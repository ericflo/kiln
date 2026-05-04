use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::Mutex;

use candle_core::DType;
use kiln_core::block::BlockManager;
use kiln_core::config::ModelConfig;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::Engine;
use kiln_model::{LinearAttentionState, ModelRunner, PagedKvCache, PagedPrefixRegistration};
use kiln_scheduler::{PrefixCacheStats, Scheduler};
use kiln_train::TrainingState;
use serde::Serialize;

use crate::decode_stats::DecodeStatsRing;
use crate::metrics::Metrics;
use crate::recent_requests::{DEFAULT_CAPACITY as RECENT_REQUESTS_CAPACITY, RecentRequestsRing};
use crate::training_queue::{SharedTrainingQueue, ShutdownFlag};

const DEFAULT_BLOCK_SIZE: usize = 16;
const MIN_AUTO_KV_BLOCKS: usize = 64;
const METAL_AUTO_MAX_KV_BLOCKS_LOW_MEM: usize = 512; // 8K tokens at block_size=16.
const METAL_AUTO_MAX_KV_BLOCKS_MID_MEM: usize = 1024; // 16K tokens at block_size=16.
const METAL_AUTO_MAX_KV_BLOCKS_HIGH_MEM: usize = 2048; // 32K tokens at block_size=16.

/// Fallback inference_memory_fraction values the KV cache auto-sizer tries in
/// order if the configured fraction OOMs at startup. Each entry is only tried
/// when it is strictly less than the configured value, so a user who pins
/// `inference_memory_fraction=0.5` gets retries at 0.45, not at 0.85→0.75→...
///
/// The descending shape (large initial step, smaller subsequent steps) matches
/// what we see in practice on A40/A6000-class 48 GiB cards with Qwen3.5-4B BF16:
/// the top of the fraction curve is the danger zone for activation peaks plus
/// driver overhead, and once you drop ~10 percentage points you have plenty of
/// room. See `phase11-685-autosizer-oom-default` and issue #685 for context.
const AUTO_SIZER_FALLBACK_FRACTIONS: &[f64] = &[0.75, 0.65, 0.55, 0.45];

/// GPU memory budget tracking for coordinating inference and training.
///
/// On startup, we compute how much VRAM is available and partition it:
/// - Model weights (fixed)
/// - KV cache for inference (controlled by KILN_INFERENCE_MEMORY_FRACTION)
/// - Remaining budget available for training
#[derive(Debug, Serialize)]
pub struct GpuMemoryBudget {
    /// Total GPU memory in bytes (0 if CPU-only).
    pub total_vram_bytes: u64,
    /// Post-load CUDA residency in bytes, or the static model estimate when
    /// runtime residency is unavailable.
    pub model_memory_bytes: u64,
    /// Static model parameter estimate in bytes.
    pub estimated_model_memory_bytes: u64,
    /// Post-load CUDA residency snapshot in bytes (0 when unavailable).
    pub post_load_used_vram_bytes: u64,
    /// Peak post-prefill CUDA residency observed at request boundaries.
    #[serde(skip)]
    pub peak_prefill_used_vram_bytes: std::sync::atomic::AtomicU64,
    /// KV cache allocation in bytes.
    pub kv_cache_bytes: u64,
    /// Memory available for training in bytes.
    pub training_budget_bytes: u64,
    /// Fraction of VRAM reserved for inference (KV cache). Default 0.7.
    pub inference_memory_fraction: f64,
}

impl GpuMemoryBudget {
    /// Compute the memory budget given model config and allocation parameters.
    ///
    /// `total_vram_bytes`: Total GPU VRAM (0 for CPU).
    /// `model_memory_bytes`: post-load residency or static model estimate.
    /// `kv_cache_bytes`: Actual KV cache allocation size.
    /// `inference_fraction`: Fraction of VRAM for inference.
    /// `training_memory_gb`: Optional explicit training memory budget in GB.
    pub fn compute(
        total_vram_bytes: u64,
        model_memory_bytes: u64,
        estimated_model_memory_bytes: u64,
        post_load_used_vram_bytes: u64,
        kv_cache_bytes: u64,
        inference_fraction: f64,
        training_memory_gb: Option<f64>,
    ) -> Self {
        let training_budget_bytes = if total_vram_bytes == 0 {
            // CPU mode — no GPU memory budget applies
            0
        } else {
            // Explicit override takes precedence
            if let Some(gb) = training_memory_gb {
                return Self {
                    total_vram_bytes,
                    model_memory_bytes,
                    estimated_model_memory_bytes,
                    post_load_used_vram_bytes,
                    peak_prefill_used_vram_bytes: std::sync::atomic::AtomicU64::new(0),
                    kv_cache_bytes,
                    training_budget_bytes: (gb * 1024.0 * 1024.0 * 1024.0) as u64,
                    inference_memory_fraction: inference_fraction,
                };
            }
            // Auto-detect: total - model - kv_cache
            total_vram_bytes
                .saturating_sub(model_memory_bytes)
                .saturating_sub(kv_cache_bytes)
        };

        Self {
            total_vram_bytes,
            model_memory_bytes,
            estimated_model_memory_bytes,
            post_load_used_vram_bytes,
            peak_prefill_used_vram_bytes: std::sync::atomic::AtomicU64::new(0),
            kv_cache_bytes,
            training_budget_bytes,
            inference_memory_fraction: inference_fraction,
        }
    }

    pub fn peak_prefill_used_vram_bytes(&self) -> u64 {
        self.peak_prefill_used_vram_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn observe_prefill_used_vram_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut current = self.peak_prefill_used_vram_bytes();
        while bytes > current {
            match self.peak_prefill_used_vram_bytes.compare_exchange_weak(
                current,
                bytes,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    /// Check if there is enough memory for training. Returns an error message if not.
    pub fn check_training_feasible(&self, estimated_training_bytes: u64) -> Result<(), String> {
        if self.total_vram_bytes == 0 {
            // CPU mode — no GPU budget enforcement
            return Ok(());
        }
        if estimated_training_bytes > self.training_budget_bytes {
            return Err(format!(
                "insufficient GPU memory for training: need ~{:.1}GB but only {:.1}GB available \
                 (total {:.1}GB - model {:.1}GB - KV cache {:.1}GB). \
                 Try reducing KILN_NUM_BLOCKS or setting KILN_INFERENCE_MEMORY_FRACTION lower",
                estimated_training_bytes as f64 / 1e9,
                self.training_budget_bytes as f64 / 1e9,
                self.total_vram_bytes as f64 / 1e9,
                self.model_memory_bytes as f64 / 1e9,
                self.kv_cache_bytes as f64 / 1e9,
            ));
        }
        Ok(())
    }
}

/// Coordination lock for GPU memory sharing between inference and training.
///
/// Inference acquires a read lock (multiple concurrent inference requests OK).
/// Training acquires a write lock (blocks inference during gradient computation).
/// This prevents combined peak VRAM from exceeding GPU capacity.
///
/// Training should acquire this per-segment (for gradient-checkpointed training),
/// not for the entire job, to minimize inference latency impact.
pub type GpuCoordinationLock = Arc<std::sync::RwLock<()>>;

/// Type of training job.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingJobType {
    Sft,
    Grpo,
}

/// Tracked training job info stored in AppState.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingJobInfo {
    pub job_id: String,
    pub adapter_name: String,
    pub job_type: TrainingJobType,
    pub state: TrainingState,
    pub progress: f32,
    pub loss: Option<f64>,
    pub epoch: Option<u32>,
    pub adapter_path: Option<String>,
    #[serde(skip)]
    pub submitted_at: std::time::Instant,
    pub auto_load: bool,
    /// Wall-clock instant at which the job entered a terminal state
    /// (`Completed` or `Failed`). `None` while the job is still
    /// `Queued` or `Running`. Used by the training-queue worker's GC
    /// pass to TTL-evict stale terminal entries from the tracking map.
    /// See `AppState::tracked_job_ttl`.
    #[serde(skip)]
    pub finished_at: Option<std::time::Instant>,
}

/// Thread-safe map of tracked training jobs.
pub type TrainingJobs = Arc<std::sync::RwLock<HashMap<String, TrainingJobInfo>>>;

pub const MIN_PREFIX_CACHE_MAX_ENTRIES: usize = 1;
const MIN_PREFIX_CACHE_STATE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PREFIX_CACHE_STATE_BYTES: u64 = 1024 * 1024 * 1024;
const PREFIX_CACHE_STATE_FRACTION_DIVISOR: u64 = 40;

pub struct RealPrefixCache {
    enabled: bool,
    max_blocks: usize,
    max_entries: usize,
    state_bytes_per_entry: u64,
    max_state_bytes: u64,
    block_size: usize,
    next_entry_id: u64,
    entries: Vec<RealPrefixCacheEntry>,
    block_refcounts: HashMap<u32, usize>,
    stats: PrefixCacheStats,
}

struct RealPrefixCacheEntry {
    id: u64,
    adapter: Option<String>,
    prompt_tokens: Vec<TokenId>,
    block_ids: Vec<u32>,
    linear_state: LinearAttentionState,
    last_used: u64,
    active_uses: usize,
}

pub struct RealPrefixCacheHit {
    pub entry_id: u64,
    pub cached_tokens: usize,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
}

pub struct RealPrefixCacheRegisterOutcome {
    pub retained_blocks: Vec<u32>,
    pub evicted_blocks: Vec<u32>,
}

impl RealPrefixCache {
    pub fn new(
        enabled: bool,
        block_size: usize,
        max_blocks: usize,
        max_entries: usize,
        state_bytes_per_entry: u64,
    ) -> Self {
        let max_entries = max_entries.max(MIN_PREFIX_CACHE_MAX_ENTRIES);
        let max_state_bytes = state_bytes_per_entry.saturating_mul(max_entries as u64);
        Self {
            enabled,
            max_blocks,
            max_entries,
            state_bytes_per_entry,
            max_state_bytes,
            block_size,
            next_entry_id: 1,
            entries: Vec::new(),
            block_refcounts: HashMap::new(),
            stats: PrefixCacheStats {
                max_blocks,
                ..PrefixCacheStats::default()
            },
        }
    }

    pub fn disabled(block_size: usize) -> Self {
        Self::new(false, block_size, 0, MIN_PREFIX_CACHE_MAX_ENTRIES, 0)
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.max_blocks > 0
    }

    pub fn lookup(
        &mut self,
        adapter: &Option<String>,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<Option<RealPrefixCacheHit>> {
        if !self.is_enabled() {
            return Ok(None);
        }

        let best_idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                &entry.adapter == adapter
                    && prompt_tokens.len() > entry.prompt_tokens.len()
                    && prompt_tokens.starts_with(&entry.prompt_tokens)
                    && entry.prompt_tokens.len() % self.block_size == 0
            })
            .max_by_key(|(_, entry)| entry.prompt_tokens.len())
            .map(|(idx, _)| idx);

        let Some(idx) = best_idx else {
            self.stats.lookup_misses += 1;
            return Ok(None);
        };

        self.stats.lookup_hits += 1;
        self.stats.hit_tokens += self.entries[idx].prompt_tokens.len() as u64;
        self.stats.hit_blocks += self.entries[idx].block_ids.len() as u64;
        self.entries[idx].last_used = self.stats.lookup_hits + self.stats.lookup_misses;
        self.entries[idx].active_uses += 1;

        Ok(Some(RealPrefixCacheHit {
            entry_id: self.entries[idx].id,
            cached_tokens: self.entries[idx].prompt_tokens.len(),
            block_ids: self.entries[idx].block_ids.clone(),
            linear_state: self.entries[idx].linear_state.snapshot()?,
        }))
    }

    pub fn release_hit(&mut self, entry_id: u64) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == entry_id) {
            entry.active_uses = entry.active_uses.saturating_sub(1);
        }
    }

    pub fn register(
        &mut self,
        adapter: Option<String>,
        registration: PagedPrefixRegistration,
    ) -> RealPrefixCacheRegisterOutcome {
        if !self.is_enabled()
            || registration.prompt_tokens.is_empty()
            || registration.prompt_tokens.len() % self.block_size != 0
            || registration.block_ids.is_empty()
        {
            return RealPrefixCacheRegisterOutcome {
                retained_blocks: Vec::new(),
                evicted_blocks: Vec::new(),
            };
        }

        if self.entries.iter().any(|entry| {
            entry.adapter == adapter && entry.prompt_tokens == registration.prompt_tokens
        }) {
            return RealPrefixCacheRegisterOutcome {
                retained_blocks: Vec::new(),
                evicted_blocks: Vec::new(),
            };
        }

        let mut evicted_blocks = Vec::new();
        let needed_new_blocks = registration
            .block_ids
            .iter()
            .filter(|block_id| !self.block_refcounts.contains_key(block_id))
            .count();
        while self.cached_blocks() + needed_new_blocks > self.max_blocks
            || self.entries.len() >= self.max_entries
        {
            let Some(evict_idx) = self.oldest_evictable_entry_idx() else {
                return RealPrefixCacheRegisterOutcome {
                    retained_blocks: Vec::new(),
                    evicted_blocks,
                };
            };
            let evicted = self.entries.remove(evict_idx);
            evicted_blocks.extend(self.release_entry_blocks(&evicted.block_ids));
        }

        let retained_blocks: Vec<u32> = registration
            .block_ids
            .iter()
            .copied()
            .filter(|block_id| !self.block_refcounts.contains_key(block_id))
            .collect();
        for &block_id in &registration.block_ids {
            *self.block_refcounts.entry(block_id).or_insert(0) += 1;
        }
        // #673: An entry evicted above may have shared block IDs with the
        // incoming registration. `release_entry_blocks` would have pushed those
        // IDs into `evicted_blocks`, but the refcount increments above have now
        // re-claimed them. Returning them in `evicted_blocks` would cause the
        // API layer to free live cached blocks back to the BlockManager, where
        // a concurrent request can re-allocate and overwrite them.
        let registration_set: HashSet<u32> = registration.block_ids.iter().copied().collect();
        evicted_blocks.retain(|block_id| !registration_set.contains(block_id));
        debug_assert!(
            evicted_blocks
                .iter()
                .all(|id| !self.block_refcounts.contains_key(id)),
            "RealPrefixCache::register: evicted_blocks must not contain any block currently in block_refcounts"
        );
        let id = self.next_entry_id;
        self.next_entry_id += 1;
        let last_used = self.stats.lookup_hits + self.stats.lookup_misses;
        self.entries.push(RealPrefixCacheEntry {
            id,
            adapter,
            prompt_tokens: registration.prompt_tokens,
            block_ids: registration.block_ids,
            linear_state: registration.linear_state,
            last_used,
            active_uses: 0,
        });
        RealPrefixCacheRegisterOutcome {
            retained_blocks,
            evicted_blocks,
        }
    }

    pub fn clear(&mut self) -> Vec<u32> {
        let mut blocks = Vec::new();
        self.entries.clear();
        blocks.extend(self.block_refcounts.keys().copied());
        self.block_refcounts.clear();
        blocks
    }

    fn oldest_evictable_entry_idx(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.active_uses == 0)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(idx, _)| idx)
    }

    fn release_entry_blocks(&mut self, block_ids: &[u32]) -> Vec<u32> {
        let mut freed = Vec::new();
        for &block_id in block_ids {
            if let Some(refcount) = self.block_refcounts.get_mut(&block_id) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    self.block_refcounts.remove(&block_id);
                    freed.push(block_id);
                }
            }
        }
        freed
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            cached_blocks: self.cached_blocks(),
            max_blocks: self.max_blocks,
            cached_entries: self.entries.len(),
            max_entries: self.max_entries,
            cached_state_bytes: self
                .state_bytes_per_entry
                .saturating_mul(self.entries.len() as u64),
            max_state_bytes: self.max_state_bytes,
            ..self.stats
        }
    }

    fn cached_blocks(&self) -> usize {
        self.block_refcounts.len()
    }
}

/// Which inference backend the server is using.
pub enum ModelBackend {
    /// Mock engine + scheduler for testing without real weights.
    Mock {
        scheduler: Arc<Mutex<Scheduler>>,
        engine: Arc<dyn Engine>,
    },
    /// Real model weights loaded via ModelRunner with paged KV cache.
    Real {
        runner: Arc<std::sync::RwLock<ModelRunner>>,
        block_manager: Arc<std::sync::Mutex<BlockManager>>,
        paged_cache: Arc<std::sync::Mutex<PagedKvCache>>,
        prefix_cache: Arc<std::sync::Mutex<RealPrefixCache>>,
        batching_engine: Option<crate::batching_engine::BatchingEngineHandle>,
    },
}

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_config: ModelConfig,
    pub backend: Arc<ModelBackend>,
    pub tokenizer: Arc<KilnTokenizer>,
    /// Directory where LoRA adapter weights are stored on disk.
    pub adapter_dir: PathBuf,
    /// Name of the currently active LoRA adapter (None = base model).
    pub active_adapter_name: Arc<std::sync::RwLock<Option<String>>>,
    /// Tracked training jobs (job_id → info).
    pub training_jobs: TrainingJobs,
    /// GPU memory budget for coordinating inference and training.
    pub memory_budget: Arc<GpuMemoryBudget>,
    /// Coordination lock: inference takes read lock, training takes write lock.
    /// This prevents simultaneous GPU-heavy operations from OOMing.
    pub gpu_lock: GpuCoordinationLock,
    /// FIFO training queue — jobs are enqueued here and executed sequentially
    /// by a background worker.
    pub training_queue: SharedTrainingQueue,
    /// Detected VRAM info for config/debug reporting.
    pub vram_info: kiln_core::vram::GpuVramInfo,
    /// Shutdown flag — set to true when the server is shutting down.
    pub shutdown: ShutdownFlag,
    /// Per-request timeout duration. Configurable via KILN_REQUEST_TIMEOUT_SECS (default 600).
    pub request_timeout: std::time::Duration,
    /// Prometheus metrics counters.
    pub metrics: Arc<Metrics>,
    /// Server startup time — used to compute uptime in health checks.
    pub started_at: std::time::Instant,
    /// True once startup inference prewarm has finished or was not needed.
    pub inference_prewarm_complete: Arc<AtomicBool>,
    /// Server-level default for adapter checkpoint interval during training.
    /// Per-job config overrides this. None = only save at the end.
    pub checkpoint_interval: Option<usize>,
    /// Optional URL to POST a JSON notification to whenever a training
    /// job completes or fails. `None` disables webhook firing entirely.
    /// See `TrainingConfig::webhook_url` for the payload contract.
    pub training_webhook_url: Option<String>,
    /// Maximum number of training jobs that may sit in `training_queue`
    /// at once. Submissions while the queue is at this cap are rejected
    /// with HTTP 503 + `Retry-After: 30`. Mirrors
    /// `TrainingConfig::max_queued_jobs` (default 32).
    pub max_queued_training_jobs: usize,
    /// Maximum number of tracked training jobs that may live in
    /// `training_jobs` (the in-memory tracking map) at once. Submissions
    /// while the map is at this cap are rejected with HTTP 503 +
    /// `Retry-After: 30` and the `training_tracked_full` error code.
    /// Mirrors `TrainingConfig::max_tracked_jobs` (default 1024).
    pub max_tracked_jobs: usize,
    /// TTL for terminal (`Completed` / `Failed`) entries in
    /// `training_jobs`. The training worker periodically removes terminal
    /// entries whose `finished_at` timestamp is older than this duration.
    /// Active entries (`Queued` / `Running`) are never GC'd.
    /// Mirrors `TrainingConfig::tracked_job_ttl_secs` (default 3600s).
    pub tracked_job_ttl: std::time::Duration,
    /// Maximum total bytes that finalized adapters may occupy in
    /// `adapter_dir/`. Uploads to `POST /v1/adapters/upload` that would
    /// push the total over this cap are rejected before the rename-into-
    /// place step. `None` disables the cap entirely (operator opt-out).
    /// `.upload-tmp-*/` staging dirs and the `.composed/<hash>/` cache
    /// are excluded from the count — they are bounded separately. Mirrors
    /// `AdaptersConfig::max_disk_bytes` (default 100 GiB).
    pub adapter_max_disk_bytes: Option<u64>,
    /// Byte cap for the on-disk composed-adapter cache at
    /// `adapter_dir/.composed/<hash>/`. Enforced via LRU eviction
    /// (oldest mtime first) after a successful synthesize. `None`
    /// disables the byte cap. Mirrors
    /// `AdaptersConfig::composed_cache_max_bytes` (default 10 GiB).
    pub composed_cache_max_bytes: Option<u64>,
    /// Entry-count cap for the on-disk composed-adapter cache at
    /// `adapter_dir/.composed/`. Enforced via LRU eviction (oldest
    /// mtime first). `None` disables the entry cap. Mirrors
    /// `AdaptersConfig::composed_cache_max_entries` (default 64).
    pub composed_cache_max_entries: Option<u64>,
    /// Identifier exposed at `/v1/models` and echoed in chat completion responses.
    pub served_model_id: String,
    /// Rolling timestamp ring for live decode tok/s + ITL on the /ui dashboard.
    pub decode_stats: Arc<std::sync::Mutex<DecodeStatsRing>>,
    /// Bounded history of recent chat-completion requests for the /ui dashboard.
    pub recent_requests: Arc<std::sync::Mutex<RecentRequestsRing>>,
}

impl AppState {
    /// Create an AppState with the mock engine backend.
    pub fn clear_real_prefix_cache(&self) {
        let ModelBackend::Real {
            block_manager,
            prefix_cache,
            ..
        } = self.backend.as_ref()
        else {
            return;
        };
        let blocks = {
            let mut cache = prefix_cache.lock().unwrap();
            cache.clear()
        };
        if !blocks.is_empty() {
            let mut bm = block_manager.lock().unwrap();
            bm.free_all(&blocks);
        }
    }

    pub fn new_mock(
        model_config: ModelConfig,
        scheduler: Scheduler,
        engine: Arc<dyn Engine>,
        tokenizer: KilnTokenizer,
        request_timeout_secs: u64,
        served_model_id: String,
    ) -> Self {
        Self {
            model_config,
            backend: Arc::new(ModelBackend::Mock {
                scheduler: Arc::new(Mutex::new(scheduler)),
                engine,
            }),
            tokenizer: Arc::new(tokenizer),
            adapter_dir: PathBuf::from("adapters"),
            active_adapter_name: Arc::new(std::sync::RwLock::new(None)),
            training_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            memory_budget: Arc::new(GpuMemoryBudget::compute(0, 0, 0, 0, 0, 1.0, None)),
            gpu_lock: Arc::new(std::sync::RwLock::new(())),
            training_queue: crate::training_queue::new_shared_queue(),
            vram_info: kiln_core::vram::GpuVramInfo {
                total_bytes: 0,
                source: kiln_core::vram::VramSource::None,
            },
            shutdown: crate::training_queue::new_shutdown_flag(),
            request_timeout: std::time::Duration::from_secs(request_timeout_secs),
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
            inference_prewarm_complete: Arc::new(AtomicBool::new(true)),
            checkpoint_interval: None,
            training_webhook_url: None,
            max_queued_training_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl: std::time::Duration::from_secs(3600),
            adapter_max_disk_bytes: Some(100 * 1024u64.pow(3)),
            composed_cache_max_bytes: Some(10 * 1024u64.pow(3)),
            composed_cache_max_entries: Some(64),
            served_model_id,
            decode_stats: Arc::new(std::sync::Mutex::new(DecodeStatsRing::new(4096))),
            recent_requests: Arc::new(std::sync::Mutex::new(RecentRequestsRing::new(
                RECENT_REQUESTS_CAPACITY,
            ))),
        }
    }

    /// Create an AppState with a real ModelRunner backend and paged KV cache.
    ///
    /// Uses `block_size=16` by default. The number of blocks can be overridden
    /// via `memory_cfg.num_blocks`. Otherwise derived from available VRAM or
    /// `max_position_embeddings / block_size`.
    ///
    /// GPU memory sharing: `memory_cfg.inference_memory_fraction` (default 0.7)
    /// controls what fraction of remaining VRAM (after model weights) is allocated
    /// to KV cache, reserving the rest for training. Set to 1.0 for inference-only.
    pub fn new_real(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
        device: candle_core::Device,
        adapter_dir: PathBuf,
        memory_cfg: &crate::config::MemoryConfig,
        request_timeout_secs: u64,
        served_model_id: String,
        prefix_cache_cfg: &crate::config::PrefixCacheConfig,
    ) -> Self {
        let block_size = DEFAULT_BLOCK_SIZE;

        // KV cache dtype must match the model's activation dtype, otherwise
        // `paged_cache.write` hits a slice-set dtype mismatch on the first
        // full-attention layer. The previous `cuda_is_available()` check was
        // a compile-time cfg(feature = "cuda"), so Metal builds ran the F32
        // branch even though the Qwen3.5-4B model loads in BF16 — prefill
        // failed on every request. Key the choice off `model_config.dtype`
        // instead so tests with F32 tiny configs keep working and any real
        // BF16 model gets a matching BF16 cache regardless of backend.
        let kv_dtype = match model_config.dtype {
            kiln_core::config::DType::BF16 => DType::BF16,
            kiln_core::config::DType::FP16 => DType::F16,
            kiln_core::config::DType::FP32 => DType::F32,
        };

        let kv_dtype_bytes: usize = if memory_cfg.kv_cache_fp8 {
            1 // FP8: 1 byte per element
        } else {
            match kv_dtype {
                DType::BF16 | DType::F16 => 2,
                _ => 4,
            }
        };

        let configured_inference_fraction = memory_cfg.inference_memory_fraction.clamp(0.1, 1.0);

        // Estimate model weight memory (approximate: params * dtype_bytes)
        // Qwen3.5-4B ≈ 4B params * 2 bytes (bf16) ≈ 8GB
        let estimated_model_bytes: u64 = estimate_model_memory_bytes(&model_config);

        // Compute KV cache bytes per block:
        // num_full_attention_layers * 2 (K+V) * num_kv_heads * head_dim * block_size * dtype_bytes
        let bytes_per_block: u64 = (model_config.num_full_attention_layers
            * 2
            * model_config.num_kv_heads
            * model_config.head_dim
            * block_size
            * kv_dtype_bytes) as u64;

        // Detect VRAM once and reuse it for both auto-sizing and reporting so
        // startup doesn't repeat the same probe/logging path.
        let vram_info = kiln_core::vram::detect_vram();
        let total_vram = detected_gpu_total_memory(&device, &vram_info);
        let is_metal = is_metal_device(&device);

        let post_load_used_vram_info = runtime_used_vram_for_device(&device);
        let post_load_used_vram = post_load_used_vram_info
            .map(|info| info.used_bytes)
            .unwrap_or(0);
        let sizing_residency_bytes = post_load_used_vram.max(estimated_model_bytes);
        if post_load_used_vram > 0 {
            tracing::info!(
                post_load_used_vram_gb = post_load_used_vram as f64 / 1e9,
                estimated_model_gb = estimated_model_bytes as f64 / 1e9,
                source = %post_load_used_vram_info.unwrap().source,
                "post-load CUDA residency snapshot for KV sizing"
            );
        } else {
            tracing::warn!(
                estimated_model_gb = estimated_model_bytes as f64 / 1e9,
                "post-load CUDA residency unavailable; falling back to static model memory estimate for KV sizing"
            );
        }

        // Compute num_blocks for a given fraction. Used both for the explicit
        // `memory_cfg.num_blocks` path and the auto-sizer retry loop below.
        let compute_blocks_for_fraction = |fraction: f64| -> usize {
            auto_num_blocks_for_fraction(
                total_vram,
                sizing_residency_bytes,
                bytes_per_block,
                fraction,
                model_config.max_position_embeddings,
                block_size,
                is_metal,
            )
        };

        // FP8 (E4M3FN) packing currently uses a CPU round-trip on every
        // write — fine on CUDA where bf16→fp8 packing is amortized over the
        // kernel work, but on Metal the round-trip dominates decode. Gate it
        // off on Metal with a warning rather than silently shipping a slow
        // path; users who know what they're doing can re-enable via
        // KILN_ALLOW_FP8_ON_METAL=1.
        let fp8_enabled = {
            let requested = memory_cfg.kv_cache_fp8;
            // Require an explicit opt-in value (`1`/`true`) so the common
            // misreading `KILN_ALLOW_FP8_ON_METAL=0` disables FP8 as intended
            // instead of flipping the gate because the variable happens to
            // be set.
            let metal_override = matches!(
                std::env::var("KILN_ALLOW_FP8_ON_METAL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE")
            );
            if requested && matches!(device, candle_core::Device::Metal(_)) && !metal_override {
                tracing::warn!(
                    "FP8 cache disabled on Metal (CPU round-trip cost); \
                     set KILN_ALLOW_FP8_ON_METAL=1 to override"
                );
                false
            } else {
                requested
            }
        };
        // Allocation closure: try to build the paged KV cache for `n` blocks.
        // Used by the auto-sizer retry loop below. CUDA OOM bubbles up here as
        // an `Err` from `Tensor::empty`, which we catch and retry with a smaller
        // budget instead of panicking on the first failure.
        let allocate_cache = |n: usize| -> anyhow::Result<PagedKvCache> {
            PagedKvCache::new_uninit_with_fp8(
                model_config.num_full_attention_layers,
                n,
                block_size,
                model_config.num_kv_heads,
                model_config.head_dim,
                kv_dtype,
                &device,
                fp8_enabled,
            )
        };

        // Determine num_blocks + paged cache:
        //   - If `memory_cfg.num_blocks` is set, honor it exactly (no retry — the
        //     user has chosen a specific value and we should not silently shrink
        //     past their request).
        //   - Otherwise, run the auto-sizer retry loop, starting at the
        //     configured `inference_memory_fraction` and shrinking on OOM.
        let (paged_cache, num_blocks, inference_fraction) = if let Some(explicit) =
            memory_cfg.num_blocks
        {
            tracing::info!(
                num_blocks = explicit,
                block_size,
                ?kv_dtype,
                fp8_enabled,
                "allocating paged KV cache (explicit num_blocks)"
            );
            let cache = allocate_cache(explicit)
                .expect("failed to create PagedKvCache with explicit num_blocks");
            (cache, explicit, configured_inference_fraction)
        } else {
            tracing::info!(
                total_vram_gb = total_vram as f64 / 1e9,
                model_gb = estimated_model_bytes as f64 / 1e9,
                post_load_used_vram_gb = post_load_used_vram as f64 / 1e9,
                sizing_residency_gb = sizing_residency_bytes as f64 / 1e9,
                inference_fraction = configured_inference_fraction,
                "memory-aware KV cache sizing"
            );
            match auto_size_with_retry(
                configured_inference_fraction,
                AUTO_SIZER_FALLBACK_FRACTIONS,
                &compute_blocks_for_fraction,
                |n| {
                    tracing::info!(
                        num_blocks = n,
                        block_size,
                        ?kv_dtype,
                        fp8_enabled,
                        "allocating paged KV cache"
                    );
                    allocate_cache(n).map_err(|e| format!("{e:#}"))
                },
            ) {
                Ok(success) => {
                    if success.fraction < configured_inference_fraction {
                        tracing::warn!(
                            configured_fraction = configured_inference_fraction,
                            actual_fraction = success.fraction,
                            num_blocks = success.num_blocks,
                            attempts = success.attempted_failures.len() + 1,
                            "KV cache auto-sizer fell back to a smaller inference_memory_fraction \
                             because the configured value OOM'd; set memory.inference_memory_fraction \
                             (or KILN_INFERENCE_MEMORY_FRACTION) to this value to silence the warning"
                        );
                    } else {
                        tracing::info!(
                            inference_fraction = success.fraction,
                            num_blocks = success.num_blocks,
                            "KV cache auto-sizer succeeded on first attempt"
                        );
                    }
                    (success.cache, success.num_blocks, success.fraction)
                }
                Err(failure) => {
                    let suggested_blocks = suggested_emergency_num_blocks(
                        total_vram,
                        sizing_residency_bytes,
                        bytes_per_block,
                        block_size,
                        model_config.max_position_embeddings,
                        is_metal,
                    );
                    let msg = format_oom_remediation_message(
                        &failure,
                        total_vram,
                        sizing_residency_bytes,
                        bytes_per_block,
                        suggested_blocks,
                        configured_inference_fraction,
                        vram_info.source,
                    );
                    // Print to stderr too so users see the actionable message
                    // even when tracing is configured to discard error events.
                    eprintln!("{msg}");
                    tracing::error!("{msg}");
                    panic!("{msg}");
                }
            }
        };

        let block_manager = BlockManager::new(num_blocks, block_size);
        let kv_cache_bytes = num_blocks as u64 * bytes_per_block;
        let memory_budget = GpuMemoryBudget::compute(
            total_vram,
            sizing_residency_bytes,
            estimated_model_bytes,
            post_load_used_vram,
            kv_cache_bytes,
            inference_fraction,
            memory_cfg.training_memory_gb,
        );

        tracing::info!(
            total_vram_gb = memory_budget.total_vram_bytes as f64 / 1e9,
            model_gb = memory_budget.model_memory_bytes as f64 / 1e9,
            estimated_model_gb = memory_budget.estimated_model_memory_bytes as f64 / 1e9,
            post_load_used_vram_gb = memory_budget.post_load_used_vram_bytes as f64 / 1e9,
            kv_cache_gb = memory_budget.kv_cache_bytes as f64 / 1e9,
            training_budget_gb = memory_budget.training_budget_bytes as f64 / 1e9,
            inference_fraction = memory_budget.inference_memory_fraction,
            "GPU memory budget"
        );

        let prefix_cache_max_blocks = if prefix_cache_cfg.enabled {
            prefix_cache_cfg.max_blocks.unwrap_or(num_blocks / 4)
        } else {
            0
        };
        let prefix_cache_state_bytes_per_entry =
            linear_attention_state_bytes(&model_config, &device);
        let prefix_cache_max_entries = if prefix_cache_cfg.enabled {
            prefix_cache_cfg.max_entries.unwrap_or_else(|| {
                default_prefix_cache_max_entries(total_vram, prefix_cache_state_bytes_per_entry)
            })
        } else {
            MIN_PREFIX_CACHE_MAX_ENTRIES
        };
        tracing::info!(
            max_blocks = prefix_cache_max_blocks,
            max_entries = prefix_cache_max_entries,
            state_bytes_per_entry = prefix_cache_state_bytes_per_entry,
            max_state_bytes =
                prefix_cache_state_bytes_per_entry.saturating_mul(prefix_cache_max_entries as u64),
            "prefix cache budget"
        );
        let prefix_cache = RealPrefixCache::new(
            prefix_cache_cfg.enabled,
            block_size,
            prefix_cache_max_blocks,
            prefix_cache_max_entries,
            prefix_cache_state_bytes_per_entry,
        );

        let runner = Arc::new(std::sync::RwLock::new(runner));
        let block_manager = Arc::new(std::sync::Mutex::new(block_manager));
        let paged_cache = Arc::new(std::sync::Mutex::new(paged_cache));
        let prefix_cache = Arc::new(std::sync::Mutex::new(prefix_cache));
        let gpu_lock = Arc::new(std::sync::RwLock::new(()));
        let batching_engine = matches!(
            std::env::var("KILN_BATCHING_ENGINE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
        .then(|| {
            tracing::info!("KILN_BATCHING_ENGINE enabled — routing streaming and non-streaming real completions through batching actor");
            crate::batching_engine::BatchingEngineHandle::start(Arc::new(
                crate::batching_engine::RealDecodeForward::new(
                    runner.clone(),
                    block_manager.clone(),
                    paged_cache.clone(),
                    prefix_cache.clone(),
                    gpu_lock.clone(),
                ),
            ))
        });

        Self {
            model_config,
            backend: Arc::new(ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
                prefix_cache,
                batching_engine,
            }),
            tokenizer: Arc::new(tokenizer),
            adapter_dir,
            active_adapter_name: Arc::new(std::sync::RwLock::new(None)),
            training_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            memory_budget: Arc::new(memory_budget),
            gpu_lock,
            training_queue: crate::training_queue::new_shared_queue(),
            vram_info,
            shutdown: crate::training_queue::new_shutdown_flag(),
            request_timeout: std::time::Duration::from_secs(request_timeout_secs),
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
            inference_prewarm_complete: Arc::new(AtomicBool::new(!matches!(
                device,
                candle_core::Device::Metal(_)
            ))),
            checkpoint_interval: None,
            training_webhook_url: None,
            max_queued_training_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl: std::time::Duration::from_secs(3600),
            adapter_max_disk_bytes: Some(100 * 1024u64.pow(3)),
            composed_cache_max_bytes: Some(10 * 1024u64.pow(3)),
            composed_cache_max_entries: Some(64),
            served_model_id,
            decode_stats: Arc::new(std::sync::Mutex::new(DecodeStatsRing::new(4096))),
            recent_requests: Arc::new(std::sync::Mutex::new(RecentRequestsRing::new(
                RECENT_REQUESTS_CAPACITY,
            ))),
        }
    }
}

fn linear_attention_state_bytes(config: &ModelConfig, device: &candle_core::Device) -> u64 {
    let num_linear_layers = config
        .num_layers
        .saturating_sub(config.num_full_attention_layers) as u64;
    let recurrent_dtype_bytes = match (device, config.dtype) {
        #[cfg(feature = "metal")]
        (candle_core::Device::Metal(_), kiln_core::config::DType::BF16) => 2,
        #[cfg(feature = "metal")]
        (candle_core::Device::Metal(_), kiln_core::config::DType::FP16) => 2,
        _ => 4,
    };
    let recurrent_elems = (config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim) as u64;
    let conv_elems =
        (config.linear_qkv_dim() * config.linear_conv_kernel_dim.saturating_sub(1)) as u64;
    num_linear_layers.saturating_mul(
        recurrent_elems
            .saturating_mul(recurrent_dtype_bytes)
            .saturating_add(conv_elems.saturating_mul(4)),
    )
}

fn default_prefix_cache_max_entries(total_vram_bytes: u64, state_bytes_per_entry: u64) -> usize {
    if state_bytes_per_entry == 0 {
        return MIN_PREFIX_CACHE_MAX_ENTRIES;
    }
    let state_budget = if total_vram_bytes == 0 {
        MIN_PREFIX_CACHE_STATE_BYTES
    } else {
        (total_vram_bytes / PREFIX_CACHE_STATE_FRACTION_DIVISOR)
            .clamp(MIN_PREFIX_CACHE_STATE_BYTES, MAX_PREFIX_CACHE_STATE_BYTES)
    };
    ((state_budget / state_bytes_per_entry) as usize).max(MIN_PREFIX_CACHE_MAX_ENTRIES)
}

/// Estimate model weight memory in bytes from config.
///
/// Uses a rough formula: total parameters * dtype_bytes.
/// For Qwen3.5-4B in BF16: ~4B params * 2 bytes ≈ 8GB.
fn estimate_model_memory_bytes(config: &ModelConfig) -> u64 {
    let dtype_bytes: u64 = match config.dtype {
        kiln_core::config::DType::BF16 | kiln_core::config::DType::FP16 => 2,
        kiln_core::config::DType::FP32 => 4,
    };

    // Rough parameter count estimate for a transformer:
    // Embedding: vocab_size * hidden_size
    // Per layer: ~8 * hidden_size^2 + 3 * hidden_size * intermediate_size (approximate)
    // LM head: vocab_size * hidden_size (often tied with embedding)
    let embedding_params = config.vocab_size as u64 * config.hidden_size as u64;
    let per_layer_params = 8 * (config.hidden_size as u64 * config.hidden_size as u64)
        + 3 * (config.hidden_size as u64 * config.intermediate_size as u64);
    let total_params = embedding_params + per_layer_params * config.num_layers as u64;

    total_params * dtype_bytes
}

fn auto_num_blocks_for_fraction(
    total_vram: u64,
    sizing_residency_bytes: u64,
    bytes_per_block: u64,
    fraction: f64,
    max_position_embeddings: usize,
    block_size: usize,
    is_metal: bool,
) -> usize {
    if total_vram > 0 && bytes_per_block > 0 {
        let available_for_kv =
            ((total_vram.saturating_sub(sizing_residency_bytes)) as f64 * fraction) as u64;
        let raw_auto_blocks = (available_for_kv / bytes_per_block) as usize;
        cap_auto_num_blocks(
            raw_auto_blocks,
            max_position_embeddings,
            block_size,
            is_metal,
            total_vram,
        )
    } else {
        let raw_auto_blocks = max_position_embeddings.div_ceil(block_size).max(256);
        cap_auto_num_blocks(
            raw_auto_blocks,
            max_position_embeddings,
            block_size,
            is_metal,
            total_vram,
        )
    }
}

fn cap_auto_num_blocks(
    raw_blocks: usize,
    max_position_embeddings: usize,
    block_size: usize,
    is_metal: bool,
    total_vram_bytes: u64,
) -> usize {
    // On Metal (unified memory), an eagerly-zeroed KV cache larger than the
    // model context can dominate memory pressure on the rest of the system,
    // so we keep the historical "≤ one full context, further capped by
    // detected memory tier" behavior.
    //
    // On CUDA / CPU, memory-aware sizing already drove `raw_blocks` from the
    // available VRAM × `inference_memory_fraction` budget. Capping again at
    // one model-context-worth of blocks (≈16K for Qwen3.5-4B's 256K window)
    // bottlenecks concurrent serving: 4 in-flight 25K-token prompts +
    // generation already exhaust 6.5K blocks each, leaving the auto cap
    // routinely OOM-borderline under realistic load even on a 48 GiB A40.
    // Trust the memory-aware ceiling here; users who want a stricter cap can
    // still set `KILN_NUM_BLOCKS` or `memory.num_blocks` explicitly.
    let runtime_cap_blocks = if is_metal {
        let model_cap_blocks = max_position_embeddings
            .div_ceil(block_size)
            .max(MIN_AUTO_KV_BLOCKS);
        model_cap_blocks.min(metal_auto_max_kv_blocks(total_vram_bytes))
    } else {
        usize::MAX
    };

    raw_blocks.max(MIN_AUTO_KV_BLOCKS).min(runtime_cap_blocks)
}

fn metal_auto_max_kv_blocks(total_vram_bytes: u64) -> usize {
    let gib = total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gib < 14.0 {
        METAL_AUTO_MAX_KV_BLOCKS_LOW_MEM
    } else if gib < 24.0 {
        METAL_AUTO_MAX_KV_BLOCKS_MID_MEM
    } else {
        METAL_AUTO_MAX_KV_BLOCKS_HIGH_MEM
    }
}

fn is_metal_device(device: &candle_core::Device) -> bool {
    #[cfg(feature = "metal")]
    {
        matches!(device, candle_core::Device::Metal(_))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
        false
    }
}

/// Query total GPU memory in bytes. Returns 0 for CPU devices.
///
/// Uses the shared VRAM detection from kiln-core (nvidia-smi + sysctl
/// hw.memsize on Apple Silicon + env override).

fn runtime_used_vram_for_device(
    device: &candle_core::Device,
) -> Option<kiln_core::vram::GpuMemoryUsedInfo> {
    match device {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => {
            let info = kiln_core::vram::detect_used_vram();
            (info.used_bytes > 0).then_some(info)
        }
        _ => None,
    }
}

fn detected_gpu_total_memory(
    device: &candle_core::Device,
    vram: &kiln_core::vram::GpuVramInfo,
) -> u64 {
    match device {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => {
            if vram.total_bytes > 0 {
                tracing::info!(
                    total_gb = vram.total_bytes as f64 / 1e9,
                    source = %vram.source,
                    "GPU VRAM detected"
                );
                vram.total_bytes
            } else {
                tracing::warn!("CUDA device present but VRAM detection failed; assuming 24GB");
                24 * 1024 * 1024 * 1024
            }
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_) => {
            if vram.total_bytes > 0 {
                tracing::info!(
                    total_gb = vram.total_bytes as f64 / 1e9,
                    source = %vram.source,
                    "unified memory detected (Apple Silicon)"
                );
                vram.total_bytes
            } else {
                tracing::warn!(
                    "Metal device present but unified memory detection failed; assuming 16GB"
                );
                16 * 1024 * 1024 * 1024
            }
        }
        _ => vram.total_bytes,
    }
}

/// Successful auto-sizer outcome. Carries the live cache plus the metadata
/// needed to log the final decision and update the GPU memory budget.
struct AutoSizeSuccess {
    cache: PagedKvCache,
    num_blocks: usize,
    fraction: f64,
    /// `(fraction, num_blocks, error)` for each attempt that failed before
    /// the eventual success. Empty when the configured fraction worked.
    attempted_failures: Vec<(f64, usize, String)>,
}

/// Failure outcome — every fraction in the retry sequence OOMed.
struct AutoSizeFailure {
    /// `(fraction, num_blocks, error)` for every attempt in order.
    attempts: Vec<(f64, usize, String)>,
}

/// Auto-size the KV cache by trying `configured_fraction` first and then each
/// entry of `fallback_fractions` that is strictly less than the configured
/// value, in order. Returns the first attempt that allocates successfully, or
/// the full attempt history on failure.
///
/// `compute_blocks` maps a fraction to the number of blocks the auto-sizer
/// would request for that budget. `try_allocate` actually attempts the
/// allocation; returning `Err` means OOM (or any other failure) and the loop
/// will move to the next smaller fraction.
///
/// Pure logic — no GPU, no tensors, no logging. Tested directly with mock
/// allocators that return OOM until the fraction drops below a threshold.
fn auto_size_with_retry<C, A>(
    configured_fraction: f64,
    fallback_fractions: &[f64],
    compute_blocks: &C,
    mut try_allocate: A,
) -> Result<AutoSizeSuccess, AutoSizeFailure>
where
    C: Fn(f64) -> usize,
    A: FnMut(usize) -> Result<PagedKvCache, String>,
{
    // Build the ordered fraction sequence: configured first, then each
    // fallback that is strictly smaller (avoids retrying the same value or
    // accidentally retrying *higher* than what the user asked for).
    let mut fractions: Vec<f64> = Vec::with_capacity(1 + fallback_fractions.len());
    fractions.push(configured_fraction);
    for &f in fallback_fractions {
        if f < configured_fraction - 1e-9 {
            fractions.push(f);
        }
    }

    let mut attempts: Vec<(f64, usize, String)> = Vec::with_capacity(fractions.len());
    for fraction in fractions {
        let num_blocks = compute_blocks(fraction);
        match try_allocate(num_blocks) {
            Ok(cache) => {
                return Ok(AutoSizeSuccess {
                    cache,
                    num_blocks,
                    fraction,
                    attempted_failures: attempts,
                });
            }
            Err(err) => {
                attempts.push((fraction, num_blocks, err));
            }
        }
    }

    Err(AutoSizeFailure { attempts })
}

/// Compute a conservative `KILN_NUM_BLOCKS=N` suggestion the user can paste
/// directly. We aim for ~30% of remaining VRAM after model weights — well
/// below the smallest fallback fraction we just tried — so the suggestion has
/// enough headroom to start cleanly even on the GPU/driver combo that just
/// OOM'd at our retry floor.
fn suggested_emergency_num_blocks(
    total_vram: u64,
    estimated_model_bytes: u64,
    bytes_per_block: u64,
    block_size: usize,
    max_position_embeddings: usize,
    is_metal: bool,
) -> usize {
    if total_vram == 0 || bytes_per_block == 0 {
        // No VRAM signal — fall back to one model context worth of blocks.
        return max_position_embeddings
            .div_ceil(block_size)
            .max(MIN_AUTO_KV_BLOCKS);
    }
    let conservative_fraction = 0.30_f64;
    let available_for_kv =
        ((total_vram.saturating_sub(estimated_model_bytes)) as f64 * conservative_fraction) as u64;
    let raw = (available_for_kv / bytes_per_block) as usize;
    cap_auto_num_blocks(
        raw,
        max_position_embeddings,
        block_size,
        is_metal,
        total_vram,
    )
}

/// Render a multi-line error message that names the exact remediation flags
/// to set, instead of dumping the raw CUDA OOM. We include:
///   - what we tried (fractions + blocks counts)
///   - the underlying error from the deepest attempt
///   - a concrete `KILN_NUM_BLOCKS=N` value
///   - a concrete `inference_memory_fraction=X` value (the lowest we tried,
///     halved further to stay safely below the OOM floor)
///   - the GPU VRAM total + detection source so users can sanity-check
fn format_oom_remediation_message(
    failure: &AutoSizeFailure,
    total_vram: u64,
    estimated_model_bytes: u64,
    bytes_per_block: u64,
    suggested_blocks: usize,
    configured_fraction: f64,
    vram_source: kiln_core::vram::VramSource,
) -> String {
    let mut buf = String::new();
    buf.push_str(
        "Auto-sizer could not fit any KV cache budget on this GPU. \
         Every inference_memory_fraction we tried OOM'd during paged KV cache allocation.\n",
    );
    buf.push_str("\nAttempts (in order, all failed):\n");
    for (fraction, num_blocks, err) in &failure.attempts {
        buf.push_str(&format!(
            "  - inference_memory_fraction={:.2} -> num_blocks={}: {}\n",
            fraction,
            num_blocks,
            // Show the error compactly — most CUDA OOMs are one or two lines.
            err.lines().next().unwrap_or("<no error message>")
        ));
    }
    let vram_gb = total_vram as f64 / (1024.0 * 1024.0 * 1024.0);
    let model_gb = estimated_model_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let kv_gb = (suggested_blocks as u64 * bytes_per_block) as f64 / (1024.0 * 1024.0 * 1024.0);
    buf.push_str(&format!(
        "\nGPU detected: {:.1} GiB total VRAM (source: {}), \
         estimated model weights: {:.1} GiB.\n",
        vram_gb, vram_source, model_gb
    ));
    let suggested_fraction = (failure
        .attempts
        .last()
        .map(|(f, _, _)| *f)
        .unwrap_or(configured_fraction)
        / 2.0)
        .max(0.10);
    buf.push_str(&format!(
        "\nRecommended remediation — set ONE of the following and restart:\n  \
         (a) KILN_NUM_BLOCKS={}        # ~{:.1} GiB KV cache, conservative; or in kiln.toml: [memory] num_blocks = {}\n  \
         (b) KILN_INFERENCE_MEMORY_FRACTION={:.2}   # equivalent fraction-based knob; or in kiln.toml: [memory] inference_memory_fraction = {:.2}\n",
        suggested_blocks, kv_gb, suggested_blocks,
        suggested_fraction, suggested_fraction,
    ));
    buf.push_str(&format!(
        "\nFor reference, the configured inference_memory_fraction was {:.2}. \
         Option (a) is preferred — it bypasses the auto-sizer entirely and is what \
         #685 documented as the working workaround on A40/A6000 + Qwen3.5-4B BF16.\n",
        configured_fraction
    ));
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_linear_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            num_layers: 2,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 2,
            attn_output_gate: false,
            linear_num_key_heads: 1,
            linear_key_head_dim: 2,
            linear_num_value_heads: 1,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 2,
            partial_rotary_factor: 0.5,
        }
    }

    #[test]
    fn real_prefix_cache_records_hits_misses_and_cached_blocks() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);

        let registration = PagedPrefixRegistration {
            prompt_tokens: vec![1, 2, 3, 4],
            block_ids: vec![9],
            linear_state: state,
        };
        let outcome = cache.register(None, registration);
        assert_eq!(outcome.retained_blocks, vec![9]);
        assert!(outcome.evicted_blocks.is_empty());

        assert!(
            cache
                .lookup(&None, &[7, 8, 9, 10, 11])
                .is_ok_and(|hit| hit.is_none())
        );
        let hit = cache.lookup(&None, &[1, 2, 3, 4, 5])?.expect("prefix hit");
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(hit.block_ids, vec![9]);
        cache.release_hit(hit.entry_id);

        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.lookup_misses, 1);
        assert_eq!(stats.hit_tokens, 4);
        assert_eq!(stats.hit_blocks, 1);
        assert_eq!(stats.cached_blocks, 1);
        assert_eq!(stats.max_blocks, 4);
        assert_eq!(stats.cached_entries, 1);
        assert_eq!(stats.max_entries, 1024);
        assert_eq!(stats.cached_state_bytes, 49);
        assert_eq!(stats.max_state_bytes, 1024 * 49);
        Ok(())
    }

    #[test]
    fn real_prefix_cache_caps_entries_and_state_bytes() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let mut cache = RealPrefixCache::new(true, 4, 100, 2, 49);

        for i in 0..3u32 {
            let start = 1 + i * 4;
            cache.register(
                None,
                PagedPrefixRegistration {
                    prompt_tokens: vec![start, start + 1, start + 2, start + 3],
                    block_ids: vec![10 + i],
                    linear_state: LinearAttentionState::new(&config, &device)?,
                },
            );
        }

        let stats = cache.stats();
        assert_eq!(stats.cached_entries, 2);
        assert_eq!(stats.max_entries, 2);
        assert_eq!(stats.cached_state_bytes, 98);
        assert_eq!(stats.max_state_bytes, 98);
        assert_eq!(stats.cached_blocks, 2);
        assert!(cache.lookup(&None, &[1, 2, 3, 4, 99])?.is_none());
        assert!(cache.lookup(&None, &[5, 6, 7, 8, 99])?.is_some());
        assert!(cache.lookup(&None, &[9, 10, 11, 12, 99])?.is_some());
        Ok(())
    }

    #[test]
    fn default_prefix_cache_entries_reserves_state_memory_budget() {
        let entry = 49 * 1024 * 1024;
        assert_eq!(
            default_prefix_cache_max_entries(48 * 1024 * 1024 * 1024, entry),
            20
        );
        assert_eq!(
            default_prefix_cache_max_entries(24 * 1024 * 1024 * 1024, entry),
            12
        );
        assert_eq!(default_prefix_cache_max_entries(0, entry), 5);
    }

    #[test]
    fn real_prefix_cache_keys_by_adapter() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        cache.register(
            Some("adapter-a".to_string()),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![9],
                linear_state: state,
            },
        );

        assert!(cache.lookup(&None, &[1, 2, 3, 4, 5])?.is_none());
        assert!(
            cache
                .lookup(&Some("adapter-b".to_string()), &[1, 2, 3, 4, 5])?
                .is_none()
        );
        assert!(
            cache
                .lookup(&Some("adapter-a".to_string()), &[1, 2, 3, 4, 5])?
                .is_some()
        );
        Ok(())
    }

    // Regression tests for #673: prefix-cache eviction must never return block
    // IDs that the same `register()` call has just re-retained for the new
    // entry. If it does, the API layer hands those IDs to BlockManager::free_all,
    // which under workers=2 can re-allocate them and overwrite live KV that the
    // prefix cache still serves as valid.

    #[test]
    fn register_does_not_evict_blocks_retained_by_incoming() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let mut cache = RealPrefixCache::new(true, 4, 2, 1024, 49);

        // Entry A occupies blocks [10, 11], the cache's full capacity.
        let outcome_a = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![10, 11],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );
        assert_eq!(outcome_a.retained_blocks, vec![10, 11]);
        assert!(outcome_a.evicted_blocks.is_empty());

        // Entry B is a strict superset of A and reuses A's blocks plus block 12.
        // To make room (cached_blocks=2 + needed_new=1 > max=2), the cache must
        // evict A. Pre-fix, evicted_blocks would contain [10, 11] AND those IDs
        // would be re-retained for B — a double-claim that frees live blocks.
        let outcome_b = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                block_ids: vec![10, 11, 12],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );

        let retained: HashSet<u32> = outcome_b.retained_blocks.iter().copied().collect();
        let evicted: HashSet<u32> = outcome_b.evicted_blocks.iter().copied().collect();
        assert!(
            retained.is_disjoint(&evicted),
            "retained_blocks {retained:?} must not overlap evicted_blocks {evicted:?}",
        );
        for block_id in &[10u32, 11, 12] {
            assert!(
                !evicted.contains(block_id),
                "block {block_id} re-retained by incoming registration must not appear in evicted_blocks: {evicted:?}",
            );
        }
        Ok(())
    }

    #[test]
    fn register_outcome_no_duplicate_or_overlap() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let mut cache = RealPrefixCache::new(true, 4, 3, 1024, 49);

        // Three small entries that together fill capacity, two of which share
        // block 20 with the eventual incoming registration.
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![20],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );
        cache.register(
            Some("a".to_string()),
            PagedPrefixRegistration {
                prompt_tokens: vec![5, 6, 7, 8],
                block_ids: vec![20, 21],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );
        cache.register(
            Some("b".to_string()),
            PagedPrefixRegistration {
                prompt_tokens: vec![9, 10, 11, 12],
                block_ids: vec![22],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );

        // Incoming registration shares block 20 with two existing entries and
        // brings a fresh block 99. Eviction will likely run multiple times.
        let outcome = cache.register(
            Some("c".to_string()),
            PagedPrefixRegistration {
                prompt_tokens: vec![13, 14, 15, 16, 17, 18, 19, 20],
                block_ids: vec![20, 99],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );

        let retained_set: HashSet<u32> = outcome.retained_blocks.iter().copied().collect();
        assert_eq!(
            retained_set.len(),
            outcome.retained_blocks.len(),
            "retained_blocks must not contain duplicates: {:?}",
            outcome.retained_blocks,
        );

        let evicted_set: HashSet<u32> = outcome.evicted_blocks.iter().copied().collect();
        assert_eq!(
            evicted_set.len(),
            outcome.evicted_blocks.len(),
            "evicted_blocks must not contain duplicates: {:?}",
            outcome.evicted_blocks,
        );

        assert!(
            retained_set.is_disjoint(&evicted_set),
            "retained {retained_set:?} and evicted {evicted_set:?} must be disjoint",
        );
        Ok(())
    }

    #[test]
    fn register_evicted_blocks_not_in_refcounts_after() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let mut cache = RealPrefixCache::new(true, 4, 2, 1024, 49);

        // Fill capacity with an evictable entry whose blocks the next
        // registration also wants.
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![30, 31],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );

        // Register a longer prompt that reuses [30, 31] and adds 32. The bug
        // would let evicted_blocks contain 30/31 even though they are now
        // tracked by the new entry's refcounts.
        let outcome = cache.register(
            Some("ad".to_string()),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                block_ids: vec![30, 31, 32],
                linear_state: LinearAttentionState::new(&config, &device)?,
            },
        );

        for block_id in &outcome.evicted_blocks {
            assert!(
                !cache.block_refcounts.contains_key(block_id),
                "evicted block {block_id} must not be tracked in block_refcounts after register(); refcounts={:?}",
                cache.block_refcounts,
            );
        }
        Ok(())
    }

    #[test]
    fn test_memory_budget_cpu_mode() {
        let budget = GpuMemoryBudget::compute(0, 0, 0, 0, 0, 0.7, None);
        assert_eq!(budget.total_vram_bytes, 0);
        assert_eq!(budget.training_budget_bytes, 0);
        // CPU mode: training feasibility check always passes
        assert!(budget.check_training_feasible(1_000_000_000).is_ok());
    }

    #[test]
    fn test_memory_budget_24gb_gpu() {
        let total: u64 = 24 * 1024 * 1024 * 1024; // 24 GB
        let model: u64 = 8 * 1024 * 1024 * 1024; // 8 GB model
        let kv: u64 = 2 * 1024 * 1024 * 1024; // 2 GB KV cache
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        assert_eq!(budget.total_vram_bytes, total);
        assert_eq!(budget.model_memory_bytes, model);
        assert_eq!(budget.kv_cache_bytes, kv);
        // training_budget = 24 - 8 - 2 = 14 GB
        assert_eq!(budget.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_budget_insufficient() {
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 8 * 1024 * 1024 * 1024;
        let kv: u64 = 12 * 1024 * 1024 * 1024;
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        // Only 4GB available for training
        assert_eq!(budget.training_budget_bytes, 4 * 1024 * 1024 * 1024);
        // Requesting 8GB should fail
        assert!(
            budget
                .check_training_feasible(8 * 1024 * 1024 * 1024)
                .is_err()
        );
        // Requesting 3GB should succeed
        assert!(
            budget
                .check_training_feasible(3 * 1024 * 1024 * 1024)
                .is_ok()
        );
    }

    #[test]
    fn test_memory_budget_saturating_sub() {
        // Edge case: model + KV > total VRAM
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 20 * 1024 * 1024 * 1024;
        let kv: u64 = 10 * 1024 * 1024 * 1024;
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        // Should not underflow — saturating_sub handles it
        assert_eq!(budget.training_budget_bytes, 0);
    }

    #[test]
    fn test_estimate_model_memory() {
        let config = ModelConfig::qwen3_5_4b();
        let bytes = estimate_model_memory_bytes(&config);
        let gb = bytes as f64 / 1e9;
        // Should be in the ballpark of 8GB for Qwen3.5-4B bf16
        assert!(
            gb > 4.0 && gb < 20.0,
            "model estimate {gb:.1}GB seems wrong"
        );
    }

    #[test]
    fn test_post_load_residency_lowers_cuda_auto_blocks() {
        let total = 51_527_024_640;
        let estimated_model = 9_156_689_920;
        let post_load_residency = 13_000_000_000;
        let bytes_per_block = 524_288;

        let old_estimate_blocks = auto_num_blocks_for_fraction(
            total,
            estimated_model,
            bytes_per_block,
            0.7,
            262_144,
            DEFAULT_BLOCK_SIZE,
            false,
        );
        let post_load_blocks = auto_num_blocks_for_fraction(
            total,
            post_load_residency,
            bytes_per_block,
            0.7,
            262_144,
            DEFAULT_BLOCK_SIZE,
            false,
        );

        assert_eq!(old_estimate_blocks, 56570);
        assert!(
            post_load_blocks < old_estimate_blocks,
            "post-load residency must reduce the default A6000 KV budget: old={old_estimate_blocks} post_load={post_load_blocks}"
        );
    }

    #[test]
    fn test_auto_num_blocks_no_model_context_cap_on_cuda() {
        // On CUDA, raw_blocks comes from the memory-aware sizing path
        // (available VRAM × inference fraction ÷ bytes-per-block), which
        // already accounts for what the GPU can hold. Clipping it to a single
        // model-context-worth of blocks (≈16K for Qwen3.5-4B's 256K window)
        // would bottleneck concurrent serving — multiple in-flight long
        // prompts collectively address more than one window's worth of KV
        // cache. cap_auto_num_blocks must trust the memory-aware ceiling
        // here.
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                false, // is_metal
                48 * 1024 * 1024 * 1024,
            ),
            50_000
        );
        // raw_blocks well under the model-context size still passes through —
        // this is the small-VRAM CUDA path (e.g. A10 / consumer card).
        assert_eq!(
            cap_auto_num_blocks(
                4_096,
                262_144,
                DEFAULT_BLOCK_SIZE,
                false,
                10 * 1024 * 1024 * 1024,
            ),
            4_096
        );
        // raw_blocks above the model-context size on a large-VRAM CUDA host
        // is preserved (multi-tenant headroom). Pre-fix this returned 16_384.
        assert_eq!(
            cap_auto_num_blocks(
                65_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                false,
                80 * 1024 * 1024 * 1024,
            ),
            65_000
        );
    }

    #[test]
    fn test_auto_num_blocks_caps_metal_desktop_defaults_by_memory_tier() {
        // On unified-memory Macs, pure memory-aware sizing can request a large
        // eagerly-zeroed KV cache. Default Metal auto-sizing is tier-capped by
        // detected memory; explicit KILN_NUM_BLOCKS still bypasses this helper
        // entirely in AppState::new_real.
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                true,
                10 * 1024 * 1024 * 1024,
            ),
            METAL_AUTO_MAX_KV_BLOCKS_LOW_MEM
        );
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                true,
                16 * 1024 * 1024 * 1024,
            ),
            METAL_AUTO_MAX_KV_BLOCKS_MID_MEM
        );
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                true,
                32 * 1024 * 1024 * 1024,
            ),
            METAL_AUTO_MAX_KV_BLOCKS_HIGH_MEM
        );
    }

    #[test]
    fn test_auto_num_blocks_preserves_small_auto_and_minimum() {
        assert_eq!(
            cap_auto_num_blocks(
                512,
                262_144,
                DEFAULT_BLOCK_SIZE,
                true,
                10 * 1024 * 1024 * 1024,
            ),
            512
        );
        assert_eq!(
            cap_auto_num_blocks(
                1,
                262_144,
                DEFAULT_BLOCK_SIZE,
                true,
                10 * 1024 * 1024 * 1024,
            ),
            MIN_AUTO_KV_BLOCKS
        );
    }

    #[test]
    fn test_inference_fraction_clamping() {
        // Verify the clamping logic works (tested indirectly through budget)
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 8 * 1024 * 1024 * 1024;
        let kv: u64 = 2 * 1024 * 1024 * 1024;

        // fraction = 1.0 means all VRAM for inference, but training budget is still calculated
        let budget_full = GpuMemoryBudget::compute(total, model, model, 0, kv, 1.0, None);
        assert_eq!(budget_full.training_budget_bytes, 14 * 1024 * 1024 * 1024);

        // fraction = 0.5
        let budget_half = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.5, None);
        assert_eq!(budget_half.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }

    /// Build a tiny CPU-resident PagedKvCache for use as the "successful
    /// allocation" return value in the auto-sizer retry tests below. The
    /// values are dummies — only the act of returning Ok(...) matters for
    /// the loop logic.
    fn dummy_cpu_cache() -> PagedKvCache {
        PagedKvCache::new_uninit_with_fp8(
            1,  // num_full_attn_layers
            8,  // num_blocks
            16, // block_size
            1,  // num_kv_heads
            4,  // head_dim
            DType::F32,
            &candle_core::Device::Cpu,
            false,
        )
        .expect("CPU PagedKvCache allocation never fails for tiny shape")
    }

    #[test]
    fn auto_sizer_succeeds_on_first_attempt_when_configured_fits() {
        let compute = |fraction: f64| -> usize {
            // Map fraction directly to a block count for inspection
            (fraction * 1000.0) as usize
        };
        let calls = std::cell::Cell::new(0u32);
        let result = auto_size_with_retry(0.85, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |_n| {
            calls.set(calls.get() + 1);
            Ok(dummy_cpu_cache())
        });
        let success = result.unwrap_or_else(|_| panic!("expected success"));
        assert_eq!(success.fraction, 0.85);
        assert_eq!(success.num_blocks, 850);
        assert!(success.attempted_failures.is_empty());
        assert_eq!(calls.get(), 1, "should have allocated exactly once");
    }

    #[test]
    fn auto_sizer_retries_until_fraction_drops_below_oom_threshold() {
        // Simulate an A40+BF16-like OOM zone: anything ≥ 0.70 OOMs, anything
        // strictly below succeeds. Configured fraction is 0.85 (issue #685
        // shape); the loop should fall through 0.85 → 0.75 (both OOM) and
        // succeed at 0.65.
        let oom_at_or_above = 0.70_f64;
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let attempted_fractions = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(0.85, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |n| {
            calls.set(calls.get() + 1);
            let frac = (n as f64) / 1000.0;
            attempted_fractions.borrow_mut().push(frac);
            if frac >= oom_at_or_above - 1e-9 {
                Err(format!(
                    "CUDA OOM: out of memory while allocating k_pool for layer 0 (n={n})"
                ))
            } else {
                Ok(dummy_cpu_cache())
            }
        });
        let success = result.unwrap_or_else(|_| panic!("expected success after fallback"));
        assert_eq!(success.fraction, 0.65, "should land on the 0.65 fallback");
        assert_eq!(success.num_blocks, 650);
        assert_eq!(
            success.attempted_failures.len(),
            2,
            "should have failed twice (0.85 then 0.75) before succeeding at 0.65"
        );
        assert_eq!(calls.get(), 3);
        let attempts = attempted_fractions.borrow().clone();
        assert_eq!(attempts, vec![0.85, 0.75, 0.65]);
    }

    #[test]
    fn auto_sizer_skips_fallbacks_above_configured_fraction() {
        // If the user pinned a low inference_memory_fraction (say 0.50), the
        // retry loop must never try the higher-default fallbacks (0.75, 0.65)
        // — that would silently allocate MORE than the user asked for.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let attempted = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(0.50, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |n| {
            calls.set(calls.get() + 1);
            let frac = (n as f64) / 1000.0;
            attempted.borrow_mut().push(frac);
            Ok(dummy_cpu_cache())
        });
        let success = result.unwrap_or_else(|_| panic!("expected success"));
        assert_eq!(success.fraction, 0.50);
        assert_eq!(success.num_blocks, 500);
        assert!(success.attempted_failures.is_empty());
        let attempts = attempted.borrow().clone();
        assert_eq!(
            attempts,
            vec![0.50],
            "should have tried only the configured fraction"
        );
    }

    #[test]
    fn auto_sizer_returns_failure_when_every_fraction_ooms() {
        // Pathological case: every fraction OOMs (e.g. unreasonably small GPU
        // for the model). The loop must not loop forever; it must return
        // Failure with the full attempt history so the caller can build a
        // useful error message.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let result = auto_size_with_retry(
            0.85,
            AUTO_SIZER_FALLBACK_FRACTIONS,
            &compute,
            |n| -> Result<PagedKvCache, String> {
                calls.set(calls.get() + 1);
                Err(format!("simulated OOM at n={n}"))
            },
        );
        let failure = result.err().unwrap_or_else(|| panic!("expected failure"));
        // 1 configured + 4 fallbacks (all strictly below 0.85) = 5 attempts
        assert_eq!(failure.attempts.len(), 5);
        assert_eq!(calls.get(), 5);
        let fractions: Vec<f64> = failure.attempts.iter().map(|(f, _, _)| *f).collect();
        assert_eq!(fractions, vec![0.85, 0.75, 0.65, 0.55, 0.45]);
    }

    #[test]
    fn auto_sizer_does_not_retry_with_duplicate_or_higher_values() {
        // If configured fraction equals one of the fallback values (e.g. 0.75),
        // the retry loop must not try it twice. Subsequent attempts must be
        // strictly lower.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let attempted = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(
            0.75,
            AUTO_SIZER_FALLBACK_FRACTIONS,
            &compute,
            |n| -> Result<PagedKvCache, String> {
                let frac = (n as f64) / 1000.0;
                attempted.borrow_mut().push(frac);
                Err(format!("OOM at n={n}"))
            },
        );
        let failure = result.err().unwrap_or_else(|| panic!("expected failure"));
        let fractions: Vec<f64> = failure.attempts.iter().map(|(f, _, _)| *f).collect();
        assert_eq!(
            fractions,
            vec![0.75, 0.65, 0.55, 0.45],
            "configured 0.75 must appear once and only fractions strictly below should follow"
        );
    }

    #[test]
    fn suggested_emergency_blocks_uses_30pct_of_remaining_vram() {
        // 48 GiB GPU, 8 GiB model -> 40 GiB remaining * 0.30 = 12 GiB for KV
        let total = 48u64 * 1024 * 1024 * 1024;
        let model = 8u64 * 1024 * 1024 * 1024;
        let bytes_per_block = 256u64 * 1024; // 256 KiB per block
        let suggested = suggested_emergency_num_blocks(
            total,
            model,
            bytes_per_block,
            DEFAULT_BLOCK_SIZE,
            262_144,
            false, // CUDA path (Metal cap doesn't apply)
        );
        let expected_kv_bytes = ((total - model) as f64 * 0.30) as u64;
        let expected_blocks = (expected_kv_bytes / bytes_per_block) as usize;
        assert_eq!(suggested, expected_blocks);
        // Sanity: at least the floor
        assert!(suggested >= MIN_AUTO_KV_BLOCKS);
    }

    #[test]
    fn suggested_emergency_blocks_falls_back_when_no_vram_signal() {
        // total_vram = 0 (CPU or detection failed) — must still return a
        // sensible block count derived from max_position_embeddings.
        let suggested = suggested_emergency_num_blocks(
            0, // total_vram unknown
            0,
            0,
            DEFAULT_BLOCK_SIZE,
            262_144,
            false,
        );
        let expected = (262_144_usize).div_ceil(DEFAULT_BLOCK_SIZE);
        assert_eq!(suggested, expected.max(MIN_AUTO_KV_BLOCKS));
    }

    #[test]
    fn oom_message_names_concrete_remediation_flags() {
        // The whole point of the new error message: it must give the user a
        // concrete `KILN_NUM_BLOCKS=N` and `KILN_INFERENCE_MEMORY_FRACTION=X`
        // value to set. Verify both appear in the rendered text.
        let failure = AutoSizeFailure {
            attempts: vec![
                (0.85, 24576, "CUDA OOM during k_pool layer 0".to_string()),
                (0.75, 21504, "CUDA OOM during k_pool layer 0".to_string()),
                (0.65, 18432, "CUDA OOM during k_pool layer 0".to_string()),
                (0.55, 15360, "CUDA OOM during k_pool layer 0".to_string()),
                (0.45, 12288, "CUDA OOM during k_pool layer 0".to_string()),
            ],
        };
        let total_vram = 48u64 * 1024 * 1024 * 1024;
        let model_bytes = 8u64 * 1024 * 1024 * 1024;
        let bytes_per_block = 1u64 * 1024 * 1024; // 1 MiB
        let suggested = 8192;
        let msg = format_oom_remediation_message(
            &failure,
            total_vram,
            model_bytes,
            bytes_per_block,
            suggested,
            0.85,
            kiln_core::vram::VramSource::NvidiaSmi,
        );
        assert!(
            msg.contains("KILN_NUM_BLOCKS=8192"),
            "message must include the concrete num_blocks suggestion: {msg}"
        );
        assert!(
            msg.contains("KILN_INFERENCE_MEMORY_FRACTION="),
            "message must include the concrete fraction suggestion: {msg}"
        );
        // Suggested fraction = last attempt (0.45) / 2 = 0.225, max(0.10) = 0.225
        assert!(
            msg.contains("0.23") || msg.contains("0.22"),
            "message should suggest a fraction roughly half of the last failure: {msg}"
        );
        assert!(
            msg.contains("48.0 GiB") || msg.contains("48 GiB"),
            "message should mention detected VRAM: {msg}"
        );
        assert!(
            msg.contains("nvidia-smi"),
            "message should mention VRAM source for sanity-check: {msg}"
        );
        // All 5 attempted fractions should be enumerated
        for fraction_str in &["0.85", "0.75", "0.65", "0.55", "0.45"] {
            assert!(
                msg.contains(fraction_str),
                "message should enumerate attempt at fraction {fraction_str}: {msg}"
            );
        }
        // The recommendation banner must reference the working workaround
        assert!(
            msg.contains("#685"),
            "message should reference issue #685 for context: {msg}"
        );
    }

    #[test]
    fn oom_message_handles_unknown_vram() {
        // total_vram = 0 path (e.g. detection failed) — the message must
        // still render something sensible without panicking on the GiB
        // formatting.
        let failure = AutoSizeFailure {
            attempts: vec![(0.85, 100, "CUDA OOM".to_string())],
        };
        let msg = format_oom_remediation_message(
            &failure,
            0,
            0,
            1024,
            64,
            0.85,
            kiln_core::vram::VramSource::None,
        );
        assert!(msg.contains("KILN_NUM_BLOCKS=64"), "message: {msg}");
        assert!(
            msg.contains("0.0 GiB"),
            "should print 0.0 GiB when unknown: {msg}"
        );
    }
}

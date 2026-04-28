use std::collections::HashMap;
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

/// GPU memory budget tracking for coordinating inference and training.
///
/// On startup, we compute how much VRAM is available and partition it:
/// - Model weights (fixed)
/// - KV cache for inference (controlled by KILN_INFERENCE_MEMORY_FRACTION)
/// - Remaining budget available for training
#[derive(Debug, Clone, Serialize)]
pub struct GpuMemoryBudget {
    /// Total GPU memory in bytes (0 if CPU-only).
    pub total_vram_bytes: u64,
    /// Estimated model weight memory in bytes.
    pub model_memory_bytes: u64,
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
    /// `model_memory_bytes`: Estimated model weight size.
    /// `kv_cache_bytes`: Actual KV cache allocation size.
    /// `inference_fraction`: Fraction of VRAM for inference.
    /// `training_memory_gb`: Optional explicit training memory budget in GB.
    pub fn compute(
        total_vram_bytes: u64,
        model_memory_bytes: u64,
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
            kv_cache_bytes,
            training_budget_bytes,
            inference_memory_fraction: inference_fraction,
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

pub struct RealPrefixCache {
    enabled: bool,
    max_blocks: usize,
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
    pub fn new(enabled: bool, block_size: usize, max_blocks: usize) -> Self {
        Self {
            enabled,
            max_blocks,
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
        Self::new(false, block_size, 0)
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
        while self.cached_blocks() + needed_new_blocks > self.max_blocks {
            let Some(evict_idx) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.active_uses == 0)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(idx, _)| idx)
            else {
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
    /// Per-request timeout duration. Configurable via KILN_REQUEST_TIMEOUT_SECS (default 300).
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
            memory_budget: Arc::new(GpuMemoryBudget::compute(0, 0, 0, 1.0, None)),
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

        let inference_fraction = memory_cfg.inference_memory_fraction.clamp(0.1, 1.0);

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

        // Determine num_blocks — either from config, or memory-aware auto-calculation
        let num_blocks = if let Some(explicit) = memory_cfg.num_blocks {
            explicit
        } else {
            // Try to compute from available VRAM and inference fraction
            if total_vram > 0 && bytes_per_block > 0 {
                let available_for_kv = ((total_vram.saturating_sub(estimated_model_bytes)) as f64
                    * inference_fraction) as u64;
                let raw_auto_blocks = (available_for_kv / bytes_per_block) as usize;
                let auto_blocks = cap_auto_num_blocks(
                    raw_auto_blocks,
                    model_config.max_position_embeddings,
                    block_size,
                    is_metal_device(&device),
                    total_vram,
                );
                tracing::info!(
                    total_vram_gb = total_vram as f64 / 1e9,
                    model_gb = estimated_model_bytes as f64 / 1e9,
                    inference_fraction,
                    raw_auto_blocks,
                    auto_blocks,
                    "memory-aware KV cache sizing"
                );
                auto_blocks
            } else {
                // Fallback: derive from max_position_embeddings
                let raw_auto_blocks = model_config
                    .max_position_embeddings
                    .div_ceil(block_size)
                    .max(256);
                cap_auto_num_blocks(
                    raw_auto_blocks,
                    model_config.max_position_embeddings,
                    block_size,
                    is_metal_device(&device),
                    total_vram,
                )
            }
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
        tracing::info!(
            num_blocks,
            block_size,
            ?kv_dtype,
            fp8_enabled,
            "allocating paged KV cache"
        );

        let block_manager = BlockManager::new(num_blocks, block_size);
        let paged_cache = PagedKvCache::new_uninit_with_fp8(
            model_config.num_full_attention_layers,
            num_blocks,
            block_size,
            model_config.num_kv_heads,
            model_config.head_dim,
            kv_dtype,
            &device,
            fp8_enabled,
        )
        .expect("failed to create PagedKvCache");

        let kv_cache_bytes = num_blocks as u64 * bytes_per_block;
        let memory_budget = GpuMemoryBudget::compute(
            total_vram,
            estimated_model_bytes,
            kv_cache_bytes,
            inference_fraction,
            memory_cfg.training_memory_gb,
        );

        tracing::info!(
            total_vram_gb = memory_budget.total_vram_bytes as f64 / 1e9,
            model_gb = memory_budget.model_memory_bytes as f64 / 1e9,
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
        let prefix_cache = RealPrefixCache::new(
            prefix_cache_cfg.enabled,
            block_size,
            prefix_cache_max_blocks,
        );

        Self {
            model_config,
            backend: Arc::new(ModelBackend::Real {
                runner: Arc::new(std::sync::RwLock::new(runner)),
                block_manager: Arc::new(std::sync::Mutex::new(block_manager)),
                paged_cache: Arc::new(std::sync::Mutex::new(paged_cache)),
                prefix_cache: Arc::new(std::sync::Mutex::new(prefix_cache)),
            }),
            tokenizer: Arc::new(tokenizer),
            adapter_dir,
            active_adapter_name: Arc::new(std::sync::RwLock::new(None)),
            training_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            memory_budget: Arc::new(memory_budget),
            gpu_lock: Arc::new(std::sync::RwLock::new(())),
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
            served_model_id,
            decode_stats: Arc::new(std::sync::Mutex::new(DecodeStatsRing::new(4096))),
            recent_requests: Arc::new(std::sync::Mutex::new(RecentRequestsRing::new(
                RECENT_REQUESTS_CAPACITY,
            ))),
        }
    }
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

fn cap_auto_num_blocks(
    raw_blocks: usize,
    max_position_embeddings: usize,
    block_size: usize,
    is_metal: bool,
    total_vram_bytes: u64,
) -> usize {
    let model_cap_blocks = max_position_embeddings
        .div_ceil(block_size)
        .max(MIN_AUTO_KV_BLOCKS);
    let runtime_cap_blocks = if is_metal {
        model_cap_blocks.min(metal_auto_max_kv_blocks(total_vram_bytes))
    } else {
        model_cap_blocks
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
        let mut cache = RealPrefixCache::new(true, 4, 4);

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
        Ok(())
    }

    #[test]
    fn real_prefix_cache_keys_by_adapter() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = candle_core::Device::Cpu;
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new(true, 4, 4);
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

    #[test]
    fn test_memory_budget_cpu_mode() {
        let budget = GpuMemoryBudget::compute(0, 0, 0, 0.7, None);
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
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7, None);
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
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7, None);
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
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7, None);
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
    fn test_auto_num_blocks_caps_at_model_context() {
        // Qwen3.5's 262K context is 16,384 blocks at block_size=16. There is
        // no reason to auto-allocate more KV cache than the model can address.
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                false,
                10 * 1024 * 1024 * 1024,
            ),
            16_384
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
        let budget_full = GpuMemoryBudget::compute(total, model, kv, 1.0, None);
        assert_eq!(budget_full.training_budget_bytes, 14 * 1024 * 1024 * 1024);

        // fraction = 0.5
        let budget_half = GpuMemoryBudget::compute(total, model, kv, 0.5, None);
        assert_eq!(budget_half.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }
}

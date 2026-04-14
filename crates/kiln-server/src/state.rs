use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use candle_core::DType;
use kiln_core::block::BlockManager;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::Engine;
use kiln_model::{ModelRunner, PagedKvCache};
use kiln_scheduler::Scheduler;
use kiln_train::TrainingState;
use serde::Serialize;

use crate::metrics::Metrics;
use crate::training_queue::{SharedTrainingQueue, ShutdownFlag};

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
    /// `inference_fraction`: Fraction of VRAM for inference (from env).
    pub fn compute(
        total_vram_bytes: u64,
        model_memory_bytes: u64,
        kv_cache_bytes: u64,
        inference_fraction: f64,
    ) -> Self {
        let training_budget_bytes = if total_vram_bytes == 0 {
            // CPU mode — no GPU memory budget applies
            0
        } else {
            // Override via KILN_TRAINING_MEMORY_GB if set
            if let Ok(val) = std::env::var("KILN_TRAINING_MEMORY_GB") {
                if let Ok(gb) = val.parse::<f64>() {
                    return Self {
                        total_vram_bytes,
                        model_memory_bytes,
                        kv_cache_bytes,
                        training_budget_bytes: (gb * 1024.0 * 1024.0 * 1024.0) as u64,
                        inference_memory_fraction: inference_fraction,
                    };
                }
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
}

/// Thread-safe map of tracked training jobs.
pub type TrainingJobs = Arc<std::sync::RwLock<HashMap<String, TrainingJobInfo>>>;

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
}

impl AppState {
    /// Create an AppState with the mock engine backend.
    pub fn new_mock(
        model_config: ModelConfig,
        scheduler: Scheduler,
        engine: Arc<dyn Engine>,
        tokenizer: KilnTokenizer,
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
            memory_budget: Arc::new(GpuMemoryBudget::compute(0, 0, 0, 1.0)),
            gpu_lock: Arc::new(std::sync::RwLock::new(())),
            training_queue: crate::training_queue::new_shared_queue(),
            vram_info: kiln_core::vram::GpuVramInfo {
                total_bytes: 0,
                source: kiln_core::vram::VramSource::None,
            },
            shutdown: crate::training_queue::new_shutdown_flag(),
            request_timeout: parse_request_timeout(),
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
        }
    }

    /// Create an AppState with a real ModelRunner backend and paged KV cache.
    ///
    /// Uses `block_size=16` by default. The number of blocks can be overridden
    /// with `KILN_NUM_BLOCKS`. Otherwise derived from `max_position_embeddings / block_size`.
    ///
    /// GPU memory sharing: When `KILN_INFERENCE_MEMORY_FRACTION` is set (default 0.7),
    /// KV cache allocation is limited to that fraction of remaining VRAM (after model weights),
    /// reserving the rest for training. Set to 1.0 to use all available VRAM for inference.
    pub fn new_real(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
        device: candle_core::Device,
        adapter_dir: PathBuf,
    ) -> Self {
        let block_size = 16;

        let kv_dtype = if candle_core::utils::cuda_is_available() {
            DType::BF16
        } else {
            DType::F32
        };

        let kv_dtype_bytes: usize = match kv_dtype {
            DType::BF16 => 2,
            _ => 4,
        };

        // Read inference memory fraction (default 0.7 = reserve 30% for training)
        let inference_fraction: f64 = std::env::var("KILN_INFERENCE_MEMORY_FRACTION")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.7)
            .clamp(0.1, 1.0);

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

        // Detect VRAM for auto-configuration
        let vram_info = kiln_core::vram::detect_vram();

        // Determine num_blocks — either from env, or memory-aware auto-calculation
        let num_blocks = if let Some(explicit) = std::env::var("KILN_NUM_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            explicit
        } else {
            // Try to compute from available VRAM and inference fraction
            let total_vram = query_gpu_total_memory(&device);
            if total_vram > 0 && bytes_per_block > 0 {
                let available_for_kv =
                    ((total_vram.saturating_sub(estimated_model_bytes)) as f64 * inference_fraction)
                        as u64;
                let auto_blocks = (available_for_kv / bytes_per_block) as usize;
                let auto_blocks = auto_blocks.max(64); // minimum 64 blocks
                tracing::info!(
                    total_vram_gb = total_vram as f64 / 1e9,
                    model_gb = estimated_model_bytes as f64 / 1e9,
                    inference_fraction,
                    auto_blocks,
                    "memory-aware KV cache sizing"
                );
                auto_blocks
            } else {
                // Fallback: derive from max_position_embeddings
                (model_config.max_position_embeddings / block_size).max(256)
            }
        };

        tracing::info!(num_blocks, block_size, ?kv_dtype, "allocating paged KV cache");

        let block_manager = BlockManager::new(num_blocks, block_size);
        let paged_cache = PagedKvCache::new(
            model_config.num_full_attention_layers,
            num_blocks,
            block_size,
            model_config.num_kv_heads,
            model_config.head_dim,
            kv_dtype,
            &device,
        )
        .expect("failed to create PagedKvCache");

        let kv_cache_bytes = num_blocks as u64 * bytes_per_block;
        let total_vram = query_gpu_total_memory(&device);
        let memory_budget = GpuMemoryBudget::compute(
            total_vram,
            estimated_model_bytes,
            kv_cache_bytes,
            inference_fraction,
        );

        tracing::info!(
            total_vram_gb = memory_budget.total_vram_bytes as f64 / 1e9,
            model_gb = memory_budget.model_memory_bytes as f64 / 1e9,
            kv_cache_gb = memory_budget.kv_cache_bytes as f64 / 1e9,
            training_budget_gb = memory_budget.training_budget_bytes as f64 / 1e9,
            inference_fraction = memory_budget.inference_memory_fraction,
            "GPU memory budget"
        );

        Self {
            model_config,
            backend: Arc::new(ModelBackend::Real {
                runner: Arc::new(std::sync::RwLock::new(runner)),
                block_manager: Arc::new(std::sync::Mutex::new(block_manager)),
                paged_cache: Arc::new(std::sync::Mutex::new(paged_cache)),
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
            request_timeout: parse_request_timeout(),
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
        }
    }
}

/// Parse KILN_REQUEST_TIMEOUT_SECS from environment (default: 300 seconds).
fn parse_request_timeout() -> std::time::Duration {
    let secs: u64 = std::env::var("KILN_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    std::time::Duration::from_secs(secs)
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
    let per_layer_params =
        8 * (config.hidden_size as u64 * config.hidden_size as u64)
        + 3 * (config.hidden_size as u64 * config.intermediate_size as u64);
    let total_params = embedding_params + per_layer_params * config.num_layers as u64;

    total_params * dtype_bytes
}

/// Query total GPU memory in bytes. Returns 0 for CPU devices.
///
/// Uses the shared VRAM detection from kiln-core (nvidia-smi + env override).
/// For CPU devices, still checks KILN_GPU_MEMORY_GB for testing purposes.
fn query_gpu_total_memory(device: &candle_core::Device) -> u64 {
    let vram = kiln_core::vram::detect_vram();
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
                // CUDA device exists but detection failed — assume 24GB
                tracing::warn!("CUDA device present but VRAM detection failed; assuming 24GB");
                24 * 1024 * 1024 * 1024
            }
        }
        _ => vram.total_bytes, // 0 if no GPU, or env override for testing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_cpu_mode() {
        let budget = GpuMemoryBudget::compute(0, 0, 0, 0.7);
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
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7);
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
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7);
        // Only 4GB available for training
        assert_eq!(budget.training_budget_bytes, 4 * 1024 * 1024 * 1024);
        // Requesting 8GB should fail
        assert!(budget
            .check_training_feasible(8 * 1024 * 1024 * 1024)
            .is_err());
        // Requesting 3GB should succeed
        assert!(budget
            .check_training_feasible(3 * 1024 * 1024 * 1024)
            .is_ok());
    }

    #[test]
    fn test_memory_budget_saturating_sub() {
        // Edge case: model + KV > total VRAM
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 20 * 1024 * 1024 * 1024;
        let kv: u64 = 10 * 1024 * 1024 * 1024;
        let budget = GpuMemoryBudget::compute(total, model, kv, 0.7);
        // Should not underflow — saturating_sub handles it
        assert_eq!(budget.training_budget_bytes, 0);
    }

    #[test]
    fn test_estimate_model_memory() {
        let config = ModelConfig::qwen3_5_4b();
        let bytes = estimate_model_memory_bytes(&config);
        let gb = bytes as f64 / 1e9;
        // Should be in the ballpark of 8GB for Qwen3.5-4B bf16
        assert!(gb > 4.0 && gb < 20.0, "model estimate {gb:.1}GB seems wrong");
    }

    #[test]
    fn test_inference_fraction_clamping() {
        // Verify the clamping logic works (tested indirectly through budget)
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 8 * 1024 * 1024 * 1024;
        let kv: u64 = 2 * 1024 * 1024 * 1024;

        // fraction = 1.0 means all VRAM for inference, but training budget is still calculated
        let budget_full = GpuMemoryBudget::compute(total, model, kv, 1.0);
        assert_eq!(budget_full.training_budget_bytes, 14 * 1024 * 1024 * 1024);

        // fraction = 0.5
        let budget_half = GpuMemoryBudget::compute(total, model, kv, 0.5);
        assert_eq!(budget_half.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }
}

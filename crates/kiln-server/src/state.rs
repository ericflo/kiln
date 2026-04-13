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
        }
    }

    /// Create an AppState with a real ModelRunner backend and paged KV cache.
    ///
    /// Uses `block_size=16` by default. The number of blocks can be overridden
    /// with `KILN_NUM_BLOCKS`. Otherwise derived from `max_position_embeddings / block_size`.
    pub fn new_real(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
        device: candle_core::Device,
        adapter_dir: PathBuf,
    ) -> Self {
        let block_size = 16;
        let num_blocks = std::env::var("KILN_NUM_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| (model_config.max_position_embeddings / block_size).max(256));

        let kv_dtype = if candle_core::utils::cuda_is_available() {
            DType::BF16
        } else {
            DType::F32
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
        }
    }
}

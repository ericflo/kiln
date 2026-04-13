use std::sync::Arc;
use tokio::sync::Mutex;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::Engine;
use kiln_model::ModelRunner;
use kiln_scheduler::Scheduler;

/// Which inference backend the server is using.
pub enum ModelBackend {
    /// Mock engine + scheduler for testing without real weights.
    Mock {
        scheduler: Arc<Mutex<Scheduler>>,
        engine: Arc<dyn Engine>,
    },
    /// Real model weights loaded via ModelRunner.
    Real(Arc<ModelRunner>),
}

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_config: ModelConfig,
    pub backend: Arc<ModelBackend>,
    pub tokenizer: Arc<KilnTokenizer>,
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
        }
    }

    /// Create an AppState with a real ModelRunner backend.
    pub fn new_real(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
    ) -> Self {
        Self {
            model_config,
            backend: Arc::new(ModelBackend::Real(Arc::new(runner))),
            tokenizer: Arc::new(tokenizer),
        }
    }
}

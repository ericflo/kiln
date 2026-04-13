use std::sync::Arc;
use tokio::sync::Mutex;

use kiln_core::config::ModelConfig;
use kiln_model::engine::Engine;
use kiln_scheduler::Scheduler;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_config: ModelConfig,
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub engine: Arc<dyn Engine>,
}

impl AppState {
    pub fn new(
        model_config: ModelConfig,
        scheduler: Scheduler,
        engine: Arc<dyn Engine>,
    ) -> Self {
        Self {
            model_config,
            scheduler: Arc::new(Mutex::new(scheduler)),
            engine,
        }
    }
}

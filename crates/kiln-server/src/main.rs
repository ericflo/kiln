use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

mod api;
mod state;

use kiln_core::config::ModelConfig;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use state::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("kiln=info".parse()?))
        .init();

    let host = std::env::var("KILN_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("KILN_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8420);

    let model_config = ModelConfig::qwen3_4b();

    let scheduler_config = SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
    };

    // Calculate number of KV cache blocks based on available memory.
    // For now, use a fixed number. Real implementation will query GPU memory.
    // Target: 131072 tokens (full 128K context) = 8192 blocks at block_size=16.
    // Qwen3-4B KV cache at 128K BF16: ~18 GB (147 KB/token * 131072 tokens).
    // On a 24GB GPU with 8GB model weights, we have ~14GB for KV — enough for
    // one full 128K sequence or several shorter ones.
    let num_blocks = 8192; // 131072 tokens of KV cache with block_size=16

    let scheduler = Scheduler::new(scheduler_config, num_blocks);
    let engine = MockEngine::new(model_config.clone());

    let state = AppState::new(model_config, scheduler, Arc::new(engine));

    let app = api::router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("kiln listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

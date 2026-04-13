use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

mod api;
mod state;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
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

    let model_config = ModelConfig::qwen3_5_4b();

    let scheduler_config = SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
    };

    // Calculate number of KV cache blocks based on available memory.
    // For now, use a fixed number. Real implementation will query GPU memory.
    // Target: 131072 tokens (full 128K context) = 8192 blocks at block_size=16.
    // Qwen3.5-4B only needs KV cache for 8 full-attention layers (out of 32 total).
    // KV at 128K BF16: ~4 GB (32 KB/token * 131072 tokens). This is the hybrid
    // architecture payoff — 128K context fits comfortably on a 24GB GPU.
    let num_blocks = 8192; // 131072 tokens of KV cache with block_size=16

    let scheduler = Scheduler::new(scheduler_config, num_blocks);
    let engine = MockEngine::new(model_config.clone());

    // Load tokenizer: try from_pretrained (HF Hub), fall back to local path, then fail gracefully.
    let model_id =
        std::env::var("KILN_MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen3.5-4B".to_string());
    let tokenizer_path = std::env::var("KILN_TOKENIZER_PATH").ok();

    let tokenizer = if let Some(path) = tokenizer_path {
        tracing::info!("loading tokenizer from {path}");
        KilnTokenizer::from_file(&path)?
    } else {
        tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
        KilnTokenizer::from_pretrained(&model_id)?
    };

    tracing::info!(
        vocab_size = tokenizer.vocab_size(),
        "tokenizer loaded successfully"
    );

    let state = AppState::new(model_config, scheduler, Arc::new(engine), tokenizer);

    let app = api::router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("kiln listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

use kiln_server::api;
use kiln_server::state;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_model::forward::GpuWeights;
use kiln_model::ModelRunner;
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
    let model_path = std::env::var("KILN_MODEL_PATH").ok();

    // Load tokenizer: try from_pretrained (HF Hub), fall back to local path, then fail gracefully.
    let model_id =
        std::env::var("KILN_MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen3.5-4B".to_string());
    let tokenizer_path = std::env::var("KILN_TOKENIZER_PATH").ok();

    let tokenizer = if let Some(path) = &tokenizer_path {
        tracing::info!("loading tokenizer from {path}");
        KilnTokenizer::from_file(path)?
    } else if let Some(ref mp) = model_path {
        // Try loading tokenizer from the model directory first
        let tok_file = Path::new(mp).join("tokenizer.json");
        if tok_file.exists() {
            tracing::info!("loading tokenizer from model directory: {}", tok_file.display());
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
            KilnTokenizer::from_pretrained(&model_id)?
        }
    } else {
        tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
        KilnTokenizer::from_pretrained(&model_id)?
    };

    tracing::info!(
        vocab_size = tokenizer.vocab_size(),
        "tokenizer loaded successfully"
    );

    let state = if let Some(ref mp) = model_path {
        // Real inference mode: load model weights and create ModelRunner.
        tracing::info!("loading model weights from {mp}");
        let model_weights = kiln_model::load_model(Path::new(mp), &model_config)?;
        let device = if candle_core::utils::cuda_is_available() {
            tracing::info!("CUDA available — using GPU device 0");
            candle_core::Device::new_cuda(0)?
        } else {
            tracing::info!("CUDA not available — using CPU");
            candle_core::Device::Cpu
        };
        let gpu_weights = GpuWeights::from_model_weights(&model_weights, &device)?;

        // ModelRunner takes ownership of a tokenizer, so load a second instance.
        let runner_tokenizer = if let Some(ref path) = tokenizer_path {
            KilnTokenizer::from_file(path)?
        } else {
            let tok_file = Path::new(mp).join("tokenizer.json");
            if tok_file.exists() {
                KilnTokenizer::from_file(tok_file.to_str().unwrap())?
            } else {
                KilnTokenizer::from_pretrained(&model_id)?
            }
        };

        let runner = ModelRunner::new(gpu_weights, runner_tokenizer, model_config.clone());

        let adapter_dir = std::env::var("KILN_ADAPTER_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(mp).join("adapters"));

        if !adapter_dir.exists() {
            tracing::info!(path = %adapter_dir.display(), "creating adapter directory");
            std::fs::create_dir_all(&adapter_dir)?;
        }

        tracing::info!(adapter_dir = %adapter_dir.display(), "model loaded — real inference mode");
        AppState::new_real(model_config, runner, tokenizer, device, adapter_dir)
    } else {
        // Mock mode: use scheduler + mock engine.
        tracing::info!("no KILN_MODEL_PATH set — running in mock mode");
        let scheduler_config = SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
        };
        let num_blocks = 8192;
        let scheduler = Scheduler::new(scheduler_config, num_blocks);
        let engine = MockEngine::new(model_config.clone());
        AppState::new_mock(model_config, scheduler, Arc::new(engine), tokenizer)
    };

    let app = api::router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("kiln listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use clap::Parser;

use kiln_server::api;
use kiln_server::cli::{self, AdapterCommands, Cli, Commands, TrainCommands};
use kiln_server::config::KilnConfig;
use kiln_server::device::select_device;
use kiln_server::state;

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::ModelRunner;
use kiln_model::engine::MockEngine;
use kiln_model::forward::GpuWeights;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use state::{AppState, ModelBackend};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        // Client-side commands (talk to a running server)
        Some(Commands::Health { ref url }) => {
            return cli::run_health(url).await;
        }
        Some(Commands::ConfigCheck { ref file }) => {
            return cli::run_config_check(file.as_deref().or(args.config.as_deref()));
        }
        Some(Commands::Train(ref train)) => match train {
            TrainCommands::Sft {
                file,
                adapter,
                lr,
                epochs,
                url,
            } => {
                return cli::run_train_sft(url, file, adapter, *lr, *epochs).await;
            }
            TrainCommands::Grpo { file, adapter, url } => {
                return cli::run_train_grpo(url, file, adapter).await;
            }
        },
        Some(Commands::Adapters(ref adapter_cmd)) => match adapter_cmd {
            AdapterCommands::List { url } => {
                return cli::run_adapters_list(url).await;
            }
            AdapterCommands::Load { name, url } => {
                return cli::run_adapters_load(url, name).await;
            }
            AdapterCommands::Unload { name, url } => {
                return cli::run_adapters_unload(url, name).await;
            }
            AdapterCommands::Delete { name, url } => {
                return cli::run_adapters_delete(url, name).await;
            }
        },
        // Serve mode (default)
        Some(Commands::Serve {
            ref served_model_id,
        }) => {
            // CLI flag wins over env/TOML; surface it via env var so the
            // config loader picks it up uniformly.
            if let Some(v) = served_model_id {
                // Safety: argv parsing happens before any threads are spawned.
                unsafe {
                    std::env::set_var("KILN_SERVED_MODEL_ID", v);
                }
            }
        }
        None => {}
    }

    // --- Server startup ---
    let config = KilnConfig::load(args.config.as_deref())?;

    kiln_server::logging::init(&config.logging.level, &config.logging.format)?;

    let host = &config.server.host;
    let port = config.server.port;

    let model_config = ModelConfig::qwen3_5_4b();
    let model_path = config.model.path.as_deref();
    let served_model_id = config.model.effective_served_model_id();
    tracing::info!(served_model_id = %served_model_id, "served model identifier");

    // Print startup banner to stderr (doesn't interfere with structured logs)
    cli::print_banner(host, port, model_path, args.config.as_deref());

    // Load tokenizer: try from_pretrained (HF Hub), fall back to local path, then fail gracefully.
    let model_id = &config.model.model_id;
    let tokenizer_path = config.model.tokenizer_path.as_deref();

    let tokenizer = if let Some(path) = tokenizer_path {
        tracing::info!("loading tokenizer from {path}");
        KilnTokenizer::from_file(path)?
    } else if let Some(mp) = model_path {
        // Try loading tokenizer from the model directory first
        let tok_file = Path::new(mp).join("tokenizer.json");
        if tok_file.exists() {
            tracing::info!(
                "loading tokenizer from model directory: {}",
                tok_file.display()
            );
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
            KilnTokenizer::from_pretrained(model_id)?
        }
    } else {
        tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
        KilnTokenizer::from_pretrained(model_id)?
    };

    tracing::info!(
        vocab_size = tokenizer.vocab_size(),
        "tokenizer loaded successfully"
    );

    let mut state = if let Some(mp) = model_path {
        // Real inference mode: load model weights and create ModelRunner.
        tracing::info!("loading model weights from {mp}");
        let load_mtp = matches!(
            config.speculative.effective_method(),
            kiln_server::config::SpecMethod::Mtp
        );
        let model_weights = kiln_model::load_model_with_options(
            Path::new(mp),
            &model_config,
            kiln_model::LoadModelOptions { load_mtp },
        )?;
        let device = select_device()?;
        let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)?;

        let runner = ModelRunner::new_with_options(
            gpu_weights,
            tokenizer.clone(),
            model_config.clone(),
            config.memory.cuda_graphs,
        );

        let adapter_dir = config
            .model
            .adapter_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(mp).join("adapters"));

        if !adapter_dir.exists() {
            tracing::info!(path = %adapter_dir.display(), "creating adapter directory");
            std::fs::create_dir_all(&adapter_dir)?;
        }

        tracing::info!(adapter_dir = %adapter_dir.display(), "model loaded — real inference mode");
        tracing::info!(
            "training endpoints available — in-process LoRA training (no sidecar needed)"
        );
        AppState::new_real(
            model_config,
            runner,
            tokenizer,
            device,
            adapter_dir,
            &config.memory,
            config.server.request_timeout_secs,
            served_model_id,
        )
    } else {
        // Mock mode: use scheduler + mock engine.
        tracing::info!("no model path set — running in mock mode");
        tracing::info!("training endpoints will return 503 in mock mode (no real weights)");
        let scheduler_config = SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: config.prefix_cache.enabled,
            prefix_cache_max_blocks: config.prefix_cache.max_blocks,
        };
        let num_blocks = 8192;
        let scheduler = Scheduler::new(scheduler_config, num_blocks);
        let engine = MockEngine::new(model_config.clone());
        AppState::new_mock(
            model_config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            config.server.request_timeout_secs,
            served_model_id,
        )
    };

    // Apply server-level checkpoint_interval from config
    state.checkpoint_interval = config.training.checkpoint_interval;

    // Spawn the background training queue worker
    let shutdown_flag = state.shutdown.clone();
    kiln_server::training_queue::spawn_training_worker(state.clone(), shutdown_flag.clone());

    let prewarm_state = state.clone();
    let app = api::router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(
        host = %host,
        port = port,
        model_path = model_path.unwrap_or("none (mock mode)"),
        "kiln listening"
    );
    spawn_backend_prewarm(prewarm_state);
    // Graceful shutdown: listen for SIGTERM/SIGINT, drain in-flight requests,
    // then force-exit after a timeout.
    let shutdown_timeout_secs = config.server.shutdown_timeout_secs;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_flag))
        .await?;

    tracing::info!(
        timeout_secs = shutdown_timeout_secs,
        "server stopped accepting connections — waiting for in-flight requests to drain"
    );

    // Give in-flight requests time to complete, then force exit
    tokio::time::sleep(std::time::Duration::from_secs(shutdown_timeout_secs)).await;
    tracing::warn!("shutdown timeout reached — exiting");

    Ok(())
}

fn spawn_backend_prewarm(state: AppState) {
    let ModelBackend::Real {
        runner,
        block_manager,
        paged_cache,
    } = state.backend.as_ref()
    else {
        return;
    };

    let is_metal = {
        let runner_guard = runner.read().unwrap();
        matches!(
            runner_guard.weights.embed_tokens.device(),
            candle_core::Device::Metal(_)
        )
    };
    if !is_metal {
        return;
    }

    let runner = runner.clone();
    let block_manager = block_manager.clone();
    let paged_cache = paged_cache.clone();
    let metrics = state.metrics.clone();

    tokio::spawn(async move {
        if metrics.request_duration_count.load(Ordering::Relaxed) > 0 {
            tracing::info!(
                "skipping background inference prewarm because the server has already handled live traffic"
            );
            return;
        }
        if metrics.active_requests.load(Ordering::Relaxed) > 0 {
            tracing::info!(
                "skipping background inference prewarm because a request is already active"
            );
            return;
        }

        tracing::info!("starting background inference prewarm");
        let prewarm = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            if metrics.request_duration_count.load(Ordering::Relaxed) > 0
                || metrics.active_requests.load(Ordering::Relaxed) > 0
            {
                return Ok(false);
            }
            let runner_guard = runner.read().unwrap();
            let mut bm_guard = block_manager.lock().unwrap();
            let mut pc_guard = paged_cache.lock().unwrap();
            let params = SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                // `max_tokens = 1` only runs prefill and samples the first
                // token. Use two tokens so Metal also compiles the decode path
                // before the first live request reaches it.
                max_tokens: 2,
                repetition_penalty: 1.0,
                stop: Vec::new(),
                seed: Some(42),
            };
            let prompt_tokens = [1_u32, 2, 3, 4, 5, 6, 7, 8];
            runner_guard.generate_from_tokens_paged(
                &prompt_tokens,
                &params,
                &mut bm_guard,
                &mut pc_guard,
            )?;
            Ok(true)
        })
        .await;

        match prewarm {
            Ok(Ok(true)) => tracing::info!("background inference prewarm complete"),
            Ok(Ok(false)) => {
                tracing::info!(
                    "skipping background inference prewarm because a request became active"
                );
            }
            Ok(Err(err)) => tracing::warn!(error = %err, "background inference prewarm failed"),
            Err(err) => tracing::warn!(error = %err, "background inference prewarm task failed"),
        }
    });
}

/// Wait for SIGTERM or SIGINT, then signal shutdown.
async fn shutdown_signal(shutdown_flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT — initiating graceful shutdown"),
        _ = terminate => tracing::info!("received SIGTERM — initiating graceful shutdown"),
    }

    // Signal the training worker to stop accepting new jobs
    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
}

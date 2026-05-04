use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use clap::Parser;

use kiln_server::api;
use kiln_server::cli::{self, AdapterCommands, Cli, Commands, TrainCommands};
use kiln_server::config::KilnConfig;
use kiln_server::device::select_device_with_options;
use kiln_server::state;

use kiln_core::block::BlockManager;
use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::ModelRunner;
use kiln_model::engine::MockEngine;
use kiln_model::forward::GpuWeights;
use kiln_model::paged_kv_cache::PagedKvCache;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use state::{AppState, ModelBackend};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        // Client-side commands (talk to a running server)
        Some(Commands::Health { ref url, json }) => {
            return cli::run_health(url, json).await;
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
                lora_rank,
                url,
            } => {
                return cli::run_train_sft(url, file, adapter, *lr, *epochs, *lora_rank).await;
            }
            TrainCommands::Grpo {
                file,
                adapter,
                lora_rank,
                url,
            } => {
                return cli::run_train_grpo(url, file, adapter, *lora_rank).await;
            }
            TrainCommands::Status { job_id, url } => {
                return cli::run_train_status(url, job_id.as_deref()).await;
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
                return cli::run_adapters_unload(url, name.as_deref()).await;
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

    let (tokenizer, chat_template_dir) = if let Some(path) = tokenizer_path {
        tracing::info!("loading tokenizer from {path}");
        let tok = KilnTokenizer::from_file(path)?;
        let dir = Path::new(path).parent().map(|p| p.to_path_buf());
        (tok, dir)
    } else if let Some(mp) = model_path {
        // Try loading tokenizer from the model directory first
        let tok_file = Path::new(mp).join("tokenizer.json");
        if tok_file.exists() {
            tracing::info!(
                "loading tokenizer from model directory: {}",
                tok_file.display()
            );
            (
                KilnTokenizer::from_file(tok_file.to_str().unwrap())?,
                Some(PathBuf::from(mp)),
            )
        } else {
            tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
            (KilnTokenizer::from_pretrained(model_id)?, None)
        }
    } else {
        tracing::info!("loading tokenizer from HuggingFace Hub: {model_id}");
        (KilnTokenizer::from_pretrained(model_id)?, None)
    };

    // Load the model's chat template (e.g. Qwen3.5's official template, which
    // appends `<think>\n` after `<|im_start|>assistant\n`). Without this,
    // `apply_chat_template` falls back to the bare ChatML stub and the model
    // is prompted out-of-distribution — Qwen3.5-4B answers "Hello!" with
    // "毎回毎回毎回..." instead of a real reply because the trained prompt
    // shape is missing the `<think>` prefix.
    let tokenizer = if let Some(dir) = chat_template_dir.as_deref() {
        match load_chat_template_from_model_dir(dir) {
            Ok(Some((source, template))) => {
                tracing::info!(
                    source = source,
                    bytes = template.len(),
                    "loaded chat template from model directory"
                );
                tokenizer.with_chat_template(template)
            }
            Ok(None) => {
                tracing::warn!(
                    dir = %dir.display(),
                    "no chat_template.jinja or tokenizer_config.json chat_template field found — \
                     falling back to bare ChatML, which produces broken output for Qwen3.5"
                );
                tokenizer
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    dir = %dir.display(),
                    "failed to load chat template — falling back to bare ChatML"
                );
                tokenizer
            }
        }
    } else {
        tokenizer
    };

    tracing::info!(
        vocab_size = tokenizer.vocab_size(),
        "tokenizer loaded successfully"
    );

    let mut state = if let Some(mp) = model_path {
        // Real inference mode: load model weights and create ModelRunner.
        tracing::info!("loading model weights from {mp}");
        let device = select_device_with_options(config.memory.cuda_graphs)?;
        let model_weights = kiln_model::load_model_with_options(
            Path::new(mp),
            &model_config,
            kiln_model::LoadModelOptions { load_mtp: false },
        )?;
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
        if matches!(
            std::env::var("KILN_BATCHING_ENGINE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        ) {
            tracing::info!("real non-streaming batching actor requested via KILN_BATCHING_ENGINE");
        }
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
            &config.prefix_cache,
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
    state.training_webhook_url = config.training.webhook_url.clone();
    state.max_queued_training_jobs = config.training.max_queued_jobs;
    state.max_tracked_jobs = config.training.max_tracked_jobs;
    state.tracked_job_ttl = std::time::Duration::from_secs(config.training.tracked_job_ttl_secs);
    state.adapter_max_disk_bytes = config.adapters.max_disk_bytes;
    state.composed_cache_max_bytes = config.adapters.composed_cache_max_bytes;
    state.composed_cache_max_entries = config.adapters.composed_cache_max_entries;
    if let Some(ref url) = state.training_webhook_url {
        tracing::info!(url = %url, "training completion webhook configured");
    }
    tracing::info!(
        cap = state.max_queued_training_jobs,
        "training queue cap configured"
    );
    tracing::info!(
        cap = state.max_tracked_jobs,
        ttl_secs = config.training.tracked_job_ttl_secs,
        "training tracked-jobs cap and TTL configured"
    );
    match state.adapter_max_disk_bytes {
        Some(cap) => tracing::info!(
            cap_bytes = cap,
            cap_gib = cap as f64 / 1024.0 / 1024.0 / 1024.0,
            "adapter_dir disk cap configured"
        ),
        None => tracing::info!("adapter_dir disk cap disabled (operator opt-out)"),
    }
    match (
        state.composed_cache_max_bytes,
        state.composed_cache_max_entries,
    ) {
        (None, None) => {
            tracing::info!("composed-adapter cache LRU eviction disabled (operator opt-out)")
        }
        (bytes, entries) => tracing::info!(
            cap_bytes = ?bytes,
            cap_gib = ?bytes.map(|b| b as f64 / 1024.0 / 1024.0 / 1024.0),
            cap_entries = ?entries,
            "composed-adapter cache LRU eviction configured"
        ),
    }

    // Spawn the background training queue worker
    let shutdown_flag = state.shutdown.clone();
    kiln_server::training_queue::spawn_training_worker(state.clone(), shutdown_flag.clone());

    let tokenizer_prewarm = state.tokenizer.clone();
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
    spawn_tokenizer_warmup(tokenizer_prewarm);
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

/// Locate the model's chat template, preferring the standalone
/// `chat_template.jinja` file (modern HF layout, e.g. Qwen3.5) and falling back
/// to the `chat_template` field in `tokenizer_config.json` (older layout). Returns
/// `Ok(None)` only when neither file is present, so the caller can warn rather
/// than silently use the bare ChatML stub.
fn load_chat_template_from_model_dir(dir: &Path) -> Result<Option<(&'static str, String)>> {
    let standalone = dir.join("chat_template.jinja");
    if standalone.exists() {
        let template = std::fs::read_to_string(&standalone)?;
        return Ok(Some(("chat_template.jinja", template)));
    }
    let config_path = dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&config_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(parsed
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| ("tokenizer_config.json", s.to_string())))
}

fn spawn_backend_prewarm(state: AppState) {
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return;
    };

    let (is_gpu, device) = {
        let runner_guard = runner.read().unwrap();
        let device = runner_guard.weights.embed_tokens.device().clone();
        let is_metal = matches!(device, candle_core::Device::Metal(_));
        #[cfg(feature = "vulkan")]
        let is_vulkan = kiln_model::backend::vulkan::vulkan_is_available();
        #[cfg(not(feature = "vulkan"))]
        let is_vulkan = false;
        (is_metal || is_vulkan, device)
    };
    if !is_gpu {
        return;
    }

    let runner = runner.clone();
    let gpu_lock = state.gpu_lock.clone();
    let prewarm_complete = state.inference_prewarm_complete.clone();

    tokio::spawn(async move {
        tracing::info!("starting background inference prewarm");
        let prewarm_start = std::time::Instant::now();
        let prewarm = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            // Pipeline compilation does not allocate KV/model working buffers, so
            // keep it outside the opportunistic GPU lock. If the first live
            // request wins the lock, it should still benefit from compiled
            // custom kernels rather than paying lazy compile latency itself.
            precompile_metal_custom_kernels(&device);

            // Prewarm is opportunistic. If a live request or training job has
            // the GPU first, skip prewarm rather than sitting in front of it.
            let Ok(_gpu_guard) = gpu_lock.try_write() else {
                tracing::info!("skipping inference prewarm because GPU is already busy");
                return Ok(());
            };

            precompile_metal_custom_kernels(&device);
            precompile_vulkan_custom_kernels(&device);
            let runner_guard = runner.read().unwrap();
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
            // Warm one full paged block plus a decode step. The previous
            // 8-token prompt missed first-block prompt shapes that desktop
            // traffic commonly hits, leaving Metal/Candle kernels to compile
            // on the first live request.
            let prompt_tokens = [1_u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let num_blocks = 2;
            // Warm the base paged path used by every desktop request. The
            // previous speculative-first prewarm made readiness wait on
            // skip-layer draft/verify work; live greedy requests can still
            // compile speculative kernels on demand without blocking startup.
            let prewarm_result = {
                let mut block_manager = BlockManager::new(num_blocks, 16);
                let mut paged_cache = PagedKvCache::new_uninit(
                    runner_guard.config.num_full_attention_layers,
                    num_blocks,
                    16,
                    runner_guard.config.num_kv_heads,
                    runner_guard.config.head_dim,
                    prewarm_kv_dtype(&runner_guard.config),
                    runner_guard.weights.embed_tokens.device(),
                )?;
                runner_guard.generate_from_tokens_paged(
                    &prompt_tokens,
                    &params,
                    &mut block_manager,
                    &mut paged_cache,
                    None,
                )
            };

            if let Err(err) = prewarm_result {
                anyhow::bail!("base paged inference prewarm failed: {err}");
            }
            Ok(())
        })
        .await;

        match prewarm {
            Ok(Ok(())) => tracing::info!(
                elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                "background inference prewarm complete"
            ),
            Ok(Err(err)) => tracing::warn!(error = %err, "background inference prewarm failed"),
            Err(err) => tracing::warn!(error = %err, "background inference prewarm task failed"),
        }
        prewarm_complete.store(true, Ordering::Release);
    });
}

fn spawn_tokenizer_warmup(tokenizer: Arc<KilnTokenizer>) {
    tokio::spawn(async move {
        let _ = tokio::task::spawn_blocking(move || warm_tokenizer(&tokenizer)).await;
    });
}

fn warm_tokenizer(tokenizer: &KilnTokenizer) {
    let start = std::time::Instant::now();
    let prompt = "Kiln tokenizer startup warmup.";
    match tokenizer.encode(prompt) {
        Ok(tokens) => {
            if let Err(err) = tokenizer.decode(&tokens) {
                tracing::warn!(error = %err, "tokenizer decode warmup failed");
            } else {
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    tokens = tokens.len(),
                    "tokenizer warmup complete"
                );
            }
        }
        Err(err) => tracing::warn!(error = %err, "tokenizer encode warmup failed"),
    }
}

#[cfg(feature = "metal")]
fn precompile_metal_custom_kernels(device: &candle_core::Device) {
    if !matches!(device, candle_core::Device::Metal(_)) {
        return;
    }

    let start = std::time::Instant::now();
    match kiln_model::backend::metal::precompile_custom_kernels(device) {
        Ok(()) => tracing::info!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Metal custom kernels precompiled during background prewarm"
        ),
        Err(err) => tracing::warn!(
            error = %err,
            "Metal custom kernel precompile failed; falling back to lazy compilation"
        ),
    }
}

#[cfg(not(feature = "metal"))]
fn precompile_metal_custom_kernels(_device: &candle_core::Device) {}

#[cfg(feature = "vulkan")]
fn precompile_vulkan_custom_kernels(_device: &candle_core::Device) {
    let start = std::time::Instant::now();
    match kiln_model::backend::vulkan::precompile_custom_kernels() {
        Ok(()) => tracing::info!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Vulkan custom kernels precompiled during background prewarm"
        ),
        Err(err) => tracing::warn!(
            error = %err,
            "Vulkan custom kernel precompile failed; falling back to lazy compilation"
        ),
    }
}

#[cfg(not(feature = "vulkan"))]
fn precompile_vulkan_custom_kernels(_device: &candle_core::Device) {}

fn prewarm_kv_dtype(config: &ModelConfig) -> candle_core::DType {
    match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    }
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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use clap::Parser;

use kiln_server::api;
use kiln_server::cli::{self, AdapterCommands, Cli, Commands, TrainCommands, TrajectoryCommands};
use kiln_server::config::KilnConfig;
use kiln_server::device::select_device_with_options;
use kiln_server::state;

use kiln_core::config::ModelConfig;
use kiln_core::env_flag::env_tristate;
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
            AdapterCommands::Verify {
                name_or_path,
                adapter_dir,
                url,
                prompt,
            } => {
                return cli::run_adapter_verify(
                    args.config.as_deref(),
                    url.as_deref(),
                    adapter_dir.as_deref(),
                    name_or_path,
                    prompt.as_deref(),
                )
                .await;
            }
        },
        Some(Commands::Trajectory(ref trajectory_cmd)) => match trajectory_cmd {
            TrajectoryCommands::Inspect {
                file,
                json,
                include_context,
                preview_tokens,
                tokenizer,
                chat_template,
                model_path,
            } => {
                return cli::run_trajectory_inspect(
                    args.config.as_deref(),
                    file.as_path(),
                    tokenizer.as_deref(),
                    chat_template.as_deref(),
                    model_path.as_deref(),
                    *json,
                    *include_context,
                    *preview_tokens,
                );
            }
        },
        // §10.14 — pi + kiln canonical pipeline subcommands.
        Some(Commands::PiSetup { ref url, ref out }) => {
            return cli::run_pi_setup(url, out.as_deref()).await;
        }
        Some(Commands::Judge(ref jc)) => {
            return cli::run_judge(jc).await;
        }
        Some(Commands::SelfImprove {
            ref url,
            ref agent,
            ref judge,
            no_crisp,
        }) => {
            return cli::run_self_improve(url, agent, judge, !no_crisp).await;
        }
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

    let level = args.effective_log_level(&config.logging.level);
    kiln_server::logging::init(level, &config.logging.format)?;

    let host = &config.server.host;
    let port = config.server.port;

    let model_config = ModelConfig::qwen3_5_4b();
    let model_path = config.model.path.as_deref();
    let served_model_id = config.model.effective_served_model_id();
    tracing::debug!(served_model_id = %served_model_id, "served model identifier");

    // Print startup banner to stderr (doesn't interfere with structured logs)
    cli::print_banner(host, port, model_path, args.config.as_deref());

    // Load tokenizer: try from_pretrained (HF Hub), fall back to local path, then fail gracefully.
    let model_id = &config.model.model_id;
    let tokenizer_path = config.model.tokenizer_path.as_deref();

    let (tokenizer, chat_template_dir) = if let Some(path) = tokenizer_path {
        tracing::debug!("loading tokenizer from {path}");
        let tok = KilnTokenizer::from_file(path)?;
        let dir = Path::new(path).parent().map(|p| p.to_path_buf());
        (tok, dir)
    } else if let Some(mp) = model_path {
        // Try loading tokenizer from the model directory first
        let tok_file = Path::new(mp).join("tokenizer.json");
        if tok_file.exists() {
            tracing::debug!(
                "loading tokenizer from model directory: {}",
                tok_file.display()
            );
            (
                KilnTokenizer::from_file(tok_file.to_str().unwrap())?,
                Some(PathBuf::from(mp)),
            )
        } else {
            tracing::debug!("loading tokenizer from HuggingFace Hub: {model_id}");
            (KilnTokenizer::from_pretrained(model_id)?, None)
        }
    } else {
        tracing::debug!("loading tokenizer from HuggingFace Hub: {model_id}");
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
                tracing::debug!(
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

    tracing::debug!(
        vocab_size = tokenizer.vocab_size(),
        "tokenizer loaded successfully"
    );

    let mut state = if let Some(mp) = model_path {
        // Real inference mode: load model weights and create ModelRunner.
        tracing::debug!("loading model weights from {mp}");
        let load_spinner = cli::make_startup_spinner("selecting device");
        let device = select_device_with_options(config.memory.cuda_graphs)?;
        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message(format!("loading model weights from {mp}"));
        }
        let model_weights = kiln_model::load_model_with_options(
            Path::new(mp),
            &model_config,
            kiln_model::LoadModelOptions { load_mtp: false },
        )?;
        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message("uploading weights to GPU");
        }
        let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)?;
        drop(model_weights);
        tracing::info!("CPU model weights dropped after GPU upload");

        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message("initializing inference runtime");
        }
        let runner = ModelRunner::new_with_options(
            gpu_weights,
            tokenizer.clone(),
            model_config.clone(),
            config.memory.cuda_graphs,
        );
        if let Some(pb) = load_spinner.as_ref() {
            pb.finish_and_clear();
        }

        let adapter_dir = config
            .model
            .adapter_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(mp).join("adapters"));

        if !adapter_dir.exists() {
            tracing::debug!(path = %adapter_dir.display(), "creating adapter directory");
            std::fs::create_dir_all(&adapter_dir)?;
        }

        tracing::debug!(adapter_dir = %adapter_dir.display(), "model loaded — real inference mode");
        tracing::debug!(
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
        tracing::debug!("no model path set — running in mock mode");
        tracing::debug!("training endpoints will return 503 in mock mode (no real weights)");
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
        tracing::debug!(url = %url, "training completion webhook configured");
    }
    tracing::debug!(
        cap = state.max_queued_training_jobs,
        "training queue cap configured"
    );
    tracing::debug!(
        cap = state.max_tracked_jobs,
        ttl_secs = config.training.tracked_job_ttl_secs,
        "training tracked-jobs cap and TTL configured"
    );

    // Restore terminal training jobs persisted from previous runs so the
    // /ui training queue still shows last week's history after a restart.
    {
        use kiln_server::training_history;
        let archived = training_history::load_all(&state.adapter_dir);
        if !archived.is_empty() {
            let mut jobs = state.training_jobs.write().unwrap();
            for job in archived.iter() {
                jobs.entry(job.job_id.clone()).or_insert_with(|| job.clone());
            }
            tracing::info!(
                count = archived.len(),
                "restored archived training jobs from disk"
            );
        }
    }

    // Restore terminal eval jobs persisted from previous runs — same
    // pattern as training, so the Evals tab survives a restart too.
    {
        use kiln_server::eval_history;
        let archived = eval_history::load_all(&state.adapter_dir);
        if !archived.is_empty() {
            let mut jobs = state.eval_jobs.write().unwrap();
            for job in archived.iter() {
                jobs.entry(job.job_id.clone()).or_insert_with(|| job.clone());
            }
            tracing::info!(
                count = archived.len(),
                "restored archived eval jobs from disk"
            );
        }
    }
    match state.adapter_max_disk_bytes {
        Some(cap) => tracing::debug!(
            cap_bytes = cap,
            cap_gib = cap as f64 / 1024.0 / 1024.0 / 1024.0,
            "adapter_dir disk cap configured"
        ),
        None => tracing::debug!("adapter_dir disk cap disabled (operator opt-out)"),
    }
    match (
        state.composed_cache_max_bytes,
        state.composed_cache_max_entries,
    ) {
        (None, None) => {
            tracing::debug!("composed-adapter cache LRU eviction disabled (operator opt-out)")
        }
        (bytes, entries) => tracing::debug!(
            cap_bytes = ?bytes,
            cap_gib = ?bytes.map(|b| b as f64 / 1024.0 / 1024.0 / 1024.0),
            cap_entries = ?entries,
            "composed-adapter cache LRU eviction configured"
        ),
    }

    // Wire on-disk eval suite + dataset + judgment registries. The shared
    // root is `<adapter_dir>/.eval/` by default; subdirs split the three
    // collections so users can audit one without listing everything.
    let eval_root = config
        .eval
        .as_ref()
        .and_then(|e| e.eval_dir.clone())
        .unwrap_or_else(|| state.adapter_dir.join(".eval"));
    let suite_dir = eval_root.join("suites");
    let dataset_dir = eval_root.join("datasets");
    let judgment_dir = eval_root.join("judgments");
    for (path, label) in &[
        (&suite_dir, "suites"),
        (&dataset_dir, "datasets"),
        (&judgment_dir, "judgments"),
    ] {
        if let Err(e) = std::fs::create_dir_all(path) {
            tracing::warn!(error = %e, path = %path.display(), kind = label, "failed to create eval subdir; that registry will be disabled");
        }
    }
    if suite_dir.exists() {
        let registry = kiln_server::eval::SuiteRegistry::new(suite_dir.clone());
        // Install the built-in Qwen3.5 agentic-core suite so users can
        // `kiln-eval run --suite qwen3.5-agentic-core` without authoring
        // anything. Idempotent — the registry's `save(force=true)` is a
        // no-op when content is unchanged at the file-system level.
        if let Err(err) = registry.install_qwen3_agentic_core() {
            tracing::warn!(
                error = %err,
                "failed to install built-in qwen3.5-agentic-core eval suite (continuing)"
            );
        }
        state.suite_registry = Some(Arc::new(registry));
    }
    if dataset_dir.exists() {
        state.dataset_registry = Some(Arc::new(kiln_server::eval::DatasetRegistry::new(
            dataset_dir.clone(),
        )));
    }
    if judgment_dir.exists() {
        state.judgment_store = Some(Arc::new(kiln_server::eval::JudgmentStore::new(
            judgment_dir.clone(),
        )));
    }
    tracing::debug!(
        eval_root = %eval_root.display(),
        "eval registries online (suites + datasets + judgments)"
    );
    if let Some(cfg) = config.eval.as_ref() {
        state.max_queued_eval_jobs = cfg.max_queued_jobs;
        state.max_tracked_eval_jobs = cfg.max_tracked_jobs;
    }

    // Spawn the background training queue worker
    let shutdown_flag = state.shutdown.clone();
    kiln_server::training_queue::spawn_training_worker(state.clone(), shutdown_flag.clone());
    // Spawn the background eval queue worker
    kiln_server::eval::spawn_eval_worker(state.clone(), shutdown_flag.clone());

    let tokenizer_prewarm = state.tokenizer.clone();
    let prewarm_state = state.clone();
    // Cheap clones so the shutdown handler can reach the batching engine
    // after `api::router` consumes the state.
    let app_state_for_shutdown = state.clone();
    let app = api::router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::debug!(
        host = %host,
        port = port,
        model_path = model_path.unwrap_or("none (mock mode)"),
        "kiln listening"
    );
    cli::print_ready_line(host, port);
    spawn_tokenizer_warmup(tokenizer_prewarm);
    spawn_backend_prewarm(prewarm_state);
    // Graceful shutdown: listen for SIGTERM/SIGINT, cancel in-flight
    // inference via the batching engine (so SSE streams terminate
    // immediately instead of holding the connection until the model
    // naturally finishes generating), then bound the drain wait with a
    // hard timeout so Ctrl+C is always responsive.
    let shutdown_timeout_secs = config.server.shutdown_timeout_secs;

    // Snapshot the batching engine handle so the signal handler can
    // proactively stop it. The handle is cheap to clone (just an mpsc
    // sender + a snapshot atomic).
    let engine_for_shutdown = match app_state_for_shutdown.backend.as_ref() {
        ModelBackend::Real {
            batching_engine, ..
        } => batching_engine.clone(),
        ModelBackend::Mock { .. } => None,
    };

    // Serve until the shutdown signal triggers + axum drains. The drain
    // is bounded by a watchdog set up *inside* `shutdown_signal` once
    // the signal actually fires — see comments there. We deliberately
    // don't wrap `axum::serve` itself in a timeout, because that would
    // cap *total server uptime*, not just the drain. (Lesson learned.)
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(
            shutdown_flag,
            engine_for_shutdown,
            shutdown_timeout_secs,
        ))
        .await?;

    tracing::info!("server stopped cleanly");
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
    let ModelBackend::Real {
        runner,
        block_manager,
        paged_cache,
        ..
    } = state.backend.as_ref()
    else {
        return;
    };

    let (is_gpu, is_vulkan, device) = {
        let runner_guard = runner.read().unwrap();
        let device = runner_guard.weights.embed_tokens.device().clone();
        let is_metal = matches!(device, candle_core::Device::Metal(_));
        let is_vulkan = runner_guard.backend_name() == "vulkan";
        (is_metal || is_vulkan, is_vulkan, device)
    };
    if !is_gpu {
        return;
    }

    let runner = runner.clone();
    let block_manager = block_manager.clone();
    let paged_cache = paged_cache.clone();
    let gpu_lock = state.gpu_lock.clone();
    let prewarm_complete = state.inference_prewarm_complete.clone();

    if vk_native_training_enabled(is_vulkan) {
        tracing::info!(
            "skipping background inference prewarm because Vulkan-native training is enabled"
        );
        prewarm_complete.store(true, Ordering::Release);
        return;
    }

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
            // Write lock — `prewarm_backend_decode_weights` now mutates
            // `weights` to stub the pre-transposed bf16 caches after Vulkan
            // upload (frees ~6-7 GB of candle CPU residency). Prewarm runs
            // once at startup so the brief exclusive lock is fine.
            let mut runner_guard = runner.write().unwrap();
            runner_guard
                .prewarm_backend_decode_weights()
                .context("backend decode weight prewarm failed")?;
            drop(runner_guard);
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
                ..SamplingParams::default()
            };
            // Warm several paged blocks plus a decode step. Short one-block
            // prewarm misses the multi-block prompt shapes that desktop chat
            // and batch traffic commonly hits, leaving Metal/Candle kernels to
            // compile on the first live request.
            let prompt_tokens: Vec<u32> = (1..=64).collect();
            // Warm the base paged path used by every desktop request. The
            // previous speculative-first prewarm made readiness wait on
            // skip-layer draft/verify work; live greedy requests can still
            // compile speculative kernels on demand without blocking startup.
            let prewarm_result = runner_guard.generate_paged_shared_tokens(
                &prompt_tokens,
                &params,
                &block_manager,
                &paged_cache,
                None,
            );

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

fn vk_native_training_enabled(is_vulkan: bool) -> bool {
    env_tristate("KILN_VK_NATIVE_TRAINING").unwrap_or_else(|| {
        #[cfg(feature = "vulkan")]
        {
            is_vulkan
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = is_vulkan;
            false
        }
    })
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

/// Wait for SIGTERM or SIGINT, then signal shutdown. Receiving a *second*
/// signal while still draining short-circuits straight to process exit so
/// users hammering Ctrl+C never have to wait.
///
/// `drain_timeout_secs` is the hard ceiling on how long we'll wait for
/// in-flight requests to finish *after* the signal fires. Once a signal
/// triggers, a detached watchdog will force-exit the process at the
/// timeout even if axum's drain hasn't returned yet.
async fn shutdown_signal(
    shutdown_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    engine: Option<kiln_server::batching_engine::BatchingEngineHandle>,
    drain_timeout_secs: u64,
) {
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

    // Tell training/eval workers + every shutdown-aware code path to stop
    // accepting work.
    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);

    // Proactively cancel every in-flight inference request. Without this
    // step, axum's graceful_shutdown waits for the model to naturally
    // finish generating on every open SSE stream — which can take a
    // minute on long completions. `BatchingEngineHandle::stop` triggers
    // `fail_all`, which calls `cancel.cancel()` + sends EngineEvent::Error
    // to every waiting/active request so connection handlers return
    // immediately and axum's drain completes promptly.
    if let Some(engine) = engine {
        match engine.stop().await {
            Ok(()) => tracing::debug!("batching engine stopped — in-flight requests cancelled"),
            Err(e) => tracing::warn!(error = %e, "batching engine stop failed (continuing)"),
        }
    }

    // Watchdog: if axum's graceful drain hasn't returned by the
    // configured timeout, force-exit. Spawned detached so we return
    // immediately and the drain can proceed in parallel.
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(drain_timeout_secs)).await;
        tracing::warn!(
            timeout_secs = drain_timeout_secs,
            "graceful-shutdown drain hit timeout — forcing exit"
        );
        std::process::exit(0);
    });

    // Watch for a second signal — if the user hammers Ctrl+C, exit
    // immediately instead of waiting for the drain. Spawning a detached
    // task here means we don't block the primary shutdown future.
    tokio::spawn(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::warn!("second SIGINT received — exiting immediately");
        std::process::exit(130);
    });
}

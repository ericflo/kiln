use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use axum::serve::ListenerExt as _;
use clap::Parser;
use socket2::{SockRef, Socket};

use kiln_server::api;
use kiln_server::cli::{
    self, AdapterCommands, Cli, Commands, HfTrainCommands, TrainCommands, TrajectoryCommands,
};
use kiln_server::config::KilnConfig;
use kiln_server::device::select_device_with_options_kt;
use kiln_server::state;

use kiln_core::config::ModelConfig;
use kiln_core::config_hashes::{ConfigHashes, kiln_env_config_hash};
use kiln_core::env_flag::env_tristate;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_model::forward::GpuWeights;
use kiln_model::{
    BackendCapabilityQueries, ModelRunner, ModelRunnerRuntimeOptions, StartupCapabilities,
};
use kiln_scheduler::{Scheduler, SchedulerConfig};
use state::{AppState, GpuCoordinationLock, ModelBackend, SpeculativeRuntimePolicy};

fn resolve_model_runner_runtime_options(
    policy: kiln_server::config::ServingRuntimePolicy,
    cuda_graphs_requested: bool,
    max_decode_batch: Option<usize>,
) -> ModelRunnerRuntimeOptions {
    if policy.live_graph_capture {
        ModelRunnerRuntimeOptions {
            cuda_graphs: cuda_graphs_requested,
            rocm_graphs: true,
            metal_graphs: true,
            max_decode_batch,
        }
    } else {
        ModelRunnerRuntimeOptions {
            max_decode_batch,
            ..ModelRunnerRuntimeOptions::eager_only()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HttpSendBufferApplication {
    /// Raw `getsockopt(SO_SNDBUF)` value reported by the operating system.
    actual_bytes: usize,
    /// Usable buffer request after normalizing platform accounting.
    effective_bytes: usize,
}

fn effective_http_send_buffer_bytes(actual_bytes: usize) -> usize {
    // Linux reports twice the requested SO_SNDBUF to include kernel
    // bookkeeping. Compare against the normalized value so a capped request
    // cannot look successful merely because the read-back is doubled.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        actual_bytes / 2
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        actual_bytes
    }
}

fn validate_http_send_buffer_readback(
    requested_bytes: usize,
    actual_bytes: usize,
) -> std::io::Result<HttpSendBufferApplication> {
    let effective_bytes = effective_http_send_buffer_bytes(actual_bytes);
    if effective_bytes < requested_bytes {
        return Err(std::io::Error::other(format!(
            "SO_SNDBUF read-back {actual_bytes} represents {effective_bytes} effective bytes, \
             below requested {requested_bytes}"
        )));
    }
    Ok(HttpSendBufferApplication {
        actual_bytes,
        effective_bytes,
    })
}

fn configure_http_send_buffer(
    socket: &Socket,
    requested_bytes: usize,
) -> std::io::Result<HttpSendBufferApplication> {
    socket.set_send_buffer_size(requested_bytes)?;
    let actual_bytes = socket.send_buffer_size()?;
    validate_http_send_buffer_readback(requested_bytes, actual_bytes)
}

fn configure_http_listener_send_buffer(
    listener: &tokio::net::TcpListener,
    requested_bytes: usize,
) -> std::io::Result<HttpSendBufferApplication> {
    configure_http_send_buffer(&SockRef::from(listener), requested_bytes)
}

fn configure_http_stream_send_buffer(
    stream: &tokio::net::TcpStream,
    requested_bytes: usize,
) -> std::io::Result<HttpSendBufferApplication> {
    configure_http_send_buffer(&SockRef::from(stream), requested_bytes)
}

fn inspect_http_send_buffer(socket: &Socket) -> (Option<usize>, Option<usize>) {
    let actual_bytes = socket.send_buffer_size().ok();
    let effective_bytes = actual_bytes.map(effective_http_send_buffer_bytes);
    (actual_bytes, effective_bytes)
}

fn preflight_http_send_buffer(
    listener: &tokio::net::TcpListener,
    requested_bytes: Option<usize>,
) -> Result<Option<HttpSendBufferApplication>> {
    let Some(requested_bytes) = requested_bytes else {
        return Ok(None);
    };

    match configure_http_listener_send_buffer(listener, requested_bytes) {
        Ok(application) => {
            tracing::info!(
                requested_bytes,
                actual_bytes = application.actual_bytes,
                effective_bytes = application.effective_bytes,
                "http_listener_send_buffer_preflight_succeeded"
            );
            Ok(Some(application))
        }
        Err(error) => {
            let socket = SockRef::from(listener);
            let (actual_bytes, effective_bytes) = inspect_http_send_buffer(&socket);
            tracing::error!(
                requested_bytes,
                actual_bytes = actual_bytes.unwrap_or_default(),
                actual_bytes_known = actual_bytes.is_some(),
                effective_bytes = effective_bytes.unwrap_or_default(),
                effective_bytes_known = effective_bytes.is_some(),
                error = %error,
                "http_listener_send_buffer_preflight_failed"
            );
            let actual_display = actual_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let effective_display = effective_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            Err(anyhow::Error::new(error).context(format!(
                "HTTP SO_SNDBUF listener preflight failed: requested_bytes={requested_bytes}, \
                 actual_bytes={actual_display}, effective_bytes={effective_display}"
            )))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let serve_cli_overrides = match &args.command {
        Some(Commands::Serve {
            served_model_id,
            eval_mode,
        }) => (served_model_id.clone(), *eval_mode),
        _ => (None, false),
    };
    let diagnostics_config_path = match &args.command {
        Some(Commands::ConfigCheck { file }) => file.as_deref().or(args.config.as_deref()),
        _ => args.config.as_deref(),
    };
    let bootstrap_logging = kiln_server::logging::bootstrap_config(diagnostics_config_path);
    let bootstrap_level = args
        .effective_log_level(&bootstrap_logging.level)
        .to_string();
    kiln_server::logging::init(&bootstrap_level, &bootstrap_logging.format)?;
    tracing::debug!(
        config_path = bootstrap_logging.config_path.as_deref().unwrap_or("<defaults>"),
        level = %bootstrap_level,
        format = %bootstrap_logging.format,
        "startup_diagnostics_initialized"
    );

    match args.command {
        // Client-side commands (talk to a running server)
        Some(Commands::Health { ref url, json }) => {
            return cli::run_health(url, json).await;
        }
        Some(Commands::EvalAdapter {
            ref url,
            ref adapter,
            ref tasks,
            seeds,
            ref request_template,
            ref scorer,
            ref output,
        }) => {
            return cli::run_eval_adapter(
                url,
                adapter,
                tasks,
                seeds,
                request_template,
                scorer,
                output,
            )
            .await;
        }
        Some(Commands::RolloutGenerate {
            ref url,
            ref adapter,
            thinking,
            thinking_budget_tokens,
            thinking_budget_ms,
            ref tasks,
            seeds,
            seed_start,
            ref request_template,
            ref scorer,
            ref output,
            ref summary_output,
        }) => {
            return cli::run_rollout_generate(
                url,
                adapter,
                thinking,
                thinking_budget_tokens,
                thinking_budget_ms,
                tasks,
                seeds,
                seed_start,
                request_template,
                scorer,
                output,
                summary_output,
            )
            .await;
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
                invalid_row_policy,
                adapter_smoke_test,
                checkpoint_interval,
                resume_checkpoint,
                url,
            } => {
                return cli::run_train_sft(
                    url,
                    file,
                    adapter,
                    *lr,
                    *epochs,
                    *lora_rank,
                    invalid_row_policy,
                    *adapter_smoke_test,
                    checkpoint_interval.map(std::num::NonZeroUsize::get),
                    resume_checkpoint.as_deref(),
                )
                .await;
            }
            TrainCommands::Grpo {
                file,
                adapter,
                lora_rank,
                adapter_smoke_test,
                checkpoint_interval,
                resume_checkpoint,
                url,
            } => {
                return cli::run_train_grpo(
                    url,
                    file,
                    adapter,
                    *lora_rank,
                    *adapter_smoke_test,
                    checkpoint_interval.map(std::num::NonZeroUsize::get),
                    resume_checkpoint.as_deref(),
                )
                .await;
            }
            TrainCommands::Opd {
                file,
                adapter,
                teacher,
                lora_rank,
                checkpoint_interval,
                resume_checkpoint,
                url,
            } => {
                return cli::run_train_opd(
                    url,
                    file,
                    adapter,
                    teacher.as_deref(),
                    *lora_rank,
                    checkpoint_interval.map(std::num::NonZeroUsize::get),
                    resume_checkpoint.as_deref(),
                )
                .await;
            }
            TrainCommands::Hf(command) => match command {
                HfTrainCommands::ExportSft {
                    file,
                    dataset,
                    name,
                    output,
                    invalid_row_policy,
                    input_adapter,
                    split_manifest,
                    keep_server_copy,
                    url,
                } => {
                    return kiln_server::hf_train_cli::run_export_sft(
                        kiln_server::hf_train_cli::ExportSftOptions {
                            url: url.clone(),
                            file: file.clone(),
                            dataset: dataset.clone(),
                            name: name.clone(),
                            output: output.clone(),
                            invalid_row_policy: invalid_row_policy.clone(),
                            input_adapter: input_adapter.clone(),
                            split_manifest: split_manifest.clone(),
                            keep_server_copy: *keep_server_copy,
                        },
                    )
                    .await;
                }
                HfTrainCommands::ExportGrpo {
                    file,
                    name,
                    output,
                    input_adapter,
                    split_manifest,
                    keep_server_copy,
                    url,
                } => {
                    return kiln_server::hf_train_cli::run_export_grpo(
                        kiln_server::hf_train_cli::ExportGrpoOptions {
                            url: url.clone(),
                            file: file.clone(),
                            name: name.clone(),
                            output: output.clone(),
                            input_adapter: input_adapter.clone(),
                            split_manifest: split_manifest.clone(),
                            keep_server_copy: *keep_server_copy,
                        },
                    )
                    .await;
                }
                HfTrainCommands::ImportPeft { bundle, name, url } => {
                    return kiln_server::hf_train_cli::run_import_peft(
                        kiln_server::hf_train_cli::ImportPeftOptions {
                            url: url.clone(),
                            bundle: bundle.clone(),
                            name: name.clone(),
                        },
                    )
                    .await;
                }
                HfTrainCommands::List { json, url } => {
                    return kiln_server::hf_train_cli::run_list(url, *json).await;
                }
                HfTrainCommands::Delete {
                    name,
                    export_sha256,
                    url,
                } => {
                    return kiln_server::hf_train_cli::run_delete(
                        url,
                        name,
                        export_sha256.as_deref(),
                    )
                    .await;
                }
            },
            TrainCommands::Status { job_id, url } => {
                return cli::run_train_status(url, job_id.as_deref()).await;
            }
            TrainCommands::Cancel { job_id, url } => {
                return cli::run_train_cancel(url, job_id).await;
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
            AdapterCommands::Restore {
                manifest,
                adapter_dir,
                name,
                overwrite,
            } => {
                return cli::run_adapter_restore(
                    args.config.as_deref(),
                    manifest,
                    adapter_dir.as_deref(),
                    name.as_deref(),
                    *overwrite,
                );
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
        Some(Commands::Serve { .. }) => {}
        None => {}
    }

    // --- Server startup ---
    let mut config = KilnConfig::load(args.config.as_deref())?;
    config.apply_serve_cli_overrides(serve_cli_overrides.0.as_deref(), serve_cli_overrides.1)?;
    let gradient_checkpoint_policy = kiln_train::GradientCheckpointPolicy::from_parts(
        config.training.grad_checkpoint_segments,
        config.training.no_grad_checkpoint,
    )
    .context("failed to resolve typed gradient-checkpoint policy")?;
    kiln_tensor::DETERMINISTIC_CACHED
        .configure(config.server.deterministic.enabled())
        .context("failed to fix deterministic tensor behavior from startup configuration")?;
    let serving_policy = config.server.serving_profile.runtime_policy();

    let validated_level = args.effective_log_level(&config.logging.level);
    if bootstrap_level != validated_level || bootstrap_logging.format != config.logging.format {
        tracing::warn!(
            bootstrap_level = %bootstrap_level,
            validated_level,
            bootstrap_format = %bootstrap_logging.format,
            validated_format = %config.logging.format,
            "logging_configuration_changed_during_startup"
        );
    }

    let serving_profile = config.server.serving_profile.diagnostics();
    let effective_policy = serving_profile.effective_policy;
    tracing::info!(
        profile = %serving_profile.profile,
        source = %serving_profile.source,
        immutable_after_startup = serving_profile.immutable_after_startup,
        request_overrides_allowed = serving_profile.request_overrides_allowed,
        effective_policy_source = serving_profile.effective_policy_source,
        inference_admission = effective_policy.inference_admission,
        training_gpu_ownership = effective_policy.training_gpu_ownership,
        adapter_weight_transitions = effective_policy.adapter_weight_transitions,
        dynamic_kv_resize = effective_policy.dynamic_kv_resize,
        allocator_reclaim = effective_policy.allocator_reclaim,
        live_graph_capture = effective_policy.live_graph_capture,
        exclusive_gpu_behavior = effective_policy.exclusive_gpu_behavior,
        "serving profile resolved"
    );
    if !effective_policy.inference_admission {
        tracing::warn!(
            profile = %serving_profile.profile,
            "inference admission is disabled; health remains non-ready until restart with a serving profile"
        );
    }

    let host = &config.server.host;
    kiln_server::api::terminal::set_bind_host(host);
    let port = config.server.port;
    // Embedded agent runs configure the spawned pi against this URL.
    kiln_server::agent_runs::set_self_url(host, port);

    let model_config = ModelConfig::qwen3_5_4b();
    config
        .speculative
        .validate_for_model(&model_config)
        .context("invalid speculative decoding configuration for the selected model")?;
    config.speculative.validate_for_serving()?;
    let model_path = config.model.path.as_deref();
    let served_model_id = config.model.effective_served_model_id();
    let model_defaults_profile = config.model.defaults_profile();
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

    let response_delivery_policy = kiln_server::batching_engine::ResponseDeliveryPolicy::from(
        config.server.stream_stall_grace_ms,
    );
    let mut model_snapshot_cleanup = None;
    let mut state = if let Some(mp) = model_path {
        // Real inference mode: load model weights and create ModelRunner.
        tracing::debug!("loading model weights from {mp}");
        let load_spinner = cli::make_startup_spinner("selecting device");
        let mut graph_options =
            resolve_model_runner_runtime_options(serving_policy, config.memory.cuda_graphs, None);
        let device_kt = select_device_with_options_kt(graph_options.cuda_graphs)?;
        let vram_probe_selector =
            kiln_server::state::ensure_accelerator_memory_probe_identity(device_kt)?;
        let physical_vram = kiln_memory::vram::detect_vram_for(vram_probe_selector);
        let capacity_resolution =
            kiln_memory::vram::resolve_vram_capacity(physical_vram, config.memory.gpu_memory_gb);
        kiln_server::state::ensure_accelerator_memory_capacity(
            device_kt,
            vram_probe_selector,
            capacity_resolution.effective,
        )?;
        kiln_server::state::ensure_accelerator_memory_floor(
            device_kt,
            capacity_resolution.effective,
            &config.memory,
        )?;
        kiln_memory::MemoryGovernor::configure_global(
            vram_probe_selector,
            config
                .memory
                .governor_config_for_capacity(capacity_resolution.effective.total_bytes),
        )
        .context("failed to configure the process-wide memory governor")?;
        let memory_governor = kiln_memory::MemoryGovernor::global();
        if vram_probe_selector != kiln_memory::vram::VramProbeSelector::None {
            let startup_memory = memory_governor.refresh();
            anyhow::ensure!(
                startup_memory.total_bytes > 0 && !startup_memory.observations.probe_failed,
                "selected-device memory probe failed before model loading"
            );
            anyhow::ensure!(
                memory_governor.start_sampler(),
                "failed to start the selected-device memory sampler before model loading"
            );
        }
        let backend_capabilities =
            kiln_model::backend::for_device_kt(&device_kt).backend_capabilities();
        let decode_batcher_policy = backend_capabilities.decode_batcher;
        let speculative_mtp_support = backend_capabilities.decode.mtp_speculative_generation;
        let startup_decode_runtime = kiln_server::batching_engine::resolve_decode_runtime_config(
            config.server.deterministic,
            config.server.max_decode_batch,
            Some(decode_batcher_policy),
            config.server.max_batch_tokens,
        );
        graph_options.max_decode_batch = Some(startup_decode_runtime.max_decode_batch.effective);
        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message(format!("loading model weights from {mp}"));
        }
        let model_weights = kiln_model::load_model_with_options_and_snapshot_dir(
            Path::new(mp),
            &model_config,
            kiln_model::LoadModelOptions { load_mtp: false },
            config.model.snapshot_dir.as_deref().map(Path::new),
        )?;
        let base_model_source_sha256 = model_weights
            .source_content_sha256
            .clone()
            .context("loaded model is missing its loader-owned source content revision")?;
        let base_weight_shard_manifest = model_weights
            .base_weight_shard_manifest()
            .cloned()
            .context("loaded model is missing its loader-owned base-weight shard manifest")?;
        anyhow::ensure!(
            base_weight_shard_manifest.aggregate_sha256 == base_model_source_sha256,
            "base-weight shard manifest aggregate does not match the loader-owned source content revision"
        );
        model_snapshot_cleanup = model_weights.snapshot_cleanup_handle();
        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message("uploading weights to GPU");
        }
        let gpu_weights =
            GpuWeights::from_model_weights_kt(&model_weights, &model_config, &device_kt)?;
        anyhow::ensure!(
            gpu_weights.base_weight_shard_manifest.as_ref() == Some(&base_weight_shard_manifest),
            "GPU weights did not retain the verified base-weight shard manifest"
        );
        model_weights
            .verify_source_content_unchanged()
            .context("model source changed between load and completed GPU upload")?;
        drop(model_weights);
        tracing::info!(
            base_model_source_sha256,
            base_weight_shard_count = base_weight_shard_manifest.shards.len(),
            base_weight_total_size_bytes = base_weight_shard_manifest.total_size_bytes,
            "CPU model weights dropped after verified GPU upload"
        );

        if let Some(pb) = load_spinner.as_ref() {
            pb.set_message("initializing inference runtime");
        }
        // The server owns the equivalent typed startup sequence above
        // (identity, resolved cap, and process governor) and AppState starts
        // its sampler. Embedded callers without that owner must use
        // `InferenceMemoryRuntime` plus `new_with_initialized_runtime`.
        let mut runner = ModelRunner::new_with_runtime_options(
            gpu_weights,
            tokenizer.clone(),
            model_config.clone(),
            graph_options,
        );
        let executable_sha256 = kiln_server::teacher_identity::current_executable_sha256()
            .context("failed to fingerprint the running server executable")?;
        let numerical_runtime_sha256 =
            kiln_server::teacher_identity::numerical_runtime_sha256(device_kt);
        let execution_provenance = kiln_server::execution_provenance::build_execution_provenance(
            &config,
            &model_config,
            &tokenizer,
            runner.backend_name(),
            device_kt,
            &executable_sha256,
            &numerical_runtime_sha256,
            runner.training_precision_policy(),
        )
        .context("failed to construct immutable execution provenance")?;
        execution_provenance
            .validate()
            .context("failed to validate immutable execution provenance")?;
        tracing::info!(
            provenance_sha256 = %execution_provenance.provenance_sha256,
            backend = %execution_provenance.backend.name,
            device = %execution_provenance.backend.device,
            executable_sha256 = %execution_provenance.build.executable_sha256,
            runtime_sha256 = %execution_provenance.backend.numerical_runtime_sha256,
            kernel_contract_sha256 = %execution_provenance.kernels.contract_sha256,
            "execution provenance initialized"
        );
        runner.weights.execution_provenance = Some(execution_provenance);
        let base_teacher_identity = Arc::new(
            kiln_server::teacher_identity::build_base_teacher_identity(
                &served_model_id,
                &base_model_source_sha256,
                &tokenizer,
                &model_config,
                runner.backend_name(),
                &executable_sha256,
                &numerical_runtime_sha256,
            )
            .context("failed to construct immutable base teacher identity")?,
        );
        tracing::info!(
            revision = %base_teacher_identity.content_revision(),
            base_model_sha256 = %base_teacher_identity.base_model_sha256(),
            tokenizer_vocab_sha256 = %base_teacher_identity.tokenizer_vocab_sha256(),
            implementation = %base_teacher_identity.implementation(),
            numerical_runtime_sha256,
            "base prompt-logprob teacher identity initialized"
        );
        if let Some(pb) = load_spinner.as_ref() {
            pb.finish_and_clear();
        }

        let adapter_dir =
            model_defaults_profile.resolve_adapter_dir(config.model.adapter_dir.as_deref(), mp);

        if !adapter_dir.exists() {
            tracing::debug!(path = %adapter_dir.display(), "creating adapter directory");
            std::fs::create_dir_all(&adapter_dir)?;
        }

        tracing::debug!(adapter_dir = %adapter_dir.display(), "model loaded — real inference mode");
        tracing::debug!(
            "training endpoints available — in-process LoRA training (no sidecar needed)"
        );
        let speculative_runtime_policy = SpeculativeRuntimePolicy::new(speculative_mtp_support);
        AppState::new_real_with_serving_profile(
            model_config,
            runner,
            tokenizer,
            device_kt,
            adapter_dir,
            &config.memory,
            response_delivery_policy,
            startup_decode_runtime,
            config.batching,
            config.speculative,
            speculative_runtime_policy,
            config.server.max_batch_tokens,
            config.server.max_prefill_tokens_per_cycle,
            config.server.max_prefill_layers_per_cycle,
            config.server.request_timeout_secs,
            served_model_id,
            &config.prefix_cache,
            Some(base_teacher_identity),
            config.server.serving_profile,
            gradient_checkpoint_policy,
        )
        .context("failed to initialize real server state")?
    } else {
        // Mock mode: use scheduler + mock engine.
        tracing::debug!("no model path set — running in mock mode");
        tracing::debug!("training endpoints will return 503 in mock mode (no real weights)");
        kiln_memory::MemoryGovernor::configure_global(
            kiln_memory::vram::VramProbeSelector::None,
            config.memory.governor_config_for_capacity(0),
        )
        .context("failed to configure the process-wide memory governor")?;
        let scheduler_config = SchedulerConfig {
            max_batch_tokens: config.server.max_batch_tokens.tokens(),
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: config.prefix_cache.enabled,
            prefix_cache_max_blocks: config.prefix_cache.max_blocks,
        };
        let num_blocks = 8192;
        let scheduler = Scheduler::new(scheduler_config, num_blocks);
        let engine = MockEngine::new(model_config.clone());
        let mut state = AppState::new_mock(
            model_config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            config.server.request_timeout_secs,
            served_model_id,
        );
        state.decode_runtime_config = kiln_server::batching_engine::resolve_decode_runtime_config(
            config.server.deterministic,
            config.server.max_decode_batch,
            None,
            config.server.max_batch_tokens,
        );
        state.batching_runtime_config = config.batching.resolve(
            kiln_server::config::BatchingBackendPolicy {
                batching_engine_default_enabled: false,
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                direct_decode_rendezvous:
                    kiln_server::config::DirectDecodeRendezvousBackendPolicy {
                        enabled: false,
                        max_batch: 1,
                        wait_us: 0,
                        mixed_seq_lens: false,
                    },
            },
            state.decode_runtime_config.max_decode_batch.effective,
        );
        state
    };

    let decode_runtime = state.decode_runtime_config;
    tracing::info!(
        deterministic = decode_runtime.deterministic.enabled,
        deterministic_source = %decode_runtime.deterministic.source,
        max_decode_batch_configured = ?decode_runtime.max_decode_batch.configured,
        max_decode_batch_configured_source = %decode_runtime.max_decode_batch.configured_source,
        max_decode_batch_backend_policy = decode_runtime.max_decode_batch.backend_policy,
        max_decode_batch_effective = decode_runtime.max_decode_batch.effective,
        max_decode_batch_effective_source = %decode_runtime.max_decode_batch.effective_source,
        "decode runtime configuration resolved"
    );

    // Apply server-level checkpoint_interval from config
    state.serving_profile = config.server.serving_profile;
    state.speculative_config = config.speculative;
    state.memory_config = config.memory.clone();
    state.training_runtime = kiln_train::TrainingRuntimeContext::new_for_device(
        state
            .training_runtime
            .runtime_device()
            .unwrap_or(kiln_tensor::Device::Cpu),
        state.vram_info,
        gradient_checkpoint_policy,
    );
    state.checkpoint_interval = config.training.checkpoint_interval;
    state.training_webhook_url = config.training.webhook_url.clone();
    state.max_queued_training_jobs = config.training.max_queued_jobs;
    state.max_tracked_jobs = config.training.max_tracked_jobs;
    state.tracked_job_ttl = std::time::Duration::from_secs(config.training.tracked_job_ttl_secs);
    state.teacher_credentials = Arc::new(config.teachers.clone());
    state.eval_mode = config.server.eval_mode;
    state.default_thinking_enabled = config.server.default_thinking_enabled;
    state.default_thinking_budget_tokens = config.server.default_thinking_budget_tokens;
    state.default_thinking_budget_ms = config.server.default_thinking_budget_ms;
    state.http_send_buffer_bytes = config.server.http_send_buffer_bytes;
    state.model_defaults_profile = model_defaults_profile;
    state.model_path = model_path.map(PathBuf::from);
    state.fold_reasoning_into_content = config.server.fold_reasoning_into_content;
    state.chat_performance_metadata = config.server.chat_performance_metadata;
    state.chat_config_hash_metadata = config.server.chat_config_hash_metadata;
    state.config_hashes = ConfigHashes::from_model_tokenizer(
        &state.model_config,
        state.tokenizer.as_ref(),
        kiln_env_config_hash(&config),
    );
    state.slow_request_warn_threshold = if config.server.slow_request_warn_secs == 0 {
        None
    } else {
        Some(std::time::Duration::from_secs(
            config.server.slow_request_warn_secs,
        ))
    };
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
    tracing::debug!(
        threshold_secs = config.server.slow_request_warn_secs,
        enabled = state.slow_request_warn_threshold.is_some(),
        "slow request watchdog configured"
    );
    tracing::debug!(
        enabled = state.chat_config_hash_metadata,
        model_config_hash = ?state.config_hashes.model_config_hash,
        tokenizer_config_hash = ?state.config_hashes.tokenizer_config_hash,
        chat_template_hash = ?state.config_hashes.chat_template_hash,
        training_chat_template_hash = ?state.config_hashes.training_chat_template_hash,
        kiln_env_config_hash = ?state.config_hashes.kiln_env_config_hash,
        "config hashes initialized"
    );
    tracing::debug!(enabled = state.eval_mode, "eval mode configured");
    tracing::info!(
        profile = state.model_defaults_profile.name,
        canonical_model_id = state.model_defaults_profile.canonical_model_id,
        canonical_served_model_id = state.model_defaults_profile.canonical_served_model_id,
        server_default_thinking_enabled = ?state.model_defaults_profile.server_default_thinking_enabled,
        template_default_thinking_enabled = state.model_defaults_profile.template_default_thinking_enabled,
        eval_mode_default_thinking_enabled = state.model_defaults_profile.eval_mode_default_thinking_enabled,
        adapter_dir = %state.adapter_dir.display(),
        adapter_dir_policy = state.model_defaults_profile.adapter_dir_policy,
        chat_template_policy = state.model_defaults_profile.chat_template_policy,
        supports_enable_thinking_kwarg = state.model_defaults_profile.supports_enable_thinking_kwarg,
        supports_tool_chat_template = state.model_defaults_profile.supports_tool_chat_template,
        "model defaults profile active"
    );
    tracing::debug!(
        default_thinking_enabled = ?state.default_thinking_enabled,
        default_thinking_budget_tokens = ?state.default_thinking_budget_tokens,
        default_thinking_budget_ms = ?state.default_thinking_budget_ms,
        "chat-template thinking default configured"
    );
    tracing::debug!(
        enabled = state.fold_reasoning_into_content,
        "reasoning-content compatibility folding configured"
    );

    // Restore terminal training jobs persisted from previous runs so the
    // /ui training queue still shows last week's history after a restart.
    {
        use kiln_server::training_history;
        let archived = training_history::load_all(&state.adapter_dir);
        if !archived.is_empty() {
            let mut jobs = state.training_jobs.write().unwrap();
            for job in archived.iter() {
                jobs.entry(job.job_id.clone())
                    .or_insert_with(|| job.clone());
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
                jobs.entry(job.job_id.clone())
                    .or_insert_with(|| job.clone());
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
        state.eval_webhook_url = cfg.webhook_url.clone();
        if let Some(ref url) = state.eval_webhook_url {
            tracing::info!(url = %url, "eval completion webhook enabled");
        }
    }

    // Durable request/response log: every inference request becomes a JSONL
    // row under <adapter_dir>/.requests (rotated + gzipped + retention-capped)
    // so production traffic is minable and trainable later.
    if config.request_log.enabled {
        let log_dir = config
            .request_log
            .dir
            .clone()
            .unwrap_or_else(|| state.adapter_dir.join(".requests"));
        match kiln_server::request_log::RequestLogger::spawn(
            log_dir.clone(),
            config.request_log.clone(),
        ) {
            Ok(logger) => {
                state.request_log = Some(logger);
                state.request_log_max_capture_bytes = config.request_log.max_capture_bytes;
                tracing::info!(dir = %log_dir.display(), "request log online");
            }
            Err(e) => {
                tracing::warn!(error = %e, dir = %log_dir.display(), "request log disabled: could not initialize");
            }
        }
    }

    // Embedded pi run engine: capacity + timeout from [agent] (defaults
    // apply when the section is omitted).
    {
        let agent_cfg = config.agent.clone().unwrap_or_default();
        state
            .agent_runs
            .apply_config(agent_cfg.max_concurrent_runs, agent_cfg.run_timeout_secs);
    }

    // Spawn the background training queue worker
    let shutdown_flag = state.shutdown.clone();
    kiln_server::training_queue::spawn_training_worker(state.clone(), shutdown_flag.clone());
    // Spawn the background eval queue worker
    kiln_server::eval::spawn_eval_worker(state.clone(), shutdown_flag.clone());

    // §10.6 flywheel scheduler: [agent] self_improve_interval_hours runs
    // the weekly loop automatically through the SAME submission path as
    // POST /v1/agent/self_improve (validation, teacher gate, queue caps).
    // DURABLE: next_run_unix_ms persists to
    // <adapter_dir>/.self_improve_scheduler.json, so frequent restarts
    // don't reset the timer (a server restarted daily would otherwise
    // NEVER fire a weekly round). Transient run failures log and retry
    // next interval; only an invalid [agent.self_improve] config stops
    // the loop (loudly).
    if let Some(agent_cfg) = config.agent.clone() {
        if let Some(hours) = agent_cfg.self_improve_interval_hours.filter(|&h| h > 0) {
            let scheduler_state = state.clone();
            let req_template = agent_cfg.self_improve.clone();
            let status_path = state.adapter_dir.join(".self_improve_scheduler.json");
            let interval_ms = hours.saturating_mul(3_600_000);
            let now_ms = kiln_server::recent_requests::now_unix_ms();
            // Resume the persisted cadence; clamp into [now, now+interval]
            // so a config change or clock jump can't park the loop years out.
            let mut status: kiln_server::state::SelfImproveSchedulerStatus =
                std::fs::read_to_string(&status_path)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
            status.interval_hours = hours;
            if status.next_run_unix_ms == 0 || status.next_run_unix_ms > now_ms + interval_ms {
                status.next_run_unix_ms = now_ms + interval_ms;
            }
            let persist = {
                let status_path = status_path.clone();
                move |s: &kiln_server::state::SelfImproveSchedulerStatus| {
                    if let Ok(body) = serde_json::to_vec(s) {
                        let _ = kiln_resource::locked_atomic_write(&status_path, &body);
                    }
                }
            };
            persist(&status);
            *scheduler_state.self_improve_scheduler.write().unwrap() = Some(status.clone());
            tracing::info!(
                interval_hours = hours,
                next_run_unix_ms = status.next_run_unix_ms,
                "self_improve scheduler armed"
            );
            tokio::spawn(async move {
                loop {
                    let now = kiln_server::recent_requests::now_unix_ms();
                    let wait_ms = status.next_run_unix_ms.saturating_sub(now).max(1_000);
                    tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                    if scheduler_state
                        .shutdown
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        break;
                    }
                    let req = match &req_template {
                        Some(v) => match serde_json::from_value(v.clone()) {
                            Ok(req) => req,
                            Err(e) => {
                                tracing::error!(
                                    error = %e,
                                    "[agent].self_improve config invalid; scheduler stopping"
                                );
                                break;
                            }
                        },
                        None => Default::default(),
                    };
                    let ran_at = kiln_server::recent_requests::now_unix_ms();
                    status.last_run_unix_ms = Some(ran_at);
                    match kiln_server::api::self_improve::submit_self_improve(&scheduler_state, req)
                    {
                        Ok(resp) => {
                            tracing::info!(
                                jobs = resp.job_ids.len(),
                                "scheduled self_improve round queued"
                            );
                            status.last_result =
                                Some(format!("queued {} job(s)", resp.job_ids.len()));
                            status.last_job_ids = resp.job_ids;
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = ?e,
                                "scheduled self_improve round failed to queue; will retry next interval"
                            );
                            status.last_result = Some(format!("failed: {e:?}"));
                            status.last_job_ids = Vec::new();
                        }
                    }
                    status.next_run_unix_ms = ran_at + interval_ms;
                    persist(&status);
                    *scheduler_state.self_improve_scheduler.write().unwrap() = Some(status.clone());
                }
            });
        }
    }

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            cli::print_bind_addr_in_use(host, port);
            std::process::exit(1);
        }
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!("failed to bind {addr}")));
        }
    };
    let http_send_buffer_preflight =
        preflight_http_send_buffer(&listener, config.server.http_send_buffer_bytes)?;
    state.http_send_buffer_preflight_actual_bytes =
        http_send_buffer_preflight.map(|application| application.actual_bytes);
    state.http_send_buffer_preflight_effective_bytes =
        http_send_buffer_preflight.map(|application| application.effective_bytes);

    let tokenizer_prewarm = state.tokenizer.clone();
    let prewarm_state = state.clone();
    // Cheap clones so the shutdown handler can reach the batching engine
    // after `api::router` consumes the state.
    let app_state_for_shutdown = state.clone();
    let app = api::router(state);

    let requested_http_send_buffer_bytes = config.server.http_send_buffer_bytes;
    let expected_http_send_buffer_application = http_send_buffer_preflight;
    let listener = listener.tap_io(move |stream| {
        let Some(requested_bytes) = requested_http_send_buffer_bytes else {
            return;
        };
        match configure_http_stream_send_buffer(stream, requested_bytes) {
            Ok(application) => {
                if Some(application) != expected_http_send_buffer_application {
                    let expected = expected_http_send_buffer_application
                        .expect("configured send buffer has a preflight result");
                    tracing::warn!(
                        requested_bytes,
                        preflight_actual_bytes = expected.actual_bytes,
                        preflight_effective_bytes = expected.effective_bytes,
                        accepted_actual_bytes = application.actual_bytes,
                        accepted_effective_bytes = application.effective_bytes,
                        "http_accepted_socket_send_buffer_preflight_mismatch"
                    );
                }
            }
            Err(error) => {
                let socket = SockRef::from(&*stream);
                let (actual_bytes, effective_bytes) = inspect_http_send_buffer(&socket);
                tracing::error!(
                    requested_bytes,
                    actual_bytes = actual_bytes.unwrap_or_default(),
                    actual_bytes_known = actual_bytes.is_some(),
                    effective_bytes = effective_bytes.unwrap_or_default(),
                    effective_bytes_known = effective_bytes.is_some(),
                    error = %error,
                    "http_accepted_socket_send_buffer_configuration_failed"
                );
                // `tap_io` cannot return a configuration error. Let the panic
                // escape Axum's accept loop so an ineffective opt-in is fatal.
                panic!(
                    "fatal accepted-socket SO_SNDBUF configuration failure: requested={requested_bytes}: {error}"
                );
            }
        }
    });
    tracing::debug!(
        host = %host,
        port = port,
        model_path = model_path.unwrap_or("none (mock mode)"),
        "kiln listening"
    );
    cli::print_ready_line(host, port);
    spawn_tokenizer_warmup(tokenizer_prewarm);
    // (#1082 Phase 2) Restore the cublasLt autotune cache from disk before
    // prewarm so this run reuses the tuned algos instead of re-running the
    // cublasLt heuristic search (~50-200ms x ~20 Qwen3.5-4B GEMM shapes).
    // Best-effort; a missing/corrupt file is ignored. Single-GPU kiln → dev 0.
    #[cfg(feature = "cuda")]
    {
        let restored = kiln_tensor::load_algo_cache_from_disk(0);
        if restored > 0 {
            tracing::info!(
                entries = restored,
                "cublaslt autotune cache restored from disk"
            );
        }
    }
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
    let (engine_for_shutdown, decode_batcher_for_shutdown) =
        match app_state_for_shutdown.backend.as_ref() {
            ModelBackend::Real {
                batching_engine,
                decode_batcher,
                ..
            } => (batching_engine.clone(), decode_batcher.clone()),
            ModelBackend::Mock { .. } => (None, None),
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

    if let Some(decode_batcher) = decode_batcher_for_shutdown {
        decode_batcher
            .shutdown()
            .context("stop and join decode batcher before accelerator teardown")?;
        tracing::debug!("decode batcher stopped and joined");
    }

    if let Some(cleanup) = model_snapshot_cleanup {
        cleanup
            .cleanup()
            .with_context(|| format!("remove model snapshot {}", cleanup.path().display()))?;
    }
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
    if let Err(error) = state.ensure_inference_admission_allowed() {
        state
            .inference_prewarm_complete
            .store(true, Ordering::Release);
        tracing::info!(
            serving_profile = %state.serving_profile.profile(),
            reason = %error,
            "skipping inference prewarm because inference admission is disabled"
        );
        return;
    }
    let ModelBackend::Real {
        runner,
        block_manager,
        paged_cache,
        ..
    } = state.backend.as_ref()
    else {
        return;
    };

    let startup_policy = {
        let runner_guard = runner.read().unwrap();
        runner_guard.backend_capabilities().startup
    };
    if !startup_policy.run_inference_prewarm {
        return;
    }

    let runner = runner.clone();
    let block_manager = block_manager.clone();
    let paged_cache = paged_cache.clone();
    let gpu_lock = state.gpu_lock.clone();
    let prewarm_complete = state.inference_prewarm_complete.clone();

    if startup_policy.decode_weight_prewarm_when_native_training
        && native_training_enabled_for_startup(startup_policy)
    {
        tracing::info!(
            "skipping synthetic inference prewarm because backend native training is enabled"
        );
        spawn_vulkan_decode_weight_prewarm(runner, gpu_lock, prewarm_complete);
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
            {
                let runner_guard = runner.read().unwrap();
                runner_guard.precompile_backend_startup_kernels()?;
            }

            // Prewarm is opportunistic. If a live request or training job has
            // the GPU first, skip prewarm rather than sitting in front of it.
            let Ok(_gpu_guard) = gpu_lock.try_write() else {
                tracing::info!("skipping inference prewarm because GPU is already busy");
                return Ok(());
            };

            {
                let runner_guard = runner.read().unwrap();
                runner_guard.precompile_backend_startup_kernels()?;
            }
            // Weight prewarm populates backend caches without replacing the
            // serving tensors; shared-tape training and portable fallback keep
            // the same authoritative values.
            let runner_guard = runner.read().unwrap();
            runner_guard
                .prewarm_backend_decode_weights()
                .context("backend decode weight prewarm failed")?;
            let params = SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                // `max_tokens = 1` only runs prefill and samples the first
                // token. Use two tokens so GPU backends also compile or tune
                // the decode path before the first live request reaches it.
                max_tokens: 2,
                repetition_penalty: 1.0,
                stop: Vec::new(),
                seed: Some(42),
                ..SamplingParams::default()
            };
            // Warm the base paged path used by every desktop request. Two
            // prompt sizes cover the short-chat decode buckets and populate
            // backend matmul autotune caches before the first live request.
            let prewarm_prompts: [Vec<u32>; 2] = [
                (1..=32).collect::<Vec<u32>>(),
                (1..=64).collect::<Vec<u32>>(),
            ];
            for prompt_tokens in prewarm_prompts {
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
            }
            Ok(())
        })
        .await;

        match prewarm {
            Ok(Ok(())) => {
                tracing::info!(
                    elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                    "background inference prewarm complete"
                );
                // (#1082 Phase 2) Prewarm just exercised every Qwen3.5-4B GEMM
                // shape, so the autotune cache is populated — flush it to disk
                // so future cold-starts skip backend heuristic search.
                #[cfg(feature = "cuda")]
                match kiln_tensor::flush_algo_cache_to_disk(0) {
                    Ok(n) if n > 0 => {
                        tracing::info!(entries = n, "cublaslt autotune cache flushed to disk")
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, "cublaslt autotune cache flush failed (continuing)")
                    }
                }
            }
            Ok(Err(err)) => tracing::warn!(error = %err, "background inference prewarm failed"),
            Err(err) => tracing::warn!(error = %err, "background inference prewarm task failed"),
        }
        prewarm_complete.store(true, Ordering::Release);
    });
}

fn spawn_vulkan_decode_weight_prewarm(
    runner: Arc<std::sync::RwLock<ModelRunner>>,
    gpu_lock: GpuCoordinationLock,
    prewarm_complete: Arc<std::sync::atomic::AtomicBool>,
) {
    tokio::spawn(async move {
        tracing::info!("starting Vulkan decode weight prewarm");
        let prewarm_start = std::time::Instant::now();
        let prewarm = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            // Pipeline compilation is cheap and independent of model working
            // buffers, so do it even if a request wins the GPU lock.
            {
                let runner_guard = runner.read().unwrap();
                runner_guard.precompile_backend_startup_kernels()?;
            }

            let Ok(_gpu_guard) = gpu_lock.try_write() else {
                tracing::info!("skipping Vulkan decode weight prewarm because GPU is already busy");
                return Ok(false);
            };

            {
                let runner_guard = runner.read().unwrap();
                runner_guard.precompile_backend_startup_kernels()?;
            }
            let runner_guard = runner.read().unwrap();
            runner_guard
                .prewarm_backend_decode_weights()
                .context("Vulkan decode weight prewarm failed")?;
            Ok(true)
        })
        .await;

        match prewarm {
            Ok(Ok(true)) => tracing::info!(
                elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                "Vulkan decode weight prewarm complete"
            ),
            Ok(Ok(false)) => tracing::info!(
                elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                "Vulkan decode weight prewarm skipped"
            ),
            Ok(Err(err)) => tracing::warn!(error = %err, "Vulkan decode weight prewarm failed"),
            Err(err) => tracing::warn!(error = %err, "Vulkan decode weight prewarm task failed"),
        }
        prewarm_complete.store(true, Ordering::Release);
    });
}

fn native_training_enabled_for_startup(policy: StartupCapabilities) -> bool {
    policy
        .native_training_env
        .and_then(env_tristate)
        .unwrap_or(policy.native_training_default_enabled)
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

    // (#1082 Phase 2) Persist the autotune cache on graceful shutdown so the
    // next cold-start reuses it — catches algos tuned during serving that the
    // post-warmup flush didn't capture. CPU-only + fast.
    #[cfg(feature = "cuda")]
    match kiln_tensor::flush_algo_cache_to_disk(0) {
        Ok(n) if n > 0 => {
            tracing::info!(entries = n, "cublaslt autotune cache flushed on shutdown")
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "cublaslt autotune cache flush failed on shutdown"),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_and_maintenance_profiles_force_eager_runners() {
        for profile in [
            kiln_server::config::ServingProfile::Stable,
            kiln_server::config::ServingProfile::Maintenance,
        ] {
            let options =
                resolve_model_runner_runtime_options(profile.runtime_policy(), true, Some(17));
            assert_eq!(
                options,
                ModelRunnerRuntimeOptions {
                    max_decode_batch: Some(17),
                    ..ModelRunnerRuntimeOptions::eager_only()
                }
            );
        }
    }

    #[test]
    fn experimental_profile_preserves_explicit_graph_eligibility() {
        let policy = kiln_server::config::ServingProfile::Experimental.runtime_policy();
        assert_eq!(
            resolve_model_runner_runtime_options(policy, true, Some(17)),
            ModelRunnerRuntimeOptions {
                cuda_graphs: true,
                rocm_graphs: true,
                metal_graphs: true,
                max_decode_batch: Some(17),
            }
        );
        assert!(!resolve_model_runner_runtime_options(policy, false, Some(17)).cuda_graphs);
    }

    #[test]
    fn send_buffer_readback_uses_platform_accounting() {
        let requested_bytes = 4096;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let actual_bytes = requested_bytes * 2;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let actual_bytes = requested_bytes;

        let application =
            validate_http_send_buffer_readback(requested_bytes, actual_bytes).unwrap();
        assert_eq!(application.actual_bytes, actual_bytes);
        assert_eq!(application.effective_bytes, requested_bytes);
    }

    #[test]
    fn send_buffer_readback_rejects_platform_normalized_clamp() {
        let requested_bytes = 6 * 1024 * 1024;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let actual_bytes = 8 * 1024 * 1024;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let actual_bytes = 4 * 1024 * 1024;

        let error = validate_http_send_buffer_readback(requested_bytes, actual_bytes).unwrap_err();
        assert!(error.to_string().contains("below requested"));
    }

    #[tokio::test]
    async fn listener_send_buffer_default_is_a_no_op() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let before = SockRef::from(&listener).send_buffer_size().unwrap();

        assert_eq!(preflight_http_send_buffer(&listener, None).unwrap(), None);

        let after = SockRef::from(&listener).send_buffer_size().unwrap();
        assert_eq!(after, before);
    }

    #[tokio::test]
    async fn listener_send_buffer_preflight_returns_structured_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let error = preflight_http_send_buffer(&listener, Some(usize::MAX)).unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("HTTP SO_SNDBUF listener preflight failed"));
        assert!(message.contains("requested_bytes="));
        assert!(message.contains("actual_bytes="));
        assert!(message.contains("effective_bytes="));
    }

    #[tokio::test]
    async fn accepted_socket_send_buffer_is_applied_and_bounded() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requested_bytes = 4096;
        let listener_application = preflight_http_send_buffer(&listener, Some(requested_bytes))
            .unwrap()
            .unwrap();
        assert!(listener_application.effective_bytes >= requested_bytes);

        let mut listener = listener.tap_io(move |stream| {
            configure_http_stream_send_buffer(stream, requested_bytes).unwrap();
        });
        let (client, (accepted, _)) =
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                tokio::join!(
                    tokio::net::TcpStream::connect(address),
                    axum::serve::Listener::accept(&mut listener),
                )
            })
            .await
            .expect("loopback accept timed out");
        let _client = client.unwrap();

        let actual_bytes = SockRef::from(&accepted).send_buffer_size().unwrap();
        let application =
            validate_http_send_buffer_readback(requested_bytes, actual_bytes).unwrap();
        assert!(application.effective_bytes >= requested_bytes);
    }
}

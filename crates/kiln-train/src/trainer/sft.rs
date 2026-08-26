use super::*;

/// Server-owned telemetry hook for the interval during which a training step
/// has exclusive GPU ownership. The returned observation is dropped before
/// the write guard, so readers can never acquire between the reported phase
/// ending and the writer actually releasing ownership.
pub trait GpuStepWriterObserver: Send + Sync {
    fn writer_acquired(self: std::sync::Arc<Self>) -> Box<dyn Send>;
}

pub(super) struct CoordinatedGpuWriteGuard {
    observation: Option<Box<dyn Send>>,
    guard: Option<tokio::sync::OwnedRwLockWriteGuard<()>>,
}

impl Drop for CoordinatedGpuWriteGuard {
    fn drop(&mut self) {
        drop(self.observation.take());
        drop(self.guard.take());
    }
}

/// Per-step GPU coordination that remains interruptible by the serving
/// backend's process-lifetime quarantine latch.
///
/// A quarantined inference request may intentionally retain its read owner
/// because dropping unknown device state is unsafe. A bare blocking write
/// would therefore strand SFT forever between steps. Polling acquisition lets
/// the trainer return the quarantine error while preserving that owner.
#[derive(Clone)]
pub struct GpuStepCoordination {
    pub(super) lock: std::sync::Arc<tokio::sync::RwLock<()>>,
    pub(super) backend_health: kiln_model::BackendHealthHandle,
    writer_observer: Option<std::sync::Arc<dyn GpuStepWriterObserver>>,
}

impl GpuStepCoordination {
    pub fn new(
        lock: std::sync::Arc<tokio::sync::RwLock<()>>,
        backend_health: kiln_model::BackendHealthHandle,
    ) -> Self {
        Self {
            lock,
            backend_health,
            writer_observer: None,
        }
    }

    pub fn with_writer_observer(
        mut self,
        observer: std::sync::Arc<dyn GpuStepWriterObserver>,
    ) -> Self {
        self.writer_observer = Some(observer);
        self
    }

    pub(super) fn blocking_write(&self) -> Result<CoordinatedGpuWriteGuard> {
        loop {
            self.backend_health.ensure_healthy()?;
            if let Ok(guard) = self.lock.clone().try_write_owned() {
                self.backend_health.ensure_healthy()?;
                let observation = self
                    .writer_observer
                    .as_ref()
                    .map(|observer| observer.clone().writer_acquired());
                return Ok(CoordinatedGpuWriteGuard {
                    observation,
                    guard: Some(guard),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub(super) fn blocking_gpu_phase<T>(
        &self,
        backend: &dyn BackendRuntime,
        workload: &'static str,
        phase: &'static str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<CoordinatedGpuPhase<T>> {
        let wait_started = Instant::now();
        let guard = self
            .blocking_write()
            .with_context(|| format!("acquire healthy backend for {workload} {phase}"))?;
        let wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;

        let held_started = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        let sync_result = ExternalYieldBackend::runtime_synchronize_external_yield(backend)
            .with_context(|| format!("synchronize backend after {workload} {phase}"));
        let held_ms = held_started.elapsed().as_secs_f64() * 1000.0;
        match (&result, &sync_result) {
            (Err(_), settlement) => {
                let sync_suffix = settlement
                    .as_ref()
                    .err()
                    .map(|error| format!("; settlement also failed: {error:#}"))
                    .unwrap_or_default();
                self.backend_health
                    .quarantine(format!("{workload} {phase} panicked{sync_suffix}"));
            }
            (Ok(_), Err(sync_error)) => self.backend_health.quarantine(format!(
                "{workload} {phase} external-yield synchronization failed: {sync_error:#}"
            )),
            (Ok(_), Ok(())) => {}
        }
        drop(guard);
        tracing::debug!(
            workload,
            phase,
            wait_ms,
            held_ms,
            "completed coordinated training GPU phase"
        );

        match result {
            Ok(operation_result) => match sync_result {
                Ok(()) => operation_result.map(|value| CoordinatedGpuPhase {
                    value,
                    wait_ms,
                    held_ms,
                }),
                Err(sync_error) => match operation_result {
                    Ok(_) => Err(sync_error),
                    Err(operation_error) => Err(anyhow::anyhow!(
                        "{workload} {phase} failed ({operation_error:#}) and backend settlement also failed ({sync_error:#})"
                    )),
                },
            },
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Run one bounded training GPU phase, settle the backend before releasing
    /// serving ownership, and quarantine the process if settlement is unknown.
    pub fn run_gpu_phase<T>(
        &self,
        backend: &dyn BackendRuntime,
        workload: &'static str,
        phase: &'static str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.blocking_gpu_phase(backend, workload, phase, operation)
            .map(|outcome| outcome.value)
    }
}

pub(super) struct CoordinatedGpuPhase<T> {
    pub(super) value: T,
    pub(super) wait_ms: f64,
    pub(super) held_ms: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct GrpoGpuWriterTimings {
    pub(super) wait_ms: f64,
    pub(super) held_ms: f64,
    pub(super) acquisitions: u64,
}

impl GrpoGpuWriterTimings {
    pub(super) fn apply_to(&self, timings: &mut GrpoBenchmarkTimings) {
        timings.gpu_writer_wait_ms = self.wait_ms;
        timings.gpu_writer_held_ms = self.held_ms;
        timings.gpu_writer_acquisitions = self.acquisitions;
    }
}

pub(super) fn run_coordinated_grpo_gpu_phase<T>(
    coordination: Option<&GpuStepCoordination>,
    backend: &dyn BackendRuntime,
    timings: &mut GrpoGpuWriterTimings,
    phase: &'static str,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(coordination) = coordination else {
        return operation();
    };

    let outcome = coordination.blocking_gpu_phase(backend, "GRPO", phase, operation)?;
    timings.wait_ms += outcome.wait_ms;
    timings.held_ms += outcome.held_ms;
    timings.acquisitions = timings.acquisitions.saturating_add(1);
    Ok(outcome.value)
}

#[allow(clippy::too_many_arguments)]
pub fn sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    sft_train_to(
        examples,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Train against adapters in `adapter_dir` while writing all new artifacts to
/// `output_adapter_dir`. The server uses this to keep an in-progress rewrite
/// invisible until its revision-barrier commit.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    sft_train_to_with_checkpoint_root(
        examples,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Staged-output SFT with a separate durable checkpoint root. Server training
/// uses this entry point so a process crash cannot discard already-published
/// resumable checkpoints with the temporary final-adapter staging tree.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    config
        .validate_native_contract()
        .context("validate native SFT profile before row admission")?;
    ensure_training_optimizer_device_supported(
        "SFT",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    ensure_training_optimizer_entry_supported(
        "SFT",
        weights,
        &runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    let prepared = crate::sft_ingestion::prepare_sft_examples(
        examples.iter().cloned(),
        tokenizer,
        config.invalid_row_policy,
        "rust_api",
        None,
    )?;
    sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
        &prepared.examples,
        &prepared.ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Standalone convenience wrapper for already-admitted SFT rows.
///
/// Server callers should use
/// [`sft_train_to_with_checkpoint_root_and_ingestion_with_runtime`] so the run
/// remains bound to their process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root_and_ingestion(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    ensure_training_optimizer_device_supported(
        "SFT",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
        examples,
        ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Server-owned SFT entry point with immutable process-lifetime runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    let runtime_device = ensure_training_optimizer_entry_supported(
        "SFT",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize SFT memory governor")?;
    config
        .validate_native_contract()
        .context("validate native SFT profile")?;
    anyhow::ensure!(
        ingestion.invalid_row_policy == config.invalid_row_policy,
        "SFT ingestion policy {} differs from trainer config {}",
        ingestion.invalid_row_policy,
        config.invalid_row_policy
    );
    crate::sft_ingestion::verify_prepared_sft_examples(examples, tokenizer, ingestion)
        .context("verify admitted SFT rows before training")?;
    sft_train_prepared_to_with_checkpoint_root(
        examples,
        ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        runtime,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sft_train_prepared_to_with_checkpoint_root(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "SFT checkpoint_interval must be greater than zero"
    );
    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = Some(ingestion.kept_corpus_sha256.clone());
    let ingestion_receipt_sha256 = crate::train_receipt::sha256_json_serializable(ingestion)
        .context("hash SFT ingestion receipt")?;
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        examples_read: ingestion.rows_read,
        examples_filtered: ingestion.rows_rejected,
        sft_ingestion: Some(ingestion.clone()),
        ..Default::default()
    };
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut lora_grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });

    // (#1082) `embed_tokens.device()` is a kt Device; the SFT path is now
    // kt-native end-to-end (kt `Parameter`s, kt AdamW state, kt tape
    // forward/backward), so keep `device` kt downstream. The only candle
    // touch left is the safetensors adapter I/O, which bridges the kt device
    // to candle locally inside `load_from_safetensors`/`save_peft`.
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "SFT",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    let backend_loss_route = TrainingLossBackend::runtime_sft_flce_loss_route(backend.as_ref());
    let sft_loss_route = runtime
        .admitted_sft_loss_route()
        .unwrap_or(backend_loss_route);
    anyhow::ensure!(
        sft_loss_route == backend_loss_route,
        "SFT loss route changed after admission: admitted `{}`, execution backend reports `{}`",
        sft_loss_route.as_str(),
        backend_loss_route.as_str(),
    );
    let bound_runtime = runtime.with_admitted_sft_loss_route(sft_loss_route);
    let runtime = &bound_runtime;
    // Training-session residency: upload a one-device copy of the weights
    // when the substrate needs it (Vulkan hybrid). Shadow `weights` so the
    // whole body trains against the resident copy; it drops at return. Route
    // drift has already failed closed before this potentially large copy.
    let resident_weights = resident_training_weights(weights, &device)?;
    let weights = resident_weights.as_ref().unwrap_or(weights);
    let checkpoint_boundary_policy = runtime.checkpoint_boundary_policy();
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);

    let learning_rate = config.effective_learning_rate();
    if let Some(explicit) = config.learning_rate
        && let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Sft),
        )
    {
        tracing::warn!(optimizer = ?config.optimizer, "SFT {warning}");
    }

    tracing::info!(
        num_examples = examples.len(),
        training_profile = %config.training_profile,
        epochs = config.epochs,
        lr = learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting SFT training"
    );

    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load SFT resume checkpoint")?;
    if config.checkpoint_interval.is_some() || resume_checkpoint.is_some() {
        validate_exact_training_provenance(weights)?;
    }
    let resume_init_seed = resume_checkpoint
        .as_ref()
        .map(|checkpoint| {
            let state = checkpoint
                .manifest
                .rng_states
                .get("lora-init")
                .context("SFT resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported SFT lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "SFT resume seed {restored} differs from requested seed {requested}"
        );
    }
    let requested_effective_seed = resume_init_seed.or(config.seed);

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_sft_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                sft_loss_route,
                config,
                config.seed,
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data_sha256,
                ingestion,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    tracing::info!(
        alpha_over_rank,
        allow_high_lora_scale = config.allow_high_lora_scale,
        "validated LoRA scaling"
    );

    // Open replay state (writes request record + lineage.json *before* the
    // optimizer step, so a crash mid-step still leaves a recoverable trail)
    // and resolve the effective seed.
    let replay_parent_adapter = resume_checkpoint
        .is_none()
        .then_some(config.base_adapter.as_deref())
        .flatten();
    let (replay_state, effective_seed) = match replay_ctx.as_ref() {
        Some(ctx) => {
            let (state, seed) = open_replay_state_to(
                ctx,
                requested_effective_seed,
                replay_parent_adapter,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            )?;
            (Some(state), Some(seed))
        }
        None => (
            None,
            Some(requested_effective_seed.unwrap_or_else(rand::random)),
        ),
    };
    let effective_seed_value = effective_seed.expect("SFT always resolves an effective seed");
    let effective_checkpoint_config =
        sft_checkpoint_effective_config(config, learning_rate, effective_seed_value)?;
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(training_data_sha256.as_deref(), "SFT training data")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_sft_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Sft,
            "resume checkpoint is not an SFT checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.effective_config == effective_checkpoint_config,
            "resume checkpoint effective SFT configuration differs from this request: checkpoint={}, request={}",
            checkpoint.manifest.effective_config,
            effective_checkpoint_config
        );
        anyhow::ensure!(
            checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint training data hash differs from this request"
        );
    }

    let base_adapter_result = if resume_checkpoint.is_some() {
        Ok(None)
    } else {
        resolve_and_validate_base_adapter_from_roots(
            config.base_adapter.as_deref(),
            adapter_dir,
            output_adapter_dir,
            adapter_name,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )
    };
    let base_adapter_dir = match base_adapter_result {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_sft_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                sft_loss_route,
                config,
                effective_seed,
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data_sha256,
                ingestion,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    // Initialize trainable LoRA parameters
    let mut params = TrainableLoraParams::initialize_seeded_with_precision_policy(
        model_config,
        weights,
        config.lora_rank,
        config.lora_alpha,
        &device,
        effective_seed,
        training_precision_policy,
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        let adapter_path =
            checkpoint.artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
        params.load_checkpoint_parameters(&adapter_path)?;
        tracing::info!(
            checkpoint = %checkpoint.root.display(),
            step = checkpoint.manifest.progress.global_step,
            "restored exact SFT adapter parameters"
        );
    } else if let Some(base_dir) = base_adapter_dir.as_deref() {
        let n_loaded = params.load_from_safetensors(base_dir, &device)?;
        tracing::info!(
            base = %base_dir.display(),
            num_tensors = n_loaded,
            "loaded base adapter — continuing SFT from those weights"
        );
    }

    // Allocate AdamW state if selected; SGD has no per-param state.
    // Register the per-param `m`/`v` device moment tensors alongside the
    // LoRA params so the on-device AdamW kernel's
    // `has_resident_activation(m/v)` gate passes (C1 fix — without this the
    // device path declines and a no-op interim corrupted the param).
    let mut opt_state = make_opt_state(&params, config.optimizer, learning_rate, &device)?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        let state_path = checkpoint
            .manifest
            .state_files
            .optimizer_state
            .as_deref()
            .map(|relative| checkpoint.artifact_path(relative))
            .transpose()?;
        match (opt_state.as_mut(), state_path) {
            (Some(state), Some(path)) => {
                let step = u32::try_from(checkpoint.manifest.progress.global_step)
                    .context("SFT resume optimizer step exceeds u32")?;
                state.load_checkpoint_state(&params, &path, step)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                anyhow::bail!("stateful SFT optimizer checkpoint has no optimizer artifact")
            }
            (None, Some(_)) => {
                anyhow::bail!("SGD SFT checkpoint unexpectedly contains optimizer state")
            }
        }
    }

    // Register only after checkpoint restoration. Registry identity is
    // process-local, so loading into already-registered tensors would leave
    // the restored host object and the resident device object out of sync.
    params.register_with_backend(&*backend)?;
    if let Some(state) = opt_state.as_ref() {
        state.register_with_backend(&*backend)?;
    }

    // Run the actual training body inside a closure so we can write the
    // outcome record (success or failure) before returning to the caller.
    let mut train_body = || -> Result<(PathBuf, f64)> {
        // Validate examples without retaining every tokenized long-context
        // payload at once. The step loop tokenizes the current example on
        // demand so full-file SFT jobs don't pin all input_ids/label masks for
        // the entire run.
        let mut valid_indices = Vec::new();
        let mut one_epoch_counts = crate::train_receipt::TokenCountReceipt::default();
        let mut max_seq_len_tokens: usize = 0;
        let mut valid_seq_lens = Vec::new();
        for (idx, ex) in examples.iter().enumerate() {
            match tokenize_for_training(ex, tokenizer) {
                Ok((input_ids, label_mask)) => {
                    let action_tokens = label_mask.iter().filter(|&&mask| mask).count() as u64;
                    one_epoch_counts.action_tokens =
                        one_epoch_counts.action_tokens.saturating_add(action_tokens);
                    one_epoch_counts.context_tokens =
                        one_epoch_counts.context_tokens.saturating_add(
                            input_ids.len().saturating_sub(action_tokens as usize) as u64,
                        );
                    if input_ids.len() > max_seq_len_tokens {
                        max_seq_len_tokens = input_ids.len();
                    }
                    valid_indices.push(idx);
                    valid_seq_lens.push(input_ids.len());
                }
                Err(e) => anyhow::bail!(
                    "admitted SFT row {} failed repeat tokenization before training: {e:#}",
                    idx + 1
                ),
            }
        }

        if valid_indices.is_empty() {
            anyhow::bail!("no valid training examples after tokenization");
        }
        anyhow::ensure!(
            valid_indices.len() == examples.len(),
            "not every admitted SFT row reached the training set"
        );
        data_stats.examples_trained = valid_indices.len().saturating_mul(config.epochs);
        token_counts.action_tokens = one_epoch_counts
            .action_tokens
            .saturating_mul(config.epochs as u64);
        token_counts.env_tokens = 0;
        token_counts.context_tokens = one_epoch_counts
            .context_tokens
            .saturating_mul(config.epochs as u64);

        // Resolve checkpointing at each step using the current example's
        // actual sequence length. The server preflight stamps the maximum
        // segment count needed for admission; treating that as a job-wide
        // fixed value makes a single very long row slow down every shorter
        // row in the same upload.
        let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
            weights,
            training_precision_policy,
            model_config_has_linear_attention(model_config),
        );
        tracing::info!(
            max_seq_len_tokens,
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            "SFT gradient checkpointing will resolve per example"
        );

        let total_steps = config
            .epochs
            .checked_mul(valid_indices.len())
            .context("SFT optimizer-step count overflow")?;
        let shuffle_seed = match resume_checkpoint.as_ref() {
            Some(checkpoint) => {
                let state = checkpoint
                    .manifest
                    .rng_states
                    .get("epoch-order")
                    .context("SFT resume checkpoint has no epoch-order RNG state")?;
                anyhow::ensure!(
                    state.algorithm == "kiln.epoch-order.v1" && state.state_file.is_none(),
                    "unsupported SFT epoch-order RNG state"
                );
                state.seed
            }
            None => effective_seed_value,
        };
        let mut has_checkpointed_step = false;
        let gradient_checkpoint_plan: Vec<_> = valid_seq_lens
            .iter()
            .map(|&seq_len| {
                let config_for_step = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    seq_len,
                    model_config.hidden_size,
                    model_config.intermediate_size,
                    model_config.vocab_size,
                    2,
                    activation_bytes_per_elem,
                    runtime,
                );
                let boundaries = checkpoint_segments_for_config(
                    weights,
                    &device,
                    seq_len,
                    config_for_step,
                    streaming_prefill,
                );
                has_checkpointed_step |= boundaries.is_some();
                serde_json::json!({
                    "seq_len": seq_len,
                    "enabled": config_for_step.enabled,
                    "num_segments": config_for_step.num_segments,
                    "auto_configured": config_for_step.auto_configured,
                    "boundaries": boundaries,
                })
            })
            .collect();
        ensure_sft_loss_route_supports_checkpointing(sft_loss_route, has_checkpointed_step)?;
        let gradient_checkpoint_plan_sha256 =
            crate::train_receipt::sha256_json_serializable(&gradient_checkpoint_plan)
                .context("hash SFT gradient-checkpoint plan")?;
        let checkpoint_descriptor = SftCheckpointDescriptor {
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: "sft-valid-example-order-v1".to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: valid_indices.len() as u64,
            },
            init_seed: effective_seed_value,
            shuffle_seed,
            optimizer: config.optimizer,
            learning_rate,
            total_steps,
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: sft_checkpoint_auxiliary_state(
                model_config,
                tokenizer,
                training_precision_policy,
                &valid_indices,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &gradient_checkpoint_plan_sha256,
                &ingestion_receipt_sha256,
                &training_runtime_planning_identity,
            )?,
        };
        if let (Some(checkpoint), Some(loop_state)) =
            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
        {
            checkpoint_descriptor.validate_resume(checkpoint, loop_state)?;
        }

        let mut global_step = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.global_step as usize);
        let mut loss_history = resume_loop_state
            .as_ref()
            .map_or_else(Vec::new, |state| state.loss_history.clone());
        let mut last_loss = resume_loop_state
            .as_ref()
            .map_or(0.0, |state| state.last_loss);
        let mut first_epoch_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.first_epoch_loss);
        let mut best_epoch_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.best_epoch_loss)
            .unwrap_or(f64::INFINITY);
        if let Some(state) = resume_loop_state.as_ref() {
            lora_grad_norms = state.lora_grad_norms.clone();
        }
        let start_epoch = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.epoch_index as usize);
        let start_cursor = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.cursor_in_epoch as usize);
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;
        let mut last_saved_step = resume_loop_state
            .as_ref()
            .map(|state| state.global_step as usize);
        const SFT_DIVERGENCE_RATIO: f64 = 8.0;
        const SFT_DIVERGENCE_MIN_INCREASE: f64 = 5.0;

        let pb = make_step_progress(total_steps, "sft training");
        if let Some(pb) = &pb {
            pb.set_position(global_step as u64);
        }

        for epoch in start_epoch..config.epochs {
            let order = epoch_order(shuffle_seed, epoch, valid_indices.len());
            let cursor_start = (epoch == start_epoch).then_some(start_cursor).unwrap_or(0);
            let mut epoch_loss = if epoch == start_epoch {
                resume_loop_state
                    .as_ref()
                    .map_or(0.0, |state| state.current_epoch_loss_sum)
            } else {
                0.0
            };
            let mut epoch_items = if epoch == start_epoch {
                resume_loop_state
                    .as_ref()
                    .map_or(0, |state| state.current_epoch_items as usize)
            } else {
                0
            };
            anyhow::ensure!(
                cursor_start <= order.len() && epoch_items == cursor_start,
                "SFT resume cursor is outside the current epoch"
            );
            let mut checkpoint_after_epoch = false;

            for (cursor, &order_idx) in order.iter().enumerate().skip(cursor_start) {
                let ex_idx = valid_indices[order_idx];
                let (input_ids, label_mask) =
                    tokenize_for_training(&examples[ex_idx], tokenizer)
                        .with_context(|| format!("retokenize SFT example {ex_idx}"))?;
                let ckpt_config = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    input_ids.len(),
                    model_config.hidden_size,
                    model_config.intermediate_size,
                    model_config.vocab_size,
                    2, // BF16 base weights (canonical kiln inference dtype)
                    activation_bytes_per_elem,
                    runtime,
                );
                let segments = checkpoint_segments_for_config(
                    weights,
                    &device,
                    input_ids.len(),
                    ckpt_config,
                    streaming_prefill,
                );
                let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
                if last_ckpt_log_key != Some(ckpt_log_key) {
                    if let Some(ref segs) = segments {
                        tracing::info!(
                            seq_len = input_ids.len(),
                            num_segments = segs.len(),
                            preflight_max_segments = ?config.grad_checkpoint_segments,
                            boundaries = ?segs,
                            "SFT gradient checkpointing enabled for step shape"
                        );
                    } else {
                        tracing::info!(
                            seq_len = input_ids.len(),
                            preflight_max_segments = ?config.grad_checkpoint_segments,
                            "SFT gradient checkpointing disabled for step shape"
                        );
                    }
                    last_ckpt_log_key = Some(ckpt_log_key);
                }
                let loss_val;

                // Per-STEP GPU coordination (the state.rs contract the
                // job-long server guard violated): hold the write lock
                // only across this step's forward/backward/optimizer so
                // in-flight inference streams interleave between steps
                // instead of freezing mid-token for the whole job.
                // Tokenization above runs lock-free (CPU work).
                let _step_gpu = match gpu_step_coordination.as_ref() {
                    Some(coordination) => Some(
                        coordination
                            .blocking_write()
                            .context("acquire healthy backend for SFT step")?,
                    ),
                    None => None,
                };

                // (#1082 candle-drop) The SFT forward/backward is now UNCONDITIONALLY
                // kt tape-authoritative — the candle checkpointed reverse + candle
                // `loss.backward()` paths are deleted, and the candle FLCE provider
                // opt-in (KILN_CUDA_FLCE) is removed (FLCE is kt-native).
                // `standard_forward_backward` and
                // `checkpointed_forward_backward_tape_authoritative_kt` both return
                // `GradSource::Kt`, consumed kt-native by the dispatchers.
                let grads: GradSource = if let Some(ref segs) = segments {
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    {
                        let (lv, kt_grads) = checkpointed_forward_backward_tape_authoritative_kt(
                            &*backend,
                            sft_loss_route,
                            &input_ids,
                            weights,
                            model_config,
                            &params,
                            &label_mask,
                            segs,
                            &device,
                            config.detect_anomaly,
                            checkpoint_boundary_policy,
                            streaming_prefill,
                        )?;
                        loss_val = lv;
                        GradSource::Kt(kt_grads)
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    {
                        // Non-GPU build: the kt tape adapters don't record on a
                        // CPU candle device, so checkpointed kt-tape backward is a
                        // GPU-only path (CUDA/Metal/Vulkan). The CPU smoke test uses
                        // the non-checkpointed `standard_forward_backward` path;
                        // reaching here means a CPU run requested checkpointing, which
                        // the candle-drop endgame does not support yet.
                        let _ = (segs, checkpoint_boundary_policy);
                        anyhow::bail!(
                            "gradient checkpointing requires a GPU feature (`cuda`, \
                             `metal`, or `vulkan`); the kt-tape checkpointed reverse \
                             is GPU-only post candle-drop)"
                        );
                    }
                } else {
                    let (lv, g) = standard_forward_backward_with_policy_and_loss_route(
                        &*backend,
                        sft_loss_route,
                        &input_ids,
                        weights,
                        model_config,
                        &params,
                        &label_mask,
                        &device,
                        config.detect_anomaly,
                        streaming_prefill,
                    )?;
                    loss_val = lv;
                    g
                };
                anyhow::ensure!(
                    loss_val.is_finite(),
                    "SFT loss became non-finite at epoch {} step {}: {loss_val}",
                    epoch + 1,
                    global_step + 1
                );
                observe_lora_grad_norms_dispatch(&mut lora_grad_norms, &params, &grads)?;
                optimizer_step_dispatch(
                    &*backend,
                    &mut params,
                    &grads,
                    learning_rate,
                    config.optimizer,
                    opt_state.as_mut(),
                )?;
                drop(_step_gpu);

                epoch_loss += loss_val;
                epoch_items += 1;
                last_loss = loss_val;
                loss_history.push(loss_val);

                global_step += 1;

                let checkpoint_due = config.checkpoint_interval.is_some_and(|interval| {
                    interval > 0 && global_step % interval == 0 && global_step < total_steps
                });
                if checkpoint_due {
                    if cursor + 1 == order.len() {
                        checkpoint_after_epoch = true;
                    } else {
                        let loop_state = SftCheckpointLoopState::capture(
                            global_step,
                            epoch,
                            cursor + 1,
                            &loss_history,
                            last_loss,
                            epoch_loss,
                            epoch_items,
                            first_epoch_loss,
                            best_epoch_loss,
                            &lora_grad_norms,
                        );
                        let path = checkpoint_descriptor.save(
                            checkpoint_output_dir,
                            &*backend,
                            &mut params,
                            &mut opt_state,
                            epoch,
                            cursor + 1,
                            &order,
                            &loop_state,
                            gpu_step_coordination.as_ref(),
                        )?;
                        last_saved_step = Some(global_step);
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved resumable SFT checkpoint"
                        );
                    }
                }

                if let Some(ref cb) = progress_cb {
                    let control = cb(TrainingProgress {
                        epoch: epoch + 1,
                        total_epochs: config.epochs,
                        step: global_step,
                        total_steps,
                        loss: loss_val,
                        progress: global_step as f32 / total_steps as f32,
                    });
                    if control == TrainControl::Stop && global_step < total_steps {
                        if last_saved_step != Some(global_step) {
                            let loop_state = SftCheckpointLoopState::capture(
                                global_step,
                                epoch,
                                cursor + 1,
                                &loss_history,
                                last_loss,
                                epoch_loss,
                                epoch_items,
                                first_epoch_loss,
                                best_epoch_loss,
                                &lora_grad_norms,
                            );
                            let path = checkpoint_descriptor.save(
                                checkpoint_output_dir,
                                &*backend,
                                &mut params,
                                &mut opt_state,
                                epoch,
                                cursor + 1,
                                &order,
                                &loop_state,
                                gpu_step_coordination.as_ref(),
                            )?;
                            tracing::info!(
                                step = global_step,
                                checkpoint = %path.display(),
                                "saved resumable SFT checkpoint before cancellation"
                            );
                        }
                        anyhow::bail!(
                            "training cancelled by user (stop requested at step boundary)"
                        );
                    }
                }

                if global_step % 10 == 0 || global_step == total_steps {
                    tracing::info!(
                        epoch = epoch + 1,
                        step = global_step,
                        total_steps,
                        loss = format!("{loss_val:.6}"),
                        "training step"
                    );
                }

                if let Some(pb) = &pb {
                    pb.set_message(format!("{loss_val:.6}"));
                    pb.inc(1);
                }
            }

            anyhow::ensure!(
                epoch_items == valid_indices.len(),
                "SFT epoch {} completed with {epoch_items} items, expected {}",
                epoch + 1,
                valid_indices.len()
            );
            let avg_loss = epoch_loss / epoch_items as f64;
            anyhow::ensure!(
                avg_loss.is_finite(),
                "SFT epoch {} average loss became non-finite: {avg_loss}",
                epoch + 1
            );
            let first_loss = *first_epoch_loss.get_or_insert(avg_loss);
            if epoch > 0
                && avg_loss > first_loss * SFT_DIVERGENCE_RATIO
                && avg_loss - best_epoch_loss > SFT_DIVERGENCE_MIN_INCREASE
            {
                anyhow::bail!(
                    "SFT loss diverged at epoch {}: avg_loss={avg_loss:.6}, \
                     first_epoch_loss={first_loss:.6}, best_epoch_loss={best_epoch_loss:.6}",
                    epoch + 1
                );
            }
            best_epoch_loss = best_epoch_loss.min(avg_loss);
            tracing::info!(
                epoch = epoch + 1,
                avg_loss = format!("{avg_loss:.6}"),
                "epoch complete"
            );
            if checkpoint_after_epoch && global_step < total_steps {
                let next_epoch = epoch + 1;
                let next_order = epoch_order(shuffle_seed, next_epoch, valid_indices.len());
                let loop_state = SftCheckpointLoopState::capture(
                    global_step,
                    next_epoch,
                    0,
                    &loss_history,
                    last_loss,
                    0.0,
                    0,
                    first_epoch_loss,
                    best_epoch_loss,
                    &lora_grad_norms,
                );
                let path = checkpoint_descriptor.save(
                    checkpoint_output_dir,
                    &*backend,
                    &mut params,
                    &mut opt_state,
                    next_epoch,
                    0,
                    &next_order,
                    &loop_state,
                    gpu_step_coordination.as_ref(),
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved resumable SFT checkpoint at epoch boundary"
                );
            }
        }

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        anyhow::ensure!(
            global_step == total_steps && loss_history.len() == total_steps,
            "SFT loop completed with inconsistent progress ({global_step}/{total_steps}, {} losses)",
            loss_history.len()
        );

        // MTP alignment phase (PR-B): train the native draft block's LoRA
        // against the freshly-tuned model so speculative decoding keeps its
        // acceptance rate under this adapter. Auto-on when the checkpoint
        // ships mtp.* tensors; config.train_mtp = false opts out. Soft-fail:
        // a draft-head alignment problem must not lose the main adapter.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        match run_mtp_alignment_phase(
            &*backend,
            weights,
            model_config,
            &mut params,
            examples,
            &valid_indices,
            tokenizer,
            config,
            &device,
            streaming_prefill,
        ) {
            Ok(Some((mtp_examples, mtp_initial_ce, mtp_final_ce))) => {
                tracing::info!(
                    examples = mtp_examples,
                    initial_ce = ?mtp_initial_ce,
                    final_ce = ?mtp_final_ce,
                    "MTP draft-block LoRA trained alongside the adapter"
                );
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "MTP alignment phase failed — saving the adapter WITHOUT \
                     a trained draft head (spec decode falls back to the base \
                     draft for this adapter)"
                );
                params.mtp = None;
            }
        }

        // Pull current Var values from registry into candle CPU
        // storage before final save_peft (the on-device optimizer
        // path leaves candle storage stale between steps).
        let final_snapshot_wait_started = Instant::now();
        let final_snapshot_gpu = gpu_step_coordination
            .as_ref()
            .map(GpuStepCoordination::blocking_write)
            .transpose()
            .context("acquire healthy backend for final SFT adapter snapshot")?;
        let final_snapshot_gpu_wait_ms = final_snapshot_wait_started.elapsed().as_millis() as u64;
        let final_snapshot_started = Instant::now();
        let synced = params
            .sync_to_master(&*backend)
            .context("capture final SFT adapter state from resident backend")?;
        let final_device_snapshot_ms = final_snapshot_started.elapsed().as_millis() as u64;
        drop(final_snapshot_gpu);
        tracing::info!(
            synced,
            final_snapshot_gpu_wait_ms,
            final_device_snapshot_ms,
            "captured final SFT adapter state before publication"
        );

        // Safetensors/config/receipt I/O consumes only the captured master
        // state and therefore cannot hold serving behind the GPU writer.
        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            "SFT training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let result = train_body();
    let adapter_smoke_test = if config.adapter_smoke_test && result.is_ok() {
        Some(run_adapter_smoke_test_best_effort(
            adapter_name,
            &*backend,
            weights,
            model_config,
            tokenizer,
            &params,
            config.adapter_smoke_prompts.as_deref(),
            streaming_prefill,
        ))
    } else {
        None
    };
    // Phase 4.1 cleanup: evict the LoRA Vars from the registry so a
    // long-running server doesn't accumulate stale entries from past
    // training jobs (each job creates fresh Vars with new TensorIds).
    // The eviction happens regardless of whether training succeeded
    // or failed.
    if let Some(state) = opt_state.as_ref() {
        state.evict_from_backend(&*backend);
    }
    params.evict_from_backend(&*backend);
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append SFT replay outcome record");
        }
    }
    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    write_sft_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        weights.base_weight_shard_manifest.as_ref(),
        weights.execution_provenance.as_ref(),
        training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
        sft_loss_route,
        config,
        effective_seed,
        Some(alpha_over_rank),
        base_adapter_dir.as_deref(),
        &output_dir,
        training_data_sha256,
        ingestion,
        data_stats,
        token_counts,
        run_started.elapsed().as_millis() as u64,
        lora_grad_norms.finish(),
        adapter_smoke_test,
        status_error,
    );
    result
        .map(|(dir, _)| dir)
        .map_err(crate::train_receipt::annotate_training_error)
}

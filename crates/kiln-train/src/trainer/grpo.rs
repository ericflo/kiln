use super::*;

/// Run GRPO training on the provided groups using the already-loaded model.
///
/// GRPO (Group Relative Policy Optimization) trains LoRA adapters by:
/// 1. Computing log-probs under the current policy (base + LoRA) for each completion
/// 2. Computing reference log-probs under the base model (no LoRA) — KL anchor
/// 3. Computing advantages from rewards normalized within each group
/// 4. Optimizing a clipped importance-sampling objective with KL penalty
///
/// Returns the path to the saved adapter directory.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_to(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
    )
}

/// Staged-output variant of [`grpo_train`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_to_with_coordination(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        None,
    )
}

/// Staged-output GRPO with bounded server GPU ownership. Direct callers should
/// normally use [`grpo_train_to`]; the server supplies coordination so
/// inference can run between optimizer groups and checkpoint snapshots.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_coordination(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
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
    grpo_train_to_with_checkpoint_root(
        groups,
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

/// Standalone staged-output GRPO with a separate durable checkpoint root.
///
/// Server callers should use [`grpo_train_to_with_checkpoint_root_and_runtime`]
/// to bind every per-group plan to their process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_checkpoint_root(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
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
        "GRPO",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    grpo_train_to_with_checkpoint_root_and_runtime(
        groups,
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

/// Server-owned inline GRPO entry point with immutable runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_checkpoint_root_and_runtime(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
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
        "GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize GRPO memory governor")?;
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "GRPO checkpoint_interval must be greater than zero"
    );
    // Fail fast on loss compositions the kt-tape path cannot train —
    // BEFORE any forward pass. The old order discovered this per-step,
    // after the rollout + reference forwards had already burned GPU time.
    let has_env_tokens = groups.iter().any(|g| {
        g.completions.iter().any(|c| {
            c.trajectory
                .iter()
                .any(|seg| seg.kind == crate::trajectory::TurnKind::Observation)
        })
    });
    config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(|e| anyhow::anyhow!("GRPO loss config: {e}"))?;
    config
        .validate_policy_config()
        .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;

    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = crate::train_receipt::sha256_json_serializable(&groups);
    let openenv_training_data = crate::openenv_training_data_provenance(groups)
        .map_err(anyhow::Error::msg)
        .context("validate inline GRPO OpenEnv corpus provenance")?;
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(training_data_sha256.as_deref(), "GRPO training data")?;
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });
    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load GRPO resume checkpoint")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_grpo_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Grpo,
            "resume checkpoint is not a GRPO checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.data.source_kind == GrpoCheckpointRoute::Inline.source_kind()
                && checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint inline GRPO data identity differs from this request"
        );
        anyhow::ensure!(
            resume_loop_state
                .as_ref()
                .is_some_and(|state| state.route == GrpoCheckpointRoute::Inline),
            "resume checkpoint was not produced by inline GRPO"
        );
    }
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
                .context("GRPO resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported GRPO lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "GRPO resume seed {restored} differs from requested seed {requested}"
        );
    }
    let requested_effective_seed = resume_init_seed.or(config.seed);
    let mut gpu_writer_timings = resume_loop_state
        .as_ref()
        .map_or_else(GrpoGpuWriterTimings::default, |state| {
            state.gpu_writer_timings.clone()
        });
    // (#1082) `embed_tokens.device()` is a kt Device; the GRPO body is now
    // kt-native (kt `Parameter`s, kt AdamW state, kt tape forward/backward),
    // so keep `device` kt downstream. No candle touch remains — the
    // safetensors adapter I/O is kt-native
    // (`kiln_tensor::safetensors::{load_cpu, save_cpu}`).
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    ensure_tape_forward_backward_supported("GRPO", weights, backend.as_ref())?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "GRPO",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    // Training-session residency: upload a one-device copy of the weights
    // when the substrate needs it (Vulkan hybrid). Shadow `weights` so the
    // whole body trains against the resident copy; it drops at return.
    let resident_weights = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "resident model setup",
        || resident_training_weights(weights, &device),
    )?;
    let weights = resident_weights.as_ref().unwrap_or(weights);
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);

    let total_completions: usize = groups.iter().map(|g| g.completions.len()).sum();
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        groups_read: groups.len(),
        completions_read: total_completions,
        ..Default::default()
    };
    let mut token_counts = resume_loop_state
        .as_ref()
        .map_or_else(crate::train_receipt::TokenCountReceipt::default, |state| {
            state.token_counts.clone()
        });
    let mut echo_metrics = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::EchoActivityMetrics::default,
        |state| state.echo_metrics.clone(),
    );
    let mut reward_stats = crate::train_receipt::RewardStatsReceipt::default();
    let mut lora_grad_norms = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::LoraGradNormAccumulator::default,
        |state| state.lora_grad_norms.clone(),
    );
    let mut policy_audit = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::GrpoPolicyAuditAccumulator::default,
        |state| state.policy_audit.clone(),
    );
    let mut phase_timings = resume_loop_state
        .as_ref()
        .map_or_else(GrpoBenchmarkTimings::default, |state| {
            state.phase_timings.clone()
        });
    let mut dynamic_groups_filtered = resume_loop_state
        .as_ref()
        .map_or(0, |state| state.dynamic_groups_filtered as usize);
    let learning_rate = config.effective_learning_rate();
    if let Some(explicit) = config.learning_rate
        && let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Grpo),
        )
    {
        tracing::warn!(optimizer = ?config.optimizer, "GRPO {warning}");
    }
    tracing::info!(
        num_groups = groups.len(),
        total_completions,
        total_input_groups = groups.len(),
        total_input_completions = total_completions,
        lr = learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting GRPO training"
    );
    tracing::info!(
        groups = groups.len(),
        completions = total_completions,
        "GRPO data loaded"
    );

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                config.seed,
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                crate::train_receipt::TrainingDataReceipt {
                    source: "inline_grpo_groups".to_string(),
                    path: None,
                    sha256: training_data_sha256,
                    openenv: openenv_training_data.clone(),
                },
                data_stats,
                reward_stats,
                token_counts,
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                dynamic_groups_filtered,
                None,
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
    // optimizer step) and resolve the effective seed.
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
    let effective_seed_value = effective_seed.expect("GRPO always resolves an effective seed");
    let effective_checkpoint_config =
        grpo_checkpoint_effective_config(config, learning_rate, effective_seed_value)?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.effective_config == effective_checkpoint_config,
            "resume checkpoint effective GRPO configuration differs from this request: checkpoint={}, request={}",
            checkpoint.manifest.effective_config,
            effective_checkpoint_config
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
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                effective_seed,
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                crate::train_receipt::TrainingDataReceipt {
                    source: "inline_grpo_groups".to_string(),
                    path: None,
                    sha256: training_data_sha256,
                    openenv: openenv_training_data.clone(),
                },
                data_stats,
                reward_stats,
                token_counts,
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                dynamic_groups_filtered,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    // Initialization, optional base-adapter upload, and registry admission all
    // mutate backend state. Keep them in one explicit setup phase, then release
    // serving before reward filtering and tokenization.
    let (mut params, mut opt_state) = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "adapter and optimizer setup",
        || {
            let mut params = TrainableLoraParams::initialize_seeded_with_precision_policy(
                model_config,
                weights,
                config.lora_rank,
                config.lora_alpha,
                &device,
                Some(effective_seed_value),
                training_precision_policy,
            )?;

            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let adapter_path = checkpoint
                    .artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
                params.load_checkpoint_parameters(&adapter_path)?;
                tracing::info!(
                    checkpoint = %checkpoint.root.display(),
                    step = checkpoint.manifest.progress.global_step,
                    "restored exact GRPO adapter parameters"
                );
            } else if let Some(base_dir) = base_adapter_dir.as_deref() {
                let n_loaded = params.load_from_safetensors(base_dir, &device)?;
                tracing::info!(
                    base = %base_dir.display(),
                    num_tensors = n_loaded,
                    "loaded base adapter — continuing GRPO from those weights"
                );
            }

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
                            .context("GRPO resume optimizer step exceeds u32")?;
                        state.load_checkpoint_state(&params, &path, step)?;
                    }
                    (None, None) => {}
                    (Some(_), None) => anyhow::bail!(
                        "stateful GRPO optimizer checkpoint has no optimizer artifact"
                    ),
                    (None, Some(_)) => {
                        anyhow::bail!("SGD GRPO checkpoint unexpectedly contains optimizer state")
                    }
                }
            }
            // Registry identity is process-local. Restore host tensors before
            // registration so resident copies cannot retain seeded state.
            params.register_with_backend(&*backend)?;
            if let Some(state) = opt_state.as_ref() {
                state.register_with_backend(&*backend)?;
            }
            Ok((params, opt_state))
        },
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    let mut train_body = || -> Result<(PathBuf, f64)> {
        let dynamic_sampling = config.dynamic_sampling;
        let mut dynamic_dropped: usize = 0;
        let mut tokenization_failed: usize = 0;
        let input_reward_groups: Vec<Vec<f64>> = groups
            .iter()
            .map(|group| {
                group
                    .completions
                    .iter()
                    .map(|completion| completion.reward)
                    .collect()
            })
            .collect();
        reward_stats = crate::train_receipt::reward_stats_from_groups_with_threshold(
            input_reward_groups.iter().map(Vec::as_slice),
            config.reward_saturation_threshold,
        );
        crate::train_receipt::warn_reward_diagnostics(
            "grpo_startup",
            adapter_name,
            &reward_stats,
            config.reward_saturation_threshold,
            config.reward_low_variance_threshold,
        );
        let reward_filter_plan = build_reward_filter_plan(
            config,
            &output_dir,
            "inline_grpo_groups",
            groups
                .iter()
                .enumerate()
                .map(|(idx, group)| RewardFilterInputGroup {
                    id: format!("group:{}", idx + 1),
                    source_index: idx + 1,
                    source_line: None,
                    reward_variance: reward_filter_variance(
                        &group
                            .completions
                            .iter()
                            .map(|completion| completion.reward)
                            .collect::<Vec<_>>(),
                    ),
                })
                .collect(),
        )?;
        if let Some(plan) = reward_filter_plan.as_ref() {
            record_reward_filter_plan(&mut data_stats, plan);
            data_stats.groups_filtered = data_stats
                .groups_filtered
                .saturating_add(plan.groups_dropped);
            tracing::info!(
                kept = plan.groups_kept,
                dropped = plan.groups_dropped,
                sidecar = %plan.sidecar_path.display(),
                "GRPO reward variance filter applied"
            );
            if let Some(reason) = plan.failure_reason.as_ref() {
                anyhow::bail!("{reason}");
            }
            if plan.skip_training {
                params.save_peft(&output_dir, model_config.num_layers)?;
                tracing::info!(
                    adapter = adapter_name,
                    path = %output_dir.display(),
                    "GRPO reward variance filter skipped training"
                );
                return Ok((output_dir.clone(), 0.0));
            }
        }

        // Tokenize all completions: for each group, tokenize prompt + each completion.
        // When dynamic_sampling is enabled (DAPO, arXiv:2503.14476), groups whose
        // completions all share the same reward are dropped before tokenization —
        // their advantage vector is uniformly zero and would contribute no
        // policy-gradient signal anyway.
        tracing::info!(
            groups = groups.len(),
            completions = total_completions,
            dynamic_sampling,
            "GRPO tokenize start"
        );
        let tokenize_all_started = Instant::now();
        let mut tokenized_groups: Vec<TokenizedGrpoGroup> = Vec::new();
        let mut trainable_source_indices: Vec<u64> = Vec::new();
        for (idx, group) in groups.iter().enumerate() {
            let source_index = idx + 1;
            if let Some(plan) = reward_filter_plan.as_ref()
                && !plan.keeps_source_index(source_index)
            {
                continue;
            }
            if dynamic_sampling && is_degenerate_grpo_group(group) {
                dynamic_dropped += 1;
                continue;
            }
            let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
            match tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut phase_timings)) {
                Ok(tgroup) => {
                    validate_tokenized_behavior_policy(&tgroup, config.behavior_policy)
                        .with_context(|| {
                            format!("validate GRPO group {source_index} behavior provenance")
                        })?;
                    tokenized_groups.push(tgroup);
                    trainable_source_indices.push(idx as u64);
                }
                Err(e) => {
                    if config.behavior_policy == BehaviorPolicy::Recorded {
                        return Err(e).with_context(|| {
                            format!(
                                "tokenize GRPO group {source_index} with required recorded behavior provenance"
                            )
                        });
                    }
                    tokenization_failed += 1;
                    tracing::warn!("skipping GRPO group: {e}");
                }
            }
        }
        if let Some(state) = resume_loop_state.as_ref() {
            anyhow::ensure!(
                state.dynamic_groups_filtered as usize == dynamic_dropped,
                "resume checkpoint dynamic-sampling selection differs from this request"
            );
        }
        dynamic_groups_filtered = dynamic_dropped;
        data_stats.groups_filtered = data_stats
            .reward_groups_filtered
            .saturating_add(dynamic_dropped)
            .saturating_add(tokenization_failed);
        let planned_token_counts = token_counts_for_grpo_groups(&tokenized_groups);
        let planned_completions: usize = tokenized_groups
            .iter()
            .map(|group| group.completions.len())
            .sum();
        if let Some(state) = resume_loop_state.as_ref() {
            let current_reward_filter_sidecar = data_stats.reward_filter_sidecar.clone();
            anyhow::ensure!(
                state.data_stats.groups_read == data_stats.groups_read
                    && state.data_stats.completions_read == data_stats.completions_read
                    && state.data_stats.groups_filtered == data_stats.groups_filtered
                    && state.data_stats.reward_groups_filtered == data_stats.reward_groups_filtered
                    && state.data_stats.reward_groups_kept == data_stats.reward_groups_kept,
                "resume checkpoint GRPO filtering statistics differ from this request"
            );
            data_stats = state.data_stats.clone();
            data_stats.reward_filter_sidecar = current_reward_filter_sidecar;
        }
        tracing::info!(
            groups = tokenized_groups.len(),
            completions = planned_completions,
            action_tokens = planned_token_counts.action_tokens,
            env_tokens = planned_token_counts.env_tokens,
            context_tokens = planned_token_counts.context_tokens,
            elapsed_ms = tokenize_all_started.elapsed().as_millis() as u64,
            "GRPO tokenize end"
        );
        crate::train_receipt::warn_echo_enabled_without_env_tokens(
            "grpo",
            config.loss.echo_enabled(),
            &planned_token_counts,
        );

        if dynamic_dropped > 0 {
            tracing::info!(
                dropped = dynamic_dropped,
                total = groups.len(),
                "GRPO dynamic sampling: dropped degenerate groups (all rewards equal)"
            );
        }

        if tokenized_groups.is_empty() {
            anyhow::bail!("no valid GRPO groups after tokenization");
        }

        // Compute the max seq_len across every completion in every group
        // so the auto-tuner sizes checkpointing against the longest path,
        // not the average.
        let max_seq_len_tokens: usize = tokenized_groups
            .iter()
            .flat_map(|g| g.completions.iter())
            .map(|c| c.input_ids.len())
            .max()
            .unwrap_or(0);

        // Resolve checkpointing per group from the actual longest
        // completion in that group. The submission preflight covers the
        // worst-case group for admission, but using that segment count for
        // every group needlessly slows shorter groups.
        let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
            weights,
            training_precision_policy,
            model_config_has_linear_attention(model_config),
        );
        tracing::info!(
            max_seq_len_tokens,
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            "GRPO gradient checkpointing will resolve per group"
        );

        let total_steps = tokenized_groups.len();
        let gradient_checkpoint_plan: Vec<_> = tokenized_groups
            .iter()
            .zip(&trainable_source_indices)
            .map(|(group, source_index)| {
                let max_seq_len = group
                    .completions
                    .iter()
                    .map(|completion| completion.input_ids.len())
                    .max()
                    .unwrap_or(0);
                let resolved = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    max_seq_len,
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
                    max_seq_len,
                    resolved,
                    streaming_prefill,
                );
                serde_json::json!({
                    "source_index": source_index,
                    "max_seq_len": max_seq_len,
                    "enabled": resolved.enabled,
                    "num_segments": resolved.num_segments,
                    "auto_configured": resolved.auto_configured,
                    "boundaries": boundaries,
                })
            })
            .collect();
        let trainable_order_sha256 =
            crate::train_receipt::sha256_json_serializable(&trainable_source_indices)
                .context("hash inline GRPO trainable order")?;
        let gradient_checkpoint_plan_sha256 =
            crate::train_receipt::sha256_json_serializable(&gradient_checkpoint_plan)
                .context("hash inline GRPO gradient-checkpoint plan")?;
        let ema_refresh_every = if config.kl_penalty_enabled() {
            match &config.kl_reference_policy {
                KlReferencePolicy::Ema { refresh_every, .. } => Some(*refresh_every),
                _ => None,
            }
        } else {
            None
        };
        let checkpoint_descriptor = GrpoCheckpointDescriptor {
            route: GrpoCheckpointRoute::Inline,
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: GrpoCheckpointRoute::Inline.source_kind().to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: total_steps as u64,
            },
            init_seed: effective_seed_value,
            optimizer: config.optimizer,
            learning_rate,
            total_steps,
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: grpo_checkpoint_auxiliary_state(
                GrpoCheckpointRoute::Inline,
                model_config,
                tokenizer,
                training_precision_policy,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &trainable_order_sha256,
                &gradient_checkpoint_plan_sha256,
                &training_runtime_planning_identity,
            ),
            ema_refresh_every,
        };
        if let (Some(checkpoint), Some(loop_state)) =
            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
        {
            checkpoint_descriptor.validate_resume(checkpoint, loop_state)?;
        }

        let mut global_step = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.global_step as usize);
        let mut processed_completions = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.processed_completions as usize);
        let mut loss_history = resume_loop_state
            .as_ref()
            .map_or_else(Vec::new, |state| state.loss_history.clone());
        let mut last_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.last_loss)
            .unwrap_or(0.0);
        let mut last_saved_step = resume_loop_state
            .as_ref()
            .map(|state| state.global_step as usize);
        anyhow::ensure!(
            global_step <= total_steps,
            "GRPO resume cursor {global_step} exceeds {total_steps} trainable groups"
        );
        let expected_processed_completions: usize = tokenized_groups
            .iter()
            .take(global_step)
            .map(|group| group.completions.len())
            .sum();
        let expected_token_counts = token_counts_for_grpo_groups(&tokenized_groups[..global_step]);
        anyhow::ensure!(
            processed_completions == expected_processed_completions
                && token_counts == expected_token_counts,
            "GRPO resume diagnostics do not match the committed trainable prefix"
        );
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;

        let pb = make_step_progress(total_steps, "grpo training");
        if let Some(pb) = &pb {
            pb.set_position(global_step as u64);
        }

        // Phase 3b: maintain an EMA-snapshot LoRA when
        // `KlReferencePolicy::Ema` is configured. Initialized eagerly to a
        // deepcopy of the (post-init, pre-train) LoRA so the very first
        // group's reference forward already runs against a frozen snapshot
        // rather than the live policy.
        let mut ema_ref_state = if config.kl_penalty_enabled() {
            match &config.kl_reference_policy {
                KlReferencePolicy::Ema {
                    decay,
                    refresh_every,
                } => {
                    let (snapshot, groups_since_refresh) =
                        if let (Some(checkpoint), Some(loop_state)) =
                            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
                        {
                            let relative = checkpoint
                                .manifest
                                .state_files
                                .reference_state
                                .as_deref()
                                .context("EMA GRPO resume checkpoint has no reference state")?;
                            let path = checkpoint.artifact_path(relative)?;
                            (
                                load_lora_reference_checkpoint(&path, &params, &device)?,
                                loop_state
                                    .ema_groups_since_refresh
                                    .context("EMA GRPO resume checkpoint has no cadence cursor")?
                                    as usize,
                            )
                        } else {
                            (
                                run_coordinated_grpo_gpu_phase(
                                    gpu_step_coordination.as_ref(),
                                    &*backend,
                                    &mut gpu_writer_timings,
                                    "initial EMA reference snapshot",
                                    || {
                                        lora_snapshot_capture_or_blend(
                                            &params, None, *decay, &device,
                                        )
                                        .context("initial EMA reference snapshot")
                                    },
                                )?,
                                0,
                            )
                        };
                    Some(EmaReferenceState {
                        snapshot,
                        groups_since_refresh,
                        refresh_every: *refresh_every,
                        decay: *decay,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        for (group_idx, tgroup) in tokenized_groups.iter().enumerate().skip(global_step) {
            let num_completions = tgroup.completions.len();
            let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
            let group_max_seq_len = tgroup
                .completions
                .iter()
                .map(|completion| completion.input_ids.len())
                .max()
                .unwrap_or(0);
            let ckpt_config = checkpoint_config_for_training_step(
                weights,
                &device,
                config.grad_checkpoint_segments,
                model_config.num_layers,
                group_max_seq_len,
                model_config.hidden_size,
                model_config.intermediate_size,
                model_config.vocab_size,
                2, // BF16 base weights
                activation_bytes_per_elem,
                runtime,
            );
            let segments = checkpoint_segments_for_config(
                weights,
                &device,
                group_max_seq_len,
                ckpt_config,
                streaming_prefill,
            );
            let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
            if last_ckpt_log_key != Some(ckpt_log_key) {
                if let Some(ref segs) = segments {
                    tracing::info!(
                        group = group_idx + 1,
                        max_seq_len = group_max_seq_len,
                        num_segments = segs.len(),
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        boundaries = ?segs,
                        "GRPO gradient checkpointing enabled for group shape"
                    );
                } else {
                    tracing::info!(
                        group = group_idx + 1,
                        max_seq_len = group_max_seq_len,
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        "GRPO gradient checkpointing disabled for group shape"
                    );
                }
                last_ckpt_log_key = Some(ckpt_log_key);
            }
            let step_report = run_coordinated_grpo_gpu_phase(
                gpu_step_coordination.as_ref(),
                &*backend,
                &mut gpu_writer_timings,
                "optimizer group",
                || {
                    let step_report = train_tokenized_grpo_group_with_grad_norms(
                        &*backend,
                        tgroup,
                        weights,
                        model_config,
                        &mut params,
                        config,
                        segments.as_deref(),
                        &device,
                        opt_state.as_mut(),
                        &mut lora_grad_norms,
                        &lora_grad_index,
                        &mut policy_audit,
                        ema_ref_state.as_ref().map(|s| &s.snapshot),
                        Some(&mut phase_timings),
                        streaming_prefill,
                    )?;

                    // Refresh while the same writer is held: both the policy
                    // update and the frozen reference transition form one
                    // exact optimizer-group boundary.
                    if let Some(state) = ema_ref_state.as_mut() {
                        state.groups_since_refresh += 1;
                        if state.groups_since_refresh >= state.refresh_every {
                            params
                                .sync_to_master(&*backend)
                                .context("sync policy before EMA reference refresh")?;
                            state.snapshot = lora_snapshot_capture_or_blend(
                                &params,
                                Some(&state.snapshot),
                                state.decay,
                                &device,
                            )
                            .context("EMA reference snapshot refresh")?;
                            state.groups_since_refresh = 0;
                            tracing::debug!(
                                group = group_idx + 1,
                                refresh_every = state.refresh_every,
                                decay = state.decay,
                                "GRPO EMA reference snapshot refreshed"
                            );
                        }
                    }
                    Ok(step_report)
                },
            )?;
            let avg_group_loss = step_report.loss;
            anyhow::ensure!(
                avg_group_loss.is_finite(),
                "GRPO loss became non-finite at group {}: {avg_group_loss}",
                group_idx + 1
            );
            echo_metrics.observe_env_ce(step_report.echo_env_ce);
            last_loss = avg_group_loss;
            loss_history.push(avg_group_loss);
            global_step += 1;
            processed_completions = processed_completions.saturating_add(num_completions);
            token_counts.add_from(&group_counts);
            data_stats.groups_trained = global_step;
            data_stats.completions_trained = processed_completions;

            let checkpoint_due = config
                .checkpoint_interval
                .is_some_and(|interval| global_step % interval == 0 && global_step < total_steps);
            if checkpoint_due {
                let mut loop_state = GrpoCheckpointLoopState::capture(
                    GrpoCheckpointRoute::Inline,
                    global_step,
                    None,
                    None,
                    processed_completions,
                    &loss_history,
                    &data_stats,
                    &token_counts,
                    dynamic_groups_filtered,
                    &echo_metrics,
                    &lora_grad_norms,
                    &policy_audit,
                    &phase_timings,
                    &gpu_writer_timings,
                    ema_ref_state.as_ref(),
                );
                let path = checkpoint_descriptor.save(
                    checkpoint_output_dir,
                    &*backend,
                    &mut params,
                    &mut opt_state,
                    ema_ref_state.as_ref(),
                    &mut loop_state,
                    gpu_step_coordination.as_ref(),
                    &mut gpu_writer_timings,
                    "checkpoint device snapshot",
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved exact GRPO training checkpoint"
                );
            }

            if let Some(ref cb) = progress_cb {
                let control = cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step: global_step,
                    total_steps,
                    loss: avg_group_loss,
                    progress: global_step as f32 / total_steps as f32,
                });
                if control == TrainControl::Stop && global_step < total_steps {
                    if last_saved_step != Some(global_step) {
                        let mut loop_state = GrpoCheckpointLoopState::capture(
                            GrpoCheckpointRoute::Inline,
                            global_step,
                            None,
                            None,
                            processed_completions,
                            &loss_history,
                            &data_stats,
                            &token_counts,
                            dynamic_groups_filtered,
                            &echo_metrics,
                            &lora_grad_norms,
                            &policy_audit,
                            &phase_timings,
                            &gpu_writer_timings,
                            ema_ref_state.as_ref(),
                        );
                        let path = checkpoint_descriptor.save(
                            checkpoint_output_dir,
                            &*backend,
                            &mut params,
                            &mut opt_state,
                            ema_ref_state.as_ref(),
                            &mut loop_state,
                            gpu_step_coordination.as_ref(),
                            &mut gpu_writer_timings,
                            "cancellation checkpoint device snapshot",
                        )?;
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved exact GRPO checkpoint before cancellation"
                        );
                    }
                    anyhow::bail!("training cancelled by user (stop requested at step boundary)");
                }
            }

            tracing::info!(
                group = group_idx + 1,
                total_groups = total_steps,
                num_completions,
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                loss = format!("{avg_group_loss:.6}"),
                "GRPO group step"
            );
            if let Some(echo_env_ce) = step_report.echo_env_ce {
                tracing::info!(
                    group = group_idx + 1,
                    total_groups = total_steps,
                    action_tokens = group_counts.action_tokens,
                    env_tokens = group_counts.env_tokens,
                    echo_env_ce,
                    "GRPO ECHO group metrics"
                );
            }

            if let Some(pb) = &pb {
                pb.set_message(format!("{avg_group_loss:.6}"));
                pb.inc(1);
            }
        }

        anyhow::ensure!(
            global_step == total_steps
                && loss_history.len() == total_steps
                && processed_completions == planned_completions
                && token_counts == planned_token_counts,
            "GRPO loop completed with inconsistent progress or diagnostics"
        );

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        // Pull current param values from the registry into kt master
        // storage before final save_peft.
        let synced = run_coordinated_grpo_gpu_phase(
            gpu_step_coordination.as_ref(),
            &*backend,
            &mut gpu_writer_timings,
            "final adapter snapshot",
            || {
                params
                    .sync_to_master(&*backend)
                    .context("capture final GRPO adapter state")
            },
        )?;
        tracing::debug!(
            synced,
            "synced LoRA params to kt master storage before GRPO save"
        );

        // Save the trained adapter
        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            "GRPO training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let mut result = train_body();
    let policy_audit = finish_grpo_policy_audit(&mut result, policy_audit);
    let mut adapter_smoke_test = None;
    let cleanup_result = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "adapter smoke test and cleanup",
        || {
            if config.adapter_smoke_test && result.is_ok() {
                adapter_smoke_test = Some(run_adapter_smoke_test_best_effort(
                    adapter_name,
                    &*backend,
                    weights,
                    model_config,
                    tokenizer,
                    &params,
                    config.adapter_smoke_prompts.as_deref(),
                    streaming_prefill,
                ));
            }
            // Registry eviction is backend mutation too; keep it within the
            // final bounded phase even after a failed training step.
            if let Some(state) = opt_state.as_ref() {
                state.evict_from_backend(&*backend);
            }
            params.evict_from_backend(&*backend);
            Ok(())
        },
    );
    if let Err(error) = cleanup_result {
        if result.is_ok() {
            result = Err(error.context("complete coordinated GRPO cleanup"));
        } else {
            tracing::warn!(error = %format!("{error:#}"), "GRPO cleanup could not acquire healthy backend");
        }
    }
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append GRPO replay outcome record");
        }
    }
    gpu_writer_timings.apply_to(&mut phase_timings);
    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    write_grpo_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        weights.base_weight_shard_manifest.as_ref(),
        weights.execution_provenance.as_ref(),
        training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
        config,
        effective_seed,
        Some(alpha_over_rank),
        base_adapter_dir.as_deref(),
        &output_dir,
        crate::train_receipt::TrainingDataReceipt {
            source: "inline_grpo_groups".to_string(),
            path: None,
            sha256: training_data_sha256,
            openenv: openenv_training_data,
        },
        data_stats,
        reward_stats,
        token_counts,
        phase_timings.to_receipt(),
        echo_metrics,
        run_started.elapsed().as_millis() as u64,
        dynamic_groups_filtered,
        adapter_smoke_test,
        lora_grad_norms.finish(),
        policy_audit,
        status_error,
    );
    result
        .map(|(dir, _)| dir)
        .map_err(crate::train_receipt::annotate_training_error)
}

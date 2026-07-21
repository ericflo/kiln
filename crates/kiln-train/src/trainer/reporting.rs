use super::*;

pub(super) fn sft_hyperparameters(
    config: &SftConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
) -> crate::train_receipt::HyperparameterReceipt {
    crate::train_receipt::HyperparameterReceipt {
        mode: "sft".to_string(),
        rank: config.lora_rank,
        alpha: config.lora_alpha,
        alpha_over_rank,
        // Receipts record the RESOLVED learning rate — the value the
        // optimizer actually stepped with — not the Option.
        learning_rate: config.effective_learning_rate(),
        epochs: config.epochs,
        seed: effective_seed,
        shuffle: true,
    }
}

pub(super) fn grpo_hyperparameters(
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
) -> crate::train_receipt::HyperparameterReceipt {
    crate::train_receipt::HyperparameterReceipt {
        mode: "grpo".to_string(),
        rank: config.lora_rank,
        alpha: config.lora_alpha,
        alpha_over_rank,
        // Resolved value, as in `sft_hyperparameters`.
        learning_rate: config.effective_learning_rate(),
        epochs: 1,
        seed: effective_seed,
        shuffle: false,
    }
}

pub(super) fn grpo_echo_receipt(config: &GrpoConfig) -> crate::train_receipt::EchoReceipt {
    match config.loss.echo.as_ref() {
        // The env-CE term is live again (resurrection PR2 #1512): `enabled`
        // records whether the term is armed for this run (λ≠0). Whether it
        // actually FIRED shows in initial/final_env_ce, filled in by the
        // EchoActivityMetrics from per-step observations — a run whose
        // rollouts carry no env tokens keeps env_ce: None and the standing
        // warn_echo_enabled_without_env_tokens diagnostic.
        Some(echo) => crate::train_receipt::EchoReceipt {
            enabled: config.loss.echo_enabled(),
            lambda: Some(echo.lambda),
            env_mask_mode: serde_json::to_value(echo.env_mask_mode)
                .ok()
                .and_then(|v| v.as_str().map(ToString::to_string)),
            warning_filter: Some(echo.warning_filter),
            initial_env_ce: None,
            final_env_ce: None,
            dropped_reason: None,
        },
        None => crate::train_receipt::EchoReceipt::disabled(),
    }
}

pub(super) fn grpo_settings_receipt(
    config: &GrpoConfig,
    dynamic_groups_filtered: usize,
) -> crate::train_receipt::GrpoReceipt {
    crate::train_receipt::GrpoReceipt {
        kl_coeff: config.kl_coeff,
        clip_epsilon: config.clip_epsilon,
        clip_eps_high: config.clip_eps_high,
        cispo_max_weight: config.cispo_max_weight,
        dynamic_sampling: config.dynamic_sampling,
        dynamic_groups_filtered,
        advantage_mode: serde_json::to_value(config.advantage_mode)
            .unwrap_or(serde_json::Value::Null),
        loss_aggregation: serde_json::to_value(config.loss_aggregation)
            .unwrap_or(serde_json::Value::Null),
        kl_estimator: serde_json::to_value(config.kl_estimator).unwrap_or(serde_json::Value::Null),
        is_level: serde_json::to_value(config.is_level).unwrap_or(serde_json::Value::Null),
        behavior_policy: serde_json::to_value(config.behavior_policy)
            .unwrap_or(serde_json::Value::Null),
        kl_reference_policy: serde_json::to_value(&config.kl_reference_policy)
            .unwrap_or(serde_json::Value::Null),
        entropy_aware_kl_quantile: config.entropy_aware_kl_quantile,
        policy_audit: None,
    }
}

#[derive(Debug, Clone)]
pub(super) struct RewardFilterInputGroup {
    pub(super) id: String,
    pub(super) source_index: usize,
    pub(super) source_line: Option<usize>,
    pub(super) reward_variance: f64,
}

#[derive(Debug, Clone)]
pub(super) struct RewardFilterPlan {
    pub(super) kept_source_indices: Vec<usize>,
    pub(super) kept_source_lines: Vec<usize>,
    pub(super) skip_training: bool,
    pub(super) failure_reason: Option<String>,
    pub(super) sidecar_path: PathBuf,
    pub(super) groups_kept: usize,
    pub(super) groups_dropped: usize,
}

impl RewardFilterPlan {
    pub(super) fn keeps_source_index(&self, source_index: usize) -> bool {
        self.kept_source_indices
            .binary_search(&source_index)
            .is_ok()
    }

    pub(super) fn keeps_source_line(&self, line_no: usize) -> bool {
        self.kept_source_lines.binary_search(&line_no).is_ok()
    }
}

pub(super) fn reward_filter_enabled(config: &GrpoConfig) -> bool {
    config.reward_filter_var_min.is_some() || config.reward_filter_var_max.is_some()
}

pub(super) fn validate_reward_filter_config(config: &GrpoConfig) -> Result<()> {
    if let Some(var_min) = config.reward_filter_var_min {
        anyhow::ensure!(
            var_min.is_finite() && var_min >= 0.0,
            "reward filter --filter-var-min must be a finite non-negative float"
        );
    }
    if let Some(var_max) = config.reward_filter_var_max {
        anyhow::ensure!(
            var_max.is_finite() && var_max >= 0.0,
            "reward filter --filter-var-max must be a finite non-negative float"
        );
    }
    if let (Some(var_min), Some(var_max)) =
        (config.reward_filter_var_min, config.reward_filter_var_max)
    {
        anyhow::ensure!(
            var_min <= var_max,
            "reward filter --filter-var-min must be <= --filter-var-max"
        );
    }
    anyhow::ensure!(
        config.reward_filter_min_groups > 0,
        "reward filter --min-groups must be at least 1"
    );
    Ok(())
}

pub(super) fn reward_filter_group_matches(
    variance: f64,
    var_min: Option<f64>,
    var_max: Option<f64>,
) -> (bool, Option<String>) {
    if let Some(min) = var_min {
        if variance < min {
            return (false, Some(format!("variance_below_min:{min}")));
        }
    }
    if let Some(max) = var_max {
        if variance > max {
            return (false, Some(format!("variance_above_max:{max}")));
        }
    }
    (true, None)
}

pub(super) fn reward_filter_variance(rewards: &[f64]) -> f64 {
    if rewards.is_empty() {
        return 0.0;
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    rewards
        .iter()
        .map(|reward| {
            let centered = *reward - mean;
            centered * centered
        })
        .sum::<f64>()
        / rewards.len() as f64
}

#[derive(Debug)]
pub(super) struct StreamedRewardStatsAccumulator {
    pub(super) count: usize,
    pub(super) mean: f64,
    pub(super) sum_squared_deviation: f64,
    pub(super) min: f64,
    pub(super) max: f64,
    pub(super) group_count: usize,
    pub(super) all_pass_group_count: usize,
    pub(super) all_fail_group_count: usize,
    pub(super) degenerate_group_count: usize,
    pub(super) variance_histogram_counts: [usize; 6],
}

impl Default for StreamedRewardStatsAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            sum_squared_deviation: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            group_count: 0,
            all_pass_group_count: 0,
            all_fail_group_count: 0,
            degenerate_group_count: 0,
            variance_histogram_counts: [0; 6],
        }
    }
}

impl StreamedRewardStatsAccumulator {
    pub(super) fn observe_group<'a, I>(&mut self, rewards: I, all_pass_threshold: f64) -> f64
    where
        I: IntoIterator<Item = &'a f64>,
    {
        let mut group_count = 0usize;
        let mut group_mean = 0.0;
        let mut group_squared_deviation = 0.0;
        let mut all_pass = true;
        let mut all_fail = true;
        for reward in rewards {
            let reward = *reward;
            group_count += 1;
            let group_delta = reward - group_mean;
            group_mean += group_delta / group_count as f64;
            group_squared_deviation += group_delta * (reward - group_mean);

            self.count += 1;
            let delta = reward - self.mean;
            self.mean += delta / self.count as f64;
            self.sum_squared_deviation += delta * (reward - self.mean);
            self.min = self.min.min(reward);
            self.max = self.max.max(reward);
            all_pass &= reward >= all_pass_threshold;
            all_fail &= reward <= 0.0;
        }
        if group_count == 0 {
            return 0.0;
        }
        self.group_count += 1;
        self.all_pass_group_count += if all_pass { 1 } else { 0 };
        self.all_fail_group_count += if all_fail { 1 } else { 0 };
        let variance = (group_squared_deviation / group_count as f64).max(0.0);
        if variance <= crate::train_receipt::REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON {
            self.degenerate_group_count += 1;
        }
        let bucket = if variance == 0.0 {
            0
        } else if variance <= 1e-6 {
            1
        } else if variance <= 0.01 {
            2
        } else if variance <= 0.25 {
            3
        } else if variance <= 1.0 {
            4
        } else {
            5
        };
        self.variance_histogram_counts[bucket] += 1;
        variance
    }

    pub(super) fn finish(self) -> crate::train_receipt::RewardStatsReceipt {
        if self.count == 0 {
            return crate::train_receipt::RewardStatsReceipt::default();
        }
        let specs = [
            ("zero", Some(0.0), Some(0.0)),
            ("tiny", Some(f64::MIN_POSITIVE), Some(1e-6)),
            ("low", Some(1e-6), Some(0.01)),
            ("medium", Some(0.01), Some(0.25)),
            ("high", Some(0.25), Some(1.0)),
            ("extreme", Some(1.0), None),
        ];
        crate::train_receipt::RewardStatsReceipt {
            count: self.count,
            mean: Some(self.mean),
            stdev: Some(
                (self.sum_squared_deviation / self.count as f64)
                    .max(0.0)
                    .sqrt(),
            ),
            min: Some(self.min),
            max: Some(self.max),
            group_count: self.group_count,
            all_pass_group_count: self.all_pass_group_count,
            all_fail_group_count: self.all_fail_group_count,
            degenerate_group_count: self.degenerate_group_count,
            group_variance_histogram: specs
                .into_iter()
                .zip(self.variance_histogram_counts)
                .map(|((label, min_inclusive, max_inclusive), count)| {
                    crate::train_receipt::HistogramBucket {
                        label: label.to_string(),
                        min_inclusive,
                        max_inclusive,
                        count,
                    }
                })
                .collect(),
        }
    }
}

pub(super) fn reward_filter_on_empty_label(mode: RewardFilterOnEmpty) -> &'static str {
    match mode {
        RewardFilterOnEmpty::Fail => "fail",
        RewardFilterOnEmpty::TrainAll => "train-all",
        RewardFilterOnEmpty::Skip => "skip",
    }
}

pub(super) fn build_reward_filter_plan(
    config: &GrpoConfig,
    output_dir: &Path,
    source: &str,
    groups: Vec<RewardFilterInputGroup>,
) -> Result<Option<RewardFilterPlan>> {
    if !reward_filter_enabled(config) {
        return Ok(None);
    }
    validate_reward_filter_config(config)?;

    let mut candidate_kept_count = 0usize;
    let mut decisions = Vec::new();
    for group in &groups {
        let variance = group.reward_variance;
        let (matched_filter, reject_reason) = reward_filter_group_matches(
            variance,
            config.reward_filter_var_min,
            config.reward_filter_var_max,
        );
        if matched_filter {
            candidate_kept_count = candidate_kept_count.saturating_add(1);
        }
        decisions.push(crate::train_receipt::RewardFilterGroupDecisionReceipt {
            id: group.id.clone(),
            source_index: group.source_index,
            source_line: group.source_line,
            reward_variance: variance,
            matched_filter,
            kept: matched_filter,
            reject_reason,
        });
    }

    let empty_filter_triggered = candidate_kept_count < config.reward_filter_min_groups;
    let on_empty = config.reward_filter_on_empty;
    let empty_filter_action = if empty_filter_triggered {
        reward_filter_on_empty_label(on_empty)
    } else {
        "use-filter"
    };

    let mut kept_ids = Vec::new();
    let mut dropped_ids = Vec::new();
    let mut kept_indices = Vec::new();
    let mut kept_lines = Vec::new();
    let mut skip_training = false;
    let mut failure_reason = None;

    if empty_filter_triggered {
        match on_empty {
            RewardFilterOnEmpty::Fail => {
                dropped_ids = groups.iter().map(|group| group.id.clone()).collect();
                for decision in &mut decisions {
                    decision.kept = false;
                    decision
                        .reject_reason
                        .get_or_insert_with(|| "below_min_groups".to_string());
                }
                failure_reason = Some(format!(
                    "reward variance filter kept {} group(s), below --min-groups {}; --on-empty-filter=fail",
                    candidate_kept_count, config.reward_filter_min_groups
                ));
            }
            RewardFilterOnEmpty::TrainAll => {
                kept_ids = groups.iter().map(|group| group.id.clone()).collect();
                for group in &groups {
                    kept_indices.push(group.source_index);
                    if let Some(line) = group.source_line {
                        kept_lines.push(line);
                    }
                }
                for decision in &mut decisions {
                    decision.kept = true;
                    decision.reject_reason = None;
                }
            }
            RewardFilterOnEmpty::Skip => {
                skip_training = true;
                dropped_ids = groups.iter().map(|group| group.id.clone()).collect();
                for decision in &mut decisions {
                    decision.kept = false;
                    decision
                        .reject_reason
                        .get_or_insert_with(|| "below_min_groups".to_string());
                }
            }
        }
    } else {
        for (group, decision) in groups.iter().zip(&decisions) {
            if decision.matched_filter {
                kept_ids.push(group.id.clone());
                kept_indices.push(group.source_index);
                if let Some(line) = group.source_line {
                    kept_lines.push(line);
                }
            } else {
                dropped_ids.push(group.id.clone());
            }
        }
    }

    let sidecar = crate::train_receipt::RewardFilterSidecar {
        schema_version: 1,
        sidecar_type: "kiln_reward_filter_groups".to_string(),
        source: source.to_string(),
        var_min: config.reward_filter_var_min,
        var_max: config.reward_filter_var_max,
        min_groups: config.reward_filter_min_groups,
        on_empty_filter: reward_filter_on_empty_label(on_empty).to_string(),
        empty_filter_triggered,
        empty_filter_action: empty_filter_action.to_string(),
        groups_read: groups.len(),
        groups_kept: kept_ids.len(),
        groups_dropped: dropped_ids.len(),
        kept_group_ids: kept_ids,
        dropped_group_ids: dropped_ids,
        groups: decisions,
    };
    let sidecar_path = crate::train_receipt::write_reward_filter_sidecar(output_dir, &sidecar)?;
    kept_indices.sort_unstable();
    kept_indices.dedup();
    kept_lines.sort_unstable();
    kept_lines.dedup();
    Ok(Some(RewardFilterPlan {
        groups_kept: sidecar.groups_kept,
        groups_dropped: sidecar.groups_dropped,
        kept_source_indices: kept_indices,
        kept_source_lines: kept_lines,
        skip_training,
        failure_reason,
        sidecar_path,
    }))
}

pub(super) fn record_reward_filter_plan(
    data_stats: &mut crate::train_receipt::DataStatsReceipt,
    plan: &RewardFilterPlan,
) {
    data_stats.reward_groups_kept = plan.groups_kept;
    data_stats.reward_groups_filtered = plan.groups_dropped;
    data_stats.reward_filter_sidecar = Some(plan.sidecar_path.display().to_string());
}

pub(super) fn run_adapter_smoke_test_best_effort(
    adapter_name: &str,
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    params: &TrainableLoraParams,
    configured_prompts: Option<&[String]>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> crate::train_receipt::AdapterSmokeTestReceipt {
    let receipt = run_adapter_smoke_test(
        backend,
        weights,
        model_config,
        tokenizer,
        params,
        configured_prompts,
        streaming_prefill,
    )
    .unwrap_or_else(|err| {
        crate::train_receipt::failed_adapter_smoke_test_receipt(format!("{err:#}"))
    });
    if receipt.passed {
        tracing::info!(
            adapter = adapter_name,
            prompts = receipt.prompts.len(),
            "adapter smoke test passed"
        );
    } else {
        for warning in &receipt.warnings {
            tracing::warn!(
                adapter = adapter_name,
                warning = %warning,
                "adapter smoke test warning"
            );
        }
    }
    receipt
}

pub(super) fn run_adapter_smoke_test(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    params: &TrainableLoraParams,
    configured_prompts: Option<&[String]>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<crate::train_receipt::AdapterSmokeTestReceipt> {
    let lora = lora_weights_detached(params);
    let smoke_prompts = adapter_smoke_test_prompts(configured_prompts)?;
    let mut prompts = Vec::with_capacity(smoke_prompts.len());

    for prompt in &smoke_prompts {
        let prompt_ids = tokenizer
            .encode(prompt)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("tokenize adapter smoke prompt {prompt:?}"))?;
        anyhow::ensure!(
            !prompt_ids.is_empty(),
            "adapter smoke prompt tokenized to zero tokens: {prompt:?}"
        );

        let base_logits = adapter_smoke_forward_logits(
            backend,
            &prompt_ids,
            weights,
            model_config,
            None,
            streaming_prefill,
        )
        .with_context(|| format!("base forward for adapter smoke prompt {prompt:?}"))?;
        let adapter_logits = adapter_smoke_forward_logits(
            backend,
            &prompt_ids,
            weights,
            model_config,
            Some(&lora),
            streaming_prefill,
        )
        .with_context(|| format!("adapter forward for adapter smoke prompt {prompt:?}"))?;
        let (finite_logits, logit_delta_l2) =
            adapter_smoke_logit_delta_l2(&base_logits, &adapter_logits)
                .with_context(|| format!("compare adapter smoke logits for {prompt:?}"))?;

        let base_generation = adapter_smoke_greedy_generate(
            backend,
            weights,
            model_config,
            tokenizer,
            prompt,
            None,
            streaming_prefill,
        )
        .with_context(|| format!("base generation for adapter smoke prompt {prompt:?}"))?;
        let adapter_generation = adapter_smoke_greedy_generate(
            backend,
            weights,
            model_config,
            tokenizer,
            prompt,
            Some(&lora),
            streaming_prefill,
        )
        .with_context(|| format!("adapter generation for adapter smoke prompt {prompt:?}"))?;

        prompts.push(crate::train_receipt::AdapterSmokePromptReceipt {
            prompt: prompt.to_string(),
            finite_logits,
            logit_delta_l2,
            generated_text_different: base_generation.output != adapter_generation.output,
            base_output: base_generation.output,
            adapter_output_chars: adapter_generation.output.chars().count(),
            adapter_output: adapter_generation.output,
            adapter_output_tokens: adapter_generation.output_tokens,
            base_generation_ms: base_generation.elapsed_ms,
            adapter_generation_ms: adapter_generation.elapsed_ms,
        });
    }

    Ok(crate::train_receipt::build_adapter_smoke_test_receipt(
        prompts,
    ))
}

pub(super) fn adapter_smoke_test_prompts(configured: Option<&[String]>) -> Result<Vec<String>> {
    let prompts = match configured {
        Some(prompts) => prompts.to_vec(),
        None => ADAPTER_SMOKE_TEST_PROMPTS
            .iter()
            .map(|prompt| (*prompt).to_string())
            .collect(),
    };
    anyhow::ensure!(
        !prompts.is_empty(),
        "adapter_smoke_prompts must contain at least one prompt"
    );
    for (index, prompt) in prompts.iter().enumerate() {
        anyhow::ensure!(
            !prompt.trim().is_empty(),
            "adapter_smoke_prompts[{index}] must not be blank"
        );
    }
    Ok(prompts)
}

pub(super) fn adapter_smoke_linear_state(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
) -> Result<LinearAttentionState> {
    // (#1082) `Tensor::device()` returns an owned kt `Device` (Copy); the
    // constructor wants `&Device`, so bind to a local and borrow.
    let kt_device = weights.embed_tokens.device();
    LinearAttentionState::new_with_batch_for_inference_runtime(model_config, 1, &kt_device, backend)
}

pub(super) fn adapter_smoke_forward_logits(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    let mut linear_state = adapter_smoke_linear_state(backend, weights, model_config)?;
    model_forward_kt_with_policy(
        backend,
        token_ids,
        weights,
        model_config,
        None,
        Some(&mut linear_state),
        lora,
        streaming_prefill,
    )
}

pub(super) fn adapter_smoke_logit_delta_l2(
    base_logits: &Tensor,
    adapter_logits: &Tensor,
) -> Result<(bool, Option<f64>)> {
    let base = adapter_smoke_last_logits(base_logits)?;
    let adapter = adapter_smoke_last_logits(adapter_logits)?;
    anyhow::ensure!(
        base.len() == adapter.len(),
        "base and adapter logits have different vocab sizes: {} vs {}",
        base.len(),
        adapter.len()
    );

    let finite_logits = base
        .iter()
        .chain(adapter.iter())
        .all(|value| value.is_finite());
    if !finite_logits {
        return Ok((false, None));
    }

    let sum_sq = base
        .iter()
        .zip(adapter.iter())
        .map(|(base, adapter)| {
            let delta = *adapter as f64 - *base as f64;
            delta * delta
        })
        .sum::<f64>();
    let l2 = sum_sq.sqrt();
    Ok((l2.is_finite(), l2.is_finite().then_some(l2)))
}

pub(super) fn adapter_smoke_last_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() >= 2,
        "adapter smoke logits must have at least 2 dimensions, got {dims:?}"
    );
    let seq_dim = dims.len() - 2;
    let seq_len = dims[seq_dim];
    anyhow::ensure!(
        seq_len > 0,
        "adapter smoke logits have zero sequence length"
    );
    Ok(logits
        .narrow(seq_dim, seq_len - 1, 1)?
        .squeeze(seq_dim)?
        .flatten_all()?
        .to_f32_dtype()?
        .to_vec1::<f32>()?)
}

pub(super) fn adapter_smoke_greedy_generate(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt: &str,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<AdapterSmokeGeneration> {
    let mut context = tokenizer
        .encode(prompt)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .with_context(|| format!("tokenize adapter smoke generation prompt {prompt:?}"))?;
    anyhow::ensure!(
        !context.is_empty(),
        "adapter smoke generation prompt tokenized to zero tokens: {prompt:?}"
    );

    let started = Instant::now();
    let mut generated = Vec::with_capacity(ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS);
    for _ in 0..ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS {
        let logits = adapter_smoke_forward_logits(
            backend,
            &context,
            weights,
            model_config,
            lora,
            streaming_prefill,
        )?;
        let token = greedy_sample(&logits)?;
        generated.push(token);
        context.push(token);
    }

    let output = tokenizer
        .decode(&generated)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .context("decode adapter smoke generated tokens")?;
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    Ok(AdapterSmokeGeneration {
        output,
        output_tokens: generated.len(),
        elapsed_ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_sft_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    sft_loss_route: SftFlceLossRoute,
    config: &SftConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data_sha256: Option<String>,
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    wall_clock_ms: u64,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    status_error: Option<String>,
) {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "sft",
        model_config,
        tokenizer,
        sft_hyperparameters(config, effective_seed, alpha_over_rank),
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.model.base_weight_shard_manifest = base_weight_shard_manifest.cloned();
    receipt.runtime.execution_provenance = execution_provenance.cloned();
    receipt.runtime.training_precision = training_precision;
    receipt.runtime.sft_loss_route = Some(sft_loss_route);
    receipt.training_data = crate::train_receipt::TrainingDataReceipt {
        source: ingestion.source.clone(),
        path: ingestion.source_locator.clone(),
        sha256: training_data_sha256,
    };
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    receipt.data = data;
    receipt.token_counts = token_counts;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    receipt.lora_grad_norms = lora_grad_norms;
    receipt.adapter_smoke_test = adapter_smoke_test;
    crate::train_receipt::log_training_token_counts("sft", &receipt.token_counts);
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_else(|err| {
                tracing::warn!(adapter = adapter_name, error = %err, "failed to summarize LoRA delta norms for train receipt");
                Vec::new()
            });
        crate::train_receipt::warn_lora_delta_norms(
            "sft",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write SFT train receipt");
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_grpo_train_receipt(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data: crate::train_receipt::TrainingDataReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    rewards: crate::train_receipt::RewardStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    phase_timings: crate::train_receipt::TrainingPhaseTimingsReceipt,
    echo_metrics: crate::train_receipt::EchoActivityMetrics,
    wall_clock_ms: u64,
    dynamic_groups_filtered: usize,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    status_error: Option<String>,
) -> crate::train_receipt::TrainReceipt {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "grpo",
        model_config,
        tokenizer,
        grpo_hyperparameters(config, effective_seed, alpha_over_rank),
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.model.base_weight_shard_manifest = base_weight_shard_manifest.cloned();
    receipt.runtime.execution_provenance = execution_provenance.cloned();
    receipt.runtime.training_precision = training_precision;
    receipt.training_data = training_data;
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    let mut grpo = grpo_settings_receipt(config, dynamic_groups_filtered);
    grpo.policy_audit = policy_audit;
    receipt.grpo = Some(grpo);
    receipt.echo = grpo_echo_receipt(config);
    echo_metrics.apply_to_echo_receipt(&mut receipt.echo);
    receipt.no_policy_loss = config.loss.no_policy_loss;
    receipt.data = data;
    receipt.rewards = rewards;
    receipt.token_counts = token_counts;
    receipt.phase_timings = phase_timings;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    receipt.adapter_smoke_test = adapter_smoke_test;
    receipt.lora_grad_norms = lora_grad_norms;
    crate::train_receipt::log_training_token_counts("grpo", &receipt.token_counts);
    crate::train_receipt::warn_echo_enabled_without_env_tokens(
        "grpo",
        config.loss.echo_enabled(),
        &receipt.token_counts,
    );
    crate::train_receipt::warn_reward_diagnostics(
        "grpo",
        adapter_name,
        &receipt.rewards,
        config.reward_saturation_threshold,
        config.reward_low_variance_threshold,
    );
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_else(|err| {
                tracing::warn!(adapter = adapter_name, error = %err, "failed to summarize LoRA delta norms for train receipt");
                Vec::new()
            });
        crate::train_receipt::warn_lora_delta_norms(
            "grpo",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    receipt
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_grpo_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data: crate::train_receipt::TrainingDataReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    rewards: crate::train_receipt::RewardStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    phase_timings: crate::train_receipt::TrainingPhaseTimingsReceipt,
    echo_metrics: crate::train_receipt::EchoActivityMetrics,
    wall_clock_ms: u64,
    dynamic_groups_filtered: usize,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    status_error: Option<String>,
) {
    if let Some(audit) = policy_audit.as_ref() {
        tracing::info!(
            schema = %audit.schema,
            ratio_scope = audit
                .importance_sampling
                .ratio_scope
                .as_deref()
                .unwrap_or("none"),
            action_tokens = audit.importance_sampling.action_tokens,
            ratio_observations = audit.importance_sampling.ratio_observations,
            mean_ratio = ?audit.importance_sampling.mean_ratio,
            outside_clip_fraction = ?audit.importance_sampling.outside_clip_fraction,
            kl_tokens = audit.kl_reference.token_observations,
            mean_kl_estimator = ?audit.kl_reference.mean_estimator,
            mean_masked_kl_estimator = ?audit.kl_reference.mean_masked_estimator,
            recorded_completions = audit.recorded_provenance.completion_count,
            behavior_sources = audit.recorded_provenance.unique_behavior_sources,
            behavior_source_manifest_sha256 = audit
                .recorded_provenance
                .behavior_source_manifest_sha256
                .as_deref()
                .unwrap_or("none"),
            "GRPO policy audit"
        );
    }
    let receipt = build_grpo_train_receipt(
        adapter_name,
        model_config,
        tokenizer,
        base_weight_shard_manifest,
        execution_provenance,
        training_precision,
        config,
        effective_seed,
        alpha_over_rank,
        base_adapter_dir,
        output_dir,
        training_data,
        data,
        rewards,
        token_counts,
        phase_timings,
        echo_metrics,
        wall_clock_ms,
        dynamic_groups_filtered,
        adapter_smoke_test,
        lora_grad_norms,
        policy_audit,
        status_error,
    );
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write GRPO train receipt");
    }
}

pub(super) fn finish_grpo_policy_audit<T>(
    training_result: &mut Result<T>,
    accumulator: crate::train_receipt::GrpoPolicyAuditAccumulator,
) -> Option<crate::train_receipt::GrpoPolicyAuditReceipt> {
    match accumulator.finish().context("finalize GRPO policy audit") {
        Ok(receipt) => Some(receipt),
        Err(error) => {
            if training_result.is_ok() {
                *training_result = Err(error);
            } else {
                tracing::warn!(error = %error, "failed to finalize partial GRPO policy audit");
            }
            None
        }
    }
}

/// Run SFT training on the provided examples using the already-loaded model.
///
/// This runs in the calling thread (blocking). The caller should spawn this
/// on a background thread to avoid blocking inference.
///
/// When `replay_ctx` is `Some`, the trainer writes a `replay.jsonl` request
/// record (with the resolved seed) and `lineage.json` into the adapter
/// directory *before* the optimizer step, then appends an outcome record
/// when training completes or fails. When `None`, no replay artifacts are
/// written — used by tests and benches that don't need replay.
///
/// Returns the path to the saved adapter directory.
/// Post-SFT MTP alignment phase (MTP training plan PR-B).
///
/// Trains LoRA on the native MTP draft block so the draft keeps
/// predicting what the freshly-tuned model would say — every LoRA step
/// moves the served distribution away from the frozen pretrained draft
/// head, so speculative-decode acceptance decays exactly in proportion
/// to personalization unless the draft trains too.
///
/// Per example: one detached no-head forward of the TUNED model gives
/// post-final-norm hiddens h (the same tensor `mtp_forward_step` consumes
/// as `h_prev` at serve time); the MTP block then trains under the kt
/// tape on `fused_t = fc(concat(norm_e(emb(tok_{t+1})), norm_h(h_t)))`
/// with the production FLCE root over the tied head — fed the shifted
/// `ids[1..]` / `mask[1..]`, which makes row t's label `ids[t+2]`: the
/// MTP objective, with zero new loss machinery. Only the seven
/// draft-block LoRA pairs receive gradients (the hiddens are detached;
/// fc / norms / tied head are frozen).
///
/// Returns `(examples_trained, initial_ce, final_ce)`; `None` when the
/// checkpoint has no MTP tensors or the phase is disabled.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_mtp_alignment_phase(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &mut TrainableLoraParams,
    examples: &[SftExample],
    valid_indices: &[usize],
    tokenizer: &KilnTokenizer,
    config: &SftConfig,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Option<(usize, Option<f64>, Option<f64>)>> {
    let enabled = config.train_mtp.unwrap_or(true);
    if !enabled || weights.mtp.is_none() {
        return Ok(None);
    }
    if !params.initialize_mtp_seeded(weights, device, Some(0x4D54_5042))? {
        return Ok(None);
    }
    let mtp = weights
        .mtp_weights()
        .context("mtp alignment: materializing mtp.* tensors")?;
    let mtp_full_attention = matches!(
        mtp.layer.attention,
        kiln_model::forward::GpuAttentionWeights::Full(_)
    );
    anyhow::ensure!(
        mtp_full_attention,
        "mtp alignment: MTP layer must be full attention"
    );

    // Move the draft-block pairs into a one-layer TrainableLoraParams so
    // the existing optimizer machinery (make_opt_state /
    // optimizer_step_dispatch, both keyed on `all_params`) drives EXACTLY
    // these seven pairs. Ownership moves — no clones — so Parameter
    // identity (tensor_id, registry residency) is preserved; the pairs
    // move back into `params.mtp` at the end for save_peft.
    let taken = params
        .mtp
        .take()
        .expect("initialize_mtp_seeded just populated params.mtp");
    let mut mtp_train = TrainableLoraParams {
        layers: vec![taken],
        mtp: None,
        rank: params.rank,
        alpha: params.alpha,
        scale: params.scale,
    };
    mtp_train
        .register_with_backend(backend)
        .context("mtp alignment: registering draft-block LoRA params with resident backend")?;

    // Serving view of the trained MAIN adapter (applied to the hiddens
    // forward) and the draft-block LoRA view (applied inside the block;
    // shares the SAME kt tensors the optimizer updates).
    let lora_view = params.as_lora_weights();
    let mtp_lora_view = {
        let mut v = mtp_train.as_lora_weights();
        v.layers.remove(0)
    };
    let lora_scale = params.scale;

    let learning_rate = config.effective_learning_rate();
    let mut opt_state = make_opt_state(&mtp_train, config.optimizer, learning_rate, device)?;
    if let Some(state) = opt_state.as_ref() {
        state.register_with_backend(backend)?;
    }
    let mut initial_ce: Option<f64> = None;
    let mut final_ce: Option<f64> = None;
    let mut trained = 0usize;
    let phase_started = Instant::now();

    for &idx in valid_indices {
        let ex = &examples[idx];
        let (input_ids, label_mask) = match tokenize_for_training(ex, tokenizer) {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        let seq_len = input_ids.len();
        // The +2 objective needs at least one supervised label at t+2.
        if seq_len < 3 || !label_mask.get(2..).is_some_and(|m| m.iter().any(|&v| v)) {
            continue;
        }

        // 1) Detached hiddens from the TUNED model — outside any tape scope.
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        let hidden = model_forward_no_head_with_policy(
            backend,
            &input_ids,
            weights,
            model_config,
            Some(&mut linear_state),
            Some(&lora_view),
            streaming_prefill,
        )
        .context("mtp alignment: no-head hiddens forward")?
        .detach();

        // 2) MTP block forward + FLCE root under the tape-authoritative scope.
        let shifted_ids: Vec<u32> = input_ids[1..].to_vec();
        let shifted_mask: Vec<bool> = label_mask[1..].to_vec();
        let positions: Vec<u32> = (0..seq_len as u32 - 1).collect();
        let result = kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions {
                detect_anomaly: config.detect_anomaly,
            },
            || {
                let to_err = |e: anyhow::Error| kiln_kt_bridge::BridgeError::new(format!("{e:#}"));
                // emb rows for tok_{1..T} — frozen embedding, plain index_select.
                let idx_t = kiln_tensor::Tensor::from_vec_on(
                    *device,
                    shifted_ids.clone(),
                    vec![seq_len - 1],
                )
                .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: idx tensor: {e}")))?;
                let emb = weights
                    .embed_tokens
                    .index_select(&idx_t, 0)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: emb select: {e}")))?;
                let norm_e = kiln_model::forward::rms_norm(
                    &emb.unsqueeze(0).map_err(|e| {
                        to_err(anyhow::anyhow!("mtp alignment: emb unsqueeze: {e}"))
                    })?,
                    &mtp.pre_fc_norm_embedding,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let h_rows = hidden
                    .narrow(1, 0, seq_len - 1)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: hidden narrow: {e}")))?;
                let norm_h = kiln_model::forward::rms_norm(
                    &h_rows,
                    &mtp.pre_fc_norm_hidden,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let concat = kiln_tensor::ops::concat(&[&norm_e, &norm_h], 2)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: concat: {e}")))?;
                let fc_t = mtp
                    .fc_t
                    .to_dtype(concat.dtype())
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: fc cast: {e}")))?;
                let fused = concat
                    .squeeze(0)
                    .and_then(|c2| c2.matmul(&fc_t))
                    .and_then(|f2| f2.unsqueeze(0))
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: fc matmul: {e}")))?;
                let block_out = kiln_model::forward::transformer_block_with_policy(
                    backend,
                    &fused,
                    &mtp.layer,
                    model_config,
                    &positions,
                    model_config.num_attention_heads,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                    model_config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    model_config.rms_norm_eps,
                    None,
                    0,
                    Some((&mtp_lora_view, lora_scale)),
                    streaming_prefill,
                )
                .map_err(to_err)?;
                let normed = kiln_model::forward::rms_norm(
                    &block_out,
                    &mtp.final_layernorm,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let loss = kiln_autograd::with_active_tape(|tape| {
                    kiln_flce_kernel::fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape(
                        &normed,
                        &weights.embed_tokens_t,
                        &shifted_ids,
                        &shifted_mask,
                        DEFAULT_CHUNK_SIZE,
                        tape,
                    )
                })
                .ok_or_else(|| {
                    kiln_kt_bridge::BridgeError::new("mtp alignment: no active kt tape".to_string())
                })?
                .map_err(|e| {
                    kiln_kt_bridge::BridgeError::new(format!("mtp alignment FLCE: {e}"))
                })?;
                let loss_val = loss
                    .to_dtype(kiln_tensor::DType::F32)
                    .and_then(|t| t.to_scalar::<f32>())
                    .map_err(|e| {
                        kiln_kt_bridge::BridgeError::new(format!("mtp alignment loss read: {e}"))
                    })? as f64;
                Ok((loss_val, loss))
            },
        );
        let (loss_val, _loss_kt, grads_by_candle_raw) = match result {
            Ok(triple) => triple,
            Err(e) => anyhow::bail!("mtp alignment step failed: {e}"),
        };

        let mut grads = kiln_autograd::GradStore::new();
        for (key_raw, kt_grad) in grads_by_candle_raw {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
            else {
                continue;
            };
            grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
        }
        anyhow::ensure!(
            !grads.is_empty(),
            "mtp alignment: tape backward produced no MTP LoRA grads — the draft \
             block's lora-linear did not record (report this; the adapter would \
             silently ship an untrained draft head)"
        );

        optimizer_step_dispatch(
            backend,
            &mut mtp_train,
            &GradSource::Kt(grads),
            learning_rate,
            config.optimizer,
            opt_state.as_mut(),
        )?;

        if initial_ce.is_none() {
            initial_ce = Some(loss_val);
        }
        final_ce = Some(loss_val);
        trained += 1;
    }

    if let Some(state) = opt_state.as_ref() {
        state.evict_from_backend(backend);
    }

    // Return the trained pairs to params.mtp — save_peft serializes them
    // under the mtp.* keys.
    params.mtp = Some(mtp_train.layers.remove(0));

    tracing::info!(
        examples = trained,
        initial_ce = ?initial_ce,
        final_ce = ?final_ce,
        elapsed_ms = phase_started.elapsed().as_millis() as u64,
        "MTP alignment phase complete"
    );
    Ok(Some((trained, initial_ce, final_ce)))
}

/// Resolve the training device against the immutable runtime binding.
/// Production training requires the runtime device to match the resident
/// model-weight device exactly. In particular, CPU-host weights are not
/// promoted into the incomplete hybrid Vulkan training substrate; every
/// device mismatch fails closed before LoRA or optimizer allocation.
pub(super) fn training_device_for_weights(
    weights: &GpuWeights,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<Device> {
    runtime.resolve_device_for_weights(weights.embed_tokens.device())
}

/// Construct the backend named by the immutable training runtime.
///
/// `kiln-model` retains CPU-as-Vulkan autodetection for compatibility
/// inference. Training cannot use that shortcut: an explicit CPU runtime stays
/// CPU, and an accelerated runtime is accepted only when the resident weight
/// device matches it exactly.
pub(crate) fn training_backend_for_device(
    device: Device,
) -> Result<std::sync::Arc<dyn BackendRuntime>> {
    backend::for_explicit_device_kt(device)
        .with_context(|| format!("initialize exact native training backend for {device}"))
}

/// Confirm training will use the already resident serving weights.
///
/// Runtime device resolution rejects mismatches before this point. Keep this
/// second gate at the former upload boundary so a future bypass cannot silently
/// start an unqualified multi-GiB full-model copy.
pub(super) fn resident_training_weights(
    weights: &GpuWeights,
    training_device: &Device,
) -> Result<Option<GpuWeights>> {
    if weights.embed_tokens.device() == *training_device {
        return Ok(None);
    }
    if weights.embed_tokens.device() == Device::Cpu && matches!(training_device, Device::Vulkan(_))
    {
        anyhow::bail!(
            "native Vulkan training is unavailable for CPU-host serving weights: the full-model resident Vulkan training substrate is not production-qualified"
        );
    }
    anyhow::bail!(
        "training device {} does not match resident model weight device {}; full-model training uploads are disabled",
        training_device.short_name(),
        weights.embed_tokens.device().short_name(),
    )
}

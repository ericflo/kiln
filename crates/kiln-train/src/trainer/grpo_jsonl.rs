use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BoundedGrpoJsonlScanStats {
    pub(super) total_bytes: u64,
    pub(super) total_lines: usize,
    pub(super) groups: usize,
    pub(super) completions: usize,
    pub(super) max_row_bytes: u64,
}

pub(super) fn scan_pinned_grpo_jsonl<F>(
    dataset_source: &PinnedGrpoJsonlSource,
    model_num_layers: usize,
    filter_enabled: bool,
    phase: &str,
    mut visit_group: F,
) -> Result<BoundedGrpoJsonlScanStats>
where
    F: FnMut(usize, usize, &GrpoGroup) -> Result<()>,
{
    use std::io::{BufRead as _, BufReader, Read as _};

    let dataset_path = dataset_source.display_path();
    let total_bytes = dataset_source.len()?;
    let file = dataset_source.reader_from_start()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut groups = 0usize;
    let mut completions = 0usize;
    let mut max_row_bytes = 0u64;

    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during {phase}",
                    dataset_path.display(),
                    line_no.saturating_add(1)
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .with_context(|| format!("GRPO JSONL line count overflow during {phase}"))?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit during {phase}",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        max_row_bytes = max_row_bytes.max(line.len() as u64);
        bytes_read = bytes_read
            .checked_add(read as u64)
            .with_context(|| format!("GRPO JSONL byte count overflow during {phase}"))?;
        anyhow::ensure!(
            bytes_read <= total_bytes,
            "GRPO JSONL dataset {} grew while scanning during {phase}",
            dataset_path.display()
        );
        streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            filter_enabled,
        )
        .with_context(|| {
            format!("bound GRPO JSONL host memory before line {line_no} during {phase}")
        })?;

        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        validate_grpo_trajectory_roles(&group, line_no)?;
        anyhow::ensure!(
            !group.completions.is_empty()
                && group.completions.len() <= crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP,
            "GRPO JSONL line {line_no} must contain 1..={} completions",
            crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP
        );
        groups = groups
            .checked_add(1)
            .with_context(|| format!("GRPO JSONL group count overflow during {phase}"))?;
        completions = completions
            .checked_add(group.completions.len())
            .with_context(|| format!("GRPO JSONL completion count overflow during {phase}"))?;
        streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound GRPO JSONL metadata at line {line_no} during {phase}"))?;
        visit_group(line_no, groups, &group)?;
    }

    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset {} changed length during {phase}: expected {total_bytes}, read {bytes_read}",
        dataset_path.display()
    );
    Ok(BoundedGrpoJsonlScanStats {
        total_bytes,
        total_lines: line_no,
        groups,
        completions,
        max_row_bytes,
    })
}

/// First-pass reward metadata for dry-run validation. Global variance is
/// supplied by a second disk pass so the legacy materialized receipt's fold
/// order remains byte-for-byte stable without retaining every reward.
#[derive(Debug)]
pub(super) struct DryRunRewardStatsAccumulator {
    pub(super) count: usize,
    pub(super) sum: f64,
    pub(super) min: f64,
    pub(super) max: f64,
    pub(super) group_count: usize,
    pub(super) all_pass_group_count: usize,
    pub(super) all_fail_group_count: usize,
    pub(super) degenerate_group_count: usize,
    pub(super) variance_histogram_counts: [usize; 6],
}

impl Default for DryRunRewardStatsAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
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

impl DryRunRewardStatsAccumulator {
    pub(super) fn observe_group(
        &mut self,
        group: &GrpoGroup,
        all_pass_threshold: f64,
    ) -> Result<f64> {
        let group_count = group.completions.len();
        anyhow::ensure!(group_count > 0, "GRPO reward group must not be empty");
        let group_mean = group
            .completions
            .iter()
            .map(|completion| completion.reward)
            .sum::<f64>()
            / group_count as f64;
        let group_variance = group
            .completions
            .iter()
            .map(|completion| {
                let centered = completion.reward - group_mean;
                centered * centered
            })
            .sum::<f64>()
            / group_count as f64;

        self.group_count = self
            .group_count
            .checked_add(1)
            .context("GRPO dry-run reward group count overflow")?;
        if group_variance <= crate::train_receipt::REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON {
            self.degenerate_group_count = self
                .degenerate_group_count
                .checked_add(1)
                .context("GRPO dry-run degenerate group count overflow")?;
        }
        if group
            .completions
            .iter()
            .all(|completion| completion.reward >= all_pass_threshold)
        {
            self.all_pass_group_count = self
                .all_pass_group_count
                .checked_add(1)
                .context("GRPO dry-run all-pass group count overflow")?;
        }
        if group
            .completions
            .iter()
            .all(|completion| completion.reward <= 0.0)
        {
            self.all_fail_group_count = self
                .all_fail_group_count
                .checked_add(1)
                .context("GRPO dry-run all-fail group count overflow")?;
        }
        let histogram_bucket = if group_variance == 0.0 {
            Some(0)
        } else if group_variance > f64::MIN_POSITIVE && group_variance <= 1e-6 {
            Some(1)
        } else if group_variance > 1e-6 && group_variance <= 0.01 {
            Some(2)
        } else if group_variance > 0.01 && group_variance <= 0.25 {
            Some(3)
        } else if group_variance > 0.25 && group_variance <= 1.0 {
            Some(4)
        } else if group_variance > 1.0 {
            Some(5)
        } else {
            None
        };
        if let Some(bucket) = histogram_bucket {
            self.variance_histogram_counts[bucket] = self.variance_histogram_counts[bucket]
                .checked_add(1)
                .context("GRPO dry-run reward histogram count overflow")?;
        }
        for completion in &group.completions {
            self.count = self
                .count
                .checked_add(1)
                .context("GRPO dry-run reward count overflow")?;
            self.sum += completion.reward;
            self.min = self.min.min(completion.reward);
            self.max = self.max.max(completion.reward);
        }
        Ok(group_variance)
    }

    pub(super) fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }

    pub(super) fn finish(
        self,
        squared_deviation_sum: f64,
    ) -> crate::train_receipt::RewardStatsReceipt {
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
            mean: Some(self.sum / self.count as f64),
            stdev: Some((squared_deviation_sum / self.count as f64).sqrt()),
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

/// Validate a streamed GRPO JSONL dataset and training configuration without
/// loading model weights or running forward/backward.
pub fn grpo_dry_run_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    allow_empty_after_filter: bool,
) -> Result<GrpoDryRunReport> {
    grpo_dry_run_jsonl_with_pass_hook(
        dataset_path,
        config,
        model_config,
        tokenizer,
        adapter_dir,
        adapter_name,
        allow_empty_after_filter,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn grpo_dry_run_jsonl_with_pass_hook(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    allow_empty_after_filter: bool,
    mut after_first_pass: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<GrpoDryRunReport> {
    let run_started = Instant::now();
    let output_dir = adapter_dir.join(adapter_name);
    let receipt_path = output_dir.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME);
    let mut training_data = crate::train_receipt::TrainingDataReceipt {
        source: "jsonl_grpo_groups_dry_run".to_string(),
        path: Some(dataset_path.display().to_string()),
        sha256: None,
        openenv: None,
    };
    let dataset_source = PinnedGrpoJsonlSource::open(dataset_path);
    let source_sha256 = dataset_source
        .as_ref()
        .map_err(|error| format!("{error:#}"))
        .and_then(|source| source.sha256().map_err(|error| format!("{error:#}")));
    training_data.sha256 = source_sha256.as_ref().ok().cloned();
    let requested_base_adapter_dir = config
        .base_adapter
        .as_deref()
        .map(|name| crate::adapter_shape::resolve_base_adapter_dir(name, adapter_dir));
    let mut data_stats = crate::train_receipt::DataStatsReceipt::default();
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut reward_stats = crate::train_receipt::RewardStatsReceipt::default();
    let mut phase_timings = GrpoBenchmarkTimings::default();
    let mut dynamic_groups_filtered = 0usize;
    let mut alpha_over_rank = None;
    let mut base_adapter_dir = None;

    let result = (|| -> Result<GrpoDryRunReport> {
        config
            .validate_policy_config()
            .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;
        let ratio = crate::lora_scaling::validate_lora_scaling(
            config.lora_rank,
            config.lora_alpha,
            config.allow_high_lora_scale,
        )?;
        alpha_over_rank = Some(ratio);
        base_adapter_dir = resolve_and_validate_base_adapter(
            config.base_adapter.as_deref(),
            adapter_dir,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )?;

        let dataset_source = dataset_source
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?;
        let source_sha256 = source_sha256
            .as_ref()
            .map_err(|error| anyhow::anyhow!("hash GRPO JSONL dataset: {error}"))?;
        let filter_enabled = reward_filter_enabled(config);
        let mut reward_accumulator = DryRunRewardStatsAccumulator::default();
        let mut reward_filter_inputs = Vec::new();
        let mut openenv_accumulator = crate::OpenEnvTrainingDataAccumulator::default();
        let first_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run reward preflight",
            |line_no, source_index, group| {
                openenv_accumulator
                    .observe_group(source_index, group)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| {
                        format!("validate OpenEnv corpus provenance at GRPO JSONL line {line_no}")
                    })?;
                data_stats.groups_read = source_index;
                data_stats.completions_read = data_stats
                    .completions_read
                    .checked_add(group.completions.len())
                    .context("GRPO dry-run completion count overflow")?;
                let reward_variance =
                    reward_accumulator.observe_group(group, config.reward_saturation_threshold)?;
                if filter_enabled {
                    reward_filter_inputs
                        .try_reserve(1)
                        .context("reserve bounded GRPO dry-run reward filter input")?;
                    reward_filter_inputs.push(RewardFilterInputGroup {
                        id: format!("line:{line_no}"),
                        source_index,
                        source_line: Some(line_no),
                        reward_variance,
                    });
                }
                Ok(())
            },
        )?;
        anyhow::ensure!(
            first_scan.groups == data_stats.groups_read
                && first_scan.completions == data_stats.completions_read,
            "GRPO dry-run reward preflight count mismatch"
        );
        training_data.openenv = openenv_accumulator
            .finish()
            .map_err(anyhow::Error::msg)
            .context("finalize GRPO dry-run OpenEnv corpus provenance")?;
        if let Some(hook) = after_first_pass.take() {
            hook()?;
        }
        anyhow::ensure!(
            dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed after dry-run reward preflight"
        );

        let reward_mean = reward_accumulator.mean();
        let mut squared_deviation_sum = 0.0;
        let variance_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run reward variance pass",
            |_line_no, _source_index, group| {
                if let Some(mean) = reward_mean {
                    for completion in &group.completions {
                        let centered = completion.reward - mean;
                        squared_deviation_sum += centered * centered;
                    }
                }
                Ok(())
            },
        )?;
        anyhow::ensure!(
            variance_scan == first_scan && dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed during dry-run reward variance pass"
        );
        reward_stats = reward_accumulator.finish(squared_deviation_sum);
        crate::train_receipt::warn_reward_diagnostics(
            "grpo_dry_run",
            adapter_name,
            &reward_stats,
            config.reward_saturation_threshold,
            config.reward_low_variance_threshold,
        );
        let reward_filter_plan = build_reward_filter_plan(
            config,
            &output_dir,
            "jsonl_grpo_groups_dry_run",
            reward_filter_inputs,
        )?;
        if let Some(plan) = reward_filter_plan.as_ref() {
            record_reward_filter_plan(&mut data_stats, plan);
            data_stats.groups_filtered = data_stats
                .groups_filtered
                .checked_add(plan.groups_dropped)
                .context("GRPO dry-run filtered group count overflow")?;
            tracing::info!(
                kept = plan.groups_kept,
                dropped = plan.groups_dropped,
                sidecar = %plan.sidecar_path.display(),
                "GRPO dry-run reward variance filter applied"
            );
            if let Some(reason) = plan.failure_reason.as_ref() {
                anyhow::bail!("{reason}");
            }
        }

        let mut processed_groups = 0usize;
        let mut processed_completions = 0usize;
        let validation_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run token and mask validation",
            |line_no, _source_index, group| {
                if let Some(plan) = reward_filter_plan.as_ref()
                    && (!plan.keeps_source_line(line_no) || plan.skip_training)
                {
                    return Ok(());
                }
                if config.dynamic_sampling && is_degenerate_grpo_group(group) {
                    dynamic_groups_filtered = dynamic_groups_filtered
                        .checked_add(1)
                        .context("GRPO dry-run dynamic filter count overflow")?;
                    data_stats.groups_filtered = data_stats
                        .groups_filtered
                        .checked_add(1)
                        .context("GRPO dry-run filtered group count overflow")?;
                    return Ok(());
                }

                let group_idx = processed_groups
                    .checked_add(1)
                    .context("GRPO dry-run processed group count overflow")?;
                let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
                let tgroup = tokenize_grpo_group_timed(
                    group,
                    tokenizer,
                    &mask_cfg,
                    Some(&mut phase_timings),
                )
                .with_context(|| {
                    format!("tokenize GRPO dry-run group {group_idx} at line {line_no}")
                })?;
                validate_tokenized_behavior_policy(&tgroup, config.behavior_policy).with_context(
                    || format!("validate GRPO dry-run group {group_idx} behavior provenance"),
                )?;
                validate_grpo_dry_run_masks(&tgroup, group_idx, line_no)?;
                let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tgroup));
                token_counts.add_from(&group_counts);
                processed_groups = group_idx;
                processed_completions = processed_completions
                    .checked_add(tgroup.completions.len())
                    .context("GRPO dry-run processed completion count overflow")?;
                Ok(())
            },
        )?;
        anyhow::ensure!(
            validation_scan == first_scan && dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed during dry-run token validation"
        );

        data_stats.groups_trained = processed_groups;
        data_stats.completions_trained = processed_completions;
        let reward_filter_skipped = reward_filter_plan
            .as_ref()
            .is_some_and(|plan| plan.skip_training);
        if processed_groups == 0 && !allow_empty_after_filter && !reward_filter_skipped {
            anyhow::bail!(
                "GRPO dry run: zero valid GRPO groups after filtering in {}; pass --allow-empty-dry-run to permit this",
                dataset_path.display()
            );
        }
        if processed_groups > 0 {
            anyhow::ensure!(
                processed_completions > 0,
                "GRPO dry run: no valid GRPO completions in {}",
                dataset_path.display()
            );
            anyhow::ensure!(
                token_counts.action_tokens > 0,
                "GRPO dry run: dataset has no action tokens after mask construction"
            );
        }

        Ok(GrpoDryRunReport {
            adapter_dir: output_dir.clone(),
            receipt_path: receipt_path.clone(),
            base_adapter_dir: base_adapter_dir.clone(),
            alpha_over_rank,
            data: data_stats.clone(),
            rewards: reward_stats.clone(),
            token_counts: token_counts.clone(),
            dynamic_groups_filtered,
        })
    })();

    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    let receipt = build_grpo_train_receipt(
        adapter_name,
        model_config,
        tokenizer,
        None,
        None,
        None,
        config,
        config.seed,
        alpha_over_rank,
        base_adapter_dir
            .as_deref()
            .or(requested_base_adapter_dir.as_deref()),
        &output_dir,
        training_data,
        data_stats,
        reward_stats,
        token_counts,
        phase_timings.to_receipt(),
        crate::train_receipt::EchoActivityMetrics::default(),
        run_started.elapsed().as_millis() as u64,
        dynamic_groups_filtered,
        None,
        Vec::new(),
        None,
        status_error,
    );
    let receipt_write = receipt
        .write_to_adapter_dir(&output_dir)
        .with_context(|| format!("write GRPO dry-run receipt {}", receipt_path.display()));

    match (result, receipt_write) {
        (Ok(report), Ok(_)) => Ok(report),
        (Ok(_), Err(err)) => Err(err),
        (Err(err), Ok(_)) => Err(crate::train_receipt::annotate_training_error(err)),
        (Err(err), Err(write_err)) => {
            tracing::warn!(
                adapter = adapter_name,
                error = %write_err,
                "failed to write GRPO dry-run receipt after validation failure"
            );
            Err(crate::train_receipt::annotate_training_error(err))
        }
    }
}

/// Maximum host-memory charge for the streamed GRPO preflight itself. Server
/// admission additionally charges the immutable disk snapshot against its
/// process-wide prepared-data cap.
pub const MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_STREAMED_GRPO_PREFLIGHT_GROUPS: usize = 1_000_000;
pub const MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS: usize = 16_000_000;

/// Conservative host peak for streamed GRPO planning.
///
/// The charge covers the compact trainable entry, reward/filter decisions and
/// sidecar serialization overlap, incremental identity hashing, one row's JSON
/// plus tokenization transients, and one group's checkpoint-boundary scratch.
/// Every operation is checked so adversarial counts fail before allocation.
pub fn streamed_grpo_preflight_host_bytes(
    groups: usize,
    completions: usize,
    max_row_bytes: u64,
    model_num_layers: usize,
    reward_filter_enabled: bool,
) -> Result<u64> {
    const BASE_BYTES: u64 = 256 * 1024;
    const TRAINABLE_PLAN_BYTES_PER_GROUP: u64 = 384;
    const FILTER_AND_SIDECAR_BYTES_PER_GROUP: u64 = 1_536;
    const COMPLETION_DIAGNOSTIC_BYTES: u64 = 8;
    const MAX_ROW_TRANSIENT_MULTIPLIER: u64 = 12;
    const CHECKPOINT_SCRATCH_BYTES_PER_LAYER: u64 = 32;

    anyhow::ensure!(
        groups <= MAX_STREAMED_GRPO_PREFLIGHT_GROUPS,
        "streamed GRPO preflight has {groups} groups; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_GROUPS}"
    );
    anyhow::ensure!(
        completions <= MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS,
        "streamed GRPO preflight has {completions} completions; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS}"
    );
    anyhow::ensure!(
        max_row_bytes <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
        "streamed GRPO preflight row has {max_row_bytes} bytes; maximum is {}",
        MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
    );

    let per_group = TRAINABLE_PLAN_BYTES_PER_GROUP
        .checked_add(if reward_filter_enabled {
            FILTER_AND_SIDECAR_BYTES_PER_GROUP
        } else {
            0
        })
        .context("streamed GRPO per-group preflight charge overflow")?;
    let group_bytes = u64::try_from(groups)
        .context("streamed GRPO group count exceeds u64")?
        .checked_mul(per_group)
        .context("streamed GRPO group-plan charge overflow")?;
    let completion_bytes = u64::try_from(completions)
        .context("streamed GRPO completion count exceeds u64")?
        .checked_mul(COMPLETION_DIAGNOSTIC_BYTES)
        .context("streamed GRPO completion charge overflow")?;
    let row_bytes = max_row_bytes
        .checked_mul(MAX_ROW_TRANSIENT_MULTIPLIER)
        .context("streamed GRPO row-transient charge overflow")?;
    let checkpoint_bytes = u64::try_from(model_num_layers.max(1))
        .context("streamed GRPO model layer count exceeds u64")?
        .checked_mul(CHECKPOINT_SCRATCH_BYTES_PER_LAYER)
        .context("streamed GRPO checkpoint scratch charge overflow")?;
    let total = BASE_BYTES
        .checked_add(group_bytes)
        .and_then(|bytes| bytes.checked_add(completion_bytes))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .and_then(|bytes| bytes.checked_add(checkpoint_bytes))
        .context("streamed GRPO preflight host-memory charge overflow")?;
    anyhow::ensure!(
        total <= MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES,
        "streamed GRPO preflight projects {total} host bytes; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES}"
    );
    Ok(total)
}

/// Disk-backed GRPO source pinned to one open file identity.
///
/// Path-based entry points construct this immediately after opening their
/// input. Server callers can instead pass an already verified handle, so later
/// preflight, resume, epoch, and receipt reads cannot be redirected by an
/// atomic pathname replacement. Reader clones keep the corpus streamed from
/// disk; the source never materializes the whole JSONL in memory.
#[derive(Debug)]
pub struct PinnedGrpoJsonlSource {
    pub(super) file: std::fs::File,
    pub(super) display_path: PathBuf,
    // `File::try_clone` shares the cursor on Unix. The streamed implementation
    // drops each phase reader before rewinding the next one; making this type
    // !Sync prevents concurrent callers from violating that order.
    pub(super) _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

impl PinnedGrpoJsonlSource {
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::OpenOptionsExt as _;

            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)
        };
        #[cfg(not(unix))]
        let file = std::fs::File::open(path);
        let file = file.with_context(|| format!("open GRPO JSONL dataset {}", path.display()))?;
        Self::from_file(file, path.to_path_buf())
    }

    pub fn from_file(file: std::fs::File, display_path: PathBuf) -> Result<Self> {
        let metadata = file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", display_path.display()))?;
        anyhow::ensure!(
            metadata.is_file(),
            "GRPO JSONL dataset {} is not a regular file",
            display_path.display()
        );
        anyhow::ensure!(
            metadata.len() <= crate::HF_TRL_GRPO_MAX_DATASET_BYTES,
            "GRPO JSONL dataset {} has {} bytes; maximum is {}",
            display_path.display(),
            metadata.len(),
            crate::HF_TRL_GRPO_MAX_DATASET_BYTES
        );
        Ok(Self {
            file,
            display_path,
            _not_sync: std::marker::PhantomData,
        })
    }

    pub fn display_path(&self) -> &Path {
        &self.display_path
    }

    pub fn len(&self) -> Result<u64> {
        self.file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", self.display_path.display()))
            .map(|metadata| metadata.len())
    }

    pub fn metadata(&self) -> Result<std::fs::Metadata> {
        self.file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", self.display_path.display()))
    }

    pub fn sha256(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        use std::io::Read as _;

        let mut file = self.reader_from_start()?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} for sha256",
                    self.display_path.display()
                )
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let digest: [u8; 32] = hasher.finalize().into();
        Ok(crate::train_receipt::format_sha256_digest(&digest))
    }

    pub(super) fn reader_from_start(&self) -> Result<std::fs::File> {
        use std::io::{Seek as _, SeekFrom};

        let mut file = self.file.try_clone().with_context(|| {
            format!(
                "clone pinned GRPO JSONL handle {}",
                self.display_path.display()
            )
        })?;
        file.seek(SeekFrom::Start(0)).with_context(|| {
            format!(
                "rewind pinned GRPO JSONL handle {}",
                self.display_path.display()
            )
        })?;
        Ok(file)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GrpoJsonlGradientCheckpointPlan {
    pub(super) config: CheckpointConfig,
    pub(super) boundaries_sha256: String,
}

pub(super) struct Sha256Writer<'a>(&'a mut sha2::Sha256);

impl std::io::Write for Sha256Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        use sha2::Digest as _;
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) struct StreamingJsonArraySha256 {
    pub(super) hasher: sha2::Sha256,
    pub(super) has_items: bool,
}

impl StreamingJsonArraySha256 {
    pub(super) fn new() -> Self {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"[");
        Self {
            hasher,
            has_items: false,
        }
    }

    pub(super) fn push<T: serde::Serialize>(&mut self, value: &T) -> Result<()> {
        use sha2::Digest as _;
        if self.has_items {
            self.hasher.update(b",");
        }
        serde_json::to_writer(Sha256Writer(&mut self.hasher), value)
            .context("serialize streamed GRPO preflight identity item")?;
        self.has_items = true;
        Ok(())
    }

    pub(super) fn finish(mut self) -> String {
        use sha2::Digest as _;
        self.hasher.update(b"]");
        let digest: [u8; 32] = self.hasher.finalize().into();
        crate::train_receipt::format_sha256_digest(&digest)
    }
}

#[derive(serde::Serialize)]
pub(super) struct GrpoJsonlOrderIdentity<'a> {
    pub(super) source_index: usize,
    pub(super) source_line: usize,
    pub(super) byte_offset: u64,
    pub(super) next_byte_offset: u64,
    pub(super) line_sha256: &'a str,
    pub(super) completions: usize,
    pub(super) token_counts: &'a crate::train_receipt::TokenCountReceipt,
    pub(super) max_seq_len: usize,
}

#[derive(serde::Serialize)]
pub(super) struct GrpoJsonlGradientIdentity<'a> {
    pub(super) source_index: usize,
    pub(super) source_line: usize,
    pub(super) max_seq_len: usize,
    pub(super) enabled: bool,
    pub(super) num_segments: usize,
    pub(super) auto_configured: bool,
    pub(super) boundaries: &'a Option<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone)]
pub(super) struct GrpoJsonlTrainablePlanEntry {
    pub(super) source_index: usize,
    pub(super) source_line: usize,
    pub(super) byte_offset: u64,
    pub(super) next_byte_offset: u64,
    pub(super) line_sha256: String,
    pub(super) completions: usize,
    pub(super) token_counts: crate::train_receipt::TokenCountReceipt,
    pub(super) max_seq_len: usize,
    pub(super) gradient_checkpoint: GrpoJsonlGradientCheckpointPlan,
}

#[derive(Debug)]
pub(super) struct GrpoJsonlPreflightPlan {
    pub(super) total_bytes: u64,
    pub(super) total_lines: usize,
    pub(super) trainable: Vec<GrpoJsonlTrainablePlanEntry>,
    pub(super) planned_completions: usize,
    pub(super) planned_token_counts: crate::train_receipt::TokenCountReceipt,
    pub(super) data_stats: crate::train_receipt::DataStatsReceipt,
    pub(super) reward_stats: crate::train_receipt::RewardStatsReceipt,
    pub(super) dynamic_groups_filtered: usize,
    pub(super) trainable_order_sha256: String,
    pub(super) gradient_checkpoint_plan_sha256: String,
    pub(super) skip_training: bool,
    pub(super) openenv: Option<crate::OpenEnvTrainingDataProvenanceV1>,
}

impl GrpoJsonlPreflightPlan {
    fn expected_cursor(&self, global_step: usize) -> Result<(u64, usize)> {
        anyhow::ensure!(
            global_step <= self.trainable.len(),
            "streamed GRPO resume cursor {global_step} exceeds {} trainable groups",
            self.trainable.len()
        );
        Ok(if global_step == 0 {
            (0, 0)
        } else {
            let previous = &self.trainable[global_step - 1];
            (previous.next_byte_offset, previous.source_line)
        })
    }

    fn prefix_diagnostics(
        &self,
        global_step: usize,
    ) -> Result<(usize, crate::train_receipt::TokenCountReceipt)> {
        anyhow::ensure!(
            global_step <= self.trainable.len(),
            "streamed GRPO diagnostic prefix {global_step} exceeds {} groups",
            self.trainable.len()
        );
        let mut completions = 0usize;
        let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
        for entry in &self.trainable[..global_step] {
            completions = completions.saturating_add(entry.completions);
            token_counts.add_from(&entry.token_counts);
        }
        Ok((completions, token_counts))
    }
}

pub(super) fn grpo_checkpoint_static_data_stats(
    mut stats: crate::train_receipt::DataStatsReceipt,
) -> crate::train_receipt::DataStatsReceipt {
    stats.groups_trained = 0;
    stats.completions_trained = 0;
    // This is a publication location, not training state. A resumed server
    // job intentionally uses a new staging directory and rewrites the same
    // deterministic sidecar there.
    stats.reward_filter_sidecar = None;
    stats
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_grpo_jsonl_preflight_plan(
    dataset_source: &PinnedGrpoJsonlSource,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    output_dir: &Path,
    adapter_name: &str,
    device: &Device,
    activation_bytes_per_elem: usize,
    runtime: &crate::TrainingRuntimeContext,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<GrpoJsonlPreflightPlan> {
    use std::io::{BufRead, BufReader, Read as _};

    let dataset_path = dataset_source.display_path();
    let file = dataset_source.reader_from_start()?;
    let total_bytes = dataset_source.len()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut source_index = 0usize;
    let mut max_row_bytes = 0u64;
    let mut data_stats = crate::train_receipt::DataStatsReceipt::default();
    let filter_enabled = reward_filter_enabled(config);
    let mut reward_stats_accumulator = StreamedRewardStatsAccumulator::default();
    let mut reward_filter_inputs = Vec::new();
    let mut openenv_accumulator = crate::OpenEnvTrainingDataAccumulator::default();

    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during preflight",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .context("streamed GRPO preflight line count overflow")?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        max_row_bytes = max_row_bytes.max(line.len() as u64);
        bytes_read = bytes_read
            .checked_add(read as u64)
            .context("streamed GRPO preflight byte count overflow")?;
        streamed_grpo_preflight_host_bytes(
            data_stats.groups_read,
            data_stats.completions_read,
            max_row_bytes,
            model_config.num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound streamed GRPO preflight before parsing line {line_no}"))?;
        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        validate_grpo_trajectory_roles(&group, line_no)?;
        anyhow::ensure!(
            !group.completions.is_empty()
                && group.completions.len() <= crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP,
            "GRPO JSONL line {line_no} must contain 1..={} completions",
            crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP
        );
        source_index = source_index
            .checked_add(1)
            .context("streamed GRPO source index overflow")?;
        openenv_accumulator
            .observe_group(source_index, &group)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!("validate OpenEnv corpus provenance at GRPO JSONL line {line_no}")
            })?;
        data_stats.groups_read = data_stats
            .groups_read
            .checked_add(1)
            .context("streamed GRPO group count overflow")?;
        data_stats.completions_read = data_stats
            .completions_read
            .checked_add(group.completions.len())
            .context("streamed GRPO completion count overflow")?;
        streamed_grpo_preflight_host_bytes(
            data_stats.groups_read,
            data_stats.completions_read,
            max_row_bytes,
            model_config.num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound streamed GRPO preflight metadata at line {line_no}"))?;
        let reward_variance = reward_stats_accumulator.observe_group(
            group
                .completions
                .iter()
                .map(|completion| &completion.reward),
            config.reward_saturation_threshold,
        );
        if filter_enabled {
            reward_filter_inputs.push(RewardFilterInputGroup {
                id: format!("line:{line_no}"),
                source_index,
                source_line: Some(line_no),
                reward_variance,
            });
        }
    }
    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset length changed during preflight: expected {total_bytes}, read {bytes_read}"
    );
    let openenv = openenv_accumulator
        .finish()
        .map_err(anyhow::Error::msg)
        .context("finalize streamed GRPO OpenEnv corpus provenance")?;

    let preflight_host_bytes = streamed_grpo_preflight_host_bytes(
        data_stats.groups_read,
        data_stats.completions_read,
        max_row_bytes,
        model_config.num_layers,
        filter_enabled,
    )?;
    tracing::debug!(
        preflight_host_bytes,
        groups = data_stats.groups_read,
        completions = data_stats.completions_read,
        max_row_bytes,
        "bounded streamed GRPO preflight host plan"
    );
    let reward_stats = reward_stats_accumulator.finish();
    crate::train_receipt::warn_reward_diagnostics(
        "streamed_grpo_startup",
        adapter_name,
        &reward_stats,
        config.reward_saturation_threshold,
        config.reward_low_variance_threshold,
    );
    let reward_filter_plan = build_reward_filter_plan(
        config,
        output_dir,
        "jsonl_grpo_groups",
        reward_filter_inputs,
    )?;
    if let Some(plan) = reward_filter_plan.as_ref() {
        record_reward_filter_plan(&mut data_stats, plan);
        data_stats.groups_filtered = plan.groups_dropped;
        tracing::info!(
            kept = plan.groups_kept,
            dropped = plan.groups_dropped,
            sidecar = %plan.sidecar_path.display(),
            "streamed GRPO reward variance filter applied"
        );
        if let Some(reason) = plan.failure_reason.as_ref() {
            anyhow::bail!("{reason}");
        }
    }
    let skip_training = reward_filter_plan
        .as_ref()
        .is_some_and(|plan| plan.skip_training);
    if skip_training {
        return Ok(GrpoJsonlPreflightPlan {
            total_bytes,
            total_lines: line_no,
            trainable: Vec::new(),
            planned_completions: 0,
            planned_token_counts: crate::train_receipt::TokenCountReceipt::default(),
            data_stats,
            reward_stats,
            dynamic_groups_filtered: 0,
            trainable_order_sha256: StreamingJsonArraySha256::new().finish(),
            gradient_checkpoint_plan_sha256: StreamingJsonArraySha256::new().finish(),
            skip_training,
            openenv,
        });
    }

    drop(reader);
    let file = dataset_source.reader_from_start()?;
    let mut reader = BufReader::new(file);
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut source_index = 0usize;
    let mut dynamic_groups_filtered = 0usize;
    let mut trainable = Vec::new();
    trainable
        .try_reserve_exact(data_stats.groups_read)
        .context("reserve bounded streamed GRPO trainable plan")?;
    let mut planned_completions = 0usize;
    let mut planned_token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut order_identity = StreamingJsonArraySha256::new();
    let mut gradient_identity = StreamingJsonArraySha256::new();

    loop {
        line.clear();
        let byte_offset = bytes_read;
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during trainable preflight",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .context("streamed GRPO trainable-pass line count overflow")?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        bytes_read = bytes_read
            .checked_add(read as u64)
            .context("streamed GRPO trainable-pass byte count overflow")?;
        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        source_index = source_index
            .checked_add(1)
            .context("streamed GRPO source index overflow")?;
        if reward_filter_plan
            .as_ref()
            .is_some_and(|plan| !plan.keeps_source_line(line_no))
        {
            continue;
        }
        if config.dynamic_sampling && is_degenerate_grpo_group(&group) {
            dynamic_groups_filtered = dynamic_groups_filtered.saturating_add(1);
            continue;
        }

        let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
        let tokenized = tokenize_grpo_group_timed(&group, tokenizer, &mask_cfg, None)
            .with_context(|| {
                format!("preflight GRPO JSONL group {source_index} at line {line_no}")
            })?;
        validate_tokenized_behavior_policy(&tokenized, config.behavior_policy).with_context(
            || {
                format!(
                    "validate preflight GRPO JSONL group {source_index} at line {line_no} behavior provenance"
                )
            },
        )?;
        let token_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tokenized));
        let completions = tokenized.completions.len();
        let max_seq_len = tokenized
            .completions
            .iter()
            .map(|completion| completion.input_ids.len())
            .max()
            .unwrap_or(0);
        let checkpoint_config = checkpoint_config_for_training_step(
            weights,
            device,
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
            device,
            max_seq_len,
            checkpoint_config,
            streaming_prefill,
        );
        let line_sha256 = crate::train_receipt::sha256_bytes(line.as_bytes());
        order_identity.push(&GrpoJsonlOrderIdentity {
            source_index,
            source_line: line_no,
            byte_offset,
            next_byte_offset: bytes_read,
            line_sha256: &line_sha256,
            completions,
            token_counts: &token_counts,
            max_seq_len,
        })?;
        gradient_identity.push(&GrpoJsonlGradientIdentity {
            source_index,
            source_line: line_no,
            max_seq_len,
            enabled: checkpoint_config.enabled,
            num_segments: checkpoint_config.num_segments,
            auto_configured: checkpoint_config.auto_configured,
            boundaries: &boundaries,
        })?;
        let boundaries_sha256 = crate::train_receipt::sha256_json_serializable(&boundaries)
            .context("hash streamed GRPO checkpoint boundaries")?;
        planned_completions = planned_completions
            .checked_add(completions)
            .context("streamed GRPO planned completion count overflow")?;
        planned_token_counts.add_from(&token_counts);
        trainable.push(GrpoJsonlTrainablePlanEntry {
            source_index,
            source_line: line_no,
            byte_offset,
            next_byte_offset: bytes_read,
            line_sha256,
            completions,
            token_counts,
            max_seq_len,
            gradient_checkpoint: GrpoJsonlGradientCheckpointPlan {
                config: checkpoint_config,
                boundaries_sha256,
            },
        });
    }
    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset length changed between preflight passes: expected {total_bytes}, read {bytes_read}"
    );
    anyhow::ensure!(
        !trainable.is_empty(),
        "grpo_train_jsonl: no valid GRPO groups in {}",
        dataset_path.display()
    );
    anyhow::ensure!(
        planned_completions > 0 && planned_token_counts.action_tokens > 0,
        "grpo_train_jsonl: trainable groups contain no completions or action tokens"
    );
    data_stats.groups_filtered = data_stats
        .reward_groups_filtered
        .saturating_add(dynamic_groups_filtered);

    Ok(GrpoJsonlPreflightPlan {
        total_bytes,
        total_lines: line_no,
        trainable,
        planned_completions,
        planned_token_counts,
        data_stats,
        reward_stats,
        dynamic_groups_filtered,
        trainable_order_sha256: order_identity.finish(),
        gradient_checkpoint_plan_sha256: gradient_identity.finish(),
        skip_training,
        openenv,
    })
}

/// Stream GRPO training from a JSONL dataset path through the kt-native route.
///
/// Each non-empty line must be one [`GrpoGroup`]. Unlike [`grpo_train`], this
/// path does not retain all parsed or tokenized groups before training.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_jsonl_to(
        dataset_path,
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

/// Staged-output variant of [`grpo_train_jsonl`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to(
    dataset_path: &Path,
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
    grpo_train_jsonl_to_with_coordination(
        dataset_path,
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

/// Streaming staged-output GRPO with bounded server GPU ownership.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_coordination(
    dataset_path: &Path,
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
    grpo_train_jsonl_to_with_checkpoint_root(
        dataset_path,
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

/// Standalone streamed GRPO with a separate durable checkpoint root.
///
/// Server callers should use
/// [`grpo_train_jsonl_to_with_checkpoint_root_and_runtime`] to bind preflight
/// and execution to the same process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_checkpoint_root(
    dataset_path: &Path,
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
        "streamed GRPO",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    grpo_train_jsonl_to_with_checkpoint_root_and_runtime(
        dataset_path,
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

/// Server-owned streamed GRPO entry point with immutable runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_checkpoint_root_and_runtime(
    dataset_path: &Path,
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
    ensure_training_optimizer_entry_supported(
        "streamed GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    let dataset_source = PinnedGrpoJsonlSource::open(dataset_path)?;
    grpo_train_pinned_jsonl_to_with_checkpoint_root_and_runtime(
        &dataset_source,
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

/// Streamed GRPO entry point for a caller-pinned file identity.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_pinned_jsonl_to_with_checkpoint_root_and_runtime(
    dataset_source: &PinnedGrpoJsonlSource,
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
    use std::io::{BufRead, BufReader, Seek, SeekFrom};

    let dataset_path = dataset_source.display_path();
    let runtime_device = ensure_training_optimizer_entry_supported(
        "streamed GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize streamed GRPO memory governor")?;
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "GRPO checkpoint_interval must be greater than zero"
    );
    // Fail fast on compositions the kt-tape path cannot train. The
    // streaming path can't cheaply pre-scan every group for Observation
    // segments, so `no_policy_loss` / reserved-OPD reject here and the
    // echo+env case rejects per-group at mask-construction time, before
    // that group's forward (plus in the dry-run gate below).
    config
        .loss
        .validate_for_kt_tape(false)
        .map_err(|e| anyhow::anyhow!("GRPO loss config: {e}"))?;
    config
        .validate_policy_config()
        .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;
    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = dataset_source
        .sha256()
        .with_context(|| format!("hash GRPO JSONL dataset {}", dataset_path.display()))?;
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(Some(&training_data_sha256), "GRPO JSONL training data")?;
    let mut training_data = crate::train_receipt::TrainingDataReceipt {
        source: "jsonl_grpo_groups".to_string(),
        path: Some(dataset_path.display().to_string()),
        sha256: Some(training_data_sha256.clone()),
        openenv: None,
    };
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });
    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load streamed GRPO resume checkpoint")?;
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
            checkpoint.manifest.data.source_kind == GrpoCheckpointRoute::Jsonl.source_kind()
                && checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint streamed GRPO data identity differs from this request"
        );
        anyhow::ensure!(
            resume_loop_state
                .as_ref()
                .is_some_and(|state| state.route == GrpoCheckpointRoute::Jsonl),
            "resume checkpoint was not produced by streamed JSONL GRPO"
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
                .context("streamed GRPO resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported streamed GRPO lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "streamed GRPO resume seed {restored} differs from requested seed {requested}"
        );
    }
    let effective_seed_value = resume_init_seed
        .or(config.seed)
        .unwrap_or_else(rand::random);
    let learning_rate = config.effective_learning_rate();
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

    // (#1082) `embed_tokens.device()` is a kt Device; the OPD/GRPO body is now
    // kt-native (kt `Parameter`s, kt AdamW state, kt tape forward/backward), so
    // keep `device` kt downstream. The only candle touch is safetensors adapter
    // I/O, which bridges kt->candle locally inside save/load.
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    ensure_tape_forward_backward_supported("streamed GRPO", weights, backend.as_ref())?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "streamed GRPO",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);
    let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy,
        model_config_has_linear_attention(model_config),
    );
    if let Some(explicit) = config.learning_rate
        && let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Grpo),
        )
    {
        tracing::warn!(optimizer = ?config.optimizer, "GRPO {warning}");
    }
    tracing::info!(
        dataset = %dataset_path.display(),
        lr = learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting streamed GRPO training"
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
                Some(effective_seed_value),
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
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
                Some(effective_seed_value),
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    let preflight = match build_grpo_jsonl_preflight_plan(
        dataset_source,
        config,
        model_config,
        weights,
        tokenizer,
        &output_dir,
        adapter_name,
        &device,
        activation_bytes_per_elem,
        runtime,
        streaming_prefill,
    ) {
        Ok(plan) => plan,
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
                Some(effective_seed_value),
                Some(alpha_over_rank),
                base_adapter_dir
                    .as_deref()
                    .or(requested_base_adapter_dir.as_deref()),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    training_data.openenv = preflight.openenv.clone();
    let post_preflight_sha256 = dataset_source
        .sha256()
        .with_context(|| format!("rehash GRPO JSONL dataset {}", dataset_path.display()))?;
    anyhow::ensure!(
        post_preflight_sha256 == training_data_sha256,
        "GRPO JSONL dataset changed while constructing the exact trainable plan"
    );
    anyhow::ensure!(
        !(preflight.skip_training && resume_checkpoint.is_some()),
        "a streamed GRPO resume checkpoint cannot target a filter-skipped run"
    );

    let current_reward_filter_sidecar = preflight.data_stats.reward_filter_sidecar.clone();
    let mut data_stats = preflight.data_stats.clone();
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut echo_metrics = crate::train_receipt::EchoActivityMetrics::default();
    let reward_stats = preflight.reward_stats.clone();
    let mut lora_grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let mut policy_audit = crate::train_receipt::GrpoPolicyAuditAccumulator::default();
    let mut phase_timings = GrpoBenchmarkTimings::default();
    let mut gpu_writer_timings = GrpoGpuWriterTimings::default();
    let mut dynamic_groups_filtered = preflight.dynamic_groups_filtered;
    if let Some(state) = resume_loop_state.as_ref() {
        anyhow::ensure!(
            grpo_checkpoint_static_data_stats(state.data_stats.clone())
                == grpo_checkpoint_static_data_stats(preflight.data_stats.clone())
                && state.dynamic_groups_filtered as usize == preflight.dynamic_groups_filtered,
            "streamed GRPO resume filtering statistics differ from the current preflight"
        );
        data_stats = state.data_stats.clone();
        data_stats.reward_filter_sidecar = current_reward_filter_sidecar;
        token_counts = state.token_counts.clone();
        echo_metrics = state.echo_metrics.clone();
        lora_grad_norms = state.lora_grad_norms.clone();
        policy_audit = state.policy_audit.clone();
        phase_timings = state.phase_timings.clone();
        gpu_writer_timings = state.gpu_writer_timings.clone();
        dynamic_groups_filtered = state.dynamic_groups_filtered as usize;
    }

    let replay_parent_adapter = resume_checkpoint
        .is_none()
        .then_some(config.base_adapter.as_deref())
        .flatten();
    let (replay_state, effective_seed) = match replay_ctx.as_ref() {
        Some(ctx) => {
            let (state, seed) = open_replay_state_to(
                ctx,
                Some(effective_seed_value),
                replay_parent_adapter,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            )?;
            anyhow::ensure!(
                seed == effective_seed_value,
                "streamed GRPO replay seed drifted"
            );
            (Some(state), Some(seed))
        }
        None => (None, Some(effective_seed_value)),
    };

    // Upload only after the checkpoint and CPU-only trainable plan have both
    // been validated. This keeps malformed resume requests out of GPU work.
    let resident_weights = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed resident model setup",
        || resident_training_weights(weights, &device),
    )?;
    let weights = resident_weights.as_ref().unwrap_or(weights);

    let (mut params, mut opt_state) = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed adapter and optimizer setup",
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
                    "restored exact streamed GRPO adapter parameters"
                );
            } else if let Some(base_dir) = base_adapter_dir.as_deref() {
                let n_loaded = params.load_from_safetensors(base_dir, &device)?;
                tracing::info!(
                    base = %base_dir.display(),
                    num_tensors = n_loaded,
                    "loaded base adapter — continuing streamed GRPO from those weights"
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
                            .context("streamed GRPO resume optimizer step exceeds u32")?;
                        state.load_checkpoint_state(&params, &path, step)?;
                    }
                    (None, None) => {}
                    (Some(_), None) => anyhow::bail!(
                        "stateful streamed GRPO optimizer checkpoint has no optimizer artifact"
                    ),
                    (None, Some(_)) => anyhow::bail!(
                        "SGD streamed GRPO checkpoint unexpectedly contains optimizer state"
                    ),
                }
            }
            params.register_with_backend(&*backend)?;
            if let Some(state) = opt_state.as_ref() {
                state.register_with_backend(&*backend)?;
            }
            Ok((params, opt_state))
        },
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized streamed GRPO trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    let ema_refresh_every = if config.kl_penalty_enabled() {
        match &config.kl_reference_policy {
            KlReferencePolicy::Ema { refresh_every, .. } => Some(*refresh_every),
            _ => None,
        }
    } else {
        None
    };
    let checkpoint_descriptor = if preflight.skip_training {
        None
    } else {
        Some(GrpoCheckpointDescriptor {
            route: GrpoCheckpointRoute::Jsonl,
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: GrpoCheckpointRoute::Jsonl.source_kind().to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: preflight.trainable.len() as u64,
            },
            init_seed: effective_seed_value,
            optimizer: config.optimizer,
            learning_rate,
            total_steps: preflight.trainable.len(),
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: grpo_checkpoint_auxiliary_state(
                GrpoCheckpointRoute::Jsonl,
                model_config,
                tokenizer,
                training_precision_policy,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &preflight.trainable_order_sha256,
                &preflight.gradient_checkpoint_plan_sha256,
                &training_runtime_planning_identity,
            ),
            ema_refresh_every,
        })
    };
    if let (Some(descriptor), Some(checkpoint), Some(loop_state)) = (
        checkpoint_descriptor.as_ref(),
        resume_checkpoint.as_ref(),
        resume_loop_state.as_ref(),
    ) {
        descriptor.validate_resume(checkpoint, loop_state)?;
    }

    let mut train_body = || -> Result<(PathBuf, f64)> {
        tracing::info!(
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            trainable_groups = preflight.trainable.len(),
            total_bytes = preflight.total_bytes,
            "streamed GRPO exact trainable plan validated"
        );

        if preflight.skip_training {
            data_stats.groups_trained = 0;
            data_stats.completions_trained = 0;
            params.save_peft(&output_dir, model_config.num_layers)?;
            tracing::info!(
                adapter = adapter_name,
                path = %output_dir.display(),
                "streamed GRPO reward variance filter skipped training"
            );
            return Ok((output_dir.clone(), 0.0));
        }
        let checkpoint_descriptor = checkpoint_descriptor
            .as_ref()
            .context("streamed GRPO trainable plan has no checkpoint descriptor")?;
        let total_steps = preflight.trainable.len();
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
        let (expected_byte_offset, expected_lines_consumed) =
            preflight.expected_cursor(global_step)?;
        let (expected_completions, expected_token_counts) =
            preflight.prefix_diagnostics(global_step)?;
        if let Some(state) = resume_loop_state.as_ref() {
            anyhow::ensure!(
                state.source_byte_offset == Some(expected_byte_offset)
                    && state.source_lines_consumed == Some(expected_lines_consumed as u64),
                "streamed GRPO resume source cursor differs from the exact trainable prefix"
            );
        }
        anyhow::ensure!(
            processed_completions == expected_completions
                && token_counts == expected_token_counts
                && data_stats.groups_trained == global_step
                && data_stats.completions_trained == processed_completions,
            "streamed GRPO resume diagnostics do not match the committed trainable prefix"
        );

        let mut file = dataset_source.reader_from_start()?;
        file.seek(SeekFrom::Start(expected_byte_offset))
            .with_context(|| {
                format!(
                    "seek GRPO JSONL dataset {} to byte {expected_byte_offset}",
                    dataset_path.display()
                )
            })?;
        tracing::info!(
            dataset = %dataset_path.display(),
            total_bytes = preflight.total_bytes,
            byte_offset = expected_byte_offset,
            lines_consumed = expected_lines_consumed,
            global_step,
            total_steps,
            "streamed GRPO data positioned at exact resume cursor"
        );
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut bytes_read = expected_byte_offset;
        let mut line_no = expected_lines_consumed;
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;

        // Phase 3b: maintain an EMA-snapshot LoRA when
        // `KlReferencePolicy::Ema` is configured (see `grpo_train` for the
        // identical pattern; streaming JSONL just iterates one group at a
        // time).
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
                                .context(
                                    "EMA streamed GRPO resume checkpoint has no reference state",
                                )?;
                            let path = checkpoint.artifact_path(relative)?;
                            (
                                load_lora_reference_checkpoint(&path, &params, &device)?,
                                loop_state.ema_groups_since_refresh.context(
                                    "EMA streamed GRPO resume checkpoint has no cadence cursor",
                                )? as usize,
                            )
                        } else {
                            (
                                run_coordinated_grpo_gpu_phase(
                                    gpu_step_coordination.as_ref(),
                                    &*backend,
                                    &mut gpu_writer_timings,
                                    "streamed initial EMA reference snapshot",
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

        loop {
            line.clear();
            let byte_offset = bytes_read;
            let read = reader.read_line(&mut line).with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {}",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
            if read == 0 {
                break;
            }
            line_no = line_no.saturating_add(1);
            bytes_read = bytes_read
                .checked_add(read as u64)
                .context("streamed GRPO training byte count overflow")?;
            let Some(entry) = preflight.trainable.get(global_step) else {
                // Consume trailing blank/filtered lines so the final file
                // cursor and hash still cover the complete source.
                continue;
            };
            if line_no < entry.source_line {
                continue;
            }
            anyhow::ensure!(
                line_no == entry.source_line
                    && byte_offset == entry.byte_offset
                    && bytes_read == entry.next_byte_offset,
                "streamed GRPO source cursor drifted before trainable group {}: expected line {} bytes {}..{}, found line {} bytes {}..{}",
                global_step + 1,
                entry.source_line,
                entry.byte_offset,
                entry.next_byte_offset,
                line_no,
                byte_offset,
                bytes_read
            );
            anyhow::ensure!(
                crate::train_receipt::sha256_bytes(line.as_bytes()) == entry.line_sha256,
                "streamed GRPO trainable line {} changed after preflight",
                entry.source_line
            );
            let group = parse_grpo_jsonl_group_line(&line, line_no)?
                .context("planned streamed GRPO line became blank")?;
            validate_grpo_trajectory_roles(&group, line_no)?;
            let group_number = global_step + 1;
            tracing::info!(
                group = group_number,
                source_index = entry.source_index,
                line = line_no,
                line_bytes = read,
                byte_offset,
                "streamed GRPO tokenize start"
            );
            let tokenize_start = Instant::now();
            let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
            let tgroup =
                tokenize_grpo_group_timed(&group, tokenizer, &mask_cfg, Some(&mut phase_timings))
                    .with_context(|| {
                    format!(
                        "tokenize GRPO JSONL group {} at line {}",
                        group_number, line_no
                    )
                })?;
            validate_tokenized_behavior_policy(&tgroup, config.behavior_policy).with_context(
                || {
                    format!(
                        "validate GRPO JSONL group {} at line {} behavior provenance",
                        group_number, line_no
                    )
                },
            )?;
            let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tgroup));
            tracing::info!(
                group = group_number,
                completions = tgroup.completions.len(),
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                context_tokens = group_counts.context_tokens,
                elapsed_ms = tokenize_start.elapsed().as_millis() as u64,
                "streamed GRPO tokenize end"
            );

            let group_max_seq_len = tgroup
                .completions
                .iter()
                .map(|completion| completion.input_ids.len())
                .max()
                .unwrap_or(0);
            anyhow::ensure!(
                tgroup.completions.len() == entry.completions
                    && group_counts == entry.token_counts
                    && group_max_seq_len == entry.max_seq_len,
                "streamed GRPO tokenization drifted from preflight at line {}",
                line_no
            );
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
            let segments_sha256 = crate::train_receipt::sha256_json_serializable(&segments)
                .context("hash streamed GRPO runtime checkpoint boundaries")?;
            anyhow::ensure!(
                ckpt_config == entry.gradient_checkpoint.config
                    && segments_sha256 == entry.gradient_checkpoint.boundaries_sha256,
                "streamed GRPO gradient-checkpoint plan drifted at line {}",
                line_no
            );
            let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
            if last_ckpt_log_key != Some(ckpt_log_key) {
                if let Some(ref segs) = segments {
                    tracing::info!(
                        group = group_number,
                        max_seq_len = group_max_seq_len,
                        num_segments = segs.len(),
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        boundaries = ?segs,
                        "streamed GRPO gradient checkpointing enabled for group shape"
                    );
                } else {
                    tracing::info!(
                        group = group_number,
                        max_seq_len = group_max_seq_len,
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        "streamed GRPO gradient checkpointing disabled for group shape"
                    );
                }
                last_ckpt_log_key = Some(ckpt_log_key);
            }

            let step_report = run_coordinated_grpo_gpu_phase(
                gpu_step_coordination.as_ref(),
                &*backend,
                &mut gpu_writer_timings,
                "streamed optimizer group",
                || {
                    let step_report = train_tokenized_grpo_group_with_grad_norms(
                        &*backend,
                        &tgroup,
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
                    if let Some(state) = ema_ref_state.as_mut() {
                        state.groups_since_refresh += 1;
                        if state.groups_since_refresh >= state.refresh_every {
                            params
                                .sync_to_master(&*backend)
                                .context("sync streamed policy before EMA reference refresh")?;
                            state.snapshot = lora_snapshot_capture_or_blend(
                                &params,
                                Some(&state.snapshot),
                                state.decay,
                                &device,
                            )
                            .context("EMA reference snapshot refresh")?;
                            state.groups_since_refresh = 0;
                            tracing::debug!(
                                group = group_number,
                                refresh_every = state.refresh_every,
                                decay = state.decay,
                                "streamed GRPO EMA reference snapshot refreshed"
                            );
                        }
                    }
                    Ok(step_report)
                },
            )?;
            let avg_group_loss = step_report.loss;
            echo_metrics.observe_env_ce(step_report.echo_env_ce);
            anyhow::ensure!(
                avg_group_loss.is_finite(),
                "grpo_train_jsonl: non-finite loss {avg_group_loss} at group {group_number}"
            );
            last_loss = avg_group_loss;
            loss_history.push(avg_group_loss);
            global_step = global_step.saturating_add(1);
            processed_completions = processed_completions.saturating_add(entry.completions);
            token_counts.add_from(&group_counts);
            data_stats.groups_trained = global_step;
            data_stats.completions_trained = processed_completions;

            let checkpoint_due = config
                .checkpoint_interval
                .is_some_and(|interval| global_step % interval == 0 && global_step < total_steps);
            if checkpoint_due {
                let mut loop_state = GrpoCheckpointLoopState::capture(
                    GrpoCheckpointRoute::Jsonl,
                    global_step,
                    Some(bytes_read),
                    Some(line_no as u64),
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
                    "streamed checkpoint device snapshot",
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved exact streamed GRPO training checkpoint"
                );
            }

            let (step, progress_total_steps, progress) =
                jsonl_byte_progress(preflight.total_bytes, bytes_read);
            if let Some(ref cb) = progress_cb {
                let control = cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step,
                    total_steps: progress_total_steps,
                    loss: avg_group_loss,
                    progress,
                });
                if control == TrainControl::Stop && global_step < total_steps {
                    if last_saved_step != Some(global_step) {
                        let mut loop_state = GrpoCheckpointLoopState::capture(
                            GrpoCheckpointRoute::Jsonl,
                            global_step,
                            Some(bytes_read),
                            Some(line_no as u64),
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
                            "streamed cancellation checkpoint device snapshot",
                        )?;
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved exact streamed GRPO checkpoint before cancellation"
                        );
                    }
                    anyhow::bail!("training cancelled by user (stop requested at step boundary)");
                }
            }

            tracing::info!(
                group = global_step,
                completions_seen = processed_completions,
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                byte_offset = bytes_read,
                total_bytes = preflight.total_bytes,
                loss = format!("{avg_group_loss:.6}"),
                "streamed GRPO group step"
            );
            if let Some(echo_env_ce) = step_report.echo_env_ce {
                tracing::info!(
                    group = global_step,
                    completions_seen = processed_completions,
                    action_tokens = group_counts.action_tokens,
                    env_tokens = group_counts.env_tokens,
                    echo_env_ce,
                    "streamed GRPO ECHO group metrics"
                );
            }
        }

        anyhow::ensure!(
            global_step == total_steps
                && loss_history.len() == total_steps
                && processed_completions == preflight.planned_completions
                && token_counts == preflight.planned_token_counts
                && bytes_read == preflight.total_bytes
                && line_no == preflight.total_lines,
            "streamed GRPO completed with inconsistent progress, diagnostics, or source cursor"
        );
        drop(reader);
        let final_training_data_sha256 = dataset_source
            .sha256()
            .with_context(|| format!("rehash GRPO JSONL dataset {}", dataset_path.display()))?;
        anyhow::ensure!(
            final_training_data_sha256 == training_data_sha256,
            "GRPO JSONL dataset changed during training"
        );
        crate::train_receipt::warn_echo_enabled_without_env_tokens(
            "streamed_grpo",
            config.loss.echo_enabled(),
            &token_counts,
        );

        let synced = run_coordinated_grpo_gpu_phase(
            gpu_step_coordination.as_ref(),
            &*backend,
            &mut gpu_writer_timings,
            "streamed final adapter snapshot",
            || {
                params
                    .sync_to_master(&*backend)
                    .context("capture final streamed GRPO adapter state")
            },
        )?;
        tracing::debug!(
            synced,
            "synced LoRA Vars to candle before streamed GRPO save"
        );

        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            processed_groups = global_step,
            processed_completions,
            "streamed GRPO training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let mut result = train_body();
    drop(train_body);
    let policy_audit = finish_grpo_policy_audit(&mut result, policy_audit);
    let mut adapter_smoke_test = None;
    let cleanup_result = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed adapter smoke test and cleanup",
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
            if let Some(state) = opt_state.as_ref() {
                state.evict_from_backend(&*backend);
            }
            params.evict_from_backend(&*backend);
            Ok(())
        },
    );
    if let Err(error) = cleanup_result {
        if result.is_ok() {
            result = Err(error.context("complete coordinated streamed GRPO cleanup"));
        } else {
            tracing::warn!(error = %format!("{error:#}"), "streamed GRPO cleanup could not acquire healthy backend");
        }
    }
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append streamed GRPO replay outcome record");
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
        training_data,
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

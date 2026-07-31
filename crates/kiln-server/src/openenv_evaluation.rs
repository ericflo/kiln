//! Paired, held-out evaluation for trained OpenEnv policies.
//!
//! This module deliberately reuses the production OpenEnv collector instead
//! of inventing an eval-only protocol path. Baseline and candidate policies
//! receive identical environment URLs, reset payloads, seeds, candidate
//! indices, generation seeds, and bounds. Each side publishes the same exact
//! dataset/replay/summary bundle as training collection, then a compact
//! comparison receipt binds the two bundles and the promotion decision.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::openenv_cli::{
    OpenEnvCollectionControl, OpenEnvCollectionProgress, OpenEnvPolicyTransport,
    OpenEnvRolloutOptions, OpenEnvRolloutRecord, collect_openenv_rollouts_with_policy,
    write_openenv_outputs,
};
use crate::state::AppState;

pub const OPENENV_ENVIRONMENT_EVAL_RECEIPT_SCHEMA_V1: &str =
    "kiln.openenv-environment-evaluation.v1";
pub const OPENENV_ENVIRONMENT_EVAL_POLICY_V1: &str = "paired_return_sign_test_v1";
pub const OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS: usize = 20;
pub const OPENENV_ENVIRONMENT_EVAL_ALPHA: f64 = 0.05;

fn default_eval_groups() -> usize {
    OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS
}

fn default_eval_group_size() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalGate {
    /// Candidate point-return floor. This is never sufficient by itself:
    /// promotion also requires a statistically significant paired win.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_mean_return: Option<f64>,
    /// Minimum candidate-minus-baseline point-return delta. Promotion still
    /// requires the fixed exact paired sign-test policy.
    #[serde(default)]
    pub min_mean_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalConfig {
    /// Number of held-out reset seeds. Defaults to the gate's minimum
    /// independent seed-group count.
    #[serde(default = "default_eval_groups")]
    pub groups: usize,
    /// Stochastic episodes per held-out seed. Baseline and candidate use the
    /// same candidate indices and therefore the same generation seeds.
    #[serde(default = "default_eval_group_size")]
    pub group_size: usize,
    /// First held-out seed. When omitted, Kiln starts immediately after the
    /// training collection's seed range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_start: Option<u64>,
    /// Optional fail-closed promotion gate. Without it the paired evaluation
    /// is diagnostic and the training request's normal auto-load behavior is
    /// preserved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<OpenEnvEnvironmentEvalGate>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvEnvironmentEvalState {
    Pending,
    CollectingBaseline,
    CollectingCandidate,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvEnvironmentEvalDecision {
    Diagnostic,
    Passed,
    Rejected,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvEnvironmentEvalOutcome {
    Diagnostic,
    Promoted,
    Kept,
    Rejected,
    Inconclusive,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvPolicyIdentity {
    /// `None` denotes the base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Exact LoRA config/tensor content identity. `None` for the base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_content_revision: Option<String>,
    /// Hash of Kiln's process-lifetime execution envelope when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provenance_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalEvidence {
    pub policy_version: String,
    pub paired_groups: usize,
    pub paired_episodes: usize,
    pub minimum_paired_groups: usize,
    pub baseline_mean_return: f64,
    pub candidate_mean_return: f64,
    pub mean_return_improvement: f64,
    pub improved_groups: usize,
    pub regressed_groups: usize,
    pub tied_groups: usize,
    pub exact_sign_test_p_value: f64,
    pub exact_sign_test_alpha: f64,
    pub decision: OpenEnvEnvironmentEvalDecision,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalProgress {
    pub state: OpenEnvEnvironmentEvalState,
    pub groups_completed: usize,
    pub groups_total: usize,
    pub rollouts_completed: usize,
    pub rollouts_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalStatus {
    pub state: OpenEnvEnvironmentEvalState,
    pub seed_start: u64,
    pub groups: usize,
    pub group_size: usize,
    pub baseline: OpenEnvPolicyIdentity,
    pub candidate: OpenEnvPolicyIdentity,
    pub progress: OpenEnvEnvironmentEvalProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<OpenEnvEnvironmentEvalEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<OpenEnvEnvironmentEvalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvEnvironmentEvalReceipt {
    pub schema: String,
    pub run_id: String,
    pub config: OpenEnvEnvironmentEvalConfig,
    pub seed_start: u64,
    pub baseline: OpenEnvPolicyIdentity,
    pub candidate: OpenEnvPolicyIdentity,
    pub baseline_summary_sha256: String,
    pub candidate_summary_sha256: String,
    pub evidence: OpenEnvEnvironmentEvalEvidence,
    pub outcome: OpenEnvEnvironmentEvalOutcome,
    pub verdict: String,
}

#[derive(Debug)]
pub(crate) struct OpenEnvEnvironmentEvalCollection {
    pub evidence: OpenEnvEnvironmentEvalEvidence,
}

pub(crate) type OpenEnvEnvironmentEvalProgressCallback =
    Arc<dyn Fn(OpenEnvEnvironmentEvalProgress) + Send + Sync>;

pub(crate) fn policy_identity(state: &AppState, adapter: &str) -> Result<OpenEnvPolicyIdentity> {
    let adapter = normalized_adapter(adapter);
    let adapter_content_revision = adapter
        .as_ref()
        .map(|name| {
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
                &state.adapter_dir.join(name),
            )
            .with_context(|| format!("resolve exact OpenEnv evaluation policy {name:?}"))
            .map(|identity| format!("sha256:{}", identity.content_revision()))
        })
        .transpose()?;
    let execution_provenance_sha256 = state
        .execution_provenance
        .as_ref()
        .map(|provenance| {
            kiln_core::config_hashes::sha256_json_serializable(provenance.as_ref())
                .context("hash OpenEnv evaluation execution provenance")
        })
        .transpose()?;
    Ok(OpenEnvPolicyIdentity {
        adapter,
        adapter_content_revision,
        execution_provenance_sha256,
    })
}

pub(crate) fn normalized_adapter(adapter: &str) -> Option<String> {
    let trimmed = adapter.trim();
    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "base" | "none" | "null"
    ) {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub(crate) async fn collect_environment_evaluation(
    policy: &OpenEnvPolicyTransport,
    baseline_options: OpenEnvRolloutOptions,
    candidate_options: OpenEnvRolloutOptions,
    cancel: Arc<AtomicBool>,
    progress: OpenEnvEnvironmentEvalProgressCallback,
    gate: Option<&OpenEnvEnvironmentEvalGate>,
) -> Result<OpenEnvEnvironmentEvalCollection> {
    anyhow::ensure!(
        baseline_options.environment_urls == candidate_options.environment_urls
            && baseline_options.groups == candidate_options.groups
            && baseline_options.group_size == candidate_options.group_size
            && baseline_options.seed_start == candidate_options.seed_start
            && baseline_options.reset_options_value == candidate_options.reset_options_value
            && baseline_options.max_steps == candidate_options.max_steps
            && baseline_options.max_action_tokens == candidate_options.max_action_tokens
            && baseline_options.temperature == candidate_options.temperature
            && baseline_options.thinking == candidate_options.thinking
            && baseline_options.protocol_error_reward == candidate_options.protocol_error_reward
            && baseline_options.max_recoverable_errors == candidate_options.max_recoverable_errors
            && baseline_options.concurrency == candidate_options.concurrency
            && baseline_options.capacity_wait_seconds == candidate_options.capacity_wait_seconds,
        "paired OpenEnv environment evaluation options drifted"
    );
    let rollouts_total = baseline_options
        .groups
        .checked_mul(baseline_options.group_size)
        .context("OpenEnv environment evaluation rollout count overflow")?;
    for path in [&baseline_options.output, &candidate_options.output] {
        let parent = path
            .parent()
            .context("OpenEnv environment evaluation output has no parent")?;
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create OpenEnv environment evaluation output directory {}",
                parent.display()
            )
        })?;
    }
    let callback = progress.clone();
    let baseline_control = OpenEnvCollectionControl::new(
        cancel.clone(),
        Some(Arc::new(move |value: OpenEnvCollectionProgress| {
            callback(OpenEnvEnvironmentEvalProgress {
                state: OpenEnvEnvironmentEvalState::CollectingBaseline,
                groups_completed: value.groups_completed,
                groups_total: value.groups_total,
                rollouts_completed: value.rollouts_completed,
                rollouts_total,
            });
        })),
        None,
    );
    let baseline =
        collect_openenv_rollouts_with_policy(&baseline_options, policy, &baseline_control).await?;
    write_openenv_outputs(
        &baseline_options,
        &baseline.groups,
        &baseline.replay,
        &baseline.summary,
    )?;

    progress(OpenEnvEnvironmentEvalProgress {
        state: OpenEnvEnvironmentEvalState::CollectingCandidate,
        groups_completed: 0,
        groups_total: candidate_options.groups,
        rollouts_completed: 0,
        rollouts_total,
    });
    let callback = progress;
    let candidate_control = OpenEnvCollectionControl::new(
        cancel,
        Some(Arc::new(move |value: OpenEnvCollectionProgress| {
            callback(OpenEnvEnvironmentEvalProgress {
                state: OpenEnvEnvironmentEvalState::CollectingCandidate,
                groups_completed: value.groups_completed,
                groups_total: value.groups_total,
                rollouts_completed: value.rollouts_completed,
                rollouts_total,
            });
        })),
        None,
    );
    let candidate =
        collect_openenv_rollouts_with_policy(&candidate_options, policy, &candidate_control)
            .await?;
    ensure_paired_environment_replays(&baseline.replay, &candidate.replay)?;
    write_openenv_outputs(
        &candidate_options,
        &candidate.groups,
        &candidate.replay,
        &candidate.summary,
    )?;

    let evidence = evaluate_paired_returns(
        &baseline.summary.rollouts,
        &candidate.summary.rollouts,
        gate,
    )?;
    Ok(OpenEnvEnvironmentEvalCollection { evidence })
}

fn ensure_paired_environment_replays(
    baseline: &crate::openenv_replay::OpenEnvReplayManifest,
    candidate: &crate::openenv_replay::OpenEnvReplayManifest,
) -> Result<()> {
    anyhow::ensure!(
        baseline.environments == candidate.environments
            && baseline.groups.len() == candidate.groups.len(),
        "OpenEnv environment identity drifted between paired baseline and candidate evaluation"
    );
    for (baseline, candidate) in baseline.groups.iter().zip(&candidate.groups) {
        anyhow::ensure!(
            baseline.group_index == candidate.group_index
                && baseline.environment_index == candidate.environment_index
                && baseline.seed == candidate.seed
                && baseline.reset_payload == candidate.reset_payload
                && baseline.reset_observation == candidate.reset_observation
                && baseline.candidates.len() == candidate.candidates.len(),
            "OpenEnv reset drifted between paired evaluation policies at group {}",
            baseline.group_index
        );
        anyhow::ensure!(
            baseline
                .candidates
                .iter()
                .zip(&candidate.candidates)
                .all(|(baseline, candidate)| baseline.candidate_index == candidate.candidate_index),
            "OpenEnv candidate identity drifted between paired evaluation policies at group {}",
            baseline.group_index
        );
    }
    Ok(())
}

pub(crate) fn evaluate_paired_returns(
    baseline: &[OpenEnvRolloutRecord],
    candidate: &[OpenEnvRolloutRecord],
    gate: Option<&OpenEnvEnvironmentEvalGate>,
) -> Result<OpenEnvEnvironmentEvalEvidence> {
    anyhow::ensure!(
        !baseline.is_empty() && baseline.len() == candidate.len(),
        "paired OpenEnv evaluation requires equal non-empty episode sets"
    );
    let mut baseline_sum = 0.0f64;
    let mut candidate_sum = 0.0f64;
    let mut grouped_returns = BTreeMap::<usize, (f64, f64, usize)>::new();
    for (baseline, candidate) in baseline.iter().zip(candidate) {
        anyhow::ensure!(
            baseline.group_index == candidate.group_index
                && baseline.candidate_index == candidate.candidate_index
                && baseline.environment_name == candidate.environment_name
                && baseline.environment_url == candidate.environment_url
                && baseline.seed == candidate.seed,
            "paired OpenEnv evaluation episode identity drifted at group {} candidate {}",
            baseline.group_index,
            baseline.candidate_index
        );
        anyhow::ensure!(
            baseline.episode_return.is_finite() && candidate.episode_return.is_finite(),
            "paired OpenEnv evaluation contains a non-finite return"
        );
        baseline_sum += baseline.episode_return;
        candidate_sum += candidate.episode_return;
        let group = grouped_returns
            .entry(baseline.group_index)
            .or_insert((0.0, 0.0, 0));
        group.0 += baseline.episode_return;
        group.1 += candidate.episode_return;
        group.2 += 1;
    }
    let mut improved_groups = 0usize;
    let mut regressed_groups = 0usize;
    let mut tied_groups = 0usize;
    for (baseline_sum, candidate_sum, episodes) in grouped_returns.values().copied() {
        let baseline_mean = baseline_sum / episodes as f64;
        let candidate_mean = candidate_sum / episodes as f64;
        match candidate_mean
            .partial_cmp(&baseline_mean)
            .context("compare finite OpenEnv per-seed mean returns")?
        {
            std::cmp::Ordering::Greater => improved_groups += 1,
            std::cmp::Ordering::Less => regressed_groups += 1,
            std::cmp::Ordering::Equal => tied_groups += 1,
        }
    }
    let paired_groups = grouped_returns.len();
    let paired_episodes = baseline.len();
    let baseline_mean_return = baseline_sum / paired_episodes as f64;
    let candidate_mean_return = candidate_sum / paired_episodes as f64;
    let mean_return_improvement = candidate_mean_return - baseline_mean_return;
    let improved_u32 =
        u32::try_from(improved_groups).context("OpenEnv improved group count exceeds u32")?;
    let regressed_u32 =
        u32::try_from(regressed_groups).context("OpenEnv regressed group count exceeds u32")?;
    let sign_test = kiln_eval::result::sign_test(improved_u32, regressed_u32);

    let (decision, reason) = match gate {
        None => (
            OpenEnvEnvironmentEvalDecision::Diagnostic,
            "paired environment evaluation completed without a promotion gate".to_string(),
        ),
        Some(_) if paired_groups < OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS => (
            OpenEnvEnvironmentEvalDecision::Inconclusive,
            format!(
                "only {paired_groups} paired seed groups; policy requires at least {OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS}"
            ),
        ),
        Some(_)
            if regressed_groups > improved_groups
                && sign_test.p_value < OPENENV_ENVIRONMENT_EVAL_ALPHA =>
        {
            (
                OpenEnvEnvironmentEvalDecision::Rejected,
                format!(
                    "significant paired return regression ({improved_groups} seed groups improved, {regressed_groups} regressed, exact p={:.6})",
                    sign_test.p_value
                ),
            )
        }
        Some(gate)
            if gate
                .min_mean_return
                .is_some_and(|floor| candidate_mean_return < floor) =>
        {
            (
                OpenEnvEnvironmentEvalDecision::Rejected,
                format!(
                    "candidate mean return {candidate_mean_return:.6} is below configured floor {:.6}",
                    gate.min_mean_return.expect("matched Some")
                ),
            )
        }
        Some(gate) if mean_return_improvement < gate.min_mean_improvement => (
            OpenEnvEnvironmentEvalDecision::Rejected,
            format!(
                "mean return improvement {mean_return_improvement:.6} is below configured minimum {:.6}",
                gate.min_mean_improvement
            ),
        ),
        Some(_)
            if improved_groups > regressed_groups
                && sign_test.p_value < OPENENV_ENVIRONMENT_EVAL_ALPHA =>
        {
            (
                OpenEnvEnvironmentEvalDecision::Passed,
                format!(
                    "significant paired return improvement ({improved_groups} seed groups improved, {regressed_groups} regressed, exact p={:.6})",
                    sign_test.p_value
                ),
            )
        }
        Some(_) => (
            OpenEnvEnvironmentEvalDecision::Inconclusive,
            format!(
                "no significant paired return improvement ({improved_groups} seed groups improved, {regressed_groups} regressed, exact p={:.6})",
                sign_test.p_value
            ),
        ),
    };

    Ok(OpenEnvEnvironmentEvalEvidence {
        policy_version: OPENENV_ENVIRONMENT_EVAL_POLICY_V1.to_string(),
        paired_groups,
        paired_episodes,
        minimum_paired_groups: OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS,
        baseline_mean_return,
        candidate_mean_return,
        mean_return_improvement,
        improved_groups,
        regressed_groups,
        tied_groups,
        exact_sign_test_p_value: sign_test.p_value,
        exact_sign_test_alpha: OPENENV_ENVIRONMENT_EVAL_ALPHA,
        decision,
        reason,
    })
}

pub(crate) fn summary_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "read OpenEnv environment evaluation summary {}",
            path.display()
        )
    })?;
    Ok(crate::openenv_replay::sha256_bytes(&bytes))
}

pub(crate) fn write_environment_evaluation_receipt(
    path: &Path,
    receipt: &OpenEnvEnvironmentEvalReceipt,
) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| {
        format!(
            "create OpenEnv environment evaluation directory {}",
            parent.display()
        )
    })?;
    let mut staged = tempfile::NamedTempFile::new_in(parent).with_context(|| {
        format!(
            "stage OpenEnv environment evaluation receipt beside {}",
            path.display()
        )
    })?;
    serde_json::to_writer_pretty(staged.as_file_mut(), receipt)
        .context("serialize OpenEnv environment evaluation receipt")?;
    staged.as_file_mut().write_all(b"\n")?;
    staged.as_file().sync_all().with_context(|| {
        format!(
            "sync OpenEnv environment evaluation receipt {}",
            path.display()
        )
    })?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "publish OpenEnv environment evaluation receipt {}",
                path.display()
            )
        })?;
    Ok(())
}

pub(crate) fn evaluation_paths(run_dir: &Path, side: &str) -> OpenEnvRolloutOptionsPaths {
    let root = run_dir.join("environment-evaluation").join(side);
    OpenEnvRolloutOptionsPaths {
        output: root.join("rollouts.jsonl"),
        replay_output: root.join("replay.json"),
        summary_output: root.join("summary.json"),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OpenEnvRolloutOptionsPaths {
    pub output: PathBuf,
    pub replay_output: PathBuf,
    pub summary_output: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::post};
    use kiln_train::OpenEnvEpisodeTerminationV1;
    use serde_json::{Value, json};

    fn record(index: usize, value: f64) -> OpenEnvRolloutRecord {
        OpenEnvRolloutRecord {
            group_index: index,
            candidate_index: 0,
            environment_name: "counter".into(),
            environment_url: "http://127.0.0.1:8000".into(),
            seed: u64::try_from(index).unwrap(),
            steps: 1,
            episode_return: value,
            terminal_done: true,
            termination: OpenEnvEpisodeTerminationV1::Done,
            protocol_error_code: None,
            recoverable_protocol_errors: 0,
            capacity_retries: 0,
            model_tokens: 1,
            model_latency_ms: 1.0,
        }
    }

    #[test]
    fn significant_paired_improvement_passes_gate() {
        let baseline = (0..20).map(|index| record(index, 0.0)).collect::<Vec<_>>();
        let candidate = (0..20).map(|index| record(index, 1.0)).collect::<Vec<_>>();
        let evidence = evaluate_paired_returns(
            &baseline,
            &candidate,
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: Some(0.5),
                min_mean_improvement: 0.5,
            }),
        )
        .unwrap();
        assert_eq!(evidence.decision, OpenEnvEnvironmentEvalDecision::Passed);
        assert!(evidence.exact_sign_test_p_value < OPENENV_ENVIRONMENT_EVAL_ALPHA);
    }

    #[test]
    fn tied_or_undersized_evidence_never_promotes() {
        let baseline = (0..20).map(|index| record(index, 1.0)).collect::<Vec<_>>();
        let tied = evaluate_paired_returns(
            &baseline,
            &baseline,
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(tied.decision, OpenEnvEnvironmentEvalDecision::Inconclusive);
        let undersized = evaluate_paired_returns(
            &baseline[..19],
            &baseline[..19],
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(
            undersized.decision,
            OpenEnvEnvironmentEvalDecision::Inconclusive
        );

        let mut one_seed_baseline = (0..20).map(|index| record(index, 0.0)).collect::<Vec<_>>();
        let mut one_seed_candidate = (0..20).map(|index| record(index, 1.0)).collect::<Vec<_>>();
        for (index, (baseline, candidate)) in one_seed_baseline
            .iter_mut()
            .zip(&mut one_seed_candidate)
            .enumerate()
        {
            baseline.group_index = 0;
            baseline.candidate_index = index;
            baseline.seed = 7;
            candidate.group_index = 0;
            candidate.candidate_index = index;
            candidate.seed = 7;
        }
        let clustered = evaluate_paired_returns(
            &one_seed_baseline,
            &one_seed_candidate,
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(clustered.paired_episodes, 20);
        assert_eq!(clustered.paired_groups, 1);
        assert_eq!(
            clustered.decision,
            OpenEnvEnvironmentEvalDecision::Inconclusive
        );
    }

    #[test]
    fn regression_and_point_threshold_fail_closed() {
        let baseline = (0..20).map(|index| record(index, 1.0)).collect::<Vec<_>>();
        let candidate = (0..20).map(|index| record(index, 0.0)).collect::<Vec<_>>();
        let regression = evaluate_paired_returns(
            &baseline,
            &candidate,
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(
            regression.decision,
            OpenEnvEnvironmentEvalDecision::Rejected
        );

        let improved = (0..20).map(|index| record(index, 2.0)).collect::<Vec<_>>();
        let below_floor = evaluate_paired_returns(
            &baseline,
            &improved,
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: Some(3.0),
                min_mean_improvement: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(
            below_floor.decision,
            OpenEnvEnvironmentEvalDecision::Rejected
        );
    }

    #[test]
    fn paired_episode_identity_drift_is_rejected() {
        let baseline = vec![record(0, 0.0)];
        let mut candidate = vec![record(0, 1.0)];
        candidate[0].seed = 99;
        assert!(evaluate_paired_returns(&baseline, &candidate, None).is_err());
    }

    #[test]
    fn evaluation_receipt_is_byte_stable_and_binds_both_summaries() {
        let baseline_records = (0..20).map(|index| record(index, 0.0)).collect::<Vec<_>>();
        let candidate_records = (0..20).map(|index| record(index, 1.0)).collect::<Vec<_>>();
        let evidence =
            evaluate_paired_returns(&baseline_records, &candidate_records, None).unwrap();
        let receipt = OpenEnvEnvironmentEvalReceipt {
            schema: OPENENV_ENVIRONMENT_EVAL_RECEIPT_SCHEMA_V1.into(),
            run_id: "run-1".into(),
            config: OpenEnvEnvironmentEvalConfig {
                groups: 20,
                group_size: 1,
                seed_start: None,
                gate: None,
            },
            seed_start: 100,
            baseline: OpenEnvPolicyIdentity {
                adapter: None,
                adapter_content_revision: None,
                execution_provenance_sha256: Some(format!("sha256:{}", "1".repeat(64))),
            },
            candidate: OpenEnvPolicyIdentity {
                adapter: Some("candidate".into()),
                adapter_content_revision: Some(format!("sha256:{}", "2".repeat(64))),
                execution_provenance_sha256: Some(format!("sha256:{}", "1".repeat(64))),
            },
            baseline_summary_sha256: format!("sha256:{}", "3".repeat(64)),
            candidate_summary_sha256: format!("sha256:{}", "4".repeat(64)),
            evidence,
            outcome: OpenEnvEnvironmentEvalOutcome::Diagnostic,
            verdict: "diagnostic".into(),
        };
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first.json");
        let second = temp.path().join("second.json");
        write_environment_evaluation_receipt(&first, &receipt).unwrap();
        write_environment_evaluation_receipt(&second, &receipt).unwrap();
        let first_bytes = std::fs::read(first).unwrap();
        assert_eq!(first_bytes, std::fs::read(second).unwrap());
        let value: Value = serde_json::from_slice(&first_bytes).unwrap();
        assert_eq!(
            value["baseline_summary_sha256"],
            format!("sha256:{}", "3".repeat(64))
        );
        assert_eq!(
            value["candidate_summary_sha256"],
            format!("sha256:{}", "4".repeat(64))
        );
    }

    /// Byte-real proof that paired evaluation uses the public OpenEnv wire
    /// path, identical held-out seeds, and environment-owned returns.
    #[tokio::test]
    #[ignore = "requires a live max-sessions=1 OpenEnv-compatible bandit server"]
    async fn paired_evaluation_drives_a_real_openenv_bandit() {
        let environment_url = std::env::var("KILN_OPENENV_INTEROP_BANDIT_URL")
            .expect("KILN_OPENENV_INTEROP_BANDIT_URL must identify the live bandit");
        let app = Router::new().route("/v1/chat/completions", post(fake_bandit_policy));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let policy = OpenEnvPolicyTransport::Http {
            client,
            kiln_url: format!("http://{address}"),
        };
        let temp = tempfile::tempdir().unwrap();
        let options = |adapter: &str, side: &str| {
            let root = temp.path().join(side);
            OpenEnvRolloutOptions {
                kiln_url: format!("http://{address}"),
                environment_urls: vec![environment_url.clone()],
                adapter: adapter.to_string(),
                groups: 20,
                group_size: 1,
                seed_start: 1000,
                reset_options: None,
                reset_options_value: None,
                max_steps: 1,
                concurrency: 1,
                max_action_tokens: 16,
                temperature: 0.0,
                thinking: false,
                protocol_error_reward: -1.0,
                max_recoverable_errors: 0,
                capacity_wait_seconds: 10,
                output: root.join("rollouts.jsonl"),
                replay_output: root.join("replay.json"),
                summary_output: root.join("summary.json"),
            }
        };
        let collection = collect_environment_evaluation(
            &policy,
            options("base", "baseline"),
            options("candidate", "candidate"),
            Arc::new(AtomicBool::new(false)),
            Arc::new(|_| {}),
            Some(&OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            collection.evidence.decision,
            OpenEnvEnvironmentEvalDecision::Passed
        );
        assert_eq!(collection.evidence.regressed_groups, 0);
        assert!(collection.evidence.improved_groups >= 15);
        assert!(temp.path().join("baseline/replay.json").is_file());
        assert!(temp.path().join("candidate/replay.json").is_file());
        server.abort();
    }

    async fn fake_bandit_policy(Json(body): Json<Value>) -> Json<Value> {
        let seed = body["seed"].as_u64().unwrap();
        let (best, worst) = bandit_extreme_arms(seed);
        let candidate = body["adapter"].as_str() == Some("candidate");
        Json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": format!(r#"{{"arm":{}}}"#, if candidate { best } else { worst })
                }
            }],
            "usage": {"total_tokens": 1}
        }))
    }

    fn bandit_extreme_arms(seed: u64) -> (usize, usize) {
        let mut state = seed;
        let mut order = (0usize..10).collect::<Vec<_>>();
        for index in (1..10).rev() {
            let swap = splitmix64(&mut state) as usize % (index + 1);
            order.swap(index, swap);
        }
        let mut payout_rank = [0usize; 10];
        for (arm, rank) in order.into_iter().enumerate() {
            payout_rank[arm] = rank;
        }
        let best = payout_rank
            .iter()
            .enumerate()
            .max_by_key(|(_, rank)| *rank)
            .unwrap()
            .0;
        let worst = payout_rank
            .iter()
            .enumerate()
            .min_by_key(|(_, rank)| *rank)
            .unwrap()
            .0;
        (best, worst)
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

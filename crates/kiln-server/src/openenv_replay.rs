//! Content-addressed OpenEnv environment transcripts.
//!
//! Canonical GRPO JSONL intentionally contains only the prompt, scored
//! trajectory, and compact OpenEnv provenance needed by training. This module
//! owns the complementary exact transcript used to verify an artifact bundle
//! offline and to replay every environment exchange against a live server.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use futures::{StreamExt, stream};
use kiln_openenv::{
    OPENENV_CLIENT_PROFILE, OpenEnvClient, OpenEnvClientError, OpenEnvErrorCode, OpenEnvInspection,
    OpenEnvObservation, OpenEnvProtocolError, OpenEnvSession,
};
use kiln_train::{AgenticGroup, OpenEnvEpisodeTerminationV1, ScoredRollout, TurnKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::openenv_cli::{OpenEnvRolloutSummary, openenv_client};

pub const OPENENV_REPLAY_SCHEMA_V1: &str = "kiln.openenv-replay.v1";
pub const OPENENV_VERIFICATION_SCHEMA_V1: &str = "kiln.openenv-verification.v1";
pub const OPENENV_REPLAY_RUN_SCHEMA_V1: &str = "kiln.openenv-replay-run.v1";

pub(crate) const MAX_OPENENV_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;
const CAPACITY_RETRY_FLOOR: Duration = Duration::from_millis(250);
const CAPACITY_RETRY_CEILING: Duration = Duration::from_secs(2);

/// Typed terminal capacity result retained through `anyhow::Context` so the
/// persisted OpenEnv workflow can expose automation-safe failure semantics.
#[derive(Debug, thiserror::Error)]
#[error(
    "OpenEnv server {base_url} remained at capacity for {waited:?} after {retries} retries (active {active_sessions:?}, max {max_sessions:?})"
)]
pub(crate) struct OpenEnvCapacityTimeout {
    base_url: String,
    waited: Duration,
    retries: usize,
    active_sessions: Option<u64>,
    max_sessions: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvReplayManifest {
    pub schema: String,
    pub client_profile: String,
    pub dataset_sha256: String,
    pub protocol_error_reward: f64,
    pub max_steps: usize,
    pub environments: Vec<OpenEnvInspection>,
    pub groups: Vec<OpenEnvReplayGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvReplayGroup {
    pub group_index: usize,
    pub environment_index: usize,
    pub seed: u64,
    pub reset_payload: Value,
    pub reset_observation: OpenEnvObservation,
    pub candidates: Vec<OpenEnvReplayCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvReplayCandidate {
    pub candidate_index: usize,
    pub exchanges: Vec<OpenEnvReplayExchange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_state: Option<Value>,
    pub episode_return: f64,
    pub terminal_done: bool,
    pub termination: OpenEnvEpisodeTerminationV1,
    pub recoverable_protocol_errors: usize,
    pub capacity_retries: usize,
    pub model_tokens: usize,
    pub model_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvReplayExchange {
    pub step_index: usize,
    pub action: Value,
    pub result: OpenEnvReplayExchangeResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OpenEnvReplayExchangeResult {
    Observation {
        observation: OpenEnvObservation,
    },
    ProtocolError {
        error: OpenEnvProtocolError,
        /// Whether the rollout continued on the same socket after this
        /// recoverable protocol error.
        continued: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvVerificationReport {
    pub schema: String,
    pub summary_path: String,
    pub dataset_path: String,
    pub replay_path: String,
    pub dataset_sha256: String,
    pub replay_sha256: String,
    pub groups: usize,
    pub rollouts: usize,
    pub environment_exchanges: usize,
}

#[derive(Debug)]
pub struct VerifiedOpenEnvArtifacts {
    pub summary: OpenEnvRolloutSummary,
    pub groups: Vec<AgenticGroup>,
    pub replay: OpenEnvReplayManifest,
    pub report: OpenEnvVerificationReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvReplayRunReport {
    pub schema: String,
    pub replay_sha256: String,
    pub environments: usize,
    pub groups: usize,
    pub rollouts: usize,
    pub environment_exchanges: usize,
    pub capacity_retries: usize,
    pub environment_prefix_only_rollouts: usize,
}

impl OpenEnvReplayManifest {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema == OPENENV_REPLAY_SCHEMA_V1,
            "unsupported OpenEnv replay schema {:?}; expected {OPENENV_REPLAY_SCHEMA_V1}",
            self.schema
        );
        anyhow::ensure!(
            self.client_profile == OPENENV_CLIENT_PROFILE,
            "unsupported OpenEnv client profile {:?}; expected {OPENENV_CLIENT_PROFILE}",
            self.client_profile
        );
        ensure_sha256("dataset_sha256", &self.dataset_sha256)?;
        anyhow::ensure!(
            self.protocol_error_reward.is_finite(),
            "OpenEnv replay protocol_error_reward must be finite"
        );
        anyhow::ensure!(
            self.max_steps > 0,
            "OpenEnv replay max_steps must be positive"
        );
        anyhow::ensure!(
            !self.environments.is_empty(),
            "OpenEnv replay must name at least one environment"
        );
        anyhow::ensure!(
            !self.groups.is_empty(),
            "OpenEnv replay must contain at least one group"
        );

        for (group_index, group) in self.groups.iter().enumerate() {
            anyhow::ensure!(
                group.group_index == group_index,
                "OpenEnv replay group indices must be contiguous: expected {group_index}, got {}",
                group.group_index
            );
            anyhow::ensure!(
                group.environment_index < self.environments.len(),
                "OpenEnv replay group {group_index} references missing environment {}",
                group.environment_index
            );
            anyhow::ensure!(
                group.reset_payload.is_object(),
                "OpenEnv replay group {group_index} reset payload must be an object"
            );
            anyhow::ensure!(
                !group.reset_observation.done,
                "OpenEnv replay group {group_index} reset observation cannot be terminal"
            );
            anyhow::ensure!(
                !group.candidates.is_empty(),
                "OpenEnv replay group {group_index} must contain candidates"
            );
            for (candidate_index, candidate) in group.candidates.iter().enumerate() {
                anyhow::ensure!(
                    candidate.candidate_index == candidate_index,
                    "OpenEnv replay group {group_index} candidate indices must be contiguous: expected {candidate_index}, got {}",
                    candidate.candidate_index
                );
                anyhow::ensure!(
                    candidate.exchanges.len() <= self.max_steps,
                    "OpenEnv replay group {group_index} candidate {candidate_index} exceeds max_steps"
                );
                anyhow::ensure!(
                    candidate.episode_return.is_finite(),
                    "OpenEnv replay group {group_index} candidate {candidate_index} has a non-finite return"
                );
                anyhow::ensure!(
                    candidate.model_latency_ms.is_finite() && candidate.model_latency_ms >= 0.0,
                    "OpenEnv replay group {group_index} candidate {candidate_index} has invalid model latency"
                );
                validate_candidate(
                    group_index,
                    candidate,
                    self.protocol_error_reward,
                    self.max_steps,
                )?;
            }
        }
        Ok(())
    }
}

fn validate_candidate(
    group_index: usize,
    candidate: &OpenEnvReplayCandidate,
    protocol_error_reward: f64,
    max_steps: usize,
) -> Result<()> {
    let candidate_index = candidate.candidate_index;
    let mut computed_return = 0.0f64;
    let mut computed_recoveries = 0usize;
    let mut saw_done = false;
    let mut exchange_ended_episode = false;
    for (step_index, exchange) in candidate.exchanges.iter().enumerate() {
        anyhow::ensure!(
            exchange.step_index == step_index,
            "OpenEnv replay group {group_index} candidate {candidate_index} step indices must be contiguous: expected {step_index}, got {}",
            exchange.step_index
        );
        anyhow::ensure!(
            exchange.action.is_object(),
            "OpenEnv replay group {group_index} candidate {candidate_index} step {step_index} action must be an object"
        );
        anyhow::ensure!(
            !exchange_ended_episode,
            "OpenEnv replay group {group_index} candidate {candidate_index} has an exchange after a terminal result"
        );
        match &exchange.result {
            OpenEnvReplayExchangeResult::Observation { observation } => {
                computed_return += observation.reward.training_value();
                saw_done = observation.done;
                exchange_ended_episode = observation.done;
            }
            OpenEnvReplayExchangeResult::ProtocolError { error, continued } => {
                anyhow::ensure!(
                    !(*continued && error.code.is_terminal()),
                    "OpenEnv replay group {group_index} candidate {candidate_index} continues after terminal protocol error {}",
                    error.code
                );
                computed_recoveries =
                    computed_recoveries.saturating_add(usize::from(!error.code.is_terminal()));
                computed_return += protocol_error_reward;
                exchange_ended_episode = !continued;
            }
        }
        anyhow::ensure!(
            computed_return.is_finite(),
            "OpenEnv replay group {group_index} candidate {candidate_index} return overflowed"
        );
    }
    if candidate.termination == OpenEnvEpisodeTerminationV1::InvalidModelAction {
        computed_return += protocol_error_reward;
    }
    anyhow::ensure!(
        computed_return.to_bits() == candidate.episode_return.to_bits(),
        "OpenEnv replay group {group_index} candidate {candidate_index} return mismatch: transcript computes {computed_return}, receipt records {}",
        candidate.episode_return
    );
    anyhow::ensure!(
        computed_recoveries == candidate.recoverable_protocol_errors,
        "OpenEnv replay group {group_index} candidate {candidate_index} recoverable-error count mismatch: transcript computes {computed_recoveries}, receipt records {}",
        candidate.recoverable_protocol_errors
    );
    anyhow::ensure!(
        candidate.terminal_done == (candidate.termination == OpenEnvEpisodeTerminationV1::Done),
        "OpenEnv replay group {group_index} candidate {candidate_index} terminal_done disagrees with termination"
    );
    anyhow::ensure!(
        saw_done == candidate.terminal_done,
        "OpenEnv replay group {group_index} candidate {candidate_index} final observation done state disagrees with termination"
    );
    if candidate.termination == OpenEnvEpisodeTerminationV1::ProtocolError {
        anyhow::ensure!(
            matches!(
                candidate.exchanges.last().map(|exchange| &exchange.result),
                Some(OpenEnvReplayExchangeResult::ProtocolError {
                    continued: false,
                    ..
                })
            ),
            "OpenEnv replay group {group_index} candidate {candidate_index} protocol termination requires a final non-continuing error"
        );
    }
    if candidate.termination == OpenEnvEpisodeTerminationV1::MaxSteps {
        anyhow::ensure!(
            candidate.exchanges.len() == max_steps,
            "OpenEnv replay group {group_index} candidate {candidate_index} max_steps termination requires exactly {max_steps} exchanges"
        );
    }
    let ended_on_terminal_protocol_error = matches!(
        candidate.exchanges.last().map(|exchange| &exchange.result),
        Some(OpenEnvReplayExchangeResult::ProtocolError {
            error,
            continued: false,
        }) if error.code.is_terminal()
    );
    anyhow::ensure!(
        candidate.final_state.is_none() == ended_on_terminal_protocol_error,
        "OpenEnv replay group {group_index} candidate {candidate_index} final-state presence disagrees with its terminal protocol state"
    );
    Ok(())
}

pub fn encode_replay(manifest: &OpenEnvReplayManifest) -> Result<Vec<u8>> {
    encode_replay_with_limit(manifest, MAX_OPENENV_ARTIFACT_BYTES)
}

fn encode_replay_with_limit(manifest: &OpenEnvReplayManifest, limit: usize) -> Result<Vec<u8>> {
    manifest.validate()?;
    let mut writer = BoundedVecWriter::new(limit);
    serde_json::to_writer_pretty(&mut writer, manifest)
        .context("serialize bounded OpenEnv replay manifest")?;
    writer
        .write_all(b"\n")
        .context("finish bounded OpenEnv replay manifest")?;
    Ok(writer.into_inner())
}

struct BoundedVecWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedVecWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedVecWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("OpenEnv replay byte count overflow"))?;
        if next > self.limit {
            return Err(std::io::Error::other(format!(
                "OpenEnv replay exceeded the {} byte artifact limit",
                self.limit
            )));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format_sha256(Sha256::digest(bytes).as_slice())
}

fn format_sha256(digest: &[u8]) -> String {
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

/// Hash one server-owned artifact without allocating an artifact-sized buffer.
///
/// Paths are required to name regular, non-symlink files and both metadata and
/// bytes read are capped. This is the publication boundary used before a run
/// advertises an artifact through its durable status manifest.
pub(crate) fn bounded_artifact_metadata(path: &Path) -> Result<(String, usize)> {
    bounded_artifact_metadata_with_limit(path, MAX_OPENENV_ARTIFACT_BYTES)
}

/// Hash one server-owned artifact under a stricter caller-specific cap.
/// Metadata is checked against that cap before hashing begins and the same cap
/// bounds growth while the descriptor is read.
pub(crate) fn bounded_artifact_metadata_with_limit(
    path: &Path,
    limit: usize,
) -> Result<(String, usize)> {
    anyhow::ensure!(
        limit <= MAX_OPENENV_ARTIFACT_BYTES,
        "OpenEnv artifact caller limit {limit} exceeds the service limit {MAX_OPENENV_ARTIFACT_BYTES}"
    );
    let mut file = open_bounded_artifact(path, limit)?;
    hash_bounded_artifact(&mut file, path, limit)
}

/// Open and rewind the exact artifact named by a published manifest entry.
///
/// The returned descriptor is the same descriptor that was hashed. Atomic path
/// replacement after verification therefore cannot substitute a different
/// inode between integrity checking and response streaming.
pub(crate) fn open_verified_artifact(
    path: &Path,
    expected_sha256: &str,
    expected_bytes: usize,
) -> Result<File> {
    ensure_sha256("artifact digest", expected_sha256)?;
    anyhow::ensure!(
        expected_bytes <= MAX_OPENENV_ARTIFACT_BYTES,
        "OpenEnv artifact manifest declares {expected_bytes} bytes; limit is {MAX_OPENENV_ARTIFACT_BYTES}"
    );
    let mut file = open_bounded_artifact(path, expected_bytes)?;
    let (actual_sha256, actual_bytes) =
        hash_bounded_artifact(&mut file, path, MAX_OPENENV_ARTIFACT_BYTES)?;
    anyhow::ensure!(
        actual_bytes == expected_bytes,
        "OpenEnv artifact {} byte count drifted: manifest {expected_bytes}, disk {actual_bytes}",
        path.display()
    );
    anyhow::ensure!(
        actual_sha256 == expected_sha256,
        "OpenEnv artifact {} digest drifted: manifest {expected_sha256}, disk {actual_sha256}",
        path.display()
    );
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind verified OpenEnv artifact {}", path.display()))?;
    Ok(file)
}

fn open_bounded_artifact(path: &Path, limit: usize) -> Result<File> {
    let path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stat OpenEnv artifact path {}", path.display()))?;
    anyhow::ensure!(
        !path_metadata.file_type().is_symlink() && path_metadata.file_type().is_file(),
        "OpenEnv artifact {} must be a regular non-symlink file",
        path.display()
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .with_context(|| format!("open OpenEnv artifact {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat opened OpenEnv artifact {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "OpenEnv artifact {} must remain a regular file after open",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= limit as u64,
        "OpenEnv artifact {} contains {} bytes; limit is {limit}",
        path.display(),
        metadata.len()
    );
    Ok(file)
}

fn hash_bounded_artifact(file: &mut File, path: &Path, limit: usize) -> Result<(String, usize)> {
    let mut hasher = Sha256::new();
    let mut total = 0usize;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read OpenEnv artifact {}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read)
            .context("OpenEnv artifact byte count overflow")?;
        anyhow::ensure!(
            total <= limit,
            "OpenEnv artifact {} grew beyond the {limit} byte limit while hashing",
            path.display()
        );
        hasher.update(&buffer[..read]);
    }
    Ok((format_sha256(hasher.finalize().as_slice()), total))
}

pub async fn connect_and_reset_with_capacity(
    client: &OpenEnvClient,
    reset_payload: &Value,
    capacity_wait: Duration,
) -> Result<(OpenEnvSession, OpenEnvObservation, usize)> {
    connect_and_reset_with_capacity_checked(client, reset_payload, capacity_wait, || Ok(())).await
}

/// Connect and reset while allowing a caller-owned lifecycle to interrupt
/// capacity backoff at every network and sleep boundary.
pub async fn connect_and_reset_with_capacity_checked(
    client: &OpenEnvClient,
    reset_payload: &Value,
    capacity_wait: Duration,
    mut ensure_active: impl FnMut() -> Result<()>,
) -> Result<(OpenEnvSession, OpenEnvObservation, usize)> {
    let started = Instant::now();
    let mut retries = 0usize;
    loop {
        ensure_active()?;
        let mut session = client
            .connect()
            .await
            .with_context(|| format!("connect OpenEnv episode at {}", client.base_url()))?;
        ensure_active()?;
        match session.reset(reset_payload).await {
            Ok(observation) => return Ok((session, observation, retries)),
            Err(OpenEnvClientError::Protocol(error))
                if error.code == OpenEnvErrorCode::CapacityReached =>
            {
                retries = retries.saturating_add(1);
                let _ = session.close().await;
                let elapsed = started.elapsed();
                if elapsed >= capacity_wait {
                    return Err(OpenEnvCapacityTimeout {
                        base_url: client.base_url().to_string(),
                        waited: capacity_wait,
                        retries,
                        active_sessions: error.active_sessions,
                        max_sessions: error.max_sessions,
                    }
                    .into());
                }
                let multiplier = u32::try_from(retries).unwrap_or(u32::MAX);
                let delay = CAPACITY_RETRY_FLOOR
                    .saturating_mul(multiplier)
                    .min(CAPACITY_RETRY_CEILING)
                    .min(capacity_wait.saturating_sub(elapsed));
                let retry_started = Instant::now();
                while retry_started.elapsed() < delay {
                    ensure_active()?;
                    tokio::time::sleep(
                        Duration::from_millis(100)
                            .min(delay.saturating_sub(retry_started.elapsed())),
                    )
                    .await;
                }
            }
            Err(error) => {
                return Err(anyhow!(error))
                    .with_context(|| format!("reset OpenEnv episode at {}", client.base_url()));
            }
        }
    }
}

pub fn verify_openenv_artifacts(
    summary_path: &Path,
    dataset_override: Option<&Path>,
    replay_override: Option<&Path>,
) -> Result<VerifiedOpenEnvArtifacts> {
    let summary_bytes = read_bounded(summary_path, MAX_OPENENV_ARTIFACT_BYTES)?;
    let summary: OpenEnvRolloutSummary = serde_json::from_slice(&summary_bytes)
        .with_context(|| format!("decode OpenEnv summary {}", summary_path.display()))?;
    match summary.schema.as_str() {
        "kiln.openenv-rollout-summary.v2" => anyhow::ensure!(
            summary.reset_options_sha256.is_some() && summary.reset_plan_sha256.is_none(),
            "OpenEnv v2 summary must contain reset_options_sha256 and omit reset_plan_sha256"
        ),
        "kiln.openenv-rollout-summary.v3"
        | "kiln.openenv-rollout-summary.v4"
        | "kiln.openenv-rollout-summary.v5" => anyhow::ensure!(
            summary.reset_options_sha256.is_none() && summary.reset_plan_sha256.is_some(),
            "OpenEnv v3/v4/v5 summary must contain reset_plan_sha256 and omit reset_options_sha256"
        ),
        schema => anyhow::bail!(
            "OpenEnv replay verification requires kiln.openenv-rollout-summary.v2, .v3, .v4, or .v5, got {schema:?}"
        ),
    }
    match summary.schema.as_str() {
        "kiln.openenv-rollout-summary.v2" | "kiln.openenv-rollout-summary.v3" => {
            anyhow::ensure!(
                summary.training_contract.is_none(),
                "OpenEnv summary {} predates embedded training contracts",
                summary.schema
            );
        }
        "kiln.openenv-rollout-summary.v4" | "kiln.openenv-rollout-summary.v5" => {
            match summary.training_contract.as_ref() {
                Some(contract) => {
                    anyhow::ensure!(
                        contract.schema == crate::openenv_cli::OPENENV_TRAINING_CONTRACT_SCHEMA_V1,
                        "OpenEnv summary has unsupported training contract schema {:?}",
                        contract.schema
                    );
                    anyhow::ensure!(
                        contract.effective_config.behavior_policy
                            == kiln_train::BehaviorPolicy::NoImportanceCorrection,
                        "OpenEnv training contract must disable importance correction"
                    );
                    anyhow::ensure!(
                        contract.effective_config.base_adapter == summary.adapter,
                        "OpenEnv training contract behavior adapter differs from the collected policy"
                    );
                    if summary.schema == "kiln.openenv-rollout-summary.v5" {
                        if let Some(contract_policy) = contract.behavior_policy.as_ref() {
                            anyhow::ensure!(
                                Some(contract_policy) == summary.behavior_policy.as_ref(),
                                "OpenEnv training contract behavior-policy revision differs from the collected policy"
                            );
                        }
                    }
                    anyhow::ensure!(
                        contract
                            .effective_config
                            .output_name
                            .as_deref()
                            .is_some_and(|name| !name.is_empty()),
                        "OpenEnv training contract has no output adapter"
                    );
                }
                None => anyhow::ensure!(
                    summary.training_submission.is_none(),
                    "OpenEnv training submission has no admitted training contract"
                ),
            }
        }
        _ => unreachable!("summary schema was validated above"),
    }
    if summary.schema == "kiln.openenv-rollout-summary.v5" {
        let behavior_policy = summary
            .behavior_policy
            .as_ref()
            .context("OpenEnv v5 summary has no behavior-policy identity")?;
        behavior_policy
            .validate()
            .map_err(anyhow::Error::msg)
            .context("OpenEnv v5 summary has an invalid behavior-policy identity")?;
        anyhow::ensure!(
            behavior_policy
                .adapter
                .as_ref()
                .map(|adapter| adapter.name.as_str())
                == summary.adapter.as_deref(),
            "OpenEnv summary behavior-policy adapter differs from its adapter selector"
        );
    } else {
        anyhow::ensure!(
            summary.behavior_policy.is_none(),
            "OpenEnv summary {} predates behavior-policy binding",
            summary.schema
        );
    }

    let dataset_path = resolve_artifact_path(
        summary_path,
        dataset_override,
        Path::new(&summary.output_path),
    );
    let replay_path = resolve_artifact_path(
        summary_path,
        replay_override,
        Path::new(&summary.replay_output_path),
    );
    let dataset_bytes = read_bounded(&dataset_path, MAX_OPENENV_ARTIFACT_BYTES)?;
    let replay_bytes = read_bounded(&replay_path, MAX_OPENENV_ARTIFACT_BYTES)?;
    let dataset_sha256 = sha256_bytes(&dataset_bytes);
    let replay_sha256 = sha256_bytes(&replay_bytes);
    anyhow::ensure!(
        dataset_sha256 == summary.dataset_sha256,
        "OpenEnv dataset digest mismatch: receipt {}, actual {dataset_sha256}",
        summary.dataset_sha256
    );
    anyhow::ensure!(
        replay_sha256 == summary.replay_sha256,
        "OpenEnv replay digest mismatch: receipt {}, actual {replay_sha256}",
        summary.replay_sha256
    );
    anyhow::ensure!(
        dataset_bytes.len() == summary.dataset_bytes,
        "OpenEnv dataset byte count mismatch: receipt {}, actual {}",
        summary.dataset_bytes,
        dataset_bytes.len()
    );
    anyhow::ensure!(
        replay_bytes.len() == summary.replay_bytes,
        "OpenEnv replay byte count mismatch: receipt {}, actual {}",
        summary.replay_bytes,
        replay_bytes.len()
    );

    let groups = decode_jsonl_groups(&dataset_bytes)?;
    let replay: OpenEnvReplayManifest = serde_json::from_slice(&replay_bytes)
        .with_context(|| format!("decode OpenEnv replay {}", replay_path.display()))?;
    replay.validate()?;
    anyhow::ensure!(
        replay.dataset_sha256 == dataset_sha256,
        "OpenEnv replay links dataset {}, but actual dataset is {dataset_sha256}",
        replay.dataset_sha256
    );
    anyhow::ensure!(
        replay.environments == summary.environments,
        "OpenEnv replay environment identities differ from the summary receipt"
    );
    anyhow::ensure!(
        replay.max_steps == summary.max_steps
            && replay.protocol_error_reward.to_bits() == summary.protocol_error_reward.to_bits(),
        "OpenEnv replay controls differ from the summary receipt"
    );
    anyhow::ensure!(
        groups.len() == summary.groups && replay.groups.len() == summary.groups,
        "OpenEnv group count differs across dataset, replay, and summary"
    );
    if matches!(
        summary.schema.as_str(),
        "kiln.openenv-rollout-summary.v3"
            | "kiln.openenv-rollout-summary.v4"
            | "kiln.openenv-rollout-summary.v5"
    ) {
        verify_reset_plan(summary.reset_plan_sha256.as_deref(), &replay)?;
    }

    let mut rollout_count = 0usize;
    let mut exchange_count = 0usize;
    for (group_index, ((group, replay_group), expected_group_index)) in groups
        .iter()
        .zip(&replay.groups)
        .zip(0..summary.groups)
        .enumerate()
    {
        debug_assert_eq!(group_index, expected_group_index);
        anyhow::ensure!(
            group.completions.len() == summary.group_size
                && replay_group.candidates.len() == summary.group_size,
            "OpenEnv group {group_index} candidate count differs across artifacts"
        );
        let expected_seed = summary
            .seed_start
            .checked_add(group_index as u64)
            .context("OpenEnv summary seed range overflow")?;
        anyhow::ensure!(
            replay_group.seed == expected_seed
                && replay_group.environment_index == group_index % replay.environments.len(),
            "OpenEnv group {group_index} seed or round-robin environment assignment differs from the summary controls"
        );
        anyhow::ensure!(
            replay_group
                .reset_payload
                .get("seed")
                .and_then(Value::as_u64)
                == Some(expected_seed),
            "OpenEnv group {group_index} reset payload does not contain its deterministic seed"
        );
        let environment = &replay.environments[replay_group.environment_index];
        anyhow::ensure!(
            group.messages
                == crate::openenv_cli::initial_messages(
                    environment,
                    &replay_group.reset_observation,
                )?,
            "OpenEnv group {group_index} prompt differs from its captured environment/reset"
        );
        for (candidate_index, (rollout, replay_candidate)) in group
            .completions
            .iter()
            .zip(&replay_group.candidates)
            .enumerate()
        {
            let provenance = rollout.openenv.as_ref().with_context(|| {
                format!("OpenEnv group {group_index} candidate {candidate_index} lacks provenance")
            })?;
            let expected_protocol_error =
                replay_candidate
                    .exchanges
                    .last()
                    .and_then(|exchange| match &exchange.result {
                        OpenEnvReplayExchangeResult::ProtocolError {
                            error,
                            continued: false,
                        } => Some(error.code.to_string()),
                        _ => None,
                    });
            anyhow::ensure!(
                rollout.reward.to_bits() == replay_candidate.episode_return.to_bits()
                    && provenance.episode_return.to_bits()
                        == replay_candidate.episode_return.to_bits()
                    && provenance.steps == replay_candidate.exchanges.len()
                    && provenance.seed == replay_group.seed
                    && provenance.termination == replay_candidate.termination
                    && provenance.terminal_done == replay_candidate.terminal_done
                    && provenance.protocol_error_code == expected_protocol_error
                    && provenance.environment_name == environment.identity.metadata.name
                    && provenance.environment_base_url == environment.identity.base_url
                    && provenance.openapi_version == environment.identity.openapi_version
                    && provenance.discovery_sha256.as_ref().is_none_or(|digest| {
                        environment.identity.discovery_sha256.as_ref() == Some(digest)
                    })
                    && provenance.environment_schema_sha256 == environment.identity.schema_sha256
                    && provenance.action_schema_sha256 == sha256_json(&environment.schema.action)?
                    && provenance.reset_sha256 == sha256_json(&replay_group.reset_payload)?,
                "OpenEnv group {group_index} candidate {candidate_index} provenance differs from replay"
            );
            if summary.schema == "kiln.openenv-rollout-summary.v5" {
                anyhow::ensure!(
                    provenance.behavior_policy.as_ref() == summary.behavior_policy.as_ref(),
                    "OpenEnv group {group_index} candidate {candidate_index} behavior policy differs from the summary receipt"
                );
            }
            verify_trajectory(group_index, candidate_index, rollout, replay_candidate)?;
            let record_index = group_index
                .checked_mul(summary.group_size)
                .and_then(|index| index.checked_add(candidate_index))
                .context("OpenEnv summary rollout index overflow")?;
            let record = summary.rollouts.get(record_index).with_context(|| {
                format!("OpenEnv summary lacks group {group_index} candidate {candidate_index}")
            })?;
            anyhow::ensure!(
                record.group_index == group_index
                    && record.candidate_index == candidate_index
                    && record.environment_name == environment.identity.metadata.name
                    && record.environment_url == environment.identity.base_url
                    && record.seed == replay_group.seed
                    && record.steps == replay_candidate.exchanges.len()
                    && record.episode_return.to_bits() == replay_candidate.episode_return.to_bits()
                    && record.terminal_done == replay_candidate.terminal_done
                    && record.termination == replay_candidate.termination
                    && record.protocol_error_code == expected_protocol_error
                    && record.recoverable_protocol_errors
                        == replay_candidate.recoverable_protocol_errors
                    && record.capacity_retries == replay_candidate.capacity_retries
                    && record.model_tokens == replay_candidate.model_tokens
                    && record.model_latency_ms.to_bits()
                        == replay_candidate.model_latency_ms.to_bits(),
                "OpenEnv summary record for group {group_index} candidate {candidate_index} differs from replay/provenance"
            );
            verify_recovery_budget(
                group_index,
                candidate_index,
                replay_candidate,
                summary.max_recoverable_errors,
            )?;
            rollout_count = rollout_count.saturating_add(1);
            exchange_count = exchange_count.saturating_add(replay_candidate.exchanges.len());
        }
    }
    anyhow::ensure!(
        rollout_count == summary.rollout_count && summary.rollouts.len() == summary.rollout_count,
        "OpenEnv rollout count differs across artifacts"
    );
    let computed_stats = crate::openenv_cli::summarize_rollouts(&summary.rollouts);
    anyhow::ensure!(
        computed_stats == summary.stats,
        "OpenEnv summary aggregate statistics do not match its rollout records: recorded={:?}, computed={computed_stats:?}",
        summary.stats
    );

    let report = OpenEnvVerificationReport {
        schema: OPENENV_VERIFICATION_SCHEMA_V1.to_string(),
        summary_path: summary_path.display().to_string(),
        dataset_path: dataset_path.display().to_string(),
        replay_path: replay_path.display().to_string(),
        dataset_sha256,
        replay_sha256,
        groups: groups.len(),
        rollouts: rollout_count,
        environment_exchanges: exchange_count,
    };
    Ok(VerifiedOpenEnvArtifacts {
        summary,
        groups,
        replay,
        report,
    })
}

fn verify_reset_plan(expected_digest: Option<&str>, replay: &OpenEnvReplayManifest) -> Result<()> {
    let mut plan = vec![None; replay.environments.len()];
    for group in &replay.groups {
        let mut template = group.reset_payload.as_object().cloned().with_context(|| {
            format!(
                "OpenEnv group {} reset payload is not one JSON object",
                group.group_index
            )
        })?;
        template.remove("seed");
        let template = Value::Object(template);
        let slot = plan.get_mut(group.environment_index).with_context(|| {
            format!(
                "OpenEnv group {} references unavailable environment {}",
                group.group_index, group.environment_index
            )
        })?;
        if let Some(previous) = slot.as_ref() {
            anyhow::ensure!(
                previous == &template,
                "OpenEnv reset template changed across groups for environment {}",
                group.environment_index
            );
        } else {
            *slot = Some(template);
        }
    }
    let plan = plan
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.with_context(|| {
                format!(
                    "OpenEnv v3 receipt cannot be verified because environment {index} was never exercised"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let actual = sha256_json(&Value::Array(plan))?;
    anyhow::ensure!(
        expected_digest == Some(actual.as_str()),
        "OpenEnv reset-plan digest mismatch: receipt {}, replay {actual}",
        expected_digest.unwrap_or("<missing>")
    );
    Ok(())
}

fn verify_recovery_budget(
    group_index: usize,
    candidate_index: usize,
    candidate: &OpenEnvReplayCandidate,
    max_recoverable_errors: usize,
) -> Result<()> {
    let mut recoverable_errors = 0usize;
    for exchange in &candidate.exchanges {
        let OpenEnvReplayExchangeResult::ProtocolError { error, continued } = &exchange.result
        else {
            continue;
        };
        let expected_continued =
            !error.code.is_terminal() && recoverable_errors < max_recoverable_errors;
        anyhow::ensure!(
            *continued == expected_continued,
            "OpenEnv group {group_index} candidate {candidate_index} step {} continuation disagrees with the recovery budget",
            exchange.step_index
        );
        recoverable_errors =
            recoverable_errors.saturating_add(usize::from(!error.code.is_terminal()));
    }
    anyhow::ensure!(
        recoverable_errors == candidate.recoverable_protocol_errors,
        "OpenEnv group {group_index} candidate {candidate_index} recoverable-error total disagrees with the transcript"
    );
    Ok(())
}

fn verify_trajectory(
    group_index: usize,
    candidate_index: usize,
    rollout: &ScoredRollout,
    replay: &OpenEnvReplayCandidate,
) -> Result<()> {
    let invalid_tail =
        usize::from(replay.termination == OpenEnvEpisodeTerminationV1::InvalidModelAction);
    let expected_segments = replay
        .exchanges
        .len()
        .checked_add(invalid_tail)
        .and_then(|pairs| pairs.checked_mul(2))
        .context("OpenEnv trajectory segment count overflow")?;
    anyhow::ensure!(
        rollout.trajectory.len() == expected_segments,
        "OpenEnv group {group_index} candidate {candidate_index} trajectory has {} segments; replay requires {expected_segments}",
        rollout.trajectory.len()
    );
    let flattened_actions = rollout
        .trajectory
        .iter()
        .filter(|segment| segment.kind == TurnKind::Action)
        .map(|segment| segment.content.as_str())
        .collect::<Vec<_>>()
        .join("<TURN_BREAK>");
    anyhow::ensure!(
        rollout.text == flattened_actions,
        "OpenEnv group {group_index} candidate {candidate_index} legacy text differs from its action trajectory"
    );

    for exchange in &replay.exchanges {
        let offset = exchange.step_index * 2;
        let action = &rollout.trajectory[offset];
        let result = &rollout.trajectory[offset + 1];
        anyhow::ensure!(
            action.role == "assistant"
                && action.kind == TurnKind::Action
                && action.tool_call_id.is_none()
                && action.warning_prefix_len.is_none(),
            "OpenEnv group {group_index} candidate {candidate_index} step {} has a malformed action segment",
            exchange.step_index
        );
        anyhow::ensure!(
            crate::openenv_cli::parse_model_action(&action.content).map_err(anyhow::Error::msg)?
                == exchange.action,
            "OpenEnv group {group_index} candidate {candidate_index} step {} action differs from replay",
            exchange.step_index
        );
        anyhow::ensure!(
            result.role == "tool"
                && result.kind == TurnKind::Observation
                && result.tool_call_id.is_none(),
            "OpenEnv group {group_index} candidate {candidate_index} step {} has a malformed result segment",
            exchange.step_index
        );
        match &exchange.result {
            OpenEnvReplayExchangeResult::Observation { observation } => {
                let policy_content =
                    crate::openenv_cli::model_observation_content("step result", observation)?;
                let legacy_content = serde_json::to_string(observation)?;
                anyhow::ensure!(
                    (result.content == policy_content || result.content == legacy_content)
                        && result.warning_prefix_len.is_none(),
                    "OpenEnv group {group_index} candidate {candidate_index} step {} result differs from replay",
                    exchange.step_index
                );
            }
            OpenEnvReplayExchangeResult::ProtocolError { error, continued } => {
                let actual: Value =
                    serde_json::from_str(&result.content).with_context(|| {
                        format!(
                            "decode OpenEnv group {group_index} candidate {candidate_index} step {} result segment",
                            exchange.step_index
                        )
                    })?;
                let expected = serde_json::json!({
                    "openenv_error": error,
                    "recoverable": continued,
                    "done": !continued,
                });
                anyhow::ensure!(
                    actual == expected && result.warning_prefix_len == Some(result.content.len()),
                    "OpenEnv group {group_index} candidate {candidate_index} step {} result differs from replay",
                    exchange.step_index
                );
            }
        }
    }

    if invalid_tail == 1 {
        let offset = replay.exchanges.len() * 2;
        let action = &rollout.trajectory[offset];
        let error = &rollout.trajectory[offset + 1];
        let error_value: Value =
            serde_json::from_str(&error.content).context("decode invalid-model-action segment")?;
        anyhow::ensure!(
            action.role == "assistant"
                && action.kind == TurnKind::Action
                && error.role == "tool"
                && error.kind == TurnKind::Observation
                && error.warning_prefix_len == Some(error.content.len())
                && error_value.pointer("/openenv_harness_error/code")
                    == Some(&Value::String("INVALID_MODEL_ACTION".to_string()))
                && error_value.get("done") == Some(&Value::Bool(true)),
            "OpenEnv group {group_index} candidate {candidate_index} invalid-model-action tail is malformed"
        );
    }
    Ok(())
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize OpenEnv value for SHA-256")?;
    Ok(sha256_bytes(&bytes))
}

pub async fn replay_openenv(
    manifest: &OpenEnvReplayManifest,
    replay_sha256: String,
    concurrency: usize,
    capacity_wait: Duration,
    credential_envs: &[Option<String>],
) -> Result<OpenEnvReplayRunReport> {
    manifest.validate()?;
    anyhow::ensure!(
        concurrency > 0,
        "OpenEnv replay concurrency must be positive"
    );
    anyhow::ensure!(
        credential_envs.is_empty() || credential_envs.len() == manifest.environments.len(),
        "OpenEnv replay credentials must be empty or contain one entry per captured environment"
    );

    let mut clients = Vec::with_capacity(manifest.environments.len());
    for (index, expected) in manifest.environments.iter().enumerate() {
        let credential_env = credential_envs
            .get(index)
            .and_then(|credential| credential.as_deref());
        let client = openenv_client(&expected.identity.base_url, credential_env)?;
        client
            .revalidate(expected)
            .await
            .with_context(|| format!("revalidate replay target {}", client.base_url()))?;
        clients.push(client);
    }

    let mut rollouts = 0usize;
    let mut exchanges = 0usize;
    let mut capacity_retries = 0usize;
    let mut prefix_only = 0usize;
    for group in &manifest.groups {
        let client = clients[group.environment_index].clone();
        let results = stream::iter(group.candidates.iter().cloned())
            .map(|candidate| {
                replay_candidate(
                    client.clone(),
                    group.group_index,
                    group.reset_payload.clone(),
                    group.reset_observation.clone(),
                    candidate,
                    capacity_wait,
                )
            })
            .buffer_unordered(concurrency.min(group.candidates.len()))
            .collect::<Vec<_>>()
            .await;
        for result in results {
            let result = result?;
            rollouts = rollouts.saturating_add(1);
            exchanges = exchanges.saturating_add(result.exchanges);
            capacity_retries = capacity_retries.saturating_add(result.capacity_retries);
            prefix_only = prefix_only.saturating_add(usize::from(result.prefix_only));
        }
    }
    Ok(OpenEnvReplayRunReport {
        schema: OPENENV_REPLAY_RUN_SCHEMA_V1.to_string(),
        replay_sha256,
        environments: manifest.environments.len(),
        groups: manifest.groups.len(),
        rollouts,
        environment_exchanges: exchanges,
        capacity_retries,
        environment_prefix_only_rollouts: prefix_only,
    })
}

struct CandidateReplayResult {
    exchanges: usize,
    capacity_retries: usize,
    prefix_only: bool,
}

async fn replay_candidate(
    client: OpenEnvClient,
    group_index: usize,
    reset_payload: Value,
    expected_reset: OpenEnvObservation,
    candidate: OpenEnvReplayCandidate,
    capacity_wait: Duration,
) -> Result<CandidateReplayResult> {
    let candidate_index = candidate.candidate_index;
    let (mut session, actual_reset, capacity_retries) =
        connect_and_reset_with_capacity(&client, &reset_payload, capacity_wait).await?;
    anyhow::ensure!(
        actual_reset == expected_reset,
        "OpenEnv replay drift at group {group_index} candidate {candidate_index} reset: expected {}, got {}",
        serde_json::to_string(&expected_reset)?,
        serde_json::to_string(&actual_reset)?
    );
    let mut terminal_protocol_error = false;
    for exchange in &candidate.exchanges {
        let context = format!(
            "group {group_index} candidate {candidate_index} step {}",
            exchange.step_index
        );
        match (&exchange.result, session.step(&exchange.action).await) {
            (
                OpenEnvReplayExchangeResult::Observation {
                    observation: expected,
                },
                Ok(actual),
            ) => {
                anyhow::ensure!(
                    actual == *expected,
                    "OpenEnv replay drift at {context}: expected {}, got {}",
                    serde_json::to_string(expected)?,
                    serde_json::to_string(&actual)?
                );
            }
            (
                OpenEnvReplayExchangeResult::ProtocolError {
                    error: expected,
                    continued: _,
                },
                Err(OpenEnvClientError::Protocol(actual)),
            ) => {
                anyhow::ensure!(
                    actual == *expected,
                    "OpenEnv replay drift at {context}: expected protocol error {}, got {}",
                    serde_json::to_string(expected)?,
                    serde_json::to_string(&actual)?
                );
                terminal_protocol_error = expected.code.is_terminal();
            }
            (expected, actual) => {
                return Err(anyhow!(
                    "OpenEnv replay drift at {context}: expected {}, got {}",
                    serde_json::to_string(expected)?,
                    match actual {
                        Ok(observation) => serde_json::to_string(&observation)?,
                        Err(error) => error.to_string(),
                    }
                ));
            }
        }
    }
    if let Some(expected_state) = &candidate.final_state {
        anyhow::ensure!(
            !terminal_protocol_error,
            "OpenEnv replay transcript requests state after a terminal protocol error"
        );
        let actual_state = session
            .state()
            .await
            .with_context(|| format!("read final OpenEnv replay state for group {group_index} candidate {candidate_index}"))?;
        anyhow::ensure!(
            actual_state == *expected_state,
            "OpenEnv replay drift at group {group_index} candidate {candidate_index} final state: expected {}, got {}",
            serde_json::to_string(expected_state)?,
            serde_json::to_string(&actual_state)?
        );
    }
    let _ = session.close().await;
    Ok(CandidateReplayResult {
        exchanges: candidate.exchanges.len(),
        capacity_retries,
        prefix_only: candidate.termination == OpenEnvEpisodeTerminationV1::InvalidModelAction,
    })
}

fn decode_jsonl_groups(bytes: &[u8]) -> Result<Vec<AgenticGroup>> {
    let text = std::str::from_utf8(bytes).context("OpenEnv dataset is not UTF-8 JSONL")?;
    let mut groups = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        anyhow::ensure!(
            !line.trim().is_empty(),
            "OpenEnv dataset contains a blank JSONL record at line {}",
            line_index + 1
        );
        groups
            .push(serde_json::from_str(line).with_context(|| {
                format!("decode OpenEnv dataset JSONL line {}", line_index + 1)
            })?);
    }
    anyhow::ensure!(!groups.is_empty(), "OpenEnv dataset is empty");
    Ok(groups)
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open OpenEnv artifact {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat OpenEnv artifact {}", path.display()))?;
    anyhow::ensure!(
        metadata.len() <= limit as u64,
        "OpenEnv artifact {} exceeds the {limit} byte limit",
        path.display()
    );
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read OpenEnv artifact {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= limit,
        "OpenEnv artifact {} grew beyond the {limit} byte limit while being read",
        path.display()
    );
    Ok(bytes)
}

fn resolve_artifact_path(
    summary_path: &Path,
    override_path: Option<&Path>,
    recorded_path: &Path,
) -> PathBuf {
    if let Some(path) = override_path {
        return path.to_path_buf();
    }
    if recorded_path.is_absolute() || recorded_path.exists() {
        return recorded_path.to_path_buf();
    }
    let sibling = summary_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(
            recorded_path
                .file_name()
                .unwrap_or(recorded_path.as_os_str()),
        );
    if sibling.exists() {
        sibling
    } else {
        recorded_path.to_path_buf()
    }
}

fn ensure_sha256(label: &str, value: &str) -> Result<()> {
    let digest = value.strip_prefix("sha256:").unwrap_or_default();
    anyhow::ensure!(
        digest.len() == 64
            && digest
                .chars()
                .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character)),
        "OpenEnv replay {label} must be a sha256:<64 hex> digest"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_openenv::{OpenEnvIdentity, OpenEnvMetadata, OpenEnvReward, OpenEnvSchema};
    use serde_json::json;

    fn inspection() -> OpenEnvInspection {
        OpenEnvInspection {
            identity: OpenEnvIdentity {
                schema: "kiln.openenv-identity.v1".to_string(),
                client_profile: OPENENV_CLIENT_PROFILE.to_string(),
                base_url: "http://127.0.0.1:8000".to_string(),
                websocket_url: "ws://127.0.0.1:8000/ws".to_string(),
                authentication: kiln_openenv::OpenEnvAuthentication::None,
                openapi_version: Some("1".to_string()),
                environments: vec!["counter".to_string()],
                metadata: OpenEnvMetadata {
                    name: "counter".to_string(),
                    description: "count".to_string(),
                    readme_content: None,
                    version: None,
                    author: None,
                    documentation_url: None,
                },
                schema_sha256: format!("sha256:{}", "a".repeat(64)),
                discovery_sha256: Some(format!("sha256:{}", "d".repeat(64))),
            },
            schema: OpenEnvSchema {
                action: json!({"type": "object"}),
                observation: json!({"type": "object"}),
                state: json!({"type": "object"}),
            },
        }
    }

    fn observation(reward: OpenEnvReward, done: bool) -> OpenEnvObservation {
        OpenEnvObservation {
            observation: json!({"count": 1}),
            reward,
            done,
            metadata: None,
        }
    }

    #[test]
    fn manifest_validates_returns_and_recovery_semantics() {
        let manifest = OpenEnvReplayManifest {
            schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
            client_profile: OPENENV_CLIENT_PROFILE.to_string(),
            dataset_sha256: format!("sha256:{}", "b".repeat(64)),
            protocol_error_reward: -1.0,
            max_steps: 3,
            environments: vec![inspection()],
            groups: vec![OpenEnvReplayGroup {
                group_index: 0,
                environment_index: 0,
                seed: 7,
                reset_payload: json!({"seed": 7}),
                reset_observation: observation(OpenEnvReward::Integer(0), false),
                candidates: vec![OpenEnvReplayCandidate {
                    candidate_index: 0,
                    exchanges: vec![
                        OpenEnvReplayExchange {
                            step_index: 0,
                            action: json!({"value": "bad"}),
                            result: OpenEnvReplayExchangeResult::ProtocolError {
                                error: OpenEnvProtocolError {
                                    code: OpenEnvErrorCode::ValidationError,
                                    message: "bad".to_string(),
                                    errors: None,
                                    active_sessions: None,
                                    max_sessions: None,
                                    factory_name: None,
                                },
                                continued: true,
                            },
                        },
                        OpenEnvReplayExchange {
                            step_index: 1,
                            action: json!({"value": 1}),
                            result: OpenEnvReplayExchangeResult::Observation {
                                observation: observation(OpenEnvReward::Integer(2), true),
                            },
                        },
                    ],
                    final_state: Some(json!({"count": 1})),
                    episode_return: 1.0,
                    terminal_done: true,
                    termination: OpenEnvEpisodeTerminationV1::Done,
                    recoverable_protocol_errors: 1,
                    capacity_retries: 0,
                    model_tokens: 8,
                    model_latency_ms: 2.0,
                }],
            }],
        };
        manifest.validate().unwrap();
        let encoded = encode_replay(&manifest).unwrap();
        assert!(encoded.ends_with(b"\n"));
        let decoded: OpenEnvReplayManifest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, manifest);
        let error = encode_replay_with_limit(&manifest, 32).unwrap_err();
        assert!(error.to_string().contains("bounded OpenEnv replay"));
        assert!(format!("{error:#}").contains("32 byte artifact limit"));
    }

    #[test]
    fn server_artifact_metadata_streams_and_verified_open_rewinds() {
        let contents = b"content-addressed OpenEnv artifact\n";
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents).unwrap();
        let expected = sha256_bytes(contents);
        let (sha256, bytes) = bounded_artifact_metadata(file.path()).unwrap();
        assert_eq!(sha256, expected);
        assert_eq!(bytes, contents.len());
        assert_eq!(
            bounded_artifact_metadata_with_limit(file.path(), contents.len()).unwrap(),
            (expected.clone(), contents.len())
        );
        let error =
            bounded_artifact_metadata_with_limit(file.path(), contents.len() - 1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("limit is {}", contents.len() - 1)),
            "{error:#}"
        );

        let mut verified = open_verified_artifact(file.path(), &expected, bytes).unwrap();
        let mut round_trip = Vec::new();
        verified.read_to_end(&mut round_trip).unwrap();
        assert_eq!(round_trip, contents);

        assert!(
            open_verified_artifact(file.path(), &format!("sha256:{}", "0".repeat(64)), bytes)
                .unwrap_err()
                .to_string()
                .contains("digest drifted")
        );
        assert!(
            open_verified_artifact(file.path(), &expected, bytes - 1)
                .unwrap_err()
                .to_string()
                .contains(&format!("contains {bytes} bytes"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn server_artifact_metadata_rejects_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        let link = directory.path().join("artifact.json");
        std::fs::write(&target, b"{}\n").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = bounded_artifact_metadata(&link).unwrap_err();
        assert!(
            error.to_string().contains("regular non-symlink file"),
            "{error:#}"
        );
    }

    #[test]
    fn manifest_rejects_return_tampering() {
        let mut manifest = OpenEnvReplayManifest {
            schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
            client_profile: OPENENV_CLIENT_PROFILE.to_string(),
            dataset_sha256: format!("sha256:{}", "b".repeat(64)),
            protocol_error_reward: -1.0,
            max_steps: 1,
            environments: vec![inspection()],
            groups: vec![OpenEnvReplayGroup {
                group_index: 0,
                environment_index: 0,
                seed: 0,
                reset_payload: json!({"seed": 0}),
                reset_observation: observation(OpenEnvReward::Integer(0), false),
                candidates: vec![OpenEnvReplayCandidate {
                    candidate_index: 0,
                    exchanges: vec![],
                    final_state: Some(json!({"count": 0})),
                    episode_return: -1.0,
                    terminal_done: false,
                    termination: OpenEnvEpisodeTerminationV1::InvalidModelAction,
                    recoverable_protocol_errors: 0,
                    capacity_retries: 0,
                    model_tokens: 1,
                    model_latency_ms: 1.0,
                }],
            }],
        };
        manifest.validate().unwrap();
        manifest.groups[0].candidates[0].episode_return = 0.0;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn v3_reset_plan_is_reconstructed_in_environment_order() {
        let replay = OpenEnvReplayManifest {
            schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
            client_profile: OPENENV_CLIENT_PROFILE.to_string(),
            dataset_sha256: format!("sha256:{}", "b".repeat(64)),
            protocol_error_reward: -1.0,
            max_steps: 1,
            environments: vec![inspection(), inspection()],
            groups: vec![
                OpenEnvReplayGroup {
                    group_index: 0,
                    environment_index: 0,
                    seed: 7,
                    reset_payload: json!({"difficulty": "hard", "seed": 7}),
                    reset_observation: observation(OpenEnvReward::Null, false),
                    candidates: Vec::new(),
                },
                OpenEnvReplayGroup {
                    group_index: 1,
                    environment_index: 1,
                    seed: 8,
                    reset_payload: json!({"seed": 8, "split": "train"}),
                    reset_observation: observation(OpenEnvReward::Null, false),
                    candidates: Vec::new(),
                },
            ],
        };
        let expected = sha256_json(&json!([
            {"difficulty": "hard"},
            {"split": "train"}
        ]))
        .unwrap();
        verify_reset_plan(Some(&expected), &replay).unwrap();
        assert!(verify_reset_plan(Some(&format!("sha256:{}", "0".repeat(64))), &replay).is_err());

        let mut inconsistent = replay.clone();
        inconsistent.groups.push(OpenEnvReplayGroup {
            group_index: 2,
            environment_index: 0,
            seed: 9,
            reset_payload: json!({"difficulty": "easy", "seed": 9}),
            reset_observation: observation(OpenEnvReward::Null, false),
            candidates: Vec::new(),
        });
        assert!(verify_reset_plan(Some(&expected), &inconsistent).is_err());
    }

    #[test]
    fn exhausted_recoverable_error_is_counted_but_does_not_continue() {
        let candidate = OpenEnvReplayCandidate {
            candidate_index: 0,
            exchanges: vec![OpenEnvReplayExchange {
                step_index: 0,
                action: json!({"value": "bad"}),
                result: OpenEnvReplayExchangeResult::ProtocolError {
                    error: OpenEnvProtocolError {
                        code: OpenEnvErrorCode::ValidationError,
                        message: "bad".to_string(),
                        errors: None,
                        active_sessions: None,
                        max_sessions: None,
                        factory_name: None,
                    },
                    continued: false,
                },
            }],
            final_state: Some(json!({"count": 0})),
            episode_return: -1.0,
            terminal_done: false,
            termination: OpenEnvEpisodeTerminationV1::ProtocolError,
            recoverable_protocol_errors: 1,
            capacity_retries: 0,
            model_tokens: 2,
            model_latency_ms: 1.0,
        };
        let manifest = OpenEnvReplayManifest {
            schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
            client_profile: OPENENV_CLIENT_PROFILE.to_string(),
            dataset_sha256: format!("sha256:{}", "b".repeat(64)),
            protocol_error_reward: -1.0,
            max_steps: 1,
            environments: vec![inspection()],
            groups: vec![OpenEnvReplayGroup {
                group_index: 0,
                environment_index: 0,
                seed: 0,
                reset_payload: json!({"seed": 0}),
                reset_observation: observation(OpenEnvReward::Integer(0), false),
                candidates: vec![candidate.clone()],
            }],
        };
        manifest.validate().unwrap();
        verify_recovery_budget(0, 0, &candidate, 0).unwrap();
        assert!(verify_recovery_budget(0, 0, &candidate, 1).is_err());
    }
}

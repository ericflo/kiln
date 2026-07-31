//! Native OpenEnv discovery, rollout, and training control plane.
//!
//! The HTTP/dashboard lifecycle is deliberately a thin orchestration layer
//! around the same protocol client, collector, chat handler, artifact writer,
//! and GRPO queue admission used by Kiln's CLI and training APIs.

use std::collections::HashMap;
use std::io::Write;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{StreamExt, stream};
use kiln_eval::EvalJobState;
use kiln_openenv::{OpenEnvClient, OpenEnvIdentity, OpenEnvInspection};
use kiln_train::{BehaviorPolicy, GrpoConfig, GrpoRequest, TrainingResponse, TrainingState};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::OpenEnvConfig;
use crate::error::ApiError;
use crate::openenv_cli::{
    OpenEnvCollectionControl, OpenEnvCollectionProgress, OpenEnvPolicyTransport,
    OpenEnvRolloutOptions, OpenEnvRolloutSummary, collect_openenv_rollouts_with_policy,
    validate_options, write_openenv_outputs, write_summary_atomic,
};
use crate::recent_requests::now_unix_ms;
use crate::state::AppState;

const OPENENV_RUN_SCHEMA_V1: &str = "kiln.openenv-run.v1";
const OPENENV_RUN_SCHEMA_V2: &str = "kiln.openenv-run.v2";
const OPENENV_RUN_LIST_SCHEMA_V2: &str = "kiln.openenv-run-list.v2";
const OPENENV_INSPECTION_SCHEMA_V1: &str = "kiln.openenv-inspection.v1";
const OPENENV_API_BODY_LIMIT: usize = 1024 * 1024;
const MAX_ENVIRONMENTS: usize = 64;
const MAX_PERSISTED_STATUS_BYTES: u64 = 2 * 1024 * 1024;
const ARTIFACT_CHUNK_BYTES: usize = 64 * 1024;
const LIFECYCLE_POLL_INTERVAL: Duration = Duration::from_millis(500);
const POST_EVAL_PUBLICATION_GRACE: Duration = Duration::from_secs(5);
const POST_EVAL_GATE_TIMEOUT: Duration = Duration::from_secs(300);

fn default_groups() -> usize {
    8
}

fn default_group_size() -> usize {
    4
}

fn default_max_steps() -> usize {
    8
}

fn default_concurrency() -> usize {
    4
}

fn default_max_action_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    1.0
}

fn default_protocol_error_reward() -> f64 {
    -1.0
}

fn default_max_recoverable_errors() -> usize {
    3
}

fn default_capacity_wait_seconds() -> u64 {
    300
}

fn default_adapter() -> String {
    "base".to_string()
}

fn default_auto_load() -> bool {
    true
}

fn default_reset_options() -> Value {
    Value::Object(Map::new())
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvRunKind {
    #[default]
    Rollout,
    Train,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvRunState {
    Queued,
    Discovering,
    Collecting,
    Submitting,
    RolloutReady,
    TrainingQueued,
    TrainingRunning,
    PostEvaluating,
    Completed,
    Failed,
    Cancelled,
}

impl OpenEnvRunState {
    fn unconditionally_terminal(self) -> bool {
        matches!(
            self,
            Self::RolloutReady | Self::Completed | Self::Failed | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRunRequest {
    #[serde(default)]
    pub kind: OpenEnvRunKind,
    #[serde(alias = "environments")]
    pub environment_urls: Vec<String>,
    #[serde(default = "default_adapter")]
    pub adapter: String,
    #[serde(default = "default_groups")]
    pub groups: usize,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    #[serde(default)]
    pub seed_start: u64,
    #[serde(default = "default_reset_options")]
    pub reset_options: Value,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_max_action_tokens")]
    pub max_action_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default = "default_protocol_error_reward")]
    pub protocol_error_reward: f64,
    #[serde(default = "default_max_recoverable_errors")]
    pub max_recoverable_errors: usize,
    #[serde(default = "default_capacity_wait_seconds")]
    pub capacity_wait_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_adapter: Option<String>,
    /// Full native GRPO configuration. Kiln owns and overrides
    /// `output_name`, `auto_load`, `base_adapter`, and `behavior_policy` so
    /// the rollout policy and submitted training job cannot diverge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<GrpoConfig>,
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvRunProgress {
    pub groups_completed: usize,
    pub groups_total: usize,
    pub rollouts_completed: usize,
    pub rollouts_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvArtifact {
    pub kind: String,
    pub url: String,
    pub sha256: String,
    pub bytes: usize,
}

/// Authoritative projection of the native trainer owned by this OpenEnv run.
///
/// The trainer remains the source of truth; this bounded snapshot makes an
/// OpenEnv workflow observable without forcing clients to stitch together two
/// unrelated control planes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvTrainingStatus {
    pub job_id: String,
    pub state: TrainingState,
    pub progress: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_eval_job_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval_verdict: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Bounded projection of one post-training evaluation linked to the trainer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvPostEvalStatus {
    pub job_id: String,
    pub suite_name: String,
    pub state: EvalJobState,
    pub examples_completed: u32,
    pub examples_total: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headline_accuracy: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenEnvRunStatus {
    pub schema: String,
    pub run_id: String,
    pub kind: OpenEnvRunKind,
    pub state: OpenEnvRunState,
    pub request: OpenEnvRunRequest,
    pub submitted_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_unix_ms: Option<u64>,
    pub progress: OpenEnvRunProgress,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environments: Vec<OpenEnvIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<OpenEnvArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_submission: Option<TrainingResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<OpenEnvTrainingStatus>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub post_evaluations: Vec<OpenEnvPostEvalStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl OpenEnvRunStatus {
    fn terminal(&self) -> bool {
        self.state.unconditionally_terminal()
            // Before v2, `training_queued` explicitly meant the OpenEnv
            // orchestrator had handed ownership off and finished. Preserve
            // that historical record on upgrade while v2 runs keep following
            // the trainer to a real terminal outcome.
            || (self.schema == OPENENV_RUN_SCHEMA_V1
                && self.state == OpenEnvRunState::TrainingQueued)
    }
}

#[derive(Debug, Serialize)]
struct OpenEnvRunList {
    schema: &'static str,
    runs: Vec<OpenEnvRunStatus>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenEnvInspectRequest {
    #[serde(alias = "environments")]
    environment_urls: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OpenEnvInspectResponse {
    schema: &'static str,
    environments: Vec<OpenEnvInspection>,
}

struct TrackedOpenEnvRun {
    status: OpenEnvRunStatus,
    cancel: Arc<AtomicBool>,
}

/// Bounded, persisted registry for server-owned OpenEnv rollout runs.
pub struct OpenEnvRunRegistry {
    root: PathBuf,
    policy: OpenEnvConfig,
    runs: RwLock<HashMap<String, TrackedOpenEnvRun>>,
}

impl std::fmt::Debug for OpenEnvRunRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvRunRegistry")
            .field("root", &self.root)
            .field("policy", &self.policy)
            .field("tracked_runs", &self.runs.read().unwrap().len())
            .finish()
    }
}

impl OpenEnvRunRegistry {
    pub fn new(adapter_dir: PathBuf) -> Self {
        Self {
            root: adapter_dir.join(".openenv").join("runs"),
            policy: OpenEnvConfig::default(),
            runs: RwLock::new(HashMap::new()),
        }
    }

    pub fn open(adapter_dir: PathBuf, policy: OpenEnvConfig) -> Result<Self> {
        let root = adapter_dir.join(".openenv").join("runs");
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create OpenEnv run root {}", root.display()))?;
        let registry = Self {
            root,
            policy,
            runs: RwLock::new(HashMap::new()),
        };
        registry.restore()?;
        Ok(registry)
    }

    pub fn policy(&self) -> &OpenEnvConfig {
        &self.policy
    }

    fn restore(&self) -> Result<()> {
        let mut restored = Vec::new();
        for entry in std::fs::read_dir(&self.root)
            .with_context(|| format!("scan OpenEnv run root {}", self.root.display()))?
        {
            let entry = entry.context("read OpenEnv run directory entry")?;
            if !entry
                .file_type()
                .context("read OpenEnv run entry type")?
                .is_dir()
            {
                continue;
            }
            let status_path = entry.path().join("run.json");
            let Ok(metadata) = std::fs::metadata(&status_path) else {
                continue;
            };
            if metadata.len() > MAX_PERSISTED_STATUS_BYTES {
                tracing::warn!(
                    path = %status_path.display(),
                    bytes = metadata.len(),
                    "ignoring oversized persisted OpenEnv run status"
                );
                continue;
            }
            let bytes = std::fs::read(&status_path)
                .with_context(|| format!("read OpenEnv status {}", status_path.display()))?;
            let mut status: OpenEnvRunStatus = serde_json::from_slice(&bytes)
                .with_context(|| format!("decode OpenEnv status {}", status_path.display()))?;
            if !status.terminal() {
                status.state = OpenEnvRunState::Failed;
                status.finished_unix_ms = Some(now_unix_ms());
                status.error =
                    Some("Kiln restarted before this OpenEnv run reached a terminal state".into());
                persist_status_to(&status_path, &status)?;
            }
            restored.push(status);
        }
        restored.sort_by_key(|status| std::cmp::Reverse(status.submitted_unix_ms));
        restored.truncate(self.policy.max_tracked_runs);
        let now = now_unix_ms();
        let ttl_ms = self.policy.tracked_run_ttl_secs.saturating_mul(1000);
        let mut runs = self.runs.write().unwrap();
        for status in restored {
            if status
                .finished_unix_ms
                .is_some_and(|finished| now.saturating_sub(finished) > ttl_ms)
            {
                continue;
            }
            runs.insert(
                status.run_id.clone(),
                TrackedOpenEnvRun {
                    status,
                    cancel: Arc::new(AtomicBool::new(false)),
                },
            );
        }
        Ok(())
    }

    fn run_dir(&self, run_id: &str) -> PathBuf {
        self.root.join(run_id)
    }

    fn status_path(&self, run_id: &str) -> PathBuf {
        self.run_dir(run_id).join("run.json")
    }

    fn prune_locked(&self, runs: &mut HashMap<String, TrackedOpenEnvRun>) {
        let now = now_unix_ms();
        let ttl_ms = self.policy.tracked_run_ttl_secs.saturating_mul(1000);
        runs.retain(|_, tracked| {
            !tracked
                .status
                .finished_unix_ms
                .is_some_and(|finished| now.saturating_sub(finished) > ttl_ms)
        });
    }

    fn make_room_locked(&self, runs: &mut HashMap<String, TrackedOpenEnvRun>) {
        while runs.len() >= self.policy.max_tracked_runs {
            let oldest_terminal = runs
                .iter()
                .filter(|(_, tracked)| tracked.status.terminal())
                .min_by_key(|(_, tracked)| tracked.status.submitted_unix_ms)
                .map(|(run_id, _)| run_id.clone());
            let Some(run_id) = oldest_terminal else {
                break;
            };
            runs.remove(&run_id);
        }
    }

    fn insert(&self, request: OpenEnvRunRequest) -> Result<(OpenEnvRunStatus, Arc<AtomicBool>)> {
        anyhow::ensure!(self.policy.enabled, "OpenEnv control plane is disabled");
        let mut runs = self.runs.write().unwrap();
        self.prune_locked(&mut runs);
        self.make_room_locked(&mut runs);
        let active = runs
            .values()
            .filter(|tracked| !tracked.status.terminal())
            .count();
        anyhow::ensure!(
            active < self.policy.max_active_runs,
            "OpenEnv active-run capacity is full ({})",
            self.policy.max_active_runs
        );
        anyhow::ensure!(
            runs.len() < self.policy.max_tracked_runs,
            "OpenEnv tracked-run capacity is full ({})",
            self.policy.max_tracked_runs
        );

        let run_id = uuid::Uuid::new_v4().to_string();
        let run_dir = self.run_dir(&run_id);
        std::fs::create_dir(&run_dir)
            .with_context(|| format!("create OpenEnv run directory {}", run_dir.display()))?;
        let rollouts_total = request
            .groups
            .checked_mul(request.group_size)
            .context("OpenEnv rollout count overflow")?;
        let status = OpenEnvRunStatus {
            schema: OPENENV_RUN_SCHEMA_V2.to_string(),
            run_id: run_id.clone(),
            kind: request.kind,
            state: OpenEnvRunState::Queued,
            submitted_unix_ms: now_unix_ms(),
            finished_unix_ms: None,
            progress: OpenEnvRunProgress {
                groups_completed: 0,
                groups_total: request.groups,
                rollouts_completed: 0,
                rollouts_total,
            },
            request,
            environments: Vec::new(),
            artifacts: Vec::new(),
            training_job_id: None,
            training_submission: None,
            training: None,
            post_evaluations: Vec::new(),
            error: None,
        };
        persist_status_to(&run_dir.join("run.json"), &status)?;
        let cancel = Arc::new(AtomicBool::new(false));
        runs.insert(
            run_id,
            TrackedOpenEnvRun {
                status: status.clone(),
                cancel: cancel.clone(),
            },
        );
        Ok((status, cancel))
    }

    fn get(&self, run_id: &str) -> Option<OpenEnvRunStatus> {
        self.runs
            .read()
            .unwrap()
            .get(run_id)
            .map(|tracked| tracked.status.clone())
    }

    fn list(&self) -> Vec<OpenEnvRunStatus> {
        let mut tracked = self.runs.write().unwrap();
        self.prune_locked(&mut tracked);
        let mut runs = tracked
            .values()
            .map(|tracked| tracked.status.clone())
            .collect::<Vec<_>>();
        runs.sort_by_key(|status| std::cmp::Reverse(status.submitted_unix_ms));
        runs
    }

    pub fn counts(&self) -> (usize, usize) {
        let mut runs = self.runs.write().unwrap();
        self.prune_locked(&mut runs);
        (
            runs.values()
                .filter(|tracked| !tracked.status.terminal())
                .count(),
            runs.len(),
        )
    }

    fn update_progress(&self, run_id: &str, progress: OpenEnvCollectionProgress) {
        if let Err(error) = self.update(run_id, |status| {
            status.state = match progress.stage {
                crate::openenv_cli::OpenEnvCollectionStage::Discovering => {
                    OpenEnvRunState::Discovering
                }
                crate::openenv_cli::OpenEnvCollectionStage::Collecting => {
                    OpenEnvRunState::Collecting
                }
            };
            status.progress.groups_completed = progress.groups_completed;
            status.progress.groups_total = progress.groups_total;
            status.progress.rollouts_completed = progress.rollouts_completed;
        }) {
            tracing::warn!(
                run_id,
                error = %error,
                "failed to persist OpenEnv collection progress"
            );
        }
    }

    fn update_environments(&self, run_id: &str, environments: Vec<OpenEnvInspection>) {
        if let Err(error) = self.update(run_id, |status| {
            status.environments = environments
                .into_iter()
                .map(|inspection| inspection.identity)
                .collect();
        }) {
            tracing::warn!(
                run_id,
                error = %error,
                "failed to persist discovered OpenEnv identities"
            );
        }
    }

    fn update(
        &self,
        run_id: &str,
        update: impl FnOnce(&mut OpenEnvRunStatus),
    ) -> Result<OpenEnvRunStatus> {
        let status = {
            let mut runs = self.runs.write().unwrap();
            let tracked = runs
                .get_mut(run_id)
                .with_context(|| format!("OpenEnv run {run_id} disappeared"))?;
            update(&mut tracked.status);
            tracked.status.clone()
        };
        persist_status_to(&self.status_path(run_id), &status)?;
        Ok(status)
    }

    fn cancel(&self, run_id: &str) -> Result<OpenEnvRunStatus> {
        let tracked = self
            .runs
            .read()
            .unwrap()
            .get(run_id)
            .map(|tracked| (tracked.status.clone(), tracked.cancel.clone()))
            .with_context(|| format!("OpenEnv run {run_id} was not found"))?;
        anyhow::ensure!(
            !tracked.0.terminal(),
            "OpenEnv run {run_id} cannot be cancelled from state {:?}",
            tracked.0.state
        );
        tracked.1.store(true, Ordering::Relaxed);
        self.update(run_id, |status| {
            status.error =
                Some("Cancellation requested; the active protocol boundary will stop".into());
        })
    }

    fn artifact_path(&self, run_id: &str, kind: &str) -> Option<(PathBuf, &'static str)> {
        let filename = match kind {
            "dataset" => ("rollouts.jsonl", "application/x-ndjson"),
            "replay" => ("replay.json", "application/json"),
            "summary" => ("summary.json", "application/json"),
            _ => return None,
        };
        self.get(run_id)?;
        Some((self.run_dir(run_id).join(filename.0), filename.1))
    }
}

fn persist_status_to(path: &Path, status: &OpenEnvRunStatus) -> Result<()> {
    let parent = path.parent().context("OpenEnv status path has no parent")?;
    let bytes = serde_json::to_vec_pretty(status).context("serialize OpenEnv run status")?;
    anyhow::ensure!(
        bytes.len() <= MAX_PERSISTED_STATUS_BYTES as usize,
        "OpenEnv run status exceeds persistence limit"
    );
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("stage OpenEnv status beside {}", path.display()))?;
    staged
        .write_all(&bytes)
        .with_context(|| format!("write staged OpenEnv status {}", path.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv status {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish OpenEnv status {}", path.display()))?;
    Ok(())
}

fn openenv_error(
    status: StatusCode,
    code: &'static str,
    message: impl std::fmt::Display,
    hint: &'static str,
) -> ApiError {
    ApiError {
        status,
        code,
        message: message.to_string(),
        hint,
        retry_after_seconds: (status == StatusCode::SERVICE_UNAVAILABLE).then_some(5),
    }
}

fn validate_environment_urls(urls: &[String], allow_remote: bool) -> Result<(), ApiError> {
    if urls.is_empty() || urls.len() > MAX_ENVIRONMENTS {
        return Err(openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_request",
            format!("environment_urls must contain 1..={MAX_ENVIRONMENTS} URLs"),
            "Provide one or more HTTP(S) OpenEnv base URLs.",
        ));
    }
    for raw in urls {
        let url = reqwest::Url::parse(raw).map_err(|error| {
            openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_url",
                format!("invalid OpenEnv URL {raw:?}: {error}"),
                "Use an absolute http:// or https:// OpenEnv base URL without credentials, query, or fragment.",
            )
        })?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_url",
                format!("OpenEnv URL {raw:?} has an unsupported scheme or URL component"),
                "Use an absolute http:// or https:// OpenEnv base URL without credentials, query, or fragment.",
            ));
        }
        if !allow_remote {
            let loopback = url.host_str().is_some_and(|host| {
                host.eq_ignore_ascii_case("localhost")
                    || host
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .parse::<IpAddr>()
                        .is_ok_and(|address| address.is_loopback())
            });
            if !loopback {
                return Err(openenv_error(
                    StatusCode::FORBIDDEN,
                    "openenv_remote_environment_forbidden",
                    format!("remote OpenEnv origin is disabled: {raw}"),
                    "Use a loopback OpenEnv server or set [openenv] allow_remote_environments=true at trusted startup.",
                ));
            }
        }
    }
    Ok(())
}

fn validate_run_request(
    request: &OpenEnvRunRequest,
    policy: &OpenEnvConfig,
) -> Result<(), ApiError> {
    validate_environment_urls(&request.environment_urls, policy.allow_remote_environments)?;
    if !matches!(
        request.adapter.trim().to_ascii_lowercase().as_str(),
        "base" | "none" | "null"
    ) {
        crate::api::adapters::validate_adapter_name(request.adapter.trim())?;
    }
    if !request.reset_options.is_object() {
        return Err(openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_request",
            "reset_options must be a JSON object",
            "Send reset_options as an object; Kiln adds the deterministic seed.",
        ));
    }
    match request.kind {
        OpenEnvRunKind::Rollout if request.output_adapter.is_some() => {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_request",
                "output_adapter is valid only for kind=train",
                "Remove output_adapter or set kind to train.",
            ));
        }
        OpenEnvRunKind::Train => {
            let output = request.output_adapter.as_deref().unwrap_or_default();
            crate::api::adapters::validate_adapter_name(output)?;
        }
        OpenEnvRunKind::Rollout => {}
    }
    validate_options(&rollout_options_for(request, Path::new("."))).map_err(|error| {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_request",
            error,
            "Correct the OpenEnv collection bounds and retry.",
        )
    })?;
    Ok(())
}

fn rollout_options_for(request: &OpenEnvRunRequest, run_dir: &Path) -> OpenEnvRolloutOptions {
    OpenEnvRolloutOptions {
        kiln_url: "in-process".to_string(),
        environment_urls: request.environment_urls.clone(),
        adapter: request.adapter.clone(),
        groups: request.groups,
        group_size: request.group_size,
        seed_start: request.seed_start,
        reset_options: None,
        reset_options_value: Some(request.reset_options.clone()),
        max_steps: request.max_steps,
        concurrency: request.concurrency,
        max_action_tokens: request.max_action_tokens,
        temperature: request.temperature,
        thinking: request.thinking,
        protocol_error_reward: request.protocol_error_reward,
        max_recoverable_errors: request.max_recoverable_errors,
        capacity_wait_seconds: request.capacity_wait_seconds,
        output: run_dir.join("rollouts.jsonl"),
        replay_output: run_dir.join("replay.json"),
        summary_output: run_dir.join("summary.json"),
    }
}

async fn inspect(
    State(state): State<AppState>,
    Json(request): Json<OpenEnvInspectRequest>,
) -> Result<Json<OpenEnvInspectResponse>, ApiError> {
    if !state.openenv_runs.policy().enabled {
        return Err(openenv_error(
            StatusCode::NOT_FOUND,
            "openenv_disabled",
            "OpenEnv control plane is disabled",
            "Set [openenv] enabled=true and restart Kiln, or use the kiln openenv CLI.",
        ));
    }
    validate_environment_urls(
        &request.environment_urls,
        state.openenv_runs.policy().allow_remote_environments,
    )?;
    let environments = stream::iter(request.environment_urls)
        .map(|url| async move {
            let client = OpenEnvClient::new(&url)?;
            client
                .inspect()
                .await
                .with_context(|| format!("inspect OpenEnv server {}", client.base_url()))
        })
        .buffered(MAX_ENVIRONMENTS)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .map_err(|error| {
            openenv_error(
                StatusCode::BAD_GATEWAY,
                "openenv_inspection_failed",
                error,
                "Check the OpenEnv server health, protocol endpoints, and network reachability.",
            )
        })?;
    Ok(Json(OpenEnvInspectResponse {
        schema: OPENENV_INSPECTION_SCHEMA_V1,
        environments,
    }))
}

async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<OpenEnvRunRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if !state.openenv_runs.policy().enabled {
        return Err(openenv_error(
            StatusCode::NOT_FOUND,
            "openenv_disabled",
            "OpenEnv control plane is disabled",
            "Set [openenv] enabled=true and restart Kiln, or use the kiln openenv CLI.",
        ));
    }
    validate_run_request(&request, state.openenv_runs.policy())?;
    let (status, cancel) = state.openenv_runs.insert(request).map_err(|error| {
        let capacity = error.to_string().contains("capacity is full");
        openenv_error(
            if capacity {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            },
            if capacity {
                "openenv_run_capacity_full"
            } else {
                "openenv_run_create_failed"
            },
            error,
            if capacity {
                "Wait for an active run to finish or for terminal history to expire, then retry."
            } else {
                "Check Kiln's adapter directory permissions and server logs."
            },
        )
    })?;
    state
        .metrics
        .openenv_runs_started
        .fetch_add(1, Ordering::Relaxed);
    let run_id = status.run_id.clone();
    tokio::spawn(async move {
        execute_run(state, run_id, cancel).await;
    });
    Ok((StatusCode::ACCEPTED, Json(status)))
}

async fn execute_run(state: AppState, run_id: String, cancel: Arc<AtomicBool>) {
    if let Err(error) = execute_run_inner(&state, &run_id, cancel.clone()).await {
        let cancelled = cancel.load(Ordering::Relaxed);
        let error_message = format!("{error:#}");
        tracing::warn!(
            run_id = %run_id,
            cancelled,
            error = %error_message,
            "OpenEnv run terminated"
        );
        if let Err(persist_error) = state.openenv_runs.update(&run_id, |status| {
            status.state = if cancelled {
                OpenEnvRunState::Cancelled
            } else {
                OpenEnvRunState::Failed
            };
            status.finished_unix_ms = Some(now_unix_ms());
            status.error = Some(error_message);
        }) {
            tracing::error!(
                run_id = %run_id,
                error = %persist_error,
                "failed to persist terminal OpenEnv run state"
            );
        }
        if cancelled {
            state
                .metrics
                .openenv_runs_cancelled
                .fetch_add(1, Ordering::Relaxed);
        } else {
            state
                .metrics
                .openenv_runs_failed
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn execute_run_inner(state: &AppState, run_id: &str, cancel: Arc<AtomicBool>) -> Result<()> {
    let request = state
        .openenv_runs
        .get(run_id)
        .context("OpenEnv run disappeared before execution")?
        .request;
    let run_dir = state.openenv_runs.run_dir(run_id);
    let options = rollout_options_for(&request, &run_dir);
    let summary_output = options.summary_output.clone();
    let registry = state.openenv_runs.clone();
    let progress_run_id = run_id.to_string();
    let progress = Arc::new(move |progress| {
        registry.update_progress(&progress_run_id, progress);
    });
    let registry = state.openenv_runs.clone();
    let discovery_run_id = run_id.to_string();
    let discovered = Arc::new(move |environments| {
        registry.update_environments(&discovery_run_id, environments);
    });
    let control = OpenEnvCollectionControl::new(cancel.clone(), Some(progress), Some(discovered));
    let policy = OpenEnvPolicyTransport::InProcess(state.clone());
    let mut collection = collect_openenv_rollouts_with_policy(&options, &policy, &control).await?;
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "OpenEnv run cancelled");
    write_openenv_outputs(
        &options,
        &collection.groups,
        &collection.replay,
        &collection.summary,
    )?;
    state.metrics.openenv_episodes_collected.fetch_add(
        u64::try_from(collection.summary.rollout_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );

    let artifacts = artifacts_for(run_id, &collection.summary);
    if request.kind == OpenEnvRunKind::Rollout {
        state.openenv_runs.update(run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.finished_unix_ms = Some(now_unix_ms());
            status.progress.groups_completed = request.groups;
            status.progress.rollouts_completed = request.groups.saturating_mul(request.group_size);
            status.environments = collection
                .summary
                .environments
                .iter()
                .map(|inspection| inspection.identity.clone())
                .collect();
            status.artifacts = artifacts;
            status.error = None;
        })?;
        state
            .metrics
            .openenv_rollouts_ready
            .fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            run_id,
            groups = request.groups,
            rollouts = request.groups.saturating_mul(request.group_size),
            "OpenEnv rollout run completed"
        );
        return Ok(());
    }

    state.openenv_runs.update(run_id, |status| {
        status.state = OpenEnvRunState::Submitting;
        status.finished_unix_ms = None;
        status.progress.groups_completed = request.groups;
        status.progress.rollouts_completed = request.groups.saturating_mul(request.group_size);
        status.environments = collection
            .summary
            .environments
            .iter()
            .map(|inspection| inspection.identity.clone())
            .collect();
        status.artifacts = artifacts;
        status.error = None;
    })?;
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "OpenEnv run cancelled");
    let output_adapter = request
        .output_adapter
        .as_ref()
        .context("OpenEnv train run has no output adapter")?;
    let mut config = request.training_config.clone().unwrap_or_default();
    config.output_name = Some(output_adapter.clone());
    config.auto_load = request.auto_load;
    config.behavior_policy = BehaviorPolicy::NoImportanceCorrection;
    config.base_adapter = if matches!(
        request.adapter.trim().to_ascii_lowercase().as_str(),
        "base" | "none" | "null"
    ) {
        None
    } else {
        Some(request.adapter.clone())
    };
    let submission = super::training::submit_grpo_request(
        state,
        GrpoRequest {
            groups: collection.groups,
            dataset_path: None,
            dataset: None,
            dataset_split: None,
            config,
            post_eval: request.post_eval.clone(),
        },
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "OpenEnv GRPO admission failed with {} {}: {}",
            error.status,
            error.code,
            error.message
        )
    })?;
    collection.summary.training_submission =
        Some(serde_json::to_value(&submission).context("serialize training submission")?);
    write_summary_atomic(&summary_output, &collection.summary)?;
    let artifacts = artifacts_for(run_id, &collection.summary);
    let training = training_status_for(state, &submission.job_id)
        .context("admitted OpenEnv training job disappeared")?;
    state.openenv_runs.update(run_id, |status| {
        status.state = OpenEnvRunState::TrainingQueued;
        status.training_job_id = Some(submission.job_id.clone());
        status.training_submission = Some(submission.clone());
        status.training = Some(training);
        status.artifacts = artifacts;
        status.error = None;
    })?;
    state
        .metrics
        .openenv_training_queued
        .fetch_add(1, Ordering::Relaxed);
    follow_openenv_training(state, run_id, &request, &submission.job_id, cancel).await?;
    state
        .metrics
        .openenv_training_completed
        .fetch_add(1, Ordering::Relaxed);
    tracing::info!(
        run_id,
        training_job_id = %submission.job_id,
        groups = request.groups,
        rollouts = request.groups.saturating_mul(request.group_size),
        "OpenEnv training run completed"
    );
    Ok(())
}

fn training_status_for(state: &AppState, job_id: &str) -> Option<OpenEnvTrainingStatus> {
    state
        .training_jobs
        .read()
        .unwrap()
        .get(job_id)
        .map(|job| OpenEnvTrainingStatus {
            job_id: job.job_id.clone(),
            state: job.state,
            progress: job.progress,
            current_loss: job.loss,
            epoch: job.epoch,
            adapter_path: job.adapter_path.clone(),
            linked_eval_job_ids: job.linked_eval_job_ids.clone(),
            post_eval_verdict: job.post_eval_verdict.clone(),
            gate_outcome: job.gate_outcome.clone(),
            error: job.error.clone(),
        })
}

fn post_eval_statuses_for(
    state: &AppState,
    job_ids: &[String],
) -> Result<Vec<OpenEnvPostEvalStatus>> {
    let jobs = state.eval_jobs.read().unwrap();
    job_ids
        .iter()
        .map(|job_id| {
            let job = jobs
                .get(job_id)
                .with_context(|| format!("linked post-eval job {job_id} disappeared"))?;
            Ok(OpenEnvPostEvalStatus {
                job_id: job.job_id.clone(),
                suite_name: job.suite_name.clone(),
                state: job.state,
                examples_completed: job.progress.examples_completed,
                examples_total: job.progress.examples_total,
                headline_accuracy: job.headline_accuracy,
                error: job.error.clone(),
            })
        })
        .collect()
}

fn publish_training_phase(
    state: &AppState,
    run_id: &str,
    run_state: OpenEnvRunState,
    training: OpenEnvTrainingStatus,
    post_evaluations: Vec<OpenEnvPostEvalStatus>,
) -> Result<()> {
    let current = state
        .openenv_runs
        .get(run_id)
        .context("OpenEnv run disappeared while following training")?;
    if current.state == run_state
        && current.training.as_ref() == Some(&training)
        && current.post_evaluations == post_evaluations
    {
        return Ok(());
    }
    state.openenv_runs.update(run_id, |status| {
        status.state = run_state;
        status.training = Some(training);
        status.post_evaluations = post_evaluations;
    })?;
    Ok(())
}

fn request_linked_cancellation(
    state: &AppState,
    training: &OpenEnvTrainingStatus,
    evals: &[OpenEnvPostEvalStatus],
) {
    if matches!(
        training.state,
        TrainingState::Queued | TrainingState::Running
    ) {
        match crate::job_cancellation::request_training_job_cancellation(state, &training.job_id) {
            Ok(_) => {}
            Err(error) => tracing::warn!(
                training_job_id = %training.job_id,
                error = %error.message,
                "OpenEnv cancellation could not be forwarded to training"
            ),
        }
    }
    for eval in evals {
        if matches!(eval.state, EvalJobState::Queued | EvalJobState::Running) {
            match crate::job_cancellation::request_eval_job_cancellation(state, &eval.job_id) {
                Ok(_) => {}
                Err(error) => tracing::warn!(
                    eval_job_id = %eval.job_id,
                    error = %error.message,
                    "OpenEnv cancellation could not be forwarded to post-evaluation"
                ),
            }
        }
    }
}

async fn follow_openenv_training(
    state: &AppState,
    run_id: &str,
    request: &OpenEnvRunRequest,
    training_job_id: &str,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let mut training_completed_at = None;
    let mut gate_wait_started = None;
    loop {
        let training = training_status_for(state, training_job_id)
            .with_context(|| format!("OpenEnv training job {training_job_id} disappeared"))?;
        let evals = post_eval_statuses_for(state, &training.linked_eval_job_ids)?;
        if cancel.load(Ordering::Relaxed) {
            request_linked_cancellation(state, &training, &evals);
        }

        match training.state {
            TrainingState::Queued => {
                publish_training_phase(
                    state,
                    run_id,
                    OpenEnvRunState::TrainingQueued,
                    training,
                    evals,
                )?;
            }
            TrainingState::Running => {
                publish_training_phase(
                    state,
                    run_id,
                    OpenEnvRunState::TrainingRunning,
                    training,
                    evals,
                )?;
            }
            TrainingState::Failed => {
                anyhow::bail!(
                    "OpenEnv training job {training_job_id} failed: {}",
                    training
                        .error
                        .as_deref()
                        .unwrap_or("trainer reported no failure detail")
                );
            }
            TrainingState::Completed => {
                if request.post_eval.is_none() {
                    state.openenv_runs.update(run_id, |status| {
                        status.state = OpenEnvRunState::Completed;
                        status.finished_unix_ms = Some(now_unix_ms());
                        status.training = Some(training);
                        status.post_evaluations = evals;
                        status.error = None;
                    })?;
                    return Ok(());
                }

                let completed_at = training_completed_at.get_or_insert_with(Instant::now);
                let expected_evals = 1 + usize::from(
                    request
                        .post_eval
                        .as_ref()
                        .is_some_and(|cfg| cfg.include_baseline),
                );
                publish_training_phase(
                    state,
                    run_id,
                    OpenEnvRunState::PostEvaluating,
                    training.clone(),
                    evals.clone(),
                )?;
                if training.linked_eval_job_ids.len() < expected_evals {
                    anyhow::ensure!(
                        completed_at.elapsed() < POST_EVAL_PUBLICATION_GRACE,
                        "OpenEnv requested {expected_evals} post-training evaluation job(s), but only {} were published",
                        training.linked_eval_job_ids.len()
                    );
                } else {
                    if let Some(failed) =
                        evals.iter().find(|eval| eval.state == EvalJobState::Failed)
                    {
                        anyhow::bail!(
                            "OpenEnv post-training evaluation {} failed: {}",
                            failed.job_id,
                            failed
                                .error
                                .as_deref()
                                .unwrap_or("evaluator reported no failure detail")
                        );
                    }
                    if let Some(cancelled) = evals
                        .iter()
                        .find(|eval| eval.state == EvalJobState::Cancelled)
                    {
                        anyhow::bail!(
                            "OpenEnv post-training evaluation {} was cancelled",
                            cancelled.job_id
                        );
                    }
                    let evaluations_done = evals
                        .iter()
                        .all(|eval| eval.state == EvalJobState::Completed);
                    let gate_done = request
                        .post_eval
                        .as_ref()
                        .is_none_or(|cfg| cfg.min_accuracy.is_none())
                        || training.gate_outcome.is_some();
                    if evaluations_done && !gate_done {
                        let started = gate_wait_started.get_or_insert_with(Instant::now);
                        anyhow::ensure!(
                            started.elapsed() < POST_EVAL_GATE_TIMEOUT,
                            "OpenEnv post-training promotion gate did not publish an outcome within {} seconds",
                            POST_EVAL_GATE_TIMEOUT.as_secs()
                        );
                    }
                    if evaluations_done && gate_done {
                        let final_training = training_status_for(state, training_job_id)
                            .context("OpenEnv training job disappeared at completion")?;
                        state.openenv_runs.update(run_id, |status| {
                            status.state = OpenEnvRunState::Completed;
                            status.finished_unix_ms = Some(now_unix_ms());
                            status.training = Some(final_training);
                            status.post_evaluations = evals;
                            status.error = None;
                        })?;
                        return Ok(());
                    }
                }
            }
        }
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL).await;
    }
}

fn artifacts_for(run_id: &str, summary: &OpenEnvRolloutSummary) -> Vec<OpenEnvArtifact> {
    let prefix = format!("/v1/openenv/runs/{run_id}/artifacts");
    let summary_bytes = std::fs::read(&summary.summary_output_path).unwrap_or_default();
    vec![
        OpenEnvArtifact {
            kind: "dataset".into(),
            url: format!("{prefix}/dataset"),
            sha256: summary.dataset_sha256.clone(),
            bytes: summary.dataset_bytes,
        },
        OpenEnvArtifact {
            kind: "replay".into(),
            url: format!("{prefix}/replay"),
            sha256: summary.replay_sha256.clone(),
            bytes: summary.replay_bytes,
        },
        OpenEnvArtifact {
            kind: "summary".into(),
            url: format!("{prefix}/summary"),
            sha256: crate::openenv_replay::sha256_bytes(&summary_bytes),
            bytes: summary_bytes.len(),
        },
    ]
}

async fn list_runs(State(state): State<AppState>) -> Json<OpenEnvRunList> {
    Json(OpenEnvRunList {
        schema: OPENENV_RUN_LIST_SCHEMA_V2,
        runs: state.openenv_runs.list(),
    })
}

async fn get_run(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<OpenEnvRunStatus>, ApiError> {
    state.openenv_runs.get(&run_id).map(Json).ok_or_else(|| {
        openenv_error(
            StatusCode::NOT_FOUND,
            "openenv_run_not_found",
            format!("OpenEnv run {run_id} was not found"),
            "List retained runs with GET /v1/openenv/runs.",
        )
    })
}

async fn cancel_run(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<OpenEnvRunStatus>, ApiError> {
    state
        .openenv_runs
        .cancel(&run_id)
        .map(Json)
        .map_err(|error| {
            let missing = state.openenv_runs.get(&run_id).is_none();
            openenv_error(
                if missing {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::CONFLICT
                },
                if missing {
                    "openenv_run_not_found"
                } else {
                    "openenv_run_not_cancellable"
                },
                error,
                if missing {
                    "List retained runs with GET /v1/openenv/runs."
                } else {
                    "Only non-terminal OpenEnv runs can be cancelled."
                },
            )
        })
}

async fn download_artifact(
    State(state): State<AppState>,
    AxumPath((run_id, kind)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    let (path, content_type) = state
        .openenv_runs
        .artifact_path(&run_id, &kind)
        .ok_or_else(|| {
            openenv_error(
                StatusCode::NOT_FOUND,
                "openenv_artifact_not_found",
                format!("OpenEnv artifact {kind:?} for run {run_id} was not found"),
                "Use dataset, replay, or summary from the run's artifacts array.",
            )
        })?;
    let file = tokio::fs::File::open(&path).await.map_err(|error| {
        openenv_error(
            StatusCode::NOT_FOUND,
            "openenv_artifact_not_ready",
            format!(
                "OpenEnv artifact {} is unavailable: {error}",
                path.display()
            ),
            "Wait for the run to reach rollout_ready or training_queued.",
        )
    })?;
    let (tx, rx) = mpsc::channel::<std::io::Result<Vec<u8>>>(8);
    tokio::spawn(async move {
        let mut file = file;
        loop {
            let mut chunk = vec![0u8; ARTIFACT_CHUNK_BYTES];
            match file.read(&mut chunk).await {
                Ok(0) => break,
                Ok(read) => {
                    chunk.truncate(read);
                    if tx.send(Ok(chunk)).await.is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    break;
                }
            }
        }
    });
    let disposition = HeaderValue::from_str(&format!(
        "attachment; filename=\"openenv-{run_id}-{kind}.{}\"",
        if kind == "dataset" { "jsonl" } else { "json" }
    ))
    .map_err(ApiError::internal)?;
    Ok((
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_str(content_type).map_err(ApiError::internal)?,
            ),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response())
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/openenv/inspect",
            post(inspect).layer(DefaultBodyLimit::max(OPENENV_API_BODY_LIMIT)),
        )
        .route(
            "/v1/openenv/runs",
            get(list_runs)
                .post(create_run)
                .layer(DefaultBodyLimit::max(OPENENV_API_BODY_LIMIT)),
        )
        .route("/v1/openenv/runs/{run_id}", get(get_run).delete(cancel_run))
        .route(
            "/v1/openenv/runs/{run_id}/artifacts/{kind}",
            get(download_artifact),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::queue::{EvalJobInfo, EvalSubmissionKind};
    use crate::state::{TrainingJobInfo, TrainingJobType};
    use axum::body::Body;
    use axum::http::Request;
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use tower::ServiceExt;

    fn test_state(temp: &tempfile::TempDir, policy: OpenEnvConfig) -> AppState {
        let model_config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(SchedulerConfig::default(), 256);
        let mut state = AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(MockEngine::new(model_config)),
            crate::api::test_tokenizer(),
            300,
            "Qwen3.5-4B".to_string(),
        );
        state.openenv_runs =
            Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap());
        state
    }

    fn request(kind: OpenEnvRunKind) -> OpenEnvRunRequest {
        OpenEnvRunRequest {
            kind,
            environment_urls: vec!["http://127.0.0.1:8000".into()],
            adapter: "base".into(),
            groups: 2,
            group_size: 3,
            seed_start: 0,
            reset_options: default_reset_options(),
            max_steps: 8,
            concurrency: 2,
            max_action_tokens: 128,
            temperature: 1.0,
            thinking: false,
            protocol_error_reward: -1.0,
            max_recoverable_errors: 3,
            capacity_wait_seconds: 30,
            output_adapter: (kind == OpenEnvRunKind::Train).then(|| "agent".into()),
            training_config: None,
            auto_load: true,
            post_eval: None,
        }
    }

    fn training_job(job_id: &str, state: TrainingState) -> TrainingJobInfo {
        TrainingJobInfo {
            job_id: job_id.into(),
            adapter_name: "agent".into(),
            job_type: TrainingJobType::Grpo,
            effective_seed: Some(17),
            state,
            progress: if state == TrainingState::Completed {
                1.0
            } else {
                0.0
            },
            loss: None,
            epoch: None,
            adapter_path: None,
            submitted_at: Instant::now(),
            submitted_unix_ms: now_unix_ms(),
            auto_load: false,
            consumed_correction_ids: Vec::new(),
            training_data: None,
            finished_at: None,
            finished_unix_ms: None,
            error: None,
            linked_eval_job_ids: Vec::new(),
            post_eval_verdict: None,
            gate_outcome: None,
            post_eval_gate_evidence: Vec::new(),
            cancel_requested: Default::default(),
            loss_history: Vec::new(),
        }
    }

    #[test]
    fn run_registry_is_bounded_persisted_and_restored() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            max_tracked_runs: 2,
            ..Default::default()
        };
        let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
        let (status, _) = registry.insert(request(OpenEnvRunKind::Rollout)).unwrap();
        assert!(registry.insert(request(OpenEnvRunKind::Rollout)).is_err());
        registry
            .update(&status.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        assert_eq!(
            restored.get(&status.run_id).unwrap().state,
            OpenEnvRunState::RolloutReady
        );
        let (second, _) = restored.insert(request(OpenEnvRunKind::Rollout)).unwrap();
        let first_submitted = status.submitted_unix_ms;
        restored
            .update(&second.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.submitted_unix_ms = first_submitted.saturating_add(1);
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        restored
            .insert(request(OpenEnvRunKind::Rollout))
            .expect("the oldest terminal status should be evicted to admit new work");
        assert!(restored.get(&status.run_id).is_none());
    }

    #[test]
    fn remote_environment_policy_is_fail_closed() {
        assert!(validate_environment_urls(&["http://127.0.0.1:1".into()], false).is_ok());
        assert!(validate_environment_urls(&["http://[::1]:1".into()], false).is_ok());
        assert!(validate_environment_urls(&["https://example.com".into()], false).is_err());
        assert!(validate_environment_urls(&["https://example.com".into()], true).is_ok());
        assert!(
            validate_environment_urls(&["http://user:secret@127.0.0.1:1".into()], true).is_err()
        );
    }

    #[test]
    fn train_requires_a_valid_output_adapter() {
        let policy = OpenEnvConfig::default();
        let mut train = request(OpenEnvRunKind::Train);
        train.output_adapter = None;
        assert!(validate_run_request(&train, &policy).is_err());
        train.output_adapter = Some("../escape".into());
        assert!(validate_run_request(&train, &policy).is_err());
    }

    #[test]
    fn collection_bounds_are_rejected_before_run_admission() {
        let policy = OpenEnvConfig::default();
        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.groups = 0;
        assert!(validate_run_request(&rollout, &policy).is_err());
        rollout.groups = 2;
        rollout.temperature = f32::NAN;
        assert!(validate_run_request(&rollout, &policy).is_err());
        rollout.temperature = 1.0;
        rollout.adapter = "../escape".into();
        assert!(validate_run_request(&rollout, &policy).is_err());
    }

    #[test]
    fn cancellation_remains_available_after_training_handoff() {
        let temp = tempfile::tempdir().unwrap();
        let registry =
            OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
        let (status, _) = registry.insert(request(OpenEnvRunKind::Train)).unwrap();
        registry
            .update(&status.run_id, |status| {
                status.state = OpenEnvRunState::Submitting;
            })
            .unwrap();
        assert!(registry.cancel(&status.run_id).is_ok());
        registry
            .update(&status.run_id, |status| {
                status.state = OpenEnvRunState::Completed;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        assert!(registry.cancel(&status.run_id).is_err());
    }

    #[test]
    fn v1_training_handoffs_remain_terminal_after_upgrade() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            ..Default::default()
        };
        let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
        let (legacy, _) = registry.insert(request(OpenEnvRunKind::Train)).unwrap();
        registry
            .update(&legacy.run_id, |status| {
                status.schema = OPENENV_RUN_SCHEMA_V1.into();
                status.state = OpenEnvRunState::TrainingQueued;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        assert_eq!(registry.counts().0, 0);
        assert!(
            registry.insert(request(OpenEnvRunKind::Train)).is_ok(),
            "a historical v1 handoff must not consume v2 active-run capacity"
        );
        let restored =
            OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).expect("restore registry");
        assert_eq!(
            restored.get(&legacy.run_id).unwrap().state,
            OpenEnvRunState::TrainingQueued
        );
    }

    #[tokio::test]
    async fn openenv_run_follows_trainer_to_actual_completion() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let (run, cancel) = state
            .openenv_runs
            .insert(request(OpenEnvRunKind::Train))
            .unwrap();
        state.training_jobs.write().unwrap().insert(
            "train-1".into(),
            training_job("train-1", TrainingState::Queued),
        );

        let followed_state = state.clone();
        let followed_run_id = run.run_id.clone();
        let follow = tokio::spawn(async move {
            follow_openenv_training(
                &followed_state,
                &followed_run_id,
                &request(OpenEnvRunKind::Train),
                "train-1",
                cancel,
            )
            .await
        });
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
        assert_eq!(
            state.openenv_runs.get(&run.run_id).unwrap().state,
            OpenEnvRunState::TrainingQueued
        );
        {
            let mut jobs = state.training_jobs.write().unwrap();
            let job = jobs.get_mut("train-1").unwrap();
            job.state = TrainingState::Running;
            job.progress = 0.5;
            job.loss = Some(0.25);
            job.epoch = Some(1);
        }
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
        let running = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(running.state, OpenEnvRunState::TrainingRunning);
        assert_eq!(running.training.unwrap().current_loss, Some(0.25));
        {
            let mut jobs = state.training_jobs.write().unwrap();
            let job = jobs.get_mut("train-1").unwrap();
            job.state = TrainingState::Completed;
            job.progress = 1.0;
            job.adapter_path = Some("/adapters/agent".into());
        }
        follow.await.unwrap().unwrap();
        let completed = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(completed.state, OpenEnvRunState::Completed);
        assert!(completed.finished_unix_ms.is_some());
        assert!(completed.terminal());
    }

    #[tokio::test]
    async fn openenv_run_waits_for_requested_post_evaluation() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let mut run_request = request(OpenEnvRunKind::Train);
        run_request.post_eval = Some(kiln_eval::PostEvalConfig {
            suite: "held-out".into(),
            data_scope: Default::default(),
            generation: None,
            min_accuracy: None,
            include_baseline: false,
        });
        let (run, cancel) = state.openenv_runs.insert(run_request.clone()).unwrap();
        let mut training = training_job("train-eval", TrainingState::Completed);
        training.linked_eval_job_ids.push("eval-1".into());
        state
            .training_jobs
            .write()
            .unwrap()
            .insert(training.job_id.clone(), training);
        state.eval_jobs.write().unwrap().insert(
            "eval-1".into(),
            EvalJobInfo::queued(
                "eval-1".into(),
                "held-out".into(),
                vec![Some("agent".into())],
                EvalSubmissionKind::PostTraining,
                Some("train-eval".into()),
                19,
            ),
        );

        let followed_state = state.clone();
        let followed_run_id = run.run_id.clone();
        let follow = tokio::spawn(async move {
            follow_openenv_training(
                &followed_state,
                &followed_run_id,
                &run_request,
                "train-eval",
                cancel,
            )
            .await
        });
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
        assert_eq!(
            state.openenv_runs.get(&run.run_id).unwrap().state,
            OpenEnvRunState::PostEvaluating
        );
        {
            let mut evals = state.eval_jobs.write().unwrap();
            let eval = evals.get_mut("eval-1").unwrap();
            eval.state = EvalJobState::Completed;
            eval.progress.examples_completed = 20;
            eval.progress.examples_total = 20;
            eval.headline_accuracy = Some(0.9);
        }
        follow.await.unwrap().unwrap();
        let completed = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(completed.state, OpenEnvRunState::Completed);
        assert_eq!(completed.post_evaluations[0].headline_accuracy, Some(0.9));
    }

    #[tokio::test]
    async fn http_surface_lists_runs_and_rejects_invalid_work_before_spawning() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let app = routes().with_state(state);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/openenv/runs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["schema"],
            OPENENV_RUN_LIST_SCHEMA_V2
        );

        let mut invalid = request(OpenEnvRunKind::Rollout);
        invalid.groups = 0;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&invalid).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(
            temp.path()
                .join(".openenv")
                .join("runs")
                .read_dir()
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[tokio::test]
    async fn disabled_http_surface_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(
            &temp,
            OpenEnvConfig {
                enabled: false,
                ..Default::default()
            },
        );
        let response = routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&request(OpenEnvRunKind::Rollout)).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

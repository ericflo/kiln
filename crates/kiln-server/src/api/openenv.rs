//! Native OpenEnv discovery, rollout, and training control plane.
//!
//! The HTTP/dashboard lifecycle is deliberately a thin orchestration layer
//! around the same protocol client, collector, chat handler, artifact writer,
//! and GRPO queue admission used by Kiln's CLI and training APIs.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, RwLock,
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
use kiln_openenv::{OpenEnvClientError, OpenEnvIdentity, OpenEnvInspection, OpenEnvTaskCatalog};
use kiln_train::{
    BehaviorPolicy, GrpoConfig, GrpoRequest, TrainingDataProvenance, TrainingResponse,
    TrainingState,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::AsyncReadExt;
use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::config::OpenEnvConfig;
use crate::error::ApiError;
use crate::openenv_cli::{
    MAX_OPENENV_TASK_PAGE_SIZE, OPENENV_TRAINING_CONTRACT_SCHEMA_V1,
    OPENENV_TRAINING_PREFLIGHT_SCHEMA_V1, OpenEnvCollectionControl, OpenEnvCollectionProgress,
    OpenEnvPolicyTransport, OpenEnvRolloutOptions, OpenEnvRolloutStats, OpenEnvRolloutSummary,
    OpenEnvTrainingCapacitySnapshot, OpenEnvTrainingContract, OpenEnvTrainingPreflightReceipt,
    OpenEnvTrainingPreflightRequest, collect_openenv_rollouts_with_policy, validate_options,
    write_openenv_outputs, write_summary_atomic,
};
use crate::openenv_evaluation::{
    OpenEnvEnvironmentEvalConfig, OpenEnvEnvironmentEvalDecision, OpenEnvEnvironmentEvalOutcome,
    OpenEnvEnvironmentEvalProgress, OpenEnvEnvironmentEvalReceipt, OpenEnvEnvironmentEvalState,
    OpenEnvEnvironmentEvalStatus, OpenEnvPolicyIdentity, collect_environment_evaluation,
    evaluation_paths, normalized_adapter, policy_identity, summary_sha256,
    write_environment_evaluation_receipt,
};
use crate::recent_requests::now_unix_ms;
use crate::state::{AppState, TrainingWorkload};

const OPENENV_RUN_SCHEMA_V1: &str = "kiln.openenv-run.v1";
const OPENENV_RUN_SCHEMA_V4: &str = "kiln.openenv-run.v4";
const OPENENV_RUN_SCHEMA_V5: &str = "kiln.openenv-run.v5";
const OPENENV_RUN_LIST_SCHEMA_V5: &str = "kiln.openenv-run-list.v5";
const OPENENV_INSPECTION_SCHEMA_V1: &str = "kiln.openenv-inspection.v1";
const OPENENV_TASK_CATALOG_SCHEMA_V1: &str = "kiln.openenv-task-catalog.v1";
const OPENENV_API_BODY_LIMIT: usize = 1024 * 1024;
const MAX_ENVIRONMENTS: usize = 64;
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 128;
const MAX_PERSISTED_STATUS_BYTES: u64 = 2 * 1024 * 1024;
const ARTIFACT_CHUNK_BYTES: usize = 64 * 1024;
const LIFECYCLE_POLL_INTERVAL: Duration = Duration::from_millis(500);
const POST_EVAL_PUBLICATION_GRACE: Duration = Duration::from_secs(5);
const POST_EVAL_GATE_TIMEOUT: Duration = Duration::from_secs(300);

mod failure;
#[cfg(test)]
mod rollout_stats_tests;
mod training_evidence;

pub use failure::{
    OPENENV_RUN_FAILURE_SCHEMA_V1, OpenEnvRunFailure, OpenEnvRunFailureCode, OpenEnvRunFailureStage,
};
use training_evidence::ensure_openenv_training_evidence;
#[cfg(test)]
use training_evidence::publish_openenv_training_evidence;

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

fn default_task_page_limit() -> usize {
    50
}

fn default_reset_options() -> Value {
    Value::Object(Map::new())
}

fn resolved_environment_eval_seed_start(request: &OpenEnvRunRequest) -> Option<u64> {
    request.environment_eval.as_ref().map(|config| {
        config.seed_start.unwrap_or_else(|| {
            request
                .seed_start
                .checked_add(u64::try_from(request.groups).unwrap_or(u64::MAX))
                .unwrap_or(u64::MAX)
        })
    })
}

fn pending_environment_evaluation(
    request: &OpenEnvRunRequest,
) -> Option<OpenEnvEnvironmentEvalStatus> {
    let config = request.environment_eval.as_ref()?;
    let seed_start = resolved_environment_eval_seed_start(request)?;
    let rollouts_total = config.groups.saturating_mul(config.group_size);
    Some(OpenEnvEnvironmentEvalStatus {
        state: OpenEnvEnvironmentEvalState::Pending,
        seed_start,
        groups: config.groups,
        group_size: config.group_size,
        baseline: OpenEnvPolicyIdentity {
            adapter: normalized_adapter(&request.adapter),
            adapter_content_revision: None,
            execution_provenance_sha256: None,
        },
        candidate: OpenEnvPolicyIdentity {
            adapter: request
                .output_adapter
                .as_deref()
                .and_then(normalized_adapter),
            adapter_content_revision: None,
            execution_provenance_sha256: None,
        },
        progress: OpenEnvEnvironmentEvalProgress {
            state: OpenEnvEnvironmentEvalState::Pending,
            groups_completed: 0,
            groups_total: config.groups,
            rollouts_completed: 0,
            rollouts_total,
        },
        baseline_stats: None,
        candidate_stats: None,
        evidence: None,
        outcome: None,
        verdict: None,
    })
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
    EnvironmentEvaluating,
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
    /// Optional opaque retry identity. While the resulting run remains
    /// retained, the same key plus normalized request returns that run and the
    /// same key with different semantics fails closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(alias = "environments")]
    pub environment_urls: Vec<String>,
    /// Optional server-configured credential handle aligned with each
    /// environment URL. An empty list means every endpoint is unauthenticated.
    /// Handles persist for audit; bearer values never enter the request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credential_ids: Vec<Option<String>>,
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
    /// Optional reset objects aligned one-for-one with `environment_urls`.
    /// Use this for heterogeneous portfolios; it is mutually exclusive with
    /// non-empty shared `reset_options`. Kiln owns and inserts `seed`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_reset_options: Vec<Value>,
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
    /// Optional paired held-out evaluation against the behavior policy after
    /// training. It reuses the same OpenEnv protocol client with a disjoint
    /// seed range and may own deferred adapter promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_eval: Option<OpenEnvEnvironmentEvalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvRunProgress {
    pub groups_completed: usize,
    pub groups_total: usize,
    pub rollouts_completed: usize,
    pub rollouts_total: usize,
}

/// Bounded server admission state for one persisted OpenEnv workflow.
///
/// Queue position is a live one-based projection and is omitted immediately
/// after the run acquires an execution slot. The admitted timestamp and wait
/// duration remain stable for the rest of the run and in retained history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenEnvRunAdmission {
    pub max_active_runs: usize,
    /// Stable monotonic FIFO order assigned by this persisted registry.
    pub sequence: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_position: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admitted_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_wait_ms: Option<u64>,
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
    /// Immutable corpus admitted by the native trainer. For OpenEnv GRPO this
    /// includes the semantic endpoint/schema/task-plan identity in addition
    /// to the exact byte-level corpus digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<TrainingDataProvenance>,
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
    pub admission: Option<OpenEnvRunAdmission>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_unix_ms: Option<u64>,
    pub progress: OpenEnvRunProgress,
    /// Reward, outcome, recovery, and policy-cost statistics from the exact
    /// artifact-published training collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_stats: Option<OpenEnvRolloutStats>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environments: Vec<OpenEnvIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<OpenEnvArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_job_id: Option<String>,
    /// Exact materialized trainer settings admitted before collection. New
    /// train runs retain this contract for their full lifecycle and execute
    /// from it rather than re-reading defaults from the original request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_contract: Option<OpenEnvTrainingContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_submission: Option<TrainingResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<OpenEnvTrainingStatus>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub post_evaluations: Vec<OpenEnvPostEvalStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_evaluation: Option<OpenEnvEnvironmentEvalStatus>,
    /// Stable, bounded terminal diagnosis for new failed runs. `error` remains
    /// as a compatibility projection for older clients and persisted records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<OpenEnvRunFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl OpenEnvRunStatus {
    fn terminal(&self) -> bool {
        self.state.unconditionally_terminal()
            // In v1, `training_queued` explicitly meant the OpenEnv
            // orchestrator had handed ownership off and finished. Preserve
            // that historical record on upgrade while newer runs keep
            // following every requested phase to a real terminal outcome.
            || (self.schema == OPENENV_RUN_SCHEMA_V1
                && self.state == OpenEnvRunState::TrainingQueued)
    }

    fn pristine_queued(&self) -> bool {
        self.state == OpenEnvRunState::Queued
            && self.admission.as_ref().is_some_and(|admission| {
                admission.admitted_unix_ms.is_none() && admission.queue_wait_ms.is_none()
            })
            && self.progress.groups_completed == 0
            && self.progress.rollouts_completed == 0
            && self.rollout_stats.is_none()
            && self.environments.is_empty()
            && self.artifacts.is_empty()
            && self.training_job_id.is_none()
            && self.training_submission.is_none()
            && self.training.is_none()
            && self.post_evaluations.is_empty()
            && self.failure.is_none()
    }

    fn safely_migratable_v4_queued(&self) -> bool {
        self.schema == OPENENV_RUN_SCHEMA_V4
            && self.training_contract.is_none()
            && self.pristine_queued()
    }

    fn safely_restartable_queued(&self) -> bool {
        self.schema == OPENENV_RUN_SCHEMA_V5
            && self.pristine_queued()
            && match self.kind {
                OpenEnvRunKind::Rollout => self.training_contract.is_none(),
                OpenEnvRunKind::Train => self.training_contract.as_ref().is_some_and(|contract| {
                    openenv_training_contract_matches_request(&self.request, contract)
                }),
            }
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
    #[serde(default)]
    credential_ids: Vec<Option<String>>,
}

#[derive(Debug, Serialize)]
struct OpenEnvInspectResponse {
    schema: &'static str,
    environments: Vec<OpenEnvInspection>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenEnvTaskCatalogRequest {
    #[serde(alias = "environments")]
    environment_urls: Vec<String>,
    #[serde(default)]
    credential_ids: Vec<Option<String>>,
    #[serde(default)]
    environment_name: Option<String>,
    #[serde(default)]
    split: Option<String>,
    #[serde(default)]
    start: u64,
    #[serde(default = "default_task_page_limit")]
    limit: usize,
}

#[derive(Debug, Serialize)]
struct OpenEnvTaskCatalogEntry {
    base_url: String,
    catalog: OpenEnvTaskCatalog,
}

#[derive(Debug, Serialize)]
struct OpenEnvTaskCatalogResponse {
    schema: &'static str,
    catalogs: Vec<OpenEnvTaskCatalogEntry>,
}

struct TrackedOpenEnvRun {
    status: OpenEnvRunStatus,
    control: OpenEnvRunControl,
}

enum OpenEnvRunInsertOutcome {
    Created {
        status: OpenEnvRunStatus,
        control: OpenEnvRunControl,
    },
    Replayed(OpenEnvRunStatus),
}

#[derive(Debug, thiserror::Error)]
#[error("OpenEnv idempotency key {key:?} is already bound to a different normalized request")]
struct OpenEnvIdempotencyConflict {
    key: String,
}

#[derive(Clone)]
struct OpenEnvRunControl {
    cancel: Arc<AtomicBool>,
    cancelled: Arc<Notify>,
}

impl OpenEnvRunControl {
    fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(Notify::new()),
        }
    }

    fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.cancelled.notify_one();
    }

    async fn wait_cancelled(&self) {
        loop {
            let notified = self.cancelled.notified();
            if self.cancel.load(Ordering::Relaxed) {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Default)]
struct OpenEnvAdmissionQueue {
    active: usize,
    next_sequence: u64,
    queued: VecDeque<String>,
}

struct OpenEnvRunPermit {
    registry: Arc<OpenEnvRunRegistry>,
}

impl Drop for OpenEnvRunPermit {
    fn drop(&mut self) {
        let mut admission = self
            .registry
            .admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        admission.active = admission.active.saturating_sub(1);
        drop(admission);
        self.registry.admission_changed.notify_waiters();
    }
}

/// Bounded, persisted registry for server-owned OpenEnv rollout runs.
pub struct OpenEnvRunRegistry {
    root: PathBuf,
    policy: OpenEnvConfig,
    runs: RwLock<HashMap<String, TrackedOpenEnvRun>>,
    admission: Mutex<OpenEnvAdmissionQueue>,
    admission_changed: Notify,
}

impl std::fmt::Debug for OpenEnvRunRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvRunRegistry")
            .field("root", &self.root)
            .field("policy", &self.policy)
            .field("tracked_runs", &self.runs.read().unwrap().len())
            .field(
                "active_runs",
                &self
                    .admission
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .active,
            )
            .finish()
    }
}

fn normalized_run_requests_match(
    left: &OpenEnvRunRequest,
    right: &OpenEnvRunRequest,
) -> Result<bool> {
    let left = serde_json::to_value(left).context("normalize submitted OpenEnv run request")?;
    let right = serde_json::to_value(right).context("normalize retained OpenEnv run request")?;
    Ok(left == right)
}

pub(crate) fn validate_openenv_idempotency_key(key: &str) -> Result<()> {
    anyhow::ensure!(
        !key.is_empty() && key.len() <= MAX_IDEMPOTENCY_KEY_BYTES,
        "OpenEnv idempotency_key must contain 1..={MAX_IDEMPOTENCY_KEY_BYTES} ASCII bytes"
    );
    anyhow::ensure!(
        key.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        }),
        "OpenEnv idempotency_key may contain only ASCII letters, digits, '.', '_', ':', or '-'"
    );
    Ok(())
}

impl OpenEnvRunRegistry {
    pub fn new(adapter_dir: PathBuf) -> Self {
        Self {
            root: adapter_dir.join(".openenv").join("runs"),
            policy: OpenEnvConfig::default(),
            runs: RwLock::new(HashMap::new()),
            admission: Mutex::new(OpenEnvAdmissionQueue::default()),
            admission_changed: Notify::new(),
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
            admission: Mutex::new(OpenEnvAdmissionQueue::default()),
            admission_changed: Notify::new(),
        };
        registry.restore()?;
        Ok(registry)
    }

    pub fn policy(&self) -> &OpenEnvConfig {
        &self.policy
    }

    fn restore(&self) -> Result<()> {
        let mut restored = Vec::new();
        let now = now_unix_ms();
        let ttl_ms = self.policy.tracked_run_ttl_secs.saturating_mul(1000);
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
            // v4 was the first restartable FIFO format, but it predated a
            // persisted effective trainer contract. Seal a pristine queued v4
            // request exactly once under the current deployment, then resume
            // only the resulting v5 record. Started or terminal history is
            // never rewritten.
            if status.safely_migratable_v4_queued() {
                match materialized_openenv_training_contract(&status.request) {
                    Ok(training_contract) => {
                        status.schema = OPENENV_RUN_SCHEMA_V5.to_string();
                        status.training_contract = training_contract;
                        persist_status_to(&status_path, &status)?;
                    }
                    Err(error) => {
                        let message = format!(
                            "Kiln could not migrate this queued OpenEnv v4 training contract: {error:#}"
                        );
                        status.state = OpenEnvRunState::Failed;
                        status.finished_unix_ms = Some(now);
                        status.failure = Some(OpenEnvRunFailure::explicit(
                            OpenEnvRunFailureCode::PersistedContractInvalid,
                            OpenEnvRunFailureStage::Restoration,
                            false,
                            &message,
                            "Correct the retained request or trainer configuration before submitting a new run.",
                            now,
                        ));
                        status.error = Some(message);
                        if let Some(admission) = status.admission.as_mut() {
                            admission.queue_position = None;
                        }
                        persist_status_to(&status_path, &status)?;
                    }
                }
            }
            if !status.terminal() && !status.safely_restartable_queued() {
                let finished_unix_ms = now_unix_ms();
                let message =
                    "Kiln restarted before this OpenEnv run reached a terminal state".to_string();
                status.state = OpenEnvRunState::Failed;
                status.finished_unix_ms = Some(finished_unix_ms);
                status.failure = Some(OpenEnvRunFailure::explicit(
                    OpenEnvRunFailureCode::RunInterrupted,
                    OpenEnvRunFailureStage::Restoration,
                    true,
                    &message,
                    "Submit a new run; OpenEnv sessions and trainer ownership cannot be assumed resumable after restart.",
                    finished_unix_ms,
                ));
                status.error = Some(message);
                if let Some(admission) = status.admission.as_mut() {
                    admission.queue_position = None;
                }
                persist_status_to(&status_path, &status)?;
            }
            if status
                .finished_unix_ms
                .is_some_and(|finished| now.saturating_sub(finished) > ttl_ms)
            {
                continue;
            }
            restored.push(status);
        }
        restored.sort_by(|left, right| {
            match (
                left.safely_restartable_queued(),
                right.safely_restartable_queued(),
            ) {
                (true, true) => left
                    .admission
                    .as_ref()
                    .map(|admission| admission.sequence)
                    .cmp(&right.admission.as_ref().map(|admission| admission.sequence))
                    .then_with(|| left.run_id.cmp(&right.run_id)),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => right.submitted_unix_ms.cmp(&left.submitted_unix_ms),
            }
        });
        let next_sequence = restored
            .iter()
            .filter_map(|status| {
                status
                    .admission
                    .as_ref()
                    .map(|admission| admission.sequence)
            })
            .max()
            .unwrap_or_default();
        for status in restored.iter_mut().skip(self.policy.max_tracked_runs) {
            if status.safely_restartable_queued() {
                let message = format!(
                    "Kiln restarted with openenv.max_tracked_runs={} before this queued run could be restored",
                    self.policy.max_tracked_runs
                );
                status.state = OpenEnvRunState::Failed;
                status.finished_unix_ms = Some(now);
                status.failure = Some(OpenEnvRunFailure::explicit(
                    OpenEnvRunFailureCode::RunAdmissionFailed,
                    OpenEnvRunFailureStage::Restoration,
                    true,
                    &message,
                    "Raise openenv.max_tracked_runs or reduce retained history before submitting a new run.",
                    now,
                ));
                status.error = Some(message);
                if let Some(admission) = status.admission.as_mut() {
                    admission.queue_position = None;
                }
                persist_status_to(&self.status_path(&status.run_id), status)?;
            }
        }
        restored.truncate(self.policy.max_tracked_runs);
        // Historical run directories are intentionally durable even after
        // their statuses fall outside the bounded registry. Validate key
        // uniqueness only after applying that retention boundary: an evicted
        // terminal run no longer owns its key, while two actually retained
        // records with one key remain a startup-failing invariant violation.
        let mut restored_idempotency_keys = HashMap::new();
        for status in &restored {
            let Some(key) = status.request.idempotency_key.as_deref() else {
                continue;
            };
            validate_openenv_idempotency_key(key).with_context(|| {
                format!(
                    "validate retained OpenEnv idempotency key for run {}",
                    status.run_id
                )
            })?;
            if let Some(previous_run_id) =
                restored_idempotency_keys.insert(key.to_string(), status.run_id.clone())
            {
                anyhow::bail!(
                    "retained OpenEnv runs {previous_run_id} and {} share idempotency key {key:?}",
                    status.run_id
                );
            }
        }
        let mut restored_queue = VecDeque::new();
        let mut runs = self.runs.write().unwrap();
        for mut status in restored {
            if status.safely_restartable_queued() {
                restored_queue.push_back(status.run_id.clone());
                if let Some(admission) = status.admission.as_mut() {
                    admission.max_active_runs = self.policy.max_active_runs;
                    admission.queue_position = Some(restored_queue.len());
                }
                persist_status_to(&self.status_path(&status.run_id), &status)?;
            }
            runs.insert(
                status.run_id.clone(),
                TrackedOpenEnvRun {
                    status,
                    control: OpenEnvRunControl::new(),
                },
            );
        }
        drop(runs);
        let mut admission = self
            .admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        admission.next_sequence = next_sequence;
        admission.queued = restored_queue;
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

    fn idempotent_status_locked(
        runs: &HashMap<String, TrackedOpenEnvRun>,
        request: &OpenEnvRunRequest,
    ) -> Result<Option<OpenEnvRunStatus>> {
        let Some(key) = request.idempotency_key.as_deref() else {
            return Ok(None);
        };
        let Some(existing) = runs
            .values()
            .find(|tracked| tracked.status.request.idempotency_key.as_deref() == Some(key))
        else {
            return Ok(None);
        };
        if !normalized_run_requests_match(request, &existing.status.request)? {
            return Err(OpenEnvIdempotencyConflict {
                key: key.to_string(),
            }
            .into());
        }
        Ok(Some(existing.status.clone()))
    }

    fn replay_idempotent(&self, request: &OpenEnvRunRequest) -> Result<Option<OpenEnvRunStatus>> {
        let mut runs = self.runs.write().unwrap();
        self.prune_locked(&mut runs);
        let status = Self::idempotent_status_locked(&runs, request)?;
        drop(runs);
        Ok(status.map(|status| self.project_admission(status)))
    }

    fn insert(
        &self,
        request: OpenEnvRunRequest,
        training_contract: Option<OpenEnvTrainingContract>,
    ) -> Result<OpenEnvRunInsertOutcome> {
        anyhow::ensure!(self.policy.enabled, "OpenEnv control plane is disabled");
        let contract_kind_valid = match (request.kind, training_contract.as_ref()) {
            (OpenEnvRunKind::Rollout, None) => true,
            (OpenEnvRunKind::Train, Some(contract)) => {
                openenv_training_contract_matches_request(&request, contract)
            }
            _ => false,
        };
        anyhow::ensure!(
            contract_kind_valid,
            "OpenEnv run kind, request, and admitted training contract disagree"
        );
        let mut runs = self.runs.write().unwrap();
        self.prune_locked(&mut runs);
        if let Some(status) = Self::idempotent_status_locked(&runs, &request)? {
            drop(runs);
            return Ok(OpenEnvRunInsertOutcome::Replayed(
                self.project_admission(status),
            ));
        }
        self.make_room_locked(&mut runs);
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
        let environment_evaluation = pending_environment_evaluation(&request);
        let submitted_unix_ms = now_unix_ms();
        let mut admission = self
            .admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        admission.next_sequence = admission
            .next_sequence
            .checked_add(1)
            .context("OpenEnv FIFO admission sequence exhausted")?;
        let sequence = admission.next_sequence;
        let queue_position = admission.queued.len().saturating_add(1);
        let status = OpenEnvRunStatus {
            schema: OPENENV_RUN_SCHEMA_V5.to_string(),
            run_id: run_id.clone(),
            kind: request.kind,
            state: OpenEnvRunState::Queued,
            submitted_unix_ms,
            admission: Some(OpenEnvRunAdmission {
                max_active_runs: self.policy.max_active_runs,
                sequence,
                queue_position: Some(queue_position),
                admitted_unix_ms: None,
                queue_wait_ms: None,
            }),
            finished_unix_ms: None,
            progress: OpenEnvRunProgress {
                groups_completed: 0,
                groups_total: request.groups,
                rollouts_completed: 0,
                rollouts_total,
            },
            rollout_stats: None,
            request,
            environments: Vec::new(),
            artifacts: Vec::new(),
            training_job_id: None,
            training_contract,
            training_submission: None,
            training: None,
            post_evaluations: Vec::new(),
            environment_evaluation,
            failure: None,
            error: None,
        };
        persist_status_to(&run_dir.join("run.json"), &status)?;
        let control = OpenEnvRunControl::new();
        admission.queued.push_back(run_id.clone());
        runs.insert(
            run_id,
            TrackedOpenEnvRun {
                status: status.clone(),
                control: control.clone(),
            },
        );
        drop(admission);
        self.admission_changed.notify_waiters();
        Ok(OpenEnvRunInsertOutcome::Created { status, control })
    }

    fn get(&self, run_id: &str) -> Option<OpenEnvRunStatus> {
        let status = self
            .runs
            .read()
            .unwrap()
            .get(run_id)
            .map(|tracked| tracked.status.clone())?;
        Some(self.project_admission(status))
    }

    fn list(&self) -> Vec<OpenEnvRunStatus> {
        let mut tracked = self.runs.write().unwrap();
        self.prune_locked(&mut tracked);
        let mut runs = tracked
            .values()
            .map(|tracked| tracked.status.clone())
            .collect::<Vec<_>>();
        runs.sort_by_key(|status| std::cmp::Reverse(status.submitted_unix_ms));
        runs.into_iter()
            .map(|status| self.project_admission(status))
            .collect()
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        let mut runs = self.runs.write().unwrap();
        self.prune_locked(&mut runs);
        let tracked = runs.len();
        drop(runs);
        let admission = self
            .admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (admission.active, admission.queued.len(), tracked)
    }

    fn queued_controls(&self) -> Vec<(String, OpenEnvRunControl)> {
        let queued = self
            .admission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .queued
            .clone();
        let runs = self.runs.read().unwrap();
        queued
            .into_iter()
            .filter_map(|run_id| {
                runs.get(&run_id)
                    .map(|tracked| (run_id, tracked.control.clone()))
            })
            .collect()
    }

    fn project_admission(&self, mut status: OpenEnvRunStatus) -> OpenEnvRunStatus {
        if let Some(admission_status) = status.admission.as_mut() {
            let admission = self
                .admission
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            admission_status.queue_position = admission
                .queued
                .iter()
                .position(|queued| queued == &status.run_id)
                .map(|position| position.saturating_add(1));
        }
        status
    }

    async fn acquire(
        self: &Arc<Self>,
        run_id: &str,
        control: &OpenEnvRunControl,
    ) -> Result<OpenEnvRunPermit> {
        loop {
            let changed = self.admission_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let admitted = {
                let mut admission = self
                    .admission
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let at_front = admission
                    .queued
                    .front()
                    .is_some_and(|queued| queued == run_id);
                if !control.cancel.load(Ordering::Relaxed)
                    && at_front
                    && admission.active < self.policy.max_active_runs
                {
                    admission.queued.pop_front();
                    admission.active = admission.active.saturating_add(1);
                    true
                } else {
                    false
                }
            };
            if admitted {
                let permit = OpenEnvRunPermit {
                    registry: self.clone(),
                };
                let admitted_unix_ms = now_unix_ms();
                self.update(run_id, |status| {
                    status.state = OpenEnvRunState::Discovering;
                    if let Some(admission) = status.admission.as_mut() {
                        admission.queue_position = None;
                        admission.admitted_unix_ms = Some(admitted_unix_ms);
                        admission.queue_wait_ms =
                            Some(admitted_unix_ms.saturating_sub(status.submitted_unix_ms));
                    }
                })?;
                self.admission_changed.notify_waiters();
                return Ok(permit);
            }
            tokio::select! {
                _ = &mut changed => {}
                _ = control.wait_cancelled() => anyhow::bail!("OpenEnv run cancelled while queued"),
            }
        }
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

    fn update_environments(&self, run_id: &str, environments: Vec<OpenEnvIdentity>) {
        if let Err(error) = self.update(run_id, |status| {
            status.environments = environments;
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

    fn cancel(&self, run_id: &str) -> Result<(OpenEnvRunStatus, bool)> {
        let tracked = self
            .runs
            .read()
            .unwrap()
            .get(run_id)
            .map(|tracked| (tracked.status.clone(), tracked.control.clone()))
            .with_context(|| format!("OpenEnv run {run_id} was not found"))?;
        anyhow::ensure!(
            !tracked.0.terminal(),
            "OpenEnv run {run_id} cannot be cancelled from state {:?}",
            tracked.0.state
        );
        tracked.1.request_cancel();
        let queued = {
            let admission = self
                .admission
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            admission.queued.iter().any(|queued| queued == run_id)
        };
        let status = self.update(run_id, |status| {
            if queued {
                status.state = OpenEnvRunState::Cancelled;
                status.finished_unix_ms = Some(now_unix_ms());
                status.failure = None;
                status.error =
                    Some("OpenEnv run cancelled while waiting for execution capacity".into());
                if let Some(admission) = status.admission.as_mut() {
                    admission.queue_position = None;
                }
            } else {
                status.error =
                    Some("Cancellation requested; the active protocol boundary will stop".into());
            }
        })?;
        if queued {
            let mut admission = self
                .admission
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(position) = admission.queued.iter().position(|queued| queued == run_id) {
                admission.queued.remove(position);
            }
        }
        self.admission_changed.notify_waiters();
        Ok((self.project_admission(status), queued))
    }

    fn artifact_path(
        &self,
        run_id: &str,
        kind: &str,
    ) -> Option<(PathBuf, &'static str, OpenEnvArtifact)> {
        let status = self.get(run_id)?;
        let artifact = status
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == kind)?
            .clone();
        let filename = match kind {
            "dataset" => ("rollouts.jsonl", "application/x-ndjson"),
            "replay" => ("replay.json", "application/json"),
            "summary" => ("summary.json", "application/json"),
            "train_receipt" => (kiln_train::TRAIN_RECEIPT_FILENAME, "application/json"),
            "adapter_manifest" => (kiln_train::ADAPTER_MANIFEST_FILENAME, "application/json"),
            "environment_eval_baseline_dataset" => (
                "environment-evaluation/baseline/rollouts.jsonl",
                "application/x-ndjson",
            ),
            "environment_eval_baseline_replay" => (
                "environment-evaluation/baseline/replay.json",
                "application/json",
            ),
            "environment_eval_baseline_summary" => (
                "environment-evaluation/baseline/summary.json",
                "application/json",
            ),
            "environment_eval_candidate_dataset" => (
                "environment-evaluation/candidate/rollouts.jsonl",
                "application/x-ndjson",
            ),
            "environment_eval_candidate_replay" => (
                "environment-evaluation/candidate/replay.json",
                "application/json",
            ),
            "environment_eval_candidate_summary" => (
                "environment-evaluation/candidate/summary.json",
                "application/json",
            ),
            "environment_eval_receipt" => {
                ("environment-evaluation/receipt.json", "application/json")
            }
            _ => return None,
        };
        Some((self.run_dir(run_id).join(filename.0), filename.1, artifact))
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

fn resolve_credential_envs(
    policy: &OpenEnvConfig,
    credential_ids: &[Option<String>],
    environment_urls: &[String],
) -> Result<Vec<Option<String>>, ApiError> {
    policy
        .resolve_credential_envs(credential_ids, environment_urls)
        .map_err(|error| {
            openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_credential",
                error,
                "Configure an origin-scoped [openenv.credentials.<id>] bearer_token_env and align credential_ids with environment_urls.",
            )
        })
}

fn validate_run_idempotency_key(request: &OpenEnvRunRequest) -> Result<(), ApiError> {
    let Some(key) = request.idempotency_key.as_deref() else {
        return Ok(());
    };
    validate_openenv_idempotency_key(key).map_err(|error| {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_idempotency_key",
            error,
            "Use 1..=128 ASCII letters, digits, '.', '_', ':', or '-'; do not place a secret in this persisted field.",
        )
    })
}

fn validate_run_request(
    request: &OpenEnvRunRequest,
    policy: &OpenEnvConfig,
) -> Result<(), ApiError> {
    validate_run_idempotency_key(request)?;
    validate_environment_urls(&request.environment_urls, policy.allow_remote_environments)?;
    let credential_envs =
        resolve_credential_envs(policy, &request.credential_ids, &request.environment_urls)?;
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
    if !request.environment_reset_options.is_empty() {
        if request.environment_reset_options.len() != request.environment_urls.len() {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_request",
                format!(
                    "environment_reset_options must contain exactly one object per environment (expected {}, got {})",
                    request.environment_urls.len(),
                    request.environment_reset_options.len()
                ),
                "Align one reset object with each environment_urls entry.",
            ));
        }
        if request
            .environment_reset_options
            .iter()
            .any(|value| !value.is_object())
        {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_request",
                "environment_reset_options entries must all be JSON objects",
                "Send one reset object per environment; Kiln adds each deterministic seed.",
            ));
        }
        if request
            .reset_options
            .as_object()
            .is_some_and(|object| !object.is_empty())
        {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_request",
                "reset_options and environment_reset_options are mutually exclusive",
                "Use either one shared reset object or one aligned object per environment.",
            ));
        }
    }
    match request.kind {
        OpenEnvRunKind::Rollout
            if request.output_adapter.is_some()
                || request.training_config.is_some()
                || request.post_eval.is_some()
                || request.environment_eval.is_some() =>
        {
            return Err(openenv_error(
                StatusCode::BAD_REQUEST,
                "openenv_invalid_request",
                "output_adapter, training_config, post_eval, and environment_eval are valid only for kind=train",
                "Remove training-only fields or set kind to train.",
            ));
        }
        OpenEnvRunKind::Train => {
            let output = request.output_adapter.as_deref().unwrap_or_default();
            crate::api::adapters::validate_adapter_name(output)?;
            if let Some(config) = &request.environment_eval {
                validate_environment_eval(request, output, config, &credential_envs)?;
            }
        }
        OpenEnvRunKind::Rollout => {}
    }
    validate_options(&rollout_options_for(
        request,
        Path::new("."),
        credential_envs,
    ))
    .map_err(|error| {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_request",
            error,
            "Correct the OpenEnv collection bounds and retry.",
        )
    })?;
    Ok(())
}

/// Materialize the exact native-GRPO configuration owned by an OpenEnv train
/// workflow. The request may customize ordinary trainer controls, while Kiln
/// fixes all fields whose meaning is determined by the live rollout.
fn materialize_openenv_grpo_config(
    mut config: GrpoConfig,
    behavior_adapter: &str,
    output_adapter: &str,
    auto_load: bool,
    external_promotion_gate_pending: bool,
) -> GrpoConfig {
    config.output_name = Some(output_adapter.to_string());
    // A native environment gate owns promotion. Keep the prior policy active
    // until paired held-out returns pass; diagnostic environment evaluation
    // preserves the ordinary training auto-load behavior.
    config.auto_load = auto_load && !external_promotion_gate_pending;
    config.behavior_policy = BehaviorPolicy::NoImportanceCorrection;
    config.base_adapter = normalized_adapter(behavior_adapter);
    config
}

fn effective_openenv_grpo_config(request: &OpenEnvRunRequest, output_adapter: &str) -> GrpoConfig {
    materialize_openenv_grpo_config(
        request.training_config.clone().unwrap_or_default(),
        &request.adapter,
        output_adapter,
        request.auto_load,
        request
            .environment_eval
            .as_ref()
            .is_some_and(|config| config.gate.is_some()),
    )
}

fn materialized_openenv_training_contract(
    request: &OpenEnvRunRequest,
) -> Result<Option<OpenEnvTrainingContract>> {
    if request.kind == OpenEnvRunKind::Rollout {
        return Ok(None);
    }
    let output_adapter = request
        .output_adapter
        .as_deref()
        .context("OpenEnv train run has no output adapter")?;
    Ok(Some(OpenEnvTrainingContract {
        schema: OPENENV_TRAINING_CONTRACT_SCHEMA_V1.to_string(),
        effective_config: effective_openenv_grpo_config(request, output_adapter),
        post_eval: request.post_eval.clone(),
        behavior_policy: None,
    }))
}

fn openenv_training_contract_matches_request(
    request: &OpenEnvRunRequest,
    contract: &OpenEnvTrainingContract,
) -> bool {
    let Some(expected_output) = request.output_adapter.as_deref() else {
        return false;
    };
    let expected_config = effective_openenv_grpo_config(request, expected_output);
    contract.schema == OPENENV_TRAINING_CONTRACT_SCHEMA_V1
        && request.kind == OpenEnvRunKind::Train
        && serde_json::to_value(&contract.effective_config).ok()
            == serde_json::to_value(&expected_config).ok()
        && serde_json::to_value(&contract.post_eval).ok()
            == serde_json::to_value(&request.post_eval).ok()
        && contract.behavior_policy.as_ref().is_none_or(|policy| {
            policy.validate().is_ok()
                && policy.adapter.as_ref().map(|adapter| adapter.name.as_str())
                    == normalized_adapter(&request.adapter).as_deref()
        })
}

fn validate_openenv_training_contract(
    state: &AppState,
    behavior_adapter: &str,
    config: &GrpoConfig,
    post_eval: Option<&kiln_eval::PostEvalConfig>,
) -> Result<kiln_train::RolloutBehaviorPolicyIdentityV1, ApiError> {
    // OpenEnv trajectories always carry explicit observation segments, so
    // validate the environment-token branch of the native kt-tape contract.
    super::training::validate_grpo_config_at_submit(config, true)?;
    super::training::validate_post_eval_suite(state, post_eval)?;
    if let Some(adapter) = normalized_adapter(behavior_adapter) {
        super::adapters::validate_loadable_adapter_dir(&state.adapter_dir.join(adapter))?;
    }
    super::training::ensure_training_backend_admission(state)?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    super::training::enforce_training_workload_admission(state, TrainingWorkload::Grpo)?;
    super::training::enforce_training_optimizer_admission(
        state,
        config.optimizer,
        config.lora_rank,
    )?;
    state
        .openenv_behavior_policy_identity(config.base_adapter.as_deref())
        .map_err(|error| {
            openenv_error(
                StatusCode::CONFLICT,
                "openenv_behavior_policy_unavailable",
                error,
                "Use a loadable immutable adapter revision and a real-model Kiln server, then retry.",
            )
        })
}

/// Reject a doomed OpenEnv train workflow before it is persisted or opens an
/// environment session. Final GRPO queue admission remains authoritative for
/// transient capacity and live memory, which may change while collection is
/// running; immutable request/backend/suite contracts are safe to prove now.
fn validate_openenv_training_preflight(
    state: &AppState,
    request: &OpenEnvRunRequest,
) -> Result<Option<OpenEnvTrainingContract>, ApiError> {
    if request.kind != OpenEnvRunKind::Train {
        return Ok(None);
    }
    let output_adapter = request.output_adapter.as_deref().ok_or_else(|| {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_request",
            "kind=train requires output_adapter",
            "Choose a path-safe output adapter name.",
        )
    })?;
    let config = effective_openenv_grpo_config(request, output_adapter);
    let behavior_policy = validate_openenv_training_contract(
        state,
        &request.adapter,
        &config,
        request.post_eval.as_ref(),
    )?;
    Ok(Some(OpenEnvTrainingContract {
        schema: OPENENV_TRAINING_CONTRACT_SCHEMA_V1.to_string(),
        effective_config: config,
        post_eval: request.post_eval.clone(),
        behavior_policy: Some(behavior_policy),
    }))
}

fn openenv_training_capacity_snapshot(
    state: &AppState,
) -> Result<OpenEnvTrainingCapacitySnapshot, ApiError> {
    // Match native admission's tracked-jobs -> queue lock order and retain
    // both guards while reading, so the returned evidence is one coherent
    // point-in-time snapshot. It deliberately stops being authoritative as
    // soon as the guards are released.
    let tracked = state
        .training_jobs
        .read()
        .map_err(|_| ApiError::internal("training job map poisoned during OpenEnv preflight"))?;
    let queue = state
        .training_queue
        .lock()
        .map_err(|_| ApiError::internal("training queue lock poisoned during OpenEnv preflight"))?;
    let queued_jobs = queue.len();
    if queued_jobs >= state.max_queued_training_jobs {
        return Err(ApiError::training_queue_full(
            state.max_queued_training_jobs,
        ));
    }
    let tracked_jobs = tracked.len();
    if tracked_jobs >= state.max_tracked_jobs {
        return Err(ApiError::training_tracked_full(state.max_tracked_jobs));
    }
    Ok(OpenEnvTrainingCapacitySnapshot {
        checked_unix_ms: now_unix_ms(),
        queued_jobs,
        max_queued_jobs: state.max_queued_training_jobs,
        tracked_jobs,
        max_tracked_jobs: state.max_tracked_jobs,
    })
}

fn preflight_training_inner(
    state: &AppState,
    request: OpenEnvTrainingPreflightRequest,
) -> Result<OpenEnvTrainingPreflightReceipt, ApiError> {
    super::adapters::validate_adapter_name(&request.output_adapter)?;
    if normalized_adapter(&request.adapter).is_some() {
        super::adapters::validate_adapter_name(request.adapter.trim())?;
    }
    let config = materialize_openenv_grpo_config(
        request.training_config,
        &request.adapter,
        &request.output_adapter,
        request.auto_load,
        false,
    );
    let post_eval = request.post_eval;
    let behavior_policy =
        validate_openenv_training_contract(state, &request.adapter, &config, post_eval.as_ref())?;
    let capacity = openenv_training_capacity_snapshot(state)?;
    tracing::info!(
        behavior_adapter = %request.adapter,
        output_adapter = %request.output_adapter,
        lora_rank = config.lora_rank,
        optimizer = ?config.optimizer.kind(),
        queued_jobs = capacity.queued_jobs,
        tracked_jobs = capacity.tracked_jobs,
        "OpenEnv training preflight accepted before collection"
    );
    Ok(OpenEnvTrainingPreflightReceipt {
        schema: OPENENV_TRAINING_PREFLIGHT_SCHEMA_V1.to_string(),
        effective_config: config,
        post_eval,
        behavior_policy: Some(behavior_policy),
        capacity,
        capacity_reserved: false,
    })
}

async fn preflight_training(
    State(state): State<AppState>,
    Json(request): Json<OpenEnvTrainingPreflightRequest>,
) -> Result<Json<OpenEnvTrainingPreflightReceipt>, ApiError> {
    match preflight_training_inner(&state, request) {
        Ok(receipt) => {
            state
                .metrics
                .openenv_training_preflights_accepted
                .fetch_add(1, Ordering::Relaxed);
            Ok(Json(receipt))
        }
        Err(error) => {
            state
                .metrics
                .openenv_training_preflights_rejected
                .fetch_add(1, Ordering::Relaxed);
            Err(error)
        }
    }
}

fn validate_environment_eval(
    request: &OpenEnvRunRequest,
    output_adapter: &str,
    config: &OpenEnvEnvironmentEvalConfig,
    credential_envs: &[Option<String>],
) -> Result<(), ApiError> {
    let fail = |message: String| {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_environment_eval",
            message,
            "Use a non-overlapping held-out seed range and valid bounded paired-evaluation settings.",
        )
    };
    if normalized_adapter(&request.adapter) == normalized_adapter(output_adapter) {
        return Err(fail(
            "environment_eval requires output_adapter to differ from the behavior adapter so the baseline revision remains evaluable".to_string(),
        ));
    }
    let training_groups = u64::try_from(request.groups)
        .map_err(|_| fail("training groups exceed the supported seed range".to_string()))?;
    let training_end = request
        .seed_start
        .checked_add(training_groups)
        .ok_or_else(|| fail("training seed range overflows u64".to_string()))?;
    let eval_start = config.seed_start.unwrap_or(training_end);
    let eval_groups = u64::try_from(config.groups)
        .map_err(|_| fail("environment_eval groups exceed the supported seed range".to_string()))?;
    let eval_end = eval_start
        .checked_add(eval_groups)
        .ok_or_else(|| fail("environment_eval seed range overflows u64".to_string()))?;
    if request.seed_start < eval_end && eval_start < training_end {
        return Err(fail(format!(
            "environment_eval seed range [{eval_start}, {eval_end}) overlaps training range [{}, {training_end})",
            request.seed_start
        )));
    }
    if let Some(gate) = &config.gate {
        if request
            .post_eval
            .as_ref()
            .is_some_and(|post_eval| post_eval.min_accuracy.is_some())
        {
            return Err(fail(
                "environment_eval.gate cannot be combined with post_eval.min_accuracy; one workflow may own only one automatic promotion gate".to_string(),
            ));
        }
        if gate.min_mean_return.is_some_and(|value| !value.is_finite())
            || !gate.min_mean_improvement.is_finite()
            || gate.min_mean_improvement < 0.0
        {
            return Err(fail(
                "environment_eval gate thresholds must be finite and min_mean_improvement must be non-negative".to_string(),
            ));
        }
        if config.groups < crate::openenv_evaluation::OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS {
            return Err(fail(format!(
                "environment_eval gate requires at least {} distinct paired seed groups, got {}",
                crate::openenv_evaluation::OPENENV_ENVIRONMENT_EVAL_MIN_GROUPS,
                config.groups
            )));
        }
    }
    let options = environment_eval_rollout_options(
        request,
        Path::new("."),
        output_adapter,
        "candidate",
        credential_envs.to_vec(),
    )
    .map_err(&fail)?;
    validate_options(&options).map_err(|error| fail(error.to_string()))
}

fn rollout_options_for(
    request: &OpenEnvRunRequest,
    run_dir: &Path,
    credential_envs: Vec<Option<String>>,
) -> OpenEnvRolloutOptions {
    let has_environment_reset_options = !request.environment_reset_options.is_empty();
    OpenEnvRolloutOptions {
        kiln_url: "in-process".to_string(),
        environment_urls: request.environment_urls.clone(),
        credential_envs,
        adapter: request.adapter.clone(),
        groups: request.groups,
        group_size: request.group_size,
        seed_start: request.seed_start,
        reset_options: None,
        reset_options_value: (!has_environment_reset_options)
            .then(|| request.reset_options.clone()),
        environment_reset_options: Vec::new(),
        environment_reset_options_values: request.environment_reset_options.clone(),
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

fn environment_eval_rollout_options(
    request: &OpenEnvRunRequest,
    run_dir: &Path,
    adapter: &str,
    side: &str,
    credential_envs: Vec<Option<String>>,
) -> Result<OpenEnvRolloutOptions, String> {
    let config = request
        .environment_eval
        .as_ref()
        .ok_or_else(|| "OpenEnv environment evaluation was not requested".to_string())?;
    let seed_start = resolved_environment_eval_seed_start(request)
        .ok_or_else(|| "OpenEnv environment evaluation seed is unavailable".to_string())?;
    let paths = evaluation_paths(run_dir, side);
    let has_environment_reset_options = !request.environment_reset_options.is_empty();
    Ok(OpenEnvRolloutOptions {
        kiln_url: "in-process".to_string(),
        environment_urls: request.environment_urls.clone(),
        credential_envs,
        adapter: adapter.to_string(),
        groups: config.groups,
        group_size: config.group_size,
        seed_start,
        reset_options: None,
        reset_options_value: (!has_environment_reset_options)
            .then(|| request.reset_options.clone()),
        environment_reset_options: Vec::new(),
        environment_reset_options_values: request.environment_reset_options.clone(),
        max_steps: request.max_steps,
        concurrency: request.concurrency.min(config.group_size),
        max_action_tokens: request.max_action_tokens,
        temperature: request.temperature,
        thinking: request.thinking,
        protocol_error_reward: request.protocol_error_reward,
        max_recoverable_errors: request.max_recoverable_errors,
        capacity_wait_seconds: request.capacity_wait_seconds,
        output: paths.output,
        replay_output: paths.replay_output,
        summary_output: paths.summary_output,
    })
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
    let credential_envs = resolve_credential_envs(
        state.openenv_runs.policy(),
        &request.credential_ids,
        &request.environment_urls,
    )?;
    if credential_envs.iter().any(Option::is_some) {
        state
            .metrics
            .openenv_authenticated_inspections
            .fetch_add(1, Ordering::Relaxed);
    }
    let environments = stream::iter(request.environment_urls)
        .zip(stream::iter(credential_envs))
        .map(|(url, credential_env)| async move {
            let client = crate::openenv_cli::openenv_client(&url, credential_env.as_deref())?;
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

async fn inspect_tasks(
    State(state): State<AppState>,
    Json(request): Json<OpenEnvTaskCatalogRequest>,
) -> Result<Json<OpenEnvTaskCatalogResponse>, ApiError> {
    state
        .metrics
        .openenv_task_catalog_inspections_started
        .fetch_add(1, Ordering::Relaxed);
    let result = inspect_tasks_inner(&state, request).await;
    if result.is_ok() {
        state
            .metrics
            .openenv_task_catalog_inspections_completed
            .fetch_add(1, Ordering::Relaxed);
    } else {
        state
            .metrics
            .openenv_task_catalog_inspections_failed
            .fetch_add(1, Ordering::Relaxed);
    }
    result.map(Json)
}

async fn inspect_tasks_inner(
    state: &AppState,
    request: OpenEnvTaskCatalogRequest,
) -> Result<OpenEnvTaskCatalogResponse, ApiError> {
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
    if !(1..=MAX_OPENENV_TASK_PAGE_SIZE).contains(&request.limit) {
        return Err(openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_task_page",
            format!(
                "limit must be in 1..={}, got {}",
                MAX_OPENENV_TASK_PAGE_SIZE, request.limit
            ),
            "Use a bounded Task API page; request another page with start.",
        ));
    }
    let credential_envs = resolve_credential_envs(
        state.openenv_runs.policy(),
        &request.credential_ids,
        &request.environment_urls,
    )?;
    if credential_envs.iter().any(Option::is_some) {
        state
            .metrics
            .openenv_authenticated_task_catalog_inspections
            .fetch_add(1, Ordering::Relaxed);
    }
    let environment_name = request.environment_name;
    let split = request.split;
    let start = request.start;
    let limit = request.limit;
    let catalogs = stream::iter(request.environment_urls)
        .zip(stream::iter(credential_envs))
        .map(|(url, credential_env)| {
            let environment_name = environment_name.clone();
            let split = split.clone();
            async move {
                let client = crate::openenv_cli::openenv_client(&url, credential_env.as_deref())?;
                let base_url = client.base_url().to_string();
                let catalog = client
                    .task_catalog(environment_name.as_deref(), split.as_deref(), start, limit)
                    .await
                    .with_context(|| format!("inspect OpenEnv Task API at {base_url}"))?;
                Ok::<_, anyhow::Error>(OpenEnvTaskCatalogEntry { base_url, catalog })
            }
        })
        .buffered(MAX_ENVIRONMENTS)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .map_err(openenv_task_catalog_error)?;
    Ok(OpenEnvTaskCatalogResponse {
        schema: OPENENV_TASK_CATALOG_SCHEMA_V1,
        catalogs,
    })
}

fn openenv_task_catalog_error(error: anyhow::Error) -> ApiError {
    let invalid_selection = error
        .downcast_ref::<OpenEnvClientError>()
        .is_some_and(|error| {
            matches!(
                error,
                OpenEnvClientError::InvalidTaskSelector { .. }
                    | OpenEnvClientError::TaskEnvironmentRequired { .. }
                    | OpenEnvClientError::UnknownTaskEnvironment { .. }
                    | OpenEnvClientError::UnknownTaskSplit { .. }
                    | OpenEnvClientError::InvalidTaskPageLimit { .. }
            )
        });
    if invalid_selection {
        openenv_error(
            StatusCode::BAD_REQUEST,
            "openenv_invalid_task_selection",
            error,
            "Choose one advertised environment and split name, then request a bounded page.",
        )
    } else {
        openenv_error(
            StatusCode::BAD_GATEWAY,
            "openenv_task_catalog_failed",
            error,
            "Check the OpenEnv Task API provider, advertised split, and network reachability. Environments without a provider should return HTTP 501 and are reported as unsupported.",
        )
    }
}

/// Restart v4 FIFO entries that provably never acquired workflow capacity.
///
/// Admitted work is never resumed: discovery and every later state may own a
/// non-resumable external episode, trainer, or evaluator and is failed during
/// registry restoration instead.
pub fn resume_queued_runs(state: &AppState) -> usize {
    if !state.openenv_runs.policy().enabled {
        return 0;
    }
    let queued = state.openenv_runs.queued_controls();
    let count = queued.len();
    for (run_id, control) in queued {
        state
            .metrics
            .openenv_runs_resumed
            .fetch_add(1, Ordering::Relaxed);
        let state = state.clone();
        tokio::spawn(async move {
            admit_and_execute_run(state, run_id, control).await;
        });
    }
    count
}

fn openenv_run_insert_error(error: anyhow::Error) -> ApiError {
    let conflict = error.downcast_ref::<OpenEnvIdempotencyConflict>().is_some();
    let capacity = error.to_string().contains("capacity is full");
    openenv_error(
        if conflict {
            StatusCode::CONFLICT
        } else if capacity {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        if conflict {
            "openenv_run_idempotency_conflict"
        } else if capacity {
            "openenv_run_capacity_full"
        } else {
            "openenv_run_create_failed"
        },
        error,
        if conflict {
            "Reuse an idempotency key only with the same normalized request, or choose a new non-secret key for distinct work."
        } else if capacity {
            "Cancel an unneeded queued run or wait for retained capacity, then retry."
        } else {
            "Check Kiln's adapter directory permissions and server logs."
        },
    )
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
    validate_run_idempotency_key(&request)?;
    if let Some(status) = state
        .openenv_runs
        .replay_idempotent(&request)
        .map_err(openenv_run_insert_error)?
    {
        state
            .metrics
            .openenv_run_idempotent_replays
            .fetch_add(1, Ordering::Relaxed);
        return Ok((StatusCode::OK, Json(status)));
    }
    validate_run_request(&request, state.openenv_runs.policy())?;
    let training_contract = match validate_openenv_training_preflight(&state, &request) {
        Ok(contract) => contract,
        Err(error) => {
            state
                .metrics
                .openenv_training_preflight_rejected
                .fetch_add(1, Ordering::Relaxed);
            state
                .metrics
                .openenv_training_preflights_rejected
                .fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }
    };
    if request.kind == OpenEnvRunKind::Train {
        state
            .metrics
            .openenv_training_preflights_accepted
            .fetch_add(1, Ordering::Relaxed);
    }
    let authenticated = request.credential_ids.iter().any(Option::is_some);
    let outcome = state
        .openenv_runs
        .insert(request, training_contract)
        .map_err(openenv_run_insert_error)?;
    let (status, control) = match outcome {
        OpenEnvRunInsertOutcome::Created { status, control } => (status, control),
        OpenEnvRunInsertOutcome::Replayed(status) => {
            state
                .metrics
                .openenv_run_idempotent_replays
                .fetch_add(1, Ordering::Relaxed);
            return Ok((StatusCode::OK, Json(status)));
        }
    };
    state
        .metrics
        .openenv_runs_started
        .fetch_add(1, Ordering::Relaxed);
    if authenticated {
        state
            .metrics
            .openenv_authenticated_runs_started
            .fetch_add(1, Ordering::Relaxed);
    }
    let run_id = status.run_id.clone();
    tokio::spawn(async move {
        admit_and_execute_run(state, run_id, control).await;
    });
    Ok((StatusCode::ACCEPTED, Json(status)))
}

async fn admit_and_execute_run(state: AppState, run_id: String, control: OpenEnvRunControl) {
    let permit = match state.openenv_runs.acquire(&run_id, &control).await {
        Ok(permit) => permit,
        Err(_) if control.cancel.load(Ordering::Relaxed) => return,
        Err(error) => {
            let error_message = format!("OpenEnv run admission failed: {error:#}");
            let occurred_unix_ms = now_unix_ms();
            let failure = OpenEnvRunFailure::explicit(
                OpenEnvRunFailureCode::RunAdmissionFailed,
                OpenEnvRunFailureStage::Admission,
                true,
                &error_message,
                "Inspect server capacity and submit a new run after admission recovers.",
                occurred_unix_ms,
            );
            tracing::error!(
                run_id = %run_id,
                error = %error_message,
                "OpenEnv run could not acquire execution capacity"
            );
            let _ = state.openenv_runs.update(&run_id, |status| {
                status.state = OpenEnvRunState::Failed;
                status.finished_unix_ms = Some(occurred_unix_ms);
                status.failure = Some(failure.clone());
                status.error = Some(error_message);
            });
            state
                .metrics
                .record_openenv_run_failure(failure.stage.as_label(), failure.retryable);
            state
                .metrics
                .openenv_runs_failed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    state
        .metrics
        .openenv_runs_admitted
        .fetch_add(1, Ordering::Relaxed);
    if let Some(queue_wait_ms) = state
        .openenv_runs
        .get(&run_id)
        .and_then(|status| status.admission)
        .and_then(|admission| admission.queue_wait_ms)
    {
        state
            .metrics
            .openenv_run_queue_wait_ms_total
            .fetch_add(queue_wait_ms, Ordering::Relaxed);
    }
    execute_run(state, run_id, control.cancel).await;
    drop(permit);
}

async fn execute_run(state: AppState, run_id: String, cancel: Arc<AtomicBool>) {
    if let Err(error) = execute_run_inner(&state, &run_id, cancel.clone()).await {
        let cancelled = cancel.load(Ordering::Relaxed);
        let occurred_unix_ms = now_unix_ms();
        let failure = (!cancelled).then(|| {
            let current = state.openenv_runs.get(&run_id);
            let run_state = current
                .as_ref()
                .map(|status| status.state)
                .unwrap_or(OpenEnvRunState::Failed);
            let collection_complete = current.as_ref().is_some_and(|status| {
                status.progress.groups_total > 0
                    && status.progress.groups_completed == status.progress.groups_total
            });
            OpenEnvRunFailure::from_error(run_state, collection_complete, &error, occurred_unix_ms)
        });
        let error_message = failure
            .as_ref()
            .map(|failure| failure.message.clone())
            .unwrap_or_else(|| format!("{error:#}"));
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
            status.finished_unix_ms = Some(occurred_unix_ms);
            status.failure = failure.clone();
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
            if let Some(failure) = &failure {
                state
                    .metrics
                    .record_openenv_run_failure(failure.stage.as_label(), failure.retryable);
            }
            state
                .metrics
                .openenv_runs_failed
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn execute_run_inner(state: &AppState, run_id: &str, cancel: Arc<AtomicBool>) -> Result<()> {
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "OpenEnv run cancelled");
    let admitted = state
        .openenv_runs
        .get(run_id)
        .context("OpenEnv run disappeared before execution")?;
    let request = admitted.request;
    let training_contract = admitted.training_contract;
    match request.kind {
        OpenEnvRunKind::Rollout => anyhow::ensure!(
            training_contract.is_none(),
            "OpenEnv rollout unexpectedly carries a training contract"
        ),
        OpenEnvRunKind::Train => anyhow::ensure!(
            training_contract.as_ref().is_some_and(|contract| {
                openenv_training_contract_matches_request(&request, contract)
            }),
            "OpenEnv train run has no valid admitted training contract"
        ),
    }
    if let Some(contract) = training_contract.as_ref() {
        if let Some(expected) = contract.behavior_policy.as_ref() {
            let current = state
                .openenv_behavior_policy_identity(contract.effective_config.base_adapter.as_deref())
                .map_err(anyhow::Error::msg)?;
            anyhow::ensure!(
                expected == &current,
                "OpenEnv behavior policy changed after training preflight and before collection began"
            );
        }
    }
    let run_dir = state.openenv_runs.run_dir(run_id);
    let credential_envs = state
        .openenv_runs
        .policy()
        .resolve_credential_envs(&request.credential_ids, &request.environment_urls)
        .map_err(anyhow::Error::msg)?;
    let options = rollout_options_for(&request, &run_dir, credential_envs);
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
    if let Some(expected) = training_contract
        .as_ref()
        .and_then(|contract| contract.behavior_policy.as_ref())
    {
        anyhow::ensure!(
            collection.summary.behavior_policy.as_ref() == Some(expected),
            "OpenEnv behavior policy changed after training preflight and before collection completed"
        );
    }
    collection.summary.training_contract = training_contract.clone();
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "OpenEnv run cancelled");
    write_openenv_outputs(
        &options,
        &collection.groups,
        &collection.replay,
        &collection.summary,
    )?;
    state
        .metrics
        .record_openenv_rollout_stats(&collection.summary.stats);

    let artifacts = artifacts_for(run_id, &collection.summary)?;
    if request.kind == OpenEnvRunKind::Rollout {
        state.openenv_runs.update(run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.finished_unix_ms = Some(now_unix_ms());
            status.progress.groups_completed = request.groups;
            status.progress.rollouts_completed = request.groups.saturating_mul(request.group_size);
            status.rollout_stats = Some(collection.summary.stats.clone());
            status.environments = collection
                .summary
                .environments
                .iter()
                .map(|inspection| inspection.identity.clone())
                .collect();
            status.artifacts = artifacts;
            status.failure = None;
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
        status.rollout_stats = Some(collection.summary.stats.clone());
        status.environments = collection
            .summary
            .environments
            .iter()
            .map(|inspection| inspection.identity.clone())
            .collect();
        status.artifacts = artifacts;
        status.failure = None;
        status.error = None;
    })?;
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "OpenEnv run cancelled");
    let training_contract =
        training_contract.context("OpenEnv train run has no admitted training contract")?;
    anyhow::ensure!(
        training_contract.schema == OPENENV_TRAINING_CONTRACT_SCHEMA_V1,
        "OpenEnv train run has unsupported training contract schema {:?}",
        training_contract.schema
    );
    let environment_gate_pending = request
        .environment_eval
        .as_ref()
        .is_some_and(|config| config.gate.is_some());
    let submission = super::training::submit_grpo_request(
        state,
        GrpoRequest {
            groups: collection.groups,
            dataset_path: None,
            dataset: None,
            dataset_split: None,
            config: training_contract.effective_config.clone(),
            post_eval: training_contract.post_eval.clone(),
        },
        environment_gate_pending,
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
    let artifacts = artifacts_for(run_id, &collection.summary)?;
    let training = training_status_for(state, &submission.job_id)
        .context("admitted OpenEnv training job disappeared")?;
    state.openenv_runs.update(run_id, |status| {
        status.state = OpenEnvRunState::TrainingQueued;
        status.training_job_id = Some(submission.job_id.clone());
        status.training_submission = Some(submission.clone());
        status.training = Some(training);
        status.artifacts = artifacts;
        status.failure = None;
        status.error = None;
    })?;
    state
        .metrics
        .openenv_training_queued
        .fetch_add(1, Ordering::Relaxed);
    follow_openenv_training(
        state,
        run_id,
        &request,
        training_contract.post_eval.as_ref(),
        &submission.job_id,
        cancel.clone(),
    )
    .await?;
    state
        .metrics
        .openenv_training_completed
        .fetch_add(1, Ordering::Relaxed);
    if request.environment_eval.is_some() {
        run_openenv_environment_evaluation(state, run_id, &request, &submission.job_id, cancel)
            .await?;
    }
    tracing::info!(
        run_id,
        training_job_id = %submission.job_id,
        groups = request.groups,
        rollouts = request.groups.saturating_mul(request.group_size),
        "OpenEnv training run completed"
    );
    Ok(())
}

fn finish_followed_training(
    state: &AppState,
    run_id: &str,
    request: &OpenEnvRunRequest,
    training: OpenEnvTrainingStatus,
    evals: Vec<OpenEnvPostEvalStatus>,
) -> Result<()> {
    state.openenv_runs.update(run_id, |status| {
        status.state = if request.environment_eval.is_some() {
            OpenEnvRunState::EnvironmentEvaluating
        } else {
            OpenEnvRunState::Completed
        };
        status.finished_unix_ms = request.environment_eval.is_none().then_some(now_unix_ms());
        status.training = Some(training);
        status.post_evaluations = evals;
        status.failure = None;
        status.error = None;
    })?;
    Ok(())
}

async fn run_openenv_environment_evaluation(
    state: &AppState,
    run_id: &str,
    request: &OpenEnvRunRequest,
    training_job_id: &str,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let config = request
        .environment_eval
        .clone()
        .context("OpenEnv environment evaluation config disappeared")?;
    let output_adapter = request
        .output_adapter
        .as_deref()
        .context("OpenEnv environment evaluation has no candidate adapter")?;
    let baseline = policy_identity(state, &request.adapter)?;
    let candidate = policy_identity(state, output_adapter)?;
    let run_dir = state.openenv_runs.run_dir(run_id);
    let credential_envs = state
        .openenv_runs
        .policy()
        .resolve_credential_envs(&request.credential_ids, &request.environment_urls)
        .map_err(anyhow::Error::msg)?;
    let baseline_options = environment_eval_rollout_options(
        request,
        &run_dir,
        &request.adapter,
        "baseline",
        credential_envs.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let candidate_options = environment_eval_rollout_options(
        request,
        &run_dir,
        output_adapter,
        "candidate",
        credential_envs,
    )
    .map_err(anyhow::Error::msg)?;
    let rollouts_total = config
        .groups
        .checked_mul(config.group_size)
        .context("OpenEnv environment evaluation rollout count overflow")?;
    state.openenv_runs.update(run_id, |status| {
        status.state = OpenEnvRunState::EnvironmentEvaluating;
        status.finished_unix_ms = None;
        status.environment_evaluation = Some(OpenEnvEnvironmentEvalStatus {
            state: OpenEnvEnvironmentEvalState::CollectingBaseline,
            seed_start: baseline_options.seed_start,
            groups: config.groups,
            group_size: config.group_size,
            baseline: baseline.clone(),
            candidate: candidate.clone(),
            progress: OpenEnvEnvironmentEvalProgress {
                state: OpenEnvEnvironmentEvalState::CollectingBaseline,
                groups_completed: 0,
                groups_total: config.groups,
                rollouts_completed: 0,
                rollouts_total,
            },
            baseline_stats: None,
            candidate_stats: None,
            evidence: None,
            outcome: None,
            verdict: None,
        });
        status.failure = None;
        status.error = None;
    })?;
    state
        .metrics
        .openenv_environment_evaluations_started
        .fetch_add(1, Ordering::Relaxed);

    let registry = state.openenv_runs.clone();
    let progress_run_id = run_id.to_string();
    let progress = Arc::new(move |progress: OpenEnvEnvironmentEvalProgress| {
        if let Err(error) = registry.update(&progress_run_id, |status| {
            status.state = OpenEnvRunState::EnvironmentEvaluating;
            if let Some(environment_evaluation) = status.environment_evaluation.as_mut() {
                environment_evaluation.state = progress.state;
                environment_evaluation.progress = progress;
            }
        }) {
            tracing::warn!(
                run_id = %progress_run_id,
                error = %error,
                "failed to persist OpenEnv environment evaluation progress"
            );
        }
    });
    let policy = OpenEnvPolicyTransport::InProcess(state.clone());
    let collection = match collect_environment_evaluation(
        &policy,
        baseline_options.clone(),
        candidate_options.clone(),
        cancel.clone(),
        progress,
        config.gate.as_ref(),
    )
    .await
    {
        Ok(collection) => collection,
        Err(error) => {
            let cancelled = cancel.load(Ordering::Relaxed);
            if !cancelled {
                state
                    .metrics
                    .openenv_environment_evaluations_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            let message = format!("{error:#}");
            let _ = state.openenv_runs.update(run_id, |status| {
                if let Some(environment_evaluation) = status.environment_evaluation.as_mut() {
                    environment_evaluation.state = if cancelled {
                        OpenEnvEnvironmentEvalState::Cancelled
                    } else {
                        OpenEnvEnvironmentEvalState::Failed
                    };
                    environment_evaluation.progress.state = environment_evaluation.state;
                    environment_evaluation.outcome =
                        (!cancelled).then_some(OpenEnvEnvironmentEvalOutcome::Error);
                    environment_evaluation.verdict = Some(message.clone());
                }
            });
            return Err(error);
        }
    };
    state
        .metrics
        .record_openenv_rollout_stats(&collection.baseline_stats);
    state
        .metrics
        .record_openenv_rollout_stats(&collection.candidate_stats);
    if cancel.load(Ordering::Relaxed) {
        let _ = state.openenv_runs.update(run_id, |status| {
            if let Some(environment_evaluation) = status.environment_evaluation.as_mut() {
                environment_evaluation.state = OpenEnvEnvironmentEvalState::Cancelled;
                environment_evaluation.progress.state = OpenEnvEnvironmentEvalState::Cancelled;
                environment_evaluation.verdict =
                    Some("OpenEnv run cancelled before environment evaluation promotion".into());
            }
        });
        anyhow::bail!("OpenEnv run cancelled before environment evaluation promotion");
    }

    let mut terminal_error = None;
    let (outcome, verdict) = match collection.evidence.decision {
        OpenEnvEnvironmentEvalDecision::Diagnostic => (
            OpenEnvEnvironmentEvalOutcome::Diagnostic,
            format!(
                "Diagnostic paired environment evaluation: {}",
                collection.evidence.reason
            ),
        ),
        OpenEnvEnvironmentEvalDecision::Rejected => (
            OpenEnvEnvironmentEvalOutcome::Rejected,
            format!(
                "Environment promotion gate rejected the candidate: {}",
                collection.evidence.reason
            ),
        ),
        OpenEnvEnvironmentEvalDecision::Inconclusive => (
            OpenEnvEnvironmentEvalOutcome::Inconclusive,
            format!(
                "Environment promotion gate was inconclusive; candidate remains unserved: {}",
                collection.evidence.reason
            ),
        ),
        OpenEnvEnvironmentEvalDecision::Passed if request.auto_load => {
            let promotion = training_status_for(state, training_job_id)
                .and_then(|training| training.adapter_path)
                .map(PathBuf::from)
                .context("OpenEnv environment gate passed but trainer published no adapter path")
                .and_then(|adapter_path| {
                    crate::adapter_swap::promote_trained_adapter(
                        state,
                        &adapter_path,
                        output_adapter,
                        true,
                    )
                    .map_err(anyhow::Error::msg)
                });
            match promotion {
                Ok(()) => (
                    OpenEnvEnvironmentEvalOutcome::Promoted,
                    format!(
                        "Environment promotion gate passed and adapter {output_adapter:?} was promoted: {}",
                        collection.evidence.reason
                    ),
                ),
                Err(error) => {
                    let message = format!(
                        "Environment promotion gate passed but adapter {output_adapter:?} could not be promoted: {error:#}"
                    );
                    terminal_error = Some(message.clone());
                    (OpenEnvEnvironmentEvalOutcome::Error, message)
                }
            }
        }
        OpenEnvEnvironmentEvalDecision::Passed => (
            OpenEnvEnvironmentEvalOutcome::Kept,
            format!(
                "Environment promotion gate passed; auto_load=false kept adapter {output_adapter:?} on disk: {}",
                collection.evidence.reason
            ),
        ),
    };

    let receipt_path = run_dir.join("environment-evaluation").join("receipt.json");
    let receipt = OpenEnvEnvironmentEvalReceipt {
        schema: crate::openenv_evaluation::OPENENV_ENVIRONMENT_EVAL_RECEIPT_SCHEMA_V1.to_string(),
        run_id: run_id.to_string(),
        config,
        seed_start: baseline_options.seed_start,
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        baseline_summary_sha256: summary_sha256(&baseline_options.summary_output)?,
        candidate_summary_sha256: summary_sha256(&candidate_options.summary_output)?,
        evidence: collection.evidence.clone(),
        outcome,
        verdict: verdict.clone(),
    };
    write_environment_evaluation_receipt(&receipt_path, &receipt)?;
    let environment_artifacts = environment_evaluation_artifacts(run_id, &run_dir)?;
    state.openenv_runs.update(run_id, |status| {
        let evaluation_failed = outcome == OpenEnvEnvironmentEvalOutcome::Error;
        status.state = if evaluation_failed {
            OpenEnvRunState::EnvironmentEvaluating
        } else {
            OpenEnvRunState::Completed
        };
        status.finished_unix_ms = (!evaluation_failed).then_some(now_unix_ms());
        status.artifacts.extend(environment_artifacts);
        status.environment_evaluation = Some(OpenEnvEnvironmentEvalStatus {
            state: if evaluation_failed {
                OpenEnvEnvironmentEvalState::Failed
            } else {
                OpenEnvEnvironmentEvalState::Completed
            },
            seed_start: baseline_options.seed_start,
            groups: receipt.config.groups,
            group_size: receipt.config.group_size,
            baseline,
            candidate,
            progress: OpenEnvEnvironmentEvalProgress {
                state: if evaluation_failed {
                    OpenEnvEnvironmentEvalState::Failed
                } else {
                    OpenEnvEnvironmentEvalState::Completed
                },
                groups_completed: receipt.config.groups,
                groups_total: receipt.config.groups,
                rollouts_completed: rollouts_total,
                rollouts_total,
            },
            baseline_stats: Some(collection.baseline_stats),
            candidate_stats: Some(collection.candidate_stats),
            evidence: Some(collection.evidence),
            outcome: Some(outcome),
            verdict: Some(verdict),
        });
        status.failure = None;
        status.error = None;
    })?;
    if terminal_error.is_some() {
        state
            .metrics
            .openenv_environment_evaluations_failed
            .fetch_add(1, Ordering::Relaxed);
    } else {
        state
            .metrics
            .openenv_environment_evaluations_completed
            .fetch_add(1, Ordering::Relaxed);
    }
    match outcome {
        OpenEnvEnvironmentEvalOutcome::Promoted | OpenEnvEnvironmentEvalOutcome::Kept => state
            .metrics
            .openenv_environment_gates_passed
            .fetch_add(1, Ordering::Relaxed),
        OpenEnvEnvironmentEvalOutcome::Rejected => state
            .metrics
            .openenv_environment_gates_rejected
            .fetch_add(1, Ordering::Relaxed),
        OpenEnvEnvironmentEvalOutcome::Inconclusive => state
            .metrics
            .openenv_environment_gates_inconclusive
            .fetch_add(1, Ordering::Relaxed),
        OpenEnvEnvironmentEvalOutcome::Diagnostic | OpenEnvEnvironmentEvalOutcome::Error => 0,
    };
    if let Some(error) = terminal_error {
        anyhow::bail!(error);
    }
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
            training_data: job.training_data.clone(),
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
    post_eval: Option<&kiln_eval::PostEvalConfig>,
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
                ensure_openenv_training_evidence(state, run_id, request, &training)
                    .map_err(failure::training_evidence_failure)?;
                if post_eval.is_none() {
                    finish_followed_training(state, run_id, request, training, evals)?;
                    return Ok(());
                }

                let completed_at = training_completed_at.get_or_insert_with(Instant::now);
                let expected_evals =
                    1 + usize::from(post_eval.is_some_and(|cfg| cfg.include_baseline));
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
                    let gate_done = post_eval.is_none_or(|cfg| cfg.min_accuracy.is_none())
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
                        finish_followed_training(state, run_id, request, final_training, evals)?;
                        return Ok(());
                    }
                }
            }
        }
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL).await;
    }
}

fn artifacts_for(run_id: &str, summary: &OpenEnvRolloutSummary) -> Result<Vec<OpenEnvArtifact>> {
    let prefix = format!("/v1/openenv/runs/{run_id}/artifacts");
    let (dataset_sha256, dataset_bytes) =
        crate::openenv_replay::bounded_artifact_metadata(Path::new(&summary.output_path))
            .context("bind published OpenEnv dataset to its on-disk bytes")?;
    anyhow::ensure!(
        dataset_sha256 == summary.dataset_sha256 && dataset_bytes == summary.dataset_bytes,
        "published OpenEnv dataset differs from its collection receipt"
    );
    let (replay_sha256, replay_bytes) =
        crate::openenv_replay::bounded_artifact_metadata(Path::new(&summary.replay_output_path))
            .context("bind published OpenEnv replay to its on-disk bytes")?;
    anyhow::ensure!(
        replay_sha256 == summary.replay_sha256 && replay_bytes == summary.replay_bytes,
        "published OpenEnv replay differs from its collection receipt"
    );
    let (summary_sha256, summary_bytes) =
        crate::openenv_replay::bounded_artifact_metadata(Path::new(&summary.summary_output_path))
            .context("bind published OpenEnv summary to its on-disk bytes")?;
    Ok(vec![
        OpenEnvArtifact {
            kind: "dataset".into(),
            url: format!("{prefix}/dataset"),
            sha256: dataset_sha256,
            bytes: dataset_bytes,
        },
        OpenEnvArtifact {
            kind: "replay".into(),
            url: format!("{prefix}/replay"),
            sha256: replay_sha256,
            bytes: replay_bytes,
        },
        OpenEnvArtifact {
            kind: "summary".into(),
            url: format!("{prefix}/summary"),
            sha256: summary_sha256,
            bytes: summary_bytes,
        },
    ])
}

fn environment_evaluation_artifacts(run_id: &str, run_dir: &Path) -> Result<Vec<OpenEnvArtifact>> {
    let root = run_dir.join("environment-evaluation");
    [
        (
            "environment_eval_baseline_dataset",
            root.join("baseline").join("rollouts.jsonl"),
        ),
        (
            "environment_eval_baseline_replay",
            root.join("baseline").join("replay.json"),
        ),
        (
            "environment_eval_baseline_summary",
            root.join("baseline").join("summary.json"),
        ),
        (
            "environment_eval_candidate_dataset",
            root.join("candidate").join("rollouts.jsonl"),
        ),
        (
            "environment_eval_candidate_replay",
            root.join("candidate").join("replay.json"),
        ),
        (
            "environment_eval_candidate_summary",
            root.join("candidate").join("summary.json"),
        ),
        ("environment_eval_receipt", root.join("receipt.json")),
    ]
    .into_iter()
    .map(|(kind, path)| {
        let (sha256, bytes) = crate::openenv_replay::bounded_artifact_metadata(&path)
            .with_context(|| {
                format!(
                    "bind OpenEnv environment evaluation artifact {}",
                    path.display()
                )
            })?;
        Ok(OpenEnvArtifact {
            kind: kind.to_string(),
            url: format!("/v1/openenv/runs/{run_id}/artifacts/{kind}"),
            sha256,
            bytes,
        })
    })
    .collect()
}

async fn list_runs(State(state): State<AppState>) -> Json<OpenEnvRunList> {
    Json(OpenEnvRunList {
        schema: OPENENV_RUN_LIST_SCHEMA_V5,
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
    let (status, settled_queued) = state.openenv_runs.cancel(&run_id).map_err(|error| {
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
    })?;
    if settled_queued {
        state
            .metrics
            .openenv_runs_cancelled
            .fetch_add(1, Ordering::Relaxed);
    }
    Ok(Json(status))
}

async fn download_artifact(
    State(state): State<AppState>,
    AxumPath((run_id, kind)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    let (path, content_type, artifact) = state
        .openenv_runs
        .artifact_path(&run_id, &kind)
        .ok_or_else(|| {
            openenv_error(
                StatusCode::NOT_FOUND,
                "openenv_artifact_not_found",
                format!("OpenEnv artifact {kind:?} for run {run_id} was not found"),
                "Use only a kind currently declared in the run's artifacts array; files are unavailable before manifest publication.",
            )
        })?;
    let verify_path = path.clone();
    let expected_sha256 = artifact.sha256.clone();
    let expected_bytes = artifact.bytes;
    let verified = tokio::task::spawn_blocking(move || {
        crate::openenv_replay::open_verified_artifact(
            &verify_path,
            &expected_sha256,
            expected_bytes,
        )
    })
    .await
    .map_err(ApiError::internal)?
    .map_err(|error| {
        tracing::warn!(
            run_id,
            artifact_kind = kind,
            error = %error,
            "refusing to stream an OpenEnv artifact that drifted from its manifest"
        );
        openenv_error(
            StatusCode::CONFLICT,
            "openenv_artifact_integrity_failed",
            format!(
                "OpenEnv artifact {kind:?} no longer matches the manifest published by run {run_id}"
            ),
            "Restore the original content-addressed artifact bundle or recollect; never edit retained artifacts in place.",
        )
    })?;
    let file = tokio::fs::File::from_std(verified);
    let (tx, rx) = mpsc::channel::<std::io::Result<Vec<u8>>>(8);
    tokio::spawn(async move {
        let mut file = file;
        let mut remaining = expected_bytes;
        while remaining > 0 {
            let mut chunk = vec![0u8; ARTIFACT_CHUNK_BYTES.min(remaining)];
            match file.read(&mut chunk).await {
                Ok(0) => {
                    let _ = tx
                        .send(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "verified OpenEnv artifact was truncated while streaming",
                        )))
                        .await;
                    break;
                }
                Ok(read) => {
                    chunk.truncate(read);
                    remaining -= read;
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
        if content_type == "application/x-ndjson" {
            "jsonl"
        } else {
            "json"
        }
    ))
    .map_err(ApiError::internal)?;
    let content_length =
        HeaderValue::from_str(&artifact.bytes.to_string()).map_err(ApiError::internal)?;
    let etag =
        HeaderValue::from_str(&format!("\"{}\"", artifact.sha256)).map_err(ApiError::internal)?;
    Ok((
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_str(content_type).map_err(ApiError::internal)?,
            ),
            (header::CONTENT_DISPOSITION, disposition),
            (header::CONTENT_LENGTH, content_length),
            (header::ETAG, etag),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, no-store"),
            ),
            (
                header::X_CONTENT_TYPE_OPTIONS,
                HeaderValue::from_static("nosniff"),
            ),
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
            "/v1/openenv/tasks",
            post(inspect_tasks).layer(DefaultBodyLimit::max(OPENENV_API_BODY_LIMIT)),
        )
        .route(
            "/v1/openenv/training/preflight",
            post(preflight_training).layer(DefaultBodyLimit::max(OPENENV_API_BODY_LIMIT)),
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
    use axum::extract::Path as AxumPath;
    use axum::http::Request;
    use axum::response::Response as AxumResponse;
    use axum::routing::{get as axum_get, post as axum_post};
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use serde_json::json;
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
            idempotency_key: None,
            environment_urls: vec!["http://127.0.0.1:8000".into()],
            credential_ids: Vec::new(),
            adapter: "base".into(),
            groups: 2,
            group_size: 3,
            seed_start: 0,
            reset_options: default_reset_options(),
            environment_reset_options: Vec::new(),
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
            environment_eval: None,
        }
    }

    fn insert_created(
        registry: &OpenEnvRunRegistry,
        request: OpenEnvRunRequest,
    ) -> (OpenEnvRunStatus, OpenEnvRunControl) {
        match insert_test_run(registry, request).unwrap() {
            OpenEnvRunInsertOutcome::Created { status, control } => (status, control),
            OpenEnvRunInsertOutcome::Replayed(_) => {
                panic!("test expected a newly created OpenEnv run")
            }
        }
    }

    fn insert_test_run(
        registry: &OpenEnvRunRegistry,
        request: OpenEnvRunRequest,
    ) -> Result<OpenEnvRunInsertOutcome> {
        let training_contract = materialized_openenv_training_contract(&request)?;
        registry.insert(request, training_contract)
    }

    fn test_training_data() -> TrainingDataProvenance {
        let groups = (0..2)
            .map(|group_index| {
                let provenance = kiln_train::OpenEnvRolloutProvenanceV1::new(
                    "CounterEnvironment",
                    "http://127.0.0.1:8000",
                    Some("1.0".into()),
                    format!("sha256:{}", "a".repeat(64)),
                    format!("sha256:{}", "b".repeat(64)),
                    format!("sha256:{}", "c".repeat(64)),
                    17 + group_index,
                    1,
                    1.0,
                    true,
                    kiln_train::OpenEnvEpisodeTerminationV1::Done,
                    None,
                )
                .unwrap();
                kiln_train::AgenticGroup {
                    messages: Vec::new(),
                    completions: (0..3)
                        .map(|_| {
                            kiln_train::ScoredRollout::legacy("{\"amount\":1}".into(), 1.0)
                                .with_openenv(provenance.clone())
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let admitted_corpus_sha256 =
            kiln_eval::sha256_json(&serde_json::to_value(&groups).unwrap());
        TrainingDataProvenance {
            source: "inline".into(),
            dataset: None,
            split: None,
            dataset_corpus_sha256: None,
            split_manifest_sha256: None,
            admitted_corpus_sha256,
            rows: groups.len() as u64,
            openenv: kiln_train::openenv_training_data_provenance(&groups).unwrap(),
        }
    }

    fn test_adapter_path(temp: &tempfile::TempDir, job_id: &str) -> PathBuf {
        temp.path().join(format!("adapter-{job_id}"))
    }

    fn write_test_training_evidence(
        temp: &tempfile::TempDir,
        job_id: &str,
        training_data: &TrainingDataProvenance,
    ) -> PathBuf {
        let adapter_path = test_adapter_path(temp, job_id);
        std::fs::create_dir_all(&adapter_path).unwrap();
        std::fs::write(
            adapter_path.join("adapter_config.json"),
            br#"{"r":8,"lora_alpha":16.0}"#,
        )
        .unwrap();
        std::fs::write(
            adapter_path.join("adapter_model.safetensors"),
            b"test adapter weights",
        )
        .unwrap();
        let mut receipt = kiln_train::TrainReceipt::new(
            "agent",
            "grpo",
            &ModelConfig::qwen3_5_4b(),
            &crate::api::test_tokenizer(),
            kiln_train::train_receipt::HyperparameterReceipt {
                mode: "grpo".into(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(17),
                shuffle: false,
            },
            json!({"output_name": "agent"}),
        );
        receipt.training_data = kiln_train::train_receipt::TrainingDataReceipt {
            source: "inline_grpo_groups".into(),
            path: None,
            sha256: Some(training_data.admitted_corpus_sha256.clone()),
            openenv: training_data.openenv.clone(),
        };
        receipt.write_to_adapter_dir(&adapter_path).unwrap();
        adapter_path
    }

    fn training_job(
        temp: &tempfile::TempDir,
        job_id: &str,
        state: TrainingState,
    ) -> TrainingJobInfo {
        let training_data = test_training_data();
        let adapter_path = write_test_training_evidence(temp, job_id, &training_data);
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
            adapter_path: (state == TrainingState::Completed)
                .then(|| adapter_path.display().to_string()),
            submitted_at: Instant::now(),
            submitted_unix_ms: now_unix_ms(),
            auto_load: false,
            consumed_correction_ids: Vec::new(),
            training_data: Some(training_data),
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

    async fn task_fixture_post(
        AxumPath((_environment, operation)): AxumPath<(String, String)>,
        Json(body): Json<Value>,
    ) -> AxumResponse {
        match operation.as_str() {
            "num_tasks" => {
                assert_eq!(body, json!({"split": "train"}));
                Json(json!({"num_tasks": 3})).into_response()
            }
            "task_range" => {
                assert_eq!(body, json!({"split": "train", "start": 1, "stop": 2}));
                Json(json!({
                    "tasks": [{"id": 1, "prompt": "2 + 2", "answer": "4"}]
                }))
                .into_response()
            }
            _ => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn task_fixture() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/list_environments",
                axum_get(|| async { Json(json!(["task_env"])) }),
            )
            .route(
                "/{environment}/splits",
                axum_get(|| async {
                    Json(json!([
                        {"name": "train", "type": "train"},
                        {"name": "holdout", "type": "validation"}
                    ]))
                }),
            )
            .route("/{environment}/{operation}", axum_post(task_fixture_post));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn run_registry_admits_fifo_and_remains_bounded_persisted_and_restored() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            max_tracked_runs: 3,
            ..Default::default()
        };
        let registry =
            Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
        let (first, first_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let (second, second_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let (third, third_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        assert!(insert_test_run(&registry, request(OpenEnvRunKind::Rollout)).is_err());
        assert_eq!(
            registry
                .get(&first.run_id)
                .unwrap()
                .admission
                .unwrap()
                .queue_position,
            Some(1)
        );
        assert_eq!(
            registry
                .get(&third.run_id)
                .unwrap()
                .admission
                .unwrap()
                .queue_position,
            Some(3)
        );

        let first_permit = registry
            .acquire(&first.run_id, &first_control)
            .await
            .unwrap();
        assert_eq!(registry.counts(), (1, 2, 3));
        let third_registry = registry.clone();
        let third_run_id = third.run_id.clone();
        let third_wait =
            tokio::spawn(
                async move { third_registry.acquire(&third_run_id, &third_control).await },
            );
        let second_registry = registry.clone();
        let second_run_id = second.run_id.clone();
        let second_wait = tokio::spawn(async move {
            second_registry
                .acquire(&second_run_id, &second_control)
                .await
        });
        tokio::task::yield_now().await;
        assert!(!second_wait.is_finished());
        assert!(!third_wait.is_finished());

        registry
            .update(&first.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.submitted_unix_ms = 1;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        drop(first_permit);
        let second_permit = tokio::time::timeout(Duration::from_secs(1), second_wait)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            !third_wait.is_finished(),
            "the third run must not bypass FIFO order"
        );
        registry
            .update(&second.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.submitted_unix_ms = 2;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        drop(second_permit);
        let third_permit = tokio::time::timeout(Duration::from_secs(1), third_wait)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        registry
            .update(&third.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.submitted_unix_ms = 3;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        drop(third_permit);

        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        assert_eq!(
            restored.get(&first.run_id).unwrap().state,
            OpenEnvRunState::RolloutReady
        );
        insert_test_run(&restored, request(OpenEnvRunKind::Rollout))
            .expect("the oldest terminal status should be evicted to admit new work");
        assert!(restored.get(&first.run_id).is_none());
    }

    #[test]
    fn run_registry_idempotency_is_atomic_conflict_safe_and_restart_durable() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig::default();
        let registry =
            Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
        let mut submitted = request(OpenEnvRunKind::Rollout);
        submitted.idempotency_key = Some("experiment:counter:17".into());
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let registry = registry.clone();
            let barrier = barrier.clone();
            let submitted = submitted.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                match insert_test_run(&registry, submitted).unwrap() {
                    OpenEnvRunInsertOutcome::Created { status, .. } => (true, status.run_id),
                    OpenEnvRunInsertOutcome::Replayed(status) => (false, status.run_id),
                }
            }));
        }
        barrier.wait();
        let first = handles.remove(0).join().unwrap();
        let second = handles.remove(0).join().unwrap();
        assert_ne!(first.0, second.0, "exactly one caller must create the run");
        assert_eq!(first.1, second.1);
        assert_eq!(registry.counts(), (0, 1, 1));
        assert_eq!(
            registry
                .replay_idempotent(&submitted)
                .unwrap()
                .unwrap()
                .run_id,
            first.1
        );

        let mut conflict = submitted.clone();
        conflict.groups += 1;
        let error = registry.replay_idempotent(&conflict).unwrap_err();
        assert!(error.downcast_ref::<OpenEnvIdempotencyConflict>().is_some());
        drop(registry);

        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        assert_eq!(
            restored
                .replay_idempotent(&submitted)
                .unwrap()
                .unwrap()
                .run_id,
            first.1
        );
        let mut duplicate = restored.get(&first.1).unwrap();
        duplicate.run_id = uuid::Uuid::new_v4().to_string();
        duplicate.admission.as_mut().unwrap().sequence += 1;
        std::fs::create_dir(restored.run_dir(&duplicate.run_id)).unwrap();
        persist_status_to(&restored.status_path(&duplicate.run_id), &duplicate).unwrap();
        drop(restored);
        let error = OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default())
            .unwrap_err();
        assert!(error.to_string().contains("share idempotency key"));
    }

    #[test]
    fn admitted_training_contract_is_persisted_and_migrates_pristine_v4_runs() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig::default();
        let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
        let mut submitted = request(OpenEnvRunKind::Train);
        submitted.auto_load = false;
        submitted.training_config = Some(GrpoConfig {
            learning_rate: Some(3e-5),
            lora_rank: 16,
            output_name: Some("ignored-request-output".into()),
            behavior_policy: BehaviorPolicy::Recorded,
            ..GrpoConfig::default()
        });
        let expected = materialized_openenv_training_contract(&submitted)
            .unwrap()
            .unwrap();
        let mut mismatched = expected.clone();
        mismatched.effective_config.lora_rank += 1;
        assert!(
            registry
                .insert(submitted.clone(), Some(mismatched))
                .is_err(),
            "persistence must reject a contract that disagrees with its owned request fields"
        );
        let mut malformed_policy = expected.clone();
        malformed_policy.behavior_policy = Some(kiln_train::RolloutBehaviorPolicyIdentityV1 {
            served_model_id: "test-model".into(),
            base_model_sha256: "not-a-digest".into(),
            adapter: None,
            inference_config_sha256: format!("sha256:{}", "a".repeat(64)),
            implementation: "kiln/test".into(),
        });
        assert!(
            registry
                .insert(submitted.clone(), Some(malformed_policy))
                .is_err(),
            "persistence must reject a malformed behavior-policy identity"
        );
        let (created, _) = insert_created(&registry, submitted);
        assert_eq!(created.schema, OPENENV_RUN_SCHEMA_V5);
        assert_eq!(
            serde_json::to_value(created.training_contract.as_ref().unwrap()).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        let persisted: OpenEnvRunStatus =
            serde_json::from_slice(&std::fs::read(registry.status_path(&created.run_id)).unwrap())
                .unwrap();
        assert_eq!(
            serde_json::to_value(persisted.training_contract.as_ref().unwrap()).unwrap(),
            serde_json::to_value(&expected).unwrap(),
            "run.json must retain the exact admitted config before collection"
        );

        registry
            .update(&created.run_id, |status| {
                status.schema = OPENENV_RUN_SCHEMA_V4.into();
                status.training_contract = None;
            })
            .unwrap();
        drop(registry);

        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        let migrated = restored.get(&created.run_id).unwrap();
        assert_eq!(migrated.schema, OPENENV_RUN_SCHEMA_V5);
        assert!(migrated.safely_restartable_queued());
        assert_eq!(
            serde_json::to_value(migrated.training_contract.unwrap()).unwrap(),
            serde_json::to_value(expected).unwrap(),
            "a pristine v4 queue entry must be sealed exactly once before resume"
        );
    }

    #[test]
    fn capacity_eviction_releases_idempotency_key_across_restart() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_tracked_runs: 1,
            ..Default::default()
        };
        let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
        let mut first_request = request(OpenEnvRunKind::Rollout);
        first_request.idempotency_key = Some("reusable:attempt".into());
        let (first, _) = insert_created(&registry, first_request.clone());
        registry.cancel(&first.run_id).unwrap();

        let mut displacement = request(OpenEnvRunKind::Rollout);
        displacement.idempotency_key = Some("displacement".into());
        let (second, _) = insert_created(&registry, displacement);
        registry.cancel(&second.run_id).unwrap();

        let (replacement, _) = insert_created(&registry, first_request.clone());
        assert_ne!(replacement.run_id, first.run_id);
        drop(registry);

        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        assert_eq!(
            restored
                .replay_idempotent(&first_request)
                .unwrap()
                .unwrap()
                .run_id,
            replacement.run_id
        );
    }

    #[tokio::test]
    async fn queued_run_cancels_immediately_without_consuming_execution_capacity() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            max_tracked_runs: 3,
            ..Default::default()
        };
        let registry =
            Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap());
        let (first, first_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let (second, second_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let first_permit = registry
            .acquire(&first.run_id, &first_control)
            .await
            .unwrap();

        let (cancelled, settled_queued) = registry.cancel(&second.run_id).unwrap();
        assert!(settled_queued);
        assert_eq!(cancelled.state, OpenEnvRunState::Cancelled);
        assert!(cancelled.finished_unix_ms.is_some());
        assert_eq!(cancelled.admission.unwrap().queue_position, None);
        assert_eq!(registry.counts(), (1, 0, 2));
        assert!(
            tokio::time::timeout(
                Duration::from_secs(1),
                registry.acquire(&second.run_id, &second_control)
            )
            .await
            .unwrap()
            .is_err()
        );
        drop(first_permit);
    }

    #[tokio::test]
    async fn restart_resumes_only_fifo_entries_that_never_acquired_capacity() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            max_tracked_runs: 3,
            ..Default::default()
        };
        let registry =
            Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
        let (active, active_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let (queued_first, _) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        let (queued_second, _) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
        assert!(
            queued_first.admission.as_ref().unwrap().sequence
                < queued_second.admission.as_ref().unwrap().sequence
        );
        let permit = registry
            .acquire(&active.run_id, &active_control)
            .await
            .unwrap();
        drop(permit);
        drop(registry);

        let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
        let interrupted = restored.get(&active.run_id).unwrap();
        assert_eq!(interrupted.state, OpenEnvRunState::Failed);
        assert!(interrupted.error.as_deref().unwrap().contains("restarted"));
        let failure = interrupted.failure.unwrap();
        assert_eq!(failure.schema, OPENENV_RUN_FAILURE_SCHEMA_V1);
        assert_eq!(failure.code, OpenEnvRunFailureCode::RunInterrupted);
        assert_eq!(failure.stage, OpenEnvRunFailureStage::Restoration);
        assert!(failure.retryable);
        let safe = restored.get(&queued_first.run_id).unwrap();
        assert_eq!(safe.state, OpenEnvRunState::Queued);
        assert_eq!(safe.admission.unwrap().queue_position, Some(1));
        assert_eq!(
            restored
                .get(&queued_second.run_id)
                .unwrap()
                .admission
                .unwrap()
                .queue_position,
            Some(2)
        );
        let queued_controls = restored.queued_controls();
        assert_eq!(queued_controls.len(), 2);
        assert_eq!(queued_controls[0].0, queued_first.run_id);
        assert_eq!(queued_controls[1].0, queued_second.run_id);
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
    fn idempotency_keys_are_bounded_non_secret_opaque_tokens() {
        for key in ["experiment-17", "client.retry_2", "capability:math:003"] {
            assert!(validate_openenv_idempotency_key(key).is_ok());
        }
        for key in ["", "contains space", "secret/token", "line\nbreak"] {
            assert!(validate_openenv_idempotency_key(key).is_err(), "{key:?}");
        }
        assert!(validate_openenv_idempotency_key(&"a".repeat(128)).is_ok());
        assert!(validate_openenv_idempotency_key(&"a".repeat(129)).is_err());

        let mut invalid = request(OpenEnvRunKind::Rollout);
        invalid.idempotency_key = Some("bad key".into());
        assert!(validate_run_request(&invalid, &OpenEnvConfig::default()).is_err());
    }

    #[test]
    fn credential_handles_are_aligned_and_resolved_before_admission() {
        let policy = OpenEnvConfig::default();
        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.credential_ids = vec![Some("missing".into())];
        let error = validate_run_request(&rollout, &policy).unwrap_err();
        assert_eq!(error.code, "openenv_invalid_credential");
        assert!(!error.message.contains("bearer"));

        rollout.credential_ids = vec![None, None];
        let error = validate_run_request(&rollout, &policy).unwrap_err();
        assert_eq!(error.code, "openenv_invalid_credential");
        assert!(error.message.contains("exactly one"));

        rollout.credential_ids = vec![None];
        assert!(validate_run_request(&rollout, &policy).is_ok());
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
    fn rollout_rejects_every_training_only_field() {
        let policy = OpenEnvConfig::default();

        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.output_adapter = Some("agent".into());
        assert!(validate_run_request(&rollout, &policy).is_err());

        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.training_config = Some(GrpoConfig::default());
        assert!(validate_run_request(&rollout, &policy).is_err());

        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.post_eval = Some(kiln_eval::PostEvalConfig {
            suite: "held-out".into(),
            data_scope: Default::default(),
            generation: None,
            min_accuracy: None,
            include_baseline: true,
        });
        assert!(validate_run_request(&rollout, &policy).is_err());

        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
            groups: 1,
            group_size: 1,
            seed_start: None,
            gate: None,
        });
        assert!(validate_run_request(&rollout, &policy).is_err());
    }

    #[test]
    fn effective_grpo_config_is_owned_by_the_live_rollout_contract() {
        let mut train = request(OpenEnvRunKind::Train);
        train.adapter = "behavior-agent".into();
        train.auto_load = true;
        train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
            groups: 1,
            group_size: 1,
            seed_start: None,
            gate: Some(crate::openenv_evaluation::OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        });
        let mut supplied = GrpoConfig::default();
        supplied.output_name = Some("ignored".into());
        supplied.auto_load = true;
        supplied.base_adapter = Some("ignored".into());
        supplied.behavior_policy = BehaviorPolicy::Recorded;
        train.training_config = Some(supplied);

        let effective = effective_openenv_grpo_config(&train, "trained-agent");
        assert_eq!(effective.output_name.as_deref(), Some("trained-agent"));
        assert_eq!(effective.base_adapter.as_deref(), Some("behavior-agent"));
        assert_eq!(
            effective.behavior_policy,
            BehaviorPolicy::NoImportanceCorrection
        );
        assert!(!effective.auto_load, "environment gate owns promotion");
    }

    #[test]
    fn train_preflight_rejects_static_failures_before_mock_backend_admission() {
        let temp = tempfile::tempdir().unwrap();
        let mut state = test_state(&temp, OpenEnvConfig::default());

        let mut invalid_config = request(OpenEnvRunKind::Train);
        let mut config = GrpoConfig::default();
        config.checkpoint_interval = Some(0);
        invalid_config.training_config = Some(config);
        let error = validate_openenv_training_preflight(&state, &invalid_config).unwrap_err();
        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("checkpoint_interval"));

        state.suite_registry = Some(Arc::new(crate::eval::SuiteRegistry::new(
            temp.path().join("suites"),
        )));
        let mut missing_suite = request(OpenEnvRunKind::Train);
        missing_suite.post_eval = Some(kiln_eval::PostEvalConfig {
            suite: "not-installed".into(),
            data_scope: Default::default(),
            generation: None,
            min_accuracy: None,
            include_baseline: true,
        });
        let error = validate_openenv_training_preflight(&state, &missing_suite).unwrap_err();
        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("not an installed eval suite"));

        let mut missing_policy = request(OpenEnvRunKind::Train);
        missing_policy.adapter = "missing-policy".into();
        let error = validate_openenv_training_preflight(&state, &missing_policy).unwrap_err();
        assert_eq!(error.code, "adapter_not_found");

        let error = validate_openenv_training_preflight(&state, &request(OpenEnvRunKind::Train))
            .unwrap_err();
        assert_eq!(error.code, "mock_mode");
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
    fn heterogeneous_reset_plan_is_aligned_exclusive_and_preserved() {
        let policy = OpenEnvConfig::default();
        let mut rollout = request(OpenEnvRunKind::Rollout);
        rollout
            .environment_urls
            .push("http://127.0.0.1:8001".into());
        rollout.environment_reset_options =
            vec![json!({"difficulty": "hard"}), json!({"split": "train"})];
        assert!(validate_run_request(&rollout, &policy).is_ok());

        let options = rollout_options_for(&rollout, Path::new("."), vec![None, None]);
        assert_eq!(
            options.environment_reset_options_values,
            rollout.environment_reset_options
        );
        assert!(options.reset_options_value.is_none());

        rollout.groups = 1;
        assert!(validate_run_request(&rollout, &policy).is_err());
        rollout.groups = 2;
        rollout.environment_reset_options.pop();
        assert!(validate_run_request(&rollout, &policy).is_err());
        rollout
            .environment_reset_options
            .push(json!(["not", "an", "object"]));
        assert!(validate_run_request(&rollout, &policy).is_err());
        rollout.environment_reset_options[1] = json!({});
        rollout.reset_options = json!({"shared": true});
        assert!(validate_run_request(&rollout, &policy).is_err());
    }

    #[test]
    fn environment_eval_requires_disjoint_seeds_and_one_gate_owner() {
        let policy = OpenEnvConfig::default();
        let mut train = request(OpenEnvRunKind::Train);
        train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
            groups: 20,
            group_size: 1,
            seed_start: None,
            gate: Some(crate::openenv_evaluation::OpenEnvEnvironmentEvalGate {
                min_mean_return: None,
                min_mean_improvement: 0.0,
            }),
        });
        assert!(validate_run_request(&train, &policy).is_ok());
        assert_eq!(
            resolved_environment_eval_seed_start(&train),
            Some(train.seed_start + train.groups as u64)
        );

        train.environment_eval.as_mut().unwrap().groups = 1;
        train.environment_eval.as_mut().unwrap().group_size = 20;
        assert!(validate_run_request(&train, &policy).is_err());
        train.environment_eval.as_mut().unwrap().groups = 20;
        train.environment_eval.as_mut().unwrap().group_size = 1;
        train.environment_eval.as_mut().unwrap().seed_start = Some(1);
        assert!(validate_run_request(&train, &policy).is_err());
        train.environment_eval.as_mut().unwrap().seed_start = Some(100);
        train.post_eval = Some(kiln_eval::PostEvalConfig {
            suite: "held-out".into(),
            data_scope: Default::default(),
            generation: None,
            min_accuracy: Some(0.8),
            include_baseline: false,
        });
        assert!(validate_run_request(&train, &policy).is_err());
    }

    #[test]
    fn environment_eval_preserves_a_distinct_baseline_revision() {
        let policy = OpenEnvConfig::default();
        let mut train = request(OpenEnvRunKind::Train);
        train.adapter = "agent".into();
        train.output_adapter = Some("agent".into());
        train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
            groups: 1,
            group_size: 1,
            seed_start: None,
            gate: None,
        });
        assert!(validate_run_request(&train, &policy).is_err());
    }

    #[test]
    fn cancellation_remains_available_after_training_handoff() {
        let temp = tempfile::tempdir().unwrap();
        let registry =
            OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
        let (status, _) = insert_created(&registry, request(OpenEnvRunKind::Train));
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
        let (legacy, _) = insert_created(&registry, request(OpenEnvRunKind::Train));
        registry
            .update(&legacy.run_id, |status| {
                status.schema = OPENENV_RUN_SCHEMA_V1.into();
                status.state = OpenEnvRunState::TrainingQueued;
                status.finished_unix_ms = Some(now_unix_ms());
            })
            .unwrap();
        assert_eq!(registry.counts().0, 0);
        assert!(
            insert_test_run(&registry, request(OpenEnvRunKind::Train)).is_ok(),
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
        let (run, cancel) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Train));
        state.training_jobs.write().unwrap().insert(
            "train-1".into(),
            training_job(&temp, "train-1", TrainingState::Queued),
        );

        let followed_state = state.clone();
        let followed_run_id = run.run_id.clone();
        let follow = tokio::spawn(async move {
            follow_openenv_training(
                &followed_state,
                &followed_run_id,
                &request(OpenEnvRunKind::Train),
                None,
                "train-1",
                cancel.cancel,
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
            job.adapter_path = Some(test_adapter_path(&temp, "train-1").display().to_string());
        }
        follow.await.unwrap().unwrap();
        let completed = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(completed.state, OpenEnvRunState::Completed);
        assert!(completed.finished_unix_ms.is_some());
        let training = completed.training.as_ref().unwrap();
        let lineage = training
            .training_data
            .as_ref()
            .and_then(|data| data.openenv.as_ref())
            .unwrap();
        assert_eq!(lineage.groups, 2);
        assert_eq!(lineage.rollouts, 6);
        assert_eq!(lineage.seed_min, 17);
        assert_eq!(lineage.seed_max, 18);
        assert_eq!(
            completed
                .artifacts
                .iter()
                .map(|artifact| artifact.kind.as_str())
                .collect::<Vec<_>>(),
            ["train_receipt", "adapter_manifest"]
        );
        for artifact in &completed.artifacts {
            let (path, _, manifest) = state
                .openenv_runs
                .artifact_path(&run.run_id, &artifact.kind)
                .unwrap();
            let (sha256, bytes) = crate::openenv_replay::bounded_artifact_metadata(&path).unwrap();
            assert_eq!(manifest, *artifact);
            assert_eq!(sha256, artifact.sha256);
            assert_eq!(bytes, artifact.bytes);
        }
        assert!(completed.terminal());
    }

    #[test]
    fn openenv_training_evidence_rejects_manifest_drift_before_publication() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let (run, _) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Train));
        let job = training_job(&temp, "train-tampered", TrainingState::Completed);
        state
            .training_jobs
            .write()
            .unwrap()
            .insert(job.job_id.clone(), job);
        let training = training_status_for(&state, "train-tampered").unwrap();
        let manifest_path =
            test_adapter_path(&temp, "train-tampered").join(kiln_train::ADAPTER_MANIFEST_FILENAME);
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["receipt_hash"] = json!(format!("sha256:{}", "0".repeat(64)));
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = publish_openenv_training_evidence(
            &state,
            &run.run_id,
            &request(OpenEnvRunKind::Train),
            &training,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("receipt hash differs"),
            "{error:#}"
        );
        assert!(
            state
                .openenv_runs
                .get(&run.run_id)
                .unwrap()
                .artifacts
                .is_empty()
        );
        assert!(
            !state
                .openenv_runs
                .run_dir(&run.run_id)
                .join(kiln_train::TRAIN_RECEIPT_FILENAME)
                .exists()
        );
    }

    #[tokio::test]
    async fn completed_training_hands_off_to_native_environment_evaluation() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let mut run_request = request(OpenEnvRunKind::Train);
        run_request.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
            groups: 20,
            group_size: 1,
            seed_start: None,
            gate: None,
        });
        let (run, cancel) = insert_created(&state.openenv_runs, run_request.clone());
        state.training_jobs.write().unwrap().insert(
            "train-environment-eval".into(),
            training_job(&temp, "train-environment-eval", TrainingState::Completed),
        );
        follow_openenv_training(
            &state,
            &run.run_id,
            &run_request,
            None,
            "train-environment-eval",
            cancel.cancel,
        )
        .await
        .unwrap();
        let handed_off = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(handed_off.state, OpenEnvRunState::EnvironmentEvaluating);
        assert!(handed_off.finished_unix_ms.is_none());
        assert!(!handed_off.terminal());
        let evaluation = handed_off.environment_evaluation.unwrap();
        assert_eq!(evaluation.seed_start, 2);
        assert_eq!(evaluation.state, OpenEnvEnvironmentEvalState::Pending);
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
        let (run, cancel) = insert_created(&state.openenv_runs, run_request.clone());
        let mut training = training_job(&temp, "train-eval", TrainingState::Completed);
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
        let post_eval = run_request.post_eval.clone();
        let follow = tokio::spawn(async move {
            follow_openenv_training(
                &followed_state,
                &followed_run_id,
                &run_request,
                post_eval.as_ref(),
                "train-eval",
                cancel.cancel,
            )
            .await
        });
        tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
        let evaluating = state.openenv_runs.get(&run.run_id).unwrap();
        assert_eq!(evaluating.state, OpenEnvRunState::PostEvaluating);
        assert!(
            ["train_receipt", "adapter_manifest"]
                .into_iter()
                .all(|kind| {
                    evaluating
                        .artifacts
                        .iter()
                        .any(|artifact| artifact.kind == kind)
                })
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
    async fn http_surface_accepts_fifo_work_when_execution_capacity_is_occupied() {
        let temp = tempfile::tempdir().unwrap();
        let policy = OpenEnvConfig {
            max_active_runs: 1,
            max_tracked_runs: 3,
            ..Default::default()
        };
        let state = test_state(&temp, policy);
        let (active, active_control) =
            insert_created(&state.openenv_runs, request(OpenEnvRunKind::Rollout));
        let active_permit = state
            .openenv_runs
            .acquire(&active.run_id, &active_control)
            .await
            .unwrap();
        let app = routes().with_state(state.clone());
        let mut submitted = request(OpenEnvRunKind::Rollout);
        submitted.idempotency_key = Some("ui:retry:17".into());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let queued: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
        assert_eq!(queued.schema, OPENENV_RUN_SCHEMA_V5);
        assert_eq!(queued.state, OpenEnvRunState::Queued);
        assert_eq!(queued.admission.as_ref().unwrap().queue_position, Some(1));
        assert_eq!(state.openenv_runs.counts(), (1, 1, 2));

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let replayed: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
        assert_eq!(replayed.run_id, queued.run_id);
        assert_eq!(state.openenv_runs.counts(), (1, 1, 2));
        assert_eq!(
            state
                .metrics
                .openenv_run_idempotent_replays
                .load(Ordering::Relaxed),
            1
        );

        let mut conflict = submitted.clone();
        conflict.groups += 1;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&conflict).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "openenv_run_idempotency_conflict"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/openenv/runs/{}", queued.run_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let cancelled: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
        assert_eq!(cancelled.state, OpenEnvRunState::Cancelled);
        assert_eq!(state.openenv_runs.counts(), (1, 0, 2));
        drop(active_permit);
    }

    #[tokio::test]
    async fn failed_discovery_persists_typed_retryable_diagnosis() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let unavailable = listener.local_addr().unwrap();
        drop(listener);
        let mut submitted = request(OpenEnvRunKind::Rollout);
        submitted.environment_urls = vec![format!("http://{unavailable}")];
        let app = routes().with_state(state.clone());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let accepted: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();

        let failed = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = state.openenv_runs.get(&accepted.run_id).unwrap();
                if status.state == OpenEnvRunState::Failed {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let failure = failed.failure.unwrap();
        assert_eq!(failure.code, OpenEnvRunFailureCode::EnvironmentUnavailable);
        assert_eq!(failure.stage, OpenEnvRunFailureStage::Discovery);
        assert!(failure.retryable);
        assert_eq!(failed.error.as_deref(), Some(failure.message.as_str()));
        assert_eq!(state.metrics.openenv_runs_failed.load(Ordering::Relaxed), 1);
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
            OPENENV_RUN_LIST_SCHEMA_V5
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
    async fn http_train_preflight_rejection_is_observable_and_persists_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let metrics = state.metrics.clone();
        let app = routes().with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/runs")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&request(OpenEnvRunKind::Train)).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "mock_mode"
        );
        assert_eq!(
            metrics
                .openenv_training_preflight_rejected
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .openenv_training_preflights_rejected
                .load(Ordering::Relaxed),
            1
        );
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
    async fn direct_training_preflight_uses_the_same_fail_closed_backend_contract() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let metrics = state.metrics.clone();
        let app = routes().with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/training/preflight")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&OpenEnvTrainingPreflightRequest {
                            adapter: "base".into(),
                            output_adapter: "direct-agent".into(),
                            training_config: GrpoConfig::default(),
                            auto_load: true,
                            post_eval: None,
                        })
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "mock_mode"
        );
        assert_eq!(
            metrics
                .openenv_training_preflights_rejected
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .openenv_training_preflights_accepted
                .load(Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn artifact_downloads_require_publication_and_reverify_the_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let (run, _) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Rollout));
        let artifact_bytes = b"{\"group\":1}\n";
        let artifact_path = state
            .openenv_runs
            .run_dir(&run.run_id)
            .join("rollouts.jsonl");
        std::fs::write(&artifact_path, artifact_bytes).unwrap();
        let artifact_url = format!("/v1/openenv/runs/{}/artifacts/dataset", run.run_id);
        let app = routes().with_state(state.clone());

        let unpublished = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&artifact_url)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unpublished.status(), StatusCode::NOT_FOUND);

        let sha256 = crate::openenv_replay::sha256_bytes(artifact_bytes);
        state
            .openenv_runs
            .update(&run.run_id, |status| {
                status.state = OpenEnvRunState::RolloutReady;
                status.artifacts = vec![OpenEnvArtifact {
                    kind: "dataset".into(),
                    url: artifact_url.clone(),
                    sha256: sha256.clone(),
                    bytes: artifact_bytes.len(),
                }];
            })
            .unwrap();

        let published = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&artifact_url)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(published.status(), StatusCode::OK);
        assert_eq!(
            published.headers()[header::CONTENT_LENGTH],
            artifact_bytes.len().to_string()
        );
        assert_eq!(published.headers()[header::ETAG], format!("\"{sha256}\""));
        assert_eq!(
            published.headers()[header::CACHE_CONTROL],
            "private, no-store"
        );
        let body = axum::body::to_bytes(published.into_body(), artifact_bytes.len())
            .await
            .unwrap();
        assert_eq!(body.as_ref(), artifact_bytes);

        std::fs::write(&artifact_path, b"{\"group\":2}\n").unwrap();
        let drifted = app
            .oneshot(
                Request::builder()
                    .uri(&artifact_url)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(drifted.status(), StatusCode::CONFLICT);
        let body = axum::body::to_bytes(drifted.into_body(), 16 * 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "openenv_artifact_integrity_failed"
        );
    }

    #[tokio::test]
    async fn task_catalog_surface_pages_reference_shaped_tasks_and_tracks_outcomes() {
        let (environment_url, server) = task_fixture().await;
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp, OpenEnvConfig::default());
        let metrics = state.metrics.clone();
        let app = routes().with_state(state);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/tasks")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "environment_urls": [environment_url],
                            "split": "TRAIN",
                            "start": 1,
                            "limit": 1
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let catalog: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(catalog["schema"], OPENENV_TASK_CATALOG_SCHEMA_V1);
        assert_eq!(catalog["catalogs"][0]["catalog"]["task_api"], "available");
        assert_eq!(catalog["catalogs"][0]["catalog"]["selected_split"], "train");
        assert_eq!(catalog["catalogs"][0]["catalog"]["num_tasks"], 3);
        assert_eq!(
            catalog["catalogs"][0]["catalog"]["tasks"][0],
            json!({"id": 1, "prompt": "2 + 2", "answer": "4"})
        );
        assert_eq!(
            metrics
                .openenv_task_catalog_inspections_started
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .openenv_task_catalog_inspections_completed
                .load(Ordering::Relaxed),
            1
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/openenv/tasks")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "environment_urls": ["http://127.0.0.1:1"],
                            "limit": MAX_OPENENV_TASK_PAGE_SIZE + 1
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            metrics
                .openenv_task_catalog_inspections_failed
                .load(Ordering::Relaxed),
            1
        );
        server.abort();
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

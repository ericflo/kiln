//! Native OpenEnv rollout collection and GRPO submission.
//!
//! The protocol boundary lives in `kiln-openenv`. This module composes it with
//! Kiln chat generation and canonical trajectory/GRPO types:
//!
//!   inspect -> reset(seed) -> model action -> step -> reward -> trajectory
//!           -> grouped JSONL -> optional `/v1/train/grpo`
//!   request -> persisted `/v1/openenv/runs` lifecycle -> verified manifest artifact

use std::fs::OpenOptions;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use clap::Subcommand;
use console::style;
use futures::{StreamExt, TryStreamExt, stream};
use kiln_openenv::{
    OPENENV_MAX_CLIENT_MESSAGE_BYTES, OpenEnvActionValidator, OpenEnvClient, OpenEnvClientError,
    OpenEnvIdentity, OpenEnvInspection, OpenEnvObservation, OpenEnvProtocolError,
    OpenEnvTaskApiSupport, OpenEnvTaskCatalog,
};
use kiln_train::{
    AgenticGroup, BehaviorPolicy, ChatMessage, GrpoConfig, OpenEnvEpisodeTerminationV1,
    OpenEnvRolloutProvenanceV1, RolloutBehaviorPolicyIdentityV1, ScoredRollout, TurnKind,
    TurnSegment,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::config::default_server_url;
use crate::openenv_replay::{
    MAX_OPENENV_ARTIFACT_BYTES, OPENENV_REPLAY_SCHEMA_V1, OpenEnvReplayCandidate,
    OpenEnvReplayExchange, OpenEnvReplayExchangeResult, OpenEnvReplayGroup, OpenEnvReplayManifest,
    connect_and_reset_with_capacity_checked, encode_replay, replay_openenv,
    sha256_bytes as replay_sha256, verify_openenv_artifacts,
};

const MAX_OPENENV_ENVIRONMENTS: usize = 64;
const MAX_OPENENV_GROUPS: usize = 16_384;
const MAX_OPENENV_GROUP_SIZE: usize = 256;
const MAX_OPENENV_ROLLOUTS: usize = 16_384;
const MAX_OPENENV_STEPS: usize = 256;
const MAX_OPENENV_CONCURRENCY: usize = 256;
const MAX_OPENENV_ACTION_TOKENS: usize = 16_384;
const MAX_OPENENV_RECOVERABLE_ERRORS: usize = 64;
const MAX_OPENENV_CAPACITY_WAIT_SECONDS: u64 = 3_600;
pub(crate) const MAX_OPENENV_TASK_PAGE_SIZE: usize = 200;
const MAX_OPENENV_DATASET_BYTES: usize = 256 * 1024 * 1024;
const MAX_OPENENV_SUMMARY_BYTES: usize = 256 * 1024 * 1024;
/// Combined serialized size of the simultaneously retained training, replay,
/// and receipt projections. Unlike the per-artifact limit, this is charged as
/// protocol turns arrive so one legal but adversarial group cannot accumulate
/// an enormous working set before artifact serialization.
const MAX_OPENENV_RETAINED_BYTES: usize = 512 * 1024 * 1024;
// Reserve one complete control-plane request for the materialized training
// contract that train workflows attach after collection. The request endpoint
// enforces this same bound, so collection + contract remains within the public
// aggregate budget even at the artifact limit.
const MAX_OPENENV_COLLECTION_RETAINED_BYTES: usize =
    MAX_OPENENV_RETAINED_BYTES - MAX_OPENENV_RUN_REQUEST_BYTES;
const MAX_OPENENV_RESET_OPTIONS_BYTES: usize = OPENENV_MAX_CLIENT_MESSAGE_BYTES - 1024;
const MAX_OPENENV_RUN_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_KILN_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const CHAT_TIMEOUT: Duration = Duration::from_secs(180);
pub const OPENENV_TRAINING_PREFLIGHT_SCHEMA_V1: &str = "kiln.openenv-training-preflight.v1";
pub const OPENENV_TRAINING_CONTRACT_SCHEMA_V1: &str = "kiln.openenv-training-contract.v1";

/// Policy inference used while an OpenEnv episode is live.
///
/// The CLI uses `Http`; the server control plane uses `InProcess` so a
/// server-owned run traverses the exact chat handler, admission, metrics, and
/// adapter-selection path without depending on a loopback listener.
#[derive(Clone)]
pub(crate) enum OpenEnvPolicyTransport {
    Http {
        client: reqwest::Client,
        kiln_url: String,
    },
    InProcess(crate::state::AppState),
}

impl OpenEnvPolicyTransport {
    #[allow(unused_mut)] // test-only mock transport removes provenance before dispatch
    async fn complete(&self, mut body: Value) -> Result<Value> {
        match self {
            Self::Http { client, kiln_url } => {
                let response = client
                    .post(format!(
                        "{}/v1/chat/completions",
                        kiln_url.trim_end_matches('/')
                    ))
                    .header("x-kiln-client", "openenv")
                    .json(&body)
                    .send()
                    .await
                    .context("send OpenEnv policy completion request")?;
                let status = response.status();
                let response_body = read_kiln_json_bounded(response, "action generation").await?;
                anyhow::ensure!(
                    status.is_success(),
                    "Kiln action generation returned HTTP {status}: {}",
                    serde_json::to_string(&response_body).unwrap_or_default()
                );
                Ok(response_body)
            }
            Self::InProcess(state) => {
                #[cfg(test)]
                if matches!(
                    state.backend.as_ref(),
                    crate::state::ModelBackend::Mock { .. }
                ) {
                    body["rollout_provenance"] = Value::Bool(false);
                    let adapter = body.get("adapter").and_then(Value::as_str);
                    let behavior_policy = state
                        .openenv_behavior_policy_identity(adapter)
                        .map_err(anyhow::Error::msg)?;
                    let mut response =
                        crate::api::completions::openenv_chat_completion(state, body).await?;
                    response["choices"][0]["rollout_provenance"] = json!({
                        "behavior_policy": behavior_policy
                    });
                    return Ok(response);
                }
                crate::api::completions::openenv_chat_completion(state, body).await
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvCollectionStage {
    Discovering,
    Collecting,
    Revalidating,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct OpenEnvCollectionProgress {
    pub stage: OpenEnvCollectionStage,
    pub groups_completed: usize,
    pub groups_total: usize,
    pub rollouts_completed: usize,
}

/// Cooperative cancellation and low-cost progress publication for a
/// server-owned OpenEnv run.
#[derive(Clone, Default)]
pub(crate) struct OpenEnvCollectionControl {
    cancel: Arc<AtomicBool>,
    progress: Option<Arc<dyn Fn(OpenEnvCollectionProgress) + Send + Sync>>,
    discovered: Option<Arc<dyn Fn(Vec<OpenEnvIdentity>) + Send + Sync>>,
}

impl OpenEnvCollectionControl {
    pub(crate) fn new(
        cancel: Arc<AtomicBool>,
        progress: Option<Arc<dyn Fn(OpenEnvCollectionProgress) + Send + Sync>>,
        discovered: Option<Arc<dyn Fn(Vec<OpenEnvIdentity>) + Send + Sync>>,
    ) -> Self {
        Self {
            cancel,
            progress,
            discovered,
        }
    }

    fn publish(&self, progress: OpenEnvCollectionProgress) {
        if let Some(callback) = &self.progress {
            callback(progress);
        }
    }

    fn publish_discovered(&self, environments: Vec<OpenEnvIdentity>) {
        if let Some(callback) = &self.discovered {
            callback(environments);
        }
    }

    fn ensure_active(&self) -> Result<()> {
        anyhow::ensure!(
            !self.cancel.load(Ordering::Relaxed),
            "OpenEnv run cancelled"
        );
        Ok(())
    }
}

pub(crate) const OPENENV_OVERVIEW: &str = r#"Inspect OpenEnv servers, collect grouped stateful episodes, and train a Kiln LoRA directly from environment-owned rewards.

Kiln discovers each environment over HTTP, including its optional bounded Task API catalog, opens one WebSocket session per episode, resets every candidate in a GRPO group with the same deterministic seed, asks the selected Kiln policy for schema-shaped JSON actions, and records every action, observation, reward, termination, environment identity, and content hash in canonical agentic trajectory JSONL. Task rows are discovery data: OpenEnv defines no automatic task-to-reset mapping, so portable training continues to use explicit reset options and seeds.

`rollout` writes the exact reusable GRPO corpus, an exact replay transcript, and a detailed summary receipt. `verify` validates the three-artifact bundle without contacting a server; `replay` re-executes the captured reset/action protocol against the content-addressed environments. `train` writes those artifacts and submits the in-memory groups to `/v1/train/grpo` with the explicit native on-policy behavior-policy contract. `start` submits the full persisted server-run contract, including paired held-out evaluation, while `artifact` atomically materializes one manifest-declared object after independent byte and SHA-256 verification. Protected environments use `--credential-env`; only the non-secret authentication method enters environment identity. Start `kiln serve` first.
"#;

pub(crate) const OPENENV_EXAMPLES: &str = r#"Examples:
  kiln openenv inspect --environment http://127.0.0.1:8000
      Check health and print the environment metadata, schemas, protocol
      profile, WebSocket URL, and content-addressed schema identity.

  kiln openenv tasks --environment http://127.0.0.1:8000 --split train
      Discover the optional Task API and print a bounded page of arbitrary
      dataset-backed task rows without treating them as reset payloads.

  kiln openenv rollout --environment http://127.0.0.1:8000 --groups 8 --group-size 4
      Collect 32 live episodes as eight seed-matched GRPO groups and write
      openenv.rollouts.jsonl plus openenv.rollout-summary.json.

  kiln openenv train --environment http://127.0.0.1:8000 --output-adapter wordle-agent
      Collect a native on-policy batch, submit GRPO training, and auto-load the
      completed adapter.

  kiln openenv inspect --environment https://arcade.example.com/openenv --credential-env ARCADE_OPENENV_TOKEN
      Authenticate HTTP discovery and the WebSocket upgrade with a bearer
      token read from the named environment variable without persisting it.

  kiln openenv runs
      List server-owned OpenEnv workflows, including live trainer and linked
      post-evaluation state.

  kiln openenv start --request openenv-run.json --idempotency-key experiment:wordle:17 --follow
      Submit a retry-safe persisted server-run request and follow collection,
      native GRPO, static evaluation, and paired held-out evaluation. Exact
      retained retries recover the original run.

  kiln openenv status 80a26e21-8451-4a64-8666-890c06fd80bd --follow
      Follow one persisted server workflow through collection, native GRPO,
      requested evaluation, and its terminal outcome.

  kiln openenv cancel 80a26e21-8451-4a64-8666-890c06fd80bd
      Cooperatively cancel whichever collection, training, or evaluation
      phase currently owns the work.

  kiln openenv artifact 80a26e21-8451-4a64-8666-890c06fd80bd environment_eval_receipt --output receipt.json
      Follow the run's returned artifact manifest, require the exact server
      length and ETag, rehash the streamed bytes, and publish atomically.

  kiln openenv verify --summary openenv.rollout-summary.json
      Verify the dataset, replay transcript, receipt hashes, rollout
      provenance, rewards, and counts entirely offline.

  kiln openenv replay --summary openenv.rollout-summary.json
      Reconnect to the captured environment identities and assert every reset,
      observation, reward, error, done flag, and final state exactly.

  kiln openenv train --environment http://127.0.0.1:8000 --environment http://127.0.0.1:8001 --adapter wordle-agent --output-adapter arcade-agent --groups 16
      Continue training an existing policy across multiple environments,
      assigning groups round-robin while preserving one environment and seed
      per comparison group.
"#;

#[derive(clap::Args, Debug, Clone)]
pub struct OpenEnvRolloutArgs {
    /// OpenEnv HTTP base URL. Repeat to assign groups round-robin across environments.
    #[arg(long = "environment", value_name = "URL", required = true)]
    environment_urls: Vec<String>,

    /// Bearer-token environment variable aligned with each --environment.
    /// Repeat once per URL when used; pass '-' for an unauthenticated slot.
    #[arg(
        long = "credential-env",
        value_name = "ENV_OR_DASH",
        allow_hyphen_values = true
    )]
    credential_envs: Vec<String>,

    /// Running Kiln server URL used for policy generation and training
    #[arg(long = "url", default_value_t = default_server_url())]
    kiln_url: String,

    /// Behavior adapter, or `base`/`none`/`null` for the base model
    #[arg(long, default_value = "base")]
    adapter: String,

    /// Number of independent task/seed comparison groups
    #[arg(long, default_value_t = 8)]
    groups: usize,

    /// Candidate episodes sharing each group's environment reset and seed
    #[arg(long = "group-size", default_value_t = 4)]
    group_size: usize,

    /// First deterministic environment seed; later groups increment by one
    #[arg(long = "seed-start", default_value_t = 0)]
    seed_start: u64,

    /// File containing one JSON object merged into every reset; Kiln sets `seed`
    #[arg(long = "reset-options", value_name = "FILE")]
    reset_options: Option<PathBuf>,

    /// JSON-object file aligned with each --environment. Repeat once per URL;
    /// pass '-' for an empty object. Mutually exclusive with --reset-options.
    #[arg(
        long = "environment-reset-options",
        value_name = "FILE_OR_DASH",
        allow_hyphen_values = true
    )]
    environment_reset_options: Vec<PathBuf>,

    /// Maximum model actions in one episode
    #[arg(long = "max-steps", default_value_t = 8)]
    max_steps: usize,

    /// Maximum simultaneous candidate sessions within a group
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Maximum tokens generated for one JSON action
    #[arg(long = "max-action-tokens", default_value_t = 256)]
    max_action_tokens: usize,

    /// Explicit maximum reasoning tokens before Kiln closes thinking. Omit
    /// for unlimited thinking; unfinished reasoning is discarded from GRPO.
    #[arg(long = "thinking-budget-tokens")]
    thinking_budget_tokens: Option<usize>,

    /// Policy sampling temperature
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Explicitly enable or disable Qwen thinking for action generation
    #[arg(
        long,
        action = clap::ArgAction::Set,
        value_parser = clap::value_parser!(bool),
        default_value_t = true
    )]
    thinking: bool,

    /// Terminal reward assigned to malformed model actions or protocol errors
    #[arg(long = "protocol-error-reward", default_value_t = -1.0)]
    protocol_error_reward: f64,

    /// Recoverable OpenEnv errors allowed before ending an episode
    #[arg(long = "max-recoverable-errors", default_value_t = 3)]
    max_recoverable_errors: usize,

    /// Maximum time to wait for a saturated OpenEnv server
    #[arg(long = "capacity-wait-seconds", default_value_t = 300)]
    capacity_wait_seconds: u64,

    /// Canonical GRPO JSONL output
    #[arg(long, default_value = "openenv.rollouts.jsonl")]
    output: PathBuf,

    /// Exact content-addressed environment replay transcript
    #[arg(long = "replay-output", default_value = "openenv.replay.json")]
    replay_output: PathBuf,

    /// Content-addressed rollout summary and provenance receipt
    #[arg(
        long = "summary-output",
        default_value = "openenv.rollout-summary.json"
    )]
    summary_output: PathBuf,
}

#[derive(Subcommand, Debug, Clone)]
pub enum OpenEnvCommands {
    /// Discover and content-address one OpenEnv server
    Inspect {
        /// OpenEnv HTTP base URL
        #[arg(long, value_name = "URL")]
        environment: String,

        /// Environment variable containing this origin's bearer token
        #[arg(long = "credential-env", value_name = "ENV")]
        credential_env: Option<String>,

        /// Emit the complete inspection as JSON
        #[arg(long)]
        json: bool,
    },

    /// Inspect an optional dataset-backed OpenEnv Task API catalog
    Tasks {
        /// OpenEnv HTTP base URL
        #[arg(long, value_name = "URL")]
        environment: String,

        /// Registered environment name; optional when exactly one is advertised
        #[arg(long = "environment-name", value_name = "NAME")]
        environment_name: Option<String>,

        /// Split to page; omit to list the advertised splits only
        #[arg(long, value_name = "SPLIT")]
        split: Option<String>,

        /// Zero-based first task in the requested page
        #[arg(long, default_value_t = 0)]
        start: u64,

        /// Maximum task rows to return (1..=200)
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Environment variable containing this origin's bearer token
        #[arg(long = "credential-env", value_name = "ENV")]
        credential_env: Option<String>,

        /// Emit the complete bounded catalog as JSON
        #[arg(long)]
        json: bool,
    },

    /// Collect seed-matched stateful episodes as canonical GRPO JSONL
    Rollout {
        #[command(flatten)]
        rollout: OpenEnvRolloutArgs,
    },

    /// Collect live episodes and immediately submit native on-policy GRPO
    Train {
        #[command(flatten)]
        rollout: OpenEnvRolloutArgs,

        /// LoRA adapter created by this training job
        #[arg(long = "output-adapter")]
        output_adapter: String,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Auto-load the new adapter after successful training
        #[arg(
            long,
            action = clap::ArgAction::Set,
            value_parser = clap::value_parser!(bool),
            default_value_t = true
        )]
        auto_load: bool,
    },

    /// List server-owned OpenEnv workflow runs
    Runs {
        /// Running Kiln server URL
        #[arg(long = "url", default_value_t = default_server_url())]
        kiln_url: String,

        /// Emit the complete retained run list as JSON
        #[arg(long)]
        json: bool,
    },

    /// Start a fully persisted server-owned OpenEnv workflow from JSON
    Start {
        /// Regular non-symlink JSON object (max 1 MiB) matching OpenEnvRunRequest
        #[arg(long, value_name = "FILE")]
        request: PathBuf,

        /// Stable non-secret retry key; must match any idempotency_key in FILE
        #[arg(long, value_name = "KEY")]
        idempotency_key: Option<String>,

        /// Running Kiln server URL
        #[arg(long = "url", default_value_t = default_server_url())]
        kiln_url: String,

        /// Follow collection, training, and requested evaluations to terminal
        #[arg(long)]
        follow: bool,

        /// Emit JSON (only the terminal snapshot when following)
        #[arg(long)]
        json: bool,
    },

    /// Inspect or follow one server-owned OpenEnv workflow
    Status {
        /// Persisted OpenEnv run UUID
        run_id: String,

        /// Running Kiln server URL
        #[arg(long = "url", default_value_t = default_server_url())]
        kiln_url: String,

        /// Poll until the complete workflow reaches a terminal outcome
        #[arg(long)]
        follow: bool,

        /// Emit JSON (only the terminal snapshot when following)
        #[arg(long)]
        json: bool,
    },

    /// Cooperatively cancel a server-owned OpenEnv workflow
    Cancel {
        /// Persisted OpenEnv run UUID
        run_id: String,

        /// Running Kiln server URL
        #[arg(long = "url", default_value_t = default_server_url())]
        kiln_url: String,

        /// Emit the updated workflow status as JSON
        #[arg(long)]
        json: bool,
    },

    /// Download and independently verify one manifest-declared run artifact
    Artifact {
        /// Persisted OpenEnv run UUID
        run_id: String,

        /// Exact kind currently returned in the run's artifacts array
        kind: String,

        /// Destination file; it must not exist unless --force is explicit
        #[arg(long, value_name = "FILE")]
        output: PathBuf,

        /// Running Kiln server URL
        #[arg(long = "url", default_value_t = default_server_url())]
        kiln_url: String,

        /// Atomically replace an existing destination
        #[arg(long)]
        force: bool,

        /// Emit the verified local artifact receipt as JSON
        #[arg(long)]
        json: bool,
    },

    /// Verify a rollout dataset, replay transcript, and summary receipt offline
    Verify {
        /// Content-addressed rollout summary receipt
        #[arg(long, default_value = "openenv.rollout-summary.json")]
        summary: PathBuf,

        /// Override the dataset path recorded in the summary
        #[arg(long)]
        dataset: Option<PathBuf>,

        /// Override the replay path recorded in the summary
        #[arg(long)]
        replay: Option<PathBuf>,

        /// Emit the verification report as JSON
        #[arg(long)]
        json: bool,
    },

    /// Replay a captured transcript against its live OpenEnv environments
    Replay {
        /// Content-addressed rollout summary receipt
        #[arg(long, default_value = "openenv.rollout-summary.json")]
        summary: PathBuf,

        /// Override the dataset path recorded in the summary
        #[arg(long)]
        dataset: Option<PathBuf>,

        /// Override the replay path recorded in the summary
        #[arg(long)]
        replay: Option<PathBuf>,

        /// Bearer-token environment variable aligned with each captured
        /// environment. Repeat once per environment; pass '-' for no token.
        #[arg(
            long = "credential-env",
            value_name = "ENV_OR_DASH",
            allow_hyphen_values = true
        )]
        credential_envs: Vec<String>,

        /// Maximum simultaneous replay sessions within a group
        #[arg(long, default_value_t = 4)]
        concurrency: usize,

        /// Maximum time to wait for a saturated OpenEnv server
        #[arg(long = "capacity-wait-seconds", default_value_t = 300)]
        capacity_wait_seconds: u64,

        /// Emit the replay report as JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone)]
pub struct OpenEnvRolloutOptions {
    pub kiln_url: String,
    pub environment_urls: Vec<String>,
    /// One secret environment-variable name per environment. An empty vector
    /// means all endpoints are unauthenticated. Values are runtime-only and
    /// never enter summaries, replay manifests, or training receipts.
    pub credential_envs: Vec<Option<String>>,
    pub adapter: String,
    pub groups: usize,
    pub group_size: usize,
    pub seed_start: u64,
    pub reset_options: Option<PathBuf>,
    /// Server/API callers provide reset options directly; CLI callers use
    /// `reset_options` as a file. The two sources are mutually exclusive.
    pub reset_options_value: Option<Value>,
    /// Optional files aligned one-for-one with `environment_urls`. `None`
    /// represents an empty object for a CLI `-` slot.
    pub environment_reset_options: Vec<Option<PathBuf>>,
    /// Server/API equivalent of `environment_reset_options`. File-backed and
    /// inline aligned plans are mutually exclusive.
    pub environment_reset_options_values: Vec<Value>,
    pub max_steps: usize,
    pub concurrency: usize,
    pub max_action_tokens: usize,
    /// Explicit reasoning limit for this collection. `None` means unlimited;
    /// Kiln does not infer a reserve or inherit a server default.
    pub thinking_budget_tokens: Option<usize>,
    pub temperature: f32,
    pub thinking: bool,
    pub protocol_error_reward: f64,
    pub max_recoverable_errors: usize,
    pub capacity_wait_seconds: u64,
    pub output: PathBuf,
    pub replay_output: PathBuf,
    pub summary_output: PathBuf,
}

impl std::fmt::Debug for OpenEnvRolloutOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvRolloutOptions")
            .field("kiln_url", &self.kiln_url)
            .field("environment_urls", &self.environment_urls)
            .field(
                "authenticated_environments",
                &self
                    .credential_envs
                    .iter()
                    .filter(|item| item.is_some())
                    .count(),
            )
            .field("adapter", &self.adapter)
            .field("groups", &self.groups)
            .field("group_size", &self.group_size)
            .field("seed_start", &self.seed_start)
            .field("max_steps", &self.max_steps)
            .field("concurrency", &self.concurrency)
            .field("max_action_tokens", &self.max_action_tokens)
            .field("thinking_budget_tokens", &self.thinking_budget_tokens)
            .field("thinking", &self.thinking)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct OpenEnvTrainOptions {
    pub rollout: OpenEnvRolloutOptions,
    pub output_adapter: String,
    pub lora_rank: Option<usize>,
    pub auto_load: bool,
}

fn default_openenv_training_adapter() -> String {
    "base".to_string()
}

fn default_openenv_training_auto_load() -> bool {
    true
}

/// Device-free OpenEnv training intent sent before any environment is
/// contacted. Kiln owns the rollout-derived GRPO fields and returns their
/// exact effective values in [`OpenEnvTrainingPreflightReceipt`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingPreflightRequest {
    #[serde(default = "default_openenv_training_adapter")]
    pub adapter: String,
    pub output_adapter: String,
    #[serde(default)]
    pub training_config: GrpoConfig,
    #[serde(default = "default_openenv_training_auto_load")]
    pub auto_load: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Point-in-time native trainer capacity observed during OpenEnv preflight.
/// It is evidence, not a reservation; final submission remains authoritative.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingCapacitySnapshot {
    pub checked_unix_ms: u64,
    pub queued_jobs: usize,
    pub max_queued_jobs: usize,
    pub tracked_jobs: usize,
    pub max_tracked_jobs: usize,
}

/// Successful server-owned admission receipt for direct OpenEnv training.
/// Both returned training fields must be submitted unchanged after collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingPreflightReceipt {
    pub schema: String,
    pub effective_config: GrpoConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
    /// Immutable policy revision admitted before the first environment reset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_policy: Option<RolloutBehaviorPolicyIdentityV1>,
    pub capacity: OpenEnvTrainingCapacitySnapshot,
    /// Always false in v1. Collection does not hold scarce trainer capacity.
    pub capacity_reserved: bool,
}

/// Immutable native-training intent admitted before OpenEnv collection.
///
/// Persisted workflows and direct CLI receipts use this same wire type so the
/// exact materialized GRPO and post-evaluation settings survive queue waits,
/// process restarts, and artifact handoff without being recomputed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingContract {
    pub schema: String,
    pub effective_config: GrpoConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
    /// Immutable policy revision that collection must retain through final
    /// native trainer admission. Absent only on legacy persisted v1 records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_policy: Option<RolloutBehaviorPolicyIdentityV1>,
}

impl OpenEnvTrainingPreflightReceipt {
    pub fn training_contract(&self) -> OpenEnvTrainingContract {
        OpenEnvTrainingContract {
            schema: OPENENV_TRAINING_CONTRACT_SCHEMA_V1.to_string(),
            effective_config: self.effective_config.clone(),
            post_eval: self.post_eval.clone(),
            behavior_policy: self.behavior_policy.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRolloutSummary {
    pub schema: String,
    pub kiln_url: String,
    pub adapter: Option<String>,
    pub adapter_label: String,
    /// Exact base-model/runtime/adapter revision that sampled every action in
    /// this collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_policy: Option<RolloutBehaviorPolicyIdentityV1>,
    pub environments: Vec<OpenEnvInspection>,
    pub groups: usize,
    pub group_size: usize,
    pub rollout_count: usize,
    pub seed_start: u64,
    pub max_steps: usize,
    pub concurrency: usize,
    pub max_action_tokens: usize,
    /// Explicit reasoning limit used by policy requests. Omitted means
    /// unlimited thinking, not an inferred final-answer reserve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<usize>,
    pub temperature: f32,
    pub thinking: bool,
    pub protocol_error_reward: f64,
    pub max_recoverable_errors: usize,
    pub capacity_wait_seconds: u64,
    /// Legacy v2 digest of the single shared reset-options object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_options_sha256: Option<String>,
    /// Digest of the ordered, seed-free reset template for every environment.
    /// Kiln inserts each deterministic group seed after computing this digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_plan_sha256: Option<String>,
    pub output_path: String,
    pub replay_output_path: String,
    pub summary_output_path: String,
    pub dataset_sha256: String,
    pub dataset_bytes: usize,
    pub replay_sha256: String,
    pub replay_bytes: usize,
    pub stats: OpenEnvRolloutStats,
    pub rollouts: Vec<OpenEnvRolloutRecord>,
    /// Present when the collection was considered for native GRPO. The
    /// artifact bundle retains every observed episode; this projection names
    /// exactly which groups/completions were eligible for optimizer work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_selection: Option<OpenEnvTrainingSelection>,
    /// Exact native training settings admitted before collection. Rollout-only
    /// receipts omit this field; train receipts retain it even if final trainer
    /// submission fails.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_contract: Option<OpenEnvTrainingContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_submission: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingSelection {
    pub groups_submitted: usize,
    pub rollouts_submitted: usize,
    pub no_output_rollouts_discarded: usize,
    pub no_usable_output_groups_skipped: usize,
    pub no_reward_gradient_groups_skipped: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRolloutStats {
    pub mean_episode_return: f64,
    pub min_episode_return: Option<f64>,
    pub max_episode_return: Option<f64>,
    pub done_count: usize,
    pub max_steps_count: usize,
    pub invalid_model_action_count: usize,
    pub protocol_error_count: usize,
    pub recoverable_protocol_error_count: usize,
    pub capacity_retry_count: usize,
    pub total_environment_steps: usize,
    pub total_model_tokens: usize,
    pub mean_model_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRolloutRecord {
    pub group_index: usize,
    pub candidate_index: usize,
    pub environment_name: String,
    pub environment_url: String,
    pub seed: u64,
    pub steps: usize,
    pub episode_return: f64,
    pub terminal_done: bool,
    pub termination: OpenEnvEpisodeTerminationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_error_code: Option<String>,
    pub recoverable_protocol_errors: usize,
    pub capacity_retries: usize,
    pub model_tokens: usize,
    pub model_latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct OpenEnvCollection {
    pub groups: Vec<AgenticGroup>,
    pub replay: OpenEnvReplayManifest,
    pub summary: OpenEnvRolloutSummary,
}

#[derive(Debug, Serialize)]
struct CandidateRollout {
    candidate_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<ChatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reset_observation: Option<OpenEnvObservation>,
    #[serde(skip)]
    messages_sha256: String,
    #[serde(skip)]
    reset_observation_sha256: String,
    rollout: ScoredRollout,
    replay: OpenEnvReplayCandidate,
    record: OpenEnvRolloutRecord,
    #[serde(skip)]
    retained_bytes: usize,
}

#[derive(Debug)]
struct OpenEnvRetainedByteBudget {
    used: AtomicUsize,
    limit: usize,
}

impl OpenEnvRetainedByteBudget {
    fn new(limit: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            limit,
        }
    }

    fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    fn charge(&self, bytes: usize, retained: &str) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(bytes)
                    .filter(|next| *next <= self.limit)
            })
            .map(|_| ())
            .map_err(|current| {
                anyhow!(
                    "OpenEnv retained rollout representations would exceed the {} byte collection budget while retaining {retained} (currently {current}, additional {bytes}); reduce groups, group size, concurrency, max steps, action size, or environment observation size",
                    self.limit
                )
            })
    }

    fn replace(&self, old_bytes: usize, new_bytes: usize, retained: &str) -> Result<()> {
        if new_bytes > old_bytes {
            self.charge(new_bytes - old_bytes, retained)
        } else {
            let released = old_bytes - new_bytes;
            let previous = self.used.fetch_sub(released, Ordering::Relaxed);
            debug_assert!(previous >= released);
            Ok(())
        }
    }
}

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("serialized byte count overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct BoundedWriter<W> {
    inner: W,
    bytes: usize,
    limit: usize,
    label: &'static str,
}

impl<W> BoundedWriter<W> {
    fn new(inner: W, limit: usize, label: &'static str) -> Self {
        Self {
            inner,
            bytes: 0,
            limit,
            label,
        }
    }
}

impl<W: Write> Write for BoundedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("bounded writer byte count overflow"))?;
        if next > self.limit {
            return Err(std::io::Error::other(format!(
                "{} exceeded the {} byte artifact limit",
                self.label, self.limit
            )));
        }
        self.inner.write_all(bytes)?;
        self.bytes = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

struct Sha256Writer<'a> {
    hasher: &'a mut Sha256,
}

impl Write for Sha256Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.hasher.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_len(value: &impl Serialize, label: &str) -> Result<usize> {
    let mut writer = CountingWriter::default();
    serde_json::to_writer(&mut writer, value)
        .with_context(|| format!("count serialized OpenEnv {label} bytes"))?;
    Ok(writer.bytes)
}

fn pretty_serialized_len(value: &impl Serialize, label: &str) -> Result<usize> {
    let mut writer = CountingWriter::default();
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("count pretty-serialized OpenEnv {label} bytes"))?;
    Ok(writer.bytes)
}

fn charge_serialized(
    budget: &OpenEnvRetainedByteBudget,
    candidate_bytes: &mut usize,
    value: &impl Serialize,
    label: &str,
) -> Result<()> {
    let bytes = serialized_len(value, label)?;
    budget.charge(bytes, label)?;
    *candidate_bytes = candidate_bytes
        .checked_add(bytes)
        .context("OpenEnv candidate retained-byte count overflow")?;
    Ok(())
}

#[derive(Debug)]
struct ModelAction {
    trajectory_content: String,
    action: Value,
    total_tokens: usize,
    latency_ms: f64,
    behavior_policy: RolloutBehaviorPolicyIdentityV1,
}

#[derive(Debug)]
enum ModelActionFailure {
    NoOutput {
        message: String,
        reasoning: Option<String>,
        total_tokens: usize,
        latency_ms: f64,
        behavior_policy: RolloutBehaviorPolicyIdentityV1,
    },
    Invalid {
        code: &'static str,
        message: String,
        raw: Option<String>,
        trajectory_content: Option<String>,
        total_tokens: usize,
        latency_ms: f64,
        behavior_policy: RolloutBehaviorPolicyIdentityV1,
    },
    Request(anyhow::Error),
}

/// Run native OpenEnv discovery, rollout collection, or rollout-and-train.
pub async fn run_openenv(command: &OpenEnvCommands) -> Result<()> {
    match command {
        OpenEnvCommands::Inspect {
            environment,
            credential_env,
            json,
        } => {
            let inspection = inspect_openenv(environment, credential_env.as_deref()).await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&inspection)?);
            } else {
                let identity = &inspection.identity;
                println!(
                    "{} OpenEnv environment {} is ready",
                    style("✓").green().bold(),
                    style(&identity.metadata.name).cyan().bold()
                );
                println!("  Base URL:       {}", identity.base_url);
                println!("  WebSocket:      {}", identity.websocket_url);
                println!("  Client profile: {}", identity.client_profile);
                println!(
                    "  OpenAPI version: {}",
                    identity.openapi_version.as_deref().unwrap_or("unspecified")
                );
                if let Some(discovery_sha256) = &identity.discovery_sha256 {
                    println!("  Discovery SHA-256: {discovery_sha256}");
                }
                println!("  Schema SHA-256: {}", identity.schema_sha256);
                println!("  Description:    {}", identity.metadata.description.trim());
                println!();
                println!("  Action schema:");
                for line in serde_json::to_string_pretty(&inspection.schema.action)?.lines() {
                    println!("    {line}");
                }
            }
        }
        OpenEnvCommands::Tasks {
            environment,
            environment_name,
            split,
            start,
            limit,
            credential_env,
            json,
        } => {
            let catalog = inspect_openenv_tasks(
                environment,
                environment_name.as_deref(),
                split.as_deref(),
                *start,
                *limit,
                credential_env.as_deref(),
            )
            .await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&catalog)?);
            } else if catalog.task_api == OpenEnvTaskApiSupport::Unsupported {
                println!(
                    "{} {} does not expose the optional OpenEnv Task API",
                    style("○").yellow().bold(),
                    style(&catalog.environment_name).cyan().bold()
                );
                println!(
                    "  Seeded reset/options training remains fully available; no task adapter is required."
                );
            } else if catalog.selected_split.is_none() {
                println!(
                    "{} OpenEnv task catalog for {}",
                    style("✓").green().bold(),
                    style(&catalog.environment_name).cyan().bold()
                );
                if catalog.splits.is_empty() {
                    println!("  No splits are advertised.");
                }
                for split in &catalog.splits {
                    println!("  {} ({})", split.name, split.split_type);
                }
                println!("  Choose a page with --split NAME.");
            } else {
                let split = catalog.selected_split.as_deref().unwrap_or_default();
                println!(
                    "{} {} / {} tasks {}..{} of {}",
                    style("✓").green().bold(),
                    style(&catalog.environment_name).cyan().bold(),
                    style(split).cyan(),
                    catalog.start.unwrap_or_default(),
                    catalog.stop.unwrap_or_default(),
                    catalog.num_tasks.unwrap_or_default()
                );
                let first = catalog.start.unwrap_or_default();
                for (offset, task) in catalog.tasks.iter().enumerate() {
                    println!(
                        "  [{}] {}",
                        first.saturating_add(offset as u64),
                        serde_json::to_string(task)?
                    );
                }
            }
        }
        OpenEnvCommands::Rollout { rollout } => {
            let summary = run_openenv_rollout(openenv_rollout_options(rollout)).await?;
            print_openenv_summary(&summary, false)?;
        }
        OpenEnvCommands::Train {
            rollout,
            output_adapter,
            lora_rank,
            auto_load,
        } => {
            let summary = run_openenv_train(OpenEnvTrainOptions {
                rollout: openenv_rollout_options(rollout),
                output_adapter: output_adapter.clone(),
                lora_rank: *lora_rank,
                auto_load: *auto_load,
            })
            .await?;
            print_openenv_summary(&summary, true)?;
        }
        OpenEnvCommands::Runs { kiln_url, json } => {
            let value = openenv_control_plane_request(kiln_url, None, reqwest::Method::GET).await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else if let Some(runs) = value.get("runs").and_then(Value::as_array) {
                if runs.is_empty() {
                    println!("No server-owned OpenEnv runs are retained.");
                }
                for run in runs {
                    print_openenv_server_run(run);
                }
            } else {
                anyhow::bail!("Kiln returned an invalid OpenEnv run-list response");
            }
        }
        OpenEnvCommands::Start {
            request,
            idempotency_key,
            kiln_url,
            follow,
            json,
        } => {
            let mut request = read_openenv_run_request(request)?;
            apply_openenv_idempotency_key(&mut request, idempotency_key.as_deref())?;
            let started = start_openenv_control_plane_run(kiln_url, &request).await?;
            let run_id = validated_openenv_server_run_id(&started, None)?.to_string();
            watch_openenv_server_run(kiln_url, &run_id, *follow, *json, Some(started)).await?;
        }
        OpenEnvCommands::Status {
            run_id,
            kiln_url,
            follow,
            json,
        } => {
            validate_openenv_run_id(run_id)?;
            watch_openenv_server_run(kiln_url, run_id, *follow, *json, None).await?;
        }
        OpenEnvCommands::Cancel {
            run_id,
            kiln_url,
            json,
        } => {
            validate_openenv_run_id(run_id)?;
            let value =
                openenv_control_plane_request(kiln_url, Some(run_id), reqwest::Method::DELETE)
                    .await?;
            validated_openenv_server_run_id(&value, Some(run_id))?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                print_openenv_server_run(&value);
            }
        }
        OpenEnvCommands::Artifact {
            run_id,
            kind,
            output,
            kiln_url,
            force,
            json,
        } => {
            let receipt =
                download_openenv_server_artifact(kiln_url, run_id, kind, output, *force).await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            } else {
                println!(
                    "{} Downloaded and verified OpenEnv artifact {}",
                    style("✓").green().bold(),
                    style(&receipt.kind).cyan().bold()
                );
                println!("  Output:  {}", receipt.output_path);
                println!("  Bytes:   {}", receipt.bytes);
                println!("  SHA-256: {}", receipt.sha256);
            }
        }
        OpenEnvCommands::Verify {
            summary,
            dataset,
            replay,
            json,
        } => {
            let verified =
                verify_openenv_artifacts(summary, dataset.as_deref(), replay.as_deref())?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&verified.report)?);
            } else {
                println!(
                    "{} Verified {} OpenEnv episodes and {} exact environment exchanges",
                    style("✓").green().bold(),
                    verified.report.rollouts,
                    verified.report.environment_exchanges
                );
                println!("  Dataset: {}", verified.report.dataset_path);
                println!("  Replay:  {}", verified.report.replay_path);
                println!("  Receipt: {}", verified.report.summary_path);
                println!("  Dataset SHA-256: {}", verified.report.dataset_sha256);
                println!("  Replay SHA-256:  {}", verified.report.replay_sha256);
            }
        }
        OpenEnvCommands::Replay {
            summary,
            dataset,
            replay,
            credential_envs,
            concurrency,
            capacity_wait_seconds,
            json,
        } => {
            anyhow::ensure!(
                *concurrency > 0 && *concurrency <= MAX_OPENENV_CONCURRENCY,
                "OpenEnv replay concurrency must be in 1..={MAX_OPENENV_CONCURRENCY}"
            );
            anyhow::ensure!(
                *capacity_wait_seconds > 0
                    && *capacity_wait_seconds <= MAX_OPENENV_CAPACITY_WAIT_SECONDS,
                "OpenEnv capacity wait must be in 1..={MAX_OPENENV_CAPACITY_WAIT_SECONDS} seconds"
            );
            let verified =
                verify_openenv_artifacts(summary, dataset.as_deref(), replay.as_deref())?;
            let credential_envs =
                parse_cli_credential_envs(credential_envs, verified.replay.environments.len())?;
            let report = replay_openenv(
                &verified.replay,
                verified.report.replay_sha256,
                *concurrency,
                Duration::from_secs(*capacity_wait_seconds),
                &credential_envs,
            )
            .await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "{} Replayed {} OpenEnv episodes and {} exact environment exchanges",
                    style("✓").green().bold(),
                    report.rollouts,
                    report.environment_exchanges
                );
                println!("  Environments:     {}", report.environments);
                println!("  Replay SHA-256:   {}", report.replay_sha256);
                println!("  Capacity retries: {}", report.capacity_retries);
                if report.environment_prefix_only_rollouts > 0 {
                    println!(
                        "  Prefix-only:       {} rollouts ended at a model-side invalid action",
                        report.environment_prefix_only_rollouts
                    );
                }
            }
        }
    }
    Ok(())
}

const OPENENV_ARTIFACT_DOWNLOAD_SCHEMA_V1: &str = "kiln.openenv-artifact-download.v1";

#[derive(Debug, Serialize, PartialEq, Eq)]
struct OpenEnvArtifactDownloadReceipt {
    schema: &'static str,
    run_id: String,
    kind: String,
    source_url: String,
    output_path: String,
    sha256: String,
    bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct OpenEnvManifestArtifact {
    url: String,
    sha256: String,
    bytes: usize,
}

fn read_openenv_run_request(path: &Path) -> Result<Value> {
    let path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stat OpenEnv run request {}", path.display()))?;
    anyhow::ensure!(
        path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink(),
        "OpenEnv run request {} must be a regular non-symlink file",
        path.display()
    );
    anyhow::ensure!(
        path_metadata.len() <= MAX_OPENENV_RUN_REQUEST_BYTES as u64,
        "OpenEnv run request {} contains {} bytes; limit is {MAX_OPENENV_RUN_REQUEST_BYTES}",
        path.display(),
        path_metadata.len()
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
        .with_context(|| format!("open OpenEnv run request {}", path.display()))?;
    let opened_metadata = file
        .metadata()
        .with_context(|| format!("stat opened OpenEnv run request {}", path.display()))?;
    anyhow::ensure!(
        opened_metadata.file_type().is_file(),
        "OpenEnv run request {} must remain a regular file after open",
        path.display()
    );
    anyhow::ensure!(
        opened_metadata.len() <= MAX_OPENENV_RUN_REQUEST_BYTES as u64,
        "OpenEnv run request {} contains {} bytes; limit is {MAX_OPENENV_RUN_REQUEST_BYTES}",
        path.display(),
        opened_metadata.len()
    );
    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.take((MAX_OPENENV_RUN_REQUEST_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read OpenEnv run request {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= MAX_OPENENV_RUN_REQUEST_BYTES,
        "OpenEnv run request {} grew beyond the {MAX_OPENENV_RUN_REQUEST_BYTES} byte limit while reading",
        path.display()
    );
    let request: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("decode OpenEnv run request {} as JSON", path.display()))?;
    anyhow::ensure!(
        request.is_object(),
        "OpenEnv run request {} must contain one JSON object",
        path.display()
    );
    Ok(request)
}

fn openenv_control_plane_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHAT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build OpenEnv control-plane client")
}

fn apply_openenv_idempotency_key(request: &mut Value, key: Option<&str>) -> Result<()> {
    let Some(key) = key else {
        return Ok(());
    };
    crate::api::openenv::validate_openenv_idempotency_key(key)?;
    let object = request
        .as_object_mut()
        .context("OpenEnv run request must be a JSON object")?;
    if let Some(existing) = object.get("idempotency_key") {
        anyhow::ensure!(
            existing.as_str() == Some(key),
            "--idempotency-key {key:?} does not match request idempotency_key {}",
            serde_json::to_string(existing).unwrap_or_default()
        );
    } else {
        object.insert("idempotency_key".into(), Value::String(key.into()));
    }
    Ok(())
}

async fn start_openenv_control_plane_run(kiln_url: &str, request: &Value) -> Result<Value> {
    let response = openenv_control_plane_client()?
        .post(format!(
            "{}/v1/openenv/runs",
            kiln_url.trim_end_matches('/')
        ))
        .header("x-kiln-client", "openenv-cli")
        .json(request)
        .send()
        .await
        .context("start persisted OpenEnv workflow")?;
    let status = response.status();
    let body = read_kiln_json_bounded(response, "workflow creation").await?;
    anyhow::ensure!(
        status.is_success(),
        "Kiln OpenEnv control plane returned HTTP {status}: {}",
        serde_json::to_string(&body).unwrap_or_default()
    );
    validated_openenv_server_run_id(&body, None)?;
    Ok(body)
}

async fn watch_openenv_server_run(
    kiln_url: &str,
    run_id: &str,
    follow: bool,
    json: bool,
    initial: Option<Value>,
) -> Result<Value> {
    let mut previous_fingerprint = None;
    let mut next = initial;
    loop {
        let value = match next.take() {
            Some(value) => value,
            None => {
                openenv_control_plane_request(kiln_url, Some(run_id), reqwest::Method::GET).await?
            }
        };
        validated_openenv_server_run_id(&value, Some(run_id))?;
        value
            .get("state")
            .and_then(Value::as_str)
            .context("Kiln OpenEnv status omitted state")?;
        let fingerprint = openenv_server_run_fingerprint(&value);
        if !json && previous_fingerprint.as_deref() != Some(fingerprint.as_str()) {
            print_openenv_server_run(&value);
        }
        if !follow || openenv_server_run_terminal(&value) {
            if json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else if previous_fingerprint.as_deref() == Some(fingerprint.as_str()) {
                print_openenv_server_run(&value);
            }
            return Ok(value);
        }
        previous_fingerprint = Some(fingerprint);
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn validated_openenv_server_run_id<'a>(run: &'a Value, expected: Option<&str>) -> Result<&'a str> {
    let run_id = run
        .get("run_id")
        .and_then(Value::as_str)
        .context("Kiln OpenEnv status omitted run_id")?;
    validate_openenv_run_id(run_id)?;
    if let Some(expected) = expected {
        anyhow::ensure!(
            run_id == expected,
            "Kiln OpenEnv status returned run ID {run_id:?}; expected {expected:?}"
        );
    }
    Ok(run_id)
}

fn manifest_artifact(run: &Value, run_id: &str, kind: &str) -> Result<OpenEnvManifestArtifact> {
    let artifact = run
        .get("artifacts")
        .and_then(Value::as_array)
        .context("Kiln OpenEnv status omitted the artifacts array")?
        .iter()
        .find(|artifact| artifact.get("kind").and_then(Value::as_str) == Some(kind))
        .with_context(|| {
            format!("OpenEnv run {run_id} has no manifest-declared artifact kind {kind:?}")
        })?;
    let url = artifact
        .get("url")
        .and_then(Value::as_str)
        .context("OpenEnv artifact manifest omitted url")?;
    let expected_url = format!("/v1/openenv/runs/{run_id}/artifacts/{kind}");
    anyhow::ensure!(
        url == expected_url,
        "OpenEnv artifact manifest URL {url:?} does not match its run and kind"
    );
    let sha256 = artifact
        .get("sha256")
        .and_then(Value::as_str)
        .context("OpenEnv artifact manifest omitted sha256")?;
    validate_openenv_sha256("artifact manifest", sha256)?;
    let bytes_u64 = artifact
        .get("bytes")
        .and_then(Value::as_u64)
        .context("OpenEnv artifact manifest omitted non-negative integer bytes")?;
    let bytes = usize::try_from(bytes_u64).context("OpenEnv artifact byte count exceeds usize")?;
    anyhow::ensure!(
        bytes <= MAX_OPENENV_ARTIFACT_BYTES,
        "OpenEnv artifact manifest declares {bytes} bytes; limit is {MAX_OPENENV_ARTIFACT_BYTES}"
    );
    Ok(OpenEnvManifestArtifact {
        url: url.to_string(),
        sha256: sha256.to_string(),
        bytes,
    })
}

fn validate_openenv_sha256(label: &str, value: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == "sha256:".len() + 64
            && value.starts_with("sha256:")
            && value["sha256:".len()..]
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "OpenEnv {label} SHA-256 is malformed"
    );
    Ok(())
}

async fn download_openenv_server_artifact(
    kiln_url: &str,
    run_id: &str,
    kind: &str,
    output: &Path,
    force: bool,
) -> Result<OpenEnvArtifactDownloadReceipt> {
    validate_openenv_run_id(run_id)?;
    anyhow::ensure!(!kind.is_empty(), "OpenEnv artifact kind cannot be empty");
    let run = openenv_control_plane_request(kiln_url, Some(run_id), reqwest::Method::GET).await?;
    validated_openenv_server_run_id(&run, Some(run_id))?;
    let artifact = manifest_artifact(&run, run_id, kind)?;
    let download_url = format!("{}{}", kiln_url.trim_end_matches('/'), artifact.url);
    let mut response = openenv_control_plane_client()?
        .get(&download_url)
        .header("x-kiln-client", "openenv-cli")
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
        .send()
        .await
        .context("download manifest-declared OpenEnv artifact")?;
    let status = response.status();
    if !status.is_success() {
        let body = read_kiln_json_bounded(response, "artifact download error").await?;
        anyhow::bail!(
            "Kiln OpenEnv artifact download returned HTTP {status}: {}",
            serde_json::to_string(&body).unwrap_or_default()
        );
    }
    let headers = response.headers();
    let content_length = headers
        .get(reqwest::header::CONTENT_LENGTH)
        .context("Kiln OpenEnv artifact response omitted Content-Length")?
        .to_str()
        .context("Kiln OpenEnv artifact Content-Length was not ASCII")?
        .parse::<usize>()
        .context("Kiln OpenEnv artifact Content-Length was not an integer")?;
    anyhow::ensure!(
        content_length == artifact.bytes,
        "Kiln OpenEnv artifact Content-Length {content_length} does not match manifest {}",
        artifact.bytes
    );
    let etag = headers
        .get(reqwest::header::ETAG)
        .context("Kiln OpenEnv artifact response omitted ETag")?
        .to_str()
        .context("Kiln OpenEnv artifact ETag was not ASCII")?;
    anyhow::ensure!(
        etag == format!("\"{}\"", artifact.sha256),
        "Kiln OpenEnv artifact ETag {etag:?} does not match manifest digest"
    );
    anyhow::ensure!(
        headers
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok())
            == Some("private, no-store"),
        "Kiln OpenEnv artifact response omitted the private, no-store cache policy"
    );
    anyhow::ensure!(
        headers
            .get(reqwest::header::X_CONTENT_TYPE_OPTIONS)
            .and_then(|value| value.to_str().ok())
            == Some("nosniff"),
        "Kiln OpenEnv artifact response omitted X-Content-Type-Options: nosniff"
    );

    let parent = output_parent(output)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create staged OpenEnv artifact beside {}", output.display()))?;
    let mut hasher = Sha256::new();
    let mut bytes = 0usize;
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read manifest-declared OpenEnv artifact")?
    {
        bytes = bytes
            .checked_add(chunk.len())
            .context("OpenEnv artifact byte count overflow")?;
        anyhow::ensure!(
            bytes <= artifact.bytes,
            "Kiln OpenEnv artifact response exceeded its manifest byte count {}",
            artifact.bytes
        );
        hasher.update(&chunk);
        staged
            .as_file_mut()
            .write_all(&chunk)
            .with_context(|| format!("write staged OpenEnv artifact for {}", output.display()))?;
    }
    anyhow::ensure!(
        bytes == artifact.bytes,
        "Kiln OpenEnv artifact response ended at {bytes} bytes; manifest declares {}",
        artifact.bytes
    );
    let sha256 = format_digest(hasher.finalize().as_slice());
    anyhow::ensure!(
        sha256 == artifact.sha256,
        "Kiln OpenEnv artifact response digest {sha256} does not match manifest {}",
        artifact.sha256
    );
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv artifact for {}", output.display()))?;
    if force {
        staged
            .persist(output)
            .map_err(|error| error.error)
            .with_context(|| format!("publish OpenEnv artifact {}", output.display()))?;
    } else {
        staged
            .persist_noclobber(output)
            .map_err(|error| error.error)
            .with_context(|| {
                format!(
                    "publish OpenEnv artifact {} without replacement; use --force to replace it deliberately",
                    output.display()
                )
            })?;
    }
    Ok(OpenEnvArtifactDownloadReceipt {
        schema: OPENENV_ARTIFACT_DOWNLOAD_SCHEMA_V1,
        run_id: run_id.to_string(),
        kind: kind.to_string(),
        source_url: artifact.url,
        output_path: output.display().to_string(),
        sha256,
        bytes,
    })
}

fn validate_openenv_run_id(run_id: &str) -> Result<()> {
    uuid::Uuid::parse_str(run_id)
        .with_context(|| format!("OpenEnv run ID {run_id:?} is not a UUID"))?;
    Ok(())
}

async fn openenv_control_plane_request(
    kiln_url: &str,
    run_id: Option<&str>,
    method: reqwest::Method,
) -> Result<Value> {
    let client = openenv_control_plane_client()?;
    let suffix = run_id.map_or_else(
        || "/v1/openenv/runs".to_string(),
        |run_id| format!("/v1/openenv/runs/{run_id}"),
    );
    let response = client
        .request(
            method,
            format!("{}{suffix}", kiln_url.trim_end_matches('/')),
        )
        .header("x-kiln-client", "openenv-cli")
        .send()
        .await
        .context("send OpenEnv control-plane request")?;
    let status = response.status();
    let body = read_kiln_json_bounded(response, "OpenEnv workflow status").await?;
    anyhow::ensure!(
        status.is_success(),
        "Kiln OpenEnv control plane returned HTTP {status}: {}",
        serde_json::to_string(&body).unwrap_or_default()
    );
    Ok(body)
}

fn openenv_server_run_terminal(run: &Value) -> bool {
    let state = run.get("state").and_then(Value::as_str).unwrap_or_default();
    matches!(
        state,
        "rollout_ready" | "completed" | "failed" | "cancelled"
    ) || (run.get("schema").and_then(Value::as_str) == Some("kiln.openenv-run.v1")
        && state == "training_queued")
}

fn openenv_server_run_fingerprint(run: &Value) -> String {
    let training_progress = run
        .pointer("/training/progress")
        .and_then(Value::as_f64)
        .map(|value| (value * 100.0).floor() as u64)
        .unwrap_or_default();
    let static_eval_done = run
        .get("post_evaluations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|evaluation| evaluation.get("examples_completed").and_then(Value::as_u64))
        .sum::<u64>();
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        run.get("state").and_then(Value::as_str).unwrap_or_default(),
        run.pointer("/training/state")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        training_progress,
        static_eval_done,
        run.pointer("/environment_evaluation/state")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        run.pointer("/environment_evaluation/progress/groups_completed")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        run.pointer("/admission/queue_position")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        run.pointer("/training/training_data/openenv/group_plan_sha256")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        run.get("artifacts")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or_default(),
        run.pointer("/failure/code")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        run.pointer("/failure/stage")
            .and_then(Value::as_str)
            .unwrap_or_default()
    )
}

fn print_openenv_server_run(run: &Value) {
    let run_id = run
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let state = run
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let kind = run.get("kind").and_then(Value::as_str).unwrap_or("unknown");
    let progress = run.get("progress").unwrap_or(&Value::Null);
    let episodes_done = progress
        .get("rollouts_completed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let episodes_total = progress
        .get("rollouts_total")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    println!(
        "{} {} {} · {episodes_done}/{episodes_total} episodes",
        style(run_id).cyan(),
        style(kind).bold(),
        state.replace('_', " ")
    );
    if let Some(admission) = run.get("admission") {
        if let Some(position) = admission.get("queue_position").and_then(Value::as_u64) {
            let slots = admission
                .get("max_active_runs")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            println!("  Admission: FIFO queue position {position} · {slots} active slots");
        } else if let Some(wait_ms) = admission.get("queue_wait_ms").and_then(Value::as_u64) {
            println!("  Admission: execution started after {wait_ms} ms");
        }
    }
    if let Some(key) = run
        .pointer("/request/idempotency_key")
        .and_then(Value::as_str)
    {
        println!("  Submission: retry key {key}");
    }
    if let Some(stats) = run.get("rollout_stats") {
        print_openenv_rollout_stats("Rollouts", stats);
    }
    if let Some(contract) = run.get("training_contract") {
        let config = contract.get("effective_config").unwrap_or(&Value::Null);
        let optimizer = config
            .pointer("/optimizer/kind")
            .and_then(Value::as_str)
            .unwrap_or("muon");
        let rank = config.get("lora_rank").and_then(Value::as_u64).unwrap_or(8);
        let output = config
            .get("output_name")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let auto_load = config
            .get("auto_load")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        println!("  Contract: {optimizer} · rank {rank} · output {output} · auto-load {auto_load}");
        if let Some(suite) = contract.pointer("/post_eval/suite").and_then(Value::as_str) {
            println!("  Contract eval: {suite}");
        }
    }
    if let Some(training) = run.get("training") {
        let training_state = training
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let training_progress = training
            .get("progress")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let loss = training
            .get("current_loss")
            .and_then(Value::as_f64)
            .map(|loss| format!(" · loss {loss:.6}"))
            .unwrap_or_default();
        println!(
            "  Trainer: {training_state} · {:.1}%{loss}",
            training_progress * 100.0
        );
        if let Some(lineage) = training.pointer("/training_data/openenv") {
            let names = lineage
                .get("environments")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|environment| environment.get("environment_name")?.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let groups = lineage
                .get("groups")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let rollouts = lineage
                .get("rollouts")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let seed_min = lineage
                .get("seed_min")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let seed_max = lineage
                .get("seed_max")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            println!(
                "  Corpus: OpenEnv · {} · {groups} groups · {rollouts} rollouts · seeds {seed_min}–{seed_max}",
                if names.is_empty() {
                    "compatible environment"
                } else {
                    &names
                }
            );
        }
        if let Some(outcome) = training.get("gate_outcome").and_then(Value::as_str) {
            println!("  Promotion gate: {outcome}");
        }
    }
    if let Some(evaluations) = run.get("post_evaluations").and_then(Value::as_array) {
        for evaluation in evaluations {
            println!(
                "  Eval {}: {} · {}/{} examples{}",
                evaluation
                    .get("suite_name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                evaluation
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                evaluation
                    .get("examples_completed")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                evaluation
                    .get("examples_total")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                evaluation
                    .get("headline_accuracy")
                    .and_then(Value::as_f64)
                    .map(|accuracy| format!(" · {:.1}% accuracy", accuracy * 100.0))
                    .unwrap_or_default()
            );
        }
    }
    if let Some(evaluation) = run.get("environment_evaluation") {
        let eval_state = evaluation
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let progress = evaluation.get("progress").unwrap_or(&Value::Null);
        let groups_done = progress
            .get("groups_completed")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let groups_total = progress
            .get("groups_total")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        println!(
            "  Environment eval: {eval_state} · {groups_done}/{groups_total} held-out seed groups"
        );
        if let Some(stats) = evaluation.get("baseline_stats") {
            print_openenv_rollout_stats("Baseline rollouts", stats);
        }
        if let Some(stats) = evaluation.get("candidate_stats") {
            print_openenv_rollout_stats("Candidate rollouts", stats);
        }
        if let Some(evidence) = evaluation.get("evidence") {
            println!(
                "    return {:.6} → {:.6} ({:+.6}) · {} seed groups improved / {} regressed / {} tied · exact p={:.6}",
                evidence
                    .get("baseline_mean_return")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                evidence
                    .get("candidate_mean_return")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                evidence
                    .get("mean_return_improvement")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                evidence
                    .get("improved_groups")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                evidence
                    .get("regressed_groups")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                evidence
                    .get("tied_groups")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                evidence
                    .get("exact_sign_test_p_value")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0),
            );
        }
        if let Some(outcome) = evaluation.get("outcome").and_then(Value::as_str) {
            println!("    Environment gate: {outcome}");
        }
    }
    if let Some(failure) = run.get("failure") {
        let code = failure
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("internal_error");
        let stage = failure
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("orchestration");
        let retryable = failure
            .get("retryable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        println!(
            "  Failure: {code} at {} · {}",
            stage.replace('_', " "),
            if retryable {
                "retryable"
            } else {
                "not retryable"
            }
        );
        if let Some(protocol_code) = failure.get("protocol_code").and_then(Value::as_str) {
            println!("    OpenEnv protocol: {protocol_code}");
        }
        if let Some(http_status) = failure.get("http_status").and_then(Value::as_u64) {
            println!("    OpenEnv HTTP: {http_status}");
        }
        if let Some(message) = failure.get("message").and_then(Value::as_str) {
            println!("    Detail: {message}");
        }
        if let Some(hint) = failure.get("hint").and_then(Value::as_str) {
            println!("    Next: {hint}");
        }
    } else if let Some(error) = run.get("error").and_then(Value::as_str) {
        println!("  Error: {error}");
    }
    if let Some(artifacts) = run.get("artifacts").and_then(Value::as_array) {
        for artifact in artifacts {
            println!(
                "  Artifact: {} · {} bytes · {}",
                artifact
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                artifact
                    .get("bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                artifact
                    .get("sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("unbound")
            );
        }
    }
}

fn print_openenv_rollout_stats(label: &str, stats: &Value) {
    let mean = stats
        .get("mean_episode_return")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let range = stats
        .get("min_episode_return")
        .and_then(Value::as_f64)
        .zip(stats.get("max_episode_return").and_then(Value::as_f64))
        .map(|(min, max)| format!(" · range {min:.6}…{max:.6}"))
        .unwrap_or_default();
    println!(
        "  {label}: mean return {mean:.6}{range} · {} done · {} max steps · {} invalid actions · {} protocol errors",
        stats.get("done_count").and_then(Value::as_u64).unwrap_or(0),
        stats
            .get("max_steps_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        stats
            .get("invalid_model_action_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        stats
            .get("protocol_error_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    );
    println!(
        "  {label} work: {} environment steps · {} model tokens · {:.3} ms mean model latency · {} recoverable errors · {} capacity retries",
        stats
            .get("total_environment_steps")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        stats
            .get("total_model_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        stats
            .get("mean_model_latency_ms")
            .and_then(Value::as_f64)
            .unwrap_or_default(),
        stats
            .get("recoverable_protocol_error_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        stats
            .get("capacity_retry_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    );
}

fn openenv_rollout_options(args: &OpenEnvRolloutArgs) -> OpenEnvRolloutOptions {
    OpenEnvRolloutOptions {
        kiln_url: args.kiln_url.clone(),
        environment_urls: args.environment_urls.clone(),
        credential_envs: parse_cli_credential_envs_unchecked(&args.credential_envs),
        adapter: args.adapter.clone(),
        groups: args.groups,
        group_size: args.group_size,
        seed_start: args.seed_start,
        reset_options: args.reset_options.clone(),
        reset_options_value: None,
        environment_reset_options: args
            .environment_reset_options
            .iter()
            .map(|path| (path.as_os_str() != "-").then(|| path.clone()))
            .collect(),
        environment_reset_options_values: Vec::new(),
        max_steps: args.max_steps,
        concurrency: args.concurrency,
        max_action_tokens: args.max_action_tokens,
        thinking_budget_tokens: args.thinking_budget_tokens,
        temperature: args.temperature,
        thinking: args.thinking,
        protocol_error_reward: args.protocol_error_reward,
        max_recoverable_errors: args.max_recoverable_errors,
        capacity_wait_seconds: args.capacity_wait_seconds,
        output: args.output.clone(),
        replay_output: args.replay_output.clone(),
        summary_output: args.summary_output.clone(),
    }
}

fn print_openenv_summary(summary: &OpenEnvRolloutSummary, submitted_training: bool) -> Result<()> {
    println!(
        "{} Collected {} OpenEnv episodes in {} seed-matched groups",
        style("✓").green().bold(),
        summary.rollout_count,
        summary.groups
    );
    println!("  Dataset:    {}", summary.output_path);
    println!("  Replay:     {}", summary.replay_output_path);
    println!("  Receipt:    {}", summary.summary_output_path);
    println!("  Dataset SHA-256: {}", summary.dataset_sha256);
    println!("  Replay SHA-256:  {}", summary.replay_sha256);
    println!(
        "  Return:     mean {:.6}, min {}, max {}",
        summary.stats.mean_episode_return,
        summary
            .stats
            .min_episode_return
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        summary
            .stats
            .max_episode_return
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  Outcomes:   {} done, {} max-steps, {} invalid actions, {} protocol errors",
        summary.stats.done_count,
        summary.stats.max_steps_count,
        summary.stats.invalid_model_action_count,
        summary.stats.protocol_error_count
    );
    if summary.stats.recoverable_protocol_error_count > 0 || summary.stats.capacity_retry_count > 0
    {
        println!(
            "  Recovery:   {} recoverable protocol errors, {} capacity retries",
            summary.stats.recoverable_protocol_error_count, summary.stats.capacity_retry_count
        );
    }
    if let Some(contract) = summary.training_contract.as_ref() {
        let config = &contract.effective_config;
        println!(
            "  Training:    {:?} · rank {} · output {} · auto-load {}{}",
            config.optimizer.kind(),
            config.lora_rank,
            config.output_name.as_deref().unwrap_or("n/a"),
            config.auto_load,
            contract
                .post_eval
                .as_ref()
                .map(|post_eval| format!(" · post-eval {}", post_eval.suite))
                .unwrap_or_default()
        );
    }
    if let Some(selection) = summary.training_selection.as_ref() {
        println!(
            "  GRPO input:  {} groups · {} rollouts · {} no-output rollouts discarded",
            selection.groups_submitted,
            selection.rollouts_submitted,
            selection.no_output_rollouts_discarded
        );
        for warning in &selection.warnings {
            println!("{} {warning}", style("!").yellow().bold());
        }
    }
    if submitted_training {
        if let Some(submission) = summary.training_submission.as_ref() {
            println!(
                "{} GRPO training submitted{}",
                style("✓").green().bold(),
                submission
                    .get("job_id")
                    .and_then(Value::as_str)
                    .map(|job_id| format!(" as {job_id}"))
                    .unwrap_or_default()
            );
        } else {
            println!(
                "{} GRPO submission skipped because the collected batch had no trainable reward group",
                style("!").yellow().bold()
            );
        }
    }
    Ok(())
}

pub async fn inspect_openenv(
    environment_url: &str,
    credential_env: Option<&str>,
) -> Result<OpenEnvInspection> {
    if let Some(name) = credential_env {
        validate_credential_envs(&[Some(name.to_owned())], 1)?;
    }
    let client = openenv_client(environment_url, credential_env)?;
    client
        .inspect()
        .await
        .with_context(|| format!("inspect OpenEnv server {}", client.base_url()))
}

pub async fn inspect_openenv_tasks(
    environment_url: &str,
    environment_name: Option<&str>,
    split: Option<&str>,
    start: u64,
    limit: usize,
    credential_env: Option<&str>,
) -> Result<OpenEnvTaskCatalog> {
    anyhow::ensure!(
        (1..=MAX_OPENENV_TASK_PAGE_SIZE).contains(&limit),
        "OpenEnv task page limit must be in 1..={}, got {limit}",
        MAX_OPENENV_TASK_PAGE_SIZE
    );
    if let Some(name) = credential_env {
        validate_credential_envs(&[Some(name.to_owned())], 1)?;
    }
    let client = openenv_client(environment_url, credential_env)?;
    client
        .task_catalog(environment_name, split, start, limit)
        .await
        .with_context(|| format!("inspect OpenEnv Task API at {}", client.base_url()))
}

pub async fn run_openenv_rollout(options: OpenEnvRolloutOptions) -> Result<OpenEnvRolloutSummary> {
    let collection = collect_openenv_rollouts(&options).await?;
    write_openenv_outputs(
        &options,
        &collection.groups,
        &collection.replay,
        &collection.summary,
    )?;
    Ok(collection.summary)
}

pub async fn run_openenv_train(options: OpenEnvTrainOptions) -> Result<OpenEnvRolloutSummary> {
    validate_output_adapter(&options.output_adapter)?;
    validate_options(&options.rollout)?;
    let preflight = preflight_openenv_training(&options).await?;
    let mut collection = collect_openenv_rollouts(&options.rollout).await?;
    anyhow::ensure!(
        collection.summary.behavior_policy == preflight.behavior_policy,
        "Kiln behavior policy changed after OpenEnv training preflight and before collection completed"
    );
    collection.summary.training_contract = Some(preflight.training_contract());
    let training_groups =
        select_openenv_training_groups(&collection.groups, &mut collection.summary);
    write_openenv_outputs(
        &options.rollout,
        &collection.groups,
        &collection.replay,
        &collection.summary,
    )?;
    if training_groups.is_empty() {
        tracing::warn!(
            groups_collected = collection.groups.len(),
            "OpenEnv batch has no trainable reward group; skipping GRPO submission"
        );
        return Ok(collection.summary);
    }
    let submission = submit_openenv_training(
        &options.rollout.kiln_url,
        &training_groups,
        &preflight.effective_config,
        preflight.post_eval.as_ref(),
    )
    .await?;
    collection.summary.training_submission = Some(submission);
    write_summary_atomic(&options.rollout.summary_output, &collection.summary)?;
    Ok(collection.summary)
}

fn rollout_has_no_model_output(rollout: &ScoredRollout) -> bool {
    rollout.trajectory.last().is_some_and(|segment| {
        serde_json::from_str::<Value>(&segment.content)
            .ok()
            .and_then(|value| {
                value
                    .pointer("/openenv_harness_error/code")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .as_deref()
            == Some("MODEL_ACTION_NO_OUTPUT")
    })
}

/// Produce the exact optimizer input from a complete diagnostic collection.
/// No-output completions are discarded individually. A group is submitted
/// only when at least two remaining rewards differ, because otherwise its
/// group-normalized advantage is uniformly zero.
pub(crate) fn select_openenv_training_groups(
    groups: &[AgenticGroup],
    summary: &mut OpenEnvRolloutSummary,
) -> Vec<AgenticGroup> {
    let (selected, selection) = openenv_training_group_selection(groups);
    summary.training_selection = Some(selection);
    selected
}

fn openenv_training_group_selection(
    groups: &[AgenticGroup],
) -> (Vec<AgenticGroup>, OpenEnvTrainingSelection) {
    let mut selected = Vec::with_capacity(groups.len());
    let mut selection = OpenEnvTrainingSelection::default();

    for (group_index, group) in groups.iter().enumerate() {
        let mut completions = Vec::with_capacity(group.completions.len());
        let mut group_no_output = 0usize;
        for rollout in &group.completions {
            if rollout_has_no_model_output(rollout) {
                group_no_output = group_no_output.saturating_add(1);
                selection.no_output_rollouts_discarded =
                    selection.no_output_rollouts_discarded.saturating_add(1);
            } else {
                completions.push(rollout.clone());
            }
        }

        let warning = if completions.is_empty() {
            selection.no_usable_output_groups_skipped =
                selection.no_usable_output_groups_skipped.saturating_add(1);
            Some(format!(
                "OpenEnv group {group_index} produced no final model actions; discarded every completion and skipped training for the group"
            ))
        } else {
            if group_no_output > 0 {
                selection.warnings.push(format!(
                    "OpenEnv group {group_index} discarded {group_no_output} completion(s) that produced no final model action before GRPO"
                ));
            }
            let first_reward = completions[0].reward;
            if completions
                .iter()
                .skip(1)
                .all(|rollout| rollout.reward == first_reward)
            {
                selection.no_reward_gradient_groups_skipped = selection
                    .no_reward_gradient_groups_skipped
                    .saturating_add(1);
                Some(format!(
                    "OpenEnv group {group_index} has no reward variation across its {} usable completion(s); skipped zero-gradient GRPO work",
                    completions.len()
                ))
            } else {
                None
            }
        };

        if let Some(warning) = warning {
            tracing::warn!(group_index, warning = %warning, "skipping OpenEnv GRPO group");
            selection.warnings.push(warning);
            continue;
        }

        selection.rollouts_submitted = selection
            .rollouts_submitted
            .saturating_add(completions.len());
        selected.push(AgenticGroup {
            messages: group.messages.clone(),
            completions,
        });
    }

    selection.groups_submitted = selected.len();
    (selected, selection)
}

pub async fn collect_openenv_rollouts(
    options: &OpenEnvRolloutOptions,
) -> Result<OpenEnvCollection> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHAT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build Kiln chat client")?;
    let policy = OpenEnvPolicyTransport::Http {
        client,
        kiln_url: options.kiln_url.clone(),
    };
    collect_openenv_rollouts_with_policy(options, &policy, &OpenEnvCollectionControl::default())
        .await
}

pub(crate) async fn collect_openenv_rollouts_with_policy(
    options: &OpenEnvRolloutOptions,
    policy: &OpenEnvPolicyTransport,
    control: &OpenEnvCollectionControl,
) -> Result<OpenEnvCollection> {
    validate_options(options)?;
    control.ensure_active()?;
    let reset_plan = read_reset_plan(options)?;
    let reset_plan_sha256 = sha256_json(&reset_plan)?;
    let retained_budget = OpenEnvRetainedByteBudget::new(MAX_OPENENV_COLLECTION_RETAINED_BYTES);
    let reset_plan_bytes = serialized_len(&reset_plan, "reset plan")?;
    retained_budget.charge(reset_plan_bytes, "the ordered reset plan")?;
    let adapter = parse_adapter_selection(&options.adapter);
    control.publish(OpenEnvCollectionProgress {
        stage: OpenEnvCollectionStage::Discovering,
        groups_completed: 0,
        groups_total: options.groups,
        rollouts_completed: 0,
    });

    let inspections = stream::iter(options.environment_urls.iter().cloned())
        .zip(stream::iter(resolved_credential_envs(options)))
        .map(|(url, credential_env)| async move {
            let client = openenv_client(&url, credential_env.as_deref())?;
            let inspection = client
                .inspect()
                .await
                .with_context(|| format!("inspect OpenEnv server {}", client.base_url()))?;
            let action_validator = inspection.action_validator().map_err(|source| {
                OpenEnvClientError::InvalidActionSchema {
                    endpoint: format!("{}/schema", client.base_url()),
                    source,
                }
            })?;
            Ok::<_, anyhow::Error>((client, inspection, action_validator))
        })
        .buffered(options.environment_urls.len())
        .try_collect::<Vec<_>>()
        .await?;
    for (_, inspection, _) in &inspections {
        retained_budget.charge(
            serialized_len(inspection, "environment inspection")?,
            "discovered environment inspection",
        )?;
    }
    control.ensure_active()?;
    control.publish_discovered(
        inspections
            .iter()
            .map(|(_, inspection, _)| inspection.identity.clone())
            .collect(),
    );
    control.publish(OpenEnvCollectionProgress {
        stage: OpenEnvCollectionStage::Collecting,
        groups_completed: 0,
        groups_total: options.groups,
        rollouts_completed: 0,
    });

    let mut groups = Vec::with_capacity(options.groups);
    let mut replay_groups = Vec::with_capacity(options.groups);
    let mut records = Vec::with_capacity(options.groups.saturating_mul(options.group_size));
    let mut dataset_bytes = 0usize;

    for group_index in 0..options.groups {
        control.ensure_active()?;
        let environment_index = group_index % inspections.len();
        let (environment, inspection, action_validator) = &inspections[environment_index];
        let seed = options
            .seed_start
            .checked_add(group_index as u64)
            .context("OpenEnv reset seed range overflow")?;
        if seed > i64::MAX as u64 {
            anyhow::bail!(
                "OpenEnv reset seed {seed} exceeds the protocol's portable signed-integer range"
            );
        }
        let reset = reset_payload(&reset_plan[environment_index], seed)?;

        let mut candidates = stream::iter(0..options.group_size)
            .map(|candidate_index| {
                run_candidate_episode(
                    policy,
                    environment,
                    inspection,
                    action_validator,
                    &adapter.request_value,
                    &adapter.label,
                    &reset,
                    seed,
                    group_index,
                    candidate_index,
                    options,
                    control,
                    &retained_budget,
                )
            })
            .buffer_unordered(options.concurrency.min(options.group_size))
            .try_collect::<Vec<_>>()
            .await?;
        candidates.sort_by_key(|candidate| candidate.candidate_index);

        let first_candidate = candidates
            .first()
            .context("OpenEnv group unexpectedly produced no candidate rollouts")?;
        debug_assert_eq!(first_candidate.candidate_index, 0);
        for candidate in &candidates {
            anyhow::ensure!(
                candidate.messages_sha256 == first_candidate.messages_sha256,
                "OpenEnv reset produced different initial observations within group {group_index}; reset(seed) must be deterministic for group-relative training"
            );
            anyhow::ensure!(
                candidate.reset_observation_sha256 == first_candidate.reset_observation_sha256,
                "OpenEnv reset produced different wire observations within group {group_index}; reset(seed) must be deterministic for exact replay"
            );
        }
        let candidate_retained_bytes = candidates.iter().try_fold(0usize, |total, candidate| {
            total
                .checked_add(candidate.retained_bytes)
                .context("OpenEnv group retained-byte count overflow")
        })?;
        let mut shared_messages = None;
        let mut reset_observation = None;
        let mut completions = Vec::with_capacity(candidates.len());
        let mut replay_candidates = Vec::with_capacity(candidates.len());
        let mut group_records = Vec::with_capacity(candidates.len());
        for (position, candidate) in candidates.into_iter().enumerate() {
            let CandidateRollout {
                candidate_index: _,
                messages,
                reset_observation: candidate_reset,
                messages_sha256: _,
                reset_observation_sha256: _,
                rollout,
                replay,
                record,
                retained_bytes: _,
            } = candidate;
            if position == 0 {
                shared_messages = messages;
                reset_observation = candidate_reset;
            } else {
                debug_assert!(messages.is_none());
                debug_assert!(candidate_reset.is_none());
            }
            completions.push(rollout);
            replay_candidates.push(replay);
            group_records.push(record);
        }
        let replay_group = OpenEnvReplayGroup {
            group_index,
            environment_index,
            seed,
            reset_payload: reset,
            reset_observation: reset_observation
                .context("OpenEnv group lost its reset observation")?,
            candidates: replay_candidates,
        };
        let group = AgenticGroup {
            messages: shared_messages.context("OpenEnv group lost its shared prompt")?,
            completions,
        };
        let group_bytes = serialized_len(&group, "GRPO group")?;
        let replay_group_bytes = serialized_len(&replay_group, "replay group")?;
        let record_bytes = serialized_len(&group_records, "rollout records")?;
        let compacted_bytes = group_bytes
            .checked_add(replay_group_bytes)
            .and_then(|bytes| bytes.checked_add(record_bytes))
            .context("OpenEnv compacted group byte count overflow")?;
        retained_budget.replace(
            candidate_retained_bytes,
            compacted_bytes,
            "the compacted seed group",
        )?;
        dataset_bytes = dataset_bytes
            .checked_add(group_bytes.saturating_add(1))
            .context("OpenEnv dataset byte count overflow")?;
        anyhow::ensure!(
            dataset_bytes <= MAX_OPENENV_DATASET_BYTES,
            "OpenEnv rollout dataset exceeded the {} byte in-memory/inline limit; reduce groups, group size, max steps, or environment observation size",
            MAX_OPENENV_DATASET_BYTES
        );
        records.extend(group_records);
        replay_groups.push(replay_group);
        groups.push(group);
        control.publish(OpenEnvCollectionProgress {
            stage: OpenEnvCollectionStage::Collecting,
            groups_completed: groups.len(),
            groups_total: options.groups,
            rollouts_completed: records.len(),
        });
    }

    // A collection can outlive an environment deployment. Re-read every
    // stable discovery surface after the final episode and before deriving or
    // publishing any artifact, so later sessions cannot be attributed to the
    // identity captured before a mid-run redeploy.
    control.publish(OpenEnvCollectionProgress {
        stage: OpenEnvCollectionStage::Revalidating,
        groups_completed: groups.len(),
        groups_total: options.groups,
        rollouts_completed: records.len(),
    });
    control.ensure_active()?;
    stream::iter(inspections.iter())
        .map(revalidate_openenv_identity)
        .buffered(inspections.len())
        .try_collect::<Vec<_>>()
        .await?;
    control.ensure_active()?;

    drop(reset_plan);
    retained_budget.replace(reset_plan_bytes, 0, "the consumed reset plan")?;

    let dataset_sha256 = sha256_jsonl(&groups)?;
    let replay_environments = inspections
        .into_iter()
        .map(|(_, inspection, _)| inspection)
        .collect::<Vec<_>>();
    retained_budget.charge(
        serialized_len(&replay_environments, "summary environment inspections")?,
        "the summary copy of environment inspections",
    )?;
    let summary_environments = replay_environments.clone();
    let replay = OpenEnvReplayManifest {
        schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
        client_profile: kiln_openenv::OPENENV_CLIENT_PROFILE.to_string(),
        dataset_sha256: dataset_sha256.clone(),
        protocol_error_reward: options.protocol_error_reward,
        max_steps: options.max_steps,
        environments: replay_environments,
        groups: replay_groups,
    };
    let replay_encoded = encode_replay(&replay)?;
    let replay_sha256 = replay_sha256(&replay_encoded);
    let replay_bytes = replay_encoded.len();
    drop(replay_encoded);
    let stats = summarize_rollouts(&records);
    let behavior_policy = kiln_train::openenv_training_data_provenance(&groups)
        .map_err(anyhow::Error::msg)
        .context("validate completed OpenEnv corpus policy identity")?
        .and_then(|provenance| provenance.behavior_policy)
        .context("current OpenEnv collection omitted its behavior-policy identity")?;
    let summary = OpenEnvRolloutSummary {
        schema: "kiln.openenv-rollout-summary.v5".to_string(),
        kiln_url: options.kiln_url.trim_end_matches('/').to_string(),
        adapter: adapter.request_value.as_str().map(ToOwned::to_owned),
        adapter_label: adapter.label,
        behavior_policy: Some(behavior_policy),
        environments: summary_environments,
        groups: options.groups,
        group_size: options.group_size,
        rollout_count: records.len(),
        seed_start: options.seed_start,
        max_steps: options.max_steps,
        concurrency: options.concurrency,
        max_action_tokens: options.max_action_tokens,
        thinking_budget_tokens: options.thinking_budget_tokens,
        temperature: options.temperature,
        thinking: options.thinking,
        protocol_error_reward: options.protocol_error_reward,
        max_recoverable_errors: options.max_recoverable_errors,
        capacity_wait_seconds: options.capacity_wait_seconds,
        reset_options_sha256: None,
        reset_plan_sha256: Some(reset_plan_sha256),
        output_path: options.output.display().to_string(),
        replay_output_path: options.replay_output.display().to_string(),
        summary_output_path: options.summary_output.display().to_string(),
        dataset_sha256,
        dataset_bytes,
        replay_sha256,
        replay_bytes,
        stats,
        rollouts: records,
        training_selection: None,
        training_contract: None,
        training_submission: None,
    };
    let summary_bytes = pretty_serialized_len(&summary, "summary")?
        .checked_add(1)
        .context("OpenEnv summary byte count overflow")?;
    anyhow::ensure!(
        summary_bytes <= MAX_OPENENV_SUMMARY_BYTES,
        "OpenEnv summary exceeded the {MAX_OPENENV_SUMMARY_BYTES} byte artifact limit"
    );
    let collection = OpenEnvCollection {
        groups,
        replay,
        summary,
    };
    retained_budget.replace(
        retained_budget.used(),
        serialized_len(&collection, "completed collection")?,
        "the completed collection",
    )?;
    Ok(collection)
}

async fn revalidate_openenv_identity(
    (environment, expected, _): &(OpenEnvClient, OpenEnvInspection, OpenEnvActionValidator),
) -> Result<()> {
    environment.revalidate(expected).await.with_context(|| {
        format!(
            "revalidate OpenEnv environment {} before artifact publication",
            environment.base_url()
        )
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_candidate_episode(
    policy: &OpenEnvPolicyTransport,
    environment: &OpenEnvClient,
    inspection: &OpenEnvInspection,
    action_validator: &OpenEnvActionValidator,
    adapter: &Value,
    adapter_label: &str,
    reset_payload: &Value,
    seed: u64,
    group_index: usize,
    candidate_index: usize,
    options: &OpenEnvRolloutOptions,
    control: &OpenEnvCollectionControl,
    retained_budget: &OpenEnvRetainedByteBudget,
) -> Result<CandidateRollout> {
    control.ensure_active()?;
    let (mut session, reset, capacity_retries) = connect_and_reset_with_capacity_checked(
        environment,
        reset_payload,
        Duration::from_secs(options.capacity_wait_seconds),
        || control.ensure_active(),
    )
    .await?;
    anyhow::ensure!(
        !reset.done,
        "OpenEnv environment {} returned done=true from reset; a trainable rollout needs at least one model action",
        inspection.identity.metadata.name
    );
    let messages = initial_messages(inspection, &reset)?;
    let mut retained_bytes = 0usize;
    charge_serialized(
        retained_budget,
        &mut retained_bytes,
        &messages,
        "candidate initial messages",
    )?;
    charge_serialized(
        retained_budget,
        &mut retained_bytes,
        &reset,
        "candidate reset observation",
    )?;
    let mut trajectory = Vec::new();
    // Reset is not a transition. Preserve its tagged reward in the initial
    // observation, but only environment step rewards contribute to return.
    let mut episode_return = 0.0;
    let mut steps = 0usize;
    let mut total_model_tokens = 0usize;
    let mut total_model_latency_ms = 0.0f64;
    let mut termination = OpenEnvEpisodeTerminationV1::MaxSteps;
    let mut protocol_error_code = None;
    let mut recoverable_protocol_errors = 0usize;
    let mut replay_exchanges = Vec::new();
    let mut terminal_protocol_error = false;
    let mut behavior_policy: Option<RolloutBehaviorPolicyIdentityV1> = None;

    for step_index in 0..options.max_steps {
        control.ensure_active()?;
        let generation_seed = generation_seed(seed, candidate_index, step_index);
        let model_action = session
            .keep_alive_while(generate_model_action(
                policy,
                action_validator,
                &messages,
                &trajectory,
                adapter,
                adapter_label,
                generation_seed,
                options.max_action_tokens,
                options.thinking_budget_tokens,
                options.temperature,
                options.thinking,
            ))
            .await
            .with_context(|| {
                format!(
                    "maintain OpenEnv environment {} while generating group {group_index} candidate {candidate_index} step {step_index}",
                    inspection.identity.metadata.name
                )
            })?;
        let model_action = match model_action {
            Ok(action) => action,
            Err(ModelActionFailure::NoOutput {
                message,
                reasoning,
                total_tokens,
                latency_ms,
                behavior_policy: action_behavior_policy,
            }) => {
                observe_openenv_behavior_policy(
                    &mut behavior_policy,
                    action_behavior_policy,
                    group_index,
                    candidate_index,
                    step_index,
                )?;
                total_model_tokens = total_model_tokens.saturating_add(total_tokens);
                total_model_latency_ms += latency_ms;
                // Preserve unfinished reasoning exactly for diagnosis, but do
                // not invent a closing </think> marker. Training selection
                // recognizes this stable code and discards the completion.
                let action_turn = action_segment(reasoning.unwrap_or_default());
                let error_turn = harness_error_segment(&json!({
                    "openenv_harness_error": {
                        "code": "MODEL_ACTION_NO_OUTPUT",
                        "message": message
                    },
                    "done": true
                }))?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &action_turn,
                    "no-output model action turn",
                )?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &error_turn,
                    "no-output model action error turn",
                )?;
                trajectory.push(action_turn);
                trajectory.push(error_turn);
                episode_return += options.protocol_error_reward;
                termination = OpenEnvEpisodeTerminationV1::InvalidModelAction;
                break;
            }
            Err(ModelActionFailure::Invalid {
                code,
                message,
                raw,
                trajectory_content,
                total_tokens,
                latency_ms,
                behavior_policy: action_behavior_policy,
            }) => {
                observe_openenv_behavior_policy(
                    &mut behavior_policy,
                    action_behavior_policy,
                    group_index,
                    candidate_index,
                    step_index,
                )?;
                total_model_tokens = total_model_tokens.saturating_add(total_tokens);
                total_model_latency_ms += latency_ms;
                let action_turn = action_segment(
                    trajectory_content
                        .or(raw)
                        .unwrap_or_else(|| invalid_action_raw(&message)),
                );
                let error_turn = harness_error_segment(&json!({
                    "openenv_harness_error": {
                        "code": code,
                        "message": message
                    },
                    "done": true
                }))?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &action_turn,
                    "invalid model action turn",
                )?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &error_turn,
                    "invalid model action error turn",
                )?;
                trajectory.push(action_turn);
                trajectory.push(error_turn);
                episode_return += options.protocol_error_reward;
                termination = OpenEnvEpisodeTerminationV1::InvalidModelAction;
                break;
            }
            Err(ModelActionFailure::Request(error)) => return Err(error),
        };
        let ModelAction {
            trajectory_content,
            action,
            total_tokens,
            latency_ms,
            behavior_policy: action_behavior_policy,
        } = model_action;
        observe_openenv_behavior_policy(
            &mut behavior_policy,
            action_behavior_policy,
            group_index,
            candidate_index,
            step_index,
        )?;
        total_model_tokens = total_model_tokens.saturating_add(total_tokens);
        total_model_latency_ms += latency_ms;
        let action_turn = action_segment(trajectory_content);
        charge_serialized(
            retained_budget,
            &mut retained_bytes,
            &action_turn,
            "model action turn",
        )?;
        charge_serialized(
            retained_budget,
            &mut retained_bytes,
            &action,
            "replay action",
        )?;
        trajectory.push(action_turn);
        steps = steps.saturating_add(1);

        match session.step(&action).await {
            Ok(observation) => {
                episode_return += observation.reward.training_value();
                anyhow::ensure!(
                    episode_return.is_finite(),
                    "OpenEnv episode return became non-finite"
                );
                let done = observation.done;
                let observation_turn = observation_segment(&observation)?;
                let result = OpenEnvReplayExchangeResult::Observation { observation };
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &observation_turn,
                    "environment observation turn",
                )?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &result,
                    "replay observation",
                )?;
                trajectory.push(observation_turn);
                replay_exchanges.push(OpenEnvReplayExchange {
                    step_index,
                    action,
                    result,
                });
                if done {
                    termination = OpenEnvEpisodeTerminationV1::Done;
                    break;
                }
            }
            Err(OpenEnvClientError::Protocol(error)) => {
                let continued = !error.code.is_terminal()
                    && recoverable_protocol_errors < options.max_recoverable_errors;
                if !error.code.is_terminal() {
                    recoverable_protocol_errors = recoverable_protocol_errors.saturating_add(1);
                }
                if !continued {
                    protocol_error_code = Some(error.code.to_string());
                    terminal_protocol_error = error.code.is_terminal();
                    termination = OpenEnvEpisodeTerminationV1::ProtocolError;
                }
                let error_turn = protocol_error_segment(&error, continued)?;
                let result = OpenEnvReplayExchangeResult::ProtocolError { error, continued };
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &error_turn,
                    "protocol error turn",
                )?;
                charge_serialized(
                    retained_budget,
                    &mut retained_bytes,
                    &result,
                    "replay protocol error",
                )?;
                trajectory.push(error_turn);
                replay_exchanges.push(OpenEnvReplayExchange {
                    step_index,
                    action,
                    result,
                });
                episode_return += options.protocol_error_reward;
                anyhow::ensure!(
                    episode_return.is_finite(),
                    "OpenEnv episode return became non-finite"
                );
                if !continued {
                    break;
                }
            }
            Err(error) => {
                return Err(anyhow!(error)).with_context(|| {
                    format!(
                        "step OpenEnv environment {} at group {group_index} candidate {candidate_index} step {step_index}",
                        inspection.identity.metadata.name
                    )
                });
            }
        }
    }
    let final_state = if terminal_protocol_error {
        None
    } else {
        Some(session.state().await.with_context(|| {
            format!(
                "capture final OpenEnv state for group {group_index} candidate {candidate_index}"
            )
        })?)
    };
    if let Some(state) = &final_state {
        charge_serialized(
            retained_budget,
            &mut retained_bytes,
            state,
            "final environment state",
        )?;
    }
    let _ = session.close().await;

    let action_schema_sha256 = sha256_json(&inspection.schema.action)?;
    let reset_sha256 = sha256_json(reset_payload)?;
    let terminal_done = matches!(termination, OpenEnvEpisodeTerminationV1::Done);
    let openenv = OpenEnvRolloutProvenanceV1::new(
        inspection.identity.metadata.name.clone(),
        inspection.identity.base_url.clone(),
        inspection.identity.openapi_version.clone(),
        inspection
            .identity
            .discovery_sha256
            .clone()
            .context("current OpenEnv inspection omitted its complete discovery digest")?,
        inspection.identity.schema_sha256.clone(),
        action_schema_sha256,
        reset_sha256,
        seed,
        steps,
        episode_return,
        terminal_done,
        termination,
        protocol_error_code.clone(),
    )
    .map_err(anyhow::Error::msg)
    .context("build OpenEnv rollout provenance")?
    .with_behavior_policy(
        behavior_policy.context("OpenEnv episode completed without a behavior-policy identity")?,
    )
    .map_err(anyhow::Error::msg)
    .context("bind OpenEnv rollout to its exact behavior policy")?;
    let rollout = ScoredRollout::from_trajectory(trajectory, episode_return).with_openenv(openenv);
    let replay = OpenEnvReplayCandidate {
        candidate_index,
        exchanges: replay_exchanges,
        final_state,
        episode_return,
        terminal_done,
        termination,
        recoverable_protocol_errors,
        capacity_retries,
        model_tokens: total_model_tokens,
        model_latency_ms: total_model_latency_ms,
    };

    let messages_sha256 = sha256_json(&messages)?;
    let reset_observation_sha256 = sha256_json(&reset)?;
    let retain_shared_reset = candidate_index == 0;
    let mut candidate = CandidateRollout {
        candidate_index,
        messages: retain_shared_reset.then_some(messages),
        reset_observation: retain_shared_reset.then_some(reset),
        messages_sha256,
        reset_observation_sha256,
        rollout,
        replay,
        record: OpenEnvRolloutRecord {
            group_index,
            candidate_index,
            environment_name: inspection.identity.metadata.name.clone(),
            environment_url: inspection.identity.base_url.clone(),
            seed,
            steps,
            episode_return,
            terminal_done,
            termination,
            protocol_error_code,
            recoverable_protocol_errors,
            capacity_retries,
            model_tokens: total_model_tokens,
            model_latency_ms: total_model_latency_ms,
        },
        retained_bytes: 0,
    };
    let exact_retained_bytes = serialized_len(&candidate, "completed candidate")?;
    retained_budget.replace(
        retained_bytes,
        exact_retained_bytes,
        "the completed candidate",
    )?;
    candidate.retained_bytes = exact_retained_bytes;
    Ok(candidate)
}

#[allow(clippy::too_many_arguments)]
async fn generate_model_action(
    policy: &OpenEnvPolicyTransport,
    action_validator: &OpenEnvActionValidator,
    messages: &[ChatMessage],
    trajectory: &[TurnSegment],
    adapter: &Value,
    adapter_label: &str,
    seed: u64,
    max_tokens: usize,
    thinking_budget_tokens: Option<usize>,
    temperature: f32,
    thinking: bool,
) -> std::result::Result<ModelAction, ModelActionFailure> {
    let mut request_messages = messages.to_vec();
    request_messages.extend(trajectory.iter().map(|segment| ChatMessage {
        role: segment.role.clone(),
        content: segment.content.clone(),
        tool_call_id: segment.tool_call_id.clone(),
        ..Default::default()
    }));
    let mut body = json!({
        "messages": request_messages,
        "adapter": adapter,
        "stream": false,
        "n": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "rollout_provenance": true,
        "chat_template_kwargs": {
            "enable_thinking": thinking
        }
    });
    // OpenEnv collection owns its reasoning semantics. An omitted run budget
    // is sent as explicit `null` so a server-wide default cannot silently
    // truncate a rollout; a numeric value is applied only when the caller set
    // it. Time budgeting is likewise explicitly disabled for this token-only
    // contract.
    body["thinking_budget_tokens"] = thinking_budget_tokens.map_or(Value::Null, Value::from);
    body["thinking_budget_ms"] = Value::Null;
    let started = Instant::now();
    let response_body = policy.complete(body).await.map_err(|error| {
        ModelActionFailure::Request(error.context(format!(
            "request OpenEnv action from Kiln using adapter {adapter_label}"
        )))
    })?;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let total_tokens = response_body
        .pointer("/usage/total_tokens")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or_default();
    let behavior_policy: RolloutBehaviorPolicyIdentityV1 = response_body
        .pointer("/choices/0/rollout_provenance/behavior_policy")
        .cloned()
        .context("Kiln response choices[0].rollout_provenance.behavior_policy is missing")
        .and_then(|value| {
            serde_json::from_value(value)
                .context("Kiln response behavior-policy identity is malformed")
        })
        .and_then(|identity: RolloutBehaviorPolicyIdentityV1| {
            identity.validate().map_err(anyhow::Error::msg)?;
            Ok(identity)
        })
        .map_err(ModelActionFailure::Request)?;
    let reasoning = response_body
        .pointer("/choices/0/message/reasoning_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    let raw = response_body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if raw.trim().is_empty() {
        let finish_reason = response_body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("unspecified");
        return Err(ModelActionFailure::NoOutput {
            message: format!(
                "policy produced no final environment action (finish_reason={finish_reason}); unfinished thinking is diagnostic data and is not eligible for GRPO"
            ),
            reasoning: reasoning.map(ToOwned::to_owned),
            total_tokens,
            latency_ms,
            behavior_policy,
        });
    }
    let trajectory_content = model_action_trajectory_content(&raw, reasoning, thinking);
    let action =
        parse_and_validate_model_action(&raw, action_validator).map_err(|(code, message)| {
            ModelActionFailure::Invalid {
                code,
                message,
                raw: Some(raw.clone()),
                trajectory_content: Some(trajectory_content.clone()),
                total_tokens,
                latency_ms,
                behavior_policy: behavior_policy.clone(),
            }
        })?;
    Ok(ModelAction {
        trajectory_content,
        action,
        total_tokens,
        latency_ms,
        behavior_policy,
    })
}

fn observe_openenv_behavior_policy(
    episode_policy: &mut Option<RolloutBehaviorPolicyIdentityV1>,
    observed: RolloutBehaviorPolicyIdentityV1,
    group_index: usize,
    candidate_index: usize,
    step_index: usize,
) -> Result<()> {
    if let Some(expected) = episode_policy.as_ref() {
        anyhow::ensure!(
            expected == &observed,
            "OpenEnv behavior policy changed during group {group_index} candidate {candidate_index} at step {step_index}: expected {expected:?}, observed {observed:?}"
        );
    } else {
        *episode_policy = Some(observed);
    }
    Ok(())
}

pub(crate) fn initial_messages(
    inspection: &OpenEnvInspection,
    reset: &OpenEnvObservation,
) -> Result<Vec<ChatMessage>> {
    let action_schema = serde_json::to_string(&inspection.schema.action)
        .context("serialize OpenEnv action schema")?;
    let observation = model_observation_content("reset result", reset)?;
    let metadata = &inspection.identity.metadata;
    Ok(vec![
        ChatMessage::new(
            "system",
            format!(
                "You are the policy acting in the OpenEnv reinforcement-learning environment {name:?}.\n\
                 Environment description: {description}\n\
                 At every turn, reply with exactly one JSON object that validates against this action schema:\n\
                 {action_schema}\n\
                 Do not use Markdown, commentary, or a code fence. A non-empty input_text field is \
                 pasted verbatim as the environment's decision prompt; otherwise the complete \
                 environment observation follows as JSON. In either case, encode your answer as \
                 the JSON object required by the action schema.",
                name = metadata.name,
                description = metadata.description
            ),
        ),
        ChatMessage::new("user", observation),
    ])
}

fn action_segment(content: String) -> TurnSegment {
    TurnSegment {
        role: "assistant".to_string(),
        content,
        kind: TurnKind::Action,
        tool_call_id: None,
        warning_prefix_len: None,
    }
}

/// Preserve the generated reasoning stream while keeping the environment
/// action independently parseable. Qwen's prompt already supplies
/// `<think>\n`; retaining the reasoning plus closing suffix (without another
/// opening tag) lets the chat template reconstruct the exact prior assistant
/// turn and makes the trajectory action mask start at the first generated
/// reasoning token rather than at prompt-owned scaffolding.
fn model_action_trajectory_content(
    action: &str,
    reasoning: Option<&str>,
    thinking: bool,
) -> String {
    if !thinking {
        return action.to_string();
    }
    // The completion splitter returns the exact bytes before and after the
    // generated close tag. Joining those parts around only the removed tag
    // reconstructs the model output byte-for-byte, including whitespace and
    // the valid immediate-close case where reasoning_content is absent.
    format!("{}</think>{action}", reasoning.unwrap_or_default())
}

fn observation_segment(observation: &OpenEnvObservation) -> Result<TurnSegment> {
    Ok(TurnSegment {
        role: "tool".to_string(),
        content: model_observation_content("step result", observation)?,
        kind: TurnKind::Observation,
        tool_call_id: None,
        warning_prefix_len: None,
    })
}

pub(crate) fn model_observation_content(
    label: &str,
    observation: &OpenEnvObservation,
) -> Result<String> {
    let complete = serde_json::to_string(observation)
        .with_context(|| format!("serialize complete OpenEnv {label}"))?;
    let input_text = observation
        .observation
        .get("input_text")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty());
    Ok(match input_text {
        Some(input_text) => input_text.to_string(),
        None => format!(
            "OpenEnv {label} (observation, reward, done, and optional metadata JSON):\n{complete}"
        ),
    })
}

fn harness_error_segment(error: &impl Serialize) -> Result<TurnSegment> {
    let content = serde_json::to_string(error)
        .context("serialize OpenEnv harness error trajectory segment")?;
    Ok(TurnSegment {
        role: "tool".to_string(),
        warning_prefix_len: Some(content.len()),
        content,
        kind: TurnKind::Observation,
        tool_call_id: None,
    })
}

fn protocol_error_segment(error: &OpenEnvProtocolError, continued: bool) -> Result<TurnSegment> {
    harness_error_segment(&json!({
        "openenv_error": error,
        "recoverable": continued,
        "done": !continued
    }))
}

fn invalid_action_raw(message: &str) -> String {
    json!({
        "invalid_model_action": message
    })
    .to_string()
}

pub(crate) fn parse_model_action(raw: &str) -> std::result::Result<Value, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("model returned an empty action".to_string());
    }
    let candidate = if let Some(fenced) = trimmed.strip_prefix("```") {
        let fenced = fenced
            .strip_prefix("json")
            .or_else(|| fenced.strip_prefix("JSON"))
            .unwrap_or(fenced)
            .trim_start();
        fenced
            .strip_suffix("```")
            .map(str::trim_end)
            .ok_or_else(|| "model action opened a code fence but did not close it".to_string())?
    } else {
        trimmed
    };
    let value: Value = serde_json::from_str(candidate)
        .map_err(|error| format!("model action is not one JSON value: {error}"))?;
    if !value.is_object() {
        return Err(format!(
            "model action must be a JSON object, got {}",
            json_type_name(&value)
        ));
    }
    Ok(value)
}

/// Parse the final environment action from a retained assistant trajectory.
/// Thinking-on trajectories contain generated reasoning and a real closing
/// delimiter before the independently parseable `content` JSON.
pub(crate) fn parse_trajectory_model_action(raw: &str) -> std::result::Result<Value, String> {
    let final_action = raw
        .split_once("</think>")
        .map_or(raw, |(_, final_action)| final_action);
    parse_model_action(final_action)
}

fn parse_and_validate_model_action(
    raw: &str,
    validator: &OpenEnvActionValidator,
) -> std::result::Result<Value, (&'static str, String)> {
    let action = parse_model_action(raw).map_err(|message| ("INVALID_MODEL_ACTION", message))?;
    validator
        .validate(&action)
        .map_err(|error| ("ACTION_SCHEMA_VALIDATION_FAILED", error.to_string()))?;
    Ok(action)
}

fn reset_payload(base: &Value, seed: u64) -> Result<Value> {
    let mut object = base
        .as_object()
        .cloned()
        .context("OpenEnv reset options must be a JSON object")?;
    object.insert("seed".to_string(), Value::from(seed));
    Ok(Value::Object(object))
}

fn read_reset_options(path: Option<&Path>, direct: Option<&Value>) -> Result<Value> {
    anyhow::ensure!(
        path.is_none() || direct.is_none(),
        "OpenEnv reset options must come from either a file or an inline value, not both"
    );
    if let Some(value) = direct {
        anyhow::ensure!(
            value.is_object(),
            "OpenEnv inline reset options must be one JSON object"
        );
        ensure_reset_options_size(value, "inline reset options")?;
        return Ok(value.clone());
    }
    let Some(path) = path else {
        return Ok(Value::Object(Map::new()));
    };
    let file = std::fs::File::open(path)
        .with_context(|| format!("open OpenEnv reset options {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat OpenEnv reset options {}", path.display()))?;
    anyhow::ensure!(
        metadata.len() <= MAX_OPENENV_RESET_OPTIONS_BYTES as u64,
        "OpenEnv reset options {} exceed the {} byte input limit",
        path.display(),
        MAX_OPENENV_RESET_OPTIONS_BYTES
    );
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .unwrap_or(MAX_OPENENV_RESET_OPTIONS_BYTES)
            .min(MAX_OPENENV_RESET_OPTIONS_BYTES),
    );
    file.take((MAX_OPENENV_RESET_OPTIONS_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read OpenEnv reset options {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= MAX_OPENENV_RESET_OPTIONS_BYTES,
        "OpenEnv reset options {} grew beyond the {} byte input limit while being read",
        path.display(),
        MAX_OPENENV_RESET_OPTIONS_BYTES
    );
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse OpenEnv reset options {}", path.display()))?;
    anyhow::ensure!(
        value.is_object(),
        "OpenEnv reset options {} must contain one JSON object",
        path.display()
    );
    ensure_reset_options_size(&value, &format!("reset options {}", path.display()))?;
    Ok(value)
}

fn ensure_reset_options_size(value: &Value, label: &str) -> Result<()> {
    let bytes = serialized_len(value, label)?;
    anyhow::ensure!(
        bytes <= MAX_OPENENV_RESET_OPTIONS_BYTES,
        "OpenEnv {label} serialize to {bytes} bytes; limit is {MAX_OPENENV_RESET_OPTIONS_BYTES} bytes so the seeded reset request remains within the protocol frame bound"
    );
    Ok(())
}

fn normalize_reset_options(mut value: Value) -> Result<Value> {
    let object = value
        .as_object_mut()
        .context("OpenEnv reset options must be a JSON object")?;
    object.remove("seed");
    Ok(value)
}

fn read_reset_plan(options: &OpenEnvRolloutOptions) -> Result<Vec<Value>> {
    let environment_count = options.environment_urls.len();
    if !options.environment_reset_options.is_empty() {
        return options
            .environment_reset_options
            .iter()
            .map(|path| read_reset_options(path.as_deref(), None).and_then(normalize_reset_options))
            .collect();
    }
    if !options.environment_reset_options_values.is_empty() {
        return options
            .environment_reset_options_values
            .iter()
            .cloned()
            .map(normalize_reset_options)
            .collect();
    }
    let shared = normalize_reset_options(read_reset_options(
        options.reset_options.as_deref(),
        options.reset_options_value.as_ref(),
    )?)?;
    Ok(vec![shared; environment_count])
}

fn parse_cli_credential_envs_unchecked(values: &[String]) -> Vec<Option<String>> {
    values
        .iter()
        .map(|value| (value != "-").then(|| value.clone()))
        .collect()
}

fn parse_cli_credential_envs(
    values: &[String],
    environment_count: usize,
) -> Result<Vec<Option<String>>> {
    let parsed = parse_cli_credential_envs_unchecked(values);
    validate_credential_envs(&parsed, environment_count)?;
    Ok(parsed)
}

fn resolved_credential_envs(options: &OpenEnvRolloutOptions) -> Vec<Option<String>> {
    if options.credential_envs.is_empty() {
        vec![None; options.environment_urls.len()]
    } else {
        options.credential_envs.clone()
    }
}

fn validate_credential_envs(
    credential_envs: &[Option<String>],
    environment_count: usize,
) -> Result<()> {
    anyhow::ensure!(
        credential_envs.is_empty() || credential_envs.len() == environment_count,
        "OpenEnv credential envs must be empty or contain exactly one entry per environment (expected {environment_count}, got {})",
        credential_envs.len()
    );
    for (index, credential_env) in credential_envs.iter().enumerate() {
        let Some(name) = credential_env.as_deref() else {
            continue;
        };
        let mut bytes = name.bytes();
        let valid_start = bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
        let valid_rest = bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
        anyhow::ensure!(
            valid_start && valid_rest && name.len() <= 128,
            "OpenEnv credential env at position {index} must name a 1..=128 character environment variable matching [A-Za-z_][A-Za-z0-9_]*"
        );
        kiln_train::validate_bearer_secret_environment(name).map_err(|error| {
            let detail = match error {
                kiln_train::CredentialLookupError::Unavailable => "is unavailable",
                kiln_train::CredentialLookupError::Empty => "is empty",
            };
            anyhow::anyhow!("OpenEnv credential at position {index} {detail}")
        })?;
    }
    Ok(())
}

pub(crate) fn openenv_client(
    environment_url: &str,
    credential_env: Option<&str>,
) -> Result<OpenEnvClient> {
    let client = OpenEnvClient::new(environment_url)?;
    let Some(name) = credential_env else {
        return Ok(client);
    };
    let token = kiln_train::bearer_secret_from_environment(name).map_err(|error| {
        let detail = match error {
            kiln_train::CredentialLookupError::Unavailable => "is unavailable",
            kiln_train::CredentialLookupError::Empty => "is empty",
        };
        anyhow::anyhow!("configured OpenEnv credential {detail}")
    })?;
    client
        .with_bearer_token(token)
        .context("configure OpenEnv bearer credential")
}

pub(crate) fn validate_options(options: &OpenEnvRolloutOptions) -> Result<()> {
    anyhow::ensure!(
        !options.environment_urls.is_empty()
            && options.environment_urls.len() <= MAX_OPENENV_ENVIRONMENTS,
        "OpenEnv requires 1..={MAX_OPENENV_ENVIRONMENTS} --environment URL values"
    );
    validate_credential_envs(&options.credential_envs, options.environment_urls.len())?;
    anyhow::ensure!(
        options.groups > 0 && options.groups <= MAX_OPENENV_GROUPS,
        "OpenEnv groups must be in 1..={MAX_OPENENV_GROUPS}"
    );
    anyhow::ensure!(
        options.groups >= options.environment_urls.len(),
        "OpenEnv groups must be at least the number of environments so every configured endpoint is exercised and receipt-verifiable (expected at least {}, got {})",
        options.environment_urls.len(),
        options.groups
    );
    let final_seed = options
        .seed_start
        .checked_add((options.groups - 1) as u64)
        .context("OpenEnv reset seed range overflow")?;
    anyhow::ensure!(
        final_seed <= i64::MAX as u64,
        "OpenEnv reset seed range must fit the protocol's portable signed-integer range"
    );
    anyhow::ensure!(
        options.group_size > 0 && options.group_size <= MAX_OPENENV_GROUP_SIZE,
        "OpenEnv group size must be in 1..={MAX_OPENENV_GROUP_SIZE}"
    );
    let rollouts = options
        .groups
        .checked_mul(options.group_size)
        .context("OpenEnv rollout count overflow")?;
    anyhow::ensure!(
        rollouts <= MAX_OPENENV_ROLLOUTS,
        "OpenEnv groups * group-size must not exceed {MAX_OPENENV_ROLLOUTS}"
    );
    anyhow::ensure!(
        options.max_steps > 0 && options.max_steps <= MAX_OPENENV_STEPS,
        "OpenEnv max steps must be in 1..={MAX_OPENENV_STEPS}"
    );
    anyhow::ensure!(
        options.concurrency > 0 && options.concurrency <= MAX_OPENENV_CONCURRENCY,
        "OpenEnv concurrency must be in 1..={MAX_OPENENV_CONCURRENCY}"
    );
    anyhow::ensure!(
        options.max_action_tokens > 0 && options.max_action_tokens <= MAX_OPENENV_ACTION_TOKENS,
        "OpenEnv max action tokens must be in 1..={MAX_OPENENV_ACTION_TOKENS}"
    );
    anyhow::ensure!(
        options.thinking || options.thinking_budget_tokens.is_none(),
        "OpenEnv thinking-budget-tokens requires thinking=true"
    );
    if let Some(budget) = options.thinking_budget_tokens {
        anyhow::ensure!(
            budget < options.max_action_tokens,
            "OpenEnv thinking-budget-tokens ({budget}) must be smaller than max-action-tokens ({}) so an explicitly budgeted request can still emit an environment action",
            options.max_action_tokens
        );
    }
    anyhow::ensure!(
        options.temperature.is_finite() && options.temperature >= 0.0,
        "OpenEnv action temperature must be finite and non-negative"
    );
    anyhow::ensure!(
        options.protocol_error_reward.is_finite(),
        "OpenEnv protocol error reward must be finite"
    );
    anyhow::ensure!(
        options.max_recoverable_errors <= MAX_OPENENV_RECOVERABLE_ERRORS,
        "OpenEnv max recoverable errors must be in 0..={MAX_OPENENV_RECOVERABLE_ERRORS}"
    );
    anyhow::ensure!(
        options.capacity_wait_seconds > 0
            && options.capacity_wait_seconds <= MAX_OPENENV_CAPACITY_WAIT_SECONDS,
        "OpenEnv capacity wait must be in 1..={MAX_OPENENV_CAPACITY_WAIT_SECONDS} seconds"
    );
    anyhow::ensure!(
        options.reset_options.is_none() || options.reset_options_value.is_none(),
        "OpenEnv reset options file and inline reset options are mutually exclusive"
    );
    anyhow::ensure!(
        options.environment_reset_options.is_empty()
            || options.environment_reset_options.len() == options.environment_urls.len(),
        "OpenEnv environment reset option files must be empty or contain exactly one entry per environment (expected {}, got {})",
        options.environment_urls.len(),
        options.environment_reset_options.len()
    );
    anyhow::ensure!(
        options.environment_reset_options_values.is_empty()
            || options.environment_reset_options_values.len() == options.environment_urls.len(),
        "OpenEnv inline environment reset options must be empty or contain exactly one object per environment (expected {}, got {})",
        options.environment_urls.len(),
        options.environment_reset_options_values.len()
    );
    anyhow::ensure!(
        options.environment_reset_options.is_empty()
            || options.environment_reset_options_values.is_empty(),
        "OpenEnv aligned reset options must come from either files or inline values, not both"
    );
    let has_shared_reset = options.reset_options.is_some() || options.reset_options_value.is_some();
    let has_aligned_reset = !options.environment_reset_options.is_empty()
        || !options.environment_reset_options_values.is_empty();
    anyhow::ensure!(
        !has_shared_reset || !has_aligned_reset,
        "OpenEnv shared reset options and aligned environment reset options are mutually exclusive"
    );
    for (index, value) in options.environment_reset_options_values.iter().enumerate() {
        anyhow::ensure!(
            value.is_object(),
            "OpenEnv inline environment reset options at position {index} must be one JSON object"
        );
    }
    anyhow::ensure!(
        options.output != options.replay_output
            && options.output != options.summary_output
            && options.replay_output != options.summary_output,
        "OpenEnv --output, --replay-output, and --summary-output must name different files"
    );
    Ok(())
}

fn validate_output_adapter(adapter: &str) -> Result<()> {
    anyhow::ensure!(
        !adapter.trim().is_empty(),
        "OpenEnv training output adapter must not be blank"
    );
    Ok(())
}

fn openenv_training_preflight_request(
    options: &OpenEnvTrainOptions,
) -> OpenEnvTrainingPreflightRequest {
    let mut training_config = GrpoConfig::default();
    if let Some(rank) = options.lora_rank {
        training_config.lora_rank = rank;
    }
    OpenEnvTrainingPreflightRequest {
        adapter: options.rollout.adapter.clone(),
        output_adapter: options.output_adapter.clone(),
        training_config,
        auto_load: options.auto_load,
        post_eval: None,
    }
}

fn validate_openenv_training_preflight_receipt(
    request: &OpenEnvTrainingPreflightRequest,
    receipt: &OpenEnvTrainingPreflightReceipt,
) -> Result<()> {
    anyhow::ensure!(
        receipt.schema == OPENENV_TRAINING_PREFLIGHT_SCHEMA_V1,
        "Kiln returned unsupported OpenEnv training preflight schema {:?}",
        receipt.schema
    );
    anyhow::ensure!(
        !receipt.capacity_reserved,
        "Kiln OpenEnv training preflight v1 cannot claim a capacity reservation"
    );
    let expected_base = if matches!(
        request.adapter.trim().to_ascii_lowercase().as_str(),
        "base" | "none" | "null"
    ) {
        None
    } else {
        Some(request.adapter.trim())
    };
    let mut expected_config = request.training_config.clone();
    expected_config.output_name = Some(request.output_adapter.clone());
    expected_config.base_adapter = expected_base.map(str::to_string);
    expected_config.auto_load = request.auto_load;
    expected_config.behavior_policy = BehaviorPolicy::NoImportanceCorrection;
    anyhow::ensure!(
        serde_json::to_value(&receipt.effective_config)? == serde_json::to_value(&expected_config)?,
        "Kiln OpenEnv training preflight changed a requested or rollout-owned GRPO field"
    );
    anyhow::ensure!(
        receipt.capacity.queued_jobs < receipt.capacity.max_queued_jobs
            && receipt.capacity.tracked_jobs < receipt.capacity.max_tracked_jobs,
        "Kiln accepted OpenEnv training preflight with no native queue capacity"
    );
    anyhow::ensure!(
        serde_json::to_value(&receipt.post_eval)? == serde_json::to_value(&request.post_eval)?,
        "Kiln OpenEnv training preflight changed the requested post-evaluation contract"
    );
    let behavior_policy = receipt
        .behavior_policy
        .as_ref()
        .context("Kiln OpenEnv training preflight omitted its behavior-policy identity")?;
    behavior_policy
        .validate()
        .map_err(anyhow::Error::msg)
        .context("Kiln OpenEnv training preflight returned an invalid behavior-policy identity")?;
    anyhow::ensure!(
        behavior_policy
            .adapter
            .as_ref()
            .map(|adapter| adapter.name.as_str())
            == expected_base,
        "Kiln OpenEnv training preflight behavior policy differs from the requested adapter"
    );
    Ok(())
}

async fn preflight_openenv_training(
    options: &OpenEnvTrainOptions,
) -> Result<OpenEnvTrainingPreflightReceipt> {
    let request = openenv_training_preflight_request(options);
    let response = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHAT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build OpenEnv training preflight client")?
        .post(format!(
            "{}/v1/openenv/training/preflight",
            options.rollout.kiln_url.trim_end_matches('/')
        ))
        .header("x-kiln-client", "openenv")
        .json(&request)
        .send()
        .await
        .context("preflight OpenEnv GRPO training before collection")?;
    let status = response.status();
    let body = read_kiln_json_bounded(response, "training preflight").await?;
    anyhow::ensure!(
        status.is_success(),
        "OpenEnv training preflight returned HTTP {status} before collection: {}",
        serde_json::to_string(&body).unwrap_or_default()
    );
    let receipt: OpenEnvTrainingPreflightReceipt =
        serde_json::from_value(body).context("decode Kiln OpenEnv training preflight receipt")?;
    validate_openenv_training_preflight_receipt(&request, &receipt)?;
    Ok(receipt)
}

struct AdapterSelection {
    request_value: Value,
    label: String,
}

fn parse_adapter_selection(adapter: &str) -> AdapterSelection {
    if matches!(
        adapter.trim().to_ascii_lowercase().as_str(),
        "base" | "none" | "null"
    ) {
        AdapterSelection {
            request_value: Value::Null,
            label: "base".to_string(),
        }
    } else {
        AdapterSelection {
            request_value: Value::String(adapter.to_string()),
            label: adapter.to_string(),
        }
    }
}

async fn submit_openenv_training(
    kiln_url: &str,
    groups: &[AgenticGroup],
    config: &GrpoConfig,
    post_eval: Option<&kiln_eval::PostEvalConfig>,
) -> Result<Value> {
    let mut body = json!({
        "groups": groups,
        "config": config
    });
    if let Some(post_eval) = post_eval {
        body["post_eval"] = serde_json::to_value(post_eval)?;
    }
    let response = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHAT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build OpenEnv training client")?
        .post(format!("{}/v1/train/grpo", kiln_url.trim_end_matches('/')))
        .header("x-kiln-client", "openenv")
        .json(&body)
        .send()
        .await
        .context("submit OpenEnv GRPO training")?;
    let status = response.status();
    let body = read_kiln_json_bounded(response, "GRPO submission").await?;
    anyhow::ensure!(
        status.is_success(),
        "OpenEnv GRPO submission returned HTTP {status}: {}",
        serde_json::to_string(&body).unwrap_or_default()
    );
    Ok(body)
}

async fn read_kiln_json_bounded(mut response: reqwest::Response, label: &str) -> Result<Value> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_KILN_RESPONSE_BYTES as u64)
    {
        anyhow::bail!(
            "Kiln OpenEnv {label} response exceeded the {MAX_KILN_RESPONSE_BYTES} byte limit"
        );
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("read Kiln OpenEnv {label} response"))?
    {
        let next_len = body
            .len()
            .checked_add(chunk.len())
            .context("Kiln OpenEnv response byte count overflow")?;
        anyhow::ensure!(
            next_len <= MAX_KILN_RESPONSE_BYTES,
            "Kiln OpenEnv {label} response exceeded the {MAX_KILN_RESPONSE_BYTES} byte limit"
        );
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body)
        .with_context(|| format!("decode Kiln OpenEnv {label} response as JSON"))
}

pub(crate) fn write_openenv_outputs(
    options: &OpenEnvRolloutOptions,
    groups: &[AgenticGroup],
    replay: &OpenEnvReplayManifest,
    summary: &OpenEnvRolloutSummary,
) -> Result<()> {
    write_groups_atomic(&options.output, groups)?;
    write_replay_atomic(&options.replay_output, replay)?;
    write_summary_atomic(&options.summary_output, summary)
}

fn write_replay_atomic(path: &Path, replay: &OpenEnvReplayManifest) -> Result<()> {
    let bytes = encode_replay(replay)?;
    let parent = output_parent(path)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create staged OpenEnv replay beside {}", path.display()))?;
    staged
        .as_file_mut()
        .write_all(&bytes)
        .with_context(|| format!("write staged OpenEnv replay for {}", path.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv replay for {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish OpenEnv replay {}", path.display()))?;
    Ok(())
}

fn write_groups_atomic(path: &Path, groups: &[AgenticGroup]) -> Result<()> {
    let parent = output_parent(path)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create staged OpenEnv dataset beside {}", path.display()))?;
    {
        let mut writer = BufWriter::new(staged.as_file_mut());
        for group in groups {
            serde_json::to_writer(&mut writer, group)
                .context("serialize OpenEnv GRPO JSONL group")?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv dataset for {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish OpenEnv dataset {}", path.display()))?;
    Ok(())
}

pub(crate) fn write_summary_atomic(path: &Path, summary: &OpenEnvRolloutSummary) -> Result<()> {
    let parent = output_parent(path)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create staged OpenEnv summary beside {}", path.display()))?;
    {
        let mut writer = BoundedWriter::new(
            staged.as_file_mut(),
            MAX_OPENENV_SUMMARY_BYTES,
            "OpenEnv summary",
        );
        serde_json::to_writer_pretty(&mut writer, summary)
            .context("serialize bounded OpenEnv rollout summary")?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv summary for {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish OpenEnv summary {}", path.display()))?;
    Ok(())
}

fn output_parent(path: &Path) -> Result<&Path> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    anyhow::ensure!(
        parent.is_dir(),
        "OpenEnv output parent {} is not a directory",
        parent.display()
    );
    Ok(parent)
}

pub(crate) fn summarize_rollouts(records: &[OpenEnvRolloutRecord]) -> OpenEnvRolloutStats {
    if records.is_empty() {
        return OpenEnvRolloutStats::default();
    }
    let sum = records
        .iter()
        .map(|record| record.episode_return)
        .sum::<f64>();
    let latency_sum = records
        .iter()
        .map(|record| record.model_latency_ms)
        .sum::<f64>();
    OpenEnvRolloutStats {
        mean_episode_return: sum / records.len() as f64,
        min_episode_return: records
            .iter()
            .map(|record| record.episode_return)
            .reduce(f64::min),
        max_episode_return: records
            .iter()
            .map(|record| record.episode_return)
            .reduce(f64::max),
        done_count: records
            .iter()
            .filter(|record| record.termination == OpenEnvEpisodeTerminationV1::Done)
            .count(),
        max_steps_count: records
            .iter()
            .filter(|record| record.termination == OpenEnvEpisodeTerminationV1::MaxSteps)
            .count(),
        invalid_model_action_count: records
            .iter()
            .filter(|record| record.termination == OpenEnvEpisodeTerminationV1::InvalidModelAction)
            .count(),
        protocol_error_count: records
            .iter()
            .filter(|record| record.termination == OpenEnvEpisodeTerminationV1::ProtocolError)
            .count(),
        recoverable_protocol_error_count: records
            .iter()
            .map(|record| record.recoverable_protocol_errors)
            .sum(),
        capacity_retry_count: records.iter().map(|record| record.capacity_retries).sum(),
        total_environment_steps: records.iter().map(|record| record.steps).sum(),
        total_model_tokens: records.iter().map(|record| record.model_tokens).sum(),
        mean_model_latency_ms: latency_sum / records.len() as f64,
    }
}

fn generation_seed(seed: u64, candidate_index: usize, step_index: usize) -> u64 {
    seed ^ (candidate_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (step_index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    let mut hasher = Sha256::new();
    serde_json::to_writer(
        Sha256Writer {
            hasher: &mut hasher,
        },
        value,
    )
    .context("serialize value for SHA-256")?;
    Ok(format_digest(hasher.finalize().as_slice()))
}

fn sha256_jsonl(groups: &[AgenticGroup]) -> Result<String> {
    let mut hasher = Sha256::new();
    {
        let mut writer = Sha256Writer {
            hasher: &mut hasher,
        };
        for group in groups {
            serde_json::to_writer(&mut writer, group)
                .context("serialize OpenEnv group for SHA-256")?;
            writer.write_all(b"\n")?;
        }
    }
    Ok(format_digest(hasher.finalize().as_slice()))
}

fn format_digest(bytes: &[u8]) -> String {
    let hex = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sha256:{hex}")
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests;

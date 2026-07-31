//! Native OpenEnv rollout collection and GRPO submission.
//!
//! The protocol boundary lives in `kiln-openenv`. This module composes it with
//! Kiln chat generation and canonical trajectory/GRPO types:
//!
//!   inspect -> reset(seed) -> model action -> step -> reward -> trajectory
//!           -> grouped JSONL -> optional `/v1/train/grpo`

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use clap::Subcommand;
use console::style;
use futures::{StreamExt, stream};
use kiln_openenv::{
    OpenEnvClient, OpenEnvClientError, OpenEnvInspection, OpenEnvObservation, OpenEnvProtocolError,
};
use kiln_train::{
    AgenticGroup, ChatMessage, OpenEnvEpisodeTerminationV1, OpenEnvRolloutProvenanceV1,
    ScoredRollout, TurnKind, TurnSegment,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::config::default_server_url;
use crate::openenv_replay::{
    OPENENV_REPLAY_SCHEMA_V1, OpenEnvReplayCandidate, OpenEnvReplayExchange,
    OpenEnvReplayExchangeResult, OpenEnvReplayGroup, OpenEnvReplayManifest,
    connect_and_reset_with_capacity, encode_replay, replay_openenv, sha256_bytes as replay_sha256,
    verify_openenv_artifacts,
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
const MAX_OPENENV_DATASET_BYTES: usize = 256 * 1024 * 1024;
const MAX_KILN_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const CHAT_TIMEOUT: Duration = Duration::from_secs(180);

pub(crate) const OPENENV_OVERVIEW: &str = r#"Inspect OpenEnv servers, collect grouped stateful episodes, and train a Kiln LoRA directly from environment-owned rewards.

Kiln discovers each environment over HTTP, opens one WebSocket session per episode, resets every candidate in a GRPO group with the same deterministic seed, asks the selected Kiln policy for schema-shaped JSON actions, and records every action, observation, reward, termination, environment identity, and content hash in canonical agentic trajectory JSONL.

`rollout` writes the exact reusable GRPO corpus, an exact replay transcript, and a detailed summary receipt. `verify` validates the three-artifact bundle without contacting a server; `replay` re-executes the captured reset/action protocol against the content-addressed environments. `train` writes those artifacts and submits the in-memory groups to `/v1/train/grpo` with the explicit native on-policy behavior-policy contract. Start `kiln serve` first.
"#;

pub(crate) const OPENENV_EXAMPLES: &str = r#"Examples:
  kiln openenv inspect --environment http://127.0.0.1:8000
      Check health and print the environment metadata, schemas, protocol
      profile, WebSocket URL, and content-addressed schema identity.

  kiln openenv rollout --environment http://127.0.0.1:8000 --groups 8 --group-size 4
      Collect 32 live episodes as eight seed-matched GRPO groups and write
      openenv.rollouts.jsonl plus openenv.rollout-summary.json.

  kiln openenv train --environment http://127.0.0.1:8000 --output-adapter wordle-agent
      Collect a native on-policy batch, submit GRPO training, and auto-load the
      completed adapter.

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

    /// JSON object merged into every reset request; Kiln always sets `seed`
    #[arg(long = "reset-options", value_name = "JSON")]
    reset_options: Option<PathBuf>,

    /// Maximum model actions in one episode
    #[arg(long = "max-steps", default_value_t = 8)]
    max_steps: usize,

    /// Maximum simultaneous candidate sessions within a group
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Maximum tokens generated for one JSON action
    #[arg(long = "max-action-tokens", default_value_t = 256)]
    max_action_tokens: usize,

    /// Policy sampling temperature
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Explicitly enable or disable Qwen thinking for action generation
    #[arg(
        long,
        action = clap::ArgAction::Set,
        value_parser = clap::value_parser!(bool),
        default_value_t = false
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

        /// Emit the complete inspection as JSON
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

#[derive(Debug, Clone)]
pub struct OpenEnvRolloutOptions {
    pub kiln_url: String,
    pub environment_urls: Vec<String>,
    pub adapter: String,
    pub groups: usize,
    pub group_size: usize,
    pub seed_start: u64,
    pub reset_options: Option<PathBuf>,
    pub max_steps: usize,
    pub concurrency: usize,
    pub max_action_tokens: usize,
    pub temperature: f32,
    pub thinking: bool,
    pub protocol_error_reward: f64,
    pub max_recoverable_errors: usize,
    pub capacity_wait_seconds: u64,
    pub output: PathBuf,
    pub replay_output: PathBuf,
    pub summary_output: PathBuf,
}

#[derive(Debug, Clone)]
pub struct OpenEnvTrainOptions {
    pub rollout: OpenEnvRolloutOptions,
    pub output_adapter: String,
    pub lora_rank: Option<usize>,
    pub auto_load: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRolloutSummary {
    pub schema: String,
    pub kiln_url: String,
    pub adapter: Option<String>,
    pub adapter_label: String,
    pub environments: Vec<OpenEnvInspection>,
    pub groups: usize,
    pub group_size: usize,
    pub rollout_count: usize,
    pub seed_start: u64,
    pub max_steps: usize,
    pub concurrency: usize,
    pub max_action_tokens: usize,
    pub temperature: f32,
    pub thinking: bool,
    pub protocol_error_reward: f64,
    pub max_recoverable_errors: usize,
    pub capacity_wait_seconds: u64,
    pub reset_options_sha256: String,
    pub output_path: String,
    pub replay_output_path: String,
    pub summary_output_path: String,
    pub dataset_sha256: String,
    pub dataset_bytes: usize,
    pub replay_sha256: String,
    pub replay_bytes: usize,
    pub stats: OpenEnvRolloutStats,
    pub rollouts: Vec<OpenEnvRolloutRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_submission: Option<Value>,
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

#[derive(Debug)]
pub struct OpenEnvCollection {
    pub groups: Vec<AgenticGroup>,
    pub replay: OpenEnvReplayManifest,
    pub summary: OpenEnvRolloutSummary,
}

#[derive(Debug)]
struct CandidateRollout {
    candidate_index: usize,
    messages: Vec<ChatMessage>,
    reset_observation: OpenEnvObservation,
    rollout: ScoredRollout,
    replay: OpenEnvReplayCandidate,
    record: OpenEnvRolloutRecord,
}

#[derive(Debug)]
struct ModelAction {
    raw: String,
    action: Value,
    total_tokens: usize,
    latency_ms: f64,
}

#[derive(Debug)]
enum ModelActionFailure {
    Invalid {
        message: String,
        raw: Option<String>,
        total_tokens: usize,
        latency_ms: f64,
    },
    Request(anyhow::Error),
}

/// Run native OpenEnv discovery, rollout collection, or rollout-and-train.
pub async fn run_openenv(command: &OpenEnvCommands) -> Result<()> {
    match command {
        OpenEnvCommands::Inspect { environment, json } => {
            let inspection = inspect_openenv(environment).await?;
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
                println!("  Schema SHA-256: {}", identity.schema_sha256);
                println!("  Description:    {}", identity.metadata.description.trim());
                println!();
                println!("  Action schema:");
                for line in serde_json::to_string_pretty(&inspection.schema.action)?.lines() {
                    println!("    {line}");
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
            let report = replay_openenv(
                &verified.replay,
                verified.report.replay_sha256,
                *concurrency,
                Duration::from_secs(*capacity_wait_seconds),
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

fn openenv_rollout_options(args: &OpenEnvRolloutArgs) -> OpenEnvRolloutOptions {
    OpenEnvRolloutOptions {
        kiln_url: args.kiln_url.clone(),
        environment_urls: args.environment_urls.clone(),
        adapter: args.adapter.clone(),
        groups: args.groups,
        group_size: args.group_size,
        seed_start: args.seed_start,
        reset_options: args.reset_options.clone(),
        max_steps: args.max_steps,
        concurrency: args.concurrency,
        max_action_tokens: args.max_action_tokens,
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
    if submitted_training {
        let submission = summary
            .training_submission
            .as_ref()
            .context("OpenEnv train completed without a training submission receipt")?;
        println!(
            "{} GRPO training submitted{}",
            style("✓").green().bold(),
            submission
                .get("job_id")
                .and_then(Value::as_str)
                .map(|job_id| format!(" as {job_id}"))
                .unwrap_or_default()
        );
    }
    Ok(())
}

pub async fn inspect_openenv(environment_url: &str) -> Result<OpenEnvInspection> {
    let client = OpenEnvClient::new(environment_url)?;
    client
        .inspect()
        .await
        .with_context(|| format!("inspect OpenEnv server {}", client.base_url()))
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
    let mut collection = collect_openenv_rollouts(&options.rollout).await?;
    write_openenv_outputs(
        &options.rollout,
        &collection.groups,
        &collection.replay,
        &collection.summary,
    )?;
    let submission = submit_openenv_training(
        &options.rollout.kiln_url,
        &collection.groups,
        &options.output_adapter,
        options.lora_rank,
        options.auto_load,
    )
    .await?;
    collection.summary.training_submission = Some(submission);
    write_summary_atomic(&options.rollout.summary_output, &collection.summary)?;
    Ok(collection.summary)
}

pub async fn collect_openenv_rollouts(
    options: &OpenEnvRolloutOptions,
) -> Result<OpenEnvCollection> {
    validate_options(options)?;
    let reset_options = read_reset_options(options.reset_options.as_deref())?;
    let reset_options_sha256 = sha256_json(&reset_options)?;
    let adapter = parse_adapter_selection(&options.adapter);
    let chat = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHAT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build Kiln chat client")?;

    let inspections = stream::iter(options.environment_urls.iter().cloned())
        .map(|url| async move {
            let client = OpenEnvClient::new(&url)?;
            let inspection = client
                .inspect()
                .await
                .with_context(|| format!("inspect OpenEnv server {}", client.base_url()))?;
            Ok::<_, anyhow::Error>((client, inspection))
        })
        .buffered(options.environment_urls.len())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let mut groups = Vec::with_capacity(options.groups);
    let mut replay_groups = Vec::with_capacity(options.groups);
    let mut records = Vec::with_capacity(options.groups.saturating_mul(options.group_size));
    let mut dataset_bytes = 0usize;

    for group_index in 0..options.groups {
        let environment_index = group_index % inspections.len();
        let (environment, inspection) = &inspections[environment_index];
        let seed = options
            .seed_start
            .checked_add(group_index as u64)
            .context("OpenEnv reset seed range overflow")?;
        if seed > i64::MAX as u64 {
            anyhow::bail!(
                "OpenEnv reset seed {seed} exceeds the protocol's portable signed-integer range"
            );
        }
        let reset = reset_payload(&reset_options, seed)?;

        let mut candidates = stream::iter(0..options.group_size)
            .map(|candidate_index| {
                run_candidate_episode(
                    &chat,
                    environment.clone(),
                    inspection.clone(),
                    adapter.request_value.clone(),
                    adapter.label.clone(),
                    reset.clone(),
                    seed,
                    group_index,
                    candidate_index,
                    options,
                )
            })
            .buffer_unordered(options.concurrency.min(options.group_size))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        candidates.sort_by_key(|candidate| candidate.candidate_index);

        let shared_messages = candidates
            .first()
            .map(|candidate| candidate.messages.clone())
            .context("OpenEnv group unexpectedly produced no candidate rollouts")?;
        for candidate in &candidates {
            anyhow::ensure!(
                candidate.messages == shared_messages,
                "OpenEnv reset produced different initial observations within group {group_index}; reset(seed) must be deterministic for group-relative training"
            );
            anyhow::ensure!(
                candidate.reset_observation == candidates[0].reset_observation,
                "OpenEnv reset produced different wire observations within group {group_index}; reset(seed) must be deterministic for exact replay"
            );
        }
        replay_groups.push(OpenEnvReplayGroup {
            group_index,
            environment_index,
            seed,
            reset_payload: reset,
            reset_observation: candidates[0].reset_observation.clone(),
            candidates: candidates
                .iter()
                .map(|candidate| candidate.replay.clone())
                .collect(),
        });
        let group = AgenticGroup {
            messages: shared_messages,
            completions: candidates
                .iter()
                .map(|candidate| candidate.rollout.clone())
                .collect(),
        };
        let encoded = serde_json::to_vec(&group).context("serialize OpenEnv GRPO group")?;
        dataset_bytes = dataset_bytes
            .checked_add(encoded.len().saturating_add(1))
            .context("OpenEnv dataset byte count overflow")?;
        anyhow::ensure!(
            dataset_bytes <= MAX_OPENENV_DATASET_BYTES,
            "OpenEnv rollout dataset exceeded the {} byte in-memory/inline limit; reduce groups, group size, max steps, or environment observation size",
            MAX_OPENENV_DATASET_BYTES
        );
        records.extend(candidates.into_iter().map(|candidate| candidate.record));
        groups.push(group);
    }

    let dataset_sha256 = sha256_jsonl(&groups)?;
    let replay = OpenEnvReplayManifest {
        schema: OPENENV_REPLAY_SCHEMA_V1.to_string(),
        client_profile: kiln_openenv::OPENENV_CLIENT_PROFILE.to_string(),
        dataset_sha256: dataset_sha256.clone(),
        protocol_error_reward: options.protocol_error_reward,
        max_steps: options.max_steps,
        environments: inspections
            .iter()
            .map(|(_, inspection)| inspection.clone())
            .collect(),
        groups: replay_groups,
    };
    let replay_encoded = encode_replay(&replay)?;
    let replay_sha256 = replay_sha256(&replay_encoded);
    let replay_bytes = replay_encoded.len();
    let stats = summarize_rollouts(&records);
    Ok(OpenEnvCollection {
        groups,
        replay,
        summary: OpenEnvRolloutSummary {
            schema: "kiln.openenv-rollout-summary.v2".to_string(),
            kiln_url: options.kiln_url.trim_end_matches('/').to_string(),
            adapter: adapter.request_value.as_str().map(ToOwned::to_owned),
            adapter_label: adapter.label,
            environments: inspections
                .iter()
                .map(|(_, inspection)| inspection.clone())
                .collect(),
            groups: options.groups,
            group_size: options.group_size,
            rollout_count: records.len(),
            seed_start: options.seed_start,
            max_steps: options.max_steps,
            concurrency: options.concurrency,
            max_action_tokens: options.max_action_tokens,
            temperature: options.temperature,
            thinking: options.thinking,
            protocol_error_reward: options.protocol_error_reward,
            max_recoverable_errors: options.max_recoverable_errors,
            capacity_wait_seconds: options.capacity_wait_seconds,
            reset_options_sha256,
            output_path: options.output.display().to_string(),
            replay_output_path: options.replay_output.display().to_string(),
            summary_output_path: options.summary_output.display().to_string(),
            dataset_sha256,
            dataset_bytes,
            replay_sha256,
            replay_bytes,
            stats,
            rollouts: records,
            training_submission: None,
        },
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_candidate_episode(
    chat: &reqwest::Client,
    environment: OpenEnvClient,
    inspection: OpenEnvInspection,
    adapter: Value,
    adapter_label: String,
    reset_payload: Value,
    seed: u64,
    group_index: usize,
    candidate_index: usize,
    options: &OpenEnvRolloutOptions,
) -> Result<CandidateRollout> {
    let (mut session, reset, capacity_retries) = connect_and_reset_with_capacity(
        &environment,
        &reset_payload,
        Duration::from_secs(options.capacity_wait_seconds),
    )
    .await?;
    anyhow::ensure!(
        !reset.done,
        "OpenEnv environment {} returned done=true from reset; a trainable rollout needs at least one model action",
        inspection.identity.metadata.name
    );
    let messages = initial_messages(&inspection, &reset)?;
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

    for step_index in 0..options.max_steps {
        let generation_seed = generation_seed(seed, candidate_index, step_index);
        let model_action = generate_model_action(
            chat,
            &options.kiln_url,
            &messages,
            &trajectory,
            &adapter,
            &adapter_label,
            generation_seed,
            options.max_action_tokens,
            options.temperature,
            options.thinking,
        )
        .await;
        let model_action = match model_action {
            Ok(action) => action,
            Err(ModelActionFailure::Invalid {
                message,
                raw,
                total_tokens,
                latency_ms,
            }) => {
                total_model_tokens = total_model_tokens.saturating_add(total_tokens);
                total_model_latency_ms += latency_ms;
                let raw = raw.unwrap_or_else(|| invalid_action_raw(&message));
                trajectory.push(action_segment(raw));
                trajectory.push(harness_error_segment(&json!({
                    "openenv_harness_error": {
                        "code": "INVALID_MODEL_ACTION",
                        "message": message
                    },
                    "done": true
                }))?);
                episode_return += options.protocol_error_reward;
                termination = OpenEnvEpisodeTerminationV1::InvalidModelAction;
                break;
            }
            Err(ModelActionFailure::Request(error)) => return Err(error),
        };
        total_model_tokens = total_model_tokens.saturating_add(model_action.total_tokens);
        total_model_latency_ms += model_action.latency_ms;
        trajectory.push(action_segment(model_action.raw));
        steps = steps.saturating_add(1);

        match session.step(&model_action.action).await {
            Ok(observation) => {
                episode_return += observation.reward.training_value();
                anyhow::ensure!(
                    episode_return.is_finite(),
                    "OpenEnv episode return became non-finite"
                );
                trajectory.push(observation_segment(&observation)?);
                replay_exchanges.push(OpenEnvReplayExchange {
                    step_index,
                    action: model_action.action,
                    result: OpenEnvReplayExchangeResult::Observation {
                        observation: observation.clone(),
                    },
                });
                if observation.done {
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
                trajectory.push(protocol_error_segment(&error, continued)?);
                replay_exchanges.push(OpenEnvReplayExchange {
                    step_index,
                    action: model_action.action,
                    result: OpenEnvReplayExchangeResult::ProtocolError { error, continued },
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
    let _ = session.close().await;

    let action_schema_sha256 = sha256_json(&inspection.schema.action)?;
    let reset_sha256 = sha256_json(&reset_payload)?;
    let terminal_done = matches!(termination, OpenEnvEpisodeTerminationV1::Done);
    let openenv = OpenEnvRolloutProvenanceV1::new(
        inspection.identity.metadata.name.clone(),
        inspection.identity.base_url.clone(),
        inspection.identity.openapi_version.clone(),
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
    .context("build OpenEnv rollout provenance")?;
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

    Ok(CandidateRollout {
        candidate_index,
        messages,
        reset_observation: reset,
        rollout,
        replay,
        record: OpenEnvRolloutRecord {
            group_index,
            candidate_index,
            environment_name: inspection.identity.metadata.name,
            environment_url: inspection.identity.base_url,
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
    })
}

#[allow(clippy::too_many_arguments)]
async fn generate_model_action(
    chat: &reqwest::Client,
    kiln_url: &str,
    messages: &[ChatMessage],
    trajectory: &[TurnSegment],
    adapter: &Value,
    adapter_label: &str,
    seed: u64,
    max_tokens: usize,
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
    let body = json!({
        "messages": request_messages,
        "adapter": adapter,
        "stream": false,
        "n": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {
            "enable_thinking": thinking
        }
    });
    let started = Instant::now();
    let response = chat
        .post(format!(
            "{}/v1/chat/completions",
            kiln_url.trim_end_matches('/')
        ))
        .header("x-kiln-client", "openenv")
        .json(&body)
        .send()
        .await
        .map_err(|error| {
            ModelActionFailure::Request(anyhow!(error).context(format!(
                "request OpenEnv action from Kiln using adapter {adapter_label}"
            )))
        })?;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = response.status();
    let response_body = read_kiln_json_bounded(response, "action generation")
        .await
        .map_err(ModelActionFailure::Request)?;
    if !status.is_success() {
        return Err(ModelActionFailure::Request(anyhow!(
            "Kiln action generation returned HTTP {status}: {}",
            serde_json::to_string(&response_body).unwrap_or_default()
        )));
    }
    let total_tokens = response_body
        .pointer("/usage/total_tokens")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or_default();
    let raw = response_body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| ModelActionFailure::Invalid {
            message: "Kiln response choices[0].message.content is missing or not text".to_string(),
            raw: None,
            total_tokens,
            latency_ms,
        })?
        .to_string();
    let action = parse_model_action(&raw).map_err(|message| ModelActionFailure::Invalid {
        message,
        raw: Some(raw.clone()),
        total_tokens,
        latency_ms,
    })?;
    Ok(ModelAction {
        raw,
        action,
        total_tokens,
        latency_ms,
    })
}

pub(crate) fn initial_messages(
    inspection: &OpenEnvInspection,
    reset: &OpenEnvObservation,
) -> Result<Vec<ChatMessage>> {
    let action_schema = serde_json::to_string(&inspection.schema.action)
        .context("serialize OpenEnv action schema")?;
    let observation =
        serde_json::to_string(reset).context("serialize complete OpenEnv reset observation")?;
    let metadata = &inspection.identity.metadata;
    Ok(vec![
        ChatMessage::new(
            "system",
            format!(
                "You are the policy acting in the OpenEnv reinforcement-learning environment {name:?}.\n\
                 Environment description: {description}\n\
                 At every turn, reply with exactly one JSON object that validates against this action schema:\n\
                 {action_schema}\n\
                 Do not use Markdown, commentary, or a code fence. The environment observation will follow.",
                name = metadata.name,
                description = metadata.description
            ),
        ),
        ChatMessage::new(
            "user",
            format!(
                "OpenEnv reset result (observation, reward, done, and optional metadata):\n{observation}\n\nChoose the next action as one JSON object."
            ),
        ),
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

fn observation_segment(observation: &impl Serialize) -> Result<TurnSegment> {
    Ok(TurnSegment {
        role: "tool".to_string(),
        content: serde_json::to_string(observation)
            .context("serialize OpenEnv observation trajectory segment")?,
        kind: TurnKind::Observation,
        tool_call_id: None,
        warning_prefix_len: None,
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

fn reset_payload(base: &Value, seed: u64) -> Result<Value> {
    let mut object = base
        .as_object()
        .cloned()
        .context("OpenEnv reset options must be a JSON object")?;
    object.insert("seed".to_string(), Value::from(seed));
    Ok(Value::Object(object))
}

fn read_reset_options(path: Option<&Path>) -> Result<Value> {
    let Some(path) = path else {
        return Ok(Value::Object(Map::new()));
    };
    let bytes = std::fs::read(path)
        .with_context(|| format!("read OpenEnv reset options {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse OpenEnv reset options {}", path.display()))?;
    anyhow::ensure!(
        value.is_object(),
        "OpenEnv reset options {} must contain one JSON object",
        path.display()
    );
    Ok(value)
}

fn validate_options(options: &OpenEnvRolloutOptions) -> Result<()> {
    anyhow::ensure!(
        !options.environment_urls.is_empty()
            && options.environment_urls.len() <= MAX_OPENENV_ENVIRONMENTS,
        "OpenEnv requires 1..={MAX_OPENENV_ENVIRONMENTS} --environment URL values"
    );
    anyhow::ensure!(
        options.groups > 0 && options.groups <= MAX_OPENENV_GROUPS,
        "OpenEnv groups must be in 1..={MAX_OPENENV_GROUPS}"
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
    output_adapter: &str,
    lora_rank: Option<usize>,
    auto_load: bool,
) -> Result<Value> {
    let mut config = json!({
        "output_name": output_adapter,
        "auto_load": auto_load,
        "behavior_policy": "no_importance_correction"
    });
    if let Some(rank) = lora_rank {
        config["lora_rank"] = Value::from(rank);
    }
    let body = json!({
        "groups": groups,
        "config": config
    });
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

fn write_openenv_outputs(
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

fn write_summary_atomic(path: &Path, summary: &OpenEnvRolloutSummary) -> Result<()> {
    let parent = output_parent(path)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create staged OpenEnv summary beside {}", path.display()))?;
    serde_json::to_writer_pretty(staged.as_file_mut(), summary)
        .context("serialize OpenEnv rollout summary")?;
    staged.as_file_mut().write_all(b"\n")?;
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
    let bytes = serde_json::to_vec(value).context("serialize value for SHA-256")?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_jsonl(groups: &[AgenticGroup]) -> Result<String> {
    let mut hasher = Sha256::new();
    for group in groups {
        let bytes = serde_json::to_vec(group).context("serialize OpenEnv group for SHA-256")?;
        hasher.update(bytes);
        hasher.update(b"\n");
    }
    Ok(format_digest(hasher.finalize().as_slice()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format_digest(Sha256::digest(bytes).as_slice())
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
mod tests {
    use super::*;
    use clap::Parser;

    use crate::cli::{Cli, Commands};

    #[test]
    fn cli_parses_inspect_rollout_and_train_as_first_class_commands() {
        let inspect = Cli::try_parse_from([
            "kiln",
            "openenv",
            "inspect",
            "--environment",
            "127.0.0.1:8990",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            inspect.command,
            Some(Commands::Openenv(OpenEnvCommands::Inspect {
                environment,
                json: true,
            })) if environment == "127.0.0.1:8990"
        ));

        let rollout = Cli::try_parse_from([
            "kiln",
            "openenv",
            "rollout",
            "--environment",
            "http://127.0.0.1:8000",
            "--environment",
            "http://127.0.0.1:8001",
            "--groups",
            "12",
            "--group-size",
            "6",
            "--thinking",
            "true",
        ])
        .unwrap();
        let Some(Commands::Openenv(OpenEnvCommands::Rollout { rollout })) = rollout.command else {
            panic!("expected openenv rollout command");
        };
        assert_eq!(rollout.environment_urls.len(), 2);
        assert_eq!(rollout.groups, 12);
        assert_eq!(rollout.group_size, 6);
        assert!(rollout.thinking);
        assert_eq!(rollout.protocol_error_reward, -1.0);
        assert_eq!(rollout.max_recoverable_errors, 3);
        assert_eq!(rollout.capacity_wait_seconds, 300);
        assert_eq!(rollout.replay_output, PathBuf::from("openenv.replay.json"));

        let train = Cli::try_parse_from([
            "kiln",
            "openenv",
            "train",
            "--environment",
            "http://127.0.0.1:8000",
            "--adapter",
            "agent-v1",
            "--output-adapter",
            "agent-v2",
            "--auto-load",
            "false",
        ])
        .unwrap();
        assert!(matches!(
            train.command,
            Some(Commands::Openenv(OpenEnvCommands::Train {
                output_adapter,
                auto_load: false,
                ..
            })) if output_adapter == "agent-v2"
        ));

        let verify = Cli::try_parse_from([
            "kiln",
            "openenv",
            "verify",
            "--summary",
            "batch.summary.json",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            verify.command,
            Some(Commands::Openenv(OpenEnvCommands::Verify {
                summary,
                json: true,
                ..
            })) if summary == PathBuf::from("batch.summary.json")
        ));

        let replay = Cli::try_parse_from([
            "kiln",
            "openenv",
            "replay",
            "--summary",
            "batch.summary.json",
            "--concurrency",
            "2",
        ])
        .unwrap();
        assert!(matches!(
            replay.command,
            Some(Commands::Openenv(OpenEnvCommands::Replay {
                concurrency: 2,
                capacity_wait_seconds: 300,
                ..
            }))
        ));

        assert!(
            Cli::try_parse_from(["kiln", "openenv", "rollout", "--groups", "2"]).is_err(),
            "an OpenEnv command without an environment must fail during parsing"
        );
    }

    #[test]
    fn action_parser_accepts_objects_and_rejects_other_shapes() {
        assert_eq!(
            parse_model_action(r#"{"answer":"B"}"#).unwrap(),
            json!({"answer": "B"})
        );
        assert_eq!(
            parse_model_action("```json\n{\"answer\":\"B\"}\n```").unwrap(),
            json!({"answer": "B"})
        );
        assert!(parse_model_action("answer B").is_err());
        assert!(parse_model_action("[1,2]").is_err());
        assert!(parse_model_action("{} trailing").is_err());
    }

    #[test]
    fn reset_seed_overrides_file_value_and_is_hashed_canonically() {
        let base = json!({"difficulty": 3, "seed": 999});
        let reset = reset_payload(&base, 7).unwrap();
        assert_eq!(reset, json!({"difficulty": 3, "seed": 7}));
        assert_eq!(sha256_json(&reset).unwrap().len(), "sha256:".len() + 64);
    }

    #[test]
    fn adapter_base_aliases_are_explicit_null() {
        for alias in ["base", "none", "null", "BASE"] {
            let adapter = parse_adapter_selection(alias);
            assert_eq!(adapter.request_value, Value::Null);
            assert_eq!(adapter.label, "base");
        }
        let named = parse_adapter_selection("agent-v2");
        assert_eq!(named.request_value, Value::String("agent-v2".into()));
    }

    #[test]
    fn generation_seeds_separate_candidates_and_steps_deterministically() {
        assert_eq!(generation_seed(42, 0, 0), 42);
        assert_eq!(generation_seed(42, 2, 3), generation_seed(42, 2, 3));
        assert_ne!(generation_seed(42, 1, 0), generation_seed(42, 0, 1));
    }
}

//! TOML configuration file support with typed validation and env var overrides.
//!
//! Configuration is loaded in this priority order (highest wins):
//! 1. Environment variables (`KILN_*`)
//! 2. TOML config file
//! 3. Built-in defaults
//!
//! The config file path is resolved as:
//! 1. Explicit path passed to `KilnConfig::load()`
//! 2. `KILN_CONFIG` environment variable
//! 3. `./kiln.toml` in the current working directory (if it exists)
//! 4. No file — use defaults only

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
pub use kiln_scheduler::DEFAULT_MAX_BATCH_TOKENS;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Smallest accepted per-connection HTTP `SO_SNDBUF` request.
pub const HTTP_SEND_BUFFER_MIN_BYTES: usize = 1024;
/// Largest accepted per-connection HTTP `SO_SNDBUF` request. This opt-in is
/// primarily for bounded transport/backpressure testing; allowing arbitrarily
/// large buffers would multiply memory use by every concurrent connection.
pub const HTTP_SEND_BUFFER_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Default continuous time a full streaming response channel may make no
/// delivery progress before the worker asks the actor to evict that request.
pub const DEFAULT_STREAM_STALL_GRACE_MS: u64 = 2_000;
/// Minimum stream-stall grace. This matches the delivery worker's fair retry
/// cadence, so every accepted value permits at least one bounded retry.
pub const STREAM_STALL_GRACE_MIN_MS: u64 = 10;
/// Maximum stream-stall grace. A stalled request retains its KV state and
/// decode slot while peers continue, so this remains a bounded safety valve.
pub const STREAM_STALL_GRACE_MAX_MS: u64 = DEFAULT_STREAM_STALL_GRACE_MS;

/// Default combined decode-plus-prefill token budget for one batching-actor
/// scheduling cycle. Decode rows consume one token each before prefill uses
/// the remainder.
pub const MAX_BATCH_TOKENS_MIN: usize = 2;
pub const MAX_BATCH_TOKENS_MAX: usize = 65_536;
/// Default prompt-token work allowed between decode cohorts. The stable
/// serving default is intentionally lower than the combined batch budget so a
/// long prompt cannot turn one actor cycle into a visible decode pause.
pub const DEFAULT_MAX_PREFILL_TOKENS_PER_CYCLE: usize = 64;
pub const MAX_PREFILL_TOKENS_PER_CYCLE_MIN: usize = 1;
pub const MAX_PREFILL_TOKENS_PER_CYCLE_MAX: usize = MAX_BATCH_TOKENS_MAX;
/// Strict startup selector for the serving-safety contract.
pub const SERVING_PROFILE_ENV: &str = "KILN_SERVING_PROFILE";

/// Provenance of a resolved startup configuration value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigValueSource {
    #[default]
    Default,
    ConfigFile,
    Environment,
}

impl fmt::Display for ConfigValueSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "default",
            Self::ConfigFile => "config_file",
            Self::Environment => "environment",
        })
    }
}

/// Process-lifetime serving policy.
///
/// The profile is immutable after startup. That keeps GPU ownership policy out
/// of individual requests and makes a restart the explicit boundary between
/// ordinary serving, development concurrency, and drained maintenance work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingProfile {
    /// Predictable inference. Dynamic physical memory operations, live graph
    /// capture, training GPU writers, and live adapter weight transitions are
    /// prohibited.
    #[default]
    Stable,
    /// Developer profile preserving concurrent inference/training and dynamic
    /// runtime mutation for controlled experiments.
    Experimental,
    /// Drained exclusive work. Inference admission is disabled while training,
    /// adapter activation, and physical memory maintenance are allowed.
    Maintenance,
}

impl ServingProfile {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "stable" => Ok(Self::Stable),
            "experimental" => Ok(Self::Experimental),
            "maintenance" => Ok(Self::Maintenance),
            _ => anyhow::bail!(
                "{label} must be one of stable, experimental, maintenance; got {raw:?}"
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Experimental => "experimental",
            Self::Maintenance => "maintenance",
        }
    }

    pub const fn runtime_policy(self) -> ServingRuntimePolicy {
        match self {
            Self::Stable => ServingRuntimePolicy {
                inference_admission: true,
                training_gpu_ownership: false,
                adapter_weight_transitions: false,
                dynamic_kv_resize: false,
                allocator_reclaim: false,
                live_graph_capture: false,
                exclusive_gpu_behavior: "reject",
            },
            Self::Experimental => ServingRuntimePolicy {
                inference_admission: true,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: true,
                exclusive_gpu_behavior: "writer_priority",
            },
            Self::Maintenance => ServingRuntimePolicy {
                inference_admission: false,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: false,
                exclusive_gpu_behavior: "inference_disabled_drain_then_exclusive",
            },
        }
    }
}

impl fmt::Display for ServingProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Fully resolved behavior derived only from [`ServingProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ServingRuntimePolicy {
    pub inference_admission: bool,
    pub training_gpu_ownership: bool,
    pub adapter_weight_transitions: bool,
    pub dynamic_kv_resize: bool,
    pub allocator_reclaim: bool,
    pub live_graph_capture: bool,
    pub exclusive_gpu_behavior: &'static str,
}

/// Operator-facing resolution report for the process-lifetime serving policy.
///
/// `source` identifies who selected `profile`. Every field in
/// `effective_policy` is derived solely from that profile, never from a
/// request, so all observability surfaces can publish one consistent contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ServingProfileDiagnostics {
    pub profile: ServingProfile,
    pub source: ConfigValueSource,
    pub immutable_after_startup: bool,
    pub request_overrides_allowed: bool,
    pub effective_policy_source: &'static str,
    pub effective_policy: ServingRuntimePolicy,
}

/// Validated serving profile plus the startup source that selected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingProfileSetting {
    profile: ServingProfile,
    source: ConfigValueSource,
}

impl ServingProfileSetting {
    /// Construct an already-resolved setting for embedders and tests that do
    /// not use [`KilnConfig::load`]. Production configuration should retain
    /// the source that actually selected the profile.
    pub const fn new(profile: ServingProfile, source: ConfigValueSource) -> Self {
        Self { profile, source }
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Ok(Self {
            profile: ServingProfile::parse(raw, SERVING_PROFILE_ENV)?,
            source: ConfigValueSource::Environment,
        })
    }

    pub const fn profile(self) -> ServingProfile {
        self.profile
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    pub const fn runtime_policy(self) -> ServingRuntimePolicy {
        self.profile.runtime_policy()
    }

    pub const fn diagnostics(self) -> ServingProfileDiagnostics {
        ServingProfileDiagnostics {
            profile: self.profile,
            source: self.source,
            immutable_after_startup: true,
            request_overrides_allowed: false,
            effective_policy_source: "serving_profile",
            effective_policy: self.runtime_policy(),
        }
    }
}

impl Default for ServingProfileSetting {
    fn default() -> Self {
        Self {
            profile: ServingProfile::Stable,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for ServingProfileSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.profile.as_str())
    }
}

impl<'de> Deserialize<'de> for ServingProfileSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let profile = ServingProfile::parse(&raw, "server.serving_profile")
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            profile,
            source: ConfigValueSource::ConfigFile,
        })
    }
}

/// Validated stream-stall grace plus the startup source that selected it.
///
/// The custom serde implementation keeps `server.stream_stall_grace_ms` an
/// ordinary TOML integer while distinguishing an explicit file value from the
/// built-in default. Environment resolution happens once in [`KilnConfig::load`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStallGrace {
    millis: u64,
    source: ConfigValueSource,
}

impl StreamStallGrace {
    fn new(millis: u64, source: ConfigValueSource) -> Result<Self> {
        validate_stream_stall_grace_ms(millis)?;
        Ok(Self { millis, source })
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        let millis = raw.trim().parse::<u64>().with_context(|| {
            format!(
                "KILN_STREAM_STALL_GRACE_MS must be a decimal integer in {}..={}, got {raw:?}",
                STREAM_STALL_GRACE_MIN_MS, STREAM_STALL_GRACE_MAX_MS
            )
        })?;
        Self::new(millis, ConfigValueSource::Environment)
            .context("invalid KILN_STREAM_STALL_GRACE_MS")
    }

    pub fn millis(self) -> u64 {
        self.millis
    }

    pub fn duration(self) -> Duration {
        Duration::from_millis(self.millis)
    }

    pub fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for StreamStallGrace {
    fn default() -> Self {
        Self {
            millis: DEFAULT_STREAM_STALL_GRACE_MS,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for StreamStallGrace {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.millis)
    }
}

impl<'de> Deserialize<'de> for StreamStallGrace {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Self::new(millis, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Validated batching token budget plus the startup source that selected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchTokenBudget {
    tokens: usize,
    source: ConfigValueSource,
}

impl BatchTokenBudget {
    pub(crate) fn new(tokens: usize, source: ConfigValueSource) -> Result<Self> {
        validate_max_batch_tokens(tokens)?;
        Ok(Self { tokens, source })
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        let tokens = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "KILN_MAX_BATCH_TOKENS must be a decimal integer in {}..={}, got {raw:?}",
                MAX_BATCH_TOKENS_MIN, MAX_BATCH_TOKENS_MAX
            )
        })?;
        Self::new(tokens, ConfigValueSource::Environment).context("invalid KILN_MAX_BATCH_TOKENS")
    }

    pub fn tokens(self) -> usize {
        self.tokens
    }

    pub fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for BatchTokenBudget {
    fn default() -> Self {
        Self {
            tokens: DEFAULT_MAX_BATCH_TOKENS,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for BatchTokenBudget {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.tokens as u64)
    }
}

impl<'de> Deserialize<'de> for BatchTokenBudget {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tokens = usize::deserialize(deserializer)?;
        Self::new(tokens, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Validated prompt-token ceiling per actor cycle plus startup provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillTokenBudget {
    tokens: usize,
    source: ConfigValueSource,
}

impl PrefillTokenBudget {
    pub(crate) fn new(tokens: usize, source: ConfigValueSource) -> Result<Self> {
        validate_max_prefill_tokens_per_cycle(tokens)?;
        Ok(Self { tokens, source })
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        let tokens = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "KILN_MAX_PREFILL_TOKENS_PER_CYCLE must be a decimal integer in {}..={}, got {raw:?}",
                MAX_PREFILL_TOKENS_PER_CYCLE_MIN, MAX_PREFILL_TOKENS_PER_CYCLE_MAX
            )
        })?;
        Self::new(tokens, ConfigValueSource::Environment)
            .context("invalid KILN_MAX_PREFILL_TOKENS_PER_CYCLE")
    }

    pub fn tokens(self) -> usize {
        self.tokens
    }

    pub fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for PrefillTokenBudget {
    fn default() -> Self {
        Self {
            tokens: DEFAULT_MAX_PREFILL_TOKENS_PER_CYCLE,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for PrefillTokenBudget {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.tokens as u64)
    }
}

impl<'de> Deserialize<'de> for PrefillTokenBudget {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tokens = usize::deserialize(deserializer)?;
        Self::new(tokens, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Top-level configuration for kiln.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct KilnConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub memory: MemoryConfig,
    pub training: TrainingConfig,
    pub logging: LoggingConfig,
    pub prefix_cache: PrefixCacheConfig,
    pub speculative: SpeculativeDecodingConfig,
    pub streaming_prefill: StreamingPrefillConfig,
    pub adapters: AdaptersConfig,
    /// Remote-teacher credentials. API clients refer to these entries by
    /// opaque id; only this server-owned configuration can name a secret
    /// environment variable or authorize the origin that receives it.
    pub teachers: TeachersConfig,
    /// Eval subsystem configuration. `None` means "use defaults" — the
    /// server still wires the eval API; only the on-disk suite registry
    /// location is left at its default `<adapter_dir>/.eval/suites`.
    #[serde(default)]
    pub eval: Option<EvalConfig>,
    /// Durable request/response log for the inference endpoints
    /// (mine→filter→train flywheel). See [`crate::request_log`].
    #[serde(default)]
    pub request_log: crate::request_log::RequestLogConfig,
    /// §10.6 self-improvement automation. `None` / omitted section means
    /// the weekly loop stays manual (`kiln self-improve`).
    #[serde(default)]
    pub agent: Option<AgentConfig>,
}

/// `[teachers]` remote-teacher trust configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TeachersConfig {
    /// Credential handles keyed by the id accepted by `POST /v1/teachers`.
    pub credentials: BTreeMap<String, TeacherCredentialConfig>,
}

/// One server-owned bearer credential and its exact authorized origin.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TeacherCredentialConfig {
    /// Canonical `scheme://host[:port]` origin. HTTPS is required unless the
    /// host is loopback.
    pub origin: String,
    /// Trusted environment-variable name. The secret itself is never stored in
    /// config, AppState, the teacher registry, responses, or receipts.
    pub api_key_env: String,
}

impl TeachersConfig {
    /// Resolve an API-visible credential handle to the trusted environment
    /// variable for one teacher URL. A handle can never be replayed to a
    /// different origin. Credential-free teachers are intentionally restricted
    /// to loopback.
    pub fn resolve_api_key_env(
        &self,
        credential_id: Option<&str>,
        teacher_url: &str,
    ) -> std::result::Result<Option<String>, String> {
        let requested_origin = canonical_teacher_origin(teacher_url)?;
        let Some(credential_id) = credential_id else {
            if teacher_url_is_loopback(teacher_url)? {
                return Ok(None);
            }
            return Err(
                "non-loopback remote teachers require a server-configured credential_id"
                    .to_string(),
            );
        };
        validate_teacher_credential_id(credential_id)?;
        let credential = self.credentials.get(credential_id).ok_or_else(|| {
            format!("teacher credential_id {credential_id:?} is not configured on this server")
        })?;
        credential.validate_definition(credential_id)?;
        if credential.origin != requested_origin {
            return Err(format!(
                "teacher credential_id {credential_id:?} is not authorized for origin {requested_origin:?}"
            ));
        }
        let secret_available =
            std::env::var(&credential.api_key_env).is_ok_and(|value| !value.trim().is_empty());
        if !secret_available {
            return Err(format!(
                "teacher credential_id {credential_id:?} is unavailable because its server-configured secret is missing or empty"
            ));
        }
        Ok(Some(credential.api_key_env.clone()))
    }

    fn validate(&self) -> Result<()> {
        for (credential_id, credential) in &self.credentials {
            validate_teacher_credential_id(credential_id)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!("invalid teachers.credentials.{credential_id} credential id")
                })?;
            credential
                .validate_definition(credential_id)
                .map_err(anyhow::Error::msg)?;
            let secret = std::env::var(&credential.api_key_env).with_context(|| {
                format!(
                    "teachers.credentials.{credential_id}.api_key_env names an environment variable that is not set"
                )
            })?;
            if secret.trim().is_empty() {
                anyhow::bail!(
                    "teachers.credentials.{credential_id}.api_key_env names an environment variable whose value is empty"
                );
            }
        }
        Ok(())
    }
}

impl TeacherCredentialConfig {
    fn validate_definition(&self, credential_id: &str) -> std::result::Result<(), String> {
        validate_teacher_api_key_env_name(&self.api_key_env).map_err(|message| {
            format!("teachers.credentials.{credential_id}.api_key_env {message}")
        })?;
        let canonical = canonical_teacher_origin(&self.origin)?;
        if self.origin != canonical {
            return Err(format!(
                "teachers.credentials.{credential_id}.origin must be the exact canonical origin {canonical:?}"
            ));
        }
        let parsed = reqwest::Url::parse(&self.origin)
            .map_err(|error| format!("invalid teacher credential origin: {error}"))?;
        if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(format!(
                "teachers.credentials.{credential_id}.origin must not contain a path, query, or fragment"
            ));
        }
        Ok(())
    }
}

/// Canonical origin used for exact credential scoping.
pub fn canonical_teacher_origin(url: &str) -> std::result::Result<String, String> {
    kiln_train::normalize_vllm_completions_url(url)?;
    let parsed = reqwest::Url::parse(url.trim())
        .map_err(|error| format!("remote teacher URL {url:?} is invalid: {error}"))?;
    Ok(parsed.origin().ascii_serialization())
}

fn teacher_url_is_loopback(url: &str) -> std::result::Result<bool, String> {
    let parsed = reqwest::Url::parse(url.trim())
        .map_err(|error| format!("remote teacher URL {url:?} is invalid: {error}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| "remote teacher URL must include a host".to_string())?;
    Ok(host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback()))
}

pub fn validate_teacher_credential_id(id: &str) -> std::result::Result<(), String> {
    let valid = !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if !valid {
        return Err(
            "teacher credential_id must be 1..=64 ASCII letters, digits, '_' or '-'".to_string(),
        );
    }
    Ok(())
}

fn validate_teacher_api_key_env_name(name: &str) -> std::result::Result<(), String> {
    let mut bytes = name.bytes();
    let valid_start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    let valid_rest = bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if !valid_start || !valid_rest || name.len() > 128 {
        return Err(
            "must be a 1..=128 character environment-variable name matching [A-Za-z_][A-Za-z0-9_]*"
                .to_string(),
        );
    }
    Ok(())
}

/// `[agent]` — the self-improvement flywheel scheduler and the
/// embedded pi run engine.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    /// Run the §10.6.2 self_improve loop every N hours (168 = weekly).
    /// Omit to disable. The first run fires one full interval after
    /// startup — never at boot, so a crash-looping server can't spam
    /// training jobs.
    #[serde(default)]
    pub self_improve_interval_hours: Option<u64>,
    /// The request the scheduler submits — same shape as
    /// POST /v1/agent/self_improve. Defaults: agent pi-coder-current,
    /// judge judge-pi-v1, CRISP on, no promotion gate. Set
    /// `post_eval = { suite = "...", min_accuracy = ... }` to make every
    /// scheduled round gated (§8.7: auto-load defers; failures demote).
    #[serde(default)]
    pub self_improve: Option<serde_json::Value>,
    /// Embedded pi runs executing at once (`POST /v1/agent/runs`).
    /// Queued runs start FIFO as slots free up. Inference batching
    /// interleaves the concurrent agents' requests.
    #[serde(default = "default_max_concurrent_runs")]
    pub max_concurrent_runs: usize,
    /// Wall-clock cap per embedded run, after which pi is aborted and
    /// the partial session is still indexed. Per-run `timeout_secs`
    /// overrides this.
    #[serde(default = "default_run_timeout_secs")]
    pub run_timeout_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            self_improve_interval_hours: None,
            self_improve: None,
            max_concurrent_runs: default_max_concurrent_runs(),
            run_timeout_secs: default_run_timeout_secs(),
        }
    }
}

fn default_max_concurrent_runs() -> usize {
    2
}

fn default_run_timeout_secs() -> u64 {
    900
}

/// Eval subsystem configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct EvalConfig {
    /// Directory where named eval suites are persisted (each as
    /// `<eval_dir>/<name>/suite.json`). `None` falls back to
    /// `<adapter_dir>/.eval/suites`.
    pub eval_dir: Option<std::path::PathBuf>,
    /// Maximum eval jobs allowed in the queue at once.
    pub max_queued_jobs: usize,
    /// Maximum tracked eval-job entries (terminal entries TTL out).
    pub max_tracked_jobs: usize,
    /// When set, every eval job that reaches a terminal state fires a
    /// POST to this URL with `{job_id, suite, adapters, status,
    /// headline_accuracy, gate_verdict, error, timestamp}` —
    /// fire-and-forget, same contract as `training.webhook_url`. Eval
    /// results become signals other systems can act on (CI gates,
    /// alerting, retrain triggers) instead of numbers in a dashboard.
    pub webhook_url: Option<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            webhook_url: None,
            eval_dir: None,
            max_queued_jobs: 32,
            max_tracked_jobs: 1024,
        }
    }
}

/// HTTP server settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Immutable serving-safety profile. Stable is the default; switching to
    /// experimental or maintenance requires an explicit file/env setting and
    /// a process restart.
    pub serving_profile: ServingProfileSetting,
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
    /// Optional `SO_SNDBUF` request applied to every accepted HTTP socket.
    /// Operating systems may round or account for bookkeeping differently;
    /// Kiln preflights the listener, normalizes platform accounting, and
    /// rejects ineffective application before advertising readiness.
    pub http_send_buffer_bytes: Option<usize>,
    /// How long a full per-request streaming response channel may make no
    /// delivery progress before the worker reports it for cancellation. The
    /// request retains KV and a decode slot during this grace; peer lanes and
    /// control commands continue independently.
    pub stream_stall_grace_ms: StreamStallGrace,
    /// Combined token budget for one production batching-actor cycle. Ready
    /// decode rows consume one token each; a resumable prefill may use only the
    /// remainder before the actor yields back to decode and control commands.
    pub max_batch_tokens: BatchTokenBudget,
    /// Independent prompt-token ceiling inside the combined actor-cycle budget.
    /// Decode rows reserve their token first; admission and resumable prefill
    /// share this remainder so a long prompt cannot monopolize the actor.
    pub max_prefill_tokens_per_cycle: PrefillTokenBudget,
    /// Enable deterministic eval-serving behavior for `kiln serve`.
    pub eval_mode: bool,
    /// Server-level default for chat-template `enable_thinking`. `None`
    /// preserves the model template default; requests can still override via
    /// `chat_template_kwargs.enable_thinking`.
    pub default_thinking_enabled: Option<bool>,
    /// Default maximum number of reasoning tokens before Kiln forces the
    /// model's `</think>` sequence. `None` leaves thinking unbounded.
    pub default_thinking_budget_tokens: Option<usize>,
    /// Default reasoning wall-clock budget in milliseconds. The clock begins
    /// at the first decode candidate, after queueing and prefill.
    pub default_thinking_budget_ms: Option<u64>,
    /// Copy separated `reasoning_content` into `content` for clients that do
    /// not understand reasoning channels. Requests can override this.
    pub fold_reasoning_into_content: bool,
    /// Include per-request performance counters in chat response metadata by
    /// default. Requests can override this with `include_performance`.
    pub chat_performance_metadata: bool,
    /// Include config hashes in chat response metadata by default. Requests can
    /// override this with `include_config_hashes`.
    pub chat_config_hash_metadata: bool,
    /// Emit a structured warning when a chat completion takes at least this
    /// many seconds. Set to 0 to disable.
    pub slow_request_warn_secs: u64,
    pub shutdown_timeout_secs: u64,
}

/// Model and tokenizer paths.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    pub path: Option<String>,
    pub model_id: String,
    pub tokenizer_path: Option<String>,
    pub adapter_dir: Option<String>,
    /// Parent directory for Kiln's private immutable model snapshot. When
    /// omitted, Kiln first tries beside the model and then the system temp
    /// directory. `KILN_MODEL_SNAPSHOT_DIR` is the explicit environment
    /// override.
    pub snapshot_dir: Option<String>,
    /// Override the string exposed at `/v1/models` and echoed in chat completion responses.
    /// When `None`, derived from `model_id` by stripping up to the last `/`.
    pub served_model_id: Option<String>,
}

/// Built-in runtime defaults for the only supported kiln model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModelDefaultsProfile {
    /// Human-readable profile name emitted in startup logs and diagnostics.
    pub name: &'static str,
    /// Canonical HuggingFace model identifier for this profile.
    pub canonical_model_id: &'static str,
    /// Canonical served model id exposed when no override is configured.
    pub canonical_served_model_id: &'static str,
    /// Server-level thinking default for ordinary serving. `None` preserves the
    /// model chat template default; explicit config/env/request values still win.
    pub server_default_thinking_enabled: Option<bool>,
    /// The official Qwen3.5-4B chat template starts assistant turns in thinking
    /// mode unless `enable_thinking=false` is supplied.
    pub template_default_thinking_enabled: bool,
    /// Eval mode should produce deterministic final-content answers for
    /// tool-agent loops unless a request explicitly opts into thinking.
    pub eval_mode_default_thinking_enabled: bool,
    /// How the adapter directory is resolved for this model profile.
    pub adapter_dir_policy: &'static str,
    /// Chat template sources accepted by startup loading, in preference order.
    pub chat_template_policy: &'static str,
    /// Whether `chat_template_kwargs.enable_thinking` is a supported template kwarg.
    pub supports_enable_thinking_kwarg: bool,
    /// Whether the bundled/official template supports OpenAI-style tool calls.
    pub supports_tool_chat_template: bool,
}

impl ModelDefaultsProfile {
    /// Qwen3.5-4B is kiln's canonical profile.
    pub const fn qwen3_5_4b() -> Self {
        Self {
            name: "Qwen3.5-4B",
            canonical_model_id: "Qwen/Qwen3.5-4B",
            canonical_served_model_id: "Qwen3.5-4B",
            server_default_thinking_enabled: None,
            template_default_thinking_enabled: true,
            eval_mode_default_thinking_enabled: false,
            adapter_dir_policy: "explicit model.adapter_dir, otherwise <model_path>/adapters",
            chat_template_policy: "prefer chat_template.jinja, fallback to tokenizer_config.json chat_template",
            supports_enable_thinking_kwarg: true,
            supports_tool_chat_template: true,
        }
    }

    /// Resolve the adapter directory according to this profile.
    pub fn resolve_adapter_dir(
        &self,
        configured_adapter_dir: Option<&str>,
        model_path: &str,
    ) -> PathBuf {
        configured_adapter_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(model_path).join("adapters"))
    }
}

impl ModelConfig {
    /// Resolve the served model identifier.
    ///
    /// Returns the explicit `served_model_id` override when set; otherwise derives
    /// it from `model_id` by stripping everything up to and including the last `/`.
    pub fn effective_served_model_id(&self) -> String {
        if let Some(ref id) = self.served_model_id {
            return id.clone();
        }
        self.model_id
            .rsplit('/')
            .next()
            .unwrap_or(&self.model_id)
            .to_string()
    }

    /// Return the active built-in defaults profile.
    ///
    /// Kiln deliberately supports Qwen3.5-4B, so every server boot uses the
    /// Qwen3.5-4B profile even when an operator overrides display identifiers.
    pub fn defaults_profile(&self) -> ModelDefaultsProfile {
        ModelDefaultsProfile::qwen3_5_4b()
    }
}

/// GPU memory allocation settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub num_blocks: Option<usize>,
    pub gpu_memory_gb: Option<f64>,
    pub inference_memory_fraction: f64,
    pub training_memory_gb: Option<f64>,
    /// Enable FP8 (E4M3FN) quantization for KV cache, halving memory usage.
    /// When enabled, K/V values are stored as 8-bit floats with per-tensor scaling.
    /// Default: false
    pub kv_cache_fp8: bool,
    /// Enable CUDA graph capture/replay for decode steps.
    /// Eliminates per-step kernel launch overhead for ~10-15% decode speedup.
    /// Automatically disabled on non-CUDA devices.
    /// Default: true
    pub cuda_graphs: bool,
}

/// Training-specific settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub grad_checkpoint_segments: Option<usize>,
    pub no_grad_checkpoint: bool,
    /// Save adapter weights every N training steps during a job.
    /// Per-job config overrides this. None = only save at the end.
    pub checkpoint_interval: Option<usize>,
    /// HTTP(S) URL to POST a JSON notification to when a training job
    /// (SFT or GRPO) completes or fails.
    ///
    /// When `None` (the default), no webhook is fired. When set, a
    /// fire-and-forget POST is sent with a 5-second timeout after the
    /// job's terminal state is recorded. Webhook failures are logged
    /// but never propagate back into the training job's outcome — a
    /// successful training job stays "completed" even if the webhook
    /// POST fails.
    ///
    /// Payload (Content-Type: application/json):
    /// ```json
    /// {
    ///   "job_id": "<uuid>",
    ///   "job_type": "sft" | "grpo",
    ///   "status": "completed" | "failed",
    ///   "adapter_name": "<name>",
    ///   "adapter_path": "<path or null>",
    ///   "error": "<message or null>",
    ///   "timestamp": "<RFC3339>"
    /// }
    /// ```
    ///
    /// Override via `KILN_TRAINING_WEBHOOK_URL`. To clear a TOML-set
    /// URL via env, set the variable to the empty string.
    pub webhook_url: Option<String>,
    /// Maximum number of training jobs that may sit in the queue at once.
    /// Submissions to `/v1/train/sft` and `/v1/train/grpo` while the queue
    /// is at this cap return HTTP 503 with a `Retry-After: 30` header
    /// instead of growing the in-memory queue without bound.
    /// Override via `KILN_TRAINING_MAX_QUEUED_JOBS`. Default: 32.
    pub max_queued_jobs: usize,
    /// Maximum number of tracked training jobs (queued, running, completed,
    /// or failed) that may live in the in-memory tracking map at once.
    /// Submissions while the tracking map is at this cap return HTTP 503
    /// with `Retry-After: 30` and the `training_tracked_full` error code.
    /// The training worker continuously evicts terminal entries older than
    /// `tracked_job_ttl_secs`, so a healthy server will rarely hit this
    /// cap. Override via `KILN_TRAINING_MAX_TRACKED_JOBS`. Default: 1024.
    pub max_tracked_jobs: usize,
    /// TTL in seconds for tracked training jobs in the `Completed` or
    /// `Failed` state. The training worker periodically removes terminal
    /// entries whose `finished_at` timestamp is older than this many
    /// seconds, bounding the steady-state size of the tracking map.
    /// Active jobs (`Queued` / `Running`) are never GC'd, regardless of
    /// age. Override via `KILN_TRAINING_TRACKED_JOB_TTL_SECS`. Default:
    /// 604800 (7 days) — long enough that the /ui still shows last
    /// week's runs, while `max_tracked_jobs` (default 1024) still bounds
    /// memory.
    pub tracked_job_ttl_secs: u64,
}

/// Logging settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

/// Prefix caching settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct PrefixCacheConfig {
    /// Enable prefix caching for shared prompt prefixes (default: true).
    /// When enabled, KV cache blocks for shared prefixes are reused across requests.
    pub enabled: bool,
    /// Maximum number of KV cache blocks the prefix cache may retain.
    /// Default: 50% of total blocks. Omit to use the default.
    pub max_blocks: Option<usize>,
    /// Maximum number of real-backend prefix entries to retain.
    /// Each entry owns a GDN linear-attention state snapshot in addition to
    /// KV blocks, so this cap prevents sustained unique-prompt traffic from
    /// accumulating unbounded device state memory. Default is memory-tiered.
    pub max_entries: Option<usize>,
}

/// Which speculative-decoding method to use when `enabled = true`.
///
/// - `Off` — no spec decoding, one token per step.
/// - `SkipLayer` — self-speculative using the first `draft_layers` of the main
///   model as a lightweight draft. Works on any checkpoint; kept as fallback
///   and A/B baseline.
/// - `Mtp` — native Multi-Token Prediction using the model's pretrained MTP
///   heads. Requires the checkpoint to contain `mtp.*` tensors (Qwen3.5-4B
///   has one MTP layer, k=1).
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpecMethod {
    Off,
    SkipLayer,
    Mtp,
}

impl Default for SpecMethod {
    fn default() -> Self {
        Self::Off
    }
}

impl SpecMethod {
    /// Parse from an env-var string. Case-insensitive; accepts common aliases.
    /// Returns `None` for unknown values so the caller can warn and fall back.
    pub fn parse_env(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" | "false" => Some(Self::Off),
            "skip_layer" | "skiplayer" | "skip-layer" | "self" => Some(Self::SkipLayer),
            "mtp" | "native_mtp" | "native-mtp" => Some(Self::Mtp),
            _ => None,
        }
    }
}

/// Speculative decoding settings.
///
/// Two implementations coexist:
///   * `SkipLayer` — the first `draft_layers` of the main model act as the
///     draft. Works on any checkpoint.
///   * `Mtp` — native MTP heads shipped with the checkpoint (Qwen3.5-4B k=1).
///     Requires `mtp.*` tensors in the weights.
///
/// `method` selects which path is active when `enabled = true`. For backward
/// compatibility, setting `enabled = true` with `method = Off` falls back to
/// `SkipLayer`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SpeculativeDecodingConfig {
    /// Enable speculative decoding (default: false).
    pub enabled: bool,
    /// Which speculative-decoding method to use. Default: `Off`.
    pub method: SpecMethod,
    /// Number of tokens the draft proposes per step (default: 256).
    /// Ignored by `Mtp` when the checkpoint has fewer MTP layers than this.
    pub num_speculative_tokens: usize,
    /// Number of layers to use for the `SkipLayer` draft (default: 8).
    pub draft_layers: usize,
}

/// Streaming/tiled prefill settings.
///
/// When enabled, long-context prefill iterates over the sequence in tiles of
/// `tile_tokens` tokens, carrying O(1) GDN recurrent state across tile
/// boundaries and writing full-attention K/V into the paged cache per tile.
/// This caps peak activation memory so that production-shaped 8k+ token
/// CUDA prefills and ≥65k-token long prefills fit on a 48 GiB A6000.
///
/// `tile_tokens` must be a positive multiple of 64 (the GDN chunk size).
///
/// Dispatch is driven by reading these environment variables directly from
/// `kiln-model` helpers; this struct is the documentation / TOML-config
/// mirror. The generic config default keeps streaming OFF unless explicitly set,
/// while runtime device policy enables streaming by default for CUDA prompts at
/// 8k+ tokens and Metal prompts after device-specific thresholds.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct StreamingPrefillConfig {
    /// Force tiled/streaming prefill on through config/env. Runtime device
    /// policy may still enable it for long CUDA/Metal prompts when unset.
    pub enabled: bool,
    /// Tile size in tokens (generic default: 8192). Must be a positive
    /// multiple of 64.
    pub tile_tokens: usize,
    /// On the final tile, compute the LM head only for the last row instead
    /// of the full hidden state. Safe for inference because RMSNorm is
    /// per-position. Default: true.
    pub last_token_lm_head: bool,
}

/// Adapter-storage settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AdaptersConfig {
    /// Maximum total size in bytes for `adapter_dir/` (excluding the
    /// `.upload-tmp-*/` staging dirs and the `.composed/<hash>/` cache —
    /// those are bounded by separate limits). Uploads to
    /// `POST /v1/adapters/upload` are rejected when the new adapter would
    /// push total finalized adapter bytes over this cap.
    ///
    /// `None` disables the cap entirely (operator opts out). The default
    /// is 100 GiB, which is large enough to hold dozens of typical LoRA
    /// adapters but small enough to catch a runaway upload loop on a
    /// home/dev box before it fills the disk. Combined with the existing
    /// per-request 4 GiB extracted-bytes limit (`ADAPTER_EXTRACT_BYTES_LIMIT`
    /// in `api/adapters.rs`), this closes the §8 disk-exhaustion finding
    /// from the v0.1 security audit.
    ///
    /// Override via `KILN_ADAPTERS_MAX_DISK_BYTES`. Set to `0` via env to
    /// disable the cap (operator-opt-out shorthand).
    pub max_disk_bytes: Option<u64>,
    /// Maximum total bytes occupied by the on-disk composed-adapter cache
    /// at `adapter_dir/.composed/<hash>/`. Each unique `(name, scale)`
    /// permutation of `adapters: [...]` on `/v1/chat/completions` writes a
    /// new entry; without a cap, a request loop with random scales fills
    /// the disk. After a successful synthesize the oldest entries (by
    /// directory mtime) are evicted until the total drops below this cap.
    /// `None` disables the byte cap (entry cap may still trigger).
    ///
    /// Default: 10 GiB. Closes the §8 / roadmap item 8 finding from the
    /// v0.1 security audit (paired with `composed_cache_max_entries`).
    /// Override via `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES`. Set to `0`
    /// via env to disable the cap (operator-opt-out shorthand).
    pub composed_cache_max_bytes: Option<u64>,
    /// Maximum number of entries (subdirectories) in the composed-adapter
    /// cache at `adapter_dir/.composed/`. Cheap independent guard against
    /// pathological permutation loops with many tiny adapters that would
    /// not blow past the byte cap quickly. Eviction order matches the
    /// byte cap (oldest mtime first). `None` disables the entry cap (byte
    /// cap may still trigger).
    ///
    /// Default: 64. Override via `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES`.
    /// Set to `0` via env to disable the cap (operator-opt-out shorthand).
    pub composed_cache_max_entries: Option<u64>,
}

// --- Defaults ---

impl Default for KilnConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            memory: MemoryConfig::default(),
            training: TrainingConfig::default(),
            logging: LoggingConfig::default(),
            prefix_cache: PrefixCacheConfig::default(),
            speculative: SpeculativeDecodingConfig::default(),
            streaming_prefill: StreamingPrefillConfig::default(),
            adapters: AdaptersConfig::default(),
            teachers: TeachersConfig::default(),
            eval: None,
            request_log: crate::request_log::RequestLogConfig::default(),
            agent: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            serving_profile: ServingProfileSetting::default(),
            host: "127.0.0.1".into(),
            port: 8420,
            request_timeout_secs: 600,
            http_send_buffer_bytes: None,
            stream_stall_grace_ms: StreamStallGrace::default(),
            max_batch_tokens: BatchTokenBudget::default(),
            max_prefill_tokens_per_cycle: PrefillTokenBudget::default(),
            eval_mode: false,
            default_thinking_enabled: None,
            default_thinking_budget_tokens: None,
            default_thinking_budget_ms: None,
            fold_reasoning_into_content: false,
            chat_performance_metadata: false,
            chat_config_hash_metadata: false,
            slow_request_warn_secs: 30,
            // Hard ceiling on graceful-shutdown drain. With proactive
            // engine.stop() on signal, real draining typically completes
            // in under a second, so anything beyond a few seconds is
            // just a safety net before forcing exit.
            shutdown_timeout_secs: 5,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: None,
            model_id: "Qwen/Qwen3.5-4B".into(),
            tokenizer_path: None,
            adapter_dir: None,
            snapshot_dir: None,
            served_model_id: None,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            num_blocks: None,
            gpu_memory_gb: None,
            inference_memory_fraction: 0.7,
            training_memory_gb: None,
            kv_cache_fp8: false,
            // Default-ON (#34): CUDA graph capture/replay is now bit-identical to
            // eager decode. BUG2 (the replay divergence) was the captured graph
            // filling its RoPE cos/sin tables with host CPU cos/sin while eager
            // uses GPU kt_cos/kt_sin — now both compute on-device. Verified
            // bit-identical over 512-token decodes (BF16 + W4A16, multiple prompts).
            cuda_graphs: true,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            grad_checkpoint_segments: None,
            no_grad_checkpoint: false,
            checkpoint_interval: None,
            webhook_url: None,
            max_queued_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl_secs: 604_800,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "auto".into(),
        }
    }
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_blocks: None,
            max_entries: None,
        }
    }
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: SpecMethod::Off,
            num_speculative_tokens: 256,
            draft_layers: 8,
        }
    }
}

impl SpeculativeDecodingConfig {
    /// Build the speculative config from defaults plus the KILN_SPEC_* env vars.
    ///
    /// This mirrors the desktop app's control surface, which drives kiln
    /// through env vars rather than CLI flags.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env_overrides();
        cfg
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("KILN_SPEC_ENABLED") {
            self.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_SPEC_METHOD") {
            if let Some(m) = SpecMethod::parse_env(&v) {
                self.method = m;
                // Asking for a method IS asking for speculative decoding:
                // `KILN_SPEC_METHOD=mtp` alone used to be a silent no-op
                // (`effective_method()` returns Off unless `enabled`),
                // which cost an operator-validation cycle to discover.
                // An explicit KILN_SPEC_ENABLED still wins (it is read
                // first above and re-checked here for ordering safety).
                if !matches!(m, SpecMethod::Off) && std::env::var("KILN_SPEC_ENABLED").is_err() {
                    self.enabled = true;
                }
            } else {
                tracing::warn!(
                    "ignoring unknown KILN_SPEC_METHOD='{}' (expected off|skip_layer|mtp)",
                    v
                );
            }
        }
        if let Ok(v) = std::env::var("KILN_SPEC_NUM_TOKENS") {
            if let Ok(n) = v.parse() {
                self.num_speculative_tokens = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_SPEC_DRAFT_LAYERS") {
            if let Ok(n) = v.parse() {
                self.draft_layers = n;
            }
        }
    }

    /// Resolve the effective speculative-decoding method.
    ///
    /// Returns `Off` if the feature is disabled; otherwise returns the
    /// configured `method`, falling back to `SkipLayer` for backward
    /// compatibility when `enabled = true` but `method = Off` (older configs
    /// and older env-var usage that predate `KILN_SPEC_METHOD`).
    pub fn effective_method(&self) -> SpecMethod {
        if !self.enabled {
            return SpecMethod::Off;
        }
        match self.method {
            SpecMethod::Off => SpecMethod::SkipLayer,
            m => m,
        }
    }
}

impl Default for StreamingPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tile_tokens: 8192,
            last_token_lm_head: true,
        }
    }
}

impl Default for AdaptersConfig {
    fn default() -> Self {
        Self {
            // 100 GiB. Large enough for many real adapters, small enough
            // to catch a runaway upload loop before it fills the disk.
            max_disk_bytes: Some(100 * 1024u64.pow(3)),
            // 10 GiB byte cap, 64 entry cap. Matches the v0.1 audit
            // recommendation (§8) and is independent of the upload cap.
            composed_cache_max_bytes: Some(10 * 1024u64.pow(3)),
            composed_cache_max_entries: Some(64),
        }
    }
}

// --- Loading and validation ---

impl KilnConfig {
    /// Load configuration from an optional file path, then apply env var overrides.
    ///
    /// Resolution order for the file path:
    /// 1. `path` argument (if `Some`)
    /// 2. `KILN_CONFIG` env var
    /// 3. `./kiln.toml` (only if it exists)
    /// 4. No file — defaults only
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = path
            .map(String::from)
            .or_else(|| std::env::var("KILN_CONFIG").ok());

        let mut config = if let Some(ref p) = config_path {
            let contents = std::fs::read_to_string(p)
                .with_context(|| format!("failed to read config file: {p}"))?;
            toml::from_str(&contents)
                .with_context(|| format!("failed to parse config file: {p}"))?
        } else if Path::new("kiln.toml").exists() {
            let contents =
                std::fs::read_to_string("kiln.toml").context("failed to read kiln.toml")?;
            toml::from_str(&contents).context("failed to parse kiln.toml")?
        } else {
            Self::default()
        };

        config.apply_env_overrides();
        config.apply_http_send_buffer_env_override()?;
        config.apply_serving_profile_env_override()?;
        config.apply_stream_stall_grace_env_override()?;
        config.apply_max_batch_tokens_env_override()?;
        config.apply_max_prefill_tokens_per_cycle_env_override()?;
        config.request_log.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Resolve the process-lifetime serving profile strictly. Unlike legacy
    /// permissive overrides, a present malformed value is always fatal.
    fn apply_serving_profile_env_override(&mut self) -> Result<()> {
        let raw = match std::env::var(SERVING_PROFILE_ENV) {
            Ok(raw) => raw,
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("{SERVING_PROFILE_ENV} must be valid UTF-8")
            }
        };
        self.apply_serving_profile_env_value(Some(&raw))
    }

    fn apply_serving_profile_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.serving_profile = ServingProfileSetting::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Override config values with KILN_* environment variables (if set).
    fn apply_env_overrides(&mut self) {
        // Server
        if let Ok(v) = std::env::var("KILN_HOST") {
            self.server.host = v;
        }
        if let Ok(v) = std::env::var("KILN_PORT") {
            if let Ok(p) = v.parse() {
                self.server.port = p;
            }
        }
        if let Ok(v) = std::env::var("KILN_REQUEST_TIMEOUT_SECS") {
            if let Ok(s) = v.parse() {
                self.server.request_timeout_secs = s;
            }
        }
        if let Ok(v) = std::env::var("KILN_EVAL_MODE") {
            self.server.eval_mode = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if std::env::var("KILN_DEFAULT_NO_THINK").is_ok() {
            self.server.default_thinking_enabled = Some(false);
        }
        if let Ok(v) = std::env::var("KILN_DEFAULT_THINKING_ENABLED")
            && let Some(enabled) = parse_bool_env(&v)
        {
            self.server.default_thinking_enabled = Some(enabled);
        }
        if let Ok(v) = std::env::var("KILN_DEFAULT_THINKING_BUDGET_TOKENS") {
            match parse_optional_usize_env(&v) {
                Ok(value) => self.server.default_thinking_budget_tokens = value,
                Err(()) => tracing::warn!(
                    value = %v,
                    "ignoring invalid KILN_DEFAULT_THINKING_BUDGET_TOKENS; expected a non-negative integer or 'unlimited'"
                ),
            }
        }
        if let Ok(v) = std::env::var("KILN_DEFAULT_THINKING_BUDGET_MS") {
            match parse_optional_u64_env(&v) {
                Ok(value) => self.server.default_thinking_budget_ms = value,
                Err(()) => tracing::warn!(
                    value = %v,
                    "ignoring invalid KILN_DEFAULT_THINKING_BUDGET_MS; expected a non-negative integer or 'unlimited'"
                ),
            }
        }
        if let Ok(v) = std::env::var("KILN_FOLD_REASONING_INTO_CONTENT")
            && let Some(enabled) = parse_bool_env(&v)
        {
            self.server.fold_reasoning_into_content = enabled;
        }
        if let Ok(v) = std::env::var("KILN_CHAT_PERFORMANCE_METADATA")
            && let Some(enabled) = parse_bool_env(&v)
        {
            self.server.chat_performance_metadata = enabled;
        }
        if let Ok(v) = std::env::var("KILN_CHAT_CONFIG_HASH_METADATA")
            && let Some(enabled) = parse_bool_env(&v)
        {
            self.server.chat_config_hash_metadata = enabled;
        }
        if let Ok(v) = std::env::var("KILN_SLOW_REQUEST_WARN_SECS") {
            if let Ok(s) = v.parse() {
                self.server.slow_request_warn_secs = s;
            }
        }
        if let Ok(v) = std::env::var("KILN_SHUTDOWN_TIMEOUT_SECS") {
            if let Ok(s) = v.parse() {
                self.server.shutdown_timeout_secs = s;
            }
        }

        // Model
        if let Ok(v) = std::env::var("KILN_MODEL_PATH") {
            self.model.path = Some(v);
        }
        if let Ok(v) = std::env::var("KILN_MODEL_ID") {
            self.model.model_id = v;
        }
        if let Ok(v) = std::env::var("KILN_TOKENIZER_PATH") {
            self.model.tokenizer_path = Some(v);
        }
        if let Ok(v) = std::env::var("KILN_ADAPTER_DIR") {
            self.model.adapter_dir = Some(v);
        }
        if let Ok(v) = std::env::var("KILN_MODEL_SNAPSHOT_DIR") {
            self.model.snapshot_dir = if v.trim().is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = std::env::var("KILN_SERVED_MODEL_ID") {
            self.model.served_model_id = Some(v);
        }

        // Memory
        if let Ok(v) = std::env::var("KILN_NUM_BLOCKS") {
            if let Ok(n) = v.parse() {
                self.memory.num_blocks = Some(n);
            }
        }
        if let Ok(v) = std::env::var("KILN_GPU_MEMORY_GB") {
            if let Ok(g) = v.parse() {
                self.memory.gpu_memory_gb = Some(g);
            }
        }
        if let Ok(v) = std::env::var("KILN_INFERENCE_MEMORY_FRACTION") {
            if let Ok(f) = v.parse::<f64>() {
                self.memory.inference_memory_fraction = f;
            }
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_MEMORY_GB") {
            if let Ok(g) = v.parse() {
                self.memory.training_memory_gb = Some(g);
            }
        }
        if let Ok(v) = std::env::var("KILN_KV_CACHE_FP8") {
            self.memory.kv_cache_fp8 = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_CUDA_GRAPHS") {
            self.memory.cuda_graphs = v == "1" || v.eq_ignore_ascii_case("true");
        }

        // Training
        if let Ok(v) = std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS") {
            if let Ok(s) = v.parse() {
                self.training.grad_checkpoint_segments = Some(s);
            }
        }
        if let Ok(v) = std::env::var("KILN_NO_GRAD_CHECKPOINT") {
            self.training.no_grad_checkpoint = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_CHECKPOINT_INTERVAL") {
            if let Ok(n) = v.parse() {
                self.training.checkpoint_interval = Some(n);
            }
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_WEBHOOK_URL") {
            // Empty string explicitly clears any TOML-set URL.
            self.training.webhook_url = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_MAX_QUEUED_JOBS") {
            if let Ok(n) = v.parse::<usize>() {
                self.training.max_queued_jobs = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_MAX_TRACKED_JOBS") {
            if let Ok(n) = v.parse::<usize>() {
                self.training.max_tracked_jobs = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_TRACKED_JOB_TTL_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                self.training.tracked_job_ttl_secs = n;
            }
        }

        // Logging
        if let Ok(v) = std::env::var("KILN_LOG_LEVEL") {
            self.logging.level = v;
        }
        if let Ok(v) = std::env::var("KILN_LOG_FORMAT") {
            self.logging.format = v;
        }

        // Prefix cache
        if let Ok(v) = std::env::var("KILN_PREFIX_CACHE_ENABLED") {
            self.prefix_cache.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_PREFIX_CACHE_MAX_BLOCKS") {
            if let Ok(n) = v.parse() {
                self.prefix_cache.max_blocks = Some(n);
            }
        }
        if let Ok(v) = std::env::var("KILN_PREFIX_CACHE_MAX_ENTRIES") {
            if let Ok(n) = v.parse() {
                self.prefix_cache.max_entries = Some(n);
            }
        }

        // Speculative decoding
        if let Ok(v) = std::env::var("KILN_SPEC_ENABLED") {
            self.speculative.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_SPEC_METHOD") {
            if let Some(m) = SpecMethod::parse_env(&v) {
                self.speculative.method = m;
            } else {
                tracing::warn!(
                    "ignoring unknown KILN_SPEC_METHOD='{}' (expected off|skip_layer|mtp)",
                    v
                );
            }
        }
        if let Ok(v) = std::env::var("KILN_SPEC_NUM_TOKENS") {
            if let Ok(n) = v.parse() {
                self.speculative.num_speculative_tokens = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_SPEC_DRAFT_LAYERS") {
            if let Ok(n) = v.parse() {
                self.speculative.draft_layers = n;
            }
        }

        // Adapters
        if let Ok(v) = std::env::var("KILN_ADAPTERS_MAX_DISK_BYTES") {
            // `0` is the operator-opt-out shorthand: disable the cap.
            // Empty string also clears any TOML-set cap.
            let trimmed = v.trim();
            if trimmed.is_empty() {
                self.adapters.max_disk_bytes = None;
            } else if let Ok(n) = trimmed.parse::<u64>() {
                self.adapters.max_disk_bytes = if n == 0 { None } else { Some(n) };
            }
        }
        if let Ok(v) = std::env::var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES") {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                self.adapters.composed_cache_max_bytes = None;
            } else if let Ok(n) = trimmed.parse::<u64>() {
                self.adapters.composed_cache_max_bytes = if n == 0 { None } else { Some(n) };
            }
        }
        if let Ok(v) = std::env::var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES") {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                self.adapters.composed_cache_max_entries = None;
            } else if let Ok(n) = trimmed.parse::<u64>() {
                self.adapters.composed_cache_max_entries = if n == 0 { None } else { Some(n) };
            }
        }

        // Streaming/tiled prefill
        if let Ok(v) = std::env::var("KILN_STREAMING_PREFILL") {
            self.streaming_prefill.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_STREAMING_TILE_TOKENS") {
            if let Ok(n) = v.parse() {
                self.streaming_prefill.tile_tokens = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_STREAMING_LAST_TOKEN_LM_HEAD") {
            self.streaming_prefill.last_token_lm_head =
                !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no");
        }
    }

    /// Apply the accepted-socket send-buffer override strictly. Unlike legacy
    /// numeric overrides, a present but malformed value is a startup error.
    fn apply_http_send_buffer_env_override(&mut self) -> Result<()> {
        let raw = match std::env::var("KILN_HTTP_SEND_BUFFER_BYTES") {
            Ok(raw) => raw,
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("KILN_HTTP_SEND_BUFFER_BYTES must be valid UTF-8 decimal bytes")
            }
        };
        let value = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "KILN_HTTP_SEND_BUFFER_BYTES must be a decimal integer in {}..={}, got {raw:?}",
                HTTP_SEND_BUFFER_MIN_BYTES, HTTP_SEND_BUFFER_MAX_BYTES
            )
        })?;
        validate_http_send_buffer_bytes(value).context("invalid KILN_HTTP_SEND_BUFFER_BYTES")?;
        self.server.http_send_buffer_bytes = Some(value);
        Ok(())
    }

    /// Resolve the batching stream-stall override strictly at startup. The
    /// actor never reads process environment and receives only this typed value.
    fn apply_stream_stall_grace_env_override(&mut self) -> Result<()> {
        let raw = match std::env::var("KILN_STREAM_STALL_GRACE_MS") {
            Ok(raw) => raw,
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("KILN_STREAM_STALL_GRACE_MS must be valid UTF-8 decimal milliseconds")
            }
        };
        self.apply_stream_stall_grace_env_value(Some(&raw))
    }

    fn apply_stream_stall_grace_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.stream_stall_grace_ms = StreamStallGrace::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Resolve chunked-prefill scheduling once at startup. The actor receives
    /// this typed value and never consults mutable process environment.
    fn apply_max_batch_tokens_env_override(&mut self) -> Result<()> {
        let raw = match std::env::var("KILN_MAX_BATCH_TOKENS") {
            Ok(raw) => raw,
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("KILN_MAX_BATCH_TOKENS must be valid UTF-8 decimal tokens")
            }
        };
        self.apply_max_batch_tokens_env_value(Some(&raw))
    }

    fn apply_max_batch_tokens_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_batch_tokens = BatchTokenBudget::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Resolve the prompt-only actor-cycle ceiling once at startup.
    fn apply_max_prefill_tokens_per_cycle_env_override(&mut self) -> Result<()> {
        let raw = match std::env::var("KILN_MAX_PREFILL_TOKENS_PER_CYCLE") {
            Ok(raw) => raw,
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!(
                "KILN_MAX_PREFILL_TOKENS_PER_CYCLE must be valid UTF-8 decimal tokens"
            ),
        };
        self.apply_max_prefill_tokens_per_cycle_env_value(Some(&raw))
    }

    fn apply_max_prefill_tokens_per_cycle_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_prefill_tokens_per_cycle =
                PrefillTokenBudget::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Validate configuration values. Returns an error describing the first invalid value.
    fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            anyhow::bail!("server.port must be > 0");
        }
        if self.server.request_timeout_secs == 0 {
            anyhow::bail!("server.request_timeout_secs must be > 0");
        }
        if let Some(bytes) = self.server.http_send_buffer_bytes {
            validate_http_send_buffer_bytes(bytes)?;
        }
        validate_stream_stall_grace_ms(self.server.stream_stall_grace_ms.millis())?;
        validate_max_batch_tokens(self.server.max_batch_tokens.tokens())?;
        validate_max_prefill_tokens_per_cycle(self.server.max_prefill_tokens_per_cycle.tokens())?;
        if self.server.shutdown_timeout_secs == 0 {
            anyhow::bail!("server.shutdown_timeout_secs must be > 0");
        }

        let f = self.memory.inference_memory_fraction;
        if !(0.0..=1.0).contains(&f) {
            anyhow::bail!("memory.inference_memory_fraction must be between 0.0 and 1.0, got {f}");
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        let level = self.logging.level.to_lowercase();
        // Allow both simple levels and tracing filter directives (contain '=')
        if !valid_levels.contains(&level.as_str()) && !level.contains('=') {
            anyhow::bail!(
                "logging.level must be one of {valid_levels:?} or a tracing filter directive, got '{}'",
                self.logging.level
            );
        }

        if self.speculative.enabled {
            if self.speculative.num_speculative_tokens == 0 {
                anyhow::bail!("speculative.num_speculative_tokens must be > 0");
            }
            if self.speculative.draft_layers == 0 {
                anyhow::bail!("speculative.draft_layers must be > 0");
            }
        }

        if self.streaming_prefill.tile_tokens == 0 || self.streaming_prefill.tile_tokens % 64 != 0 {
            anyhow::bail!(
                "streaming_prefill.tile_tokens must be a positive multiple of 64, got {}",
                self.streaming_prefill.tile_tokens
            );
        }

        self.teachers.validate()?;

        Ok(())
    }
}

fn validate_http_send_buffer_bytes(bytes: usize) -> Result<()> {
    if !(HTTP_SEND_BUFFER_MIN_BYTES..=HTTP_SEND_BUFFER_MAX_BYTES).contains(&bytes) {
        anyhow::bail!(
            "server.http_send_buffer_bytes must be between {} and {} bytes, got {bytes}",
            HTTP_SEND_BUFFER_MIN_BYTES,
            HTTP_SEND_BUFFER_MAX_BYTES
        );
    }
    Ok(())
}

fn validate_stream_stall_grace_ms(millis: u64) -> Result<()> {
    if !(STREAM_STALL_GRACE_MIN_MS..=STREAM_STALL_GRACE_MAX_MS).contains(&millis) {
        anyhow::bail!(
            "server.stream_stall_grace_ms must be between {} and {} milliseconds, got {millis}",
            STREAM_STALL_GRACE_MIN_MS,
            STREAM_STALL_GRACE_MAX_MS
        );
    }
    Ok(())
}

fn validate_max_batch_tokens(tokens: usize) -> Result<()> {
    if !(MAX_BATCH_TOKENS_MIN..=MAX_BATCH_TOKENS_MAX).contains(&tokens) {
        anyhow::bail!(
            "server.max_batch_tokens must be between {} and {} tokens, got {tokens}",
            MAX_BATCH_TOKENS_MIN,
            MAX_BATCH_TOKENS_MAX
        );
    }
    Ok(())
}

fn validate_max_prefill_tokens_per_cycle(tokens: usize) -> Result<()> {
    if !(MAX_PREFILL_TOKENS_PER_CYCLE_MIN..=MAX_PREFILL_TOKENS_PER_CYCLE_MAX).contains(&tokens) {
        anyhow::bail!(
            "server.max_prefill_tokens_per_cycle must be between {} and {} tokens, got {tokens}",
            MAX_PREFILL_TOKENS_PER_CYCLE_MIN,
            MAX_PREFILL_TOKENS_PER_CYCLE_MAX
        );
    }
    Ok(())
}

fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn optional_limit_is_unlimited(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "none" | "null" | "off" | "unlimited"
    )
}

fn parse_optional_usize_env(value: &str) -> Result<Option<usize>, ()> {
    if optional_limit_is_unlimited(value) {
        Ok(None)
    } else {
        value.trim().parse::<usize>().map(Some).map_err(|_| ())
    }
}

fn parse_optional_u64_env(value: &str) -> Result<Option<u64>, ()> {
    if optional_limit_is_unlimited(value) {
        Ok(None)
    } else {
        value.trim().parse::<u64>().map(Some).map_err(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serializes tests that mutate the process-wide environment. cargo nextest
    // and `cargo test` run tests in parallel by default, so any test that
    // calls `std::env::set_var` / `std::env::remove_var` races with siblings
    // touching the same variables. Acquire this lock for the full duration of
    // the test (bind to a named guard, NOT `_`) before mutating env state.
    // `unwrap_or_else(|e| e.into_inner())` recovers from poisoning so a single
    // panicking test doesn't cascade into the rest of the suite.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_defaults() {
        let config = KilnConfig::default();
        assert_eq!(
            config.server.serving_profile.profile(),
            ServingProfile::Stable
        );
        assert_eq!(
            config.server.serving_profile.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.server.request_timeout_secs, 600);
        assert_eq!(config.server.http_send_buffer_bytes, None);
        assert_eq!(
            config.server.stream_stall_grace_ms.millis(),
            DEFAULT_STREAM_STALL_GRACE_MS
        );
        assert_eq!(
            config.server.stream_stall_grace_ms.source(),
            ConfigValueSource::Default
        );
        assert!(!config.server.eval_mode);
        assert_eq!(config.server.default_thinking_enabled, None);
        assert_eq!(config.server.default_thinking_budget_tokens, None);
        assert_eq!(config.server.default_thinking_budget_ms, None);
        assert!(!config.server.fold_reasoning_into_content);
        assert!(!config.server.chat_performance_metadata);
        assert!(!config.server.chat_config_hash_metadata);
        assert_eq!(config.server.slow_request_warn_secs, 30);
        assert_eq!(config.server.shutdown_timeout_secs, 5);
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
        assert!(config.model.path.is_none());
        assert!(config.model.tokenizer_path.is_none());
        assert!(config.model.adapter_dir.is_none());
        assert!(config.memory.num_blocks.is_none());
        assert_eq!(config.memory.inference_memory_fraction, 0.7);
        assert!(!config.memory.kv_cache_fp8);
        assert!(config.memory.cuda_graphs); // #34: default-ON
        assert!(!config.training.no_grad_checkpoint);
        assert!(config.training.checkpoint_interval.is_none());
        assert!(config.training.webhook_url.is_none());
        assert_eq!(config.training.max_queued_jobs, 32);
        assert_eq!(config.training.max_tracked_jobs, 1024);
        assert_eq!(config.training.tracked_job_ttl_secs, 604_800);
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.format, "auto");
        assert!(config.prefix_cache.enabled);
        assert!(config.prefix_cache.max_blocks.is_none());
        assert!(!config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 256);
        assert_eq!(config.speculative.draft_layers, 8);
        assert!(!config.streaming_prefill.enabled);
        assert_eq!(config.streaming_prefill.tile_tokens, 8192);
        assert!(config.streaming_prefill.last_token_lm_head);
        assert_eq!(
            config.adapters.max_disk_bytes,
            Some(100 * 1024u64.pow(3)),
            "default adapter disk cap should be 100 GiB"
        );
        assert_eq!(
            config.adapters.composed_cache_max_bytes,
            Some(10 * 1024u64.pow(3)),
            "default composed-cache byte cap should be 10 GiB"
        );
        assert_eq!(
            config.adapters.composed_cache_max_entries,
            Some(64),
            "default composed-cache entry cap should be 64"
        );
        assert!(config.teachers.credentials.is_empty());
    }

    #[test]
    fn teacher_credentials_parse_and_resolve_only_for_the_exact_origin() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const ENV: &str = "KILN_TEST_SCOPED_TEACHER_SECRET";
        unsafe {
            std::env::set_var(ENV, "test-secret");
        }
        let config: KilnConfig = toml::from_str(&format!(
            r#"
[teachers.credentials.primary-vllm]
origin = "https://vllm.example.com:8443"
api_key_env = "{ENV}"
"#
        ))
        .unwrap();
        config.validate().unwrap();
        assert_eq!(
            config
                .teachers
                .resolve_api_key_env(
                    Some("primary-vllm"),
                    "https://vllm.example.com:8443/tenant/a"
                )
                .unwrap()
                .as_deref(),
            Some(ENV)
        );
        let error = config
            .teachers
            .resolve_api_key_env(Some("primary-vllm"), "https://other.example.com:8443")
            .unwrap_err();
        assert!(error.contains("not authorized"), "{error}");
        assert!(!error.contains(ENV), "credential internals leaked: {error}");

        assert_eq!(
            config
                .teachers
                .resolve_api_key_env(None, "http://127.0.0.1:8000")
                .unwrap(),
            None
        );
        assert!(
            config
                .teachers
                .resolve_api_key_env(None, "https://vllm.example.com")
                .unwrap_err()
                .contains("credential_id")
        );
        unsafe {
            std::env::remove_var(ENV);
        }
    }

    #[test]
    fn teacher_credentials_reject_invalid_definitions_and_missing_secrets() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const ENV: &str = "KILN_TEST_MISSING_TEACHER_SECRET";
        unsafe {
            std::env::remove_var(ENV);
        }

        let mut config = KilnConfig::default();
        config.teachers.credentials.insert(
            "bad.id".into(),
            TeacherCredentialConfig {
                origin: "https://vllm.example.com".into(),
                api_key_env: ENV.into(),
            },
        );
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("credential id")
        );

        config.teachers.credentials.clear();
        config.teachers.credentials.insert(
            "valid-id".into(),
            TeacherCredentialConfig {
                origin: "https://vllm.example.com/".into(),
                api_key_env: ENV.into(),
            },
        );
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("canonical")
        );

        config
            .teachers
            .credentials
            .get_mut("valid-id")
            .unwrap()
            .origin = "https://vllm.example.com".into();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("not set"), "{error}");

        unsafe {
            std::env::set_var(ENV, "   ");
        }
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("empty"), "{error}");

        unsafe {
            std::env::set_var(ENV, "secret");
        }
        config
            .teachers
            .credentials
            .get_mut("valid-id")
            .unwrap()
            .api_key_env = "1INVALID".into();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("[A-Za-z_]"), "{error}");
        unsafe {
            std::env::remove_var(ENV);
        }
    }

    #[test]
    fn test_qwen35_defaults_profile() {
        let config = KilnConfig::default();
        let profile = config.model.defaults_profile();

        assert_eq!(profile.name, "Qwen3.5-4B");
        assert_eq!(profile.canonical_model_id, "Qwen/Qwen3.5-4B");
        assert_eq!(profile.canonical_served_model_id, "Qwen3.5-4B");
        assert_eq!(profile.server_default_thinking_enabled, None);
        assert!(profile.template_default_thinking_enabled);
        assert!(!profile.eval_mode_default_thinking_enabled);
        assert!(profile.supports_enable_thinking_kwarg);
        assert!(profile.supports_tool_chat_template);
        assert_eq!(
            profile.resolve_adapter_dir(None, "/models/Qwen3.5-4B"),
            PathBuf::from("/models/Qwen3.5-4B/adapters")
        );
        assert_eq!(
            profile.resolve_adapter_dir(Some("/tmp/adapters"), "/models/Qwen3.5-4B"),
            PathBuf::from("/tmp/adapters")
        );
    }

    #[test]
    fn test_parse_full_toml() {
        let toml_str = r#"
[server]
host = "127.0.0.1"
port = 9000
request_timeout_secs = 60
http_send_buffer_bytes = 8192
stream_stall_grace_ms = 1500
max_batch_tokens = 1024
max_prefill_tokens_per_cycle = 192
eval_mode = true
default_thinking_enabled = false
default_thinking_budget_tokens = 256
default_thinking_budget_ms = 1500
fold_reasoning_into_content = true
chat_performance_metadata = true
chat_config_hash_metadata = true
slow_request_warn_secs = 15
shutdown_timeout_secs = 10

[model]
path = "/models/qwen"
model_id = "custom/model"
tokenizer_path = "/models/tokenizer.json"
adapter_dir = "/models/adapters"

[memory]
num_blocks = 128
gpu_memory_gb = 24.0
inference_memory_fraction = 0.5
training_memory_gb = 6.0
kv_cache_fp8 = true
cuda_graphs = false

[training]
grad_checkpoint_segments = 8
no_grad_checkpoint = false
checkpoint_interval = 50
webhook_url = "https://example.com/hook"
max_queued_jobs = 4
max_tracked_jobs = 16
tracked_job_ttl_secs = 120

[logging]
level = "debug"
format = "pretty"

[prefix_cache]
enabled = false
max_blocks = 32
max_entries = 8

[speculative]
enabled = true
num_speculative_tokens = 6
draft_layers = 10

[streaming_prefill]
enabled = true
tile_tokens = 4096
last_token_lm_head = false

[adapters]
max_disk_bytes = 5368709120
composed_cache_max_bytes = 1073741824
composed_cache_max_entries = 8
"#;
        let config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.request_timeout_secs, 60);
        assert_eq!(config.server.http_send_buffer_bytes, Some(8192));
        assert_eq!(config.server.stream_stall_grace_ms.millis(), 1500);
        assert_eq!(
            config.server.stream_stall_grace_ms.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(config.server.max_batch_tokens.tokens(), 1024);
        assert_eq!(
            config.server.max_batch_tokens.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(config.server.max_prefill_tokens_per_cycle.tokens(), 192);
        assert_eq!(
            config.server.max_prefill_tokens_per_cycle.source(),
            ConfigValueSource::ConfigFile
        );
        assert!(config.server.eval_mode);
        assert_eq!(config.server.default_thinking_enabled, Some(false));
        assert_eq!(config.server.default_thinking_budget_tokens, Some(256));
        assert_eq!(config.server.default_thinking_budget_ms, Some(1500));
        assert!(config.server.fold_reasoning_into_content);
        assert!(config.server.chat_performance_metadata);
        assert!(config.server.chat_config_hash_metadata);
        assert_eq!(config.server.slow_request_warn_secs, 15);
        assert_eq!(config.model.path.as_deref(), Some("/models/qwen"));
        assert_eq!(config.model.model_id, "custom/model");
        assert_eq!(config.memory.num_blocks, Some(128));
        assert_eq!(config.memory.gpu_memory_gb, Some(24.0));
        assert_eq!(config.memory.inference_memory_fraction, 0.5);
        assert_eq!(config.memory.training_memory_gb, Some(6.0));
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs);
        assert_eq!(config.training.grad_checkpoint_segments, Some(8));
        assert_eq!(config.training.checkpoint_interval, Some(50));
        assert_eq!(
            config.training.webhook_url.as_deref(),
            Some("https://example.com/hook")
        );
        assert_eq!(config.training.max_queued_jobs, 4);
        assert_eq!(config.training.max_tracked_jobs, 16);
        assert_eq!(config.training.tracked_job_ttl_secs, 120);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.format, "pretty");
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(32));
        assert_eq!(config.prefix_cache.max_entries, Some(8));
        assert!(config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);
        assert!(config.streaming_prefill.enabled);
        assert_eq!(config.streaming_prefill.tile_tokens, 4096);
        assert!(!config.streaming_prefill.last_token_lm_head);
        assert_eq!(config.adapters.max_disk_bytes, Some(5_368_709_120));
        assert_eq!(
            config.adapters.composed_cache_max_bytes,
            Some(1_073_741_824)
        );
        assert_eq!(config.adapters.composed_cache_max_entries, Some(8));
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[server]
port = 3000
"#;
        let config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.server.host, "127.0.0.1"); // default (loopback)
        assert_eq!(config.server.request_timeout_secs, 600); // default
        assert_eq!(config.server.http_send_buffer_bytes, None); // default
        assert_eq!(
            config.server.stream_stall_grace_ms,
            StreamStallGrace::default()
        );
        assert_eq!(config.server.max_batch_tokens, BatchTokenBudget::default());
        assert_eq!(
            config.server.max_prefill_tokens_per_cycle,
            PrefillTokenBudget::default()
        );
        assert!(!config.server.eval_mode); // default
        assert_eq!(config.server.default_thinking_enabled, None); // default
        assert!(!config.server.fold_reasoning_into_content); // default
        assert_eq!(config.server.slow_request_warn_secs, 30); // default
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B"); // default
        assert_eq!(config.memory.inference_memory_fraction, 0.7); // default
        assert_eq!(config.logging.level, "info"); // default
    }

    #[test]
    fn test_validation_rejects_port_zero() {
        let mut config = KilnConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_fraction_above_one() {
        let mut config = KilnConfig::default();
        config.memory.inference_memory_fraction = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_fraction() {
        let mut config = KilnConfig::default();
        config.memory.inference_memory_fraction = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_bad_log_level() {
        let mut config = KilnConfig::default();
        config.logging.level = "banana".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_accepts_filter_directive() {
        let mut config = KilnConfig::default();
        config.logging.level = "kiln=trace,tower_http=warn".into();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_bad_streaming_tile_tokens() {
        let mut config = KilnConfig::default();
        config.streaming_prefill.tile_tokens = 0;
        assert!(config.validate().is_err());

        let mut config2 = KilnConfig::default();
        config2.streaming_prefill.tile_tokens = 100; // not a multiple of 64
        assert!(config2.validate().is_err());

        let mut config3 = KilnConfig::default();
        config3.streaming_prefill.tile_tokens = 64;
        assert!(config3.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_zero_timeout() {
        let mut config = KilnConfig::default();
        config.server.request_timeout_secs = 0;
        assert!(config.validate().is_err());

        let mut config2 = KilnConfig::default();
        config2.server.shutdown_timeout_secs = 0;
        assert!(config2.validate().is_err());
    }

    #[test]
    fn serving_profile_policy_matrix_is_fail_closed() {
        assert_eq!(
            ServingProfile::Stable.runtime_policy(),
            ServingRuntimePolicy {
                inference_admission: true,
                training_gpu_ownership: false,
                adapter_weight_transitions: false,
                dynamic_kv_resize: false,
                allocator_reclaim: false,
                live_graph_capture: false,
                exclusive_gpu_behavior: "reject",
            }
        );
        assert_eq!(
            ServingProfile::Experimental.runtime_policy(),
            ServingRuntimePolicy {
                inference_admission: true,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: true,
                exclusive_gpu_behavior: "writer_priority",
            }
        );
        assert_eq!(
            ServingProfile::Maintenance.runtime_policy(),
            ServingRuntimePolicy {
                inference_admission: false,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: false,
                exclusive_gpu_behavior: "inference_disabled_drain_then_exclusive",
            }
        );
    }

    #[test]
    fn serving_profile_diagnostics_bind_source_and_effective_policy() {
        let diagnostics =
            ServingProfileSetting::new(ServingProfile::Maintenance, ConfigValueSource::Environment)
                .diagnostics();

        assert_eq!(diagnostics.profile, ServingProfile::Maintenance);
        assert_eq!(diagnostics.source, ConfigValueSource::Environment);
        assert!(diagnostics.immutable_after_startup);
        assert!(!diagnostics.request_overrides_allowed);
        assert_eq!(diagnostics.effective_policy_source, "serving_profile");
        assert_eq!(
            diagnostics.effective_policy,
            ServingProfile::Maintenance.runtime_policy()
        );

        let json = serde_json::to_value(diagnostics).unwrap();
        assert_eq!(json["profile"], "maintenance");
        assert_eq!(json["source"], "environment");
        assert_eq!(json["effective_policy"]["inference_admission"], false);
        assert_eq!(
            json["effective_policy"]["exclusive_gpu_behavior"],
            "inference_disabled_drain_then_exclusive"
        );
    }

    #[test]
    fn serving_profile_toml_is_typed_and_source_tracked() {
        for (raw, expected) in [
            ("stable", ServingProfile::Stable),
            ("experimental", ServingProfile::Experimental),
            ("maintenance", ServingProfile::Maintenance),
        ] {
            let config: KilnConfig =
                toml::from_str(&format!("[server]\nserving_profile = {raw:?}\n")).unwrap();
            assert_eq!(config.server.serving_profile.profile(), expected);
            assert_eq!(
                config.server.serving_profile.source(),
                ConfigValueSource::ConfigFile
            );
            let serialized = toml::to_string(&config).unwrap();
            assert!(serialized.contains(&format!("serving_profile = {raw:?}")));
        }

        let error =
            toml::from_str::<KilnConfig>("[server]\nserving_profile = \"fast\"\n").unwrap_err();
        assert!(
            error.to_string().contains("server.serving_profile"),
            "{error:#}"
        );
    }

    #[test]
    fn serving_profile_env_override_is_strict_and_source_tracked() {
        let mut config: KilnConfig =
            toml::from_str("[server]\nserving_profile = \"experimental\"\n").unwrap();
        config
            .apply_serving_profile_env_value(Some(" maintenance "))
            .unwrap();
        assert_eq!(
            config.server.serving_profile.profile(),
            ServingProfile::Maintenance
        );
        assert_eq!(
            config.server.serving_profile.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "fast", "prod", "maintenance-now"] {
            let error = ServingProfileSetting::from_environment_value(invalid).unwrap_err();
            let detail = format!("{error:#}");
            assert!(detail.contains(SERVING_PROFILE_ENV), "{detail}");
            assert!(detail.contains(&format!("{invalid:?}")), "{detail}");
        }
    }

    #[test]
    fn load_rejects_malformed_serving_profile_environment() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        unsafe {
            std::env::set_var(SERVING_PROFILE_ENV, "prod-ish");
        }
        let error = KilnConfig::load(Some(path.to_str().unwrap())).unwrap_err();
        unsafe {
            std::env::remove_var(SERVING_PROFILE_ENV);
        }
        let detail = format!("{error:#}");
        assert!(detail.contains(SERVING_PROFILE_ENV), "{detail}");
        assert!(detail.contains("prod-ish"), "{detail}");
    }

    #[test]
    fn test_http_send_buffer_env_override_is_strict() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        let path = path.to_str().unwrap();

        unsafe {
            std::env::remove_var("KILN_HTTP_SEND_BUFFER_BYTES");
        }
        let config = KilnConfig::load(Some(path)).unwrap();
        assert_eq!(config.server.http_send_buffer_bytes, None);

        unsafe {
            std::env::set_var("KILN_HTTP_SEND_BUFFER_BYTES", "4096");
        }
        let config = KilnConfig::load(Some(path)).unwrap();
        assert_eq!(config.server.http_send_buffer_bytes, Some(4096));

        for invalid in ["", "0", "1023", "16777217", "not-a-number"] {
            unsafe {
                std::env::set_var("KILN_HTTP_SEND_BUFFER_BYTES", invalid);
            }
            let error = KilnConfig::load(Some(path)).unwrap_err();
            assert!(
                format!("{error:#}").contains("HTTP_SEND_BUFFER"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
        unsafe {
            std::env::remove_var("KILN_HTTP_SEND_BUFFER_BYTES");
        }
    }

    #[test]
    fn test_http_send_buffer_validation_bounds() {
        let mut config = KilnConfig::default();

        for valid in [HTTP_SEND_BUFFER_MIN_BYTES, HTTP_SEND_BUFFER_MAX_BYTES] {
            config.server.http_send_buffer_bytes = Some(valid);
            assert!(config.validate().is_ok(), "rejected valid bound {valid}");
        }

        for invalid in [
            HTTP_SEND_BUFFER_MIN_BYTES - 1,
            HTTP_SEND_BUFFER_MAX_BYTES + 1,
        ] {
            config.server.http_send_buffer_bytes = Some(invalid);
            let error = config.validate().unwrap_err();
            assert!(
                format!("{error:#}").contains("server.http_send_buffer_bytes"),
                "unexpected error for {invalid}: {error:#}"
            );
        }
    }

    #[test]
    fn test_stream_stall_grace_env_override_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
stream_stall_grace_ms = 1000
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.stream_stall_grace_ms.source(),
            ConfigValueSource::ConfigFile
        );

        config
            .apply_stream_stall_grace_env_value(Some(" 50 "))
            .unwrap();
        assert_eq!(config.server.stream_stall_grace_ms.millis(), 50);
        assert_eq!(
            config.server.stream_stall_grace_ms.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "0", "9", "2001", "-1", "not-a-number"] {
            let error = StreamStallGrace::from_environment_value(invalid).unwrap_err();
            assert!(
                format!("{error:#}").contains("KILN_STREAM_STALL_GRACE_MS"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
    }

    #[test]
    fn test_stream_stall_grace_toml_validation_bounds() {
        for valid in [STREAM_STALL_GRACE_MIN_MS, STREAM_STALL_GRACE_MAX_MS] {
            let config: KilnConfig =
                toml::from_str(&format!("[server]\nstream_stall_grace_ms = {valid}\n")).unwrap();
            assert_eq!(config.server.stream_stall_grace_ms.millis(), valid);
            assert_eq!(
                config.server.stream_stall_grace_ms.source(),
                ConfigValueSource::ConfigFile
            );
        }

        for invalid in [STREAM_STALL_GRACE_MIN_MS - 1, STREAM_STALL_GRACE_MAX_MS + 1] {
            let error = toml::from_str::<KilnConfig>(&format!(
                "[server]\nstream_stall_grace_ms = {invalid}\n"
            ))
            .unwrap_err();
            assert!(
                error.to_string().contains("server.stream_stall_grace_ms"),
                "unexpected error for {invalid}: {error:#}"
            );
        }
    }

    #[test]
    fn test_max_batch_tokens_env_override_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
max_batch_tokens = 1024
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.max_batch_tokens.source(),
            ConfigValueSource::ConfigFile
        );

        config
            .apply_max_batch_tokens_env_value(Some(" 256 "))
            .unwrap();
        assert_eq!(config.server.max_batch_tokens.tokens(), 256);
        assert_eq!(
            config.server.max_batch_tokens.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "0", "1", "65537", "-1", "not-a-number"] {
            let error = BatchTokenBudget::from_environment_value(invalid).unwrap_err();
            assert!(
                format!("{error:#}").contains("KILN_MAX_BATCH_TOKENS"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
    }

    #[test]
    fn test_max_batch_tokens_toml_validation_bounds() {
        for valid in [MAX_BATCH_TOKENS_MIN, MAX_BATCH_TOKENS_MAX] {
            let config: KilnConfig =
                toml::from_str(&format!("[server]\nmax_batch_tokens = {valid}\n")).unwrap();
            assert_eq!(config.server.max_batch_tokens.tokens(), valid);
            assert_eq!(
                config.server.max_batch_tokens.source(),
                ConfigValueSource::ConfigFile
            );
        }

        for invalid in [MAX_BATCH_TOKENS_MIN - 1, MAX_BATCH_TOKENS_MAX + 1] {
            let error =
                toml::from_str::<KilnConfig>(&format!("[server]\nmax_batch_tokens = {invalid}\n"))
                    .unwrap_err();
            assert!(
                error.to_string().contains("server.max_batch_tokens"),
                "unexpected error for {invalid}: {error:#}"
            );
        }
    }

    #[test]
    fn test_max_prefill_tokens_env_override_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
max_prefill_tokens_per_cycle = 256
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.max_prefill_tokens_per_cycle.source(),
            ConfigValueSource::ConfigFile
        );

        config
            .apply_max_prefill_tokens_per_cycle_env_value(Some(" 64 "))
            .unwrap();
        assert_eq!(config.server.max_prefill_tokens_per_cycle.tokens(), 64);
        assert_eq!(
            config.server.max_prefill_tokens_per_cycle.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "0", "65537", "-1", "not-a-number"] {
            let error = PrefillTokenBudget::from_environment_value(invalid).unwrap_err();
            assert!(
                format!("{error:#}").contains("KILN_MAX_PREFILL_TOKENS_PER_CYCLE"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
    }

    #[test]
    fn test_max_prefill_tokens_toml_validation_bounds() {
        for valid in [
            MAX_PREFILL_TOKENS_PER_CYCLE_MIN,
            MAX_PREFILL_TOKENS_PER_CYCLE_MAX,
        ] {
            let config: KilnConfig = toml::from_str(&format!(
                "[server]\nmax_prefill_tokens_per_cycle = {valid}\n"
            ))
            .unwrap();
            assert_eq!(config.server.max_prefill_tokens_per_cycle.tokens(), valid);
            assert_eq!(
                config.server.max_prefill_tokens_per_cycle.source(),
                ConfigValueSource::ConfigFile
            );
        }

        for invalid in [
            MAX_PREFILL_TOKENS_PER_CYCLE_MIN - 1,
            MAX_PREFILL_TOKENS_PER_CYCLE_MAX + 1,
        ] {
            let error = toml::from_str::<KilnConfig>(&format!(
                "[server]\nmax_prefill_tokens_per_cycle = {invalid}\n"
            ))
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("server.max_prefill_tokens_per_cycle"),
                "unexpected error for {invalid}: {error:#}"
            );
        }
    }

    #[test]
    fn test_env_var_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Safety: these tests manipulate env vars which is unsafe in Rust 1.78+.
        // They are safe because ENV_LOCK serializes env-mutating tests.
        unsafe {
            std::env::set_var("KILN_HOST", "10.0.0.1");
            std::env::set_var("KILN_PORT", "7777");
            std::env::set_var("KILN_EVAL_MODE", "true");
            std::env::set_var("KILN_DEFAULT_THINKING_ENABLED", "false");
            std::env::set_var("KILN_FOLD_REASONING_INTO_CONTENT", "true");
            std::env::set_var("KILN_CHAT_PERFORMANCE_METADATA", "true");
            std::env::set_var("KILN_CHAT_CONFIG_HASH_METADATA", "true");
            std::env::set_var("KILN_SLOW_REQUEST_WARN_SECS", "12");
            std::env::set_var("KILN_MODEL_PATH", "/tmp/model");
            std::env::set_var("KILN_INFERENCE_MEMORY_FRACTION", "0.9");
            std::env::set_var("KILN_LOG_LEVEL", "debug");
            std::env::set_var("KILN_NO_GRAD_CHECKPOINT", "1");
            std::env::set_var("KILN_CHECKPOINT_INTERVAL", "25");
            std::env::set_var("KILN_TRAINING_WEBHOOK_URL", "https://hook.example/notify");
            std::env::set_var("KILN_TRAINING_MAX_QUEUED_JOBS", "7");
            std::env::set_var("KILN_TRAINING_MAX_TRACKED_JOBS", "9");
            std::env::set_var("KILN_TRAINING_TRACKED_JOB_TTL_SECS", "11");
            std::env::set_var("KILN_KV_CACHE_FP8", "1");
            std::env::set_var("KILN_CUDA_GRAPHS", "false");
            std::env::set_var("KILN_PREFIX_CACHE_ENABLED", "false");
            std::env::set_var("KILN_PREFIX_CACHE_MAX_BLOCKS", "128");
            std::env::set_var("KILN_SPEC_ENABLED", "1");
            std::env::set_var("KILN_SPEC_NUM_TOKENS", "6");
            std::env::set_var("KILN_SPEC_DRAFT_LAYERS", "10");
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "2048");
            std::env::set_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD", "0");
        }

        let mut config = KilnConfig::default();
        config.apply_env_overrides();

        assert_eq!(config.server.host, "10.0.0.1");
        assert_eq!(config.server.port, 7777);
        assert!(config.server.eval_mode);
        assert_eq!(config.server.default_thinking_enabled, Some(false));
        assert!(config.server.fold_reasoning_into_content);
        assert!(config.server.chat_performance_metadata);
        assert!(config.server.chat_config_hash_metadata);
        assert_eq!(config.server.slow_request_warn_secs, 12);
        assert_eq!(config.model.path.as_deref(), Some("/tmp/model"));
        assert_eq!(config.memory.inference_memory_fraction, 0.9);
        assert_eq!(config.logging.level, "debug");
        assert!(config.training.no_grad_checkpoint);
        assert_eq!(config.training.checkpoint_interval, Some(25));
        assert_eq!(
            config.training.webhook_url.as_deref(),
            Some("https://hook.example/notify")
        );
        assert_eq!(config.training.max_queued_jobs, 7);
        assert_eq!(config.training.max_tracked_jobs, 9);
        assert_eq!(config.training.tracked_job_ttl_secs, 11);
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs); // env sets KILN_CUDA_GRAPHS=false -> override wins
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(128));
        assert!(config.prefix_cache.max_entries.is_none());
        assert!(config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);
        assert!(config.streaming_prefill.enabled);
        assert_eq!(config.streaming_prefill.tile_tokens, 2048);
        assert!(!config.streaming_prefill.last_token_lm_head);

        // Clean up
        unsafe {
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_EVAL_MODE");
            std::env::remove_var("KILN_DEFAULT_THINKING_ENABLED");
            std::env::remove_var("KILN_FOLD_REASONING_INTO_CONTENT");
            std::env::remove_var("KILN_CHAT_PERFORMANCE_METADATA");
            std::env::remove_var("KILN_CHAT_CONFIG_HASH_METADATA");
            std::env::remove_var("KILN_SLOW_REQUEST_WARN_SECS");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_INFERENCE_MEMORY_FRACTION");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
            std::env::remove_var("KILN_CHECKPOINT_INTERVAL");
            std::env::remove_var("KILN_TRAINING_WEBHOOK_URL");
            std::env::remove_var("KILN_TRAINING_MAX_QUEUED_JOBS");
            std::env::remove_var("KILN_TRAINING_MAX_TRACKED_JOBS");
            std::env::remove_var("KILN_TRAINING_TRACKED_JOB_TTL_SECS");
            std::env::remove_var("KILN_KV_CACHE_FP8");
            std::env::remove_var("KILN_CUDA_GRAPHS");
            std::env::remove_var("KILN_PREFIX_CACHE_ENABLED");
            std::env::remove_var("KILN_PREFIX_CACHE_MAX_BLOCKS");
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD");
        }
    }

    #[test]
    fn test_legacy_default_no_think_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("KILN_DEFAULT_NO_THINK", "1");
        }

        let mut config = KilnConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.server.default_thinking_enabled, Some(false));

        unsafe {
            std::env::set_var("KILN_DEFAULT_THINKING_ENABLED", "true");
        }
        config.apply_env_overrides();
        assert_eq!(
            config.server.default_thinking_enabled,
            Some(true),
            "new explicit env var should win over the legacy disable switch"
        );

        unsafe {
            std::env::remove_var("KILN_DEFAULT_NO_THINK");
            std::env::remove_var("KILN_DEFAULT_THINKING_ENABLED");
        }
    }

    #[test]
    fn test_adapters_max_disk_bytes_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = KilnConfig::default();
        // Default is 100 GiB.
        assert_eq!(config.adapters.max_disk_bytes, Some(100 * 1024u64.pow(3)));

        unsafe {
            std::env::set_var("KILN_ADAPTERS_MAX_DISK_BYTES", "1073741824");
        }
        config.apply_env_overrides();
        assert_eq!(config.adapters.max_disk_bytes, Some(1_073_741_824));

        // `0` disables the cap (operator-opt-out shorthand).
        unsafe {
            std::env::set_var("KILN_ADAPTERS_MAX_DISK_BYTES", "0");
        }
        config.apply_env_overrides();
        assert!(config.adapters.max_disk_bytes.is_none());

        // Empty string also clears the cap.
        unsafe {
            std::env::set_var("KILN_ADAPTERS_MAX_DISK_BYTES", "");
        }
        config.adapters.max_disk_bytes = Some(123);
        config.apply_env_overrides();
        assert!(config.adapters.max_disk_bytes.is_none());

        unsafe {
            std::env::remove_var("KILN_ADAPTERS_MAX_DISK_BYTES");
        }
    }

    #[test]
    fn test_adapters_composed_cache_max_bytes_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = KilnConfig::default();
        // Default is 10 GiB.
        assert_eq!(
            config.adapters.composed_cache_max_bytes,
            Some(10 * 1024u64.pow(3))
        );

        unsafe {
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "536870912");
        }
        config.apply_env_overrides();
        assert_eq!(config.adapters.composed_cache_max_bytes, Some(536_870_912));

        unsafe {
            std::env::remove_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES");
        }
    }

    #[test]
    fn test_adapters_composed_cache_max_entries_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = KilnConfig::default();
        // Default is 64.
        assert_eq!(config.adapters.composed_cache_max_entries, Some(64));

        unsafe {
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "12");
        }
        config.apply_env_overrides();
        assert_eq!(config.adapters.composed_cache_max_entries, Some(12));

        unsafe {
            std::env::remove_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES");
        }
    }

    #[test]
    fn test_adapters_composed_cache_zero_disables() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut config = KilnConfig::default();
        assert!(config.adapters.composed_cache_max_bytes.is_some());
        assert!(config.adapters.composed_cache_max_entries.is_some());

        // `0` is the operator-opt-out shorthand for both caps.
        unsafe {
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "0");
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "0");
        }
        config.apply_env_overrides();
        assert!(config.adapters.composed_cache_max_bytes.is_none());
        assert!(config.adapters.composed_cache_max_entries.is_none());

        // Empty string also clears.
        config.adapters.composed_cache_max_bytes = Some(123);
        config.adapters.composed_cache_max_entries = Some(7);
        unsafe {
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "");
            std::env::set_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "");
        }
        config.apply_env_overrides();
        assert!(config.adapters.composed_cache_max_bytes.is_none());
        assert!(config.adapters.composed_cache_max_entries.is_none());

        unsafe {
            std::env::remove_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES");
            std::env::remove_var("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES");
        }
    }

    #[test]
    fn test_training_webhook_env_empty_string_clears_toml_value() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let toml_str = r#"
[training]
webhook_url = "https://from-toml.example/hook"
"#;
        let mut config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.training.webhook_url.as_deref(),
            Some("https://from-toml.example/hook")
        );

        unsafe {
            std::env::set_var("KILN_TRAINING_WEBHOOK_URL", "");
        }
        config.apply_env_overrides();
        assert!(
            config.training.webhook_url.is_none(),
            "empty env var should clear the TOML-set webhook URL"
        );
        unsafe {
            std::env::remove_var("KILN_TRAINING_WEBHOOK_URL");
        }
    }

    #[test]
    fn test_load_missing_file_returns_defaults() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // With no file and no KILN_CONFIG env var, should return defaults
        unsafe {
            std::env::remove_var("KILN_CONFIG");
            // Clear env vars that would override defaults
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_EVAL_MODE");
            std::env::remove_var("KILN_SLOW_REQUEST_WARN_SECS");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_LOG_FORMAT");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
        }
        unsafe {
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
        }
        // Load from a path that doesn't exist via the CWD fallback (kiln.toml won't exist in test dir)
        let config = KilnConfig::load(None).unwrap();
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn test_load_explicit_path() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(
            &path,
            r#"
[server]
port = 5555

[logging]
level = "warn"
"#,
        )
        .unwrap();

        unsafe {
            // Clear env vars so they don't interfere
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_LOG_FORMAT");
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
        }

        let config = KilnConfig::load(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(config.server.port, 5555);
        assert_eq!(config.logging.level, "warn");
        assert_eq!(config.server.host, "127.0.0.1"); // default (loopback)
    }

    #[test]
    fn test_load_nonexistent_explicit_path_errors() {
        let result = KilnConfig::load(Some("/no/such/file.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_served_model_id_default_derivation() {
        let config = ModelConfig::default();
        assert_eq!(config.effective_served_model_id(), "Qwen3.5-4B");
    }

    #[test]
    fn test_served_model_id_preserves_no_slash() {
        let config = ModelConfig {
            model_id: "Qwen3.5-4B".into(),
            ..ModelConfig::default()
        };
        assert_eq!(config.effective_served_model_id(), "Qwen3.5-4B");
    }

    #[test]
    fn test_served_model_id_derives_from_nested_path() {
        let config = ModelConfig {
            model_id: "Org/Subdir/Model-Foo_7B".into(),
            ..ModelConfig::default()
        };
        assert_eq!(config.effective_served_model_id(), "Model-Foo_7B");
    }

    #[test]
    fn test_served_model_id_explicit_override_passes_through() {
        let config = ModelConfig {
            model_id: "Qwen/Qwen3.5-4B".into(),
            served_model_id: Some("My-Custom_Name".into()),
            ..ModelConfig::default()
        };
        assert_eq!(config.effective_served_model_id(), "My-Custom_Name");
    }

    #[test]
    fn test_served_model_id_env_var_overrides_toml() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let toml_str = r#"
[model]
served_model_id = "from-toml"
"#;
        let mut config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model.served_model_id.as_deref(), Some("from-toml"));

        unsafe {
            std::env::set_var("KILN_SERVED_MODEL_ID", "from-env");
        }
        config.apply_env_overrides();
        assert_eq!(
            config.model.effective_served_model_id(),
            "from-env",
            "env var should override TOML value"
        );
        unsafe {
            std::env::remove_var("KILN_SERVED_MODEL_ID");
        }
    }
}

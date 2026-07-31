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

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
pub use kiln_scheduler::DEFAULT_MAX_BATCH_TOKENS;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Default loopback interface used by a fresh server or desktop install.
pub const DEFAULT_SERVER_HOST: &str = "127.0.0.1";
/// Hostname used by local CLI clients when no server URL is supplied.
pub const DEFAULT_SERVER_CLIENT_HOST: &str = "localhost";
/// Shared default listen/client port. Keep this aligned with
/// `contracts/runtime-defaults-v1.json` and the desktop conformance checks.
pub const DEFAULT_SERVER_PORT: u16 = 8420;
/// Adapter-library endpoint advertised by the contract-only library API.
pub const DEFAULT_ADAPTER_LIBRARY_URL: &str = "https://library.kiln.run";

/// Default base URL used by local HTTP clients.
pub fn default_server_url() -> String {
    format!("http://{DEFAULT_SERVER_CLIENT_HOST}:{DEFAULT_SERVER_PORT}")
}

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
/// Default prompt-token work allowed between decode cohorts. This matches the
/// ROCm numerical prefill tile; backends that require route-invariant prompt
/// partitioning validate the relationship after backend selection.
pub const DEFAULT_MAX_PREFILL_TOKENS_PER_CYCLE: usize = 256;
pub const MAX_PREFILL_TOKENS_PER_CYCLE_MIN: usize = 1;
pub const MAX_PREFILL_TOKENS_PER_CYCLE_MAX: usize = MAX_BATCH_TOKENS_MAX;
/// Default transformer-layer work allowed before a partial prefill yields to
/// the next decode cohort. Four layers keeps a Qwen3.5-4B group well below the
/// measured token-only fixed-cost problem without repeating the 64-token chunk;
/// the hardware qualification gate determines whether it is sufficient.
pub const DEFAULT_MAX_PREFILL_LAYERS_PER_CYCLE: usize = 4;
pub const MAX_PREFILL_LAYERS_PER_CYCLE_MIN: usize = 1;
pub const MAX_PREFILL_LAYERS_PER_CYCLE_MAX: usize = 1_024;
/// Mechanically derived public environment spelling for the serving profile.
pub const SERVING_PROFILE_ENV: &str = "KILN_SERVER_SERVING_PROFILE";
/// Mechanically derived public environment spelling for deterministic serving.
pub const DETERMINISTIC_ENV: &str = "KILN_SERVER_DETERMINISTIC";
/// Mechanically derived public environment spelling for decode width.
pub const MAX_DECODE_BATCH_ENV: &str = "KILN_SERVER_MAX_DECODE_BATCH";
pub const MAX_DECODE_BATCH_MIN: usize = 1;
pub const MAX_DECODE_BATCH_MAX: usize = MAX_BATCH_TOKENS_MAX;
/// Latency-oriented prompt-admission default for backends that do not ask the
/// batching actor to fill the effective decode width before the first decode.
pub const DEFAULT_PREFILL_ADMISSION_QUANTUM: usize = 4;
pub const PREFILL_ADMISSION_QUANTUM_MIN: usize = 1;
pub const PREFILL_ADMISSION_QUANTUM_MAX: usize = MAX_BATCH_TOKENS_MAX;
/// Default cooperative idle inserted after an actor cycle that advanced model
/// work. Zero preserves the unpaced production scheduler.
pub const DEFAULT_ACTOR_CYCLE_IDLE_MS: u64 = 0;
/// Keep an accidentally large duty-cycle value from making the actor appear
/// unavailable for minutes. Command polling remains independently bounded.
pub const ACTOR_CYCLE_IDLE_MAX_MS: u64 = 60_000;
/// Maximum interval between control-command polls during cooperative idle.
pub const ACTOR_CYCLE_IDLE_COMMAND_POLL_MS: u64 = 5;
/// Bounds for the direct-stream greedy decode rendezvous worker. Its effective
/// width is also clamped to the already-resolved process decode ceiling.
/// Stable admission default used by the production batching actor.
pub const DEFAULT_PREFIX_AWARE_ADMISSION: bool = true;
/// Stable decode default: issue one true batched forward for all ready rows.
pub const DEFAULT_ROWWISE_DECODE: bool = false;
/// Mechanically derived public environment spelling for the token budget.
pub const DEFAULT_THINKING_BUDGET_TOKENS_ENV: &str = "KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS";
/// Mechanically derived public environment spelling for the decode-time budget.
pub const DEFAULT_THINKING_BUDGET_MS_ENV: &str = "KILN_SERVER_DEFAULT_THINKING_BUDGET_MS";
/// Stable default number of process-lifetime ROCm graph-cache entries.
pub const DEFAULT_ROCM_GRAPH_CACHE_ENTRIES: usize = 8;
pub const ROCM_GRAPH_CACHE_ENTRIES_MIN: usize = 1;
pub const ROCM_GRAPH_CACHE_ENTRIES_MAX: usize = 64;
/// Stable retained-device-byte budget for the experimental ROCm graph cache.
pub const DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;
pub const ROCM_GRAPH_CACHE_MAX_BYTES_MIN: u64 = 64 * 1024 * 1024;
pub const ROCM_GRAPH_CACHE_MAX_BYTES_MAX: u64 = 16 * 1024 * 1024 * 1024;
/// Versioned schema identity shared by config, health, and debug diagnostics.
pub const ACCELERATOR_RUNTIME_POLICY_SCHEMA_ID: &str = "kiln.accelerator-runtime-policy.v16";
pub const ACCELERATOR_RUNTIME_POLICY_VERSION: u32 = 16;

/// Stable operator-facing default for sparse SFT checkpoint-boundary anchors.
pub const DEFAULT_CHECKPOINT_BOUNDARY_CACHE_GB: f64 = 6.0;
const GIB_BYTES_F64: f64 = 1024.0 * 1024.0 * 1024.0;

/// Provenance of a resolved startup configuration value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigValueSource {
    #[default]
    Default,
    ConfigFile,
    Environment,
    CommandLine,
}

pub const EFFECTIVE_CONFIGURATION_SCHEMA_ID: &str = "kiln.effective-configuration.v1";
pub const EFFECTIVE_CONFIGURATION_SCHEMA_VERSION: u32 = 1;
pub const EFFECTIVE_CONFIGURATION_FIXED_FIELD_COUNT: usize = 117;

/// One post-precedence typed startup value in the complete configuration dump.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectiveConfigurationField {
    /// The typed value used by startup, or `null` when the value is redacted.
    pub effective_value: serde_json::Value,
    pub source: ConfigValueSource,
    pub canonical_environment: Option<String>,
    pub compatibility_environment: Vec<&'static str>,
    pub redacted: bool,
    pub restart_required_to_change: bool,
}

/// Complete, deterministic, source-aware view of the typed startup object.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectiveConfiguration {
    pub schema_id: &'static str,
    pub schema_version: u32,
    pub effective_config_hash: Option<String>,
    pub fixed_field_count: usize,
    pub dynamic_field_count: usize,
    pub all_fields_restart_required_to_change: bool,
    pub fields: BTreeMap<String, EffectiveConfigurationField>,
}

/// Source-tracked override for Kiln's shared persistent cache root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRootSetting {
    configured: Option<PathBuf>,
    source: ConfigValueSource,
}

impl CacheRootSetting {
    fn from_path(path: PathBuf, source: ConfigValueSource, name: &str) -> Result<Self> {
        if path.as_os_str().is_empty() {
            anyhow::bail!("{name} must be a non-empty path");
        }
        Ok(Self {
            configured: Some(path),
            source,
        })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Self::from_path(PathBuf::from(raw), ConfigValueSource::Environment, name)
    }

    pub fn configured(&self) -> Option<&Path> {
        self.configured.as_deref()
    }

    pub const fn source(&self) -> ConfigValueSource {
        self.source
    }
}

impl Default for CacheRootSetting {
    fn default() -> Self {
        Self {
            configured: None,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for CacheRootSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.configured.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CacheRootSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let path = PathBuf::deserialize(deserializer)?;
        Self::from_path(path, ConfigValueSource::ConfigFile, "paths.cache_root")
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for ConfigValueSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "default",
            Self::ConfigFile => "config_file",
            Self::Environment => "environment",
            Self::CommandLine => "command_line",
        })
    }
}

/// Validated process-lifetime deterministic-inference selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicInference {
    enabled: bool,
    source: ConfigValueSource,
}

impl DeterministicInference {
    pub const fn new(enabled: bool, source: ConfigValueSource) -> Self {
        Self { enabled, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let enabled = parse_bool_env(raw).with_context(|| {
            format!("{name} must be one of true, false, 1, 0, yes, no, on, off; got {raw:?}")
        })?;
        Ok(Self::new(enabled, ConfigValueSource::Environment))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value(DETERMINISTIC_ENV, raw)
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    pub const fn diagnostics(self) -> DeterministicInferenceDiagnostics {
        DeterministicInferenceDiagnostics {
            enabled: self.enabled,
            source: self.source,
        }
    }
}

impl Default for DeterministicInference {
    fn default() -> Self {
        Self::new(false, ConfigValueSource::Default)
    }
}

impl Serialize for DeterministicInference {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(self.enabled)
    }
}

impl<'de> Deserialize<'de> for DeterministicInference {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            bool::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

/// Health/config representation of the deterministic-inference selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeterministicInferenceDiagnostics {
    pub enabled: bool,
    pub source: ConfigValueSource,
}

/// Optional operator ceiling for concurrent decode rows.
///
/// `None` means the active backend policy selects the width. A present value
/// remains visible even when deterministic inference or the combined actor
/// token budget lowers the effective width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxDecodeBatch {
    limit: Option<usize>,
    source: ConfigValueSource,
}

impl MaxDecodeBatch {
    pub(crate) fn new(limit: Option<usize>, source: ConfigValueSource) -> Result<Self> {
        if let Some(limit) = limit {
            validate_max_decode_batch(limit)?;
        }
        Ok(Self { limit, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if is_auto_decode_batch(trimmed) {
            return Ok(Self {
                limit: None,
                source: ConfigValueSource::Environment,
            });
        }
        let limit = trimmed.parse::<usize>().with_context(|| {
            format!(
                "{name} must be 'auto' or a decimal integer in {MAX_DECODE_BATCH_MIN}..={MAX_DECODE_BATCH_MAX}, got {raw:?}"
            )
        })?;
        Self::new(Some(limit), ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value(MAX_DECODE_BATCH_ENV, raw)
    }

    pub const fn limit(self) -> Option<usize> {
        self.limit
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for MaxDecodeBatch {
    fn default() -> Self {
        Self {
            limit: None,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for MaxDecodeBatch {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.limit {
            Some(limit) => serializer.serialize_u64(limit as u64),
            None => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawMaxDecodeBatch {
    Limit(usize),
    Mode(String),
}

impl<'de> Deserialize<'de> for MaxDecodeBatch {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawMaxDecodeBatch::deserialize(deserializer)?;
        match raw {
            RawMaxDecodeBatch::Limit(limit) => {
                Self::new(Some(limit), ConfigValueSource::ConfigFile)
                    .map_err(serde::de::Error::custom)
            }
            RawMaxDecodeBatch::Mode(mode) if is_auto_decode_batch(&mode) => Ok(Self {
                limit: None,
                source: ConfigValueSource::ConfigFile,
            }),
            RawMaxDecodeBatch::Mode(mode) => Err(serde::de::Error::custom(format!(
                "server.max_decode_batch must be 'auto' or an integer in {MAX_DECODE_BATCH_MIN}..={MAX_DECODE_BATCH_MAX}, got {mode:?}"
            ))),
        }
    }
}

fn is_auto_decode_batch(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "auto" | "backend" | "backend_policy"
    )
}

/// Boolean batching setting represented as a TOML boolean with provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchingToggle {
    enabled: bool,
    source: ConfigValueSource,
}

impl BatchingToggle {
    pub const fn new(enabled: bool, source: ConfigValueSource) -> Self {
        Self { enabled, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let enabled = parse_required_bool_env(name, raw)?;
        Ok(Self::new(enabled, ConfigValueSource::Environment))
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    pub const fn diagnostics(self) -> BatchingToggleDiagnostics {
        BatchingToggleDiagnostics {
            enabled: self.enabled,
            source: self.source,
        }
    }
}

impl Default for BatchingToggle {
    fn default() -> Self {
        Self::new(false, ConfigValueSource::Default)
    }
}

impl Serialize for BatchingToggle {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(self.enabled)
    }
}

impl<'de> Deserialize<'de> for BatchingToggle {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            bool::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

/// Optional prompt-admission quantum plus the startup source that selected it.
/// `None` preserves the selected backend's admission policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillAdmissionQuantum {
    configured: Option<usize>,
    source: ConfigValueSource,
}

impl PrefillAdmissionQuantum {
    pub fn new(configured: Option<usize>, source: ConfigValueSource) -> Result<Self> {
        if let Some(quantum) = configured {
            validate_prefill_admission_quantum(quantum)?;
        }
        Ok(Self { configured, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            return Ok(Self {
                configured: None,
                source: ConfigValueSource::Environment,
            });
        }
        let quantum = trimmed.parse::<usize>().with_context(|| {
            format!(
                "{name} must be 'auto' or a decimal integer in {PREFILL_ADMISSION_QUANTUM_MIN}..={PREFILL_ADMISSION_QUANTUM_MAX}, got {raw:?}"
            )
        })?;
        Self::new(Some(quantum), ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }

    pub const fn configured(self) -> Option<usize> {
        self.configured
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for PrefillAdmissionQuantum {
    fn default() -> Self {
        Self {
            configured: None,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for PrefillAdmissionQuantum {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.configured {
            Some(quantum) => serializer.serialize_u64(quantum as u64),
            None => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawPrefillAdmissionQuantum {
    Quantum(usize),
    Mode(String),
}

impl<'de> Deserialize<'de> for PrefillAdmissionQuantum {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match RawPrefillAdmissionQuantum::deserialize(deserializer)? {
            RawPrefillAdmissionQuantum::Quantum(quantum) => {
                Self::new(Some(quantum), ConfigValueSource::ConfigFile)
                    .map_err(serde::de::Error::custom)
            }
            RawPrefillAdmissionQuantum::Mode(mode) if mode.trim().eq_ignore_ascii_case("auto") => {
                Ok(Self {
                    configured: None,
                    source: ConfigValueSource::ConfigFile,
                })
            }
            RawPrefillAdmissionQuantum::Mode(mode) => Err(serde::de::Error::custom(format!(
                "batching.prefill_admission_quantum must be 'auto' or an integer in {PREFILL_ADMISSION_QUANTUM_MIN}..={PREFILL_ADMISSION_QUANTUM_MAX}, got {mode:?}"
            ))),
        }
    }
}

/// Cooperative idle after a batching actor cycle that performed accelerator
/// work, plus the startup source that selected it.
///
/// Zero disables pacing. Nonzero values intentionally trade throughput and
/// latency for a lower sustained accelerator duty cycle without suspending the
/// process or hiding the wait from control-plane diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActorCycleIdle {
    millis: u64,
    source: ConfigValueSource,
}

impl ActorCycleIdle {
    fn new(millis: u64, source: ConfigValueSource) -> Result<Self> {
        validate_actor_cycle_idle_ms(millis)?;
        Ok(Self { millis, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let millis = raw.trim().parse::<u64>().with_context(|| {
            format!(
                "{name} must be a decimal integer in 0..={ACTOR_CYCLE_IDLE_MAX_MS}, got {raw:?}"
            )
        })?;
        Self::new(millis, ConfigValueSource::Environment).with_context(|| format!("invalid {name}"))
    }

    pub const fn millis(self) -> u64 {
        self.millis
    }

    pub fn duration(self) -> Duration {
        Duration::from_millis(self.millis)
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for ActorCycleIdle {
    fn default() -> Self {
        Self {
            millis: DEFAULT_ACTOR_CYCLE_IDLE_MS,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for ActorCycleIdle {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.millis)
    }
}

impl<'de> Deserialize<'de> for ActorCycleIdle {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Self::new(millis, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Authority that selected an effective batching value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchingEffectiveSource {
    Default,
    BackendPolicy,
    ConfigFile,
    Environment,
    CommandLine,
    EffectiveDecodeWidth,
}

impl fmt::Display for BatchingEffectiveSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "default",
            Self::BackendPolicy => "backend_policy",
            Self::ConfigFile => "config_file",
            Self::Environment => "environment",
            Self::CommandLine => "command_line",
            Self::EffectiveDecodeWidth => "effective_decode_width",
        })
    }
}

/// Backend-owned batching capabilities used to resolve the operator's `auto`
/// settings. Named fields prevent independent boolean capabilities from being
/// swapped at the startup boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchingBackendPolicy {
    pub use_decode_width_prefill_admission: bool,
    pub burst_prefill_admission: bool,
    pub actor_prefill_tile_alignment_required: bool,
}

impl BatchingBackendPolicy {
    /// Project the model backend's complete decode-execution policy onto the
    /// narrower server-owned settings needed for startup resolution.
    pub const fn from_decode_execution_policy(policy: kiln_model::DecodeExecutionPolicy) -> Self {
        Self {
            use_decode_width_prefill_admission: policy.use_decode_width_prefill_admission,
            burst_prefill_admission: policy.burst_prefill_admission,
            actor_prefill_tile_alignment_required: policy.actor_prefill_tile_alignment_required,
        }
    }
}

/// Runtime-ready batching policy resolved once after backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchingRuntimeConfig {
    pub rowwise_decode: BatchingToggleDiagnostics,
    pub prefix_aware_admission: BatchingToggleDiagnostics,
    pub prefill_admission_quantum: PrefillAdmissionQuantumDiagnostics,
    /// Intentional safe-boundary delay after actor cycles that advanced model
    /// work. This is zero unless explicitly configured.
    pub actor_cycle_idle: ActorCycleIdleDiagnostics,
    /// Backend-owned refill behavior, carried here so actor construction does
    /// not retain a second copy of the backend decode policy.
    pub burst_prefill_admission: bool,
    /// Whether startup must prove that actor prompt chunks preserve the
    /// backend-qualified streaming-prefill numerical boundary.
    pub actor_prefill_tile_alignment_required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ActorCycleIdleDiagnostics {
    pub milliseconds: u64,
    pub source: ConfigValueSource,
    pub enabled: bool,
    pub command_poll_milliseconds: u64,
}

impl Default for ActorCycleIdleDiagnostics {
    fn default() -> Self {
        Self {
            milliseconds: DEFAULT_ACTOR_CYCLE_IDLE_MS,
            source: ConfigValueSource::Default,
            enabled: false,
            command_poll_milliseconds: ACTOR_CYCLE_IDLE_COMMAND_POLL_MS,
        }
    }
}

impl BatchingRuntimeConfig {
    /// Project the resolved settings owned by the batching actor's admission
    /// loop. Actor activation and rowwise model dispatch are enforced at their
    /// separate startup boundaries and therefore are not part of this type.
    pub const fn actor_admission_config(self) -> BatchingActorAdmissionConfig {
        BatchingActorAdmissionConfig {
            prefix_aware_admission: self.prefix_aware_admission.enabled,
            prefill_admission_quantum: self.prefill_admission_quantum.effective,
            burst_prefill_admission: self.burst_prefill_admission,
        }
    }
}

/// Effective settings consumed specifically by the batching actor's admission
/// loop. Keeping this surface narrower than [`BatchingRuntimeConfig`] prevents
/// actor construction from appearing to apply activation or decode-dispatch
/// settings that belong to other startup boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchingActorAdmissionConfig {
    pub prefix_aware_admission: bool,
    pub prefill_admission_quantum: usize,
    pub burst_prefill_admission: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchingToggleDiagnostics {
    pub enabled: bool,
    pub source: ConfigValueSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefillAdmissionQuantumDiagnostics {
    pub configured: Option<usize>,
    pub configured_source: ConfigValueSource,
    pub backend_policy: usize,
    pub effective: usize,
    pub effective_source: BatchingEffectiveSource,
}

/// Final authority that selected the effective concurrent decode width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeBatchEffectiveSource {
    BackendPolicy,
    ConfigFile,
    Environment,
    CommandLine,
    Deterministic,
    MaxBatchTokens,
}

impl fmt::Display for DecodeBatchEffectiveSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BackendPolicy => "backend_policy",
            Self::ConfigFile => "config_file",
            Self::Environment => "environment",
            Self::CommandLine => "command_line",
            Self::Deterministic => "deterministic",
            Self::MaxBatchTokens => "max_batch_tokens",
        })
    }
}

/// Resolved process-lifetime decode policy exposed by startup, health, config,
/// and debug diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DecodeRuntimeConfig {
    pub deterministic: DeterministicInferenceDiagnostics,
    pub max_decode_batch: MaxDecodeBatchDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MaxDecodeBatchDiagnostics {
    pub configured: Option<usize>,
    pub configured_source: ConfigValueSource,
    pub backend_policy: usize,
    pub effective: usize,
    pub effective_source: DecodeBatchEffectiveSource,
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
                vulkan_resident_prefill: false,
                exclusive_gpu_behavior: "reject",
            },
            Self::Experimental => ServingRuntimePolicy {
                inference_admission: true,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: true,
                vulkan_resident_prefill: true,
                exclusive_gpu_behavior: "writer_priority",
            },
            Self::Maintenance => ServingRuntimePolicy {
                inference_admission: false,
                training_gpu_ownership: true,
                adapter_weight_transitions: true,
                dynamic_kv_resize: true,
                allocator_reclaim: true,
                live_graph_capture: false,
                vulkan_resident_prefill: false,
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
    /// Admit the corrected Vulkan-native token-prefill route. Stable serving
    /// remains on generic prefill until the full release qualification closes.
    pub vulkan_resident_prefill: bool,
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

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self {
            profile: ServingProfile::parse(raw, name)?,
            source: ConfigValueSource::Environment,
        })
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value(SERVING_PROFILE_ENV, raw)
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

/// Startup-authoritative kiln-tensor adapter route selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KtApiMode {
    /// Use qualified defaults: stable routes are active while experimental
    /// matmul and paged-KV routes remain inactive.
    #[default]
    Auto,
    /// Activate every adapter route. Requires the experimental profile.
    All,
    /// Disable every adapter route. Requires the experimental profile.
    Disabled,
}

impl KtApiMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            "disabled" => Ok(Self::Disabled),
            _ => anyhow::bail!("{label} must be one of auto, all, or disabled; got {raw:?}"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::All => "all",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for KtApiMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked kiln-tensor adapter route setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KtApiModeSetting {
    mode: KtApiMode,
    source: ConfigValueSource,
}

impl KtApiModeSetting {
    pub const fn new(mode: KtApiMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            KtApiMode::parse(raw, name)?,
            ConfigValueSource::Environment,
        ))
    }

    pub const fn mode(self) -> KtApiMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for KtApiModeSetting {
    fn default() -> Self {
        Self::new(KtApiMode::Auto, ConfigValueSource::Default)
    }
}

impl Serialize for KtApiModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for KtApiModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode =
            KtApiMode::parse(&raw, "accelerator.kt_api_mode").map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// ROCm host/stream synchronization policy selected at process startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmSynchronizationMode {
    /// Preserve the qualified host-visible barriers used by the existing
    /// backend while stream-ordered execution is being qualified.
    #[default]
    LegacyHostBarriers,
    /// Use stream dependencies for device work and reserve host waits for
    /// true external-yield and readback boundaries.
    StreamOrdered,
}

impl RocmSynchronizationMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "legacy_host_barriers" => Ok(Self::LegacyHostBarriers),
            "stream_ordered" => Ok(Self::StreamOrdered),
            _ => anyhow::bail!(
                "{label} must be one of legacy_host_barriers or stream_ordered; got {raw:?}"
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyHostBarriers => "legacy_host_barriers",
            Self::StreamOrdered => "stream_ordered",
        }
    }
}

impl fmt::Display for RocmSynchronizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked ROCm synchronization setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmSynchronizationModeSetting {
    mode: RocmSynchronizationMode,
    source: ConfigValueSource,
}

impl RocmSynchronizationModeSetting {
    pub const fn new(mode: RocmSynchronizationMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            RocmSynchronizationMode::parse(raw, name)?,
            ConfigValueSource::Environment,
        ))
    }

    pub const fn mode(self) -> RocmSynchronizationMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmSynchronizationModeSetting {
    fn default() -> Self {
        Self::new(
            RocmSynchronizationMode::LegacyHostBarriers,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for RocmSynchronizationModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for RocmSynchronizationModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode = RocmSynchronizationMode::parse(&raw, "accelerator.rocm_synchronization_mode")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Startup-authoritative route selection for ROCm strided-batched matmul.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmStridedBatchedMatmulMode {
    /// Always use the strided-batched hipBLASLt route when batch is greater
    /// than one. This requires the experimental serving profile.
    Enabled,
    /// Always issue one hipBLASLt operation per logical batch row.
    #[default]
    Disabled,
}

impl RocmStridedBatchedMatmulMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "enabled" => Ok(Self::Enabled),
            "disabled" => Ok(Self::Disabled),
            _ => anyhow::bail!("{label} must be one of enabled or disabled; got {raw:?}"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for RocmStridedBatchedMatmulMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked ROCm strided-batched matmul route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmStridedBatchedMatmulModeSetting {
    mode: RocmStridedBatchedMatmulMode,
    source: ConfigValueSource,
}

impl RocmStridedBatchedMatmulModeSetting {
    pub const fn new(mode: RocmStridedBatchedMatmulMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mode = RocmStridedBatchedMatmulMode::parse(raw, name)?;
        Ok(Self::new(mode, ConfigValueSource::Environment))
    }

    pub const fn mode(self) -> RocmStridedBatchedMatmulMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmStridedBatchedMatmulModeSetting {
    fn default() -> Self {
        Self::new(
            RocmStridedBatchedMatmulMode::Disabled,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for RocmStridedBatchedMatmulModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for RocmStridedBatchedMatmulModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode = RocmStridedBatchedMatmulMode::parse(
            &raw,
            "accelerator.rocm_strided_batched_matmul_mode",
        )
        .map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Startup-authoritative BF16-output route for ROCm matmul.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmBf16MatmulOutputMode {
    /// Always request native BF16 output from hipBLASLt. This requires the
    /// experimental serving profile.
    NativeBf16,
    /// Always request F32 output and cast it to BF16 on-device.
    #[default]
    F32ThenCast,
}

impl RocmBf16MatmulOutputMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "native_bf16" => Ok(Self::NativeBf16),
            "f32_then_cast" => Ok(Self::F32ThenCast),
            _ => anyhow::bail!("{label} must be one of native_bf16 or f32_then_cast; got {raw:?}"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeBf16 => "native_bf16",
            Self::F32ThenCast => "f32_then_cast",
        }
    }
}

impl fmt::Display for RocmBf16MatmulOutputMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked ROCm BF16-output matmul route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmBf16MatmulOutputModeSetting {
    mode: RocmBf16MatmulOutputMode,
    source: ConfigValueSource,
}

impl RocmBf16MatmulOutputModeSetting {
    pub const fn new(mode: RocmBf16MatmulOutputMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mode = RocmBf16MatmulOutputMode::parse(raw, name)?;
        Ok(Self::new(mode, ConfigValueSource::Environment))
    }

    pub const fn mode(self) -> RocmBf16MatmulOutputMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmBf16MatmulOutputModeSetting {
    fn default() -> Self {
        Self::new(
            RocmBf16MatmulOutputMode::F32ThenCast,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for RocmBf16MatmulOutputModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for RocmBf16MatmulOutputModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode =
            RocmBf16MatmulOutputMode::parse(&raw, "accelerator.rocm_bf16_matmul_output_mode")
                .map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Closed process-lifetime CUDA backend-kernel route set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaKernelProfile {
    /// Preserve the previously default-on native CUDA backend routes.
    #[default]
    NativeDefault,
    /// Decline all profile-governed routes and use portable fallbacks.
    PortableFallback,
}

impl CudaKernelProfile {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "native_default" => Ok(Self::NativeDefault),
            "portable_fallback" => Ok(Self::PortableFallback),
            _ => {
                anyhow::bail!(
                    "{label} must be one of native_default or portable_fallback; got {raw:?}"
                )
            }
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeDefault => "native_default",
            Self::PortableFallback => "portable_fallback",
        }
    }
}

impl fmt::Display for CudaKernelProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked CUDA backend-kernel profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelProfileSetting {
    profile: CudaKernelProfile,
    source: ConfigValueSource,
}

impl CudaKernelProfileSetting {
    pub const fn new(profile: CudaKernelProfile, source: ConfigValueSource) -> Self {
        Self { profile, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let profile = CudaKernelProfile::parse(raw, name)?;
        Ok(Self::new(profile, ConfigValueSource::Environment))
    }

    pub const fn profile(self) -> CudaKernelProfile {
        self.profile
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for CudaKernelProfileSetting {
    fn default() -> Self {
        Self::new(CudaKernelProfile::NativeDefault, ConfigValueSource::Default)
    }
}

impl Serialize for CudaKernelProfileSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.profile.as_str())
    }
}

impl<'de> Deserialize<'de> for CudaKernelProfileSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let profile = CudaKernelProfile::parse(&raw, "accelerator.cuda_kernel_profile")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(profile, ConfigValueSource::ConfigFile))
    }
}

/// Closed process-lifetime CUDA Marlin projection layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaMarlinProfile {
    /// Preserve BF16 projections; historical default.
    #[default]
    Disabled,
    /// Pack full-attention Q and every MLP projection as Marlin W4A16.
    AttentionMlp,
    /// Also pack the quality-sensitive GDN output projection.
    AttentionMlpGdn,
}

impl CudaMarlinProfile {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "disabled" => Ok(Self::Disabled),
            "attention_mlp" => Ok(Self::AttentionMlp),
            "attention_mlp_gdn" => Ok(Self::AttentionMlpGdn),
            _ => anyhow::bail!(
                "{label} must be one of disabled, attention_mlp, or attention_mlp_gdn; got {raw:?}"
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::AttentionMlp => "attention_mlp",
            Self::AttentionMlpGdn => "attention_mlp_gdn",
        }
    }
}

impl fmt::Display for CudaMarlinProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked CUDA Marlin projection profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMarlinProfileSetting {
    profile: CudaMarlinProfile,
    source: ConfigValueSource,
}

impl CudaMarlinProfileSetting {
    pub const fn new(profile: CudaMarlinProfile, source: ConfigValueSource) -> Self {
        Self { profile, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            CudaMarlinProfile::parse(raw, name)?,
            ConfigValueSource::Environment,
        ))
    }

    pub const fn profile(self) -> CudaMarlinProfile {
        self.profile
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for CudaMarlinProfileSetting {
    fn default() -> Self {
        Self::new(CudaMarlinProfile::Disabled, ConfigValueSource::Default)
    }
}

impl Serialize for CudaMarlinProfileSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.profile.as_str())
    }
}

impl<'de> Deserialize<'de> for CudaMarlinProfileSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let profile = CudaMarlinProfile::parse(&raw, "accelerator.cuda_marlin_profile")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(profile, ConfigValueSource::ConfigFile))
    }
}

/// CUDA FlashAttention backward accumulation mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaFlashBackwardMode {
    /// Historical fast accumulation path.
    #[default]
    Fast,
    /// Deterministic split accumulation for exact replay and diagnosis.
    Deterministic,
}

impl CudaFlashBackwardMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fast" => Ok(Self::Fast),
            "deterministic" => Ok(Self::Deterministic),
            _ => anyhow::bail!("{label} must be one of fast or deterministic; got {raw:?}"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Deterministic => "deterministic",
        }
    }
}

impl fmt::Display for CudaFlashBackwardMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked CUDA FlashAttention backward mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaFlashBackwardModeSetting {
    mode: CudaFlashBackwardMode,
    source: ConfigValueSource,
}

impl CudaFlashBackwardModeSetting {
    pub const fn new(mode: CudaFlashBackwardMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            CudaFlashBackwardMode::parse(raw, name)?,
            ConfigValueSource::Environment,
        ))
    }

    pub const fn mode(self) -> CudaFlashBackwardMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for CudaFlashBackwardModeSetting {
    fn default() -> Self {
        Self::new(CudaFlashBackwardMode::Fast, ConfigValueSource::Default)
    }
}

impl Serialize for CudaFlashBackwardModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for CudaFlashBackwardModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode = CudaFlashBackwardMode::parse(&raw, "accelerator.cuda_flash_backward_mode")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Closed process-lifetime Metal backend-kernel route set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalKernelProfile {
    /// Preserve the native Metal routes active before policy consolidation.
    #[default]
    NativeDefault,
    /// Decline all profile-governed routes and use portable fallbacks.
    PortableFallback,
}

impl MetalKernelProfile {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "native_default" => Ok(Self::NativeDefault),
            "portable_fallback" => Ok(Self::PortableFallback),
            _ => {
                anyhow::bail!(
                    "{label} must be one of native_default or portable_fallback; got {raw:?}"
                )
            }
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeDefault => "native_default",
            Self::PortableFallback => "portable_fallback",
        }
    }
}

impl fmt::Display for MetalKernelProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked Metal backend-kernel profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKernelProfileSetting {
    profile: MetalKernelProfile,
    source: ConfigValueSource,
}

impl MetalKernelProfileSetting {
    pub const fn new(profile: MetalKernelProfile, source: ConfigValueSource) -> Self {
        Self { profile, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let profile = MetalKernelProfile::parse(raw, name)?;
        Ok(Self::new(profile, ConfigValueSource::Environment))
    }

    pub const fn profile(self) -> MetalKernelProfile {
        self.profile
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for MetalKernelProfileSetting {
    fn default() -> Self {
        Self::new(
            MetalKernelProfile::NativeDefault,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for MetalKernelProfileSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.profile.as_str())
    }
}

impl<'de> Deserialize<'de> for MetalKernelProfileSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let profile = MetalKernelProfile::parse(&raw, "accelerator.metal_kernel_profile")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(profile, ConfigValueSource::ConfigFile))
    }
}

/// Closed process-lifetime ROCm kernel route set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmKernelProfile {
    /// Decline all profile-governed routes and use portable model fallbacks.
    #[default]
    PortableFallback,
}

impl RocmKernelProfile {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "portable_fallback" => Ok(Self::PortableFallback),
            _ => anyhow::bail!(
                "{label} must be portable_fallback; machine-qualified profiles are not product configuration; got {raw:?}"
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PortableFallback => "portable_fallback",
        }
    }
}

impl fmt::Display for RocmKernelProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked ROCm kernel profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmKernelProfileSetting {
    profile: RocmKernelProfile,
    source: ConfigValueSource,
}

impl RocmKernelProfileSetting {
    pub const fn new(profile: RocmKernelProfile, source: ConfigValueSource) -> Self {
        Self { profile, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let profile = RocmKernelProfile::parse(raw, name)?;
        Ok(Self::new(profile, ConfigValueSource::Environment))
    }

    pub const fn profile(self) -> RocmKernelProfile {
        self.profile
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmKernelProfileSetting {
    fn default() -> Self {
        Self::new(
            RocmKernelProfile::PortableFallback,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for RocmKernelProfileSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.profile.as_str())
    }
}

impl<'de> Deserialize<'de> for RocmKernelProfileSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let profile = RocmKernelProfile::parse(&raw, "accelerator.rocm_kernel_profile")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(profile, ConfigValueSource::ConfigFile))
    }
}

/// Configured ROCm graph lifecycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmGraphMode {
    /// Derive the effective lifecycle from the immutable serving profile.
    #[default]
    Profile,
    Disabled,
    /// Run one graph-shaped eager warmup and remain eager without capturing.
    WarmupThenEager,
    /// Lazily capture and replay shapes while serving.
    LazyCaptureReplay,
}

impl RocmGraphMode {
    fn parse(raw: &str, label: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "profile" => Ok(Self::Profile),
            "disabled" => Ok(Self::Disabled),
            "warmup_then_eager" => Ok(Self::WarmupThenEager),
            "lazy_capture_replay" => Ok(Self::LazyCaptureReplay),
            _ => anyhow::bail!(
                "{label} must be one of profile, disabled, warmup_then_eager, or lazy_capture_replay; got {raw:?}"
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Disabled => "disabled",
            Self::WarmupThenEager => "warmup_then_eager",
            Self::LazyCaptureReplay => "lazy_capture_replay",
        }
    }
}

impl fmt::Display for RocmGraphMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked ROCm graph lifecycle setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmGraphModeSetting {
    mode: RocmGraphMode,
    source: ConfigValueSource,
}

impl RocmGraphModeSetting {
    pub const fn new(mode: RocmGraphMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mode = RocmGraphMode::parse(raw, name)?;
        Ok(Self::new(mode, ConfigValueSource::Environment))
    }

    pub const fn mode(self) -> RocmGraphMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmGraphModeSetting {
    fn default() -> Self {
        Self::new(RocmGraphMode::Profile, ConfigValueSource::Default)
    }
}

impl Serialize for RocmGraphModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for RocmGraphModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode = RocmGraphMode::parse(&raw, "accelerator.rocm_graph_mode")
            .map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Validated source-tracked process-wide ROCm graph-cache capacity.
///
/// Saturated admission reclaims idle owners before minimum fair-LRU active
/// entries and preserves one projected graph for every active owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmGraphCacheEntries {
    entries: usize,
    source: ConfigValueSource,
}

impl RocmGraphCacheEntries {
    pub fn new(entries: usize, source: ConfigValueSource) -> Result<Self> {
        validate_rocm_graph_cache_entries(entries)?;
        Ok(Self { entries, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let entries = parse_decimal_env::<usize>(name, raw, "a decimal integer")?;
        Self::new(entries, ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }

    pub const fn entries(self) -> usize {
        self.entries
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmGraphCacheEntries {
    fn default() -> Self {
        Self {
            entries: DEFAULT_ROCM_GRAPH_CACHE_ENTRIES,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for RocmGraphCacheEntries {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.entries as u64)
    }
}

impl<'de> Deserialize<'de> for RocmGraphCacheEntries {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = usize::deserialize(deserializer)?;
        Self::new(entries, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Validated source-tracked retained-device-byte budget for ROCm graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmGraphCacheMaxBytes {
    bytes: u64,
    source: ConfigValueSource,
}

impl RocmGraphCacheMaxBytes {
    pub fn new(bytes: u64, source: ConfigValueSource) -> Result<Self> {
        validate_rocm_graph_cache_max_bytes(bytes)?;
        Ok(Self { bytes, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let bytes = parse_decimal_env::<u64>(name, raw, "a decimal integer byte count")?;
        Self::new(bytes, ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for RocmGraphCacheMaxBytes {
    fn default() -> Self {
        Self {
            bytes: DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for RocmGraphCacheMaxBytes {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.bytes)
    }
}

impl<'de> Deserialize<'de> for RocmGraphCacheMaxBytes {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = u64::deserialize(deserializer)?;
        Self::new(bytes, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Validated process-lifetime ceiling for exact full-attention score scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullAttentionScoreBudgetMib {
    mib: usize,
    source: ConfigValueSource,
}

impl FullAttentionScoreBudgetMib {
    pub fn new(mib: usize, source: ConfigValueSource) -> Result<Self> {
        kiln_model::validate_full_attention_score_budget_mib(mib)
            .with_context(|| format!("invalid full-attention score budget {mib} MiB"))?;
        Ok(Self { mib, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mib = parse_decimal_env::<usize>(name, raw, "a decimal integer MiB count")?;
        Self::new(mib, ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }

    pub const fn mib(self) -> usize {
        self.mib
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for FullAttentionScoreBudgetMib {
    fn default() -> Self {
        Self {
            mib: kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for FullAttentionScoreBudgetMib {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.mib as u64)
    }
}

impl<'de> Deserialize<'de> for FullAttentionScoreBudgetMib {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mib = usize::deserialize(deserializer)?;
        Self::new(mib, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Source-tracked Vulkan physical-device selection. `None` means automatic
/// discrete-GPU preference; an explicit index is validated against the devices
/// enumerated by Vulkan during startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanDeviceIndexSetting {
    index: Option<usize>,
    source: ConfigValueSource,
}

impl VulkanDeviceIndexSetting {
    pub const fn new(index: Option<usize>, source: ConfigValueSource) -> Self {
        Self { index, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let index = if raw.trim().eq_ignore_ascii_case("auto") {
            None
        } else {
            Some(parse_decimal_env::<usize>(
                name,
                raw,
                "'auto' or a zero-based decimal physical-device index",
            )?)
        };
        Ok(Self::new(index, ConfigValueSource::Environment))
    }

    pub const fn index(self) -> Option<usize> {
        self.index
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for VulkanDeviceIndexSetting {
    fn default() -> Self {
        Self::new(None, ConfigValueSource::Default)
    }
}

impl Serialize for VulkanDeviceIndexSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.index {
            Some(index) => serializer.serialize_u64(index as u64),
            None => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum VulkanDeviceIndexInput {
    Index(usize),
    Name(String),
}

impl<'de> Deserialize<'de> for VulkanDeviceIndexSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let index = match VulkanDeviceIndexInput::deserialize(deserializer)? {
            VulkanDeviceIndexInput::Index(index) => Some(index),
            VulkanDeviceIndexInput::Name(name) if name.eq_ignore_ascii_case("auto") => None,
            VulkanDeviceIndexInput::Name(name) => {
                return Err(serde::de::Error::custom(format!(
                    "accelerator.vulkan_device_index must be 'auto' or a zero-based integer, got {name:?}"
                )));
            }
        };
        Ok(Self::new(index, ConfigValueSource::ConfigFile))
    }
}

/// Source-tracked Vulkan validation-layer startup setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanValidationSetting {
    enabled: bool,
    source: ConfigValueSource,
}

impl VulkanValidationSetting {
    pub const fn new(enabled: bool, source: ConfigValueSource) -> Self {
        Self { enabled, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            parse_required_bool_env(name, raw)?,
            ConfigValueSource::Environment,
        ))
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for VulkanValidationSetting {
    fn default() -> Self {
        Self::new(false, ConfigValueSource::Default)
    }
}

impl Serialize for VulkanValidationSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(self.enabled)
    }
}

impl<'de> Deserialize<'de> for VulkanValidationSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            bool::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

/// One configured/effective accelerator policy leaf and its startup source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResolvedAcceleratorValue<T> {
    pub configured: T,
    pub effective: T,
    pub source: ConfigValueSource,
}

/// Versioned process-lifetime accelerator policy for config and health APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResolvedAcceleratorRuntimePolicy {
    pub schema_id: &'static str,
    pub version: u32,
    pub vulkan_kernel_policy_schema_id: &'static str,
    pub vulkan_device_policy_schema_id: &'static str,
    pub serving_profile: ServingProfile,
    pub serving_profile_source: ConfigValueSource,
    pub kt_api_mode: ResolvedAcceleratorValue<KtApiMode>,
    pub full_attention_score_budget_mib: ResolvedAcceleratorValue<usize>,
    pub vulkan_device_index: ResolvedAcceleratorValue<Option<usize>>,
    pub vulkan_validation: ResolvedAcceleratorValue<bool>,
    pub cuda_kernel_profile: ResolvedAcceleratorValue<CudaKernelProfile>,
    pub cuda_marlin_profile: ResolvedAcceleratorValue<CudaMarlinProfile>,
    pub cuda_flash_backward_mode: ResolvedAcceleratorValue<CudaFlashBackwardMode>,
    pub metal_kernel_profile: ResolvedAcceleratorValue<MetalKernelProfile>,
    pub rocm_synchronization_mode: ResolvedAcceleratorValue<RocmSynchronizationMode>,
    pub rocm_strided_batched_matmul_mode: ResolvedAcceleratorValue<RocmStridedBatchedMatmulMode>,
    pub rocm_bf16_matmul_output_mode: ResolvedAcceleratorValue<RocmBf16MatmulOutputMode>,
    pub rocm_kernel_profile: ResolvedAcceleratorValue<RocmKernelProfile>,
    pub rocm_graph_mode: ResolvedAcceleratorValue<RocmGraphMode>,
    pub rocm_graph_cache_entries: ResolvedAcceleratorValue<usize>,
    pub rocm_graph_cache_max_bytes: ResolvedAcceleratorValue<u64>,
}

/// Process-lifetime accelerator policy. Canonical startup overrides use
/// `KILN_ACCELERATOR_<FIELD>`; alternate spellings are not accepted.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AcceleratorRuntimeConfig {
    pub kt_api_mode: KtApiModeSetting,
    pub full_attention_score_budget_mib: FullAttentionScoreBudgetMib,
    pub vulkan_device_index: VulkanDeviceIndexSetting,
    pub vulkan_validation: VulkanValidationSetting,
    pub cuda_kernel_profile: CudaKernelProfileSetting,
    pub cuda_marlin_profile: CudaMarlinProfileSetting,
    pub cuda_flash_backward_mode: CudaFlashBackwardModeSetting,
    pub metal_kernel_profile: MetalKernelProfileSetting,
    pub rocm_synchronization_mode: RocmSynchronizationModeSetting,
    pub rocm_strided_batched_matmul_mode: RocmStridedBatchedMatmulModeSetting,
    pub rocm_bf16_matmul_output_mode: RocmBf16MatmulOutputModeSetting,
    pub rocm_kernel_profile: RocmKernelProfileSetting,
    pub rocm_graph_mode: RocmGraphModeSetting,
    pub rocm_graph_cache_entries: RocmGraphCacheEntries,
    pub rocm_graph_cache_max_bytes: RocmGraphCacheMaxBytes,
}

impl AcceleratorRuntimeConfig {
    /// Resolve all profile-derived values into an immutable runtime/API report.
    pub const fn resolved_policy(
        &self,
        serving_profile: ServingProfileSetting,
    ) -> ResolvedAcceleratorRuntimePolicy {
        let configured_graph_mode = self.rocm_graph_mode.mode();
        let effective_graph_mode = match configured_graph_mode {
            RocmGraphMode::Profile => match serving_profile.profile() {
                ServingProfile::Experimental => RocmGraphMode::LazyCaptureReplay,
                ServingProfile::Stable | ServingProfile::Maintenance => RocmGraphMode::Disabled,
            },
            explicit => explicit,
        };
        ResolvedAcceleratorRuntimePolicy {
            schema_id: ACCELERATOR_RUNTIME_POLICY_SCHEMA_ID,
            version: ACCELERATOR_RUNTIME_POLICY_VERSION,
            vulkan_kernel_policy_schema_id: kiln_model::VULKAN_KERNEL_POLICY_SCHEMA_ID,
            vulkan_device_policy_schema_id: kiln_model::VULKAN_DEVICE_POLICY_SCHEMA_ID,
            serving_profile: serving_profile.profile(),
            serving_profile_source: serving_profile.source(),
            kt_api_mode: ResolvedAcceleratorValue {
                configured: self.kt_api_mode.mode(),
                effective: self.kt_api_mode.mode(),
                source: self.kt_api_mode.source(),
            },
            full_attention_score_budget_mib: ResolvedAcceleratorValue {
                configured: self.full_attention_score_budget_mib.mib(),
                effective: self.full_attention_score_budget_mib.mib(),
                source: self.full_attention_score_budget_mib.source(),
            },
            vulkan_device_index: ResolvedAcceleratorValue {
                configured: self.vulkan_device_index.index(),
                effective: self.vulkan_device_index.index(),
                source: self.vulkan_device_index.source(),
            },
            vulkan_validation: ResolvedAcceleratorValue {
                configured: self.vulkan_validation.enabled(),
                effective: self.vulkan_validation.enabled(),
                source: self.vulkan_validation.source(),
            },
            cuda_kernel_profile: ResolvedAcceleratorValue {
                configured: self.cuda_kernel_profile.profile(),
                effective: self.cuda_kernel_profile.profile(),
                source: self.cuda_kernel_profile.source(),
            },
            cuda_marlin_profile: ResolvedAcceleratorValue {
                configured: self.cuda_marlin_profile.profile(),
                effective: self.cuda_marlin_profile.profile(),
                source: self.cuda_marlin_profile.source(),
            },
            cuda_flash_backward_mode: ResolvedAcceleratorValue {
                configured: self.cuda_flash_backward_mode.mode(),
                effective: self.cuda_flash_backward_mode.mode(),
                source: self.cuda_flash_backward_mode.source(),
            },
            metal_kernel_profile: ResolvedAcceleratorValue {
                configured: self.metal_kernel_profile.profile(),
                effective: self.metal_kernel_profile.profile(),
                source: self.metal_kernel_profile.source(),
            },
            rocm_synchronization_mode: ResolvedAcceleratorValue {
                configured: self.rocm_synchronization_mode.mode(),
                effective: self.rocm_synchronization_mode.mode(),
                source: self.rocm_synchronization_mode.source(),
            },
            rocm_strided_batched_matmul_mode: ResolvedAcceleratorValue {
                configured: self.rocm_strided_batched_matmul_mode.mode(),
                effective: self.rocm_strided_batched_matmul_mode.mode(),
                source: self.rocm_strided_batched_matmul_mode.source(),
            },
            rocm_bf16_matmul_output_mode: ResolvedAcceleratorValue {
                configured: self.rocm_bf16_matmul_output_mode.mode(),
                effective: self.rocm_bf16_matmul_output_mode.mode(),
                source: self.rocm_bf16_matmul_output_mode.source(),
            },
            rocm_kernel_profile: ResolvedAcceleratorValue {
                configured: self.rocm_kernel_profile.profile(),
                effective: self.rocm_kernel_profile.profile(),
                source: self.rocm_kernel_profile.source(),
            },
            rocm_graph_mode: ResolvedAcceleratorValue {
                configured: configured_graph_mode,
                effective: effective_graph_mode,
                source: self.rocm_graph_mode.source(),
            },
            rocm_graph_cache_entries: ResolvedAcceleratorValue {
                configured: self.rocm_graph_cache_entries.entries(),
                effective: self.rocm_graph_cache_entries.entries(),
                source: self.rocm_graph_cache_entries.source(),
            },
            rocm_graph_cache_max_bytes: ResolvedAcceleratorValue {
                configured: self.rocm_graph_cache_max_bytes.bytes(),
                effective: self.rocm_graph_cache_max_bytes.bytes(),
                source: self.rocm_graph_cache_max_bytes.source(),
            },
        }
    }

    /// Fail closed when an experimental accelerator behavior is requested
    /// under a profile that does not permit live accelerator experiments.
    pub fn validate_for_serving_profile(&self, profile: ServingProfile) -> Result<()> {
        if self.kt_api_mode.mode() != KtApiMode::Auto && profile != ServingProfile::Experimental {
            anyhow::bail!(
                "accelerator.kt_api_mode={} requires server.serving_profile=experimental; got {profile}",
                self.kt_api_mode.mode()
            );
        }
        if self.vulkan_validation.enabled() && profile != ServingProfile::Experimental {
            anyhow::bail!(
                "accelerator.vulkan_validation=true requires server.serving_profile=experimental; got {profile}"
            );
        }
        if self.cuda_marlin_profile.profile() != CudaMarlinProfile::Disabled
            && profile != ServingProfile::Experimental
        {
            anyhow::bail!(
                "accelerator.cuda_marlin_profile={} requires server.serving_profile=experimental; got {profile}",
                self.cuda_marlin_profile.profile()
            );
        }
        if self.rocm_synchronization_mode.mode() == RocmSynchronizationMode::StreamOrdered
            && profile != ServingProfile::Experimental
        {
            anyhow::bail!(
                "accelerator.rocm_synchronization_mode=stream_ordered requires server.serving_profile=experimental; got {profile}"
            );
        }
        if self.rocm_strided_batched_matmul_mode.mode() == RocmStridedBatchedMatmulMode::Enabled
            && profile != ServingProfile::Experimental
        {
            anyhow::bail!(
                "accelerator.rocm_strided_batched_matmul_mode={} requires server.serving_profile=experimental; got {profile}",
                self.rocm_strided_batched_matmul_mode.mode()
            );
        }
        if self.rocm_bf16_matmul_output_mode.mode() == RocmBf16MatmulOutputMode::NativeBf16
            && profile != ServingProfile::Experimental
        {
            anyhow::bail!(
                "accelerator.rocm_bf16_matmul_output_mode={} requires server.serving_profile=experimental; got {profile}",
                self.rocm_bf16_matmul_output_mode.mode()
            );
        }
        if matches!(
            self.rocm_graph_mode.mode(),
            RocmGraphMode::WarmupThenEager | RocmGraphMode::LazyCaptureReplay
        ) && profile != ServingProfile::Experimental
        {
            anyhow::bail!(
                "accelerator.rocm_graph_mode={} requires server.serving_profile=experimental; got {profile}",
                self.rocm_graph_mode.mode()
            );
        }
        validate_rocm_graph_cache_entries(self.rocm_graph_cache_entries.entries())?;
        validate_rocm_graph_cache_max_bytes(self.rocm_graph_cache_max_bytes.bytes())?;
        kiln_model::validate_full_attention_score_budget_mib(
            self.full_attention_score_budget_mib.mib(),
        )
    }
}

impl Default for AcceleratorRuntimeConfig {
    fn default() -> Self {
        Self {
            kt_api_mode: KtApiModeSetting::default(),
            full_attention_score_budget_mib: FullAttentionScoreBudgetMib::default(),
            vulkan_device_index: VulkanDeviceIndexSetting::default(),
            vulkan_validation: VulkanValidationSetting::default(),
            cuda_kernel_profile: CudaKernelProfileSetting::default(),
            cuda_marlin_profile: CudaMarlinProfileSetting::default(),
            cuda_flash_backward_mode: CudaFlashBackwardModeSetting::default(),
            metal_kernel_profile: MetalKernelProfileSetting::default(),
            rocm_synchronization_mode: RocmSynchronizationModeSetting::default(),
            rocm_strided_batched_matmul_mode: RocmStridedBatchedMatmulModeSetting::default(),
            rocm_bf16_matmul_output_mode: RocmBf16MatmulOutputModeSetting::default(),
            rocm_kernel_profile: RocmKernelProfileSetting::default(),
            rocm_graph_mode: RocmGraphModeSetting::default(),
            rocm_graph_cache_entries: RocmGraphCacheEntries::default(),
            rocm_graph_cache_max_bytes: RocmGraphCacheMaxBytes::default(),
        }
    }
}

/// Validated memory-governor reclaim mode plus startup provenance.
///
/// The memory crate deliberately has no serialization dependency, so the
/// server-owned startup boundary wraps its runtime enum and performs all TOML
/// and environment parsing here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryReclaimModeSetting {
    mode: kiln_memory::MemoryReclaimMode,
    source: ConfigValueSource,
}

impl MemoryReclaimModeSetting {
    pub const fn new(mode: kiln_memory::MemoryReclaimMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mode = kiln_memory::MemoryReclaimMode::parse(raw).map_err(|_| {
            anyhow::anyhow!("{name} must be one of off, on-demand, automatic; got {raw:?}")
        })?;
        Ok(Self::new(mode, ConfigValueSource::Environment))
    }

    pub const fn mode(self) -> kiln_memory::MemoryReclaimMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for MemoryReclaimModeSetting {
    fn default() -> Self {
        Self::new(
            kiln_memory::MemoryReclaimMode::Off,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for MemoryReclaimModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for MemoryReclaimModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let mode = kiln_memory::MemoryReclaimMode::parse(&raw).map_err(serde::de::Error::custom)?;
        Ok(Self::new(mode, ConfigValueSource::ConfigFile))
    }
}

/// Immutable KV autoscaler request and the startup source that selected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvAutoscaleSetting {
    enabled: bool,
    source: ConfigValueSource,
}

impl KvAutoscaleSetting {
    pub const fn new(enabled: bool, source: ConfigValueSource) -> Self {
        Self { enabled, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let enabled = parse_required_bool_env(name, raw)?;
        Ok(Self::new(enabled, ConfigValueSource::Environment))
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for KvAutoscaleSetting {
    fn default() -> Self {
        Self::new(true, ConfigValueSource::Default)
    }
}

impl Serialize for KvAutoscaleSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(self.enabled)
    }
}

impl<'de> Deserialize<'de> for KvAutoscaleSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            bool::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

/// One-shot startup KV resize target. Zero disables the forced resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvForceBlocksSetting {
    blocks: usize,
    source: ConfigValueSource,
}

impl KvForceBlocksSetting {
    pub const fn new(blocks: usize, source: ConfigValueSource) -> Self {
        Self { blocks, source }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let blocks = parse_decimal_env::<usize>(name, raw, "a non-negative decimal integer")?;
        Ok(Self::new(blocks, ConfigValueSource::Environment))
    }

    pub const fn blocks(self) -> usize {
        self.blocks
    }

    pub const fn target(self) -> Option<usize> {
        if self.blocks == 0 {
            None
        } else {
            Some(self.blocks)
        }
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for KvForceBlocksSetting {
    fn default() -> Self {
        Self::new(0, ConfigValueSource::Default)
    }
}

impl Serialize for KvForceBlocksSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.blocks as u64)
    }
}

impl<'de> Deserialize<'de> for KvForceBlocksSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            usize::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
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

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let millis = raw.trim().parse::<u64>().with_context(|| {
            format!(
                "{name} must be a decimal integer in {}..={}, got {raw:?}",
                STREAM_STALL_GRACE_MIN_MS, STREAM_STALL_GRACE_MAX_MS
            )
        })?;
        Self::new(millis, ConfigValueSource::Environment).with_context(|| format!("invalid {name}"))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value("KILN_SERVER_STREAM_STALL_GRACE_MS", raw)
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

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let tokens = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "{name} must be a decimal integer in {}..={}, got {raw:?}",
                MAX_BATCH_TOKENS_MIN, MAX_BATCH_TOKENS_MAX
            )
        })?;
        Self::new(tokens, ConfigValueSource::Environment).with_context(|| format!("invalid {name}"))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value("KILN_SERVER_MAX_BATCH_TOKENS", raw)
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

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let tokens = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "{name} must be a decimal integer in {}..={}, got {raw:?}",
                MAX_PREFILL_TOKENS_PER_CYCLE_MIN, MAX_PREFILL_TOKENS_PER_CYCLE_MAX
            )
        })?;
        Self::new(tokens, ConfigValueSource::Environment).with_context(|| format!("invalid {name}"))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value("KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE", raw)
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

/// Validated transformer-layer ceiling per prefill actor cycle plus source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillLayerBudget {
    layers: usize,
    source: ConfigValueSource,
}

impl PrefillLayerBudget {
    pub(crate) fn new(layers: usize, source: ConfigValueSource) -> Result<Self> {
        validate_max_prefill_layers_per_cycle(layers)?;
        Ok(Self { layers, source })
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let layers = raw.trim().parse::<usize>().with_context(|| {
            format!(
                "{name} must be a decimal integer in {}..={}, got {raw:?}",
                MAX_PREFILL_LAYERS_PER_CYCLE_MIN, MAX_PREFILL_LAYERS_PER_CYCLE_MAX
            )
        })?;
        Self::new(layers, ConfigValueSource::Environment).with_context(|| format!("invalid {name}"))
    }

    fn from_environment_value(raw: &str) -> Result<Self> {
        Self::from_named_environment_value("KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE", raw)
    }

    pub fn layers(self) -> usize {
        self.layers
    }

    pub fn source(self) -> ConfigValueSource {
        self.source
    }
}

impl Default for PrefillLayerBudget {
    fn default() -> Self {
        Self {
            layers: DEFAULT_MAX_PREFILL_LAYERS_PER_CYCLE,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for PrefillLayerBudget {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.layers as u64)
    }
}

impl<'de> Deserialize<'de> for PrefillLayerBudget {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let layers = usize::deserialize(deserializer)?;
        Self::new(layers, ConfigValueSource::ConfigFile).map_err(serde::de::Error::custom)
    }
}

/// Top-level configuration for kiln.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct KilnConfig {
    pub server: ServerConfig,
    pub accelerator: AcceleratorRuntimeConfig,
    pub batching: BatchingConfig,
    pub model: ModelConfig,
    /// Shared persistent filesystem locations resolved once before any model
    /// or accelerator cache is opened.
    pub paths: PathsConfig,
    pub memory: MemoryConfig,
    pub training: TrainingConfig,
    /// Native OpenEnv rollout and training control-plane policy.
    pub openenv: OpenEnvConfig,
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
    /// (mine→filter→train flywheel). Canonical startup overrides use
    /// `KILN_REQUEST_LOG_<FIELD>`. See [`crate::request_log`].
    #[serde(default)]
    pub request_log: crate::request_log::RequestLogConfig,
    /// §10.6 self-improvement automation. `None` / omitted section means
    /// the weekly loop stays manual (`kiln self-improve`).
    #[serde(default)]
    pub agent: Option<AgentConfig>,
    /// Loader-owned provenance is excluded from the typed configuration value
    /// and its hash. It exists only to produce the complete effective dump.
    #[serde(skip)]
    value_sources: BTreeMap<String, ConfigValueSource>,
}

/// `[paths]` process-lifetime application filesystem policy.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PathsConfig {
    /// Root for autotune, Vulkan pipeline, and transposed-weight caches.
    /// Relative paths are anchored to the startup working directory.
    pub cache_root: CacheRootSetting,
}

/// Effective application paths installed before cache consumers start.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedApplicationPaths {
    pub cache_root: PathBuf,
    pub cache_root_source: ConfigValueSource,
    pub restart_required_to_change: bool,
}

impl PathsConfig {
    pub fn resolve(&self) -> Result<ResolvedApplicationPaths> {
        let cache_root =
            kiln_resource::resolve_application_cache_root(self.cache_root.configured())
                .context("failed to resolve paths.cache_root")?;
        Ok(ResolvedApplicationPaths {
            cache_root,
            cache_root_source: self.cache_root.source(),
            restart_required_to_change: true,
        })
    }
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
        if kiln_train::validate_bearer_secret_environment(&credential.api_key_env).is_err() {
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
            kiln_train::validate_bearer_secret_environment(&credential.api_key_env).map_err(
                |error| {
                    let detail = match error {
                        kiln_train::CredentialLookupError::Unavailable => "is not set",
                        kiln_train::CredentialLookupError::Empty => "has an empty value",
                    };
                    anyhow::anyhow!(
                        "teachers.credentials.{credential_id}.api_key_env names an environment variable that {detail}"
                    )
                },
            )?;
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
    /// Access policy for arbitrary-code-execution-grade embedded runs.
    #[serde(default)]
    pub runs_access: LocalCapabilityAccess,
    /// Explicit `pi` executable. Omit to resolve `pi` from the startup PATH.
    #[serde(default)]
    pub pi_bin: Option<PathBuf>,
    /// External pi session directory used for trace discovery. Omit to use the
    /// operating-system account's `.pi/agent/sessions` directory.
    #[serde(default)]
    pub pi_sessions_dir: Option<PathBuf>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            self_improve_interval_hours: None,
            self_improve: None,
            max_concurrent_runs: default_max_concurrent_runs(),
            run_timeout_secs: default_run_timeout_secs(),
            runs_access: LocalCapabilityAccess::default(),
            pi_bin: None,
            pi_sessions_dir: None,
        }
    }
}

/// Access policy for local capabilities that can execute arbitrary code.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalCapabilityAccess {
    /// Enable only when the immutable listen host is loopback.
    #[default]
    LoopbackOnly,
    /// Enable even on a network bind.
    Enabled,
    /// Disable even on loopback.
    Disabled,
}

impl LocalCapabilityAccess {
    pub fn parse(name: &str, raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "loopback_only" | "auto" => Ok(Self::LoopbackOnly),
            "enabled" | "true" | "1" | "yes" | "on" => Ok(Self::Enabled),
            "disabled" | "false" | "0" | "no" | "off" => Ok(Self::Disabled),
            _ => anyhow::bail!("{name} must be loopback_only, enabled, or disabled, got {raw:?}"),
        }
    }

    pub fn enabled_for_host(self, host: &str) -> bool {
        match self {
            Self::LoopbackOnly => host_is_loopback(host),
            Self::Enabled => true,
            Self::Disabled => false,
        }
    }
}

pub fn host_is_loopback(host: &str) -> bool {
    let host = host.trim();
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.trim_matches(['[', ']'])
        .parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_loopback())
}

/// Immutable operational policy resolved once after the adapter root is known.
#[derive(Debug, Clone, Serialize)]
pub struct OperationalRuntimeConfig {
    pub bind_host: String,
    pub terminal_access: LocalCapabilityAccess,
    pub terminal_enabled: bool,
    pub agent_runs_access: LocalCapabilityAccess,
    pub agent_runs_enabled: bool,
    pub pi_bin: Option<PathBuf>,
    pub pi_sessions_dir: PathBuf,
    pub adapter_library_url: String,
    pub logit_cache_dir: PathBuf,
}

impl Default for OperationalRuntimeConfig {
    fn default() -> Self {
        let bind_host = DEFAULT_SERVER_HOST.to_owned();
        let terminal_access = LocalCapabilityAccess::default();
        let agent_runs_access = LocalCapabilityAccess::default();
        Self {
            terminal_enabled: terminal_access.enabled_for_host(&bind_host),
            agent_runs_enabled: agent_runs_access.enabled_for_host(&bind_host),
            bind_host,
            terminal_access,
            agent_runs_access,
            pi_bin: None,
            pi_sessions_dir: PathBuf::from("/tmp/pi/agent/sessions"),
            adapter_library_url: DEFAULT_ADAPTER_LIBRARY_URL.to_owned(),
            logit_cache_dir: PathBuf::from("logit-cache"),
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
#[serde(default, deny_unknown_fields)]
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

/// HTTP server settings. Canonical startup overrides use
/// `KILN_SERVER_<FIELD>`; alternate spellings are not accepted.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Immutable serving-safety profile. Stable is the default; switching to
    /// experimental or maintenance requires an explicit file/env setting and
    /// a process restart.
    pub serving_profile: ServingProfileSetting,
    /// Process-lifetime reproducibility envelope. Deterministic inference uses
    /// deterministic tensor paths and a single concurrent decode row.
    pub deterministic: DeterministicInference,
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
    /// Access policy for the embedded interactive terminal.
    pub terminal_access: LocalCapabilityAccess,
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
    /// decode rows consume one token each; newly selected prompt chunks use the
    /// remainder once, while retained layer groups are bounded by the separate
    /// layer ceiling.
    pub max_batch_tokens: BatchTokenBudget,
    /// Independent new-prompt-token ceiling inside the combined actor-cycle
    /// budget. Admission and new resumable chunks share this remainder so a
    /// long prompt cannot monopolize the actor; retained layer groups do not
    /// pay for the same tokens twice.
    pub max_prefill_tokens_per_cycle: PrefillTokenBudget,
    /// Transformer layers executed for an in-flight prefill chunk before the
    /// hidden state yields back to decode without completing/repeating tokens.
    pub max_prefill_layers_per_cycle: PrefillLayerBudget,
    /// Optional operator ceiling for concurrent decode rows. `auto` delegates
    /// to the active backend policy; deterministic mode and the combined token
    /// budget can still lower the effective value.
    pub max_decode_batch: MaxDecodeBatch,
    /// Enable deterministic eval-serving behavior for `kiln serve`.
    pub eval_mode: bool,
    /// Expose the trusted `/v1/debug/model-state` diagnostics endpoint without
    /// enabling eval-mode request semantics. The endpoint never includes prompt
    /// or user-message contents.
    pub debug_model_state: bool,
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

/// Model and tokenizer paths. Startup overrides use `KILN_MODEL_<FIELD>`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
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
    /// Optional average read rate for immutable snapshot copying and each
    /// loader-owned full checkpoint verification pass. Omission leaves reads
    /// unlimited while retaining cooperative cancellation between chunks.
    pub checkpoint_read_mib_per_second: Option<u64>,
    /// Optional average source-weight rate for eager accelerator upload.
    /// Omission leaves startup unlimited. A configured rate bounds sustained
    /// pressure and checks shutdown at least every 25 ms between layer uploads.
    pub accelerator_weight_upload_mib_per_second: Option<u64>,
    /// Populate Vulkan's backend-private decode-weight caches during startup.
    /// Disable only when startup latency matters more than first-request
    /// latency; ordinary serving keeps this enabled.
    pub vulkan_decode_weight_prewarm: bool,
    /// Average materialization rate for Vulkan decode-weight caches. Pacing is
    /// cancellation-aware and bounds sustained startup memory/thermal pressure.
    pub vulkan_decode_weight_prewarm_mib_per_second: u64,
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

/// GPU memory allocation settings. Startup overrides use `KILN_MEMORY_<FIELD>`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub num_blocks: Option<usize>,
    pub gpu_memory_gb: Option<f64>,
    pub inference_memory_fraction: f64,
    pub training_memory_gb: Option<f64>,
    /// Process-wide Vulkan scratch recycler cap. Active operations may exceed
    /// this amount, but idle buffers beyond it are not retained.
    pub vulkan_buffer_pool_gb: f64,
    /// GiB withheld by the process-wide memory governor after live probing.
    pub floor_gb: f64,
    /// Minimum interval between live OS/driver memory probes.
    pub probe_ms: u64,
    /// Whether allocator reclaim is disabled, explicit only, or automatic.
    pub reclaim_mode: MemoryReclaimModeSetting,
    /// Enable pressure-driven physical KV-cache resizing when the serving
    /// profile and backend both permit it.
    pub kv_autoscale: KvAutoscaleSetting,
    /// One-shot startup KV resize target. Zero disables the forced resize.
    pub kv_force_blocks: KvForceBlocksSetting,
    /// Enable FP8 (E4M3FN) quantization for KV cache, halving memory usage.
    /// When enabled, K/V values are stored as 8-bit floats with per-tensor scaling.
    /// Default: false
    pub kv_cache_fp8: bool,
    /// Enable CUDA graph capture/replay for decode steps.
    /// Eliminates per-step kernel launch overhead for ~10-15% decode speedup.
    /// Automatically disabled on non-CUDA devices.
    /// Default: true
    pub cuda_graphs: bool,
    /// Maximum retained single-row CUDA decode graphs. Each entry owns graph
    /// handles and graph-stable device buffers; the bound is fixed at startup.
    /// Valid range: 1..=64. Default: 8.
    pub cuda_graph_cache_entries: usize,
}

/// Training-specific settings. Startup overrides use `KILN_TRAINING_<FIELD>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointBoundaryRecomputeSetting {
    mode: kiln_train::CheckpointBoundaryRecomputeMode,
    source: ConfigValueSource,
}

impl CheckpointBoundaryRecomputeSetting {
    pub const fn new(
        mode: kiln_train::CheckpointBoundaryRecomputeMode,
        source: ConfigValueSource,
    ) -> Self {
        Self { mode, source }
    }

    pub const fn mode(self) -> kiln_train::CheckpointBoundaryRecomputeMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn parse_config(raw: &str) -> Result<kiln_train::CheckpointBoundaryRecomputeMode> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(kiln_train::CheckpointBoundaryRecomputeMode::Auto),
            "enabled" => Ok(kiln_train::CheckpointBoundaryRecomputeMode::Enabled),
            "disabled" => Ok(kiln_train::CheckpointBoundaryRecomputeMode::Disabled),
            _ => anyhow::bail!(
                "training.recompute_checkpoint_boundaries must be one of auto, enabled, or disabled, got {raw:?}"
            ),
        }
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let mode = Self::parse_config(raw).with_context(|| format!("invalid {name}"))?;
        Ok(Self::new(mode, ConfigValueSource::Environment))
    }
}

impl Default for CheckpointBoundaryRecomputeSetting {
    fn default() -> Self {
        Self::new(
            kiln_train::CheckpointBoundaryRecomputeMode::Auto,
            ConfigValueSource::Default,
        )
    }
}

impl Serialize for CheckpointBoundaryRecomputeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for CheckpointBoundaryRecomputeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::new(
            Self::parse_config(&raw).map_err(serde::de::Error::custom)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointBoundaryThresholdSetting {
    tokens: usize,
    source: ConfigValueSource,
}

impl CheckpointBoundaryThresholdSetting {
    pub fn new(tokens: usize, source: ConfigValueSource) -> Result<Self> {
        if tokens == 0 {
            anyhow::bail!(
                "training.recompute_boundary_threshold_tokens must be a positive integer, got {tokens}"
            );
        }
        Ok(Self { tokens, source })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let tokens = raw
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{name} must be a positive decimal integer, got {raw:?}"))?;
        Self::new(tokens, ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }
}

impl Default for CheckpointBoundaryThresholdSetting {
    fn default() -> Self {
        Self {
            tokens: kiln_train::DEFAULT_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for CheckpointBoundaryThresholdSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.tokens as u64)
    }
}

impl<'de> Deserialize<'de> for CheckpointBoundaryThresholdSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(
            usize::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointBoundaryAnchorStrideSetting {
    configured: Option<usize>,
    source: ConfigValueSource,
}

impl CheckpointBoundaryAnchorStrideSetting {
    pub fn new(configured: Option<usize>, source: ConfigValueSource) -> Result<Self> {
        if configured == Some(0) {
            anyhow::bail!(
                "training.checkpoint_boundary_anchor_stride must be 'auto' or a positive integer, got 0"
            );
        }
        Ok(Self { configured, source })
    }

    pub const fn configured(self) -> Option<usize> {
        self.configured
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            return Ok(Self {
                configured: None,
                source: ConfigValueSource::Environment,
            });
        }
        let stride = trimmed.parse::<usize>().with_context(|| {
            format!("{name} must be 'auto' or a positive decimal integer, got {raw:?}")
        })?;
        Self::new(Some(stride), ConfigValueSource::Environment)
            .with_context(|| format!("invalid {name} value {raw:?}"))
    }
}

impl Default for CheckpointBoundaryAnchorStrideSetting {
    fn default() -> Self {
        Self {
            configured: None,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for CheckpointBoundaryAnchorStrideSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.configured {
            Some(stride) => serializer.serialize_u64(stride as u64),
            None => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawCheckpointBoundaryAnchorStride {
    Stride(usize),
    Mode(String),
}

impl<'de> Deserialize<'de> for CheckpointBoundaryAnchorStrideSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match RawCheckpointBoundaryAnchorStride::deserialize(deserializer)? {
            RawCheckpointBoundaryAnchorStride::Stride(stride) => {
                Self::new(Some(stride), ConfigValueSource::ConfigFile)
                    .map_err(serde::de::Error::custom)
            }
            RawCheckpointBoundaryAnchorStride::Mode(mode)
                if mode.trim().eq_ignore_ascii_case("auto") =>
            {
                Ok(Self {
                    configured: None,
                    source: ConfigValueSource::ConfigFile,
                })
            }
            RawCheckpointBoundaryAnchorStride::Mode(mode) => {
                Err(serde::de::Error::custom(format!(
                    "training.checkpoint_boundary_anchor_stride must be 'auto' or a positive integer, got {mode:?}"
                )))
            }
        }
    }
}

fn checkpoint_boundary_cache_bytes(field: &str, gib: f64) -> Result<u64> {
    let bytes = gib * GIB_BYTES_F64;
    if !gib.is_finite()
        || gib <= 0.0
        || !bytes.is_finite()
        || bytes < 1.0
        || bytes >= u64::MAX as f64
    {
        anyhow::bail!(
            "{field} must be finite, > 0, and convert to between 1 and {} bytes, got {gib}",
            u64::MAX
        );
    }
    Ok(bytes as u64)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckpointBoundaryCacheGbSetting {
    gib: f64,
    bytes: u64,
    source: ConfigValueSource,
}

impl CheckpointBoundaryCacheGbSetting {
    pub fn new(gib: f64, source: ConfigValueSource) -> Result<Self> {
        let bytes = checkpoint_boundary_cache_bytes("training.checkpoint_boundary_cache_gb", gib)?;
        Ok(Self { gib, bytes, source })
    }

    pub const fn gib(self) -> f64 {
        self.gib
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        let gib = raw
            .trim()
            .parse::<f64>()
            .with_context(|| format!("{name} must be a decimal GiB value, got {raw:?}"))?;
        let bytes = checkpoint_boundary_cache_bytes(name, gib)
            .with_context(|| format!("invalid {name} value {raw:?}"))?;
        Ok(Self {
            gib,
            bytes,
            source: ConfigValueSource::Environment,
        })
    }
}

impl Default for CheckpointBoundaryCacheGbSetting {
    fn default() -> Self {
        Self {
            gib: DEFAULT_CHECKPOINT_BOUNDARY_CACHE_GB,
            bytes: kiln_train::DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
            source: ConfigValueSource::Default,
        }
    }
}

impl Serialize for CheckpointBoundaryCacheGbSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.gib)
    }
}

impl<'de> Deserialize<'de> for CheckpointBoundaryCacheGbSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(
            f64::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Copyable configured/source snapshot used by config, health, and debug APIs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CheckpointBoundaryConfigDiagnostics {
    pub recompute_checkpoint_boundaries: kiln_train::CheckpointBoundaryRecomputeMode,
    pub recompute_checkpoint_boundaries_source: ConfigValueSource,
    pub recompute_boundary_threshold_tokens: usize,
    pub recompute_boundary_threshold_tokens_source: ConfigValueSource,
    pub checkpoint_boundary_anchor_stride: Option<usize>,
    pub checkpoint_boundary_anchor_stride_source: ConfigValueSource,
    pub checkpoint_boundary_cache_gb: f64,
    pub checkpoint_boundary_cache_gb_source: ConfigValueSource,
    pub checkpoint_boundary_cache_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub grad_checkpoint_segments: Option<usize>,
    pub no_grad_checkpoint: bool,
    /// Retain every checkpoint boundary, replay sparse boundaries, or select
    /// automatically from `recompute_boundary_threshold_tokens`.
    pub recompute_checkpoint_boundaries: CheckpointBoundaryRecomputeSetting,
    /// Sequence length at which automatic checkpoint-boundary replay starts.
    pub recompute_boundary_threshold_tokens: CheckpointBoundaryThresholdSetting,
    /// Explicit sparse-anchor stride, or `"auto"` to derive it from the cache
    /// target and admitted tensor shape.
    pub checkpoint_boundary_anchor_stride: CheckpointBoundaryAnchorStrideSetting,
    /// GiB target used to derive an automatic sparse-anchor stride.
    pub checkpoint_boundary_cache_gb: CheckpointBoundaryCacheGbSetting,
    /// Save a checkpoint every N committed optimizer steps. SFT and GRPO
    /// checkpoints are exact and resumable; modes not yet migrated emit PEFT
    /// snapshots. Per-job config overrides this. `None` disables periodic
    /// checkpoints.
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
    /// Root for immutable teacher-logit cache entries. Omit to place it next
    /// to the adapter directory.
    pub logit_cache_dir: Option<PathBuf>,
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

/// `[openenv]` server-owned rollout orchestration policy.
///
/// CLI-driven OpenEnv workflows remain available independently. These limits
/// govern only the asynchronous HTTP/dashboard control plane.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenEnvConfig {
    /// Enable OpenEnv discovery and run lifecycle API routes.
    pub enabled: bool,
    /// Maximum concurrent rollout collectors. Training jobs submitted by a
    /// run subsequently use the normal bounded training queue.
    pub max_active_runs: usize,
    /// Maximum in-memory tracked run records, including terminal runs.
    pub max_tracked_runs: usize,
    /// Retention window for terminal run records.
    pub tracked_run_ttl_secs: u64,
    /// Permit the server control plane to connect to non-loopback OpenEnv
    /// origins. Disabled by default to make the HTTP API safe against SSRF;
    /// the local CLI remains unrestricted.
    pub allow_remote_environments: bool,
}

/// Logging settings. Startup overrides use `KILN_LOGGING_<FIELD>`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

/// Prefix caching settings. Canonical startup overrides use
/// `KILN_PREFIX_CACHE_<FIELD>`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
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

/// Production scheduling and admission settings for the single batching
/// actor. Canonical startup overrides are mechanically derived as
/// `KILN_BATCHING_<FIELD>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BatchingConfig {
    /// Issue each row of a ready decode cohort separately. This is an emergency
    /// comparison/fallback switch; true batched decode is the stable default.
    pub rowwise_decode: BatchingToggle,
    /// Defer a queued strict same-adapter descendant while an active prefix can
    /// still become reusable; independent rows remain eligible for admission.
    pub prefix_aware_admission: BatchingToggle,
    /// Number of queued prompts admitted before yielding to decode. `auto`
    /// uses either the effective decode width or the latency-oriented default,
    /// according to backend policy.
    pub prefill_admission_quantum: PrefillAdmissionQuantum,
    /// Cooperative idle in milliseconds after an actor cycle that advanced
    /// prefill or decode. Zero disables pacing. Default: 0.
    pub actor_cycle_idle_ms: ActorCycleIdle,
}

impl BatchingConfig {
    /// Resolve backend-dependent settings after the backend has selected its
    /// batching default and effective decode width.
    pub fn resolve(
        self,
        backend_policy: BatchingBackendPolicy,
        effective_decode_width: usize,
    ) -> BatchingRuntimeConfig {
        let effective_decode_width = effective_decode_width.max(1);
        let backend_quantum = if backend_policy.use_decode_width_prefill_admission {
            effective_decode_width
        } else {
            DEFAULT_PREFILL_ADMISSION_QUANTUM
        };
        let selected_quantum = self
            .prefill_admission_quantum
            .configured()
            .unwrap_or(backend_quantum);
        let effective_quantum = selected_quantum.clamp(1, effective_decode_width);
        let quantum_effective_source = if effective_quantum != selected_quantum {
            BatchingEffectiveSource::EffectiveDecodeWidth
        } else if self.prefill_admission_quantum.configured().is_some() {
            effective_source_for_explicit_value(self.prefill_admission_quantum.source())
        } else {
            BatchingEffectiveSource::BackendPolicy
        };

        BatchingRuntimeConfig {
            rowwise_decode: self.rowwise_decode.diagnostics(),
            prefix_aware_admission: self.prefix_aware_admission.diagnostics(),
            prefill_admission_quantum: PrefillAdmissionQuantumDiagnostics {
                configured: self.prefill_admission_quantum.configured(),
                configured_source: self.prefill_admission_quantum.source(),
                backend_policy: backend_quantum,
                effective: effective_quantum,
                effective_source: quantum_effective_source,
            },
            actor_cycle_idle: ActorCycleIdleDiagnostics {
                milliseconds: self.actor_cycle_idle_ms.millis(),
                source: self.actor_cycle_idle_ms.source(),
                enabled: self.actor_cycle_idle_ms.millis() > 0,
                command_poll_milliseconds: ACTOR_CYCLE_IDLE_COMMAND_POLL_MS,
            },
            burst_prefill_admission: backend_policy.burst_prefill_admission,
            actor_prefill_tile_alignment_required: backend_policy
                .actor_prefill_tile_alignment_required,
        }
    }
}

fn effective_source_for_explicit_value(source: ConfigValueSource) -> BatchingEffectiveSource {
    match source {
        ConfigValueSource::Default => BatchingEffectiveSource::Default,
        ConfigValueSource::ConfigFile => BatchingEffectiveSource::ConfigFile,
        ConfigValueSource::Environment => BatchingEffectiveSource::Environment,
        ConfigValueSource::CommandLine => BatchingEffectiveSource::CommandLine,
    }
}

/// Configured speculative-decoding intent. `Off` disables the feature.
///
/// - `Off` — no spec decoding, one token per step.
/// - `SkipLayer` — self-speculative using the first `draft_layers` of the main
///   model as a lightweight draft in isolated qualification.
/// - `Mtp` — native Multi-Token Prediction using the model's pretrained MTP
///   heads in isolated qualification (Qwen3.5-4B has one MTP layer, k=1).
///
/// Serving currently accepts only effective `Off`; every other value is
/// rejected before model loading.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpecMethod {
    Off,
    SkipLayer,
    Mtp,
}

/// Largest verifier window accepted by configuration and low-level probes.
///
/// This remains aligned with the planned local qualification matrix. Raising it
/// requires new accelerator evidence because verifier work and synchronization
/// scale with K.
pub const MAX_SPECULATIVE_DRAFT_TOKENS: usize = kiln_model::speculative::MAX_SPECULATIVE_TOKENS;

impl Default for SpecMethod {
    fn default() -> Self {
        Self::Off
    }
}

impl SpecMethod {
    /// Parse the canonical environment grammar. Returns `None` for unknown values.
    pub fn parse_env(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "skip_layer" => Some(Self::SkipLayer),
            "mtp" => Some(Self::Mtp),
            _ => None,
        }
    }
}

/// Speculative decoding settings. Startup overrides use
/// `KILN_SPECULATIVE_<FIELD>`.
///
/// Two qualification implementations coexist:
///   * `SkipLayer` — the first `draft_layers` of the main model act as the
///     draft. Works on any checkpoint.
///   * `Mtp` — native MTP heads shipped with the checkpoint (Qwen3.5-4B k=1).
///     Requires `mtp.*` tensors in the weights.
///
/// `method` is the single source of truth: `Off` disables speculative decoding,
/// while either non-off method records intended qualification behavior. Serving
/// rejects both non-off implementations before model loading; the draft-window
/// default and hard ceiling are both K=4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeDecodingConfig {
    /// Which speculative-decoding method to use. `Off` is the disabled default.
    pub method: SpecMethod,
    /// Number of tokens the draft proposes per step (default: 4).
    /// Ignored by the k=1 `Mtp` research path.
    pub num_speculative_tokens: usize,
    /// Number of layers to use for the `SkipLayer` draft (default: 8).
    pub draft_layers: usize,
}

/// Operator intent for whether the selected backend may use streaming prefill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingPrefillMode {
    /// Defer to the selected backend's immutable policy.
    #[default]
    Auto,
    Enabled,
    Disabled,
}

impl StreamingPrefillMode {
    fn parse_config(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "enabled" => Ok(Self::Enabled),
            "disabled" => Ok(Self::Disabled),
            _ => anyhow::bail!(
                "streaming_prefill.mode must be one of auto, enabled, or disabled, got {raw:?}"
            ),
        }
    }

    fn parse_environment(name: &str, raw: &str) -> Result<Self> {
        Self::parse_config(raw).with_context(|| format!("invalid {name}"))
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for StreamingPrefillMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Source-tracked streaming-prefill mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingPrefillModeSetting {
    mode: StreamingPrefillMode,
    source: ConfigValueSource,
}

impl StreamingPrefillModeSetting {
    pub const fn new(mode: StreamingPrefillMode, source: ConfigValueSource) -> Self {
        Self { mode, source }
    }

    pub const fn mode(self) -> StreamingPrefillMode {
        self.mode
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            StreamingPrefillMode::parse_environment(name, raw)?,
            ConfigValueSource::Environment,
        ))
    }
}

impl Default for StreamingPrefillModeSetting {
    fn default() -> Self {
        Self::new(StreamingPrefillMode::Auto, ConfigValueSource::Default)
    }
}

impl Serialize for StreamingPrefillModeSetting {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.mode.as_str())
    }
}

impl<'de> Deserialize<'de> for StreamingPrefillModeSetting {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::new(
            StreamingPrefillMode::parse_config(&raw).map_err(serde::de::Error::custom)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawStreamingPrefillTokenSetting {
    Tokens(usize),
    Mode(String),
}

macro_rules! define_streaming_prefill_token_setting {
    ($name:ident, $field:literal, $validator:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            configured: Option<usize>,
            source: ConfigValueSource,
        }

        impl $name {
            pub fn new(configured: Option<usize>, source: ConfigValueSource) -> Result<Self> {
                if let Some(tokens) = configured {
                    $validator($field, tokens)?;
                }
                Ok(Self { configured, source })
            }

            pub const fn configured(self) -> Option<usize> {
                self.configured
            }

            pub const fn source(self) -> ConfigValueSource {
                self.source
            }

            fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
                let trimmed = raw.trim();
                if trimmed.eq_ignore_ascii_case("auto") {
                    return Ok(Self {
                        configured: None,
                        source: ConfigValueSource::Environment,
                    });
                }
                let tokens = trimmed.parse::<usize>().with_context(|| {
                    format!("{name} must be 'auto' or a positive decimal integer, got {raw:?}")
                })?;
                Self::new(Some(tokens), ConfigValueSource::Environment)
                    .with_context(|| format!("invalid {name} value {raw:?}"))
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    configured: None,
                    source: ConfigValueSource::Default,
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                match self.configured {
                    Some(tokens) => serializer.serialize_u64(tokens as u64),
                    None => serializer.serialize_str("auto"),
                }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                match RawStreamingPrefillTokenSetting::deserialize(deserializer)? {
                    RawStreamingPrefillTokenSetting::Tokens(tokens) => {
                        Self::new(Some(tokens), ConfigValueSource::ConfigFile)
                            .map_err(serde::de::Error::custom)
                    }
                    RawStreamingPrefillTokenSetting::Mode(mode)
                        if mode.trim().eq_ignore_ascii_case("auto") =>
                    {
                        Ok(Self {
                            configured: None,
                            source: ConfigValueSource::ConfigFile,
                        })
                    }
                    RawStreamingPrefillTokenSetting::Mode(mode) => {
                        Err(serde::de::Error::custom(format!(
                            "{} must be 'auto' or a positive integer, got {mode:?}",
                            $field
                        )))
                    }
                }
            }
        }
    };
}

fn validate_streaming_prefill_positive_tokens(field: &str, tokens: usize) -> Result<()> {
    if tokens == 0 {
        anyhow::bail!("{field} must be a positive integer, got {tokens}");
    }
    Ok(())
}

fn validate_streaming_prefill_tile_tokens(field: &str, tokens: usize) -> Result<()> {
    if tokens == 0 || tokens % 64 != 0 {
        anyhow::bail!("{field} must be a positive multiple of 64, got {tokens}");
    }
    Ok(())
}

define_streaming_prefill_token_setting!(
    StreamingPrefillThresholdTokens,
    "streaming_prefill.threshold_tokens",
    validate_streaming_prefill_positive_tokens
);
define_streaming_prefill_token_setting!(
    StreamingPrefillTileTokens,
    "streaming_prefill.tile_tokens",
    validate_streaming_prefill_tile_tokens
);
define_streaming_prefill_token_setting!(
    StreamingPrefillTapeTileTokens,
    "streaming_prefill.tape_tile_tokens",
    validate_streaming_prefill_tile_tokens
);
define_streaming_prefill_token_setting!(
    StreamingPrefillDetachedFullAttnTileTokens,
    "streaming_prefill.detached_full_attn_tile_tokens",
    validate_streaming_prefill_tile_tokens
);

/// Source-tracked last-token LM-head optimization selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingPrefillLastTokenLmHead {
    enabled: bool,
    source: ConfigValueSource,
}

impl StreamingPrefillLastTokenLmHead {
    pub const fn new(enabled: bool, source: ConfigValueSource) -> Self {
        Self { enabled, source }
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn source(self) -> ConfigValueSource {
        self.source
    }

    fn from_named_environment_value(name: &str, raw: &str) -> Result<Self> {
        Ok(Self::new(
            parse_required_bool_env(name, raw)?,
            ConfigValueSource::Environment,
        ))
    }
}

impl Default for StreamingPrefillLastTokenLmHead {
    fn default() -> Self {
        Self::new(true, ConfigValueSource::Default)
    }
}

impl Serialize for StreamingPrefillLastTokenLmHead {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(self.enabled)
    }
}

impl<'de> Deserialize<'de> for StreamingPrefillLastTokenLmHead {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(
            bool::deserialize(deserializer)?,
            ConfigValueSource::ConfigFile,
        ))
    }
}

/// Streaming/tiled prefill settings. Startup overrides are derived
/// mechanically as `KILN_STREAMING_PREFILL_<FIELD>`.
///
/// `auto` values preserve backend policy. Every concrete tile size must be a
/// positive multiple of 64, the recurrent-attention chunk size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreamingPrefillConfig {
    pub mode: StreamingPrefillModeSetting,
    pub threshold_tokens: StreamingPrefillThresholdTokens,
    pub tile_tokens: StreamingPrefillTileTokens,
    pub tape_tile_tokens: StreamingPrefillTapeTileTokens,
    pub detached_full_attn_tile_tokens: StreamingPrefillDetachedFullAttnTileTokens,
    pub last_token_lm_head: StreamingPrefillLastTokenLmHead,
}

/// Final authority for a resolved streaming-prefill value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingPrefillEffectiveSource {
    BackendPolicy,
    Default,
    ConfigFile,
    Environment,
    CommandLine,
    InheritedFromTileTokensDefault,
    InheritedFromTileTokensConfigFile,
    InheritedFromTileTokensEnvironment,
    InheritedFromTileTokensCommandLine,
}

impl fmt::Display for StreamingPrefillEffectiveSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BackendPolicy => "backend_policy",
            Self::Default => "default",
            Self::ConfigFile => "config_file",
            Self::Environment => "environment",
            Self::CommandLine => "command_line",
            Self::InheritedFromTileTokensDefault => "inherited_from_tile_tokens_default",
            Self::InheritedFromTileTokensConfigFile => "inherited_from_tile_tokens_config_file",
            Self::InheritedFromTileTokensEnvironment => "inherited_from_tile_tokens_environment",
            Self::InheritedFromTileTokensCommandLine => "inherited_from_tile_tokens_command_line",
        })
    }
}

const fn streaming_prefill_explicit_source(
    source: ConfigValueSource,
) -> StreamingPrefillEffectiveSource {
    match source {
        ConfigValueSource::Default => StreamingPrefillEffectiveSource::Default,
        ConfigValueSource::ConfigFile => StreamingPrefillEffectiveSource::ConfigFile,
        ConfigValueSource::Environment => StreamingPrefillEffectiveSource::Environment,
        ConfigValueSource::CommandLine => StreamingPrefillEffectiveSource::CommandLine,
    }
}

const fn streaming_prefill_inherited_tile_source(
    source: ConfigValueSource,
) -> StreamingPrefillEffectiveSource {
    match source {
        ConfigValueSource::Default => {
            StreamingPrefillEffectiveSource::InheritedFromTileTokensDefault
        }
        ConfigValueSource::ConfigFile => {
            StreamingPrefillEffectiveSource::InheritedFromTileTokensConfigFile
        }
        ConfigValueSource::Environment => {
            StreamingPrefillEffectiveSource::InheritedFromTileTokensEnvironment
        }
        ConfigValueSource::CommandLine => {
            StreamingPrefillEffectiveSource::InheritedFromTileTokensCommandLine
        }
    }
}

/// Stable machine-readable dispatch rule for diagnostics and the website UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillDispatchRuleDiagnostics {
    pub policy: StreamingPrefillDispatchPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_prompt_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingPrefillDispatchPolicy {
    Never,
    AllNonEmpty,
    PromptTokensAtLeast,
}

impl StreamingPrefillDispatchRuleDiagnostics {
    const fn from_auto_dispatch(dispatch: kiln_model::StreamingPrefillAutoDispatch) -> Self {
        match dispatch {
            kiln_model::StreamingPrefillAutoDispatch::Never => Self {
                policy: StreamingPrefillDispatchPolicy::Never,
                minimum_prompt_tokens: None,
            },
            kiln_model::StreamingPrefillAutoDispatch::PromptTokensAtLeast(tokens) => Self {
                policy: StreamingPrefillDispatchPolicy::PromptTokensAtLeast,
                minimum_prompt_tokens: Some(tokens),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillDispatchDiagnostics {
    pub configured_mode: StreamingPrefillMode,
    pub configured_source: ConfigValueSource,
    pub backend_policy: StreamingPrefillDispatchRuleDiagnostics,
    pub effective: StreamingPrefillDispatchRuleDiagnostics,
    pub effective_source: StreamingPrefillEffectiveSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillThresholdDiagnostics {
    pub configured: Option<usize>,
    pub configured_source: ConfigValueSource,
    pub backend_policy: Option<usize>,
    pub effective_for_auto_mode: Option<usize>,
    pub override_applied_to_backend_auto_policy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillTileDiagnostics {
    pub configured: Option<usize>,
    pub configured_source: ConfigValueSource,
    pub backend_policy: usize,
    pub effective: usize,
    pub effective_source: StreamingPrefillEffectiveSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillDerivedTileDiagnostics {
    pub backend_policy: usize,
    pub effective: usize,
    pub effective_source: StreamingPrefillEffectiveSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillToggleDiagnostics {
    pub configured: bool,
    pub configured_source: ConfigValueSource,
    pub effective: bool,
    pub effective_source: StreamingPrefillEffectiveSource,
}

/// Process-lifetime streaming-prefill policy resolved once after backend
/// selection. The serialized fields are the complete operator-facing contract;
/// the private execution value is the exact policy injected into model code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamingPrefillRuntimeConfig {
    pub dispatch: StreamingPrefillDispatchDiagnostics,
    pub threshold_tokens: StreamingPrefillThresholdDiagnostics,
    pub tile_tokens: StreamingPrefillTileDiagnostics,
    pub tape_tile_tokens: StreamingPrefillTileDiagnostics,
    pub detached_full_attn_tile_tokens: StreamingPrefillTileDiagnostics,
    pub detached_full_attn_boundary_tile_tokens: StreamingPrefillDerivedTileDiagnostics,
    pub detached_full_attn_tape_replay_tile_tokens: StreamingPrefillDerivedTileDiagnostics,
    pub last_token_lm_head: StreamingPrefillToggleDiagnostics,
    pub immutable_after_startup: bool,
    pub restart_required_to_change: bool,
    #[serde(skip)]
    execution_policy: kiln_model::StreamingPrefillExecutionPolicy,
}

impl StreamingPrefillRuntimeConfig {
    pub const fn execution_policy(self) -> kiln_model::StreamingPrefillExecutionPolicy {
        self.execution_policy
    }
}

/// Reject batching-actor settings that can partition the same prompt
/// differently from tiled streaming prefill on a backend whose deterministic
/// output is sensitive to that boundary.
pub fn validate_actor_prefill_tile_contract(
    batching: BatchingRuntimeConfig,
    streaming_prefill: StreamingPrefillRuntimeConfig,
    max_batch_tokens: BatchTokenBudget,
    max_prefill_tokens_per_cycle: PrefillTokenBudget,
    max_decode_batch: usize,
) -> Result<()> {
    if !batching.actor_prefill_tile_alignment_required {
        return Ok(());
    }

    let tile_tokens = streaming_prefill.tile_tokens.effective;
    anyhow::ensure!(
        max_prefill_tokens_per_cycle.tokens() == tile_tokens,
        "server.max_prefill_tokens_per_cycle={} must equal the backend's effective streaming_prefill.tile_tokens={tile_tokens} because the batching actor requires route-invariant prefill chunks",
        max_prefill_tokens_per_cycle.tokens()
    );

    let dispatch_covers_first_split = match streaming_prefill.dispatch.effective.policy {
        StreamingPrefillDispatchPolicy::AllNonEmpty => true,
        StreamingPrefillDispatchPolicy::PromptTokensAtLeast => streaming_prefill
            .dispatch
            .effective
            .minimum_prompt_tokens
            .is_some_and(|minimum| minimum <= tile_tokens),
        StreamingPrefillDispatchPolicy::Never => false,
    };
    anyhow::ensure!(
        dispatch_covers_first_split,
        "streaming_prefill.mode and streaming_prefill.threshold_tokens must enable tiled streaming prefill no later than the first {tile_tokens}-token actor chunk because the batching actor requires route-invariant prefill chunks"
    );

    let required_batch_tokens = tile_tokens
        .checked_add(max_decode_batch)
        .context("actor prefill tile plus decode width overflowed usize")?;
    anyhow::ensure!(
        max_batch_tokens.tokens() >= required_batch_tokens,
        "server.max_batch_tokens={} must be at least {required_batch_tokens} (streaming prefill tile {tile_tokens} + effective decode width {max_decode_batch}) because the batching actor requires one full route-invariant prefill chunk beside ready decode rows",
        max_batch_tokens.tokens()
    );
    Ok(())
}

impl StreamingPrefillConfig {
    /// Resolve typed operator intent over one selected backend's immutable
    /// policy. No lower execution path needs to know how TOML or environment
    /// syntax supplied the values.
    pub fn resolve(
        self,
        backend: kiln_model::StreamingPrefillBackendPolicy,
    ) -> StreamingPrefillRuntimeConfig {
        let execution_mode = match self.mode.mode() {
            StreamingPrefillMode::Auto => kiln_model::StreamingPrefillMode::Auto,
            StreamingPrefillMode::Enabled => kiln_model::StreamingPrefillMode::Enabled,
            StreamingPrefillMode::Disabled => kiln_model::StreamingPrefillMode::Disabled,
        };
        let execution_policy = kiln_model::StreamingPrefillExecutionPolicy::resolve(
            backend,
            execution_mode,
            self.threshold_tokens.configured(),
            self.tile_tokens.configured(),
            self.tape_tile_tokens.configured(),
            self.detached_full_attn_tile_tokens.configured(),
            self.last_token_lm_head.enabled(),
        );

        let backend_dispatch =
            StreamingPrefillDispatchRuleDiagnostics::from_auto_dispatch(backend.auto_dispatch);
        let threshold_override_applied = self.threshold_tokens.configured().is_some()
            && matches!(
                backend.auto_dispatch,
                kiln_model::StreamingPrefillAutoDispatch::PromptTokensAtLeast(_)
            );
        let effective_auto_threshold = execution_policy.threshold_tokens();
        let (effective_dispatch, dispatch_source) = match self.mode.mode() {
            StreamingPrefillMode::Enabled => (
                StreamingPrefillDispatchRuleDiagnostics {
                    policy: StreamingPrefillDispatchPolicy::AllNonEmpty,
                    minimum_prompt_tokens: None,
                },
                streaming_prefill_explicit_source(self.mode.source()),
            ),
            StreamingPrefillMode::Disabled => (
                StreamingPrefillDispatchRuleDiagnostics {
                    policy: StreamingPrefillDispatchPolicy::Never,
                    minimum_prompt_tokens: None,
                },
                streaming_prefill_explicit_source(self.mode.source()),
            ),
            StreamingPrefillMode::Auto => {
                let source = if threshold_override_applied {
                    streaming_prefill_explicit_source(self.threshold_tokens.source())
                } else {
                    StreamingPrefillEffectiveSource::BackendPolicy
                };
                (
                    effective_auto_threshold.map_or(
                        StreamingPrefillDispatchRuleDiagnostics {
                            policy: StreamingPrefillDispatchPolicy::Never,
                            minimum_prompt_tokens: None,
                        },
                        |tokens| StreamingPrefillDispatchRuleDiagnostics {
                            policy: StreamingPrefillDispatchPolicy::PromptTokensAtLeast,
                            minimum_prompt_tokens: Some(tokens),
                        },
                    ),
                    source,
                )
            }
        };

        let base_configured = self.tile_tokens.configured();
        let base_source = base_configured
            .map_or(StreamingPrefillEffectiveSource::BackendPolicy, |_| {
                streaming_prefill_explicit_source(self.tile_tokens.source())
            });
        let tape_source = if self.tape_tile_tokens.configured().is_some() {
            streaming_prefill_explicit_source(self.tape_tile_tokens.source())
        } else if base_configured.is_some() {
            streaming_prefill_inherited_tile_source(self.tile_tokens.source())
        } else {
            StreamingPrefillEffectiveSource::BackendPolicy
        };
        let detached_source = if self.detached_full_attn_tile_tokens.configured().is_some() {
            streaming_prefill_explicit_source(self.detached_full_attn_tile_tokens.source())
        } else if base_configured.is_some() {
            streaming_prefill_inherited_tile_source(self.tile_tokens.source())
        } else {
            StreamingPrefillEffectiveSource::BackendPolicy
        };

        StreamingPrefillRuntimeConfig {
            dispatch: StreamingPrefillDispatchDiagnostics {
                configured_mode: self.mode.mode(),
                configured_source: self.mode.source(),
                backend_policy: backend_dispatch,
                effective: effective_dispatch,
                effective_source: dispatch_source,
            },
            threshold_tokens: StreamingPrefillThresholdDiagnostics {
                configured: self.threshold_tokens.configured(),
                configured_source: self.threshold_tokens.source(),
                backend_policy: backend.auto_dispatch.minimum_prompt_tokens(),
                effective_for_auto_mode: effective_auto_threshold,
                override_applied_to_backend_auto_policy: threshold_override_applied,
            },
            tile_tokens: StreamingPrefillTileDiagnostics {
                configured: base_configured,
                configured_source: self.tile_tokens.source(),
                backend_policy: backend.base_tile_tokens,
                effective: execution_policy.base_tile_tokens(),
                effective_source: base_source,
            },
            tape_tile_tokens: StreamingPrefillTileDiagnostics {
                configured: self.tape_tile_tokens.configured(),
                configured_source: self.tape_tile_tokens.source(),
                backend_policy: backend.tape_tile_tokens,
                effective: execution_policy.tape_tile_tokens(),
                effective_source: tape_source,
            },
            detached_full_attn_tile_tokens: StreamingPrefillTileDiagnostics {
                configured: self.detached_full_attn_tile_tokens.configured(),
                configured_source: self.detached_full_attn_tile_tokens.source(),
                backend_policy: backend.detached_full_attn_tile_tokens,
                effective: execution_policy.detached_full_attn_tile_tokens(),
                effective_source: detached_source,
            },
            detached_full_attn_boundary_tile_tokens: StreamingPrefillDerivedTileDiagnostics {
                backend_policy: backend.detached_full_attn_boundary_tile_tokens,
                effective: execution_policy.detached_full_attn_boundary_tile_tokens(),
                effective_source: detached_source,
            },
            detached_full_attn_tape_replay_tile_tokens: StreamingPrefillDerivedTileDiagnostics {
                backend_policy: backend.detached_full_attn_tape_replay_tile_tokens,
                effective: execution_policy.detached_full_attn_tape_replay_tile_tokens(),
                effective_source: detached_source,
            },
            last_token_lm_head: StreamingPrefillToggleDiagnostics {
                configured: self.last_token_lm_head.enabled(),
                configured_source: self.last_token_lm_head.source(),
                effective: execution_policy.last_token_lm_head(),
                effective_source: streaming_prefill_explicit_source(
                    self.last_token_lm_head.source(),
                ),
            },
            immutable_after_startup: true,
            restart_required_to_change: true,
            execution_policy,
        }
    }
}

/// Adapter-storage settings. Canonical startup overrides use
/// `KILN_ADAPTERS_<FIELD>`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdaptersConfig {
    /// Adapter-library service URL. The current API publishes this resolved
    /// endpoint but remains contract-only until the library backend launches.
    pub library_url: String,
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

fn parse_public_text(_name: &str, raw: &str) -> Result<String> {
    Ok(raw.to_owned())
}

fn parse_public_some_text(_name: &str, raw: &str) -> Result<Option<String>> {
    Ok(Some(raw.to_owned()))
}

fn parse_public_snapshot_dir(_name: &str, raw: &str) -> Result<Option<String>> {
    Ok((!raw.trim().is_empty()).then(|| raw.to_owned()))
}

fn parse_public_empty_clears_text(_name: &str, raw: &str) -> Result<Option<String>> {
    Ok((!raw.is_empty()).then(|| raw.to_owned()))
}

fn parse_public_request_log_dir(name: &str, raw: &str) -> Result<Option<PathBuf>> {
    if raw.trim().is_empty() {
        anyhow::bail!("{name} must be a non-empty path, got {raw:?}");
    }
    Ok(Some(PathBuf::from(raw)))
}

fn parse_public_optional_path(name: &str, raw: &str) -> Result<Option<PathBuf>> {
    if raw.trim().is_empty() {
        anyhow::bail!("{name} must be a non-empty path, got {raw:?}");
    }
    Ok(Some(PathBuf::from(raw)))
}

fn parse_public_local_capability_access(name: &str, raw: &str) -> Result<LocalCapabilityAccess> {
    LocalCapabilityAccess::parse(name, raw)
}

fn parse_public_bool(name: &str, raw: &str) -> Result<bool> {
    parse_required_bool_env(name, raw)
}

fn parse_public_some_bool(name: &str, raw: &str) -> Result<Option<bool>> {
    parse_required_bool_env(name, raw).map(Some)
}

fn parse_public_decimal<T>(name: &str, raw: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: fmt::Display + Send + Sync + 'static,
{
    parse_decimal_env(name, raw, "a decimal value")
}

fn parse_public_some_decimal<T>(name: &str, raw: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: fmt::Display + Send + Sync + 'static,
{
    parse_public_decimal(name, raw).map(Some)
}

fn parse_public_http_send_buffer(name: &str, raw: &str) -> Result<Option<usize>> {
    let value = parse_public_decimal(name, raw)?;
    validate_http_send_buffer_bytes(value)
        .with_context(|| format!("invalid {name} value {raw:?}"))?;
    Ok(Some(value))
}

fn parse_public_spec_method(name: &str, raw: &str) -> Result<SpecMethod> {
    SpecMethod::parse_env(raw)
        .with_context(|| format!("{name} must be off, skip_layer, or mtp, got {raw:?}"))
}

type ApplyPublicEnvValue = fn(&mut KilnConfig, &str, &str) -> Result<()>;

/// One fixed typed leaf in the mechanically named public environment contract.
struct PublicEnvField {
    section: &'static str,
    field: &'static str,
    apply: ApplyPublicEnvValue,
}

impl PublicEnvField {
    fn canonical_name(&self) -> String {
        canonical_env_name(self.section, self.field)
    }

    fn field_path(&self) -> String {
        format!("{}.{}", self.section, self.field)
    }

    fn apply_from_environment(&self, config: &mut KilnConfig) -> Result<bool> {
        let canonical_name = self.canonical_name();
        let Some(raw) = read_optional_unicode_env(&canonical_name)? else {
            return Ok(false);
        };
        (self.apply)(config, &canonical_name, &raw)?;
        Ok(true)
    }
}

/// Derive the public environment spelling mechanically from a typed leaf.
fn canonical_env_name(section: &str, field: &str) -> String {
    format!(
        "KILN_{}_{}",
        section.to_ascii_uppercase(),
        field.to_ascii_uppercase()
    )
}

macro_rules! public_env_parser {
    (text) => {
        parse_public_text
    };
    (some_text) => {
        parse_public_some_text
    };
    (snapshot_dir) => {
        parse_public_snapshot_dir
    };
    (empty_clears_text) => {
        parse_public_empty_clears_text
    };
    (request_log_dir) => {
        parse_public_request_log_dir
    };
    (optional_path) => {
        parse_public_optional_path
    };
    (cache_root) => {
        CacheRootSetting::from_named_environment_value
    };
    (local_capability_access) => {
        parse_public_local_capability_access
    };
    (bool) => {
        parse_public_bool
    };
    (some_bool) => {
        parse_public_some_bool
    };
    (u16) => {
        parse_public_decimal::<u16>
    };
    (u64) => {
        parse_public_decimal::<u64>
    };
    (usize) => {
        parse_public_decimal::<usize>
    };
    (some_usize) => {
        parse_public_some_decimal::<usize>
    };
    (f64) => {
        parse_public_decimal::<f64>
    };
    (some_f64) => {
        parse_public_some_decimal::<f64>
    };
    (optional_capacity) => {
        parse_optional_capacity_env
    };
    (optional_usize) => {
        parse_optional_usize_env
    };
    (optional_u64) => {
        parse_optional_u64_env
    };
    (serving_profile) => {
        ServingProfileSetting::from_named_environment_value
    };
    (kt_api_mode) => {
        KtApiModeSetting::from_named_environment_value
    };
    (full_attention_score_budget_mib) => {
        FullAttentionScoreBudgetMib::from_named_environment_value
    };
    (vulkan_device_index) => {
        VulkanDeviceIndexSetting::from_named_environment_value
    };
    (vulkan_validation) => {
        VulkanValidationSetting::from_named_environment_value
    };
    (cuda_kernel_profile) => {
        CudaKernelProfileSetting::from_named_environment_value
    };
    (cuda_marlin_profile) => {
        CudaMarlinProfileSetting::from_named_environment_value
    };
    (cuda_flash_backward_mode) => {
        CudaFlashBackwardModeSetting::from_named_environment_value
    };
    (metal_kernel_profile) => {
        MetalKernelProfileSetting::from_named_environment_value
    };
    (rocm_synchronization_mode) => {
        RocmSynchronizationModeSetting::from_named_environment_value
    };
    (rocm_strided_batched_matmul_mode) => {
        RocmStridedBatchedMatmulModeSetting::from_named_environment_value
    };
    (rocm_bf16_matmul_output_mode) => {
        RocmBf16MatmulOutputModeSetting::from_named_environment_value
    };
    (rocm_kernel_profile) => {
        RocmKernelProfileSetting::from_named_environment_value
    };
    (rocm_graph_mode) => {
        RocmGraphModeSetting::from_named_environment_value
    };
    (rocm_graph_cache_entries) => {
        RocmGraphCacheEntries::from_named_environment_value
    };
    (rocm_graph_cache_max_bytes) => {
        RocmGraphCacheMaxBytes::from_named_environment_value
    };
    (memory_reclaim_mode) => {
        MemoryReclaimModeSetting::from_named_environment_value
    };
    (kv_autoscale) => {
        KvAutoscaleSetting::from_named_environment_value
    };
    (kv_force_blocks) => {
        KvForceBlocksSetting::from_named_environment_value
    };
    (deterministic) => {
        DeterministicInference::from_named_environment_value
    };
    (http_send_buffer) => {
        parse_public_http_send_buffer
    };
    (stream_stall) => {
        StreamStallGrace::from_named_environment_value
    };
    (batch_tokens) => {
        BatchTokenBudget::from_named_environment_value
    };
    (prefill_tokens) => {
        PrefillTokenBudget::from_named_environment_value
    };
    (prefill_layers) => {
        PrefillLayerBudget::from_named_environment_value
    };
    (max_decode_batch) => {
        MaxDecodeBatch::from_named_environment_value
    };
    (batching_toggle) => {
        BatchingToggle::from_named_environment_value
    };
    (actor_cycle_idle) => {
        ActorCycleIdle::from_named_environment_value
    };
    (prefill_admission_quantum) => {
        PrefillAdmissionQuantum::from_named_environment_value
    };
    (checkpoint_boundary_recompute) => {
        CheckpointBoundaryRecomputeSetting::from_named_environment_value
    };
    (checkpoint_boundary_threshold) => {
        CheckpointBoundaryThresholdSetting::from_named_environment_value
    };
    (checkpoint_boundary_anchor_stride) => {
        CheckpointBoundaryAnchorStrideSetting::from_named_environment_value
    };
    (checkpoint_boundary_cache_gb) => {
        CheckpointBoundaryCacheGbSetting::from_named_environment_value
    };
    (streaming_prefill_mode) => {
        StreamingPrefillModeSetting::from_named_environment_value
    };
    (streaming_prefill_threshold_tokens) => {
        StreamingPrefillThresholdTokens::from_named_environment_value
    };
    (streaming_prefill_tile_tokens) => {
        StreamingPrefillTileTokens::from_named_environment_value
    };
    (streaming_prefill_tape_tile_tokens) => {
        StreamingPrefillTapeTileTokens::from_named_environment_value
    };
    (streaming_prefill_detached_full_attn_tile_tokens) => {
        StreamingPrefillDetachedFullAttnTileTokens::from_named_environment_value
    };
    (streaming_prefill_last_token_lm_head) => {
        StreamingPrefillLastTokenLmHead::from_named_environment_value
    };
    (spec_method) => {
        parse_public_spec_method
    };
}

macro_rules! public_env_field {
    ($kind:ident, $section:ident.$field:ident) => {
        PublicEnvField {
            section: stringify!($section),
            field: stringify!($field),
            apply: |config, name, raw| {
                let value = (public_env_parser!($kind))(name, raw)?;
                config.$section.$field = value;
                Ok(())
            },
        }
    };
}

macro_rules! optional_section_public_env_field {
    ($kind:ident, $section:ident.$field:ident) => {
        PublicEnvField {
            section: stringify!($section),
            field: stringify!($field),
            apply: |config, name, raw| {
                let value = (public_env_parser!($kind))(name, raw)?;
                config.$section.get_or_insert_with(Default::default).$field = value;
                Ok(())
            },
        }
    };
}
/// Complete public environment contract for fixed typed leaves. Every name is
/// derived as `KILN_<SECTION>_<FIELD>`; no alternate spellings are accepted.
static PUBLIC_ENV_FIELDS: &[PublicEnvField] = &[
    public_env_field!(serving_profile, server.serving_profile),
    public_env_field!(kt_api_mode, accelerator.kt_api_mode),
    public_env_field!(
        full_attention_score_budget_mib,
        accelerator.full_attention_score_budget_mib
    ),
    public_env_field!(vulkan_device_index, accelerator.vulkan_device_index),
    public_env_field!(vulkan_validation, accelerator.vulkan_validation),
    public_env_field!(cuda_kernel_profile, accelerator.cuda_kernel_profile),
    public_env_field!(cuda_marlin_profile, accelerator.cuda_marlin_profile),
    public_env_field!(
        cuda_flash_backward_mode,
        accelerator.cuda_flash_backward_mode
    ),
    public_env_field!(metal_kernel_profile, accelerator.metal_kernel_profile),
    public_env_field!(
        rocm_synchronization_mode,
        accelerator.rocm_synchronization_mode
    ),
    public_env_field!(
        rocm_strided_batched_matmul_mode,
        accelerator.rocm_strided_batched_matmul_mode
    ),
    public_env_field!(
        rocm_bf16_matmul_output_mode,
        accelerator.rocm_bf16_matmul_output_mode
    ),
    public_env_field!(rocm_kernel_profile, accelerator.rocm_kernel_profile),
    public_env_field!(rocm_graph_mode, accelerator.rocm_graph_mode),
    public_env_field!(
        rocm_graph_cache_entries,
        accelerator.rocm_graph_cache_entries
    ),
    public_env_field!(
        rocm_graph_cache_max_bytes,
        accelerator.rocm_graph_cache_max_bytes
    ),
    public_env_field!(deterministic, server.deterministic),
    public_env_field!(text, server.host),
    public_env_field!(u16, server.port),
    public_env_field!(u64, server.request_timeout_secs),
    public_env_field!(local_capability_access, server.terminal_access),
    public_env_field!(http_send_buffer, server.http_send_buffer_bytes),
    public_env_field!(stream_stall, server.stream_stall_grace_ms),
    public_env_field!(batch_tokens, server.max_batch_tokens),
    public_env_field!(prefill_tokens, server.max_prefill_tokens_per_cycle),
    public_env_field!(prefill_layers, server.max_prefill_layers_per_cycle),
    public_env_field!(max_decode_batch, server.max_decode_batch),
    public_env_field!(bool, server.eval_mode),
    public_env_field!(bool, server.debug_model_state),
    public_env_field!(some_bool, server.default_thinking_enabled),
    public_env_field!(optional_usize, server.default_thinking_budget_tokens),
    public_env_field!(optional_u64, server.default_thinking_budget_ms),
    public_env_field!(bool, server.fold_reasoning_into_content),
    public_env_field!(bool, server.chat_performance_metadata),
    public_env_field!(bool, server.chat_config_hash_metadata),
    public_env_field!(u64, server.slow_request_warn_secs),
    public_env_field!(u64, server.shutdown_timeout_secs),
    public_env_field!(batching_toggle, batching.rowwise_decode),
    public_env_field!(batching_toggle, batching.prefix_aware_admission),
    public_env_field!(
        prefill_admission_quantum,
        batching.prefill_admission_quantum
    ),
    public_env_field!(actor_cycle_idle, batching.actor_cycle_idle_ms),
    public_env_field!(some_text, model.path),
    public_env_field!(text, model.model_id),
    public_env_field!(some_text, model.tokenizer_path),
    public_env_field!(some_text, model.adapter_dir),
    public_env_field!(snapshot_dir, model.snapshot_dir),
    public_env_field!(optional_u64, model.checkpoint_read_mib_per_second),
    public_env_field!(optional_u64, model.accelerator_weight_upload_mib_per_second),
    public_env_field!(bool, model.vulkan_decode_weight_prewarm),
    public_env_field!(u64, model.vulkan_decode_weight_prewarm_mib_per_second),
    public_env_field!(some_text, model.served_model_id),
    public_env_field!(cache_root, paths.cache_root),
    public_env_field!(some_usize, memory.num_blocks),
    public_env_field!(some_f64, memory.gpu_memory_gb),
    public_env_field!(f64, memory.inference_memory_fraction),
    public_env_field!(some_f64, memory.training_memory_gb),
    public_env_field!(f64, memory.vulkan_buffer_pool_gb),
    public_env_field!(f64, memory.floor_gb),
    public_env_field!(u64, memory.probe_ms),
    public_env_field!(memory_reclaim_mode, memory.reclaim_mode),
    public_env_field!(kv_autoscale, memory.kv_autoscale),
    public_env_field!(kv_force_blocks, memory.kv_force_blocks),
    public_env_field!(bool, memory.kv_cache_fp8),
    public_env_field!(bool, memory.cuda_graphs),
    public_env_field!(usize, memory.cuda_graph_cache_entries),
    public_env_field!(some_usize, training.grad_checkpoint_segments),
    public_env_field!(bool, training.no_grad_checkpoint),
    public_env_field!(
        checkpoint_boundary_recompute,
        training.recompute_checkpoint_boundaries
    ),
    public_env_field!(
        checkpoint_boundary_threshold,
        training.recompute_boundary_threshold_tokens
    ),
    public_env_field!(
        checkpoint_boundary_anchor_stride,
        training.checkpoint_boundary_anchor_stride
    ),
    public_env_field!(
        checkpoint_boundary_cache_gb,
        training.checkpoint_boundary_cache_gb
    ),
    public_env_field!(some_usize, training.checkpoint_interval),
    public_env_field!(empty_clears_text, training.webhook_url),
    public_env_field!(optional_path, training.logit_cache_dir),
    public_env_field!(usize, training.max_queued_jobs),
    public_env_field!(usize, training.max_tracked_jobs),
    public_env_field!(u64, training.tracked_job_ttl_secs),
    public_env_field!(bool, openenv.enabled),
    public_env_field!(usize, openenv.max_active_runs),
    public_env_field!(usize, openenv.max_tracked_runs),
    public_env_field!(u64, openenv.tracked_run_ttl_secs),
    public_env_field!(bool, openenv.allow_remote_environments),
    public_env_field!(text, logging.level),
    public_env_field!(text, logging.format),
    public_env_field!(bool, prefix_cache.enabled),
    public_env_field!(some_usize, prefix_cache.max_blocks),
    public_env_field!(some_usize, prefix_cache.max_entries),
    public_env_field!(spec_method, speculative.method),
    public_env_field!(usize, speculative.num_speculative_tokens),
    public_env_field!(usize, speculative.draft_layers),
    public_env_field!(streaming_prefill_mode, streaming_prefill.mode),
    public_env_field!(
        streaming_prefill_threshold_tokens,
        streaming_prefill.threshold_tokens
    ),
    public_env_field!(streaming_prefill_tile_tokens, streaming_prefill.tile_tokens),
    public_env_field!(
        streaming_prefill_tape_tile_tokens,
        streaming_prefill.tape_tile_tokens
    ),
    public_env_field!(
        streaming_prefill_detached_full_attn_tile_tokens,
        streaming_prefill.detached_full_attn_tile_tokens
    ),
    public_env_field!(
        streaming_prefill_last_token_lm_head,
        streaming_prefill.last_token_lm_head
    ),
    public_env_field!(optional_capacity, adapters.max_disk_bytes),
    public_env_field!(optional_capacity, adapters.composed_cache_max_bytes),
    public_env_field!(optional_capacity, adapters.composed_cache_max_entries),
    public_env_field!(text, adapters.library_url),
    public_env_field!(bool, request_log.enabled),
    public_env_field!(request_log_dir, request_log.dir),
    public_env_field!(u64, request_log.max_file_bytes),
    public_env_field!(u64, request_log.max_total_bytes),
    public_env_field!(bool, request_log.compress),
    public_env_field!(usize, request_log.max_capture_bytes),
    optional_section_public_env_field!(optional_u64, agent.self_improve_interval_hours),
    optional_section_public_env_field!(usize, agent.max_concurrent_runs),
    optional_section_public_env_field!(u64, agent.run_timeout_secs),
    optional_section_public_env_field!(local_capability_access, agent.runs_access),
    optional_section_public_env_field!(optional_path, agent.pi_bin),
    optional_section_public_env_field!(optional_path, agent.pi_sessions_dir),
];

const CONFIG_FILE_ONLY_FIXED_FIELDS: &[&str] = &[
    "agent.self_improve",
    "eval.eval_dir",
    "eval.max_queued_jobs",
    "eval.max_tracked_jobs",
    "eval.webhook_url",
];

const REDACTED_EFFECTIVE_CONFIGURATION_FIELDS: &[&str] = &[
    "agent.self_improve",
    "eval.webhook_url",
    "training.webhook_url",
];

// --- Defaults ---

impl Default for KilnConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            accelerator: AcceleratorRuntimeConfig::default(),
            batching: BatchingConfig::default(),
            model: ModelConfig::default(),
            paths: PathsConfig::default(),
            memory: MemoryConfig::default(),
            training: TrainingConfig::default(),
            openenv: OpenEnvConfig::default(),
            logging: LoggingConfig::default(),
            prefix_cache: PrefixCacheConfig::default(),
            speculative: SpeculativeDecodingConfig::default(),
            streaming_prefill: StreamingPrefillConfig::default(),
            adapters: AdaptersConfig::default(),
            teachers: TeachersConfig::default(),
            eval: None,
            request_log: crate::request_log::RequestLogConfig::default(),
            agent: None,
            value_sources: BTreeMap::new(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            serving_profile: ServingProfileSetting::default(),
            deterministic: DeterministicInference::default(),
            host: DEFAULT_SERVER_HOST.into(),
            port: DEFAULT_SERVER_PORT,
            request_timeout_secs: 600,
            terminal_access: LocalCapabilityAccess::default(),
            http_send_buffer_bytes: None,
            stream_stall_grace_ms: StreamStallGrace::default(),
            max_batch_tokens: BatchTokenBudget::default(),
            max_prefill_tokens_per_cycle: PrefillTokenBudget::default(),
            max_prefill_layers_per_cycle: PrefillLayerBudget::default(),
            max_decode_batch: MaxDecodeBatch::default(),
            eval_mode: false,
            debug_model_state: false,
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
            checkpoint_read_mib_per_second: None,
            accelerator_weight_upload_mib_per_second: None,
            vulkan_decode_weight_prewarm: true,
            vulkan_decode_weight_prewarm_mib_per_second: 256,
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
            vulkan_buffer_pool_gb: 3.0,
            floor_gb: 1.0,
            probe_ms: 500,
            reclaim_mode: MemoryReclaimModeSetting::default(),
            kv_autoscale: KvAutoscaleSetting::default(),
            kv_force_blocks: KvForceBlocksSetting::default(),
            kv_cache_fp8: false,
            // Default-ON (#34): CUDA graph capture/replay is now bit-identical to
            // eager decode. BUG2 (the replay divergence) was the captured graph
            // filling its RoPE cos/sin tables with host CPU cos/sin while eager
            // uses GPU kt_cos/kt_sin — now both compute on-device. Verified
            // bit-identical over 512-token decodes (BF16 + W4A16, multiple prompts).
            cuda_graphs: true,
            cuda_graph_cache_entries: 8,
        }
    }
}

impl MemoryConfig {
    /// Configured Vulkan recycler cap in bytes.
    pub fn vulkan_buffer_pool_bytes(&self) -> u64 {
        (self.vulkan_buffer_pool_gb * 1024.0 * 1024.0 * 1024.0).round() as u64
    }

    /// Configured governor floor in bytes, using the same GiB conversion and
    /// rounding as the installed runtime policy.
    pub fn floor_bytes(&self) -> u64 {
        (self.floor_gb * 1024.0 * 1024.0 * 1024.0).round() as u64
    }

    /// Translate the validated typed startup configuration into the
    /// dependency-light governor's immutable runtime policy.
    pub fn governor_config(&self) -> kiln_memory::GovernorConfig {
        let mut governor = kiln_memory::GovernorConfig::default();
        governor.floor_bytes = self.floor_bytes();
        governor.ttl = std::time::Duration::from_millis(self.probe_ms);
        governor.reclaim_mode = self.reclaim_mode.mode();
        governor
    }

    /// Bind the governor to the already-resolved cap-only effective capacity.
    /// A zero capacity remains an explicit fail-closed ceiling.
    pub fn governor_config_for_capacity(
        &self,
        effective_capacity_bytes: u64,
    ) -> kiln_memory::GovernorConfig {
        let mut governor = self.governor_config();
        governor.capacity_limit_bytes = Some(effective_capacity_bytes);
        governor
    }
}

impl TrainingConfig {
    /// Resolve the typed startup configuration into the immutable policy used
    /// by admission, execution, and exact-resume planning identity.
    pub fn checkpoint_boundary_policy(&self) -> Result<kiln_train::CheckpointBoundaryPolicy> {
        kiln_train::CheckpointBoundaryPolicy::from_parts(
            self.recompute_checkpoint_boundaries.mode(),
            self.recompute_boundary_threshold_tokens.tokens(),
            self.checkpoint_boundary_anchor_stride.configured(),
            self.checkpoint_boundary_cache_gb.bytes(),
        )
        .context("invalid training checkpoint-boundary policy")
    }

    pub const fn checkpoint_boundary_diagnostics(&self) -> CheckpointBoundaryConfigDiagnostics {
        CheckpointBoundaryConfigDiagnostics {
            recompute_checkpoint_boundaries: self.recompute_checkpoint_boundaries.mode(),
            recompute_checkpoint_boundaries_source: self.recompute_checkpoint_boundaries.source(),
            recompute_boundary_threshold_tokens: self.recompute_boundary_threshold_tokens.tokens(),
            recompute_boundary_threshold_tokens_source: self
                .recompute_boundary_threshold_tokens
                .source(),
            checkpoint_boundary_anchor_stride: self.checkpoint_boundary_anchor_stride.configured(),
            checkpoint_boundary_anchor_stride_source: self
                .checkpoint_boundary_anchor_stride
                .source(),
            checkpoint_boundary_cache_gb: self.checkpoint_boundary_cache_gb.gib(),
            checkpoint_boundary_cache_gb_source: self.checkpoint_boundary_cache_gb.source(),
            checkpoint_boundary_cache_bytes: self.checkpoint_boundary_cache_gb.bytes(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            grad_checkpoint_segments: None,
            no_grad_checkpoint: false,
            recompute_checkpoint_boundaries: CheckpointBoundaryRecomputeSetting::default(),
            recompute_boundary_threshold_tokens: CheckpointBoundaryThresholdSetting::default(),
            checkpoint_boundary_anchor_stride: CheckpointBoundaryAnchorStrideSetting::default(),
            checkpoint_boundary_cache_gb: CheckpointBoundaryCacheGbSetting::default(),
            checkpoint_interval: None,
            webhook_url: None,
            logit_cache_dir: None,
            max_queued_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl_secs: 604_800,
        }
    }
}

impl Default for OpenEnvConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_active_runs: 4,
            max_tracked_runs: 128,
            tracked_run_ttl_secs: 604_800,
            allow_remote_environments: false,
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

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            rowwise_decode: BatchingToggle::new(DEFAULT_ROWWISE_DECODE, ConfigValueSource::Default),
            prefix_aware_admission: BatchingToggle::new(
                DEFAULT_PREFIX_AWARE_ADMISSION,
                ConfigValueSource::Default,
            ),
            prefill_admission_quantum: PrefillAdmissionQuantum::default(),
            actor_cycle_idle_ms: ActorCycleIdle::default(),
        }
    }
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            method: SpecMethod::Off,
            num_speculative_tokens: MAX_SPECULATIVE_DRAFT_TOKENS,
            draft_layers: 8,
        }
    }
}

impl SpeculativeDecodingConfig {
    /// Validate model-dependent draft geometry before weights are loaded.
    ///
    /// MTP still uses the skip-layer implementation as its long-prompt
    /// fallback, so every enabled method must carry a valid draft depth.
    pub fn validate_for_model(&self, model: &kiln_core::config::ModelConfig) -> Result<()> {
        if self.method == SpecMethod::Off {
            return Ok(());
        }
        if self.draft_layers >= model.num_layers {
            anyhow::bail!(
                "speculative.draft_layers must be less than the selected model's {} layers when speculative decoding is enabled, got {}",
                model.num_layers,
                self.draft_layers
            );
        }
        Ok(())
    }

    /// Reject serving methods that have not passed the local accelerator gate.
    ///
    /// Keep this policy beside the typed setting so `serve`, `config check`,
    /// and other product surfaces cannot disagree about availability.
    pub fn validate_for_serving(&self) -> Result<()> {
        let requested = self.method;
        if requested != SpecMethod::Off {
            anyhow::bail!(
                "speculative decoding method {requested:?} is not available for serving until its cancellation, owner-settlement, EOS, context-capacity, and burst-admission contracts pass local accelerator qualification; set speculative.method=\"off\""
            );
        }
        Ok(())
    }

    /// Resolve the effective speculative-decoding method.
    pub fn effective_method(&self) -> SpecMethod {
        self.method
    }
}

impl Default for StreamingPrefillConfig {
    fn default() -> Self {
        Self {
            mode: StreamingPrefillModeSetting::default(),
            threshold_tokens: StreamingPrefillThresholdTokens::default(),
            tile_tokens: StreamingPrefillTileTokens::default(),
            tape_tile_tokens: StreamingPrefillTapeTileTokens::default(),
            detached_full_attn_tile_tokens: StreamingPrefillDetachedFullAttnTileTokens::default(),
            last_token_lm_head: StreamingPrefillLastTokenLmHead::default(),
        }
    }
}

impl Default for AdaptersConfig {
    fn default() -> Self {
        Self {
            library_url: DEFAULT_ADAPTER_LIBRARY_URL.to_owned(),
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

fn collect_toml_configuration_paths(
    prefix: &str,
    value: &toml::Value,
    paths: &mut BTreeSet<String>,
) {
    // This field is one typed JSON value. Its internal keys are request data,
    // not additional configuration-schema leaves.
    if prefix == "agent.self_improve" {
        paths.insert(prefix.to_owned());
        return;
    }
    if let toml::Value::Table(table) = value {
        for (name, child) in table {
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            collect_toml_configuration_paths(&path, child, paths);
        }
    } else if !prefix.is_empty() {
        paths.insert(prefix.to_owned());
    }
}

fn json_value_at_configuration_path<'a>(
    root: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    path.split('.')
        .try_fold(root, |value, component| value.as_object()?.get(component))
}

impl KilnConfig {
    /// Load configuration from an optional file path, then apply env var overrides.
    ///
    /// Resolution order for the file path:
    /// 1. `path` argument (if `Some`)
    /// 2. `KILN_CONFIG` env var
    /// 3. `./kiln.toml` (only if it exists)
    /// 4. No file — defaults only
    pub fn load(path: Option<&str>) -> Result<Self> {
        let result = Self::load_inner(path);
        if let Err(error) = &result {
            let config_path = path
                .map(String::from)
                .or_else(|| std::env::var("KILN_CONFIG").ok())
                .or_else(|| {
                    Path::new("kiln.toml")
                        .exists()
                        .then(|| "kiln.toml".to_string())
                });
            tracing::error!(
                config_path = config_path.as_deref().unwrap_or("<defaults>"),
                error = %format!("{error:#}"),
                "configuration_load_failed"
            );
        }
        result
    }

    /// Apply `serve` command-line values after file and environment
    /// resolution. This keeps CLI precedence typed and avoids process-global
    /// environment mutation.
    pub fn apply_serve_cli_overrides(
        &mut self,
        served_model_id: Option<&str>,
        eval_mode: bool,
    ) -> Result<()> {
        if let Some(served_model_id) = served_model_id {
            self.model.served_model_id = Some(served_model_id.to_owned());
            self.value_sources.insert(
                "model.served_model_id".to_owned(),
                ConfigValueSource::CommandLine,
            );
        }
        if eval_mode {
            self.server.eval_mode = true;
            self.value_sources.insert(
                "server.eval_mode".to_owned(),
                ConfigValueSource::CommandLine,
            );
        }
        self.validate()
    }

    fn load_inner(path: Option<&str>) -> Result<Self> {
        let config_path = match path {
            Some(path) => Some(path.to_string()),
            None => read_optional_unicode_env("KILN_CONFIG")?,
        };

        let mut config = if let Some(ref p) = config_path {
            let contents = std::fs::read_to_string(p)
                .with_context(|| format!("failed to read config file: {p}"))?;
            Self::from_toml_with_sources(&contents)
                .with_context(|| format!("failed to parse config file: {p}"))?
        } else if Path::new("kiln.toml").exists() {
            let contents =
                std::fs::read_to_string("kiln.toml").context("failed to read kiln.toml")?;
            Self::from_toml_with_sources(&contents).context("failed to parse kiln.toml")?
        } else {
            Self::default()
        };

        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    fn from_toml_with_sources(contents: &str) -> Result<Self> {
        let document = toml::from_str::<toml::Value>(contents)?;
        let mut configured_paths = BTreeSet::new();
        collect_toml_configuration_paths("", &document, &mut configured_paths);
        let mut config: Self = document.try_into()?;
        config.value_sources = configured_paths
            .into_iter()
            .map(|path| (path, ConfigValueSource::ConfigFile))
            .collect();
        Ok(config)
    }

    /// Return every typed startup leaf after file, environment, and CLI
    /// precedence. Runtime-derived backend/capacity adjustments remain beside
    /// this object in `/v1/config`; this snapshot is their complete input.
    pub fn effective_configuration(&self) -> Result<EffectiveConfiguration> {
        let mut serialized = serde_json::to_value(self)
            .context("failed to serialize the effective typed configuration")?;
        let root = serialized
            .as_object_mut()
            .context("effective typed configuration did not serialize as an object")?;
        if root.get("eval").is_none_or(serde_json::Value::is_null) {
            root.insert(
                "eval".to_owned(),
                serde_json::to_value(EvalConfig::default())?,
            );
        }
        if root.get("agent").is_none_or(serde_json::Value::is_null) {
            root.insert(
                "agent".to_owned(),
                serde_json::to_value(AgentConfig::default())?,
            );
        }

        let mut fixed_paths = PUBLIC_ENV_FIELDS
            .iter()
            .map(PublicEnvField::field_path)
            .chain(
                CONFIG_FILE_ONLY_FIXED_FIELDS
                    .iter()
                    .map(|path| (*path).to_owned()),
            )
            .collect::<Vec<_>>();
        fixed_paths.sort();
        fixed_paths.dedup();
        anyhow::ensure!(
            fixed_paths.len() == EFFECTIVE_CONFIGURATION_FIXED_FIELD_COUNT,
            "effective configuration registry expected {} fixed fields, found {}",
            EFFECTIVE_CONFIGURATION_FIXED_FIELD_COUNT,
            fixed_paths.len()
        );

        let mut fields = BTreeMap::new();
        for path in fixed_paths {
            let value = json_value_at_configuration_path(&serialized, &path)
                .with_context(|| format!("effective configuration is missing typed field {path}"))?
                .clone();
            let (section, field_name) = path
                .split_once('.')
                .with_context(|| format!("invalid effective configuration field path {path}"))?;
            let public = PUBLIC_ENV_FIELDS
                .iter()
                .find(|field| field.section == section && field.field == field_name);
            let redacted = REDACTED_EFFECTIVE_CONFIGURATION_FIELDS.contains(&path.as_str());
            fields.insert(
                path.clone(),
                EffectiveConfigurationField {
                    effective_value: if redacted {
                        serde_json::Value::Null
                    } else {
                        value
                    },
                    source: self
                        .value_sources
                        .get(&path)
                        .copied()
                        .unwrap_or(ConfigValueSource::Default),
                    canonical_environment: public.map(PublicEnvField::canonical_name),
                    compatibility_environment: Vec::new(),
                    redacted,
                    restart_required_to_change: true,
                },
            );
        }

        for (credential_id, credential) in &self.teachers.credentials {
            for (field, value, redacted) in [
                (
                    "origin",
                    serde_json::Value::String(credential.origin.clone()),
                    false,
                ),
                ("api_key_env", serde_json::Value::Null, true),
            ] {
                let path = format!("teachers.credentials.{credential_id}.{field}");
                fields.insert(
                    path.clone(),
                    EffectiveConfigurationField {
                        effective_value: value,
                        source: self
                            .value_sources
                            .get(&path)
                            .copied()
                            .unwrap_or(ConfigValueSource::Default),
                        canonical_environment: None,
                        compatibility_environment: Vec::new(),
                        redacted,
                        restart_required_to_change: true,
                    },
                );
            }
        }

        Ok(EffectiveConfiguration {
            schema_id: EFFECTIVE_CONFIGURATION_SCHEMA_ID,
            schema_version: EFFECTIVE_CONFIGURATION_SCHEMA_VERSION,
            effective_config_hash: kiln_core::config_hashes::effective_config_hash(self),
            fixed_field_count: EFFECTIVE_CONFIGURATION_FIXED_FIELD_COUNT,
            dynamic_field_count: fields
                .len()
                .saturating_sub(EFFECTIVE_CONFIGURATION_FIXED_FIELD_COUNT),
            all_fields_restart_required_to_change: true,
            fields,
        })
    }

    fn apply_serving_profile_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.serving_profile = ServingProfileSetting::from_environment_value(raw)?;
        }
        Ok(())
    }

    fn apply_deterministic_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.deterministic = DeterministicInference::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Resolve the complete fixed-field public startup environment contract.
    fn apply_env_overrides(&mut self) -> Result<()> {
        for field in PUBLIC_ENV_FIELDS {
            if field.apply_from_environment(self)? {
                self.value_sources
                    .insert(field.field_path(), ConfigValueSource::Environment);
            }
        }
        Ok(())
    }

    fn apply_stream_stall_grace_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.stream_stall_grace_ms = StreamStallGrace::from_environment_value(raw)?;
        }
        Ok(())
    }

    fn apply_max_batch_tokens_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_batch_tokens = BatchTokenBudget::from_environment_value(raw)?;
        }
        Ok(())
    }

    fn apply_max_prefill_tokens_per_cycle_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_prefill_tokens_per_cycle =
                PrefillTokenBudget::from_environment_value(raw)?;
        }
        Ok(())
    }

    fn apply_max_prefill_layers_per_cycle_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_prefill_layers_per_cycle =
                PrefillLayerBudget::from_environment_value(raw)?;
        }
        Ok(())
    }

    fn apply_max_decode_batch_env_value(&mut self, raw: Option<&str>) -> Result<()> {
        if let Some(raw) = raw {
            self.server.max_decode_batch = MaxDecodeBatch::from_environment_value(raw)?;
        }
        Ok(())
    }

    /// Resolve request-handler operational policy once, after the effective
    /// adapter directory is known. This is the only post-load startup boundary
    /// that consults PATH; handlers retain the immutable result.
    pub fn resolve_operational_runtime(
        &self,
        adapter_dir: &Path,
    ) -> Result<OperationalRuntimeConfig> {
        let agent = self.agent.clone().unwrap_or_default();
        let absolute = |path: PathBuf| std::path::absolute(&path).unwrap_or(path);
        let configured_pi = agent.pi_bin.map(absolute);
        let search_path = std::env::var_os("PATH");
        let pi_bin =
            crate::pi_rpc::find_pi(configured_pi.as_deref(), search_path.as_deref()).map(absolute);
        let pi_sessions_dir = agent
            .pi_sessions_dir
            .map(absolute)
            .or_else(|| {
                kiln_resource::user_home_dir()
                    .map(|home| absolute(home.join(".pi").join("agent").join("sessions")))
            })
            .unwrap_or_else(|| {
                std::env::temp_dir()
                    .join("pi")
                    .join("agent")
                    .join("sessions")
            });
        let logit_cache_dir = self
            .training
            .logit_cache_dir
            .clone()
            .map(absolute)
            .unwrap_or_else(|| {
                absolute(
                    adapter_dir
                        .parent()
                        .map(|parent| parent.join("logit-cache"))
                        .unwrap_or_else(|| PathBuf::from("logit-cache")),
                )
            });
        let bind_host = self.server.host.clone();

        Ok(OperationalRuntimeConfig {
            terminal_access: self.server.terminal_access,
            terminal_enabled: self.server.terminal_access.enabled_for_host(&bind_host),
            agent_runs_access: agent.runs_access,
            agent_runs_enabled: agent.runs_access.enabled_for_host(&bind_host),
            bind_host,
            pi_bin,
            pi_sessions_dir,
            adapter_library_url: self.adapters.library_url.clone(),
            logit_cache_dir,
        })
    }

    /// Validate configuration values. Returns an error describing the first invalid value.
    fn validate(&self) -> Result<()> {
        self.accelerator
            .validate_for_serving_profile(self.server.serving_profile.profile())?;
        if self.server.host.trim().is_empty() {
            anyhow::bail!("server.host must be non-empty, got {:?}", self.server.host);
        }
        if self.server.port == 0 {
            anyhow::bail!("server.port must be > 0, got {}", self.server.port);
        }
        if self.server.request_timeout_secs == 0 {
            anyhow::bail!(
                "server.request_timeout_secs must be > 0, got {}",
                self.server.request_timeout_secs
            );
        }
        if let Some(bytes) = self.server.http_send_buffer_bytes {
            validate_http_send_buffer_bytes(bytes)?;
        }
        validate_stream_stall_grace_ms(self.server.stream_stall_grace_ms.millis())?;
        validate_max_batch_tokens(self.server.max_batch_tokens.tokens())?;
        validate_max_prefill_tokens_per_cycle(self.server.max_prefill_tokens_per_cycle.tokens())?;
        validate_max_prefill_layers_per_cycle(self.server.max_prefill_layers_per_cycle.layers())?;
        if let Some(quantum) = self.batching.prefill_admission_quantum.configured() {
            validate_prefill_admission_quantum(quantum)?;
        }
        validate_actor_cycle_idle_ms(self.batching.actor_cycle_idle_ms.millis())?;
        if self.server.shutdown_timeout_secs == 0 {
            anyhow::bail!(
                "server.shutdown_timeout_secs must be > 0, got {}",
                self.server.shutdown_timeout_secs
            );
        }

        if self.model.model_id.trim().is_empty() {
            anyhow::bail!(
                "model.model_id must be non-empty, got {:?}",
                self.model.model_id
            );
        }
        if self
            .model
            .served_model_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            anyhow::bail!(
                "model.served_model_id must be non-empty when set, got {:?}",
                self.model.served_model_id
            );
        }
        for (field, value) in [
            ("model.path", self.model.path.as_deref()),
            ("model.tokenizer_path", self.model.tokenizer_path.as_deref()),
            ("model.adapter_dir", self.model.adapter_dir.as_deref()),
            ("model.snapshot_dir", self.model.snapshot_dir.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                anyhow::bail!("{field} must be non-empty when set, got {value:?}");
            }
        }
        for (field, rate) in [
            (
                "model.checkpoint_read_mib_per_second",
                self.model.checkpoint_read_mib_per_second,
            ),
            (
                "model.accelerator_weight_upload_mib_per_second",
                self.model.accelerator_weight_upload_mib_per_second,
            ),
        ] {
            if let Some(rate) = rate
                && !(1..=16_384).contains(&rate)
            {
                anyhow::bail!("{field} must be in 1..=16384 when set, got {rate}");
            }
        }
        if !(1..=16_384).contains(&self.model.vulkan_decode_weight_prewarm_mib_per_second) {
            anyhow::bail!(
                "model.vulkan_decode_weight_prewarm_mib_per_second must be in 1..=16384, got {}",
                self.model.vulkan_decode_weight_prewarm_mib_per_second
            );
        }

        if self.memory.num_blocks == Some(0) {
            anyhow::bail!("memory.num_blocks must be > 0 when set, got Some(0)");
        }
        for (field, value) in [
            ("memory.gpu_memory_gb", self.memory.gpu_memory_gb),
            ("memory.training_memory_gb", self.memory.training_memory_gb),
        ] {
            if let Some(value) = value
                && (!value.is_finite()
                    || value <= 0.0
                    || value > u64::MAX as f64 / (1024.0 * 1024.0 * 1024.0))
            {
                anyhow::bail!(
                    "{field} must be finite, > 0, and representable as bytes when set, got {value}"
                );
            }
        }

        let f = self.memory.inference_memory_fraction;
        if !f.is_finite() || !(0.0..=1.0).contains(&f) {
            anyhow::bail!("memory.inference_memory_fraction must be between 0.0 and 1.0, got {f}");
        }
        let floor_gb = self.memory.floor_gb;
        let max_floor_gb = u64::MAX as f64 / (1024.0 * 1024.0 * 1024.0);
        if !floor_gb.is_finite() || floor_gb < 0.0 || floor_gb > max_floor_gb {
            anyhow::bail!(
                "memory.floor_gb must be finite, non-negative, and representable as bytes, got {floor_gb}"
            );
        }
        let vulkan_pool_gb = self.memory.vulkan_buffer_pool_gb;
        if !vulkan_pool_gb.is_finite() || vulkan_pool_gb < 0.0 || vulkan_pool_gb > max_floor_gb {
            anyhow::bail!(
                "memory.vulkan_buffer_pool_gb must be finite, non-negative, and representable as bytes, got {vulkan_pool_gb}"
            );
        }
        if self.memory.probe_ms == 0 {
            anyhow::bail!("memory.probe_ms must be > 0, got 0");
        }
        if !(1..=kiln_model::CudaGraphExecutionPolicy::MAX_CACHED_GRAPHS)
            .contains(&self.memory.cuda_graph_cache_entries)
        {
            anyhow::bail!(
                "memory.cuda_graph_cache_entries must be in 1..={}, got {}",
                kiln_model::CudaGraphExecutionPolicy::MAX_CACHED_GRAPHS,
                self.memory.cuda_graph_cache_entries
            );
        }
        if self.memory.kv_force_blocks.target().is_some() {
            if !self.memory.kv_autoscale.enabled() {
                anyhow::bail!("memory.kv_force_blocks requires memory.kv_autoscale=true");
            }
            if self.server.serving_profile.profile() != ServingProfile::Maintenance {
                anyhow::bail!("memory.kv_force_blocks requires server.serving_profile=maintenance");
            }
        }

        if self.training.grad_checkpoint_segments == Some(0) {
            anyhow::bail!("training.grad_checkpoint_segments must be > 0 when set, got Some(0)");
        }
        self.training.checkpoint_boundary_policy()?;
        if self.training.checkpoint_interval == Some(0) {
            anyhow::bail!("training.checkpoint_interval must be > 0 when set, got Some(0)");
        }
        if self.training.max_queued_jobs == 0 {
            anyhow::bail!(
                "training.max_queued_jobs must be > 0, got {}",
                self.training.max_queued_jobs
            );
        }
        if self.training.max_tracked_jobs == 0 {
            anyhow::bail!(
                "training.max_tracked_jobs must be > 0, got {}",
                self.training.max_tracked_jobs
            );
        }
        if self.training.tracked_job_ttl_secs == 0 {
            anyhow::bail!(
                "training.tracked_job_ttl_secs must be > 0, got {}",
                self.training.tracked_job_ttl_secs
            );
        }
        if self.training.max_tracked_jobs < self.training.max_queued_jobs {
            anyhow::bail!(
                "training.max_tracked_jobs must be at least training.max_queued_jobs ({}), got {}",
                self.training.max_queued_jobs,
                self.training.max_tracked_jobs
            );
        }
        if self.openenv.max_active_runs == 0 {
            anyhow::bail!(
                "openenv.max_active_runs must be > 0, got {}",
                self.openenv.max_active_runs
            );
        }
        if self.openenv.max_tracked_runs == 0 {
            anyhow::bail!(
                "openenv.max_tracked_runs must be > 0, got {}",
                self.openenv.max_tracked_runs
            );
        }
        if self.openenv.max_tracked_runs < self.openenv.max_active_runs {
            anyhow::bail!(
                "openenv.max_tracked_runs must be at least openenv.max_active_runs ({}), got {}",
                self.openenv.max_active_runs,
                self.openenv.max_tracked_runs
            );
        }
        if self.openenv.tracked_run_ttl_secs == 0 {
            anyhow::bail!(
                "openenv.tracked_run_ttl_secs must be > 0, got {}",
                self.openenv.tracked_run_ttl_secs
            );
        }
        validate_optional_webhook_url(
            "training.webhook_url",
            self.training.webhook_url.as_deref(),
        )?;
        if self
            .training
            .logit_cache_dir
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            anyhow::bail!("training.logit_cache_dir must be non-empty when set");
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        let level = self.logging.level.to_ascii_lowercase();
        if !valid_levels.contains(&level.as_str()) && !self.logging.level.contains('=') {
            anyhow::bail!(
                "logging.level must be one of {valid_levels:?} or a tracing filter directive, got {:?}",
                self.logging.level
            );
        }
        tracing_subscriber::EnvFilter::try_new(&self.logging.level).with_context(|| {
            format!(
                "logging.level must be a valid tracing filter directive, got {:?}",
                self.logging.level
            )
        })?;

        let valid_formats = ["auto", "json", "pretty", "text", "human"];
        if !valid_formats.contains(&self.logging.format.as_str()) {
            anyhow::bail!(
                "logging.format must be one of {valid_formats:?}, got {:?}",
                self.logging.format
            );
        }

        if self.speculative.num_speculative_tokens == 0 {
            anyhow::bail!(
                "speculative.num_speculative_tokens must be > 0, got {}",
                self.speculative.num_speculative_tokens
            );
        }
        if self.speculative.num_speculative_tokens > MAX_SPECULATIVE_DRAFT_TOKENS {
            anyhow::bail!(
                "speculative.num_speculative_tokens must be <= {}, got {}",
                MAX_SPECULATIVE_DRAFT_TOKENS,
                self.speculative.num_speculative_tokens
            );
        }
        if self.speculative.draft_layers == 0 {
            anyhow::bail!(
                "speculative.draft_layers must be > 0, got {}",
                self.speculative.draft_layers
            );
        }
        if let Some(tokens) = self.streaming_prefill.threshold_tokens.configured() {
            validate_streaming_prefill_positive_tokens(
                "streaming_prefill.threshold_tokens",
                tokens,
            )?;
        }
        for (field, tokens) in [
            (
                "streaming_prefill.tile_tokens",
                self.streaming_prefill.tile_tokens.configured(),
            ),
            (
                "streaming_prefill.tape_tile_tokens",
                self.streaming_prefill.tape_tile_tokens.configured(),
            ),
            (
                "streaming_prefill.detached_full_attn_tile_tokens",
                self.streaming_prefill
                    .detached_full_attn_tile_tokens
                    .configured(),
            ),
        ] {
            if let Some(tokens) = tokens {
                validate_streaming_prefill_tile_tokens(field, tokens)?;
            }
        }

        if self.prefix_cache.max_blocks == Some(0) {
            anyhow::bail!("prefix_cache.max_blocks must be > 0 when set, got Some(0)");
        }
        if self.prefix_cache.max_entries == Some(0) {
            anyhow::bail!("prefix_cache.max_entries must be > 0 when set, got Some(0)");
        }

        validate_required_http_url("adapters.library_url", &self.adapters.library_url)?;

        self.teachers.validate()?;

        if let Some(eval) = &self.eval {
            if eval
                .eval_dir
                .as_ref()
                .is_some_and(|path| path.as_os_str().is_empty())
            {
                anyhow::bail!("eval.eval_dir must be non-empty when set, got an empty path");
            }
            if eval.max_queued_jobs == 0 {
                anyhow::bail!(
                    "eval.max_queued_jobs must be > 0, got {}",
                    eval.max_queued_jobs
                );
            }
            if eval.max_tracked_jobs == 0 {
                anyhow::bail!(
                    "eval.max_tracked_jobs must be > 0, got {}",
                    eval.max_tracked_jobs
                );
            }
            if eval.max_tracked_jobs < eval.max_queued_jobs {
                anyhow::bail!(
                    "eval.max_tracked_jobs must be at least eval.max_queued_jobs ({}), got {}",
                    eval.max_queued_jobs,
                    eval.max_tracked_jobs
                );
            }
            validate_optional_webhook_url("eval.webhook_url", eval.webhook_url.as_deref())?;
        }

        if let Some(agent) = &self.agent {
            if agent.max_concurrent_runs == 0 {
                anyhow::bail!(
                    "agent.max_concurrent_runs must be > 0, got {}",
                    agent.max_concurrent_runs
                );
            }
            if agent.run_timeout_secs < 10 {
                anyhow::bail!(
                    "agent.run_timeout_secs must be at least 10, got {}",
                    agent.run_timeout_secs
                );
            }
            for (field, path) in [
                ("agent.pi_bin", agent.pi_bin.as_ref()),
                ("agent.pi_sessions_dir", agent.pi_sessions_dir.as_ref()),
            ] {
                if path.is_some_and(|path| path.as_os_str().is_empty()) {
                    anyhow::bail!("{field} must be non-empty when set");
                }
            }
            if let Some(pi_bin) = &agent.pi_bin
                && !pi_bin.is_file()
            {
                anyhow::bail!(
                    "agent.pi_bin must name an existing file, got {}",
                    pi_bin.display()
                );
            }
        }

        self.request_log.validate()?;

        Ok(())
    }
}

fn validate_optional_webhook_url(field: &str, value: Option<&str>) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        anyhow::bail!("{field} must be a non-empty HTTP(S) URL when set, got {value:?}");
    }
    let parsed = reqwest::Url::parse(value)
        .with_context(|| format!("{field} must be a valid HTTP(S) URL, got {value:?}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("{field} must use the http or https scheme, got {value:?}");
    }
    Ok(())
}

fn validate_required_http_url(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{field} must be a non-empty HTTP(S) URL, got {value:?}");
    }
    let parsed = reqwest::Url::parse(value)
        .with_context(|| format!("{field} must be a valid HTTP(S) URL, got {value:?}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("{field} must use the http or https scheme, got {value:?}");
    }
    Ok(())
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

fn validate_rocm_graph_cache_entries(entries: usize) -> Result<()> {
    if !(ROCM_GRAPH_CACHE_ENTRIES_MIN..=ROCM_GRAPH_CACHE_ENTRIES_MAX).contains(&entries) {
        anyhow::bail!(
            "accelerator.rocm_graph_cache_entries must be between {} and {} entries, got {entries}",
            ROCM_GRAPH_CACHE_ENTRIES_MIN,
            ROCM_GRAPH_CACHE_ENTRIES_MAX
        );
    }
    Ok(())
}

fn validate_rocm_graph_cache_max_bytes(bytes: u64) -> Result<()> {
    if !(ROCM_GRAPH_CACHE_MAX_BYTES_MIN..=ROCM_GRAPH_CACHE_MAX_BYTES_MAX).contains(&bytes) {
        anyhow::bail!(
            "accelerator.rocm_graph_cache_max_bytes must be between {} and {} bytes, got {bytes}",
            ROCM_GRAPH_CACHE_MAX_BYTES_MIN,
            ROCM_GRAPH_CACHE_MAX_BYTES_MAX
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

fn validate_actor_cycle_idle_ms(millis: u64) -> Result<()> {
    if millis > ACTOR_CYCLE_IDLE_MAX_MS {
        anyhow::bail!(
            "batching.actor_cycle_idle_ms must be between 0 and {} milliseconds, got {millis}",
            ACTOR_CYCLE_IDLE_MAX_MS
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

fn validate_max_prefill_layers_per_cycle(layers: usize) -> Result<()> {
    if !(MAX_PREFILL_LAYERS_PER_CYCLE_MIN..=MAX_PREFILL_LAYERS_PER_CYCLE_MAX).contains(&layers) {
        anyhow::bail!(
            "server.max_prefill_layers_per_cycle must be between {} and {} layers, got {layers}",
            MAX_PREFILL_LAYERS_PER_CYCLE_MIN,
            MAX_PREFILL_LAYERS_PER_CYCLE_MAX
        );
    }
    Ok(())
}

fn validate_max_decode_batch(limit: usize) -> Result<()> {
    if !(MAX_DECODE_BATCH_MIN..=MAX_DECODE_BATCH_MAX).contains(&limit) {
        anyhow::bail!(
            "server.max_decode_batch must be between {} and {} rows, got {limit}",
            MAX_DECODE_BATCH_MIN,
            MAX_DECODE_BATCH_MAX
        );
    }
    Ok(())
}

fn validate_prefill_admission_quantum(quantum: usize) -> Result<()> {
    if !(PREFILL_ADMISSION_QUANTUM_MIN..=PREFILL_ADMISSION_QUANTUM_MAX).contains(&quantum) {
        anyhow::bail!(
            "batching.prefill_admission_quantum must be between {} and {} requests, got {quantum}",
            PREFILL_ADMISSION_QUANTUM_MIN,
            PREFILL_ADMISSION_QUANTUM_MAX
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

fn parse_required_bool_env(name: &str, value: &str) -> Result<bool> {
    parse_bool_env(value).with_context(|| {
        format!("{name} must be one of true, false, 1, 0, yes, no, on, or off, got {value:?}")
    })
}

fn parse_decimal_env<T>(name: &str, value: &str, expected: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display + Send + Sync + 'static,
{
    value
        .trim()
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("{name} must be {expected}, got {value:?}: {error}"))
}

fn parse_optional_capacity_env(name: &str, value: &str) -> Result<Option<u64>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let capacity = parse_decimal_env::<u64>(
        name,
        value,
        "an empty value, 0, or a non-negative decimal integer",
    )?;
    Ok((capacity != 0).then_some(capacity))
}

fn read_optional_unicode_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(raw) => Ok(Some(raw)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("{name} must be valid UTF-8"),
    }
}

fn optional_limit_is_unlimited(value: &str) -> bool {
    value.trim().eq_ignore_ascii_case("unlimited")
}

fn parse_optional_usize_env(name: &str, value: &str) -> Result<Option<usize>> {
    if optional_limit_is_unlimited(value) {
        return Ok(None);
    }
    value.trim().parse::<usize>().map(Some).with_context(|| {
        format!("{name} must be 'unlimited' or a non-negative decimal integer, got {value:?}")
    })
}

fn parse_optional_u64_env(name: &str, value: &str) -> Result<Option<u64>> {
    if optional_limit_is_unlimited(value) {
        return Ok(None);
    }
    value.trim().parse::<u64>().map(Some).with_context(|| {
        format!("{name} must be 'unlimited' or a non-negative decimal integer, got {value:?}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TEST_ENV_LOCK as ENV_LOCK;
    use std::ffi::{OsStr, OsString};

    const EXPECTED_PUBLIC_ENV_NAMES: &[&str] = &[
        "KILN_ACCELERATOR_FULL_ATTENTION_SCORE_BUDGET_MIB",
        "KILN_ACCELERATOR_KT_API_MODE",
        "KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE",
        "KILN_ACCELERATOR_CUDA_KERNEL_PROFILE",
        "KILN_ACCELERATOR_CUDA_MARLIN_PROFILE",
        "KILN_ACCELERATOR_METAL_KERNEL_PROFILE",
        "KILN_ACCELERATOR_ROCM_BF16_MATMUL_OUTPUT_MODE",
        "KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES",
        "KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES",
        "KILN_ACCELERATOR_ROCM_GRAPH_MODE",
        "KILN_ACCELERATOR_ROCM_KERNEL_PROFILE",
        "KILN_ACCELERATOR_ROCM_STRIDED_BATCHED_MATMUL_MODE",
        "KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE",
        "KILN_ACCELERATOR_VULKAN_DEVICE_INDEX",
        "KILN_ACCELERATOR_VULKAN_VALIDATION",
        "KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES",
        "KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES",
        "KILN_ADAPTERS_LIBRARY_URL",
        "KILN_ADAPTERS_MAX_DISK_BYTES",
        "KILN_AGENT_MAX_CONCURRENT_RUNS",
        "KILN_AGENT_PI_BIN",
        "KILN_AGENT_PI_SESSIONS_DIR",
        "KILN_AGENT_RUNS_ACCESS",
        "KILN_AGENT_RUN_TIMEOUT_SECS",
        "KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS",
        "KILN_BATCHING_ACTOR_CYCLE_IDLE_MS",
        "KILN_BATCHING_PREFIX_AWARE_ADMISSION",
        "KILN_BATCHING_PREFILL_ADMISSION_QUANTUM",
        "KILN_BATCHING_ROWWISE_DECODE",
        "KILN_LOGGING_FORMAT",
        "KILN_LOGGING_LEVEL",
        "KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES",
        "KILN_MEMORY_CUDA_GRAPHS",
        "KILN_MEMORY_FLOOR_GB",
        "KILN_MEMORY_GPU_MEMORY_GB",
        "KILN_MEMORY_INFERENCE_MEMORY_FRACTION",
        "KILN_MEMORY_KV_AUTOSCALE",
        "KILN_MEMORY_KV_CACHE_FP8",
        "KILN_MEMORY_KV_FORCE_BLOCKS",
        "KILN_MEMORY_NUM_BLOCKS",
        "KILN_MEMORY_PROBE_MS",
        "KILN_MEMORY_RECLAIM_MODE",
        "KILN_MEMORY_TRAINING_MEMORY_GB",
        "KILN_MEMORY_VULKAN_BUFFER_POOL_GB",
        "KILN_MODEL_ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND",
        "KILN_MODEL_ADAPTER_DIR",
        "KILN_MODEL_CHECKPOINT_READ_MIB_PER_SECOND",
        "KILN_MODEL_MODEL_ID",
        "KILN_MODEL_PATH",
        "KILN_MODEL_SERVED_MODEL_ID",
        "KILN_MODEL_SNAPSHOT_DIR",
        "KILN_MODEL_TOKENIZER_PATH",
        "KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM",
        "KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND",
        "KILN_OPENENV_ALLOW_REMOTE_ENVIRONMENTS",
        "KILN_OPENENV_ENABLED",
        "KILN_OPENENV_MAX_ACTIVE_RUNS",
        "KILN_OPENENV_MAX_TRACKED_RUNS",
        "KILN_OPENENV_TRACKED_RUN_TTL_SECS",
        "KILN_PATHS_CACHE_ROOT",
        "KILN_PREFIX_CACHE_ENABLED",
        "KILN_PREFIX_CACHE_MAX_BLOCKS",
        "KILN_PREFIX_CACHE_MAX_ENTRIES",
        "KILN_REQUEST_LOG_COMPRESS",
        "KILN_REQUEST_LOG_DIR",
        "KILN_REQUEST_LOG_ENABLED",
        "KILN_REQUEST_LOG_MAX_CAPTURE_BYTES",
        "KILN_REQUEST_LOG_MAX_FILE_BYTES",
        "KILN_REQUEST_LOG_MAX_TOTAL_BYTES",
        "KILN_SERVER_CHAT_CONFIG_HASH_METADATA",
        "KILN_SERVER_CHAT_PERFORMANCE_METADATA",
        "KILN_SERVER_DEFAULT_THINKING_BUDGET_MS",
        "KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS",
        "KILN_SERVER_DEFAULT_THINKING_ENABLED",
        "KILN_SERVER_DEBUG_MODEL_STATE",
        "KILN_SERVER_DETERMINISTIC",
        "KILN_SERVER_EVAL_MODE",
        "KILN_SERVER_FOLD_REASONING_INTO_CONTENT",
        "KILN_SERVER_HOST",
        "KILN_SERVER_HTTP_SEND_BUFFER_BYTES",
        "KILN_SERVER_MAX_BATCH_TOKENS",
        "KILN_SERVER_MAX_DECODE_BATCH",
        "KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE",
        "KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE",
        "KILN_SERVER_PORT",
        "KILN_SERVER_REQUEST_TIMEOUT_SECS",
        "KILN_SERVER_SERVING_PROFILE",
        "KILN_SERVER_SHUTDOWN_TIMEOUT_SECS",
        "KILN_SERVER_SLOW_REQUEST_WARN_SECS",
        "KILN_SERVER_STREAM_STALL_GRACE_MS",
        "KILN_SERVER_TERMINAL_ACCESS",
        "KILN_SPECULATIVE_DRAFT_LAYERS",
        "KILN_SPECULATIVE_METHOD",
        "KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS",
        "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
        "KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD",
        "KILN_STREAMING_PREFILL_MODE",
        "KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS",
        "KILN_STREAMING_PREFILL_THRESHOLD_TOKENS",
        "KILN_STREAMING_PREFILL_TILE_TOKENS",
        "KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE",
        "KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB",
        "KILN_TRAINING_CHECKPOINT_INTERVAL",
        "KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS",
        "KILN_TRAINING_MAX_QUEUED_JOBS",
        "KILN_TRAINING_MAX_TRACKED_JOBS",
        "KILN_TRAINING_NO_GRAD_CHECKPOINT",
        "KILN_TRAINING_LOGIT_CACHE_DIR",
        "KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS",
        "KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES",
        "KILN_TRAINING_TRACKED_JOB_TTL_SECS",
        "KILN_TRAINING_WEBHOOK_URL",
    ];

    const RETIRED_PUBLIC_ENVIRONMENT_ALIASES: &[&str] = &[
        "KILN_ADAPTER_DIR",
        "KILN_ADAPTER_LIBRARY_URL",
        "KILN_AGENT_RUNS",
        "KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH",
        "KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS",
        "KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE",
        "KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US",
        "KILN_BATCHING_ENGINE",
        "KILN_BATCHING_MODE",
        "KILN_BATCH_DECODE_ROWWISE",
        "KILN_BATCH_PREFILL_ADMISSION_QUANTUM",
        "KILN_BATCH_PREFIX_AWARE_ADMISSION",
        "KILN_CHAT_CONFIG_HASH_METADATA",
        "KILN_CHAT_PERFORMANCE_METADATA",
        "KILN_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE",
        "KILN_CHECKPOINT_BOUNDARY_CACHE_GB",
        "KILN_CHECKPOINT_INTERVAL",
        "KILN_CUDA_GRAPHS",
        "KILN_DECODE_BATCHER",
        "KILN_DECODE_BATCH_MAX",
        "KILN_DECODE_BATCH_MIXED_SEQ",
        "KILN_DECODE_BATCH_WAIT_US",
        "KILN_DEFAULT_NO_THINK",
        "KILN_DEFAULT_THINKING_BUDGET_MS",
        "KILN_DEFAULT_THINKING_BUDGET_TOKENS",
        "KILN_DEFAULT_THINKING_ENABLED",
        "KILN_DETERMINISTIC",
        "KILN_DETACHED_FULL_ATTN_TILE_TOKENS",
        "KILN_DISABLE_ROCM_BF16_MATMUL_F32_OUTPUT",
        "KILN_DISABLE_ROCM_STRIDED_BATCHED_MATMUL",
        "KILN_EVAL_MODE",
        "KILN_FOLD_REASONING_INTO_CONTENT",
        "KILN_FORCE_ROCM_BF16_MATMUL_F32_OUTPUT",
        "KILN_FORCE_ROCM_STRIDED_BATCHED_MATMUL",
        "KILN_GPU_MEMORY_GB",
        "KILN_GRAD_CHECKPOINT_SEGMENTS",
        "KILN_HOST",
        "KILN_HTTP_SEND_BUFFER_BYTES",
        "KILN_INFERENCE_MEMORY_FRACTION",
        "KILN_KV_AUTOSCALE",
        "KILN_KV_CACHE_FP8",
        "KILN_KV_FORCE_BLOCKS",
        "KILN_LOGIT_CACHE_DIR",
        "KILN_LOG_FORMAT",
        "KILN_LOG_LEVEL",
        "KILN_MAX_BATCH_TOKENS",
        "KILN_MAX_DECODE_BATCH",
        "KILN_MAX_PREFILL_LAYERS_PER_CYCLE",
        "KILN_MAX_PREFILL_TOKENS_PER_CYCLE",
        "KILN_MODEL_ID",
        "KILN_NO_GRAD_CHECKPOINT",
        "KILN_NUM_BLOCKS",
        "KILN_PI_BIN",
        "KILN_PI_SESSIONS_DIR",
        "KILN_PORT",
        "KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS",
        "KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES",
        "KILN_REQUEST_TIMEOUT_SECS",
        "KILN_ROCM_GRAPHS",
        "KILN_ROCM_GRAPH_CACHE_MAX",
        "KILN_ROCM_GRAPH_CAPTURE",
        "KILN_SERVED_MODEL_ID",
        "KILN_SERVING_PROFILE",
        "KILN_SHUTDOWN_TIMEOUT_SECS",
        "KILN_SLOW_REQUEST_WARN_SECS",
        "KILN_SPEC_DRAFT_LAYERS",
        "KILN_SPEC_ENABLED",
        "KILN_SPEC_METHOD",
        "KILN_SPEC_NUM_TOKENS",
        "KILN_SPECULATIVE_ENABLED",
        "KILN_STREAMING_LAST_TOKEN_LM_HEAD",
        "KILN_STREAMING_PREFILL",
        "KILN_STREAMING_PREFILL_ENABLED",
        "KILN_STREAMING_TILE_TOKENS",
        "KILN_STREAM_STALL_GRACE_MS",
        "KILN_TAPE_STREAMING_TILE_TOKENS",
        "KILN_TERMINAL",
        "KILN_TOKENIZER_PATH",
        "KILN_TRAINING_MEMORY_GB",
        "KILN_VULKAN_BUFFER_POOL_GB",
        "KILN_VULKAN_DEVICE",
        "KILN_VULKAN_VALIDATION",
    ];

    const INTENTIONALLY_UNMAPPED_ENV_TARGETS: &[&str] = &[
        "KILN_EVAL_EVAL_DIR",
        "KILN_EVAL_MAX_QUEUED_JOBS",
        "KILN_EVAL_MAX_TRACKED_JOBS",
        "KILN_EVAL_WEBHOOK_URL",
        "KILN_AGENT_SELF_IMPROVE",
        "KILN_TEACHERS_CREDENTIALS",
    ];

    const DYNAMIC_CONFIG_FIELDS: &[&str] = &[
        "teachers.credentials.<id>.api_key_env",
        "teachers.credentials.<id>.origin",
    ];

    fn collect_json_leaf_paths(prefix: &str, value: &serde_json::Value, paths: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(fields) => {
                for (field, value) in fields {
                    let path = if prefix.is_empty() {
                        field.clone()
                    } else {
                        format!("{prefix}.{field}")
                    };
                    collect_json_leaf_paths(&path, value, paths);
                }
            }
            _ => paths.push(prefix.to_owned()),
        }
    }

    struct ScopedConfigEnvironment {
        saved: Vec<(String, Option<OsString>)>,
    }

    impl ScopedConfigEnvironment {
        fn isolated() -> Self {
            Self::isolated_with(&[])
        }

        fn isolated_with(additional_names: &[&str]) -> Self {
            let mut names = vec!["KILN_CONFIG".to_owned()];
            for field in PUBLIC_ENV_FIELDS {
                names.push(field.canonical_name());
            }
            names.extend(
                INTENTIONALLY_UNMAPPED_ENV_TARGETS
                    .iter()
                    .map(|name| (*name).to_owned()),
            );
            names.extend(additional_names.iter().map(|name| (*name).to_owned()));
            names.sort();
            names.dedup();

            let saved = names
                .into_iter()
                .map(|name| {
                    let value = std::env::var_os(&name);
                    unsafe {
                        std::env::remove_var(&name);
                    }
                    (name, value)
                })
                .collect();
            Self { saved }
        }

        fn set(&self, name: &str, value: &str) {
            unsafe {
                std::env::set_var(name, value);
            }
        }

        fn set_os(&self, name: &str, value: &OsStr) {
            unsafe {
                std::env::set_var(name, value);
            }
        }

        fn remove(&self, name: &str) {
            unsafe {
                std::env::remove_var(name);
            }
        }
    }

    impl Drop for ScopedConfigEnvironment {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                unsafe {
                    if let Some(value) = value {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
        }
    }

    // Serializes tests that mutate the process-wide environment. cargo nextest
    // and `cargo test` run tests in parallel by default, so any test that
    // calls `std::env::set_var` / `std::env::remove_var` races with siblings
    // touching the same variables. Acquire this lock for the full duration of
    // the test (bind to a named guard, NOT `_`) before mutating env state.
    // `unwrap_or_else(|e| e.into_inner())` recovers from poisoning so a single
    // panicking test doesn't cascade into the rest of the suite.
    #[test]
    fn runtime_defaults_contract_matches_server_and_client_defaults() {
        let contract: serde_json::Value =
            serde_json::from_str(include_str!("../../../contracts/runtime-defaults-v1.json"))
                .unwrap();

        assert_eq!(contract["contract_version"], 1);
        assert_eq!(contract["server"]["bind_host"], DEFAULT_SERVER_HOST);
        assert_eq!(
            contract["server"]["client_host"],
            DEFAULT_SERVER_CLIENT_HOST
        );
        assert_eq!(contract["server"]["port"], DEFAULT_SERVER_PORT);
        assert_eq!(
            default_server_url(),
            format!("http://{DEFAULT_SERVER_CLIENT_HOST}:{DEFAULT_SERVER_PORT}")
        );
    }

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
        assert!(!config.server.deterministic.enabled());
        assert_eq!(
            config.server.deterministic.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.server.max_decode_batch.limit(), None);
        assert_eq!(
            config.server.max_decode_batch.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.server.host, DEFAULT_SERVER_HOST);
        assert_eq!(config.server.port, DEFAULT_SERVER_PORT);
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
        assert_eq!(
            config.server.max_batch_tokens.tokens(),
            DEFAULT_MAX_BATCH_TOKENS
        );
        assert_eq!(
            config.server.max_prefill_tokens_per_cycle.tokens(),
            DEFAULT_MAX_PREFILL_TOKENS_PER_CYCLE
        );
        assert_eq!(config.server.max_prefill_tokens_per_cycle.tokens(), 256);
        assert!(!config.server.eval_mode);
        assert!(!config.server.debug_model_state);
        assert_eq!(config.server.default_thinking_enabled, None);
        assert_eq!(config.server.default_thinking_budget_tokens, None);
        assert_eq!(config.server.default_thinking_budget_ms, None);
        assert!(!config.server.fold_reasoning_into_content);
        assert!(!config.server.chat_performance_metadata);
        assert!(!config.server.chat_config_hash_metadata);
        assert_eq!(config.server.slow_request_warn_secs, 30);
        assert_eq!(config.server.shutdown_timeout_secs, 5);
        assert_eq!(config.accelerator.kt_api_mode.mode(), KtApiMode::Auto);
        assert_eq!(config.accelerator.vulkan_device_index.index(), None);
        assert!(!config.accelerator.vulkan_validation.enabled());
        assert_eq!(
            config.accelerator.cuda_kernel_profile.profile(),
            CudaKernelProfile::NativeDefault
        );
        assert_eq!(
            config.accelerator.cuda_marlin_profile.profile(),
            CudaMarlinProfile::Disabled
        );
        assert_eq!(
            config.accelerator.cuda_flash_backward_mode.mode(),
            CudaFlashBackwardMode::Fast
        );
        assert_eq!(
            config.accelerator.metal_kernel_profile.profile(),
            MetalKernelProfile::NativeDefault
        );
        assert_eq!(
            config.accelerator.full_attention_score_budget_mib.mib(),
            kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB
        );
        assert_eq!(
            config.accelerator.full_attention_score_budget_mib.source(),
            ConfigValueSource::Default
        );
        assert_eq!(
            config.accelerator.rocm_synchronization_mode.mode(),
            RocmSynchronizationMode::LegacyHostBarriers
        );
        assert_eq!(
            config.accelerator.rocm_strided_batched_matmul_mode.mode(),
            RocmStridedBatchedMatmulMode::Disabled
        );
        assert_eq!(
            config.accelerator.rocm_bf16_matmul_output_mode.mode(),
            RocmBf16MatmulOutputMode::F32ThenCast
        );
        assert_eq!(
            config.accelerator.rocm_kernel_profile.profile(),
            RocmKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.rocm_graph_mode.mode(),
            RocmGraphMode::Profile
        );
        assert_eq!(
            config.accelerator.rocm_graph_cache_entries.entries(),
            DEFAULT_ROCM_GRAPH_CACHE_ENTRIES
        );
        assert_eq!(
            config.accelerator.rocm_graph_cache_max_bytes.bytes(),
            DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES
        );
        for source in [
            config.accelerator.kt_api_mode.source(),
            config.accelerator.full_attention_score_budget_mib.source(),
            config.accelerator.vulkan_device_index.source(),
            config.accelerator.vulkan_validation.source(),
            config.accelerator.cuda_kernel_profile.source(),
            config.accelerator.cuda_marlin_profile.source(),
            config.accelerator.cuda_flash_backward_mode.source(),
            config.accelerator.metal_kernel_profile.source(),
            config.accelerator.rocm_synchronization_mode.source(),
            config.accelerator.rocm_strided_batched_matmul_mode.source(),
            config.accelerator.rocm_bf16_matmul_output_mode.source(),
            config.accelerator.rocm_kernel_profile.source(),
            config.accelerator.rocm_graph_mode.source(),
            config.accelerator.rocm_graph_cache_entries.source(),
            config.accelerator.rocm_graph_cache_max_bytes.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Default);
        }
        assert!(!config.batching.rowwise_decode.enabled());
        assert_eq!(
            config.batching.rowwise_decode.source(),
            ConfigValueSource::Default
        );
        assert!(config.batching.prefix_aware_admission.enabled());
        assert_eq!(
            config.batching.prefix_aware_admission.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.batching.prefill_admission_quantum.configured(), None);
        assert_eq!(
            config.batching.prefill_admission_quantum.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.batching.actor_cycle_idle_ms.millis(), 0);
        assert_eq!(
            config.batching.actor_cycle_idle_ms.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
        assert!(config.model.path.is_none());
        assert!(config.model.tokenizer_path.is_none());
        assert!(config.model.adapter_dir.is_none());
        assert!(config.memory.num_blocks.is_none());
        assert_eq!(config.memory.inference_memory_fraction, 0.7);
        assert_eq!(config.memory.vulkan_buffer_pool_gb, 3.0);
        assert_eq!(
            config.memory.vulkan_buffer_pool_bytes(),
            3 * 1024 * 1024 * 1024
        );
        assert_eq!(config.memory.floor_gb, 1.0);
        assert_eq!(config.memory.probe_ms, 500);
        assert_eq!(
            config.memory.reclaim_mode.mode(),
            kiln_memory::MemoryReclaimMode::Off
        );
        assert_eq!(
            config.memory.reclaim_mode.source(),
            ConfigValueSource::Default
        );
        assert!(config.memory.kv_autoscale.enabled());
        assert_eq!(
            config.memory.kv_autoscale.source(),
            ConfigValueSource::Default
        );
        assert_eq!(config.memory.kv_force_blocks.target(), None);
        assert_eq!(
            config.memory.kv_force_blocks.source(),
            ConfigValueSource::Default
        );
        assert!(!config.memory.kv_cache_fp8);
        assert!(config.memory.cuda_graphs); // #34: default-ON
        assert_eq!(config.memory.cuda_graph_cache_entries, 8);
        assert!(!config.training.no_grad_checkpoint);
        assert_eq!(
            config.training.recompute_checkpoint_boundaries.mode(),
            kiln_train::CheckpointBoundaryRecomputeMode::Auto
        );
        assert_eq!(
            config.training.recompute_boundary_threshold_tokens.tokens(),
            kiln_train::DEFAULT_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS
        );
        assert_eq!(
            config
                .training
                .checkpoint_boundary_anchor_stride
                .configured(),
            None
        );
        assert_eq!(
            config.training.checkpoint_boundary_cache_gb.gib(),
            DEFAULT_CHECKPOINT_BOUNDARY_CACHE_GB
        );
        assert_eq!(
            config.training.checkpoint_boundary_cache_gb.bytes(),
            kiln_train::DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES
        );
        for source in [
            config.training.recompute_checkpoint_boundaries.source(),
            config.training.recompute_boundary_threshold_tokens.source(),
            config.training.checkpoint_boundary_anchor_stride.source(),
            config.training.checkpoint_boundary_cache_gb.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Default);
        }
        assert_eq!(
            config.training.checkpoint_boundary_policy().unwrap(),
            kiln_train::CheckpointBoundaryPolicy::default()
        );
        assert!(config.training.checkpoint_interval.is_none());
        assert!(config.training.webhook_url.is_none());
        assert_eq!(config.training.max_queued_jobs, 32);
        assert_eq!(config.training.max_tracked_jobs, 1024);
        assert_eq!(config.training.tracked_job_ttl_secs, 604_800);
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.format, "auto");
        assert!(config.prefix_cache.enabled);
        assert!(config.prefix_cache.max_blocks.is_none());
        assert_eq!(config.speculative.method, SpecMethod::Off);
        assert_eq!(
            config.speculative.num_speculative_tokens,
            MAX_SPECULATIVE_DRAFT_TOKENS
        );
        assert_eq!(config.speculative.draft_layers, 8);
        assert_eq!(
            config.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Auto
        );
        assert_eq!(config.streaming_prefill.threshold_tokens.configured(), None);
        assert_eq!(config.streaming_prefill.tile_tokens.configured(), None);
        assert_eq!(config.streaming_prefill.tape_tile_tokens.configured(), None);
        assert_eq!(
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
            None
        );
        assert!(config.streaming_prefill.last_token_lm_head.enabled());
        for source in [
            config.streaming_prefill.mode.source(),
            config.streaming_prefill.threshold_tokens.source(),
            config.streaming_prefill.tile_tokens.source(),
            config.streaming_prefill.tape_tile_tokens.source(),
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
            config.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Default);
        }
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
    fn accelerator_policy_defaults_resolve_by_serving_profile_and_serialize() {
        let accelerator = AcceleratorRuntimeConfig::default();
        for (profile, expected_graph_mode) in [
            (ServingProfile::Stable, RocmGraphMode::Disabled),
            (
                ServingProfile::Experimental,
                RocmGraphMode::LazyCaptureReplay,
            ),
            (ServingProfile::Maintenance, RocmGraphMode::Disabled),
        ] {
            let source = if profile == ServingProfile::Experimental {
                ConfigValueSource::ConfigFile
            } else {
                ConfigValueSource::Default
            };
            let resolved = accelerator.resolved_policy(ServingProfileSetting::new(profile, source));
            assert_eq!(resolved.schema_id, ACCELERATOR_RUNTIME_POLICY_SCHEMA_ID);
            assert_eq!(resolved.version, ACCELERATOR_RUNTIME_POLICY_VERSION);
            assert_eq!(
                resolved.vulkan_kernel_policy_schema_id,
                kiln_model::VULKAN_KERNEL_POLICY_SCHEMA_ID
            );
            assert_eq!(
                resolved.vulkan_device_policy_schema_id,
                kiln_model::VULKAN_DEVICE_POLICY_SCHEMA_ID
            );
            assert_eq!(resolved.serving_profile, profile);
            assert_eq!(resolved.serving_profile_source, source);
            assert_eq!(
                resolved.kt_api_mode,
                ResolvedAcceleratorValue {
                    configured: KtApiMode::Auto,
                    effective: KtApiMode::Auto,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.full_attention_score_budget_mib,
                ResolvedAcceleratorValue {
                    configured: kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB,
                    effective: kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.vulkan_device_index,
                ResolvedAcceleratorValue {
                    configured: None,
                    effective: None,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.vulkan_validation,
                ResolvedAcceleratorValue {
                    configured: false,
                    effective: false,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.cuda_kernel_profile,
                ResolvedAcceleratorValue {
                    configured: CudaKernelProfile::NativeDefault,
                    effective: CudaKernelProfile::NativeDefault,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.cuda_marlin_profile,
                ResolvedAcceleratorValue {
                    configured: CudaMarlinProfile::Disabled,
                    effective: CudaMarlinProfile::Disabled,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.cuda_flash_backward_mode,
                ResolvedAcceleratorValue {
                    configured: CudaFlashBackwardMode::Fast,
                    effective: CudaFlashBackwardMode::Fast,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.metal_kernel_profile,
                ResolvedAcceleratorValue {
                    configured: MetalKernelProfile::NativeDefault,
                    effective: MetalKernelProfile::NativeDefault,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_synchronization_mode,
                ResolvedAcceleratorValue {
                    configured: RocmSynchronizationMode::LegacyHostBarriers,
                    effective: RocmSynchronizationMode::LegacyHostBarriers,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_strided_batched_matmul_mode,
                ResolvedAcceleratorValue {
                    configured: RocmStridedBatchedMatmulMode::Disabled,
                    effective: RocmStridedBatchedMatmulMode::Disabled,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_bf16_matmul_output_mode,
                ResolvedAcceleratorValue {
                    configured: RocmBf16MatmulOutputMode::F32ThenCast,
                    effective: RocmBf16MatmulOutputMode::F32ThenCast,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_kernel_profile,
                ResolvedAcceleratorValue {
                    configured: RocmKernelProfile::PortableFallback,
                    effective: RocmKernelProfile::PortableFallback,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_graph_mode,
                ResolvedAcceleratorValue {
                    configured: RocmGraphMode::Profile,
                    effective: expected_graph_mode,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_graph_cache_entries,
                ResolvedAcceleratorValue {
                    configured: DEFAULT_ROCM_GRAPH_CACHE_ENTRIES,
                    effective: DEFAULT_ROCM_GRAPH_CACHE_ENTRIES,
                    source: ConfigValueSource::Default,
                }
            );
            assert_eq!(
                resolved.rocm_graph_cache_max_bytes,
                ResolvedAcceleratorValue {
                    configured: DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES,
                    effective: DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES,
                    source: ConfigValueSource::Default,
                }
            );
        }

        let json = serde_json::to_value(accelerator.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Experimental,
            ConfigValueSource::Environment,
        )))
        .unwrap();
        assert_eq!(json["schema_id"], "kiln.accelerator-runtime-policy.v16");
        assert_eq!(json["version"], 16);
        assert_eq!(
            json["vulkan_kernel_policy_schema_id"],
            "kiln.vulkan-kernel-policy.v6"
        );
        assert_eq!(
            json["vulkan_device_policy_schema_id"],
            "kiln.vulkan-device-policy.v1"
        );
        assert_eq!(json["serving_profile"], "experimental");
        assert_eq!(json["serving_profile_source"], "environment");
        assert_eq!(json["kt_api_mode"]["configured"], "auto");
        assert_eq!(json["kt_api_mode"]["effective"], "auto");
        assert_eq!(json["kt_api_mode"]["source"], "default");
        assert_eq!(json["full_attention_score_budget_mib"]["effective"], 2048);
        assert!(json["vulkan_device_index"]["effective"].is_null());
        assert_eq!(json["vulkan_validation"]["effective"], false);
        assert_eq!(json["cuda_kernel_profile"]["configured"], "native_default");
        assert_eq!(json["cuda_kernel_profile"]["effective"], "native_default");
        assert_eq!(json["cuda_kernel_profile"]["source"], "default");
        assert_eq!(json["cuda_marlin_profile"]["configured"], "disabled");
        assert_eq!(json["cuda_marlin_profile"]["effective"], "disabled");
        assert_eq!(json["cuda_marlin_profile"]["source"], "default");
        assert_eq!(json["cuda_flash_backward_mode"]["configured"], "fast");
        assert_eq!(json["cuda_flash_backward_mode"]["effective"], "fast");
        assert_eq!(json["cuda_flash_backward_mode"]["source"], "default");
        assert_eq!(json["metal_kernel_profile"]["configured"], "native_default");
        assert_eq!(json["metal_kernel_profile"]["effective"], "native_default");
        assert_eq!(json["metal_kernel_profile"]["source"], "default");
        assert_eq!(
            json["rocm_kernel_profile"]["configured"],
            "portable_fallback"
        );
        assert_eq!(
            json["rocm_kernel_profile"]["effective"],
            "portable_fallback"
        );
        assert_eq!(json["rocm_kernel_profile"]["source"], "default");
        assert_eq!(json["rocm_graph_mode"]["configured"], "profile");
        assert_eq!(json["rocm_graph_mode"]["effective"], "lazy_capture_replay");
        assert_eq!(json["rocm_graph_mode"]["source"], "default");
    }

    #[test]
    fn accelerator_toml_is_strict_source_tracked_bounded_and_profile_gated() {
        let config: KilnConfig = toml::from_str(
            r#"
[server]
serving_profile = "experimental"

[accelerator]
kt_api_mode = "all"
full_attention_score_budget_mib = 64
vulkan_device_index = 2
vulkan_validation = true
cuda_kernel_profile = "portable_fallback"
cuda_marlin_profile = "attention_mlp_gdn"
cuda_flash_backward_mode = "deterministic"
metal_kernel_profile = "portable_fallback"
rocm_synchronization_mode = "stream_ordered"
rocm_strided_batched_matmul_mode = "disabled"
rocm_bf16_matmul_output_mode = "f32_then_cast"
rocm_kernel_profile = "portable_fallback"
rocm_graph_mode = "warmup_then_eager"
rocm_graph_cache_entries = 64
rocm_graph_cache_max_bytes = 17179869184
"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.accelerator.kt_api_mode.mode(), KtApiMode::All);
        assert_eq!(config.accelerator.full_attention_score_budget_mib.mib(), 64);
        assert_eq!(config.accelerator.vulkan_device_index.index(), Some(2));
        assert!(config.accelerator.vulkan_validation.enabled());
        assert_eq!(
            config.accelerator.cuda_kernel_profile.profile(),
            CudaKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.cuda_marlin_profile.profile(),
            CudaMarlinProfile::AttentionMlpGdn
        );
        assert_eq!(
            config.accelerator.cuda_flash_backward_mode.mode(),
            CudaFlashBackwardMode::Deterministic
        );
        assert_eq!(
            config.accelerator.metal_kernel_profile.profile(),
            MetalKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.full_attention_score_budget_mib.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(
            config.accelerator.rocm_synchronization_mode.mode(),
            RocmSynchronizationMode::StreamOrdered
        );
        assert_eq!(
            config.accelerator.rocm_strided_batched_matmul_mode.mode(),
            RocmStridedBatchedMatmulMode::Disabled
        );
        assert_eq!(
            config.accelerator.rocm_bf16_matmul_output_mode.mode(),
            RocmBf16MatmulOutputMode::F32ThenCast
        );
        assert_eq!(
            config.accelerator.rocm_kernel_profile.profile(),
            RocmKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.rocm_graph_mode.mode(),
            RocmGraphMode::WarmupThenEager
        );
        assert_eq!(config.accelerator.rocm_graph_cache_entries.entries(), 64);
        assert_eq!(
            config.accelerator.rocm_graph_cache_max_bytes.bytes(),
            ROCM_GRAPH_CACHE_MAX_BYTES_MAX
        );
        for source in [
            config.accelerator.kt_api_mode.source(),
            config.accelerator.vulkan_device_index.source(),
            config.accelerator.vulkan_validation.source(),
            config.accelerator.cuda_kernel_profile.source(),
            config.accelerator.cuda_marlin_profile.source(),
            config.accelerator.cuda_flash_backward_mode.source(),
            config.accelerator.metal_kernel_profile.source(),
            config.accelerator.rocm_synchronization_mode.source(),
            config.accelerator.rocm_strided_batched_matmul_mode.source(),
            config.accelerator.rocm_bf16_matmul_output_mode.source(),
            config.accelerator.rocm_kernel_profile.source(),
            config.accelerator.rocm_graph_mode.source(),
            config.accelerator.rocm_graph_cache_entries.source(),
            config.accelerator.rocm_graph_cache_max_bytes.source(),
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }

        for mib in [
            kiln_model::MIN_FULL_ATTENTION_SCORE_BUDGET_MIB,
            kiln_model::MAX_FULL_ATTENTION_SCORE_BUDGET_MIB,
        ] {
            let parsed: KilnConfig = toml::from_str(&format!(
                "[accelerator]\nfull_attention_score_budget_mib = {mib}\n"
            ))
            .unwrap();
            assert_eq!(
                parsed.accelerator.full_attention_score_budget_mib.mib(),
                mib
            );
        }

        for entries in [ROCM_GRAPH_CACHE_ENTRIES_MIN, ROCM_GRAPH_CACHE_ENTRIES_MAX] {
            let parsed: KilnConfig = toml::from_str(&format!(
                "[accelerator]\nrocm_graph_cache_entries = {entries}\n"
            ))
            .unwrap();
            assert_eq!(
                parsed.accelerator.rocm_graph_cache_entries.entries(),
                entries
            );
        }

        for bytes in [
            ROCM_GRAPH_CACHE_MAX_BYTES_MIN,
            ROCM_GRAPH_CACHE_MAX_BYTES_MAX,
        ] {
            let parsed: KilnConfig = toml::from_str(&format!(
                "[accelerator]\nrocm_graph_cache_max_bytes = {bytes}\n"
            ))
            .unwrap();
            assert_eq!(parsed.accelerator.rocm_graph_cache_max_bytes.bytes(), bytes);
        }

        for document in [
            "[accelerator]\nkt_api_mode = \"sometimes\"\n".to_owned(),
            "[accelerator]\nkt_api_mode = true\n".to_owned(),
            "[accelerator]\nfull_attention_score_budget_mib = 63\n".to_owned(),
            "[accelerator]\nfull_attention_score_budget_mib = 2049\n".to_owned(),
            "[accelerator]\nfull_attention_score_budget_mib = \"2048\"\n".to_owned(),
            "[accelerator]\nvulkan_device_index = \"gpu\"\n".to_owned(),
            "[accelerator]\nvulkan_validation = \"true\"\n".to_owned(),
            "[accelerator]\ncuda_kernel_profile = \"individual_switches\"\n".to_owned(),
            "[accelerator]\ncuda_kernel_profile = true\n".to_owned(),
            "[accelerator]\ncuda_marlin_profile = \"everything\"\n".to_owned(),
            "[accelerator]\ncuda_marlin_profile = true\n".to_owned(),
            "[accelerator]\ncuda_flash_backward_mode = \"auto\"\n".to_owned(),
            "[accelerator]\ncuda_flash_backward_mode = true\n".to_owned(),
            "[accelerator]\nmetal_kernel_profile = \"individual_switches\"\n".to_owned(),
            "[accelerator]\nmetal_kernel_profile = true\n".to_owned(),
            "[accelerator]\nrocm_synchronization_mode = \"eventually\"\n".to_owned(),
            "[accelerator]\nrocm_synchronization_mode = true\n".to_owned(),
            "[accelerator]\nrocm_strided_batched_matmul_mode = \"sometimes\"\n".to_owned(),
            "[accelerator]\nrocm_strided_batched_matmul_mode = true\n".to_owned(),
            "[accelerator]\nrocm_bf16_matmul_output_mode = \"bf16\"\n".to_owned(),
            "[accelerator]\nrocm_bf16_matmul_output_mode = false\n".to_owned(),
            "[accelerator]\nrocm_kernel_profile = \"individual_switches\"\n".to_owned(),
            "[accelerator]\nrocm_kernel_profile = true\n".to_owned(),
            "[accelerator]\nrocm_graph_mode = \"automatic\"\n".to_owned(),
            "[accelerator]\nrocm_graph_mode = false\n".to_owned(),
            "[accelerator]\nrocm_graph_cache_entries = 0\n".to_owned(),
            format!(
                "[accelerator]\nrocm_graph_cache_entries = {}\n",
                ROCM_GRAPH_CACHE_ENTRIES_MAX + 1
            ),
            "[accelerator]\nrocm_graph_cache_entries = \"8\"\n".to_owned(),
            format!(
                "[accelerator]\nrocm_graph_cache_max_bytes = {}\n",
                ROCM_GRAPH_CACHE_MAX_BYTES_MIN - 1
            ),
            format!(
                "[accelerator]\nrocm_graph_cache_max_bytes = {}\n",
                ROCM_GRAPH_CACHE_MAX_BYTES_MAX + 1
            ),
            "[accelerator]\nrocm_graph_cache_max_bytes = \"1073741824\"\n".to_owned(),
            "[accelerator]\nunknown = true\n".to_owned(),
        ] {
            let error = toml::from_str::<KilnConfig>(&document).unwrap_err();
            let detail = error.to_string();
            assert!(
                detail.contains("accelerator")
                    || detail.contains("full-attention score budget")
                    || detail.contains("invalid type")
                    || detail.contains("unknown field"),
                "unexpected error for {document:?}: {error:#}"
            );
        }

        for profile in [ServingProfile::Stable, ServingProfile::Maintenance] {
            for mode in [KtApiMode::All, KtApiMode::Disabled] {
                let mut gated = KilnConfig::default();
                gated.server.serving_profile =
                    ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
                gated.accelerator.kt_api_mode =
                    KtApiModeSetting::new(mode, ConfigValueSource::ConfigFile);
                let detail = gated.validate().unwrap_err().to_string();
                assert!(detail.contains("accelerator.kt_api_mode"), "{detail}");
                assert!(detail.contains("experimental"), "{detail}");
            }

            for mode in [
                RocmGraphMode::WarmupThenEager,
                RocmGraphMode::LazyCaptureReplay,
            ] {
                let mut gated = KilnConfig::default();
                gated.server.serving_profile =
                    ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
                gated.accelerator.rocm_graph_mode =
                    RocmGraphModeSetting::new(mode, ConfigValueSource::ConfigFile);
                let detail = gated.validate().unwrap_err().to_string();
                assert!(detail.contains("accelerator.rocm_graph_mode"), "{detail}");
                assert!(detail.contains("experimental"), "{detail}");
            }

            let mut gated = KilnConfig::default();
            gated.server.serving_profile =
                ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
            gated.accelerator.cuda_marlin_profile = CudaMarlinProfileSetting::new(
                CudaMarlinProfile::AttentionMlp,
                ConfigValueSource::ConfigFile,
            );
            let detail = gated.validate().unwrap_err().to_string();
            assert!(
                detail.contains("accelerator.cuda_marlin_profile"),
                "{detail}"
            );
            assert!(detail.contains("experimental"), "{detail}");

            let mut gated = KilnConfig::default();
            gated.server.serving_profile =
                ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
            gated.accelerator.rocm_synchronization_mode = RocmSynchronizationModeSetting::new(
                RocmSynchronizationMode::StreamOrdered,
                ConfigValueSource::ConfigFile,
            );
            let detail = gated.validate().unwrap_err().to_string();
            assert!(
                detail.contains("accelerator.rocm_synchronization_mode"),
                "{detail}"
            );
            assert!(detail.contains("experimental"), "{detail}");

            let mut gated = KilnConfig::default();
            gated.server.serving_profile =
                ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
            gated.accelerator.vulkan_validation =
                VulkanValidationSetting::new(true, ConfigValueSource::ConfigFile);
            let detail = gated.validate().unwrap_err().to_string();
            assert!(detail.contains("accelerator.vulkan_validation"), "{detail}");
            assert!(detail.contains("experimental"), "{detail}");

            for (strided_mode, output_mode, expected_field) in [
                (
                    RocmStridedBatchedMatmulMode::Enabled,
                    RocmBf16MatmulOutputMode::F32ThenCast,
                    "accelerator.rocm_strided_batched_matmul_mode",
                ),
                (
                    RocmStridedBatchedMatmulMode::Disabled,
                    RocmBf16MatmulOutputMode::NativeBf16,
                    "accelerator.rocm_bf16_matmul_output_mode",
                ),
            ] {
                let mut gated = KilnConfig::default();
                gated.server.serving_profile =
                    ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
                gated.accelerator.rocm_strided_batched_matmul_mode =
                    RocmStridedBatchedMatmulModeSetting::new(
                        strided_mode,
                        ConfigValueSource::ConfigFile,
                    );
                gated.accelerator.rocm_bf16_matmul_output_mode =
                    RocmBf16MatmulOutputModeSetting::new(
                        output_mode,
                        ConfigValueSource::ConfigFile,
                    );
                let detail = gated.validate().unwrap_err().to_string();
                assert!(detail.contains(expected_field), "{detail}");
                assert!(detail.contains("experimental"), "{detail}");
            }

            for allowed_mode in [RocmGraphMode::Profile, RocmGraphMode::Disabled] {
                let mut allowed = KilnConfig::default();
                allowed.server.serving_profile =
                    ServingProfileSetting::new(profile, ConfigValueSource::ConfigFile);
                allowed.accelerator.rocm_graph_mode =
                    RocmGraphModeSetting::new(allowed_mode, ConfigValueSource::ConfigFile);
                allowed.validate().unwrap();

                allowed.accelerator.rocm_kernel_profile = RocmKernelProfileSetting::new(
                    RocmKernelProfile::PortableFallback,
                    ConfigValueSource::ConfigFile,
                );
                allowed.accelerator.cuda_flash_backward_mode = CudaFlashBackwardModeSetting::new(
                    CudaFlashBackwardMode::Deterministic,
                    ConfigValueSource::ConfigFile,
                );
                allowed.validate().unwrap();
            }
        }
    }

    #[test]
    fn cuda_kernel_profile_environment_is_canonical_strict_and_source_tracked() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        for (raw, expected) in [
            ("native_default", CudaKernelProfile::NativeDefault),
            ("portable_fallback", CudaKernelProfile::PortableFallback),
        ] {
            environment.set("KILN_ACCELERATOR_CUDA_KERNEL_PROFILE", raw);
            let mut config = KilnConfig::default();
            config.apply_env_overrides().unwrap();
            assert_eq!(config.accelerator.cuda_kernel_profile.profile(), expected);
            assert_eq!(
                config.accelerator.cuda_kernel_profile.source(),
                ConfigValueSource::Environment
            );
            environment.remove("KILN_ACCELERATOR_CUDA_KERNEL_PROFILE");
        }

        environment.set("KILN_ACCELERATOR_CUDA_KERNEL_PROFILE", "custom");
        let detail = KilnConfig::default()
            .apply_env_overrides()
            .unwrap_err()
            .to_string();
        assert!(
            detail.contains("KILN_ACCELERATOR_CUDA_KERNEL_PROFILE"),
            "{detail}"
        );
        assert!(detail.contains("native_default"), "{detail}");
    }

    #[test]
    fn cuda_marlin_profile_environment_is_canonical_strict_and_source_tracked() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        for (raw, expected) in [
            ("disabled", CudaMarlinProfile::Disabled),
            ("attention_mlp", CudaMarlinProfile::AttentionMlp),
            ("attention_mlp_gdn", CudaMarlinProfile::AttentionMlpGdn),
        ] {
            environment.set("KILN_ACCELERATOR_CUDA_MARLIN_PROFILE", raw);
            let mut config = KilnConfig::default();
            config.apply_env_overrides().unwrap();
            assert_eq!(config.accelerator.cuda_marlin_profile.profile(), expected);
            assert_eq!(
                config.accelerator.cuda_marlin_profile.source(),
                ConfigValueSource::Environment
            );
            environment.remove("KILN_ACCELERATOR_CUDA_MARLIN_PROFILE");
        }

        environment.set("KILN_ACCELERATOR_CUDA_MARLIN_PROFILE", "custom");
        let detail = KilnConfig::default()
            .apply_env_overrides()
            .unwrap_err()
            .to_string();
        assert!(
            detail.contains("KILN_ACCELERATOR_CUDA_MARLIN_PROFILE"),
            "{detail}"
        );
        assert!(detail.contains("attention_mlp_gdn"), "{detail}");
    }

    #[test]
    fn cuda_flash_backward_mode_environment_is_canonical_strict_and_source_tracked() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        for (raw, expected) in [
            ("fast", CudaFlashBackwardMode::Fast),
            ("deterministic", CudaFlashBackwardMode::Deterministic),
        ] {
            environment.set("KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE", raw);
            let mut config = KilnConfig::default();
            config.apply_env_overrides().unwrap();
            assert_eq!(config.accelerator.cuda_flash_backward_mode.mode(), expected);
            assert_eq!(
                config.accelerator.cuda_flash_backward_mode.source(),
                ConfigValueSource::Environment
            );
            environment.remove("KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE");
        }

        environment.set("KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE", "custom");
        let detail = KilnConfig::default()
            .apply_env_overrides()
            .unwrap_err()
            .to_string();
        assert!(
            detail.contains("KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE"),
            "{detail}"
        );
        assert!(detail.contains("deterministic"), "{detail}");
    }

    #[test]
    fn metal_kernel_profile_environment_is_canonical_strict_and_source_tracked() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        for (raw, expected) in [
            ("native_default", MetalKernelProfile::NativeDefault),
            ("portable_fallback", MetalKernelProfile::PortableFallback),
        ] {
            environment.set("KILN_ACCELERATOR_METAL_KERNEL_PROFILE", raw);
            let mut config = KilnConfig::default();
            config.apply_env_overrides().unwrap();
            assert_eq!(config.accelerator.metal_kernel_profile.profile(), expected);
            assert_eq!(
                config.accelerator.metal_kernel_profile.source(),
                ConfigValueSource::Environment
            );
            environment.remove("KILN_ACCELERATOR_METAL_KERNEL_PROFILE");
        }

        environment.set("KILN_ACCELERATOR_METAL_KERNEL_PROFILE", "custom");
        let detail = KilnConfig::default()
            .apply_env_overrides()
            .unwrap_err()
            .to_string();
        assert!(
            detail.contains("KILN_ACCELERATOR_METAL_KERNEL_PROFILE"),
            "{detail}"
        );
        assert!(detail.contains("native_default"), "{detail}");
    }

    #[test]
    fn rocm_kernel_profile_environment_is_canonical_strict_and_source_tracked() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        environment.set("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE", "portable_fallback");
        let mut config = KilnConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(
            config.accelerator.rocm_kernel_profile.profile(),
            RocmKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.rocm_kernel_profile.source(),
            ConfigValueSource::Environment
        );
        environment.remove("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE");

        for retired in ["qualified", "experimental_multiblock", "custom"] {
            environment.set("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE", retired);
            let detail = KilnConfig::default()
                .apply_env_overrides()
                .unwrap_err()
                .to_string();
            assert!(
                detail.contains("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE"),
                "{detail}"
            );
            assert!(detail.contains("portable_fallback"), "{detail}");
        }
    }

    #[test]
    fn vulkan_device_environment_is_typed_and_strict() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();

        environment.set("KILN_ACCELERATOR_VULKAN_DEVICE_INDEX", "3");
        environment.set("KILN_ACCELERATOR_VULKAN_VALIDATION", "true");
        let mut config = KilnConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.accelerator.vulkan_device_index.index(), Some(3));
        assert!(config.accelerator.vulkan_validation.enabled());
        assert_eq!(
            config.accelerator.vulkan_device_index.source(),
            ConfigValueSource::Environment
        );
        assert_eq!(
            config.accelerator.vulkan_validation.source(),
            ConfigValueSource::Environment
        );

        environment.set("KILN_ACCELERATOR_VULKAN_DEVICE_INDEX", "gpu");
        let detail = KilnConfig::default()
            .apply_env_overrides()
            .unwrap_err()
            .to_string();
        assert!(detail.contains("zero-based"), "{detail}");
    }

    #[test]
    fn speculative_draft_geometry_is_validated_against_the_selected_model() {
        let model = kiln_core::config::ModelConfig::qwen3_5_4b();
        let disabled = SpeculativeDecodingConfig {
            draft_layers: model.num_layers,
            ..SpeculativeDecodingConfig::default()
        };
        assert!(disabled.validate_for_model(&model).is_ok());

        for method in [SpecMethod::SkipLayer, SpecMethod::Mtp] {
            let enabled = SpeculativeDecodingConfig {
                method,
                draft_layers: model.num_layers,
                ..SpeculativeDecodingConfig::default()
            };
            let error = enabled.validate_for_model(&model).unwrap_err().to_string();
            assert!(error.contains("speculative.draft_layers"));
            assert!(error.contains(&model.num_layers.to_string()));
        }

        let valid = SpeculativeDecodingConfig {
            method: SpecMethod::SkipLayer,
            draft_layers: model.num_layers - 1,
            ..SpeculativeDecodingConfig::default()
        };
        assert!(valid.validate_for_model(&model).is_ok());
    }

    #[test]
    fn speculative_serving_fails_closed_until_accelerator_qualification() {
        for method in [SpecMethod::SkipLayer, SpecMethod::Mtp] {
            let speculative = SpeculativeDecodingConfig {
                method,
                ..SpeculativeDecodingConfig::default()
            };
            let error = speculative.validate_for_serving().unwrap_err().to_string();
            assert!(error.contains(&format!("{method:?}")));
            assert!(error.contains("owner-settlement"));
            assert!(error.contains("local accelerator qualification"));
        }

        SpeculativeDecodingConfig::default()
            .validate_for_serving()
            .unwrap();
    }

    #[test]
    fn public_env_registry_is_mechanical_complete_and_unique() {
        let mut names = PUBLIC_ENV_FIELDS
            .iter()
            .map(PublicEnvField::canonical_name)
            .collect::<Vec<_>>();
        let original_len = names.len();
        names.sort();
        names.dedup();

        let mut expected = EXPECTED_PUBLIC_ENV_NAMES
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(original_len, 112);
        assert_eq!(names.len(), original_len, "canonical names must be unique");
        assert_eq!(names, expected);

        for field in PUBLIC_ENV_FIELDS {
            assert_eq!(
                field.canonical_name(),
                format!(
                    "KILN_{}_{}",
                    field.section.to_ascii_uppercase(),
                    field.field.to_ascii_uppercase()
                )
            );
        }
    }

    #[test]
    fn retired_public_environment_aliases_are_unique_disjoint_and_inert() {
        let mut retired = RETIRED_PUBLIC_ENVIRONMENT_ALIASES.to_vec();
        retired.sort_unstable();
        retired.dedup();
        assert_eq!(retired.len(), 82);
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../contracts/kiln-config-v1.schema.json"
        ))
        .unwrap();
        let mut schema_retired = schema["x-kiln-retired-environment-replacements"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        schema_retired.sort_unstable();
        assert_eq!(retired, schema_retired);
        assert!(retired.iter().all(|name| {
            !EXPECTED_PUBLIC_ENV_NAMES.contains(name)
                && !INTENTIONALLY_UNMAPPED_ENV_TARGETS.contains(name)
        }));

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment =
            ScopedConfigEnvironment::isolated_with(RETIRED_PUBLIC_ENVIRONMENT_ALIASES);
        let expected = KilnConfig::default();
        let expected_json = serde_json::to_value(&expected).unwrap();
        let expected_hash = expected
            .effective_configuration()
            .unwrap()
            .effective_config_hash;
        for name in RETIRED_PUBLIC_ENVIRONMENT_ALIASES {
            environment.set(name, "definitely-retired");
        }

        let mut actual = KilnConfig::default();
        actual.apply_env_overrides().unwrap();
        assert_eq!(serde_json::to_value(&actual).unwrap(), expected_json);
        let bootstrap_logging = crate::logging::bootstrap_config(None);
        assert_eq!(bootstrap_logging.level, expected.logging.level);
        assert_eq!(bootstrap_logging.format, expected.logging.format);
        let effective = actual.effective_configuration().unwrap();
        assert_eq!(effective.effective_config_hash, expected_hash);
        assert!(effective.fields.values().all(|field| {
            field.source == ConfigValueSource::Default && field.compatibility_environment.is_empty()
        }));
    }

    #[test]
    fn public_env_every_fixed_typed_leaf_has_an_explicit_classification() {
        let mut config = KilnConfig::default();
        config.eval = Some(EvalConfig::default());
        config.agent = Some(AgentConfig::default());

        let mut serialized_leaves = Vec::new();
        collect_json_leaf_paths(
            "",
            &serde_json::to_value(&config).unwrap(),
            &mut serialized_leaves,
        );
        serialized_leaves.sort();
        assert_eq!(
            serde_json::to_value(&config)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            17
        );
        assert_eq!(serialized_leaves.len(), 117);
        assert_eq!(CONFIG_FILE_ONLY_FIXED_FIELDS.len(), 5);

        let mut classified = PUBLIC_ENV_FIELDS
            .iter()
            .map(PublicEnvField::field_path)
            .chain(
                CONFIG_FILE_ONLY_FIXED_FIELDS
                    .iter()
                    .map(|path| (*path).to_owned()),
            )
            .collect::<Vec<_>>();
        classified.sort();
        classified.dedup();
        assert_eq!(serialized_leaves, classified);

        config.teachers.credentials.insert(
            "probe".to_owned(),
            TeacherCredentialConfig {
                origin: "https://teacher.example".to_owned(),
                api_key_env: "PROBE_API_KEY".to_owned(),
            },
        );
        let mut teacher_leaves = Vec::new();
        collect_json_leaf_paths(
            "teachers",
            &serde_json::to_value(&config.teachers).unwrap(),
            &mut teacher_leaves,
        );
        for path in &mut teacher_leaves {
            *path = path.replace(".probe.", ".<id>.");
        }
        teacher_leaves.sort();
        assert_eq!(
            teacher_leaves,
            DYNAMIC_CONFIG_FIELDS
                .iter()
                .map(|path| (*path).to_owned())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn effective_configuration_is_complete_source_aware_and_redacted() {
        const SECRET_ENV: &str = "KILN_TEST_EFFECTIVE_CONFIG_SECRET";

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated_with(&[SECRET_ENV]);
        environment.set(SECRET_ENV, "must-not-appear-in-the-dump");
        environment.set("KILN_SERVER_HOST", "127.0.0.9");

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
port = 9001

[eval]
max_queued_jobs = 9
webhook_url = "https://eval.example.invalid/private"

[training]
webhook_url = "https://training.example.invalid/private"

[agent]
self_improve = {{ task = "private-task" }}

[teachers.credentials.primary]
origin = "http://127.0.0.1:8000"
api_key_env = "{SECRET_ENV}"
"#
            ),
        )
        .unwrap();

        let mut config = KilnConfig::load(path.to_str()).unwrap();
        config
            .apply_serve_cli_overrides(Some("operator-model-id"), false)
            .unwrap();
        let effective = config.effective_configuration().unwrap();

        assert_eq!(effective.schema_id, EFFECTIVE_CONFIGURATION_SCHEMA_ID);
        assert_eq!(effective.schema_version, 1);
        assert_eq!(effective.fixed_field_count, 117);
        assert_eq!(effective.dynamic_field_count, 2);
        assert_eq!(effective.fields.len(), 119);
        assert!(effective.all_fields_restart_required_to_change);
        assert!(
            effective
                .fields
                .values()
                .all(|field| field.restart_required_to_change)
        );
        assert_eq!(
            effective.effective_config_hash,
            kiln_core::config_hashes::effective_config_hash(&config)
        );
        assert_eq!(effective, config.effective_configuration().unwrap());
        let sources = std::mem::take(&mut config.value_sources);
        assert_eq!(
            effective.effective_config_hash,
            kiln_core::config_hashes::effective_config_hash(&config),
            "per-field source metadata must not change typed config identity"
        );
        config.value_sources = sources;

        let default = &effective.fields["server.request_timeout_secs"];
        assert_eq!(default.source, ConfigValueSource::Default);
        assert_eq!(default.effective_value, serde_json::json!(600));

        let file = &effective.fields["server.port"];
        assert_eq!(file.source, ConfigValueSource::ConfigFile);
        assert_eq!(file.effective_value, serde_json::json!(9001));
        assert_eq!(
            file.canonical_environment.as_deref(),
            Some("KILN_SERVER_PORT")
        );

        let environment = &effective.fields["server.host"];
        assert_eq!(environment.source, ConfigValueSource::Environment);
        assert_eq!(environment.effective_value, serde_json::json!("127.0.0.9"));
        assert_eq!(
            environment.canonical_environment.as_deref(),
            Some("KILN_SERVER_HOST")
        );
        assert!(environment.compatibility_environment.is_empty());

        let command_line = &effective.fields["model.served_model_id"];
        assert_eq!(command_line.source, ConfigValueSource::CommandLine);
        assert_eq!(
            command_line.effective_value,
            serde_json::json!("operator-model-id")
        );

        let credential_origin = &effective.fields["teachers.credentials.primary.origin"];
        assert_eq!(credential_origin.source, ConfigValueSource::ConfigFile);
        assert_eq!(
            credential_origin.effective_value,
            serde_json::json!("http://127.0.0.1:8000")
        );
        let credential_secret = &effective.fields["teachers.credentials.primary.api_key_env"];
        assert_eq!(credential_secret.source, ConfigValueSource::ConfigFile);
        assert!(credential_secret.redacted);
        assert!(credential_secret.effective_value.is_null());

        for path in [
            "agent.self_improve",
            "eval.webhook_url",
            "training.webhook_url",
        ] {
            let field = &effective.fields[path];
            assert_eq!(field.source, ConfigValueSource::ConfigFile, "{path}");
            assert!(field.redacted, "{path}");
            assert!(field.effective_value.is_null(), "{path}");
        }

        let serialized = serde_json::to_string(&effective).unwrap();
        for secret in [
            "must-not-appear-in-the-dump",
            SECRET_ENV,
            "private-task",
            "eval.example.invalid",
            "training.example.invalid",
        ] {
            assert!(!serialized.contains(secret), "dump leaked {secret:?}");
        }
    }

    #[test]
    fn public_env_canonical_only_loads_all_public_fields() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, value) in [
            ("KILN_SERVER_SERVING_PROFILE", "experimental"),
            ("KILN_ACCELERATOR_KT_API_MODE", "all"),
            ("KILN_ACCELERATOR_FULL_ATTENTION_SCORE_BUDGET_MIB", "512"),
            (
                "KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE",
                "stream_ordered",
            ),
            ("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE", "portable_fallback"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_MODE", "lazy_capture_replay"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES", "13"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES", "536870912"),
            ("KILN_SERVER_DETERMINISTIC", "true"),
            ("KILN_SERVER_HOST", "127.0.0.2"),
            ("KILN_SERVER_PORT", "9444"),
            ("KILN_SERVER_REQUEST_TIMEOUT_SECS", "321"),
            ("KILN_SERVER_TERMINAL_ACCESS", "disabled"),
            ("KILN_SERVER_HTTP_SEND_BUFFER_BYTES", "4096"),
            ("KILN_SERVER_STREAM_STALL_GRACE_MS", "100"),
            ("KILN_SERVER_MAX_BATCH_TOKENS", "1024"),
            ("KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE", "128"),
            ("KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE", "8"),
            ("KILN_SERVER_MAX_DECODE_BATCH", "backend_policy"),
            ("KILN_SERVER_EVAL_MODE", "true"),
            ("KILN_SERVER_DEBUG_MODEL_STATE", "true"),
            ("KILN_SERVER_DEFAULT_THINKING_ENABLED", "false"),
            ("KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS", "7"),
            ("KILN_SERVER_DEFAULT_THINKING_BUDGET_MS", "20"),
            ("KILN_SERVER_FOLD_REASONING_INTO_CONTENT", "true"),
            ("KILN_SERVER_CHAT_PERFORMANCE_METADATA", "true"),
            ("KILN_SERVER_CHAT_CONFIG_HASH_METADATA", "true"),
            ("KILN_SERVER_SLOW_REQUEST_WARN_SECS", "0"),
            ("KILN_SERVER_SHUTDOWN_TIMEOUT_SECS", "9"),
            ("KILN_BATCHING_ROWWISE_DECODE", "true"),
            ("KILN_BATCHING_PREFIX_AWARE_ADMISSION", "false"),
            ("KILN_BATCHING_PREFILL_ADMISSION_QUANTUM", "16"),
            ("KILN_BATCHING_ACTOR_CYCLE_IDLE_MS", "75"),
            ("KILN_MODEL_PATH", "/tmp/canonical-model"),
            ("KILN_MODEL_MODEL_ID", "Canonical/Test-Model"),
            ("KILN_MODEL_TOKENIZER_PATH", "/tmp/canonical-tokenizer"),
            ("KILN_MODEL_ADAPTER_DIR", "/tmp/canonical-adapters"),
            ("KILN_MODEL_SNAPSHOT_DIR", "/tmp/canonical-snapshot"),
            ("KILN_MODEL_SERVED_MODEL_ID", "canonical-served"),
            ("KILN_PATHS_CACHE_ROOT", "/tmp/canonical-kiln-cache"),
            ("KILN_MEMORY_NUM_BLOCKS", "123"),
            ("KILN_MEMORY_GPU_MEMORY_GB", "64"),
            ("KILN_MEMORY_INFERENCE_MEMORY_FRACTION", "0.6"),
            ("KILN_MEMORY_TRAINING_MEMORY_GB", "8"),
            ("KILN_MEMORY_VULKAN_BUFFER_POOL_GB", "1.75"),
            ("KILN_MEMORY_FLOOR_GB", "2.5"),
            ("KILN_MEMORY_PROBE_MS", "750"),
            ("KILN_MEMORY_RECLAIM_MODE", "on-demand"),
            ("KILN_MEMORY_KV_AUTOSCALE", "false"),
            ("KILN_MEMORY_KV_FORCE_BLOCKS", "0"),
            ("KILN_MEMORY_KV_CACHE_FP8", "true"),
            ("KILN_MEMORY_CUDA_GRAPHS", "false"),
            ("KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES", "13"),
            ("KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS", "4"),
            ("KILN_TRAINING_NO_GRAD_CHECKPOINT", "true"),
            ("KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES", "enabled"),
            ("KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS", "4096"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE", "3"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB", "2.5"),
            ("KILN_TRAINING_CHECKPOINT_INTERVAL", "5"),
            ("KILN_TRAINING_WEBHOOK_URL", "https://hook.example/test"),
            ("KILN_TRAINING_LOGIT_CACHE_DIR", "/tmp/canonical-logits"),
            ("KILN_TRAINING_MAX_QUEUED_JOBS", "7"),
            ("KILN_TRAINING_MAX_TRACKED_JOBS", "9"),
            ("KILN_TRAINING_TRACKED_JOB_TTL_SECS", "11"),
            ("KILN_LOGGING_LEVEL", "debug"),
            ("KILN_LOGGING_FORMAT", "json"),
            ("KILN_PREFIX_CACHE_ENABLED", "false"),
            ("KILN_PREFIX_CACHE_MAX_BLOCKS", "64"),
            ("KILN_PREFIX_CACHE_MAX_ENTRIES", "12"),
            ("KILN_SPECULATIVE_METHOD", "mtp"),
            ("KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS", "3"),
            ("KILN_SPECULATIVE_DRAFT_LAYERS", "2"),
            ("KILN_STREAMING_PREFILL_MODE", "enabled"),
            ("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS", "1024"),
            ("KILN_STREAMING_PREFILL_TILE_TOKENS", "2048"),
            ("KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS", "4096"),
            (
                "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
                "512",
            ),
            ("KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD", "false"),
            ("KILN_ADAPTERS_MAX_DISK_BYTES", "1024"),
            ("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "512"),
            ("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "6"),
            ("KILN_ADAPTERS_LIBRARY_URL", "https://library.example/test"),
            ("KILN_REQUEST_LOG_ENABLED", "false"),
            ("KILN_REQUEST_LOG_DIR", "/tmp/canonical-request-log"),
            ("KILN_REQUEST_LOG_MAX_FILE_BYTES", "8192"),
            ("KILN_REQUEST_LOG_MAX_TOTAL_BYTES", "16384"),
            ("KILN_REQUEST_LOG_COMPRESS", "false"),
            ("KILN_REQUEST_LOG_MAX_CAPTURE_BYTES", "2048"),
            ("KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS", "12"),
            ("KILN_AGENT_MAX_CONCURRENT_RUNS", "3"),
            ("KILN_AGENT_RUN_TIMEOUT_SECS", "45"),
            ("KILN_AGENT_RUNS_ACCESS", "enabled"),
            ("KILN_AGENT_PI_BIN", "/bin/sh"),
            ("KILN_AGENT_PI_SESSIONS_DIR", "/tmp/canonical-pi-sessions"),
        ] {
            environment.set(name, value);
        }

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        let config = KilnConfig::load(Some(path.to_str().unwrap())).unwrap();

        assert_eq!(
            config.server.serving_profile.profile(),
            ServingProfile::Experimental
        );
        assert_eq!(config.accelerator.kt_api_mode.mode(), KtApiMode::All);
        assert_eq!(
            config.accelerator.full_attention_score_budget_mib.mib(),
            512
        );
        assert_eq!(
            config.accelerator.full_attention_score_budget_mib.source(),
            ConfigValueSource::Environment
        );
        assert_eq!(
            config.accelerator.rocm_synchronization_mode.mode(),
            RocmSynchronizationMode::StreamOrdered
        );
        assert_eq!(
            config.accelerator.rocm_kernel_profile.profile(),
            RocmKernelProfile::PortableFallback
        );
        assert_eq!(
            config.accelerator.rocm_graph_mode.mode(),
            RocmGraphMode::LazyCaptureReplay
        );
        assert_eq!(config.accelerator.rocm_graph_cache_entries.entries(), 13);
        assert_eq!(
            config.accelerator.rocm_graph_cache_max_bytes.bytes(),
            536_870_912
        );
        assert!(config.server.deterministic.enabled());
        assert_eq!(config.server.host, "127.0.0.2");
        assert_eq!(config.server.port, 9444);
        assert_eq!(config.server.request_timeout_secs, 321);
        assert_eq!(
            config.server.terminal_access,
            LocalCapabilityAccess::Disabled
        );
        assert_eq!(config.server.http_send_buffer_bytes, Some(4096));
        assert_eq!(config.server.stream_stall_grace_ms.millis(), 100);
        assert_eq!(config.server.max_batch_tokens.tokens(), 1024);
        assert_eq!(config.server.max_prefill_tokens_per_cycle.tokens(), 128);
        assert_eq!(config.server.max_prefill_layers_per_cycle.layers(), 8);
        assert_eq!(config.server.max_decode_batch.limit(), None);
        assert!(config.server.eval_mode);
        assert!(config.server.debug_model_state);
        assert_eq!(config.server.default_thinking_enabled, Some(false));
        assert_eq!(config.server.default_thinking_budget_tokens, Some(7));
        assert_eq!(config.server.default_thinking_budget_ms, Some(20));
        assert!(config.server.fold_reasoning_into_content);
        assert!(config.server.chat_performance_metadata);
        assert!(config.server.chat_config_hash_metadata);
        assert_eq!(config.server.slow_request_warn_secs, 0);
        assert_eq!(config.server.shutdown_timeout_secs, 9);
        assert!(config.batching.rowwise_decode.enabled());
        assert!(!config.batching.prefix_aware_admission.enabled());
        assert_eq!(
            config.batching.prefill_admission_quantum.configured(),
            Some(16)
        );
        assert_eq!(config.batching.actor_cycle_idle_ms.millis(), 75);
        assert_eq!(
            config.batching.actor_cycle_idle_ms.source(),
            ConfigValueSource::Environment
        );
        assert_eq!(config.model.path.as_deref(), Some("/tmp/canonical-model"));
        assert_eq!(config.model.model_id, "Canonical/Test-Model");
        assert_eq!(
            config.model.tokenizer_path.as_deref(),
            Some("/tmp/canonical-tokenizer")
        );
        assert_eq!(
            config.model.adapter_dir.as_deref(),
            Some("/tmp/canonical-adapters")
        );
        assert_eq!(
            config.model.snapshot_dir.as_deref(),
            Some("/tmp/canonical-snapshot")
        );
        assert_eq!(
            config.model.served_model_id.as_deref(),
            Some("canonical-served")
        );
        assert_eq!(
            config.paths.cache_root.configured(),
            Some(Path::new("/tmp/canonical-kiln-cache"))
        );
        assert_eq!(
            config.paths.cache_root.source(),
            ConfigValueSource::Environment
        );
        assert_eq!(config.memory.num_blocks, Some(123));
        assert_eq!(config.memory.gpu_memory_gb, Some(64.0));
        assert_eq!(config.memory.inference_memory_fraction, 0.6);
        assert_eq!(config.memory.training_memory_gb, Some(8.0));
        assert_eq!(config.memory.vulkan_buffer_pool_gb, 1.75);
        assert_eq!(config.memory.floor_gb, 2.5);
        assert_eq!(config.memory.probe_ms, 750);
        assert_eq!(
            config.memory.reclaim_mode.mode(),
            kiln_memory::MemoryReclaimMode::OnDemand
        );
        assert_eq!(
            config.memory.reclaim_mode.source(),
            ConfigValueSource::Environment
        );
        assert!(!config.memory.kv_autoscale.enabled());
        assert_eq!(
            config.memory.kv_autoscale.source(),
            ConfigValueSource::Environment
        );
        assert_eq!(config.memory.kv_force_blocks.target(), None);
        assert_eq!(
            config.memory.kv_force_blocks.source(),
            ConfigValueSource::Environment
        );
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs);
        assert_eq!(config.memory.cuda_graph_cache_entries, 13);
        assert_eq!(config.training.grad_checkpoint_segments, Some(4));
        assert!(config.training.no_grad_checkpoint);
        assert_eq!(
            config.training.recompute_checkpoint_boundaries.mode(),
            kiln_train::CheckpointBoundaryRecomputeMode::Enabled
        );
        assert_eq!(
            config.training.recompute_boundary_threshold_tokens.tokens(),
            4096
        );
        assert_eq!(
            config
                .training
                .checkpoint_boundary_anchor_stride
                .configured(),
            Some(3)
        );
        assert_eq!(config.training.checkpoint_boundary_cache_gb.gib(), 2.5);
        assert_eq!(
            config.training.checkpoint_boundary_cache_gb.bytes(),
            2_684_354_560
        );
        assert_eq!(config.training.checkpoint_interval, Some(5));
        assert_eq!(
            config.training.webhook_url.as_deref(),
            Some("https://hook.example/test")
        );
        assert_eq!(
            config.training.logit_cache_dir.as_deref(),
            Some(Path::new("/tmp/canonical-logits"))
        );
        assert_eq!(config.training.max_queued_jobs, 7);
        assert_eq!(config.training.max_tracked_jobs, 9);
        assert_eq!(config.training.tracked_job_ttl_secs, 11);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.format, "json");
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(64));
        assert_eq!(config.prefix_cache.max_entries, Some(12));
        assert_eq!(config.speculative.method, SpecMethod::Mtp);
        assert_eq!(config.speculative.num_speculative_tokens, 3);
        assert_eq!(config.speculative.draft_layers, 2);
        assert_eq!(
            config.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Enabled
        );
        assert_eq!(
            config.streaming_prefill.threshold_tokens.configured(),
            Some(1024)
        );
        assert_eq!(
            config.streaming_prefill.tile_tokens.configured(),
            Some(2048)
        );
        assert_eq!(
            config.streaming_prefill.tape_tile_tokens.configured(),
            Some(4096)
        );
        assert_eq!(
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
            Some(512)
        );
        assert!(!config.streaming_prefill.last_token_lm_head.enabled());
        assert_eq!(config.adapters.max_disk_bytes, Some(1024));
        assert_eq!(config.adapters.composed_cache_max_bytes, Some(512));
        assert_eq!(config.adapters.composed_cache_max_entries, Some(6));
        assert_eq!(config.adapters.library_url, "https://library.example/test");
        assert!(!config.request_log.enabled);
        assert_eq!(
            config.request_log.dir.as_deref(),
            Some(Path::new("/tmp/canonical-request-log"))
        );
        assert_eq!(config.request_log.max_file_bytes, 8192);
        assert_eq!(config.request_log.max_total_bytes, 16384);
        assert!(!config.request_log.compress);
        assert_eq!(config.request_log.max_capture_bytes, 2048);
        let agent = config.agent.as_ref().unwrap();
        assert_eq!(agent.self_improve_interval_hours, Some(12));
        assert_eq!(agent.max_concurrent_runs, 3);
        assert_eq!(agent.run_timeout_secs, 45);
        assert_eq!(agent.runs_access, LocalCapabilityAccess::Enabled);
        assert_eq!(agent.pi_bin.as_deref(), Some(Path::new("/bin/sh")));
        assert_eq!(
            agent.pi_sessions_dir.as_deref(),
            Some(Path::new("/tmp/canonical-pi-sessions"))
        );

        for source in [
            config.server.serving_profile.source(),
            config.accelerator.kt_api_mode.source(),
            config.accelerator.rocm_synchronization_mode.source(),
            config.accelerator.rocm_kernel_profile.source(),
            config.accelerator.rocm_graph_mode.source(),
            config.accelerator.rocm_graph_cache_entries.source(),
            config.accelerator.rocm_graph_cache_max_bytes.source(),
            config.server.deterministic.source(),
            config.server.stream_stall_grace_ms.source(),
            config.server.max_batch_tokens.source(),
            config.server.max_prefill_tokens_per_cycle.source(),
            config.server.max_prefill_layers_per_cycle.source(),
            config.server.max_decode_batch.source(),
            config.batching.rowwise_decode.source(),
            config.batching.prefix_aware_admission.source(),
            config.batching.prefill_admission_quantum.source(),
            config.memory.kv_autoscale.source(),
            config.memory.kv_force_blocks.source(),
            config.training.recompute_checkpoint_boundaries.source(),
            config.training.recompute_boundary_threshold_tokens.source(),
            config.training.checkpoint_boundary_anchor_stride.source(),
            config.training.checkpoint_boundary_cache_gb.source(),
            config.streaming_prefill.mode.source(),
            config.streaming_prefill.threshold_tokens.source(),
            config.streaming_prefill.tile_tokens.source(),
            config.streaming_prefill.tape_tile_tokens.source(),
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
            config.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }
    }

    #[test]
    fn local_capability_access_recognizes_all_loopback_ip_forms() {
        for host in [
            "localhost",
            "LOCALHOST",
            "127.0.0.1",
            "127.42.0.9",
            "::1",
            "[::1]",
        ] {
            assert!(host_is_loopback(host), "{host}");
        }
        for host in ["0.0.0.0", "192.168.1.10", "::", "kiln.internal"] {
            assert!(!host_is_loopback(host), "{host}");
        }
    }

    #[test]
    fn public_env_malformed_canonical_inputs_fail_closed_and_name_the_input() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, invalid) in [
            ("KILN_SERVER_SERVING_PROFILE", "sometimes"),
            ("KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE", "eventually"),
            ("KILN_ACCELERATOR_ROCM_KERNEL_PROFILE", "custom"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_MODE", "automatic"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES", "0"),
            ("KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES", "67108863"),
            ("KILN_SERVER_PORT", "nine-thousand"),
            ("KILN_SERVER_DETERMINISTIC", "maybe"),
            ("KILN_SERVER_TERMINAL_ACCESS", "sometimes"),
            ("KILN_SERVER_DEFAULT_THINKING_BUDGET_MS", "2.5"),
            ("KILN_BATCHING_ROWWISE_DECODE", "row-by-row-ish"),
            ("KILN_BATCHING_PREFIX_AWARE_ADMISSION", "preferably"),
            ("KILN_BATCHING_PREFILL_ADMISSION_QUANTUM", "0"),
            ("KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES", "true"),
            ("KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS", "0"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE", "0"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB", "inf"),
            ("KILN_TRAINING_LOGIT_CACHE_DIR", ""),
            ("KILN_STREAMING_PREFILL_MODE", "true"),
            ("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS", "0"),
            ("KILN_STREAMING_PREFILL_TILE_TOKENS", "65"),
            ("KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS", "127"),
            (
                "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
                "not-auto",
            ),
            ("KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD", "sometimes"),
            ("KILN_MEMORY_RECLAIM_MODE", "whenever"),
            ("KILN_MEMORY_KV_AUTOSCALE", "sometimes"),
            ("KILN_MEMORY_KV_FORCE_BLOCKS", "-1"),
            ("KILN_PATHS_CACHE_ROOT", ""),
            ("KILN_SPECULATIVE_METHOD", "guessing"),
            ("KILN_REQUEST_LOG_COMPRESS", "occasionally"),
            ("KILN_AGENT_RUNS_ACCESS", "trusted-ish"),
            ("KILN_AGENT_PI_BIN", ""),
            ("KILN_AGENT_PI_SESSIONS_DIR", ""),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains(invalid), "{name}: {detail}");
        }
    }

    #[test]
    fn cache_root_toml_is_source_tracked_and_resolves_absolutely() {
        let config: KilnConfig = toml::from_str("[paths]\ncache_root = 'relative-cache'\n")
            .expect("relative cache root parses");
        assert_eq!(
            config.paths.cache_root.configured(),
            Some(Path::new("relative-cache"))
        );
        assert_eq!(
            config.paths.cache_root.source(),
            ConfigValueSource::ConfigFile
        );
        let resolved = config.paths.resolve().expect("cache root resolves");
        assert_eq!(
            resolved.cache_root,
            std::env::current_dir().unwrap().join("relative-cache")
        );
        assert_eq!(resolved.cache_root_source, ConfigValueSource::ConfigFile);
        assert!(resolved.restart_required_to_change);

        assert!(toml::from_str::<KilnConfig>("[paths]\ncache_root = ''\n").is_err());
    }

    #[test]
    fn batching_toml_is_strict_source_tracked_and_bounded() {
        for quantum in [PREFILL_ADMISSION_QUANTUM_MIN, PREFILL_ADMISSION_QUANTUM_MAX] {
            let config: KilnConfig = toml::from_str(&format!(
                "[batching]\nprefill_admission_quantum = {quantum}\n"
            ))
            .unwrap();
            assert_eq!(
                config.batching.prefill_admission_quantum.configured(),
                Some(quantum)
            );
            assert_eq!(
                config.batching.prefill_admission_quantum.source(),
                ConfigValueSource::ConfigFile
            );
        }

        let auto: KilnConfig =
            toml::from_str("[batching]\nprefill_admission_quantum = \"auto\"\n").unwrap();
        assert_eq!(auto.batching.prefill_admission_quantum.configured(), None);
        assert_eq!(
            auto.batching.prefill_admission_quantum.source(),
            ConfigValueSource::ConfigFile
        );

        for document in [
            "[batching]\nmode = \"enabled\"\n".to_owned(),
            "[batching]\nrowwise_decode = \"true\"\n".to_owned(),
            "[batching]\nprefix_aware_admission = 1\n".to_owned(),
            "[batching]\nprefill_admission_quantum = 0\n".to_owned(),
            format!(
                "[batching]\nprefill_admission_quantum = {}\n",
                PREFILL_ADMISSION_QUANTUM_MAX + 1
            ),
            "[batching]\nprefill_admission_quantum = \"backend-ish\"\n".to_owned(),
        ] {
            let error = toml::from_str::<KilnConfig>(&document).unwrap_err();
            let detail = error.to_string();
            assert!(
                detail.contains("batching")
                    || detail.contains("invalid type")
                    || detail.contains("unknown field"),
                "unexpected error for {document:?}: {error:#}"
            );
        }
    }

    #[test]
    fn batching_environment_overrides_toml_with_environment_provenance() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, value) in [
            ("KILN_BATCHING_ROWWISE_DECODE", "yes"),
            ("KILN_BATCHING_PREFIX_AWARE_ADMISSION", "off"),
            ("KILN_BATCHING_PREFILL_ADMISSION_QUANTUM", "9"),
        ] {
            environment.set(name, value);
        }

        let mut config: KilnConfig = toml::from_str(
            r#"
[batching]
rowwise_decode = false
prefix_aware_admission = true
prefill_admission_quantum = "auto"
"#,
        )
        .unwrap();
        config.apply_env_overrides().unwrap();

        assert!(config.batching.rowwise_decode.enabled());
        assert!(!config.batching.prefix_aware_admission.enabled());
        assert_eq!(
            config.batching.prefill_admission_quantum.configured(),
            Some(9)
        );
        for source in [
            config.batching.rowwise_decode.source(),
            config.batching.prefix_aware_admission.source(),
            config.batching.prefill_admission_quantum.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }

        for name in [
            "KILN_BATCHING_ROWWISE_DECODE",
            "KILN_BATCHING_PREFIX_AWARE_ADMISSION",
            "KILN_BATCHING_PREFILL_ADMISSION_QUANTUM",
        ] {
            environment.remove(name);
        }
        environment.set("KILN_BATCHING_PREFILL_ADMISSION_QUANTUM", " AUTO ");
        let mut auto = KilnConfig::default();
        auto.apply_env_overrides().unwrap();
        assert_eq!(auto.batching.prefill_admission_quantum.configured(), None);
        assert_eq!(
            auto.batching.prefill_admission_quantum.source(),
            ConfigValueSource::Environment
        );
    }

    #[test]
    fn batching_malformed_environment_values_fail_closed() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, invalid) in [
            ("KILN_BATCHING_ROWWISE_DECODE", "maybe"),
            ("KILN_BATCHING_PREFIX_AWARE_ADMISSION", ""),
            ("KILN_BATCHING_PREFILL_ADMISSION_QUANTUM", "65537"),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains(&format!("{invalid:?}")), "{name}: {detail}");
        }
    }

    #[test]
    fn batching_runtime_resolution_preserves_actor_policy_and_clamps_quantum() {
        let defaults = BatchingConfig::default();
        let latency_backend = defaults.resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
            },
            64,
        );
        assert_eq!(latency_backend.prefill_admission_quantum.backend_policy, 4);
        assert_eq!(latency_backend.prefill_admission_quantum.effective, 4);
        assert_eq!(
            latency_backend.prefill_admission_quantum.effective_source,
            BatchingEffectiveSource::BackendPolicy
        );

        let throughput_backend = defaults.resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: true,
                burst_prefill_admission: true,
                actor_prefill_tile_alignment_required: false,
            },
            64,
        );
        assert!(throughput_backend.burst_prefill_admission);
        assert_eq!(
            throughput_backend.prefill_admission_quantum.backend_policy,
            64
        );
        assert_eq!(throughput_backend.prefill_admission_quantum.effective, 64);
        assert_eq!(
            throughput_backend.actor_admission_config(),
            BatchingActorAdmissionConfig {
                prefix_aware_admission: true,
                prefill_admission_quantum: 64,
                burst_prefill_admission: true,
            }
        );

        let explicit: KilnConfig = toml::from_str(
            r#"
[batching]
rowwise_decode = true
prefix_aware_admission = false
prefill_admission_quantum = 100
"#,
        )
        .unwrap();
        let resolved = explicit.batching.resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: true,
                burst_prefill_admission: true,
                actor_prefill_tile_alignment_required: false,
            },
            16,
        );
        assert!(resolved.rowwise_decode.enabled);
        assert!(!resolved.prefix_aware_admission.enabled);
        assert!(resolved.burst_prefill_admission);
        assert_eq!(resolved.prefill_admission_quantum.configured, Some(100));
        assert_eq!(resolved.prefill_admission_quantum.effective, 16);
        assert_eq!(
            resolved.prefill_admission_quantum.effective_source,
            BatchingEffectiveSource::EffectiveDecodeWidth
        );
    }

    #[test]
    fn memory_governor_environment_values_are_validated_at_startup() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();

        for (name, invalid, field) in [
            ("KILN_MEMORY_FLOOR_GB", "-0.5", "memory.floor_gb"),
            ("KILN_MEMORY_FLOOR_GB", "inf", "memory.floor_gb"),
            ("KILN_MEMORY_PROBE_MS", "0", "memory.probe_ms"),
            (
                "KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES",
                "0",
                "memory.cuda_graph_cache_entries",
            ),
            (
                "KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES",
                "65",
                "memory.cuda_graph_cache_entries",
            ),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::load(Some(path.to_str().unwrap())).unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(field), "{name}: {detail}");
            assert!(detail.contains(invalid), "{name}: {detail}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn public_env_non_unicode_canonical_input_is_fatal() {
        use std::os::unix::ffi::OsStringExt;

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let invalid = OsString::from_vec(vec![b'9', 0xff]);
        environment.set_os("KILN_SERVER_PORT", &invalid);

        let error = KilnConfig::default().apply_env_overrides().unwrap_err();
        let detail = format!("{error:#}");
        assert!(detail.contains("KILN_SERVER_PORT"), "{detail}");
        assert!(detail.contains("UTF-8"), "{detail}");
    }

    #[cfg(unix)]
    #[test]
    fn streaming_prefill_non_unicode_canonical_inputs_are_fatal() {
        use std::os::unix::ffi::OsStringExt;

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for name in [
            "KILN_STREAMING_PREFILL_MODE",
            "KILN_STREAMING_PREFILL_THRESHOLD_TOKENS",
            "KILN_STREAMING_PREFILL_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD",
        ] {
            let invalid = OsString::from_vec(vec![b'1', 0xff]);
            environment.set_os(name, &invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains("UTF-8"), "{name}: {detail}");
        }
    }

    #[test]
    fn public_env_config_file_only_and_dynamic_targets_are_not_public() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for name in INTENTIONALLY_UNMAPPED_ENV_TARGETS {
            environment.set(name, "1");
        }

        let mut config = KilnConfig::default();
        config.apply_env_overrides().unwrap();
        assert!(config.eval.is_none());
        assert!(config.agent.is_none());
        assert!(config.teachers.credentials.is_empty());
        assert!(PUBLIC_ENV_FIELDS.iter().all(|field| {
            !INTENTIONALLY_UNMAPPED_ENV_TARGETS.contains(&field.canonical_name().as_str())
        }));
    }

    #[test]
    fn public_env_typed_serve_cli_overrides_win_without_env_mutation() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        environment.set("KILN_MODEL_SERVED_MODEL_ID", "from-env");
        environment.set("KILN_SERVER_EVAL_MODE", "false");

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(
            &path,
            "[server]\neval_mode = false\n[model]\nserved_model_id = 'from-toml'\n",
        )
        .unwrap();
        let mut config = KilnConfig::load(Some(path.to_str().unwrap())).unwrap();
        config
            .apply_serve_cli_overrides(Some("from-cli"), true)
            .unwrap();
        assert_eq!(config.model.served_model_id.as_deref(), Some("from-cli"));
        assert!(config.server.eval_mode);
        assert_eq!(
            std::env::var("KILN_MODEL_SERVED_MODEL_ID").unwrap(),
            "from-env"
        );
        assert_eq!(std::env::var("KILN_SERVER_EVAL_MODE").unwrap(), "false");

        environment.set("KILN_SERVER_EVAL_MODE", "true");
        let mut config = KilnConfig::load(Some(path.to_str().unwrap())).unwrap();
        let resolved_id = config.model.served_model_id.clone();
        config.apply_serve_cli_overrides(None, false).unwrap();
        assert_eq!(config.model.served_model_id, resolved_id);
        assert!(config.server.eval_mode);

        let error = config
            .apply_serve_cli_overrides(Some("   "), false)
            .unwrap_err();
        assert!(format!("{error:#}").contains("model.served_model_id"));

        let mut invalid = KilnConfig::default();
        invalid.server.port = 0;
        let error = invalid.apply_serve_cli_overrides(None, false).unwrap_err();
        assert!(format!("{error:#}").contains("server.port"));
    }

    #[test]
    fn public_env_canonical_clear_semantics_match_existing_contract() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let mut config: KilnConfig = toml::from_str(
            "[model]\nsnapshot_dir = '/tmp/from-toml'\n\
             [training]\nwebhook_url = 'https://hook.example/toml'\n\
             [adapters]\nmax_disk_bytes = 99\n",
        )
        .unwrap();

        environment.set("KILN_MODEL_SNAPSHOT_DIR", "   ");
        environment.set("KILN_TRAINING_WEBHOOK_URL", "");
        environment.set("KILN_ADAPTERS_MAX_DISK_BYTES", "0");
        config.apply_env_overrides().unwrap();
        assert!(config.model.snapshot_dir.is_none());
        assert!(config.training.webhook_url.is_none());
        assert!(config.adapters.max_disk_bytes.is_none());

        environment.set("KILN_REQUEST_LOG_DIR", "   ");
        let error = KilnConfig::default().apply_env_overrides().unwrap_err();
        let detail = format!("{error:#}");
        assert!(detail.contains("KILN_REQUEST_LOG_DIR"), "{detail}");
        assert!(detail.contains("non-empty path"), "{detail}");
    }

    #[test]
    fn teacher_credentials_parse_and_resolve_only_for_the_exact_origin() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const ENV: &str = "KILN_TEST_SCOPED_TEACHER_SECRET";
        let environment = ScopedConfigEnvironment::isolated_with(&[ENV]);
        environment.set(ENV, "test-secret");
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
    }

    #[test]
    fn teacher_credentials_reject_invalid_definitions_and_missing_secrets() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const ENV: &str = "KILN_TEST_MISSING_TEACHER_SECRET";
        let environment = ScopedConfigEnvironment::isolated_with(&[ENV]);

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

        environment.set(ENV, "   ");
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("empty"), "{error}");

        environment.set(ENV, "secret");
        config
            .teachers
            .credentials
            .get_mut("valid-id")
            .unwrap()
            .api_key_env = "1INVALID".into();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("[A-Za-z_]"), "{error}");
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
deterministic = true
host = "127.0.0.1"
port = 9000
request_timeout_secs = 60
http_send_buffer_bytes = 8192
stream_stall_grace_ms = 1500
max_batch_tokens = 1024
max_prefill_tokens_per_cycle = 192
max_prefill_layers_per_cycle = 6
max_decode_batch = 24
eval_mode = true
default_thinking_enabled = false
default_thinking_budget_tokens = 256
default_thinking_budget_ms = 1500
fold_reasoning_into_content = true
chat_performance_metadata = true
chat_config_hash_metadata = true
slow_request_warn_secs = 15
shutdown_timeout_secs = 10

[batching]
rowwise_decode = true
prefix_aware_admission = false
prefill_admission_quantum = 12

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
vulkan_buffer_pool_gb = 1.5
floor_gb = 2.0
probe_ms = 250
reclaim_mode = "automatic"
kv_autoscale = false
kv_force_blocks = 0
kv_cache_fp8 = true
cuda_graphs = false
cuda_graph_cache_entries = 16

[training]
grad_checkpoint_segments = 8
no_grad_checkpoint = false
recompute_checkpoint_boundaries = "disabled"
recompute_boundary_threshold_tokens = 4096
checkpoint_boundary_anchor_stride = 4
checkpoint_boundary_cache_gb = 2.5
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
method = "skip_layer"
num_speculative_tokens = 6
draft_layers = 10

[streaming_prefill]
mode = "enabled"
threshold_tokens = 1024
tile_tokens = 4096
tape_tile_tokens = 2048
detached_full_attn_tile_tokens = 512
last_token_lm_head = false

[adapters]
max_disk_bytes = 5368709120
composed_cache_max_bytes = 1073741824
composed_cache_max_entries = 8
"#;
        let config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(config.server.deterministic.enabled());
        assert_eq!(
            config.server.deterministic.source(),
            ConfigValueSource::ConfigFile
        );
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
        assert_eq!(config.server.max_prefill_layers_per_cycle.layers(), 6);
        assert_eq!(
            config.server.max_prefill_layers_per_cycle.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(config.server.max_decode_batch.limit(), Some(24));
        assert_eq!(
            config.server.max_decode_batch.source(),
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
        assert!(config.batching.rowwise_decode.enabled());
        assert_eq!(
            config.batching.rowwise_decode.source(),
            ConfigValueSource::ConfigFile
        );
        assert!(!config.batching.prefix_aware_admission.enabled());
        assert_eq!(
            config.batching.prefix_aware_admission.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(
            config.batching.prefill_admission_quantum.configured(),
            Some(12)
        );
        assert_eq!(
            config.batching.prefill_admission_quantum.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(config.model.path.as_deref(), Some("/models/qwen"));
        assert_eq!(config.model.model_id, "custom/model");
        assert_eq!(config.memory.num_blocks, Some(128));
        assert_eq!(config.memory.gpu_memory_gb, Some(24.0));
        assert_eq!(config.memory.inference_memory_fraction, 0.5);
        assert_eq!(config.memory.training_memory_gb, Some(6.0));
        assert_eq!(config.memory.vulkan_buffer_pool_gb, 1.5);
        assert_eq!(config.memory.floor_gb, 2.0);
        assert_eq!(config.memory.probe_ms, 250);
        assert_eq!(
            config.memory.reclaim_mode.mode(),
            kiln_memory::MemoryReclaimMode::Automatic
        );
        assert_eq!(
            config.memory.reclaim_mode.source(),
            ConfigValueSource::ConfigFile
        );
        assert!(!config.memory.kv_autoscale.enabled());
        assert_eq!(
            config.memory.kv_autoscale.source(),
            ConfigValueSource::ConfigFile
        );
        assert_eq!(config.memory.kv_force_blocks.target(), None);
        assert_eq!(
            config.memory.kv_force_blocks.source(),
            ConfigValueSource::ConfigFile
        );
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs);
        assert_eq!(config.memory.cuda_graph_cache_entries, 16);
        assert_eq!(config.training.grad_checkpoint_segments, Some(8));
        assert_eq!(
            config.training.recompute_checkpoint_boundaries.mode(),
            kiln_train::CheckpointBoundaryRecomputeMode::Disabled
        );
        assert_eq!(
            config.training.recompute_boundary_threshold_tokens.tokens(),
            4096
        );
        assert_eq!(
            config
                .training
                .checkpoint_boundary_anchor_stride
                .configured(),
            Some(4)
        );
        assert_eq!(config.training.checkpoint_boundary_cache_gb.gib(), 2.5);
        assert_eq!(
            config.training.checkpoint_boundary_policy().unwrap(),
            kiln_train::CheckpointBoundaryPolicy::from_parts(
                kiln_train::CheckpointBoundaryRecomputeMode::Disabled,
                4096,
                Some(4),
                2_684_354_560,
            )
            .unwrap()
        );
        for source in [
            config.training.recompute_checkpoint_boundaries.source(),
            config.training.recompute_boundary_threshold_tokens.source(),
            config.training.checkpoint_boundary_anchor_stride.source(),
            config.training.checkpoint_boundary_cache_gb.source(),
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }
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
        assert_eq!(config.speculative.method, SpecMethod::SkipLayer);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);
        assert_eq!(
            config.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Enabled
        );
        assert_eq!(
            config.streaming_prefill.threshold_tokens.configured(),
            Some(1024)
        );
        assert_eq!(
            config.streaming_prefill.tile_tokens.configured(),
            Some(4096)
        );
        assert_eq!(
            config.streaming_prefill.tape_tile_tokens.configured(),
            Some(2048)
        );
        assert_eq!(
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
            Some(512)
        );
        assert!(!config.streaming_prefill.last_token_lm_head.enabled());
        for source in [
            config.streaming_prefill.mode.source(),
            config.streaming_prefill.threshold_tokens.source(),
            config.streaming_prefill.tile_tokens.source(),
            config.streaming_prefill.tape_tile_tokens.source(),
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
            config.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }
        assert_eq!(config.adapters.max_disk_bytes, Some(5_368_709_120));
        assert_eq!(
            config.adapters.composed_cache_max_bytes,
            Some(1_073_741_824)
        );
        assert_eq!(config.adapters.composed_cache_max_entries, Some(8));
    }

    #[test]
    fn checkpoint_boundary_toml_is_strict_source_tracked_and_resolves_once() {
        for (raw, expected) in [
            ("auto", kiln_train::CheckpointBoundaryRecomputeMode::Auto),
            (
                "enabled",
                kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
            ),
            (
                "disabled",
                kiln_train::CheckpointBoundaryRecomputeMode::Disabled,
            ),
        ] {
            let config: KilnConfig = toml::from_str(&format!(
                "[training]\nrecompute_checkpoint_boundaries = {raw:?}\n"
            ))
            .unwrap();
            assert_eq!(
                config.training.recompute_checkpoint_boundaries.mode(),
                expected
            );
            assert_eq!(
                config.training.recompute_checkpoint_boundaries.source(),
                ConfigValueSource::ConfigFile
            );
        }

        let config: KilnConfig = toml::from_str(
            r#"
[training]
recompute_checkpoint_boundaries = "enabled"
recompute_boundary_threshold_tokens = 16384
checkpoint_boundary_anchor_stride = "auto"
checkpoint_boundary_cache_gb = 0.5
"#,
        )
        .unwrap();
        let diagnostics = config.training.checkpoint_boundary_diagnostics();
        assert_eq!(
            diagnostics.recompute_checkpoint_boundaries,
            kiln_train::CheckpointBoundaryRecomputeMode::Enabled
        );
        assert_eq!(diagnostics.recompute_boundary_threshold_tokens, 16_384);
        assert_eq!(diagnostics.checkpoint_boundary_anchor_stride, None);
        assert_eq!(diagnostics.checkpoint_boundary_cache_gb, 0.5);
        assert_eq!(diagnostics.checkpoint_boundary_cache_bytes, 536_870_912);
        for source in [
            diagnostics.recompute_checkpoint_boundaries_source,
            diagnostics.recompute_boundary_threshold_tokens_source,
            diagnostics.checkpoint_boundary_anchor_stride_source,
            diagnostics.checkpoint_boundary_cache_gb_source,
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }
        assert_eq!(
            config.training.checkpoint_boundary_policy().unwrap(),
            kiln_train::CheckpointBoundaryPolicy::from_parts(
                kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
                16_384,
                None,
                536_870_912,
            )
            .unwrap()
        );
        assert_eq!(
            serde_json::to_value(&config.training).unwrap()["checkpoint_boundary_anchor_stride"],
            serde_json::json!("auto")
        );

        for (field, input) in [
            (
                "recompute_checkpoint_boundaries",
                "[training]\nrecompute_checkpoint_boundaries = 'sometimes'",
            ),
            (
                "recompute_boundary_threshold_tokens",
                "[training]\nrecompute_boundary_threshold_tokens = 0",
            ),
            (
                "checkpoint_boundary_anchor_stride",
                "[training]\ncheckpoint_boundary_anchor_stride = 0",
            ),
            (
                "checkpoint_boundary_anchor_stride",
                "[training]\ncheckpoint_boundary_anchor_stride = 'sometimes'",
            ),
            (
                "checkpoint_boundary_cache_gb",
                "[training]\ncheckpoint_boundary_cache_gb = 0.0",
            ),
            (
                "checkpoint_boundary_cache_gb",
                "[training]\ncheckpoint_boundary_cache_gb = nan",
            ),
            (
                "checkpoint_boundary_cache_gb",
                "[training]\ncheckpoint_boundary_cache_gb = 1e-12",
            ),
            (
                "checkpoint_boundary_cache_gb",
                "[training]\ncheckpoint_boundary_cache_gb = 2e10",
            ),
        ] {
            let error = toml::from_str::<KilnConfig>(input).unwrap_err().to_string();
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn checkpoint_boundary_environment_overrides_toml_with_environment_sources() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, value) in [
            ("KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES", "enabled"),
            ("KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS", "4096"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE", "5"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB", "1.25"),
        ] {
            environment.set(name, value);
        }

        let mut config: KilnConfig = toml::from_str(
            r#"
[training]
recompute_checkpoint_boundaries = "disabled"
recompute_boundary_threshold_tokens = 2048
checkpoint_boundary_anchor_stride = "auto"
checkpoint_boundary_cache_gb = 6.0
"#,
        )
        .unwrap();
        config.apply_env_overrides().unwrap();
        assert_eq!(
            config.training.checkpoint_boundary_policy().unwrap(),
            kiln_train::CheckpointBoundaryPolicy::from_parts(
                kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
                4096,
                Some(5),
                1_342_177_280,
            )
            .unwrap()
        );
        let diagnostics = config.training.checkpoint_boundary_diagnostics();
        for source in [
            diagnostics.recompute_checkpoint_boundaries_source,
            diagnostics.recompute_boundary_threshold_tokens_source,
            diagnostics.checkpoint_boundary_anchor_stride_source,
            diagnostics.checkpoint_boundary_cache_gb_source,
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }
    }

    #[test]
    fn checkpoint_boundary_malformed_environment_values_fail_closed() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, invalid) in [
            ("KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES", "on"),
            (
                "KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS",
                "not-a-number",
            ),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE", "0"),
            ("KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB", "1e-12"),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains(invalid), "{name}: {detail}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_boundary_non_unicode_environment_inputs_are_fatal() {
        use std::os::unix::ffi::OsStringExt;

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for name in [
            "KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES",
            "KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS",
            "KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE",
            "KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB",
        ] {
            let invalid = OsString::from_vec(vec![b'1', 0xff]);
            environment.set_os(name, &invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains("UTF-8"), "{name}: {detail}");
        }
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
        assert_eq!(
            config.server.max_prefill_layers_per_cycle,
            PrefillLayerBudget::default()
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
    fn test_validation_rejects_invalid_memory_governor_tuning() {
        let mut negative_floor = KilnConfig::default();
        negative_floor.memory.floor_gb = -0.1;
        assert!(negative_floor.validate().is_err());

        let mut non_finite_floor = KilnConfig::default();
        non_finite_floor.memory.floor_gb = f64::INFINITY;
        assert!(non_finite_floor.validate().is_err());

        let mut negative_vulkan_pool = KilnConfig::default();
        negative_vulkan_pool.memory.vulkan_buffer_pool_gb = -0.1;
        assert!(negative_vulkan_pool.validate().is_err());

        let mut non_finite_vulkan_pool = KilnConfig::default();
        non_finite_vulkan_pool.memory.vulkan_buffer_pool_gb = f64::INFINITY;
        assert!(non_finite_vulkan_pool.validate().is_err());

        let mut zero_probe = KilnConfig::default();
        zero_probe.memory.probe_ms = 0;
        assert!(zero_probe.validate().is_err());

        for invalid in [
            0,
            kiln_model::CudaGraphExecutionPolicy::MAX_CACHED_GRAPHS + 1,
        ] {
            let mut config = KilnConfig::default();
            config.memory.cuda_graph_cache_entries = invalid;
            let detail = config.validate().unwrap_err().to_string();
            assert!(detail.contains("memory.cuda_graph_cache_entries"));
            assert!(detail.contains(&invalid.to_string()));
        }

        let mut unrepresentable_capacity = KilnConfig::default();
        unrepresentable_capacity.memory.gpu_memory_gb = Some(f64::MAX);
        assert!(unrepresentable_capacity.validate().is_err());

        let mut force_without_autoscale = KilnConfig::default();
        force_without_autoscale.memory.kv_autoscale =
            KvAutoscaleSetting::new(false, ConfigValueSource::ConfigFile);
        force_without_autoscale.memory.kv_force_blocks =
            KvForceBlocksSetting::new(1, ConfigValueSource::ConfigFile);
        assert!(
            force_without_autoscale
                .validate()
                .unwrap_err()
                .to_string()
                .contains("memory.kv_autoscale=true")
        );

        let mut force_outside_maintenance = KilnConfig::default();
        force_outside_maintenance.memory.kv_force_blocks =
            KvForceBlocksSetting::new(1, ConfigValueSource::ConfigFile);
        assert!(
            force_outside_maintenance
                .validate()
                .unwrap_err()
                .to_string()
                .contains("server.serving_profile=maintenance")
        );

        let mut maintenance_force = KilnConfig::default();
        maintenance_force.server.serving_profile =
            ServingProfileSetting::new(ServingProfile::Maintenance, ConfigValueSource::ConfigFile);
        maintenance_force.memory.kv_force_blocks =
            KvForceBlocksSetting::new(1, ConfigValueSource::ConfigFile);
        maintenance_force.validate().unwrap();
    }

    #[test]
    fn test_memory_reclaim_mode_rejects_invalid_toml_value() {
        let error = toml::from_str::<KilnConfig>(
            r#"
[memory]
reclaim_mode = "whenever"
"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("memory.reclaim_mode"));
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
    fn streaming_prefill_toml_is_strict_and_source_tracked() {
        for mode in ["auto", "enabled", "disabled"] {
            let config: KilnConfig =
                toml::from_str(&format!("[streaming_prefill]\nmode = {mode:?}\n")).unwrap();
            assert_eq!(config.streaming_prefill.mode.mode().as_str(), mode);
            assert_eq!(
                config.streaming_prefill.mode.source(),
                ConfigValueSource::ConfigFile
            );
        }

        let explicit: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
threshold_tokens = 1
tile_tokens = 64
tape_tile_tokens = 128
detached_full_attn_tile_tokens = 192
last_token_lm_head = false
"#,
        )
        .unwrap();
        assert_eq!(
            explicit.streaming_prefill.threshold_tokens.configured(),
            Some(1)
        );
        assert_eq!(
            explicit.streaming_prefill.tile_tokens.configured(),
            Some(64)
        );
        assert_eq!(
            explicit.streaming_prefill.tape_tile_tokens.configured(),
            Some(128)
        );
        assert_eq!(
            explicit
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
            Some(192)
        );
        assert!(!explicit.streaming_prefill.last_token_lm_head.enabled());
        for source in [
            explicit.streaming_prefill.threshold_tokens.source(),
            explicit.streaming_prefill.tile_tokens.source(),
            explicit.streaming_prefill.tape_tile_tokens.source(),
            explicit
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
            explicit.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }

        let automatic: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
threshold_tokens = "auto"
tile_tokens = "auto"
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
"#,
        )
        .unwrap();
        for configured in [
            automatic.streaming_prefill.threshold_tokens.configured(),
            automatic.streaming_prefill.tile_tokens.configured(),
            automatic.streaming_prefill.tape_tile_tokens.configured(),
            automatic
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
        ] {
            assert_eq!(configured, None);
        }
        for source in [
            automatic.streaming_prefill.threshold_tokens.source(),
            automatic.streaming_prefill.tile_tokens.source(),
            automatic.streaming_prefill.tape_tile_tokens.source(),
            automatic
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
        ] {
            assert_eq!(source, ConfigValueSource::ConfigFile);
        }

        for document in [
            "[streaming_prefill]\nmode = 'sometimes'\n",
            "[streaming_prefill]\nmode = true\n",
            "[streaming_prefill]\nenabled = true\n",
            "[streaming_prefill]\nthreshold_tokens = 0\n",
            "[streaming_prefill]\nthreshold_tokens = 'many'\n",
            "[streaming_prefill]\ntile_tokens = 0\n",
            "[streaming_prefill]\ntile_tokens = 63\n",
            "[streaming_prefill]\ntape_tile_tokens = 65\n",
            "[streaming_prefill]\ndetached_full_attn_tile_tokens = 127\n",
            "[streaming_prefill]\nlast_token_lm_head = 'true'\n",
        ] {
            let error = toml::from_str::<KilnConfig>(document).unwrap_err();
            let detail = error.to_string();
            assert!(
                detail.contains("streaming_prefill")
                    || detail.contains("unknown field")
                    || detail.contains("invalid type")
                    || detail.contains("data did not match"),
                "unexpected error for {document:?}: {error:#}"
            );
        }
    }

    #[test]
    fn streaming_prefill_runtime_resolution_preserves_backend_policy_and_provenance() {
        let config: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
mode = "auto"
threshold_tokens = 1024
tile_tokens = 256
last_token_lm_head = false
"#,
        )
        .unwrap();
        let backend = kiln_model::StreamingPrefillBackendPolicy::for_backend(
            "cuda",
            kiln_tensor::Device::Cuda(0),
        );
        let runtime = config.streaming_prefill.resolve(backend);

        assert_eq!(runtime.dispatch.configured_mode, StreamingPrefillMode::Auto);
        assert_eq!(
            runtime.dispatch.backend_policy,
            StreamingPrefillDispatchRuleDiagnostics {
                policy: StreamingPrefillDispatchPolicy::PromptTokensAtLeast,
                minimum_prompt_tokens: Some(2048),
            }
        );
        assert_eq!(
            runtime.dispatch.effective,
            StreamingPrefillDispatchRuleDiagnostics {
                policy: StreamingPrefillDispatchPolicy::PromptTokensAtLeast,
                minimum_prompt_tokens: Some(1024),
            }
        );
        assert_eq!(
            runtime.dispatch.effective_source,
            StreamingPrefillEffectiveSource::ConfigFile
        );
        assert!(
            runtime
                .threshold_tokens
                .override_applied_to_backend_auto_policy
        );
        assert_eq!(runtime.tile_tokens.effective, 256);
        assert_eq!(runtime.tape_tile_tokens.effective, 256);
        assert_eq!(runtime.detached_full_attn_tile_tokens.effective, 256);
        assert_eq!(
            runtime.detached_full_attn_boundary_tile_tokens.effective,
            256
        );
        assert_eq!(
            runtime.detached_full_attn_tape_replay_tile_tokens.effective,
            256
        );
        for source in [
            runtime.tape_tile_tokens.effective_source,
            runtime.detached_full_attn_tile_tokens.effective_source,
            runtime
                .detached_full_attn_boundary_tile_tokens
                .effective_source,
            runtime
                .detached_full_attn_tape_replay_tile_tokens
                .effective_source,
        ] {
            assert_eq!(
                source,
                StreamingPrefillEffectiveSource::InheritedFromTileTokensConfigFile
            );
        }
        assert!(!runtime.last_token_lm_head.effective);
        assert!(!runtime.execution_policy().enabled_for(1023));
        assert!(runtime.execution_policy().enabled_for(1024));
        assert!(runtime.immutable_after_startup);
        assert!(runtime.restart_required_to_change);

        let json = serde_json::to_value(runtime).unwrap();
        assert_eq!(
            json["dispatch"]["effective"]["policy"],
            "prompt_tokens_at_least"
        );
        assert_eq!(
            json["tape_tile_tokens"]["effective_source"],
            "inherited_from_tile_tokens_config_file"
        );
        assert!(json.get("execution_policy").is_none());
    }

    #[test]
    fn streaming_prefill_runtime_resolution_reports_ignored_auto_thresholds() {
        let config: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
threshold_tokens = 1
"#,
        )
        .unwrap();
        let runtime = config.streaming_prefill.resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_backend("cpu", kiln_tensor::Device::Cpu),
        );

        assert_eq!(
            runtime.dispatch.effective.policy,
            StreamingPrefillDispatchPolicy::Never
        );
        assert_eq!(
            runtime.dispatch.effective_source,
            StreamingPrefillEffectiveSource::BackendPolicy
        );
        assert_eq!(runtime.threshold_tokens.configured, Some(1));
        assert_eq!(runtime.threshold_tokens.backend_policy, None);
        assert_eq!(runtime.threshold_tokens.effective_for_auto_mode, None);
        assert!(
            !runtime
                .threshold_tokens
                .override_applied_to_backend_auto_policy
        );
        assert!(!runtime.execution_policy().enabled_for(usize::MAX));
    }

    #[test]
    fn streaming_prefill_forced_mode_overrides_a_backend_never_rule() {
        let config: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
mode = "enabled"
"#,
        )
        .unwrap();
        let runtime = config.streaming_prefill.resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_backend(
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
            ),
        );

        assert_eq!(
            runtime.dispatch.backend_policy.policy,
            StreamingPrefillDispatchPolicy::Never
        );
        assert_eq!(
            runtime.dispatch.effective.policy,
            StreamingPrefillDispatchPolicy::AllNonEmpty
        );
        assert_eq!(
            runtime.dispatch.effective_source,
            StreamingPrefillEffectiveSource::ConfigFile
        );
        assert!(!runtime.execution_policy().enabled_for(0));
        assert!(runtime.execution_policy().enabled_for(1));
    }

    #[test]
    fn rocm_actor_prefill_contract_accepts_only_route_invariant_chunking() {
        let decode =
            kiln_model::DecodeExecutionPolicy::for_backend("rocm", kiln_tensor::Device::Rocm(0));
        let backend_policy = BatchingBackendPolicy::from_decode_execution_policy(decode);
        let batching = BatchingConfig::default().resolve(backend_policy, 8);
        let streaming = StreamingPrefillConfig::default().resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_backend(
                "rocm",
                kiln_tensor::Device::Rocm(0),
            ),
        );
        let max_batch = BatchTokenBudget::new(512, ConfigValueSource::Default).unwrap();
        let aligned_prefill = PrefillTokenBudget::new(256, ConfigValueSource::Default).unwrap();

        validate_actor_prefill_tile_contract(batching, streaming, max_batch, aligned_prefill, 8)
            .unwrap();
        assert!(batching.actor_prefill_tile_alignment_required);
        assert_eq!(streaming.tile_tokens.effective, 256);
        assert_eq!(
            streaming.threshold_tokens.effective_for_auto_mode,
            Some(256)
        );

        let error = validate_actor_prefill_tile_contract(
            batching,
            streaming,
            max_batch,
            PrefillTokenBudget::new(64, ConfigValueSource::ConfigFile).unwrap(),
            8,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("server.max_prefill_tokens_per_cycle=64 must equal")
        );

        let delayed_streaming = StreamingPrefillConfig {
            threshold_tokens: StreamingPrefillThresholdTokens::new(
                Some(512),
                ConfigValueSource::ConfigFile,
            )
            .unwrap(),
            ..StreamingPrefillConfig::default()
        }
        .resolve(kiln_model::StreamingPrefillBackendPolicy::for_backend(
            "rocm",
            kiln_tensor::Device::Rocm(0),
        ));
        let error = validate_actor_prefill_tile_contract(
            batching,
            delayed_streaming,
            max_batch,
            aligned_prefill,
            8,
        )
        .unwrap_err();
        assert!(error.to_string().contains("enable tiled streaming prefill"));

        let narrow_batch = BatchTokenBudget::new(263, ConfigValueSource::ConfigFile).unwrap();
        let error = validate_actor_prefill_tile_contract(
            batching,
            streaming,
            narrow_batch,
            aligned_prefill,
            8,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("server.max_batch_tokens=263 must be at least 264")
        );
    }

    #[test]
    fn actor_prefill_contract_is_inert_when_backend_requirement_is_off() {
        let batching = BatchingConfig::default().resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
            },
            8,
        );
        let never_streaming = StreamingPrefillConfig::default().resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_backend("cpu", kiln_tensor::Device::Cpu),
        );
        validate_actor_prefill_tile_contract(
            batching,
            never_streaming,
            BatchTokenBudget::new(2, ConfigValueSource::ConfigFile).unwrap(),
            PrefillTokenBudget::new(1, ConfigValueSource::ConfigFile).unwrap(),
            8,
        )
        .unwrap();
    }

    #[test]
    fn streaming_prefill_environment_overrides_toml_with_environment_provenance() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, value) in [
            ("KILN_STREAMING_PREFILL_MODE", "enabled"),
            ("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS", "1024"),
            ("KILN_STREAMING_PREFILL_TILE_TOKENS", "2048"),
            ("KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS", "4096"),
            (
                "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
                "512",
            ),
            ("KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD", "off"),
        ] {
            environment.set(name, value);
        }

        let mut config: KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
mode = "disabled"
threshold_tokens = "auto"
tile_tokens = "auto"
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
last_token_lm_head = true
"#,
        )
        .unwrap();
        config.apply_env_overrides().unwrap();

        assert_eq!(
            config.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Enabled
        );
        assert_eq!(
            config.streaming_prefill.threshold_tokens.configured(),
            Some(1024)
        );
        assert_eq!(
            config.streaming_prefill.tile_tokens.configured(),
            Some(2048)
        );
        assert_eq!(
            config.streaming_prefill.tape_tile_tokens.configured(),
            Some(4096)
        );
        assert_eq!(
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
            Some(512)
        );
        assert!(!config.streaming_prefill.last_token_lm_head.enabled());
        for source in [
            config.streaming_prefill.mode.source(),
            config.streaming_prefill.threshold_tokens.source(),
            config.streaming_prefill.tile_tokens.source(),
            config.streaming_prefill.tape_tile_tokens.source(),
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
            config.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }

        for name in [
            "KILN_STREAMING_PREFILL_MODE",
            "KILN_STREAMING_PREFILL_THRESHOLD_TOKENS",
            "KILN_STREAMING_PREFILL_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD",
        ] {
            environment.remove(name);
        }
        environment.set("KILN_STREAMING_PREFILL_MODE", "disabled");
        let mut disabled = KilnConfig::default();
        disabled.apply_env_overrides().unwrap();
        assert_eq!(
            disabled.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Disabled
        );
        assert_eq!(
            disabled.streaming_prefill.mode.source(),
            ConfigValueSource::Environment
        );

        environment.remove("KILN_STREAMING_PREFILL_MODE");
        for name in [
            "KILN_STREAMING_PREFILL_MODE",
            "KILN_STREAMING_PREFILL_THRESHOLD_TOKENS",
            "KILN_STREAMING_PREFILL_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
        ] {
            environment.set(name, "auto");
        }
        let mut automatic = KilnConfig::default();
        automatic.apply_env_overrides().unwrap();
        assert_eq!(
            automatic.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Auto
        );
        for configured in [
            automatic.streaming_prefill.threshold_tokens.configured(),
            automatic.streaming_prefill.tile_tokens.configured(),
            automatic.streaming_prefill.tape_tile_tokens.configured(),
            automatic
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .configured(),
        ] {
            assert_eq!(configured, None);
        }
        for source in [
            automatic.streaming_prefill.mode.source(),
            automatic.streaming_prefill.threshold_tokens.source(),
            automatic.streaming_prefill.tile_tokens.source(),
            automatic.streaming_prefill.tape_tile_tokens.source(),
            automatic
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }
    }

    #[test]
    fn streaming_prefill_malformed_environment_values_fail_closed() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, invalid) in [
            ("KILN_STREAMING_PREFILL_MODE", "automatic-ish"),
            ("KILN_STREAMING_PREFILL_TILE_TOKENS", "65"),
            ("KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS", "127"),
            (
                "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
                "-1",
            ),
            ("KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD", "sometimes"),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{name}: {detail}");
            assert!(detail.contains(&format!("{invalid:?}")), "{name}: {detail}");
        }
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
                vulkan_resident_prefill: false,
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
                vulkan_resident_prefill: true,
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
                vulkan_resident_prefill: false,
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
        assert_eq!(json["effective_policy"]["vulkan_resident_prefill"], false);
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
        let environment = ScopedConfigEnvironment::isolated();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        environment.set(SERVING_PROFILE_ENV, "prod-ish");
        let error = KilnConfig::load(Some(path.to_str().unwrap())).unwrap_err();
        let detail = format!("{error:#}");
        assert!(detail.contains(SERVING_PROFILE_ENV), "{detail}");
        assert!(detail.contains("prod-ish"), "{detail}");
    }

    #[test]
    fn load_rejects_malformed_decode_runtime_environment() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        let path = path.to_str().unwrap();

        for (name, invalid) in [
            (DETERMINISTIC_ENV, "sometimes"),
            (MAX_DECODE_BATCH_ENV, "wide-ish"),
            (MAX_DECODE_BATCH_ENV, "0"),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::load(Some(path)).unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{detail}");
            assert!(detail.contains(invalid), "{detail}");
        }
    }

    #[test]
    fn thinking_budget_environment_grammar_is_strict() {
        assert_eq!(
            parse_optional_usize_env(DEFAULT_THINKING_BUDGET_TOKENS_ENV, " 0 ").unwrap(),
            Some(0)
        );
        assert_eq!(
            parse_optional_usize_env(DEFAULT_THINKING_BUDGET_TOKENS_ENV, "42").unwrap(),
            Some(42)
        );
        assert_eq!(
            parse_optional_u64_env(DEFAULT_THINKING_BUDGET_MS_ENV, " 1500 ").unwrap(),
            Some(1500)
        );
        assert_eq!(
            parse_optional_u64_env(DEFAULT_THINKING_BUDGET_MS_ENV, " UnLiMiTeD ").unwrap(),
            None
        );

        for invalid in ["", "off", "none", "null", "-1", "1.5", "12ms"] {
            for error in [
                parse_optional_usize_env(DEFAULT_THINKING_BUDGET_TOKENS_ENV, invalid).unwrap_err(),
                parse_optional_u64_env(DEFAULT_THINKING_BUDGET_MS_ENV, invalid).unwrap_err(),
            ] {
                let detail = format!("{error:#}");
                assert!(detail.contains(&format!("{invalid:?}")), "{detail}");
                assert!(
                    detail.contains(DEFAULT_THINKING_BUDGET_TOKENS_ENV)
                        || detail.contains(DEFAULT_THINKING_BUDGET_MS_ENV),
                    "{detail}"
                );
            }
        }
    }

    #[test]
    fn thinking_budget_environment_overrides_toml_and_invalid_values_fail_load() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(
            &path,
            "[server]\ndefault_thinking_budget_tokens = 64\ndefault_thinking_budget_ms = 2500\n",
        )
        .unwrap();
        let path = path.to_str().unwrap();

        environment.set(DEFAULT_THINKING_BUDGET_TOKENS_ENV, "unlimited");
        environment.set(DEFAULT_THINKING_BUDGET_MS_ENV, "0");
        let config = KilnConfig::load(Some(path)).unwrap();
        assert_eq!(config.server.default_thinking_budget_tokens, None);
        assert_eq!(config.server.default_thinking_budget_ms, Some(0));

        environment.remove(DEFAULT_THINKING_BUDGET_TOKENS_ENV);
        environment.remove(DEFAULT_THINKING_BUDGET_MS_ENV);
        for (name, invalid) in [
            (DEFAULT_THINKING_BUDGET_TOKENS_ENV, "64 tokens"),
            (DEFAULT_THINKING_BUDGET_MS_ENV, "2.5"),
        ] {
            environment.set(name, invalid);
            let error = KilnConfig::load(Some(path)).unwrap_err();
            environment.remove(name);
            let detail = format!("{error:#}");
            assert!(detail.contains(name), "{detail}");
            assert!(detail.contains(&format!("{invalid:?}")), "{detail}");
        }
    }

    #[test]
    fn test_http_send_buffer_env_override_is_strict() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kiln.toml");
        std::fs::write(&path, "").unwrap();
        let path = path.to_str().unwrap();

        let config = KilnConfig::load(Some(path)).unwrap();
        assert_eq!(config.server.http_send_buffer_bytes, None);

        environment.set("KILN_SERVER_HTTP_SEND_BUFFER_BYTES", "4096");
        let config = KilnConfig::load(Some(path)).unwrap();
        assert_eq!(config.server.http_send_buffer_bytes, Some(4096));

        for invalid in ["", "0", "1023", "16777217", "not-a-number"] {
            environment.set("KILN_SERVER_HTTP_SEND_BUFFER_BYTES", invalid);
            let error = KilnConfig::load(Some(path)).unwrap_err();
            assert!(
                format!("{error:#}").contains("HTTP_SEND_BUFFER"),
                "unexpected error for {invalid:?}: {error:#}"
            );
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
                format!("{error:#}").contains("KILN_SERVER_STREAM_STALL_GRACE_MS"),
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
                format!("{error:#}").contains("KILN_SERVER_MAX_BATCH_TOKENS"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
    }

    #[test]
    fn deterministic_inference_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
deterministic = true
"#,
        )
        .unwrap();
        assert!(config.server.deterministic.enabled());
        assert_eq!(
            config.server.deterministic.source(),
            ConfigValueSource::ConfigFile
        );

        config.apply_deterministic_env_value(Some(" off ")).unwrap();
        assert!(!config.server.deterministic.enabled());
        assert_eq!(
            config.server.deterministic.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "2", "truthy", "false-ish"] {
            let error = DeterministicInference::from_environment_value(invalid).unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains(DETERMINISTIC_ENV) && message.contains(&format!("{invalid:?}")),
                "unexpected error for {invalid:?}: {message}"
            );
        }
    }

    #[test]
    fn max_decode_batch_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
max_decode_batch = 24
"#,
        )
        .unwrap();
        assert_eq!(config.server.max_decode_batch.limit(), Some(24));
        assert_eq!(
            config.server.max_decode_batch.source(),
            ConfigValueSource::ConfigFile
        );

        config
            .apply_max_decode_batch_env_value(Some(" 12 "))
            .unwrap();
        assert_eq!(config.server.max_decode_batch.limit(), Some(12));
        assert_eq!(
            config.server.max_decode_batch.source(),
            ConfigValueSource::Environment
        );

        config
            .apply_max_decode_batch_env_value(Some(" auto "))
            .unwrap();
        assert_eq!(config.server.max_decode_batch.limit(), None);
        assert_eq!(
            config.server.max_decode_batch.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "0", "65537", "-1", "not-a-number"] {
            let error = MaxDecodeBatch::from_environment_value(invalid).unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains(MAX_DECODE_BATCH_ENV) && message.contains(&format!("{invalid:?}")),
                "unexpected error for {invalid:?}: {message}"
            );
        }
    }

    #[test]
    fn max_decode_batch_toml_supports_auto_and_validates_bounds() {
        for mode in ["auto", "backend", "backend_policy"] {
            let config: KilnConfig =
                toml::from_str(&format!("[server]\nmax_decode_batch = {mode:?}\n")).unwrap();
            assert_eq!(config.server.max_decode_batch.limit(), None);
            assert_eq!(
                config.server.max_decode_batch.source(),
                ConfigValueSource::ConfigFile
            );
        }

        for valid in [MAX_DECODE_BATCH_MIN, MAX_DECODE_BATCH_MAX] {
            let config: KilnConfig =
                toml::from_str(&format!("[server]\nmax_decode_batch = {valid}\n")).unwrap();
            assert_eq!(config.server.max_decode_batch.limit(), Some(valid));
        }

        for invalid in [0, MAX_DECODE_BATCH_MAX + 1] {
            let error =
                toml::from_str::<KilnConfig>(&format!("[server]\nmax_decode_batch = {invalid}\n"))
                    .unwrap_err();
            assert!(
                error.to_string().contains("server.max_decode_batch"),
                "unexpected error for {invalid}: {error:#}"
            );
        }

        let error =
            toml::from_str::<KilnConfig>("[server]\nmax_decode_batch = \"unbounded-ish\"\n")
                .unwrap_err();
        assert!(error.to_string().contains("server.max_decode_batch"));
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
                format!("{error:#}").contains("KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE"),
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
    fn test_max_prefill_layers_env_override_is_strict_and_source_tracked() {
        let mut config: KilnConfig = toml::from_str(
            r#"
[server]
max_prefill_layers_per_cycle = 8
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.max_prefill_layers_per_cycle.source(),
            ConfigValueSource::ConfigFile
        );

        config
            .apply_max_prefill_layers_per_cycle_env_value(Some(" 4 "))
            .unwrap();
        assert_eq!(config.server.max_prefill_layers_per_cycle.layers(), 4);
        assert_eq!(
            config.server.max_prefill_layers_per_cycle.source(),
            ConfigValueSource::Environment
        );

        for invalid in ["", "0", "1025", "-1", "not-a-number"] {
            let error = PrefillLayerBudget::from_environment_value(invalid).unwrap_err();
            assert!(
                format!("{error:#}").contains("KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE"),
                "unexpected error for {invalid:?}: {error:#}"
            );
        }
    }

    #[test]
    fn test_max_prefill_layers_toml_validation_bounds() {
        for valid in [
            MAX_PREFILL_LAYERS_PER_CYCLE_MIN,
            MAX_PREFILL_LAYERS_PER_CYCLE_MAX,
        ] {
            let config: KilnConfig = toml::from_str(&format!(
                "[server]\nmax_prefill_layers_per_cycle = {valid}\n"
            ))
            .unwrap();
            assert_eq!(config.server.max_prefill_layers_per_cycle.layers(), valid);
            assert_eq!(
                config.server.max_prefill_layers_per_cycle.source(),
                ConfigValueSource::ConfigFile
            );
        }

        for invalid in [
            MAX_PREFILL_LAYERS_PER_CYCLE_MIN - 1,
            MAX_PREFILL_LAYERS_PER_CYCLE_MAX + 1,
        ] {
            let error = toml::from_str::<KilnConfig>(&format!(
                "[server]\nmax_prefill_layers_per_cycle = {invalid}\n"
            ))
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("server.max_prefill_layers_per_cycle"),
                "unexpected error for {invalid}: {error:#}"
            );
        }
    }

    #[test]
    fn test_env_var_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        for (name, value) in [
            ("KILN_SERVER_HOST", "10.0.0.1"),
            ("KILN_SERVER_PORT", "7777"),
            ("KILN_SERVER_EVAL_MODE", "true"),
            ("KILN_SERVER_DEFAULT_THINKING_ENABLED", "false"),
            ("KILN_SERVER_FOLD_REASONING_INTO_CONTENT", "true"),
            ("KILN_SERVER_CHAT_PERFORMANCE_METADATA", "true"),
            ("KILN_SERVER_CHAT_CONFIG_HASH_METADATA", "true"),
            ("KILN_SERVER_SLOW_REQUEST_WARN_SECS", "12"),
            ("KILN_MODEL_PATH", "/tmp/model"),
            ("KILN_MEMORY_INFERENCE_MEMORY_FRACTION", "0.9"),
            ("KILN_LOGGING_LEVEL", "debug"),
            ("KILN_TRAINING_NO_GRAD_CHECKPOINT", "1"),
            ("KILN_TRAINING_CHECKPOINT_INTERVAL", "25"),
            ("KILN_TRAINING_WEBHOOK_URL", "https://hook.example/notify"),
            ("KILN_TRAINING_MAX_QUEUED_JOBS", "7"),
            ("KILN_TRAINING_MAX_TRACKED_JOBS", "9"),
            ("KILN_TRAINING_TRACKED_JOB_TTL_SECS", "11"),
            ("KILN_MEMORY_KV_CACHE_FP8", "1"),
            ("KILN_MEMORY_CUDA_GRAPHS", "false"),
            ("KILN_PREFIX_CACHE_ENABLED", "false"),
            ("KILN_PREFIX_CACHE_MAX_BLOCKS", "128"),
            ("KILN_SPECULATIVE_METHOD", "skip_layer"),
            ("KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS", "6"),
            ("KILN_SPECULATIVE_DRAFT_LAYERS", "10"),
            ("KILN_STREAMING_PREFILL_MODE", "enabled"),
            ("KILN_STREAMING_PREFILL_TILE_TOKENS", "2048"),
            ("KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD", "0"),
        ] {
            environment.set(name, value);
        }

        let mut config = KilnConfig::default();
        config.apply_env_overrides().unwrap();

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
        assert!(!config.memory.cuda_graphs); // env sets KILN_MEMORY_CUDA_GRAPHS=false -> override wins
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(128));
        assert!(config.prefix_cache.max_entries.is_none());
        assert_eq!(config.speculative.method, SpecMethod::SkipLayer);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);
        assert_eq!(
            config.streaming_prefill.mode.mode(),
            StreamingPrefillMode::Enabled
        );
        assert_eq!(
            config.streaming_prefill.tile_tokens.configured(),
            Some(2048)
        );
        assert!(!config.streaming_prefill.last_token_lm_head.enabled());
        for source in [
            config.streaming_prefill.mode.source(),
            config.streaming_prefill.tile_tokens.source(),
            config.streaming_prefill.last_token_lm_head.source(),
        ] {
            assert_eq!(source, ConfigValueSource::Environment);
        }
    }

    #[test]
    fn test_adapters_max_disk_bytes_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let mut config = KilnConfig::default();
        // Default is 100 GiB.
        assert_eq!(config.adapters.max_disk_bytes, Some(100 * 1024u64.pow(3)));

        environment.set("KILN_ADAPTERS_MAX_DISK_BYTES", "1073741824");
        config.apply_env_overrides().unwrap();
        assert_eq!(config.adapters.max_disk_bytes, Some(1_073_741_824));

        // `0` disables the cap (operator-opt-out shorthand).
        environment.set("KILN_ADAPTERS_MAX_DISK_BYTES", "0");
        config.apply_env_overrides().unwrap();
        assert!(config.adapters.max_disk_bytes.is_none());

        // Empty string also clears the cap.
        environment.set("KILN_ADAPTERS_MAX_DISK_BYTES", "");
        config.adapters.max_disk_bytes = Some(123);
        config.apply_env_overrides().unwrap();
        assert!(config.adapters.max_disk_bytes.is_none());
    }

    #[test]
    fn test_adapters_composed_cache_max_bytes_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let mut config = KilnConfig::default();
        // Default is 10 GiB.
        assert_eq!(
            config.adapters.composed_cache_max_bytes,
            Some(10 * 1024u64.pow(3))
        );

        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "536870912");
        config.apply_env_overrides().unwrap();
        assert_eq!(config.adapters.composed_cache_max_bytes, Some(536_870_912));
    }

    #[test]
    fn test_adapters_composed_cache_max_entries_env_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let mut config = KilnConfig::default();
        // Default is 64.
        assert_eq!(config.adapters.composed_cache_max_entries, Some(64));

        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "12");
        config.apply_env_overrides().unwrap();
        assert_eq!(config.adapters.composed_cache_max_entries, Some(12));
    }

    #[test]
    fn test_adapters_composed_cache_zero_disables() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let mut config = KilnConfig::default();
        assert!(config.adapters.composed_cache_max_bytes.is_some());
        assert!(config.adapters.composed_cache_max_entries.is_some());

        // `0` is the operator-opt-out shorthand for both caps.
        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "0");
        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "0");
        config.apply_env_overrides().unwrap();
        assert!(config.adapters.composed_cache_max_bytes.is_none());
        assert!(config.adapters.composed_cache_max_entries.is_none());

        // Empty string also clears.
        config.adapters.composed_cache_max_bytes = Some(123);
        config.adapters.composed_cache_max_entries = Some(7);
        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES", "");
        environment.set("KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES", "");
        config.apply_env_overrides().unwrap();
        assert!(config.adapters.composed_cache_max_bytes.is_none());
        assert!(config.adapters.composed_cache_max_entries.is_none());
    }

    #[test]
    fn test_training_webhook_env_empty_string_clears_toml_value() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let toml_str = r#"
[training]
webhook_url = "https://from-toml.example/hook"
"#;
        let mut config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.training.webhook_url.as_deref(),
            Some("https://from-toml.example/hook")
        );

        environment.set("KILN_TRAINING_WEBHOOK_URL", "");
        config.apply_env_overrides().unwrap();
        assert!(
            config.training.webhook_url.is_none(),
            "empty env var should clear the TOML-set webhook URL"
        );
    }

    #[test]
    fn test_load_missing_file_returns_defaults() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _environment = ScopedConfigEnvironment::isolated();
        // Load from a path that doesn't exist via the CWD fallback (kiln.toml won't exist in test dir)
        let config = KilnConfig::load(None).unwrap();
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn test_load_explicit_path() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _environment = ScopedConfigEnvironment::isolated();
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
        assert_eq!(config.checkpoint_read_mib_per_second, None);
        assert_eq!(config.accelerator_weight_upload_mib_per_second, None);
        assert!(config.vulkan_decode_weight_prewarm);
        assert_eq!(config.vulkan_decode_weight_prewarm_mib_per_second, 256);
    }

    #[test]
    fn accelerator_weight_upload_rate_is_optional_and_bounded() {
        for invalid in [0, 16_385] {
            let mut config = KilnConfig::default();
            config.model.accelerator_weight_upload_mib_per_second = Some(invalid);
            let error = config.validate().unwrap_err().to_string();
            assert!(
                error.contains("model.accelerator_weight_upload_mib_per_second"),
                "{error}"
            );
        }

        let mut config = KilnConfig::default();
        config.model.accelerator_weight_upload_mib_per_second = Some(256);
        config.validate().unwrap();
    }

    #[test]
    fn checkpoint_read_rate_is_optional_and_bounded() {
        for invalid in [0, 16_385] {
            let mut config = KilnConfig::default();
            config.model.checkpoint_read_mib_per_second = Some(invalid);
            let error = config.validate().unwrap_err().to_string();
            assert!(
                error.contains("model.checkpoint_read_mib_per_second"),
                "{error}"
            );
        }

        let mut config = KilnConfig::default();
        config.model.checkpoint_read_mib_per_second = Some(256);
        config.validate().unwrap();
    }

    #[test]
    fn vulkan_decode_weight_prewarm_rate_is_bounded() {
        for invalid in [0, 16_385] {
            let mut config = KilnConfig::default();
            config.model.vulkan_decode_weight_prewarm_mib_per_second = invalid;
            let error = config.validate().unwrap_err().to_string();
            assert!(
                error.contains("model.vulkan_decode_weight_prewarm_mib_per_second"),
                "{error}"
            );
        }
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
        let environment = ScopedConfigEnvironment::isolated();
        let toml_str = r#"
[model]
served_model_id = "from-toml"
"#;
        let mut config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model.served_model_id.as_deref(), Some("from-toml"));

        environment.set("KILN_MODEL_SERVED_MODEL_ID", "from-env");
        config.apply_env_overrides().unwrap();
        assert_eq!(
            config.model.effective_served_model_id(),
            "from-env",
            "env var should override TOML value"
        );
    }

    #[test]
    fn malformed_environment_overrides_are_fatal_and_identify_the_input() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        let names = [
            "KILN_SERVER_PORT",
            "KILN_SERVER_REQUEST_TIMEOUT_SECS",
            "KILN_SERVER_EVAL_MODE",
            "KILN_SERVER_DEFAULT_THINKING_ENABLED",
            "KILN_SERVER_FOLD_REASONING_INTO_CONTENT",
            "KILN_SERVER_CHAT_PERFORMANCE_METADATA",
            "KILN_SERVER_CHAT_CONFIG_HASH_METADATA",
            "KILN_SERVER_SLOW_REQUEST_WARN_SECS",
            "KILN_SERVER_SHUTDOWN_TIMEOUT_SECS",
            "KILN_MEMORY_NUM_BLOCKS",
            "KILN_MEMORY_GPU_MEMORY_GB",
            "KILN_MEMORY_INFERENCE_MEMORY_FRACTION",
            "KILN_MEMORY_TRAINING_MEMORY_GB",
            "KILN_MEMORY_KV_CACHE_FP8",
            "KILN_MEMORY_CUDA_GRAPHS",
            "KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS",
            "KILN_TRAINING_NO_GRAD_CHECKPOINT",
            "KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES",
            "KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS",
            "KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE",
            "KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB",
            "KILN_TRAINING_CHECKPOINT_INTERVAL",
            "KILN_TRAINING_MAX_QUEUED_JOBS",
            "KILN_TRAINING_MAX_TRACKED_JOBS",
            "KILN_TRAINING_TRACKED_JOB_TTL_SECS",
            "KILN_PREFIX_CACHE_ENABLED",
            "KILN_PREFIX_CACHE_MAX_BLOCKS",
            "KILN_PREFIX_CACHE_MAX_ENTRIES",
            "KILN_SPECULATIVE_METHOD",
            "KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS",
            "KILN_SPECULATIVE_DRAFT_LAYERS",
            "KILN_ADAPTERS_MAX_DISK_BYTES",
            "KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES",
            "KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES",
            "KILN_STREAMING_PREFILL_MODE",
            "KILN_STREAMING_PREFILL_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS",
            "KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD",
        ];

        for name in names {
            environment.set(name, "definitely-invalid");
            let error = KilnConfig::default().apply_env_overrides().unwrap_err();
            environment.remove(name);
            let message = format!("{error:#}");
            assert!(message.contains(name), "{name}: {message}");
            assert!(message.contains("definitely-invalid"), "{name}: {message}");
        }
    }

    #[test]
    fn unknown_toml_fields_are_rejected_at_every_config_level() {
        for (name, input) in [
            ("root_typo", "root_typo = 17"),
            ("port_typo", "[server]\nport_typo = 17"),
            ("path_typo", "[model]\npath_typo = 'bad-model'"),
            ("bytes_typo", "[request_log]\nbytes_typo = 17"),
            ("jobs_typo", "[eval]\njobs_typo = 17"),
        ] {
            let error = toml::from_str::<KilnConfig>(input).unwrap_err().to_string();
            assert!(error.contains(name), "{name}: {error}");
        }
    }

    #[test]
    fn semantically_invalid_toml_fields_name_the_field_and_value() {
        for (field, value, input) in [
            ("server.host", "empty", "[server]\nhost = ''"),
            ("server.port", "0", "[server]\nport = 0"),
            ("model.model_id", "empty", "[model]\nmodel_id = ''"),
            ("memory.num_blocks", "0", "[memory]\nnum_blocks = 0"),
            (
                "training.max_queued_jobs",
                "0",
                "[training]\nmax_queued_jobs = 0",
            ),
            ("logging.format", "bogus", "[logging]\nformat = 'bogus'"),
            (
                "logging.level",
                "kiln=definitely-not-a-level",
                "[logging]\nlevel = 'kiln=definitely-not-a-level'",
            ),
            (
                "training.webhook_url",
                "smtp://bad",
                "[training]\nwebhook_url = 'smtp://bad'",
            ),
            (
                "prefix_cache.max_blocks",
                "0",
                "[prefix_cache]\nmax_blocks = 0",
            ),
            (
                "speculative.num_speculative_tokens",
                "0",
                "[speculative]\nnum_speculative_tokens = 0",
            ),
            (
                "speculative.num_speculative_tokens",
                "5",
                "[speculative]\nnum_speculative_tokens = 5",
            ),
            (
                "request_log.max_file_bytes",
                "1024",
                "[request_log]\nmax_file_bytes = 1024",
            ),
            ("eval.max_queued_jobs", "0", "[eval]\nmax_queued_jobs = 0"),
            (
                "eval.webhook_url",
                "not a URL",
                "[eval]\nwebhook_url = 'not a URL'",
            ),
            (
                "agent.max_concurrent_runs",
                "0",
                "[agent]\nmax_concurrent_runs = 0",
            ),
        ] {
            let config = toml::from_str::<KilnConfig>(input).unwrap();
            let error = config.validate().unwrap_err().to_string();
            assert!(error.contains(field), "{field}: {error}");
            assert!(error.contains(value), "{field}: {error}");
        }
    }

    #[test]
    fn actor_cycle_idle_is_bounded_and_source_tracked() {
        for value in [0, 75, ACTOR_CYCLE_IDLE_MAX_MS] {
            let config: KilnConfig =
                toml::from_str(&format!("[batching]\nactor_cycle_idle_ms = {value}\n")).unwrap();
            assert_eq!(config.batching.actor_cycle_idle_ms.millis(), value);
            assert_eq!(
                config.batching.actor_cycle_idle_ms.source(),
                ConfigValueSource::ConfigFile
            );
        }

        let error = toml::from_str::<KilnConfig>(&format!(
            "[batching]\nactor_cycle_idle_ms = {}\n",
            ACTOR_CYCLE_IDLE_MAX_MS + 1
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("batching.actor_cycle_idle_ms"), "{error}");
        assert!(error.contains("60001"), "{error}");

        let error = ActorCycleIdle::from_named_environment_value(
            "KILN_BATCHING_ACTOR_CYCLE_IDLE_MS",
            "not-a-duration",
        )
        .unwrap_err();
        let detail = format!("{error:#}");
        assert!(
            detail.contains("KILN_BATCHING_ACTOR_CYCLE_IDLE_MS"),
            "{detail}"
        );
        assert!(detail.contains("not-a-duration"), "{detail}");
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_config_environment_is_fatal() {
        use std::os::unix::ffi::OsStringExt;

        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let environment = ScopedConfigEnvironment::isolated();
        environment.set_os(
            "KILN_CONFIG",
            &std::ffi::OsString::from_vec(vec![b'/', b't', b'm', b'p', b'/', 0xff]),
        );
        let error = KilnConfig::load(None).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("KILN_CONFIG"), "{message}");
        assert!(message.contains("UTF-8"), "{message}");
    }

    #[test]
    fn checked_in_server_configs_parse_and_validate() {
        let config_dir = PathBuf::from("../../qualification/server-config");
        let mut config_paths = std::fs::read_dir(&config_dir)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to read checked-in server config directory {}: {error}",
                    config_dir.display()
                )
            })
            .map(|entry| {
                entry
                    .expect("failed to read server config directory entry")
                    .path()
            })
            .filter(|path| path.extension() == Some(OsStr::new("toml")))
            .collect::<Vec<_>>();
        config_paths.sort();
        assert!(
            !config_paths.is_empty(),
            "no checked-in server configs found in {}",
            config_dir.display()
        );

        for path in config_paths {
            let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!(
                    "failed to read checked-in config {}: {error}",
                    path.display()
                )
            });
            let config = KilnConfig::from_toml_with_sources(&contents).unwrap_or_else(|error| {
                panic!(
                    "failed to parse checked-in config {}: {error}",
                    path.display()
                )
            });
            config.validate().unwrap_or_else(|error| {
                panic!(
                    "failed to validate checked-in config {}: {error}",
                    path.display()
                )
            });
        }
    }
}

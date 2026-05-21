//! Stable training receipts for GRPO/SFT adapter runs.
//!
//! This is intentionally separate from `receipt.json`: `receipt.json` is the
//! older high-level reproducibility artifact used by distillation recipes,
//! while `train_receipt.json` is the machine-readable forensic record that cap
//! scripts can parse without scraping logs.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use kiln_core::config::ModelConfig;
use kiln_core::config_hashes::ConfigHashes;
use kiln_core::tokenizer::KilnTokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const TRAIN_RECEIPT_FILENAME: &str = "train_receipt.json";
pub const ADAPTER_CANARY_STATUS_FILENAME: &str = "adapter_canary_status.json";
pub const REWARD_FILTER_SIDECAR_FILENAME: &str = "reward_filter_groups.json";
pub const TRAIN_RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const ADAPTER_CANARY_STATUS_SCHEMA_VERSION: u32 = 1;
pub const ADAPTER_SMOKE_LOGIT_DELTA_EPSILON: f64 = 1e-6;
pub const ADAPTER_SMOKE_OUTPUT_TOKEN_LIMIT: usize = 4;
pub const ADAPTER_SMOKE_OUTPUT_CHAR_LIMIT: usize = 512;
pub const ADAPTER_SMOKE_LATENCY_MULTIPLIER: u64 = 4;
pub const ADAPTER_SMOKE_LATENCY_ABSOLUTE_MS: u64 = 500;
pub const LORA_DELTA_NEAR_ZERO_EPSILON: f64 = 1e-12;
pub const LORA_DELTA_EXTREME_SCALE_MULTIPLIER: f64 = 100.0;
pub const DEFAULT_REWARD_SATURATION_THRESHOLD: f64 = 0.95;
pub const DEFAULT_REWARD_LOW_VARIANCE_THRESHOLD: f64 = 1e-4;
pub const REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON: f64 = 1e-12;
pub const REWARD_MOST_GROUPS_FRACTION: f64 = 0.5;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainReceiptStatus {
    Success,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainFailureReason {
    DataSchemaError,
    AdapterLoadFailed,
    ZeroGroups,
    ZeroActionTokens,
    ZeroEnvTokens,
    NanLoss,
    Oom,
    ShapeMismatch,
    UnsafeLoraScale,
    BaseAdapterMissing,
    TrainingError,
}

impl TrainFailureReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DataSchemaError => "data_schema_error",
            Self::AdapterLoadFailed => "adapter_load_failed",
            Self::ZeroGroups => "zero_groups",
            Self::ZeroActionTokens => "zero_action_tokens",
            Self::ZeroEnvTokens => "zero_env_tokens",
            Self::NanLoss => "nan_loss",
            Self::Oom => "oom",
            Self::ShapeMismatch => "shape_mismatch",
            Self::UnsafeLoraScale => "unsafe_lora_scale",
            Self::BaseAdapterMissing => "base_adapter_missing",
            Self::TrainingError => "training_error",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainReceipt {
    pub schema_version: u32,
    pub receipt_type: String,
    pub adapter_name: String,
    pub produced_at: String,
    pub status: TrainReceiptStatus,
    pub failure_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_message: Option<String>,
    pub kiln: KilnSourceReceipt,
    pub model: ModelReceipt,
    pub tokenizer: TokenizerReceipt,
    pub adapters: AdapterReceiptSet,
    pub training_data: TrainingDataReceipt,
    pub hyperparameters: HyperparameterReceipt,
    pub grpo: Option<GrpoReceipt>,
    pub echo: EchoReceipt,
    pub no_policy_loss: bool,
    pub data: DataStatsReceipt,
    pub rewards: RewardStatsReceipt,
    pub token_counts: TokenCountReceipt,
    #[serde(default)]
    pub phase_timings: TrainingPhaseTimingsReceipt,
    pub runtime: RuntimeReceipt,
    #[serde(default)]
    pub config_hashes: ConfigHashes,
    pub lora_delta_norms: Vec<LoraDeltaNormSummary>,
    #[serde(default)]
    pub lora_grad_norms: Vec<LoraGradNormSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_smoke_test: Option<AdapterSmokeTestReceipt>,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KilnSourceReceipt {
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub git_source: Option<String>,
    pub package_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_config_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelReceipt {
    pub path: Option<String>,
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerReceipt {
    /// Backward-compatible combined hash of tokenizer JSON plus chat template.
    pub config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterReceiptSet {
    pub base: AdapterFileReceipt,
    pub output: AdapterFileReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterFileReceipt {
    pub path: Option<String>,
    pub adapter_model_sha256: Option<String>,
    pub adapter_model_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingDataReceipt {
    pub source: String,
    pub path: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperparameterReceipt {
    pub mode: String,
    pub rank: usize,
    pub alpha: f32,
    pub alpha_over_rank: Option<f32>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoReceipt {
    pub kl_coeff: f64,
    pub clip_epsilon: f64,
    pub clip_eps_high: Option<f64>,
    pub dynamic_sampling: bool,
    pub dynamic_groups_filtered: usize,
    pub advantage_mode: serde_json::Value,
    pub loss_aggregation: serde_json::Value,
    pub kl_estimator: serde_json::Value,
    pub is_level: serde_json::Value,
    pub reference_policy: serde_json::Value,
    pub entropy_aware_kl_quantile: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EchoReceipt {
    pub enabled: bool,
    pub lambda: Option<f64>,
    pub env_mask_mode: Option<String>,
    pub warning_filter: Option<bool>,
    #[serde(default)]
    pub initial_env_ce: Option<f64>,
    #[serde(default)]
    pub final_env_ce: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DataStatsReceipt {
    pub examples_read: usize,
    pub examples_filtered: usize,
    pub examples_trained: usize,
    pub groups_read: usize,
    pub groups_filtered: usize,
    pub groups_trained: usize,
    pub completions_read: usize,
    pub completions_trained: usize,
    #[serde(default)]
    pub reward_groups_filtered: usize,
    #[serde(default)]
    pub reward_groups_kept: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_filter_sidecar: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardStatsReceipt {
    pub count: usize,
    pub mean: Option<f64>,
    pub stdev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    #[serde(default)]
    pub group_count: usize,
    #[serde(default)]
    pub all_pass_group_count: usize,
    #[serde(default)]
    pub all_fail_group_count: usize,
    #[serde(default)]
    pub degenerate_group_count: usize,
    pub group_variance_histogram: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramBucket {
    pub label: String,
    pub min_inclusive: Option<f64>,
    pub max_inclusive: Option<f64>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardFilterSidecar {
    pub schema_version: u32,
    pub sidecar_type: String,
    pub source: String,
    pub var_min: Option<f64>,
    pub var_max: Option<f64>,
    pub min_groups: usize,
    pub on_empty_filter: String,
    pub empty_filter_triggered: bool,
    pub empty_filter_action: String,
    pub groups_read: usize,
    pub groups_kept: usize,
    pub groups_dropped: usize,
    pub kept_group_ids: Vec<String>,
    pub dropped_group_ids: Vec<String>,
    pub groups: Vec<RewardFilterGroupDecisionReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardFilterGroupDecisionReceipt {
    pub id: String,
    pub source_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_line: Option<usize>,
    pub reward_variance: f64,
    pub matched_filter: bool,
    pub kept: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TokenCountReceipt {
    pub action_tokens: u64,
    /// Active env-CE tokens after warning-prefix filtering. Kept as the
    /// legacy field name for stable consumers.
    pub env_tokens: u64,
    /// Observation tokens before warning-prefix filtering. For trajectories
    /// without warning prefixes this equals `env_tokens`.
    #[serde(default)]
    pub env_tokens_before_warning_filter: u64,
    /// Explicit alias for the filtered env-CE token count so receipts expose
    /// both sides of the warning-filter gate without overloading names.
    #[serde(default)]
    pub env_tokens_after_warning_filter: u64,
    /// Tokens excluded from env-CE by `warning_prefix_len` filtering.
    #[serde(default)]
    pub warning_tokens_filtered: u64,
    pub context_tokens: u64,
}

impl TokenCountReceipt {
    pub fn observe_completion(
        &mut self,
        seq_len: usize,
        action_tokens: u64,
        env_tokens_after_warning_filter: u64,
        env_tokens_before_warning_filter: u64,
    ) {
        let env_before = env_tokens_before_warning_filter.max(env_tokens_after_warning_filter);
        let context = seq_len
            .saturating_sub(action_tokens as usize)
            .saturating_sub(env_tokens_after_warning_filter as usize) as u64;

        self.action_tokens = self.action_tokens.saturating_add(action_tokens);
        self.env_tokens = self
            .env_tokens
            .saturating_add(env_tokens_after_warning_filter);
        self.env_tokens_after_warning_filter = self
            .env_tokens_after_warning_filter
            .saturating_add(env_tokens_after_warning_filter);
        self.env_tokens_before_warning_filter = self
            .env_tokens_before_warning_filter
            .saturating_add(env_before);
        self.warning_tokens_filtered = self
            .warning_tokens_filtered
            .saturating_add(env_before.saturating_sub(env_tokens_after_warning_filter));
        self.context_tokens = self.context_tokens.saturating_add(context);
    }

    pub fn add_from(&mut self, other: &Self) {
        self.action_tokens = self.action_tokens.saturating_add(other.action_tokens);
        self.env_tokens = self.env_tokens.saturating_add(other.env_tokens);
        self.env_tokens_before_warning_filter = self
            .env_tokens_before_warning_filter
            .saturating_add(other.env_tokens_before_warning_filter);
        self.env_tokens_after_warning_filter = self
            .env_tokens_after_warning_filter
            .saturating_add(other.env_tokens_after_warning_filter);
        self.warning_tokens_filtered = self
            .warning_tokens_filtered
            .saturating_add(other.warning_tokens_filtered);
        self.context_tokens = self.context_tokens.saturating_add(other.context_tokens);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingPhaseTimingsReceipt {
    pub tokenize_ms: f64,
    pub mask_build_ms: f64,
    pub reference_forward_ms: f64,
    pub policy_forward_ms: f64,
    pub backward_ms: f64,
    pub optimizer_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EchoActivityMetrics {
    pub initial_env_ce: Option<f64>,
    pub final_env_ce: Option<f64>,
    pub measurements: usize,
}

impl EchoActivityMetrics {
    pub fn observe_env_ce(&mut self, env_ce: Option<f64>) {
        let Some(env_ce) = env_ce else {
            return;
        };
        if !env_ce.is_finite() {
            return;
        }
        if self.initial_env_ce.is_none() {
            self.initial_env_ce = Some(env_ce);
        }
        self.final_env_ce = Some(env_ce);
        self.measurements = self.measurements.saturating_add(1);
    }

    pub fn apply_to_echo_receipt(&self, receipt: &mut EchoReceipt) {
        if self.measurements == 0 {
            return;
        }
        receipt.initial_env_ce = self.initial_env_ce;
        receipt.final_env_ce = self.final_env_ce;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeReceipt {
    pub wall_clock_ms: u64,
    pub peak_vram_mib: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraDeltaNormSummary {
    pub module: String,
    pub pair_count: usize,
    pub a_l2_mean: f64,
    pub a_l2_max: f64,
    pub b_l2_mean: f64,
    pub b_l2_max: f64,
    pub delta_l2_upper_bound_mean: f64,
    pub delta_l2_upper_bound_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraGradNormSummary {
    pub module: String,
    pub sample_count: usize,
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterSmokeTestReceipt {
    pub enabled: bool,
    pub passed: bool,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_diagnostics: Vec<AdapterSmokePromptDiagnosisReceipt>,
    pub prompts: Vec<AdapterSmokePromptReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterSmokePromptReceipt {
    pub prompt: String,
    pub finite_logits: bool,
    pub logit_delta_l2: Option<f64>,
    pub generated_text_different: bool,
    pub base_output: String,
    pub adapter_output: String,
    pub adapter_output_tokens: usize,
    pub adapter_output_chars: usize,
    #[serde(default)]
    pub base_generation_ms: u64,
    #[serde(default)]
    pub adapter_generation_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterSmokePromptDiagnosisReceipt {
    pub prompt: String,
    pub outcome: AdapterSmokePromptDiagnosis,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdapterSmokePromptDiagnosis {
    NonFiniteLogits,
    EmptyAdapterOutput,
    LogitsChangedTextChanged,
    LogitsChangedTextIdentical,
    TextChangedWithoutMeasurableLogitDelta,
    NoLogitChange,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdapterCanaryState {
    Passed,
    Quarantined,
}

impl AdapterCanaryState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Quarantined => "quarantined",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterCanaryStatusReceipt {
    pub schema_version: u32,
    pub receipt_type: String,
    pub adapter_name: String,
    pub produced_at: String,
    pub source: String,
    pub status: AdapterCanaryState,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub checks: Vec<AdapterCanaryCheckReceipt>,
    #[serde(default)]
    pub prompt_diagnostics: Vec<AdapterSmokePromptDiagnosisReceipt>,
}

impl AdapterCanaryStatusReceipt {
    pub fn is_quarantined(&self) -> bool {
        self.status == AdapterCanaryState::Quarantined
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterCanaryCheckReceipt {
    pub name: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

impl Default for RewardStatsReceipt {
    fn default() -> Self {
        Self {
            count: 0,
            mean: None,
            stdev: None,
            min: None,
            max: None,
            group_count: 0,
            all_pass_group_count: 0,
            all_fail_group_count: 0,
            degenerate_group_count: 0,
            group_variance_histogram: variance_histogram(&[]),
        }
    }
}

impl TrainReceipt {
    pub fn new(
        adapter_name: impl Into<String>,
        mode: impl Into<String>,
        model_config: &ModelConfig,
        tokenizer: &KilnTokenizer,
        hyperparameters: HyperparameterReceipt,
        config: serde_json::Value,
    ) -> Self {
        let config_hashes = ConfigHashes::from_model_tokenizer(
            model_config,
            tokenizer,
            kiln_core::config_hashes::kiln_env_config_hash(&config),
        );
        Self {
            schema_version: TRAIN_RECEIPT_SCHEMA_VERSION,
            receipt_type: "kiln_train_receipt".to_string(),
            adapter_name: adapter_name.into(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            status: TrainReceiptStatus::Success,
            failure_reason: None,
            failure_message: None,
            kiln: detect_kiln_source(config_hashes.kiln_env_config_hash.clone()),
            model: ModelReceipt {
                path: detect_model_path(),
                config_hash: config_hashes.model_config_hash.clone(),
            },
            tokenizer: TokenizerReceipt {
                config_hash: tokenizer.config_sha256().ok(),
                tokenizer_config_hash: config_hashes.tokenizer_config_hash.clone(),
                chat_template_hash: config_hashes.chat_template_hash.clone(),
            },
            adapters: AdapterReceiptSet {
                base: AdapterFileReceipt::none(),
                output: AdapterFileReceipt::none(),
            },
            training_data: TrainingDataReceipt {
                source: mode.into(),
                path: None,
                sha256: None,
            },
            hyperparameters,
            grpo: None,
            echo: EchoReceipt::disabled(),
            no_policy_loss: false,
            data: DataStatsReceipt::default(),
            rewards: RewardStatsReceipt::default(),
            token_counts: TokenCountReceipt::default(),
            phase_timings: TrainingPhaseTimingsReceipt::default(),
            runtime: RuntimeReceipt {
                wall_clock_ms: 0,
                peak_vram_mib: None,
            },
            config_hashes,
            lora_delta_norms: Vec::new(),
            lora_grad_norms: Vec::new(),
            adapter_smoke_test: None,
            config,
        }
    }

    pub fn mark_failed(self, message: impl Into<String>) -> Self {
        let message = message.into();
        let reason = classify_training_failure(&message);
        self.mark_failed_with_reason(reason, message)
    }

    pub fn mark_failed_with_reason(
        mut self,
        reason: TrainFailureReason,
        message: impl Into<String>,
    ) -> Self {
        self.status = TrainReceiptStatus::Failed;
        self.failure_reason = Some(reason.as_str().to_string());
        self.failure_message = Some(message.into());
        self
    }

    pub fn write_to_adapter_dir(&self, adapter_dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(adapter_dir).with_context(|| {
            format!(
                "create adapter dir {} for train receipt",
                adapter_dir.display()
            )
        })?;
        let path = adapter_dir.join(TRAIN_RECEIPT_FILENAME);
        let json = serde_json::to_string_pretty(self).context("serialize train receipt")?;
        std::fs::write(&path, json)
            .with_context(|| format!("write train receipt {}", path.display()))?;
        if let Some(status) = adapter_canary_status_from_train_receipt(self) {
            write_adapter_canary_status(adapter_dir, &status)?;
        }
        Ok(path)
    }

    pub fn read_from_adapter_dir(adapter_dir: &Path) -> Result<Option<Self>> {
        let path = adapter_dir.join(TRAIN_RECEIPT_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read train receipt {}", path.display()))?;
        let receipt = serde_json::from_slice(&bytes)
            .with_context(|| format!("deserialize train receipt {}", path.display()))?;
        Ok(Some(receipt))
    }
}

pub fn adapter_canary_status_from_train_receipt(
    receipt: &TrainReceipt,
) -> Option<AdapterCanaryStatusReceipt> {
    receipt.adapter_smoke_test.as_ref().map(|smoke| {
        build_adapter_canary_status_receipt(&receipt.adapter_name, &receipt.produced_at, smoke)
    })
}

pub fn build_adapter_canary_status_receipt(
    adapter_name: &str,
    produced_at: &str,
    smoke: &AdapterSmokeTestReceipt,
) -> AdapterCanaryStatusReceipt {
    let checks = adapter_canary_checks_from_smoke_test(smoke);
    let checks_passed = checks.iter().all(|check| check.passed);
    let passed = smoke.passed && checks_passed;
    let status = if passed {
        AdapterCanaryState::Passed
    } else {
        AdapterCanaryState::Quarantined
    };
    let failure_reason = if passed {
        None
    } else {
        smoke
            .warnings
            .first()
            .cloned()
            .or_else(|| checks.iter().find_map(|check| check.failure_reason.clone()))
            .or_else(|| Some("adapter canary failed".to_string()))
    };

    AdapterCanaryStatusReceipt {
        schema_version: ADAPTER_CANARY_STATUS_SCHEMA_VERSION,
        receipt_type: "kiln_adapter_canary_status".to_string(),
        adapter_name: adapter_name.to_string(),
        produced_at: produced_at.to_string(),
        source: "adapter_smoke_test".to_string(),
        status,
        passed,
        failure_reason,
        warnings: smoke.warnings.clone(),
        notes: smoke.notes.clone(),
        checks,
        prompt_diagnostics: smoke.prompt_diagnostics.clone(),
    }
}

pub fn write_adapter_canary_status(
    adapter_dir: &Path,
    status: &AdapterCanaryStatusReceipt,
) -> Result<PathBuf> {
    std::fs::create_dir_all(adapter_dir).with_context(|| {
        format!(
            "create adapter dir {} for adapter canary status",
            adapter_dir.display()
        )
    })?;
    let path = adapter_dir.join(ADAPTER_CANARY_STATUS_FILENAME);
    let json = serde_json::to_string_pretty(status).context("serialize adapter canary status")?;
    std::fs::write(&path, json)
        .with_context(|| format!("write adapter canary status {}", path.display()))?;
    Ok(path)
}

pub fn read_adapter_canary_status_from_adapter_dir(
    adapter_dir: &Path,
) -> Result<Option<AdapterCanaryStatusReceipt>> {
    let path = adapter_dir.join(ADAPTER_CANARY_STATUS_FILENAME);
    if path.exists() {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read adapter canary status {}", path.display()))?;
        let status = serde_json::from_slice(&bytes)
            .with_context(|| format!("deserialize adapter canary status {}", path.display()))?;
        return Ok(Some(status));
    }

    let Some(receipt) = TrainReceipt::read_from_adapter_dir(adapter_dir)? else {
        return Ok(None);
    };
    Ok(adapter_canary_status_from_train_receipt(&receipt))
}

pub fn classify_training_failure(message: &str) -> TrainFailureReason {
    let lower = message.to_ascii_lowercase();

    if lower.contains("unsafe lora scaling") || lower.contains("lora rank must") {
        return TrainFailureReason::UnsafeLoraScale;
    }
    if lower.contains("out of memory")
        || lower.contains("cuda error: out of memory")
        || lower.contains("cudnn_status_alloc_failed")
        || lower.contains("cublas_status_alloc_failed")
        || lower.contains("allocation failed")
        || lower.contains("oom")
    {
        return TrainFailureReason::Oom;
    }
    if lower.contains("echo is enabled")
        && (lower.contains("env_mask is empty")
            || lower.contains("no environment tokens")
            || lower.contains("env tokens")
            || lower.contains("environment tokens"))
    {
        return TrainFailureReason::ZeroEnvTokens;
    }
    if lower.contains("no action tokens")
        || lower.contains("zero action tokens")
        || lower.contains("empty action_mask")
        || lower.contains("action_mask is empty")
        || lower.contains("no supervised assistant tokens")
    {
        return TrainFailureReason::ZeroActionTokens;
    }
    if lower.contains("no valid grpo groups")
        || lower.contains("zero valid grpo groups")
        || lower.contains("no grpo groups")
        || lower.contains("no valid training examples")
        || lower.contains("prompts must be non-empty")
        || lower.contains("no valid prompts")
        || lower.contains("no valid grpo completions")
        || lower.contains("no valid completions")
        || lower.contains("reward variance filter kept")
        || lower.contains("below --min-groups")
    {
        return TrainFailureReason::ZeroGroups;
    }
    if lower.contains("base adapter")
        && (lower.contains("shape mismatch")
            || lower.contains("rank mismatch")
            || lower.contains("target_modules mismatch")
            || lower.contains("missing tensor")
            || lower.contains("unexpected tensor")
            || lower.contains("shape conversion"))
    {
        return TrainFailureReason::ShapeMismatch;
    }
    if lower.contains("base adapter")
        && (lower.contains("no such file")
            || lower.contains("not found")
            || lower.contains("does not exist")
            || lower.contains("read base adapter config")
            || lower.contains("read base adapter tensors"))
    {
        return TrainFailureReason::BaseAdapterMissing;
    }
    if (lower.contains("adapter") || lower.contains("safetensors"))
        && (lower.contains("load")
            || lower.contains("read")
            || lower.contains("parse")
            || lower.contains("deserialize"))
    {
        return TrainFailureReason::AdapterLoadFailed;
    }
    if lower.contains("non-finite loss")
        || lower.contains("nan loss")
        || lower.contains("loss nan")
        || lower.contains("infinite loss")
    {
        return TrainFailureReason::NanLoss;
    }
    if lower.contains("malformed trajectory role")
        || lower.contains("invalid trajectory")
        || lower.contains("data schema")
        || lower.contains("missing field")
        || (lower.contains("json") && (lower.contains("parse") || lower.contains("deserialize")))
    {
        return TrainFailureReason::DataSchemaError;
    }

    TrainFailureReason::TrainingError
}

pub fn training_failure_error_message(message: &str) -> String {
    let reason = classify_training_failure(message);
    format!("failure_reason={}: {message}", reason.as_str())
}

pub fn annotate_training_error(err: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("{}", training_failure_error_message(&format!("{err:#}")))
}

pub fn write_reward_filter_sidecar(
    adapter_dir: &Path,
    sidecar: &RewardFilterSidecar,
) -> Result<PathBuf> {
    std::fs::create_dir_all(adapter_dir).with_context(|| {
        format!(
            "create adapter dir {} for reward filter sidecar",
            adapter_dir.display()
        )
    })?;
    let path = adapter_dir.join(REWARD_FILTER_SIDECAR_FILENAME);
    let json = serde_json::to_string_pretty(sidecar).context("serialize reward filter sidecar")?;
    std::fs::write(&path, json)
        .with_context(|| format!("write reward filter sidecar {}", path.display()))?;
    Ok(path)
}

impl AdapterFileReceipt {
    pub fn none() -> Self {
        Self {
            path: None,
            adapter_model_sha256: None,
            adapter_model_bytes: None,
        }
    }
}

impl EchoReceipt {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            lambda: None,
            env_mask_mode: None,
            warning_filter: None,
            initial_env_ce: None,
            final_env_ce: None,
        }
    }
}

pub fn log_training_token_counts(mode: &str, token_counts: &TokenCountReceipt) {
    tracing::info!(
        mode,
        action_tokens = token_counts.action_tokens,
        env_tokens = token_counts.env_tokens,
        env_tokens_before_warning_filter = token_counts.env_tokens_before_warning_filter,
        env_tokens_after_warning_filter = token_counts.env_tokens_after_warning_filter,
        warning_tokens_filtered = token_counts.warning_tokens_filtered,
        context_tokens = token_counts.context_tokens,
        "training token counts"
    );
    warn_if_warning_filter_stripped_most_env_tokens(mode, token_counts);
}

pub fn warn_echo_enabled_without_env_tokens(
    mode: &str,
    echo_enabled: bool,
    token_counts: &TokenCountReceipt,
) {
    if echo_enabled && token_counts.env_tokens == 0 {
        tracing::warn!(
            mode,
            action_tokens = token_counts.action_tokens,
            env_tokens = token_counts.env_tokens,
            context_tokens = token_counts.context_tokens,
            "ECHO is enabled but no environment tokens were observed; env-CE is inactive"
        );
    }
}

pub fn warning_filter_stripped_most_env_tokens(token_counts: &TokenCountReceipt) -> bool {
    token_counts.env_tokens_before_warning_filter > 0
        && token_counts.warning_tokens_filtered.saturating_mul(2)
            > token_counts.env_tokens_before_warning_filter
}

pub fn warn_if_warning_filter_stripped_most_env_tokens(
    mode: &str,
    token_counts: &TokenCountReceipt,
) {
    if warning_filter_stripped_most_env_tokens(token_counts) {
        tracing::warn!(
            mode,
            env_tokens_before_warning_filter = token_counts.env_tokens_before_warning_filter,
            env_tokens_after_warning_filter = token_counts.env_tokens_after_warning_filter,
            warning_tokens_filtered = token_counts.warning_tokens_filtered,
            "warning-prefix filter stripped most environment tokens; verify harness warnings are not dominating ECHO data"
        );
    }
}

pub fn warn_lora_delta_norms(
    mode: &str,
    adapter_name: &str,
    summaries: &[LoraDeltaNormSummary],
    alpha_over_rank: f64,
) {
    for warning in lora_delta_norm_warnings(summaries, alpha_over_rank) {
        tracing::warn!(
            mode,
            adapter = adapter_name,
            warning = %warning,
            "LoRA delta norm warning"
        );
    }
}

pub fn warn_reward_diagnostics(
    mode: &str,
    adapter_name: &str,
    rewards: &RewardStatsReceipt,
    saturation_threshold: f64,
    low_variance_threshold: f64,
) {
    for warning in reward_diagnostic_warnings(rewards, saturation_threshold, low_variance_threshold)
    {
        tracing::warn!(
            mode,
            adapter = adapter_name,
            warning = %warning,
            "GRPO reward diagnostic warning"
        );
    }
}

pub fn build_adapter_smoke_test_receipt(
    prompts: Vec<AdapterSmokePromptReceipt>,
) -> AdapterSmokeTestReceipt {
    let mut warnings = Vec::new();
    let mut notes = Vec::new();
    let prompt_diagnostics: Vec<_> = prompts.iter().map(diagnose_adapter_smoke_prompt).collect();

    if prompts.is_empty() {
        warnings.push("adapter smoke test did not run any prompts".to_string());
    }

    if prompts
        .iter()
        .any(|prompt| !prompt.finite_logits || prompt.logit_delta_l2.is_none())
    {
        warnings.push("adapter smoke test observed non-finite logits".to_string());
    }

    let checked_logit_delta = prompts.iter().any(|prompt| {
        prompt
            .logit_delta_l2
            .is_some_and(|delta| delta > ADAPTER_SMOKE_LOGIT_DELTA_EPSILON)
    });
    if !checked_logit_delta {
        warnings.push(format!(
            "adapter smoke test observed no nonzero logit delta from base (all logit_delta_l2 <= {ADAPTER_SMOKE_LOGIT_DELTA_EPSILON:e})"
        ));
    }

    let measurable_effect = prompts.iter().any(|prompt| {
        prompt
            .logit_delta_l2
            .is_some_and(|delta| delta > ADAPTER_SMOKE_LOGIT_DELTA_EPSILON)
            || prompt.generated_text_different
    });
    if !measurable_effect {
        warnings.push(format!(
            "adapter smoke test observed no measurable adapter effect (all logit_delta_l2 <= {ADAPTER_SMOKE_LOGIT_DELTA_EPSILON:e} and generated text matched base); inspect adapter load path, lora_grad_norms, lora_delta_norms, KL/clipping, masks, and LoRA scale"
        ));
    }

    if prompts
        .iter()
        .any(|prompt| prompt.adapter_output_tokens == 0 || prompt.adapter_output.trim().is_empty())
    {
        warnings
            .push("adapter smoke test produced empty adapter output on canary prompt".to_string());
    }

    if prompts.iter().any(adapter_smoke_output_too_long) {
        warnings.push(format!(
            "adapter smoke test output length sanity failed (adapter output exceeded {ADAPTER_SMOKE_OUTPUT_CHAR_LIMIT} chars or the bounded token budget)"
        ));
    }

    if prompts.iter().any(adapter_smoke_latency_regressed) {
        warnings.push(format!(
            "adapter smoke test latency sanity failed (adapter generation exceeded base generation by more than {ADAPTER_SMOKE_LATENCY_MULTIPLIER}x + {ADAPTER_SMOKE_LATENCY_ABSOLUTE_MS}ms)"
        ));
    }

    if prompt_diagnostics.iter().any(|diagnostic| {
        diagnostic.outcome == AdapterSmokePromptDiagnosis::LogitsChangedTextIdentical
    }) {
        notes.push(
            "adapter smoke test observed changed logits with byte-identical greedy text; deterministic argmax decoding can keep selecting the same top token even when lower-ranked logits move, so this is not by itself evidence of a no-op adapter".to_string(),
        );
    }

    if prompt_diagnostics
        .iter()
        .any(|diagnostic| diagnostic.outcome == AdapterSmokePromptDiagnosis::NoLogitChange)
    {
        notes.push(
            "adapter smoke test observed no checked-logit movement on at least one prompt; use lora_grad_norms and lora_delta_norms to distinguish zero gradients from an adapter loading, masking, KL/clipping, or scale issue".to_string(),
        );
    }

    AdapterSmokeTestReceipt {
        enabled: true,
        passed: warnings.is_empty(),
        warnings,
        notes,
        prompt_diagnostics,
        prompts,
    }
}

fn adapter_smoke_output_too_long(prompt: &AdapterSmokePromptReceipt) -> bool {
    prompt.adapter_output_tokens > 0
        && (prompt.adapter_output_tokens > ADAPTER_SMOKE_OUTPUT_TOKEN_LIMIT
            || prompt.adapter_output_chars > ADAPTER_SMOKE_OUTPUT_CHAR_LIMIT)
}

fn adapter_smoke_latency_regressed(prompt: &AdapterSmokePromptReceipt) -> bool {
    prompt.adapter_generation_ms
        > prompt
            .base_generation_ms
            .saturating_mul(ADAPTER_SMOKE_LATENCY_MULTIPLIER)
            .saturating_add(ADAPTER_SMOKE_LATENCY_ABSOLUTE_MS)
}

pub fn adapter_canary_checks_from_smoke_test(
    smoke: &AdapterSmokeTestReceipt,
) -> Vec<AdapterCanaryCheckReceipt> {
    let any_prompt = smoke
        .prompts
        .iter()
        .any(|prompt| prompt.adapter_output_tokens > 0 && !prompt.adapter_output.trim().is_empty());
    let tool_prompt = smoke.prompts.iter().any(|prompt| {
        let lower = prompt.prompt.to_ascii_lowercase();
        (lower.contains("tool") || lower.contains("json"))
            && prompt.adapter_output_tokens > 0
            && !prompt.adapter_output.trim().is_empty()
    });
    let output_length_ok = !smoke.prompts.is_empty()
        && smoke.prompts.iter().all(|prompt| {
            prompt.adapter_output_tokens > 0
                && prompt.adapter_output_tokens <= ADAPTER_SMOKE_OUTPUT_TOKEN_LIMIT
                && prompt.adapter_output_chars <= ADAPTER_SMOKE_OUTPUT_CHAR_LIMIT
        });
    let finite_logits = !smoke.prompts.is_empty()
        && smoke
            .prompts
            .iter()
            .all(|prompt| prompt.finite_logits && prompt.logit_delta_l2.is_some());
    let latency_ok = !smoke.prompts.is_empty()
        && smoke
            .prompts
            .iter()
            .all(|prompt| !adapter_smoke_latency_regressed(prompt));
    let nonzero_logit_delta = smoke.prompts.iter().any(|prompt| {
        prompt
            .logit_delta_l2
            .is_some_and(|delta| delta > ADAPTER_SMOKE_LOGIT_DELTA_EPSILON)
    });

    vec![
        canary_check(
            "simple_short_completion",
            any_prompt,
            "no canary prompt produced non-empty adapter text",
        ),
        canary_check(
            "simple_tool_call_shaped_prompt",
            tool_prompt,
            "no tool-call-shaped canary prompt produced non-empty adapter text",
        ),
        canary_check(
            "output_length_sanity",
            output_length_ok,
            "adapter canary output was empty or exceeded the bounded length sanity limit",
        ),
        canary_check(
            "finite_logits",
            finite_logits,
            "adapter canary observed non-finite logits",
        ),
        canary_check(
            "latency_sanity",
            latency_ok,
            "adapter canary latency exceeded the relative sanity threshold",
        ),
        canary_check(
            "nonzero_logit_delta_from_base",
            nonzero_logit_delta,
            "adapter canary observed no nonzero logit delta from base",
        ),
    ]
}

fn canary_check(name: &str, passed: bool, failure_reason: &str) -> AdapterCanaryCheckReceipt {
    AdapterCanaryCheckReceipt {
        name: name.to_string(),
        passed,
        failure_reason: (!passed).then(|| failure_reason.to_string()),
    }
}

pub fn diagnose_adapter_smoke_prompt(
    prompt: &AdapterSmokePromptReceipt,
) -> AdapterSmokePromptDiagnosisReceipt {
    let logit_delta_changed = prompt
        .logit_delta_l2
        .is_some_and(|delta| delta > ADAPTER_SMOKE_LOGIT_DELTA_EPSILON);
    let adapter_output_empty =
        prompt.adapter_output_tokens == 0 || prompt.adapter_output.trim().is_empty();

    let (outcome, explanation) = if !prompt.finite_logits || prompt.logit_delta_l2.is_none() {
        (
            AdapterSmokePromptDiagnosis::NonFiniteLogits,
            "base-vs-adapter logits could not be compared because at least one checked logit was non-finite".to_string(),
        )
    } else if adapter_output_empty {
        (
            AdapterSmokePromptDiagnosis::EmptyAdapterOutput,
            "adapter generation produced no non-whitespace text for this canary prompt".to_string(),
        )
    } else if logit_delta_changed && prompt.generated_text_different {
        (
            AdapterSmokePromptDiagnosis::LogitsChangedTextChanged,
            "adapter changed checked logits and the greedy output differed from base".to_string(),
        )
    } else if logit_delta_changed {
        (
            AdapterSmokePromptDiagnosis::LogitsChangedTextIdentical,
            "adapter changed checked logits but greedy output stayed byte-identical; deterministic decoding can preserve the selected token sequence despite logit movement".to_string(),
        )
    } else if prompt.generated_text_different {
        (
            AdapterSmokePromptDiagnosis::TextChangedWithoutMeasurableLogitDelta,
            "greedy output differed even though the single checked logit vector did not exceed the logit-delta threshold; inspect later decode positions or sampling settings".to_string(),
        )
    } else {
        (
            AdapterSmokePromptDiagnosis::NoLogitChange,
            "adapter did not move checked logits above threshold and greedy output matched base on this prompt".to_string(),
        )
    };

    AdapterSmokePromptDiagnosisReceipt {
        prompt: prompt.prompt.clone(),
        outcome,
        explanation,
    }
}

pub fn failed_adapter_smoke_test_receipt(error: impl Into<String>) -> AdapterSmokeTestReceipt {
    AdapterSmokeTestReceipt {
        enabled: true,
        passed: false,
        warnings: vec![format!(
            "adapter smoke test failed before metrics were recorded: {}",
            error.into()
        )],
        notes: Vec::new(),
        prompt_diagnostics: Vec::new(),
        prompts: Vec::new(),
    }
}

pub fn adapter_file_receipt(adapter_dir: Option<&Path>) -> AdapterFileReceipt {
    let Some(dir) = adapter_dir else {
        return AdapterFileReceipt::none();
    };
    let adapter_model = dir.join("adapter_model.safetensors");
    let (hash, bytes) = if adapter_model.exists() {
        (
            sha256_file(&adapter_model).ok(),
            adapter_model.metadata().ok().map(|m| m.len()),
        )
    } else {
        (None, None)
    };
    AdapterFileReceipt {
        path: Some(dir.display().to_string()),
        adapter_model_sha256: hash,
        adapter_model_bytes: bytes,
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {} for sha256", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {} for sha256", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("sha256:{}", hex_digest(h.finalize().as_slice())))
}

pub fn sha256_json_value(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let digest = Sha256::digest(&bytes);
    format!("sha256:{}", hex_digest(digest.as_slice()))
}

pub fn sha256_json_serializable<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .map(|value| sha256_json_value(&value))
}

pub fn reward_stats_from_groups<'a, I>(groups: I) -> RewardStatsReceipt
where
    I: IntoIterator<Item = &'a [f64]>,
{
    reward_stats_from_groups_with_threshold(groups, DEFAULT_REWARD_SATURATION_THRESHOLD)
}

pub fn reward_stats_from_groups_with_threshold<'a, I>(
    groups: I,
    all_pass_threshold: f64,
) -> RewardStatsReceipt
where
    I: IntoIterator<Item = &'a [f64]>,
{
    let mut rewards = Vec::new();
    let mut variances = Vec::new();
    let mut group_count = 0usize;
    let mut all_pass_group_count = 0usize;
    let mut all_fail_group_count = 0usize;
    let mut degenerate_group_count = 0usize;
    for group_rewards in groups {
        if group_rewards.is_empty() {
            continue;
        }
        group_count += 1;
        rewards.extend_from_slice(group_rewards);
        let group_variance = population_variance(group_rewards);
        variances.push(group_variance);
        if group_variance <= REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON {
            degenerate_group_count += 1;
        }
        if group_rewards
            .iter()
            .all(|reward| *reward >= all_pass_threshold)
        {
            all_pass_group_count += 1;
        }
        if group_rewards.iter().all(|reward| *reward <= 0.0) {
            all_fail_group_count += 1;
        }
    }
    let count = rewards.len();
    if count == 0 {
        return RewardStatsReceipt::default();
    }
    let mean = rewards.iter().sum::<f64>() / count as f64;
    let variance = rewards
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;
    let min = rewards.iter().copied().fold(f64::INFINITY, f64::min);
    let max = rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    RewardStatsReceipt {
        count,
        mean: Some(mean),
        stdev: Some(variance.sqrt()),
        min: Some(min),
        max: Some(max),
        group_count,
        all_pass_group_count,
        all_fail_group_count,
        degenerate_group_count,
        group_variance_histogram: variance_histogram(&variances),
    }
}

pub fn reward_diagnostic_warnings(
    rewards: &RewardStatsReceipt,
    saturation_threshold: f64,
    low_variance_threshold: f64,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if rewards.group_count > 0 {
        let saturated_groups = rewards
            .all_pass_group_count
            .saturating_add(rewards.all_fail_group_count);
        let saturated_fraction = saturated_groups as f64 / rewards.group_count as f64;
        if saturated_fraction > REWARD_MOST_GROUPS_FRACTION {
            warnings.push(format!(
                "most GRPO reward groups are all-pass or all-fail ({saturated_groups}/{} = {:.1}%); policy-gradient signal may be saturated, so consider `--no-policy-loss` with ECHO or collect harder data",
                rewards.group_count,
                saturated_fraction * 100.0
            ));
        }
    }

    if let (Some(mean), Some(stdev)) = (rewards.mean, rewards.stdev) {
        let variance = stdev * stdev;
        if mean >= saturation_threshold && variance <= low_variance_threshold {
            warnings.push(format!(
                "reward mean {mean:.4} is above saturation threshold {saturation_threshold:.4} while variance {variance:.3e} is below {low_variance_threshold:.3e}; consider `--no-policy-loss` or harder data"
            ));
        }
    }
    warnings
}

pub fn lora_delta_norm_summary_from_adapter(
    adapter_dir: &Path,
    alpha_over_rank: f64,
) -> Result<Vec<LoraDeltaNormSummary>> {
    let adapter_model = adapter_dir.join("adapter_model.safetensors");
    if !adapter_model.exists() {
        return Ok(Vec::new());
    }
    let tensors = candle_core::safetensors::load(&adapter_model, &Device::Cpu)
        .with_context(|| format!("load adapter tensors {}", adapter_model.display()))?;

    let mut pairs: BTreeMap<(usize, String), ProjectionPair> = BTreeMap::new();
    for (key, tensor) in tensors {
        let Some(parsed) = parse_peft_lora_key(&key) else {
            continue;
        };
        let norm = tensor_l2_norm(&tensor)
            .with_context(|| format!("compute l2 norm for adapter tensor {key}"))?;
        let pair = pairs.entry((parsed.layer, parsed.module)).or_default();
        match parsed.kind {
            LoraTensorKind::A => pair.a_l2 = Some(norm),
            LoraTensorKind::B => pair.b_l2 = Some(norm),
        }
    }

    let mut by_module: BTreeMap<String, ModuleNormAccumulator> = BTreeMap::new();
    for ((_layer, module), pair) in pairs {
        let (Some(a_l2), Some(b_l2)) = (pair.a_l2, pair.b_l2) else {
            continue;
        };
        by_module
            .entry(module)
            .or_default()
            .push(a_l2, b_l2, alpha_over_rank);
    }

    Ok(by_module
        .into_iter()
        .map(|(module, acc)| acc.finish(module))
        .collect())
}

pub fn lora_delta_norm_warnings(
    summaries: &[LoraDeltaNormSummary],
    alpha_over_rank: f64,
) -> Vec<String> {
    let finite: Vec<&LoraDeltaNormSummary> = summaries
        .iter()
        .filter(|summary| summary.delta_l2_upper_bound_max.is_finite())
        .collect();
    if finite.is_empty() {
        return Vec::new();
    }

    let mut warnings = Vec::new();
    let max_delta = finite
        .iter()
        .map(|summary| summary.delta_l2_upper_bound_max)
        .fold(0.0_f64, f64::max);
    if max_delta <= LORA_DELTA_NEAR_ZERO_EPSILON {
        warnings.push(format!(
            "all LoRA delta norms are near zero (max_delta_l2_upper_bound={max_delta:.3e}, threshold={LORA_DELTA_NEAR_ZERO_EPSILON:.3e}); adapter weights may not have moved enough to affect inference"
        ));
    }

    let scale = alpha_over_rank.abs().max(f64::MIN_POSITIVE);
    let extreme_threshold = scale * LORA_DELTA_EXTREME_SCALE_MULTIPLIER;
    for summary in finite {
        if summary.delta_l2_upper_bound_max > extreme_threshold {
            warnings.push(format!(
                "LoRA delta norm for module {} is extreme relative to initialized scale (delta_l2_upper_bound_max={:.3e}, threshold={:.3e}, alpha_over_rank={:.3e})",
                summary.module,
                summary.delta_l2_upper_bound_max,
                extreme_threshold,
                alpha_over_rank
            ));
        }
    }
    warnings
}

fn detect_kiln_source(env_config_hash: Option<String>) -> KilnSourceReceipt {
    let repo_root = std::env::var("KILN_REPO_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .map(Path::to_path_buf)
        });

    let (git_commit, git_dirty, git_source) = if let Some(root) = repo_root.as_deref() {
        let commit = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .filter(|s| !s.is_empty());
        let dirty = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| !out.stdout.is_empty());
        let source = commit.as_ref().map(|_| root.display().to_string());
        (commit, dirty, source)
    } else {
        (None, None, None)
    };

    KilnSourceReceipt {
        git_commit: git_commit.or_else(|| Some(crate::replay::kiln_commit())),
        git_dirty,
        git_source,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        env_config_hash,
    }
}

fn detect_model_path() -> Option<String> {
    ["KILN_MODEL_PATH", "MODEL_PATH", "HF_MODEL_PATH"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok())
        .filter(|value| !value.trim().is_empty())
}

fn variance_histogram(variances: &[f64]) -> Vec<HistogramBucket> {
    let specs = [
        ("zero", Some(0.0), Some(0.0)),
        ("tiny", Some(f64::MIN_POSITIVE), Some(1e-6)),
        ("low", Some(1e-6), Some(0.01)),
        ("medium", Some(0.01), Some(0.25)),
        ("high", Some(0.25), Some(1.0)),
        ("extreme", Some(1.0), None),
    ];
    specs
        .into_iter()
        .map(|(label, min, max)| {
            let count = variances
                .iter()
                .filter(|value| histogram_contains(**value, min, max))
                .count();
            HistogramBucket {
                label: label.to_string(),
                min_inclusive: min,
                max_inclusive: max,
                count,
            }
        })
        .collect()
}

fn histogram_contains(value: f64, min: Option<f64>, max: Option<f64>) -> bool {
    if value == 0.0 {
        return min == Some(0.0) && max == Some(0.0);
    }
    if let Some(min) = min {
        if value <= min {
            return false;
        }
    }
    if let Some(max) = max {
        if value > max {
            return false;
        }
    }
    true
}

fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64
}

pub(crate) fn tensor_l2_norm(tensor: &Tensor) -> Result<f64> {
    let sum_sq = tensor
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok((sum_sq as f64).sqrt())
}

#[derive(Debug, Default, Clone)]
pub struct LoraGradNormAccumulator {
    by_module: BTreeMap<String, GradNormAccumulator>,
}

impl LoraGradNormAccumulator {
    pub fn observe(&mut self, module: impl Into<String>, norm: f64) {
        if !norm.is_finite() {
            return;
        }
        self.by_module.entry(module.into()).or_default().push(norm);
    }

    pub fn finish(&self) -> Vec<LoraGradNormSummary> {
        self.by_module
            .iter()
            .map(|(module, acc)| acc.finish(module.clone()))
            .collect()
    }
}

#[derive(Debug, Default, Clone)]
struct GradNormAccumulator {
    sample_count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl GradNormAccumulator {
    fn push(&mut self, norm: f64) {
        if self.sample_count == 0 {
            self.min = norm;
            self.max = norm;
        } else {
            self.min = self.min.min(norm);
            self.max = self.max.max(norm);
        }
        self.sample_count += 1;
        self.sum += norm;
    }

    fn finish(&self, module: String) -> LoraGradNormSummary {
        let denom = self.sample_count.max(1) as f64;
        LoraGradNormSummary {
            module,
            sample_count: self.sample_count,
            min: self.min,
            mean: self.sum / denom,
            max: self.max,
        }
    }
}

fn parse_peft_lora_key(key: &str) -> Option<ParsedLoraKey> {
    let parts: Vec<&str> = key.split('.').collect();
    let layer_pos = parts.iter().position(|part| *part == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse().ok()?;
    let lora_pos = parts
        .iter()
        .position(|part| *part == "lora_A" || *part == "lora_B")?;
    if parts.get(lora_pos + 1)? != &"weight" || lora_pos == 0 {
        return None;
    }
    let module = parts.get(lora_pos - 1)?.to_string();
    let kind = match parts[lora_pos] {
        "lora_A" => LoraTensorKind::A,
        "lora_B" => LoraTensorKind::B,
        _ => return None,
    };
    Some(ParsedLoraKey {
        layer,
        module,
        kind,
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct ParsedLoraKey {
    layer: usize,
    module: String,
    kind: LoraTensorKind,
}

#[derive(Clone, Copy)]
enum LoraTensorKind {
    A,
    B,
}

#[derive(Default)]
struct ProjectionPair {
    a_l2: Option<f64>,
    b_l2: Option<f64>,
}

#[derive(Default)]
struct ModuleNormAccumulator {
    pair_count: usize,
    a_l2_sum: f64,
    a_l2_max: f64,
    b_l2_sum: f64,
    b_l2_max: f64,
    delta_l2_upper_bound_sum: f64,
    delta_l2_upper_bound_max: f64,
}

impl ModuleNormAccumulator {
    fn push(&mut self, a_l2: f64, b_l2: f64, alpha_over_rank: f64) {
        let delta = a_l2 * b_l2 * alpha_over_rank;
        self.pair_count += 1;
        self.a_l2_sum += a_l2;
        self.a_l2_max = self.a_l2_max.max(a_l2);
        self.b_l2_sum += b_l2;
        self.b_l2_max = self.b_l2_max.max(b_l2);
        self.delta_l2_upper_bound_sum += delta;
        self.delta_l2_upper_bound_max = self.delta_l2_upper_bound_max.max(delta);
    }

    fn finish(self, module: String) -> LoraDeltaNormSummary {
        let denom = self.pair_count.max(1) as f64;
        LoraDeltaNormSummary {
            module,
            pair_count: self.pair_count,
            a_l2_mean: self.a_l2_sum / denom,
            a_l2_max: self.a_l2_max,
            b_l2_mean: self.b_l2_sum / denom,
            b_l2_max: self.b_l2_max,
            delta_l2_upper_bound_mean: self.delta_l2_upper_bound_sum / denom,
            delta_l2_upper_bound_max: self.delta_l2_upper_bound_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn train_receipt_success_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let receipt = TrainReceipt::new(
            "adapter-a",
            "sft",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(42),
            },
            serde_json::json!({"epochs": 1}),
        );
        let path = receipt.write_to_adapter_dir(dir.path())?;
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some(TRAIN_RECEIPT_FILENAME)
        );
        let loaded = TrainReceipt::read_from_adapter_dir(dir.path())?.expect("receipt exists");
        assert_eq!(loaded.schema_version, TRAIN_RECEIPT_SCHEMA_VERSION);
        assert_eq!(loaded.status, TrainReceiptStatus::Success);
        assert_eq!(loaded.adapter_name, "adapter-a");
        assert_eq!(loaded.hyperparameters.rank, 8);
        assert!(loaded.config_hashes.model_config_hash.is_some());
        assert_eq!(
            loaded.model.config_hash,
            loaded.config_hashes.model_config_hash
        );
        assert!(loaded.config_hashes.tokenizer_config_hash.is_some());
        assert_eq!(
            loaded.tokenizer.tokenizer_config_hash,
            loaded.config_hashes.tokenizer_config_hash
        );
        assert!(loaded.config_hashes.chat_template_hash.is_none());
        assert!(loaded.config_hashes.kiln_env_config_hash.is_some());
        assert_eq!(
            loaded.kiln.env_config_hash,
            loaded.config_hashes.kiln_env_config_hash
        );
        assert!(loaded.lora_grad_norms.is_empty());
        assert!(loaded.adapter_smoke_test.is_none());
        Ok(())
    }

    #[test]
    fn train_receipt_writes_adapter_canary_status_sidecar() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let mut receipt = TrainReceipt::new(
            "adapter-canary",
            "sft",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(7),
            },
            serde_json::json!({"adapter_smoke_test": true}),
        );
        receipt.adapter_smoke_test = Some(build_adapter_smoke_test_receipt(vec![
            AdapterSmokePromptReceipt {
                prompt: "In one short sentence, name a primary color:".to_string(),
                finite_logits: true,
                logit_delta_l2: Some(0.25),
                generated_text_different: true,
                base_output: "Blue".to_string(),
                adapter_output: "Red".to_string(),
                adapter_output_tokens: 1,
                adapter_output_chars: 3,
                base_generation_ms: 10,
                adapter_generation_ms: 11,
            },
            AdapterSmokePromptReceipt {
                prompt: "Return a compact JSON tool call for weather.".to_string(),
                finite_logits: true,
                logit_delta_l2: Some(0.10),
                generated_text_different: true,
                base_output: "{}".to_string(),
                adapter_output: r#"{"tool":"weather"}"#.to_string(),
                adapter_output_tokens: 4,
                adapter_output_chars: 18,
                base_generation_ms: 12,
                adapter_generation_ms: 13,
            },
        ]));

        receipt.write_to_adapter_dir(dir.path())?;
        let status = read_adapter_canary_status_from_adapter_dir(dir.path())?.unwrap();
        assert_eq!(status.status, AdapterCanaryState::Passed);
        assert!(status.passed);
        assert!(dir.path().join(ADAPTER_CANARY_STATUS_FILENAME).is_file());
        assert!(
            status
                .checks
                .iter()
                .any(|check| check.name == "simple_tool_call_shaped_prompt" && check.passed)
        );
        Ok(())
    }

    #[test]
    fn train_receipt_failed_status_is_stable() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let receipt = TrainReceipt::new(
            "adapter-b",
            "grpo",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "grpo".to_string(),
                rank: 4,
                alpha: 8.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-5,
                epochs: 1,
                seed: None,
            },
            serde_json::json!({}),
        )
        .mark_failed("no valid GRPO groups after tokenization");
        receipt.write_to_adapter_dir(dir.path())?;
        let json = std::fs::read_to_string(dir.path().join(TRAIN_RECEIPT_FILENAME))?;
        assert!(json.contains("\"status\": \"failed\""));
        assert!(json.contains("\"failure_reason\": \"zero_groups\""));
        assert!(json.contains("\"failure_message\": \"no valid GRPO groups after tokenization\""));
        Ok(())
    }

    #[test]
    fn training_failure_reason_classifier_covers_standard_reasons() {
        let cases = [
            (
                "parse GRPO JSONL dataset line 7: missing field `completions`",
                TrainFailureReason::DataSchemaError,
            ),
            (
                "load adapter tensors: safetensors deserialize failed",
                TrainFailureReason::AdapterLoadFailed,
            ),
            (
                "GRPO dry run: zero valid GRPO groups after filtering",
                TrainFailureReason::ZeroGroups,
            ),
            (
                "GRPO dry run: dataset has no action tokens after mask construction",
                TrainFailureReason::ZeroActionTokens,
            ),
            (
                "GRPO dry run: ECHO is enabled but env_mask is empty across all valid groups",
                TrainFailureReason::ZeroEnvTokens,
            ),
            (
                "grpo_train_jsonl: non-finite loss NaN at group 2",
                TrainFailureReason::NanLoss,
            ),
            ("CUDA error: out of memory", TrainFailureReason::Oom),
            (
                "base adapter tensor shape mismatch for q_proj",
                TrainFailureReason::ShapeMismatch,
            ),
            (
                "read base adapter config /tmp/missing/adapter_config.json",
                TrainFailureReason::BaseAdapterMissing,
            ),
            (
                "unsafe LoRA scaling: alpha/rank = 3.000 exceeds the default limit",
                TrainFailureReason::UnsafeLoraScale,
            ),
        ];

        for (message, reason) in cases {
            assert_eq!(classify_training_failure(message), reason, "{message}");
        }
    }

    #[test]
    fn lora_grad_norm_accumulator_summarizes_by_module() {
        let mut acc = LoraGradNormAccumulator::default();
        acc.observe("q_proj", 3.0);
        acc.observe("q_proj", 5.0);
        acc.observe("q_proj", f64::NAN);
        acc.observe("down_proj", 2.0);

        let summaries = acc.finish();
        assert_eq!(summaries.len(), 2);
        let down = summaries
            .iter()
            .find(|summary| summary.module == "down_proj")
            .expect("down_proj summary");
        assert_eq!(down.sample_count, 1);
        assert_eq!(down.min, 2.0);
        assert_eq!(down.mean, 2.0);
        assert_eq!(down.max, 2.0);

        let q = summaries
            .iter()
            .find(|summary| summary.module == "q_proj")
            .expect("q_proj summary");
        assert_eq!(q.sample_count, 2);
        assert_eq!(q.min, 3.0);
        assert_eq!(q.mean, 4.0);
        assert_eq!(q.max, 5.0);
    }

    #[test]
    fn lora_delta_norm_warnings_cover_near_zero_and_extreme() {
        let near_zero = vec![LoraDeltaNormSummary {
            module: "q_proj".to_string(),
            pair_count: 1,
            a_l2_mean: 1.0,
            a_l2_max: 1.0,
            b_l2_mean: 0.0,
            b_l2_max: 0.0,
            delta_l2_upper_bound_mean: 0.0,
            delta_l2_upper_bound_max: 0.0,
        }];
        let warnings = lora_delta_norm_warnings(&near_zero, 2.0);
        assert!(warnings.iter().any(|warning| warning.contains("near zero")));

        let extreme = vec![LoraDeltaNormSummary {
            module: "down_proj".to_string(),
            pair_count: 1,
            a_l2_mean: 1.0,
            a_l2_max: 1.0,
            b_l2_mean: 1.0,
            b_l2_max: 1.0,
            delta_l2_upper_bound_mean: 250.0,
            delta_l2_upper_bound_max: 250.0,
        }];
        let warnings = lora_delta_norm_warnings(&extreme, 2.0);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("down_proj") && warning.contains("extreme"))
        );
    }

    #[test]
    fn echo_activity_metrics_populate_receipt_env_ce_bounds() {
        let mut metrics = EchoActivityMetrics::default();
        metrics.observe_env_ce(Some(2.5));
        metrics.observe_env_ce(None);
        metrics.observe_env_ce(Some(f64::NAN));
        metrics.observe_env_ce(Some(1.25));

        let mut receipt = EchoReceipt {
            enabled: true,
            lambda: Some(0.05),
            env_mask_mode: Some("env_only".to_string()),
            warning_filter: Some(true),
            initial_env_ce: None,
            final_env_ce: None,
        };
        metrics.apply_to_echo_receipt(&mut receipt);

        assert_eq!(receipt.initial_env_ce, Some(2.5));
        assert_eq!(receipt.final_env_ce, Some(1.25));
        assert_eq!(metrics.measurements, 2);
    }

    #[test]
    fn token_count_receipt_records_warning_filter_before_after_counts() {
        let mut counts = TokenCountReceipt::default();
        counts.observe_completion(100, 10, 20, 70);

        assert_eq!(counts.action_tokens, 10);
        assert_eq!(counts.env_tokens, 20);
        assert_eq!(counts.env_tokens_after_warning_filter, 20);
        assert_eq!(counts.env_tokens_before_warning_filter, 70);
        assert_eq!(counts.warning_tokens_filtered, 50);
        assert_eq!(counts.context_tokens, 70);
        assert!(warning_filter_stripped_most_env_tokens(&counts));

        let mut total = TokenCountReceipt::default();
        total.add_from(&counts);
        assert_eq!(total, counts);
    }

    #[test]
    fn adapter_smoke_receipt_passes_when_effect_is_measurable() {
        let receipt = build_adapter_smoke_test_receipt(vec![AdapterSmokePromptReceipt {
            prompt: "Say hello.".to_string(),
            finite_logits: true,
            logit_delta_l2: Some(0.25),
            generated_text_different: true,
            base_output: "Hello".to_string(),
            adapter_output: "Hi".to_string(),
            adapter_output_tokens: 1,
            adapter_output_chars: 2,
            base_generation_ms: 10,
            adapter_generation_ms: 11,
        }]);

        assert!(receipt.enabled);
        assert!(receipt.passed);
        assert!(receipt.warnings.is_empty());
        assert_eq!(
            receipt.prompt_diagnostics[0].outcome,
            AdapterSmokePromptDiagnosis::LogitsChangedTextChanged
        );
    }

    #[test]
    fn adapter_smoke_receipt_warns_on_no_effect() {
        let receipt = build_adapter_smoke_test_receipt(vec![AdapterSmokePromptReceipt {
            prompt: "Say hello.".to_string(),
            finite_logits: true,
            logit_delta_l2: Some(0.0),
            generated_text_different: false,
            base_output: "Hello".to_string(),
            adapter_output: "Hello".to_string(),
            adapter_output_tokens: 1,
            adapter_output_chars: 5,
            base_generation_ms: 10,
            adapter_generation_ms: 11,
        }]);

        assert!(!receipt.passed);
        assert!(
            receipt
                .warnings
                .iter()
                .any(|warning| warning.contains("no measurable adapter effect"))
        );
        assert!(
            receipt
                .notes
                .iter()
                .any(|note| note.contains("lora_grad_norms"))
        );
        assert_eq!(
            receipt.prompt_diagnostics[0].outcome,
            AdapterSmokePromptDiagnosis::NoLogitChange
        );
    }

    #[test]
    fn failed_adapter_smoke_status_quarantines_adapter() {
        let receipt = build_adapter_smoke_test_receipt(vec![AdapterSmokePromptReceipt {
            prompt: "Return a compact JSON tool call for weather.".to_string(),
            finite_logits: true,
            logit_delta_l2: Some(0.0),
            generated_text_different: false,
            base_output: "{}".to_string(),
            adapter_output: "{}".to_string(),
            adapter_output_tokens: 1,
            adapter_output_chars: 2,
            base_generation_ms: 10,
            adapter_generation_ms: 5000,
        }]);
        let status =
            build_adapter_canary_status_receipt("adapter-failed", "2026-05-21T00:00:00Z", &receipt);

        assert_eq!(status.status, AdapterCanaryState::Quarantined);
        assert!(!status.passed);
        assert!(
            status
                .failure_reason
                .as_deref()
                .unwrap_or_default()
                .contains("no nonzero logit delta")
        );
        assert!(
            status
                .checks
                .iter()
                .any(|check| check.name == "latency_sanity" && !check.passed)
        );
    }

    #[test]
    fn adapter_smoke_receipt_notes_logits_changed_but_text_identical() {
        let receipt = build_adapter_smoke_test_receipt(vec![AdapterSmokePromptReceipt {
            prompt: "Say hello.".to_string(),
            finite_logits: true,
            logit_delta_l2: Some(0.25),
            generated_text_different: false,
            base_output: "Hello".to_string(),
            adapter_output: "Hello".to_string(),
            adapter_output_tokens: 1,
            adapter_output_chars: 5,
            base_generation_ms: 10,
            adapter_generation_ms: 11,
        }]);

        assert!(receipt.passed);
        assert!(receipt.warnings.is_empty());
        assert!(
            receipt
                .notes
                .iter()
                .any(|note| note.contains("deterministic argmax"))
        );
        assert_eq!(
            receipt.prompt_diagnostics[0].outcome,
            AdapterSmokePromptDiagnosis::LogitsChangedTextIdentical
        );
    }

    #[test]
    fn adapter_smoke_receipt_warns_on_non_finite_or_empty_output() {
        let receipt = build_adapter_smoke_test_receipt(vec![AdapterSmokePromptReceipt {
            prompt: "Say hello.".to_string(),
            finite_logits: false,
            logit_delta_l2: None,
            generated_text_different: true,
            base_output: "Hello".to_string(),
            adapter_output: String::new(),
            adapter_output_tokens: 0,
            adapter_output_chars: 0,
            base_generation_ms: 10,
            adapter_generation_ms: 11,
        }]);

        assert!(!receipt.passed);
        assert!(
            receipt
                .warnings
                .iter()
                .any(|warning| warning.contains("non-finite logits"))
        );
        assert!(
            receipt
                .warnings
                .iter()
                .any(|warning| warning.contains("empty adapter output"))
        );
        assert_eq!(
            receipt.prompt_diagnostics[0].outcome,
            AdapterSmokePromptDiagnosis::NonFiniteLogits
        );
    }

    #[test]
    fn sha256_json_value_is_prefixed_and_stable() {
        let a = sha256_json_value(&serde_json::json!({"x": 1, "y": [true, false]}));
        let b = sha256_json_value(&serde_json::json!({"x": 1, "y": [true, false]}));
        assert!(a.starts_with("sha256:"));
        assert_eq!(a, b);
    }

    #[test]
    fn reward_stats_include_variance_histogram() {
        let groups = [vec![1.0, 0.0], vec![0.25, 0.25], vec![2.0, -2.0]];
        let slices: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
        let stats = reward_stats_from_groups(slices);
        assert_eq!(stats.count, 6);
        assert!(stats.mean.unwrap().abs() > 0.0);
        assert_eq!(stats.min, Some(-2.0));
        assert_eq!(stats.max, Some(2.0));
        assert_eq!(stats.group_count, 3);
        assert_eq!(stats.degenerate_group_count, 1);
        assert_eq!(stats.group_variance_histogram.len(), 6);
        assert_eq!(
            stats
                .group_variance_histogram
                .iter()
                .map(|b| b.count)
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn reward_diagnostics_warn_on_degenerate_and_saturated_rewards() {
        let groups = [vec![1.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]];
        let slices: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
        let stats = reward_stats_from_groups_with_threshold(slices, 0.95);
        assert_eq!(stats.group_count, 3);
        assert_eq!(stats.all_pass_group_count, 2);
        assert_eq!(stats.all_fail_group_count, 1);

        let warnings = reward_diagnostic_warnings(&stats, 0.95, 1e-4);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("all-pass or all-fail"))
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("`--no-policy-loss`"))
        );

        let saturated_groups = [vec![1.0, 0.99], vec![0.98, 1.0]];
        let slices: Vec<&[f64]> = saturated_groups.iter().map(Vec::as_slice).collect();
        let stats = reward_stats_from_groups_with_threshold(slices, 0.95);
        let warnings = reward_diagnostic_warnings(&stats, 0.95, 1e-3);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("reward mean"))
        );
    }

    fn minimal_tokenizer() -> Result<KilnTokenizer> {
        let json = br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]"
            }
        }"#;
        KilnTokenizer::from_bytes(json).map_err(|err| anyhow::anyhow!("{err}"))
    }
}

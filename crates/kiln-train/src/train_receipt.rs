//! Stable training receipts for GRPO/SFT adapter runs.
//!
//! This is intentionally separate from `receipt.json`: `receipt.json` is the
//! older high-level audit artifact used by distillation recipes,
//! while `train_receipt.json` is the machine-readable forensic record that cap
//! scripts can parse without scraping logs.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
// NOTE(#1082 Wave E4): the `cd_types` facade now resolves bare
// `Tensor` / `DType` to **kt** (matching the workspace post-flip
// convention). The candle `tensor_l2_norm(&candle_core::Tensor)` helper
// and its candle-parity test were DELETED in the candle drop — the
// trainer's candle-autograd gradient-norm fallback is gone (`GradSource`
// has no candle variant), so every grad-norm / adapter-norm caller is
// now kt-native via `tensor_l2_norm_kt`.
use kiln_core::config::ModelConfig;
use kiln_core::config_hashes::ConfigHashes;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_tensor as kt;
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
pub const REWARD_SATURATION_RECOMMENDATION: &str = "policy-gradient may be harmful; collect harder tasks, use stronger rubric gates, switch to OPD/teacher distillation, or use `--no-policy-loss` with ECHO";
pub const GRPO_POLICY_AUDIT_SCHEMA_V1: &str = "kiln.grpo-policy-audit.v1";
const GRPO_BEHAVIOR_SOURCE_SCHEMA_V1: &str = "kiln.grpo-behavior-source.v1";
const GRPO_BEHAVIOR_SOURCE_MANIFEST_SCHEMA_V1: &str = "kiln.grpo-behavior-source-manifest.v1";

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
    /// The operator cancelled the running job (DELETE
    /// /v1/train/queue/{id}) — the trainer aborted cooperatively at a
    /// step boundary.
    Cancelled,
    /// The loss composition asked for a term the kt-tape path cannot
    /// train (no_policy_loss / reserved OPD slot) — see
    /// `LossConfig::validate_for_kt_tape`.
    UnsupportedLossConfig,
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
            Self::Cancelled => "cancelled",
            Self::UnsupportedLossConfig => "unsupported_loss_config",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opd: Option<OpdReceipt>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_weight_shard_manifest: Option<kiln_core::model_provenance::BaseWeightShardManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerReceipt {
    /// Backward-compatible combined hash of tokenizer JSON plus chat template.
    pub config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_chat_template_hash: Option<String>,
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
    /// Semantic OpenEnv corpus identity when every admitted GRPO completion
    /// carries validated OpenEnv episode provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openenv: Option<crate::OpenEnvTrainingDataProvenanceV1>,
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
    /// True when example order was shuffled per epoch (SFT). Recorded so a
    /// receipt records the shuffle selector. Reconstructing the exact order
    /// also requires identical input data and ordering code.
    #[serde(default)]
    pub shuffle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoReceipt {
    pub kl_coeff: f64,
    pub clip_epsilon: f64,
    pub clip_eps_high: Option<f64>,
    #[serde(default = "crate::default_cispo_max_weight")]
    pub cispo_max_weight: f64,
    pub dynamic_sampling: bool,
    pub dynamic_groups_filtered: usize,
    pub advantage_mode: serde_json::Value,
    pub loss_aggregation: serde_json::Value,
    pub kl_estimator: serde_json::Value,
    pub is_level: serde_json::Value,
    #[serde(default)]
    pub behavior_policy: serde_json::Value,
    #[serde(default, alias = "reference_policy")]
    pub kl_reference_policy: serde_json::Value,
    pub entropy_aware_kl_quantile: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_audit: Option<GrpoPolicyAuditReceipt>,
}

/// Versioned observed-policy diagnostics for a GRPO run.
///
/// Importance ratios are always relative to the configured behavior policy;
/// KL values are always relative to the independently configured frozen
/// reference. Keeping them in distinct records prevents receipts from
/// accidentally presenting one denominator as the other.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoPolicyAuditReceipt {
    pub schema: String,
    pub importance_sampling: GrpoImportanceSamplingMetricsReceipt,
    pub kl_reference: GrpoKlReferenceMetricsReceipt,
    pub recorded_provenance: GrpoRecordedProvenanceReceipt,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GrpoImportanceSamplingMetricsReceipt {
    /// `token` for token PPO/CISPO, `sequence` for GSPO.
    pub ratio_scope: Option<String>,
    pub action_tokens: u64,
    pub ratio_observations: u64,
    pub mean_ratio: Option<f64>,
    pub min_ratio: Option<f64>,
    pub max_ratio: Option<f64>,
    /// Ratios below the PPO/GSPO lower bound. Always zero for CISPO, which
    /// deliberately has no lower importance-weight floor.
    pub below_clip_count: u64,
    /// Ratios above the PPO/GSPO upper bound or the absolute CISPO weight cap.
    pub above_clip_count: u64,
    /// Fraction outside the two-sided PPO/GSPO interval, or above the
    /// upper-only CISPO cap.
    pub outside_clip_fraction: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GrpoKlReferenceMetricsReceipt {
    pub token_observations: u64,
    pub entropy_mask_applied_tokens: u64,
    pub mean_policy_reference_log_ratio: Option<f64>,
    /// Configured K1/K3 value before the optional entropy-aware mask.
    pub mean_estimator: Option<f64>,
    /// Configured K1/K3 value after masking, still normalized by all action
    /// tokens so it matches the loss contribution before `kl_coeff`.
    pub mean_masked_estimator: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GrpoRecordedProvenanceReceipt {
    pub completion_count: u64,
    pub sampled_action_tokens: u64,
    pub forced_action_tokens: u64,
    pub unique_behavior_sources: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_source_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub behavior_sources: Vec<GrpoRecordedBehaviorSourceReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoRecordedBehaviorSourceReceipt {
    pub behavior_source_sha256: String,
    pub completion_count: u64,
    pub sampled_action_tokens: u64,
    pub forced_action_tokens: u64,
    pub behavior_policy: crate::RolloutBehaviorPolicyIdentityV1,
    pub tokenizer: crate::RolloutTokenizerIdentityV1,
    pub template_invocation: crate::RolloutChatTemplateInvocationV1,
    pub sampling: crate::RolloutSamplingConfigV1,
    pub generation_backend: String,
}

#[derive(Serialize)]
struct GrpoBehaviorSourceIdentityV1<'a> {
    schema: &'static str,
    behavior_policy: &'a crate::RolloutBehaviorPolicyIdentityV1,
    tokenizer: &'a crate::RolloutTokenizerIdentityV1,
    template_invocation: &'a crate::RolloutChatTemplateInvocationV1,
    sampling: &'a crate::RolloutSamplingConfigV1,
    generation_backend: &'a str,
}

#[derive(Serialize)]
struct GrpoBehaviorSourceManifestV1<'a> {
    schema: &'static str,
    sources: &'a [GrpoRecordedBehaviorSourceReceipt],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GrpoRecordedBehaviorSourceAccumulator {
    receipt: GrpoRecordedBehaviorSourceReceipt,
}

/// Compact behavior-source identity retained beside a tokenized completion.
/// It deliberately excludes the rollout's potentially long token arrays.
#[derive(Debug, Clone)]
pub(crate) struct GrpoRecordedBehaviorSourceObservation {
    identity_sha256: String,
    sampled_action_tokens: u64,
    forced_action_tokens: u64,
    behavior_policy: crate::RolloutBehaviorPolicyIdentityV1,
    tokenizer: crate::RolloutTokenizerIdentityV1,
    template_invocation: crate::RolloutChatTemplateInvocationV1,
    sampling: crate::RolloutSamplingConfigV1,
    generation_backend: String,
}

impl GrpoRecordedBehaviorSourceObservation {
    pub(crate) fn from_provenance(provenance: &crate::RolloutProvenanceV1) -> Result<Self> {
        provenance
            .validate()
            .map_err(anyhow::Error::msg)
            .context("validate GRPO policy-audit rollout provenance")?;
        let identity = GrpoBehaviorSourceIdentityV1 {
            schema: GRPO_BEHAVIOR_SOURCE_SCHEMA_V1,
            behavior_policy: &provenance.behavior_policy,
            tokenizer: &provenance.tokenizer,
            template_invocation: &provenance.template_invocation,
            sampling: &provenance.sampling,
            generation_backend: &provenance.generation_backend,
        };
        let encoded = serde_json::to_vec(&identity).context("serialize GRPO behavior source")?;
        let sampled_action_tokens = provenance.sampled_action_tokens().count() as u64;
        Ok(Self {
            identity_sha256: kiln_core::config_hashes::sha256_bytes(&encoded),
            sampled_action_tokens,
            forced_action_tokens: provenance.action_tokens.len() as u64 - sampled_action_tokens,
            behavior_policy: provenance.behavior_policy.clone(),
            tokenizer: provenance.tokenizer.clone(),
            template_invocation: provenance.template_invocation.clone(),
            sampling: provenance.sampling.clone(),
            generation_backend: provenance.generation_backend.clone(),
        })
    }
}

/// Accumulates receipt-grade GRPO diagnostics without retaining per-token
/// values. The trainer feeds it selected policy log-probabilities already
/// computed by the loss path, so this contract requires no second model
/// forward.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct GrpoPolicyAuditAccumulator {
    ratio_scope: Option<String>,
    action_tokens: u64,
    ratio_observations: u64,
    ratio_sum: f64,
    ratio_min: Option<f64>,
    ratio_max: Option<f64>,
    below_clip_count: u64,
    above_clip_count: u64,
    kl_token_observations: u64,
    kl_mask_applied_tokens: u64,
    kl_log_ratio_sum: f64,
    kl_estimator_sum: f64,
    kl_masked_estimator_sum: f64,
    recorded_completions: u64,
    recorded_sampled_actions: u64,
    recorded_forced_actions: u64,
    behavior_sources: BTreeMap<String, GrpoRecordedBehaviorSourceAccumulator>,
}

impl GrpoPolicyAuditAccumulator {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe_policy_values(
        &mut self,
        policy_log_probs: &[f32],
        behavior_log_probs: Option<&[f32]>,
        kl_reference_log_probs: Option<&[f32]>,
        is_level: crate::IsLevel,
        clip_low: f64,
        clip_high: f64,
        kl_estimator: crate::KlEstimator,
        entropy_aware_kl_quantile: Option<f32>,
    ) -> Result<()> {
        if policy_log_probs.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(
            policy_log_probs.iter().all(|value| value.is_finite()),
            "GRPO policy audit received a non-finite policy log-probability"
        );
        anyhow::ensure!(
            clip_low.is_finite() && clip_high.is_finite(),
            "GRPO policy audit received non-finite clip bounds"
        );

        let importance_log_ratios = match behavior_log_probs {
            Some(behavior) => {
                anyhow::ensure!(
                    behavior.len() == policy_log_probs.len(),
                    "GRPO policy audit behavior length {} differs from policy length {}",
                    behavior.len(),
                    policy_log_probs.len()
                );
                anyhow::ensure!(
                    behavior.iter().all(|value| value.is_finite()),
                    "GRPO policy audit received a non-finite behavior log-probability"
                );
                policy_log_probs
                    .iter()
                    .zip(behavior)
                    .map(|(&policy, &old)| f64::from(policy - old))
                    .collect::<Vec<_>>()
            }
            // Explicit no-importance-correction mode fixes the ratio at one;
            // it must not use the KL reference as an implicit denominator.
            None => vec![0.0; policy_log_probs.len()],
        };
        let scope = match is_level {
            crate::IsLevel::Sequence => "sequence",
            crate::IsLevel::Token | crate::IsLevel::Cispo => "token",
        };
        let ratio_scope_matches = match self.ratio_scope.as_deref() {
            None => true,
            Some(current) => current == scope,
        };
        anyhow::ensure!(
            ratio_scope_matches,
            "GRPO policy audit ratio scope changed within one run"
        );
        self.ratio_scope = Some(scope.to_string());
        self.action_tokens = self
            .action_tokens
            .saturating_add(policy_log_probs.len() as u64);

        let (lower_clip, upper_clip) = match is_level {
            crate::IsLevel::Cispo => (None, clip_high),
            crate::IsLevel::Token | crate::IsLevel::Sequence => {
                (Some(1.0 - clip_low), 1.0 + clip_high)
            }
        };
        anyhow::ensure!(
            upper_clip.is_finite() && upper_clip > 0.0,
            "GRPO policy audit received an invalid upper clip bound {upper_clip}"
        );

        if is_level == crate::IsLevel::Sequence {
            let mean_log_ratio =
                importance_log_ratios.iter().sum::<f64>() / importance_log_ratios.len() as f64;
            self.observe_ratio(mean_log_ratio.exp(), lower_clip, upper_clip)?;
        } else {
            for log_ratio in importance_log_ratios {
                self.observe_ratio(log_ratio.exp(), lower_clip, upper_clip)?;
            }
        }

        if let Some(reference) = kl_reference_log_probs {
            anyhow::ensure!(
                reference.len() == policy_log_probs.len(),
                "GRPO policy audit KL-reference length {} differs from policy length {}",
                reference.len(),
                policy_log_probs.len()
            );
            anyhow::ensure!(
                reference.iter().all(|value| value.is_finite()),
                "GRPO policy audit received a non-finite KL-reference log-probability"
            );
            let mask = entropy_aware_kl_mask_host(policy_log_probs, entropy_aware_kl_quantile);
            for ((&policy, &reference), apply) in policy_log_probs.iter().zip(reference).zip(mask) {
                let log_ratio = f64::from(policy - reference);
                let estimator = match kl_estimator {
                    crate::KlEstimator::None => 0.0,
                    crate::KlEstimator::K1 => log_ratio,
                    crate::KlEstimator::K3 => (-log_ratio).exp() - 1.0 + log_ratio,
                };
                anyhow::ensure!(
                    log_ratio.is_finite() && estimator.is_finite(),
                    "GRPO policy audit produced a non-finite KL observation"
                );
                self.kl_token_observations = self.kl_token_observations.saturating_add(1);
                self.kl_log_ratio_sum += log_ratio;
                self.kl_estimator_sum += estimator;
                if apply {
                    self.kl_mask_applied_tokens = self.kl_mask_applied_tokens.saturating_add(1);
                    self.kl_masked_estimator_sum += estimator;
                }
            }
        }
        Ok(())
    }

    fn observe_ratio(
        &mut self,
        ratio: f64,
        lower_clip: Option<f64>,
        upper_clip: f64,
    ) -> Result<()> {
        anyhow::ensure!(
            ratio.is_finite(),
            "GRPO policy audit produced a non-finite importance ratio"
        );
        self.ratio_observations = self.ratio_observations.saturating_add(1);
        self.ratio_sum += ratio;
        self.ratio_min = Some(self.ratio_min.map_or(ratio, |current| current.min(ratio)));
        self.ratio_max = Some(self.ratio_max.map_or(ratio, |current| current.max(ratio)));
        if lower_clip.is_some_and(|lower| ratio < lower) {
            self.below_clip_count = self.below_clip_count.saturating_add(1);
        }
        if ratio > upper_clip {
            self.above_clip_count = self.above_clip_count.saturating_add(1);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn observe_provenance(
        &mut self,
        provenance: &crate::RolloutProvenanceV1,
    ) -> Result<()> {
        let observation = GrpoRecordedBehaviorSourceObservation::from_provenance(provenance)?;
        self.observe_recorded_behavior_source(&observation);
        Ok(())
    }

    pub(crate) fn observe_recorded_behavior_source(
        &mut self,
        observation: &GrpoRecordedBehaviorSourceObservation,
    ) {
        let entry = self
            .behavior_sources
            .entry(observation.identity_sha256.clone())
            .or_insert_with(|| GrpoRecordedBehaviorSourceAccumulator {
                receipt: GrpoRecordedBehaviorSourceReceipt {
                    behavior_source_sha256: observation.identity_sha256.clone(),
                    completion_count: 0,
                    sampled_action_tokens: 0,
                    forced_action_tokens: 0,
                    behavior_policy: observation.behavior_policy.clone(),
                    tokenizer: observation.tokenizer.clone(),
                    template_invocation: observation.template_invocation.clone(),
                    sampling: observation.sampling.clone(),
                    generation_backend: observation.generation_backend.clone(),
                },
            });
        entry.receipt.completion_count = entry.receipt.completion_count.saturating_add(1);
        entry.receipt.sampled_action_tokens = entry
            .receipt
            .sampled_action_tokens
            .saturating_add(observation.sampled_action_tokens);
        entry.receipt.forced_action_tokens = entry
            .receipt
            .forced_action_tokens
            .saturating_add(observation.forced_action_tokens);
        self.recorded_completions = self.recorded_completions.saturating_add(1);
        self.recorded_sampled_actions = self
            .recorded_sampled_actions
            .saturating_add(observation.sampled_action_tokens);
        self.recorded_forced_actions = self
            .recorded_forced_actions
            .saturating_add(observation.forced_action_tokens);
    }

    pub(crate) fn finish(self) -> Result<GrpoPolicyAuditReceipt> {
        let ratio_mean =
            (self.ratio_observations > 0).then(|| self.ratio_sum / self.ratio_observations as f64);
        let outside_clip = self.below_clip_count.saturating_add(self.above_clip_count);
        let outside_clip_fraction = (self.ratio_observations > 0)
            .then(|| outside_clip as f64 / self.ratio_observations as f64);
        let kl_mean = (self.kl_token_observations > 0)
            .then(|| self.kl_log_ratio_sum / self.kl_token_observations as f64);
        let estimator_mean = (self.kl_token_observations > 0)
            .then(|| self.kl_estimator_sum / self.kl_token_observations as f64);
        let masked_estimator_mean = (self.kl_token_observations > 0)
            .then(|| self.kl_masked_estimator_sum / self.kl_token_observations as f64);

        let behavior_sources = self
            .behavior_sources
            .into_values()
            .map(|source| source.receipt)
            .collect::<Vec<_>>();
        let behavior_source_manifest_sha256 = if behavior_sources.is_empty() {
            None
        } else {
            let manifest = GrpoBehaviorSourceManifestV1 {
                schema: GRPO_BEHAVIOR_SOURCE_MANIFEST_SCHEMA_V1,
                sources: &behavior_sources,
            };
            let encoded =
                serde_json::to_vec(&manifest).context("serialize GRPO behavior-source manifest")?;
            Some(kiln_core::config_hashes::sha256_bytes(&encoded))
        };

        Ok(GrpoPolicyAuditReceipt {
            schema: GRPO_POLICY_AUDIT_SCHEMA_V1.to_string(),
            importance_sampling: GrpoImportanceSamplingMetricsReceipt {
                ratio_scope: self.ratio_scope,
                action_tokens: self.action_tokens,
                ratio_observations: self.ratio_observations,
                mean_ratio: ratio_mean,
                min_ratio: self.ratio_min,
                max_ratio: self.ratio_max,
                below_clip_count: self.below_clip_count,
                above_clip_count: self.above_clip_count,
                outside_clip_fraction,
            },
            kl_reference: GrpoKlReferenceMetricsReceipt {
                token_observations: self.kl_token_observations,
                entropy_mask_applied_tokens: self.kl_mask_applied_tokens,
                mean_policy_reference_log_ratio: kl_mean,
                mean_estimator: estimator_mean,
                mean_masked_estimator: masked_estimator_mean,
            },
            recorded_provenance: GrpoRecordedProvenanceReceipt {
                completion_count: self.recorded_completions,
                sampled_action_tokens: self.recorded_sampled_actions,
                forced_action_tokens: self.recorded_forced_actions,
                unique_behavior_sources: behavior_sources.len(),
                behavior_source_manifest_sha256,
                behavior_sources,
            },
        })
    }
}

fn entropy_aware_kl_mask_host(policy_log_probs: &[f32], quantile: Option<f32>) -> Vec<bool> {
    let Some(q) = quantile.filter(|q| q.is_finite() && (0.0..1.0).contains(q)) else {
        return vec![true; policy_log_probs.len()];
    };
    let mut negative = policy_log_probs
        .iter()
        .map(|value| -f64::from(*value))
        .collect::<Vec<_>>();
    negative.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((q as f64) * policy_log_probs.len().saturating_sub(1) as f64).round() as usize;
    let threshold = negative[index.min(negative.len().saturating_sub(1))];
    policy_log_probs
        .iter()
        .map(|value| -f64::from(*value) >= threshold)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpdReceipt {
    pub training_mode: String,
    pub objective: String,
    pub loss_granularity: String,
    pub teacher_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_content_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_identity: Option<crate::TeacherIdentityV1>,
    pub top_k: Option<usize>,
    pub samples_per_prompt: usize,
    pub action_tokens: u64,
    pub env_tokens: u64,
    pub echo_combined: bool,
    pub echo_lambda: Option<f64>,
    pub initial_opd_loss: Option<f64>,
    pub final_opd_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EchoReceipt {
    /// Whether the env-CE term actually FIRED during this run — not
    /// whether it was requested. A requested-but-dropped term records
    /// `enabled: false` plus `dropped_reason`.
    pub enabled: bool,
    pub lambda: Option<f64>,
    pub env_mask_mode: Option<String>,
    pub warning_filter: Option<bool>,
    #[serde(default)]
    pub initial_env_ce: Option<f64>,
    #[serde(default)]
    pub final_env_ce: Option<f64>,
    /// Why a configured ECHO term did not contribute to the loss (e.g. no
    /// kt-tape gradient root post candle-drop, #1082). `None` when the
    /// term fired or was never requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_reason: Option<String>,
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
    /// SFT-only admission evidence. Legacy receipts and non-SFT modes omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sft_ingestion: Option<crate::sft_ingestion::SftIngestionReceipt>,
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
    /// Time spent waiting for server inference readers before a bounded GRPO
    /// GPU phase. Zero for direct trainer calls without server coordination.
    #[serde(default)]
    pub gpu_writer_wait_ms: f64,
    /// Time for which bounded GRPO phases held the server GPU writer.
    #[serde(default)]
    pub gpu_writer_held_ms: f64,
    /// Number of separately scheduled GRPO GPU phases.
    #[serde(default)]
    pub gpu_writer_acquisitions: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
    /// Backend-owned route used by this SFT run. Server jobs pin and revalidate
    /// it at admission; standalone jobs resolve it once before execution.
    /// Absent for non-SFT and legacy receipts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sft_loss_route: Option<kiln_model::backend::SftFlceLossRoute>,
    /// Complete process/backend envelope for the resident weights. Absent only
    /// for legacy receipts and explicitly synthetic or dry-run paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provenance: Option<kiln_core::execution_provenance::ExecutionProvenanceV1>,
    /// Concrete dtypes actually captured from trainable parameters and
    /// optimizer state after setup. Early preflight failures may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
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
    fn validate_envelope(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == TRAIN_RECEIPT_SCHEMA_VERSION,
            "unsupported train-receipt schema_version {}; expected {}",
            self.schema_version,
            TRAIN_RECEIPT_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.receipt_type == "kiln_train_receipt",
            "invalid train-receipt receipt_type {:?}; expected \"kiln_train_receipt\"",
            self.receipt_type
        );
        anyhow::ensure!(
            !self.adapter_name.trim().is_empty(),
            "train-receipt adapter_name must not be empty"
        );
        chrono::DateTime::parse_from_rfc3339(&self.produced_at)
            .context("train-receipt produced_at must be an RFC3339 timestamp")?;
        match self.status {
            TrainReceiptStatus::Success => {
                anyhow::ensure!(
                    self.failure_reason.is_none() && self.failure_message.is_none(),
                    "successful train receipt must not carry failure_reason or failure_message"
                );
            }
            TrainReceiptStatus::Failed => {
                anyhow::ensure!(
                    self.failure_reason
                        .as_deref()
                        .is_some_and(|reason| !reason.trim().is_empty()),
                    "failed train receipt must carry a non-empty failure_reason"
                );
            }
        }
        Ok(())
    }

    pub fn new(
        adapter_name: impl Into<String>,
        mode: impl Into<String>,
        model_config: &ModelConfig,
        tokenizer: &KilnTokenizer,
        hyperparameters: HyperparameterReceipt,
        config: serde_json::Value,
    ) -> Self {
        let mode = mode.into();
        let config_hashes = ConfigHashes::from_model_tokenizer(
            model_config,
            tokenizer,
            kiln_core::config_hashes::effective_config_hash(&config),
        );
        Self {
            schema_version: TRAIN_RECEIPT_SCHEMA_VERSION,
            receipt_type: "kiln_train_receipt".to_string(),
            adapter_name: adapter_name.into(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            status: TrainReceiptStatus::Success,
            failure_reason: None,
            failure_message: None,
            kiln: detect_kiln_source(config_hashes.effective_config_hash.clone()),
            model: ModelReceipt {
                path: detect_model_path(),
                config_hash: config_hashes.model_config_hash.clone(),
                base_weight_shard_manifest: None,
            },
            tokenizer: TokenizerReceipt {
                config_hash: tokenizer.config_sha256().ok(),
                tokenizer_config_hash: config_hashes.tokenizer_config_hash.clone(),
                chat_template_hash: config_hashes.chat_template_hash.clone(),
                training_chat_template_hash: (mode == "sft")
                    .then(|| config_hashes.training_chat_template_hash.clone())
                    .flatten(),
            },
            adapters: AdapterReceiptSet {
                base: AdapterFileReceipt::none(),
                output: AdapterFileReceipt::none(),
            },
            training_data: TrainingDataReceipt {
                source: mode,
                path: None,
                sha256: None,
                openenv: None,
            },
            hyperparameters,
            grpo: None,
            opd: None,
            echo: EchoReceipt::disabled(),
            no_policy_loss: false,
            data: DataStatsReceipt::default(),
            rewards: RewardStatsReceipt::default(),
            token_counts: TokenCountReceipt::default(),
            phase_timings: TrainingPhaseTimingsReceipt::default(),
            runtime: RuntimeReceipt {
                wall_clock_ms: 0,
                peak_vram_mib: None,
                sft_loss_route: None,
                execution_provenance: None,
                training_precision: None,
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

    /// Validate a parsed receipt without performing another filesystem read.
    /// Callers that already own bounded, integrity-checked bytes can therefore
    /// prove that validation and subsequent use cover the same content.
    pub fn validate(&self) -> Result<()> {
        self.validate_envelope()
            .context("validate train-receipt envelope")?;
        self.validate_training_chat_template_identity()
            .context("validate train-receipt training chat template identity")?;
        if let Some(ingestion) = self.data.sft_ingestion.as_ref() {
            ingestion
                .validate()
                .context("validate train-receipt SFT ingestion evidence")?;
        }
        self.validate_sft_ingestion_binding()
            .context("validate train-receipt SFT ingestion binding")?;
        if let Some(manifest) = self.model.base_weight_shard_manifest.as_ref() {
            manifest
                .validate()
                .context("validate train-receipt base-weight shard manifest")?;
        }
        if let Some(provenance) = self.runtime.execution_provenance.as_ref() {
            provenance
                .validate()
                .context("validate train-receipt execution provenance")?;
        }
        if let Some(precision) = self.runtime.training_precision.as_ref() {
            precision
                .validate()
                .context("validate train-receipt precision")?;
        }
        if let Some(openenv) = self.training_data.openenv.as_ref() {
            openenv
                .validate()
                .map_err(anyhow::Error::msg)
                .context("validate train-receipt OpenEnv training-data provenance")?;
        }
        Ok(())
    }

    pub fn write_to_adapter_dir(&self, adapter_dir: &Path) -> Result<PathBuf> {
        self.validate()?;
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
        crate::adapter_output::write_adapter_manifest_from_train_receipt(adapter_dir, self)?;
        Ok(path)
    }

    pub fn read_from_adapter_dir(adapter_dir: &Path) -> Result<Option<Self>> {
        let path = adapter_dir.join(TRAIN_RECEIPT_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read train receipt {}", path.display()))?;
        let receipt: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("deserialize train receipt {}", path.display()))?;
        receipt
            .validate_envelope()
            .with_context(|| format!("validate train-receipt envelope in {}", path.display()))?;
        receipt
            .validate_training_chat_template_identity()
            .with_context(|| {
                format!(
                    "validate training chat template identity in train receipt {}",
                    path.display()
                )
            })?;
        if let Some(ingestion) = receipt.data.sft_ingestion.as_ref() {
            ingestion.validate().with_context(|| {
                format!(
                    "validate SFT ingestion evidence in train receipt {}",
                    path.display()
                )
            })?;
        }
        receipt.validate_sft_ingestion_binding().with_context(|| {
            format!(
                "validate SFT ingestion binding in train receipt {}",
                path.display()
            )
        })?;
        if let Some(manifest) = receipt.model.base_weight_shard_manifest.as_ref() {
            manifest.validate().with_context(|| {
                format!(
                    "validate base-weight shard manifest in train receipt {}",
                    path.display()
                )
            })?;
        }
        if let Some(provenance) = receipt.runtime.execution_provenance.as_ref() {
            provenance.validate().with_context(|| {
                format!(
                    "validate execution provenance in train receipt {}",
                    path.display()
                )
            })?;
        }
        if let Some(precision) = receipt.runtime.training_precision.as_ref() {
            precision.validate().with_context(|| {
                format!(
                    "validate training precision in train receipt {}",
                    path.display()
                )
            })?;
        }
        if let Some(openenv) = receipt.training_data.openenv.as_ref() {
            openenv
                .validate()
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!(
                        "validate OpenEnv training-data provenance in train receipt {}",
                        path.display()
                    )
                })?;
        }
        Ok(Some(receipt))
    }

    fn validate_training_chat_template_identity(&self) -> Result<()> {
        let direct = self.tokenizer.training_chat_template_hash.as_deref();
        let config = self.config_hashes.training_chat_template_hash.as_deref();
        let execution = self
            .runtime
            .execution_provenance
            .as_ref()
            .and_then(|provenance| provenance.model.training_chat_template_sha256.as_deref());

        for (field, value) in [
            ("tokenizer.training_chat_template_hash", direct),
            ("config_hashes.training_chat_template_hash", config),
            (
                "runtime.execution_provenance.model.training_chat_template_sha256",
                execution,
            ),
        ] {
            if let Some(value) = value {
                validate_prefixed_sha256(field, value)?;
            }
        }
        for (left_name, left, right_name, right) in [
            (
                "tokenizer.training_chat_template_hash",
                direct,
                "config_hashes.training_chat_template_hash",
                config,
            ),
            (
                "tokenizer.training_chat_template_hash",
                direct,
                "runtime.execution_provenance.model.training_chat_template_sha256",
                execution,
            ),
            (
                "config_hashes.training_chat_template_hash",
                config,
                "runtime.execution_provenance.model.training_chat_template_sha256",
                execution,
            ),
        ] {
            if let (Some(left), Some(right)) = (left, right)
                && left != right
            {
                bail!("{left_name} ({left}) differs from {right_name} ({right})");
            }
        }
        Ok(())
    }

    fn validate_sft_ingestion_binding(&self) -> Result<()> {
        let Some(ingestion) = self.data.sft_ingestion.as_ref() else {
            return Ok(());
        };
        anyhow::ensure!(
            self.hyperparameters.mode == "sft",
            "SFT ingestion evidence is attached to mode {:?}",
            self.hyperparameters.mode
        );
        anyhow::ensure!(
            self.training_data.source == ingestion.source,
            "training_data.source differs from SFT ingestion source"
        );
        anyhow::ensure!(
            self.training_data.path == ingestion.source_locator,
            "training_data.path differs from SFT ingestion source locator"
        );
        anyhow::ensure!(
            self.training_data.sha256.as_deref() == Some(&ingestion.kept_corpus_sha256),
            "training_data.sha256 differs from SFT kept-corpus identity"
        );
        anyhow::ensure!(
            self.data.examples_read == ingestion.rows_read,
            "data.examples_read differs from SFT ingestion rows_read"
        );
        anyhow::ensure!(
            self.data.examples_filtered == ingestion.rows_rejected,
            "data.examples_filtered differs from SFT ingestion rows_rejected"
        );
        let configured_policy = self
            .config
            .get("invalid_row_policy")
            .context("SFT receipt config lacks invalid_row_policy")?;
        let configured_policy: crate::SftInvalidRowPolicy =
            serde_json::from_value(configured_policy.clone())
                .context("parse SFT receipt config invalid_row_policy")?;
        anyhow::ensure!(
            configured_policy == ingestion.invalid_row_policy,
            "receipt config invalid_row_policy differs from SFT ingestion evidence"
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn test_execution_provenance() -> kiln_core::execution_provenance::ExecutionProvenanceV1
{
    use std::collections::BTreeMap;

    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
        ExecutionProvenanceV1,
    };

    let hash = |byte: char| format!("sha256:{}", byte.to_string().repeat(64));
    ExecutionProvenanceV1::new(
        ExecutionBackendIdentity {
            name: "test".into(),
            device: "cpu".into(),
            numerical_runtime_sha256: hash('1'),
        },
        ExecutionBuildIdentity {
            package_version: env!("CARGO_PKG_VERSION").into(),
            target: "linux-x86_64".into(),
            executable_sha256: hash('2'),
            git_commit: Some("test-commit".into()),
            source_tree_sha256: Some(hash('3')),
            source_dirty: Some(false),
        },
        ExecutionModelIdentity {
            model_config_sha256: hash('4'),
            tokenizer_vocab_sha256: hash('5'),
            tokenizer_config_sha256: hash('6'),
            chat_template_sha256: Some(hash('7')),
            training_chat_template_sha256: Some(hash('8')),
        },
        ExecutionPrecisionIdentity {
            inference_dtype: "f32".into(),
            training_policy: "cpu_f32_reference".into(),
        },
        ExecutionKernelIdentity::new(
            BTreeMap::from([("kiln-train".into(), env!("CARGO_PKG_VERSION").into())]),
            Vec::new(),
        )
        .unwrap(),
        ExecutionConfigurationIdentity {
            effective_server_config_sha256: hash('8'),
            effective_environment_sha256: hash('9'),
        },
    )
    .unwrap()
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
    // Operator cancellation: match FIRST — the message is unambiguous and
    // nothing else should reclassify it.
    if lower.contains("cancelled by user") {
        return TrainFailureReason::Cancelled;
    }

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
    // Untrainable loss compositions (validate_for_kt_tape) — must match
    // BEFORE the zero-env-tokens patterns since both mention env tokens.
    if lower.contains("no_policy_loss (verifier-free")
        || lower.contains("not yet re-wired on the kt-tape path")
        || lower.contains("loss.opd composition on grpo is reserved")
    {
        return TrainFailureReason::UnsupportedLossConfig;
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
            dropped_reason: None,
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
    let digest: [u8; 32] = h.finalize().into();
    Ok(format_sha256_digest(&digest))
}

pub(crate) fn validate_prefixed_sha256(field: &str, value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        bail!("{field} must use the sha256:<64 lowercase hex> format");
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must use the sha256:<64 lowercase hex> format");
    }
    Ok(())
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    format_sha256_digest(&digest)
}

pub fn sha256_json_value(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
}

/// Encode an already-computed SHA-256 digest in the receipt wire format.
pub fn format_sha256_digest(digest: &[u8; 32]) -> String {
    format!("sha256:{}", hex_digest(digest))
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
                "most GRPO reward groups are all-pass or all-fail ({saturated_groups}/{} = {:.1}%); {}",
                rewards.group_count,
                saturated_fraction * 100.0,
                REWARD_SATURATION_RECOMMENDATION
            ));
        }
    }

    if let (Some(mean), Some(stdev)) = (rewards.mean, rewards.stdev) {
        let variance = stdev * stdev;
        if mean >= saturation_threshold && variance <= low_variance_threshold {
            warnings.push(format!(
                "reward mean {mean:.4} is above saturation threshold {saturation_threshold:.4} while variance {variance:.3e} is below {low_variance_threshold:.3e}; {}",
                REWARD_SATURATION_RECOMMENDATION
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
    // (#1082) Adapter L2-norm receipt path uses kt safetensors + kt L2 norm
    // computation. CPU-only, autograd-free diagnostic loop. The trainer's
    // gradient-norm path is also kt-native now (`tensor_l2_norm_kt`).
    let tensors = kt::safetensors::load_cpu(&adapter_model)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("load adapter tensors {}", adapter_model.display()))?;

    let mut pairs: BTreeMap<(usize, String), ProjectionPair> = BTreeMap::new();
    for (key, tensor) in tensors {
        let Some(parsed) = parse_peft_lora_key(&key) else {
            continue;
        };
        let norm = tensor_l2_norm_kt(&tensor)
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

/// kt-native LoRA / gradient L2 norm.
///
/// (#1082) The candle counterpart `tensor_l2_norm(&candle_core::Tensor)` was
/// deleted in the candle drop — every production grad-norm caller
/// (`observe_lora_grad_norms_from_kt_grad_store`, `accumulate_lora_grad_sum_sq`)
/// is kt-native, and `GradSource` no longer has a candle variant.
///
/// Used by [`lora_delta_norm_summary_from_adapter`] after its
/// safetensors loader migrated from `candle_core::safetensors::load`
/// to `kt::safetensors::load_cpu` (#1082). Casts to F32 first so the
/// accumulator preserves precision when the underlying adapter weight
/// is BF16/F16. Squaring and summation stay on the tensor's backend and only
/// the scalar sum is copied to CPU, so receipt collection does not require a
/// CPU-backed gradient tensor.
pub(crate) fn tensor_l2_norm_kt(tensor: &kt::Tensor) -> Result<f64> {
    anyhow::ensure!(
        tensor.elem_count() > 0,
        "cannot compute the L2 norm of an empty tensor"
    );
    let f32_t = if tensor.dtype() == kt::DType::F32 {
        tensor.clone()
    } else {
        kt::ops::cast::cast(tensor, kt::DType::F32)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("cast adapter tensor to f32 for l2 norm")?
    };
    let squared = kt::ops::mul(&f32_t, &f32_t)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("square tensor for L2 norm")?;
    let sum = kt::ops::sum_all(&squared)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("sum squared tensor for L2 norm")?;
    let sum = sum
        .to_device(kt::Device::Cpu)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("copy L2 squared sum to CPU")?
        .to_scalar::<f32>()?;
    anyhow::ensure!(
        sum.is_finite() && sum >= 0.0,
        "invalid L2 squared sum {sum}"
    );
    Ok(f64::from(sum).sqrt())
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraGradNormAccumulator {
    by_module: BTreeMap<String, GradNormAccumulator>,
}

impl LoraGradNormAccumulator {
    pub fn observe(&mut self, module: impl Into<String>, norm: f64) {
        // Device norm reductions are f32. Canonicalize back to that source
        // precision so host sqrt/transfer tails cannot make forensic receipts
        // differ by an information-free f64 ULP across equivalent runs.
        let norm = f64::from(norm as f32);
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

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GradNormAccumulator {
    sample_count: usize,
    #[serde(with = "f64_ieee_bits")]
    sum: f64,
    #[serde(with = "f64_ieee_bits")]
    min: f64,
    #[serde(with = "f64_ieee_bits")]
    max: f64,
}

mod f64_ieee_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{:016x}", value.to_bits()))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum WireValue {
            Bits(String),
            LegacyNumber(f64),
        }

        let value = match WireValue::deserialize(deserializer)? {
            WireValue::Bits(encoded) => {
                let digits = encoded.strip_prefix("0x").ok_or_else(|| {
                    serde::de::Error::custom("IEEE-754 f64 bits must start with 0x")
                })?;
                if digits.len() != 16 {
                    return Err(serde::de::Error::custom(
                        "IEEE-754 f64 bits must contain exactly 16 hex digits",
                    ));
                }
                let bits = u64::from_str_radix(digits, 16)
                    .map_err(|_| serde::de::Error::custom("invalid IEEE-754 f64 bits"))?;
                f64::from_bits(bits)
            }
            WireValue::LegacyNumber(value) => value,
        };
        if !value.is_finite() {
            return Err(serde::de::Error::custom(
                "gradient-norm accumulator values must be finite",
            ));
        }
        Ok(value)
    }
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

    fn base_weight_manifest() -> kiln_core::model_provenance::BaseWeightShardManifest {
        kiln_core::model_provenance::BaseWeightShardManifest::new(vec![
            kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                "model.safetensors",
                11,
                [0x42; 32],
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn full_sha256(fill: char) -> String {
        format!("sha256:{}", fill.to_string().repeat(64))
    }

    fn audit_provenance(adapter_revision_fill: char, seed: u64) -> crate::RolloutProvenanceV1 {
        crate::RolloutProvenanceV1::new(
            vec![1, 2, 3, 4],
            2,
            full_sha256('1'),
            full_sha256('2'),
            vec![
                crate::RolloutActionTokenV1::sampled(2, 3, -0.25),
                crate::RolloutActionTokenV1::forced(3, 4),
            ],
            crate::RolloutBehaviorPolicyIdentityV1 {
                served_model_id: "policy-model".to_string(),
                base_model_sha256: full_sha256('3'),
                adapter: Some(crate::RolloutAdapterIdentityV1 {
                    name: "policy-adapter".to_string(),
                    content_sha256: full_sha256(adapter_revision_fill),
                }),
                inference_config_sha256: full_sha256('4'),
                implementation: "kiln-test".to_string(),
            },
            crate::RolloutTokenizerIdentityV1 {
                vocab_sha256: full_sha256('5'),
                config_sha256: full_sha256('6'),
                chat_template_sha256: full_sha256('7'),
            },
            crate::RolloutSamplingConfigV1 {
                temperature: 0.9,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                max_tokens: 16,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                stop: Vec::new(),
                thinking_budget: Some(crate::RolloutThinkingBudgetV1 {
                    max_tokens: Some(8),
                    max_time_ms: None,
                    close_token_ids: vec![4],
                }),
            },
            seed,
            "rocm",
        )
        .unwrap()
    }

    fn assert_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-7,
            "actual {actual} != expected {expected}"
        );
    }

    #[test]
    fn grpo_policy_audit_keeps_behavior_ratios_and_kl_reference_distinct() -> Result<()> {
        let policy = [-1.0_f32, -2.0];
        let behavior = [-1.2_f32, -1.8];
        let kl_reference = [-0.7_f32, -2.3];
        let mut audit = GrpoPolicyAuditAccumulator::default();
        audit.observe_policy_values(
            &policy,
            Some(&behavior),
            Some(&kl_reference),
            crate::IsLevel::Token,
            0.2,
            0.2,
            crate::KlEstimator::K3,
            Some(0.5),
        )?;
        let receipt = audit.finish()?;

        assert_eq!(receipt.schema, GRPO_POLICY_AUDIT_SCHEMA_V1);
        let importance = receipt.importance_sampling;
        assert_eq!(importance.ratio_scope.as_deref(), Some("token"));
        assert_eq!(importance.action_tokens, 2);
        assert_eq!(importance.ratio_observations, 2);
        assert_eq!(importance.below_clip_count, 0);
        assert_eq!(importance.above_clip_count, 1);
        assert_near(
            importance.mean_ratio.unwrap(),
            (0.2_f64.exp() + (-0.2_f64).exp()) / 2.0,
        );
        assert_near(importance.outside_clip_fraction.unwrap(), 0.5);

        let kl = receipt.kl_reference;
        assert_eq!(kl.token_observations, 2);
        assert_eq!(kl.entropy_mask_applied_tokens, 1);
        assert_near(kl.mean_policy_reference_log_ratio.unwrap(), 0.0);
        let first = 0.3_f64.exp() - 1.0 - 0.3;
        let second = (-0.3_f64).exp() - 1.0 + 0.3;
        assert_near(kl.mean_estimator.unwrap(), (first + second) / 2.0);
        assert_near(kl.mean_masked_estimator.unwrap(), second / 2.0);
        Ok(())
    }

    #[test]
    fn grpo_policy_audit_accumulator_round_trips_strictly_for_resume() -> Result<()> {
        let mut audit = GrpoPolicyAuditAccumulator::default();
        audit.observe_policy_values(
            &[-1.0, -2.0],
            Some(&[-1.2, -1.8]),
            Some(&[-0.7, -2.3]),
            crate::IsLevel::Token,
            0.2,
            0.2,
            crate::KlEstimator::K3,
            Some(0.5),
        )?;

        let encoded = serde_json::to_value(&audit)?;
        let restored: GrpoPolicyAuditAccumulator = serde_json::from_value(encoded.clone())?;
        assert_eq!(restored, audit);

        let mut with_unknown = encoded;
        with_unknown
            .as_object_mut()
            .expect("audit checkpoint state must be an object")
            .insert("unknown".to_string(), serde_json::Value::Bool(true));
        assert!(
            serde_json::from_value::<GrpoPolicyAuditAccumulator>(with_unknown).is_err(),
            "resume state must reject unknown accumulator fields"
        );
        Ok(())
    }

    #[test]
    fn grpo_policy_audit_no_correction_fixes_ratio_without_deleting_kl() -> Result<()> {
        let mut audit = GrpoPolicyAuditAccumulator::default();
        audit.observe_policy_values(
            &[-1.0, -2.0],
            None,
            Some(&[-1.5, -2.5]),
            crate::IsLevel::Sequence,
            0.2,
            0.2,
            crate::KlEstimator::K1,
            None,
        )?;
        let receipt = audit.finish()?;

        assert_eq!(
            receipt.importance_sampling.ratio_scope.as_deref(),
            Some("sequence")
        );
        assert_eq!(receipt.importance_sampling.action_tokens, 2);
        assert_eq!(receipt.importance_sampling.ratio_observations, 1);
        assert_eq!(receipt.importance_sampling.mean_ratio, Some(1.0));
        assert_eq!(receipt.kl_reference.token_observations, 2);
        assert_eq!(receipt.kl_reference.mean_estimator, Some(0.5));
        assert_eq!(receipt.kl_reference.mean_masked_estimator, Some(0.5));
        Ok(())
    }

    #[test]
    fn grpo_policy_audit_cispo_reports_only_the_absolute_upper_cap() -> Result<()> {
        let mut audit = GrpoPolicyAuditAccumulator::default();
        audit.observe_policy_values(
            &[-2.0, -0.1],
            Some(&[-1.0, -1.0]),
            None,
            crate::IsLevel::Cispo,
            0.2,
            1.5,
            crate::KlEstimator::None,
            None,
        )?;
        let importance = audit.finish()?.importance_sampling;

        assert_eq!(importance.ratio_scope.as_deref(), Some("token"));
        assert_eq!(importance.ratio_observations, 2);
        assert_eq!(importance.below_clip_count, 0);
        assert_eq!(importance.above_clip_count, 1);
        assert_near(importance.min_ratio.unwrap(), (-1.0_f64).exp());
        assert_near(importance.max_ratio.unwrap(), 0.9_f64.exp());
        assert_near(importance.outside_clip_fraction.unwrap(), 0.5);
        Ok(())
    }

    #[test]
    fn grpo_policy_audit_behavior_manifest_is_exact_and_order_independent() -> Result<()> {
        let first = audit_provenance('8', 10);
        let same_source = audit_provenance('8', 11);
        let revised = audit_provenance('9', 12);

        let mut forward = GrpoPolicyAuditAccumulator::default();
        forward.observe_provenance(&first)?;
        forward.observe_provenance(&same_source)?;
        forward.observe_provenance(&revised)?;
        let forward = forward.finish()?.recorded_provenance;

        let mut reverse = GrpoPolicyAuditAccumulator::default();
        reverse.observe_provenance(&revised)?;
        reverse.observe_provenance(&same_source)?;
        reverse.observe_provenance(&first)?;
        let reverse = reverse.finish()?.recorded_provenance;

        assert_eq!(forward, reverse);
        assert_eq!(forward.completion_count, 3);
        assert_eq!(forward.sampled_action_tokens, 3);
        assert_eq!(forward.forced_action_tokens, 3);
        assert_eq!(forward.unique_behavior_sources, 2);
        assert!(forward.behavior_source_manifest_sha256.is_some());
        let counts = forward
            .behavior_sources
            .iter()
            .map(|source| source.completion_count)
            .collect::<Vec<_>>();
        assert!(counts.contains(&1));
        assert!(counts.contains(&2));
        assert_ne!(
            forward.behavior_sources[0].behavior_source_sha256,
            forward.behavior_sources[1].behavior_source_sha256
        );
        Ok(())
    }

    #[test]
    fn legacy_grpo_receipt_maps_reference_to_kl_and_leaves_behavior_unknown() {
        let receipt: GrpoReceipt = serde_json::from_value(serde_json::json!({
            "kl_coeff": 0.1,
            "clip_epsilon": 0.2,
            "clip_eps_high": null,
            "dynamic_sampling": true,
            "dynamic_groups_filtered": 0,
            "advantage_mode": "dr_grpo",
            "loss_aggregation": "token_level",
            "kl_estimator": "k1",
            "is_level": "token",
            "reference_policy": {"kind": "base_per_step"},
            "entropy_aware_kl_quantile": null
        }))
        .unwrap();
        assert_eq!(receipt.behavior_policy, serde_json::Value::Null);
        assert_eq!(receipt.cispo_max_weight, 5.0);
        assert_eq!(
            receipt.kl_reference_policy,
            serde_json::json!({"kind": "base_per_step"})
        );

        assert!(receipt.policy_audit.is_none());
        let wire = serde_json::to_value(&receipt).unwrap();
        assert!(wire.get("reference_policy").is_none());
        assert_eq!(wire["behavior_policy"], serde_json::Value::Null);
        assert_eq!(wire["cispo_max_weight"], 5.0);
    }

    #[test]
    fn train_receipt_success_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let mut receipt = TrainReceipt::new(
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
                shuffle: false,
            },
            serde_json::json!({"epochs": 1}),
        );
        let base_weights = base_weight_manifest();
        receipt.model.base_weight_shard_manifest = Some(base_weights.clone());
        let execution_provenance = test_execution_provenance();
        receipt.runtime.execution_provenance = Some(execution_provenance.clone());
        receipt.runtime.training_precision = Some(crate::checkpoint::TrainingCheckpointPrecision {
            parameter_dtype: "f32".into(),
            optimizer_state_dtype: "f32".into(),
            activation_dtype: "f32".into(),
            gradient_dtype: "f32".into(),
            stochastic_rounding: serde_json::json!({"mode": "round_to_nearest"}),
        });
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
        assert_eq!(
            loaded.model.base_weight_shard_manifest.as_ref(),
            Some(&base_weights)
        );
        assert_eq!(
            loaded
                .runtime
                .execution_provenance
                .as_ref()
                .map(|value| value.provenance_sha256.as_str()),
            Some(execution_provenance.provenance_sha256.as_str())
        );
        assert_eq!(
            loaded
                .runtime
                .training_precision
                .as_ref()
                .map(|value| value.parameter_dtype.as_str()),
            Some("f32")
        );
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
        assert!(loaded.config_hashes.effective_config_hash.is_some());
        assert_eq!(
            loaded.kiln.env_config_hash,
            loaded.config_hashes.effective_config_hash
        );
        assert!(loaded.lora_grad_norms.is_empty());
        assert!(loaded.adapter_smoke_test.is_none());

        let mut tampered_precision = serde_json::to_value(&loaded)?;
        tampered_precision["runtime"]["training_precision"]["parameter_dtype"] =
            serde_json::json!("");
        std::fs::write(
            dir.path().join(TRAIN_RECEIPT_FILENAME),
            serde_json::to_vec_pretty(&tampered_precision)?,
        )?;
        let error = TrainReceipt::read_from_adapter_dir(dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("validate training precision"));

        let mut tampered_execution = serde_json::to_value(&loaded)?;
        tampered_execution["runtime"]["execution_provenance"]["backend"]["device"] =
            serde_json::json!("tampered");
        std::fs::write(
            dir.path().join(TRAIN_RECEIPT_FILENAME),
            serde_json::to_vec_pretty(&tampered_execution)?,
        )?;
        let error = TrainReceipt::read_from_adapter_dir(dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("validate execution provenance"));
        Ok(())
    }

    #[test]
    fn train_receipt_reader_rejects_invalid_envelope_identity_and_status() -> Result<()> {
        let dir = tempdir()?;
        let receipt = TrainReceipt::new(
            "adapter-a",
            "sft",
            &ModelConfig::qwen3_5_4b(),
            &minimal_tokenizer()?,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(42),
                shuffle: false,
            },
            serde_json::json!({"epochs": 1}),
        );
        let path = dir.path().join(TRAIN_RECEIPT_FILENAME);
        let valid = serde_json::to_value(receipt)?;

        for (field, value, expected) in [
            (
                "schema_version",
                serde_json::json!(TRAIN_RECEIPT_SCHEMA_VERSION + 1),
                "unsupported train-receipt schema_version",
            ),
            (
                "receipt_type",
                serde_json::json!("other_receipt"),
                "invalid train-receipt receipt_type",
            ),
            (
                "produced_at",
                serde_json::json!("not-a-timestamp"),
                "must be an RFC3339 timestamp",
            ),
        ] {
            let mut changed = valid.clone();
            changed[field] = value;
            std::fs::write(&path, serde_json::to_vec_pretty(&changed)?)?;
            let error = format!(
                "{:#}",
                TrainReceipt::read_from_adapter_dir(dir.path()).unwrap_err()
            );
            assert!(error.contains(expected), "{error}");
        }

        let mut inconsistent_success = valid;
        inconsistent_success["failure_reason"] = serde_json::json!("training_error");
        std::fs::write(&path, serde_json::to_vec_pretty(&inconsistent_success)?)?;
        let error = format!(
            "{:#}",
            TrainReceipt::read_from_adapter_dir(dir.path()).unwrap_err()
        );
        assert!(
            error.contains("successful train receipt must not carry"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn sft_ingestion_binding_round_trips_and_rejects_cross_field_tampering() -> Result<()> {
        let dir = tempdir()?;
        let tokenizer = minimal_tokenizer()?.with_chat_template(
            "{% for message in messages %}{{ message.content }} {% endfor %}".to_string(),
        );
        let prepared = crate::prepare_sft_examples(
            vec![
                crate::SftExample {
                    messages: vec![
                        crate::ChatMessage::new("user", "hello"),
                        crate::ChatMessage::new("assistant", "hello"),
                    ],
                },
                crate::SftExample { messages: vec![] },
            ],
            &tokenizer,
            crate::SftInvalidRowPolicy::Skip,
            "inline",
            None,
        )?;
        let config = crate::SftConfig {
            invalid_row_policy: crate::SftInvalidRowPolicy::Skip,
            ..Default::default()
        };
        let mut receipt = TrainReceipt::new(
            "adapter-ingestion",
            "sft",
            &ModelConfig::qwen3_5_4b(),
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(7),
                shuffle: false,
            },
            serde_json::to_value(&config)?,
        );
        receipt.training_data = TrainingDataReceipt {
            source: prepared.ingestion.source.clone(),
            path: prepared.ingestion.source_locator.clone(),
            sha256: Some(prepared.ingestion.kept_corpus_sha256.clone()),
            openenv: None,
        };
        receipt.data.examples_read = prepared.ingestion.rows_read;
        receipt.data.examples_filtered = prepared.ingestion.rows_rejected;
        receipt.data.examples_trained = prepared.ingestion.rows_kept;
        receipt.data.sft_ingestion = Some(prepared.ingestion.clone());
        receipt.write_to_adapter_dir(dir.path())?;
        let loaded = TrainReceipt::read_from_adapter_dir(dir.path())?.unwrap();
        assert_eq!(loaded.data.sft_ingestion, Some(prepared.ingestion));

        let path = dir.path().join(TRAIN_RECEIPT_FILENAME);
        let mut tampered = serde_json::to_value(&loaded)?;
        tampered["training_data"]["sha256"] = serde_json::json!(full_sha256('f'));
        std::fs::write(&path, serde_json::to_vec_pretty(&tampered)?)?;
        let error = TrainReceipt::read_from_adapter_dir(dir.path()).unwrap_err();
        assert!(
            format!("{error:#}").contains("training_data.sha256 differs"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn sft_receipt_records_and_validates_effective_training_template_identity() -> Result<()> {
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?.with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );
        let hyperparameters = HyperparameterReceipt {
            mode: "sft".to_string(),
            rank: 2,
            alpha: 4.0,
            alpha_over_rank: Some(2.0),
            learning_rate: 1e-4,
            epochs: 1,
            seed: Some(7),
            shuffle: false,
        };
        let mut sft = TrainReceipt::new(
            "adapter-sft",
            "sft",
            &model,
            &tokenizer,
            hyperparameters.clone(),
            serde_json::json!({}),
        );
        assert_eq!(
            sft.tokenizer.training_chat_template_hash,
            sft.config_hashes.training_chat_template_hash
        );
        assert!(sft.tokenizer.training_chat_template_hash.is_some());

        let grpo = TrainReceipt::new(
            "adapter-grpo",
            "grpo",
            &model,
            &tokenizer,
            hyperparameters,
            serde_json::json!({}),
        );
        assert!(grpo.tokenizer.training_chat_template_hash.is_none());
        assert!(grpo.config_hashes.training_chat_template_hash.is_some());

        sft.tokenizer.training_chat_template_hash = Some(format!("sha256:{}", "0".repeat(64)));
        let dir = tempdir()?;
        let error = format!("{:#}", sft.write_to_adapter_dir(dir.path()).unwrap_err());
        assert!(error.contains("differs from config_hashes.training_chat_template_hash"));
        assert!(!dir.path().join(TRAIN_RECEIPT_FILENAME).exists());
        Ok(())
    }

    #[test]
    fn train_receipt_read_rejects_tampered_base_weight_manifest() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let mut receipt = TrainReceipt::new(
            "adapter-tampered",
            "sft",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 2,
                alpha: 4.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(1),
                shuffle: false,
            },
            serde_json::json!({}),
        );
        receipt.model.base_weight_shard_manifest = Some(base_weight_manifest());
        let mut value = serde_json::to_value(receipt)?;
        value["model"]["base_weight_shard_manifest"]["total_size_bytes"] = serde_json::json!(12);
        std::fs::write(
            dir.path().join(TRAIN_RECEIPT_FILENAME),
            serde_json::to_vec_pretty(&value)?,
        )?;

        let error = TrainReceipt::read_from_adapter_dir(dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("validate base-weight shard manifest"));
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
                shuffle: false,
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
    fn train_receipt_writes_adapter_manifest_when_adapter_files_exist() -> Result<()> {
        let dir = tempdir()?;
        std::fs::write(
            dir.path().join("adapter_config.json"),
            serde_json::json!({
                "r": 2,
                "lora_alpha": 4.0,
                "target_modules": ["q_proj"],
            })
            .to_string(),
        )?;
        std::fs::write(dir.path().join("adapter_model.safetensors"), "weights")?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let mut receipt = TrainReceipt::new(
            "adapter-manifest",
            "sft",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 2,
                alpha: 4.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(9),
                shuffle: false,
            },
            serde_json::json!({"base_adapter": "parent"}),
        );
        receipt.kiln.git_commit = Some("abc123".to_string());
        receipt.training_data.sha256 = Some("sha256:data".to_string());

        receipt.write_to_adapter_dir(dir.path())?;

        let manifest_path = dir
            .path()
            .join(crate::adapter_output::ADAPTER_MANIFEST_FILENAME);
        assert!(manifest_path.is_file());
        let manifest = crate::adapter_output::read_adapter_manifest(&manifest_path)?;
        assert_eq!(manifest.adapter_name, "adapter-manifest");
        assert_eq!(manifest.parent_adapter.as_deref(), Some("parent"));
        assert_eq!(manifest.kiln_commit.as_deref(), Some("abc123"));
        assert_eq!(manifest.training_data_hash.as_deref(), Some("sha256:data"));
        assert!(manifest.receipt_hash.is_some());
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
                shuffle: false,
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
    fn lora_grad_norm_accumulator_continues_bit_exactly_after_json_restore() {
        let first = 0.1076551472980432_f64;
        let second = 0.07733242672018717_f64;
        let mut uninterrupted = LoraGradNormAccumulator::default();
        uninterrupted.observe("k_proj", first);
        uninterrupted.observe("k_proj", second);

        let mut resumed = LoraGradNormAccumulator::default();
        resumed.observe("k_proj", first);
        let encoded = serde_json::to_string(&resumed).unwrap();
        assert!(
            encoded.contains("0x"),
            "checkpoint floats must use raw bits"
        );
        let mut resumed: LoraGradNormAccumulator = serde_json::from_str(&encoded).unwrap();
        resumed.observe("k_proj", second);
        assert_eq!(resumed, uninterrupted);
    }

    #[test]
    fn lora_grad_norm_accumulator_accepts_legacy_numeric_state() {
        let legacy = r#"{
            "by_module": {
                "q_proj": {
                    "sample_count": 1,
                    "sum": 0.25,
                    "min": 0.25,
                    "max": 0.25
                }
            }
        }"#;
        let restored: LoraGradNormAccumulator = serde_json::from_str(legacy).unwrap();
        let summary = restored.finish().pop().unwrap();
        assert_eq!(summary.mean, 0.25);
    }

    #[test]
    fn lora_grad_norm_accumulator_canonicalizes_f64_tail_noise() {
        let mut left = LoraGradNormAccumulator::default();
        let mut right = LoraGradNormAccumulator::default();
        left.observe("gate_proj", 0.00728748675095551);
        right.observe("gate_proj", 0.007287486750955509);
        assert_eq!(left, right);
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

    /// (#1082) Numeric check for `tensor_l2_norm_kt` against an independent
    /// closed-form L2 norm. (The candle oracle `tensor_l2_norm` was deleted in
    /// the candle drop; the closed-form `expected` is now the sole oracle.)
    #[test]
    fn tensor_l2_norm_kt_matches_closed_form_for_f32_inputs() -> Result<()> {
        let xs: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let expected = ((1.0_f64).powi(2)
            + (2.0_f64).powi(2)
            + (3.0_f64).powi(2)
            + (4.0_f64).powi(2)
            + (5.0_f64).powi(2))
        .sqrt();

        let kt_t = kt::Tensor::from_vec(xs, vec![5]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let kt_norm = tensor_l2_norm_kt(&kt_t)?;
        assert!(
            (kt_norm - expected).abs() < 1e-5,
            "kt norm {kt_norm} != expected {expected}"
        );
        Ok(())
    }

    /// (#1082) End-to-end check that `lora_delta_norm_summary_from_adapter`
    /// still produces correct PEFT-shaped summaries after migrating its
    /// safetensors loader from `candle_core::safetensors::load` to
    /// `kt::safetensors::load_cpu`.
    #[test]
    fn lora_delta_norm_summary_from_adapter_kt_loader_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("adapter_model.safetensors");

        // Two PEFT-shaped LoRA pairs for layer 0 / q_proj, with known
        // L2 norms (3/4/5 right triangles).
        let a: Vec<f32> = vec![3.0, 0.0, 0.0, 4.0]; // 2x2, L2 = 5
        let b: Vec<f32> = vec![6.0, 0.0, 0.0, 8.0]; // 2x2, L2 = 10
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for f in v {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out
        };
        let entries: Vec<(String, Vec<u8>)> = vec![
            (
                "base_model.model.model.layers.0.q_proj.lora_A.weight".to_string(),
                to_bytes(&a),
            ),
            (
                "base_model.model.model.layers.0.q_proj.lora_B.weight".to_string(),
                to_bytes(&b),
            ),
        ];
        let views: Vec<(&str, ::safetensors::tensor::TensorView<'_>)> = entries
            .iter()
            .map(|(k, bytes)| {
                let v = ::safetensors::tensor::TensorView::new(
                    ::safetensors::Dtype::F32,
                    vec![2, 2],
                    bytes,
                )
                .expect("tensor view");
                (k.as_str(), v)
            })
            .collect();
        ::safetensors::serialize_to_file(views, None, &path)
            .map_err(|e| anyhow::anyhow!("serialize_to_file: {e}"))?;

        let summaries = lora_delta_norm_summary_from_adapter(dir.path(), 2.0)?;
        assert_eq!(summaries.len(), 1);
        let s = &summaries[0];
        assert_eq!(s.module, "q_proj");
        assert_eq!(s.pair_count, 1);
        assert!((s.a_l2_mean - 5.0).abs() < 1e-5, "a={} != 5", s.a_l2_mean);
        assert!((s.b_l2_mean - 10.0).abs() < 1e-5, "b={} != 10", s.b_l2_mean);
        // delta upper bound = a_l2 * b_l2 * alpha_over_rank = 5 * 10 * 2 = 100
        assert!(
            (s.delta_l2_upper_bound_max - 100.0).abs() < 1e-4,
            "delta_max={} != 100",
            s.delta_l2_upper_bound_max
        );
        Ok(())
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
            dropped_reason: None,
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
                .any(|warning| warning.contains("policy-gradient may be harmful"))
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("harder tasks"))
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("stronger rubric gates"))
        );
        assert!(warnings.iter().any(|warning| warning.contains("OPD")));
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

    #[test]
    fn runtime_receipt_sft_loss_route_is_typed_and_legacy_optional() -> Result<()> {
        let tokenizer = minimal_tokenizer()?;
        let mut receipt = TrainReceipt::new(
            "route-receipt",
            "sft",
            &ModelConfig::qwen3_5_4b(),
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(1),
                shuffle: true,
            },
            serde_json::json!({}),
        );
        let legacy_json = serde_json::to_value(&receipt)?;
        assert!(legacy_json["runtime"].get("sft_loss_route").is_none());

        receipt.runtime.sft_loss_route =
            Some(kiln_model::backend::SftFlceLossRoute::VulkanActiveRows);
        let encoded = serde_json::to_value(&receipt)?;
        assert_eq!(encoded["runtime"]["sft_loss_route"], "vulkan_active_rows");
        assert_eq!(serde_json::from_value::<TrainReceipt>(encoded)?, receipt);
        Ok(())
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

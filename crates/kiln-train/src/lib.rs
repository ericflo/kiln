//! Training for Kiln — pure Rust, in-process LoRA SFT and GRPO.
//!
//! This crate defines the training API types AND implements the actual training
//! loop using candle autograd. Training runs in the same process as inference,
//! operating on the already-loaded model weights. No Python sidecar needed.

#[cfg(feature = "cuda")]
pub mod cuda_train;
pub mod diagnostics;
pub mod logit_cache;
pub mod logit_source;
pub mod opd;
pub mod receipt;
pub mod remote_teacher;
pub mod replay;
pub mod trainer;
#[cfg(feature = "vulkan")]
pub mod vk_train;

pub use logit_cache::{CacheEntry, CacheStats, CachedLogitSource, LogitCache, hash_prefix};
pub use remote_teacher::{CostTally, RemoteProvider, RemoteTeacher, RemoteTeacherConfig};
pub use receipt::{
    AdapterReceipt, DiagnosticSummary, PromptSourceDescriptor, RECEIPT_SCHEMA_VERSION,
    TeacherDescriptor,
};

pub use diagnostics::{
    DIVERSITY_COLLAPSE_THRESHOLD, DIVERSITY_COLLAPSE_WINDOW, GuardrailDecision, GuardrailTrigger,
    LengthInflationGuardrail, OpdDiagnosticSnapshot, REPETITION_GUARDRAIL_THRESHOLD, RolloutSummary,
    SELF_PLAY_SATURATION_THRESHOLD, SELF_PLAY_SATURATION_WINDOW, build_snapshot, repetition_rate,
    rollout_diversity, truncation_rate,
};
pub use logit_source::{
    DeterministicUniformLogitSource, LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch,
    TopKLogprobs,
};
pub use opd::{
    AgenticLossInputs, AgenticLossWeights, COLD_START_DEFAULT_EPOCHS, COLD_START_DEFAULT_PROMPTS,
    COLD_START_OVERLAP_THRESHOLD, ColdStartDecision, DistillMergeRequest, DistillMergeSource,
    DistillPumpMode, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest,
    NewKnowledgeSource, OpdConfig, OpdLossGranularity, OpdPrompt, OpdRequest, SelfDistillMode,
    StableOpdCoefficients, StableOpdLossInputs, StableOpdLossOutputs, TipTokenClass,
    cold_start_probe, cold_start_probe_default, compute_agentic_loss_weights,
    compute_initial_overlap, compute_stable_opd_loss, default_beta_kl,
    default_lambda_sft, default_lambda_verifier, default_opd_samples_per_prompt,
    default_opd_top_k, default_score_decay_steps, default_score_earliest_weight,
    default_tip_tool_call_weight, default_tip_tool_name_weight,
};

pub use replay::{
    BaseModel, Lineage, OutcomeRecord, OutcomeStatus, ParentLora, ReplayKind, ReplayLog,
    ReplayRecord, RequestRecord,
};
pub use trainer::CheckpointConfig;

use serde::{Deserialize, Serialize};

/// A chat message in a training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// An SFT training example — a conversation with the correct assistant response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftExample {
    pub messages: Vec<ChatMessage>,
}

/// Request to run SFT training on submitted examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftRequest {
    pub examples: Vec<SftExample>,
    #[serde(default)]
    pub config: SftConfig,
    /// Optional auto-eval hook: when set, the training queue worker enqueues
    /// an eval against the produced adapter once training completes. Lets
    /// callers chain `train → eval` in a single API call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Optimizer selection for training.
///
/// `Sgd` is plain stochastic gradient descent (`param -= lr * grad`) — the
/// historical default; dispatched on-device via `dispatch_sgd_step` when the
/// backend supports residency, otherwise via candle CPU autograd.
///
/// `AdamW` is decoupled-weight-decay Adam (Loshchilov & Hutter 2019);
/// dispatched on-device via `dispatch_adamw_step` when the backend supports
/// residency. The trainer allocates per-parameter first/second moment Vars at
/// init, registers them in the resident-activation registry alongside the
/// param/grad, and updates all three in-place per step. The CPU fallback runs
/// the same update via candle ops.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Optimizer {
    Sgd,
    AdamW {
        #[serde(default = "default_beta1")]
        beta1: f32,
        #[serde(default = "default_beta2")]
        beta2: f32,
        #[serde(default = "default_eps")]
        eps: f32,
        #[serde(default = "default_weight_decay")]
        weight_decay: f32,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::AdamW {
            beta1: default_beta1(),
            beta2: default_beta2(),
            eps: default_eps(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_eps() -> f32 {
    1e-8
}
fn default_weight_decay() -> f32 {
    0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftConfig {
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_sft_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    /// If set, continue training from this adapter instead of starting fresh.
    pub base_adapter: Option<String>,
    /// Name for the output adapter. Auto-generated if not set.
    pub output_name: Option<String>,
    /// Automatically load the resulting adapter when training completes (default true).
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,
    /// Save adapter weights every N training steps. None = only save at the end.
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// Deterministic seed for LoRA init and any RNG-dependent steps. If
    /// `None`, the trainer generates one and records it in `replay.jsonl`
    /// so the run is still exactly reproducible.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optimizer selection. Defaults to AdamW (decoupled weight decay) per
    /// LoRA fine-tuning best practice. Plain SGD is available via
    /// `{"optimizer": {"kind": "sgd"}}` for backwards-compatible runs.
    #[serde(default)]
    pub optimizer: Optimizer,
}

fn default_auto_load() -> bool {
    true
}
fn default_epochs() -> usize {
    3
}
fn default_sft_lr() -> f64 {
    1e-4
}
fn default_rank() -> usize {
    16
}
fn default_alpha() -> f32 {
    32.0
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            learning_rate: default_sft_lr(),
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            seed: None,
            optimizer: Optimizer::default(),
        }
    }
}

/// A scored completion for GRPO training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCompletion {
    pub text: String,
    pub reward: f64,
}

/// A group of completions for one prompt (GRPO operates on groups).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoGroup {
    /// The prompt that generated these completions.
    pub messages: Vec<ChatMessage>,
    /// Multiple completions with their rewards.
    pub completions: Vec<ScoredCompletion>,
}

/// Request to run a GRPO training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoRequest {
    #[serde(default)]
    pub groups: Vec<GrpoGroup>,
    /// Optional server-local JSONL dataset path. Each non-empty line is one
    /// `GrpoGroup`. Used by Vulkan-native GRPO to stream large datasets without
    /// retaining every group in memory.
    #[serde(default)]
    pub dataset_path: Option<String>,
    #[serde(default)]
    pub config: GrpoConfig,
    /// Optional auto-eval hook (see `SftRequest::post_eval`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Group-relative advantage normalization mode.
///
/// `Vanilla` is the DeepSeekMath / R1 default: `A_i = (r_i - mean(r)) / (std(r) + eps)`.
/// `DrGrpo` follows arXiv:2503.20783 ("Understanding R1-Zero-Like Training") and
/// drops the `std` normalization to eliminate question-difficulty bias and the
/// numerical blow-up when within-group rewards are nearly uniform.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdvantageMode {
    /// `A_i = (r_i - mean(r)) / (std(r) + 1e-8)`. The historical kiln default.
    #[default]
    Vanilla,
    /// `A_i = r_i - mean(r)`. Recommended by Dr. GRPO (arXiv:2503.20783) and
    /// the SimpleRL-Zoo replication (arXiv:2503.18892).
    DrGrpo,
}

/// How the GRPO surrogate loss is aggregated across the completions in a group.
///
/// `PerSample` mirrors the historical kiln behavior: each completion's loss is
/// the per-token mean, then the group reports the mean across completions, and
/// the optimizer steps once per completion. This is the DeepSeekMath / R1
/// default. It systematically under-penalizes long incorrect completions and
/// is the source of the documented GRPO "length-drift" failure mode.
///
/// `TokenLevel` is the DAPO Token-Level Policy Gradient Loss (arXiv:2503.14476):
/// per-token surrogates are accumulated across all completions in the group
/// and divided by the total active completion token count, with a single
/// optimizer step per group. Removes per-sample length bias.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LossAggregation {
    /// Per-completion mean-of-tokens, per-completion optimizer step.
    #[default]
    PerSample,
    /// DAPO Token-Level Loss: sum-over-all-tokens-in-group divided by
    /// total active tokens, single optimizer step per group.
    /// (Candle path only — the vk-native path falls back to `PerSample`.)
    TokenLevel,
}

/// Which KL estimator to use for the per-token penalty.
///
/// All three options leave the reference forward pass intact (the reference
/// log-probs are needed for the importance-sampling ratio). They only change
/// the per-token KL term added to the surrogate.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KlEstimator {
    /// Schulman k1: `KL_t = log_ratio_t`. Gradient-correct (matches the
    /// existing kiln implementation and is the recommended default).
    #[default]
    K1,
    /// Schulman k3: `KL_t = exp(-log_ratio_t) - 1 + log_ratio_t`. Always
    /// non-negative; value-correct but the gradient is biased. DeepSeekMath
    /// uses this form. Candle path only; not implemented on vk-native.
    K3,
    /// No KL penalty. The reference forward still runs (still needed for the
    /// importance ratio); only the KL contribution to the loss is zeroed.
    /// Equivalent in effect to `kl_coeff = 0` but expresses the intent
    /// explicitly.
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    #[serde(default = "default_grpo_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_kl_coeff")]
    pub kl_coeff: f64,
    /// Symmetric clip epsilon. When `clip_eps_high` is `None`, both the lower
    /// and upper PPO clip bounds use this value (`[1-ε, 1+ε]`). When
    /// `clip_eps_high` is `Some(h)`, this field provides the lower epsilon
    /// (`1-ε_low`) and `h` provides the upper (`1+ε_high`). DAPO's
    /// "Clip-Higher" recommendation is `clip_epsilon = 0.20`,
    /// `clip_eps_high = Some(0.28)` (arXiv:2503.14476).
    #[serde(default = "default_clip_eps")]
    pub clip_epsilon: f64,
    /// Upper PPO clip epsilon for the asymmetric Clip-Higher recipe. `None`
    /// (default) preserves symmetric clipping using `clip_epsilon` on both
    /// sides.
    #[serde(default)]
    pub clip_eps_high: Option<f64>,
    /// Advantage normalization mode. Defaults to `Vanilla` (historical
    /// kiln behavior). Set to `DrGrpo` to drop std-normalization.
    #[serde(default)]
    pub advantage_mode: AdvantageMode,
    /// Surrogate-loss aggregation mode. Defaults to `PerSample` (historical
    /// kiln behavior). Set to `TokenLevel` for the DAPO Token-Level Loss.
    #[serde(default)]
    pub loss_aggregation: LossAggregation,
    /// KL penalty estimator. Defaults to `K1` (historical kiln behavior).
    #[serde(default)]
    pub kl_estimator: KlEstimator,
    /// When true, groups whose completions all share the same reward (and
    /// therefore produce a uniformly-zero advantage vector) are skipped
    /// before training. Implements DAPO's Dynamic Sampling filter. Default
    /// `false` for backward compatibility; turning this on is essentially
    /// always a stability win.
    #[serde(default)]
    pub dynamic_sampling: bool,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    pub base_adapter: Option<String>,
    pub output_name: Option<String>,
    /// Automatically load the resulting adapter when training completes (default true).
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,
    /// Save adapter weights every N training steps. None = only save at the end.
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// Deterministic seed for LoRA init and any RNG-dependent steps. If
    /// `None`, the trainer generates one and records it in `replay.jsonl`
    /// so the run is still exactly reproducible.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optimizer selection — see `SftConfig::optimizer`.
    #[serde(default)]
    pub optimizer: Optimizer,
}

fn default_grpo_lr() -> f64 {
    1e-5
}
fn default_kl_coeff() -> f64 {
    0.1
}
fn default_clip_eps() -> f64 {
    0.2
}

impl GrpoConfig {
    /// Resolved PPO clip epsilon bounds `(ε_low, ε_high)`. The PPO clip range
    /// is `[1 - ε_low, 1 + ε_high]`. When `clip_eps_high` is `None`, both
    /// sides use `clip_epsilon` (symmetric, historical behavior).
    pub fn clip_bounds(&self) -> (f64, f64) {
        (
            self.clip_epsilon,
            self.clip_eps_high.unwrap_or(self.clip_epsilon),
        )
    }
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_grpo_lr(),
            kl_coeff: default_kl_coeff(),
            clip_epsilon: default_clip_eps(),
            clip_eps_high: None,
            advantage_mode: AdvantageMode::default(),
            loss_aggregation: LossAggregation::default(),
            kl_estimator: KlEstimator::default(),
            dynamic_sampling: false,
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            seed: None,
            optimizer: Optimizer::default(),
        }
    }
}

/// Status of an ongoing training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub job_id: String,
    pub state: TrainingState,
    pub progress: f32,
    pub current_loss: Option<f64>,
    pub adapter_name: Option<String>,
    pub started_at: String,
    pub elapsed_secs: f64,
    /// Wall-clock submit time as Unix milliseconds — survives server restarts
    /// so the /ui can render real timestamps for archived terminal jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_unix_ms: Option<u64>,
    /// Wall-clock terminal-transition time as Unix milliseconds. `None` while
    /// the job is still active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_unix_ms: Option<u64>,
    /// "sft" or "grpo" — populated by the server side; absent on older
    /// payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_type: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingState {
    Queued,
    Running,
    Completed,
    Failed,
}

/// Response after submitting a training request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResponse {
    pub job_id: String,
    pub state: TrainingState,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sft_config_default_checkpoint_interval_is_none() {
        let config = SftConfig::default();
        assert!(config.checkpoint_interval.is_none());
    }

    #[test]
    fn test_grpo_config_default_checkpoint_interval_is_none() {
        let config = GrpoConfig::default();
        assert!(config.checkpoint_interval.is_none());
    }

    #[test]
    fn test_sft_config_deserialize_with_checkpoint_interval() {
        let json = r#"{"checkpoint_interval": 25}"#;
        let config: SftConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.checkpoint_interval, Some(25));
        assert_eq!(config.epochs, 3); // default preserved
    }

    #[test]
    fn test_sft_config_deserialize_without_checkpoint_interval() {
        let json = r#"{"epochs": 5}"#;
        let config: SftConfig = serde_json::from_str(json).unwrap();
        assert!(config.checkpoint_interval.is_none());
        assert_eq!(config.epochs, 5);
    }

    #[test]
    fn test_grpo_config_deserialize_with_checkpoint_interval() {
        let json = r#"{"checkpoint_interval": 10}"#;
        let config: GrpoConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.checkpoint_interval, Some(10));
        assert_eq!(config.kl_coeff, 0.1); // default preserved
    }
}

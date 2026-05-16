//! Training for Kiln — pure Rust, in-process LoRA SFT and GRPO.
//!
//! This crate defines the training API types AND implements the actual training
//! loop using candle autograd. Training runs in the same process as inference,
//! operating on the already-loaded model weights. No Python sidecar needed.

#[cfg(feature = "cuda")]
pub mod cuda_train;
pub mod diagnostics;
pub mod logit_source;
pub mod opd;
pub mod receipt;
pub mod replay;
pub mod trainer;
#[cfg(feature = "vulkan")]
pub mod vk_train;

pub use receipt::{
    AdapterReceipt, DiagnosticSummary, PromptSourceDescriptor, RECEIPT_SCHEMA_VERSION,
    TeacherDescriptor,
};

pub use diagnostics::{
    GuardrailDecision, GuardrailTrigger, LengthInflationGuardrail, OpdDiagnosticSnapshot,
    REPETITION_GUARDRAIL_THRESHOLD, RolloutSummary, build_snapshot, repetition_rate,
    truncation_rate,
};
pub use logit_source::{
    LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs,
};
pub use opd::{
    COLD_START_DEFAULT_EPOCHS, COLD_START_DEFAULT_PROMPTS, COLD_START_OVERLAP_THRESHOLD,
    ColdStartDecision, DistillMergeRequest, DistillMergeSource, DistillPumpMode,
    DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest, NewKnowledgeSource, OpdConfig,
    OpdLossGranularity, OpdPrompt, OpdRequest, SelfDistillMode, StableOpdCoefficients,
    StableOpdLossInputs, StableOpdLossOutputs, cold_start_probe, cold_start_probe_default,
    compute_initial_overlap, compute_stable_opd_loss, default_beta_kl, default_lambda_sft,
    default_opd_samples_per_prompt, default_opd_top_k,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    #[serde(default = "default_grpo_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_kl_coeff")]
    pub kl_coeff: f64,
    #[serde(default = "default_clip_eps")]
    pub clip_epsilon: f64,
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

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_grpo_lr(),
            kl_coeff: default_kl_coeff(),
            clip_epsilon: default_clip_eps(),
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

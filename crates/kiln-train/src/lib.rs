//! Training for Kiln — pure Rust, in-process LoRA SFT and GRPO.
//!
//! This crate defines the training API types AND implements the actual training
//! loop using candle autograd. Training runs in the same process as inference,
//! operating on the already-loaded model weights. No Python sidecar needed.

pub mod trainer;

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
}

fn default_auto_load() -> bool { true }
fn default_epochs() -> usize { 3 }
fn default_sft_lr() -> f64 { 1e-4 }
fn default_rank() -> usize { 16 }
fn default_alpha() -> f32 { 32.0 }

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
    pub groups: Vec<GrpoGroup>,
    #[serde(default)]
    pub config: GrpoConfig,
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
}

fn default_grpo_lr() -> f64 { 1e-5 }
fn default_kl_coeff() -> f64 { 0.1 }
fn default_clip_eps() -> f64 { 0.2 }

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

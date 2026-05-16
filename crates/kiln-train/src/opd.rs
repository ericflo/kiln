//! On-Policy Distillation trainer.
//!
//! Implements the §3.1 pseudocode from the grand plan:
//!
//! ```text
//! # Initialize teacher client (LogitSource).
//! # Sample trajectories (do_group_rollout on the student).
//! # Compute reward (per-token reverse KL via the OPD-loss kernel).
//! # Train with RL (importance-sampling loss; advantage = -reverse_kl).
//! ```
//!
//! The mathematical guarantee from Lu (2025) is that this is "literally
//! one line on top of the GRPO trainer" — swap `reward - baseline` for
//! `-reverse_kl_per_token`. That observation is what makes this module
//! a sibling of `trainer.rs::grpo_loss` rather than a fork.
//!
//! # Milestone 2 (this commit)
//!
//! What lands now:
//!
//! * Request / config types matching `SftRequest` / `GrpoRequest` shape
//!   (the §4 endpoint payload).
//! * `OpdLossGranularity` enum (`SampledToken` / `TeacherTopK` /
//!   `FullVocab`) with the §6 defaults.
//! * `opd_step_loss` — one OPD step's loss + per-position KL, given a
//!   fully tokenized rollout, the student's hidden states at the
//!   rollout's positions, and a `LogitSource` to query the teacher.
//!   Returns the scalar loss + per-position KL vector for diagnostics
//!   (`overlap_ratio`, `entropy_gap`, etc. — §3.8 wires off these).
//!
//! What's deferred to the next commit:
//!
//! * The full `opd_train` function that loops the §3.1 pseudocode over
//!   prompts × rollouts × optimizer steps with hot-swap and replay log
//!   integration — that step depends on the trainer's
//!   `checkpointed_grpo_forward_backward` body, which the wiring will
//!   factor out so OPD and GRPO share the segment-checkpointing path
//!   (no duplication).
//!
//! # Math
//!
//! For each active position `t` in a rollout, we have:
//!
//! * `student_logprobs[t]` — log p_student(sampled_token_t | prefix_t).
//! * `teacher_topk_indices[t, :K]`, `teacher_topk_logprobs[t, :K]` —
//!   the teacher's top-K support and full-vocab logprobs there.
//!
//! Three granularities produce different per-token KL:
//!
//! 1. **`SampledToken`** (Lu's default for the reasoning case).
//!    `KL_t = student_logprobs[t] - teacher_logprob_at_sampled[t]`.
//!    Cheap (no kernel), brittle on long rollouts (§3.1 of Fu et al.).
//! 2. **`TeacherTopK`** (§6 default, robust). Renormalise both
//!    distributions over the teacher's K support, then compute
//!    `KL(p_hat || q_hat)`. Uses `kiln-opd-loss-kernel`.
//! 3. **`FullVocab`** (corporate tier per §5.1.2 of deepseek_v4).
//!    The K support is the full vocabulary; same kernel, K = V.
//!
//! Per-token advantage for the importance-sampling loss is
//! `A_t = -KL_t`. The trainer's GRPO loss machinery (`grpo_loss` in
//! `trainer.rs`) is reused verbatim with this advantage.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::logit_source::{LogitSource, LogprobBatch};
use crate::{ChatMessage, Optimizer, default_alpha, default_rank};
use kiln_opd_loss_kernel::{opd_top_k_reverse_kl_phase_a_per_position, DEFAULT_CHUNK_SIZE};

/// §6 default top-K for the `TeacherTopK` loss path. Picked by Fu et al.
/// (2026) ablation table 3: K = 32 is the optimum across math and
/// agentic OPD; K = 16 underperforms; K = 64 buys no further gain.
pub const fn default_opd_top_k() -> usize {
    32
}

/// §3.1 default rollouts-per-prompt. Lu (2025) used 4; auto-scales to
/// 16/64 when the dataset is small (`data_multiplier` mode, §3.5.4).
/// The auto-scale logic is a trainer-level decision and lives in
/// `opd_train`, not in the per-step kernel.
pub const fn default_opd_samples_per_prompt() -> usize {
    4
}

/// §3.1 loss granularity selector. Defaults to `TeacherTopK` (§6).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpdLossGranularity {
    /// Single-token reverse KL — `student_lp - teacher_lp_at_sampled`.
    /// Lu's default; brittle on long rollouts per Fu et al. 2026.
    SampledToken,
    /// Top-K renormalised reverse KL. **The default.** Fu et al. 2026
    /// teacher Top-K local support matching.
    TeacherTopK,
    /// Full-vocab reverse KL (K = V). DeepSeek-V4 corporate-tier.
    FullVocab,
}

impl Default for OpdLossGranularity {
    fn default() -> Self {
        Self::TeacherTopK
    }
}

/// §3.1 Stable-OPD knob set (Luo et al. 2026). When `Auto`, the
/// `LengthInflation` guardrail (§3.9, lands in #24) toggles these
/// based on RepRate observations. Power users can pin
/// `Off` or `Manual { kl, sft }`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum StableOpdMode {
    /// Disable Stable-OPD entirely. Only useful for parity studies vs
    /// the Lu (2025) baseline; never recommended for real runs.
    Off,
    /// Auto-engage based on the diagnostic stack — §3.9 default.
    Auto,
    /// Pin both knobs.
    Manual {
        /// β — reference-KL penalty weight. Luo et al. default 0.01.
        kl_beta: f64,
        /// λ — golden-trajectory mixture fraction. Luo et al. default 0.1.
        sft_lambda: f64,
    },
}

impl Default for StableOpdMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// One prompt's worth of OPD input — the same minimal shape as
/// `SftExample`. The trainer rolls out `samples_per_prompt` completions
/// for each prompt, runs the teacher over each, computes per-token
/// reverse KL, and turns each completion into an importance-sampling
/// step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpdPrompt {
    pub messages: Vec<ChatMessage>,
}

/// Reference to a new-knowledge dataset for [`DistillRefreshRequest`].
/// Two shapes:
/// - `dataset_name` — server-registered dataset (the eval-dataset
///   upload path produces these); the trainer resolves to a JSONL
///   file on disk.
/// - `examples` — inline list of `OpdPrompt`s. Useful for tiny
///   personalisation runs and unit tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NewKnowledgeSource {
    Dataset {
        /// Server-side registered dataset name. Resolves to a path
        /// inside the server's eval-datasets directory.
        dataset: String,
    },
    Inline {
        /// Inline prompts (and optional teacher completions when
        /// `mode=behaviour_recovery_only`).
        examples: Vec<OpdPrompt>,
    },
}

/// HTTP-shaped request for `POST /v1/distill/refresh` — the §3.6
/// continual-learning recipe (Lu 2025 instruction-following recovery
/// experiment).
///
/// Two-phase pipeline:
/// 1. **Mid-train** on `new_data` mixed with `background_chat`
///    (default Tulu3) — student learns the new knowledge but
///    typically degrades on instruction-following.
/// 2. **OPD-recover** against `behavioural_teacher` (the prior
///    checkpoint of the model itself, per Lu's recipe) — reverse-KL
///    on Tulu3-flavoured prompts. The instruction-following is
///    restored without losing the new knowledge.
///
/// Both phases pre-eval (baseline) and post-eval (after each phase),
/// gated on the two thresholds. The new adapter is only published
/// when both gates pass; otherwise the run is marked failed and the
/// previous adapter remains active. §8.7 auto-rollback contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillRefreshRequest {
    /// Name of the existing adapter being refreshed. The new adapter
    /// is named with the same prefix + a version suffix (e.g.
    /// `company-assistant@v18`).
    pub name: String,
    /// New knowledge to mid-train on.
    pub new_data: NewKnowledgeSource,
    /// Alias of the prior-self teacher to recover against. Per
    /// Lu (2025): "an earlier version of the model itself." Typically
    /// resolves to the previous `name@vN` checkpoint via the §3.2
    /// teacher registry.
    pub behavioural_teacher: String,
    /// Background-chat distribution for mid-training (§3.1 / §3.6
    /// regulariser). Default `tulu3`; resolves server-side.
    #[serde(default = "default_background_chat")]
    pub background_chat: String,
    /// Required fractional recovery on the instruction-following
    /// eval suite before the refreshed adapter is published.
    /// `0.95` (default) means "IF-eval after refresh must be ≥ 95%
    /// of the pre-refresh score" — Lu's recipe target.
    #[serde(default = "default_require_if_eval_recovery")]
    pub require_if_eval_recovery: f64,
    /// Required absolute gain on the new-knowledge eval suite.
    /// `0.05` (default) = 5 percentage points.
    #[serde(default = "default_require_new_knowledge_gain")]
    pub require_internal_qa_gain: f64,
    /// Per-job OPD config (knobs for the recovery phase). Defaults
    /// match `OpdConfig::default`.
    #[serde(default)]
    pub config: OpdConfig,
    /// Optional auto-eval hook firing after the run completes
    /// (instruction-following + new-knowledge suites are eval'd
    /// internally regardless; this is the extra one the dashboard
    /// uses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

fn default_background_chat() -> String {
    "tulu3".to_string()
}
fn default_require_if_eval_recovery() -> f64 {
    0.95
}
fn default_require_new_knowledge_gain() -> f64 {
    0.05
}

// ---------------------------------------------------------------------------
// /v1/adapters/distill_merge — behaviour-space merge (§3.4)
// ---------------------------------------------------------------------------

/// One source adapter for the [`DistillMergeRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillMergeSource {
    /// Adapter name on disk.
    pub adapter: String,
    /// Mixture weight when sampling prompts owned by this source.
    /// Defaults to 1.0 — equal weighting across sources.
    #[serde(default = "default_merge_source_weight")]
    pub weight: f64,
}

fn default_merge_source_weight() -> f64 {
    1.0
}

/// `POST /v1/adapters/distill_merge` payload — §3.4 "killer feature."
///
/// Each source LoRA is treated as a teacher over its retained training-
/// prompt distribution. The merged adapter is OPD-trained to match
/// each teacher on the prompts that teacher was good at. Multi-teacher
/// reverse-KL weighting from DeepSeek-V4 §5.1.2 with per-prompt routing
/// resolved by source-of-origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillMergeRequest {
    /// Output adapter name.
    pub name: String,
    /// Source adapters to merge. Each becomes a "teacher" over its
    /// own prompt history.
    pub sources: Vec<DistillMergeSource>,
    /// Optional warm-start: adapter to initialise from before
    /// running multi-teacher OPD. Default `"base"` (the unmodified
    /// model).
    #[serde(default = "default_merge_student")]
    pub student: String,
    /// Total rollout budget across all sources (per the §3.4
    /// example: `5000`).
    #[serde(default = "default_merge_rollout_budget")]
    pub rollout_budget: usize,
    /// Per-job OPD config; loss granularity defaults to
    /// `teacher_top_k` so each source's distillation honours the §6
    /// fast path.
    #[serde(default)]
    pub config: OpdConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

fn default_merge_student() -> String {
    "base".to_string()
}
fn default_merge_rollout_budget() -> usize {
    5_000
}

// ---------------------------------------------------------------------------
// /v1/distill/pump — 27B → 4B Knowledge Pump (§3.5)
// ---------------------------------------------------------------------------

/// Three modes of operation per §3.5:
/// - `Domain { domain }` — targeted-domain (canonical seed corpus).
/// - `Wide` — wide-corpus generalist pump.
/// - `Examples { examples }` — auto-domain from user prompts;
///   data-multiplier mode auto-engages when `|examples| < 200`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DistillPumpMode {
    Domain { domain: String },
    Examples { examples: Vec<OpdPrompt> },
    Wide { wide: bool },
}

/// `POST /v1/distill/pump` payload — §3.5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillPumpRequest {
    pub name: String,
    /// Teacher alias.
    pub teacher: String,
    /// One of the three mode variants (§3.5.{1,2,3}).
    pub mode: DistillPumpMode,
    /// LoRA rank — overrides the OpdConfig default when set.
    #[serde(default)]
    pub rank: Option<usize>,
    /// Total rollout budget.
    #[serde(default = "default_pump_rollout_budget")]
    pub rollout_budget: usize,
    /// Honor the canonical cache + the community cache (§3.3).
    /// Default `true` — pit-of-success pre-amortise.
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub config: OpdConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

fn default_pump_rollout_budget() -> usize {
    50_000
}
fn default_use_cache() -> bool {
    true
}

// ---------------------------------------------------------------------------
// /v1/distill/self — Privileged-Information self-distillation (§3.12)
// ---------------------------------------------------------------------------

/// One of the four PI modes described in §3.12.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelfDistillMode {
    /// OPSD: ground-truth answer as privileged context to the
    /// teacher copy of the model.
    GroundTruthConditioning,
    /// CRISP: condition on "be concise" — teaches concision. Lu et
    /// al. report 57% token reduction at +9% accuracy.
    Conciseness,
    /// GATES: retrieval context to teacher only.
    DocumentAsPi,
    /// RLRT: reversed teacher signal.
    ReverseTeacher,
}

/// `POST /v1/distill/self` payload — §3.12 PI self-distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillSelfRequest {
    pub name: String,
    /// Self-distillation mode.
    pub mode: SelfDistillMode,
    /// Prompts to OPD over. `None` = use the most recent training
    /// dataset registered with the server.
    #[serde(default)]
    pub prompts: Option<Vec<OpdPrompt>>,
    /// Optional ground-truth answers for `GroundTruthConditioning`.
    /// Must match `prompts` length when both are set.
    #[serde(default)]
    pub ground_truth: Option<Vec<String>>,
    /// Optional retrieval context for `DocumentAsPi`.
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    #[serde(default)]
    pub config: OpdConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// HTTP-shaped request for `POST /v1/train/opd`. Mirror of `GrpoRequest`,
/// adapted to the OPD recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpdRequest {
    /// Prompts to OPD-train on. May be empty if `dataset_path` is set.
    #[serde(default)]
    pub prompts: Vec<OpdPrompt>,
    /// Optional server-local JSONL dataset path. Each non-empty line is
    /// one [`OpdPrompt`]. Used for large prompt sets that shouldn't be
    /// held in memory.
    #[serde(default)]
    pub dataset_path: Option<String>,
    /// Alias of the `LogitSource` registered via `/v1/teachers` (§3.2).
    /// Resolved server-side at the start of the run.
    pub teacher: String,
    #[serde(default)]
    pub config: OpdConfig,
    /// Optional auto-eval hook (see `SftRequest::post_eval`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// §3.1 + §6 default config for the OPD trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpdConfig {
    /// Loss granularity (§3.1). Defaults to `TeacherTopK` per §6.
    #[serde(default)]
    pub loss: OpdLossGranularity,

    /// Top-K size when `loss = TeacherTopK`. Default 32 (Fu et al. 2026
    /// ablation). Ignored when `loss = SampledToken` or `FullVocab`.
    #[serde(default = "default_opd_top_k")]
    pub top_k: usize,

    /// Rollouts per prompt — Lu (2025) default 4; auto-scaled by the
    /// trainer to 16/64 when the prompt count is small (§3.5.4
    /// data-multiplier).
    #[serde(default = "default_opd_samples_per_prompt")]
    pub samples_per_prompt: usize,

    /// Rollout sampling temperature. §6 default 1.0 (Fu et al.
    /// ablation table 3).
    #[serde(default = "default_opd_temperature")]
    pub temperature: f64,

    /// Rollout nucleus-sampling cutoff. §6 default 0.9 (Fu et al.
    /// ablation table 3 — essential for stability).
    #[serde(default = "default_opd_top_p")]
    pub top_p: f64,

    /// Rollout length cap. §6 default 7K (Li et al. 2026 §6.1 — teacher
    /// reward decays past this point; explicit override required for
    /// longer).
    #[serde(default = "default_opd_max_tokens")]
    pub max_tokens: usize,

    /// Stable-OPD mode (§3.1, §3.9). Defaults to `Auto`.
    #[serde(default)]
    pub stable_opd: StableOpdMode,

    /// Discount factor on the per-token reverse KL. Lu (2025) §3.1
    /// picks γ = 0 ("variance dominates the bias gain"). Exposed for
    /// research parity but the production default is 0.
    #[serde(default)]
    pub discount: f64,

    /// PPO-style clip epsilon for the importance-sampling ratio.
    /// Inherited from the GRPO loss; default matches `GrpoConfig`.
    #[serde(default = "default_opd_clip_eps")]
    pub clip_epsilon: f64,

    /// Learning rate. §6: 10× the FullFT optimum (Schulman 2025 LoRA
    /// Without Regret), which for kiln's bf16 training is ~1e-5. The
    /// per-model auto-pick rule lives in `kiln-server`; this is the
    /// safe fallback.
    #[serde(default = "default_opd_lr")]
    pub learning_rate: f64,

    /// LoRA rank. §6: 16 (laptop) / 32 (prosumer) / 64+ (corporate).
    /// The trainer doesn't know its tier; the kiln-server endpoint
    /// picks per-tier and overrides.
    #[serde(default = "default_rank")]
    pub lora_rank: usize,

    /// LoRA α. §6: 32 (community standard).
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,

    /// If set, continue training from this adapter (continual learning,
    /// §3.6 refresh recipe). When the teacher is the user's *prior*
    /// adapter, this is Lu's `recover-instruction-following` recipe.
    pub base_adapter: Option<String>,

    /// Output adapter name. Auto-generated if not set.
    pub output_name: Option<String>,

    /// Automatically load the resulting adapter when training completes.
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,

    /// Auto-checkpoint cadence (§3.9 auto-rollback). Every N steps.
    /// Default 10. Corporate-tier picks 5.
    #[serde(default = "default_opd_checkpoint_interval")]
    pub checkpoint_interval: Option<usize>,

    /// Deterministic seed. If `None`, the trainer picks one and records
    /// it in the replay log.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Optimizer. Defaults to AdamW per kiln convention.
    #[serde(default)]
    pub optimizer: Optimizer,

    /// Hard cap on remote-teacher $ spend, in USD. §8.6 cost lock.
    /// `None` falls back to the server-wide default. Applies only when
    /// the resolved `LogitSource` reports a non-zero $/token cost.
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
}

fn default_opd_temperature() -> f64 {
    1.0
}
fn default_opd_top_p() -> f64 {
    0.9
}
fn default_opd_max_tokens() -> usize {
    7168
}
fn default_opd_clip_eps() -> f64 {
    0.2
}
fn default_opd_lr() -> f64 {
    1e-5
}
fn default_opd_checkpoint_interval() -> Option<usize> {
    Some(10)
}
fn default_auto_load() -> bool {
    true
}

impl Default for OpdConfig {
    fn default() -> Self {
        Self {
            loss: OpdLossGranularity::default(),
            top_k: default_opd_top_k(),
            samples_per_prompt: default_opd_samples_per_prompt(),
            temperature: default_opd_temperature(),
            top_p: default_opd_top_p(),
            max_tokens: default_opd_max_tokens(),
            stable_opd: StableOpdMode::default(),
            discount: 0.0,
            clip_epsilon: default_opd_clip_eps(),
            learning_rate: default_opd_lr(),
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: default_opd_checkpoint_interval(),
            seed: None,
            optimizer: Optimizer::default(),
            max_cost_usd: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-step loss computation
// ---------------------------------------------------------------------------

/// Inputs to one OPD step's loss computation. A "step" here is one
/// student-sampled trajectory (one rollout). The trainer collects N
/// rollouts per prompt × M prompts per batch and reduces over them.
///
/// The split between this struct and the trainer's surrounding code
/// matches `trainer.rs::grpo_loss`: the trainer owns the sampling,
/// hidden-state forward pass, optimizer step, and replay log; this
/// function owns the per-rollout teacher query + KL math.
#[derive(Debug)]
pub struct OpdStepInputs<'a> {
    /// Full tokenized rollout (prompt + completion).
    pub tokens: &'a [u32],
    /// Positions in `tokens` that contribute to the loss. Typically the
    /// completion's assistant-token positions. The OPD-loss kernel
    /// expects `label_mask` of length `tokens.len()`, but the rollout's
    /// trainer constructs that mask once per group; here we accept the
    /// explicit position list and build the mask internally to match
    /// the §3.1 pseudocode interface.
    pub active_positions: &'a [usize],
    /// Student hidden states at *all* `tokens` positions, shape
    /// `[1, tokens.len(), hidden_size]`. Produced by the trainer's
    /// segment-checkpointed forward pass (`trainer.rs` already produces
    /// this — `opd_train` will plumb it in).
    pub student_hidden: &'a Tensor,
    /// Frozen LM head weights, shape `[H, V]`. Matches the layout used
    /// by `kiln-flce-kernel` and `kiln-model::forward::embed_tokens_t`.
    pub head_t: &'a Tensor,
    /// Teacher source. Queried for top-K logprobs at `active_positions`.
    pub teacher: Arc<dyn LogitSource>,
    /// Loss granularity (§3.1).
    pub loss: OpdLossGranularity,
    /// Top-K size for `TeacherTopK`. Ignored for `SampledToken` /
    /// `FullVocab`.
    pub top_k: usize,
    /// Chunk size along the active-token axis for the kernel. Falls
    /// back to `DEFAULT_CHUNK_SIZE` (= 4096) when 0.
    pub chunk_size: usize,
}

/// Output of one OPD step's loss computation.
///
/// The trainer feeds:
/// * `per_position_kl` → into the GRPO importance-sampling loss as the
///   per-token advantage (negated; §3.1 step 4).
/// * `mean_kl` → into the optimizer-tracked loss metric and into the
///   §3.8 diagnostic stack.
/// * `student_logprob_sum`, `teacher_logprob_sum` → into the §3.8
///   diagnostic stack (entropy gap, overlap ratio computed in #23).
#[derive(Debug)]
pub struct OpdStepOutputs {
    /// Per-position reverse KL, shape `[T_active]` f32. Lives on the
    /// same device as `student_hidden`.
    pub per_position_kl: Tensor,
    /// Mean over active positions of `per_position_kl`. Scalar f32.
    pub mean_kl: Tensor,
    /// Number of active positions (T_active).
    pub active_count: usize,
}

/// Compute the per-position reverse-KL loss for one OPD step.
///
/// This is the §3.1 pseudocode's "compute reward" step, made explicit:
///
/// 1. Build the `label_mask` from `active_positions`.
/// 2. Query the teacher for top-K logprobs at those positions.
/// 3. Run the OPD-loss kernel.
/// 4. Return `(per_position_kl, mean_kl)`.
///
/// The trainer's surrounding code is responsible for:
///
/// * Sampling the rollout (`do_group_rollout` equivalent in
///   `trainer.rs`).
/// * Producing `student_hidden` via the segment-checkpointed forward
///   pass.
/// * Multiplying `-per_position_kl` into the importance-sampling
///   `grpo_loss` via per-position advantage broadcast (existing GRPO
///   path uses a scalar advantage — the OPD path generalises that to a
///   vector. The refactor is `grpo_loss(... advantage: Tensor ...)`;
///   the call site is unchanged for the scalar case).
pub fn opd_step_loss(inputs: OpdStepInputs<'_>) -> Result<OpdStepOutputs> {
    let OpdStepInputs {
        tokens,
        active_positions,
        student_hidden,
        head_t,
        teacher,
        loss,
        top_k,
        chunk_size,
    } = inputs;

    if active_positions.is_empty() {
        return Err(anyhow!(
            "opd_step_loss called with no active positions — caller should short-circuit"
        ));
    }
    let seq_len = tokens.len();
    let label_mask: Vec<bool> = {
        let mut m = vec![false; seq_len];
        for &p in active_positions {
            if p >= seq_len {
                return Err(anyhow!(
                    "active position {} out of range for seq_len {}",
                    p,
                    seq_len
                ));
            }
            m[p] = true;
        }
        m
    };
    let active_count = active_positions.len();

    // Caps check: pick the actual top-K from the teacher.
    let caps = teacher.capabilities();
    let resolved_top_k = match loss {
        OpdLossGranularity::SampledToken => 1,
        OpdLossGranularity::TeacherTopK => top_k.min(caps.max_top_k),
        OpdLossGranularity::FullVocab => caps.vocab_size,
    };

    // Query the teacher.
    let request_top_k = match loss {
        OpdLossGranularity::FullVocab => None,
        _ => Some(resolved_top_k),
    };
    let batch = teacher
        .fetch_logprobs(tokens, active_positions, request_top_k)
        .with_context(|| {
            format!(
                "fetch_logprobs from teacher {:?} for {} positions",
                caps.teacher_id, active_count
            )
        })?;

    let (teacher_topk_indices, teacher_topk_logprobs) = match batch {
        LogprobBatch::TopK(t) => (t.indices, t.logprobs),
        LogprobBatch::FullVocab { logprobs, vocab_size: _ } => {
            // For full-vocab, indices are 0..V repeated per position.
            let mut indices = Vec::with_capacity(active_count * resolved_top_k);
            for _ in 0..active_count {
                for v in 0..resolved_top_k as u32 {
                    indices.push(v);
                }
            }
            (indices, logprobs)
        }
    };

    let device = student_hidden.device().clone();
    // chunk_size is plumbed through to the kernel when we switch to
    // Phase B end-to-end (next milestone). Phase A doesn't chunk —
    // the autograd graph holds the whole `[T_active, K]` tensor live
    // anyway, so chunking buys nothing. Keep the field on the
    // `OpdStepInputs` so the call site doesn't churn when we flip.
    let _chunk_size = if chunk_size == 0 {
        DEFAULT_CHUNK_SIZE
    } else {
        chunk_size
    };

    // Phase A is the default until the CUDA kernel lands; the trainer
    // can opt into Phase B via the env var. Phase A's autograd graph
    // is built off `student_hidden`'s autograd parents (LoRA Vars), so
    // the resulting `mean_kl.backward()` flows gradients into the LoRA
    // parameters the trainer is optimizing.
    let per_position_kl = opd_top_k_reverse_kl_phase_a_per_position(
        student_hidden,
        head_t,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        &label_mask,
        resolved_top_k,
        &device,
    )?;
    let mean_kl = per_position_kl
        .mean_all()
        .context("mean per-position KL")?;

    Ok(OpdStepOutputs {
        per_position_kl,
        mean_kl,
        active_count,
    })
}

/// Convenience: build an `OpdStepInputs` and run the loss + mean_kl
/// in one call. Used by unit tests.
#[allow(clippy::too_many_arguments)]
pub fn opd_step_loss_simple(
    tokens: &[u32],
    active_positions: &[usize],
    student_hidden: &Tensor,
    head_t: &Tensor,
    teacher: Arc<dyn LogitSource>,
    loss: OpdLossGranularity,
    top_k: usize,
) -> Result<OpdStepOutputs> {
    opd_step_loss(OpdStepInputs {
        tokens,
        active_positions,
        student_hidden,
        head_t,
        teacher,
        loss,
        top_k,
        chunk_size: DEFAULT_CHUNK_SIZE,
    })
}

// ---------------------------------------------------------------------------
// Stable-OPD loss composition (Luo et al. 2026 §4.2 + grand plan §3.1)
// ---------------------------------------------------------------------------

/// §6 default β_kl — reference-policy KL coefficient. Luo et al. 2026
/// ablation table 2; engages via the `LengthInflation` guardrail when
/// `stable_opd = "auto"`.
pub const fn default_beta_kl() -> f64 {
    0.01
}

/// §6 default λ_sft — golden-trajectory mixture weight. Luo et al. 2026.
pub const fn default_lambda_sft() -> f64 {
    0.1
}

/// Resolved Stable-OPD coefficients for one training step. The trainer
/// constructs this from the configured [`StableOpdMode`] and the
/// guardrail engine's auto-tuning decisions.
#[derive(Debug, Clone, Copy)]
pub struct StableOpdCoefficients {
    /// β — weight on the reference-policy KL term.
    pub beta_kl: f64,
    /// λ — weight on the SFT golden-trajectory term.
    pub lambda_sft: f64,
}

impl StableOpdCoefficients {
    /// Off-mode: both terms zero (Luo et al. baseline).
    pub const fn off() -> Self {
        Self {
            beta_kl: 0.0,
            lambda_sft: 0.0,
        }
    }

    /// Auto-mode default values (paper-cited, §6).
    pub const fn auto_default() -> Self {
        Self {
            beta_kl: default_beta_kl(),
            lambda_sft: default_lambda_sft(),
        }
    }

    /// §3.9 `BumpStableOpd` action: double both coefficients. Used by
    /// the guardrail when RepRate stays high.
    pub fn doubled(self) -> Self {
        Self {
            beta_kl: self.beta_kl * 2.0,
            lambda_sft: self.lambda_sft * 2.0,
        }
    }

    /// Resolve from configured mode (the `auto` path uses the §6
    /// defaults until the guardrail bumps them).
    pub fn from_mode(mode: StableOpdMode) -> Self {
        match mode {
            StableOpdMode::Off => Self::off(),
            StableOpdMode::Auto => Self::auto_default(),
            StableOpdMode::Manual { kl_beta, sft_lambda } => Self {
                beta_kl: kl_beta,
                lambda_sft: sft_lambda,
            },
        }
    }
}

impl Default for StableOpdCoefficients {
    fn default() -> Self {
        Self::auto_default()
    }
}

/// Inputs to the per-step Stable-OPD loss composition.
///
/// All tensors live on the same device. The OPD per-position KL is
/// what `opd_step_loss` already returns. The two extra Stable-OPD
/// terms — reference-KL and golden-SFT — are independent autograd
/// graphs the trainer maintains. We compose the three into a single
/// scalar that the trainer's `.backward()` can root on.
#[derive(Debug)]
pub struct StableOpdLossInputs<'a> {
    /// Per-position OPD KL from the loss kernel, shape `[T_active]`.
    pub per_position_kl: &'a Tensor,
    /// Optional `KL(π_θ || π_ref)` per-position tensor of the same
    /// shape. Caller computes this from a reference forward pass (the
    /// previous-checkpoint adapter held as `π_ref`). If `None`, the
    /// β·KL_ref term is skipped (equivalent to β_kl = 0 for this
    /// step).
    pub per_position_kl_ref: Option<&'a Tensor>,
    /// Optional SFT loss on a golden-trajectory minibatch (already a
    /// scalar tensor). Caller computes this from a separate
    /// `kiln-flce-kernel` cross-entropy pass on the golden batch. If
    /// `None`, the λ·SFT term is skipped.
    pub sft_loss: Option<&'a Tensor>,
    /// Stable-OPD coefficients as resolved by the guardrail engine.
    pub coefficients: StableOpdCoefficients,
}

/// Output of [`compute_stable_opd_loss`].
#[derive(Debug)]
pub struct StableOpdLossOutputs {
    /// `L_total` — the scalar the trainer calls `.backward()` on.
    pub total: Tensor,
    /// `mean(L_OPD)` — the per-token reverse KL piece (the
    /// "headline" metric the dashboard shows).
    pub mean_opd: Tensor,
    /// `β · mean(KL_ref)` — the reference-policy regularizer piece,
    /// or zeros tensor when omitted.
    pub mean_kl_ref: Tensor,
    /// `λ · sft_loss` — the golden-SFT piece, or zeros tensor when
    /// omitted.
    pub sft_term: Tensor,
}

// ---------------------------------------------------------------------------
// Cold-start auto-injection (§3.1 + §8.10)
// ---------------------------------------------------------------------------

/// §3.1 default threshold: median initial overlap below this triggers
/// cold-start (Li et al. 2026 §5.1 — "thinking-pattern mismatch"
/// indicator).
pub const COLD_START_OVERLAP_THRESHOLD: f64 = 0.5;

/// §3.1 default cold-start length when triggered: ~2 epochs over
/// 5–10K teacher rollouts. We pick the lower bound here as the
/// pit-of-success default; the trainer can scale up if the user has
/// a larger seed corpus.
pub const COLD_START_DEFAULT_PROMPTS: usize = 5_000;
pub const COLD_START_DEFAULT_EPOCHS: usize = 2;

/// Decision returned by [`cold_start_probe`]. The trainer consults
/// this before the first OPD step:
///
/// - `Skip`: initial overlap is already ≥ threshold; OPD can run
///   directly.
/// - `InjectSft { ... }`: silently run the prescribed SFT pre-phase
///   first. The user sees one progress bar labelled "preparing your
///   model"; auto-cold-start is the §8.10 mandate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColdStartDecision {
    Skip,
    InjectSft {
        prompts: usize,
        epochs: usize,
        observed_overlap: f64,
    },
}

/// Compute initial overlap from a slice of (student top-K, teacher
/// top-K) pairs. The "overlap" is |S^p ∩ S^q| / K per position, then
/// the median over the probe positions.
///
/// `pairs[i] = (student_topk_indices, teacher_topk_indices)` — both
/// length K. Caller is responsible for producing the student's
/// top-K (via a full-vocab forward pass over a small probe set; the
/// trainer-side helper that does this lives in the trainer's
/// forward-pass module which uses `kiln-model::forward::model_forward_*`).
pub fn compute_initial_overlap(pairs: &[(Vec<u32>, Vec<u32>)]) -> f64 {
    if pairs.is_empty() {
        return 1.0; // Vacuously: no positions to test, assume aligned.
    }
    let mut ratios: Vec<f64> = pairs
        .iter()
        .map(|(s, q)| {
            if s.is_empty() {
                return 1.0;
            }
            let q_set: std::collections::HashSet<u32> = q.iter().copied().collect();
            let inter = s.iter().filter(|i| q_set.contains(i)).count() as f64;
            inter / (s.len() as f64)
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = ratios.len() / 2;
    if ratios.len() % 2 == 0 && ratios.len() > 1 {
        (ratios[mid - 1] + ratios[mid]) / 2.0
    } else {
        ratios[mid]
    }
}

/// Run the §3.1 cold-start probe against a set of student/teacher
/// top-K pairs. Returns a [`ColdStartDecision`] the trainer respects
/// before the first OPD step.
///
/// `threshold` is [`COLD_START_OVERLAP_THRESHOLD`] by default;
/// `prompts` / `epochs` are the cold-start phase length used when the
/// probe triggers. The trainer prepares the (student_topk,
/// teacher_topk) pairs by:
/// 1. Sampling ~50 prompts from the user's training set.
/// 2. Forwarding them through the student (full-vocab logits, then
///    `topk(k=top_k)`).
/// 3. Querying the teacher for its top-K at the same positions.
/// 4. Pairing position-wise and calling this function.
///
/// Per the grand plan §3.1: "the user sees one progress bar labelled
/// 'preparing your model' and never learns that two distinct training
/// paradigms are running back-to-back." This function is the policy
/// component; the orchestration (running the SFT phase silently)
/// belongs to the trainer body.
pub fn cold_start_probe(
    pairs: &[(Vec<u32>, Vec<u32>)],
    threshold: f64,
    prompts: usize,
    epochs: usize,
) -> ColdStartDecision {
    let overlap = compute_initial_overlap(pairs);
    if overlap >= threshold {
        ColdStartDecision::Skip
    } else {
        ColdStartDecision::InjectSft {
            prompts,
            epochs,
            observed_overlap: overlap,
        }
    }
}

/// Same as [`cold_start_probe`] with the §3.1 default thresholds.
pub fn cold_start_probe_default(pairs: &[(Vec<u32>, Vec<u32>)]) -> ColdStartDecision {
    cold_start_probe(
        pairs,
        COLD_START_OVERLAP_THRESHOLD,
        COLD_START_DEFAULT_PROMPTS,
        COLD_START_DEFAULT_EPOCHS,
    )
}

/// Compose the §3.1 Stable-OPD loss:
///
/// ```text
/// L_total = mean(per_position_kl)
///         + β_kl · mean(per_position_kl_ref)
///         + λ_sft · sft_loss
/// ```
///
/// All terms autograd-attach to their respective parents (student LoRA
/// Vars for the OPD term, reference-policy graph for KL_ref, golden-
/// SFT graph for sft_loss). The trainer calls `.backward()` on
/// `output.total`.
///
/// When `per_position_kl_ref` is `None` we treat the β term as zero
/// without building any graph for it (Stable-OPD `Off` mode, or when
/// no reference adapter is configured). Same for `sft_loss`.
pub fn compute_stable_opd_loss(inputs: StableOpdLossInputs<'_>) -> Result<StableOpdLossOutputs> {
    let device = inputs.per_position_kl.device();
    let mean_opd = inputs
        .per_position_kl
        .mean_all()
        .context("mean(per_position_kl)")?;

    let mean_kl_ref = match inputs.per_position_kl_ref {
        Some(t) => t.mean_all().context("mean(per_position_kl_ref)")?,
        None => Tensor::new(0.0_f32, device).context("zero kl_ref scalar")?,
    };

    let sft_term = match inputs.sft_loss {
        Some(t) => t.clone(),
        None => Tensor::new(0.0_f32, device).context("zero sft scalar")?,
    };

    let beta = inputs.coefficients.beta_kl;
    let lambda = inputs.coefficients.lambda_sft;

    // total = mean_opd + β · mean_kl_ref + λ · sft_term.
    let kl_ref_scaled = if beta == 0.0 {
        Tensor::new(0.0_f32, device)?
    } else {
        mean_kl_ref.affine(beta, 0.0)?
    };
    let sft_scaled = if lambda == 0.0 {
        Tensor::new(0.0_f32, device)?
    } else {
        sft_term.affine(lambda, 0.0)?
    };
    let total = (&mean_opd + &kl_ref_scaled)?.add(&sft_scaled)?;
    Ok(StableOpdLossOutputs {
        total,
        mean_opd,
        mean_kl_ref,
        sft_term,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logit_source::FixtureLogitSource;
    use candle_core::{DType, Device};

    /// End-to-end: feed a tokenized rollout, a fixture teacher, and a
    /// random `student_hidden` into `opd_step_loss`, confirm we get a
    /// scalar mean_kl + per-position vector of the right shape and
    /// that the math matches the Phase A reference directly.
    #[test]
    fn opd_step_loss_matches_kernel_directly() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 8;
        let hidden_size = 8;
        let vocab_size = 64;
        let top_k = 4;

        // Hidden + head are arbitrary smooth tensors.
        let hidden_vec: Vec<f32> = (0..(seq_len * hidden_size))
            .map(|i| (i as f32 * 0.011).sin() * 0.3)
            .collect();
        let student_hidden = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size), &device)?;
        let head_vec: Vec<f32> = (0..(hidden_size * vocab_size))
            .map(|i| ((i as f32 + 11.0) * 0.005).cos() * 0.2)
            .collect();
        let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size), &device)?;

        let tokens: Vec<u32> = (0..seq_len).map(|i| ((i * 7 + 3) % vocab_size) as u32).collect();
        let active_positions = vec![3, 5, 7]; // arbitrary completion tokens

        // Build a fixture teacher with deterministic top-K at each active position.
        let mut fixture = FixtureLogitSource::uniform_topk("test", vocab_size, top_k);
        let h = FixtureLogitSource::hash_tokens(&tokens);
        for &pos in &active_positions {
            let idx: Vec<u32> = (0..top_k as u32)
                .map(|k| (pos as u32 * 5 + k * 11) % vocab_size as u32)
                .collect();
            let lp: Vec<f32> = (0..top_k)
                .map(|k| -((pos + 1) as f32).ln() - (k as f32) * 0.3)
                .collect();
            fixture.insert(h, pos, idx, lp);
        }
        let teacher: Arc<dyn LogitSource> = Arc::new(fixture);

        let out = opd_step_loss_simple(
            &tokens,
            &active_positions,
            &student_hidden,
            &head_t,
            teacher.clone(),
            OpdLossGranularity::TeacherTopK,
            top_k,
        )?;

        // Per-position vector must have length T_active.
        let per_pos: Vec<f32> = out.per_position_kl.to_vec1()?;
        assert_eq!(per_pos.len(), active_positions.len());
        // Mean matches the per-position recompute.
        let mean_recomputed = per_pos.iter().sum::<f32>() / (per_pos.len() as f32);
        let mean_kl = out.mean_kl.to_scalar::<f32>()?;
        assert!(
            (mean_kl - mean_recomputed).abs() < 1e-5,
            "mean_kl {mean_kl} vs recompute {mean_recomputed}"
        );
        // KL is non-negative.
        for (i, k) in per_pos.iter().enumerate() {
            assert!(*k >= -1e-5, "per_position_kl[{i}] = {k} went negative");
        }
        Ok(())
    }

    #[test]
    fn opd_step_loss_rejects_empty_positions() {
        let device = Device::Cpu;
        let student_hidden = Tensor::zeros((1, 4, 8), DType::F32, &device).unwrap();
        let head_t = Tensor::zeros((8, 16), DType::F32, &device).unwrap();
        let fixture = FixtureLogitSource::uniform_topk("test", 16, 4);
        let teacher: Arc<dyn LogitSource> = Arc::new(fixture);
        let err = opd_step_loss_simple(
            &[1, 2, 3, 4],
            &[],
            &student_hidden,
            &head_t,
            teacher,
            OpdLossGranularity::TeacherTopK,
            4,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no active positions"));
    }

    #[test]
    fn opd_request_round_trips_through_serde() {
        let req = OpdRequest {
            prompts: vec![OpdPrompt {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: "Evaluate ∫_0^∞ e^{-x^2} dx".into(),
                }],
            }],
            dataset_path: None,
            teacher: "qwen3.6-27b@local".into(),
            config: OpdConfig::default(),
            post_eval: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: OpdRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.teacher, req.teacher);
        assert_eq!(parsed.config.top_k, default_opd_top_k());
        assert_eq!(parsed.config.samples_per_prompt, default_opd_samples_per_prompt());
        assert!(matches!(parsed.config.loss, OpdLossGranularity::TeacherTopK));
    }

    #[test]
    fn opd_config_defaults_match_grand_plan_section_6() {
        let cfg = OpdConfig::default();
        assert_eq!(cfg.top_k, 32);
        assert_eq!(cfg.samples_per_prompt, 4);
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.max_tokens, 7168);
        assert_eq!(cfg.discount, 0.0);
        assert_eq!(cfg.clip_epsilon, 0.2);
        assert!(matches!(cfg.stable_opd, StableOpdMode::Auto));
        assert!(matches!(cfg.loss, OpdLossGranularity::TeacherTopK));
        assert_eq!(cfg.checkpoint_interval, Some(10));
    }

    #[test]
    fn stable_opd_defaults_match_section_6_paper_cites() {
        let auto = StableOpdCoefficients::auto_default();
        assert!((auto.beta_kl - 0.01).abs() < 1e-9);
        assert!((auto.lambda_sft - 0.1).abs() < 1e-9);
        let off = StableOpdCoefficients::off();
        assert_eq!(off.beta_kl, 0.0);
        assert_eq!(off.lambda_sft, 0.0);
    }

    #[test]
    fn stable_opd_doubled_doubles_both() {
        let auto = StableOpdCoefficients::auto_default();
        let doubled = auto.doubled();
        assert!((doubled.beta_kl - 0.02).abs() < 1e-9);
        assert!((doubled.lambda_sft - 0.2).abs() < 1e-9);
        // Doubling twice should land at 0.04 / 0.4 — same as the
        // §3.9 guardrail's runaway escalation.
        let four = doubled.doubled();
        assert!((four.beta_kl - 0.04).abs() < 1e-9);
        assert!((four.lambda_sft - 0.4).abs() < 1e-9);
    }

    #[test]
    fn stable_opd_loss_composition_matches_paper_formula() -> Result<()> {
        let device = Device::Cpu;
        // OPD per-position: 5 active positions, values 0.1..0.5
        let kl = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5], 5, &device)?;
        let kl_ref = Tensor::from_vec(vec![0.05f32, 0.05, 0.05, 0.05, 0.05], 5, &device)?;
        let sft = Tensor::new(2.0_f32, &device)?;
        let coeffs = StableOpdCoefficients {
            beta_kl: 0.01,
            lambda_sft: 0.1,
        };
        let out = compute_stable_opd_loss(StableOpdLossInputs {
            per_position_kl: &kl,
            per_position_kl_ref: Some(&kl_ref),
            sft_loss: Some(&sft),
            coefficients: coeffs,
        })?;
        // mean(opd) = (0.1+0.2+0.3+0.4+0.5)/5 = 0.3
        // mean(kl_ref) = 0.05
        // total = 0.3 + 0.01 * 0.05 + 0.1 * 2.0 = 0.30 + 0.0005 + 0.20 = 0.5005
        let total = out.total.to_scalar::<f32>()?;
        let mean_opd = out.mean_opd.to_scalar::<f32>()?;
        let mean_ref = out.mean_kl_ref.to_scalar::<f32>()?;
        assert!((mean_opd - 0.3).abs() < 1e-5);
        assert!((mean_ref - 0.05).abs() < 1e-5);
        assert!((total - 0.5005).abs() < 1e-4, "got total = {total}");
        Ok(())
    }

    #[test]
    fn stable_opd_loss_omits_optional_terms_when_none() -> Result<()> {
        let device = Device::Cpu;
        let kl = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], 3, &device)?;
        let coeffs = StableOpdCoefficients::off();
        let out = compute_stable_opd_loss(StableOpdLossInputs {
            per_position_kl: &kl,
            per_position_kl_ref: None,
            sft_loss: None,
            coefficients: coeffs,
        })?;
        // Off-mode: total = mean(opd) = 0.2
        let total = out.total.to_scalar::<f32>()?;
        let mean_kl_ref = out.mean_kl_ref.to_scalar::<f32>()?;
        let sft_term = out.sft_term.to_scalar::<f32>()?;
        assert!((total - 0.2).abs() < 1e-5);
        assert_eq!(mean_kl_ref, 0.0);
        assert_eq!(sft_term, 0.0);
        Ok(())
    }

    #[test]
    fn cold_start_overlap_zero_when_no_intersection() {
        // K=4 each; no overlap.
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![10u32, 11, 12, 13]),
            (vec![5u32, 6, 7, 8], vec![20u32, 21, 22, 23]),
        ];
        assert!(compute_initial_overlap(&pairs).abs() < 1e-9);
    }

    #[test]
    fn cold_start_overlap_one_when_identical() {
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 4]),
            (vec![5u32, 6, 7, 8], vec![5u32, 6, 7, 8]),
        ];
        assert!((compute_initial_overlap(&pairs) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cold_start_overlap_is_median() {
        // Three positions, overlap ratios 0.25, 0.5, 0.75. Median = 0.5.
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]),     // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 11, 12]),       // 0.5
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 12]),        // 0.75
        ];
        assert!((compute_initial_overlap(&pairs) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cold_start_probe_triggers_below_threshold() {
        // Median overlap < 0.5 — should inject SFT.
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]), // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]), // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 11, 12]),   // 0.5
        ];
        // Median of [0.25, 0.25, 0.5] = 0.25 < 0.5.
        match cold_start_probe_default(&pairs) {
            ColdStartDecision::InjectSft {
                prompts,
                epochs,
                observed_overlap,
            } => {
                assert_eq!(prompts, COLD_START_DEFAULT_PROMPTS);
                assert_eq!(epochs, COLD_START_DEFAULT_EPOCHS);
                assert!((observed_overlap - 0.25).abs() < 1e-9);
            }
            other => panic!("expected InjectSft, got {other:?}"),
        }
    }

    #[test]
    fn cold_start_probe_skips_above_threshold() {
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 12]), // 0.75
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 12]), // 0.75
        ];
        assert_eq!(
            cold_start_probe_default(&pairs),
            ColdStartDecision::Skip
        );
    }

    #[test]
    fn cold_start_probe_empty_pairs_is_skip() {
        // Edge case: no probe positions ⇒ vacuously aligned ⇒ Skip.
        assert_eq!(
            cold_start_probe_default(&[]),
            ColdStartDecision::Skip
        );
    }

    #[test]
    fn distill_refresh_request_round_trips() {
        let req = DistillRefreshRequest {
            name: "company-assistant".into(),
            new_data: NewKnowledgeSource::Dataset {
                dataset: "q4-2026-internal-docs".into(),
            },
            behavioural_teacher: "company-assistant@v17".into(),
            background_chat: "tulu3".into(),
            require_if_eval_recovery: 0.95,
            require_internal_qa_gain: 0.05,
            config: OpdConfig::default(),
            post_eval: None,
        };
        let s = serde_json::to_string(&req).unwrap();
        let parsed: DistillRefreshRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.name, req.name);
        assert_eq!(parsed.behavioural_teacher, "company-assistant@v17");
        assert_eq!(parsed.background_chat, "tulu3");
        assert!((parsed.require_if_eval_recovery - 0.95).abs() < 1e-9);
        assert!((parsed.require_internal_qa_gain - 0.05).abs() < 1e-9);
        match parsed.new_data {
            NewKnowledgeSource::Dataset { dataset } => {
                assert_eq!(dataset, "q4-2026-internal-docs");
            }
            other => panic!("expected Dataset, got {other:?}"),
        }
    }

    #[test]
    fn distill_merge_request_round_trips() {
        let req = DistillMergeRequest {
            name: "unified-coder".into(),
            sources: vec![
                DistillMergeSource { adapter: "rust-helper".into(), weight: 1.0 },
                DistillMergeSource { adapter: "python-helper".into(), weight: 1.0 },
                DistillMergeSource { adapter: "sql-helper".into(), weight: 0.7 },
            ],
            student: "base".into(),
            rollout_budget: 5000,
            config: OpdConfig::default(),
            post_eval: None,
        };
        let s = serde_json::to_string(&req).unwrap();
        let parsed: DistillMergeRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.sources.len(), 3);
        assert!((parsed.sources[2].weight - 0.7).abs() < 1e-9);
        assert_eq!(parsed.rollout_budget, 5000);
    }

    #[test]
    fn distill_pump_request_three_modes_parse() {
        let domain: DistillPumpRequest = serde_json::from_str(
            r#"{"name":"math-lora","teacher":"qwen3.6@local","mode":{"domain":"math_reasoning"}}"#,
        )
        .unwrap();
        match domain.mode {
            DistillPumpMode::Domain { domain } => assert_eq!(domain, "math_reasoning"),
            other => panic!("expected Domain, got {other:?}"),
        }
        assert!(domain.use_cache);

        let wide: DistillPumpRequest = serde_json::from_str(
            r#"{"name":"gen-lora","teacher":"qwen3.6@local","mode":{"wide":true}}"#,
        )
        .unwrap();
        assert!(matches!(wide.mode, DistillPumpMode::Wide { wide: true }));

        let examples: DistillPumpRequest = serde_json::from_str(
            r#"{"name":"my-style","teacher":"x","mode":{"examples":[{"messages":[{"role":"user","content":"hi"}]}]}}"#,
        )
        .unwrap();
        match examples.mode {
            DistillPumpMode::Examples { examples } => assert_eq!(examples.len(), 1),
            other => panic!("expected Examples, got {other:?}"),
        }
    }

    #[test]
    fn distill_self_request_four_modes_parse() {
        for mode_str in [
            "ground_truth_conditioning",
            "conciseness",
            "document_as_pi",
            "reverse_teacher",
        ] {
            let json = format!(r#"{{"name":"x","mode":"{mode_str}"}}"#);
            let req: DistillSelfRequest = serde_json::from_str(&json).unwrap();
            // Mode enum parses without panic; round-trip back to the same string.
            let s = serde_json::to_string(&req).unwrap();
            assert!(
                s.contains(mode_str),
                "expected mode {mode_str} in serialized {s}"
            );
        }
    }

    #[test]
    fn distill_refresh_request_accepts_inline_examples() {
        let json = r#"{
            "name": "demo",
            "new_data": {"examples": [
                {"messages":[{"role":"user","content":"hi"}]}
            ]},
            "behavioural_teacher": "self@v1"
        }"#;
        let req: DistillRefreshRequest = serde_json::from_str(json).unwrap();
        match req.new_data {
            NewKnowledgeSource::Inline { examples } => {
                assert_eq!(examples.len(), 1);
                assert_eq!(examples[0].messages[0].content, "hi");
            }
            other => panic!("expected Inline, got {other:?}"),
        }
        // Defaults applied.
        assert_eq!(req.background_chat, "tulu3");
        assert!((req.require_if_eval_recovery - 0.95).abs() < 1e-9);
    }

    /// Stable-OPD `BumpStableOpd` semantics: doubled coefficients
    /// scale the total linearly when β=0 (so doubling is a no-op),
    /// but matter when β>0.
    #[test]
    fn stable_opd_doubled_changes_loss_when_ref_present() -> Result<()> {
        let device = Device::Cpu;
        let kl = Tensor::from_vec(vec![0.1_f32; 4], 4, &device)?;
        let kl_ref = Tensor::from_vec(vec![1.0_f32; 4], 4, &device)?;
        let base = StableOpdCoefficients::auto_default();
        let doubled = base.doubled();
        let out_base = compute_stable_opd_loss(StableOpdLossInputs {
            per_position_kl: &kl,
            per_position_kl_ref: Some(&kl_ref),
            sft_loss: None,
            coefficients: base,
        })?;
        let out_doubled = compute_stable_opd_loss(StableOpdLossInputs {
            per_position_kl: &kl,
            per_position_kl_ref: Some(&kl_ref),
            sft_loss: None,
            coefficients: doubled,
        })?;
        let base_total = out_base.total.to_scalar::<f32>()?;
        let doubled_total = out_doubled.total.to_scalar::<f32>()?;
        // base: 0.1 + 0.01 * 1.0 = 0.11
        // doubled: 0.1 + 0.02 * 1.0 = 0.12
        assert!((base_total - 0.11).abs() < 1e-5);
        assert!((doubled_total - 0.12).abs() < 1e-5);
        Ok(())
    }
}

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
//! # What's in the module
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
//! * `opd_train` — the full §3.1 trainer body. Mirrors `sft_train`'s
//!   structure but with `opd_step_loss` for the per-step loss.
//! * `build_local_teacher_fixture` — in-process LocalTeacher path: run
//!   the loaded model forward once per prompt, stash top-K teacher
//!   logprobs in a `FixtureLogitSource` keyed by tokens_hash. Used by
//!   `run_distill_refresh` / `run_distill_pump` / `run_distill_self`
//!   to materialise the §3.2 Local teacher when the registered alias
//!   resolves to a Local kind.
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
///
/// `teacher_extra_messages`, when present, are **prepended only on the
/// teacher's side** of the OPD step. The student rolls out from
/// `messages` alone; the teacher computes its logprobs as if it had
/// also seen `teacher_extra_messages` — typically a few pristine
/// examples, an expanded schema, a style guide, or anti-pattern
/// call-outs. The asymmetry sharpens the teacher's distribution at
/// rollout positions, giving reverse-KL a tighter target. Critical for
/// self-distillation, where teacher == student weights and the
/// asymmetric prefix is the only source of gradient signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpdPrompt {
    pub messages: Vec<ChatMessage>,
    /// Asymmetric teacher-only context. Empty = symmetric (default).
    /// When non-empty, the teacher's logprobs are computed against
    /// `teacher_extra_messages ++ messages ++ rollout` while the
    /// student still rolls out from `messages` alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub teacher_extra_messages: Vec<ChatMessage>,
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
    /// Name of the registered eval suite used to measure
    /// instruction-following recovery. When set, the refreshed
    /// adapter is queued for a post-training eval against this suite
    /// with `min_accuracy = baseline * require_if_eval_recovery`
    /// (the prior adapter `name` is the baseline). On failure, the
    /// refreshed adapter is renamed with `.failed` — §8.7 auto-
    /// rollback contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub if_eval_suite: Option<String>,
    /// Name of the registered eval suite used to measure new-knowledge
    /// gain on the mid-trained material. When set, the refreshed
    /// adapter is queued for a post-training eval against this suite
    /// with `min_accuracy = baseline + require_internal_qa_gain`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_knowledge_eval_suite: Option<String>,
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

    /// Permit alpha/rank above the default safety limit for deliberate
    /// experiments.
    #[serde(default)]
    pub allow_high_lora_scale: bool,

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

    /// Number of training epochs. Default 1.
    #[serde(default = "default_opd_epochs")]
    pub epochs: usize,

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
    // Mid-flight checkpoint cadence. With OPD at ~150s/step (asymmetric
    // teacher conditioning, no skip-rate floor) a 25-step interval saves
    // every ~60 minutes — small enough that a wall-time kill or crash
    // doesn't burn more than ~60 minutes of LoRA progress, large enough
    // that I/O isn't a hot path.
    Some(25)
}
fn default_auto_load() -> bool {
    true
}
fn default_opd_epochs() -> usize {
    1
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
            allow_high_lora_scale: false,
            base_adapter: None,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: default_opd_checkpoint_interval(),
            seed: None,
            optimizer: Optimizer::default(),
            max_cost_usd: None,
            epochs: 1,
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
    /// Full tokenized rollout (prompt + completion). Used for the
    /// student-side hidden state alignment, the label_mask, and as the
    /// teacher query token sequence *unless* `teacher_tokens` is set
    /// (asymmetric teacher conditioning).
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
    /// Teacher source. Queried for top-K logprobs at `active_positions`
    /// (or `teacher_active_positions` if set).
    pub teacher: Arc<dyn LogitSource>,
    /// Loss granularity (§3.1).
    pub loss: OpdLossGranularity,
    /// Top-K size for `TeacherTopK`. Ignored for `SampledToken` /
    /// `FullVocab`.
    pub top_k: usize,
    /// Chunk size along the active-token axis for the kernel. Falls
    /// back to `DEFAULT_CHUNK_SIZE` (= 4096) when 0.
    pub chunk_size: usize,
    /// Optional asymmetric teacher token sequence. When `Some`, this is
    /// what's sent to `teacher.fetch_logprobs`; `tokens` continues to
    /// drive the student-side state. Typical shape:
    /// `teacher_prefix_tokens ++ tokens` — i.e. the same rollout, but
    /// preceded by privileged context only the teacher sees. Length
    /// must match the longest active position in `teacher_active_positions`.
    pub teacher_tokens: Option<&'a [u32]>,
    /// Active positions in `teacher_tokens`'s frame. Pair-wise aligned
    /// with `active_positions` — position `i` in the loss kernel reads
    /// student logits at `active_positions[i]` (within `tokens`) and
    /// teacher logprobs at `teacher_active_positions[i]` (within
    /// `teacher_tokens`). Required when `teacher_tokens` is set; ignored
    /// otherwise.
    pub teacher_active_positions: Option<&'a [usize]>,
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
        teacher_tokens,
        teacher_active_positions,
    } = inputs;

    // Pair-wise alignment check for asymmetric teacher conditioning.
    if let (Some(t_tok), Some(t_act)) = (teacher_tokens, teacher_active_positions) {
        if t_act.len() != active_positions.len() {
            return Err(anyhow!(
                "opd_step_loss: teacher_active_positions.len() ({}) != active_positions.len() ({})",
                t_act.len(),
                active_positions.len()
            ));
        }
        for &p in t_act {
            if p >= t_tok.len() {
                return Err(anyhow!(
                    "opd_step_loss: teacher active position {} out of range for teacher_tokens.len() {}",
                    p,
                    t_tok.len()
                ));
            }
        }
    } else if teacher_tokens.is_some() ^ teacher_active_positions.is_some() {
        return Err(anyhow!(
            "opd_step_loss: teacher_tokens and teacher_active_positions must be set together"
        ));
    }
    let (query_tokens, query_positions): (&[u32], &[usize]) =
        match (teacher_tokens, teacher_active_positions) {
            (Some(t), Some(a)) => (t, a),
            _ => (tokens, active_positions),
        };

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
        .fetch_logprobs(query_tokens, query_positions, request_top_k)
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
        teacher_tokens: None,
        teacher_active_positions: None,
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

// ---------------------------------------------------------------------------
// §10.4 agentic OPD loss-shaping — SCoRe earliest-error + TIP tool-call
// reweighting + optional verifier reward blend.
// ---------------------------------------------------------------------------

/// §6 default — SCoRe earliest-error weight. Lyu et al. 2025.
/// Per-token loss multiplier within the earliest-divergence
/// neighbourhood (gradients up-weighted on the earliest "wrong"
/// token).
pub const fn default_score_earliest_weight() -> f64 {
    3.0
}

/// §6 default — SCoRe decay rate. Loss multiplier decays linearly
/// from `score_earliest_weight` at the earliest-divergence position
/// to `1.0` after `score_decay_steps` later positions.
pub const fn default_score_decay_steps() -> usize {
    16
}

/// §6 default — TIP tool-call upweight. Higher than prose because
/// "a wrong tool call sends the agent down a bad branch (not
/// recoverable in the current rollout)." §10.4 C.
pub const fn default_tip_tool_call_weight() -> f64 {
    2.0
}

/// §6 default — TIP tool-call-name upweight (vs tool-call-params).
/// Names are even higher-stakes than parameters because picking
/// the wrong tool is irrecoverable.
pub const fn default_tip_tool_name_weight() -> f64 {
    3.0
}

/// §6 default — verifier-reward blend coefficient λ_verifier.
/// §10.4 D: "anchors on outcome, doesn't drown out the per-token
/// signal." Auto-tuned.
pub const fn default_lambda_verifier() -> f64 {
    0.3
}

/// One position's token class for the TIP loss reweighting (§10.4 C).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TipTokenClass {
    /// Reasoning prose; weight = 1.0.
    Prose,
    /// Function/tool name token; higher stakes; weight =
    /// `tip_tool_name_weight`.
    ToolCallName,
    /// Parameters (JSON keys/values); weight = `tip_tool_call_weight`.
    ToolCallParams,
    /// Result tokens from a tool; masked from loss entirely (these
    /// are inputs the model didn't generate).
    ToolResult,
}

impl TipTokenClass {
    /// Loss multiplier for this class given the §10.4 C weights.
    pub fn multiplier(
        &self,
        tool_call_weight: f64,
        tool_name_weight: f64,
    ) -> f64 {
        match self {
            Self::Prose => 1.0,
            Self::ToolCallName => tool_name_weight,
            Self::ToolCallParams => tool_call_weight,
            // ToolResult positions are masked, not weighted —
            // callers should remove them from the active-position
            // set before constructing the per-position weights.
            Self::ToolResult => 0.0,
        }
    }
}

/// Per-token weights produced by [`compute_agentic_loss_weights`].
/// One f64 per active position; multiplied into the per-position
/// reverse-KL before it becomes the importance-sampling advantage.
pub type AgenticLossWeights = Vec<f64>;

/// Inputs to [`compute_agentic_loss_weights`] — the §10.4 A+B+C
/// shaping function.
#[derive(Debug, Clone)]
pub struct AgenticLossInputs {
    /// Per-position token class. Length = T_active.
    pub token_classes: Vec<TipTokenClass>,
    /// 0-indexed earliest-divergence position relative to the
    /// active-position set. `None` when SCoRe is disabled or the
    /// trajectory had no detected divergence.
    pub earliest_divergence: Option<usize>,
    /// SCoRe weights (defaults to `default_score_*`).
    pub score_earliest_weight: f64,
    /// SCoRe decay steps.
    pub score_decay_steps: usize,
    /// TIP weights (defaults to `default_tip_*`).
    pub tip_tool_call_weight: f64,
    pub tip_tool_name_weight: f64,
}

impl Default for AgenticLossInputs {
    fn default() -> Self {
        Self {
            token_classes: Vec::new(),
            earliest_divergence: None,
            score_earliest_weight: default_score_earliest_weight(),
            score_decay_steps: default_score_decay_steps(),
            tip_tool_call_weight: default_tip_tool_call_weight(),
            tip_tool_name_weight: default_tip_tool_name_weight(),
        }
    }
}

/// Compute the per-token loss multiplier vector that §10.4 wants
/// the trainer to apply to the per-position reverse KL before
/// turning it into the importance-sampling advantage.
///
/// Combines §10.4 B (SCoRe earliest-error decay) and §10.4 C (TIP
/// tool-call class weighting) multiplicatively per position.
pub fn compute_agentic_loss_weights(inputs: &AgenticLossInputs) -> AgenticLossWeights {
    let n = inputs.token_classes.len();
    let mut weights = vec![1.0_f64; n];
    // TIP per-class multiplier.
    for (i, class) in inputs.token_classes.iter().enumerate() {
        weights[i] *= class.multiplier(
            inputs.tip_tool_call_weight,
            inputs.tip_tool_name_weight,
        );
    }
    // SCoRe earliest-divergence schedule.
    if let Some(idx) = inputs.earliest_divergence {
        for i in idx..n.min(idx + inputs.score_decay_steps + 1) {
            let dist = i - idx;
            let frac = if inputs.score_decay_steps == 0 {
                0.0
            } else {
                (dist as f64) / (inputs.score_decay_steps as f64)
            };
            let extra = inputs.score_earliest_weight
                + (1.0 - inputs.score_earliest_weight) * frac.min(1.0);
            // The schedule modulates ON TOP of TIP. Multiplicative
            // composition: a tool-call-name position at earliest
            // divergence gets `score_earliest * tool_name_weight`.
            weights[i] *= extra.max(0.0);
        }
    }
    weights
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

/// §3.1 end-to-end algorithmic-loop validation. Given a synthetic
/// teacher (FixtureLogitSource), a trainable `head_t` proxy, and an
/// AdamW optimizer, run K optimization steps and verify:
///
/// 1. The loss strictly decreases over the run.
/// 2. The final loss is small (the student's K-support softmax
///    matches the teacher).
///
/// This is the smallest unit that proves the per-token reverse-KL
/// kernel + StableOpd loss composition + candle autograd + AdamW
/// state machinery together produce a real training signal. The
/// trainer body integration (#31) wires this primitive into the
/// kiln-model forward path + LoRA Vars + adapter save.
#[cfg(test)]
pub fn opd_train_synthetic_validation(
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    top_k: usize,
    num_steps: usize,
    learning_rate: f64,
) -> Result<(f32, f32)> {
    use candle_core::Var;
    use candle_nn::optim::{AdamW, ParamsAdamW};
    use candle_nn::Optimizer;

    let device = candle_core::Device::Cpu;

    // Trainable: a hidden-state-shaped Var. We're proving the loss
    // gradient flows back into something the optimizer can move.
    let init: Vec<f32> = (0..(seq_len * hidden_size))
        .map(|i| (i as f32 * 0.013).sin() * 0.3)
        .collect();
    let hidden = Var::from_vec(init, (1, seq_len, hidden_size), &device)?;

    // Frozen head — represents the LM projection. We pick a fixed
    // random head; the training surface is the hidden Var.
    let head_vec: Vec<f32> = (0..(hidden_size * vocab_size))
        .map(|i| ((i as f32 + 7.0) * 0.0007).cos() * 0.2)
        .collect();
    let head_t = candle_core::Tensor::from_vec(
        head_vec,
        (hidden_size, vocab_size),
        &device,
    )?;

    // Synthetic teacher: pick K random vocab indices per active
    // position; logprobs uniform over the K support (so the
    // student's target softmax is uniform-over-K).
    let mut label_mask = vec![false; seq_len];
    for i in 0..seq_len {
        if i % 2 == 1 {
            label_mask[i] = true;
        }
    }
    let active: Vec<usize> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    let active_count = active.len();

    let mut indices: Vec<u32> = Vec::with_capacity(active_count * top_k);
    let mut logprobs: Vec<f32> = Vec::with_capacity(active_count * top_k);
    for (row, _) in active.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k as u32 {
            let mut idx = (row as u32 * 17 + k * 31 + 5) % vocab_size as u32;
            while !seen.insert(idx) {
                idx = (idx + 1) % vocab_size as u32;
            }
            indices.push(idx);
            logprobs.push(-(top_k as f32).ln());
        }
    }

    // AdamW with default betas.
    let params_adamw = ParamsAdamW {
        lr: learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut optimizer = AdamW::new(vec![hidden.clone()], params_adamw)?;

    let mut first_loss: f32 = f32::NAN;
    let mut last_loss: f32 = f32::NAN;
    for step in 0..num_steps {
        let loss = opd_top_k_reverse_kl_phase_b(
            hidden.as_tensor(),
            &head_t,
            &indices,
            &logprobs,
            &label_mask,
            top_k,
            &device,
            16,
        )
        .with_context(|| format!("forward step {step}"))?;
        let lv = loss.to_scalar::<f32>()?;
        if step == 0 {
            first_loss = lv;
        }
        last_loss = lv;
        optimizer.backward_step(&loss).with_context(|| format!("backward step {step}"))?;
    }

    Ok((first_loss, last_loss))
}

/// Build an in-process "local teacher" FixtureLogitSource by running
/// the loaded model forward on each prompt and extracting top-K
/// logprobs at the active (assistant-token) positions.
///
/// This is the production §3.2 LocalTeacher path materialised as a
/// pre-compute step: instead of holding a long-lived `&GpuWeights`
/// reference inside a `LogitSource` impl (which would require
/// refactoring `GpuWeights` to `Arc<GpuWeights>` everywhere it's
/// owned), we run all teacher forwards up-front and stash the
/// answers into a `FixtureLogitSource`. The trainer then queries it
/// with the same `LogitSource` trait shape it uses for remote
/// teachers — no API drift between local and remote.
///
/// `teacher_lora`, when `Some`, is applied during the teacher forward
/// (so callers can build a teacher that's "the model with a specific
/// LoRA"). `None` means base-model teacher — Lu (2025)'s behavioural-
/// recovery recipe.
///
/// `prompt_modifier` is a per-prompt token-stream transformer for the
/// §3.12 privileged-context modes (GroundTruthConditioning, etc.).
/// Default is `identity` — student and teacher see the same tokens.
///
/// Returns a `FixtureLogitSource` already populated with entries for
/// every (prompt's tokenization, active position). Callers pass this
/// as the `teacher` argument to `opd_train`.
#[allow(clippy::too_many_arguments)]
pub fn build_local_teacher_fixture(
    teacher_id: impl Into<String>,
    prompts_and_active: &[(Vec<u32>, Vec<usize>)],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    teacher_lora: Option<&kiln_model::lora_loader::LoraWeights>,
    top_k: usize,
    tokenizer_hash: Option<String>,
) -> Result<crate::logit_source::FixtureLogitSource> {
    use kiln_model::backend;
    use kiln_model::forward::{LinearAttentionState, model_forward};

    let device = weights.embed_tokens.device().clone();
    let backend_rt = backend::for_device(&device);

    let vocab_size = model_config.vocab_size;
    if top_k > vocab_size {
        return Err(anyhow!(
            "build_local_teacher_fixture: top_k {top_k} > vocab_size {vocab_size}"
        ));
    }

    let mut fixture =
        crate::logit_source::FixtureLogitSource::uniform_topk(teacher_id, vocab_size, top_k);
    // Replace caps with full ones so the `top_k` cap mirrors actual.
    let caps_clone = fixture.capabilities();
    // We can't directly mutate fixture's caps from outside, but the
    // top_k is already set correctly by uniform_topk. Just attach the
    // tokenizer hash via a fresh insertion path.
    let _ = (caps_clone, tokenizer_hash);

    for (tokens, active_positions) in prompts_and_active {
        if active_positions.is_empty() {
            continue;
        }
        let mut linear_state = LinearAttentionState::new(model_config, &device)?;
        let logits = model_forward(
            &*backend_rt,
            tokens,
            weights,
            model_config,
            None,
            Some(&mut linear_state),
            teacher_lora,
        )
        .context("local-teacher forward pass")?;
        // logits shape: [1, T, V]. Detach autograd — no gradients needed.
        let logits = logits.detach();
        let log_probs = candle_nn::ops::log_softmax(&logits, 2)
            .context("local-teacher log_softmax")?;
        let log_probs_2d = log_probs
            .squeeze(0)
            .context("local-teacher squeeze batch dim")?
            .to_dtype(candle_core::DType::F32)
            .context("local-teacher to_dtype f32")?;
        let log_probs_host: Vec<Vec<f32>> = log_probs_2d
            .to_vec2::<f32>()
            .context("local-teacher logprobs to host")?;
        let tokens_hash = crate::logit_source::FixtureLogitSource::hash_tokens(tokens);
        for &pos in active_positions {
            if pos >= log_probs_host.len() {
                continue;
            }
            let row = &log_probs_host[pos];
            let mut indexed: Vec<(u32, f32)> = row
                .iter()
                .copied()
                .enumerate()
                .map(|(i, lp)| (i as u32, lp))
                .collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(top_k);
            let indices: Vec<u32> = indexed.iter().map(|(i, _)| *i).collect();
            let logprobs: Vec<f32> = indexed.iter().map(|(_, lp)| *lp).collect();
            fixture.insert(tokens_hash, pos, indices, logprobs);
        }
    }

    Ok(fixture)
}

/// Run OPD training on the provided prompts using the already-loaded model.
///
/// This is the §3.1 trainer body. It mirrors `trainer::sft_train`'s
/// structure (LoRA params + AdamW state + per-prompt forward/backward +
/// `save_peft`), but the per-step loss is `opd_step_loss` (reverse-KL
/// against the teacher) rather than fused linear cross-entropy.
///
/// The teacher is queried only at *active* (assistant) token positions —
/// the prompt tokens carry no loss. For the milestone-13 wire-up the
/// "rollout" is the assistant turn already present in each
/// `OpdPrompt`'s messages (the same shape as an `SftExample`); the full
/// on-policy student-sampler is a follow-up that swaps which token IDs
/// feed `model_forward_no_head` without changing the loss path.
///
/// Replay artifacts and the §8.11 receipt are *not* written by this
/// function — the caller (kiln-server's `run_opd`) wraps it with
/// `open_replay_state` / `close_replay_state` and `AdapterReceipt`.
///
/// Sample a student rollout under the active LoRA.
///
/// Uses the segmented forward path (`model_forward_segment`) so that
/// `KILN_STREAMING_PREFILL=1` actually applies — without it the
/// monolithic GDN prefill materializes F32 intermediates over the
/// full prompt length which blows past a 48 GiB GPU once the teacher
/// is also resident. The trade-off is O(N²): we re-run the prefix
/// forward at every decode step instead of carrying a KV cache.
/// For the short rollouts OPD uses (~32-256 tokens) this is the
/// memory-safe choice. KV-cached decode can be added later as a
/// fast-path when more VRAM is available.
#[allow(clippy::too_many_arguments)]
fn sample_student_rollout(
    backend: &dyn kiln_model::backend::BackendRuntime,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    lora: &kiln_model::lora::LoraWeights,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
) -> Result<Vec<u32>> {
    use kiln_model::forward::{
        LinearAttentionState, model_forward_embed, model_forward_final_norm,
        model_forward_segment,
    };
    use kiln_model::sampling::sample_step;
    use kiln_core::sampling::SamplingParams;

    anyhow::ensure!(!prompt_tokens.is_empty(), "sample_student_rollout: prompt empty");

    let device = weights.embed_tokens.device().clone();
    let head_t = weights.embed_tokens_t.clone();

    let mut params = SamplingParams::default();
    params.temperature = temperature;
    params.top_p = top_p;
    params.seed = seed;
    let mut step_seed = seed;

    // Walk the prompt + generated suffix through the streaming
    // segmented forward each step. We chunk the layers (same pattern
    // as `opd_train`'s checkpointed forward) so intermediate buffers
    // from each chunk free before the next; a single call covering
    // all layers holds the GDN intermediates live for the duration
    // and OOMs on a 48 GiB GPU once the teacher is resident.
    //
    // We chunk more aggressively here than the training path's 8
    // segments — sampling doesn't need to hold an autograd graph, so
    // every-2-layers is the safe default for memory headroom under
    // long contexts (700+ tokens with a 27B teacher resident). The
    // env override `KILN_OPD_SAMPLER_SEGMENTS` lets users dial it.
    let default_segments = std::env::var("KILN_OPD_SAMPLER_SEGMENTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(18);
    let num_segments = default_segments.min(model_config.num_layers);
    let segments = crate::trainer::compute_segment_boundaries(
        model_config.num_layers,
        num_segments,
    );
    let run_forward = |seq: &[u32]| -> Result<Tensor> {
        let positions: Vec<u32> = (0..seq.len() as u32).collect();
        let (embed_hidden, _) = model_forward_embed(seq, weights)?;
        let mut linear_state = LinearAttentionState::new_for_inference(model_config, &device)?;
        let mut current = embed_hidden;
        for &(start, end) in &segments {
            current = model_forward_segment(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(lora),
            )
            .with_context(|| format!("on-policy rollout segment [{start},{end})"))?;
            // Detach between chunks so the candle buffer cache can
            // free intermediates — bit-exact with monolithic since
            // we don't need gradients here.
            current = current.detach();
        }
        let normed = model_forward_final_norm(&current, weights, model_config)?;
        // normed: [1, T, H] — slice last position → [1, 1, H] → [H]
        let t = normed.dim(1)?;
        let last = normed.narrow(1, t - 1, 1)?.squeeze(1)?; // [1, H]
        let logits = last.matmul(&head_t)?; // [1, V]
        Ok(logits)
    };

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut working: Vec<u32> = prompt_tokens.to_vec();
    let mut next = sample_step(&run_forward(&working)?, &params, step_seed, &[])
        .context("on-policy rollout first-token sample")?;

    for _ in 0..max_new_tokens {
        if eos_token_ids.contains(&next) {
            break;
        }
        generated.push(next);
        working.push(next);
        if let Some(s) = step_seed.as_mut() {
            *s = s.wrapping_add(1);
        }
        next = sample_step(&run_forward(&working)?, &params, step_seed, &generated)
            .context("on-policy rollout next-token sample")?;
    }
    Ok(generated)
}

#[allow(clippy::too_many_arguments)]
pub fn opd_train(
    prompts: &[OpdPrompt],
    config: &OpdConfig,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher: Arc<dyn LogitSource>,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: Option<crate::trainer::ProgressCallback>,
) -> Result<std::path::PathBuf> {
    let run_started = std::time::Instant::now();
    let output_dir = adapter_dir.join(adapter_name);
    let training_data_sha256 = crate::train_receipt::sha256_json_serializable(&prompts);
    let requested_base_adapter_dir = config
        .base_adapter
        .as_deref()
        .map(|name| crate::adapter_shape::resolve_base_adapter_dir(name, adapter_dir));
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        examples_read: prompts.len(),
        ..Default::default()
    };
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();

    use crate::SftExample;
    use crate::trainer::{
        TrainableLoraParams, accumulate_grads, compute_segment_boundaries, lora_weights_detached,
        tokenize_for_training,
    };
    use crate::Optimizer;
    use kiln_model::backend;
    use kiln_model::forward::{
        LinearAttentionState, model_forward_embed, model_forward_final_norm,
        model_forward_segment,
    };
    use candle_core::Var;
    use std::collections::HashMap;

    if prompts.is_empty() {
        let message = "opd_train: prompts must be non-empty";
        write_opd_train_receipt_best_effort(
            adapter_name,
            model_config,
            tokenizer,
            config,
            &output_dir,
            requested_base_adapter_dir.as_deref(),
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            None,
            Some(message.to_string()),
        );
        anyhow::bail!("{}", crate::train_receipt::training_failure_error_message(message));
    }

    let device = weights.embed_tokens.device().clone();
    let backend_rt = backend::for_device(&device);

    // §6 data-multiplier: auto-scale samples_per_prompt when the
    // dataset is small. Lu (2025) §3.5.4: 4 if |prompts| ≥ 200,
    // 16 if 50 ≤ |prompts| < 200, 64 if |prompts| < 50. We respect
    // any non-default user override (≠ default_opd_samples_per_prompt
    // = 4) and only auto-scale when the user didn't ask for a
    // specific count.
    let effective_samples_per_prompt = if config.samples_per_prompt == default_opd_samples_per_prompt() {
        if prompts.len() < 50 {
            64
        } else if prompts.len() < 200 {
            16
        } else {
            default_opd_samples_per_prompt()
        }
    } else {
        config.samples_per_prompt
    };

    tracing::info!(
        num_prompts = prompts.len(),
        samples_per_prompt = effective_samples_per_prompt,
        config_samples_per_prompt = config.samples_per_prompt,
        top_k = config.top_k,
        loss = ?config.loss,
        lr = config.learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting OPD training"
    );

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            write_opd_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                config,
                &output_dir,
                requested_base_adapter_dir.as_deref(),
                training_data_sha256,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                None,
                Some(format!("{err:#}")),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    let effective_seed = config.seed;

    let params = TrainableLoraParams::initialize_seeded(
        model_config,
        weights,
        config.lora_rank,
        config.lora_alpha,
        &device,
        effective_seed,
    )?;
    params.register_with_backend(&*backend_rt)?;

    let mut opt_state = match config.optimizer {
        Optimizer::Sgd => None,
        Optimizer::AdamW { .. } => {
            let state = params.allocate_adamw_state(&device)?;
            state.register_with_backend(&*backend_rt)?;
            Some(state)
        }
    };

    // Tokenize every prompt up-front (cheap relative to the forward
    // pass) and skip any prompts that produce no supervised assistant
    // tokens — same shape as sft_train's validity probe.
    let mut tokenized: Vec<(Vec<u32>, Vec<bool>)> = Vec::with_capacity(prompts.len());
    for (idx, prompt) in prompts.iter().enumerate() {
        let example = SftExample {
            messages: prompt.messages.clone(),
        };
        match tokenize_for_training(&example, tokenizer) {
            Ok(pair) => tokenized.push(pair),
            Err(e) => {
                tracing::warn!(prompt_idx = idx, error = %e, "skipping OPD prompt");
            }
        }
    }
    if tokenized.is_empty() {
        data_stats.examples_filtered = prompts.len();
        let message = "opd_train: no valid prompts after tokenization";
        write_opd_train_receipt_best_effort(
            adapter_name,
            model_config,
            tokenizer,
            config,
            &output_dir,
            requested_base_adapter_dir.as_deref(),
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            Some(alpha_over_rank),
            Some(message.to_string()),
        );
        anyhow::bail!("{}", crate::train_receipt::training_failure_error_message(message));
    }
    data_stats.examples_filtered = prompts.len().saturating_sub(tokenized.len());

    let epochs = config.epochs.max(1);
    let total_steps = epochs * tokenized.len() * effective_samples_per_prompt.max(1);
    let mut global_step = 0usize;
    let mut last_loss = 0.0_f64;

    let head_t = weights.embed_tokens_t.clone();

    // §3.9 guardrail observer + per-step rollout summary buffer.
    // The guardrail watches loss / repetition / overlap signals every
    // step and produces a `GuardrailDecision`; on any non-Ok decision
    // we log the trigger via tracing so the dashboard / receipt can
    // pick it up. Programmatic in-process rollback to the last
    // passing checkpoint is the remaining §3.9 wire-up; for now the
    // detector fires and the user sees it in the run log.
    let mut guardrail = crate::diagnostics::LengthInflationGuardrail::default();
    let validation_cadence: u64 = 5;

    // Resolve EOS token ids once for rollout termination.
    let eos_token_ids: Vec<u32> = tokenizer.eos_token_ids();
    let on_policy_enabled = std::env::var("KILN_OPD_OFF_POLICY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .map(|off| !off)
        .unwrap_or(true);

    // Generation-prompt suffix that turns the prompt boundary into the
    // same context the model sees at inference under
    // `enable_thinking=False`. Without it, the student samples from a
    // `<|im_end|>\n` boundary with no assistant cue and emits EOS
    // immediately (~87% of prompts skipped in the JSON-schema-adherence
    // capability run). With the suffix the rollout starts where eval
    // actually generates, so most prompts produce real trajectories.
    //
    // We encode `<|im_start|>assistant\n<think>\n\n</think>\n\n`. If the
    // tokenizer doesn't recognise the special tokens (e.g. a model that
    // has no `<think>` markers), the encode either returns the prefix
    // without thinking tokens or errors — we fall back to empty in the
    // error case so the trainer keeps working.
    // Render the *exact* prompt the model sees at inference time under
    // `enable_thinking=false`, per-prompt, via the chat template. The
    // returned token sequence ends with the assistant cue marker
    // (`<|im_start|>assistant\n<think>\n\n</think>\n\n` for Qwen3.5) so
    // the student samples from the same boundary it would at inference.
    //
    // This replaces an earlier hack that appended a hand-encoded suffix
    // string via raw `tokenizer.encode()`, which byte-tokenized the
    // `<|im_start|>` special instead of resolving it to its single id.
    // Result of the bug: ~97% skip rate on terse-list capabilities
    // because the rollout prompt ended at `<|im_end|>\n` (the end of the
    // user turn) with no assistant cue, so the student EOS'd first thing.
    //
    // Per-prompt pre-render of the rollout prefix. Drops the last
    // assistant message (if any), passes `enable_thinking=false`, lets
    // the template emit the proper marker tokens, encodes to ids.
    use kiln_core::tokenizer::{ChatMessage as CoreChatMessage, ChatTemplateOptions};
    // OPT-IN via env var. The chat-template render path is correct (verified
    // via `examples/test_render`) but produces broken adapters end-to-end on
    // structured-list capabilities (see opd-cap.code-symbol-extraction/
    // failure_mode.md). Probably a kernel-vs-prompt-length interaction the
    // root cause isn't fully understood for. Default to legacy
    // orig_input_ids[..first_label] path until the kernel side is audited.
    let use_chat_template_render = std::env::var("KILN_OPD_USE_CHAT_TEMPLATE_RENDER")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let rollout_prompt_prefixes: Vec<Vec<u32>> = if !use_chat_template_render {
        // Empty → fallback path uses orig_input_ids[..first_label_mask_true]
        // per the legacy behavior. Same as pre-fix.
        vec![Vec::new(); prompts.len()]
    } else {
    prompts
        .iter()
        .map(|p| {
            // Drop trailing assistant message(s) if present; we want the
            // chat template to *insert* the assistant cue via
            // add_generation_prompt rather than echo the (dummy) one we
            // have in the prompt. Convert kiln-train ChatMessage to the
            // kiln-core variant the template engine expects.
            let mut msgs: Vec<CoreChatMessage> = p
                .messages
                .iter()
                .map(|m| CoreChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    ..Default::default()
                })
                .collect();
            while msgs.last().map(|m| m.role.as_str()) == Some("assistant") {
                msgs.pop();
            }
            let opts = ChatTemplateOptions {
                template_kwargs: serde_json::Map::from_iter([(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(false),
                )]),
            };
            let text = tokenizer
                .apply_chat_template_full_with_options(msgs.as_slice(), None, None, opts)
                .map_err(|e| anyhow!("apply_chat_template_full_with_options: {e}"))
                .unwrap_or_default();
            tokenizer.encode(&text).unwrap_or_default()
        })
        .collect()
    };
    if use_chat_template_render {
        let avg_prefix_len = rollout_prompt_prefixes
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
            / rollout_prompt_prefixes.len().max(1);
        tracing::info!(
            n_prompts = rollout_prompt_prefixes.len(),
            avg_prefix_tokens = avg_prefix_len,
            "opd_train: pre-rendered rollout prompts via chat template (enable_thinking=false). \
             EXPERIMENTAL — known to break LoRAs on structured-list capabilities; see failure_mode.md."
        );
    }

    // §20 teacher-side conditioning: per-prompt asymmetric teacher
    // prompt tokens (the FULL teacher render, not just a prefix).
    // Empty Vec for prompts without `teacher_extra_messages`.
    //
    // When non-empty, the teacher's query at training time uses
    // `teacher_prompt_tokens ++ sampled_rollout` instead of the
    // student's `prompt_only ++ sampled_rollout`. The student keeps a
    // clean deployment-realistic prompt; the teacher sees a richer
    // version with the privileged content merged into its first system
    // message.
    //
    // We MERGE rather than prepend because most chat templates (Qwen3.5
    // included) only allow one system message at position 0. The merge
    // concatenates the teacher_extra content into the existing system
    // message (or creates one when the student has none).
    let teacher_prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            if p.teacher_extra_messages.is_empty() {
                return Vec::new();
            }
            // Build the merged messages: drop the dummy trailing
            // assistant (the student's rollout will replace it); merge
            // each teacher_extra content into the system position.
            let mut merged: Vec<CoreChatMessage> = p
                .messages
                .iter()
                .map(|m| CoreChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    ..Default::default()
                })
                .collect();
            while merged.last().map(|m| m.role.as_str()) == Some("assistant") {
                merged.pop();
            }
            // Collect the teacher_extra content into a single string,
            // joined by blank lines.
            let extras_text = p
                .teacher_extra_messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            // Merge or create the first system message.
            if let Some(first) = merged.first_mut() {
                if first.role == "system" {
                    first.content = format!("{}\n\n{}", extras_text, first.content);
                } else {
                    merged.insert(
                        0,
                        CoreChatMessage {
                            role: "system".to_string(),
                            content: extras_text,
                            ..Default::default()
                        },
                    );
                }
            } else {
                merged.push(CoreChatMessage {
                    role: "system".to_string(),
                    content: extras_text,
                    ..Default::default()
                });
            }
            // Render with the same options the student's rollout prompt
            // would use, so the teacher and student see the SAME chat
            // template framing — only the system-message content differs.
            let opts = ChatTemplateOptions {
                template_kwargs: serde_json::Map::from_iter([(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(false),
                )]),
            };
            let text = match tokenizer.apply_chat_template_full_with_options(
                merged.as_slice(),
                None,
                None,
                opts,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "teacher render failed; falling back to symmetric for this prompt");
                    return Vec::new();
                }
            };
            tokenizer.encode(&text).unwrap_or_default()
        })
        .collect();
    let any_asymmetric = teacher_prompt_tokens.iter().any(|v| !v.is_empty());
    if any_asymmetric {
        let n_with_extra = teacher_prompt_tokens.iter().filter(|v| !v.is_empty()).count();
        let avg_extra: usize = teacher_prompt_tokens
            .iter()
            .filter(|v| !v.is_empty())
            .map(|v| v.len())
            .sum::<usize>()
            / n_with_extra.max(1);
        eprintln!(
            "opd_train: asymmetric teacher conditioning ACTIVE — {} prompts, avg teacher prompt = {} tokens. §20",
            n_with_extra, avg_extra
        );
        tracing::info!(
            n_prompts_with_teacher_extra = n_with_extra,
            avg_teacher_prompt_tokens = avg_extra,
            "opd_train: asymmetric teacher conditioning ACTIVE — \
             teacher sees richer system message than student. §20"
        );
    }

    for epoch in 0..epochs {
        for (prompt_idx, (orig_input_ids, label_mask)) in tokenized.iter().enumerate() {
            // Build the rollout prompt from the pre-rendered chat-template
            // prefix (see above — drops the dummy assistant turn and lets
            // the template emit the proper assistant cue marker tokens).
            // Falls back to the legacy `orig_input_ids[..first_label]`
            // path if rendering failed.
            let prompt_only: Vec<u32> = if !rollout_prompt_prefixes[prompt_idx].is_empty() {
                rollout_prompt_prefixes[prompt_idx].clone()
            } else {
                // Fallback path — preserves prior behaviour rather than
                // failing hard if a chat template is missing.
                let prompt_end = label_mask
                    .iter()
                    .position(|&m| m)
                    .unwrap_or(orig_input_ids.len());
                if prompt_end == 0 || prompt_end >= orig_input_ids.len() {
                    tracing::warn!(prompt_idx, "skipping prompt with no prompt/assistant split");
                    continue;
                }
                orig_input_ids[..prompt_end].to_vec()
            };
            let rollout_prompt_len = prompt_only.len();

            for sample_idx in 0..effective_samples_per_prompt.max(1) {
                // §3.1 step 1: sample a fresh student trajectory under
                // the current LoRA. Replaces the off-policy passthrough
                // of the teacher-authored assistant turn with the
                // student's own tokens — the defining property of
                // on-policy distillation per Lu (2025) §1.
                let (input_ids_owned, active_positions): (Vec<u32>, Vec<usize>) = if on_policy_enabled {
                    let lora_for_sample = params.as_lora_weights();
                    let step_seed = effective_seed.map(|s| {
                        s.wrapping_add(global_step as u64)
                            .wrapping_add(prompt_idx as u64 * 1_000_003)
                            .wrapping_add(sample_idx as u64 * 1_000_033)
                    });
                    let sampled = sample_student_rollout(
                        &*backend_rt,
                        weights,
                        model_config,
                        &lora_for_sample,
                        &prompt_only,
                        config.max_tokens,
                        &eos_token_ids,
                        config.temperature as f32,
                        config.top_p as f32,
                        step_seed,
                    )
                    .with_context(|| format!("on-policy rollout for prompt {prompt_idx}"))?;
                    if sampled.is_empty() {
                        tracing::warn!(prompt_idx, sample_idx, "student produced 0 new tokens; skipping step");
                        continue;
                    }
                    let mut full = prompt_only.clone();
                    full.extend_from_slice(&sampled);
                    // Active positions: the student-sampled tokens, which
                    // start at `rollout_prompt_len` (after the original
                    // prompt + the generation-prompt suffix).
                    let active: Vec<usize> =
                        (rollout_prompt_len..rollout_prompt_len + sampled.len()).collect();
                    (full, active)
                } else {
                    let active: Vec<usize> = label_mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &m)| if m { Some(i) } else { None })
                        .collect();
                    (orig_input_ids.clone(), active)
                };
                let input_ids: &[u32] = &input_ids_owned;
                if active_positions.is_empty() {
                    continue;
                }
                token_counts.action_tokens = token_counts
                    .action_tokens
                    .saturating_add(active_positions.len() as u64);
                token_counts.context_tokens = token_counts.context_tokens.saturating_add(
                    input_ids.len().saturating_sub(active_positions.len()) as u64,
                );
                // Gradient-checkpointed step. Runs the model forward in
                // detached segments to bound activation memory, computes the
                // OPD top-K reverse-KL loss at the final boundary, then
                // walks segments in reverse with autograd ON to accumulate
                // LoRA gradients. Mirrors the SFT path's
                // `checkpointed_forward_backward` so long-context OPD fits
                // on a single 48GB GPU.
                let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
                let lora_detached = lora_weights_detached(&params);
                let num_segments = 8usize.min(model_config.num_layers);
                let segments = compute_segment_boundaries(model_config.num_layers, num_segments);

                // === Step 1: detached forward; save segment boundaries ===
                let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
                let mut boundary_states: Vec<Tensor> =
                    Vec::with_capacity(segments.len() + 1);
                boundary_states.push(embed_hidden.detach());
                {
                    let mut current = boundary_states[0].clone();
                    let mut linear_state = LinearAttentionState::new(model_config, &device)?;
                    for &(start, end) in &segments {
                        current = model_forward_segment(
                            &*backend_rt,
                            current,
                            weights,
                            model_config,
                            &positions,
                            start,
                            end,
                            Some(&mut linear_state),
                            Some(&lora_detached),
                        )
                        .context("OPD checkpointed segmented forward")?;
                        boundary_states.push(current.detach());
                        current = boundary_states.last().unwrap().clone();
                    }
                }
                let final_hidden = boundary_states.last().unwrap().clone();

                // === Step 2: OPD loss at the final boundary ===
                // Build a Var so candle autograd routes the kernel's backward
                // into `final_var.grad()`.
                let final_var = Var::from_tensor(&final_hidden)?;
                let normed = model_forward_final_norm(
                    final_var.as_tensor(),
                    weights,
                    model_config,
                )?;
                // §20 asymmetric teacher conditioning: if this prompt
                // declared `teacher_extra_messages`, build the teacher's
                // token sequence as `teacher_prompt_tokens ++ sampled`.
                // The student's input_ids = prompt_only ++ sampled has
                // the student's own prompt framing; the teacher's view
                // swaps the prompt half for the merged-extras version
                // while keeping the SAME sampled rollout. Active
                // positions are remapped to the teacher's frame.
                let teacher_prompt: &[u32] = &teacher_prompt_tokens[prompt_idx];
                let (teacher_full_tokens_owned, teacher_shifted_positions): (Vec<u32>, Vec<usize>) =
                    if teacher_prompt.is_empty() {
                        (Vec::new(), Vec::new())
                    } else {
                        // sampled portion of student's input_ids = the
                        // tokens after the student's prompt prefix.
                        let sampled = &input_ids[rollout_prompt_len..];
                        let mut t = Vec::with_capacity(teacher_prompt.len() + sampled.len());
                        t.extend_from_slice(teacher_prompt);
                        t.extend_from_slice(sampled);
                        // Student active positions live in
                        // [rollout_prompt_len, rollout_prompt_len + sampled.len()).
                        // Map each to teacher frame:
                        // teacher_pos = teacher_prompt.len() + (student_pos - rollout_prompt_len).
                        let pos: Vec<usize> = active_positions
                            .iter()
                            .map(|p| teacher_prompt.len() + (*p - rollout_prompt_len))
                            .collect();
                        (t, pos)
                    };
                let (teacher_tokens_opt, teacher_active_opt): (Option<&[u32]>, Option<&[usize]>) =
                    if teacher_prompt.is_empty() {
                        (None, None)
                    } else {
                        (
                            Some(teacher_full_tokens_owned.as_slice()),
                            Some(teacher_shifted_positions.as_slice()),
                        )
                    };
                let out = opd_step_loss(OpdStepInputs {
                    tokens: input_ids,
                    active_positions: &active_positions,
                    student_hidden: &normed,
                    head_t: &head_t,
                    teacher: teacher.clone(),
                    loss: config.loss,
                    top_k: config.top_k,
                    chunk_size: 0,
                    teacher_tokens: teacher_tokens_opt,
                    teacher_active_positions: teacher_active_opt,
                })?;
                let loss_val = out.mean_kl.to_scalar::<f32>()? as f64;
                let active_count = out.active_count;
                let head_grads = out
                    .mean_kl
                    .backward()
                    .context("OPD loss kernel backward at final boundary")?;
                let mut upstream_grad = head_grads
                    .get(final_var.as_tensor())
                    .ok_or_else(|| anyhow!("no gradient at final_hidden Var"))?
                    .clone()
                    .detach();
                drop(head_grads);
                drop(out);
                drop(normed);
                drop(final_var);

                // === Step 3: walk segments in reverse, accumulate LoRA grads ===
                let mut accumulated_grads: HashMap<candle_core::TensorId, Tensor> =
                    HashMap::new();
                let all_vars = params.all_vars();
                for seg_idx in (0..segments.len()).rev() {
                    let (seg_start, seg_end) = segments[seg_idx];
                    let seg_input = boundary_states[seg_idx].clone();
                    let seg_input_var = Var::from_tensor(&seg_input)?;
                    let mut state = LinearAttentionState::new(model_config, &device)?;
                    let lora_for_seg = params.as_lora_weights();
                    let seg_output = model_forward_segment(
                        &*backend_rt,
                        seg_input_var.as_tensor().clone(),
                        weights,
                        model_config,
                        &positions,
                        seg_start,
                        seg_end,
                        Some(&mut state),
                        Some(&lora_for_seg),
                    )
                    .with_context(|| {
                        format!(
                            "OPD checkpointed reverse segment [{seg_start},{seg_end})"
                        )
                    })?;

                    // Inject upstream gradient: scalar = sum(seg_output * upstream)
                    let scalar = (&seg_output * &upstream_grad)?
                        .sum_all()
                        .context("OPD reverse segment gradient injection")?;
                    let seg_grads = scalar
                        .backward()
                        .with_context(|| format!("OPD reverse segment backward {seg_idx}"))?;
                    accumulate_grads(&mut accumulated_grads, &seg_grads, &all_vars)?;
                    upstream_grad = seg_grads
                        .get(seg_input_var.as_tensor())
                        .ok_or_else(|| {
                            anyhow!("no grad at seg_input_var (seg_idx={seg_idx})")
                        })?
                        .clone()
                        .detach();
                    drop(seg_grads);
                    drop(seg_output);
                }

                // Move CPU-spilled gradients back to the device for the optimizer step.
                let grads_on_device: HashMap<candle_core::TensorId, Tensor> = accumulated_grads
                    .into_iter()
                    .map(|(k, v)| {
                        let v = if v.device().same_device(&device) {
                            v
                        } else {
                            v.to_device(&device).unwrap_or(v)
                        };
                        (k, v)
                    })
                    .collect();
                crate::trainer::optimizer_step_from_map(
                    &*backend_rt,
                    &params,
                    &grads_on_device,
                    config.learning_rate,
                    config.optimizer,
                    opt_state.as_mut(),
                )?;

                last_loss = loss_val;
                global_step += 1;

                // Periodic adapter checkpoint — mirrors sft_train. Lets a
                // long OPD run survive mid-flight kills (SIGTERM, OOM,
                // wall-time exhaustion) without losing all training
                // signal. The most recent checkpoint dir is a complete
                // PEFT adapter that can be loaded as if training had
                // finished there. Disabled when `config.checkpoint_interval`
                // is None or 0.
                if let Some(interval) = config.checkpoint_interval {
                    if interval > 0
                        && global_step % interval == 0
                        && global_step < total_steps
                    {
                        let ckpt_dir = adapter_dir
                            .join(format!("{adapter_name}-checkpoint-{global_step}"));
                        if let Err(e) = params.sync_to_candle(&*backend_rt) {
                            tracing::warn!(
                                step = global_step,
                                error = %e,
                                "checkpoint: failed to sync LoRA Vars to candle"
                            );
                        }
                        match params.save_peft(&ckpt_dir, model_config.num_layers) {
                            Ok(_) => {
                                eprintln!(
                                    "opd_train: checkpoint step={}/{} dir={}",
                                    global_step,
                                    total_steps,
                                    ckpt_dir.display()
                                );
                                tracing::info!(
                                    step = global_step,
                                    total_steps,
                                    path = %ckpt_dir.display(),
                                    "OPD checkpoint saved"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    step = global_step,
                                    error = %e,
                                    "checkpoint: save_peft failed"
                                );
                            }
                        }
                    }
                }

                // §3.9 guardrail observation on the validation
                // cadence. We build a snapshot from the active-token
                // tokens-as-bytes proxy (since opd_train uses the
                // ground-truth assistant turn as the rollout for the
                // milestone wire-up — true student-sampled rollouts
                // arrive with the rollout sampler). The repetition /
                // truncation signals are best-effort against the
                // ground-truth tail; the kl / loss signals are real.
                if global_step as u64 % validation_cadence == 0 {
                    let rollout = crate::diagnostics::RolloutSummary::from_tokens(
                        input_ids, true,
                    );
                    let snapshot = crate::diagnostics::build_snapshot(
                        global_step as u64,
                        std::slice::from_ref(&rollout),
                        loss_val,
                    );
                    let decision = guardrail.observe(&snapshot);
                    if !matches!(decision, crate::diagnostics::GuardrailDecision::Ok) {
                        tracing::warn!(
                            step = global_step,
                            decision = ?decision,
                            "§3.9 guardrail fired"
                        );
                    }
                }

                if let Some(ref cb) = progress_cb {
                    cb(crate::trainer::TrainingProgress {
                        epoch: epoch + 1,
                        total_epochs: epochs,
                        step: global_step,
                        total_steps,
                        loss: loss_val,
                        progress: global_step as f32 / total_steps.max(1) as f32,
                    });
                }

                if global_step % 10 == 0 || global_step == total_steps {
                    tracing::info!(
                        prompt = prompt_idx,
                        sample = sample_idx,
                        step = global_step,
                        total_steps,
                        loss = format!("{loss_val:.6}"),
                        active = active_count,
                        "OPD step"
                    );
                }
            }
        }
    }

    // Pull on-device values back into candle CPU storage before
    // `save_peft` serializes them — mirrors sft_train's final sync.
    let _synced = params.sync_to_candle(&*backend_rt).unwrap_or(0);

    params.save_peft(&output_dir, model_config.num_layers)?;
    data_stats.examples_trained = global_step;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        steps = global_step,
        "OPD training complete"
    );

    write_opd_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        config,
        &output_dir,
        requested_base_adapter_dir.as_deref(),
        training_data_sha256,
        data_stats,
        token_counts,
        run_started.elapsed().as_millis() as u64,
        Some(alpha_over_rank),
        None,
    );

    Ok(output_dir)
}

#[allow(clippy::too_many_arguments)]
fn write_opd_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &kiln_core::config::ModelConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    config: &OpdConfig,
    output_dir: &std::path::Path,
    base_adapter_dir: Option<&std::path::Path>,
    training_data_sha256: Option<String>,
    data: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    wall_clock_ms: u64,
    alpha_over_rank: Option<f32>,
    status_error: Option<String>,
) {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "opd",
        model_config,
        tokenizer,
        crate::train_receipt::HyperparameterReceipt {
            mode: "opd".to_string(),
            rank: config.lora_rank,
            alpha: config.lora_alpha,
            alpha_over_rank,
            learning_rate: config.learning_rate,
            epochs: config.epochs.max(1),
            seed: config.seed,
        },
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.training_data = crate::train_receipt::TrainingDataReceipt {
        source: "inline_opd_prompts".to_string(),
        path: None,
        sha256: training_data_sha256,
    };
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    receipt.data = data;
    receipt.token_counts = token_counts;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_default();
        crate::train_receipt::warn_lora_delta_norms(
            "opd",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write OPD train receipt");
    }
}

#[cfg(test)]
use kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b;

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
    fn opd_step_loss_asymmetric_teacher_pairs_by_index() -> Result<()> {
        // Verify that when teacher_tokens / teacher_active_positions are
        // set, the kernel queries the teacher at the SHIFTED positions
        // and pairs them back to the student's active positions by index.
        let device = Device::Cpu;
        let vocab_size = 16usize;
        let hidden_size = 8usize;
        let top_k = 4usize;
        let student_seq_len = 10usize;
        let teacher_prefix_len = 5usize;
        let teacher_seq_len = teacher_prefix_len + student_seq_len;

        let student_hidden_vec: Vec<f32> = (0..1 * student_seq_len * hidden_size)
            .map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0)
            .collect();
        let student_hidden =
            Tensor::from_vec(student_hidden_vec, (1, student_seq_len, hidden_size), &device)?;
        let head_vec: Vec<f32> = (0..hidden_size * vocab_size)
            .map(|i| ((i * 11 + 5) % 1000) as f32 / 1000.0)
            .collect();
        let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size), &device)?;

        let student_tokens: Vec<u32> = (0..student_seq_len)
            .map(|i| ((i * 7 + 3) % vocab_size) as u32)
            .collect();
        let active_positions = vec![6, 8]; // two completion positions

        let teacher_tokens: Vec<u32> = {
            let mut v = Vec::with_capacity(teacher_seq_len);
            for i in 0..teacher_prefix_len {
                v.push(((i * 17 + 1) % vocab_size) as u32);
            }
            v.extend_from_slice(&student_tokens);
            v
        };
        let teacher_active: Vec<usize> = active_positions
            .iter()
            .map(|&p| p + teacher_prefix_len)
            .collect();

        // Build a fixture teacher keyed off the *teacher* tokens. The
        // OPD kernel must hand the fixture the teacher_tokens (not the
        // student tokens) and the SHIFTED positions for the lookup to
        // succeed — that's the asymmetric path under test.
        let mut fixture = FixtureLogitSource::uniform_topk("asym-test", vocab_size, top_k);
        let h = FixtureLogitSource::hash_tokens(&teacher_tokens);
        for &pos in &teacher_active {
            let idx: Vec<u32> = (0..top_k as u32)
                .map(|k| (pos as u32 * 5 + k * 11) % vocab_size as u32)
                .collect();
            let lp: Vec<f32> = (0..top_k)
                .map(|k| -((pos + 1) as f32).ln() - (k as f32) * 0.3)
                .collect();
            fixture.insert(h, pos, idx, lp);
        }
        let teacher: Arc<dyn LogitSource> = Arc::new(fixture);

        let out = opd_step_loss(OpdStepInputs {
            tokens: &student_tokens,
            active_positions: &active_positions,
            student_hidden: &student_hidden,
            head_t: &head_t,
            teacher: teacher.clone(),
            loss: OpdLossGranularity::TeacherTopK,
            top_k,
            chunk_size: 0,
            teacher_tokens: Some(&teacher_tokens),
            teacher_active_positions: Some(&teacher_active),
        })?;

        // Two active positions → two per-position KL values.
        let per_pos: Vec<f32> = out.per_position_kl.to_vec1()?;
        assert_eq!(per_pos.len(), 2);
        // All non-negative.
        for (i, k) in per_pos.iter().enumerate() {
            assert!(*k >= -1e-5, "per_position_kl[{i}] = {k} went negative");
        }
        Ok(())
    }

    #[test]
    fn opd_step_loss_rejects_mismatched_asymmetric_lengths() {
        let device = Device::Cpu;
        let student_hidden = Tensor::zeros((1, 4, 8), DType::F32, &device).unwrap();
        let head_t = Tensor::zeros((8, 16), DType::F32, &device).unwrap();
        let fixture = FixtureLogitSource::uniform_topk("test", 16, 4);
        let teacher: Arc<dyn LogitSource> = Arc::new(fixture);
        // teacher_active_positions has 1 entry but active_positions has 2.
        let err = opd_step_loss(OpdStepInputs {
            tokens: &[1, 2, 3, 4],
            active_positions: &[1, 2],
            student_hidden: &student_hidden,
            head_t: &head_t,
            teacher,
            loss: OpdLossGranularity::TeacherTopK,
            top_k: 4,
            chunk_size: 0,
            teacher_tokens: Some(&[1, 2, 3, 4, 5, 6]),
            teacher_active_positions: Some(&[3]),
        })
        .unwrap_err();
        assert!(err.to_string().contains("teacher_active_positions"));
    }

    #[test]
    fn opd_request_round_trips_through_serde() {
        let req = OpdRequest {
            prompts: vec![OpdPrompt {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: "Evaluate ∫_0^∞ e^{-x^2} dx".into(),
                }],
                teacher_extra_messages: vec![],
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
        assert_eq!(cfg.checkpoint_interval, Some(25));
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

    /// §3.1 end-to-end algorithmic-loop validation — the most
    /// load-bearing test in the whole OPD stack. Proves that the
    /// per-token reverse-KL kernel + autograd + AdamW together
    /// produce a real training signal that drives the loss down.
    ///
    /// Without this passing, every other §3.1 / §31 milestone is
    /// just plumbing.
    #[test]
    fn opd_synthetic_training_loop_actually_trains() -> Result<()> {
        // Small enough to converge fast on CPU; representative of
        // the gather + matmul + softmax dynamics.
        let (first, last) = opd_train_synthetic_validation(
            /* seq_len = */ 16,
            /* hidden_size = */ 8,
            /* vocab_size = */ 64,
            /* top_k = */ 8,
            /* num_steps = */ 50,
            /* learning_rate = */ 0.05,
        )?;
        assert!(
            last < first,
            "loss did not decrease: first={first:.6} last={last:.6}"
        );
        // Strong-form: loss drops by at least 30% over 50 steps
        // when the teacher is uniform-over-K. With a high enough
        // learning rate the student's K-support softmax should
        // approach the uniform target.
        assert!(
            last < first * 0.7,
            "loss decreased but not enough: first={first:.6} last={last:.6}"
        );
        Ok(())
    }

    #[test]
    fn agentic_weights_prose_only_default_to_one() {
        let inputs = AgenticLossInputs {
            token_classes: vec![TipTokenClass::Prose; 4],
            ..Default::default()
        };
        let w = compute_agentic_loss_weights(&inputs);
        assert_eq!(w, vec![1.0; 4]);
    }

    #[test]
    fn agentic_weights_tool_call_upweighted_per_class() {
        let inputs = AgenticLossInputs {
            token_classes: vec![
                TipTokenClass::Prose,
                TipTokenClass::ToolCallName,
                TipTokenClass::ToolCallParams,
                TipTokenClass::ToolResult,
            ],
            ..Default::default()
        };
        let w = compute_agentic_loss_weights(&inputs);
        assert_eq!(w[0], 1.0); // prose
        assert_eq!(w[1], 3.0); // tool name (default 3.0)
        assert_eq!(w[2], 2.0); // tool params (default 2.0)
        assert_eq!(w[3], 0.0); // tool result — masked
    }

    #[test]
    fn agentic_weights_score_schedule_decays_to_one() {
        // 8 prose positions, earliest divergence at position 0,
        // decay steps = 4. Position 0 weight = 3.0; position 4
        // back to ~1.0; later positions still at 1.0.
        let inputs = AgenticLossInputs {
            token_classes: vec![TipTokenClass::Prose; 8],
            earliest_divergence: Some(0),
            score_earliest_weight: 3.0,
            score_decay_steps: 4,
            ..Default::default()
        };
        let w = compute_agentic_loss_weights(&inputs);
        assert!((w[0] - 3.0).abs() < 1e-9);
        // Linear decay 3.0 → 1.0 over 4 steps.
        // dist=2 -> frac=0.5 -> 3 + (1-3)*0.5 = 2.0
        assert!((w[2] - 2.0).abs() < 1e-9);
        assert!((w[4] - 1.0).abs() < 1e-9);
        assert!((w[7] - 1.0).abs() < 1e-9, "beyond decay window stays at 1");
    }

    #[test]
    fn agentic_weights_score_and_tip_compose_multiplicatively() {
        // §10.4: earliest-divergence position on a tool-call-name
        // token gets BOTH SCoRe boost AND TIP boost.
        let inputs = AgenticLossInputs {
            token_classes: vec![
                TipTokenClass::Prose,
                TipTokenClass::ToolCallName,
                TipTokenClass::Prose,
            ],
            earliest_divergence: Some(1),
            score_earliest_weight: 3.0,
            score_decay_steps: 4,
            tip_tool_name_weight: 3.0,
            ..Default::default()
        };
        let w = compute_agentic_loss_weights(&inputs);
        // pos 1: prose=1.0 -> ToolCallName 3.0 -> SCoRe earliest 3.0 = 9.0.
        assert!((w[1] - 9.0).abs() < 1e-9, "got {}", w[1]);
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
            if_eval_suite: None,
            new_knowledge_eval_suite: None,
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

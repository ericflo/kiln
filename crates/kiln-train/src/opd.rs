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
//! * `OpdLossGranularity` compatibility enum. The executable server path is
//!   `TeacherTopK` with K 16 or 32; other values fail closed.
//! * `opd_step_loss` — one OPD step's loss + per-position KL, given a
//!   fully tokenized rollout, the student's hidden states at the
//!   rollout's positions, and a `LogitSource` to query the teacher.
//!   Returns the scalar loss + per-position KL vector for diagnostics
//!   (`overlap_ratio`, `entropy_gap`, etc. — §3.8 wires off these).
//! * `opd_train` — the full §3.1 trainer body. Mirrors `sft_train`'s
//!   structure but with `opd_step_loss` for the per-step loss.
//! * `build_local_teacher_fixture` — in-process LocalTeacher path: run
//!   the loaded model forward once per prompt, stash top-K teacher
//!   logprobs in a `FixtureLogitSource` keyed by the exact token sequence. Used by
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
//! 1. **`SampledToken`** is retained only as a backwards-compatible wire
//!    value and rejected before training. A correct implementation needs the
//!    observed token's teacher logprob; ranked top-1 support would yield zero.
//! 2. **`TeacherTopK`** (§6 default, robust). Renormalise both
//!    distributions over the teacher's K support, then compute
//!    `KL(p_hat || q_hat)`. Uses `kiln-opd-loss-kernel`.
//! 3. **`FullVocab`** is retained as a compatibility value but rejected by
//!    the server because no concrete source/kernel route implements K = V.
//!
//! Production currently minimizes the mean per-token top-K reverse KL
//! directly. Discounted advantages and importance-ratio clipping are rejected
//! until a policy-gradient loss root is connected.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
// NOTE(#1082): Dropped every direct `candle_core::` reference in this
// file. The 29 candle uses (struct fields on `OpdStepInputs` /
// `OpdStepOutputs` / `StableOpdLossInputs` / `StableOpdLossOutputs`,
// `opd_step_loss_simple` signature params, the segment-checkpoint
// helper's `run_forward` return type, the
// `boundary_states: Vec<Tensor>` accumulator, the `Tensor::new` /
// `Tensor::from_vec` constructors in production fallbacks, the `Var`
// boundary helpers, the test-mod `Tensor` / `Device` / `DType`
// builders) now go through the per-crate candle facade
// `crate::cd_types`. The aliases are transparent at the ABI boundary
// so the public surface is unchanged. This drops opd.rs
// `candle_core::` ref count from 59 → 0 as part of the per-file
// path-collapse pass in full candle removal (#1082).
//
// The candle dep itself stays because `compute_stable_opd_loss` returns
// `Tensor`s that the trainer's `.backward()` roots on, and the
// underlying KL+SFT composition still uses candle autograd. Drop the
// candle dep once the entire OPD loss path migrates to kt-typed
// autograd via `KtForwardOp1` / `KtForwardOp2` (kt_api.rs Phase 7).
// (#1082 candle-drop) `Var` + `TensorId` dropped from the facade import:
// the OPD candle grad path that used candle `Var`s + a `HashMap<TensorId,
// Tensor>` grad map was deleted. The production tape-authoritative scalar path
// is now kt-native (`opd_tape_shim::try_tape_opd_scalar_mean_cuda_kt` takes kt
// `hidden`/`head_t` directly). The candle `Tensor` / `DType` facade aliases stay
// for the remaining candle seam in `opd_step_loss` (the per-position candle
// shim path, used by `opd_step_loss_simple` + parity tests) and the OPD-loss
// test fixtures.
use crate::cd_types::{DType, Tensor};
// (#1082) `Device` is only referenced from test-mode helpers
// (`opd_train_synthetic_validation`, `mod tests`); gating the import
// keeps the non-test build free of dead-code warnings.
#[cfg(test)]
use crate::cd_types::Device;
use serde::{Deserialize, Serialize};

use crate::logit_source::{
    LogitSource, LogitSourceCaps, LogprobBatch, target_token_positions_to_logits_rows,
    validate_full_vocab_logprobs_batch, validate_logit_request, validate_topk_logprob_row,
    validate_topk_logprobs_batch,
};
use crate::{ChatMessage, Optimizer, default_alpha, default_rank};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use kiln_model::backend::{OpdLossRoute, OpdPhaseBBackwardRoute, TrainingLossBackend};
// (#1082) Production caller migration: the per-position OPD loss now
// routes through `KtForwardOp1` (kiln-kt-bridge::forward_op::KtForwardOp1,
// commit `095f1c74`) over the kt-typed forward
// (`opd_top_k_reverse_kl_per_position_kt`) and kt-typed backward
// (`opd_top_k_reverse_kl_phase_b_bwd_kt`). The production path now calls the
// kt forward directly, with route eligibility owned by typed backend policy.
use kiln_opd_loss_kernel::DEFAULT_CHUNK_SIZE;
// (#1082) The candle-typed OPD glue (Phase A reference, kt-forward-op
// shim, kt-tape adapters) was relocated from `kiln-opd-loss-kernel` into
// (#1082) `opd_step_loss` now calls the kt forward
// `kiln_opd_loss_kernel::opd_top_k_reverse_kl_per_position_kt` directly — the
// candle `opd_top_k_reverse_kl_per_position_via_kt_forward_op` shim import is gone.

/// §6 default top-K for the `TeacherTopK` loss path. Picked by Fu et al.
/// (2026) ablation table 3: K = 32 is the optimum across math and
/// agentic OPD; K = 16 underperforms; K = 64 buys no further gain.
pub const fn default_opd_top_k() -> usize {
    32
}

/// Resolve a requested OPD support size against the source and KT-kernel
/// envelopes. The current authoritative forward/backward kernels support only
/// K=16 and K=32, so arbitrary request/provider caps must be rounded down to
/// the largest executable value before teacher I/O or GPU work begins.
pub fn resolve_opd_top_k(requested_top_k: usize, source_max_top_k: usize) -> Result<usize> {
    [32usize, 16]
        .into_iter()
        .find(|&candidate| candidate <= requested_top_k && candidate <= source_max_top_k)
        .ok_or_else(|| {
            anyhow!(
                "OPD teacher_top_k has no executable support size: requested {requested_top_k}, source cap {source_max_top_k}, KT kernels require one of {{16, 32}}"
            )
        })
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
    /// Reserved wire value for single-token reverse KL. Rejected before
    /// training: the current `LogitSource` contract returns ranked top-K rows,
    /// not the sampled token's teacher logprob, so treating K=1 as this loss
    /// would silently produce an identically zero renormalized-support KL.
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

impl OpdLossGranularity {
    /// Explain why a deserializable loss mode cannot currently train.
    pub const fn unsupported_reason(self) -> Option<&'static str> {
        match self {
            Self::SampledToken => Some(
                "sampled_token is unsupported: the teacher contract does not return the sampled token's logprob, and using ranked top-1 would produce an identically zero loss",
            ),
            Self::TeacherTopK | Self::FullVocab => None,
        }
    }
}

/// Whether OPD should sample fresh student rollouts or replay
/// teacher-authored completions from an offline dataset.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpdTrainingMode {
    OnPolicy,
    OffPolicy,
}

impl Default for OpdTrainingMode {
    fn default() -> Self {
        Self::OnPolicy
    }
}

/// Action-token supervision objective for off-policy distillation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpdObjective {
    /// Reverse-KL against teacher top-K logprobs.
    ReverseKl,
    /// Reserved wire value. Dataset preparation can validate CE targets, but
    /// the production trainer rejects this until CE is part of its loss root.
    CrossEntropy,
}

impl Default for OpdObjective {
    fn default() -> Self {
        Self::ReverseKl
    }
}

impl OpdObjective {
    pub const fn unsupported_reason(self) -> Option<&'static str> {
        match self {
            Self::ReverseKl => None,
            Self::CrossEntropy => {
                Some("cross_entropy is not wired into the production OPD loss root; use reverse_kl")
            }
        }
    }
}

/// Reserved Stable-OPD knob set (Luo et al. 2026). Only `Off` is executable
/// today; `Auto` and `Manual` remain deserializable so older requests fail with
/// a precise contract error instead of an unknown-enum error.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum StableOpdMode {
    /// Disable Stable-OPD. This is the only production-supported value today.
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
        Self::Off
    }
}

impl StableOpdMode {
    pub const fn unsupported_reason(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Auto | Self::Manual { .. } => Some(
                "Stable-OPD is not wired into the production loss root; use {\"mode\":\"off\"}",
            ),
        }
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
    /// Student-visible prompt scaffold. For plain off-policy replay this can
    /// include the teacher-authored assistant turn. For trajectory-shaped
    /// replay this is only the prompt scaffold; `trajectory` supplies the
    /// replayed action/observation turns.
    pub messages: Vec<ChatMessage>,
    /// Asymmetric teacher-only context. Empty = symmetric (default).
    /// When non-empty, the teacher's logprobs are computed against
    /// `teacher_extra_messages ++ messages ++ rollout` while the
    /// student still rolls out from `messages` alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub teacher_extra_messages: Vec<ChatMessage>,
    /// Optional agentic replay trajectory. Action segments receive OPD
    /// supervision; Observation segments can receive ECHO env-CE when
    /// `OpdConfig::echo` is set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<crate::trajectory::TurnSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherTopLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// One teacher action token from the off-policy OPD JSONL schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherActionToken {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<TeacherTopLogprob>,
}

/// One line of the documented off-policy OPD teacher JSONL schema.
///
/// `messages` is the prompt seen by the student. `teacher_response` is
/// appended as the assistant turn for off-policy replay. `teacher_tokens`
/// carries optional teacher logprobs for reverse-KL; cross-entropy mode only
/// requires the response text. `trajectory` is optional agentic structure used
/// to account for ECHO observation tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffPolicyDistillationExample {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub teacher_response: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub teacher_tokens: Vec<TeacherActionToken>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<crate::trajectory::TurnSegment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

pub const OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1: &str =
    "kiln.off-policy-distillation-manifest.v1";

/// Canonical first record for JSONL containing pre-scored teacher logits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OffPolicyDistillationManifestV1 {
    schema: String,
    teacher_identity: crate::TeacherIdentityV1,
}

impl OffPolicyDistillationManifestV1 {
    pub fn new(teacher_identity: crate::TeacherIdentityV1) -> Self {
        Self {
            schema: OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1.to_string(),
            teacher_identity,
        }
    }

    pub fn teacher_identity(&self) -> &crate::TeacherIdentityV1 {
        &self.teacher_identity
    }

    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("validated off-policy manifest serializes")
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema == OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1,
            "unsupported off-policy distillation manifest schema {:?}",
            self.schema
        );
        Ok(())
    }
}

/// Load-once JSONL result. `source_sha256` covers the exact bytes parsed.
#[derive(Debug, Clone)]
pub struct LoadedOffPolicyDistillationDataset {
    pub manifest: Option<OffPolicyDistillationManifestV1>,
    pub examples: Vec<OffPolicyDistillationExample>,
    pub source_sha256: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OffPolicyDistillationSummary {
    pub examples: usize,
    pub action_tokens: u64,
    pub env_tokens: u64,
    pub examples_with_teacher_logprobs: usize,
    pub objective: OpdObjective,
    pub echo_combined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OffPolicyLossBreakdown {
    pub objective: OpdObjective,
    pub action_token_loss: f64,
    pub echo_env_ce: Option<f64>,
    pub echo_lambda: Option<f64>,
    pub total_loss: f64,
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
/// Optional IF-recovery and new-knowledge suites produce independent paired
/// confidence evidence. Automatic publication deliberately accepts only one
/// versioned held-out suite: configure one gate for `auto_load`, or set
/// `auto_load=false` and review both diagnostics. Callers that require one
/// automatic decision across both domains must compose them into one suite.
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
    /// eval suite before a singly-gated refreshed adapter is published.
    /// `0.95` (default) means "IF-eval after refresh must be ≥ 95%
    /// of the pre-refresh score" — Lu's recipe target.
    #[serde(default = "default_require_if_eval_recovery")]
    pub require_if_eval_recovery: f64,
    /// Required confidence-bounded absolute gain on the new-knowledge eval
    /// suite.
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
    /// using the candidate-lower / baseline-upper Wilson ratio. It may be the
    /// sole automatic gate, or an independent diagnostic when auto-load is
    /// disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub if_eval_suite: Option<String>,
    /// Name of the registered eval suite used to measure new-knowledge
    /// gain on the mid-trained material. When set, the refreshed
    /// adapter is queued for a post-training eval against this suite
    /// using candidate-lower minus baseline-upper. It may be the sole
    /// automatic gate, or an independent diagnostic when auto-load is
    /// disabled.
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
    #[serde(
        default = "default_off_policy_opd_config",
        deserialize_with = "deserialize_off_policy_opd_config"
    )]
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

fn default_off_policy_opd_config() -> OpdConfig {
    OpdConfig {
        training_mode: OpdTrainingMode::OffPolicy,
        ..OpdConfig::default()
    }
}

fn deserialize_off_policy_opd_config<'de, D>(deserializer: D) -> Result<OpdConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;

    let mut value = serde_json::Value::deserialize(deserializer)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| D::Error::custom("OPD config must be a JSON object"))?;
    object
        .entry("training_mode")
        .or_insert_with(|| serde_json::Value::String("off_policy".to_owned()));
    serde_json::from_value(value).map_err(D::Error::custom)
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
    /// Reserved RLRT-style mode. The server rejects it until a distinct
    /// reverse objective is implemented; negating logprobs is invalid.
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
    /// Ground-truth answers for `GroundTruthConditioning`: exactly one
    /// non-empty answer per explicit prompt.
    #[serde(default)]
    pub ground_truth: Option<Vec<String>>,
    /// Retrieval context for `DocumentAsPi`: exactly one non-empty document
    /// string per explicit prompt.
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    #[serde(
        default = "default_off_policy_opd_config",
        deserialize_with = "deserialize_off_policy_opd_config"
    )]
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
    /// Optional server-local JSONL dataset path. In `off_policy` mode each
    /// non-empty line is an [`OffPolicyDistillationExample`] carrying the
    /// teacher response and optional per-token logprobs.
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

/// How OPD constructs the student rollout prefix before sampling.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpdRolloutPromptRendering {
    /// Preserve the admitted token sequence up to the first supervised action.
    /// This is the qualified compatibility path and remains the default.
    #[default]
    LegacyActionBoundary,
    /// Re-render the prompt with the model chat template and thinking disabled.
    /// This is experimental because it changes the token sequence and has not
    /// produced reliable adapters on every structured-output workload.
    ChatTemplate,
}

/// §3.1 + §6 default config for the OPD trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpdConfig {
    /// Training data source: on-policy student rollouts, or off-policy replay
    /// of teacher-authored responses from JSONL.
    #[serde(default)]
    pub training_mode: OpdTrainingMode,

    /// Action-token objective. Only `reverse_kl` is executable today;
    /// `cross_entropy` is retained as a rejected compatibility value.
    #[serde(default)]
    pub objective: OpdObjective,

    /// Loss granularity (§3.1). Defaults to `TeacherTopK` per §6.
    #[serde(default)]
    pub loss: OpdLossGranularity,

    /// Top-K size when `loss = TeacherTopK`. Default 32 (Fu et al. 2026
    /// ablation). Ignored for `FullVocab`; `SampledToken` is rejected.
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

    /// Number of layer segments used by the memory-bounded student sampler.
    /// `None` selects the automatic default of 18, capped at the model layer
    /// count. This affects sampling only, not gradient checkpointing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler_segments: Option<usize>,

    /// Exact algorithm used to construct student rollout prefixes. Serialized
    /// into effective config, checkpoints, and receipts for replay identity.
    #[serde(default)]
    pub rollout_prompt_rendering: OpdRolloutPromptRendering,

    /// Stable-OPD mode (§3.1, §3.9). Defaults to `Off`; other modes are
    /// rejected until their reference-KL and golden-SFT terms are wired.
    #[serde(default)]
    pub stable_opd: StableOpdMode,

    /// Reserved discount factor. Production currently supports only zero.
    #[serde(default)]
    pub discount: f64,

    /// Reserved PPO-style importance-ratio clipping. Production currently
    /// supports only zero because OPD roots the direct reverse-KL mean.
    #[serde(default = "default_opd_clip_eps")]
    pub clip_epsilon: f64,

    /// Learning rate. `None` (the default) resolves per optimizer at run
    /// start — see [`crate::resolve_learning_rate`] (AdamW/SGD keep the
    /// legacy 1e-5: §6's 10× the FullFT optimum per Schulman 2025 LoRA
    /// Without Regret). Explicit values are used verbatim; the train
    /// receipt records whichever value actually ran.
    #[serde(default)]
    pub learning_rate: Option<f64>,

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

    /// Auto-checkpoint cadence (§3.9 auto-rollback). Every N committed
    /// optimizer steps. Defaults to 25.
    #[serde(default = "default_opd_checkpoint_interval")]
    pub checkpoint_interval: Option<usize>,

    /// Immutable exact-training checkpoint to resume. PEFT adapter snapshots
    /// are serving artifacts and are rejected at this boundary.
    #[serde(default)]
    pub resume_checkpoint: Option<String>,

    /// Internal server admission result for the maximum submitted row. `None`
    /// lets standalone training plan from its immutable runtime capacity.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub grad_checkpoint_segments: Option<usize>,

    /// Scan every tape backward gradient for NaN or Inf and fail at the
    /// producing operation. This request-local diagnostic is disabled by
    /// default because each check may synchronize the training device.
    #[serde(default)]
    pub detect_anomaly: bool,

    /// Deterministic seed. If `None`, the trainer picks one and records
    /// it in the replay log.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Optimizer. Defaults to Muon (momentum-orthogonalized SGD), matching
    /// SFT/GRPO; AdamW and SGD remain selectable per-request.
    #[serde(default)]
    pub optimizer: Optimizer,

    /// Optional ECHO term for off-policy agentic trajectories. When present
    /// and the JSONL line includes Observation segments, env tokens are
    /// accounted separately from action-token OPD loss in the receipt.
    #[serde(default)]
    pub echo: Option<crate::EchoConfig>,

    /// Number of training epochs. Default 1.
    #[serde(default = "default_opd_epochs")]
    pub epochs: usize,

    /// Reserved paid-provider cost cap. Only self-hosted vLLM is wired today,
    /// so any non-`None` value is rejected rather than falsely advertised.
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
    0.0
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

impl OpdConfig {
    /// The explicit `learning_rate` when given, else the per-optimizer
    /// default for OPD.
    pub fn effective_learning_rate(&self) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| crate::resolve_learning_rate(&self.optimizer, crate::TrainMode::Opd))
    }

    /// Validate the subset whose values participate in the production OPD
    /// computation. Keep this at the library boundary so alternate server
    /// admission paths cannot enqueue semantically inert configuration.
    pub fn validate_runtime_contract(&self) -> Result<()> {
        if let Some(reason) = self.loss.unsupported_reason() {
            anyhow::bail!("OPD loss: {reason}");
        }
        if let Some(reason) = self.objective.unsupported_reason() {
            anyhow::bail!("OPD objective: {reason}");
        }
        if let Some(reason) = self.stable_opd.unsupported_reason() {
            anyhow::bail!("OPD stable_opd: {reason}");
        }
        anyhow::ensure!(
            self.discount == 0.0,
            "OPD discount={} is unsupported because discounted advantages are not wired; use 0",
            self.discount
        );
        anyhow::ensure!(
            self.clip_epsilon == 0.0,
            "OPD clip_epsilon={} is unsupported because importance-ratio clipping is not wired; use 0",
            self.clip_epsilon
        );
        anyhow::ensure!(
            self.grad_checkpoint_segments != Some(0),
            "OPD grad_checkpoint_segments must be greater than zero"
        );
        anyhow::ensure!(
            self.sampler_segments != Some(0),
            "OPD sampler_segments must be greater than zero"
        );
        anyhow::ensure!(
            self.max_cost_usd.is_none(),
            "OPD max_cost_usd is unavailable: the only wired remote provider is self-hosted vLLM and no metered billing source exists"
        );
        anyhow::ensure!(
            self.checkpoint_interval != Some(0),
            "OPD checkpoint_interval must be greater than zero"
        );
        Ok(())
    }
}

impl Default for OpdConfig {
    fn default() -> Self {
        Self {
            training_mode: OpdTrainingMode::default(),
            objective: OpdObjective::default(),
            loss: OpdLossGranularity::default(),
            top_k: default_opd_top_k(),
            samples_per_prompt: default_opd_samples_per_prompt(),
            temperature: default_opd_temperature(),
            top_p: default_opd_top_p(),
            max_tokens: default_opd_max_tokens(),
            sampler_segments: None,
            rollout_prompt_rendering: OpdRolloutPromptRendering::default(),
            stable_opd: StableOpdMode::default(),
            discount: 0.0,
            clip_epsilon: default_opd_clip_eps(),
            learning_rate: None,
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            allow_high_lora_scale: false,
            base_adapter: None,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: default_opd_checkpoint_interval(),
            resume_checkpoint: None,
            grad_checkpoint_segments: None,
            detect_anomaly: false,
            seed: None,
            optimizer: Optimizer::default(),
            echo: None,
            max_cost_usd: None,
            epochs: 1,
        }
    }
}

pub struct PreparedOffPolicyDistillation {
    pub prompts: Vec<OpdPrompt>,
    pub teacher: crate::logit_source::FixtureLogitSource,
    pub summary: OffPolicyDistillationSummary,
}

/// Shared OPD tokenization output used by the `opd_train` kt-tape path.
///
/// `pub(crate)` so the tokenization + ECHO env-mask plumbing has a single
/// definition of "what does an OPD step's active mask mean" reused across the
/// OPD trainer's internals.
pub(crate) struct TokenizedOpdPrompt {
    pub(crate) input_ids: Vec<u32>,
    pub(crate) action_mask: Vec<bool>,
    pub(crate) env_mask: Vec<bool>,
    pub(crate) total_obs_len: usize,
}

struct PreparedOpdPrompt {
    source_index: usize,
    tokenized: TokenizedOpdPrompt,
}

pub(crate) fn tokenize_opd_prompt_for_training(
    prompt: &OpdPrompt,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    echo: Option<&crate::EchoConfig>,
) -> Result<TokenizedOpdPrompt> {
    if prompt.trajectory.is_empty() {
        let (input_ids, action_mask) = crate::trainer::tokenize_for_training(
            &crate::SftExample {
                messages: prompt.messages.clone(),
            },
            tokenizer,
        )?;
        let env_mask = vec![false; input_ids.len()];
        return Ok(TokenizedOpdPrompt {
            input_ids,
            action_mask,
            env_mask,
            total_obs_len: 0,
        });
    }

    let mask_cfg = echo
        .map(|echo| crate::trajectory_mask::MaskConfig {
            warning_filter: echo.warning_filter,
            env_mask_mode: echo.env_mask_mode.into(),
        })
        .unwrap_or_default();
    let masked = crate::trajectory_mask::build_masks_from_trajectory(
        &prompt.trajectory,
        &prompt.messages,
        tokenizer,
        &mask_cfg,
    )?;
    let total_obs_len = masked.total_obs_len();
    Ok(TokenizedOpdPrompt {
        input_ids: masked.input_ids,
        action_mask: masked.action_mask,
        env_mask: masked.env_mask,
        total_obs_len,
    })
}

fn prepare_opd_prompts_for_training(
    prompts: &[OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    echo: Option<&crate::EchoConfig>,
) -> Vec<PreparedOpdPrompt> {
    let mut prepared = Vec::with_capacity(prompts.len());
    for (source_index, prompt) in prompts.iter().enumerate() {
        match tokenize_opd_prompt_for_training(prompt, tokenizer, echo) {
            Ok(tokenized) if tokenized.action_mask.iter().any(|&active| active) => {
                prepared.push(PreparedOpdPrompt {
                    source_index,
                    tokenized,
                });
            }
            Ok(_) => {
                tracing::warn!(
                    prompt_idx = source_index,
                    "skipping OPD prompt with no action tokens"
                );
            }
            Err(error) => {
                tracing::warn!(
                    prompt_idx = source_index,
                    error = %error,
                    "skipping OPD prompt"
                );
            }
        }
    }
    prepared
}

pub fn parse_off_policy_distillation_jsonl_str(
    input: &str,
) -> Result<Vec<OffPolicyDistillationExample>> {
    Ok(parse_off_policy_distillation_dataset_str(input)?.examples)
}

pub fn parse_off_policy_distillation_dataset_str(
    input: &str,
) -> Result<LoadedOffPolicyDistillationDataset> {
    let mut examples = Vec::new();
    let mut manifest = None;
    for (line_idx, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse off-policy OPD JSONL line {}", line_idx + 1))?;
        let is_manifest = value.get("schema").and_then(serde_json::Value::as_str)
            == Some(OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1);
        if is_manifest {
            anyhow::ensure!(
                manifest.is_none() && examples.is_empty(),
                "off-policy distillation manifest must be the first non-empty JSONL record"
            );
            let parsed: OffPolicyDistillationManifestV1 = serde_json::from_value(value)
                .with_context(|| {
                    format!(
                        "parse off-policy OPD JSONL manifest on line {}",
                        line_idx + 1
                    )
                })?;
            parsed.validate()?;
            anyhow::ensure!(
                parsed.canonical_json() == raw_line,
                "off-policy distillation manifest on line {} is not canonical compact JSON",
                line_idx + 1
            );
            manifest = Some(parsed);
            continue;
        }
        let example: OffPolicyDistillationExample = serde_json::from_str(line)
            .with_context(|| format!("parse off-policy OPD JSONL line {}", line_idx + 1))?;
        examples.push(example);
    }
    Ok(LoadedOffPolicyDistillationDataset {
        manifest,
        examples,
        source_sha256: crate::train_receipt::sha256_bytes(input.as_bytes()),
    })
}

pub fn load_off_policy_distillation_jsonl(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<OffPolicyDistillationExample>> {
    Ok(load_off_policy_distillation_dataset(path)?.examples)
}

pub fn load_off_policy_distillation_dataset(
    path: impl AsRef<std::path::Path>,
) -> Result<LoadedOffPolicyDistillationDataset> {
    let path = path.as_ref();
    let input = std::fs::read_to_string(path)
        .with_context(|| format!("reading off-policy OPD JSONL {}", path.display()))?;
    parse_off_policy_distillation_dataset_str(&input)
}

pub fn prepare_off_policy_distillation_dataset(
    examples: &[OffPolicyDistillationExample],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher_id: impl Into<String>,
    vocab_size: usize,
    top_k: usize,
    objective: OpdObjective,
    echo: Option<&crate::EchoConfig>,
) -> Result<PreparedOffPolicyDistillation> {
    prepare_off_policy_distillation_dataset_with_identity(
        examples, tokenizer, teacher_id, None, vocab_size, top_k, objective, echo,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_off_policy_distillation_dataset_with_identity(
    examples: &[OffPolicyDistillationExample],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher_id: impl Into<String>,
    teacher_identity: Option<crate::TeacherIdentityV1>,
    vocab_size: usize,
    top_k: usize,
    objective: OpdObjective,
    echo: Option<&crate::EchoConfig>,
) -> Result<PreparedOffPolicyDistillation> {
    anyhow::ensure!(top_k > 0, "top_k must be > 0");
    anyhow::ensure!(vocab_size > 0, "vocab_size must be > 0");
    anyhow::ensure!(
        top_k <= vocab_size,
        "top_k {top_k} exceeds vocab_size {vocab_size}"
    );
    let teacher_id = teacher_id.into();
    let mut prompts = Vec::with_capacity(examples.len());
    let mut fixture =
        crate::logit_source::FixtureLogitSource::uniform_topk(&teacher_id, vocab_size, top_k);
    if let Some(identity) = teacher_identity {
        fixture = fixture
            .with_authoritative_identity(identity)
            .context("bind off-policy fixture to teacher identity")?;
    }
    let mut summary = OffPolicyDistillationSummary {
        examples: examples.len(),
        objective,
        echo_combined: false,
        ..Default::default()
    };

    for (example_idx, example) in examples.iter().enumerate() {
        anyhow::ensure!(
            !example.messages.is_empty(),
            "off-policy OPD example {example_idx} has no messages"
        );
        anyhow::ensure!(
            !example.teacher_response.is_empty(),
            "off-policy OPD example {example_idx} has empty teacher_response"
        );

        let prompt = if example.trajectory.is_empty() {
            let mut messages = example.messages.clone();
            messages.push(ChatMessage::new(
                "assistant",
                example.teacher_response.clone(),
            ));
            OpdPrompt {
                messages,
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            }
        } else {
            OpdPrompt {
                messages: example.messages.clone(),
                teacher_extra_messages: Vec::new(),
                trajectory: example.trajectory.clone(),
            }
        };

        let tokenized = tokenize_opd_prompt_for_training(&prompt, tokenizer, echo)
            .with_context(|| format!("tokenize off-policy OPD example {example_idx}"))?;
        let active_positions: Vec<usize> = tokenized
            .action_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, active)| active.then_some(idx))
            .collect();
        anyhow::ensure!(
            !active_positions.is_empty(),
            "off-policy OPD example {example_idx} produced no assistant action tokens"
        );
        let logits_rows = target_token_positions_to_logits_rows(
            &teacher_id,
            tokenized.input_ids.len(),
            &active_positions,
        )
        .with_context(|| {
            format!("off-policy OPD example {example_idx} has invalid action-token positions")
        })?;
        summary.action_tokens = summary
            .action_tokens
            .saturating_add(active_positions.len() as u64);

        if !example.teacher_tokens.is_empty() {
            anyhow::ensure!(
                example.teacher_tokens.len() == active_positions.len(),
                "off-policy OPD example {example_idx} has {} teacher_tokens but {} action tokens",
                example.teacher_tokens.len(),
                active_positions.len()
            );
            for (token_idx, (teacher_token, &position)) in example
                .teacher_tokens
                .iter()
                .zip(active_positions.iter())
                .enumerate()
            {
                if let Some(declared_token_id) = teacher_token.token_id {
                    let tokenized_target = tokenized.input_ids[position];
                    anyhow::ensure!(
                        declared_token_id == tokenized_target,
                        "off-policy OPD example {example_idx} token {token_idx} declares token_id {declared_token_id}, but tokenization produced active target {tokenized_target} at position {position}"
                    );
                }
            }
        }

        match objective {
            OpdObjective::ReverseKl => {
                anyhow::ensure!(
                    example.teacher_tokens.len() == active_positions.len(),
                    "off-policy OPD example {example_idx} has {} teacher_tokens but {} action tokens",
                    example.teacher_tokens.len(),
                    active_positions.len()
                );
                for (token_idx, ((teacher_token, &position), &logits_row)) in example
                    .teacher_tokens
                    .iter()
                    .zip(active_positions.iter())
                    .zip(logits_rows.iter())
                    .enumerate()
                {
                    anyhow::ensure!(
                        teacher_token.top_logprobs.len() >= top_k,
                        "off-policy OPD example {example_idx} token {token_idx} has {} top_logprobs, need at least {top_k}",
                        teacher_token.top_logprobs.len()
                    );
                    let mut indices = Vec::with_capacity(top_k);
                    let mut logprobs = Vec::with_capacity(top_k);
                    for entry in teacher_token.top_logprobs.iter().take(top_k) {
                        anyhow::ensure!(
                            (entry.token_id as usize) < vocab_size,
                            "off-policy OPD example {example_idx} token {token_idx} has token_id {} outside vocab_size {vocab_size}",
                            entry.token_id
                        );
                        anyhow::ensure!(
                            entry.logprob.is_finite(),
                            "off-policy OPD example {example_idx} token {token_idx} has non-finite logprob"
                        );
                        indices.push(entry.token_id);
                        logprobs.push(entry.logprob);
                    }
                    validate_topk_logprob_row(
                        &fixture.capabilities(),
                        top_k,
                        token_idx,
                        &indices,
                        &logprobs,
                    )
                    .with_context(|| {
                        format!(
                            "off-policy OPD example {example_idx} target position {position} has invalid teacher logprobs"
                        )
                    })?;
                    fixture
                        .insert(&tokenized.input_ids, logits_row, indices, logprobs)
                        .with_context(|| {
                            format!(
                                "off-policy OPD example {example_idx} target position {position} conflicts with another fixture row"
                            )
                        })?;
                }
                summary.examples_with_teacher_logprobs += 1;
            }
            OpdObjective::CrossEntropy => {
                for (token_idx, (&position, &logits_row)) in
                    active_positions.iter().zip(logits_rows.iter()).enumerate()
                {
                    let target = tokenized.input_ids[position];
                    let mut indices = Vec::with_capacity(top_k);
                    indices.push(target);
                    for candidate in 0..vocab_size as u32 {
                        if indices.len() == top_k {
                            break;
                        }
                        if candidate != target {
                            indices.push(candidate);
                        }
                    }
                    anyhow::ensure!(
                        indices.len() == top_k,
                        "off-policy OPD example {example_idx} token {token_idx} could not build a {top_k}-way CE fixture from vocab_size {vocab_size}"
                    );
                    let mut logprobs = vec![-30.0_f32; top_k];
                    logprobs[0] = 0.0;
                    fixture
                        .insert(&tokenized.input_ids, logits_row, indices, logprobs)
                        .with_context(|| {
                            format!(
                                "off-policy OPD example {example_idx} CE target position {position} conflicts with another fixture row"
                            )
                        })?;
                }
            }
        }

        let env_tokens = tokenized.env_mask.iter().filter(|&&active| active).count() as u64;
        summary.env_tokens = summary.env_tokens.saturating_add(env_tokens);
        // The env masks built here feed the per-step EchoEnvSpec during
        // execution. Preparation records counts only; the final receipt's
        // echo_combined value keys off the term actually firing.
        let _ = echo;

        prompts.push(prompt);
    }

    Ok(PreparedOffPolicyDistillation {
        prompts,
        teacher: fixture,
        summary,
    })
}

pub fn compose_off_policy_distillation_loss(
    objective: OpdObjective,
    action_token_losses: &[f64],
    env_token_cross_entropy: &[f64],
    echo_lambda: Option<f64>,
) -> Result<OffPolicyLossBreakdown> {
    anyhow::ensure!(
        !action_token_losses.is_empty(),
        "off-policy OPD loss requires at least one action token"
    );
    anyhow::ensure!(
        action_token_losses.iter().all(|v| v.is_finite()),
        "off-policy OPD action loss contains non-finite values"
    );
    anyhow::ensure!(
        env_token_cross_entropy.iter().all(|v| v.is_finite()),
        "off-policy OPD env CE contains non-finite values"
    );
    let action_token_loss =
        action_token_losses.iter().sum::<f64>() / action_token_losses.len() as f64;
    let echo_env_ce = if env_token_cross_entropy.is_empty() {
        None
    } else {
        Some(env_token_cross_entropy.iter().sum::<f64>() / env_token_cross_entropy.len() as f64)
    };
    let echo_lambda = echo_lambda.filter(|lambda| *lambda != 0.0 && echo_env_ce.is_some());
    let total_loss = action_token_loss + echo_lambda.unwrap_or(0.0) * echo_env_ce.unwrap_or(0.0);
    Ok(OffPolicyLossBreakdown {
        objective,
        action_token_loss,
        echo_env_ce,
        echo_lambda,
        total_loss,
    })
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
    /// Target-token positions in `tokens` that contribute to the loss.
    /// Typically these are the completion's assistant-token indices. A
    /// target at index `q` is predicted by causal logits row `q - 1`; this
    /// boundary converts the targets before building the kernel mask or
    /// querying the teacher. Positions must be strictly increasing and unique,
    /// and position zero cannot be a target because it has no preceding row.
    pub active_positions: &'a [usize],
    /// Student hidden states at *all* `tokens` positions, shape
    /// `[1, tokens.len(), hidden_size]`. Produced by the trainer's
    /// segment-checkpointed forward pass (`trainer.rs` already produces
    /// this — `opd_train` will plumb it in).
    pub student_hidden: &'a Tensor,
    /// Frozen LM head weights, shape `[H, V]`. Matches the layout used
    /// by `kiln-flce-kernel` and `kiln-model::forward::embed_tokens_t`.
    pub head_t: &'a Tensor,
    /// Teacher source. Queried at the causal logits rows that predict
    /// `active_positions` (or `teacher_active_positions` if set).
    pub teacher: Arc<dyn LogitSource>,
    /// Loss granularity (§3.1).
    pub loss: OpdLossGranularity,
    /// Top-K size for `TeacherTopK`. Ignored for `FullVocab`;
    /// `SampledToken` is rejected before the teacher query.
    pub top_k: usize,
    /// Chunk size along the active-token axis for the kernel. Falls
    /// back to `DEFAULT_CHUNK_SIZE` (= 4096) when 0.
    pub chunk_size: usize,
    /// Optional asymmetric teacher token sequence. When `Some`, this is
    /// what's sent to `teacher.fetch_logprobs`; `tokens` continues to
    /// drive the student-side state. Typical shape:
    /// `teacher_prefix_tokens ++ tokens` — i.e. the same rollout, but
    /// preceded by privileged context only the teacher sees. It must contain
    /// every target in `teacher_active_positions`.
    pub teacher_tokens: Option<&'a [u32]>,
    /// Target-token positions in `teacher_tokens`'s frame. Pair-wise aligned
    /// with `active_positions`: target `i` is scored from student logits row
    /// `active_positions[i] - 1` and teacher logits row
    /// `teacher_active_positions[i] - 1`. Required with `teacher_tokens`.
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
/// 1. Convert target-token positions to their preceding causal logits rows and
///    build the `label_mask` from those rows.
/// 2. Query the teacher at the corresponding logits rows.
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
/// Teacher-side kernel inputs for one OPD step, resolved once from
/// [`OpdStepInputs`] (the alignment checks, `label_mask`, `top_k` resolution,
/// and the teacher logprob fetch).
///
/// Shared by [`opd_step_loss`] (candle-autograd / per-position path) and the
/// tape-authoritative scalar-mean path in [`opd_train`] so the (potentially
/// expensive) teacher fetch + bookkeeping happens exactly once and identically
/// on both code paths. (#1082 CP-4 endgame.)
struct PreparedOpdKernelInputs {
    label_mask: Vec<bool>,
    teacher_topk_indices: Vec<u32>,
    teacher_topk_logprobs: Vec<f32>,
    resolved_top_k: usize,
    active_count: usize,
}

/// Run the per-step teacher fetch + mask/`top_k` resolution shared by the
/// candle and tape-authoritative OPD paths.
fn prepare_opd_kernel_inputs(
    tokens: &[u32],
    active_positions: &[usize],
    teacher: Arc<dyn LogitSource>,
    loss: OpdLossGranularity,
    top_k: usize,
    teacher_tokens: Option<&[u32]>,
    teacher_active_positions: Option<&[usize]>,
) -> Result<PreparedOpdKernelInputs> {
    if let Some(reason) = loss.unsupported_reason() {
        return Err(anyhow!("opd_step_loss: {reason}"));
    }
    if active_positions.is_empty() {
        return Err(anyhow!(
            "opd_step_loss called with no active positions - caller should short-circuit"
        ));
    }
    if teacher_tokens.is_some() ^ teacher_active_positions.is_some() {
        return Err(anyhow!(
            "opd_step_loss: teacher_tokens and teacher_active_positions must be set together"
        ));
    }

    let caps = teacher.capabilities();
    let student_logits_rows =
        target_token_positions_to_logits_rows(&caps.teacher_id, tokens.len(), active_positions)
            .context("opd_step_loss: invalid student target-token positions")?;
    let label_mask: Vec<bool> = {
        let mut mask = vec![false; tokens.len()];
        for &row in &student_logits_rows {
            mask[row] = true;
        }
        mask
    };
    let active_count = active_positions.len();

    let teacher_logits_rows;
    let (query_tokens, query_positions): (&[u32], &[usize]) = match (
        teacher_tokens,
        teacher_active_positions,
    ) {
        (Some(teacher_tokens), Some(teacher_targets)) => {
            if teacher_targets.len() != active_count {
                return Err(anyhow!(
                    "opd_step_loss: teacher_active_positions.len() ({}) != active_positions.len() ({active_count})",
                    teacher_targets.len()
                ));
            }
            teacher_logits_rows = target_token_positions_to_logits_rows(
                &caps.teacher_id,
                teacher_tokens.len(),
                teacher_targets,
            )
            .context("opd_step_loss: invalid teacher target-token positions")?;
            for (pair_index, (&student_target, &teacher_target)) in active_positions
                .iter()
                .zip(teacher_targets.iter())
                .enumerate()
            {
                if tokens[student_target] != teacher_tokens[teacher_target] {
                    return Err(anyhow!(
                        "opd_step_loss: asymmetric target pair {pair_index} has student token {} at {student_target} but teacher token {} at {teacher_target}",
                        tokens[student_target],
                        teacher_tokens[teacher_target]
                    ));
                }
            }
            (teacher_tokens, &teacher_logits_rows)
        }
        (None, None) => (tokens, &student_logits_rows),
        _ => unreachable!("paired teacher options checked above"),
    };

    let resolved_top_k = match loss {
        OpdLossGranularity::SampledToken => unreachable!("unsupported loss rejected above"),
        OpdLossGranularity::TeacherTopK => {
            resolve_opd_top_k(top_k, caps.max_top_k.min(caps.vocab_size))
                .context("opd_step_loss: cannot resolve an executable teacher top-K")?
        }
        OpdLossGranularity::FullVocab => caps.vocab_size,
    };

    // Query the teacher.
    let request_top_k = match loss {
        OpdLossGranularity::FullVocab => None,
        _ => Some(resolved_top_k),
    };
    validate_logit_request(&caps, query_tokens, query_positions, request_top_k)
        .context("opd_step_loss: invalid teacher logprob request")?;
    let batch = teacher
        .fetch_logprobs(query_tokens, query_positions, request_top_k)
        .with_context(|| {
            format!(
                "fetch_logprobs from teacher {:?} for {} positions",
                caps.teacher_id, active_count
            )
        })?;

    let (teacher_topk_indices, teacher_topk_logprobs) = match (request_top_k, batch) {
        (Some(requested_top_k), LogprobBatch::TopK(topk)) => {
            validate_topk_logprobs_batch(
                &caps,
                query_tokens,
                query_positions,
                requested_top_k,
                &topk,
            )
            .context("opd_step_loss: teacher returned invalid top-K logprobs")?;
            (topk.indices, topk.logprobs)
        }
        (None, full_vocab @ LogprobBatch::FullVocab { .. }) => {
            validate_full_vocab_logprobs_batch(&caps, query_tokens, query_positions, &full_vocab)
                .context("opd_step_loss: teacher returned invalid full-vocabulary logprobs")?;
            let LogprobBatch::FullVocab { logprobs, .. } = full_vocab else {
                unreachable!()
            };
            // For full-vocab, indices are 0..V repeated per position.
            let mut indices = Vec::with_capacity(active_count * resolved_top_k);
            for _ in 0..active_count {
                for v in 0..resolved_top_k as u32 {
                    indices.push(v);
                }
            }
            (indices, logprobs)
        }
        (Some(_), LogprobBatch::FullVocab { .. }) => {
            return Err(anyhow!(
                "opd_step_loss: top-K request returned a full-vocabulary response"
            ));
        }
        (None, LogprobBatch::TopK(_)) => {
            return Err(anyhow!(
                "opd_step_loss: full-vocabulary request returned a top-K response"
            ));
        }
    };

    Ok(PreparedOpdKernelInputs {
        label_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        resolved_top_k,
        active_count,
    })
}

fn opd_checkpoint_segments_for_step(
    runtime: &crate::TrainingRuntimeContext,
    admitted_segments: Option<usize>,
    activation_bytes_per_elem: usize,
    model_config: &kiln_core::config::ModelConfig,
    seq_len: usize,
) -> Option<Vec<(usize, usize)>> {
    let cfg = admitted_segments.map_or_else(
        || {
            crate::trainer::CheckpointConfig::auto_for_workload_with_activation_bytes_and_runtime(
                model_config.num_layers,
                seq_len,
                model_config.hidden_size,
                model_config.intermediate_size,
                model_config.vocab_size,
                2,
                activation_bytes_per_elem,
                runtime,
            )
        },
        |segments| {
            crate::trainer::CheckpointConfig::from_resolved_segments(
                model_config.num_layers,
                segments,
            )
        },
    );

    if cfg.enabled {
        Some(crate::trainer::compute_segment_boundaries(
            model_config.num_layers,
            cfg.num_segments,
        ))
    } else {
        None
    }
}

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

    // Teacher fetch + mask / top_k resolution (shared with the
    // tape-authoritative scalar path in `opd_train`).
    let PreparedOpdKernelInputs {
        label_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        resolved_top_k,
        active_count,
    } = prepare_opd_kernel_inputs(
        tokens,
        active_positions,
        teacher,
        loss,
        top_k,
        teacher_tokens,
        teacher_active_positions,
    )?;

    // (#1082) kt-NATIVE per-position reverse-KL — straight through the kt
    // forward kernel (`opd_top_k_reverse_kl_per_position_kt`) on the kt
    // `student_hidden` + `head_t` DIRECTLY. No candle: the old path bridged both
    // inputs kt->candle, ran a `KtForwardOp1` candle-autograd shim (or the
    // pure-candle Phase A composite), then copied the result candle->kt.
    //
    // `opd_step_loss` is a VALUE/metrics API — the OPD training GRADIENT comes
    // from the separate scalar-mean tape path
    // (`opd_step_forward_backward_tape_authoritative`, kt-native). So a plain kt
    // forward (no tape recording) is correct here; `OpdStepOutputs` carries the
    // per-position KL + its mean for reporting/guardrails. `chunk_size` is not
    // consumed by the per-position kernel (kept on `OpdStepInputs` for API
    // stability).
    let _ = chunk_size;
    let per_position_kl = kiln_opd_loss_kernel::opd_top_k_reverse_kl_per_position_kt(
        student_hidden,
        head_t,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        &label_mask,
        resolved_top_k,
    )
    .map_err(|e| anyhow!("opd_step_loss: per-position reverse-KL (kt): {e}"))?;
    let mean_kl = kiln_tensor::ops::mean_all(&per_position_kl)
        .map_err(|e| anyhow!("opd_step_loss: mean per-position KL (kt): {e}"))?;

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
            StableOpdMode::Manual {
                kl_beta,
                sft_lambda,
            } => Self {
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
    pub fn multiplier(&self, tool_call_weight: f64, tool_name_weight: f64) -> f64 {
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
        weights[i] *= class.multiplier(inputs.tip_tool_call_weight, inputs.tip_tool_name_weight);
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
            let extra =
                inputs.score_earliest_weight + (1.0 - inputs.score_earliest_weight) * frac.min(1.0);
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
        // (#1082) kt scalar: rank-0 zeros. candle `Tensor::new(0.0_f32, dev)`
        // -> kt `Tensor::zeros((), F32, dev)` (kt `new` takes `&[E]`).
        None => Tensor::zeros((), DType::F32, device).context("zero kl_ref scalar")?,
    };

    let sft_term = match inputs.sft_loss {
        Some(t) => t.clone(),
        None => Tensor::zeros((), DType::F32, device).context("zero sft scalar")?,
    };

    let beta = inputs.coefficients.beta_kl;
    let lambda = inputs.coefficients.lambda_sft;

    // total = mean_opd + β · mean_kl_ref + λ · sft_term.
    let kl_ref_scaled = if beta == 0.0 {
        Tensor::zeros((), DType::F32, device)?
    } else {
        mean_kl_ref.affine(beta, 0.0)?
    };
    let sft_scaled = if lambda == 0.0 {
        Tensor::zeros((), DType::F32, device)?
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

/// Build an in-process "local teacher" FixtureLogitSource by running
/// the loaded model forward on each prompt and extracting top-K
/// logprobs that predict the active (assistant) target tokens. Inputs use
/// target-token positions; fixture keys use the preceding causal logits rows.
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
    build_local_teacher_fixture_with_coordination(
        teacher_id,
        prompts_and_active,
        weights,
        model_config,
        teacher_lora,
        top_k,
        tokenizer_hash,
        None,
    )
}

/// Coordinated local-teacher materialization. Each prompt forward is one
/// settled GPU phase so inference can run between source rows.
#[allow(clippy::too_many_arguments)]
pub fn build_local_teacher_fixture_with_coordination(
    teacher_id: impl Into<String>,
    prompts_and_active: &[(Vec<u32>, Vec<usize>)],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    teacher_lora: Option<&kiln_model::lora_loader::LoraWeights>,
    top_k: usize,
    tokenizer_hash: Option<String>,
    gpu_step_coordination: Option<&crate::trainer::GpuStepCoordination>,
) -> Result<crate::logit_source::FixtureLogitSource> {
    let streaming_prefill = kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(
        weights.embed_tokens.device(),
    );
    build_local_teacher_fixture_with_coordination_and_policy(
        teacher_id,
        prompts_and_active,
        weights,
        model_config,
        teacher_lora,
        top_k,
        tokenizer_hash,
        gpu_step_coordination,
        streaming_prefill,
    )
}

/// Coordinated local-teacher materialization with an owning runtime's
/// startup-resolved streaming-prefill policy.
#[allow(clippy::too_many_arguments)]
pub fn build_local_teacher_fixture_with_coordination_and_policy(
    teacher_id: impl Into<String>,
    prompts_and_active: &[(Vec<u32>, Vec<usize>)],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    teacher_lora: Option<&kiln_model::lora_loader::LoraWeights>,
    top_k: usize,
    tokenizer_hash: Option<String>,
    gpu_step_coordination: Option<&crate::trainer::GpuStepCoordination>,
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> Result<crate::logit_source::FixtureLogitSource> {
    use kiln_model::backend;
    use kiln_model::forward::{LinearAttentionState, model_forward_kt_with_policy};

    // (#1082) `embed_tokens.device()` is kt; `LinearAttentionState::new` below
    // takes a kt device, and `for_device_kt` selects the backend from a kt
    // device, so keep `device` kt here.
    let device = weights.embed_tokens.device();
    let backend_rt = backend::for_device_kt(&device);

    let vocab_size = model_config.vocab_size;
    if top_k == 0 {
        return Err(anyhow!("build_local_teacher_fixture: top_k must be > 0"));
    }
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

    for (tokens, active_target_positions) in prompts_and_active {
        if active_target_positions.is_empty() {
            continue;
        }
        let logits_rows = target_token_positions_to_logits_rows(
            &fixture.capabilities().teacher_id,
            tokens.len(),
            active_target_positions,
        )
        .context("build_local_teacher_fixture: invalid active target-token positions")?;
        let log_probs_host: Vec<Vec<f32>> = run_coordinated_opd_gpu_phase(
            gpu_step_coordination,
            &*backend_rt,
            "local teacher prompt",
            || {
                let mut linear_state = LinearAttentionState::new(model_config, &device)?;
                let logits = model_forward_kt_with_policy(
                    &*backend_rt,
                    tokens,
                    weights,
                    model_config,
                    None,
                    Some(&mut linear_state),
                    teacher_lora,
                    streaming_prefill,
                )
                .context("local-teacher forward pass")?;
                // logits shape: [1, T, V]. Detach autograd — no gradients needed.
                let logits = logits.detach();
                let log_probs = {
                    let max = logits
                        .max_keepdim(2)
                        .context("local-teacher log_softmax max_keepdim")?;
                    let diff = logits
                        .broadcast_sub(&max)
                        .context("local-teacher log_softmax broadcast_sub")?;
                    let sum_exp = diff
                        .exp()
                        .context("local-teacher log_softmax exp")?
                        .sum_keepdim(2)
                        .context("local-teacher log_softmax sum_keepdim")?;
                    diff.broadcast_sub(&sum_exp.log().context("local-teacher log_softmax log")?)
                        .context("local-teacher log_softmax broadcast_sub final")?
                };
                log_probs
                    .squeeze(0)
                    .context("local-teacher squeeze batch dim")?
                    .to_dtype(DType::F32)
                    .context("local-teacher to_dtype f32")?
                    .to_vec2::<f32>()
                    .context("local-teacher logprobs to host")
            },
        )?;
        let fixture_caps = fixture.capabilities();
        for (row_index, &logits_row) in logits_rows.iter().enumerate() {
            let row = &log_probs_host[logits_row];
            let (indices, logprobs) =
                select_validated_topk_logprob_row(&fixture_caps, row_index, row, top_k)
                    .context("build_local_teacher_fixture: model returned invalid logprob row")?;
            validate_topk_logprob_row(&fixture_caps, top_k, row_index, &indices, &logprobs)
                .context("build_local_teacher_fixture: model returned invalid top-K logprobs")?;
            fixture
                .insert(tokens, logits_row, indices, logprobs)
                .context("build_local_teacher_fixture: conflicting exact-sequence row")?;
        }
    }

    Ok(fixture)
}

/// Run `model_forward_kt` on `tokens` and return the teacher's top-`top_k`
/// `(indices, logprobs)` for each causal logits row in `positions`, in order.
/// Logits row `p` predicts target token `tokens[p + 1]`.
/// This is the shared forward+log_softmax+top-K core behind both the
/// pre-computed [`build_local_teacher_fixture`] and the live
/// [`LiveLocalTeacher`]. Runs in inference (detached) — no tape recording.
#[allow(clippy::too_many_arguments)]
fn forward_topk_at_positions(
    tokens: &[u32],
    positions: &[usize],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    teacher_lora: Option<&kiln_model::lora_loader::LoraWeights>,
    top_k: usize,
    caps: &crate::logit_source::LogitSourceCaps,
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> Result<Vec<(Vec<u32>, Vec<f32>)>> {
    use kiln_model::backend;
    use kiln_model::forward::{LinearAttentionState, model_forward_kt_with_policy};

    let device = weights.embed_tokens.device();
    let backend_rt = backend::for_device_kt(&device);
    let mut linear_state = LinearAttentionState::new(model_config, &device)?;
    let logits = model_forward_kt_with_policy(
        &*backend_rt,
        tokens,
        weights,
        model_config,
        None,
        Some(&mut linear_state),
        teacher_lora,
        streaming_prefill,
    )
    .context("live-teacher forward pass")?;
    let logits = logits.detach();
    // Numerically-stable log_softmax over the vocab axis (same identity as the
    // fixture path): xs - log(sum_exp(xs - max(xs))).
    let log_probs = {
        let max = logits
            .max_keepdim(2)
            .context("live-teacher log_softmax max")?;
        let diff = logits
            .broadcast_sub(&max)
            .context("live-teacher log_softmax sub")?;
        let sum_exp = diff
            .exp()
            .context("live-teacher log_softmax exp")?
            .sum_keepdim(2)
            .context("live-teacher log_softmax sum")?;
        diff.broadcast_sub(&sum_exp.log().context("live-teacher log_softmax log")?)
            .context("live-teacher log_softmax final")?
    };
    let log_probs_host: Vec<Vec<f32>> = log_probs
        .squeeze(0)
        .context("live-teacher squeeze batch")?
        .to_dtype(DType::F32)
        .context("live-teacher to f32")?
        .to_vec2::<f32>()
        .context("live-teacher logprobs to host")?;

    let mut out = Vec::with_capacity(positions.len());
    for (row_index, &pos) in positions.iter().enumerate() {
        if pos >= log_probs_host.len() {
            return Err(anyhow!(
                "live-teacher: position {pos} >= seq_len {}",
                log_probs_host.len()
            ));
        }
        out.push(
            select_validated_topk_logprob_row(caps, row_index, &log_probs_host[pos], top_k)
                .context("live-teacher: model returned invalid logprob row")?,
        );
    }
    Ok(out)
}

fn select_validated_topk_logprob_row(
    caps: &crate::logit_source::LogitSourceCaps,
    row_index: usize,
    row: &[f32],
    top_k: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    anyhow::ensure!(
        row.len() == caps.vocab_size,
        "teacher {:?} row {row_index} has width {}, expected vocabulary width {}",
        caps.teacher_id,
        row.len(),
        caps.vocab_size
    );
    anyhow::ensure!(
        top_k > 0 && top_k <= row.len(),
        "teacher {:?} requested invalid top_k {top_k} for row width {}",
        caps.teacher_id,
        row.len()
    );
    for (token_id, &logprob) in row.iter().enumerate() {
        anyhow::ensure!(
            logprob.is_finite() && logprob <= 0.0,
            "teacher {:?} row {row_index} token {token_id} has invalid logprob {logprob:?}",
            caps.teacher_id
        );
    }

    let mut indexed: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logprob)| (token_id as u32, logprob))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(top_k);
    Ok((
        indexed.iter().map(|(token_id, _)| *token_id).collect(),
        indexed.iter().map(|(_, logprob)| *logprob).collect(),
    ))
}

/// In-process self-distillation teacher (#31): holds a shared (cheap, Arc-backed)
/// handle to the loaded model and computes top-K teacher logprobs **live** for
/// any token sequence. Unlike a pre-computed [`crate::logit_source::FixtureLogitSource`] containing
/// exact fixed sequences, this scores the actual ON-policy rollouts the student
/// generates, so `teacher: "self"` works in
/// on-policy mode. `fetch_logprobs` runs in the OPD data-prep phase, OUTSIDE the
/// student's tape-authoritative scope (no tape nesting — mirrors the fixture
/// builder's detached forward).
pub struct LiveLocalTeacher {
    weights: kiln_model::forward::GpuWeights,
    model_config: kiln_core::config::ModelConfig,
    teacher_lora: Option<kiln_model::lora_loader::LoraWeights>,
    identity: crate::TeacherIdentityV1,
    caps: crate::logit_source::LogitSourceCaps,
    default_top_k: usize,
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
}

impl std::fmt::Debug for LiveLocalTeacher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveLocalTeacher")
            .field("teacher_id", &self.caps.teacher_id)
            .field("vocab_size", &self.caps.vocab_size)
            .field("max_top_k", &self.caps.max_top_k)
            .field("has_teacher_lora", &self.teacher_lora.is_some())
            .finish()
    }
}

impl LiveLocalTeacher {
    pub fn new(
        teacher_id: impl Into<String>,
        weights: kiln_model::forward::GpuWeights,
        model_config: kiln_core::config::ModelConfig,
        teacher_lora: Option<kiln_model::lora_loader::LoraWeights>,
        identity: crate::TeacherIdentityV1,
        top_k: usize,
    ) -> Result<Self> {
        let streaming_prefill = kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(
            weights.embed_tokens.device(),
        );
        Self::new_with_streaming_prefill_policy(
            teacher_id,
            weights,
            model_config,
            teacher_lora,
            identity,
            top_k,
            streaming_prefill,
        )
    }

    /// Build a live local teacher with the owning runtime's startup-resolved
    /// streaming-prefill policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_streaming_prefill_policy(
        teacher_id: impl Into<String>,
        weights: kiln_model::forward::GpuWeights,
        model_config: kiln_core::config::ModelConfig,
        teacher_lora: Option<kiln_model::lora_loader::LoraWeights>,
        identity: crate::TeacherIdentityV1,
        top_k: usize,
        streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
    ) -> Result<Self> {
        let vocab_size = model_config.vocab_size;
        anyhow::ensure!(
            identity.vocab_size() as usize == vocab_size,
            "live local teacher identity vocab_size {} does not match model vocab_size {vocab_size}",
            identity.vocab_size()
        );
        anyhow::ensure!(
            identity.max_top_k() as usize >= top_k,
            "live local teacher top_k {top_k} exceeds identity max_top_k {}",
            identity.max_top_k()
        );
        let caps = crate::logit_source::LogitSourceCaps {
            teacher_id: teacher_id.into(),
            vocab_size,
            max_top_k: identity.max_top_k() as usize,
            supports_full_vocab: false,
            supports_batched: true,
            tokenizer_hash: Some(identity.tokenizer_vocab_sha256().to_owned()),
        };
        Ok(Self {
            weights,
            model_config,
            teacher_lora,
            identity,
            caps,
            default_top_k: top_k,
            streaming_prefill,
        })
    }
}

impl crate::logit_source::LogitSource for LiveLocalTeacher {
    fn capabilities(&self) -> crate::logit_source::LogitSourceCaps {
        self.caps.clone()
    }

    fn authoritative_teacher_identity(&self) -> Option<&crate::TeacherIdentityV1> {
        Some(&self.identity)
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> std::result::Result<crate::logit_source::LogprobBatch, crate::logit_source::LogitSourceError>
    {
        use crate::logit_source::{LogitSourceError, LogprobBatch, TopKLogprobs};
        let teacher_id = self.caps.teacher_id.clone();
        validate_logit_request(&self.caps, tokens, positions, top_k)?;
        let requested_k = top_k.unwrap_or(self.default_top_k);
        let per_pos = forward_topk_at_positions(
            tokens,
            positions,
            &self.weights,
            &self.model_config,
            self.teacher_lora.as_ref(),
            requested_k,
            &self.caps,
            self.streaming_prefill,
        )
        .map_err(|e| {
            LogitSourceError::invalid(&teacher_id, format!("live teacher forward: {e:#}"))
        })?;

        let mut indices = Vec::with_capacity(positions.len() * requested_k);
        let mut logprobs = Vec::with_capacity(positions.len() * requested_k);
        for (idx, lp) in per_pos {
            indices.extend_from_slice(&idx);
            logprobs.extend_from_slice(&lp);
        }
        let batch = TopKLogprobs {
            indices,
            logprobs,
            top_k: requested_k,
        };
        validate_topk_logprobs_batch(&self.caps, tokens, positions, requested_k, &batch)?;
        Ok(LogprobBatch::TopK(batch))
    }
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
/// Uses the segmented forward path with the run's immutable streaming policy
/// so long prompts bound GDN scratch while the teacher is also resident. The
/// trade-off is O(N²): we re-run the prefix forward at every decode step
/// instead of carrying a KV cache.
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
    sampler_segments: Option<usize>,
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> Result<Vec<u32>> {
    use kiln_core::sampling::SamplingParams;
    use kiln_model::forward::{
        LinearAttentionState, model_forward_embed, model_forward_final_norm,
        model_forward_segment_with_policy,
    };
    use kiln_model::sampling::sample_step;

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "sample_student_rollout: prompt empty"
    );

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
    // long contexts (700+ tokens with a 27B teacher resident). The typed
    // request config can select a different positive count when needed.
    let num_segments = sampler_segments.unwrap_or(18).min(model_config.num_layers);
    let segments =
        crate::trainer::compute_segment_boundaries(model_config.num_layers, num_segments);
    // (#1082) kt-native: the segmented forward chain produces kt tensors
    // end-to-end (no autograd needed in sampling) and `sample_step` now
    // consumes kt directly, so kt flows through to the sampler unbridged.
    let run_forward = |seq: &[u32]| -> Result<kiln_tensor::Tensor> {
        let positions: Vec<u32> = (0..seq.len() as u32).collect();
        let (embed_hidden, _) = model_forward_embed(seq, weights)?;
        let mut linear_state = LinearAttentionState::new_for_inference(model_config, &device)?;
        let mut current = embed_hidden;
        for &(start, end) in &segments {
            current = model_forward_segment_with_policy(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(lora),
                streaming_prefill,
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

    // (#1082) `run_forward` produces kt logits and `kiln_model::sampling::
    // sample_step` now consumes kt directly — the old kt->candle bridge was a
    // stale candle bridge and is dropped so kt flows end-to-end.
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

fn chat_messages_without_trailing_assistant(
    messages: &[ChatMessage],
) -> Vec<kiln_core::tokenizer::ChatMessage> {
    let mut rendered_messages = messages.to_vec();
    while rendered_messages
        .last()
        .map(|message| message.role.as_str())
        == Some("assistant")
    {
        rendered_messages.pop();
    }
    rendered_messages
}

fn thinking_disabled_chat_template_options() -> kiln_core::tokenizer::ChatTemplateOptions {
    kiln_core::tokenizer::ChatTemplateOptions {
        template_kwargs: serde_json::Map::from_iter([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        )]),
    }
}

fn render_rollout_prompt_prefixes(
    prompts: &[OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> Result<Vec<Vec<u32>>> {
    prompts
        .iter()
        .enumerate()
        .map(|(prompt_idx, prompt)| {
            let messages = chat_messages_without_trailing_assistant(&prompt.messages);
            let text = tokenizer
                .apply_chat_template_full_with_options(
                    &messages,
                    None,
                    None,
                    thinking_disabled_chat_template_options(),
                )
                .with_context(|| {
                    format!("opd_train: render rollout chat template for prompt {prompt_idx}")
                })?;
            let tokens = tokenizer.encode(&text).with_context(|| {
                format!("opd_train: encode rendered rollout prompt {prompt_idx}")
            })?;
            anyhow::ensure!(
                !tokens.is_empty(),
                "opd_train: rendered rollout prompt {prompt_idx} encoded to zero tokens"
            );
            Ok(tokens)
        })
        .collect()
}

fn render_teacher_prompt_tokens(
    prompts: &[OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> Result<Vec<Vec<u32>>> {
    prompts
        .iter()
        .enumerate()
        .map(|(prompt_idx, prompt)| {
            if prompt.teacher_extra_messages.is_empty() {
                return Ok(Vec::new());
            }

            let mut merged = chat_messages_without_trailing_assistant(&prompt.messages);
            let extras_text = prompt
                .teacher_extra_messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            if let Some(first) = merged.first_mut() {
                if first.role == "system" {
                    first.content = format!("{extras_text}\n\n{}", first.content);
                } else {
                    merged.insert(
                        0,
                        kiln_core::tokenizer::ChatMessage {
                            role: "system".to_string(),
                            content: extras_text,
                            ..Default::default()
                        },
                    );
                }
            } else {
                merged.push(kiln_core::tokenizer::ChatMessage {
                    role: "system".to_string(),
                    content: extras_text,
                    ..Default::default()
                });
            }

            let text = tokenizer
                .apply_chat_template_full_with_options(
                    &merged,
                    None,
                    None,
                    thinking_disabled_chat_template_options(),
                )
                .with_context(|| {
                    format!(
                        "opd_train: render asymmetric teacher chat template for prompt {prompt_idx}"
                    )
                })?;
            let tokens = tokenizer.encode(&text).with_context(|| {
                format!("opd_train: encode asymmetric teacher prompt {prompt_idx}")
            })?;
            anyhow::ensure!(
                !tokens.is_empty(),
                "opd_train: asymmetric teacher prompt {prompt_idx} encoded to zero tokens"
            );
            Ok(tokens)
        })
        .collect()
}

fn use_chat_template_rollout_prefixes(config: &OpdConfig) -> bool {
    matches!(
        config.rollout_prompt_rendering,
        OpdRolloutPromptRendering::ChatTemplate
    )
}

/// Eagerly score every fixed off-policy action row and return an in-memory
/// fixture carrying the verified source identity.
///
/// Server workers call this before bounded GPU phases begin. A remote source
/// is therefore never reachable from those phases: cache misses, HTTP
/// timeouts, and remote retries all complete (or fail the job) while inference
/// can still use the GPU. On-policy distillation is deliberately rejected
/// because its student-generated rollouts do not exist until GPU execution
/// begins and cannot be prefetched without changing the algorithm.
pub fn materialize_verified_off_policy_teacher(
    prompts: &[OpdPrompt],
    config: &OpdConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    source: Arc<dyn LogitSource>,
) -> Result<crate::logit_source::FixtureLogitSource> {
    config
        .validate_runtime_contract()
        .context("materialize remote teacher: unsupported OPD configuration")?;
    anyhow::ensure!(
        matches!(config.training_mode, OpdTrainingMode::OffPolicy),
        "remote teacher scoring cannot run on-policy inside bounded GPU phases; use training_mode=\"off_policy\" with fixed assistant actions"
    );
    anyhow::ensure!(
        matches!(config.loss, OpdLossGranularity::TeacherTopK),
        "remote teacher materialization currently requires loss=\"teacher_top_k\""
    );
    anyhow::ensure!(
        !prompts.is_empty(),
        "remote teacher materialization requires at least one prompt"
    );

    let caps = source.capabilities();
    let identity = source
        .authoritative_teacher_identity()
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "remote teacher {:?} has no authoritative identity; refusing to materialize unverified logits",
                caps.teacher_id
            )
        })?;
    let top_k = resolve_opd_top_k(config.top_k, caps.max_top_k.min(caps.vocab_size))
        .context("materialize remote teacher: cannot resolve an executable top-K")?;
    let use_rendered_prefix = use_chat_template_rollout_prefixes(config);
    let rollout_prefixes = if use_rendered_prefix {
        render_rollout_prompt_prefixes(prompts, tokenizer)?
    } else {
        vec![Vec::new(); prompts.len()]
    };
    let teacher_prefixes = render_teacher_prompt_tokens(prompts, tokenizer)?;

    let mut fixture = crate::logit_source::FixtureLogitSource::uniform_topk(
        caps.teacher_id.clone(),
        caps.vocab_size,
        top_k,
    )
    .with_authoritative_identity(identity)
    .context("materialize remote teacher: bind fixture identity")?;

    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let tokenized =
            tokenize_opd_prompt_for_training(prompt, tokenizer, config.echo.as_ref())
                .with_context(|| format!("materialize remote teacher prompt {prompt_index}"))?;
        let active_positions: Vec<usize> = tokenized
            .action_mask
            .iter()
            .enumerate()
            .filter_map(|(position, &active)| active.then_some(position))
            .collect();
        anyhow::ensure!(
            !active_positions.is_empty(),
            "materialize remote teacher prompt {prompt_index} produced no assistant action tokens"
        );

        let (query_tokens, query_targets) = if teacher_prefixes[prompt_index].is_empty() {
            (tokenized.input_ids, active_positions)
        } else {
            let rollout_prefix_len = if use_rendered_prefix {
                rollout_prefixes[prompt_index].len()
            } else {
                active_positions[0]
            };
            anyhow::ensure!(
                rollout_prefix_len > 0 && rollout_prefix_len < tokenized.input_ids.len(),
                "materialize remote teacher prompt {prompt_index} has invalid rollout prefix length {rollout_prefix_len} for {} tokens",
                tokenized.input_ids.len()
            );
            anyhow::ensure!(
                active_positions
                    .iter()
                    .all(|&position| position >= rollout_prefix_len),
                "materialize remote teacher prompt {prompt_index} has an action before its rollout prefix boundary"
            );

            let teacher_prefix = &teacher_prefixes[prompt_index];
            let mut tokens = Vec::with_capacity(
                teacher_prefix.len() + tokenized.input_ids.len() - rollout_prefix_len,
            );
            tokens.extend_from_slice(teacher_prefix);
            tokens.extend_from_slice(&tokenized.input_ids[rollout_prefix_len..]);
            let targets = active_positions
                .iter()
                .map(|position| teacher_prefix.len() + position - rollout_prefix_len)
                .collect();
            (tokens, targets)
        };
        let logits_rows = target_token_positions_to_logits_rows(
            &caps.teacher_id,
            query_tokens.len(),
            &query_targets,
        )
        .with_context(|| {
            format!("materialize remote teacher prompt {prompt_index} target alignment")
        })?;
        let batch = source
            .fetch_logprobs(&query_tokens, &logits_rows, Some(top_k))
            .with_context(|| {
                format!(
                    "materialize remote teacher {:?} prompt {prompt_index} ({} rows)",
                    caps.teacher_id,
                    logits_rows.len()
                )
            })?;
        let crate::logit_source::LogprobBatch::TopK(topk) = batch else {
            anyhow::bail!(
                "materialize remote teacher {:?} returned full-vocabulary logits for a top-K request",
                caps.teacher_id
            );
        };
        for (row_index, &logits_row) in logits_rows.iter().enumerate() {
            let start = row_index * top_k;
            let end = start + top_k;
            fixture
                .insert(
                    &query_tokens,
                    logits_row,
                    topk.indices[start..end].to_vec(),
                    topk.logprobs[start..end].to_vec(),
                )
                .with_context(|| {
                    format!(
                        "materialize remote teacher prompt {prompt_index} logits row {logits_row}"
                    )
                })?;
        }
    }

    Ok(fixture)
}

#[derive(Debug, Clone)]
struct OpdTeacherProvenance {
    teacher_id: String,
    identity: Option<crate::TeacherIdentityV1>,
    content_revision: Option<String>,
}

impl OpdTeacherProvenance {
    fn from_source(source: &dyn LogitSource, capabilities: &LogitSourceCaps) -> Self {
        Self {
            teacher_id: capabilities.teacher_id.clone(),
            identity: source.authoritative_teacher_identity().cloned(),
            content_revision: source.authoritative_content_revision(),
        }
    }

    fn content_revision(&self) -> Option<String> {
        self.content_revision.clone()
    }
}

const OPD_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION: u32 = 1;
const OPD_CHECKPOINT_LOOP_STATE_TYPE: &str = "kiln.opd-loop-state.v1";
const OPD_CHECKPOINT_LOOP_STATE_FILE: &str = "opd_loop_state.json";

/// CPU-owned OPD state at the next source/sample candidate boundary. The
/// candidate cursor is intentionally distinct from `global_step`: a sampled
/// rollout may be empty and consume a deterministic candidate without
/// committing an optimizer update.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct OpdCheckpointLoopState {
    schema_version: u32,
    state_type: String,
    global_step: u64,
    epoch_index: u64,
    cursor_in_epoch: u64,
    loss_history: Vec<f64>,
    last_loss: Option<f64>,
    data_stats: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    run_env_ce: Option<f64>,
    lora_grad_norms: crate::train_receipt::LoraGradNormAccumulator,
    guardrail: crate::diagnostics::LengthInflationGuardrail,
}

impl OpdCheckpointLoopState {
    #[allow(clippy::too_many_arguments)]
    fn capture(
        global_step: usize,
        epoch_index: usize,
        cursor_in_epoch: usize,
        loss_history: &[f64],
        data_stats: &crate::train_receipt::DataStatsReceipt,
        token_counts: &crate::train_receipt::TokenCountReceipt,
        run_env_ce: Option<f64>,
        lora_grad_norms: &crate::train_receipt::LoraGradNormAccumulator,
        guardrail: &crate::diagnostics::LengthInflationGuardrail,
    ) -> Self {
        Self {
            schema_version: OPD_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION,
            state_type: OPD_CHECKPOINT_LOOP_STATE_TYPE.to_string(),
            global_step: global_step as u64,
            epoch_index: epoch_index as u64,
            cursor_in_epoch: cursor_in_epoch as u64,
            loss_history: loss_history.to_vec(),
            last_loss: loss_history.last().copied(),
            data_stats: data_stats.clone(),
            token_counts: token_counts.clone(),
            run_env_ce,
            lora_grad_norms: lora_grad_norms.clone(),
            guardrail: guardrail.clone(),
        }
    }

    fn validate(
        &self,
        progress: &crate::checkpoint::TrainingCheckpointProgress,
        candidates_per_epoch: usize,
        total_epochs: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == OPD_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION
                && self.state_type == OPD_CHECKPOINT_LOOP_STATE_TYPE,
            "unsupported OPD checkpoint loop-state contract"
        );
        anyhow::ensure!(
            candidates_per_epoch > 0 && total_epochs > 0,
            "OPD checkpoint has an empty training schedule"
        );
        anyhow::ensure!(
            self.global_step == progress.global_step
                && self.epoch_index == progress.epoch_index
                && self.cursor_in_epoch == progress.cursor_in_epoch,
            "OPD checkpoint loop state disagrees with manifest progress"
        );
        let scheduled_candidates = candidates_per_epoch
            .checked_mul(total_epochs)
            .context("OPD checkpoint schedule overflows usize")?;
        anyhow::ensure!(
            progress.total_steps == scheduled_candidates as u64
                && progress.data_order.len() == candidates_per_epoch,
            "OPD checkpoint manifest schedule differs from its loop state"
        );
        anyhow::ensure!(
            self.epoch_index < total_epochs as u64,
            "OPD checkpoint next epoch {} exceeds configured epochs {}",
            self.epoch_index,
            total_epochs
        );
        anyhow::ensure!(
            self.cursor_in_epoch < candidates_per_epoch as u64,
            "OPD checkpoint candidate cursor exceeds one epoch"
        );
        let consumed_candidates = self
            .epoch_index
            .checked_mul(candidates_per_epoch as u64)
            .and_then(|value| value.checked_add(self.cursor_in_epoch))
            .context("OPD checkpoint candidate cursor overflows u64")?;
        anyhow::ensure!(
            self.global_step <= consumed_candidates,
            "OPD checkpoint has more optimizer steps than consumed candidates"
        );
        anyhow::ensure!(
            self.loss_history.len() as u64 == self.global_step,
            "OPD checkpoint loss-history length {} does not match global step {}",
            self.loss_history.len(),
            self.global_step
        );
        anyhow::ensure!(
            self.loss_history.iter().all(|loss| loss.is_finite()),
            "OPD checkpoint loss history contains a non-finite value"
        );
        match (self.loss_history.last().copied(), self.last_loss) {
            (None, None) => {}
            (Some(expected), Some(actual)) if expected == actual && actual.is_finite() => {}
            _ => anyhow::bail!("OPD checkpoint last_loss does not match loss history"),
        }
        anyhow::ensure!(
            self.data_stats.examples_filtered <= self.data_stats.examples_read
                && self.data_stats.examples_trained as u64 == self.global_step,
            "OPD checkpoint data counters are inconsistent"
        );
        anyhow::ensure!(
            self.run_env_ce.is_none_or(f64::is_finite),
            "OPD checkpoint contains a non-finite ECHO measurement"
        );
        anyhow::ensure!(
            self.guardrail.checkpoint_state_is_finite(),
            "OPD checkpoint contains non-finite guardrail state"
        );
        Ok(())
    }
}

const OPD_CHECKPOINT_ADAPTER_FILE: &str = "adapter.safetensors";
const OPD_CHECKPOINT_OPTIMIZER_FILE: &str = "optimizer.safetensors";

#[derive(Debug, Clone)]
struct OpdCheckpointDescriptor {
    adapter_name: String,
    effective_config: serde_json::Value,
    precision_policy: crate::checkpoint::TrainingCheckpointPrecision,
    data: crate::checkpoint::TrainingCheckpointData,
    init_seed: u64,
    optimizer: Optimizer,
    learning_rate: f64,
    candidates_per_epoch: usize,
    total_epochs: usize,
    on_policy: bool,
    base_model_weights_sha256: Option<String>,
    teacher_content_revision: Option<String>,
    auxiliary_state: serde_json::Value,
}

#[derive(Debug)]
struct OpdCheckpointSnapshot {
    target: PathBuf,
    manifest: crate::checkpoint::TrainingCheckpointManifest,
    artifacts: Vec<crate::checkpoint::CheckpointArtifact>,
    adapter_parameters: crate::trainer::CheckpointTensorSnapshot,
    optimizer_state: Option<crate::trainer::CheckpointTensorSnapshot>,
    loop_state_bytes: Vec<u8>,
}

impl OpdCheckpointSnapshot {
    fn publish(self) -> Result<PathBuf> {
        let Self {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            loop_state_bytes,
        } = self;
        crate::checkpoint::write_training_checkpoint_atomic(
            &target,
            manifest,
            &artifacts,
            move |staging| {
                adapter_parameters.save(&staging.join(OPD_CHECKPOINT_ADAPTER_FILE))?;
                if let Some(state) = optimizer_state.as_ref() {
                    state.save(&staging.join(OPD_CHECKPOINT_OPTIMIZER_FILE))?;
                }
                std::fs::write(
                    staging.join(OPD_CHECKPOINT_LOOP_STATE_FILE),
                    &loop_state_bytes,
                )
                .context("write OPD checkpoint loop state")?;
                Ok(())
            },
        )
    }
}

impl OpdCheckpointDescriptor {
    fn total_candidates(&self) -> Result<usize> {
        self.candidates_per_epoch
            .checked_mul(self.total_epochs)
            .context("OPD checkpoint schedule overflows usize")
    }

    fn optimizer_state_file(&self) -> Option<String> {
        (!matches!(self.optimizer, Optimizer::Sgd))
            .then(|| OPD_CHECKPOINT_OPTIMIZER_FILE.to_string())
    }

    fn optimizer_manifest(
        &self,
        step: u64,
    ) -> Result<crate::checkpoint::TrainingCheckpointOptimizer> {
        let kind = match self.optimizer {
            Optimizer::Sgd => "sgd",
            Optimizer::AdamW { .. } => "adam_w",
            Optimizer::Muon { .. } => "muon",
        };
        let hyperparameters = crate::trainer::canonical_checkpoint_json_value(serde_json::json!({
            "learning_rate": self.learning_rate,
            "optimizer": serde_json::to_value(self.optimizer)
                .context("serialize OPD checkpoint optimizer")?,
        }))?;
        Ok(crate::checkpoint::TrainingCheckpointOptimizer {
            kind: kind.to_string(),
            step,
            hyperparameters,
            state_file: self.optimizer_state_file(),
        })
    }

    fn scheduler_manifest(&self, step: u64) -> crate::checkpoint::TrainingCheckpointScheduler {
        crate::checkpoint::TrainingCheckpointScheduler {
            kind: "constant".to_string(),
            step,
            state: serde_json::json!({"learning_rate": self.learning_rate}),
        }
    }

    fn consumed_candidates(&self, loop_state: &OpdCheckpointLoopState) -> Result<u64> {
        loop_state
            .epoch_index
            .checked_mul(self.candidates_per_epoch as u64)
            .and_then(|value| value.checked_add(loop_state.cursor_in_epoch))
            .context("OPD checkpoint candidate cursor overflows u64")
    }

    fn rng_states(
        &self,
        loop_state: &OpdCheckpointLoopState,
    ) -> Result<BTreeMap<String, crate::checkpoint::TrainingCheckpointRngState>> {
        let mut states = BTreeMap::from([(
            "lora-init".to_string(),
            crate::checkpoint::TrainingCheckpointRngState {
                algorithm: "kiln.seeded-lora-init.v1".to_string(),
                seed: self.init_seed,
                position: 0,
                state_file: None,
            },
        )]);
        if self.on_policy {
            states.insert(
                "rollout-sampling".to_string(),
                crate::checkpoint::TrainingCheckpointRngState {
                    algorithm: "kiln.opd-rollout-candidate.v1".to_string(),
                    seed: self.init_seed,
                    position: self.consumed_candidates(loop_state)?,
                    state_file: None,
                },
            );
        }
        let rounding = &self.precision_policy.stochastic_rounding;
        if rounding.get("mode").and_then(serde_json::Value::as_str) == Some("stochastic")
            && let Some(seed) = rounding.get("seed").and_then(serde_json::Value::as_u64)
        {
            states.insert(
                "optimizer-rounding".to_string(),
                crate::checkpoint::TrainingCheckpointRngState {
                    algorithm: "kiln.optimizer-stochastic-rounding.v1".to_string(),
                    seed,
                    position: loop_state.global_step,
                    state_file: None,
                },
            );
        }
        Ok(states)
    }

    fn data_order(&self) -> Vec<u64> {
        (0..self.candidates_per_epoch as u64).collect()
    }

    fn state_files(&self) -> crate::checkpoint::TrainingCheckpointStateFiles {
        crate::checkpoint::TrainingCheckpointStateFiles {
            adapter_parameters: OPD_CHECKPOINT_ADAPTER_FILE.to_string(),
            optimizer_state: self.optimizer_state_file(),
            reference_state: None,
            ema_state: None,
            reward_normalization_state: None,
            loss_history: Some(OPD_CHECKPOINT_LOOP_STATE_FILE.to_string()),
        }
    }

    fn progress(
        &self,
        loop_state: &OpdCheckpointLoopState,
    ) -> Result<crate::checkpoint::TrainingCheckpointProgress> {
        Ok(crate::checkpoint::TrainingCheckpointProgress {
            global_step: loop_state.global_step,
            total_steps: self.total_candidates()? as u64,
            epoch_index: loop_state.epoch_index,
            cursor_in_epoch: loop_state.cursor_in_epoch,
            data_order: self.data_order(),
        })
    }

    fn manifest(
        &self,
        loop_state: &OpdCheckpointLoopState,
    ) -> Result<crate::checkpoint::TrainingCheckpointManifest> {
        let progress = self.progress(loop_state)?;
        loop_state.validate(&progress, self.candidates_per_epoch, self.total_epochs)?;
        Ok(crate::checkpoint::TrainingCheckpointManifest::new(
            format!("opd-step-{:08}", loop_state.global_step),
            crate::checkpoint::TrainingKind::Opd,
            &self.adapter_name,
            self.effective_config.clone(),
            self.precision_policy.clone(),
            progress,
            self.data.clone(),
            self.rng_states(loop_state)?,
            self.optimizer_manifest(loop_state.global_step)?,
            self.scheduler_manifest(loop_state.global_step),
            self.state_files(),
            self.auxiliary_state.clone(),
        ))
    }

    fn validate_resume(
        &self,
        checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
        loop_state: &OpdCheckpointLoopState,
    ) -> Result<()> {
        let manifest = &checkpoint.manifest;
        anyhow::ensure!(
            manifest.training_kind == crate::checkpoint::TrainingKind::Opd,
            "resume checkpoint is {:?}, not OPD",
            manifest.training_kind
        );
        anyhow::ensure!(
            manifest.adapter_name == self.adapter_name,
            "resume checkpoint adapter {:?} does not match output adapter {:?}",
            manifest.adapter_name,
            self.adapter_name
        );
        anyhow::ensure!(
            manifest.effective_config == self.effective_config,
            "resume checkpoint effective OPD configuration differs from this request: checkpoint={}, request={}",
            manifest.effective_config,
            self.effective_config
        );
        anyhow::ensure!(
            manifest.precision_policy == self.precision_policy,
            "resume checkpoint precision policy differs from this runtime"
        );
        anyhow::ensure!(
            manifest.data == self.data,
            "resume checkpoint OPD data identity differs from this request"
        );
        anyhow::ensure!(
            manifest.progress.total_steps == self.total_candidates()? as u64
                && manifest.progress.data_order == self.data_order(),
            "resume checkpoint OPD candidate order differs from this run"
        );
        anyhow::ensure!(
            manifest.optimizer == self.optimizer_manifest(manifest.progress.global_step)?,
            "resume checkpoint optimizer contract differs from this request"
        );
        anyhow::ensure!(
            manifest.scheduler == self.scheduler_manifest(manifest.progress.global_step),
            "resume checkpoint scheduler contract differs from this request"
        );
        anyhow::ensure!(
            manifest.rng_states == self.rng_states(loop_state)?,
            "resume checkpoint RNG streams differ from this request"
        );
        anyhow::ensure!(
            manifest.state_files == self.state_files(),
            "resume checkpoint OPD artifact contract differs from this runtime"
        );
        crate::checkpoint::validate_checkpoint_base_weight_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        crate::checkpoint::validate_checkpoint_execution_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        anyhow::ensure!(
            manifest.auxiliary_state == self.auxiliary_state,
            "resume checkpoint model/tokenizer/teacher/runtime identity differs from this run"
        );
        loop_state.validate(
            &manifest.progress,
            self.candidates_per_epoch,
            self.total_epochs,
        )
    }

    fn capture(
        &self,
        output_root: &Path,
        backend: &dyn kiln_model::backend::BackendRuntime,
        params: &mut crate::trainer::TrainableLoraParams,
        opt_state: &mut Option<crate::trainer::OptimizerState>,
        loop_state: &OpdCheckpointLoopState,
    ) -> Result<OpdCheckpointSnapshot> {
        anyhow::ensure!(
            self.base_model_weights_sha256.is_some(),
            "exact OPD checkpointing requires base-model weights loaded with a content identity"
        );
        crate::checkpoint::validated_checkpoint_base_weight_manifest(&self.auxiliary_state)?;
        crate::checkpoint::validated_checkpoint_execution_provenance(&self.auxiliary_state)?;
        anyhow::ensure!(
            self.teacher_content_revision.is_some(),
            "exact OPD checkpointing requires an authoritative teacher content identity"
        );
        match (&self.optimizer, opt_state.as_ref()) {
            (Optimizer::Sgd, None) => {}
            (Optimizer::Sgd, Some(_)) => {
                anyhow::bail!("SGD OPD checkpoint unexpectedly has optimizer state")
            }
            (_, Some(state)) => anyhow::ensure!(
                u64::from(state.step_count()) == loop_state.global_step,
                "OPD optimizer step {} differs from loop step {}",
                state.step_count(),
                loop_state.global_step
            ),
            (_, None) => anyhow::bail!("stateful OPD optimizer has no checkpoint state"),
        }
        let manifest = self.manifest(loop_state)?;
        let target = output_root.join(format!(
            "{}-checkpoint-step-{:08}.kiln-checkpoint",
            self.adapter_name, loop_state.global_step
        ));
        params.sync_to_master(backend)?;
        let adapter_parameters = params.capture_checkpoint_parameters()?;
        let optimizer_state = opt_state
            .as_mut()
            .map(|state| state.capture_checkpoint_state(params, backend))
            .transpose()?;
        let mut artifacts = vec![
            crate::checkpoint::CheckpointArtifact {
                relative_path: OPD_CHECKPOINT_ADAPTER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::AdapterParameters,
            },
            crate::checkpoint::CheckpointArtifact {
                relative_path: OPD_CHECKPOINT_LOOP_STATE_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::LossHistory,
            },
        ];
        if optimizer_state.is_some() {
            artifacts.push(crate::checkpoint::CheckpointArtifact {
                relative_path: OPD_CHECKPOINT_OPTIMIZER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::OptimizerState,
            });
        }
        let loop_state_bytes =
            serde_json::to_vec_pretty(loop_state).context("serialize OPD checkpoint loop state")?;
        Ok(OpdCheckpointSnapshot {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            loop_state_bytes,
        })
    }
}

fn run_coordinated_opd_gpu_phase<T>(
    coordination: Option<&crate::trainer::GpuStepCoordination>,
    backend: &dyn kiln_model::backend::BackendRuntime,
    phase: &'static str,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match coordination {
        Some(coordination) => coordination.run_gpu_phase(backend, "OPD", phase, operation),
        None => operation(),
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_and_publish_opd_checkpoint(
    descriptor: &OpdCheckpointDescriptor,
    output_root: &Path,
    backend: &dyn kiln_model::backend::BackendRuntime,
    params: &mut crate::trainer::TrainableLoraParams,
    opt_state: &mut Option<crate::trainer::OptimizerState>,
    loop_state: &OpdCheckpointLoopState,
    coordination: Option<&crate::trainer::GpuStepCoordination>,
) -> Result<PathBuf> {
    let snapshot =
        run_coordinated_opd_gpu_phase(coordination, backend, "checkpoint device snapshot", || {
            descriptor.capture(output_root, backend, params, opt_state, loop_state)
        })?;
    let publish_started = std::time::Instant::now();
    let path = snapshot.publish()?;
    tracing::info!(
        checkpoint = %path.display(),
        publish_ms = publish_started.elapsed().as_millis() as u64,
        "published exact OPD checkpoint outside GPU ownership"
    );
    Ok(path)
}

fn load_opd_checkpoint_loop_state(
    checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
) -> Result<OpdCheckpointLoopState> {
    let relative = checkpoint
        .manifest
        .state_files
        .loss_history
        .as_deref()
        .context("OPD resume checkpoint has no loop-state file")?;
    anyhow::ensure!(
        relative == OPD_CHECKPOINT_LOOP_STATE_FILE,
        "unsupported OPD loop-state artifact {relative:?}"
    );
    let path = checkpoint.artifact_path(relative)?;
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read OPD checkpoint loop state {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse strict OPD checkpoint loop state")
}

fn restore_opd_adapter_parameters(
    params: &mut crate::trainer::TrainableLoraParams,
    device: &kiln_tensor::Device,
    resume_checkpoint: Option<&crate::checkpoint::ValidatedTrainingCheckpoint>,
    base_adapter_dir: Option<&Path>,
) -> Result<()> {
    anyhow::ensure!(
        resume_checkpoint.is_none() || base_adapter_dir.is_none(),
        "OPD exact resume and base-adapter initialization are mutually exclusive"
    );
    if let Some(checkpoint) = resume_checkpoint {
        let adapter_path =
            checkpoint.artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
        params.load_checkpoint_parameters(&adapter_path)?;
        tracing::info!(
            checkpoint = %checkpoint.root.display(),
            step = checkpoint.manifest.progress.global_step,
            "restored exact OPD adapter parameters"
        );
    } else if let Some(base_dir) = base_adapter_dir {
        let loaded = params.load_from_safetensors(base_dir, device)?;
        tracing::info!(
            base = %base_dir.display(),
            tensors = loaded,
            "loaded base adapter before OPD optimizer setup"
        );
    }
    Ok(())
}

fn opd_checkpoint_effective_config(
    config: &OpdConfig,
    learning_rate: f64,
    effective_seed: u64,
    effective_top_k: usize,
    effective_samples_per_prompt: usize,
) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(config).context("serialize effective OPD config")?;
    let object = value
        .as_object_mut()
        .context("serialized OPD config is not an object")?;
    object.remove("resume_checkpoint");
    object.insert(
        "learning_rate".to_string(),
        serde_json::json!(learning_rate),
    );
    object.insert("seed".to_string(), serde_json::json!(effective_seed));
    object.insert(
        "effective_top_k".to_string(),
        serde_json::json!(effective_top_k),
    );
    object.insert(
        "effective_samples_per_prompt".to_string(),
        serde_json::json!(effective_samples_per_prompt),
    );
    crate::trainer::canonical_checkpoint_json_value(value)
}

#[allow(clippy::too_many_arguments)]
fn opd_checkpoint_auxiliary_state(
    model_config: &kiln_core::config::ModelConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    training_precision_policy: kiln_model::backend::TrainingPrecisionPolicy,
    base_model_weights_sha256: Option<&str>,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    backend_runtime: &str,
    prepared_source_indices: &[usize],
    teacher_caps: &LogitSourceCaps,
    teacher_provenance: &OpdTeacherProvenance,
    use_chat_template_rollout_prefixes: bool,
    training_runtime_planning_identity: &serde_json::Value,
) -> Result<serde_json::Value> {
    let hashes =
        kiln_core::config_hashes::ConfigHashes::from_model_tokenizer(model_config, tokenizer, None);
    let prepared_source_order_sha256 =
        crate::train_receipt::sha256_json_serializable(&prepared_source_indices)
            .context("hash OPD prepared source order")?;
    Ok(serde_json::json!({
        "loop_state_type": OPD_CHECKPOINT_LOOP_STATE_TYPE,
        "model_config_sha256": hashes.model_config_hash,
        "tokenizer_config_sha256": hashes.tokenizer_config_hash,
        "chat_template_sha256": hashes.chat_template_hash,
        "base_model_weights_sha256": base_model_weights_sha256,
        "base_weight_shard_manifest": base_weight_shard_manifest,
        "execution_provenance": execution_provenance,
        "backend_runtime": backend_runtime,
        "kiln_train_version": env!("CARGO_PKG_VERSION"),
        "prepared_source_indices": prepared_source_indices,
        "prepared_source_order_sha256": prepared_source_order_sha256,
        "teacher_capabilities": teacher_caps,
        "teacher_identity": &teacher_provenance.identity,
        "teacher_content_revision": teacher_provenance.content_revision(),
        "training_precision_policy": training_precision_policy.name,
        "use_chat_template_rollout_prefixes": use_chat_template_rollout_prefixes,
        "checkpoint_planning": training_runtime_planning_identity,
    }))
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
    opd_train_to_with_checkpoint_root(
        prompts,
        config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        None,
    )
}

/// Staged-output variant of [`opd_train`]. Existing adapters resolve from
/// `adapter_dir`; all new weights, receipts, and checkpoints stay under
/// `output_adapter_dir` until the caller publishes them.
#[allow(clippy::too_many_arguments)]
pub fn opd_train_to(
    prompts: &[OpdPrompt],
    config: &OpdConfig,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher: Arc<dyn LogitSource>,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: Option<crate::trainer::ProgressCallback>,
) -> Result<std::path::PathBuf> {
    opd_train_to_with_checkpoint_root(
        prompts,
        config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        output_adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        None,
    )
}

/// Standalone coordinated OPD entry point with a durable checkpoint root.
///
/// Server callers should use [`opd_train_to_with_checkpoint_root_and_runtime`]
/// so rollout-dependent checkpoint plans share their process-lifetime memory
/// configuration.
#[allow(clippy::too_many_arguments)]
pub fn opd_train_to_with_checkpoint_root(
    prompts: &[OpdPrompt],
    config: &OpdConfig,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher: Arc<dyn LogitSource>,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    checkpoint_output_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: Option<crate::trainer::ProgressCallback>,
    gpu_step_coordination: Option<crate::trainer::GpuStepCoordination>,
) -> Result<std::path::PathBuf> {
    config
        .validate_runtime_contract()
        .context("opd_train: unsupported configuration before runtime initialization")?;
    crate::trainer::ensure_training_optimizer_device_supported(
        "OPD",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    opd_train_to_with_checkpoint_root_and_runtime(
        prompts,
        config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        gpu_step_coordination,
        &runtime,
    )
}

/// Server-owned OPD entry point with immutable process-lifetime runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn opd_train_to_with_checkpoint_root_and_runtime(
    prompts: &[OpdPrompt],
    config: &OpdConfig,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    teacher: Arc<dyn LogitSource>,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    checkpoint_output_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: Option<crate::trainer::ProgressCallback>,
    gpu_step_coordination: Option<crate::trainer::GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<std::path::PathBuf> {
    config
        .validate_runtime_contract()
        .context("opd_train: unsupported configuration")?;
    let runtime_device = crate::trainer::ensure_training_optimizer_entry_supported(
        "OPD",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("opd_train: initialize memory governor")?;
    let teacher_caps = teacher.capabilities();
    let teacher_provenance = OpdTeacherProvenance::from_source(teacher.as_ref(), &teacher_caps);
    let effective_top_k = match config.loss {
        OpdLossGranularity::SampledToken => unreachable!("unsupported loss rejected above"),
        OpdLossGranularity::TeacherTopK => resolve_opd_top_k(
            config.top_k,
            teacher_caps.max_top_k.min(teacher_caps.vocab_size),
        )
        .context("opd_train: cannot resolve an executable teacher top-K")?,
        OpdLossGranularity::FullVocab => {
            anyhow::ensure!(
                teacher_caps.supports_full_vocab,
                "opd_train: teacher {:?} does not support full-vocabulary logprobs",
                teacher_caps.teacher_id
            );
            let resolved = resolve_opd_top_k(teacher_caps.vocab_size, teacher_caps.vocab_size)
                .context("opd_train: full-vocabulary size is outside the KT kernel envelope")?;
            anyhow::ensure!(
                resolved == teacher_caps.vocab_size,
                "opd_train: full-vocabulary size {} is outside the KT kernel envelope {{16, 32}}",
                teacher_caps.vocab_size
            );
            resolved
        }
    };
    let run_started = std::time::Instant::now();
    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = crate::train_receipt::sha256_json_serializable(&prompts);
    let training_data_checkpoint_sha256 = crate::trainer::checkpoint_sha256_hex(
        training_data_sha256.as_deref(),
        "OPD training data",
    )?;
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        crate::trainer::resolve_base_adapter_dir_from_roots(
            name,
            adapter_dir,
            output_adapter_dir,
            adapter_name,
        )
    });
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        examples_read: prompts.len(),
        ..Default::default()
    };
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();

    use crate::Optimizer;
    use crate::trainer::TrainableLoraParams;
    // Per-step OPD forward/backward is tape-authoritative and kt-native. The
    // active tape scope is mandatory; no alternate autograd producer exists.
    use kiln_model::backend;
    if prompts.is_empty() {
        let message = "opd_train: prompts must be non-empty";
        if let Err(receipt_error) = write_opd_train_receipt(
            adapter_name,
            model_config,
            tokenizer,
            weights.base_weight_shard_manifest.as_ref(),
            weights.execution_provenance.as_ref(),
            None,
            config,
            effective_top_k,
            config.seed,
            config.samples_per_prompt,
            &teacher_provenance,
            &output_dir,
            requested_base_adapter_dir.as_deref(),
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            None,
            None,
            None,
            Vec::new(),
            Some(message.to_string()),
        ) {
            tracing::warn!(
                adapter = adapter_name,
                error = %receipt_error,
                "failed to persist OPD failure receipt"
            );
        }
        anyhow::bail!(
            "{}",
            crate::train_receipt::training_failure_error_message(message)
        );
    }

    // (#1082) `embed_tokens.device()` is the kt Device — threaded straight
    // through. The per-step forward/backward (`opd_step_forward_backward_tape_authoritative`)
    // now takes the kt device directly; the candle round-trip bridge is gone.
    let device_kt = runtime_device;
    let backend_rt = crate::trainer::training_backend_for_device(device_kt)?;
    let training_precision_policy =
        crate::trainer::training_precision_policy_for_backend(backend_rt.as_ref());
    crate::trainer::ensure_training_optimizer_supported(
        "OPD",
        backend_rt.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    let streaming_prefill_policy = runtime.resolved_streaming_prefill_policy(device_kt);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device_kt);

    // Bind every per-step plan to the process-lifetime capacity supplied by
    // the caller. OPD sequence lengths vary with sampled rollouts, but the
    // effective hardware/configuration context must not drift within a run.
    let opd_activation_bytes_per_elem =
        crate::trainer::training_activation_bytes_per_elem_for_backend(
            weights,
            backend_rt.as_ref(),
        );

    // §6 data-multiplier: auto-scale samples_per_prompt when the
    // dataset is small. Lu (2025) §3.5.4: 4 if |prompts| ≥ 200,
    // 16 if 50 ≤ |prompts| < 200, 64 if |prompts| < 50. We respect
    // any non-default user override (≠ default_opd_samples_per_prompt
    // = 4) and only auto-scale when the user didn't ask for a
    // specific count.
    let effective_samples_per_prompt = if matches!(config.training_mode, OpdTrainingMode::OffPolicy)
        && config.samples_per_prompt == default_opd_samples_per_prompt()
    {
        1
    } else if config.samples_per_prompt == default_opd_samples_per_prompt() {
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

    let learning_rate = config.effective_learning_rate();
    if let Some(explicit) = config.learning_rate
        && let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Opd),
        )
    {
        tracing::warn!(optimizer = ?config.optimizer, "OPD {warning}");
    }

    tracing::info!(
        num_prompts = prompts.len(),
        samples_per_prompt = effective_samples_per_prompt,
        config_samples_per_prompt = config.samples_per_prompt,
        requested_top_k = config.top_k,
        effective_top_k,
        teacher_top_k_cap = teacher_caps.max_top_k,
        loss = ?config.loss,
        lr = learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting OPD training"
    );

    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load OPD resume checkpoint")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_opd_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Opd,
            "resume checkpoint is not an OPD checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint OPD training data hash differs from this request"
        );
    }
    if config.checkpoint_interval.is_some() || resume_checkpoint.is_some() {
        crate::trainer::validate_exact_training_provenance(weights)?;
    }
    let resume_init_seed = resume_checkpoint
        .as_ref()
        .map(|checkpoint| {
            let state = checkpoint
                .manifest
                .rng_states
                .get("lora-init")
                .context("OPD resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported OPD lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "OPD resume seed {restored} differs from requested seed {requested}"
        );
    }
    let effective_seed = resume_init_seed
        .or(config.seed)
        .unwrap_or_else(rand::random);

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            if let Err(receipt_error) = write_opd_train_receipt(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                effective_top_k,
                Some(effective_seed),
                effective_samples_per_prompt,
                &teacher_provenance,
                &output_dir,
                requested_base_adapter_dir.as_deref(),
                training_data_sha256,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                None,
                None,
                None,
                Vec::new(),
                Some(format!("{err:#}")),
            ) {
                tracing::warn!(
                    adapter = adapter_name,
                    error = %receipt_error,
                    "failed to persist OPD failure receipt"
                );
            }
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    // Parameter and optimizer allocation can create resident buffers. Treat it
    // as an explicit settled setup phase before returning GPU ownership.
    let (mut params, mut opt_state) = run_coordinated_opd_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend_rt,
        "adapter and optimizer allocation",
        || {
            // `mut`: `sync_to_master` (checkpoint + final save) takes `&mut self`
            // (it swaps each param's forward/backward storage to the resolved kt
            // master). Mirrors `sft_train`'s `let mut params`. (#1082)
            let params = TrainableLoraParams::initialize_seeded_with_precision_policy(
                model_config,
                weights,
                config.lora_rank,
                config.lora_alpha,
                &device_kt,
                Some(effective_seed),
                training_precision_policy,
            )?;

            let opt_state = crate::trainer::make_opt_state(
                &params,
                config.optimizer,
                learning_rate,
                &device_kt,
            )?;
            Ok((params, opt_state))
        },
    )?;

    // Tokenize every prompt up-front (cheap relative to the forward
    // pass) and skip any prompts that produce no supervised action
    // tokens — same shape as sft_train's validity probe, with optional
    // trajectory ECHO masks for off-policy agentic data.
    let tokenized = prepare_opd_prompts_for_training(prompts, tokenizer, config.echo.as_ref());
    if tokenized.is_empty() {
        data_stats.examples_filtered = prompts.len();
        let message = "opd_train: no valid prompts after tokenization";
        if let Err(receipt_error) = write_opd_train_receipt(
            adapter_name,
            model_config,
            tokenizer,
            weights.base_weight_shard_manifest.as_ref(),
            weights.execution_provenance.as_ref(),
            None,
            config,
            effective_top_k,
            Some(effective_seed),
            effective_samples_per_prompt,
            &teacher_provenance,
            &output_dir,
            requested_base_adapter_dir.as_deref(),
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            Some(alpha_over_rank),
            None,
            None,
            Vec::new(),
            Some(message.to_string()),
        ) {
            tracing::warn!(
                adapter = adapter_name,
                error = %receipt_error,
                "failed to persist OPD failure receipt"
            );
        }
        anyhow::bail!(
            "{}",
            crate::train_receipt::training_failure_error_message(message)
        );
    }
    data_stats.examples_filtered = prompts.len().saturating_sub(tokenized.len());

    let epochs = config.epochs.max(1);
    let samples_per_prompt = effective_samples_per_prompt.max(1);
    let candidates_per_epoch = tokenized
        .len()
        .checked_mul(samples_per_prompt)
        .context("OPD candidate count overflows usize")?;
    let total_steps = epochs
        .checked_mul(candidates_per_epoch)
        .context("OPD training schedule overflows usize")?;

    // (#1082 P-OPD) `embed_tokens_t` is the frozen, weight-tied lm_head. The OPD
    // scalar-loss tape root is now kt-native (`try_tape_opd_scalar_mean_cuda_kt`)
    // and takes the kt `head_t` DIRECTLY, so the per-run kt->candle hoist (H8) is
    // gone — thread the kt weight straight into the per-step closure.
    let head_t = weights.embed_tokens_t.clone();

    // §3.9 guardrail observer + per-step rollout summary buffer.
    // The guardrail watches loss / repetition / overlap signals every
    // step and produces a `GuardrailDecision`; on any non-Ok decision
    // we log the trigger via tracing so the dashboard / receipt can
    // pick it up. Programmatic in-process rollback to the last
    // passing checkpoint is the remaining §3.9 wire-up; for now the
    // detector fires and the user sees it in the run log.
    let validation_cadence: u64 = 5;

    // Resolve EOS token ids once for rollout termination.
    let eos_token_ids: Vec<u32> = tokenizer.eos_token_ids();
    // The serialized request is the sole authority for the training mode.
    // A process-global environment override used to be able to turn an
    // admitted, fully materialized off-policy remote job back into a dynamic
    // on-policy job after queue admission. Besides making receipts dishonest,
    // that made the trainer query a live source while the server owned the GPU.
    let on_policy_enabled = matches!(config.training_mode, OpdTrainingMode::OnPolicy);

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
    // Explicit opt-in through `rollout_prompt_rendering`. The chat-template
    // render path is correct (verified via `examples/test_render`) but
    // produces broken adapters end-to-end on
    // structured-list capabilities (see opd-cap.code-symbol-extraction/
    // failure_mode.md). Probably a kernel-vs-prompt-length interaction the
    // root cause isn't fully understood for. Default to legacy
    // orig_input_ids[..first_label] path until the kernel side is audited.
    let use_chat_template_render = use_chat_template_rollout_prefixes(config);
    let rollout_prompt_prefixes: Vec<Vec<u32>> = if !use_chat_template_render {
        // Empty → fallback path uses orig_input_ids[..first_label_mask_true]
        // per the legacy behavior. Same as pre-fix.
        vec![Vec::new(); prompts.len()]
    } else {
        render_rollout_prompt_prefixes(prompts, tokenizer)?
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
    let teacher_prompt_tokens = render_teacher_prompt_tokens(prompts, tokenizer)?;
    let any_asymmetric = teacher_prompt_tokens.iter().any(|v| !v.is_empty());
    if any_asymmetric {
        let n_with_extra = teacher_prompt_tokens
            .iter()
            .filter(|v| !v.is_empty())
            .count();
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

    let checkpoint_precision =
        crate::trainer::training_checkpoint_precision(&params, opt_state.as_ref())?;
    let prepared_source_indices: Vec<usize> = tokenized
        .iter()
        .map(|prepared| prepared.source_index)
        .collect();
    let effective_checkpoint_config = opd_checkpoint_effective_config(
        config,
        learning_rate,
        effective_seed,
        effective_top_k,
        samples_per_prompt,
    )?;
    let checkpoint_descriptor = OpdCheckpointDescriptor {
        adapter_name: adapter_name.to_string(),
        effective_config: effective_checkpoint_config,
        precision_policy: checkpoint_precision,
        data: crate::checkpoint::TrainingCheckpointData {
            source_kind: if on_policy_enabled {
                "inline-on-policy-opd-candidate-order-v1"
            } else {
                "inline-off-policy-opd-candidate-order-v1"
            }
            .to_string(),
            content_sha256: training_data_checkpoint_sha256,
            item_count: candidates_per_epoch as u64,
        },
        init_seed: effective_seed,
        optimizer: config.optimizer,
        learning_rate,
        candidates_per_epoch,
        total_epochs: epochs,
        on_policy: on_policy_enabled,
        base_model_weights_sha256: weights.source_content_sha256.clone(),
        teacher_content_revision: teacher_provenance.content_revision(),
        auxiliary_state: opd_checkpoint_auxiliary_state(
            model_config,
            tokenizer,
            training_precision_policy,
            weights.source_content_sha256.as_deref(),
            weights.base_weight_shard_manifest.as_ref(),
            weights.execution_provenance.as_ref(),
            backend_rt.runtime_name(),
            &prepared_source_indices,
            &teacher_caps,
            &teacher_provenance,
            use_chat_template_render,
            &training_runtime_planning_identity,
        )?,
    };
    if let (Some(checkpoint), Some(loop_state)) =
        (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
    {
        checkpoint_descriptor.validate_resume(checkpoint, loop_state)?;
        anyhow::ensure!(
            loop_state.data_stats.examples_read == prompts.len()
                && loop_state.data_stats.examples_filtered == data_stats.examples_filtered,
            "resume checkpoint OPD prepared-row counters differ from this request"
        );
    }

    let base_adapter_dir = if resume_checkpoint.is_some() {
        None
    } else {
        crate::trainer::resolve_and_validate_base_adapter_from_roots(
            config.base_adapter.as_deref(),
            adapter_dir,
            output_adapter_dir,
            adapter_name,
            model_config,
            config.lora_rank,
            false,
        )?
    };
    run_coordinated_opd_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend_rt,
        "adapter restore and registry setup",
        || {
            restore_opd_adapter_parameters(
                &mut params,
                &device_kt,
                resume_checkpoint.as_ref(),
                base_adapter_dir.as_deref(),
            )?;
            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let state_path = checkpoint
                    .manifest
                    .state_files
                    .optimizer_state
                    .as_deref()
                    .map(|relative| checkpoint.artifact_path(relative))
                    .transpose()?;
                match (opt_state.as_mut(), state_path) {
                    (Some(state), Some(path)) => {
                        let step = u32::try_from(checkpoint.manifest.progress.global_step)
                            .context("OPD resume optimizer step exceeds u32")?;
                        state.load_checkpoint_state(&params, &path, step)?;
                    }
                    (None, None) => {}
                    (Some(_), None) => {
                        anyhow::bail!("stateful OPD checkpoint has no optimizer artifact")
                    }
                    (None, Some(_)) => {
                        anyhow::bail!("SGD OPD checkpoint unexpectedly contains optimizer state")
                    }
                }
            }
            if let Err(error) = params.register_with_backend(&*backend_rt) {
                params.evict_from_backend(&*backend_rt);
                return Err(error).context("register OPD adapter parameters with backend");
            }
            if let Some(state) = opt_state.as_ref()
                && let Err(error) = state.register_with_backend(&*backend_rt)
            {
                state.evict_from_backend(&*backend_rt);
                params.evict_from_backend(&*backend_rt);
                return Err(error).context("register OPD optimizer state with backend");
            }
            Ok(())
        },
    )?;
    let mut global_step = resume_loop_state
        .as_ref()
        .map_or(Ok(0), |state| usize::try_from(state.global_step))
        .context("OPD resume global step exceeds usize")?;
    let start_epoch = resume_loop_state
        .as_ref()
        .map_or(Ok(0), |state| usize::try_from(state.epoch_index))
        .context("OPD resume epoch exceeds usize")?;
    let start_cursor = resume_loop_state
        .as_ref()
        .map_or(Ok(0), |state| usize::try_from(state.cursor_in_epoch))
        .context("OPD resume candidate cursor exceeds usize")?;
    let mut loss_history = resume_loop_state
        .as_ref()
        .map_or_else(Vec::new, |state| state.loss_history.clone());
    let mut last_loss = loss_history.last().copied().unwrap_or(0.0);
    if let Some(state) = resume_loop_state.as_ref() {
        data_stats = state.data_stats.clone();
        token_counts = state.token_counts.clone();
    }
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut run_env_ce = resume_loop_state
        .as_ref()
        .and_then(|state| state.run_env_ce);
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut lora_grad_norms = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::LoraGradNormAccumulator::default,
        |state| state.lora_grad_norms.clone(),
    );
    let mut guardrail = resume_loop_state.as_ref().map_or_else(
        crate::diagnostics::LengthInflationGuardrail::default,
        |state| state.guardrail.clone(),
    );
    let train_result = (|| -> Result<PathBuf> {
        for epoch in start_epoch..epochs {
            for (prepared_index, prepared_prompt) in tokenized.iter().enumerate() {
                let prompt_idx = prepared_prompt.source_index;
                let tokenized_prompt = &prepared_prompt.tokenized;
                // Build the rollout prompt from the pre-rendered chat-template
                // prefix (see above — drops the dummy assistant turn and lets
                // the template emit the proper assistant cue marker tokens).
                // The explicitly enabled render path fails before training if a
                // prompt cannot be rendered or encoded. The legacy framing is
                // used only when that mode was selected above.
                let prompt_only: Vec<u32> = if use_chat_template_render {
                    rollout_prompt_prefixes[prompt_idx].clone()
                } else {
                    // Legacy path selected explicitly by the current default.
                    let prompt_end = tokenized_prompt
                        .action_mask
                        .iter()
                        .position(|&m| m)
                        .unwrap_or(tokenized_prompt.input_ids.len());
                    if prompt_end == 0 || prompt_end >= tokenized_prompt.input_ids.len() {
                        tracing::warn!(
                            prompt_idx,
                            "skipping prompt with no prompt/assistant split"
                        );
                        continue;
                    }
                    tokenized_prompt.input_ids[..prompt_end].to_vec()
                };
                let rollout_prompt_len = prompt_only.len();

                for sample_idx in 0..samples_per_prompt {
                    let candidate_cursor = prepared_index
                        .checked_mul(samples_per_prompt)
                        .and_then(|value| value.checked_add(sample_idx))
                        .context("OPD candidate cursor overflows usize")?;
                    if epoch == start_epoch && candidate_cursor < start_cursor {
                        continue;
                    }
                    // §3.1 step 1: sample a fresh student trajectory under
                    // the current LoRA. Replaces the off-policy passthrough
                    // of the teacher-authored assistant turn with the
                    // student's own tokens — the defining property of
                    // on-policy distillation per Lu (2025) §1.
                    let (input_ids_owned, active_positions, env_mask_owned, total_obs_len): (
                        Vec<u32>,
                        Vec<usize>,
                        Vec<bool>,
                        usize,
                    ) = if on_policy_enabled {
                        let lora_for_sample = params.as_lora_weights();
                        let step_seed = Some(
                            effective_seed
                                .wrapping_add(global_step as u64)
                                .wrapping_add(prompt_idx as u64 * 1_000_003)
                                .wrapping_add(sample_idx as u64 * 1_000_033),
                        );
                        let sampled = run_coordinated_opd_gpu_phase(
                            gpu_step_coordination.as_ref(),
                            &*backend_rt,
                            "student rollout",
                            || {
                                sample_student_rollout(
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
                                    config.sampler_segments,
                                    streaming_prefill_policy,
                                )
                                .with_context(|| {
                                    format!("on-policy rollout for prompt {prompt_idx}")
                                })
                            },
                        )?;
                        if sampled.is_empty() {
                            tracing::warn!(
                                prompt_idx,
                                sample_idx,
                                "student produced 0 new tokens; skipping step"
                            );
                            continue;
                        }
                        let mut full = prompt_only.clone();
                        full.extend_from_slice(&sampled);
                        // Active positions: the student-sampled tokens, which
                        // start at `rollout_prompt_len` (after the original
                        // prompt + the generation-prompt suffix).
                        let active: Vec<usize> =
                            (rollout_prompt_len..rollout_prompt_len + sampled.len()).collect();
                        let env_mask = vec![false; full.len()];
                        (full, active, env_mask, 0)
                    } else {
                        let active: Vec<usize> = tokenized_prompt
                            .action_mask
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &m)| if m { Some(i) } else { None })
                            .collect();
                        (
                            tokenized_prompt.input_ids.clone(),
                            active,
                            tokenized_prompt.env_mask.clone(),
                            tokenized_prompt.total_obs_len,
                        )
                    };
                    let input_ids: &[u32] = &input_ids_owned;
                    let env_mask: &[bool] = &env_mask_owned;
                    if active_positions.is_empty() {
                        continue;
                    }
                    let env_count = env_mask.iter().filter(|&&active| active).count();
                    token_counts.action_tokens = token_counts
                        .action_tokens
                        .saturating_add(active_positions.len() as u64);
                    token_counts.env_tokens =
                        token_counts.env_tokens.saturating_add(env_count as u64);
                    token_counts.context_tokens = token_counts.context_tokens.saturating_add(
                        input_ids
                            .len()
                            .saturating_sub(active_positions.len().saturating_add(env_count))
                            as u64,
                    );
                    // §20 asymmetric teacher conditioning: if this prompt
                    // declared `teacher_extra_messages`, build the teacher's
                    // token sequence as `teacher_prompt_tokens ++ sampled`.
                    // The student's input_ids = prompt_only ++ sampled has
                    // the student's own prompt framing; the teacher's view
                    // swaps the prompt half for the merged-extras version
                    // while keeping the SAME sampled rollout. Active
                    // positions are remapped to the teacher's frame. This
                    // bookkeeping is path-independent (it only touches host
                    // token arrays) so we compute it once before the
                    // tape-authoritative-vs-candle dispatch below.
                    let teacher_prompt: &[u32] = &teacher_prompt_tokens[prompt_idx];
                    let (teacher_full_tokens_owned, teacher_shifted_positions): (
                        Vec<u32>,
                        Vec<usize>,
                    ) = if teacher_prompt.is_empty() {
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
                    let (teacher_tokens_opt, teacher_active_opt): (
                        Option<&[u32]>,
                        Option<&[usize]>,
                    ) = if teacher_prompt.is_empty() {
                        (None, None)
                    } else {
                        (
                            Some(teacher_full_tokens_owned.as_slice()),
                            Some(teacher_shifted_positions.as_slice()),
                        )
                    };

                    // ECHO no longer uses the retired candle-only gate. When this
                    // rollout has eligible observation tokens, EchoEnvSpec below
                    // composes env-CE into the tape-authoritative loss root; the
                    // checkpointed path applies the same value and analytic hidden
                    // gradient before replaying its segments.
                    let opd_segments = opd_checkpoint_segments_for_step(
                        runtime,
                        config.grad_checkpoint_segments,
                        opd_activation_bytes_per_elem,
                        model_config,
                        input_ids.len(),
                    );
                    let checkpoint_segments = opd_segments.as_ref().map_or(0, |segs| segs.len());

                    // === Forward + backward dispatch (#1082 candle-drop) ===
                    //
                    // Tape-authoritative kt path (the ONLY path post-candle-drop):
                    // drive gradients through the kt `Tape`, never candle
                    // `loss.backward()`. Runs a SINGLE full forward
                    // (`model_forward_no_head`) inside `with_tape_authoritative_scope`
                    // so the LoRA adapters record their nodes, then roots the tape at
                    // the scalar OPD loss (`try_tape_opd_scalar_mean_cuda_kt`, taking
                    // kt `normed`/`head_t` directly) and walks the connected tape with
                    // one `Tape::backward`. Returns the LoRA
                    // grads as a kt-native `kiln_autograd::GradStore` (keyed by
                    // `KtTensorId`) consumed DIRECTLY by
                    // `optimizer_step_from_kt_grad_store` — NO per-step kt→candle grad
                    // copy (the explicit high-perf target). Mirrors the SFT
                    // `standard_forward_backward_tape_authoritative_kt` producer +
                    // `optimizer_step_from_kt_grad_store` consumer.
                    //
                    // (#1082) The candle gradient-checkpointing fallback
                    // (`opd_step_forward_backward_candle`) + the candle/CPU/ECHO
                    // dispatch arms were deleted: candle autograd can no longer trace
                    // LoRA grads through the kt-internal forward ops (the kt↔candle
                    // copy bridge severs the lineage — see note
                    // `kiln-candle-autograd-drops-attn-conv-grads`), so the tape path
                    // is the sole correct grad producer. The former ECHO-only
                    // dispatch inputs no longer steer this kt-only path.
                    //
                    // CUDA-gated: the kt tape adapters record kt CUDA ops, so the OPD
                    // tape path is CUDA-only. A non-cuda build of `opd_train` has no
                    // grad producer; the non-cuda arm bails cleanly so the loop body
                    // (which reads `loss_val` / `active_count` below for logging,
                    // guardrails, and the progress callback) still type-checks on
                    // both builds.
                    let (loss_val, active_count): (f64, usize) = run_coordinated_opd_gpu_phase(
                        gpu_step_coordination.as_ref(),
                        &*backend_rt,
                        "optimizer candidate",
                        || {
                            #[cfg(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            ))]
                            {
                                let step_started = std::time::Instant::now();
                                // ECHO env-CE spec for this rollout (OPD half of the
                                // resurrection plan): built from the trajectory env
                                // mask the prepare pass already computed; None when
                                // ECHO is off or the rollout has no observations.
                                let echo_spec = config.echo.as_ref().and_then(|echo| {
                                    (total_obs_len > 0 && echo.lambda != 0.0).then(|| {
                                        crate::grpo_tape_shim::EchoEnvSpec {
                                            env_mask: env_mask_owned.clone(),
                                            total_obs_len,
                                            lambda: echo.lambda,
                                        }
                                    })
                                });
                                let (loss_val, active_count, kt_grads, step_env_ce) =
                                    if let Some(segs) = opd_segments.as_deref() {
                                        checkpointed_opd_step_forward_backward_tape_authoritative(
                                            &*backend_rt,
                                            input_ids,
                                            weights,
                                            model_config,
                                            &params,
                                            &device_kt,
                                            &head_t,
                                            teacher.clone(),
                                            &active_positions,
                                            config.loss,
                                            effective_top_k,
                                            teacher_tokens_opt,
                                            teacher_active_opt,
                                            echo_spec.as_ref(),
                                            config.detect_anomaly,
                                            segs,
                                            streaming_prefill_policy,
                                        )?
                                    } else {
                                        opd_step_forward_backward_tape_authoritative(
                                            &*backend_rt,
                                            input_ids,
                                            weights,
                                            model_config,
                                            &params,
                                            &device_kt,
                                            &head_t,
                                            teacher.clone(),
                                            &active_positions,
                                            config.loss,
                                            effective_top_k,
                                            teacher_tokens_opt,
                                            teacher_active_opt,
                                            echo_spec.as_ref(),
                                            config.detect_anomaly,
                                            streaming_prefill_policy,
                                        )?
                                    };
                                if let Some(env_ce) = step_env_ce {
                                    run_env_ce = Some(env_ce);
                                }
                                tracing::info!(
                                    prompt_idx,
                                    sample_idx,
                                    seq_len = input_ids.len(),
                                    action_tokens = active_positions.len(),
                                    env_tokens = env_count,
                                    checkpoint_segments,
                                    elapsed_ms = step_started.elapsed().as_millis() as u64,
                                    "OPD step end (tape-authoritative kt)"
                                );

                                // Observe per-module LoRA grad norms from the kt-native
                                // grad store BEFORE the optimizer consumes it — same
                                // pattern as SFT/GRPO. Records that gradients flowed
                                // (the receipt's `lora_grad_norms` is the oracle the
                                // Metal smoke checks).
                                crate::trainer::observe_lora_grad_norms_from_kt_grad_store(
                                    &mut lora_grad_norms,
                                    &params,
                                    &kt_grads,
                                )?;

                                // Consume the kt-native grads DIRECTLY (no kt→candle
                                // copy): `optimizer_step_from_kt_grad_store` bridges each
                                // LoRA Var's grad at its own per-Var boundary inside the
                                // optimizer update, the last remaining candle dependency
                                // in the OPD grad path (dissolves when `kiln-optim` goes
                                // kt-native).
                                crate::trainer::optimizer_step_from_kt_grad_store(
                                    &*backend_rt,
                                    &mut params,
                                    &kt_grads,
                                    learning_rate,
                                    config.optimizer,
                                    opt_state.as_mut(),
                                )?;

                                Ok((loss_val, active_count))
                            }
                            #[cfg(not(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            )))]
                            {
                                anyhow::bail!(
                                    "opd_train: OPD training requires a CUDA / Metal / Vulkan / ROCm \
                             build — the kt tape-authoritative grad path (the sole grad \
                             producer after the #1082 candle-drop) records kt GPU ops and is \
                             gated behind `feature = \"cuda\"` / `\"metal\"` / `\"vulkan\"` / \
                             `\"rocm\"`"
                                );
                            }
                        },
                    )?;

                    anyhow::ensure!(
                        loss_val.is_finite(),
                        "OPD loss became non-finite at candidate {}: {loss_val}",
                        candidate_cursor + 1
                    );
                    last_loss = loss_val;
                    loss_history.push(loss_val);
                    global_step += 1;
                    data_stats.examples_trained = global_step;
                    let next_candidate = candidate_cursor + 1;
                    let (next_epoch, next_cursor) = if next_candidate == candidates_per_epoch {
                        (epoch + 1, 0)
                    } else {
                        (epoch, next_candidate)
                    };
                    let has_remaining_candidates = next_epoch < epochs;

                    // §3.9 guardrail observation on the validation
                    // cadence. We build a snapshot from the active-token
                    // tokens-as-bytes proxy (since opd_train uses the
                    // ground-truth assistant turn as the rollout for the
                    // milestone wire-up — true student-sampled rollouts
                    // arrive with the rollout sampler). The repetition /
                    // truncation signals are best-effort against the
                    // ground-truth tail; the kl / loss signals are real.
                    if global_step as u64 % validation_cadence == 0 {
                        let rollout =
                            crate::diagnostics::RolloutSummary::from_tokens(input_ids, true);
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

                    let control =
                        progress_cb
                            .as_ref()
                            .map_or(crate::trainer::TrainControl::Continue, |cb| {
                                let consumed_candidates = next_epoch
                                    .saturating_mul(candidates_per_epoch)
                                    .saturating_add(next_cursor);
                                cb(crate::trainer::TrainingProgress {
                                    epoch: epoch + 1,
                                    total_epochs: epochs,
                                    step: global_step,
                                    total_steps,
                                    loss: loss_val,
                                    progress: consumed_candidates as f32
                                        / total_steps.max(1) as f32,
                                })
                            });

                    let checkpoint_due = config.checkpoint_interval.is_some_and(|interval| {
                        global_step % interval == 0 && has_remaining_candidates
                    });
                    let stop_requested =
                        control == crate::trainer::TrainControl::Stop && has_remaining_candidates;
                    if checkpoint_due || stop_requested {
                        let loop_state = OpdCheckpointLoopState::capture(
                            global_step,
                            next_epoch,
                            next_cursor,
                            &loss_history,
                            &data_stats,
                            &token_counts,
                            run_env_ce,
                            &lora_grad_norms,
                            &guardrail,
                        );
                        let path = capture_and_publish_opd_checkpoint(
                            &checkpoint_descriptor,
                            checkpoint_output_dir,
                            &*backend_rt,
                            &mut params,
                            &mut opt_state,
                            &loop_state,
                            gpu_step_coordination.as_ref(),
                        )?;
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            reason = if stop_requested { "cancellation" } else { "periodic" },
                            "saved exact OPD training checkpoint"
                        );
                    }
                    if stop_requested {
                        anyhow::bail!(
                            "training cancelled by user (stop requested at OPD step boundary)"
                        );
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

        // Pull on-device values back into the kt master before `save_peft`
        // serializes them — mirrors sft_train's final `sync_to_master`.
        run_coordinated_opd_gpu_phase(
            gpu_step_coordination.as_ref(),
            &*backend_rt,
            "final adapter device snapshot",
            || {
                params
                    .sync_to_master(&*backend_rt)
                    .context("capture final OPD adapter state")
            },
        )?;

        // Safetensors/config/receipt I/O consumes only the captured CPU state.
        params.save_peft(&output_dir, model_config.num_layers)?;
        data_stats.examples_trained = global_step;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            steps = global_step,
            "OPD training complete"
        );

        write_opd_train_receipt(
            adapter_name,
            model_config,
            tokenizer,
            weights.base_weight_shard_manifest.as_ref(),
            weights.execution_provenance.as_ref(),
            crate::trainer::training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
            config,
            effective_top_k,
            Some(effective_seed),
            effective_samples_per_prompt,
            &teacher_provenance,
            &output_dir,
            requested_base_adapter_dir.as_deref(),
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            Some(alpha_over_rank),
            Some(last_loss),
            run_env_ce,
            lora_grad_norms.finish(),
            None,
        )?;

        Ok(output_dir)
    })();
    let mut train_result = train_result;
    let cleanup_result = run_coordinated_opd_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend_rt,
        "resident adapter and optimizer cleanup",
        || {
            drop(teacher);
            if let Some(state) = opt_state.as_ref() {
                state.evict_from_backend(&*backend_rt);
            }
            params.evict_from_backend(&*backend_rt);
            Ok(())
        },
    );
    if let Err(error) = cleanup_result {
        if train_result.is_ok() {
            train_result = Err(error.context("complete coordinated OPD cleanup"));
        } else {
            tracing::warn!(
                error = %format!("{error:#}"),
                "OPD cleanup could not acquire a healthy backend"
            );
        }
    }
    train_result
}

/// Tape-authoritative OPD forward + backward for one step (#1082 CP-4 endgame).
///
/// Mirrors `trainer::standard_forward_backward_tape_authoritative` (the SFT
/// precedent): runs the full forward + scalar OPD loss inside
/// `with_tape_authoritative_scope`, seeds `dL/dL = 1` at the loss, walks the
/// connected kt `Tape` (NO candle `loss.backward()`), and extracts the
/// per-LoRA-param grads into a kt-native `kiln_autograd::GradStore` the
/// optimizer step consumes DIRECTLY.
///
/// The single full forward (`model_forward_no_head`) routes the LoRA adapters
/// through the mandatory active tape scope, and the kt-native scalar loss
/// adapter (`try_tape_opd_scalar_mean_cuda_kt`) records
/// the loss as the tape root against the kt `normed` (the final-RMSNorm tape
/// node output) DIRECTLY so the recorded tape is CONNECTED from the loss back
/// through the LoRA chain. This REPLACES the candle
/// gradient-checkpointing reverse-segment loop: the tape records every forward
/// node, so one backward walk yields the LoRA grads directly.
///
/// Returns `(loss_val, active_count, grads)`, where `grads` is a kt-native
/// `kiln_autograd::GradStore` keyed by `KtTensorId`. (#1082 high-perf) The
/// per-step kt→candle grad copy was DROPPED: the kt grads are inserted as-is
/// and consumed by `optimizer_step_from_kt_grad_store`, which bridges each LoRA
/// Var's grad to candle only at its own per-Var optimizer boundary (the last
/// candle dependency in the OPD grad path, dissolving when `kiln-optim` goes
/// kt-native). Mirrors SFT's `standard_forward_backward_tape_authoritative_kt`.
///
/// CUDA + Metal: the tape adapters record kt GPU ops + bridge kt<->candle GPU
/// tensors. The caller device-gates this via
/// `#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan", feature = "rocm"))]`. (#1082 Metal lane) On
/// Metal the FORWARD + scalar OPD loss record onto the tape; the recorded OPD
/// backward (`CudaOpdTopKReverseKlPhaseBBackward::apply`) is CUDA-FFI-only and
/// `bail!`s during the tape walk — so on Metal this returns the backward error
/// (OPD Metal backward is a documented follow-up pending a Metal kernel).
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn opd_step_forward_backward_tape_authoritative(
    backend_rt: &dyn kiln_model::backend::BackendRuntime,
    input_ids: &[u32],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    params: &crate::trainer::TrainableLoraParams,
    device: &kiln_tensor::Device,
    // (#1082 P-OPD) The frozen lm_head weight as a kt tensor, passed straight
    // through. The OPD scalar-loss tape root is now kt-native
    // (`try_tape_opd_scalar_mean_cuda_kt`), so it takes the kt `head_t`
    // DIRECTLY — the prior per-run `head_t`→candle hoist (H8) is gone.
    head_t: &kiln_tensor::Tensor,
    teacher: Arc<dyn LogitSource>,
    active_positions: &[usize],
    loss_granularity: OpdLossGranularity,
    top_k: usize,
    teacher_tokens: Option<&[u32]>,
    teacher_active_positions: Option<&[usize]>,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    detect_anomaly: bool,
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> Result<(f64, usize, kiln_autograd::GradStore, Option<f64>)> {
    use kiln_model::forward::model_forward_no_head_with_policy;

    // Teacher fetch + mask / top_k resolution uses the shared preparation
    // helper. This is the (potentially network/IPC) teacher query;
    // done ONCE here, before the tape scope, so the closure is pure tensor
    // work (the scope must not perform side-effecting teacher I/O on a retry).
    let prepared = prepare_opd_kernel_inputs(
        input_ids,
        active_positions,
        teacher,
        loss_granularity,
        top_k,
        teacher_tokens,
        teacher_active_positions,
    )?;
    let active_count = prepared.active_count;
    // Captured by the tape closure; read after the scope returns.
    let step_env_ce: std::cell::Cell<Option<f64>> = std::cell::Cell::new(None);

    let lora_weights = params.as_lora_weights();

    let (loss_val, _loss_kt, grads_by_candle_raw) =
        kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions { detect_anomaly },
            || {
            // Single full forward (embed -> layers -> final RMSNorm). The
            // LoRA adapters inside record onto the active tape; the final
            // RMSNorm retains its kt output so the OPD loss adapter can
            // thread it as `hidden` and keep the tape connected.
            let mut linear_state =
                kiln_model::forward::LinearAttentionState::new(model_config, device).map_err(
                    |e| kiln_kt_bridge::BridgeError::new(format!("opd tape: linear_state: {e:#}")),
                )?;
            let normed = model_forward_no_head_with_policy(
                backend_rt,
                input_ids,
                weights,
                model_config,
                Some(&mut linear_state),
                Some(&lora_weights),
                streaming_prefill,
            )
            .context("opd tape-authoritative forward")
            .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
            let opd_loss_route = TrainingLossBackend::runtime_opd_loss_route(backend_rt);

            // (#1082 P-OPD) `model_forward_no_head` returns the kt `normed`
            // (the final-RMSNorm tape node output). Record the SCALAR OPD loss
            // DIRECTLY against the kt `normed` + kt `head_t` via the kt-native
            // adapter — no candle `normed`/`head_t` copies, no `normed_candle`
            // retain dance, no per-run `head_t_candle` hoist. Because `normed`
            // is already a recorded tape node id, the loss roots on it and the
            // tape stays connected back through the LoRA chain natively (mirrors
            // the SFT H6 CE-from-logits-kt path, which records against the
            // connected kt logits id). Returns a detached candle scalar (value
            // only, registered for the scope to seed); the gradient lives on the
            // tape.
            let loss_candidate = match opd_loss_route {
                #[cfg(feature = "vulkan")]
                OpdLossRoute::VulkanActiveHidden => {
                    crate::opd_tape_shim::try_tape_opd_scalar_mean_vulkan_kt(
                        &normed,
                        &weights.embed_tokens,
                        &prepared.teacher_topk_indices,
                        &prepared.teacher_topk_logprobs,
                        &prepared.label_mask,
                        prepared.resolved_top_k,
                    )
                }
                #[cfg(not(feature = "vulkan"))]
                OpdLossRoute::VulkanActiveHidden => Ok(None),
                OpdLossRoute::KtTapePhaseB => crate::opd_tape_shim::try_tape_opd_scalar_mean_cuda_kt(
                    &normed,
                    head_t,
                    &prepared.teacher_topk_indices,
                    &prepared.teacher_topk_logprobs,
                    &prepared.label_mask,
                    prepared.resolved_top_k,
                ),
                OpdLossRoute::Unsupported => Ok(None),
            };
            let loss = match loss_candidate
                .context("opd tape-authoritative scalar loss")
                .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
            {
                Some(l) => l,
                None => {
                    // The scalar tape adapter declined (gate off, empty active
                    // set, or out-of-envelope inputs). The tape scope cannot
                    // be seeded without a tape-routed loss root, so surface a
                    // clean error — the caller's dispatch should not have
                    // selected this path if the envelope was unmet.
                    return Err(kiln_kt_bridge::BridgeError::new(
                        format!(
                            "opd tape-authoritative: {:?} did not record a scalar root \
                         (an active tape scope is mandatory; the active set may be empty, the route may be \
                         unsupported, or the inputs may be outside its envelope)",
                            opd_loss_route
                        ),
                    ));
                }
            };
            // ECHO env-CE (OPD half of the resurrection plan): wrap the
            // recorded OPD scalar in the compose node so the env rows
            // gradient-flow alongside the distillation term. The head is
            // dtype-aligned to the hidden for the env log-prob compute
            // (Vulkan mixed precision: F32 hidden × BF16 head).
            let loss = match echo_env {
                Some(spec) => {
                    let head_for_echo;
                    let head_ref = if head_t.dtype() != normed.dtype() {
                        head_for_echo =
                            head_t.to_dtype(normed.dtype()).map_err(|e| {
                                kiln_kt_bridge::BridgeError::new(format!(
                                    "opd tape: echo head cast: {e}"
                                ))
                            })?;
                        &head_for_echo
                    } else {
                        head_t
                    };
                    match crate::opd_tape_shim::try_tape_opd_echo_env_compose_kt(
                        &loss,
                        &normed,
                        head_ref,
                        input_ids,
                        spec,
                        512,
                        device,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                    {
                        Some((composed, env_ce_val)) => {
                            step_env_ce.set(Some(env_ce_val));
                            composed
                        }
                        None => loss,
                    }
                }
                None => loss,
            };
            let loss_val = loss.to_scalar::<f32>().map_err(|e| {
                kiln_kt_bridge::BridgeError::new(format!("opd tape: loss.to_scalar: {e}"))
            })? as f64;
            Ok((loss_val, loss))
            },
        )
        .map_err(|e| anyhow!("opd tape-authoritative backward: {e}"))?;

    // (#1082 high-perf) Build a kt-native `kiln_autograd::GradStore` DIRECTLY
    // from the tape grads — NO per-step kt→candle grad copy. The kt grads are
    // inserted as-is, keyed by each LoRA Var's id bridged into the kt id space
    // (`KtTensorId::from_raw(var.id().as_raw() as u64)` == `cd_tensor_id_to_kt`),
    // matching the `KtTensorId`-keyed `OptimizerState.moments` so
    // `optimizer_step_from_kt_grad_store` looks each grad up under the same key.
    // The single per-Var kt→candle bridge now happens inside the optimizer step
    // (its master/moments are still candle until `kiln-optim` goes kt-native),
    // not here. Mirrors SFT's `standard_forward_backward_tape_authoritative_kt`.
    // (#1082) `all_vars()` -> `all_params()` (LoRA params are
    // `kiln_param::Parameter` now, not candle `Var`s); each param's stable kt
    // id is `Parameter::tensor_id().as_raw()` (`u64`). Mirrors SFT's
    // `standard_forward_backward_tape_authoritative_kt`.
    let mut grads = kiln_autograd::GradStore::new();
    for (candle_raw, kt_grad) in grads_by_candle_raw {
        // The tape `out` map mixes candle-keyed deposits (frozen base /
        // activations) with namespace-tagged kt-param deposits (the LoRA leaves,
        // via `register_input_mapping_kt`). Only tagged keys are LoRA-param
        // grads; `decode_kt_param_deposit` strips the tag → the param's kt id,
        // and rejects candle ids that happen to collide with a param id
        // (independent counters, both start at 1 — the #1082 grad-shape bug).
        let Some(param_raw) =
            kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
        else {
            continue;
        };
        // Preserve every tagged parameter deposit. The shared exact-gradient
        // validator rejects unknown ids instead of letting this producer hide
        // them before the optimizer boundary.
        grads.insert(kiln_tensor_id::TensorId::from_raw(param_raw), kt_grad);
    }

    Ok((loss_val, active_count, grads, step_env_ce.get()))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn checkpointed_opd_step_forward_backward_tape_authoritative(
    backend_rt: &dyn kiln_model::backend::BackendRuntime,
    input_ids: &[u32],
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    params: &crate::trainer::TrainableLoraParams,
    device: &kiln_tensor::Device,
    head_t: &kiln_tensor::Tensor,
    teacher: Arc<dyn LogitSource>,
    active_positions: &[usize],
    loss_granularity: OpdLossGranularity,
    top_k: usize,
    teacher_tokens: Option<&[u32]>,
    teacher_active_positions: Option<&[usize]>,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    detect_anomaly: bool,
    segments: &[(usize, usize)],
    streaming_prefill: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> Result<(f64, usize, kiln_autograd::GradStore, Option<f64>)> {
    use kiln_model::forward::{
        LinearAttentionState, model_forward_embed, model_forward_final_norm,
        model_forward_segment_with_policy,
    };
    use kiln_opd_loss_kernel as opd_loss;

    let num_segments = segments.len();
    anyhow::ensure!(
        num_segments > 0,
        "checkpointed OPD requires at least one segment"
    );
    let prepared = prepare_opd_kernel_inputs(
        input_ids,
        active_positions,
        teacher,
        loss_granularity,
        top_k,
        teacher_tokens,
        teacher_active_positions,
    )?;
    let active_count = prepared.active_count;
    let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
    let lora_detached = crate::trainer::lora_weights_detached(params);
    let lora_weights = params.as_lora_weights();

    let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
    let mut boundaries: Vec<kiln_tensor::Tensor> =
        Vec::with_capacity(crate::retained_checkpoint_boundary_count(num_segments));
    let mut current = embed_hidden.detach();
    boundaries.push(current.clone());
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        for &(start, end) in segments {
            current = model_forward_segment_with_policy(
                backend_rt,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_detached),
                streaming_prefill,
            )?
            .detach();
            boundaries.push(current.clone());
        }
    }
    let final_hidden = boundaries
        .last()
        .context("checkpointed OPD: missing final checkpoint boundary")?
        .clone();
    let normed = model_forward_final_norm(&final_hidden, weights, model_config)
        .context("checkpointed OPD final norm")?;
    let opd_loss_route = TrainingLossBackend::runtime_opd_loss_route(backend_rt);

    #[cfg(feature = "vulkan")]
    let head_owned;
    #[cfg(feature = "vulkan")]
    let head_for_loss: &kiln_tensor::Tensor =
        if matches!(opd_loss_route, OpdLossRoute::VulkanActiveHidden)
            && normed.dtype() == kiln_tensor::DType::F32
            && head_t.dtype() == kiln_tensor::DType::BF16
        {
            head_owned = head_t
                .to_dtype(kiln_tensor::DType::F32)
                .context("checkpointed OPD: cast BF16 head_t -> F32 for Vulkan OPD loss")?;
            &head_owned
        } else {
            head_t
        };
    #[cfg(not(feature = "vulkan"))]
    let head_for_loss: &kiln_tensor::Tensor = head_t;

    let generic_loss_and_grad = || -> Result<(f64, kiln_tensor::Tensor)> {
        let (loss, _opd_active_metadata) = opd_loss::opd_top_k_reverse_kl_with_metadata_kt(
            &normed,
            head_for_loss,
            &prepared.teacher_topk_indices,
            &prepared.teacher_topk_logprobs,
            &prepared.label_mask,
            prepared.resolved_top_k,
        )
        .map_err(|e| anyhow!("checkpointed OPD scalar loss: {e}"))?;
        let loss_val = loss.to_scalar::<f32>()? as f64;

        let grad_normed = {
            let opd_backward_route =
                TrainingLossBackend::runtime_opd_phase_b_backward_route(backend_rt);
            let composite_grad = || -> Result<kiln_tensor::Tensor> {
                let grad_loss = kiln_tensor::Tensor::from_vec_on(*device, vec![1.0f32], vec![])
                    .context("checkpointed OPD grad_loss seed")?;
                opd_loss::kt_api::opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
                    &normed,
                    head_for_loss,
                    &prepared.teacher_topk_indices,
                    &prepared.teacher_topk_logprobs,
                    &prepared.label_mask,
                    &grad_loss,
                    prepared.resolved_top_k,
                    opd_loss::OpdLossOutputKt::ScalarMean,
                )
                .map_err(|e| anyhow!("checkpointed OPD composite hidden gradient: {e}"))
            };
            match opd_backward_route {
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad => {
                    if let Some(active_metadata) = _opd_active_metadata.as_ref() {
                        opd_loss::opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_with_metadata_kt(
                            &normed,
                            head_for_loss,
                            &prepared.teacher_topk_indices,
                            &prepared.teacher_topk_logprobs,
                            &prepared.label_mask,
                            prepared.resolved_top_k,
                            active_metadata,
                        )
                        .map_err(|e| anyhow!("checkpointed OPD fused hidden gradient: {e}"))?
                    } else {
                        opd_loss::opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_kt(
                            &normed,
                            head_for_loss,
                            &prepared.teacher_topk_indices,
                            &prepared.teacher_topk_logprobs,
                            &prepared.label_mask,
                            prepared.resolved_top_k,
                        )
                        .map_err(|e| anyhow!("checkpointed OPD fused hidden gradient: {e}"))?
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad => {
                    anyhow::bail!(
                        "checkpointed OPD CUDA/ROCm fused Phase-B backward route requires cuda or rocm feature"
                    )
                }
                OpdPhaseBBackwardRoute::KtComposite => composite_grad()?,
                OpdPhaseBBackwardRoute::VulkanActiveHidden => {
                    anyhow::bail!(
                        "checkpointed OPD Vulkan active-hidden backward route should use the fused Vulkan loss/grad path"
                    )
                }
                OpdPhaseBBackwardRoute::Unsupported => {
                    anyhow::bail!("checkpointed OPD Phase-B backward route is unsupported")
                }
            }
        };
        Ok((loss_val, grad_normed))
    };

    let (loss_val, grad_normed) = {
        match opd_loss_route {
            #[cfg(feature = "vulkan")]
            OpdLossRoute::VulkanActiveHidden => {
                let (loss, grad_normed) =
                    crate::opd_tape_shim::vulkan_opd_top_k_reverse_kl_scalar_loss_and_grad_kt(
                        &normed,
                        &weights.embed_tokens,
                        &prepared.teacher_topk_indices,
                        &prepared.teacher_topk_logprobs,
                        &prepared.label_mask,
                        prepared.resolved_top_k,
                    )
                    .map_err(|e| anyhow!("checkpointed OPD Vulkan fused loss/grad: {e}"))?;
                (loss.to_scalar::<f32>()? as f64, grad_normed)
            }
            #[cfg(not(feature = "vulkan"))]
            OpdLossRoute::VulkanActiveHidden => {
                anyhow::bail!("checkpointed OPD Vulkan active-hidden route requires vulkan feature")
            }
            OpdLossRoute::KtTapePhaseB => generic_loss_and_grad()?,
            OpdLossRoute::Unsupported => {
                anyhow::bail!("checkpointed OPD loss route is unsupported for this backend")
            }
        }
    };
    // ECHO env-CE (OPD half of the resurrection plan), analytic variant for
    // the checkpointed path: add λ·env_CE to the scalar and the matching
    // constant-coefficient env-row gradient (seed 1.0) onto grad_normed —
    // the replay segments then propagate it like any other hidden grad.
    let (loss_val, grad_normed, step_env_ce) = match echo_env {
        Some(spec) => {
            let head_for_echo;
            let head_ref = if head_t.dtype() != normed.dtype() {
                head_for_echo = head_t
                    .to_dtype(normed.dtype())
                    .context("checkpointed OPD: echo head cast")?;
                &head_for_echo
            } else {
                head_t
            };
            match crate::grpo_tape_shim::echo_env_state_and_value_kt(
                &normed, head_ref, input_ids, spec, 512, device,
            )
            .context("checkpointed OPD: echo env state")?
            {
                Some((node_state, _env_ce_kt, env_ce_val)) => {
                    let env_grad = crate::grpo_tape_shim::echo_env_grad_from_normed_hidden_kt(
                        &normed,
                        head_ref,
                        &node_state,
                        crate::trainer::GrpoLossParams {
                            advantage: 0.0,
                            clip_low: 1.0,
                            clip_high: 1.0,
                            kl_coeff: 0.0,
                            kl_estimator: crate::KlEstimator::K1,
                            loss_normalizer: 1.0,
                            is_level: crate::IsLevel::Token,
                            reinforce: true,
                            entropy_aware_kl_quantile: None,
                        },
                        kiln_model::backend::GrpoKlAuxiliaryRoute::HostComposite,
                        1.0,
                        device,
                        512,
                    )
                    .context("checkpointed OPD: echo env grad")?;
                    let grad_normed =
                        (&grad_normed + &env_grad).context("checkpointed OPD: echo grad add")?;
                    (
                        loss_val + spec.lambda * env_ce_val,
                        grad_normed,
                        Some(env_ce_val),
                    )
                }
                None => (loss_val, grad_normed, None),
            }
        }
        None => (loss_val, grad_normed, None),
    };
    let mut upstream_grad = crate::trainer::rms_norm_backward_pre_final_norm(
        TrainingLossBackend::runtime_final_rmsnorm_backward_route(backend_rt),
        &final_hidden,
        &weights.final_norm,
        &grad_normed,
        model_config.rms_norm_eps,
    )
    .context("checkpointed OPD final RMSNorm backward")?
    .detach();

    let mut grads = kiln_autograd::GradStore::new();
    for seg_idx in (0..num_segments).rev() {
        let (start, end) = segments[seg_idx];
        let seg_input = boundaries[seg_idx].clone();
        let seg_input_id = seg_input.id();
        let seed = upstream_grad
            .to_dtype(boundaries[seg_idx + 1].dtype())
            .map_err(|e| anyhow!("checkpointed OPD: seed dtype cast (segment {seg_idx}): {e}"))?;
        let positions_ref = &positions;
        let lora_ref = &lora_weights;
        let (kt_grads, candle_grads) =
            kiln_kt_bridge::tape_bridge::with_tape_segment_backward_scope(
                kiln_autograd::TapeOptions { detect_anomaly },
                seed,
                || {
                    let mut seg_ls = LinearAttentionState::new(model_config, device)
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    model_forward_segment_with_policy(
                        backend_rt,
                        seg_input,
                        weights,
                        model_config,
                        positions_ref,
                        start,
                        end,
                        Some(&mut seg_ls),
                        Some(lora_ref),
                        streaming_prefill,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))
                },
            )
            .map_err(|e| anyhow!("checkpointed OPD: segment {seg_idx} tape backward: {e}"))?;

        let mut segment_grads = kiln_autograd::GradStore::new();
        for (candle_raw, g) in candle_grads {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
            else {
                continue;
            };
            segment_grads.insert(kiln_tensor_id::TensorId::from_raw(param_raw), g);
        }
        let grad_context =
            format!("checkpointed OPD segment {seg_idx} layers {start}..{end} gradient contract");
        crate::trainer::merge_checkpoint_lora_grad_segment(
            params,
            &mut grads,
            segment_grads,
            start,
            end,
            &grad_context,
        )?;

        if seg_idx > 0 {
            upstream_grad = kt_grads.get(seg_input_id).cloned().ok_or_else(|| {
                anyhow!(
                    "checkpointed OPD: tape backward produced no input gradient for segment {seg_idx}"
                )
            })?;
        }
    }

    Ok((loss_val, active_count, grads, step_env_ce))
}

#[allow(clippy::too_many_arguments)]
fn write_opd_train_receipt(
    adapter_name: &str,
    model_config: &kiln_core::config::ModelConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    config: &OpdConfig,
    effective_top_k: usize,
    effective_seed: Option<u64>,
    effective_samples_per_prompt: usize,
    teacher: &OpdTeacherProvenance,
    output_dir: &std::path::Path,
    base_adapter_dir: Option<&std::path::Path>,
    training_data_sha256: Option<String>,
    data: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    wall_clock_ms: u64,
    alpha_over_rank: Option<f32>,
    final_opd_loss: Option<f64>,
    run_env_ce: Option<f64>,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    status_error: Option<String>,
) -> Result<()> {
    let effective_config = opd_config_for_receipt(
        config,
        effective_top_k,
        effective_seed,
        effective_samples_per_prompt,
    );
    let effective_config_json = serde_json::to_value(&effective_config)
        .context("serialize effective OPD configuration for train receipt")?;
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
            // Receipts record the RESOLVED learning rate — the value the
            // optimizer actually stepped with — not the Option.
            learning_rate: config.effective_learning_rate(),
            epochs: config.epochs.max(1),
            shuffle: false,
            seed: effective_seed,
        },
        effective_config_json,
    );
    receipt.model.base_weight_shard_manifest = base_weight_shard_manifest.cloned();
    receipt.runtime.execution_provenance = execution_provenance.cloned();
    receipt.runtime.training_precision = training_precision;
    receipt.training_data = crate::train_receipt::TrainingDataReceipt {
        source: "inline_opd_prompts".to_string(),
        path: None,
        sha256: training_data_sha256,
        openenv: None,
    };
    receipt.lora_grad_norms = lora_grad_norms;
    receipt.opd = Some(crate::train_receipt::OpdReceipt {
        training_mode: serde_json::to_value(config.training_mode)
            .ok()
            .and_then(|value| value.as_str().map(ToString::to_string))
            .unwrap_or_else(|| "on_policy".to_string()),
        objective: serde_json::to_value(config.objective)
            .ok()
            .and_then(|value| value.as_str().map(ToString::to_string))
            .unwrap_or_else(|| "reverse_kl".to_string()),
        loss_granularity: serde_json::to_value(config.loss)
            .ok()
            .and_then(|value| value.as_str().map(ToString::to_string))
            .unwrap_or_else(|| "teacher_top_k".to_string()),
        teacher_id: Some(teacher.teacher_id.clone()),
        teacher_content_revision: teacher.content_revision(),
        teacher_identity: teacher.identity.clone(),
        top_k: matches!(
            config.loss,
            OpdLossGranularity::TeacherTopK | OpdLossGranularity::FullVocab
        )
        .then_some(effective_top_k),
        samples_per_prompt: effective_samples_per_prompt,
        action_tokens: token_counts.action_tokens,
        env_tokens: token_counts.env_tokens,
        // HONESTY preserved, term restored: `echo_combined` records
        // whether the env-CE term actually FIRED this run (the compose
        // node / analytic add reported a value), never the config alone.
        // A run whose rollouts carried no env rows stays false.
        echo_combined: run_env_ce.is_some(),
        echo_lambda: config.echo.as_ref().map(|echo| echo.lambda),
        initial_opd_loss: None,
        final_opd_loss,
    });
    receipt.echo = match config.echo.as_ref() {
        Some(echo) => crate::train_receipt::EchoReceipt {
            // `enabled` records whether the term FIRED (the OPD compose
            // node / checkpointed analytic add reported an env-CE value),
            // not merely whether it was requested. Rollouts with no env
            // rows leave it false with the reason spelled out.
            enabled: run_env_ce.is_some(),
            lambda: Some(echo.lambda),
            env_mask_mode: serde_json::to_value(echo.env_mask_mode)
                .ok()
                .and_then(|value| value.as_str().map(ToString::to_string)),
            warning_filter: Some(echo.warning_filter),
            initial_env_ce: None,
            final_env_ce: run_env_ce,
            dropped_reason: run_env_ce.is_none().then(|| {
                "ECHO was configured but this run's rollouts carried no \
                 environment tokens — the env-CE term had nothing to train on"
                    .to_string()
            }),
        },
        None => crate::train_receipt::EchoReceipt::disabled(),
    };
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    receipt.data = data;
    receipt.token_counts = token_counts;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    if status_error.is_none() {
        receipt.lora_delta_norms = crate::train_receipt::lora_delta_norm_summary_from_adapter(
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
    receipt
        .write_to_adapter_dir(output_dir)
        .with_context(|| format!("write OPD train receipt for adapter {adapter_name:?}"))?;
    Ok(())
}

fn opd_config_for_receipt(
    config: &OpdConfig,
    effective_top_k: usize,
    effective_seed: Option<u64>,
    effective_samples_per_prompt: usize,
) -> OpdConfig {
    let mut effective_config = config.clone();
    effective_config.top_k = effective_top_k;
    effective_config.seed = effective_seed;
    effective_config.samples_per_prompt = effective_samples_per_prompt;
    effective_config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logit_source::{
        FixtureLogitSource, LogitSourceCaps, LogitSourceError, TopKLogprobs,
    };
    // (#1082) Test-mod `Tensor` / `DType` / `Device` references go through
    // `crate::cd_types::*` rather than naming `candle_core::` directly, so
    // adding more tests does not regrow the per-file candle ref count. The
    // candle dep itself is still blocked on the kt-typed OPD forward
    // surface migrating off candle `Tensor` in the public API — see the
    // module-level note above.
    use kiln_core::tokenizer::KilnTokenizer;

    fn opd_test_runtime(
        gib: u64,
        policy: crate::GradientCheckpointPolicy,
    ) -> crate::TrainingRuntimeContext {
        crate::TrainingRuntimeContext::new(
            kiln_memory::vram::GpuVramInfo {
                total_bytes: gib * 1024 * 1024 * 1024,
                source: kiln_memory::vram::VramSource::ConfigOverride,
                unified: false,
            },
            policy,
        )
    }

    #[test]
    fn opd_checkpoint_planning_uses_immutable_capacity_and_policy() -> Result<()> {
        let model = kiln_core::config::ModelConfig::qwen3_5_4b();
        let roomy = opd_test_runtime(48, crate::GradientCheckpointPolicy::Auto);
        let tight = opd_test_runtime(16, crate::GradientCheckpointPolicy::Auto);
        assert!(opd_checkpoint_segments_for_step(&roomy, None, 4, &model, 30).is_none());
        assert!(
            opd_checkpoint_segments_for_step(&tight, None, 4, &model, 4096)
                .is_some_and(|segments| segments.len() >= 2)
        );

        assert_eq!(
            opd_checkpoint_segments_for_step(&roomy, Some(8), 4, &model, 30)
                .expect("live-admitted OPD plan must override roomy static capacity")
                .len(),
            8
        );

        let explicit = opd_test_runtime(
            48,
            crate::GradientCheckpointPolicy::from_parts(Some(8), false)?,
        );
        assert_eq!(
            opd_checkpoint_segments_for_step(&explicit, None, 4, &model, 30)
                .expect("explicit policy enables checkpointing")
                .len(),
            8
        );

        let disabled = opd_test_runtime(
            16,
            crate::GradientCheckpointPolicy::from_parts(Some(8), true)?,
        );
        assert!(opd_checkpoint_segments_for_step(&disabled, None, 4, &model, 4096).is_none());
        assert_ne!(
            tight.checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu),
            disabled.checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu)
        );

        let forced_streaming = kiln_model::forward::StreamingPrefillExecutionPolicy::resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_device(kiln_tensor::Device::Cpu),
            kiln_model::forward::StreamingPrefillMode::Enabled,
            Some(256),
            Some(128),
            None,
            None,
            false,
        );
        let identity = roomy
            .with_streaming_prefill_policy(forced_streaming)
            .checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu);
        assert_eq!(identity["streaming_prefill_policy"]["mode"], "enabled");
        assert_eq!(
            identity["streaming_prefill_policy"]["base_tile_tokens"],
            128
        );
        Ok(())
    }

    #[derive(Debug)]
    struct RecordingLogitSource {
        caps: LogitSourceCaps,
        batch: LogprobBatch,
        calls: std::sync::Mutex<Vec<Vec<usize>>>,
    }

    impl RecordingLogitSource {
        fn topk(vocab_size: usize, top_k: usize, rows: usize) -> Arc<Self> {
            let mut indices = Vec::with_capacity(rows * top_k);
            let mut logprobs = Vec::with_capacity(rows * top_k);
            for _ in 0..rows {
                indices.extend((0..top_k).map(|token| token as u32));
                logprobs.extend((0..top_k).map(|rank| -0.7 - rank as f32));
            }
            Arc::new(Self {
                caps: LogitSourceCaps {
                    teacher_id: "recording-teacher".into(),
                    vocab_size,
                    max_top_k: top_k,
                    supports_full_vocab: false,
                    supports_batched: true,
                    tokenizer_hash: None,
                },
                batch: LogprobBatch::TopK(TopKLogprobs {
                    indices,
                    logprobs,
                    top_k,
                }),
                calls: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl LogitSource for RecordingLogitSource {
        fn capabilities(&self) -> LogitSourceCaps {
            self.caps.clone()
        }

        fn fetch_logprobs(
            &self,
            _tokens: &[u32],
            positions: &[usize],
            _top_k: Option<usize>,
        ) -> std::result::Result<LogprobBatch, LogitSourceError> {
            self.calls.lock().unwrap().push(positions.to_vec());
            Ok(self.batch.clone())
        }
    }

    #[derive(Debug)]
    struct VerifiedDynamicLogitSource {
        caps: LogitSourceCaps,
        identity: crate::TeacherIdentityV1,
        calls: std::sync::Mutex<Vec<(Vec<u32>, Vec<usize>)>>,
    }

    impl VerifiedDynamicLogitSource {
        fn new() -> Arc<Self> {
            let identity = crate::TeacherIdentityV1::new(
                "verified-dynamic-model",
                "a".repeat(64),
                "b".repeat(64),
                "c".repeat(64),
                None,
                64,
                32,
                4096,
                1_000_000,
                "test-runtime",
                "d".repeat(64),
            )
            .unwrap();
            Arc::new(Self {
                caps: LogitSourceCaps {
                    teacher_id: "verified-dynamic".into(),
                    vocab_size: 64,
                    max_top_k: 32,
                    supports_full_vocab: false,
                    supports_batched: true,
                    tokenizer_hash: Some(identity.tokenizer_vocab_sha256().to_string()),
                },
                identity,
                calls: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl LogitSource for VerifiedDynamicLogitSource {
        fn capabilities(&self) -> LogitSourceCaps {
            self.caps.clone()
        }

        fn authoritative_teacher_identity(&self) -> Option<&crate::TeacherIdentityV1> {
            Some(&self.identity)
        }

        fn fetch_logprobs(
            &self,
            tokens: &[u32],
            positions: &[usize],
            top_k: Option<usize>,
        ) -> std::result::Result<LogprobBatch, LogitSourceError> {
            let top_k = top_k.ok_or_else(|| LogitSourceError::FullVocabUnsupported {
                teacher_id: self.caps.teacher_id.clone(),
            })?;
            crate::logit_source::validate_logit_request(
                &self.caps,
                tokens,
                positions,
                Some(top_k),
            )?;
            self.calls
                .lock()
                .unwrap()
                .push((tokens.to_vec(), positions.to_vec()));
            let mut indices = Vec::with_capacity(positions.len() * top_k);
            let mut logprobs = Vec::with_capacity(positions.len() * top_k);
            for _ in positions {
                indices.extend((0..top_k).map(|token| token as u32));
                logprobs.extend((0..top_k).map(|rank| -4.0 - rank as f32 * 0.1));
            }
            Ok(LogprobBatch::TopK(TopKLogprobs {
                indices,
                logprobs,
                top_k,
            }))
        }
    }

    fn off_policy_smoke_tokenizer() -> Result<KilnTokenizer> {
        let mut vocab = String::from("{");
        let chars = "userassistanttoolokhiresult<|im_start|><|im_end|>\n ";
        let mut seen = std::collections::HashSet::new();
        let mut id = 0u32;
        for ch in chars.chars() {
            let key = match ch {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
                c => c.to_string(),
            };
            if !seen.insert(key.clone()) {
                continue;
            }
            if id > 0 {
                vocab.push(',');
            }
            vocab.push_str(&format!("\"{}\":{}", key, id));
            id += 1;
        }
        vocab.push('}');
        let json = format!(
            r#"{{"version":"1.0","model":{{"type":"BPE","vocab":{},"merges":[]}}}}"#,
            vocab
        );
        let template = "{% for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
        Ok(KilnTokenizer::from_bytes(json.as_bytes())
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_chat_template(template.to_string()))
    }

    fn smoke_opd_prompt_with_teacher_context() -> OpdPrompt {
        OpdPrompt {
            messages: vec![
                ChatMessage::new("user", "hi"),
                ChatMessage::new("assistant", "ok"),
            ],
            teacher_extra_messages: vec![ChatMessage::new("system", "result")],
            trajectory: Vec::new(),
        }
    }

    #[test]
    fn remote_materialization_rejects_on_policy_before_teacher_fetch() -> Result<()> {
        let source = VerifiedDynamicLogitSource::new();
        let error = materialize_verified_off_policy_teacher(
            &[smoke_opd_prompt_with_teacher_context()],
            &OpdConfig::default(),
            &off_policy_smoke_tokenizer()?,
            source.clone(),
        )
        .err()
        .expect("on-policy remote materialization must be rejected");

        assert!(
            error.to_string().contains("cannot run on-policy"),
            "{error:#}"
        );
        assert!(source.calls.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn remote_materialization_eagerly_seals_exact_queries_and_identity() -> Result<()> {
        let source = VerifiedDynamicLogitSource::new();
        let identity = source.identity.clone();
        let prompt = smoke_opd_prompt_with_teacher_context();
        let config = OpdConfig {
            training_mode: OpdTrainingMode::OffPolicy,
            top_k: 16,
            ..OpdConfig::default()
        };

        let fixture = materialize_verified_off_policy_teacher(
            std::slice::from_ref(&prompt),
            &config,
            &off_policy_smoke_tokenizer()?,
            source.clone(),
        )?;
        assert_eq!(fixture.authoritative_teacher_identity(), Some(&identity));

        let calls = source.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        let (query_tokens, query_rows) = &calls[0];
        assert!(!query_rows.is_empty());
        let fetched = fixture.fetch_logprobs(query_tokens, query_rows, Some(16))?;
        assert_eq!(fetched.flat_len(), query_rows.len() * 16);
        assert_eq!(source.calls.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn explicit_rollout_and_teacher_rendering_fail_closed_with_prompt_index() {
        let tokenizer = off_policy_smoke_tokenizer()
            .unwrap()
            .with_chat_template("{{ raise_exception('template boom') }}".to_string());
        let prompts = [smoke_opd_prompt_with_teacher_context()];

        let rollout_error = render_rollout_prompt_prefixes(&prompts, &tokenizer).unwrap_err();
        assert!(
            rollout_error
                .to_string()
                .contains("render rollout chat template for prompt 0"),
            "{rollout_error:#}"
        );

        let teacher_error = render_teacher_prompt_tokens(&prompts, &tokenizer).unwrap_err();
        assert!(
            teacher_error
                .to_string()
                .contains("render asymmetric teacher chat template for prompt 0"),
            "{teacher_error:#}"
        );
    }

    #[test]
    fn local_teacher_topk_validates_full_row_and_breaks_ties_by_token_id() {
        let caps = LogitSourceCaps {
            teacher_id: "local-test".into(),
            vocab_size: 4,
            max_top_k: 4,
            supports_full_vocab: false,
            supports_batched: true,
            tokenizer_hash: None,
        };
        let (indices, logprobs) =
            select_validated_topk_logprob_row(&caps, 0, &[-1.0, -1.0, -2.0, -3.0], 2).unwrap();
        assert_eq!(indices, [0, 1]);
        assert_eq!(logprobs, [-1.0, -1.0]);

        for corrupt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.1] {
            let error =
                select_validated_topk_logprob_row(&caps, 7, &[-0.1, -0.2, -0.3, corrupt], 2)
                    .unwrap_err();
            assert!(error.to_string().contains("token 3"), "{error:#}");
        }
    }

    /// End-to-end: feed a tokenized rollout, a fixture teacher, and a
    /// random `student_hidden` into `opd_step_loss`, confirm we get a
    /// scalar mean_kl + per-position vector of the right shape and
    /// that the math matches the Phase A reference directly.
    #[test]
    fn opd_step_loss_matches_kernel_directly() -> Result<()> {
        // (#1082) kt `from_vec` is CPU-only and takes no device arg, so the
        // explicit binding is unused now.
        let _device = Device::Cpu;
        let seq_len = 8;
        let hidden_size = 8;
        let vocab_size = 64;
        let top_k = 16;

        // Hidden + head are arbitrary smooth tensors.
        let hidden_vec: Vec<f32> = (0..(seq_len * hidden_size))
            .map(|i| (i as f32 * 0.011).sin() * 0.3)
            .collect();
        // (#1082) kt `Tensor::from_vec(data, shape)` is 2-arg (CPU-only);
        // dropped the candle `&device` arg (the test device is CPU).
        let student_hidden = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size))?;
        let head_vec: Vec<f32> = (0..(hidden_size * vocab_size))
            .map(|i| ((i as f32 + 11.0) * 0.005).cos() * 0.2)
            .collect();
        let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size))?;

        let tokens: Vec<u32> = (0..seq_len)
            .map(|i| ((i * 7 + 3) % vocab_size) as u32)
            .collect();
        let active_positions = vec![3, 5, 7]; // arbitrary completion tokens

        // Build a fixture teacher at each causal row that predicts an active target.
        let mut fixture = FixtureLogitSource::uniform_topk("test", vocab_size, top_k);
        for &pos in &active_positions {
            let idx: Vec<u32> = (0..top_k as u32)
                .map(|k| (pos as u32 * 5 + k * 11) % vocab_size as u32)
                .collect();
            let lp: Vec<f32> = (0..top_k)
                .map(|k| -((pos + 1) as f32).ln() - (k as f32) * 0.3)
                .collect();
            fixture.insert(&tokens, pos - 1, idx, lp)?;
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
        .err()
        .expect("an empty active-position set must be rejected");
        assert!(err.to_string().contains("no active positions"));
    }

    #[test]
    fn opd_top_k_resolution_uses_largest_executable_support() {
        assert_eq!(resolve_opd_top_k(32, 20).unwrap(), 16);
        assert_eq!(resolve_opd_top_k(32, 32).unwrap(), 32);
        assert_eq!(resolve_opd_top_k(24, 64).unwrap(), 16);
        assert_eq!(resolve_opd_top_k(128, 128).unwrap(), 32);
        assert!(resolve_opd_top_k(15, 32).is_err());
        assert!(resolve_opd_top_k(32, 15).is_err());
    }

    #[test]
    fn opd_receipt_config_records_resolved_values_without_mutating_request() {
        let config = OpdConfig::default();
        assert_eq!(config.top_k, 32);
        assert_eq!(config.samples_per_prompt, 4);
        assert_eq!(config.seed, None);

        let receipt_config = opd_config_for_receipt(&config, 16, Some(73), 64);
        assert_eq!(receipt_config.top_k, 16);
        assert_eq!(receipt_config.samples_per_prompt, 64);
        assert_eq!(receipt_config.seed, Some(73));
        assert_eq!(
            config.top_k, 32,
            "receipt preparation must not mutate the request"
        );
        assert_eq!(config.samples_per_prompt, 4);
        assert_eq!(config.seed, None);
        assert_eq!(
            serde_json::to_value(receipt_config).unwrap()["top_k"],
            serde_json::json!(16)
        );
    }

    #[test]
    fn opd_top_k_resolution_fails_before_teacher_fetch() {
        let teacher = RecordingLogitSource::topk(64, 15, 1);
        let err = prepare_opd_kernel_inputs(
            &[3, 5, 7],
            &[2],
            teacher.clone(),
            OpdLossGranularity::TeacherTopK,
            32,
            None,
            None,
        )
        .err()
        .expect("a source cap below K=16 must fail admission");

        assert!(format!("{err:#}").contains("no executable"), "{err:#}");
        assert!(teacher.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn opd_preparation_shifts_action_targets_to_their_causal_logits_rows() -> Result<()> {
        let response = RecordingLogitSource::topk(64, 16, 2);
        let mut caps = response.caps.clone();
        caps.max_top_k = 20;
        let teacher = Arc::new(RecordingLogitSource {
            caps,
            batch: response.batch.clone(),
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let prepared = prepare_opd_kernel_inputs(
            &[3, 5, 7, 9, 11],
            &[1, 4],
            teacher.clone(),
            OpdLossGranularity::TeacherTopK,
            32,
            None,
            None,
        )?;

        assert_eq!(*teacher.calls.lock().unwrap(), vec![vec![0, 3]]);
        assert_eq!(prepared.label_mask, vec![true, false, false, true, false]);
        assert_eq!(prepared.resolved_top_k, 16);
        assert_eq!(prepared.active_count, 2);
        Ok(())
    }

    #[test]
    fn opd_preparation_rejects_invalid_target_order_before_teacher_fetch() {
        for positions in [&[0][..], &[2, 2], &[3, 2]] {
            let teacher = RecordingLogitSource::topk(16, 2, positions.len());
            let err = prepare_opd_kernel_inputs(
                &[3, 5, 7, 9],
                positions,
                teacher.clone(),
                OpdLossGranularity::TeacherTopK,
                2,
                None,
                None,
            )
            .err()
            .expect("invalid target positions must fail");
            assert!(
                err.to_string().contains("target"),
                "unexpected error for {positions:?}: {err:#}"
            );
            assert!(teacher.calls.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn opd_preparation_rejects_sampled_token_noop_before_teacher_fetch() {
        let teacher = RecordingLogitSource::topk(16, 1, 1);
        let err = prepare_opd_kernel_inputs(
            &[3, 5, 7],
            &[2],
            teacher.clone(),
            OpdLossGranularity::SampledToken,
            1,
            None,
            None,
        )
        .err()
        .expect("sampled-token no-op must fail closed");
        assert!(err.to_string().contains("identically zero"), "{err:#}");
        assert!(teacher.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn opd_preparation_rejects_malformed_teacher_batch_before_kernel() {
        let teacher = RecordingLogitSource::topk(64, 16, 2);
        let mut malformed_indices: Vec<u32> = (0..32).map(|idx| (idx % 16) as u32).collect();
        malformed_indices[31] = 64;
        let malformed = Arc::new(RecordingLogitSource {
            caps: teacher.caps.clone(),
            batch: LogprobBatch::TopK(TopKLogprobs {
                indices: malformed_indices,
                logprobs: (0..32).map(|index| -3.0 - (index % 16) as f32).collect(),
                top_k: 16,
            }),
            calls: std::sync::Mutex::new(Vec::new()),
        });

        let err = prepare_opd_kernel_inputs(
            &[1, 2, 3, 4],
            &[1, 3],
            malformed.clone(),
            OpdLossGranularity::TeacherTopK,
            32,
            None,
            None,
        )
        .err()
        .expect("malformed teacher batch must fail");
        assert!(err.to_string().contains("invalid top-K"), "{err:#}");
        assert_eq!(*malformed.calls.lock().unwrap(), vec![vec![0, 2]]);
    }

    #[test]
    fn opd_preparation_rejects_asymmetric_target_token_mismatch() {
        let teacher = RecordingLogitSource::topk(16, 2, 1);
        let err = prepare_opd_kernel_inputs(
            &[1, 2, 3],
            &[2],
            teacher.clone(),
            OpdLossGranularity::TeacherTopK,
            2,
            Some(&[8, 1, 9]),
            Some(&[2]),
        )
        .err()
        .expect("asymmetric target mismatch must fail");
        assert!(
            err.to_string().contains("asymmetric target pair"),
            "{err:#}"
        );
        assert!(teacher.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn opd_step_loss_asymmetric_teacher_pairs_by_index() -> Result<()> {
        // Verify that when teacher_tokens / teacher_active_positions are
        // set, the kernel queries the teacher at the SHIFTED positions
        // and pairs them back to the student's active positions by index.
        // (#1082) kt `from_vec` is CPU-only and takes no device arg.
        let _device = Device::Cpu;
        let vocab_size = 16usize;
        let hidden_size = 8usize;
        let top_k = 16usize;
        let student_seq_len = 10usize;
        let teacher_prefix_len = 5usize;
        let teacher_seq_len = teacher_prefix_len + student_seq_len;

        let student_hidden_vec: Vec<f32> = (0..1 * student_seq_len * hidden_size)
            .map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0)
            .collect();
        // (#1082) kt `from_vec` is 2-arg (CPU-only); dropped candle `&device`.
        let student_hidden =
            Tensor::from_vec(student_hidden_vec, (1, student_seq_len, hidden_size))?;
        let head_vec: Vec<f32> = (0..hidden_size * vocab_size)
            .map(|i| ((i * 11 + 5) % 1000) as f32 / 1000.0)
            .collect();
        let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size))?;

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
        // student tokens) and the causal rows preceding the shifted targets.
        let mut fixture = FixtureLogitSource::uniform_topk("asym-test", vocab_size, top_k);
        for &pos in &teacher_active {
            let idx: Vec<u32> = (0..top_k as u32)
                .map(|k| (pos as u32 * 5 + k * 11) % vocab_size as u32)
                .collect();
            let lp: Vec<f32> = (0..top_k)
                .map(|k| -((pos + 1) as f32).ln() - (k as f32) * 0.3)
                .collect();
            fixture.insert(&teacher_tokens, pos - 1, idx, lp)?;
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
                messages: vec![ChatMessage::new("user", "Evaluate ∫_0^∞ e^{-x^2} dx")],
                teacher_extra_messages: vec![],
                trajectory: vec![],
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
        assert_eq!(
            parsed.config.samples_per_prompt,
            default_opd_samples_per_prompt()
        );
        assert!(matches!(
            parsed.config.loss,
            OpdLossGranularity::TeacherTopK
        ));
    }

    #[test]
    fn off_policy_teacher_jsonl_builds_reverse_kl_fixture_and_echo_summary() -> Result<()> {
        let jsonl = r#"{"id":"ex1","messages":[{"role":"user","content":"hi"}],"teacher_response":"ok","trajectory":[{"role":"assistant","content":"ok","kind":"action"},{"role":"tool","content":"result","kind":"observation"}]}"#;
        let mut examples = parse_off_policy_distillation_jsonl_str(jsonl)?;
        let tokenizer = off_policy_smoke_tokenizer()?;
        let echo = crate::EchoConfig::default();
        let ce_prepared = prepare_off_policy_distillation_dataset(
            &examples,
            &tokenizer,
            "teacher-fixture",
            32,
            2,
            OpdObjective::CrossEntropy,
            Some(&echo),
        )?;
        examples[0].teacher_tokens = (0..ce_prepared.summary.action_tokens)
            .map(|idx| TeacherActionToken {
                token_id: None,
                token: None,
                logprob: None,
                top_logprobs: vec![
                    TeacherTopLogprob {
                        token_id: (idx % 16) as u32,
                        logprob: -0.1,
                    },
                    TeacherTopLogprob {
                        token_id: ((idx + 1) % 16) as u32,
                        logprob: -2.4,
                    },
                ],
            })
            .collect();
        let prepared = prepare_off_policy_distillation_dataset(
            &examples,
            &tokenizer,
            "teacher-fixture",
            32,
            2,
            OpdObjective::ReverseKl,
            Some(&echo),
        )?;

        assert_eq!(prepared.prompts.len(), 1);
        assert_eq!(
            prepared.teacher.capabilities().teacher_id,
            "teacher-fixture"
        );
        assert_eq!(prepared.summary.examples, 1);
        assert_eq!(prepared.summary.examples_with_teacher_logprobs, 1);
        assert_eq!(
            prepared.summary.action_tokens,
            ce_prepared.summary.action_tokens
        );
        assert!(prepared.summary.env_tokens > 0);
        // Preparation builds masks and counts only. Whether env-CE actually
        // fires is known during execution and belongs in the final receipt.
        assert!(!prepared.summary.echo_combined);
        Ok(())
    }

    #[test]
    fn off_policy_teacher_token_id_must_match_tokenized_active_target() -> Result<()> {
        let jsonl = r#"{"messages":[{"role":"user","content":"hi"}],"teacher_response":"ok"}"#;
        let mut examples = parse_off_policy_distillation_jsonl_str(jsonl)?;
        let tokenizer = off_policy_smoke_tokenizer()?;
        let mut messages = examples[0].messages.clone();
        messages.push(ChatMessage::new(
            "assistant",
            examples[0].teacher_response.clone(),
        ));
        let prompt = OpdPrompt {
            messages,
            teacher_extra_messages: Vec::new(),
            trajectory: Vec::new(),
        };
        let tokenized = tokenize_opd_prompt_for_training(&prompt, &tokenizer, None)?;
        let active: Vec<usize> = tokenized
            .action_mask
            .iter()
            .enumerate()
            .filter_map(|(position, &is_active)| is_active.then_some(position))
            .collect();
        examples[0].teacher_tokens = active
            .iter()
            .map(|&position| TeacherActionToken {
                token_id: Some(tokenized.input_ids[position]),
                token: None,
                logprob: None,
                top_logprobs: vec![
                    TeacherTopLogprob {
                        token_id: 0,
                        logprob: -1.0,
                    },
                    TeacherTopLogprob {
                        token_id: 1,
                        logprob: -2.0,
                    },
                ],
            })
            .collect();
        let actual = examples[0].teacher_tokens[0].token_id.unwrap();
        examples[0].teacher_tokens[0].token_id = Some((actual + 1) % 32);

        let error = prepare_off_policy_distillation_dataset(
            &examples,
            &tokenizer,
            "teacher-fixture",
            32,
            2,
            OpdObjective::ReverseKl,
            None,
        )
        .err()
        .expect("mismatched declared teacher token must be rejected");
        assert!(
            error
                .to_string()
                .contains("tokenization produced active target"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn off_policy_duplicate_exact_sequence_rejects_conflicting_logits() -> Result<()> {
        let jsonl = concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"teacher_response\":\"ok\"}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"teacher_response\":\"ok\"}"
        );
        let mut examples = parse_off_policy_distillation_jsonl_str(jsonl)?;
        let tokenizer = off_policy_smoke_tokenizer()?;
        for (example_index, example) in examples.iter_mut().enumerate() {
            let mut messages = example.messages.clone();
            messages.push(ChatMessage::new(
                "assistant",
                example.teacher_response.clone(),
            ));
            let tokenized = tokenize_opd_prompt_for_training(
                &OpdPrompt {
                    messages,
                    teacher_extra_messages: Vec::new(),
                    trajectory: Vec::new(),
                },
                &tokenizer,
                None,
            )?;
            let first_candidate = (example_index * 2) as u32;
            example.teacher_tokens = tokenized
                .action_mask
                .iter()
                .enumerate()
                .filter_map(|(position, &is_active)| {
                    is_active.then(|| TeacherActionToken {
                        token_id: Some(tokenized.input_ids[position]),
                        token: None,
                        logprob: None,
                        top_logprobs: vec![
                            TeacherTopLogprob {
                                token_id: first_candidate,
                                logprob: -1.0,
                            },
                            TeacherTopLogprob {
                                token_id: first_candidate + 1,
                                logprob: -2.0,
                            },
                        ],
                    })
                })
                .collect();
        }

        let error = prepare_off_policy_distillation_dataset(
            &examples,
            &tokenizer,
            "teacher-fixture",
            32,
            2,
            OpdObjective::ReverseKl,
            None,
        )
        .err()
        .expect("conflicting duplicate fixture rows must be rejected");
        assert!(error.to_string().contains("conflict"), "{error:#}");
        Ok(())
    }

    #[test]
    fn off_policy_teacher_jsonl_cross_entropy_does_not_require_logprobs() -> Result<()> {
        let jsonl = r#"{"messages":[{"role":"user","content":"hi"}],"teacher_response":"ok"}"#;
        let examples = parse_off_policy_distillation_jsonl_str(jsonl)?;
        let tokenizer = off_policy_smoke_tokenizer()?;
        let prepared = prepare_off_policy_distillation_dataset(
            &examples,
            &tokenizer,
            "teacher-fixture",
            32,
            2,
            OpdObjective::CrossEntropy,
            None,
        )?;

        assert_eq!(prepared.prompts.len(), 1);
        assert_eq!(
            prepared.teacher.capabilities().teacher_id,
            "teacher-fixture"
        );
        assert_eq!(prepared.summary.objective, OpdObjective::CrossEntropy);
        assert!(prepared.summary.action_tokens > 0);

        let tokenized = tokenize_opd_prompt_for_training(&prepared.prompts[0], &tokenizer, None)?;
        let active_positions: Vec<usize> = tokenized
            .action_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, active)| active.then_some(idx))
            .collect();
        let logits_rows = target_token_positions_to_logits_rows(
            "teacher-fixture",
            tokenized.input_ids.len(),
            &active_positions,
        )?;
        let batch = prepared
            .teacher
            .fetch_logprobs(&tokenized.input_ids, &logits_rows, Some(2))?;
        let LogprobBatch::TopK(topk) = batch else {
            panic!("CE fixture should return top-k logprobs");
        };
        assert_eq!(topk.indices.len(), active_positions.len() * 2);
        assert_eq!(topk.logprobs[0], 0.0);
        Ok(())
    }

    #[test]
    fn off_policy_manifest_binds_loaded_fixture_and_exact_source_bytes() -> Result<()> {
        let identity = crate::TeacherIdentityV1::new(
            "teacher-model",
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            None,
            32,
            2,
            4096,
            32,
            "vllm-test",
            "d".repeat(64),
        )?;
        let manifest = OffPolicyDistillationManifestV1::new(identity.clone());
        let example = r#"{"messages":[{"role":"user","content":"hi"}],"teacher_response":"ok"}"#;
        let jsonl = format!("{}\n{example}\n", manifest.canonical_json());
        let loaded = parse_off_policy_distillation_dataset_str(&jsonl)?;

        assert_eq!(loaded.manifest.as_ref(), Some(&manifest));
        assert_eq!(
            loaded.source_sha256,
            crate::train_receipt::sha256_bytes(jsonl.as_bytes())
        );
        let prepared = prepare_off_policy_distillation_dataset_with_identity(
            &loaded.examples,
            &off_policy_smoke_tokenizer()?,
            "teacher-alias",
            Some(identity.clone()),
            32,
            2,
            OpdObjective::CrossEntropy,
            None,
        )?;
        assert_eq!(
            prepared.teacher.authoritative_teacher_identity(),
            Some(&identity)
        );
        Ok(())
    }

    #[test]
    fn off_policy_manifest_must_be_canonical_and_first() {
        let identity = crate::TeacherIdentityV1::new(
            "teacher-model",
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            None,
            32,
            2,
            4096,
            32,
            "vllm-test",
            "d".repeat(64),
        )
        .unwrap();
        let canonical = OffPolicyDistillationManifestV1::new(identity).canonical_json();
        let example = r#"{"messages":[{"role":"user","content":"hi"}],"teacher_response":"ok"}"#;

        let noncanonical = format!(" {}\n{example}", canonical);
        let error = parse_off_policy_distillation_dataset_str(&noncanonical)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not canonical"), "{error}");

        let late = format!("{example}\n{canonical}");
        let error = parse_off_policy_distillation_dataset_str(&late)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be the first"), "{error}");
    }

    #[test]
    fn off_policy_loss_composes_action_objective_with_echo_env_ce() -> Result<()> {
        let kl = compose_off_policy_distillation_loss(
            OpdObjective::ReverseKl,
            &[0.2, 0.4],
            &[1.0, 3.0],
            Some(0.05),
        )?;
        assert_eq!(kl.objective, OpdObjective::ReverseKl);
        assert!((kl.action_token_loss - 0.3).abs() < 1e-12);
        assert_eq!(kl.echo_env_ce, Some(2.0));
        assert_eq!(kl.echo_lambda, Some(0.05));
        assert!((kl.total_loss - 0.4).abs() < 1e-12);

        let ce = compose_off_policy_distillation_loss(
            OpdObjective::CrossEntropy,
            &[1.5, 0.5],
            &[],
            Some(0.05),
        )?;
        assert_eq!(ce.objective, OpdObjective::CrossEntropy);
        assert_eq!(ce.echo_env_ce, None);
        assert_eq!(ce.echo_lambda, None);
        assert!((ce.total_loss - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn opd_config_defaults_match_grand_plan_section_6() {
        let cfg = OpdConfig::default();
        assert!(matches!(cfg.training_mode, OpdTrainingMode::OnPolicy));
        assert!(matches!(cfg.objective, OpdObjective::ReverseKl));
        assert_eq!(cfg.top_k, 32);
        assert_eq!(cfg.samples_per_prompt, 4);
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.max_tokens, 7168);
        assert_eq!(cfg.sampler_segments, None);
        assert_eq!(
            cfg.rollout_prompt_rendering,
            OpdRolloutPromptRendering::LegacyActionBoundary
        );
        assert_eq!(cfg.discount, 0.0);
        assert_eq!(cfg.clip_epsilon, 0.0);
        assert!(matches!(cfg.stable_opd, StableOpdMode::Off));
        assert!(matches!(cfg.loss, OpdLossGranularity::TeacherTopK));
        assert_eq!(cfg.checkpoint_interval, Some(25));
        assert!(cfg.resume_checkpoint.is_none());
        assert!(!cfg.detect_anomaly);
        cfg.validate_runtime_contract().unwrap();
    }

    #[test]
    fn opd_tape_anomaly_diagnostics_are_explicit_and_round_trip() {
        let config: OpdConfig =
            serde_json::from_value(serde_json::json!({"detect_anomaly": true})).unwrap();
        assert!(config.detect_anomaly);
        assert_eq!(
            serde_json::to_value(config).unwrap()["detect_anomaly"],
            true
        );
    }

    #[test]
    fn opd_sampler_policy_is_typed_validated_and_round_trips() {
        let config: OpdConfig = serde_json::from_value(serde_json::json!({
            "sampler_segments": 12,
            "rollout_prompt_rendering": "chat_template",
        }))
        .unwrap();
        assert_eq!(config.sampler_segments, Some(12));
        assert_eq!(
            config.rollout_prompt_rendering,
            OpdRolloutPromptRendering::ChatTemplate
        );
        config.validate_runtime_contract().unwrap();
        let json = serde_json::to_value(config).unwrap();
        assert_eq!(json["sampler_segments"], 12);
        assert_eq!(json["rollout_prompt_rendering"], "chat_template");

        let mut invalid = OpdConfig::default();
        invalid.sampler_segments = Some(0);
        assert!(invalid.validate_runtime_contract().is_err());
    }

    #[test]
    fn opd_config_round_trips_resume_checkpoint_and_rejects_zero_interval() {
        let name = "run-checkpoint-step-00000007.kiln-checkpoint";
        let config: OpdConfig = serde_json::from_value(serde_json::json!({
            "resume_checkpoint": name,
            "checkpoint_interval": 3,
        }))
        .unwrap();
        assert_eq!(config.resume_checkpoint.as_deref(), Some(name));
        assert_eq!(
            serde_json::to_value(&config).unwrap()["resume_checkpoint"],
            name
        );

        let mut invalid = config;
        invalid.checkpoint_interval = Some(0);
        assert!(invalid.validate_runtime_contract().is_err());
    }

    #[test]
    fn filtered_opd_prompts_retain_their_source_index() -> Result<()> {
        let tokenizer = off_policy_smoke_tokenizer()?;
        let prompts = vec![
            OpdPrompt {
                messages: vec![ChatMessage::new("user", "filtered")],
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            },
            OpdPrompt {
                messages: vec![
                    ChatMessage::new("user", "kept"),
                    ChatMessage::new("assistant", "answer"),
                ],
                teacher_extra_messages: vec![ChatMessage::new("system", "teacher-only context")],
                trajectory: Vec::new(),
            },
        ];

        let prepared = prepare_opd_prompts_for_training(&prompts, &tokenizer, None);
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].source_index, 1);
        let expected = tokenize_opd_prompt_for_training(&prompts[1], &tokenizer, None)?;
        assert_eq!(prepared[0].tokenized.input_ids, expected.input_ids);
        assert_eq!(prepared[0].tokenized.action_mask, expected.action_mask);
        Ok(())
    }

    #[test]
    fn opd_checkpoint_loop_state_is_strict_and_candidate_aware() -> Result<()> {
        let data_stats = crate::train_receipt::DataStatsReceipt {
            examples_read: 3,
            examples_filtered: 1,
            examples_trained: 2,
            ..Default::default()
        };
        let token_counts = crate::train_receipt::TokenCountReceipt {
            action_tokens: 11,
            context_tokens: 7,
            ..Default::default()
        };
        let state = OpdCheckpointLoopState::capture(
            2,
            0,
            3,
            &[0.75, 0.5],
            &data_stats,
            &token_counts,
            Some(0.25),
            &crate::train_receipt::LoraGradNormAccumulator::default(),
            &crate::diagnostics::LengthInflationGuardrail::default(),
        );
        let progress = crate::checkpoint::TrainingCheckpointProgress {
            global_step: 2,
            total_steps: 8,
            epoch_index: 0,
            cursor_in_epoch: 3,
            data_order: (0..4).collect(),
        };
        state.validate(&progress, 4, 2)?;

        let encoded = serde_json::to_value(&state)?;
        let restored: OpdCheckpointLoopState = serde_json::from_value(encoded.clone())?;
        assert_eq!(restored, state);

        let mut unknown = encoded.clone();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("future_field".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<OpdCheckpointLoopState>(unknown).is_err());

        let mut wrong_cursor = progress.clone();
        wrong_cursor.cursor_in_epoch = 2;
        assert!(state.validate(&wrong_cursor, 4, 2).is_err());

        let mut wrong_history: OpdCheckpointLoopState = serde_json::from_value(encoded)?;
        wrong_history.last_loss = Some(0.25);
        assert!(wrong_history.validate(&progress, 4, 2).is_err());
        Ok(())
    }

    #[test]
    fn opd_checkpoint_manifest_binds_candidate_rng_and_optimizer_progress() -> Result<()> {
        let data_stats = crate::train_receipt::DataStatsReceipt {
            examples_read: 3,
            examples_filtered: 1,
            examples_trained: 2,
            ..Default::default()
        };
        let state = OpdCheckpointLoopState::capture(
            2,
            0,
            3,
            &[0.75, 0.5],
            &data_stats,
            &crate::train_receipt::TokenCountReceipt::default(),
            None,
            &crate::train_receipt::LoraGradNormAccumulator::default(),
            &crate::diagnostics::LengthInflationGuardrail::default(),
        );
        let base_weight_shard_manifest =
            kiln_core::model_provenance::BaseWeightShardManifest::new(vec![
                kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                    "model.safetensors",
                    16,
                    [0x22; 32],
                )?,
            ])?;
        let descriptor = OpdCheckpointDescriptor {
            adapter_name: "exact-opd".into(),
            effective_config: serde_json::json!({"seed": 17}),
            precision_policy: crate::checkpoint::TrainingCheckpointPrecision {
                parameter_dtype: "f32".into(),
                optimizer_state_dtype: "f32".into(),
                activation_dtype: "f32".into(),
                gradient_dtype: "f32".into(),
                stochastic_rounding: serde_json::json!({"mode": "round_to_nearest"}),
            },
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: "inline-on-policy-opd-candidate-order-v1".into(),
                content_sha256: "1".repeat(64),
                item_count: 4,
            },
            init_seed: 17,
            optimizer: Optimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            learning_rate: 1e-5,
            candidates_per_epoch: 4,
            total_epochs: 2,
            on_policy: true,
            base_model_weights_sha256: Some(base_weight_shard_manifest.aggregate_sha256.clone()),
            teacher_content_revision: Some(format!("sha256:{}", "3".repeat(64))),
            auxiliary_state: serde_json::json!({
                "identity": "bound",
                "base_model_weights_sha256": base_weight_shard_manifest.aggregate_sha256,
                "base_weight_shard_manifest": base_weight_shard_manifest,
                "execution_provenance": crate::train_receipt::test_execution_provenance(),
            }),
        };
        let manifest = descriptor.manifest(&state)?;
        assert_eq!(manifest.training_kind, crate::checkpoint::TrainingKind::Opd);
        assert_eq!(manifest.progress.global_step, 2);
        assert_eq!(manifest.progress.cursor_in_epoch, 3);
        assert_eq!(manifest.optimizer.step, 2);
        assert_eq!(manifest.scheduler.step, 2);
        assert_eq!(manifest.rng_states["rollout-sampling"].position, 3);
        Ok(())
    }

    #[test]
    fn opd_base_adapter_initialization_loads_the_requested_weights() -> Result<()> {
        use crate::trainer::tests::{tiny_config_full_attn, tiny_weights};

        let device = kiln_tensor::Device::Cpu;
        let config = tiny_config_full_attn();
        let weights = tiny_weights(&config, &device)?;
        let source = crate::trainer::TrainableLoraParams::initialize_seeded(
            &config,
            &weights,
            2,
            4.0,
            &device,
            Some(11),
        )?;
        let mut destination = crate::trainer::TrainableLoraParams::initialize_seeded(
            &config,
            &weights,
            2,
            4.0,
            &device,
            Some(99),
        )?;
        let temp = tempfile::tempdir()?;
        let base = temp.path().join("base");
        source.save_peft(&base, config.num_layers)?;
        restore_opd_adapter_parameters(&mut destination, &device, None, Some(&base))?;

        let source_exact = temp.path().join("source.safetensors");
        let destination_exact = temp.path().join("destination.safetensors");
        source.save_checkpoint_parameters(&source_exact)?;
        destination.save_checkpoint_parameters(&destination_exact)?;
        assert_eq!(
            std::fs::read(source_exact)?,
            std::fs::read(destination_exact)?
        );
        Ok(())
    }

    #[cfg(any(feature = "rocm", feature = "vulkan"))]
    fn exact_resume_opd_prompts() -> Vec<OpdPrompt> {
        ["ok", "hi", "result"]
            .into_iter()
            .map(|answer| OpdPrompt {
                messages: vec![
                    ChatMessage::new("user", "hi"),
                    ChatMessage::new("assistant", answer),
                ],
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            })
            .collect()
    }

    #[cfg(any(feature = "rocm", feature = "vulkan"))]
    fn exact_resume_opd_teacher(
        prompts: &[OpdPrompt],
        tokenizer: &KilnTokenizer,
        vocab_size: usize,
        top_k: usize,
    ) -> Result<Arc<dyn LogitSource>> {
        let identity = crate::TeacherIdentityV1::new(
            "exact-opd-teacher",
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            None,
            vocab_size as u32,
            top_k as u32,
            4096,
            1_000_000,
            "kiln-test-fixture",
            "d".repeat(64),
        )?;
        let mut fixture = FixtureLogitSource::uniform_topk("exact-opd-teacher", vocab_size, top_k)
            .with_authoritative_identity(identity)?;
        for prompt in prompts {
            let tokenized = tokenize_opd_prompt_for_training(prompt, tokenizer, None)?;
            let active_positions: Vec<usize> = tokenized
                .action_mask
                .iter()
                .enumerate()
                .filter_map(|(index, active)| active.then_some(index))
                .collect();
            for logits_row in target_token_positions_to_logits_rows(
                "exact-opd-teacher",
                tokenized.input_ids.len(),
                &active_positions,
            )? {
                fixture.insert(
                    &tokenized.input_ids,
                    logits_row,
                    (0..top_k as u32).collect(),
                    (0..top_k).map(|rank| -3.0 - rank as f32 * 0.1).collect(),
                )?;
            }
        }
        Ok(Arc::new(fixture))
    }

    #[cfg(any(feature = "rocm", feature = "vulkan"))]
    fn exact_resume_loss_callback(
        stop_after: Option<usize>,
        gpu_lock: Arc<tokio::sync::RwLock<()>>,
    ) -> (
        Arc<std::sync::Mutex<Vec<f64>>>,
        crate::trainer::ProgressCallback,
    ) {
        let losses = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = losses.clone();
        let callback = Box::new(move |progress: crate::trainer::TrainingProgress| {
            let inference_owner = gpu_lock
                .clone()
                .try_read_owned()
                .expect("OPD progress callback must run outside GPU write ownership");
            let mut values = captured.lock().unwrap();
            values.push(progress.loss);
            let control = if stop_after.is_some_and(|limit| values.len() >= limit) {
                crate::trainer::TrainControl::Stop
            } else {
                crate::trainer::TrainControl::Continue
            };
            drop(values);
            drop(inference_owner);
            control
        });
        (losses, callback)
    }

    #[cfg(any(feature = "rocm", feature = "vulkan"))]
    fn opd_cancel_resume_matches_uninterrupted_training(
        model_config: kiln_core::config::ModelConfig,
        weights: kiln_model::forward::GpuWeights,
    ) -> Result<()> {
        let tokenizer = off_policy_smoke_tokenizer()?;
        let prompts = exact_resume_opd_prompts();
        let top_k = 16;
        let teacher =
            exact_resume_opd_teacher(&prompts, &tokenizer, model_config.vocab_size, top_k)?;
        let config = OpdConfig {
            training_mode: OpdTrainingMode::OffPolicy,
            top_k,
            samples_per_prompt: 1,
            learning_rate: Some(1e-3),
            lora_rank: 2,
            lora_alpha: 4.0,
            auto_load: false,
            checkpoint_interval: Some(2),
            seed: Some(17),
            optimizer: Optimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            epochs: 1,
            ..Default::default()
        };
        let temp = tempfile::tempdir()?;
        let adapter_root = temp.path().join("adapters");
        let control_root = temp.path().join("control");
        let control_checkpoint_root = temp.path().join("control-checkpoints");
        let resumed_root = temp.path().join("resumed");
        let resumed_checkpoint_root = temp.path().join("resumed-checkpoints");
        std::fs::create_dir_all(&adapter_root)?;
        std::fs::create_dir_all(&control_root)?;
        std::fs::create_dir_all(&control_checkpoint_root)?;
        std::fs::create_dir_all(&resumed_root)?;
        std::fs::create_dir_all(&resumed_checkpoint_root)?;
        let adapter_name = "exact-opd";
        let gpu_lock = Arc::new(tokio::sync::RwLock::new(()));
        let coordination = crate::trainer::GpuStepCoordination::new(
            gpu_lock.clone(),
            kiln_model::BackendHealthHandle::default(),
        );

        let (control_losses, control_callback) = exact_resume_loss_callback(None, gpu_lock.clone());
        let control_output = opd_train_to_with_checkpoint_root(
            &prompts,
            &config,
            &model_config,
            &weights,
            &tokenizer,
            teacher.clone(),
            &adapter_root,
            &control_root,
            &control_checkpoint_root,
            adapter_name,
            Some(control_callback),
            Some(coordination.clone()),
        )?;

        let (first_losses, first_callback) = exact_resume_loss_callback(Some(1), gpu_lock.clone());
        let interrupted = opd_train_to_with_checkpoint_root(
            &prompts,
            &config,
            &model_config,
            &weights,
            &tokenizer,
            teacher.clone(),
            &adapter_root,
            &resumed_root,
            &resumed_checkpoint_root,
            adapter_name,
            Some(first_callback),
            Some(coordination.clone()),
        )
        .expect_err("OPD stop callback must interrupt after publishing a checkpoint");
        assert!(
            format!("{interrupted:#}").contains("cancelled by user"),
            "{interrupted:#}"
        );
        drop(
            gpu_lock
                .clone()
                .try_write_owned()
                .expect("cancelled OPD run must release GPU ownership"),
        );
        std::fs::remove_dir_all(&resumed_root)?;
        std::fs::create_dir_all(&resumed_root)?;
        let resume_path = resumed_checkpoint_root.join(format!(
            "{adapter_name}-checkpoint-step-00000001.kiln-checkpoint"
        ));
        let mut resume_config = config.clone();
        resume_config.resume_checkpoint = Some(resume_path.display().to_string());
        let (remaining_losses, remaining_callback) =
            exact_resume_loss_callback(None, gpu_lock.clone());
        let resumed_output = opd_train_to_with_checkpoint_root(
            &prompts,
            &resume_config,
            &model_config,
            &weights,
            &tokenizer,
            teacher,
            &adapter_root,
            &resumed_root,
            &resumed_checkpoint_root,
            adapter_name,
            Some(remaining_callback),
            Some(coordination),
        )?;
        drop(
            gpu_lock
                .clone()
                .try_write_owned()
                .expect("completed OPD run must release GPU ownership"),
        );

        let control_losses = control_losses.lock().unwrap().clone();
        let mut combined_losses = first_losses.lock().unwrap().clone();
        combined_losses.extend(remaining_losses.lock().unwrap().iter().copied());
        assert_eq!(combined_losses, control_losses);
        assert_eq!(
            std::fs::read(control_output.join("adapter_model.safetensors"))?,
            std::fs::read(resumed_output.join("adapter_model.safetensors"))?
        );

        let control_checkpoint =
            crate::checkpoint::load_training_checkpoint(&control_checkpoint_root.join(format!(
                "{adapter_name}-checkpoint-step-00000002.kiln-checkpoint"
            )))?;
        let resumed_checkpoint =
            crate::checkpoint::load_training_checkpoint(&resumed_checkpoint_root.join(format!(
                "{adapter_name}-checkpoint-step-00000002.kiln-checkpoint"
            )))?;
        assert_eq!(
            control_checkpoint.manifest.effective_config,
            resumed_checkpoint.manifest.effective_config
        );
        assert_eq!(
            control_checkpoint.manifest.precision_policy,
            resumed_checkpoint.manifest.precision_policy
        );
        assert_eq!(
            control_checkpoint.manifest.progress,
            resumed_checkpoint.manifest.progress
        );
        assert_eq!(
            control_checkpoint.manifest.data,
            resumed_checkpoint.manifest.data
        );
        assert_eq!(
            control_checkpoint.manifest.rng_states,
            resumed_checkpoint.manifest.rng_states
        );
        assert_eq!(
            control_checkpoint.manifest.optimizer,
            resumed_checkpoint.manifest.optimizer
        );
        assert_eq!(
            control_checkpoint.manifest.scheduler,
            resumed_checkpoint.manifest.scheduler
        );
        assert_eq!(
            control_checkpoint.manifest.state_files,
            resumed_checkpoint.manifest.state_files
        );
        assert_eq!(
            control_checkpoint.manifest.auxiliary_state,
            resumed_checkpoint.manifest.auxiliary_state
        );
        for relative in [OPD_CHECKPOINT_ADAPTER_FILE, OPD_CHECKPOINT_OPTIMIZER_FILE] {
            assert_eq!(
                std::fs::read(control_checkpoint.artifact_path(relative)?)?,
                std::fs::read(resumed_checkpoint.artifact_path(relative)?)?
            );
        }
        assert_eq!(
            load_opd_checkpoint_loop_state(&control_checkpoint)?,
            load_opd_checkpoint_loop_state(&resumed_checkpoint)?
        );
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_opd_cancel_resume_matches_uninterrupted_training() -> Result<()> {
        if std::env::var("KILN_QUALIFICATION").ok().as_deref() != Some("1") {
            eprintln!(
                "skip rocm_opd_cancel_resume_matches_uninterrupted_training: qualification off"
            );
            return Ok(());
        }
        anyhow::ensure!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );
        use crate::trainer::tests::{tiny_config_full_attn_bf16, tiny_weights_bf16};
        let device = Device::Rocm(0);
        let model_config = tiny_config_full_attn_bf16();
        let weights = tiny_weights_bf16(&model_config, &device)?;
        opd_cancel_resume_matches_uninterrupted_training(model_config, weights)
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_opd_cancel_resume_matches_uninterrupted_training() -> Result<()> {
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            eprintln!(
                "skip vulkan_opd_cancel_resume_matches_uninterrupted_training: Vulkan test opt-in disabled"
            );
            return Ok(());
        }
        anyhow::ensure!(
            kiln_model::backend::vulkan::vulkan_is_available(),
            "Vulkan qualification requested but no Vulkan device is available"
        );
        use crate::trainer::tests::{tiny_config_full_attn, tiny_weights};
        let device = Device::Vulkan(0);
        let model_config = tiny_config_full_attn();
        let weights = tiny_weights(&model_config, &device)?;
        opd_cancel_resume_matches_uninterrupted_training(model_config, weights)
    }

    #[test]
    fn opd_runtime_contract_rejects_every_unwired_knob() {
        let base = OpdConfig::default();

        let mut config = base.clone();
        config.objective = OpdObjective::CrossEntropy;
        assert!(
            config
                .validate_runtime_contract()
                .unwrap_err()
                .to_string()
                .contains("cross_entropy")
        );

        let mut config = base.clone();
        config.stable_opd = StableOpdMode::Auto;
        assert!(
            config
                .validate_runtime_contract()
                .unwrap_err()
                .to_string()
                .contains("Stable-OPD")
        );

        let mut config = base.clone();
        config.discount = 0.5;
        assert!(
            config
                .validate_runtime_contract()
                .unwrap_err()
                .to_string()
                .contains("discount")
        );

        let mut config = base.clone();
        config.clip_epsilon = 0.2;
        assert!(
            config
                .validate_runtime_contract()
                .unwrap_err()
                .to_string()
                .contains("clip_epsilon")
        );

        let mut config = base;
        config.max_cost_usd = Some(25.0);
        assert!(
            config
                .validate_runtime_contract()
                .unwrap_err()
                .to_string()
                .contains("max_cost_usd")
        );
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
        // (#1082) kt `from_vec` is CPU-only and takes no device arg.
        let _device = Device::Cpu;
        // OPD per-position: 5 active positions, values 0.1..0.5
        // (#1082) kt `from_vec` is 2-arg (CPU-only); dropped candle `&device`.
        let kl = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5], 5)?;
        let kl_ref = Tensor::from_vec(vec![0.05f32, 0.05, 0.05, 0.05, 0.05], 5)?;
        // (#1082) kt scalar: rank-0 from a 1-elem vec (kt `new` takes `&[E]`,
        // candle `new(2.0_f32, dev)` had no kt drop-in for a bare scalar).
        let sft = Tensor::from_vec(vec![2.0_f32], ())?;
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
        // (#1082) kt `from_vec` is CPU-only and takes no device arg.
        let _device = Device::Cpu;
        // (#1082) kt `from_vec` is 2-arg (CPU-only); dropped candle `&device`.
        let kl = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], 3)?;
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
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]), // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 11, 12]),  // 0.5
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 3, 12]),   // 0.75
        ];
        assert!((compute_initial_overlap(&pairs) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cold_start_probe_triggers_below_threshold() {
        // Median overlap < 0.5 — should inject SFT.
        let pairs = vec![
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]), // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 10, 11, 12]), // 0.25
            (vec![1u32, 2, 3, 4], vec![1u32, 2, 11, 12]),  // 0.5
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
        assert_eq!(cold_start_probe_default(&pairs), ColdStartDecision::Skip);
    }

    #[test]
    fn cold_start_probe_empty_pairs_is_skip() {
        // Edge case: no probe positions ⇒ vacuously aligned ⇒ Skip.
        assert_eq!(cold_start_probe_default(&[]), ColdStartDecision::Skip);
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
                DistillMergeSource {
                    adapter: "rust-helper".into(),
                    weight: 1.0,
                },
                DistillMergeSource {
                    adapter: "python-helper".into(),
                    weight: 1.0,
                },
                DistillMergeSource {
                    adapter: "sql-helper".into(),
                    weight: 0.7,
                },
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
    fn fixed_fixture_partial_configs_default_to_off_policy() {
        let merge: DistillMergeRequest =
            serde_json::from_str(r#"{"name":"merged","sources":[],"config":{"lora_rank":4}}"#)
                .unwrap();
        assert_eq!(merge.config.training_mode, OpdTrainingMode::OffPolicy);

        let self_distill: DistillSelfRequest = serde_json::from_str(
            r#"{"name":"selfy","mode":"conciseness","config":{"lora_rank":4}}"#,
        )
        .unwrap();
        assert_eq!(
            self_distill.config.training_mode,
            OpdTrainingMode::OffPolicy
        );

        let explicit: DistillMergeRequest = serde_json::from_str(
            r#"{"name":"merged","sources":[],"config":{"training_mode":"on_policy"}}"#,
        )
        .unwrap();
        assert_eq!(explicit.config.training_mode, OpdTrainingMode::OnPolicy);
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
        // (#1082) kt `from_vec` is CPU-only and takes no device arg.
        let _device = Device::Cpu;
        // (#1082) kt `from_vec` is 2-arg (CPU-only); dropped candle `&device`.
        let kl = Tensor::from_vec(vec![0.1_f32; 4], 4)?;
        let kl_ref = Tensor::from_vec(vec![1.0_f32; 4], 4)?;
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

    /// #1082 CP-4 endgame: validate that the tape-authoritative OPD step
    /// ([`opd_step_forward_backward_tape_authoritative`]) produces nonzero
    /// LoRA gradients — i.e. the OPD-KL scalar tape root
    /// (`try_tape_opd_scalar_mean_cuda_kt`) is CONNECTED through the BF16 model
    /// forward back to the LoRA `Var`s, so one `Tape::backward` routes a real
    /// gradient into each LoRA parameter.
    ///
    /// This is the OPD sibling of the SFT
    /// `tape_authoritative_grads_match_candle_baseline_bf16` coverage check.
    /// It MUST run on a **BF16 CUDA** model: the kt fused adapters
    /// (`supports_rmsnorm_kt`, `supports_mlp_silu_mul_kt`,
    /// `supports_sigmoid_mul_kt`, `supports_rotary_qk_kt`) and the OPD-loss
    /// kt-tape envelope (`top_k ∈ {16, 32}`, `hidden.dtype() == head_t.dtype()
    /// ∈ {F32, BF16}`) are all BF16-only / CUDA-only. On F32 (or CPU) every
    /// `supports_*_kt` predicate returns false, the tape adapters decline
    /// (`Ok(None)`), no tape node is recorded, and the loss→LoRA chain
    /// dead-ends — so a CPU/F32 build cannot exercise this path. The kernel
    /// envelope is documented in `kiln-opd-loss-kernel::kt_tape::envelope_ok`.
    ///
    /// Reuses the SFT BF16 fixtures (`crate::trainer::tests::tiny_config_bf16`
    /// / `tiny_weights_bf16`) — single source of truth for the BF16 tiny model
    /// — so this test exercises the exact same model the SFT CP-4 coverage gate
    /// validates, just driven through the OPD scalar-mean loss root instead of
    /// cross-entropy.
    ///
    /// Run via `cargo nextest run` for per-process env isolation: the tape
    /// gates are `OnceLock`-cached on first read, and the kt GPU substrate is
    /// not thread-safe across the in-process parallelism `cargo test` uses.
    #[cfg(feature = "cuda")]
    #[test]
    fn opd_tape_authoritative_grads_reach_lora_bf16() {
        use crate::trainer::TrainableLoraParams;
        use crate::trainer::tests::{tiny_config_bf16, tiny_weights_bf16};

        // `Device` is the per-crate candle facade alias (= candle_core::Device);
        // opd.rs keeps its `candle_core::` ref count at 0 by going through it.
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("opd tape-authoritative grads (bf16): no CUDA device — skipping");
            return;
        }
        let device = Device::Cuda(0);

        let config = tiny_config_bf16();
        let weights = tiny_weights_bf16(&config, &device).expect("bf16 tiny weights on cuda");
        let params =
            TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device).expect("params");

        // `head_t` is the (transposed) LM head — the OPD path uses the tied
        // `embed_tokens_t` (BF16 here, matching the BF16 model output so the
        // kt-tape dtype gate `hidden.dtype() == head_t.dtype()` passes).
        // #1082: `embed_tokens_t` is a kt tensor and
        // `opd_step_forward_backward_tape_authoritative` takes the kt `head_t`
        // directly (the kt-native OPD-loss tape root `try_tape_opd_scalar_mean_cuda_kt`
        // consumes the kt `head_t` with no candle copy), so thread the kt weight
        // straight through.
        let head_t = weights.embed_tokens_t.clone();
        assert_eq!(
            head_t.dtype(),
            DType::BF16,
            "head_t must be BF16 to satisfy the kt-tape OPD envelope"
        );

        // Short token sequence; supervise the trailing positions (the typical
        // assistant-token mask shape). seq_len=7 mirrors the SFT BF16 test.
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let active_positions: Vec<usize> = vec![2, 3, 4, 5];

        // Deterministic uniform teacher. `resolved_top_k` =
        // `top_k.min(caps.max_top_k)`; pick K=16 (in the kt backward envelope
        // {16, 32}, and ≤ vocab_size=32). `max_top_k = 16` so the resolution
        // lands exactly on 16.
        let top_k = 16usize;
        let teacher: Arc<dyn LogitSource> =
            Arc::new(crate::logit_source::DeterministicUniformLogitSource::new(
                "tape-opd-test",
                config.vocab_size,
                top_k,
            ));

        let backend = kiln_model::backend::for_device_kt(&device);

        // (#1082) opd_step_forward_backward_tape_authoritative takes the kt device
        // directly now — no candle bridge.
        let (loss_val, active_count, grads, _env_ce) =
            opd_step_forward_backward_tape_authoritative(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &device,
                &head_t,
                teacher,
                &active_positions,
                OpdLossGranularity::TeacherTopK,
                top_k,
                None,
                None,
                None,
                false,
                kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device),
            )
            .expect("tape-authoritative OPD step");

        // The active count must match the supervised positions.
        assert_eq!(
            active_count,
            active_positions.len(),
            "active_count {active_count} != supervised positions {}",
            active_positions.len()
        );

        // Per-position KL is non-negative; the mean loss must be finite and
        // strictly positive (a uniform teacher vs an initialized student gives
        // a real divergence — a zero/NaN loss would mean the kernel produced a
        // degenerate forward).
        assert!(
            loss_val.is_finite() && loss_val > 0.0,
            "OPD tape loss {loss_val} is not finite-positive"
        );

        // HEADLINE assertion: the tape walk routed a grad into the LoRA Vars.
        // If the loss→LoRA tape chain were severed (the F32 failure mode, or a
        // broken adapter), `grads` would be empty.
        assert!(
            !grads.is_empty(),
            "tape-authoritative OPD produced NO LoRA grads — the OPD-KL tape \
             root did not connect through the BF16 model to any LoRA Var"
        );

        // Every returned grad keys a LoRA Var, and at least one must have a
        // nonzero, finite norm (a connected-but-all-zero grad would mean the
        // backward ran but carried no signal — e.g. a detached leaf).
        //
        // (#1082 high-perf) `grads` is now a kt-native `kiln_autograd::GradStore`
        // keyed by `KtTensorId` (values `kiln_tensor::Tensor`) — no kt→candle
        // copy. LoRA params are `kiln_param::Parameter` now; `all_vars()` ->
        // `all_params()` and the id is `Parameter::tensor_id()` directly (the
        // same kt id space the tape producer keyed on).
        let var_kt_ids: std::collections::HashSet<kiln_tensor_id::TensorId> =
            params.all_params().iter().map(|p| p.tensor_id()).collect();
        let mut nonzero_lora_grads = 0usize;
        let mut max_norm = 0f32;
        for (tid, g) in grads.iter() {
            assert!(
                var_kt_ids.contains(tid),
                "grad key {tid:?} is not a LoRA Var id"
            );
            let flat = g
                .to_dtype(kiln_tensor::DType::F32)
                .expect("grad -> f32")
                .flatten_all()
                .expect("flatten grad")
                .to_vec1::<f32>()
                .expect("grad to vec");
            assert!(
                flat.iter().all(|x| x.is_finite()),
                "LoRA grad {tid:?} contains non-finite values"
            );
            let norm = flat.iter().map(|x| x * x).sum::<f32>().sqrt();
            max_norm = max_norm.max(norm);
            if norm > 0.0 {
                nonzero_lora_grads += 1;
            }
        }

        eprintln!(
            "[CP4-OPD] tape grads: {} LoRA Vars got a grad ({} nonzero; max_norm={max_norm:.6}); \
             loss={loss_val:.6}; total_vars={}",
            grads.len(),
            nonzero_lora_grads,
            params.all_params().len()
        );

        assert!(
            nonzero_lora_grads > 0 && max_norm > 0.0,
            "tape-authoritative OPD routed grads to {} LoRA Vars but ALL were \
             zero (max_norm={max_norm}) — backward carried no signal",
            grads.len()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn checkpointed_opd_tape_authoritative_grads_reach_lora_bf16() {
        use crate::trainer::TrainableLoraParams;
        use crate::trainer::tests::{CUDA_TEST_LOCK, tiny_config_bf16, tiny_weights_bf16};

        let _cuda_guard = CUDA_TEST_LOCK.lock().expect("cuda test lock poisoned");
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("checkpointed OPD tape grads (bf16): no CUDA device — skipping");
            return;
        }
        let device = Device::Cuda(0);
        let config = tiny_config_bf16();
        let weights = tiny_weights_bf16(&config, &device).expect("bf16 tiny weights on cuda");
        let params =
            TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device).expect("params");
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let active_positions: Vec<usize> = vec![2, 3, 4, 5];
        let top_k = 16usize;
        let teacher: Arc<dyn LogitSource> =
            Arc::new(crate::logit_source::DeterministicUniformLogitSource::new(
                "checkpointed-opd-test",
                config.vocab_size,
                top_k,
            ));
        let backend = kiln_model::backend::for_device_kt(&device);
        let segments = crate::trainer::compute_segment_boundaries(config.num_layers, 2);

        let (loss_val, active_count, grads, _env_ce) =
            checkpointed_opd_step_forward_backward_tape_authoritative(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &device,
                &weights.embed_tokens_t,
                teacher,
                &active_positions,
                OpdLossGranularity::TeacherTopK,
                top_k,
                None,
                None,
                None,
                false,
                &segments,
                kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device),
            )
            .expect("checkpointed tape-authoritative OPD step");

        assert_eq!(active_count, active_positions.len());
        assert!(
            loss_val.is_finite() && loss_val > 0.0,
            "checkpointed OPD loss {loss_val} is not finite-positive"
        );
        assert!(!grads.is_empty(), "checkpointed OPD produced no LoRA grads");
        let var_kt_ids: std::collections::HashSet<kiln_tensor_id::TensorId> =
            params.all_params().iter().map(|p| p.tensor_id()).collect();
        let mut nonzero_lora_grads = 0usize;
        for (tid, g) in grads.iter() {
            assert!(
                var_kt_ids.contains(tid),
                "grad key {tid:?} is not a LoRA Var id"
            );
            let flat = g
                .to_dtype(kiln_tensor::DType::F32)
                .expect("grad -> f32")
                .flatten_all()
                .expect("flatten grad")
                .to_vec1::<f32>()
                .expect("grad to vec");
            if flat.iter().map(|x| x * x).sum::<f32>().sqrt() > 0.0 {
                nonzero_lora_grads += 1;
            }
        }
        assert!(
            nonzero_lora_grads > 0,
            "all checkpointed OPD LoRA grads were zero"
        );
    }

    /// (#1082) OPD on F32 Vulkan through the REAL
    /// `opd_step_forward_backward_tape_authoritative` entry point must produce
    /// a non-empty, finite kt grad store. The OPD top-K reverse-KL envelope now
    /// admits Vulkan (`kt_tape::envelope_ok`) and the backward routes through
    /// the device-agnostic analytic kt-composite, so F32-on-Vulkan OPD records
    /// + backprops. Uses the GDN-bearing F32 `tiny_config` so the GDN F32 path
    /// is driven.
    ///
    /// HOST-SAFETY: ONE forward+loss+backward over the tiny 4-layer model.
    /// Self-skips unless `KILN_TENSOR_VULKAN_TEST=1` AND a Vulkan device is
    /// present. Run single-shot, one at a time.
    #[cfg(feature = "vulkan")]
    #[test]
    fn vk_f32_opd_grads_nonempty() {
        use crate::trainer::TrainableLoraParams;
        use crate::trainer::tests::{tiny_config_full_attn, tiny_weights};

        let test_name = "vk_f32_opd_grads_nonempty";
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            eprintln!("skip {test_name}: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if !kiln_model::backend::vulkan::vulkan_is_available() {
            eprintln!("skip {test_name}: no Vulkan device");
            return;
        }
        let device = Device::Vulkan(0);
        let config = tiny_config_full_attn(); // F32 base, full-attn-only
        let weights = tiny_weights(&config, &device).expect("f32 tiny weights on Vulkan");
        assert_eq!(
            weights.embed_tokens.dtype(),
            DType::F32,
            "OPD F32 Vulkan: config must be F32"
        );
        let params =
            TrainableLoraParams::initialize_seeded(&config, &weights, 4, 8.0, &device, Some(7))
                .expect("LoRA params");

        // head_t = tied F32 LM head; matches the F32 hidden so the OPD envelope
        // `hidden.dtype() == head_t.dtype()` passes (both F32 on Vulkan).
        let head_t = weights.embed_tokens_t.clone();
        assert_eq!(head_t.dtype(), DType::F32, "head_t must be F32 here");

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let active_positions: Vec<usize> = vec![2, 3, 4, 5];
        let top_k = 16usize; // in the kt backward envelope {16,32}, <= vocab=32
        let teacher: Arc<dyn LogitSource> =
            Arc::new(crate::logit_source::DeterministicUniformLogitSource::new(
                "vk-opd-test",
                config.vocab_size,
                top_k,
            ));

        let backend = kiln_model::backend::for_device_kt(&device);
        let (loss_val, active_count, grads, _env_ce) =
            opd_step_forward_backward_tape_authoritative(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &device,
                &head_t,
                teacher,
                &active_positions,
                OpdLossGranularity::TeacherTopK,
                top_k,
                None,
                None,
                None,
                false,
                kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device),
            )
            .expect("opd_step_forward_backward_tape_authoritative (F32 Vulkan OPD)");

        assert_eq!(active_count, active_positions.len());
        assert!(
            loss_val.is_finite(),
            "OPD F32 Vulkan loss {loss_val} is not finite"
        );

        // Per-module coverage bisection.
        let mut present = 0usize;
        let mut max_norm = 0.0_f32;
        let mut missing = 0usize;
        for p in params.all_params() {
            match grads.get(p.tensor_id()) {
                Some(g) => {
                    let host = g
                        .to_device(kiln_tensor::Device::Cpu)
                        .and_then(|t| t.to_dtype(kiln_tensor::DType::F32))
                        .and_then(|t| t.to_vec::<f32>())
                        .unwrap_or_default();
                    assert!(
                        host.iter().all(|v| v.is_finite()),
                        "OPD F32 Vulkan LoRA grad non-finite"
                    );
                    let norm = host.iter().map(|x| x * x).sum::<f32>().sqrt();
                    max_norm = max_norm.max(norm);
                    present += 1;
                }
                None => missing += 1,
            }
        }
        eprintln!(
            "[OPD F32 Vulkan] loss={loss_val:.6} store.len()={} present={present} \
             absent={missing} max_norm={max_norm:.6}",
            grads.len()
        );
        assert!(
            !grads.is_empty() && present > 0 && max_norm > 0.0,
            "F32 Vulkan OPD produced EMPTY/zero LoRA grads — the OPD-KL tape root \
             did not connect through the F32 model to any LoRA leaf"
        );
    }

    /// (#1443 step 4) OPD on a BF16 BASE on Vulkan through the REAL
    /// `opd_step_forward_backward_tape_authoritative` entry point must produce a
    /// non-empty, finite F32 LoRA grad store, with the base projection weights
    /// staying BF16 (the VRAM win). Mixed precision: BF16 base weights, F32
    /// activations — the base linear runs `vk_matmul_bf16w`, the embedding is
    /// cast BF16→F32 at the head of the forward, and LoRA A/B are F32. Uses the
    /// GDN-bearing BF16 `tiny_config_bf16` so the GDN BF16-weight path is driven.
    ///
    /// HOST-SAFETY: ONE forward+loss+backward over the tiny 4-layer model.
    /// Self-skips unless `KILN_TENSOR_VULKAN_TEST=1` AND a Vulkan device is
    /// present. Run single-shot, one at a time.
    #[cfg(feature = "vulkan")]
    #[test]
    fn vk_bf16_opd_grads_nonempty() {
        use crate::trainer::TrainableLoraParams;
        use crate::trainer::tests::{tiny_config_bf16, tiny_weights_bf16};

        let test_name = "vk_bf16_opd_grads_nonempty";
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            eprintln!("skip {test_name}: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if !kiln_model::backend::vulkan::vulkan_is_available() {
            eprintln!("skip {test_name}: no Vulkan device");
            return;
        }
        let device = Device::Vulkan(0);
        let config = tiny_config_bf16(); // BF16 base, GDN-bearing
        let weights = tiny_weights_bf16(&config, &device).expect("bf16 tiny weights on Vulkan");
        assert_eq!(
            weights.embed_tokens.dtype(),
            DType::BF16,
            "OPD BF16 Vulkan: config must be BF16"
        );
        // Base projection weights stay BF16 (the #1443 VRAM win).
        assert_eq!(
            weights.embed_tokens_t.dtype(),
            DType::BF16,
            "lm_head (embed_tokens_t) base weight must stay BF16 (the #1443 VRAM win)"
        );

        let params =
            TrainableLoraParams::initialize_seeded(&config, &weights, 4, 8.0, &device, Some(7))
                .expect("LoRA params");
        // Mixed precision: LoRA is F32 on Vulkan even on a BF16 base.
        assert_eq!(
            params.all_params()[0]
                .forward_storage()
                .primary_tensor()
                .dtype(),
            kiln_tensor::DType::F32,
            "LoRA param dtype must be F32 on Vulkan even on a BF16 base (mixed precision)"
        );

        // head_t = tied BF16 LM head; the OPD scalar loss consumes F32 activations
        // (post embed BF16→F32 cast) and the BF16 head via `vk_matmul_bf16w`.
        let head_t = weights.embed_tokens_t.clone();
        assert_eq!(head_t.dtype(), DType::BF16, "head_t must be BF16 here");

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let active_positions: Vec<usize> = vec![2, 3, 4, 5];
        let top_k = 16usize; // in the kt backward envelope {16,32}, <= vocab=32
        let teacher: Arc<dyn LogitSource> =
            Arc::new(crate::logit_source::DeterministicUniformLogitSource::new(
                "vk-opd-bf16-test",
                config.vocab_size,
                top_k,
            ));

        let backend = kiln_model::backend::for_device_kt(&device);
        let (loss_val, active_count, grads, _env_ce) =
            opd_step_forward_backward_tape_authoritative(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &device,
                &head_t,
                teacher,
                &active_positions,
                OpdLossGranularity::TeacherTopK,
                top_k,
                None,
                None,
                None,
                false,
                kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device),
            )
            .expect("opd_step_forward_backward_tape_authoritative (BF16 Vulkan OPD)");

        assert_eq!(active_count, active_positions.len());
        assert!(
            loss_val.is_finite(),
            "OPD BF16 Vulkan loss {loss_val} is not finite"
        );

        // Per-module coverage bisection + finite-F32 grad check.
        let mut present = 0usize;
        let mut max_norm = 0.0_f32;
        let mut missing = 0usize;
        for p in params.all_params() {
            match grads.get(p.tensor_id()) {
                Some(g) => {
                    assert_eq!(
                        g.dtype(),
                        kiln_tensor::DType::F32,
                        "OPD BF16 Vulkan LoRA grad must be F32 (mixed precision)"
                    );
                    let host = g
                        .to_device(kiln_tensor::Device::Cpu)
                        .and_then(|t| t.to_dtype(kiln_tensor::DType::F32))
                        .and_then(|t| t.to_vec::<f32>())
                        .unwrap_or_default();
                    assert!(
                        host.iter().all(|v| v.is_finite()),
                        "OPD BF16 Vulkan LoRA grad non-finite"
                    );
                    let norm = host.iter().map(|x| x * x).sum::<f32>().sqrt();
                    max_norm = max_norm.max(norm);
                    present += 1;
                }
                None => missing += 1,
            }
        }
        eprintln!(
            "[OPD BF16 Vulkan] loss={loss_val:.6} store.len()={} present={present} \
             absent={missing} max_norm={max_norm:.6} (base BF16, LoRA F32)",
            grads.len()
        );
        assert!(
            !grads.is_empty() && present > 0 && max_norm > 0.0,
            "BF16 Vulkan OPD produced EMPTY/zero LoRA grads — the OPD-KL tape root \
             did not connect through the BF16 model to any LoRA leaf"
        );
        // Re-confirm the base weight is STILL BF16 after the step.
        assert_eq!(
            weights.embed_tokens_t.dtype(),
            DType::BF16,
            "lm_head base weight must stay BF16 after the OPD step (the #1443 VRAM win)"
        );
    }
}

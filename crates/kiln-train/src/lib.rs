//! Training for Kiln — pure Rust, in-process LoRA SFT and GRPO.
//!
//! This crate defines the training API types AND implements the actual training
//! loop using candle autograd. Training runs in the same process as inference,
//! operating on the already-loaded model weights. No Python sidecar needed.

pub mod adapter_output;
pub mod adapter_shape;
// (#1082) Per-crate candle facade — every `candle_core::` path that
// `trainer.rs` previously held inline (type aliases, generic constructor
// helpers, safetensors I/O shims, `cd_bail!` macro) now lives in this
// module. Keeps `trainer.rs` at one direct candle reference (the
// `CustomOp1` trait impl, which cannot be type-aliased on stable Rust).
pub(crate) mod cd_types;
#[cfg(feature = "cuda")]
pub mod cuda_train;
pub mod diagnostics;
// (#1082) `pub mod echo;` deleted: ECHO's only caller was the OPD candle
// gradient-checkpointing path (`opd_step_forward_backward_candle`), which
// the candle-drop removed. ECHO's FLCE composite has no kt-tape coverage,
// so it dropped out with the candle path. Re-add a kt-native ECHO module
// when the FLCE env-CE term gets a tape adapter.
// (#1082) Candle↔kt boundary for the SFT/FLCE trainer — relocated out of
// `kiln-flce-kernel` so that kernel crate became 100% candle-free (the
// THIRD kernel-crate candle drop, after `kiln-opd-loss-kernel` and
// `kiln-rmsnorm-kernel`). Holds the candle-typed `FlceMatmulProvider` trait,
// the pure-candle Phase A reference, the Phase B candle `CustomOp1`, the
// (#1082 candle-drop) `flce_candle_shim` deleted — the candle FLCE CustomOp +
// `FlceMatmulProvider` (the `KILN_CUDA_FLCE` provider opt-in) are gone; FLCE is
// kt-native via `kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_kt`.
// (#1082) CP-4: GRPO policy-gradient scalar-loss tape root. The
// candle↔kt boundary for the GRPO trainer's tape-authoritative path —
// a single fused tape node taking the full `[1, T, V]` policy logits and
// producing the scalar PG (+ optional KL) loss, whose backward recomputes
// the candle GRPO forward with autograd ON to yield `dL/dlogits`. Mirrors
// `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda` (the
// SFT loss root) and `opd_tape_shim::try_tape_opd_scalar_mean_cuda` (OPD).
pub mod grpo_tape_shim;
pub mod logit_cache;
pub mod logit_source;
pub mod long_context_fixture;
pub mod lora_scaling;
pub mod opd;
// (#1082) Candle↔kt boundary for the OPD trainer — relocated out of
// `kiln-opd-loss-kernel` so that kernel crate became 100% candle-free
// (the first kernel-crate candle drop). Holds the pure-candle Phase A
// reference path, the candle `CustomOp1`-based kt-forward-op shim, and
// the kt-tape production-caller adapters. See module docstring. (#1082)
pub mod opd_tape_shim;
pub mod pi_trajectory;
pub mod receipt;
pub mod remote_teacher;
pub mod replay;
pub mod sft_tape_shim;
// CP-4 substrate pilot — `kiln_autograd::Tape`-based parallel training
// entry. Sits alongside the candle-typed `trainer` module so future PRs
// can extend it to cover the full per-step graph. See module docstring
// + `docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`. (#1082)
pub mod tape_step;
pub mod train_receipt;
pub mod trainer;
pub mod trajectory;
pub mod trajectory_inspect;
pub mod trajectory_mask;

pub use logit_cache::{CacheEntry, CacheStats, CachedLogitSource, LogitCache, hash_prefix};
pub use receipt::{
    AdapterReceipt, DiagnosticSummary, PromptSourceDescriptor, RECEIPT_SCHEMA_VERSION,
    TeacherDescriptor,
};
pub use remote_teacher::{CostTally, RemoteProvider, RemoteTeacher, RemoteTeacherConfig};

pub use adapter_output::{
    ADAPTER_MANIFEST_FILENAME, ADAPTER_MANIFEST_SCHEMA_VERSION, ADAPTER_RECEIPT_FILENAME,
    AdapterManifest, AdapterManifestFiles, AdapterOutputReceipt, AdapterRestoreOptions,
    AdapterRestoreReceipt, ResolvedSftOutputLayout, install_adapter_symlink,
    read_adapter_manifest, read_adapter_manifest_from_adapter_dir,
    resolve_sft_output_layout, restore_adapter_from_manifest,
    validate_adapter_output_dir, validate_install_adapter_name,
    write_adapter_manifest_from_train_receipt, write_adapter_output_receipt,
};
pub use adapter_shape::{
    ALLOW_ADAPTER_SHAPE_CONVERSION_FLAG, BaseAdapterCompatibility, TRAINABLE_TARGET_MODULES,
    resolve_base_adapter_dir, validate_base_adapter_compatibility,
};
pub use diagnostics::{
    DIVERSITY_COLLAPSE_THRESHOLD, DIVERSITY_COLLAPSE_WINDOW, GuardrailDecision, GuardrailTrigger,
    LengthInflationGuardrail, OpdDiagnosticSnapshot, REPETITION_GUARDRAIL_THRESHOLD,
    RolloutSummary, SELF_PLAY_SATURATION_THRESHOLD, SELF_PLAY_SATURATION_WINDOW, build_snapshot,
    repetition_rate, rollout_diversity, truncation_rate,
};
pub use logit_source::{
    DeterministicUniformLogitSource, LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch,
    TopKLogprobs,
};
pub use lora_scaling::{
    ALLOW_HIGH_LORA_SCALE_FLAG, MAX_LORA_ALPHA_OVER_RANK, alpha_over_rank, validate_lora_scaling,
};
pub use opd::{
    AgenticLossInputs, AgenticLossWeights, COLD_START_DEFAULT_EPOCHS, COLD_START_DEFAULT_PROMPTS,
    COLD_START_OVERLAP_THRESHOLD, ColdStartDecision, DistillMergeRequest, DistillMergeSource,
    DistillPumpMode, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest,
    NewKnowledgeSource, OffPolicyDistillationExample, OffPolicyDistillationSummary,
    OffPolicyLossBreakdown, OpdConfig, OpdLossGranularity, OpdObjective, OpdPrompt, OpdRequest,
    OpdTrainingMode, PreparedOffPolicyDistillation, SelfDistillMode, StableOpdCoefficients,
    StableOpdLossInputs, StableOpdLossOutputs, TeacherActionToken, TeacherTopLogprob,
    TipTokenClass, cold_start_probe, cold_start_probe_default,
    compose_off_policy_distillation_loss, compute_agentic_loss_weights, compute_initial_overlap,
    compute_stable_opd_loss, default_beta_kl, default_lambda_sft, default_lambda_verifier,
    default_opd_samples_per_prompt, default_opd_top_k, default_score_decay_steps,
    default_score_earliest_weight, default_tip_tool_call_weight, default_tip_tool_name_weight,
    load_off_policy_distillation_jsonl, parse_off_policy_distillation_jsonl_str,
    prepare_off_policy_distillation_dataset,
};
pub use train_receipt::{
    ADAPTER_CANARY_STATUS_FILENAME, AdapterCanaryCheckReceipt, AdapterCanaryState,
    AdapterCanaryStatusReceipt, AdapterSmokePromptDiagnosis, AdapterSmokePromptDiagnosisReceipt,
    AdapterSmokePromptReceipt, AdapterSmokeTestReceipt, OpdReceipt, TRAIN_RECEIPT_FILENAME,
    TRAIN_RECEIPT_SCHEMA_VERSION, TrainFailureReason, TrainReceipt, TrainReceiptStatus,
    read_adapter_canary_status_from_adapter_dir,
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
    #[serde(default)]
    pub examples: Vec<SftExample>,
    /// Optional name of an uploaded dataset (the eval dataset store) to train
    /// on instead of inline `examples`. The server resolves the name and reads
    /// every row — callers never round-trip rows through the client, so large
    /// datasets train whole (no preview-endpoint truncation) and the request
    /// stays a few bytes. Mutually exclusive with `examples`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
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
/// `Muon` (Bernstein-Newhouse 2024 / Jordan et al. 2024) is the
/// **default**: momentum-orthogonalized SGD. It keeps a single per-param
/// heavy-ball momentum buffer, takes a (Nesterov) look-ahead, and — for
/// the rank-2 LoRA A/B weight matrices — projects it onto the nearest
/// semi-orthogonal matrix via a Newton-Schulz iteration before stepping,
/// then rescales by `sqrt(max(rows, cols))` so the update magnitude is
/// shape-independent. Dispatched on-device via `dispatch_muon_step`
/// (fused per-matrix Newton-Schulz kernels — CUDA / ROCm / Vulkan /
/// Metal) when operands are resident; otherwise the `kiln_optim::Muon`
/// CPU reference. Muon converges LoRA fine-tunes in fewer steps than
/// AdamW at roughly half the optimizer state (one momentum buffer vs
/// Adam's m+v).
///
/// `AdamW` is decoupled-weight-decay Adam (Loshchilov & Hutter 2019);
/// dispatched on-device via `dispatch_adamw_step` when the backend
/// supports residency. The trainer allocates per-parameter first/second
/// moment tensors at init, registers them in the resident-activation
/// registry alongside the param/grad, and updates all three in-place per
/// step. Select with `{"optimizer": {"kind": "adam_w"}}`.
///
/// `Sgd` is plain stochastic gradient descent (`param -= lr * grad`);
/// dispatched on-device via `dispatch_sgd_step` when the backend
/// supports residency. Select with `{"optimizer": {"kind": "sgd"}}`.
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
    /// Momentum-orthogonalized SGD (the default). `momentum` is the
    /// heavy-ball coefficient; `nesterov` toggles the look-ahead;
    /// `ns_iters` is the Newton-Schulz iteration count (paper uses 5);
    /// `weight_decay` is decoupled.
    Muon {
        #[serde(default = "default_muon_momentum")]
        momentum: f32,
        #[serde(default = "default_muon_nesterov")]
        nesterov: bool,
        #[serde(default = "default_muon_ns_iters")]
        ns_iters: u32,
        #[serde(default = "default_muon_weight_decay")]
        weight_decay: f32,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Muon {
            momentum: default_muon_momentum(),
            nesterov: default_muon_nesterov(),
            ns_iters: default_muon_ns_iters(),
            weight_decay: default_muon_weight_decay(),
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
fn default_muon_momentum() -> f32 {
    0.95
}
fn default_muon_nesterov() -> bool {
    true
}
fn default_muon_ns_iters() -> u32 {
    5
}
fn default_muon_weight_decay() -> f32 {
    0.0
}

/// Which training mode a learning rate is being resolved for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainMode {
    Sft,
    Grpo,
    Opd,
}

/// Per-optimizer learning-rate default, used when a config omits
/// `learning_rate`.
///
/// AdamW and SGD keep the legacy defaults (SFT 1e-4, GRPO/OPD 1e-5).
/// Muon's orthogonalized, RMS-matched update is ~unit scale and wants a
/// much larger step: SFT uses `kiln_optim::Muon::default_hp()`'s 2e-2,
/// and GRPO/OPD scale the legacy AdamW SFT:GRPO ratio (10x down) by the
/// same factor. The Muon GRPO/OPD band is an initial heuristic pending an
/// empirical sweep; train receipts record the resolved value, so every
/// run stays auditable either way.
pub fn resolve_learning_rate(optimizer: &Optimizer, mode: TrainMode) -> f64 {
    match (optimizer, mode) {
        (Optimizer::Muon { .. }, TrainMode::Sft) => 2e-2,
        (Optimizer::Muon { .. }, TrainMode::Grpo | TrainMode::Opd) => 2e-3,
        (_, TrainMode::Sft) => 1e-4,
        (_, TrainMode::Grpo | TrainMode::Opd) => 1e-5,
    }
}

/// Warn when an explicit learning rate sits far outside the selected
/// optimizer's band. The classic failure is an AdamW-era value (1e-4)
/// submitted to a Muon run (band 2e-2) — it trains 200x too cold and
/// silently produces a near-no-op adapter. Returns `None` inside the
/// 50x band; ordinary tuning (a few x either way) never trips it.
pub fn learning_rate_band_warning(explicit: f64, resolved_default: f64) -> Option<String> {
    const BAND_RATIO: f64 = 50.0;
    if !explicit.is_finite() || explicit <= 0.0 || resolved_default <= 0.0 {
        return None;
    }
    let ratio = explicit / resolved_default;
    if ratio > BAND_RATIO {
        Some(format!(
            "learning_rate {explicit:e} is {ratio:.0}x larger than this optimizer's \
             default {resolved_default:e} — the run may diverge; omit learning_rate \
             to use the per-optimizer default"
        ))
    } else if ratio < 1.0 / BAND_RATIO {
        Some(format!(
            "learning_rate {explicit:e} is {:.0}x smaller than this optimizer's \
             default {resolved_default:e} — the run may train too cold to matter; \
             omit learning_rate to use the per-optimizer default",
            1.0 / ratio
        ))
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftConfig {
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Learning rate. `None` (the default) resolves per optimizer at run
    /// start — see [`resolve_learning_rate`]. Explicit values are used
    /// verbatim; the train receipt records whichever value actually ran.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    /// If set, continue training from this adapter instead of starting fresh.
    pub base_adapter: Option<String>,
    /// Reserved escape hatch for future explicit shape conversion. Today,
    /// incompatible base adapters still fail before optimizer setup.
    #[serde(default)]
    pub allow_adapter_shape_conversion: bool,
    /// Permit alpha/rank above the default safety limit for deliberate
    /// experiments.
    #[serde(default)]
    pub allow_high_lora_scale: bool,
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
    /// Optimizer selection. Defaults to Muon (momentum-orthogonalized SGD
    /// with fused on-device Newton-Schulz). AdamW remains available via
    /// `{"optimizer": {"kind": "adam_w"}}` and plain SGD via
    /// `{"optimizer": {"kind": "sgd"}}` for backwards-compatible runs.
    #[serde(default)]
    pub optimizer: Optimizer,
    /// After successful training, run a small base-vs-adapter canary check and
    /// record the result in `train_receipt.json`.
    #[serde(default)]
    pub adapter_smoke_test: bool,
}

fn default_auto_load() -> bool {
    true
}
fn default_epochs() -> usize {
    3
}
fn default_rank() -> usize {
    16
}
fn default_alpha() -> f32 {
    32.0
}

impl SftConfig {
    /// The explicit `learning_rate` when given, else the per-optimizer
    /// default for SFT.
    pub fn effective_learning_rate(&self) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| resolve_learning_rate(&self.optimizer, TrainMode::Sft))
    }
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            learning_rate: None,
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            allow_adapter_shape_conversion: false,
            allow_high_lora_scale: false,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            seed: None,
            optimizer: Optimizer::default(),
            adapter_smoke_test: false,
        }
    }
}

// ScoredCompletion and GrpoGroup are now re-exports of the canonical
// trajectory-aware types defined in `crate::trajectory`. The old field
// shape (`{text, reward}` / `{messages, completions}`) is preserved
// byte-identical, plus an optional `trajectory` field on each rollout
// that ECHO consumes. Legacy callers see no field-name change.
//
// See `docs/plans/echo-integration-plan.md` §2 and §B.1 for the design.

pub use crate::trajectory::{AgenticGroup, ScoredRollout, TurnKind, TurnSegment};

/// Legacy alias for [`ScoredRollout`]. Use the canonical name in new code.
pub type ScoredCompletion = ScoredRollout;

/// Legacy alias for [`AgenticGroup`]. Use the canonical name in new code.
pub type GrpoGroup = AgenticGroup;

/// Request to run a GRPO training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoRequest {
    /// Accepts `agentic_groups` as an alias so the /v1/train/agentic
    /// route's JSON body uses the semantically-meaningful name. Legacy
    /// clients posting `groups` continue to deserialize unchanged.
    #[serde(default, alias = "agentic_groups")]
    pub groups: Vec<GrpoGroup>,
    /// Optional server-local JSONL dataset path. Each non-empty line is one
    /// `GrpoGroup`. Used by Vulkan-native GRPO to stream large datasets without
    /// retaining every group in memory.
    #[serde(default)]
    pub dataset_path: Option<String>,
    /// Optional name of an uploaded dataset (the eval dataset store) to train
    /// on instead of inline `groups` / a raw path. The server resolves the
    /// name to its on-disk JSONL and streams it like `dataset_path`. Mutually
    /// exclusive with both `groups` and `dataset_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default)]
    pub config: GrpoConfig,
    /// Optional auto-eval hook (see `SftRequest::post_eval`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Group-relative advantage normalization mode.
///
/// `Vanilla` is the original DeepSeekMath / R1 form: `A_i = (r_i - mean(r)) /
/// (std(r) + eps)`. `DrGrpo` follows arXiv:2503.20783 ("Understanding
/// R1-Zero-Like Training") and drops the `std` normalization to eliminate
/// question-difficulty bias and the numerical blow-up when within-group
/// rewards are nearly uniform.
///
/// Default: `DrGrpo`. Phase 1 ablation on Qwen3.5-4B (the kiln backbone)
/// showed clean training under both modes; dropping std normalization is a
/// strict improvement at the math level (removes the small-std blow-up and
/// the question-difficulty bias from arXiv:2503.20783) with zero compute
/// cost, so we ship Dr. GRPO as the default.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdvantageMode {
    /// `A_i = (r_i - mean(r)) / (std(r) + 1e-8)`. The original DeepSeekMath /
    /// R1 form; retained for ablation and back-compat.
    Vanilla,
    /// `A_i = r_i - mean(r)`. Recommended by Dr. GRPO (arXiv:2503.20783) and
    /// the SimpleRL-Zoo replication (arXiv:2503.18892). The kiln default
    /// from Phase 1 onward.
    #[default]
    DrGrpo,
}

/// How the GRPO surrogate loss is aggregated across the completions in a group.
///
/// `PerSample` is the original DeepSeekMath / R1 form: each completion's loss
/// is the per-token mean, then the group reports the mean across completions,
/// and the optimizer steps once per completion. It systematically
/// under-penalizes long incorrect completions and is the documented source of
/// the GRPO "length-drift" failure mode.
///
/// `TokenLevel` is the DAPO Token-Level Policy Gradient Loss (arXiv:2503.14476):
/// per-token surrogates are accumulated across all completions in the group
/// and divided by the total active completion token count, with a single
/// optimizer step per group. Removes per-sample length bias.
///
/// Default: `TokenLevel`. Phase 1 ablation on Qwen3.5-4B showed a ~2× speedup
/// over `PerSample` (one optimizer step per group, plus dynamic-sampling
/// filtering) with clean loss curves; combined with the literature consensus
/// against per-sample averaging, this becomes the kiln default. The
/// vk-native path falls back to `PerSample` (kernel work pending).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LossAggregation {
    /// Per-completion mean-of-tokens, per-completion optimizer step. The
    /// original DeepSeekMath / R1 form; retained for ablation and as the
    /// vk-native fallback.
    PerSample,
    /// DAPO Token-Level Loss: sum-over-all-tokens-in-group divided by
    /// total active tokens, single optimizer step per group. The kiln
    /// default from Phase 1 onward (candle path).
    #[default]
    TokenLevel,
}

/// How the importance-sampling ratio is computed and clipped.
///
/// `Token` is the historical kiln behavior: the IS ratio is per-token, the
/// PPO surrogate clips per-token, and clipped tokens drop out of the
/// gradient. The DeepSeekMath / R1 default.
///
/// `Sequence` implements GSPO (arXiv:2507.18071): the ratio is computed once
/// per sequence as `(π_θ(y) / π_old(y))^(1/|y|)`, clip and surrogate live at
/// the sequence level, and every token in the sequence sees the same scalar
/// gradient coefficient `s · d_surrogate / |y|`. Originally proposed for MoE
/// (Qwen3-MoE production); kiln's Qwen3.5-4B is dense so the MoE argument
/// does not apply, but the variance-reduction argument on long CoT may.
/// Treat as a controlled experiment until ablated on the actual workload.
///
/// `Cispo` implements CISPO (MiniMax-M1, arXiv:2506.13585): per-token gradient
/// flow with the IS *weight* clipped (not the surrogate). Every token
/// contributes a gradient; the clipped IS weight bounds variance. Less
/// invasive than GSPO and keeps per-token gradient diversity.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IsLevel {
    /// Historical per-token IS with PPO min(surrogate, clipped_surrogate).
    #[default]
    Token,
    /// GSPO sequence-level IS (arXiv:2507.18071).
    Sequence,
    /// CISPO clipped-weight IS (arXiv:2506.13585).
    Cispo,
}

/// What policy the reference forward uses, and how often it refreshes.
///
/// `BasePerStep` reproduces the historical kiln behavior: the reference is
/// the base model (no LoRA), recomputed for every completion. The IS ratio
/// is therefore `π_θ / π_base` and the KL term anchors the policy to the
/// base model.
///
/// `None` skips the reference forward entirely and uses a fixed log-ratio
/// of 0 for every token. Combined with `KlEstimator::None` this reduces
/// GRPO to pure REINFORCE with group-relative advantages — useful as an
/// ablation baseline. The reference forward is the most expensive single
/// component of a kiln GRPO step, so this mode is also a meaningful
/// speedup when the KL anchor is not load-bearing for stability.
///
/// `Ema` snapshots the LoRA-applied policy every `refresh_every` optimizer
/// steps into a frozen reference. `decay` controls EMA blending across
/// successive snapshots: `decay = 0.0` resets the snapshot at every
/// refresh; `decay = 0.9` keeps 90% of the prior snapshot and blends in
/// 10% of the current policy. This implements the "stronger reference"
/// idea from the post-DAPO line of work — the reference policy tracks
/// actual training progress instead of pulling toward the (potentially
/// pre-reasoning) base model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReferencePolicy {
    /// Base model (no LoRA), recomputed per completion. Historical default.
    BasePerStep,
    /// No reference forward; ratio fixed at 1.0, KL forced off.
    None,
    /// EMA snapshot of the LoRA-applied policy.
    Ema {
        #[serde(default = "default_ema_decay")]
        decay: f32,
        #[serde(default = "default_ema_refresh")]
        refresh_every: usize,
    },
}

impl Default for ReferencePolicy {
    fn default() -> Self {
        ReferencePolicy::BasePerStep
    }
}

fn default_ema_decay() -> f32 {
    0.0
}
fn default_ema_refresh() -> usize {
    32
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
    /// uses this form. Implemented on the shared kt tape path and the
    /// Vulkan kernel (`VK_GRPO_KL_MODE_K3`).
    K3,
    /// No KL penalty. The reference forward still runs (still needed for the
    /// importance ratio); only the KL contribution to the loss is zeroed.
    /// Equivalent in effect to `kl_coeff = 0` but expresses the intent
    /// explicitly.
    None,
}

/// Behavior when reward-variance filtering would leave too few groups.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RewardFilterOnEmpty {
    /// Fail the run with a receipt and reward-filter sidecar.
    #[default]
    Fail,
    /// Ignore the reward filter for this run and train on every group.
    TrainAll,
    /// Skip optimizer work and emit an untrained/base adapter plus receipt.
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Learning rate. `None` (the default) resolves per optimizer at run
    /// start — see [`resolve_learning_rate`]. Explicit values are used
    /// verbatim; the train receipt records whichever value actually ran.
    #[serde(default)]
    pub learning_rate: Option<f64>,
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
    /// Advantage normalization mode. Defaults to `DrGrpo` (drops
    /// std-normalization per arXiv:2503.20783). Set to `Vanilla` for the
    /// historical DeepSeekMath/R1 form.
    #[serde(default)]
    pub advantage_mode: AdvantageMode,
    /// Surrogate-loss aggregation mode. Defaults to `TokenLevel` (the DAPO
    /// Token-Level Loss, arXiv:2503.14476). Set to `PerSample` for the
    /// historical per-completion form.
    #[serde(default)]
    pub loss_aggregation: LossAggregation,
    /// KL penalty estimator. Defaults to `K1` (historical kiln behavior).
    #[serde(default)]
    pub kl_estimator: KlEstimator,
    /// When true (the kiln default since Phase 1), groups whose completions
    /// all share the same reward (and therefore produce a uniformly-zero
    /// advantage vector) are skipped before training. Implements DAPO's
    /// Dynamic Sampling filter (arXiv:2503.14476). Degenerate groups have
    /// no policy-gradient signal under any [`AdvantageMode`]; the only
    /// thing they contribute is a spurious KL pull, so dropping them is
    /// strictly compute-saving and a stability win. Phase 1 ablation on
    /// Qwen3.5-4B observed ~60-70% of humaneval-style groups filtered as
    /// degenerate at typical reward distributions.
    #[serde(default = "default_dynamic_sampling")]
    pub dynamic_sampling: bool,
    /// Importance-sampling level. Defaults to `Token` (historical kiln
    /// behavior). Set to `Sequence` for GSPO or `Cispo` for CISPO.
    #[serde(default)]
    pub is_level: IsLevel,
    /// Reference-policy selection. Defaults to `BasePerStep` (historical
    /// kiln behavior, KL anchored to the base model). Set to `None` for
    /// REINFORCE-like training without an IS ratio, or `Ema` for a moving
    /// snapshot of the LoRA-applied policy.
    #[serde(default)]
    pub reference_policy: ReferencePolicy,
    /// Phase 3c — selective-KL entropy regulation. When `Some(q)`, only
    /// tokens whose proxy entropy (defined as `-policy_log_prob` for the
    /// selected token) is at or above the `q`-quantile across the active
    /// tokens contribute to the KL penalty. This is a cheap approximation
    /// of the Cui et al. "high-entropy minority tokens drive effective RL"
    /// finding (arXiv:2506.01939): the full Clip-Cov / KL-Cov estimators
    /// require per-token vocab covariance, which doubles the analytic tail
    /// cost; selecting by negative chosen log-prob instead requires no
    /// extra forward work since policy log-probs are already materialized
    /// for the IS ratio. Typical values: `0.8` (apply KL on top-20% tokens
    /// by uncertainty), `None` = full-token KL (historical behavior).
    /// Only meaningful when `kl_estimator != None`.
    #[serde(default)]
    pub entropy_aware_kl_quantile: Option<f32>,
    /// Mean reward threshold above which low-variance groups are considered
    /// saturated. Used only for diagnostics and train receipts.
    #[serde(default = "default_reward_saturation_threshold")]
    pub reward_saturation_threshold: f64,
    /// Population variance threshold used with `reward_saturation_threshold`
    /// to warn about low-signal reward distributions.
    #[serde(default = "default_reward_low_variance_threshold")]
    pub reward_low_variance_threshold: f64,
    /// Drop groups whose population reward variance is below this threshold.
    /// Disabled when both reward-filter variance bounds are unset.
    #[serde(default)]
    pub reward_filter_var_min: Option<f64>,
    /// Drop groups whose population reward variance is above this threshold.
    /// Usually left unset; useful for excluding pathological reward outliers.
    #[serde(default)]
    pub reward_filter_var_max: Option<f64>,
    /// Minimum number of groups that must remain after reward filtering.
    #[serde(default = "default_reward_filter_min_groups")]
    pub reward_filter_min_groups: usize,
    /// Explicit behavior when reward filtering keeps fewer than
    /// `reward_filter_min_groups`.
    #[serde(default)]
    pub reward_filter_on_empty: RewardFilterOnEmpty,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    pub base_adapter: Option<String>,
    /// Reserved escape hatch for future explicit shape conversion. Today,
    /// incompatible base adapters still fail before optimizer setup.
    #[serde(default)]
    pub allow_adapter_shape_conversion: bool,
    /// Permit alpha/rank above the default safety limit for deliberate
    /// experiments.
    #[serde(default)]
    pub allow_high_lora_scale: bool,
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
    /// After successful training, run a small base-vs-adapter canary check and
    /// record the result in `train_receipt.json`.
    #[serde(default)]
    pub adapter_smoke_test: bool,
    /// Composition of per-token training objectives. ECHO is OFF by
    /// default until its env-CE term regains a kt-tape gradient root
    /// (#1082); OPD's slot is reserved but empty. See `LossConfig`.
    #[serde(default)]
    pub loss: LossConfig,
}

fn default_kl_coeff() -> f64 {
    0.1
}
fn default_clip_eps() -> f64 {
    0.2
}
fn default_dynamic_sampling() -> bool {
    true
}
fn default_reward_saturation_threshold() -> f64 {
    crate::train_receipt::DEFAULT_REWARD_SATURATION_THRESHOLD
}
fn default_reward_low_variance_threshold() -> f64 {
    crate::train_receipt::DEFAULT_REWARD_LOW_VARIANCE_THRESHOLD
}
fn default_reward_filter_min_groups() -> usize {
    1
}

// ---- ECHO + LossConfig ----------------------------------------------------
//
// `LossConfig` composes three loss objectives that share one forward pass:
//
//   L_total = L_policy(actions)              [the GRPO surrogate]
//           + λ_echo · L_envCE(observations) [paper: Shrivastava 2026]
//           + λ_opd  · L_revKL(actions)      [paper: Lu 2025; wired in OPD merge]
//
// `LossConfig::default()` has ECHO ON (λ=0.05, paper §3.3): the env-CE
// term regained its gradient root on the fused GRPO tape node (ECHO
// resurrection PR2 — `grpo_tape_shim::EchoEnvSpec`), so default-config
// agentic GRPO on real pi trajectories trains the observation tokens
// again. Rollouts with no env tokens pay nothing (the term contributes
// exactly zero and the receipt's env_ce stays None).
// `opd` is a placeholder None; its shape is reserved so the composition
// stays orthogonal.

/// What positions in an observation segment contribute to the env_mask.
///
/// Paper §3.2: terminal warning prefixes memorize within ~60 steps and stop
/// providing useful gradient. `EnvOnly` (default) strips the harness
/// warning prefix from each Observation; `FullObs` (debug only) covers
/// the full observation including warnings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvMaskMode {
    #[default]
    EnvOnly,
    FullObs,
}

/// Configuration for ECHO — the auxiliary environment cross-entropy loss
/// on tool/observation tokens. See
/// `docs/papers/echo/echo_paper.md` and `docs/plans/echo-integration-plan.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoConfig {
    /// Mixing coefficient for the env-CE term: `L_total = L_policy + λ · L_envCE`.
    /// Paper §3.3 default: 0.05. Productive range: 0.01–0.05. ≥0.1 risks
    /// degrading the policy objective; 0.2 collapses to predictable-output
    /// rollouts (mode collapse).
    #[serde(default = "default_echo_lambda")]
    pub lambda: f64,
    /// Env_only (default) strips the harness warning prefix. FullObs (debug)
    /// covers the full observation including warnings.
    #[serde(default)]
    pub env_mask_mode: EnvMaskMode,
    /// Whether the trajectory_mask's warning_filter is on. Defaults to true.
    /// Set to false for debugging or for environments whose tool output
    /// never includes a `WARNINGS:` prefix.
    #[serde(default = "default_warning_filter")]
    pub warning_filter: bool,
}

fn default_echo_lambda() -> f64 {
    0.05
}
fn default_warning_filter() -> bool {
    true
}
/// ECHO default: ON at λ=0.05 (paper §3.3). The term was forced OFF
/// between the #1082 candle drop (which deleted the only env-CE gradient
/// producer) and resurrection PR2 (which rebuilt it as constant-coefficient
/// `(softmax − onehot)` rows on the fused GRPO tape root). Legacy
/// single-turn rollouts carry no env tokens, so the default costs them
/// exactly nothing.
fn default_echo_some() -> Option<EchoConfig> {
    Some(EchoConfig::default())
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            lambda: default_echo_lambda(),
            env_mask_mode: EnvMaskMode::default(),
            warning_filter: default_warning_filter(),
        }
    }
}

/// Placeholder for OPD's auxiliary term on the LossConfig — populated when
/// the OPD branch rebases on top of this work. The fields here mirror
/// `kiln_train::opd::OpdConfig`'s most relevant knobs; the exact shape will
/// be reconciled during the OPD merge. Default: None.
///
/// Reserving this slot now means `L = L_policy + λ_echo · L_envCE +
/// λ_opd · L_revKL` is structurally encoded in the type system, so OPD
/// composition is mechanical when its branch lands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpdAuxConfig {
    #[serde(default)]
    pub lambda: f64,
}

/// Composition of per-token training objectives. Each branch contributes to
/// `L_total = L_policy + λ_echo · L_envCE + λ_opd · L_revKL` where inactive
/// branches contribute zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Action-token policy-gradient term — knobs already on GrpoConfig
    /// (clip_eps, kl_coeff, kl_estimator, advantage_mode, is_level,
    /// reference_policy, etc.). LossConfig doesn't duplicate them; the
    /// trainer reads them from the surrounding GrpoConfig.
    ///
    /// Observation-token cross-entropy (paper: Shrivastava et al. 2026).
    /// Default: `Some(EchoConfig::default())` — ON at λ=0.05 (resurrection
    /// PR2 rebuilt the env-CE gradient on the fused GRPO tape root after
    /// the #1082 candle drop severed it). Rollouts without environment
    /// tokens pay nothing; set to `null` (or pass `--no-echo`) to train
    /// the action-token policy loss alone.
    #[serde(default = "default_echo_some")]
    pub echo: Option<EchoConfig>,
    /// Action-token reverse-KL to a teacher (OPD; Lu 2025). Default: None.
    /// Wired in by the OPD branch rebase; LossConfig holds the field so the
    /// composition is structurally orthogonal.
    #[serde(default)]
    pub opd: Option<OpdAuxConfig>,
    /// Verifier-free env-only adaptation mode (paper §5.5). When `true`,
    /// the trainer masks out the GRPO policy-gradient term entirely and
    /// trains *only* on ECHO's env-CE objective. Useful for adapting an
    /// already-trained agentic policy on tasks where no programmatic
    /// verifier is available — the model improves purely by learning to
    /// predict the consequences of its own actions. Default: false.
    #[serde(default)]
    pub no_policy_loss: bool,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            echo: default_echo_some(),
            opd: None,
            no_policy_loss: false,
        }
    }
}

impl LossConfig {
    /// Validate this composition against the kt-tape training path BEFORE
    /// any GPU work. The ECHO env-CE term is live again (resurrection PR2:
    /// constant-coefficient rows on the fused GRPO tape root), so
    /// echo-enabled configs with environment tokens TRAIN now. Still
    /// rejected: `no_policy_loss` (verifier-free ECHO-only training needs
    /// the PG coefficients zeroed while the env rows drive — not yet
    /// re-wired) and the reserved `opd` slot.
    /// `has_env_tokens` = the training data carries trajectory
    /// Observation/tool segments that the env mask would cover.
    pub fn validate_for_kt_tape(&self, _has_env_tokens: bool) -> Result<(), String> {
        if self.no_policy_loss {
            return Err(
                "loss.no_policy_loss (verifier-free, ECHO-only training) is not yet \
                 re-wired on the kt-tape path: it needs the policy-gradient \
                 coefficients zeroed while the env-CE rows drive the update. Remove \
                 no_policy_loss — the ECHO env-CE term itself is live again and \
                 composes with the standard policy loss."
                    .to_string(),
            );
        }
        if self.opd.is_some() {
            return Err(
                "loss.opd composition on GRPO is reserved but not wired — use \
                 POST /v1/train/opd for on-policy distillation instead."
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Lambda for the ECHO term, or 0.0 if ECHO is disabled.
    pub fn echo_lambda(&self) -> f64 {
        self.echo.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
    /// Lambda for the OPD auxiliary term, or 0.0 if OPD is disabled.
    pub fn opd_lambda(&self) -> f64 {
        self.opd.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
    /// True when the ECHO term is configured to be applied.
    pub fn echo_enabled(&self) -> bool {
        self.echo.is_some() && self.echo_lambda() != 0.0
    }

    /// Apply environment-variable overrides on top of an existing
    /// `LossConfig`. Honors:
    ///
    /// - `KILN_ECHO_ENABLED` — `0`/`false`/`no` → disable ECHO
    ///   (`loss.echo = None`); `1`/`true`/`yes` → enable with current
    ///   `lambda` (or default 0.05 if previously disabled).
    /// - `KILN_ECHO_LAMBDA` — overrides `lambda` on the existing
    ///   `EchoConfig`; if ECHO is disabled and this is set, ECHO is
    ///   re-enabled with the given lambda.
    /// - `KILN_ECHO_ENV_MASK_MODE` — `env_only` (default) | `full_obs`.
    /// - `KILN_ECHO_WARNING_FILTER` — bool, default true.
    ///
    /// Call this from CLI entry points (cuda_grpo_ablation, vk_train CLI,
    /// future kiln-server route handlers) so users can toggle ECHO from
    /// the shell without editing JSON.
    pub fn apply_kiln_echo_env_overrides(&mut self) {
        // KILN_ECHO_ENABLED — explicit disable wins over anything else.
        if let Ok(val) = std::env::var("KILN_ECHO_ENABLED") {
            let v = val.to_lowercase();
            if v == "0" || v == "false" || v == "no" {
                self.echo = None;
            } else if v == "1" || v == "true" || v == "yes" {
                self.echo.get_or_insert_with(EchoConfig::default);
            }
        }

        // The remaining knobs only make sense when ECHO is on. Apply
        // them on the current EchoConfig (creating one if needed when
        // any knob is set).
        let lambda_override = std::env::var("KILN_ECHO_LAMBDA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let mask_mode_override =
            std::env::var("KILN_ECHO_ENV_MASK_MODE").ok().and_then(|v| {
                match v.to_lowercase().as_str() {
                    "env_only" | "envonly" => Some(EnvMaskMode::EnvOnly),
                    "full_obs" | "fullobs" => Some(EnvMaskMode::FullObs),
                    _ => None,
                }
            });
        let warning_filter_override =
            std::env::var("KILN_ECHO_WARNING_FILTER")
                .ok()
                .and_then(|v| match v.to_lowercase().as_str() {
                    "0" | "false" | "no" => Some(false),
                    "1" | "true" | "yes" => Some(true),
                    _ => None,
                });

        if lambda_override.is_some()
            || mask_mode_override.is_some()
            || warning_filter_override.is_some()
        {
            let cfg = self.echo.get_or_insert_with(EchoConfig::default);
            if let Some(lambda) = lambda_override {
                cfg.lambda = lambda;
            }
            if let Some(mode) = mask_mode_override {
                cfg.env_mask_mode = mode;
            }
            if let Some(wf) = warning_filter_override {
                cfg.warning_filter = wf;
            }
        }
    }
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

    /// The explicit `learning_rate` when given, else the per-optimizer
    /// default for GRPO.
    pub fn effective_learning_rate(&self) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| resolve_learning_rate(&self.optimizer, TrainMode::Grpo))
    }
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: None,
            kl_coeff: default_kl_coeff(),
            clip_epsilon: default_clip_eps(),
            clip_eps_high: None,
            advantage_mode: AdvantageMode::default(),
            loss_aggregation: LossAggregation::default(),
            kl_estimator: KlEstimator::default(),
            dynamic_sampling: default_dynamic_sampling(),
            is_level: IsLevel::default(),
            reference_policy: ReferencePolicy::default(),
            entropy_aware_kl_quantile: None,
            reward_saturation_threshold: default_reward_saturation_threshold(),
            reward_low_variance_threshold: default_reward_low_variance_threshold(),
            reward_filter_var_min: None,
            reward_filter_var_max: None,
            reward_filter_min_groups: default_reward_filter_min_groups(),
            reward_filter_on_empty: RewardFilterOnEmpty::default(),
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            allow_adapter_shape_conversion: false,
            allow_high_lora_scale: false,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            seed: None,
            optimizer: Optimizer::default(),
            adapter_smoke_test: false,
            loss: LossConfig::default(),
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
    /// Failure detail when `state == failed`; absent otherwise and on
    /// payloads that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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

    /// Legacy `TrainingStatus` payloads (pre-`error` archives, older
    /// servers) must keep deserializing, and the field must stay off the
    /// wire when `None` so old dashboards see an unchanged shape.
    #[test]
    fn training_status_error_field_is_optional_and_skipped_when_none() {
        let legacy = serde_json::json!({
            "job_id": "job-1",
            "state": "failed",
            "progress": 0.5,
            "current_loss": null,
            "adapter_name": "a",
            "started_at": "3s ago",
            "elapsed_secs": 3.0
        });
        let status: TrainingStatus = serde_json::from_value(legacy).unwrap();
        assert!(status.error.is_none());

        let none_wire = serde_json::to_value(&status).unwrap();
        assert!(none_wire.get("error").is_none(), "None must be omitted");

        let failed = TrainingStatus {
            error: Some("trainer exploded".to_string()),
            ..status
        };
        let wire = serde_json::to_value(&failed).unwrap();
        assert_eq!(wire["error"], "trainer exploded");
        let back: TrainingStatus = serde_json::from_value(wire).unwrap();
        assert_eq!(back.error.as_deref(), Some("trainer exploded"));
    }

    /// Pin the serde shape of the training-policy defaults. A future default
    /// flip must update this snapshot in the same commit, which forces the
    /// field/enum docs (and CHANGELOG) to be revisited together — the Muon
    /// flip left "Defaults to AdamW" doc-rot behind precisely because nothing
    /// tied the default to a literal.
    #[test]
    fn training_policy_defaults_match_pinned_snapshots() {
        assert_eq!(
            serde_json::to_value(Optimizer::default()).unwrap(),
            serde_json::json!({
                "kind": "muon",
                // f32 fields widen to f64 in serde_json, so pin via casts.
                "momentum": 0.95f32 as f64,
                "nesterov": true,
                "ns_iters": 5,
                "weight_decay": 0.0f32 as f64
            })
        );
        assert_eq!(
            serde_json::to_value(AdvantageMode::default()).unwrap(),
            serde_json::json!("dr_grpo")
        );
        assert_eq!(
            serde_json::to_value(LossAggregation::default()).unwrap(),
            serde_json::json!("token_level")
        );
        assert_eq!(
            serde_json::to_value(KlEstimator::default()).unwrap(),
            serde_json::json!("k1")
        );
    }

    /// Pin the full per-optimizer learning-rate table. AdamW/SGD are the
    /// unchanged legacy defaults; Muon's SFT value is kiln-optim's
    /// `Muon::default_hp()` lr, with GRPO/OPD scaled down by the legacy
    /// AdamW SFT:GRPO ratio.
    #[test]
    fn learning_rate_resolution_table_snapshot() {
        let adamw: Optimizer = serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap();
        let muon = Optimizer::default();
        let table = [
            (&muon, TrainMode::Sft, 2e-2),
            (&muon, TrainMode::Grpo, 2e-3),
            (&muon, TrainMode::Opd, 2e-3),
            (&adamw, TrainMode::Sft, 1e-4),
            (&adamw, TrainMode::Grpo, 1e-5),
            (&adamw, TrainMode::Opd, 1e-5),
            (&Optimizer::Sgd, TrainMode::Sft, 1e-4),
            (&Optimizer::Sgd, TrainMode::Grpo, 1e-5),
            (&Optimizer::Sgd, TrainMode::Opd, 1e-5),
        ];
        for (optimizer, mode, expected) in table {
            let resolved = resolve_learning_rate(optimizer, mode);
            assert_eq!(
                resolved, expected,
                "resolve_learning_rate({optimizer:?}, {mode:?})"
            );
        }
    }

    #[test]
    fn learning_rate_serde_back_compat() {
        // Explicit wire values still deserialize (full back-compat) …
        let sft: SftConfig = serde_json::from_str(r#"{"learning_rate": 5e-5}"#).unwrap();
        assert_eq!(sft.learning_rate, Some(5e-5));
        let grpo: GrpoConfig = serde_json::from_str(r#"{"learning_rate": 5e-5}"#).unwrap();
        assert_eq!(grpo.learning_rate, Some(5e-5));
        // … and omitting the field means "resolve per optimizer".
        let sft: SftConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(sft.learning_rate, None);
        let grpo: GrpoConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(grpo.learning_rate, None);
    }

    #[test]
    fn effective_learning_rate_prefers_explicit_value() {
        let mut sft = SftConfig::default();
        assert_eq!(sft.effective_learning_rate(), 2e-2); // Muon default
        sft.learning_rate = Some(3e-4);
        assert_eq!(sft.effective_learning_rate(), 3e-4);

        let mut grpo = GrpoConfig::default();
        assert_eq!(grpo.effective_learning_rate(), 2e-3); // Muon default
        grpo.learning_rate = Some(7e-6);
        assert_eq!(grpo.effective_learning_rate(), 7e-6);

        let adamw_sft = SftConfig {
            optimizer: serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap(),
            ..SftConfig::default()
        };
        assert_eq!(adamw_sft.effective_learning_rate(), 1e-4);
    }

    #[test]
    fn learning_rate_band_warning_fires_only_past_50x() {
        // The motivating bug: AdamW-era 1e-4 against Muon's 2e-2 (200x cold).
        assert!(learning_rate_band_warning(1e-4, 2e-2).is_some());
        // 200x hot fires too.
        assert!(learning_rate_band_warning(4.0, 2e-2).is_some());
        // Ordinary tuning (10x either way) stays quiet.
        assert!(learning_rate_band_warning(2e-3, 2e-2).is_none());
        assert!(learning_rate_band_warning(2e-1, 2e-2).is_none());
        // Exactly on band, and degenerate inputs, stay quiet.
        assert!(learning_rate_band_warning(2e-2, 2e-2).is_none());
        assert!(learning_rate_band_warning(0.0, 2e-2).is_none());
        assert!(learning_rate_band_warning(f64::NAN, 2e-2).is_none());
    }

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

    #[test]
    fn optimizer_default_is_muon() {
        match Optimizer::default() {
            Optimizer::Muon {
                momentum,
                nesterov,
                ns_iters,
                weight_decay,
            } => {
                assert_eq!(momentum, 0.95);
                assert!(nesterov);
                assert_eq!(ns_iters, 5);
                assert_eq!(weight_decay, 0.0);
            }
            other => panic!("default optimizer should be Muon, got {other:?}"),
        }
    }

    #[test]
    fn optimizer_muon_serializes_with_kind_muon() {
        let opt = Optimizer::default();
        let v: serde_json::Value = serde_json::to_value(opt).unwrap();
        assert_eq!(v["kind"], "muon");
        // momentum is an f32 widened to f64 in JSON — compare with tolerance.
        assert!((v["momentum"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(v["nesterov"], true);
        assert_eq!(v["ns_iters"], 5);
    }

    #[test]
    fn optimizer_muon_round_trips() {
        let opt = Optimizer::Muon {
            momentum: 0.9,
            nesterov: false,
            ns_iters: 7,
            weight_decay: 0.01,
        };
        let json = serde_json::to_string(&opt).unwrap();
        let back: Optimizer = serde_json::from_str(&json).unwrap();
        assert_eq!(opt, back);
    }

    #[test]
    fn optimizer_muon_minimal_json_fills_defaults() {
        // Bare `{"kind": "muon"}` must fill every Muon field from defaults.
        let opt: Optimizer = serde_json::from_str(r#"{"kind": "muon"}"#).unwrap();
        assert_eq!(opt, Optimizer::default());
    }

    #[test]
    fn optimizer_adamw_and_sgd_still_deserialize() {
        // Back-compat: the old optimizer kinds remain selectable.
        let adamw: Optimizer = serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap();
        assert!(matches!(adamw, Optimizer::AdamW { .. }));
        let sgd: Optimizer = serde_json::from_str(r#"{"kind": "sgd"}"#).unwrap();
        assert_eq!(sgd, Optimizer::Sgd);
    }

    #[test]
    fn configs_default_to_muon_optimizer() {
        // Every training-mode config defaults to Muon when no optimizer is given.
        assert!(matches!(
            SftConfig::default().optimizer,
            Optimizer::Muon { .. }
        ));
        assert!(matches!(
            GrpoConfig::default().optimizer,
            Optimizer::Muon { .. }
        ));
        let sft: SftConfig = serde_json::from_str(r#"{"epochs": 2}"#).unwrap();
        assert!(matches!(sft.optimizer, Optimizer::Muon { .. }));
    }

    #[test]
    fn test_grpo_reward_diagnostic_threshold_defaults() {
        let config = GrpoConfig::default();
        assert!(
            (config.reward_saturation_threshold
                - crate::train_receipt::DEFAULT_REWARD_SATURATION_THRESHOLD)
                .abs()
                < 1e-12
        );
        assert!(
            (config.reward_low_variance_threshold
                - crate::train_receipt::DEFAULT_REWARD_LOW_VARIANCE_THRESHOLD)
                .abs()
                < 1e-12
        );
        assert!(config.reward_filter_var_min.is_none());
        assert!(config.reward_filter_var_max.is_none());
        assert_eq!(config.reward_filter_min_groups, 1);
        assert_eq!(config.reward_filter_on_empty, RewardFilterOnEmpty::Fail);
    }

    #[test]
    fn test_grpo_reward_diagnostic_thresholds_deserialize() {
        let json = r#"{
            "reward_saturation_threshold": 0.8,
            "reward_low_variance_threshold": 0.002,
            "reward_filter_var_min": 0.01,
            "reward_filter_var_max": 0.25,
            "reward_filter_min_groups": 3,
            "reward_filter_on_empty": "train-all"
        }"#;
        let config: GrpoConfig = serde_json::from_str(json).unwrap();
        assert!((config.reward_saturation_threshold - 0.8).abs() < 1e-12);
        assert!((config.reward_low_variance_threshold - 0.002).abs() < 1e-12);
        assert_eq!(config.reward_filter_var_min, Some(0.01));
        assert_eq!(config.reward_filter_var_max, Some(0.25));
        assert_eq!(config.reward_filter_min_groups, 3);
        assert_eq!(config.reward_filter_on_empty, RewardFilterOnEmpty::TrainAll);
        assert_eq!(config.kl_coeff, 0.1);
    }

    // ECHO LossConfig + env-var override tests. Env-var manipulation is
    // process-global; each test scrubs the env vars at start to keep the
    // suite hermetic regardless of order. We use the modern `unsafe`
    // wrappers on Rust 1.83+ to make the mutation explicit.
    //
    // ENV_LOCK serializes the env-var-touching tests against each other
    // so cargo test's default parallel runner doesn't race on the
    // process-global env state. Mirrors `trainer::tests::ENV_LOCK`.
    // Acquire via `_env_guard = ENV_LOCK.lock().unwrap_or_else(...)` at
    // the top of every test that calls `clear_kiln_echo_env_vars` or
    // `std::env::set_var`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_kiln_echo_env_vars() {
        unsafe {
            std::env::remove_var("KILN_ECHO_ENABLED");
            std::env::remove_var("KILN_ECHO_LAMBDA");
            std::env::remove_var("KILN_ECHO_ENV_MASK_MODE");
            std::env::remove_var("KILN_ECHO_WARNING_FILTER");
        }
    }

    #[test]
    fn loss_config_default_has_echo_on_again() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig::default();
        // Resurrection PR2: the env-CE term regained its gradient root on
        // the fused GRPO tape node, so the paper default (λ=0.05) is the
        // out-of-the-box behavior again — and a default config with env
        // tokens VALIDATES.
        let echo = cfg.echo.as_ref().expect("ECHO defaults ON post-resurrection");
        assert!((echo.lambda - 0.05).abs() < 1e-12, "paper §3.3 default λ");
        assert!(cfg.opd.is_none(), "OPD should be off by default");
        assert!(cfg.validate_for_kt_tape(true).is_ok());
    }

    #[test]
    fn loss_config_validation_rejects_untrainable_compositions() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();

        // ECHO + env tokens trains again (resurrection PR2) — both shapes
        // validate.
        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        assert!(cfg.validate_for_kt_tape(false).is_ok());
        assert!(
            cfg.validate_for_kt_tape(true).is_ok(),
            "echo + env tokens is the flagship agentic shape — must validate"
        );

        // no_policy_loss: verifier-free ECHO-only training is not yet
        // re-wired (PG coefficients must be zeroed while env rows drive).
        let mut cfg = LossConfig::default();
        cfg.no_policy_loss = true;
        let err = cfg.validate_for_kt_tape(false).unwrap_err();
        assert!(err.contains("no_policy_loss"), "{err}");

        // Reserved OPD slot: point at the real endpoint.
        let mut cfg = LossConfig::default();
        cfg.opd = Some(OpdAuxConfig { lambda: 1.0 });
        let err = cfg.validate_for_kt_tape(false).unwrap_err();
        assert!(err.contains("/v1/train/opd"), "{err}");
    }

    #[test]
    fn loss_config_echo_lambda_returns_zero_when_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let mut cfg = LossConfig::default();
        cfg.echo = None;
        assert_eq!(cfg.echo_lambda(), 0.0);
        assert!(!cfg.echo_enabled());
    }

    #[test]
    fn kiln_echo_enabled_false_disables_echo() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_ENABLED", "false") };
        // Start from an explicitly-enabled config (the default is now OFF)
        // so the env override has something to disable.
        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_none(), "KILN_ECHO_ENABLED=false should disable");
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_lambda_overrides_value() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_LAMBDA", "0.02") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        assert!(
            (cfg.echo_lambda() - 0.02).abs() < 1e-12,
            "KILN_ECHO_LAMBDA=0.02 should override; got {}",
            cfg.echo_lambda()
        );
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_lambda_re_enables_when_previously_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_LAMBDA", "0.03") };
        let mut cfg = LossConfig::default();
        cfg.echo = None; // explicitly disabled before override
        cfg.apply_kiln_echo_env_overrides();
        assert!(
            cfg.echo.is_some(),
            "setting LAMBDA env var should also re-enable ECHO"
        );
        assert!((cfg.echo_lambda() - 0.03).abs() < 1e-12);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_env_mask_mode_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_ENV_MASK_MODE", "full_obs") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        let echo = cfg.echo.expect("ECHO should still be on");
        assert_eq!(echo.env_mask_mode, EnvMaskMode::FullObs);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_warning_filter_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_WARNING_FILTER", "false") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        let echo = cfg.echo.expect("ECHO should still be on");
        assert!(!echo.warning_filter);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_enabled_true_with_lambda_combo() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe {
            std::env::set_var("KILN_ECHO_ENABLED", "true");
            std::env::set_var("KILN_ECHO_LAMBDA", "0.1");
        }
        let mut cfg = LossConfig::default();
        cfg.echo = None;
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_some());
        assert!((cfg.echo_lambda() - 0.1).abs() < 1e-12);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_no_env_vars_leaves_config_unchanged() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        // Both the (ON again) default and an explicit config survive a
        // pass with no env vars set.
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_some(), "default ON stays ON");

        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        let original_lambda = cfg.echo_lambda();
        cfg.apply_kiln_echo_env_overrides();
        assert!((cfg.echo_lambda() - original_lambda).abs() < 1e-12);
        assert!(cfg.echo.is_some(), "explicit opt-in stays on");
    }

    #[test]
    fn loss_config_no_policy_loss_default_is_false() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig::default();
        assert!(!cfg.no_policy_loss);
    }

    #[test]
    fn loss_config_no_policy_loss_serde_round_trip() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig {
            echo: Some(EchoConfig::default()),
            opd: None,
            no_policy_loss: true,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: LossConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.no_policy_loss);
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn loss_config_legacy_payload_without_no_policy_loss_defaults_false() {
        // Old payloads that don't include `no_policy_loss` at all still parse.
        let json = r#"{"echo": {"lambda": 0.05}, "opd": null}"#;
        let cfg: LossConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.no_policy_loss);
    }

    #[test]
    fn echo_config_custom_fields_survive_json_roundtrip() {
        // Every EchoConfig field set away from its default — pin the
        // wire format so a future field rename or default change is a
        // loud test failure rather than a silent compatibility break.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let custom = EchoConfig {
            lambda: 0.027,
            env_mask_mode: EnvMaskMode::FullObs,
            warning_filter: false,
        };
        let json = serde_json::to_string(&custom).unwrap();
        // The serialized form should mention every non-default field.
        assert!(json.contains("0.027"), "lambda missing from {json}");
        assert!(
            json.contains("full_obs"),
            "env_mask_mode missing from {json}"
        );
        assert!(
            json.contains("\"warning_filter\":false"),
            "warning_filter missing from {json}"
        );

        let parsed: EchoConfig = serde_json::from_str(&json).unwrap();
        assert!((parsed.lambda - 0.027).abs() < 1e-12);
        assert_eq!(parsed.env_mask_mode, EnvMaskMode::FullObs);
        assert!(!parsed.warning_filter);
    }

    #[test]
    fn echo_config_partial_payload_fills_defaults() {
        // HTTP clients may send only `{"lambda": ...}` — the other fields
        // must fall back to their `#[serde(default = ...)]` values.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let json = r#"{"lambda": 0.012}"#;
        let parsed: EchoConfig = serde_json::from_str(json).unwrap();
        assert!((parsed.lambda - 0.012).abs() < 1e-12);
        assert_eq!(parsed.env_mask_mode, EnvMaskMode::default());
        assert!(parsed.warning_filter, "warning_filter must default to true");
    }

    #[test]
    fn grpo_request_accepts_agentic_groups_alias() {
        // The /v1/train/agentic route reads the same `GrpoRequest` struct
        // as /v1/train/grpo, but clients calling the "agentic" route may
        // semantically prefer `agentic_groups`. Both must parse.
        let body_legacy = r#"{"groups": []}"#;
        let body_agentic = r#"{"agentic_groups": []}"#;
        let parsed_legacy: GrpoRequest = serde_json::from_str(body_legacy).unwrap();
        let parsed_agentic: GrpoRequest = serde_json::from_str(body_agentic).unwrap();
        assert_eq!(parsed_legacy.groups.len(), 0);
        assert_eq!(parsed_agentic.groups.len(), 0);
    }

    /// Capability authors ship `capability.config.json` files with a
    /// `training_phase1_defaults.loss` block that the cap launcher
    /// passes verbatim to `train_tokenized_grpo_group`. Pin the
    /// contract: the shapes used in `pi-terminal-bench-lite` and
    /// `pi-script-fixup` must always deserialize as a valid LossConfig.
    /// If either ever drifts the assertion fires.
    #[test]
    fn cap_config_loss_blocks_deserialize_as_lossconfig() {
        // pi-terminal-bench-lite: ECHO on, OPD off, no_policy_loss absent.
        let tblite = r#"{
            "echo": {
                "lambda": 0.05,
                "env_mask_mode": "env_only",
                "warning_filter": true
            },
            "opd": null
        }"#;
        let parsed: LossConfig = serde_json::from_str(tblite).unwrap();
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
        assert!(parsed.opd.is_none());
        assert!(!parsed.no_policy_loss);
        let echo = parsed.echo.as_ref().unwrap();
        assert_eq!(echo.env_mask_mode, EnvMaskMode::EnvOnly);
        assert!(echo.warning_filter);

        // pi-script-fixup: same shape plus no_policy_loss = true.
        let fixup = r#"{
            "echo": {
                "lambda": 0.05,
                "env_mask_mode": "env_only",
                "warning_filter": true
            },
            "opd": null,
            "no_policy_loss": true
        }"#;
        let parsed: LossConfig = serde_json::from_str(fixup).unwrap();
        assert!(
            parsed.no_policy_loss,
            "verifier-free cap must parse with no_policy_loss=true"
        );
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
    }
}

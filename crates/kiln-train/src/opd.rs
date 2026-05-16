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
}

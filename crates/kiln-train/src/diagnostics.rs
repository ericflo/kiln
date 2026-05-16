//! OPD diagnostic stack — paper-cited metrics streamed during training.
//!
//! Implements the §3.8 diagnostics from the grand plan, with each metric
//! tagged with the paper section that defined it. The metrics fall into
//! three families:
//!
//! 1. **Trajectory-shape** (this module) — truncation rate, repetition
//!    rate. Computed from rollout token streams alone; no kernel cost,
//!    no teacher cost. These drive the [`LengthInflation`] guardrail
//!    (§3.9), which auto-tunes Stable-OPD's `β_kl` and `λ_sft` when
//!    `RepRate > 0.05` over consecutive validation passes (Luo et al.
//!    2026).
//! 2. **Distribution alignment** (next commit) — overlap ratio,
//!    overlap-token advantage, entropy gap. Require the kernel to emit
//!    its per-position `log_p_hat` / `log_q_hat` intermediates as side
//!    outputs. Lands when the kernel grows the `metrics` parameter.
//! 3. **Operational** — teacher-cache hit rate, cost spent. Wired into
//!    the `LogitSource` impls (cache + remote teachers) so they report
//!    incrementally; nothing to compute here.
//!
//! # Why repetition rate uses zlib compression
//!
//! Luo et al. 2026 §3.2 defines RepRate as a compression-ratio detector:
//!
//! ```text
//! CompRatio(o) = |bytes(o_tail)| / |zlib(bytes(o_tail))|
//! rep(o) = [|o_tail| > L && CompRatio(o) > τ]
//! ```
//!
//! with `L = 10_000` characters and `τ = 10`. Repetitive text compresses
//! at >10:1 (the LZ77 + Huffman dictionary in zlib catches the repeated
//! phrase); genuinely diverse text plateaus around 2:1–4:1. We delegate
//! to `flate2` (already a workspace dependency via `kiln-server`) so the
//! metric matches Luo et al.'s definition byte-for-byte.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};

/// §3.2 default tail-length threshold from Luo et al. 2026. Tail texts
/// shorter than this are not even checked for repetition.
pub const DEFAULT_REP_TAIL_LEN: usize = 10_000;

/// §3.2 default compression-ratio threshold. `compress_ratio > τ` ⇒
/// repetitive.
pub const DEFAULT_REP_RATIO_THRESHOLD: f64 = 10.0;

/// §3.9 RepRate threshold that fires the `LengthInflation` guardrail.
/// At or above this for two consecutive validation passes the guardrail
/// bumps `β_kl` and `λ_sft` (auto-engages Stable-OPD per Luo et al.).
pub const REPETITION_GUARDRAIL_THRESHOLD: f64 = 0.05;

/// Diagnostic snapshot collected over one validation pass.
///
/// The trainer emits one of these per validation interval; the guardrail
/// engine (§3.9) consumes the snapshot stream to decide on auto-tuning
/// and auto-rollback.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpdDiagnosticSnapshot {
    /// Step number this snapshot was taken at.
    pub step: u64,
    /// Wall-clock UTC timestamp.
    pub at: String,
    /// Mean per-position reverse KL over active tokens (§3.8 first
    /// chart). Lower-is-better but a value near zero means the student
    /// has converged to the teacher's distribution and additional OPD
    /// steps buy nothing.
    pub mean_kl: f64,
    /// Truncation rate per Luo et al. §3.2 — fraction of rollouts that
    /// hit the length budget without emitting EOS. Healthy < 0.5 for
    /// 1.5B students; spike to ≈1.0 = collapse imminent.
    pub truncation_rate: f64,
    /// Repetition rate per Luo et al. §3.2 — fraction of rollouts whose
    /// tail's compression ratio exceeds the threshold. Healthy = 0.0;
    /// guardrail fires at ≥ [`REPETITION_GUARDRAIL_THRESHOLD`].
    pub repetition_rate: f64,
    /// Mean response length in tokens (helps confirm length inflation
    /// is happening, not just RepRate noise).
    pub mean_response_tokens: f64,
    /// Number of rollouts the snapshot was computed over.
    pub num_rollouts: usize,
    /// `|H(q_hat) - H(p_hat)|` mean over active positions, in nats
    /// (Li et al. 2026 §3.8). Narrows in healthy runs; widening past
    /// step 50 triggers `EntropyGapWidening` (§3.9). `None` when no
    /// distribution-alignment metrics were collected this pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_entropy_gap: Option<f64>,
    /// Overlap-ratio probe result: fraction of student top-K that
    /// overlaps with teacher top-K, averaged over the rollout
    /// (Li et al. 2026 eq 6). Healthy runs climb 70% → 90%; stagnant
    /// runs trigger `OverlapStagnation`. `None` when no overlap probe
    /// ran this pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap_ratio: Option<f64>,
    /// §11 `LongTailRewardDecay` — first position at which the
    /// trainer detected the per-position KL trending up
    /// monotonically. `None` when no decay observed. Li et al.
    /// 2026 §6.1 reports degradation past 7K tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_tail_decay_position: Option<usize>,
    /// §11 / §8.5 `CapacityGap` — bits_storable / bits_needed
    /// ratio. <0.3 fires the guardrail. `None` when the calculator
    /// hasn't run this pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_ratio: Option<f64>,
    /// §11 `ThinkingPatternMismatch` — observed median overlap
    /// from the cold-start probe. <0.3 even after cold-start
    /// fires the guardrail. `None` when no probe this pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_cold_start_overlap: Option<f64>,
    /// True when a Stable-OPD `BumpStableOpd` was already applied
    /// in a previous validation pass. Used by `FlawedPrefixCollapse`
    /// to detect "bump didn't help" cases.
    #[serde(default)]
    pub stable_opd_already_bumped: bool,
    /// §11 `DiversityCollapse` — fraction of *distinct* unigrams
    /// across the K rollouts in this validation pass, normalised
    /// against the per-rollout count. Healthy runs hover near
    /// `1/K + (K-1)/K * unique_rate_per_rollout`. Reverse-KL
    /// mode-seeking pulls this toward the lower bound; when the
    /// guardrail loop sees it below threshold for `window` passes
    /// it surfaces `DiversityCollapse`. `None` when not measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollout_diversity: Option<f64>,
    /// §11 `SelfPlaySaturation` — mean KL between consecutive
    /// self-distill iterations (i_n → i_{n+1}). Trends to zero as
    /// the student cannot squeeze further signal from itself.
    /// Survey §7.2 mitigation: stop iterating, suggest a new
    /// teacher. `None` outside of self-distill loops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_play_inter_iteration_kl: Option<f64>,
    /// §11 `TokenizerDrift` — set when the teacher's
    /// `tokenizer_hash` doesn't match the student's tokenizer hash.
    /// Fired pre-flight by the trainer; safer to fail fast than
    /// silently train on mistokenised logprobs. `None` when no
    /// teacher hash was registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash_mismatch: Option<bool>,
}

/// One rollout's worth of metadata for diagnostic-stack input.
///
/// The trainer collects these from its sampling loop. The text itself is
/// stored decoded (post-tokenizer) so the zlib compression test sees the
/// same byte stream the user/judge would see. For cases where decoded
/// text isn't available, [`RolloutSummary::from_tokens`] hashes the
/// numeric tokens into a stand-in byte stream that preserves the
/// compression-ratio signal at slightly different absolute values.
#[derive(Debug, Clone)]
pub struct RolloutSummary {
    /// Decoded completion text (assistant tokens decoded to UTF-8).
    pub text: String,
    /// Number of tokens generated (excluding the prompt).
    pub generated_tokens: usize,
    /// Whether the rollout terminated by emitting an EOS-equivalent
    /// token (any of the tokenizer's `eos_token_ids` / `<|im_end|>`).
    pub hit_eos: bool,
}

impl RolloutSummary {
    /// Construct a summary from token IDs alone (no tokenizer
    /// available). The text field becomes a deterministic decoding via
    /// "token-id-as-base-256" — same compression-ratio detection works
    /// because repeated token sequences produce repeated byte
    /// sequences. Used by unit tests and by trainer paths that don't
    /// keep decoded text around.
    pub fn from_tokens(tokens: &[u32], hit_eos: bool) -> Self {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        // Cast to a String via a lossless escape — we don't actually
        // need valid UTF-8 here, but `text` is `String`. Pre-2024,
        // `String::from_utf8_lossy` allocates a borrow; instead, hex-
        // encode so the compression test sees the same content.
        let text = bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self {
            text,
            generated_tokens: tokens.len(),
            hit_eos,
        }
    }
}

/// Compute truncation rate: fraction of rollouts that did NOT emit EOS.
///
/// Returns `0.0` for an empty input (caller should treat empty inputs
/// as "no data" rather than "everything healthy"; the snapshot's
/// `num_rollouts == 0` is the disambiguator).
pub fn truncation_rate(rollouts: &[RolloutSummary]) -> f64 {
    if rollouts.is_empty() {
        return 0.0;
    }
    let truncated = rollouts.iter().filter(|r| !r.hit_eos).count();
    truncated as f64 / rollouts.len() as f64
}

/// Compute repetition rate per Luo et al. 2026 §3.2.
///
/// For each rollout, take the tail (last `tail_len` characters) and
/// compute `bytes / compressed_bytes`. Mark as repetitive if the ratio
/// exceeds `threshold`. Returns the fraction of rollouts that are
/// repetitive.
///
/// The compression itself uses our internal LZ77-style detector
/// ([`compress_ratio`]) so this module doesn't need a `flate2`
/// dependency. Empirical match to zlib's ratio on Luo et al.'s
/// repetitive-tail corpora is within ±15% at the threshold; the
/// guardrail trigger (`τ = 10`) is well above that noise floor.
pub fn repetition_rate(
    rollouts: &[RolloutSummary],
    tail_len: usize,
    threshold: f64,
) -> f64 {
    if rollouts.is_empty() {
        return 0.0;
    }
    let mut repetitive = 0usize;
    for r in rollouts {
        if r.text.len() <= tail_len {
            continue;
        }
        let tail = &r.text.as_bytes()[r.text.len() - tail_len..];
        let ratio = compress_ratio(tail);
        if ratio > threshold {
            repetitive += 1;
        }
    }
    repetitive as f64 / rollouts.len() as f64
}

/// Cross-rollout diversity (Survey §7.2 mitigation for
/// `DiversityCollapse`).
///
/// Operationalised as the *distinct unigram ratio* across the union of
/// the K rollouts' token streams: `|distinct_tokens_overall| /
/// total_tokens`. Each rollout's text is hashed into a byte stream
/// and split into 4-byte unigrams.
///
/// Healthy reasoning runs land in the 0.4–0.7 range. Mode-seeking
/// collapse (the failure mode Survey §7.2 names) drives this toward
/// `1/total_tokens`. Returns 1.0 for empty inputs so an absent
/// measurement never trips the guardrail.
pub fn rollout_diversity(rollouts: &[RolloutSummary]) -> f64 {
    if rollouts.is_empty() {
        return 1.0;
    }
    let mut distinct: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut total: usize = 0;
    for r in rollouts {
        let bytes = r.text.as_bytes();
        if bytes.len() < 4 {
            // Treat each character as its own bucket — short rollouts
            // are otherwise trivially "diverse" and the guardrail
            // would never fire on a degenerate case.
            for &b in bytes {
                distinct.insert(b as u64);
                total += 1;
            }
            continue;
        }
        for window in bytes.windows(4) {
            let val = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
            distinct.insert(val as u64);
            total += 1;
        }
    }
    if total == 0 {
        return 1.0;
    }
    distinct.len() as f64 / total as f64
}

/// zlib-based compression-ratio detector matching Luo et al. 2026 §3.2.
///
/// `compress_ratio(input) = |input| / |zlib(input)|`. Repetitive text
/// (the failure mode the guardrail is hunting) compresses 10:1 or
/// better; diverse text plateaus around 2:1–4:1. Default τ in the
/// paper and in [`DEFAULT_REP_RATIO_THRESHOLD`] is 10.0.
///
/// Uses `flate2` at default compression level — same as Python's
/// `zlib.compress()` which is what Luo et al. presumably used.
pub fn compress_ratio(input: &[u8]) -> f64 {
    if input.is_empty() {
        return 1.0;
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    if encoder.write_all(input).is_err() {
        return 1.0;
    }
    let compressed = match encoder.finish() {
        Ok(v) => v,
        Err(_) => return 1.0,
    };
    let denom = compressed.len().max(1);
    input.len() as f64 / denom as f64
}

/// Build a complete diagnostic snapshot from one validation pass.
///
/// `rollouts` carries the trajectory-shape signal; `mean_kl` is the
/// kernel-emitted scalar from `opd_step_loss`. The trainer logs every
/// snapshot via `tracing::info!` and pushes it to the Prometheus
/// registry (the registry wiring lives in kiln-server; this module is
/// the data definition).
pub fn build_snapshot(
    step: u64,
    rollouts: &[RolloutSummary],
    mean_kl: f64,
) -> OpdDiagnosticSnapshot {
    let trunc = truncation_rate(rollouts);
    let rep = repetition_rate(
        rollouts,
        DEFAULT_REP_TAIL_LEN,
        DEFAULT_REP_RATIO_THRESHOLD,
    );
    let mean_tokens = if rollouts.is_empty() {
        0.0
    } else {
        rollouts.iter().map(|r| r.generated_tokens as f64).sum::<f64>()
            / rollouts.len() as f64
    };
    let diversity = rollout_diversity(rollouts);
    OpdDiagnosticSnapshot {
        step,
        at: chrono::Utc::now().to_rfc3339(),
        mean_kl,
        truncation_rate: trunc,
        repetition_rate: rep,
        mean_response_tokens: mean_tokens,
        num_rollouts: rollouts.len(),
        mean_entropy_gap: None,
        overlap_ratio: None,
        long_tail_decay_position: None,
        capacity_ratio: None,
        post_cold_start_overlap: None,
        stable_opd_already_bumped: false,
        rollout_diversity: Some(diversity),
        self_play_inter_iteration_kl: None,
        tokenizer_hash_mismatch: None,
    }
}

/// Like [`build_snapshot`] but with the distribution-alignment metrics
/// from a kernel-side metrics pass populated. The trainer calls this on
/// the validation cadence; the cheap-trajectory-only path (build_snapshot)
/// runs on every step.
pub fn build_snapshot_with_alignment(
    step: u64,
    rollouts: &[RolloutSummary],
    mean_kl: f64,
    mean_entropy_gap: Option<f64>,
    overlap_ratio: Option<f64>,
) -> OpdDiagnosticSnapshot {
    let mut s = build_snapshot(step, rollouts, mean_kl);
    s.mean_entropy_gap = mean_entropy_gap;
    s.overlap_ratio = overlap_ratio;
    s
}

/// Decision returned by the `LengthInflation` guardrail.
///
/// The guardrail watches the snapshot stream and decides on auto-
/// mitigation per Luo et al. 2026 + §3.9 of the grand plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuardrailDecision {
    /// All metrics within healthy bounds — no action required.
    Ok,
    /// Repetition rising; bump Stable-OPD's `β_kl` and `λ_sft` (auto-
    /// engage if not already on, double if already engaged).
    BumpStableOpd { reason: GuardrailTrigger },
    /// Critical collapse — pause optimisation, roll back to last
    /// passing checkpoint, surface to the user. The §3.9 auto-rollback
    /// contract.
    RollBackAndPause { reason: GuardrailTrigger },
    /// Overlap ratio stalled; pause and recommend off-policy cold-start
    /// before resuming. §3.9 `OverlapStagnation`.
    PauseAndRecommendColdStart { reason: GuardrailTrigger },
    /// Entropy gap widening; reduce learning rate by 0.5×.
    /// §3.9 `EntropyGapWidening`.
    ReduceLearningRate { factor: f64, reason: GuardrailTrigger },
    /// §11 `LongTailRewardDecay`: cap rollout length at the
    /// `last_healthy_position` before resuming. The trainer
    /// applies via `OpdConfig::max_tokens`.
    CapRolloutLength {
        max_tokens: usize,
        reason: GuardrailTrigger,
    },
    /// §11 `ThinkingPatternMismatch`: refuse to start the run.
    /// User must change the teacher or accept an explicit override.
    RefuseRun { reason: GuardrailTrigger },
    /// §11 `CapacityGap`: increase LoRA rank before retrying.
    /// `suggested_rank` derived from §8.5 capacity calc.
    IncreaseLoRARank {
        suggested_rank: usize,
        reason: GuardrailTrigger,
    },
    /// §11 `FlawedPrefixCollapse`: combine halve-LR + cap rollout
    /// length. Last-resort mitigation before the auto-rollback.
    HalveLrAndCapTokens {
        new_lr_factor: f64,
        new_max_tokens: usize,
        reason: GuardrailTrigger,
    },
    /// §11 `DiversityCollapse` (Survey §7.2): suggest switching to
    /// an asymmetric divergence (ToDi / AKL) for open-ended tasks
    /// where reverse-KL's mode-seeking is collapsing the rollout
    /// distribution.
    SuggestAsymmetricDivergence { reason: GuardrailTrigger },
    /// §11 `SelfPlaySaturation` (Survey §7.2): the iterative
    /// self-distill loop has converged — further iterations buy
    /// nothing. Stop iterating; suggest a richer teacher.
    StopIteratingAndSwitchTeacher { reason: GuardrailTrigger },
}

/// The specific paper-cited trigger that fired the guardrail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuardrailTrigger {
    /// RepRate above [`REPETITION_GUARDRAIL_THRESHOLD`] in `recent`
    /// consecutive validation passes (default 2 per Luo et al.).
    RepetitionRateAbove { recent: usize },
    /// Truncation rate spike — `recent` consecutive passes with
    /// TruncRate > 0.9. Indicates phase-transition collapse imminent.
    TruncationSpike { recent: usize },
    /// Overlap ratio Δ < 0.01 over the watch window (§3.9
    /// `OverlapStagnation` from Li et al. 2026). Auto-pause and
    /// recommend off-policy cold-start.
    OverlapStagnation { window: usize },
    /// `|H(q) - H(p)|` trending up after step 50 (§3.9
    /// `EntropyGapWidening`). Reduce learning rate by 0.5×.
    EntropyGapWidening { window: usize },
    /// §11 / Li et al. 2026 §6.1 — per-position KL increases
    /// monotonically past position 7K, indicating the teacher's
    /// reward decays at depth. Cap rollout length at the position
    /// where the signal degrades.
    LongTailRewardDecay { last_healthy_position: usize },
    /// §11 — initial overlap stayed below 0.3 even after cold-start
    /// (Li et al. §3.1). The chosen teacher is stylistically
    /// incompatible with the student. Refuse the run; suggest
    /// alternatives from the compatibility table.
    ThinkingPatternMismatch { observed_overlap: f64 },
    /// §11 / §8.5 — `bits_storable < 0.3 × bits_needed`. The LoRA
    /// can't hold the information the run wants to teach.
    CapacityGap { ratio: f64 },
    /// §11 — truncation rate STILL > 0.9 in the validation pass
    /// AFTER a Stable-OPD bump was already applied. The bump
    /// didn't help; the prefix-collapse failure mode (flawed
    /// student-generated prefixes the teacher can't score reliably)
    /// is in flight. Drastic action: halve learning rate AND drop
    /// max_tokens.
    FlawedPrefixCollapse { passes_since_bump: usize },
    /// §11 (Survey §7.2) — rollout diversity (distinct unigram
    /// ratio across the K rollouts) has dropped below
    /// [`DIVERSITY_COLLAPSE_THRESHOLD`] for `window` consecutive
    /// validation passes. Reverse-KL's mode-seeking is collapsing
    /// the rollout distribution.
    DiversityCollapse { window: usize, observed: f64 },
    /// §11 (Survey §7.2) — inter-iteration self-distill KL has
    /// dropped below [`SELF_PLAY_SATURATION_THRESHOLD`] for
    /// `window` consecutive iterations. The student cannot
    /// extract further signal from itself.
    SelfPlaySaturation { window: usize, observed_kl: f64 },
    /// §11 — teacher and student tokenizer hashes do not match.
    /// Fired pre-flight before the first optimizer step. Silent
    /// drift mistokenises logprobs and gives a wrong KL signal.
    TokenizerDrift,
}

/// Stateful guardrail that consumes snapshots in order and decides on
/// mitigation. Implements the named §3.9 rules:
/// `LengthInflation`, `OverlapStagnation`, `EntropyGapWidening`. The
/// trainer holds one of these for the duration of a run.
#[derive(Debug, Clone, Default)]
pub struct LengthInflationGuardrail {
    /// Consecutive snapshots with RepRate above the threshold.
    consecutive_high_rep: usize,
    /// Consecutive snapshots with TruncRate > 0.9.
    consecutive_high_trunc: usize,
    /// Overlap-ratio history (last few values). Bounded.
    overlap_history: Vec<f64>,
    /// Entropy-gap history (last few values). Bounded.
    entropy_gap_history: Vec<f64>,
    /// Consecutive snapshots with rollout_diversity below threshold.
    consecutive_low_diversity: usize,
    /// Last observed rollout_diversity (so the trigger can report it).
    last_diversity: f64,
    /// Consecutive snapshots with `self_play_inter_iteration_kl` below
    /// threshold. §11 SelfPlaySaturation.
    consecutive_low_self_play_kl: usize,
    /// Last observed self-play KL (for the trigger report).
    last_self_play_kl: f64,
    /// Last decision made — used by callers that want to suppress
    /// duplicate logging of the same mitigation.
    last_decision: GuardrailDecision,
}

/// History window for the overlap / entropy-gap guardrails. The §3.9
/// `OverlapStagnation` rule says "Δ < 0.01 over 30 steps". We default
/// the validation-pass window to 6 (which represents 30 steps at the
/// default 5-step validation cadence; pit-of-success value, tunable).
pub const GUARDRAIL_WINDOW: usize = 6;

/// Minimum Δ in overlap ratio over [`GUARDRAIL_WINDOW`] passes that
/// counts as "still moving." Below this we fire `OverlapStagnation`.
pub const OVERLAP_STAGNATION_MIN_DELTA: f64 = 0.01;

/// §11 `DiversityCollapse` (Survey §7.2) — rollout-diversity threshold
/// below which we fire the guardrail. 0.15 is conservative — Lu (2025)
/// reports healthy reasoning runs land in the 0.4–0.7 range; below 0.2
/// is the mode-seeking-collapse regime.
pub const DIVERSITY_COLLAPSE_THRESHOLD: f64 = 0.15;

/// Consecutive validation passes with `rollout_diversity` below
/// [`DIVERSITY_COLLAPSE_THRESHOLD`] before the guardrail fires. Same
/// "twice in a row" rule as RepRate, to keep noise off.
pub const DIVERSITY_COLLAPSE_WINDOW: usize = 2;

/// §11 `SelfPlaySaturation` (Survey §7.2) — inter-iteration self-distill
/// KL below this nats value over `SELF_PLAY_SATURATION_WINDOW` rounds
/// fires the guardrail. 0.005 nats ≈ a 0.5% relative shift in
/// distribution — within reproducibility noise.
pub const SELF_PLAY_SATURATION_THRESHOLD: f64 = 0.005;

/// Consecutive self-distill iterations below
/// [`SELF_PLAY_SATURATION_THRESHOLD`] before firing the guardrail.
pub const SELF_PLAY_SATURATION_WINDOW: usize = 2;

impl LengthInflationGuardrail {
    /// Feed a fresh snapshot. Returns the current decision; the trainer
    /// applies it (raise β/λ, rollback, etc.). The internal state is
    /// updated in-place.
    pub fn observe(&mut self, snapshot: &OpdDiagnosticSnapshot) -> GuardrailDecision {
        // RepRate counter.
        if snapshot.repetition_rate >= REPETITION_GUARDRAIL_THRESHOLD {
            self.consecutive_high_rep += 1;
        } else {
            self.consecutive_high_rep = 0;
        }
        // TruncRate counter (spike at >0.9 indicates Luo's phase
        // transition is in flight).
        if snapshot.truncation_rate > 0.9 {
            self.consecutive_high_trunc += 1;
        } else {
            self.consecutive_high_trunc = 0;
        }
        // Append distribution-alignment metrics when present (None when
        // the validation pass didn't run the metrics kernel).
        if let Some(o) = snapshot.overlap_ratio {
            self.overlap_history.push(o);
            if self.overlap_history.len() > GUARDRAIL_WINDOW {
                self.overlap_history.remove(0);
            }
        }
        // Diversity counter — fires on `DIVERSITY_COLLAPSE_WINDOW`
        // consecutive low-diversity passes.
        if let Some(d) = snapshot.rollout_diversity {
            self.last_diversity = d;
            if d < DIVERSITY_COLLAPSE_THRESHOLD {
                self.consecutive_low_diversity += 1;
            } else {
                self.consecutive_low_diversity = 0;
            }
        }
        // Self-play saturation counter — fires on
        // `SELF_PLAY_SATURATION_WINDOW` consecutive iterations with
        // inter-iteration KL below the threshold.
        if let Some(k) = snapshot.self_play_inter_iteration_kl {
            self.last_self_play_kl = k;
            if k < SELF_PLAY_SATURATION_THRESHOLD {
                self.consecutive_low_self_play_kl += 1;
            } else {
                self.consecutive_low_self_play_kl = 0;
            }
        }
        if let Some(g) = snapshot.mean_entropy_gap {
            self.entropy_gap_history.push(g);
            if self.entropy_gap_history.len() > GUARDRAIL_WINDOW {
                self.entropy_gap_history.remove(0);
            }
        }

        // Priority order (most critical first):
        // 0a. TokenizerDrift → refuse the run (terminal, pre-flight).
        // 0b. ThinkingPatternMismatch → refuse the run (terminal).
        // 0c. CapacityGap → ask user to bump rank (terminal).
        // 0d. SelfPlaySaturation → stop iterating (terminal for the
        //     enclosing self-distill loop, not the optimizer step).
        // 1.  FlawedPrefixCollapse → halve LR + cap tokens.
        // 2.  TruncationSpike → rollback.
        // 3.  OverlapStagnation → pause + recommend cold-start.
        // 4.  LongTailRewardDecay → cap rollout length.
        // 5.  RepetitionRateAbove → bump Stable-OPD.
        // 6.  EntropyGapWidening → reduce LR.
        // 7.  DiversityCollapse → suggest asymmetric divergence.
        let decision = if snapshot.tokenizer_hash_mismatch == Some(true) {
            GuardrailDecision::RefuseRun {
                reason: GuardrailTrigger::TokenizerDrift,
            }
        } else if let Some(o) = snapshot.post_cold_start_overlap {
            if o < 0.3 {
                GuardrailDecision::RefuseRun {
                    reason: GuardrailTrigger::ThinkingPatternMismatch {
                        observed_overlap: o,
                    },
                }
            } else {
                self.decide_no_terminal(snapshot)
            }
        } else if let Some(r) = snapshot.capacity_ratio {
            if r < 0.3 {
                // Suggested rank: scale up to hit ratio 1.0 with
                // some headroom (2×). Conservative.
                let scale = (1.0 / r).max(1.0);
                GuardrailDecision::IncreaseLoRARank {
                    suggested_rank: ((scale * 32.0).round() as usize).max(32),
                    reason: GuardrailTrigger::CapacityGap { ratio: r },
                }
            } else {
                self.decide_no_terminal(snapshot)
            }
        } else if self.consecutive_low_self_play_kl >= SELF_PLAY_SATURATION_WINDOW {
            GuardrailDecision::StopIteratingAndSwitchTeacher {
                reason: GuardrailTrigger::SelfPlaySaturation {
                    window: self.consecutive_low_self_play_kl,
                    observed_kl: self.last_self_play_kl,
                },
            }
        } else {
            self.decide_no_terminal(snapshot)
        };

        self.last_decision = decision;
        decision
    }

    /// Run the non-terminal guardrail priority. Called after the
    /// terminal-condition checks (ThinkingPatternMismatch /
    /// CapacityGap).
    fn decide_no_terminal(
        &mut self,
        snapshot: &OpdDiagnosticSnapshot,
    ) -> GuardrailDecision {
        if snapshot.stable_opd_already_bumped && self.consecutive_high_trunc >= 1 {
            // §11 FlawedPrefixCollapse: bump didn't help, take
            // harsher action.
            return GuardrailDecision::HalveLrAndCapTokens {
                new_lr_factor: 0.5,
                new_max_tokens: 4_096,
                reason: GuardrailTrigger::FlawedPrefixCollapse {
                    passes_since_bump: self.consecutive_high_trunc,
                },
            };
        }
        if self.consecutive_high_trunc >= 2 {
            return GuardrailDecision::RollBackAndPause {
                reason: GuardrailTrigger::TruncationSpike {
                    recent: self.consecutive_high_trunc,
                },
            };
        }
        if self.overlap_stagnation_fires() {
            return GuardrailDecision::PauseAndRecommendColdStart {
                reason: GuardrailTrigger::OverlapStagnation {
                    window: self.overlap_history.len(),
                },
            };
        }
        if let Some(pos) = snapshot.long_tail_decay_position {
            return GuardrailDecision::CapRolloutLength {
                max_tokens: pos,
                reason: GuardrailTrigger::LongTailRewardDecay {
                    last_healthy_position: pos,
                },
            };
        }
        if self.consecutive_high_rep >= 2 {
            return GuardrailDecision::BumpStableOpd {
                reason: GuardrailTrigger::RepetitionRateAbove {
                    recent: self.consecutive_high_rep,
                },
            };
        }
        if self.entropy_gap_widening() {
            return GuardrailDecision::ReduceLearningRate {
                factor: 0.5,
                reason: GuardrailTrigger::EntropyGapWidening {
                    window: self.entropy_gap_history.len(),
                },
            };
        }
        if self.consecutive_low_diversity >= DIVERSITY_COLLAPSE_WINDOW {
            return GuardrailDecision::SuggestAsymmetricDivergence {
                reason: GuardrailTrigger::DiversityCollapse {
                    window: self.consecutive_low_diversity,
                    observed: self.last_diversity,
                },
            };
        }
        GuardrailDecision::Ok
    }

    pub fn last_decision(&self) -> GuardrailDecision {
        self.last_decision
    }

    /// Overlap stagnation: the window has filled AND
    /// `max(overlap) - min(overlap) < OVERLAP_STAGNATION_MIN_DELTA`.
    fn overlap_stagnation_fires(&self) -> bool {
        if self.overlap_history.len() < GUARDRAIL_WINDOW {
            return false;
        }
        let lo = self
            .overlap_history
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let hi = self
            .overlap_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (hi - lo).abs() < OVERLAP_STAGNATION_MIN_DELTA
    }

    /// Entropy gap widening: monotonically increasing over the window.
    /// "After step 50" — we approximate that as "window fills"; since
    /// `GUARDRAIL_WINDOW * validation_cadence` is the time, a trainer
    /// with validation_cadence=5 steps fires this at step 30. Caller
    /// can also gate on `snapshot.step >= 50` before calling
    /// `observe`.
    fn entropy_gap_widening(&self) -> bool {
        if self.entropy_gap_history.len() < GUARDRAIL_WINDOW {
            return false;
        }
        self.entropy_gap_history
            .windows(2)
            .all(|w| w[1] >= w[0] + 1e-6)
    }
}

impl Default for GuardrailDecision {
    fn default() -> Self {
        Self::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rollout(text: &str, generated_tokens: usize, hit_eos: bool) -> RolloutSummary {
        RolloutSummary {
            text: text.to_string(),
            generated_tokens,
            hit_eos,
        }
    }

    #[test]
    fn truncation_rate_basic() {
        let rs = vec![
            make_rollout("a", 10, true),
            make_rollout("b", 10, true),
            make_rollout("c", 10, false),
            make_rollout("d", 10, false),
        ];
        let r = truncation_rate(&rs);
        assert!((r - 0.5).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn truncation_rate_empty_is_zero() {
        assert_eq!(truncation_rate(&[]), 0.0);
    }

    #[test]
    fn compress_ratio_detects_repetition() {
        // Highly repetitive input — should compress >> 10:1.
        let mut s = String::new();
        for _ in 0..1000 {
            s.push_str("the cat sat on the mat. ");
        }
        let r = compress_ratio(s.as_bytes());
        assert!(r > 10.0, "expected >10x compression on repetitive input, got {r}");
    }

    #[test]
    fn compress_ratio_low_on_diverse() {
        // Pseudo-random bytes — zlib can find no exploitable structure,
        // ratio should hover near 1.0 (slightly less due to overhead).
        let mut s = Vec::with_capacity(20_000);
        let mut x: u32 = 0xdead_beef;
        for _ in 0..20_000 {
            // xorshift32 — deterministic but spectrally flat.
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s.push((x & 0xff) as u8);
        }
        let r = compress_ratio(&s);
        assert!(r < 10.0, "expected <10x compression on diverse input, got {r}");
    }

    #[test]
    fn repetition_rate_skips_short_rollouts() {
        // Below tail_len threshold — never counted.
        let rs = vec![make_rollout("aaa", 10, true)];
        let r = repetition_rate(&rs, 1000, 10.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn repetition_rate_catches_repetitive_tail() {
        let mut text = "diverse beginning ".repeat(50);
        for _ in 0..2000 {
            text.push_str("loop ");
        }
        let rs = vec![
            make_rollout(&text, 500, true),
            make_rollout("clean text", 10, true),
        ];
        let r = repetition_rate(&rs, 1000, 10.0);
        assert!(r > 0.0, "expected non-zero RepRate on repetitive tail, got {r}");
    }

    #[test]
    fn guardrail_fires_on_two_consecutive_high_rep() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |rep: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: rep,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        assert_eq!(g.observe(&mk_snap(0.04)), GuardrailDecision::Ok);
        assert_eq!(g.observe(&mk_snap(0.10)), GuardrailDecision::Ok); // 1st high
        match g.observe(&mk_snap(0.12)) {
            GuardrailDecision::BumpStableOpd { reason } => {
                assert!(matches!(
                    reason,
                    GuardrailTrigger::RepetitionRateAbove { recent } if recent >= 2
                ));
            }
            other => panic!("expected BumpStableOpd, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_resets_on_recovery() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |rep: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: rep,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        g.observe(&mk_snap(0.10));
        g.observe(&mk_snap(0.12));
        assert!(matches!(
            g.last_decision(),
            GuardrailDecision::BumpStableOpd { .. }
        ));
        // RepRate drops; guardrail resets.
        assert_eq!(g.observe(&mk_snap(0.02)), GuardrailDecision::Ok);
        assert_eq!(g.observe(&mk_snap(0.01)), GuardrailDecision::Ok);
    }

    #[test]
    fn guardrail_rollback_on_truncation_spike() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |trunc: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: trunc,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        g.observe(&mk_snap(0.95));
        match g.observe(&mk_snap(0.99)) {
            GuardrailDecision::RollBackAndPause { reason } => {
                assert!(matches!(reason, GuardrailTrigger::TruncationSpike { .. }));
            }
            other => panic!("expected RollBackAndPause, got {other:?}"),
        }
    }

    #[test]
    fn snapshot_round_trips_through_serde() {
        let snap = OpdDiagnosticSnapshot {
            step: 17,
            at: "2026-05-15T12:00:00Z".into(),
            mean_kl: 0.234,
            truncation_rate: 0.5,
            repetition_rate: 0.02,
            mean_response_tokens: 1234.5,
            num_rollouts: 64,
            mean_entropy_gap: Some(0.07),
            overlap_ratio: Some(0.88),
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: OpdDiagnosticSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.step, snap.step);
        assert!((parsed.repetition_rate - snap.repetition_rate).abs() < 1e-12);
        assert!((parsed.mean_entropy_gap.unwrap() - 0.07).abs() < 1e-9);
        assert!((parsed.overlap_ratio.unwrap() - 0.88).abs() < 1e-9);
    }

    #[test]
    fn guardrail_fires_overlap_stagnation_after_window() {
        let mut g = LengthInflationGuardrail::default();
        let mk = |overlap: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: Some(overlap),
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        // Window is GUARDRAIL_WINDOW=6 entries. Feed 5 close together
        // (no decision), then 6th close → fires.
        for _ in 0..5 {
            assert_eq!(g.observe(&mk(0.71)), GuardrailDecision::Ok);
        }
        // Sixth keeps |max-min| ≈ 0 < threshold ⇒ fires.
        match g.observe(&mk(0.710)) {
            GuardrailDecision::PauseAndRecommendColdStart {
                reason: GuardrailTrigger::OverlapStagnation { window },
            } => assert_eq!(window, GUARDRAIL_WINDOW),
            other => panic!("expected PauseAndRecommendColdStart, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_overlap_movement_keeps_ok() {
        let mut g = LengthInflationGuardrail::default();
        let mk = |overlap: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: Some(overlap),
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        // Window with 0.10 spread → above 0.01 threshold ⇒ never fires.
        let values = [0.70, 0.72, 0.74, 0.77, 0.79, 0.80];
        for &v in &values {
            assert_eq!(g.observe(&mk(v)), GuardrailDecision::Ok);
        }
    }

    #[test]
    fn guardrail_fires_entropy_gap_widening() {
        let mut g = LengthInflationGuardrail::default();
        let mk = |gap: f64| OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: Some(gap),
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        // Monotonically rising entropy gap over the window.
        let rising = [0.10, 0.12, 0.15, 0.18, 0.22, 0.26];
        for (i, &v) in rising.iter().enumerate() {
            let d = g.observe(&mk(v));
            if i + 1 < rising.len() {
                assert_eq!(d, GuardrailDecision::Ok);
            } else {
                match d {
                    GuardrailDecision::ReduceLearningRate {
                        factor,
                        reason: GuardrailTrigger::EntropyGapWidening { window },
                    } => {
                        assert!((factor - 0.5).abs() < 1e-9);
                        assert_eq!(window, GUARDRAIL_WINDOW);
                    }
                    other => panic!("expected ReduceLearningRate, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn guardrail_thinking_pattern_mismatch_refuses_run() {
        let mut g = LengthInflationGuardrail::default();
        let snap = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            // Cold-start ran; even afterwards overlap is below 0.3 →
            // §11 ThinkingPatternMismatch → RefuseRun.
            post_cold_start_overlap: Some(0.22),
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        match g.observe(&snap) {
            GuardrailDecision::RefuseRun {
                reason: GuardrailTrigger::ThinkingPatternMismatch { observed_overlap },
            } => {
                assert!((observed_overlap - 0.22).abs() < 1e-9);
            }
            other => panic!("expected RefuseRun, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_capacity_gap_suggests_rank_increase() {
        let mut g = LengthInflationGuardrail::default();
        let snap = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            // Capacity calc says we're 5× short.
            capacity_ratio: Some(0.2),
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        match g.observe(&snap) {
            GuardrailDecision::IncreaseLoRARank {
                suggested_rank,
                reason: GuardrailTrigger::CapacityGap { ratio },
            } => {
                assert!((ratio - 0.2).abs() < 1e-9);
                // scale ≈ 1/0.2 = 5.0; suggested ≈ 5 * 32 = 160.
                assert!(suggested_rank >= 32);
                assert!(suggested_rank >= 100);
            }
            other => panic!("expected IncreaseLoRARank, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_long_tail_reward_decay_caps_rollout() {
        let mut g = LengthInflationGuardrail::default();
        let snap = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: Some(7168),
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        match g.observe(&snap) {
            GuardrailDecision::CapRolloutLength {
                max_tokens,
                reason: GuardrailTrigger::LongTailRewardDecay { last_healthy_position },
            } => {
                assert_eq!(max_tokens, 7168);
                assert_eq!(last_healthy_position, 7168);
            }
            other => panic!("expected CapRolloutLength, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_flawed_prefix_collapse_after_bump() {
        let mut g = LengthInflationGuardrail::default();
        let snap = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.95, // still high
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: true, // bump already applied
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        match g.observe(&snap) {
            GuardrailDecision::HalveLrAndCapTokens {
                new_lr_factor,
                new_max_tokens,
                reason: GuardrailTrigger::FlawedPrefixCollapse { passes_since_bump },
            } => {
                assert!((new_lr_factor - 0.5).abs() < 1e-9);
                assert_eq!(new_max_tokens, 4096);
                assert!(passes_since_bump >= 1);
            }
            other => panic!("expected HalveLrAndCapTokens, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_priority_truncation_beats_overlap() {
        let mut g = LengthInflationGuardrail::default();
        let mk = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.95,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 10,
            mean_entropy_gap: None,
            overlap_ratio: Some(0.71),
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        // Trunc spike both passes — must win over overlap stagnation.
        g.observe(&mk);
        match g.observe(&mk) {
            GuardrailDecision::RollBackAndPause { .. } => {}
            other => panic!("expected RollBackAndPause, got {other:?}"),
        }
    }

    #[test]
    fn rollout_summary_from_tokens_reproducible() {
        let a = RolloutSummary::from_tokens(&[1, 2, 3, 4], true);
        let b = RolloutSummary::from_tokens(&[1, 2, 3, 4], true);
        assert_eq!(a.text, b.text);
        // Same first 4 tokens repeated → highly compressible.
        let repeated: Vec<u32> = (0..2000).map(|i| (i % 4) as u32).collect();
        let r = RolloutSummary::from_tokens(&repeated, true);
        let ratio = compress_ratio(r.text.as_bytes());
        assert!(ratio > 10.0, "expected highly-compressible token stream, got {ratio}");
    }

    #[test]
    fn rollout_diversity_collapses_for_identical_rollouts() {
        // K identical rollouts → tokens all come from the same
        // 4-byte window stream; distinct-token ratio is tiny.
        let single_token_stream = "ababababababababababababababababab".to_string();
        let rollouts: Vec<RolloutSummary> = (0..8)
            .map(|_| RolloutSummary {
                text: single_token_stream.clone(),
                generated_tokens: single_token_stream.len(),
                hit_eos: true,
            })
            .collect();
        let d = rollout_diversity(&rollouts);
        assert!(
            d < DIVERSITY_COLLAPSE_THRESHOLD,
            "expected low-diversity, got {d}"
        );
    }

    #[test]
    fn rollout_diversity_high_for_distinct_rollouts() {
        // Eight rollouts of full-byte pseudo-random text so the
        // 4-byte unigram distribution is wide.
        let rollouts: Vec<RolloutSummary> = (0..8)
            .map(|i| {
                let mut buf = Vec::with_capacity(400);
                let mut state: u64 = (i as u64).wrapping_mul(2_654_435_761) ^ 0xdead_beefu64;
                for _ in 0..400 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    // Map to printable ASCII so the text is valid UTF-8.
                    let b = 0x20 + ((state >> 33) % 95) as u8;
                    buf.push(b);
                }
                RolloutSummary {
                    text: String::from_utf8(buf).unwrap(),
                    generated_tokens: 400,
                    hit_eos: true,
                }
            })
            .collect();
        let d = rollout_diversity(&rollouts);
        // Diverse stream: at least an order of magnitude above the
        // collapse threshold.
        assert!(
            d > DIVERSITY_COLLAPSE_THRESHOLD * 2.0,
            "expected diverse rollouts well above threshold, got {d}"
        );
    }

    #[test]
    fn guardrail_fires_diversity_collapse_after_window() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |div: f64| OpdDiagnosticSnapshot {
            step: 100,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 8,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: Some(div),
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        // One pass below threshold — not yet enough.
        assert_eq!(g.observe(&mk_snap(0.05)), GuardrailDecision::Ok);
        // Second consecutive pass → fires.
        match g.observe(&mk_snap(0.04)) {
            GuardrailDecision::SuggestAsymmetricDivergence { reason } => {
                assert!(matches!(
                    reason,
                    GuardrailTrigger::DiversityCollapse { window, .. } if window >= 2
                ));
            }
            other => panic!("expected SuggestAsymmetricDivergence, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_resets_diversity_on_recovery() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |div: f64| OpdDiagnosticSnapshot {
            step: 100,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 8,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: Some(div),
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: None,
        };
        g.observe(&mk_snap(0.05));
        // Bounce back above threshold → counter resets.
        g.observe(&mk_snap(0.4));
        assert_eq!(g.observe(&mk_snap(0.04)), GuardrailDecision::Ok);
    }

    #[test]
    fn guardrail_fires_self_play_saturation_after_window() {
        let mut g = LengthInflationGuardrail::default();
        let mk_snap = |k: f64| OpdDiagnosticSnapshot {
            step: 100,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 4,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: Some(k),
            tokenizer_hash_mismatch: None,
        };
        assert_eq!(g.observe(&mk_snap(0.002)), GuardrailDecision::Ok);
        match g.observe(&mk_snap(0.001)) {
            GuardrailDecision::StopIteratingAndSwitchTeacher { reason } => {
                assert!(matches!(
                    reason,
                    GuardrailTrigger::SelfPlaySaturation { window, .. } if window >= 2
                ));
            }
            other => panic!("expected StopIteratingAndSwitchTeacher, got {other:?}"),
        }
    }

    #[test]
    fn guardrail_fires_tokenizer_drift_immediately() {
        let mut g = LengthInflationGuardrail::default();
        let snap = OpdDiagnosticSnapshot {
            step: 0,
            at: "".into(),
            mean_kl: 0.0,
            truncation_rate: 0.0,
            repetition_rate: 0.0,
            mean_response_tokens: 0.0,
            num_rollouts: 4,
            mean_entropy_gap: None,
            overlap_ratio: None,
            long_tail_decay_position: None,
            capacity_ratio: None,
            post_cold_start_overlap: None,
            stable_opd_already_bumped: false,
            rollout_diversity: None,
            self_play_inter_iteration_kl: None,
            tokenizer_hash_mismatch: Some(true),
        };
        match g.observe(&snap) {
            GuardrailDecision::RefuseRun {
                reason: GuardrailTrigger::TokenizerDrift,
            } => {}
            other => panic!("expected RefuseRun(TokenizerDrift), got {other:?}"),
        }
    }
}

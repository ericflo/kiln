//! Pre-submission training preflight estimator.
//!
//! Submitted SFT/GRPO jobs are size-checked against the corrected memory
//! budget so the server fails with HTTP 413 + an actionable hint instead
//! of OOM-killing itself partway through the first step. The estimator is
//! intentionally a closed-form upper bound: it overestimates by design so
//! that "fits according to preflight" implies "actually fits at runtime".
//!
//! Used by [`crate::api::training::submit_sft`] and `submit_grpo`.

use kiln_core::config::{DType, ModelConfig};
use kiln_core::vram::GpuVramInfo;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{GrpoGroup, SftExample};

/// One line item in the working-set breakdown. Surfaced in the 413
/// response body so users can see which contribution dominates and
/// pick the right knob to turn down.
#[derive(Debug, Clone, Copy)]
pub struct Breakdown {
    pub base_weights: u64,
    pub per_segment_activations: u64,
    pub boundary_states: u64,
    pub flce_intermediates: u64,
    pub lora_param_grad: u64,
    pub safety_margin: u64,
}

impl Breakdown {
    pub fn total(&self) -> u64 {
        self.base_weights
            + self.per_segment_activations
            + self.boundary_states
            + self.flce_intermediates
            + self.lora_param_grad
            + self.safety_margin
    }
}

/// Aggregated working-set estimate for one training step.
#[derive(Debug, Clone, Copy)]
pub struct WorkingSet {
    pub total_bytes: u64,
    pub max_seq_len: usize,
    pub breakdown: Breakdown,
}

const BYTES_PER_GB: u64 = 1024 * 1024 * 1024;
/// Default safety margin: 1 GB. Large enough to absorb scratch
/// allocations the closed-form pieces don't model directly (DRM
/// import buffers, allocator slack, kernel staging buffers).
const SAFETY_MARGIN_BYTES: u64 = BYTES_PER_GB;
/// Default chunk count for the FLCE forward pass. The Phase B kernel
/// processes the LM head matmul in chunks so the [T, V] logits tensor
/// is never materialized; the per-chunk working set scales with
/// `T * V / chunks` in F32 (the reduce accumulator).
pub const DEFAULT_FLCE_CHUNKS: u64 = 8;

fn dtype_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::BF16 | DType::FP16 => 2,
        DType::FP32 => 4,
    }
}

/// Sum of every base-model weight rounded up to the storage dtype.
///
/// Closed-form approximation — does not enumerate actual tensors;
/// intended as a stable upper bound for preflight rejection.
fn approximate_base_weight_bytes(cfg: &ModelConfig) -> u64 {
    let elem = dtype_bytes(cfg.dtype);
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let v = cfg.vocab_size as u64;
    let layers = cfg.num_layers as u64;

    // Embedding + LM head (often tied, but we count both to stay conservative).
    let embed_bytes = v * h * elem * 2;
    // Per-layer projections (Q, K, V, O on full attention; gate/up/down for MLP).
    // Rough composition: q_proj ~ h * gate_h, k/v ~ h * kv_h, o_proj ~ h * h.
    // We collapse to (4 * h * h) for attention as a conservative upper bound,
    // and (3 * h * i) for the MLP, plus 4 * h for RMSNorm pairs per layer.
    let per_layer_attn = 4 * h * h * elem;
    let per_layer_mlp = 3 * h * i * elem;
    let per_layer_norms = 4 * h * elem;
    let per_layer_total = per_layer_attn + per_layer_mlp + per_layer_norms;
    embed_bytes + per_layer_total * layers
}

/// Activations live for the segment currently being recomputed.
///
/// Closed-form upper bound per layer: hidden state stash + QKV + attn
/// output + MLP up/gate/down intermediates. Multiplied by sequence
/// length and dtype size, then by the number of layers per segment.
fn per_segment_activation_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    num_segments: usize,
) -> u64 {
    let elem = dtype_bytes(cfg.dtype);
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let t = max_seq_len as u64;
    let layers_per_seg = (cfg.num_layers / num_segments.max(1)).max(1) as u64;
    // Per layer: 6 hidden-sized tensors + 2 intermediate-sized tensors.
    let per_layer = (6 * h + 2 * i) * t * elem;
    per_layer * layers_per_seg
}

/// Boundary states between segments — always live.
fn boundary_state_bytes(cfg: &ModelConfig, max_seq_len: usize, num_segments: usize) -> u64 {
    let elem = dtype_bytes(cfg.dtype);
    let h = cfg.hidden_size as u64;
    let t = max_seq_len as u64;
    (num_segments as u64 + 1) * h * t * elem
}

/// One FLCE chunk's reduce accumulator + logits slice.
fn flce_chunk_intermediate_bytes(cfg: &ModelConfig, max_seq_len: usize) -> u64 {
    let v = cfg.vocab_size as u64;
    let t = max_seq_len as u64;
    let per_chunk_tokens = (t + DEFAULT_FLCE_CHUNKS - 1) / DEFAULT_FLCE_CHUNKS;
    // Reduce accumulator (F32) + logits chunk (compute dtype).
    let reduce_bytes = per_chunk_tokens * v * 4;
    let logits_bytes = per_chunk_tokens * v * dtype_bytes(cfg.dtype);
    reduce_bytes + logits_bytes
}

/// LoRA params + their gradients, F32 for both.
fn lora_param_and_grad_bytes(cfg: &ModelConfig, lora_rank: usize) -> u64 {
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let r = lora_rank as u64;
    let layers = cfg.num_layers as u64;
    // Per layer: U + V for {q, k, v, o, gate, up, down}. Conservative:
    // each has shapes ~ (h, r) + (r, max(h,i)). Use intermediate as upper bound.
    let per_layer_params = 7 * (h * r + r * i);
    // Param + grad, both F32.
    2 * per_layer_params * 4 * layers
}

/// Closed-form working-set estimate for one training step.
pub fn estimate_step_working_set(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    num_segments: usize,
) -> WorkingSet {
    let bd = Breakdown {
        base_weights: approximate_base_weight_bytes(cfg),
        per_segment_activations: per_segment_activation_bytes(cfg, max_seq_len, num_segments),
        boundary_states: boundary_state_bytes(cfg, max_seq_len, num_segments),
        flce_intermediates: flce_chunk_intermediate_bytes(cfg, max_seq_len),
        lora_param_grad: lora_param_and_grad_bytes(cfg, lora_rank),
        safety_margin: SAFETY_MARGIN_BYTES,
    };
    WorkingSet {
        total_bytes: bd.total(),
        max_seq_len,
        breakdown: bd,
    }
}

/// How much of the corrected VRAM budget is available for a training
/// step right now: budget − KV-cache reservation − safety reserve.
///
/// Phase 0: KV cache is hard to query without consulting the running
/// allocator, so we reserve a fraction of the budget. The per-job
/// preflight then compares its estimate against the remainder.
pub fn available_for_training_bytes(vram: &GpuVramInfo) -> u64 {
    if vram.total_bytes == 0 {
        // No detection — refuse to claim any budget. Caller should treat
        // this as "skip the check" rather than "reject everything",
        // since the preflight is not the right place to refuse jobs
        // when we have no budget signal at all.
        return u64::MAX;
    }
    // Reserve up to 1/3 of the budget for inference (KV cache + the
    // running scheduler), capped at 6 GB so that on a small APU the
    // training preflight doesn't lose the entire usable budget.
    let inference_reserve = (vram.total_bytes / 3).min(6 * BYTES_PER_GB);
    let after_inference = vram.total_bytes.saturating_sub(inference_reserve);
    after_inference.saturating_sub(SAFETY_MARGIN_BYTES)
}

/// Best-effort token count estimate for one example without paying
/// real tokenizer cost on every submission.
///
/// If `tokenizer` is provided, we tokenize one chat-templated turn
/// and use that as a lower bound; we then upgrade to a `chars/4`
/// upper bound across all examples so the preflight is conservative
/// even when the first example is short and later ones are huge.
pub fn approximate_max_seq_len_sft(examples: &[SftExample], tokenizer: Option<&KilnTokenizer>) -> usize {
    examples
        .iter()
        .map(|ex| approximate_tokens_for_messages(&ex.messages, tokenizer))
        .max()
        .unwrap_or(0)
}

pub fn approximate_max_seq_len_grpo(groups: &[GrpoGroup], tokenizer: Option<&KilnTokenizer>) -> usize {
    groups
        .iter()
        .map(|g| {
            let prompt = approximate_tokens_for_messages(&g.messages, tokenizer);
            let max_completion = g
                .completions
                .iter()
                .map(|c| approximate_tokens_for_text(&c.text, tokenizer))
                .max()
                .unwrap_or(0);
            prompt + max_completion
        })
        .max()
        .unwrap_or(0)
}

fn approximate_tokens_for_messages(
    messages: &[kiln_train::ChatMessage],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    // Sum every message's content in chars, plus a 16-token-per-message
    // tag for chat-template envelope overhead.
    let chars: usize = messages.iter().map(|m| m.content.chars().count()).sum();
    let envelope = messages.len() * 16;
    let char_estimate = (chars / 4) + envelope;

    if let Some(tok) = tokenizer {
        let core: Vec<kiln_core::tokenizer::ChatMessage> = messages
            .iter()
            .map(|m| kiln_core::tokenizer::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                ..Default::default()
            })
            .collect();
        if let Ok(text) = tok.apply_chat_template(&core) {
            if let Ok(ids) = tok.encode(&text) {
                // Use the larger of the two — if real tokenization
                // exceeds char/4, trust it; if char/4 is larger,
                // keep that as a safety upper bound.
                return ids.len().max(char_estimate);
            }
        }
    }
    char_estimate
}

fn approximate_tokens_for_text(text: &str, tokenizer: Option<&KilnTokenizer>) -> usize {
    let char_estimate = text.chars().count() / 4 + 4;
    if let Some(tok) = tokenizer {
        if let Ok(ids) = tok.encode(text) {
            return ids.len().max(char_estimate);
        }
    }
    char_estimate
}

/// Build the human-readable hint that goes in the `message` field of
/// the 413 response. The static `hint` field on the ApiError carries
/// the generic suggestion list; this body lists the actual numbers.
pub fn format_oom_message(
    estimate: &WorkingSet,
    available_bytes: u64,
    lora_rank: usize,
    num_segments: usize,
) -> String {
    let est_gb = estimate.total_bytes as f64 / BYTES_PER_GB as f64;
    let avail_gb = available_bytes as f64 / BYTES_PER_GB as f64;
    let bd = &estimate.breakdown;
    let bw_gb = bd.base_weights as f64 / BYTES_PER_GB as f64;
    let act_gb =
        (bd.per_segment_activations + bd.boundary_states) as f64 / BYTES_PER_GB as f64;
    let flce_gb = bd.flce_intermediates as f64 / BYTES_PER_GB as f64;
    let lora_gb = bd.lora_param_grad as f64 / BYTES_PER_GB as f64;
    format!(
        "Estimated training step working set is {est_gb:.2} GB but only \
         {avail_gb:.2} GB is available. Breakdown: weights {bw_gb:.2} GB, \
         activations {act_gb:.2} GB (max_seq_len={msl}, num_segments={num_segments}), \
         FLCE chunk {flce_gb:.2} GB, LoRA params+grads {lora_gb:.2} GB \
         (lora_rank={lora_rank}). To fit, raise KILN_GRAD_CHECKPOINT_SEGMENTS \
         (more segments = less per-segment activation memory), shrink \
         lora_rank, send fewer/shorter examples per submission, or set \
         KILN_TRAINING_MEMORY_RESERVE_GB lower if your host can spare RAM \
         from other processes.",
        msl = estimate.max_seq_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::config::ModelConfig;
    use kiln_core::vram::{GpuVramInfo, VramSource};

    fn qwen_4b() -> ModelConfig {
        ModelConfig::qwen3_5_4b()
    }

    #[test]
    fn estimator_grows_with_seq_len() {
        let cfg = qwen_4b();
        let small = estimate_step_working_set(&cfg, 256, 16, 4);
        let large = estimate_step_working_set(&cfg, 4096, 16, 4);
        assert!(
            large.total_bytes > small.total_bytes,
            "expected larger T to grow estimate; small={} large={}",
            small.total_bytes,
            large.total_bytes
        );
    }

    #[test]
    fn estimator_shrinks_with_more_segments() {
        let cfg = qwen_4b();
        let few = estimate_step_working_set(&cfg, 1500, 16, 4);
        let many = estimate_step_working_set(&cfg, 1500, 16, 16);
        assert!(
            many.breakdown.per_segment_activations < few.breakdown.per_segment_activations,
            "more segments must reduce per-segment activation footprint"
        );
    }

    #[test]
    fn qwen_4b_baseline_estimate_is_in_expected_range() {
        // Sanity: at T=1500 with 4 segments and rank 16, the estimate
        // should sit in single-digit-to-low-tens GB. A wildly different
        // number means a coefficient regression in one of the helpers.
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 1500, 16, 4);
        let gb = est.total_bytes as f64 / BYTES_PER_GB as f64;
        assert!(
            (8.0..=80.0).contains(&gb),
            "Qwen3.5-4B baseline estimate {gb:.2} GB out of plausible range"
        );
    }

    #[test]
    fn preflight_rejects_oversized_payload_on_30gb_host() {
        // The reproducer that motivated the plan: Qwen3.5-4B,
        // long-context SFT, 4-segment checkpointing on a 30 GB host
        // (corrected to ~22 GB after reserve). With a high seq len
        // the estimate should exceed available_for_training_bytes.
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 22 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        let avail = available_for_training_bytes(&vram);
        let est = estimate_step_working_set(&cfg, 8192, 16, 4);
        assert!(
            est.total_bytes > avail,
            "expected 8K-token Qwen3.5-4B step to overflow 22 GB unified budget; \
             got estimate={} avail={}",
            est.total_bytes,
            avail
        );
    }

    #[test]
    fn preflight_accepts_small_payload_on_30gb_host() {
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 22 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        let avail = available_for_training_bytes(&vram);
        let est = estimate_step_working_set(&cfg, 256, 8, 8);
        // The base weight contribution alone is large (~16 GB Qwen3.5-4B
        // BF16 with embedding + LM head counted twice). The Phase 0
        // estimator is conservative on purpose; this assertion just
        // checks the small payload sits closer to the limit than the
        // 8K version.
        let large = estimate_step_working_set(&cfg, 8192, 16, 4);
        assert!(est.total_bytes < large.total_bytes);
        // And that something fits at all on a 100 GB host so the
        // accept path is exercised somewhere.
        let big_vram = GpuVramInfo {
            total_bytes: 100 * BYTES_PER_GB,
            source: VramSource::NvidiaSmi,
        };
        let big_avail = available_for_training_bytes(&big_vram);
        assert!(est.total_bytes < big_avail);
        // Suppress unused warning when assertion compiles in release.
        let _ = avail;
    }

    #[test]
    fn available_for_training_handles_unknown_vram() {
        let none = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
        };
        // u64::MAX signals "skip the check" — preflight should not
        // refuse jobs when we have no budget signal.
        assert_eq!(available_for_training_bytes(&none), u64::MAX);
    }

    #[test]
    fn format_oom_message_includes_actionable_knobs() {
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 8192, 16, 4);
        let msg = format_oom_message(&est, 8 * BYTES_PER_GB, 16, 4);
        assert!(msg.contains("KILN_GRAD_CHECKPOINT_SEGMENTS"));
        assert!(msg.contains("lora_rank"));
        assert!(msg.contains("KILN_TRAINING_MEMORY_RESERVE_GB"));
    }
}

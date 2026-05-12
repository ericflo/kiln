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
use kiln_core::vram::{GpuVramInfo, VramSource};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{GrpoGroup, SftExample};

/// What the trainer can rely on being deduplicated across CPU and GPU.
///
/// On Vulkan today (Phase 0) every base weight lives BOTH as a candle
/// CPU tensor AND as one or two `VulkanBuffer` mirrors on the device.
/// On a unified-memory APU those mirrors are backed by the same
/// physical RAM, so the working set must count weights as if they
/// were resident twice over. After Phase 1.2-1.4 lands the resident
/// registry, the candle storage is stubbed and weights live in
/// exactly one place — the preflight then drops the multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightResidency {
    /// CUDA / Metal / dGPU paths where candle owns the weights and the
    /// device backend doesn't keep parallel copies.
    SingleCopy,
    /// Vulkan without the resident registry: weights live in candle CPU
    /// storage AND in `VulkanBuffer` caches simultaneously.
    DualResidentCpuAndVulkan,
}

impl WeightResidency {
    /// What multiplier to apply to base weight bytes when computing the
    /// working-set estimate.
    fn weight_multiplier(self) -> u64 {
        match self {
            // 1x for the candle copy + ~1x for the cached VulkanBuffer.
            // Add a small headroom (0.25x) for the bf16-packed cache that
            // is computed alongside the f32 cache for many weights.
            Self::DualResidentCpuAndVulkan => 2,
            Self::SingleCopy => 1,
        }
    }

    /// Inferred from the corrected VRAM source: if detection labelled
    /// the device as a unified-memory APU, weights are dual-resident
    /// today.
    pub fn for_vram_source(source: VramSource) -> Self {
        match source {
            VramSource::LinuxDrmSysfsUnified | VramSource::AppleSilicon => {
                Self::DualResidentCpuAndVulkan
            }
            // Discrete: candle keeps weights in CPU RAM but the GPU's
            // separate VRAM pool is its own memory; only the CPU copy
            // counts against the same budget the trainer estimates
            // against on the host. SingleCopy is honest there.
            _ => Self::SingleCopy,
        }
    }
}

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

#[derive(Debug, Clone, Copy, Default)]
pub struct EstimateOptions {
    pub max_supervised_tokens: Option<usize>,
    pub recompute_boundaries: bool,
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
fn boundary_state_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    num_segments: usize,
    recompute_boundaries: bool,
) -> u64 {
    if recompute_boundaries {
        let h = cfg.hidden_size as u64;
        let t = max_seq_len as u64;
        // Long-context SFT recomputes segment inputs on demand. At peak it
        // keeps the upstream hidden gradient (F32) plus one detached segment
        // input (model dtype), with a one-extra-buffer cushion for allocator
        // overlap during recompute.
        return 3 * h * t * 4;
    }
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
///
/// `residency` controls how many copies of the base weights to count.
/// Until the Phase 1 resident registry is deployed, callers on Vulkan
/// must pass `WeightResidency::DualResidentCpuAndVulkan` so the host
/// RAM pressure from both the candle CPU mirror and the device-side
/// `VulkanBuffer` caches is reflected. After Phase 1 lands and the
/// candle storage is stubbed, callers switch to `SingleCopy`.
///
/// `weights_already_resident` should be `true` when the available
/// budget already accounts for the loaded model (e.g. `MemAvailable`
/// at submission time, with the model already in candle/Vulkan
/// caches). In that case the base-weight contribution is excluded
/// from the working-set estimate to avoid double-counting them
/// against a budget that's already deducted them. For static budgets
/// (e.g. discrete-GPU VRAM total minus a fraction reserve) the
/// weights are still pending in the budget, so pass `false`.
pub fn estimate_step_working_set(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    num_segments: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
) -> WorkingSet {
    estimate_step_working_set_with_options(
        cfg,
        max_seq_len,
        lora_rank,
        num_segments,
        residency,
        weights_already_resident,
        EstimateOptions::default(),
    )
}

pub fn estimate_step_working_set_with_options(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    num_segments: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
    options: EstimateOptions,
) -> WorkingSet {
    let base_weights = if weights_already_resident {
        0
    } else {
        approximate_base_weight_bytes(cfg).saturating_mul(residency.weight_multiplier())
    };
    let flce_tokens = options.max_supervised_tokens.unwrap_or(max_seq_len).max(1);
    let bd = Breakdown {
        base_weights,
        per_segment_activations: per_segment_activation_bytes(cfg, max_seq_len, num_segments),
        boundary_states: boundary_state_bytes(
            cfg,
            max_seq_len,
            num_segments,
            options.recompute_boundaries,
        ),
        flce_intermediates: flce_chunk_intermediate_bytes(cfg, flce_tokens),
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
/// step right now.
///
/// On unified-memory APUs this consults `/proc/meminfo` MemAvailable
/// at submission time so the preflight reflects what's actually free
/// — the static VRAM number is the absolute ceiling but inference
/// has typically already eaten KV cache + Vulkan weight caches +
/// candle CPU storage from that pool by the time training is
/// submitted.
///
/// Discrete GPUs and the no-detection path keep the static behavior:
/// reserve a fraction of the budget for inference, return the rest.
pub fn available_for_training_bytes(vram: &GpuVramInfo) -> u64 {
    available_for_training_bytes_with_meminfo(vram, query_linux_mem_available_bytes())
}

#[doc(hidden)]
pub fn available_for_training_bytes_with_meminfo(
    vram: &GpuVramInfo,
    mem_available_bytes: Option<u64>,
) -> u64 {
    if vram.total_bytes == 0 {
        // No detection — refuse to claim any budget. Caller should treat
        // this as "skip the check" rather than "reject everything",
        // since the preflight is not the right place to refuse jobs
        // when we have no budget signal at all.
        return u64::MAX;
    }

    // Unified memory: training and inference share the same physical
    // pool, so MemAvailable_now is the truth. Cap by the corrected
    // VRAM budget so a misconfigured host can't somehow report more
    // available than the GPU can address.
    if matches!(
        vram.source,
        VramSource::LinuxDrmSysfsUnified | VramSource::AppleSilicon
    ) {
        if let Some(mem_avail) = mem_available_bytes {
            let live = mem_avail.saturating_sub(SAFETY_MARGIN_BYTES);
            return live.min(vram.total_bytes.saturating_sub(SAFETY_MARGIN_BYTES));
        }
        // No /proc/meminfo (non-Linux Apple Silicon, or read failed):
        // fall through to the conservative static path.
    }

    // Discrete-GPU / unknown path: reserve a fraction of the budget
    // for inference (KV cache + the running scheduler), capped at
    // 6 GB so that on a small device the training preflight doesn't
    // lose the entire usable budget.
    let inference_reserve = (vram.total_bytes / 3).min(6 * BYTES_PER_GB);
    let after_inference = vram.total_bytes.saturating_sub(inference_reserve);
    after_inference.saturating_sub(SAFETY_MARGIN_BYTES)
}

#[cfg(target_os = "linux")]
fn query_linux_mem_available_bytes() -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())?;
            return Some(kib * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn query_linux_mem_available_bytes() -> Option<u64> {
    None
}

/// Best-effort token count estimate for one example without paying
/// real tokenizer cost on every submission.
///
/// If `tokenizer` is provided, use the real chat-template token count for
/// each example. If tokenization is unavailable, fall back to a chars/4
/// estimate plus template-envelope overhead.
pub fn approximate_max_seq_len_sft(examples: &[SftExample], tokenizer: Option<&KilnTokenizer>) -> usize {
    examples
        .iter()
        .map(|ex| approximate_tokens_for_messages(&ex.messages, tokenizer))
        .max()
        .unwrap_or(0)
}

pub fn approximate_max_supervised_tokens_sft(
    examples: &[SftExample],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    examples
        .iter()
        .map(|ex| {
            ex.messages
                .iter()
                .filter(|message| message.role == "assistant")
                .map(|message| approximate_tokens_for_text(&message.content, tokenizer))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0)
}

pub fn recompute_checkpoint_boundaries_for_seq_len(max_seq_len: usize) -> bool {
    if let Some(forced) = kiln_core::env_flag::env_tristate("KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES")
    {
        return forced;
    }
    let threshold = std::env::var("KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8192);
    max_seq_len >= threshold
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
                return ids.len();
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
    format_oom_message_with_source(estimate, available_bytes, lora_rank, num_segments, None)
}

/// Variant that includes the VRAM detection source in the message
/// when supplied. On unified-memory APUs the operator benefits from
/// knowing the available number is `MemTotal − reserve` rather than
/// the raw GPU report — that's what `KILN_TRAINING_MEMORY_RESERVE_GB`
/// actually controls, and seeing it in the rejection message makes
/// the actionable knob obvious.
pub fn format_oom_message_with_source(
    estimate: &WorkingSet,
    available_bytes: u64,
    lora_rank: usize,
    num_segments: usize,
    vram_source: Option<kiln_core::vram::VramSource>,
) -> String {
    let est_gb = estimate.total_bytes as f64 / BYTES_PER_GB as f64;
    let avail_gb = available_bytes as f64 / BYTES_PER_GB as f64;
    let bd = &estimate.breakdown;
    let bw_gb = bd.base_weights as f64 / BYTES_PER_GB as f64;
    let act_gb =
        (bd.per_segment_activations + bd.boundary_states) as f64 / BYTES_PER_GB as f64;
    let flce_gb = bd.flce_intermediates as f64 / BYTES_PER_GB as f64;
    let lora_gb = bd.lora_param_grad as f64 / BYTES_PER_GB as f64;
    let source_clause = match vram_source {
        Some(src) => format!(" (vram_source={src})"),
        None => String::new(),
    };
    format!(
        "Estimated training step working set is {est_gb:.2} GB but only \
         {avail_gb:.2} GB is available{source_clause}. Breakdown: weights {bw_gb:.2} GB, \
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
        let small = estimate_step_working_set(&cfg, 256, 16, 4, WeightResidency::SingleCopy, false);
        let large = estimate_step_working_set(&cfg, 4096, 16, 4, WeightResidency::SingleCopy, false);
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
        let few = estimate_step_working_set(&cfg, 1500, 16, 4, WeightResidency::SingleCopy, false);
        let many = estimate_step_working_set(&cfg, 1500, 16, 16, WeightResidency::SingleCopy, false);
        assert!(
            many.breakdown.per_segment_activations < few.breakdown.per_segment_activations,
            "more segments must reduce per-segment activation footprint"
        );
    }

    #[test]
    fn estimator_uses_supervised_tokens_for_flce() {
        let cfg = qwen_4b();
        let full_prompt = estimate_step_working_set(&cfg, 8192, 16, 8, WeightResidency::SingleCopy, false);
        let sparse_labels = estimate_step_working_set_with_options(
            &cfg,
            8192,
            16,
            8,
            WeightResidency::SingleCopy,
            false,
            EstimateOptions {
                max_supervised_tokens: Some(512),
                recompute_boundaries: false,
            },
        );
        assert!(
            sparse_labels.breakdown.flce_intermediates < full_prompt.breakdown.flce_intermediates,
            "FLCE estimate should scale with supervised tokens"
        );
    }

    #[test]
    fn estimator_recompute_boundaries_does_not_scale_with_segment_count() {
        let cfg = qwen_4b();
        let cached = estimate_step_working_set(&cfg, 8192, 16, 32, WeightResidency::SingleCopy, false);
        let recompute = estimate_step_working_set_with_options(
            &cfg,
            8192,
            16,
            32,
            WeightResidency::SingleCopy,
            false,
            EstimateOptions {
                max_supervised_tokens: None,
                recompute_boundaries: true,
            },
        );
        assert!(
            recompute.breakdown.boundary_states < cached.breakdown.boundary_states,
            "recomputed-boundary estimate should not charge all segment boundaries"
        );
    }

    #[test]
    fn qwen_4b_baseline_estimate_is_in_expected_range() {
        // Sanity: at T=1500 with 4 segments and rank 16, the estimate
        // should sit in single-digit-to-low-tens GB. A wildly different
        // number means a coefficient regression in one of the helpers.
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 1500, 16, 4, WeightResidency::SingleCopy, false);
        let gb = est.total_bytes as f64 / BYTES_PER_GB as f64;
        assert!(
            (8.0..=80.0).contains(&gb),
            "Qwen3.5-4B baseline estimate {gb:.2} GB out of plausible range"
        );
    }

    #[test]
    fn dual_residency_doubles_base_weight_contribution() {
        let cfg = qwen_4b();
        let single = estimate_step_working_set(&cfg, 1500, 16, 4, WeightResidency::SingleCopy, false);
        let dual = estimate_step_working_set(
            &cfg,
            1500,
            16,
            4,
            WeightResidency::DualResidentCpuAndVulkan,
            false,
        );
        assert_eq!(dual.breakdown.base_weights, single.breakdown.base_weights * 2);
        assert!(dual.total_bytes > single.total_bytes);
    }

    #[test]
    fn weight_residency_is_dual_for_unified_memory() {
        assert_eq!(
            WeightResidency::for_vram_source(VramSource::LinuxDrmSysfsUnified),
            WeightResidency::DualResidentCpuAndVulkan
        );
        assert_eq!(
            WeightResidency::for_vram_source(VramSource::AppleSilicon),
            WeightResidency::DualResidentCpuAndVulkan
        );
        assert_eq!(
            WeightResidency::for_vram_source(VramSource::NvidiaSmi),
            WeightResidency::SingleCopy
        );
        assert_eq!(
            WeightResidency::for_vram_source(VramSource::LinuxDrmSysfs),
            WeightResidency::SingleCopy
        );
    }

    /// Regression test for the 2026-05-10 host crash. The crash
    /// happened on the pre-Phase-0 codebase: weights were dual-
    /// resident (no embed_tokens stub, no projection drop), the
    /// preflight didn't exist, and MemAvailable at submission time
    /// (after model+KV-cache load on the old code) was only ~5 GB.
    /// Reconstructing those exact conditions, the preflight must
    /// reject — proving the same crash today wouldn't reach the
    /// kernel OOM path.
    #[test]
    fn preflight_rejects_repro_payload_on_strix_halo() {
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 24_944_216_064,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        // Pre-Phase-1.2-1 codebase had ~5 GB MemAvailable at
        // submission time (model loaded twice in CPU+Vulkan, KV
        // cache resident).
        let mem_available = 5 * BYTES_PER_GB;
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(mem_available));
        // weights_already_resident=false models the old codebase's
        // budget accounting — the dual-resident weights are still
        // counted in the estimate against a static budget, mirroring
        // the residency model that crashed.
        let est = estimate_step_working_set(
            &cfg,
            2500,
            16,
            8,
            WeightResidency::DualResidentCpuAndVulkan,
            false,
        );
        assert!(
            est.total_bytes > avail,
            "preflight must reject the 2026-05-10 repro payload; \
             got estimate={} ({:.2} GB), avail={} ({:.2} GB)",
            est.total_bytes,
            est.total_bytes as f64 / BYTES_PER_GB as f64,
            avail,
            avail as f64 / BYTES_PER_GB as f64,
        );
    }

    #[test]
    fn preflight_rejects_oversized_payload_on_30gb_host() {
        // Long-context payload on the same 30 GB unified host: must
        // overflow even more dramatically than the repro.
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 22 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(8 * BYTES_PER_GB));
        let est = estimate_step_working_set(
            &cfg,
            8192,
            16,
            4,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
        );
        assert!(
            est.total_bytes > avail,
            "expected 8K-token Qwen3.5-4B step to overflow 22 GB unified budget; \
             got estimate={} avail={}",
            est.total_bytes,
            avail
        );
    }

    /// Companion to the repro-rejection test: a SMALL payload (1
    /// example, ~36 tokens) on the same Strix Halo unified budget
    /// must be ACCEPTED. The earlier hardening over-corrected by
    /// double-counting base weights against MemAvailable; the
    /// `weights_already_resident` flag fixes that.
    #[test]
    fn preflight_accepts_tiny_payload_on_strix_halo_post_load() {
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 24_944_216_064,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        // Post-load MemAvailable on the actual hardware was ~17 GB
        // (the model + Vulkan caches were already loaded).
        let mem_available = 17 * BYTES_PER_GB;
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(mem_available));
        let est = estimate_step_working_set(
            &cfg,
            36, // 1 short example
            4,  // tiny rank
            8,
            WeightResidency::DualResidentCpuAndVulkan,
            true, // weights already resident
        );
        assert!(
            est.total_bytes < avail,
            "small payload on Strix Halo post-load must fit; \
             got estimate={} ({:.2} GB), avail={} ({:.2} GB)",
            est.total_bytes,
            est.total_bytes as f64 / BYTES_PER_GB as f64,
            avail,
            avail as f64 / BYTES_PER_GB as f64,
        );
    }

    #[test]
    fn preflight_still_accepts_on_a_big_dgpu() {
        // Discrete A6000-class card (48 GB VRAM) with a small payload:
        // SingleCopy residency, generous available, must accept.
        let cfg = qwen_4b();
        let big_vram = GpuVramInfo {
            total_bytes: 48 * BYTES_PER_GB,
            source: VramSource::NvidiaSmi,
        };
        let big_avail = available_for_training_bytes_with_meminfo(&big_vram, None);
        let est = estimate_step_working_set(&cfg, 256, 8, 8, WeightResidency::SingleCopy, false);
        assert!(
            est.total_bytes < big_avail,
            "small payload on 48 GB dGPU must fit; estimate={} avail={}",
            est.total_bytes,
            big_avail
        );
    }

    #[test]
    fn preflight_rejects_long_context_on_a6000_cuda_budget() {
        // CUDA/discrete GPUs use SingleCopy residency, but the same
        // fit-before-run gate must still reject payloads whose activations
        // and FLCE chunk working set exceed the post-reserve A6000 budget.
        let cfg = qwen_4b();
        let a6000_vram = GpuVramInfo {
            total_bytes: 48 * BYTES_PER_GB,
            source: VramSource::NvidiaSmi,
        };
        let available = available_for_training_bytes_with_meminfo(&a6000_vram, None);
        let est = estimate_step_working_set(
            &cfg,
            65_536,
            16,
            4,
            WeightResidency::SingleCopy,
            false,
        );
        assert!(
            est.total_bytes > available,
            "64k-token Qwen3.5-4B SFT must be rejected on an A6000 CUDA budget; \
             estimate={} ({:.2} GB), available={} ({:.2} GB)",
            est.total_bytes,
            est.total_bytes as f64 / BYTES_PER_GB as f64,
            available,
            available as f64 / BYTES_PER_GB as f64,
        );
    }

    #[test]
    fn available_for_training_handles_unknown_vram() {
        let none = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
        };
        assert_eq!(available_for_training_bytes(&none), u64::MAX);
    }

    #[test]
    fn unified_memory_uses_meminfo_when_present() {
        let vram = GpuVramInfo {
            total_bytes: 25 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        // 8 GB free at submission time, capped by VRAM ceiling.
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(8 * BYTES_PER_GB));
        let expected = 8u64 * BYTES_PER_GB - SAFETY_MARGIN_BYTES;
        assert_eq!(avail, expected);
    }

    #[test]
    fn unified_memory_caps_meminfo_at_vram_ceiling() {
        let vram = GpuVramInfo {
            total_bytes: 10 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        // 50 GB MemAvailable on a 10 GB ceiling — cap to ceiling.
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(50 * BYTES_PER_GB));
        assert_eq!(avail, 10u64 * BYTES_PER_GB - SAFETY_MARGIN_BYTES);
    }

    #[test]
    fn format_oom_message_includes_actionable_knobs() {
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 8192, 16, 4, WeightResidency::SingleCopy, false);
        let msg = format_oom_message(&est, 8 * BYTES_PER_GB, 16, 4);
        assert!(msg.contains("KILN_GRAD_CHECKPOINT_SEGMENTS"));
        assert!(msg.contains("lora_rank"));
        assert!(msg.contains("KILN_TRAINING_MEMORY_RESERVE_GB"));
    }

    #[test]
    fn format_oom_message_with_source_surfaces_unified_memory_signal() {
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 8192, 16, 4, WeightResidency::SingleCopy, false);
        // On a unified-memory APU the rejection message must call out
        // the corrected source so the operator knows the
        // KILN_TRAINING_MEMORY_RESERVE_GB knob is the relevant one.
        let msg = format_oom_message_with_source(
            &est,
            8 * BYTES_PER_GB,
            16,
            4,
            Some(VramSource::LinuxDrmSysfsUnified),
        );
        assert!(
            msg.contains("vram_source=linux-drm-sysfs-unified"),
            "expected unified-memory provenance, got: {msg}"
        );
        assert!(msg.contains("KILN_TRAINING_MEMORY_RESERVE_GB"));

        // None preserves the legacy message — no provenance clause.
        let no_src = format_oom_message_with_source(&est, 8 * BYTES_PER_GB, 16, 4, None);
        assert!(!no_src.contains("vram_source"));
    }
}

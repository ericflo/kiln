use super::*;

/// Gradient checkpointing configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointConfig {
    /// Number of segments to split layers into.
    pub num_segments: usize,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
    /// Whether num_segments was auto-configured from VRAM detection.
    pub auto_configured: bool,
}

impl CheckpointConfig {
    pub fn from_resolved_segments(num_layers: usize, num_segments: usize) -> Self {
        let num_segments = num_segments.min(num_layers).max(1);
        Self {
            num_segments,
            enabled: num_segments > 1,
            auto_configured: true,
        }
    }

    /// Standalone constructor with VRAM-aware automatic defaults.
    ///
    /// This is the *VRAM-only* path. Callers that know the workload's
    /// `max_seq_len` should prefer [`CheckpointConfig::auto_for_workload`],
    /// which can additionally choose to *disable* checkpointing when the
    /// activation tape comfortably fits in available VRAM (typical on big
    /// GPUs with short prompts).
    pub fn standalone(num_layers: usize) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::from_runtime(num_layers, &runtime)
    }

    /// Standalone compatibility with an already-resolved capacity.
    pub fn standalone_with_vram(num_layers: usize, vram: &kiln_memory::vram::GpuVramInfo) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::from_runtime(num_layers, &runtime)
    }

    /// Deprecated compatibility name. This function does not read environment.
    #[deprecated(note = "use CheckpointConfig::standalone or CheckpointConfig::from_runtime")]
    pub fn from_env(num_layers: usize) -> Self {
        Self::standalone(num_layers)
    }

    /// Deprecated compatibility name. This function does not read environment.
    #[deprecated(note = "use CheckpointConfig::standalone_with_vram or from_runtime")]
    pub fn from_env_with_vram(num_layers: usize, vram: &kiln_memory::vram::GpuVramInfo) -> Self {
        Self::standalone_with_vram(num_layers, vram)
    }

    /// Resolve a VRAM-only checkpoint configuration from immutable inputs.
    pub fn from_runtime(num_layers: usize, runtime: &crate::TrainingRuntimeContext) -> Self {
        use crate::GradientCheckpointPolicy;

        let vram = runtime.effective_vram();
        match runtime.gradient_checkpoint_policy() {
            GradientCheckpointPolicy::ExplicitSegments { segments } => {
                let mut config = Self::from_resolved_segments(num_layers, segments.get());
                config.auto_configured = false;
                return config;
            }
            GradientCheckpointPolicy::Disabled {
                segments: Some(segments),
            } => {
                return Self {
                    num_segments: segments.get().min(num_layers).max(1),
                    enabled: false,
                    auto_configured: false,
                };
            }
            GradientCheckpointPolicy::Auto
            | GradientCheckpointPolicy::Disabled { segments: None } => {}
        }

        // VRAM-aware auto-configuration
        let num_segments = kiln_memory::vram::recommended_checkpoint_segments(vram)
            .unwrap_or(4) // conservative fallback when capacity is unknown
            .min(num_layers)
            .max(1);

        let auto_configured = vram.source != kiln_memory::vram::VramSource::None;

        if auto_configured {
            tracing::info!(
                num_segments,
                vram_gb = vram.total_bytes as f64 / 1e9,
                source = %vram.source,
                "auto-configured gradient checkpoint segments for detected VRAM"
            );
        }

        Self {
            num_segments,
            enabled: !runtime.gradient_checkpoint_policy().is_disabled(),
            auto_configured,
        }
    }

    /// Create config with **VRAM + workload-shape** auto-tuning. Preferred over
    /// [`CheckpointConfig::standalone`] for trainer call sites that have the
    /// `max_seq_len` available after tokenization.
    ///
    /// Standalone wrappers detect physical capacity once into a
    /// [`crate::TrainingRuntimeContext`]. Runtime-aware callers receive the
    /// server's resolved typed configuration.
    ///
    /// In auto mode this calls [`kiln_memory::vram::recommended_checkpoint_plan`]
    ///   which can *disable* checkpointing entirely when the activation tape
    ///   comfortably fits in available VRAM. On A6000 + Qwen3.5-4B, this
    ///   skips checkpointing for sequences up to ~12K tokens and only
    ///   engages it (with the right number of segments) for longer contexts.
    ///
    /// `bytes_per_base_param` is used to estimate base-model footprint —
    /// pass 2 for BF16 (canonical kiln inference dtype) or 4 for F32.
    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            4,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_vram(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        vram: &kiln_memory::vram::GpuVramInfo,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            4,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            activation_bytes_per_elem,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes_and_vram(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
        vram: &kiln_memory::vram::GpuVramInfo,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            activation_bytes_per_elem,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes_and_runtime(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
        runtime: &crate::TrainingRuntimeContext,
    ) -> Self {
        match runtime.gradient_checkpoint_policy() {
            crate::GradientCheckpointPolicy::Auto => {}
            crate::GradientCheckpointPolicy::ExplicitSegments { segments } => {
                let mut config = Self::from_resolved_segments(num_layers, segments.get());
                config.auto_configured = false;
                return config;
            }
            crate::GradientCheckpointPolicy::Disabled { segments } => {
                return Self {
                    num_segments: segments.map_or(1, |value| value.get().min(num_layers).max(1)),
                    enabled: false,
                    auto_configured: false,
                };
            }
        }

        let vram = runtime.effective_vram();

        let base_bytes = kiln_memory::vram::estimate_base_model_bytes(
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
        );

        match kiln_memory::vram::recommended_checkpoint_plan_with_activation_bytes(
            vram,
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            base_bytes,
            activation_bytes_per_elem,
        ) {
            None => Self::from_runtime(num_layers, runtime),
            Some(kiln_memory::vram::CheckpointPlan::Disabled {
                max_act_gib,
                available_gib,
            }) => {
                tracing::info!(
                    max_seq_len_tokens,
                    activation_bytes_per_elem,
                    activation_tape_gib = format!("{max_act_gib:.2}"),
                    available_gib = format!("{available_gib:.2}"),
                    vram_total_gb = vram.total_bytes as f64 / 1e9,
                    vram_source = %vram.source,
                    "auto-tuned: gradient checkpointing DISABLED — activation tape fits comfortably in available VRAM"
                );
                Self {
                    num_segments: 1,
                    enabled: false,
                    auto_configured: true,
                }
            }
            Some(kiln_memory::vram::CheckpointPlan::Enabled {
                num_segments,
                max_act_gib,
                per_segment_gib,
                available_gib,
            }) => {
                tracing::info!(
                    num_segments,
                    max_seq_len_tokens,
                    activation_bytes_per_elem,
                    activation_tape_gib = format!("{max_act_gib:.2}"),
                    per_segment_gib = format!("{per_segment_gib:.2}"),
                    available_gib = format!("{available_gib:.2}"),
                    vram_total_gb = vram.total_bytes as f64 / 1e9,
                    vram_source = %vram.source,
                    "auto-tuned: gradient checkpointing engaged for workload shape"
                );
                Self {
                    num_segments,
                    enabled: true,
                    auto_configured: true,
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn checkpoint_config_for_training_step(
    weights: &GpuWeights,
    device: &Device,
    preflight_resolved_segments: Option<usize>,
    num_layers: usize,
    seq_len_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    bytes_per_base_param: usize,
    activation_bytes_per_elem: usize,
    runtime: &crate::TrainingRuntimeContext,
) -> CheckpointConfig {
    match runtime.gradient_checkpoint_policy() {
        crate::GradientCheckpointPolicy::Auto => {}
        crate::GradientCheckpointPolicy::ExplicitSegments { segments } => {
            let mut config = CheckpointConfig::from_resolved_segments(
                num_layers,
                preflight_resolved_segments.unwrap_or(segments.get()),
            );
            config.auto_configured = false;
            return config;
        }
        crate::GradientCheckpointPolicy::Disabled { segments } => {
            return CheckpointConfig {
                num_segments: segments
                    .map(std::num::NonZeroUsize::get)
                    .or(preflight_resolved_segments)
                    .unwrap_or(1)
                    .min(num_layers)
                    .max(1),
                enabled: false,
                auto_configured: false,
            };
        }
    }

    if let Some(resolved_segments) = preflight_resolved_segments {
        // Server admission resolves against live memory after the model and KV
        // cache are resident. That exact plan is stricter than replanning from
        // the immutable startup capacity and must remain authoritative.
        return CheckpointConfig::from_resolved_segments(num_layers, resolved_segments);
    }

    let mut cfg = CheckpointConfig::auto_for_workload_with_activation_bytes_and_runtime(
        num_layers,
        seq_len_tokens,
        hidden_size,
        intermediate_size,
        vocab_size,
        bytes_per_base_param,
        activation_bytes_per_elem,
        runtime,
    );

    if let Some(num_segments) =
        long_context_full_attention_forced_checkpoint_segments(weights, device, seq_len_tokens)
        && (!cfg.enabled || cfg.num_segments < num_segments)
    {
        tracing::info!(
            seq_len_tokens,
            num_segments,
            "auto-tuned: gradient checkpointing engaged for long-context full-attention tape pressure"
        );
        cfg.enabled = num_segments > 1;
        cfg.num_segments = num_segments;
        cfg.auto_configured = true;
    }

    cfg
}

pub(super) fn long_context_full_attention_forced_checkpoint_segments(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
) -> Option<usize> {
    const MIN_TOKENS: usize = 8 * 1024;

    if seq_len_tokens < MIN_TOKENS {
        return None;
    }
    if !matches!(
        device,
        Device::Cuda(_) | Device::Rocm(_) | Device::Metal(_) | Device::Vulkan(_)
    ) {
        return None;
    }
    let full_attention_layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count();
    if full_attention_layers == 0 {
        return None;
    }

    Some(weights.layers.len().max(1))
}

pub(super) fn checkpoint_segments_for_config(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    ckpt_config: CheckpointConfig,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Option<Vec<(usize, usize)>> {
    if !ckpt_config.enabled {
        return None;
    }
    let mut boundaries = compute_segment_boundaries(weights.layers.len(), ckpt_config.num_segments);
    if ckpt_config.auto_configured
        && (materialized_full_attention_checkpoint_refinement_needed(
            weights,
            device,
            seq_len_tokens,
            streaming_prefill,
        ) || rocm_online_full_attention_checkpoint_refinement_needed(
            weights,
            device,
            seq_len_tokens,
            streaming_prefill,
        ))
    {
        let refined = refine_segments_for_materialized_full_attention(weights, &boundaries);
        if refined.len() > boundaries.len() {
            tracing::info!(
                seq_len = seq_len_tokens,
                original_segments = boundaries.len(),
                refined_segments = refined.len(),
                "refined gradient checkpoint boundaries for materialized full-attention replay"
            );
            boundaries = refined;
        }
    }
    Some(boundaries)
}

pub(super) fn rocm_online_full_attention_checkpoint_refinement_needed(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> bool {
    const MIN_TOKENS: usize = 8 * 1024;

    if seq_len_tokens < MIN_TOKENS || !streaming_prefill.enabled_for(seq_len_tokens) {
        return false;
    }
    if !matches!(device, Device::Rocm(_)) {
        return false;
    }
    weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count()
        > 1
}

pub(super) fn materialized_full_attention_checkpoint_refinement_needed(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> bool {
    if !streaming_prefill.enabled_for(seq_len_tokens) {
        return false;
    }
    if !matches!(device, Device::Metal(_) | Device::Vulkan(_)) {
        return false;
    }
    weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count()
        > 1
}

pub(super) fn refine_segments_for_materialized_full_attention(
    weights: &GpuWeights,
    boundaries: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut refined = Vec::with_capacity(boundaries.len());
    for &(start, end) in boundaries {
        if start >= end {
            continue;
        }
        let mut seg_start = start;
        let mut full_attn_in_segment = 0usize;
        for layer_idx in start..end {
            if matches!(
                weights.layers[layer_idx].attention,
                GpuAttentionWeights::Full(_)
            ) {
                if full_attn_in_segment > 0 {
                    refined.push((seg_start, layer_idx));
                    seg_start = layer_idx;
                    full_attn_in_segment = 0;
                }
                full_attn_in_segment += 1;
            }
        }
        if seg_start < end {
            refined.push((seg_start, end));
        }
    }
    refined
}

/// Compute segment boundaries for gradient checkpointing.
///
/// Returns a list of `(start_layer, end_layer)` pairs that partition
/// `[0..num_layers)` into `num_segments` roughly-equal segments.
pub(crate) fn compute_segment_boundaries(
    num_layers: usize,
    num_segments: usize,
) -> Vec<(usize, usize)> {
    let seg_size = num_layers / num_segments;
    let remainder = num_layers % num_segments;
    let mut boundaries = Vec::with_capacity(num_segments);
    let mut start = 0;
    for i in 0..num_segments {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + seg_size + extra;
        boundaries.push((start, end));
        start = end;
    }
    boundaries
}

/// Returns true when every transformer layer in `weights` uses linear (GDN)
/// attention — i.e., the model has **no** full-attention layers anywhere.
///
/// The training-time time-axis tile path
/// ([`tiled_segment_recompute_and_backward`]) thread `LinearAttentionState`
/// across tiles to keep GDN forward bit-exact, but full-attention layers have
/// no analogous KV-cache thread at training time (training does not allocate
/// a paged KV cache). Within a tile a full-attention layer would attend only
/// inside the tile and produce different logits, breaking both per-tile loss
/// and any LoRA gradient that flows through it.
///
/// Per-segment iteration also runs **later** segments detached on the tile's
/// output, so even a segment that is itself GDN-only would dispatch into
/// later full-attention layers under tiling — which would also break parity.
/// The cleanest correctness invariant is therefore "no full-attention layers
/// anywhere in the model".
#[allow(dead_code)]
pub(super) fn model_is_gdn_only(weights: &GpuWeights) -> bool {
    weights
        .layers
        .iter()
        .all(|l| matches!(l.attention, GpuAttentionWeights::Linear(_)))
}

/// Build a [`LoraWeights`] view whose `a` / `b` projections are **detached**
/// from the LoRA Vars' autograd graph.
///
/// Used by [`layer_pair_tiled_segment_recompute_and_backward`] for forwards
/// whose backward should NOT produce LoRA gradients — specifically, the
/// tail forward (whose only useful output is the gradient at the segment-
/// output Var) and the block-boundary forward in Step 2 (which only
/// computes activation VALUES). Without this, those backward passes would
/// produce LoRA gradients that would then be discarded — wasted compute,
/// and a correctness hazard if the discard is forgotten.
pub(crate) fn lora_weights_detached(params: &TrainableLoraParams) -> LoraWeights {
    let layers: Vec<LoraLayerWeights> = params
        .layers
        .iter()
        .map(|lp| {
            // (#1082) kt `Tensor::detach()` — the detached forward LoRA used by
            // the checkpointed Step-1 boundary forward (no grad recording).
            let make_proj =
                |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                    pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                        a: a.forward_storage().primary_tensor().detach(),
                        b: b.forward_storage().primary_tensor().detach(),
                    })
                };
            LoraLayerWeights {
                q_proj: make_proj(&lp.q_proj),
                k_proj: make_proj(&lp.k_proj),
                v_proj: make_proj(&lp.v_proj),
                o_proj: make_proj(&lp.o_proj),
                in_proj_qkv: make_proj(&lp.in_proj_qkv),
                in_proj_z: make_proj(&lp.in_proj_z),
                gdn_out_proj: make_proj(&lp.gdn_out_proj),
                gate_proj: make_proj(&lp.gate_proj),
                up_proj: make_proj(&lp.up_proj),
                down_proj: make_proj(&lp.down_proj),
                ..Default::default()
            }
        })
        .collect();

    LoraWeights {
        layers,
        mtp: None,
        rank: params.rank,
        alpha: params.alpha,
        scale: params.scale,
        source_identity: None,
    }
}

/// Attention kind of a single transformer layer for the layer-pair tiled path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttnKind {
    Gdn,
    FullAttn,
}

pub(super) fn attn_kind_at(weights: &GpuWeights, layer_idx: usize) -> AttnKind {
    match &weights.layers[layer_idx].attention {
        GpuAttentionWeights::Linear(_) => AttnKind::Gdn,
        GpuAttentionWeights::Full(_) => AttnKind::FullAttn,
    }
}

/// Partition `[seg_start, seg_end)` into maximal contiguous runs of the same
/// attention kind. Each entry is `(kind, layer_range)` where `layer_range` is
/// a sub-range of the segment with all layers of the same kind.
///
/// Used by the layer-pair tiled path to process GDN sub-blocks (time-tile)
/// and full-attention sub-blocks (monolithic) sequentially within one
/// segment-recompute pass.
pub(super) fn partition_segment_layers_by_attn_type(
    weights: &GpuWeights,
    seg_start: usize,
    seg_end: usize,
) -> Vec<(AttnKind, std::ops::Range<usize>)> {
    debug_assert!(seg_start < seg_end);
    let mut blocks: Vec<(AttnKind, std::ops::Range<usize>)> = Vec::new();
    let mut block_start = seg_start;
    let mut current_kind = attn_kind_at(weights, seg_start);
    for i in (seg_start + 1)..seg_end {
        let kind = attn_kind_at(weights, i);
        if kind != current_kind {
            blocks.push((current_kind, block_start..i));
            block_start = i;
            current_kind = kind;
        }
    }
    blocks.push((current_kind, block_start..seg_end));
    blocks
}

/// Determine whether a time-axis tile path applies for this training step.
///
/// Returns `Some(tile_size)` when:
/// 1. The injected streaming-prefill policy is enabled at this `seq_len`.
/// 2. The tile size is a positive multiple of `GDN_CHUNK_SIZE` (enforced by
///    typed startup validation) and strictly less than `seq_len`.
///
/// Caller routes between two implementations based on
/// [`model_is_gdn_only`]:
/// * GDN-only models use [`tiled_segment_recompute_and_backward`], which is
///   bit-exact against monolithic and skips gradient injection (cheaper).
/// * Hybrid GDN + full-attn models use
///   [`layer_pair_tiled_segment_recompute_and_backward`], which partitions
///   each segment into contiguous-attention-type blocks and processes them
///   with gradient injection so the tiled path can fire on production
///   models like Qwen3.5-4B (24 GDN + 8 full-attn).
#[allow(dead_code)]
pub(super) fn tiled_training_tile_size(
    weights: &GpuWeights,
    seq_len: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Option<usize> {
    let _ = weights; // signature retained for callers; gating moved to the dispatcher.
    if !streaming_prefill.enabled_for(seq_len) {
        return None;
    }
    let tile = streaming_prefill.base_tile_tokens();
    if tile == 0 || !tile.is_multiple_of(GDN_CHUNK_SIZE) || tile >= seq_len {
        return None;
    }
    Some(tile)
}

// (#1082) Deleted five orphaned residues of the removed exact_gdn tiled-reverse
// machinery — all had zero callers after the candle-drop:
//   * `profile_exact_gdn_reverse_tiles`
//   * `exact_gdn_split_recurrent_backward_enabled`
//   * `finish_exact_gdn_reverse_tile_stage` (its only call was to the already
//     deleted `synchronize_checkpoint_boundary`)
//   * `exact_gdn_reverse_tile_size`
//   * `exact_gdn_backward_tile_tokens_for`

// `InjectTensorGradient` (struct + impl candle_core::CustomOp1) was
// deleted as part of the #1082 CP-4 step 2-3 caller flip. All 6 call
// sites in `full_attention_single_layer_tiled_mlp_reverse` now use
// `kiln_kt_bridge::inject_grad_shim::inject_gradient_via_shim` which
// produces a bit-equivalent candle Tensor (the shim's `bwd` returns
// the precomputed `upstream`, byte-for-byte matching the previous
// in-trainer impl). With this deletion, `kiln-train::trainer` has
// zero production `candle_core::CustomOp1` impls and the crate's
// `candle-core` dep can move to `[dev-dependencies]`. See commits
// e2f8723c (substrate revision), 07afd64a (IO mapping removal),
// a6531830 (shim hoist), and the InjectTensorGradient flip
// commit itself. (#1082)

/// (#1082) Whether the kt tape grad-delivery path supports this base model's
/// dtype on this device. The decisive dtype is the **activation** dtype, which
/// follows the BASE model weights (`embed_tokens` dtype) — NOT the LoRA Vars,
/// which now FOLLOW the base dtype (see `initialize_seeded`).
///
/// BF16 is supported by the kt tape adapters. F32 is supported only when the
/// backend-owned precision policy declares F32 activations for mixed base
/// weights. Other dtypes fail before training work begins.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn base_dtype_supports_tape_for_policy(
    weights: &GpuWeights,
    policy: TrainingPrecisionPolicy,
) -> bool {
    // (#1082) `embed_tokens.dtype()` is now kt `DType`.
    match weights.embed_tokens.dtype() {
        kiln_tensor::DType::BF16 => true,
        kiln_tensor::DType::F32 => policy.uses_f32_activations_for_mixed_base_weights(),
        _ => false,
    }
}

pub(super) fn ensure_sft_loss_route_supports_checkpointing(
    route: SftFlceLossRoute,
    checkpointed: bool,
) -> Result<()> {
    anyhow::ensure!(
        !checkpointed || route != SftFlceLossRoute::FullLogits,
        "checkpointed SFT does not support loss route `{}`: its loss-value path \
         requires an active kt tape, while checkpoint tails run outside segment \
         tapes; disable gradient checkpointing or use a backend with a \
         checkpoint-compatible SFT loss route",
        route.as_str()
    );
    Ok(())
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn ensure_tape_forward_backward_supported(
    workload: &str,
    weights: &GpuWeights,
    backend: &dyn BackendRuntime,
) -> Result<()> {
    let route = TrainingLossBackend::runtime_tape_forward_backward_route(backend);
    anyhow::ensure!(
        matches!(route, TrainingTapeRoute::KtTapeAuthoritative),
        "{workload}: kt tape-authoritative training is required, but the backend \
         advertises tape route `{}`",
        route.as_str()
    );
    let precision_policy = training_precision_policy_for_backend(backend);
    anyhow::ensure!(
        base_dtype_supports_tape_for_policy(weights, precision_policy),
        "{workload}: base activation dtype {:?} is incompatible with backend \
         training precision policy `{}` for kt tape-authoritative training",
        weights.embed_tokens.dtype(),
        precision_policy.name
    );
    Ok(())
}

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
)))]
pub(super) fn ensure_tape_forward_backward_supported(
    workload: &str,
    _weights: &GpuWeights,
    _backend: &dyn BackendRuntime,
) -> Result<()> {
    anyhow::bail!(
        "{workload}: kt tape-authoritative training requires a CUDA, ROCm, Metal, \
         or Vulkan backend build"
    )
}

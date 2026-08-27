//! Pre-submission training preflight estimator.
//!
//! Submitted SFT/GRPO jobs are size-checked against the corrected memory
//! budget so the server fails with HTTP 413 + an actionable hint instead
//! of OOM-killing itself partway through the first step. The estimator is
//! intentionally a closed-form upper bound: it overestimates by design so
//! that "fits according to preflight" implies "actually fits at runtime".
//!
//! Used by `crate::api::training::submit_sft` and `submit_grpo`.

use kiln_core::config::{DType, ModelConfig};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_memory::vram::GpuVramInfo;
use kiln_model::backend::SftFlceLossRoute;
use kiln_train::{CheckpointBoundaryPolicy, GrpoGroup, Optimizer, SftExample};

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

/// Backend ownership policy for trainable LoRA tensors.
///
/// CUDA, ROCm, and Metal residency aliases tensor storage. Vulkan's resident
/// registry currently allocates and uploads a distinct device-local buffer, so
/// every registered parameter and optimizer state has two physical copies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LoraResidency {
    #[default]
    StorageOwned,
    RegistryMirrored,
}

impl LoraResidency {
    /// Resolve from the selected runtime backend, independent of memory
    /// topology. Unknown accelerator backends fail closed as registry-mirrored.
    pub fn for_backend_name(backend_name: &str) -> Self {
        match kiln_model::backend::residency::resident_ownership_for_backend(backend_name) {
            kiln_model::backend::residency::ResidentOwnership::StorageOwned => Self::StorageOwned,
            kiln_model::backend::residency::ResidentOwnership::RegistryOwned => {
                Self::RegistryMirrored
            }
        }
    }

    fn param_and_grad_f32_copies(self) -> u64 {
        match self {
            // Parameter storage + gradient storage.
            Self::StorageOwned => 2,
            // Parameter storage + registry mirror + gradient storage.
            Self::RegistryMirrored => 3,
        }
    }

    fn optimizer_state_f32_copies(self) -> u64 {
        match self {
            Self::StorageOwned => 1,
            // State storage + registry mirror.
            Self::RegistryMirrored => 2,
        }
    }
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

    /// Unified-memory systems keep CPU and accelerator copies in the same
    /// physical pool regardless of how the effective capacity was configured.
    pub fn for_vram(vram: &GpuVramInfo) -> Self {
        if vram.unified {
            Self::DualResidentCpuAndVulkan
        } else {
            // Discrete: candle keeps weights in CPU RAM but the GPU's
            // separate VRAM pool is its own memory; only the CPU copy
            // counts against the same budget the trainer estimates
            // against on the host. SingleCopy is honest there.
            Self::SingleCopy
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
    pub loss_workspace: u64,
    pub lora_param_grad: u64,
    pub lora_optimizer_state: u64,
    pub lora_registry_scratch: u64,
    pub safety_margin: u64,
}

impl Breakdown {
    pub fn total(&self) -> u64 {
        self.base_weights
            .saturating_add(self.per_segment_activations)
            .saturating_add(self.boundary_states)
            .saturating_add(self.loss_workspace)
            .saturating_add(self.lora_param_grad)
            .saturating_add(self.lora_optimizer_state)
            .saturating_add(self.lora_registry_scratch)
            .saturating_add(self.safety_margin)
    }

    /// The part of the estimate that does not scale with LoRA rank.
    pub fn fixed_bytes(&self) -> u64 {
        self.base_weights
            .saturating_add(self.per_segment_activations)
            .saturating_add(self.boundary_states)
            .saturating_add(self.loss_workspace)
            .saturating_add(self.safety_margin)
    }
}

/// Aggregated working-set estimate for one training step.
#[derive(Debug, Clone, Copy)]
pub struct WorkingSet {
    pub total_bytes: u64,
    pub max_seq_len: usize,
    pub sft_loss_route: Option<SftFlceLossRoute>,
    pub breakdown: Breakdown,
}

/// SFT-only inputs that must remain bound to the backend's loss route.
///
/// Keeping these values together prevents admission from applying a sparse
/// active-token estimate to an unknown or different runtime loss path.
#[derive(Debug, Clone, Copy)]
pub struct SftEstimateOptions {
    pub max_active_tokens: usize,
    pub loss_route: SftFlceLossRoute,
    pub checkpoint_boundary_policy: CheckpointBoundaryPolicy,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EstimateOptions {
    /// SFT-specific loss and sparse-boundary inputs. GRPO and OPD leave this
    /// as `None` and retain every checkpoint boundary during reverse replay.
    pub sft: Option<SftEstimateOptions>,
    pub activation_bytes_per_elem: Option<usize>,
    pub streaming_gdn_tile_tokens: Option<usize>,
    /// Persistent optimizer state that must coexist with params and grads.
    pub optimizer: Optimizer,
    /// Whether backend residency aliases tensor storage or mirrors it.
    pub lora_residency: LoraResidency,
}

/// Model- and resource-derived upper bounds for a uniformly applied LoRA rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraRankCeiling {
    /// Largest rank that remains low-rank for every trained projection.
    pub model: usize,
    /// Largest rank whose params, grads, and optimizer state fit the budget.
    pub resource: usize,
    /// The enforceable ceiling (`min(model, resource)`).
    pub effective: usize,
    /// Rank-linear LoRA working-set bytes charged for each rank unit.
    pub bytes_per_rank: u64,
}

const BYTES_PER_GB: u64 = 1024 * 1024 * 1024;
/// Default safety margin: 1 GB. Large enough to absorb scratch
/// allocations the closed-form pieces don't model directly (DRM
/// import buffers, allocator slack, kernel staging buffers).
const SAFETY_MARGIN_BYTES: u64 = BYTES_PER_GB;
const FLCE_MAX_AUTO_CHUNK: usize = 4096;
const FLCE_FALLBACK_SCRATCH_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

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
    let h = usize_to_u64_saturating(cfg.hidden_size);
    let i = usize_to_u64_saturating(cfg.intermediate_size);
    let v = usize_to_u64_saturating(cfg.vocab_size);
    let layers = usize_to_u64_saturating(cfg.num_layers);

    // Embedding + LM head (often tied, but we count both to stay conservative).
    let embed_bytes = v.saturating_mul(h).saturating_mul(elem).saturating_mul(2);
    // Per-layer projections (Q, K, V, O on full attention; gate/up/down for MLP).
    // Rough composition: q_proj ~ h * gate_h, k/v ~ h * kv_h, o_proj ~ h * h.
    // We collapse to (4 * h * h) for attention as a conservative upper bound,
    // and (3 * h * i) for the MLP, plus 4 * h for RMSNorm pairs per layer.
    let per_layer_attn = 4u64
        .saturating_mul(h)
        .saturating_mul(h)
        .saturating_mul(elem);
    let per_layer_mlp = 3u64
        .saturating_mul(h)
        .saturating_mul(i)
        .saturating_mul(elem);
    let per_layer_norms = 4u64.saturating_mul(h).saturating_mul(elem);
    let per_layer_total = per_layer_attn
        .saturating_add(per_layer_mlp)
        .saturating_add(per_layer_norms);
    embed_bytes.saturating_add(per_layer_total.saturating_mul(layers))
}

/// Activations live for the segment currently being recomputed.
///
/// Closed-form upper bound per layer: hidden state stash + QKV + attn
/// output + MLP up/gate/down intermediates. Multiplied by sequence
/// length and dtype size, then by the number of layers per segment.
fn estimate_activation_bytes_per_elem(cfg: &ModelConfig, options: EstimateOptions) -> u64 {
    options
        .activation_bytes_per_elem
        .map(|bytes| usize_to_u64_saturating(bytes.max(1)))
        .unwrap_or_else(|| dtype_bytes(cfg.dtype))
}

fn per_segment_activation_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    num_segments: usize,
    activation_bytes_per_elem: u64,
) -> u64 {
    let elem = activation_bytes_per_elem.max(1);
    let h = usize_to_u64_saturating(cfg.hidden_size);
    let i = usize_to_u64_saturating(cfg.intermediate_size);
    let t = usize_to_u64_saturating(max_seq_len);
    let layers_per_seg =
        usize_to_u64_saturating(cfg.num_layers.div_ceil(num_segments.max(1)).max(1));
    // Per layer: 6 hidden-sized tensors + 2 intermediate-sized tensors.
    let per_layer_width = 6u64
        .saturating_mul(h)
        .saturating_add(2u64.saturating_mul(i));
    per_layer_width
        .saturating_mul(t)
        .saturating_mul(elem)
        .saturating_mul(layers_per_seg)
}

/// Boundary states between segments — always live.
fn boundary_state_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    num_segments: usize,
    sft_checkpoint_boundary_policy: Option<CheckpointBoundaryPolicy>,
    activation_bytes_per_elem: u64,
) -> u64 {
    let elem = activation_bytes_per_elem.max(1);
    if let Some(policy) = sft_checkpoint_boundary_policy
        && policy.recompute_for(max_seq_len)
    {
        let h = usize_to_u64_saturating(cfg.hidden_size);
        let t = usize_to_u64_saturating(max_seq_len);
        let anchor_stride = policy.anchor_stride_for_shape(
            max_seq_len,
            num_segments,
            cfg.hidden_size,
            activation_bytes_per_elem as usize,
        );
        let anchor_count = usize_to_u64_saturating(checkpoint_boundary_anchor_count(
            num_segments,
            anchor_stride,
        ));
        let hidden_tokens = h.saturating_mul(t);
        let anchor_bytes = anchor_count
            .saturating_mul(hidden_tokens)
            .saturating_mul(elem);
        // Long-context SFT recomputes segment inputs on demand. At peak it
        // keeps sparse boundary anchors, the upstream hidden gradient (F32),
        // one detached segment input (model dtype), and one F32-sized cushion
        // for allocator overlap during replay/backprop.
        return anchor_bytes
            .saturating_add(hidden_tokens.saturating_mul(2).saturating_mul(4))
            .saturating_add(hidden_tokens.saturating_mul(elem));
    }
    let h = usize_to_u64_saturating(cfg.hidden_size);
    let t = usize_to_u64_saturating(max_seq_len);
    usize_to_u64_saturating(kiln_train::retained_checkpoint_boundary_count(num_segments))
        .saturating_mul(h)
        .saturating_mul(t)
        .saturating_mul(elem)
}

pub fn checkpoint_boundary_anchor_count(num_segments: usize, anchor_stride: usize) -> usize {
    let anchor_stride = anchor_stride.max(1);
    (num_segments / anchor_stride).saturating_add(1)
}

/// Legacy generic chunked-loss estimate used by non-SFT workloads until their
/// loss routes are represented explicitly.
///
/// The vk-native FLCE chunks the LM-head matmul along the VOCAB axis
/// (shape-aware columns per chunk, processed sequentially).
/// Peak memory at any moment is therefore approximately:
///
///   per-chunk logits `[num_active, chunk_len]` (F32, in-place reused as
///       grad-logits during backward) +
///   weight slice `[chunk_len, hidden]` (F32, copied via `vk_narrow_lastdim`) +
///   grad-hidden accumulator `[num_active, hidden]` (F32, lives across the
///       whole vocab loop).
///
/// Uses the same shape-aware policy as the runtime path.
fn flce_chunk_intermediate_bytes(cfg: &ModelConfig, max_seq_len: usize) -> u64 {
    let h = usize_to_u64_saturating(cfg.hidden_size);
    let t = usize_to_u64_saturating(max_seq_len);
    let chunk_len = usize_to_u64_saturating(active_flce_chunk_len(cfg, max_seq_len));
    let per_chunk_logits = t.saturating_mul(chunk_len).saturating_mul(4); // F32 logits / grad-logits
    // FLCE now slices `[chunk_len, hidden]` and transposes only that
    // chunk to `[hidden, chunk_len]`; both buffers can be live at once.
    let per_chunk_weight = 2u64
        .saturating_mul(chunk_len)
        .saturating_mul(h)
        .saturating_mul(4);
    let grad_hidden = t.saturating_mul(h).saturating_mul(4); // accumulator across vocab loop
    per_chunk_logits
        .saturating_add(per_chunk_weight)
        .saturating_add(grad_hidden)
}

fn active_flce_chunk_len(cfg: &ModelConfig, max_active_tokens: usize) -> usize {
    let active = usize_to_u64_saturating(max_active_tokens.max(1));
    let hidden = usize_to_u64_saturating(cfg.hidden_size.max(1));
    let bytes_per_vocab_col = 4u64.saturating_mul(active.saturating_add(hidden.saturating_mul(2)));
    let by_memory = (FLCE_FALLBACK_SCRATCH_BUDGET_BYTES / bytes_per_vocab_col).max(1) as usize;
    let raw = cfg
        .vocab_size
        .max(1)
        .min(FLCE_MAX_AUTO_CHUNK)
        .min(by_memory)
        .max(1);
    let rounded = if raw <= 1 {
        1
    } else {
        1usize << (usize::BITS - 1 - raw.leading_zeros())
    };

    rounded.max(1).min(cfg.vocab_size.max(1))
}

fn sft_loss_workspace_bytes(cfg: &ModelConfig, max_seq_len: usize, sft: SftEstimateOptions) -> u64 {
    let elem = dtype_bytes(cfg.dtype);
    let t = usize_to_u64_saturating(max_seq_len.max(1));
    let active = usize_to_u64_saturating(sft.max_active_tokens.max(1));
    let hidden = usize_to_u64_saturating(cfg.hidden_size.max(1));
    let vocab = usize_to_u64_saturating(cfg.vocab_size.max(1));
    let chunk = usize_to_u64_saturating(cfg.vocab_size.max(1).min(FLCE_MAX_AUTO_CHUNK));

    match sft.loss_route {
        SftFlceLossRoute::KtTapeFlce => {
            // CUDA/ROCm kt FLCE promotes a BF16/FP16 tied head to one full F32
            // [H,V] tensor. The remaining terms cover the raw active gather,
            // one F32 head chunk, generic ROCm chunk temporaries, active hidden
            // accumulators, the scattered full hidden gradient, and metadata.
            let head_promotion = if elem < 4 {
                4u64.saturating_mul(hidden).saturating_mul(vocab)
            } else {
                0
            };
            let raw_active_hidden = elem.saturating_mul(active).saturating_mul(hidden);
            let f32_elements = hidden
                .saturating_mul(chunk)
                .saturating_add(5u64.saturating_mul(active).saturating_mul(chunk))
                .saturating_add(5u64.saturating_mul(active).saturating_mul(hidden))
                .saturating_add(t.saturating_mul(hidden))
                .saturating_add(8u64.saturating_mul(active));
            head_promotion
                .saturating_add(raw_active_hidden)
                .saturating_add(4u64.saturating_mul(f32_elements))
        }
        SftFlceLossRoute::VulkanActiveRows => {
            // Vulkan caps both automatic and forced chunks at min(V, 4096).
            // Charge the F32 weight slice plus transpose, active-row logits,
            // active/gradient hidden buffers, full hidden output, and metadata.
            let f32_elements = 2u64
                .saturating_mul(hidden)
                .saturating_mul(chunk)
                .saturating_add(active.saturating_mul(chunk))
                .saturating_add(5u64.saturating_mul(active).saturating_mul(hidden))
                .saturating_add(t.saturating_mul(hidden))
                .saturating_add(8u64.saturating_mul(active));
            4u64.saturating_mul(f32_elements)
        }
        SftFlceLossRoute::FullLogits => {
            // The standard tape route retains model-dtype [T,V] logits. Its CE
            // forward/backward also owns an active-row gather, five active F32
            // buffers, and five dense F32 shifted/full-gradient buffers. BF16
            // and FP16 add a model-dtype cast-back gradient.
            let model_dtype_elements = t
                .saturating_mul(vocab)
                .saturating_add(active.saturating_mul(vocab));
            let cast_back_gradient = if elem < 4 {
                elem.saturating_mul(t).saturating_mul(vocab)
            } else {
                0
            };
            let f32_elements = 5u64
                .saturating_mul(active)
                .saturating_mul(vocab)
                .saturating_add(5u64.saturating_mul(t).saturating_mul(vocab))
                .saturating_add(8u64.saturating_mul(active));
            elem.saturating_mul(model_dtype_elements)
                .saturating_add(cast_back_gradient)
                .saturating_add(4u64.saturating_mul(f32_elements))
        }
    }
}

fn loss_workspace_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    sft: Option<SftEstimateOptions>,
) -> u64 {
    match sft {
        Some(sft) => sft_loss_workspace_bytes(cfg, max_seq_len, sft),
        None => flce_chunk_intermediate_bytes(cfg, max_seq_len.max(1)),
    }
}

/// Shape-derived rank-linear upper bound for trainable LoRA elements.
///
/// Each projection pair contains `(in_features + out_features) * rank`
/// elements. One additional full-attention layer is charged for SFT's optional
/// MTP alignment phase: the main adapter and its optimizer state remain live
/// while the draft-block LoRA is trained.
fn lora_parameter_elements_upper_bound(cfg: &ModelConfig, lora_rank: usize) -> u64 {
    if cfg.full_attention_interval == 0 {
        return u64::MAX;
    }
    let h = usize_to_u64_saturating(cfg.hidden_size);
    let i = usize_to_u64_saturating(cfg.intermediate_size);
    let r = usize_to_u64_saturating(lora_rank);
    let layers = usize_to_u64_saturating(cfg.num_layers);
    let full_layers = usize_to_u64_saturating(cfg.num_layers / cfg.full_attention_interval);
    let linear_layers = layers.saturating_sub(full_layers);
    let q_out = usize_to_u64_saturating(cfg.num_attention_heads.saturating_mul(cfg.head_dim));
    let full_q = q_out.saturating_mul(if cfg.attn_output_gate { 2 } else { 1 });
    let kv = usize_to_u64_saturating(cfg.num_kv_heads.saturating_mul(cfg.head_dim));
    let linear_qk = usize_to_u64_saturating(
        cfg.linear_num_key_heads
            .saturating_mul(cfg.linear_key_head_dim),
    );
    let linear_v = usize_to_u64_saturating(
        cfg.linear_num_value_heads
            .saturating_mul(cfg.linear_value_head_dim),
    );
    let linear_qkv = linear_qk.saturating_mul(2).saturating_add(linear_v);

    let mlp_per_layer = h.saturating_add(i).saturating_mul(3);
    let full_attention_per_layer = h
        .saturating_add(full_q)
        .saturating_add(h.saturating_add(kv).saturating_mul(2))
        .saturating_add(q_out.saturating_add(h));
    let linear_attention_per_layer = h
        .saturating_add(linear_qkv)
        .saturating_add(h.saturating_add(linear_v))
        .saturating_add(linear_v.saturating_add(h));
    let main_adapter = mlp_per_layer
        .saturating_mul(layers)
        .saturating_add(full_attention_per_layer.saturating_mul(full_layers))
        .saturating_add(linear_attention_per_layer.saturating_mul(linear_layers));
    let optional_mtp = mlp_per_layer.saturating_add(full_attention_per_layer);
    main_adapter.saturating_add(optional_mtp).saturating_mul(r)
}

/// LoRA params + their gradients, conservatively F32 for every physical copy.
fn lora_param_and_grad_bytes(cfg: &ModelConfig, lora_rank: usize, residency: LoraResidency) -> u64 {
    lora_parameter_elements_upper_bound(cfg, lora_rank)
        .saturating_mul(residency.param_and_grad_f32_copies())
        .saturating_mul(4)
}

fn optimizer_state_tensor_count(optimizer: Optimizer) -> u64 {
    match optimizer {
        Optimizer::Sgd => 0,
        Optimizer::Muon { .. } => 1,
        Optimizer::AdamW { .. } => 2,
    }
}

/// Persistent optimizer state is always a full F32 tensor per LoRA parameter:
/// one momentum tensor for Muon and first/second moments for AdamW.
fn lora_optimizer_state_bytes_for_residency(
    cfg: &ModelConfig,
    lora_rank: usize,
    optimizer: Optimizer,
    residency: LoraResidency,
) -> u64 {
    lora_parameter_elements_upper_bound(cfg, lora_rank)
        .saturating_mul(optimizer_state_tensor_count(optimizer))
        .saturating_mul(residency.optimizer_state_f32_copies())
        .saturating_mul(4)
}

/// Vulkan's initial registry upload and each optimizer dispatch temporarily
/// mirror one tensor at a time. Charge the largest possible LoRA matrix as the
/// one-at-a-time registry scratch peak.
fn lora_registry_scratch_bytes(
    cfg: &ModelConfig,
    lora_rank: usize,
    residency: LoraResidency,
) -> u64 {
    if residency != LoraResidency::RegistryMirrored {
        return 0;
    }
    let largest_feature_dim = [
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads.saturating_mul(cfg.head_dim),
        cfg.num_attention_heads
            .saturating_mul(cfg.head_dim)
            .saturating_mul(if cfg.attn_output_gate { 2 } else { 1 }),
        cfg.num_kv_heads.saturating_mul(cfg.head_dim),
        cfg.linear_num_key_heads
            .saturating_mul(cfg.linear_key_head_dim)
            .saturating_mul(2)
            .saturating_add(
                cfg.linear_num_value_heads
                    .saturating_mul(cfg.linear_value_head_dim),
            ),
        cfg.linear_num_value_heads
            .saturating_mul(cfg.linear_value_head_dim),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    usize_to_u64_saturating(largest_feature_dim)
        .saturating_mul(usize_to_u64_saturating(lora_rank))
        .saturating_mul(4)
}

fn lora_working_set_bytes_per_rank(
    cfg: &ModelConfig,
    optimizer: Optimizer,
    residency: LoraResidency,
) -> u64 {
    lora_param_and_grad_bytes(cfg, 1, residency)
        .saturating_add(lora_optimizer_state_bytes_for_residency(
            cfg, 1, optimizer, residency,
        ))
        .saturating_add(lora_registry_scratch_bytes(cfg, 1, residency))
}

/// Largest useful uniform rank for the projection set trained by Kiln.
///
/// A rank above either matrix dimension is no longer a low-rank update. Since
/// one rank is applied to every target, the smallest trained projection is the
/// architecture-derived ceiling. Invalid or degenerate model shapes fail
/// closed with a zero ceiling.
pub fn model_lora_rank_ceiling(cfg: &ModelConfig) -> usize {
    if cfg.num_layers == 0 || cfg.full_attention_interval == 0 {
        return 0;
    }

    let mut ceiling = cfg.hidden_size.min(cfg.intermediate_size);
    if ceiling == 0 {
        return 0;
    }

    let mut include_projection = |in_features: usize, out_features: usize| {
        ceiling = ceiling.min(in_features).min(out_features);
    };
    let kv_dim = cfg.num_kv_heads.saturating_mul(cfg.head_dim);
    let q_out_dim = cfg.num_attention_heads.saturating_mul(cfg.head_dim);
    let full_q_dim = q_out_dim.saturating_mul(if cfg.attn_output_gate { 2 } else { 1 });
    let linear_qk_dim = cfg
        .linear_num_key_heads
        .saturating_mul(cfg.linear_key_head_dim);
    let linear_v_dim = cfg
        .linear_num_value_heads
        .saturating_mul(cfg.linear_value_head_dim);
    let linear_qkv_dim = linear_qk_dim.saturating_mul(2).saturating_add(linear_v_dim);

    let full_layer_count = cfg.num_layers / cfg.full_attention_interval;
    // SFT may train one MTP full-attention layer while the main adapter and
    // optimizer remain live, so its projection dimensions always participate.
    include_projection(cfg.hidden_size, full_q_dim);
    include_projection(cfg.hidden_size, kv_dim);
    include_projection(cfg.hidden_size, kv_dim);
    include_projection(q_out_dim, cfg.hidden_size);
    if full_layer_count < cfg.num_layers {
        include_projection(cfg.hidden_size, linear_qkv_dim);
        include_projection(cfg.hidden_size, linear_v_dim);
        include_projection(linear_v_dim, cfg.hidden_size);
    }
    ceiling
}

/// Derive the rank ceiling for an already-shaped working-set plan.
///
/// `estimate.breakdown.fixed_bytes()` contains activations and other costs
/// for the selected checkpoint plan. The remainder is divided by the exact
/// rank-linear upper bound for params, grads, registry scratch, and this
/// optimizer's persistent state. A zero/overflowed shape produces a zero
/// resource ceiling.
pub fn lora_rank_ceiling_for_budget(
    cfg: &ModelConfig,
    optimizer: Optimizer,
    residency: LoraResidency,
    available_bytes: u64,
    estimate: &WorkingSet,
) -> LoraRankCeiling {
    let model = model_lora_rank_ceiling(cfg);
    let bytes_per_rank = lora_working_set_bytes_per_rank(cfg, optimizer, residency);
    let rank_budget = available_bytes.saturating_sub(estimate.breakdown.fixed_bytes());
    let resource_u64 = if bytes_per_rank == 0 {
        0
    } else {
        rank_budget / bytes_per_rank
    };
    let resource = usize::try_from(resource_u64).unwrap_or(usize::MAX);
    LoraRankCeiling {
        model,
        resource,
        effective: model.min(resource),
        bytes_per_rank,
    }
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
    let activation_bytes_per_elem = estimate_activation_bytes_per_elem(cfg, options);
    let per_segment_activations = options
        .streaming_gdn_tile_tokens
        .filter(|&tile| tile > 0 && tile < max_seq_len)
        .map(|tile| {
            checkpointed_layerwise_streaming_activation_bytes(cfg, max_seq_len, num_segments, tile)
        })
        .unwrap_or_else(|| {
            per_segment_activation_bytes(cfg, max_seq_len, num_segments, activation_bytes_per_elem)
        });
    let bd = Breakdown {
        base_weights,
        per_segment_activations,
        boundary_states: boundary_state_bytes(
            cfg,
            max_seq_len,
            num_segments,
            options.sft.map(|sft| sft.checkpoint_boundary_policy),
            activation_bytes_per_elem,
        ),
        loss_workspace: loss_workspace_bytes(cfg, max_seq_len, options.sft),
        lora_param_grad: lora_param_and_grad_bytes(cfg, lora_rank, options.lora_residency),
        lora_optimizer_state: lora_optimizer_state_bytes_for_residency(
            cfg,
            lora_rank,
            options.optimizer,
            options.lora_residency,
        ),
        lora_registry_scratch: lora_registry_scratch_bytes(cfg, lora_rank, options.lora_residency),
        safety_margin: SAFETY_MARGIN_BYTES,
    };
    WorkingSet {
        total_bytes: bd.total(),
        max_seq_len,
        sft_loss_route: options.sft.map(|sft| sft.loss_route),
        breakdown: bd,
    }
}

// The flat argument list mirrors the CLI-flag/API field set 1:1; a parameter struct would obscure that correspondence, and changing the signature would be a breaking API change.
#[allow(clippy::too_many_arguments)]
pub fn auto_fit_checkpoint_segments(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    max_segments: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
    options: EstimateOptions,
    available_bytes: u64,
) -> (usize, WorkingSet) {
    let max_segments = max_segments.max(1);
    let mut last = estimate_step_working_set_with_options(
        cfg,
        max_seq_len,
        lora_rank,
        1,
        residency,
        weights_already_resident,
        options,
    );
    if last.total_bytes <= available_bytes {
        return (1, last);
    }
    if options
        .sft
        .is_some_and(|sft| sft.loss_route == SftFlceLossRoute::FullLogits)
    {
        // The checkpoint tail runs outside an active tape, so FullLogits has
        // no executable checkpointed loss-value path. Never manufacture an
        // admission plan the trainer must reject.
        return (1, last);
    }
    for num_segments in 2..=max_segments {
        let estimate = estimate_step_working_set_with_options(
            cfg,
            max_seq_len,
            lora_rank,
            num_segments,
            residency,
            weights_already_resident,
            options,
        );
        if estimate.total_bytes <= available_bytes {
            return (num_segments, estimate);
        }
        last = estimate;
    }
    (max_segments, last)
}

fn f32_matrix_bytes(rows: usize, cols: usize) -> u64 {
    usize_to_u64_saturating(rows)
        .saturating_mul(usize_to_u64_saturating(cols))
        .saturating_mul(4)
}

fn ceil_div_u64(n: u64, d: u64) -> u64 {
    if d == 0 {
        0
    } else {
        (n / d).saturating_add(u64::from(!n.is_multiple_of(d)))
    }
}

/// Peak activation estimate for one replayed layer/subblock.
///
/// This shape model replays one subgraph at a time:
///   - Full-attention layers split attention, MLP gate/up, and MLP down.
///   - GDN chunkwise backward saves recurrent state snapshots and
///     recomputes per-chunk intermediates during backward.
///
/// The estimate is intentionally shape-based rather than layer-count
/// based: peak memory is dominated by one replayed layer/subblock, not
/// by the number of layers in a segment.
fn layerwise_recompute_activation_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    streaming_gdn_tile_tokens: Option<usize>,
) -> u64 {
    let t = max_seq_len;
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let q_dim = cfg.num_attention_heads.saturating_mul(cfg.head_dim);
    let kv_dim = cfg.num_kv_heads.saturating_mul(cfg.head_dim);
    let q_raw_dim = q_dim.saturating_mul(if cfg.attn_output_gate { 2 } else { 1 });

    let hidden = f32_matrix_bytes(t, h);
    let intermediate = f32_matrix_bytes(t, i);
    let q = f32_matrix_bytes(t, q_dim);
    let q_raw = f32_matrix_bytes(t, q_raw_dim);
    let kv = f32_matrix_bytes(t, kv_dim);

    // Forward attention block + recomputed exact flash-style SDPA
    // backward. The implementation does not materialize a [T,T]
    // score matrix and replays attention as a split subgraph, so peak
    // residency is the Q/gate/Q-norm/RoPE path, K/V, attention output,
    // and upstream/boundary tensors rather than all intermediates at
    // once.
    let full_attention_peak = hidden
        .saturating_mul(5)
        .saturating_add(q_raw)
        .saturating_add(q.saturating_mul(5))
        .saturating_add(kv.saturating_mul(4));
    // Split SwiGLU: gate/up/silu/gated dominate. Down-proj replay is
    // lower but still included for non-Qwen shapes.
    let mlp_gate_up_peak = hidden
        .saturating_mul(4)
        .saturating_add(intermediate.saturating_mul(4));
    let mlp_down_peak = hidden
        .saturating_mul(4)
        .saturating_add(intermediate.saturating_mul(3));

    let gdn_t = streaming_gdn_tile_tokens
        .filter(|&tile| tile > 0 && tile < t)
        .unwrap_or(t);
    let hidden_tile = f32_matrix_bytes(gdn_t, h);
    let linear_qk_dim = cfg
        .linear_num_key_heads
        .saturating_mul(cfg.linear_key_head_dim);
    let linear_v_dim = cfg
        .linear_num_value_heads
        .saturating_mul(cfg.linear_value_head_dim);
    let linear_qkv_dim = linear_qk_dim.saturating_mul(2).saturating_add(linear_v_dim);
    let linear_qkv = f32_matrix_bytes(gdn_t, linear_qkv_dim);
    let linear_qk = f32_matrix_bytes(gdn_t, linear_qk_dim);
    let linear_v = f32_matrix_bytes(gdn_t, linear_v_dim);
    let gdn_chunks = ceil_div_u64(usize_to_u64_saturating(gdn_t), 64);
    let gdn_state_snapshots = gdn_chunks
        .saturating_mul(usize_to_u64_saturating(cfg.linear_num_value_heads))
        .saturating_mul(usize_to_u64_saturating(cfg.linear_key_head_dim))
        .saturating_mul(usize_to_u64_saturating(cfg.linear_value_head_dim))
        .saturating_mul(4);
    // The vk-native GDN backward is split into exact subgraphs instead
    // of replaying the whole GDN layer at once. The largest pieces are:
    // no-grad normed recompute for frozen out-proj, chunkwise backward
    // with recurrent snapshots, and conv/split/repeat backward from
    // q/k/v to mixed_qkv.
    let gdn_normed_recompute_peak = hidden
        .saturating_mul(4)
        .saturating_add(hidden_tile)
        .saturating_add(linear_qkv.saturating_mul(2))
        .saturating_add(linear_v.saturating_mul(3))
        .saturating_add(linear_qk.saturating_mul(2));
    let gdn_chunkwise_split_peak = gdn_state_snapshots
        .saturating_add(hidden.saturating_mul(2))
        .saturating_add(hidden_tile)
        .saturating_add(linear_qkv)
        .saturating_add(linear_v.saturating_mul(3))
        .saturating_add(linear_qk.saturating_mul(2));
    let gdn_conv_split_peak = hidden
        .saturating_mul(2)
        .saturating_add(hidden_tile)
        .saturating_add(linear_qkv.saturating_mul(2))
        .saturating_add(linear_v.saturating_mul(3))
        .saturating_add(linear_qk.saturating_mul(2));
    let gdn_peak = gdn_normed_recompute_peak
        .max(gdn_chunkwise_split_peak)
        .max(gdn_conv_split_peak);

    full_attention_peak
        .max(mlp_gate_up_peak)
        .max(mlp_down_peak)
        .max(gdn_peak)
}

/// Peak activation estimate for segment-checkpointed kt tape when GDN
/// replay is time-tiled by the active backend policy.
///
/// Unlike the generic GDN tape fallback, this mirrors the production
/// long-context path: full-attention and MLP subgraphs still scale with the
/// full sequence, while GDN's q/k/v/recurrent intermediates scale with the
/// tape tile. When a checkpoint segment spans multiple layers, multiply the
/// layer/subblock peak by the largest possible layer count in a segment.
fn checkpointed_layerwise_streaming_activation_bytes(
    cfg: &ModelConfig,
    max_seq_len: usize,
    num_segments: usize,
    streaming_gdn_tile_tokens: usize,
) -> u64 {
    let layers_per_seg =
        usize_to_u64_saturating(cfg.num_layers.div_ceil(num_segments.max(1)).max(1));
    layerwise_recompute_activation_bytes(cfg, max_seq_len, Some(streaming_gdn_tile_tokens))
        .saturating_mul(layers_per_seg)
}

/// Peak activation estimate for the vk-native exact layerwise
/// reverse-recompute trainer.
fn vk_native_recompute_activation_bytes(cfg: &ModelConfig, max_seq_len: usize) -> u64 {
    layerwise_recompute_activation_bytes(cfg, max_seq_len, None)
}

/// Closed-form working-set estimate for the vk-native exact
/// layerwise reverse-recompute path used by hybrid Vulkan training.
pub fn estimate_vk_native_recompute_working_set(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
) -> WorkingSet {
    estimate_vk_native_recompute_working_set_with_optimizer(
        cfg,
        max_seq_len,
        lora_rank,
        residency,
        weights_already_resident,
        Optimizer::default(),
    )
}

pub fn estimate_vk_native_recompute_working_set_with_optimizer(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
    optimizer: Optimizer,
) -> WorkingSet {
    estimate_vk_native_recompute_working_set_with_residency(
        cfg,
        max_seq_len,
        lora_rank,
        residency,
        weights_already_resident,
        optimizer,
        LoraResidency::RegistryMirrored,
    )
}

pub fn estimate_vk_native_recompute_working_set_with_residency(
    cfg: &ModelConfig,
    max_seq_len: usize,
    lora_rank: usize,
    residency: WeightResidency,
    weights_already_resident: bool,
    optimizer: Optimizer,
    lora_residency: LoraResidency,
) -> WorkingSet {
    let base_weights = if weights_already_resident {
        0
    } else {
        approximate_base_weight_bytes(cfg).saturating_mul(residency.weight_multiplier())
    };
    let bd = Breakdown {
        base_weights,
        per_segment_activations: vk_native_recompute_activation_bytes(cfg, max_seq_len),
        boundary_states: 0,
        loss_workspace: flce_chunk_intermediate_bytes(cfg, max_seq_len),
        lora_param_grad: lora_param_and_grad_bytes(cfg, lora_rank, lora_residency),
        lora_optimizer_state: lora_optimizer_state_bytes_for_residency(
            cfg,
            lora_rank,
            optimizer,
            lora_residency,
        ),
        lora_registry_scratch: lora_registry_scratch_bytes(cfg, lora_rank, lora_residency),
        safety_margin: if weights_already_resident {
            0
        } else {
            SAFETY_MARGIN_BYTES
        },
    };
    WorkingSet {
        total_bytes: bd.total(),
        max_seq_len,
        sft_loss_route: None,
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
/// Discrete GPUs keep the static behavior: reserve a fraction of the budget
/// for inference, return the rest. Missing detection fails closed with zero.
pub fn available_for_training_bytes(vram: &GpuVramInfo) -> u64 {
    available_for_training_bytes_with_meminfo_details(
        vram,
        query_linux_mem_available_bytes(),
        query_linux_mem_total_bytes(),
    )
}

#[doc(hidden)]
pub fn available_for_training_bytes_with_meminfo(
    vram: &GpuVramInfo,
    mem_available_bytes: Option<u64>,
) -> u64 {
    available_for_training_bytes_with_meminfo_details(vram, mem_available_bytes, None)
}

#[doc(hidden)]
pub fn available_for_training_bytes_with_meminfo_details(
    vram: &GpuVramInfo,
    mem_available_bytes: Option<u64>,
    _mem_total_bytes: Option<u64>,
) -> u64 {
    if vram.total_bytes == 0 {
        return 0;
    }

    // Unified memory: training and inference share the same physical
    // pool, so MemAvailable_now is the truth. Cap by the effective capacity so
    // a host cannot report more available memory than the GPU can address.
    if vram.unified
        && let Some(mem_avail) = mem_available_bytes
    {
        let live = mem_avail.saturating_sub(SAFETY_MARGIN_BYTES);
        return live.min(vram.total_bytes.saturating_sub(SAFETY_MARGIN_BYTES));
    }
    // No live host signal (for example non-Linux Apple Silicon): fall
    // through to the conservative static path. Capacity detection already
    // retained unified-memory system headroom.

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
    query_linux_meminfo_kib("MemAvailable:").map(|kib| kib.saturating_mul(1024))
}

#[cfg(target_os = "linux")]
fn query_linux_mem_total_bytes() -> Option<u64> {
    query_linux_meminfo_kib("MemTotal:").map(|kib| kib.saturating_mul(1024))
}

#[cfg(target_os = "linux")]
fn query_linux_meminfo_kib(prefix: &str) -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())?;
            return Some(kib);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn query_linux_mem_available_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "linux"))]
fn query_linux_mem_total_bytes() -> Option<u64> {
    None
}

/// Best-effort token count estimate for one example without paying
/// real tokenizer cost on every submission.
///
/// If `tokenizer` is provided, use the real chat-template token count for
/// each example. If tokenization is unavailable, fall back to a chars/4
/// estimate plus template-envelope overhead.
pub fn approximate_max_seq_len_sft(
    examples: &[SftExample],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    examples
        .iter()
        .map(|example| {
            tokenizer
                .and_then(|tokenizer| {
                    kiln_train::trainer::tokenize_for_training(example, tokenizer)
                        .ok()
                        .map(|(input_ids, _)| input_ids.len())
                })
                .unwrap_or_else(|| approximate_tokens_for_messages(&example.messages, tokenizer))
        })
        .max()
        .unwrap_or(0)
}

pub fn approximate_max_supervised_tokens_sft(
    examples: &[SftExample],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    examples
        .iter()
        .map(|example| {
            tokenizer
                .and_then(|tokenizer| {
                    kiln_train::trainer::tokenize_for_training(example, tokenizer)
                        .ok()
                        .map(|(_, label_mask)| {
                            label_mask.into_iter().filter(|active| *active).count()
                        })
                })
                .unwrap_or_else(|| {
                    example
                        .messages
                        .iter()
                        .filter(|message| message.role == "assistant")
                        .map(|message| approximate_tokens_for_text(&message.content, tokenizer))
                        .sum::<usize>()
                })
        })
        .max()
        .unwrap_or(0)
}

pub fn approximate_max_seq_len_grpo(
    groups: &[GrpoGroup],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    groups
        .iter()
        .map(|g| approximate_max_seq_len_grpo_group(g, tokenizer))
        .max()
        .unwrap_or(0)
}

/// Approximate the longest training sequence an OPD job builds: the longest
/// chat-templated prompt plus the rollout budget (`max_tokens`). Off-policy
/// dataset jobs (empty `prompts`, fed by `dataset_path`) fall back to the
/// rollout budget alone — an under-estimate the trainer's own guard backstops,
/// but enough to give the governor a non-zero working-set reservation.
pub fn approximate_max_seq_len_opd(
    prompts: &[kiln_train::opd::OpdPrompt],
    max_tokens: usize,
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    let longest_prompt = prompts
        .iter()
        .map(|p| approximate_tokens_for_messages(&p.messages, tokenizer))
        .max()
        .unwrap_or(0);
    longest_prompt.saturating_add(max_tokens)
}

pub fn approximate_max_seq_len_grpo_group(
    group: &GrpoGroup,
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    let prompt = approximate_tokens_for_messages(&group.messages, tokenizer);
    let max_completion = group
        .completions
        .iter()
        .map(|c| approximate_tokens_for_text(&c.text, tokenizer))
        .max()
        .unwrap_or(0);
    prompt.saturating_add(max_completion)
}

fn approximate_tokens_for_messages(
    messages: &[kiln_train::ChatMessage],
    tokenizer: Option<&KilnTokenizer>,
) -> usize {
    // Sum every message's content in chars, plus a 16-token-per-message
    // tag for chat-template envelope overhead.
    let chars = messages.iter().fold(0usize, |total, message| {
        let tool_chars = message
            .tool_calls
            .as_ref()
            .and_then(|tool_calls| serde_json::to_string(tool_calls).ok())
            .map_or(0, |value| value.chars().count());
        total.saturating_add(message.content.chars().count().saturating_add(tool_chars))
    });
    let envelope = messages.len().saturating_mul(16);
    let char_estimate = (chars / 4).saturating_add(envelope);

    if let Some(tok) = tokenizer {
        let core: Vec<kiln_core::tokenizer::ChatMessage> = messages.to_vec();
        if let Ok(text) = tok.apply_chat_template(&core)
            && let Ok(ids) = tok.encode(&text)
        {
            return ids.len();
        }
    }
    char_estimate
}

fn approximate_tokens_for_text(text: &str, tokenizer: Option<&KilnTokenizer>) -> usize {
    let char_estimate = (text.chars().count() / 4).saturating_add(4);
    if let Some(tok) = tokenizer
        && let Ok(ids) = tok.encode(text)
    {
        return ids.len().max(char_estimate);
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
/// knowing whether the available number came from the live UMA signal
/// rather than a static discrete-GPU budget.
pub fn format_oom_message_with_source(
    estimate: &WorkingSet,
    available_bytes: u64,
    lora_rank: usize,
    num_segments: usize,
    vram_source: Option<kiln_memory::vram::VramSource>,
) -> String {
    let est_gb = estimate.total_bytes as f64 / BYTES_PER_GB as f64;
    let avail_gb = available_bytes as f64 / BYTES_PER_GB as f64;
    let bd = &estimate.breakdown;
    let bw_gb = bd.base_weights as f64 / BYTES_PER_GB as f64;
    let act_gb = bd
        .per_segment_activations
        .saturating_add(bd.boundary_states) as f64
        / BYTES_PER_GB as f64;
    let loss_workspace_gb = bd.loss_workspace as f64 / BYTES_PER_GB as f64;
    let loss_route = estimate
        .sft_loss_route
        .map(SftFlceLossRoute::as_str)
        .unwrap_or("generic_chunked");
    let lora_gb = bd.lora_param_grad as f64 / BYTES_PER_GB as f64;
    let optimizer_gb = bd.lora_optimizer_state as f64 / BYTES_PER_GB as f64;
    let registry_scratch_gb = bd.lora_registry_scratch as f64 / BYTES_PER_GB as f64;
    let source_clause = match vram_source {
        Some(src) => format!(" (vram_source={src})"),
        None => String::new(),
    };
    format!(
        "Estimated training step working set is {est_gb:.2} GB but only \
         {avail_gb:.2} GB is available{source_clause}. Breakdown: weights {bw_gb:.2} GB, \
         activations {act_gb:.2} GB (max_seq_len={msl}, num_segments={num_segments}), \
         loss workspace {loss_workspace_gb:.2} GB (route={loss_route}), \
         LoRA params+grads {lora_gb:.2} GB, \
         optimizer state {optimizer_gb:.2} GB, residency scratch {registry_scratch_gb:.2} GB \
         (lora_rank={lora_rank}). Dynamic checkpointing already tried up to \
         this segment count. To fit, shrink lora_rank, send fewer/shorter \
         examples per submission, or free memory from other processes.",
        msl = estimate.max_seq_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::config::ModelConfig;
    use kiln_memory::vram::{GpuVramInfo, VramSource};

    fn qwen_4b() -> ModelConfig {
        ModelConfig::qwen3_5_4b()
    }

    fn adamw() -> Optimizer {
        serde_json::from_str(r#"{"kind":"adam_w"}"#).unwrap()
    }

    fn sft_estimate(
        max_active_tokens: usize,
        loss_route: SftFlceLossRoute,
        checkpoint_boundary_policy: CheckpointBoundaryPolicy,
    ) -> SftEstimateOptions {
        SftEstimateOptions {
            max_active_tokens,
            loss_route,
            checkpoint_boundary_policy,
        }
    }

    fn vulkan_sft(max_active_tokens: usize) -> SftEstimateOptions {
        sft_estimate(
            max_active_tokens,
            SftFlceLossRoute::VulkanActiveRows,
            CheckpointBoundaryPolicy::default(),
        )
    }

    #[test]
    fn sft_preflight_uses_training_render_and_exact_assistant_mask() {
        let tokenizer = KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "x": 2},
                    "merges": []
                }
            }"#,
        )
        .unwrap()
        .with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}\
             {% if add_generation_prompt %}x{% endif %}"
                .to_string(),
        );
        let examples = [SftExample {
            messages: vec![
                kiln_train::ChatMessage::new("user", "a"),
                kiln_train::ChatMessage::new("assistant", "b"),
            ],
        }];

        assert_eq!(approximate_max_seq_len_sft(&examples, Some(&tokenizer)), 2);
        assert_eq!(
            approximate_max_supervised_tokens_sft(&examples, Some(&tokenizer)),
            1
        );
        assert_eq!(
            approximate_tokens_for_messages(&examples[0].messages, Some(&tokenizer)),
            3
        );
    }

    #[test]
    fn estimator_grows_with_seq_len() {
        let cfg = qwen_4b();
        let small = estimate_step_working_set(&cfg, 256, 16, 4, WeightResidency::SingleCopy, false);
        let large =
            estimate_step_working_set(&cfg, 4096, 16, 4, WeightResidency::SingleCopy, false);
        assert!(
            large.total_bytes > small.total_bytes,
            "expected larger T to grow estimate; small={} large={}",
            small.total_bytes,
            large.total_bytes
        );
    }

    #[test]
    fn optimizer_state_is_charged_as_full_f32_lora_tensors() {
        let cfg = qwen_4b();
        let estimate = |optimizer| {
            estimate_step_working_set_with_options(
                &cfg,
                256,
                16,
                4,
                WeightResidency::SingleCopy,
                true,
                EstimateOptions {
                    optimizer,
                    ..Default::default()
                },
            )
        };

        let sgd = estimate(Optimizer::Sgd);
        let muon = estimate(Optimizer::default());
        let adamw_estimate = estimate(adamw());
        assert_eq!(sgd.breakdown.lora_optimizer_state, 0);
        assert_eq!(
            muon.breakdown.lora_param_grad,
            sgd.breakdown.lora_param_grad
        );
        assert_eq!(
            adamw_estimate.breakdown.lora_param_grad,
            sgd.breakdown.lora_param_grad
        );
        assert_eq!(
            muon.breakdown.lora_optimizer_state,
            sgd.breakdown.lora_param_grad / 2,
            "Muon owns one F32 state tensor while params+grads charge two"
        );
        assert_eq!(
            adamw_estimate.breakdown.lora_optimizer_state, sgd.breakdown.lora_param_grad,
            "AdamW owns two F32 state tensors"
        );
        assert!(sgd.total_bytes < muon.total_bytes);
        assert!(muon.total_bytes < adamw_estimate.total_bytes);

        let vk_adamw = estimate_vk_native_recompute_working_set_with_optimizer(
            &cfg,
            256,
            16,
            WeightResidency::SingleCopy,
            true,
            adamw(),
        );
        assert_eq!(
            vk_adamw.breakdown.lora_optimizer_state,
            adamw_estimate
                .breakdown
                .lora_optimizer_state
                .saturating_mul(2),
            "vk-native admission must charge storage plus the registry mirror"
        );
        assert!(vk_adamw.breakdown.lora_registry_scratch > 0);
    }

    #[test]
    fn lora_residency_is_selected_by_backend_not_memory_topology() {
        for backend in ["cuda", "cuda-portable", "rocm", "metal", "metal-portable"] {
            assert_eq!(
                LoraResidency::for_backend_name(backend),
                LoraResidency::StorageOwned,
                "{backend} residency must alias tensor storage"
            );
        }
        for backend in [
            "vulkan",
            "vulkan-portable",
            "cpu",
            "portable",
            "future-unknown",
        ] {
            assert_eq!(
                LoraResidency::for_backend_name(backend),
                LoraResidency::RegistryMirrored,
                "{backend} must charge a registry-owned mirror"
            );
        }
    }

    #[test]
    fn vulkan_charges_three_five_seven_persistent_lora_copies() {
        let cfg = qwen_4b();
        let rank = 16;
        let p = lora_parameter_elements_upper_bound(&cfg, rank).saturating_mul(4);
        let estimate = |optimizer, lora_residency| {
            estimate_step_working_set_with_options(
                &cfg,
                256,
                rank,
                4,
                WeightResidency::SingleCopy,
                true,
                EstimateOptions {
                    optimizer,
                    lora_residency,
                    ..Default::default()
                },
            )
        };

        let rocm = LoraResidency::for_backend_name("rocm");
        let cuda = LoraResidency::for_backend_name("cuda");
        let vulkan = LoraResidency::for_backend_name("vulkan");
        let rocm_sgd = estimate(Optimizer::Sgd, rocm);
        let rocm_muon = estimate(Optimizer::default(), rocm);
        let rocm_adamw = estimate(adamw(), rocm);
        let cuda_adamw = estimate(adamw(), cuda);
        let vulkan_sgd = estimate(Optimizer::Sgd, vulkan);
        let vulkan_muon = estimate(Optimizer::default(), vulkan);
        let vulkan_adamw = estimate(adamw(), vulkan);
        let persistent = |working_set: &WorkingSet| {
            working_set
                .breakdown
                .lora_param_grad
                .saturating_add(working_set.breakdown.lora_optimizer_state)
        };

        assert_eq!(persistent(&rocm_sgd), p.saturating_mul(2));
        assert_eq!(persistent(&rocm_muon), p.saturating_mul(3));
        assert_eq!(persistent(&rocm_adamw), p.saturating_mul(4));
        assert_eq!(persistent(&cuda_adamw), p.saturating_mul(4));
        assert_eq!(persistent(&vulkan_sgd), p.saturating_mul(3));
        assert_eq!(persistent(&vulkan_muon), p.saturating_mul(5));
        assert_eq!(persistent(&vulkan_adamw), p.saturating_mul(7));
        assert_eq!(rocm_adamw.breakdown.lora_registry_scratch, 0);
        assert!(vulkan_adamw.breakdown.lora_registry_scratch > 0);
    }

    #[test]
    fn lora_rank_ceiling_is_derived_from_model_and_optimizer_budget() {
        let cfg = qwen_4b();
        let estimate = estimate_step_working_set_with_options(
            &cfg,
            256,
            1,
            4,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                optimizer: Optimizer::Sgd,
                ..Default::default()
            },
        );
        assert_eq!(model_lora_rank_ceiling(&cfg), 1024);

        let sgd_per_rank =
            lora_working_set_bytes_per_rank(&cfg, Optimizer::Sgd, LoraResidency::StorageOwned);
        let budget = estimate
            .breakdown
            .fixed_bytes()
            .saturating_add(sgd_per_rank.saturating_mul(120));
        let sgd = lora_rank_ceiling_for_budget(
            &cfg,
            Optimizer::Sgd,
            LoraResidency::StorageOwned,
            budget,
            &estimate,
        );
        let muon = lora_rank_ceiling_for_budget(
            &cfg,
            Optimizer::default(),
            LoraResidency::StorageOwned,
            budget,
            &estimate,
        );
        let adamw = lora_rank_ceiling_for_budget(
            &cfg,
            adamw(),
            LoraResidency::StorageOwned,
            budget,
            &estimate,
        );
        let vulkan_sgd = lora_rank_ceiling_for_budget(
            &cfg,
            Optimizer::Sgd,
            LoraResidency::RegistryMirrored,
            budget,
            &estimate,
        );

        assert_eq!(sgd.resource, 120);
        assert_eq!(muon.resource, 80);
        assert_eq!(adamw.resource, 60);
        assert_eq!(sgd.effective, 120);
        assert_eq!(muon.effective, 80);
        assert_eq!(adamw.effective, 60);
        assert!(
            vulkan_sgd.resource < sgd.resource,
            "registry mirrors and scratch must lower Vulkan's resource-derived rank ceiling"
        );
    }

    #[test]
    fn lora_rank_ceiling_fails_closed_for_degenerate_or_exhausted_shapes() {
        let mut invalid = qwen_4b();
        invalid.num_kv_heads = 0;
        assert_eq!(model_lora_rank_ceiling(&invalid), 0);

        let cfg = qwen_4b();
        let estimate =
            estimate_step_working_set(&cfg, 256, 1, 4, WeightResidency::SingleCopy, true);
        let ceiling = lora_rank_ceiling_for_budget(
            &cfg,
            Optimizer::default(),
            LoraResidency::StorageOwned,
            estimate.breakdown.fixed_bytes().saturating_sub(1),
            &estimate,
        );
        assert_eq!(ceiling.resource, 0);
        assert_eq!(ceiling.effective, 0);
    }

    #[test]
    fn estimator_overflow_saturates_toward_rejection() {
        let mut cfg = qwen_4b();
        cfg.hidden_size = usize::MAX;
        cfg.intermediate_size = usize::MAX;
        cfg.num_layers = usize::MAX;
        cfg.vocab_size = usize::MAX;

        let estimate = estimate_step_working_set_with_options(
            &cfg,
            usize::MAX,
            usize::MAX,
            1,
            WeightResidency::DualResidentCpuAndVulkan,
            false,
            EstimateOptions {
                optimizer: adamw(),
                ..Default::default()
            },
        );
        assert_eq!(estimate.total_bytes, u64::MAX);
        assert_eq!(estimate.breakdown.lora_param_grad, u64::MAX);
        assert_eq!(estimate.breakdown.lora_optimizer_state, u64::MAX);
    }

    #[test]
    fn estimator_shrinks_with_more_segments() {
        let cfg = qwen_4b();
        let few = estimate_step_working_set(&cfg, 1500, 16, 4, WeightResidency::SingleCopy, false);
        let many =
            estimate_step_working_set(&cfg, 1500, 16, 16, WeightResidency::SingleCopy, false);
        assert!(
            many.breakdown.per_segment_activations < few.breakdown.per_segment_activations,
            "more segments must reduce per-segment activation footprint"
        );
    }

    #[test]
    fn estimator_charges_largest_ceiling_segment() {
        let cfg = qwen_4b();
        let seq_len = 1024usize;
        let segments = 5usize;
        let estimate = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            8,
            segments,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                activation_bytes_per_elem: Some(1),
                ..Default::default()
            },
        );
        let expected_layers = cfg.num_layers.div_ceil(segments) as u64;
        let per_layer =
            (6 * cfg.hidden_size as u64 + 2 * cfg.intermediate_size as u64) * seq_len as u64;
        assert_eq!(
            estimate.breakdown.per_segment_activations,
            per_layer * expected_layers
        );
    }

    #[test]
    fn auto_fit_segments_raises_segments_until_long_context_fits() {
        let cfg = qwen_4b();
        let max_seq_len = 104_412;
        let available = 21 * BYTES_PER_GB;
        let four_segment = estimate_step_working_set_with_options(
            &cfg,
            max_seq_len,
            8,
            4,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
            EstimateOptions {
                sft: Some(vulkan_sft(512)),
                activation_bytes_per_elem: Some(2),
                ..Default::default()
            },
        );
        assert!(
            four_segment.total_bytes > available,
            "fixed 4-segment plan should reproduce the rejected long-context case"
        );

        let (segments, fit) = auto_fit_checkpoint_segments(
            &cfg,
            max_seq_len,
            8,
            cfg.num_layers,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
            EstimateOptions {
                sft: Some(vulkan_sft(512)),
                activation_bytes_per_elem: Some(2),
                ..Default::default()
            },
            available,
        );
        assert!(
            segments > 4,
            "auto-fit should raise segments beyond the VRAM-only default"
        );
        assert!(
            fit.total_bytes <= available,
            "auto-fit plan must fit: segments={segments}, estimate={}, available={available}",
            fit.total_bytes
        );
    }

    #[test]
    fn auto_fit_long_gdn_context_rejects_when_even_one_layer_exceeds_available() {
        let cfg = qwen_4b();
        let max_seq_len = 104_412;
        let available = 21 * BYTES_PER_GB;
        let options = EstimateOptions {
            sft: Some(vulkan_sft(512)),
            activation_bytes_per_elem: Some(10),
            ..Default::default()
        };
        let (segments, fit) = auto_fit_checkpoint_segments(
            &cfg,
            max_seq_len,
            8,
            cfg.num_layers,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
            options,
            available,
        );
        assert_eq!(segments, cfg.num_layers);
        assert!(
            fit.total_bytes > available,
            "auto-fit must reject instead of accepting a GDN long-context plan: estimate={}, available={available}",
            fit.total_bytes
        );
    }

    #[test]
    fn streaming_gdn_long_context_requires_route_complete_uma_budget() {
        let cfg = qwen_4b();
        let max_seq_len = 104_412;
        let options = EstimateOptions {
            sft: Some(vulkan_sft(512)),
            activation_bytes_per_elem: Some(10),
            streaming_gdn_tile_tokens: Some(1024),
            ..Default::default()
        };
        let (segments_at_30_gib, rejected) = auto_fit_checkpoint_segments(
            &cfg,
            max_seq_len,
            8,
            cfg.num_layers,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
            options,
            30 * BYTES_PER_GB,
        );
        assert_eq!(segments_at_30_gib, cfg.num_layers);
        assert!(
            rejected.total_bytes > 30 * BYTES_PER_GB,
            "the complete route-specific working set must reject the stale 30 GiB acceptance claim"
        );

        let (segments_at_31_gib, accepted) = auto_fit_checkpoint_segments(
            &cfg,
            max_seq_len,
            8,
            cfg.num_layers,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
            options,
            31 * BYTES_PER_GB,
        );
        assert_eq!(segments_at_31_gib, cfg.num_layers);
        assert_eq!(accepted.total_bytes, rejected.total_bytes);
        assert!(
            accepted.total_bytes <= 31 * BYTES_PER_GB,
            "streaming GDN estimate should accept the 104k-token rank-8 repro only with the next complete GiB of live budget: estimate={} ({:.2} GiB)",
            accepted.total_bytes,
            accepted.total_bytes as f64 / BYTES_PER_GB as f64,
        );
        assert!(
            accepted.breakdown.loss_workspace >= 4 * max_seq_len as u64 * cfg.hidden_size as u64,
            "Vulkan loss workspace must retain the full-sequence F32 gradient upper bound"
        );
        assert!(
            accepted.breakdown.per_segment_activations < 30 * BYTES_PER_GB,
            "streaming GDN should not charge full-sequence GDN intermediates"
        );
    }

    #[test]
    fn estimator_uses_active_tokens_for_sft_loss_workspace() {
        let cfg = qwen_4b();
        let estimate = |max_active_tokens| {
            estimate_step_working_set_with_options(
                &cfg,
                8192,
                16,
                8,
                WeightResidency::SingleCopy,
                false,
                EstimateOptions {
                    sft: Some(vulkan_sft(max_active_tokens)),
                    ..Default::default()
                },
            )
        };
        let full_prompt = estimate(8192);
        let sparse_labels = estimate(512);
        assert!(
            sparse_labels.breakdown.loss_workspace < full_prompt.breakdown.loss_workspace,
            "SFT loss estimate should scale with active tokens"
        );
    }

    #[test]
    fn sft_loss_workspace_matches_route_specific_upper_bounds() {
        let mut cfg = qwen_4b();
        cfg.hidden_size = 8;
        cfg.vocab_size = 16;
        cfg.dtype = DType::BF16;
        let policy = CheckpointBoundaryPolicy::default();
        let workspace =
            |loss_route| sft_loss_workspace_bytes(&cfg, 4, sft_estimate(2, loss_route, policy));

        assert_eq!(workspace(SftFlceLossRoute::KtTapeFlce), 2_208);
        assert_eq!(workspace(SftFlceLossRoute::VulkanActiveRows), 1_664);
        assert_eq!(workspace(SftFlceLossRoute::FullLogits), 2_304);
    }

    #[test]
    fn full_logits_keeps_dense_sequence_workspace_when_labels_are_sparse() {
        let mut cfg = qwen_4b();
        cfg.hidden_size = 8;
        cfg.vocab_size = 16;
        let policy = CheckpointBoundaryPolicy::default();
        let sparse = sft_loss_workspace_bytes(
            &cfg,
            32,
            sft_estimate(1, SftFlceLossRoute::FullLogits, policy),
        );
        let dense = sft_loss_workspace_bytes(
            &cfg,
            32,
            sft_estimate(32, SftFlceLossRoute::FullLogits, policy),
        );

        assert!(sparse < dense);
        assert!(sparse >= 32 * 16 * dtype_bytes(cfg.dtype));
    }

    #[test]
    fn every_sft_loss_route_saturates_toward_rejection() {
        let mut cfg = qwen_4b();
        cfg.hidden_size = usize::MAX;
        cfg.vocab_size = usize::MAX;
        let policy = CheckpointBoundaryPolicy::default();
        for route in [
            SftFlceLossRoute::KtTapeFlce,
            SftFlceLossRoute::VulkanActiveRows,
            SftFlceLossRoute::FullLogits,
        ] {
            assert_eq!(
                sft_loss_workspace_bytes(&cfg, usize::MAX, sft_estimate(usize::MAX, route, policy),),
                u64::MAX,
                "route {} must saturate instead of wrapping",
                route.as_str()
            );
        }
    }

    #[test]
    fn full_logits_auto_fit_never_selects_checkpointing() {
        let cfg = qwen_4b();
        let (segments, estimate) = auto_fit_checkpoint_segments(
            &cfg,
            8192,
            16,
            cfg.num_layers,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                sft: Some(sft_estimate(
                    512,
                    SftFlceLossRoute::FullLogits,
                    CheckpointBoundaryPolicy::default(),
                )),
                ..Default::default()
            },
            0,
        );

        assert_eq!(segments, 1);
        assert_eq!(estimate.sft_loss_route, Some(SftFlceLossRoute::FullLogits));
        assert!(estimate.total_bytes > 0);
    }

    #[test]
    fn flce_chunk_len_shrinks_for_long_context_without_env_tuning() {
        let cfg = qwen_4b();
        let short = active_flce_chunk_len(&cfg, 512);
        let long = active_flce_chunk_len(&cfg, 65_536);

        assert!(
            long < short,
            "shape-aware FLCE chunking should reduce chunk size for long contexts: short={short}, long={long}"
        );
        assert!(long > 0);
    }

    #[test]
    fn vk_native_recompute_has_no_segment_boundaries() {
        let cfg = qwen_4b();
        let est = estimate_vk_native_recompute_working_set(
            &cfg,
            8192,
            16,
            WeightResidency::SingleCopy,
            true,
        );
        assert_eq!(est.breakdown.boundary_states, 0);
        assert!(
            est.breakdown.per_segment_activations > 0,
            "recompute estimate must still count replayed subgraph activations"
        );
    }

    #[test]
    fn estimator_recompute_boundaries_does_not_scale_with_segment_count() {
        let cfg = qwen_4b();
        let seq_len = 104_412;
        let cached =
            estimate_step_working_set(&cfg, seq_len, 16, 32, WeightResidency::SingleCopy, false);
        let recompute = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            16,
            32,
            WeightResidency::SingleCopy,
            false,
            EstimateOptions {
                sft: Some(vulkan_sft(seq_len)),
                ..Default::default()
            },
        );
        assert!(
            recompute.breakdown.boundary_states < cached.breakdown.boundary_states,
            "recomputed-boundary estimate should not charge all segment boundaries"
        );
    }

    #[test]
    fn retained_boundary_estimate_matches_grpo_and_opd_runtime_contract() {
        let cfg = qwen_4b();
        let seq_len = 104_412usize;
        let num_segments = 32usize;
        let estimate = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            16,
            num_segments,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions::default(),
        );
        let runtime_boundary_count =
            kiln_train::retained_checkpoint_boundary_count(num_segments) as u64;
        let expected = runtime_boundary_count
            * cfg.hidden_size as u64
            * seq_len as u64
            * dtype_bytes(cfg.dtype);
        assert_eq!(estimate.breakdown.boundary_states, expected);

        let disabled_sft = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            16,
            num_segments,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                sft: Some(sft_estimate(
                    seq_len,
                    SftFlceLossRoute::VulkanActiveRows,
                    CheckpointBoundaryPolicy::from_parts(
                        kiln_train::CheckpointBoundaryRecomputeMode::Disabled,
                        1,
                        None,
                        1,
                    )
                    .expect("disabled SFT boundary policy"),
                )),
                ..Default::default()
            },
        );
        assert_eq!(disabled_sft.breakdown.boundary_states, expected);

        let sparse = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            16,
            num_segments,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                sft: Some(vulkan_sft(seq_len)),
                ..Default::default()
            },
        );
        assert!(
            estimate.breakdown.boundary_states > sparse.breakdown.boundary_states,
            "the GRPO/OPD retained-boundary contract must not use SFT's sparse estimate"
        );
    }

    #[test]
    fn recompute_boundary_estimate_charges_sparse_anchors_and_segment_input() {
        let cfg = qwen_4b();
        let seq_len = 8192usize;
        let num_segments = 32usize;
        let policy = CheckpointBoundaryPolicy::from_parts(
            kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
            1,
            Some(4),
            1,
        )
        .expect("explicit sparse-boundary policy");
        let estimate = estimate_step_working_set_with_options(
            &cfg,
            seq_len,
            16,
            num_segments,
            WeightResidency::SingleCopy,
            true,
            EstimateOptions {
                sft: Some(sft_estimate(
                    seq_len,
                    SftFlceLossRoute::VulkanActiveRows,
                    policy,
                )),
                ..Default::default()
            },
        );
        let h = cfg.hidden_size as u64;
        let t = seq_len as u64;
        let elem = dtype_bytes(cfg.dtype);
        let anchor_stride =
            policy.anchor_stride_for_shape(seq_len, num_segments, cfg.hidden_size, elem as usize);
        let anchor_count = checkpoint_boundary_anchor_count(num_segments, anchor_stride) as u64;
        let expected = anchor_count * h * t * elem + 2 * h * t * 4 + h * t * elem;
        assert_eq!(estimate.breakdown.boundary_states, expected);
    }

    #[test]
    fn vk_native_recompute_is_lower_than_four_layer_segment_at_long_context() {
        let cfg = qwen_4b();
        let segmented =
            estimate_step_working_set(&cfg, 8192, 16, 4, WeightResidency::SingleCopy, true);
        let recompute = estimate_vk_native_recompute_working_set(
            &cfg,
            8192,
            16,
            WeightResidency::SingleCopy,
            true,
        );
        assert!(
            recompute.breakdown.per_segment_activations
                < segmented.breakdown.per_segment_activations + segmented.breakdown.boundary_states,
            "exact layerwise recompute should estimate below four-layer segment residency"
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
        let single =
            estimate_step_working_set(&cfg, 1500, 16, 4, WeightResidency::SingleCopy, false);
        let dual = estimate_step_working_set(
            &cfg,
            1500,
            16,
            4,
            WeightResidency::DualResidentCpuAndVulkan,
            false,
        );
        assert_eq!(
            dual.breakdown.base_weights,
            single.breakdown.base_weights * 2
        );
        assert!(dual.total_bytes > single.total_bytes);
    }

    #[test]
    fn configured_capacity_does_not_hide_weight_memory_topology() {
        let unified = GpuVramInfo {
            total_bytes: 96 * BYTES_PER_GB,
            source: VramSource::ConfigOverride,
            unified: true,
        };
        let discrete = GpuVramInfo {
            total_bytes: 16 * BYTES_PER_GB,
            source: VramSource::ConfigOverride,
            unified: false,
        };

        assert_eq!(
            WeightResidency::for_vram(&unified),
            WeightResidency::DualResidentCpuAndVulkan
        );
        assert_eq!(
            WeightResidency::for_vram(&discrete),
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
            unified: true,
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
    fn unified_preflight_never_ignores_live_host_memory() {
        let vram = GpuVramInfo {
            total_bytes: 120 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let avail = available_for_training_bytes_with_meminfo_details(
            &vram,
            Some(16 * BYTES_PER_GB),
            Some(32 * BYTES_PER_GB),
        );
        assert_eq!(
            avail,
            16 * BYTES_PER_GB - SAFETY_MARGIN_BYTES,
            "unified address-space counters must not override live host availability"
        );
    }

    #[test]
    fn configured_unified_capacity_remains_bounded_by_live_host_memory() {
        let vram = GpuVramInfo {
            total_bytes: 120 * BYTES_PER_GB,
            source: VramSource::ConfigOverride,
            unified: true,
        };
        let avail = available_for_training_bytes_with_meminfo_details(
            &vram,
            Some(16 * BYTES_PER_GB),
            Some(32 * BYTES_PER_GB),
        );

        assert_eq!(avail, 16 * BYTES_PER_GB - SAFETY_MARGIN_BYTES);
    }

    #[test]
    fn preflight_uses_host_memavailable_for_true_unified_memory() {
        let vram = GpuVramInfo {
            total_bytes: 24 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let avail = available_for_training_bytes_with_meminfo_details(
            &vram,
            Some(17 * BYTES_PER_GB),
            Some(32 * BYTES_PER_GB),
        );
        assert_eq!(avail, 17 * BYTES_PER_GB - SAFETY_MARGIN_BYTES);
    }

    #[test]
    fn preflight_rejects_oversized_payload_on_30gb_host() {
        // Long-context payload on the same 30 GB unified host: must
        // overflow even more dramatically than the repro.
        let cfg = qwen_4b();
        let vram = GpuVramInfo {
            total_bytes: 22 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(8 * BYTES_PER_GB));
        let est = estimate_step_working_set(
            &cfg,
            32768,
            16,
            4,
            WeightResidency::DualResidentCpuAndVulkan,
            true,
        );
        assert!(
            est.total_bytes > avail,
            "expected 32K-token Qwen3.5-4B step to overflow 22 GB unified budget; \
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
            unified: true,
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
            unified: false,
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
            unified: false,
        };
        let available = available_for_training_bytes_with_meminfo(&a6000_vram, None);
        let est =
            estimate_step_working_set(&cfg, 65_536, 16, 4, WeightResidency::SingleCopy, false);
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
    fn available_for_training_fails_closed_for_unknown_vram() {
        let none = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
            unified: false,
        };
        assert_eq!(available_for_training_bytes(&none), 0);
    }

    #[test]
    fn unified_memory_uses_meminfo_when_present() {
        let vram = GpuVramInfo {
            total_bytes: 25 * BYTES_PER_GB,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
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
            unified: true,
        };
        // 50 GB MemAvailable on a 10 GB ceiling — cap to ceiling.
        let avail = available_for_training_bytes_with_meminfo(&vram, Some(50 * BYTES_PER_GB));
        assert_eq!(avail, 10u64 * BYTES_PER_GB - SAFETY_MARGIN_BYTES);
    }

    #[test]
    fn unified_memory_without_meminfo_uses_conservative_static_budget() {
        let vram = GpuVramInfo {
            total_bytes: 10 * BYTES_PER_GB,
            source: VramSource::AppleSilicon,
            unified: true,
        };

        let avail = available_for_training_bytes_with_meminfo(&vram, None);

        assert_eq!(
            avail,
            10 * BYTES_PER_GB - (10 * BYTES_PER_GB) / 3 - SAFETY_MARGIN_BYTES
        );
    }

    #[test]
    fn format_oom_message_includes_actionable_knobs() {
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 8192, 16, 4, WeightResidency::SingleCopy, false);
        let msg = format_oom_message(&est, 8 * BYTES_PER_GB, 16, 4);
        assert!(msg.contains("Dynamic checkpointing already tried"));
        assert!(msg.contains("lora_rank"));
        assert!(msg.contains("optimizer state"));
        assert!(msg.contains("residency scratch"));
        assert!(!msg.contains("KILN_TRAINING_MEMORY_RESERVE_GB"));
    }

    #[test]
    fn format_oom_message_with_source_surfaces_unified_memory_signal() {
        let cfg = qwen_4b();
        let est = estimate_step_working_set(&cfg, 8192, 16, 4, WeightResidency::SingleCopy, false);
        // On a unified-memory APU the rejection message must call out
        // the corrected source so the operator knows the available
        // number is coming from live UMA memory, not static dGPU VRAM.
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
        assert!(!msg.contains("KILN_TRAINING_MEMORY_RESERVE_GB"));

        // None preserves the legacy message — no provenance clause.
        let no_src = format_oom_message_with_source(&est, 8 * BYTES_PER_GB, 16, 4, None);
        assert!(!no_src.contains("vram_source"));
    }
}

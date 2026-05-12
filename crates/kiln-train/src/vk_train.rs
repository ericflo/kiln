//! Vulkan-native training step.
//!
//! Self-contained training loop on `VkTensor` parameters: forward via
//! `vk_model_forward_loss`, backward via `vk_backward`, optimizer step
//! via the on-device AdamW kernel called directly with `VulkanBuffer`
//! handles (skipping the candle TensorId → registry indirection).
//!
//! For a full Qwen3.5 SFT run, the caller wires `VkModelWeights` from
//! their existing `GpuWeights` (one-time upload at training start) and
//! constructs `VkLoraLayer` per layer. The trainer then drives:
//!
//! ```text
//! for epoch:
//!   for batch (input_ids):
//!     loss = vk_model_forward_loss(weights, lora, ids)
//!     grads = vk_step_backward(loss)
//!     for (param_id, grad) in grads:
//!       lookup VkAdamWState by param_id
//!       dispatch_adamw_step_f32 in place
//! at end:
//!   for each lora pair: VkTensor.to_candle() → safetensors save
//! ```

use anyhow::{bail, Context, Result};
use candle_core::TensorId;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use kiln_model::vk_forward::{
    vk_count_gdn_layers, vk_model_forward_loss, vk_model_forward_loss_with_state, vk_step_backward,
    VkLayerWeights, VkLoraLayer, VkLoraPair, VkModelWeights,
};
use kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState;
use kiln_vulkan_kernel::kernels::dispatch_adamw_step_f32;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanBuffer, VulkanDevice};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::trainer::{tokenize_for_training, ProgressCallback, TrainingProgress};
use crate::{SftConfig, SftExample};

/// Per-parameter AdamW state held entirely on the GPU.
pub struct VkAdamWState {
    pub m: Arc<VulkanBuffer>,
    pub v: Arc<VulkanBuffer>,
    pub n_elements: usize,
}

impl VkAdamWState {
    pub fn zeros_for(device: &Arc<VulkanDevice>, n_elements: usize) -> Result<Self> {
        let bytes = (n_elements * 4).max(4) as u64;
        let m = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes,
        )
        .context("VkAdamWState: alloc m")?;
        let v = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes,
        )
        .context("VkAdamWState: alloc v")?;
        // Zero them via the existing fill shader.
        let zero_bytes: Vec<u8> = vec![0u8; n_elements * 4];
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &m,
            &zero_bytes,
        )?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &v,
            &zero_bytes,
        )?;
        Ok(Self {
            m: Arc::new(m),
            v: Arc::new(v),
            n_elements,
        })
    }
}

/// All AdamW state for a model — one entry per trainable param.
pub type VkAdamWBook = HashMap<TensorId, VkAdamWState>;

pub fn allocate_adamw_state(
    device: &Arc<VulkanDevice>,
    lora_layers: &[VkLoraLayer],
) -> Result<VkAdamWBook> {
    let mut book = HashMap::new();
    for layer in lora_layers {
        for proj in [
            layer.q_proj.as_ref(),
            layer.k_proj.as_ref(),
            layer.v_proj.as_ref(),
            layer.o_proj.as_ref(),
            layer.gate_proj.as_ref(),
            layer.up_proj.as_ref(),
            layer.down_proj.as_ref(),
        ]
        .iter()
        .flatten()
        {
            book.insert(proj.a_id, VkAdamWState::zeros_for(device, proj.a.num_elements())?);
            book.insert(proj.b_id, VkAdamWState::zeros_for(device, proj.b.num_elements())?);
        }
    }
    Ok(book)
}

#[derive(Clone, Copy, Debug)]
pub struct VkAdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for VkAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

fn lora_pairs<'a>(layers: &'a [VkLoraLayer]) -> impl Iterator<Item = &'a VkLoraPair> + 'a {
    layers.iter().flat_map(|l| {
        [
            l.q_proj.as_ref(),
            l.k_proj.as_ref(),
            l.v_proj.as_ref(),
            l.o_proj.as_ref(),
            l.gate_proj.as_ref(),
            l.up_proj.as_ref(),
            l.down_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
    })
}

/// Compute segment boundary layer indices.
///
/// Returns a Vec of length `num_segments` where each entry is the
/// **end index** (exclusive) of that segment. So segment 0 covers
/// layers `[0, boundaries[0])`, segment 1 covers `[boundaries[0],
/// boundaries[1])`, etc.
pub fn vk_compute_segment_boundaries(num_layers: usize, num_segments: usize) -> Vec<usize> {
    let n = num_segments.clamp(1, num_layers.max(1));
    let chunk = (num_layers + n - 1) / n;
    let mut out = Vec::with_capacity(n);
    let mut acc = chunk;
    for _ in 0..n {
        out.push(acc.min(num_layers));
        acc += chunk;
    }
    if let Some(last) = out.last_mut() {
        *last = num_layers;
    }
    out
}

/// Recommended number of gradient-checkpoint segments for a given
/// model layer count. Mirrors `trainer::compute_segment_boundaries`'s
/// default (clamps to 1..=num_layers).
pub fn vk_recommended_checkpoint_segments(num_layers: usize) -> usize {
    std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(1, num_layers))
        .unwrap_or_else(|| {
            // Same heuristic as the candle path's
            // kiln_core::vram::recommended_checkpoint_segments default.
            // 4 segments fits comfortably under 22 GB at T=918 for
            // Qwen3.5-4B if the per-segment intermediate budget is
            // ~5 GB.
            4.min(num_layers).max(1)
        })
}

/// Single-step training (no gradient checkpointing — full forward
/// tape held in GPU memory).
///
/// Returns the scalar loss value.
pub fn vk_train_step(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
) -> Result<f32> {
    vk_train_step_with_state(weights, lora_layers, input_ids, None, adamw_state, cfg, step)
}

/// Mint a synthetic candle TensorId — used to wrap a boundary
/// activation as a parameter leaf so its gradient can be captured
/// from a sub-tape.
fn mint_fresh_tensor_id() -> Result<TensorId> {
    use candle_core::{Device, Tensor, Var};
    let dummy = Tensor::from_vec(vec![0.0_f32], (1,), &Device::Cpu)?;
    Ok(Var::from_tensor(&dummy)?.id())
}

/// Gradient-checkpointed training step.
///
/// Forward through layers in `num_segments` chunks, only saving boundary
/// states (peak memory ≈ one segment's intermediates). Backward in
/// reverse: rebuild forward + backward per segment, capture the
/// boundary-input gradient via the scalar trick (seg_loss = sum(seg_out
/// · upstream_grad)), use that as upstream for the next-earlier
/// segment.
///
/// Does NOT yet support GDN layers (they need state plumbing through
/// the recompute path; the recurrent state would have to be snapshot
/// per segment).
#[allow(clippy::too_many_arguments)]
pub fn vk_checkpointed_train_step(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
    num_segments: usize,
) -> Result<f32> {
    use kiln_model::vk_forward::{
        vk_compute_rope_tables, vk_full_attention_layer_with_rope, VkLayerWeights,
    };
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
    use kiln_vulkan_kernel::vk_ops::embedding::{
        upload_u32_ids, vk_embedding_lookup_bf16, vk_embedding_lookup_f32,
    };
    use kiln_vulkan_kernel::vk_ops::flce::{vk_flce_loss, FLCE_DEFAULT_CHUNK};
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
    use kiln_vulkan_kernel::VkTensor;

    // GDN unsupported in this path for v1 (would need state snapshot per segment)
    for layer in &weights.layers {
        if !matches!(layer, VkLayerWeights::FullAttention(_)) {
            anyhow::bail!(
                "vk_checkpointed_train_step: GDN layers not yet supported in checkpointed path \
                 (use vk_train_step for hybrid models — checkpointing for hybrid models is a \
                 follow-up that needs per-segment state snapshots)"
            );
        }
    }

    let num_layers = weights.layers.len();
    let segments = vk_compute_segment_boundaries(num_layers, num_segments);
    let device = weights.embed_tokens.device();

    // Precompute RoPE tables once
    let t_rope = input_ids.len();
    let rope_tables = if !weights.rotary_inv_freq.is_empty() && weights.rotary_dim > 0 {
        Some(vk_compute_rope_tables(
            device,
            &weights.rotary_inv_freq,
            t_rope,
        )?)
    } else {
        None
    };

    // Phase A: segmented forward, save detached boundary inputs
    let ids = upload_u32_ids(device, input_ids)?;
    let h_init = match weights.embed_dtype {
        VkDType::F32 => {
            vk_embedding_lookup_f32(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)?
        }
        VkDType::Bf16 => {
            vk_embedding_lookup_bf16(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)?
        }
    };
    // boundaries[k] = activations entering segment k (boundaries[0] = embedding output)
    let mut boundaries: Vec<VkTensor> = vec![h_init.detach()];
    let mut h = h_init;

    for (seg_idx, &end_layer) in segments.iter().enumerate() {
        let start_layer = if seg_idx == 0 { 0 } else { segments[seg_idx - 1] };
        for layer_idx in start_layer..end_layer {
            match &weights.layers[layer_idx] {
                VkLayerWeights::FullAttention(full) => {
                    let rope_arg = rope_tables
                        .as_ref()
                        .map(|(c, s)| (c, s, weights.rotary_dim));
                    h = vk_full_attention_layer_with_rope(
                        &h,
                        full,
                        &lora_layers[layer_idx],
                        rope_arg,
                    )?;
                }
                VkLayerWeights::LinearAttention(_) => unreachable!(),
            }
        }
        if seg_idx + 1 < segments.len() {
            // Detach: drops the prior segment's tape via Arc refcount
            h = h.detach();
            boundaries.push(h.clone());
        }
    }
    // h still holds the last segment's tape, ready for loss + backward
    let final_h = h;

    // Phase B: compute loss + backward last segment
    let h_norm = vk_rmsnorm(&final_h, &weights.final_norm_weight, 1e-5)?;
    let t_in = input_ids.len();
    let mut labels: Vec<u32> = input_ids[1..].to_vec();
    while labels.len() < t_in {
        labels.push((weights.vocab.saturating_sub(1)) as u32);
    }
    let loss = vk_flce_loss(&h_norm, &weights.lm_head, &labels, FLCE_DEFAULT_CHUNK)?;
    let loss_val = loss.to_vec_f32()?[0];

    // For the LAST segment we wrap its boundary input as a parameter
    // leaf so we can grab the upstream gradient. But the loss tape
    // doesn't pass through that wrapping — it passes through `final_h`.
    // Workaround: the grad returned by vk_backward contains LoRA grads
    // for params in the last segment. To get grad-at-boundary for
    // segment k-1, we'd need to back-propagate INTO the boundary tensor.
    //
    // Trick: re-wrap last-segment forward starting from a fresh leaf,
    // following the same upstream-grad pattern as the other segments.
    // That avoids special-casing the last segment.

    // Accumulate grads into a shared store
    use std::collections::HashMap;
    let mut shared_grads: HashMap<TensorId, VkTensor> = HashMap::new();

    // Backward LAST segment first (using loss directly).
    // Wrap boundaries[N-1] as a parameter leaf; rebuild forward; backward(loss).
    let mut upstream_grad: Option<VkTensor> = {
        let last_boundary = boundaries.last().unwrap();
        let last_id = mint_fresh_tensor_id()?;
        let h_param = VkTensor::parameter(
            Arc::clone(last_boundary.buffer()),
            last_boundary.shape().to_vec(),
            last_boundary.dtype(),
            Arc::clone(last_boundary.device()),
            last_id,
        );
        // Recompute last segment with autograd, using h_param as the boundary
        let last_seg = segments.len() - 1;
        let start_layer = if last_seg == 0 { 0 } else { segments[last_seg - 1] };
        let end_layer = segments[last_seg];
        let mut h = h_param;
        for layer_idx in start_layer..end_layer {
            match &weights.layers[layer_idx] {
                VkLayerWeights::FullAttention(full) => {
                    let rope_arg = rope_tables
                        .as_ref()
                        .map(|(c, s)| (c, s, weights.rotary_dim));
                    h = vk_full_attention_layer_with_rope(
                        &h,
                        full,
                        &lora_layers[layer_idx],
                        rope_arg,
                    )?;
                }
                VkLayerWeights::LinearAttention(_) => unreachable!(),
            }
        }
        let h_norm2 = vk_rmsnorm(&h, &weights.final_norm_weight, 1e-5)?;
        let loss2 = vk_flce_loss(&h_norm2, &weights.lm_head, &labels, FLCE_DEFAULT_CHUNK)?;
        let grads = vk_backward(&loss2)?;
        let upstream = grads.get(last_id).cloned();
        // Accumulate non-boundary grads
        for (pid, g) in grads.iter() {
            if *pid != last_id {
                shared_grads.insert(*pid, g.clone());
            }
        }
        upstream
    };

    // Earlier segments in reverse
    for seg_idx in (0..segments.len() - 1).rev() {
        let upstream = upstream_grad
            .clone()
            .ok_or_else(|| anyhow::anyhow!("checkpoint: missing upstream grad at seg {seg_idx}"))?;
        let boundary = &boundaries[seg_idx];
        let boundary_id = mint_fresh_tensor_id()?;
        let h_param = VkTensor::parameter(
            Arc::clone(boundary.buffer()),
            boundary.shape().to_vec(),
            boundary.dtype(),
            Arc::clone(boundary.device()),
            boundary_id,
        );
        let start_layer = if seg_idx == 0 { 0 } else { segments[seg_idx - 1] };
        let end_layer = segments[seg_idx];
        let mut h = h_param;
        for layer_idx in start_layer..end_layer {
            match &weights.layers[layer_idx] {
                VkLayerWeights::FullAttention(full) => {
                    let rope_arg = rope_tables
                        .as_ref()
                        .map(|(c, s)| (c, s, weights.rotary_dim));
                    h = vk_full_attention_layer_with_rope(
                        &h,
                        full,
                        &lora_layers[layer_idx],
                        rope_arg,
                    )?;
                }
                VkLayerWeights::LinearAttention(_) => unreachable!(),
            }
        }
        // Scalar trick: scalar = sum(h * upstream)
        let prod = vk_mul(&h, &upstream)?;
        let scalar = vk_sum_all(&prod)?;
        let grads = vk_backward(&scalar)?;
        upstream_grad = grads.get(boundary_id).cloned();
        for (pid, g) in grads.iter() {
            if *pid != boundary_id {
                // Accumulate (sum) — same param may already be in store from later seg
                if let Some(existing) = shared_grads.get(pid).cloned() {
                    let summed = kiln_vulkan_kernel::vk_ops::elementwise::vk_add_no_grad(
                        &existing, g,
                    )?;
                    shared_grads.insert(*pid, summed);
                } else {
                    shared_grads.insert(*pid, g.clone());
                }
            }
        }
    }

    // Optimizer step from accumulated grads
    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = shared_grads.get(&pid) else {
                continue;
            };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "vk_checkpointed_train_step: AdamW F32 only"
            );
            let s = adamw_state
                .get(&pid)
                .with_context(|| format!("missing AdamW state for {:?}", pid))?;
            dispatch_adamw_step_f32(
                weights.embed_tokens.device(),
                param.buffer(),
                grad.buffer(),
                &s.m,
                &s.v,
                param.num_elements(),
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                step,
            )
            .context("dispatch_adamw_step_f32 (checkpointed)")?;
        }
    }

    Ok(loss_val)
}

/// Same as `vk_train_step` but threads optional GDN state. For
/// hybrid models (Qwen3.5-4B), pass a freshly-zeroed
/// `VkLinearAttentionState`. The state is mutated in place and can
/// be discarded after the step (training treats each example as
/// starting from zero state).
#[allow(clippy::too_many_arguments)]
pub fn vk_train_step_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    gdn_state: Option<&mut VkLinearAttentionState>,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
) -> Result<f32> {
    let loss = vk_model_forward_loss_with_state(weights, lora_layers, input_ids, gdn_state)?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = vk_step_backward(&loss)?;

    // Dispatch AdamW per parameter. We assume F32 storage; BF16
    // variant just swaps the kernel name.
    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = grads.get(pid) else { continue };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "vk_train_step: AdamW F32 only for Phase F (got {:?}/{:?})",
                param.dtype(),
                grad.dtype()
            );
            anyhow::ensure!(
                param.num_elements() == grad.num_elements(),
                "vk_train_step: param/grad element-count mismatch"
            );
            let state = adamw_state
                .get(&pid)
                .with_context(|| format!("missing AdamW state for param {:?}", pid))?;
            anyhow::ensure!(
                state.n_elements == param.num_elements(),
                "AdamW state size mismatch"
            );
            dispatch_adamw_step_f32(
                weights.embed_tokens.device(),
                param.buffer(),
                grad.buffer(),
                &state.m,
                &state.v,
                param.num_elements(),
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                step,
            )
            .context("dispatch_adamw_step_f32")?;
        }
    }

    Ok(loss_val)
}

// ---------------------------------------------------------------------------
// LoRA initialization (one VkLoraLayer per model layer)
// ---------------------------------------------------------------------------

/// Initialize LoRA params for every layer in the model.
///
/// Targets the canonical SFT modules (q/k/v/o + gate/up/down) on
/// FullAttention layers. LinearAttention (GDN) layers currently get
/// empty LoRA — Phase 5 will populate `in_proj_qkv` / `in_proj_z` /
/// `gdn_out_proj`. Callers using a pure-FullAttn model (e.g.
/// synthetic test or a non-hybrid Qwen variant) get full LoRA
/// coverage; hybrid Qwen3.5-4B will train only the 8 FullAttn layers
/// until Phase 5.
pub fn vk_init_lora_layers(
    device: &Arc<VulkanDevice>,
    model_weights: &VkModelWeights,
    model_config: &ModelConfig,
    rank: usize,
    alpha: f32,
    seed: u64,
) -> Result<Vec<VkLoraLayer>> {
    let hidden = model_config.hidden_size;
    let head_dim = model_config.head_dim;
    let q_dim = model_config.num_attention_heads * head_dim;
    let kv_dim = model_config.num_kv_heads * head_dim;
    let q_out_dim = if model_config.attn_output_gate {
        q_dim * 2
    } else {
        q_dim
    };
    let intermediate = model_config.intermediate_size;

    let mut out = Vec::with_capacity(model_weights.layers.len());
    for (li, layer) in model_weights.layers.iter().enumerate() {
        let mk = |idx: usize, in_features: usize, out_features: usize| -> Result<VkLoraPair> {
            // Combine seed with layer + module index for deterministic init
            let s = seed
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add((li as u64).wrapping_mul(7))
                .wrapping_add(idx as u64);
            VkLoraPair::init_kaiming(device, in_features, out_features, rank, alpha, s)
        };
        match layer {
            VkLayerWeights::FullAttention(_) => {
                out.push(VkLoraLayer {
                    q_proj: Some(mk(1, hidden, q_out_dim)?),
                    k_proj: Some(mk(2, hidden, kv_dim)?),
                    v_proj: Some(mk(3, hidden, kv_dim)?),
                    o_proj: Some(mk(4, q_dim, hidden)?),
                    gate_proj: Some(mk(5, hidden, intermediate)?),
                    up_proj: Some(mk(6, hidden, intermediate)?),
                    down_proj: Some(mk(7, intermediate, hidden)?),
                    ..Default::default()
                });
            }
            VkLayerWeights::LinearAttention(_) => {
                // Phase 5 will populate in_proj_qkv / in_proj_z /
                // gdn_out_proj LoRA. For now GDN layers get no LoRA —
                // they still forward (once Phase 5 wires it) but
                // train no params.
                out.push(VkLoraLayer::default());
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// vk-native SFT trainer (multi-epoch, single-step optimizer)
// ---------------------------------------------------------------------------

/// Vulkan-native SFT training loop.
///
/// Mirrors `trainer::sft_train()`'s shape (same args, same callback
/// contract, same on-disk adapter format) but executes the entire
/// per-step forward → backward → AdamW chain on GPU buffers via
/// `VkTensor` + `vk_backward` + `dispatch_adamw_step_f32`. No candle
/// `Var` registry indirection, no `CpuStorage` intermediates.
///
/// Hybrid Qwen3.5-4B currently bails because the GDN layer arm of
/// `vk_transformer_layer` is not yet implemented (Phase 5 wires it).
/// All-FullAttn models train end-to-end.
pub fn vk_native_sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    // Probe Vulkan device — if absent, refuse loud rather than silently
    // falling back to candle (the env flag explicitly asked for
    // vk-native; the user wants to know when it's not happening).
    if !VulkanDevice::probe() {
        bail!(
            "vk_native_sft_train: no Vulkan device available — \
             unset KILN_VK_NATIVE_TRAINING to use the candle path"
        );
    }
    let vk_device = Arc::new(
        VulkanDevice::new().context("vk_native_sft_train: failed to create Vulkan device")?,
    );

    tracing::info!(
        num_examples = examples.len(),
        epochs = config.epochs,
        lr = config.learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting vk-native SFT training"
    );

    let effective_seed = config.seed.unwrap_or_else(|| {
        // Same fallback as sft_train: a deterministic-enough default.
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdeadbeef)
    });

    // Upload candle GpuWeights → vk-native VkModelWeights (one-time).
    let vk_weights = VkModelWeights::from_gpu_weights(weights, model_config, &vk_device)
        .context("vk_native_sft_train: VkModelWeights::from_gpu_weights")?;

    // Initialize LoRA params for each layer.
    let lora_layers = vk_init_lora_layers(
        &vk_device,
        &vk_weights,
        model_config,
        config.lora_rank,
        config.lora_alpha,
        effective_seed,
    )?;

    // Allocate AdamW state per LoRA pair.
    let mut adamw = allocate_adamw_state(&vk_device, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: config.learning_rate as f32,
        ..Default::default()
    };

    // Tokenize all examples up front.
    let tokenized: Vec<(Vec<u32>, Vec<bool>)> = examples
        .iter()
        .filter_map(|ex| match tokenize_for_training(ex, tokenizer) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("vk_native: skipping example: {e}");
                None
            }
        })
        .collect();
    if tokenized.is_empty() {
        bail!("vk_native_sft_train: no valid training examples after tokenization");
    }

    let total_steps = config.epochs * tokenized.len();
    let mut global_step: u32 = 0;
    let mut last_loss = 0.0f32;

    // Pre-compute GDN state shape from model_config for per-example
    // allocation. For hybrid Qwen3.5-4B this allocates state for the
    // 24 GDN layers.
    let num_gdn_layers = vk_count_gdn_layers(&vk_weights);
    let needs_state = num_gdn_layers > 0;
    let conv_kernel = model_config.linear_conv_kernel_dim;

    // Pick gradient-checkpoint segments. KILN_GRAD_CHECKPOINT_SEGMENTS
    // env var (matches the candle path's convention) overrides the
    // default 4-segment heuristic. Set to 1 to disable
    // checkpointing entirely.
    //
    // The checkpointed path is FullAttn-only for now; hybrid models
    // (with GDN layers) fall back to vk_train_step.
    let ckpt_segments = if needs_state || std::env::var("KILN_NO_GRAD_CHECKPOINT").is_ok() {
        None
    } else {
        let segs = vk_recommended_checkpoint_segments(model_config.num_layers);
        if segs > 1 {
            Some(segs)
        } else {
            None
        }
    };
    if let Some(segs) = ckpt_segments {
        tracing::info!(
            num_segments = segs,
            "vk-native gradient checkpointing enabled"
        );
    } else {
        tracing::info!("vk-native gradient checkpointing disabled");
    }

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        for (input_ids, _label_mask) in tokenized.iter() {
            global_step += 1;

            // Fresh GDN state per example (training is short-context;
            // we don't carry state across examples).
            let mut maybe_state = if needs_state {
                let conv_channels = 2 * model_config.linear_num_key_heads * model_config.linear_key_head_dim
                    + model_config.linear_num_value_heads * model_config.linear_value_head_dim;
                let state = VkLinearAttentionState::zeros(
                    &vk_device,
                    num_gdn_layers,
                    1, // batch
                    model_config.linear_num_value_heads,
                    model_config.linear_key_head_dim,
                    model_config.linear_value_head_dim,
                    conv_channels,
                    conv_kernel,
                )?;
                Some(state)
            } else {
                None
            };

            // For Phase 1 we ignore label_mask and run FLCE on all
            // positions (matches the synthetic smoke tests). Real SFT
            // training masks prompt tokens; that's a Phase 1.7 follow-up
            // — vk_index_select_rows already exists for the gather.
            let loss = if let Some(segs) = ckpt_segments {
                // Checkpointed path (FullAttn-only)
                vk_checkpointed_train_step(
                    &vk_weights,
                    &lora_layers,
                    input_ids,
                    &mut adamw,
                    &cfg,
                    global_step,
                    segs,
                )
                .with_context(|| {
                    format!(
                        "vk_checkpointed_train_step at epoch {} step {}",
                        epoch + 1,
                        global_step
                    )
                })?
            } else {
                vk_train_step_with_state(
                    &vk_weights,
                    &lora_layers,
                    input_ids,
                    maybe_state.as_mut(),
                    &mut adamw,
                    &cfg,
                    global_step,
                )
                .with_context(|| {
                    format!(
                        "vk_train_step at epoch {} step {}",
                        epoch + 1,
                        global_step
                    )
                })?
            };

            anyhow::ensure!(
                loss.is_finite(),
                "vk_native_sft_train: non-finite loss {loss} at step {global_step}"
            );
            epoch_loss += loss;
            last_loss = loss;

            // Periodic checkpoint
            if let Some(interval) = config.checkpoint_interval {
                if interval > 0
                    && (global_step as usize) % interval == 0
                    && (global_step as usize) < total_steps
                {
                    let ckpt_dir =
                        adapter_dir.join(format!("{adapter_name}-checkpoint-{global_step}"));
                    if let Err(e) = std::fs::create_dir_all(&ckpt_dir) {
                        tracing::warn!(error = %e, "create checkpoint dir failed");
                    } else {
                        let ckpt_path = ckpt_dir.join("adapter_model.safetensors");
                        if let Err(e) = save_vk_lora_adapter(
                            &lora_layers,
                            config.lora_rank,
                            config.lora_alpha,
                            &ckpt_path,
                        ) {
                            tracing::warn!(error = %e, "save checkpoint failed");
                        } else {
                            tracing::info!(
                                step = global_step,
                                path = %ckpt_path.display(),
                                "saved vk-native training checkpoint"
                            );
                        }
                    }
                }
            }

            if let Some(ref cb) = progress_cb {
                cb(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.epochs,
                    step: global_step as usize,
                    total_steps,
                    loss: loss as f64,
                    progress: (global_step as f32) / (total_steps as f32),
                });
            }

            if (global_step as usize) % 10 == 0 || (global_step as usize) == total_steps {
                tracing::info!(
                    epoch = epoch + 1,
                    step = global_step,
                    total_steps,
                    loss = format!("{loss:.6}"),
                    "vk-native training step"
                );
            }
        }
        let avg = epoch_loss / (tokenized.len() as f32);
        tracing::info!(
            epoch = epoch + 1,
            avg_loss = format!("{avg:.6}"),
            "vk-native epoch complete"
        );
    }

    // Final adapter save
    let output_dir = adapter_dir.join(adapter_name);
    std::fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "vk_native_sft_train: create adapter dir {}",
            output_dir.display()
        )
    })?;
    let adapter_path = output_dir.join("adapter_model.safetensors");
    save_vk_lora_adapter(&lora_layers, config.lora_rank, config.lora_alpha, &adapter_path)
        .with_context(|| format!("save final adapter to {}", adapter_path.display()))?;

    // Write a minimal adapter_config.json mirroring trainer::save_peft.
    write_vk_adapter_config(
        &output_dir,
        config.lora_rank,
        config.lora_alpha,
    )?;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        "vk-native SFT training complete"
    );

    Ok(output_dir)
}

fn write_vk_adapter_config(output_dir: &Path, rank: usize, alpha: f32) -> Result<()> {
    let cfg = serde_json::json!({
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": null,
        "peft_type": "LORA",
        "r": rank,
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    });
    let path = output_dir.join("adapter_config.json");
    let s = serde_json::to_string_pretty(&cfg)
        .with_context(|| format!("serialize adapter_config.json {}", path.display()))?;
    std::fs::write(&path, s)
        .with_context(|| format!("write adapter_config.json {}", path.display()))?;
    Ok(())
}

/// Save LoRA adapter to safetensors via candle. Each VkTensor is read
/// back to CPU once.
pub fn save_vk_lora_adapter(
    lora_layers: &[VkLoraLayer],
    rank: usize,
    alpha: f32,
    output_path: &std::path::Path,
) -> Result<()> {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (li, layer) in lora_layers.iter().enumerate() {
        for (name, proj) in [
            ("q_proj", layer.q_proj.as_ref()),
            ("k_proj", layer.k_proj.as_ref()),
            ("v_proj", layer.v_proj.as_ref()),
            ("o_proj", layer.o_proj.as_ref()),
            ("gate_proj", layer.gate_proj.as_ref()),
            ("up_proj", layer.up_proj.as_ref()),
            ("down_proj", layer.down_proj.as_ref()),
        ] {
            let Some(p) = proj else { continue };
            let a_t = p.a.to_candle()?.to_device(&Device::Cpu)?;
            let b_t = p.b.to_candle()?.to_device(&Device::Cpu)?;
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_A.weight",
                    li, name
                ),
                a_t,
            );
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_B.weight",
                    li, name
                ),
                b_t,
            );
        }
    }
    candle_core::safetensors::save(&tensors, output_path)
        .with_context(|| format!("save_vk_lora_adapter: {}", output_path.display()))?;
    let _ = (rank, alpha); // adapter_config.json could be written here if desired
    Ok(())
}

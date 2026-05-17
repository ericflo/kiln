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
//!   for each lora pair: VkTensor readback → direct safetensors save
//! ```

use anyhow::{Context, Result, bail};
use candle_core::TensorId;
use kiln_core::config::ModelConfig;
use kiln_core::env_flag::env_flag;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use kiln_model::vk_forward::{
    VkGrpoReferencePrefix, VkLayerWeights, VkLinearAttentionWeights, VkLoraLayer, VkLoraPair,
    VkModelWeights, vk_count_gdn_layers, vk_grpo_reference_log_probs_from_prefix,
    vk_grpo_reference_log_probs_full_sequence, vk_grpo_reference_prefill_prompt,
    vk_linear_with_lora, vk_model_forward_final_norm_with_state,
    vk_model_forward_loss_masked_with_state, vk_model_forward_loss_with_state, vk_step_backward,
};
use kiln_vulkan_kernel::kernels::{dispatch_adamw_step_f32, dispatch_sgd_step_f32};
use kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState;
use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanBuffer, VulkanDevice};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::trainer::{ProgressCallback, TrainingProgress, tokenize_for_training};
use crate::{
    AdvantageMode, GrpoConfig, GrpoGroup, KlEstimator, LossAggregation, Optimizer, SftConfig,
    SftExample,
};

struct TokenizedSftExample {
    input_ids: Vec<u32>,
    label_mask: Vec<bool>,
    original_index: usize,
}

struct TokenizedVkGrpoCompletion {
    input_ids: Vec<u32>,
    completion_mask: Vec<bool>,
}

struct TokenizedVkGrpoGroup {
    prompt_ids: Vec<u32>,
    completions: Vec<TokenizedVkGrpoCompletion>,
    rewards: Vec<f64>,
}

fn to_core_messages(msgs: &[crate::ChatMessage]) -> Vec<kiln_core::tokenizer::ChatMessage> {
    msgs.iter()
        .map(|m| kiln_core::tokenizer::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            ..Default::default()
        })
        .collect()
}

fn tokenize_vk_grpo_group(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
) -> Result<TokenizedVkGrpoGroup> {
    if group.completions.is_empty() {
        bail!("GRPO group has no completions");
    }

    let prompt_messages = to_core_messages(&group.messages);
    let prompt_text = tokenizer
        .apply_chat_template(&prompt_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rewards = Vec::with_capacity(group.completions.len());
    let mut full_message_batches = Vec::with_capacity(group.completions.len());
    for scored in &group.completions {
        let mut full_messages = prompt_messages.clone();
        full_messages.push(kiln_core::tokenizer::ChatMessage {
            role: "assistant".to_string(),
            content: scored.text.clone(),
            ..Default::default()
        });
        full_message_batches.push(full_messages);
        rewards.push(scored.reward);
    }

    let full_texts = tokenizer
        .apply_chat_template_batch(&full_message_batches)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let full_id_batches = tokenizer
        .encode_batch(&full_texts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut completions = Vec::with_capacity(full_id_batches.len());
    for full_ids in full_id_batches {
        if full_ids.len() < 2 {
            bail!("GRPO completion tokenized to fewer than 2 tokens");
        }
        anyhow::ensure!(
            full_ids.len() >= prompt_ids.len() && full_ids[..prompt_ids.len()] == prompt_ids[..],
            "GRPO completion tokenization did not preserve prompt prefix"
        );
        let mut mask = vec![false; full_ids.len()];
        for slot in mask.iter_mut().skip(prompt_ids.len()) {
            *slot = true;
        }
        completions.push(TokenizedVkGrpoCompletion {
            input_ids: full_ids,
            completion_mask: mask,
        });
    }

    Ok(TokenizedVkGrpoGroup {
        prompt_ids,
        completions,
        rewards,
    })
}

/// Effective KL coefficient applied by the vk-native shader path.
///
/// The shader implements the K1 estimator (`+kl_coeff * log_ratio` per token).
/// `KlEstimator::None` is realized by passing `0.0` so the shader's KL term
/// vanishes; `KlEstimator::K1` passes the raw coefficient; `KlEstimator::K3`
/// is rejected earlier in `vk_native_grpo_train*`.
fn vk_effective_kl_coeff(config: &GrpoConfig) -> f64 {
    match config.kl_estimator {
        KlEstimator::None => 0.0,
        KlEstimator::K1 | KlEstimator::K3 => config.kl_coeff,
    }
}

/// Mirrors `trainer::is_degenerate_grpo_group` for the vk-native path. A
/// group whose completions all share the same reward produces a uniformly
/// zero advantage vector under any AdvantageMode and is dropped by
/// Dynamic Sampling (DAPO, arXiv:2503.14476).
fn is_degenerate_vk_grpo_group(group: &GrpoGroup) -> bool {
    let mut rewards = group.completions.iter().map(|c| c.reward);
    let Some(first) = rewards.next() else {
        return true;
    };
    rewards.all(|r| r == first)
}

fn compute_vk_grpo_advantages(rewards: &[f64], mode: AdvantageMode) -> Vec<f64> {
    let n = rewards.len() as f64;
    if n <= 1.0 {
        return vec![0.0; rewards.len()];
    }
    let mean = rewards.iter().sum::<f64>() / n;
    let centered: Vec<f64> = rewards.iter().map(|r| r - mean).collect();
    match mode {
        AdvantageMode::DrGrpo => centered,
        AdvantageMode::Vanilla => {
            let var = centered.iter().map(|c| c * c).sum::<f64>() / n;
            let std = var.sqrt();
            centered.into_iter().map(|c| c / (std + 1e-8)).collect()
        }
    }
}

fn grpo_active_rows_and_labels(
    input_ids: &[u32],
    completion_mask: &[bool],
) -> Result<(Vec<u32>, Vec<u32>)> {
    anyhow::ensure!(
        input_ids.len() == completion_mask.len(),
        "GRPO completion mask length {} != input length {}",
        completion_mask.len(),
        input_ids.len()
    );
    anyhow::ensure!(
        input_ids.len() >= 2,
        "GRPO completion needs at least 2 tokens"
    );
    let active_rows: Vec<u32> = completion_mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(idx, &active)| active.then_some(idx as u32))
        .collect();
    anyhow::ensure!(
        !active_rows.is_empty(),
        "GRPO completion has no active completion tokens"
    );
    let labels = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();
    Ok((active_rows, labels))
}

fn ensure_grpo_completion_scoring_layout(prompt_len: usize, active_rows: &[u32]) -> Result<()> {
    anyhow::ensure!(
        prompt_len > 0,
        "GRPO completion scoring requires at least one prompt token"
    );
    anyhow::ensure!(
        !active_rows.is_empty(),
        "GRPO completion scoring requires active rows"
    );
    let first = (prompt_len - 1) as u32;
    for (idx, &row) in active_rows.iter().enumerate() {
        anyhow::ensure!(
            row == first + idx as u32,
            "GRPO active rows are not a contiguous completion continuation from prompt: \
             prompt_len={prompt_len}, idx={idx}, row={row}, expected={}",
            first + idx as u32
        );
    }
    Ok(())
}

const DEFAULT_GRPO_PREFIX_REFERENCE_MAX_DECODE_TOKENS: usize = 8;
const LONG_PROMPT_REFERENCE_MIN_TOKENS: usize = 4096;
const LONG_PROMPT_PREFIX_REFERENCE_MAX_DECODE_TOKENS: usize = 256;

fn grpo_prefix_reference_max_decode_tokens() -> usize {
    std::env::var("KILN_VK_GRPO_PREFIX_REFERENCE_MAX_DECODE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GRPO_PREFIX_REFERENCE_MAX_DECODE_TOKENS)
}

fn grpo_long_prompt_prefix_reference_max_decode_tokens() -> usize {
    std::env::var("KILN_VK_GRPO_LONG_PROMPT_PREFIX_REFERENCE_MAX_DECODE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(LONG_PROMPT_PREFIX_REFERENCE_MAX_DECODE_TOKENS)
}

fn grpo_use_prefix_reference(
    prompt_len: usize,
    active_label_count: usize,
    completions_in_group: usize,
) -> bool {
    let decode_steps = active_label_count.saturating_sub(1);
    if decode_steps <= grpo_prefix_reference_max_decode_tokens() {
        return true;
    }

    // Full-sequence reference scoring replays the whole prompt once per
    // completion. For long-prompt GRPO groups, prefer a shared prompt prefix
    // state for moderate completion tails so the reference pass does not
    // duplicate tens of thousands of prompt tokens across completions.
    prompt_len >= LONG_PROMPT_REFERENCE_MIN_TOKENS
        && completions_in_group > 1
        && decode_steps <= grpo_long_prompt_prefix_reference_max_decode_tokens()
}

fn grpo_group_needs_prefix_reference(group: &TokenizedVkGrpoGroup) -> bool {
    let completions_in_group = group.completions.len();
    group.completions.iter().any(|completion| {
        let active_labels = completion
            .completion_mask
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .filter(|&&active| active)
            .count();
        grpo_use_prefix_reference(group.prompt_ids.len(), active_labels, completions_in_group)
    })
}

#[cfg(test)]
mod reference_path_tests {
    use super::*;

    #[test]
    fn short_completion_tail_uses_prefix_reference() {
        assert!(grpo_use_prefix_reference(128, 9, 1));
    }

    #[test]
    fn long_prompt_group_keeps_moderate_tail_on_prefix_reference() {
        assert!(grpo_use_prefix_reference(22_700, 22, 2));
    }

    #[test]
    fn long_completion_tail_uses_full_sequence_reference() {
        assert!(!grpo_use_prefix_reference(22_700, 300, 2));
    }
}

#[allow(clippy::too_many_arguments)]
fn vk_grpo_reference_log_probs_dynamic(
    weights: &VkModelWeights,
    ref_prefix: Option<&VkGrpoReferencePrefix>,
    input_ids: &[u32],
    active_rows: &[u32],
    labels: &[u32],
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    prompt_len: usize,
    completions_in_group: usize,
) -> Result<(VkTensor, &'static str)> {
    if grpo_use_prefix_reference(prompt_len, labels.len(), completions_in_group) {
        let prefix = ref_prefix.ok_or_else(|| {
            anyhow::anyhow!("vk-native GRPO prefix reference path selected without prompt prefix")
        })?;
        let log_probs = vk_grpo_reference_log_probs_from_prefix(weights, prefix, labels)?;
        Ok((log_probs, "prefix_decode"))
    } else {
        let log_probs = vk_grpo_reference_log_probs_full_sequence(
            weights,
            input_ids,
            active_rows,
            labels,
            model_config,
            num_gdn_layers,
        )?;
        Ok((log_probs, "full_sequence"))
    }
}

pub fn validate_vk_grpo_seq_lens(
    seq_lens: &[usize],
    max_position_embeddings: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        max_position_embeddings > 0,
        "{context}: model max_position_embeddings must be positive"
    );
    let Some(max_seq_len) = seq_lens.iter().copied().max() else {
        anyhow::bail!("{context}: no GRPO sequence lengths to validate");
    };
    anyhow::ensure!(
        max_seq_len <= max_position_embeddings,
        "{context}: tokenized GRPO sequence length {max_seq_len} exceeds model max_position_embeddings {max_position_embeddings}; shorten or split the offending group"
    );
    Ok(())
}

fn validate_vk_grpo_tokenized_group_context(
    group: &TokenizedVkGrpoGroup,
    model_config: &ModelConfig,
    context: &str,
) -> Result<()> {
    let seq_lens = group
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .collect::<Vec<_>>();
    validate_vk_grpo_seq_lens(&seq_lens, model_config.max_position_embeddings, context)
}

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
            layer.in_proj_qkv.as_ref(),
            layer.in_proj_z.as_ref(),
            layer.gdn_out_proj.as_ref(),
        ]
        .iter()
        .flatten()
        {
            book.insert(
                proj.a_id,
                VkAdamWState::zeros_for(device, proj.a.num_elements())?,
            );
            book.insert(
                proj.b_id,
                VkAdamWState::zeros_for(device, proj.b.num_elements())?,
            );
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
            l.in_proj_qkv.as_ref(),
            l.in_proj_z.as_ref(),
            l.gdn_out_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
    })
}

fn detach_lora_pair(pair: &VkLoraPair) -> VkLoraPair {
    VkLoraPair {
        a: pair.a.detach(),
        b: pair.b.detach(),
        a_id: pair.a_id,
        b_id: pair.b_id,
        scale: pair.scale,
    }
}

fn detach_lora_layers(layers: &[VkLoraLayer]) -> Vec<VkLoraLayer> {
    layers
        .iter()
        .map(|l| VkLoraLayer {
            q_proj: l.q_proj.as_ref().map(detach_lora_pair),
            k_proj: l.k_proj.as_ref().map(detach_lora_pair),
            v_proj: l.v_proj.as_ref().map(detach_lora_pair),
            o_proj: l.o_proj.as_ref().map(detach_lora_pair),
            gate_proj: l.gate_proj.as_ref().map(detach_lora_pair),
            up_proj: l.up_proj.as_ref().map(detach_lora_pair),
            down_proj: l.down_proj.as_ref().map(detach_lora_pair),
            in_proj_qkv: l.in_proj_qkv.as_ref().map(detach_lora_pair),
            in_proj_z: l.in_proj_z.as_ref().map(detach_lora_pair),
            gdn_out_proj: l.gdn_out_proj.as_ref().map(detach_lora_pair),
        })
        .collect()
}

fn fresh_gdn_state(
    device: &Arc<VulkanDevice>,
    model_config: &ModelConfig,
    num_gdn_layers: usize,
) -> Result<Option<VkLinearAttentionState>> {
    if num_gdn_layers == 0 {
        return Ok(None);
    }
    let conv_channels = 2 * model_config.linear_num_key_heads * model_config.linear_key_head_dim
        + model_config.linear_num_value_heads * model_config.linear_value_head_dim;
    Ok(Some(VkLinearAttentionState::zeros(
        device,
        num_gdn_layers,
        1,
        model_config.linear_num_value_heads,
        model_config.linear_key_head_dim,
        model_config.linear_value_head_dim,
        conv_channels,
        model_config.linear_conv_kernel_dim,
    )?))
}

fn scalar_product_sum(lhs: &VkTensor, rhs: &VkTensor) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    let prod = vk_mul(lhs, rhs)?;
    vk_sum_all(&prod)
}

fn add_scalar(acc: Option<VkTensor>, next: VkTensor) -> Result<Option<VkTensor>> {
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_add;
    match acc {
        Some(prev) => Ok(Some(vk_add(&prev, &next)?)),
        None => Ok(Some(next)),
    }
}

fn parameter_like(t: &VkTensor) -> Result<(VkTensor, TensorId)> {
    let id = mint_fresh_tensor_id()?;
    Ok((
        VkTensor::parameter(
            Arc::clone(t.buffer()),
            t.shape().to_vec(),
            t.dtype(),
            Arc::clone(t.device()),
            id,
        ),
        id,
    ))
}

fn accumulate_grad(
    shared_grads: &mut HashMap<TensorId, VkTensor>,
    pid: TensorId,
    grad: &VkTensor,
) -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_add_no_grad;
    if let Some(existing) = shared_grads.get(&pid).cloned() {
        let summed = vk_add_no_grad(&existing, grad)?;
        shared_grads.insert(pid, summed);
    } else {
        shared_grads.insert(pid, grad.clone());
    }
    Ok(())
}

fn accumulate_grads_except(
    shared_grads: &mut HashMap<TensorId, VkTensor>,
    grads: &kiln_vulkan_kernel::vk_autograd::VkGradStore,
    skip_id: TensorId,
) -> Result<()> {
    for (pid, grad) in grads.iter() {
        if *pid != skip_id {
            accumulate_grad(shared_grads, *pid, grad)?;
        }
    }
    Ok(())
}

fn vk_optimizer_step_from_grads(
    device: &VulkanDevice,
    lora_layers: &[VkLoraLayer],
    grads: &HashMap<TensorId, VkTensor>,
    adamw_state: &mut VkAdamWBook,
    lr: f32,
    optimizer: Optimizer,
    step: u32,
    context: &str,
) -> Result<()> {
    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = grads.get(&pid) else {
                continue;
            };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "{context}: optimizer F32 only"
            );
            anyhow::ensure!(
                param.num_elements() == grad.num_elements(),
                "{context}: param/grad element-count mismatch"
            );
            match optimizer {
                Optimizer::Sgd => {
                    dispatch_sgd_step_f32(
                        device,
                        param.buffer(),
                        grad.buffer(),
                        param.num_elements(),
                        lr,
                    )
                    .with_context(|| format!("dispatch_sgd_step_f32 ({context})"))?;
                }
                Optimizer::AdamW {
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                } => {
                    let state = adamw_state
                        .get(&pid)
                        .with_context(|| format!("missing AdamW state for param {:?}", pid))?;
                    dispatch_adamw_step_f32(
                        device,
                        param.buffer(),
                        grad.buffer(),
                        &state.m,
                        &state.v,
                        param.num_elements(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                    )
                    .with_context(|| format!("dispatch_adamw_step_f32 ({context})"))?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gdn_qkv_from_mixed(
    mixed_qkv: &VkTensor,
    w: &VkLinearAttentionWeights,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
) -> Result<(VkTensor, VkTensor, VkTensor)> {
    use kiln_vulkan_kernel::vk_ops::conv1d::vk_causal_conv1d;
    use kiln_vulkan_kernel::vk_ops::l2norm::vk_l2_norm_lastdim;
    use kiln_vulkan_kernel::vk_ops::narrow::vk_narrow_lastdim;
    use kiln_vulkan_kernel::vk_ops::permute::{vk_permute_rh_to_hr, vk_repeat_kv_heads};
    use kiln_vulkan_kernel::vk_ops::shape::{vk_reshape, vk_transpose_2d};

    let t = mixed_qkv.shape()[0];
    let batch = 1usize;
    let dk = w.head_dim_k;
    let dv = w.head_dim_v;
    let nk = w.heads_k;
    let nv = w.heads_v;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;
    let qkv_dim = 2 * qk_dim + v_dim;

    let mixed_ct = vk_transpose_2d(mixed_qkv)?;
    let mixed_chw_t = vk_reshape(&mixed_ct, &[batch, qkv_dim, t])?;
    let conv_out = vk_causal_conv1d(
        &mixed_chw_t,
        &w.conv1d,
        &state.layers[gdn_layer_idx].conv_state,
        batch,
        qkv_dim,
        t,
        w.conv_kernel,
    )?;
    let conv_ct = vk_reshape(&conv_out, &[qkv_dim, t])?;
    let conv_tc = vk_transpose_2d(&conv_ct)?;
    let conv_btc_t = vk_reshape(&conv_tc, &[batch, t, qkv_dim])?;
    let conv_2d = vk_reshape(&conv_btc_t, &[batch * t, qkv_dim])?;

    let q_flat = vk_narrow_lastdim(&conv_2d, 0, qk_dim)?;
    let k_flat = vk_narrow_lastdim(&conv_2d, qk_dim, qk_dim)?;
    let v_flat = vk_narrow_lastdim(&conv_2d, 2 * qk_dim, v_dim)?;

    let q_thd = vk_reshape(&q_flat, &[t, nk, dk])?;
    let k_thd = vk_reshape(&k_flat, &[t, nk, dk])?;
    let v_thd = vk_reshape(&v_flat, &[t, nv, dv])?;
    let q_htd = vk_permute_rh_to_hr(&q_thd)?;
    let k_htd = vk_permute_rh_to_hr(&k_thd)?;
    let v_htd = vk_permute_rh_to_hr(&v_thd)?;
    let q_bh = vk_reshape(&q_htd, &[batch, nk, t, dk])?;
    let k_bh = vk_reshape(&k_htd, &[batch, nk, t, dk])?;
    let v_bh = vk_reshape(&v_htd, &[batch, nv, t, dv])?;

    if nk < nv {
        let groups = nv / nk;
        let q_3d = vk_reshape(&q_bh, &[nk, t, dk])?;
        let q_rep = vk_repeat_kv_heads(&q_3d, groups)?;
        let q_expanded = vk_reshape(&q_rep, &[batch, nv, t, dk])?;
        let k_3d = vk_reshape(&k_bh, &[nk, t, dk])?;
        let k_rep = vk_repeat_kv_heads(&k_3d, groups)?;
        let k_expanded = vk_reshape(&k_rep, &[batch, nv, t, dk])?;
        let q_normed = vk_l2_norm_lastdim(&q_expanded, 1.0 / (dk as f32).sqrt(), 1e-6)?;
        let k_normed = vk_l2_norm_lastdim(&k_expanded, 1.0, 1e-6)?;
        Ok((q_normed, k_normed, v_bh))
    } else {
        let q_normed = vk_l2_norm_lastdim(&q_bh, 1.0 / (dk as f32).sqrt(), 1e-6)?;
        let k_normed = vk_l2_norm_lastdim(&k_bh, 1.0, 1e-6)?;
        Ok((q_normed, k_normed, v_bh))
    }
}

fn gdn_gates_from_ab(
    a_proj: &VkTensor,
    b_proj: &VkTensor,
    w: &VkLinearAttentionWeights,
) -> Result<(VkTensor, VkTensor)> {
    use kiln_vulkan_kernel::vk_ops::gdn_gates::vk_gdn_gates;
    use kiln_vulkan_kernel::vk_ops::shape::{vk_reshape, vk_transpose_2d};

    let t = a_proj.shape()[0];
    let batch = 1usize;
    let nv = w.heads_v;
    let a_3 = vk_reshape(a_proj, &[batch, t, nv])?;
    let b_3 = vk_reshape(b_proj, &[batch, t, nv])?;
    let (beta_tn, g_tn) = vk_gdn_gates(&a_3, &b_3, &w.a_log, &w.dt_bias, nv)?;
    let beta_2d = vk_reshape(&beta_tn, &[t, nv])?;
    let g_2d = vk_reshape(&g_tn, &[t, nv])?;
    let beta_t = vk_transpose_2d(&beta_2d)?;
    let g_t = vk_transpose_2d(&g_2d)?;
    Ok((
        vk_reshape(&beta_t, &[batch, nv, t])?,
        vk_reshape(&g_t, &[batch, nv, t])?,
    ))
}

fn gdn_state_tensor(
    x: &VkTensor,
    w: &VkLinearAttentionWeights,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
) -> VkTensor {
    VkTensor::from_buffer(
        Arc::clone(&state.layers[gdn_layer_idx].recurrent_state),
        vec![1, w.heads_v, w.head_dim_k, w.head_dim_v],
        VkDType::F32,
        Arc::clone(x.device()),
    )
}

fn gdn_chunkwise_from_parts(
    x_for_device: &VkTensor,
    w: &VkLinearAttentionWeights,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    beta: &VkTensor,
    g: &VkTensor,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise;
    let t = q.shape()[2];
    let mut state_t = gdn_state_tensor(x_for_device, w, state, gdn_layer_idx);
    let chunk_c = if t < 64 { t.max(1) } else { 64 };
    vk_gdn_chunkwise(q, k, v, beta, g, &mut state_t, chunk_c)
}

fn gdn_normed_from_chunk_and_z(
    out_chunkwise: &VkTensor,
    z_raw: &VkTensor,
    w: &VkLinearAttentionWeights,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::gdn_gated_rms_norm::vk_gdn_gated_rms_norm;
    use kiln_vulkan_kernel::vk_ops::permute::vk_permute_hr_to_rh;
    use kiln_vulkan_kernel::vk_ops::shape::vk_reshape;

    let t = z_raw.shape()[0];
    let batch = 1usize;
    let nv = w.heads_v;
    let dv = w.head_dim_v;
    let v_dim = nv * dv;

    let out_3 = vk_reshape(out_chunkwise, &[nv, t, dv])?;
    let out_t_nv_dv = vk_permute_hr_to_rh(&out_3)?;
    let flat_t = vk_reshape(&out_t_nv_dv, &[batch * t, v_dim])?;
    let flat_per_head = vk_reshape(&flat_t, &[batch * t * nv, dv])?;
    let z_per_head = vk_reshape(z_raw, &[batch * t * nv, dv])?;
    let normed_per_head = vk_gdn_gated_rms_norm(&flat_per_head, &z_per_head, &w.gated_norm, w.eps)?;
    vk_reshape(&normed_per_head, &[batch * t, v_dim])
}

fn gdn_compute_h_norm(x: &VkTensor, w: &VkLinearAttentionWeights) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
    vk_rmsnorm(x, &w.layer_norm, w.eps)
}

fn gdn_compute_normed_no_grad(
    x: &VkTensor,
    w: &VkLinearAttentionWeights,
    lora: &VkLoraLayer,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
) -> Result<VkTensor> {
    let h_norm = gdn_compute_h_norm(x, w)?;
    let mixed_qkv = vk_linear_with_lora(&h_norm, &w.in_proj_qkv, lora.in_proj_qkv.as_ref())?;
    let z_raw = vk_linear_with_lora(&h_norm, &w.in_proj_z, lora.in_proj_z.as_ref())?;
    let a_proj = vk_linear_with_lora(&h_norm, &w.in_proj_a, None)?;
    let b_proj = vk_linear_with_lora(&h_norm, &w.in_proj_b, None)?;
    let (q, k, v) = gdn_qkv_from_mixed(&mixed_qkv, w, state, gdn_layer_idx)?;
    let (beta, g) = gdn_gates_from_ab(&a_proj, &b_proj, w)?;
    let out_chunk = gdn_chunkwise_from_parts(x, w, state, gdn_layer_idx, &q, &k, &v, &beta, &g)?;
    gdn_normed_from_chunk_and_z(&out_chunk, &z_raw, w)
}

fn gdn_attention_block_value(
    x: &VkTensor,
    w: &VkLinearAttentionWeights,
    lora: &VkLoraLayer,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_add;

    let normed = gdn_compute_normed_no_grad(x, w, lora, state, gdn_layer_idx)?;
    let out_proj = vk_linear_with_lora(&normed, &w.out_proj, lora.gdn_out_proj.as_ref())?;
    vk_add(x, &out_proj)
}

#[allow(clippy::too_many_arguments)]
fn vk_gdn_layer_backward_split(
    boundary: &VkTensor,
    upstream: &VkTensor,
    w: &VkLinearAttentionWeights,
    lora: &VkLoraLayer,
    state: &VkLinearAttentionState,
    gdn_layer_idx: usize,
    shared_grads: &mut HashMap<TensorId, VkTensor>,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_add_no_grad;
    use kiln_vulkan_kernel::vk_ops::shape::vk_reshape;

    let detached_lora = VkLoraLayer {
        in_proj_qkv: lora.in_proj_qkv.as_ref().map(detach_lora_pair),
        in_proj_z: lora.in_proj_z.as_ref().map(detach_lora_pair),
        gdn_out_proj: lora.gdn_out_proj.as_ref().map(detach_lora_pair),
        ..Default::default()
    };

    // 1. Residual + out_proj. Capture dL/d(normed) and any out_proj LoRA grads.
    let normed = gdn_compute_normed_no_grad(boundary, w, &detached_lora, state, gdn_layer_idx)?;
    let (normed_param, normed_id) = parameter_like(&normed)?;
    let out_proj = vk_linear_with_lora(&normed_param, &w.out_proj, lora.gdn_out_proj.as_ref())?;
    let scalar = scalar_product_sum(&out_proj, upstream)?;
    let grads = vk_backward(&scalar).context("vk_gdn_layer_backward_split: out_proj")?;
    let grad_normed = grads
        .get(normed_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing normed grad for GDN layer"))?;
    accumulate_grads_except(shared_grads, &grads, normed_id)?;
    drop(grads);
    drop(out_proj);
    drop(normed_param);
    drop(normed);

    // 2. Gated RMSNorm. Capture dL/d(chunkwise_out) and dL/d(z_raw).
    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let mixed_qkv =
        vk_linear_with_lora(&h_norm, &w.in_proj_qkv, detached_lora.in_proj_qkv.as_ref())?;
    let z_raw = vk_linear_with_lora(&h_norm, &w.in_proj_z, detached_lora.in_proj_z.as_ref())?;
    let a_proj = vk_linear_with_lora(&h_norm, &w.in_proj_a, None)?;
    let b_proj = vk_linear_with_lora(&h_norm, &w.in_proj_b, None)?;
    let (q, k, v) = gdn_qkv_from_mixed(&mixed_qkv, w, state, gdn_layer_idx)?;
    let (beta, g) = gdn_gates_from_ab(&a_proj, &b_proj, w)?;
    let out_chunk =
        gdn_chunkwise_from_parts(boundary, w, state, gdn_layer_idx, &q, &k, &v, &beta, &g)?;
    let (out_chunk_param, out_chunk_id) = parameter_like(&out_chunk)?;
    let (z_param, z_id) = parameter_like(&z_raw)?;
    let normed_2 = gdn_normed_from_chunk_and_z(&out_chunk_param, &z_param, w)?;
    let scalar = scalar_product_sum(&normed_2, &grad_normed)?;
    let grads = vk_backward(&scalar).context("vk_gdn_layer_backward_split: gated norm")?;
    let grad_out_chunk = grads
        .get(out_chunk_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing chunkwise-output grad for GDN layer"))?;
    let grad_z = grads
        .get(z_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing z grad for GDN layer"))?;
    drop(grads);
    drop(normed_2);
    drop(out_chunk_param);
    drop(z_param);
    drop(out_chunk);
    drop(q);
    drop(k);
    drop(v);
    drop(beta);
    drop(g);
    drop(a_proj);
    drop(b_proj);
    drop(z_raw);
    drop(mixed_qkv);
    drop(h_norm);

    // 3. Chunkwise recurrence. Capture dL/d(q,k,v,beta,g).
    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let mixed_qkv =
        vk_linear_with_lora(&h_norm, &w.in_proj_qkv, detached_lora.in_proj_qkv.as_ref())?;
    let a_proj = vk_linear_with_lora(&h_norm, &w.in_proj_a, None)?;
    let b_proj = vk_linear_with_lora(&h_norm, &w.in_proj_b, None)?;
    let (q, k, v) = gdn_qkv_from_mixed(&mixed_qkv, w, state, gdn_layer_idx)?;
    let (beta, g) = gdn_gates_from_ab(&a_proj, &b_proj, w)?;
    let (q_param, q_id) = parameter_like(&q)?;
    let (k_param, k_id) = parameter_like(&k)?;
    let (v_param, v_id) = parameter_like(&v)?;
    let (beta_param, beta_id) = parameter_like(&beta)?;
    let (g_param, g_id) = parameter_like(&g)?;
    let out_chunk_2 = gdn_chunkwise_from_parts(
        boundary,
        w,
        state,
        gdn_layer_idx,
        &q_param,
        &k_param,
        &v_param,
        &beta_param,
        &g_param,
    )?;
    let scalar = scalar_product_sum(&out_chunk_2, &grad_out_chunk)?;
    let grads = vk_backward(&scalar).context("vk_gdn_layer_backward_split: chunkwise")?;
    let grad_q = grads
        .get(q_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing q grad for GDN layer"))?;
    let grad_k = grads
        .get(k_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing k grad for GDN layer"))?;
    let grad_v = grads
        .get(v_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing v grad for GDN layer"))?;
    let grad_beta = grads
        .get(beta_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing beta grad for GDN layer"))?;
    let grad_g = grads
        .get(g_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing g grad for GDN layer"))?;
    drop(grads);
    drop(out_chunk_2);
    drop(q_param);
    drop(k_param);
    drop(v_param);
    drop(beta_param);
    drop(g_param);
    drop(q);
    drop(k);
    drop(v);
    drop(beta);
    drop(g);
    drop(a_proj);
    drop(b_proj);
    drop(mixed_qkv);
    drop(h_norm);

    // 4. Conv/split/repeat path. Capture dL/d(mixed_qkv), then project
    // that back through the qkv linear to h_norm and qkv LoRA.
    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let mixed_qkv =
        vk_linear_with_lora(&h_norm, &w.in_proj_qkv, detached_lora.in_proj_qkv.as_ref())?;
    let (mixed_param, mixed_id) = parameter_like(&mixed_qkv)?;
    let (q_2, k_2, v_2) = gdn_qkv_from_mixed(&mixed_param, w, state, gdn_layer_idx)?;
    let mut scalar = None;
    scalar = add_scalar(scalar, scalar_product_sum(&q_2, &grad_q)?)?;
    scalar = add_scalar(scalar, scalar_product_sum(&k_2, &grad_k)?)?;
    scalar = add_scalar(scalar, scalar_product_sum(&v_2, &grad_v)?)?;
    let grads = vk_backward(&scalar.expect("scalar is populated"))
        .context("vk_gdn_layer_backward_split: qkv conv")?;
    let grad_mixed = grads
        .get(mixed_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing mixed_qkv grad for GDN layer"))?;
    drop(grads);
    drop(q_2);
    drop(k_2);
    drop(v_2);
    drop(mixed_param);
    drop(mixed_qkv);
    drop(h_norm);

    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let (h_param, h_id) = parameter_like(&h_norm)?;
    let mixed_from_h = vk_linear_with_lora(&h_param, &w.in_proj_qkv, lora.in_proj_qkv.as_ref())?;
    let scalar = scalar_product_sum(&mixed_from_h, &grad_mixed)?;
    let grads =
        vk_backward(&scalar).context("vk_gdn_layer_backward_split: qkv projection to h_norm")?;
    let mut grad_h_norm = grads
        .get(h_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing h_norm qkv grad for GDN layer"))?;
    accumulate_grads_except(shared_grads, &grads, h_id)?;
    drop(grads);
    drop(mixed_from_h);
    drop(h_param);
    drop(h_norm);

    // 5. Gate path: beta/g -> a/b, then frozen a/b projections -> h_norm.
    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let a_proj = vk_linear_with_lora(&h_norm, &w.in_proj_a, None)?;
    let b_proj = vk_linear_with_lora(&h_norm, &w.in_proj_b, None)?;
    let (a_param, a_id) = parameter_like(&a_proj)?;
    let (b_param, b_id) = parameter_like(&b_proj)?;
    let (beta_2, g_2) = gdn_gates_from_ab(&a_param, &b_param, w)?;
    let mut scalar = None;
    scalar = add_scalar(scalar, scalar_product_sum(&beta_2, &grad_beta)?)?;
    scalar = add_scalar(scalar, scalar_product_sum(&g_2, &grad_g)?)?;
    let grads = vk_backward(&scalar.expect("scalar is populated"))
        .context("vk_gdn_layer_backward_split: gates")?;
    let grad_a = grads
        .get(a_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing a grad for GDN layer"))?;
    let grad_b = grads
        .get(b_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing b grad for GDN layer"))?;
    drop(grads);
    drop(beta_2);
    drop(g_2);
    drop(a_param);
    drop(b_param);
    drop(a_proj);
    drop(b_proj);
    drop(h_norm);

    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let (h_param, h_id) = parameter_like(&h_norm)?;
    let a_from_h = vk_linear_with_lora(&h_param, &w.in_proj_a, None)?;
    let b_from_h = vk_linear_with_lora(&h_param, &w.in_proj_b, None)?;
    let mut scalar = None;
    scalar = add_scalar(scalar, scalar_product_sum(&a_from_h, &grad_a)?)?;
    scalar = add_scalar(scalar, scalar_product_sum(&b_from_h, &grad_b)?)?;
    let grads = vk_backward(&scalar.expect("scalar is populated"))
        .context("vk_gdn_layer_backward_split: gate projections to h_norm")?;
    let grad_h_gate = grads
        .get(h_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing h_norm gate grad for GDN layer"))?;
    grad_h_norm = vk_add_no_grad(&grad_h_norm, &grad_h_gate)?;
    drop(grads);
    drop(a_from_h);
    drop(b_from_h);
    drop(h_param);
    drop(h_norm);

    // 6. z projection -> h_norm and z LoRA.
    let h_norm = gdn_compute_h_norm(boundary, w)?;
    let (h_param, h_id) = parameter_like(&h_norm)?;
    let z_from_h = vk_linear_with_lora(&h_param, &w.in_proj_z, lora.in_proj_z.as_ref())?;
    let scalar = scalar_product_sum(&z_from_h, &grad_z)?;
    let grads =
        vk_backward(&scalar).context("vk_gdn_layer_backward_split: z projection to h_norm")?;
    let grad_h_z = grads
        .get(h_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing h_norm z grad for GDN layer"))?;
    grad_h_norm = vk_add_no_grad(&grad_h_norm, &grad_h_z)?;
    accumulate_grads_except(shared_grads, &grads, h_id)?;
    drop(grads);
    drop(z_from_h);
    drop(h_param);
    drop(h_norm);

    // 7. Input RMSNorm -> boundary, then add residual gradient.
    let (x_param, x_id) = parameter_like(boundary)?;
    let h_from_x = gdn_compute_h_norm(&x_param, w)?;
    let grad_h_norm = vk_reshape(&grad_h_norm, h_from_x.shape())?;
    let scalar = scalar_product_sum(&h_from_x, &grad_h_norm)?;
    let grads = vk_backward(&scalar).context("vk_gdn_layer_backward_split: input rmsnorm")?;
    let grad_x_norm = grads
        .get(x_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing input grad for GDN layer"))?;
    vk_add_no_grad(upstream, &grad_x_norm)
}

fn gdn_layer_index_map(weights: &VkModelWeights) -> Vec<Option<usize>> {
    let mut next = 0usize;
    weights
        .layers
        .iter()
        .map(|layer| {
            if matches!(layer, VkLayerWeights::LinearAttention(_)) {
                let idx = next;
                next += 1;
                Some(idx)
            } else {
                None
            }
        })
        .collect()
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
    vk_train_step_with_state(
        weights,
        lora_layers,
        input_ids,
        None,
        adamw_state,
        cfg,
        step,
    )
}

/// Mint a synthetic candle TensorId — used to wrap a boundary
/// activation as a parameter leaf so its gradient can be captured
/// from a sub-tape.
fn mint_fresh_tensor_id() -> Result<TensorId> {
    use candle_core::{Device, Tensor, Var};
    let dummy = Tensor::from_vec(vec![0.0_f32], (1,), &Device::Cpu)?;
    Ok(Var::from_tensor(&dummy)?.id())
}

fn vk_embedding_hidden(weights: &VkModelWeights, input_ids: &[u32]) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::embedding::{
        upload_u32_ids, vk_embedding_lookup_bf16, vk_embedding_lookup_f32,
    };
    let device = weights.embed_tokens.device();
    let ids = upload_u32_ids(device, input_ids)?;
    match weights.embed_dtype {
        VkDType::F32 => {
            vk_embedding_lookup_f32(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)
        }
        VkDType::Bf16 => {
            vk_embedding_lookup_bf16(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vk_forward_to_layer_input(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    end_layer: usize,
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    gdn_map: &[Option<usize>],
    rope_tables: Option<(&VkTensor, &VkTensor)>,
) -> Result<(VkTensor, Option<VkLinearAttentionState>)> {
    use kiln_model::vk_forward::{
        vk_full_attention_layer_with_rope, vk_transformer_layer_with_state,
    };
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;

    anyhow::ensure!(
        end_layer <= weights.layers.len(),
        "vk_forward_to_layer_input: end_layer {} > {}",
        end_layer,
        weights.layers.len()
    );
    let mut h = vk_embedding_hidden(weights, input_ids)?;
    let mut state = fresh_gdn_state(weights.embed_tokens.device(), model_config, num_gdn_layers)?;
    let profile_layers = env_flag("KILN_PROFILE_VK_RECOMPUTE_LAYERS", false);
    let profile_finite = env_flag("KILN_PROFILE_VK_RECOMPUTE_FINITE", false);
    for layer_idx in 0..end_layer {
        if profile_layers {
            let layer_kind = match &weights.layers[layer_idx] {
                VkLayerWeights::FullAttention(_) => "full_attention",
                VkLayerWeights::LinearAttention(_) => "linear_attention",
            };
            tracing::info!(
                end_layer,
                layer_idx,
                layer_kind,
                seq_len = input_ids.len(),
                "vk-native recompute forward layer begin"
            );
        }
        h = match &weights.layers[layer_idx] {
            VkLayerWeights::FullAttention(full) => {
                let rope_arg = rope_tables.map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                vk_full_attention_layer_with_rope(&h, full, &lora_layers[layer_idx], rope_arg)?
            }
            VkLayerWeights::LinearAttention(_) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let s = state
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("GDN layer {layer_idx} requires state"))?;
                vk_transformer_layer_with_state(
                    &h,
                    &weights.layers[layer_idx],
                    &lora_layers[layer_idx],
                    Some((s, gdn_idx)),
                )?
            }
        };
        if profile_layers {
            if profile_finite {
                let hidden_sum = vk_sum_all(&h)?.to_vec_f32()?[0];
                tracing::info!(
                    end_layer,
                    layer_idx,
                    seq_len = input_ids.len(),
                    hidden_sum = format!("{hidden_sum:.6e}"),
                    hidden_sum_finite = hidden_sum.is_finite(),
                    "vk-native recompute forward layer done"
                );
            } else {
                tracing::info!(
                    end_layer,
                    layer_idx,
                    seq_len = input_ids.len(),
                    "vk-native recompute forward layer done"
                );
            }
        }
    }
    Ok((h, state))
}

#[allow(clippy::too_many_arguments)]
fn vk_forward_layer_boundaries(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    gdn_map: &[Option<usize>],
    rope_tables: Option<(&VkTensor, &VkTensor)>,
) -> Result<(Vec<VkTensor>, Option<VkLinearAttentionState>)> {
    use kiln_model::vk_forward::{
        vk_full_attention_layer_with_rope, vk_transformer_layer_with_state,
    };
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;

    let mut h = vk_embedding_hidden(weights, input_ids)?;
    let mut boundaries = Vec::with_capacity(weights.layers.len() + 1);
    boundaries.push(h.clone());
    let mut state = fresh_gdn_state(weights.embed_tokens.device(), model_config, num_gdn_layers)?;
    let profile_layers = env_flag("KILN_PROFILE_VK_RECOMPUTE_LAYERS", false);
    let profile_finite = env_flag("KILN_PROFILE_VK_RECOMPUTE_FINITE", false);
    for layer_idx in 0..weights.layers.len() {
        if profile_layers {
            let layer_kind = match &weights.layers[layer_idx] {
                VkLayerWeights::FullAttention(_) => "full_attention",
                VkLayerWeights::LinearAttention(_) => "linear_attention",
            };
            tracing::info!(
                end_layer = weights.layers.len(),
                layer_idx,
                layer_kind,
                seq_len = input_ids.len(),
                boundary_cache = true,
                "vk-native recompute forward layer begin"
            );
        }
        h = match &weights.layers[layer_idx] {
            VkLayerWeights::FullAttention(full) => {
                let rope_arg = rope_tables.map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                vk_full_attention_layer_with_rope(&h, full, &lora_layers[layer_idx], rope_arg)?
            }
            VkLayerWeights::LinearAttention(_) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let s = state
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("GDN layer {layer_idx} requires state"))?;
                vk_transformer_layer_with_state(
                    &h,
                    &weights.layers[layer_idx],
                    &lora_layers[layer_idx],
                    Some((s, gdn_idx)),
                )?
            }
        };
        if profile_layers {
            if profile_finite {
                let hidden_sum = vk_sum_all(&h)?.to_vec_f32()?[0];
                tracing::info!(
                    end_layer = weights.layers.len(),
                    layer_idx,
                    seq_len = input_ids.len(),
                    boundary_cache = true,
                    hidden_sum = format!("{hidden_sum:.6e}"),
                    hidden_sum_finite = hidden_sum.is_finite(),
                    "vk-native recompute forward layer done"
                );
            } else {
                tracing::info!(
                    end_layer = weights.layers.len(),
                    layer_idx,
                    seq_len = input_ids.len(),
                    boundary_cache = true,
                    "vk-native recompute forward layer done"
                );
            }
        }
        boundaries.push(h.clone());
    }
    Ok((boundaries, state))
}

fn recompute_boundary_cache_limit_bytes() -> usize {
    if let Some(limit) = std::env::var("KILN_VK_RECOMPUTE_BOUNDARY_CACHE_GB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as usize)
    {
        return limit;
    }
    auto_recompute_boundary_cache_limit_bytes()
}

#[cfg(target_os = "linux")]
fn auto_recompute_boundary_cache_limit_bytes() -> usize {
    let raw = match std::fs::read_to_string("/proc/meminfo") {
        Ok(raw) => raw,
        Err(_) => return 0,
    };
    let mut mem_available = None;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_available = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .map(|kib| kib.saturating_mul(1024));
            break;
        }
    }
    let Some(available) = mem_available else {
        return 0;
    };
    const GIB: usize = 1024 * 1024 * 1024;
    // Keep at least 4 GiB outside the optional boundary cache for the
    // compositor, driver, base model buffers, and transient dispatch scratch.
    // Cap at 10 GiB because larger examples can always fall back to exact
    // layerwise recompute instead of pinning every boundary.
    available.saturating_sub(4 * GIB).min(10 * GIB)
}

#[cfg(not(target_os = "linux"))]
fn auto_recompute_boundary_cache_limit_bytes() -> usize {
    0
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
        VkLayerWeights, vk_compute_rope_tables, vk_full_attention_layer_with_rope,
    };
    use kiln_vulkan_kernel::VkTensor;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
    use kiln_vulkan_kernel::vk_ops::embedding::{
        upload_u32_ids, vk_embedding_lookup_bf16, vk_embedding_lookup_f32,
    };
    use kiln_vulkan_kernel::vk_ops::flce::{flce_recommended_chunk_len_for_tensors, vk_flce_loss};
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;

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
        let start_layer = if seg_idx == 0 {
            0
        } else {
            segments[seg_idx - 1]
        };
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
    let loss = vk_flce_loss(
        &h_norm,
        &weights.lm_head,
        &labels,
        flce_recommended_chunk_len_for_tensors(&h_norm, &weights.lm_head),
    )?;
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
        let start_layer = if last_seg == 0 {
            0
        } else {
            segments[last_seg - 1]
        };
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
        let loss2 = vk_flce_loss(
            &h_norm2,
            &weights.lm_head,
            &labels,
            flce_recommended_chunk_len_for_tensors(&h_norm2, &weights.lm_head),
        )?;
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
        let start_layer = if seg_idx == 0 {
            0
        } else {
            segments[seg_idx - 1]
        };
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
                    let summed =
                        kiln_vulkan_kernel::vk_ops::elementwise::vk_add_no_grad(&existing, g)?;
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

/// Exact layerwise reverse-recompute training step.
///
/// This path is deliberately slower than segment checkpointing, but it keeps
/// only one transformer layer's tracked tape live at a time and supports hybrid
/// FullAttention+GDN models. It is the vk-native long-context path: prefix
/// activations are recomputed with detached LoRA tensors, then each layer is
/// replayed once with a synthetic boundary leaf and the upstream-gradient
/// scalar trick.
#[allow(clippy::too_many_arguments)]
pub fn vk_recompute_train_step_with_state_masked(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    label_mask: &[bool],
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
) -> Result<f32> {
    use kiln_model::vk_forward::{
        vk_compute_rope_tables, vk_full_attention_attention_block_with_rope,
        vk_full_attention_mlp_down_from_gated, vk_full_attention_mlp_gated,
        vk_linear_attention_mlp_down_from_gated, vk_linear_attention_mlp_gated,
    };
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::{vk_add_no_grad, vk_mul};
    use kiln_vulkan_kernel::vk_ops::flce::{flce_recommended_chunk_len_for_tensors, vk_flce_loss};
    use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;

    anyhow::ensure!(
        label_mask.len() == input_ids.len(),
        "vk_recompute_train_step: label mask length {} != input length {}",
        label_mask.len(),
        input_ids.len()
    );
    anyhow::ensure!(
        input_ids.len() >= 2,
        "vk_recompute_train_step: need at least 2 tokens"
    );
    let active_rows: Vec<u32> = label_mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(i, &active)| active.then_some(i as u32))
        .collect();
    anyhow::ensure!(
        !active_rows.is_empty(),
        "vk_recompute_train_step: no active label positions"
    );
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();

    let device = weights.embed_tokens.device();
    let detached_lora = detach_lora_layers(lora_layers);
    let gdn_map = gdn_layer_index_map(weights);
    let rope_tables = if !weights.rotary_inv_freq.is_empty() && weights.rotary_dim > 0 {
        Some(vk_compute_rope_tables(
            device,
            &weights.rotary_inv_freq,
            input_ids.len(),
        )?)
    } else {
        None
    };
    let rope_refs = rope_tables.as_ref().map(|(cos, sin)| (cos, sin));
    let profile = env_flag("KILN_PROFILE_VK_RECOMPUTE", false);
    let boundary_cache_limit = recompute_boundary_cache_limit_bytes();
    let boundary_cache_bytes = (weights.layers.len() + 1)
        .saturating_mul(input_ids.len())
        .saturating_mul(weights.hidden)
        .saturating_mul(std::mem::size_of::<f32>());
    let use_boundary_cache = kiln_core::env_flag::env_tristate("KILN_VK_RECOMPUTE_BOUNDARY_CACHE")
        .unwrap_or(true)
        && boundary_cache_limit > 0
        && boundary_cache_bytes <= boundary_cache_limit;

    // Seed upstream with d(loss)/d(pre_final_norm_hidden).
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active_labels = active_rows.len(),
            boundary_cache = use_boundary_cache,
            boundary_cache_bytes,
            boundary_cache_limit,
            "vk-native recompute final forward begin"
        );
    }
    let (final_hidden, boundary_cache) = if use_boundary_cache {
        let (boundaries, _state) = vk_forward_layer_boundaries(
            weights,
            &detached_lora,
            input_ids,
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        let final_hidden = boundaries
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("vk_recompute_train_step: empty boundary cache"))?;
        (final_hidden, Some(boundaries))
    } else {
        let (final_hidden, _state) = vk_forward_to_layer_input(
            weights,
            &detached_lora,
            input_ids,
            weights.layers.len(),
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        (final_hidden, None)
    };
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            boundary_cache = use_boundary_cache,
            "vk-native recompute final forward done"
        );
    }
    let final_id = mint_fresh_tensor_id()?;
    let final_param = VkTensor::parameter(
        Arc::clone(final_hidden.buffer()),
        final_hidden.shape().to_vec(),
        final_hidden.dtype(),
        Arc::clone(final_hidden.device()),
        final_id,
    );
    let h_norm = vk_rmsnorm(&final_param, &weights.final_norm_weight, 1e-5)?;
    let active_h = vk_index_select_rows(&h_norm, &active_rows)?;
    let loss = vk_flce_loss(
        &active_h,
        &weights.lm_head,
        &labels,
        flce_recommended_chunk_len_for_tensors(&active_h, &weights.lm_head),
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active_labels = active_rows.len(),
            loss = format!("{loss_val:.6}"),
            "vk-native recompute finite loss computed"
        );
    }
    let final_grads = vk_backward(&loss)?;
    let mut upstream = final_grads
        .get(final_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("vk_recompute_train_step: missing final upstream grad"))?;

    let mut shared_grads: HashMap<TensorId, VkTensor> = HashMap::new();

    for layer_idx in (0..weights.layers.len()).rev() {
        if profile {
            tracing::info!(
                step,
                layer_idx,
                seq_len = input_ids.len(),
                boundary_cache = use_boundary_cache,
                "vk-native recompute reverse layer begin"
            );
        }
        let (boundary, state) = if let Some(boundaries) = boundary_cache.as_ref() {
            let boundary = boundaries
                .get(layer_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing cached boundary for layer {layer_idx}"))?;
            let state =
                fresh_gdn_state(weights.embed_tokens.device(), model_config, num_gdn_layers)?;
            (boundary, state)
        } else {
            vk_forward_to_layer_input(
                weights,
                &detached_lora,
                input_ids,
                layer_idx,
                model_config,
                num_gdn_layers,
                &gdn_map,
                rope_refs,
            )
            .with_context(|| format!("vk_recompute_train_step: prefix to layer {layer_idx}"))?
        };
        if profile {
            tracing::info!(
                step,
                layer_idx,
                seq_len = input_ids.len(),
                boundary_cache = use_boundary_cache,
                "vk-native recompute reverse prefix done"
            );
        }
        match &weights.layers[layer_idx] {
            VkLayerWeights::FullAttention(full) => {
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native recompute reverse full-attention block begin"
                    );
                }
                let rope_arg = rope_refs.map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                let after_attn_value = vk_full_attention_attention_block_with_rope(
                    &boundary,
                    full,
                    &detached_lora[layer_idx],
                    rope_arg,
                )
                .with_context(|| {
                    format!("vk_recompute_train_step: full-attn prefix block {layer_idx}")
                })?;
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native recompute reverse full-attention block done"
                    );
                }

                let gated_value =
                    vk_full_attention_mlp_gated(&after_attn_value, full, &detached_lora[layer_idx])
                        .with_context(|| {
                            format!(
                                "vk_recompute_train_step: full layer {layer_idx} MLP gated value"
                            )
                        })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_full_attention_mlp_down_from_gated(
                    &gated_param,
                    full,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_train_step: backward full layer {layer_idx} MLP down")
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid == gated_id {
                        continue;
                    }
                    if let Some(existing) = shared_grads.get(pid).cloned() {
                        let summed = vk_add_no_grad(&existing, grad)?;
                        shared_grads.insert(*pid, summed);
                    } else {
                        shared_grads.insert(*pid, grad.clone());
                    }
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_full_attention_mlp_gated(&after_param, full, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_train_step: backward full layer {layer_idx} MLP gate/up")
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for full layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid == after_id {
                        continue;
                    }
                    if let Some(existing) = shared_grads.get(pid).cloned() {
                        let summed = vk_add_no_grad(&existing, grad)?;
                        shared_grads.insert(*pid, summed);
                    } else {
                        shared_grads.insert(*pid, grad.clone());
                    }
                }

                let boundary_id = mint_fresh_tensor_id()?;
                let h_param = VkTensor::parameter(
                    Arc::clone(boundary.buffer()),
                    boundary.shape().to_vec(),
                    boundary.dtype(),
                    Arc::clone(boundary.device()),
                    boundary_id,
                );
                let after_attn = vk_full_attention_attention_block_with_rope(
                    &h_param,
                    full,
                    &lora_layers[layer_idx],
                    rope_arg,
                )?;
                let prod = vk_mul(&after_attn, &upstream_after_attn)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_train_step: backward full layer {layer_idx} attention")
                })?;
                upstream = grads.get(boundary_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing boundary grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid == boundary_id {
                        continue;
                    }
                    if let Some(existing) = shared_grads.get(pid).cloned() {
                        let summed = vk_add_no_grad(&existing, grad)?;
                        shared_grads.insert(*pid, summed);
                    } else {
                        shared_grads.insert(*pid, grad.clone());
                    }
                }
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native recompute reverse full layer done"
                    );
                }
                continue;
            }
            VkLayerWeights::LinearAttention(linear) => {
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native recompute reverse GDN layer begin"
                    );
                }
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let s = state
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GDN layer {layer_idx} requires state"))?;

                let after_attn_value = gdn_attention_block_value(
                    &boundary,
                    linear,
                    &detached_lora[layer_idx],
                    s,
                    gdn_idx,
                )
                .with_context(|| {
                    format!("vk_recompute_train_step: GDN layer {layer_idx} attention value")
                })?;

                let gated_value = vk_linear_attention_mlp_gated(
                    &after_attn_value,
                    linear,
                    &detached_lora[layer_idx],
                )
                .with_context(|| {
                    format!("vk_recompute_train_step: GDN layer {layer_idx} MLP gated value")
                })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_linear_attention_mlp_down_from_gated(
                    &gated_param,
                    linear,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_train_step: backward GDN layer {layer_idx} MLP down")
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for GDN layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid == gated_id {
                        continue;
                    }
                    accumulate_grad(&mut shared_grads, *pid, grad)?;
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_linear_attention_mlp_gated(&after_param, linear, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_train_step: backward GDN layer {layer_idx} MLP gate/up")
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for GDN layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid == after_id {
                        continue;
                    }
                    accumulate_grad(&mut shared_grads, *pid, grad)?;
                }

                upstream = vk_gdn_layer_backward_split(
                    &boundary,
                    &upstream_after_attn,
                    linear,
                    &lora_layers[layer_idx],
                    s,
                    gdn_idx,
                    &mut shared_grads,
                )
                .with_context(|| {
                    format!("vk_recompute_train_step: split backward GDN layer {layer_idx}")
                })?;
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native recompute reverse GDN layer done"
                    );
                }
                continue;
            }
        }
    }

    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = shared_grads.get(&pid) else {
                continue;
            };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "vk_recompute_train_step: AdamW F32 only"
            );
            anyhow::ensure!(
                param.num_elements() == grad.num_elements(),
                "vk_recompute_train_step: param/grad element-count mismatch"
            );
            let state = adamw_state
                .get(&pid)
                .with_context(|| format!("missing AdamW state for param {:?}", pid))?;
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
            .context("dispatch_adamw_step_f32 (recompute)")?;
        }
    }

    Ok(loss_val)
}

/// Exact layerwise reverse-recompute GRPO step.
///
/// This is the GRPO counterpart to `vk_recompute_train_step_with_state_masked`:
/// reference log-probs are supplied by the caller, the final loss is the
/// clipped GRPO objective plus KL penalty, and the transformer body is replayed
/// one layer at a time so policy training does not keep a full long-context
/// forward tape alive.
#[allow(clippy::too_many_arguments)]
pub fn vk_recompute_grpo_train_step_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    active_rows: &[u32],
    labels: &[u32],
    ref_log_probs: &VkTensor,
    advantage: f32,
    clip_epsilon: f32,
    kl_coeff: f32,
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    optimizer: Optimizer,
    step: u32,
) -> Result<f32> {
    use kiln_model::vk_forward::{
        vk_compute_rope_tables, vk_full_attention_attention_block_with_rope,
        vk_full_attention_mlp_down_from_gated, vk_full_attention_mlp_gated,
        vk_linear_attention_mlp_down_from_gated, vk_linear_attention_mlp_gated,
    };
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::{vk_add_no_grad, vk_mul};
    use kiln_vulkan_kernel::vk_ops::flce::{flce_recommended_chunk_len_for_tensors, vk_grpo_loss};
    use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;

    anyhow::ensure!(
        input_ids.len() >= 2,
        "vk_recompute_grpo_train_step: need at least 2 tokens"
    );
    anyhow::ensure!(
        active_rows.len() == labels.len(),
        "vk_recompute_grpo_train_step: active row count {} != label count {}",
        active_rows.len(),
        labels.len()
    );
    anyhow::ensure!(
        !active_rows.is_empty(),
        "vk_recompute_grpo_train_step: no active GRPO tokens"
    );
    for &row in active_rows {
        anyhow::ensure!(
            (row as usize) + 1 < input_ids.len(),
            "vk_recompute_grpo_train_step: active row {row} out of range for {} tokens",
            input_ids.len()
        );
    }

    let device = weights.embed_tokens.device();
    let detached_lora = detach_lora_layers(lora_layers);
    let gdn_map = gdn_layer_index_map(weights);
    let rope_tables = if !weights.rotary_inv_freq.is_empty() && weights.rotary_dim > 0 {
        Some(vk_compute_rope_tables(
            device,
            &weights.rotary_inv_freq,
            input_ids.len(),
        )?)
    } else {
        None
    };
    let rope_refs = rope_tables.as_ref().map(|(cos, sin)| (cos, sin));
    let profile = env_flag("KILN_PROFILE_VK_RECOMPUTE", false);
    let boundary_cache_limit = recompute_boundary_cache_limit_bytes();
    let boundary_cache_bytes = (weights.layers.len() + 1)
        .saturating_mul(input_ids.len())
        .saturating_mul(weights.hidden)
        .saturating_mul(std::mem::size_of::<f32>());
    let use_boundary_cache = kiln_core::env_flag::env_tristate("KILN_VK_RECOMPUTE_BOUNDARY_CACHE")
        .unwrap_or(true)
        && boundary_cache_limit > 0
        && boundary_cache_bytes <= boundary_cache_limit;

    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active_labels = active_rows.len(),
            boundary_cache = use_boundary_cache,
            boundary_cache_bytes,
            boundary_cache_limit,
            "vk-native GRPO recompute final forward begin"
        );
    }
    let (final_hidden, boundary_cache) = if use_boundary_cache {
        let (boundaries, _state) = vk_forward_layer_boundaries(
            weights,
            &detached_lora,
            input_ids,
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        let final_hidden = boundaries
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("vk_recompute_grpo_train_step: empty boundary cache"))?;
        (final_hidden, Some(boundaries))
    } else {
        let (final_hidden, _state) = vk_forward_to_layer_input(
            weights,
            &detached_lora,
            input_ids,
            weights.layers.len(),
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        (final_hidden, None)
    };
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            boundary_cache = use_boundary_cache,
            "vk-native GRPO recompute final forward done"
        );
    }

    let final_id = mint_fresh_tensor_id()?;
    let final_param = VkTensor::parameter(
        Arc::clone(final_hidden.buffer()),
        final_hidden.shape().to_vec(),
        final_hidden.dtype(),
        Arc::clone(final_hidden.device()),
        final_id,
    );
    let h_norm = vk_rmsnorm(&final_param, &weights.final_norm_weight, 1e-5)?;
    let active_h = vk_index_select_rows(&h_norm, active_rows)?;
    let loss = vk_grpo_loss(
        &active_h,
        &weights.lm_head,
        labels,
        ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        flce_recommended_chunk_len_for_tensors(&active_h, &weights.lm_head),
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active_labels = active_rows.len(),
            loss = format!("{loss_val:.6}"),
            "vk-native GRPO recompute finite loss computed"
        );
    }
    let final_grads = vk_backward(&loss)?;
    let mut upstream = final_grads.get(final_id).cloned().ok_or_else(|| {
        anyhow::anyhow!("vk_recompute_grpo_train_step: missing final upstream grad")
    })?;

    let mut shared_grads: HashMap<TensorId, VkTensor> = HashMap::new();

    for layer_idx in (0..weights.layers.len()).rev() {
        if profile {
            tracing::info!(
                step,
                layer_idx,
                seq_len = input_ids.len(),
                boundary_cache = use_boundary_cache,
                "vk-native GRPO recompute reverse layer begin"
            );
        }
        let (boundary, state) = if let Some(boundaries) = boundary_cache.as_ref() {
            let boundary = boundaries
                .get(layer_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing cached boundary for layer {layer_idx}"))?;
            let state =
                fresh_gdn_state(weights.embed_tokens.device(), model_config, num_gdn_layers)?;
            (boundary, state)
        } else {
            vk_forward_to_layer_input(
                weights,
                &detached_lora,
                input_ids,
                layer_idx,
                model_config,
                num_gdn_layers,
                &gdn_map,
                rope_refs,
            )
            .with_context(|| format!("vk_recompute_grpo_train_step: prefix to layer {layer_idx}"))?
        };
        if profile {
            tracing::info!(
                step,
                layer_idx,
                seq_len = input_ids.len(),
                boundary_cache = use_boundary_cache,
                "vk-native GRPO recompute reverse prefix done"
            );
        }
        match &weights.layers[layer_idx] {
            VkLayerWeights::FullAttention(full) => {
                let rope_arg = rope_refs.map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                let after_attn_value = vk_full_attention_attention_block_with_rope(
                    &boundary,
                    full,
                    &detached_lora[layer_idx],
                    rope_arg,
                )
                .with_context(|| {
                    format!("vk_recompute_grpo_train_step: full-attn prefix block {layer_idx}")
                })?;

                let gated_value = vk_full_attention_mlp_gated(
                    &after_attn_value,
                    full,
                    &detached_lora[layer_idx],
                )
                .with_context(|| {
                    format!("vk_recompute_grpo_train_step: full layer {layer_idx} MLP gated value")
                })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_full_attention_mlp_down_from_gated(
                    &gated_param,
                    full,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_grpo_train_step: backward full layer {layer_idx} MLP down"
                    )
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != gated_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_full_attention_mlp_gated(&after_param, full, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_grpo_train_step: backward full layer {layer_idx} MLP gate/up"
                    )
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for full layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid != after_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let boundary_id = mint_fresh_tensor_id()?;
                let h_param = VkTensor::parameter(
                    Arc::clone(boundary.buffer()),
                    boundary.shape().to_vec(),
                    boundary.dtype(),
                    Arc::clone(boundary.device()),
                    boundary_id,
                );
                let after_attn = vk_full_attention_attention_block_with_rope(
                    &h_param,
                    full,
                    &lora_layers[layer_idx],
                    rope_arg,
                )?;
                let prod = vk_mul(&after_attn, &upstream_after_attn)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_grpo_train_step: backward full layer {layer_idx} attention"
                    )
                })?;
                upstream = grads.get(boundary_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing boundary grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != boundary_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native GRPO recompute reverse full layer done"
                    );
                }
                continue;
            }
            VkLayerWeights::LinearAttention(linear) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let s = state
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GDN layer {layer_idx} requires state"))?;

                let after_attn_value = gdn_attention_block_value(
                    &boundary,
                    linear,
                    &detached_lora[layer_idx],
                    s,
                    gdn_idx,
                )
                .with_context(|| {
                    format!("vk_recompute_grpo_train_step: GDN layer {layer_idx} attention value")
                })?;

                let gated_value = vk_linear_attention_mlp_gated(
                    &after_attn_value,
                    linear,
                    &detached_lora[layer_idx],
                )
                .with_context(|| {
                    format!("vk_recompute_grpo_train_step: GDN layer {layer_idx} MLP gated value")
                })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_linear_attention_mlp_down_from_gated(
                    &gated_param,
                    linear,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_grpo_train_step: backward GDN layer {layer_idx} MLP down")
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for GDN layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != gated_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_linear_attention_mlp_gated(&after_param, linear, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_grpo_train_step: backward GDN layer {layer_idx} MLP gate/up"
                    )
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for GDN layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid != after_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                upstream = vk_gdn_layer_backward_split(
                    &boundary,
                    &upstream_after_attn,
                    linear,
                    &lora_layers[layer_idx],
                    s,
                    gdn_idx,
                    &mut shared_grads,
                )
                .with_context(|| {
                    format!("vk_recompute_grpo_train_step: split backward GDN layer {layer_idx}")
                })?;
                if profile {
                    tracing::info!(
                        step,
                        layer_idx,
                        seq_len = input_ids.len(),
                        "vk-native GRPO recompute reverse GDN layer done"
                    );
                }
                continue;
            }
        }
    }

    vk_optimizer_step_from_grads(
        weights.embed_tokens.device(),
        lora_layers,
        &shared_grads,
        adamw_state,
        cfg.lr,
        optimizer,
        step,
        "vk_recompute_grpo_train_step",
    )?;

    Ok(loss_val)
}

/// Exact layerwise reverse-recompute OPD step — the gradient-checkpointed
/// counterpart of `vk_opd_train_step_with_state`.
///
/// The non-checkpointed `vk_opd_train_step_with_state` keeps every layer's
/// activation tape alive through the OPD reverse-KL backward and OOMs at
/// long context. This variant mirrors `vk_recompute_grpo_train_step_with_state`
/// exactly:
///
/// 1. Forward through every layer with autograd OFF (using the detached
///    LoRA copy) to get `final_hidden`.
/// 2. Re-attach `final_hidden` as a fresh parameter, apply final RMSNorm
///    + index_select(active_rows), call `vk_opd_top_k_reverse_kl_loss`,
///    and backward to get the upstream gradient at `final_hidden`.
/// 3. Walk layers in REVERSE order; at each layer recompute the layer's
///    forward with autograd ON (against the live LoRA parameters) and
///    propagate the upstream gradient backward through that layer's
///    attention + MLP. Accumulate LoRA gradients into `shared_grads`
///    chunk-by-chunk so peak memory stays at one layer's tape, not 32.
/// 4. Single optimizer step at the end against `shared_grads`.
///
/// Mirrors the CUDA-side `opd_train` gradient-checkpointing pattern Eric
/// landed in `opd: gradient-checkpointed trainer + example binaries`.
/// Same `(teacher_topk_indices, teacher_topk_logprobs, top_k)` contract
/// as the non-checkpointed variant.
#[allow(clippy::too_many_arguments)]
pub fn vk_recompute_opd_train_step_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    active_rows: &[u32],
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
    model_config: &ModelConfig,
    num_gdn_layers: usize,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    optimizer: Optimizer,
    step: u32,
) -> Result<f32> {
    use kiln_model::vk_forward::{
        vk_compute_rope_tables, vk_full_attention_attention_block_with_rope,
        vk_full_attention_mlp_down_from_gated, vk_full_attention_mlp_gated,
        vk_linear_attention_mlp_down_from_gated, vk_linear_attention_mlp_gated,
    };
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::{vk_add_no_grad, vk_mul};
    use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;
    use kiln_vulkan_kernel::vk_ops::opd::vk_opd_top_k_reverse_kl_loss;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;

    anyhow::ensure!(
        input_ids.len() >= 2,
        "vk_recompute_opd_train_step: need at least 2 tokens"
    );
    anyhow::ensure!(
        !active_rows.is_empty(),
        "vk_recompute_opd_train_step: no active OPD tokens"
    );
    anyhow::ensure!(
        top_k == 16 || top_k == 32,
        "vk_recompute_opd_train_step: top_k must be 16 or 32 (got {top_k})"
    );
    let expected_lpq = active_rows.len() * top_k;
    anyhow::ensure!(
        teacher_topk_indices.len() == expected_lpq,
        "vk_recompute_opd_train_step: teacher_topk_indices.len() {} != active_rows * top_k = {}",
        teacher_topk_indices.len(),
        expected_lpq
    );
    anyhow::ensure!(
        teacher_topk_logprobs.len() == expected_lpq,
        "vk_recompute_opd_train_step: teacher_topk_logprobs.len() {} != active_rows * top_k = {}",
        teacher_topk_logprobs.len(),
        expected_lpq
    );
    for &row in active_rows {
        anyhow::ensure!(
            (row as usize) < input_ids.len(),
            "vk_recompute_opd_train_step: active row {row} out of range for {} tokens",
            input_ids.len()
        );
    }

    let device = weights.embed_tokens.device();
    let detached_lora = detach_lora_layers(lora_layers);
    let gdn_map = gdn_layer_index_map(weights);
    let rope_tables = if !weights.rotary_inv_freq.is_empty() && weights.rotary_dim > 0 {
        Some(vk_compute_rope_tables(
            device,
            &weights.rotary_inv_freq,
            input_ids.len(),
        )?)
    } else {
        None
    };
    let rope_refs = rope_tables.as_ref().map(|(cos, sin)| (cos, sin));
    let profile = env_flag("KILN_PROFILE_VK_RECOMPUTE", false);
    let boundary_cache_limit = recompute_boundary_cache_limit_bytes();
    let boundary_cache_bytes = (weights.layers.len() + 1)
        .saturating_mul(input_ids.len())
        .saturating_mul(weights.hidden)
        .saturating_mul(std::mem::size_of::<f32>());
    let use_boundary_cache = kiln_core::env_flag::env_tristate("KILN_VK_RECOMPUTE_BOUNDARY_CACHE")
        .unwrap_or(true)
        && boundary_cache_limit > 0
        && boundary_cache_bytes <= boundary_cache_limit;

    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active = active_rows.len(),
            top_k,
            boundary_cache = use_boundary_cache,
            "vk-native OPD recompute final forward begin"
        );
    }
    let (final_hidden, boundary_cache) = if use_boundary_cache {
        let (boundaries, _state) = vk_forward_layer_boundaries(
            weights,
            &detached_lora,
            input_ids,
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        let final_hidden = boundaries
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("vk_recompute_opd_train_step: empty boundary cache"))?;
        (final_hidden, Some(boundaries))
    } else {
        let (final_hidden, _state) = vk_forward_to_layer_input(
            weights,
            &detached_lora,
            input_ids,
            weights.layers.len(),
            model_config,
            num_gdn_layers,
            &gdn_map,
            rope_refs,
        )?;
        (final_hidden, None)
    };

    let final_id = mint_fresh_tensor_id()?;
    let final_param = VkTensor::parameter(
        Arc::clone(final_hidden.buffer()),
        final_hidden.shape().to_vec(),
        final_hidden.dtype(),
        Arc::clone(final_hidden.device()),
        final_id,
    );
    let h_norm = vk_rmsnorm(&final_param, &weights.final_norm_weight, 1e-5)?;
    let active_h = vk_index_select_rows(&h_norm, active_rows)?;
    let loss = vk_opd_top_k_reverse_kl_loss(
        &active_h,
        &weights.lm_head,
        teacher_topk_indices,
        teacher_topk_logprobs,
        top_k,
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    if profile {
        tracing::info!(
            step,
            seq_len = input_ids.len(),
            active = active_rows.len(),
            loss = format!("{loss_val:.6}"),
            "vk-native OPD recompute reverse-KL computed"
        );
    }
    let final_grads = vk_backward(&loss)?;
    let mut upstream = final_grads.get(final_id).cloned().ok_or_else(|| {
        anyhow::anyhow!("vk_recompute_opd_train_step: missing final upstream grad")
    })?;

    let mut shared_grads: HashMap<TensorId, VkTensor> = HashMap::new();

    for layer_idx in (0..weights.layers.len()).rev() {
        if profile {
            tracing::info!(
                step,
                layer_idx,
                seq_len = input_ids.len(),
                "vk-native OPD recompute reverse layer begin"
            );
        }
        let (boundary, state) = if let Some(boundaries) = boundary_cache.as_ref() {
            let boundary = boundaries
                .get(layer_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing cached boundary for layer {layer_idx}"))?;
            let state =
                fresh_gdn_state(weights.embed_tokens.device(), model_config, num_gdn_layers)?;
            (boundary, state)
        } else {
            vk_forward_to_layer_input(
                weights,
                &detached_lora,
                input_ids,
                layer_idx,
                model_config,
                num_gdn_layers,
                &gdn_map,
                rope_refs,
            )
            .with_context(|| format!("vk_recompute_opd_train_step: prefix to layer {layer_idx}"))?
        };
        match &weights.layers[layer_idx] {
            VkLayerWeights::FullAttention(full) => {
                let rope_arg = rope_refs.map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                let after_attn_value = vk_full_attention_attention_block_with_rope(
                    &boundary,
                    full,
                    &detached_lora[layer_idx],
                    rope_arg,
                )
                .with_context(|| {
                    format!("vk_recompute_opd_train_step: full-attn prefix block {layer_idx}")
                })?;

                let gated_value = vk_full_attention_mlp_gated(
                    &after_attn_value,
                    full,
                    &detached_lora[layer_idx],
                )
                .with_context(|| {
                    format!("vk_recompute_opd_train_step: full layer {layer_idx} MLP gated value")
                })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_full_attention_mlp_down_from_gated(
                    &gated_param,
                    full,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_opd_train_step: backward full layer {layer_idx} MLP down")
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != gated_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_full_attention_mlp_gated(&after_param, full, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_opd_train_step: backward full layer {layer_idx} MLP gate/up"
                    )
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for full layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid != after_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let boundary_id = mint_fresh_tensor_id()?;
                let h_param = VkTensor::parameter(
                    Arc::clone(boundary.buffer()),
                    boundary.shape().to_vec(),
                    boundary.dtype(),
                    Arc::clone(boundary.device()),
                    boundary_id,
                );
                let after_attn = vk_full_attention_attention_block_with_rope(
                    &h_param,
                    full,
                    &lora_layers[layer_idx],
                    rope_arg,
                )?;
                let prod = vk_mul(&after_attn, &upstream_after_attn)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_opd_train_step: backward full layer {layer_idx} attention")
                })?;
                upstream = grads.get(boundary_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing boundary grad for full layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != boundary_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }
                continue;
            }
            VkLayerWeights::LinearAttention(linear) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let s = state
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GDN layer {layer_idx} requires state"))?;

                let after_attn_value = gdn_attention_block_value(
                    &boundary,
                    linear,
                    &detached_lora[layer_idx],
                    s,
                    gdn_idx,
                )
                .with_context(|| {
                    format!("vk_recompute_opd_train_step: GDN layer {layer_idx} attention value")
                })?;

                let gated_value = vk_linear_attention_mlp_gated(
                    &after_attn_value,
                    linear,
                    &detached_lora[layer_idx],
                )
                .with_context(|| {
                    format!("vk_recompute_opd_train_step: GDN layer {layer_idx} MLP gated value")
                })?;

                let gated_id = mint_fresh_tensor_id()?;
                let gated_param = VkTensor::parameter(
                    Arc::clone(gated_value.buffer()),
                    gated_value.shape().to_vec(),
                    gated_value.dtype(),
                    Arc::clone(gated_value.device()),
                    gated_id,
                );
                let down_out = vk_linear_attention_mlp_down_from_gated(
                    &gated_param,
                    linear,
                    &lora_layers[layer_idx],
                )?;
                let prod = vk_mul(&down_out, &upstream)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!("vk_recompute_opd_train_step: backward GDN layer {layer_idx} MLP down")
                })?;
                let upstream_gated = grads.get(gated_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing gated grad for GDN layer {layer_idx}")
                })?;
                for (pid, grad) in grads.iter() {
                    if *pid != gated_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                let after_id = mint_fresh_tensor_id()?;
                let after_param = VkTensor::parameter(
                    Arc::clone(after_attn_value.buffer()),
                    after_attn_value.shape().to_vec(),
                    after_attn_value.dtype(),
                    Arc::clone(after_attn_value.device()),
                    after_id,
                );
                let gated =
                    vk_linear_attention_mlp_gated(&after_param, linear, &lora_layers[layer_idx])?;
                let prod = vk_mul(&gated, &upstream_gated)?;
                let scalar = vk_sum_all(&prod)?;
                let grads = vk_backward(&scalar).with_context(|| {
                    format!(
                        "vk_recompute_opd_train_step: backward GDN layer {layer_idx} MLP gate/up"
                    )
                })?;
                let mlp_grad_after = grads.get(after_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("missing after-attn MLP grad for GDN layer {layer_idx}")
                })?;
                let upstream_after_attn = vk_add_no_grad(&upstream, &mlp_grad_after)?;
                for (pid, grad) in grads.iter() {
                    if *pid != after_id {
                        accumulate_grad(&mut shared_grads, *pid, grad)?;
                    }
                }

                upstream = vk_gdn_layer_backward_split(
                    &boundary,
                    &upstream_after_attn,
                    linear,
                    &lora_layers[layer_idx],
                    s,
                    gdn_idx,
                    &mut shared_grads,
                )
                .with_context(|| {
                    format!("vk_recompute_opd_train_step: split backward GDN layer {layer_idx}")
                })?;
                continue;
            }
        }
    }

    vk_optimizer_step_from_grads(
        weights.embed_tokens.device(),
        lora_layers,
        &shared_grads,
        adamw_state,
        cfg.lr,
        optimizer,
        step,
        "vk_recompute_opd_train_step",
    )?;

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

/// SFT training step that honors the assistant-only label mask.
#[allow(clippy::too_many_arguments)]
pub fn vk_train_step_with_state_masked(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    label_mask: &[bool],
    gdn_state: Option<&mut VkLinearAttentionState>,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
) -> Result<f32> {
    let loss = vk_model_forward_loss_masked_with_state(
        weights,
        lora_layers,
        input_ids,
        label_mask,
        gdn_state,
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = vk_step_backward(&loss)?;

    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = grads.get(pid) else {
                continue;
            };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "vk_train_step_masked: AdamW F32 only for Phase F (got {:?}/{:?})",
                param.dtype(),
                grad.dtype()
            );
            anyhow::ensure!(
                param.num_elements() == grad.num_elements(),
                "vk_train_step_masked: param/grad element-count mismatch"
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
            .context("dispatch_adamw_step_f32 (masked)")?;
        }
    }

    Ok(loss_val)
}

#[allow(clippy::too_many_arguments)]
pub fn vk_grpo_train_step_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    active_rows: &[u32],
    labels: &[u32],
    ref_log_probs: &VkTensor,
    advantage: f32,
    clip_epsilon: f32,
    kl_coeff: f32,
    gdn_state: Option<&mut VkLinearAttentionState>,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    optimizer: Optimizer,
    step: u32,
) -> Result<f32> {
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::flce::{flce_recommended_chunk_len_for_tensors, vk_grpo_loss};

    let h = vk_model_forward_final_norm_with_state(weights, lora_layers, input_ids, gdn_state)?;
    let active_h = vk_index_select_rows(&h, active_rows)?;
    let loss = vk_grpo_loss(
        &active_h,
        &weights.lm_head,
        labels,
        ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        flce_recommended_chunk_len_for_tensors(&active_h, &weights.lm_head),
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = vk_backward(&loss)?;
    let mut shared_grads = HashMap::new();
    for (pid, grad) in grads.iter() {
        shared_grads.insert(*pid, grad.clone());
    }
    vk_optimizer_step_from_grads(
        weights.embed_tokens.device(),
        lora_layers,
        &shared_grads,
        adamw_state,
        cfg.lr,
        optimizer,
        step,
        "vk_grpo_train_step",
    )?;
    Ok(loss_val)
}

/// Vulkan-native OPD training step.
///
/// Mirrors `vk_grpo_train_step_with_state` but uses the fused OPD top-K
/// reverse-KL loss from `kiln_vulkan_kernel::vk_ops::opd` instead of the GRPO
/// importance-sampling head. One forward through the model → gather active
/// rows → fused-kernel forward+backward against the teacher's top-K → AdamW
/// (or SGD) on the LoRA adapter pairs.
///
/// Arguments:
/// - `active_rows`: row indices into `hidden` that contribute to the loss
///   (the trainer's "active" positions — typically assistant tokens). The
///   order must match the row order of `teacher_topk_indices` /
///   `teacher_topk_logprobs`.
/// - `teacher_topk_indices`: flattened `[active_rows.len() * top_k]` u32
///   teacher top-K vocab indices.
/// - `teacher_topk_logprobs`: flattened `[active_rows.len() * top_k]` f32
///   teacher logprobs at those indices (full-vocab `log_softmax`; the
///   kernel renormalises over the K support).
/// - `top_k`: 16 or 32 (the supported K envelope).
///
/// Returns the scalar mean reverse-KL after the optimizer step.
#[allow(clippy::too_many_arguments)]
pub fn vk_opd_train_step_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    active_rows: &[u32],
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
    gdn_state: Option<&mut VkLinearAttentionState>,
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    optimizer: Optimizer,
    step: u32,
) -> Result<f32> {
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::opd::vk_opd_top_k_reverse_kl_loss;

    let h = vk_model_forward_final_norm_with_state(weights, lora_layers, input_ids, gdn_state)?;
    let active_h = vk_index_select_rows(&h, active_rows)?;
    let loss = vk_opd_top_k_reverse_kl_loss(
        &active_h,
        &weights.lm_head,
        teacher_topk_indices,
        teacher_topk_logprobs,
        top_k,
    )?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = vk_backward(&loss)?;
    let mut shared_grads = HashMap::new();
    for (pid, grad) in grads.iter() {
        shared_grads.insert(*pid, grad.clone());
    }
    vk_optimizer_step_from_grads(
        weights.embed_tokens.device(),
        lora_layers,
        &shared_grads,
        adamw_state,
        cfg.lr,
        optimizer,
        step,
        "vk_opd_train_step",
    )?;
    Ok(loss_val)
}

// ---------------------------------------------------------------------------
// LoRA initialization (one VkLoraLayer per model layer)
// ---------------------------------------------------------------------------

/// Initialize LoRA params for every layer in the model.
///
/// Targets the canonical SFT modules on every attention block:
/// q/k/v/o + gate/up/down on FullAttention layers, and
/// in_proj_qkv/in_proj_z/out_proj + gate/up/down on LinearAttention (GDN)
/// layers.
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
            VkLayerWeights::LinearAttention(gdn) => {
                let dims = |name: &str, weight: &VkTensor| -> Result<(usize, usize)> {
                    let shape = weight.shape();
                    anyhow::ensure!(
                        shape.len() == 2,
                        "expected rank-2 {name} for layer {li}, got {:?}",
                        shape
                    );
                    Ok((shape[1], shape[0]))
                };
                let (qkv_in, qkv_out) = dims("in_proj_qkv", &gdn.in_proj_qkv)?;
                let (z_in, z_out) = dims("in_proj_z", &gdn.in_proj_z)?;
                let (out_in, out_out) = dims("out_proj", &gdn.out_proj)?;
                out.push(VkLoraLayer {
                    in_proj_qkv: Some(mk(8, qkv_in, qkv_out)?),
                    in_proj_z: Some(mk(9, z_in, z_out)?),
                    gdn_out_proj: Some(mk(10, out_in, out_out)?),
                    gate_proj: Some(mk(5, hidden, intermediate)?),
                    up_proj: Some(mk(6, hidden, intermediate)?),
                    down_proj: Some(mk(7, intermediate, hidden)?),
                    ..Default::default()
                });
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// vk-native SFT trainer (multi-epoch, single-step optimizer)
// ---------------------------------------------------------------------------

fn parse_grpo_jsonl_group_line(line: &str, line_no: usize) -> Result<Option<GrpoGroup>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<GrpoGroup>(trimmed)
        .map(Some)
        .with_context(|| format!("parse GRPO JSONL group at line {line_no}"))
}

/// Return `(groups, completions)` for a GRPO JSONL file without retaining the
/// parsed groups. Each non-empty line must be one `GrpoGroup`.
pub fn grpo_jsonl_stats(path: &Path) -> Result<(usize, usize)> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file =
        File::open(path).with_context(|| format!("open GRPO JSONL dataset {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut groups = 0usize;
    let mut completions = 0usize;
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "read GRPO JSONL dataset {} line {}",
                path.display(),
                idx + 1
            )
        })?;
        if let Some(group) = parse_grpo_jsonl_group_line(&line, idx + 1)? {
            groups += 1;
            completions += group.completions.len();
        }
    }
    anyhow::ensure!(
        groups > 0,
        "GRPO JSONL dataset {} has no groups",
        path.display()
    );
    anyhow::ensure!(
        completions > 0,
        "GRPO JSONL dataset {} has no completions",
        path.display()
    );
    Ok((groups, completions))
}

fn jsonl_byte_progress(total_bytes: u64, offset: u64) -> (usize, usize, f32) {
    let total = total_bytes.max(1);
    let clamped = offset.min(total);
    let total_steps = total.min(usize::MAX as u64).max(1) as usize;
    let step = clamped.min(usize::MAX as u64).max(1) as usize;
    let progress = (clamped as f64 / total as f64).min(0.999) as f32;
    (step, total_steps, progress)
}

/// Vulkan-native GRPO training loop.
///
/// This path keeps reference forward, selected-token logprob extraction,
/// GRPO loss/backward, gradient accumulation, optimizer updates, and adapter
/// save in the vk-native tensor stack. Policy backward uses exact layerwise
/// recompute so long completions do not require retaining the full forward
/// tape.
pub fn vk_native_grpo_train(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    anyhow::ensure!(!groups.is_empty(), "vk_native_grpo_train: no GRPO groups");
    if !VulkanDevice::probe() {
        bail!(
            "vk_native_grpo_train: no Vulkan device available - \
             unset KILN_VK_NATIVE_TRAINING to use the candle path"
        );
    }
    let vk_device = Arc::new(
        VulkanDevice::new().context("vk_native_grpo_train: failed to create Vulkan device")?,
    );

    // The vk-native GRPO step bakes forward + backward + optimizer step into a
    // single call (vk_recompute_grpo_train_step_with_state). Per-completion
    // stepping is therefore structurally tied to PerSample loss aggregation;
    // and the current shader supports K1 (kl_coeff scaled log_ratio) or No-KL
    // only, with symmetric clipping. Fall back / error early rather than
    // silently producing wrong gradients.
    if matches!(config.loss_aggregation, LossAggregation::TokenLevel) {
        anyhow::bail!(
            "vk-native GRPO does not yet support LossAggregation::TokenLevel; \
             use the candle path (unset KILN_VK_NATIVE_TRAINING) or set \
             loss_aggregation = per_sample"
        );
    }
    if matches!(config.kl_estimator, KlEstimator::K3) {
        anyhow::bail!(
            "vk-native GRPO does not yet support KlEstimator::K3; use \
             KlEstimator::K1 (default) or KlEstimator::None"
        );
    }
    if config.clip_eps_high.is_some_and(|hi| hi != config.clip_epsilon) {
        anyhow::bail!(
            "vk-native GRPO does not yet support asymmetric Clip-Higher; \
             use the candle path or leave clip_eps_high = None / equal to \
             clip_epsilon"
        );
    }

    let effective_seed = config.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x6752_504f)
    });

    let upload_start = std::time::Instant::now();
    tracing::info!("uploading frozen model weights into vk-native GRPO tensors");
    let vk_weights = VkModelWeights::from_gpu_weights(weights, model_config, &vk_device)
        .context("vk_native_grpo_train: VkModelWeights::from_gpu_weights")?;
    tracing::info!(
        elapsed_ms = upload_start.elapsed().as_millis() as u64,
        "vk-native GRPO frozen model weights ready"
    );
    let lora_layers = vk_init_lora_layers(
        &vk_device,
        &vk_weights,
        model_config,
        config.lora_rank,
        config.lora_alpha,
        effective_seed,
    )?;
    let mut adamw = allocate_adamw_state(&vk_device, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: config.learning_rate as f32,
        ..Default::default()
    };
    let num_gdn_layers = vk_count_gdn_layers(&vk_weights);
    let total_completions: usize = groups.iter().map(|g| g.completions.len()).sum();
    tracing::info!(
        num_groups = groups.len(),
        total_completions,
        lr = config.learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting vk-native GRPO training"
    );

    let mut optimizer_step = 0u32;
    let mut last_loss = 0.0f32;

    for (group_idx, group) in groups.iter().enumerate() {
        let group_step = group_idx + 1;
        if config.dynamic_sampling && is_degenerate_vk_grpo_group(group) {
            tracing::debug!(
                group = group_step,
                "vk-native GRPO dynamic sampling: skipping degenerate group"
            );
            continue;
        }
        let tgroup = tokenize_vk_grpo_group(group, tokenizer)
            .with_context(|| format!("tokenize GRPO group {group_step}"))?;
        validate_vk_grpo_tokenized_group_context(
            &tgroup,
            model_config,
            &format!("vk-native GRPO group {group_step}"),
        )?;
        let advantages = compute_vk_grpo_advantages(&tgroup.rewards, config.advantage_mode);
        let mut group_loss_sum = 0.0f64;
        let ref_prefix = if grpo_group_needs_prefix_reference(&tgroup) {
            Some(
                vk_grpo_reference_prefill_prompt(
                    &vk_weights,
                    &tgroup.prompt_ids,
                    model_config,
                    num_gdn_layers,
                )
                .with_context(|| {
                    format!(
                        "vk-native GRPO reference prompt prefill group {}",
                        group_step
                    )
                })?,
            )
        } else {
            tracing::info!(
                group = group_step,
                prompt_len = tgroup.prompt_ids.len(),
                "vk-native GRPO reference prompt prefill skipped for full-sequence scoring"
            );
            None
        };

        for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
            optimizer_step += 1;
            let (active_rows, labels) =
                grpo_active_rows_and_labels(&comp.input_ids, &comp.completion_mask)?;
            ensure_grpo_completion_scoring_layout(tgroup.prompt_ids.len(), &active_rows)?;
            let (ref_log_probs, _reference_path) = vk_grpo_reference_log_probs_dynamic(
                &vk_weights,
                ref_prefix.as_ref(),
                &comp.input_ids,
                &active_rows,
                &labels,
                model_config,
                num_gdn_layers,
                tgroup.prompt_ids.len(),
                tgroup.completions.len(),
            )
            .with_context(|| {
                format!(
                    "vk-native GRPO reference logprobs group {} completion {}",
                    group_step,
                    comp_idx + 1
                )
            })?;

            let loss = vk_recompute_grpo_train_step_with_state(
                &vk_weights,
                &lora_layers,
                &comp.input_ids,
                &active_rows,
                &labels,
                &ref_log_probs,
                advantages[comp_idx] as f32,
                config.clip_epsilon as f32,
                vk_effective_kl_coeff(config) as f32,
                model_config,
                num_gdn_layers,
                &mut adamw,
                &cfg,
                config.optimizer,
                optimizer_step,
            )
            .with_context(|| {
                format!(
                    "vk-native GRPO policy step group {} completion {}",
                    group_step,
                    comp_idx + 1
                )
            })?;
            anyhow::ensure!(
                loss.is_finite(),
                "vk_native_grpo_train: non-finite loss {loss} at optimizer step {optimizer_step}"
            );
            last_loss = loss;
            group_loss_sum += loss as f64;
        }

        let avg_group_loss = if tgroup.completions.is_empty() {
            0.0
        } else {
            group_loss_sum / tgroup.completions.len() as f64
        };
        if let Some(ref cb) = progress_cb {
            cb(TrainingProgress {
                epoch: 1,
                total_epochs: 1,
                step: group_step,
                total_steps: groups.len(),
                loss: avg_group_loss,
                progress: group_step as f32 / groups.len().max(1) as f32,
            });
        }
        tracing::info!(
            group = group_step,
            total_groups = groups.len(),
            completions = tgroup.completions.len(),
            loss = format!("{avg_group_loss:.6}"),
            "vk-native GRPO group step"
        );

        if let Some(interval) = config.checkpoint_interval {
            if interval > 0 && group_step % interval == 0 && group_step < groups.len() {
                let ckpt_dir = adapter_dir.join(format!("{adapter_name}-checkpoint-{group_step}"));
                std::fs::create_dir_all(&ckpt_dir).with_context(|| {
                    format!(
                        "create vk-native GRPO checkpoint dir {}",
                        ckpt_dir.display()
                    )
                })?;
                save_vk_lora_adapter(
                    &lora_layers,
                    config.lora_rank,
                    config.lora_alpha,
                    &ckpt_dir.join("adapter_model.safetensors"),
                )?;
                write_vk_adapter_config(&ckpt_dir, config.lora_rank, config.lora_alpha)?;
            }
        }
    }

    let output_dir = adapter_dir.join(adapter_name);
    std::fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "vk_native_grpo_train: create adapter dir {}",
            output_dir.display()
        )
    })?;
    save_vk_lora_adapter(
        &lora_layers,
        config.lora_rank,
        config.lora_alpha,
        &output_dir.join("adapter_model.safetensors"),
    )?;
    write_vk_adapter_config(&output_dir, config.lora_rank, config.lora_alpha)?;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        "vk-native GRPO training complete"
    );

    Ok(output_dir)
}

/// Vulkan-native GRPO training loop over a JSONL dataset.
///
/// The file is streamed one non-empty line at a time. Each line is a
/// `GrpoGroup`; no vector of all groups is retained during tokenization,
/// reference forward, policy loss, backward, or optimizer update.
pub fn vk_native_grpo_train_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    if !VulkanDevice::probe() {
        bail!(
            "vk_native_grpo_train_jsonl: no Vulkan device available - \
             unset KILN_VK_NATIVE_TRAINING to use the candle path"
        );
    }
    let vk_device = Arc::new(
        VulkanDevice::new()
            .context("vk_native_grpo_train_jsonl: failed to create Vulkan device")?,
    );

    if matches!(config.loss_aggregation, LossAggregation::TokenLevel) {
        anyhow::bail!(
            "vk-native GRPO does not yet support LossAggregation::TokenLevel; \
             use the candle path (unset KILN_VK_NATIVE_TRAINING) or set \
             loss_aggregation = per_sample"
        );
    }
    if matches!(config.kl_estimator, KlEstimator::K3) {
        anyhow::bail!(
            "vk-native GRPO does not yet support KlEstimator::K3; use \
             KlEstimator::K1 (default) or KlEstimator::None"
        );
    }
    if config.clip_eps_high.is_some_and(|hi| hi != config.clip_epsilon) {
        anyhow::bail!(
            "vk-native GRPO does not yet support asymmetric Clip-Higher; \
             use the candle path or leave clip_eps_high = None / equal to \
             clip_epsilon"
        );
    }

    let effective_seed = config.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x6752_504f)
    });

    let upload_start = std::time::Instant::now();
    tracing::info!("uploading frozen model weights into streamed vk-native GRPO tensors");
    let vk_weights = VkModelWeights::from_gpu_weights(weights, model_config, &vk_device)
        .context("vk_native_grpo_train_jsonl: VkModelWeights::from_gpu_weights")?;
    tracing::info!(
        elapsed_ms = upload_start.elapsed().as_millis() as u64,
        "streamed vk-native GRPO frozen model weights ready"
    );
    let lora_layers = vk_init_lora_layers(
        &vk_device,
        &vk_weights,
        model_config,
        config.lora_rank,
        config.lora_alpha,
        effective_seed,
    )?;
    let mut adamw = allocate_adamw_state(&vk_device, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: config.learning_rate as f32,
        ..Default::default()
    };
    let num_gdn_layers = vk_count_gdn_layers(&vk_weights);
    let file = File::open(dataset_path)
        .with_context(|| format!("open GRPO JSONL dataset {}", dataset_path.display()))?;
    let total_bytes = file.metadata().map(|m| m.len()).unwrap_or(0).max(1);
    tracing::info!(
        dataset = %dataset_path.display(),
        total_bytes,
        lr = config.learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting streamed vk-native GRPO training"
    );

    let mut reader = BufReader::new(file);
    let mut optimizer_step = 0u32;
    let mut last_loss = 0.0f32;
    let mut processed_groups = 0usize;
    let mut processed_completions = 0usize;
    let mut bytes_read = 0u64;
    let mut line_no = 0usize;
    let mut line = String::new();

    loop {
        line.clear();
        let line_start = bytes_read;
        let read = reader.read_line(&mut line).with_context(|| {
            format!(
                "read GRPO JSONL dataset {} line {}",
                dataset_path.display(),
                line_no + 1
            )
        })?;
        if read == 0 {
            break;
        }
        line_no += 1;
        bytes_read = bytes_read.saturating_add(read as u64);
        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        if config.dynamic_sampling && is_degenerate_vk_grpo_group(&group) {
            tracing::debug!(
                line = line_no,
                "streamed vk-native GRPO dynamic sampling: skipping degenerate group"
            );
            continue;
        }
        processed_groups += 1;
        tracing::info!(
            group = processed_groups,
            line = line_no,
            line_bytes = read,
            byte_offset = line_start,
            "streamed vk-native GRPO group tokenization begin"
        );
        let tokenize_start = std::time::Instant::now();
        let tgroup = tokenize_vk_grpo_group(&group, tokenizer).with_context(|| {
            format!(
                "tokenize GRPO JSONL group {} at line {}",
                processed_groups, line_no
            )
        })?;
        validate_vk_grpo_tokenized_group_context(
            &tgroup,
            model_config,
            &format!("vk-native GRPO JSONL group {processed_groups} line {line_no}"),
        )?;
        tracing::info!(
            group = processed_groups,
            completions = tgroup.completions.len(),
            seq_lens = ?tgroup
                .completions
                .iter()
                .map(|c| c.input_ids.len())
                .collect::<Vec<_>>(),
            elapsed_ms = tokenize_start.elapsed().as_millis() as u64,
            "streamed vk-native GRPO group tokenized"
        );
        let advantages = compute_vk_grpo_advantages(&tgroup.rewards, config.advantage_mode);
        let mut group_loss_sum = 0.0f64;
        let ref_prefix = if grpo_group_needs_prefix_reference(&tgroup) {
            tracing::info!(
                group = processed_groups,
                prompt_len = tgroup.prompt_ids.len(),
                "streamed vk-native GRPO reference prompt prefill begin"
            );
            let ref_prefix_start = std::time::Instant::now();
            let ref_prefix = vk_grpo_reference_prefill_prompt(
                &vk_weights,
                &tgroup.prompt_ids,
                model_config,
                num_gdn_layers,
            )
            .with_context(|| {
                format!(
                    "vk-native GRPO JSONL reference prompt prefill group {}",
                    processed_groups
                )
            })?;
            tracing::info!(
                group = processed_groups,
                prompt_len = tgroup.prompt_ids.len(),
                elapsed_ms = ref_prefix_start.elapsed().as_millis() as u64,
                "streamed vk-native GRPO reference prompt prefill done"
            );
            Some(ref_prefix)
        } else {
            tracing::info!(
                group = processed_groups,
                prompt_len = tgroup.prompt_ids.len(),
                "streamed vk-native GRPO reference prompt prefill skipped for full-sequence scoring"
            );
            None
        };

        let group_substeps = tgroup.completions.len().saturating_mul(2).max(1);
        let line_span = bytes_read.saturating_sub(line_start);
        for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
            optimizer_step += 1;
            let (active_rows, labels) =
                grpo_active_rows_and_labels(&comp.input_ids, &comp.completion_mask)?;
            ensure_grpo_completion_scoring_layout(tgroup.prompt_ids.len(), &active_rows)?;
            let completed_before = comp_idx.saturating_mul(2);
            let progress_offset = line_start.saturating_add(
                line_span.saturating_mul(completed_before as u64) / group_substeps as u64,
            );
            let (step, total_steps, progress) = jsonl_byte_progress(total_bytes, progress_offset);
            if let Some(ref cb) = progress_cb {
                cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step,
                    total_steps,
                    loss: last_loss as f64,
                    progress,
                });
            }
            tracing::info!(
                group = processed_groups,
                completion = comp_idx + 1,
                completions = tgroup.completions.len(),
                optimizer_step,
                seq_len = comp.input_ids.len(),
                active_labels = labels.len(),
                reference_path = if grpo_use_prefix_reference(
                    tgroup.prompt_ids.len(),
                    labels.len(),
                    tgroup.completions.len(),
                ) {
                    "prefix_decode"
                } else {
                    "full_sequence"
                },
                "streamed vk-native GRPO reference logprobs begin"
            );
            let ref_start = std::time::Instant::now();
            let (ref_log_probs, reference_path) = vk_grpo_reference_log_probs_dynamic(
                &vk_weights,
                ref_prefix.as_ref(),
                &comp.input_ids,
                &active_rows,
                &labels,
                model_config,
                num_gdn_layers,
                tgroup.prompt_ids.len(),
                tgroup.completions.len(),
            )
            .with_context(|| {
                format!(
                    "vk-native GRPO JSONL reference logprobs group {} completion {}",
                    processed_groups,
                    comp_idx + 1
                )
            })?;
            tracing::info!(
                group = processed_groups,
                completion = comp_idx + 1,
                optimizer_step,
                seq_len = comp.input_ids.len(),
                active_labels = labels.len(),
                reference_path,
                elapsed_ms = ref_start.elapsed().as_millis() as u64,
                "streamed vk-native GRPO reference logprobs done"
            );
            if let Some(ref cb) = progress_cb {
                let completed = completed_before + 1;
                let progress_offset = line_start.saturating_add(
                    line_span.saturating_mul(completed as u64) / group_substeps as u64,
                );
                let (step, total_steps, progress) =
                    jsonl_byte_progress(total_bytes, progress_offset);
                cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step,
                    total_steps,
                    loss: last_loss as f64,
                    progress,
                });
            }

            tracing::info!(
                group = processed_groups,
                completion = comp_idx + 1,
                completions = tgroup.completions.len(),
                optimizer_step,
                seq_len = comp.input_ids.len(),
                active_labels = labels.len(),
                "streamed vk-native GRPO policy step begin"
            );
            let policy_start = std::time::Instant::now();
            let loss = vk_recompute_grpo_train_step_with_state(
                &vk_weights,
                &lora_layers,
                &comp.input_ids,
                &active_rows,
                &labels,
                &ref_log_probs,
                advantages[comp_idx] as f32,
                config.clip_epsilon as f32,
                vk_effective_kl_coeff(config) as f32,
                model_config,
                num_gdn_layers,
                &mut adamw,
                &cfg,
                config.optimizer,
                optimizer_step,
            )
            .with_context(|| {
                format!(
                    "vk-native GRPO JSONL policy step group {} completion {}",
                    processed_groups,
                    comp_idx + 1
                )
            })?;
            tracing::info!(
                group = processed_groups,
                completion = comp_idx + 1,
                optimizer_step,
                seq_len = comp.input_ids.len(),
                active_labels = labels.len(),
                loss = format!("{loss:.6}"),
                elapsed_ms = policy_start.elapsed().as_millis() as u64,
                "streamed vk-native GRPO policy step done"
            );
            anyhow::ensure!(
                loss.is_finite(),
                "vk_native_grpo_train_jsonl: non-finite loss {loss} at optimizer step {optimizer_step}"
            );
            last_loss = loss;
            group_loss_sum += loss as f64;
            if let Some(ref cb) = progress_cb {
                let completed = completed_before + 2;
                let progress_offset = line_start.saturating_add(
                    line_span.saturating_mul(completed as u64) / group_substeps as u64,
                );
                let (step, total_steps, progress) =
                    jsonl_byte_progress(total_bytes, progress_offset);
                cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step,
                    total_steps,
                    loss: last_loss as f64,
                    progress,
                });
            }
        }
        processed_completions = processed_completions.saturating_add(tgroup.completions.len());

        let avg_group_loss = if tgroup.completions.is_empty() {
            0.0
        } else {
            group_loss_sum / tgroup.completions.len() as f64
        };
        if let Some(ref cb) = progress_cb {
            let (step, total_steps, progress) = jsonl_byte_progress(total_bytes, bytes_read);
            cb(TrainingProgress {
                epoch: 1,
                total_epochs: 1,
                step,
                total_steps,
                loss: avg_group_loss,
                progress,
            });
        }
        tracing::info!(
            group = processed_groups,
            completions_seen = processed_completions,
            completions = tgroup.completions.len(),
            byte_offset = bytes_read,
            total_bytes,
            loss = format!("{avg_group_loss:.6}"),
            "streamed vk-native GRPO group step"
        );

        if let Some(interval) = config.checkpoint_interval {
            if interval > 0 && processed_groups % interval == 0 && bytes_read < total_bytes {
                let ckpt_dir =
                    adapter_dir.join(format!("{adapter_name}-checkpoint-{processed_groups}"));
                std::fs::create_dir_all(&ckpt_dir).with_context(|| {
                    format!(
                        "create streamed vk-native GRPO checkpoint dir {}",
                        ckpt_dir.display()
                    )
                })?;
                save_vk_lora_adapter(
                    &lora_layers,
                    config.lora_rank,
                    config.lora_alpha,
                    &ckpt_dir.join("adapter_model.safetensors"),
                )?;
                write_vk_adapter_config(&ckpt_dir, config.lora_rank, config.lora_alpha)?;
            }
        }
    }

    anyhow::ensure!(
        processed_groups > 0 && optimizer_step > 0,
        "vk_native_grpo_train_jsonl: no valid GRPO groups in {}",
        dataset_path.display()
    );

    let output_dir = adapter_dir.join(adapter_name);
    std::fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "vk_native_grpo_train_jsonl: create adapter dir {}",
            output_dir.display()
        )
    })?;
    save_vk_lora_adapter(
        &lora_layers,
        config.lora_rank,
        config.lora_alpha,
        &output_dir.join("adapter_model.safetensors"),
    )?;
    write_vk_adapter_config(&output_dir, config.lora_rank, config.lora_alpha)?;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        processed_groups,
        processed_completions,
        "streamed vk-native GRPO training complete"
    );

    Ok(output_dir)
}

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

    // Tokenize all examples up front. The Vulkan-native path never truncates
    // or windows examples; length sorting only changes step order so long
    // runs reach finite progress on shorter full-context examples before the
    // longest O(T^2) attention examples.
    let mut tokenized: Vec<TokenizedSftExample> = examples
        .iter()
        .enumerate()
        .filter_map(
            |(original_index, ex)| match tokenize_for_training(ex, tokenizer) {
                Ok((input_ids, label_mask)) => Some(TokenizedSftExample {
                    input_ids,
                    label_mask,
                    original_index,
                }),
                Err(e) => {
                    tracing::warn!("vk_native: skipping example: {e}");
                    None
                }
            },
        )
        .collect();
    if tokenized.is_empty() {
        bail!("vk_native_sft_train: no valid training examples after tokenization");
    }
    let min_seq_len = tokenized
        .iter()
        .map(|ex| ex.input_ids.len())
        .min()
        .unwrap_or(0);
    let max_seq_len = tokenized
        .iter()
        .map(|ex| ex.input_ids.len())
        .max()
        .unwrap_or(0);
    let length_sorted = kiln_core::env_flag::env_tristate("KILN_VK_LENGTH_SORT_SFT")
        .unwrap_or(max_seq_len >= 8192 && tokenized.len() > 1);
    if length_sorted {
        tokenized.sort_by_key(|ex| (ex.input_ids.len(), ex.original_index));
    }
    tracing::info!(
        examples = tokenized.len(),
        min_seq_len,
        max_seq_len,
        first_seq_len = tokenized[0].input_ids.len(),
        first_original_index = tokenized[0].original_index,
        length_sorted,
        "vk-native tokenized full SFT examples"
    );

    let total_steps = config.epochs * tokenized.len();
    let mut global_step: u32 = 0;
    let mut last_loss = 0.0f32;

    // Pre-compute GDN state shape from model_config for per-example
    // allocation. For hybrid Qwen3.5-4B this allocates state for the
    // 24 GDN layers.
    let num_gdn_layers = vk_count_gdn_layers(&vk_weights);
    let needs_state = num_gdn_layers > 0;

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
        if segs > 1 { Some(segs) } else { None }
    };
    if let Some(segs) = ckpt_segments {
        tracing::info!(
            num_segments = segs,
            "vk-native gradient checkpointing enabled"
        );
    } else if needs_state {
        tracing::info!("vk-native exact layerwise recompute enabled for hybrid GDN model");
    } else {
        tracing::info!("vk-native gradient checkpointing disabled");
    }

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        for ex in tokenized.iter() {
            global_step += 1;
            let input_ids = &ex.input_ids;
            let label_mask = &ex.label_mask;
            tracing::info!(
                epoch = epoch + 1,
                step = global_step,
                total_steps,
                seq_len = input_ids.len(),
                original_index = ex.original_index,
                "vk-native SFT step begin"
            );

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
            } else if needs_state {
                vk_recompute_train_step_with_state_masked(
                    &vk_weights,
                    &lora_layers,
                    input_ids,
                    label_mask,
                    model_config,
                    num_gdn_layers,
                    &mut adamw,
                    &cfg,
                    global_step,
                )
                .with_context(|| {
                    format!(
                        "vk_recompute_train_step at epoch {} step {}",
                        epoch + 1,
                        global_step
                    )
                })?
            } else {
                vk_train_step_with_state_masked(
                    &vk_weights,
                    &lora_layers,
                    input_ids,
                    label_mask,
                    None,
                    &mut adamw,
                    &cfg,
                    global_step,
                )
                .with_context(|| {
                    format!("vk_train_step at epoch {} step {}", epoch + 1, global_step)
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
    save_vk_lora_adapter(
        &lora_layers,
        config.lora_rank,
        config.lora_alpha,
        &adapter_path,
    )
    .with_context(|| format!("save final adapter to {}", adapter_path.display()))?;

    // Write a minimal adapter_config.json mirroring trainer::save_peft.
    write_vk_adapter_config(&output_dir, config.lora_rank, config.lora_alpha)?;

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
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "out_proj"
        ],
    });
    let path = output_dir.join("adapter_config.json");
    let s = serde_json::to_string_pretty(&cfg)
        .with_context(|| format!("serialize adapter_config.json {}", path.display()))?;
    std::fs::write(&path, s)
        .with_context(|| format!("write adapter_config.json {}", path.display()))?;
    Ok(())
}

/// Save LoRA adapter to safetensors. Each VkTensor is read back to CPU once
/// for file serialization, but no Candle tensors or autograd path are used.
pub fn save_vk_lora_adapter(
    lora_layers: &[VkLoraLayer],
    rank: usize,
    alpha: f32,
    output_path: &std::path::Path,
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    let mut byte_storage: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut push_tensor = |name: String, tensor: &VkTensor| -> Result<()> {
        let data = tensor
            .to_vec_f32()
            .with_context(|| format!("read back Vulkan adapter tensor {name}"))?;
        let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f32>());
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        byte_storage.push((name, tensor.shape().to_vec(), bytes));
        Ok(())
    };

    for (li, layer) in lora_layers.iter().enumerate() {
        for (submodule, name, proj) in [
            ("self_attn", "q_proj", layer.q_proj.as_ref()),
            ("self_attn", "k_proj", layer.k_proj.as_ref()),
            ("self_attn", "v_proj", layer.v_proj.as_ref()),
            ("self_attn", "o_proj", layer.o_proj.as_ref()),
            ("mlp", "gate_proj", layer.gate_proj.as_ref()),
            ("mlp", "up_proj", layer.up_proj.as_ref()),
            ("mlp", "down_proj", layer.down_proj.as_ref()),
            ("self_attn", "in_proj_qkv", layer.in_proj_qkv.as_ref()),
            ("self_attn", "in_proj_z", layer.in_proj_z.as_ref()),
            ("self_attn", "out_proj", layer.gdn_out_proj.as_ref()),
        ] {
            let Some(p) = proj else { continue };
            push_tensor(
                format!(
                    "base_model.model.model.layers.{}.{}.{}.lora_A.weight",
                    li, submodule, name
                ),
                &p.a,
            )?;
            push_tensor(
                format!(
                    "base_model.model.model.layers.{}.{}.{}.lora_B.weight",
                    li, submodule, name
                ),
                &p.b,
            )?;
        }
    }
    drop(push_tensor);

    let views: Vec<(String, TensorView<'_>)> = byte_storage
        .iter()
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| anyhow::anyhow!("building safetensors view for {name}: {e}"))?;
            Ok::<_, anyhow::Error>((name.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<(&str, TensorView<'_>)> = views
        .iter()
        .map(|(name, view)| (name.as_str(), view.clone()))
        .collect();
    let serialized =
        safetensors::tensor::serialize(refs, None).context("serialize Vulkan LoRA safetensors")?;
    std::fs::write(output_path, serialized)
        .with_context(|| format!("save_vk_lora_adapter: {}", output_path.display()))?;
    let _ = (rank, alpha); // adapter_config.json could be written here if desired
    Ok(())
}

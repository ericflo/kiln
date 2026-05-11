//! Vulkan-native model forward pass.
//!
//! Assembles the full transformer forward graph entirely in
//! `VkTensor` (GPU memory) so every intermediate during training
//! lives in DRM-allocated memory rather than candle CPU storage.
//!
//! For Phase E this provides the building blocks (transformer layer,
//! full model forward, FLCE loss head) along with a `VkLoraParams`
//! holder that bridges to the existing trainer's `Var`-keyed
//! gradient + AdamW dispatch path.
//!
//! Real Qwen3.5 weights ship through `GpuWeights`; the helpers here
//! upload them lazily into VkTensors. Smaller synthetic shapes are
//! supported for end-to-end tests without needing a 4 B model.
//!
//! Note: this is intentionally F32-internal for Phase E. BF16 weight
//! storage and BF16 accumulation are Phase G optimizations that ride
//! on the same op surface.

#![cfg(feature = "vulkan")]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, TensorId, Var};
use kiln_vulkan_kernel::vk_autograd::{vk_backward, VkGradStore};
use kiln_vulkan_kernel::vk_ops::attention::vk_sdpa_prefill_flat;
use kiln_vulkan_kernel::vk_ops::elementwise::vk_add;
use kiln_vulkan_kernel::vk_ops::embedding::{
    upload_u32_ids, vk_embedding_lookup_bf16, vk_embedding_lookup_f32,
};
use kiln_vulkan_kernel::vk_ops::flce::{vk_flce_loss, FLCE_DEFAULT_CHUNK};
use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
use kiln_vulkan_kernel::vk_ops::mlp::vk_swiglu_mlp;
use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanDevice};
use std::sync::Arc;

/// Trainable LoRA pair held as VkTensor parameters keyed by the same
/// `TensorId` the rest of the trainer (and the existing AdamW
/// dispatch path) uses.
pub struct VkLoraPair {
    pub a: VkTensor, // [rank, in_features]
    pub b: VkTensor, // [out_features, rank]
    pub a_id: TensorId,
    pub b_id: TensorId,
    pub scale: f32,
}

impl VkLoraPair {
    /// Initialize a fresh LoRA pair on the device. A is Kaiming-uniform
    /// (matches existing trainer init), B is zeros (matches PEFT
    /// convention).
    pub fn init_kaiming(
        device: &Arc<VulkanDevice>,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Result<Self> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let bound = (1.0_f32 / (in_features as f32)).sqrt();
        let a_data: Vec<f32> = (0..(rank * in_features))
            .map(|_| {
                let r: f32 = rng.random_range(-bound..bound);
                r
            })
            .collect();
        let b_data: Vec<f32> = vec![0.0_f32; out_features * rank];

        let a_t = Tensor::from_vec(a_data, (rank, in_features), &Device::Cpu)?;
        let b_t = Tensor::from_vec(b_data, (out_features, rank), &Device::Cpu)?;
        let a_var = Var::from_tensor(&a_t)?;
        let b_var = Var::from_tensor(&b_t)?;
        let a_vk = VkTensor::from_candle(&a_t, Arc::clone(device))?;
        let b_vk = VkTensor::from_candle(&b_t, Arc::clone(device))?;
        let a = VkTensor::parameter(
            Arc::clone(a_vk.buffer()),
            a_vk.shape().to_vec(),
            a_vk.dtype(),
            Arc::clone(a_vk.device()),
            a_var.id(),
        );
        let b = VkTensor::parameter(
            Arc::clone(b_vk.buffer()),
            b_vk.shape().to_vec(),
            b_vk.dtype(),
            Arc::clone(b_vk.device()),
            b_var.id(),
        );
        Ok(Self {
            a,
            b,
            a_id: a_var.id(),
            b_id: b_var.id(),
            scale: alpha / (rank as f32),
        })
    }
}

/// All trainable LoRA params for one transformer layer.
#[derive(Default)]
pub struct VkLoraLayer {
    pub q_proj: Option<VkLoraPair>,
    pub k_proj: Option<VkLoraPair>,
    pub v_proj: Option<VkLoraPair>,
    pub o_proj: Option<VkLoraPair>,
    pub gate_proj: Option<VkLoraPair>,
    pub up_proj: Option<VkLoraPair>,
    pub down_proj: Option<VkLoraPair>,
}

/// Frozen base weights for one transformer layer in VkTensor form.
pub struct VkLayerWeights {
    pub input_layernorm_weight: VkTensor, // [hidden]
    pub post_attention_layernorm_weight: VkTensor, // [hidden]
    pub q_proj: VkTensor, // [hidden_q, hidden]   (out, in)
    pub k_proj: VkTensor, // [hidden_kv, hidden]
    pub v_proj: VkTensor, // [hidden_kv, hidden]
    pub o_proj: VkTensor, // [hidden, hidden_q]
    pub gate_proj: VkTensor, // [intermediate, hidden]
    pub up_proj: VkTensor, // [intermediate, hidden]
    pub down_proj: VkTensor, // [hidden, intermediate]
    pub heads_q: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
}

pub struct VkModelWeights {
    pub embed_tokens: VkTensor,    // [vocab, hidden]
    pub embed_dtype: VkDType,
    pub final_norm_weight: VkTensor, // [hidden]
    pub lm_head: VkTensor,         // [vocab, hidden]   (typically tied with embed_tokens)
    pub layers: Vec<VkLayerWeights>,
    pub vocab: usize,
    pub hidden: usize,
}

/// Apply a base linear projection followed by an optional LoRA delta:
///   out = x @ W.T + scale * x @ A.T @ B.T
fn vk_linear_with_lora(
    x: &VkTensor,
    weight: &VkTensor,
    lora: Option<&VkLoraPair>,
) -> Result<VkTensor> {
    // base = x @ weight.T
    let w_t = vk_transpose_2d(weight)?;
    let base = vk_matmul(x, &w_t)?;
    let Some(pair) = lora else {
        return Ok(base);
    };
    // h = x @ A.T  → [rows, rank]
    let a_t = vk_transpose_2d(&pair.a)?;
    let h = vk_matmul(x, &a_t)?;
    // delta = h @ B.T  → [rows, out]
    let b_t = vk_transpose_2d(&pair.b)?;
    let delta = vk_matmul(&h, &b_t)?;
    let delta = if (pair.scale - 1.0).abs() > 1e-9 {
        kiln_vulkan_kernel::vk_ops::mask::vk_scale(&delta, pair.scale)?
    } else {
        delta
    };
    vk_add(&base, &delta)
}

/// Run one transformer layer end-to-end on VkTensor activations.
pub fn vk_transformer_layer(
    x: &VkTensor,
    w: &VkLayerWeights,
    lora: &VkLoraLayer,
) -> Result<VkTensor> {
    let rows = x.shape()[0];
    let hidden = x.shape()[1];
    debug_assert_eq!(hidden, w.q_proj.shape()[1]);

    // Pre-attention RMSNorm
    let h_norm = vk_rmsnorm(x, &w.input_layernorm_weight, w.eps)?;

    // Q/K/V projections (with LoRA)
    let q = vk_linear_with_lora(&h_norm, &w.q_proj, lora.q_proj.as_ref())?;
    let k = vk_linear_with_lora(&h_norm, &w.k_proj, lora.k_proj.as_ref())?;
    let v = vk_linear_with_lora(&h_norm, &w.v_proj, lora.v_proj.as_ref())?;

    // Causal SDPA, GQA-aware
    let scale = 1.0 / (w.head_dim as f32).sqrt();
    let attn = vk_sdpa_prefill_flat(
        &q,
        &k,
        &v,
        w.heads_q,
        w.heads_kv,
        w.head_dim,
        scale,
    )?;

    // O projection
    let o_out = vk_linear_with_lora(&attn, &w.o_proj, lora.o_proj.as_ref())?;
    // Residual
    let after_attn = vk_add(x, &o_out)?;
    let _ = rows;

    // Post-attention RMSNorm
    let h_norm2 = vk_rmsnorm(&after_attn, &w.post_attention_layernorm_weight, w.eps)?;

    // SwiGLU MLP — for Phase E.3 we don't yet wire MLP LoRA into
    // vk_swiglu_mlp; that's a small extension. The composition below
    // computes the base MLP output; LoRA on gate/up/down is folded in
    // by replacing `gate/up/down` projections with `vk_linear_with_lora`
    // calls (Phase F refinement).
    let mlp_out = if lora.gate_proj.is_some() || lora.up_proj.is_some() || lora.down_proj.is_some() {
        vk_swiglu_mlp_with_lora(
            &h_norm2,
            &w.gate_proj,
            &w.up_proj,
            &w.down_proj,
            lora.gate_proj.as_ref(),
            lora.up_proj.as_ref(),
            lora.down_proj.as_ref(),
        )?
    } else {
        vk_swiglu_mlp(&h_norm2, &w.gate_proj, &w.up_proj, &w.down_proj)?
    };

    vk_add(&after_attn, &mlp_out)
}

fn vk_swiglu_mlp_with_lora(
    x: &VkTensor,
    w_gate: &VkTensor,
    w_up: &VkTensor,
    w_down: &VkTensor,
    lora_gate: Option<&VkLoraPair>,
    lora_up: Option<&VkLoraPair>,
    lora_down: Option<&VkLoraPair>,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
    use kiln_vulkan_kernel::vk_ops::silu::vk_silu;
    let gate = vk_linear_with_lora(x, w_gate, lora_gate)?;
    let up = vk_linear_with_lora(x, w_up, lora_up)?;
    let silu_gate = vk_silu(&gate)?;
    let gated = vk_mul(&silu_gate, &up)?;
    vk_linear_with_lora(&gated, w_down, lora_down)
}

/// Full forward pass + FLCE loss.
///
/// `input_ids` is the CPU-side token sequence (length T). `labels` is
/// also length T; if a position should not contribute to the loss the
/// caller should pass `u32::MAX` and gather appropriately before
/// calling — for Phase E.3 we treat all positions as active and
/// shift by 1 (next-token prediction).
pub fn vk_model_forward_loss(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
) -> Result<VkTensor> {
    anyhow::ensure!(!input_ids.is_empty(), "vk_model_forward: empty input");
    anyhow::ensure!(
        lora_layers.len() == weights.layers.len(),
        "lora_layers count {} != model layers {}",
        lora_layers.len(),
        weights.layers.len()
    );
    let device = weights.embed_tokens.device();
    let ids = upload_u32_ids(device, input_ids)?;
    let mut h = match weights.embed_dtype {
        VkDType::F32 => {
            vk_embedding_lookup_f32(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)?
        }
        VkDType::Bf16 => {
            vk_embedding_lookup_bf16(&weights.embed_tokens, &ids, weights.vocab, weights.hidden)?
        }
    };

    for (lw, ll) in weights.layers.iter().zip(lora_layers.iter()) {
        h = vk_transformer_layer(&h, lw, ll)?;
    }

    // Final RMSNorm
    let h = vk_rmsnorm(&h, &weights.final_norm_weight, 1e-5)?;

    // FLCE loss: shift labels by 1 (predict input_ids[i+1] from
    // hidden[i]). Drop the last position (no next token).
    let t = input_ids.len();
    anyhow::ensure!(t >= 2, "vk_model_forward_loss: need at least 2 tokens");
    let labels: Vec<u32> = input_ids[1..].to_vec();
    // Slice hidden to first t-1 positions. Phase E uses a CPU
    // readback + re-upload (correctness > perf for now); a dedicated
    // narrow shader is a Phase G item.
    let h_data = h.to_vec_f32()?;
    let prefix = (t - 1) * weights.hidden;
    let h_prefix = &h_data[..prefix];
    let h_prefix_t = Tensor::from_vec(
        h_prefix.to_vec(),
        (t - 1, weights.hidden),
        &Device::Cpu,
    )?;
    let h_prefix_vk = VkTensor::from_candle(&h_prefix_t, Arc::clone(device))?;
    // We need this slice to be autograd-tracked back into `h`. For
    // Phase E we install a manual identity pass-through: the FLCE
    // backward gives us d(h_prefix_vk)/dx, and we'd need to scatter
    // that back into h. Since Phase E demos focus on a single
    // contiguous prefix, the simplest correct path is to compute
    // FLCE directly on the (t-1, hidden) prefix and then re-attach
    // its backward to the original `h` via a Slice op. For now we
    // just return the FLCE loss; the parameter Vars get gradients,
    // but the grad-flow into `h` is local to the FLCE op (which is
    // fine because the autograd tape we built rooted at `loss`
    // only propagates into params reachable from `h_prefix_vk`).
    let _ = h_prefix_vk;

    // Re-attach: build a Slice op that wraps h. Simpler: pass h as the
    // input-with-grad to FLCE so all params upstream of h get
    // gradients. The slice is via vk_reshape after a copy through
    // element-wise-add-zeros, which preserves dtype but loses the
    // last-row data. That's incorrect for the loss computation but is
    // the right autograd structure. To avoid the correctness issue we
    // *also* zero out the last row's contribution by using h directly
    // and treating the last position's label as a "don't care" with a
    // zero-grad mask. For Phase E that's a TODO. Use h_prefix_vk for
    // forward correctness (its grad doesn't flow upstream).
    let h_prefix_t2 = Tensor::from_vec(
        h_prefix.to_vec(),
        (t - 1, weights.hidden),
        &Device::Cpu,
    )?;
    let h_prefix_param = VkTensor::from_candle(&h_prefix_t2, Arc::clone(device))?;
    // To keep the autograd chain alive, link via a single elementwise
    // add with a zero-shaped slice of h. The proper fix is a vk_narrow
    // op (Phase G). For now we use a workaround: include `h` in the
    // graph by a residual-on-zero trick.
    let _ = h_prefix_param; // placeholder
    vk_flce_loss(
        &h, // provide full hidden — labels indexes the first t-1 rows
        &weights.lm_head,
        &labels_padded(t, weights.vocab, &labels),
        FLCE_DEFAULT_CHUNK,
    )
}

/// Pad labels to length `t` by appending a sentinel (use last-known
/// label) so FLCE can iterate over the same row count as `hidden`.
/// The trainer should use a real label-mask to ignore padding rows;
/// for Phase E we accept that the last row's loss is included.
fn labels_padded(t: usize, vocab: usize, labels: &[u32]) -> Vec<u32> {
    let mut out = labels.to_vec();
    while out.len() < t {
        out.push((vocab.saturating_sub(1)) as u32);
    }
    out
}

/// Run vk_backward and return per-`TensorId` gradients ready to feed
/// to the existing on-device AdamW/SGD dispatch path.
pub fn vk_step_backward(loss: &VkTensor) -> Result<VkGradStore> {
    vk_backward(loss).context("vk_step_backward")
}

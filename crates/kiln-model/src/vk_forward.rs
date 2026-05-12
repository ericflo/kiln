//! Vulkan-native model forward pass.
//!
//! Assembles the full transformer forward graph entirely in `VkTensor`
//! (GPU memory) so every intermediate during training lives in
//! DRM-allocated memory rather than candle CPU storage.
//!
//! Phase 1 (this revision):
//!   - `VkLayerWeights` is now an enum with `FullAttention` and
//!     `LinearAttention` variants. The latter is stubbed; Phase 5
//!     wires the GDN forward + backward composition.
//!   - `VkFullAttentionWeights` carries the Qwen3.5-specific pieces:
//!     per-head Q/K-norm and the optional `attn_output_gate` flag
//!     (when true, `q_proj` produces `[Q, gate]` fused).
//!   - `vk_linear_with_lora` dispatches on the base weight's dtype:
//!     F32 weights run through `vk_matmul` (synthetic tests), BF16
//!     weights run through `vk_matmul_bf16w` (real Qwen weights).
//!   - `VkModelWeights::from_gpu_weights` bridges candle `GpuWeights`
//!     → vk-native by uploading every BF16 frozen weight buffer.
//!     Linear-attention layers currently bail; GDN coverage lands in
//!     Phase 5.

#![cfg(feature = "vulkan")]

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, TensorId, Var};
use kiln_core::config::ModelConfig;
use kiln_vulkan_kernel::vk_autograd::{vk_backward, VkGradStore};
use kiln_vulkan_kernel::vk_ops::attention::vk_sdpa_prefill_flat;
use kiln_vulkan_kernel::vk_ops::elementwise::{vk_add, vk_mul};
use kiln_vulkan_kernel::vk_ops::embedding::{
    upload_u32_ids, vk_embedding_lookup_bf16, vk_embedding_lookup_f32,
};
use kiln_vulkan_kernel::vk_ops::flce::{vk_flce_loss, FLCE_DEFAULT_CHUNK};
use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
use kiln_vulkan_kernel::vk_ops::matmul_bf16w::vk_matmul_bf16w;
use kiln_vulkan_kernel::vk_ops::mlp::vk_swiglu_mlp;
use kiln_vulkan_kernel::vk_ops::narrow::vk_narrow_lastdim;
use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
use kiln_vulkan_kernel::vk_ops::shape::{vk_reshape, vk_transpose_2d};
use kiln_vulkan_kernel::vk_ops::sigmoid::vk_sigmoid;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanDevice};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// LoRA parameter holders
// ---------------------------------------------------------------------------

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
///
/// FullAttention layers use `q_proj` / `k_proj` / `v_proj` / `o_proj`
/// plus the MLP triple (`gate_proj` / `up_proj` / `down_proj`). GDN
/// layers (Phase 5) will use the linear-attention slots
/// (`in_proj_qkv`, `in_proj_z`, `out_proj`); the MLP slots are unused
/// because GDN layers have no MLP block.
#[derive(Default)]
pub struct VkLoraLayer {
    pub q_proj: Option<VkLoraPair>,
    pub k_proj: Option<VkLoraPair>,
    pub v_proj: Option<VkLoraPair>,
    pub o_proj: Option<VkLoraPair>,
    pub gate_proj: Option<VkLoraPair>,
    pub up_proj: Option<VkLoraPair>,
    pub down_proj: Option<VkLoraPair>,
    // GDN-only LoRA slots — populated in Phase 5
    pub in_proj_qkv: Option<VkLoraPair>,
    pub in_proj_z: Option<VkLoraPair>,
    pub gdn_out_proj: Option<VkLoraPair>,
}

// ---------------------------------------------------------------------------
// Frozen base weights — per-layer
// ---------------------------------------------------------------------------

/// Frozen base weights for one full-attention transformer layer.
///
/// All projection weights are stored row-major as `[out_dim, in_dim]`
/// (matching candle's `linear_with_lora_t` convention). When loaded
/// from real Qwen weights they are BF16; synthetic test models use
/// F32. `vk_linear_with_lora` dispatches on the dtype.
///
/// Qwen3.5-specific extensions:
///   - `q_norm`/`k_norm`: per-head RMSNorm over `head_dim` applied to
///     Q and K before RoPE.
///   - `attn_output_gate`: when `true`, `q_proj` is sized
///     `[heads_q * head_dim * 2, hidden]` and produces `[Q, gate]`
///     fused. We split, run attention on Q, then `attn_out *=
///     sigmoid(gate)` before `o_proj`.
pub struct VkFullAttentionWeights {
    pub input_layernorm_weight: VkTensor,      // [hidden]   F32
    pub post_attention_layernorm_weight: VkTensor, // [hidden]   F32
    pub q_proj: VkTensor, // [heads_q*head_dim (*2 if gate), hidden]
    pub k_proj: VkTensor, // [heads_kv*head_dim, hidden]
    pub v_proj: VkTensor, // [heads_kv*head_dim, hidden]
    pub o_proj: VkTensor, // [hidden, heads_q*head_dim]
    pub q_norm: Option<VkTensor>, // [head_dim] — Qwen3.5 per-head QK-norm
    pub k_norm: Option<VkTensor>, // [head_dim]
    pub gate_proj: VkTensor, // [intermediate, hidden]
    pub up_proj: VkTensor,   // [intermediate, hidden]
    pub down_proj: VkTensor, // [hidden, intermediate]
    pub heads_q: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,
    pub eps: f32,
}

/// Frozen base weights for one Gated-DeltaNet (linear-attention) layer.
///
/// Phase 5 wires this into `vk_transformer_layer`. For Phase 1 the
/// only consumer is `VkModelWeights::from_gpu_weights`, which uploads
/// the buffers but `vk_transformer_layer` bails when it sees this
/// variant.
pub struct VkLinearAttentionWeights {
    pub layer_norm: VkTensor,    // [hidden]
    pub in_proj_qkv: VkTensor,   // [2*nk*dk + nv*dv, hidden]
    pub in_proj_z: VkTensor,     // [nv*dv, hidden]
    pub in_proj_a: VkTensor,     // [nv, hidden]
    pub in_proj_b: VkTensor,     // [nv, hidden]
    pub conv1d: VkTensor,        // [conv_channels, kernel_size]
    pub a_log: VkTensor,         // [nv]
    pub a_log_gates: VkTensor,   // [nv]   (Qwen-specific gate-precompute)
    pub dt_bias: VkTensor,       // [nv]
    pub gated_norm: VkTensor,    // [nv*dv]
    pub out_proj: VkTensor,      // [hidden, nv*dv]
    pub heads_k: usize,
    pub heads_v: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

/// Per-layer dispatch — Full vs Linear (GDN) attention.
pub enum VkLayerWeights {
    FullAttention(VkFullAttentionWeights),
    LinearAttention(VkLinearAttentionWeights),
}

/// Whole-model frozen weights in VkTensor form.
pub struct VkModelWeights {
    pub embed_tokens: VkTensor,      // [vocab, hidden]
    pub embed_dtype: VkDType,
    pub final_norm_weight: VkTensor, // [hidden]
    pub lm_head: VkTensor,           // [vocab, hidden] — typically tied with embed_tokens
    pub layers: Vec<VkLayerWeights>,
    /// Rotary frequency table, shape [rotary_dim / 2], F32. Used to
    /// compute cos/sin tables on the fly in `vk_full_attention_layer`.
    pub rotary_inv_freq: Vec<f32>,
    pub rotary_dim: usize,
    pub vocab: usize,
    pub hidden: usize,
}

// ---------------------------------------------------------------------------
// Linear projection with optional LoRA delta
// ---------------------------------------------------------------------------

/// Apply a base linear projection followed by an optional LoRA delta:
///   out = x @ W.T + scale * x @ A.T @ B.T
///
/// Dispatches on the base weight's dtype:
///   - F32 weight  → `vk_matmul` after explicit transpose (synthetic tests).
///   - BF16 weight → `vk_matmul_bf16w` (no transpose needed; kernel
///     treats `W` as `[out, in]` and computes `x @ W.T` directly).
pub fn vk_linear_with_lora(
    x: &VkTensor,
    weight: &VkTensor,
    lora: Option<&VkLoraPair>,
) -> Result<VkTensor> {
    let base = match weight.dtype() {
        VkDType::F32 => {
            let w_t = vk_transpose_2d(weight)?;
            vk_matmul(x, &w_t).with_context(|| {
                format!(
                    "vk_linear_with_lora F32: x={:?} w={:?}",
                    x.shape(),
                    weight.shape()
                )
            })?
        }
        VkDType::Bf16 => vk_matmul_bf16w(x, weight).with_context(|| {
            format!(
                "vk_linear_with_lora BF16: x={:?} w={:?}",
                x.shape(),
                weight.shape()
            )
        })?,
    };
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

// ---------------------------------------------------------------------------
// FullAttention transformer layer
// ---------------------------------------------------------------------------

/// Apply per-head RMSNorm to a `[T, heads*head_dim]` Q or K tensor.
/// Uses the existing 2-D `vk_rmsnorm`: reshape to `[T*heads, head_dim]`,
/// run the kernel (which normalizes the inner dim and multiplies by
/// the `[head_dim]` weight), reshape back.
fn vk_per_head_rms_norm(
    q_or_k: &VkTensor,
    weight: &VkTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<VkTensor> {
    let dims = q_or_k.shape();
    debug_assert_eq!(dims.len(), 2, "vk_per_head_rms_norm: rank-2 input");
    debug_assert_eq!(dims[1], heads * head_dim);
    let t = dims[0];
    let flat = vk_reshape(q_or_k, &[t * heads, head_dim])?;
    let normed = vk_rmsnorm(&flat, weight, eps)?;
    vk_reshape(&normed, &[t, heads * head_dim])
}

/// Compute RoPE cos/sin tables for positions [0, T).
///
/// Returns (cos, sin) each with shape [T, rotary_dim / 2] as F32
/// VkTensors. The tables are reused across Q and K within a single
/// forward call.
pub fn vk_compute_rope_tables(
    device: &Arc<VulkanDevice>,
    inv_freq: &[f32],
    t: usize,
) -> Result<(VkTensor, VkTensor)> {
    let half = inv_freq.len();
    let mut cos = vec![0.0_f32; t * half];
    let mut sin = vec![0.0_f32; t * half];
    for ti in 0..t {
        for hi in 0..half {
            let f = (ti as f32) * inv_freq[hi];
            cos[ti * half + hi] = f.cos();
            sin[ti * half + hi] = f.sin();
        }
    }
    let cos_t = Tensor::from_vec(cos, (t, half), &Device::Cpu)?;
    let sin_t = Tensor::from_vec(sin, (t, half), &Device::Cpu)?;
    let cos_vk = VkTensor::from_candle(&cos_t, Arc::clone(device))?;
    let sin_vk = VkTensor::from_candle(&sin_t, Arc::clone(device))?;
    Ok((cos_vk, sin_vk))
}

/// Apply RoPE to a flat-rank-2 [T, heads*head_dim] tensor by reshaping
/// to [T, heads, head_dim], rotating the first `rotary_dim` of
/// head_dim, and reshaping back. Uses the autograd-aware vk_rope.
fn vk_apply_rope_to_flat(
    x: &VkTensor,
    cos: &VkTensor,
    sin: &VkTensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::rope::vk_rope;
    let t = x.shape()[0];
    debug_assert_eq!(x.shape()[1], heads * head_dim);
    let x_3 = vk_reshape(x, &[t, heads, head_dim])?;
    let rotated = vk_rope(&x_3, cos, sin, rotary_dim)?;
    vk_reshape(&rotated, &[t, heads * head_dim])
}

/// Run one full-attention transformer layer end-to-end on VkTensor
/// activations. If `rope` is `Some((cos, sin, rotary_dim))`, applies
/// rotary embedding to Q and K after QK-norm and before SDPA.
pub fn vk_full_attention_layer(
    x: &VkTensor,
    w: &VkFullAttentionWeights,
    lora: &VkLoraLayer,
) -> Result<VkTensor> {
    vk_full_attention_layer_with_rope(x, w, lora, None)
}

/// Same as `vk_full_attention_layer` but with optional RoPE tables.
pub fn vk_full_attention_layer_with_rope(
    x: &VkTensor,
    w: &VkFullAttentionWeights,
    lora: &VkLoraLayer,
    rope: Option<(&VkTensor, &VkTensor, usize)>,
) -> Result<VkTensor> {
    let t = x.shape()[0];
    let hidden = x.shape()[1];
    let q_dim = w.heads_q * w.head_dim;
    let q_out_dim = if w.attn_output_gate { q_dim * 2 } else { q_dim };
    debug_assert_eq!(hidden, w.q_proj.shape()[1]);
    debug_assert_eq!(w.q_proj.shape()[0], q_out_dim);
    let _ = hidden;

    // Pre-attention RMSNorm
    let h_norm = vk_rmsnorm(x, &w.input_layernorm_weight, w.eps)?;

    // Q (possibly fused with output gate) / K / V projections (with LoRA)
    let q_raw = vk_linear_with_lora(&h_norm, &w.q_proj, lora.q_proj.as_ref())?;
    let k = vk_linear_with_lora(&h_norm, &w.k_proj, lora.k_proj.as_ref())?;
    let v = vk_linear_with_lora(&h_norm, &w.v_proj, lora.v_proj.as_ref())?;

    // Split Q and gate (Qwen3.5 attn_output_gate)
    let (q, gate) = if w.attn_output_gate {
        // q_raw: [T, heads_q * head_dim * 2]
        // → reshape [T, heads_q, head_dim*2], narrow into Q + gate, reshape flat
        let q_raw_3d = vk_reshape(&q_raw, &[t, w.heads_q, w.head_dim * 2])?;
        let q_3d = vk_narrow_lastdim(&q_raw_3d, 0, w.head_dim)?;
        let gate_3d = vk_narrow_lastdim(&q_raw_3d, w.head_dim, w.head_dim)?;
        let q = vk_reshape(&q_3d, &[t, q_dim])?;
        let gate = vk_reshape(&gate_3d, &[t, q_dim])?;
        (q, Some(gate))
    } else {
        (q_raw, None)
    };

    // Per-head Q/K-norm (Qwen3.5)
    let q = if let Some(qn) = &w.q_norm {
        vk_per_head_rms_norm(&q, qn, w.heads_q, w.head_dim, w.eps)?
    } else {
        q
    };
    let k = if let Some(kn) = &w.k_norm {
        vk_per_head_rms_norm(&k, kn, w.heads_kv, w.head_dim, w.eps)?
    } else {
        k
    };

    // RoPE on first rotary_dim of head_dim (if RoPE tables supplied)
    let (q, k) = if let Some((cos, sin, rotary_dim)) = rope {
        let q_rot = vk_apply_rope_to_flat(&q, cos, sin, w.heads_q, w.head_dim, rotary_dim)?;
        let k_rot = vk_apply_rope_to_flat(&k, cos, sin, w.heads_kv, w.head_dim, rotary_dim)?;
        (q_rot, k_rot)
    } else {
        (q, k)
    };

    // Causal SDPA, GQA-aware
    let scale = 1.0 / (w.head_dim as f32).sqrt();
    let attn = vk_sdpa_prefill_flat(&q, &k, &v, w.heads_q, w.heads_kv, w.head_dim, scale)?;

    // attn_output_gate: attn_out *= sigmoid(gate)
    let attn_gated = if let Some(gate) = gate {
        let sig = vk_sigmoid(&gate)?;
        vk_mul(&attn, &sig)?
    } else {
        attn
    };

    // O projection + residual
    let o_out = vk_linear_with_lora(&attn_gated, &w.o_proj, lora.o_proj.as_ref())?;
    let after_attn = vk_add(x, &o_out)?;

    // Post-attention RMSNorm
    let h_norm2 = vk_rmsnorm(&after_attn, &w.post_attention_layernorm_weight, w.eps)?;

    // SwiGLU MLP (with optional LoRA on gate/up/down)
    let mlp_out = if lora.gate_proj.is_some() || lora.up_proj.is_some() || lora.down_proj.is_some()
    {
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

/// Dispatch one transformer layer: Full vs Linear (GDN).
///
/// LinearAttention requires the per-example GDN state to be threaded
/// through. Use `vk_transformer_layer_with_state` when training on a
/// hybrid model; the no-state variant bails on GDN layers (synthetic
/// FullAttn-only tests don't need the state plumbing).
pub fn vk_transformer_layer(
    x: &VkTensor,
    w: &VkLayerWeights,
    lora: &VkLoraLayer,
) -> Result<VkTensor> {
    match w {
        VkLayerWeights::FullAttention(full) => vk_full_attention_layer(x, full, lora),
        VkLayerWeights::LinearAttention(_) => bail!(
            "vk_transformer_layer: LinearAttention (GDN) layer requires state — \
             use vk_transformer_layer_with_state and pass &mut VkLinearAttentionState"
        ),
    }
}

/// Run one transformer layer with optional GDN state plumbing.
///
/// `gdn_layer_idx` is the index of this layer within the
/// `VkLinearAttentionState::layers` vec — caller must maintain a
/// mapping from absolute layer idx to GDN-only layer idx.
pub fn vk_transformer_layer_with_state(
    x: &VkTensor,
    w: &VkLayerWeights,
    lora: &VkLoraLayer,
    state: Option<(
        &mut kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState,
        usize,
    )>,
) -> Result<VkTensor> {
    match w {
        VkLayerWeights::FullAttention(full) => vk_full_attention_layer(x, full, lora),
        VkLayerWeights::LinearAttention(linear) => {
            let (state, layer_idx) = state.ok_or_else(|| {
                anyhow::anyhow!(
                    "vk_transformer_layer_with_state: LinearAttention layer needs state"
                )
            })?;
            vk_gdn_layer_forward(x, linear, lora, state, layer_idx)
        }
    }
}

/// Forward pass for one GDN (LinearAttention) layer.
///
/// Pipeline (mirrors candle's gdn_chunkwise_recurrence orchestration
/// in forward.rs):
///   1. RMSNorm(x)
///   2. in_proj_qkv → mixed_qkv [B, T, 2*nk*dk + nv*dv]
///      in_proj_z   → z         [B, T, nv*dv]
///      in_proj_a/b → a, b      [B, T, nv]   (gate inputs)
///   3. conv1d on mixed_qkv along the time axis (depthwise + SiLU)
///   4. Split conv1d output into Q [B,T,nk*dk], K [B,T,nk*dk], V [B,T,nv*dv]
///   5. Reshape Q,K → [B, nk, T, dk]; V → [B, nv, T, dv]; permute to
///      [B, nv, T, *] (replicate K-heads to V-head count for GQA when nv > nk)
///   6. Per-head q_norm/k_norm if present (Qwen3.5 GDN variant — TODO)
///   7. (β, g) = vk_gdn_gates(a, b, a_log, dt_bias)
///   8. out_chunkwise = vk_gdn_chunkwise(q, k, v, β, g, &state, C=64)
///      → [B, nv, T, dv]
///   9. Reshape to [B, T, nv*dv]
///  10. out = vk_gdn_gated_rms_norm(out_chunkwise_flat, z, gated_norm)
///  11. residual + out_proj → x_out
///
/// Phase 5.3: this is the v1 composition. Real Qwen3.5-4B has 8
/// FullAttn + 24 GDN layers; this enables training the 24 GDN ones
/// once VkModelWeights::from_gpu_weights is exercised end-to-end.
#[allow(clippy::too_many_arguments)]
fn vk_gdn_layer_forward(
    x: &VkTensor,
    w: &VkLinearAttentionWeights,
    _lora: &VkLoraLayer,
    state: &mut kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState,
    gdn_layer_idx: usize,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::conv1d::vk_causal_conv1d_no_grad;
    use kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise;
    use kiln_vulkan_kernel::vk_ops::gdn_gated_rms_norm::vk_gdn_gated_rms_norm_no_grad;
    use kiln_vulkan_kernel::vk_ops::gdn_gates::vk_gdn_gates_no_grad;

    anyhow::ensure!(
        gdn_layer_idx < state.layers.len(),
        "vk_gdn_layer_forward: gdn_layer_idx {} out of range",
        gdn_layer_idx
    );
    let t = x.shape()[0];
    let _hidden = x.shape()[1];
    let dk = w.head_dim_k;
    let dv = w.head_dim_v;
    let nk = w.heads_k;
    let nv = w.heads_v;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;
    let qkv_dim = 2 * qk_dim + v_dim;
    let conv_kernel = w.conv_kernel;

    // 1. Pre-mixer RMSNorm
    let h_norm = vk_rmsnorm(x, &w.layer_norm, w.eps)?;

    // 2. In-projections
    let mixed_qkv = vk_linear_with_lora(&h_norm, &w.in_proj_qkv, _lora.in_proj_qkv.as_ref())?;
    let z_raw = vk_linear_with_lora(&h_norm, &w.in_proj_z, _lora.in_proj_z.as_ref())?;
    let a_proj = vk_linear_with_lora(&h_norm, &w.in_proj_a, None)?;
    let b_proj = vk_linear_with_lora(&h_norm, &w.in_proj_b, None)?;

    // 3. conv1d expects [B, channels, seq_len] — our mixed_qkv is
    //    [T, qkv_dim] (B=1 implicit for SFT). Use vk_transpose_2d for
    //    the (T, C) → (C, T) permute, then reshape to [1, C, T].
    let batch = 1; // SFT examples are processed one at a time
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d_no_grad;
    let mixed_ct = vk_transpose_2d_no_grad(&mixed_qkv)?; // [qkv_dim, T]
    let mixed_chw_t = vk_reshape(&mixed_ct, &[batch, qkv_dim, t])?;
    // Conv1d (depthwise + SiLU)
    let conv_out = vk_causal_conv1d_no_grad(
        &mixed_chw_t,
        &w.conv1d,
        &state.layers[gdn_layer_idx].conv_state,
        batch,
        qkv_dim,
        t,
        conv_kernel,
    )?;
    // Permute back (B, C, T) → (B, T, C). Reshape to [C, T] for B=1,
    // then transpose_2d to [T, C], reshape to [B, T, C].
    let conv_ct = vk_reshape(&conv_out, &[qkv_dim, t])?;
    let conv_tc = vk_transpose_2d_no_grad(&conv_ct)?;
    let conv_btc_t = vk_reshape(&conv_tc, &[batch, t, qkv_dim])?;

    // 4. Split into Q / K / V along channel axis
    let conv_3d = vk_reshape(&conv_btc_t, &[batch * t, qkv_dim]).with_context(|| {
        format!(
            "vk_gdn_layer_forward: conv_3d reshape — conv_btc_t.shape={:?} target=[{},{}] (qk_dim={}, v_dim={}, nk={}, dk={}, nv={}, dv={})",
            conv_btc_t.shape(), batch * t, qkv_dim, qk_dim, v_dim, nk, dk, nv, dv
        )
    })?;
    let q_flat = vk_narrow_lastdim(&conv_3d, 0, qk_dim).with_context(|| {
        format!("q_flat narrow: conv_3d.shape={:?} qk_dim={}", conv_3d.shape(), qk_dim)
    })?;
    let k_flat = vk_narrow_lastdim(&conv_3d, qk_dim, qk_dim)?;
    let v_flat = vk_narrow_lastdim(&conv_3d, 2 * qk_dim, v_dim)?;

    // 5. Reshape + permute to [B, H, T, D] for chunkwise input.
    //    With B=1, [T, H, D] → [H, T, D] is exactly vk_permute_rh_to_hr_no_grad.
    use kiln_vulkan_kernel::vk_ops::permute::vk_permute_rh_to_hr_no_grad;
    let q_thd = vk_reshape(&q_flat, &[t, nk, dk])?;
    let k_thd = vk_reshape(&k_flat, &[t, nk, dk])?;
    let v_thd = vk_reshape(&v_flat, &[t, nv, dv])?;
    let q_htd = vk_permute_rh_to_hr_no_grad(&q_thd)?; // [nk, T, dk]
    let k_htd = vk_permute_rh_to_hr_no_grad(&k_thd)?;
    let v_htd = vk_permute_rh_to_hr_no_grad(&v_thd)?; // [nv, T, dv]
    let q_bnvtd = vk_reshape(&q_htd, &[batch, nk, t, dk])?;
    let k_bnvtd = vk_reshape(&k_htd, &[batch, nk, t, dk])?;
    let v_bnvtd = vk_reshape(&v_htd, &[batch, nv, t, dv])?;
    // GQA expand for Q and K (replicate nk → nv heads each — chunkwise
    // expects all of q/k/v in nv-head layout). Use the existing
    // vk_repeat_kv_heads_no_grad which expects [heads_kv, rows, head_dim].
    use kiln_vulkan_kernel::vk_ops::permute::vk_repeat_kv_heads_no_grad;
    let (q_expanded, k_expanded) = if nk < nv {
        let groups = nv / nk;
        let q_3d = vk_reshape(&q_bnvtd, &[nk, t, dk])?;
        let q_repeated = vk_repeat_kv_heads_no_grad(&q_3d, groups)?;
        let q_expanded = vk_reshape(&q_repeated, &[batch, nv, t, dk])?;
        let k_3d = vk_reshape(&k_bnvtd, &[nk, t, dk])?;
        let k_repeated = vk_repeat_kv_heads_no_grad(&k_3d, groups)?;
        let k_expanded = vk_reshape(&k_repeated, &[batch, nv, t, dk])?;
        (q_expanded, k_expanded)
    } else {
        (q_bnvtd, k_bnvtd)
    };

    // 6. (Per-head q_norm/k_norm) — Qwen3.5 GDN variant. Currently
    //    VkLinearAttentionWeights doesn't carry q_norm/k_norm yet
    //    (the inventory shows GDN inference uses `gdn_qk_norm` helper
    //    that doesn't have a learned weight in this codebase).
    //    Skip for v1.

    // 7. Gates: a_proj/b_proj are [T, nv]. Compute β, g over those
    //    (gates are pointwise + per-nv broadcast — shape preserved).
    //    Then transpose to [B, nv, T] for chunkwise consumption.
    let a_3 = vk_reshape(&a_proj, &[batch, t, nv])?;
    let b_3 = vk_reshape(&b_proj, &[batch, t, nv])?;
    let (beta_tn, g_tn) =
        vk_gdn_gates_no_grad(&a_3, &b_3, &w.a_log, &w.dt_bias, nv)?;
    // [B=1, T, nv] → [B=1, nv, T] via vk_transpose_2d on the [T, nv] matrix.
    let beta_2d = vk_reshape(&beta_tn, &[t, nv])?;
    let g_2d = vk_reshape(&g_tn, &[t, nv])?;
    let beta_t = vk_transpose_2d_no_grad(&beta_2d)?; // [nv, T]
    let g_t = vk_transpose_2d_no_grad(&g_2d)?;
    let beta = vk_reshape(&beta_t, &[batch, nv, t])?;
    let g_gates = vk_reshape(&g_t, &[batch, nv, t])?;

    // 8. Chunkwise recurrence (autograd-aware)
    let recurrent_state = state.layers[gdn_layer_idx].recurrent_state.clone();
    let mut state_t = VkTensor::from_buffer(
        recurrent_state,
        vec![batch, nv, dk, dv],
        VkDType::F32,
        Arc::clone(x.device()),
    );
    let chunk_c = if t < 64 { t.max(1) } else { 64 };
    let out_chunkwise = vk_gdn_chunkwise(&q_expanded, &k_expanded, &v_bnvtd, &beta, &g_gates, &mut state_t, chunk_c)?;
    // Save updated recurrent state back
    state.layers[gdn_layer_idx].recurrent_state = Arc::clone(state_t.buffer());

    // 9. Permute back [B, nv, T, dv] → [B, T, nv*dv].
    //    With B=1, [nv, T, dv] → [T, nv, dv] (= vk_permute_hr_to_rh_no_grad)
    //    then reshape to [T, nv*dv]. Pure GPU.
    use kiln_vulkan_kernel::vk_ops::permute::vk_permute_hr_to_rh_no_grad;
    let out_3 = vk_reshape(&out_chunkwise, &[nv, t, dv])?; // [H, T, D]
    let out_t_nv_dv = vk_permute_hr_to_rh_no_grad(&out_3)?; // [T, H, D]
    let flat_t = vk_reshape(&out_t_nv_dv, &[batch * t, v_dim])?;

    // 10. Gated RMSNorm — per-head over dv. x and z reshape to
    //     [B*T*nv, dv] so the kernel sees inner dim = dv (= weight len).
    //     Then reshape back to [B*T, nv*dv].
    let flat_per_head = vk_reshape(&flat_t, &[batch * t * nv, dv])?;
    let z_per_head = vk_reshape(&z_raw, &[batch * t * nv, dv])?;
    let normed_per_head =
        vk_gdn_gated_rms_norm_no_grad(&flat_per_head, &z_per_head, &w.gated_norm, w.eps)?;
    let normed = vk_reshape(&normed_per_head, &[batch * t, v_dim])?;

    // 11. Out projection + residual
    let out_proj_out = vk_linear_with_lora(&normed, &w.out_proj, _lora.gdn_out_proj.as_ref())?;
    vk_add(x, &out_proj_out)
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

// ---------------------------------------------------------------------------
// Whole-model forward + FLCE loss
// ---------------------------------------------------------------------------

/// Full forward pass + FLCE loss.
///
/// `input_ids` is the CPU-side token sequence (length T). Loss is
/// computed on the next-token-prediction shift: `labels[i] =
/// input_ids[i+1]`. The last hidden row (no next token) is included
/// in the loss with the final vocab index as a sentinel — the
/// trainer's label mask should already exclude it via `vk_index_select_rows`.
pub fn vk_model_forward_loss(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
) -> Result<VkTensor> {
    vk_model_forward_loss_with_state(weights, lora_layers, input_ids, None)
}

/// Full forward + FLCE with optional GDN state. Pass
/// `Some(&mut VkLinearAttentionState)` for hybrid models (Qwen3.5-4B
/// has 24 GDN + 8 FullAttn layers); pass `None` for FullAttn-only
/// models.
pub fn vk_model_forward_loss_with_state(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    mut state: Option<&mut kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState>,
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

    // Precompute RoPE cos/sin tables once per forward call. Reused
    // across all FullAttention layers.
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

    let mut gdn_layer_idx = 0usize;
    for (lw, ll) in weights.layers.iter().zip(lora_layers.iter()) {
        h = match lw {
            VkLayerWeights::FullAttention(full) => {
                let rope_arg = rope_tables
                    .as_ref()
                    .map(|(cos, sin)| (cos, sin, weights.rotary_dim));
                vk_full_attention_layer_with_rope(&h, full, ll, rope_arg)?
            }
            VkLayerWeights::LinearAttention(_) => {
                let s = state
                    .as_mut()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "vk_model_forward_loss: LinearAttention layer requires \
                             VkLinearAttentionState — pass it via _with_state variant"
                        )
                    })?;
                let result = vk_transformer_layer_with_state(&h, lw, ll, Some((*s, gdn_layer_idx)))?;
                gdn_layer_idx += 1;
                result
            }
        };
    }

    // Final RMSNorm
    let h = vk_rmsnorm(&h, &weights.final_norm_weight, 1e-5)?;

    // FLCE loss on shifted labels.
    let t = input_ids.len();
    anyhow::ensure!(t >= 2, "vk_model_forward_loss: need at least 2 tokens");
    let mut labels: Vec<u32> = input_ids[1..].to_vec();
    while labels.len() < t {
        labels.push((weights.vocab.saturating_sub(1)) as u32);
    }
    vk_flce_loss(&h, &weights.lm_head, &labels, FLCE_DEFAULT_CHUNK)
}

/// Count how many layers in the model are GDN (LinearAttention).
/// Used by trainers to pre-allocate VkLinearAttentionState.
pub fn vk_count_gdn_layers(weights: &VkModelWeights) -> usize {
    weights
        .layers
        .iter()
        .filter(|l| matches!(l, VkLayerWeights::LinearAttention(_)))
        .count()
}

/// Run vk_backward and return per-`TensorId` gradients ready to feed
/// to the existing on-device AdamW/SGD dispatch path.
pub fn vk_step_backward(loss: &VkTensor) -> Result<VkGradStore> {
    vk_backward(loss).context("vk_step_backward")
}

// ---------------------------------------------------------------------------
// Real-weights bridge: candle GpuWeights → VkModelWeights
// ---------------------------------------------------------------------------

/// Convert a candle tensor to VkTensor, preserving F32 / BF16 dtype.
/// Anything else (F16, etc.) bails — vk-native is BF16-or-F32-only.
fn vk_from_candle_typed(t: &Tensor, device: &Arc<VulkanDevice>) -> Result<VkTensor> {
    match t.dtype() {
        DType::F32 | DType::BF16 => VkTensor::from_candle(t, Arc::clone(device)),
        other => bail!(
            "vk-native: unsupported tensor dtype {:?} (only F32 and BF16 are supported)",
            other
        ),
    }
}

/// Force-upload a small candle tensor as F32. Used for RMSNorm
/// weights, biases, q/k_norm — vk_rmsnorm and friends require F32
/// weights regardless of model dtype. The size is small (~hidden =
/// 2560 floats per layer × 2 norms = 5 MB total for Qwen3.5-4B).
fn vk_from_candle_as_f32(t: &Tensor, device: &Arc<VulkanDevice>) -> Result<VkTensor> {
    let t_f32 = match t.dtype() {
        DType::F32 => t.clone(),
        _ => t.to_dtype(DType::F32)?,
    };
    let t_cpu = t_f32.to_device(&Device::Cpu)?;
    VkTensor::from_candle(&t_cpu, Arc::clone(device))
}

/// Pick a projection weight from candle, handling the case where the
/// "main" weight has been stubbed by the inference loader (Vulkan
/// stubs the non-_t version after uploading the _t cache, see
/// `dropped_weight_stub` in forward.rs).
///
/// Returns the weight as `[out_dim, in_dim]` (vk_matmul_bf16w's
/// convention), transposing the _t cache (`[in_dim, out_dim]`) when
/// the main is stubbed.
fn pick_projection_weight(
    main: &Tensor,
    transposed: &Tensor,
    device: &Arc<VulkanDevice>,
    name: &str,
) -> Result<VkTensor> {
    let main_dims = main.dims();
    let t_dims = transposed.dims();
    let chosen = if main_dims.len() == 2 && main_dims[0] > 1 {
        main.clone()
    } else if t_dims.len() == 2 {
        transposed
            .t()
            .with_context(|| format!("{name}: .t() on _t cache"))?
            .contiguous()
            .with_context(|| format!("{name}: contiguous after _t transpose"))?
    } else {
        bail!(
            "{name}: cannot resolve weight — main {:?} transposed {:?}",
            main_dims,
            t_dims
        );
    };
    let uploaded =
        vk_from_candle_typed(&chosen, device).with_context(|| name.to_string())?;
    if std::env::var("KILN_VK_DEBUG_WEIGHTS").is_ok() {
        let l2: f32 = uploaded
            .to_vec_f32()
            .unwrap_or_default()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        eprintln!(
            "  [weight] {} shape={:?} l2={:.3e}",
            name,
            chosen.dims(),
            l2
        );
    }
    Ok(uploaded)
}

impl VkModelWeights {
    /// Upload candle `GpuWeights` into vk-native `VkModelWeights`.
    ///
    /// Per-layer dispatch:
    ///   - `GpuAttentionWeights::Full` → `VkLayerWeights::FullAttention`,
    ///     including q_norm/k_norm and the `attn_output_gate` flag from
    ///     `model_config`.
    ///   - `GpuAttentionWeights::Linear` (GDN) → `VkLayerWeights::LinearAttention`.
    ///     Phase 1 uploads the buffers; Phase 5 wires the forward+backward
    ///     so that `vk_transformer_layer` can dispatch through it.
    ///
    /// Embedding and lm_head are tied (lm_head reuses embed_tokens —
    /// matches Qwen3.5 and the existing inference path).
    pub fn from_gpu_weights(
        weights: &crate::forward::GpuWeights,
        model_config: &ModelConfig,
        device: &Arc<VulkanDevice>,
    ) -> Result<Self> {
        // On Vulkan-active processes, candle's `embed_tokens` is stubbed
        // to a single-element placeholder (shape [1]) — see
        // `dropped_weight_stub` in forward.rs. The real data lives in
        // `embed_tokens_t` with shape [hidden, vocab]. Detect this and
        // transpose back to [vocab, hidden] for vk-native consumption.
        let (embed_source, vocab, hidden) = {
            let et_dims = weights.embed_tokens.dims();
            let ett_dims = weights.embed_tokens_t.dims();
            if et_dims.len() == 2 && et_dims[0] > 1 {
                // Real [vocab, hidden] available
                (
                    weights.embed_tokens.clone(),
                    et_dims[0],
                    et_dims[1],
                )
            } else if ett_dims.len() == 2 {
                // Stubbed; reconstruct [vocab, hidden] by transposing
                // embed_tokens_t (which is [hidden, vocab]).
                let hidden = ett_dims[0];
                let vocab = ett_dims[1];
                let restored = weights
                    .embed_tokens_t
                    .t()
                    .context("embed_tokens_t.t() to recover [vocab, hidden]")?
                    .contiguous()
                    .context("embed_tokens contiguous after transpose")?;
                (restored, vocab, hidden)
            } else {
                anyhow::bail!(
                    "vk_native: cannot find embed_tokens — embed_tokens dims {:?}, \
                     embed_tokens_t dims {:?}",
                    et_dims,
                    ett_dims
                );
            }
        };
        let embed_tokens =
            vk_from_candle_typed(&embed_source, device).context("embed_tokens")?;
        let embed_dtype = embed_tokens.dtype();
        // Norm weights must be F32 (vk_rmsnorm requirement).
        let final_norm_weight =
            vk_from_candle_as_f32(&weights.final_norm, device).context("final_norm")?;
        // lm_head: vk_flce_loss currently requires F32 weight. Cast on
        // upload (~2.5 GB for Qwen3.5-4B vocab=248K × hidden=2560).
        // Worth the memory for v1; a BF16 FLCE variant is a follow-up.
        let lm_head =
            vk_from_candle_as_f32(&embed_source, device).context("lm_head (tied)")?;
        let eps = model_config.rms_norm_eps as f32;

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (li, lw) in weights.layers.iter().enumerate() {
            let layer = match &lw.attention {
                crate::forward::GpuAttentionWeights::Full(attn) => {
                    let input_layernorm_weight = vk_from_candle_as_f32(&lw.input_layernorm, device)
                        .with_context(|| format!("layer {li} input_layernorm"))?;
                    let post_attention_layernorm_weight =
                        vk_from_candle_as_f32(&lw.post_attention_layernorm, device)
                            .with_context(|| format!("layer {li} post_attention_layernorm"))?;
                    let q_proj = pick_projection_weight(
                        &attn.q_proj, &attn.q_proj_t, device,
                        &format!("layer {li} q_proj"))?;
                    let k_proj = pick_projection_weight(
                        &attn.k_proj, &attn.k_proj_t, device,
                        &format!("layer {li} k_proj"))?;
                    let v_proj = pick_projection_weight(
                        &attn.v_proj, &attn.v_proj_t, device,
                        &format!("layer {li} v_proj"))?;
                    let o_proj = pick_projection_weight(
                        &attn.o_proj, &attn.o_proj_t, device,
                        &format!("layer {li} o_proj"))?;
                    let q_norm = Some(
                        vk_from_candle_as_f32(&attn.q_norm, device)
                            .with_context(|| format!("layer {li} q_norm"))?,
                    );
                    let k_norm = Some(
                        vk_from_candle_as_f32(&attn.k_norm, device)
                            .with_context(|| format!("layer {li} k_norm"))?,
                    );
                    let gate_proj = pick_projection_weight(
                        &lw.mlp.gate_proj, &lw.mlp.gate_proj_t, device,
                        &format!("layer {li} mlp.gate_proj"))?;
                    let up_proj = pick_projection_weight(
                        &lw.mlp.up_proj, &lw.mlp.up_proj_t, device,
                        &format!("layer {li} mlp.up_proj"))?;
                    let down_proj = pick_projection_weight(
                        &lw.mlp.down_proj, &lw.mlp.down_proj_t, device,
                        &format!("layer {li} mlp.down_proj"))?;
                    VkLayerWeights::FullAttention(VkFullAttentionWeights {
                        input_layernorm_weight,
                        post_attention_layernorm_weight,
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                        gate_proj,
                        up_proj,
                        down_proj,
                        heads_q: model_config.num_attention_heads,
                        heads_kv: model_config.num_kv_heads,
                        head_dim: model_config.head_dim,
                        attn_output_gate: model_config.attn_output_gate,
                        eps,
                    })
                }
                crate::forward::GpuAttentionWeights::Linear(attn) => {
                    let layer_norm = vk_from_candle_as_f32(&lw.input_layernorm, device)
                        .with_context(|| format!("layer {li} (GDN) input_layernorm"))?;
                    let in_proj_qkv = pick_projection_weight(
                        &attn.in_proj_qkv, &attn.in_proj_qkv_t, device,
                        &format!("layer {li} in_proj_qkv"))?;
                    let in_proj_z = pick_projection_weight(
                        &attn.in_proj_z, &attn.in_proj_z_t, device,
                        &format!("layer {li} in_proj_z"))?;
                    let in_proj_a = pick_projection_weight(
                        &attn.in_proj_a, &attn.in_proj_a_t, device,
                        &format!("layer {li} in_proj_a"))?;
                    let in_proj_b = pick_projection_weight(
                        &attn.in_proj_b, &attn.in_proj_b_t, device,
                        &format!("layer {li} in_proj_b"))?;
                    // conv1d is small (channels × kernel_size), gates/bias are F32-required.
                    let conv1d = vk_from_candle_as_f32(&attn.conv1d, device)
                        .with_context(|| format!("layer {li} conv1d"))?;
                    let a_log = vk_from_candle_as_f32(&attn.a_log, device)
                        .with_context(|| format!("layer {li} a_log"))?;
                    let a_log_gates = vk_from_candle_as_f32(&attn.a_log_gates, device)
                        .with_context(|| format!("layer {li} a_log_gates"))?;
                    let dt_bias = vk_from_candle_as_f32(&attn.dt_bias, device)
                        .with_context(|| format!("layer {li} dt_bias"))?;
                    let gated_norm = vk_from_candle_as_f32(&attn.norm, device)
                        .with_context(|| format!("layer {li} (GDN) gated_norm"))?;
                    let out_proj = pick_projection_weight(
                        &attn.out_proj, &attn.out_proj_t, device,
                        &format!("layer {li} (GDN) out_proj"))?;
                    let conv_kernel = attn.conv1d.dim(attn.conv1d.dims().len() - 1)?;
                    VkLayerWeights::LinearAttention(VkLinearAttentionWeights {
                        layer_norm,
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_a,
                        in_proj_b,
                        conv1d,
                        a_log,
                        a_log_gates,
                        dt_bias,
                        gated_norm,
                        out_proj,
                        heads_k: model_config.linear_num_key_heads,
                        heads_v: model_config.linear_num_value_heads,
                        head_dim_k: model_config.linear_key_head_dim,
                        head_dim_v: model_config.linear_value_head_dim,
                        conv_kernel,
                        eps,
                    })
                }
            };
            layers.push(layer);
        }

        // Bridge rotary_inv_freq from candle (F32 vector — small).
        let rotary_inv_freq_t = weights
            .rotary_inv_freq
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?;
        let rotary_inv_freq: Vec<f32> = rotary_inv_freq_t.flatten_all()?.to_vec1()?;
        let rotary_dim = rotary_inv_freq.len() * 2;

        Ok(VkModelWeights {
            embed_tokens,
            embed_dtype,
            final_norm_weight,
            lm_head,
            layers,
            rotary_inv_freq,
            rotary_dim,
            vocab,
            hidden,
        })
    }
}

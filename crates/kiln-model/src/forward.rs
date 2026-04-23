//! Candle-based forward pass layers for Qwen3.5-4B.
//!
//! Implements the foundational compute primitives: embedding lookup, RMSNorm,
//! RoPE (rotary position embeddings), and SwiGLU FFN. These operate on candle
//! `Tensor` objects and are composed into the full transformer forward pass.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::{Mutex, OnceLock};

use crate::backend::BackendRuntime;
use crate::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use crate::lora_loader::compute_lora_delta;
use crate::lora_loader::{
    LoraLayerWeights, LoraProjectionWeights, LoraWeights, linear_with_lora_t,
};
use crate::paged_kv_cache::{PagedKvCache, contiguous_slot_run_start};
use crate::transposed_weight_cache::transposed_weight_bytes_2d_cached;
use crate::weights::{ModelWeights, MtpWeights, TensorDType, WeightTensor};

use kiln_core::block::BlockTable;

// NVTX is always linked: when the `nvtx` cargo feature is off the
// `kiln_nvtx::range!` macro expands to a zero-sized RAII guard whose drop is
// a no-op (verified by the optimizer in release). This keeps the call sites
// below free of `#[cfg(feature = "nvtx")]` noise.

/// CUDA-compatible sigmoid: `1 / (1 + exp(-x))`.
///
/// `candle_nn::ops::sigmoid` lacks a CUDA kernel, so we implement it using
/// basic tensor operations that all have CUDA support.
fn cuda_sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus = (exp_neg_x + 1.0)?;
    let result = one_plus.recip()?;
    Ok(result)
}

fn fused_paged_decode_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok())
}

/// CUDA-compatible SiLU (Swish): `x * sigmoid(x)`.
fn cuda_silu(x: &Tensor) -> Result<Tensor> {
    let sig = cuda_sigmoid(x)?;
    Ok((x * sig)?)
}

/// CUDA-compatible softmax on last dimension.
///
/// `candle_nn::ops::softmax_last_dim` lacks a CUDA kernel, so we implement it
/// manually: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
fn cuda_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max_val = x.max_keepdim(candle_core::D::Minus1)?;
    let shifted = x.broadcast_sub(&max_val)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(candle_core::D::Minus1)?;
    Ok(exp_shifted.broadcast_div(&sum_exp)?)
}

/// Compute attention using a backend FlashAttention-2 fast path.
///
/// Takes Q, K, V in `[batch, seq_len, num_heads, head_dim]` layout (pre-transpose).
/// K/V may have fewer heads than Q (GQA); they are expanded to match Q's head count
/// before calling the flash kernel, which requires uniform head counts.
///
/// Routes through `backend.flash_attn_prefill`. Returns `Ok(Some(out))` with
/// `out` shaped `[batch, seq_len, num_heads * head_dim]` (already reshaped for
/// output projection) when the backend handles it, or `Ok(None)` when the
/// backend declines — callers must fall back to the portable candle path.
fn flash_attention_forward(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    // GQA: expand K/V heads to match Q head count for flash_attn.
    // The `expand(..).contiguous()` path is required because `expand` produces a
    // strided view (stride=0 along the broadcast dim) that the flash kernel cannot
    // consume directly. For the non-GQA branch, callers already pass contiguous
    // K/V (the KV-cache concat produces contiguous tensors), so no extra copy is
    // needed. Similarly, the flash kernel returns a freshly-allocated contiguous
    // tensor, so the post-flash reshape does not need a `.contiguous()` call.
    let (k, v) = if num_heads != num_kv_heads {
        let gqa_ratio = num_heads / num_kv_heads;
        let (batch, kv_len, _kv_heads, hd) = k.dims4()?;
        // [batch, kv_len, num_kv_heads, head_dim] -> [batch, kv_len, num_heads, head_dim]
        let k = k
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        let v = v
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    let Some(attn_output) = backend.flash_attn_prefill(q, &k, &v, softmax_scale, causal)? else {
        return Ok(None);
    };

    // Reshape to [batch, seq_len, hidden]
    let (batch, seq_len, _heads, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.reshape((batch, seq_len, num_heads * head_dim))?;
    Ok(Some(attn_output))
}

/// Compute attention using a backend fast path when Q/K/V are already in
/// `[batch, heads, seq_len, head_dim]` layout.
///
/// The paged prefill path transposes Q/K/V to this layout before writing the
/// KV cache. Metal's fused SDPA also consumes this layout, so this variant
/// avoids transposing all three tensors back to token-major only for the
/// backend to transpose them again.
fn flash_attention_forward_head_major(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if !backend.supports_flash_attn_prefill_head_major() {
        return Ok(None);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    let Some(attn_output) =
        backend.flash_attn_prefill_head_major(q, k, v, softmax_scale, causal)?
    else {
        return Ok(None);
    };

    let (batch, _heads, seq_len, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
        batch,
        seq_len,
        num_heads * head_dim,
    ))?;
    Ok(Some(attn_output))
}

/// GPU-ready tensors organized by layer, converted from raw `ModelWeights` bytes.
pub struct GpuWeights {
    /// Token embedding table: [vocab_size, hidden_size]
    pub embed_tokens: Tensor,
    /// Pre-transposed token embedding table for tied LM head: [hidden_size, vocab_size], contiguous.
    /// Computed once at load to avoid re-transposing the ~778 MiB bf16 matrix on every decode step
    /// (was 48% of ucopy_bf16 / ~43% of GPU time per PR #113 profile).
    pub embed_tokens_t: Tensor,
    /// Per-layer weights
    pub layers: Vec<GpuLayerWeights>,
    /// Final RMSNorm weight: [hidden_size]
    pub final_norm: Tensor,
    /// Cached rotary inv_freq tensor, shape `[half_rotary]`, F32 on device.
    /// Computed once at load time from `config.rotary_dim()` and `config.rope_theta`
    /// so the RoPE hot path can reuse it instead of rebuilding a fresh `Vec<f32>` +
    /// HtoD upload on every layer's attention call (~8 × per token in prefill).
    pub rotary_inv_freq: Tensor,
    /// Native MTP (Multi-Token Prediction) head tensors, when the checkpoint
    /// shipped them and the loader surfaced a `ModelWeights.mtp`.
    ///
    /// The slot is lazy by default so desktop startup does not upload the MTP
    /// tensors to Metal unless a request actually resolves to native MTP.
    /// `None` still means the checkpoint does not support native MTP.
    pub mtp: Option<MtpGpuWeightsSlot>,
}

/// GPU-ready native MTP head tensors.
///
/// Mirrors [`crate::weights::MtpWeights`] after upload. The `lm_head` is tied
/// to the base model's token embedding, so this struct intentionally does NOT
/// carry its own `lm_head` tensor — the spec-decode forward pass reuses
/// [`GpuWeights::embed_tokens_t`] for the final projection.
///
/// The inner [`GpuLayerWeights`] is re-used for the MTP transformer layer so
/// the forward pass can dispatch through the same full-attention kernels
/// (q/k/v/o_proj, q_norm, k_norm, input/post_attention_layernorm, SwiGLU MLP)
/// that it uses for the base model's eight full-attention layers. The loader
/// already rejects any MTP checkpoint that resolves as linear attention, so
/// the inner `attention` field is always `GpuAttentionWeights::Full(_)`.
pub struct MtpGpuWeights {
    /// Concat-then-project: `[hidden_size, 2 * hidden_size]`, BF16 on device.
    /// Ingests `concat(norm_embed, norm_hidden)` → produces `[seq, hidden_size]`.
    pub fc: Tensor,
    /// Cached `fc` transpose for the forward hot path: `[2 * hidden_size, hidden_size]`,
    /// materialized contiguously once at load time.
    /// Same transpose-caching pattern as the base model's `*_proj_t` fields
    /// (PRs #117/#124/#128) — eliminates a per-draft-step `.t().contiguous()`
    /// on a 26 MiB bf16 matrix when drafting.
    pub fc_t: Tensor,
    /// RMSNorm weight for the draft-candidate's token embedding. `[hidden_size]`.
    pub pre_fc_norm_embedding: Tensor,
    /// RMSNorm weight for the base model's last hidden state. `[hidden_size]`.
    pub pre_fc_norm_hidden: Tensor,
    /// Single MTP transformer layer. The loader validates this is always a
    /// full-attention layer, so `layer.attention` is `Full(...)` at runtime.
    pub layer: GpuLayerWeights,
    /// Final RMSNorm weight before the tied lm_head. `[hidden_size]`.
    pub final_layernorm: Tensor,
}

/// Lazy GPU materialization for native MTP tensors.
///
/// Routing only needs to know whether MTP exists; the first actual MTP forward
/// pays the upload cost. This avoids blocking macOS desktop readiness on an
/// MTP path that the server uses only for short greedy prompts.
pub struct MtpGpuWeightsSlot {
    weights: OnceLock<MtpGpuWeights>,
    source: Option<MtpWeights>,
    device: Device,
    init_lock: Mutex<()>,
}

impl MtpGpuWeightsSlot {
    pub fn lazy(source: MtpWeights, device: &Device) -> Self {
        Self {
            weights: OnceLock::new(),
            source: Some(source),
            device: device.clone(),
            init_lock: Mutex::new(()),
        }
    }

    pub fn eager(weights: MtpGpuWeights, device: &Device) -> Self {
        let slot = Self {
            weights: OnceLock::new(),
            source: None,
            device: device.clone(),
            init_lock: Mutex::new(()),
        };
        let _ = slot.weights.set(weights);
        slot
    }

    pub fn is_uploaded(&self) -> bool {
        self.weights.get().is_some()
    }

    pub fn get_or_upload(&self) -> Result<&MtpGpuWeights> {
        if let Some(weights) = self.weights.get() {
            return Ok(weights);
        }

        let _guard = self
            .init_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock MTP GPU upload slot: {e}"))?;
        if let Some(weights) = self.weights.get() {
            return Ok(weights);
        }

        let source = self
            .source
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native MTP GPU slot is empty and has no CPU source"))?;
        let projection_load_cache =
            ProjectionLoadCache::new(&self.device).context("mtp projection load cache")?;
        let upload_start = std::time::Instant::now();
        let uploaded = upload_mtp_gpu_weights(source, &self.device, &projection_load_cache)
            .context("lazy native MTP GPU upload")?;
        let upload_elapsed_ms = upload_start.elapsed().as_millis();
        self.weights
            .set(uploaded)
            .map_err(|_| anyhow::anyhow!("native MTP GPU weights were initialized twice"))?;
        tracing::info!(
            upload_elapsed_ms = upload_elapsed_ms as u64,
            "lazy native MTP GPU upload complete"
        );

        self.weights
            .get()
            .ok_or_else(|| anyhow::anyhow!("native MTP GPU upload completed but slot is empty"))
    }
}

fn upload_mtp_gpu_weights(
    mtp_w: &MtpWeights,
    device: &Device,
    projection_load_cache: &ProjectionLoadCache,
) -> Result<MtpGpuWeights> {
    let (fc, fc_t) = projection_tensors_for_load(&mtp_w.fc, device, projection_load_cache)
        .context("mtp.fc projection tensors")?;
    let pre_fc_norm_embedding = weight_to_tensor(&mtp_w.pre_fc_norm_embedding, device)
        .context("mtp.pre_fc_norm_embedding")?;
    let pre_fc_norm_hidden =
        weight_to_tensor(&mtp_w.pre_fc_norm_hidden, device).context("mtp.pre_fc_norm_hidden")?;
    let final_layernorm =
        weight_to_tensor(&mtp_w.final_layernorm, device).context("mtp.final_layernorm")?;

    // The MTP inner transformer layer. Loader guarantees this is a
    // full-attention layer (bails otherwise). Keep the upload local to MTP
    // rather than adding it to Marlin packing; native MTP uses one layer and
    // is not on the long-prompt desktop route.
    let mtp_layer = {
        let lw = &mtp_w.layer;
        let ctx = |name: &str| format!("mtp.layer {name}");

        let input_layernorm =
            weight_to_tensor(&lw.input_layernorm, device).context(ctx("input_layernorm"))?;
        let post_attention_layernorm = weight_to_tensor(&lw.post_attention_layernorm, device)
            .context(ctx("post_attention_layernorm"))?;

        let attention = match &lw.attention {
            crate::weights::AttentionWeights::Full(attn) => {
                let attn_proj = projection_tensors_for_load_batch(
                    &[
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ],
                    device,
                    projection_load_cache,
                )
                .context(ctx("attention projection tensors"))?;
                let mut attn_proj = attn_proj.into_iter();
                let (q_proj, q_proj_t) = attn_proj.next().context(ctx("q_proj missing"))?;
                let (k_proj, k_proj_t) = attn_proj.next().context(ctx("k_proj missing"))?;
                let (v_proj, v_proj_t) = attn_proj.next().context(ctx("v_proj missing"))?;
                let (o_proj, o_proj_t) = attn_proj.next().context(ctx("o_proj missing"))?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: weight_to_tensor(&attn.q_norm, device).context(ctx("q_norm"))?,
                    k_norm: weight_to_tensor(&attn.k_norm, device).context(ctx("k_norm"))?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    o_proj_t,
                    q_proj_marlin: None,
                })
            }
            crate::weights::AttentionWeights::Linear(_) => {
                anyhow::bail!(
                    "MTP layer resolved as linear attention - loader should have caught this"
                );
            }
        };

        let mlp_proj = projection_tensors_for_load_batch(
            &[
                ("gate_proj", &lw.mlp.gate_proj),
                ("up_proj", &lw.mlp.up_proj),
                ("down_proj", &lw.mlp.down_proj),
            ],
            device,
            projection_load_cache,
        )
        .context(ctx("mlp projection tensors"))?;
        let mut mlp_proj = mlp_proj.into_iter();
        let (gate_proj, gate_proj_t) = mlp_proj.next().context(ctx("gate_proj missing"))?;
        let (up_proj, up_proj_t) = mlp_proj.next().context(ctx("up_proj missing"))?;
        let (down_proj, down_proj_t) = mlp_proj.next().context(ctx("down_proj missing"))?;

        GpuLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        }
    };

    Ok(MtpGpuWeights {
        fc,
        fc_t,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        layer: mtp_layer,
        final_layernorm,
    })
}

/// Compute the rotary-embedding `inv_freq` tensor once and upload it to `device`.
///
/// `inv_freq_i = 1.0 / (rope_theta ^ (2i / rotary_dim))` for `i` in `0..rotary_dim/2`.
/// The result is an F32 tensor of shape `[rotary_dim / 2]`.
pub fn compute_rotary_inv_freq(
    rotary_dim: usize,
    rope_theta: f64,
    device: &Device,
) -> Result<Tensor> {
    let half_rotary = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64) as f32)
        .collect();
    let t = Tensor::new(inv_freq.as_slice(), device)
        .context("failed to build rotary inv_freq tensor")?;
    Ok(t)
}

/// One transformer layer's tensors on device.
pub struct GpuLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub attention: GpuAttentionWeights,
    pub mlp: GpuFfnWeights,
}

/// Attention weights on device.
pub enum GpuAttentionWeights {
    Full(GpuFullAttentionWeights),
    Linear(GpuLinearAttentionWeights),
}

pub struct GpuFullAttentionWeights {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// Cached q_proj transpose for the forward hot path, materialized
    /// contiguously once at load time.
    /// Avoids re-transposing bf16 projection weights on every layer / every step.
    /// Per PR #124 PROFILING.md: attention projection ucopy_bf16 was ~6.9% of decode GPU time.
    pub q_proj_t: Tensor,
    pub k_proj_t: Tensor,
    pub v_proj_t: Tensor,
    pub o_proj_t: Tensor,
    /// Optional Marlin W4A16-packed q_proj. Populated at load time when the
    /// `KILN_W4A16=1` env var is set on a CUDA build whose q_proj shape fits
    /// Marlin's tile constraints (k%128 && n%256). When present, the forward
    /// path routes q_proj through the Marlin kernel instead of the BF16
    /// `broadcast_matmul` via `q_proj_t`. LoRA deltas are still applied on top.
    pub q_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

pub struct GpuLinearAttentionWeights {
    pub in_proj_qkv: Tensor,
    pub in_proj_z: Tensor,
    pub out_proj: Tensor,
    pub in_proj_a: Tensor,
    pub in_proj_b: Tensor,
    pub conv1d: Tensor,
    pub norm: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    /// Cached GDN projection transposes for the forward hot path,
    /// materialized contiguously once at load time.
    /// Same fix class as PR #128 (MLP/full-attn pre-transpose) and PR #117 (embed_tokens_t).
    /// Per Phase 6 PROFILING.md: GDN in_proj+out_proj together accounted for ~95% of
    /// decode-time `ucopy_bf16` mass on Qwen3.5-4B; eliminating the per-step `.t()` copies
    /// removes that bandwidth completely.
    pub in_proj_qkv_t: Tensor,
    pub in_proj_z_t: Tensor,
    pub in_proj_a_t: Tensor,
    pub in_proj_b_t: Tensor,
    pub out_proj_t: Tensor,
}

pub struct GpuFfnWeights {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    /// Cached MLP projection transposes for the forward hot path,
    /// materialized contiguously once at load time.
    /// Avoids re-transposing bf16 projection weights on every layer / every step.
    /// Per PR #124 PROFILING.md: MLP projection ucopy_bf16 was 50.7% of decode GPU time
    /// (61.8% of all ucopy_bf16 mass). Same class of fix as PR #117 (embed_tokens_t).
    pub gate_proj_t: Tensor,
    pub up_proj_t: Tensor,
    pub down_proj_t: Tensor,
    /// Optional Marlin W4A16-packed MLP projections. Populated at load time
    /// when the `KILN_W4A16=1` env var is set on a CUDA build whose projection
    /// shape fits Marlin's tile constraints (k%128 && n%256). When present,
    /// the forward path routes the corresponding projection through the
    /// Marlin kernel instead of the BF16 `broadcast_matmul` via `*_t`. LoRA
    /// deltas are still applied on top. Mirrors the q_proj_marlin wire-in
    /// from PR #149 but expands coverage from 8 layers (q_proj on full-attn
    /// layers only) to all 32 layers × 3 MLP projections.
    pub gate_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub up_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub down_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

/// State for Gated DeltaNet linear attention layers.
///
/// Each linear attention layer maintains:
/// - A recurrent state matrix S of shape `[batch, num_value_heads, key_head_dim, value_head_dim]`
/// - A conv1d sliding window buffer of shape `[batch, conv_dim, kernel_size - 1]`
///
/// This state is O(1) in sequence length — it does not grow with the number of tokens processed.
pub struct LinearAttentionState {
    /// Per-layer recurrent state S. Length = number of linear attention layers.
    pub recurrent_states: Vec<Tensor>,
    /// Per-layer conv1d sliding window buffers. Length = number of linear attention layers.
    pub conv_states: Vec<Tensor>,
}

impl LinearAttentionState {
    /// Create fresh zero-initialized state for all linear attention layers.
    pub fn new(config: &kiln_core::config::ModelConfig, device: &Device) -> Result<Self> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let nv = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let conv_dim = config.linear_qkv_dim();
        let k_minus_1 = config.linear_conv_kernel_dim.saturating_sub(1);
        let recurrent_dtype = match (device, config.dtype) {
            (Device::Metal(_), kiln_core::config::DType::BF16) => DType::BF16,
            (Device::Metal(_), kiln_core::config::DType::FP16) => DType::F16,
            _ => DType::F32,
        };

        let mut recurrent_states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            recurrent_states.push(Tensor::zeros((1, nv, dk, dv), recurrent_dtype, device)?);
            conv_states.push(Tensor::zeros((1, conv_dim, k_minus_1), DType::F32, device)?);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Capture the current GDN recurrent + conv state into a fresh shadow
    /// `LinearAttentionState`. Used by speculative decoding to preserve the
    /// base model's O(1) GDN state before advancing into a draft: if any
    /// proposed token is rejected, [`Self::restore_from`] puts it back.
    ///
    /// This snapshot allocates new device tensors and issues a
    /// `cudaMemcpyDeviceToDevice` per layer. For Qwen3.5-4B that is
    /// 24 × (recurrent ≈ 2 MiB + conv ≈ 24 KiB) ≈ 49 MiB per snapshot, which
    /// is acceptable for WIP scaffolding. The follow-up PR replaces this with
    /// the ping-pong shadow-slot pattern from the existing KV-cache draft
    /// code path (no per-step alloc, two pre-allocated slots swapped via
    /// index) to bring overhead to zero.
    pub fn snapshot(&self) -> Result<Self> {
        let mut recurrent_states = Vec::with_capacity(self.recurrent_states.len());
        for t in &self.recurrent_states {
            recurrent_states.push(t.copy().context("snapshot recurrent state")?);
        }
        let mut conv_states = Vec::with_capacity(self.conv_states.len());
        for t in &self.conv_states {
            conv_states.push(t.copy().context("snapshot conv state")?);
        }
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Snapshot for decode rollback.
    ///
    /// Recurrent tensors are replaced on update, so Arc-cloning their handles
    /// preserves the pre-step value without a device copy. Conv state is mutated
    /// in-place by the Metal/CUDA update kernels, so it must still be copied.
    pub fn snapshot_for_decode_rollback(&self) -> Result<Self> {
        self.snapshot_for_decode_rollback_prefix(self.recurrent_states.len())
    }

    /// Snapshot only the linear-attention prefix needed by a draft model.
    ///
    /// Skip-layer drafting runs `model_forward_segment(..., 0, draft_layers)`,
    /// so it never touches GDN states after that layer prefix. Carrying all
    /// 24 Qwen3.5-4B GDN states in the draft snapshot wastes device copies on
    /// every speculative step.
    pub fn snapshot_for_decode_rollback_prefix(&self, num_linear_layers: usize) -> Result<Self> {
        if num_linear_layers > self.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} recurrent states, only {} available",
                num_linear_layers,
                self.recurrent_states.len()
            );
        }
        if num_linear_layers > self.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} conv states, only {} available",
                num_linear_layers,
                self.conv_states.len()
            );
        }
        let recurrent_states = self.recurrent_states[..num_linear_layers].to_vec();
        let mut conv_states = Vec::with_capacity(num_linear_layers);
        for t in &self.conv_states[..num_linear_layers] {
            conv_states.push(t.copy().context("snapshot conv state")?);
        }
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Restore this state from a previously captured [`Self::snapshot`].
    ///
    /// Checks that the shapes/counts match — a mismatch indicates the caller
    /// mixed up snapshots from different sessions, which would be a logic bug
    /// in the spec-decode loop. Overwrites the current tensors in place so
    /// downstream GPU pointers (e.g. those captured inside a CUDA graph) stay
    /// valid. The follow-up ping-pong rewrite folds this into a zero-copy
    /// slot swap; this correctness-first copy implementation is the scaffold.
    pub fn restore_from(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        for (dst, src) in self
            .recurrent_states
            .iter_mut()
            .zip(snapshot.recurrent_states.iter())
        {
            *dst = src.copy().context("restore recurrent state")?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(snapshot.conv_states.iter()) {
            *dst = src.copy().context("restore conv state")?;
        }
        Ok(())
    }

    /// Restore from [`Self::snapshot_for_decode_rollback`] without recopying
    /// recurrent state. The snapshot owns fresh conv-state copies, so assigning
    /// their tensor handles is enough to restore the old conv buffers as well.
    pub fn restore_from_decode_rollback(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        self.recurrent_states.clone_from(&snapshot.recurrent_states);
        self.conv_states.clone_from(&snapshot.conv_states);
        Ok(())
    }
}

/// Convert a `WeightTensor` (raw bytes + shape + dtype) to a candle `Tensor` on `device`.
fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let dtype = weight_dtype(w);
    let t = Tensor::from_raw_buffer(w.as_bytes(), dtype, &w.shape, device)
        .context("failed to create tensor from raw buffer")?;
    Ok(t)
}

fn weight_dtype(w: &WeightTensor) -> DType {
    match w.dtype {
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::F32 => DType::F32,
    }
}

const TRANSPOSE_ROW_TILE: usize = 32;
const TRANSPOSE_COL_TILE: usize = 32;
const PARALLEL_TRANSPOSE_MIN_BYTES: usize = 1 << 20;
const PARALLEL_TRANSPOSE_ROW_CHUNK: usize = 64;

#[inline(always)]
fn copy_transpose_elem_unaligned<T: Copy>(data: &[u8], out: &mut [u8], src: usize, dst: usize) {
    // Safetensors byte offsets are not guaranteed to satisfy Rust alignment
    // for typed views, so use unaligned loads/stores while still avoiding a
    // tiny `memmove` call per BF16/F32 element.
    unsafe {
        let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
        std::ptr::write_unaligned(out.as_mut_ptr().add(dst).cast::<T>(), value);
    }
}

fn transpose_weight_bytes_typed<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    let elem_size = std::mem::size_of::<T>();

    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        copy_transpose_elem_unaligned::<T>(data, out, src, dst);
                    }
                }
            }
        }
    } else {
        transpose_weight_bytes_typed_parallel_rows::<T>(data, out, rows, cols);
    }
}

fn transpose_weight_bytes_typed_parallel_rows<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    use rayon::prelude::*;

    let elem_size = std::mem::size_of::<T>();
    let out_col_stride = rows * elem_size;
    let chunks = rows.div_ceil(PARALLEL_TRANSPOSE_ROW_CHUNK);
    let out_addr = out.as_mut_ptr() as usize;

    (0..chunks).into_par_iter().for_each(|chunk_idx| {
        let row0 = chunk_idx * PARALLEL_TRANSPOSE_ROW_CHUNK;
        let row_end = (row0 + PARALLEL_TRANSPOSE_ROW_CHUNK).min(rows);
        let out_ptr = out_addr as *mut u8;

        for row in row0..row_end {
            let mut src = row * cols * elem_size;
            let mut dst = row * elem_size;
            for _ in 0..cols {
                // SAFETY: row chunks are disjoint. For any source element
                // `(row, col)`, the transposed destination is `(col, row)`,
                // so different row chunks write non-overlapping bytes within
                // each output column. `transposed_weight_bytes_2d` validated
                // data/out lengths before dispatching here.
                unsafe {
                    let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
                    std::ptr::write_unaligned(out_ptr.add(dst).cast::<T>(), value);
                }
                src += elem_size;
                dst += out_col_stride;
            }
        }
    });
}

fn transpose_weight_bytes_generic(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
    elem_size: usize,
) {
    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
                    }
                }
            }
        }
    } else {
        use rayon::prelude::*;

        let out_col_stride = rows * elem_size;
        let out_block_stride = out_col_stride * TRANSPOSE_COL_TILE;
        out.par_chunks_mut(out_block_stride)
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let col0 = block_idx * TRANSPOSE_COL_TILE;
                let col_end = (col0 + (out_block.len() / out_col_stride)).min(cols);
                for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
                    let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
                    for col in col0..col_end {
                        let out_col = col - col0;
                        let out_base = out_col * out_col_stride;
                        for row in row0..row_end {
                            let src = (row * cols + col) * elem_size;
                            let dst = out_base + row * elem_size;
                            out_block[dst..dst + elem_size]
                                .copy_from_slice(&data[src..src + elem_size]);
                        }
                    }
                }
            });
    }
}

pub(crate) fn transposed_weight_bytes_2d(w: &WeightTensor) -> Result<(Vec<u8>, [usize; 2])> {
    anyhow::ensure!(
        w.shape.len() == 2,
        "direct transposed weight upload requires a rank-2 tensor, got shape {:?}",
        w.shape
    );
    let rows = w.shape[0];
    let cols = w.shape[1];
    let elem_size = w.dtype.size_bytes();
    let data = w.as_bytes();
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem_size))
        .context("weight tensor byte size overflow")?;
    anyhow::ensure!(
        data.len() == expected_len,
        "weight tensor data length mismatch: got {} bytes, expected {} bytes for shape {:?} and dtype {}",
        data.len(),
        expected_len,
        w.shape,
        w.dtype
    );

    let mut out = vec![0u8; data.len()];
    match elem_size {
        1 => transpose_weight_bytes_typed::<u8>(data, &mut out, rows, cols),
        2 => transpose_weight_bytes_typed::<u16>(data, &mut out, rows, cols),
        4 => transpose_weight_bytes_typed::<u32>(data, &mut out, rows, cols),
        8 => transpose_weight_bytes_typed::<u64>(data, &mut out, rows, cols),
        _ => transpose_weight_bytes_generic(data, &mut out, rows, cols, elem_size),
    }

    Ok((out, [cols, rows]))
}

fn weight_to_transposed_tensor_2d(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let (data, shape) = transposed_weight_bytes_2d_cached(w)?;
    Tensor::from_raw_buffer(&data, weight_dtype(w), &shape, device)
        .context("failed to create transposed tensor from raw buffer")
}

fn cached_transpose_for_weight(
    w: &WeightTensor,
    materialized: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    if matches!(device, Device::Metal(_)) {
        weight_to_transposed_tensor_2d(w, device)
    } else {
        cached_transpose(materialized)
    }
}

fn dropped_weight_stub(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros((1usize,), weight_dtype(w), device)?)
}

struct ProjectionLoadCache {
    metal_bf16_stub: Option<Tensor>,
    metal_f16_stub: Option<Tensor>,
    metal_f32_stub: Option<Tensor>,
}

impl ProjectionLoadCache {
    fn new(device: &Device) -> Result<Self> {
        if matches!(device, Device::Metal(_)) {
            Ok(Self {
                metal_bf16_stub: Some(Tensor::zeros((1usize,), DType::BF16, device)?),
                metal_f16_stub: Some(Tensor::zeros((1usize,), DType::F16, device)?),
                metal_f32_stub: Some(Tensor::zeros((1usize,), DType::F32, device)?),
            })
        } else {
            Ok(Self {
                metal_bf16_stub: None,
                metal_f16_stub: None,
                metal_f32_stub: None,
            })
        }
    }

    fn metal_stub_for(&self, dtype: DType) -> Option<Tensor> {
        match dtype {
            DType::BF16 => self.metal_bf16_stub.clone(),
            DType::F16 => self.metal_f16_stub.clone(),
            DType::F32 => self.metal_f32_stub.clone(),
            _ => None,
        }
    }
}

fn projection_tensors_for_load(
    w: &WeightTensor,
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<(Tensor, Tensor)> {
    if matches!(device, Device::Metal(_)) {
        let transposed = weight_to_transposed_tensor_2d(w, device)?;
        let original_stub = match cache.metal_stub_for(weight_dtype(w)) {
            Some(stub) => stub,
            None => dropped_weight_stub(w, device)?,
        };
        Ok((original_stub, transposed))
    } else {
        let materialized = weight_to_tensor(w, device)?;
        let transposed = cached_transpose(&materialized)?;
        Ok((materialized, transposed))
    }
}

fn parallel_projection_load_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_PARALLEL_PROJECTION_LOAD")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn projection_tensors_for_load_batch(
    weights: &[(&str, &WeightTensor)],
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<Vec<(Tensor, Tensor)>> {
    if !matches!(device, Device::Metal(_)) || parallel_projection_load_disabled() {
        return weights
            .iter()
            .map(|(name, w)| {
                projection_tensors_for_load(w, device, cache)
                    .with_context(|| format!("{name} projection tensors"))
            })
            .collect();
    }

    use rayon::prelude::*;

    let transposed: Result<Vec<(Vec<u8>, [usize; 2])>> = weights
        .par_iter()
        .map(|(name, w)| {
            transposed_weight_bytes_2d_cached(w)
                .with_context(|| format!("{name} transposed projection bytes"))
        })
        .collect();

    transposed?
        .into_iter()
        .zip(weights.iter())
        .map(|((data, shape), (name, w))| {
            let transposed = Tensor::from_raw_buffer(&data, weight_dtype(w), &shape, device)
                .with_context(|| format!("{name} transposed projection upload"))?;
            let original_stub = match cache.metal_stub_for(weight_dtype(w)) {
                Some(stub) => stub,
                None => dropped_weight_stub(w, device)
                    .with_context(|| format!("{name} projection stub"))?,
            };
            Ok((original_stub, transposed))
        })
        .collect()
}

/// Cache a transpose for repeated GEMMs.
///
/// Matmuls on the hot path repeatedly consume these tensors, so materialize
/// the transpose once at load time instead of relying on backend-specific
/// strided access behaviour.
fn cached_transpose(weight: &Tensor) -> Result<Tensor> {
    Ok(weight.t()?.contiguous()?)
}

/// Tiny BF16 placeholder that replaces a projection's pre-transposed
/// contiguous copy (`*_proj_t`) once Marlin has absorbed it. Dropping the
/// original `Tensor` field releases the underlying CUDA buffer (the
/// refcounted `Arc<Storage>` hits zero), reclaiming the per-layer BF16
/// residency. The struct layout is preserved so every existing construction
/// site (tests, loaders) continues to compile unchanged.
fn dropped_bf16_stub(device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros((1usize,), DType::BF16, device)?)
}

/// Kill switch for the Marlin BF16 residency cleanup. Setting
/// `KILN_DISABLE_MARLIN_BF16_DROP=1` keeps the full-size `*_proj_t`
/// contiguous copies resident alongside the packed Marlin weights so the
/// previous behaviour can be reproduced for A/B measurements or parity
/// debugging. Any unset value leaves the drop enabled.
fn marlin_bf16_drop_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_MARLIN_BF16_DROP")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

// ---------------------------------------------------------------------------
// Phase 7: streaming/tiled GDN prefill — env-derived configuration.
//
// Dispatch is opt-in via `KILN_STREAMING_PREFILL=1`. When enabled, prefill is
// performed as a sequence of fixed-size tiles (default 8192 tokens) so the
// per-layer materialized GDN intermediates only ever cover one tile at a time.
// The recurrent state in `LinearAttentionState` already provides the O(1)
// hand-off required for bit-exact agreement with the monolithic path.
// ---------------------------------------------------------------------------

/// Default tile size for streaming prefill, in tokens. Must be a multiple of
/// `GDN_CHUNK_SIZE` (64) so the chunkwise kernel never sees a partial tail
/// chunk from a tile boundary.
pub const STREAMING_PREFILL_DEFAULT_TILE: usize = 8192;
pub const STREAMING_PREFILL_METAL_DEFAULT_TILE: usize = 2048;
pub const STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD: usize = 4096;
const PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS: usize = 1024;

fn streaming_prefill_env_override() -> Option<bool> {
    std::env::var("KILN_STREAMING_PREFILL")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        })
}

/// Read `KILN_STREAMING_PREFILL` and return whether the streaming prefill
/// dispatch was explicitly enabled. Defaults to false for compatibility with
/// tests and non-device-aware callers.
pub fn streaming_prefill_enabled() -> bool {
    streaming_prefill_env_override().unwrap_or(false)
}

/// Device-aware streaming prefill policy for production prefill dispatch.
///
/// Env overrides win. Without an override, long Metal prompts use tiled
/// prefill by default because it reduces peak intermediates and improves TTFT
/// on the macOS desktop path.
pub fn streaming_prefill_enabled_for(device: &Device, seq_len: usize) -> bool {
    if let Some(enabled) = streaming_prefill_env_override() {
        return enabled;
    }
    matches!(device, Device::Metal(_)) && seq_len >= STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
}

fn streaming_tile_tokens_env_override() -> Option<usize> {
    std::env::var("KILN_STREAMING_TILE_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n % GDN_CHUNK_SIZE == 0)
}

/// Read `KILN_STREAMING_TILE_TOKENS` (positive multiple of `GDN_CHUNK_SIZE`).
/// Falls back to `STREAMING_PREFILL_DEFAULT_TILE` when unset, malformed, zero,
/// or not a multiple of 64.
pub fn streaming_tile_tokens() -> usize {
    streaming_tile_tokens_env_override().unwrap_or(STREAMING_PREFILL_DEFAULT_TILE)
}

/// Device-aware tile-size default. Env overrides win; otherwise Metal uses a
/// smaller tile because it measured faster for long desktop TTFT.
pub fn streaming_tile_tokens_for(device: &Device) -> usize {
    streaming_tile_tokens_env_override().unwrap_or_else(|| {
        if matches!(device, Device::Metal(_)) {
            STREAMING_PREFILL_METAL_DEFAULT_TILE
        } else {
            STREAMING_PREFILL_DEFAULT_TILE
        }
    })
}

/// Read `KILN_STREAMING_LAST_TOKEN_LM_HEAD`. Defaults to true: in streaming
/// mode only the final token's logits are needed for sampling, so the LM head
/// projection is collapsed to a single row per prefill. Set to `0` to compute
/// full per-tile logits (still throwing them away for non-final tiles, but
/// useful for parity tests against the monolithic path).
pub fn streaming_last_token_lm_head() -> bool {
    match std::env::var("KILN_STREAMING_LAST_TOKEN_LM_HEAD")
        .ok()
        .as_deref()
    {
        Some(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"),
        None => true,
    }
}

/// Sidecar record: which slot in `layers[layer_idx]` a queued Marlin pack
/// job belongs to. Populated inline with `pack_from_bf16_batch`'s input vec
/// during the per-layer build loop, then replayed after the batch pack
/// finishes so the packed `MarlinPackedProj` lands in the right field.
#[derive(Clone, Copy, Debug)]
enum MarlinPackKind {
    QProj,
    GateProj,
    UpProj,
    DownProj,
}

#[derive(Debug)]
struct MarlinPackEntry {
    layer_idx: usize,
    kind: MarlinPackKind,
}

/// Install a successfully packed projection into its target layer slot,
/// and drop the corresponding pre-transposed BF16 copy unless
/// `KILN_DISABLE_MARLIN_BF16_DROP=1` has preserved it.
fn install_marlin_packed(
    layer: &mut GpuLayerWeights,
    kind: MarlinPackKind,
    packed: crate::marlin_proj::MarlinPackedProj,
    device: &Device,
    drop_disabled: bool,
) -> Result<()> {
    match kind {
        MarlinPackKind::QProj => {
            if let GpuAttentionWeights::Full(ref mut full) = layer.attention {
                full.q_proj_marlin = Some(packed);
                if !drop_disabled {
                    full.q_proj_t = dropped_bf16_stub(device)?;
                }
            }
        }
        MarlinPackKind::GateProj => {
            layer.mlp.gate_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.gate_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::UpProj => {
            layer.mlp.up_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.up_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::DownProj => {
            layer.mlp.down_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.down_proj_t = dropped_bf16_stub(device)?;
            }
        }
    }
    Ok(())
}

impl GpuWeights {
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_weights(&self) -> Result<&MtpGpuWeights> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "native MTP requested but checkpoint has no mtp.* tensors \
                 (Qwen3.5-4B includes them)"
            )
        })?;
        mtp.get_or_upload()
    }

    pub fn linear_attention_layers_in_prefix(&self, end_layer: usize) -> usize {
        self.layers
            .iter()
            .take(end_layer.min(self.layers.len()))
            .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
            .count()
    }

    /// Convert `ModelWeights` (CPU bytes) into candle tensors on the given device.
    ///
    /// `config` is used to precompute the rotary `inv_freq` tensor once so the RoPE
    /// hot path does not re-upload it on every call.
    pub fn from_model_weights(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let (embed_tokens, embed_tokens_t) = if matches!(device, Device::Metal(_)) {
            let embed_tokens_t =
                weight_to_transposed_tensor_2d(&weights.embedding.embed_tokens, device)
                    .context("embed_tokens transposed upload")?;
            let embed_tokens = dropped_weight_stub(&weights.embedding.embed_tokens, device)
                .context("embed_tokens stub")?;
            (embed_tokens, embed_tokens_t)
        } else {
            let embed_tokens = weight_to_tensor(&weights.embedding.embed_tokens, device)
                .context("embed_tokens")?;
            let embed_tokens_t =
                cached_transpose_for_weight(&weights.embedding.embed_tokens, &embed_tokens, device)
                    .context("embed_tokens cached transpose")?;
            (embed_tokens, embed_tokens_t)
        };
        let final_norm = weight_to_tensor(&weights.final_norm, device).context("final_norm")?;
        let rotary_inv_freq =
            compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .context("rotary_inv_freq")?;
        let projection_load_cache =
            ProjectionLoadCache::new(device).context("projection load cache")?;

        // Per-layer `pack_from_bf16` used to run inline during weight load,
        // serializing ~104 calls (8 × q_proj + 96 × MLP gate/up/down) behind
        // a single thread. At ~58s cold load on the Qwen3.5-4B A6000 build
        // this is a significant fraction of server startup. Sidecar the
        // pack inputs here, batch-pack via rayon after the layer loop, and
        // install results into the per-layer slots.
        let w4a16_enabled = crate::marlin_proj::env_enabled();
        let mut marlin_pack_inputs: Vec<(Tensor, i32)> = Vec::new();
        let mut marlin_pack_meta: Vec<MarlinPackEntry> = Vec::new();

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, lw) in weights.layers.iter().enumerate() {
            let ctx = |name: &str| format!("layer {i} {name}");

            let input_layernorm =
                weight_to_tensor(&lw.input_layernorm, device).context(ctx("input_layernorm"))?;
            let post_attention_layernorm = weight_to_tensor(&lw.post_attention_layernorm, device)
                .context(ctx("post_attention_layernorm"))?;

            let attention = match &lw.attention {
                crate::weights::AttentionWeights::Full(attn) => {
                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("q_proj", &attn.q_proj),
                            ("k_proj", &attn.k_proj),
                            ("v_proj", &attn.v_proj),
                            ("o_proj", &attn.o_proj),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (q_proj, q_proj_t) = attn_proj.next().context(ctx("q_proj missing"))?;
                    let (k_proj, k_proj_t) = attn_proj.next().context(ctx("k_proj missing"))?;
                    let (v_proj, v_proj_t) = attn_proj.next().context(ctx("v_proj missing"))?;
                    let (o_proj, o_proj_t) = attn_proj.next().context(ctx("o_proj missing"))?;
                    // KILN_W4A16=1 opt-in: queue q_proj for the post-loop
                    // Marlin batch pack. The packed weight (and the BF16
                    // drop) are installed after the layer loop via
                    // `install_marlin_packed`, so `q_proj_marlin` starts as
                    // None and `q_proj_t` keeps the BF16 copy until then.
                    if w4a16_enabled {
                        marlin_pack_inputs.push((q_proj_t.clone(), 128));
                        marlin_pack_meta.push(MarlinPackEntry {
                            layer_idx: i,
                            kind: MarlinPackKind::QProj,
                        });
                    }
                    GpuAttentionWeights::Full(GpuFullAttentionWeights {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm: weight_to_tensor(&attn.q_norm, device).context(ctx("q_norm"))?,
                        k_norm: weight_to_tensor(&attn.k_norm, device).context(ctx("k_norm"))?,
                        q_proj_t,
                        k_proj_t,
                        v_proj_t,
                        o_proj_t,
                        q_proj_marlin: None,
                    })
                }
                crate::weights::AttentionWeights::Linear(attn) => {
                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("in_proj_qkv", &attn.in_proj_qkv),
                            ("in_proj_z", &attn.in_proj_z),
                            ("out_proj", &attn.out_proj),
                            ("in_proj_a", &attn.in_proj_a),
                            ("in_proj_b", &attn.in_proj_b),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("linear attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (in_proj_qkv, in_proj_qkv_t) =
                        attn_proj.next().context(ctx("in_proj_qkv missing"))?;
                    let (in_proj_z, in_proj_z_t) =
                        attn_proj.next().context(ctx("in_proj_z missing"))?;
                    let (out_proj, out_proj_t) =
                        attn_proj.next().context(ctx("out_proj missing"))?;
                    let (in_proj_a, in_proj_a_t) =
                        attn_proj.next().context(ctx("in_proj_a missing"))?;
                    let (in_proj_b, in_proj_b_t) =
                        attn_proj.next().context(ctx("in_proj_b missing"))?;
                    GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                        in_proj_qkv,
                        in_proj_z,
                        out_proj,
                        in_proj_a,
                        in_proj_b,
                        conv1d: weight_to_tensor(&attn.conv1d, device).context(ctx("conv1d"))?,
                        norm: weight_to_tensor(&attn.norm, device).context(ctx("gdn_norm"))?,
                        a_log: weight_to_tensor(&attn.a_log, device).context(ctx("a_log"))?,
                        dt_bias: weight_to_tensor(&attn.dt_bias, device).context(ctx("dt_bias"))?,
                        in_proj_qkv_t,
                        in_proj_z_t,
                        in_proj_a_t,
                        in_proj_b_t,
                        out_proj_t,
                    })
                }
            };

            let mlp_proj = projection_tensors_for_load_batch(
                &[
                    ("gate_proj", &lw.mlp.gate_proj),
                    ("up_proj", &lw.mlp.up_proj),
                    ("down_proj", &lw.mlp.down_proj),
                ],
                device,
                &projection_load_cache,
            )
            .context(ctx("mlp projection tensors"))?;
            let mut mlp_proj = mlp_proj.into_iter();
            let (gate_proj, gate_proj_t) = mlp_proj.next().context(ctx("gate_proj missing"))?;
            let (up_proj, up_proj_t) = mlp_proj.next().context(ctx("up_proj missing"))?;
            let (down_proj, down_proj_t) = mlp_proj.next().context(ctx("down_proj missing"))?;
            // KILN_W4A16=1 opt-in: queue each MLP projection for the
            // post-loop Marlin batch pack. See the q_proj comment above —
            // the `*_proj_marlin` fields start as None, and
            // `install_marlin_packed` drops `*_proj_t` after the batch runs
            // (unless `KILN_DISABLE_MARLIN_BF16_DROP=1`).
            if w4a16_enabled {
                marlin_pack_inputs.push((gate_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::GateProj,
                });
                marlin_pack_inputs.push((up_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::UpProj,
                });
                marlin_pack_inputs.push((down_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::DownProj,
                });
            }
            let mlp = GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            };

            layers.push(GpuLayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention,
                mlp,
            });
        }

        // Batch-pack the queued Marlin projections in parallel. On
        // Qwen3.5-4B this is 8 × q_proj + 96 × MLP = 104 projections. The
        // CPU-bound `quantize_and_pack` work now runs across every
        // available worker thread (rayon's default pool) while the
        // GPU↔CPU copies stay sequential inside
        // `pack_from_bf16_batch`. Set `KILN_DISABLE_PARALLEL_PACK=1` to
        // force the legacy serial pack for A/B measurements or rollback.
        if w4a16_enabled && !marlin_pack_inputs.is_empty() {
            let pack_start = std::time::Instant::now();
            let packed = crate::marlin_proj::pack_from_bf16_batch(&marlin_pack_inputs)
                .context("marlin batch pack")?;
            let pack_elapsed_ms = pack_start.elapsed().as_millis();
            let parallel = !crate::marlin_proj::parallel_pack_disabled();
            let n_inputs = marlin_pack_inputs.len();
            let n_packed = packed.iter().filter(|p| p.is_some()).count();
            tracing::info!(
                n_inputs,
                n_packed,
                pack_elapsed_ms = pack_elapsed_ms as u64,
                parallel,
                "marlin batch pack complete"
            );
            eprintln!(
                "[kiln] marlin batch pack: {n_packed}/{n_inputs} projections in {pack_elapsed_ms} ms ({})",
                if parallel { "parallel" } else { "serial" }
            );

            let drop_disabled = marlin_bf16_drop_disabled();
            for (entry, maybe_packed) in marlin_pack_meta.into_iter().zip(packed.into_iter()) {
                if let Some(p) = maybe_packed {
                    install_marlin_packed(
                        &mut layers[entry.layer_idx],
                        entry.kind,
                        p,
                        device,
                        drop_disabled,
                    )
                    .with_context(|| {
                        format!(
                            "install marlin {:?} on layer {}",
                            entry.kind, entry.layer_idx
                        )
                    })?;
                }
            }
        }

        // Keep MTP routing support visible but do not upload native MTP tensors
        // during model load. The macOS desktop default only uses native MTP for
        // short greedy prompts; long prompts route to skip-layer, so eager MTP
        // upload slows common startup/readiness without warming the hot path.
        let mtp = weights
            .mtp
            .as_ref()
            .map(|mtp_w| MtpGpuWeightsSlot::lazy(mtp_w.clone(), device));

        Ok(Self {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward pass primitives
// ---------------------------------------------------------------------------

/// Look up token embeddings from the embedding table.
///
/// `token_ids`: 1-D slice of token IDs.
/// `embed_weights`: [vocab_size, hidden_size] embedding matrix.
///
/// Returns: [seq_len, hidden_size] tensor.
pub fn embedding_lookup(token_ids: &[u32], embed_weights: &Tensor) -> Result<Tensor> {
    let index = Tensor::new(token_ids, embed_weights.device())?;
    let out = embed_weights.index_select(&index, 0)?;
    Ok(out)
}

fn embedding_lookup_from_weights(token_ids: &[u32], weights: &GpuWeights) -> Result<Tensor> {
    let t_dims = weights.embed_tokens_t.dims();
    if t_dims.len() == 2 {
        let expected_embed_dims = [t_dims[1], t_dims[0]];
        if weights.embed_tokens.dims() != expected_embed_dims.as_slice() {
            return embedding_lookup_from_transposed(token_ids, &weights.embed_tokens_t);
        }
    }
    embedding_lookup(token_ids, &weights.embed_tokens)
}

fn embedding_lookup_from_transposed(token_ids: &[u32], embed_tokens_t: &Tensor) -> Result<Tensor> {
    let index = Tensor::new(token_ids, embed_tokens_t.device())?;
    let gathered = embed_tokens_t.index_select(&index, 1)?;
    Ok(gathered.t()?.contiguous()?)
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps).
///
/// `x`: [..., hidden_size]
/// `weight`: [hidden_size] (learnable scale)
/// `eps`: small constant for numerical stability (1e-6 for Qwen3.5-4B)
///
/// Returns: same shape as `x`.
///
/// On CUDA+bf16 inputs within the kernel envelope (hidden <= 8192), dispatches
/// to `kiln_rmsnorm_kernel::fused_rmsnorm`, which collapses the ~11 candle op
/// launches (to_dtype, sqr, mean_keepdim, +eps, sqrt, recip, broadcast_mul,
/// to_dtype, ones_like + w, broadcast_mul, to_dtype) into a single fused kernel.
/// Falls back to the candle-op path on CPU, non-bf16, or out-of-envelope inputs.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        if !disabled && kiln_rmsnorm_kernel::supports(x, weight) {
            return kiln_rmsnorm_kernel::fused_rmsnorm(x, weight, eps as f32)
                .context("fused_rmsnorm kernel failed");
        }
    }
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_rms_norm_supports(x, weight) {
            return crate::backend::metal::metal_rms_norm_bf16(x, weight, eps as f32)
                .context("metal rms_norm kernel failed");
        }
    }
    rms_norm_fallback(x, weight, eps)
}

/// Candle-op reference RMSNorm. Kept as the CPU path and as the correctness
/// oracle for the fused CUDA kernel. Matches HF semantics exactly:
/// `out = (1 + w) * x * rsqrt(mean(x^2) + eps)` with F32 reduction and epilogue.
pub fn rms_norm_fallback(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    // Qwen3.5 RMSNorm stores weights centered around 0 and applies as (1 + w) * x_normed.
    // Keep everything in F32 for precision (matches HF: `output * (1.0 + self.weight.float())`),
    // then cast back to input dtype at the end.
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let out = normed.broadcast_mul(&w_plus_one)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// `q`: [batch, seq_len, num_heads, head_dim]
/// `k`: [batch, seq_len, num_kv_heads, head_dim]
/// `positions`: position index for each token in the sequence (length = seq_len)
/// `head_dim`: dimension of each attention head
/// `rotary_dim`: number of head dimensions to apply rotation to (the rest pass through unchanged).
///   For Qwen3.5-4B: 64 (partial_rotary_factor=0.25, so 0.25 * 256 = 64).
/// `inv_freq`: cached frequency table of shape `[rotary_dim / 2]` (F32 on same device as `q`/`k`).
///   Build once via [`compute_rotary_inv_freq`] and reuse across calls.
///
/// Returns: (rotated_q, rotated_k) with same shapes.
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();

    // Position tensor
    let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos = Tensor::new(pos_f32.as_slice(), device)?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: [seq_len, half_rotary]
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?; // [seq_len, half_rotary]
    let sin = freqs.sin()?; // [seq_len, half_rotary]

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Same as [`rotary_embedding`] but accepts positions as a pre-allocated GPU tensor
/// instead of a CPU slice. This is critical for CUDA graph compatibility: the tensor's
/// GPU address stays stable across graph replays, and its contents can be updated via
/// `cudaMemcpyAsync` outside the captured graph.
///
/// `positions_tensor`: f32 tensor on device, shape [seq_len]
/// `inv_freq`: cached frequency table, shape `[rotary_dim / 2]`, F32 on device.
pub fn rotary_embedding_from_tensor(
    q: &Tensor,
    k: &Tensor,
    positions_tensor: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // positions_tensor is [seq_len], unsqueeze to [seq_len, 1]
    let pos = positions_tensor.unsqueeze(1)?;

    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

fn rotary_tables_from_tensor(
    positions_tensor: &Tensor,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let pos = positions_tensor.unsqueeze(1)?;
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn rotary_embedding_from_tables(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_rotary_embedding_supports(
            q, k, cos, sin, head_dim, rotary_dim,
        ) {
            return crate::backend::metal::metal_rotary_embedding_bf16(
                q, k, cos, sin, head_dim, rotary_dim,
            )
            .context("metal rotary embedding kernel failed");
        }
    }
    let rotated_q = apply_rope(q, cos, sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, cos, sin, head_dim, rotary_dim)?;
    Ok((rotated_q, rotated_k))
}

/// Apply the rotation to a single tensor, supporting partial rotary embeddings.
/// `x`: [batch, seq_len, num_heads, head_dim]
/// `cos`, `sin`: [seq_len, half_rotary]
/// `head_dim`: total dimension per head
/// `rotary_dim`: number of dimensions to rotate (must be even). The first `rotary_dim` dims
///   are rotated; the remaining `head_dim - rotary_dim` dims pass through unchanged.
fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    let half_rotary = rotary_dim / 2;
    let x_dtype = x.dtype();

    // Work in f32 for precision
    let x = x.to_dtype(DType::F32)?;

    // Split into rotary portion and passthrough portion
    let x_rot = x.narrow(candle_core::D::Minus1, 0, rotary_dim)?; // [..., :rotary_dim]
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?) // [..., rotary_dim:]
    } else {
        None
    };

    // Split rotary portion into two halves
    let x1 = x_rot.narrow(candle_core::D::Minus1, 0, half_rotary)?; // [..., :half_rotary]
    let x2 = x_rot.narrow(candle_core::D::Minus1, half_rotary, half_rotary)?; // [..., half_rotary:rotary_dim]

    // cos/sin are [seq_len, half_rotary], need to broadcast to [batch, seq_len, num_heads, half_rotary]
    // Reshape to [1, seq_len, 1, half_rotary]
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;

    // Standard RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    // Concatenate rotated dims + passthrough dims
    let out = match x_pass {
        Some(pass) => Tensor::cat(&[&r1, &r2, &pass], candle_core::D::Minus1)?,
        None => Tensor::cat(&[&r1, &r2], candle_core::D::Minus1)?,
    };
    Ok(out.to_dtype(x_dtype)?)
}

/// SwiGLU feed-forward network.
///
/// Computes: down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
///
/// `x`: [batch, seq_len, hidden_size]
/// `mlp`: MLP weight bundle, including optional Marlin W4A16-packed projections.
///
/// Dispatch each projection through the Marlin W4A16 path when the matching
/// `*_marlin` field is `Some`, else the existing BF16 `broadcast_matmul(*_t)`
/// path. LoRA deltas are always added on top so behaviour matches
/// `linear_with_lora_t` in the absence of Marlin weights. Mirrors
/// `q_proj_forward`'s Marlin routing from PR #149.
///
/// Returns: [batch, seq_len, hidden_size]
pub fn swiglu_ffn(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    // x @ gate_proj_t -> [batch, seq_len, intermediate_size]
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward(
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    // SiLU activation: x * sigmoid(x)
    let gate = cuda_silu(&gate)?;
    // x @ up_proj_t -> [batch, seq_len, intermediate_size]
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward(
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    // Element-wise multiply
    let hidden = (gate * up)?;
    // hidden @ down_proj_t -> [batch, seq_len, hidden_size]
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        mlp_proj_forward(
            &hidden,
            &mlp.down_proj_t,
            mlp.down_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.down_proj.as_ref()),
            lora_scale,
        )?
    };
    Ok(out)
}

/// Phase B12: sub-op-tapping variant of [`swiglu_ffn`]. Structurally
/// identical — same projections, same SiLU, same `gate * up` elementwise,
/// same down projection — but with three [`capture_b12_gqa_tap`] calls so
/// the HF comparator can localize drift to one of mlp_gate / mlp_up /
/// mlp_down on layer 31.
///
/// Called from [`transformer_block_paged`] only when
/// [`crate::mtp_debug::current_b12_layer_is_31`] is true, so the hot
/// production path continues to go through `swiglu_ffn` untouched.
fn swiglu_ffn_b12_tapped(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    // mlp_gate: output of the gate projection BEFORE SiLU. This matches the
    // HF reference which taps `self.gate_proj(x)` pre-activation.
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward(
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_gate", &gate)?;
    let gate = cuda_silu(&gate)?;
    // mlp_up: output of the up projection.
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward(
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_up", &up)?;
    let hidden = (gate * up)?;
    // mlp_down: final hidden-size output after the down projection.
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        mlp_proj_forward(
            &hidden,
            &mlp.down_proj_t,
            mlp.down_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.down_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_down", &out)?;
    Ok(out)
}

/// Route a single MLP projection through Marlin W4A16 when packed weights are
/// present, else fall back to the BF16 `linear_with_lora_t` path. LoRA deltas
/// are added on top of either base matmul. Mirrors `q_proj_forward`'s routing.
fn mlp_proj_forward(
    x: &Tensor,
    weight_t: &Tensor,
    marlin: Option<&crate::marlin_proj::MarlinPackedProj>,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(packed) = marlin {
        let base = crate::marlin_proj::matmul_bf16(x, packed)
            .context("mlp_proj_forward: marlin matmul")?;
        if let Some(proj) = lora {
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("mlp_proj_forward: lora delta")?;
            return Ok((base + delta).context("mlp_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    // Non-CUDA builds never carry Marlin weights; reference the parameter so
    // the signature stays unified without a dead_code warning.
    let _ = marlin;
    linear_with_lora_t(x, weight_t, lora, lora_scale)
}

fn lm_head_forward(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_bf16(x, embed_tokens_t)
                .context("metal lm_head kernel failed");
        }
    }
    Ok(x.broadcast_matmul(embed_tokens_t)?)
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) linear attention primitives
// ---------------------------------------------------------------------------

/// L2 normalize the last dimension: x / sqrt(sum(x^2) + eps).
/// Returns result in F32 regardless of input dtype.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq_sum = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let norm = (sq_sum + 1e-6)?.sqrt()?;
    let normalized = x_f32.broadcast_div(&norm)?;
    Ok(normalized)
}

fn gdn_qk_norm(q: &Tensor, k: &Tensor, input_dtype: DType, scale: f64) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "metal")]
    {
        if input_dtype == DType::BF16 && crate::backend::metal::metal_gdn_qk_norm_supports(q, k) {
            return crate::backend::metal::metal_gdn_qk_norm_f32_bf16(q, k, scale as f32, 1e-6)
                .context("metal gdn qk_norm kernel failed");
        }
    }

    #[cfg(feature = "cuda")]
    {
        let enabled = std::env::var("KILN_ENABLE_FUSED_L2_QK_NORM").is_ok();
        if enabled && input_dtype == DType::BF16 && kiln_rmsnorm_kernel::supports_l2_qk_norm(q, k) {
            return kiln_rmsnorm_kernel::fused_l2_qk_norm(q, k, scale as f32, 1e-6)
                .context("fused_l2_qk_norm kernel failed");
        }
    }

    let q = l2_normalize(q)?; // F32
    let k = l2_normalize(k)?; // F32
    let q = (q * scale)?.to_dtype(input_dtype)?;
    let k = k.to_dtype(input_dtype)?;
    Ok((q, k))
}

/// softplus(x) = ln(1 + exp(x)), numerically stable for all x.
///
/// Uses the identity: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
/// Since exp(-|x|) ∈ (0, 1], no overflow is possible.
/// This matches PyTorch's F.softplus output (which clamps to linear for x > 20).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let relu_x = x.maximum(&zeros)?;
    // |x| = relu(x) + relu(-x)
    let neg_x = x.neg()?;
    let relu_neg_x = neg_x.maximum(&zeros)?;
    let abs_x = (relu_x.clone() + relu_neg_x)?;
    let neg_abs = abs_x.neg()?;
    // log(1 + exp(-|x|)) — always stable since exp(-|x|) ∈ (0, 1]
    let log_term = (neg_abs.exp()? + 1.0)?.log()?;
    Ok((relu_x + log_term)?)
}

fn gated_rms_norm(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if backend.supports_gdn_gated_rms_norm() {
        if let Some(out) = backend.gdn_gated_rms_norm(x, z, weight, eps)? {
            return Ok(out);
        }
    }
    gated_rms_norm_fallback(x, z, weight, eps)
}

/// Gated RMSNorm: rms_norm(x, weight) * silu(z).
///
/// Applied per-group on the last dimension. Returns F32.
///
/// `x`: [..., dim] — attention output
/// `z`: [..., dim] — output gate (from in_proj_z)
/// `weight`: [dim] — learnable scale
fn gated_rms_norm_fallback(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let z_f32 = z.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;

    // RMS norm on last dimension
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    let normed = normed.broadcast_mul(&w_f32)?;

    // Output gate: silu(z) = z * sigmoid(z)
    let gate = cuda_silu(&z_f32)?;
    let out = (normed * gate)?;
    Ok(out)
}

/// Causal depthwise conv1d for prefill (seq_len > 1).
///
/// `x`: [batch, channels, seq_len]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated to last K-1 inputs
fn causal_conv1d_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let compute_dtype = causal_conv1d_prefill_compute_dtype(x, weight, conv_state, kernel_size);
    causal_conv1d_prefill_with_dtype(x, weight, conv_state, kernel_size, compute_dtype)
}

fn causal_conv1d_prefill_compute_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> DType {
    if matches!(x.device(), Device::Metal(_))
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && conv_state.dtype() == DType::F32
        && kernel_size == 4
    {
        DType::BF16
    } else {
        DType::F32
    }
}

fn causal_conv1d_prefill_with_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
    compute_dtype: DType,
) -> Result<Tensor> {
    let (_batch, channels, seq_len) = x.dims3()?;
    let x_compute = x.to_dtype(compute_dtype)?;
    let x_state_f32 = if compute_dtype == DType::F32 {
        x_compute.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    // Squeeze [channels, 1, kernel_size] -> [channels, kernel_size]
    let w_compute = weight
        .to_dtype(compute_dtype)?
        .reshape((channels, kernel_size))?;
    let k_minus_1 = kernel_size - 1;

    // Left-pad with conv_state (previous K-1 inputs, or zeros for fresh state)
    let x_padded = Tensor::cat(&[&conv_state.to_dtype(compute_dtype)?, &x_compute], 2)?;

    // Depthwise conv: output[t] = sum_{j=0}^{K-1} weight[j] * x_padded[t+j]
    let mut output = Tensor::zeros_like(&x_compute)?;
    for j in 0..kernel_size {
        let x_slice = x_padded.narrow(2, j, seq_len)?;
        let w_j = w_compute.narrow(1, j, 1)?.unsqueeze(0)?; // [1, channels, 1]
        output = (output + x_slice.broadcast_mul(&w_j)?)?;
    }

    // Update conv_state to the last K-1 input positions
    if seq_len >= k_minus_1 {
        *conv_state = x_state_f32
            .narrow(2, seq_len - k_minus_1, k_minus_1)?
            .contiguous()?;
    } else {
        // Fewer new tokens than buffer size: shift old state and append new
        let keep = k_minus_1 - seq_len;
        let old_part = conv_state.narrow(2, seq_len, keep)?;
        *conv_state = Tensor::cat(&[&old_part, &x_state_f32], 2)?.contiguous()?;
    }

    Ok(output)
}

/// Causal depthwise conv1d for decode (seq_len == 1).
///
/// `x`: [batch, channels, 1]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated
fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (_batch, channels, _one) = x.dims3()?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight
        .to_dtype(DType::F32)?
        .reshape((channels, kernel_size))?;

    // Full window = [conv_state | x] -> [batch, channels, kernel_size]
    let window = Tensor::cat(&[&conv_state.to_dtype(DType::F32)?, &x_f32], 2)?;

    // Dot product per channel: sum over kernel dimension
    let w_expanded = w_f32.unsqueeze(0)?; // [1, channels, kernel_size]
    let output = window.broadcast_mul(&w_expanded)?.sum(2)?; // [batch, channels]
    let output = output.unsqueeze(2)?; // [batch, channels, 1]

    // Update conv_state: drop oldest, append newest
    *conv_state = window.narrow(2, 1, kernel_size - 1)?.contiguous()?;

    Ok(output)
}

// ---------------------------------------------------------------------------
// GDN chunkwise analytical recurrence (Phase 6, approach (b) in the chunkwise
// plan). Replaces the per-token `for t in 0..seq_len` loop inside
// `gated_deltanet_forward` with an unrolled form that processes up to
// `GDN_CHUNK_SIZE` tokens per heavy matmul, dropping the number of GPU kernel
// launches from O(T) to O(T / C) per layer.
// ---------------------------------------------------------------------------

/// Chunk size for the analytical GDN recurrence. C = 64 balances:
///   - intra-chunk [C, dk] × [dk, C] matmuls large enough to saturate tensor
///     cores on A5000/4090-class GPUs for dk = dv = 128,
///   - a small-enough forward-substitution inner loop so the Vec<Tensor> cat
///     churn stays bounded.
const GDN_CHUNK_SIZE: usize = 64;
const GDN_RECURRENT_PREFILL_MAX_TOKENS: usize = 2048;

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row > col else 0.0.
/// Used for the strictly lower-triangular `A_strict` mask (i < t, exclusive).
fn strict_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let t = Tensor::arange(0u32, n as u32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(rows.gt(&cols)?.to_dtype(dtype)?)
}

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row >= col else 0.0.
/// Used for the causal (inclusive) lower-triangular `B_mask` mask (i <= t).
fn causal_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let t = Tensor::arange(0u32, n as u32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(rows.ge(&cols)?.to_dtype(dtype)?)
}

/// Compute the chunk-local W = (I + A_strict)^{-1} (beta * V_prime) by
/// forward substitution. On backends that advertise
/// `supports_gdn_forward_substitution()` (CUDA/Metal bf16 today), dispatches
/// to the fused kernel (one kernel block per (batch, head)) when
/// `chunk_size <= 128`. Otherwise it falls back to the per-token candle loop.
fn compute_w_chunk(
    backend: &dyn BackendRuntime,
    a_strict: &Tensor, // [B, nv, C, C]
    v_prime: &Tensor,  // [B, nv, C, dv]
    beta_c: &Tensor,   // [B, nv, C]
    c: usize,
) -> Result<Tensor> {
    // The kernel envelope is C <= 128; callers enforce this precondition so
    // we never pay for a backend call we know will decline.
    if c <= 128 && backend.supports_gdn_forward_substitution() {
        kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
        if let Some(out) = backend.gdn_forward_substitution(a_strict, v_prime, beta_c)? {
            return Ok(out);
        }
    }
    compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)
}

/// Reference per-token forward substitution. Kept as the CPU path and as
/// the correctness oracle for the fused CUDA kernel.
fn compute_w_chunk_fallback(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta_c: &Tensor,
    c: usize,
) -> Result<Tensor> {
    let beta_col = beta_c.unsqueeze(3)?; // [B, nv, C, 1]
    let mut w_rows: Vec<Tensor> = Vec::with_capacity(c);
    for t in 0..c {
        let vp_t = v_prime.narrow(2, t, 1)?; // [B, nv, 1, dv]
        let beta_t = beta_col.narrow(2, t, 1)?; // [B, nv, 1, 1]
        let w_t = if t == 0 {
            vp_t.broadcast_mul(&beta_t)?
        } else {
            let a_row = a_strict.narrow(2, t, 1)?.narrow(3, 0, t)?.contiguous()?;
            let w_prev = Tensor::cat(&w_rows, 2)?;
            let sub = a_row.matmul(&w_prev)?; // [B, nv, 1, dv]
            (vp_t - sub)?.broadcast_mul(&beta_t)?
        };
        w_rows.push(w_t);
    }
    Ok(Tensor::cat(&w_rows, 2)?)
}

fn compute_chunk_body_reference(
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta_c: &Tensor,
    decay_last_col_u: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let c = v_prime.dim(2)?;
    let w = compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)?;
    let intra = b_mask.matmul(&w)?;
    let out_chunk = (q_s_scaled + &intra)?;
    let w_weighted = w.broadcast_mul(decay_last_col_u)?.contiguous()?;
    Ok((out_chunk, w_weighted))
}

/// Specialized single-token GDN recurrence.
///
/// This is the non-CUDA fast path for `seq_len == 1`, avoiding the chunkwise
/// prep work (`KKT`, `QKT`, masks, triangular solve) that is only worthwhile
/// when a chunk contains multiple tokens.
fn gdn_single_token_recurrence(
    q: &Tensor,         // [B, nv, 1, dk]
    k: &Tensor,         // [B, nv, 1, dk]
    v: &Tensor,         // [B, nv, 1, dv]
    beta: &Tensor,      // [B, nv, 1]
    g: &Tensor,         // [B, nv, 1]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Tensor> {
    let dtype = q.dtype();

    let p = g.to_dtype(DType::F32)?.exp()?.to_dtype(dtype)?;
    let p_u = p.unsqueeze(3)?; // [B, nv, 1, 1]

    let k_t = k.transpose(2, 3)?.contiguous()?; // [B, nv, dk, 1]
    let ks_entry = k.matmul(&*state)?; // [B, nv, 1, dv]
    let q_s = q.matmul(&*state)?; // [B, nv, 1, dv]

    let v_prime = (v - ks_entry.broadcast_mul(&p_u)?)?;
    let w = v_prime.broadcast_mul(&beta.unsqueeze(3)?)?; // [B, nv, 1, dv]
    let qk = q.matmul(&k_t)?; // [B, nv, 1, 1]
    let out = (q_s.broadcast_mul(&p_u)? + qk.matmul(&w)?)?;

    let state_scaled = state.broadcast_mul(&p_u)?;
    let delta_state = k_t.matmul(&w)?;
    *state = (state_scaled + delta_state)?;

    Ok(out)
}

/// Analytical chunkwise form of the Gated DeltaNet recurrence.
///
/// The per-token recurrence is
///
/// ```text
///   S_t   = exp(g_t) * S_{t-1}  +  k_t ⊗ delta_t
///   delta_t = beta_t * (v_t - k_t · (exp(g_t) * S_{t-1}))
///   out_t = q_t · S_t
/// ```
///
/// Within a chunk of up to `chunk_size` tokens, let `G[t] = cumsum(g)[t]`.
/// The per-token recurrence unrolls into the closed form (derived from the
/// standard GLA / chunk_gla_fwd identity used in fla-org and RWKV-5):
///
/// 1. Inter-chunk carry
///    ```text
///      V'[t] = v[t] - exp(G[t]) * (k[t] · S_entry)
///    ```
/// 2. Strict intra-chunk decay mask
///    ```text
///      A_strict[t, i] = exp(G[t] - G[i]) * (k[t] · k[i])   for i < t, else 0
///    ```
/// 3. Forward-substitution / triangular solve for W[t]
///    ```text
///      W[t] = beta[t] * ( V'[t] - Σ_{i<t} A_strict[t, i] * W[i] )
///    ```
/// 4. Output
///    ```text
///      B_mask[t, i] = exp(G[t] - G[i]) * (q[t] · k[i])     for i <= t, else 0
///      out[t] = exp(G[t]) * (q[t] · S_entry) + Σ_{i<=t} B_mask[t, i] * W[i]
///    ```
/// 5. State exit
///    ```text
///      S_new = exp(G[C-1]) * S_entry + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
///    ```
///
/// This is numerically equivalent to the per-token loop (modulo rounding in
/// the bf16 hot path) and matches the pre-existing sequential code exactly
/// for chunk_size = 1 (decode path).
///
/// Inputs are already transposed to `[B, nv, T, *]` layout. `state` is
/// mutated in place and must be in the hot-path dtype (bf16 in production,
/// F32 on CPU tests); the caller is responsible for preserving the external
/// F32-state invariant.
///
/// Returns: `[B, nv, T, dv]`.
fn gdn_chunkwise_recurrence(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Tensor> {
    let (_, _, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    let device = q.device();

    // Single-token decode fast path. The chunkwise machinery (preshape,
    // decay matrix, KKT, forward sub, B_mask) costs more than the per-token
    // recurrence itself when seq_len == 1, which is the cause of the −54%
    // decode regression in PR #80. The backend's `gdn_recurrent_step`
    // kernel (CUDA today) collapses the whole recurrence into one block
    // per (B,H).
    if seq_len == 1 {
        if dtype == DType::BF16
            && state.dtype() == DType::BF16
            && backend.supports_gdn_recurrent_step()
        {
            // The five squeeze+contiguous calls below each emit a bf16 ucopy
            // kernel before the recurrent forward runs. PROFILING.md (PR #107)
            // marks this block as the suspected source of the 24-GDN-layer
            // ucopy_bf16 slice; the dedicated NVTX range lets nsys attribute
            // it separately from the kernel itself.
            let (q1, k1, v1, beta1, g1) = {
                kiln_nvtx::range!(c"kiln/attn/gdn/precopy");
                (
                    q.squeeze(2)?.contiguous()?,
                    k.squeeze(2)?.contiguous()?,
                    v.squeeze(2)?.contiguous()?,
                    beta.squeeze(2)?.contiguous()?,
                    g.squeeze(2)?.contiguous()?,
                )
            };
            let out_opt = {
                kiln_nvtx::range!(c"kiln/attn/gdn/recurrent");
                backend.gdn_recurrent_step(&q1, &k1, &v1, &beta1, &g1, state)?
            };
            if let Some(out) = out_opt {
                return Ok(out.unsqueeze(2)?);
            }
        }

        return gdn_single_token_recurrence(q, k, v, beta, g, state);
    }

    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;

    // Slice full chunks directly. On macOS Metal this avoids the large upfront
    // pre-permute copies that dominated long-prompt GDN recurrence time.

    let mut out_chunks: Vec<Tensor> = Vec::with_capacity(seq_len.div_ceil(chunk_size));

    for ci in 0..(full_chunks + if tail > 0 { 1 } else { 0 }) {
        let is_tail = ci >= full_chunks;
        let c = if is_tail { tail } else { chunk_size };

        let (q_c, k_c, v_c, beta_c, g_c) = if is_tail {
            let t_start = full_chunks * chunk_size;
            (
                q.narrow(2, t_start, tail)?.contiguous()?,
                k.narrow(2, t_start, tail)?.contiguous()?,
                v.narrow(2, t_start, tail)?.contiguous()?,
                beta.narrow(2, t_start, tail)?.contiguous()?,
                g.narrow(2, t_start, tail)?.contiguous()?,
            )
        } else {
            let t_start = ci * chunk_size;
            (
                q.narrow(2, t_start, chunk_size)?.contiguous()?,
                k.narrow(2, t_start, chunk_size)?.contiguous()?,
                v.narrow(2, t_start, chunk_size)?.contiguous()?,
                beta.narrow(2, t_start, chunk_size)?.contiguous()?,
                g.narrow(2, t_start, chunk_size)?.contiguous()?,
            )
        };

        // Matmuls first — these are well-tuned cuBLAS GEMMs and stay on
        // candle. K^T is reused for KKT (intra-chunk similarities) and the
        // final outer product into the state update.
        let k_t_mat = k_c.transpose(2, 3)?.contiguous()?; // [B, nv, dk, C]
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = k_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let qkt = q_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]

        if !is_tail && c == 64 && backend.supports_gdn_full_chunk_forward() && dtype == DType::BF16
        {
            if let Some(out_chunk) = backend.gdn_full_chunk_forward(
                &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state,
            )? {
                out_chunks.push(out_chunk);
                continue;
            }
        }

        // Fused prep: cumsum + decay + exp + masked scales + v_prime +
        // q_s_scaled + decay_last_col + p_last in a single CUDA launch.
        // Falls back to the candle-op chain when the backend declines
        // (non-CUDA, non-bf16, envelope violation).
        //
        // Post-conditions on all four paths:
        //   a_strict:         [B, nv, C, C] bf16 — kkt * decay * strict_lower
        //   b_mask:           [B, nv, C, C] bf16 — qkt * decay * causal_lower
        //   v_prime:          [B, nv, C, dv] bf16 — v - ks_entry * p
        //   q_s_scaled:       [B, nv, C, dv] bf16 — q_s * p
        //   decay_last_col_u: [B, nv, C, 1]  bf16 — exp(big_g[C-1] - big_g[i])
        //   p_last_u:         [B, nv, 1, 1]  bf16 — exp(big_g[C-1])
        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col_u, p_last_u) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk_prep");
            let prep_out = if backend.supports_gdn_chunk_prep() && dtype == DType::BF16 {
                backend.gdn_chunk_prep(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?
            } else {
                None
            };
            match prep_out {
                Some((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)) => {
                    let decay_last_col_u = decay_last_col.unsqueeze(3)?; // [B,nv,C,1]
                    let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?; // [B,nv,1,1]
                    (
                        a_strict.contiguous()?,
                        b_mask.contiguous()?,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
                None => {
                    // Cumulative decay G[t] = Σ_{s=0..t} g[s].  Done in F32:
                    // exp() of the cumulative sum is the only place bf16
                    // would lose meaningful precision (G can reach -10 or
                    // more across a full 64-token chunk).
                    let g_f32 = g_c.to_dtype(DType::F32)?;
                    let big_g = g_f32.cumsum(candle_core::D::Minus1)?; // [B, nv, C], F32

                    // Decay matrix D[t, i] = exp(G[t] - G[i]).
                    let big_g_col = big_g.unsqueeze(3)?; // [B, nv, C, 1]
                    let big_g_row = big_g.unsqueeze(2)?; // [B, nv, 1, C]
                    let decay_f32 = big_g_col.broadcast_sub(&big_g_row)?.exp()?;
                    let decay = decay_f32.to_dtype(dtype)?; // back to hot dtype

                    // p[t] = exp(G[t]).
                    let p = big_g.exp()?.to_dtype(dtype)?; // [B, nv, C]
                    let p_col = p.unsqueeze(3)?; // [B, nv, C, 1]

                    let strict_mask = strict_lower_tri_mask(c, dtype, device)?;
                    let causal_mask = causal_lower_tri_mask(c, dtype, device)?;

                    let v_prime = (&v_c - ks_entry.broadcast_mul(&p_col)?)?;
                    let a_strict = kkt
                        .broadcast_mul(&decay)?
                        .broadcast_mul(&strict_mask)?
                        .contiguous()?;
                    let b_mask = qkt
                        .broadcast_mul(&decay)?
                        .broadcast_mul(&causal_mask)?
                        .contiguous()?;
                    let q_s_scaled = q_s.broadcast_mul(&p_col)?;

                    let g_last = big_g.narrow(2, c - 1, 1)?; // [B, nv, 1]
                    let decay_last_col_u = g_last
                        .broadcast_sub(&big_g)?
                        .exp()?
                        .to_dtype(dtype)?
                        .unsqueeze(3)?; // [B, nv, C, 1]
                    let p_last_u = g_last.exp()?.to_dtype(dtype)?.unsqueeze(3)?; // [B,nv,1,1]

                    (
                        a_strict,
                        b_mask,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
            }
        };

        let decay_last_col = decay_last_col_u.squeeze(3)?;
        let (out_chunk, w_weighted) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
            if backend.supports_gdn_chunk_scan() && dtype == DType::BF16 {
                match backend.gdn_chunk_scan(
                    &a_strict,
                    &b_mask,
                    &v_prime,
                    &q_s_scaled,
                    &beta_c,
                    &decay_last_col,
                )? {
                    Some((out_chunk, w_weighted)) => (out_chunk, w_weighted),
                    None => {
                        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                        let intra = b_mask.matmul(&w)?;
                        let out_chunk = (&q_s_scaled + &intra)?;
                        let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                        (out_chunk, w_weighted)
                    }
                }
            } else {
                let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                let intra = b_mask.matmul(&w)?;
                let out_chunk = (&q_s_scaled + &intra)?;
                let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                (out_chunk, w_weighted)
            }
        };

        out_chunks.push(out_chunk); // [B, nv, C, dv]

        // State update:
        //   S_new = exp(G[C-1]) * S_entry
        //         + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
        let state_scaled = state.broadcast_mul(&p_last_u)?; // [B, nv, dk, dv]
        let delta_state = k_t_mat.matmul(&w_weighted)?; // [B, nv, dk, dv]
        *state = (state_scaled + delta_state)?;
    }

    Ok(Tensor::cat(&out_chunks, 2)?)
}

fn gdn_recurrent_prefill_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, _, seq_len, _) = q.dims4()?;
    if seq_len <= 1
        || q.dtype() != DType::BF16
        || state.dtype() != DType::BF16
        || !backend.supports_gdn_recurrent_prefill_head_last()
    {
        return Ok(None);
    }
    backend.gdn_recurrent_prefill_head_last(q, k, v, beta, g, state)
}

fn gdn_recurrent_prefill_native_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, T, nk, dk]
    k: &Tensor,         // [B, T, nk, dk]
    v: &Tensor,         // [B, T, nv, dv]
    beta: &Tensor,      // [B, T, nv]
    g: &Tensor,         // [B, T, nv]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, seq_len, _, _) = q.dims4()?;
    if seq_len <= 1
        || q.dtype() != DType::BF16
        || state.dtype() != DType::BF16
        || !backend.supports_gdn_recurrent_prefill_native_head_last()
    {
        return Ok(None);
    }
    backend.gdn_recurrent_prefill_native_head_last(q, k, v, beta, g, state)
}

/// Metal BF16 fast path for full 64-token chunks.
///
/// Returns a contiguous head-last `[B, T, nv, dv]` tensor so the caller can feed
/// Metal gated RMSNorm without the `[B,nv,T,dv]` cat + transpose + contiguous
/// copy chain.
fn gdn_chunkwise_recurrence_head_last_full_chunks(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    let (batch, heads, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    if chunk_size != 64
        || seq_len <= 1
        || seq_len % chunk_size != 0
        || dtype != DType::BF16
        || state.dtype() != DType::BF16
        || !backend.supports_gdn_full_chunk_forward_head_last()
    {
        return Ok(None);
    }

    let dv = v.dim(3)?;
    let out = Tensor::zeros((batch, seq_len, heads, dv), DType::BF16, q.device())?;

    for ci in 0..(seq_len / chunk_size) {
        let t_start = ci * chunk_size;
        let q_c = q.narrow(2, t_start, chunk_size)?.contiguous()?;
        let k_c = k.narrow(2, t_start, chunk_size)?.contiguous()?;
        let v_c = v.narrow(2, t_start, chunk_size)?.contiguous()?;
        let beta_c = beta.narrow(2, t_start, chunk_size)?.contiguous()?;
        let g_c = g.narrow(2, t_start, chunk_size)?.contiguous()?;

        let k_t_mat = k_c.transpose(2, 3)?.contiguous()?; // [B, nv, dk, C]
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = k_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let qkt = q_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]

        if !backend.gdn_full_chunk_forward_head_last_into(
            &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state, &out, t_start,
            seq_len,
        )? {
            if ci == 0 {
                return Ok(None);
            }
            anyhow::bail!("backend declined GDN head-last full-chunk path mid-sequence");
        }
    }

    Ok(Some(out))
}

/// Gated DeltaNet (GDN) linear attention forward pass.
///
/// Implements the recurrent linear attention mechanism used by 24/32 layers in Qwen3.5-4B.
/// Uses data-dependent gating (alpha/beta) and a delta rule update for the recurrent state.
///
/// `x`: [batch, seq_len, hidden_size]
/// `weights`: linear attention projection weights
/// `config`: model configuration
/// `recurrent_state`: [batch, nv, dk, dv] — mutable recurrent state, updated in place
/// `conv_state`: [batch, conv_dim, kernel_size-1] — mutable conv buffer, updated in place
///
/// Returns: [batch, seq_len, hidden_size]

/// Candle-op reference path for the Step-6 GDN gates. This is the original
/// Phase-6 implementation; it's kept as a fallback for shapes/dtypes outside
/// the fused kernel's envelope and as the algorithmic oracle for parity tests.
///
/// beta = sigmoid(b)                                // bf16
/// g    = -exp(A_log) * softplus(a + dt_bias)       // bf16 (F32 intermediates)
fn gated_deltanet_gates_fallback(
    a: &Tensor,
    b: &Tensor,
    weights: &GpuLinearAttentionWeights,
    input_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let beta = cuda_sigmoid(b)?; // [B, T, nv], bf16
    let a_f32 = a.to_dtype(DType::F32)?;
    let a_log_f32 = weights.a_log.to_dtype(DType::F32)?;
    let dt_bias_f32 = weights.dt_bias.to_dtype(DType::F32)?;
    let g = {
        let a_biased = a_f32.broadcast_add(&dt_bias_f32)?;
        let sp = softplus(&a_biased)?;
        let neg_decay = a_log_f32.exp()?.neg()?; // -exp(A_log)
        sp.broadcast_mul(&neg_decay)?
    }
    .to_dtype(input_dtype)?;
    Ok((beta, g))
}

pub fn gated_deltanet_forward(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    capture_b11_taps: bool,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let input_dtype = x.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;

    // --- Step 1: Input projections ---
    // Use the pre-transposed weight cache (Phase 6) so we don't pay a `.t().contiguous()`
    // ucopy_bf16 copy on every layer / every step. Same fix class as PR #128 (MLP/full-attn).
    let (mixed_qkv, z, a, b) = {
        kiln_nvtx::range!(c"kiln/gdn/in_proj");
        let mixed_qkv = x.broadcast_matmul(&weights.in_proj_qkv_t)?; // [B, T, qkv_dim]
        let z = x.broadcast_matmul(&weights.in_proj_z_t)?; // [B, T, v_dim]
        let a = x.broadcast_matmul(&weights.in_proj_a_t)?; // [B, T, nv]
        let b = x.broadcast_matmul(&weights.in_proj_b_t)?; // [B, T, nv]
        (mixed_qkv, z, a, b)
    };

    // Phase B11b tap: `gdn_in_proj`. Matches the HF reference layout
    // `concat([in_proj_qkvz(x), in_proj_ba(x)], dim=-1)` = [q, k, v, z, b, a]
    // along the last axis. Capture once here so subsequent post-split
    // transforms don't alter what we're attributing divergence to.
    if capture_b11_taps {
        let gdn_in_proj = Tensor::cat(&[&mixed_qkv, &z, &b, &a], candle_core::D::Minus1)?;
        crate::mtp_debug::capture_b11_layer0_tap("gdn_in_proj", &gdn_in_proj)?;
    }

    // --- Step 2: Causal depthwise conv1d + SiLU on fused QKV ---
    //
    // Decode fast path: backend-side `causal_conv1d_update` collapses the
    // to_f32 / cat / sum / narrow / silu chain into one fused update per
    // (batch, channel). It returns F32 with SiLU already fused, so the
    // subsequent `cuda_silu(.to_dtype(F32))` step is skipped. Unsupported
    // backends, non-bf16, kernel_size != 4, and the `KILN_DISABLE_FUSED_CONV1D`
    // kill switch all route through the portable candle path below — which is the
    // parity oracle.
    let mixed_qkv = {
        kiln_nvtx::range!(c"kiln/gdn/conv");
        // Transpose to [B, channels, T] for conv
        let mixed_qkv_ct = mixed_qkv.transpose(1, 2)?.contiguous()?;
        let post_silu = if seq_len == 1 && backend.supports_causal_conv1d_update() {
            match backend.causal_conv1d_update(
                &mixed_qkv_ct,
                &weights.conv1d,
                conv_state,
                kernel_size,
            )? {
                Some(out) => out, // F32, SiLU fused into the kernel epilogue
                None => {
                    let y = causal_conv1d_decode(
                        &mixed_qkv_ct,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                    )?;
                    cuda_silu(&y.to_dtype(DType::F32)?)?
                }
            }
        } else if seq_len > 1 {
            if backend.supports_causal_conv1d_prefill() {
                match backend.causal_conv1d_prefill(
                    &mixed_qkv_ct,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                )? {
                    Some(out) => out, // F32, SiLU fused into the kernel epilogue
                    None => {
                        let y = causal_conv1d_prefill(
                            &mixed_qkv_ct,
                            &weights.conv1d,
                            conv_state,
                            kernel_size,
                        )?;
                        cuda_silu(&y)?
                    }
                }
            } else {
                let y =
                    causal_conv1d_prefill(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
                cuda_silu(&y)?
            }
        } else {
            let y = causal_conv1d_decode(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
            cuda_silu(&y.to_dtype(DType::F32)?)?
        };
        // Transpose back to [B, T, qkv_dim]
        post_silu.transpose(1, 2)?
    };

    // Phase B11b tap: `gdn_conv`. Output of the causal depthwise conv1d +
    // SiLU, matching HF's `mixed_qkv` after `self.conv1d(...)[:T]` +
    // `F.silu(...)` (shape [B, T, qkv_dim]).
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_conv", &mixed_qkv)?;
    }

    // --- Step 3: Split into Q, K, V and reshape to heads ---
    let (q, k, v, z) = {
        kiln_nvtx::range!(c"kiln/gdn/qkv_split");
        let q = mixed_qkv
            .narrow(2, 0, qk_dim)?
            .reshape((batch, seq_len, nk, dk))?;
        let k = mixed_qkv
            .narrow(2, qk_dim, qk_dim)?
            .reshape((batch, seq_len, nk, dk))?;
        let v = mixed_qkv
            .narrow(2, 2 * qk_dim, v_dim)?
            .reshape((batch, seq_len, nv, dv))?;
        let z = z.reshape((batch, seq_len, nv, dv))?;
        (q, k, v, z)
    };

    // --- Step 4/5: GQA head repeat (nk → nv), L2 normalize Q/K, scale Q ---
    //
    // Fast paths: Metal defaults to a fused F32->BF16 kernel for the desktop
    // hot path, and CUDA keeps its opt-in `kiln_rmsnorm_kernel::fused_l2_qk_norm`.
    // Both collapse the l2-normalize(Q) + scale(Q) + l2-normalize(K) +
    // dtype-cast chain (~11 candle launches on tiny per-row tensors at decode
    // shape) into a single launch.
    //
    // The CUDA fused kernel remains **opt-in via
    // `KILN_ENABLE_FUSED_L2_QK_NORM=1`**. Phase 6 wallclock validation on Arm B
    // (RTX A6000, KILN_W4A16=1
    // KILN_CUDA_GRAPHS=true, paged 512/128, 3 paired runs) measured a
    // median speedup of 1.0093x — well below the task's 1.05x abort floor.
    // Mean ITL improved by 0.18ms (0.92%); only p99 ITL showed a
    // meaningful win (24.78ms -> 20.25ms, -18% tail latency) and the
    // run-to-run variance tightened. The kernel is correct (parity tests
    // and the full nextest suite pass) but the wallclock impact at the
    // Qwen3.5-4B GDN decode shape does not meet
    // the bar to engage by default. See PROFILING.md "Phase 6 fused
    // qk_norm null result" for the full numbers and analysis.
    //
    // Both paths produce bf16 outputs in `input_dtype`; only the kernel
    // path skips the F32 round-trip through HBM. The candle path is the
    // parity oracle exercised by `kiln-rmsnorm-kernel`'s
    // `parity_l2_qk_norm_*` tests.
    let scale = 1.0 / (dk as f64).sqrt();
    let recurrent_prefill_unexpanded_qk = input_dtype == DType::BF16
        && seq_len > 1
        && seq_len <= GDN_RECURRENT_PREFILL_MAX_TOKENS
        && dk == 128
        && gqa_ratio > 1
        && !capture_b11_taps
        && backend.supports_gdn_recurrent_prefill_head_last();
    let (q, k, qk_expanded) = {
        #[cfg(feature = "metal")]
        {
            if recurrent_prefill_unexpanded_qk {
                kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                (q, k, false)
            } else if input_dtype == DType::BF16
                && gqa_ratio > 1
                && crate::backend::metal::metal_gdn_qk_norm_gqa_supports(&q, &k, nv)
            {
                kiln_nvtx::range!(c"kiln/gdn/qk_norm_gqa");
                crate::backend::metal::metal_gdn_qk_norm_gqa_f32_bf16(
                    &q,
                    &k,
                    nv,
                    scale as f32,
                    1e-6,
                )
                .context("metal gdn qk_norm gqa kernel failed")
                .map(|(q, k)| (q, k, true))?
            } else {
                let (q, k) = {
                    kiln_nvtx::range!(c"kiln/gdn/head_expand");
                    if gqa_ratio > 1 {
                        let q = q
                            .unsqueeze(3)?
                            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                            .contiguous()?
                            .reshape((batch, seq_len, nv, dk))?;
                        let k = k
                            .unsqueeze(3)?
                            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                            .contiguous()?
                            .reshape((batch, seq_len, nv, dk))?;
                        (q, k)
                    } else {
                        (q.contiguous()?, k.contiguous()?)
                    }
                };
                kiln_nvtx::range!(c"kiln/gdn/qk_norm");
                let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                (q, k, true)
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            let (q, k) = {
                kiln_nvtx::range!(c"kiln/gdn/head_expand");
                if gqa_ratio > 1 {
                    let q = q
                        .unsqueeze(3)?
                        .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                        .contiguous()?
                        .reshape((batch, seq_len, nv, dk))?;
                    let k = k
                        .unsqueeze(3)?
                        .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                        .contiguous()?
                        .reshape((batch, seq_len, nv, dk))?;
                    (q, k)
                } else {
                    (q.contiguous()?, k.contiguous()?)
                }
            };
            kiln_nvtx::range!(c"kiln/gdn/qk_norm");
            let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
            (q, k, true)
        }
    };

    // Phase B11b taps: `gdn_qk_norm_q` / `gdn_qk_norm_k`. Both are post-L2
    // normalization (+ Q scaled by 1/sqrt(dk)). Shapes [B, T, nv, dk] (the
    // GQA head-expand above brought nk→nv). HF mirror: `query` / `key` after
    // `query.normalize(dim=-1)` / `key.normalize(dim=-1)` and the Q-scale.
    if capture_b11_taps && qk_expanded {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_qk_norm_q", &q)?;
        crate::mtp_debug::capture_b11_layer0_tap("gdn_qk_norm_k", &k)?;
    }

    // --- Step 6: Compute gates ---
    //
    // Two paths: a fused backend kernel (`backend.gdn_gates`) that collapses
    // the sigmoid + softplus + exp + mul chain into one launch, and the
    // candle-op reference path for everything outside the kernel's
    // envelope (unsupported backend, non-bf16, nv > 256, or kill switches
    // like `KILN_DISABLE_FUSED_GDN_GATES=1` /
    // `KILN_DISABLE_METAL_GDN_GATES=1`). The two are algorithmically
    // identical — the reference path is the original Phase-6 implementation
    // and remains the parity oracle.
    let (beta, g) = {
        kiln_nvtx::range!(c"kiln/gdn/gates");
        if backend.supports_gdn_gates() {
            if let Some((beta, g)) = backend.gdn_gates(&a, &b, &weights.a_log, &weights.dt_bias)? {
                (beta, g)
            } else {
                gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)?
            }
        } else {
            gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)?
        }
    };

    // Phase B11b taps: `gdn_gate_beta` = sigmoid(b), `gdn_gate_g` =
    // -exp(A_log) * softplus(a + dt_bias) (the log-decay scalar fed into the
    // recurrence). Shapes [B, T, nv]. HF mirror: `beta = b.sigmoid()` and
    // `g = -A_log.exp() * F.softplus(a + dt_bias)`.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_gate_beta", &beta)?;
        crate::mtp_debug::capture_b11_layer0_tap("gdn_gate_g", &g)?;
    }

    // --- Step 7: Chunkwise analytical recurrence (Phase 6, approach (b)) ---
    // The recurrent state is stored in F32 externally (across layers/steps)
    // for accumulator stability, but we run the recurrence in bf16 to reclaim
    // the ~66% of prefill GPU time previously spent in bmul_f32 /
    // fast_sum_f32 / badd_f32 (see PROFILING.md recommendation #2). State is
    // cast to bf16 at entry and restored to F32 at exit so the external
    // invariant holds.
    //
    // PR #72 introduced the bf16 hot path. PR #74 replaced the read/write
    // broadcast_mul+sum pairs with batched matmuls but left the O(T)
    // sequential chain. This PR (Phase 6) unrolls the per-chunk recurrence
    // analytically: within each C = GDN_CHUNK_SIZE chunk we build a
    // triangular decay matrix and solve for the per-token updates in a small
    // number of heavy matmuls, cutting the number of GPU kernel launches
    // from O(T) to O(T / C) per layer.
    //
    // The within-chunk forward substitution still walks token-by-token, but
    // each step only does a [1, t] @ [t, dv] matmul over the already-built
    // prefix — orders of magnitude cheaper than the full [dk, dv] state
    // update that was previously done per token.
    let state_external_dtype = recurrent_state.dtype();
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(input_dtype)?;
    }

    let native_recurrent_prefill = if recurrent_prefill_unexpanded_qk {
        let v_recur = v.to_dtype(input_dtype)?;
        gdn_recurrent_prefill_native_head_last(
            backend,
            &q,
            &k,
            &v_recur,
            &beta,
            &g,
            recurrent_state,
        )?
    } else {
        None
    };

    // Cast v back to input_dtype so the recurrence stays in bf16. The
    // portable F32 causal-conv fallback can still produce F32 mixed_qkv;
    // without this cast the subtract `(v - exp(G) * (K @ S_entry))` below
    // hits a dtype mismatch on bf16 GPU runs, because the state-derived
    // tensor inherits the (now bf16) state dtype.
    let (attn_out, attn_out_head_last) = if let Some(attn_out) = native_recurrent_prefill {
        (attn_out, true) // [B, T, nv, dv], contiguous
    } else {
        let (q, k, v, beta, g) = {
            kiln_nvtx::range!(c"kiln/gdn/recur_prep");
            let v = v.to_dtype(input_dtype)?;

            // Transpose to [B, nv, T, dim] for per-head processing.
            let q = q.transpose(1, 2)?; // [B, nv, T, dk]
            let k = k.transpose(1, 2)?; // [B, nv, T, dk]
            let v = v.transpose(1, 2)?; // [B, nv, T, dv]
            let beta = beta.transpose(1, 2)?; // [B, nv, T]
            let g = g.transpose(1, 2)?; // [B, nv, T]
            (q, k, v, beta, g)
        };

        if let Some(attn_out) =
            gdn_recurrent_prefill_head_last(backend, &q, &k, &v, &beta, &g, recurrent_state)?
        {
            (attn_out, true) // [B, T, nv, dv], contiguous
        } else {
            match gdn_chunkwise_recurrence_head_last_full_chunks(
                backend,
                &q,
                &k,
                &v,
                &beta,
                &g,
                recurrent_state,
                GDN_CHUNK_SIZE,
            )? {
                Some(attn_out) => (attn_out, true), // [B, T, nv, dv], contiguous
                None => (
                    gdn_chunkwise_recurrence(
                        backend,
                        &q,
                        &k,
                        &v,
                        &beta,
                        &g,
                        recurrent_state,
                        GDN_CHUNK_SIZE,
                    )?,
                    false,
                ), // [B, nv, T, dv]
            }
        }
    };

    // Restore state to its original dtype so the caller's F32 invariant holds
    // across layer calls and across prefill/decode steps.
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(state_external_dtype)?;
    }

    // Transpose to [B, T, nv, dv] unless the Metal full-chunk path already
    // wrote that contiguous layout directly.
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/post_transpose");
        if attn_out_head_last {
            attn_out
        } else {
            attn_out.transpose(1, 2)?
        }
    };

    // Phase B11b tap: `gdn_recur_out`. Captured post-transpose (shape
    // [B, T, nv, dv]) so the layout matches the input HF passes to its
    // GatedRMSNorm — i.e. the recurrence output transposed into the
    // "head-last" layout. Capturing here (rather than pre-transpose) lets
    // the HF reference mirror this tensor via a single
    // `norm.register_forward_pre_hook`, which sees exactly the same shape.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_recur_out", &attn_out)?;
    }

    // --- Step 8: Gated RMSNorm — norm(attn_out) * silu(z) ---
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/gated_norm");
        let attn_out = gated_rms_norm(backend, &attn_out, &z, &weights.norm, config.rms_norm_eps)?;
        // Reshape to [B, T, v_dim] and cast back to input dtype
        attn_out
            .reshape((batch, seq_len, v_dim))?
            .to_dtype(input_dtype)?
    };

    // Phase B11b tap: `gdn_gated_norm`. Output of the GatedRMSNorm /
    // `norm(attn_out) * silu(z)` block, reshaped and cast back to input
    // dtype. Shape [B, T, v_dim]. HF mirror: `core_attn_out` after
    // `self.norm(core_attn_out, z)`.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_gated_norm", &attn_out)?;
    }

    // --- Step 9: Output projection ---
    // NOTE: conv1d bias is not loaded by the weight loader. If the model has one,
    // it should be added to GpuLinearAttentionWeights and applied after conv1d.
    // Pre-transposed cache (see Step 1 note).
    let out = {
        kiln_nvtx::range!(c"kiln/gdn/out_proj");
        attn_out.broadcast_matmul(&weights.out_proj_t)?
    };

    // Phase B11b tap: `gdn_out_proj`. Output of the final `out_proj` linear
    // (shape [B, T, hidden]) — this is what the caller adds to the residual
    // stream. HF mirror: `self.out_proj(core_attn_out)`.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_out_proj", &out)?;
    }

    Ok(out)
}

/// Grouped-Query Attention (GQA).
///
/// Computes scaled dot-product attention with fewer KV heads than Q heads.
/// Each group of `num_heads / num_kv_heads` query heads shares one KV head.
///
/// `x`: [batch, seq_len, hidden_size]
/// `attn_weights`: Q/K/V/O projection weights plus per-head RMSNorm weights
/// `positions`: position indices for RoPE (length = seq_len, absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `inv_freq`: cached RoPE frequency table (built once via [`compute_rotary_inv_freq`])
/// `rms_norm_eps`: epsilon for Q/K head norms
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array (only full-attn layers)
///
/// Dispatch `q_proj` through the Marlin W4A16 path when available, else the
/// existing BF16 `broadcast_matmul(q_proj_t)` path. LoRA deltas are always
/// added after the base matmul so behaviour matches `linear_with_lora_t` in
/// the absence of Marlin weights.
pub fn q_proj_forward(
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(ref packed) = attn_weights.q_proj_marlin {
        let base =
            crate::marlin_proj::matmul_bf16(x, packed).context("q_proj_forward: marlin matmul")?;
        if let Some(proj) = lora {
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("q_proj_forward: lora delta")?;
            return Ok((base + delta).context("q_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    linear_with_lora_t(x, &attn_weights.q_proj_t, lora, lora_scale)
}

/// Returns: [batch, seq_len, hidden_size]
pub fn gqa_attention(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Project to Q, K, V (with optional LoRA delta)
    // When attn_output_gate is true, q_proj outputs [Q, gate] fused:
    //   q_proj: [num_heads * head_dim * 2, hidden_size]
    //   Split into Q [num_heads, head_dim] and gate [num_heads, head_dim]
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        let q_raw = q_proj_forward(
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )?;
        let k = linear_with_lora_t(
            x,
            &attn_weights.k_proj_t,
            lora_layer.and_then(|l| l.k_proj.as_ref()),
            lora_scale,
        )?;
        let v = linear_with_lora_t(
            x,
            &attn_weights.v_proj_t,
            lora_layer.and_then(|l| l.v_proj.as_ref()),
            lora_scale,
        )?;
        (q_raw, k, v)
    };

    // Split Q and gate if output gate is enabled
    let (q, gate) = if attn_output_gate {
        // q_raw: [batch, seq_len, num_heads * head_dim * 2]
        // Reshape to [batch, seq_len, num_heads, head_dim * 2] then split
        let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        // gate needs to be [batch, seq_len, num_heads * head_dim] for later
        let gate = gate
            .contiguous()?
            .reshape(((), seq_len, num_heads * head_dim))?;
        (q.contiguous()?, Some(gate))
    } else {
        let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
        (q, None)
    };

    // Reshape K, V to [batch, seq_len, num_heads, head_dim]
    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // Apply per-head RMSNorm to Q and K (Qwen3.5 uses QK-norm)
    // q_norm/k_norm are [head_dim] — broadcast over [batch, seq_len, num_heads, head_dim]
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;

    // Apply RoPE (positions are absolute, so cached tokens get correct embeddings)
    // Only rotate first rotary_dim dimensions; the rest pass through unchanged.
    let (q, k) = rotary_embedding(&q, &k, positions, head_dim, rotary_dim, inv_freq)?;

    // Fused-attention path for prefill (seq_len > 1, no KV cache).
    // Takes [batch, seq_len, num_heads, head_dim] — the layout we already
    // have. When a KV cache is present we fall through to the naive path,
    // which handles the cache update and Q_len != KV_len masking correctly.
    // Backend declines (returns None) on dtype mismatch so non-BF16 configs
    // (e.g. tests on F32) transparently fall back to naive softmax+matmul.
    if seq_len > 1 && kv_cache.is_none() && backend.supports_flash_attn_prefill() {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            // Apply output gate: attn_output * sigmoid(gate)
            let attn_output = if let Some(ref gate) = gate {
                let sigmoid_gate = cuda_sigmoid(gate)?;
                (attn_output * sigmoid_gate)?
            } else {
                attn_output
            };
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t(
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            return Ok(out);
        }
    }

    // Transpose to [batch, heads, seq_len, head_dim] for naive attention
    let q = q.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len, head_dim]
    let k = k.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]
    let v = v.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]

    // If KV cache is provided, update it and use full cached K/V
    let (k, v, kv_len) = if let Some(cache) = kv_cache {
        let (full_k, full_v) = cache
            .update(full_attn_layer_idx, &k, &v)
            .context("KV cache update failed")?;
        let kv_len = full_k.dim(2)?;
        (full_k, full_v, kv_len)
    } else {
        (k, v, seq_len)
    };

    // GQA head expansion: repeat K/V to match Q head count
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        // Expand [batch, num_kv_heads, kv_len, head_dim] -> [batch, num_heads, kv_len, head_dim]
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    // Q: [batch, num_heads, seq_len, head_dim]
    // K: [batch, num_heads, kv_len, head_dim]
    // scores: [batch, num_heads, seq_len, kv_len]
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?;
    let attn_scores = (attn_scores / scale)?;

    // Apply causal mask (handles Q_len != KV_len for cached decoding)
    let past_len = kv_len - seq_len;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;

    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?; // [batch, num_heads, seq_len, head_dim]

    // Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
    let attn_output =
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape(((), seq_len, num_heads * head_dim))?;

    // Apply output gate: attn_output * sigmoid(gate)
    let attn_output = if let Some(ref gate) = gate {
        let sigmoid_gate = cuda_sigmoid(gate)?;
        (attn_output * sigmoid_gate)?
    } else {
        attn_output
    };

    // Output projection
    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t(
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    Ok(out)
}

/// Try the fused paged-decode flash-attention kernel.
///
/// Returns `Ok(Some(output))` on success and `Ok(None)` when the kernel
/// preconditions cannot be satisfied (forcing the caller to fall back to the
/// materializing slow path).
///
/// ### Preconditions checked here
///   * `block_size` divides `kBlockN = 128`
///   * Within each `kBlockN`-wide chunk of the block table, the underlying
///     physical pages are contiguous in the pool. The FA2 splitkv paged kernel
///     reads only one block-table entry per kBlockN chunk and assumes the next
///     `kBlockN / block_size` pages are physically contiguous (see
///     `flash_fwd_kernel.h` lines 587-596 and 770-779).
///
/// ### Output
/// `[batch, 1, num_heads * head_dim]` after o_proj (matches the slow path).
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_paged_decode(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    total_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gate: Option<&Tensor>,
    attn_weights: &GpuFullAttentionWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
) -> Result<Option<Tensor>> {
    const K_BLOCK_N: usize = 128;

    let block_size = paged_cache.block_size();
    if block_size == 0 || K_BLOCK_N % block_size != 0 {
        return Ok(None);
    }
    let pages_per_chunk = K_BLOCK_N / block_size;

    // q here is [batch, num_heads, 1, head_dim] after the transpose at the
    // call site. Flash-attn wants [batch, 1, num_heads, head_dim].
    let (batch, q_heads, q_len, q_hd) = q.dims4()?;
    if q_len != 1 || q_heads != num_heads || q_hd != head_dim {
        return Ok(None);
    }
    if batch != 1 {
        // Multi-sequence dispatch needs a per-sequence block_table tensor.
        // Defer to the slow path until the scheduler exercises it.
        return Ok(None);
    }

    let (k_pool, v_pool) = match paged_cache.pool_tensors(full_attn_layer_idx) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Common macOS/desktop case: a single sequence receives freshly-allocated
    // blocks, so its whole live KV window is already one contiguous run in the
    // pool. In that case we can bypass the paged gather path entirely and feed
    // the fused prefill kernel a direct `[1, total_seq_len, kv_heads, head_dim]`
    // narrow of the live K/V window.
    if !paged_cache.is_fp8() {
        if let Some(start_slot) =
            contiguous_slot_run_start(block_table, block_size, 0, total_seq_len)
        {
            let k_live = k_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
            let v_live = v_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
            let attn_output = if backend.supports_flash_attn_prefill_head_major() {
                // Q is already head-major at the call site. Keep K/V grouped
                // instead of routing through `flash_attention_forward`, which
                // expands GQA K/V before Metal SDPA and defeats Candle's
                // native vector-attention GQA path.
                let k_head = k_live.transpose(1, 2)?.contiguous()?;
                let v_head = v_live.transpose(1, 2)?.contiguous()?;
                flash_attention_forward_head_major(
                    backend, q, &k_head, &v_head, num_heads, head_dim,
                )?
            } else {
                None
            };
            let attn_output = if attn_output.is_some() {
                attn_output
            } else {
                // Reshape Q for the fused-attention APIs only when the
                // head-major path declined. The common Metal desktop path
                // returns above and should not pay this transpose/copy.
                let q_fa = {
                    kiln_nvtx::range!(c"kiln/attn/q_fa_transpose");
                    q.transpose(1, 2)?.contiguous()?
                };
                flash_attention_forward(
                    backend,
                    &q_fa,
                    &k_live,
                    &v_live,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?
            };
            if let Some(attn_output) = attn_output {
                // The flash-attention helpers already reshape to
                // [batch, seq_len, num_heads * head_dim].
                let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

                let attn_output = if let Some(gate) = gate {
                    let sigmoid_gate = cuda_sigmoid(gate)?;
                    (attn_output * sigmoid_gate)?
                } else {
                    attn_output
                };
                let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);

                let out = {
                    kiln_nvtx::range!(c"kiln/proj/o");
                    linear_with_lora_t(
                        &attn_output,
                        &attn_weights.o_proj_t,
                        lora_layer.and_then(|l| l.o_proj.as_ref()),
                        lora_scale,
                    )?
                };
                let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
                return Ok(Some(out));
            }
        }
    }

    // Verify intra-chunk contiguity. The kernel reads block_table[c * 8] only
    // (for block_size=16) and assumes pages [c*8 .. c*8+7] are physically
    // contiguous in the pool. kiln's `BlockManager` allocates blocks
    // sequentially from a free list, so a single freshly-allocated sequence
    // satisfies this trivially. After eviction or interleaved allocation the
    // condition may not hold, in which case we fall back.
    let n_chunks = total_seq_len.div_ceil(K_BLOCK_N);
    let blocks = &block_table.blocks;
    let allocated = blocks.len();
    if allocated < n_chunks * pages_per_chunk && allocated < total_seq_len.div_ceil(block_size) {
        // Block table too short for the requested seqlen.
        return Ok(None);
    }
    for c in 0..n_chunks {
        let base_idx = c * pages_per_chunk;
        if base_idx >= allocated {
            break;
        }
        let base_phys = blocks[base_idx];
        for i in 1..pages_per_chunk {
            let idx = base_idx + i;
            if idx >= allocated {
                break;
            }
            if blocks[idx] != base_phys + i as u32 {
                return Ok(None);
            }
        }
    }

    // Build a padded block_table tensor sized [1, n_chunks * pages_per_chunk].
    // Only the entries at indices c * pages_per_chunk are read by the kernel,
    // but we copy the active prefix of the kiln block table and pad the tail
    // by continuing the contiguous run from the last valid block (so any
    // stray reads stay within the cache pool).
    //
    // The scheduler may over-allocate blocks (blocks.len() > max_blocks_per_seq)
    // when it reserves capacity ahead of the current decode position. Those
    // extra blocks are not part of this iteration's active attention window,
    // so we truncate to max_blocks_per_seq before copying. Without this,
    // `reshape((1, max_blocks_per_seq))` crashes when allocated > max
    // (observed: 40 blocks vs max 32 at block 3 of full-attention layers).
    let max_blocks_per_seq = n_chunks * pages_per_chunk;
    let take = max_blocks_per_seq.min(blocks.len());
    let mut padded: Vec<u32> = Vec::with_capacity(max_blocks_per_seq);
    padded.extend_from_slice(&blocks[..take]);
    if padded.is_empty() {
        return Ok(None);
    }
    while padded.len() < max_blocks_per_seq {
        let next = padded.last().copied().unwrap_or(0).wrapping_add(1);
        padded.push(next);
    }

    let device = q.device();
    let bt_tensor =
        Tensor::new(padded.as_slice(), device)?.reshape((1usize, max_blocks_per_seq))?;

    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Reshape Q for the fused paged-decode APIs: [batch, num_heads, 1, head_dim]
    // -> [batch, 1, num_heads, head_dim]. Build it lazily so the contiguous-KV
    // Metal path above can avoid a dead transpose/copy per full-attention layer.
    let q_fa = {
        kiln_nvtx::range!(c"kiln/attn/q_fa_transpose");
        q.transpose(1, 2)?.contiguous()?
    };

    let attn_out = match backend.flash_attn_paged_decode(
        &q_fa,
        k_pool,
        v_pool,
        &bt_tensor,
        total_seq_len,
        block_size,
        softmax_scale,
        true,
    )? {
        Some(t) => t,
        None => return Ok(None),
    };

    // attn_out is [batch, 1, num_heads, head_dim] bf16. Reshape to
    // [batch, 1, num_heads * head_dim] for the gate / o_proj path.
    let _ = num_kv_heads; // unused — kept in signature for symmetry / future use
    let attn_output = attn_out.reshape((batch, 1usize, num_heads * head_dim))?;
    let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

    let attn_output = if let Some(gate) = gate {
        let sigmoid_gate = cuda_sigmoid(gate)?;
        (attn_output * sigmoid_gate)?
    } else {
        attn_output
    };
    let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);

    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t(
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
    Ok(Some(out))
}

/// Grouped-query attention using a paged KV cache.
///
/// Same computation as [`gqa_attention`] but reads/writes K/V through a
/// [`PagedKvCache`] and [`BlockTable`] instead of a contiguous [`KvCache`].
/// This enables multiple concurrent sequences to share a fixed KV cache pool.
///
/// The caller must ensure the block table has enough blocks allocated for all
/// positions up to `positions.last() + 1`.
pub fn gqa_attention_paged(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    gqa_attention_paged_with_rope_tables(
        backend,
        x,
        attn_weights,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        None,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        attn_output_gate,
        lora,
    )
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_paged_with_rope_tables(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rope_tables: Option<(&Tensor, &Tensor)>,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Project to Q, K, V (with optional LoRA delta and output gate split)
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        let q_raw = q_proj_forward(
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )?;
        let k = linear_with_lora_t(
            x,
            &attn_weights.k_proj_t,
            lora_layer.and_then(|l| l.k_proj.as_ref()),
            lora_scale,
        )?;
        let v = linear_with_lora_t(
            x,
            &attn_weights.v_proj_t,
            lora_layer.and_then(|l| l.v_proj.as_ref()),
            lora_scale,
        )?;
        (q_raw, k, v)
    };
    // Phase B7b sub-op taps: post-projection (pre-split). `q_raw` may include
    // the gate half when `attn_output_gate` is on, so its trailing dim is 2H.
    let _ = crate::mtp_debug::capture_subop("post_q_proj_raw", &q_raw);
    let _ = crate::mtp_debug::capture_subop("post_k_proj", &k);
    let _ = crate::mtp_debug::capture_subop("post_v_proj", &v);
    // Phase B9 H3 alias: pre_gated_attn_split is the q_raw tensor before the
    // (q, gate) narrow split. Captured as alias of post_q_proj_raw so the
    // comparator can locate H3 zone divergence by name.
    let _ = crate::mtp_debug::capture_subop("pre_gated_attn_split", &q_raw);
    // Phase B12 layer-31 GQA taps: q_proj / k_proj / v_proj. These are
    // the post-projection tensors before the gate split. No-op unless
    // layer 31 is executing with B12 capture armed.
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("q_proj", &q_raw)?;
        crate::mtp_debug::capture_b12_gqa_tap("k_proj", &k)?;
        crate::mtp_debug::capture_b12_gqa_tap("v_proj", &v)?;
    }

    let (q, gate) = {
        kiln_nvtx::range!(c"kiln/proj/qkv_split");
        if attn_output_gate {
            let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
            let q = q_raw.narrow(3, 0, head_dim)?;
            let gate = q_raw.narrow(3, head_dim, head_dim)?;
            let gate = gate
                .contiguous()?
                .reshape(((), seq_len, num_heads * head_dim))?;
            (q.contiguous()?, Some(gate))
        } else {
            let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
            (q, None)
        }
    };
    // After the gate split, q is the rotation target.
    let _ = crate::mtp_debug::capture_subop("post_q_split", &q);
    // Phase B9 H3 alias: post_gated_attn_split_value mirrors post_q_split.
    let _ = crate::mtp_debug::capture_subop("post_gated_attn_split_value", &q);
    if let Some(ref g) = gate {
        let _ = crate::mtp_debug::capture_subop("post_gate_split", g);
        // Phase B9 H3 alias: post_gated_attn_split_gate mirrors post_gate_split.
        let _ = crate::mtp_debug::capture_subop("post_gated_attn_split_gate", g);
    }

    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // Phase B9 H2 taps: pre_qk_norm_{q,k} are the per-head reshaped tensors
    // immediately before per-head RMSNorm. pre_qk_norm_q is alias of
    // post_q_split; pre_qk_norm_k is genuinely new (post_k_proj is pre-reshape).
    let _ = crate::mtp_debug::capture_subop("pre_qk_norm_q", &q);
    let _ = crate::mtp_debug::capture_subop("pre_qk_norm_k", &k);

    // QK-norm
    let (q, k) = {
        kiln_nvtx::range!(c"kiln/attn/qk_norm");
        let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
        let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
        (q, k)
    };
    let _ = crate::mtp_debug::capture_subop("post_q_norm", &q);
    let _ = crate::mtp_debug::capture_subop("post_k_norm", &k);
    // Phase B9 H2 aliases: post_qk_norm_{q,k} mirror post_{q,k}_norm.
    let _ = crate::mtp_debug::capture_subop("post_qk_norm_q", &q);
    let _ = crate::mtp_debug::capture_subop("post_qk_norm_k", &k);
    // Phase B12 layer-31 GQA taps: qk_norm_q / qk_norm_k. Post per-head
    // RMSNorm, pre-RoPE. Shape [B, T, num_heads, head_dim] /
    // [B, T, num_kv_heads, head_dim].
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("qk_norm_q", &q)?;
        crate::mtp_debug::capture_b12_gqa_tap("qk_norm_k", &k)?;
    }

    // RoPE — only rotate first rotary_dim dimensions
    // Use the GPU tensor variant so positions remain at a stable GPU address
    // (critical for CUDA graph replay correctness)
    let (q, k) = {
        kiln_nvtx::range!(c"kiln/attn/rope");
        if let Some((cos, sin)) = rope_tables {
            rotary_embedding_from_tables(&q, &k, cos, sin, head_dim, rotary_dim)?
        } else {
            rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, inv_freq)?
        }
    };
    let _ = crate::mtp_debug::capture_subop("post_q_rope", &q);
    let _ = crate::mtp_debug::capture_subop("post_k_rope", &k);
    // Phase B12 layer-31 GQA taps: rope_q / rope_k. Post-RoPE, pre-transpose.
    // These are intermediates that HF can only expose via a forward hook on
    // the attention module's q_proj/k_proj output + manual re-run of the
    // rotary function in the comparator — the Python dump script emits a
    // NOTE rather than failing when these HF taps are absent.
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("rope_q", &q)?;
        crate::mtp_debug::capture_b12_gqa_tap("rope_k", &k)?;
    }

    // Keep the cache-native token-major K/V views for paged writes. Attention
    // still wants head-major tensors, but the cache pool stores
    // `[slot, kv_head, dim]`, so using these avoids a transpose back during
    // prefill.
    let k_cache_token_major = k.clone();
    let v_cache_token_major = v.clone();

    // Transpose Q to [batch, heads, seq_len, head_dim]. K/V are transposed
    // lazily only on paths that consume the current tile directly; later
    // prefill tiles and speculative verifier windows read full head-major K/V
    // back from the paged cache instead.
    let q = {
        kiln_nvtx::range!(c"kiln/attn/qkv_transpose");
        let q = q.transpose(1, 2)?.contiguous()?;
        q
    };

    let total_seq_len = start_pos + seq_len;

    // Initial prefill fast path: when there is no prefix history yet
    // (`start_pos == 0`), the current K/V tensors already cover the entire
    // attention window. Route prefill through the backend flash-attn path
    // directly and only write K/V into the paged cache once for future decode.
    // This avoids a pointless write-then-read round-trip through
    // `PagedKvCache` on the first prompt tile.
    if seq_len > 1
        && start_pos == 0
        && (backend.supports_flash_attn_prefill_head_major()
            || backend.supports_flash_attn_prefill())
    {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_initial");
        let k_head = k_cache_token_major.transpose(1, 2)?.contiguous()?;
        let v_head = v_cache_token_major.transpose(1, 2)?.contiguous()?;
        let attn_output = if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k_head, &v_head, num_heads, head_dim)?
        {
            Some(attn_output)
        } else if backend.supports_flash_attn_prefill() {
            let q_prefill = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
            let k_prefill = k_cache_token_major.contiguous()?; // [batch, seq_len, num_kv_heads, head_dim]
            let v_prefill = v_cache_token_major.contiguous()?; // [batch, seq_len, num_kv_heads, head_dim]
            flash_attention_forward(
                backend,
                &q_prefill,
                &k_prefill,
                &v_prefill,
                num_heads,
                num_kv_heads,
                head_dim,
            )?
        } else {
            None
        };

        if let Some(attn_output) = attn_output {
            {
                kiln_nvtx::range!(c"kiln/kv/copy");
                if !paged_cache.write_token_major_native(
                    full_attn_layer_idx,
                    block_table,
                    start_pos,
                    &k_cache_token_major,
                    &v_cache_token_major,
                )? {
                    paged_cache
                        .write(
                            full_attn_layer_idx,
                            block_table,
                            start_pos,
                            &k_head,
                            &v_head,
                        )
                        .context("paged KV cache write failed")?;
                }
            }

            // Phase B12 layer-31 GQA tap: attn_out. Captured AFTER the gate
            // multiply (if `attn_output_gate`) and BEFORE o_proj, so it
            // matches the HF reference's `attn_output = ... * sigmoid_gate`
            // tap point. Shape: [B, T, num_heads * head_dim].
            let attn_output = if let Some(ref gate) = gate {
                let sigmoid_gate = cuda_sigmoid(gate)?;
                (attn_output * sigmoid_gate)?
            } else {
                attn_output
            };
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t(
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            // Phase B12 layer-31 GQA tap: o_proj output (post-o_proj).
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // Phase C8: when the MTP forward step has armed single-token
    // self-attention, the MTP layer attends only to the just-computed K/V
    // (kv_len = 1, no history). Skip the paged-cache write/read and the
    // fused paged-decode kernel entirely so the per-step (k, v) above
    // becomes the SDPA input. Cleared back to `false` by the matching
    // `disarm_mtp_single_token_self_attn` in `mtp_forward_step`, so non-MTP
    // attention calls on this thread are unaffected.
    let single_token_self_attn = crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // Write new K/V into paged cache.
    if !single_token_self_attn {
        kiln_nvtx::range!(c"kiln/kv/copy");
        if !paged_cache.write_token_major_native(
            full_attn_layer_idx,
            block_table,
            start_pos,
            &k_cache_token_major,
            &v_cache_token_major,
        )? {
            let k_head = k_cache_token_major.transpose(1, 2)?.contiguous()?;
            let v_head = v_cache_token_major.transpose(1, 2)?.contiguous()?;
            paged_cache
                .write(
                    full_attn_layer_idx,
                    block_table,
                    start_pos,
                    &k_head,
                    &v_head,
                )
                .context("paged KV cache write failed")?;
        }
    }

    // Fast path: fused paged-decode flash-attention kernel.
    // Eliminates the materializing `paged_cache.read()` (an `index_select` /
    // u8→bf16 dequant) on the decode hot path. Limited to:
    //   * Backends that advertise `supports_flash_attn_paged_decode()`
    //     (CUDA + bf16 today)
    //   * Decode steps (seq_len == 1)
    //   * Non-FP8 caches (the kernel reads bf16 pool slots directly)
    //   * Page sizes that divide kBlockN=128 (block_size=16 satisfies this)
    //   * Single sequence with physically contiguous block allocation
    //     (kiln's BlockManager allocates blocks in order from a free list, so
    //     a freshly-allocated single sequence is always contiguous)
    //   * Phase C8: not in single-token self-attn mode (kernel reads the
    //     full cache history, defeating the kv_len = 1 contract).
    if seq_len == 1
        && !single_token_self_attn
        && !paged_cache.is_fp8()
        && (num_heads / num_kv_heads) > 1
        && !fused_paged_decode_disabled()
        && backend.supports_flash_attn_paged_decode()
        && !crate::mtp_debug::is_c7_sdpa_capture_armed()
    {
        // Open the fused-decode range around the call so the kernel work is
        // attributed to it. When the eligibility checks inside reject (return
        // None) the range still closes here and the fallback range below
        // takes over for the rest of the iteration. Eligibility-rejection is
        // cheap so the over-attribution is small.
        let out_opt = {
            kiln_nvtx::range!(c"kiln/attn/full/decode_fused");
            try_flash_attn_paged_decode(
                backend,
                &q,
                paged_cache,
                block_table,
                full_attn_layer_idx,
                total_seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                gate.as_ref(),
                attn_weights,
                lora_layer,
                lora_scale,
            )?
        };
        if let Some(out) = out_opt {
            return Ok(out);
        }
    }

    // Open the fallback-decode range BEFORE the paged_cache.read so the read's
    // gather/dequant ucopy is attributed to it. The range stays open through
    // the GQA decode work below; it harmlessly also covers the prefill FA-2
    // path (which has its own inner range and returns from inside it). The
    // range is bound to the function scope so it always closes on return.
    let _decode_fallback_nvtx = if seq_len == 1 {
        Some(kiln_nvtx::Range::push(c"kiln/attn/full/decode_fallback"))
    } else {
        None
    };

    // Read full K/V from paged cache (all positions 0..start_pos+seq_len).
    // Phase C8: when single_token_self_attn is armed (MTP inner GQA call),
    // attend only to the just-computed (k, v) — kv_len = 1, no cache read.
    // This matches the Qwen3-Next MTP reference contract where the inner
    // block performs single-token self-attention without a growing KV history.
    let (k, v, kv_len) = if single_token_self_attn {
        (
            k_cache_token_major.transpose(1, 2)?.contiguous()?,
            v_cache_token_major.transpose(1, 2)?.contiguous()?,
            1usize,
        )
    } else {
        let fast_read = if seq_len > 1
            && seq_len >= PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS
            && !paged_cache.is_fp8()
            && backend.supports_paged_kv_head_major_read()
            && backend.supports_flash_attn_prefill_head_major()
        {
            contiguous_slot_run_start(block_table, paged_cache.block_size(), 0, total_seq_len)
                .and_then(|start_slot| {
                    paged_cache
                        .pool_tensors(full_attn_layer_idx)
                        .map(|(k_pool, v_pool)| (start_slot, k_pool, v_pool))
                })
                .map(|(start_slot, k_pool, v_pool)| {
                    backend.paged_kv_head_major_read(k_pool, v_pool, start_slot, total_seq_len)
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        let (k, v) = match fast_read {
            Some((k, v)) => (k, v),
            None => paged_cache
                .read(full_attn_layer_idx, block_table, total_seq_len)
                .context("paged KV cache read failed")?,
        };
        (k, v, total_seq_len)
    };

    // Multi-token append / speculative verify with prefix history. `read`
    // already returns head-major K/V; on Metal, keep Q/K/V in that layout and
    // avoid token-major transposes plus GQA K/V expansion.
    if seq_len > 1
        && backend.supports_flash_attn_prefill_head_major()
        && !crate::mtp_debug::is_c7_sdpa_capture_armed()
    {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_head_major");
        if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k, &v, num_heads, head_dim)?
        {
            let attn_output = if let Some(ref gate) = gate {
                let sigmoid_gate = cuda_sigmoid(gate)?;
                (attn_output * sigmoid_gate)?
            } else {
                attn_output
            };
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t(
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // Fused-attention path for prefill with existing prefix history
    // (`start_pos > 0`). Initial prefill is special-cased above so we do not
    // materialize the same K/V we just produced.
    // Paged cache returns [batch, heads, kv_len, head_dim] — transpose to
    // [batch, kv_len, heads, head_dim] for the backend kernel.
    if seq_len > 1 && backend.supports_flash_attn_prefill() {
        kiln_nvtx::range!(c"kiln/attn/full/prefill");
        let q = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            // Apply output gate: attn_output * sigmoid(gate)
            let attn_output = if let Some(ref gate) = gate {
                let sigmoid_gate = cuda_sigmoid(gate)?;
                (attn_output * sigmoid_gate)?
            } else {
                attn_output
            };
            // Phase B12 layer-31 GQA tap (secondary prefill path).
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t(
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            if crate::mtp_debug::current_b12_layer_is_31() {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // GQA head expansion and attention
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;

    // Optimized decode path (seq_len == 1): reshape Q instead of expanding K/V.
    // Q is [batch, num_heads, 1, head_dim] (1 token) while K/V is
    // [batch, num_kv_heads, kv_len, head_dim] (full history). Expanding K/V
    // copies kv_len * head_dim * num_kv_heads data gqa_ratio times.
    // Instead, group Q heads to match KV heads and compute per-group attention.
    if seq_len == 1 && gqa_ratio > 1 {
        let scale = (head_dim as f64).sqrt();

        // Phase C7 SDPA bisect: capture pre-SDPA Q/K/V and causal-mask taps
        // BEFORE the grouping reshape, in the canonical HF shapes:
        //   Q: [batch, num_heads, q_len=1, head_dim]
        //   K: [batch, num_kv_heads, kv_len, head_dim] (unexpanded — HF
        //      reference dumps the same pre-repeat_kv form)
        //   V: same shape as K
        //   causal_mask: scalar 0 placeholder (decode has q_len=1 and attends
        //      to all kv_len positions, so no mask is applied)
        let c7_armed = crate::mtp_debug::is_c7_sdpa_capture_armed();
        if c7_armed {
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_q", &q)?;
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_k", &k)?;
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_v", &v)?;
            let empty_mask = candle_core::Tensor::zeros((), candle_core::DType::F32, q.device())?;
            crate::mtp_debug::capture_c7_sdpa_tap("causal_mask", &empty_mask)?;
        }

        // Reshape Q: [batch, num_heads, 1, head_dim]
        //          -> [batch, num_kv_heads, gqa_ratio, 1, head_dim]
        //          -> [batch * num_kv_heads, gqa_ratio, 1, head_dim]
        // K:         [batch, num_kv_heads, kv_len, head_dim]
        //          -> [batch * num_kv_heads, kv_len, head_dim]
        // V:         same as K
        let q_grouped = q
            .reshape((batch, num_kv_heads, gqa_ratio, 1, head_dim))?
            .reshape((batch * num_kv_heads, gqa_ratio, 1, head_dim))?
            .contiguous()?;
        // Unsqueeze K/V to [batch*num_kv_heads, 1, kv_len, head_dim] so that
        // broadcast_matmul pairs each Q group with its own KV head (dim 0),
        // broadcasting over the gqa_ratio dim (dim 1).  Without the unsqueeze
        // the 3-D K would be padded to [1, batch*num_kv_heads, ...] and the
        // gqa_ratio dim would incorrectly index into different KV heads.
        let k_flat = k
            .reshape((batch * num_kv_heads, kv_len, head_dim))?
            .unsqueeze(1)?
            .contiguous()?;
        let v_flat = v
            .reshape((batch * num_kv_heads, kv_len, head_dim))?
            .unsqueeze(1)?
            .contiguous()?;

        // Attention scores: [batch*num_kv_heads, gqa_ratio, 1, kv_len]
        let attn_scores = q_grouped.broadcast_matmul(&k_flat.transpose(2, 3)?.contiguous()?)?;
        let attn_scores = (attn_scores / scale)?;

        // Phase C7: reshape grouped scores back to canonical
        // [batch, num_heads, 1, kv_len] for diff against HF.
        if c7_armed {
            let scores_canonical = attn_scores
                .reshape((batch, num_kv_heads, gqa_ratio, 1, kv_len))?
                .reshape((batch, num_heads, 1, kv_len))?;
            crate::mtp_debug::capture_c7_sdpa_tap("attn_scores_pre_softmax", &scores_canonical)?;
        }

        // No causal mask needed for decode (q_len=1 attends to everything)
        let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;

        // Phase C7: reshape grouped probs back to canonical
        // [batch, num_heads, 1, kv_len] for diff against HF.
        if c7_armed {
            let probs_canonical = attn_weights_softmax
                .reshape((batch, num_kv_heads, gqa_ratio, 1, kv_len))?
                .reshape((batch, num_heads, 1, kv_len))?;
            crate::mtp_debug::capture_c7_sdpa_tap("attn_probs", &probs_canonical)?;
        }

        // Weighted sum: [batch*num_kv_heads, gqa_ratio, 1, head_dim]
        let attn_output = attn_weights_softmax.broadcast_matmul(&v_flat)?;

        // Reshape back: -> [batch, num_kv_heads * gqa_ratio, 1, head_dim]
        //               == [batch, num_heads, 1, head_dim]
        let attn_output = attn_output
            .reshape((batch, num_heads, 1, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, num_heads * head_dim))?;
        let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

        // Phase C7: final SDPA output tap at the same point as post_attn_raw,
        // shape [batch, q_len=1, num_heads*head_dim] = [1, 1, 4096].
        if c7_armed {
            crate::mtp_debug::capture_c7_sdpa_tap("attn_out", &attn_output)?;
        }

        let attn_output = if let Some(ref gate) = gate {
            let sigmoid_gate = cuda_sigmoid(gate)?;
            (attn_output * sigmoid_gate)?
        } else {
            attn_output
        };
        let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);
        // Phase B12 layer-31 GQA tap (grouped decode path).
        if crate::mtp_debug::current_b12_layer_is_31() {
            crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
        }
        let out = {
            kiln_nvtx::range!(c"kiln/proj/o");
            linear_with_lora_t(
                &attn_output,
                &attn_weights.o_proj_t,
                lora_layer.and_then(|l| l.o_proj.as_ref()),
                lora_scale,
            )?
        };
        let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
        if crate::mtp_debug::current_b12_layer_is_31() {
            crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
        }
        return Ok(out);
    }

    // Standard path (prefill without flash-attn, or gqa_ratio == 1)
    let (k, v) = if gqa_ratio > 1 {
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?;
    let attn_scores = (attn_scores / scale)?;

    let past_len = kv_len - seq_len;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;

    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?;

    // Transpose back and output projection
    let attn_output =
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape(((), seq_len, num_heads * head_dim))?;
    let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

    let attn_output = if let Some(ref gate) = gate {
        let sigmoid_gate = cuda_sigmoid(gate)?;
        (attn_output * sigmoid_gate)?
    } else {
        attn_output
    };
    let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);
    // Phase B12 layer-31 GQA tap (standard fallback path).
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
    }

    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t(
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
    }
    Ok(out)
}

/// Apply a causal (lower-triangular) mask to attention scores.
/// Sets future positions to -inf so softmax zeroes them out.
fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
    apply_causal_mask_with_offset(scores, seq_len, seq_len, 0)
}

/// Apply a causal mask with support for KV cache offset.
///
/// When using a KV cache, Q has `q_len` new positions and K/V has `kv_len` total
/// positions (past_len cached + q_len new). Each query position `i` (representing
/// absolute position `past_len + i`) can attend to all KV positions up to and
/// including itself: positions `0..past_len + i + 1`.
///
/// `scores`: [batch, heads, q_len, kv_len]
/// `q_len`: number of new query positions
/// `kv_len`: total KV length (past_len + q_len)
/// `past_len`: number of cached positions before the new tokens
fn apply_causal_mask_with_offset(
    scores: &Tensor,
    q_len: usize,
    kv_len: usize,
    past_len: usize,
) -> Result<Tensor> {
    if q_len <= 1 && kv_len <= 1 {
        return Ok(scores.clone());
    }
    // During decode (q_len=1), the single new token can attend to all kv_len
    // positions (all past + itself), so no masking needed.
    if q_len == 1 {
        return Ok(scores.clone());
    }
    let device = scores.device();
    // Build a [q_len, kv_len] mask: 0 for allowed, -inf for masked
    // Query position i (absolute: past_len + i) can attend to KV positions 0..past_len+i+1
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let max_kv = past_len + i + 1; // last allowed KV position (exclusive)
            (0..kv_len).map(move |j| if j < max_kv { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    let mask = Tensor::new(mask, device)?.reshape((1, 1, q_len, kv_len))?;
    let mask = mask.to_dtype(scores.dtype())?;
    let out = scores.broadcast_add(&mask)?;
    Ok(out)
}

/// Single transformer block: norm -> attention -> residual -> norm -> FFN -> residual.
///
/// `x`: [batch, seq_len, hidden_size]
/// `layer`: weights for this transformer layer
/// `positions`: position indices for RoPE (absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rotary_dim`: number of head dims to rotate (partial RoPE)
/// `inv_freq`: cached RoPE frequency table (built once via [`compute_rotary_inv_freq`])
/// `rms_norm_eps`: epsilon for RMSNorm
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array
///
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!("transformer_block only supports full attention layers (not linear/GDN)")
        }
    };

    // Pre-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };

    // Self-attention
    let attn_out = gqa_attention(
        backend,
        &normed,
        attn_weights,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
    )?;

    // Residual connection
    let x = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + attn_out)?
    };

    // Post-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
    };

    // Feed-forward network
    let ffn_out = swiglu_ffn(&normed, &layer.mlp, lora)?;

    // Residual connection
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + ffn_out)?
    };
    Ok(out)
}

/// Transformer block using paged KV cache.
///
/// Same as [`transformer_block`] but reads/writes K/V through a [`PagedKvCache`]
/// and [`BlockTable`] instead of a contiguous [`KvCache`].
pub fn transformer_block_paged(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    transformer_block_paged_with_rope_tables(
        backend,
        x,
        layer,
        config,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        None,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        lora,
    )
}

#[allow(clippy::too_many_arguments)]
fn transformer_block_paged_with_rope_tables(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rope_tables: Option<(&Tensor, &Tensor)>,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!(
                "transformer_block_paged only supports full attention layers (not linear/GDN)"
            )
        }
    };

    // Pre-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };
    let _ = crate::mtp_debug::capture_subop("post_pre_attn_norm", &normed);
    // Phase B12: layer-31 GQA sub-op tap #1. Named `post_input_norm` to
    // match the HF reference-dump naming. No-op unless we are on base-model
    // layer 31 with the B12 capture window armed.
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("post_input_norm", &normed)?;
    }

    // Self-attention with paged cache
    let attn_out = gqa_attention_paged_with_rope_tables(
        backend,
        &normed,
        attn_weights,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rope_tables,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
    )?;
    let _ = crate::mtp_debug::capture_subop("post_attn_block", &attn_out);

    // Residual connection
    let x = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + attn_out)?
    };
    let _ = crate::mtp_debug::capture_subop("post_attn_residual", &x);

    // Post-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
    };
    let _ = crate::mtp_debug::capture_subop("post_pre_mlp_norm", &normed);
    // Phase B12: layer-31 GQA sub-op tap — post_attn_norm. Named to match
    // the HF reference. No-op unless layer 31 + armed.
    if crate::mtp_debug::current_b12_layer_is_31() {
        crate::mtp_debug::capture_b12_gqa_tap("post_attn_norm", &normed)?;
    }

    // Feed-forward network. For layer 31 with B12 armed, route through a
    // sub-op-tapping path that exposes mlp_gate / mlp_up / mlp_down; the
    // standard `swiglu_ffn` fuses those and is fine for everyone else.
    let ffn_out = if crate::mtp_debug::current_b12_layer_is_31() {
        swiglu_ffn_b12_tapped(&normed, &layer.mlp, lora)?
    } else {
        swiglu_ffn(&normed, &layer.mlp, lora)?
    };
    let _ = crate::mtp_debug::capture_subop("post_mlp", &ffn_out);

    // Residual connection
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + ffn_out)?
    };
    // Note: the final block output (`out`) is dumped as `post_layer` at the
    // outer MTP call site, so we do not re-capture it here.
    Ok(out)
}

/// Full model forward pass: embedding → N transformer blocks → final norm → LM head → logits.
///
/// `token_ids`: 1-D slice of token IDs for the input sequence.
/// `weights`: pre-loaded GPU tensors for all model parameters.
/// `config`: model architecture configuration.
/// `kv_cache`: optional KV cache for incremental decoding. When provided, `token_ids`
///   should contain only the new (not yet cached) tokens, and positions are computed
///   starting from `kv_cache.seq_len()`.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
///
/// Notes:
/// - Qwen3.5-4B uses weight tying: the LM head reuses `embed_tokens` transposed.
/// - Linear attention (Gated DeltaNet) layers are not yet implemented and will
///   be skipped with an identity pass-through.
/// - After this function returns, the caller must call `kv_cache.advance(token_ids.len())`
///   to update the cached sequence length.
pub fn model_forward(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mut kv_cache: Option<&mut KvCache>,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let seq_len = token_ids.len();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position indices for RoPE — absolute positions accounting for cached tokens
    let offset = kv_cache.as_ref().map_or(0, |c| c.seq_len());
    let positions: Vec<u32> = (offset..offset + seq_len).map(|p| p as u32).collect();

    // 2. Loop through all transformer layers
    // Track full-attention layer index (0-based counter of only full-attn layers)
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Reborrow the cache for each layer call
                let cache_ref = kv_cache.as_mut().map(|c| &mut **c);
                hidden = transformer_block(
                    backend,
                    &hidden,
                    layer,
                    config,
                    &positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    config.rms_norm_eps,
                    cache_ref,
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                // Pre-attention RMSNorm
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                // Gated DeltaNet linear attention
                let attn_out = gated_deltanet_forward(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    /* capture_b11_taps = */ false,
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention)"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                // Post-attention RMSNorm + FFN
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    // 3. Final RMSNorm + 4. LM head projection (weight-tied: embed_tokens^T)
    // hidden: [1, seq_len, hidden_size], embed_tokens: [vocab_size, hidden_size]
    // logits = hidden @ embed_tokens^T -> [1, seq_len, vocab_size]
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head");
        hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        lm_head_forward(&hidden, &weights.embed_tokens_t)?
    };

    Ok(logits)
}

/// Run a subset of transformer layers on an existing hidden state.
///
/// Processes layers `[start_layer..end_layer)` without embedding or LM head.
/// Used by gradient checkpointing to recompute individual segments.
///
/// `hidden`: [1, seq_len, hidden_size] — input hidden state.
/// `positions`: absolute position indices for RoPE.
/// `linear_state`: mutable linear attention state (only entries for layers in range are touched).
///
/// Returns: [1, seq_len, hidden_size] — output hidden state.
pub fn model_forward_segment(
    backend: &dyn BackendRuntime,
    mut hidden: Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    start_layer: usize,
    end_layer: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    // Count full-attention and linear-attention layers before start_layer
    // so we index into the right KV cache / linear state slots.
    let mut full_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Full(_)))
        .count();
    let mut linear_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Linear(_)))
        .count();

    for i in start_layer..end_layer {
        let layer = &weights.layers[i];
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Training doesn't use KV cache
                hidden = transformer_block(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    config.rms_norm_eps,
                    None, // no KV cache for training
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("segment transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                let attn_out = gated_deltanet_forward(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    /* capture_b11_taps = */ false,
                )
                .with_context(|| format!("segment gated deltanet layer {i}"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Compute embedding lookup and add batch dimension.
///
/// Returns `([1, seq_len, hidden_size], positions)` — the initial hidden state
/// and position indices for RoPE (starting from position 0, no KV cache offset).
pub fn model_forward_embed(token_ids: &[u32], weights: &GpuWeights) -> Result<(Tensor, Vec<u32>)> {
    let seq_len = token_ids.len();
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;
    hidden = hidden.unsqueeze(0)?;
    let positions: Vec<u32> = (0..seq_len).map(|p| p as u32).collect();
    Ok((hidden, positions))
}

/// Apply final RMSNorm and LM head projection.
///
/// `hidden`: [1, seq_len, hidden_size]
/// Returns: [1, seq_len, vocab_size] logits.
pub fn model_forward_head(
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/lm_head");
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    let logits = lm_head_forward(&normed, &weights.embed_tokens_t)?;
    Ok(logits)
}

/// Apply only the final RMSNorm (no LM head projection).
///
/// Used by the FLCE training path to produce the post-final-RMSNorm hidden
/// state that `fused_linear_cross_entropy` consumes. Mirrors the RMSNorm
/// step inside [`model_forward_head`] without the vocab-dim matmul.
pub fn model_forward_final_norm(
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/final_rmsnorm");
    rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)
}

/// Full training-path forward WITHOUT the LM head projection.
///
/// Runs embedding -> transformer layers -> final RMSNorm, returning the
/// post-final-RMSNorm hidden state `[1, seq_len, hidden_size]`. This is the
/// input the Fused Linear Cross-Entropy path consumes, avoiding the
/// `[1, seq_len, vocab_size]` logits materialization that dominates peak
/// VRAM at long context on the Qwen3.5-4B head (V=151936).
///
/// Call site is the trainer (SFT and GRPO) behind the `KILN_USE_FLCE`
/// environment flag. No KV cache is used (matches `standard_forward_backward`).
pub fn model_forward_no_head(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let (hidden, positions) = model_forward_embed(token_ids, weights)?;
    let num_layers = weights.layers.len();
    let hidden = model_forward_segment(
        backend,
        hidden,
        weights,
        config,
        &positions,
        0,
        num_layers,
        linear_state,
        lora,
    )?;
    let normed = {
        kiln_nvtx::range!(c"kiln/final_rmsnorm");
        rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
    };
    Ok(normed)
}

/// Full model forward pass using paged KV cache.
///
/// Same as [`model_forward`] but uses a [`PagedKvCache`] and [`BlockTable`]
/// for KV storage. The caller provides `start_pos` (the absolute position of
/// the first token in `token_ids`) instead of relying on `kv_cache.seq_len()`.
///
/// `positions_gpu`: optional pre-allocated f32 tensor on device with shape [seq_len].
/// When provided, this tensor is used for RoPE instead of creating a new one.
/// This is required for CUDA graph replay: the tensor's GPU address must remain
/// stable so the captured graph reads updated position values on replay.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
pub fn model_forward_paged(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let (logits, _hidden) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
        LmHeadMode::Full,
    )?;
    // `LmHeadMode::Full` always returns Some.
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// Paged-KV forward pass for generation prefill when only the next-token
/// distribution is needed.
///
/// This runs the same layer loop and paged KV writes as [`model_forward_paged`]
/// but only projects the final hidden row through the LM head, returning
/// logits with shape `[1, 1, vocab_size]`.
pub fn model_forward_paged_last_token(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let (logits, _hidden) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
        LmHeadMode::LastRowOnly,
    )?;
    Ok(logits.expect("LmHeadMode::LastRowOnly always produces logits"))
}

/// Paged-KV forward pass that ALSO returns the last-row pre-final-norm hidden state.
///
/// Same semantics as [`model_forward_paged`] (identical layer loop, RoPE,
/// paged KV writes), but extracts the last token's hidden state BEFORE
/// `final_norm` is applied. This is the `h_prev` input the native MTP head
/// consumes for speculative decoding: see [`mtp_forward_step`].
///
/// Returns `(logits[1, seq_len, V], hidden_last[1, 1, H])`. Logits are
/// returned per-position so MTP speculative verification can compare the
/// draft token against position 0 (`logits[:, 0, :]` predicts what should
/// follow the last committed token) and sample a bonus token from position
/// `seq_len - 1` on full acceptance.
pub fn model_forward_paged_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    // Phase B10: arm the base-model per-layer hidden-state capture window
    // when `KILN_MTP_DUMP_HIDDEN_STATES=1`. The arm is a no-op when the env
    // var is unset, so production cost is a single TLS borrow + env lookup.
    // The inner forward pass fills the window with boundary-layer last-row
    // slices plus `h_post_final_norm` (C18 — formerly `h_pre_final_norm`
    // before kiln started returning post-final-norm `h_prev`); the window
    // is drained in
    // `mtp_forward_step`'s dump block so the taps appear alongside the
    // standard 8 MTP taps in the same safetensors file. The next call to
    // this function re-arms the window, overwriting any stale buffer from
    // a prior call whose dump did not fire (e.g. non-targeted `mtp_pos`).
    crate::mtp_debug::arm_h_main_capture();
    // Phase B11: stash the exact input tokens that fed this forward pass so
    // the MTP dump can serialize them, letting the HF reference replay the
    // same prompt instead of its canonical fallback greeting. No-op when
    // h_main capture is disarmed.
    crate::mtp_debug::stash_h_main_replay_context(token_ids);
    // Phase B11b: arm the layer-0 GDN sub-op capture window in the same
    // place as h_main so both capture modes drain together inside the MTP
    // dump block. No-op unless `KILN_MTP_DUMP_B11_TAPS=1`, so production
    // decode pays only a single TLS borrow + env-var lookup.
    crate::mtp_debug::arm_b11_layer0_capture();
    // Phase B12: arm the layer-31 GQA sub-op capture window. Same pattern
    // as B11 — no-op unless `KILN_MTP_DUMP_B12_GQA_TAPS=1`. The h_main
    // capture is gated to also include layers 24..30 when this flag is on,
    // giving the comparator both per-layer h_layer_<idx> taps for the GQA
    // tail and per-sub-op taps inside layer 31.
    crate::mtp_debug::arm_b12_gqa_capture();
    let (logits, hidden) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
        LmHeadMode::FullWithLastHidden,
    )?;
    Ok((
        logits.expect("LmHeadMode::FullWithLastHidden always produces logits"),
        hidden.expect("LmHeadMode::FullWithLastHidden always produces hidden"),
    ))
}

/// Paged-KV forward pass for MTP prefill.
///
/// Returns only the last-row logits plus the last-row pre-final-norm hidden
/// state. MTP prefill does not need per-position logits, so this avoids
/// projecting every prompt row through the large tied LM head.
pub fn model_forward_paged_last_token_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    crate::mtp_debug::arm_h_main_capture();
    crate::mtp_debug::stash_h_main_replay_context(token_ids);
    crate::mtp_debug::arm_b11_layer0_capture();
    let (logits, hidden) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
        LmHeadMode::LastRowWithLastHidden,
    )?;
    Ok((
        logits.expect("LmHeadMode::LastRowWithLastHidden always produces logits"),
        hidden.expect("LmHeadMode::LastRowWithLastHidden always produces hidden"),
    ))
}

/// Single-step native MTP (Multi-Token Prediction) forward pass.
///
/// Implements the Qwen3-Next-style MTP head described in the vLLM reference
/// (`qwen3_next_mtp.py`): given the previously generated token and the base
/// model's pre-final-norm hidden state, project them through the MTP fusion
/// layer and a single full-attention transformer block to produce logits for
/// the NEXT token, plus an updated hidden state that can be fed back for
/// multi-step drafting (when `num_nextn_predict_layers > 1`; Qwen3.5-4B ships
/// `k=1` so drafts are exactly one token deep).
///
/// Fusion pipeline:
///
/// 1. `token_emb  = embed_tokens[draft_token_id]`   # [1, 1, H]
/// 2. `norm_emb   = rms_norm(token_emb, pre_fc_norm_embedding)`
/// 3. `norm_h     = rms_norm(h_prev,    pre_fc_norm_hidden)`
/// 4. `fused      = concat([norm_emb, norm_h], dim=-1) @ fc_t`   # [1,1,2H]→[1,1,H]
/// 5. `hidden     = transformer_block_paged(mtp_layer, fused, mtp_cache, mtp_pos)`
/// 6. `logits     = rms_norm(hidden, final_layernorm) @ embed_tokens_t`  # tied head
///
/// Returns `(logits[1,1,V], new_hidden[1,1,H])`. `new_hidden` is the
/// pre-final-norm output of the MTP transformer block and is the `h_prev`
/// input for the next MTP step (unused when k=1).
///
/// ## KV cache discipline
///
/// The MTP layer maintains its own `PagedKvCache` with exactly ONE full-attn
/// layer slot. `mtp_pos` is the absolute position at which to write this
/// step's KV. Callers advance `mtp_pos` by +1 ONLY when the draft token is
/// accepted; on rejection `mtp_pos` stays unchanged and the next call
/// overwrites the just-written KV slot (the paged writes are idempotent at a
/// given position, so rejection is implicit — no explicit rollback needed).
///
/// ## Marlin / LoRA
///
/// The MTP layer is NOT currently Marlin-packed (deferred to a follow-up PR —
/// Marlin adds substantial pack latency at model load and the MTP layer is a
/// small fraction of per-step cost). LoRA is not applied to MTP.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_step(
    backend: &dyn BackendRuntime,
    draft_token_id: u32,
    h_prev: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mtp_cache: &mut PagedKvCache,
    mtp_block_table: &BlockTable,
    base_pos: usize,
    mtp_pos: usize,
) -> Result<(Tensor, Tensor)> {
    kiln_nvtx::range!(c"kiln/mtp/step");
    let mtp = weights.mtp_weights()?;
    let device = weights.embed_tokens.device();

    // Phase C6 dump pre-flight: consume the dump slot up-front so we can arm
    // the pre-RoPE capture window BEFORE any of the 5 pre-RoPE tensors
    // (token_emb, norm_emb, norm_h, concat, fused) are materialized. The slot
    // is one-shot per (process, mtp_pos), so `should_dump` threads through to
    // the dump block below without being re-consumed. When
    // `KILN_MTP_DUMP_PRE_ROPE` is unset the arm is a no-op and the per-tap
    // `capture_pre_rope_tap` calls short-circuit on a closed TLS window.
    // Phase C13: `KILN_MTP_DUMP_SPLICE=1` is a meta-flag that fires up to N=8
    // (configurable) dumps per targeted position (default `{0, 2}`) instead of
    // the one-shot latch used by earlier phases. When the splice lane takes a
    // slot, `splice_step` is `Some(step)` and the dump path can substitute
    // `{step}` alongside `{pos}`. When splice is disabled we fall through to
    // the existing one-shot latch so prior flows (B6/B7/C6/C7) behave
    // unchanged.
    let splice_step = crate::mtp_debug::try_consume_splice_slot(mtp_pos);
    let should_dump = if splice_step.is_some() {
        true
    } else if crate::mtp_debug::is_dump_splice_enabled() {
        // Splice is on but this position/step is not eligible — suppress the
        // legacy one-shot latch so only the splice lane controls dumping.
        false
    } else {
        crate::mtp_debug::try_consume_dump_slot_for_pos(mtp_pos)
    };
    let dump_pre_rope = should_dump && crate::mtp_debug::is_dump_pre_rope_effectively_enabled();
    if dump_pre_rope {
        crate::mtp_debug::arm_pre_rope_capture();
    }
    // Phase C7 dump pre-flight: arm the SDPA-internal capture BEFORE the
    // inner transformer block runs, so the 7 taps inside `gqa_attention_paged`
    // can record Q/K/V, scores, probs, and the raw attention output. Armed
    // independently of C6 because the two capture windows bracket different
    // regions of the forward: C6 captures MTP fc inputs pre-RoPE, C7
    // captures SDPA inside the inner block post-RoPE. Arming C7 also acts as
    // a signal to the GQA path to bypass the fused flash-attention paged
    // decode kernel (which doesn't materialize the intermediates we need)
    // and take the unfused grouped-decode Candle path instead.
    let dump_c7_sdpa = should_dump && crate::mtp_debug::is_dump_c7_sdpa_enabled();
    if dump_c7_sdpa {
        crate::mtp_debug::arm_c7_sdpa_capture();
    }

    // Phase C14 post-block splice-dump pre-flight. Mirrors the C7 arm above.
    // Gated on `should_dump` AND either the explicit
    // `KILN_MTP_DUMP_C14_POST_BLOCK=1` opt-in or (via OR-composition inside
    // `is_dump_c14_post_block_effectively_enabled`) the C13 splice meta-flag
    // `KILN_MTP_DUMP_SPLICE=1`. When armed, we capture three taps after the
    // MTP transformer block returns: `post_block` (pre-norm hidden),
    // `post_norm` (post-final-norm hidden, pre-lm_head), and `logits`
    // (post-lm_head, pre-softmax). This is the extension of the splice
    // window past the `c6__fused` exit that C13 certified clean.
    let dump_c14_post_block =
        should_dump && crate::mtp_debug::is_dump_c14_post_block_effectively_enabled();
    if dump_c14_post_block {
        crate::mtp_debug::arm_c14_post_block_capture();
    }

    // 1. Token embedding for the draft token. `embedding_lookup` returns
    //    shape [1, H]; unsqueeze to [1, 1, H] to match transformer-block I/O.
    let token_ids = [draft_token_id];
    let token_emb = embedding_lookup_from_weights(&token_ids, weights)?; // [1, H]
    let token_emb = token_emb.unsqueeze(0)?; // [1, 1, H]
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("token_emb", &token_emb);
    }

    // 2-3. Dual RMSNorms. `h_prev` is [1, 1, H] pre-final-norm.
    //
    // `KILN_MTP_SWAP_FC_NORMS=1` swaps which RMSNorm weight is applied to
    // which half. This is the Phase B2 secondary-hypothesis A/B: if the
    // loader paired the two `pre_fc_norm_*` tensors to the wrong halves of
    // the `fc` input (plausible since both are [H]-vectors and
    // distinguishable only by name), swap-on should materially change α.
    // If α is unchanged the hypothesis is disproven.
    let swap_fc_norms = crate::mtp_debug::is_swap_fc_norms_enabled();
    let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
        (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
    } else {
        (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
    };
    let norm_emb = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_emb");
        rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("norm_emb", &norm_emb);
    }
    let norm_h = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
        rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("norm_h", &norm_h);
    }

    // 4. Concat along the hidden dim and fuse: [1, 1, 2H] @ fc_t[2H, H] -> [1, 1, H]
    //
    // We keep the concat alive (named `concat`) so the Phase B6 dump can
    // capture the exact bytes fed into `fc.weight` as `fc_input`.
    let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("concat", &concat);
    }
    // Phase C12: `KILN_MTP_FP32_HEAD=1` subsumes `KILN_MTP_FC_FP32_ACCUM=1` —
    // the full-head kill switch always includes fc_input/fc_output in f32,
    // so either flag alone is sufficient to trigger the fp32 fc path.
    let fp32_head = crate::mtp_debug::is_mtp_fp32_head_enabled();
    let fused = {
        kiln_nvtx::range!(c"kiln/mtp/fc");
        if crate::mtp_debug::is_mtp_fc_fp32_accum_enabled() || fp32_head {
            // Phase C9 falsification: promote inputs to f32, matmul in f32,
            // cast the result back to the input dtype. Eliminates the bf16
            // accumulation noise visible at the `fused` / `fc_output` tap
            // (max|Δ| ~1.6e-2 against the HF bf16 reference). The
            // [1, 1, 2H] @ [2H, H] shape is tiny (~13M FLOPs for 4B), so
            // the per-step cost is negligible and there is no hot-path
            // regression to worry about.
            let in_dtype = concat.dtype();
            let concat_f32 = concat.to_dtype(candle_core::DType::F32)?;
            let fc_t_f32 = mtp.fc_t.to_dtype(candle_core::DType::F32)?;
            concat_f32.broadcast_matmul(&fc_t_f32)?.to_dtype(in_dtype)?
        } else {
            concat.broadcast_matmul(&mtp.fc_t)?
        }
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("fused", &fused);
    }

    // Phase B7 sub-op capture pre-flight. `should_dump` was already consumed
    // above (moved up to bracket Phase C6 pre-RoPE arming). `dump_subops` is
    // a strict subset: only true when KILN_MTP_DUMP_SUBOPS=1 AND we are about
    // to dump anyway. This keeps the production path entirely free of sub-op
    // capture overhead — the TLS check inside `capture_subop` is a no-op when
    // the window is closed.
    let dump_subops = should_dump && crate::mtp_debug::is_dump_subops_enabled();
    if dump_subops {
        crate::mtp_debug::arm_subop_capture();
    }

    // 5. Single full-attention transformer block with its own paged cache.
    //
    //    Two distinct position counters are in play here:
    //
    //    * `base_pos + mtp_pos` — the ABSOLUTE sequence position the draft
    //      token would occupy in the prompt+decode stream. This is what
    //      RoPE must use so the MTP head sees the same rotation angles the
    //      base Qwen3-Next block would have applied at that position. The
    //      PyTorch reference (`scripts/mtp_reference_dump.py`) applies RoPE
    //      at the absolute position; Phase B7a (PR #276) confirmed kiln's
    //      prior use of bare `mtp_pos` here caused monotonic `post_layer`
    //      drift at pos=1,2 — the RoPE-wrong-position signature.
    //
    //    * `mtp_pos` — the LOCAL slot index into the MTP paged KV cache.
    //      The MTP cache is its own isolated address space (distinct from
    //      the base KV cache); slot `mtp_pos` is the right write target
    //      regardless of where the token sits in absolute stream order.
    //
    //    MTP is not CUDA-graph-captured, so rebuilding the position tensor
    //    per step is fine.
    let abs_pos = base_pos + mtp_pos;
    let positions = Tensor::new(&[abs_pos as f32][..], device)?;
    // Phase C8: arm single-token self-attention for the MTP inner GQA call.
    // The Qwen3-Next reference contract (see `scripts/mtp_reference_dump.py`
    // and HF/vLLM `Qwen3NextMultiTokenPredictor`) runs the MTP inner
    // attention as kv_len = 1 — Q·K^T is a 1×1 scalar, softmax = 1.0,
    // attn_out = V. Phase C7 (PR #319) localized the mtp_pos > 0
    // attn_out divergence to kiln attending over the growing MTP paged
    // cache (kv_len = mtp_pos + 1) instead of single-token self-attn.
    // Arming this flag flips `gqa_attention_paged` onto the per-step
    // K/V scratch path for the MTP layer only; the disarm below clears
    // it before any non-MTP attention path on this thread can observe it.
    crate::mtp_debug::arm_mtp_single_token_self_attn();
    // Phase C12: arm fp32-head BEFORE the inner block so that every
    // projection matmul inside `gqa_attention_paged` (q/k/v/o) and the
    // MLP (`swiglu_ffn`'s gate/up/down) sees the TLS flag armed. This is
    // the cleanest minimal invasive cast point: `linear_with_lora_t` is
    // the single chokepoint for all of those matmuls, and the non-MTP
    // paths never observe the armed flag because it is disarmed below
    // before we return. The MTP head is not Marlin-packed today, so in
    // practice the flag gates the straight BF16 `broadcast_matmul`; if a
    // future PR adds Marlin to MTP, the marlin path in `q_proj_forward`
    // will need an analogous upcast branch.
    if fp32_head {
        crate::mtp_debug::arm_mtp_fp32_head();
    }
    let mtp_hidden_result = transformer_block_paged(
        backend,
        &fused,
        &mtp.layer,
        config,
        &positions,
        mtp_pos,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim(),
        &weights.rotary_inv_freq,
        config.rms_norm_eps,
        mtp_cache,
        mtp_block_table,
        /* full_attn_layer_idx = */ 0,
        /* lora = */ None,
    );
    // Always disarm in both success and error paths (`mtp_hidden_result`
    // is `?`-propagated below, so we cannot rely on the function tail).
    crate::mtp_debug::disarm_mtp_single_token_self_attn();
    if fp32_head {
        crate::mtp_debug::disarm_mtp_fp32_head();
    }

    // Drain the sub-op capture window in BOTH success and error paths so the
    // TLS slot is not left armed for the next draft step (which would corrupt
    // the next dump or leak captures into other transformer block calls).
    //
    // Phase B10 appends per-layer base-model hidden-state taps captured during
    // the prior `model_forward_paged_with_last_hidden` call on this thread.
    // These live in a distinct TLS slot (`H_MAIN_CAPTURE`) gated on
    // `KILN_MTP_DUMP_HIDDEN_STATES=1`; `drain_h_main_capture` returns empty
    // when the slot was never armed, so disarmed runs pay zero cost.
    let mut extra_subops = if dump_subops {
        crate::mtp_debug::drain_subop_capture()
    } else {
        Vec::new()
    };
    extra_subops.extend(crate::mtp_debug::drain_h_main_capture());
    let mtp_hidden = mtp_hidden_result.context("mtp transformer block")?;
    // Phase C14 tap 1/3: pre-final-norm output of the MTP transformer block.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("post_block", &mtp_hidden);
    }

    // 6. Final RMSNorm + weight-tied LM head (reuses base embed_tokens_t).
    //
    // We split `normed` out as a distinct bind (rather than inlining into the
    // `logits` block) so the Phase B6 dump can capture `post_final_ln` ahead
    // of the `lm_head` matmul. No semantic change vs the previous inlined
    // form: `rms_norm` has no side effects, and `normed` is only used once.
    let normed = {
        kiln_nvtx::range!(c"kiln/mtp/final_layernorm");
        rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?
    };
    // Phase C14 tap 2/3: post-final-norm hidden state, pre-lm_head.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("post_norm", &normed);
    }
    let logits = {
        kiln_nvtx::range!(c"kiln/mtp/lm_head");
        lm_head_forward(&normed, &weights.embed_tokens_t)?
    };
    // Phase C14 tap 3/3: post-lm_head logits, pre-softmax / pre-sampler.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("logits", &logits);
    }

    // Phase B6/B7 numerical-bisect dump. Fires once per (process, mtp_pos)
    // pair when `KILN_MTP_DUMP_PATH` is set and the current `mtp_pos` is
    // listed in `KILN_MTP_DUMP_POS` (defaults to "0" for B6 compatibility).
    // Writes one safetensors file per targeted position with the 8 outer
    // taps enumerated in `write_mtp_dump` plus integer metadata (draft
    // token id, `mtp_pos`, `swap_fc_norms`). When `KILN_MTP_DUMP_SUBOPS=1`
    // is also set, per-sub-op activations from inside the MTP transformer
    // block are appended (Phase B7b).
    //
    // Use `KILN_MTP_DUMP_PATH=/path/dump_pos{pos}.st` plus
    // `KILN_MTP_DUMP_POS=0,1,2` to capture three positions in one process.
    // The companion Python reference (`scripts/mtp_reference_dump.py`)
    // produces same-shaped files for the same prompt + seed;
    // `scripts/mtp_compare.py` prints a per-tap first-divergence table.
    // Failure to dump is logged but non-fatal — we never want an
    // instrumentation bug to break decode.
    if should_dump {
        // Phase C13: when the splice meta-flag is driving this step, the path
        // can substitute `{step}` alongside `{pos}` so each of the up-to-8
        // per-position dumps lands in its own file. Falls back to the legacy
        // `{pos}`-only substitution when splice is off.
        let dump_path_opt = crate::mtp_debug::dump_path_for_pos_and_step(mtp_pos, splice_step);
        // Always drain the C7 TLS slot before returning. If we entered the
        // armed C7 path above but `dump_path_for_pos` returned None (pos not
        // listed in `KILN_MTP_DUMP_POS`), the slot would otherwise leak
        // captured tensors into the next draft step's dump. Dropping the
        // drained vec here is cheap and keeps the invariant "armed ⇒ drained
        // within the same mtp_forward_step".
        if dump_c7_sdpa && dump_path_opt.is_none() {
            let _ = crate::mtp_debug::drain_c7_sdpa_capture();
        }
        // Phase C14: mirror the C7 defensive drain so the post-block TLS slot
        // is not left armed for the next draft step when the path is filtered
        // out for this pos.
        if dump_c14_post_block && dump_path_opt.is_none() {
            let _ = crate::mtp_debug::drain_c14_post_block_capture();
        }
        if let Some(path) = dump_path_opt {
            let taps: [(&str, &Tensor); 8] = [
                ("h_main", h_prev),
                ("tok_embed", &token_emb),
                ("fc_input", &concat),
                ("fc_output", &fused),
                ("pre_layer", &fused),
                ("post_layer", &mtp_hidden),
                ("post_final_ln", &normed),
                ("mtp_logits", &logits),
            ];
            // Phase B11: drain any prompt tokens stashed by the preceding
            // `model_forward_paged_with_last_hidden` call. Empty on legacy
            // paths / when h_main capture was never armed, which matches
            // the pre-B11 dump format (no `prompt_tokens` tensor emitted).
            let prompt_tokens = crate::mtp_debug::drain_h_main_prompt_tokens();
            let replay_tokens = crate::mtp_debug::drain_h_main_replay_tokens();
            // Phase B11b: drain any layer-0 GDN sub-op taps stashed during
            // the base-model forward. Empty when `KILN_MTP_DUMP_B11_TAPS`
            // is unset, which keeps the dump format bit-identical to B11.
            let b11_taps = crate::mtp_debug::drain_b11_layer0_capture();
            // Phase B12: drain any layer-31 GQA sub-op taps stashed during
            // the base-model forward. Empty when
            // `KILN_MTP_DUMP_B12_GQA_TAPS` is unset, which keeps the dump
            // format bit-identical to the pre-B12 layout.
            let b12_taps = crate::mtp_debug::drain_b12_gqa_capture();
            // Phase C6: drain the 5 pre-RoPE MTP input taps (token_emb,
            // norm_emb, norm_h, concat, fused) captured above. Empty when
            // `KILN_MTP_DUMP_PRE_ROPE` is unset, which keeps the dump format
            // bit-identical to the pre-C6 layout.
            let c6_taps = crate::mtp_debug::drain_pre_rope_capture();
            // Phase C7: drain the 7 SDPA-internal taps (pre_sdpa_q/k/v,
            // causal_mask, attn_scores_pre_softmax, attn_probs, attn_out)
            // captured inside `gqa_attention_paged`. Empty when
            // `KILN_MTP_DUMP_C7_SDPA` is unset, which keeps the dump format
            // bit-identical to the pre-C7 layout.
            let c7_taps = crate::mtp_debug::drain_c7_sdpa_capture();
            // Phase C14: drain the 3 post-MTP-transformer-block taps
            // (post_block, post_norm, logits) captured above. Empty when
            // neither `KILN_MTP_DUMP_C14_POST_BLOCK` nor the C13 splice
            // meta-flag is set, which keeps the dump format bit-identical
            // to the pre-C14 layout.
            let c14_taps = crate::mtp_debug::drain_c14_post_block_capture();
            match crate::mtp_debug::write_mtp_dump(
                &path,
                draft_token_id,
                mtp_pos,
                base_pos,
                swap_fc_norms,
                &taps,
                &extra_subops,
                &prompt_tokens,
                &replay_tokens,
                &b11_taps,
                &b12_taps,
                &c6_taps,
                &c7_taps,
                &c14_taps,
            ) {
                Ok(()) => tracing::info!(
                    target: "kiln::mtp_debug",
                    path = %path,
                    draft_token_id,
                    mtp_pos,
                    splice_step = ?splice_step,
                    subops = extra_subops.len(),
                    prompt_tokens_len = prompt_tokens.len(),
                    replay_tokens_len = replay_tokens.len(),
                    b11_taps = b11_taps.len(),
                    b12_taps = b12_taps.len(),
                    c6_taps = c6_taps.len(),
                    c7_taps = c7_taps.len(),
                    c14_taps = c14_taps.len(),
                    "mtp_b7_dump_written"
                ),
                Err(e) => tracing::warn!(
                    target: "kiln::mtp_debug",
                    error = %e,
                    "mtp_b7_dump_failed"
                ),
            }
        }
    } else if dump_c7_sdpa || dump_c14_post_block {
        // Defensive: drain C7 / C14 captures even when `should_dump` is false
        // to avoid leaving the TLS slots armed for the next draft step. This
        // branch should be unreachable (both flags AND with `should_dump`),
        // but is cheap insurance against future refactors that could break
        // the invariant.
        if dump_c7_sdpa {
            let _ = crate::mtp_debug::drain_c7_sdpa_capture();
        }
        if dump_c14_post_block {
            let _ = crate::mtp_debug::drain_c14_post_block_capture();
        }
    }

    // Optional Phase B instrumentation. Off by default; enabled with
    // `KILN_MTP_DEBUG=1`. See `crate::mtp_debug` for the rate-limited path.
    //
    // Phase B2 additions: halves-L2 on the `fc` input (to quantify the
    // embed-dominance hypothesis) and L2 on the fused output (to rule out
    // explode/collapse failure modes inside the fc matmul). `halves_ratio`
    // is `norm_emb_l2 / norm_h_l2`; values far from 1.0 are evidence the
    // two halves have mismatched magnitudes feeding `fc`.
    if crate::mtp_debug::should_log() {
        let h_norm = crate::mtp_debug::tensor_l2_norm(h_prev).unwrap_or(f32::NAN);
        let norm_emb_l2 = crate::mtp_debug::tensor_l2_norm(&norm_emb).unwrap_or(f32::NAN);
        let norm_h_l2 = crate::mtp_debug::tensor_l2_norm(&norm_h).unwrap_or(f32::NAN);
        let fused_l2 = crate::mtp_debug::tensor_l2_norm(&fused).unwrap_or(f32::NAN);
        let halves_ratio = if norm_h_l2 > 0.0 {
            norm_emb_l2 / norm_h_l2
        } else {
            f32::NAN
        };
        let logits_norm = crate::mtp_debug::tensor_l2_norm(&logits).unwrap_or(f32::NAN);
        let top = crate::mtp_debug::top_k_logits(&logits, 5)
            .map(|t| crate::mtp_debug::format_top_k(&t))
            .unwrap_or_else(|e| format!("<top_k err: {e}>"));
        tracing::info!(
            target: "kiln::mtp_debug",
            mtp_pos = mtp_pos,
            last_token = draft_token_id,
            swap_fc_norms = swap_fc_norms,
            h_prev_l2 = h_norm,
            norm_emb_l2 = norm_emb_l2,
            norm_h_l2 = norm_h_l2,
            halves_ratio = halves_ratio,
            fused_l2 = fused_l2,
            mtp_logits_l2 = logits_norm,
            mtp_top5 = %top,
            "mtp_draft"
        );
    }

    Ok((logits, mtp_hidden))
}

/// Controls the LM head behaviour at the end of a paged forward pass.
///
/// The streaming/tiled prefill path needs to skip the LM head entirely on
/// every non-final tile (its outputs are discarded by the caller) and
/// optionally collapse the final tile's projection to a single row, since
/// only the last token's logits feed sampling. Both shortcuts preserve
/// bit-exact agreement with the monolithic path on the values that are
/// actually consumed downstream.
#[derive(Clone, Copy, Debug)]
enum LmHeadMode {
    /// Compute the LM head over every position. Result has shape
    /// `[1, seq_len, vocab_size]`. This is the legacy `model_forward_paged`
    /// behaviour and the only mode used by training / parity verification.
    Full,
    /// Compute the LM head over the final token only. Result has shape
    /// `[1, 1, vocab_size]`. Numerically identical to slicing the last row
    /// of `Full` because RMSNorm is per-position and the matmul reduces
    /// along `hidden_size` only.
    LastRowOnly,
    /// Compute the LM head over every position AND return the last-row
    /// pre-final-norm hidden state. Used by
    /// [`model_forward_paged_with_last_hidden`] to surface per-position logits
    /// for MTP speculative verification at position 0 (draft comparison) and
    /// position 1 (bonus), plus `h_prev` for the next MTP step.
    FullWithLastHidden,
    /// Compute the LM head over the final token only AND return the last-row
    /// pre-final-norm hidden state. Used by MTP prefill, which only consumes
    /// the next-token distribution for the prompt's final row.
    LastRowWithLastHidden,
    /// Skip RMSNorm + LM head entirely and return `None`. Used for non-final
    /// tiles where the caller throws away the logits.
    Skip,
}

/// Internal per-tile forward pass shared by `model_forward_paged` and
/// `model_forward_paged_streaming`. `lm_head_mode` controls whether the
/// final RMSNorm + LM head projection runs and over how many positions.
///
/// Pure code motion from the original `model_forward_paged` — the layer
/// loop, RoPE position tensor handling, and per-layer dispatch are unchanged.
/// The only difference is the LM head section at the bottom, which becomes
/// a `match` over `lm_head_mode`.
#[allow(clippy::too_many_arguments)]
fn model_forward_paged_inner(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
    lm_head_mode: LmHeadMode,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let seq_len = token_ids.len();
    let device = weights.embed_tokens.device();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Phase B11b tap: `tok_embed`. Output of `embed_tokens(input_ids)` with a
    // leading batch dim. Taken once at layer 0 entry so both kiln and the HF
    // reference dump compare the exact same pre-layer hidden state. Shape
    // [1, T, hidden].
    crate::mtp_debug::capture_b11_layer0_tap("tok_embed", &hidden)?;

    // Position tensor for RoPE — use pre-allocated GPU tensor if provided,
    // otherwise create one from scratch. The pre-allocated path is essential
    // for CUDA graph replay where the tensor pointer must be stable.
    let positions_owned;
    let positions: &Tensor = match positions_gpu {
        Some(t) => t,
        None => {
            let pos_f32: Vec<f32> = (start_pos..start_pos + seq_len).map(|p| p as f32).collect();
            positions_owned = Tensor::new(pos_f32.as_slice(), device)?;
            &positions_owned
        }
    };
    let rope_tables_owned = if positions_gpu.is_none() {
        Some(rotary_tables_from_tensor(
            positions,
            &weights.rotary_inv_freq,
        )?)
    } else {
        None
    };
    let rope_tables = rope_tables_owned
        .as_ref()
        .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor));

    // 2. Loop through all transformer layers
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Phase B12: tell the capture layer that we are entering the
                // base-model layer `i`. `capture_b12_gqa_tap` call sites inside
                // `gqa_attention_paged` / `transformer_block_paged` gate on
                // this TLS slot + the armed capture window so that only
                // layer 31 emits sub-op taps. No-op on the production path.
                crate::mtp_debug::enter_b12_layer_scope(i);
                let block_result = transformer_block_paged_with_rope_tables(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    start_pos,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    rope_tables,
                    config.rms_norm_eps,
                    paged_cache,
                    block_table,
                    full_attn_idx,
                    layer_lora,
                );
                crate::mtp_debug::exit_b12_layer_scope();
                hidden = block_result
                    .with_context(|| format!("transformer block {i} (full attention, paged)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                // Phase B11b tap: `layer_0_post_input_norm`. Captured only on
                // layer 0 (the B10 scan localized divergence there) — pre-GDN
                // input LayerNorm output. Shape [1, T, hidden]. HF mirror:
                // `hidden_states` after `self.input_layernorm(...)` at layer 0.
                if crate::mtp_debug::should_capture_b11_tap_for_layer(i) {
                    crate::mtp_debug::capture_b11_layer0_tap("layer_0_post_input_norm", &normed)?;
                }
                let attn_out = gated_deltanet_forward(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    /* capture_b11_taps = */
                    crate::mtp_debug::should_capture_b11_tap_for_layer(i),
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention, paged)"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }

        // Phase B10: capture last-row hidden state at boundary layers when
        // `KILN_MTP_DUMP_HIDDEN_STATES=1` and a capture window has been armed
        // (done by `model_forward_paged_with_last_hidden`). Gate is a cheap
        // TLS-borrow + array-contains check when disarmed; zero cost in
        // production. The narrow+contiguous copies ~H floats per captured
        // layer (5 layers × 2560 f32 ≈ 50 KiB total) which is negligible
        // next to the full hidden tensor.
        if crate::mtp_debug::should_capture_hidden_state_for_layer(i) {
            let last_row = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
            let _ = crate::mtp_debug::capture_h_main_tap(&format!("h_layer_{i}"), &last_row);
        }
    }

    // 3. Final RMSNorm + 4. LM head projection (weight-tied)
    //
    // `Full` matches the legacy code path exactly. `LastRowOnly` slices the
    // hidden tensor to the last position before the projection so we only
    // do `vocab_size * hidden_size` MACs instead of `seq_len * vocab_size *
    // hidden_size` — bit-exact with `Full`'s last row because RMSNorm is
    // per-position and the matmul reduces along `hidden_size` only. `Skip`
    // returns `None` and is used by the streaming dispatcher for every tile
    // whose logits the caller will throw away.
    match lm_head_mode {
        LmHeadMode::Full => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward(&hidden, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), None))
        }
        LmHeadMode::LastRowOnly => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                let last = hidden.narrow(1, seq_len - 1, 1)?;
                let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward(&normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), None))
        }
        LmHeadMode::FullWithLastHidden => {
            // Phase C18: `h_prev` must be returned POST-final-norm.
            // vLLM (`Qwen3_5MultiTokenPredictor.forward`) and SGLang consume
            // the base model's `last_hidden_state` (post-`model.norm`) as the
            // input to `pre_fc_norm_hidden`. C17 cross-referenced the upstream
            // contract and the C15 numerical fingerprint (2.0–2.4× kiln/HF
            // magnitude ratio) confirmed kiln was one RMSNorm behind. We now
            // apply `final_norm` ONCE and slice the last row from the normed
            // tensor for both the logits projection and the returned h_prev.
            let normed = {
                kiln_nvtx::range!(c"kiln/final_norm");
                rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
            };
            let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
            if crate::mtp_debug::is_h_main_capture_armed() {
                let _ = crate::mtp_debug::capture_h_main_tap("h_post_final_norm", &last_hidden);
            }
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward(&normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), Some(last_hidden)))
        }
        LmHeadMode::LastRowWithLastHidden => {
            // Phase C18: same frame fix as `FullWithLastHidden`. For the
            // single-row variant we still only materialise the last row before
            // `final_norm` (cheap) — that row, once normed, is the canonical
            // post-final-norm `h_prev` the MTP head expects.
            let last_pre_norm = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
            let last_hidden = {
                kiln_nvtx::range!(c"kiln/final_norm");
                rms_norm(&last_pre_norm, &weights.final_norm, config.rms_norm_eps)?
            };
            if crate::mtp_debug::is_h_main_capture_armed() {
                let _ = crate::mtp_debug::capture_h_main_tap("h_post_final_norm", &last_hidden);
            }
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward(&last_hidden, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), Some(last_hidden)))
        }
        LmHeadMode::Skip => Ok((None, None)),
    }
}

/// Streaming/tiled paged prefill — the Phase 7 long-context entry point.
///
/// Iterates `token_ids` in fixed-size tiles (default 8192 tokens, configurable
/// via `KILN_STREAMING_TILE_TOKENS`, must be a multiple of `GDN_CHUNK_SIZE`)
/// and dispatches each tile through `model_forward_paged_inner`. The
/// `LinearAttentionState` carries GDN recurrent + conv state across tile
/// boundaries; the paged KV cache is filled tile-by-tile via `start_pos +
/// cursor`. Only the final tile runs the LM head — non-final tiles use
/// `LmHeadMode::Skip`. When `KILN_STREAMING_LAST_TOKEN_LM_HEAD=0` the final
/// tile uses `LmHeadMode::Full` instead so callers can compare per-position
/// logits against the monolithic path.
///
/// Returns logits with shape `[1, 1, vocab_size]` (last-token only) or
/// `[1, last_tile_len, vocab_size]` when full LM head is requested.
///
/// `positions_gpu` is intentionally not threaded through to per-tile calls —
/// each tile builds its own per-tile position vector inside the inner fn.
/// Streaming prefill is incompatible with CUDA graph replay (which requires
/// a stable shape per call) and is only used outside of graph-captured paths.
pub fn model_forward_paged_streaming(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_streaming_with(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        streaming_tile_tokens_for(weights.embed_tokens.device()),
        streaming_last_token_lm_head(),
    )
}

/// Streaming/tiled MTP prefill.
///
/// Same tiled execution as [`model_forward_paged_streaming`], but the final
/// tile returns both last-token logits and the post-final-norm `h_prev` needed
/// to seed native MTP decoding.
pub fn model_forward_paged_streaming_last_token_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<(Tensor, Tensor)> {
    model_forward_paged_streaming_last_token_with_last_hidden_with(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        streaming_tile_tokens_for(weights.embed_tokens.device()),
    )
}

/// Explicit-tile variant of
/// [`model_forward_paged_streaming_last_token_with_last_hidden`].
pub fn model_forward_paged_streaming_last_token_with_last_hidden_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
) -> Result<(Tensor, Tensor)> {
    let total = token_ids.len();
    if total == 0 {
        anyhow::bail!(
            "model_forward_paged_streaming_last_token_with_last_hidden requires at least one token"
        );
    }
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }

    let mut last_logits: Option<Tensor> = None;
    let mut last_hidden: Option<Tensor> = None;
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let is_last_tile = end == total;
        let mode = if is_last_tile {
            crate::mtp_debug::arm_h_main_capture();
            crate::mtp_debug::stash_h_main_replay_context(token_ids);
            crate::mtp_debug::arm_b11_layer0_capture();
            LmHeadMode::LastRowWithLastHidden
        } else {
            LmHeadMode::Skip
        };

        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();
        let (tile_logits, tile_hidden) = model_forward_paged_inner(
            backend,
            &token_ids[cursor..end],
            weights,
            config,
            paged_cache,
            block_table,
            start_pos + cursor,
            state_for_tile,
            lora,
            None,
            mode,
        )
        .with_context(|| {
            format!(
                "streaming MTP prefill tile [{cursor}, {end}) of {total} (start_pos={})",
                start_pos + cursor
            )
        })?;

        if is_last_tile {
            last_logits = tile_logits;
            last_hidden = tile_hidden;
        }

        cursor = end;
    }

    Ok((
        last_logits.context("streaming MTP prefill produced no logits")?,
        last_hidden.context("streaming MTP prefill produced no h_prev")?,
    ))
}

/// Explicit-parameter variant of [`model_forward_paged_streaming`] used by
/// tests that need to exercise specific tile sizes without manipulating
/// process-wide env vars (which would race under parallel test runners).
///
/// `tile_size` must be a positive multiple of `GDN_CHUNK_SIZE`.
pub fn model_forward_paged_streaming_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
    last_token_only: bool,
) -> Result<Tensor> {
    let total = token_ids.len();
    if total == 0 {
        anyhow::bail!("model_forward_paged_streaming requires at least one token");
    }
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }

    let mut last_logits: Option<Tensor> = None;
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let is_last_tile = end == total;
        let mode = if is_last_tile {
            if last_token_only {
                LmHeadMode::LastRowOnly
            } else {
                LmHeadMode::Full
            }
        } else {
            LmHeadMode::Skip
        };

        // Re-borrow the optional `&mut LinearAttentionState` for this tile.
        // `Option<&mut T>::as_deref_mut()` produces `Option<&mut T>` again.
        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();

        let (tile_logits, _tile_hidden) = model_forward_paged_inner(
            backend,
            &token_ids[cursor..end],
            weights,
            config,
            paged_cache,
            block_table,
            start_pos + cursor,
            state_for_tile,
            lora,
            None,
            mode,
        )
        .with_context(|| {
            format!(
                "streaming prefill tile [{cursor}, {end}) of {total} (start_pos={})",
                start_pos + cursor
            )
        })?;

        if is_last_tile {
            last_logits = tile_logits;
        }

        cursor = end;
    }

    last_logits.context("streaming prefill produced no logits (empty token_ids)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    /// Tests all run on `Device::Cpu`, so the `CpuBackend` (all kernel methods
    /// return `Ok(None)`) is the right dispatch target.
    fn test_backend(device: &Device) -> CpuBackend {
        CpuBackend::new(device.clone())
    }

    #[test]
    fn test_embedding_lookup() -> Result<()> {
        let device = Device::Cpu;
        // vocab_size=5, hidden_size=3
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            0.4, 0.5, 0.6, // token 1
            0.7, 0.8, 0.9, // token 2
            1.0, 1.1, 1.2, // token 3
            1.3, 1.4, 1.5, // token 4
        ];
        let embed = Tensor::new(embed_data, &device)?.reshape((5, 3))?;

        let result = embedding_lookup(&[2, 0, 4], &embed)?;
        assert_eq!(result.dims(), &[3, 3]); // [seq_len=3, hidden_size=3]

        let vals = result.to_vec2::<f32>()?;
        // Token 2
        assert!((vals[0][0] - 0.7).abs() < 1e-6);
        assert!((vals[0][1] - 0.8).abs() < 1e-6);
        assert!((vals[0][2] - 0.9).abs() < 1e-6);
        // Token 0
        assert!((vals[1][0] - 0.1).abs() < 1e-6);
        // Token 4
        assert!((vals[2][0] - 1.3).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_lookup_from_transposed_matches_table() -> Result<()> {
        let device = Device::Cpu;
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
            0.7, 0.8, 0.9, //
            1.0, 1.1, 1.2, //
            1.3, 1.4, 1.5,
        ];
        let embed = Tensor::new(embed_data, &device)?.reshape((5, 3))?;
        let embed_t = embed.t()?.contiguous()?;

        let direct = embedding_lookup(&[2, 0, 4], &embed)?;
        let transposed = embedding_lookup_from_transposed(&[2, 0, 4], &embed_t)?;

        assert_eq!(transposed.dims(), direct.dims());
        assert_eq!(transposed.to_vec2::<f32>()?, direct.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_known_values() -> Result<()> {
        let device = Device::Cpu;
        // x = [1, 2, 3], weight = [0, 0, 0], eps = 0
        // Effective weight = 1 + w = [1, 1, 1]
        // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.1602
        // normed = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?.unsqueeze(0)?; // [1, 3]
        let w = Tensor::new(&[0.0_f32, 0.0, 0.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        let rms = (14.0_f64 / 3.0).sqrt();
        assert!((vals[0][0] as f64 - 1.0 / rms).abs() < 1e-4);
        assert!((vals[0][1] as f64 - 2.0 / rms).abs() < 1e-4);
        assert!((vals[0][2] as f64 - 3.0 / rms).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_rms_norm_with_weight() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[2.0_f32, 2.0, 2.0], &device)?.unsqueeze(0)?;
        let w = Tensor::new(&[0.5_f32, 1.0, 2.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        // RMS of [2,2,2] = 2.0, so normed = [1,1,1]
        // Effective weight = 1 + w = [1.5, 2.0, 3.0]
        // After weight: [1.5, 2.0, 3.0]
        assert!((vals[0][0] - 1.5).abs() < 1e-4);
        assert!((vals[0][1] - 2.0).abs() < 1e-4);
        assert!((vals[0][2] - 3.0).abs() < 1e-4);

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gated_rms_norm_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_gated_rms_norm_matches_fallback");
            return Ok(());
        };
        let backend = crate::backend::for_device(&device);
        if !backend.supports_gdn_gated_rms_norm() {
            eprintln!("Metal gated RMSNorm disabled, skipping parity test");
            return Ok(());
        }

        let batch = 1usize;
        let seq_len = 3usize;
        let heads = 32usize;
        let hidden = 128usize;
        let elems = batch * seq_len * heads * hidden;

        let mut rng = StdRng::seed_from_u64(0x6A7E_DA75);
        let x_data: Vec<f32> = (0..elems).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let z_data: Vec<f32> = (0..elems).map(|_| rng.gen_range(-2.0f32..2.0f32)).collect();
        let w_data: Vec<f32> = (0..hidden).map(|_| rng.gen_range(0.5f32..1.5f32)).collect();

        let x = Tensor::from_slice(&x_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let z = Tensor::from_slice(&z_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&w_data, (hidden,), &device)?.to_dtype(DType::BF16)?;

        let fallback = gated_rms_norm_fallback(&x, &z, &weight, 1e-6)?;
        let fused = backend
            .gdn_gated_rms_norm(&x, &z, &weight, 1e-6)?
            .context("Metal backend declined gated RMSNorm test shape")?;

        assert_eq!(fused.dims(), fallback.dims());
        assert_eq!(fused.dtype(), DType::BF16);

        let diff = (fused.to_dtype(DType::F32)?
            - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("gated_rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 5e-3,
            "Metal gated_rms_norm max_abs_diff={max:e} exceeds 5e-3"
        );
        assert!(
            mean < 5e-4,
            "Metal gated_rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_rms_norm_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_rms_norm_matches_fallback");
            return Ok(());
        };

        let batch = 2usize;
        let seq_len = 3usize;
        let hidden = 4096usize;
        let elems = batch * seq_len * hidden;

        let mut rng = StdRng::seed_from_u64(0xA11CE);
        let x_data: Vec<f32> = (0..elems).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let w_data: Vec<f32> = (0..hidden)
            .map(|_| rng.gen_range(-0.2f32..0.2f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, seq_len, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&w_data, (hidden,), &device)?.to_dtype(DType::BF16)?;

        assert!(crate::backend::metal::metal_rms_norm_supports(&x, &weight));
        let fallback = rms_norm_fallback(&x, &weight, 1e-6)?;
        let fused = crate::backend::metal::metal_rms_norm_bf16(&x, &weight, 1e-6)?;

        assert_eq!(fused.dims(), fallback.dims());
        assert_eq!(fused.dtype(), DType::BF16);

        let diff = (fused.to_dtype(DType::F32)?
            - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 5e-3,
            "Metal rms_norm max_abs_diff={max:e} exceeds 5e-3"
        );
        assert!(
            mean < 5e-4,
            "Metal rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_rotary_embedding_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_rotary_embedding_matches_fallback");
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 5usize;
        let q_heads = 4usize;
        let k_heads = 2usize;
        let head_dim = 16usize;
        let rotary_dim = 8usize;
        let mut rng = StdRng::seed_from_u64(0xA07A_7E55);
        let q_data: Vec<f32> = (0..batch * seq_len * q_heads * head_dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let k_data: Vec<f32> = (0..batch * seq_len * k_heads * head_dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let q = Tensor::from_slice(&q_data, (batch, seq_len, q_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (batch, seq_len, k_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let positions: Vec<f32> = (11..11 + seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_slice(&positions, (seq_len,), &device)?;
        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (cos, sin) = rotary_tables_from_tensor(&positions, &inv_freq)?;

        assert!(crate::backend::metal::metal_rotary_embedding_supports(
            &q, &k, &cos, &sin, head_dim, rotary_dim,
        ));
        let (q_fused, k_fused) = crate::backend::metal::metal_rotary_embedding_bf16(
            &q, &k, &cos, &sin, head_dim, rotary_dim,
        )?;
        let q_ref = apply_rope(&q, &cos, &sin, head_dim, rotary_dim)?;
        let k_ref = apply_rope(&k, &cos, &sin, head_dim, rotary_dim)?;

        let q_diff = (q_fused.to_dtype(DType::F32)?
            - q_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
        .abs()?;
        let k_diff = (k_fused.to_dtype(DType::F32)?
            - k_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
        .abs()?;
        let q_max = q_diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let k_max = k_diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(q_max < 1e-6, "Metal rotary Q max_abs_diff={q_max:e}");
        assert!(k_max < 1e-6, "Metal rotary K max_abs_diff={k_max:e}");

        Ok(())
    }

    #[test]
    fn test_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 8;

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(
            0.0_f32,
            1.0,
            (batch, seq_len, num_kv_heads, head_dim),
            &device,
        )?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, head_dim, &inv_freq)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_rope_position_zero_is_identity() -> Result<()> {
        let device = Device::Cpu;
        // At position 0, cos=1 and sin=0, so rotation should be identity
        let head_dim = 4;
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, _rk) = rotary_embedding(&q, &k, &[0], head_dim, head_dim, &inv_freq)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        for i in 0..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-5,
                "Position 0 should be identity, dim {i}: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_rope_different_positions_differ() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let q = Tensor::ones((1, 2, 1, head_dim), DType::F32, &device)?;
        let k = q.clone();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, _) = rotary_embedding(&q, &k, &[0, 100], head_dim, head_dim, &inv_freq)?;
        // rq shape: [1, 2, 1, 8] — extract pos 0 and pos 100
        let pos0 = rq.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let pos100 = rq.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;

        let diff: f32 = pos0.iter().zip(&pos100).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "Different positions should produce different embeddings"
        );

        Ok(())
    }

    #[test]
    fn test_partial_rope_passthrough_dims_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let rotary_dim = 4; // only rotate first 4 dims, last 4 pass through
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        // Position 100 — the rotary dims should change, passthrough dims should not
        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (rq, _) = rotary_embedding(&q, &k, &[100], head_dim, rotary_dim, &inv_freq)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        // First rotary_dim dims should be different at non-zero position
        let rotary_diff: f32 = (0..rotary_dim).map(|i| (orig[i] - rotated[i]).abs()).sum();
        assert!(
            rotary_diff > 0.01,
            "Rotary dims should change at position 100"
        );

        // Passthrough dims (rotary_dim..head_dim) must be identical
        for i in rotary_dim..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-6,
                "Passthrough dim {i} should be unchanged: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_partial_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 16;
        let rotary_dim = 4; // partial rotation

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(
            0.0_f32,
            1.0,
            (batch, seq_len, num_kv_heads, head_dim),
            &device,
        )?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, rotary_dim, &inv_freq)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_swiglu_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let gate = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let up = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let down = Tensor::randn(0.0_f32, 0.1, (hidden, intermediate), &device)?;
        let gate_t = gate.t()?.contiguous()?;
        let up_t = up.t()?.contiguous()?;
        let down_t = down.t()?.contiguous()?;

        let mlp = GpuFfnWeights {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            gate_proj_t: gate_t,
            up_proj_t: up_t,
            down_proj_t: down_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let result = swiglu_ffn(&x, &mlp, None)?;
        assert_eq!(result.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_swiglu_zero_gate_gives_zero() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        // Gate weights all zero -> silu(0) = 0 -> output is zero regardless of up/down
        let gate = Tensor::zeros((intermediate, hidden), DType::F32, &device)?;
        let up = Tensor::ones((intermediate, hidden), DType::F32, &device)?;
        let down = Tensor::ones((hidden, intermediate), DType::F32, &device)?;
        let gate_t = gate.t()?.contiguous()?;
        let up_t = up.t()?.contiguous()?;
        let down_t = down.t()?.contiguous()?;

        let mlp = GpuFfnWeights {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            gate_proj_t: gate_t,
            up_proj_t: up_t,
            down_proj_t: down_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let result = swiglu_ffn(&x, &mlp, None)?;
        let vals = result.to_vec3::<f32>()?;

        for v in &vals[0][0] {
            assert!(
                v.abs() < 1e-6,
                "SwiGLU with zero gate should produce zero, got {v}"
            );
        }

        Ok(())
    }

    /// Create a minimal config for tests (no output gate, simple dims).
    fn make_test_config(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
    ) -> kiln_core::config::ModelConfig {
        kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 4,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: hidden * 2,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0, // tests use full rotation by default
        }
    }

    #[test]
    fn test_linear_attention_state_prefix_snapshot_truncates_draft_state() -> Result<()> {
        let device = Device::Cpu;
        let config = make_test_config(2, 1, 4, 8);
        let state = LinearAttentionState::new(&config, &device)?;

        assert_eq!(state.recurrent_states.len(), 3);
        assert_eq!(state.conv_states.len(), 3);

        let draft = state.snapshot_for_decode_rollback_prefix(1)?;
        assert_eq!(draft.recurrent_states.len(), 1);
        assert_eq!(draft.conv_states.len(), 1);
        Ok(())
    }

    fn make_test_attn_weights(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<GpuFullAttentionWeights> {
        let q_proj = Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, hidden), device)?;
        let k_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
        let v_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
        let o_proj = Tensor::randn(0.0_f32, 0.02, (hidden, num_heads * head_dim), device)?;
        let q_proj_t = q_proj.t()?.contiguous()?;
        let k_proj_t = k_proj.t()?.contiguous()?;
        let v_proj_t = v_proj.t()?.contiguous()?;
        let o_proj_t = o_proj.t()?.contiguous()?;
        Ok(GpuFullAttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            q_proj_t,
            k_proj_t,
            v_proj_t,
            o_proj_t,
            q_proj_marlin: None,
        })
    }

    #[test]
    fn test_gqa_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim; // 32

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_gqa_head_expansion() -> Result<()> {
        // Verify GQA works: 4 Q heads, 2 KV heads (ratio=2)
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 0.5, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        // Output should be finite and not all zeros
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output should be finite"
        );
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(sum > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_gqa_single_token() -> Result<()> {
        // Single token should work (no causal masking needed)
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 1.0, (1, 1, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &[0],
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[1, 1, hidden]);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        // A 3x3 score matrix
        let scores = Tensor::ones((1, 1, 3, 3), DType::F32, &device)?;
        let masked = apply_causal_mask(&scores, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0: [1, -inf, -inf]
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert!(vals[2].is_infinite() && vals[2] < 0.0);
        // Row 1: [1, 1, -inf]
        assert!((vals[3] - 1.0).abs() < 1e-6);
        assert!((vals[4] - 1.0).abs() < 1e-6);
        assert!(vals[5].is_infinite() && vals[5] < 0.0);
        // Row 2: [1, 1, 1]
        assert!((vals[6] - 1.0).abs() < 1e-6);
        assert!((vals[7] - 1.0).abs() < 1e-6);
        assert!((vals[8] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_transformer_block_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(make_test_attn_weights(
                num_heads,
                num_kv_heads,
                head_dim,
                hidden,
                &device,
            )?),
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_transformer_block_residual_connections() -> Result<()> {
        // With residual connections, output should differ from zero even with small weights
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        // Input with known non-zero values
        let x = Tensor::ones((1, 2, hidden), DType::F32, &device)?;
        let positions = vec![0u32, 1];

        let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(make_test_attn_weights(
                num_heads,
                num_kv_heads,
                head_dim,
                hidden,
                &device,
            )?),
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        )?;

        // Output should not be zero (residual adds input through)
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(
            sum > 0.1,
            "residual connections should keep output non-zero, got sum={sum}"
        );
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output should be finite"
        );

        Ok(())
    }

    #[test]
    fn test_transformer_block_rejects_linear_attention() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 8;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_z: Tensor::zeros((1, 1), DType::F32, &device)?,
                out_proj: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_a: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_b: Tensor::zeros((1, 1), DType::F32, &device)?,
                conv1d: Tensor::zeros((1, 1), DType::F32, &device)?,
                norm: Tensor::zeros((1,), DType::F32, &device)?,
                a_log: Tensor::zeros((1,), DType::F32, &device)?,
                dt_bias: Tensor::zeros((1,), DType::F32, &device)?,
                in_proj_qkv_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_z_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_a_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_b_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                out_proj_t: Tensor::zeros((1, 1), DType::F32, &device)?,
            }),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                up_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                down_proj: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                gate_proj_t: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                up_proj_t: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                down_proj_t: Tensor::zeros((1, hidden), DType::F32, &device)?,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        let cfg = make_test_config(2, 1, 4, 8);
        let inv_freq = compute_rotary_inv_freq(4, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let result = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &[0],
            2,
            1,
            4,
            4,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        );
        assert!(result.is_err(), "should reject linear attention layers");

        Ok(())
    }

    #[test]
    fn test_weight_to_tensor_f32() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let t = weight_to_tensor(&wt, &device)?;
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);

        let vals = t.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.0).abs() < 1e-6);
        assert!((vals[1][2] - 6.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_weight_to_transposed_tensor_2d_f32_matches_cached_transpose() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let direct = weight_to_transposed_tensor_2d(&wt, &device)?;
        let baseline = cached_transpose(&weight_to_tensor(&wt, &device)?)?;

        assert!(direct.is_contiguous());
        assert_eq!(direct.dims(), &[3, 2]);
        assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
        assert_eq!(
            direct.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    #[test]
    fn test_transposed_weight_bytes_2d_preserves_two_byte_elements() -> Result<()> {
        let values: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::BF16,
            source: None,
        };

        let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;
        let got: Vec<u16> = transposed
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        assert_eq!(shape, [3, 2]);
        assert_eq!(got, vec![1, 4, 2, 5, 3, 6]);
        Ok(())
    }

    #[test]
    fn test_transposed_weight_bytes_2d_parallel_preserves_two_byte_elements() -> Result<()> {
        let rows = 513usize;
        let cols = 1025usize;
        let values: Vec<u16> = (0..rows * cols).map(|idx| idx as u16).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(bytes.len() >= PARALLEL_TRANSPOSE_MIN_BYTES);
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![rows, cols],
            dtype: TensorDType::BF16,
            source: None,
        };

        let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;

        assert_eq!(shape, [cols, rows]);
        for col in 0..cols {
            for row in 0..rows {
                let got_offset = (col * rows + row) * 2;
                let got = u16::from_le_bytes([transposed[got_offset], transposed[got_offset + 1]]);
                assert_eq!(got, values[row * cols + col]);
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_weight_to_transposed_tensor_2d_metal_matches_cpu_cached_transpose() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let data: Vec<f32> = vec![1.0, -2.0, 3.5, 4.25, 5.0, -6.75];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let direct = weight_to_transposed_tensor_2d(&wt, &device)?.to_device(&cpu)?;
        let baseline = cached_transpose(&weight_to_tensor(&wt, &cpu)?)?;

        assert!(direct.is_contiguous());
        assert_eq!(direct.dims(), &[3, 2]);
        assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_cached_transpose_materializes_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let tt = cached_transpose(&t)?;

        assert!(tt.is_contiguous());
        assert_eq!(tt.dims(), &[3, 2]);
        assert_eq!(
            tt.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cached_transpose_materializes_on_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let t = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let tt = cached_transpose(&t)?;

        assert!(tt.is_contiguous());
        assert_eq!(tt.dims(), &[3, 2]);
        assert_eq!(
            tt.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    /// Helper: build tiny GpuWeights for testing model_forward shape propagation.
    /// Uses full-attention layers only (no linear attention) with small dimensions.
    fn make_tiny_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
            let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
            let q_proj_t = q_proj.t()?.contiguous()?;
            let k_proj_t = k_proj.t()?.contiguous()?;
            let v_proj_t = v_proj.t()?.contiguous()?;
            let o_proj_t = o_proj.t()?.contiguous()?;
            let gate_proj = randn(&[intermediate_size, hidden_size])?;
            let up_proj = randn(&[intermediate_size, hidden_size])?;
            let down_proj = randn(&[hidden_size, intermediate_size])?;
            let gate_proj_t = gate_proj.t()?.contiguous()?;
            let up_proj_t = up_proj.t()?.contiguous()?;
            let down_proj_t = down_proj.t()?.contiguous()?;
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    o_proj_t,
                    q_proj_marlin: None,
                }),
                mlp: GpuFfnWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                    gate_proj_t,
                    up_proj_t,
                    down_proj_t,
                    gate_proj_marlin: None,
                    up_proj_marlin: None,
                    down_proj_marlin: None,
                },
            });
        }

        // Tests using this helper all set `partial_rotary_factor = 1.0` and
        // `rope_theta = 10000.0`, so rotate every head_dim with base 10k.
        let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[test]
    fn test_model_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1, // every layer is full attention
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let backend = test_backend(&device);
        let logits = model_forward(&backend, &token_ids, &weights, &config, None, None, None)?;

        // Expected shape: [1, seq_len, vocab_size]
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        Ok(())
    }

    #[test]
    fn test_model_forward_single_token() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            1, // single layer
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let backend = test_backend(&device);
        let logits = model_forward(&backend, &[7], &weights, &config, None, None, None)?;
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);

        // Logits should be finite
        let vals = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_equivalence() -> Result<()> {
        // Verify that model_forward with KV cache produces the same last-position
        // logits as without KV cache, for a multi-token sequence processed
        // incrementally (prefill + decode steps).
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![1, 5, 3, 10, 7];
        let backend = test_backend(&device);

        // Reference: full forward pass without KV cache
        let logits_ref = model_forward(&backend, &tokens, &weights, &config, None, None, None)?;
        // Extract last position logits: [1, 5, vocab] -> last position
        let last_ref = logits_ref.narrow(1, tokens.len() - 1, 1)?; // [1, 1, vocab]
        let last_ref_vals = last_ref.flatten_all()?.to_vec1::<f32>()?;

        // With KV cache: prefill first 4 tokens, then decode the 5th
        let mut kv_cache =
            KvCache::new(num_layers, num_kv_heads, head_dim, 32, DType::F32, &device)?;

        // Prefill
        let _prefill_logits = model_forward(
            &backend,
            &tokens[..4],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(4);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode the 5th token
        let decode_logits = model_forward(
            &backend,
            &tokens[4..],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);
        assert_eq!(kv_cache.seq_len(), 5);

        let last_cached_vals = decode_logits
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Compare: should be identical (within floating point tolerance)
        assert_eq!(last_ref_vals.len(), last_cached_vals.len());
        for (i, (r, c)) in last_ref_vals.iter().zip(&last_cached_vals).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}, diff={}",
                (r - c).abs()
            );
        }

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_token_by_token() -> Result<()> {
        // Verify that processing tokens one-by-one with KV cache matches
        // processing all at once without cache.
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            1,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![3, 7, 1];
        let backend = test_backend(&device);

        // Reference
        let logits_ref = model_forward(&backend, &tokens, &weights, &config, None, None, None)?;
        let last_ref = logits_ref
            .narrow(1, 2, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // KV cache: process token by token
        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 16, DType::F32, &device)?;

        // Token 0
        let _ = model_forward(
            &backend,
            &[3],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        // Token 1
        let _ = model_forward(
            &backend,
            &[7],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        // Token 2
        let logits_cached = model_forward(
            &backend,
            &[1],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        let last_cached = logits_cached
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for (i, (r, c)) in last_ref.iter().zip(&last_cached).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}",
            );
        }

        Ok(())
    }

    /// Helper: build tiny GpuWeights with a mix of full and linear attention layers.
    fn make_hybrid_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        full_attention_interval: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        // For linear attention: nk heads with key_head_dim, nv heads with value_head_dim
        // Use same dims as full attention for simplicity
        let nk = num_kv_heads;
        let nv = num_heads;
        let dk = head_dim;
        let dv = head_dim;
        let qkv_dim = nk * dk + nk * dk + nv * dv; // Q + K + V fused
        let conv_kernel = 4;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let is_full = (i + 1) % full_attention_interval == 0;
            let attention = if is_full {
                let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
                let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
                let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
                let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
                let q_proj_t = q_proj.t()?.contiguous()?;
                let k_proj_t = k_proj.t()?.contiguous()?;
                let v_proj_t = v_proj.t()?.contiguous()?;
                let o_proj_t = o_proj.t()?.contiguous()?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    o_proj_t,
                    q_proj_marlin: None,
                })
            } else {
                let in_proj_qkv = randn(&[qkv_dim, hidden_size])?;
                let in_proj_z = randn(&[nv * dv, hidden_size])?;
                let out_proj = randn(&[hidden_size, nv * dv])?;
                let in_proj_a = randn(&[nv, hidden_size])?;
                let in_proj_b = randn(&[nv, hidden_size])?;
                let in_proj_qkv_t = in_proj_qkv.t()?.contiguous()?;
                let in_proj_z_t = in_proj_z.t()?.contiguous()?;
                let in_proj_a_t = in_proj_a.t()?.contiguous()?;
                let in_proj_b_t = in_proj_b.t()?.contiguous()?;
                let out_proj_t = out_proj.t()?.contiguous()?;
                GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                    in_proj_qkv,
                    in_proj_z,
                    out_proj,
                    in_proj_a,
                    in_proj_b,
                    conv1d: randn(&[qkv_dim, 1, conv_kernel])?,
                    norm: Tensor::ones(dk, DType::F32, device)?,
                    a_log: Tensor::zeros(nv, DType::F32, device)?,
                    dt_bias: Tensor::zeros(nv, DType::F32, device)?,
                    in_proj_qkv_t,
                    in_proj_z_t,
                    in_proj_a_t,
                    in_proj_b_t,
                    out_proj_t,
                })
            };

            let gate_proj = randn(&[intermediate_size, hidden_size])?;
            let up_proj = randn(&[intermediate_size, hidden_size])?;
            let down_proj = randn(&[hidden_size, intermediate_size])?;
            let gate_proj_t = gate_proj.t()?.contiguous()?;
            let up_proj_t = up_proj.t()?.contiguous()?;
            let down_proj_t = down_proj.t()?.contiguous()?;
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention,
                mlp: GpuFfnWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                    gate_proj_t,
                    up_proj_t,
                    down_proj_t,
                    gate_proj_marlin: None,
                    up_proj_marlin: None,
                    down_proj_marlin: None,
                },
            });
        }

        // Tests using this helper set `partial_rotary_factor = 1.0` and
        // `rope_theta = 10000.0`, so rotary_dim = head_dim with base 10k.
        let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[test]
    fn test_model_forward_hybrid_layers() -> Result<()> {
        // Test model_forward with a mix of full and linear (GDN) attention layers
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4; // layer 3 is full, layers 0,1,2 are linear

        let weights = make_hybrid_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
            full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut linear_state = LinearAttentionState::new(&config, &device)?;

        // Prefill with multiple tokens
        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let backend = test_backend(&device);
        let logits = model_forward(
            &backend,
            &token_ids,
            &weights,
            &config,
            None,
            Some(&mut linear_state),
            None,
        )?;
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        // All values should be finite (no NaN/Inf)
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "logits contain non-finite values"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    struct ParityScenario {
        label: &'static str,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        full_attention_interval: usize,
        token_ids: Vec<u32>,
        max_abs_diff: f32,
    }

    /// Runs `model_forward` on CPU and Metal with matching random-weight
    /// models and asserts the logits agree within `scenario.max_abs_diff`.
    /// Drives both parity tests below; the scenario controls whether the
    /// `MetalBackend` SDPA path activates (head_dim ∈ whitelist) or whether
    /// the portable candle fallback runs.
    ///
    /// Returns `Ok(())` without running if Metal isn't available so the
    /// suite stays portable on Linux + CUDA hosts.
    #[cfg(feature = "metal")]
    fn run_cpu_metal_parity(scenario: ParityScenario) -> Result<()> {
        let Some(metal_device) = crate::backend::metal::try_new_metal() else {
            eprintln!("skipping parity test '{}'", scenario.label);
            return Ok(());
        };
        let cpu_device = Device::Cpu;

        let weights_cpu = make_hybrid_gpu_weights(
            &cpu_device,
            scenario.vocab_size,
            scenario.hidden_size,
            scenario.num_heads,
            scenario.num_kv_heads,
            scenario.head_dim,
            scenario.intermediate_size,
            scenario.num_layers,
            scenario.full_attention_interval,
        )?;
        let weights_metal = make_hybrid_gpu_weights(
            &metal_device,
            scenario.vocab_size,
            scenario.hidden_size,
            scenario.num_heads,
            scenario.num_kv_heads,
            scenario.head_dim,
            scenario.intermediate_size,
            scenario.num_layers,
            scenario.full_attention_interval,
        )?;

        // Linear attention dims are 0 when full_attention_interval == 1 (no
        // GDN layers in the model); otherwise set to head_dim so GDN state
        // is shaped for the fallback path.
        let has_linear_layers = scenario.full_attention_interval > 1;
        let linear_num_kv_heads = if has_linear_layers {
            scenario.num_kv_heads
        } else {
            0
        };
        let linear_num_value_heads = if has_linear_layers {
            scenario.num_heads
        } else {
            0
        };
        let linear_head_dim = if has_linear_layers {
            scenario.head_dim
        } else {
            0
        };
        let linear_conv_kernel_dim = if has_linear_layers { 4 } else { 0 };

        let config = kiln_core::config::ModelConfig {
            hidden_size: scenario.hidden_size,
            num_layers: scenario.num_layers,
            num_attention_heads: scenario.num_heads,
            num_kv_heads: scenario.num_kv_heads,
            head_dim: scenario.head_dim,
            intermediate_size: scenario.intermediate_size,
            vocab_size: scenario.vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: if has_linear_layers {
                1
            } else {
                scenario.num_layers
            },
            full_attention_interval: scenario.full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: linear_num_kv_heads,
            linear_key_head_dim: linear_head_dim,
            linear_num_value_heads,
            linear_value_head_dim: linear_head_dim,
            linear_conv_kernel_dim,
            partial_rotary_factor: 1.0,
        };

        let cpu_backend = test_backend(&cpu_device);
        let mut cpu_linear = LinearAttentionState::new(&config, &cpu_device)?;
        let logits_cpu = model_forward(
            &cpu_backend,
            &scenario.token_ids,
            &weights_cpu,
            &config,
            None,
            Some(&mut cpu_linear),
            None,
        )?;

        let metal_backend = crate::backend::for_device(&metal_device);
        let mut metal_linear = LinearAttentionState::new(&config, &metal_device)?;
        let logits_metal = model_forward(
            &*metal_backend,
            &scenario.token_ids,
            &weights_metal,
            &config,
            None,
            Some(&mut metal_linear),
            None,
        )?;

        assert_eq!(logits_cpu.dims(), logits_metal.dims());

        let cpu_flat = logits_cpu.flatten_all()?.to_vec1::<f32>()?;
        let metal_flat = logits_metal
            .to_device(&cpu_device)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert!(
            cpu_flat.iter().all(|v| v.is_finite()),
            "{}: CPU logits non-finite",
            scenario.label
        );
        assert!(
            metal_flat.iter().all(|v| v.is_finite()),
            "{}: Metal logits non-finite",
            scenario.label
        );

        let max_abs_diff = cpu_flat
            .iter()
            .zip(metal_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < scenario.max_abs_diff,
            "{}: CPU vs Metal logits diverge: max abs diff = {max_abs_diff} (bound {})",
            scenario.label,
            scenario.max_abs_diff,
        );
        Ok(())
    }

    /// Qwen-shaped: GQA ratio 4, head_dim 128, full attention only. Exercises
    /// `MetalBackend::flash_attn_prefill` (candle SDPA) directly — head_dim
    /// 128 is in the SDPA whitelist, seq_len 12 > 8 for the full SDPA kernel
    /// (not the vector path).
    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_parity_sdpa_path() -> Result<()> {
        run_cpu_metal_parity(ParityScenario {
            label: "sdpa_path",
            vocab_size: 32,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 128,
            hidden_size: 512,
            intermediate_size: 1024,
            num_layers: 2,
            full_attention_interval: 1,
            token_ids: (0..12u32).collect(),
            // SDPA internally accumulates at FP32 but softmax rounds differently
            // from the naive CPU path. 1e-2 accommodates M1 drift; tighten if
            // later hardware proves it's conservative.
            max_abs_diff: 1e-2,
        })
    }

    /// Hybrid full + GDN layers with head_dim 4, below the SDPA whitelist.
    /// `MetalBackend` declines into the portable fallback, so this validates
    /// that the whole candle composition (embed, RMSNorm, RoPE, SwiGLU, naive
    /// softmax+matmul, GDN recurrent loop) runs correctly on Apple Silicon.
    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_parity_cpu_vs_metal() -> Result<()> {
        run_cpu_metal_parity(ParityScenario {
            label: "portable_fallback",
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            num_layers: 4,
            full_attention_interval: 4,
            token_ids: vec![1, 5, 3, 10],
            max_abs_diff: 1e-3,
        })
    }

    #[test]
    fn test_model_forward_hybrid_decode() -> Result<()> {
        // Test prefill + decode with linear attention state persistence
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4;

        let weights = make_hybrid_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
            full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 32, DType::F32, &device)?;
        let mut linear_state = LinearAttentionState::new(&config, &device)?;
        let backend = test_backend(&device);

        // Prefill
        let prefill_logits = model_forward(
            &backend,
            &[1, 5, 3],
            &weights,
            &config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
        )?;
        kv_cache.advance(3);
        assert_eq!(prefill_logits.dims(), &[1, 3, vocab_size]);

        // Decode: single token should work with persisted linear state
        let decode_logits = model_forward(
            &backend,
            &[10],
            &weights,
            &config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
        )?;
        kv_cache.advance(1);
        assert_eq!(decode_logits.dims(), &[1, 1, vocab_size]);

        // Both should produce finite values
        let flat = decode_logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "decode logits contain non-finite values"
        );

        Ok(())
    }

    #[test]
    fn test_linear_attention_state_new() -> Result<()> {
        let device = Device::Cpu;
        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let state = LinearAttentionState::new(&config, &device)?;
        // 3 linear layers (layers 0,1,2; layer 3 is full)
        assert_eq!(state.recurrent_states.len(), 3);
        assert_eq!(state.conv_states.len(), 3);
        // Recurrent state shape: [1, nv, dk, dv]
        assert_eq!(state.recurrent_states[0].dims(), &[1, 4, 4, 4]);
        assert_eq!(state.recurrent_states[0].dtype(), DType::F32);
        // Conv state shape: [1, qkv_dim, kernel_size-1] where qkv_dim = 2*(nk*dk) + nv*dv = 2*8+16=32
        let qkv_dim = 2 * (2 * 4) + 4 * 4; // 32
        assert_eq!(state.conv_states[0].dims(), &[1, qkv_dim, 3]);
        assert_eq!(state.conv_states[0].dtype(), DType::F32);

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_linear_attention_state_uses_bf16_on_metal_for_bf16_models() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };

        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let state = LinearAttentionState::new(&config, &device)?;
        assert_eq!(state.recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(state.conv_states[0].dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn test_causal_mask_with_offset() -> Result<()> {
        let device = Device::Cpu;
        // Simulate decode: 1 new query, 4 total KV (3 cached + 1 new)
        let scores = Tensor::ones((1, 1, 1, 4), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 1, 4, 3)?;
        // Single query should attend to all 4 positions (no masking for q_len=1)
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| (*v - 1.0).abs() < 1e-6),
            "single query token should attend to all KV positions"
        );

        // Simulate prefill with offset: 2 new queries, 5 total KV (3 cached + 2 new)
        let scores = Tensor::ones((1, 1, 2, 5), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 2, 5, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0 (abs pos 3): can attend to positions 0..4 (first 4), mask position 4
        assert!((vals[0] - 1.0).abs() < 1e-6); // pos 0: ok
        assert!((vals[1] - 1.0).abs() < 1e-6); // pos 1: ok
        assert!((vals[2] - 1.0).abs() < 1e-6); // pos 2: ok
        assert!((vals[3] - 1.0).abs() < 1e-6); // pos 3 (self): ok
        assert!(vals[4].is_infinite() && vals[4] < 0.0); // pos 4: masked
        // Row 1 (abs pos 4): can attend to all 5 positions
        assert!(vals[5..10].iter().all(|v| (*v - 1.0).abs() < 1e-6));

        Ok(())
    }

    // ------------------------------------------------------------------
    // GDN chunkwise correctness test (Phase 6)
    // ------------------------------------------------------------------

    /// Reference per-token GDN recurrence, mirroring the pre-Phase-6 loop
    /// that used to live in `gated_deltanet_forward`. Kept in the test
    /// module (never called from production) so the chunkwise implementation
    /// can be cross-checked against the arithmetically simple form.
    ///
    /// Inputs are already transposed to [B, nv, T, *]; state is [B, nv, dk, dv].
    fn gdn_sequential_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Tensor> {
        let (_, _, seq_len, _) = q.dims4()?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1)?; // [B, nv, 1, dk]
            let k_t = k.narrow(2, t, 1)?; // [B, nv, 1, dk]
            let v_t = v.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dv]
            let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]
            let g_t = g.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]

            let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?; // [B, nv, 1, 1]
            *state = state.broadcast_mul(&g_exp)?;

            let kv_mem = k_t.matmul(&*state)?.squeeze(2)?; // [B, nv, dv]
            let delta: Tensor = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?; // [B, nv, dv]

            let k_col = k_t.squeeze(2)?.unsqueeze(3)?; // [B, nv, dk, 1]
            let outer = k_col.broadcast_mul(&delta.unsqueeze(2)?)?; // [B, nv, dk, dv]
            *state = (&*state + &outer)?;

            let out_t = q_t.matmul(&*state)?; // [B, nv, 1, dv]
            outputs.push(out_t);
        }
        Ok(Tensor::cat(&outputs, 2)?)
    }

    /// Deterministic tensor of the given shape filled with values from a
    /// simple hash of the index. Avoids depending on candle's RNG (which
    /// uses process-global state) and keeps the test reproducible.
    fn det_tensor(shape: &[usize], scale: f32, bias: f32, device: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                // Cheap mixable pseudo-random: stretch i through two sin
                // waves of different frequencies. Gives values in roughly
                // [-1, 1] with no exact repeats for small n.
                let x = (i as f32 * 0.7283).sin() + (i as f32 * 1.3719).cos();
                (x * 0.5) * scale + bias
            })
            .collect();
        Ok(Tensor::from_vec(data, shape, device)?)
    }

    #[test]
    fn test_gdn_chunkwise_matches_sequential() -> Result<()> {
        // Small, fully-on-CPU shapes. We use F32 here so the comparison
        // is against the same numerical path the chunkwise form takes
        // for its decay cumulative products; the task spec's bf16
        // tolerance (<1e-3) is comfortably satisfied in F32 as well.
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 8;
        let dk = 4;
        let dv = 4;
        let chunk_size = 4;

        let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        // beta ∈ (0, 1): pass through sigmoid-like shift.
        let beta_raw = det_tensor(&[b, nv, t], 2.0, 0.0, &device)?.to_dtype(dtype)?;
        let beta = {
            let ones = Tensor::ones_like(&beta_raw)?;
            (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
        };
        // g ∈ (-0.2, 0): small negative decays so cumulative sum stays sane.
        let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
        let g = (g_raw.abs()? * (-1.0_f64))?;

        let state_init = Tensor::zeros((b, nv, dk, dv), dtype, &device)?;
        let backend = test_backend(&device);

        let mut state_chunk = state_init.clone();
        let out_chunk = gdn_chunkwise_recurrence(
            &backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_chunk,
            chunk_size,
        )?;

        let mut state_seq = state_init.clone();
        let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

        let out_diff = (&out_chunk - &out_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let state_diff = (&state_chunk - &state_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;

        // Task acceptance: max abs diff < 1e-3 in bf16. We run the test in
        // F32 so the actual tolerance is much tighter; guard against both
        // silent divergence and silent upgrade of the bf16 tolerance bound.
        assert!(
            out_diff < 1e-3,
            "chunkwise vs sequential output diff too large: {out_diff}",
        );
        assert!(
            state_diff < 1e-3,
            "chunkwise vs sequential state diff too large: {state_diff}",
        );

        // Also test chunk_size >= seq_len (single-chunk path) and
        // chunk_size == 1 (decode-like path) for coverage.
        for &cs in &[1usize, t] {
            let mut state_a = state_init.clone();
            let out_a =
                gdn_chunkwise_recurrence(&backend, &q, &k, &v, &beta, &g, &mut state_a, cs)?;
            let mut state_b = state_init.clone();
            let out_b = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_b)?;
            let d = (&out_a - &out_b)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let sd = (&state_a - &state_b)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(d < 1e-3, "chunkwise(cs={cs}) output diff {d}");
            assert!(sd < 1e-3, "chunkwise(cs={cs}) state diff {sd}");
        }

        Ok(())
    }

    #[test]
    fn test_gdn_single_token_matches_sequential() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 1;
        let dk = 4;
        let dv = 4;

        let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 0.8, 0.1, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 0.6, -0.2, &device)?.to_dtype(dtype)?;
        let beta_raw = det_tensor(&[b, nv, t], 1.5, 0.0, &device)?.to_dtype(dtype)?;
        let beta = {
            let ones = Tensor::ones_like(&beta_raw)?;
            (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
        };
        let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
        let g = (g_raw.abs()? * (-1.0_f64))?;

        let state_init = det_tensor(&[b, nv, dk, dv], 0.1, 0.0, &device)?.to_dtype(dtype)?;
        let backend = test_backend(&device);

        let mut state_fast = state_init.clone();
        let out_fast = gdn_chunkwise_recurrence(
            &backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_fast,
            GDN_CHUNK_SIZE,
        )?;

        let mut state_seq = state_init.clone();
        let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

        let out_diff = (&out_fast - &out_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let state_diff = (&state_fast - &state_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;

        assert!(
            out_diff < 1e-5,
            "single-token fast path output drifted: max_abs_diff={out_diff:e}"
        );
        assert!(
            state_diff < 1e-5,
            "single-token fast path state drifted: max_abs_diff={state_diff:e}"
        );
        Ok(())
    }

    /// Correctness test for the vendored kiln-gdn-kernel CUDA fused
    /// forward-substitution kernel.
    ///
    /// Compares the fused kernel output against the per-token candle
    /// fallback on the same random bf16 inputs at kiln's exact GDN config
    /// (B=1, nv=32, C=64, dv=128). Asserts max abs diff < 1e-2 and mean
    /// abs diff < 1e-3 — the fused path uses F32 accumulators and
    /// per-token bf16 round-trips, so finite-precision drift is bounded
    /// by bf16 rounding noise.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_kernel_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_kernel_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);

        let n_a = b * nv * c * c;
        let n_v = b * nv * c * dv;
        let n_b = b * nv * c;

        let a_data: Vec<f32> = (0..n_a).map(|_| rng.gen_range(-0.05f32..0.05f32)).collect();
        let v_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(0.5f32..1.5f32)).collect();

        let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c), &device)?;

        // Make A_strict actually strictly lower triangular (matches what
        // the recurrence produces upstream of compute_w_chunk).
        let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
        let a_f32 = a_f32.broadcast_mul(&mask)?;

        let a = a_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;

        let backend = crate::backend::for_device(&device);
        let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?; // CUDA kernel
        let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?; // candle per-token

        let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!("gdn-kernel vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}");

        assert!(
            max < 1e-2,
            "kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );

        Ok(())
    }

    /// Correctness test for the Metal fused forward-substitution kernel.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gdn_forward_substitution_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_gdn_forward_substitution_matches_fallback"
            );
            return Ok(());
        };

        let b = 1usize;
        let nv = 8usize;
        let c = 16usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xFACE_FEED_u64);

        let n_a = b * nv * c * c;
        let n_v = b * nv * c * dv;
        let n_b = b * nv * c;

        let a_data: Vec<f32> = (0..n_a).map(|_| rng.gen_range(-0.05f32..0.05f32)).collect();
        let v_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(0.5f32..1.5f32)).collect();

        let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c), &device)?;

        let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
        let a_f32 = a_f32.broadcast_mul(&mask)?;

        let a = a_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;

        let backend = crate::backend::for_device(&device);
        let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?;
        let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?;

        let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "metal gdn-forward-sub vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}"
        );

        assert!(
            max < 1e-2,
            "metal forward-sub kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "metal forward-sub kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );

        Ok(())
    }

    /// Parity check for the single-token recurrent CUDA kernel.
    ///
    /// Compares output and final state of `gdn_chunkwise_recurrence` with
    /// the new fused recurrent kernel against `gdn_sequential_reference`
    /// at kiln's exact GDN config (B=1, nv=32, dk=128, dv=128, T=1).
    /// Tolerance matches the chunkwise CUDA kernel test.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_recurrent_kernel_matches_reference() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_gdn_recurrent_kernel_matches_reference"
                );
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let t = 1usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xDECAFBADu64);

        let n_qk = b * nv * t * dk;
        let n_v = b * nv * t * dv;
        let n_b = b * nv * t;
        let n_s = b * nv * dk * dv;

        let q_data: Vec<f32> = (0..n_qk).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let k_data: Vec<f32> = (0..n_qk).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let v_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(0.3f32..1.2f32)).collect();
        // Small negative gates so exp(g) stays in (~0.8, 1.0).
        let g_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(-0.2f32..0.0f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();

        let q_f32 = Tensor::from_slice(&q_data, (b, nv, t, dk), &device)?;
        let k_f32 = Tensor::from_slice(&k_data, (b, nv, t, dk), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, t, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, t), &device)?;
        let g_f32 = Tensor::from_slice(&g_data, (b, nv, t), &device)?;
        let state_f32 = Tensor::from_slice(&s_data, (b, nv, dk, dv), &device)?;

        let q = q_f32.to_dtype(DType::BF16)?;
        let k = k_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;
        let g = g_f32.to_dtype(DType::BF16)?;
        let state_bf16 = state_f32.to_dtype(DType::BF16)?;

        // Reference path: F32 sequential recurrence on the same numerical
        // inputs (cast back to F32 from the bf16 round-trip so the bf16
        // quantization is shared between the two paths and only the kernel
        // arithmetic differs).
        let q_ref = q.to_dtype(DType::F32)?;
        let k_ref = k.to_dtype(DType::F32)?;
        let v_ref = v.to_dtype(DType::F32)?;
        let beta_ref = beta.to_dtype(DType::F32)?;
        let g_ref = g.to_dtype(DType::F32)?;
        let mut state_ref = state_bf16.to_dtype(DType::F32)?;
        let out_ref =
            gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

        // Kernel path: chunkwise dispatcher with seq_len == 1 routes to
        // the new fused recurrent kernel. Make sure no prior test left the
        // kill-switch set in this process.
        // SAFETY: cargo test is single-threaded per test by default and we
        // are only mutating an env var that the dispatcher reads at the top
        // of the same call below. No other thread observes it concurrently.
        unsafe {
            std::env::remove_var("KILN_DISABLE_GDN_KERNEL");
        }
        let backend = crate::backend::for_device(&device);
        let mut state_kernel = state_bf16.clone();
        let out_kernel =
            gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

        let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
        let abs = out_diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
        let s_abs = s_diff.abs()?;
        let s_max = s_abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let s_mean = s_abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
        );

        assert!(
            max < 1e-2,
            "recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );
        assert!(
            s_max < 1e-2,
            "recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
        );
        assert!(
            s_mean < 1e-3,
            "recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
        );

        Ok(())
    }

    /// Parity check for the single-token recurrent Metal kernel.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gdn_recurrent_kernel_matches_reference() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_gdn_recurrent_kernel_matches_reference"
            );
            return Ok(());
        };

        let b = 1usize;
        let nv = 16usize;
        let t = 1usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xBEEFu64);

        let n_qk = b * nv * t * dk;
        let n_v = b * nv * t * dv;
        let n_b = b * nv * t;
        let n_s = b * nv * dk * dv;

        let q_data: Vec<f32> = (0..n_qk).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let k_data: Vec<f32> = (0..n_qk).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let v_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(0.3f32..1.2f32)).collect();
        let g_data: Vec<f32> = (0..n_b).map(|_| rng.gen_range(-0.2f32..0.0f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();

        let q = Tensor::from_slice(&q_data, (b, nv, t, dk), &device)?.to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (b, nv, t, dk), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, t, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, t), &device)?.to_dtype(DType::BF16)?;
        let g = Tensor::from_slice(&g_data, (b, nv, t), &device)?.to_dtype(DType::BF16)?;
        let state_bf16 =
            Tensor::from_slice(&s_data, (b, nv, dk, dv), &device)?.to_dtype(DType::BF16)?;

        let q_ref = q.to_dtype(DType::F32)?;
        let k_ref = k.to_dtype(DType::F32)?;
        let v_ref = v.to_dtype(DType::F32)?;
        let beta_ref = beta.to_dtype(DType::F32)?;
        let g_ref = g.to_dtype(DType::F32)?;
        let mut state_ref = state_bf16.to_dtype(DType::F32)?;
        let out_ref =
            gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

        let backend = crate::backend::for_device(&device);
        if !backend.supports_gdn_recurrent_step() {
            eprintln!("Metal recurrent kernel disabled, skipping parity test");
            return Ok(());
        }
        let mut state_kernel = state_bf16.clone();
        let out_kernel =
            gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

        let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
        let abs = out_diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
        let s_abs = s_diff.abs()?;
        let s_max = s_abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let s_mean = s_abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "metal gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
        );

        assert!(
            max < 1e-2,
            "metal recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "metal recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );
        assert!(
            s_max < 1e-2,
            "metal recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
        );
        assert!(
            s_mean < 1e-3,
            "metal recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
        );

        Ok(())
    }

    /// Parity check for the fused chunk-prep CUDA kernel.
    ///
    /// Generates random bf16 `kkt`, `qkt`, `ks_entry`, `q_s`, `v`, `g` at
    /// kiln's GDN prefill shape (B=1, nv=32, C=64, dv=128), then asserts
    /// that the fused `gdn_chunk_prep` kernel produces the same six
    /// output tensors as the candle-op reference chain it replaces.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_chunk_prep_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_chunk_prep_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xB00B1E5_u64);

        let n_g = b * nv * c;
        let n_v = b * nv * c * dv;
        let n_cc = b * nv * c * c;

        // Small negative gates so big_g stays in a reasonable range — the
        // recurrence produces g_t near zero so the cumulative sum caps
        // around -10 at most.
        let g_data: Vec<f32> = (0..n_g).map(|_| rng.gen_range(-0.15f32..0.0f32)).collect();
        let v_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let kkt_data: Vec<f32> = (0..n_cc).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let qkt_data: Vec<f32> = (0..n_cc).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let ks_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let qs_data: Vec<f32> = (0..n_v).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();

        let g = Tensor::from_slice(&g_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let kkt = Tensor::from_slice(&kkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let qkt = Tensor::from_slice(&qkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let ks_entry =
            Tensor::from_slice(&ks_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s = Tensor::from_slice(&qs_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;

        // Kernel path.
        let (a_strict_k, b_mask_k, v_prime_k, q_s_scaled_k, decay_last_col_k, p_last_k) =
            kiln_gdn_kernel::gdn_chunk_prep(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;

        // Candle reference chain — mirrors the else branch in
        // gdn_chunkwise_recurrence.
        let g_f32 = g.to_dtype(DType::F32)?;
        let big_g = g_f32.cumsum(candle_core::D::Minus1)?; // [B, nv, C] F32
        let big_g_col = big_g.unsqueeze(3)?;
        let big_g_row = big_g.unsqueeze(2)?;
        let decay_f32 = big_g_col.broadcast_sub(&big_g_row)?.exp()?;
        let decay = decay_f32.to_dtype(DType::BF16)?;
        let p = big_g.exp()?.to_dtype(DType::BF16)?;
        let p_col = p.unsqueeze(3)?;

        let strict_mask = strict_lower_tri_mask(c, DType::BF16, &device)?;
        let causal_mask = causal_lower_tri_mask(c, DType::BF16, &device)?;

        let v_prime_ref = (&v - ks_entry.broadcast_mul(&p_col)?)?;
        let a_strict_ref = kkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&strict_mask)?
            .contiguous()?;
        let b_mask_ref = qkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&causal_mask)?
            .contiguous()?;
        let q_s_scaled_ref = q_s.broadcast_mul(&p_col)?;

        let g_last = big_g.narrow(2, c - 1, 1)?; // [B, nv, 1]
        let decay_last_col_ref = g_last.broadcast_sub(&big_g)?.exp()?.to_dtype(DType::BF16)?; // [B, nv, C]
        let p_last_ref = g_last.squeeze(2)?.exp()?.to_dtype(DType::BF16)?; // [B, nv]

        let check = |name: &str, k: &Tensor, r: &Tensor| -> Result<()> {
            let diff = (k.to_dtype(DType::F32)? - r.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!("gdn-chunk-prep {name}: max={max:e} mean={mean:e}");
            assert!(
                max < 1e-2,
                "chunk-prep {name} max_abs_diff {max:e} exceeds 1e-2"
            );
            assert!(
                mean < 1e-3,
                "chunk-prep {name} mean_abs_diff {mean:e} exceeds 1e-3"
            );
            Ok(())
        };

        check("a_strict", &a_strict_k, &a_strict_ref)?;
        check("b_mask", &b_mask_k, &b_mask_ref)?;
        check("v_prime", &v_prime_k, &v_prime_ref)?;
        check("q_s_scaled", &q_s_scaled_k, &q_s_scaled_ref)?;
        check("decay_last_col", &decay_last_col_k, &decay_last_col_ref)?;
        check("p_last", &p_last_k, &p_last_ref)?;

        Ok(())
    }

    /// Parity check for the fused post-prep prefill chunk body.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_chunk_body_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_chunk_body_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xFACE1234_u64);
        let n_cc = b * nv * c * c;
        let n_cdv = b * nv * c * dv;
        let n_c = b * nv * c;

        let a_data: Vec<f32> = (0..n_cc)
            .map(|idx| {
                let t = (idx / c) % c;
                let i = idx % c;
                if i < t {
                    rng.gen_range(-0.15f32..0.15f32)
                } else {
                    0.0
                }
            })
            .collect();
        let b_data: Vec<f32> = (0..n_cc).map(|_| rng.gen_range(-0.2f32..0.2f32)).collect();
        let v_data: Vec<f32> = (0..n_cdv).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let qss_data: Vec<f32> = (0..n_cdv).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let beta_data: Vec<f32> = (0..n_c).map(|_| rng.gen_range(0.3f32..1.1f32)).collect();
        let decay_data: Vec<f32> = (0..n_c).map(|_| rng.gen_range(0.6f32..1.0f32)).collect();

        let a_strict =
            Tensor::from_slice(&a_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let b_mask = Tensor::from_slice(&b_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let v_prime =
            Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s_scaled =
            Tensor::from_slice(&qss_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let decay_last_col =
            Tensor::from_slice(&decay_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;

        let (out_kernel, ww_kernel) = kiln_gdn_kernel::gdn_chunk_scan(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col,
        )?;

        let (out_ref, ww_ref) = compute_chunk_body_reference(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col.unsqueeze(3)?,
        )?;

        let check = |name: &str, got: &Tensor, want: &Tensor| -> Result<()> {
            let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!("gdn-chunk-body {name}: max={max:e} mean={mean:e}");
            assert!(
                max < 2e-2,
                "chunk-body {name} max_abs_diff {max:e} exceeds 2e-2"
            );
            assert!(
                mean < 2e-3,
                "chunk-body {name} mean_abs_diff {mean:e} exceeds 2e-3"
            );
            Ok(())
        };

        check("out_chunk", &out_kernel, &out_ref)?;
        check("w_weighted", &ww_kernel, &ww_ref)?;
        Ok(())
    }

    /// Parity check for the fused full-chunk CUDA prefill path.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_full_chunk_forward_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_gdn_full_chunk_forward_matches_fallback"
                );
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0x5EED_CAFE_u64);
        let n_c = b * nv * c;
        let n_cdv = b * nv * c * dv;
        let n_cc = b * nv * c * c;
        let n_dkc = b * nv * dk * c;
        let n_dkdv = b * nv * dk * dv;

        let g_data: Vec<f32> = (0..n_c).map(|_| rng.gen_range(-0.15f32..0.0f32)).collect();
        let v_data: Vec<f32> = (0..n_cdv).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let kkt_data: Vec<f32> = (0..n_cc).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let qkt_data: Vec<f32> = (0..n_cc).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let ks_data: Vec<f32> = (0..n_cdv).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let qs_data: Vec<f32> = (0..n_cdv).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let beta_data: Vec<f32> = (0..n_c).map(|_| rng.gen_range(0.3f32..1.1f32)).collect();
        let kt_data: Vec<f32> = (0..n_dkc).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let state_data: Vec<f32> = (0..n_dkdv)
            .map(|_| rng.gen_range(-0.25f32..0.25f32))
            .collect();

        let g = Tensor::from_slice(&g_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let kkt = Tensor::from_slice(&kkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let qkt = Tensor::from_slice(&qkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let ks_entry =
            Tensor::from_slice(&ks_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s = Tensor::from_slice(&qs_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let k_t = Tensor::from_slice(&kt_data, (b, nv, dk, c), &device)?.to_dtype(DType::BF16)?;
        let mut state_kernel =
            Tensor::from_slice(&state_data, (b, nv, dk, dv), &device)?.to_dtype(DType::BF16)?;
        let state_ref = state_kernel.clone();

        let out_kernel = kiln_gdn_kernel::gdn_full_chunk_forward(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &mut state_kernel,
        )?;

        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
            kiln_gdn_kernel::gdn_chunk_prep(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;
        let (out_ref, ww_ref) = compute_chunk_body_reference(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col.unsqueeze(3)?,
        )?;
        let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?;
        let state_expected =
            (state_ref.broadcast_mul(&p_last_u)? + k_t.matmul(&ww_ref)?)?.contiguous()?;

        let check =
            |name: &str, got: &Tensor, want: &Tensor, max_tol: f32, mean_tol: f32| -> Result<()> {
                let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?;
                let abs = diff.abs()?;
                let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
                eprintln!("gdn-full-chunk {name}: max={max:e} mean={mean:e}");
                assert!(
                    max < max_tol,
                    "full-chunk {name} max_abs_diff {max:e} exceeds {max_tol:e}"
                );
                assert!(
                    mean < mean_tol,
                    "full-chunk {name} mean_abs_diff {mean:e} exceeds {mean_tol:e}"
                );
                Ok(())
            };

        check("out_chunk", &out_kernel, &out_ref, 2e-2, 2e-3)?;
        check("state", &state_kernel, &state_expected, 3.5e-2, 4e-3)?;
        Ok(())
    }

    /// Parity check for the fused causal_conv1d_update kernel against the
    /// portable `causal_conv1d_decode` + `cuda_silu` chain, at Qwen3.5-4B's
    /// exact decode shape: B=1, C=linear_qkv_dim=8192, K=4.
    ///
    /// Verifies (a) the silu-fused F32 output matches within bf16-rounding
    /// noise and (b) the mutated conv_state matches bit-for-bit (both paths
    /// write the same K-1 previous inputs from the same bf16 source).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_causal_conv1d_update_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_causal_conv1d_update_matches_fallback"
                );
                return Ok(());
            }
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
        let n_x = batch * channels * 1;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let w_data: Vec<f32> = (0..n_w).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.3f32..0.3f32)).collect();

        let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1), &device)?;
        let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let x = x_f32.to_dtype(DType::BF16)?;
        let w = w_f32.to_dtype(DType::BF16)?;

        // Fallback path: candle decode + silu in F32.
        let mut s_fb = s_init.clone();
        let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
        let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

        // Fused kernel path via the backend dispatch.
        let backend = crate::backend::for_device(&device);
        if !backend.supports_causal_conv1d_update() {
            eprintln!(
                "backend declines causal_conv1d_update (KILN_DISABLE_FUSED_CONV1D?); skipping"
            );
            return Ok(());
        }
        let mut s_k = s_init.clone();
        let out_k = match backend.causal_conv1d_update(&x, &w, &mut s_k, kernel_size)? {
            Some(t) => t,
            None => {
                eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
                return Ok(());
            }
        };

        // Output parity (silu fused on the kernel side).
        let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-3,
            "fused conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
        );
        assert!(
            mean < 5e-4,
            "fused conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
        );

        // State parity — both paths write the same K-1 previous inputs.
        let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_update state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-5,
            "fused conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
        );

        Ok(())
    }

    /// Metal parity check for `backend.causal_conv1d_update` against the same
    /// portable `causal_conv1d_decode` + `cuda_silu` oracle used by CUDA.
    #[cfg(feature = "metal")]
    #[test]
    fn test_causal_conv1d_update_matches_fallback_metal() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping test_causal_conv1d_update_matches_fallback_metal"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
        let n_x = batch * channels;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let w_data: Vec<f32> = (0..n_w).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.3f32..0.3f32)).collect();

        let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1), &device)?;
        let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let x = x_f32.to_dtype(DType::BF16)?;
        let w = w_f32.to_dtype(DType::BF16)?;

        let mut s_fb = s_init.clone();
        let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
        let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

        let backend = crate::backend::for_device(&device);
        if !backend.supports_causal_conv1d_update() {
            eprintln!(
                "backend declines causal_conv1d_update (KILN_DISABLE_FUSED_CONV1D?); skipping"
            );
            return Ok(());
        }
        let mut s_k = s_init.clone();
        let out_k = match backend.causal_conv1d_update(&x, &w, &mut s_k, kernel_size)? {
            Some(t) => t,
            None => {
                eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
                return Ok(());
            }
        };

        let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-3,
            "metal conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
        );
        assert!(
            mean < 5e-4,
            "metal conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
        );

        let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_update state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-5,
            "metal conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_causal_conv1d_prefill_bf16_parity_on_metal() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_causal_conv1d_prefill_bf16_parity_on_metal"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let seq_len = 16usize;
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xBF16_C0DE);
        let n_x = batch * channels * seq_len;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let w_data: Vec<f32> = (0..n_w).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.3f32..0.3f32)).collect();

        let x = Tensor::from_slice(&x_data, (batch, channels, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let mut s_ref = s_init.clone();
        let out_ref =
            causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
        let out_ref = cuda_silu(&out_ref)?;

        let mut s_bf16 = s_init.clone();
        assert_eq!(
            causal_conv1d_prefill_compute_dtype(&x, &w, &s_bf16, kernel_size),
            DType::BF16
        );
        let out_bf16 = causal_conv1d_prefill(&x, &w, &mut s_bf16, kernel_size)?;
        assert_eq!(out_bf16.dtype(), DType::BF16);
        assert_eq!(s_bf16.dtype(), DType::F32);
        let out_bf16 = cuda_silu(&out_bf16)?;

        let diff = (out_bf16.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill bf16 vs f32: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-2,
            "bf16 prefill output max_abs_diff={max:e} exceeds 2e-2"
        );
        assert!(
            mean < 2e-3,
            "bf16 prefill output mean_abs_diff={mean:e} exceeds 2e-3"
        );

        let sdiff = (s_bf16.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill bf16 state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-6,
            "bf16 prefill state max_abs_diff={smax:e} exceeds 1e-6"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_causal_conv1d_prefill_kernel_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_causal_conv1d_prefill_kernel_matches_fallback"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let seq_len = 16usize;
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE_8175);
        let n_x = batch * channels * seq_len;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x).map(|_| rng.gen_range(-0.5f32..0.5f32)).collect();
        let w_data: Vec<f32> = (0..n_w).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect();
        let s_data: Vec<f32> = (0..n_s).map(|_| rng.gen_range(-0.3f32..0.3f32)).collect();

        let x = Tensor::from_slice(&x_data, (batch, channels, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let mut s_ref = s_init.clone();
        let out_ref =
            causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
        let out_ref = cuda_silu(&out_ref)?;

        let backend = crate::backend::for_device(&device);
        assert!(backend.supports_causal_conv1d_prefill());
        let mut s_kernel = s_init.clone();
        let out_kernel = match backend.causal_conv1d_prefill(&x, &w, &mut s_kernel, kernel_size)? {
            Some(out) => out,
            None => {
                eprintln!("Metal backend declined causal_conv1d_prefill; skipping");
                return Ok(());
            }
        };
        assert_eq!(out_kernel.dtype(), DType::F32);
        assert_eq!(s_kernel.dtype(), DType::F32);

        let diff = (out_kernel.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!(
            "metal conv1d_prefill kernel vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max < 1e-5,
            "metal prefill output max_abs_diff={max:e} exceeds 1e-5"
        );
        assert!(
            mean < 1e-6,
            "metal prefill output mean_abs_diff={mean:e} exceeds 1e-6"
        );

        let sdiff = (s_kernel.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_prefill kernel state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-6,
            "metal prefill state max_abs_diff={smax:e} exceeds 1e-6"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase 7: streaming/tiled GDN prefill — CPU parity tests.
    //
    // Each test compares the monolithic `model_forward_paged` against
    // `model_forward_paged_streaming_with` running multiple tiles. Both runs
    // start from fresh `LinearAttentionState` + `PagedKvCache` so the
    // recurrent state hand-off and per-tile paged writes are exercised end
    // to end. Tests use `last_token_only=false` so we can compare the full
    // last-tile logits row-by-row against the matching slice of the
    // monolithic logits.
    // -----------------------------------------------------------------------

    /// Shared config for all streaming parity tests. Picks a hybrid layer
    /// stack (3 GDN + 1 full attention with `full_attention_interval=4`,
    /// scaled to 8 layers so we get 6 GDN layers exercising the recurrent
    /// hand-off across tile boundaries).
    fn streaming_test_config() -> kiln_core::config::ModelConfig {
        let num_layers = 8;
        let full_attention_interval = 4; // layers 3, 7 are full → 2 full + 6 linear
        kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 2,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        }
    }

    /// Build a paged cache + sequential block table sized for `seq_len` tokens
    /// with `block_size`-token blocks (block_size = GDN_CHUNK_SIZE so block
    /// boundaries coincide with the smallest legal tile boundary).
    fn make_paged_setup(
        config: &kiln_core::config::ModelConfig,
        seq_len: usize,
        block_size: usize,
        device: &Device,
    ) -> Result<(PagedKvCache, BlockTable)> {
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            DType::F32,
            device,
        )?;
        let mut block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            block_table.push(i);
        }
        Ok((cache, block_table))
    }

    /// Deterministic token sequence for parity testing. Stays inside vocab.
    fn deterministic_tokens(seq_len: usize, vocab_size: u32) -> Vec<u32> {
        (0..seq_len)
            .map(|i| ((i as u32 * 13 + 7) % vocab_size).max(1))
            .collect()
    }

    /// Run monolithic vs streaming on the same config + tokens, return
    /// `(monolithic_full_logits[1, T, V], streaming_full_last_tile_logits[1, last_tile_len, V])`
    /// where the streaming pass uses `tile_size` and `last_token_only=false`.
    fn run_parity(
        config: &kiln_core::config::ModelConfig,
        token_ids: &[u32],
        tile_size: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        // Monolithic: single forward pass, full LM head.
        let (mut mono_cache, mono_bt) =
            make_paged_setup(config, token_ids.len(), block_size, &device)?;
        let mut mono_state = LinearAttentionState::new(config, &device)?;
        let mono_logits = model_forward_paged(
            &backend,
            token_ids,
            &weights,
            config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming: tiled prefill with last_token_only=false so the final
        // tile produces a full per-position logits slice we can compare
        // against the matching window of the monolithic output.
        let (mut stream_cache, stream_bt) =
            make_paged_setup(config, token_ids.len(), block_size, &device)?;
        let mut stream_state = LinearAttentionState::new(config, &device)?;
        let stream_logits = model_forward_paged_streaming_with(
            &backend,
            token_ids,
            &weights,
            config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile_size,
            false,
        )?;

        Ok((mono_logits, stream_logits))
    }

    /// Compare the streaming last-tile full logits against the matching
    /// slice of the monolithic logits.
    fn assert_last_tile_matches(
        mono_logits: &Tensor,
        stream_logits: &Tensor,
        total_len: usize,
        tile_size: usize,
        tol: f32,
    ) -> Result<()> {
        // Last tile spans [last_start, total_len).
        let last_start = total_len - ((total_len - 1) % tile_size + 1);
        let last_len = total_len - last_start;
        let mono_slice = mono_logits
            .narrow(1, last_start, last_len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let stream_slice = stream_logits.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            mono_slice.len(),
            stream_slice.len(),
            "last tile length mismatch"
        );
        let mut max_abs = 0f32;
        for (a, b) in mono_slice.iter().zip(stream_slice.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs <= tol,
            "streaming vs monolithic max_abs_diff={max_abs:e} exceeds {tol:e}"
        );
        Ok(())
    }

    #[test]
    fn test_streaming_matches_monolithic_cpu_small() -> Result<()> {
        let config = streaming_test_config();
        let total = 128;
        let tile = 64;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
        assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
        assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_streaming_last_hidden_matches_monolithic_cpu() -> Result<()> {
        let config = streaming_test_config();
        let device = Device::Cpu;
        let total = GDN_CHUNK_SIZE * 2 + 7;
        let tile = GDN_CHUNK_SIZE;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let (mono_logits, mono_hidden) = model_forward_paged_last_token_with_last_hidden(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let (stream_logits, stream_hidden) =
            model_forward_paged_streaming_last_token_with_last_hidden_with(
                &backend,
                &tokens,
                &weights,
                &config,
                &mut stream_cache,
                &stream_bt,
                0,
                Some(&mut stream_state),
                None,
                tile,
            )?;

        assert_eq!(stream_logits.dims(), &[1, 1, config.vocab_size]);
        assert_eq!(stream_hidden.dims(), &[1, 1, config.hidden_size]);
        let logits_diff = (&mono_logits - &stream_logits)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let hidden_diff = (&mono_hidden - &stream_hidden)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            logits_diff <= 1e-5,
            "streaming MTP prefill logits drifted: max_abs_diff={logits_diff:e}"
        );
        assert!(
            hidden_diff <= 1e-5,
            "streaming MTP prefill h_prev drifted: max_abs_diff={hidden_diff:e}"
        );
        Ok(())
    }

    #[test]
    fn test_streaming_matches_monolithic_cpu_mid() -> Result<()> {
        let config = streaming_test_config();
        let total = 512;
        let tile = 128;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
        assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
        assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_streaming_tile_invariance_cpu() -> Result<()> {
        // For a fixed token sequence, the last token's logits must agree
        // across every legal tile size (multiples of GDN_CHUNK_SIZE that
        // divide or partition `total`). The monolithic run is the reference;
        // every tile size collapses to the same final-token logits.
        let config = streaming_test_config();
        let total = 256;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);

        // Monolithic reference: take the last row of [1, total, V] logits.
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_logits = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;
        let reference_last = mono_logits
            .narrow(1, total - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for tile in [64usize, 128, 256] {
            let (mut cache, bt) = make_paged_setup(&config, total, 64, &device)?;
            let mut state = LinearAttentionState::new(&config, &device)?;
            let logits = model_forward_paged_streaming_with(
                &backend,
                &tokens,
                &weights,
                &config,
                &mut cache,
                &bt,
                0,
                Some(&mut state),
                None,
                tile,
                true, // last_token_only — matches production dispatch
            )?;
            assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
            let last = logits.flatten_all()?.to_vec1::<f32>()?;
            let max_abs = reference_last
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-5,
                "tile={tile} last-token max_abs_diff={max_abs:e} exceeds 1e-5"
            );
        }
        Ok(())
    }

    #[test]
    fn test_model_forward_paged_last_token_matches_full_last_row_cpu() -> Result<()> {
        let config = streaming_test_config();
        let total = 128;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        let (mut full_cache, full_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut full_state = LinearAttentionState::new(&config, &device)?;
        let full_logits = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut full_cache,
            &full_bt,
            0,
            Some(&mut full_state),
            None,
            None,
        )?;
        let reference_last = full_logits
            .narrow(1, total - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let (mut last_cache, last_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut last_state = LinearAttentionState::new(&config, &device)?;
        let last_logits = model_forward_paged_last_token(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut last_cache,
            &last_bt,
            0,
            Some(&mut last_state),
            None,
            None,
        )?;
        assert_eq!(last_logits.dims(), &[1, 1, config.vocab_size]);
        let last = last_logits.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = reference_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "last-token prefill max_abs_diff={max_abs:e} exceeds 1e-5"
        );

        Ok(())
    }

    #[test]
    fn test_streaming_preserves_state_cpu() -> Result<()> {
        // After prefill, run a single decode step on top of the resulting
        // (paged_cache, linear_state). If state was preserved bit-exact
        // across tile boundaries, the decode-token logits must agree with
        // the monolithic reference.
        let config = streaming_test_config();
        let total = 192;
        let tile = 64;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let next_token: u32 = 11;

        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        // Monolithic prefill, then 1 decode step.
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let _ = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;
        let mono_decode = model_forward_paged(
            &backend,
            &[next_token],
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            total,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming prefill, then 1 decode step.
        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let _ = model_forward_paged_streaming_with(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile,
            true,
        )?;
        let stream_decode = model_forward_paged(
            &backend,
            &[next_token],
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            total,
            Some(&mut stream_state),
            None,
            None,
        )?;

        let a = mono_decode.flatten_all()?.to_vec1::<f32>()?;
        let b = stream_decode.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len());
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "decode-after-streaming max_abs_diff={max_abs:e} exceeds 1e-5 \
             (state was not bit-exact preserved across tile boundaries)"
        );
        Ok(())
    }

    /// CUDA parity for streaming/tiled GDN prefill.
    ///
    /// Mirrors `test_streaming_matches_monolithic_cpu_mid` but on CUDA at
    /// T=2048, tile=512 (the configuration the Phase 7 GPU spike validates).
    /// Asserts (1) full-tile logits match the matching slice of the
    /// monolithic logits, and (2) `LinearAttentionState.recurrent_states[l]`
    /// and `state.conv_states[l]` are equal across the two paths after
    /// prefill — the state hand-off is the load-bearing part of streaming.
    ///
    /// Tolerance: 1e-4. The design doc (PROFILING.md §c "CUDA parity")
    /// argues bit-exactness is achievable because GDN recurrent state stays
    /// in F32 and the conv1d F32 promotion makes the conv path
    /// deterministic. In practice, candle CUDA matmul reduction order can
    /// vary with shape, so we use a small FP32 tolerance rather than
    /// strict equality.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_streaming_matches_monolithic_cuda() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_streaming_matches_monolithic_cuda");
                return Ok(());
            }
        };

        let config = streaming_test_config();
        let total = 2048usize;
        let tile = 512usize;
        let block_size = 64usize; // == GDN_CHUNK_SIZE
        let tokens = deterministic_tokens(total, config.vocab_size as u32);

        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = crate::backend::for_device(&device);

        // Monolithic: single forward pass, full LM head.
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, block_size, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_logits = model_forward_paged(
            &*backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming: tiled prefill, last_token_only=false so we get a full
        // last-tile logits slice for row-by-row comparison.
        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, block_size, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let stream_logits = model_forward_paged_streaming_with(
            &*backend,
            &tokens,
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile,
            false,
        )?;

        assert_eq!(mono_logits.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream_logits.dims(), &[1, tile, config.vocab_size]);

        // (1) Last-tile logits parity.
        assert_last_tile_matches(&mono_logits, &stream_logits, total, tile, 1e-4)?;

        // (2) Per-layer state parity (recurrent + conv).
        assert_eq!(
            mono_state.recurrent_states.len(),
            stream_state.recurrent_states.len(),
            "recurrent_states layer count mismatch"
        );
        assert_eq!(
            mono_state.conv_states.len(),
            stream_state.conv_states.len(),
            "conv_states layer count mismatch"
        );
        for (l, (m, s)) in mono_state
            .recurrent_states
            .iter()
            .zip(stream_state.recurrent_states.iter())
            .enumerate()
        {
            let m_v = m.flatten_all()?.to_vec1::<f32>()?;
            let s_v = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                m_v.len(),
                s_v.len(),
                "recurrent_states[{l}] length mismatch"
            );
            let max_abs = m_v
                .iter()
                .zip(s_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "recurrent_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
            );
        }
        for (l, (m, s)) in mono_state
            .conv_states
            .iter()
            .zip(stream_state.conv_states.iter())
            .enumerate()
        {
            let m_v = m.flatten_all()?.to_vec1::<f32>()?;
            let s_v = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(m_v.len(), s_v.len(), "conv_states[{l}] length mismatch");
            let max_abs = m_v
                .iter()
                .zip(s_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "conv_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
            );
        }

        Ok(())
    }

    #[test]
    fn test_streaming_prefill_env_helpers() {
        // Each nextest test runs in its own process, so env-var manipulation
        // here is safe. We verify the dispatch helpers return what
        // `model_forward_paged_streaming` reads from the environment.
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD");
        }
        assert!(!streaming_prefill_enabled(), "default must be disabled");
        assert!(!streaming_prefill_enabled_for(
            &Device::Cpu,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));
        assert_eq!(streaming_tile_tokens(), STREAMING_PREFILL_DEFAULT_TILE);
        assert_eq!(
            streaming_tile_tokens_for(&Device::Cpu),
            STREAMING_PREFILL_DEFAULT_TILE
        );
        assert!(streaming_last_token_lm_head(), "default must be true");

        #[cfg(feature = "metal")]
        if let Some(device) = crate::backend::metal::try_new_metal() {
            assert!(!streaming_prefill_enabled_for(
                &device,
                STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD - 1
            ));
            assert!(streaming_prefill_enabled_for(
                &device,
                STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
            ));
            assert_eq!(
                streaming_tile_tokens_for(&device),
                STREAMING_PREFILL_METAL_DEFAULT_TILE
            );
        }

        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
        }
        assert!(streaming_prefill_enabled());
        assert!(streaming_prefill_enabled_for(&Device::Cpu, 1));

        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "0");
        }
        assert!(!streaming_prefill_enabled());
        assert!(!streaming_prefill_enabled_for(
            &Device::Cpu,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));

        unsafe {
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "256");
        }
        assert_eq!(streaming_tile_tokens(), 256);
        assert_eq!(streaming_tile_tokens_for(&Device::Cpu), 256);

        // Bad value (not a multiple of GDN_CHUNK_SIZE) falls back to default.
        unsafe {
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "65");
        }
        assert_eq!(streaming_tile_tokens(), STREAMING_PREFILL_DEFAULT_TILE);

        unsafe {
            std::env::set_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD", "0");
        }
        assert!(!streaming_last_token_lm_head());

        // Cleanup so this test does not leak state to peers (defensive even
        // though nextest isolates by process).
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD");
        }
    }
}

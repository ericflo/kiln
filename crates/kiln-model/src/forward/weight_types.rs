use super::*;

/// GPU-ready tensors organized by layer, converted from raw `ModelWeights` bytes.
///
/// `Clone` is cheap: every field bottoms out in `Tensor`, which is `Arc`-backed,
/// so a clone bumps refcounts rather than copying device memory. Used by the
/// in-process self-distillation teacher (`LiveLocalTeacher`) to hold a shared
/// handle to the loaded model and score rollouts on demand.
#[derive(Clone)]
pub struct GpuWeights {
    /// Content revision of the exact base-model shard bytes that produced
    /// these tensors. Production loaders always populate it; synthetic test
    /// weights may leave it absent. Exact training checkpoints bind to this
    /// identity so shape-compatible model replacements cannot resume silently.
    pub source_content_sha256: Option<String>,
    /// Canonical per-shard provenance retained from the same loader hash pass
    /// that produced `source_content_sha256`.
    pub base_weight_shard_manifest: Option<BaseWeightShardManifest>,
    /// Process-lifetime execution envelope stamped after backend startup.
    /// Synthetic weights may leave it absent.
    pub execution_provenance: Option<ExecutionProvenanceV1>,
    /// Token embedding table: [vocab_size, hidden_size]
    pub embed_tokens: Tensor,
    /// Pre-transposed token embedding table for tied LM head: [hidden_size, vocab_size], contiguous.
    /// Computed once at load to avoid re-transposing the ~778 MiB bf16 matrix on every decode step
    /// (was 48% of ucopy_bf16 / ~43% of GPU time per PR #113 profile).
    pub embed_tokens_t: Tensor,
    /// Optional ROCm-only row-wise W8A16 tied LM head. Stores the original
    /// embedding rows `[vocab_size, hidden_size]` for fused greedy argmax.
    pub lm_head_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
    /// Per-layer weights
    pub layers: Vec<GpuLayerWeights>,
    /// Final RMSNorm weight: [hidden_size]
    pub final_norm: Tensor,
    /// Cached rotary inv_freq tensor, shape `[half_rotary]`, F32 on device.
    /// Computed once at load time from `config.rotary_dim()` and `config.rope_theta`
    /// so the RoPE hot path can reuse it instead of rebuilding a fresh `Vec<f32>` +
    /// HtoD upload on every layer's attention call (~8 × per token in prefill).
    pub rotary_inv_freq: Tensor,
    /// Native MTP (Multi-Token Prediction) head tensors, when a model-library
    /// caller requests them and the checkpoint supplies `ModelWeights.mtp`.
    ///
    /// The serving binary defers MTP loading and rejects every enabled
    /// speculative method at startup. A deferred checkpoint source therefore
    /// still produces a lazy slot here; explicit offline qualification and
    /// model-library callers may materialize it later. `None` means the
    /// checkpoint does not expose an MTP source.
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
/// Explicit model-library and offline qualification callers pay the upload cost
/// on their first MTP forward. Serving retains this latent slot but neither
/// routes inference through it nor permits server SFT to train it while the MTP
/// ownership and memory contracts remain unqualified.
pub struct MtpGpuWeightsSlot {
    weights: OnceLock<MtpGpuWeights>,
    source: Option<MtpGpuSource>,
    device: Device,
    init_lock: Mutex<()>,
}

impl Clone for MtpGpuWeightsSlot {
    /// The slot lazily derives its GPU weights from `source` on first use, so a
    /// clone carries `source` + `device` and starts with a FRESH (empty) cache
    /// that re-derives on demand. (The self-distillation teacher that drives
    /// `GpuWeights: Clone` never touches MTP, so this never re-uploads there.)
    fn clone(&self) -> Self {
        Self {
            weights: OnceLock::new(),
            source: self.source.clone(),
            device: self.device.clone(),
            init_lock: Mutex::new(()),
        }
    }
}

#[derive(Clone)]
pub(super) enum MtpGpuSource {
    Loaded(MtpWeights),
    Deferred(DeferredMtpSource),
}

impl MtpGpuWeightsSlot {
    pub fn lazy(source: MtpWeights, device: &Device) -> Self {
        Self {
            weights: OnceLock::new(),
            source: Some(MtpGpuSource::Loaded(source)),
            device: device.clone(),
            init_lock: Mutex::new(()),
        }
    }

    pub fn lazy_deferred(source: DeferredMtpSource, device: &Device) -> Self {
        Self {
            weights: OnceLock::new(),
            source: Some(MtpGpuSource::Deferred(source)),
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

    /// Training-session residency companion to
    /// [`GpuWeights::to_device_deep`]. An uploaded slot deep-copies its
    /// tensors onto `device` (eager); a lazy slot keeps its CPU source and
    /// re-targets the device.
    pub fn to_device_deep(
        &self,
        device: Device,
        mv_layer: &dyn Fn(&GpuLayerWeights) -> Result<GpuLayerWeights>,
    ) -> Result<MtpGpuWeightsSlot> {
        if let Some(mtp) = self.weights.get() {
            let mv = |t: &Tensor| -> Result<Tensor> {
                t.to_device(device)
                    .map_err(|e| anyhow::anyhow!("mtp to_device_deep: {e}"))
            };
            let moved = MtpGpuWeights {
                fc: mv(&mtp.fc)?,
                fc_t: mv(&mtp.fc_t)?,
                pre_fc_norm_embedding: mv(&mtp.pre_fc_norm_embedding)?,
                pre_fc_norm_hidden: mv(&mtp.pre_fc_norm_hidden)?,
                layer: mv_layer(&mtp.layer)?,
                final_layernorm: mv(&mtp.final_layernorm)?,
            };
            return Ok(MtpGpuWeightsSlot::eager(moved, &device));
        }
        Ok(MtpGpuWeightsSlot {
            weights: OnceLock::new(),
            source: self.source.clone(),
            device,
            init_lock: Mutex::new(()),
        })
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
        let mtp_weights = match source {
            MtpGpuSource::Loaded(weights) => weights.clone(),
            MtpGpuSource::Deferred(source) => {
                let load_start = std::time::Instant::now();
                let loaded = crate::loader::load_deferred_mtp(source)
                    .context("deferred native MTP CPU load")?;
                tracing::info!(
                    load_elapsed_ms = load_start.elapsed().as_millis() as u64,
                    "deferred native MTP CPU load complete"
                );
                loaded
            }
        };
        // Explicit model-library and qualification callers may trigger this
        // lazy upload on their first MTP forward. Cache misses must remain
        // read-only; serving does not route requests through this path.
        let projection_load_cache = ProjectionLoadCache::for_lazy_mtp_upload(&self.device)
            .context("mtp projection load cache")?;
        let upload_start = std::time::Instant::now();
        let uploaded = upload_mtp_gpu_weights(&mtp_weights, &self.device, &projection_load_cache)
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

pub(super) fn upload_mtp_gpu_weights(
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
    // is currently an explicit offline/model-library path, not a serving route.
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
                    qkv_proj_t: None,
                    o_proj_t,
                    qkv_proj_w8: None,
                    o_proj_w8: None,
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
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
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

/// The kt `Device` the loader places its host-resident kt tensors on,
/// derived from the model's logical device. (#1082 Vulkan loader fix.)
///
/// On Vulkan the kt-tensor *substrate* storage is not yet wired
/// (`Tensor::zeros_on(Vulkan)` etc. deliberately `Err` to surface
/// accidental routing — see `kiln_tensor::Tensor::zeros_on`), and the
/// production Vulkan inference path keeps every weight as a **CPU-host**
/// kt tensor that the `vk::Device` backend uploads into its own buffer
/// cache lazily. So loader-internal kt constructors (stubs, the rotary
/// `inv_freq` table) must land on `Device::Cpu` on Vulkan rather than on
/// the logical `Device::Vulkan(_)`. CUDA / Metal / CPU pass through
/// unchanged. Revisit when Phase 1.8 lands real kt Vulkan storage.
#[inline]
pub(super) fn loader_kt_device(device: &Device) -> Device {
    match device {
        Device::Vulkan(_) => Device::Cpu,
        d => *d,
    }
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
    // kt `new` takes `Device` by value (kt Device is Copy) (#1082).
    let t = Tensor::new(inv_freq.as_slice(), loader_kt_device(device))
        .context("failed to build rotary inv_freq tensor")?;
    Ok(t)
}

/// One transformer layer's tensors on device.
#[derive(Clone)]
pub struct GpuLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub attention: GpuAttentionWeights,
    pub mlp: GpuFfnWeights,
}

/// Attention weights on device.
#[derive(Clone)]
pub enum GpuAttentionWeights {
    Full(GpuFullAttentionWeights),
    Linear(GpuLinearAttentionWeights),
}

#[derive(Clone)]
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
    /// Optional cached `[hidden, q_raw + k + v]` transpose for CUDA decode.
    /// This combines the full-attention Q/K/V projections into one matmul on
    /// forward-only single-token fast paths without disturbing the separate
    /// transposes used by training, LoRA, Marlin, and debug captures.
    pub qkv_proj_t: Option<Tensor>,
    pub o_proj_t: Tensor,
    /// Optional ROCm-only row-wise W8A16 full-attention decode projections.
    /// `qkv_proj_w8` stores `[q_raw | k | v]` rows; `o_proj_w8` stores
    /// `[hidden, num_heads * head_dim]`. Populated only when
    /// Enabled by the qualified immutable ROCm kernel profile for decode.
    pub qkv_proj_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
    pub o_proj_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
    /// Optional Marlin W4A16-packed q_proj. Populated when the installed CUDA
    /// Marlin profile includes attention Q and the shape fits Marlin's tile
    /// constraints (k%128 && n%256). When present, the forward
    /// path routes q_proj through the Marlin kernel instead of the BF16
    /// `broadcast_matmul` via `q_proj_t`. LoRA deltas are still applied on top.
    pub q_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

impl GpuFullAttentionWeights {
    /// kt-native view of the pre-transposed full-attention q projection
    /// (#1082, GQA full-attention region migration — region 3).
    ///
    /// Returns `q_proj_t` (`[hidden, num_heads * head_dim (+gate)]`,
    /// contiguous since load) as a contiguous `KtTensor`. The tensor is already
    /// kt-native, so backend eligibility belongs to the call site's request
    /// dispatch instead of this accessor.
    pub fn q_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.q_proj_t, "q_proj_t_kt")
    }

    /// kt-native view of the pre-transposed full-attention k projection
    /// (#1082, region 3). Same kt-native contiguity contract as
    /// [`GpuFullAttentionWeights::q_proj_t_kt`].
    pub fn k_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.k_proj_t, "k_proj_t_kt")
    }

    /// kt-native view of the pre-transposed full-attention v projection
    /// (#1082, region 3). Same kt-native contiguity contract as
    /// [`GpuFullAttentionWeights::q_proj_t_kt`].
    pub fn v_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.v_proj_t, "v_proj_t_kt")
    }

    /// kt-native view of the pre-transposed full-attention output
    /// projection (#1082, region 3). Shape `[num_heads * head_dim, hidden]`.
    /// Same kt-native contiguity contract as
    /// [`GpuFullAttentionWeights::q_proj_t_kt`]; provided for parity with the
    /// region-1/2 accessors and future consolidation of the o_proj matmul.
    pub fn o_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.o_proj_t, "o_proj_t_kt")
    }

    /// kt-native view of the q_norm RMSNorm weight (#1082, region 3).
    /// Shape `[head_dim]`. Same kt-native contiguity contract as
    /// [`GpuFullAttentionWeights::q_proj_t_kt`]; provided for parity with the
    /// region-1/2 accessors (the q/k norm already routes through the
    /// default-on `rms_norm` kt gate inside the prepare step, so the SDPA
    /// matmul helper does not need this — it is kept for the future
    /// consolidation of the prepare region).
    pub fn q_norm_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.q_norm, "q_norm_kt")
    }

    /// kt-native view of the k_norm RMSNorm weight (#1082, region 3).
    /// Shape `[head_dim]`. Same contract as
    /// [`GpuFullAttentionWeights::q_norm_kt`].
    pub fn k_norm_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.k_norm, "k_norm_kt")
    }
}

#[derive(Clone)]
pub struct GpuLinearAttentionWeights {
    pub in_proj_qkv: Tensor,
    pub in_proj_z: Tensor,
    pub out_proj: Tensor,
    pub in_proj_a: Tensor,
    pub in_proj_b: Tensor,
    pub conv1d: Tensor,
    pub norm: Tensor,
    pub a_log: Tensor,
    pub a_log_gates: Tensor,
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
    /// Optional cached `[hidden, 2 * nv]` transpose that combines the small
    /// prefill/decode A/B projections into one matmul on backend fast paths.
    pub in_proj_ab_t: Option<Tensor>,
    pub out_proj_t: Tensor,
    /// Optional Marlin W4A16-packed GDN out_proj. Populated when the expanded
    /// CUDA Marlin profile includes it and its shape fits Marlin's tile
    /// constraints (`k%128 && n%256`).
    /// When present, the GDN forward path uses Marlin for the projection
    /// instead of `broadcast_matmul` via `out_proj_t`. The GDN out_proj is
    /// the last linear layer in the GDN block before the
    /// residual add — int4 quantization there is more sensitive to
    /// quality drift than the in-projections or the MLP, so deployments
    /// opt in only after their own quality A/B passes.
    pub out_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    /// Optional ROCm-only row-wise W8A16 fused input projection storing
    /// `[qkv | z | a | b]` rows. Enabled by default on ROCm decode; set
    /// The immutable ROCm kernel profile owns packing and dispatch.
    pub in_proj_qkvzab_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
}

impl GpuLinearAttentionWeights {
    /// kt-native view of the fused GDN input projection (`in_proj_qkv`,
    /// the row-major `[out, hidden]` load-time weight) (#1082, GDN
    /// linear-attention region migration).
    ///
    /// Returns the contiguous kt field as a `KtTensor`, mirroring
    /// [`GpuFullAttentionWeights::q_proj_t_kt`] and
    /// [`GpuFfnWeights::gate_proj_t_kt`]. Provided so the GDN region's
    /// kt weight boundary lives in one place. The tensor is already kt-native;
    /// backend eligibility belongs to the request-dispatch call site. The hot
    /// path uses the pre-transposed `*_t` variants.
    pub fn in_proj_qkv_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_qkv, "in_proj_qkv_kt")
    }

    /// kt-native view of the GDN `in_proj_z` (gate) projection weight
    /// (#1082, GDN region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn in_proj_z_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_z, "in_proj_z_kt")
    }

    /// kt-native view of the GDN output projection weight (`out_proj`,
    /// the row-major `[hidden, value_dim]` load-time weight) (#1082, GDN
    /// region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn out_proj_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.out_proj, "out_proj_kt")
    }

    /// kt-native view of the GDN `in_proj_a` projection weight (#1082,
    /// GDN region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn in_proj_a_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_a, "in_proj_a_kt")
    }

    /// kt-native view of the GDN `in_proj_b` projection weight (#1082,
    /// GDN region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn in_proj_b_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_b, "in_proj_b_kt")
    }

    /// kt-native view of the GDN depthwise `conv1d` weight (#1082, GDN
    /// region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn conv1d_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.conv1d, "conv1d_kt")
    }

    /// kt-native view of the GDN gated-RMSNorm `norm` weight (#1082, GDN
    /// region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn norm_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.norm, "norm_kt")
    }

    /// kt-native view of the GDN `a_log` decay parameter (#1082, GDN
    /// region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn a_log_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.a_log, "a_log_kt")
    }

    /// kt-native view of the GDN `a_log_gates` decay parameter (#1082,
    /// GDN region). Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn a_log_gates_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.a_log_gates, "a_log_gates_kt")
    }

    /// kt-native view of the GDN `dt_bias` parameter (#1082, GDN region).
    /// Same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn dt_bias_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.dt_bias, "dt_bias_kt")
    }

    /// kt-native view of the pre-transposed GDN fused input projection
    /// (`in_proj_qkv_t`, `[hidden, out]`, contiguous since load) (#1082,
    /// GDN region). This is the hot-path transpose used for the decode
    /// `x @ in_proj_qkv_t` matmul; same kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_kt`].
    pub fn in_proj_qkv_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_qkv_t, "in_proj_qkv_t_kt")
    }

    /// kt-native view of the pre-transposed GDN gate projection
    /// (`in_proj_z_t`, `[hidden, nv]`) (#1082, GDN region). Same
    /// kt-native contiguity contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_t_kt`].
    pub fn in_proj_z_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_z_t, "in_proj_z_t_kt")
    }

    /// kt-native view of the pre-transposed GDN `in_proj_a` projection
    /// (`in_proj_a_t`) (#1082, GDN region). Same kt-native contiguity
    /// contract as [`GpuLinearAttentionWeights::in_proj_qkv_t_kt`].
    pub fn in_proj_a_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_a_t, "in_proj_a_t_kt")
    }

    /// kt-native view of the pre-transposed GDN `in_proj_b` projection
    /// (`in_proj_b_t`) (#1082, GDN region). Same kt-native contiguity
    /// contract as [`GpuLinearAttentionWeights::in_proj_qkv_t_kt`].
    pub fn in_proj_b_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.in_proj_b_t, "in_proj_b_t_kt")
    }

    /// kt-native view of the optional pre-transposed combined A/B
    /// projection (`in_proj_ab_t`, `[hidden, 2 * nv]`) (#1082, GDN
    /// region). Returns `Ok(None)` when the fused A/B transpose was not
    /// materialized at load time. When present, same kt-native contiguity
    /// contract as
    /// [`GpuLinearAttentionWeights::in_proj_qkv_t_kt`].
    pub fn in_proj_ab_t_kt(&self) -> Result<Option<KtTensor>> {
        let Some(ab_t) = self.in_proj_ab_t.as_ref() else {
            return Ok(None);
        };
        Ok(Some(kt_contiguous(ab_t, "in_proj_ab_t_kt")?))
    }

    /// kt-native view of the pre-transposed GDN output projection
    /// (`out_proj_t`, `[value_dim, hidden]`, contiguous since load)
    /// (#1082, GDN region). This is the hot-path transpose used for the
    /// `hidden @ out_proj_t` matmul; same kt-native contiguity contract
    /// as [`GpuLinearAttentionWeights::in_proj_qkv_t_kt`].
    pub fn out_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.out_proj_t, "out_proj_t_kt")
    }
}

#[derive(Clone)]
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
    /// Optional cached `[hidden, 2 * intermediate]` transpose combining
    /// `gate_proj_t` and `up_proj_t` along the output dim. Populated at load
    /// time on CUDA so prefill can issue a single `[B*T, hidden] @
    /// [hidden, 2*intermediate]` BF16 GEMM and slice gate/up halves out of the
    /// result, instead of two separate `[B*T, hidden] @ [hidden, intermediate]`
    /// matmuls. Mirrors the [`GpuFullAttentionWeights::qkv_proj_t`] decode
    /// fast path. Skipped when LoRA or Marlin are configured for either of
    /// the gate/up projections (those paths need the standalone transposes).
    pub gate_up_proj_t: Option<Tensor>,
    /// Optional Marlin W4A16-packed MLP projections. Populated when the CUDA
    /// Marlin profile includes them and their shapes fit Marlin's tile
    /// constraints (k%128 && n%256). When present,
    /// the forward path routes the corresponding projection through the
    /// Marlin kernel instead of the BF16 `broadcast_matmul` via `*_t`. LoRA
    /// deltas are still applied on top. Mirrors the q_proj_marlin wire-in
    /// from PR #149 but expands coverage from 8 layers (q_proj on full-attn
    /// layers only) to all 32 layers × 3 MLP projections.
    pub gate_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub up_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub down_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    /// Optional ROCm-only row-wise W8A16 decode projections. `gate_up_proj_w8`
    /// stores `[2 * intermediate, hidden]`; `down_proj_w8` stores
    /// `[hidden, intermediate]`. Enabled by default on ROCm decode; set
    /// The immutable ROCm kernel profile owns packing and dispatch.
    pub gate_up_proj_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
    pub down_proj_w8: Option<crate::rocm_w8_proj::RocmW8Proj>,
}

impl GpuFfnWeights {
    /// kt-native view of the pre-transposed SwiGLU gate projection
    /// (#1082, MLP/FFN region migration — region 2).
    ///
    /// Returns `gate_proj_t` (`[hidden, intermediate]`, contiguous since
    /// load) as a contiguous `KtTensor`. The tensor is already kt-native, so
    /// backend eligibility stays with the caller's request dispatch.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub fn gate_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.gate_proj_t, "gate_proj_t_kt")
    }

    /// kt-native view of the pre-transposed SwiGLU up projection
    /// (#1082, MLP/FFN region migration — region 2). Same kt-native
    /// contiguity contract as [`GpuFfnWeights::gate_proj_t_kt`]; used for
    /// the `x @ up_proj_t` matmul in [`kt_swiglu_ffn_native`].
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub fn up_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.up_proj_t, "up_proj_t_kt")
    }

    /// kt-native view of the pre-transposed SwiGLU down projection
    /// (#1082, MLP/FFN region migration — region 2). Shape
    /// `[intermediate, hidden]`. Same kt-native contiguity contract as
    /// [`GpuFfnWeights::gate_proj_t_kt`]; used for the final
    /// `hidden @ down_proj_t` matmul in [`kt_swiglu_ffn_native`].
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub fn down_proj_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.down_proj_t, "down_proj_t_kt")
    }
}

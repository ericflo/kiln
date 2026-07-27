use super::*;

/// Strict batched single-token paged decode up through the final transformer
/// block.
///
/// This is the model-loop counterpart to
/// [`transformer_block_paged_decode_contiguous_batch`]. It accepts one token
/// per batch row, a block table per row, a common decode position, and an
/// optional batch-shaped [`LinearAttentionState`]. It returns final hidden
/// states with shape `[batch, 1, hidden_size]`.
///
/// The helper is deliberately narrower than the general scheduler contract:
/// every row must share the same `start_pos`, full-attention rows must satisfy
/// the contiguous paged-KV constraints enforced by E340/E341, and LoRA/debug
/// capture paths remain owned by the rowwise entry points until scheduler
/// integration needs them.
#[allow(clippy::too_many_arguments)]
pub(super) fn model_forward_paged_decode_contiguous_batch_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_decode_contiguous_batch_hidden_with_ids(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state.as_deref_mut(),
        lora,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch_hidden_with_ids(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    row_ids: Option<&[u64]>,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    {
        if let Some(hidden) = try_vulkan_resident_batched_decode_hidden(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            start_positions,
            row_ids,
            linear_state.as_deref(),
            lora,
        )? {
            return Ok(hidden);
        }

        if native_resident_decode_required(backend, token_ids, start_positions, config, lora) {
            anyhow::bail!(
                "batched hidden decode declined native resident path; \
                 generic fallback disabled by backend policy"
            );
        }
    }

    model_forward_paged_decode_contiguous_batch_hidden_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state.as_deref_mut(),
        lora,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "metal")]
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_decode_contiguous_batch_hidden_with_stable_buffers(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    stable_positions_gpu: &Tensor,
    stable_token_ids_gpu: &Tensor,
    stable_block_table_gpu: &Tensor,
    stable_seqused_k_gpu: &Tensor,
    stable_attn_out_gpu: &[Tensor],
    stable_softmax_lse_gpu: &[Tensor],
    stable_rotary_cos_gpu: &Tensor,
    stable_rotary_sin_gpu: &Tensor,
    stable_kv_slot_gpu: &Tensor,
    #[cfg(feature = "metal")] metal_icb_inputs: Option<MetalPagedDecodeIcbInputs<'_>>,
) -> Result<Tensor> {
    model_forward_paged_decode_contiguous_batch_hidden_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state.as_deref_mut(),
        lora,
        Some(stable_positions_gpu),
        Some(stable_token_ids_gpu),
        Some(stable_block_table_gpu),
        Some(stable_seqused_k_gpu),
        Some(stable_attn_out_gpu),
        Some(stable_softmax_lse_gpu),
        Some(stable_rotary_cos_gpu),
        Some(stable_rotary_sin_gpu),
        Some(stable_kv_slot_gpu),
        #[cfg(feature = "metal")]
        metal_icb_inputs,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_decode_contiguous_batch_greedy_with_stable_buffers(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    stable_positions_gpu: &Tensor,
    stable_token_ids_gpu: &Tensor,
    stable_block_table_gpu: &Tensor,
    stable_seqused_k_gpu: &Tensor,
    stable_attn_out_gpu: &[Tensor],
    stable_softmax_lse_gpu: &[Tensor],
    stable_rotary_cos_gpu: &Tensor,
    stable_rotary_sin_gpu: &Tensor,
    stable_kv_slot_gpu: &Tensor,
    #[cfg(feature = "metal")] metal_icb_inputs: Option<MetalPagedDecodeIcbInputs<'_>>,
) -> Result<Vec<u32>> {
    let hidden = model_forward_paged_decode_contiguous_batch_hidden_with_stable_buffers(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
        stable_positions_gpu,
        stable_token_ids_gpu,
        stable_block_table_gpu,
        stable_seqused_k_gpu,
        stable_attn_out_gpu,
        stable_softmax_lse_gpu,
        stable_rotary_cos_gpu,
        stable_rotary_sin_gpu,
        stable_kv_slot_gpu,
        #[cfg(feature = "metal")]
        metal_icb_inputs,
    )?;
    let token_ids = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_argmax_decode_stable_buffers");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        match rocm_w8_lm_head_argmax_rows(backend, &normed, weights)? {
            Some(tokens) => tokens,
            None => lm_head_argmax_rows_backend_decode_if(
                Some(backend),
                &normed,
                &weights.embed_tokens_t,
            )?,
        }
    };
    Ok(token_ids)
}

#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch_sample_with_ids(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&LinearAttentionState>,
    lora: Option<&LoraWeights>,
    row_ids: Option<&[u64]>,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    #[cfg(feature = "vulkan")]
    {
        return try_vulkan_resident_batched_decode_sample(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            start_positions,
            row_ids,
            linear_state,
            lora,
            history_rows,
            history_indices,
            history_counts,
            repetition_penalties,
            presence_penalties,
            frequency_penalties,
            temperatures,
            top_k,
            top_p,
            min_p,
            seeds,
        );
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            start_positions,
            linear_state,
            lora,
            row_ids,
            history_rows,
            history_indices,
            history_counts,
            repetition_penalties,
            presence_penalties,
            frequency_penalties,
            temperatures,
            top_k,
            top_p,
            min_p,
            seeds,
        );
        Ok(None)
    }
}

/// Implementation backing `model_forward_paged_decode_contiguous_batch_hidden`
/// plus the upcoming batched CUDA graph wrapper. When
/// `stable_positions_gpu` / `stable_token_ids_gpu` are `Some`, the
/// function skips the per-step host→device builds for those tensors
/// and reads from the caller-owned device pointers instead — exactly
/// the invariant CUDA graph capture/replay needs.
///
/// When `stable_block_table_gpu` / `stable_seqused_k_gpu` are also
/// `Some`, the function uses [`CachedPagedDecodeMeta::build_with_stable_buffers`]
/// in place of [`CachedPagedDecodeMeta::build`], so the captured
/// graph reads the per-step paged-decode metadata from the caller-
/// owned device tensors instead of from transient `Tensor::from_slice`
/// allocations that would be `cudaFree`'d when the per-call meta
/// drops at end of capture. This pins the `ILLEGAL_ADDRESS` fault
/// documented in `bench-results/cuda-graph-bs2-memcheck.md` (#1082).
/// All four stable parameters are independent: callers may pass any
/// subset that matches the buffers their captured-region setup
/// pre-allocates and re-fills.
#[allow(clippy::too_many_arguments)]
pub(super) fn model_forward_paged_decode_contiguous_batch_hidden_inner(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    stable_positions_gpu: Option<&Tensor>,
    stable_token_ids_gpu: Option<&Tensor>,
    stable_block_table_gpu: Option<&Tensor>,
    stable_seqused_k_gpu: Option<&Tensor>,
    // CUDA-graph-stable per-full-attn-layer `(attn_out, softmax_lse)`
    // scratch tensors. When `Some`, the caller has pre-allocated one
    // pair per full-attn layer and threads them through the captured
    // forward so the bs>1 paged-decode kernel writes/reads from
    // stable buffers instead of allocating fresh `Tensor::zeros`
    // inside the captured region. Mirrors the bs=1
    // `PagedDecodeGraphInputs::{attn_out, softmax_lse}` plumbing
    // (#1082 suspects 3+4).
    stable_attn_out_gpu: Option<&[Tensor]>,
    stable_softmax_lse_gpu: Option<&[Tensor]>,
    // CUDA-graph-stable RoPE `[batch, rotary_dim/2]` cos/sin tables.
    // When `Some`, the caller has pre-allocated runner-owned device
    // tensors and refills them via
    // `CudaGraphRunner::update_batched_rotary_buffers` before each
    // graph replay. The captured RoPE step reads from those stable
    // pointers via `rotary_embedding_from_tables`, instead of the
    // legacy `rotary_embedding_from_tensor` path that allocates fresh
    // `freqs/cos/sin` `cudaMalloc` tensors inside the captured region.
    // Mirrors the bs=1 `PagedDecodeGraphInputs::{rotary_cos,
    // rotary_sin}` plumbing into `gqa_attention_paged_with_rope_tables`
    // (#1082 suspect 2 — see
    // `bench-results/cuda-graph-bs2-secondary-audit.md`).
    stable_rotary_cos_gpu: Option<&Tensor>,
    stable_rotary_sin_gpu: Option<&Tensor>,
    // Runner-owned or per-step `[batch]` u32 per-row KV-write slot tensor.
    // When `Some`, the bs>1 KV-write step
    // dispatches the fused batched-slot kernel
    // (`PagedKvCache::write_token_major_native_batch_graph_slot`) so
    // every layer reads its per-row destination slot from one device tensor.
    // Captured paths supply a stable runner-owned buffer; ROCm eager decode
    // builds one per step and reuses it across all full-attention layers.
    stable_kv_slot_gpu: Option<&Tensor>,
    #[cfg(feature = "metal")] mut metal_icb_inputs: Option<MetalPagedDecodeIcbInputs<'_>>,
) -> Result<Tensor> {
    let batch = token_ids.len();
    anyhow::ensure!(batch > 0, "batched paged decode requires a non-empty batch");
    anyhow::ensure!(
        block_tables.len() == batch && start_positions.len() == batch,
        "batched paged decode metadata length mismatch"
    );
    if weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
    {
        let state_batch = linear_state
            .as_ref()
            .context("batched paged decode requires LinearAttentionState for GDN layers")?
            .batch_size()?;
        anyhow::ensure!(
            state_batch == batch,
            "batched paged decode LinearAttentionState batch mismatch ({state_batch} vs {batch})"
        );
    }

    let device = weights.embed_tokens.device();
    // Embedding lookup. When the caller supplies a stable `[batch] u32`
    // token-id tensor on the device (CUDA graph capture path), use it
    // via the index-based lookup — the device pointer stays valid
    // across replays. Otherwise build fresh from the host slice.
    let mut hidden = if let Some(token_ids_gpu) = stable_token_ids_gpu {
        embedding_lookup_from_weights_with_index(token_ids_gpu, weights)?.unsqueeze(1)?
    } else {
        embedding_lookup_from_weights(token_ids, weights)?.unsqueeze(1)?
    };
    // When every row decodes at the same position (the common case — all
    // requests admitted with same-length prompts or all admitted at the same
    // decode step), pass a single-element positions tensor so the full-attn
    // RoPE picks the fast scalar-broadcast path (`positions.elem_count() == 1`)
    // and skips the 4-transpose+contig dance the per-row path needs to align
    // cos/sin with the batch dim. nsys at bs=16 (post-broadcast-matmul fix)
    // showed ~32 RoPE transpose+contig copies per decode step routing through
    // copy2d_bf16; this elides them when positions happen to be uniform.
    //
    // CUDA graph capture path: when `stable_positions_gpu` is `Some`,
    // skip both branches above and use the caller-owned device buffer
    // so the captured RoPE kernels read from a graph-stable pointer.
    // The bench shows the per-step `Tensor::from_slice` here is a tiny
    // HtoD launch (one per step), so the win comes from graph
    // captureability, not from elimination of the copy itself.
    let first_pos = start_positions[0];
    let positions_uniform = start_positions.iter().all(|&p| p == first_pos);
    let positions_owned: Option<Tensor> = if stable_positions_gpu.is_none() {
        Some(if positions_uniform {
            Tensor::from_vec_on(device, vec![first_pos as f32], vec![1])?
        } else {
            let positions_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
            Tensor::from_vec_on(device, positions_f32, vec![batch])?
        })
    } else {
        None
    };
    let positions: &Tensor = stable_positions_gpu.unwrap_or_else(|| {
        positions_owned
            .as_ref()
            .expect("positions_owned built above when stable was None")
    });
    let use_metal_decode_ffn = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_runtime::single_token_self_attention_active();

    // Build the per-step paged-decode metadata once when there are any
    // full-attention layers in the model. Each gqa call within this step
    // would otherwise rebuild the seqused_k + padded block_table tensors
    // (one HtoD launch each) per layer (8× on Qwen3.5-4B); hoisting saves
    // 14 launches per step. Skip the build entirely on linear-only models.
    let has_full_attention_layer = weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)));

    // The physical destination slots are identical for every full-attention
    // layer in one decode step. ROCm's fused device-slot writer consumes this
    // tensor without a host readback, so build it once here rather than once
    // per layer. A graph runner's stable slot buffer remains authoritative
    // when supplied.
    #[cfg(feature = "rocm")]
    let rocm_kv_slot_owned: Option<Tensor> = if stable_kv_slot_gpu.is_none()
        && has_full_attention_layer
        && matches!(device, Device::Rocm(_))
    {
        let slots = paged_cache.resolve_unique_decode_slots(block_tables, start_positions)?;
        Some(
            Tensor::from_vec_on(device, slots, vec![batch])?
                .contiguous()
                .context("build ROCm batched decode KV slot tensor")?,
        )
    } else {
        None
    };
    #[cfg(feature = "rocm")]
    let effective_kv_slot_gpu = stable_kv_slot_gpu.or(rocm_kv_slot_owned.as_ref());
    #[cfg(not(feature = "rocm"))]
    let effective_kv_slot_gpu = stable_kv_slot_gpu;

    let cached_paged_meta: Option<CachedPagedDecodeMeta> = if has_full_attention_layer {
        // CUDA graph capture path: when the caller supplies stable
        // `block_table` + `seqused_k` device buffers, build the meta
        // around those instead of allocating fresh per-step tensors
        // via `Tensor::from_slice`. The transient-allocation path bakes
        // dangling pointers into the captured graph (#1082, see
        // `bench-results/cuda-graph-bs2-memcheck.md`). Both stable
        // tensors must be supplied together — they're a single
        // logical "meta" input pair. If only one is provided, fall
        // back to the regular build path to keep the code obviously
        // correct rather than mixing stable and transient storage.
        match (stable_block_table_gpu, stable_seqused_k_gpu) {
            (Some(bt), Some(sk)) => Some(
                CachedPagedDecodeMeta::build_with_stable_buffers(
                    paged_cache,
                    block_tables,
                    start_positions,
                    bt,
                    sk,
                    // Phase 7 #1082: kt twin threading deferred — caller does not
                    // yet expose `kt_paged_cache` to the inner. `None` keeps the
                    // `paged_cache.block_size()` read on the candle path (the
                    // helper short-circuits when kt is `None`).
                    #[cfg(feature = "cuda")]
                    None,
                )
                .context("build cached paged decode metadata for batched step (stable buffers)")?,
            ),
            _ => Some(
                CachedPagedDecodeMeta::build(
                    &device,
                    paged_cache,
                    block_tables,
                    start_positions,
                    // Phase 7 #1082: kt twin threading deferred — caller does not
                    // yet expose `kt_paged_cache` to the inner. `None` keeps the
                    // `paged_cache.block_size()` read on the candle path (the
                    // helper short-circuits when kt is `None`).
                    #[cfg(feature = "cuda")]
                    None,
                )
                .context("build cached paged decode metadata for batched step")?,
            ),
        }
    } else {
        None
    };
    let cached_paged_meta_ref = cached_paged_meta.as_ref();

    let mut full_attn_idx = 0usize;
    let mut linear_attn_idx = 0usize;
    for (i, layer) in weights.layers.iter().enumerate() {
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Pick out the per-layer graph-stable paged-decode scratch
                // tensors when the caller threaded them in. The captured
                // CUDA graph re-uses these across replays so the kernel
                // doesn't dangle pointers when its transient `Tensor::zeros`
                // scratch drops at end of capture (#1082 suspects 3+4).
                let layer_graph_outputs: Option<(&Tensor, &Tensor)> =
                    match (stable_attn_out_gpu, stable_softmax_lse_gpu) {
                        (Some(attn_outs), Some(lses)) => {
                            let attn_out = attn_outs.get(full_attn_idx).with_context(|| {
                                format!(
                                    "stable attn_out missing for full-attn layer {full_attn_idx}"
                                )
                            })?;
                            let lse = lses.get(full_attn_idx).with_context(|| {
                                format!(
                                    "stable softmax_lse missing for full-attn layer {full_attn_idx}"
                                )
                            })?;
                            Some((attn_out, lse))
                        }
                        _ => None,
                    };
                // Pull the runner-owned stable rotary cos/sin tables
                // through. Both must be provided together — they're a
                // logical pair populated by the same
                // `update_batched_rotary_buffers` step (#1082 suspect 2).
                let layer_rope_tables: Option<(&Tensor, &Tensor)> =
                    match (stable_rotary_cos_gpu, stable_rotary_sin_gpu) {
                        (Some(cos), Some(sin)) => Some((cos, sin)),
                        _ => None,
                    };
                #[cfg(feature = "metal")]
                let layer_metal_icb = match metal_icb_inputs.as_mut() {
                    Some(inputs) => {
                        let q = inputs.q.get(full_attn_idx).with_context(|| {
                            format!(
                                "Metal ICB stable Q missing for full-attn layer {full_attn_idx}"
                            )
                        })?;
                        let k = inputs.k.get(full_attn_idx).with_context(|| {
                            format!(
                                "Metal ICB stable K missing for full-attn layer {full_attn_idx}"
                            )
                        })?;
                        let v = inputs.v.get(full_attn_idx).with_context(|| {
                            format!(
                                "Metal ICB stable V missing for full-attn layer {full_attn_idx}"
                            )
                        })?;
                        let graph = inputs.graphs.get_mut(full_attn_idx).with_context(|| {
                            format!(
                                "Metal ICB graph slot missing for full-attn layer {full_attn_idx}"
                            )
                        })?;
                        Some(MetalPagedDecodeIcbLayer { q, k, v, graph })
                    }
                    None => None,
                };
                hidden = transformer_block_paged_decode_contiguous_batch(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    start_positions,
                    &weights.rotary_inv_freq,
                    paged_cache,
                    block_tables,
                    full_attn_idx,
                    layer_lora,
                    cached_paged_meta_ref,
                    layer_graph_outputs,
                    layer_rope_tables,
                    effective_kv_slot_gpu,
                    #[cfg(feature = "metal")]
                    layer_metal_icb,
                    #[cfg(feature = "cuda")]
                    None,
                )
                .with_context(|| {
                    format!("batched transformer block {i} (full attention, paged)")
                })?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("batched linear attention state required for GDN layer {i}")
                })?;
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn_batch_decode");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    use_metal_decode_ffn,
                    use_metal_decode_ffn,
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| {
                    format!("batched gated deltanet layer {i} (linear attention, paged)")
                })?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual_batch_decode");
                    residual_add(hidden, attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp_batch_decode");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual_batch_decode");
                    residual_add(hidden, ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Strict batched single-token paged decode for model-forward integration.
///
/// This is the model-loop counterpart to
/// [`transformer_block_paged_decode_contiguous_batch`]. It accepts one token
/// per batch row, a block table per row, per-row decode positions, and an
/// optional batch-shaped [`LinearAttentionState`]. It returns full logits with
/// shape `[batch, 1, vocab_size]`.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let hidden = model_forward_paged_decode_contiguous_batch_hidden(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
    )?;
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_decode");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
    };
    Ok(logits)
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_vulkan_resident_batched_decode_argmax(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    row_ids: Option<&[u64]>,
    linear_state: Option<&LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Option<Vec<u32>>> {
    let batch = token_ids.len();
    if batch == 0
        || block_tables.len() != batch
        || start_positions.len() != batch
        || row_ids.is_some_and(|ids| ids.len() != batch)
        || start_positions.iter().any(|&p| p == 0)
        || lora.is_some()
        || !config.attn_output_gate
        || !vulkan_resident_decode_enabled()
        || !ReplayBackend::runtime_supports_resident_decode(backend)
        || !resident_decode_pool_ready(backend, config)
        || crate::mtp_runtime::single_token_self_attention_active()
    {
        return Ok(None);
    }

    let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
        .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
    else {
        return Ok(None);
    };
    let Some(vk_device) = vk_backend.vulkan_device() else {
        return Ok(None);
    };
    let Some(vk_kv_cache) = vk_backend.vk_paged_kv_cache(
        config.num_full_attention_layers,
        paged_cache.num_blocks(),
        paged_cache.block_size(),
        config.num_kv_heads,
        config.head_dim,
    ) else {
        return Ok(None);
    };
    let has_linear_layers = weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)));
    let empty_states: &[Tensor] = &[];
    let (recurrent_states, conv_states): (&[Tensor], &[Tensor]) = if has_linear_layers {
        let Some(state) = linear_state else {
            return Ok(None);
        };
        if state.batch_size()? != batch {
            return Ok(None);
        }
        (
            state.recurrent_states.as_slice(),
            state.conv_states.as_slice(),
        )
    } else {
        (empty_states, empty_states)
    };

    let single_unidentified_row = row_ids.is_none() && batch == 1;
    if single_unidentified_row {
        vk_backend.note_resident_session(start_positions[0]);
    } else if row_ids.is_none() {
        vk_backend.reset_resident_decode_row_seeded();
    }
    let mut full_attn_idx = 0usize;
    for layer in weights.layers.iter() {
        if matches!(layer.attention, GpuAttentionWeights::Full(_)) {
            let mut seed_rows = Vec::new();
            let mut seed_tables = Vec::new();
            for row_idx in 0..batch {
                let should_seed = if single_unidentified_row {
                    !vk_backend.full_attn_layer_seeded(full_attn_idx)
                } else {
                    row_ids
                        .map(|ids| {
                            !vk_backend.resident_decode_row_seeded(full_attn_idx, ids[row_idx])
                        })
                        .unwrap_or(true)
                };
                if should_seed {
                    seed_rows.push(row_idx);
                    seed_tables.push(block_tables[row_idx]);
                }
            }
            if !seed_tables.is_empty() {
                crate::vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_batched_tables(
                    vk_device,
                    vk_kv_cache,
                    paged_cache,
                    full_attn_idx,
                    &seed_tables,
                )?;
                if single_unidentified_row {
                    vk_backend.mark_full_attn_layer_seeded(full_attn_idx);
                } else if let Some(ids) = row_ids {
                    for row_idx in seed_rows {
                        vk_backend.mark_resident_decode_row_seeded(full_attn_idx, ids[row_idx]);
                    }
                }
            }
            full_attn_idx += 1;
        }
    }

    crate::vk_decode_resident::submit_transformer_stack_batched_argmax_from_tokens(
        vk_backend,
        vk_device,
        token_ids,
        block_tables,
        start_positions,
        paged_cache.block_size(),
        weights,
        config,
        vk_kv_cache,
        recurrent_states,
        conv_states,
    )
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_vulkan_resident_batched_decode_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    row_ids: Option<&[u64]>,
    linear_state: Option<&LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Option<Tensor>> {
    let batch = token_ids.len();
    if batch == 0
        || block_tables.len() != batch
        || start_positions.len() != batch
        || row_ids.is_some_and(|ids| ids.len() != batch)
        || start_positions.iter().any(|&p| p == 0)
        || lora.is_some()
        || !config.attn_output_gate
        || !vulkan_resident_decode_enabled()
        || !ReplayBackend::runtime_supports_resident_decode(backend)
        || !resident_decode_pool_ready(backend, config)
        || crate::mtp_runtime::single_token_self_attention_active()
    {
        return Ok(None);
    }

    let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
        .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
    else {
        return Ok(None);
    };
    let Some(vk_device) = vk_backend.vulkan_device() else {
        return Ok(None);
    };
    let Some(vk_kv_cache) = vk_backend.vk_paged_kv_cache(
        config.num_full_attention_layers,
        paged_cache.num_blocks(),
        paged_cache.block_size(),
        config.num_kv_heads,
        config.head_dim,
    ) else {
        return Ok(None);
    };
    let has_linear_layers = weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)));
    let empty_states: &[Tensor] = &[];
    let (recurrent_states, conv_states): (&[Tensor], &[Tensor]) = if has_linear_layers {
        let Some(state) = linear_state else {
            return Ok(None);
        };
        if state.batch_size()? != batch {
            return Ok(None);
        }
        (
            state.recurrent_states.as_slice(),
            state.conv_states.as_slice(),
        )
    } else {
        (empty_states, empty_states)
    };

    let single_unidentified_row = row_ids.is_none() && batch == 1;
    if single_unidentified_row {
        vk_backend.note_resident_session(start_positions[0]);
    } else if row_ids.is_none() {
        vk_backend.reset_resident_decode_row_seeded();
    }
    let mut full_attn_idx = 0usize;
    for layer in weights.layers.iter() {
        if matches!(layer.attention, GpuAttentionWeights::Full(_)) {
            let mut seed_rows = Vec::new();
            let mut seed_tables = Vec::new();
            for row_idx in 0..batch {
                let should_seed = if single_unidentified_row {
                    !vk_backend.full_attn_layer_seeded(full_attn_idx)
                } else {
                    row_ids
                        .map(|ids| {
                            !vk_backend.resident_decode_row_seeded(full_attn_idx, ids[row_idx])
                        })
                        .unwrap_or(true)
                };
                if should_seed {
                    seed_rows.push(row_idx);
                    seed_tables.push(block_tables[row_idx]);
                }
            }
            if !seed_tables.is_empty() {
                crate::vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_batched_tables(
                    vk_device,
                    vk_kv_cache,
                    paged_cache,
                    full_attn_idx,
                    &seed_tables,
                )?;
                if single_unidentified_row {
                    vk_backend.mark_full_attn_layer_seeded(full_attn_idx);
                } else if let Some(ids) = row_ids {
                    for row_idx in seed_rows {
                        vk_backend.mark_resident_decode_row_seeded(full_attn_idx, ids[row_idx]);
                    }
                }
            }
            full_attn_idx += 1;
        }
    }

    let Some(hidden_rows) =
        crate::vk_decode_resident::submit_transformer_stack_batched_hidden_from_tokens(
            vk_backend,
            vk_device,
            token_ids,
            block_tables,
            start_positions,
            paged_cache.block_size(),
            weights,
            config,
            vk_kv_cache,
            recurrent_states,
            conv_states,
        )?
    else {
        return Ok(None);
    };
    let device = weights.embed_tokens.device();
    let out = Tensor::from_vec_on(device, hidden_rows, vec![batch, 1usize, config.hidden_size])?;
    Ok(Some(out))
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_vulkan_resident_batched_decode_sample(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    row_ids: Option<&[u64]>,
    linear_state: Option<&LinearAttentionState>,
    lora: Option<&LoraWeights>,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    let batch = token_ids.len();
    if batch == 0
        || block_tables.len() != batch
        || start_positions.len() != batch
        || row_ids.is_some_and(|ids| ids.len() != batch)
        || start_positions.iter().any(|&p| p == 0)
        || lora.is_some()
        || !config.attn_output_gate
        || !vulkan_resident_decode_enabled()
        || !ReplayBackend::runtime_supports_resident_decode(backend)
        || !resident_decode_pool_ready(backend, config)
        || crate::mtp_runtime::single_token_self_attention_active()
    {
        return Ok(None);
    }

    let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
        .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
    else {
        return Ok(None);
    };
    let Some(vk_device) = vk_backend.vulkan_device() else {
        return Ok(None);
    };
    let Some(vk_kv_cache) = vk_backend.vk_paged_kv_cache(
        config.num_full_attention_layers,
        paged_cache.num_blocks(),
        paged_cache.block_size(),
        config.num_kv_heads,
        config.head_dim,
    ) else {
        return Ok(None);
    };
    let has_linear_layers = weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)));
    let empty_states: &[Tensor] = &[];
    let (recurrent_states, conv_states): (&[Tensor], &[Tensor]) = if has_linear_layers {
        let Some(state) = linear_state else {
            return Ok(None);
        };
        if state.batch_size()? != batch {
            return Ok(None);
        }
        (
            state.recurrent_states.as_slice(),
            state.conv_states.as_slice(),
        )
    } else {
        (empty_states, empty_states)
    };

    let single_unidentified_row = row_ids.is_none() && batch == 1;
    if single_unidentified_row {
        vk_backend.note_resident_session(start_positions[0]);
    } else if row_ids.is_none() {
        vk_backend.reset_resident_decode_row_seeded();
    }
    let mut full_attn_idx = 0usize;
    for layer in weights.layers.iter() {
        if matches!(layer.attention, GpuAttentionWeights::Full(_)) {
            let mut seed_rows = Vec::new();
            let mut seed_tables = Vec::new();
            for row_idx in 0..batch {
                let should_seed = if single_unidentified_row {
                    !vk_backend.full_attn_layer_seeded(full_attn_idx)
                } else {
                    row_ids
                        .map(|ids| {
                            !vk_backend.resident_decode_row_seeded(full_attn_idx, ids[row_idx])
                        })
                        .unwrap_or(true)
                };
                if should_seed {
                    seed_rows.push(row_idx);
                    seed_tables.push(block_tables[row_idx]);
                }
            }
            if !seed_tables.is_empty() {
                crate::vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_batched_tables(
                    vk_device,
                    vk_kv_cache,
                    paged_cache,
                    full_attn_idx,
                    &seed_tables,
                )?;
                if single_unidentified_row {
                    vk_backend.mark_full_attn_layer_seeded(full_attn_idx);
                } else if let Some(ids) = row_ids {
                    for row_idx in seed_rows {
                        vk_backend.mark_resident_decode_row_seeded(full_attn_idx, ids[row_idx]);
                    }
                }
            }
            full_attn_idx += 1;
        }
    }

    crate::vk_decode_resident::submit_transformer_stack_batched_sample_from_tokens(
        vk_backend,
        vk_device,
        token_ids,
        block_tables,
        start_positions,
        paged_cache.block_size(),
        weights,
        config,
        vk_kv_cache,
        recurrent_states,
        conv_states,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )
}

#[cfg(feature = "vulkan")]
pub(super) fn native_resident_decode_required(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    start_positions: &[usize],
    config: &kiln_core::config::ModelConfig,
    lora: Option<&LoraWeights>,
) -> bool {
    !token_ids.is_empty()
        && start_positions.len() == token_ids.len()
        && start_positions.iter().all(|&pos| pos > 0)
        && lora.is_none()
        && config.attn_output_gate
        && vulkan_resident_decode_enabled()
        && ReplayBackend::runtime_supports_resident_decode(backend)
        && !crate::mtp_runtime::single_token_self_attention_active()
}

/// Strict batched single-token paged decode that returns greedy next-token IDs
/// without materializing full logits when a backend has a fused argmax path.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch_greedy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Vec<u32>> {
    model_forward_paged_decode_contiguous_batch_greedy_with_ids(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch_greedy_with_ids(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    row_ids: Option<&[u64]>,
) -> Result<Vec<u32>> {
    #[cfg(feature = "vulkan")]
    {
        if let Some(next_tokens) = try_vulkan_resident_batched_decode_argmax(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            start_positions,
            row_ids,
            linear_state.as_deref(),
            lora,
        )? {
            return Ok(next_tokens);
        }

        if native_resident_decode_required(backend, token_ids, start_positions, config, lora) {
            anyhow::bail!(
                "greedy decode declined native resident path; \
                 generic fallback disabled by backend policy"
            );
        }
    }

    let hidden = model_forward_paged_decode_contiguous_batch_hidden(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
    )?;
    let token_ids = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_argmax_decode");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        match rocm_w8_lm_head_argmax_rows(backend, &normed, weights)? {
            Some(tokens) => tokens,
            None => lm_head_argmax_rows_backend_decode_if(
                Some(backend),
                &normed,
                &weights.embed_tokens_t,
            )?,
        }
    };
    Ok(token_ids)
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
///
/// #1082: kt-native full (non-paged) forward pass — the sole entry point. The
/// old candle-returning `model_forward` shim + its kt→candle bridge helpers were
/// removed once the kt tape (kiln_autograd) became the sole grad producer. The
/// forward internals produce kt tensors (bare `Tensor` = `kiln_tensor::Tensor`)
/// and this returns kt logits `[1, seq_len, vocab_size]` directly — no candle
/// round-trip. Callers in generate/speculative/server consume the kt tensor
/// through `kiln_tensor::ops::*`.
pub fn model_forward_kt(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    kv_cache: Option<&mut KvCache>,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_kt_with_policy(
        backend,
        token_ids,
        weights,
        config,
        kv_cache,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`model_forward_kt`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_kt_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mut kv_cache: Option<&mut KvCache>,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    let seq_len = token_ids.len();

    // 1. Embedding lookup: [seq_len, hidden_size]
    // The weight-aware lookup applies the backend activation precision policy.
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position indices for RoPE — absolute positions accounting for cached tokens
    let offset = kv_cache.as_ref().map_or(0, |c| c.seq_len());
    let positions: Vec<u32> = (offset..offset + seq_len).map(|p| p as u32).collect();
    let use_metal_decode_ffn =
        seq_len == 1 && offset > 0 && !crate::mtp_runtime::single_token_self_attention_active();

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
                hidden = transformer_block_with_policy(
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
                    streaming_prefill,
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
                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    /* use_fused_gdn_gates = */ true,
                    use_metal_decode_ffn,
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention)"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, attn_out)?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn attention_residual"),
                    &hidden,
                )?;
                // Post-attention RMSNorm + FFN
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = if use_metal_decode_ffn {
                    swiglu_ffn_metal_decode(&normed_post, &layer.mlp, layer_lora)?
                } else {
                    swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?
                };
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, ffn_out)?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn output"),
                    &hidden,
                )?;
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
        lm_head_forward_backend_decode_if(Some(backend), &hidden, &weights.embed_tokens_t)?
    };

    // #1082 forward-flip: `logits` is a kt tensor — return it directly. The
    // kt tape (kiln_autograd) is the sole grad producer; there is no longer
    // a candle consumer to bridge to.
    Ok(logits)
}

// (#1082) Deleted `model_forward_logits_kt_to_candle`: `model_forward_kt`
//   returns kt logits directly now — no candle bridge.

// (#1082, PR7 cleanup) Deleted the dead `kt_logits_to_candle` /
//   `candle_to_kt_activation` kt-seam passthroughs. Their only caller was the
//   legacy `vk_forward.rs`, deleted in PR7; the vulkan lane is now fully
//   kt-native (no candle round-trip), so both helpers were unreferenced
//   dead code (dead-code warnings since PR7).

/// kt has no infer-from-hole reshape (candle's `((), d1, d2)`). These
/// helpers reproduce a rank-3 / rank-4 reshape where the leading
/// dimension is inferred from the element count, divided by the product
/// of the explicit trailing dims (#1082 forward-flip). The inferred dim
/// is placed in slot 0, matching every candle `reshape(((), ..))` site
/// flipped here (all of which infer batch in position 0).
#[inline]
pub(super) fn reshape_hole0_3(t: &Tensor, d1: usize, d2: usize) -> Result<Tensor> {
    let n = t.element_count();
    Ok(t.reshape((n / (d1 * d2), d1, d2))?)
}

#[inline]
pub(super) fn reshape_hole0_4(t: &Tensor, d1: usize, d2: usize, d3: usize) -> Result<Tensor> {
    let n = t.element_count();
    Ok(t.reshape((n / (d1 * d2 * d3), d1, d2, d3))?)
}

// (#1082) Deleted the old `model_forward_kt` wrapper (it delegated to the
//   removed candle-returning `model_forward` shim). `model_forward_kt` is now
//   the primary kt-native forward, defined above.

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
    hidden: Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    start_layer: usize,
    end_layer: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_segment_with_policy(
        backend,
        hidden,
        weights,
        config,
        positions,
        start_layer,
        end_layer,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`model_forward_segment`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_segment_with_policy(
    backend: &dyn BackendRuntime,
    hidden: Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    start_layer: usize,
    end_layer: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    // (#1443 step 3) Defensive policy-owned activation cast for direct callers
    // (gradient-checkpointing recompute) that may hand in a pre-cast hidden.
    let mut hidden = cast_embedding_output_to_policy_activation(hidden)?;
    // Count full-attention and linear-attention layers before start_layer
    // so we index into the right KV cache / linear state slots.
    let mut full_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Full(_)))
        .count();
    let mut linear_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Linear(_)))
        .count();

    // Phase 10: training-time streaming GDN prefill.
    //
    // When streaming prefill is enabled and the segment's seq_len exceeds the
    // configured tile size, GDN layers run as a sequence of smaller tiles
    // threading `LinearAttentionState` per tile. Detached full-attention
    // boundary forwards on capable backends also tile query blocks while
    // preserving each tile's full causal prefix; tape-authoritative reverse
    // forwards use the kt flash-attention recorder where available. Inter-layer
    // hidden activations still stay at full T shape, but the large per-call
    // attention/GDN scratch allocations are bounded by the backend tile policy.
    let (_, seq_len, _) = hidden.dims3()?;
    let streaming = streaming_prefill.enabled_for(seq_len);
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let tape_scope_active = crate::tape_forward::tape_scope_active();
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let tape_scope_active = false;
    let stream_tile = if streaming {
        if tape_scope_active {
            streaming_prefill.tape_tile_tokens()
        } else {
            streaming_prefill.base_tile_tokens_for(seq_len)
        }
    } else {
        0
    };
    let stream_active = streaming && stream_tile > 0 && seq_len > stream_tile;

    for i in start_layer..end_layer {
        let layer = &weights.layers[i];
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Training doesn't use KV cache
                hidden = transformer_block_with_policy(
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
                    streaming_prefill,
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
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn pre_attn_norm"),
                    &normed,
                )?;
                let attn_out = if stream_active {
                    gated_deltanet_forward_streaming(
                        backend,
                        &normed,
                        lin_weights,
                        config,
                        &mut state.recurrent_states[linear_attn_idx],
                        &mut state.conv_states[linear_attn_idx],
                        stream_tile,
                        layer_lora,
                    )
                    .with_context(|| format!("segment streaming gated deltanet layer {i}"))?
                } else {
                    gated_deltanet_forward_decode_if(
                        backend,
                        &normed,
                        lin_weights,
                        config,
                        &mut state.recurrent_states[linear_attn_idx],
                        &mut state.conv_states[linear_attn_idx],
                        /* use_fused_gdn_gates = */ true,
                        /* use_metal_decode_gemv = */ false,
                        /* allow_forward_only_fastpaths = */ true,
                        /* allow_prefill_recurrent_kernel = */ true,
                        layer_lora,
                    )
                    .with_context(|| format!("segment gated deltanet layer {i}"))?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn attention"),
                    &attn_out,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, attn_out)?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn attention_residual"),
                    &hidden,
                )?;
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn post_attn_norm"),
                    &normed_post,
                )?;
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    false,
                )?;
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn mlp"),
                    &ffn_out,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, ffn_out)?
                };
                synchronize_tensor_ready_for_model_handoff(
                    &format!("layer {i} gdn output"),
                    &hidden,
                )?;
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
    // The weight-aware lookup applies the backend activation precision policy.
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
    model_forward_head_backend_decode_if(None, hidden, weights, config)
}

pub fn model_forward_head_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/lm_head");
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    #[cfg(feature = "rocm")]
    if let Some(lm_head_w8) = weights.lm_head_w8.as_ref() {
        let dims = normed.dims();
        let lead: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        if lead == 1
            && normed.dtype() == DType::BF16
            && !normed.track_op()
            && matches!(normed.device(), Device::Rocm(_))
        {
            let normed = normed
                .contiguous()
                .context("rocm w8 sampled lm_head normed contiguous")?;
            return crate::rocm_w8_proj::matmul_bf16(&normed, lm_head_w8)
                .context("rocm w8 sampled lm_head");
        }
    }
    let logits = lm_head_forward_backend_decode_if(backend, &normed, &weights.embed_tokens_t)?;
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

/// Project post-final-RMSNorm hidden states through the tied LM head.
///
/// Unlike [`model_forward_head`], this function does not apply final RMSNorm.
/// It is intended for callers that run [`model_forward_no_head`] once and then
/// project bounded sequence chunks, avoiding simultaneous residency of the
/// full `[sequence, vocabulary]` logits tensor. The leading dimensions are
/// preserved and only the last dimension is projected from hidden size to
/// vocabulary size.
pub fn model_forward_project_normalized_hidden(
    backend: &dyn BackendRuntime,
    normalized_hidden: &Tensor,
    weights: &GpuWeights,
) -> Result<Tensor> {
    let hidden_size = normalized_hidden
        .dims()
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("normalized hidden tensor must have rank >= 1"))?;
    let projection_hidden_size = weights
        .embed_tokens_t
        .dims()
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("LM-head projection tensor must have rank >= 1"))?;
    if hidden_size != projection_hidden_size {
        anyhow::bail!(
            "normalized hidden width {hidden_size} did not match LM-head input width \
             {projection_hidden_size}"
        );
    }

    kiln_nvtx::range!(c"kiln/lm_head_normalized_chunk");
    lm_head_forward_backend_decode_if(Some(backend), normalized_hidden, &weights.embed_tokens_t)
}

/// Full training-path forward WITHOUT the LM head projection.
///
/// Runs embedding -> transformer layers -> final RMSNorm, returning the
/// post-final-RMSNorm hidden state `[1, seq_len, hidden_size]`. This is the
/// input the Fused Linear Cross-Entropy path consumes, avoiding the
/// `[1, seq_len, vocab_size]` logits materialization that dominates peak
/// VRAM at long context on the Qwen3.5-4B head (V=151936).
///
/// SFT and GRPO call this when their backend-owned typed loss route consumes
/// normalized hidden states directly. No KV cache is used (matches
/// `standard_forward_backward`).
pub fn model_forward_no_head(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_no_head_with_policy(
        backend,
        token_ids,
        weights,
        config,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`model_forward_no_head`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_no_head_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    let (hidden, positions) = model_forward_embed(token_ids, weights)?;
    let num_layers = weights.layers.len();
    let hidden = model_forward_segment_with_policy(
        backend,
        hidden,
        weights,
        config,
        &positions,
        0,
        num_layers,
        linear_state,
        lora,
        streaming_prefill,
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    // Native single-submit Vulkan-resident decode fast-path. For
    // `seq_len == 1` (decode hot path) the LastRowOnly logits returned
    // by the native path are shape-equivalent to LmHeadMode::Full's
    // single-row output, so callers see identical behaviour.
    #[cfg(feature = "vulkan")]
    {
        if token_ids.len() == 1
            && start_pos > 0
            && lora.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && config.attn_output_gate
            && vulkan_resident_decode_enabled()
            && ReplayBackend::runtime_supports_resident_decode(backend)
            && resident_decode_pool_ready(backend, config)
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(logits) = model_forward_paged_last_token_resident_native_vk(
                    vk_backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    start_pos,
                    linear_state.as_deref(),
                )? {
                    return Ok(logits);
                }
            }
        }

        if native_resident_decode_required(backend, token_ids, &[start_pos], config, lora) {
            anyhow::bail!(
                "decode declined native resident path; \
                 generic fallback disabled by backend policy"
            );
        }
    }

    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::Full,
    )?;
    // `LmHeadMode::Full` always returns Some.
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// Paged-KV forward pass with an optional [`PagedKvCacheKt`] twin threaded
/// through to the per-layer GQA-attention writer.
///
/// Same contract as [`model_forward_paged`] but lets the caller pass a kt
/// cache that will be written-mirrored alongside the candle [`PagedKvCache`].
/// When `kt_paged_cache` is `None` (or built features are non-CUDA), behavior
/// is bit-identical to [`model_forward_paged`] — the candle writer is the only
/// thing that runs. When `Some(&kt)` is passed AND
/// `accelerator.kt_api_mode = "all"` is on, every paged-KV write inside
/// [`gqa_attention_paged_with_rope_tables`] mirrors into the kt cache via
/// `try_kt_paged_kv_write_token_major_native_graph_slot`.
///
/// # Why this exists
///
/// Phase 7 #1082 staging step: the writer plumbing (commits `7dd0009c`,
/// `d67b6096`) and the inner-fn parameter (the commit before this one) are
/// landed, but no caller passes `Some(&kt)`. This sibling function is the
/// first public entry point that does — bench/latency code (`kiln-server`)
/// can opt into the mirrored write path by allocating a kt twin alongside
/// its `PagedKvCache` and routing through this fn instead of
/// [`model_forward_paged`]. Every other production caller keeps using
/// [`model_forward_paged`] unchanged, so the kt path stays opt-in.
///
/// # Vulkan fast-path skipped
///
/// Unlike [`model_forward_paged`], this fn does *not* dispatch to the
/// Vulkan native single-submit resident-decode path: the kt twin is
/// CUDA-only (see `PagedKvCacheKt`), so passing it on a Vulkan device is
/// already meaningless. Callers that want the Vulkan fast path should
/// keep using [`model_forward_paged`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_with_kt(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
    // Phase 7 #1082: kt twin of `paged_cache`. `None` => behavior matches
    // `model_forward_paged` exactly; `Some(&kt)` => every paged-KV write
    // inside the full-attention layers mirrors into the kt cache.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Tensor> {
    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        #[cfg(feature = "cuda")]
        kt_paged_cache,
        LmHeadMode::Full,
    )?;
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// Paged-KV forward pass that returns the post-final-RMSNorm hidden state at
/// every position, skipping the LM head matmul entirely.
///
/// Used by the GRPO reference-forward path that shares the prompt's K/V across
/// all completions in a group: forward the prompt once with `start_pos == 0`,
/// snapshot the linear state, then forward each completion's tokens with
/// `start_pos == prompt_len` so the paged cache's prompt K/V is reused for
/// cross-attention. The returned hidden state feeds
/// `chunked_log_probs_for_completion` directly (avoids the full LM head over
/// every position when only completion log-probs are needed).
///
/// Returns hidden tensor with shape `[1, seq_len, hidden_size]`.
pub fn model_forward_paged_normed_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let (_logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        None,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::HiddenOnly,
    )?;
    let hidden = hidden.expect("LmHeadMode::HiddenOnly always returns hidden");
    let normed = {
        kiln_nvtx::range!(c"kiln/final_norm");
        rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
    };
    Ok(normed)
}

/// Paged-KV prefill that returns only the last pre-final-RMSNorm hidden row,
/// skipping the LM head matmul entirely.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_last_token_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let seq_len = token_ids.len();
    if seq_len == 0 {
        anyhow::bail!("model_forward_paged_last_token_hidden requires at least one token");
    }
    let (_logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        None,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::HiddenOnly,
    )?;
    let hidden = hidden.expect("LmHeadMode::HiddenOnly always returns hidden");
    hidden
        .narrow(1, seq_len - 1, 1)?
        .contiguous()
        .context("paged last-token hidden contiguous")
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    // Native single-submit Vulkan-resident decode fast-path. Records
    // all 32 layers into one `CommandBatch` and submits once. Falls
    // back transparently when not feasible.
    #[cfg(feature = "vulkan")]
    {
        if token_ids.len() == 1
            && start_pos > 0
            && lora.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && config.attn_output_gate
            && vulkan_resident_decode_enabled()
            && ReplayBackend::runtime_supports_resident_decode(backend)
            && resident_decode_pool_ready(backend, config)
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(logits) = model_forward_paged_last_token_resident_native_vk(
                    vk_backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    start_pos,
                    linear_state.as_deref(),
                )? {
                    return Ok(logits);
                }
            }
        }

        if native_resident_decode_required(backend, token_ids, &[start_pos], config, lora) {
            anyhow::bail!(
                "last-token decode declined native resident path; \
                 generic fallback disabled by backend policy"
            );
        }
    }

    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowOnly,
    )?;
    Ok(logits.expect("LmHeadMode::LastRowOnly always produces logits"))
}

/// Qualification-only eager forward that retains the final row at the
/// embedding, every transformer-layer output, and the final RMSNorm.
///
/// The returned snapshots are F32 device tensors ordered as
/// `embedding`, `layer_0` through `layer_N-1`, then `final_norm`. Production
/// serving does not call this synchronization-free diagnostic surface.
#[cfg(feature = "rocm")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_last_token_with_layer_snapshots(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<(Tensor, Vec<Tensor>)> {
    let mut snapshots = Vec::with_capacity(weights.layers.len() + 2);
    let embedding = embedding_lookup_from_weights(token_ids, weights)?.unsqueeze(0)?;
    snapshots.push(qualification_layer_last_row(&embedding, token_ids.len())?);
    let mut resume = None;
    let mut final_hidden = None;
    for layer_index in 0..weights.layers.len() {
        let final_layer = layer_index + 1 == weights.layers.len();
        let progress = model_forward_paged_inner_bounded(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_table,
            start_pos,
            linear_state.as_deref_mut(),
            lora,
            None,
            None,
            None,
            if final_layer {
                LmHeadMode::HiddenOnly
            } else {
                LmHeadMode::LastRowOnly
            },
            resume.take(),
            1,
        )?;
        if final_layer {
            final_hidden = progress.hidden;
        } else {
            let state = progress
                .state
                .context("layer snapshot forward did not retain its next layer")?;
            snapshots.push(qualification_layer_last_row(
                &state.hidden,
                token_ids.len(),
            )?);
            resume = Some(state);
        }
    }
    let final_hidden = final_hidden.context("layer snapshot forward returned no final hidden")?;
    snapshots.push(qualification_layer_last_row(
        &final_hidden,
        token_ids.len(),
    )?);
    let last = final_hidden.narrow(1, token_ids.len() - 1, 1)?;
    let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
    snapshots.push(qualification_layer_last_row(&normed, 1)?);
    let logits =
        lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?;
    anyhow::ensure!(
        snapshots.len() == weights.layers.len() + 2,
        "layer snapshot count is inconsistent"
    );
    Ok((logits, snapshots))
}

#[cfg(feature = "rocm")]
pub(super) fn qualification_layer_last_row(hidden: &Tensor, seq_len: usize) -> Result<Tensor> {
    hidden
        .narrow(1, seq_len - 1, 1)?
        .to_dtype(DType::F32)?
        .contiguous()
        .context("qualification layer last row contiguous")
}

/// Vulkan-resident decode entry-point. Same signature as
/// [`model_forward_paged_last_token`]; routes through the Vulkan-resident
/// dispatchers when the backend supports it AND the per-step buffer pool
/// is feasible. Resident decode declines are visible by default; use the
/// backend-owned decode hot-path fallback env for explicit A/B fallback.
///
/// Gate (a)/(c) of `docs/vk_resident_decode_plan.md`. The runtime predicate
/// `Backend::supports_resident_decode()` returns `false` on CPU / CUDA /
/// Metal so those backends keep using the existing `model_forward_paged_last_token`
/// path. On Vulkan the predicate returns `true` when the logical device is
/// up; pool feasibility is checked the first time this fn is called and
/// cached on the backend.
///
/// This entry point is a strict superset of `model_forward_paged_last_token`:
/// the resident path is a fast-path overlay for non-Vulkan backends. On Vulkan,
/// native-eligible decode failures stop instead of silently moving to a slower
/// generic route, so occupancy regressions are caught where they happen.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_last_token_resident(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    // Resident path requires backend support AND the buffer pool to fit.
    // Only consumed on `#[cfg(feature = "vulkan")]` below; the allow
    // silences the unused-variable warning on CUDA-only builds.
    #[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
    let route_resident = ReplayBackend::runtime_supports_resident_decode(backend)
        && resident_decode_pool_ready(backend, config);

    // Native single-submit orchestrator: chains all 32 layers' dispatches
    // into one `CommandBatch`, eliminating per-layer Tensor bridging and
    // submit overhead. Governed by the Vulkan policy; falls back
    // transparently to the per-layer fast-path embedded
    // in `model_forward_paged_inner` on any decline.
    #[cfg(feature = "vulkan")]
    {
        if route_resident
            && token_ids.len() == 1
            && start_pos > 0
            && lora.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && config.attn_output_gate
            && vulkan_resident_decode_enabled()
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(logits) = model_forward_paged_last_token_resident_native_vk(
                    vk_backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    start_pos,
                    linear_state.as_deref(),
                )? {
                    return Ok(logits);
                }
            }
        }
    }

    model_forward_paged_last_token(
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
    )
}

/// Native single-submit Vulkan-resident decode forward.
///
/// Records all 32 layer blocks' dispatches into one `CommandBatch`,
/// alternating between two pool buffers for layer-to-layer x; pre-seeds
/// any first-use KV-pool / GDN-state layers, pre-uploads RoPE/block_table/
/// seq_lens, submits once, reads back the final hidden state, and runs
/// the final RMSNorm + LM head through the legacy path (cheap one-shot).
///
/// Returns `Ok(None)` on any unsupported configuration so the caller
/// falls back to `model_forward_paged_inner` bit-identically.
#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn model_forward_paged_last_token_resident_native_vk(
    vk_backend: &crate::backend::vulkan::VulkanBackend,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&LinearAttentionState>,
) -> Result<Option<Tensor>> {
    use kiln_vulkan_kernel::CommandBatch;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    let timing_enabled =
        kiln_vulkan_kernel::kernels::vulkan_kernel_policy().profile_resident_decode_timing;
    static EMBED_NS: AtomicU64 = AtomicU64::new(0);
    static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
    static SEED_NS: AtomicU64 = AtomicU64::new(0);
    static RECORD_NS: AtomicU64 = AtomicU64::new(0);
    static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    static READBACK_NS: AtomicU64 = AtomicU64::new(0);
    static LMHEAD_NS: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    let t0 = std::time::Instant::now();
    let Some(vk_device) = vk_backend.vulkan_device() else {
        return Ok(None);
    };
    let Some(state) = linear_state else {
        return Ok(None);
    };
    if token_ids.len() != 1 {
        return Ok(None);
    }
    let hidden_size = config.hidden_size;
    let device = weights.embed_tokens.device();

    // 1. Token embedding is recorded into the main command batch.
    if timing_enabled {
        EMBED_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_rope = std::time::Instant::now();

    // 2. RoPE tables are recorded into the main command batch.
    if timing_enabled {
        ROPE_NS.fetch_add(t_rope.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_upload = std::time::Instant::now();

    let block_tables = [block_table];
    let start_positions = [start_pos];
    let step = crate::vk_decode_resident::prepare_batched_resident_decode_token_step_buffers(
        vk_backend,
        token_ids,
        hidden_size,
        config.rotary_dim(),
    )?;
    let meta = crate::vk_decode_resident::prepare_batched_resident_decode_meta_buffers(
        vk_backend,
        &block_tables,
        &start_positions,
        paged_cache.block_size(),
    )?;
    if timing_enabled {
        UPLOAD_NS.fetch_add(t_upload.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_seed = std::time::Instant::now();

    // 6. VkPagedKvCache (size matches paged cache).
    let vk_kv_cache_arc = match vk_backend.vk_paged_kv_cache(
        config.num_full_attention_layers,
        paged_cache.num_blocks(),
        paged_cache.block_size(),
        config.num_kv_heads,
        config.head_dim,
    ) {
        Some(c) => c.clone(),
        None => return Ok(None),
    };
    let vk_kv_cache: &kiln_vulkan_kernel::VkPagedKvCache = &vk_kv_cache_arc;

    // Detect a fresh request via `start_pos` continuity. Within one
    // request the resident decode advances by 1 per token; a jump
    // (boot, or a new /v1/chat/completions whose first decode step
    // doesn't follow the previous request's last) clears the seeded
    // sets so we re-seed from this request's prefill rather than
    // reusing the prior request's stale K/V data.
    vk_backend.note_resident_session(start_pos);

    // 7. Pre-seed any first-use full-attn layers from the kt paged
    // cache. The per-block seed is bounded by `block_table` so a fresh
    // request copies ~64 KB × num_blocks_used per layer (≈ 1 MB total
    // on Qwen3.5-4B with a 32-token prompt) instead of the multi-GB
    // full-pool slab.
    let active_blocks = block_table.blocks.as_slice();
    let mut full_attn_idx: usize = 0;
    for layer in weights.layers.iter() {
        if let GpuAttentionWeights::Full(_) = &layer.attention {
            if !vk_backend.full_attn_layer_seeded(full_attn_idx) {
                crate::vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_kt(
                    vk_device,
                    vk_kv_cache,
                    paged_cache,
                    full_attn_idx,
                    active_blocks,
                )?;
                vk_backend.mark_full_attn_layer_seeded(full_attn_idx);
            }
            full_attn_idx += 1;
        }
    }

    if timing_enabled {
        SEED_NS.fetch_add(t_seed.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_record = std::time::Instant::now();

    // 8. Build ONE CommandBatch with token embedding plus all layer
    // blocks recorded.
    let mut batch = CommandBatch::new(vk_device)?;
    let Some(final_in_input) =
        crate::vk_decode_resident::record_transformer_stack_batched_hidden_from_tokens_into(
            vk_backend,
            &mut batch,
            token_ids,
            &step.token_ids,
            &step.input,
            &step.scratch,
            weights,
            config,
            1,
            meta.max_blocks_per_seq,
            meta.block_size,
            vk_kv_cache,
            &step.rope_cos,
            &step.rope_sin,
            &meta.block_table,
            &meta.seq_lens,
            &meta.slots,
            state.recurrent_states.as_slice(),
            state.conv_states.as_slice(),
        )?
    else {
        return Ok(None);
    };
    let from_buf = if final_in_input {
        &step.input
    } else {
        &step.scratch
    };

    // Fold final RMSNorm + LM head GEMM into the same CommandBatch —
    // no intermediate readback or Tensor bridge between the last
    // transformer block and the lm_head. The legacy fast-path was
    // costing ~12 ms / token; baking these two dispatches into the
    // batch turns that into ~5 ms of pure GPU compute on the same
    // queue submission.
    // (#1082) kt-keyed weight caches: extract bytes straight from kt storage,
    // no full candle copy of the weight (the lm_head/embed_tokens_t is 778 MB —
    // a per-model candle copy was pure memory waste on a unified-memory APU).
    let final_norm_buf = vk_backend.cached_f32_weight_buffer_kt(&weights.final_norm)?;
    let lm_head_w_buf = vk_backend.cached_bf16_packed_weight_buffer_kt(&weights.embed_tokens_t)?;
    let vocab_size = weights.embed_tokens_t.dims().last().copied().unwrap_or(0);
    if vocab_size == 0 {
        return Ok(None);
    }
    let normed_final_buf =
        vk_backend.acquire_resident_scratch("native_final_normed", (hidden_size * 4) as u64)?;
    let logits_buf =
        vk_backend.acquire_resident_scratch("native_logits", (vocab_size * 4) as u64)?;
    // Final RMSNorm: from_buf → normed_final_buf
    batch.record_shader(
        kiln_vulkan_kernel::shaders::QWEN_RMSNORM_FORWARD,
        &[
            from_buf.handle(),
            final_norm_buf.handle(),
            normed_final_buf.handle(),
        ],
        &[
            1u32,
            hidden_size as u32,
            (config.rms_norm_eps as f32).to_bits(),
        ],
        kiln_vulkan_kernel::Workgroups::OneD(1),
    )?;
    // LM head GEMM (bf16w b=1): normed_final_buf → logits_buf
    batch.record_shader(
        // LM head out_dim = vocab_size (151936 for Qwen3.5-4B) — the wide
        // 64-col variant gives ~2374 workgroups (full SM saturation) AND
        // 100% cache-line utilization per warp memory read.
        kiln_vulkan_kernel::shaders::LINEAR_DECODE_BF16W_WIDE,
        &[
            normed_final_buf.handle(),
            lm_head_w_buf.handle(),
            logits_buf.handle(),
        ],
        &[hidden_size as u32, vocab_size as u32],
        kiln_vulkan_kernel::Workgroups::OneD(vocab_size.div_ceil(16) as u32),
    )?;

    if timing_enabled {
        RECORD_NS.fetch_add(t_record.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_submit = std::time::Instant::now();

    // Submit + wait — one queue submission covers all 32 layer blocks
    // PLUS the final norm + lm_head.
    // Fold the logits readback into the main batch: record a
    // cmd_copy_buffer from the device-local logits buffer into a
    // persistent host-visible staging buffer. After
    // submit_and_wait, we just `map_memory` on the staging buffer —
    // no separate queue submission for the readback.
    let logits_bytes_len = (vocab_size * 4) as u64;
    let logits_staging = vk_backend
        .acquire_resident_scratch_host_visible("native_logits_staging", logits_bytes_len)?;
    batch
        .record_copy_buffer(&logits_buf, &logits_staging, logits_bytes_len)
        .context("native: record logits copy to staging")?;

    batch
        .submit_and_wait("vk-resident native full-token forward")
        .context("native: submit full-token CommandBatch")?;
    if timing_enabled {
        SUBMIT_NS.fetch_add(t_submit.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_readback = std::time::Instant::now();

    // Map the staging buffer (already populated by the batch's
    // recorded cmd_copy_buffer above).
    let out_bytes = logits_staging
        .read_mapped(logits_bytes_len as usize)
        .context("native: map logits staging buffer")?;
    let n = out_bytes.len() / 4;
    let mut out_f32: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 4];
        b.copy_from_slice(&out_bytes[i * 4..i * 4 + 4]);
        out_f32.push(f32::from_le_bytes(b));
    }
    // Logits are produced as f32 by the bf16w GEMM; keep them as f32
    // to avoid a wasteful to_dtype() conversion (the caller's argmax
    // / greedy_sample works in f32 either way).
    let logits = Tensor::from_vec_on(device, out_f32, vec![1usize, 1usize, vocab_size])?;
    if timing_enabled {
        READBACK_NS.fetch_add(t_readback.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_lmhead = std::time::Instant::now();

    // (the rms_norm + lm_head are now part of the main batch; this
    // phase just exists to keep the phase timing layout stable)
    if timing_enabled {
        LMHEAD_NS.fetch_add(t_lmhead.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        // Print every call so each token's breakdown is visible.
        let ms = |ns: u64| (ns as f64) / 1e6;
        eprintln!(
            "[vk-native-phase] call={calls} embed={:.2} rope={:.2} upload={:.2} seed={:.2} record={:.2} submit={:.2} readback={:.2} lmhead={:.2} (ms; cumulative across all native calls)",
            ms(EMBED_NS.load(Ordering::Relaxed)),
            ms(ROPE_NS.load(Ordering::Relaxed)),
            ms(UPLOAD_NS.load(Ordering::Relaxed)),
            ms(SEED_NS.load(Ordering::Relaxed)),
            ms(RECORD_NS.load(Ordering::Relaxed)),
            ms(SUBMIT_NS.load(Ordering::Relaxed)),
            ms(READBACK_NS.load(Ordering::Relaxed)),
            ms(LMHEAD_NS.load(Ordering::Relaxed)),
        );
    }
    Ok(Some(logits))
}

/// Try the Vulkan-resident full-attention decode block. Returns
/// `Ok(Some(out))` when the resident path successfully produced this
/// layer's post-MLP residual; `Ok(None)` when the resident helper
/// declined (caller falls back to the nonresident block). Errors propagate.
///
/// On first use per layer per session, this seeds the resident KV pool
/// from the kt paged pool so any prefill K/V already written are
/// visible to subsequent resident attention reads. Once seeded, the
/// resident block writes per-step K/V into the VRAM pool only — the
/// kt paged pool is no longer authoritative for that layer.
#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_resident_block_full_attn_b1(
    vk_backend: &crate::backend::vulkan::VulkanBackend,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    inv_freq: &Tensor,
    rope_tables: Option<(&Tensor, &Tensor)>,
) -> Result<Option<Tensor>> {
    // The resident KV cache geometry must match the kt paged pool's.
    let Some(vk_kv_cache) = vk_backend.vk_paged_kv_cache(
        config.num_full_attention_layers,
        paged_cache.num_blocks(),
        paged_cache.block_size(),
        config.num_kv_heads,
        config.head_dim,
    ) else {
        return Ok(None);
    };

    // Build/derive RoPE cos/sin for the current single position. If the
    // outer loop already built rope_tables we slice the active row out;
    // otherwise we compute them here.
    let (cos, sin) = if let Some((c, s)) = rope_tables {
        // rope_tables is (num_positions, rotary_dim/2). For seq_len=1
        // they should be 1 row already.
        (c.clone(), s.clone())
    } else {
        rotary_tables_from_tensor(positions, inv_freq)?
    };

    // Seed this layer's resident K/V pool from the kt paged pool
    // on first use per session. `start_pos` continuity detects a fresh
    // request and clears the per-layer seeded flags so this layer
    // re-seeds from the new request's prefill instead of reusing the
    // previous request's slot data.
    let Some(vk_device) = vk_backend.vulkan_device() else {
        return Ok(None);
    };
    vk_backend.note_resident_session(start_pos);
    if !vk_backend.full_attn_layer_seeded(full_attn_layer_idx) {
        crate::vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_kt(
            vk_device,
            vk_kv_cache,
            paged_cache,
            full_attn_layer_idx,
            block_table.blocks.as_slice(),
        )?;
        vk_backend.mark_full_attn_layer_seeded(full_attn_layer_idx);
    }

    crate::vk_decode_resident::transformer_block_paged_decode_full_attn_resident_b1(
        vk_backend,
        x,
        layer,
        config,
        start_pos,
        block_table,
        full_attn_layer_idx,
        paged_cache,
        vk_kv_cache,
        &cos,
        &sin,
    )
}

/// First-use feasibility check for the Vulkan-resident decode pool.
///
/// Returns true when the backend's `decode_resident_pool_ready` predicate
/// confirms the buffer-pool ring fits in the device-local memory budget.
/// On CPU / CUDA / Metal the trait default returns false; only Vulkan
/// constructs (and caches) the pool here.
pub(super) fn resident_decode_pool_ready(
    backend: &dyn BackendRuntime,
    config: &kiln_core::config::ModelConfig,
) -> bool {
    // The pool is sized off (hidden, intermediate, max_batch). Max
    // batch defaults to 64 per docs/vk_resident_decode_plan.md gate
    // (b). At runtime, an iGPU near its UMA limit lands `None` and
    // routes to the per-call Tensor path.
    ReplayBackend::runtime_decode_resident_pool_ready(
        backend,
        config.hidden_size,
        config.intermediate_size,
        64,
    )
}

/// Paged-KV forward pass for greedy generation prefill.
///
/// This runs the same prefill work as [`model_forward_paged_last_token`] but
/// fuses the final-row LM-head projection with argmax when the backend supports
/// it, avoiding a `[1, 1, vocab_size]` logits tensor that greedy sampling would
/// immediately reduce.
pub fn model_forward_paged_last_token_greedy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<u32> {
    // Native single-submit Vulkan-resident decode fast-path. For the
    // hot decode loop the bench drives through this entry (since Vulkan
    // exposes `supports_linear_decode_argmax = true`), so the native
    // path has to land here too — wiring it only into
    // `model_forward_paged` / `_last_token` misses the production
    // generation loop entirely.
    #[cfg(feature = "vulkan")]
    {
        let block_tables = [block_table];
        let start_positions = [start_pos];
        if let Some(next_tokens) = try_vulkan_resident_batched_decode_argmax(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            &block_tables,
            &start_positions,
            None,
            linear_state.as_deref(),
            lora,
        )? {
            if let Some(&token) = next_tokens.first() {
                return Ok(token);
            }
        }

        if token_ids.len() == 1
            && start_pos > 0
            && lora.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && config.attn_output_gate
            && vulkan_resident_decode_enabled()
            && ReplayBackend::runtime_supports_resident_decode(backend)
            && resident_decode_pool_ready(backend, config)
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(logits) = model_forward_paged_last_token_resident_native_vk(
                    vk_backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    start_pos,
                    linear_state.as_deref(),
                )? {
                    // Materialize argmax on the returned logits.
                    let logits_flat = logits.flatten_all()?;
                    let logits_vec: Vec<f32> = logits_flat.to_vec1()?;
                    let mut best_idx = 0u32;
                    let mut best_val = f32::NEG_INFINITY;
                    for (i, &v) in logits_vec.iter().enumerate() {
                        if v > best_val {
                            best_val = v;
                            best_idx = i as u32;
                        }
                    }
                    return Ok(best_idx);
                }
            }
        }

        if native_resident_decode_required(backend, token_ids, &start_positions, config, lora) {
            anyhow::bail!(
                "greedy decode declined native resident path; \
                 generic fallback disabled by backend policy"
            );
        }
    }

    let (_logits, _hidden, token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowArgmaxOnly,
    )?;
    token.context("LmHeadMode::LastRowArgmaxOnly always produces a token")
}

/// Paged-KV single-token decode for greedy sampling.
///
/// This keeps the existing logits APIs intact but, on the Metal BF16 decode
/// path, fuses the LM-head projection with argmax so generation does not
/// materialize `[1, 1, vocab_size]` logits only to immediately reduce them.
pub fn model_forward_paged_next_token_greedy(
    backend: &dyn BackendRuntime,
    token_id: u32,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<u32> {
    model_forward_paged_last_token_greedy(
        backend,
        &[token_id],
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
    )
}

/// #1082 box-102 FIX — graph-capture twin of
/// [`model_forward_paged_with_graph_inputs`] that stops at the
/// **pre-final-norm** hidden state (`LmHeadMode::HiddenOnly`) instead of
/// projecting to logits.
///
/// The bs=1 paged-decode CUDA graph captures THIS forward (transformer blocks
/// only — paged-KV + GDN recurrent state still advance exactly as before), so
/// the large-N (`vocab = 151936`) cublasLt lm_head GEMV is NOT recorded into
/// the graph. The caller (`CudaGraphRunner::{try_capture, decode_step_paged}`)
/// runs `final_norm` + lm_head EAGERLY on the replayed hidden via
/// [`lm_head_from_hidden_eager`].
///
/// Why: replaying the captured lm_head matmul produced WRONG logits
/// (token-doubling, "BUG2") despite a bit-identical input hidden. The slot-40
/// sign-sum probe proved the lm_head INPUT matches eager-vs-replay on every
/// step while the OUTPUT logits diverge up to 97%; transformer blocks 0-31 also
/// match. The large-N cublasLt algo is not CUDA-graph-replay-deterministic; the
/// small per-layer matmuls are. Moving final_norm + lm_head out of the captured
/// region is the structural fix — the captured 32-layer transformer (the actual
/// decode win) is preserved.
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_hidden_with_graph_inputs(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: &Tensor,
    positions_gpu: &Tensor,
    graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
) -> Result<Tensor> {
    let (_logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        Some(token_ids_gpu),
        Some(positions_gpu),
        graph_inputs,
        // Phase 7 #1082: no kt twin from this caller — forward `None` so the
        // candle writer remains authoritative (mirrors the logits twin).
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::HiddenOnly,
    )?;
    Ok(hidden.expect("LmHeadMode::HiddenOnly always returns hidden"))
}

/// #1082 box-102 FIX — run the final RMSNorm + LM head projection EAGERLY
/// (outside any captured CUDA graph) on the **pre-final-norm** hidden state the
/// captured bs=1 decode graph produced ([`model_forward_paged_hidden_with_graph_inputs`]).
///
/// Numerically identical to the `LmHeadMode::Full` arm of
/// [`model_forward_paged_inner`] (same `final_norm` RMSNorm, same
/// `lm_head_forward_backend_decode_if`), so the logits match the eager decode
/// path exactly — but the lm_head cublasLt GEMV runs eagerly, off the graph,
/// sidestepping the replay-nondeterminism that doubled output ("BUG2"). Cost is
/// one extra eager vocab GEMV + one RMSNorm per decode token — negligible next
/// to the captured 32-layer transformer.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn lm_head_from_hidden_eager(
    backend: &dyn BackendRuntime,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/lm_head_eager");
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn lm_head_argmax_from_hidden_eager(
    backend: &dyn BackendRuntime,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<u32> {
    kiln_nvtx::range!(c"kiln/lm_head_argmax_eager");
    if let Some(token) =
        lm_head_weighted_prep_argmax(hidden, &weights.final_norm, &weights.embed_tokens_t)?
    {
        return Ok(token);
    }
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    #[cfg(feature = "rocm")]
    if let Some(lm_head_w8) = weights.lm_head_w8.as_ref() {
        if normed.dtype() == DType::BF16
            && !normed.track_op()
            && matches!(normed.device(), Device::Rocm(_))
        {
            let normed = normed
                .contiguous()
                .context("rocm w8 lm_head argmax normed contiguous")?;
            return crate::rocm_w8_proj::argmax_bf16(&normed, lm_head_w8)
                .context("rocm w8 lm_head argmax");
        }
    }
    lm_head_argmax_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn model_forward_paged_with_graph_inputs(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: &Tensor,
    positions_gpu: &Tensor,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
) -> Result<Tensor> {
    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        Some(token_ids_gpu),
        Some(positions_gpu),
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        graph_inputs,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::Full,
    )?;
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// #1082 boxes 432/433 (STEPS 2-3) — the batched HiddenOnly forward the
/// captured `bs > 1` decode graph records.
///
/// Stops BEFORE the final RMSNorm + lm_head: it writes the PRE-final-norm
/// transformer-stack hidden (`[batch, 1, hidden_size]`) into the caller-owned
/// stable [`BatchedPagedDecodeGraphInputs::output_hidden`] buffer via
/// `slice_set` and returns `Ok(())`. NO `rms_norm`, NO
/// `lm_head_forward_backend_decode_if` inside — `final_norm` + the large-N
/// (`vocab = 151936`) cublasLt lm_head GEMV run EAGERLY off the captured graph
/// via [`lm_head_from_batched_hidden_eager`].
///
/// This mirrors the bs=1 `HiddenOnly` contract
/// ([`model_forward_paged_hidden_with_graph_inputs`] →
/// `model_forward_paged_inner(..., LmHeadMode::HiddenOnly)`): the lm_head GEMV
/// is not CUDA-graph-replay-deterministic at large N, so moving it out of the
/// captured region is the structural fix for the "BUG2" token-doubling.
///
/// STEPS 2-3 wire this into
/// [`crate::cuda_graph::CudaGraphRunner::try_capture_batched`] /
/// `decode_step_paged_batched`, replacing the former in-graph logits twin.
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_batched_hidden_with_graph_inputs(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    sequence_lengths: &[usize],
    lora: Option<&LoraWeights>,
    graph_inputs: &mut BatchedPagedDecodeGraphInputs<'_>,
) -> Result<()> {
    // Run the bs>1 hidden path with the persistent linear-state slot and the
    // graph-stable token-id / position / block_table / seqused_k / per-layer
    // paged-decode scratch / rotary / kv_slot device buffers threaded through.
    // The captured graph reads the per-step paged-decode metadata from
    // caller-owned device tensors (`graph_inputs.block_table` +
    // `graph_inputs.seqused_k`), writes/reads the per-layer flash-attn
    // paged-decode `attn_out` + `softmax_lse` through caller-owned tensors,
    // reads its RoPE cos/sin tables from `graph_inputs.rotary_cos` /
    // `.rotary_sin` (refreshed before each replay via
    // `CudaGraphRunner::update_batched_rotary_buffers`), and dispatches the
    // fused batched-slot KV writer from `graph_inputs.kv_slot`. Any of these
    // allocated *inside* the captured region would otherwise `cudaFree` their
    // storage at end of capture, leaving the graph with dangling pointers (the
    // `ILLEGAL_ADDRESS` faults documented in
    // `bench-results/cuda-graph-bs2-memcheck.md` and suspects 1-4 in
    // `bench-results/cuda-graph-bs2-secondary-audit.md`, #1082). This twin
    // diverges from the eager batched forward ONLY in what it does with the
    // resulting `hidden`: it stops before final_norm + lm_head and writes the
    // hidden to the stable `output_hidden` buffer.
    let hidden = model_forward_paged_decode_contiguous_batch_hidden_inner(
        backend,
        input_tokens,
        weights,
        config,
        paged_cache,
        block_tables,
        sequence_lengths,
        Some(graph_inputs.linear_state),
        lora,
        Some(graph_inputs.positions),
        Some(graph_inputs.token_ids),
        Some(graph_inputs.block_table),
        Some(graph_inputs.seqused_k),
        Some(graph_inputs.attn_out),
        Some(graph_inputs.softmax_lse),
        Some(graph_inputs.rotary_cos),
        Some(graph_inputs.rotary_sin),
        Some(graph_inputs.kv_slot),
        #[cfg(feature = "metal")]
        None,
    )?;
    // #1082 box-102 BUG2 fix (batched): write the PRE-final-norm hidden into
    // the caller-owned stable `output_hidden` buffer (`[batch, 1, hidden]`)
    // and return. `final_norm` + lm_head run EAGERLY later via
    // `lm_head_from_batched_hidden_eager` — mirrors the bs=1 contract
    // (`lm_head_from_hidden_eager`). CRITICAL: nothing here may do a
    // device→host transfer (e.g. `to_vec1`, `to_scalar`, host-side argmax) —
    // a synchronous DtoH during CUDA stream capture is not recorded cleanly by
    // the driver (the first wiring attempt hit `CUDA_ERROR_ILLEGAL_ADDRESS` on
    // step 2). Argmax + DtoH is the caller's job, run outside the captured
    // region.
    #[cfg(feature = "rocm")]
    kiln_tensor::rocm_slice_set_dim0(graph_inputs.output_hidden, &hidden, 0)
        .context("copy ROCm graph-wrapper hidden into stable output_hidden buffer")?;
    #[cfg(not(feature = "rocm"))]
    graph_inputs
        .output_hidden
        .slice_set(&hidden, 0, 0)
        .context("copy graph-wrapper hidden into stable output_hidden buffer")?;
    Ok(())
}

/// #1082 boxes 432/433 (STEPS 2-3) — batched twin of
/// [`lm_head_from_hidden_eager`].
///
/// Runs the final RMSNorm + lm_head projection EAGERLY (outside any captured
/// CUDA graph) on the **pre-final-norm** batched hidden state the captured
/// batched decode graph produced into
/// [`BatchedPagedDecodeGraphInputs::output_hidden`] (via
/// [`model_forward_paged_batched_hidden_with_graph_inputs`]).
///
/// `output_hidden` is `[batch, 1, hidden_size]`; the returned logits are
/// `[batch, 1, vocab_size]`. The runner argmax-reduces per row to produce the
/// `[batch]` next-token IDs. Numerically identical to the eager batched
/// forward's tail (`rms_norm` + `lm_head_forward_backend_decode_if` on the same
/// `final_norm` / `embed_tokens_t`) — only the lm_head cublasLt GEMV now runs
/// eagerly off the graph, sidestepping the replay-nondeterminism that doubled
/// output ("BUG2"). Cost is one extra eager batched vocab GEMV + one RMSNorm
/// per decode step.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn lm_head_from_batched_hidden_eager(
    backend: &dyn BackendRuntime,
    output_hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/lm_head_eager_batched");
    // Same `final_norm` RMSNorm + lm_head as the bs=1 eager twin, but the
    // input/output carry the leading `batch` dim ([batch, 1, hidden] →
    // [batch, 1, vocab]). `rms_norm` normalizes over the last (hidden) axis
    // and `lm_head_forward_backend_decode_if` projects the last axis to
    // vocab, both of which are rank-agnostic over the leading dims, so no
    // reshape is needed (mirrors how the eager batched forward feeds its
    // `[batch, 1, hidden]` `hidden` straight through `rms_norm` +
    // `lm_head_forward_backend_decode_if`).
    let normed = rms_norm(output_hidden, &weights.final_norm, config.rms_norm_eps)?;
    lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)
}

/// Token-only sibling of [`lm_head_from_batched_hidden_eager`]. ROCm uses the
/// packed W8 batched LM-head/argmax pipeline when available, avoiding the
/// `[batch, vocab]` logits allocation and returning one token vector.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn lm_head_argmax_from_batched_hidden_eager(
    backend: &dyn BackendRuntime,
    output_hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Vec<u32>> {
    kiln_nvtx::range!(c"kiln/lm_head_argmax_eager_batched");
    let normed = rms_norm(output_hidden, &weights.final_norm, config.rms_norm_eps)?;
    match rocm_w8_lm_head_argmax_rows(backend, &normed, weights)? {
        Some(tokens) => Ok(tokens),
        None => {
            lm_head_argmax_rows_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)
        }
    }
}

/// Batched paged decode API for real continuous-batching work.
///
/// Keeps the existing [`PagedKvCache`] API and its caller-held mutex: each
/// request still has its own [`BlockTable`] and KV window, but the dominant
/// GDN/MLP layers run as one batch-shaped forward. Full-attention layers stay
/// row-wise because each request has distinct paged KV metadata; this avoids
/// the batch-8 paged-attention workspace blow-up while still removing the 24
/// GDN-layer row loop that made streaming throughput flat.
///
/// CUDA graphs are deliberately not used for `batch_size > 1` here; the graph
/// runner is currently captured for the batch-1 decode shape only. TODO(phase2
/// continuous batching): add graph capture/replay keyed by decode batch shape.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_batched_decode(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[BlockTable],
    sequence_lengths: &[usize],
    linear_states: &mut [&mut LinearAttentionState],
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let hidden = model_forward_paged_batched_decode_hidden(
        backend,
        input_tokens,
        weights,
        config,
        paged_cache,
        block_tables,
        sequence_lengths,
        linear_states,
        lora,
    )?;
    model_forward_head_backend_decode_if(Some(backend), &hidden, weights, config)
        .context("batched decode lm head")
}

/// Batched paged decode through the transformer stack, stopping before the
/// final LM head.
///
/// Returning `[batch, 1, hidden]` lets the caller choose between projecting the
/// whole batch with a backend-aware LM head or sampling rows independently when
/// bounded LM-head workspace is more important than projection throughput.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_batched_decode_hidden(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[BlockTable],
    sequence_lengths: &[usize],
    linear_states: &mut [&mut LinearAttentionState],
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let batch_size = input_tokens.len();
    anyhow::ensure!(batch_size > 0, "batched decode requires at least one token");
    anyhow::ensure!(
        block_tables.len() == batch_size,
        "batched decode block_tables length {} != input_tokens length {batch_size}",
        block_tables.len()
    );
    anyhow::ensure!(
        sequence_lengths.len() == batch_size,
        "batched decode sequence_lengths length {} != input_tokens length {batch_size}",
        sequence_lengths.len()
    );
    anyhow::ensure!(
        linear_states.len() == batch_size,
        "batched decode linear_states length {} != input_tokens length {batch_size}",
        linear_states.len()
    );

    if batch_size == 1 {
        let (_, hidden, _) = model_forward_paged_inner(
            backend,
            &[input_tokens[0]],
            weights,
            config,
            paged_cache,
            &block_tables[0],
            sequence_lengths[0],
            Some(&mut *linear_states[0]),
            lora,
            None,
            None,
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            None,
            // Phase 7 #1082: no kt twin from this caller — forward
            // `None` so the candle writer remains authoritative.
            #[cfg(feature = "cuda")]
            None,
            LmHeadMode::HiddenOnly,
        )?;
        return hidden.context("batched decode hidden skipped lm head");
    }

    let device = weights.embed_tokens.device();
    let mut hidden = embedding_lookup_from_weights(input_tokens, weights)?;
    hidden = hidden.unsqueeze(1)?;
    let use_metal_decode_ffn = sequence_lengths.iter().all(|&p| p > 0)
        && !crate::mtp_runtime::single_token_self_attention_active();

    let mut full_attn_idx = 0usize;
    let mut linear_attn_idx = 0usize;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(layer_idx).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Linear(lin_weights) => {
                let normed = {
                    kiln_nvtx::range!(c"kiln/batched_decode/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };

                // Defensive dtype normalization (same rationale as
                // LinearAttentionState::from_batch_rows): cast any drifted rows
                // back to row 0's dtype before cat, so a stray BF16 row from a
                // prior aborted decode does not break the slow path either.
                let mut recurrent_state = {
                    let target_dtype = linear_states[0].recurrent_states[linear_attn_idx].dtype();
                    let mut owned: Vec<Tensor> = Vec::with_capacity(linear_states.len());
                    for (row_idx, state) in linear_states.iter().enumerate() {
                        let t = &state.recurrent_states[linear_attn_idx];
                        if t.dtype() != target_dtype {
                            tracing::debug!(
                                layer = layer_idx,
                                row = row_idx,
                                from = ?t.dtype(),
                                to = ?target_dtype,
                                "batched_decode: normalizing recurrent state dtype before cat"
                            );
                            owned.push(t.to_dtype(target_dtype).with_context(|| {
                                format!(
                                    "cast recurrent state row {row_idx} to {target_dtype:?} for GDN layer {layer_idx}"
                                )
                            })?);
                        } else {
                            owned.push(t.clone());
                        }
                    }
                    let refs: Vec<&Tensor> = owned.iter().collect();
                    Tensor::cat(&refs, 0).with_context(|| {
                        format!("cat batched recurrent state for GDN layer {layer_idx}")
                    })?
                };
                let mut conv_state = {
                    let target_dtype = linear_states[0].conv_states[linear_attn_idx].dtype();
                    let mut owned: Vec<Tensor> = Vec::with_capacity(linear_states.len());
                    for (row_idx, state) in linear_states.iter().enumerate() {
                        let t = &state.conv_states[linear_attn_idx];
                        if t.dtype() != target_dtype {
                            tracing::debug!(
                                layer = layer_idx,
                                row = row_idx,
                                from = ?t.dtype(),
                                to = ?target_dtype,
                                "batched_decode: normalizing conv state dtype before cat"
                            );
                            owned.push(t.to_dtype(target_dtype).with_context(|| {
                                format!(
                                    "cast conv state row {row_idx} to {target_dtype:?} for GDN layer {layer_idx}"
                                )
                            })?);
                        } else {
                            owned.push(t.clone());
                        }
                    }
                    let refs: Vec<&Tensor> = owned.iter().collect();
                    Tensor::cat(&refs, 0).with_context(|| {
                        format!("cat batched conv state for GDN layer {layer_idx}")
                    })?
                };

                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut recurrent_state,
                    &mut conv_state,
                    true,
                    false,
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| format!("batched GDN layer {layer_idx}"))?;

                for (row_idx, state) in linear_states.iter_mut().enumerate() {
                    state.recurrent_states[linear_attn_idx] =
                        LinearAttentionState::detached_batch_row(&recurrent_state, row_idx)
                            .with_context(|| {
                                format!(
                                    "split recurrent state row {row_idx} for GDN layer {layer_idx}"
                                )
                            })?;
                    state.conv_states[linear_attn_idx] =
                        LinearAttentionState::detached_batch_row(&conv_state, row_idx)
                            .with_context(|| {
                                format!("split conv state row {row_idx} for GDN layer {layer_idx}")
                            })?;
                }

                hidden = {
                    kiln_nvtx::range!(c"kiln/batched_decode/residual/attn");
                    residual_add(hidden, attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/batched_decode/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/batched_decode/residual/mlp");
                    residual_add(hidden, ffn_out)?
                };
                linear_attn_idx += 1;
            }
            GpuAttentionWeights::Full(_) => {
                let positions_f32: Vec<f32> = sequence_lengths.iter().map(|&p| p as f32).collect();
                let positions = Tensor::from_vec_on(device, positions_f32, vec![batch_size])?;
                let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
                match transformer_block_paged_decode_contiguous_batch(
                    backend,
                    &hidden,
                    layer,
                    config,
                    &positions,
                    sequence_lengths,
                    &weights.rotary_inv_freq,
                    paged_cache,
                    &block_table_refs,
                    full_attn_idx,
                    layer_lora,
                    None,
                    None,
                    None,
                    None,
                    #[cfg(feature = "metal")]
                    None,
                    #[cfg(feature = "cuda")]
                    None,
                ) {
                    Ok(out) => hidden = out,
                    Err(err) => {
                        if native_decode_attention_required(backend) {
                            return Err(err).with_context(|| {
                                format!(
                                    "batched full-attention decode layer {layer_idx} declined; \
                                     rowwise fallback disabled by backend policy"
                                )
                            });
                        }
                        tracing::debug!(
                            layer = layer_idx,
                            error = %err,
                            "batched full-attention decode declined; falling back to rowwise"
                        );
                        let mut rows = Vec::with_capacity(batch_size);
                        for row_idx in 0..batch_size {
                            let row_hidden = hidden.narrow(0, row_idx, 1)?.contiguous()?;
                            let row_position = Tensor::from_vec_on(
                                device,
                                vec![sequence_lengths[row_idx] as f32],
                                vec![1],
                            )?;
                            let row = transformer_block_paged(
                                backend,
                                &row_hidden,
                                layer,
                                config,
                                &row_position,
                                sequence_lengths[row_idx],
                                config.num_attention_heads,
                                config.num_kv_heads,
                                config.head_dim,
                                config.rotary_dim(),
                                &weights.rotary_inv_freq,
                                config.rms_norm_eps,
                                paged_cache,
                                &block_tables[row_idx],
                                full_attn_idx,
                                layer_lora,
                            )
                            .with_context(|| {
                                format!(
                                    "rowwise fallback transformer block {layer_idx} row {row_idx} (full attention, paged)"
                                )
                            })?;
                            rows.push(row);
                        }
                        let row_refs: Vec<&Tensor> = rows.iter().collect();
                        hidden = Tensor::cat(&row_refs, 0).with_context(|| {
                            format!("cat rowwise fallback transformer block {layer_idx} outputs")
                        })?;
                    }
                }
                full_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin from this caller — forward
        // `None` so the candle writer remains authoritative.
        #[cfg(feature = "cuda")]
        None,
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
    mtp_cache: &PagedKvCache,
    mtp_block_table: &BlockTable,
    base_pos: usize,
    mtp_pos: usize,
    lora: Option<&crate::lora_loader::LoraWeights>,
) -> Result<(Tensor, Tensor)> {
    kiln_nvtx::range!(c"kiln/mtp/step");
    let mtp = weights.mtp_weights()?;
    let device = weights.embed_tokens.device();

    // 1. Token embedding for the draft token. `embedding_lookup` returns
    //    shape [1, H]; unsqueeze to [1, 1, H] to match transformer-block I/O.
    let token_ids = [draft_token_id];
    let token_emb = embedding_lookup_from_weights(&token_ids, weights)?; // [1, H]
    let token_emb = token_emb.unsqueeze(0)?; // [1, 1, H]

    // 2-3. Dual RMSNorms. `h_prev` is [1, 1, H] pre-final-norm.
    let norm_emb = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_emb");
        rms_norm(&token_emb, &mtp.pre_fc_norm_embedding, config.rms_norm_eps)?
    };
    let norm_h = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
        rms_norm(h_prev, &mtp.pre_fc_norm_hidden, config.rms_norm_eps)?
    };

    // 4. Concat along the hidden dim and fuse: [1, 1, 2H] @ fc_t[2H, H] -> [1, 1, H]
    //
    // Phase 7 (#1082): for rank-3 tensors axis-2 is the last dim, so
    // route the cat through `try_kt_concat_last_dim`. Both pieces are
    // produced by the RMSNorm above and are contiguous CUDA tensors of
    // matching dtype + rank; the helper's preconditions are satisfied
    // on the production decode path. The follow-up `.contiguous()?` is
    // kept because `try_kt_concat_last_dim` outputs are already
    // contiguous but callers downstream rely on the layout assertion.
    let concat = {
        #[cfg(feature = "cuda")]
        {
            let pieces: [&Tensor; 2] = [&norm_emb, &norm_h];
            match try_kt_concat_last_dim(&pieces)? {
                Some(out) => out,
                None => Tensor::cat(&[&norm_emb, &norm_h], 2)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Tensor::cat(&[&norm_emb, &norm_h], 2)?
        }
    }
    .contiguous()?;
    let fused = {
        kiln_nvtx::range!(c"kiln/mtp/fc");
        runtime_matmul_or_broadcast(backend, &concat, &mtp.fc_t)?
    };

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
    let mtp_hidden = {
        let _attention_scope = crate::mtp_runtime::MtpAttentionScope::enter();
        transformer_block_paged(
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
            // The adapter's MTP draft-block LoRA, when the adapter was
            // trained with MTP alignment. Absent means base draft weights;
            // verification remains exact while acceptance may degrade.
            lora.and_then(|l| l.mtp.as_ref().map(|m| (m, l.scale))),
        )
        .context("mtp transformer block")?
    };

    // 6. Final RMSNorm + weight-tied LM head.
    let normed = {
        kiln_nvtx::range!(c"kiln/mtp/final_layernorm");
        rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?
    };
    let logits = {
        kiln_nvtx::range!(c"kiln/mtp/lm_head");
        lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
    };

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
pub(super) enum LmHeadMode {
    /// Compute the LM head over every position. Result has shape
    /// `[1, seq_len, vocab_size]`. This is the legacy `model_forward_paged`
    /// behaviour and the only mode used by training / parity verification.
    Full,
    /// Compute the LM head over the final token only. Result has shape
    /// `[1, 1, vocab_size]`. Numerically identical to slicing the last row
    /// of `Full` because RMSNorm is per-position and the matmul reduces
    /// along `hidden_size` only.
    LastRowOnly,
    /// Compute the final token's greedy argmax without materializing logits
    /// when a backend-specific fused head supports it. Used only by greedy
    /// single-token decode.
    LastRowArgmaxOnly,
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
    /// Skip RMSNorm + LM head but return the final hidden state. Used by the
    /// batched decode actor so it can project/sample rows with bounded LM-head
    /// workspace after the batch-shaped transformer pass.
    HiddenOnly,
}

/// GPU ownership retained while one token chunk yields between transformer
/// layer groups. The linear-attention and paged-KV state remain owned by the
/// caller; this state carries only the forward-local tensors and layer cursors.
pub(crate) struct PagedLayerForwardState {
    hidden: Tensor,
    positions: Tensor,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
    next_layer: usize,
    full_attn_idx: usize,
    linear_attn_idx: usize,
}

/// Result of one bounded layer group for last-row paged prefill.
pub(crate) struct PagedLayerForwardProgress {
    pub(crate) logits: Option<Tensor>,
    pub(crate) state: Option<PagedLayerForwardState>,
    pub(crate) layers_processed: usize,
}

pub(super) struct PagedForwardProgress {
    logits: Option<Tensor>,
    hidden: Option<Tensor>,
    token: Option<u32>,
    state: Option<PagedLayerForwardState>,
    layers_processed: usize,
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
pub(super) fn model_forward_paged_inner(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: Option<&Tensor>,
    positions_gpu: Option<&Tensor>,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
    // Phase 7 #1082: kt twin of `paged_cache` plumbed through to the
    // per-layer `transformer_block_paged_with_rope_tables` so the kt
    // cache can mirror the CUDA-graph paged-KV write performed inside
    // `gqa_attention_paged_with_rope_tables`. `None` keeps the candle
    // writer authoritative — same gating playbook as `graph_inputs`.
    // CUDA-gated since `PagedKvCacheKt` itself is CUDA-only.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    lm_head_mode: LmHeadMode,
) -> Result<(Option<Tensor>, Option<Tensor>, Option<u32>)> {
    let progress = model_forward_paged_inner_bounded(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        token_ids_gpu,
        positions_gpu,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        graph_inputs,
        #[cfg(feature = "cuda")]
        kt_paged_cache,
        lm_head_mode,
        None,
        usize::MAX,
    )?;
    debug_assert!(progress.state.is_none());
    Ok((progress.logits, progress.hidden, progress.token))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn model_forward_paged_inner_bounded(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: Option<&Tensor>,
    positions_gpu: Option<&Tensor>,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    lm_head_mode: LmHeadMode,
    resume: Option<PagedLayerForwardState>,
    max_layers: usize,
) -> Result<PagedForwardProgress> {
    anyhow::ensure!(max_layers > 0, "paged layer quantum must be positive");
    let seq_len = token_ids.len();
    let device = weights.embed_tokens.device();

    let (
        mut hidden,
        positions_owned,
        rope_tables_owned,
        layer_start,
        mut full_attn_idx,
        mut linear_attn_idx,
    ) = if let Some(resume) = resume {
        anyhow::ensure!(
            token_ids_gpu.is_none() && positions_gpu.is_none(),
            "resumed paged forward cannot replace retained token or position tensors"
        );
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        anyhow::ensure!(
            graph_inputs.is_none(),
            "graph-backed paged forward cannot yield between layers"
        );
        #[cfg(feature = "cuda")]
        anyhow::ensure!(
            kt_paged_cache.is_none(),
            "CUDA twin-cache paged forward cannot yield between layers"
        );
        (
            resume.hidden,
            Some(resume.positions),
            Some((resume.rotary_cos, resume.rotary_sin)),
            resume.next_layer,
            resume.full_attn_idx,
            resume.linear_attn_idx,
        )
    } else {
        // 1. Embedding lookup: [seq_len, hidden_size]
        let hidden = match token_ids_gpu {
            Some(index) => embedding_lookup_from_weights_with_index(index, weights)?,
            None => embedding_lookup_from_weights(token_ids, weights)?,
        }
        .unsqueeze(0)?;

        // Phase B11b tap: `tok_embed`. Output of `embed_tokens(input_ids)`
        // with a leading batch dim. A resumed layer group must not repeat it.

        let positions_owned = if positions_gpu.is_none() {
            let pos_f32: Vec<f32> = (start_pos..start_pos + seq_len)
                .map(|position| position as f32)
                .collect();
            Some(Tensor::new(pos_f32.as_slice(), device)?)
        } else {
            None
        };
        let positions = positions_gpu
            .or(positions_owned.as_ref())
            .context("paged forward positions are missing")?;
        let graph_rope_tables = {
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                graph_inputs.map(|inputs| (inputs.rotary_cos, inputs.rotary_sin))
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                Option::<(&Tensor, &Tensor)>::None
            }
        };
        let rope_tables_owned = if positions_gpu.is_none() && graph_rope_tables.is_none() {
            Some(rotary_tables_from_tensor(
                positions,
                &weights.rotary_inv_freq,
            )?)
        } else {
            None
        };
        (hidden, positions_owned, rope_tables_owned, 0, 0, 0)
    };

    // Position tensor for RoPE — graph callers retain their preallocated
    // tensor, while resumable prefill owns the generated tensor in its state.
    let positions = positions_gpu
        .or(positions_owned.as_ref())
        .context("paged forward positions are missing")?;
    let graph_rope_tables = {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            graph_inputs.map(|inputs| (inputs.rotary_cos, inputs.rotary_sin))
        }
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        {
            Option::<(&Tensor, &Tensor)>::None
        }
    };
    let rope_tables = graph_rope_tables.or_else(|| {
        rope_tables_owned
            .as_ref()
            .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor))
    });

    // 2. Loop through all transformer layers
    let layer_end = layer_start
        .saturating_add(max_layers)
        .min(weights.layers.len());
    for (i, layer) in weights
        .layers
        .iter()
        .enumerate()
        .take(layer_end)
        .skip(layer_start)
    {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                hidden = transformer_block_paged_with_rope_tables(
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
                    #[cfg(any(feature = "cuda", feature = "rocm"))]
                    graph_inputs,
                    // Phase 7 #1082: forward the inner-fn kt twin
                    // parameter down to the per-layer block call so the
                    // kt cache mirrors the candle CUDA-graph paged-KV
                    // write. `None` when the gate is off or the caller
                    // hasn't migrated; the candle writer is still
                    // authoritative either way.
                    #[cfg(feature = "cuda")]
                    kt_paged_cache,
                )
                .with_context(|| format!("transformer block {i} (full attention, paged)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                // Vulkan-resident full-block GDN fast-path. Gates: seq_len=1
                // decode hot path, start_pos > 0, no MTP, no LoRA, and
                // qualified resident decode enabled.
                // Bypasses the legacy pre-norm/residual/post-norm/MLP/final-residual
                // candle path entirely when active.
                #[cfg(feature = "vulkan")]
                {
                    if seq_len == 1
                        && start_pos > 0
                        && lora.is_none()
                        && !crate::mtp_runtime::single_token_self_attention_active()
                        && vulkan_resident_decode_enabled()
                    {
                        if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                            .downcast_ref::<crate::backend::vulkan::VulkanBackend>(
                        ) {
                            let recurrent_t = &state.recurrent_states[linear_attn_idx];
                            let conv_t = &state.conv_states[linear_attn_idx];
                            if let Some(out) =
                                crate::vk_decode_resident::transformer_block_paged_decode_gdn_resident_b1_kt(
                                    vk_backend,
                                    &hidden,
                                    layer,
                                    config,
                                    recurrent_t,
                                    conv_t,
                                )?
                            {
                                hidden = out;
                                linear_attn_idx += 1;
                                continue;
                            }
                        }
                    }
                }
                let use_metal_decode_ffn = seq_len == 1
                    && start_pos > 0
                    && !crate::mtp_runtime::single_token_self_attention_active();
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                let attn_out = {
                    let _prefill_resident_layer_scope =
                        GdnPrefillResidentStateLayerScope::new(backend, linear_attn_idx);
                    gated_deltanet_forward_decode_if(
                        backend,
                        &normed,
                        lin_weights,
                        config,
                        &mut state.recurrent_states[linear_attn_idx],
                        &mut state.conv_states[linear_attn_idx],
                        true,
                        use_metal_decode_ffn,
                        true,
                        true,
                        layer_lora,
                    )
                    .with_context(|| {
                        format!("gated deltanet layer {i} (linear attention, paged)")
                    })?
                };
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    residual_add(hidden, ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    let layers_processed = layer_end.saturating_sub(layer_start);
    if layer_end < weights.layers.len() {
        anyhow::ensure!(
            matches!(lm_head_mode, LmHeadMode::LastRowOnly),
            "only last-row paged prefill may yield between transformer layers"
        );
        anyhow::ensure!(
            token_ids_gpu.is_none() && positions_gpu.is_none(),
            "layer-yielding paged prefill must own token and position inputs"
        );
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        anyhow::ensure!(
            graph_inputs.is_none(),
            "graph-backed paged forward cannot yield between layers"
        );
        #[cfg(feature = "cuda")]
        anyhow::ensure!(
            kt_paged_cache.is_none(),
            "CUDA twin-cache paged forward cannot yield between layers"
        );
        let positions = positions_owned.context("yielding paged prefill lost its positions")?;
        let (rotary_cos, rotary_sin) =
            rope_tables_owned.context("yielding paged prefill lost its rotary tables")?;
        return Ok(PagedForwardProgress {
            logits: None,
            hidden: None,
            token: None,
            state: Some(PagedLayerForwardState {
                hidden,
                positions,
                rotary_cos,
                rotary_sin,
                next_layer: layer_end,
                full_attn_idx,
                linear_attn_idx,
            }),
            layers_processed,
        });
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
    let (logits, output_hidden, token) = match lm_head_mode {
        LmHeadMode::Full => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward_backend_decode_if(Some(backend), &hidden, &weights.embed_tokens_t)?
            };
            Ok::<(Option<Tensor>, Option<Tensor>, Option<u32>), anyhow::Error>((
                Some(logits),
                None,
                None,
            ))
        }
        LmHeadMode::LastRowOnly => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                let last = hidden.narrow(1, seq_len - 1, 1)?;
                let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), None, None))
        }
        LmHeadMode::LastRowArgmaxOnly => {
            let token = {
                kiln_nvtx::range!(c"kiln/lm_head_argmax");
                let last = hidden.narrow(1, seq_len - 1, 1)?;
                if let Some(token) = lm_head_weighted_prep_argmax(
                    &last,
                    &weights.final_norm,
                    &weights.embed_tokens_t,
                )? {
                    return Ok(PagedForwardProgress {
                        logits: None,
                        hidden: None,
                        token: Some(token),
                        state: None,
                        layers_processed,
                    });
                }
                let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
                #[cfg(feature = "rocm")]
                if let Some(lm_head_w8) = weights.lm_head_w8.as_ref() {
                    if normed.dtype() == DType::BF16
                        && !normed.track_op()
                        && matches!(normed.device(), Device::Rocm(_))
                    {
                        let normed = normed
                            .contiguous()
                            .context("rocm w8 lm_head argmax normed contiguous")?;
                        let token = crate::rocm_w8_proj::argmax_bf16(&normed, lm_head_w8)
                            .context("rocm w8 lm_head argmax")?;
                        return Ok(PagedForwardProgress {
                            logits: None,
                            hidden: None,
                            token: Some(token),
                            state: None,
                            layers_processed,
                        });
                    }
                }
                lm_head_argmax_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((None, None, Some(token)))
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
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), Some(last_hidden), None))
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
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward_backend_decode_if(
                    Some(backend),
                    &last_hidden,
                    &weights.embed_tokens_t,
                )?
            };
            Ok((Some(logits), Some(last_hidden), None))
        }
        LmHeadMode::Skip => Ok((None, None, None)),
        LmHeadMode::HiddenOnly => Ok((None, Some(hidden), None)),
    }?;
    Ok(PagedForwardProgress {
        logits,
        hidden: output_hidden,
        token,
        state: None,
        layers_processed,
    })
}

/// Advance one paged prefill token chunk through at most `max_layers`
/// transformer layers. A partial result owns the intermediate hidden and RoPE
/// tensors so the serving actor can run a decode cohort before resuming it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_last_token_layer_group(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: &mut LinearAttentionState,
    lora: Option<&LoraWeights>,
    state: Option<PagedLayerForwardState>,
    max_layers: usize,
) -> Result<PagedLayerForwardProgress> {
    let progress = model_forward_paged_inner_bounded(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        Some(linear_state),
        lora,
        None,
        None,
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowOnly,
        state,
        max_layers,
    )?;
    anyhow::ensure!(
        progress.hidden.is_none() && progress.token.is_none(),
        "last-row layer-group prefill returned an incompatible output"
    );
    Ok(PagedLayerForwardProgress {
        logits: progress.logits,
        state: progress.state,
        layers_processed: progress.layers_processed,
    })
}

/// Streaming/tiled paged prefill — the Phase 7 long-context entry point.
///
/// Iterates `token_ids` in fixed-size tiles (the portable default is 8192
/// tokens and every configured value must be a multiple of `GDN_CHUNK_SIZE`)
/// and dispatches each tile through `model_forward_paged_inner`. The
/// `LinearAttentionState` carries GDN recurrent + conv state across tile
/// boundaries; the paged KV cache is filled tile-by-tile via `start_pos +
/// cursor`. Only the final tile runs the LM head — non-final tiles use
/// `LmHeadMode::Skip`. The injected policy may instead select
/// `LmHeadMode::Full` on the final tile for per-position parity comparisons
/// against the monolithic path.
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`model_forward_paged_streaming`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_progress_and_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        streaming_prefill,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_with_progress(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    progress: Option<&crate::cancel::CancelHandle>,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_progress_and_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        progress,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Progress-aware explicit-policy streaming prefill.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_with_progress_and_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    progress: Option<&crate::cancel::CancelHandle>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_progress_offset_and_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        progress,
        0,
        streaming_prefill,
    )
}

/// Progress-aware tiled prefill with an existing request-local progress base.
///
/// Prefix-cache callers use this for a tail pass after a separately executed
/// head pass, so progress remains cumulative across both forwards.
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_streaming_with_progress_offset(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    progress: Option<&crate::cancel::CancelHandle>,
    progress_offset: u64,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_progress_offset_and_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        progress,
        progress_offset,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn model_forward_paged_streaming_with_progress_offset_and_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    progress: Option<&crate::cancel::CancelHandle>,
    progress_offset: u64,
    streaming_prefill: StreamingPrefillExecutionPolicy,
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
        streaming_prefill.base_tile_tokens_for(token_ids.len()),
        streaming_prefill.last_token_lm_head(),
        progress,
        progress_offset,
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<(Tensor, Tensor)> {
    model_forward_paged_streaming_last_token_with_last_hidden_with_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of
/// [`model_forward_paged_streaming_last_token_with_last_hidden`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_last_token_with_last_hidden_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
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
        streaming_prefill.base_tile_tokens_for(token_ids.len()),
    )
}

/// Streaming/tiled paged prefill that returns only the final pre-final-RMSNorm
/// hidden row, skipping the LM head matmul entirely.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_last_token_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_streaming_last_token_hidden_with_policy(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`model_forward_paged_streaming_last_token_hidden`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_last_token_hidden_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    model_forward_paged_streaming_last_token_hidden_with(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        streaming_prefill.base_tile_tokens_for(token_ids.len()),
    )
}

/// Explicit-tile variant of [`model_forward_paged_streaming_last_token_hidden`].
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_last_token_hidden_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
) -> Result<Tensor> {
    let total = token_ids.len();
    if total == 0 {
        anyhow::bail!(
            "model_forward_paged_streaming_last_token_hidden requires at least one token"
        );
    }
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }

    let mut last_hidden: Option<Tensor> = None;
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let is_last_tile = end == total;
        let mode = if is_last_tile {
            LmHeadMode::HiddenOnly
        } else {
            LmHeadMode::Skip
        };

        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();
        let (_tile_logits, tile_hidden, _token) = model_forward_paged_inner(
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
            None,
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            None,
            #[cfg(feature = "cuda")]
            None,
            mode,
        )
        .with_context(|| {
            format!(
                "streaming hidden prefill tile [{cursor}, {end}) of {total} (start_pos={})",
                start_pos + cursor
            )
        })?;

        if is_last_tile {
            let tile_hidden = tile_hidden.context("streaming hidden prefill produced no hidden")?;
            last_hidden = Some(
                tile_hidden
                    .narrow(1, end - cursor - 1, 1)?
                    .contiguous()
                    .context("streaming last-token hidden contiguous")?,
            );
        }

        cursor = end;
    }

    last_hidden.context("streaming hidden prefill produced no hidden")
}

/// Explicit-tile variant of
/// [`model_forward_paged_streaming_last_token_with_last_hidden`].
pub fn model_forward_paged_streaming_last_token_with_last_hidden_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
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
            LmHeadMode::LastRowWithLastHidden
        } else {
            LmHeadMode::Skip
        };

        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();
        let (tile_logits, tile_hidden, _token) = model_forward_paged_inner(
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
            None,
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            None,
            // Phase 7 #1082: no kt twin from this caller — forward
            // `None` so the candle writer remains authoritative.
            #[cfg(feature = "cuda")]
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
    last_token_only: bool,
    progress: Option<&crate::cancel::CancelHandle>,
    progress_offset: u64,
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
        if progress.is_some_and(crate::cancel::CancelHandle::is_cancelled) {
            anyhow::bail!("streaming prefill cancelled by caller");
        }
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

        let (tile_logits, _tile_hidden, _token) = model_forward_paged_inner(
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
            None,
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            None,
            // Phase 7 #1082: no kt twin from this caller — forward
            // `None` so the candle writer remains authoritative.
            #[cfg(feature = "cuda")]
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
        if let Some(progress) = progress {
            progress.report_prefill_tokens_completed(progress_offset.saturating_add(cursor as u64));
            if progress.is_cancelled() {
                anyhow::bail!("streaming prefill cancelled by caller");
            }
        }
    }

    last_logits.context("streaming prefill produced no logits (empty token_ids)")
}

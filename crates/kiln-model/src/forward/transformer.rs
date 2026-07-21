use super::*;

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
    transformer_block_with_policy(
        backend,
        x,
        layer,
        config,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache,
        full_attn_layer_idx,
        lora,
        StreamingPrefillExecutionPolicy::for_runtime(backend),
    )
}

/// Explicit-policy variant of [`transformer_block`].
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_with_policy(
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
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!("transformer_block only supports full attention layers (not linear/GDN)")
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_ffn = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_runtime::single_token_self_attention_active();

    if let Some(out) = transformer_block_detached_prefill_chunked(
        backend,
        x,
        layer,
        config,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache.is_some(),
        full_attn_layer_idx,
        lora,
        streaming_prefill,
    )? {
        return Ok(out);
    }

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
    //
    // CP-4 (#1082) Increment 2: route through `residual_add` (which threads
    // `try_tape_add_cuda` onto the kt `Tape`) instead of a raw candle `+`.
    // The GDN block already used `residual_add`; `transformer_block` used a
    // raw `+`, which fragmented the tape at the full-attn residual so grads
    // never reached attention / FFN projections below the loss.
    let x = {
        kiln_nvtx::range!(c"kiln/residual");
        residual_add(x.clone(), attn_out)?
    };
    synchronize_tensor_ready_for_model_handoff(
        &format!("layer {full_attn_layer_idx} attention_residual"),
        &x,
    )?;

    // Post-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
    };

    // Feed-forward network. Inference must thread the active backend through
    // every base projection: Vulkan's F32 activations and BF16 resident weights
    // cannot use the portable equal-dtype matmul fallback. Keep the established
    // tape-training route unchanged so its recorder remains authoritative.
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
    let ffn_out = if tape_scope_active && use_metal_decode_ffn {
        swiglu_ffn_metal_decode(&normed, &layer.mlp, lora)?
    } else if tape_scope_active {
        swiglu_ffn(&normed, &layer.mlp, lora)?
    } else {
        swiglu_ffn_backend_profiled(backend, &normed, &layer.mlp, lora, use_metal_decode_ffn)?
    };

    // Residual connection
    //
    // CP-4 (#1082) Increment 2: same as the attention residual above — route
    // through `residual_add` so the FFN residual stays on the kt `Tape`. This
    // is the link that lets `loss → lm_head → final_norm → FFN residual →
    // down_proj` connect, making the last full-attn layer's down_proj LoRA
    // Vars reachable (`tape_has_grad` rises from 0).
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        residual_add(x, ffn_out)?
    };
    synchronize_tensor_ready_for_model_handoff(
        &format!("layer {full_attn_layer_idx} output"),
        &out,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transformer_block_detached_prefill_chunked(
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
    has_kv_cache: bool,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Option<Tensor>> {
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

    if !detached_chunked_prefill_supported(backend) || has_kv_cache {
        return Ok(None);
    }
    let (batch, seq_len, _hidden) = x.dims3()?;
    if !streaming_prefill.enabled_for(seq_len) {
        return Ok(None);
    }
    let mode = if tape_scope_active {
        FullAttnChunkMode::TapeReplay
    } else {
        if x.track_op() {
            return Ok(None);
        }
        FullAttnChunkMode::DetachedBoundary
    };
    let base_tile_size = match mode {
        FullAttnChunkMode::DetachedBoundary => {
            streaming_prefill.detached_full_attn_boundary_tile_tokens()
        }
        FullAttnChunkMode::TapeReplay => {
            streaming_prefill.detached_full_attn_tape_replay_tile_tokens()
        }
    };
    if base_tile_size == 0 {
        return Ok(None);
    }
    let GpuAttentionWeights::Full(attn_weights) = &layer.attention else {
        return Ok(None);
    };
    let materialized_score_budget_mb = full_attn_materialized_score_budget_mib();
    let materialized_scratch_buffers =
        mode.materialized_scratch_buffers_for_tile_plan(backend, &x.device(), x.dtype(), head_dim);
    if materialized_scratch_buffers > 0 {
        let minimum_query_tile = seq_len.min(MATERIALIZED_FULL_ATTN_TILE_GRANULARITY);
        let minimum_bounded_scratch_bytes = full_attn_materialized_scratch_bytes(
            x.dtype(),
            batch,
            num_heads,
            minimum_query_tile,
            seq_len,
            materialized_scratch_buffers,
        )
        .unwrap_or(u64::MAX);
        let published_budget_bytes = u64::try_from(materialized_score_budget_mb)
            .unwrap_or(u64::MAX)
            .saturating_mul(1024 * 1024);
        if published_budget_bytes < minimum_bounded_scratch_bytes {
            anyhow::bail!(
                "full-attention prefill rejected: published memory budget cannot sustain a bounded tile plan (budget_bytes={published_budget_bytes}, minimum_scratch_bytes={minimum_bounded_scratch_bytes}, batch={batch}, sequence_tokens={seq_len}, heads={num_heads}, minimum_query_tile={minimum_query_tile})"
            );
        }
    }
    let (tile_count, first_tile_size, min_tile_size, max_tile_size, peak_scratch_bytes) =
        full_attn_adaptive_tile_plan_summary(
            &x.device(),
            x.dtype(),
            batch,
            seq_len,
            num_heads,
            base_tile_size,
            materialized_scratch_buffers,
            materialized_score_budget_mb,
        );
    let peak_scratch_bytes = peak_scratch_bytes.ok_or_else(|| {
        anyhow::anyhow!(
            "full-attention prefill rejected: materialized scratch-byte calculation overflowed"
        )
    })?;

    tracing::info!(
        layer = full_attn_layer_idx,
        seq_len,
        base_tile_size,
        tile_count,
        first_tile_size,
        min_tile_size,
        max_tile_size,
        score_budget_mb = materialized_score_budget_mb,
        scratch_buffers = materialized_scratch_buffers,
        peak_scratch_bytes,
        flash_tile_guaranteed =
            mode.flash_tile_guaranteed(backend, &x.device(), x.dtype(), head_dim),
        mode = mode.label(),
        "detached full-attention prefill chunked"
    );

    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn_chunked");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention pre_attn_norm"),
        &normed,
    )?;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let k_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        &normed,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention k projection")?;
    let k = tape_reshape_full_attn(
        &k_flat,
        &[
            ReshapeArg::Infer,
            seq_len.into(),
            num_kv_heads.into(),
            head_dim.into(),
        ],
    )
    .context("chunked full-attention k reshape")?;
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention k projection"),
        &k,
    )?;
    let v_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        &normed,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention v projection")?;
    let v = tape_reshape_full_attn(
        &v_flat,
        &[
            ReshapeArg::Infer,
            seq_len.into(),
            num_kv_heads.into(),
            head_dim.into(),
        ],
    )
    .context("chunked full-attention v reshape")?;
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention v"),
        &v,
    )?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)
        .context("chunked full-attention k norm")?;
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention k_norm"),
        &k,
    )?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)
        .context("chunked full-attention k rotary")?;
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention k_rotary"),
        &k,
    )?;
    let mut output_tiles = Vec::with_capacity(seq_len.div_ceil(base_tile_size));
    let mut tile_start = 0usize;
    while tile_start < seq_len {
        let remaining = seq_len - tile_start;
        let tile_len = full_attn_adaptive_tile_len_with_budget(
            &x.device(),
            x.dtype(),
            batch,
            tile_start,
            remaining,
            num_heads,
            base_tile_size,
            materialized_scratch_buffers,
            materialized_score_budget_mb,
        );
        let tile_end = tile_start + tile_len;
        let normed_tile = tape_narrow_contig_full_attn(
            &normed,
            1,
            tile_start,
            tile_len,
            &format!("chunked full-attention normed tile [{tile_start}, {tile_end})"),
            true,
        )?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention normed tile [{tile_start}, {tile_end})"
            ),
            &normed_tile,
        )?;
        let split_q_gate = if config.attn_output_gate {
            split_q_gate_training_bf16(
                backend,
                false,
                &normed_tile,
                attn_weights,
                lora_layer.and_then(|l| l.q_proj.as_ref()),
                lora_scale,
                tile_len,
                num_heads,
                head_dim,
            )
            .with_context(|| {
                format!("chunked full-attention split q/gate [{tile_start}, {tile_end})")
            })?
        } else {
            None
        };
        let (q_tile, gate_tile) = if let Some((q, gate)) = split_q_gate {
            (q, Some(gate))
        } else {
            let q_raw = q_proj_forward_decode_if(
                Some(backend),
                false,
                &normed_tile,
                attn_weights,
                lora_layer.and_then(|l| l.q_proj.as_ref()),
                lora_scale,
            )
            .with_context(|| {
                format!("chunked full-attention q projection [{tile_start}, {tile_end})")
            })?;
            if config.attn_output_gate {
                let q_raw = tape_reshape_full_attn(
                    &q_raw,
                    &[
                        ReshapeArg::Infer,
                        tile_len.into(),
                        num_heads.into(),
                        (head_dim * 2).into(),
                    ],
                )
                .with_context(|| {
                    format!("chunked full-attention q/gate reshape [{tile_start}, {tile_end})")
                })?;
                let q = tape_narrow_contig_full_attn(
                    &q_raw,
                    3,
                    0,
                    head_dim,
                    &format!("chunked full-attention q split [{tile_start}, {tile_end})"),
                    true,
                )?;
                let gate_split = tape_narrow_contig_full_attn(
                    &q_raw,
                    3,
                    head_dim,
                    head_dim,
                    &format!("chunked full-attention gate split [{tile_start}, {tile_end})"),
                    true,
                )?;
                let gate = tape_reshape_full_attn(
                    &gate_split,
                    &[
                        ReshapeArg::Infer,
                        tile_len.into(),
                        (num_heads * head_dim).into(),
                    ],
                )
                .context("chunked full-attention gate reshape")?;
                (q, Some(gate))
            } else {
                (
                    tape_reshape_full_attn(
                        &q_raw,
                        &[
                            ReshapeArg::Infer,
                            tile_len.into(),
                            num_heads.into(),
                            head_dim.into(),
                        ],
                    )
                    .with_context(|| {
                        format!("chunked full-attention q reshape [{tile_start}, {tile_end})")
                    })?,
                    None,
                )
            }
        };
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention q projection tile [{tile_start}, {tile_end})"
            ),
            &q_tile,
        )?;
        let q_tile = rms_norm(&q_tile, &attn_weights.q_norm, rms_norm_eps)
            .context("chunked full-attention q norm")?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention q_norm input tile [{tile_start}, {tile_end})"
            ),
            &q_tile,
        )?;
        if let Some(gate) = gate_tile.as_ref() {
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!(
                    "layer {full_attn_layer_idx} chunked full-attention gate tile [{tile_start}, {tile_end})"
                ),
                gate,
            )?;
        }
        let tile_positions = &positions[tile_start..tile_end];
        let (q_tile, _) = rotary_embedding(
            &q_tile,
            &q_tile,
            tile_positions,
            head_dim,
            rotary_dim,
            inv_freq,
        )
        .with_context(|| format!("chunked full-attention q rotary [{tile_start}, {tile_end})"))?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention q_rotary tile [{tile_start}, {tile_end})"
            ),
            &q_tile,
        )?;
        let k_prefix = tape_narrow_contig_full_attn(
            &k,
            1,
            0,
            tile_end,
            &format!("chunked full-attention k prefix [0, {tile_end}) for tile {tile_start}"),
            false,
        )?;
        let v_prefix = tape_narrow_contig_full_attn(
            &v,
            1,
            0,
            tile_end,
            &format!("chunked full-attention v prefix [0, {tile_end}) for tile {tile_start}"),
            false,
        )?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention k_prefix [0, {tile_end}) tile [{tile_start}, {tile_end})"
            ),
            &k_prefix,
        )?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention v_prefix [0, {tile_end}) tile [{tile_start}, {tile_end})"
            ),
            &v_prefix,
        )?;
        let tile_prepared = GqaAttentionPrepared {
            q: q_tile,
            k: k_prefix,
            v: v_prefix,
            gate: None,
        };
        let flash_tile_guaranteed =
            mode.flash_tile_guaranteed(backend, &x.device(), x.dtype(), head_dim);
        let flash_tile_allowed = flash_tile_guaranteed
            && flash_prefill_allowed_for_shape(
                backend,
                &x.device(),
                x.dtype(),
                head_dim,
                tile_len,
                tile_end,
            );
        let attn_core = if flash_tile_allowed {
            match mode {
                FullAttnChunkMode::DetachedBoundary => {
                    let q_detached = tile_prepared.q.detach();
                    let k_detached = tile_prepared.k.detach();
                    let v_detached = tile_prepared.v.detach();
                    let q_flash = fresh_contig_full_attn(
                        &q_detached,
                        &format!("chunked full-attention flash q tile [{tile_start}, {tile_end})"),
                    )?;
                    let k_flash = fresh_contig_full_attn(
                        &k_detached,
                        &format!("chunked full-attention flash k prefix [0, {tile_end})"),
                    )?;
                    let v_flash = fresh_contig_full_attn(
                        &v_detached,
                        &format!("chunked full-attention flash v prefix [0, {tile_end})"),
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention flash q tile [{tile_start}, {tile_end})"
                        ),
                        &q_flash,
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention flash k prefix [0, {tile_end})"
                        ),
                        &k_flash,
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention flash v prefix [0, {tile_end})"
                        ),
                        &v_flash,
                    )?;
                    let flash_result = flash_attention_forward(
                        backend,
                        &q_flash,
                        &k_flash,
                        &v_flash,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )
                    .with_context(|| {
                        format!("chunked full-attention flash core tile [{tile_start}, {tile_end})")
                    })?;
                    match flash_result {
                        Some(out) => out,
                        None => gqa_attention_core_prefill(
                            backend,
                            &tile_prepared,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        )
                        .with_context(|| {
                            format!("chunked full-attention core tile [{tile_start}, {tile_end})")
                        })?,
                    }
                }
                FullAttnChunkMode::TapeReplay => {
                    let q_flash = tape_contig_full_attn(
                        &tile_prepared.q,
                        &format!(
                            "chunked full-attention tape flash q tile [{tile_start}, {tile_end})"
                        ),
                    )?;
                    let k_flash = tape_contig_full_attn(
                        &tile_prepared.k,
                        &format!("chunked full-attention tape flash k prefix [0, {tile_end})"),
                    )?;
                    let v_flash = tape_contig_full_attn(
                        &tile_prepared.v,
                        &format!("chunked full-attention tape flash v prefix [0, {tile_end})"),
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention tape flash q tile [{tile_start}, {tile_end})"
                        ),
                        &q_flash,
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention tape flash k prefix [0, {tile_end})"
                        ),
                        &k_flash,
                    )?;
                    synchronize_tensor_ready_for_full_attn_handoff(
                        &format!(
                            "layer {full_attn_layer_idx} chunked full-attention tape flash v prefix [0, {tile_end})"
                        ),
                        &v_flash,
                    )?;
                    #[cfg(any(feature = "cuda", feature = "rocm"))]
                    {
                        let flash_result = crate::tape_forward::try_tape_flash_attn_kt(
                        &q_flash,
                        &k_flash,
                        &v_flash,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )
                    .with_context(|| {
                        format!(
                            "chunked full-attention tape flash core tile [{tile_start}, {tile_end})"
                        )
                    })?;
                        if let Some(attn_output) = flash_result {
                            let (rb, rs, rh, rd) = attn_output.dims4()?;
                            tape_reshape_full_attn(
                                &attn_output,
                                &[rb.into(), rs.into(), (rh * rd).into()],
                            )?
                        } else {
                            gqa_attention_core_prefill(
                                backend,
                                &tile_prepared,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                            )
                            .with_context(|| {
                                format!(
                                    "chunked full-attention core tile [{tile_start}, {tile_end})"
                                )
                            })?
                        }
                    }
                    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                    {
                        gqa_attention_core_prefill(
                            backend,
                            &tile_prepared,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        )
                        .with_context(|| {
                            format!("chunked full-attention core tile [{tile_start}, {tile_end})")
                        })?
                    }
                }
            }
        } else {
            gqa_attention_core_prefill(backend, &tile_prepared, num_heads, num_kv_heads, head_dim)
                .with_context(|| {
                format!("chunked full-attention core tile [{tile_start}, {tile_end})")
            })?
        };
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention attn_core tile [{tile_start}, {tile_end})"
            ),
            &attn_core,
        )?;
        let attn_output = gqa_attention_apply_output_gate(attn_core, gate_tile.as_ref())
            .with_context(|| {
                format!("chunked full-attention gate tile [{tile_start}, {tile_end})")
            })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention attn_output tile [{tile_start}, {tile_end})"
            ),
            &attn_output,
        )?;
        let attn_out =
            gqa_attention_output_projection(backend, &attn_output, attn_weights, false, lora)
                .with_context(|| {
                    format!("chunked full-attention o-proj tile [{tile_start}, {tile_end})")
                })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention o_proj tile [{tile_start}, {tile_end})"
            ),
            &attn_out,
        )?;
        let x_tile = tape_narrow_contig_full_attn(
            x,
            1,
            tile_start,
            tile_len,
            &format!("chunked full-attention residual tile [{tile_start}, {tile_end})"),
            true,
        )?;
        let residual = residual_add(x_tile, attn_out).with_context(|| {
            format!("chunked full-attention attention residual tile [{tile_start}, {tile_end})")
        })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention attention_residual tile [{tile_start}, {tile_end})"
            ),
            &residual,
        )?;
        let normed_post = rms_norm(&residual, &layer.post_attention_layernorm, rms_norm_eps)
            .with_context(|| {
                format!("chunked full-attention post norm tile [{tile_start}, {tile_end})")
            })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention post_norm tile [{tile_start}, {tile_end})"
            ),
            &normed_post,
        )?;
        let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, lora).with_context(|| {
            format!("chunked full-attention MLP tile [{tile_start}, {tile_end})")
        })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention mlp tile [{tile_start}, {tile_end})"
            ),
            &ffn_out,
        )?;
        let out_tile = residual_add(residual, ffn_out).with_context(|| {
            format!("chunked full-attention output tile [{tile_start}, {tile_end})")
        })?;
        synchronize_tensor_ready_for_full_attn_handoff(
            &format!(
                "layer {full_attn_layer_idx} chunked full-attention output tile [{tile_start}, {tile_end})"
            ),
            &out_tile,
        )?;
        let out_tile = if mode.detach_outputs() {
            let detached = out_tile.detach();
            let fresh = fresh_contig_full_attn(
                &detached,
                &format!("chunked full-attention detached output tile [{tile_start}, {tile_end})"),
            )?;
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!(
                    "layer {full_attn_layer_idx} chunked full-attention detached output tile [{tile_start}, {tile_end})"
                ),
                &fresh,
            )?;
            fresh
        } else {
            out_tile
        };
        output_tiles.push(out_tile);
        tile_start = tile_end;
    }

    let output_refs: Vec<&Tensor> = output_tiles.iter().collect();
    let output = Tensor::cat(&output_refs, 1).context("chunked full-attention output cat")?;
    synchronize_tensor_ready_for_full_attn_handoff(
        &format!("layer {full_attn_layer_idx} chunked full-attention output cat"),
        &output,
    )?;
    let output = match mode {
        FullAttnChunkMode::DetachedBoundary => {
            let detached = output.detach();
            let fresh = fresh_contig_full_attn(
                &detached,
                &format!("layer {full_attn_layer_idx} detached chunked output"),
            )?;
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!("layer {full_attn_layer_idx} detached chunked output"),
                &fresh,
            )?;
            fresh
        }
        FullAttnChunkMode::TapeReplay => {
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            {
                crate::tape_forward::try_tape_concat_kt(&output_refs, 1, &output)
                    .context("chunked full-attention output cat try_tape_concat_kt")?
                    .context("chunked full-attention tape replay failed to record output cat")?
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            {
                output
            }
        }
    };
    Ok(Some(output))
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
    paged_cache: &PagedKvCache,
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
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin plumbed through this wrapper yet —
        // the cache-owning struct migration that allocates one is a
        // follow-up commit. Default `None` keeps this path on the
        // candle writer only.
        #[cfg(feature = "cuda")]
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transformer_block_paged_with_rope_tables(
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
    // Phase 7 #1082: kt twin of `paged_cache`, plumbed through to the
    // GQA paged-attention call below so the kt cache can mirror the
    // CUDA-graph paged-KV write. `None` keeps the path on the candle
    // writer only; same migration playbook as `graph_inputs`.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!(
                "transformer_block_paged only supports full attention layers (not linear/GDN)"
            )
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Vulkan-resident decode fast-path. Gates: seq_len = 1 (decode hot
    // path), start_pos > 0 (need at least one prefill K/V), no LoRA,
    // no MTP, no CUDA graph inputs, and a resident backend implementation.
    #[cfg(feature = "vulkan")]
    {
        if seq_len == 1
            && start_pos > 0
            && lora.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && config.attn_output_gate
            && qualified_vulkan_resident_decode_enabled()
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(out) = try_resident_block_full_attn_b1(
                    vk_backend,
                    x,
                    layer,
                    config,
                    positions,
                    start_pos,
                    paged_cache,
                    block_table,
                    full_attn_layer_idx,
                    inv_freq,
                    rope_tables,
                )? {
                    return Ok(out);
                }
            }
        }
    }
    let use_metal_decode_ffn =
        seq_len == 1 && start_pos > 0 && !crate::mtp_runtime::single_token_self_attention_active();

    // Pre-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };
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
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        graph_inputs,
        // Phase 7 #1082: forward the kt twin through to the GQA
        // attention call. None on this path until the cache-owning
        // model struct migration plumbs an `Option<&PagedKvCacheKt>`
        // through every transformer-block invocation.
        #[cfg(feature = "cuda")]
        kt_paged_cache,
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
    let ffn_out = if use_metal_decode_ffn {
        swiglu_ffn_backend_profiled(backend, &normed, &layer.mlp, lora, true)?
    } else {
        swiglu_ffn_backend_profiled(backend, &normed, &layer.mlp, lora, false)?
    };

    // Residual connection
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + ffn_out)?
    };
    // Note: the final block output (`out`) is dumped as `post_layer` at the
    // outer MTP call site, so we do not re-capture it here.
    Ok(out)
}

/// Batched decode variant of [`transformer_block_paged`] for full-attention
/// layers whose paged-KV windows are contiguous and share one decode position.
///
/// This wraps [`gqa_attention_paged_decode_contiguous_batch`] with the block's
/// pre-attention norm, residuals, post-attention norm, and MLP. Linear/GDN
/// layers are intentionally out of scope; they use `LinearAttentionState`
/// batching instead.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_positions: &[usize],
    inv_freq: &Tensor,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
    cached_meta: Option<&CachedPagedDecodeMeta>,
    // CUDA-graph-stable `(attn_out, softmax_lse)` scratch for the bs>1
    // captured forward path. Forwarded as-is to
    // `gqa_attention_paged_decode_contiguous_batch` (#1082).
    graph_outputs: Option<(&Tensor, &Tensor)>,
    // CUDA-graph-stable RoPE cos/sin tables for the bs>1 captured
    // forward path. Forwarded as-is to
    // `gqa_attention_paged_decode_contiguous_batch` (#1082 suspect 2).
    rope_tables: Option<(&Tensor, &Tensor)>,
    // CUDA-graph-stable `[batch]` u32 per-row KV-write slot tensor.
    // Forwarded as-is to `gqa_attention_paged_decode_contiguous_batch`
    // (#1082 suspect 1).
    kv_slot: Option<&Tensor>,
    #[cfg(feature = "metal")] metal_icb_layer: Option<MetalPagedDecodeIcbLayer<'_>>,
    // Phase 7 #1082: kt twin of `paged_cache` forwarded as-is to the
    // GQA layer. `None` (default) keeps the candle accessor path.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!(
                "batched contiguous paged transformer decode only supports full attention layers"
            )
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;
    anyhow::ensure!(
        seq_len == 1,
        "batched contiguous paged transformer decode requires one token per row"
    );
    anyhow::ensure!(
        !start_positions.is_empty(),
        "batched contiguous paged transformer decode requires a non-empty batch"
    );
    // Phase 12-B-prime: per-row start positions are allowed; the SwiGLU MLP
    // decode-gemv hint must hold for every row, so require all > 0.
    let use_metal_decode_ffn = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_runtime::single_token_self_attention_active();

    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn_batch_decode");
        rms_norm(x, &layer.input_layernorm, config.rms_norm_eps)?
    };
    let attn_out = gqa_attention_paged_decode_contiguous_batch(
        backend,
        &normed,
        attn_weights,
        positions,
        start_positions,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim(),
        inv_freq,
        config.rms_norm_eps,
        paged_cache,
        block_tables,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
        cached_meta,
        graph_outputs,
        rope_tables,
        kv_slot,
        #[cfg(feature = "metal")]
        metal_icb_layer,
        #[cfg(feature = "cuda")]
        kt_paged_cache,
    )?;
    let x = {
        kiln_nvtx::range!(c"kiln/residual_batch_decode");
        (x + attn_out)?
    };
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp_batch_decode");
        rms_norm(&x, &layer.post_attention_layernorm, config.rms_norm_eps)?
    };
    let ffn_out =
        swiglu_ffn_backend_profiled(backend, &normed, &layer.mlp, lora, use_metal_decode_ffn)?;
    let out = {
        kiln_nvtx::range!(c"kiln/residual_batch_decode");
        (x + ffn_out)?
    };
    Ok(out)
}

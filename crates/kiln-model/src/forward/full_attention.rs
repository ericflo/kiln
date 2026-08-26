use super::*;

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
    q_proj_forward_decode_if(None, false, x, attn_weights, lora, lora_scale)
}

pub(super) fn q_proj_forward_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if crate::tape_forward::tape_scope_active() && attn_weights.q_proj_marlin.is_some() {
        anyhow::bail!("active tape scope cannot use forward-only Marlin query projection");
    }
    #[cfg(feature = "cuda")]
    if !crate::tape_forward::tape_scope_active()
        && let Some(ref packed) = attn_weights.q_proj_marlin
    {
        // #1082 H2: kt-native Marlin matmul — no kt->candle->candle->kt
        // round-trip on the per-token activation/result. Runs ×8 full-attn
        // layers/token. The kt-native LoRA delta/add chain runs directly on
        // the kt base.
        let base = crate::marlin_proj::matmul_bf16_kt(x, packed)
            .context("q_proj_forward: kt-native marlin matmul")?;
        if let Some(proj) = lora {
            if let Some(delta) =
                try_kt_lora_delta(x, proj, lora_scale).context("q_proj_forward: kt lora delta")?
            {
                let delta = if delta.dtype() == base.dtype() {
                    delta
                } else {
                    delta
                        .to_dtype(base.dtype())
                        .context("q_proj_forward: kt lora delta cast")?
                };
                if let Some(out) =
                    try_kt_lora_add(&base, &delta).context("q_proj_forward: kt lora add")?
                {
                    return Ok(out);
                }
                return Ok((base + delta).context("q_proj_forward: add lora delta")?);
            }
            // kt-native composite LoRA delta fallback (#1082: compute_lora_delta
            // is kt now — pass kt `x` directly).
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("q_proj_forward: lora delta")?;
            return Ok((base + delta).context("q_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    linear_with_lora_t_backend_decode_if(
        backend,
        use_metal_decode_gemv,
        x,
        &attn_weights.q_proj_t,
        lora,
        lora_scale,
    )
}

pub(super) fn split_q_gate_training_disabled(device: &Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().split_q_gate_training;
    }
    false
}

pub(super) fn split_q_gate_output_chunk_features_for_device(
    device: &Device,
    full_dim: usize,
) -> usize {
    if matches!(device, Device::Rocm(_)) {
        return crate::rocm_policy::current_rocm_kernel_policy()
            .split_q_gate_output_chunk_features
            .min(full_dim)
            .max(1);
    }
    match device {
        // ROCm policy is handled above.
        Device::Rocm(_) => unreachable!("ROCm policy returned above"),
        // Vulkan's linear offset path already has submit-size ceilings. Use
        // the same split contract when BF16 projection slices are available.
        Device::Vulkan(_) => full_dim.min(1024).max(1),
        _ => full_dim.max(1),
    }
}

pub(super) fn lora_projection_slice(
    proj: &LoraProjectionWeights,
    chunk_start: usize,
    chunk_len: usize,
) -> Result<LoraProjectionWeights> {
    Ok(LoraProjectionWeights {
        a: proj.a.clone(),
        b: proj
            .b
            .narrow(0, chunk_start, chunk_len)
            .context("split q/gate LoRA B slice")?
            .contiguous()
            .context("split q/gate LoRA B slice contiguous")?,
    })
}

pub(super) fn q_gate_projection_weight_slice(
    full_weight_t: &Tensor,
    x_dtype: DType,
    chunk_start: usize,
    chunk_len: usize,
    context: &str,
) -> Result<Tensor> {
    let chunk = full_weight_t
        .narrow(1, chunk_start, chunk_len)
        .with_context(|| format!("{context} narrow weight chunk"))?
        .contiguous()
        .with_context(|| format!("{context} contiguous weight chunk"))?;
    if chunk.dtype() == x_dtype {
        Ok(chunk)
    } else {
        chunk
            .to_dtype(x_dtype)
            .with_context(|| format!("{context} cast weight chunk"))
    }
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_split_q_gate_row_tile_tokens() -> usize {
    crate::rocm_policy::current_rocm_kernel_policy().split_q_gate_row_tile_tokens
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_q_gate_projection_slice_bf16_via_f32(
    x: &Tensor,
    full_weight_t: &Tensor,
    chunk_start: usize,
    chunk_len: usize,
) -> Result<Option<Tensor>> {
    if !crate::rocm_policy::current_rocm_kernel_policy().split_q_gate_f32_output
        || !matches!(x.device(), Device::Rocm(_))
        || x.dtype() != DType::BF16
        || full_weight_t.dtype() != DType::BF16
        || full_weight_t.dims().len() != 2
        || chunk_len == 0
        || chunk_start >= full_weight_t.dims()[1]
        || chunk_start + chunk_len > full_weight_t.dims()[1]
    {
        return Ok(None);
    }
    let x_dims = x.dims().to_vec();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    let k = *x_dims.last().unwrap();
    if k != full_weight_t.dims()[0] {
        return Ok(None);
    }
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    let weight_t = q_gate_projection_weight_slice(
        full_weight_t,
        x.dtype(),
        chunk_start,
        chunk_len,
        "rocm split q/gate f32-output projection",
    )?;
    let x_contig = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()
            .context("rocm split q/gate f32-output x contiguous")?
    };
    let x2d = x_contig
        .reshape((rows, k))
        .context("rocm split q/gate f32-output x reshape")?;
    let row_tile = rocm_split_q_gate_row_tile_tokens().min(rows).max(1);
    let out2d = if rows > row_tile {
        let mut pieces = Vec::with_capacity(rows.div_ceil(row_tile));
        let mut row_start = 0usize;
        while row_start < rows {
            let row_len = (rows - row_start).min(row_tile);
            let x_tile = x2d
                .narrow(0, row_start, row_len)
                .with_context(|| {
                    format!(
                        "rocm split q/gate f32-output x row tile [{row_start}, {})",
                        row_start + row_len
                    )
                })?
                .contiguous()
                .with_context(|| {
                    format!(
                        "rocm split q/gate f32-output x row tile [{row_start}, {}) contiguous",
                        row_start + row_len
                    )
                })?;
            let out_tile = kiln_tensor::rocm_bf16_matmul_bf16_out(&x_tile, &weight_t)
                .with_context(|| {
                    format!(
                        "rocm split q/gate bf16 fallback matmul row tile [{row_start}, {})",
                        row_start + row_len
                    )
                })?;
            pieces.push(out_tile);
            row_start += row_len;
        }
        let piece_refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&piece_refs, 0).context("rocm split q/gate f32-output row tile cat")?
    } else {
        kiln_tensor::rocm_bf16_matmul_bf16_out(&x2d, &weight_t)
            .context("rocm split q/gate bf16 fallback matmul")?
    };
    let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
    out_shape.push(chunk_len);
    Ok(Some(
        out2d
            .reshape(out_shape)
            .context("rocm split q/gate f32-output reshape")?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linear_with_lora_t_backend_decode_output_slice(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    full_weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    chunk_start: usize,
    chunk_len: usize,
) -> Result<Tensor> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let weight_t = q_gate_projection_weight_slice(
            full_weight_t,
            x.dtype(),
            chunk_start,
            chunk_len,
            "split q/gate tape projection",
        )?;
        return require_active_tape_output(
            crate::tape_forward::try_tape_lora_linear_output_slice_kt(
                x,
                &weight_t,
                lora,
                lora_scale,
                chunk_start,
            )
            .context("split q/gate projection try_tape_lora_linear_output_slice_kt")?,
            "split query/gate projection",
        );
    }

    // Inference does not need parameter identity or a full-shape gradient, so
    // materialize the narrow B view only after the authoritative tape branch.
    let lora_slice = lora
        .map(|proj| lora_projection_slice(proj, chunk_start, chunk_len))
        .transpose()?;

    #[cfg(feature = "rocm")]
    if let Some(base) =
        rocm_q_gate_projection_slice_bf16_via_f32(x, full_weight_t, chunk_start, chunk_len)?
    {
        let out = add_lora_delta_to_base(Some(backend), base, x, lora_slice.as_ref(), lora_scale)?;
        return Ok(out);
    }

    if let Some(base) = LinearBackend::runtime_linear_prefill_apply_offset(
        backend,
        x,
        full_weight_t,
        chunk_start,
        chunk_len,
    )? {
        let out = add_lora_delta_to_base(Some(backend), base, x, lora_slice.as_ref(), lora_scale)?;
        return Ok(out);
    }

    let weight_t = q_gate_projection_weight_slice(
        full_weight_t,
        x.dtype(),
        chunk_start,
        chunk_len,
        "split q/gate projection",
    )?;
    linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &weight_t,
        lora_slice.as_ref(),
        lora_scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linear_with_lora_t_backend_decode_output_range_chunked(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    full_weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    range_start: usize,
    range_len: usize,
) -> Result<Tensor> {
    let chunk_features = split_q_gate_output_chunk_features_for_device(&x.device(), range_len);
    if chunk_features >= range_len {
        let out = linear_with_lora_t_backend_decode_output_slice(
            backend,
            use_metal_decode_gemv,
            x,
            full_weight_t,
            lora,
            lora_scale,
            range_start,
            range_len,
        )?;
        return Ok(out);
    }

    let mut chunks = Vec::with_capacity(range_len.div_ceil(chunk_features));
    let mut offset = 0usize;
    while offset < range_len {
        let cur_len = (range_len - offset).min(chunk_features);
        let chunk_start = range_start + offset;
        let chunk = linear_with_lora_t_backend_decode_output_slice(
            backend,
            use_metal_decode_gemv,
            x,
            full_weight_t,
            lora,
            lora_scale,
            chunk_start,
            cur_len,
        )?;
        chunks.push(chunk);
        offset += cur_len;
    }

    let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
    let out = Tensor::cat(&chunk_refs, x.rank() - 1).context("split q/gate projection concat")?;
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_concat_kt(&chunk_refs, out.rank() - 1, &out)
                .context("split q/gate projection concat try_tape_concat_kt")?,
            "split query/gate projection concatenation",
        );
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn split_q_gate_training_bf16(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    if split_q_gate_training_disabled(&x.device())
        || x.dtype() != DType::BF16
        || !matches!(
            x.device(),
            Device::Cuda(_) | Device::Rocm(_) | Device::Vulkan(_)
        )
        || attn_weights.q_proj_marlin.is_some()
    {
        return Ok(None);
    }

    let q_dim = num_heads * head_dim;
    let Ok((_, q_out_dim)) = attn_weights.q_proj_t.dims2() else {
        return Ok(None);
    };
    if q_out_dim != q_dim * 2 {
        return Ok(None);
    }

    if let Some(proj) = lora {
        let Ok((b_out, _rank)) = proj.b.dims2() else {
            return Ok(None);
        };
        if b_out != q_dim * 2 {
            return Ok(None);
        }
    }

    let q_flat = linear_with_lora_t_backend_decode_output_range_chunked(
        backend,
        use_metal_decode_gemv,
        x,
        &attn_weights.q_proj_t,
        lora,
        lora_scale,
        0,
        q_dim,
    )?;
    let gate = linear_with_lora_t_backend_decode_output_range_chunked(
        backend,
        use_metal_decode_gemv,
        x,
        &attn_weights.q_proj_t,
        lora,
        lora_scale,
        q_dim,
        q_dim,
    )?;
    let q = tape_reshape_full_attn(
        &q_flat,
        &[
            ReshapeArg::Infer,
            seq_len.into(),
            num_heads.into(),
            head_dim.into(),
        ],
    )?;
    let gate = tape_reshape_full_attn(&gate, &[ReshapeArg::Infer, seq_len.into(), q_dim.into()])?;
    Ok(Some((q, gate)))
}

pub struct GqaAttentionPrepared {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub gate: Option<Tensor>,
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q_gate_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<(Tensor, Option<Tensor>)> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    anyhow::ensure!(
        !crate::tape_forward::tape_scope_active(),
        "gqa_attention_q_gate_prefill is inference-only until every reshape and narrow is tape-recorded"
    );
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv = false;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = if attn_output_gate {
        split_q_gate_training_bf16(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
            seq_len,
            num_heads,
            head_dim,
        )?
    } else {
        None
    };

    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else {
        let q_raw = q_proj_forward_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )?;
        if attn_output_gate {
            let q_raw = reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim * 2)?;
            let q = q_raw.narrow(3, 0, head_dim)?;
            let gate = q_raw.narrow(3, head_dim, head_dim)?;
            let gate = reshape_hole0_3(&gate.contiguous()?, seq_len, num_heads * head_dim)?;
            (q.contiguous()?, Some(gate))
        } else {
            (reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim)?, None)
        }
    };

    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let (q, _) = rotary_embedding(&q, &q, positions, head_dim, rotary_dim, inv_freq)?;
    Ok((q, gate))
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_kv_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<(Tensor, Tensor)> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    anyhow::ensure!(
        !crate::tape_forward::tape_scope_active(),
        "gqa_attention_kv_prefill is inference-only until every reshape is tape-recorded"
    );
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let k_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )?;
    let k = reshape_hole0_4(&k_flat, seq_len, num_kv_heads, head_dim)?;
    let v_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )?;
    let v = reshape_hole0_4(&v_flat, seq_len, num_kv_heads, head_dim)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)?;
    Ok((k, v))
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_prepare_prefill(
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
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<GqaAttentionPrepared> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    anyhow::ensure!(
        !crate::tape_forward::tape_scope_active(),
        "gqa_attention_prepare_prefill is inference-only until every reshape and narrow is tape-recorded"
    );
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv = false;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = if attn_output_gate {
        split_q_gate_training_bf16(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
            seq_len,
            num_heads,
            head_dim,
        )?
    } else {
        None
    };

    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        if split_q_gate.is_some() {
            let k = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.k_proj_t,
                lora_layer.and_then(|l| l.k_proj.as_ref()),
                lora_scale,
            )?;
            let v = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.v_proj_t,
                lora_layer.and_then(|l| l.v_proj.as_ref()),
                lora_scale,
            )?;
            (None, k, v)
        } else {
            let (q_raw, k, v) = full_attn_qkv_proj_decode_if(
                backend,
                use_metal_decode_gemv,
                x,
                attn_weights,
                lora_layer,
                lora_scale,
            )?;
            (Some(q_raw), k, v)
        }
    };

    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else if attn_output_gate {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when split_q_gate is inactive");
        let q_raw = reshape_hole0_4(q_raw, seq_len, num_heads, head_dim * 2)?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        let gate = reshape_hole0_3(&gate.contiguous()?, seq_len, num_heads * head_dim)?;
        (q.contiguous()?, Some(gate))
    } else {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when attention output gate is disabled");
        let q = reshape_hole0_4(q_raw, seq_len, num_heads, head_dim)?;
        (q, None)
    };

    let k = reshape_hole0_4(&k, seq_len, num_kv_heads, head_dim)?;
    let v = reshape_hole0_4(&v, seq_len, num_kv_heads, head_dim)?;
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
    let (q, k) = rotary_embedding(&q, &k, positions, head_dim, rotary_dim, inv_freq)?;

    Ok(GqaAttentionPrepared { q, k, v, gate })
}

/// Phase 7 default-on (#1082) — region-3 kt-native consolidation of the
/// GQA full-attention **naive-SDPA fallback**'s two cublasLt matmuls.
///
/// Computes the non-flash fallback `attn_output = softmax((q @ kᵀ)/√hd
/// + causal_mask) @ v` for the GQA-expanded, head-FIRST operands
/// (`q`/`k`/`v` all `[B, nq, T, hd]`), routing the **score matmul**
/// (`q @ kᵀ`) and the **value matmul** (`p @ v`) through the kt substrate
/// (`kiln_tensor::ops::{matmul_rhs_transposed, matmul}`) with the active
/// backend's matmul contract owning native dispatch. The scale,
/// causal mask, and softmax between the two matmuls are computed by the
/// EXISTING candle / kt-softmax-gated ops (`affine` div, the additive `-inf`
/// `broadcast_add` mask via [`apply_causal_mask_with_offset`], and
/// [`cuda_softmax_last_dim`]) — left candle on purpose so the path stays
/// bit-exact with the candle parity oracle (reproducing the affine div and
/// the broadcast `-inf` mask in a different kt kernel risks rounding drift).
///
/// Returns the head-FIRST `attn_output` `[B, nq, T, hd]` (BEFORE the caller's
/// transpose+reshape-back and BEFORE the `try_tape_sdpa_fallback_cuda`
/// adapter) so the caller's tape/decline/reshape logic is byte-for-byte
/// unchanged. Returns `Ok(None)` — falling through to the caller's existing
/// candle `broadcast_matmul` pair — on any of:
/// - `accelerator.kt_api_mode = "disabled"`;
/// - non-CUDA device, non-{BF16,F16,F32} / mixed dtype, or a non-contiguous
///   operand;
/// - autograd-tracked `q` (the candle `loss.backward()` parity oracle keeps
///   the differentiable composite) OR an active tape recording scope (so the
///   caller's `try_tape_sdpa_fallback_cuda` records the analytic backward on
///   the candle-computed `attn_output`, exactly as today);
/// - any kt borrow / matmul / copy-back failure (the candle path then runs).
///
/// Bit-exact to the candle fallback by construction: on CUDA the request ops
/// bottom out in the same cublasLt GEMM family the candle `broadcast_matmul`
/// lowers to, and the score matmul uses the RHS-transposed entry to avoid
/// materialising `k^T`. The intervening scale/mask/softmax ops are physically
/// unchanged. NVTX range `kiln/gqa_sdpa_kt` brackets the migrated region.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_gqa_sdpa_matmuls(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    scale: f64,
) -> Result<Option<(Tensor, Option<kiln_memory::governor::Reservation<'static>>)>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    // The autograd / tape paths must run the candle composite: an
    // autograd-tracked `q` drives the candle `loss.backward()` parity gate
    // (a kt copy-out would sever it), and an active tape scope means the
    // caller's `try_tape_sdpa_fallback_cuda` will record the analytic
    // backward on the candle `attn_output` — so the candle ops must run.
    if q.track_op() || crate::tape_forward::tape_scope_active() {
        return Ok(None);
    }
    // Envelope: head-FIRST, GQA-EXPANDED operands, all [B, nq, T, hd].
    let (qb, qh, qt, qd) = match q.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (kb, kh, kt_, kd) = match k.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (vb, vh, vt, vd) = match v.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    // q/k/v must already be GQA-expanded to identical [B, nq, T, hd] shapes
    // (the caller expands before this call); bail otherwise.
    if qb != kb
        || qb != vb
        || qh != kh
        || qh != vh
        || qt != seq_len
        || kt_ != seq_len
        || vt != seq_len
        || qd != kd
        || qd != vd
    {
        return Ok(None);
    }
    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || k.dtype() != dtype
        || v.dtype() != dtype
        || !matches!(q.device(), Device::Cuda(_))
        || !matches!(k.device(), Device::Cuda(_))
        || !matches!(v.device(), Device::Cuda(_))
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(None);
    }
    let materialized_scratch_reservation = reserve_gqa_materialized_scratch(q, k)?;

    kiln_nvtx::range!(c"kiln/gqa_sdpa_kt");

    // --- Score matmul: scores = q @ kᵀ, [B, nq, T, T] ---
    // #1082 forward-flip: q/k/v are already kt; run the GEMMs, scale/mask/
    // softmax all kt-native — no candle borrow / copy-out round-trips.
    let attn_scores = match kiln_tensor::ops::matmul_rhs_transposed(q, k) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // --- Scale + causal mask + softmax (kt-native) ---
    // kt has no `Tensor / f64`; `x / scale == x * (1/scale)` via affine.
    let attn_scores = attn_scores.affine(1.0 / scale, 0.0)?;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, seq_len, 0)?;
    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;

    // --- Value matmul: attn_output = p @ v, [B, nq, T, hd] ---
    let p_contig = match attn_weights_softmax.contiguous() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let attn_output = match kiln_tensor::ops::matmul(&p_contig, v) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    Ok(Some((attn_output, materialized_scratch_reservation)))
}

#[cfg(feature = "rocm")]
pub(super) fn try_rocm_gqa_sdpa_f32_materialized(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    kv_len: usize,
    scale: f64,
) -> Result<Option<Tensor>> {
    if !crate::rocm_policy::current_rocm_kernel_policy().gqa_sdpa_f32_materialized {
        return Ok(None);
    }
    let dtype = q.dtype();
    if !matches!(q.device(), Device::Rocm(_))
        || !matches!(k.device(), Device::Rocm(_))
        || !matches!(v.device(), Device::Rocm(_))
        || !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || k.dtype() != dtype
        || v.dtype() != dtype
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(None);
    }
    let _materialized_scratch_reservation = reserve_gqa_materialized_scratch(q, k)?;

    let q_f32 = if dtype == DType::F32 {
        q.clone()
    } else {
        q.to_dtype(DType::F32)?
    };
    let q_f32 = fresh_contig_full_attn(&q_f32, "gqa rocm f32 sdpa q cast")?;
    let k_f32 = if dtype == DType::F32 {
        k.clone()
    } else {
        k.to_dtype(DType::F32)?
    };
    let k_f32 = fresh_contig_full_attn(&k_f32, "gqa rocm f32 sdpa k cast")?;
    let v_f32 = if dtype == DType::F32 {
        v.clone()
    } else {
        v.to_dtype(DType::F32)?
    };
    let v_f32 = fresh_contig_full_attn(&v_f32, "gqa rocm f32 sdpa v cast")?;
    synchronize_tensor_ready_for_full_attn_handoff("gqa rocm f32 sdpa q cast", &q_f32)?;
    synchronize_tensor_ready_for_full_attn_handoff("gqa rocm f32 sdpa k cast", &k_f32)?;
    synchronize_tensor_ready_for_full_attn_handoff("gqa rocm f32 sdpa v cast", &v_f32)?;

    let (batch, heads, _, _) = q_f32.dims4()?;
    let mut batch_outputs = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut head_outputs = Vec::with_capacity(heads);
        for h in 0..heads {
            // hipBLASLt strided-batched GEMM has shown silent zeros and
            // impossible finite values for very large SDPA score shapes on
            // ROCm. Keep the exact math, but run batch/head slices as
            // batch-count=1 GEMMs and concatenate only the final outputs.
            let q_bh = q_f32.narrow(0, b, 1)?.narrow(1, h, 1)?;
            let q_bh =
                fresh_contig_full_attn(&q_bh, &format!("ROCm GQA SDPA q slice b={b} h={h}"))?;
            let k_bh = k_f32.narrow(0, b, 1)?.narrow(1, h, 1)?;
            let k_bh =
                fresh_contig_full_attn(&k_bh, &format!("ROCm GQA SDPA k slice b={b} h={h}"))?;
            let v_bh = v_f32.narrow(0, b, 1)?.narrow(1, h, 1)?;
            let v_bh =
                fresh_contig_full_attn(&v_bh, &format!("ROCm GQA SDPA v slice b={b} h={h}"))?;

            let attn_scores = kiln_tensor::rocm_matmul_rhs_transposed(&q_bh, &k_bh)
                .with_context(|| format!("ROCm GQA SDPA score matmul b={b} h={h}"))?;
            synchronize_tensor_ready_for_full_attn_handoff(
                "gqa rocm f32 sdpa score matmul",
                &attn_scores,
            )?;
            let attn_scores = attn_scores.affine(1.0 / scale, 0.0)?;
            let attn_scores =
                apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, kv_len - seq_len)?;
            let attn_scores = fresh_contig_full_attn(
                &attn_scores,
                &format!("gqa rocm f32 sdpa masked scores contiguous b={b} h={h}"),
            )?;
            let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
            synchronize_tensor_ready_for_full_attn_handoff(
                "gqa rocm f32 sdpa softmax",
                &attn_weights_softmax,
            )?;
            let attn_weights_softmax = fresh_contig_full_attn(
                &attn_weights_softmax,
                &format!("gqa rocm f32 sdpa softmax contiguous b={b} h={h}"),
            )?;
            let out_bh = kiln_tensor::rocm_matmul(&attn_weights_softmax, &v_bh)
                .with_context(|| format!("ROCm GQA SDPA value matmul b={b} h={h}"))?;
            synchronize_tensor_ready_for_full_attn_handoff(
                "gqa rocm f32 sdpa value matmul f32",
                &out_bh,
            )?;
            head_outputs.push(out_bh);
        }
        let head_refs: Vec<&Tensor> = head_outputs.iter().collect();
        let heads_out = Tensor::cat(&head_refs, 1)?;
        let heads_out =
            fresh_contig_full_attn(&heads_out, &format!("gqa rocm f32 sdpa head concat b={b}"))?;
        batch_outputs.push(heads_out);
    }
    let batch_refs: Vec<&Tensor> = batch_outputs.iter().collect();
    let out_f32 = Tensor::cat(&batch_refs, 0)?;
    let out_f32 = fresh_contig_full_attn(&out_f32, "gqa rocm f32 sdpa batch concat f32")?;
    synchronize_tensor_ready_for_full_attn_handoff("gqa rocm f32 sdpa concat f32", &out_f32)?;
    let out = if dtype == DType::F32 {
        out_f32
    } else {
        out_f32.to_dtype(dtype)?
    };
    synchronize_tensor_ready_for_full_attn_handoff("gqa rocm f32 sdpa value matmul", &out)?;
    Ok(Some(out))
}

pub(super) fn align_gqa_kv_to_query(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let k = if k.dtype() == q.dtype() {
        k.clone()
    } else {
        k.to_dtype(q.dtype())?
    };
    let v = if v.dtype() == q.dtype() {
        v.clone()
    } else {
        v.to_dtype(q.dtype())?
    };
    // Vulkan LoRA decode keeps the projected query on-device while the
    // portable paged-cache reader materializes K/V on the CPU. Normalize that
    // hybrid boundary before either grouped decode or materialized SDPA.
    #[cfg(feature = "vulkan")]
    let (k, v) = {
        let mut k = k;
        let mut v = v;
        if matches!(q.device(), Device::Cpu | Device::Vulkan(_)) {
            if k.device() != q.device() && matches!(k.device(), Device::Cpu | Device::Vulkan(_)) {
                k = k
                    .to_device(q.device())
                    .context("align Vulkan/CPU SDPA key to query device")?;
            }
            if v.device() != q.device() && matches!(v.device(), Device::Cpu | Device::Vulkan(_)) {
                v = v
                    .to_device(q.device())
                    .context("align Vulkan/CPU SDPA value to query device")?;
            }
        }
        (k, v)
    };
    Ok((k, v))
}

pub(super) fn gqa_sdpa_materialized_default(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    kv_len: usize,
    scale: f64,
) -> Result<Tensor> {
    let (k, v) = align_gqa_kv_to_query(q, k, v)?;
    let attn_scores = kiln_tensor::ops::matmul_rhs_transposed(q, &k)?;
    let attn_scores = attn_scores.affine(1.0 / scale, 0.0)?;
    let attn_scores =
        apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, kv_len - seq_len)?;
    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let out = attn_weights_softmax.broadcast_matmul(&v)?; // [B, nq, T, hd]
    synchronize_tensor_ready_for_full_attn_handoff("gqa sdpa fallback value matmul", &out)?;
    Ok(out)
}

pub(super) fn reserve_gqa_materialized_scratch(
    q: &Tensor,
    k: &Tensor,
) -> Result<Option<kiln_memory::governor::Reservation<'static>>> {
    if !full_attn_materialized_scores_for_device(&q.device()) {
        return Ok(None);
    }
    let selector = q.device().memory_probe_selector();
    if kiln_memory::MemoryGovernor::global_configuration().selector != selector {
        anyhow::bail!(
            "materialized attention rejected: the memory governor does not match the active device (device={:?}, expected_selector={selector:?})",
            q.device()
        );
    }
    let (batch, heads, query_tokens, _) = q.dims4()?;
    let (_, _, key_tokens, _) = k.dims4()?;
    let scratch_bytes = full_attn_materialized_scratch_bytes(
        q.dtype(),
        batch,
        heads,
        query_tokens,
        key_tokens,
        MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
    )
    .ok_or_else(|| anyhow::anyhow!("materialized attention scratch-byte calculation overflowed"))?;
    let reservation = kiln_memory::MemoryGovernor::try_global_cached_reserve(scratch_bytes)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "materialized attention rejected: concurrent memory reservations exhausted the published scratch budget (requested_bytes={scratch_bytes}, batch={batch}, heads={heads}, query_tokens={query_tokens}, key_tokens={key_tokens})"
            )
        })?;
    Ok(Some(reservation))
}

pub(super) fn gqa_sdpa_materialized_exact(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    kv_len: usize,
    scale: f64,
) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    if let Some(out) = try_rocm_gqa_sdpa_f32_materialized(q, k, v, seq_len, kv_len, scale)? {
        return Ok(out);
    }

    let _materialized_scratch_reservation = reserve_gqa_materialized_scratch(q, k)?;
    gqa_sdpa_materialized_default(q, k, v, seq_len, kv_len, scale)
}

pub fn gqa_attention_core_prefill(
    backend: &dyn BackendRuntime,
    prepared: &GqaAttentionPrepared,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (_batch, seq_len, _heads, _hd) = prepared.q.dims4()?;
    if seq_len > 1 && AttentionBackend::runtime_supports_flash_attn_prefill(backend) {
        let q = tape_contig_full_attn(&prepared.q, "gqa prefill q")?;
        let k = tape_contig_full_attn(&prepared.k, "gqa prefill k")?;
        let v = tape_contig_full_attn(&prepared.v, "gqa prefill v")?;
        // Route the GQA flash path through the kt Tape when a scope is active,
        // so a
        // tape-authoritative backward reaches the q/k/v (LoRA) projections.
        // No-ops (returns None) in every other configuration — default
        // training/inference is unchanged and falls through below.
        // #1082: tape flash-attn + the training CustomOp are candle islands.
        // Bridge q/k/v kt->candle only when the tape is active / training.
        // #1082 seam flip: kt-native flash-attn + reshape recorders — no kt->candle->kt
        // at the attention seam (q/k/v + the attn output stay kt; the downstream
        // reshape chains kt-native to o_proj).
        #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
        if crate::tape_forward::tape_scope_active() {
            if let Some(attn_output) = crate::tape_forward::try_tape_flash_attn_kt(
                &q,
                &k,
                &v,
                num_heads,
                num_kv_heads,
                head_dim,
            )? {
                let (rb, rs, rh, rd) = attn_output.dims4()?;
                let flat = rh * rd;
                return require_active_tape_output(
                    crate::tape_forward::try_tape_reshape_kt(&attn_output, vec![rb, rs, flat])
                        .context("gqa flash output try_tape_reshape_kt")?,
                    "GQA flash-attention output reshape",
                );
            }
        }
        // (#1082) Deleted the dead candle-CustomOp `cuda_flash_attention_training_bf16`
        // branch: the kt tape's `try_tape_flash_attn_kt` above is the sole
        // flash-attention autograd producer.
        //
        // (#1082) SKIP the leaf `flash_attention_forward` kernel when a tape
        // scope is active: it neither records a backward NOR (on Vulkan)
        // preserves device residency — the Vulkan flash-attn prefill kernel
        // returns a CPU-host kt tensor at the kt<->vk seam, which then breaks
        // the very next op (o_proj matmul: a=cpu, b=vulkan). During training we
        // fall through to the device-resident, tape-recording SDPA composite +
        // `try_tape_sdpa_fallback_kt` below. Inference (no tape scope) keeps the
        // fast leaf kernel unchanged on every backend. On a default build
        // (no tape-feature) `tape_forward` is cfg'd out, so the leaf always runs.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        let skip_leaf_flash = crate::tape_forward::tape_scope_active();
        #[cfg(not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )))]
        let skip_leaf_flash = false;
        let leaf_flash_allowed = flash_prefill_allowed_for_shape(
            backend,
            &prepared.q.device(),
            prepared.q.dtype(),
            head_dim,
            seq_len,
            prepared.k.dim(1).unwrap_or(seq_len),
        );
        if !skip_leaf_flash
            && leaf_flash_allowed
            && let Some(attn_output) =
                flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            return Ok(attn_output);
        }
    }

    // Head-FIRST, PRE-GQA-expand q/k/v ([B, nq, T, hd] / [B, nkv, T, hd]):
    // the naive SDPA fallback consumes these. Keep references to them for the
    // #1082 CP-4 kt-Tape SDPA adapter below (it derives the analytic backward
    // from these pre-expand tensors and GQA-collapses dk/dv back to nkv).
    let q = tape_transpose_contig_full_attn(&prepared.q, 1, 2)?;
    let k_he = tape_transpose_contig_full_attn(&prepared.k, 1, 2)?;
    let v_he = tape_transpose_contig_full_attn(&prepared.v, 1, 2)?;
    let kv_len = k_he.dim(2)?;
    anyhow::ensure!(
        v_he.dim(2)? == kv_len,
        "gqa_attention_core_prefill: k/v prefix lengths differ: k={kv_len} v={}",
        v_he.dim(2)?
    );
    anyhow::ensure!(
        kv_len >= seq_len,
        "gqa_attention_core_prefill: prefix attention requires kv_len >= q_len, got q_len={seq_len} kv_len={kv_len}"
    );
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k_he.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        let k = k_he
            .unsqueeze(2)?
            .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v_he
            .unsqueeze(2)?
            .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k_he.contiguous()?, v_he.contiguous()?)
    };
    let scale = (head_dim as f64).sqrt();
    // #1082 region 3: consolidate the fallback's two cublasLt matmuls
    // (`q @ kᵀ` and `softmax @ v`) into the kt substrate, keeping the
    // intervening scale / causal-mask / softmax on candle for bit-exactness.
    // Fires ONLY on the plain inference path (gate on, !track_op, no tape
    // scope); declines to `None` otherwise so the candle composite below runs
    // unchanged — including for the tape/decline paths the adapter relies on.
    // The helper returns the SAME head-FIRST `[B, nq, T, hd]` `attn_output`,
    // so the `try_tape_sdpa_fallback_cuda` adapter + transpose/reshape-back
    // logic below is byte-for-byte identical regardless of which path ran.
    #[cfg(feature = "cuda")]
    let mut _cuda_fast_scratch_reservation = None;
    let attn_output = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_gqa_sdpa_matmuls(&q, &k, &v, seq_len, scale)? {
                Some((out, reservation)) => {
                    _cuda_fast_scratch_reservation = reservation;
                    out
                }
                None => gqa_sdpa_materialized_exact(&q, &k, &v, seq_len, kv_len, scale)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            gqa_sdpa_materialized_exact(&q, &k, &v, seq_len, kv_len, scale)?
        }
    };

    // Route the GQA SDPA fallback through the kt Tape when a scope is active,
    // so a
    // tape-authoritative backward reaches the q/k/v (LoRA) projections on the
    // NON-flash path (head_dim ∉ {128,256}). No-ops (returns None) in every
    // other configuration — default training/inference is unchanged and falls
    // through to the plain candle transpose+reshape below. Records on the
    // head-FIRST `attn_output` (BEFORE the reshape-back), with the pre-expand
    // head-first q/k_he/v_he as inputs; then chains the transpose+reshape so
    // the tape stays connected to o_proj (else it fragments at the reshape).
    // #1082 seam flip: kt-native SDPA-fallback + transpose + reshape recorders — no
    // kt->candle->kt at the SDPA seam (q/k/v + attn output stay kt; the downstream
    // transpose->reshape chains kt-native to o_proj).
    // (#1082) Vulkan added: the flash-attn path (CUDA FFI) declines on Vulkan,
    // so the device-agnostic `try_tape_sdpa_fallback_kt` is the attention
    // backward producer on Vulkan; without it, grads sever between v_proj and
    // o_proj.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let tape_attn = require_active_tape_output(
            crate::tape_forward::try_tape_sdpa_fallback_kt(
                &q,
                &k_he,
                &v_he,
                head_dim,
                &attn_output,
            )
            .context("gqa SDPA try_tape_sdpa_fallback_kt")?,
            "GQA SDPA fallback",
        )?;
        let transposed = require_active_tape_output(
            crate::tape_forward::try_tape_transpose_kt(&tape_attn, 1, 2)
                .context("gqa SDPA output try_tape_transpose_kt")?,
            "GQA SDPA output transpose",
        )?;
        let (tb, tt, th, td) = transposed.dims4()?;
        let flat = th * td;
        return require_active_tape_output(
            crate::tape_forward::try_tape_reshape_kt(&transposed, vec![tb, tt, flat])
                .context("gqa SDPA output try_tape_reshape_kt")?,
            "GQA SDPA output reshape",
        );
    }

    Ok(reshape_hole0_3(
        &attn_output.transpose(1, 2)?.contiguous()?,
        seq_len,
        num_heads * head_dim,
    )?)
}

pub fn gqa_attention_apply_output_gate(
    attn_output: Tensor,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    attention_output_gate_decode_if(false, attn_output, gate)
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_pre_o_chunked_prefill(
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
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    tile_size: usize,
) -> Result<Tensor> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    anyhow::ensure!(
        !crate::tape_forward::tape_scope_active(),
        "gqa_attention_pre_o_chunked_prefill is inference-only until tiled narrow, reshape, and concat operations are tape-recorded"
    );
    let (_batch, seq_len, _hidden) = x.dims3()?;
    if tile_size == 0 || tile_size >= seq_len {
        return gqa_attention_pre_o(
            backend,
            x,
            attn_weights,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            inv_freq,
            rms_norm_eps,
            None,
            0,
            attn_output_gate,
            lora,
        );
    }

    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };

    let k_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention pre-o k projection")?;
    let k = reshape_hole0_4(&k_flat, seq_len, num_kv_heads, head_dim)
        .context("chunked full-attention pre-o k reshape")?;
    let v_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention pre-o v projection")?;
    let v = reshape_hole0_4(&v_flat, seq_len, num_kv_heads, head_dim)
        .context("chunked full-attention pre-o v reshape")?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)
        .context("chunked full-attention pre-o k norm")?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)
        .context("chunked full-attention pre-o k rotary")?;

    let mut output_tiles = Vec::with_capacity(seq_len.div_ceil(tile_size));
    let mut tile_start = 0usize;
    while tile_start < seq_len {
        let tile_len = (seq_len - tile_start).min(tile_size);
        let tile_end = tile_start + tile_len;

        let x_tile = x.narrow(1, tile_start, tile_len).with_context(|| {
            format!("chunked full-attention pre-o input tile [{tile_start}, {tile_end})")
        })?;
        let q_raw = q_proj_forward_decode_if(
            Some(backend),
            false,
            &x_tile,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )
        .with_context(|| {
            format!("chunked full-attention pre-o q projection [{tile_start}, {tile_end})")
        })?;
        let (q_tile, gate_tile) = if attn_output_gate {
            let q_raw =
                reshape_hole0_4(&q_raw, tile_len, num_heads, head_dim * 2).with_context(|| {
                    format!(
                        "chunked full-attention pre-o q/gate reshape [{tile_start}, {tile_end})"
                    )
                })?;
            let q = q_raw
                .narrow(3, 0, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o q split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention pre-o q contiguous")?;
            let gate_split = q_raw
                .narrow(3, head_dim, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o gate split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention pre-o gate contiguous")?;
            let gate = reshape_hole0_3(&gate_split, tile_len, num_heads * head_dim)
                .context("chunked full-attention pre-o gate reshape")?;
            (q, Some(gate))
        } else {
            (
                reshape_hole0_4(&q_raw, tile_len, num_heads, head_dim).with_context(|| {
                    format!("chunked full-attention pre-o q reshape [{tile_start}, {tile_end})")
                })?,
                None,
            )
        };
        let q_tile = rms_norm(&q_tile, &attn_weights.q_norm, rms_norm_eps)
            .context("chunked full-attention pre-o q norm")?;
        let tile_positions = &positions[tile_start..tile_end];
        let (q_tile, _) = rotary_embedding(
            &q_tile,
            &q_tile,
            tile_positions,
            head_dim,
            rotary_dim,
            inv_freq,
        )
        .with_context(|| {
            format!("chunked full-attention pre-o q rotary [{tile_start}, {tile_end})")
        })?;
        let k_prefix = k.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention pre-o k prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let v_prefix = v.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention pre-o v prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let tile_prepared = GqaAttentionPrepared {
            q: q_tile,
            k: k_prefix,
            v: v_prefix,
            gate: None,
        };
        let attn_core =
            gqa_attention_core_prefill(backend, &tile_prepared, num_heads, num_kv_heads, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o core tile [{tile_start}, {tile_end})")
                })?;
        synchronize_tensor_ready_for_model_handoff(
            &format!("chunked full-attention pre-o core tile [{tile_start}, {tile_end})"),
            &attn_core,
        )?;
        let attn_output = gqa_attention_apply_output_gate(attn_core, gate_tile.as_ref())
            .with_context(|| {
                format!("chunked full-attention pre-o gate tile [{tile_start}, {tile_end})")
            })?;
        synchronize_tensor_ready_for_model_handoff(
            &format!("chunked full-attention pre-o output tile [{tile_start}, {tile_end})"),
            &attn_output,
        )?;
        output_tiles.push(attn_output);

        tile_start = tile_end;
    }

    let output_refs: Vec<&Tensor> = output_tiles.iter().collect();
    let output = Tensor::cat(&output_refs, 1).context("chunked full-attention pre-o cat")?;
    synchronize_tensor_ready_for_model_handoff("chunked full-attention pre-o cat", &output)?;
    Ok(output)
}

/// CP-4 (#1082) Increment 7 helper: reshape a candle tensor, routing through
/// the kt `Tape` (`try_tape_reshape_cuda`) when a tape scope is active so the
/// reshape stays connected to the upstream producer (chains its input) and
/// becomes a retained output the downstream consumer can pick up. Falls through
/// to a plain candle `.reshape()` when the tape adapter declines (no scope,
/// non-CUDA, or envelope miss).
///
/// `dims` is the candle reshape spec (may contain one inferred `()` axis); the
/// kt reshape needs concrete dims, so the inferred axis is resolved from the
/// input's element count before recording.
pub(super) fn tape_reshape_full_attn(x: &Tensor, dims: &[ReshapeArg]) -> Result<Tensor> {
    // #1082 seam flip: kt-native reshape recorder — no kt->candle->kt.
    // (#1082) Vulkan added: device-agnostic pure-kt recorder.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let concrete = resolve_reshape_dims(x.elem_count(), dims)
            .context("active tape scope could not resolve full-attention reshape dimensions")?;
        return require_active_tape_output(
            crate::tape_forward::try_tape_reshape_kt(x, concrete)
                .context("tape_reshape_full_attn try_tape_reshape_kt")?,
            "full-attention reshape",
        );
    }
    // kt reshape with the (possibly inferred) spec.
    candle_reshape_with_spec(x, dims)
}

pub(super) fn tape_narrow_contig_full_attn(
    x: &Tensor,
    axis: usize,
    offset: usize,
    length: usize,
    context: &str,
    contiguous_without_tape: bool,
) -> Result<Tensor> {
    let narrowed = x
        .narrow(axis, offset, length)
        .with_context(|| format!("{context} narrow"))?;
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        if crate::tape_forward::tape_scope_active() {
            let narrowed =
                crate::tape_forward::try_tape_narrow_kt(x, axis, offset, length, &narrowed)
                    .with_context(|| format!("{context} try_tape_narrow_kt"))?
                    .with_context(|| format!("{context} failed to record tape narrow"))?;
            return crate::tape_forward::try_tape_contiguous_kt(&narrowed)
                .with_context(|| format!("{context} try_tape_contiguous_kt"))?
                .with_context(|| format!("{context} failed to record tape contiguous"));
        }
    }
    if contiguous_without_tape {
        fresh_contig_full_attn(&narrowed, context)
    } else {
        Ok(narrowed)
    }
}

pub(super) fn tape_contig_full_attn(x: &Tensor, context: &str) -> Result<Tensor> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        if crate::tape_forward::tape_scope_active() {
            return crate::tape_forward::try_tape_contiguous_kt(x)
                .with_context(|| format!("{context} try_tape_contiguous_kt"))?
                .with_context(|| format!("{context} failed to record tape contiguous"));
        }
    }
    x.contiguous()
        .with_context(|| format!("{context} contiguous"))
}

pub(super) fn fresh_contig_full_attn(x: &Tensor, context: &str) -> Result<Tensor> {
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            return kiln_tensor::cuda_contiguous(x)
                .with_context(|| format!("{context} fresh cuda contiguous"));
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            return kiln_tensor::rocm_contiguous(x)
                .with_context(|| format!("{context} fresh rocm contiguous"));
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => {
            return kiln_tensor::vulkan_contiguous(x)
                .with_context(|| format!("{context} fresh vulkan contiguous"));
        }
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            return kiln_tensor::metal_deep_copy(x)
                .with_context(|| format!("{context} fresh metal contiguous"));
        }
        _ => {}
    }
    x.contiguous()
        .with_context(|| format!("{context} fresh contiguous"))
}

/// CP-4 (#1082) Increment 7 helper: `transpose(axis_a, axis_b)` + `contiguous`,
/// routing through the kt `Tape` (`try_tape_transpose_cuda`, which materialises
/// a contiguous output) when a tape scope is active so the chain stays
/// connected across the naive-SDPA layout transpose. Falls through to the plain
/// candle `transpose().contiguous()` otherwise.
pub(super) fn tape_transpose_contig_full_attn(
    x: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> Result<Tensor> {
    // #1082 seam flip: kt-native transpose recorder — no kt->candle->kt.
    // (#1082) Vulkan added: device-agnostic pure-kt recorder.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_transpose_kt(x, axis_a, axis_b)
                .context("tape_transpose_contig_full_attn try_tape_transpose_kt")?,
            "full-attention transpose",
        );
    }
    Ok(x.transpose(axis_a, axis_b)?.contiguous()?)
}

/// A single axis spec for [`tape_reshape_full_attn`]: either a fixed size or the
/// single inferred axis (`Infer`, the candle `()` placeholder).
#[derive(Clone, Copy)]
pub(super) enum ReshapeArg {
    Size(usize),
    Infer,
}

impl From<usize> for ReshapeArg {
    fn from(v: usize) -> Self {
        ReshapeArg::Size(v)
    }
}

/// Resolve a reshape spec (with at most one `Infer` axis) to concrete dims given
/// the total element count. Returns `None` if the spec is inconsistent (e.g. the
/// fixed axes don't divide the element count, or more than one `Infer`).
pub(super) fn resolve_reshape_dims(elem_count: usize, dims: &[ReshapeArg]) -> Option<Vec<usize>> {
    let mut infer_idx: Option<usize> = None;
    let mut known: usize = 1;
    for (i, d) in dims.iter().enumerate() {
        match d {
            ReshapeArg::Size(s) => known = known.checked_mul(*s)?,
            ReshapeArg::Infer => {
                if infer_idx.is_some() {
                    return None; // more than one inferred axis
                }
                infer_idx = Some(i);
            }
        }
    }
    let mut out: Vec<usize> = Vec::with_capacity(dims.len());
    match infer_idx {
        Some(idx) => {
            if known == 0 || elem_count % known != 0 {
                return None;
            }
            let inferred = elem_count / known;
            for (i, d) in dims.iter().enumerate() {
                out.push(if i == idx {
                    inferred
                } else if let ReshapeArg::Size(s) = d {
                    *s
                } else {
                    unreachable!()
                });
            }
        }
        None => {
            if known != elem_count {
                return None;
            }
            for d in dims {
                if let ReshapeArg::Size(s) = d {
                    out.push(*s);
                }
            }
        }
    }
    Some(out)
}

/// Candle reshape honouring the [`ReshapeArg`] spec (resolves the inferred axis
/// the same way candle's tuple-with-`()` reshape does).
pub(super) fn candle_reshape_with_spec(x: &Tensor, dims: &[ReshapeArg]) -> Result<Tensor> {
    match resolve_reshape_dims(x.elem_count(), dims) {
        Some(concrete) => Ok(x.reshape(concrete)?),
        None => anyhow::bail!(
            "tape_reshape_full_attn: inconsistent reshape spec for {} elements",
            x.elem_count()
        ),
    }
}

/// Returns the gated attention value before the final output projection:
/// [batch, seq_len, num_heads * head_dim].
pub fn gqa_attention_pre_o(
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
    let use_metal_decode_gemv = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_runtime::single_token_self_attention_active();

    // Project to Q, K, V (with optional LoRA delta)
    // When attn_output_gate is true, q_proj outputs [Q, gate] fused:
    //   q_proj: [num_heads * head_dim * 2, hidden_size]
    //   Split into Q [num_heads, head_dim] and gate [num_heads, head_dim]
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = if attn_output_gate {
        split_q_gate_training_bf16(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
            seq_len,
            num_heads,
            head_dim,
        )?
    } else {
        None
    };

    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        if split_q_gate.is_some() {
            let k = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.k_proj_t,
                lora_layer.and_then(|l| l.k_proj.as_ref()),
                lora_scale,
            )?;
            let v = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.v_proj_t,
                lora_layer.and_then(|l| l.v_proj.as_ref()),
                lora_scale,
            )?;
            (None, k, v)
        } else {
            let (q_raw, k, v) = full_attn_qkv_proj_decode_if(
                backend,
                use_metal_decode_gemv,
                x,
                attn_weights,
                lora_layer,
                lora_scale,
            )?;
            (Some(q_raw), k, v)
        }
    };
    // Split Q and gate if output gate is enabled
    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else if attn_output_gate {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when split_q_gate is inactive");
        // q_raw: [batch, seq_len, num_heads * head_dim * 2]
        // Reshape to [batch, seq_len, num_heads, head_dim * 2] then split
        let q_raw = tape_reshape_full_attn(
            q_raw,
            &[
                ReshapeArg::Infer,
                seq_len.into(),
                num_heads.into(),
                (head_dim * 2).into(),
            ],
        )?;
        let q =
            tape_narrow_contig_full_attn(&q_raw, 3, 0, head_dim, "full-attention q split", true)?;
        let gate = tape_narrow_contig_full_attn(
            &q_raw,
            3,
            head_dim,
            head_dim,
            "full-attention gate split",
            true,
        )?;
        // gate needs to be [batch, seq_len, num_heads * head_dim] for later
        let gate = tape_reshape_full_attn(
            &gate,
            &[
                ReshapeArg::Infer,
                seq_len.into(),
                (num_heads * head_dim).into(),
            ],
        )?;
        (q, Some(gate))
    } else {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when attention output gate is disabled");
        // CP-4 (#1082) Increment 7: route the q-projection reshape
        // ([B, S, H*hd] -> [B, S, H, hd]) through the kt `Tape` so the chain
        // from q_proj (the LoRA keystone) stays connected into q_norm/rope/SDPA
        // on the naive SDPA-fallback path (tiny-model head_dim<128). Plain
        // candle `.reshape()` mints a fresh id that severs the tape, leaving
        // q_norm's input a fresh-borrow island. Falls through to candle when
        // the gate is off / no tape scope / non-CUDA.
        let q = tape_reshape_full_attn(
            q_raw,
            &[
                ReshapeArg::Infer,
                seq_len.into(),
                num_heads.into(),
                head_dim.into(),
            ],
        )?;
        (q, None)
    };

    // Reshape K, V to [batch, seq_len, num_heads, head_dim]
    let (k, v) = {
        // CP-4 (#1082) Increment 7: same as the q reshape above — tape-wire the
        // K/V reshapes so the chain from k_proj/v_proj reaches k_norm (K) and
        // the value matmul (V) on the SDPA-fallback path.
        (
            tape_reshape_full_attn(
                &k,
                &[
                    ReshapeArg::Infer,
                    seq_len.into(),
                    num_kv_heads.into(),
                    head_dim.into(),
                ],
            )?,
            tape_reshape_full_attn(
                &v,
                &[
                    ReshapeArg::Infer,
                    seq_len.into(),
                    num_kv_heads.into(),
                    head_dim.into(),
                ],
            )?,
        )
    };

    // Apply per-head RMSNorm to Q and K (Qwen3.5 uses QK-norm)
    // q_norm/k_norm are [head_dim] — broadcast over [batch, seq_len, num_heads, head_dim]
    let (q, k) = {
        (
            rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?,
            rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?,
        )
    };

    // Apply RoPE (positions are absolute, so cached tokens get correct embeddings)
    // Only rotate first rotary_dim dimensions; the rest pass through unchanged.
    let (q, k) = { rotary_embedding(&q, &k, positions, head_dim, rotary_dim, inv_freq)? };

    // Fused-attention path for prefill (seq_len > 1, no KV cache).
    // Takes [batch, seq_len, num_heads, head_dim] — the layout we already
    // have. When a KV cache is present we fall through to the naive path,
    // which handles the cache update and Q_len != KV_len masking correctly.
    // Backend declines (returns None) on dtype mismatch so non-BF16 configs
    // (e.g. tests on F32) transparently fall back to naive softmax+matmul.
    if seq_len > 1
        && kv_cache.is_none()
        && AttentionBackend::runtime_supports_flash_attn_prefill(backend)
    {
        let q = tape_contig_full_attn(&q, "full-attention flash q")?;
        let k = tape_contig_full_attn(&k, "full-attention flash k")?;
        let v = tape_contig_full_attn(&v, "full-attention flash v")?;
        // kt-Tape flash path (see gqa_attention_core_prefill). No-op without an
        // active tape scope. NOTE:
        // when an attention output gate is present, the gate multiply is not
        // yet tape-recorded, so the tape chain ends at the reshape for
        // gate-on models (a follow-up gate adapter closes that); gate-off
        // (e.g. Qwen3.5-4B default) chains straight through to o_proj.
        // #1082 seam flip: kt-native flash-attn + reshape recorders — no kt->candle->kt.
        #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
        if crate::tape_forward::tape_scope_active() {
            if let Some(attn_output) = crate::tape_forward::try_tape_flash_attn_kt(
                &q,
                &k,
                &v,
                num_heads,
                num_kv_heads,
                head_dim,
            )? {
                let (rb, rs, rh, rd) = attn_output.dims4()?;
                let flat = rh * rd;
                let attn_kt = require_active_tape_output(
                    crate::tape_forward::try_tape_reshape_kt(&attn_output, vec![rb, rs, flat])
                        .context("full-attention flash output try_tape_reshape_kt")?,
                    "full-attention flash output reshape",
                )?;
                let attn_kt = attention_output_gate_decode_if(false, attn_kt, gate.as_ref())?;
                return Ok(attn_kt);
            }
        }
        // (#1082) Deleted the dead candle-CustomOp `cuda_flash_attention_training_bf16`
        // branch: `try_tape_flash_attn_kt` above is the sole flash-attn autograd
        // producer.
        //
        // (#1082) SKIP the leaf `flash_attention_forward` kernel when a tape
        // scope is active: it neither records a backward NOR (on Vulkan)
        // preserves device residency — the Vulkan flash-attn prefill kernel
        // returns a CPU-host kt tensor at the kt<->vk seam, which then breaks
        // the very next op (o_proj matmul: a=cpu, b=vulkan). During training we
        // fall through to the device-resident, tape-recording SDPA composite +
        // `try_tape_sdpa_fallback_kt` below. Inference (no tape scope) keeps the
        // fast leaf kernel unchanged on every backend. On a default build
        // (no tape-feature) `tape_forward` is cfg'd out, so the leaf always runs.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        let skip_leaf_flash = crate::tape_forward::tape_scope_active();
        #[cfg(not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )))]
        let skip_leaf_flash = false;
        if !skip_leaf_flash
            && let Some(attn_output) =
                flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            return Ok(attn_output);
        }
    }

    // Transpose to [batch, heads, seq_len, head_dim] for naive attention.
    //
    // CP-4 (#1082) Increment 7: route the layout transpose through the kt
    // `Tape` (`tape_transpose_contig_full_attn`) so the chain from rope (Q/K)
    // and the V reshape stays connected to the SDPA-fallback inputs. A plain
    // candle `transpose().contiguous()` would mint a fresh id and sever the
    // tape between rope/reshape and SDPA. Falls through to candle otherwise.
    let (q, k, v) = {
        (
            tape_transpose_contig_full_attn(&q, 1, 2)?,
            tape_transpose_contig_full_attn(&k, 1, 2)?,
            tape_transpose_contig_full_attn(&v, 1, 2)?,
        )
    };

    // CP-4 (#1082) Increment 7: keep references to the head-FIRST, PRE-GQA-
    // expand q/k/v ([B, nq, T, hd] / [B, nkv, T, hd]) so the naive-SDPA tape
    // adapter (below) can record `SdpaBackward` from them — mirroring
    // `gqa_attention_core_prefill`, which records on the pre-expand tensors and
    // GQA-collapses dk/dv back to nkv. Only meaningful when there's no KV cache
    // (the tape-authoritative SFT path), so capture before the cache update.
    // (#1082) Vulkan added: the SDPA fallback is the attention backward producer
    // on Vulkan (the flash leaf is skipped under a tape scope above).
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let sdpa_pre_expand = if kv_cache.is_none() {
        Some((q.clone(), k.clone(), v.clone()))
    } else {
        None
    };

    // If KV cache is provided, update it and use full cached K/V
    let (k, v, kv_len) = if let Some(cache) = kv_cache {
        // #1082: the contiguous `KvCache` is now kt-native (token-major
        // storage, kt `slice_set`/`narrow`). K/V flow straight through — no
        // candle bridge.
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
        (
            k.unsqueeze(2)?
                .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
                .contiguous()?
                .reshape((batch, num_heads, kv_len, head_dim))?,
            v.unsqueeze(2)?
                .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
                .contiguous()?
                .reshape((batch, num_heads, kv_len, head_dim))?,
        )
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    // Q: [batch, num_heads, seq_len, head_dim]
    // K: [batch, num_heads, kv_len, head_dim]
    // scores: [batch, num_heads, seq_len, kv_len]
    let scale = (head_dim as f64).sqrt();
    let attn_scores = {
        let out = kiln_tensor::ops::matmul_rhs_transposed(&q, &k)?;
        // kt has no `Tensor / f64`; `x / scale == x * (1/scale)` via affine.
        out.affine(1.0 / scale, 0.0)?
    };

    // Apply causal mask (handles Q_len != KV_len for cached decoding)
    let past_len = kv_len - seq_len;
    let attn_scores = { apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)? };

    let attn_weights_softmax = { cuda_softmax_last_dim(&attn_scores)? };
    let attn_output = {
        let out = attn_weights_softmax.broadcast_matmul(&v)?; // [batch, num_heads, seq_len, head_dim]
        out
    };

    // Route the naive SDPA fallback through the kt `Tape` when a scope is
    // active, so a
    // tape-authoritative backward reaches the q/k/v (LoRA) projections on the
    // full-attention SDPA-fallback path (head_dim ∉ {128,256}, e.g. the tiny
    // test model). Records on the head-FIRST `attn_output` (BEFORE the
    // transpose-back), with the pre-GQA-expand head-first q/k/v as inputs (the
    // `SdpaBackward` adjoint GQA-collapses dk/dv back to nkv), then chains the
    // transpose-back + reshape so the chain reaches o_proj. No-ops (returns
    // None) in every other configuration; mirrors `gqa_attention_core_prefill`.
    // (#1082) Vulkan added: device-agnostic SDPA-fallback + transpose + reshape
    // recorders — the attention backward producer on Vulkan.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    // #1082 seam flip: kt-native SDPA-fallback + transpose + reshape recorders.
    if crate::tape_forward::tape_scope_active() {
        let (q_pe, k_pe, v_pe) = sdpa_pre_expand
            .as_ref()
            .context("active tape scope requires pre-expand SDPA inputs")?;
        let tape_attn = require_active_tape_output(
            crate::tape_forward::try_tape_sdpa_fallback_kt(
                q_pe,
                k_pe,
                v_pe,
                head_dim,
                &attn_output,
            )
            .context("full-attention SDPA try_tape_sdpa_fallback_kt")?,
            "full-attention SDPA fallback",
        )?;
        let transposed = require_active_tape_output(
            crate::tape_forward::try_tape_transpose_kt(&tape_attn, 1, 2)
                .context("full-attention SDPA output try_tape_transpose_kt")?,
            "full-attention SDPA output transpose",
        )?;
        let (tb, tt, th, td) = transposed.dims4()?;
        let flat = th * td;
        let reshaped = require_active_tape_output(
            crate::tape_forward::try_tape_reshape_kt(&transposed, vec![tb, tt, flat])
                .context("full-attention SDPA output try_tape_reshape_kt")?,
            "full-attention SDPA output reshape",
        )?;
        let attn_output =
            attention_output_gate_decode_if(use_metal_decode_gemv, reshaped, gate.as_ref())?;
        return Ok(attn_output);
    }

    // Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
    let attn_output = {
        reshape_hole0_3(
            &attn_output.transpose(1, 2)?.contiguous()?,
            seq_len,
            num_heads * head_dim,
        )?
    };

    let attn_output =
        attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())?;
    Ok(attn_output)
}

pub fn gqa_attention_output_projection(
    backend: &dyn BackendRuntime,
    attn_output: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    use_metal_decode_gemv: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    kiln_nvtx::range!(c"kiln/proj/o");
    #[cfg(feature = "rocm")]
    if !crate::tape_forward::tape_scope_active()
        && lora_layer.is_none()
        && attn_output.dtype() == DType::BF16
        && !attn_output.track_op()
        && matches!(attn_output.device(), Device::Rocm(_))
        && let Some(o_w8) = attn_weights.o_proj_w8.as_ref()
    {
        if let Ok((_, seq_len, hidden)) = attn_output.dims3() {
            if seq_len == 1 && hidden == o_w8.k {
                return crate::rocm_w8_proj::matmul_bf16(attn_output, o_w8)
                    .context("rocm w8 full-attn output projection");
            }
        }
    }
    linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        attn_output,
        &attn_weights.o_proj_t,
        lora_layer.and_then(|l| l.o_proj.as_ref()),
        lora_scale,
    )
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
    let use_metal_decode_gemv = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_runtime::single_token_self_attention_active();
    let attn_output = gqa_attention_pre_o(
        backend,
        x,
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
        attn_output_gate,
        lora,
    )?;
    let projected = gqa_attention_output_projection(
        backend,
        &attn_output,
        attn_weights,
        use_metal_decode_gemv,
        lora,
    )?;
    Ok(projected)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct PagedDecodeGraphInputs<'a> {
    pub block_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub kv_slot: &'a Tensor,
    pub max_seqlen_k: usize,
    pub rotary_cos: &'a Tensor,
    pub rotary_sin: &'a Tensor,
    pub attn_out: &'a [Tensor],
    pub softmax_lse: &'a [Tensor],
}

/// Marker returned when a ROCm graph-backed decode would fall through to the
/// sequence-length-shaped attention implementation. Capturing that fallback
/// bakes the current K/V length into tensor shapes, so replaying it for a later
/// token is incorrect even though the HIP graph launch itself succeeds.
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub(crate) struct RocmGraphShapeDependentAttention;

#[cfg(feature = "rocm")]
impl std::fmt::Display for RocmGraphShapeDependentAttention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "ROCm graph capture requires graph-stable native paged attention; \
             the current decode geometry reached shape-dependent attention",
        )
    }
}

#[cfg(feature = "rocm")]
impl std::error::Error for RocmGraphShapeDependentAttention {}

#[cfg(feature = "rocm")]
pub(crate) fn is_rocm_graph_shape_dependent_attention(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<RocmGraphShapeDependentAttention>()
        .is_some()
}

/// Stable per-step inputs threaded through batched CUDA/ROCm graph
/// capture/replay path. Mirrors [`PagedDecodeGraphInputs`] but every
/// tensor is shaped for `[batch, …]`. The graph runner pre-allocates
/// these on the device once per `batch_size` bucket; per-step updates
/// rewrite their contents in place so the
/// captured kernels read from the same device pointers on every replay.
/// Consumed by [`model_forward_paged_batched_hidden_with_graph_inputs`]
/// (the HiddenOnly batched capture forward — #1082 boxes 432/433).
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[allow(dead_code)]
pub(crate) struct BatchedPagedDecodeGraphInputs<'a> {
    /// `[batch]` u32 token-id buffer.
    pub token_ids: &'a Tensor,
    /// `[batch]` f32 per-row decode position.
    pub positions: &'a Tensor,
    /// `[batch, max_blocks_per_seq]` u32 padded block table.
    pub block_table: &'a Tensor,
    /// `[batch]` i32 per-row K/V length.
    pub seqused_k: &'a Tensor,
    /// `[batch]` u32 per-row current KV-write slot.
    pub kv_slot: &'a Tensor,
    /// Max K/V length baked into the captured kernel launch shape.
    pub max_seqlen_k: usize,
    /// `[batch, rotary_dim/2]` rotary cosine table.
    pub rotary_cos: &'a Tensor,
    /// `[batch, rotary_dim/2]` rotary sine table.
    pub rotary_sin: &'a Tensor,
    /// Per-full-attention-layer paged decode output buffers, shape
    /// `[batch, 1, n_heads, head_dim]`.
    pub attn_out: &'a [Tensor],
    /// Per-full-attention-layer paged decode LSE scratch, shape
    /// `[batch, n_heads, 1]`.
    pub softmax_lse: &'a [Tensor],
    /// `[batch, 1, hidden_size]` stable PRE-final-norm hidden buffer
    /// (#1082 box-102 BUG2 fix port — STEPS 2-3). The
    /// [`model_forward_paged_batched_hidden_with_graph_inputs`] HiddenOnly
    /// forward writes the transformer-stack output here via `slice_set` and
    /// returns WITHOUT running `final_norm` / lm_head. `final_norm` + the
    /// large-N (`vocab = 151936`) cublasLt lm_head GEMV then run EAGERLY off
    /// the captured graph via [`lm_head_from_batched_hidden_eager`],
    /// mirroring the bs=1 `HiddenOnly` contract ([`PagedDecodeGraphInputs`]
    /// has no logits buffer; the bs=1 hidden lives on
    /// [`crate::cuda_graph::CapturedDecodeGraph::output_hidden`]). This is the
    /// structural fix for the lm_head replay-nondeterminism ("BUG2"
    /// token-doubling). STEPS 2-3 dropped the former in-graph `output_logits`
    /// buffer (and the in-graph lm_head twin) — the batched capture/replay
    /// path is now HiddenOnly + eager-lm_head.
    pub output_hidden: &'a mut Tensor,
    /// Persistent batched [`LinearAttentionState`] slot used by the
    /// captured forward. Lifetime is the graph runner's; the captured
    /// graph reads recurrent/conv state from these device pointers.
    pub linear_state: &'a mut LinearAttentionState,
}

/// Try the fused paged-decode flash-attention kernel.
///
/// Returns `Ok(Some(output))` on success and `Ok(None)` when the kernel
/// preconditions cannot be satisfied (forcing the caller to fall back to the
/// materializing slow path).
///
/// ### Preconditions checked here
///   * `block_size` divides `kBlockN` (`FA2_KBLOCK_N` = 64 for the hdim256
///     model — see its doc in generate.rs)
///   * Within each `kBlockN`-wide chunk of the block table, the underlying
///     physical pages are contiguous in the pool. The FA2 splitkv paged kernel
///     reads only one block-table entry per kBlockN chunk and assumes the next
///     `kBlockN / block_size` pages are physically contiguous (see
///     `flash_fwd_kernel.h` lines 587-596 and 770-779). With the #1082 default
///     `block_size = 64` this is one page per chunk → vacuously satisfied.
///
/// ### Output
/// `[batch, 1, num_heads * head_dim]` after o_proj (matches the slow path).
#[allow(clippy::too_many_arguments)]
pub(super) fn try_flash_attn_paged_decode(
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
    use_metal_decode_gemv: bool,
    attn_weights: &GpuFullAttentionWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
    // Phase 7 #1082: kt twin of `paged_cache` for parity-checked
    // accessor reads. When `Some` AND the env gate is on,
    // `paged_cache.block_size()` and `paged_cache.is_fp8()` are
    // re-routed through `try_kt_paged_kv_*` helpers. CUDA-gated
    // since `PagedKvCacheKt` is CUDA-only. `None` on the default
    // path (caller not migrated yet or gate off) keeps every
    // accessor on the candle path unchanged.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Option<Tensor>> {
    // #1082: the real FA2 tile width for hdim256 (was a conservative 128).
    const K_BLOCK_N: usize = crate::generate::FA2_KBLOCK_N;

    #[cfg(feature = "cuda")]
    let block_size = try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache);
    #[cfg(not(feature = "cuda"))]
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

    let kt_pools = paged_cache.pool_tensors(full_attn_layer_idx);
    #[cfg(feature = "cuda")]
    try_kt_paged_kv_pool_tensors_present(kt_pools.is_some(), full_attn_layer_idx, kt_paged_cache);
    let (k_pool, v_pool) = match kt_pools {
        Some(p) => p,
        None => return Ok(None),
    };
    // #26: `pool_tensors` now returns OWNED clones (cheap Arc bumps; the pools
    // live behind an RwLock for live resize). Re-borrow as `&KtTensor` so every
    // downstream consumer below is unchanged — the owned tensors stay alive in
    // this scope and keep the pool storage pinned for the whole decode step.
    let (k_pool, v_pool) = (&k_pool, &v_pool);

    // Common macOS/desktop case: a single sequence receives freshly-allocated
    // blocks, so its whole live KV window is already one contiguous run in the
    // pool. In that case we can bypass the paged gather path entirely and feed
    // the fused prefill kernel a direct `[1, total_seq_len, kv_heads, head_dim]`
    // narrow of the live K/V window.
    // CUDA and Vulkan have native GQA paged-decode kernels. The contiguous
    // branch below is useful for backends with an implemented contiguous decode
    // kernel, but otherwise it can build compact K/V views before reaching the
    // real paged path.
    let use_direct_paged_decode = direct_paged_decode_attention_enabled(backend);
    #[cfg(feature = "cuda")]
    let is_fp8 = try_kt_paged_kv_is_fp8(paged_cache.is_fp8(), kt_paged_cache);
    #[cfg(not(feature = "cuda"))]
    let is_fp8 = paged_cache.is_fp8();
    if !is_fp8
        && !use_direct_paged_decode
        && let Some(start_slot) =
            contiguous_slot_run_start(block_table, block_size, 0, total_seq_len)
    {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = {
            kiln_nvtx::range!(c"kiln/attn/paged_decode_contiguous");
            AttentionBackend::runtime_flash_attn_paged_decode_contiguous(
                backend,
                q,
                k_pool,
                v_pool,
                start_slot,
                total_seq_len,
                softmax_scale,
            )?
        };
        let attn_output = if attn_output.is_some() {
            attn_output
        } else {
            let can_read_head_major =
                AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend)
                    && PagedKvBackend::runtime_supports_paged_kv_head_major_read(backend);
            let fast_head_major = if can_read_head_major {
                kiln_nvtx::range!(c"kiln/kv/head_major_read_decode");
                PagedKvBackend::runtime_paged_kv_head_major_read(
                    backend,
                    k_pool,
                    v_pool,
                    start_slot,
                    total_seq_len,
                )?
            } else {
                None
            };
            if AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend) {
                // Q is already head-major at the call site. Keep K/V grouped
                // instead of routing through `flash_attention_forward`, which
                // expands GQA K/V before Metal SDPA and defeats Candle's
                // native vector-attention GQA path.
                let (k_head, v_head) = match fast_head_major {
                    Some(kv) => kv,
                    None => {
                        let k_live = k_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                        let v_live = v_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                        (
                            k_live.transpose(1, 2)?.contiguous()?,
                            v_live.transpose(1, 2)?.contiguous()?,
                        )
                    }
                };
                flash_attention_forward_head_major(
                    backend, q, &k_head, &v_head, num_heads, head_dim,
                )?
            } else {
                None
            }
        };
        let attn_output = if attn_output.is_some() {
            attn_output
        } else {
            // Reshape Q for the fused-attention APIs only when the
            // head-major path declined. The common Metal desktop path
            // returns above and should not pay this transpose/copy.
            let k_live = k_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
            let v_live = v_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
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
            let attn_output =
                attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate)?;
            let out = gqa_attention_output_projection(
                backend,
                &attn_output,
                attn_weights,
                use_metal_decode_gemv,
                lora_layer.map(|l| (l, lora_scale)),
            )?;
            return Ok(Some(out));
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
    if paged_decode_requires_contiguous_kv_chunks(backend) {
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
    // #1082: pad the tail by REPEATING the last real block id, never `last + 1`.
    // At block_size >= kBlockN (the #1082 default 64) pages_per_chunk = 1, so the
    // FA2 split-KV kernel reads EVERY block_table entry [0..n_block_max-1] as a
    // raw physical page id. An incrementing pad off the last real page can exceed
    // num_blocks (~12174 at block_size=64) — harmless when rebuilt eagerly each
    // step, but FATAL once baked into a captured CUDA graph (CUDA_ERROR_ILLEGAL_
    // ADDRESS on the first replay under concurrency, where blocks.len() <
    // max_blocks_per_seq so padding is actually appended). Repeat-last keeps every
    // entry a valid in-pool page; the padded tail is beyond actual_seqlen_k so it
    // is masked and never semantically read. Matches the box-102 fix in
    // cuda_graph.rs::padded_block_table.
    let pad_block = *padded.last().expect("padded is non-empty (checked above)");
    while padded.len() < max_blocks_per_seq {
        padded.push(pad_block);
    }

    let device = q.device();
    let bt_tensor_owned;
    let bt_tensor = {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            if let Some(inputs) = graph_inputs {
                inputs.block_table
            } else {
                bt_tensor_owned = Tensor::new(padded.as_slice(), device)?
                    .reshape((1usize, max_blocks_per_seq))?;
                &bt_tensor_owned
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        {
            bt_tensor_owned =
                Tensor::new(padded.as_slice(), device)?.reshape((1usize, max_blocks_per_seq))?;
            &bt_tensor_owned
        }
    };

    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Reshape Q for the fused paged-decode APIs: [batch, num_heads, 1, head_dim]
    // -> [batch, 1, num_heads, head_dim]. Build it lazily so the contiguous-KV
    // Metal path above can avoid a dead transpose/copy per full-attention layer.
    let q_fa = {
        kiln_nvtx::range!(c"kiln/attn/q_fa_transpose");
        q.transpose(1, 2)?.contiguous()?
    };
    let attn_out = {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            if let Some(inputs) = graph_inputs {
                let attn_out = inputs.attn_out.get(full_attn_layer_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing graph paged decode output buffer for full-attention layer {full_attn_layer_idx}"
                    )
                })?;
                let softmax_lse = inputs.softmax_lse.get(full_attn_layer_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing graph paged decode LSE buffer for full-attention layer {full_attn_layer_idx}"
                    )
                })?;
                // Phase 7 (#1082): route through the new kt-typed
                // `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_
                // outputs` entry (`aab07fa7`). Bit-exact: bottoms out
                // in the same FFI symbol as the candle wrapper. The
                // kt entry writes through the caller-owned `(attn_out,
                // softmax_lse)` pinned by the captured-graph runner,
                // preserving the dangling-pointer-fix contract from
                // `bench-results/cuda-graph-bs2-secondary-audit.md`
                // suspects 3+4.
                kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs");
                // #1082: every arg is already kt here — `q_fa` (kt param `q`
                // transposed), `k_pool`/`v_pool` (bridged to kt at the top of
                // this fn), and `inputs.*` / `attn_out` / `softmax_lse`
                // (PagedDecodeGraphInputs is kt-typed). Pass directly.
                kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs(
                    &q_fa,
                    k_pool,
                    v_pool,
                    inputs.block_table,
                    inputs.seqused_k,
                    attn_out,
                    softmax_lse,
                    inputs.max_seqlen_k,
                    block_size,
                    softmax_scale,
                    true,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "forward kt: flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs: {e}"
                    )
                })?;
                attn_out.clone()
            } else {
                match AttentionBackend::runtime_flash_attn_paged_decode(
                    backend,
                    &q_fa,
                    k_pool,
                    v_pool,
                    bt_tensor,
                    total_seq_len,
                    block_size,
                    softmax_scale,
                    true,
                )? {
                    Some(t) => t,
                    None => return Ok(None),
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        {
            match AttentionBackend::runtime_flash_attn_paged_decode(
                backend,
                &q_fa,
                k_pool,
                v_pool,
                bt_tensor,
                total_seq_len,
                block_size,
                softmax_scale,
                true,
            )? {
                Some(t) => t,
                None => return Ok(None),
            }
        }
    };

    // attn_out is [batch, 1, num_heads, head_dim] bf16. Reshape to
    // [batch, 1, num_heads * head_dim] for the gate / o_proj path.
    let _ = num_kv_heads; // unused — kept in signature for symmetry / future use
    let attn_output = attn_out.reshape((batch, 1usize, num_heads * head_dim))?;
    let attn_output = attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate)?;
    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t_backend_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    Ok(Some(out))
}

/// Per-decode-step metadata that is identical across every full-attention
/// layer (8× on Qwen3.5-4B). Building the `seqused_k` and padded
/// `block_table` tensors costs one `cudaMemcpyHtoD` each per build, and was
/// being repeated per layer — nsys at bs=16 attributed ~11% of GPU time to
/// these `copy2d_bf16` launches. Hoisted to once-per-step via this struct.
pub struct CachedPagedDecodeMeta {
    /// Padded `[batch, max_blocks_per_seq]` u32 tensor indexing the paged
    /// KV pool. Same for every full-attn layer within a step.
    pub block_table_tensor: Tensor,
    /// Per-row K/V length `[batch]` i32 tensor.
    pub seqused_k_tensor: Tensor,
    /// Max K/V length across rows in the batch (`max(start_pos) + 1`).
    pub max_seqlen_k: usize,
    /// Launch bound supplied to the dynamic-length paged-decode kernel.
    /// Eager metadata uses the exact maximum above. Graph-stable metadata uses
    /// the whole capture bucket so replay can grow row lengths without
    /// retaining the capture step's maximum as a baked kernel argument.
    pub kernel_max_seqlen_k: usize,
    /// Padded block-table width (in pages).
    pub max_blocks_per_seq: usize,
    /// Whether every row's `start_pos` is identical — when true, the strict
    /// uniform-length path is preferred over `dyn_seqlen`.
    pub uniform_start_pos: bool,
    /// When the uniform-length path is reachable, the per-row contiguous
    /// slot start positions (built via `paged_cache.contiguous_slot_run_starts`).
    /// Cached here so the strict fallback skips its own build too.
    pub strict_start_slots: Option<Vec<u32>>,
}

impl CachedPagedDecodeMeta {
    /// Build the shared metadata once for the current decode step. Mirrors
    /// the inline build inside `gqa_attention_paged_decode_contiguous_batch`,
    /// but yields tensors the caller can pass into every full-attn layer.
    ///
    /// `kt_paged_cache`: Phase 7 #1082 — kt twin of `paged_cache` for the
    /// parity-checked `paged_cache.block_size()` read. `None` (default
    /// path) keeps the accessor on the candle path unchanged.
    pub fn build(
        device: &Device,
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
        #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    ) -> Result<Self> {
        let batch = start_positions.len();
        anyhow::ensure!(
            batch > 0,
            "CachedPagedDecodeMeta requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == batch,
            "CachedPagedDecodeMeta metadata length mismatch ({} vs {batch})",
            block_tables.len()
        );

        let max_start_pos = *start_positions
            .iter()
            .max()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let min_start_pos = *start_positions
            .iter()
            .min()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let uniform_start_pos = max_start_pos == min_start_pos;
        let max_seqlen_k = max_start_pos + 1;

        #[cfg(feature = "cuda")]
        let page_block_size = try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache);
        #[cfg(not(feature = "cuda"))]
        let page_block_size = paged_cache.block_size();
        let max_blocks_per_seq = ((max_seqlen_k + page_block_size - 1) / page_block_size).max(1);
        let mut block_table_vec = Vec::<u32>::with_capacity(batch * max_blocks_per_seq);
        // #1082: kt flash-attn requires `seqused_k` U32 (see
        // kiln-flash-attn/src/kt_api.rs); the count is a non-negative
        // sequence length, so u32 is the faithful kt dtype. candle's old
        // i32 storage and the FFI `*const i32` share the bit pattern for
        // these small non-negative values.
        let mut seqused_k_vec = Vec::<u32>::with_capacity(batch);
        for (row_idx, bt) in block_tables.iter().enumerate() {
            let row_seqlen = start_positions[row_idx] + 1;
            seqused_k_vec.push(
                u32::try_from(row_seqlen)
                    .context("CachedPagedDecodeMeta: seqused_k exceeds u32 range")?,
            );
            let row_blocks = bt.blocks.as_slice();
            anyhow::ensure!(
                row_blocks.len() * page_block_size >= row_seqlen,
                "CachedPagedDecodeMeta row {row_idx}: block_table covers {} tokens but row needs {}",
                row_blocks.len() * page_block_size,
                row_seqlen,
            );
            let pad_block = *row_blocks.last().unwrap_or(&0);
            for slot in 0..max_blocks_per_seq {
                let phys = if slot < row_blocks.len() {
                    row_blocks[slot]
                } else {
                    pad_block
                };
                block_table_vec.push(phys);
            }
        }

        let strict_start_slots: Option<Vec<u32>> = if uniform_start_pos {
            let live_window_starts = vec![0usize; batch];
            match paged_cache.contiguous_slot_run_starts(
                block_tables,
                &live_window_starts,
                max_seqlen_k,
            ) {
                Some(slots) => {
                    let v: Result<Vec<u32>> = slots
                        .iter()
                        .map(|&slot| {
                            u32::try_from(slot)
                                .context("CachedPagedDecodeMeta: start slot exceeds u32 range")
                        })
                        .collect();
                    Some(v?)
                }
                None => None,
            }
        } else {
            None
        };

        let block_table_tensor = Tensor::from_vec_on(
            device.clone(),
            block_table_vec,
            vec![batch, max_blocks_per_seq],
        )?
        .contiguous()?;
        let seqused_k_tensor =
            Tensor::from_vec_on(device.clone(), seqused_k_vec, vec![batch])?.contiguous()?;

        Ok(Self {
            block_table_tensor,
            seqused_k_tensor,
            max_seqlen_k,
            kernel_max_seqlen_k: max_seqlen_k,
            max_blocks_per_seq,
            uniform_start_pos,
            strict_start_slots,
        })
    }

    /// Build the per-step paged-decode metadata using caller-owned stable
    /// device buffers for `block_table_tensor` and `seqused_k_tensor` instead
    /// of allocating fresh ones via `Tensor::from_slice`.
    ///
    /// This is the CUDA graph capture path's entry point: during capture the
    /// regular [`Self::build`] would call `Tensor::from_slice` inside the
    /// captured stream window, baking transient `cudaMalloc`-backed device
    /// pointers into the captured kernel arguments. When the per-call
    /// `CachedPagedDecodeMeta` local drops at end of capture, candle
    /// `cudaFree`s the storage and the captured graph is left holding
    /// dangling pointers — the first `cuGraphLaunch` then faults with
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` (see `bench-results/cuda-graph-bs2-memcheck.md`,
    /// issue #1082).
    ///
    /// Callers must pre-populate the stable buffers with the same per-row
    /// data the regular [`Self::build`] would write (the CUDA graph runner
    /// re-fills them before every replay via `cudaMemcpyHtoDAsync`). The
    /// tensors are taken by clone (cheap — they're storage handles, not
    /// data copies) and stored on the returned `CachedPagedDecodeMeta` so
    /// downstream call sites (e.g. `gqa_attention_paged_decode_contiguous_batch`)
    /// can keep their existing `&meta.block_table_tensor` access pattern.
    ///
    /// Shape contract (the caller is responsible for upholding it):
    ///   * `stable_block_table_gpu` must be
    ///     `[batch, (bucket_max_seqlen_k / kBlockN) * (kBlockN / paged_cache.block_size())]` u32
    ///   * `stable_seqused_k_gpu` must be `[batch]` i32
    /// where `bucket_max_seqlen_k = ceil((max(start_positions) + 1) / kBlockN) * kBlockN`
    /// — the K/V chunk-bucketed value, identical to
    /// `CudaBatchedGraphKey::new`'s formula in `cuda_graph.rs`. The bucket
    /// ensures one captured graph can serve every decode step within
    /// the bucket without re-capture. The struct's `max_seqlen_k`
    /// field continues to hold the *actual* `max(start_positions) + 1`
    /// (needed by the strict-path kernel). `kernel_max_seqlen_k` holds the
    /// bucketed value because the dynamic-length kernel's launch bound is
    /// captured by value and must remain valid as replay lengths grow.
    pub fn build_with_stable_buffers(
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
        stable_block_table_gpu: &Tensor,
        stable_seqused_k_gpu: &Tensor,
        // Phase 7 #1082: kt twin of `paged_cache` for the parity-checked
        // `paged_cache.block_size()` read inside this fn. `None` (default
        // path) keeps the accessor on the candle path unchanged. CUDA-only
        // since `PagedKvCacheKt` is `cfg(feature = "cuda")`.
        #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    ) -> Result<Self> {
        let batch = start_positions.len();
        anyhow::ensure!(
            batch > 0,
            "CachedPagedDecodeMeta requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == batch,
            "CachedPagedDecodeMeta metadata length mismatch ({} vs {batch})",
            block_tables.len()
        );

        let max_start_pos = *start_positions
            .iter()
            .max()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let min_start_pos = *start_positions
            .iter()
            .min()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let uniform_start_pos = max_start_pos == min_start_pos;
        let max_seqlen_k = max_start_pos + 1;

        #[cfg(feature = "cuda")]
        let page_block_size = try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache);
        #[cfg(not(feature = "cuda"))]
        let page_block_size = paged_cache.block_size();

        // Actual per-step `max_blocks_per_seq` — what the downstream
        // strict path needs (`flash_attn_paged_decode_contiguous_batch`
        // reads exactly `max_seqlen_k` tokens from each row's
        // contiguous slot run; a bucketed value would over-read past
        // actual seqlen into garbage slots). Used for the struct's
        // public field consumed by both the strict and dyn_seqlen
        // dispatches below.
        let max_blocks_per_seq = ((max_seqlen_k + page_block_size - 1) / page_block_size).max(1);

        // The stable buffer is sized via the same bucketed formula as the
        // graph keys (`CudaBatchedGraphKey` / MetalGraphKey):
        //   `bucket_max_seqlen_k = ceil(max_seqlen_k/kBlockN) * kBlockN`
        //   `stable_max_blocks_per_seq = (bucket_max_seqlen_k / kBlockN)
        //                                * (kBlockN / page_block_size)`
        // Prior to this commit the assert below used `max_blocks_per_seq`
        // (exact ceil), which mismatched the bucketed stable buffer
        // shape on every step except the final one of each kBlockN-token
        // bucket. The `anyhow::ensure!` failure was silently swallowed
        // by `cuda_graph.rs:1617` (`.context("batched forward failed
        // during graph capture")`), causing every concurrent decode
        // request ≥2 to fall back to eager and corrupt the CUDA
        // context. Surfaced by the diagnostic bench after `909e2e61`
        // added `{e:#}` formatting to the swallowed warn line.
        //
        // Fix: compute the bucketed shape separately and use *that*
        // for the buffer assert. Don't propagate the bucketed value
        // into the struct field — the strict-path kernel needs the
        // actual exact value (see comment on `max_blocks_per_seq`
        // above + the `flash_attn_paged_decode_contiguous_batch` use
        // at `forward.rs:~18265`).
        let stable_kblock_n = crate::generate::FA2_KBLOCK_N;
        let stable_bucket_max_seqlen_k = max_seqlen_k.div_ceil(stable_kblock_n) * stable_kblock_n;
        let stable_pages_per_chunk = stable_kblock_n / page_block_size;
        let stable_max_blocks_per_seq =
            ((stable_bucket_max_seqlen_k / stable_kblock_n) * stable_pages_per_chunk).max(1);

        // Verify per-row block-table coverage matches the regular build path
        // even though we don't materialize the device tensors here. This
        // catches inconsistent block_tables / start_positions before they
        // reach the captured kernel.
        for (row_idx, bt) in block_tables.iter().enumerate() {
            let row_seqlen = start_positions[row_idx] + 1;
            let row_blocks = bt.blocks.as_slice();
            anyhow::ensure!(
                row_blocks.len() * page_block_size >= row_seqlen,
                "CachedPagedDecodeMeta row {row_idx}: block_table covers {} tokens but row needs {}",
                row_blocks.len() * page_block_size,
                row_seqlen,
            );
        }

        // Validate the stable buffer shapes match what the captured kernels
        // expect. The stable buffer is sized to the BUCKETED width (the
        // captured graph key is bucketed so one captured graph can
        // serve every step within the kBlockN-token bucket without
        // re-capture). A shape mismatch here is what was rejecting
        // every bs≥2 step prior to the bucketed-formula fix above.
        let block_table_dims = stable_block_table_gpu.dims();
        anyhow::ensure!(
            block_table_dims == [batch, stable_max_blocks_per_seq],
            "CachedPagedDecodeMeta stable block_table shape mismatch: got {:?}, expected [{batch}, {stable_max_blocks_per_seq}] (bucketed from actual max_seqlen_k={max_seqlen_k}, max_blocks_per_seq={max_blocks_per_seq})",
            block_table_dims,
        );
        let seqused_k_dims = stable_seqused_k_gpu.dims();
        anyhow::ensure!(
            seqused_k_dims == [batch],
            "CachedPagedDecodeMeta stable seqused_k shape mismatch: got {:?}, expected [{batch}]",
            seqused_k_dims,
        );

        let strict_start_slots: Option<Vec<u32>> = if uniform_start_pos {
            let live_window_starts = vec![0usize; batch];
            // Use the *actual* sequence length here — `contiguous_slot_run_starts`
            // walks the cache to verify the slots are filled up to `len`. A
            // bucketed value would ask about slots beyond the current decode
            // position that haven't been written yet.
            match paged_cache.contiguous_slot_run_starts(
                block_tables,
                &live_window_starts,
                max_seqlen_k,
            ) {
                Some(slots) => {
                    let v: Result<Vec<u32>> = slots
                        .iter()
                        .map(|&slot| {
                            u32::try_from(slot)
                                .context("CachedPagedDecodeMeta: start slot exceeds u32 range")
                        })
                        .collect();
                    Some(v?)
                }
                None => None,
            }
        } else {
            None
        };

        Ok(Self {
            block_table_tensor: stable_block_table_gpu.clone(),
            seqused_k_tensor: stable_seqused_k_gpu.clone(),
            // The strict path needs the actual value. The dynamic-length path
            // must launch for the entire graph bucket: this scalar becomes a
            // captured kernel argument, while `seqused_k_tensor` masks each
            // row to its current active prefix.
            max_seqlen_k,
            kernel_max_seqlen_k: stable_bucket_max_seqlen_k,
            max_blocks_per_seq,
            uniform_start_pos,
            strict_start_slots,
        })
    }
}

/// Batched full-attention decode for rows whose live paged-KV windows can be
/// addressed through a block table. Uniform contiguous rows use the strict
/// faster path when available; divergent row lengths use the backend's
/// dyn-seqlen path.
///
/// This is the scheduler-facing low-level primitive for true decode batching:
/// it projects Q/K/V for `[batch, 1, hidden]`, writes one K/V row per request
/// into the shared paged cache, runs the batched contiguous paged-attention
/// backend kernel, then applies the attention output gate and `o_proj`.
///
/// Current backend constraints are intentionally narrow:
/// - one decode token per row,
/// - non-FP8 paged cache,
/// - either each row's live `0..start_pos+1` KV window is one contiguous pool
///   run with a uniform length, or the backend accepts
///   `flash_attn_paged_decode_contiguous_batch_dyn_seqlen`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_positions: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    cached_meta: Option<&CachedPagedDecodeMeta>,
    // CUDA-graph-stable paged-decode scratch tensors for the dyn_seqlen
    // backend path. When `Some((attn_out, softmax_lse))`, the captured
    // kernel writes/reads through caller-owned buffers re-used across
    // graph replays instead of allocating fresh `Tensor::zeros` inside
    // the captured region (#1082, see
    // `bench-results/cuda-graph-bs2-secondary-audit.md` suspects 3+4).
    // `None` reproduces the legacy per-call allocation behavior.
    graph_outputs: Option<(&Tensor, &Tensor)>,
    // CUDA-graph-stable RoPE cos/sin tables for the bs>1 captured
    // forward. When `Some((cos, sin))` (shape `[batch, rotary_dim/2]`,
    // typically `graph_inputs.rotary_cos`/`.rotary_sin`), the RoPE
    // step consumes the caller-owned tables via
    // `rotary_embedding_from_tables` instead of calling
    // `rotary_embedding_from_tensor`, which builds fresh
    // `cudaMalloc`-backed `freqs/cos/sin` tensors inside the captured
    // region (#1082 suspect 2 — see
    // `bench-results/cuda-graph-bs2-secondary-audit.md`). `None`
    // reproduces the legacy positions-based per-call build.
    rope_tables: Option<(&Tensor, &Tensor)>,
    // Graph-stable `[batch]` u32 per-row KV-write slot tensor. Metal dispatches
    // the fused batched kernel
    // (`PagedKvCache::write_token_major_native_batch_graph_slot`) instead
    // of the per-row host loop that calls
    // `paged_kv_write_token_major_bf16` with a baked-immediate slot.
    // The baked-immediate form is a CUDA-graph replay-correctness bug:
    // the captured kernel records the capture-time slot index as a
    // launch immediate, so replays at a different decode position
    // write into the wrong KV-cache slot. The fused-slot kernel reads
    // its destination slots from this device tensor on every replay
    // (refreshed via `update_cuda_scalar` outside the captured region),
    // closing suspect 1 in `bench-results/cuda-graph-bs2-secondary-audit.md`
    // for #1082. `None` reproduces the legacy per-row writer.
    kv_slot: Option<&Tensor>,
    #[cfg(feature = "metal")] mut metal_icb_layer: Option<MetalPagedDecodeIcbLayer<'_>>,
    // Phase 7 #1082: kt twin of `paged_cache` for parity-checked
    // accessor reads. When `Some` AND the env gate is on, accessor
    // calls (`is_fp8`, `num_layers`) are mirrored through
    // `try_kt_paged_kv_*` helpers. `None` keeps every accessor on
    // the candle path unchanged.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    anyhow::ensure!(batch > 0, "batched paged decode requires a non-empty batch");
    anyhow::ensure!(
        seq_len == 1,
        "batched contiguous paged attention requires one decode token per row"
    );
    anyhow::ensure!(
        block_tables.len() == batch && start_positions.len() == batch,
        "batched contiguous paged attention metadata length mismatch"
    );
    anyhow::ensure!(
        positions.elem_count() == 1 || positions.elem_count() == batch,
        "batched contiguous paged attention positions tensor must hold either a shared scalar or one entry per row"
    );
    #[cfg(feature = "cuda")]
    {
        anyhow::ensure!(
            !try_kt_paged_kv_is_fp8(paged_cache.is_fp8(), kt_paged_cache),
            "batched contiguous paged attention does not support FP8 caches"
        );
        anyhow::ensure!(
            full_attn_layer_idx
                < try_kt_paged_kv_num_layers(paged_cache.num_layers(), kt_paged_cache),
            "batched contiguous paged attention layer index out of range"
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        anyhow::ensure!(
            !paged_cache.is_fp8(),
            "batched contiguous paged attention does not support FP8 caches"
        );
        anyhow::ensure!(
            full_attn_layer_idx < paged_cache.num_layers(),
            "batched contiguous paged attention layer index out of range"
        );
    }

    // Phase 12-B-prime: drop the uniform-start_pos assertion stack. Per-row
    // K/V lengths are encoded via `seqused_k`, and per-row start positions
    // (used by RoPE + paged-KV slot indexing) are passed through as-is.
    //
    // When `cached_meta` is provided (set by the top-level batched decode
    // entry point), we skip the per-layer rebuild of the seqused_k /
    // block_table tensors. The cache is invariant across the 8 full-attn
    // layers within a step, so building once saves 7 HtoD launches per
    // step at bs > 1.
    let (
        max_seqlen_k,
        kernel_max_seqlen_k,
        uniform_start_pos,
        max_blocks_per_seq,
        own_block_table_tensor,
        own_seqused_k_tensor,
        own_strict_start_slots,
    ): (
        usize,
        usize,
        bool,
        usize,
        Option<Tensor>,
        Option<Tensor>,
        Option<Vec<u32>>,
    ) = match cached_meta {
        Some(meta) => (
            meta.max_seqlen_k,
            meta.kernel_max_seqlen_k,
            meta.uniform_start_pos,
            meta.max_blocks_per_seq,
            None,
            None,
            None,
        ),
        None => {
            let max_start_pos = *start_positions
                .iter()
                .max()
                .context("batched paged decode requires non-empty start_positions")?;
            let min_start_pos = *start_positions
                .iter()
                .min()
                .context("batched paged decode requires non-empty start_positions")?;
            let uniform_start_pos = max_start_pos == min_start_pos;
            let max_seqlen_k = max_start_pos + 1;

            // Build varlen metadata: per-row seqused_k tensor and a padded
            // [batch, max_blocks_per_seq] block_table tensor that indexes the
            // paged KV pool. `flash_attn_paged_decode_dyn_seqlen` masks padding
            // beyond each row's seqused_k.
            #[cfg(feature = "cuda")]
            let page_block_size =
                try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache);
            #[cfg(not(feature = "cuda"))]
            let page_block_size = paged_cache.block_size();
            let max_blocks_per_seq =
                ((max_seqlen_k + page_block_size - 1) / page_block_size).max(1);
            let mut block_table_vec = Vec::<u32>::with_capacity(batch * max_blocks_per_seq);
            // #1082: kt flash-attn requires `seqused_k` U32 (kt_api.rs); the
            // count is a non-negative sequence length, faithful as u32.
            let mut seqused_k_vec = Vec::<u32>::with_capacity(batch);
            for (row_idx, bt) in block_tables.iter().enumerate() {
                let row_seqlen = start_positions[row_idx] + 1;
                seqused_k_vec.push(
                    u32::try_from(row_seqlen).context(
                        "batched contiguous paged attention seqused_k exceeds u32 range",
                    )?,
                );
                let row_blocks = bt.blocks.as_slice();
                anyhow::ensure!(
                    row_blocks.len() * page_block_size >= row_seqlen,
                    "batched contiguous paged attention row {row_idx}: block_table covers {} tokens but row needs {}",
                    row_blocks.len() * page_block_size,
                    row_seqlen,
                );
                let pad_block = *row_blocks.last().unwrap_or(&0);
                for slot in 0..max_blocks_per_seq {
                    let phys = if slot < row_blocks.len() {
                        row_blocks[slot]
                    } else {
                        pad_block
                    };
                    block_table_vec.push(phys);
                }
            }

            // Strict-path slot_run vector kept as a fallback for when the
            // dyn_seqlen backend declines (e.g. kill switch armed). Only valid
            // when the live window is uniform across rows.
            let strict_start_slots: Option<Vec<u32>> = if uniform_start_pos {
                let live_window_starts = vec![0usize; batch];
                match paged_cache.contiguous_slot_run_starts(
                    block_tables,
                    &live_window_starts,
                    max_seqlen_k,
                ) {
                    Some(slots) => {
                        let v: Result<Vec<u32>> = slots
                            .iter()
                            .map(|&slot| {
                                u32::try_from(slot).context(
                                    "batched contiguous paged attention start slot exceeds u32 range",
                                )
                            })
                            .collect();
                        Some(v?)
                    }
                    None => None,
                }
            } else {
                None
            };

            let block_table_tensor =
                Tensor::from_vec_on(x.device(), block_table_vec, vec![batch, max_blocks_per_seq])?
                    .contiguous()?;
            let seqused_k_tensor =
                Tensor::from_vec_on(x.device(), seqused_k_vec, vec![batch])?.contiguous()?;

            (
                max_seqlen_k,
                max_seqlen_k,
                uniform_start_pos,
                max_blocks_per_seq,
                Some(block_table_tensor),
                Some(seqused_k_tensor),
                strict_start_slots,
            )
        }
    };
    let _ = max_blocks_per_seq; // shape already baked into the cached tensor

    let use_metal_decode_gemv = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_runtime::single_token_self_attention_active();

    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv_batch_decode");
        full_attn_qkv_proj_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer,
            lora_scale,
        )
        .context("gqa paged qkv projection")?
    };

    let (q, gate) = {
        if attn_output_gate {
            let q_raw = reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim * 2)?;
            let q = q_raw.narrow(3, 0, head_dim)?;
            let gate = q_raw.narrow(3, head_dim, head_dim)?;
            let gate = reshape_hole0_3(&gate.contiguous()?, seq_len, num_heads * head_dim)?;
            (q.contiguous()?, Some(gate))
        } else {
            (reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim)?, None)
        }
    };
    let k = reshape_hole0_4(&k, seq_len, num_kv_heads, head_dim)?;
    let v = reshape_hole0_4(&v, seq_len, num_kv_heads, head_dim)?;

    let (q, k) = {
        let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
        let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
        (q, k)
    };
    let (q, k) = {
        if let Some((cos, sin)) = rope_tables {
            // CUDA-graph capture path: the runner pre-allocates
            // `[batch, rotary_dim/2]` cos/sin tables and re-fills them via
            // `update_batched_rotary_buffers` before every replay, so the
            // captured RoPE kernel reads from a stable device pointer.
            // The table shape matches the per-row layout (one (cos, sin)
            // row per start_position, even when positions are uniform),
            // so we swap batch <-> seq_len so `apply_rope` /
            // `fused_rotary_qk` broadcast `[batch, half]` against
            // `[1, batch, num_heads, half]`. This bypasses the
            // `rotary_embedding_from_tensor` inner allocation of
            // freshly-allocated `freqs/cos/sin` tensors inside the
            // captured region — the bug documented as suspect 2 in
            // `bench-results/cuda-graph-bs2-secondary-audit.md` (#1082).
            let q_swap = q.transpose(0, 1)?.contiguous()?;
            let k_swap = k.transpose(0, 1)?.contiguous()?;
            let (q_rot, k_rot) =
                rotary_embedding_from_tables(&q_swap, &k_swap, cos, sin, head_dim, rotary_dim)?;
            (
                q_rot.transpose(0, 1)?.contiguous()?,
                k_rot.transpose(0, 1)?.contiguous()?,
            )
        } else if positions.elem_count() == 1 {
            // Shared scalar position: reuse the existing seq_len-major rope
            // path. cos/sin shape [1, half_rotary] broadcasts cleanly across
            // [batch, 1, num_heads, half_rotary].
            rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, inv_freq)?
        } else {
            // Per-row positions: swap batch <-> seq_len so cos/sin built from
            // [batch, half_rotary] aligns with the second dim of the q/k
            // tensors expected by `apply_rope`. After RoPE swap back.
            let q_swap = q.transpose(0, 1)?.contiguous()?;
            let k_swap = k.transpose(0, 1)?.contiguous()?;
            let (q_rot, k_rot) = rotary_embedding_from_tensor(
                &q_swap, &k_swap, positions, head_dim, rotary_dim, inv_freq,
            )?;
            (
                q_rot.transpose(0, 1)?.contiguous()?,
                k_rot.transpose(0, 1)?.contiguous()?,
            )
        }
    };
    // Q stays in [batch, 1, num_heads, head_dim] for the dyn_seqlen path; the
    // strict fallback below transposes lazily into the head-major layout it
    // requires.

    // #1082: `PagedKvCacheKt::pool_tensors` returns kt pool references already;
    // the candle borrow/bridge dance is gone — bind the kt pools directly.
    let (k_pool, v_pool) = paged_cache
        .pool_tensors(full_attn_layer_idx)
        .context("batched contiguous paged attention layer index out of range")?;
    // #26: pool_tensors returns owned clones now — re-borrow so downstream is
    // unchanged (owned tensors stay alive for this scope, pinning the storage).
    let (k_pool, v_pool) = (&k_pool, &v_pool);
    #[cfg(feature = "cuda")]
    let page_block_size = try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache);
    #[cfg(not(feature = "cuda"))]
    let page_block_size = paged_cache.block_size();
    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Prefer the once-per-step cached tensors when the caller built them;
    // otherwise use the per-layer ones we built above.
    let block_table_tensor: &Tensor = match (cached_meta, own_block_table_tensor.as_ref()) {
        (Some(meta), _) => &meta.block_table_tensor,
        (None, Some(t)) => t,
        (None, None) => unreachable!("cached_meta=None branch must build the block_table tensor"),
    };
    let seqused_k_tensor: &Tensor = match (cached_meta, own_seqused_k_tensor.as_ref()) {
        (Some(meta), _) => &meta.seqused_k_tensor,
        (None, Some(t)) => t,
        (None, None) => unreachable!("cached_meta=None branch must build the seqused_k tensor"),
    };

    #[cfg(feature = "metal")]
    let mut metal_icb_attn_output: Option<Tensor> = None;
    #[cfg(feature = "metal")]
    if let Some(layer) = metal_icb_layer.as_mut() {
        match try_metal_paged_decode_icb_attention(
            &q,
            &k,
            &v,
            layer,
            graph_outputs,
            kv_slot,
            k_pool,
            v_pool,
            block_table_tensor,
            seqused_k_tensor,
            block_tables,
            start_positions,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
        ) {
            Ok(Some(out)) => metal_icb_attn_output = Some(out),
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(
                    layer = full_attn_layer_idx,
                    error = %err,
                    "Metal paged decode ICB attention failed; using eager attention path"
                );
                if let Some(layer) = metal_icb_layer.as_mut() {
                    *layer.graph = None;
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    let run_eager_kv_and_attention = metal_icb_attn_output.is_none();
    #[cfg(not(feature = "metal"))]
    let run_eager_kv_and_attention = true;
    if run_eager_kv_and_attention {
        let kv_write_done = {
            #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
            {
                // #1082 suspect 1: when the runner has wired the
                // `[batch] u32` per-row slot device buffer through
                // `BatchedPagedDecodeGraphInputs.kv_slot`/Metal graph inputs,
                // write via the fused batched-slot kernel so the captured graph
                // re-reads fresh slots on every replay. The unqualified CUDA
                // batched graph route remains unavailable; Metal uses the
                // slot-buffer writer whenever the runner supplies it.
                if let Some(slot_tensor) = kv_slot {
                    let use_graph_slot_writer = match k.device() {
                        #[cfg(feature = "cuda")]
                        Device::Cuda(_) => false,
                        #[cfg(feature = "metal")]
                        Device::Metal(_) => true,
                        #[cfg(feature = "rocm")]
                        Device::Rocm(_) => true,
                        _ => false,
                    };
                    if use_graph_slot_writer {
                        // #1082: `PagedKvCacheKt::write_token_major_native_batch_graph_slot`
                        // takes kt tensors; `k`/`v`/`slot_tensor` are already kt, so
                        // pass them through with no candle bridge.
                        paged_cache.write_token_major_native_batch_graph_slot(
                            full_attn_layer_idx,
                            &k,
                            &v,
                            slot_tensor,
                        )?
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "rocm")))]
            {
                false
            }
        };
        if !kv_write_done {
            // #1082: `PagedKvCacheKt::write_token_major_native_batch` takes kt
            // tensors; `k`/`v` are already kt, so pass them with no candle bridge.
            // CUDA uses its host-slot writer; Metal uses its batched token-major
            // kernel; unsupported native BF16 placements fall back inside the
            // cache writer to the generic kt head-major scatter.
            if !paged_cache.write_token_major_native_batch(
                full_attn_layer_idx,
                block_tables,
                start_positions,
                &k,
                &v,
            )? {
                anyhow::bail!("batched contiguous paged attention KV write declined");
            }
        }
    }

    let strict_start_slots: Option<&[u32]> = match (cached_meta, own_strict_start_slots.as_ref()) {
        (Some(meta), _) => meta.strict_start_slots.as_deref(),
        (None, Some(v)) => Some(v.as_slice()),
        (None, None) => None,
    };
    let run_eager_paged_decode_attention = || -> Result<Tensor> {
        // Phase 12-B-prime perf gate: dyn_seqlen handles divergent per-row
        // start_pos correctly but regressed synthetic c=8 throughput by ~61%
        // versus the post-#996 strict-path baseline under uniform load (which
        // is the common synthetic + most-production case). Route through the
        // strict head-major path when start_pos is uniform across the batch
        // (the pre-12-B-prime working path) and only fall through to
        // dyn_seqlen when rows actually diverge.
        //
        // ROCm and CUDA use immutable profile policy. Other backends retain
        // automatic strict-versus-dynamic selection from request geometry.
        let rocm_policy = BackendIdentity::runtime_name(backend) == "rocm";
        let kill_dyn_seqlen = if rocm_policy {
            !crate::rocm_policy::current_rocm_kernel_policy().paged_decode_dyn_seqlen_batch
        } else if BackendIdentity::runtime_name(backend) == "cuda" {
            !crate::cuda_policy::current_cuda_kernel_policy().paged_decode_dyn_seqlen_batch
        } else {
            false
        };
        let force_dyn_seqlen = false;
        // Short-circuit the strict probe on backends that have no
        // strict_paged_decode_contiguous_batch kernel impl. The probe
        // would `Tensor::from_slice` build a `[batch] u32 start_slots`
        // tensor before calling the backend, and on CUDA the backend
        // declines via the trait default — but under CUDA graph
        // capture, the `Tensor::from_slice`'s `cudaMemcpyHtoDAsync`
        // is captured by the stream and on replay writes to a
        // recycled VA (suspect 6 in
        // `bench-results/cuda-graph-bs2-secondary-audit.md`, #1082).
        // The strict kernel exists on Metal so the predicate defaults
        // `true` and Metal paths keep their preferred-strict
        // dispatch.
        let backend_supports_strict =
            AttentionBackend::runtime_supports_strict_paged_decode_contiguous_batch(backend);
        let prefer_strict = !force_dyn_seqlen
            && uniform_start_pos
            && strict_start_slots.is_some()
            && backend_supports_strict;

        let try_strict = |out_acc: &mut Option<Tensor>| -> Result<()> {
            // Strict contiguous-batch path: pre-12-B-prime code path that
            // delivered PR #996's +10.76% c=8 throughput win. Requires
            // uniform start_pos + contiguous live KV. The strict kernel
            // expects head-major [batch, num_heads, 1, head_dim].
            let strict_slots = strict_start_slots.context(
                "batched contiguous paged attention requires uniform start_pos for the strict path",
            )?;
            let start_slots = Tensor::from_vec_on(x.device(), strict_slots.to_vec(), vec![batch])?
                .contiguous()?;
            let q_strict = { q.transpose(1, 2)?.contiguous()? };
            *out_acc = AttentionBackend::runtime_flash_attn_paged_decode_contiguous_batch(
                backend,
                &q_strict,
                k_pool,
                v_pool,
                &start_slots,
                max_seqlen_k,
                softmax_scale,
            )?;
            Ok(())
        };

        let try_dyn_seqlen = |out_acc: &mut Option<Tensor>| -> Result<()> {
            // When the caller threaded in graph-stable `(attn_out,
            // softmax_lse)` scratch tensors, use the variant that
            // consumes them so the captured kernel writes to a
            // runner-owned destination across replays (#1082 suspects
            // 3+4). Otherwise the kernel wrapper allocates fresh
            // `Tensor::zeros` inside the captured region, which the
            // captured graph then dangles when the tensors drop.
            *out_acc =
                ReplayBackend::runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
                    backend,
                    &q,
                    k_pool,
                    v_pool,
                    block_table_tensor,
                    seqused_k_tensor,
                    graph_outputs,
                    kernel_max_seqlen_k,
                    page_block_size,
                    softmax_scale,
                    true,
                )?;
            Ok(())
        };

        let mut out: Option<Tensor> = None;
        if kill_dyn_seqlen {
            // Strict-only; non-uniform batches will surface the strict_slots
            // context error.
            try_strict(&mut out)?;
        } else if prefer_strict {
            try_strict(&mut out)?;
            if out.is_none() {
                // Strict backend declined (e.g. Metal CPU fallback path).
                // Fall through to dyn_seqlen.
                try_dyn_seqlen(&mut out)?;
            }
        } else {
            try_dyn_seqlen(&mut out)?;
            if out.is_none()
                && uniform_start_pos
                && strict_start_slots.is_some()
                && backend_supports_strict
            {
                // dyn_seqlen backend declined; the strict path can still
                // serve uniform batches. Divergent batches have no fallback
                // and will surface as the final context error below. CUDA
                // skips this fallback because its strict trait method has
                // no impl (declines via the default `Ok(None)`) and the
                // intermediate `Tensor::from_slice` would emit a captured
                // HtoD to a recycled VA (#1082 suspect 6).
                try_strict(&mut out)?;
            }
        }
        out.context("backend declined batched contiguous paged attention")
    };
    let attn_output = {
        #[cfg(feature = "metal")]
        {
            if let Some(out) = metal_icb_attn_output {
                out
            } else {
                run_eager_paged_decode_attention()?
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            run_eager_paged_decode_attention()?
        }
    };

    // Both kernels feed o_proj a row-major [batch, 1, num_heads * head_dim].
    // The Metal strict kernel already returns that 3-D shape; the dyn_seqlen
    // kernel returns 4-D [batch, 1, num_heads, head_dim], so flatten the trailing
    // axes here. The reshape is a no-op for the 3-D case.
    let attn_output = if attn_output.dims().len() == 4 {
        attn_output.reshape((batch, seq_len, num_heads * head_dim))?
    } else {
        attn_output
    };

    let attn_output =
        { attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())? };
    let out = {
        kiln_nvtx::range!(c"kiln/proj/o_batch_decode");
        gqa_attention_output_projection(
            backend,
            &attn_output,
            attn_weights,
            use_metal_decode_gemv,
            lora_layer.map(|l| (l, lora_scale)),
        )?
    };
    Ok(out)
}

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_metal_paged_decode_icb_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    layer: &mut MetalPagedDecodeIcbLayer<'_>,
    graph_outputs: Option<(&Tensor, &Tensor)>,
    kv_slot: Option<&Tensor>,
    k_pool: &Tensor,
    v_pool: &Tensor,
    block_table_tensor: &Tensor,
    seqused_k_tensor: &Tensor,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<Option<Tensor>> {
    let Ok((batch, q_len, q_heads, head_dim)) = q.dims4() else {
        return Ok(None);
    };
    if batch == 0 || q_len != 1 || q_heads != 16 || head_dim != 256 {
        return Ok(None);
    }
    if block_tables.len() != batch || start_positions.len() != batch {
        return Ok(None);
    }
    let Some((attn_out, _softmax_lse)) = graph_outputs else {
        return Ok(None);
    };
    let Some(slots) = kv_slot else {
        return Ok(None);
    };
    if !matches!(q.device(), Device::Metal(_))
        || !matches!(k.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
        || !matches!(attn_out.device(), Device::Metal(_))
        || !matches!(slots.device(), Device::Metal(_))
    {
        return Ok(None);
    }

    kiln_tensor::metal_copy_in_place(q, layer.q).context("refresh Metal ICB stable Q")?;
    kiln_tensor::metal_copy_in_place(k, layer.k).context("refresh Metal ICB stable K")?;
    kiln_tensor::metal_copy_in_place(v, layer.v).context("refresh Metal ICB stable V")?;

    if layer.graph.is_none() {
        *layer.graph = Some(
            crate::backend::metal::metal_record_paged_decode_icb_graph(
                layer.q,
                k_pool,
                v_pool,
                block_table_tensor,
                seqused_k_tensor,
                attn_out,
                layer.k,
                layer.v,
                slots,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
            )
            .context("record Metal paged decode ICB graph")?,
        );
    }
    let graph = layer
        .graph
        .as_ref()
        .context("Metal ICB graph missing after record")?;
    crate::metal_graph::replay_paged_decode_icb_graph_through_replay_plan(
        graph,
        max_seqlen_k as u32,
        softmax_scale,
    )
    .context("replay Metal paged decode ICB graph")?;
    Ok(Some(attn_out.clone()))
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
    paged_cache: &PagedKvCache,
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
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None,
        // Phase 7 #1082: no kt twin plumbed through this wrapper yet
        // — the cache-owning struct migration that allocates one via
        // `try_kt_paged_kv_cache_new` is a follow-up commit. Default
        // `None` keeps this path on the candle writer only.
        #[cfg(feature = "cuda")]
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_attention_paged_with_rope_tables(
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
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    #[cfg(any(feature = "cuda", feature = "rocm"))] graph_inputs: Option<
        &PagedDecodeGraphInputs<'_>,
    >,
    // Phase 7 #1082: kt twin of `paged_cache` used to mirror the
    // CUDA-graph paged-KV write to the kt cache when
    // `accelerator.kt_api_mode = "all"`. `None` means "policy disabled,
    // non-CUDA device, or caller hasn't been migrated yet" — the
    // candle writer below runs unchanged in that case. CUDA-gated
    // since `PagedKvCacheKt` itself is CUDA-only.
    #[cfg(feature = "cuda")] kt_paged_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv =
        seq_len == 1 && start_pos > 0 && !crate::mtp_runtime::single_token_self_attention_active();

    // Project to Q, K, V (with optional LoRA delta and output gate split)
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k_raw, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        full_attn_qkv_proj_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer,
            lora_scale,
        )?
    };
    let fused_qkv_prep: Option<(Tensor, Tensor, Option<Tensor>)> = {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            if seq_len == 1
                && !gpu_fused_attn_decode_qkv_prep_disabled(&x.device())
                && !any_kt_tensor_tracks_op(&[
                    &q_raw,
                    &k_raw,
                    &attn_weights.q_norm,
                    &attn_weights.k_norm,
                ])
            {
                if let Some((cos, sin)) = rope_tables {
                    if let (
                        Some(q_raw_kt),
                        Some(k_raw_kt),
                        Some(q_norm_kt),
                        Some(k_norm_kt),
                        Some(cos_kt),
                        Some(sin_kt),
                    ) = (
                        try_borrow_kt_cuda(&q_raw),
                        try_borrow_kt_cuda(&k_raw),
                        try_borrow_kt_cuda(&attn_weights.q_norm),
                        try_borrow_kt_cuda(&attn_weights.k_norm),
                        try_borrow_kt_cuda(cos),
                        try_borrow_kt_cuda(sin),
                    ) {
                        if kiln_rmsnorm_kernel::supports_attn_decode_qkv_prep_kt(
                            &q_raw_kt,
                            &k_raw_kt,
                            &q_norm_kt,
                            &k_norm_kt,
                            &cos_kt,
                            &sin_kt,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            rotary_dim,
                            attn_output_gate,
                        ) {
                            kiln_nvtx::range!(c"kiln/attn/qkv_prep_gpu_fused");
                            let (q_kt, k_kt, gate_kt) =
                                kiln_rmsnorm_kernel::attn_decode_qkv_split_qk_norm_rope_kt(
                                    &q_raw_kt,
                                    &k_raw_kt,
                                    &q_norm_kt,
                                    &k_norm_kt,
                                    &cos_kt,
                                    &sin_kt,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    rotary_dim,
                                    attn_output_gate,
                                    rms_norm_eps as f32,
                                )
                                .map_err(|e| anyhow::anyhow!("kt attn_decode_qkv_prep: {e}"))?;
                            // #1082: keep the fused qkv-prep outputs as kt — the
                            // `fused_qkv_prep` consumer and the else-branch below are
                            // kt, so the candle copy-out is gone.
                            let q = q_kt;
                            let k = k_kt;
                            let gate = gate_kt;
                            Some((q, k, gate))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        {
            None
        }
    };

    let (q, k, gate) = if let Some((q, k, gate)) = fused_qkv_prep {
        (q, k, gate)
    } else {
        let (q, gate) = {
            kiln_nvtx::range!(c"kiln/proj/qkv_split");
            if attn_output_gate {
                let q_raw = reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim * 2)
                    .context("gqa paged q split reshape")?;
                let q = q_raw
                    .narrow(3, 0, head_dim)
                    .context("gqa paged q split value narrow")?;
                let gate = q_raw
                    .narrow(3, head_dim, head_dim)
                    .context("gqa paged q split gate narrow")?;
                let gate = reshape_hole0_3(
                    &gate
                        .contiguous()
                        .context("gqa paged q split gate contiguous")?,
                    seq_len,
                    num_heads * head_dim,
                )
                .context("gqa paged q split gate reshape")?;
                (
                    q.contiguous()
                        .context("gqa paged q split value contiguous")?,
                    Some(gate),
                )
            } else {
                let q = reshape_hole0_4(&q_raw, seq_len, num_heads, head_dim)
                    .context("gqa paged q reshape")?;
                (q, None)
            }
        };
        let k = reshape_hole0_4(&k_raw, seq_len, num_kv_heads, head_dim)
            .context("gqa paged k reshape")?;

        // QK-norm
        let (q, k) = {
            kiln_nvtx::range!(c"kiln/attn/qk_norm");
            let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps).context("gqa paged q norm")?;
            let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps).context("gqa paged k norm")?;
            (q, k)
        };
        // Phase B9 H2 aliases: post_qk_norm_{q,k} mirror post_{q,k}_norm.
        // Phase B12 layer-31 GQA taps: qk_norm_q / qk_norm_k. Post per-head
        // RMSNorm, pre-RoPE. Shape [B, T, num_heads, head_dim] /
        // [B, T, num_kv_heads, head_dim].

        // RoPE — only rotate first rotary_dim dimensions
        // Use the GPU tensor variant so positions remain at a stable GPU address
        // (critical for CUDA graph replay correctness)
        let (q, k) = {
            kiln_nvtx::range!(c"kiln/attn/rope");
            if let Some((cos, sin)) = rope_tables {
                rotary_embedding_from_tables(&q, &k, cos, sin, head_dim, rotary_dim)
                    .context("gqa paged rope tables")?
            } else {
                rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, inv_freq)
                    .context("gqa paged rope tensor")?
            }
        };
        (q, k, gate)
    };

    let v = reshape_hole0_4(&v, seq_len, num_kv_heads, head_dim).context("gqa paged v reshape")?;

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
        q.transpose(1, 2)
            .context("gqa paged q transpose view")?
            .contiguous()
            .context("gqa paged q transpose contiguous")?
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
        && (AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend)
            || AttentionBackend::runtime_supports_flash_attn_prefill(backend))
    {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_initial");
        let k_head = k_cache_token_major.transpose(1, 2)?.contiguous()?;
        let v_head = v_cache_token_major.transpose(1, 2)?.contiguous()?;
        let attn_output = if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k_head, &v_head, num_heads, head_dim)?
        {
            Some(attn_output)
        } else if AttentionBackend::runtime_supports_flash_attn_prefill(backend) {
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
                // #1082: `PagedKvCacheKt` write methods take kt tensors; the
                // token-major / head-major K/V are already kt, so pass them
                // through with no candle bridge.
                // (#1082 all-hardware) `write_token_major_native` / `write` are
                // CUDA-kernel methods (`#[cfg(feature = "cuda")]`). The Vulkan
                // backend never reaches this generic prefill/decode fallback at
                // runtime — it dispatches the single-submit resident path
                // (`model_forward_paged_last_token_resident_native_vk`) which
                // owns KV in `VkPagedKvCache`. The non-CUDA arm bails.
                #[cfg(feature = "cuda")]
                {
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
                #[cfg(not(feature = "cuda"))]
                {
                    // (#1082 DoD-100) `write_token_major_native` is a CUDA-kernel
                    // fast path; on the non-CUDA path use the device-agnostic
                    // native `write` (BF16 paged-KV scatter via kt `slice_set` +
                    // host slot math). The Vulkan backend never reaches this
                    // generic fallback at runtime (it owns KV in `VkPagedKvCache`
                    // via the resident path), but this still compiles for it.
                    let _ = (&k_cache_token_major, &v_cache_token_major);
                    paged_cache
                        .write(
                            full_attn_layer_idx,
                            block_table,
                            start_pos,
                            &k_head,
                            &v_head,
                        )
                        .context("paged KV cache write (non-CUDA) failed")?;
                }
            }

            // Phase B12 layer-31 GQA tap: attn_out. Captured AFTER the gate
            // multiply (if `attn_output_gate`) and BEFORE o_proj, so it
            // matches the HF reference's `attn_output = ... * sigmoid_gate`
            // tap point. Shape: [B, T, num_heads * head_dim].
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            // Phase B12 layer-31 GQA tap: o_proj output (post-o_proj).
            return Ok(out);
        }
    }

    // The scoped MTP draft block attends only to the just-computed K/V
    // (kv_len = 1, no history). Skip the paged-cache write/read and the fused
    // paged-decode kernel so the per-step (k, v) above becomes the SDPA input.
    // `MtpAttentionScope` restores ordinary attention on every exit path.
    let single_token_self_attn = crate::mtp_runtime::single_token_self_attention_active();

    // Write new K/V into paged cache.
    if !single_token_self_attn {
        kiln_nvtx::range!(c"kiln/kv/copy");
        let graph_write_done = {
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                if let Some(inputs) = graph_inputs {
                    // #1082: `PagedKvCacheKt::write_token_major_native_graph_slot`
                    // takes kt tensors; `k_cache_token_major`/`v_cache_token_major`
                    // and `inputs.kv_slot` are already kt, so pass them with no
                    // candle bridge. ROCm: the primary cache is already a
                    // `PagedKvCacheKt`, so the kt-twin mirror below is CUDA-only.
                    let done = paged_cache.write_token_major_native_graph_slot(
                        full_attn_layer_idx,
                        &k_cache_token_major,
                        &v_cache_token_major,
                        inputs.kv_slot,
                    )?;
                    #[cfg(feature = "cuda")]
                    // Phase 7 #1082: when the legacy PagedKvCache writer
                    // succeeded and a kt twin cache is plumbed through (i.e. the
                    // `accelerator.kt_api_mode = "all"` gate is on and the
                    // owning struct allocated a kt cache via
                    // `try_kt_paged_kv_cache_new`), mirror the same write
                    // into the kt cache through
                    // `try_kt_paged_kv_write_token_major_native_graph_slot`.
                    // Both caches hold the same K/V/slot device storage
                    // (the helper *borrows* k/v/slot rather than copying),
                    // so any divergence between the two writers surfaces
                    // immediately in downstream reads. When `kt_paged_cache`
                    // is `None` (default), the helper short-circuits to
                    // `Ok(false)` and this branch is zero overhead — the
                    // legacy cache write above is the only thing that ran.
                    if done && kt_paged_cache.is_some() {
                        let _kt_done = try_kt_paged_kv_write_token_major_native_graph_slot(
                            kt_paged_cache,
                            full_attn_layer_idx,
                            &k_cache_token_major,
                            &v_cache_token_major,
                            inputs.kv_slot,
                        )?;
                    }
                    done
                } else {
                    false
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                false
            }
        };
        // #1082: bridge kt K/V to candle for the candle-island write (see the
        // initial-prefill write above). kt twin cache write happens separately.
        // Bridge lazily so the graph-slot fast path (`graph_write_done`) skips
        // the candle copy entirely.
        if !graph_write_done {
            // #1082: `PagedKvCacheKt` write methods take kt tensors;
            // `k_cache_token_major`/`v_cache_token_major` are already kt, so
            // pass them through with no candle bridge.
            // (#1082 all-hardware) CUDA-kernel writers; the Vulkan backend uses
            // the resident-decode path (`VkPagedKvCache`) and never reaches this
            // generic fallback, so the non-CUDA arm bails.
            #[cfg(feature = "cuda")]
            {
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
            #[cfg(not(feature = "cuda"))]
            {
                // (#1082 DoD-100) CPU path: device-agnostic native `write`
                // (BF16), mirroring the CUDA fallback's token-major -> head-major
                // transpose. Vulkan never reaches this generic fallback at
                // runtime (resident path), but it compiles.
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
                    .context("paged KV cache write (non-CUDA) failed")?;
            }
        }
    }

    // Fast path: fused paged-decode flash-attention kernel.
    // Eliminates the materializing `paged_cache.read()` (an `index_select` /
    // u8→bf16 dequant) on the decode hot path. Limited to:
    //   * Backends that advertise `supports_flash_attn_paged_decode()`
    //   * Decode steps (seq_len == 1)
    //   * Non-FP8 caches (the kernel reads bf16 pool slots directly)
    //   * Page sizes that divide kBlockN=128 (block_size=16 satisfies this)
    //   * CUDA still requires one physically-contiguous page run per kBlockN
    //     chunk. Vulkan uses the block-table gather kernel and supports
    //     non-contiguous physical pages.
    //   * Phase C8: not in single-token self-attn mode (kernel reads the
    //     full cache history, defeating the kv_len = 1 contract).
    if seq_len == 1
        && !single_token_self_attn
        && {
            #[cfg(feature = "cuda")]
            {
                !try_kt_paged_kv_is_fp8(paged_cache.is_fp8(), kt_paged_cache)
            }
            // ROCm: the rocm SDPA paged-decode dequantizes U8 (FP8 E4M3FN) pool
            // slots to BF16 right after the gather, so FP8 caches take the fused
            // fast path here too — which is bucketed (max_seqlen_k) and therefore
            // HIP-graph-capturable, unlike the eager fallback's seq-len-dependent
            // broadcast attention. Other non-CUDA backends still exclude FP8.
            #[cfg(all(feature = "rocm", not(feature = "cuda")))]
            {
                let _ = paged_cache.is_fp8();
                true
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                !paged_cache.is_fp8()
            }
        }
        && (num_heads / num_kv_heads) > 1
        && !fused_paged_decode_disabled(q.device())
        && AttentionBackend::runtime_supports_flash_attn_paged_decode(backend)
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
                use_metal_decode_gemv,
                attn_weights,
                lora_layer,
                lora_scale,
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                graph_inputs,
                #[cfg(feature = "cuda")]
                kt_paged_cache,
            )?
        };
        if let Some(out) = out_opt {
            return Ok(out);
        }
        let portable_lora_fallback = lora_layer.is_some() && portable_lora_decode_allowed(backend);
        if native_decode_attention_required(backend) && !portable_lora_fallback {
            anyhow::bail!(
                "native paged decode declined native paged-attention path; \
                 generic fallback disabled by backend policy"
            );
        }
        #[cfg(feature = "vulkan")]
        if portable_lora_fallback {
            record_vulkan_lora_paged_decode_fallback(full_attn_layer_idx, total_seq_len);
        }
    }

    #[cfg(feature = "rocm")]
    if graph_inputs.is_some() {
        return Err(RocmGraphShapeDependentAttention.into());
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
        let prefix_only_prefill = seq_len > 1
            && start_pos > 0
            && {
                #[cfg(feature = "cuda")]
                {
                    !try_kt_paged_kv_is_fp8(paged_cache.is_fp8(), kt_paged_cache)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    !paged_cache.is_fp8()
                }
            }
            && AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend);
        let append_head_major_read_supported =
            PagedKvBackend::runtime_supports_paged_kv_head_major_read_append_token_major(backend);
        let prefix_append_fast = if prefix_only_prefill
            && start_pos >= PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS
            && append_head_major_read_supported
        {
            contiguous_slot_run_start(
                block_table,
                {
                    #[cfg(feature = "cuda")]
                    {
                        try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        paged_cache.block_size()
                    }
                },
                0,
                start_pos,
            )
            .and_then(|start_slot| {
                paged_cache
                    .pool_tensors(full_attn_layer_idx)
                    .map(|(k_pool, v_pool)| (start_slot, k_pool, v_pool))
            })
            .map(|(start_slot, k_pool, v_pool)| {
                // #1082: `PagedKvCacheKt::pool_tensors` already yields kt pool
                // references; pass them straight to the kt backend read.
                PagedKvBackend::runtime_paged_kv_head_major_read_append_token_major(
                    backend,
                    &k_pool,
                    &v_pool,
                    start_slot,
                    start_pos,
                    &k_cache_token_major,
                    &v_cache_token_major,
                )
            })
            .transpose()?
            .flatten()
        } else {
            None
        };
        let fast_read_len = if prefix_only_prefill {
            start_pos
        } else {
            total_seq_len
        };
        let fast_read = if seq_len > 1
            && fast_read_len >= PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS
            && {
                #[cfg(feature = "cuda")]
                {
                    !try_kt_paged_kv_is_fp8(paged_cache.is_fp8(), kt_paged_cache)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    !paged_cache.is_fp8()
                }
            }
            && PagedKvBackend::runtime_supports_paged_kv_head_major_read(backend)
            && AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend)
        {
            contiguous_slot_run_start(
                block_table,
                {
                    #[cfg(feature = "cuda")]
                    {
                        try_kt_paged_kv_block_size(paged_cache.block_size(), kt_paged_cache)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        paged_cache.block_size()
                    }
                },
                0,
                fast_read_len,
            )
            .and_then(|start_slot| {
                paged_cache
                    .pool_tensors(full_attn_layer_idx)
                    .map(|(k_pool, v_pool)| (start_slot, k_pool, v_pool))
            })
            .map(|(start_slot, k_pool, v_pool)| {
                // #1082: `PagedKvCacheKt::pool_tensors` already yields kt pool
                // references; pass them straight to the kt backend read.
                PagedKvBackend::runtime_paged_kv_head_major_read(
                    backend,
                    &k_pool,
                    &v_pool,
                    start_slot,
                    fast_read_len,
                )
            })
            .transpose()?
            .flatten()
        } else {
            None
        };
        let (k, v) = if prefix_only_prefill {
            match prefix_append_fast {
                Some((k, v)) => (k, v),
                None => {
                    let (prefix_k, prefix_v) = match fast_read {
                        Some((k, v)) => (k, v),
                        None => {
                            #[cfg(feature = "cuda")]
                            {
                                try_kt_paged_kv_read(
                                    paged_cache,
                                    kt_paged_cache,
                                    full_attn_layer_idx,
                                    block_table,
                                    start_pos,
                                )
                                .context("paged KV cache prefix read failed")?
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                try_kt_paged_kv_read(
                                    paged_cache,
                                    full_attn_layer_idx,
                                    block_table,
                                    start_pos,
                                )
                                .context("paged KV cache prefix read failed")?
                            }
                        }
                    };
                    let current_k = k_cache_token_major.transpose(1, 2)?.contiguous()?;
                    let current_v = v_cache_token_major.transpose(1, 2)?.contiguous()?;
                    (
                        Tensor::cat(&[&prefix_k, &current_k], 2)?,
                        Tensor::cat(&[&prefix_v, &current_v], 2)?,
                    )
                }
            }
        } else {
            match fast_read {
                Some((k, v)) => (k, v),
                None => {
                    #[cfg(feature = "cuda")]
                    let out = try_kt_paged_kv_read(
                        paged_cache,
                        kt_paged_cache,
                        full_attn_layer_idx,
                        block_table,
                        total_seq_len,
                    )
                    .context("paged KV cache read failed")?;
                    #[cfg(not(feature = "cuda"))]
                    let out = try_kt_paged_kv_read(
                        paged_cache,
                        full_attn_layer_idx,
                        block_table,
                        total_seq_len,
                    )
                    .context("paged KV cache read failed")?;
                    out
                }
            }
        };
        (k, v, total_seq_len)
    };

    // Multi-token append / speculative verify with prefix history. `read`
    // already returns head-major K/V; on Metal, keep Q/K/V in that layout and
    // avoid token-major transposes plus GQA K/V expansion.
    if seq_len > 1 && AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend) {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_head_major");
        if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k, &v, num_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            return Ok(out);
        }
    }

    // Fused-attention path for prefill with existing prefix history
    // (`start_pos > 0`). Initial prefill is special-cased above so we do not
    // materialize the same K/V we just produced.
    // Paged cache returns [batch, heads, kv_len, head_dim] — transpose to
    // [batch, kv_len, heads, head_dim] for the backend kernel.
    if seq_len > 1 && AttentionBackend::runtime_supports_flash_attn_prefill(backend) {
        kiln_nvtx::range!(c"kiln/attn/full/prefill");
        let q = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            // Phase B12 layer-31 GQA tap (secondary prefill path).
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            return Ok(out);
        }
    }

    // GQA head expansion and attention
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;

    // (#1082 Vulkan) The paged-cache read returns K/V in the pool dtype (BF16),
    // but the Vulkan attention computes in F32 (q comes from F32 projections)
    // and the kt matmul requires equal operand dtype and device. Normalize K/V
    // against q so the manual GQA SDPA (Q@Kᵀ and probs@V below) remains
    // coherent. No-op on CUDA when q/k/v already match. This is the
    // cross-length prefill / split-tail fallback the self-attention flash
    // kernel declines.
    let (k, v) = align_gqa_kv_to_query(&q, &k, &v)?;

    // Optimized decode path (seq_len == 1): reshape Q instead of expanding K/V.
    // Q is [batch, num_heads, 1, head_dim] (1 token) while K/V is
    // [batch, num_kv_heads, kv_len, head_dim] (full history). Expanding K/V
    // copies kv_len * head_dim * num_kv_heads data gqa_ratio times.
    // Instead, group Q heads to match KV heads and compute per-group attention.
    if seq_len == 1 && gqa_ratio > 1 {
        let scale = (head_dim as f64).sqrt();

        // Reshape Q: [batch, num_heads, 1, head_dim]
        //          -> [batch, num_kv_heads, gqa_ratio, 1, head_dim]
        //          -> [batch * num_kv_heads, gqa_ratio, 1, head_dim]
        // K:         [batch, num_kv_heads, kv_len, head_dim]
        //          -> [batch * num_kv_heads, kv_len, head_dim]
        // V:         same as K
        let (q_grouped, k_flat, v_flat) = {
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
            (q_grouped, k_flat, v_flat)
        };

        // Attention scores: [batch*num_kv_heads, gqa_ratio, 1, kv_len]
        let attn_scores = {
            let k_grouped = k_flat
                .broadcast_as((batch * num_kv_heads, gqa_ratio, kv_len, head_dim))?
                .contiguous()?;
            let attn_scores = kiln_tensor::ops::matmul_rhs_transposed(&q_grouped, &k_grouped)?;
            // kt has no `Tensor / f64`; `x / scale == x * (1/scale)` via affine.
            attn_scores.affine(1.0 / scale, 0.0)?
        };

        // No causal mask needed for decode (q_len=1 attends to everything)
        let attn_weights_softmax = { cuda_softmax_last_dim(&attn_scores)? };

        // Weighted sum: [batch*num_kv_heads, gqa_ratio, 1, head_dim]
        let attn_output = {
            let attn_output = attn_weights_softmax.broadcast_matmul(&v_flat)?;

            // Reshape back: -> [batch, num_kv_heads * gqa_ratio, 1, head_dim]
            //               == [batch, num_heads, 1, head_dim]
            attn_output
                .reshape((batch, num_heads, 1, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, 1, num_heads * head_dim))?
        };

        // Phase C7: final SDPA output tap at the same point as post_attn_raw,
        // shape [batch, q_len=1, num_heads*head_dim] = [1, 1, 4096].

        let attn_output =
            { attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())? };
        // Phase B12 layer-31 GQA tap (grouped decode path).
        let out = {
            kiln_nvtx::range!(c"kiln/proj/o");
            linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                &attn_output,
                &attn_weights.o_proj_t,
                lora_layer.and_then(|l| l.o_proj.as_ref()),
                lora_scale,
            )?
        };
        return Ok(out);
    }

    // Standard path (prefill without flash-attn, or gqa_ratio == 1)
    let (k, v) = if gqa_ratio > 1 {
        let k = k
            .unsqueeze(2)?
            .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand([batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn_scores = {
        let attn_scores = kiln_tensor::ops::matmul_rhs_transposed(&q, &k)?;
        // kt has no `Tensor / f64`; `x / scale == x * (1/scale)` via affine.
        attn_scores.affine(1.0 / scale, 0.0)?
    };

    let past_len = kv_len - seq_len;
    let attn_scores = { apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)? };

    let attn_weights_softmax = { cuda_softmax_last_dim(&attn_scores)? };
    let attn_output = { attn_weights_softmax.broadcast_matmul(&v)? };

    // Transpose back and output projection
    let attn_output = {
        reshape_hole0_3(
            &attn_output.transpose(1, 2)?.contiguous()?,
            seq_len,
            num_heads * head_dim,
        )?
    };

    let attn_output =
        { attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())? };
    // Phase B12 layer-31 GQA tap (standard fallback path).

    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t_backend_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    Ok(out)
}

/// Apply a causal (lower-triangular) mask to attention scores.
/// Sets future positions to -inf so softmax zeroes them out.
#[allow(dead_code)]
pub(super) fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
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
pub(super) fn apply_causal_mask_with_offset(
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
    let mask = Tensor::new(&mask, device)?.reshape((1, 1, q_len, kv_len))?;
    let mask = mask.to_dtype(scores.dtype())?;
    let out = scores.broadcast_add(&mask)?;
    Ok(out)
}

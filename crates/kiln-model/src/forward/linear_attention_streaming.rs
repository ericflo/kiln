use super::*;

/// Streaming/tiled wrapper around [`gated_deltanet_forward`] for the
/// training-time forward path.
///
/// Slices `x: [B, T, hidden]` along T into tiles of `tile_size` (the last
/// tile may be partial), calls [`gated_deltanet_forward`] per tile threading
/// `recurrent_state` and `conv_state` across tile boundaries, and
/// concatenates the per-tile outputs back into `[B, T, hidden]` along T.
///
/// Tiling reduces peak transient activation memory: GDN's F32 intermediates
/// inside the conv1d / l2_normalize / chunkwise paths allocate per-call
/// buffers sized by the input length, so smaller tiles → smaller transient
/// allocations. The `LinearAttentionState` recurrent + conv state hand-off
/// makes this bit-exact with the monolithic call by construction
/// (the inference path uses the same hand-off in
/// [`model_forward_paged_streaming_with`]).
///
/// `tile_size` must be a positive multiple of `GDN_CHUNK_SIZE`. The last
/// tile may be smaller; partial tile lengths are handled by
/// [`gated_deltanet_forward`] itself (the same way the inference streaming
/// path handles a non-aligned final tile).
///
/// Used by [`model_forward_segment_with_policy`] when the injected policy
/// enables streaming and the segment's sequence length exceeds `tile_size`.
// Intentionally wide forward signature; bundling would break the public API.
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_forward_streaming(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    tile_size: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    if tile_size == 0 || !tile_size.is_multiple_of(GDN_CHUNK_SIZE) {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }
    let (_b, total, _h) = x.dims3()?;
    if total == 0 {
        anyhow::bail!("gated_deltanet_forward_streaming requires at least one token");
    }
    if total <= tile_size {
        // Single tile — no benefit from the cat overhead, defer to the
        // monolithic path so behavior matches the env-off case bit-exactly.
        return gated_deltanet_forward_decode_if(
            backend,
            x,
            weights,
            config,
            recurrent_state,
            conv_state,
            true,
            false,
            true,
            true,
            lora,
        );
    }

    let cap = total.div_ceil(tile_size);
    let mut tile_outs: Vec<Tensor> = Vec::with_capacity(cap);
    let tile_device = x.device();
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let len = end - cursor;
        let allow_forward_only_fastpaths =
            streaming_gdn_forward_only_fastpaths_allowed(&tile_device);
        let allow_prefill_recurrent_kernel = allow_forward_only_fastpaths;
        let mut run_tile = || -> Result<Tensor> {
            let tile_in = x.narrow(1, cursor, len)?;
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let tile_in = if crate::tape_forward::tape_scope_active() {
                require_active_tape_output(
                    crate::tape_forward::try_tape_narrow_kt(x, 1, cursor, len, &tile_in)
                        .with_context(|| {
                            format!("streaming GDN input tile [{cursor}, {end}) tape narrow")
                        })?,
                    "streaming GDN input narrow",
                )?
            } else {
                tile_in
            };
            gated_deltanet_forward_decode_if(
                backend,
                &tile_in,
                weights,
                config,
                recurrent_state,
                conv_state,
                true,
                false,
                allow_forward_only_fastpaths,
                allow_prefill_recurrent_kernel,
                lora,
            )
            .with_context(|| {
                format!("streaming GDN tile [{cursor}, {end}) of {total} (tile_size={tile_size})")
            })
        };
        let tile_out = {
            #[cfg(feature = "metal")]
            {
                if let Device::Metal(mi) = tile_device {
                    let tile_out = metal_autoreleasepool(|| run_tile())?;
                    // #1082: kt-native device sync — wait on the Metal
                    // companion's command queue (candle's `Device::synchronize`
                    // is gone). Bounds the streaming-GDN tile's GPU work before
                    // the host reads the next tile.
                    kiln_tensor::primary_metal_companion(mi)
                        .and_then(|c| c.wait_until_completed())
                        .with_context(|| {
                            format!("synchronize streaming GDN tile [{cursor}, {end}) of {total}")
                        })?;
                    tile_out
                } else {
                    run_tile()?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                run_tile()?
            }
        };
        #[cfg(feature = "rocm")]
        if matches!(tile_device, Device::Rocm(_)) {
            // Streaming GDN threads recurrent/conv state across tile calls. The
            // tile body mixes hipBLASLt GEMMs and kt kernels, so a tensor-stream
            // sync can miss producer work on the ROCm companion/default stream.
            // Use the same stronger ROCm barrier as long full-attention handoffs;
            // this preserves exact math while making the inter-tile dependency
            // explicit.
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!("streaming GDN tile [{cursor}, {end}) output"),
                &tile_out,
            )?;
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!("streaming GDN tile [{cursor}, {end}) recurrent state"),
                &*recurrent_state,
            )?;
            synchronize_tensor_ready_for_full_attn_handoff(
                &format!("streaming GDN tile [{cursor}, {end}) conv state"),
                &*conv_state,
            )?;
        }
        tile_outs.push(tile_out);
        cursor = end;
    }

    let tile_refs: Vec<&Tensor> = tile_outs.iter().collect();
    // Phase 7 (#1082): when stable KT routes are enabled and all tile outputs are
    // contiguous CUDA tensors of a supported dtype, route the
    // `Tensor::cat(&tile_refs, 1)` step through
    // `kiln_tensor::cuda_concat(_, 1)` via the kt-bridge borrow
    // adapter. Falls through to the candle composite when any
    // precondition fails.
    let out = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) =
                try_kt_cat_dim1(&tile_refs).context("streaming GDN try_kt_cat_dim1")?
            {
                out
            } else {
                Tensor::cat(&tile_refs, 1).context("streaming GDN cat tile outputs along T axis")?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Tensor::cat(&tile_refs, 1).context("streaming GDN cat tile outputs along T axis")?
        }
    };
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_concat_kt(&tile_refs, 1, &out)
                .context("streaming GDN tile concat try_tape_concat_kt")?,
            "streaming GDN tile concatenation",
        );
    }
    Ok(out)
}

pub(super) fn replace_gdn_recurrent_state_handle(
    backend: &dyn BackendRuntime,
    state: &mut Tensor,
    replacement: Tensor,
    operation: &'static str,
) -> Result<()> {
    ResidencyBackend::runtime_rekey_gdn_recurrent_resident_state(backend, state, &replacement)
        .with_context(|| format!("GDN resident-state ownership transfer during {operation}"))?;
    *state = replacement;
    Ok(())
}

pub(super) struct GdnPrefillResidentStateLayerScope<'a> {
    backend: &'a dyn BackendRuntime,
    active: bool,
}

impl<'a> GdnPrefillResidentStateLayerScope<'a> {
    pub(super) fn new(backend: &'a dyn BackendRuntime, layer_idx: usize) -> Self {
        let active = ResidencyBackend::runtime_enter_gdn_prefill_resident_state_layer_scope(
            backend, layer_idx,
        );
        Self { backend, active }
    }
}

impl Drop for GdnPrefillResidentStateLayerScope<'_> {
    fn drop(&mut self) {
        if self.active {
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_layer_scope(self.backend);
        }
    }
}

/// #1082 box-102 BUG2 fix: wrap the GDN decode so the recurrent + conv state
/// update lands IN-PLACE in the caller's persistent buffers. The inner decode
/// updates them functionally (`*state = <new tensor>`); under CUDA-graph capture
/// that Rust reassignment never runs on replay, so the next replay reads a stale
/// state → the GDN state freezes across replays → token-doubling (the original
/// diagnosis observed state norms changing only at re-capture boundaries).
/// Snapshot the persistent buffers, run the
/// decode, then copy the new state back into them in-place via `slice_set` — a
/// captured device→device copy that survives replay — and restore the slots.
/// `Tensor::clone` shares the storage Arc + copies the id, so an unchanged slot
/// (already in-place, e.g. the Vulkan resident path) is detected by id-equality
/// and skipped. Eager decode is value-identical (the copy preserves the state).
#[allow(clippy::too_many_arguments)]
pub(super) fn gated_deltanet_forward_decode_if(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    use_fused_gdn_gates: bool,
    use_metal_decode_gemv: bool,
    allow_forward_only_fastpaths: bool,
    allow_prefill_recurrent_kernel: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let rs_persist = recurrent_state.clone();
    let cv_persist = conv_state.clone();
    let out = gated_deltanet_forward_decode_if_inner(
        backend,
        x,
        weights,
        config,
        recurrent_state,
        conv_state,
        use_fused_gdn_gates,
        use_metal_decode_gemv,
        allow_forward_only_fastpaths,
        allow_prefill_recurrent_kernel,
        lora,
    )?;
    if recurrent_state.id() != rs_persist.id() {
        if ResidencyBackend::runtime_rekey_gdn_recurrent_resident_state(
            backend,
            recurrent_state,
            &rs_persist,
        )
        .context("GDN resident-state ownership transfer to persistent slot")?
        {
            // The backend buffer is authoritative while resident. Restoring the
            // persistent metadata handle must not copy its stale host payload.
            *recurrent_state = rs_persist;
        } else {
            let src = recurrent_state.contiguous()?;
            #[cfg(feature = "rocm")]
            if matches!(src.device(), Device::Rocm(_)) {
                synchronize_tensor_ready_for_model_handoff(
                    "box-102 GDN recurrent-state restore src",
                    &src,
                )?;
            }
            // (#1082) Vulkan lacks `Tensor::slice_set`; the in-place buffer restore
            // is only needed to preserve the persistent buffer identity on backends
            // that support it. Assign the updated state directly when it is on Vulkan
            // OR when it moved devices vs the persistent slot (#1443: the real-model
            // GDN state is CPU-initialized but the forward produces a Vulkan state,
            // so `slice_set` on the CPU `rs_persist` with a Vulkan `src` hit a device
            // mismatch — gating on `rs_persist.device()` was the wrong tensor). The
            // caller's `LinearAttentionState` holds the tensor by value and adopts
            // the new one, so identity preservation is unnecessary for correctness.
            if matches!(src.device(), Device::Vulkan(_)) || rs_persist.device() != src.device() {
                *recurrent_state = src;
            } else {
                rs_persist
                    .slice_set(&src, 0, 0)
                    .context("box-102: in-place GDN recurrent-state restore")?;
                #[cfg(feature = "rocm")]
                if matches!(rs_persist.device(), Device::Rocm(_)) {
                    synchronize_tensor_ready_for_model_handoff(
                        "box-102 GDN recurrent-state restore dst",
                        &rs_persist,
                    )?;
                }
                *recurrent_state = rs_persist;
            }
        }
    }
    if conv_state.id() != cv_persist.id() {
        let src = conv_state.contiguous()?;
        #[cfg(feature = "rocm")]
        if matches!(src.device(), Device::Rocm(_)) {
            synchronize_tensor_ready_for_model_handoff("box-102 GDN conv-state restore src", &src)?;
        }
        if matches!(src.device(), Device::Vulkan(_)) || cv_persist.device() != src.device() {
            *conv_state = src;
        } else {
            cv_persist
                .slice_set(&src, 0, 0)
                .context("box-102: in-place GDN conv-state restore")?;
            #[cfg(feature = "rocm")]
            if matches!(cv_persist.device(), Device::Rocm(_)) {
                synchronize_tensor_ready_for_model_handoff(
                    "box-102 GDN conv-state restore dst",
                    &cv_persist,
                )?;
            }
            *conv_state = cv_persist;
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gated_deltanet_forward_decode_if_inner(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    use_fused_gdn_gates: bool,
    use_metal_decode_gemv: bool,
    allow_forward_only_fastpaths: bool,
    allow_prefill_recurrent_kernel: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let input_dtype = x.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let qkv_dim = config.linear_qkv_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;
    // CP-4 (#1082): the GDN forward-only fast paths (fused conv+split, backend
    // causal_conv1d_prefill, fused gates, unexpanded-qk recurrence) fuse ops the
    // kt Tape can't see and bypass the unfused, tape-wired slow path. They gate on
    // `!x.track_op()` — but the tape-authoritative path's intermediates are detached
    // (track_op==false), so they would fire and sever the GDN chain (conv1d never
    // records → in_proj_qkv disconnects). Disable them when a tape recording scope
    // is active so the unfused, fully-tape-wired GDN forward runs instead. Default
    // (no tape scope) behaviour is unchanged.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let tape_recording_active = crate::tape_forward::tape_scope_active();
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let tape_recording_active = false;
    let gdn_forward_only_fastpaths =
        allow_forward_only_fastpaths && !x.track_op() && !tape_recording_active;
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    let in_proj_qkv_lora = lora_layer.and_then(|l| l.in_proj_qkv.as_ref());
    let in_proj_z_lora = lora_layer.and_then(|l| l.in_proj_z.as_ref());
    let gdn_out_lora = lora_layer.and_then(|l| l.gdn_out_proj.as_ref());
    let has_gdn_in_lora = in_proj_qkv_lora.is_some() || in_proj_z_lora.is_some();

    // Vulkan-resident decode fast-path for GDN. Same shape contract as
    // the full-attn fast-path in transformer_block_paged_with_rope_tables:
    // declines (`Ok(None)`) on any unsupported config so the legacy
    // path below runs unchanged.
    #[cfg(feature = "vulkan")]
    {
        if batch == 1
            && seq_len == 1
            && lora.is_none()
            && gdn_forward_only_fastpaths
            && vulkan_resident_decode_enabled()
        {
            if let Some(vk_backend) = BackendIdentity::runtime_as_any(backend)
                .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
            {
                if let Some(out) =
                    crate::vk_decode_resident::gated_deltanet_forward_decode_resident_b1_kt(
                        vk_backend,
                        x,
                        weights,
                        config,
                        recurrent_state,
                        conv_state,
                    )?
                {
                    return Ok(out);
                }
            }
        }
    }
    // --- Step 1: Input projections ---
    // Use the pre-transposed weight cache (Phase 6) so we don't pay a `.t().contiguous()`
    // ucopy_bf16 copy on every layer / every step. Same fix class as PR #128 (MLP/full-attn).
    let (mixed_qkv, z, a, b, prefill_ab_for_gates) = {
        kiln_nvtx::range!(c"kiln/gdn/in_proj");
        if !has_gdn_in_lora
            && gdn_forward_only_fastpaths
            && seq_len == 1
            && x.dtype() == DType::BF16
            && !x.track_op()
            && let Some(w8) = weights.in_proj_qkvzab_w8.as_ref()
        {
            let fused = crate::rocm_w8_proj::matmul_bf16(x, w8)?;
            let mixed_qkv = fused.narrow(2, 0, qkv_dim)?;
            let z = fused.narrow(2, qkv_dim, v_dim)?;
            let a = fused.narrow(2, qkv_dim + v_dim, nv)?;
            let b = fused.narrow(2, qkv_dim + v_dim + nv, nv)?;
            (mixed_qkv, z, a, b, None::<Tensor>)
        } else if !has_gdn_in_lora
            && gdn_forward_only_fastpaths
            && let Some((mixed_qkv, z, a, b)) = GdnBackend::runtime_gdn_in_proj_decode(
                backend,
                x,
                &weights.in_proj_qkv_t,
                &weights.in_proj_z_t,
                &weights.in_proj_a_t,
                &weights.in_proj_b_t,
            )?
        {
            (mixed_qkv, z, a, b, None::<Tensor>)
        } else {
            let mixed_qkv = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &weights.in_proj_qkv_t,
                in_proj_qkv_lora,
                lora_scale,
            )?; // [B, T, qkv_dim]
            let z = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &weights.in_proj_z_t,
                in_proj_z_lora,
                lora_scale,
            )?; // [B, T, v_dim]
            let prefill_ab: Option<(Tensor, Tensor, Tensor)> = {
                if gdn_forward_only_fastpaths {
                    match weights.in_proj_ab_t.as_ref() {
                        Some(in_proj_ab_t) => GdnBackend::runtime_gdn_ab_in_proj_prefill(
                            backend,
                            x,
                            in_proj_ab_t,
                            nv,
                            seq_len,
                        )?,
                        None => None,
                    }
                } else {
                    None
                }
            };
            if let Some((ab, a, b)) = prefill_ab {
                (mixed_qkv, z, a, b, Some(ab))
            } else {
                let a = gdn_in_proj_matmul(backend, x, &weights.in_proj_a_t)?; // [B, T, nv]
                let b = gdn_in_proj_matmul(backend, x, &weights.in_proj_b_t)?; // [B, T, nv]
                (mixed_qkv, z, a, b, None::<Tensor>)
            }
        }
    };
    #[cfg(not(feature = "metal"))]
    let _ = &prefill_ab_for_gates;

    let scale = 1.0 / (dk as f64).sqrt();
    let recurrent_unexpanded_qk = matches!(input_dtype, DType::BF16 | DType::F32)
        && gdn_forward_only_fastpaths
        && (1..=GDN_RECURRENT_PREFILL_MAX_TOKENS).contains(&seq_len)
        && dk == 128
        && gqa_ratio > 1
        && GdnBackend::runtime_supports_gdn_recurrent_prefill_native_head_last(backend);
    let fused_decode_unexpanded_qk = input_dtype == DType::BF16
        && gdn_forward_only_fastpaths
        && seq_len == 1
        && dk == 128
        && gqa_ratio > 1
        && GdnBackend::runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk(backend);
    #[cfg(feature = "metal")]
    let use_unexpanded_qk = recurrent_unexpanded_qk || fused_decode_unexpanded_qk;
    let fused_decode_qkv_conv_norm = {
        #[cfg(feature = "metal")]
        {
            if use_unexpanded_qk
                && gdn_forward_only_fastpaths
                && crate::backend::metal::metal_gdn_decode_qkv_conv_norm_supports(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                )
            {
                kiln_nvtx::range!(c"kiln/gdn/qkv_conv_norm");
                let (q, k, v) = crate::backend::metal::metal_gdn_decode_qkv_conv_norm_bf16(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                    scale as f32,
                    1e-6,
                )
                .context("metal gdn decode qkv conv/norm kernel failed")?;
                let z = z.reshape((batch, seq_len, nv, dv))?;
                Some((q, k, v, z, false, false, false))
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let fused_prefill_qkv_conv_split = {
        #[cfg(feature = "metal")]
        {
            if fused_decode_qkv_conv_norm.is_none()
                && recurrent_unexpanded_qk
                && gdn_forward_only_fastpaths
                && seq_len > 1
                && crate::backend::metal::metal_gdn_prefill_qkv_conv_split_supports(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                )
            {
                kiln_nvtx::range!(c"kiln/gdn/qkv_conv_split");
                let (q, k, v) =
                    crate::backend::metal::metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
                        &mixed_qkv,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                        nk,
                        dk,
                        nv,
                        dv,
                    )
                    .context("metal gdn prefill qkv conv-split kernel failed")?;
                let (q, k) = {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    gdn_qk_norm(&q, &k, input_dtype, scale)?
                };
                let z = z.reshape((batch, seq_len, nv, dv))?;
                Some((q, k, v, z, false, false, false))
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let (
        q,
        k,
        v,
        z,
        qk_expanded,
        qk_norm_deferred_to_recurrent,
        qk_norm_deferred_to_native_recurrent,
    ) = if let Some(fused) = fused_decode_qkv_conv_norm {
        fused
    } else if let Some(fused) = fused_prefill_qkv_conv_split {
        fused
    } else {
        // --- Step 2: Causal depthwise conv1d + SiLU on fused QKV ---
        //
        // Decode fast path: backend-side `causal_conv1d_update` collapses the
        // to_f32 / cat / sum / narrow / silu chain into one fused update per
        // (batch, channel). It returns F32 with SiLU already fused, so the
        // subsequent `cuda_silu(.to_dtype(F32))` step is skipped. Unsupported
        // backends, non-bf16, and kernel_size != 4 all route through the
        // portable candle path below, which is the parity oracle.
        let mixed_qkv = {
            kiln_nvtx::range!(c"kiln/gdn/conv");
            // Transpose to [B, channels, T] for conv. At seq_len == 1 the
            // [B, 1, C] -> [B, C, 1] axis swap is a no-data-move shape
            // reinterpretation: in row-major, element[b, 0, c] sits at the
            // same offset as element[b, c, 0]. `reshape` on a contiguous
            // input produces a view (no copy); the conv kernel's strict
            // [B, C, 1] dims check accepts the view. Saves the
            // transpose + `contiguous` copy that nsys flagged as ~3 ms /
            // bench-bs=16 in `kiln/gdn/conv/layout`.
            let mixed_qkv_ct = {
                kiln_nvtx::range!(c"kiln/gdn/conv/layout");
                if seq_len == 1 && mixed_qkv.is_contiguous() {
                    let (b, _t, c) = mixed_qkv.dims3()?;
                    let reshaped = mixed_qkv.reshape((b, c, 1))?;
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    if tape_recording_active {
                        require_active_tape_output(
                            crate::tape_forward::try_tape_reshape_kt(&mixed_qkv, vec![b, c, 1])
                                .context("gdn conv layout try_tape_reshape_kt")?,
                            "GDN single-token conv input reshape",
                        )?
                    } else {
                        reshaped
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    reshaped
                } else {
                    // CP-4 Increment 3 (#1082): the [B,T,C]->[B,C,T] conv-input
                    // transpose mints a fresh candle id between the in_proj_qkv
                    // keystone output (`mixed_qkv`) and the conv. Wrap it so the
                    // conv backward's input grad flows back to in_proj_qkv. The
                    // transpose adapter materialises a contiguous copy (matching
                    // the `.contiguous()` here), so it's value-faithful. No-op +
                    // candle fallback unless the gate is on + a tape scope is
                    // active.
                    // #1082 seam flip: kt-native transpose recorder — no kt->candle->kt.
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    {
                        if tape_recording_active {
                            require_active_tape_output(
                                crate::tape_forward::try_tape_transpose_kt(&mixed_qkv, 1, 2)
                                    .context("gdn conv transpose try_tape_transpose_kt")?,
                                "GDN conv input transpose",
                            )?
                        } else {
                            mixed_qkv.transpose(1, 2)?.contiguous()?
                        }
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    {
                        mixed_qkv.transpose(1, 2)?.contiguous()?
                    }
                }
            };
            let record_prefill_conv = |out: &Tensor| -> Result<Tensor> {
                #[cfg(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                ))]
                if tape_recording_active {
                    return require_active_tape_output(
                        crate::tape_forward::try_tape_causal_conv1d_prefill_kt(
                            &mixed_qkv_ct,
                            &weights.conv1d,
                            out,
                            kernel_size,
                        )
                        .context("gdn conv prefill try_tape_causal_conv1d_prefill_kt")?,
                        "GDN causal conv1d prefill",
                    );
                }
                Ok(out.clone())
            };
            let post_silu = if seq_len == 1
                && gdn_forward_only_fastpaths
                && ConvBackend::runtime_supports_causal_conv1d_update(backend)
            {
                let conv_update = {
                    kiln_nvtx::range!(c"kiln/gdn/conv/update");
                    ConvBackend::runtime_causal_conv1d_update(
                        backend,
                        &mixed_qkv_ct,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                    )?
                };
                match conv_update {
                    Some(out) => out, // F32, SiLU fused into the kernel epilogue
                    None => {
                        kiln_nvtx::range!(c"kiln/gdn/conv/fallback_decode");
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
                #[cfg(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                ))]
                let tape_fused_prefill_conv = tape_recording_active;
                #[cfg(not(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                )))]
                let tape_fused_prefill_conv = false;
                if (gdn_forward_only_fastpaths || tape_fused_prefill_conv)
                    && ConvBackend::runtime_supports_causal_conv1d_prefill(backend)
                {
                    let conv_entry_state = conv_state.clone();
                    let conv_prefill = {
                        kiln_nvtx::range!(c"kiln/gdn/conv/prefill_update");
                        ConvBackend::runtime_causal_conv1d_prefill(
                            backend,
                            &mixed_qkv_ct,
                            &weights.conv1d,
                            conv_state,
                            kernel_size,
                        )?
                    };
                    match conv_prefill {
                        Some(out) => {
                            // F32, SiLU fused into the kernel epilogue. In a tape
                            // scope, record a fused conv+SiLU backward that
                            // recomputes the pre-SiLU conv activation from the
                            // saved entry state. If the recorder declines, restore
                            // the state and fall back to the pre-SiLU recorded path.
                            #[cfg(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            ))]
                            {
                                if tape_recording_active {
                                    let recorded =
                                        crate::tape_forward::try_tape_causal_conv1d_prefill_silu_kt(
                                                &mixed_qkv_ct,
                                                &weights.conv1d,
                                                &conv_entry_state,
                                                &out,
                                                kernel_size,
                                            )?;
                                    if let Some(recorded) = recorded {
                                        recorded
                                    } else {
                                        *conv_state = conv_entry_state;
                                        let y = causal_conv1d_prefill(
                                            &mixed_qkv_ct,
                                            &weights.conv1d,
                                            conv_state,
                                            kernel_size,
                                        )?;
                                        let y = record_prefill_conv(&y)?;
                                        cuda_silu(&y)?
                                    }
                                } else {
                                    out
                                }
                            }
                            #[cfg(not(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            )))]
                            {
                                out
                            }
                        }
                        None => {
                            kiln_nvtx::range!(c"kiln/gdn/conv/fallback_prefill");
                            let y = causal_conv1d_prefill(
                                &mixed_qkv_ct,
                                &weights.conv1d,
                                conv_state,
                                kernel_size,
                            )?;
                            // CP-4 Increment 3 (#1082): wire the prefill conv +
                            // its SiLU onto the kt Tape. See the comment on the
                            // sibling fallback branch below.
                            // Device-agnostic backward (CUDA FFI / kt composite), so Vulkan is in.
                            let y = record_prefill_conv(&y)?;
                            #[cfg(feature = "cuda")]
                            {
                                // #1082 seam flip: cuda_silu records the kt-native
                                // SiluBackward on the active tape internally.
                                cuda_silu(&y)?
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                cuda_silu(&y)?
                            }
                        }
                    }
                } else {
                    kiln_nvtx::range!(c"kiln/gdn/conv/fallback_prefill");
                    let y = causal_conv1d_prefill(
                        &mixed_qkv_ct,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                    )?;
                    // CP-4 Increment 3 (#1082): THE single largest GDN gap. The
                    // prefill conv is the bottom-most op separating q/k/v from
                    // in_proj_qkv. Record a CausalConv1dPrefillInputBackward
                    // (wraps the proven `[rows,channels]` bwd-input kernel with a
                    // [B,C,T]<->[rows,C] layout transform) so the recurrence's
                    // dq/dk/dv reach in_proj_qkv. Then wire the SiLU so the chain
                    // stays connected to the qkv_split. Both no-op outside a
                    // tape scope. This is the training path
                    // (track_op=true -> gdn_forward_only_fastpaths=false).
                    // Device-agnostic backward (CUDA FFI / kt composite), so Vulkan is in.
                    let y = record_prefill_conv(&y)?;
                    #[cfg(feature = "cuda")]
                    {
                        // #1082 seam flip: cuda_silu records the kt-native
                        // SiluBackward on the active tape internally.
                        cuda_silu(&y)?
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        cuda_silu(&y)?
                    }
                }
            } else if tape_recording_active {
                // A partial streaming tail can contain exactly one token (for
                // example 1025 tokens with a 256-token tile). Decode's
                // in-place state path is intentionally forward-only, but the
                // portable prefill composite supports T=1 and has the
                // authoritative tape recorder. Use it here so the tail stays
                // connected to in_proj_qkv instead of rejecting an otherwise
                // valid training trajectory. Its shift-and-append state update
                // is algebraically identical to causal_conv1d_decode at T=1.
                kiln_nvtx::range!(c"kiln/gdn/conv/fallback_prefill_single_token_tape");
                let y =
                    causal_conv1d_prefill(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
                let y = record_prefill_conv(&y)?;
                cuda_silu(&y)?
            } else {
                kiln_nvtx::range!(c"kiln/gdn/conv/fallback_decode");
                let y =
                    causal_conv1d_decode(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
                cuda_silu(&y.to_dtype(DType::F32)?)?
            };
            // Transpose back to [B, T, qkv_dim].
            //
            // CP-4 Increment 3 (#1082): the [B,C,T]->[B,T,C] conv-output
            // transpose mints a fresh candle id between the wired SiLU and the
            // qkv_split. Wrap it so the split's narrow grads flow back through
            // the conv. No-op + candle fallback unless the gate is on + a tape
            // scope is active.
            // #1082 seam flip: kt-native transpose recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            {
                if tape_recording_active {
                    require_active_tape_output(
                        crate::tape_forward::try_tape_transpose_kt(&post_silu, 1, 2)
                            .context("gdn conv-out transpose try_tape_transpose_kt")?,
                        "GDN conv output transpose",
                    )?
                } else {
                    post_silu.transpose(1, 2)?
                }
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            {
                post_silu.transpose(1, 2)?
            }
        };

        // Phase B11b tap: `gdn_conv`. Output of the causal depthwise conv1d +
        // SiLU, matching HF's `mixed_qkv` after `self.conv1d(...)[:T]` +
        // `F.silu(...)` (shape [B, T, qkv_dim]).

        // --- Step 3: Split into Q, K, V and reshape to heads ---
        let (q, k, v, z) = {
            kiln_nvtx::range!(c"kiln/gdn/qkv_split");
            // CP-4 Increment 3 (#1082): the narrow (QKV split) + reshape ops mint
            // fresh candle ids between the wired conv output (`mixed_qkv`) and the
            // head_expand / recur_prep. Wrap each on the kt Tape (narrow adjoint =
            // zero-pad; reshape adjoint = inverse reshape) so the recurrence's
            // dq/dk/dv flow back into `mixed_qkv` (and thence conv → in_proj_qkv).
            // The z reshape connects the in_proj_z keystone output to the
            // gated-RMSNorm gate input. No-op + candle fallback unless the gate is
            // on + a tape scope is active.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let narrow_then_reshape = |src: &Tensor,
                                       offset: usize,
                                       length: usize,
                                       shape: (usize, usize, usize, usize)|
             -> Result<Tensor> {
                // Materialise contiguous: `narrow` returns a strided view, but
                // the narrow adapter borrows `out` as a kt tensor (which
                // requires contiguous) and the downstream reshape needs it too.
                // Without this the narrow adapter declines (borrow fails) and
                // the q/k/v become fresh-borrow islands, severing in_proj_qkv
                // from the loss. Value-preserving. (#1082 CP-4 Increment 5)
                let nar = src.narrow(2, offset, length)?.contiguous()?;
                // #1082 seam flip: kt-native NarrowCompositeBackward recorder — no kt->candle->kt.
                let nar = if tape_recording_active {
                    require_active_tape_output(
                        crate::tape_forward::try_tape_narrow_kt(src, 2, offset, length, &nar)
                            .context("gdn qkv narrow try_tape_narrow_kt")?,
                        "GDN QKV narrow",
                    )?
                } else {
                    nar
                };
                let resh = nar.reshape(shape)?;
                // #1082 seam flip: kt-native reshape recorder — no kt->candle->kt.
                if tape_recording_active {
                    require_active_tape_output(
                        crate::tape_forward::try_tape_reshape_kt(
                            &nar,
                            vec![shape.0, shape.1, shape.2, shape.3],
                        )
                        .context("gdn qkv reshape try_tape_reshape_kt")?,
                        "GDN QKV reshape",
                    )
                } else {
                    Ok(resh)
                }
            };
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            let narrow_then_reshape = |src: &Tensor,
                                       offset: usize,
                                       length: usize,
                                       shape: (usize, usize, usize, usize)|
             -> Result<Tensor> {
                Ok(src.narrow(2, offset, length)?.reshape(shape)?)
            };
            let q = narrow_then_reshape(&mixed_qkv, 0, qk_dim, (batch, seq_len, nk, dk))?;
            let k = narrow_then_reshape(&mixed_qkv, qk_dim, qk_dim, (batch, seq_len, nk, dk))?;
            let v = narrow_then_reshape(&mixed_qkv, 2 * qk_dim, v_dim, (batch, seq_len, nv, dv))?;
            let z_reshaped = z.reshape((batch, seq_len, nv, dv))?;
            // #1082 seam flip: kt-native reshape recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let z_reshaped = if tape_recording_active {
                require_active_tape_output(
                    crate::tape_forward::try_tape_reshape_kt(&z, vec![batch, seq_len, nv, dv])
                        .context("gdn z reshape try_tape_reshape_kt")?,
                    "GDN z reshape",
                )?
            } else {
                z_reshaped
            };
            (q, k, v, z_reshaped)
        };

        // --- Step 4/5: GQA head repeat (nk → nv), L2 normalize Q/K, scale Q ---
        //
        // Fast paths: Metal/CUDA/ROCm default to fused F32->BF16 kernels for
        // supported bf16 tensors. These collapse the l2-normalize(Q) + scale(Q) +
        // l2-normalize(K) + dtype-cast chain (~11 candle launches on tiny per-row
        // tensors at decode shape) into a single launch. Backend route profiles
        // own the accelerated/fallback selection for the process lifetime.
        //
        // Both paths produce bf16 outputs in `input_dtype`; only the kernel
        // path skips the F32 round-trip through HBM. The candle path is the
        // parity oracle exercised by `kiln-rmsnorm-kernel`'s
        // `parity_l2_qk_norm_*` tests.
        let defer_backend_qk_norm_to_recurrent = {
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                seq_len == 1
                    && gdn_forward_only_fastpaths
                    && fused_decode_unexpanded_qk
                    && input_dtype == DType::BF16
                    && GdnBackend::runtime_supports_gdn_decode_qk_norm_gates_recurrent(backend)
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                false
            }
        };
        let defer_native_qk_norm_to_recurrent = seq_len == 1
            && gdn_forward_only_fastpaths
            && recurrent_unexpanded_qk
            && input_dtype == DType::BF16
            && GdnBackend::runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last(backend);
        let normalize_before_gqa_expand_for_tape = tape_recording_active && gqa_ratio > 1;
        let normalize_then_expand_qk_for_tape =
            |q_src: &Tensor, k_src: &Tensor| -> Result<(Tensor, Tensor)> {
                let (q_norm, k_norm) = gdn_qk_norm(q_src, k_src, input_dtype, scale)?;
                let expand = |src: &Tensor, label: &'static str| -> Result<Tensor> {
                    let expanded = src
                        .unsqueeze(3)?
                        .expand([batch, seq_len, nk, gqa_ratio, dk])?
                        .contiguous()?
                        .reshape((batch, seq_len, nv, dk))?;
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    {
                        require_active_tape_output(
                            crate::tape_forward::try_tape_gqa_expand_kt(src, gqa_ratio, &expanded)
                                .with_context(|| {
                                    format!("gdn qk-norm-before-expand tape expand {label}")
                                })?,
                            &format!("GDN {label} GQA expansion"),
                        )
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    {
                        let _ = label;
                        Ok(expanded)
                    }
                };
                Ok((expand(&q_norm, "q")?, expand(&k_norm, "k")?))
            };
        let (q, k, qk_expanded, qk_norm_deferred, qk_norm_deferred_to_native_recurrent) = {
            #[cfg(feature = "metal")]
            {
                if use_unexpanded_qk {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, false, false, false)
                } else if normalize_before_gqa_expand_for_tape {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_pre_expand_tape");
                    let (q, k) = normalize_then_expand_qk_for_tape(&q, &k)?;
                    (q, k, true, false, false)
                } else if input_dtype == DType::BF16
                    && gdn_forward_only_fastpaths
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
                    .map(|(q, k)| (q, k, true, false, false))?
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
                    (q, k, true, false, false)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let fused_gqa = {
                    #[cfg(feature = "cuda")]
                    {
                        if !fused_decode_unexpanded_qk
                            && gdn_forward_only_fastpaths
                            && crate::cuda_policy::current_cuda_kernel_policy().fused_l2_qk_norm
                            && input_dtype == DType::BF16
                            && gqa_ratio > 1
                        {
                            let q_contig = q.contiguous()?;
                            let k_contig = k.contiguous()?;
                            if let (Some(q_kt), Some(k_kt)) =
                                (try_borrow_kt_cuda(&q_contig), try_borrow_kt_cuda(&k_contig))
                            {
                                if kiln_rmsnorm_kernel::supports_l2_qk_norm_gqa_kt(&q_kt, &k_kt, nv)
                                {
                                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_gqa");
                                    // #1082: keep the fused L2-QK-norm output as kt —
                                    // the downstream qk_norm tuple arms are kt, so the
                                    // candle copy-out is gone.
                                    let (q, k) = kiln_rmsnorm_kernel::fused_l2_qk_norm_gqa_kt(
                                        &q_kt,
                                        &k_kt,
                                        nv,
                                        scale as f32,
                                        1e-6,
                                    )
                                    .map_err(|e| anyhow::anyhow!("kt fused_l2_qk_norm_gqa: {e}"))?;
                                    Some((q, k))
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
                    #[cfg(not(feature = "cuda"))]
                    {
                        None
                    }
                };

                if defer_backend_qk_norm_to_recurrent {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_deferred");
                    (q, k, false, true, false)
                } else if defer_native_qk_norm_to_recurrent {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_deferred_native");
                    (q, k, false, false, true)
                } else if fused_decode_unexpanded_qk {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, false, false, false)
                } else if normalize_before_gqa_expand_for_tape {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_pre_expand_tape");
                    let (q, k) = normalize_then_expand_qk_for_tape(&q, &k)?;
                    (q, k, true, false, false)
                } else if let Some((q, k)) = fused_gqa {
                    (q, k, true, false, false)
                } else {
                    let (q, k) = {
                        kiln_nvtx::range!(c"kiln/gdn/head_expand");
                        if gqa_ratio > 1 {
                            let q_exp = q
                                .unsqueeze(3)?
                                .expand([batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            let k_exp = k
                                .unsqueeze(3)?
                                .expand([batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            // CP-4 Increment 3 (#1082): the unsqueeze+expand+
                            // contiguous+reshape chain mints a fresh candle id
                            // between the wired qk_split (below) and qk_norm. A
                            // single GqaExpandBackward (adjoint = reshape+sum
                            // over the broadcast head sub-dim) keeps the chain
                            // connected so the recurrence's dq/dk reach the
                            // post-split q/k (and thence in_proj_qkv). No-op +
                            // candle fallback unless the gate is on + a tape
                            // scope is active.
                            // #1082 seam flip: kt-native GqaExpandBackward recorder — no kt->candle->kt.
                            #[cfg(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            ))]
                            let q_exp = if tape_recording_active {
                                require_active_tape_output(
                                    crate::tape_forward::try_tape_gqa_expand_kt(
                                        &q, gqa_ratio, &q_exp,
                                    )
                                    .context("gdn gqa-expand try_tape_gqa_expand_kt q")?,
                                    "GDN query GQA expansion",
                                )?
                            } else {
                                q_exp
                            };
                            #[cfg(any(
                                feature = "cuda",
                                feature = "metal",
                                feature = "vulkan",
                                feature = "rocm"
                            ))]
                            let k_exp = if tape_recording_active {
                                require_active_tape_output(
                                    crate::tape_forward::try_tape_gqa_expand_kt(
                                        &k, gqa_ratio, &k_exp,
                                    )
                                    .context("gdn gqa-expand try_tape_gqa_expand_kt k")?,
                                    "GDN key GQA expansion",
                                )?
                            } else {
                                k_exp
                            };
                            (q_exp, k_exp)
                        } else {
                            (q.contiguous()?, k.contiguous()?)
                        }
                    };
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, true, false, false)
                }
            }
        };
        (
            q,
            k,
            v,
            z,
            qk_expanded,
            qk_norm_deferred,
            qk_norm_deferred_to_native_recurrent,
        )
    };
    // Phase B11b taps: `gdn_qk_norm_q` / `gdn_qk_norm_k`. Both are post-L2
    // normalization (+ Q scaled by 1/sqrt(dk)). Shapes [B, T, nv, dk] (the
    // GQA head-expand above brought nk→nv). HF mirror: `query` / `key` after
    // `query.normalize(dim=-1)` / `key.normalize(dim=-1)` and the Q-scale.

    // --- Step 7: Chunkwise analytical recurrence (Phase 6, approach (b)) ---
    // The recurrent state dtype is the accumulator policy. Inference backends
    // that allocate bf16 state keep the bf16 hot path; training backends allocate
    // F32 state so long-row GDN recurrence stays stable and matches the analytic
    // backward's F32 replay.
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
    let recurrence_dtype = if state_external_dtype == DType::F32 {
        DType::F32
    } else {
        input_dtype
    };
    if state_external_dtype != recurrence_dtype {
        let normalized_state = recurrent_state.to_dtype(recurrence_dtype)?;
        replace_gdn_recurrent_state_handle(
            backend,
            recurrent_state,
            normalized_state,
            "normalize recurrent dtype",
        )?;
    }

    let fused_decode_gates_recurrent_rmsnorm = {
        let mut fused = {
            #[cfg(feature = "metal")]
            {
                if recurrent_unexpanded_qk
                    && seq_len == 1
                    && crate::backend::metal::metal_gdn_decode_gates_recurrent_rmsnorm_supports(
                        &q,
                        &k,
                        &v,
                        &a,
                        &b,
                        &weights.a_log,
                        &weights.dt_bias,
                        recurrent_state,
                        &z,
                        &weights.norm,
                    )
                {
                    kiln_nvtx::range!(c"kiln/gdn/gates_recur_gated_norm");
                    let out = crate::backend::metal::metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
                        &q,
                        &k,
                        &v,
                        &a,
                        &b,
                        &weights.a_log,
                        &weights.dt_bias,
                        recurrent_state,
                        &z,
                        &weights.norm,
                        config.rms_norm_eps as f32,
                    )
                    .context("metal gdn decode gates+recurrent+gated-rmsnorm kernel failed")?;
                    Some(out)
                } else {
                    None
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                None
            }
        };
        if fused.is_none() && qk_norm_deferred_to_recurrent {
            kiln_nvtx::range!(c"kiln/gdn/qk_norm_gates_recur_gated_norm");
            fused = GdnBackend::runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
                backend,
                &q,
                &k,
                &v,
                &a,
                &b,
                &weights.a_log_gates,
                &weights.dt_bias,
                recurrent_state,
                &z,
                &weights.norm,
                scale,
                1e-6,
                config.rms_norm_eps,
            )?;
        }
        fused
    };

    let fused_decode_gates_recurrent = {
        if fused_decode_gates_recurrent_rmsnorm.is_none()
            && gdn_forward_only_fastpaths
            && seq_len == 1
        {
            if qk_norm_deferred_to_recurrent {
                kiln_nvtx::range!(c"kiln/gdn/qk_norm_gates_recur");
                let out = if let Some(out) = GdnBackend::runtime_gdn_decode_qk_norm_gates_recurrent(
                    backend,
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &weights.a_log_gates,
                    &weights.dt_bias,
                    recurrent_state,
                    scale,
                    1e-6,
                )? {
                    out
                } else {
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    GdnBackend::runtime_gdn_decode_gates_recurrent(
                        backend,
                        &q,
                        &k,
                        &v,
                        &a,
                        &b,
                        &weights.a_log_gates,
                        &weights.dt_bias,
                        recurrent_state,
                        &z,
                        &weights.norm,
                        config.rms_norm_eps,
                    )?
                    .context("CUDA deferred qk_norm fallback recurrent path declined")?
                };
                Some(out)
            } else if let Some(out) = GdnBackend::runtime_gdn_decode_gates_recurrent(
                backend,
                &q,
                &k,
                &v,
                &a,
                &b,
                &weights.a_log_gates,
                &weights.dt_bias,
                recurrent_state,
                &z,
                &weights.norm,
                config.rms_norm_eps,
            )? {
                kiln_nvtx::range!(c"kiln/gdn/gates_recur");
                Some(out)
            } else {
                #[cfg(feature = "metal")]
                {
                    if recurrent_unexpanded_qk
                        && crate::backend::metal::metal_gdn_decode_gates_recurrent_supports(
                            &q,
                            &k,
                            &v,
                            &a,
                            &b,
                            &weights.a_log,
                            &weights.dt_bias,
                            recurrent_state,
                        )
                    {
                        kiln_nvtx::range!(c"kiln/gdn/gates_recur");
                        let out = crate::backend::metal::metal_gdn_decode_gates_recurrent_bf16(
                            &q,
                            &k,
                            &v,
                            &a,
                            &b,
                            &weights.a_log,
                            &weights.dt_bias,
                            recurrent_state,
                        )
                        .context("metal gdn decode gates+recurrent kernel failed")?;
                        Some(out)
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    None
                }
            }
        } else {
            None
        }
    };

    let fused_prefill_decay_recurrent = {
        #[cfg(feature = "metal")]
        {
            if fused_decode_gates_recurrent_rmsnorm.is_none()
                && fused_decode_gates_recurrent.is_none()
                && !tape_recording_active
                && recurrent_unexpanded_qk
                && seq_len > 1
                && use_fused_gdn_gates
                && crate::backend::metal::metal_gdn_gates_decay_supports(
                    &a,
                    &b,
                    &weights.a_log,
                    &weights.dt_bias,
                )
            {
                let v_recur = v.to_dtype(input_dtype)?;
                if crate::backend::metal::metal_gdn_recurrent_prefill_native_head_last_decay_supports(
                    &q,
                    &k,
                    &v_recur,
                    &a,
                    &b,
                    recurrent_state,
                ) {
                    let (beta, decay) = {
                        kiln_nvtx::range!(c"kiln/gdn/gates");
                        if let Some(ab) = prefill_ab_for_gates.as_ref() {
                            if crate::backend::metal::metal_gdn_gates_decay_ab_supports(
                                ab,
                                &weights.a_log,
                                &weights.dt_bias,
                                nv,
                            ) {
                                crate::backend::metal::metal_gdn_gates_decay_ab_bf16(
                                    ab,
                                    &weights.a_log,
                                    &weights.dt_bias,
                                    nv,
                                )
                                .context("metal gdn prefill A/B gates decay kernel failed")?
                            } else {
                                crate::backend::metal::metal_gdn_gates_decay_bf16(
                                    &a,
                                    &b,
                                    &weights.a_log,
                                    &weights.dt_bias,
                                )
                                .context("metal gdn prefill gates decay kernel failed")?
                            }
                        } else {
                            crate::backend::metal::metal_gdn_gates_decay_bf16(
                                &a,
                                &b,
                                &weights.a_log,
                                &weights.dt_bias,
                            )
                            .context("metal gdn prefill gates decay kernel failed")?
                        }
                    };

                    kiln_nvtx::range!(c"kiln/gdn/recurrent");
                    let attn_out =
                        crate::backend::metal::metal_gdn_recurrent_prefill_native_head_last_decay_bf16(
                            &q,
                            &k,
                            &v_recur,
                            &beta,
                            &decay,
                            recurrent_state,
                        )
                        .context("metal gdn prefill recurrent decay kernel failed")?;
                    Some(attn_out)
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let expanded_qk_for_split = |q: Tensor, k: Tensor| -> Result<(Tensor, Tensor)> {
        if qk_expanded {
            Ok((q, k))
        } else {
            kiln_nvtx::range!(c"kiln/gdn/head_expand_recur_fallback");
            let q_exp = q
                .unsqueeze(3)?
                .expand([batch, seq_len, nk, gqa_ratio, dk])?
                .contiguous()?
                .reshape((batch, seq_len, nv, dk))?;
            let k_exp = k
                .unsqueeze(3)?
                .expand([batch, seq_len, nk, gqa_ratio, dk])?
                .contiguous()?
                .reshape((batch, seq_len, nv, dk))?;
            // CP-4 Increment 3 (#1082): same head-expand chaining fix as the
            // main `kiln/gdn/head_expand` site, for the deferred/native
            // recurrence path (qk_expanded == false). Off the BF16-prefill
            // training path (which sets qk_expanded == true) but wired for
            // parity across configs.
            // #1082 seam flip: kt-native GqaExpandBackward recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let q_exp = if tape_recording_active {
                require_active_tape_output(
                    crate::tape_forward::try_tape_gqa_expand_kt(&q, gqa_ratio, &q_exp)
                        .context("gdn gqa-expand-recur try_tape_gqa_expand_kt q")?,
                    "GDN recurrent query GQA expansion",
                )?
            } else {
                q_exp
            };
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let k_exp = if tape_recording_active {
                require_active_tape_output(
                    crate::tape_forward::try_tape_gqa_expand_kt(&k, gqa_ratio, &k_exp)
                        .context("gdn gqa-expand-recur try_tape_gqa_expand_kt k")?,
                    "GDN recurrent key GQA expansion",
                )?
            } else {
                k_exp
            };
            Ok((q_exp, k_exp))
        }
    };

    let (attn_out, attn_out_head_last, attn_out_already_gated_norm) = if let Some(attn_out) =
        fused_decode_gates_recurrent_rmsnorm
    {
        (attn_out, true, true) // [B, T, nv, dv], contiguous and gated-normalized
    } else if let Some(attn_out) = fused_decode_gates_recurrent {
        (attn_out, true, false) // [B, T, nv, dv], contiguous
    } else if let Some(attn_out) = fused_prefill_decay_recurrent {
        (attn_out, true, false) // [B, T, nv, dv], contiguous
    } else {
        // --- Step 6: Compute gates ---
        //
        // Two paths: a fused backend kernel (`backend.gdn_gates`) that collapses
        // the sigmoid + softplus + exp + mul chain into one launch, and the
        // candle-op reference path for everything outside the kernel's
        // envelope (unsupported backend, non-bf16, nv > 256, or a disabled
        // backend policy route). The two are algorithmically
        // identical — the reference path is the original Phase-6 implementation
        // and remains the parity oracle.
        let (beta, g) = {
            kiln_nvtx::range!(c"kiln/gdn/gates");
            if gdn_forward_only_fastpaths
                && use_fused_gdn_gates
                && GdnBackend::runtime_supports_gdn_gates(backend)
            {
                if let Some((beta, g)) = GdnBackend::runtime_gdn_gates(
                    backend,
                    &a,
                    &b,
                    &weights.a_log_gates,
                    &weights.dt_bias,
                )
                .context("gdn decode gates fused backend")?
                {
                    (beta, g)
                } else {
                    gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)
                        .context("gdn decode gates fallback after backend miss")?
                }
            } else {
                gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)
                    .context("gdn decode gates fallback")?
            }
        };

        // Phase B11b taps: `gdn_gate_beta` = sigmoid(b), `gdn_gate_g` =
        // -exp(A_log) * softplus(a + dt_bias) (the log-decay scalar fed into the
        // recurrence). Shapes [B, T, nv]. HF mirror: `beta = b.sigmoid()` and
        // `g = -A_log.exp() * F.softplus(a + dt_bias)`.
        let native_recurrent_result = if tape_recording_active {
            None
        } else if qk_norm_deferred_to_native_recurrent {
            let v_recur = v.to_dtype(input_dtype)?;
            match GdnBackend::runtime_gdn_recurrent_qk_norm_prefill_native_head_last(
                backend,
                &q,
                &k,
                &v_recur,
                &beta,
                &g,
                recurrent_state,
                scale,
                1e-6,
            )? {
                Some(attn_out) => Some(attn_out),
                None => {
                    let (q_norm, k_norm) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    let Some(attn_out) = gdn_recurrent_prefill_native_head_last(
                        backend,
                        &q_norm,
                        &k_norm,
                        &v_recur,
                        &beta,
                        &g,
                        recurrent_state,
                    )?
                    else {
                        anyhow::bail!(
                            "backend declined GDN qk-norm recurrent fallback after qk_norm deferral"
                        );
                    };
                    Some(attn_out)
                }
            }
        } else if recurrent_unexpanded_qk {
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

        if let Some(attn_out) = native_recurrent_result {
            (attn_out, true, false) // [B, T, nv, dv], contiguous
        } else {
            let (q, k) = expanded_qk_for_split(q, k)?;

            // Snapshot the recurrent state BEFORE any dispatch below mutates
            // it in place — the GDN tape backward (`GdnRecurrentBackward`)
            // needs the entry state. Cheap clone (a Tensor handle); only the
            // CUDA tape path reads it, but we take it unconditionally so the
            // wiring stays a single no-op call below.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let gdn_entry_state = recurrent_state.clone();

            // Cast recurrence inputs to the state accumulator dtype. Training
            // keeps F32 state, so q/k/v/beta/g stay in F32 through the chunkwise
            // recurrence; bf16 inference state keeps the original bf16 hot path.
            let (q, k, v, beta, g) = {
                kiln_nvtx::range!(c"kiln/gdn/recur_prep");
                let cast_for_recurrence = |src: &Tensor, label: &'static str| -> Result<Tensor> {
                    if src.dtype() == recurrence_dtype {
                        return Ok(src.clone());
                    }
                    let casted = src.to_dtype(recurrence_dtype)?;
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    {
                        if tape_recording_active {
                            require_active_tape_output(
                                crate::tape_forward::try_tape_cast_kt(src, &casted).with_context(
                                    || format!("gdn recur {label} cast try_tape_cast_kt"),
                                )?,
                                &format!("GDN recurrent {label} cast"),
                            )
                        } else {
                            Ok(casted)
                        }
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    {
                        let _ = label;
                        Ok(casted)
                    }
                };
                let q = cast_for_recurrence(&q, "q")?;
                let k = cast_for_recurrence(&k, "k")?;
                let v = cast_for_recurrence(&v, "v")?;
                let beta = cast_for_recurrence(&beta, "beta")?;
                let g = cast_for_recurrence(&g, "g")?;

                // #1082 seam flip: kt-native CastCompositeBackward recorder — no kt->candle->kt.

                // Transpose to [B, nv, T, dim] for per-head processing.
                // #1082 seam flip: kt-native transpose recorder — no kt->candle->kt.
                #[cfg(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                ))]
                let transpose12 = |t: &Tensor| -> Result<Tensor> {
                    if tape_recording_active {
                        require_active_tape_output(
                            crate::tape_forward::try_tape_transpose_kt(t, 1, 2)
                                .context("gdn recur transpose try_tape_transpose_kt")?,
                            "GDN recurrent input transpose",
                        )
                    } else {
                        Ok(t.transpose(1, 2)?)
                    }
                };
                #[cfg(not(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                )))]
                let transpose12 = |t: &Tensor| -> Result<Tensor> { Ok(t.transpose(1, 2)?) };

                let q = transpose12(&q)?; // [B, nv, T, dk]
                let k = transpose12(&k)?; // [B, nv, T, dk]
                let v = transpose12(&v)?; // [B, nv, T, dv]
                let beta = transpose12(&beta)?; // [B, nv, T]
                let g = transpose12(&g)?; // [B, nv, T]
                (q, k, v, beta, g)
            };
            let recurrent_result = if allow_prefill_recurrent_kernel
                && let Some(attn_out) = gdn_recurrent_prefill_head_last(
                    backend,
                    &q,
                    &k,
                    &v,
                    &beta,
                    &g,
                    recurrent_state,
                )? {
                (attn_out, true, false) // [B, T, nv, dv], contiguous
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
                    Some(attn_out) => (attn_out, true, false), // [B, T, nv, dv], contiguous
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
                        false,
                    ), // [B, nv, T, dv]
                }
            };

            // Record a `GdnRecurrentBackward` node for the recurrence output
            // the dispatch above just produced, using the head-FIRST
            // q/k/v/beta/g and the head_last flag (tuple element 1) so the
            // backward can transpose a head-last grad to head-first. No-op
            // unless a thread-local `Tape` is active; never re-runs the
            // recurrence. The production output (`recurrent_result.0`) is
            // untouched. See `crate::tape_forward` module docs.
            //
            // #1082 P4-full: record kt DIRECTLY — the recorder
            // (`tape_record_gdn_recurrent_kt`) is kt-native, so we no longer
            // bridge the 7 saved tensors (out + q/k/v/beta/g + entry_state)
            // kt->candle here (~7 DtoD copies per GDN layer per step, ×24 GDN
            // layers). `recurrent_result.0`'s id is the production recurrence
            // output that flows downstream (the post-transpose's
            // `kt_logits_to_candle` retains it for chaining), so recording it
            // as the node output keeps the recurrence→transpose seam connected.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            if crate::tape_forward::tape_scope_active() {
                let recorded = crate::tape_forward::tape_record_gdn_recurrent_kt(
                    &recurrent_result.0,
                    recurrent_result.1, // head_last
                    &q,
                    &k,
                    &v,
                    &beta,
                    &g,
                    &gdn_entry_state,
                    &q.device(),
                )?;
                anyhow::ensure!(
                    recorded,
                    "active tape scope could not record GDN recurrence"
                );
            }

            recurrent_result
        }
    };

    // Restore state to its original dtype so the caller's F32 invariant holds
    // across layer calls and across prefill/decode steps.
    if state_external_dtype != recurrence_dtype {
        let external_state = recurrent_state.to_dtype(state_external_dtype)?;
        replace_gdn_recurrent_state_handle(
            backend,
            recurrent_state,
            external_state,
            "restore external recurrent dtype",
        )?;
    }

    // Transpose to [B, T, nv, dv] unless the Metal full-chunk path already
    // wrote that contiguous layout directly.
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/post_transpose");
        if attn_out_head_last {
            attn_out
        } else {
            // Phase 6a/CP-4 (#1082) chaining-gap fix: when the recurrence
            // output is head-FIRST (the chunkwise fallback), this transpose to
            // head-LAST mints a fresh candle id. Route it through the kt Tape so
            // the downstream gated-RMSNorm adapter's `tape_kt_input` chains back
            // to the recurrence node (else the tape fragments here and the GDN
            // LoRA grads never flow). Outside a tape scope it falls through to
            // the plain transpose.
            // #1082 seam flip: kt-native transpose recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            {
                if tape_recording_active {
                    require_active_tape_output(
                        crate::tape_forward::try_tape_transpose_kt(&attn_out, 1, 2)
                            .context("gdn attn_out transpose try_tape_transpose_kt")?,
                        "GDN recurrence output transpose",
                    )?
                } else {
                    attn_out.transpose(1, 2)?
                }
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            {
                attn_out.transpose(1, 2)?
            }
        }
    };
    // Phase B11b tap: `gdn_recur_out`. Captured post-transpose (shape
    // [B, T, nv, dv]) so the layout matches the input HF passes to its
    // GatedRMSNorm — i.e. the recurrence output transposed into the
    // "head-last" layout. Capturing here (rather than pre-transpose) lets
    // the HF reference mirror this tensor via a single
    // `norm.register_forward_pre_hook`, which sees exactly the same shape.

    // --- Step 8: Gated RMSNorm — norm(attn_out) * silu(z) ---
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/gated_norm");
        anyhow::ensure!(
            !(tape_recording_active && attn_out_already_gated_norm),
            "active tape scope reached a fused GDN gated norm without a recorder"
        );
        let attn_out = if attn_out_already_gated_norm {
            attn_out
        } else {
            let gated = gated_rms_norm(backend, &attn_out, &z, &weights.norm, config.rms_norm_eps)?;
            // Phase 6a/CP-4 (#1082): record a GdnGatedRmsNormBackward node for
            // the gated-RMSNorm output the production op just produced (BEFORE
            // the reshape below, so x/z/out shapes still match [B,T,nv,dv]).
            // No-op (returns `Ok(None)`) unless a tape scope is active; the
            // production output (`gated`) is untouched either way.
            // #1082 seam flip: kt-native GdnGatedRmsNormBackward recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let gated = if crate::tape_forward::tape_scope_active() {
                require_active_tape_output(
                    crate::tape_forward::try_tape_gdn_gated_rms_norm_kt(
                        &attn_out,
                        &z,
                        &weights.norm,
                        config.rms_norm_eps,
                        &gated,
                    )
                    .context("gdn gated norm try_tape_gdn_gated_rms_norm_kt")?,
                    "GDN gated RMSNorm",
                )?
            } else {
                gated
            };
            gated
        };
        // Reshape to [B, T, v_dim] and cast back to input dtype.
        //
        // CP-4 Increment 3 (#1082): both ops mint fresh candle ids that sit
        // between the wired gated-RMSNorm node and the out_proj keystone. Wrap
        // them on the kt Tape (reshape adjoint = inverse reshape; cast adjoint =
        // dtype round-trip) so the out_proj LoRA grad's `dL/dx` flows back into
        // the gated-RMSNorm node (and thence z + the recurrence chain). No-op +
        // the normal forward fallback outside a tape scope.
        let reshaped = attn_out.reshape((batch, seq_len, v_dim))?;
        // #1082 seam flip: kt-native reshape recorder — no kt->candle->kt.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        let reshaped = if tape_recording_active {
            require_active_tape_output(
                crate::tape_forward::try_tape_reshape_kt(&attn_out, vec![batch, seq_len, v_dim])
                    .context("gdn gated-norm reshape try_tape_reshape_kt")?,
                "GDN gated-norm output reshape",
            )?
        } else {
            reshaped
        };
        // (#1082) GDN-on-Vulkan: a same-dtype `to_dtype` is a kt no-op that still
        // mints a NEW tensor id, and the cast tape recorder skips no-op casts —
        // leaving `casted` an unrecorded island that severs out_proj's dx from the
        // gated-RMSNorm node (and thence the recurrence → in_proj_qkv/z chain). On
        // F32 Vulkan `input_dtype == reshaped.dtype()`, so skip the cast entirely
        // and keep `reshaped` (the recorded tensor) flowing. Only cast when the
        // dtype actually changes (BF16↔F32 boundary), where the recorder fires.
        let casted = if reshaped.dtype() == input_dtype {
            reshaped
        } else {
            let casted = reshaped.to_dtype(input_dtype)?;
            // #1082 seam flip: kt-native CastCompositeBackward recorder — no kt->candle->kt.
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            let casted = if tape_recording_active {
                require_active_tape_output(
                    crate::tape_forward::try_tape_cast_kt(&reshaped, &casted)
                        .context("gdn gated-norm cast try_tape_cast_kt")?,
                    "GDN gated-norm output cast",
                )?
            } else {
                casted
            };
            casted
        };
        #[cfg(feature = "vulkan")]
        let casted = if matches!(x.device(), Device::Vulkan(_)) && casted.device() != x.device() {
            casted.to_device(x.device())?
        } else {
            casted
        };
        casted
    };
    // Phase B11b tap: `gdn_gated_norm`. Output of the GatedRMSNorm /
    // `norm(attn_out) * silu(z)` block, reshaped and cast back to input
    // dtype. Shape [B, T, v_dim]. HF mirror: `core_attn_out` after
    // `self.norm(core_attn_out, z)`.

    // --- Step 9: Output projection ---
    // NOTE: conv1d bias is not loaded by the weight loader. If the model has one,
    // it should be added to GpuLinearAttentionWeights and applied after conv1d.
    // Pre-transposed cache (see Step 1 note).
    let out = {
        kiln_nvtx::range!(c"kiln/gdn/out_proj");
        mlp_proj_forward_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_out,
            &weights.out_proj_t,
            weights.out_proj_marlin.as_ref(),
            gdn_out_lora,
            lora_scale,
        )?
    };
    // Phase B11b tap: `gdn_out_proj`. Output of the final `out_proj` linear
    // (shape [B, T, hidden]) — this is what the caller adds to the residual
    // stream. HF mirror: `self.out_proj(core_attn_out)`.

    Ok(out)
}

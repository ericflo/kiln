use super::*;

/// (#1082 Increment-0 PR2) kt-native sibling of
/// [`standard_forward_backward_tape_authoritative`]: delivers the
/// tape-authoritative LoRA gradients into a kt-native
/// [`kiln_autograd::GradStore`] keyed by [`KtTensorId`], WITHOUT the candle
/// `loss.backward()` GradStore-container hack and WITHOUT the per-grad
/// `kt -> candle` copy.
///
/// The kt grads produced by `with_tape_authoritative_scope` are the
/// authoritative output; they are inserted as-is (the candle `loss` is used
/// only for the `loss_val` scalar readback). The optimizer bridges each grad
/// to candle at its per-Var boundary (`optimizer_step_from_kt_grad_store`,
/// Inc-0 PR3) until the optimizer itself goes kt-native via `kiln-optim`.
///
/// This is the perf-correct grad-delivery path AND the structural gate for the
/// forward.rs type-flip: it removes the dependency on a candle `loss` existing
/// to call `.backward()` on (post-flip `model_forward` returns kt, so there is
/// no candle loss to instantiate a candle `GradStore` from). The grad keys
/// match the PR1 `KtTensorId`-keyed `OptimizerState.moments`
/// (`KtTensorId::from_raw(var.id().as_raw() as u64)` ==
/// `cd_tensor_id_to_kt(var.id())`), so PR3's consumer looks moments up by the
/// same key. (#1082 Inc-0 PR4) NOW WIRED IN: `standard_forward_backward`'s
/// tape-authoritative CUDA branch calls this and returns `GradSource::Kt`, so
/// the SFT loop + the CP-4 gates exercise this kt-native path.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn standard_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, kiln_autograd::GradStore)> {
    let lora_weights = params.as_lora_weights();
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    ensure_sft_loss_route_supports_checkpointing(sft_loss_route, false)?;

    let (loss_val, _loss_kt, grads_by_candle_raw) =
        kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions { detect_anomaly },
            || {
            let loss_kt = match sft_loss_route {
                SftFlceLossRoute::KtTapeFlce => {
                    let normed = model_forward_no_head_with_policy(
                        backend,
                        input_ids,
                        weights,
                        model_config,
                        Some(&mut linear_state),
                        Some(&lora_weights),
                        streaming_prefill,
                    )
                    .context("tape-authoritative(kt) no-head forward")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    // Default SFT records the kt FLCE loss root against final normed hidden
                    // instead of materializing `[1, T, V]` logits. The frozen tied head
                    // receives no gradient; the FLCE tape node returns `dhidden`, keeping
                    // the LoRA path connected through `model_forward_no_head`.
                    let loss = kiln_autograd::with_active_tape(|tape| {
                        kiln_flce_kernel::fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape(
                            &normed,
                            &weights.embed_tokens_t,
                            input_ids,
                            label_mask,
                            DEFAULT_CHUNK_SIZE,
                            tape,
                        )
                    })
                    .ok_or_else(|| {
                        kiln_kt_bridge::BridgeError::new(
                            "tape-authoritative(kt) SFT FLCE: no active kt tape".to_string(),
                        )
                    })?
                    .map_err(|e| {
                        kiln_kt_bridge::BridgeError::new(format!(
                            "tape-authoritative(kt) SFT FLCE kt-tape: {e}"
                        ))
                    })?;
                    loss
                }
                SftFlceLossRoute::VulkanActiveRows => {
                    #[cfg(feature = "vulkan")]
                    {
                        let normed = model_forward_no_head_with_policy(
                            backend,
                            input_ids,
                            weights,
                            model_config,
                            Some(&mut linear_state),
                            Some(&lora_weights),
                            streaming_prefill,
                        )
                        .context("tape-authoritative(kt) no-head Vulkan forward")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                        // Vulkan has its own fused FLCE shaders over active rows
                        // and canonical tied weight [V, H], so route the SFT root
                        // there instead of materializing [1, T, V] logits.
                        crate::sft_tape_shim::try_tape_sft_flce_vulkan_kt(
                            &normed,
                            &weights.embed_tokens,
                            input_ids,
                            label_mask,
                        )
                        .context("tape-authoritative(kt) Vulkan SFT FLCE")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                        .ok_or_else(|| {
                            kiln_kt_bridge::BridgeError::new(
                                "tape-authoritative(kt) Vulkan SFT FLCE returned None".to_string(),
                            )
                        })?
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        return Err(kiln_kt_bridge::BridgeError::new(
                            "backend requested Vulkan SFT FLCE without the vulkan feature"
                                .to_string(),
                        ));
                    }
                }
                SftFlceLossRoute::FullLogits => {
                    let logits = model_forward_kt_with_policy(
                        backend,
                        input_ids,
                        weights,
                        model_config,
                        None,
                        Some(&mut linear_state),
                        Some(&lora_weights),
                        streaming_prefill,
                    )
                    .context("tape-authoritative(kt) fallback full forward")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    kiln_model::tape_forward::try_tape_cross_entropy_from_logits_kt(
                        &logits,
                        input_ids,
                        label_mask,
                    )
                    .context("tape-authoritative(kt) cross_entropy_from_logits_kt")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                    .ok_or_else(|| {
                        kiln_kt_bridge::BridgeError::new(
                            "tape-authoritative(kt) SFT: cross_entropy_from_logits_kt returned None \
                             (kt CE envelope declined — expected [1, T, V] CUDA logits)"
                                .to_string(),
                        )
                    })?
                }
            };
            let loss_val = loss_kt
                .to_scalar::<f32>()
                .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("loss_kt.to_scalar: {e}")))?
                as f64;
            Ok((loss_val, loss_kt))
            },
        )
        .map_err(|e| anyhow::anyhow!("tape-authoritative(kt) backward: {e}"))?;

    // (#1082) Build a kt-native GradStore from the tape grads, keyed by each
    // LoRA `Parameter::tensor_id()`. The tape's `out` map mixes candle-keyed
    // deposits (frozen base/activation/norm tensors via `register_input_mapping`)
    // and kt-param deposits (LoRA leaves via `register_input_mapping_kt`, which
    // namespace-tags the key with `KT_PARAM_DEPOSIT_TAG`). Decode each key: only
    // tagged entries are genuine LoRA-param grads — `decode_kt_param_deposit`
    // strips the tag and yields the param's kt id, so a candle id that happens to
    // equal a param id (independent counters, both start at 1) is rejected. This
    // is the read side of the #1082 collision fix (a frozen RMSNorm `[hidden]`
    // grad was aliasing the `in_proj_z` LoRA-B `[out, rank]` slot → AdamW shape
    // mismatch `[32] != [32, 4]`).
    let mut grads = kiln_autograd::GradStore::new();
    for (key_raw, kt_grad) in grads_by_candle_raw {
        let Some(param_raw) = kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
        else {
            continue;
        };
        grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
    }
    Ok((loss_val, grads))
}

/// Gradient-checkpointed SFT forward/backward via the kt autograd tape (#1082).
///
/// The kt-native replacement for the legacy candle gradient-checkpointing
/// reverse (`checkpointed_forward_backward`), which the forward.rs candle→kt
/// flip grad-severed: candle `.backward()` can no longer trace through the now
/// kt-internal `model_forward_segment` (the kt↔candle copy bridge breaks the
/// autograd lineage). This routes each checkpoint segment's backward through
/// the kt tape instead — the same validated grad producer as the monolithic
/// `standard_forward_backward_tape_authoritative_kt`, just applied per segment
/// so only one segment's activations are resident at a time (the whole point of
/// gradient checkpointing).
///
/// Flow (mirrors `checkpointed_forward_backward` Steps 1-2, replaces Step 3):
///  1. One detached forward → kt boundary activations (one per segment start +
///     the final pre-final-norm hidden). No tape recording (memory-bounded).
///  2. Loss at the final boundary + the analytic tail seed `d(loss)/d(hidden)`
///     through final-RMSNorm + tied LM-head + masked next-token cross-entropy
///     (a candle island; bridged to kt to seed the tape).
///  3. Walk segments in reverse: re-run each segment's forward UNDER A FRESH
///     thread-local tape (recording only that segment), seed the tape backward
///     at the segment output with the upstream grad, read out (a) the LoRA `Var`
///     grads for that segment and (b) the segment-INPUT grad to chain into the
///     previous segment. The fresh-tape-per-segment design bounds memory.
///
/// Returns the LoRA grads as a kt-native `kiln_autograd::GradStore` (keyed by
/// `KtTensorId`), consumed directly by `optimizer_step_from_kt_grad_store` — no
/// candle `loss.backward()` and no kt→candle grad copy.
///
/// The dispatch below uses backend training capabilities plus precision policy
/// for tape eligibility, then keeps local loss-shape exclusions such as ECHO.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn checkpointed_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    segments: &[(usize, usize)],
    device: &Device,
    detect_anomaly: bool,
    checkpoint_boundary_policy: crate::CheckpointBoundaryPolicy,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, kiln_autograd::GradStore)> {
    let num_segments = segments.len();
    anyhow::ensure!(
        num_segments > 0,
        "checkpointed (kt-tape) SFT requires at least one segment"
    );
    anyhow::ensure!(
        input_ids.len() == label_mask.len(),
        "input_ids/label_mask length mismatch: {} vs {}",
        input_ids.len(),
        label_mask.len()
    );
    anyhow::ensure!(
        has_supervised_shifted_labels(label_mask),
        "checkpointed (kt-tape) SFT called with no supervised shifted-label positions"
    );
    ensure_tape_forward_backward_supported("checkpointed SFT", weights, backend)?;
    ensure_sft_loss_route_supports_checkpointing(sft_loss_route, true)?;

    let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
    let lora_detached = lora_weights_detached(params);
    let lora_weights = params.as_lora_weights();

    // Step 1: detached forward → kt boundary activations (one per segment start
    // + the final pre-final-norm hidden). NOT under a tape scope, so nothing is
    // recorded — only the boundary tensors are kept (the checkpointing memory
    // profile). A single threaded `LinearAttentionState` is fine: each GDN
    // layer's recurrence is internal to its own full-sequence pass.
    let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
    let spool_boundaries = if checkpoint_boundary_policy.recompute_for(input_ids.len()) {
        let resident_device_storage =
            ResidencyBackend::runtime_supports_resident_activation(backend);
        let anchor_stride = checkpoint_boundary_policy.anchor_stride_for_shape(
            input_ids.len(),
            num_segments,
            model_config.hidden_size,
            dtype_size_bytes(embed_hidden.dtype()),
        );
        Some(StoredCheckpointBoundaries::new(
            num_segments,
            resident_device_storage,
            anchor_stride,
        ))
    } else {
        None
    };
    let mut boundaries: Vec<Option<kiln_tensor::Tensor>> = Vec::with_capacity(num_segments + 1);
    let mut boundary_dtypes: Vec<DType> = Vec::with_capacity(num_segments + 1);
    let mut current = embed_hidden.detach();
    synchronize_training_tensor_ready("embed_hidden", &current)?;
    boundary_dtypes.push(current.dtype());
    if let Some(spool) = spool_boundaries.as_ref() {
        spool.save(0, &current)?;
        boundaries.push(None);
    } else {
        boundaries.push(Some(current.clone()));
    }
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        for (seg_idx, &(start, end)) in segments.iter().enumerate() {
            current = model_forward_segment_with_policy(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_detached),
                streaming_prefill,
            )?
            .detach();
            let boundary_label = format!("boundary_segment[{seg_idx}] layers {start}..{end}");
            synchronize_training_tensor_ready(&boundary_label, &current)?;
            boundary_dtypes.push(current.dtype());
            if let Some(spool) = spool_boundaries.as_ref() {
                spool.save(seg_idx + 1, &current)?;
                boundaries.push(None);
            } else {
                boundaries.push(Some(current.clone()));
            }
        }
    }
    let final_hidden_kt = current.clone();

    // Step 2: real loss at the final boundary + the exact FLCE/RMSNorm tail
    // seed. When the CUDA FLCE loss-value path computes final-norm output, the
    // analytic tail reuses that `normed` hidden for the FLCE backward seed
    // instead of recomputing the same [1, T, H] RMSNorm. The tail then applies
    // the shared final-RMSNorm backward to return d(loss)/d(pre-final-norm
    // hidden) as kt [1, T, H] (BF16 on the fused GPU path, F32 on the composite
    // fallback) — exactly the upstream grad to seed the LAST segment's backward
    // (its output IS that hidden). The loss value here is ONLY consumed as a
    // scalar `loss_val`; the gradient comes entirely from this tail seed,
    // outside the per-segment tape scopes.
    let mut normed_for_tail = None;
    let mut flce_active_metadata_for_tail = None;
    let tail_grad_override: Option<Tensor>;
    let loss_val = match sft_loss_route {
        SftFlceLossRoute::KtTapeFlce => {
            tail_grad_override = None;
            // (#1082 H-FLCE / candle-drop) FLCE loss-VALUE via the kt-native forward
            // `fused_linear_cross_entropy_phase_b_kt` — taking the kt `normed` hidden
            // and the kt `embed_tokens_t` head DIRECTLY (no candle `cd_out` copy, no
            // ~780MB/step `embed_tokens_t` kt->candle copy, no candle device bridge).
            // Only the resulting scalar crosses back to host. The same `normed`
            // tensor is retained for the FLCE/RMSNorm tail seed below.
            // The candle FLCE provider opt-in (`KILN_CUDA_FLCE`) was removed in the
            // candle drop — this is now the sole FLCE path.
            synchronize_training_tensor_ready("tail_pre_final_norm_hidden", &final_hidden_kt)?;
            let normed = model_forward_final_norm(&final_hidden_kt, weights, model_config)?;
            synchronize_training_tensor_ready("tail_final_norm", &normed)?;
            let (loss_kt, active_metadata) =
                kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_with_metadata_kt(
                    &normed,
                    &weights.embed_tokens_t,
                    input_ids,
                    label_mask,
                    DEFAULT_CHUNK_SIZE,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ckpt-kt kt-native fused linear cross-entropy (final boundary): {e}"
                    )
                })?;
            synchronize_training_tensor_ready("tail_flce_loss_scalar", &loss_kt)?;
            let loss_val = loss_kt.to_scalar::<f32>()? as f64;
            flce_active_metadata_for_tail = active_metadata;
            normed_for_tail = Some(normed);
            loss_val
        }
        SftFlceLossRoute::VulkanActiveRows => {
            #[cfg(feature = "vulkan")]
            {
                synchronize_training_tensor_ready("tail_pre_final_norm_hidden", &final_hidden_kt)?;
                let normed = model_forward_final_norm(&final_hidden_kt, weights, model_config)?;
                synchronize_training_tensor_ready("tail_final_norm", &normed)?;
                let (loss_kt, grad_normed) =
                    crate::sft_tape_shim::vulkan_sft_flce_loss_and_grad_kt(
                        &normed,
                        &weights.embed_tokens,
                        input_ids,
                        label_mask,
                    )
                    .map_err(|e| anyhow::anyhow!("ckpt-kt Vulkan fused SFT FLCE tail: {e}"))?;
                synchronize_training_tensor_ready("tail_vulkan_flce_loss_scalar", &loss_kt)?;
                let loss_val = loss_kt.to_scalar::<f32>()? as f64;
                tail_grad_override = Some(
                    rms_norm_backward_pre_final_norm(
                        final_rmsnorm_backward_route_for_backend(backend),
                        &final_hidden_kt,
                        &weights.final_norm,
                        &grad_normed,
                        model_config.rms_norm_eps,
                    )
                    .context("ckpt-kt Vulkan final RMSNorm backward")?,
                );
                loss_val
            }
            #[cfg(not(feature = "vulkan"))]
            {
                anyhow::bail!("backend requested Vulkan SFT FLCE without the vulkan feature");
            }
        }
        SftFlceLossRoute::FullLogits => {
            anyhow::bail!(
                "checkpointed SFT reached unsupported loss route `{}` after its entry guard",
                sft_loss_route.as_str()
            )
        }
    };
    anyhow::ensure!(
        loss_val.is_finite(),
        "SFT loss became non-finite before backward: loss={loss_val} route={} seq_len={} segments={}",
        sft_loss_route.as_str(),
        input_ids.len(),
        num_segments
    );
    let tail_grad = if let Some(tail_grad) = tail_grad_override {
        Ok(tail_grad)
    } else {
        match normed_for_tail.as_ref() {
            Some(normed) => analytic_sft_tail_grad_from_normed_pre_final_norm_with_flce_metadata(
                final_rmsnorm_backward_route_for_backend(backend),
                &final_hidden_kt,
                normed,
                &weights.final_norm,
                &weights.embed_tokens_t,
                input_ids,
                label_mask,
                model_config.rms_norm_eps,
                DEFAULT_CHUNK_SIZE,
                flce_active_metadata_for_tail.as_ref(),
            ),
            None => analytic_sft_tail_grad_pre_final_norm(
                final_rmsnorm_backward_route_for_backend(backend),
                &final_hidden_kt,
                &weights.final_norm,
                &weights.embed_tokens_t,
                input_ids,
                label_mask,
                model_config.rms_norm_eps,
                DEFAULT_CHUNK_SIZE,
            ),
        }
    };
    let mut upstream_grad = tail_grad
        .context("ckpt-kt FLCE/RMSNorm SFT tail gradient")?
        .detach();
    drop(final_hidden_kt);
    drop(current);
    // Step 3: reverse pass over segments via the kt tape. Each segment is
    // re-run under its OWN fresh tape (memory bounded to one segment), seeded at
    // its output with the upstream grad; we read the LoRA Var grads and the
    // segment-input grad (to chain) out of the walk.
    // (#1082) keyed by `Parameter::tensor_id()`.
    let mut grads = kiln_autograd::GradStore::new();
    for seg_idx in (0..num_segments).rev() {
        let (start, end) = segments[seg_idx];
        let seg_input = if let Some(spool) = spool_boundaries.as_ref() {
            load_or_recompute_checkpoint_boundary(
                spool,
                seg_idx,
                backend,
                weights,
                model_config,
                &positions,
                segments,
                &lora_detached,
                device,
                streaming_prefill,
            )?
        } else {
            boundaries[seg_idx]
                .as_ref()
                .context("ckpt-kt: missing in-memory checkpoint boundary")?
                .clone()
        };
        let seg_input_id = seg_input.id();
        // Match the seed dtype to the segment output (the model hidden dtype);
        // the analytic tail is F32 and chained grads may differ.
        let seg_output_dtype = boundary_dtypes[seg_idx + 1];
        let seed = upstream_grad
            .to_dtype(seg_output_dtype)
            .map_err(|e| anyhow::anyhow!("ckpt-kt: seed dtype cast (segment {seg_idx}): {e}"))?;
        let positions_ref = &positions;
        let lora_ref = &lora_weights;
        let (kt_grads, candle_grads) =
            kiln_kt_bridge::tape_bridge::with_tape_segment_backward_scope(
                kiln_autograd::TapeOptions { detect_anomaly },
                seed,
                || {
                    // Fresh recurrence state per segment (GDN recurrence is internal
                    // to each layer's full-sequence pass — see Step 1 note).
                    let mut seg_ls = LinearAttentionState::new(model_config, device)
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    model_forward_segment_with_policy(
                        backend,
                        seg_input,
                        weights,
                        model_config,
                        positions_ref,
                        start,
                        end,
                        Some(&mut seg_ls),
                        Some(lora_ref),
                        streaming_prefill,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))
                },
            )
            .map_err(|e| anyhow::anyhow!("ckpt-kt: segment {seg_idx} tape backward: {e}"))?;

        // Decode every tagged parameter deposit into a segment-local store.
        // The exact segment contract below rejects missing leaves, deposits for
        // another layer range, and any unknown tagged parameter before merge.
        let mut segment_grads = kiln_autograd::GradStore::new();
        for (candle_raw, g) in candle_grads {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
            else {
                continue;
            };
            segment_grads.insert(KtTensorId::from_raw(param_raw), g);
        }
        let grad_context =
            format!("checkpointed SFT segment {seg_idx} layers {start}..{end} gradient contract");
        merge_checkpoint_lora_grad_segment(
            params,
            &mut grads,
            segment_grads,
            start,
            end,
            &grad_context,
        )?;

        // Chain the upstream grad into the previous (earlier) segment.
        if seg_idx > 0 {
            upstream_grad = kt_grads.get(seg_input_id).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "ckpt-kt: tape backward produced no input gradient for segment {seg_idx}"
                )
            })?;
        }
    }

    Ok((loss_val, grads))
}

pub fn standard_forward_backward(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
) -> Result<(f64, GradSource)> {
    standard_forward_backward_with_policy(
        backend,
        input_ids,
        weights,
        model_config,
        params,
        label_mask,
        device,
        StreamingPrefillExecutionPolicy::for_device(*device),
    )
}

/// Explicit-policy variant of [`standard_forward_backward`].
#[allow(clippy::too_many_arguments)]
pub fn standard_forward_backward_with_policy(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, GradSource)> {
    standard_forward_backward_with_policy_and_loss_route(
        backend,
        TrainingLossBackend::runtime_sft_flce_loss_route(backend),
        input_ids,
        weights,
        model_config,
        params,
        label_mask,
        device,
        false,
        streaming_prefill,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn standard_forward_backward_with_policy_and_loss_route(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, GradSource)> {
    // (#1082 candle-drop) The SFT forward/backward is now UNCONDITIONALLY
    // kt tape-authoritative when the backend capability and precision policy
    // allow it. The candle producers
    // (`standard_forward_backward_tape_authoritative` F32-hack,
    // `standard_forward_backward_via_tape_bridge`, the inline candle
    // `loss.backward()` path) are all DELETED. Unsupported backend/dtype
    // combinations fail before the kt step is attempted.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        ensure_tape_forward_backward_supported("standard_forward_backward", weights, backend)?;
        let (loss_val, kt_grads) = standard_forward_backward_tape_authoritative_kt(
            backend,
            sft_loss_route,
            input_ids,
            weights,
            model_config,
            params,
            label_mask,
            device,
            detect_anomaly,
            streaming_prefill,
        )?;
        Ok((loss_val, GradSource::Kt(kt_grads)))
    }
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    {
        let _ = (
            backend,
            sft_loss_route,
            input_ids,
            weights,
            model_config,
            params,
            label_mask,
            device,
            detect_anomaly,
            streaming_prefill,
        );
        anyhow::bail!(
            "standard_forward_backward: SFT training requires a GPU backend feature \
             post candle-drop because the candle `loss.backward()` path was removed."
        )
    }
}

/// kt-native sibling of [`grpo_step_forward_backward_tape_authoritative`]
/// (#1082 Inc-0 PR5). Identical GRPO policy-gradient tape-authoritative
/// forward/backward, but delivers the LoRA grads in a kt-native
/// [`kiln_autograd::GradStore`] (keyed by [`KtTensorId`]) DIRECTLY from the
/// tape — NO candle `loss.backward()` GradStore-container hack and NO per-grad
/// `kt -> candle` copy. This is the exact GRPO analogue of the SFT kt producer
/// [`standard_forward_backward_tape_authoritative_kt`].
///
/// As with SFT, this is the perf-correct grad-delivery path AND the structural
/// gate for the forward.rs type-flip: it removes the dependency on a candle
/// `loss` existing to call `.backward()` on (post-flip `model_forward` returns
/// kt, so there is no candle loss to seed a candle `GradStore` from). The grad
/// keys match the PR1 `KtTensorId`-keyed `OptimizerState.moments`
/// (`KtTensorId::from_raw(var.id().as_raw() as u64)` ==
/// `cd_tensor_id_to_kt(var.id())`), so the kt consumers
/// ([`optimizer_step_from_kt_grad_store`],
/// [`observe_lora_grad_norms_from_kt_grad_store`], and the kt accumulate path)
/// look grads/moments up by the same key.
///
/// The backend-selected loss route must record inside the active tape scope;
/// there is no environment opt-out or alternate autograd producer. ECHO, when
/// configured, is composed into that same tape-rooted loss.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn grpo_step_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    action_mask: &[bool],
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    device: &Device,
    comp_idx: usize,
    num_active: usize,
    comp_env_count: usize,
    streaming_tile_tokens: usize,
    checkpoint_segments: usize,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    no_pg: bool,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(
    f64,
    Option<f64>,
    kiln_autograd::GradStore,
    kiln_tensor::Tensor,
)> {
    let lora_weights = params.as_lora_weights();
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    let step_started = Instant::now();

    let ((loss_val, env_ce, policy_log_probs), _loss_kt, grads_by_candle_raw) =
        kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions { detect_anomaly },
            || {
                // Single policy forward through final RMSNorm, without materializing
                // `[1, T, V]` logits. The GRPO loss root chunks the frozen tied head
                // internally and records `dL/d(normed_hidden)` directly.
                let policy_hidden = model_forward_no_head_with_policy(
                    backend,
                    input_ids,
                    weights,
                    model_config,
                    Some(&mut linear_state),
                    Some(&lora_weights),
                    streaming_prefill,
                )
                .context("GRPO tape-authoritative(kt) no-head policy forward")
                .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;

                // The Vulkan fused active-rows root carries no env rows — an
                // ECHO-active step takes the KtComposite root below instead.
                #[cfg(feature = "vulkan")]
                let mut loss_opt = match TrainingLossBackend::runtime_grpo_loss_route(backend) {
                    GrpoLossRoute::VulkanActiveRows if echo_env.is_none() => {
                        crate::grpo_tape_shim::try_tape_grpo_pg_loss_from_normed_hidden_vulkan_kt(
                            &policy_hidden,
                            &weights.embed_tokens,
                            input_ids,
                            action_mask,
                            behavior_log_probs,
                            kl_reference_log_probs,
                            loss_params,
                        )
                        .context("GRPO tape-authoritative(kt) Vulkan fused scalar loss")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                        .map(|(loss, policy_log_probs)| (loss, None, policy_log_probs))
                    }
                    _ => None,
                };
                #[cfg(not(feature = "vulkan"))]
                let mut loss_opt = None;
                if loss_opt.is_none() {
                    loss_opt = crate::grpo_tape_shim::try_tape_grpo_pg_loss_from_normed_hidden_kt(
                        &policy_hidden,
                        &weights.embed_tokens_t,
                        input_ids,
                        action_mask,
                        behavior_log_probs,
                        kl_reference_log_probs,
                        loss_params,
                        grpo_kl_auxiliary_route_for_backend(backend),
                        device,
                        DEFAULT_CHUNK_SIZE,
                        echo_env,
                        no_pg,
                    )
                    .context("GRPO tape-authoritative(kt) scalar loss")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                }

                let (loss, env_ce, policy_log_probs) = match loss_opt {
                    Some(values) => values,
                    None => {
                        return Err(kiln_kt_bridge::BridgeError::new(
                            "GRPO tape-authoritative(kt): the selected loss route did not record a \
                         scalar root (an active tape scope is mandatory; the active set may be \
                         empty or the hidden/head tensors may be outside the route envelope)",
                        ));
                    }
                };
                let loss_val = loss.to_scalar::<f32>().map_err(|e| {
                    kiln_kt_bridge::BridgeError::new(format!("GRPO(kt) loss.to_scalar: {e}"))
                })? as f64;
                Ok(((loss_val, env_ce, policy_log_probs), loss))
            },
        )
        .map_err(|e| anyhow::anyhow!("GRPO tape-authoritative(kt) backward: {e}"))?;

    // Build a kt-native GradStore DIRECTLY from the tape grads. No
    // `loss.backward()` container hack (`GradStore::new()` on kiln_autograd is
    // public, unlike candle's) and no `kt -> candle` grad copy: the kt grads are
    // inserted as-is, keyed by each LoRA Var's id bridged into the kt id space
    // (matching the PR1 `KtTensorId`-keyed moments). Identical shape to the SFT
    // kt producer.
    // (#1082) keyed by `Parameter::tensor_id()` (== the LoRA primary kt
    // tensor id the tape adapter registered as the candle-input key).
    let mut grads = kiln_autograd::GradStore::new();
    for (key_raw, kt_grad) in grads_by_candle_raw {
        let Some(param_raw) = kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
        else {
            continue;
        };
        grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
    }

    let step_elapsed = step_started.elapsed();
    if let Some(t) = timings.as_deref_mut() {
        // The tape walk owns the backward internally so we can't break the step
        // into policy_forward / backward here; bucket the full wall-clock against
        // the backward timer so the GRPO benchmark accounting still totals
        // correctly when this path is exercised.
        t.add_backward(step_elapsed);
    }
    tracing::info!(
        comp_idx,
        seq_len = input_ids.len(),
        action_tokens = num_active,
        env_tokens = comp_env_count,
        checkpoint_segments,
        streaming_prefill = streaming_prefill.enabled_for(input_ids.len()),
        streaming_tile_tokens,
        elapsed_ms = step_elapsed.as_millis() as u64,
        "GRPO step end (tape-authoritative kt)"
    );

    Ok((loss_val, env_ce, grads, policy_log_probs))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn checkpointed_grpo_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    action_mask: &[bool],
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    segments: &[(usize, usize)],
    device: &Device,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    no_pg: bool,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(
    f64,
    Option<f64>,
    kiln_autograd::GradStore,
    kiln_tensor::Tensor,
)> {
    let num_segments = segments.len();
    anyhow::ensure!(
        num_segments > 0,
        "checkpointed GRPO requires at least one segment"
    );
    anyhow::ensure!(
        input_ids.len() == action_mask.len(),
        "input_ids/action_mask length mismatch: {} vs {}",
        input_ids.len(),
        action_mask.len()
    );
    anyhow::ensure!(
        action_mask.get(1..).is_some_and(|m| m.iter().any(|&v| v)),
        "checkpointed GRPO called with no active shifted action positions"
    );

    let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
    let lora_detached = lora_weights_detached(params);
    let lora_weights = params.as_lora_weights();

    let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
    let mut boundaries: Vec<Tensor> =
        Vec::with_capacity(crate::retained_checkpoint_boundary_count(num_segments));
    let mut current = embed_hidden.detach();
    boundaries.push(current.clone());
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        for &(start, end) in segments {
            current = model_forward_segment_with_policy(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_detached),
                streaming_prefill,
            )?
            .detach();
            boundaries.push(current.clone());
        }
    }
    let final_hidden = boundaries
        .last()
        .context("checkpointed GRPO: missing final checkpoint boundary")?
        .clone();

    let normed = model_forward_final_norm(&final_hidden, weights, model_config)
        .context("checkpointed GRPO final norm")?;
    // The Vulkan fused active-rows tail carries no env rows — ECHO-active
    // steps take the KtComposite tail instead.
    #[cfg(feature = "vulkan")]
    let fused_vulkan_tail = if echo_env.is_some() {
        None
    } else {
        match TrainingLossBackend::runtime_grpo_loss_route(backend) {
            GrpoLossRoute::VulkanActiveRows => {
                crate::grpo_tape_shim::vulkan_grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
                    &normed,
                    &weights.embed_tokens,
                    input_ids,
                    action_mask,
                    behavior_log_probs,
                    kl_reference_log_probs,
                    loss_params,
                )
                .context("checkpointed GRPO Vulkan fused tail loss/gradient")?
            }
            GrpoLossRoute::KtComposite => None,
        }
    };
    #[cfg(not(feature = "vulkan"))]
    let fused_vulkan_tail = None;
    let (loss_kt, grad_normed, env_ce, policy_log_probs) = match fused_vulkan_tail {
        Some((loss, grad, policy_log_probs)) => (loss, grad, None, policy_log_probs),
        None => crate::grpo_tape_shim::grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
            &normed,
            &weights.embed_tokens_t,
            input_ids,
            action_mask,
            behavior_log_probs,
            kl_reference_log_probs,
            loss_params,
            grpo_kl_auxiliary_route_for_backend(backend),
            1.0,
            device,
            DEFAULT_CHUNK_SIZE,
            echo_env,
            no_pg,
        )
        .context("checkpointed GRPO tail loss/gradient")?,
    };
    let loss_val = loss_kt.to_scalar::<f32>()? as f64;
    let mut upstream_grad = rms_norm_backward_pre_final_norm(
        final_rmsnorm_backward_route_for_backend(backend),
        &final_hidden,
        &weights.final_norm,
        &grad_normed,
        model_config.rms_norm_eps,
    )
    .context("checkpointed GRPO final RMSNorm backward")?
    .detach();

    let mut grads = kiln_autograd::GradStore::new();
    for seg_idx in (0..num_segments).rev() {
        let (start, end) = segments[seg_idx];
        let seg_input = boundaries[seg_idx].clone();
        let seg_input_id = seg_input.id();
        let seg_output_dtype = boundaries[seg_idx + 1].dtype();
        let seed = upstream_grad.to_dtype(seg_output_dtype).map_err(|e| {
            anyhow::anyhow!("checkpointed GRPO: seed dtype cast (segment {seg_idx}): {e}")
        })?;
        let positions_ref = &positions;
        let lora_ref = &lora_weights;
        let (kt_grads, candle_grads) =
            kiln_kt_bridge::tape_bridge::with_tape_segment_backward_scope(
                kiln_autograd::TapeOptions { detect_anomaly },
                seed,
                || {
                    let mut seg_ls = LinearAttentionState::new(model_config, device)
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    model_forward_segment_with_policy(
                        backend,
                        seg_input,
                        weights,
                        model_config,
                        positions_ref,
                        start,
                        end,
                        Some(&mut seg_ls),
                        Some(lora_ref),
                        streaming_prefill,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))
                },
            )
            .map_err(|e| {
                anyhow::anyhow!("checkpointed GRPO: segment {seg_idx} tape backward: {e}")
            })?;

        let mut segment_grads = kiln_autograd::GradStore::new();
        for (candle_raw, g) in candle_grads {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
            else {
                continue;
            };
            segment_grads.insert(KtTensorId::from_raw(param_raw), g);
        }
        let grad_context =
            format!("checkpointed GRPO segment {seg_idx} layers {start}..{end} gradient contract");
        merge_checkpoint_lora_grad_segment(
            params,
            &mut grads,
            segment_grads,
            start,
            end,
            &grad_context,
        )?;

        if seg_idx > 0 {
            upstream_grad = kt_grads.get(seg_input_id).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "checkpointed GRPO: tape backward produced no input gradient for segment {seg_idx}"
                )
            })?;
        }
    }

    Ok((loss_val, env_ce, grads, policy_log_probs))
}

/// Bundled parameters for the GRPO surrogate / KL loss.
///
/// `loss_normalizer` is the scalar applied to the *sum* of per-token loss
/// contributions before backward. For `LossAggregation::PerSample` it is
/// `1 / num_active_tokens` for the current completion (recovering the
/// historical kiln per-completion mean). For `LossAggregation::TokenLevel`
/// it is `1 / group_total_active_tokens` so the per-token contributions sum
/// across the entire group to a DAPO-style token-level mean.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GrpoLossParams {
    pub advantage: f64,
    pub clip_low: f64,
    /// Additive PPO upper epsilon for token/sequence GRPO; absolute upper
    /// importance-weight cap for CISPO.
    pub clip_high: f64,
    pub kl_coeff: f64,
    pub kl_estimator: KlEstimator,
    pub loss_normalizer: f64,
    /// Importance-sampling level (Phase 2). `Token` is the historical
    /// per-token PPO surrogate; `Sequence` computes the IS ratio at the
    /// sequence level (GSPO, arXiv:2507.18071); `Cispo` clips the IS
    /// weight rather than the surrogate (arXiv:2506.13585).
    pub is_level: IsLevel,
    /// When true, the IS ratio is forced to 1.0 and the surrogate reduces to
    /// `advantage` per token. KL selection remains independent.
    pub reinforce: bool,
    /// Phase 3c — entropy-aware KL quantile. `None` = full-token KL; when
    /// `Some(q)`, tokens whose `-policy_log_prob` is below the q-quantile
    /// across this loss-instance's active tokens get zero KL contribution
    /// (and zero KL gradient). Approximates the Cui et al. selective-KL
    /// idea (arXiv:2506.01939).
    pub entropy_aware_kl_quantile: Option<f32>,
}

impl GrpoLossParams {
    pub(super) fn from_config(config: &GrpoConfig, advantage: f64, loss_normalizer: f64) -> Self {
        let (clip_low, ppo_clip_high) = config.clip_bounds();
        let clip_high = if matches!(config.is_level, IsLevel::Cispo) {
            config.cispo_max_weight
        } else {
            ppo_clip_high
        };
        let reinforce = matches!(
            config.behavior_policy,
            BehaviorPolicy::NoImportanceCorrection
        );
        let kl_estimator = config.kl_estimator;
        // Entropy-aware KL only makes sense when KL is actually being
        // applied; gate it off otherwise so the quantile compute doesn't
        // run for nothing.
        let entropy_aware_kl_quantile = if matches!(kl_estimator, KlEstimator::None) {
            None
        } else {
            config.entropy_aware_kl_quantile
        };
        Self {
            advantage,
            clip_low,
            clip_high,
            kl_coeff: config.kl_coeff,
            kl_estimator,
            loss_normalizer,
            is_level: config.is_level,
            reinforce,
            entropy_aware_kl_quantile,
        }
    }
}

pub(super) fn entropy_aware_kl_threshold_from_policy_log_probs(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    q: f32,
    num_active: usize,
) -> Result<f32> {
    anyhow::ensure!(
        num_active > 0,
        "entropy-aware KL threshold requires at least one active token"
    );
    let idx = ((q as f64) * (num_active.saturating_sub(1)) as f64).round() as usize;
    let idx = idx.min(num_active.saturating_sub(1));

    // Reuse backend top-k kernels when the requested quantile rank is small.
    // The kernels are intentionally k-pass, so for large ranks a single host
    // threshold read + CPU sort is less work than asking for most of the vector.
    if idx < 1024
        && matches!(
            grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath
        )
    {
        let flat = policy_log_probs
            .to_f32_dtype()?
            .flatten_all()?
            .reshape(vec![num_active])?
            .contiguous()?;
        match try_topk_on_device(&flat, idx + 1) {
            Ok(pairs) => {
                let threshold = pairs.get(idx).map(|(_, value)| *value).ok_or_else(|| {
                    anyhow::anyhow!("entropy-aware KL top-k returned too few values")
                })?;
                return Ok(threshold);
            }
            Err(err) => {
                tracing::debug!(
                    error = %err,
                    "entropy-aware KL device top-k declined; falling back to host threshold sort"
                );
            }
        }
    }

    let plp_host: Vec<f32> = policy_log_probs
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()?;
    anyhow::ensure!(
        plp_host.len() == num_active,
        "entropy-aware KL threshold plp len {} != num_active {num_active}",
        plp_host.len()
    );
    let mut neg = plp_host.iter().map(|p| -(*p as f64)).collect::<Vec<_>>();
    neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thr = neg[idx];
    Ok((-thr) as f32)
}

pub(crate) fn entropy_aware_kl_mask_kt(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(q) = params.entropy_aware_kl_quantile else {
        return Ok(None);
    };
    if !(q.is_finite() && (0.0..1.0).contains(&q)) {
        return Ok(None);
    }

    let num_active = policy_log_probs.elem_count();
    if num_active == 0 {
        return Ok(Some(zeros_f32_on(policy_log_probs.shape(), device)?));
    }

    let threshold = entropy_aware_kl_threshold_from_policy_log_probs(
        grpo_kl_auxiliary_route,
        policy_log_probs,
        q,
        num_active,
    )?;
    let policy_f32 = policy_log_probs
        .to_f32_dtype()?
        .flatten_all()?
        .reshape(vec![num_active])?
        .contiguous()?;
    let threshold_tensor = policy_f32.affine(0.0, threshold as f64)?.contiguous()?;
    let keep_kl = kiln_tensor::ops::le(&policy_f32, &threshold_tensor)?;
    let ones = policy_f32.affine(0.0, 1.0)?.contiguous()?;
    let zeros = policy_f32.affine(0.0, 0.0)?.contiguous()?;
    let mask = keep_kl.where_cond(&ones, &zeros)?;
    mask.reshape(policy_log_probs.dims().to_vec())?
        .contiguous()
        .map(Some)
        .map_err(Into::into)
}

/// Compute the GRPO loss from policy, behavior-policy, and KL-reference
/// log-probs.
///
/// Returns a scalar loss tensor suitable for backward(). The scalar is
/// `params.loss_normalizer * sum_over_active_tokens(per_token_loss)`.
///
/// The structure of `per_token_loss` depends on `params.is_level`:
///   * `IsLevel::Token` — historical per-token PPO `min(r·A, clip(r)·A)`.
///   * `IsLevel::Sequence` — GSPO sequence-level scalar ratio
///     `s = exp(mean(log_ratio))`, then `min(s·A, clip(s)·A)` broadcast to
///     every active token before the configured loss aggregation.
///   * `IsLevel::Cispo` — CISPO weight clipping: the per-token gradient
///     factor `stop_grad(min(r, cispo_max_weight))·A` multiplies `log π_θ`,
///     so every token contributes a gradient without a lower weight floor.
// `pub(crate)` so the GRPO tape-authoritative loss-root shim
// (`crate::grpo_tape_shim`) can recompute the EXACT same scalar PG (+ KL)
// loss inside its candle-autograd backward composite (#1082 CP-4).
pub(crate) fn grpo_loss(
    policy_log_probs: &Tensor,
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Tensor> {
    grpo_loss_with_kl_auxiliary_route(
        GrpoKlAuxiliaryRoute::HostComposite,
        policy_log_probs,
        behavior_log_probs,
        kl_reference_log_probs,
        params,
        device,
    )
}

pub(crate) fn grpo_loss_with_kl_auxiliary_route(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Tensor> {
    let num_active = policy_log_probs.elem_count();
    if num_active == 0 {
        // Scalar zero loss (empty active set). kt-native.
        return zeros_f32_on((), device).map_err(Into::into);
    }

    anyhow::ensure!(
        behavior_log_probs.elem_count() == num_active,
        "GRPO behavior log-probability count {} did not match policy count {num_active}",
        behavior_log_probs.elem_count()
    );
    anyhow::ensure!(
        kl_reference_log_probs.elem_count() == num_active,
        "GRPO KL-reference log-probability count {} did not match policy count {num_active}",
        kl_reference_log_probs.elem_count()
    );

    // `reinforce` is the explicit no-importance-correction mode. Its ratio is
    // one by value while retaining the policy gradient; the independently
    // configured KL term below still uses `kl_reference_log_probs`.
    let importance_log_ratio = if params.reinforce {
        (policy_log_probs - policy_log_probs.detach())?
    } else {
        (policy_log_probs - behavior_log_probs)?
    };
    let ratio = importance_log_ratio.exp()?;
    let ratio_shape = ratio.dims().to_vec();
    let kl_log_ratio = (policy_log_probs - kl_reference_log_probs)?;

    // Asymmetric PPO clip range: [1 - clip_low, 1 + clip_high]. CISPO
    // interprets clip_high separately below as its absolute upper weight cap.
    let lo_val = 1.0 - params.clip_low;
    let hi_val = 1.0 + params.clip_high;

    // Per-token KL term selected by KlEstimator (shared across IS levels).
    let kl_penalty_raw = match params.kl_estimator {
        KlEstimator::None => zeros_f32_on(ratio.shape(), device)?,
        KlEstimator::K1 => kl_log_ratio.affine(params.kl_coeff, 0.0)?,
        KlEstimator::K3 => {
            let neg_log_ratio = kl_log_ratio.neg()?;
            let term = (neg_log_ratio.exp()?.affine(1.0, -1.0)? + &kl_log_ratio)?;
            term.affine(params.kl_coeff, 0.0)?
        }
    };
    // Phase 3c — selective KL gating: zero KL on tokens below the proxy-entropy threshold.
    let kl_penalty = if let Some(q) = params.entropy_aware_kl_quantile {
        if q.is_finite() && (0.0..1.0).contains(&q) {
            let mask = entropy_aware_kl_mask_kt(
                grpo_kl_auxiliary_route,
                policy_log_probs,
                params,
                device,
            )?
            .ok_or_else(|| anyhow::anyhow!("entropy-aware KL mask unexpectedly absent"))?;
            (&kl_penalty_raw * &mask)?
        } else {
            kl_penalty_raw
        }
    } else {
        kl_penalty_raw
    };

    let neg_surrogate = if params.reinforce {
        ratio.affine(-params.advantage, 0.0)?
    } else {
        match params.is_level {
            IsLevel::Token => {
                // Per-token surrogate: -min(r·A, clip(r)·A).
                // (#1082) kt `clamp` takes scalar bounds directly; advantage folds
                // into `affine` (constant scalar, gradient flows through the ratio).
                let clipped_ratio = ratio.clamp(lo_val, hi_val)?;
                let surr1 = ratio.affine(params.advantage, 0.0)?;
                let surr2 = clipped_ratio.affine(params.advantage, 0.0)?;
                let surrogate = surr1.minimum(&surr2)?;
                surrogate.neg()?
            }
            IsLevel::Sequence => {
                // GSPO: s = exp(mean(log_ratio)), surrogate at sequence level,
                // gradient distributed back equally to every active token.
                //
                // The sequence surrogate is replicated over active tokens;
                // the outer loss normalizer performs the sequence mean.
                let u = importance_log_ratio.mean_keepdim(0)?;
                let s = u.exp()?;
                // (#1082) kt scalar clamp + scalar `affine` for the constant
                // advantage (gradient flows through `s`).
                let clipped = s.clamp(lo_val, hi_val)?;
                let surr1 = s.affine(params.advantage, 0.0)?;
                let surr2 = clipped.affine(params.advantage, 0.0)?;
                let surrogate = surr1.minimum(&surr2)?;
                // Repeat the sequence loss across its active token positions.
                // The outer per-sample normalizer divides the sum by
                // num_active, matching TRL/GSPO. The derivative of the shared
                // sequence ratio already contributes its own 1/num_active.
                let neg = surrogate.neg()?;
                neg.broadcast_as(&ratio_shape)?
            }
            IsLevel::Cispo => {
                // CISPO: gradient through `log π_θ` only; the IS weight is the
                // *clipped* ratio with stop-gradient. The total loss contribution
                // is `-stop_grad(clip(r)) · A · log π_θ` per token.
                // (#1082) kt scalar clamp; advantage folds into `affine`. `weight`
                // is detached either way, so the constant scalar mul is exact.
                let clipped_ratio = ratio.clamp(0.0, params.clip_high)?.detach();
                // log π_θ = policy_log_probs (already in tensor form).
                let weight = clipped_ratio.affine(params.advantage, 0.0)?.detach();
                (&weight * policy_log_probs)?.neg()?
            }
        }
    };

    let per_token_loss = (&neg_surrogate + &kl_penalty)?;
    let total = per_token_loss.sum_all()?;
    total
        .affine(params.loss_normalizer, 0.0)
        .map_err(Into::into)
}

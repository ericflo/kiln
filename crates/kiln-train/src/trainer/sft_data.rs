use super::*;

/// Tokenize a training example into (input_ids, label_mask).
///
/// The label_mask indicates which tokens are part of assistant responses
/// (true = compute loss here, false = ignore).
pub fn tokenize_for_training(
    example: &SftExample,
    tokenizer: &KilnTokenizer,
) -> Result<(Vec<u32>, Vec<bool>)> {
    let core_messages = to_core_messages(&example.messages);

    // Build the full conversation text using the chat template
    let (full_text, template_assistant_spans) = tokenizer
        .apply_chat_template_for_training_with_spans(&core_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (input_ids, offsets) = tokenizer
        .encode_with_offsets(&full_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if input_ids.is_empty() {
        anyhow::bail!("empty tokenization result");
    }

    let assistant_count = core_messages
        .iter()
        .filter(|message| message.role == "assistant")
        .count();
    let mut label_mask = if let Some(spans) = template_assistant_spans {
        anyhow::ensure!(
            spans.len() == assistant_count,
            "training template returned {} assistant spans for {assistant_count} assistant messages",
            spans.len()
        );
        let mut mask = vec![false; input_ids.len()];
        for (start, end) in spans {
            mark_offsets_overlapping_span(&mut mask, &offsets, start, end);
        }
        mask
    } else {
        label_mask_from_rendered_assistant_spans(
            &full_text,
            &offsets,
            input_ids.len(),
            assistant_count,
        )
        .unwrap_or_else(|| vec![false; input_ids.len()])
    };
    // ChatML/Qwen-style templates are handled directly from the single rendered
    // full example. This avoids prefix renders that are not stable when
    // templates append generation prompts or rewrite post-tool turns.
    if !label_mask.iter().any(|&marked| marked) {
        let mut prefix_messages: Vec<kiln_core::tokenizer::ChatMessage> = Vec::new();
        for msg in &core_messages {
            if msg.role == "assistant" {
                let before_text = if prefix_messages.is_empty() {
                    String::new()
                } else {
                    tokenizer
                        .apply_chat_template_for_training(&prefix_messages)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                };

                prefix_messages.push(msg.clone());
                let prefix_text = tokenizer
                    .apply_chat_template_for_training(&prefix_messages)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                if !full_text.starts_with(&prefix_text) || before_text.len() > prefix_text.len() {
                    label_mask = label_mask_by_prefix_tokenization(
                        input_ids.len(),
                        &core_messages,
                        tokenizer,
                    )?;
                    break;
                }

                let start = before_text.len();
                let end = prefix_text.len().min(full_text.len());
                for (i, &(token_start, token_end)) in offsets.iter().enumerate() {
                    if token_start == token_end {
                        continue;
                    }
                    if token_start < end && token_end > start {
                        label_mask[i] = true;
                    }
                }
            } else {
                prefix_messages.push(msg.clone());
            }
        }
    }

    // For next-token prediction, we need at least 2 tokens
    if input_ids.len() < 2 {
        anyhow::bail!("example too short ({} tokens)", input_ids.len());
    }
    if !has_supervised_shifted_labels(&label_mask) {
        anyhow::bail!("example has no supervised assistant tokens after next-token shift");
    }

    Ok((input_ids, label_mask))
}

pub(super) fn label_mask_from_rendered_assistant_spans(
    full_text: &str,
    offsets: &[(usize, usize)],
    input_len: usize,
    expected_assistant_spans: usize,
) -> Option<Vec<bool>> {
    const ASSISTANT_START: &str = "<|im_start|>assistant\n";
    const MESSAGE_END: &str = "<|im_end|>";

    if expected_assistant_spans == 0 {
        return Some(vec![false; input_len]);
    }

    let mut label_mask = vec![false; input_len];
    let mut search_from = 0usize;
    let mut found = 0usize;

    while let Some(relative_start) = full_text[search_from..].find(ASSISTANT_START) {
        let start = search_from + relative_start;
        let content_start = start + ASSISTANT_START.len();
        let Some(relative_end) = full_text[content_start..].find(MESSAGE_END) else {
            break;
        };
        let mut end = content_start + relative_end + MESSAGE_END.len();
        if full_text[end..].starts_with('\n') {
            end += 1;
        }

        // TRL's Qwen3.5 training template opens its generation span after the
        // assistant role header and closes it after the message terminator.
        mark_offsets_overlapping_span(&mut label_mask, offsets, content_start, end);
        found += 1;
        search_from = end;
    }

    (found == expected_assistant_spans).then_some(label_mask)
}

pub(super) fn mark_offsets_overlapping_span(
    label_mask: &mut [bool],
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
) {
    for (index, &(token_start, token_end)) in offsets.iter().enumerate() {
        if index >= label_mask.len() || token_start == token_end {
            continue;
        }
        if token_start < end && token_end > start {
            label_mask[index] = true;
        }
    }
}

pub(super) fn label_mask_by_prefix_tokenization(
    input_len: usize,
    core_messages: &[kiln_core::tokenizer::ChatMessage],
    tokenizer: &KilnTokenizer,
) -> Result<Vec<bool>> {
    let mut label_mask = vec![false; input_len];
    let mut prefix_messages: Vec<kiln_core::tokenizer::ChatMessage> = Vec::new();
    for msg in core_messages {
        prefix_messages.push(msg.clone());
        if msg.role == "assistant" {
            let prefix_text = tokenizer
                .apply_chat_template_for_training(&prefix_messages)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let prefix_ids = tokenizer
                .encode(&prefix_text)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let before_messages: Vec<_> = prefix_messages[..prefix_messages.len() - 1].to_vec();
            let before_text = if before_messages.is_empty() {
                String::new()
            } else {
                tokenizer
                    .apply_chat_template_for_training(&before_messages)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };
            let before_ids = if before_text.is_empty() {
                Vec::new()
            } else {
                tokenizer
                    .encode(&before_text)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            let start = before_ids.len();
            let end = prefix_ids.len().min(input_len);
            label_mask[start..end].fill(true);
        }
    }
    Ok(label_mask)
}

pub(super) fn has_supervised_shifted_labels(label_mask: &[bool]) -> bool {
    label_mask.get(1..).is_some_and(|m| m.iter().any(|&v| v))
}

/// Compute cross-entropy loss on masked positions.
///
/// `logits`: [1, seq_len, vocab_size] — model output
/// `input_ids`: token IDs (used as labels, shifted by 1)
/// `label_mask`: which positions to include in the loss
///
/// SFT next-token cross-entropy loss VALUE (scalar `f64`), kt-native.
///
/// (#1082 candle-drop) This is a value-only reader: it returns the scalar loss
/// for logging / the gradient-checkpoint final-boundary readback. The
/// *differentiable* CE root is `try_tape_cross_entropy_from_logits_kt` recorded
/// DIRECTLY by the SFT/GRPO/OPD `with_tape_authoritative_scope_kt` closures — it
/// does NOT go through here. The old candle `[1, T, V]` bridge + candle
/// log-sum-exp/gather composite + the candle `try_tape_cross_entropy_cuda`
/// adapter are deleted; unsupported backend/dtype combinations are rejected by
/// the backend tape route plus `TrainingPrecisionPolicy`. The kt CE math itself
/// is covered by `tape_forward_parity`
/// (`tape_forward_cross_entropy_matches_reference`,
/// `tape_backward_cross_entropy_matches_analytic_gradient`).
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn cross_entropy_loss(
    logits: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    _device: &Device,
) -> Result<f64> {
    let loss_kt = kiln_model::tape_forward::try_tape_cross_entropy_from_logits_kt(
        logits, input_ids, label_mask,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "cross_entropy_loss: kt CE-from-logits declined (requires CUDA BF16 [1, T, V] \
             logits; F32/CPU cross-entropy was dropped in the candle drop, #1082)"
        )
    })?;
    Ok(loss_kt.to_scalar::<f32>()? as f64)
}

/// Analytic SFT tail seed: `d loss / d hidden` for final RMSNorm + tied
/// LM-head + next-token cross-entropy.
///
/// This mirrors [`cross_entropy_loss`] / FLCE shifted-label semantics while
/// chunking over vocab so the full `[T, V]` logits tensor is never
/// materialized. The returned tensor is F32 with shape `[1, T, H]`; inactive
/// shifted-label rows and the final sequence row are zero.
pub(super) fn synchronize_tail_chunk(_context: &'static str) -> Result<()> {
    // (#1082) kt `Device` has no per-device `synchronize()` (candle-only API);
    // the old chunk-tail sync point is retained as a named no-op for caller
    // structure without branching on backend identity here.
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analytic_sft_tail_grad_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        None,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    let normed = rms_norm(hidden, final_norm_weight, rms_norm_eps)
        .context("analytic SFT tail final RMSNorm")?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        &normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analytic_sft_tail_grad_from_normed_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        Some(normed),
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analytic_sft_tail_grad_from_normed_pre_final_norm_with_flce_metadata(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
    active_metadata: Option<&kiln_flce_kernel::kt_api::FlceActiveMetadata>,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        Some(normed),
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        active_metadata,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
    active_metadata: Option<&kiln_flce_kernel::kt_api::FlceActiveMetadata>,
) -> Result<Tensor> {
    let grad_normed = if let Some(active_metadata) = active_metadata {
        kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt(
            normed, head_t, input_ids, label_mask, chunk_size, active_metadata,
        )
    } else {
        kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
            normed, head_t, input_ids, label_mask, chunk_size,
        )
    }
    .map_err(|e| anyhow::anyhow!("analytic SFT tail FLCE hidden gradient: {e}"))?;
    rms_norm_backward_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        final_norm_weight,
        &grad_normed,
        rms_norm_eps,
    )
    .context("analytic SFT tail final RMSNorm backward")
}

pub(super) fn validate_analytic_sft_tail_grad_inputs(
    hidden: &Tensor,
    normed: Option<&Tensor>,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<()> {
    let seq_len = input_ids.len();
    if seq_len < 2 {
        anyhow::bail!("analytic SFT tail gradient requires at least 2 tokens");
    }
    if chunk_size == 0 {
        anyhow::bail!("analytic SFT tail gradient chunk_size must be > 0");
    }
    if label_mask.len() != seq_len {
        anyhow::bail!(
            "label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len
        );
    }

    let dims = hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != seq_len {
        anyhow::bail!(
            "hidden must have shape [1, seq_len, hidden_size], got {:?} for seq_len {}",
            dims,
            seq_len
        );
    }
    if let Some(normed) = normed
        && normed.dims() != hidden.dims()
    {
        anyhow::bail!(
            "normed hidden shape {:?} does not match hidden shape {:?}",
            normed.dims(),
            hidden.dims()
        );
    }
    let hidden_size = dims[2];
    if final_norm_weight.dims() != [hidden_size] {
        anyhow::bail!(
            "final_norm_weight shape {:?} does not match hidden size {}",
            final_norm_weight.dims(),
            hidden_size
        );
    }
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        anyhow::bail!(
            "head_t must have shape [hidden_size, vocab_size], got {:?}",
            head_t.dims()
        );
    }

    Ok(())
}

pub(crate) fn rms_norm_backward_pre_final_norm(
    _final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    final_norm_weight: &Tensor,
    grad_normed: &Tensor,
    rms_norm_eps: f64,
) -> Result<Tensor> {
    let dims = hidden.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3,
        "rms_norm_backward_pre_final_norm: hidden must be [batch, seq, hidden], got {dims:?}"
    );
    anyhow::ensure!(
        grad_normed.dims() == hidden.dims(),
        "rms_norm_backward_pre_final_norm: grad_normed shape {:?} != hidden shape {:?}",
        grad_normed.dims(),
        hidden.dims()
    );
    let hidden_size = dims[2];
    anyhow::ensure!(
        final_norm_weight.dims() == [hidden_size],
        "rms_norm_backward_pre_final_norm: final_norm_weight shape {:?} != hidden size {hidden_size}",
        final_norm_weight.dims()
    );

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let same_device = final_norm_weight.device() == hidden.device()
            && grad_normed.device() == hidden.device();
        let non_empty_rows = dims[0] > 0 && dims[1] > 0;
        let fused_envelope = matches!(
            _final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::CudaRocmFusedTail
        ) && same_device
            && non_empty_rows
            && kiln_rmsnorm_kernel::supports_rmsnorm_kt(hidden, final_norm_weight)
            && grad_normed.dtype() == KtDType::BF16
            && grad_normed.is_contiguous();

        if fused_envelope {
            let grad_hidden = kiln_rmsnorm_kernel::fused_rmsnorm_backward_dx_kt(
                hidden,
                final_norm_weight,
                grad_normed,
                rms_norm_eps as f32,
            )
            .map_err(|e| anyhow::anyhow!("fused final RMSNorm backward: {e}"))?;
            return Ok(grad_hidden.detach());
        }
    }

    let hidden_f32 = hidden.to_f32_dtype()?;
    let grad_normed_f32 = grad_normed.to_f32_dtype()?;
    let norm_weight = final_norm_weight.to_f32_dtype()?;
    let norm_weight_plus_one = (norm_weight.ones_like()? + norm_weight)?;
    let variance = hidden_f32.sqr()?.mean_keepdim(LAST_DIM)?;
    let rms_inv = (variance + rms_norm_eps)?.sqrt()?.recip()?;

    // Qwen RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
    // Given dL/dy, the pre-norm gradient is:
    // u * r - x * r^3 / H * sum(u * x), where u = dL/dy * (1 + w).
    let u = grad_normed_f32.broadcast_mul(&norm_weight_plus_one)?;
    let dot = (&u * &hidden_f32)?.sum_keepdim(LAST_DIM)?;
    let rms_inv_sq = rms_inv.sqr()?;
    let rms_inv_cubed = rms_inv_sq.broadcast_mul(&rms_inv)?;
    let correction_scale = rms_inv_cubed.affine(1.0f64 / hidden_size as f64, 0.0)?;
    let correction = hidden_f32.broadcast_mul(&dot.broadcast_mul(&correction_scale)?)?;
    Ok((u.broadcast_mul(&rms_inv)? - correction)?.detach())
}

pub(super) fn synchronize_training_tensor_ready(label: &str, tensor: &Tensor) -> Result<()> {
    let _ = label;
    match tensor.device() {
        Device::Cpu => Ok(()),
        #[cfg(feature = "cuda")]
        Device::Cuda(idx) => kiln_tensor::cuda_synchronize_default_stream_for(
            idx,
            kiln_tensor::CudaSyncReason::TensorHandoff,
        )
        .with_context(|| format!("{label}: synchronize CUDA tensor readiness")),
        #[cfg(feature = "rocm")]
        Device::Rocm(idx) => {
            if kiln_tensor::rocm_capture_arena_active() {
                Ok(())
            } else {
                kiln_tensor::rocm_synchronize_default_stream(idx)
                    .with_context(|| format!("{label}: synchronize ROCm tensor readiness"))
            }
        }
        #[cfg(feature = "metal")]
        Device::Metal(idx) => kiln_tensor::primary_metal_companion(idx)
            .and_then(|companion| companion.wait_until_completed())
            .with_context(|| format!("{label}: synchronize Metal tensor readiness")),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(idx) => kiln_tensor::vulkan_synchronize_queue(idx)
            .with_context(|| format!("{label}: synchronize Vulkan tensor readiness")),
        _ => Ok(()),
    }
}

pub(super) fn summarize_sft_debug_values(tensor: &Tensor) -> Result<(bool, String)> {
    let host = tensor
        .to_device(Device::Cpu)
        .context("copy SFT debug tensor to CPU")?
        .to_dtype(DType::F32)
        .context("cast SFT debug tensor to f32")?
        .contiguous()
        .context("make SFT debug CPU tensor contiguous")?;
    let values = host
        .to_vec::<f32>()
        .context("read SFT debug CPU tensor values")?;
    let mut first_bad: Option<(usize, f32)> = None;
    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (idx, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            let abs = value.abs();
            if abs > max_abs {
                max_abs = abs;
                max_abs_idx = idx;
            }
        } else if first_bad.is_none() {
            first_bad = Some((idx, value));
        }
    }
    let shape = tensor.shape();
    let coord = |mut idx: usize| -> Vec<usize> {
        let mut out = vec![0usize; shape.len()];
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis].max(1);
            out[axis] = idx % dim;
            idx /= dim;
        }
        out
    };
    let (bad_idx, bad_value) = first_bad.unwrap_or((usize::MAX, f32::NAN));
    let summary = format!(
        "first_bad_flat={} first_bad_coord={:?} first_bad_value={} max_finite_abs={} max_finite_abs_flat={} max_finite_abs_coord={:?}",
        bad_idx,
        if bad_idx == usize::MAX {
            Vec::new()
        } else {
            coord(bad_idx)
        },
        bad_value,
        max_abs,
        max_abs_idx,
        coord(max_abs_idx)
    );
    Ok((first_bad.is_none(), summary))
}

pub(super) fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::BF16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::U8 => 1,
        DType::U32 => 4,
        DType::I64 => 8,
        _ => 4,
    }
}

pub(super) struct StoredCheckpointBoundaries {
    pub(super) tensors: std::cell::RefCell<Vec<Option<Tensor>>>,
    pub(super) resident_device_storage: bool,
    pub(super) anchor_stride: usize,
}

impl StoredCheckpointBoundaries {
    pub(super) fn new(
        num_segments: usize,
        resident_device_storage: bool,
        anchor_stride: usize,
    ) -> Self {
        Self {
            tensors: std::cell::RefCell::new(vec![None; num_segments + 1]),
            resident_device_storage,
            anchor_stride: anchor_stride.max(1),
        }
    }

    pub(super) fn should_store(&self, boundary_idx: usize) -> bool {
        boundary_idx == 0 || boundary_idx.is_multiple_of(self.anchor_stride)
    }

    pub(super) fn anchor_for_boundary(&self, boundary_idx: usize) -> usize {
        (boundary_idx / self.anchor_stride) * self.anchor_stride
    }

    // Long-context checkpoint boundaries are too large to retain at every
    // segment boundary. Keep sparse anchors in process memory and replay from
    // the nearest anchor on demand.
    pub(super) fn save(&self, boundary_idx: usize, tensor: &Tensor) -> Result<()> {
        if !self.should_store(boundary_idx) {
            return Ok(());
        }
        let stored = if self.resident_device_storage {
            tensor
                .contiguous()
                .map_err(|e| anyhow::anyhow!("checkpoint boundary save: contiguous: {e}"))?
        } else {
            tensor
                .to_device(kiln_tensor::Device::Cpu)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("checkpoint boundary save: to cpu: {e}"))?
        };
        let mut tensors = self.tensors.borrow_mut();
        let slot = tensors.get_mut(boundary_idx).ok_or_else(|| {
            anyhow::anyhow!("checkpoint boundary index {boundary_idx} out of storage range")
        })?;
        *slot = Some(stored);
        Ok(())
    }

    pub(super) fn load_stored(
        &self,
        boundary_idx: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let tensors = self.tensors.borrow();
        let Some(slot) = tensors.get(boundary_idx) else {
            anyhow::bail!("checkpoint boundary index {boundary_idx} out of spool range");
        };
        let Some(hidden) = slot.as_ref() else {
            return Ok(None);
        };
        if self.resident_device_storage {
            return Ok(Some(hidden.clone()));
        }
        Ok(Some(hidden.to_device(*device).map_err(|e| {
            anyhow::anyhow!("checkpoint boundary load: move to device: {e}")
        })?))
    }

    pub(super) fn load(&self, boundary_idx: usize, device: &Device) -> Result<Tensor> {
        self.load_stored(boundary_idx, device)?.ok_or_else(|| {
            anyhow::anyhow!("checkpoint boundary {boundary_idx} missing hidden tensor")
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_or_recompute_checkpoint_boundary(
    spool: &StoredCheckpointBoundaries,
    boundary_idx: usize,
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    positions: &[u32],
    segments: &[(usize, usize)],
    lora_detached: &LoraWeights,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    anyhow::ensure!(
        boundary_idx <= segments.len(),
        "checkpoint boundary {boundary_idx} out of range for {} segments",
        segments.len()
    );
    if let Some(stored) = spool.load_stored(boundary_idx, device)? {
        return Ok(stored);
    }

    let anchor_idx = spool.anchor_for_boundary(boundary_idx);
    let mut current = spool.load(anchor_idx, device)?;
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    for (offset, &(start, end)) in segments[anchor_idx..boundary_idx].iter().enumerate() {
        let replay_idx = anchor_idx + offset;
        current = model_forward_segment_with_policy(
            backend,
            current,
            weights,
            model_config,
            positions,
            start,
            end,
            Some(&mut linear_state),
            Some(lora_detached),
            streaming_prefill,
        )
        .with_context(|| {
            format!("checkpoint boundary replay segment {replay_idx} layers {start}..{end}")
        })?
        .detach();
    }
    Ok(current)
}

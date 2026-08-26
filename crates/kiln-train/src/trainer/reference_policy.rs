use super::*;

/// Compute group-normalized advantages from rewards.
///
/// advantage_i = (reward_i - mean(rewards)) / (std(rewards) + 1e-8)
/// Deep-copy a tensor so the result's backing storage is independent of the
/// input's (which may be a [`Var`]'s storage that subsequent optimizer steps
/// can replace). Goes via host-side `f32` round-trip; restores the original
/// dtype on the way out.
///
/// Used by [`lora_snapshot_capture_or_blend`] to materialize a reference
/// LoRA that won't silently track future policy updates.
pub(super) fn deepcopy_tensor_for_snapshot(t: &Tensor, snapshot_device: &Device) -> Result<Tensor> {
    let dtype = t.dtype();
    let shape = t.dims().to_vec();
    let host: Vec<f32> = t
        .to_f32_dtype()?
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()
        .context("snapshot: read tensor to host f32 vec")?;
    // (#1082) kt-native rebuild on the source device (no candle constructor).
    let rebuilt = Tensor::from_vec_on(snapshot_device.clone(), host, shape)?;
    if dtype == DType::F32 {
        Ok(rebuilt.detach())
    } else {
        Ok(rebuilt.to_dtype(dtype)?.detach())
    }
}

/// EMA blend two tensors: `new = decay * old + (1 - decay) * current`. The
/// result has the same dtype as `old` and is independent of either input's
/// storage (the affine + add chain materializes a fresh tensor).
pub(super) fn ema_blend_tensor(
    old: &Tensor,
    current: &Tensor,
    decay: f32,
    snapshot_device: &Device,
) -> Result<Tensor> {
    let dtype = old.dtype();
    let a = old
        .to_device(snapshot_device.clone())?
        .to_f32_dtype()?
        .affine(decay as f64, 0.0)?;
    let b = current
        .to_device(snapshot_device.clone())?
        .to_f32_dtype()?
        .affine((1.0 - decay) as f64, 0.0)?;
    let blended = (a + b)?;
    let out = if dtype == DType::F32 {
        blended
    } else {
        blended.to_dtype(dtype)?
    };
    Ok(out.detach())
}

/// Per-projection helper used by [`lora_snapshot_capture_or_blend`]. Given a
/// current trainable Var pair and an optional prior snapshot projection,
/// produces a fresh `LoraProjectionWeights` whose tensors are EMA-blended
/// from the snapshot toward the current params (or a pure deepcopy of
/// current if no prior snapshot exists).
pub(super) fn snapshot_projection(
    cur: &Option<(Parameter, Parameter)>,
    prior: Option<&LoraProjectionWeights>,
    decay: f32,
    snapshot_device: &Device,
) -> Result<Option<LoraProjectionWeights>> {
    let Some((cur_a, cur_b)) = cur else {
        return Ok(None);
    };
    // (#1082) The EMA blend / deepcopy helpers (`ema_blend_tensor` /
    // `deepcopy_tensor_for_snapshot`) are now kt-native, and the param's
    // primary tensor + `LoraProjectionWeights.a/.b` are kt, so the whole
    // snapshot blend runs in kt with no candle bridge.
    let cur_a_kt = cur_a.forward_storage().primary_tensor();
    let cur_b_kt = cur_b.forward_storage().primary_tensor();
    let (a, b) = match prior {
        Some(prior) => (
            ema_blend_tensor(&prior.a, cur_a_kt, decay, snapshot_device)?,
            ema_blend_tensor(&prior.b, cur_b_kt, decay, snapshot_device)?,
        ),
        None => (
            deepcopy_tensor_for_snapshot(cur_a_kt, snapshot_device)?,
            deepcopy_tensor_for_snapshot(cur_b_kt, snapshot_device)?,
        ),
    };
    anyhow::ensure!(
        a.device() == *snapshot_device && b.device() == *snapshot_device,
        "GRPO EMA snapshot landed on {}/{} instead of {}",
        a.device(),
        b.device(),
        snapshot_device
    );
    Ok(Some(LoraProjectionWeights { a, b }))
}

/// Capture an EMA snapshot of the current LoRA params, blending with a prior
/// snapshot if provided.
///
/// * `prior = None`: a fresh deepcopy of `current` becomes the snapshot.
/// * `prior = Some(snap)`: returns `decay * snap + (1 - decay) * current`,
///   blended per-tensor.
///
/// Returned `LoraWeights` is fully owned (no aliasing of `current`'s Var
/// storage) and safe to pass as the reference into `model_forward_no_head`
/// across subsequent optimizer steps on `current`.
///
/// Used by [`KlReferencePolicy::Ema`] in `grpo_train` and `grpo_train_jsonl`.
pub(super) fn lora_snapshot_capture_or_blend(
    current: &TrainableLoraParams,
    prior: Option<&LoraWeights>,
    decay: f32,
    snapshot_device: &Device,
) -> Result<LoraWeights> {
    let layers = current
        .layers
        .iter()
        .enumerate()
        .map(|(layer_idx, lp)| {
            let snap_layer = prior.and_then(|p| p.layers.get(layer_idx));
            // For each named projection, blend or deepcopy.
            let mk = |which: fn(&LoraLayerWeights) -> Option<&LoraProjectionWeights>,
                      cur: &Option<(Parameter, Parameter)>|
             -> Result<Option<LoraProjectionWeights>> {
                snapshot_projection(cur, snap_layer.and_then(which), decay, snapshot_device)
            };
            Ok::<LoraLayerWeights, anyhow::Error>(LoraLayerWeights {
                q_proj: mk(|l| l.q_proj.as_ref(), &lp.q_proj)?,
                k_proj: mk(|l| l.k_proj.as_ref(), &lp.k_proj)?,
                v_proj: mk(|l| l.v_proj.as_ref(), &lp.v_proj)?,
                o_proj: mk(|l| l.o_proj.as_ref(), &lp.o_proj)?,
                in_proj_qkv: mk(|l| l.in_proj_qkv.as_ref(), &lp.in_proj_qkv)?,
                in_proj_z: mk(|l| l.in_proj_z.as_ref(), &lp.in_proj_z)?,
                gdn_out_proj: mk(|l| l.gdn_out_proj.as_ref(), &lp.gdn_out_proj)?,
                gate_proj: mk(|l| l.gate_proj.as_ref(), &lp.gate_proj)?,
                up_proj: mk(|l| l.up_proj.as_ref(), &lp.up_proj)?,
                down_proj: mk(|l| l.down_proj.as_ref(), &lp.down_proj)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoraWeights {
        layers,
        mtp: None,
        rank: current.rank,
        alpha: current.alpha,
        scale: current.scale,
        source_identity: None,
    })
}

pub(super) fn capture_lora_reference_checkpoint(
    snapshot: &LoraWeights,
) -> Result<CheckpointTensorSnapshot> {
    anyhow::ensure!(
        snapshot.mtp.is_none(),
        "GRPO EMA reference checkpoint must not contain MTP weights"
    );
    let mut tensors = Vec::new();
    for (layer_idx, layer) in snapshot.layers.iter().enumerate() {
        for (module, projection) in [
            ("q_proj", &layer.q_proj),
            ("k_proj", &layer.k_proj),
            ("v_proj", &layer.v_proj),
            ("o_proj", &layer.o_proj),
            ("in_proj_qkv", &layer.in_proj_qkv),
            ("in_proj_z", &layer.in_proj_z),
            ("out_proj", &layer.gdn_out_proj),
            ("gate_proj", &layer.gate_proj),
            ("up_proj", &layer.up_proj),
            ("down_proj", &layer.down_proj),
        ] {
            let Some(projection) = projection else {
                continue;
            };
            for (matrix, tensor) in [("A", &projection.a), ("B", &projection.b)] {
                let key = checkpoint_parameter_key(layer_idx, module, matrix);
                let tensor = tensor
                    .to_device(kiln_tensor::Device::Cpu)
                    .and_then(|tensor| tensor.contiguous())
                    .map_err(|error| {
                        anyhow::anyhow!("capture GRPO EMA reference tensor {key}: {error}")
                    })?;
                checkpoint_ensure_finite_tensor(&tensor, &key)?;
                tensors.push((key, tensor));
            }
        }
    }
    CheckpointTensorSnapshot::new(tensors, "GRPO EMA reference")
}

pub(super) fn restore_lora_reference_tensor(
    loaded: &mut HashMap<String, KtTensor>,
    key: &str,
    current: &KtTensor,
    snapshot_device: &Device,
) -> Result<KtTensor> {
    let tensor = loaded
        .remove(key)
        .with_context(|| format!("GRPO EMA reference tensor {key} missing"))?;
    anyhow::ensure!(
        tensor.dims() == current.dims(),
        "GRPO EMA reference tensor {key} shape mismatch: expected {:?}, found {:?}",
        current.dims(),
        tensor.dims()
    );
    anyhow::ensure!(
        tensor.dtype() == current.dtype(),
        "GRPO EMA reference tensor {key} dtype mismatch: expected {}, found {}",
        current.dtype(),
        tensor.dtype()
    );
    checkpoint_ensure_finite_tensor(&tensor, key)?;
    tensor
        .to_device(snapshot_device.clone())
        .and_then(|tensor| tensor.contiguous())
        .map_err(|error| anyhow::anyhow!("restore GRPO EMA reference tensor {key}: {error}"))
}

pub(super) fn restore_lora_reference_projection(
    loaded: &mut HashMap<String, KtTensor>,
    layer_idx: usize,
    module: &str,
    current: &Option<(Parameter, Parameter)>,
    snapshot_device: &Device,
) -> Result<Option<LoraProjectionWeights>> {
    let Some((current_a, current_b)) = current else {
        return Ok(None);
    };
    let a_key = checkpoint_parameter_key(layer_idx, module, "A");
    let b_key = checkpoint_parameter_key(layer_idx, module, "B");
    Ok(Some(LoraProjectionWeights {
        a: restore_lora_reference_tensor(
            loaded,
            &a_key,
            current_a.forward_storage().primary_tensor(),
            snapshot_device,
        )?,
        b: restore_lora_reference_tensor(
            loaded,
            &b_key,
            current_b.forward_storage().primary_tensor(),
            snapshot_device,
        )?,
    }))
}

pub(super) fn load_lora_reference_checkpoint(
    path: &Path,
    current: &TrainableLoraParams,
    snapshot_device: &Device,
) -> Result<LoraWeights> {
    let mut loaded = kiln_tensor::safetensors::load_cpu(path)
        .map_err(|error| anyhow::anyhow!("load GRPO EMA reference checkpoint: {error}"))?;
    let expected: BTreeSet<_> = current.checkpoint_param_keys().into_iter().collect();
    let actual: BTreeSet<_> = loaded.keys().cloned().collect();
    anyhow::ensure!(
        actual == expected,
        "GRPO EMA reference tensor set mismatch: expected {expected:?}, found {actual:?}"
    );

    let layers = current
        .layers
        .iter()
        .enumerate()
        .map(|(layer_idx, layer)| {
            Ok::<_, anyhow::Error>(LoraLayerWeights {
                q_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "q_proj",
                    &layer.q_proj,
                    snapshot_device,
                )?,
                k_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "k_proj",
                    &layer.k_proj,
                    snapshot_device,
                )?,
                v_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "v_proj",
                    &layer.v_proj,
                    snapshot_device,
                )?,
                o_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "o_proj",
                    &layer.o_proj,
                    snapshot_device,
                )?,
                in_proj_qkv: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "in_proj_qkv",
                    &layer.in_proj_qkv,
                    snapshot_device,
                )?,
                in_proj_z: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "in_proj_z",
                    &layer.in_proj_z,
                    snapshot_device,
                )?,
                gdn_out_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "out_proj",
                    &layer.gdn_out_proj,
                    snapshot_device,
                )?,
                gate_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "gate_proj",
                    &layer.gate_proj,
                    snapshot_device,
                )?,
                up_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "up_proj",
                    &layer.up_proj,
                    snapshot_device,
                )?,
                down_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "down_proj",
                    &layer.down_proj,
                    snapshot_device,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(loaded.is_empty(), "unconsumed GRPO EMA reference tensors");
    Ok(LoraWeights {
        layers,
        mtp: None,
        rank: current.rank,
        alpha: current.alpha,
        scale: current.scale,
        source_identity: None,
    })
}

/// State threaded through a GRPO run to support `KlReferencePolicy::Ema`.
///
/// Captures the most recent snapshot of the LoRA params and the number of
/// completed groups since the last refresh. When `groups_since_refresh
/// >= refresh_every`, the outer caller calls
/// [`lora_snapshot_capture_or_blend`] with the current params and decay,
/// then resets the counter.
pub(super) struct EmaReferenceState {
    pub(super) snapshot: LoraWeights,
    pub(super) groups_since_refresh: usize,
    pub(super) refresh_every: usize,
    pub(super) decay: f32,
}

/// Returns true when every completion in `group` carries the same reward.
/// Such a group produces a uniformly-zero advantage vector under any of the
/// supported [`AdvantageMode`]s and contributes no policy-gradient signal,
/// only a spurious KL update. Dropped by Dynamic Sampling
/// (DAPO, arXiv:2503.14476) when `GrpoConfig::dynamic_sampling` is true.
pub(super) fn is_degenerate_grpo_group(group: &GrpoGroup) -> bool {
    let mut rewards = group.completions.iter().map(|c| c.reward);
    let Some(first) = rewards.next() else {
        return true;
    };
    rewards.all(|r| r == first)
}

pub(super) fn compute_advantages(rewards: &[f64], mode: AdvantageMode) -> Vec<f64> {
    let n = rewards.len() as f64;
    if n <= 1.0 {
        return vec![0.0; rewards.len()];
    }
    let mean = rewards.iter().sum::<f64>() / n;
    let centered: Vec<f64> = rewards.iter().map(|r| r - mean).collect();
    match mode {
        AdvantageMode::DrGrpo => centered,
        AdvantageMode::Vanilla => {
            let var = centered.iter().map(|c| c * c).sum::<f64>() / n;
            let std = var.sqrt();
            centered.into_iter().map(|c| c / (std + 1e-8)).collect()
        }
    }
}

/// Compute per-token log-probabilities for the tokens indicated by the mask.
///
/// Returns a 1-D tensor of log-probs for only the masked (completion) positions.
/// Uses the next-token prediction convention: logits[i] predicts token[i+1].
// `pub(crate)` so the GRPO tape-authoritative loss-root shim
// (`crate::grpo_tape_shim`) can recompute the EXACT same policy log-probs
// inside its candle-autograd backward composite (#1082 CP-4).
pub(crate) fn token_log_probs(
    logits: &Tensor,
    input_ids: &[u32],
    mask: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let seq_len = input_ids.len();
    let logits = logits.squeeze(0)?; // [seq_len, vocab_size]

    // Next-token prediction: logits[i] predicts input_ids[i+1]
    // So for completion token at position j, use logits[j-1]
    let shift_logits = logits.narrow(0, 0, seq_len - 1)?; // [seq_len-1, vocab_size]
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = mask[1..].to_vec();

    // Find active positions (completion tokens)
    let active_positions: Vec<usize> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    if active_positions.is_empty() {
        // Return a zero tensor if no completion tokens
        return zeros_f32_on(1, device);
    }

    // Gather active logits
    let active_idx_u32: Vec<u32> = active_positions.iter().map(|&i| i as u32).collect();
    let n_active_idx = active_idx_u32.len();
    let indices = Tensor::from_vec_on(*device, active_idx_u32, vec![n_active_idx])?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [num_active, vocab_size]

    let active_labels: Vec<u32> = active_positions.iter().map(|&i| shift_labels[i]).collect();

    // log_softmax denominator (CUDA-capable reduce).
    let active_logits_f32 = active_logits.to_f32_dtype()?;
    let log_sum_exp = active_logits_f32.log_sum_exp(LAST_DIM)?; // [num_active]

    // correct_logits[a] = active_logits[a, label_a]. (#1082) kt `gather` is
    // CPU-only (gather.rs requires both indices AND x be CpuStorage), whereas
    // the candle gather it replaced ran on CUDA — so a direct `.gather` here
    // breaks the CUDA GRPO path. Select via a FLAT `index_select` instead, which
    // is CUDA-capable (the `shift_logits.index_select` above already established
    // that on-device U32 indices work) and stays on-device (a CPU round-trip of
    // the [num_active, vocab=248320] active logits would be prohibitive). Flatten
    // [num_active, vocab] -> [num_active*vocab] and index at a*vocab + label_a.
    let vocab_size = *active_logits_f32
        .dims()
        .last()
        .expect("active_logits_f32 has a last dim");
    let flat_idx: Vec<u32> = active_labels
        .iter()
        .enumerate()
        .map(|(a, &lbl)| (a * vocab_size + lbl as usize) as u32)
        .collect();
    let flat_indices = Tensor::from_vec_on(*device, flat_idx, vec![n_active_idx])?;
    let correct_logits = active_logits_f32
        .contiguous()?
        .flatten_all()? // [num_active*vocab]
        .index_select(&flat_indices, 0)?; // [num_active]

    // log_prob = logit - log_sum_exp
    let log_probs = (correct_logits - log_sum_exp)?;

    Ok(log_probs)
}

/// Select one logit per row from a chunked `[rows, chunk_len]` logits tile.
///
/// This is the chunked analogue of [`token_log_probs`]' flat-index
/// `index_select`: it avoids materializing a dense `[rows, chunk_len]`
/// one-hot tensor just to pick sparse target columns. Labels outside this
/// chunk contribute zero, so summing the returned chunks recovers the selected
/// full-vocab logits.
pub(crate) fn selected_logits_from_chunk_sparse(
    logits_chunk: &Tensor,
    target_ids: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    vocab_size: usize,
    device: &Device,
    caller: &str,
) -> Result<Tensor> {
    let num_rows = target_ids.len();
    let dims = logits_chunk.dims();
    anyhow::ensure!(
        dims == [num_rows, chunk_len],
        "{caller}: logits_chunk shape {dims:?} != [{num_rows}, {chunk_len}]"
    );

    let mut row_indices = Vec::new();
    let mut flat_indices = Vec::new();
    for (row_idx, &label) in target_ids.iter().enumerate() {
        let label = label as usize;
        if label >= vocab_size {
            anyhow::bail!("{caller}: label {label} is outside vocab size {vocab_size}");
        }
        if label >= chunk_start && label < chunk_start + chunk_len {
            let rel = label - chunk_start;
            let flat = row_idx
                .checked_mul(chunk_len)
                .and_then(|base| base.checked_add(rel))
                .ok_or_else(|| anyhow::anyhow!("{caller}: flat selected-logit index overflow"))?;
            row_indices.push(
                u32::try_from(row_idx)
                    .with_context(|| format!("{caller}: row index {row_idx} exceeds u32 range"))?,
            );
            flat_indices.push(
                u32::try_from(flat)
                    .with_context(|| format!("{caller}: flat index {flat} exceeds u32 range"))?,
            );
        }
    }

    if flat_indices.is_empty() {
        return Tensor::zeros(vec![num_rows, 1], DType::F32, *device).map_err(Into::into);
    }

    let n_selected = flat_indices.len();
    let flat_idx = Tensor::from_vec_on(*device, flat_indices, vec![n_selected])?;
    let selected = logits_chunk
        .contiguous()?
        .flatten_all()?
        .index_select(&flat_idx, 0)?;
    let row_idx = Tensor::from_vec_on(*device, row_indices, vec![n_selected])?;
    let selected_rows = kiln_tensor::ops::scatter_add(&selected, 0, &row_idx, num_rows)?;
    selected_rows.unsqueeze(1).map_err(Into::into)
}

/// Compute selected next-token log-probs from post-final-RMSNorm hidden states
/// without materializing the full `[seq_len, vocab_size]` logits tensor.
pub(super) fn selected_log_probs_from_normed_hidden_chunked(
    normed_hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    mask: &[bool],
    chunk_size: usize,
) -> Result<Tensor> {
    let device = normed_hidden.device();
    let seq_len = input_ids.len();
    if seq_len < 2 {
        anyhow::bail!("selected log-probs require at least 2 tokens");
    }
    if mask.len() != seq_len {
        anyhow::bail!(
            "selected log-prob mask length {} does not match input length {}",
            mask.len(),
            seq_len
        );
    }
    if chunk_size == 0 {
        anyhow::bail!("selected log-prob chunk_size must be > 0");
    }

    let dims = normed_hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != seq_len {
        anyhow::bail!(
            "normed_hidden must have shape [1, seq_len, hidden_size], got {:?}",
            dims
        );
    }
    let hidden_size = dims[2];
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        anyhow::bail!(
            "head_t must have shape [hidden_size, vocab_size], got {:?}",
            head_t.dims()
        );
    }

    let active_positions: Vec<u32> = mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    if active_positions.is_empty() {
        return zeros_f32_on(1, &device);
    }
    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| input_ids[i as usize + 1])
        .collect();

    let hidden_2d = normed_hidden.squeeze(0)?;
    let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
    let n_pos = active_positions.len();
    let active_indices = Tensor::from_vec_on(device, active_positions.clone(), vec![n_pos])?;
    let active_hidden = shift_hidden
        .index_select(&active_indices, 0)?
        .to_f32_dtype()?;

    let head_t_f32 = head_t.to_f32_dtype()?;
    let vocab_size = head_t_f32.dim(1)?;
    if vocab_size == 0 {
        anyhow::bail!("head_t vocab dimension is zero");
    }

    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut correct_logits: Option<Tensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        {
            let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
            let logits_chunk = active_hidden.matmul(&head_chunk)?;
            let chunk_max = logits_chunk.max_keepdim(LAST_DIM)?;
            let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
                (None, None) => {
                    let shifted =
                        (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    (chunk_max.detach(), chunk_sumexp.detach())
                }
                (Some(prev_max), Some(prev_sumexp)) => {
                    let new_max = prev_max.maximum(&chunk_max)?;
                    let prev_scale = (prev_max - &new_max)?.exp()?;
                    let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                    let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    let new_sumexp = (scaled_prev + chunk_sumexp)?;
                    (new_max.detach(), new_sumexp.detach())
                }
                _ => unreachable!("running max/sumexp are set together"),
            };
            running_max = Some(new_max);
            running_sumexp = Some(new_sumexp);

            let chunk_correct = selected_logits_from_chunk_sparse(
                &logits_chunk,
                &active_labels,
                chunk_start,
                chunk_len,
                vocab_size,
                &device,
                "selected_log_probs_from_normed_hidden_chunked",
            )?;
            correct_logits = Some(match correct_logits.as_ref() {
                Some(prev) => (prev + chunk_correct)?.detach(),
                None => chunk_correct.detach(),
            });
        }
        synchronize_tail_chunk("synchronize selected log-prob chunk")?;
        chunk_start = chunk_end;
    }

    let running_max = running_max.context("vocab_size was zero")?;
    let running_sumexp = running_sumexp.context("vocab_size was zero")?;
    let correct_logits = correct_logits.context("vocab_size was zero")?;
    let log_sum_exp = (running_max + running_sumexp.log()?)?;
    Ok((correct_logits - log_sum_exp)?.squeeze(1)?)
}

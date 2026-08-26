use super::*;

/// Tokenized data for a single completion within a GRPO group.
///
/// Carries two parallel masks:
/// - `action_mask` — true at policy-gradient target positions (assistant
///   tokens). Equivalent to the pre-ECHO `completion_mask` for legacy
///   single-turn rollouts.
/// - `env_mask` — true at environment-observation target positions (tool
///   results). All-false when the rollout has no trajectory or when the
///   trajectory is single-turn Action-only. ECHO's env-CE consumes this.
pub(super) struct TokenizedGrpoCompletion {
    /// Full input_ids: prompt + completion tokens.
    pub(super) input_ids: Vec<u32>,
    /// Exact end of the prompt prefix, independent of the first sampled
    /// action position. Forced controller tokens may appear after this point.
    pub(super) prompt_token_count: usize,
    /// Mask of positions the model generated (assistant turns).
    /// Targets of the GRPO policy-gradient objective.
    pub(super) action_mask: Vec<bool>,
    /// Mask of positions the environment produced (tool-result turns).
    /// Targets of ECHO's env-CE auxiliary loss. All-false for legacy
    /// single-turn rollouts.
    pub(super) env_mask: Vec<bool>,
    /// Total observation length |O| for paper §3.1 length normalization
    /// in the ECHO term. Counts every Observation token regardless of the
    /// warning_filter trim — `env_mask` may be a strict subset of |O|.
    pub(super) total_obs_len: usize,
    /// Behavior-policy log-probabilities in sampled-action order. `None`
    /// means the rollout was admitted only under an explicit
    /// no-importance-correction policy.
    pub(super) recorded_behavior_log_probs: Option<Vec<f32>>,
    /// Content-addressed rollout source identity without the provenance
    /// record's potentially long token arrays.
    pub(super) recorded_behavior_source:
        Option<crate::train_receipt::GrpoRecordedBehaviorSourceObservation>,
}

/// A tokenized GRPO group ready for training.
pub(super) struct TokenizedGrpoGroup {
    pub(super) completions: Vec<TokenizedGrpoCompletion>,
    pub(super) rewards: Vec<f64>,
}

pub(super) fn validate_tokenized_behavior_policy(
    group: &TokenizedGrpoGroup,
    behavior_policy: BehaviorPolicy,
) -> Result<()> {
    for (completion_idx, completion) in group.completions.iter().enumerate() {
        anyhow::ensure!(
            completion.prompt_token_count > 0
                && completion.prompt_token_count <= completion.input_ids.len(),
            "GRPO completion {completion_idx} has invalid prompt_token_count {} for {} input tokens",
            completion.prompt_token_count,
            completion.input_ids.len()
        );
        let sampled_tokens = completion
            .action_mask
            .get(1..)
            .map_or(0, |mask| mask.iter().filter(|&&active| active).count());
        if behavior_policy == BehaviorPolicy::Recorded {
            let log_probs = completion.recorded_behavior_log_probs.as_ref().with_context(|| {
                format!(
                    "GRPO completion {completion_idx} is missing exact rollout provenance required by behavior_policy=recorded"
                )
            })?;
            anyhow::ensure!(
                log_probs.len() == sampled_tokens,
                "GRPO completion {completion_idx} has {} recorded behavior log-probabilities for {sampled_tokens} sampled action tokens",
                log_probs.len()
            );
            anyhow::ensure!(
                log_probs
                    .iter()
                    .all(|value| value.is_finite() && *value <= 1e-6),
                "GRPO completion {completion_idx} contains an invalid recorded behavior log-probability"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct GrpoGroupStepReport {
    pub(super) loss: f64,
    pub(super) echo_env_ce: Option<f64>,
}

pub(super) fn token_counts_for_grpo_groups(
    groups: &[TokenizedGrpoGroup],
) -> crate::train_receipt::TokenCountReceipt {
    let mut counts = crate::train_receipt::TokenCountReceipt::default();
    for group in groups {
        for completion in &group.completions {
            let action = completion
                .action_mask
                .iter()
                .filter(|&&active| active)
                .count() as u64;
            let env = completion.env_mask.iter().filter(|&&active| active).count() as u64;
            let env_before = (completion.total_obs_len as u64).max(env);
            counts.observe_completion(completion.input_ids.len(), action, env, env_before);
        }
    }
    counts
}

pub(super) fn grpo_benchmark_report_from_tokenized(
    tgroup: &TokenizedGrpoGroup,
    timings: GrpoBenchmarkTimings,
    loss: Option<f64>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    elapsed: Duration,
) -> GrpoBenchmarkReport {
    let counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
    let min_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .min()
        .unwrap_or(0);
    let max_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0);
    let total_tokens = counts
        .action_tokens
        .saturating_add(counts.env_tokens)
        .saturating_add(counts.context_tokens);
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let tokens_per_sec = if total_ms > 0.0 {
        total_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    GrpoBenchmarkReport {
        completions: tgroup.completions.len(),
        min_seq_len,
        max_seq_len,
        total_tokens,
        action_tokens: counts.action_tokens,
        env_tokens: counts.env_tokens,
        context_tokens: counts.context_tokens,
        loss,
        policy_audit,
        timings,
        total_ms,
        tokens_per_sec,
    }
}

pub fn grpo_benchmark_tokenization(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
) -> Result<GrpoBenchmarkReport> {
    let started = Instant::now();
    let mut timings = GrpoBenchmarkTimings::default();
    let mask_cfg = crate::trajectory_mask::MaskConfig::default();
    let tgroup = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut timings))?;
    Ok(grpo_benchmark_report_from_tokenized(
        &tgroup,
        timings,
        None,
        None,
        started.elapsed(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn grpo_benchmark_training_step(
    backend: &dyn BackendRuntime,
    group: &GrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    // (#1082) `&mut` — the GRPO step mutates each LoRA `Parameter` in place.
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    tokenizer: &KilnTokenizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<GrpoBenchmarkReport> {
    grpo_benchmark_training_step_with_policy(
        backend,
        group,
        weights,
        model_config,
        params,
        config,
        segments,
        device,
        tokenizer,
        opt_state,
        StreamingPrefillExecutionPolicy::for_device(*device),
    )
}

/// Explicit-policy variant of [`grpo_benchmark_training_step`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_benchmark_training_step_with_policy(
    backend: &dyn BackendRuntime,
    group: &GrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    tokenizer: &KilnTokenizer,
    opt_state: Option<&mut OptimizerState>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<GrpoBenchmarkReport> {
    let started = Instant::now();
    let mut timings = GrpoBenchmarkTimings::default();
    let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
    let tgroup = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut timings))?;
    let mut grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let mut policy_audit = crate::train_receipt::GrpoPolicyAuditAccumulator::default();
    let lora_grad_index = LoraGradNormIndex::new(params);
    let step_report = train_tokenized_grpo_group_with_grad_norms(
        backend,
        &tgroup,
        weights,
        model_config,
        params,
        config,
        segments,
        device,
        opt_state,
        &mut grad_norms,
        &lora_grad_index,
        &mut policy_audit,
        None,
        Some(&mut timings),
        streaming_prefill,
    )?;
    let policy_audit = policy_audit
        .finish()
        .context("finish GRPO benchmark policy audit")?;
    Ok(grpo_benchmark_report_from_tokenized(
        &tgroup,
        timings,
        Some(step_report.loss),
        Some(policy_audit),
        started.elapsed(),
    ))
}

pub(super) fn validate_grpo_trajectory_roles(group: &GrpoGroup, line_no: usize) -> Result<()> {
    for (rollout_idx, rollout) in group.completions.iter().enumerate() {
        for (segment_idx, segment) in rollout.trajectory.iter().enumerate() {
            let role = segment.role.trim();
            anyhow::ensure!(
                !role.is_empty(),
                "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: role must be non-empty"
            );
            match segment.kind {
                TurnKind::Action => anyhow::ensure!(
                    role.eq_ignore_ascii_case("assistant"),
                    "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: Action segment must use role \"assistant\", got {:?}",
                    segment.role
                ),
                TurnKind::Observation => anyhow::ensure!(
                    role.eq_ignore_ascii_case("tool"),
                    "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: Observation segment must use role \"tool\", got {:?}",
                    segment.role
                ),
                TurnKind::Context => {}
            }
        }
    }
    Ok(())
}

pub(super) fn validate_grpo_dry_run_masks(
    group: &TokenizedGrpoGroup,
    group_idx: usize,
    line_no: usize,
) -> Result<()> {
    for (completion_idx, completion) in group.completions.iter().enumerate() {
        anyhow::ensure!(
            completion.action_mask.len() == completion.input_ids.len(),
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} action_mask length {} does not match input_ids length {}",
            completion.action_mask.len(),
            completion.input_ids.len()
        );
        anyhow::ensure!(
            completion.env_mask.len() == completion.input_ids.len(),
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} env_mask length {} does not match input_ids length {}",
            completion.env_mask.len(),
            completion.input_ids.len()
        );
        let action_tokens = completion
            .action_mask
            .iter()
            .filter(|&&active| active)
            .count();
        anyhow::ensure!(
            action_tokens > 0,
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} has empty action_mask"
        );
    }
    Ok(())
}

pub(super) fn parse_grpo_jsonl_group_line(line: &str, line_no: usize) -> Result<Option<GrpoGroup>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<GrpoGroup>(trimmed)
        .map(Some)
        .with_context(|| format!("parse GRPO JSONL group at line {line_no}"))
}

pub(super) fn jsonl_byte_progress(total_bytes: u64, offset: u64) -> (usize, usize, f32) {
    let total = total_bytes.max(1);
    let clamped = offset.min(total);
    let total_steps = total.min(usize::MAX as u64).max(1) as usize;
    let step = clamped.min(usize::MAX as u64).max(1) as usize;
    let progress = (clamped as f64 / total as f64).min(0.999) as f32;
    (step, total_steps, progress)
}

/// Page size used by the GRPO shared-prompt-prefix paged cache. Matches the
/// production server / bench setting so the same FA fast paths fire (#1082:
/// 16 -> 64 so each FA2 kBlockN=64 tile is one page; keeps parity with
/// `DEFAULT_BLOCK_SIZE`).
pub(super) const GRPO_REF_PAGED_BLOCK_SIZE: usize = 64;

pub(super) fn grpo_shared_prefix_tile_tokens(
    streaming_prefill: StreamingPrefillExecutionPolicy,
    seq_len: usize,
) -> Result<Option<usize>> {
    if !streaming_prefill.enabled_for(seq_len) {
        return Ok(None);
    }
    let tile_tokens = streaming_prefill.base_tile_tokens_for(seq_len);
    anyhow::ensure!(
        tile_tokens > 0,
        "GRPO shared-prefix streaming tile size must be greater than zero"
    );
    Ok((tile_tokens < seq_len).then_some(tile_tokens))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn model_forward_paged_normed_hidden_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    paged_cache: &PagedKvCacheKt,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    ema_ref_lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    anyhow::ensure!(
        !token_ids.is_empty(),
        "GRPO shared-prefix paged forward requires at least one token"
    );
    let Some(tile_tokens) = grpo_shared_prefix_tile_tokens(streaming_prefill, token_ids.len())?
    else {
        return model_forward_paged_normed_hidden(
            backend,
            token_ids,
            weights,
            model_config,
            paged_cache,
            block_table,
            start_pos,
            linear_state,
            ema_ref_lora,
        );
    };

    let mut tile_hidden = Vec::with_capacity(token_ids.len().div_ceil(tile_tokens));
    let mut cursor = 0usize;
    while cursor < token_ids.len() {
        let end = (cursor + tile_tokens).min(token_ids.len());
        tile_hidden.push(
            model_forward_paged_normed_hidden(
                backend,
                &token_ids[cursor..end],
                weights,
                model_config,
                paged_cache,
                block_table,
                start_pos + cursor,
                linear_state.as_deref_mut(),
                ema_ref_lora,
            )
            .with_context(|| {
                format!(
                    "GRPO shared-prefix streaming tile [{cursor}, {end}) of {}",
                    token_ids.len()
                )
            })?,
        );
        cursor = end;
    }
    let refs: Vec<&Tensor> = tile_hidden.iter().collect();
    cat_tensors(&refs, 1).context("GRPO shared-prefix: concatenate streaming hidden tiles")
}

/// Compute the reference-policy log probs for every completion in a GRPO
/// group, sharing the prompt-prefix forward across all completions.
///
/// All completions in a GRPO group share an identical prompt prefix
/// (tokenize_grpo_group computes `prompt_ids` once and reuses it). The legacy
/// path ran `model_forward_no_head` over `[prompt | completion]` once per
/// completion (4× per group at default settings), redoing the
/// O(prompt_len²) full-attention and O(prompt_len) GDN work each time.
///
/// This helper runs the prompt forward exactly once via the paged path,
/// snapshots the GDN linear state at `prompt_len`, then forwards only each
/// completion's tokens with `start_pos == prompt_len`. The paged cache
/// transparently feeds the prompt's K/V into the full-attention layers as
/// "prefix history", so FlashAttention prefill-with-prefix runs at
/// O(comp_len × prompt_len) instead of O(prompt_len²) per completion. The
/// GDN linear state is restored from the post-prompt snapshot before each
/// completion so its recurrent state starts from the correct point.
///
/// The shared-prefix path requires no gradient (this is the reference
/// forward), so the paged inference kernels are used directly. Total
/// reference-forward attention work drops from `n_comp × (P + C)²` to
/// `P² + n_comp × C × (P + C)` — roughly a `n_comp×` speedup when
/// `C << P`, which is the production regime for pi-compaction.
///
/// All returned log-prob tensors are detached (the ratio computation in
/// `grpo_loss` only needs the policy side to track gradients).
pub(super) fn compute_ref_log_probs_shared_prefix(
    backend: &dyn BackendRuntime,
    tgroup: &TokenizedGrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    ema_ref_lora: Option<&LoraWeights>,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Vec<Tensor>> {
    if tgroup.completions.is_empty() {
        return Ok(Vec::new());
    }

    let first = &tgroup.completions[0];
    let prompt_len = first.prompt_token_count;
    if prompt_len < 1 {
        anyhow::bail!("GRPO shared-prefix ref forward requires prompt_len >= 1, got {prompt_len}");
    }

    // Validate the prefix invariant — every completion must share the same
    // prompt prefix or the shared-prefix path is unsound.
    for (idx, comp) in tgroup.completions.iter().enumerate() {
        let comp_prompt_len = comp.prompt_token_count;
        anyhow::ensure!(
            comp_prompt_len == prompt_len,
            "GRPO completions have different prompt lengths ({prompt_len} vs {comp_prompt_len} for completion {idx})"
        );
        anyhow::ensure!(
            comp.input_ids.len() >= prompt_len,
            "completion {idx} input_ids shorter than prompt_len {prompt_len}"
        );
        anyhow::ensure!(
            comp.input_ids[..prompt_len] == first.input_ids[..prompt_len],
            "completion {idx} prompt token ids differ from completion 0",
        );
    }

    let prompt_ids: &[u32] = &first.input_ids[..prompt_len];
    let max_total = tgroup
        .completions
        .iter()
        .map(|c| c.input_ids.len())
        .max()
        .unwrap_or(prompt_len)
        .max(prompt_len);

    let dtype = match model_config.dtype {
        kiln_core::config::DType::BF16 => DType::BF16,
        kiln_core::config::DType::FP16 => DType::F16,
        kiln_core::config::DType::FP32 => DType::F32,
    };

    let num_blocks = max_total.div_ceil(GRPO_REF_PAGED_BLOCK_SIZE);
    // (#1082) The candle `PagedKvCache::new` took a candle device; its kt twin
    // `PagedKvCacheKt::new` allocates its pools on the model's runtime `Device`.
    // `device` is a kt `Device` (Copy) — pass it through so the pools land on
    // the same device as the model's tensors (CPU model → CPU pools, etc.).
    let paged_cache = PagedKvCacheKt::new(
        model_config.num_full_attention_layers,
        num_blocks,
        GRPO_REF_PAGED_BLOCK_SIZE,
        model_config.num_kv_heads,
        model_config.head_dim,
        dtype,
        *device,
    )
    .context("GRPO shared-prefix: build PagedKvCacheKt")?;
    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }

    let mut linear_state = LinearAttentionState::new(model_config, device)
        .context("GRPO shared-prefix: build LinearAttentionState")?;

    // Phase 1: prompt forward — populates the paged cache for positions
    // [0..prompt_len) and advances the GDN linear state past the prompt.
    let prompt_hidden = model_forward_paged_normed_hidden_with_policy(
        backend,
        prompt_ids,
        weights,
        model_config,
        &paged_cache,
        &block_table,
        0,
        Some(&mut linear_state),
        ema_ref_lora,
        streaming_prefill,
    )
    .context("GRPO shared-prefix: prompt forward")?;

    // The position that predicts the first completion token (input_ids[prompt_len])
    // is prompt_len - 1. Capture its normed hidden state as a detached, stable
    // owning tensor so the rest of the prompt_hidden allocation can be freed.
    // (#1082) `prompt_hidden` is kt (kt-flipped `model_forward_paged_normed_hidden`);
    // the downstream GRPO ref log-prob math (`cat_tensors`,
    // `chunked_log_probs_for_completion`) is now kt-native too, so keep it kt.
    let last_prompt_hidden = prompt_hidden
        .narrow(1, prompt_len - 1, 1)
        .context("GRPO shared-prefix: narrow last prompt hidden")?
        .contiguous()
        .context("GRPO shared-prefix: contiguous last prompt hidden")?;
    drop(prompt_hidden);

    // Snapshot the GDN linear state at end-of-prompt so each completion can
    // restore from this point before running its own forward.
    let linear_snap = linear_state
        .snapshot()
        .context("GRPO shared-prefix: snapshot linear state")?;

    let mut ref_log_probs_per_comp = Vec::with_capacity(tgroup.completions.len());
    for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
        let full_len = comp.input_ids.len();
        let comp_len = full_len - prompt_len;
        if comp_len == 0 {
            // No completion tokens — placeholder zero tensor matches the legacy
            // path's behaviour for empty active completions.
            ref_log_probs_per_comp.push(zeros_f32_on(1, device)?.detach());
            continue;
        }

        // Restore GDN state to end-of-prompt. The paged-cache full-attn K/V
        // for positions [0..prompt_len) is preserved (writes only target
        // start_pos..start_pos+seq_len, never the prefix), so the cache is
        // implicitly reset by passing start_pos = prompt_len. Each completion
        // overwrites the cache slots at [prompt_len..prompt_len+comp_len), but
        // that region is throw-away — we never read another completion's K/V.
        linear_state.restore_from(&linear_snap).with_context(|| {
            format!("GRPO shared-prefix: restore linear state for completion {comp_idx}")
        })?;

        let completion_ids = &comp.input_ids[prompt_len..];

        let comp_hidden = {
            let kt = model_forward_paged_normed_hidden_with_policy(
                backend,
                completion_ids,
                weights,
                model_config,
                &paged_cache,
                &block_table,
                prompt_len,
                Some(&mut linear_state),
                ema_ref_lora,
                streaming_prefill,
            )
            .with_context(|| format!("GRPO shared-prefix: completion {comp_idx} forward"))?;
            // (#1082) kt forward output flows straight into the kt log-prob path.
            kt
        };

        // Build the "active hidden" tensor: aligned with the completion tokens
        // we want to compute log-probs for. Following the legacy convention in
        // `selected_log_probs_from_normed_hidden_chunked`, hidden[i] is the
        // normed pre-LM-head state that predicts the token at position i+1.
        //
        //   active_hidden[0]            = last prompt hidden (predicts input_ids[prompt_len])
        //   active_hidden[1..comp_len]  = comp_hidden[0..comp_len-1]
        //                                 (predicts input_ids[prompt_len+1..full_len])
        //
        // Shape: [1, comp_len, hidden_size]. comp_hidden[comp_len-1] is dropped
        // because there's no token after the last completion token to predict.
        let active_hidden = if comp_len == 1 {
            last_prompt_hidden.clone()
        } else {
            let comp_prefix = comp_hidden.narrow(1, 0, comp_len - 1).with_context(|| {
                format!("GRPO shared-prefix: narrow comp prefix completion {comp_idx}")
            })?;
            cat_tensors(&[&last_prompt_hidden, &comp_prefix], 1).with_context(|| {
                format!("GRPO shared-prefix: concat active hidden completion {comp_idx}")
            })?
        };
        drop(comp_hidden);

        // (#1082) `embed_tokens_t` and the chunked log-prob helper are both
        // kt now; pass the kt head weight straight through.
        let log_probs = chunked_log_probs_for_completion(
            &active_hidden,
            &weights.embed_tokens_t,
            completion_ids,
            DEFAULT_CHUNK_SIZE,
            device,
        )
        .with_context(|| format!("GRPO shared-prefix: chunked log-probs completion {comp_idx}"))?;

        ref_log_probs_per_comp.push(log_probs.detach());
    }

    Ok(ref_log_probs_per_comp)
}

/// Compute per-target-token log probs from a pre-shifted normed-hidden tensor.
///
/// `active_hidden` is `[1, n_targets, hidden_size]` and is assumed to be the
/// post-final-RMSNorm hidden state at exactly the positions that need a
/// log-prob (one row per target token). `target_ids` is the actual token id
/// each row predicts. This is the chunked-softmax core that
/// [`selected_log_probs_from_normed_hidden_chunked`] also uses, but without
/// the position-selection / shift bookkeeping (the caller has already aligned
/// rows with targets).
pub(super) fn chunked_log_probs_for_completion(
    active_hidden: &Tensor,
    head_t: &Tensor,
    target_ids: &[u32],
    chunk_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let n_targets = target_ids.len();
    if n_targets == 0 {
        return zeros_f32_on(1, device);
    }
    if chunk_size == 0 {
        anyhow::bail!("chunked_log_probs_for_completion chunk_size must be > 0");
    }

    let dims = active_hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != n_targets {
        anyhow::bail!(
            "active_hidden must have shape [1, n_targets={n_targets}, hidden_size], got {:?}",
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

    let hidden_2d = active_hidden.squeeze(0)?.to_f32_dtype()?;
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
            let logits_chunk = hidden_2d.matmul(&head_chunk)?;
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
                target_ids,
                chunk_start,
                chunk_len,
                vocab_size,
                device,
                "chunked_log_probs_for_completion",
            )?;
            correct_logits = Some(match correct_logits.as_ref() {
                Some(prev) => (prev + chunk_correct)?.detach(),
                None => chunk_correct.detach(),
            });
        }
        synchronize_tail_chunk("synchronize chunked_log_probs_for_completion")?;
        chunk_start = chunk_end;
    }

    let running_max = running_max.context("vocab_size was zero")?;
    let running_sumexp = running_sumexp.context("vocab_size was zero")?;
    let correct_logits = correct_logits.context("vocab_size was zero")?;
    let log_sum_exp = (running_max + running_sumexp.log()?)?;
    Ok((correct_logits - log_sum_exp)?.squeeze(1)?)
}

pub(super) fn observe_grpo_policy_audit_completion(
    policy_audit: &mut crate::train_receipt::GrpoPolicyAuditAccumulator,
    policy_log_probs: &Tensor,
    behavior_log_probs: Option<&[f32]>,
    kl_reference_log_probs: Option<&Tensor>,
    loss_params: GrpoLossParams,
    behavior_source: Option<&crate::train_receipt::GrpoRecordedBehaviorSourceObservation>,
) -> Result<()> {
    let policy_log_probs_host = policy_log_probs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()?;
    let kl_reference_log_probs_host = kl_reference_log_probs
        .map(|reference| {
            reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_device(cpu_device())?
                .to_vec1::<f32>()
        })
        .transpose()?;
    policy_audit.observe_policy_values(
        &policy_log_probs_host,
        behavior_log_probs,
        kl_reference_log_probs_host.as_deref(),
        loss_params.is_level,
        loss_params.clip_low,
        loss_params.clip_high,
        loss_params.kl_estimator,
        loss_params.entropy_aware_kl_quantile,
    )?;
    if let Some(source) = behavior_source {
        policy_audit.observe_recorded_behavior_source(source);
    }
    Ok(())
}

// Non-GPU builds: the cfg'd `unreachable!` arm below makes the rest of the
// body unreachable (callers bail on the group-entry capability check), so
// the structural `unreachable_code` it triggers is allowed for that feature
// set only; under cuda/metal/vulkan/rocm the code after the arm is live.
#[cfg_attr(
    not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )),
    allow(unreachable_code)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn train_tokenized_grpo_group_with_grad_norms(
    backend: &dyn BackendRuntime,
    tgroup: &TokenizedGrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    // (#1082) `&mut` — the optimizer step mutates each LoRA `Parameter`'s
    // kt master in place.
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    opt_state: Option<&mut OptimizerState>,
    grad_norms: &mut crate::train_receipt::LoraGradNormAccumulator,
    lora_grad_index: &LoraGradNormIndex,
    // Observed only in the GPU-feature step body; non-GPU builds bail before it.
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_variables)
    )]
    policy_audit: &mut crate::train_receipt::GrpoPolicyAuditAccumulator,
    // Optional EMA-snapshot LoRA used as the KL reference when
    // `config.kl_reference_policy == KlReferencePolicy::Ema`. None means the
    // KL-reference forward runs without LoRA (`BasePerStep`) or is skipped.
    ema_ref_lora: Option<&LoraWeights>,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
    streaming_prefill_policy: StreamingPrefillExecutionPolicy,
) -> Result<GrpoGroupStepReport> {
    validate_tokenized_behavior_policy(tgroup, config.behavior_policy)
        .context("validate GRPO behavior-policy provenance")?;
    let skip_kl_reference = !config.kl_penalty_enabled()
        || matches!(config.kl_reference_policy, KlReferencePolicy::None);

    // Learning-rate resolution is request-local and independent of runtime
    // execution policy.
    let learning_rate = config.effective_learning_rate();
    let advantages = compute_advantages(&tgroup.rewards, config.advantage_mode);
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut group_loss_sum = 0.0;

    // Active token counts per completion (matches the next-token-shift convention
    // used by token_log_probs and the analytic tail: action_mask[1..]).
    let per_comp_active: Vec<usize> = tgroup
        .completions
        .iter()
        .map(|c| {
            c.action_mask
                .get(1..)
                .map_or(0, |m| m.iter().filter(|&&v| v).count())
        })
        .collect();
    let group_total_active: usize = per_comp_active.iter().sum();
    if group_total_active == 0 {
        return Ok(GrpoGroupStepReport {
            loss: 0.0,
            echo_env_ce: None,
        });
    }
    ensure_tape_forward_backward_supported("GRPO group step", weights, backend)?;
    let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
    let group_max_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0);
    let checkpoint_segments = segments.map_or(0, |segs| segs.len());
    let streaming_tile_tokens = streaming_prefill_policy.base_tile_tokens();
    let streaming_prefill = streaming_prefill_policy.enabled_for(group_max_seq_len);

    let token_level = matches!(config.loss_aggregation, LossAggregation::TokenLevel);
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut group_accum: GradMap = HashMap::new();
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut group_echo_ce_sum = 0.0f64;
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_mut)
    )]
    let mut group_echo_ce_weight = 0usize;

    // Shared-prefix optimization: when the reference policy is active and the
    // group has more than one completion, run the prompt forward exactly once
    // (paged path) and reuse its K/V + GDN state across all completions. The
    // legacy per-completion `model_forward_no_head` loop is kept as the
    // fallback below when (a) reference is skipped, or (b) the group has a
    // single completion (no sharing to be had), or (c) the shared-prefix path
    // is explicitly disabled.
    let use_shared_prefix = !skip_kl_reference
        && tgroup.completions.len() > 1
        // The paged shared-prefix reference path still mixes a host
        // broadcast temporary into a fully resident Vulkan graph. Keep the
        // exact per-completion path until paged Vulkan reference parity is
        // independently qualified.
        && !matches!(device, Device::Vulkan(_))
        && config.shared_prefix_reference;
    let shared_prefix_log_probs: Option<Vec<Tensor>> = if use_shared_prefix {
        let started = Instant::now();
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            checkpoint_segments,
            streaming_prefill,
            streaming_tile_tokens,
            "GRPO ref forward start"
        );
        let log_probs = compute_ref_log_probs_shared_prefix(
            backend,
            tgroup,
            weights,
            model_config,
            ema_ref_lora,
            device,
            streaming_prefill_policy,
        )
        .context("GRPO shared-prefix reference forward")?;
        let elapsed = started.elapsed();
        if let Some(t) = timings.as_deref_mut() {
            t.add_reference_forward(elapsed);
        }
        tracing::info!(
            n_completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            checkpoint_segments,
            streaming_prefill,
            streaming_tile_tokens,
            elapsed_ms = elapsed.as_millis() as u64,
            "GRPO ref forward end"
        );
        Some(log_probs)
    } else {
        None
    };

    for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
        let num_active = per_comp_active[comp_idx];
        if num_active == 0 {
            continue;
        }
        let loss_normalizer = if token_level {
            1.0 / group_total_active as f64
        } else {
            1.0 / num_active as f64
        };
        // Fed only into the GPU-feature kt-tape dispatch below.
        #[cfg_attr(
            not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )),
            allow(unused_variables)
        )]
        let loss_params =
            GrpoLossParams::from_config(config, advantages[comp_idx], loss_normalizer);
        let comp_env_count = comp
            .env_mask
            .get(1..)
            .map_or(0, |m| m.iter().filter(|&&v| v).count());
        #[cfg_attr(
            not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )),
            allow(unused_mut, unused_variables)
        )]
        let mut comp_echo_env_ce: Option<f64> = None;

        let kl_reference_log_probs = if skip_kl_reference {
            // KL is disabled, so the placeholder is never inspected by the
            // loss. Behavior-policy probabilities are prepared separately.
            zeros_f32_on(num_active, device)?.detach()
        } else if let Some(shared) = shared_prefix_log_probs.as_ref() {
            // The shared-prefix output is one log-prob per completion-span
            // position (predicting input_ids[prompt_len + i] for
            // i in 0..comp_len). For trajectory-aware rollouts that
            // include Observation segments, we need only the Action
            // positions to match policy_log_probs's shape; for legacy
            // single-turn rollouts the action_mask is true at every
            // completion-span position and this filter is a no-op.
            let span = &shared[comp_idx];
            let comp_prompt_len = comp.prompt_token_count;
            let active_indices: Vec<u32> = (0..span.dim(0)?)
                .filter(|&i| {
                    comp.action_mask
                        .get(comp_prompt_len + i)
                        .copied()
                        .unwrap_or(false)
                })
                .map(|i| i as u32)
                .collect();
            if active_indices.len() == span.dim(0)? {
                // Legacy fast-path: every span position is active. No need
                // to allocate an indices tensor or do an index_select.
                span.clone()
            } else if active_indices.is_empty() {
                // Defensive: shouldn't happen (num_active > 0 was checked
                // above), but handle it cleanly.
                zeros_f32_on(num_active, device)?.detach()
            } else {
                let n_idx = active_indices.len();
                let indices = Tensor::from_vec_on(*device, active_indices, vec![n_idx])?;
                span.index_select(&indices, 0)?.detach()
            }
        } else {
            let ref_started = Instant::now();
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                checkpoint_segments,
                streaming_prefill = streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                streaming_tile_tokens,
                "GRPO ref forward start"
            );
            let mut ref_linear_state = LinearAttentionState::new(model_config, device)?;
            // BasePerStep (None ema_ref_lora) → base model (no LoRA).
            // Ema (Some(snapshot)) → frozen snapshot of the LoRA from a
            // prior training point.
            // (#1082) `model_forward_no_head` and
            // `selected_log_probs_from_normed_hidden_chunked` are both kt-native;
            // the kt hidden + kt `embed_tokens_t` head weight flow through
            // directly (no candle bridge).
            let ref_hidden = model_forward_no_head_with_policy(
                backend,
                &comp.input_ids,
                weights,
                model_config,
                Some(&mut ref_linear_state),
                ema_ref_lora,
                streaming_prefill_policy,
            )
            .context("GRPO reference forward pass")?
            .contiguous()
            .context("GRPO ref hidden contiguous")?;
            let ref_log_probs = selected_log_probs_from_normed_hidden_chunked(
                &ref_hidden,
                &weights.embed_tokens_t,
                &comp.input_ids,
                &comp.action_mask,
                DEFAULT_CHUNK_SIZE,
            )?
            .detach();
            if let Some(t) = timings.as_deref_mut() {
                t.add_reference_forward(ref_started.elapsed());
            }
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                checkpoint_segments,
                streaming_prefill = streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                streaming_tile_tokens,
                elapsed_ms = ref_started.elapsed().as_millis() as u64,
                "GRPO ref forward end"
            );
            ref_log_probs
        };

        let behavior_log_probs = match config.behavior_policy {
            BehaviorPolicy::NoImportanceCorrection => zeros_f32_on(num_active, device)?.detach(),
            BehaviorPolicy::Recorded => {
                let values = comp.recorded_behavior_log_probs.as_ref().with_context(|| {
                    format!("completion {comp_idx} is missing behavior-policy log-probabilities")
                })?;
                Tensor::from_vec_on(*device, values.clone(), vec![num_active])?.detach()
            }
        };

        // (#1082 candle-drop) GRPO per-completion step is now UNCONDITIONALLY
        // kt tape-authoritative. The candle gradient-checkpointed GRPO reverse
        // (`checkpointed_grpo_forward_backward` + analytic ECHO tail), the
        // candle tape-bridge producer, and the inline candle `loss.backward()`
        // path are all DELETED. ECHO env-CE has no kt tape root, so an
        // ECHO-active GRPO step is not supported on the kt-only path (the
        // candle ECHO term was a candle-authoritative feature dropped in the
        // candle drop). `grpo_step_forward_backward_tape_authoritative_kt`
        // returns `GradSource::Kt`, consumed kt-native by the dispatchers.
        // Assigned/read only inside the GPU-feature branch (and the code after
        // the non-GPU `unreachable!` guard); kept declared here so both builds
        // type-check.
        #[cfg_attr(
            not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )),
            allow(unused_variables)
        )]
        let loss_val: f64;
        let (grads, policy_log_probs): (GradSource, Tensor) = {
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            {
                // ECHO env-CE spec (resurrection PR2): built when the term
                // is enabled and this completion actually has env rows; the
                // fused loss roots add λ·env_CE to the value and the matching
                // constant-coefficient rows to the gradient.
                let echo_env_spec =
                    if config.loss.echo_enabled() && comp_env_count > 0 && comp.total_obs_len > 0 {
                        Some(crate::grpo_tape_shim::EchoEnvSpec {
                            env_mask: comp.env_mask.clone(),
                            total_obs_len: comp.total_obs_len,
                            lambda: config.loss.echo_lambda(),
                        })
                    } else {
                        None
                    };
                let (lv, env_ce, kt_grads, policy_log_probs) = if let Some(segs) = segments {
                    let step_started = Instant::now();
                    let out = checkpointed_grpo_forward_backward_tape_authoritative_kt(
                        backend,
                        &comp.input_ids,
                        weights,
                        model_config,
                        params,
                        &comp.action_mask,
                        &behavior_log_probs,
                        &kl_reference_log_probs,
                        loss_params,
                        segs,
                        device,
                        echo_env_spec.as_ref(),
                        config.loss.no_policy_loss,
                        config.detect_anomaly,
                        streaming_prefill_policy,
                    )?;
                    let step_elapsed = step_started.elapsed();
                    if let Some(t) = timings.as_deref_mut() {
                        t.add_backward(step_elapsed);
                    }
                    tracing::info!(
                        comp_idx,
                        seq_len = comp.input_ids.len(),
                        action_tokens = num_active,
                        env_tokens = comp_env_count,
                        checkpoint_segments,
                        streaming_prefill =
                            streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                        streaming_tile_tokens,
                        elapsed_ms = step_elapsed.as_millis() as u64,
                        "GRPO step end (checkpointed tape-authoritative kt)"
                    );
                    out
                } else {
                    grpo_step_forward_backward_tape_authoritative_kt(
                        backend,
                        &comp.input_ids,
                        weights,
                        model_config,
                        params,
                        &comp.action_mask,
                        &behavior_log_probs,
                        &kl_reference_log_probs,
                        loss_params,
                        device,
                        comp_idx,
                        num_active,
                        comp_env_count,
                        streaming_tile_tokens,
                        checkpoint_segments,
                        timings.as_deref_mut(),
                        echo_env_spec.as_ref(),
                        config.loss.no_policy_loss,
                        config.detect_anomaly,
                        streaming_prefill_policy,
                    )?
                };
                loss_val = lv;
                comp_echo_env_ce = env_ce;
                (GradSource::Kt(kt_grads), policy_log_probs)
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            {
                // The group-entry capability check already bailed without a GPU
                // backend feature. This arm keeps `loss_val` definitely assigned.
                let _ = (
                    &behavior_log_probs,
                    &kl_reference_log_probs,
                    num_active,
                    comp_env_count,
                    comp_idx,
                );
                unreachable!("GRPO kt path requires a GPU backend feature");
            }
        };
        anyhow::ensure!(
            policy_log_probs.elem_count() == num_active,
            "GRPO loss returned {} selected policy log-probabilities for {num_active} active tokens",
            policy_log_probs.elem_count()
        );
        let behavior_log_probs_host = match config.behavior_policy {
            BehaviorPolicy::NoImportanceCorrection => None,
            BehaviorPolicy::Recorded => comp.recorded_behavior_log_probs.as_deref(),
        };
        observe_grpo_policy_audit_completion(
            policy_audit,
            &policy_log_probs,
            behavior_log_probs_host,
            (!skip_kl_reference).then_some(&kl_reference_log_probs),
            loss_params,
            comp.recorded_behavior_source.as_ref(),
        )
        .with_context(|| format!("record GRPO policy metrics for completion {comp_idx}"))?;
        if token_level {
            // Cross-completion grad accumulation into the kt `GradMap`
            // (keyed by `Parameter::tensor_id()`).
            accumulate_grads_dispatch(&mut group_accum, &grads, params)?;
        } else {
            observe_lora_grad_norms_dispatch(grad_norms, params, &grads)?;
            let optimizer_started = Instant::now();
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                optimizer = ?config.optimizer,
                "GRPO optimizer start"
            );
            optimizer_step_dispatch(
                backend,
                params,
                &grads,
                learning_rate,
                config.optimizer,
                opt_state.as_deref_mut(),
            )?;
            if let Some(t) = timings.as_deref_mut() {
                t.add_optimizer(optimizer_started.elapsed());
            }
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                optimizer = ?config.optimizer,
                elapsed_ms = optimizer_started.elapsed().as_millis() as u64,
                "GRPO optimizer end"
            );
        }

        group_loss_sum += loss_val;
        if let Some(env_ce) = comp_echo_env_ce
            && comp.total_obs_len > 0
        {
            group_echo_ce_sum += env_ce * comp.total_obs_len as f64;
            group_echo_ce_weight = group_echo_ce_weight.saturating_add(comp.total_obs_len);
        }
    }

    if token_level && !group_accum.is_empty() {
        observe_lora_grad_norms_from_map(grad_norms, lora_grad_index, &group_accum)?;
        let optimizer_started = Instant::now();
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            optimizer = ?config.optimizer,
            "GRPO optimizer start"
        );
        optimizer_step_from_map(
            backend,
            params,
            &group_accum,
            learning_rate,
            config.optimizer,
            opt_state,
        )?;
        if let Some(t) = timings {
            t.add_optimizer(optimizer_started.elapsed());
        }
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            optimizer = ?config.optimizer,
            elapsed_ms = optimizer_started.elapsed().as_millis() as u64,
            "GRPO optimizer end"
        );
    }

    let loss = if tgroup.completions.is_empty() {
        0.0
    } else if token_level {
        // Per-completion loss_val is already its share of the group-level mean
        // (each was scaled by 1/group_total_active). Sum across completions
        // gives the true group-level mean.
        group_loss_sum
    } else {
        group_loss_sum / tgroup.completions.len() as f64
    };
    let echo_env_ce = if group_echo_ce_weight > 0 {
        Some(group_echo_ce_sum / group_echo_ce_weight as f64)
    } else {
        None
    };
    Ok(GrpoGroupStepReport { loss, echo_env_ce })
}

// (#1082) `merge_grad_maps` removed: its sole caller was the candle
// gradient-checkpointed GRPO path (`checkpointed_grpo_forward_backward`),
// which was deleted in the candle drop. The kt-only GRPO token-level path
// accumulates directly via `accumulate_grads_dispatch`.

pub(super) struct LoraGradNormIndex {
    // (#1082) keyed by each LoRA `Parameter::tensor_id()` (kt).
    pub(super) modules_by_param: HashMap<KtTensorId, &'static str>,
}

impl LoraGradNormIndex {
    pub(super) fn new(params: &TrainableLoraParams) -> Self {
        Self {
            modules_by_param: params
                .all_params_with_modules()
                .into_iter()
                .map(|entry| (entry.param.tensor_id(), entry.module))
                .collect(),
        }
    }
}

pub(super) fn observe_lora_grad_norms_from_map(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    index: &LoraGradNormIndex,
    grads: &GradMap,
) -> Result<()> {
    let mut sum_sq_by_module: BTreeMap<&'static str, f64> = BTreeMap::new();
    for (id, grad) in grads {
        if let Some(module) = index.modules_by_param.get(id).copied() {
            accumulate_lora_grad_sum_sq(&mut sum_sq_by_module, module, grad)?;
        }
    }
    observe_lora_grad_module_norms(accumulator, sum_sq_by_module);
    Ok(())
}

/// (#1082) kt-native LoRA grad-norm observer: reads each LoRA
/// `Parameter`'s gradient from a kt-native [`kiln_autograd::GradStore`]
/// (keyed by `Parameter::tensor_id()`) and accumulates its squared L2
/// norm per module. The per-param norm is computed KT-NATIVELY via
/// `train_receipt::tensor_l2_norm_kt` (cast-to-F32 on-device + single D2H
/// scalar readback) — NO full-tensor kt->candle grad copy.
pub(crate) fn observe_lora_grad_norms_from_kt_grad_store(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
) -> Result<()> {
    let mut sum_sq_by_module: BTreeMap<&'static str, f64> = BTreeMap::new();
    for entry in params.all_params_with_modules() {
        if let Some(kt_grad) = grads.get(entry.param.tensor_id()) {
            let norm = crate::train_receipt::tensor_l2_norm_kt(kt_grad).with_context(|| {
                format!("compute LoRA grad l2 norm (kt) for module {}", entry.module)
            })?;
            if norm.is_finite() {
                *sum_sq_by_module.entry(entry.module).or_insert(0.0) += norm * norm;
            } else {
                let value_summary = summarize_sft_debug_values(kt_grad)
                    .map(|(_, summary)| summary)
                    .unwrap_or_else(|e| format!("stats_error={e:#}"));
                tracing::warn!(
                    layer = entry.layer_idx,
                    module = entry.module,
                    matrix = entry.matrix,
                    tensor_id = entry.param.tensor_id().as_raw(),
                    dtype = ?kt_grad.dtype(),
                    shape = ?kt_grad.shape(),
                    device = %kt_grad.device(),
                    value_summary,
                    "skipping non-finite LoRA grad norm sample (kt)"
                );
            }
        }
    }
    observe_lora_grad_module_norms(accumulator, sum_sq_by_module);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_lora_gradient_metadata(
    context: &str,
    leaf: &str,
    expected_shape: &[usize],
    expected_dtype: KtDType,
    expected_device: kiln_tensor::Device,
    observed_shape: &[usize],
    observed_dtype: KtDType,
    observed_device: kiln_tensor::Device,
) -> Result<()> {
    anyhow::ensure!(
        observed_shape == expected_shape,
        "{context}: LoRA gradient shape mismatch for {leaf}: expected={expected_shape:?} observed={observed_shape:?}"
    );
    anyhow::ensure!(
        observed_dtype == expected_dtype,
        "{context}: LoRA gradient dtype mismatch for {leaf}: expected={expected_dtype:?} observed={observed_dtype:?}"
    );
    anyhow::ensure!(
        observed_device == expected_device,
        "{context}: LoRA gradient device mismatch for {leaf}: expected={expected_device} observed={observed_device}"
    );
    Ok(())
}

pub(super) fn validate_lora_gradient_tensor(
    context: &str,
    entry: &LoraParamRef<'_>,
    grad: &KtTensor,
    check_finite_values: bool,
) -> Result<()> {
    let id = entry.param.tensor_id();
    let leaf = format!(
        "layer={} module={} matrix={} tensor_id={}",
        entry.layer_idx, entry.module, entry.matrix, id
    );
    let master = entry.param.backward_storage().ok_or_else(|| {
        anyhow::anyhow!("{context}: configured trainable LoRA leaf has no master storage: {leaf}")
    })?;
    validate_lora_gradient_metadata(
        context,
        &leaf,
        master.shape(),
        entry.param.amp_policy().backward_compute_dtype,
        master.device(),
        grad.shape(),
        grad.dtype(),
        grad.device(),
    )?;
    if !check_finite_values {
        return Ok(());
    }

    let fast_finite = grad
        .all_finite()
        .with_context(|| format!("{context}: finite scan failed for LoRA gradient {leaf}"))?;
    if fast_finite {
        return Ok(());
    }

    let (cpu_finite, value_summary) = summarize_sft_debug_values(grad)
        .with_context(|| format!("{context}: CPU-confirming finite scan failed for {leaf}"))?;
    if cpu_finite {
        tracing::warn!(
            layer = entry.layer_idx,
            module = entry.module,
            matrix = entry.matrix,
            tensor_id = id.as_raw(),
            dtype = ?grad.dtype(),
            shape = ?grad.shape(),
            device = %grad.device(),
            value_summary,
            "{context}: backend finite reducer reported a non-finite LoRA gradient but CPU confirmation was finite"
        );
        return Ok(());
    }

    anyhow::bail!(
        "{context}: non-finite LoRA gradient {leaf} dtype={:?} shape={:?} device={} {}",
        grad.dtype(),
        grad.shape(),
        grad.device(),
        value_summary
    )
}

#[derive(Clone, Copy)]
pub(super) enum ExpectedLoraGradientSet {
    WholeAdapter,
    CheckpointLayerRange,
}

pub(super) fn validate_exact_lora_gradients<'p, 'g>(
    expected_entries: impl IntoIterator<Item = LoraParamRef<'p>>,
    observed_gradients: impl IntoIterator<Item = (KtTensorId, &'g KtTensor)>,
    context: &str,
    expected_set: ExpectedLoraGradientSet,
    check_finite_values: bool,
) -> Result<()> {
    let mut expected = BTreeMap::new();
    for entry in expected_entries {
        let id = entry.param.tensor_id();
        if let Some(previous) = expected.insert(id, entry) {
            anyhow::bail!(
                "{context}: duplicate configured LoRA tensor_id={id}: first=layer={} module={} matrix={} second=layer={} module={} matrix={}",
                previous.layer_idx,
                previous.module,
                previous.matrix,
                expected[&id].layer_idx,
                expected[&id].module,
                expected[&id].matrix
            );
        }
    }
    if matches!(expected_set, ExpectedLoraGradientSet::WholeAdapter) {
        anyhow::ensure!(
            !expected.is_empty(),
            "{context}: configured trainable LoRA leaf set is empty"
        );
    }
    let observed: BTreeMap<_, _> = observed_gradients.into_iter().collect();

    let missing = expected
        .iter()
        .filter(|(id, _)| !observed.contains_key(id))
        .map(|(id, entry)| {
            format!(
                "layer={} module={} matrix={} tensor_id={id}",
                entry.layer_idx, entry.module, entry.matrix
            )
        })
        .collect::<Vec<_>>();
    let unknown = observed
        .keys()
        .filter(|id| !expected.contains_key(id))
        .map(|id| format!("tensor_id={id}"))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty() && unknown.is_empty(),
        "{context}: exact LoRA gradient identity mismatch: configured={} observed={} missing=[{}] unknown=[{}]",
        expected.len(),
        observed.len(),
        missing.join(", "),
        unknown.join(", ")
    );

    for (id, entry) in &expected {
        let grad = observed
            .get(id)
            .expect("exact LoRA gradient identity check established membership");
        validate_lora_gradient_tensor(context, entry, grad, check_finite_values)?;
    }
    Ok(())
}

pub(crate) fn validate_exact_lora_grad_store(
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        true,
    )
}

pub(super) fn validate_exact_lora_grad_store_metadata(
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        false,
    )
}

pub(super) fn validate_exact_lora_grad_map(
    params: &TrainableLoraParams,
    grads: &GradMap,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        true,
    )
}

pub(super) fn validate_exact_lora_grad_map_metadata(
    params: &TrainableLoraParams,
    grads: &GradMap,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        false,
    )
}

pub(crate) fn merge_checkpoint_lora_grad_segment(
    params: &TrainableLoraParams,
    accumulated: &mut kiln_autograd::GradStore,
    segment: kiln_autograd::GradStore,
    start_layer: usize,
    end_layer: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        start_layer < end_layer && end_layer <= params.layers.len(),
        "{context}: invalid checkpoint layer range {start_layer}..{end_layer} for {} layers",
        params.layers.len()
    );
    validate_exact_lora_gradients(
        params
            .all_params_with_modules()
            .into_iter()
            .filter(|entry| entry.layer_idx >= start_layer && entry.layer_idx < end_layer),
        segment.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::CheckpointLayerRange,
        false,
    )?;
    let segment = segment.into_inner();
    let mut duplicate_ids = segment
        .keys()
        .copied()
        .filter(|id| accumulated.contains(*id))
        .collect::<Vec<_>>();
    duplicate_ids.sort_unstable();
    anyhow::ensure!(
        duplicate_ids.is_empty(),
        "{context}: duplicate checkpoint LoRA gradient tensor IDs across layer segments: [{}]",
        duplicate_ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (id, grad) in segment {
        accumulated.insert(id, grad);
    }
    Ok(())
}

pub(super) fn accumulate_lora_grad_sum_sq(
    sum_sq_by_module: &mut BTreeMap<&'static str, f64>,
    module: &'static str,
    grad: &KtTensor,
) -> Result<()> {
    // (#1082) kt grad now; norm computed kt-natively.
    let norm = crate::train_receipt::tensor_l2_norm_kt(grad)
        .with_context(|| format!("compute LoRA grad l2 norm for module {module}"))?;
    if norm.is_finite() {
        *sum_sq_by_module.entry(module).or_insert(0.0) += norm * norm;
    } else {
        tracing::warn!(module, "skipping non-finite LoRA grad norm sample");
    }
    Ok(())
}

pub(super) fn observe_lora_grad_module_norms(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    sum_sq_by_module: BTreeMap<&'static str, f64>,
) {
    for (module, sum_sq) in sum_sq_by_module {
        accumulator.observe(module, sum_sq.sqrt());
    }
}

/// Tokenize a GRPO group: prompt messages + each completion text.
///
/// When a rollout carries a populated `trajectory` field, this routes
/// through `crate::trajectory_mask::build_masks_from_trajectory` so the
/// resulting `TokenizedGrpoCompletion` carries proper `action_mask` and
/// `env_mask` separations. ECHO consumes both masks; the legacy GRPO
/// policy-gradient path consumes `action_mask` (aliased to
/// `completion_mask` for back-compat).
///
/// When a rollout has no trajectory (legacy single-string `text` only),
/// behaviour is bit-identical to the pre-ECHO path: `action_mask` is "true
/// after the prompt" and `env_mask` is all-false.
pub(super) fn tokenize_grpo_group(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
) -> Result<TokenizedGrpoGroup> {
    let mask_cfg = crate::trajectory_mask::MaskConfig::default();
    tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, None)
}

/// Validate one GRPO group's policy/provenance contract without loading model
/// weights or touching an accelerator. API admission uses this for recorded
/// behavior data so a doomed job cannot sit in the training queue.
pub fn validate_grpo_group_policy_data(
    group: &GrpoGroup,
    config: &GrpoConfig,
    tokenizer: &KilnTokenizer,
) -> Result<()> {
    validate_grpo_group_policy_data_and_max_seq_len(group, config, tokenizer, 1).map(|_| ())
}

/// Validate one streamed GRPO row and return its exact longest tokenized
/// completion length. Server admission uses this while scanning the complete
/// JSONL source, so its memory plan is based on the same tokenizer and masks as
/// the trainer rather than a character-count estimate.
pub fn validate_grpo_group_policy_data_and_max_seq_len(
    group: &GrpoGroup,
    config: &GrpoConfig,
    tokenizer: &KilnTokenizer,
    source_line: usize,
) -> Result<usize> {
    config
        .validate_policy_config()
        .map_err(|error| anyhow::anyhow!("GRPO policy config: {error}"))?;
    validate_grpo_trajectory_roles(group, source_line)?;
    let has_env_tokens = group.completions.iter().any(|completion| {
        completion
            .trajectory
            .iter()
            .any(|segment| segment.kind == TurnKind::Observation)
    });
    config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(|error| anyhow::anyhow!("GRPO loss config: {error}"))?;
    let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
    let tokenized = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, None)?;
    validate_tokenized_behavior_policy(&tokenized, config.behavior_policy)?;
    validate_grpo_dry_run_masks(&tokenized, source_line, source_line)?;
    Ok(tokenized
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0))
}

pub(super) fn tokenize_grpo_group_timed(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
    mask_cfg: &crate::trajectory_mask::MaskConfig,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
) -> Result<TokenizedGrpoGroup> {
    if group.completions.is_empty() {
        anyhow::bail!("GRPO group has no completions");
    }

    let prompt_messages = to_core_messages(&group.messages);

    // Tokenize the prompt (without any assistant response). Used by the
    // legacy single-string path to find where the completion begins; the
    // trajectory-aware path computes its own boundaries via the mask
    // builder so it doesn't need this.
    let prompt_tokenize_started = Instant::now();
    let prompt_text = tokenizer
        .apply_chat_template(&prompt_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(t) = timings.as_deref_mut() {
        t.add_tokenize(prompt_tokenize_started.elapsed());
    }

    let mut raw_rewards = Vec::with_capacity(group.completions.len());
    let mut full_message_batches = Vec::with_capacity(group.completions.len());

    // Pre-built per-completion (input_ids, action_mask, env_mask,
    // total_obs_len) for the trajectory-aware path, indexed parallel to
    // `full_message_batches`. `None` means "use the legacy single-string
    // path for this rollout".
    let mut prebuilt: Vec<Option<crate::trajectory_mask::MaskedRollout>> =
        Vec::with_capacity(group.completions.len());

    for scored in &group.completions {
        if scored.has_trajectory() {
            // Trajectory-aware path: build masks from the explicit
            // segment structure. The MaskedRollout is the canonical
            // output; full_message_batches gets a stub so the indices
            // stay aligned with `full_id_batches` below.
            let (masked, mask_timings) = crate::trajectory_mask::build_masks_from_trajectory_timed(
                &scored.trajectory,
                &group.messages,
                tokenizer,
                mask_cfg,
            )?;
            if let Some(t) = timings.as_deref_mut() {
                t.tokenize_ms += mask_timings.tokenize_ms;
                t.mask_build_ms += mask_timings.mask_build_ms;
            }
            prebuilt.push(Some(masked));
            // Placeholder; not used when prebuilt is Some.
            full_message_batches.push(prompt_messages.clone());
        } else if scored.provenance.is_some() {
            // Exact provenance owns the model sequence. Keep a cheap prompt
            // placeholder in the parallel batch; re-rendering completed text
            // is not equivalent to the sequence inference consumed because
            // chat templates append generation prefixes.
            full_message_batches.push(prompt_messages.clone());
            prebuilt.push(None);
        } else {
            // Legacy single-string path: assemble [prompt + assistant
            // completion] and tokenize as one chat batch (cheap because
            // we batch all completions in one apply_chat_template_batch
            // call).
            let mut full_messages = prompt_messages.clone();
            full_messages.push(kiln_core::tokenizer::ChatMessage {
                role: "assistant".to_string(),
                content: scored.text.clone(),
                ..Default::default()
            });
            full_message_batches.push(full_messages);
            prebuilt.push(None);
        }
        raw_rewards.push(scored.reward);
    }

    let batch_tokenize_started = Instant::now();
    let full_texts = tokenizer
        .apply_chat_template_batch(&full_message_batches)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let full_id_batches = tokenizer
        .encode_batch(&full_texts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(t) = timings.as_deref_mut() {
        t.add_tokenize(batch_tokenize_started.elapsed());
    }
    let mut completions = Vec::with_capacity(full_id_batches.len());
    let mut rewards = Vec::with_capacity(full_id_batches.len());
    for (completion_idx, ((full_ids, reward), pre)) in full_id_batches
        .into_iter()
        .zip(raw_rewards)
        .zip(prebuilt)
        .enumerate()
    {
        if let Some(provenance) = group.completions[completion_idx].provenance.as_ref() {
            provenance.validate().map_err(|error| {
                anyhow::anyhow!(
                    "completion {completion_idx} has invalid rollout provenance: {error}"
                )
            })?;

            let vocab_sha256 = tokenizer.vocab_identity_sha256();
            let config_sha256 = tokenizer
                .tokenizer_config_sha256()
                .map_err(|error| anyhow::anyhow!("hash tokenizer config: {error}"))?;
            let chat_template_sha256 = tokenizer.chat_template_sha256().with_context(|| {
                format!(
                    "completion {completion_idx} has recorded provenance but the training tokenizer has no chat template"
                )
            })?;
            anyhow::ensure!(
                provenance.tokenizer.vocab_sha256 == vocab_sha256,
                "completion {completion_idx} rollout tokenizer vocabulary identity mismatch: provenance={}, training={vocab_sha256}",
                provenance.tokenizer.vocab_sha256
            );
            anyhow::ensure!(
                provenance.tokenizer.config_sha256 == config_sha256,
                "completion {completion_idx} rollout tokenizer config identity mismatch: provenance={}, training={config_sha256}",
                provenance.tokenizer.config_sha256
            );
            anyhow::ensure!(
                provenance.tokenizer.chat_template_sha256 == chat_template_sha256,
                "completion {completion_idx} rollout chat-template identity mismatch: provenance={}, training={chat_template_sha256}",
                provenance.tokenizer.chat_template_sha256
            );

            let prompt_messages_sha256 = crate::rollout_prompt_messages_sha256(&group.messages)
                .map_err(anyhow::Error::msg)?;
            let scored_payload_sha256 =
                crate::scored_rollout_payload_sha256(&group.completions[completion_idx])
                    .map_err(anyhow::Error::msg)?;
            anyhow::ensure!(
                provenance.prompt_messages_sha256 == prompt_messages_sha256,
                "completion {completion_idx} prompt messages differ from rollout provenance"
            );
            anyhow::ensure!(
                provenance.scored_payload_sha256 == scored_payload_sha256,
                "completion {completion_idx} scored text/trajectory differs from rollout provenance"
            );

            let recorded_prompt_text = tokenizer
                .apply_chat_template_full_with_options(
                    &prompt_messages,
                    (!provenance.template_invocation.tools.is_empty())
                        .then_some(provenance.template_invocation.tools.as_slice()),
                    provenance.template_invocation.tool_choice.as_ref(),
                    kiln_core::tokenizer::ChatTemplateOptions {
                        template_kwargs: provenance
                            .template_invocation
                            .template_kwargs
                            .clone(),
                    },
                )
                .map_err(|error| {
                    anyhow::anyhow!(
                        "completion {completion_idx} could not replay its recorded chat-template invocation: {error}"
                    )
                })?;
            let recorded_prompt_ids = tokenizer.encode(&recorded_prompt_text).map_err(|error| {
                anyhow::anyhow!(
                    "completion {completion_idx} could not tokenize its replayed rollout prompt: {error}"
                )
            })?;
            anyhow::ensure!(
                provenance.prompt_token_count == recorded_prompt_ids.len(),
                "completion {completion_idx} rollout prompt boundary {} differs from the rendered prompt length {}",
                provenance.prompt_token_count,
                recorded_prompt_ids.len()
            );
            anyhow::ensure!(
                provenance.input_token_ids[..provenance.prompt_token_count] == recorded_prompt_ids,
                "completion {completion_idx} rollout input prefix differs from the rendered prompt tokens"
            );

            let (rendered_trajectory_ids, trajectory_action_mask, env_mask, total_obs_len) =
                if let Some(masked) = pre {
                    let total_obs_len = masked.total_obs_len();
                    let crate::trajectory_mask::MaskedRollout {
                        input_ids,
                        action_mask,
                        env_mask,
                        segment_spans: _,
                    } = masked;
                    (Some(input_ids), Some(action_mask), env_mask, total_obs_len)
                } else {
                    (None, None, vec![false; provenance.input_token_ids.len()], 0)
                };

            if let Some(rendered_input_ids) = rendered_trajectory_ids.as_ref() {
                anyhow::ensure!(
                    rendered_input_ids == &provenance.input_token_ids,
                    "completion {completion_idx} trajectory rendering differs from its exact rollout token sequence; exact observation masks cannot be recovered safely"
                );
            }
            let rendered_action_indices =
                if let Some(expected_action_mask) = trajectory_action_mask.as_ref() {
                    expected_action_mask
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &active)| active.then_some(index))
                        .collect::<Vec<_>>()
                } else {
                    (provenance.prompt_token_count..provenance.input_token_ids.len())
                        .collect::<Vec<_>>()
                };
            let provenance_action_indices = provenance
                .action_tokens
                .iter()
                .map(|token| token.sequence_index)
                .collect::<Vec<_>>();
            anyhow::ensure!(
                rendered_action_indices == provenance_action_indices,
                "completion {completion_idx} scored payload action positions differ from rollout provenance"
            );

            let mut action_mask = vec![false; provenance.input_token_ids.len()];
            let mut behavior_log_probs = Vec::new();
            for action in &provenance.action_tokens {
                if action.source == crate::RolloutActionTokenSourceV1::Sampled {
                    action_mask[action.sequence_index] = true;
                    let logprob = action.behavior_logprob.with_context(|| {
                        format!(
                            "completion {completion_idx} sampled token {} is missing behavior_logprob",
                            action.sequence_index
                        )
                    })? as f32;
                    anyhow::ensure!(
                        logprob.is_finite(),
                        "completion {completion_idx} behavior_logprob at token {} cannot be represented as f32",
                        action.sequence_index
                    );
                    behavior_log_probs.push(logprob);
                }
            }
            anyhow::ensure!(
                action_mask
                    .iter()
                    .zip(env_mask.iter())
                    .all(|(&action, &env)| !(action && env)),
                "completion {completion_idx} has overlapping sampled-action and environment masks"
            );
            completions.push(TokenizedGrpoCompletion {
                input_ids: provenance.input_token_ids.clone(),
                prompt_token_count: provenance.prompt_token_count,
                action_mask,
                env_mask,
                total_obs_len,
                recorded_behavior_log_probs: Some(behavior_log_probs),
                recorded_behavior_source: Some(
                    crate::train_receipt::GrpoRecordedBehaviorSourceObservation::from_provenance(
                        provenance,
                    )
                    .with_context(|| {
                        format!(
                            "build completion {completion_idx} GRPO behavior-source observation"
                        )
                    })?,
                ),
            });
            rewards.push(reward);
            continue;
        }

        // Trajectory-aware path: prebuilt MaskedRollout overrides the
        // batch-tokenized full_ids. The mask builder rendered the
        // conversation itself, so its input_ids are authoritative for
        // the trajectory case.
        if let Some(masked) = pre {
            if masked.input_ids.len() < 2 {
                tracing::warn!(
                    "skipping trajectory completion: too short ({} tokens)",
                    masked.input_ids.len()
                );
                continue;
            }
            let total_obs_len = masked.total_obs_len();
            let crate::trajectory_mask::MaskedRollout {
                input_ids,
                action_mask,
                env_mask,
                segment_spans: _,
            } = masked;
            // The full trajectory render is authoritative for agentic data.
            // A separately rendered inference prompt can end in a different
            // template suffix (for example Qwen's enable_thinking=false
            // close sequence versus its default open <think> block). The
            // first action-mask token is the exact boundary between shared
            // policy context and model-produced trajectory tokens.
            let prompt_token_count = action_mask
                .iter()
                .position(|&active| active)
                .with_context(|| {
                    format!(
                        "trajectory completion {completion_idx} has no action token from which to derive its prompt boundary"
                    )
                })?;
            anyhow::ensure!(
                prompt_token_count > 0,
                "trajectory completion {completion_idx} action mask starts before any prompt token"
            );
            completions.push(TokenizedGrpoCompletion {
                input_ids,
                prompt_token_count,
                action_mask,
                env_mask,
                total_obs_len,
                recorded_behavior_log_probs: None,
                recorded_behavior_source: None,
            });
            rewards.push(reward);
            continue;
        }

        if full_ids.len() < 2 {
            tracing::warn!("skipping completion: too short ({} tokens)", full_ids.len());
            continue;
        }

        // Legacy single-string path: tokens after the prompt are
        // action tokens; there are no observation tokens.
        let mask_started = Instant::now();
        tracing::info!(
            seq_len = full_ids.len(),
            prompt_tokens = prompt_ids.len(),
            completion_tokens = full_ids.len().saturating_sub(prompt_ids.len()),
            "GRPO mask build start"
        );
        let mut action_mask = vec![false; full_ids.len()];
        action_mask[prompt_ids.len()..].fill(true);
        let env_mask = vec![false; full_ids.len()];
        let mask_elapsed = mask_started.elapsed();
        if let Some(t) = timings.as_deref_mut() {
            t.add_mask_build(mask_elapsed);
        }
        tracing::info!(
            seq_len = full_ids.len(),
            action_tokens = action_mask.iter().filter(|&&active| active).count(),
            env_tokens = 0usize,
            elapsed_ms = mask_elapsed.as_secs_f64() * 1000.0,
            "GRPO mask build end"
        );

        completions.push(TokenizedGrpoCompletion {
            input_ids: full_ids,
            prompt_token_count: prompt_ids.len(),
            action_mask,
            env_mask,
            total_obs_len: 0,
            recorded_behavior_log_probs: None,
            recorded_behavior_source: None,
        });
        rewards.push(reward);
    }

    if completions.is_empty() {
        anyhow::bail!("no valid completions in GRPO group after tokenization");
    }

    Ok(TokenizedGrpoGroup {
        completions,
        rewards,
    })
}

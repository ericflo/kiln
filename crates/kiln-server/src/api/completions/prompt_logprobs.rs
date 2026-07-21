use super::*;

pub(super) fn completion_system_fingerprint(state: &AppState) -> Result<Option<String>, ApiError> {
    canonical_completion_fingerprint(
        state.base_teacher_identity.as_deref(),
        matches!(state.backend.as_ref(), ModelBackend::Real { .. }),
    )
}

pub(super) fn canonical_completion_fingerprint(
    identity: Option<&kiln_train::TeacherIdentityV1>,
    required: bool,
) -> Result<Option<String>, ApiError> {
    match (identity, required) {
        (Some(identity), _) => Ok(Some(identity.fingerprint())),
        (None, false) => Ok(None),
        (None, true) => Err(ApiError::internal(
            "real prompt-logprob backend has no verified base teacher identity; restart with a loader-owned model source revision",
        )),
    }
}

pub(super) fn prompt_logprob_projection_chunk_tokens(vocab_size: usize) -> usize {
    let bytes_per_row = vocab_size
        .saturating_mul(2 * std::mem::size_of::<f32>())
        .max(1);
    (PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET / bytes_per_row)
        .clamp(1, MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS)
}

pub(super) fn validate_prompt_logprobs_top_k(
    state: &AppState,
    top_k: usize,
) -> Result<(), ApiError> {
    if top_k > MAX_COMPLETION_PROMPT_LOGPROBS {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt_logprobs {top_k} exceeds kiln's cap of {MAX_COMPLETION_PROMPT_LOGPROBS}"
        )));
    }
    if top_k > state.model_config.vocab_size {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt_logprobs {top_k} exceeds vocab size {}",
            state.model_config.vocab_size
        )));
    }
    Ok(())
}

pub(super) fn tokens_for_text_completion_prompt(
    state: &AppState,
    prompt: &TextCompletionPrompt,
    add_special_tokens: bool,
) -> Result<Vec<TokenId>, ApiError> {
    match prompt {
        TextCompletionPrompt::TokenIds(tokens) => Ok(tokens.clone()),
        TextCompletionPrompt::Text(text) => state
            .tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(ApiError::tokenization_failed),
    }
}

pub(super) fn decode_prompt_logprob_token(
    token_id: TokenId,
    preceding_actual_ids: &[TokenId],
    decode_token: impl FnOnce(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<String, ApiError> {
    decode_token(token_id, preceding_actual_ids).map_err(|err| {
        ApiError::tokenization_failed(format!(
            "failed to render prompt-logprob token id {token_id}: {err}"
        ))
    })
}

#[derive(Clone, Copy)]
pub(super) struct ValidatedPromptLogprobRow<'a> {
    values: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompactPromptLogprobEntry {
    pub(super) token_id: TokenId,
    pub(super) logprob: f32,
    pub(super) rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompactPromptLogprobSelection {
    pub(super) entries: Vec<CompactPromptLogprobEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PromptLogprobRankCandidate {
    token_id: TokenId,
    logit: f32,
}

impl Eq for PromptLogprobRankCandidate {}

impl Ord for PromptLogprobRankCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap::peek returns the greatest item. Reverse score order and
        // keep token-ID order forward so the root is the worst retained
        // candidate: lowest logit, then highest token ID.
        other
            .logit
            .partial_cmp(&self.logit)
            .expect("prompt-logprob rank candidates are validated finite")
            .then_with(|| self.token_id.cmp(&other.token_id))
    }
}

impl PartialOrd for PromptLogprobRankCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
pub(super) fn top_k_logprob_map_with_decoder(
    logit_row: &[f32],
    logprob_row: &[f32],
    expected_vocab_size: usize,
    observed_token_id: TokenId,
    top_k: usize,
    preceding_actual_ids: &[TokenId],
    mut decode_token: impl FnMut(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<PromptLogprobMap, ApiError> {
    let logit_row = validate_prompt_logprob_row(logit_row, expected_vocab_size)?;
    let logprob_row = validate_prompt_logprob_row(logprob_row, expected_vocab_size)?;
    let selection = select_prompt_logprobs_from_validated_rows(
        logit_row,
        logprob_row,
        observed_token_id,
        top_k,
    )?;
    prompt_logprob_map_from_selection_with_decoder(
        &selection,
        preceding_actual_ids,
        &mut decode_token,
    )
}

pub(super) fn validate_prompt_logprob_row<'a>(
    row: &'a [f32],
    expected_vocab_size: usize,
) -> Result<ValidatedPromptLogprobRow<'a>, ApiError> {
    if row.len() != expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs row width {} did not match model vocabulary size {expected_vocab_size}",
            row.len()
        )));
    }

    if let Some((token_id, value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs row contained non-finite value {value:?} at token id {token_id}"
        )));
    }

    Ok(ValidatedPromptLogprobRow { values: row })
}

pub(super) fn select_prompt_logprobs_from_validated_rows(
    logit_row: ValidatedPromptLogprobRow<'_>,
    logprob_row: ValidatedPromptLogprobRow<'_>,
    observed_token_id: TokenId,
    top_k: usize,
) -> Result<CompactPromptLogprobSelection, ApiError> {
    let logits = logit_row.values;
    let logprobs = logprob_row.values;
    if logits.len() != logprobs.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs logits width {} did not match log-probability width {}",
            logits.len(),
            logprobs.len()
        )));
    }
    if top_k > logits.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "requested {top_k} prompt-logprob candidates from a vocabulary row with width {}",
            logits.len()
        )));
    }
    let observed_index = observed_token_id as usize;
    let observed_logit = logits.get(observed_index).copied().ok_or_else(|| {
        ApiError::generation_failed(anyhow::anyhow!(
            "observed prompt token id {observed_token_id} was outside vocabulary row width {}",
            logits.len()
        ))
    })?;
    let observed_logprob = logprobs[observed_index];

    if top_k == 0 {
        let observed_rank = logits
            .iter()
            .filter(|&&logit| logit >= observed_logit)
            .count();
        return Ok(CompactPromptLogprobSelection {
            entries: vec![CompactPromptLogprobEntry {
                token_id: observed_token_id,
                logprob: observed_logprob,
                rank: observed_rank,
            }],
        });
    }

    let compare_rank = |a: &(TokenId, f32), b: &(TokenId, f32)| {
        b.1.partial_cmp(&a.1)
            .expect("validated prompt-logprob rows contain only finite values")
            .then_with(|| a.0.cmp(&b.0))
    };
    let mut heap = std::collections::BinaryHeap::with_capacity(top_k);
    for (token_id, &logit) in logits.iter().enumerate() {
        let candidate = PromptLogprobRankCandidate {
            token_id: token_id as TokenId,
            logit,
        };
        if heap.len() < top_k {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| {
            compare_rank(
                &(candidate.token_id, candidate.logit),
                &(worst.token_id, worst.logit),
            )
            .is_lt()
        }) {
            heap.pop();
            heap.push(candidate);
        }
    }
    let mut pairs = heap
        .into_iter()
        .map(|candidate| (candidate.token_id, candidate.logit))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by(compare_rank);

    let observed_top_rank = pairs
        .iter()
        .position(|(token_id, _)| *token_id == observed_token_id)
        .map(|rank| rank + 1);
    let observed_rank = observed_top_rank.unwrap_or_else(|| {
        logits
            .iter()
            .filter(|&&logit| logit >= observed_logit)
            .count()
    });

    // vLLM inserts the observed token first and then its top-K alternatives.
    // A duplicate observed token in top-K overwrites the value at the same key;
    // represent that final distinct-token result directly here.
    let mut entries = Vec::with_capacity(top_k + usize::from(observed_top_rank.is_none()));
    entries.push(CompactPromptLogprobEntry {
        token_id: observed_token_id,
        logprob: observed_logprob,
        rank: observed_rank,
    });
    for (rank, (token_id, _)) in pairs.into_iter().enumerate() {
        if token_id != observed_token_id {
            entries.push(CompactPromptLogprobEntry {
                token_id,
                logprob: logprobs[token_id as usize],
                rank: rank + 1,
            });
        }
    }

    let expected_len = top_k + usize::from(observed_top_rank.is_none());
    if entries.len() != expected_len {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs selection returned {} distinct candidates instead of {expected_len}",
            entries.len()
        )));
    }
    Ok(CompactPromptLogprobSelection { entries })
}

pub(super) fn select_prompt_logprobs_from_device_row(
    row: &kiln_tensor::DevicePromptLogprobRow,
    expected_vocab_size: usize,
    observed_token_id: TokenId,
    top_k: usize,
) -> Result<CompactPromptLogprobSelection, ApiError> {
    if top_k > expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "requested {top_k} prompt-logprob candidates from a vocabulary row with width {expected_vocab_size}"
        )));
    }
    if observed_token_id as usize >= expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "observed prompt token id {observed_token_id} was outside vocabulary row width {expected_vocab_size}"
        )));
    }
    if row.candidates.len() != top_k {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned {} candidates instead of {top_k}",
            row.candidates.len()
        )));
    }
    if !row.observed_logit.is_finite() || !row.observed_logprob.is_finite() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned non-finite observed values"
        )));
    }
    if !(1..=expected_vocab_size).contains(&row.observed_full_rank) {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob observed rank {} was outside 1..={expected_vocab_size}",
            row.observed_full_rank
        )));
    }

    let observed_top_rank = row
        .candidates
        .iter()
        .position(|candidate| candidate.token_id == observed_token_id)
        .map(|rank| rank + 1);
    if let Some(rank) = observed_top_rank {
        let candidate = &row.candidates[rank - 1];
        if candidate.logit != row.observed_logit || candidate.logprob != row.observed_logprob {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob observed candidate disagreed with observed row statistics"
            )));
        }
    }

    let observed_rank = observed_top_rank.unwrap_or(row.observed_full_rank);
    let mut entries = Vec::with_capacity(top_k + usize::from(observed_top_rank.is_none()));
    entries.push(CompactPromptLogprobEntry {
        token_id: observed_token_id,
        logprob: row.observed_logprob,
        rank: observed_rank,
    });
    for (rank, candidate) in row.candidates.iter().enumerate() {
        if candidate.token_id as usize >= expected_vocab_size {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob candidate token id {} was outside vocabulary row width {expected_vocab_size}",
                candidate.token_id
            )));
        }
        if !candidate.logit.is_finite() || !candidate.logprob.is_finite() {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob candidate at rank {} was non-finite",
                rank + 1
            )));
        }
        if candidate.token_id != observed_token_id {
            entries.push(CompactPromptLogprobEntry {
                token_id: candidate.token_id,
                logprob: candidate.logprob,
                rank: rank + 1,
            });
        }
    }

    let expected_len = top_k + usize::from(observed_top_rank.is_none());
    if entries.len() != expected_len {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned {} distinct candidates instead of {expected_len}",
            entries.len()
        )));
    }
    Ok(CompactPromptLogprobSelection { entries })
}

pub(super) fn prompt_logprob_map_from_selection(
    state: &AppState,
    selection: &CompactPromptLogprobSelection,
    preceding_actual_ids: &[TokenId],
) -> Result<PromptLogprobMap, ApiError> {
    let mut decode_token = |token_id, context: &[TokenId]| {
        state
            .tokenizer
            .decode_token_for_display_with_context(token_id, context)
    };
    prompt_logprob_map_from_selection_with_decoder(
        selection,
        preceding_actual_ids,
        &mut decode_token,
    )
}

pub(super) fn prompt_logprob_map_from_selection_with_decoder(
    selection: &CompactPromptLogprobSelection,
    preceding_actual_ids: &[TokenId],
    decode_token: &mut impl FnMut(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<PromptLogprobMap, ApiError> {
    let mut out = BTreeMap::new();
    for entry in &selection.entries {
        out.insert(
            entry.token_id.to_string(),
            PromptLogprobEntry {
                logprob: entry.logprob,
                rank: entry.rank,
                decoded_token: decode_prompt_logprob_token(
                    entry.token_id,
                    preceding_actual_ids,
                    &mut *decode_token,
                )?,
            },
        );
    }
    if out.len() != selection.entries.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs rendering collapsed distinct token ids"
        )));
    }
    Ok(out)
}

pub(super) fn prompt_logprobs_from_selections(
    state: &AppState,
    prompt_tokens: &[TokenId],
    selections: &[CompactPromptLogprobSelection],
    deadline: Option<tokio::time::Instant>,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let expected_selections = prompt_tokens.len().saturating_sub(1);
    if selections.len() != expected_selections {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs selection count {} did not match scored prompt position count {expected_selections}",
            selections.len()
        )));
    }

    let mut out = Vec::with_capacity(prompt_tokens.len());
    out.push(None);
    for (selection_index, selection) in selections.iter().enumerate() {
        if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            return Err(ApiError::request_timeout(state.request_timeout.as_secs()));
        }
        let prompt_position = selection_index + 1;
        out.push(Some(prompt_logprob_map_from_selection(
            state,
            selection,
            &prompt_tokens[..prompt_position],
        )?));
    }
    Ok(out)
}

#[cfg(test)]
pub(super) fn prompt_logprobs_from_rows(
    state: &AppState,
    prompt_tokens: &[TokenId],
    logit_rows: &[Vec<f32>],
    logprob_rows: &[Vec<f32>],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let expected_rows = prompt_tokens.len().saturating_sub(1);
    if logit_rows.len() != expected_rows || logprob_rows.len() != expected_rows {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs logits/log-probability row counts {}/{} did not match scored prompt position count {expected_rows}",
            logit_rows.len(),
            logprob_rows.len()
        )));
    }

    let validated_logits = logit_rows
        .iter()
        .map(|row| validate_prompt_logprob_row(row, state.model_config.vocab_size))
        .collect::<Result<Vec<_>, _>>()?;
    let validated_rows = logprob_rows
        .iter()
        .map(|row| validate_prompt_logprob_row(row, state.model_config.vocab_size))
        .collect::<Result<Vec<_>, _>>()?;

    let mut selections = Vec::with_capacity(prompt_tokens.len().saturating_sub(1));
    for pos in 1..prompt_tokens.len() {
        selections.push(select_prompt_logprobs_from_validated_rows(
            validated_logits[pos - 1],
            validated_rows[pos - 1],
            prompt_tokens[pos],
            top_k,
        )?);
    }
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, None)
}

pub(super) fn mock_prompt_logprobs(
    state: &AppState,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let vocab_size = state.model_config.vocab_size.max(1);
    let mut selections = Vec::with_capacity(prompt_tokens.len().saturating_sub(1));
    for pos in 1..prompt_tokens.len() {
        // The mock distribution is score-descending in token-id order:
        // logprob(token_id) = -token_id. This keeps behavior deterministic
        // while exercising the same K/K+1 observed-token rule as real models.
        let observed_token_id = prompt_tokens[pos];
        let observed_in_top_k = (observed_token_id as usize) < top_k;
        let mut entries = Vec::with_capacity(top_k + usize::from(!observed_in_top_k));
        entries.push(CompactPromptLogprobEntry {
            token_id: observed_token_id,
            logprob: -(observed_token_id as f32),
            rank: observed_token_id as usize + 1,
        });
        for rank in 0..top_k.min(vocab_size) {
            let token_id = rank as TokenId;
            if token_id != observed_token_id {
                entries.push(CompactPromptLogprobEntry {
                    token_id,
                    logprob: -(token_id as f32),
                    rank: rank + 1,
                });
            }
        }
        selections.push(CompactPromptLogprobSelection { entries });
    }
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, None)
}

pub(super) fn prompt_logprob_tensor_rows_to_f32(
    tensor: &kiln_tensor::Tensor,
) -> anyhow::Result<Vec<Vec<f32>>> {
    match tensor.dtype() {
        kiln_tensor::DType::F32 => tensor.to_vec2::<f32>(),
        kiln_tensor::DType::BF16 => tensor.to_vec2::<half::bf16>().map(|rows| {
            rows.into_iter()
                .map(|row| row.into_iter().map(half::bf16::to_f32).collect())
                .collect()
        }),
        kiln_tensor::DType::F16 => tensor.to_vec2::<half::f16>().map(|rows| {
            rows.into_iter()
                .map(|row| row.into_iter().map(half::f16::to_f32).collect())
                .collect()
        }),
        dtype => anyhow::bail!("prompt-logprobs logits had unsupported dtype {dtype}"),
    }
    .context("prompt-logprobs tensor rows to F32 host values")
}

pub(super) fn ensure_prompt_logprob_scoring_active(cancel: &CancelHandle) -> anyhow::Result<()> {
    if cancel.is_cancelled() {
        anyhow::bail!("prompt-logprobs scoring cancelled");
    }
    Ok(())
}

pub(super) async fn validate_prompt_logprob_runner_admission(
    runner: &std::sync::RwLock<ModelRunner>,
    deadline: tokio::time::Instant,
    timeout: std::time::Duration,
) -> Result<(), ApiError> {
    loop {
        match runner.try_read() {
            Ok(runner_guard) => {
                runner_guard
                    .ensure_backend_healthy()
                    .map_err(ApiError::generation_failed)?;
                if runner_guard.active_lora().is_some() {
                    return Err(ApiError::completion_invalid_request(
                        "prompt-logprobs scoring is base-model only until adapter revision identity is pinned; unload the active adapter",
                    ));
                }
                return Ok(());
            }
            Err(std::sync::TryLockError::Poisoned(error)) => {
                return Err(ApiError::internal(format!(
                    "prompt-logprobs runner lock poisoned: {error}"
                )));
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }

        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
        tokio::time::sleep_until(std::cmp::min(
            deadline,
            now + std::time::Duration::from_millis(1),
        ))
        .await;
    }
}

pub(super) fn prompt_logprob_runner_read<'a>(
    runner: &'a std::sync::RwLock<ModelRunner>,
    cancel: &CancelHandle,
) -> anyhow::Result<std::sync::RwLockReadGuard<'a, ModelRunner>> {
    loop {
        ensure_prompt_logprob_scoring_active(cancel)?;
        match runner.try_read() {
            Ok(runner_guard) => return Ok(runner_guard),
            Err(std::sync::TryLockError::Poisoned(error)) => {
                anyhow::bail!("prompt-logprobs runner lock poisoned: {error}")
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

pub(super) struct PromptLogprobWorkerOwnership {
    _gpu_guard: tokio::sync::OwnedRwLockWriteGuard<()>,
    linear_state: Option<kiln_model::forward::LinearAttentionState>,
    normalized_hidden: Option<kiln_tensor::Tensor>,
    hidden_chunk: Option<kiln_tensor::Tensor>,
    logits: Option<kiln_tensor::Tensor>,
    cpu_logits: Option<kiln_tensor::Tensor>,
    logits_2d: Option<kiln_tensor::Tensor>,
    log_probs: Option<kiln_tensor::Tensor>,
    log_probs_2d: Option<kiln_tensor::Tensor>,
}

impl PromptLogprobWorkerOwnership {
    fn new(gpu_guard: tokio::sync::OwnedRwLockWriteGuard<()>) -> Self {
        Self {
            _gpu_guard: gpu_guard,
            linear_state: None,
            normalized_hidden: None,
            hidden_chunk: None,
            logits: None,
            cpu_logits: None,
            logits_2d: None,
            log_probs: None,
            log_probs_2d: None,
        }
    }

    fn clear_completed_chunk(&mut self) {
        self.log_probs_2d = None;
        self.log_probs = None;
        self.logits_2d = None;
        self.cpu_logits = None;
        self.logits = None;
        self.hidden_chunk = None;
    }
}

pub(super) fn run_prompt_logprob_worker_with_panic_fence<T, O>(
    backend_health: &kiln_model::BackendHealthHandle,
    mut ownership: O,
    work: impl FnOnce(&mut O) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut ownership))) {
        Ok(result) => {
            if backend_health.snapshot().quarantined {
                std::mem::forget(ownership);
            }
            result
        }
        Err(_) => {
            let reason = "prompt-logprobs scorer or settlement panicked; backend completion and GPU ownership are unknown";
            backend_health.quarantine(reason);
            std::mem::forget(ownership);
            Err(anyhow::anyhow!(reason))
        }
    }
}

pub(super) fn score_real_prompt_logprob_rows(
    runner: &ModelRunner,
    prompt_tokens: &[TokenId],
    scored_tokens: &[TokenId],
    expected_vocab_size: usize,
    top_k: usize,
    cancel: &CancelHandle,
    metrics: &crate::metrics::Metrics,
    ownership: &mut PromptLogprobWorkerOwnership,
) -> anyhow::Result<Vec<CompactPromptLogprobSelection>> {
    runner.ensure_backend_healthy()?;
    if runner.active_lora().is_some() {
        anyhow::bail!("prompt-logprobs active adapter changed after base-model admission");
    }
    if prompt_tokens.len() != scored_tokens.len().saturating_add(1) {
        anyhow::bail!(
            "prompt-logprobs prompt/scored token counts {}/{} did not preserve the causal one-row offset",
            prompt_tokens.len(),
            scored_tokens.len()
        );
    }

    let device = runner.weights.device_kt();
    let backend = runner.backend_runtime();
    ownership.linear_state = Some(
        kiln_model::forward::LinearAttentionState::new_with_batch_for_inference_runtime(
            &runner.config,
            1,
            &device,
            backend,
        )?,
    );
    let normalized_hidden = kiln_model::forward::model_forward_no_head_with_policy(
        backend,
        scored_tokens,
        &runner.weights,
        &runner.config,
        ownership.linear_state.as_mut(),
        None,
        runner.streaming_prefill_policy(),
    )
    .context("prompt-logprobs forward pass")?;
    ownership.normalized_hidden = Some(normalized_hidden);
    ensure_prompt_logprob_scoring_active(cancel)?;

    let (batch_size, scored_sequence_len, hidden_size) = ownership
        .normalized_hidden
        .as_ref()
        .context("prompt-logprobs normalized hidden owner missing")?
        .dims3()
        .context("prompt-logprobs normalized hidden shape")?;
    if batch_size != 1
        || scored_sequence_len != scored_tokens.len()
        || hidden_size != runner.config.hidden_size
    {
        anyhow::bail!(
            "prompt-logprobs normalized hidden shape {:?} did not match expected [1, {}, {}]",
            ownership
                .normalized_hidden
                .as_ref()
                .context("prompt-logprobs normalized hidden owner missing")?
                .dims(),
            scored_tokens.len(),
            runner.config.hidden_size
        );
    }

    let mut selections = Vec::with_capacity(scored_sequence_len);
    let mut validated_row_count = 0usize;
    let projection_chunk_tokens = prompt_logprob_projection_chunk_tokens(expected_vocab_size);
    for chunk_start in (0..scored_sequence_len).step_by(projection_chunk_tokens) {
        ensure_prompt_logprob_scoring_active(cancel)?;
        let chunk_len = (scored_sequence_len - chunk_start).min(projection_chunk_tokens);
        ownership.hidden_chunk = Some(
            ownership
                .normalized_hidden
                .as_ref()
                .context("prompt-logprobs normalized hidden owner missing")?
                .narrow(1, chunk_start, chunk_len)
                .context("prompt-logprobs narrow normalized hidden chunk")?
                .contiguous()
                .context("prompt-logprobs normalized hidden chunk contiguous")?,
        );
        ownership.logits = Some(
            kiln_model::forward::model_forward_project_normalized_hidden(
                backend,
                ownership
                    .hidden_chunk
                    .as_ref()
                    .context("prompt-logprobs hidden chunk owner missing")?,
                &runner.weights,
            )
            .with_context(|| {
                format!(
                    "prompt-logprobs LM-head projection for rows {chunk_start}..{}",
                    chunk_start + chunk_len
                )
            })?,
        );
        let expected_logits_shape = [1, chunk_len, expected_vocab_size];
        let logits = ownership
            .logits
            .as_ref()
            .context("prompt-logprobs logits owner missing")?;
        if logits.dims() != expected_logits_shape {
            anyhow::bail!(
                "prompt-logprobs logits shape {:?} did not match expected {:?}",
                logits.dims(),
                expected_logits_shape
            );
        }

        ownership.logits_2d = Some(
            logits
                .squeeze(0)
                .context("prompt-logprobs squeeze logits batch dim")?,
        );
        let observed_token_ids = &prompt_tokens[chunk_start + 1..chunk_start + chunk_len + 1];
        let logits_2d = ownership
            .logits_2d
            .as_ref()
            .context("prompt-logprobs logits view owner missing")?;
        let device_rows: Option<Vec<kiln_tensor::DevicePromptLogprobRow>> = match logits_2d.device()
        {
            #[cfg(feature = "cuda")]
            kiln_tensor::Device::Cuda(_) => Some(
                kiln_tensor::cuda_prompt_logprobs(logits_2d, observed_token_ids, top_k)
                    .context("CUDA compact prompt-logprob selection")?,
            ),
            #[cfg(feature = "rocm")]
            kiln_tensor::Device::Rocm(_) => Some(
                kiln_tensor::rocm_prompt_logprobs(logits_2d, observed_token_ids, top_k)
                    .context("ROCm compact prompt-logprob selection")?,
            ),
            _ => None,
        };

        let selection_route;
        if let Some(device_rows) = device_rows {
            selection_route = crate::metrics::PromptLogprobSelectionRoute::CompactDevice;
            if device_rows.len() != chunk_len {
                anyhow::bail!(
                    "compact prompt-logprob selection returned {} rows instead of {chunk_len}",
                    device_rows.len()
                );
            }
            for (chunk_row, row) in device_rows.iter().enumerate() {
                ensure_prompt_logprob_scoring_active(cancel)?;
                selections.push(
                    select_prompt_logprobs_from_device_row(
                        row,
                        expected_vocab_size,
                        observed_token_ids[chunk_row],
                        top_k,
                    )
                    .map_err(|error| anyhow::anyhow!(error.message))?,
                );
                validated_row_count += 1;
            }
        } else {
            selection_route = crate::metrics::PromptLogprobSelectionRoute::BoundedHostFallback;
            // Selection and full rank use original logits. F32 log-softmax
            // values can collapse distinct far-tail logits after the LSE
            // subtraction, so ranking the rendered values is not equivalent.
            // Vulkan's log-softmax is host-backed; keep that result on CPU
            // instead of bouncing it back to the device and immediately out.
            let host_logit_rows = if matches!(logits.device(), kiln_tensor::Device::Vulkan(_)) {
                ownership.cpu_logits = Some(
                    logits
                        .to_device(kiln_tensor::Device::Cpu)
                        .context("prompt-logprobs Vulkan logits to host")?,
                );
                ownership.logits_2d = Some(
                    ownership
                        .cpu_logits
                        .as_ref()
                        .context("prompt-logprobs CPU logits owner missing")?
                        .squeeze(0)
                        .context("prompt-logprobs squeeze CPU logits batch dim")?,
                );
                let host_rows = prompt_logprob_tensor_rows_to_f32(
                    ownership
                        .logits_2d
                        .as_ref()
                        .context("prompt-logprobs logits view owner missing")?,
                )?;
                ownership.log_probs = Some(
                    kiln_tensor::ops::log_softmax_last_dim_f32(
                        ownership
                            .cpu_logits
                            .as_ref()
                            .context("prompt-logprobs CPU logits owner missing")?,
                    )
                    .context("prompt-logprobs host log_softmax_last_dim_f32")?,
                );
                host_rows
            } else {
                let host_rows = prompt_logprob_tensor_rows_to_f32(
                    ownership
                        .logits_2d
                        .as_ref()
                        .context("prompt-logprobs logits view owner missing")?,
                )?;
                ownership.log_probs = Some(
                    kiln_tensor::ops::log_softmax_last_dim_f32(logits)
                        .context("prompt-logprobs log_softmax_last_dim_f32")?,
                );
                host_rows
            };
            let log_probs = ownership
                .log_probs
                .as_ref()
                .context("prompt-logprobs log-probability owner missing")?;
            if log_probs.dims() != expected_logits_shape
                || log_probs.dtype() != kiln_tensor::DType::F32
            {
                anyhow::bail!(
                    "prompt-logprobs F32 log-softmax produced shape {:?} and dtype {:?}; expected {:?} and F32",
                    log_probs.dims(),
                    log_probs.dtype(),
                    expected_logits_shape
                );
            }
            ownership.log_probs_2d = Some(
                log_probs
                    .squeeze(0)
                    .context("prompt-logprobs squeeze log-probability batch dim")?,
            );
            let host_logprob_rows = ownership
                .log_probs_2d
                .as_ref()
                .context("prompt-logprobs log-probability view owner missing")?
                .to_vec2::<f32>()
                .context("prompt-logprobs F32 chunk to host")?;

            ensure_prompt_logprob_scoring_active(cancel)?;
            if host_logit_rows.len() != chunk_len || host_logprob_rows.len() != chunk_len {
                anyhow::bail!(
                    "prompt-logprobs host chunks returned logits/log-probability row counts {}/{} instead of {chunk_len}",
                    host_logit_rows.len(),
                    host_logprob_rows.len()
                );
            }
            for (chunk_row, (logit_row, logprob_row)) in
                host_logit_rows.iter().zip(&host_logprob_rows).enumerate()
            {
                ensure_prompt_logprob_scoring_active(cancel)?;
                let global_row = chunk_start + chunk_row;
                let validated_logits = validate_prompt_logprob_row(logit_row, expected_vocab_size)
                    .map_err(|error| anyhow::anyhow!(error.message))?;
                let validated_logprobs =
                    validate_prompt_logprob_row(logprob_row, expected_vocab_size)
                        .map_err(|error| anyhow::anyhow!(error.message))?;
                selections.push(
                    select_prompt_logprobs_from_validated_rows(
                        validated_logits,
                        validated_logprobs,
                        prompt_tokens[global_row + 1],
                        top_k,
                    )
                    .map_err(|error| anyhow::anyhow!(error.message))?,
                );
                validated_row_count += 1;
            }
        }
        // Host readback proves the requested values are available, but the
        // backend may own auxiliary stream/cache work. Use the repository's
        // explicit external-yield fence before dropping this chunk's device
        // owners and reusing their bounded memory budget.
        runner.synchronize_external_yield("prompt-logprobs projection chunk")?;
        metrics.record_prompt_logprob_selection(selection_route, chunk_len);
        ownership.clear_completed_chunk();
    }

    if validated_row_count != scored_sequence_len {
        anyhow::bail!(
            "prompt-logprobs validated {validated_row_count} rows instead of {scored_sequence_len}"
        );
    }
    Ok(selections)
}

pub(super) async fn real_prompt_logprobs(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    if prompt_tokens.len() == 1 {
        return Ok(vec![None]);
    }

    let timeout = state.request_timeout;
    let deadline = tokio::time::Instant::now() + timeout;
    let cancel = CancelHandle::new();
    let _cancel_on_drop = CancelOnDrop::new(cancel.clone());
    let backend_health = match state.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => backend_health.clone(),
        ModelBackend::Mock { .. } => {
            return Err(ApiError::internal(
                "real prompt-logprobs scorer received the mock backend",
            ));
        }
    };
    backend_health
        .ensure_healthy()
        .map_err(ApiError::backend_quarantined)?;
    validate_prompt_logprob_runner_admission(runner, deadline, timeout).await?;
    // Prompt scoring performs a full prefill plus vocabulary-wide projection.
    // Take exclusive GPU admission so concurrent scoring/generation/training
    // cannot multiply its bounded scratch residency into an OOM or a visible
    // mid-inference allocator stall.
    let gpu_guard = match tokio::time::timeout_at(
        deadline,
        gpu_coordination_write_guard_while_healthy_async(&state.gpu_lock, &backend_health),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(error)) => return Err(ApiError::backend_quarantined(error)),
        Err(_) => return Err(ApiError::request_timeout(timeout.as_secs())),
    };

    let runner = runner.clone();
    let prompt_tokens_owned = prompt_tokens.to_vec();
    let scored_tokens = prompt_tokens[..prompt_tokens.len() - 1].to_vec();
    let expected_vocab_size = state.model_config.vocab_size;
    let cancel_inner = cancel.clone();
    let worker_backend_health = backend_health.clone();
    let metrics = std::sync::Arc::clone(&state.metrics);
    let handle = tokio::task::spawn_blocking(
        move || -> anyhow::Result<Vec<CompactPromptLogprobSelection>> {
            run_prompt_logprob_worker_with_panic_fence(
                &worker_backend_health,
                PromptLogprobWorkerOwnership::new(gpu_guard),
                |ownership| {
                    ensure_prompt_logprob_scoring_active(&cancel_inner)?;
                    let runner_guard = prompt_logprob_runner_read(&runner, &cancel_inner)?;
                    let scoring_result = score_real_prompt_logprob_rows(
                        &runner_guard,
                        &prompt_tokens_owned,
                        &scored_tokens,
                        expected_vocab_size,
                        top_k,
                        &cancel_inner,
                        &metrics,
                        ownership,
                    );

                    // Every ordinary result, including cancellation and
                    // validation failure, must settle submitted backend work
                    // before exclusive admission becomes reusable.
                    let synchronized =
                        runner_guard.synchronize_external_yield("prompt-logprobs scoring");
                    drop(runner_guard);
                    synchronized?;
                    scoring_result
                },
            )
        },
    );

    tokio::pin!(handle);
    let selections = match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(Ok(selections))) => selections,
        Ok(Ok(Err(err))) => return Err(ApiError::generation_failed(err)),
        Ok(Err(err)) => {
            backend_health.quarantine(format!(
                "prompt-logprobs blocking worker terminated without settlement: {err}"
            ));
            return Err(ApiError::internal(format!("join error: {err}")));
        }
        Err(_) => {
            cancel.cancel();
            if let Err(err) = handle.await {
                backend_health.quarantine(format!(
                    "timed-out prompt-logprobs worker terminated without settlement: {err}"
                ));
            }
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, Some(deadline))
}

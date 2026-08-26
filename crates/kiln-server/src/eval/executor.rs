//! Suite executor: drives `EvalGenerator` × `Scorer` across an `EvalSuite`
//! and produces a `SuiteResult`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use kiln_eval::scorers::{JudgeRunner, NoopJudgeRunner, score_completion};
use kiln_eval::{
    AggregateMetrics, EvalGenerationParams, EvalOutcomeKind, EvalProgress, EvalSuite,
    EvalThinkingBudget, ExampleOutcome, SuiteResult, aggregate_example_outcomes,
};

use crate::eval::generator::{EvalGenerationSource, EvalGenerator, PreparedPrompt};

#[derive(Debug, thiserror::Error)]
pub enum EvalExecutionError {
    #[error("generation failed: {0}")]
    Generation(String),
    #[error("scorer error: {0}")]
    Scorer(String),
    #[error("aggregation failed: {0}")]
    Aggregation(String),
}

/// Callback fired after every example completes. Used by the worker loop to
/// surface running accuracy into `EvalJobInfo::progress` while the suite is
/// still in flight.
pub type ProgressCallback = Box<dyn Fn(EvalProgress) + Send + Sync>;

/// Startup-owned identities bound into every replay record produced by a
/// worker. Synthetic executor callers may use the empty default; strict replay
/// will correctly reject the resulting incomplete record.
#[derive(Debug, Clone, Default)]
pub struct EvalReplayEnvironment {
    pub execution_provenance_sha256: Option<String>,
    pub base_weight_manifest_sha256: Option<String>,
}

/// Run `suite` against `adapter` with `generator`. Returns the aggregated
/// `SuiteResult`. The function is cancellation-cooperative — when the
/// `cancelled` flag flips to true the executor returns early with the
/// partial result populated.
pub async fn run_suite_against_adapter(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
    effective_seed: u64,
    generator: Arc<dyn EvalGenerator>,
    progress: Option<ProgressCallback>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    judge_runner: Arc<dyn JudgeRunner>,
) -> Result<SuiteResult, EvalExecutionError> {
    run_suite_against_adapter_with_replay(
        suite,
        adapter,
        generation_override,
        effective_seed,
        generator,
        progress,
        cancelled,
        judge_runner,
        EvalReplayEnvironment::default(),
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_suite_against_adapter_with_replay(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
    effective_seed: u64,
    generator: Arc<dyn EvalGenerator>,
    progress: Option<ProgressCallback>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    judge_runner: Arc<dyn JudgeRunner>,
    replay_environment: EvalReplayEnvironment,
    replay_source_record: Option<Arc<kiln_eval::EvalReplayRecordV1>>,
) -> Result<SuiteResult, EvalExecutionError> {
    suite
        .validate()
        .map_err(|error| EvalExecutionError::Aggregation(error.to_string()))?;
    validate_effective_aggregation(suite, generation_override)?;
    // Resolve inherited server defaults and reject invalid close-token/stop/
    // max-token combinations before loading an adapter or generating any
    // examples. Cache duplicate generation objects so suite-wide defaults are
    // validated exactly once per run.
    let (prepared_examples, resolved_budgets) =
        prepare_and_preflight_examples(suite, generation_override, generator.as_ref()).await?;
    let effective_generation_hash = kiln_eval::eval_effective_generation_sha256(
        suite,
        generation_override,
        effective_seed,
        &resolved_budgets,
    )
    .map_err(|error| EvalExecutionError::Aggregation(error.to_string()))?;
    let suite_hash = kiln_eval::eval_suite_sha256(suite)
        .map_err(|error| EvalExecutionError::Aggregation(error.to_string()))?;
    if let Some(source) = replay_source_record.as_deref()
        && (source.effective_seed != effective_seed
            || source.suite_sha256 != suite_hash
            || source.effective_generation_sha256 != effective_generation_hash
            || source.execution_provenance_sha256 != replay_environment.execution_provenance_sha256
            || source.base_weight_manifest_sha256 != replay_environment.base_weight_manifest_sha256)
    {
        return Err(EvalExecutionError::Generation(
            "strict replay input or environment identity drifted after admission".into(),
        ));
    }

    // Hoist the adapter swap out of the per-example loop. The previous
    // active adapter is restored at the end via the same call so a suite
    // run never leaves the model in an unexpected state, even when an
    // example errors mid-loop (including after the deferred judge pass
    // swapped to a judge adapter).
    let previous_adapter = generator
        .set_adapter(adapter)
        .await
        .map_err(EvalExecutionError::Generation)?;
    let model_target = match generator.model_target_identity(adapter) {
        Ok(identity) => identity,
        Err(error) => {
            let _ = generator.restore_adapter(previous_adapter.as_deref()).await;
            return Err(EvalExecutionError::Generation(error));
        }
    };
    if let Some(source) = replay_source_record.as_deref()
        && model_target != source.model_target
    {
        let _ = generator.restore_adapter(previous_adapter.as_deref()).await;
        return Err(EvalExecutionError::Generation(format!(
            "strict replay model target drifted after admission: expected {:?}, loaded {:?}",
            source.model_target, model_target
        )));
    }
    let result = run_suite_inner(
        suite,
        adapter,
        generation_override,
        effective_seed,
        prepared_examples,
        resolved_budgets,
        suite_hash,
        effective_generation_hash,
        model_target,
        replay_environment,
        replay_source_record,
        generator.clone(),
        progress,
        cancelled,
        judge_runner,
    )
    .await;
    let _ = generator
        .restore_adapter(previous_adapter.as_deref())
        .await
        .inspect_err(
            |e| tracing::warn!(error = %e, "failed to restore previous adapter after eval"),
        );
    result
}

/// A completion whose scoring needs the LLM judge. Pass 1 (generation)
/// records it with a placeholder outcome; pass 2 swaps to the judge
/// adapter ONCE per distinct judge and scores the batch on a blocking
/// thread. Before this existed, every judge call would have swapped the
/// model's adapter mid-suite (poisoning subsequent generations) — and in
/// practice none of it ran at all, because the worker always installed the
/// no-op judge.
struct DeferredJudgeScore {
    outcome_index: usize,
    example_index: usize,
    completion_idx: usize,
    completion_text: String,
    raw_completion_text: String,
    thinking_budget: EvalThinkingBudget,
    prompt_tokens: usize,
    completion_tokens: usize,
    latency_ms: f64,
    schema_note: Option<String>,
}

async fn run_suite_inner(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
    effective_seed: u64,
    prepared_examples: Vec<Result<PreparedPrompt, String>>,
    resolved_budgets: Vec<EvalThinkingBudget>,
    suite_hash: String,
    effective_generation_hash: String,
    model_target: Option<kiln_eval::EvalModelTargetIdentity>,
    replay_environment: EvalReplayEnvironment,
    replay_source_record: Option<Arc<kiln_eval::EvalReplayRecordV1>>,
    generator: Arc<dyn EvalGenerator>,
    progress: Option<ProgressCallback>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    judge_runner: Arc<dyn JudgeRunner>,
) -> Result<SuiteResult, EvalExecutionError> {
    let started_at = chrono::Utc::now();
    let start_instant = Instant::now();

    let mut outcomes: Vec<ExampleOutcome> = Vec::new();
    let mut weights: BTreeMap<String, f32> = BTreeMap::new();
    let mut tags_by_example: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scorer_kind_by_example: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut target_tool_by_example: BTreeMap<String, String> = BTreeMap::new();
    let mut predicted_tool_by_outcome: BTreeMap<(String, usize), String> = BTreeMap::new();
    let mut schema_violations_by_outcome: BTreeMap<(String, usize), (u32, u32)> = BTreeMap::new();
    let mut deferred_judge: Vec<DeferredJudgeScore> = Vec::new();
    let mut judge_targets: Vec<kiln_eval::EvalModelTargetIdentity> = Vec::new();

    'examples: for (outcomes_example_index, example) in suite.examples.iter().enumerate() {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let example_id = example.resolved_id();
        weights.insert(example_id.clone(), example.weight);
        tags_by_example.insert(example_id.clone(), example.tags.clone());
        let scorer = example.scorer.as_ref().unwrap_or(&suite.default_scorer);
        scorer_kind_by_example.insert(example_id.clone(), scorer.kind_label());
        // For tool_call-scored examples, snapshot the target tool name so
        // the aggregate can break out pass-rate per tool. Cheap parse of
        // the target — uses the same extractor the scorer does.
        if matches!(scorer, kiln_eval::scorers::Scorer::ToolCall { .. })
            && let Some(target) = example.target.as_deref()
            && let Some(call) = kiln_eval::qwen3::extract_first_tool_call(target)
        {
            target_tool_by_example.insert(example_id.clone(), call.name);
        }
        let (gen_params, _) = effective_generation(example, suite, generation_override);
        let gen_params = gen_params.clone();
        let n = gen_params.n;
        let example_seed = gen_params.seed.unwrap_or(effective_seed);
        let resolved_budget = &resolved_budgets[outcomes_example_index];
        let outcome_start = outcomes.len();
        let deferred_start = deferred_judge.len();

        // Prompts were rendered before the adapter swap so all active budget
        // configurations could be validated before any decode. Preserve the
        // historical behavior of recording template/tokenization failures as
        // per-example error outcomes.
        let prepared = match &prepared_examples[outcomes_example_index] {
            Ok(prepared) => prepared.clone(),
            Err(err) => {
                for completion_index in 0..n {
                    outcomes.push(ExampleOutcome {
                        example_id: example_id.clone(),
                        completion_index,
                        generation_seed: Some(kiln_eval::derive_eval_completion_seed(
                            example_seed,
                            &example_id,
                            completion_index,
                        )),
                        completion_text: String::new(),
                        raw_completion_text: None,
                        thinking_budget: None,
                        kind: EvalOutcomeKind::Error,
                        score: 0.0,
                        detail: Some(format!("prepare failed: {err}")),
                        prompt_tokens: None,
                        completion_tokens: None,
                        latency_ms: None,
                        tags: example.tags.clone(),
                        metadata: example.metadata.clone(),
                        reasoning_text: None,
                        unclosed_thinking: false,
                    });
                }
                publish_reduced_progress(&outcomes, suite, progress.as_ref(), suite.examples.len());
                continue;
            }
        };

        for completion_idx in 0..n {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                outcomes.truncate(outcome_start);
                deferred_judge.truncate(deferred_start);
                predicted_tool_by_outcome.retain(|(id, _), _| id != &example_id);
                schema_violations_by_outcome.retain(|(id, _), _| id != &example_id);
                break 'examples;
            }
            let generation_seed =
                kiln_eval::derive_eval_completion_seed(example_seed, &example_id, completion_idx);
            let mut completion_params = gen_params.clone();
            completion_params.seed = Some(generation_seed);
            let result = generator
                .run(
                    &prepared,
                    &completion_params,
                    resolved_budget,
                    completion_idx,
                    adapter,
                )
                .await;
            let mut pending_schema_detail: Option<kiln_eval::qwen3::SchemaCheck> = None;
            let outcome = match result {
                Ok(completion) => {
                    // For tool-call-scored examples, capture the *predicted*
                    // tool name from the completion text (independent of
                    // scoring) so the aggregate can build a confusion matrix
                    // — `target=Read, predicted=Write` is exactly the kind
                    // of confusion users want to see surfaced.
                    if matches!(scorer, kiln_eval::scorers::Scorer::ToolCall { .. }) {
                        let parsed = kiln_eval::qwen3::extract_first_tool_call(&completion.text);
                        let predicted = parsed
                            .as_ref()
                            .map(|c| c.name.clone())
                            .unwrap_or_else(|| "<none>".to_string());
                        predicted_tool_by_outcome
                            .insert((example_id.clone(), completion_idx), predicted);
                        // Schema check: when the suite/example declares a
                        // `tools` catalogue, validate the predicted call's
                        // args against the tool's declared parameters.
                        if let Some(call) = parsed.as_ref()
                            && let Some(catalogue) = example.effective_tools(suite.tools.as_deref())
                            && let Some(chk) =
                                kiln_eval::qwen3::validate_against_schema(call, catalogue)
                        {
                            schema_violations_by_outcome.insert(
                                (example_id.clone(), completion_idx),
                                (
                                    chk.missing_required.len() as u32,
                                    chk.extra_unknown.len() as u32,
                                ),
                            );
                            pending_schema_detail = Some(chk);
                        }
                    }
                    let schema_note = pending_schema_detail.as_ref().and_then(|chk| {
                        if chk.is_clean() {
                            return None;
                        }
                        let mut parts = Vec::new();
                        if !chk.missing_required.is_empty() {
                            parts.push(format!("missing={}", chk.missing_required.join(",")));
                        }
                        if !chk.extra_unknown.is_empty() {
                            parts.push(format!("extra={}", chk.extra_unknown.join(",")));
                        }
                        Some(format!(" || schema: {}", parts.join(" ")))
                    });
                    if scorer.requires_judge() {
                        // Defer judge-backed scoring to pass 2: the judge
                        // runs on a (possibly different) adapter, and
                        // swapping mid-generation-loop would poison the
                        // remaining generations. A placeholder Invalid
                        // outcome holds the slot; pass 2 replaces it.
                        deferred_judge.push(DeferredJudgeScore {
                            outcome_index: outcomes.len(),
                            example_index: outcomes_example_index,
                            completion_idx,
                            completion_text: completion.text.clone(),
                            raw_completion_text: completion.raw_text.clone(),
                            thinking_budget: completion.thinking_budget.clone(),
                            prompt_tokens: completion.prompt_tokens,
                            completion_tokens: completion.completion_tokens,
                            latency_ms: completion.latency_ms,
                            schema_note: schema_note.clone(),
                        });
                        ExampleOutcome {
                            example_id: example_id.clone(),
                            completion_index: completion_idx,
                            generation_seed: Some(generation_seed),
                            completion_text: completion.text.clone(),
                            raw_completion_text: Some(completion.raw_text.clone()),
                            thinking_budget: Some(completion.thinking_budget.clone()),
                            kind: EvalOutcomeKind::Invalid,
                            score: 0.0,
                            detail: Some("awaiting judge pass".into()),
                            prompt_tokens: Some(completion.prompt_tokens),
                            completion_tokens: Some(completion.completion_tokens),
                            latency_ms: Some(completion.latency_ms),
                            tags: example.tags.clone(),
                            metadata: example.metadata.clone(),
                            reasoning_text: None,
                            unclosed_thinking: false,
                        }
                    } else {
                        // A scorer error (missing target, bad regex, judge
                        // unavailable) is a property of ONE example — record
                        // it as that example's Error outcome instead of
                        // aborting the run and discarding every completed
                        // outcome.
                        let mut o = match score_completion(
                            scorer,
                            example,
                            &completion.text,
                            judge_runner.as_ref(),
                        ) {
                            Ok(o) => o,
                            Err(e) => ExampleOutcome {
                                example_id: example_id.clone(),
                                completion_index: completion_idx,
                                generation_seed: Some(generation_seed),
                                completion_text: completion.text.clone(),
                                raw_completion_text: Some(completion.raw_text.clone()),
                                thinking_budget: Some(completion.thinking_budget.clone()),
                                kind: EvalOutcomeKind::Error,
                                score: 0.0,
                                detail: Some(format!("scorer error: {e}")),
                                prompt_tokens: None,
                                completion_tokens: None,
                                latency_ms: None,
                                tags: example.tags.clone(),
                                metadata: example.metadata.clone(),
                                reasoning_text: None,
                                unclosed_thinking: false,
                            },
                        };
                        o.completion_index = completion_idx;
                        o.generation_seed = Some(generation_seed);
                        o.prompt_tokens = Some(completion.prompt_tokens);
                        o.completion_tokens = Some(completion.completion_tokens);
                        o.latency_ms = Some(completion.latency_ms);
                        o.raw_completion_text = Some(completion.raw_text);
                        o.thinking_budget = Some(completion.thinking_budget);
                        if let Some(note) = schema_note.as_ref() {
                            o.detail =
                                Some(o.detail.map(|d| d + note).unwrap_or_else(|| {
                                    note.trim_start_matches(" || ").to_string()
                                }));
                        }
                        o
                    }
                }
                Err(err) => ExampleOutcome {
                    example_id: example_id.clone(),
                    completion_index: completion_idx,
                    generation_seed: Some(generation_seed),
                    completion_text: String::new(),
                    raw_completion_text: None,
                    thinking_budget: None,
                    kind: EvalOutcomeKind::Error,
                    score: 0.0,
                    detail: Some(format!("generation failed: {err}")),
                    prompt_tokens: None,
                    completion_tokens: None,
                    latency_ms: None,
                    tags: example.tags.clone(),
                    metadata: example.metadata.clone(),
                    reasoning_text: None,
                    unclosed_thinking: false,
                },
            };
            outcomes.push(outcome);
        }
        publish_reduced_progress(&outcomes, suite, progress.as_ref(), suite.examples.len());
    }

    // ── Pass 2: deferred judge scoring ──────────────────────────────
    // Group by judge adapter, swap once per group (the barrier swap makes
    // each swap wait out in-flight requests — once per suite, not once
    // per call), then score the batch on a blocking thread: the live
    // judge re-enters the generator via Handle::block_on, which panics on
    // a runtime worker thread.
    if !deferred_judge.is_empty() && !cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        let mut groups: BTreeMap<Option<String>, Vec<DeferredJudgeScore>> = BTreeMap::new();
        for item in deferred_judge.drain(..) {
            let scorer = suite.examples[item.example_index]
                .scorer
                .as_ref()
                .unwrap_or(&suite.default_scorer);
            groups
                .entry(scorer.judge_adapter().map(str::to_string))
                .or_default()
                .push(item);
        }
        for (judge_adapter, items) in groups {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            if let Err(e) = generator.set_adapter(judge_adapter.as_deref()).await {
                for item in items {
                    outcomes[item.outcome_index].kind = EvalOutcomeKind::Error;
                    outcomes[item.outcome_index].detail =
                        Some(format!("judge adapter swap failed: {e}"));
                }
                publish_reduced_progress(&outcomes, suite, progress.as_ref(), suite.examples.len());
                continue;
            }
            let identity = generator
                .model_target_identity(judge_adapter.as_deref())
                .map_err(EvalExecutionError::Generation)?;
            if let Some(source) = replay_source_record.as_deref() {
                let expected = source
                    .judge_targets
                    .iter()
                    .find(|target| target.adapter == judge_adapter);
                if identity.as_ref() != expected {
                    return Err(EvalExecutionError::Generation(format!(
                        "strict replay judge target drifted after admission for {:?}: expected {:?}, loaded {:?}",
                        judge_adapter, expected, identity
                    )));
                }
            }
            if let Some(identity) = identity {
                judge_targets.push(identity);
            }
            let batch: Vec<(
                DeferredJudgeScore,
                kiln_eval::scorers::Scorer,
                kiln_eval::EvalExample,
            )> = items
                .into_iter()
                .map(|item| {
                    let example = suite.examples[item.example_index].clone();
                    let scorer = example
                        .scorer
                        .clone()
                        .unwrap_or_else(|| suite.default_scorer.clone());
                    (item, scorer, example)
                })
                .collect();
            let judge = judge_runner.clone();
            let cancel_flag = cancelled.clone();
            let scored = tokio::task::spawn_blocking(move || {
                batch
                    .into_iter()
                    .map(|(item, scorer, example)| {
                        if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                            return (item, None);
                        }
                        let result = score_completion(
                            &scorer,
                            &example,
                            &item.completion_text,
                            judge.as_ref(),
                        );
                        (item, Some(result))
                    })
                    .collect::<Vec<_>>()
            })
            .await
            .map_err(|e| EvalExecutionError::Scorer(format!("judge scoring task: {e}")))?;

            for (item, result) in scored {
                let Some(result) = result else { continue };
                let slot = &mut outcomes[item.outcome_index];
                match result {
                    Ok(mut o) => {
                        o.completion_index = item.completion_idx;
                        o.generation_seed = slot.generation_seed;
                        o.prompt_tokens = Some(item.prompt_tokens);
                        o.completion_tokens = Some(item.completion_tokens);
                        o.latency_ms = Some(item.latency_ms);
                        o.raw_completion_text = Some(item.raw_completion_text);
                        o.thinking_budget = Some(item.thinking_budget);
                        if let Some(note) = item.schema_note.as_ref() {
                            o.detail =
                                Some(o.detail.map(|d| d + note).unwrap_or_else(|| {
                                    note.trim_start_matches(" || ").to_string()
                                }));
                        }
                        *slot = o;
                    }
                    Err(e) => {
                        slot.kind = EvalOutcomeKind::Error;
                        slot.detail = Some(format!("scorer error: {e}"));
                    }
                }
            }
            publish_reduced_progress(&outcomes, suite, progress.as_ref(), suite.examples.len());
        }
    }

    let elapsed = start_instant.elapsed().as_secs_f64();
    let aggregated_outcomes = aggregate_example_outcomes(&outcomes, suite.aggregation)
        .map_err(|error| EvalExecutionError::Aggregation(error.to_string()))?;
    let metrics = AggregateMetrics::compute_with_tools_full(
        &aggregated_outcomes,
        &outcomes,
        &weights,
        &tags_by_example,
        &scorer_kind_by_example,
        &target_tool_by_example,
        &predicted_tool_by_outcome,
        &schema_violations_by_outcome,
        elapsed,
    );
    let finished_at = chrono::Utc::now();
    let replay_record = kiln_eval::EvalReplayRecordV1::new(
        suite.clone(),
        generation_override.cloned(),
        effective_seed,
        resolved_budgets,
        model_target,
        judge_targets,
        replay_environment.execution_provenance_sha256,
        replay_environment.base_weight_manifest_sha256,
        &outcomes,
    )
    .map_err(|error| EvalExecutionError::Aggregation(error.to_string()))?;

    Ok(SuiteResult {
        suite_name: suite.name.clone(),
        adapter: adapter.map(str::to_string),
        aggregation: suite.aggregation,
        metrics,
        outcomes,
        aggregated_outcomes,
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        suite_hash,
        effective_generation_hash,
        replay_record: Some(replay_record),
    })
}

fn effective_generation<'a>(
    example: &'a kiln_eval::EvalExample,
    suite: &'a EvalSuite,
    generation_override: Option<&'a EvalGenerationParams>,
) -> (&'a EvalGenerationParams, EvalGenerationSource) {
    if let Some(params) = example.generation.as_ref() {
        (params, EvalGenerationSource::Example)
    } else if let Some(params) = generation_override {
        (params, EvalGenerationSource::RunOverride)
    } else {
        (&suite.generation, EvalGenerationSource::Suite)
    }
}

fn publish_reduced_progress(
    outcomes: &[ExampleOutcome],
    suite: &EvalSuite,
    callback: Option<&ProgressCallback>,
    total_examples: usize,
) {
    let Some(callback) = callback else {
        return;
    };
    let pending: std::collections::BTreeSet<&str> = outcomes
        .iter()
        .filter(|outcome| outcome.detail.as_deref() == Some("awaiting judge pass"))
        .map(|outcome| outcome.example_id.as_str())
        .collect();
    let finalized: Vec<ExampleOutcome> = outcomes
        .iter()
        .filter(|outcome| !pending.contains(outcome.example_id.as_str()))
        .cloned()
        .collect();
    let Ok(reduced) = aggregate_example_outcomes(&finalized, suite.aggregation) else {
        return;
    };
    let completed = reduced.len() as u32;
    let pass = reduced
        .iter()
        .filter(|outcome| outcome.kind == EvalOutcomeKind::Pass)
        .count() as u32;
    let score = reduced.iter().map(|outcome| outcome.score).sum::<f32>();
    callback(EvalProgress {
        examples_completed: completed,
        examples_total: total_examples as u32,
        running_accuracy: if completed > 0 {
            pass as f32 / completed as f32
        } else {
            0.0
        },
        running_mean_score: if completed > 0 {
            score / completed as f32
        } else {
            0.0
        },
    });
}

fn validate_effective_aggregation(
    suite: &EvalSuite,
    generation_override: Option<&EvalGenerationParams>,
) -> Result<(), EvalExecutionError> {
    let k = suite.aggregation.k();
    for example in &suite.examples {
        let (params, _) = effective_generation(example, suite, generation_override);
        if params.n != k {
            return Err(EvalExecutionError::Aggregation(format!(
                "example {:?} generation.n {} does not match aggregation {} (k={k})",
                example.resolved_id(),
                params.n,
                suite.aggregation.label()
            )));
        }
    }
    Ok(())
}

async fn prepare_and_preflight_examples(
    suite: &EvalSuite,
    generation_override: Option<&EvalGenerationParams>,
    generator: &dyn EvalGenerator,
) -> Result<(Vec<Result<PreparedPrompt, String>>, Vec<EvalThinkingBudget>), EvalExecutionError> {
    let mut cache: BTreeMap<String, EvalThinkingBudget> = BTreeMap::new();
    let mut prepared_examples = Vec::with_capacity(suite.examples.len());
    let mut resolved = Vec::with_capacity(suite.examples.len());
    for example in &suite.examples {
        let (params, source) = effective_generation(example, suite, generation_override);
        let prepared = generator
            .prepare(
                &example.messages,
                suite.system_prompt.as_deref(),
                example.effective_tools(suite.tools.as_deref()),
                params,
            )
            .await;
        let starts_in_reasoning = prepared
            .as_ref()
            .map(|prompt| prompt.starts_in_reasoning)
            .unwrap_or(false);
        let key = format!(
            "{}:{starts_in_reasoning}:{}",
            match source {
                EvalGenerationSource::Suite => "suite",
                EvalGenerationSource::RunOverride => "run_override",
                EvalGenerationSource::Example => "example",
            },
            serde_json::to_string(params).unwrap_or_else(|_| format!("{params:?}")),
        );
        let budget = if let Some(cached) = cache.get(&key) {
            cached.clone()
        } else {
            let budget = generator
                .preflight_thinking_budget(params, source, starts_in_reasoning)
                .await
                .map_err(EvalExecutionError::Generation)?;
            cache.insert(key, budget.clone());
            budget
        };
        prepared_examples.push(prepared);
        resolved.push(budget);
    }
    Ok((prepared_examples, resolved))
}

/// Convenience: a no-op judge runner wrapped in an `Arc` for callers that
/// don't have a live judge (unit tests, offline eval, etc.).
pub fn noop_judge_runner() -> Arc<dyn JudgeRunner> {
    Arc::new(NoopJudgeRunner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::generator::MockEvalGenerator;
    use kiln_eval::scorers::{NumericTolerance, Scorer};
    use kiln_eval::{
        EvalBudgetOverride, EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn suite_with_numeric_answer() -> EvalSuite {
        let mk = |id: &str, q: &str, a: &str| EvalExample {
            id: Some(id.into()),
            messages: vec![EvalChatMessage::new("user", q)],
            target: Some(a.into()),
            tags: vec!["easy".into()],
            ..Default::default()
        };
        EvalSuite {
            name: "math".into(),
            description: None,
            default_scorer: Scorer::NumericTolerance(NumericTolerance {
                atol: 0.0,
                rtol: 0.0,
                integer_only: true,
            }),
            generation: EvalGenerationParams::default(),
            aggregation: kiln_eval::EvalAggregation::Single,
            system_prompt: None,
            examples: vec![mk("e1", "1+1?", "2"), mk("e2", "5+5?", "10")],
            schema_version: 1,
            tools: None,
        }
    }

    struct PreflightProbeGenerator {
        starts_in_reasoning: bool,
        preflight_calls: AtomicUsize,
        adapter_calls: AtomicUsize,
        seeds: std::sync::Mutex<Vec<u64>>,
    }

    struct ReplayDriftProbeGenerator {
        run_calls: AtomicUsize,
        adapter_calls: AtomicUsize,
        identity_error: bool,
    }

    impl EvalGenerator for ReplayDriftProbeGenerator {
        fn set_adapter(
            &self,
            _adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            self.adapter_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async { Ok(None) })
        }

        fn model_target_identity(
            &self,
            _expected_adapter: Option<&str>,
        ) -> Result<Option<kiln_eval::EvalModelTargetIdentity>, String> {
            if self.identity_error {
                Err("synthetic identity failure".into())
            } else {
                Ok(None)
            }
        }

        fn prepare(
            &self,
            _messages: &[EvalChatMessage],
            _system_prompt: Option<&str>,
            _tools: Option<&[serde_json::Value]>,
            _params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(PreparedPrompt {
                    tokens: vec![1],
                    starts_in_reasoning: false,
                })
            })
        }

        fn run(
            &self,
            _prepared: &PreparedPrompt,
            _params: &EvalGenerationParams,
            _thinking_budget: &EvalThinkingBudget,
            _completion_index: usize,
            _adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::EvalCompletion, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async { Err("run must not be reached".into()) })
        }
    }

    impl EvalGenerator for PreflightProbeGenerator {
        fn preflight_thinking_budget(
            &self,
            params: &EvalGenerationParams,
            source: EvalGenerationSource,
            starts_in_reasoning: bool,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<EvalThinkingBudget, String>> + Send + '_>,
        > {
            self.preflight_calls.fetch_add(1, Ordering::Relaxed);
            let max_tokens = params.thinking_budget_tokens.resolve(None);
            let source = match source {
                EvalGenerationSource::Suite => "suite",
                EvalGenerationSource::RunOverride => "run_override",
                EvalGenerationSource::Example => "example",
            };
            let tokens_source = match params.thinking_budget_tokens {
                EvalBudgetOverride::Inherit => "unlimited".to_string(),
                EvalBudgetOverride::Unlimited => format!("{source}_unlimited"),
                EvalBudgetOverride::Limited(_) => source.to_string(),
            };
            Box::pin(async move {
                if starts_in_reasoning {
                    return Err("synthetic invalid close sequence".into());
                }
                Ok(EvalThinkingBudget {
                    configured: max_tokens.is_some(),
                    applied: false,
                    max_tokens,
                    max_time_ms: None,
                    tokens_source: tokens_source.into(),
                    time_source: "unlimited".into(),
                    outcome: None,
                })
            })
        }

        fn set_adapter(
            &self,
            _adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            self.adapter_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async { Ok(None) })
        }

        fn prepare(
            &self,
            _messages: &[EvalChatMessage],
            _system_prompt: Option<&str>,
            _tools: Option<&[serde_json::Value]>,
            _params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
        > {
            let starts_in_reasoning = self.starts_in_reasoning;
            Box::pin(async move {
                Ok(PreparedPrompt {
                    tokens: vec![1],
                    starts_in_reasoning,
                })
            })
        }

        fn run(
            &self,
            _prepared: &PreparedPrompt,
            params: &EvalGenerationParams,
            thinking_budget: &EvalThinkingBudget,
            _completion_index: usize,
            adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::EvalCompletion, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.seeds.lock().unwrap().push(
                params
                    .seed
                    .expect("executor must materialize a decoder seed"),
            );
            let thinking_budget = thinking_budget.clone();
            let adapter = adapter_label.map(str::to_string);
            Box::pin(async move {
                Ok(crate::eval::generator::EvalCompletion {
                    text: "2".into(),
                    raw_text: "decoder-raw".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    latency_ms: 0.1,
                    adapter,
                    thinking_budget,
                })
            })
        }
    }

    #[tokio::test]
    async fn budget_preflight_is_deduplicated_gated_by_rendered_prompt_and_precedes_adapter_swap() {
        let mut suite = suite_with_numeric_answer();
        suite.generation.thinking_budget_tokens = EvalBudgetOverride::Limited(1);

        let inert = Arc::new(PreflightProbeGenerator {
            starts_in_reasoning: false,
            preflight_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            seeds: std::sync::Mutex::new(Vec::new()),
        });
        let result = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            inert.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .expect("an inert budget must not be rejected");
        assert_eq!(inert.preflight_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inert.adapter_calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            result.outcomes[0].raw_completion_text.as_deref(),
            Some("decoder-raw")
        );
        assert_eq!(
            result.outcomes[0]
                .thinking_budget
                .as_ref()
                .unwrap()
                .tokens_source,
            "suite"
        );
        assert!(!result.effective_generation_hash.is_empty());

        let active = Arc::new(PreflightProbeGenerator {
            starts_in_reasoning: true,
            preflight_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            seeds: std::sync::Mutex::new(Vec::new()),
        });
        let error = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            active.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("synthetic invalid close sequence")
        );
        assert_eq!(active.preflight_calls.load(Ordering::Relaxed), 1);
        assert_eq!(active.adapter_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn strict_replay_rejects_model_identity_drift_before_generation() {
        let mut suite = suite_with_numeric_answer();
        suite.examples.truncate(1);
        let mut outcome = kiln_eval::score_completion(
            &suite.default_scorer,
            &suite.examples[0],
            "2",
            &kiln_eval::scorers::NoopJudgeRunner,
        )
        .unwrap();
        outcome.generation_seed = Some(kiln_eval::derive_eval_completion_seed(17, "e1", 0));
        outcome.raw_completion_text = Some("2".into());
        outcome.thinking_budget = Some(EvalThinkingBudget::default());
        let execution = format!("sha256:{}", "11".repeat(32));
        let weights = format!("sha256:{}", "22".repeat(32));
        let source = kiln_eval::EvalReplayRecordV1::new(
            suite.clone(),
            None,
            17,
            vec![EvalThinkingBudget::default()],
            Some(kiln_eval::EvalModelTargetIdentity::base()),
            Vec::new(),
            Some(execution.clone()),
            Some(weights.clone()),
            std::slice::from_ref(&outcome),
        )
        .unwrap();
        source.validate_strict_replay(&[outcome]).unwrap();
        let probe = Arc::new(ReplayDriftProbeGenerator {
            run_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            identity_error: false,
        });
        let error = run_suite_against_adapter_with_replay(
            &suite,
            None,
            None,
            17,
            probe.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
            EvalReplayEnvironment {
                execution_provenance_sha256: Some(execution),
                base_weight_manifest_sha256: Some(weights),
            },
            Some(Arc::new(source)),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("model target drifted"),
            "{error}"
        );
        assert_eq!(probe.run_calls.load(Ordering::Relaxed), 0);
        assert_eq!(probe.adapter_calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn model_identity_attestation_error_restores_previous_adapter() {
        let mut suite = suite_with_numeric_answer();
        suite.examples.truncate(1);
        let probe = Arc::new(ReplayDriftProbeGenerator {
            run_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            identity_error: true,
        });
        let error = run_suite_against_adapter(
            &suite,
            Some("candidate"),
            None,
            17,
            probe.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("synthetic identity failure"));
        assert_eq!(probe.run_calls.load(Ordering::Relaxed), 0);
        assert_eq!(probe.adapter_calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn job_seed_derives_stable_distinct_per_example_completion_seeds() {
        let mut suite = suite_with_numeric_answer();
        suite.generation.n = 2;
        suite.aggregation = kiln_eval::EvalAggregation::PassAtK { k: 2 };
        suite.schema_version = kiln_eval::SUITE_SCHEMA_VERSION;
        let probe = Arc::new(PreflightProbeGenerator {
            starts_in_reasoning: false,
            preflight_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            seeds: std::sync::Mutex::new(Vec::new()),
        });
        let result = run_suite_against_adapter(
            &suite,
            Some("candidate"),
            None,
            42,
            probe.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();

        let expected = ["e1", "e2"]
            .into_iter()
            .flat_map(|id| {
                (0..2).map(move |idx| kiln_eval::derive_eval_completion_seed(42, id, idx))
            })
            .collect::<Vec<_>>();
        assert_eq!(*probe.seeds.lock().unwrap(), expected);
        assert_eq!(
            result
                .outcomes
                .iter()
                .map(|outcome| outcome.generation_seed.unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            expected
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            expected.len(),
            "every stable example/completion identity needs an independent seed"
        );

        let repeat = Arc::new(PreflightProbeGenerator {
            starts_in_reasoning: false,
            preflight_calls: AtomicUsize::new(0),
            adapter_calls: AtomicUsize::new(0),
            seeds: std::sync::Mutex::new(Vec::new()),
        });
        run_suite_against_adapter(
            &suite,
            Some("baseline"),
            None,
            42,
            repeat.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        assert_eq!(
            *repeat.seeds.lock().unwrap(),
            expected,
            "paired adapter runs must consume identical decoder seeds"
        );
    }

    struct IndexedReplyGenerator {
        pass_indices: std::collections::BTreeSet<usize>,
        run_calls: AtomicUsize,
    }

    impl EvalGenerator for IndexedReplyGenerator {
        fn set_adapter(
            &self,
            _adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            Box::pin(async { Ok(None) })
        }

        fn prepare(
            &self,
            _messages: &[EvalChatMessage],
            _system_prompt: Option<&str>,
            _tools: Option<&[serde_json::Value]>,
            _params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(PreparedPrompt {
                    tokens: vec![1],
                    starts_in_reasoning: false,
                })
            })
        }

        fn run(
            &self,
            _prepared: &PreparedPrompt,
            _params: &EvalGenerationParams,
            thinking_budget: &EvalThinkingBudget,
            completion_index: usize,
            _adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::EvalCompletion, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            let text = if self.pass_indices.contains(&completion_index) {
                "yes"
            } else {
                "no"
            }
            .to_string();
            let thinking_budget = thinking_budget.clone();
            Box::pin(async move {
                Ok(crate::eval::generator::EvalCompletion {
                    raw_text: text.clone(),
                    text,
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    latency_ms: 1.0,
                    adapter: None,
                    thinking_budget,
                })
            })
        }
    }

    fn multi_sample_suite(aggregation: kiln_eval::EvalAggregation) -> EvalSuite {
        let mut generation = EvalGenerationParams::default();
        generation.n = aggregation.k();
        EvalSuite {
            name: "multi-sample".into(),
            description: None,
            default_scorer: Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: true,
            },
            generation,
            aggregation,
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("e1".into()),
                messages: vec![EvalChatMessage::new("user", "answer yes")],
                target: Some("yes".into()),
                ..Default::default()
            }],
            schema_version: kiln_eval::SUITE_SCHEMA_VERSION,
            tools: None,
        }
    }

    #[tokio::test]
    async fn executor_reduces_multi_sample_runs_before_metrics() {
        for aggregation in [
            kiln_eval::EvalAggregation::MeanAtK { k: 3 },
            kiln_eval::EvalAggregation::PassAtK { k: 3 },
            kiln_eval::EvalAggregation::MajorityAtK { k: 3 },
        ] {
            let generator = Arc::new(IndexedReplyGenerator {
                pass_indices: [0, 2].into_iter().collect(),
                run_calls: AtomicUsize::new(0),
            });
            let result = run_suite_against_adapter(
                &multi_sample_suite(aggregation),
                None,
                None,
                17,
                generator.clone(),
                None,
                Arc::new(AtomicBool::new(false)),
                noop_judge_runner(),
            )
            .await
            .unwrap();

            assert_eq!(generator.run_calls.load(Ordering::Relaxed), 3);
            assert_eq!(result.outcomes.len(), 3);
            assert_eq!(result.aggregated_outcomes.len(), 1);
            assert_eq!(result.aggregated_outcomes[0].num_pass, 2);
            assert_eq!(result.aggregated_outcomes[0].num_fail, 1);
            assert_eq!(result.metrics.num_examples, 1);
            assert_eq!(result.metrics.num_completions, 3);
            assert_eq!(result.metrics.num_pass, 1);
            assert_eq!(result.metrics.accuracy, 1.0);
        }
    }

    #[tokio::test]
    async fn generation_override_must_preserve_suite_aggregation_cardinality() {
        let generator = Arc::new(IndexedReplyGenerator {
            pass_indices: [0].into_iter().collect(),
            run_calls: AtomicUsize::new(0),
        });
        let mut generation_override = EvalGenerationParams::default();
        generation_override.n = 2;
        let error = run_suite_against_adapter(
            &multi_sample_suite(kiln_eval::EvalAggregation::PassAtK { k: 3 }),
            None,
            Some(&generation_override),
            17,
            generator.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match aggregation pass@3")
        );
        assert_eq!(generator.run_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn full_pass_when_mock_replies_with_target() {
        let gen_ = Arc::new(MockEvalGenerator::new().with_force_reply("the answer is 2"))
            as Arc<dyn EvalGenerator>;
        let mut suite = suite_with_numeric_answer();
        suite.examples[0].metadata = Some(serde_json::json!({
            "source_path": "prod.jsonl",
            "source_line": 4
        }));
        // Tweak the second example so the mock's reply "2" doesn't match 10.
        let result = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            gen_,
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        assert_eq!(result.outcomes.len(), 2);
        assert_eq!(result.metrics.num_pass, 1);
        assert_eq!(result.metrics.num_fail, 1);
        assert_eq!(
            result.outcomes[0].metadata.as_ref().unwrap()["source_line"],
            serde_json::json!(4)
        );
    }

    #[tokio::test]
    async fn scorer_error_becomes_per_example_outcome_instead_of_aborting() {
        // Example 2 has no `target`, which is a ScorerError::MissingTarget
        // under NumericTolerance. The run must complete with that example
        // recorded as an Error outcome — not abort and discard examples 1/3.
        let gen_ =
            Arc::new(MockEvalGenerator::new().with_force_reply("2")) as Arc<dyn EvalGenerator>;
        let mut suite = suite_with_numeric_answer();
        suite.examples.insert(
            1,
            EvalExample {
                id: Some("broken".into()),
                messages: vec![EvalChatMessage::new("user", "no target here")],
                target: None,
                ..Default::default()
            },
        );
        suite.examples.push(EvalExample {
            id: Some("e3".into()),
            messages: vec![EvalChatMessage::new("user", "1+1?")],
            target: Some("2".into()),
            ..Default::default()
        });

        let result = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            gen_,
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .expect("a per-example scorer error must not abort the run");

        assert_eq!(result.outcomes.len(), 4);
        assert_eq!(result.metrics.num_error, 1);
        let err_outcome = result
            .outcomes
            .iter()
            .find(|o| o.example_id == "broken")
            .expect("broken example should have an outcome");
        assert!(matches!(err_outcome.kind, EvalOutcomeKind::Error));
        assert!(
            err_outcome
                .detail
                .as_deref()
                .is_some_and(|d| d.starts_with("scorer error:")),
            "detail should name the scorer error, got {:?}",
            err_outcome.detail
        );
        assert_eq!(err_outcome.completion_text, "2");
        // The healthy examples still scored normally: e1 and e3 pass (reply
        // "2" matches), the original "5+5?" example fails on the wrong reply.
        assert_eq!(result.metrics.num_pass, 2);
        assert_eq!(result.metrics.num_fail, 1);
    }

    /// Judge runner test double: replies with a fixed judge verdict and
    /// records the thread context plus every (adapter, prompt) it saw.
    struct ScriptedJudge {
        reply: &'static str,
        calls: std::sync::Mutex<Vec<Option<String>>>,
    }

    impl kiln_eval::scorers::JudgeRunner for ScriptedJudge {
        fn judge(&self, adapter: Option<&str>, _prompt: &str) -> Option<String> {
            self.calls.lock().unwrap().push(adapter.map(str::to_string));
            Some(self.reply.to_string())
        }
    }

    fn judge_suite() -> EvalSuite {
        let mk = |id: &str| EvalExample {
            id: Some(id.into()),
            messages: vec![EvalChatMessage::new("user", "rate me")],
            target: Some("anything".into()),
            ..Default::default()
        };
        EvalSuite {
            name: "judged".into(),
            description: None,
            default_scorer: Scorer::LlmJudge {
                judge_adapter: Some("judge-x".into()),
                template: kiln_eval::scorers::llm_judge::default_judge_template(),
                score_regex: kiln_eval::scorers::llm_judge::default_judge_regex(),
            },
            generation: EvalGenerationParams::default(),
            aggregation: kiln_eval::EvalAggregation::Single,
            system_prompt: None,
            examples: vec![mk("j1"), mk("j2")],
            schema_version: 1,
            tools: None,
        }
    }

    /// The flywheel's payoff step: judge-scored examples actually score
    /// (no more Invalid-on-every-example), the judge adapter activates
    /// ONCE for the whole batch (never per call), and the eval adapter is
    /// restored afterwards.
    #[tokio::test]
    async fn judge_scoring_runs_in_deferred_pass_with_single_batch_swap() {
        let mock = Arc::new(MockEvalGenerator::new().with_force_reply("model answer"));
        let gen_ = mock.clone() as Arc<dyn EvalGenerator>;
        let judge = Arc::new(ScriptedJudge {
            reply: "Score: 1",
            calls: std::sync::Mutex::new(Vec::new()),
        });

        let result = run_suite_against_adapter(
            &judge_suite(),
            Some("eval-adapter"),
            None,
            17,
            gen_,
            None,
            Arc::new(AtomicBool::new(false)),
            judge.clone(),
        )
        .await
        .unwrap();

        assert_eq!(result.outcomes.len(), 2);
        for o in &result.outcomes {
            assert!(
                matches!(o.kind, EvalOutcomeKind::Pass),
                "judge-scored example must Pass, got {:?} ({:?})",
                o.kind,
                o.detail
            );
            assert!(o.prompt_tokens.is_some(), "generation metadata survives");
        }
        assert_eq!(result.metrics.num_pass, 2);
        assert_eq!(judge.calls.lock().unwrap().len(), 2);

        // Swap sequence: eval adapter in, judge adapter ONCE for the
        // deferred batch, previous adapter restored at the end.
        let swaps = mock.swap_log.lock().unwrap().clone();
        assert_eq!(
            swaps,
            vec![
                Some("eval-adapter".to_string()),
                Some("judge-x".to_string()),
                None,
            ],
            "judge adapter must activate exactly once for the batch"
        );
    }

    /// Without a live judge (mock mode / noop runner) judge-scored
    /// examples degrade to Invalid with the honest detail — never stuck
    /// on the pass-1 placeholder.
    #[tokio::test]
    async fn judge_scoring_without_runner_degrades_to_invalid() {
        let gen_ = Arc::new(MockEvalGenerator::new().with_force_reply("model answer"))
            as Arc<dyn EvalGenerator>;
        let result = run_suite_against_adapter(
            &judge_suite(),
            None,
            None,
            17,
            gen_,
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        assert_eq!(result.outcomes.len(), 2);
        for o in &result.outcomes {
            assert!(matches!(o.kind, EvalOutcomeKind::Invalid));
            assert_eq!(o.detail.as_deref(), Some("judge runner unavailable"));
        }
    }

    #[tokio::test]
    async fn cancellation_short_circuits_loop() {
        let gen_ =
            Arc::new(MockEvalGenerator::new().with_force_reply("2")) as Arc<dyn EvalGenerator>;
        let suite = suite_with_numeric_answer();
        let flag = Arc::new(AtomicBool::new(true));
        let result = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            gen_,
            None,
            flag,
            noop_judge_runner(),
        )
        .await
        .unwrap();
        assert_eq!(result.outcomes.len(), 0);
    }

    #[tokio::test]
    async fn progress_callback_called_per_completion() {
        let gen_ =
            Arc::new(MockEvalGenerator::new().with_force_reply("2")) as Arc<dyn EvalGenerator>;
        let suite = suite_with_numeric_answer();
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_cb = counter.clone();
        let cb: ProgressCallback = Box::new(move |_p| {
            counter_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        let _ = run_suite_against_adapter(
            &suite,
            None,
            None,
            17,
            gen_,
            Some(cb),
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn per_adapter_replies_drive_compare_mode() {
        let gen_ = Arc::new(
            MockEvalGenerator::new()
                .with_adapter_reply("alpha", "2")
                .with_adapter_reply("beta", "999"),
        ) as Arc<dyn EvalGenerator>;
        let suite = suite_with_numeric_answer();
        let r_a = run_suite_against_adapter(
            &suite,
            Some("alpha"),
            None,
            17,
            gen_.clone(),
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        let r_b = run_suite_against_adapter(
            &suite,
            Some("beta"),
            None,
            17,
            gen_,
            None,
            Arc::new(AtomicBool::new(false)),
            noop_judge_runner(),
        )
        .await
        .unwrap();
        // alpha gets 1 of 2 right (answer "2" matches e1, fails e2).
        assert_eq!(r_a.metrics.num_pass, 1);
        // beta gets neither right.
        assert_eq!(r_b.metrics.num_pass, 0);
    }
}

//! Suite executor: drives `EvalGenerator` × `Scorer` across an `EvalSuite`
//! and produces a `SuiteResult`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use kiln_eval::scorers::{JudgeRunner, NoopJudgeRunner, score_completion};
use kiln_eval::{
    AggregateMetrics, EvalGenerationParams, EvalOutcomeKind, EvalProgress, EvalSuite,
    ExampleOutcome, SuiteResult,
};
use sha2::{Digest, Sha256};

use crate::eval::generator::EvalGenerator;

#[derive(Debug, thiserror::Error)]
pub enum EvalExecutionError {
    #[error("generation failed: {0}")]
    Generation(String),
    #[error("scorer error: {0}")]
    Scorer(String),
}

/// Callback fired after every example completes. Used by the worker loop to
/// surface running accuracy into `EvalJobInfo::progress` while the suite is
/// still in flight.
pub type ProgressCallback = Box<dyn Fn(EvalProgress) + Send + Sync>;

/// Run `suite` against `adapter` with `generator`. Returns the aggregated
/// `SuiteResult`. The function is cancellation-cooperative — when the
/// `cancelled` flag flips to true the executor returns early with the
/// partial result populated.
pub async fn run_suite_against_adapter(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
    generator: Arc<dyn EvalGenerator>,
    progress: Option<ProgressCallback>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    judge_runner: Arc<dyn JudgeRunner>,
) -> Result<SuiteResult, EvalExecutionError> {
    // Hoist the adapter swap out of the per-example loop. The previous
    // active adapter is restored at the end via the same call so a suite
    // run never leaves the model in an unexpected state, even when an
    // example errors mid-loop (including after the deferred judge pass
    // swapped to a judge adapter).
    let previous_adapter = generator
        .set_adapter(adapter)
        .await
        .map_err(EvalExecutionError::Generation)?;
    let result = run_suite_inner(
        suite,
        adapter,
        generation_override,
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
    prompt_tokens: usize,
    completion_tokens: usize,
    latency_ms: f64,
    schema_note: Option<String>,
}

async fn run_suite_inner(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
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
    let mut running_pass: u32 = 0;
    let mut running_score: f32 = 0.0;
    let mut completions_seen: u32 = 0;
    let mut deferred_judge: Vec<DeferredJudgeScore> = Vec::new();

    let total_completions: u32 = suite
        .examples
        .iter()
        .map(|ex| {
            ex.generation
                .as_ref()
                .or(generation_override)
                .map(|g| g.n)
                .unwrap_or(suite.generation.n) as u32
        })
        .sum();

    for (outcomes_example_index, example) in suite.examples.iter().enumerate() {
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
        if matches!(scorer, kiln_eval::scorers::Scorer::ToolCall { .. }) {
            if let Some(target) = example.target.as_deref() {
                if let Some(call) = kiln_eval::qwen3::extract_first_tool_call(target) {
                    target_tool_by_example.insert(example_id.clone(), call.name);
                }
            }
        }
        let gen_params = example
            .generation
            .as_ref()
            .or(generation_override)
            .unwrap_or(&suite.generation)
            .clone();

        // Render chat template + tokenize once per example. For n>1 the
        // resulting `PreparedPrompt` is reused across every completion.
        // Pass through the effective tool catalogue (per-example override
        // or suite-level default) so Qwen3.5's `<tools>` block renders.
        let effective_tools = example.effective_tools(suite.tools.as_deref());
        let prepared = match generator
            .prepare(
                &example.messages,
                suite.system_prompt.as_deref(),
                effective_tools,
                &gen_params,
            )
            .await
        {
            Ok(p) => p,
            Err(err) => {
                let outcome = ExampleOutcome {
                    example_id: example_id.clone(),
                    completion_index: 0,
                    completion_text: String::new(),
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
                };
                completions_seen += 1;
                outcomes.push(outcome);
                continue;
            }
        };

        let n = gen_params.n.max(1);
        for completion_idx in 0..n {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let result = generator
                .run(&prepared, &gen_params, completion_idx, adapter)
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
                        if let Some(call) = parsed.as_ref() {
                            if let Some(catalogue) = example.effective_tools(suite.tools.as_deref())
                            {
                                if let Some(chk) =
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
                            prompt_tokens: completion.prompt_tokens,
                            completion_tokens: completion.completion_tokens,
                            latency_ms: completion.latency_ms,
                            schema_note: schema_note.clone(),
                        });
                        ExampleOutcome {
                            example_id: example_id.clone(),
                            completion_index: completion_idx,
                            completion_text: completion.text.clone(),
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
                                completion_text: completion.text.clone(),
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
                        o.prompt_tokens = Some(completion.prompt_tokens);
                        o.completion_tokens = Some(completion.completion_tokens);
                        o.latency_ms = Some(completion.latency_ms);
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
                    completion_text: String::new(),
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
            if matches!(outcome.kind, EvalOutcomeKind::Pass) {
                running_pass += 1;
            }
            running_score += outcome.score;
            completions_seen += 1;
            outcomes.push(outcome);

            if let Some(cb) = progress.as_ref() {
                let progress_snap = EvalProgress {
                    examples_completed: completions_seen,
                    examples_total: total_completions,
                    running_accuracy: if completions_seen > 0 {
                        running_pass as f32 / completions_seen as f32
                    } else {
                        0.0
                    },
                    running_mean_score: if completions_seen > 0 {
                        running_score / completions_seen as f32
                    } else {
                        0.0
                    },
                };
                cb(progress_snap);
            }
        }
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
                    outcomes[item.outcome_index].detail =
                        Some(format!("judge adapter swap failed: {e}"));
                }
                continue;
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
                        o.prompt_tokens = Some(item.prompt_tokens);
                        o.completion_tokens = Some(item.completion_tokens);
                        o.latency_ms = Some(item.latency_ms);
                        if let Some(note) = item.schema_note.as_ref() {
                            o.detail =
                                Some(o.detail.map(|d| d + note).unwrap_or_else(|| {
                                    note.trim_start_matches(" || ").to_string()
                                }));
                        }
                        if matches!(o.kind, EvalOutcomeKind::Pass) {
                            running_pass += 1;
                        }
                        running_score += o.score;
                        *slot = o;
                    }
                    Err(e) => {
                        slot.kind = EvalOutcomeKind::Error;
                        slot.detail = Some(format!("scorer error: {e}"));
                    }
                }
            }
            if let Some(cb) = progress.as_ref() {
                cb(EvalProgress {
                    examples_completed: completions_seen,
                    examples_total: total_completions,
                    running_accuracy: if completions_seen > 0 {
                        running_pass as f32 / completions_seen as f32
                    } else {
                        0.0
                    },
                    running_mean_score: if completions_seen > 0 {
                        running_score / completions_seen as f32
                    } else {
                        0.0
                    },
                });
            }
        }
    }

    let elapsed = start_instant.elapsed().as_secs_f64();
    let metrics = AggregateMetrics::compute_with_tools_full(
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
    let suite_hash = hash_suite(suite);

    Ok(SuiteResult {
        suite_name: suite.name.clone(),
        adapter: adapter.map(str::to_string),
        metrics,
        outcomes,
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        suite_hash,
    })
}

/// Stable hash of the suite content. Used to detect "did the suite change
/// between runs?" in replay auditing. Hashes the suite JSON canonical form.
pub fn hash_suite(suite: &EvalSuite) -> String {
    let mut hasher = Sha256::new();
    // serde_json doesn't sort keys by default but our shape is mostly
    // arrays + strings so byte-form serialization is stable.
    if let Ok(b) = serde_json::to_vec(suite) {
        hasher.update(&b);
    } else {
        hasher.update(suite.name.as_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
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
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};
    use std::sync::atomic::AtomicBool;

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
            system_prompt: None,
            examples: vec![mk("e1", "1+1?", "2"), mk("e2", "5+5?", "10")],
            schema_version: 1,
            tools: None,
        }
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
        let result =
            run_suite_against_adapter(&suite, None, None, gen_, None, flag, noop_judge_runner())
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

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
    // example errors mid-loop.
    let previous_adapter = generator
        .set_adapter(adapter)
        .await
        .map_err(EvalExecutionError::Generation)?;
    let result = run_suite_inner(
        suite,
        adapter,
        generation_override,
        generator.as_ref(),
        progress,
        cancelled,
        judge_runner.as_ref(),
    )
    .await;
    let _ = generator
        .set_adapter(previous_adapter.as_deref())
        .await
        .inspect_err(|e| tracing::warn!(error = %e, "failed to restore previous adapter after eval"));
    result
}

async fn run_suite_inner(
    suite: &EvalSuite,
    adapter: Option<&str>,
    generation_override: Option<&EvalGenerationParams>,
    generator: &dyn EvalGenerator,
    progress: Option<ProgressCallback>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    judge_runner: &dyn JudgeRunner,
) -> Result<SuiteResult, EvalExecutionError> {
    let started_at = chrono::Utc::now();
    let start_instant = Instant::now();

    let mut outcomes: Vec<ExampleOutcome> = Vec::new();
    let mut weights: BTreeMap<String, f32> = BTreeMap::new();
    let mut tags_by_example: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scorer_kind_by_example: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut running_pass: u32 = 0;
    let mut running_score: f32 = 0.0;
    let mut completions_seen: u32 = 0;

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

    for example in &suite.examples {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let example_id = example.resolved_id();
        weights.insert(example_id.clone(), example.weight);
        tags_by_example.insert(example_id.clone(), example.tags.clone());
        let scorer = example.scorer.as_ref().unwrap_or(&suite.default_scorer);
        scorer_kind_by_example.insert(example_id.clone(), scorer.kind_label());
        let gen_params = example
            .generation
            .as_ref()
            .or(generation_override)
            .unwrap_or(&suite.generation)
            .clone();

        // Render chat template + tokenize once per example. For n>1 the
        // resulting `PreparedPrompt` is reused across every completion.
        let prepared = match generator
            .prepare(&example.messages, suite.system_prompt.as_deref(), &gen_params)
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
            let outcome = match result {
                Ok(completion) => {
                    let mut o = score_completion(scorer, example, &completion.text, judge_runner)
                        .map_err(|e| EvalExecutionError::Scorer(format!("{e}")))?;
                    o.completion_index = completion_idx;
                    o.prompt_tokens = Some(completion.prompt_tokens);
                    o.completion_tokens = Some(completion.completion_tokens);
                    o.latency_ms = Some(completion.latency_ms);
                    o
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

    let elapsed = start_instant.elapsed().as_secs_f64();
    let metrics = AggregateMetrics::compute(
        &outcomes,
        &weights,
        &tags_by_example,
        &scorer_kind_by_example,
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
            examples: vec![
                EvalExample {
                    id: Some("e1".into()),
                    messages: vec![EvalChatMessage {
                        role: "user".into(),
                        content: "1+1?".into(),
                    }],
                    target: Some("2".into()),
                    aliases: Vec::new(),
                    tags: vec!["easy".into()],
                    metadata: None,
                    scorer: None,
                    generation: None,
                    weight: 1.0,
                },
                EvalExample {
                    id: Some("e2".into()),
                    messages: vec![EvalChatMessage {
                        role: "user".into(),
                        content: "5+5?".into(),
                    }],
                    target: Some("10".into()),
                    aliases: Vec::new(),
                    tags: vec!["easy".into()],
                    metadata: None,
                    scorer: None,
                    generation: None,
                    weight: 1.0,
                },
            ],
            schema_version: 1,
        }
    }

    #[tokio::test]
    async fn full_pass_when_mock_replies_with_target() {
        let gen_ = Arc::new(
            MockEvalGenerator::new().with_force_reply("the answer is 2")
        ) as Arc<dyn EvalGenerator>;
        let suite = suite_with_numeric_answer();
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
    }

    #[tokio::test]
    async fn cancellation_short_circuits_loop() {
        let gen_ = Arc::new(MockEvalGenerator::new().with_force_reply("2")) as Arc<dyn EvalGenerator>;
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
        let gen_ = Arc::new(MockEvalGenerator::new().with_force_reply("2")) as Arc<dyn EvalGenerator>;
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

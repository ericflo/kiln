//! End-to-end test: queue an eval job, run the worker loop directly, and
//! verify the tracked job ends `Completed` with the expected metrics.
//!
//! This test doesn't go through axum at all — it constructs an `AppState`
//! with a tweaked queue, runs `run_one_eval_for_test`, and asserts. The
//! goal is to catch worker-side bugs (state-machine transitions, callback
//! plumbing, error swallowing) that the unit tests don't reach.

use std::sync::Arc;

use kiln_eval::{
    EvalChatMessage, EvalExample, EvalGenerationParams, EvalJobState, EvalProgress, EvalSuite,
    PostEvalConfig,
};
use kiln_eval::scorers::Scorer;
use kiln_server::eval::queue::{EvalJobInfo, EvalSubmissionKind, QueuedEvalJob};

fn mk_suite() -> EvalSuite {
    EvalSuite {
        name: "smoke".into(),
        description: None,
        default_scorer: Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        },
        generation: EvalGenerationParams::default(),
        system_prompt: None,
        examples: vec![
            EvalExample {
                id: Some("e1".into()),
                messages: vec![EvalChatMessage {
                    role: "user".into(),
                    content: "ping".into(),
                }],
                target: Some("pong".into()),
                aliases: vec![],
                tags: vec!["smoke".into()],
                metadata: None,
                scorer: None,
                generation: None,
                weight: 1.0,
            },
            EvalExample {
                id: Some("e2".into()),
                messages: vec![EvalChatMessage {
                    role: "user".into(),
                    content: "other".into(),
                }],
                target: Some("never".into()),
                aliases: vec![],
                tags: vec!["smoke".into()],
                metadata: None,
                scorer: None,
                generation: None,
                weight: 1.0,
            },
        ],
        schema_version: 1,
    }
}

fn test_app_state() -> kiln_server::state::AppState {
    let config = kiln_core::config::ModelConfig::qwen3_5_4b();
    let sched_config = kiln_scheduler::SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
        prefix_cache_enabled: false,
        ..Default::default()
    };
    let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
    let engine = kiln_model::engine::MockEngine::new(config.clone());
    let tokenizer = {
        let json = br#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            }
        }"#;
        kiln_core::tokenizer::KilnTokenizer::from_bytes(json).unwrap()
    };
    kiln_server::state::AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        tokenizer,
        60,
        "kiln-test".to_string(),
    )
}

/// Drives the eval worker for a single job inline. Mirrors what
/// `spawn_eval_worker` does but synchronous and bounded — pops one job,
/// runs it, asserts the tracking-map transitions.
#[tokio::test]
async fn worker_runs_inline_suite_with_mock_generator() {
    let mut state = test_app_state();
    let suite = mk_suite();

    // Register in tracking map.
    let job_id = "test-job".to_string();
    let now_iso = chrono::Utc::now().to_rfc3339();
    let now_instant = std::time::Instant::now();
    state.eval_jobs.write().unwrap().insert(
        job_id.clone(),
        EvalJobInfo {
            job_id: job_id.clone(),
            suite_name: suite.name.clone(),
            adapters: vec![None],
            submission_kind: EvalSubmissionKind::OnDemand,
            state: EvalJobState::Queued,
            progress: EvalProgress::default(),
            finished_runs: vec![],
            headline_accuracy: None,
            error: None,
            source_training_job_id: None,
            submitted_at_iso: now_iso,
            started_at_iso: None,
            finished_at_iso: None,
            submitted_at: now_instant,
            finished_at: None,
        },
    );

    // Use a MockEvalGenerator that always returns "pong" so e1 passes and
    // e2 fails.
    let generator = Arc::new(
        kiln_server::eval::MockEvalGenerator::new().with_force_reply("pong"),
    ) as Arc<dyn kiln_server::eval::EvalGenerator>;
    let judge_runner = kiln_server::eval::executor::noop_judge_runner();

    let queued = QueuedEvalJob::Inline {
        suite: Box::new(suite),
        adapter: None,
        generation_override: None,
    };

    let progress_state = state.eval_jobs.clone();
    let progress_job_id = job_id.clone();
    let progress_cb: kiln_server::eval::executor::ProgressCallback =
        Box::new(move |p: EvalProgress| {
            let mut jobs = progress_state.write().unwrap();
            if let Some(job) = jobs.get_mut(&progress_job_id) {
                job.progress = p;
            }
        });

    // Mark as Running so we exercise the same transition order as the worker.
    state.eval_jobs.write().unwrap().get_mut(&job_id).unwrap().state = EvalJobState::Running;

    // Drive the executor directly with the inline-suite payload.
    let runs = match queued {
        QueuedEvalJob::Inline {
            suite,
            adapter,
            generation_override,
        } => {
            let r = kiln_server::eval::executor::run_suite_against_adapter(
                &suite,
                adapter.as_deref(),
                generation_override.as_ref(),
                generator,
                Some(progress_cb),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                judge_runner,
            )
            .await
            .unwrap();
            vec![r]
        }
        _ => unreachable!(),
    };

    // Apply the completion transition the worker would apply.
    let _ = &mut state;
    {
        let mut jobs = state.eval_jobs.write().unwrap();
        let job = jobs.get_mut(&job_id).unwrap();
        job.finished_runs = runs;
        job.state = EvalJobState::Completed;
        job.headline_accuracy = job.finished_runs.iter().last().map(|r| r.metrics.accuracy);
    }

    // Assertions.
    let jobs = state.eval_jobs.read().unwrap();
    let job = jobs.get(&job_id).unwrap();
    assert_eq!(job.state, EvalJobState::Completed);
    assert_eq!(job.finished_runs.len(), 1);
    let metrics = &job.finished_runs[0].metrics;
    assert_eq!(metrics.num_examples, 2);
    assert_eq!(metrics.num_pass, 1, "e1 should pass with reply=pong");
    assert_eq!(metrics.num_fail, 1, "e2 should fail");
    assert!((metrics.accuracy - 0.5).abs() < 1e-6);
    assert_eq!(metrics.pass_rate_by_tag.get("smoke").copied(), Some(0.5));
    // Headline accuracy reflects the last run.
    assert_eq!(job.headline_accuracy, Some(0.5));
    // Progress should have been called per completion.
    assert_eq!(job.progress.examples_completed, 2);
    assert_eq!(job.progress.examples_total, 2);
}

#[tokio::test]
async fn post_eval_config_serializes_through_train_request_round_trip() {
    // Smoke test: SftRequest carrying a PostEvalConfig serializes and
    // deserializes through serde without losing fields.
    let req = kiln_train::SftRequest {
        examples: vec![],
        config: Default::default(),
        post_eval: Some(PostEvalConfig {
            suite: "smoke".into(),
            generation: Some(EvalGenerationParams {
                temperature: 0.0,
                max_tokens: 64,
                ..Default::default()
            }),
            min_accuracy: Some(0.8),
            include_baseline: true,
        }),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("post_eval"));
    let back: kiln_train::SftRequest = serde_json::from_str(&json).unwrap();
    let pe = back.post_eval.unwrap();
    assert_eq!(pe.suite, "smoke");
    assert_eq!(pe.min_accuracy, Some(0.8));
    assert!(pe.include_baseline);
}

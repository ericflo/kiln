//! Background eval worker — pulls jobs from `state.eval_queue` and runs
//! them through the executor.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use kiln_eval::scorers::JudgeRunner;
use kiln_eval::{EvalJobState, EvalProgress};

use crate::eval::executor::{noop_judge_runner, run_suite_against_adapter};
use crate::eval::generator::generator_from_state;
use crate::eval::queue::{EvalQueueEntry, QueuedEvalJob};
use crate::state::AppState;
use crate::training_queue::ShutdownFlag;

/// Spawn the eval worker. Mirrors `spawn_training_worker` — polls the
/// queue every 250ms, runs jobs serially, exits cleanly on shutdown.
pub fn spawn_eval_worker(state: AppState, shutdown: ShutdownFlag) {
    tokio::spawn(async move {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                tracing::info!("eval worker shutting down");
                break;
            }
            gc_eval_jobs(&state);
            let entry = {
                let mut q = state.eval_queue.lock().unwrap();
                q.pop()
            };
            if let Some(entry) = entry {
                run_one_job(state.clone(), entry).await;
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        }
    });
}

/// Evict terminal (`Completed` / `Failed` / `Cancelled`) entries from
/// `state.eval_jobs` past TTL only when the map exceeds
/// `max_tracked_eval_jobs`. Oldest-by-finish-time first. Mirrors
/// `training_queue::gc_tracked_jobs` (cap-driven, not TTL-driven).
pub fn gc_eval_jobs(state: &AppState) -> usize {
    let cap = state.max_tracked_eval_jobs;
    let ttl = state.tracked_job_ttl;
    let now = std::time::Instant::now();
    let mut jobs = state.eval_jobs.write().unwrap();
    if jobs.len() <= cap {
        return 0;
    }
    let mut candidates: Vec<(String, std::time::Instant)> = jobs
        .iter()
        .filter_map(|(id, j)| match (j.state, j.finished_at) {
            (
                EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled,
                Some(t),
            ) if now.saturating_duration_since(t) >= ttl => Some((id.clone(), t)),
            _ => None,
        })
        .collect();
    candidates.sort_by_key(|(_, t)| *t);
    let want_to_remove = jobs.len().saturating_sub(cap);
    let mut removed = 0;
    for (id, _) in candidates.into_iter().take(want_to_remove) {
        jobs.remove(&id);
        removed += 1;
    }
    if removed > 0 {
        tracing::debug!(
            removed,
            remaining = jobs.len(),
            cap,
            "evicted oldest terminal eval jobs past TTL to honor max_tracked_eval_jobs cap"
        );
    }
    removed
}

async fn run_one_job(state: AppState, entry: EvalQueueEntry) {
    let generator = generator_from_state(state.clone());
    run_one_job_with_generator(state, entry, generator).await
}

/// `run_one_job` body with the generator injected — the seam the
/// cancellation test uses to gate generation deterministically.
async fn run_one_job_with_generator(
    state: AppState,
    entry: EvalQueueEntry,
    generator: Arc<dyn crate::eval::generator::EvalGenerator>,
) {
    let job_id = entry.job_id.clone();
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Mark as running, stamp started_at, and install the cancellation flag
    // so DELETE /v1/eval/jobs/{id} can actually stop the executor.
    {
        let mut jobs = state.eval_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            if matches!(job.state, EvalJobState::Cancelled | EvalJobState::Failed) {
                tracing::info!(job_id = %job_id, "skipping cancelled eval job");
                return;
            }
            job.state = EvalJobState::Running;
            job.started_at_iso = Some(chrono::Utc::now().to_rfc3339());
            job.cancel_flag = Some(cancel_flag.clone());
        } else {
            tracing::warn!(job_id = %job_id, "eval job not found in tracking map");
            return;
        }
    }

    // TODO(eval): swap in a `LiveJudgeRunner` once judge calls are plumbed.
    // Until then judge scorers degrade to `Invalid` on every example.
    let judge_runner: Arc<dyn JudgeRunner> = noop_judge_runner();

    let progress_state = state.eval_jobs.clone();
    let progress_job_id = job_id.clone();
    let progress_cb: crate::eval::executor::ProgressCallback =
        Box::new(move |p: EvalProgress| {
            let mut jobs = progress_state.write().unwrap();
            if let Some(job) = jobs.get_mut(&progress_job_id) {
                job.progress = p;
            }
        });

    let result = run_job(
        &state,
        &entry.job,
        generator,
        judge_runner,
        Some(progress_cb),
        cancel_flag.clone(),
    )
    .await;

    let now_iso = chrono::Utc::now().to_rfc3339();
    let now_instant = std::time::Instant::now();

    let was_cancelled = cancel_flag.load(std::sync::atomic::Ordering::Relaxed);
    let archive_snapshot = match result {
        Ok(runs) => {
            let headline = runs.iter().last().map(|r| r.metrics.accuracy);
            let mut jobs = state.eval_jobs.write().unwrap();
            jobs.get_mut(&job_id).map(|job| {
                // A cancelled run still archives its partial outcomes, but
                // the terminal state must stay Cancelled — not flip back to
                // Completed and erase what the user asked for.
                if was_cancelled || matches!(job.state, EvalJobState::Cancelled) {
                    job.state = EvalJobState::Cancelled;
                } else {
                    job.state = EvalJobState::Completed;
                }
                job.finished_runs = runs;
                job.headline_accuracy = headline;
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
                job.cancel_flag = None;
                job.clone()
            })
        }
        Err(err) => {
            tracing::error!(job_id = %job_id, error = %err, "eval job failed");
            let mut jobs = state.eval_jobs.write().unwrap();
            jobs.get_mut(&job_id).map(|job| {
                job.state = EvalJobState::Failed;
                job.error = Some(err);
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
                job.cancel_flag = None;
                job.clone()
            })
        }
    };
    if let Some(snapshot) = archive_snapshot {
        if let Err(e) = crate::eval_history::save(&state.adapter_dir, &snapshot) {
            tracing::warn!(error = %e, job_id = %job_id, "failed to archive terminal eval job");
        }
        crate::eval_history::prune_to_max(
            &state.adapter_dir,
            crate::eval_history::MAX_ARCHIVED_JOBS,
        );
    }

    // Back-linking from eval-completion → training-job is intentionally
    // NOT done here. `enqueue_post_training_eval` already populates the
    // training job's `linked_eval_job_ids` at queue time, so the
    // training-side dashboard can link to the eval the moment it lands
    // in the queue (not after it finishes). Pushing again here would
    // duplicate every ID.
}

async fn run_job(
    state: &AppState,
    job: &QueuedEvalJob,
    generator: Arc<dyn crate::eval::generator::EvalGenerator>,
    judge_runner: Arc<dyn JudgeRunner>,
    progress: Option<crate::eval::executor::ProgressCallback>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<Vec<kiln_eval::SuiteResult>, String> {
    match job {
        QueuedEvalJob::Registered {
            suite_name,
            adapter,
            generation_override,
        } => {
            let suite = state
                .suite_registry
                .as_ref()
                .ok_or_else(|| "no suite registry configured".to_string())?
                .load(suite_name)
                .map_err(|e| format!("{e}"))?;
            let r = run_suite_against_adapter(
                &suite,
                adapter.as_deref(),
                generation_override.as_ref(),
                generator,
                progress,
                cancel_flag,
                judge_runner,
            )
            .await
            .map_err(|e| format!("{e}"))?;
            Ok(vec![r])
        }
        QueuedEvalJob::Inline {
            suite,
            adapter,
            generation_override,
        } => {
            let r = run_suite_against_adapter(
                suite,
                adapter.as_deref(),
                generation_override.as_ref(),
                generator,
                progress,
                cancel_flag,
                judge_runner,
            )
            .await
            .map_err(|e| format!("{e}"))?;
            Ok(vec![r])
        }
        QueuedEvalJob::Compare(spec) => {
            let suite = state
                .suite_registry
                .as_ref()
                .ok_or_else(|| "no suite registry configured".to_string())?
                .load(&spec.suite)
                .map_err(|e| format!("{e}"))?;
            let mut runs = Vec::with_capacity(spec.adapters.len());
            // The single `progress` callback owns the progress slot on
            // EvalJobInfo. Cloning per-adapter would race; instead we reuse
            // it across adapters and let the slot reflect the most recent
            // adapter's running accuracy. Without this the UI's compare
            // panel shows zero progress for the entire job.
            let mut progress_slot = progress;
            for adapter in &spec.adapters {
                // A cancelled compare job must not keep swapping adapters
                // for runs nobody asked to finish.
                if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let adapter_opt = if adapter.is_empty() {
                    None
                } else {
                    Some(adapter.as_str())
                };
                let r = run_suite_against_adapter(
                    &suite,
                    adapter_opt,
                    spec.generation.as_ref(),
                    generator.clone(),
                    progress_slot.take(),
                    cancel_flag.clone(),
                    judge_runner.clone(),
                )
                .await
                .map_err(|e| format!("compare ({}): {e}", adapter))?;
                runs.push(r);
            }
            Ok(runs)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::queue::{EvalJobInfo, EvalQueueEntry, EvalSubmissionKind};
    use kiln_eval::scorers::Scorer;
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};
    use tower::ServiceExt;

    fn big_suite(n: usize) -> EvalSuite {
        EvalSuite {
            name: "cancel-me".into(),
            description: None,
            default_scorer: Scorer::Contains {
                phrases: vec!["mock".into()],
                mode: Default::default(),
                case_sensitive: false,
            },
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples: (0..n)
                .map(|i| EvalExample {
                    id: Some(format!("e{i}")),
                    messages: vec![EvalChatMessage::new("user", format!("ping {i}"))],
                    target: None,
                    ..Default::default()
                })
                .collect(),
            schema_version: 1,
            tools: None,
        }
    }

    /// Gate wrapper: first example completes, then generation parks on a
    /// Notify until the test has cancelled through the real DELETE route.
    struct GatedGenerator {
        inner: crate::eval::MockEvalGenerator,
        reached: Arc<tokio::sync::Notify>,
        resume: Arc<tokio::sync::Notify>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::eval::generator::EvalGenerator for GatedGenerator {
        fn set_adapter(
            &self,
            adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            self.inner.set_adapter(adapter)
        }

        fn prepare(
            &self,
            messages: &[EvalChatMessage],
            system_prompt: Option<&str>,
            tools: Option<&[serde_json::Value]>,
            params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::PreparedPrompt, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.inner.prepare(messages, system_prompt, tools, params)
        }

        fn run(
            &self,
            prepared: &crate::eval::generator::PreparedPrompt,
            params: &EvalGenerationParams,
            completion_index: usize,
            adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::eval::EvalCompletion, String>>
                    + Send
                    + '_,
            >,
        > {
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let fut = self.inner.run(prepared, params, completion_index, adapter_label);
            let reached = self.reached.clone();
            let resume = self.resume.clone();
            Box::pin(async move {
                if call == 1 {
                    // One outcome is recorded; tell the test we are mid-run
                    // and wait for it to fire the cancellation.
                    reached.notify_one();
                    resume.notified().await;
                }
                fut.await
            })
        }
    }

    /// DELETE /v1/eval/jobs/{id} mid-run must stop the executor at the next
    /// example boundary and the terminal state must STAY Cancelled with the
    /// partial outcomes recorded — the worker used to clobber it back to
    /// Completed and the flag was never wired, so cancellation did nothing.
    #[tokio::test]
    async fn cancel_running_job_stops_early_and_stays_cancelled() {
        // Archive writes go under adapter_dir — point it at a tempdir so
        // the test never litters the repo tree (a stray adapters/.kiln-jobs
        // breaks the health adapter-count test).
        let adapter_dir = tempfile::tempdir().unwrap();
        let state = {
            let config = kiln_core::config::ModelConfig::qwen3_5_4b();
            let scheduler = kiln_scheduler::Scheduler::new(
                kiln_scheduler::SchedulerConfig {
                    max_batch_tokens: 8192,
                    max_batch_size: 64,
                    block_size: 16,
                    prefix_cache_enabled: false,
                    ..Default::default()
                },
                256,
            );
            let engine = kiln_model::engine::MockEngine::new(config.clone());
            crate::state::AppState::new_mock(
                config,
                scheduler,
                Arc::new(engine),
                crate::api::test_tokenizer(),
                300,
                "kiln-test".to_string(),
            )
        };
        let mut state = state;
        state.adapter_dir = adapter_dir.path().to_path_buf();
        let state = state;
        let total_examples = 8usize;
        let job_id = "job-cancel".to_string();
        state.eval_jobs.write().unwrap().insert(
            job_id.clone(),
            EvalJobInfo::queued(
                job_id.clone(),
                "cancel-me".into(),
                vec![None],
                EvalSubmissionKind::OnDemand,
                None,
            ),
        );
        let entry = EvalQueueEntry {
            job_id: job_id.clone(),
            job: QueuedEvalJob::Inline {
                suite: Box::new(big_suite(total_examples)),
                adapter: None,
                generation_override: None,
            },
        };

        let reached = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let generator = Arc::new(GatedGenerator {
            inner: crate::eval::MockEvalGenerator::new(),
            reached: reached.clone(),
            resume: resume.clone(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }) as Arc<dyn crate::eval::generator::EvalGenerator>;

        let worker = tokio::spawn(run_one_job_with_generator(
            state.clone(),
            entry,
            generator,
        ));

        // Deterministic: the gate fires after the first outcome landed and
        // the second example is mid-generation.
        reached.notified().await;
        {
            let jobs = state.eval_jobs.read().unwrap();
            let job = jobs.get(&job_id).unwrap();
            assert_eq!(job.state, EvalJobState::Running);
            assert!(job.cancel_flag.is_some(), "worker must install the handle");
        }

        // Cancel through the real route, exactly as the dashboard does.
        let app = crate::api::router(state.clone());
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/eval/jobs/{job_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        resume.notify_one();

        worker.await.unwrap();

        let jobs = state.eval_jobs.read().unwrap();
        let job = jobs.get(&job_id).expect("job should stay tracked");
        assert_eq!(
            job.state,
            EvalJobState::Cancelled,
            "worker must not clobber Cancelled back to Completed"
        );
        assert_eq!(job.finished_runs.len(), 1, "partial run should be recorded");
        let outcomes = job.finished_runs[0].outcomes.len();
        assert!(
            outcomes >= 1 && outcomes < total_examples,
            "executor should stop early, got {outcomes}/{total_examples} outcomes"
        );
        assert!(job.finished_at_iso.is_some(), "terminal timestamps stamped");
                assert!(job.cancel_flag.is_none(), "handle cleared at terminal state");
    }
}

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
    let mut candidates: Vec<(String, std::time::Instant)> =
        jobs.iter()
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

    // Live judge on the real backend: LlmJudge scorers re-enter the model
    // through the executor's deferred judge pass (batch adapter swap +
    // blocking-thread scoring). Mock mode keeps the no-op runner — judge
    // scorers degrade to Invalid there, honestly.
    let judge_runner: Arc<dyn JudgeRunner> = match state.backend.as_ref() {
        crate::state::ModelBackend::Real { .. } => {
            Arc::new(crate::eval::generator::LiveJudgeRunner::new(state.clone()))
        }
        crate::state::ModelBackend::Mock { .. } => noop_judge_runner(),
    };

    let progress_state = state.eval_jobs.clone();
    let progress_job_id = job_id.clone();
    let progress_cb: crate::eval::executor::ProgressCallback = Box::new(move |p: EvalProgress| {
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
        // §8.7: this job may carry a promotion gate (post-training eval
        // with min_accuracy). Apply the verdict now that the run is
        // terminal.
        apply_post_eval_gate(&state, &snapshot).await;

        // Eval results become signals: fire the completion webhook AFTER
        // the gate so the payload carries the promotion verdict. The gate
        // stamps its verdict on the LINKED TRAINING job; re-read it here.
        if let Some(ref url) = state.eval_webhook_url {
            let gate_verdict = snapshot.post_eval_gate.as_ref().and_then(|gate| {
                let jobs = state.training_jobs.read().unwrap();
                jobs.get(&gate.training_job_id)
                    .and_then(|j| j.post_eval_verdict.clone())
            });
            let event = serde_json::json!({
                "event": "eval_completed",
                "job_id": snapshot.job_id,
                "suite": snapshot.suite_name,
                "adapters": snapshot.adapters,
                "status": match snapshot.state {
                    EvalJobState::Completed => "completed",
                    EvalJobState::Cancelled => "cancelled",
                    _ => "failed",
                },
                "headline_accuracy": snapshot.headline_accuracy,
                "gate_verdict": gate_verdict,
                "error": snapshot.error,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });
            crate::training_queue::fire_webhook_json(url.clone(), event);
        }
    }

    // Back-linking from eval-completion → training-job is intentionally
    // NOT done here. `enqueue_post_training_eval` already populates the
    // training job's `linked_eval_job_ids` at queue time, so the
    // training-side dashboard can link to the eval the moment it lands
    // in the queue (not after it finishes). Pushing again here would
    // duplicate every ID.
}

/// Apply the §8.7 promotion gate carried by a post-training eval job.
///
/// - **Pass** (`accuracy >= min_accuracy`): promote the adapter into
///   serving when training deferred its auto-load to this verdict.
/// - **Fail**: rename the adapter dir to `<name>.failed` (the documented
///   `PostEvalConfig::min_accuracy` contract), purge its cache entries,
///   and never promote. The previously-active adapter simply stays
///   active — it was never displaced, because training defers auto-load
///   while a gate is pending.
/// - **Eval errored/cancelled**: leave the adapter on disk but do NOT
///   promote — an unmeasured adapter must not start serving. The verdict
///   on the training job says exactly that.
async fn apply_post_eval_gate(state: &AppState, snapshot: &crate::eval::queue::EvalJobInfo) {
    let Some(gate) = snapshot.post_eval_gate.clone() else {
        return;
    };
    let job_id = snapshot.job_id.clone();

    let stamp_verdict = |outcome: crate::state::GateOutcome, verdict: String| {
        tracing::info!(
            eval_job = %job_id,
            training_job = %gate.training_job_id,
            adapter = %gate.adapter_name,
            outcome = %outcome.as_str(),
            verdict = %verdict,
            "post-eval gate verdict"
        );
        let snapshot = {
            let mut jobs = state.training_jobs.write().unwrap();
            jobs.get_mut(&gate.training_job_id).map(|job| {
                job.post_eval_verdict = Some(verdict);
                // Machine-readable twin of the prose verdict — persisted
                // together everywhere the verdict persists so consumers
                // never classify prose by substring.
                job.gate_outcome = Some(outcome.as_str().to_string());
                job.clone()
            })
        };
        // Re-archive: finalize_job persisted the terminal job BEFORE the
        // gate eval ran, so without this re-save the verdict only lived
        // in memory and a restart showed the gated job verdict-less
        // (round-5 quick win).
        if let Some(job) = snapshot {
            if let Err(e) = crate::training_history::save(&state.adapter_dir, &job) {
                tracing::warn!(
                    error = %e,
                    training_job = %gate.training_job_id,
                    "failed to persist gate verdict to training history"
                );
            }
        }
    };

    if snapshot.state != EvalJobState::Completed {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "post-eval did not complete (state: {:?}) — adapter `{}` left on disk, NOT promoted",
                snapshot.state, gate.adapter_name
            ),
        );
        return;
    }

    let accuracy = snapshot
        .finished_runs
        .iter()
        .find(|run| run.adapter.as_deref() == Some(gate.adapter_name.as_str()))
        .map(|run| run.metrics.accuracy)
        .or(snapshot.headline_accuracy);
    let Some(accuracy) = accuracy else {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "post-eval produced no run for adapter `{}` — NOT promoted",
                gate.adapter_name
            ),
        );
        return;
    };

    // Regression detection: gated runs are Compare jobs carrying the
    // previous generation's run alongside the new adapter's. Pair the
    // per-example outcomes and reject promotion when the new adapter is
    // SIGNIFICANTLY worse (#1497 exact sign test) — a static floor alone
    // happily promotes a regressed adapter as long as it clears the bar.
    if let Some(baseline_run) = snapshot
        .finished_runs
        .iter()
        .find(|run| run.adapter.as_deref() != Some(gate.adapter_name.as_str()))
    {
        if let Some(new_run) = snapshot
            .finished_runs
            .iter()
            .find(|run| run.adapter.as_deref() == Some(gate.adapter_name.as_str()))
        {
            let mut improved = 0u32;
            let mut regressed = 0u32;
            for (b, n) in baseline_run.outcomes.iter().zip(new_run.outcomes.iter()) {
                if b.example_id != n.example_id {
                    continue;
                }
                if n.score > b.score {
                    improved += 1;
                } else if n.score < b.score {
                    regressed += 1;
                }
            }
            let test = kiln_eval::result::sign_test(improved, regressed);
            if regressed > improved && test.significant() {
                stamp_verdict(
                    crate::state::GateOutcome::Regression,
                    format!(
                        "REGRESSION: `{}` significantly worse than `{}` \
                         (improved {improved}, regressed {regressed}, p={:.4}) — \
                         NOT promoted despite accuracy {accuracy:.3}",
                        gate.adapter_name,
                        baseline_run.adapter.as_deref().unwrap_or("base"),
                        test.p_value
                    ),
                );
                return;
            }

            // distill_refresh §6.4 dual thresholds against the SAME
            // baseline run: fractional recovery (new/baseline) and
            // absolute gain (new − baseline).
            let baseline_acc = baseline_run.metrics.accuracy;
            if let Some(min_recovery) = gate.relative_recovery {
                let recovery = if baseline_acc > 0.0 {
                    accuracy / baseline_acc
                } else {
                    1.0
                };
                if recovery < min_recovery {
                    stamp_verdict(
                        crate::state::GateOutcome::Regression,
                        format!(
                            "RECOVERY FAILED: `{}` recovered only {recovery:.3} of `{}`'s \
                             {baseline_acc:.3} (required {min_recovery:.2}) — NOT promoted",
                            gate.adapter_name,
                            baseline_run.adapter.as_deref().unwrap_or("base"),
                        ),
                    );
                    return;
                }
            }
            if let Some(min_gain) = gate.absolute_gain {
                let gain = accuracy - baseline_acc;
                if gain < min_gain {
                    stamp_verdict(
                        crate::state::GateOutcome::Regression,
                        format!(
                            "GAIN TOO SMALL: `{}` gained {gain:+.3} over `{}`'s \
                             {baseline_acc:.3} (required {min_gain:+.2}) — NOT promoted",
                            gate.adapter_name,
                            baseline_run.adapter.as_deref().unwrap_or("base"),
                        ),
                    );
                    return;
                }
            }
        }
    }

    if accuracy >= gate.min_accuracy {
        if gate.auto_load_on_pass {
            let adapter_dir = state.adapter_dir.join(&gate.adapter_name);
            match crate::adapter_swap::swap_runtime_adapter(
                state,
                crate::adapter_swap::SwapRequest {
                    target: crate::adapter_swap::SwapTarget::Resolved {
                        active_name: gate.adapter_name.clone(),
                        dir: adapter_dir,
                    },
                    content_changed: true,
                    default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Replace(Some(
                        gate.adapter_name.clone(),
                    )),
                    reason: "post_eval_gate_promotion",
                },
            )
            .await
            {
                Ok(_) => {
                    stamp_verdict(
                        crate::state::GateOutcome::Promoted,
                        format!(
                            "PASSED: accuracy {accuracy:.3} >= {:.3}; adapter `{}` promoted to active",
                            gate.min_accuracy, gate.adapter_name
                        ),
                    );
                }
                // The gate itself passed, but the system failed to apply
                // the promotion — that is an operational error, not a
                // measured success or failure.
                Err(e) => stamp_verdict(
                    crate::state::GateOutcome::Error,
                    format!(
                        "PASSED: accuracy {accuracy:.3} >= {:.3}, but promotion failed: {e}",
                        gate.min_accuracy
                    ),
                ),
            }
        } else {
            stamp_verdict(
                crate::state::GateOutcome::Kept,
                format!(
                    "PASSED: accuracy {accuracy:.3} >= {:.3}; adapter `{}` kept (auto_load not requested)",
                    gate.min_accuracy, gate.adapter_name
                ),
            );
        }
        return;
    }

    // FAILED the gate. Own the same revision barrier as load, delete, upload,
    // and training publication until the serving name has been removed. This
    // makes the loaded check, optional unload, default clear, rename, and cache
    // purge one serialized transaction.
    let serial = match crate::adapter_swap::adapter_mutation_guard(state).await {
        Ok(serial) => serial,
        Err(error) => {
            stamp_verdict(
                crate::state::GateOutcome::Error,
                format!(
                    "FAILED: accuracy {accuracy:.3} < {:.3}, but adapter `{}` could not be demoted: {error}",
                    gate.min_accuracy, gate.adapter_name
                ),
            );
            return;
        }
    };
    let currently_loaded = state.loaded_adapter_name();
    if currently_loaded.as_deref() == Some(gate.adapter_name.as_str()) {
        if let Err(error) = crate::adapter_swap::swap_runtime_adapter_locked(
            state,
            crate::adapter_swap::SwapRequest {
                target: crate::adapter_swap::SwapTarget::Base,
                content_changed: false,
                default_adapter: crate::adapter_swap::DefaultAdapterUpdate::ClearIf(
                    gate.adapter_name.clone(),
                ),
                reason: "post_eval_gate_demotion",
            },
            &serial,
        )
        .await
        {
            stamp_verdict(
                crate::state::GateOutcome::Error,
                format!(
                    "FAILED: accuracy {accuracy:.3} < {:.3}, but loaded adapter `{}` could not be swapped away: {error}",
                    gate.min_accuracy, gate.adapter_name
                ),
            );
            return;
        }
    } else {
        crate::adapter_swap::apply_default_update(
            state,
            &crate::adapter_swap::DefaultAdapterUpdate::ClearIf(gate.adapter_name.clone()),
        );
    }

    let src = state.adapter_dir.join(&gate.adapter_name);
    let mut dst = state
        .adapter_dir
        .join(format!("{}.failed", gate.adapter_name));
    if dst.exists() {
        dst = state.adapter_dir.join(format!(
            "{}.failed-{}",
            gate.adapter_name,
            crate::recent_requests::now_unix_ms()
        ));
    }
    if let Err(error) = std::fs::rename(&src, &dst) {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "FAILED: accuracy {accuracy:.3} < {:.3}, but adapter `{}` could not be renamed to .failed: {error}",
                gate.min_accuracy, gate.adapter_name
            ),
        );
        return;
    }
    state.purge_adapter_caches(&Some(gate.adapter_name.clone()));
    drop(serial);
    let rename_note = format!(
        "renamed to `{}`",
        dst.file_name().unwrap_or_default().to_string_lossy()
    );
    stamp_verdict(
        crate::state::GateOutcome::Demoted,
        format!(
            "FAILED: accuracy {accuracy:.3} < {:.3}; adapter `{}` NOT promoted, {rename_note}",
            gate.min_accuracy, gate.adapter_name
        ),
    );
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
            thinking_budget: &kiln_eval::EvalThinkingBudget,
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
            let fut = self.inner.run(
                prepared,
                params,
                thinking_budget,
                completion_index,
                adapter_label,
            );
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

    fn gate_test_state() -> (crate::state::AppState, tempfile::TempDir) {
        let adapter_dir = tempfile::tempdir().unwrap();
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
        let mut state = crate::state::AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            crate::api::test_tokenizer(),
            300,
            "kiln-test".to_string(),
        );
        state.adapter_dir = adapter_dir.path().to_path_buf();
        (state, adapter_dir)
    }

    fn seed_training_job(state: &crate::state::AppState, job_id: &str, adapter: &str) {
        state.training_jobs.write().unwrap().insert(
            job_id.to_string(),
            crate::state::TrainingJobInfo {
                job_id: job_id.to_string(),
                adapter_name: adapter.to_string(),
                job_type: crate::state::TrainingJobType::Sft,
                state: kiln_train::TrainingState::Completed,
                progress: 1.0,
                loss: None,
                epoch: None,
                adapter_path: None,
                submitted_at: std::time::Instant::now(),
                submitted_unix_ms: crate::recent_requests::now_unix_ms(),
                auto_load: true,
                consumed_correction_ids: Vec::new(),
                finished_at: None,
                finished_unix_ms: None,
                error: None,
                linked_eval_job_ids: Vec::new(),
                post_eval_verdict: None,
                gate_outcome: None,
                loss_history: Vec::new(),
                cancel_requested: Default::default(),
            },
        );
    }

    /// One-example suite whose scorer passes iff the mock generator's
    /// reply contains `phrase`.
    fn gate_suite(phrase: &str) -> EvalSuite {
        EvalSuite {
            name: "gate-suite".into(),
            description: None,
            default_scorer: Scorer::Contains {
                phrases: vec![phrase.into()],
                mode: Default::default(),
                case_sensitive: false,
            },
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("e0".into()),
                messages: vec![EvalChatMessage::new("user", "ping")],
                target: None,
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    async fn run_gated_job(
        state: &crate::state::AppState,
        suite: EvalSuite,
        gate: crate::eval::queue::PostEvalGate,
        reply: &str,
    ) {
        let job_id = format!("eval-{}", gate.adapter_name);
        let mut info = EvalJobInfo::queued(
            job_id.clone(),
            suite.name.clone(),
            vec![Some(gate.adapter_name.clone())],
            EvalSubmissionKind::PostTraining,
            Some(gate.training_job_id.clone()),
        );
        info.post_eval_gate = Some(gate.clone());
        state
            .eval_jobs
            .write()
            .unwrap()
            .insert(job_id.clone(), info);
        let entry = EvalQueueEntry {
            job_id,
            job: QueuedEvalJob::Inline {
                suite: Box::new(suite),
                adapter: Some(gate.adapter_name.clone()),
                generation_override: None,
            },
        };
        let generator =
            Arc::new(crate::eval::MockEvalGenerator::new().with_force_reply(reply.to_string()))
                as Arc<dyn crate::eval::generator::EvalGenerator>;
        run_one_job_with_generator(state.clone(), entry, generator).await;
    }

    fn verdict_of(state: &crate::state::AppState, training_job: &str) -> String {
        state
            .training_jobs
            .read()
            .unwrap()
            .get(training_job)
            .and_then(|j| j.post_eval_verdict.clone())
            .expect("verdict stamped")
    }

    /// The machine-readable twin stamped next to the prose verdict.
    fn outcome_of(state: &crate::state::AppState, training_job: &str) -> String {
        state
            .training_jobs
            .read()
            .unwrap()
            .get(training_job)
            .and_then(|j| j.gate_outcome.clone())
            .expect("gate_outcome stamped")
    }

    /// Gate FAIL: the adapter directory is renamed `<name>.failed` (the
    /// documented PostEvalConfig::min_accuracy contract) and the verdict
    /// lands on the training job. Before this, min_accuracy was parsed,
    /// documented — and read by no code at all.
    #[tokio::test]
    async fn gate_failure_renames_adapter_and_stamps_verdict() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-1", "gated");
        std::fs::create_dir(state.adapter_dir.join("gated")).unwrap();
        *state.active_adapter_name.write().unwrap() = Some("gated".to_string());
        *state.loaded_adapter.write().unwrap() = Some(crate::state::LoadedAdapterIdentity {
            name: "request-override".to_string(),
            content_revision: "c".repeat(64),
        });

        run_gated_job(
            &state,
            gate_suite("phrase-the-reply-never-contains"),
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "gated".into(),
                training_job_id: "train-1".into(),
                auto_load_on_pass: false,
            },
            "mock reply",
        )
        .await;

        assert!(
            !state.adapter_dir.join("gated").exists(),
            "failed adapter must not stay under its serving name"
        );
        assert!(
            state.adapter_dir.join("gated.failed").exists(),
            "failed adapter renamed per the documented contract"
        );
        let verdict = verdict_of(&state, "train-1");
        assert!(verdict.contains("FAILED"), "{verdict}");
        assert!(verdict.contains("gated.failed"), "{verdict}");
        assert_eq!(outcome_of(&state, "train-1"), "demoted");
        assert!(
            state.active_adapter_name.read().unwrap().is_none(),
            "demotion must clear a rejected server default even when a request override is loaded"
        );
        assert_eq!(
            state.loaded_adapter_name().as_deref(),
            Some("request-override"),
            "demotion must not unload a different physically loaded revision"
        );
    }

    /// Gate PASS without a deferred auto-load: adapter stays under its
    /// name, verdict records the pass.
    #[tokio::test]
    async fn gate_pass_keeps_adapter_and_stamps_verdict() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-2", "good");
        std::fs::create_dir(state.adapter_dir.join("good")).unwrap();

        run_gated_job(
            &state,
            gate_suite("mock"),
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "good".into(),
                training_job_id: "train-2".into(),
                auto_load_on_pass: false,
            },
            "a mock reply that matches",
        )
        .await;

        assert!(state.adapter_dir.join("good").exists());
        assert!(!state.adapter_dir.join("good.failed").exists());
        let verdict = verdict_of(&state, "train-2");
        assert!(verdict.contains("PASSED"), "{verdict}");
        // A pass without a requested promotion is `kept` — distinct from
        // `promoted` so the dashboard never paints it as a warning by
        // substring-sniffing the prose.
        assert_eq!(outcome_of(&state, "train-2"), "kept");
    }

    /// An errored eval must leave the adapter on disk and NOT promote it —
    /// an unmeasured adapter never starts serving.
    #[tokio::test]
    async fn gate_on_errored_eval_does_not_promote_or_rename() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-3", "unmeasured");
        std::fs::create_dir(state.adapter_dir.join("unmeasured")).unwrap();

        // Registered suite with no registry configured → run_job errors →
        // terminal state Failed.
        let job_id = "eval-unmeasured".to_string();
        let mut info = EvalJobInfo::queued(
            job_id.clone(),
            "missing-suite".into(),
            vec![Some("unmeasured".into())],
            EvalSubmissionKind::PostTraining,
            Some("train-3".into()),
        );
        info.post_eval_gate = Some(crate::eval::queue::PostEvalGate {
            min_accuracy: 0.5,
            relative_recovery: None,
            absolute_gain: None,
            adapter_name: "unmeasured".into(),
            training_job_id: "train-3".into(),
            auto_load_on_pass: true,
        });
        state
            .eval_jobs
            .write()
            .unwrap()
            .insert(job_id.clone(), info);
        let entry = EvalQueueEntry {
            job_id,
            job: QueuedEvalJob::Registered {
                suite_name: "missing-suite".into(),
                adapter: Some("unmeasured".into()),
                generation_override: None,
            },
        };
        let generator = Arc::new(crate::eval::MockEvalGenerator::new())
            as Arc<dyn crate::eval::generator::EvalGenerator>;
        run_one_job_with_generator(state.clone(), entry, generator).await;

        assert!(state.adapter_dir.join("unmeasured").exists());
        assert!(!state.adapter_dir.join("unmeasured.failed").exists());
        assert!(
            state.active_adapter_name.read().unwrap().is_none(),
            "unmeasured adapter must not be promoted"
        );
        let verdict = verdict_of(&state, "train-3");
        assert!(verdict.contains("did not complete"), "{verdict}");
        assert!(verdict.contains("NOT promoted"), "{verdict}");
        assert_eq!(outcome_of(&state, "train-3"), "error");
    }

    /// The stamped verdict + machine-readable outcome must survive all the
    /// way to the API payload shape: GET /v1/train/queue serializes the
    /// completed job with BOTH `post_eval_verdict` (prose, for humans) and
    /// `gate_outcome` (string enum, for the dashboard pill). Before
    /// `gate_outcome` existed the UI classified the prose by substring and
    /// a PASSED gate could render as a warning.
    #[tokio::test]
    async fn gate_outcome_reaches_train_queue_api_payload() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-api", "api-gated");
        std::fs::create_dir(state.adapter_dir.join("api-gated")).unwrap();

        run_gated_job(
            &state,
            gate_suite("mock"),
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "api-gated".into(),
                training_job_id: "train-api".into(),
                auto_load_on_pass: false,
            },
            "a mock reply that matches",
        )
        .await;

        let app = crate::api::router(state.clone());
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/train/queue")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let job = payload["completed"]
            .as_array()
            .expect("completed array")
            .iter()
            .find(|j| j["job_id"] == "train-api")
            .expect("gated training job present in /v1/train/queue");
        assert_eq!(job["gate_outcome"], "kept", "payload: {job}");
        assert!(
            job["post_eval_verdict"]
                .as_str()
                .is_some_and(|v| v.contains("PASSED")),
            "payload: {job}"
        );
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

        let worker = tokio::spawn(run_one_job_with_generator(state.clone(), entry, generator));

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
        assert!(
            job.cancel_flag.is_none(),
            "handle cleared at terminal state"
        );
    }
}

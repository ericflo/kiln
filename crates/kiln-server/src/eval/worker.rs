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
/// `state.eval_jobs` whose `finished_at` is older than `tracked_job_ttl`.
/// Mirrors `training_queue::gc_tracked_jobs`. Without this, terminal
/// entries pile up until `max_tracked_eval_jobs` rejects new submissions.
pub fn gc_eval_jobs(state: &AppState) -> usize {
    let ttl = state.tracked_job_ttl;
    let now = std::time::Instant::now();
    let mut jobs = state.eval_jobs.write().unwrap();
    let before = jobs.len();
    jobs.retain(|_id, job| match job.state {
        EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled => {
            match job.finished_at {
                None => true,
                Some(t) => now.saturating_duration_since(t) < ttl,
            }
        }
        EvalJobState::Queued | EvalJobState::Running => true,
    });
    let removed = before - jobs.len();
    if removed > 0 {
        tracing::debug!(removed, remaining = jobs.len(), "GC'd terminal eval jobs past TTL");
    }
    removed
}

async fn run_one_job(state: AppState, entry: EvalQueueEntry) {
    let job_id = entry.job_id.clone();
    // Mark as running and stamp started_at.
    {
        let mut jobs = state.eval_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            if matches!(job.state, EvalJobState::Cancelled | EvalJobState::Failed) {
                tracing::info!(job_id = %job_id, "skipping cancelled eval job");
                return;
            }
            job.state = EvalJobState::Running;
            job.started_at_iso = Some(chrono::Utc::now().to_rfc3339());
        } else {
            tracing::warn!(job_id = %job_id, "eval job not found in tracking map");
            return;
        }
    }

    let generator = generator_from_state(state.clone());
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

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Wire a tracking-side cancellation flag: the cancel endpoint sets it on
    // the job-info before re-checking.
    // (We re-derive `cancel_flag` from job state below — for now, jobs
    // marked Cancelled by the API are skipped on the next iteration.)

    let result = run_job(
        &state,
        &entry.job,
        generator,
        judge_runner,
        Some(progress_cb),
        cancel_flag,
    )
    .await;

    let now_iso = chrono::Utc::now().to_rfc3339();
    let now_instant = std::time::Instant::now();

    match result {
        Ok(runs) => {
            let headline = runs.iter().last().map(|r| r.metrics.accuracy);
            let mut jobs = state.eval_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = EvalJobState::Completed;
                job.finished_runs = runs;
                job.headline_accuracy = headline;
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
            }
        }
        Err(err) => {
            tracing::error!(job_id = %job_id, error = %err, "eval job failed");
            let mut jobs = state.eval_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = EvalJobState::Failed;
                job.error = Some(err);
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
            }
        }
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


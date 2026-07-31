//! Shared cooperative cancellation for authoritative training and eval jobs.
//!
//! Workflow-level orchestrators must stop the executor that owns the work,
//! rather than merely changing their own status record. The HTTP handlers and
//! server-owned OpenEnv runs therefore use these same primitives.

use kiln_eval::EvalJobState;
use kiln_train::TrainingState;

use crate::error::ApiError;
use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::state::{AppState, TrainingJobType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainingCancellation {
    Cancelling,
    Cancelled,
}

pub(crate) fn request_training_job_cancellation(
    state: &AppState,
    job_id: &str,
) -> Result<TrainingCancellation, ApiError> {
    {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(job_id)
            .ok_or_else(|| ApiError::training_job_not_found(job_id))?;
        if job.state == TrainingState::Running {
            job.cancel_requested
                .store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(job_id, "cancellation requested for running training job");
            return Ok(TrainingCancellation::Cancelling);
        }
        if job.state != TrainingState::Queued {
            return Err(ApiError::training_job_not_cancellable(
                job_id,
                format!("{:?}", job.state),
            ));
        }
    }

    let removed = state.training_queue.lock().unwrap().remove(job_id);
    if !removed {
        return Err(ApiError::training_job_already_started(job_id));
    }

    let metric_type = {
        let mut jobs = state.training_jobs.write().unwrap();
        let job_type = jobs.get(job_id).map(|job| job.job_type);
        if let Some(job) = jobs.get_mut(job_id) {
            job.state = TrainingState::Failed;
            job.error = Some("cancelled while queued".to_string());
            job.finished_at = Some(std::time::Instant::now());
            job.finished_unix_ms = Some(crate::recent_requests::now_unix_ms());
        }
        job_type
    };
    if let Some(job_type) = metric_type {
        let metric_type = match job_type {
            TrainingJobType::Sft => TrainingMetricType::Sft,
            TrainingJobType::Grpo => TrainingMetricType::Grpo,
            TrainingJobType::Opd => TrainingMetricType::Opd,
        };
        state
            .metrics
            .inc_training(metric_type, TrainingMetricStatus::Cancelled);
    }
    Ok(TrainingCancellation::Cancelled)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalCancellation {
    Cancelling,
    Cancelled { was_in_queue: bool },
}

/// Request cancellation without applying the eval DELETE endpoint's distinct
/// terminal-record deletion semantics.
pub(crate) fn request_eval_job_cancellation(
    state: &AppState,
    job_id: &str,
) -> Result<Option<EvalCancellation>, ApiError> {
    let removed = state.eval_queue.lock().unwrap().remove(job_id);
    let current_state = state
        .eval_jobs
        .read()
        .unwrap()
        .get(job_id)
        .map(|job| job.state);
    match current_state {
        None => Err(ApiError::eval_job_not_found(job_id)),
        Some(EvalJobState::Queued) => {
            if let Some(job) = state.eval_jobs.write().unwrap().get_mut(job_id) {
                job.state = EvalJobState::Cancelled;
                job.finished_at_iso = Some(chrono::Utc::now().to_rfc3339());
                job.finished_at = Some(std::time::Instant::now());
            }
            Ok(Some(EvalCancellation::Cancelled {
                was_in_queue: removed,
            }))
        }
        Some(EvalJobState::Running) => {
            if let Some(job) = state.eval_jobs.write().unwrap().get_mut(job_id) {
                job.state = EvalJobState::Cancelled;
                if let Some(flag) = job.cancel_flag.as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            Ok(Some(EvalCancellation::Cancelling))
        }
        Some(EvalJobState::Cancelled | EvalJobState::Completed | EvalJobState::Failed) => Ok(None),
    }
}

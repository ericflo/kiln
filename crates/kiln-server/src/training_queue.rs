//! FIFO training job queue — accepts SFT and GRPO jobs, runs them sequentially.
//!
//! The queue ensures only one training job runs at a time, preventing GPU memory
//! conflicts between concurrent training jobs. Jobs are executed in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use kiln_model::lora_loader::LoraWeights;
use kiln_train::{GrpoRequest, SftRequest, TrainingState};
use kiln_train::trainer;

use crate::state::{AppState, ModelBackend};

/// A pending training job in the queue.
pub enum QueuedJob {
    Sft(SftRequest),
    Grpo(GrpoRequest),
}

/// Entry in the training queue.
pub struct QueueEntry {
    pub job_id: String,
    pub job: QueuedJob,
}

/// Thread-safe training queue.
pub struct TrainingQueue {
    pub(crate) queue: VecDeque<QueueEntry>,
}

impl TrainingQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Add a job to the back of the queue.
    pub fn push(&mut self, entry: QueueEntry) {
        self.queue.push_back(entry);
    }

    /// Take the next job from the front of the queue.
    pub fn pop(&mut self) -> Option<QueueEntry> {
        self.queue.pop_front()
    }

    /// Number of jobs waiting in the queue (not including the currently running job).
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Remove a queued job by ID. Returns true if found and removed.
    pub fn remove(&mut self, job_id: &str) -> bool {
        let before = self.queue.len();
        self.queue.retain(|e| e.job_id != job_id);
        self.queue.len() < before
    }
}

pub type SharedTrainingQueue = Arc<std::sync::Mutex<TrainingQueue>>;

/// Create a new shared training queue.
pub fn new_shared_queue() -> SharedTrainingQueue {
    Arc::new(std::sync::Mutex::new(TrainingQueue::new()))
}

/// Spawn the background training worker that pulls jobs from the queue.
///
/// This runs as a tokio task that polls the queue every 500ms. When a job is
/// found, it executes it on a blocking thread (training is CPU/GPU-bound).
pub fn spawn_training_worker(state: AppState) {
    tokio::spawn(async move {
        loop {
            // Check for next job
            let entry = {
                let mut q = state.training_queue.lock().unwrap();
                q.pop()
            };

            if let Some(entry) = entry {
                // Execute the job on a blocking thread
                let state_clone = state.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    execute_job(state_clone, entry);
                });
                // Wait for completion before pulling the next job
                if let Err(e) = handle.await {
                    tracing::error!("training worker task panicked: {e}");
                }
            } else {
                // No jobs — sleep briefly before checking again
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    });
}

/// Execute a single training job (runs on a blocking thread).
fn execute_job(state: AppState, entry: QueueEntry) {
    let job_id = entry.job_id.clone();

    // Mark as running
    {
        let mut jobs = state.training_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            // Check if it was cancelled while queued
            if job.state == TrainingState::Failed {
                tracing::info!(job_id = %job_id, "skipping cancelled job");
                return;
            }
            job.state = TrainingState::Running;
        } else {
            tracing::warn!(job_id = %job_id, "job not found in tracking map, skipping");
            return;
        }
    }

    // Extract model weights reference
    let runner_arc = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner.clone(),
        ModelBackend::Mock { .. } => {
            let mut jobs = state.training_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = TrainingState::Failed;
            }
            tracing::error!(job_id = %job_id, "training requires real model weights");
            return;
        }
    };

    let (weights_ref, num_layers) = {
        let guard = runner_arc.read().unwrap();
        (runner_arc.clone(), guard.config.num_layers)
    };

    // Get auto_load and adapter_name from the job info
    let (auto_load, adapter_name) = {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs.get(&job_id).unwrap();
        (job.auto_load, job.adapter_name.clone())
    };

    // Set up progress callback
    let training_jobs_cb = state.training_jobs.clone();
    let job_id_cb = job_id.clone();
    let progress_cb = Box::new(move |progress: trainer::TrainingProgress| {
        let mut jobs = training_jobs_cb.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id_cb) {
            job.progress = progress.progress;
            job.loss = Some(progress.loss);
            job.epoch = Some(progress.epoch as u32);
        }
    });

    // Run the actual training under GPU write lock
    let result: Result<PathBuf, String> = match entry.job {
        QueuedJob::Sft(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            trainer::sft_train(
                &req.examples,
                &req.config,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"))
        }
        QueuedJob::Grpo(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            trainer::grpo_train(
                &req.groups,
                &req.config,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"))
        }
    };

    match result {
        Ok(adapter_path) => {
            let path_str = adapter_path.display().to_string();
            tracing::info!(job_id = %job_id, adapter = %adapter_name, path = %path_str, "training completed");

            {
                let mut jobs = state.training_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.state = TrainingState::Completed;
                    job.progress = 1.0;
                    job.adapter_path = Some(path_str);
                }
            }

            if auto_load {
                if let Err(e) = auto_load_adapter(
                    &weights_ref,
                    &state.active_adapter_name,
                    &adapter_path,
                    &adapter_name,
                    num_layers,
                ) {
                    tracing::error!(job_id = %job_id, "auto-load failed: {e}");
                } else {
                    tracing::info!(job_id = %job_id, "auto-loaded trained adapter");
                }
            }
        }
        Err(e) => {
            tracing::error!(job_id = %job_id, "training failed: {e}");
            let mut jobs = state.training_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = TrainingState::Failed;
            }
        }
    }
}

/// Load a LoRA adapter using the two-phase RwLock pattern.
fn auto_load_adapter(
    runner: &Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    active_adapter_name: &Arc<std::sync::RwLock<Option<String>>>,
    adapter_path: &std::path::Path,
    adapter_name: &str,
    num_layers: usize,
) -> Result<(), String> {
    let device = {
        let guard = runner.read().unwrap();
        guard.weights.embed_tokens.device().clone()
    };

    let lora = LoraWeights::load(adapter_path, num_layers, &device)
        .map_err(|e| format!("failed to load adapter: {e}"))?;

    {
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
    }
    *active_adapter_name.write().unwrap() = Some(adapter_name.to_string());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_fifo_order() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });

        assert_eq!(q.len(), 3);
        assert_eq!(q.pop().unwrap().job_id, "job-1");
        assert_eq!(q.pop().unwrap().job_id, "job-2");
        assert_eq!(q.pop().unwrap().job_id, "job-3");
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_queue_remove() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });

        // Remove middle job
        assert!(q.remove("job-2"));
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().unwrap().job_id, "job-1");
        assert_eq!(q.pop().unwrap().job_id, "job-3");

        // Remove non-existent
        assert!(!q.remove("job-99"));
    }

    #[test]
    fn test_queue_empty() {
        let mut q = TrainingQueue::new();
        assert_eq!(q.len(), 0);
        assert!(q.pop().is_none());
        assert!(!q.remove("nonexistent"));
    }
}

//! Client for the Python training sidecar process.
//!
//! Communicates over a unix domain socket using a JSON-line protocol
//! (one JSON object per line, newline-delimited).

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Messages sent from Rust to the Python sidecar.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SidecarRequest {
    SftRequest {
        job_id: String,
        examples: Vec<serde_json::Value>,
        adapter_name: String,
        epochs: usize,
        learning_rate: f64,
        lora_rank: usize,
        lora_alpha: f32,
    },
    GrpoRequest {
        job_id: String,
        groups: Vec<serde_json::Value>,
        adapter_name: String,
        learning_rate: f64,
        lora_rank: usize,
        lora_alpha: f32,
    },
    StatusQuery {
        job_id: String,
    },
    Shutdown,
}

/// Messages received from the Python sidecar.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SidecarResponse {
    JobAccepted {
        job_id: String,
        #[serde(default)]
        message: Option<String>,
    },
    JobStatus {
        job_id: String,
        state: String,
        progress: f32,
        #[serde(default)]
        epoch: Option<u32>,
        #[serde(default)]
        loss: Option<f64>,
        #[serde(default)]
        adapter_path: Option<String>,
        #[serde(default)]
        message: Option<String>,
    },
    JobComplete {
        job_id: String,
        adapter_path: String,
    },
    Error {
        job_id: String,
        message: String,
    },
}

/// Client for communicating with the Python training sidecar.
#[derive(Debug, Clone)]
pub struct SidecarClient {
    socket_path: PathBuf,
}

impl SidecarClient {
    pub fn new(socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
        }
    }

    /// Check if the sidecar socket exists (sidecar is likely running).
    pub fn is_available(&self) -> bool {
        self.socket_path.exists()
    }

    /// Send a request to the sidecar and read the response.
    async fn send(&self, request: &SidecarRequest) -> Result<SidecarResponse, SidecarError> {
        let stream = UnixStream::connect(&self.socket_path).await.map_err(|e| {
            SidecarError::Connection(format!(
                "failed to connect to sidecar at {}: {}",
                self.socket_path.display(),
                e
            ))
        })?;

        let (read_half, mut write_half) = stream.into_split();

        // Send the JSON-line request
        let mut msg = serde_json::to_vec(request).map_err(|e| {
            SidecarError::Protocol(format!("failed to serialize request: {e}"))
        })?;
        msg.push(b'\n');
        write_half.write_all(&msg).await.map_err(|e| {
            SidecarError::Connection(format!("failed to write to sidecar: {e}"))
        })?;
        write_half.flush().await.map_err(|e| {
            SidecarError::Connection(format!("failed to flush sidecar socket: {e}"))
        })?;

        // Read one JSON-line response
        let mut reader = BufReader::new(read_half);
        let mut line = String::new();
        let bytes_read = tokio::time::timeout(Duration::from_secs(30), reader.read_line(&mut line))
            .await
            .map_err(|_| SidecarError::Timeout("sidecar response timed out (30s)".into()))?
            .map_err(|e| SidecarError::Connection(format!("failed to read from sidecar: {e}")))?;

        if bytes_read == 0 {
            return Err(SidecarError::Connection(
                "sidecar closed connection without responding".into(),
            ));
        }

        serde_json::from_str(line.trim()).map_err(|e| {
            SidecarError::Protocol(format!("invalid sidecar response: {e}: {line}"))
        })
    }

    /// Submit an SFT training job.
    pub async fn submit_sft(
        &self,
        job_id: &str,
        examples: Vec<serde_json::Value>,
        adapter_name: &str,
        epochs: usize,
        learning_rate: f64,
        lora_rank: usize,
        lora_alpha: f32,
    ) -> Result<SidecarResponse, SidecarError> {
        self.send(&SidecarRequest::SftRequest {
            job_id: job_id.to_string(),
            examples,
            adapter_name: adapter_name.to_string(),
            epochs,
            learning_rate,
            lora_rank,
            lora_alpha,
        })
        .await
    }

    /// Submit a GRPO training job.
    pub async fn submit_grpo(
        &self,
        job_id: &str,
        groups: Vec<serde_json::Value>,
        adapter_name: &str,
        learning_rate: f64,
        lora_rank: usize,
        lora_alpha: f32,
    ) -> Result<SidecarResponse, SidecarError> {
        self.send(&SidecarRequest::GrpoRequest {
            job_id: job_id.to_string(),
            groups,
            adapter_name: adapter_name.to_string(),
            learning_rate,
            lora_rank,
            lora_alpha,
        })
        .await
    }

    /// Query the status of a training job.
    pub async fn query_status(&self, job_id: &str) -> Result<SidecarResponse, SidecarError> {
        self.send(&SidecarRequest::StatusQuery {
            job_id: job_id.to_string(),
        })
        .await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("sidecar connection error: {0}")]
    Connection(String),
    #[error("sidecar protocol error: {0}")]
    Protocol(String),
    #[error("sidecar timeout: {0}")]
    Timeout(String),
}

/// Spawn a background task that polls tracked training jobs and auto-loads
/// completed adapters via the two-phase RwLock hot-swap pattern.
pub fn spawn_training_watcher(state: crate::state::AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;

            let client = match &state.sidecar {
                Some(c) if c.is_available() => c,
                _ => continue,
            };

            // Collect job IDs that need polling.
            let jobs_to_poll: Vec<(String, bool)> = {
                let jobs = state.training_jobs.read().unwrap();
                jobs.values()
                    .filter(|j| {
                        j.state == kiln_train::TrainingState::Queued
                            || j.state == kiln_train::TrainingState::Running
                    })
                    .map(|j| (j.job_id.clone(), j.auto_load))
                    .collect()
            };

            for (job_id, auto_load) in jobs_to_poll {
                let resp = match client.query_status(&job_id).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::debug!(job_id = %job_id, "watcher poll error: {e}");
                        continue;
                    }
                };

                match resp {
                    SidecarResponse::JobStatus {
                        job_id,
                        state: ref sidecar_state,
                        progress,
                        epoch,
                        loss,
                        adapter_path,
                        ..
                    } => {
                        let new_state = match sidecar_state.as_str() {
                            "completed" => kiln_train::TrainingState::Completed,
                            "failed" => kiln_train::TrainingState::Failed,
                            "running" => kiln_train::TrainingState::Running,
                            _ => kiln_train::TrainingState::Queued,
                        };

                        // Update tracked job state.
                        {
                            let mut jobs = state.training_jobs.write().unwrap();
                            if let Some(job) = jobs.get_mut(&job_id) {
                                job.state = new_state;
                                job.progress = progress;
                                job.epoch = epoch;
                                job.loss = loss;
                                job.adapter_path = adapter_path.clone();
                            }
                        }

                        if new_state == kiln_train::TrainingState::Completed {
                            if let Some(ref path) = adapter_path {
                                if auto_load {
                                    let adapter_name = {
                                        state
                                            .training_jobs
                                            .read()
                                            .unwrap()
                                            .get(&job_id)
                                            .map(|j| j.adapter_name.clone())
                                            .unwrap_or_default()
                                    };
                                    if let Err(e) =
                                        auto_load_adapter(&state, path, &adapter_name).await
                                    {
                                        tracing::error!(
                                            job_id = %job_id,
                                            adapter = %adapter_name,
                                            "auto-load failed: {e}"
                                        );
                                    } else {
                                        tracing::info!(
                                            "auto-loaded adapter {} from training job {}",
                                            adapter_name,
                                            job_id
                                        );
                                    }
                                }
                            }
                        }
                    }
                    SidecarResponse::JobComplete {
                        job_id,
                        adapter_path,
                    } => {
                        let (adapter_name, auto_load_flag) = {
                            let mut jobs = state.training_jobs.write().unwrap();
                            if let Some(job) = jobs.get_mut(&job_id) {
                                job.state = kiln_train::TrainingState::Completed;
                                job.progress = 1.0;
                                job.adapter_path = Some(adapter_path.clone());
                                (job.adapter_name.clone(), job.auto_load)
                            } else {
                                continue;
                            }
                        };
                        if auto_load_flag {
                            if let Err(e) =
                                auto_load_adapter(&state, &adapter_path, &adapter_name).await
                            {
                                tracing::error!(
                                    job_id = %job_id,
                                    adapter = %adapter_name,
                                    "auto-load failed: {e}"
                                );
                            } else {
                                tracing::info!(
                                    "auto-loaded adapter {} from training job {}",
                                    adapter_name,
                                    job_id
                                );
                            }
                        }
                    }
                    SidecarResponse::Error { job_id, message } => {
                        let mut jobs = state.training_jobs.write().unwrap();
                        if let Some(job) = jobs.get_mut(&job_id) {
                            job.state = kiln_train::TrainingState::Failed;
                        }
                        tracing::warn!(job_id = %job_id, "training job failed: {message}");
                    }
                    _ => {}
                }
            }
        }
    });
}

/// Load a LoRA adapter using the two-phase RwLock pattern (same as adapters.rs).
async fn auto_load_adapter(
    state: &crate::state::AppState,
    adapter_path: &str,
    adapter_name: &str,
) -> Result<(), String> {
    use crate::state::ModelBackend;
    use kiln_model::lora_loader::LoraWeights;

    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner.clone(),
        ModelBackend::Mock { .. } => {
            // In mock mode, just update the active adapter name.
            *state.active_adapter_name.write().unwrap() = Some(adapter_name.to_string());
            return Ok(());
        }
    };

    let path = std::path::PathBuf::from(adapter_path);
    if !path.exists() {
        return Err(format!(
            "adapter path does not exist: {}",
            path.display()
        ));
    }

    // Phase 1: read device/num_layers under brief read lock.
    let (device, num_layers) = {
        let guard = runner.read().unwrap();
        (
            guard.weights.embed_tokens.device().clone(),
            guard.config.num_layers,
        )
    };

    // Phase 2: load weights outside any lock.
    let name = adapter_name.to_string();
    let active_adapter = state.active_adapter_name.clone();
    tokio::task::spawn_blocking(move || {
        let lora =
            LoraWeights::load(&path, num_layers, &device).map_err(|e| format!("{e}"))?;
        // Brief write lock to swap.
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
        // Update active adapter name.
        *active_adapter.write().unwrap() = Some(name);
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

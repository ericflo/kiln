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

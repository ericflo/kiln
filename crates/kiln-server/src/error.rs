//! Structured API error responses.
//!
//! Every error returned by the API is a JSON object with a consistent shape:
//! ```json
//! {
//!   "error": {
//!     "code": "adapter_not_found",
//!     "message": "Adapter 'foo' does not exist",
//!     "hint": "List available adapters with GET /v1/adapters"
//!   }
//! }
//! ```

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// JSON error body shape, matching OpenAI's convention.
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
    hint: &'static str,
}

/// Structured API error with HTTP status, machine-readable code, human message,
/// and an actionable hint.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub hint: &'static str,
}

impl ApiError {
    // ── Chat completions ────────────────────────────────────────────

    pub fn chat_template_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_messages",
            message: format!("Failed to apply chat template: {detail}"),
            hint: "Check that each message has a valid 'role' (system, user, assistant) and non-empty 'content'.",
        }
    }

    pub fn tokenization_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "tokenization_error",
            message: format!("Tokenization failed: {detail}"),
            hint: "This is a server-side error. If it persists, check that the tokenizer files are not corrupted.",
        }
    }

    pub fn generation_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "generation_error",
            message: format!("Text generation failed: {detail}"),
            hint: "Retry the request. If the error mentions OOM, try reducing max_tokens or freeing GPU memory.",
        }
    }

    pub fn request_timeout(timeout_secs: u64) -> Self {
        Self {
            status: StatusCode::REQUEST_TIMEOUT,
            code: "request_timeout",
            message: format!("Request timed out after {timeout_secs} seconds"),
            hint: "Try reducing max_tokens, or increase the server's request_timeout_secs in the config file.",
        }
    }

    pub fn streaming_not_supported_mock() -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "streaming_not_supported",
            message: "Streaming is not supported with the mock backend".to_string(),
            hint: "Start the server with a real model (set model.path in config) to enable streaming.",
        }
    }

    // ── Adapters ────────────────────────────────────────────────────

    pub fn adapter_not_found(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "adapter_not_found",
            message: format!("Adapter '{name}' does not exist"),
            hint: "List available adapters with GET /v1/adapters.",
        }
    }

    pub fn adapter_load_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_load_failed",
            message: format!("Failed to load adapter: {detail}"),
            hint: "Check that the adapter directory contains adapter_config.json and adapter_model.safetensors.",
        }
    }

    pub fn adapter_active(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_active",
            message: format!("Adapter '{name}' is currently active and cannot be deleted"),
            hint: "Unload the adapter first with POST /v1/adapters/unload, then retry the delete.",
        }
    }

    pub fn adapter_delete_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_delete_failed",
            message: format!("Failed to delete adapter directory: {detail}"),
            hint: "Check file permissions on the adapter directory.",
        }
    }

    pub fn adapter_merge_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_merge_invalid",
            message: format!("Cannot merge adapters: {detail}"),
            hint: "All sources must share the same rank, target_modules, base_model, and tensor shapes. Linear interpolation requires identical adapter layouts.",
        }
    }

    pub fn adapter_merge_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_merge_failed",
            message: format!("Adapter merge failed: {detail}"),
            hint: "Check server logs for the underlying I/O or serialization error.",
        }
    }

    pub fn adapter_merge_output_exists(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_merge_output_exists",
            message: format!("Output adapter '{name}' already exists"),
            hint: "Choose a different output_name, or delete the existing adapter first with DELETE /v1/adapters/{name}.",
        }
    }

    pub fn adapter_merge_bad_name(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_merge_bad_name",
            message: format!("Invalid output_name '{name}'"),
            hint: "output_name must be non-empty, contain no path separators, and not be '.' or '..'.",
        }
    }

    pub fn mock_mode_no_adapters() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "mock_mode",
            message: "Adapter management is not supported in mock mode".to_string(),
            hint: "Start the server with a real model (set model.path in config) to use adapters.",
        }
    }

    // ── Training ────────────────────────────────────────────────────

    pub fn shutting_down() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "server_shutting_down",
            message: "Server is shutting down — not accepting new requests".to_string(),
            hint: "Wait for the server to restart, then retry.",
        }
    }

    pub fn mock_mode_no_training() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "mock_mode",
            message: "Training requires real model weights (not available in mock mode)".to_string(),
            hint: "Start the server with a real model (set model.path in config) to use training.",
        }
    }

    pub fn training_job_not_found(job_id: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "training_job_not_found",
            message: format!("Training job '{job_id}' not found"),
            hint: "List all training jobs with GET /v1/train/status.",
        }
    }

    pub fn training_job_not_cancellable(
        job_id: impl std::fmt::Display,
        state: impl std::fmt::Display,
    ) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "training_job_not_cancellable",
            message: format!("Cannot cancel job '{job_id}': current state is {state}"),
            hint: "Only jobs in 'queued' state can be cancelled. Running jobs must complete.",
        }
    }

    pub fn training_job_already_started(job_id: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "training_job_already_started",
            message: format!("Job '{job_id}' was not found in the queue (it may have already started)"),
            hint: "Check job status with GET /v1/train/status/{job_id}.",
        }
    }

    // ── Generic ─────────────────────────────────────────────────────

    pub fn internal(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: format!("Internal error: {detail}"),
            hint: "This is unexpected. Check server logs for details.",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                code: self.code,
                message: self.message,
                hint: self.hint,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

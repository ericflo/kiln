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

use axum::Json;
use axum::http::StatusCode;
use axum::http::header::{HeaderValue, RETRY_AFTER};
use axum::response::{IntoResponse, Response};
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
    /// When set, emit a `Retry-After: <N>` header (seconds) on the response.
    /// Used by 503 errors that want to suggest a backoff to the client.
    pub retry_after_seconds: Option<u64>,
}

impl ApiError {
    // ── Chat completions ────────────────────────────────────────────

    pub fn chat_template_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_messages",
            message: format!("Failed to apply chat template: {detail}"),
            hint: "Check that each message has a valid 'role' (system, user, assistant) and non-empty 'content'.",
            retry_after_seconds: None,
        }
    }

    pub fn tokenization_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "tokenization_error",
            message: format!("Tokenization failed: {detail}"),
            hint: "This is a server-side error. If it persists, check that the tokenizer files are not corrupted.",
            retry_after_seconds: None,
        }
    }

    pub fn generation_failed(detail: anyhow::Error) -> Self {
        // `{:#}` flattens the anyhow Context chain so the client sees the
        // root cause, not just the outermost wrapper. Without this, errors
        // like "prefill forward pass (paged) failed" surface with no hint
        // about which kernel / op actually failed.
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "generation_error",
            message: format!("Text generation failed: {detail:#}"),
            hint: "Retry the request. If the error mentions OOM, try reducing max_tokens or freeing GPU memory.",
            retry_after_seconds: None,
        }
    }

    pub fn request_timeout(timeout_secs: u64) -> Self {
        Self {
            status: StatusCode::REQUEST_TIMEOUT,
            code: "request_timeout",
            message: format!("Request timed out after {timeout_secs} seconds"),
            hint: "Try reducing max_tokens, or increase the server's request_timeout_secs in the config file.",
            retry_after_seconds: None,
        }
    }

    pub fn streaming_not_supported_mock() -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "streaming_not_supported",
            message: "Streaming is not supported with the mock backend".to_string(),
            hint: "Start the server with a real model (set model.path in config) to enable streaming.",
            retry_after_seconds: None,
        }
    }

    pub fn chat_invalid_request(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "chat_invalid_request",
            message: format!("Invalid chat completion request: {detail}"),
            hint: "POST {messages: [{role,content}, ...], n: <int>=1, ...sampling}. Non-streaming n choices must stay within the configured cap.",
            retry_after_seconds: None,
        }
    }

    // ── Adapters ────────────────────────────────────────────────────

    pub fn adapter_not_found(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "adapter_not_found",
            message: format!("Adapter '{name}' does not exist"),
            hint: "List available adapters with GET /v1/adapters.",
            retry_after_seconds: None,
        }
    }

    pub fn invalid_adapter_name(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_adapter_name",
            message: format!("Invalid adapter name '{name}'"),
            hint: "Adapter names must be a single path segment: no '/', '\\', or '..', and not absolute.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_export_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_export_failed",
            message: format!("Failed to export adapter: {detail}"),
            hint: "Check server logs and that the adapter directory is readable.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_load_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_load_failed",
            message: format!("Failed to load adapter: {detail}"),
            hint: "Check that the adapter directory contains adapter_config.json and adapter_model.safetensors.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_quarantined(
        name: impl std::fmt::Display,
        reason: impl std::fmt::Display,
    ) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_quarantined",
            message: format!("Adapter '{name}' is quarantined by failed canary checks: {reason}"),
            hint: "Inspect GET /v1/adapters for canary_failure_reason. To load anyway, POST /v1/adapters/load with allow_quarantined=true.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_layout_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_layout_invalid",
            message: format!("Invalid adapter directory layout: {detail}"),
            hint: "Point /v1/adapters/load at the actual adapter directory containing adapter_config.json and adapter_model.safetensors, not its parent output directory.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_active(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_active",
            message: format!("Adapter '{name}' is currently active and cannot be deleted"),
            hint: "Unload the adapter first with POST /v1/adapters/unload, then retry the delete.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_delete_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_delete_failed",
            message: format!("Failed to delete adapter directory: {detail}"),
            hint: "Check file permissions on the adapter directory.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_merge_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_merge_invalid",
            message: format!("Cannot merge adapters: {detail}"),
            hint: "All sources must share the same rank, target_modules, base_model, and tensor shapes. Linear interpolation requires identical adapter layouts.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_merge_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_merge_failed",
            message: format!("Adapter merge failed: {detail}"),
            hint: "Check server logs for the underlying I/O or serialization error.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_merge_output_exists(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_merge_output_exists",
            message: format!("Output adapter '{name}' already exists"),
            hint: "Choose a different output_name, or delete the existing adapter first with DELETE /v1/adapters/{name}.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_merge_bad_name(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_merge_bad_name",
            message: format!("Invalid output_name '{name}'"),
            hint: "output_name must be non-empty, contain no path separators, and not be '.' or '..'.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_already_exists(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "adapter_already_exists",
            message: format!("Adapter '{name}' already exists"),
            hint: "Run `curl -X DELETE /v1/adapters/{name}` to remove the existing adapter before re-uploading, or upload under a different name.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_import_failed(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "adapter_import_failed",
            message: format!("Failed to import adapter: {detail}"),
            hint: "Check that the uploaded archive is a valid tar.gz produced by GET /v1/adapters/{name}/download. Server logs have details.",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_import_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "adapter_import_invalid",
            message: format!("Invalid adapter upload: {detail}"),
            hint: "POST multipart/form-data with two fields: 'name' (text, single segment) and 'archive' (file, gzipped tar produced by GET /v1/adapters/{name}/download).",
            retry_after_seconds: None,
        }
    }

    pub fn adapter_disk_quota_exceeded(detail: impl std::fmt::Display) -> Self {
        Self {
            // 507 Insufficient Storage — the request is well-formed but the
            // server cannot accept it without exceeding the configured cap.
            status: StatusCode::INSUFFICIENT_STORAGE,
            code: "adapter_disk_quota_exceeded",
            message: format!("Adapter upload would exceed adapter_dir disk cap: {detail}"),
            hint: "Delete unused adapters with `curl -X DELETE /v1/adapters/{name}`, or raise the cap with `KILN_ADAPTERS_MAX_DISK_BYTES` (set to 0 to disable).",
            retry_after_seconds: None,
        }
    }

    pub fn invalid_compose_request(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_compose_request",
            message: format!("Invalid adapter composition request: {detail}"),
            hint: "Specify either 'adapter' (single name) or 'adapters' (non-empty list of {name, scale}), not both. The composed adapter is merged once and cached on disk.",
            retry_after_seconds: None,
        }
    }

    pub fn mock_mode_no_adapters() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "mock_mode",
            message: "Adapter management is not supported in mock mode".to_string(),
            hint: "Start the server with a real model (set model.path in config) to use adapters.",
            retry_after_seconds: None,
        }
    }

    // ── Training ────────────────────────────────────────────────────

    pub fn shutting_down() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "server_shutting_down",
            message: "Server is shutting down — not accepting new requests".to_string(),
            hint: "Wait for the server to restart, then retry.",
            retry_after_seconds: None,
        }
    }

    pub fn mock_mode_no_training() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "mock_mode",
            message: "Training requires real model weights (not available in mock mode)"
                .to_string(),
            hint: "Start the server with a real model (set model.path in config) to use training.",
            retry_after_seconds: None,
        }
    }

    /// 503 returned when the in-memory training queue has reached its
    /// configured cap. Carries `Retry-After: 30` so polite clients back
    /// off automatically. The cap is `training.max_queued_jobs` (default
    /// 32, override `KILN_TRAINING_MAX_QUEUED_JOBS`).
    pub fn training_queue_full(max: usize) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "training_queue_full",
            message: format!("Training queue is at capacity ({max} jobs queued)"),
            hint: "Wait for in-flight jobs to drain, or raise training.max_queued_jobs in the config (env: KILN_TRAINING_MAX_QUEUED_JOBS).",
            retry_after_seconds: Some(30),
        }
    }

    /// 503 returned when the in-memory training-jobs tracking map has
    /// reached its configured cap. Distinct from `training_queue_full`:
    /// this fires when too many terminal (`Completed` / `Failed`) entries
    /// are still resident waiting for the TTL GC to evict them. Carries
    /// `Retry-After: 30` so polite clients back off automatically. The
    /// cap is `training.max_tracked_jobs` (default 1024, override
    /// `KILN_TRAINING_MAX_TRACKED_JOBS`); the TTL is
    /// `training.tracked_job_ttl_secs` (default 3600, override
    /// `KILN_TRAINING_TRACKED_JOB_TTL_SECS`).
    pub fn training_tracked_full(max: usize) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "training_tracked_full",
            message: format!("Training tracking map is at capacity ({max} tracked jobs)"),
            hint: "Wait for terminal entries to TTL out, or raise training.max_tracked_jobs in the config (env: KILN_TRAINING_MAX_TRACKED_JOBS).",
            retry_after_seconds: Some(30),
        }
    }

    /// 413 returned when the training preflight estimator concludes
    /// the submitted job's working set won't fit in the available
    /// memory budget. The detailed numbers (estimate, available,
    /// breakdown) live in `message`; the static `hint` lists the
    /// knobs the caller can turn down.
    pub fn training_will_not_fit(detailed_message: String) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "training_will_not_fit",
            message: detailed_message,
            hint: "Lower the per-step memory by raising KILN_GRAD_CHECKPOINT_SEGMENTS, shrinking lora_rank, sending fewer/shorter examples, or — only if your host can spare RAM — overriding KILN_TRAINING_MEMORY_RESERVE_GB.",
            retry_after_seconds: None,
        }
    }

    pub fn training_invalid_request(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "training_invalid_request",
            message: format!("Invalid training request: {detail}"),
            hint: "For GRPO, submit either a groups array or a dataset_path JSONL file path. Do not send both.",
            retry_after_seconds: None,
        }
    }

    pub fn training_job_not_found(job_id: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "training_job_not_found",
            message: format!("Training job '{job_id}' not found"),
            hint: "List all training jobs with GET /v1/train/status.",
            retry_after_seconds: None,
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
            retry_after_seconds: None,
        }
    }

    pub fn training_job_already_started(job_id: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "training_job_already_started",
            message: format!(
                "Job '{job_id}' was not found in the queue (it may have already started)"
            ),
            hint: "Check job status with GET /v1/train/status/{job_id}.",
            retry_after_seconds: None,
        }
    }

    // ── Eval ────────────────────────────────────────────────────────

    pub fn eval_invalid_request(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "eval_invalid_request",
            message: format!("Invalid eval request: {detail}"),
            hint: "Send either a registered suite name OR an inline suite document. See docs/EVAL_GUIDE.md for the schema.",
            retry_after_seconds: None,
        }
    }

    pub fn eval_suite_not_found(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "eval_suite_not_found",
            message: format!("Eval suite '{name}' not found"),
            hint: "List registered suites with GET /v1/eval/suites.",
            retry_after_seconds: None,
        }
    }

    pub fn eval_suite_exists(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "eval_suite_exists",
            message: format!("Eval suite '{name}' already exists"),
            hint: "Use ?force=true on POST /v1/eval/suites to overwrite.",
            retry_after_seconds: None,
        }
    }

    pub fn eval_job_not_found(job_id: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "eval_job_not_found",
            message: format!("Eval job '{job_id}' not found"),
            hint: "List jobs with GET /v1/eval/jobs.",
            retry_after_seconds: None,
        }
    }

    pub fn eval_queue_full(max: usize) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "eval_queue_full",
            message: format!("Eval queue is at capacity ({max} jobs queued)"),
            hint: "Wait for in-flight evals to drain, or raise eval.max_queued_jobs in the config.",
            retry_after_seconds: Some(15),
        }
    }

    pub fn eval_tracked_full(max: usize) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "eval_tracked_full",
            message: format!("Eval tracking map is at capacity ({max} tracked jobs)"),
            hint: "Wait for terminal eval entries to TTL out, or raise eval.max_tracked_jobs.",
            retry_after_seconds: Some(15),
        }
    }

    pub fn eval_registry_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "eval_registry_unavailable",
            message: "Server has no eval directory configured".to_string(),
            hint: "Start the server with an eval_dir set in kiln.toml, or POST an inline suite instead of a registered name.",
            retry_after_seconds: None,
        }
    }

    pub fn dataset_not_found(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "dataset_not_found",
            message: format!("Eval dataset '{name}' not found"),
            hint: "List datasets with GET /v1/eval/datasets.",
            retry_after_seconds: None,
        }
    }

    pub fn dataset_exists(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "dataset_exists",
            message: format!("Eval dataset '{name}' already exists"),
            hint: "Delete or rename the existing dataset, or use a different name.",
            retry_after_seconds: None,
        }
    }

    pub fn dataset_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "dataset_invalid",
            message: format!("Invalid eval dataset: {detail}"),
            hint: "Each line must be a valid JSON object. SFT datasets must have a `messages` array.",
            retry_after_seconds: None,
        }
    }

    pub fn dataset_registry_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "dataset_registry_unavailable",
            message: "Server has no dataset directory configured".to_string(),
            hint: "Start the server with an eval_dir set or restart to let kiln create the default `<adapter_dir>/.eval/` location.",
            retry_after_seconds: None,
        }
    }

    pub fn judgment_store_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "judgment_store_unavailable",
            message: "Server has no judgment store configured".to_string(),
            hint: "Same setup as the dataset registry — the judgment store lives under `<adapter_dir>/.eval/judgments/`.",
            retry_after_seconds: None,
        }
    }

    pub fn judgment_not_found(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "judgment_not_found",
            message: format!("Judgment dataset '{name}' not found"),
            hint: "List judgments with GET /v1/judgments.",
            retry_after_seconds: None,
        }
    }

    pub fn judgment_invalid(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "judgment_invalid",
            message: format!("Invalid judgment: {detail}"),
            hint: "Required fields: prompt, response_a, response_b, winner (one of `a` | `b` | `tie` | `skip`).",
            retry_after_seconds: None,
        }
    }

    pub fn invalid_suite_name(name: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_suite_name",
            message: format!("Invalid suite name '{name}'"),
            hint: "Suite names must be non-empty, not contain path separators or '..', and not be absolute paths.",
            retry_after_seconds: None,
        }
    }

    // ── Batch completions ───────────────────────────────────────────

    pub fn batch_invalid_request(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "batch_invalid_request",
            message: format!("Invalid batch completion request: {detail}"),
            hint: "POST {prompts: [[{role,content}, ...], ...], n: <int>=1, ...sampling}. Total outputs = prompts.len() * n must not exceed the configured cap.",
            retry_after_seconds: None,
        }
    }

    pub fn batch_too_large(requested: usize, cap: usize) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "batch_too_large",
            message: format!(
                "Batch would produce {requested} completions, which exceeds the cap of {cap}"
            ),
            hint: "Reduce prompts.len() or n so prompts.len() * n <= cap, or split into multiple smaller batch requests.",
            retry_after_seconds: None,
        }
    }

    // ── Generic ─────────────────────────────────────────────────────

    pub fn internal(detail: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: format!("Internal error: {detail}"),
            hint: "This is unexpected. Check server logs for details.",
            retry_after_seconds: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry_after = self.retry_after_seconds;
        let body = ErrorBody {
            error: ErrorDetail {
                code: self.code,
                message: self.message,
                hint: self.hint,
            },
        };
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = retry_after {
            // HeaderValue::from(u64) is infallible (base-10 ASCII).
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

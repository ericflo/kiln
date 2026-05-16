//! Eval HTTP API. The full endpoint matrix:
//!
//! | Method | Path                                | Description                                  |
//! | ------ | ----------------------------------- | -------------------------------------------- |
//! | GET    | /v1/eval/suites                     | List registered suites                       |
//! | GET    | /v1/eval/suites/:name               | Fetch one suite's full JSON                  |
//! | POST   | /v1/eval/suites                     | Register a suite (body = `EvalSuite`)        |
//! | DELETE | /v1/eval/suites/:name               | Delete a registered suite                    |
//! | POST   | /v1/eval/run                        | Submit an eval job (registered or inline)    |
//! | POST   | /v1/eval/compare                    | Submit a compare-mode job (A/B/C adapters)   |
//! | GET    | /v1/eval/jobs                       | List all eval jobs                           |
//! | GET    | /v1/eval/jobs/:job_id               | Per-job status + results                     |
//! | DELETE | /v1/eval/jobs/:job_id               | Cancel a queued / running eval job           |

use std::sync::atomic::Ordering;

use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use kiln_eval::synthesis::SynthesisConfig;
use kiln_eval::{
    EvalCompareSpec, EvalGenerationParams, EvalJobState, EvalResult, EvalSuite, EvalSuiteSummary,
};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::eval::datasets::{DatasetFormat, DatasetManifest};
use crate::eval::judgments::{
    JudgmentManifest, JudgmentMessage, JudgmentRow, JudgmentWinner, build_validation_suite,
    compile_judgments_to_sft, format_judge_prompt,
};
use crate::eval::queue::{EvalJobInfo, EvalSubmissionKind, QueuedEvalJob};
use crate::eval::synthesis_driver::{preview_synthesis, synthesize_and_save};
use crate::state::AppState;

/// Submission body for `POST /v1/eval/run`.
///
/// Exactly one of `suite` (registered name) or `inline_suite` must be set.
/// Adapter selection is optional: omit to evaluate the currently active
/// adapter (or the base model when no adapter is loaded).
#[derive(Debug, Deserialize)]
pub struct EvalRunRequest {
    /// Registered suite name. Mutually exclusive with `inline_suite`.
    #[serde(default)]
    pub suite: Option<String>,
    /// Inline suite document. Mutually exclusive with `suite`.
    #[serde(default)]
    pub inline_suite: Option<EvalSuite>,
    /// Adapter to evaluate. `null` (or omitted) means "evaluate the
    /// currently active adapter, whatever that is". Empty string means
    /// "base model (no adapter)".
    #[serde(default)]
    pub adapter: Option<String>,
    /// Suite-wide generation override. Per-example overrides on the suite
    /// itself still win over this.
    #[serde(default)]
    pub generation: Option<EvalGenerationParams>,
}

#[derive(Debug, Serialize)]
pub struct EvalRunResponse {
    pub job_id: String,
    pub state: EvalJobState,
    pub message: String,
}

/// Optional query for `POST /v1/eval/suites?force=true`.
#[derive(Debug, Deserialize, Default)]
struct SuiteUpsertQuery {
    #[serde(default)]
    force: Option<bool>,
}

#[derive(Debug, Serialize)]
struct SuiteSaveResponse {
    name: String,
    path: String,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct SuiteListResponse {
    suites: Vec<EvalSuiteSummary>,
}

#[derive(Debug, Serialize)]
struct EvalJobListResponse {
    jobs: Vec<EvalJobInfo>,
}

const EVAL_BODY_LIMIT: usize = 32 * 1024 * 1024;
const SUITE_BODY_LIMIT: usize = 32 * 1024 * 1024;
/// 1 GiB cap on uploaded datasets (matches `datasets::DATASET_MAX_BYTES`).
const DATASET_BODY_LIMIT: usize = 1024 * 1024 * 1024;

async fn list_suites(State(state): State<AppState>) -> Result<Json<SuiteListResponse>, ApiError> {
    let Some(reg) = state.suite_registry.as_ref() else {
        // No registry = no suites, but we still return 200 with an empty list
        // so dashboards have something to render in mock mode.
        return Ok(Json(SuiteListResponse { suites: vec![] }));
    };
    Ok(Json(SuiteListResponse {
        suites: reg.list(),
    }))
}

async fn get_suite(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<EvalSuite>, ApiError> {
    let reg = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    let suite = reg
        .load(&name)
        .map_err(|e| match e {
            crate::eval::SuiteRegistryError::NotFound(_) => ApiError::eval_suite_not_found(&name),
            crate::eval::SuiteRegistryError::InvalidName(n) => ApiError::invalid_suite_name(n),
            other => ApiError::eval_invalid_request(format!("{other}")),
        })?;
    Ok(Json(suite))
}

async fn save_suite(
    State(state): State<AppState>,
    Query(query): Query<SuiteUpsertQuery>,
    Json(suite): Json<EvalSuite>,
) -> Result<Json<SuiteSaveResponse>, ApiError> {
    let reg = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    let force = query.force.unwrap_or(false);
    let path = reg.save(&suite, force).map_err(|e| match e {
        crate::eval::SuiteRegistryError::InvalidName(n) => ApiError::invalid_suite_name(n),
        crate::eval::SuiteRegistryError::AlreadyExists(n) => ApiError::eval_suite_exists(n),
        other => ApiError::eval_invalid_request(format!("{other}")),
    })?;
    Ok(Json(SuiteSaveResponse {
        name: suite.name,
        path: path.display().to_string(),
        status: if force { "overwritten" } else { "created" },
    }))
}

async fn delete_suite(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let reg = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    reg.delete(&name).map_err(|e| match e {
        crate::eval::SuiteRegistryError::InvalidName(n) => ApiError::invalid_suite_name(n),
        crate::eval::SuiteRegistryError::NotFound(_) => ApiError::eval_suite_not_found(&name),
        other => ApiError::eval_invalid_request(format!("{other}")),
    })?;
    Ok(Json(serde_json::json!({"status": "deleted", "name": name})))
}

async fn submit_eval(
    State(state): State<AppState>,
    Json(req): Json<EvalRunRequest>,
) -> Result<Json<EvalRunResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    let (suite_present, inline_present) = (req.suite.is_some(), req.inline_suite.is_some());
    if suite_present == inline_present {
        return Err(ApiError::eval_invalid_request(
            "exactly one of `suite` or `inline_suite` must be set",
        ));
    }

    // Reject when at queue / tracked cap.
    let qlen = state.eval_queue.lock().unwrap().len();
    if qlen >= state.max_queued_eval_jobs {
        return Err(ApiError::eval_queue_full(state.max_queued_eval_jobs));
    }
    let tlen = state.eval_jobs.read().unwrap().len();
    if tlen >= state.max_tracked_eval_jobs {
        return Err(ApiError::eval_tracked_full(state.max_tracked_eval_jobs));
    }

    // Normalize the adapter selection: empty string means base model.
    let adapter = req.adapter.and_then(|s| if s.is_empty() { None } else { Some(s) });
    let adapters = vec![adapter.clone()];

    let (suite_name, queued_job) = if let Some(name) = req.suite {
        let reg = state
            .suite_registry
            .as_ref()
            .ok_or_else(ApiError::eval_registry_unavailable)?;
        // Validate that the suite exists at submit time so callers get a 404
        // immediately rather than waiting for the worker to fail it.
        reg.load(&name).map_err(|e| match e {
            crate::eval::SuiteRegistryError::NotFound(_) => ApiError::eval_suite_not_found(&name),
            crate::eval::SuiteRegistryError::InvalidName(n) => ApiError::invalid_suite_name(n),
            other => ApiError::eval_invalid_request(format!("{other}")),
        })?;
        let queued = QueuedEvalJob::Registered {
            suite_name: name.clone(),
            adapter,
            generation_override: req.generation,
        };
        (name, queued)
    } else {
        let suite = req
            .inline_suite
            .ok_or_else(|| ApiError::eval_invalid_request("inline_suite missing"))?;
        let suite_name = suite.name.clone();
        let queued = QueuedEvalJob::Inline {
            suite: Box::new(suite),
            adapter,
            generation_override: req.generation,
        };
        (suite_name, queued)
    };

    let job_id = state.enqueue_eval(
        suite_name.clone(),
        adapters,
        EvalSubmissionKind::OnDemand,
        None,
        queued_job,
    );
    tracing::info!(job_id = %job_id, suite = %suite_name, "eval job queued");
    Ok(Json(EvalRunResponse {
        job_id,
        state: EvalJobState::Queued,
        message: format!("Queued eval against suite `{suite_name}`"),
    }))
}

async fn submit_compare(
    State(state): State<AppState>,
    Json(spec): Json<EvalCompareSpec>,
) -> Result<Json<EvalRunResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    if spec.adapters.is_empty() {
        return Err(ApiError::eval_invalid_request(
            "compare requires at least one adapter",
        ));
    }
    if spec.adapters.len() > 8 {
        return Err(ApiError::eval_invalid_request(
            "compare supports at most 8 adapters per submission",
        ));
    }
    let qlen = state.eval_queue.lock().unwrap().len();
    if qlen >= state.max_queued_eval_jobs {
        return Err(ApiError::eval_queue_full(state.max_queued_eval_jobs));
    }
    let tlen = state.eval_jobs.read().unwrap().len();
    if tlen >= state.max_tracked_eval_jobs {
        return Err(ApiError::eval_tracked_full(state.max_tracked_eval_jobs));
    }
    // Compare always references a registered suite — there's no concise way
    // to inline a multi-adapter run.
    let reg = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    reg.load(&spec.suite).map_err(|e| match e {
        crate::eval::SuiteRegistryError::NotFound(_) => ApiError::eval_suite_not_found(&spec.suite),
        crate::eval::SuiteRegistryError::InvalidName(n) => ApiError::invalid_suite_name(n),
        other => ApiError::eval_invalid_request(format!("{other}")),
    })?;
    let adapters: Vec<Option<String>> = spec
        .adapters
        .iter()
        .map(|a| if a.is_empty() { None } else { Some(a.clone()) })
        .collect();
    let suite_name = spec.suite.clone();
    let job_id = state.enqueue_eval(
        suite_name.clone(),
        adapters,
        EvalSubmissionKind::Compare,
        None,
        QueuedEvalJob::Compare(spec),
    );
    Ok(Json(EvalRunResponse {
        job_id,
        state: EvalJobState::Queued,
        message: format!("Queued compare-mode eval against suite `{suite_name}`"),
    }))
}

async fn list_jobs(State(state): State<AppState>) -> Json<EvalJobListResponse> {
    let jobs = state.eval_jobs.read().unwrap();
    let mut jobs: Vec<EvalJobInfo> = jobs.values().cloned().collect();
    jobs.sort_by(|a, b| b.submitted_at_iso.cmp(&a.submitted_at_iso));
    Json(EvalJobListResponse { jobs })
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<EvalResult>, ApiError> {
    let jobs = state.eval_jobs.read().unwrap();
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| ApiError::eval_job_not_found(&job_id))?;
    Ok(Json(job.to_result()))
}

/// `POST /v1/eval/jobs/:job_id/rerun` — re-runs the failing examples from
/// a completed job. Body picks the adapter (defaults to whatever the
/// original ran against) and optionally an `outcome_kinds` filter
/// (defaults to ["fail", "invalid", "error"]). Returns a new `job_id`.
#[derive(Debug, Deserialize, Default)]
struct RerunBody {
    /// Adapter to evaluate. `null` = use the original job's adapter.
    /// Empty string = base model.
    #[serde(default)]
    adapter: Option<String>,
    /// Outcome kinds to include in the re-run. Defaults to everything
    /// that didn't pass.
    #[serde(default)]
    outcome_kinds: Option<Vec<String>>,
    /// When true, also include "pass" outcomes. False by default — the
    /// whole point of this endpoint is debugging failures.
    #[serde(default)]
    include_pass: bool,
}

async fn rerun_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
    Json(body): Json<RerunBody>,
) -> Result<Json<EvalRunResponse>, ApiError> {
    let suites = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    let (suite_name, adapter, example_ids) = {
        let jobs = state.eval_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::eval_job_not_found(&job_id))?;
        let want_kinds: std::collections::HashSet<String> = body
            .outcome_kinds
            .clone()
            .unwrap_or_else(|| vec!["fail".into(), "invalid".into(), "error".into()])
            .into_iter()
            .collect();
        let mut ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for run in &job.finished_runs {
            for o in &run.outcomes {
                let kind = match o.kind {
                    kiln_eval::EvalOutcomeKind::Pass => "pass",
                    kiln_eval::EvalOutcomeKind::Fail => "fail",
                    kiln_eval::EvalOutcomeKind::Invalid => "invalid",
                    kiln_eval::EvalOutcomeKind::Error => "error",
                };
                if !body.include_pass && kind == "pass" {
                    continue;
                }
                if want_kinds.contains(kind) || (body.include_pass && kind == "pass") {
                    ids.insert(o.example_id.clone());
                }
            }
        }
        let suite_name = job.suite_name.clone();
        let adapter = body
            .adapter
            .clone()
            .or_else(|| job.adapters.first().and_then(|a| a.clone()));
        (suite_name, adapter, ids.into_iter().collect::<Vec<_>>())
    };
    if example_ids.is_empty() {
        return Err(ApiError::eval_invalid_request(
            "no matching outcomes to re-run",
        ));
    }
    let inline = crate::eval::rerun_filtered_suite(suites, &suite_name, &example_ids)
        .map_err(|e| match e {
            crate::eval::rerun::RerunError::Registry(
                crate::eval::SuiteRegistryError::NotFound(_),
            ) => ApiError::eval_suite_not_found(&suite_name),
            other => ApiError::eval_invalid_request(format!("{other}")),
        })?;
    let suite_label = inline.name.clone();
    let new_job_id = state.enqueue_eval(
        suite_label.clone(),
        vec![adapter.clone()],
        EvalSubmissionKind::OnDemand,
        None,
        QueuedEvalJob::Inline {
            suite: Box::new(inline),
            adapter,
            generation_override: None,
        },
    );
    Ok(Json(EvalRunResponse {
        job_id: new_job_id,
        state: EvalJobState::Queued,
        message: format!(
            "Queued re-run of {} example(s) from `{suite_name}`",
            example_ids.len()
        ),
    }))
}

async fn cancel_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Remove from queue if still pending.
    let removed = {
        let mut q = state.eval_queue.lock().unwrap();
        q.remove(&job_id)
    };
    // Look up current state. We may mutate (cancel) or delete (terminal).
    let current_state = {
        let jobs = state.eval_jobs.read().unwrap();
        jobs.get(&job_id).map(|j| j.state)
    };
    match current_state {
        None => Err(ApiError::eval_job_not_found(&job_id)),
        Some(EvalJobState::Queued) => {
            let mut jobs = state.eval_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = EvalJobState::Cancelled;
                job.finished_at_iso = Some(chrono::Utc::now().to_rfc3339());
                job.finished_at = Some(std::time::Instant::now());
            }
            Ok(Json(serde_json::json!({
                "status": "cancelled",
                "job_id": job_id,
                "was_in_queue": removed,
            })))
        }
        Some(EvalJobState::Running) => {
            // Running cancellation is best-effort: the worker checks the
            // tracked state at iteration boundaries.
            let mut jobs = state.eval_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = EvalJobState::Cancelled;
            }
            Ok(Json(serde_json::json!({
                "status": "cancelling",
                "job_id": job_id,
                "note": "running job will exit at the next example boundary",
            })))
        }
        Some(
            EvalJobState::Cancelled | EvalJobState::Completed | EvalJobState::Failed,
        ) => {
            // Terminal — DELETE means "remove from tracking + archive".
            {
                let mut jobs = state.eval_jobs.write().unwrap();
                jobs.remove(&job_id);
            }
            let archive_path = crate::eval_history::archive_dir(&state.adapter_dir)
                .join(format!("{job_id}.json"));
            let removed_file = match std::fs::remove_file(&archive_path) {
                Ok(_) => true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
                Err(e) => {
                    return Err(ApiError::internal(format!(
                        "failed to delete archive file {}: {}",
                        archive_path.display(),
                        e
                    )));
                }
            };
            Ok(Json(serde_json::json!({
                "status": "deleted",
                "job_id": job_id,
                "removed_archive_file": removed_file,
            })))
        }
    }
}

// ── Datasets ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct DatasetListResponse {
    datasets: Vec<DatasetManifest>,
}

async fn list_datasets(State(state): State<AppState>) -> Json<DatasetListResponse> {
    let datasets = state
        .dataset_registry
        .as_ref()
        .map(|r| r.list())
        .unwrap_or_default();
    Json(DatasetListResponse { datasets })
}

async fn get_dataset(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<DatasetManifest>, ApiError> {
    let reg = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let m = reg
        .load_manifest(&name)
        .map_err(|_| ApiError::dataset_not_found(&name))?;
    Ok(Json(m))
}

/// `GET /v1/eval/datasets/:name/rows?limit=N` — returns the first `limit`
/// rows of an uploaded SFT dataset as a JSON array. Used by the Training
/// tab's dataset picker so users can train directly off an uploaded
/// dataset without round-tripping through copy-paste.
#[derive(Debug, Deserialize)]
struct DatasetRowsQuery {
    #[serde(default = "default_rows_limit")]
    limit: usize,
}
fn default_rows_limit() -> usize {
    100
}

async fn dataset_rows(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Query(q): Query<DatasetRowsQuery>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let reg = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let limit = q.limit.clamp(1, 5000);
    let convs = reg.head_sft(&name, limit).map_err(|e| match e {
        crate::eval::DatasetError::NotFound(_) => ApiError::dataset_not_found(&name),
        crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(&name),
        other => ApiError::dataset_invalid(format!("{other}")),
    })?;
    let out: Vec<serde_json::Value> = convs
        .into_iter()
        .map(|c| serde_json::to_value(&c).unwrap_or(serde_json::json!({})))
        .collect();
    Ok(Json(out))
}

async fn delete_dataset(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let reg = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    reg.delete(&name).map_err(|e| match e {
        crate::eval::DatasetError::NotFound(_) => ApiError::dataset_not_found(&name),
        crate::eval::DatasetError::InvalidName(n) => ApiError::dataset_invalid(n),
        other => ApiError::dataset_invalid(format!("{other}")),
    })?;
    Ok(Json(serde_json::json!({"status": "deleted", "name": name})))
}

/// `POST /v1/eval/datasets/upload` — multipart with fields `name`,
/// `format` (`sft_chat` | `grpo_groups` | `raw`), optional `description`,
/// and `file`.
async fn upload_dataset(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<DatasetManifest>, ApiError> {
    let reg = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let mut name: Option<String> = None;
    let mut format_str: Option<String> = None;
    let mut description: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::dataset_invalid(format!("multipart: {e}")))?
    {
        let field_name = match field.name() {
            Some(n) => n.to_string(),
            None => continue,
        };
        match field_name.as_str() {
            "name" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| ApiError::dataset_invalid(format!("name: {e}")))?;
                name = Some(v.trim().to_string());
            }
            "format" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| ApiError::dataset_invalid(format!("format: {e}")))?;
                format_str = Some(v.trim().to_string());
            }
            "description" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| ApiError::dataset_invalid(format!("description: {e}")))?;
                if !v.trim().is_empty() {
                    description = Some(v);
                }
            }
            "file" => {
                let mut buf = Vec::new();
                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|e| ApiError::dataset_invalid(format!("file chunk: {e}")))?
                {
                    if buf.len() + chunk.len() > crate::eval::datasets::DATASET_MAX_BYTES as usize {
                        return Err(ApiError::dataset_invalid("upload exceeds 1 GiB cap"));
                    }
                    buf.extend_from_slice(&chunk);
                }
                file_bytes = Some(buf);
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let name = name.ok_or_else(|| ApiError::dataset_invalid("missing `name` field"))?;
    let file_bytes = file_bytes.ok_or_else(|| ApiError::dataset_invalid("missing `file` field"))?;
    let format = match format_str.as_deref().unwrap_or("sft_chat") {
        "sft_chat" | "sft" => DatasetFormat::SftChat,
        "grpo_groups" | "grpo" => DatasetFormat::GrpoGroups,
        "raw" => DatasetFormat::Raw,
        other => return Err(ApiError::dataset_invalid(format!("unknown format `{other}`"))),
    };
    let manifest = reg.create(&name, format, description, &file_bytes).map_err(|e| {
        match e {
            crate::eval::DatasetError::AlreadyExists(_) => ApiError::dataset_exists(&name),
            crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(&name),
            crate::eval::DatasetError::QuotaExceeded(m) => ApiError::dataset_invalid(m),
            other => ApiError::dataset_invalid(format!("{other}")),
        }
    })?;
    Ok(Json(manifest))
}

#[derive(Debug, Deserialize)]
struct SynthesisPreviewBody {
    #[serde(flatten)]
    config: SynthesisConfig,
    /// How many conversations to scan from the head when previewing.
    /// Defaults to 5 — enough to render a quick "what'll come out" panel.
    #[serde(default = "default_preview_head")]
    head_n: usize,
}

fn default_preview_head() -> usize {
    5
}

async fn synthesize_dataset_preview(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<SynthesisPreviewBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let datasets = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let preview = preview_synthesis(datasets, &name, &body.config, body.head_n).map_err(|e| {
        match e {
            crate::eval::SynthesisDriverError::Dataset(crate::eval::DatasetError::NotFound(_)) => {
                ApiError::dataset_not_found(&name)
            }
            other => ApiError::dataset_invalid(format!("{other}")),
        }
    })?;
    Ok(Json(serde_json::to_value(&preview).unwrap_or_default()))
}

#[derive(Debug, Deserialize)]
struct SynthesizeBody {
    #[serde(flatten)]
    config: SynthesisConfig,
    /// Overwrite an existing suite by the same name.
    #[serde(default)]
    force: bool,
    /// When set, immediately enqueue an eval against this adapter as
    /// well — turning a 3-click flow into a 1-click flow.
    #[serde(default)]
    run_against: Option<Vec<String>>,
}

async fn synthesize_dataset(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<SynthesizeBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let datasets = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let suites = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    if datasets.load_manifest(&name).is_err() {
        return Err(ApiError::dataset_not_found(&name));
    }
    let outcome = synthesize_and_save(datasets, suites, &name, &body.config, body.force)
        .map_err(|e| ApiError::dataset_invalid(format!("{e}")))?;
    // Optional auto-run: queue eval jobs against each requested adapter.
    let queued_jobs: Vec<String> = body
        .run_against
        .unwrap_or_default()
        .into_iter()
        .map(|adapter| {
            let adapter_opt = if adapter.is_empty() { None } else { Some(adapter) };
            state.enqueue_eval(
                outcome.suite.name.clone(),
                vec![adapter_opt.clone()],
                EvalSubmissionKind::OnDemand,
                None,
                QueuedEvalJob::Registered {
                    suite_name: outcome.suite.name.clone(),
                    adapter: adapter_opt,
                    generation_override: None,
                },
            )
        })
        .collect();
    Ok(Json(serde_json::json!({
        "suite": outcome.suite,
        "stats": outcome.stats,
        "queued_eval_job_ids": queued_jobs,
    })))
}

// ── Judgments ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct JudgmentListResponse {
    judgments: Vec<JudgmentManifest>,
}

async fn list_judgments(State(state): State<AppState>) -> Json<JudgmentListResponse> {
    let judgments = state
        .judgment_store
        .as_ref()
        .map(|s| s.list())
        .unwrap_or_default();
    Json(JudgmentListResponse { judgments })
}

#[derive(Debug, Deserialize)]
struct CreateJudgmentBody {
    name: String,
    #[serde(default)]
    description: Option<String>,
}

async fn create_judgment_dataset(
    State(state): State<AppState>,
    Json(body): Json<CreateJudgmentBody>,
) -> Result<Json<JudgmentManifest>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    let m = store.create(&body.name, body.description).map_err(|e| match e {
        crate::eval::JudgmentError::AlreadyExists(_) => ApiError::dataset_exists(&body.name),
        crate::eval::JudgmentError::InvalidName(n) => ApiError::dataset_invalid(n),
        other => ApiError::judgment_invalid(format!("{other}")),
    })?;
    Ok(Json(m))
}

#[derive(Debug, Deserialize)]
struct AppendJudgmentBody {
    #[serde(default)]
    id: Option<String>,
    prompt: Vec<JudgmentMessage>,
    #[serde(default)]
    adapter_a: Option<String>,
    #[serde(default)]
    adapter_b: Option<String>,
    response_a: String,
    response_b: String,
    winner: JudgmentWinner,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

async fn append_judgment(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<AppendJudgmentBody>,
) -> Result<Json<JudgmentManifest>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    let row = JudgmentRow {
        id: body.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        prompt: body.prompt,
        adapter_a: body.adapter_a.filter(|s| !s.is_empty()),
        adapter_b: body.adapter_b.filter(|s| !s.is_empty()),
        response_a: body.response_a,
        response_b: body.response_b,
        winner: body.winner,
        note: body.note,
        tags: body.tags,
        submitted_at: chrono::Utc::now().to_rfc3339(),
    };
    let m = store.append(&name, &row).map_err(|e| match e {
        crate::eval::JudgmentError::NotFound(_) => ApiError::judgment_not_found(&name),
        crate::eval::JudgmentError::InvalidName(n) => ApiError::dataset_invalid(n),
        other => ApiError::judgment_invalid(format!("{other}")),
    })?;
    Ok(Json(m))
}

async fn remove_judgment_row(
    State(state): State<AppState>,
    AxumPath((name, judgment_id)): AxumPath<(String, String)>,
) -> Result<Json<JudgmentManifest>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    let m = store.remove(&name, &judgment_id).map_err(|e| match e {
        crate::eval::JudgmentError::NotFound(_) => ApiError::judgment_not_found(&name),
        crate::eval::JudgmentError::InvalidName(n) => ApiError::dataset_invalid(n),
        other => ApiError::judgment_invalid(format!("{other}")),
    })?;
    Ok(Json(m))
}

async fn delete_judgment_dataset(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    store.delete(&name).map_err(|e| match e {
        crate::eval::JudgmentError::NotFound(_) => ApiError::judgment_not_found(&name),
        crate::eval::JudgmentError::InvalidName(n) => ApiError::dataset_invalid(n),
        other => ApiError::judgment_invalid(format!("{other}")),
    })?;
    Ok(Json(serde_json::json!({"status": "deleted", "name": name})))
}

#[derive(Debug, Deserialize)]
struct CompileJudgmentBody {
    /// Name of the SFT dataset that will be created.
    output_dataset: String,
    /// Include `skip` judgments in the compilation. Off by default.
    #[serde(default)]
    include_skips: bool,
}

async fn compile_judgment(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<CompileJudgmentBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    let datasets = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let compiled = compile_judgments_to_sft(store, datasets, &name, &body.output_dataset, body.include_skips)
        .map_err(|e| ApiError::judgment_invalid(format!("{e}")))?;
    let manifest = datasets
        .load_manifest(&body.output_dataset)
        .map_err(|e| ApiError::judgment_invalid(format!("{e}")))?;
    Ok(Json(serde_json::json!({
        "status": "compiled",
        "rows": compiled,
        "dataset": manifest,
    })))
}

#[derive(Debug, Deserialize)]
struct PromoteJudgmentBody {
    /// Adapter to validate (the just-trained judge LoRA).
    adapter: String,
    /// How many of the most-recent judgments to use as the validation set.
    #[serde(default = "default_holdout")]
    holdout_n: usize,
}

fn default_holdout() -> usize {
    20
}

async fn validate_judgment_adapter(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<PromoteJudgmentBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state
        .judgment_store
        .as_ref()
        .ok_or_else(ApiError::judgment_store_unavailable)?;
    let suite = build_validation_suite(store, &name, body.holdout_n)
        .map_err(|e| ApiError::judgment_invalid(format!("{e}")))?;
    let suite_name = suite.name.clone();
    let job_id = state.enqueue_eval(
        suite_name,
        vec![Some(body.adapter.clone())],
        EvalSubmissionKind::OnDemand,
        None,
        QueuedEvalJob::Inline {
            suite: Box::new(suite),
            adapter: Some(body.adapter),
            generation_override: None,
        },
    );
    Ok(Json(serde_json::json!({
        "status": "queued",
        "eval_job_id": job_id,
        "validation_suite": format!("judge-validate-{name}"),
    })))
}

/// `POST /v1/judgments/render_prompt` — given a candidate prompt + two
/// replies, render the judging prompt string the way the SFT compiler
/// would. Lets the UI display the exact text that the judge LoRA will
/// see, before the user commits a judgment.
async fn render_judgment_prompt(
    Json(body): Json<AppendJudgmentBody>,
) -> Json<serde_json::Value> {
    let row = JudgmentRow {
        id: body.id.unwrap_or_default(),
        prompt: body.prompt,
        adapter_a: body.adapter_a,
        adapter_b: body.adapter_b,
        response_a: body.response_a,
        response_b: body.response_b,
        winner: body.winner,
        note: body.note,
        tags: body.tags,
        submitted_at: String::new(),
    };
    Json(serde_json::json!({"prompt": format_judge_prompt(&row)}))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/eval/suites", get(list_suites))
        .route(
            "/v1/eval/suites",
            post(save_suite).layer(DefaultBodyLimit::max(SUITE_BODY_LIMIT)),
        )
        .route("/v1/eval/suites/{name}", get(get_suite).delete(delete_suite))
        .route(
            "/v1/eval/run",
            post(submit_eval).layer(DefaultBodyLimit::max(EVAL_BODY_LIMIT)),
        )
        .route(
            "/v1/eval/compare",
            post(submit_compare).layer(DefaultBodyLimit::max(EVAL_BODY_LIMIT)),
        )
        .route("/v1/eval/jobs", get(list_jobs))
        .route("/v1/eval/jobs/{job_id}", get(get_job).delete(cancel_job))
        .route("/v1/eval/jobs/{job_id}/rerun", post(rerun_job))
        // Datasets
        .route("/v1/eval/datasets", get(list_datasets))
        .route(
            "/v1/eval/datasets/upload",
            post(upload_dataset).layer(DefaultBodyLimit::max(DATASET_BODY_LIMIT)),
        )
        .route(
            "/v1/eval/datasets/{name}",
            get(get_dataset).delete(delete_dataset),
        )
        .route("/v1/eval/datasets/{name}/rows", get(dataset_rows))
        .route(
            "/v1/eval/datasets/{name}/preview",
            post(synthesize_dataset_preview).layer(DefaultBodyLimit::max(EVAL_BODY_LIMIT)),
        )
        .route(
            "/v1/eval/datasets/{name}/synthesize",
            post(synthesize_dataset).layer(DefaultBodyLimit::max(EVAL_BODY_LIMIT)),
        )
        // Judgments — the flywheel
        .route("/v1/judgments", get(list_judgments).post(create_judgment_dataset))
        .route("/v1/judgments/render_prompt", post(render_judgment_prompt))
        .route(
            "/v1/judgments/{name}",
            delete(delete_judgment_dataset),
        )
        .route("/v1/judgments/{name}/rows", post(append_judgment))
        .route(
            "/v1/judgments/{name}/rows/{judgment_id}",
            delete(remove_judgment_row),
        )
        .route("/v1/judgments/{name}/compile", post(compile_judgment))
        .route("/v1/judgments/{name}/validate", post(validate_judgment_adapter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::test_tokenizer;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use kiln_core::config::ModelConfig;
    use kiln_eval::scorers::{NumericTolerance, Scorer};
    use kiln_eval::{EvalChatMessage, EvalExample};
    use tower::ServiceExt;

    fn mk_state() -> AppState {
        let config = ModelConfig::qwen3_5_4b();
        let sched_config = kiln_scheduler::SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        AppState::new_mock(
            config,
            scheduler,
            std::sync::Arc::new(engine),
            test_tokenizer(),
            60,
            "test-model".into(),
        )
    }

    fn mk_state_with_registry(dir: &std::path::Path) -> AppState {
        let mut state = mk_state();
        state.suite_registry = Some(std::sync::Arc::new(crate::eval::SuiteRegistry::new(
            dir.to_path_buf(),
        )));
        state
    }

    fn mk_inline_suite() -> EvalSuite {
        EvalSuite {
            name: "inline-math".into(),
            description: None,
            default_scorer: Scorer::NumericTolerance(NumericTolerance {
                atol: 0.0,
                rtol: 0.0,
                integer_only: true,
            }),
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("e1".into()),
                messages: vec![EvalChatMessage::new("user", "1+1?")],
                target: Some("2".into()),
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    #[tokio::test]
    async fn inline_suite_submission_returns_job_id() {
        let state = mk_state();
        let router = routes().with_state(state.clone());
        let body = serde_json::json!({
            "inline_suite": mk_inline_suite(),
        });
        let res = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(res.into_body(), 1 << 16).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(resp["job_id"].as_str().is_some());
        assert_eq!(resp["state"], "queued");
        // Tracked in the map.
        assert_eq!(state.eval_jobs.read().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn run_rejects_both_suite_and_inline() {
        let state = mk_state();
        let router = routes().with_state(state);
        let body = serde_json::json!({
            "suite": "foo",
            "inline_suite": mk_inline_suite(),
        });
        let res = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn run_rejects_neither_suite_nor_inline() {
        let state = mk_state();
        let router = routes().with_state(state);
        let res = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn registered_suite_run_fails_without_registry_configured() {
        let state = mk_state();
        let router = routes().with_state(state);
        let body = serde_json::json!({
            "suite": "foo",
        });
        let res = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn list_jobs_empty_then_one_after_submit() {
        let state = mk_state();
        let router = routes().with_state(state.clone());
        let res = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/eval/jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = serde_json::json!({"inline_suite": mk_inline_suite()});
        let _ = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let res = router
            .oneshot(
                Request::builder()
                    .uri("/v1/eval/jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp["jobs"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cancel_queued_job_marks_cancelled() {
        let state = mk_state();
        let router = routes().with_state(state.clone());
        let body = serde_json::json!({"inline_suite": mk_inline_suite()});
        let res = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/run")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let job_id = v["job_id"].as_str().unwrap().to_string();

        let res = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/eval/jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "cancelled");
    }
}

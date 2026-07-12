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
    /// Optional job-level seed. Unlike a full generation override, this does
    /// not replace the suite's temperature/top-p/max-token settings.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Suite-wide generation override. Per-example overrides on the suite
    /// itself still win over this.
    #[serde(default)]
    pub generation: Option<EvalGenerationParams>,
}

#[derive(Debug, Serialize)]
pub struct EvalRunResponse {
    pub job_id: String,
    pub state: EvalJobState,
    #[serde(with = "kiln_eval::result::u64_decimal")]
    pub effective_seed: u64,
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

fn map_eval_enqueue_error(state: &AppState, error: anyhow::Error) -> ApiError {
    if state.ensure_inference_admission_allowed().is_err() {
        ApiError::inference_disabled_by_profile(state.serving_profile.profile())
    } else {
        ApiError::internal(error)
    }
}

async fn list_suites(State(state): State<AppState>) -> Result<Json<SuiteListResponse>, ApiError> {
    let Some(reg) = state.suite_registry.as_ref() else {
        // No registry = no suites, but we still return 200 with an empty list
        // so dashboards have something to render in mock mode.
        return Ok(Json(SuiteListResponse { suites: vec![] }));
    };
    Ok(Json(SuiteListResponse { suites: reg.list() }))
}

async fn get_suite(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<EvalSuite>, ApiError> {
    let reg = state
        .suite_registry
        .as_ref()
        .ok_or_else(ApiError::eval_registry_unavailable)?;
    let suite = reg.load(&name).map_err(|e| match e {
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
    let requested_job_seed = req.seed;
    let adapter = req
        .adapter
        .and_then(|s| if s.is_empty() { None } else { Some(s) });
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
        suite.validate().map_err(|error| {
            ApiError::eval_invalid_request(format!("invalid inline suite: {error}"))
        })?;
        let suite_name = suite.name.clone();
        let queued = QueuedEvalJob::Inline {
            suite: Box::new(suite),
            adapter,
            generation_override: req.generation,
        };
        (suite_name, queued)
    };

    let enqueued = match requested_job_seed {
        Some(seed) => state.enqueue_eval_with_effective_seed(
            suite_name.clone(),
            adapters.clone(),
            EvalSubmissionKind::OnDemand,
            None,
            queued_job,
            seed,
        ),
        None => state.enqueue_eval(
            suite_name.clone(),
            adapters,
            EvalSubmissionKind::OnDemand,
            None,
            queued_job,
        ),
    }
    .map_err(|error| map_eval_enqueue_error(&state, error))?;
    tracing::info!(
        job_id = %enqueued.job_id,
        effective_seed = enqueued.effective_seed,
        suite = %suite_name,
        "eval job queued"
    );
    Ok(Json(EvalRunResponse {
        job_id: enqueued.job_id,
        state: EvalJobState::Queued,
        effective_seed: enqueued.effective_seed,
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
    let enqueued = state
        .enqueue_eval(
            suite_name.clone(),
            adapters,
            EvalSubmissionKind::Compare,
            None,
            QueuedEvalJob::Compare(spec),
        )
        .map_err(|error| map_eval_enqueue_error(&state, error))?;
    Ok(Json(EvalRunResponse {
        job_id: enqueued.job_id,
        state: EvalJobState::Queued,
        effective_seed: enqueued.effective_seed,
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
    /// Override the original job seed. Omitted reuses it exactly.
    #[serde(default)]
    seed: Option<u64>,
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
    let (suite_name, adapter, example_ids, original_effective_seed) = {
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
        (
            suite_name,
            adapter,
            ids.into_iter().collect::<Vec<_>>(),
            job.effective_seed,
        )
    };
    if example_ids.is_empty() {
        return Err(ApiError::eval_invalid_request(
            "no matching outcomes to re-run",
        ));
    }
    let mut inline = crate::eval::rerun_filtered_suite(suites, &suite_name, &example_ids).map_err(
        |e| match e {
            crate::eval::rerun::RerunError::Registry(
                crate::eval::SuiteRegistryError::NotFound(_),
            ) => ApiError::eval_suite_not_found(&suite_name),
            other => ApiError::eval_invalid_request(format!("{other}")),
        },
    )?;
    if let Some(seed) = body.seed.or(original_effective_seed) {
        inline.generation.seed = Some(seed);
    }
    let suite_label = inline.name.clone();
    let enqueued = state
        .enqueue_eval(
            suite_label.clone(),
            vec![adapter.clone()],
            EvalSubmissionKind::OnDemand,
            None,
            QueuedEvalJob::Inline {
                suite: Box::new(inline),
                adapter,
                generation_override: None,
            },
        )
        .map_err(|error| map_eval_enqueue_error(&state, error))?;
    Ok(Json(EvalRunResponse {
        job_id: enqueued.job_id,
        state: EvalJobState::Queued,
        effective_seed: enqueued.effective_seed,
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
            // Cooperative cancellation: flip the executor's flag (checked at
            // example boundaries) and mark the tracked state. The worker
            // preserves Cancelled when the run returns, archiving whatever
            // partial outcomes completed.
            let mut jobs = state.eval_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = EvalJobState::Cancelled;
                if let Some(flag) = job.cancel_flag.as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            Ok(Json(serde_json::json!({
                "status": "cancelling",
                "job_id": job_id,
                "note": "running job will exit at the next example boundary",
            })))
        }
        Some(EvalJobState::Cancelled | EvalJobState::Completed | EvalJobState::Failed) => {
            // Terminal — DELETE means "remove from tracking + archive".
            {
                let mut jobs = state.eval_jobs.write().unwrap();
                jobs.remove(&job_id);
            }
            let archive_path =
                crate::eval_history::archive_dir(&state.adapter_dir).join(format!("{job_id}.json"));
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
        other => {
            return Err(ApiError::dataset_invalid(format!(
                "unknown format `{other}`"
            )));
        }
    };
    let file_bytes = normalize_dataset_upload(format, file_bytes)?;
    let manifest = reg
        .create(&name, format, description, &file_bytes)
        .map_err(|e| match e {
            crate::eval::DatasetError::AlreadyExists(_) => ApiError::dataset_exists(&name),
            crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(&name),
            other => ApiError::dataset_invalid(format!("{other}")),
        })?;
    Ok(Json(manifest))
}

fn normalize_dataset_upload(
    format: DatasetFormat,
    file_bytes: Vec<u8>,
) -> Result<Vec<u8>, ApiError> {
    if matches!(format, DatasetFormat::Raw) {
        return Ok(file_bytes);
    }
    let first = file_bytes
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace());
    if first != Some(b'[') {
        return Ok(file_bytes);
    }
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&file_bytes)
        .map_err(|e| ApiError::dataset_invalid(format!("dataset JSON array: {e}")))?;
    let mut jsonl = Vec::with_capacity(file_bytes.len());
    for row in rows {
        serde_json::to_writer(&mut jsonl, &row)
            .map_err(|e| ApiError::dataset_invalid(format!("dataset JSONL encode: {e}")))?;
        jsonl.push(b'\n');
    }
    Ok(jsonl)
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
    let preview =
        preview_synthesis(datasets, &name, &body.config, body.head_n).map_err(|e| match e {
            crate::eval::SynthesisDriverError::Dataset(crate::eval::DatasetError::NotFound(_)) => {
                ApiError::dataset_not_found(&name)
            }
            other => ApiError::dataset_invalid(format!("{other}")),
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
            let adapter_opt = if adapter.is_empty() {
                None
            } else {
                Some(adapter)
            };
            state
                .enqueue_eval(
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
                .map(|enqueued| enqueued.job_id)
                .map_err(|error| map_eval_enqueue_error(&state, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
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
    let m = store
        .create(&body.name, body.description)
        .map_err(|e| match e {
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

/// Response for `POST /v1/judgments/{name}/rows`. Additive over the bare
/// manifest the endpoint used to return: `judgment_id` is the id assigned
/// to the row that was just appended, so the UI can offer Undo via
/// `DELETE /v1/judgments/{name}/rows/{judgment_id}` without re-reading the
/// dataset. Every pre-existing manifest field stays at the top level.
#[derive(Debug, Serialize)]
struct AppendJudgmentResponse {
    judgment_id: String,
    #[serde(flatten)]
    manifest: JudgmentManifest,
}

async fn append_judgment(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(body): Json<AppendJudgmentBody>,
) -> Result<Json<AppendJudgmentResponse>, ApiError> {
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
    Ok(Json(AppendJudgmentResponse {
        judgment_id: row.id,
        manifest: m,
    }))
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
    /// Most-recent rows to EXCLUDE from compilation so `validate` scores
    /// the judge on data it never trained on. Omitted → automatic:
    /// `min(20, rows/5)`, so bootstrap datasets (a handful of picks)
    /// still compile fully while grown datasets get a real holdout.
    #[serde(default)]
    holdout_n: Option<usize>,
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
    let total_rows = store
        .load_manifest(&name)
        .map(|m| m.num_rows as usize)
        .unwrap_or(0);
    let holdout_n = body
        .holdout_n
        .unwrap_or_else(|| (total_rows / 5).min(default_holdout()));
    let mut warnings: Vec<String> = Vec::new();
    if holdout_n == 0 {
        warnings.push(
            "no holdout reserved — validation against this dataset will score the judge \
             on its own training rows until you have more judgments"
                .to_string(),
        );
    }
    let (compiled, split) = compile_judgments_to_sft(
        store,
        datasets,
        &name,
        &body.output_dataset,
        body.include_skips,
        holdout_n,
    )
    .map_err(|e| ApiError::judgment_invalid(format!("{e}")))?;
    let manifest = datasets
        .load_manifest(&body.output_dataset)
        .map_err(|e| ApiError::judgment_invalid(format!("{e}")))?;
    Ok(Json(serde_json::json!({
        "status": "compiled",
        "rows": compiled,
        "holdout_n": holdout_n,
        "train_validation_split": split,
        "dataset": manifest,
        "warnings": warnings,
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
    // Validation rows are the last `holdout_n`; compilation trained on
    // rows [0, split). They overlap when total - holdout_n < split — the
    // accuracy would then be partly measured on training data.
    let mut warnings: Vec<String> = Vec::new();
    if let Ok(manifest) = store.load_manifest(&name) {
        if let Some(split) = manifest.last_compiled_split {
            let total = manifest.num_rows;
            if total.saturating_sub(body.holdout_n as u64) < split {
                warnings.push(format!(
                    "validation overlaps the compiled training set: last compile trained on \
                     rows [0, {split}) of {total}, validation uses the last {} — recompile \
                     with holdout_n >= {} or validate with fewer rows",
                    body.holdout_n, body.holdout_n
                ));
            }
        } else {
            warnings.push(
                "judgments were never compiled with a holdout split — if the judge trained \
                 on this dataset, validation accuracy includes training rows"
                    .to_string(),
            );
        }
    }
    let suite_name = suite.name.clone();
    let enqueued = state
        .enqueue_eval(
            suite_name,
            vec![Some(body.adapter.clone())],
            EvalSubmissionKind::OnDemand,
            None,
            QueuedEvalJob::Inline {
                suite: Box::new(suite),
                adapter: Some(body.adapter),
                generation_override: None,
            },
        )
        .map_err(|error| map_eval_enqueue_error(&state, error))?;
    Ok(Json(serde_json::json!({
        "status": "queued",
        "eval_job_id": enqueued.job_id,
        "effective_seed": enqueued.effective_seed.to_string(),
        "validation_suite": format!("judge-validate-{name}"),
        "warnings": warnings,
    })))
}

/// `POST /v1/judgments/render_prompt` — given a candidate prompt + two
/// replies, render the judging prompt string the way the SFT compiler
/// would. Lets the UI display the exact text that the judge LoRA will
/// see, before the user commits a judgment.
async fn render_judgment_prompt(Json(body): Json<AppendJudgmentBody>) -> Json<serde_json::Value> {
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
        .route(
            "/v1/eval/suites/{name}",
            get(get_suite).delete(delete_suite),
        )
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
            post(upload_dataset).layer(DefaultBodyLimit::disable()),
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
        .route(
            "/v1/judgments",
            get(list_judgments).post(create_judgment_dataset),
        )
        .route("/v1/judgments/render_prompt", post(render_judgment_prompt))
        .route("/v1/judgments/{name}", delete(delete_judgment_dataset))
        .route("/v1/judgments/{name}/rows", post(append_judgment))
        .route(
            "/v1/judgments/{name}/rows/{judgment_id}",
            delete(remove_judgment_row),
        )
        .route("/v1/judgments/{name}/compile", post(compile_judgment))
        .route(
            "/v1/judgments/{name}/validate",
            post(validate_judgment_adapter),
        )
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
        let body_bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(resp["job_id"].as_str().is_some());
        assert_eq!(resp["state"], "queued");
        // Tracked in the map.
        assert_eq!(state.eval_jobs.read().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn maintenance_profile_rejects_eval_before_queue_publication() {
        let mut state = mk_state();
        state.serving_profile = crate::config::ServingProfileSetting::new(
            crate::config::ServingProfile::Maintenance,
            crate::config::ConfigValueSource::ConfigFile,
        );
        let router = routes().with_state(state.clone());
        let body = serde_json::json!({"inline_suite": mk_inline_suite()});
        let response = router
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
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{json}");
        assert_eq!(json["error"]["code"], "inference_disabled_by_profile");
        assert!(state.eval_jobs.read().unwrap().is_empty());
        assert!(state.eval_queue.lock().unwrap().is_empty());
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
        let submitted = router
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
        let submitted = axum::body::to_bytes(submitted.into_body(), 1 << 16)
            .await
            .unwrap();
        let submitted: serde_json::Value = serde_json::from_slice(&submitted).unwrap();
        let effective_seed = submitted["effective_seed"]
            .as_str()
            .and_then(|value| value.parse::<u64>().ok())
            .expect("submission must materialize an effective seed");
        let res = router
            .oneshot(
                Request::builder()
                    .uri("/v1/eval/jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp["jobs"].as_array().unwrap().len(), 1);
        assert_eq!(
            resp["jobs"][0]["effective_seed"],
            effective_seed.to_string()
        );
    }

    #[tokio::test]
    async fn explicit_eval_seed_is_immutable_across_response_tracking_and_queue() {
        let state = mk_state();
        let router = routes().with_state(state.clone());
        let body = serde_json::json!({
            "inline_suite": mk_inline_suite(),
            "seed": u64::MAX,
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
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["effective_seed"], u64::MAX.to_string());
        let job_id = response["job_id"].as_str().unwrap();
        assert_eq!(
            state
                .eval_jobs
                .read()
                .unwrap()
                .get(job_id)
                .unwrap()
                .effective_seed,
            Some(u64::MAX)
        );
        let entry = state.eval_queue.lock().unwrap().pop().unwrap();
        assert_eq!(entry.job_id, job_id);
        assert_eq!(entry.effective_seed, u64::MAX);
    }

    #[tokio::test]
    async fn compare_inherits_suite_seed_and_accepts_job_only_override() {
        let temp = tempfile::tempdir().unwrap();
        let state = mk_state_with_registry(temp.path());
        let mut suite = mk_inline_suite();
        suite.generation.seed = Some(444);
        state
            .suite_registry
            .as_ref()
            .unwrap()
            .save(&suite, false)
            .unwrap();
        let router = routes().with_state(state.clone());

        for (body, expected) in [
            (
                serde_json::json!({
                    "suite": suite.name.clone(),
                    "adapters": ["", "candidate"]
                }),
                444u64,
            ),
            (
                serde_json::json!({
                    "suite": suite.name.clone(),
                    "adapters": ["", "candidate"],
                    "generation": {
                        "temperature": 0.25,
                        "top_p": 0.9,
                        "top_k": 20,
                        "max_tokens": 32,
                        "n": 1
                    }
                }),
                444u64,
            ),
            (
                serde_json::json!({
                    "suite": suite.name.clone(),
                    "adapters": ["", "candidate"],
                    "seed": 555u64
                }),
                555u64,
            ),
        ] {
            let res = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/eval/compare")
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
                .await
                .unwrap();
            let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(response["effective_seed"], expected.to_string());
        }
        let queue = &mut state.eval_queue.lock().unwrap();
        assert_eq!(queue.pop().unwrap().effective_seed, 444);
        assert_eq!(queue.pop().unwrap().effective_seed, 444);
        assert_eq!(queue.pop().unwrap().effective_seed, 555);
    }

    #[tokio::test]
    async fn filtered_rerun_reuses_the_original_effective_seed() {
        let temp = tempfile::tempdir().unwrap();
        let state = mk_state_with_registry(temp.path());
        let suite = mk_inline_suite();
        state
            .suite_registry
            .as_ref()
            .unwrap()
            .save(&suite, false)
            .unwrap();
        let mut outcome = kiln_eval::score_completion(
            &suite.default_scorer,
            &suite.examples[0],
            "3",
            &kiln_eval::scorers::NoopJudgeRunner,
        )
        .unwrap();
        outcome.generation_seed = Some(kiln_eval::derive_eval_completion_seed(777, "e1", 0));
        let mut original = EvalJobInfo::queued(
            "original".into(),
            suite.name.clone(),
            vec![None],
            EvalSubmissionKind::OnDemand,
            None,
            777,
        );
        original.state = EvalJobState::Completed;
        original.finished_runs.push(kiln_eval::SuiteResult {
            suite_name: suite.name.clone(),
            adapter: None,
            metrics: kiln_eval::AggregateMetrics::default(),
            outcomes: vec![outcome],
            started_at: "2026-07-10T00:00:00Z".into(),
            finished_at: "2026-07-10T00:00:01Z".into(),
            suite_hash: "suite".into(),
            effective_generation_hash: "generation".into(),
        });
        state
            .eval_jobs
            .write()
            .unwrap()
            .insert("original".into(), original);

        let res = routes()
            .with_state(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/eval/jobs/original/rerun")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["effective_seed"], "777");
        assert_eq!(
            state
                .eval_queue
                .lock()
                .unwrap()
                .pop()
                .unwrap()
                .effective_seed,
            777
        );
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
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
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
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "cancelled");
    }
}

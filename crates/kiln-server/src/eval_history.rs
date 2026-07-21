//! On-disk archive for terminal eval jobs.
//!
//! Mirrors `training_history`: terminal (Completed / Failed / Cancelled)
//! eval jobs are written as one JSON file per job under
//! `<adapter_dir>/.kiln-jobs/eval/<job_id>.json`. The archive is loaded at
//! server startup so the /ui Evals tab still shows past runs across
//! restarts.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::eval::queue::EvalJobInfo;
use kiln_eval::EvalJobState;

const SUBDIR: &str = ".kiln-jobs/eval";

/// Hard cap on archived terminal eval jobs retained on disk. Past this,
/// oldest entries are deleted first.
pub const MAX_ARCHIVED_JOBS: usize = 1024;

pub fn archive_dir(adapter_dir: &Path) -> PathBuf {
    adapter_dir.join(SUBDIR)
}

fn job_path(adapter_dir: &Path, job_id: &str) -> PathBuf {
    archive_dir(adapter_dir).join(format!("{job_id}.json"))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_artifact_provenance(job: &EvalJobInfo, require_replay_records: bool) -> io::Result<()> {
    if job.schema_version != kiln_eval::EVAL_RESULT_SCHEMA_VERSION {
        return Err(invalid_data(format!(
            "unsupported archived eval result schema_version {}; expected {}; legacy multi-completion results are ambiguous and must be rerun",
            job.schema_version,
            kiln_eval::EVAL_RESULT_SCHEMA_VERSION
        )));
    }
    if let Some(manifest) = job.base_weight_shard_manifest.as_ref() {
        manifest
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    }
    if let Some(provenance) = job.execution_provenance.as_ref() {
        provenance
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    }
    for (run_index, run) in job.finished_runs.iter().enumerate() {
        match run.replay_record.as_ref() {
            Some(record) => {
                record.validate(&run.outcomes).map_err(|error| {
                    invalid_data(format!(
                        "eval run {run_index} has invalid replay evidence: {error}"
                    ))
                })?;
                if run.suite_hash != record.suite_sha256
                    || run.effective_generation_hash != record.effective_generation_sha256
                {
                    return Err(invalid_data(format!(
                        "eval run {run_index} hashes do not match its replay record"
                    )));
                }
            }
            None if require_replay_records => {
                return Err(invalid_data(format!(
                    "new eval archive run {run_index} is missing kiln.eval-replay.v1 evidence"
                )));
            }
            None => {}
        }
    }
    match (&job.replay_expectation, &job.replay_verdict) {
        (Some(expectation), Some(verdict)) => {
            verdict.validate(expectation).map_err(|error| {
                invalid_data(format!("eval replay verdict is invalid: {error}"))
            })?;
            match (job.state, verdict.status) {
                (EvalJobState::Completed, kiln_eval::EvalReplayStatus::Matched)
                | (EvalJobState::Completed, kiln_eval::EvalReplayStatus::Mismatch)
                | (EvalJobState::Failed, kiln_eval::EvalReplayStatus::Error)
                | (EvalJobState::Cancelled, kiln_eval::EvalReplayStatus::Error) => {}
                (state, status) => {
                    return Err(invalid_data(format!(
                        "eval replay state {state:?} is inconsistent with verdict {status:?}"
                    )));
                }
            }
        }
        (Some(expectation), None) => {
            expectation.validate().map_err(|error| {
                invalid_data(format!("eval replay expectation is invalid: {error}"))
            })?;
            if job.state != EvalJobState::Cancelled {
                return Err(invalid_data(
                    "terminal replay archive is missing its replay verdict",
                ));
            }
        }
        (None, Some(_)) => {
            return Err(invalid_data(
                "eval replay verdict is present without an expectation",
            ));
        }
        (None, None) => {}
    }
    Ok(())
}

fn decode_archive(bytes: &[u8]) -> io::Result<EvalJobInfo> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if version != Some(kiln_eval::EVAL_RESULT_SCHEMA_VERSION as u64) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported archived eval result schema_version {}; expected {}; legacy multi-completion results are ambiguous and must be rerun",
                version
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "<missing>".to_string()),
                kiln_eval::EVAL_RESULT_SCHEMA_VERSION
            ),
        ));
    }
    serde_json::from_value(value).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn save(adapter_dir: &Path, job: &EvalJobInfo) -> io::Result<()> {
    if !matches!(
        job.state,
        EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled
    ) {
        return Ok(());
    }
    validate_artifact_provenance(job, true)?;
    let dir = archive_dir(adapter_dir);
    fs::create_dir_all(&dir)?;
    let path = job_path(adapter_dir, &job.job_id);
    let tmp = path.with_extension("json.tmp");
    let payload = serde_json::to_vec_pretty(job)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&tmp, payload)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn load_all(adapter_dir: &Path) -> Vec<EvalJobInfo> {
    let dir = archive_dir(adapter_dir);
    let entries = match fs::read_dir(&dir) {
        Ok(it) => it,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(error = %e, dir = %dir.display(), "failed to read eval archive dir");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match fs::read(&path).and_then(|b| {
            let job = decode_archive(&b)?;
            validate_artifact_provenance(&job, false)?;
            Ok(job)
        }) {
            Ok(job) => out.push(job),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    file = %path.display(),
                    "failed to load archived eval job — skipping"
                );
            }
        }
    }
    out
}

pub fn prune_to_max(adapter_dir: &Path, max: usize) {
    let dir = archive_dir(adapter_dir);
    let entries: Vec<(PathBuf, SystemTime)> = match fs::read_dir(&dir) {
        Ok(it) => it
            .flatten()
            .filter_map(|e| {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("json") {
                    return None;
                }
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((path, mtime))
            })
            .collect(),
        Err(_) => return,
    };
    if entries.len() <= max {
        return;
    }
    let mut sorted = entries;
    sorted.sort_by_key(|(_, mtime)| *mtime);
    let to_delete = sorted.len() - max;
    for (path, _) in sorted.into_iter().take(to_delete) {
        if let Err(e) = fs::remove_file(&path) {
            tracing::warn!(error = %e, file = %path.display(), "failed to prune archived eval job");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::queue::EvalSubmissionKind;
    use kiln_eval::scorers::Scorer;

    fn base_weight_manifest() -> kiln_core::model_provenance::BaseWeightShardManifest {
        kiln_core::model_provenance::BaseWeightShardManifest::new(vec![
            kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                "model.safetensors",
                11,
                [0x42; 32],
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn completed_job() -> EvalJobInfo {
        let mut job = EvalJobInfo::queued(
            "eval-provenance".to_string(),
            "suite".to_string(),
            vec![None],
            EvalSubmissionKind::OnDemand,
            None,
            17,
        );
        job.state = EvalJobState::Completed;
        job.base_weight_shard_manifest = Some(base_weight_manifest());
        job.execution_provenance = Some(crate::execution_provenance::test_execution_provenance());
        job
    }

    fn completed_job_with_replay() -> EvalJobInfo {
        let mut job = completed_job();
        let suite = kiln_eval::EvalSuite {
            name: "replay-suite".into(),
            description: None,
            default_scorer: Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: false,
            },
            generation: kiln_eval::EvalGenerationParams::default(),
            aggregation: kiln_eval::EvalAggregation::Single,
            system_prompt: None,
            examples: vec![kiln_eval::EvalExample {
                id: Some("one".into()),
                messages: vec![kiln_eval::EvalChatMessage::new("user", "say x")],
                target: Some("x".into()),
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        };
        let mut outcome = kiln_eval::score_completion(
            &suite.default_scorer,
            &suite.examples[0],
            "x",
            &kiln_eval::scorers::NoopJudgeRunner,
        )
        .unwrap();
        outcome.generation_seed = Some(kiln_eval::derive_eval_completion_seed(17, "one", 0));
        outcome.raw_completion_text = Some("x".into());
        outcome.thinking_budget = Some(kiln_eval::EvalThinkingBudget::default());
        let record = kiln_eval::EvalReplayRecordV1::new(
            suite.clone(),
            None,
            17,
            vec![kiln_eval::EvalThinkingBudget::default()],
            Some(kiln_eval::EvalModelTargetIdentity::base()),
            Vec::new(),
            Some(
                job.execution_provenance
                    .as_ref()
                    .unwrap()
                    .provenance_sha256
                    .clone(),
            ),
            Some(
                job.base_weight_shard_manifest
                    .as_ref()
                    .unwrap()
                    .aggregate_sha256
                    .clone(),
            ),
            std::slice::from_ref(&outcome),
        )
        .unwrap();
        let aggregated_outcomes = kiln_eval::aggregate_example_outcomes(
            std::slice::from_ref(&outcome),
            suite.aggregation,
        )
        .unwrap();
        job.finished_runs.push(kiln_eval::SuiteResult {
            suite_name: suite.name,
            adapter: None,
            aggregation: suite.aggregation,
            metrics: kiln_eval::AggregateMetrics::default(),
            outcomes: vec![outcome],
            aggregated_outcomes,
            started_at: "2026-07-20T00:00:00Z".into(),
            finished_at: "2026-07-20T00:00:01Z".into(),
            suite_hash: record.suite_sha256.clone(),
            effective_generation_hash: record.effective_generation_sha256.clone(),
            replay_record: Some(record),
        });
        job
    }

    #[test]
    fn eval_archive_round_trips_and_validates_artifact_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let job = completed_job();
        save(temp.path(), &job).unwrap();

        let loaded = load_all(temp.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].base_weight_shard_manifest,
            job.base_weight_shard_manifest
        );
        assert_eq!(loaded[0].execution_provenance, job.execution_provenance);

        let path = job_path(temp.path(), &job.job_id);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["base_weight_shard_manifest"]["total_size_bytes"] = serde_json::json!(12);
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(load_all(temp.path()).is_empty());

        save(temp.path(), &job).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["execution_provenance"]["backend"]["device"] = serde_json::json!("tampered:0");
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(load_all(temp.path()).is_empty());
    }

    #[test]
    fn eval_archive_refuses_to_save_tampered_execution_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let mut job = completed_job();
        job.execution_provenance.as_mut().unwrap().backend.device = "tampered:0".to_string();
        let error = save(temp.path(), &job).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn eval_archive_rejects_tampered_raw_completion_replay_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let job = completed_job_with_replay();
        save(temp.path(), &job).unwrap();
        let path = job_path(temp.path(), &job.job_id);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["finished_runs"][0]["outcomes"][0]["raw_completion_text"] =
            serde_json::json!("tampered");
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(load_all(temp.path()).is_empty());
    }

    #[test]
    fn eval_archive_requires_replay_evidence_for_every_new_finished_run() {
        let temp = tempfile::tempdir().unwrap();
        let mut job = completed_job_with_replay();
        job.finished_runs[0].replay_record = None;
        let error = save(temp.path(), &job).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("missing kiln.eval-replay.v1"));
    }

    #[test]
    fn eval_archive_rejects_missing_and_legacy_result_versions() {
        let temp = tempfile::tempdir().unwrap();
        let job = completed_job();
        save(temp.path(), &job).unwrap();
        let path = job_path(temp.path(), &job.job_id);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();

        value.as_object_mut().unwrap().remove("schema_version");
        let error = decode_archive(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("schema_version <missing>"));

        value["schema_version"] = serde_json::json!(1);
        let error = decode_archive(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("schema_version 1"));
        assert!(error.to_string().contains("ambiguous"));
    }
}

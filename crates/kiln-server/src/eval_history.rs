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

fn validate_base_weight_provenance(job: &EvalJobInfo) -> io::Result<()> {
    if let Some(manifest) = job.base_weight_shard_manifest.as_ref() {
        manifest
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    }
    Ok(())
}

pub fn save(adapter_dir: &Path, job: &EvalJobInfo) -> io::Result<()> {
    if !matches!(
        job.state,
        EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled
    ) {
        return Ok(());
    }
    validate_base_weight_provenance(job)?;
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
            let job: EvalJobInfo = serde_json::from_slice(&b)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            validate_base_weight_provenance(&job)?;
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
        job
    }

    #[test]
    fn eval_archive_round_trips_and_validates_base_weight_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let job = completed_job();
        save(temp.path(), &job).unwrap();

        let loaded = load_all(temp.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].base_weight_shard_manifest,
            job.base_weight_shard_manifest
        );

        let path = job_path(temp.path(), &job.job_id);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["base_weight_shard_manifest"]["total_size_bytes"] = serde_json::json!(12);
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(load_all(temp.path()).is_empty());
    }
}

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

pub fn save(adapter_dir: &Path, job: &EvalJobInfo) -> io::Result<()> {
    if !matches!(
        job.state,
        EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled
    ) {
        return Ok(());
    }
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
            serde_json::from_slice::<EvalJobInfo>(&b)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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

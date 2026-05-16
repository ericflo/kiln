//! On-disk archive for terminal training jobs.
//!
//! The in-memory `training_jobs` map is bounded by `max_tracked_jobs` and
//! TTL-evicted by the queue worker, so without persistence the dashboard
//! "forgets" past runs as soon as TTL expires (or the server restarts).
//! This module mirrors every terminal (`Completed` / `Failed`) job into
//! `<adapter_dir>/.kiln-jobs/training/<job_id>.json` so the queue panel
//! survives restarts and long-running sessions.
//!
//! Format: one JSON file per job. Filename is the job id with a `.json`
//! suffix. Mtime tracks the wall-clock terminal-transition time so the
//! oldest-first eviction can use the cheap `metadata().modified()` instead
//! of opening every file.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::state::TrainingJobInfo;

const SUBDIR: &str = ".kiln-jobs/training";

/// Hard cap on archived terminal training jobs retained on disk. Past this,
/// oldest entries are deleted first.
pub const MAX_ARCHIVED_JOBS: usize = 1024;

pub fn archive_dir(adapter_dir: &Path) -> PathBuf {
    adapter_dir.join(SUBDIR)
}

fn job_path(adapter_dir: &Path, job_id: &str) -> PathBuf {
    archive_dir(adapter_dir).join(format!("{job_id}.json"))
}

/// Persist a terminal training job to disk. No-op for non-terminal jobs —
/// the worker GC handles eviction of live entries.
///
/// Atomic write via tempfile + rename; never partially-written files. Errors
/// are returned to the caller for logging — never panic in the hot path.
pub fn save(adapter_dir: &Path, job: &TrainingJobInfo) -> io::Result<()> {
    if !matches!(
        job.state,
        kiln_train::TrainingState::Completed | kiln_train::TrainingState::Failed
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

/// Load every archived terminal training job from disk. Silently skips
/// individual files that fail to parse (a malformed entry from a previous
/// schema shouldn't take the server down on startup).
pub fn load_all(adapter_dir: &Path) -> Vec<TrainingJobInfo> {
    let dir = archive_dir(adapter_dir);
    let entries = match fs::read_dir(&dir) {
        Ok(it) => it,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(error = %e, dir = %dir.display(), "failed to read training archive dir");
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
            serde_json::from_slice::<TrainingJobInfo>(&b)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }) {
            Ok(job) => out.push(job),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    file = %path.display(),
                    "failed to load archived training job — skipping"
                );
            }
        }
    }
    out
}

/// Delete oldest archived jobs (by mtime) until at most `max` remain.
/// Best-effort: errors are logged but not propagated, so a single
/// unreadable file doesn't stop pruning the rest.
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
    sorted.sort_by_key(|(_, mtime)| *mtime); // oldest first
    let to_delete = sorted.len() - max;
    for (path, _) in sorted.into_iter().take(to_delete) {
        if let Err(e) = fs::remove_file(&path) {
            tracing::warn!(error = %e, file = %path.display(), "failed to prune archived training job");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{TrainingJobInfo, TrainingJobType};
    use kiln_train::TrainingState;

    fn make_job(id: &str, state: TrainingState) -> TrainingJobInfo {
        TrainingJobInfo {
            job_id: id.into(),
            adapter_name: format!("adapter-{id}"),
            job_type: TrainingJobType::Sft,
            state,
            progress: 1.0,
            loss: Some(0.5),
            epoch: Some(1),
            adapter_path: None,
            submitted_at: std::time::Instant::now(),
            submitted_unix_ms: 1_000,
            auto_load: false,
            finished_at: None,
            finished_unix_ms: Some(2_000),
            linked_eval_job_ids: vec![],
            loss_history: vec![],
        }
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let job = make_job("abc", TrainingState::Completed);
        save(tmp.path(), &job).unwrap();
        let loaded = load_all(tmp.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].job_id, "abc");
        assert_eq!(loaded[0].adapter_name, "adapter-abc");
        assert_eq!(loaded[0].submitted_unix_ms, 1_000);
        assert_eq!(loaded[0].finished_unix_ms, Some(2_000));
    }

    #[test]
    fn save_skips_non_terminal_jobs() {
        let tmp = tempfile::tempdir().unwrap();
        save(tmp.path(), &make_job("x", TrainingState::Queued)).unwrap();
        save(tmp.path(), &make_job("y", TrainingState::Running)).unwrap();
        assert!(load_all(tmp.path()).is_empty());
    }

    #[test]
    fn prune_keeps_newest() {
        let tmp = tempfile::tempdir().unwrap();
        for id in &["a", "b", "c", "d", "e"] {
            save(tmp.path(), &make_job(id, TrainingState::Completed)).unwrap();
            // Force mtime ordering so prune picks "older" deterministically.
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        prune_to_max(tmp.path(), 3);
        let remaining: Vec<String> = load_all(tmp.path())
            .into_iter()
            .map(|j| j.job_id)
            .collect();
        assert_eq!(remaining.len(), 3);
        // The first two ("a", "b") should have been pruned as oldest.
        assert!(!remaining.contains(&"a".to_string()));
        assert!(!remaining.contains(&"b".to_string()));
    }

    #[test]
    fn load_missing_dir_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let nonexistent = tmp.path().join("nothing");
        assert!(load_all(&nonexistent).is_empty());
    }
}

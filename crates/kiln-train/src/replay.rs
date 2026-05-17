//! Deterministic replay storage for LoRA adapters.
//!
//! Every adapter directory written by SFT or GRPO carries enough state to be
//! re-trained from scratch:
//!
//! - `replay.jsonl`: one record per accepted training request, atomically
//!   appended *before* the optimizer step runs. A second `outcome` record is
//!   appended after the step completes (or fails). The two-record split means
//!   a crash mid-step still leaves a recoverable trail of what was attempted.
//! - `lineage.json`: parent-LoRA pointer, base-model identity, kiln commit,
//!   and a content-addressed `replay_hash` derived from the parent hash plus
//!   every request record in this adapter's `replay.jsonl`.
//!
//! Together these let `kiln-replay` walk the parent chain, re-apply each
//! recorded request against the base model, and verify reproducibility.
//!
//! ## Atomic append guarantee
//!
//! `ReplayLog::append_request` opens `replay.jsonl` with `O_APPEND` (the
//! POSIX guarantee that small writes are atomic when smaller than `PIPE_BUF`,
//! typically 4 KiB) and calls `sync_data` before returning. Larger records
//! still serialize fine but lose the strict atomicity-under-concurrent-write
//! guarantee — for our use case there is only one writer (the GPU write lock
//! serializes training jobs), so even a multi-page request record is durable
//! before the optimizer step runs.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// File names written into every adapter directory.
pub const REPLAY_LOG_FILE: &str = "replay.jsonl";
pub const LINEAGE_FILE: &str = "lineage.json";

/// Current schema version for `lineage.json`. Bump when the on-disk shape
/// changes in an incompatible way.
pub const LINEAGE_SCHEMA_VERSION: u32 = 1;

/// Identity of the base model the adapter was trained against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaseModel {
    /// HuggingFace-style identifier, e.g. "Qwen/Qwen3.5-4B".
    pub id: String,
    /// HuggingFace revision (commit/branch). `None` if unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Optional digest of the model config (hidden_size, num_layers, etc.) so
    /// replay can detect mismatched architectures even when `id` matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_digest: Option<String>,
}

/// Pointer to the parent LoRA when the adapter was trained on top of another.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParentLora {
    /// Logical adapter name (matches the parent directory name).
    pub name: String,
    /// Parent's `replay_hash`. Required; this is what binds the chain.
    pub replay_hash: String,
}

/// On-disk lineage manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Lineage {
    /// Schema version — see [`LINEAGE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Logical name of this adapter (matches its directory name).
    pub adapter_name: String,
    /// Base model identity.
    pub base_model: BaseModel,
    /// Optional pointer to a parent LoRA. `None` when training from the base.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_lora: Option<ParentLora>,
    /// kiln crate version (or build-stamped commit) at training time.
    pub kiln_commit: String,
    /// RFC3339 timestamp when training started.
    pub created_at: String,
    /// Content-addressed hash binding the parent hash, base model, and every
    /// request record in this adapter's `replay.jsonl`.
    pub replay_hash: String,
}

/// Training kind tag for replay records.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReplayKind {
    Sft,
    Grpo,
}

/// Outcome state for a request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeStatus {
    Completed,
    Failed,
}

/// Either record that may appear on a `replay.jsonl` line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ReplayRecord {
    Request(RequestRecord),
    Outcome(OutcomeRecord),
}

/// A request record captured before the optimizer step runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestRecord {
    /// UUID matching the training job id.
    pub request_id: String,
    /// Training kind.
    pub kind: ReplayKind,
    /// Verbatim request body as deserialized from the HTTP submission.
    pub request_body: serde_json::Value,
    /// Effective seed for this request. Always populated, even when the user
    /// did not supply one (we generate and record one so replay is exact).
    pub seed: u64,
    /// kiln crate version (or build-stamped commit).
    pub kiln_commit: String,
    /// RFC3339 timestamp when the record was appended.
    pub submitted_at: String,
}

/// Outcome record appended after the optimizer step finishes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutcomeRecord {
    /// Matches the corresponding `RequestRecord::request_id`.
    pub request_id: String,
    pub status: OutcomeStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_secs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Append-only writer over an adapter directory's `replay.jsonl`.
///
/// Cheap to construct; opens the file lazily on each append so concurrent
/// readers (including parent-chain hashing) see flushed data.
pub struct ReplayLog {
    dir: PathBuf,
}

impl ReplayLog {
    /// Wrap an existing adapter directory. Creates the directory if missing.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating replay dir: {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn log_path(&self) -> PathBuf {
        self.dir.join(REPLAY_LOG_FILE)
    }

    /// Append a record line atomically (single `write_all` of one line ending
    /// in `\n`) and `fsync(data)` before returning.
    fn append_line(&self, line: &str) -> Result<()> {
        let path = self.log_path();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening replay log: {}", path.display()))?;

        let mut buf = String::with_capacity(line.len() + 1);
        buf.push_str(line);
        buf.push('\n');

        file.write_all(buf.as_bytes())
            .with_context(|| format!("appending to replay log: {}", path.display()))?;
        file.sync_data()
            .with_context(|| format!("fsync replay log: {}", path.display()))?;
        Ok(())
    }

    /// Append a request record. Must be called *before* the optimizer step
    /// runs so a crash mid-step still leaves the request on disk.
    pub fn append_request(&self, record: &RequestRecord) -> Result<()> {
        let line = serde_json::to_string(&ReplayRecord::Request(record.clone()))
            .context("serializing replay request record")?;
        self.append_line(&line)
    }

    /// Append an outcome record after the optimizer step finishes.
    pub fn append_outcome(&self, record: &OutcomeRecord) -> Result<()> {
        let line = serde_json::to_string(&ReplayRecord::Outcome(record.clone()))
            .context("serializing replay outcome record")?;
        self.append_line(&line)
    }
}

/// Read all records from an adapter's `replay.jsonl`. Missing log => empty.
pub fn read_replay_log(adapter_dir: &Path) -> Result<Vec<ReplayRecord>> {
    let path = adapter_dir.join(REPLAY_LOG_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut buf = String::new();
    File::open(&path)
        .with_context(|| format!("opening replay log: {}", path.display()))?
        .read_to_string(&mut buf)
        .with_context(|| format!("reading replay log: {}", path.display()))?;
    let mut out = Vec::new();
    for (lineno, line) in buf.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: ReplayRecord = serde_json::from_str(line).with_context(|| {
            format!(
                "parsing replay log line {} of {}",
                lineno + 1,
                path.display()
            )
        })?;
        out.push(rec);
    }
    Ok(out)
}

/// Write `lineage.json` atomically (write to temp file, fsync, rename).
pub fn write_lineage(adapter_dir: &Path, lineage: &Lineage) -> Result<()> {
    std::fs::create_dir_all(adapter_dir)
        .with_context(|| format!("creating adapter dir: {}", adapter_dir.display()))?;
    let final_path = adapter_dir.join(LINEAGE_FILE);
    let tmp_path = adapter_dir.join(format!("{LINEAGE_FILE}.tmp"));
    let body = serde_json::to_string_pretty(lineage).context("serializing lineage")?;
    {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("creating tmp lineage: {}", tmp_path.display()))?;
        f.write_all(body.as_bytes())
            .with_context(|| format!("writing tmp lineage: {}", tmp_path.display()))?;
        f.sync_data()
            .with_context(|| format!("fsync tmp lineage: {}", tmp_path.display()))?;
    }
    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "renaming {} -> {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

/// Read `lineage.json` for an adapter directory.
pub fn read_lineage(adapter_dir: &Path) -> Result<Lineage> {
    let path = adapter_dir.join(LINEAGE_FILE);
    let mut buf = String::new();
    File::open(&path)
        .with_context(|| format!("opening lineage: {}", path.display()))?
        .read_to_string(&mut buf)
        .with_context(|| format!("reading lineage: {}", path.display()))?;
    serde_json::from_str(&buf).with_context(|| format!("parsing lineage: {}", path.display()))
}

/// Compute the content-addressed replay hash for an adapter.
///
/// Hash inputs (in this order, separated by NUL bytes):
///
/// 1. `parent_replay_hash` (or empty string if root)
/// 2. `base_model.id`
/// 3. `base_model.revision` (or empty string)
/// 4. `base_model.config_digest` (or empty string)
/// 5. JSON-canonicalized bytes of every `RequestRecord` in chronological
///    order (one per line), joined with `\n`.
///
/// Outcome records are intentionally excluded so that a successful re-run
/// produces the same hash as the original (failures don't change identity).
pub fn compute_replay_hash(
    parent_replay_hash: Option<&str>,
    base: &BaseModel,
    requests: &[&RequestRecord],
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(parent_replay_hash.unwrap_or("").as_bytes());
    hasher.update([0u8]);
    hasher.update(base.id.as_bytes());
    hasher.update([0u8]);
    hasher.update(base.revision.as_deref().unwrap_or("").as_bytes());
    hasher.update([0u8]);
    hasher.update(base.config_digest.as_deref().unwrap_or("").as_bytes());
    hasher.update([0u8]);
    for req in requests {
        let line =
            serde_json::to_string(req).context("canonicalizing request record for replay hash")?;
        hasher.update(line.as_bytes());
        hasher.update([b'\n']);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Walk the parent chain from `adapter_dir` up toward the root, returning
/// `(adapter_dir, lineage)` pairs ordered root-first.
///
/// `parent_dir_resolver` maps a parent's logical name to its directory path
/// (typically `|name| adapter_root.join(name)`).
pub fn walk_parent_chain(
    adapter_dir: &Path,
    parent_dir_resolver: impl Fn(&str) -> PathBuf,
) -> Result<Vec<(PathBuf, Lineage)>> {
    let mut chain = Vec::new();
    let mut current = adapter_dir.to_path_buf();
    let mut visited = std::collections::HashSet::new();
    loop {
        if !visited.insert(current.clone()) {
            anyhow::bail!("cycle detected in parent chain at {}", current.display());
        }
        let lineage = read_lineage(&current)
            .with_context(|| format!("reading lineage for {}", current.display()))?;
        let parent_name = lineage.parent_lora.as_ref().map(|p| p.name.clone());
        chain.push((current.clone(), lineage));
        match parent_name {
            None => break,
            Some(name) => {
                current = parent_dir_resolver(&name);
            }
        }
    }
    chain.reverse();
    Ok(chain)
}

/// Verify that an adapter's `replay_hash` matches the recomputed hash from
/// `replay.jsonl` + parent. Does not load the model.
pub fn verify_chain_integrity(
    adapter_dir: &Path,
    parent_dir_resolver: impl Fn(&str) -> PathBuf,
) -> Result<()> {
    let chain = walk_parent_chain(adapter_dir, &parent_dir_resolver)?;
    let mut prev_hash: Option<String> = None;
    for (dir, lineage) in &chain {
        // Verify the parent pointer matches the previously-walked hash.
        match (&lineage.parent_lora, &prev_hash) {
            (None, None) => {}
            (Some(p), Some(prev)) => {
                if &p.replay_hash != prev {
                    anyhow::bail!(
                        "parent_lora.replay_hash mismatch at {}: lineage says {} but parent computed {}",
                        dir.display(),
                        p.replay_hash,
                        prev
                    );
                }
            }
            (Some(p), None) => anyhow::bail!(
                "lineage at {} declares parent {} but chain walk hit no predecessor",
                dir.display(),
                p.name
            ),
            (None, Some(_)) => anyhow::bail!(
                "lineage at {} has no parent but chain walk produced one",
                dir.display()
            ),
        }

        let records = read_replay_log(dir)?;
        let requests: Vec<&RequestRecord> = records
            .iter()
            .filter_map(|r| match r {
                ReplayRecord::Request(req) => Some(req),
                ReplayRecord::Outcome(_) => None,
            })
            .collect();
        let parent_hash = lineage.parent_lora.as_ref().map(|p| p.replay_hash.as_str());
        let expected = compute_replay_hash(parent_hash, &lineage.base_model, &requests)?;
        if expected != lineage.replay_hash {
            anyhow::bail!(
                "replay_hash mismatch at {}: lineage says {} but recomputed {}",
                dir.display(),
                lineage.replay_hash,
                expected
            );
        }
        prev_hash = Some(lineage.replay_hash.clone());
    }
    Ok(())
}

/// kiln version stamped into request records and lineage. Single-source so
/// changing it later (e.g. wiring vergen for git sha) is one-liner.
pub fn kiln_commit() -> String {
    if let Ok(v) = std::env::var("KILN_COMMIT") {
        if !v.is_empty() {
            return v;
        }
    }
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_request(id: &str, kind: ReplayKind, seed: u64) -> RequestRecord {
        RequestRecord {
            request_id: id.to_string(),
            kind,
            request_body: serde_json::json!({
                "examples": [{"messages": [{"role": "user", "content": "hi"}]}],
                "config": {"epochs": 1}
            }),
            seed,
            kiln_commit: "test-0.0.0".to_string(),
            submitted_at: "2026-05-09T00:00:00Z".to_string(),
        }
    }

    fn sample_outcome(id: &str, status: OutcomeStatus) -> OutcomeRecord {
        OutcomeRecord {
            request_id: id.to_string(),
            status,
            final_loss: Some(0.42),
            elapsed_secs: Some(1.0),
            error: None,
        }
    }

    fn sample_base() -> BaseModel {
        BaseModel {
            id: "Qwen/Qwen3.5-4B".to_string(),
            revision: Some("main".to_string()),
            config_digest: Some("digest-deadbeef".to_string()),
        }
    }

    #[test]
    fn append_request_and_read_back() {
        let tmp = TempDir::new().unwrap();
        let log = ReplayLog::new(tmp.path()).unwrap();
        let req = sample_request("req-1", ReplayKind::Sft, 7);
        log.append_request(&req).unwrap();

        let records = read_replay_log(tmp.path()).unwrap();
        assert_eq!(records.len(), 1);
        match &records[0] {
            ReplayRecord::Request(r) => assert_eq!(r, &req),
            _ => panic!("expected request record"),
        }
    }

    #[test]
    fn append_request_and_outcome_in_order() {
        let tmp = TempDir::new().unwrap();
        let log = ReplayLog::new(tmp.path()).unwrap();
        log.append_request(&sample_request("a", ReplayKind::Sft, 1))
            .unwrap();
        log.append_outcome(&sample_outcome("a", OutcomeStatus::Completed))
            .unwrap();
        log.append_request(&sample_request("b", ReplayKind::Grpo, 2))
            .unwrap();
        log.append_outcome(&sample_outcome("b", OutcomeStatus::Failed))
            .unwrap();
        let records = read_replay_log(tmp.path()).unwrap();
        assert_eq!(records.len(), 4);
        assert!(matches!(records[0], ReplayRecord::Request(_)));
        assert!(matches!(records[1], ReplayRecord::Outcome(_)));
        assert!(matches!(records[2], ReplayRecord::Request(_)));
        assert!(matches!(records[3], ReplayRecord::Outcome(_)));
    }

    #[test]
    fn append_survives_simulated_panic_between_records() {
        // Simulate a panic between the request append and the optimizer step
        // by simply not appending an outcome. The request record must still
        // be on disk and parseable so a follow-up replay can detect the
        // missing outcome.
        let tmp = TempDir::new().unwrap();
        let log = ReplayLog::new(tmp.path()).unwrap();
        let req = sample_request("interrupted", ReplayKind::Sft, 99);
        log.append_request(&req).unwrap();
        // Drop the writer (closes the file) — emulates process exit.
        drop(log);

        let records = read_replay_log(tmp.path()).unwrap();
        assert_eq!(records.len(), 1);
        match &records[0] {
            ReplayRecord::Request(r) => assert_eq!(r.request_id, "interrupted"),
            _ => panic!("expected request"),
        }
    }

    #[test]
    fn lineage_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let lineage = Lineage {
            schema_version: LINEAGE_SCHEMA_VERSION,
            adapter_name: "child".to_string(),
            base_model: sample_base(),
            parent_lora: Some(ParentLora {
                name: "parent".to_string(),
                replay_hash: "abc123".to_string(),
            }),
            kiln_commit: "kiln-0.2.13".to_string(),
            created_at: "2026-05-09T00:00:00Z".to_string(),
            replay_hash: "deadbeef".to_string(),
        };
        write_lineage(tmp.path(), &lineage).unwrap();
        let read = read_lineage(tmp.path()).unwrap();
        assert_eq!(read, lineage);
    }

    #[test]
    fn replay_hash_is_deterministic_and_chains() {
        let base = sample_base();
        let r1 = sample_request("a", ReplayKind::Sft, 1);
        let r2 = sample_request("b", ReplayKind::Grpo, 2);

        let h_root = compute_replay_hash(None, &base, &[&r1]).unwrap();
        let h_root_again = compute_replay_hash(None, &base, &[&r1]).unwrap();
        assert_eq!(h_root, h_root_again, "hash must be deterministic");

        let h_child_a = compute_replay_hash(Some(&h_root), &base, &[&r2]).unwrap();
        let h_child_b = compute_replay_hash(Some(&h_root), &base, &[&r2]).unwrap();
        assert_eq!(h_child_a, h_child_b);

        // Changing the parent hash changes the child.
        let h_other = compute_replay_hash(Some("zzz"), &base, &[&r2]).unwrap();
        assert_ne!(h_child_a, h_other);

        // Changing the base model changes the hash.
        let mut base2 = base.clone();
        base2.revision = Some("other".to_string());
        let h_other_base = compute_replay_hash(None, &base2, &[&r1]).unwrap();
        assert_ne!(h_root, h_other_base);

        // Changing record order changes the hash.
        let h_swapped = compute_replay_hash(None, &base, &[&r2, &r1]).unwrap();
        let h_normal = compute_replay_hash(None, &base, &[&r1, &r2]).unwrap();
        assert_ne!(h_normal, h_swapped);
    }

    #[test]
    fn replay_hash_ignores_outcome_records() {
        // The chain hash is computed over requests only, so a successful
        // re-run that adds an outcome should not change the hash.
        let tmp = TempDir::new().unwrap();
        let log = ReplayLog::new(tmp.path()).unwrap();
        let r1 = sample_request("a", ReplayKind::Sft, 1);
        log.append_request(&r1).unwrap();
        let base = sample_base();
        let before = compute_replay_hash(None, &base, &[&r1]).unwrap();

        log.append_outcome(&sample_outcome("a", OutcomeStatus::Completed))
            .unwrap();
        let records = read_replay_log(tmp.path()).unwrap();
        let requests: Vec<&RequestRecord> = records
            .iter()
            .filter_map(|r| match r {
                ReplayRecord::Request(req) => Some(req),
                _ => None,
            })
            .collect();
        let after = compute_replay_hash(None, &base, &requests).unwrap();
        assert_eq!(before, after);
    }

    fn build_two_step_chain(root: &Path) -> (PathBuf, PathBuf, String, String) {
        // Parent adapter
        let parent_dir = root.join("parent");
        let parent_log = ReplayLog::new(&parent_dir).unwrap();
        let r1 = sample_request("p1", ReplayKind::Sft, 1);
        parent_log.append_request(&r1).unwrap();
        parent_log
            .append_outcome(&sample_outcome("p1", OutcomeStatus::Completed))
            .unwrap();
        let base = sample_base();
        let parent_hash = compute_replay_hash(None, &base, &[&r1]).unwrap();
        let parent_lineage = Lineage {
            schema_version: LINEAGE_SCHEMA_VERSION,
            adapter_name: "parent".to_string(),
            base_model: base.clone(),
            parent_lora: None,
            kiln_commit: "test".to_string(),
            created_at: "2026-05-09T00:00:00Z".to_string(),
            replay_hash: parent_hash.clone(),
        };
        write_lineage(&parent_dir, &parent_lineage).unwrap();

        // Child adapter
        let child_dir = root.join("child");
        let child_log = ReplayLog::new(&child_dir).unwrap();
        let r2 = sample_request("c1", ReplayKind::Grpo, 2);
        child_log.append_request(&r2).unwrap();
        child_log
            .append_outcome(&sample_outcome("c1", OutcomeStatus::Completed))
            .unwrap();
        let child_hash = compute_replay_hash(Some(&parent_hash), &base, &[&r2]).unwrap();
        let child_lineage = Lineage {
            schema_version: LINEAGE_SCHEMA_VERSION,
            adapter_name: "child".to_string(),
            base_model: base,
            parent_lora: Some(ParentLora {
                name: "parent".to_string(),
                replay_hash: parent_hash.clone(),
            }),
            kiln_commit: "test".to_string(),
            created_at: "2026-05-09T00:00:01Z".to_string(),
            replay_hash: child_hash.clone(),
        };
        write_lineage(&child_dir, &child_lineage).unwrap();

        (parent_dir, child_dir, parent_hash, child_hash)
    }

    #[test]
    fn walk_parent_chain_root_first() {
        let tmp = TempDir::new().unwrap();
        let (parent_dir, child_dir, _, _) = build_two_step_chain(tmp.path());
        let resolver = |name: &str| tmp.path().join(name);
        let chain = walk_parent_chain(&child_dir, resolver).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].0, parent_dir);
        assert_eq!(chain[1].0, child_dir);
        assert!(chain[0].1.parent_lora.is_none());
        assert_eq!(
            chain[1].1.parent_lora.as_ref().unwrap().name.as_str(),
            "parent"
        );
    }

    #[test]
    fn verify_chain_integrity_passes_for_valid_chain() {
        let tmp = TempDir::new().unwrap();
        let (_, child_dir, _, _) = build_two_step_chain(tmp.path());
        let resolver = |name: &str| tmp.path().join(name);
        verify_chain_integrity(&child_dir, resolver).unwrap();
    }

    #[test]
    fn verify_chain_integrity_detects_tampered_request() {
        let tmp = TempDir::new().unwrap();
        let (_, child_dir, _, _) = build_two_step_chain(tmp.path());

        // Tamper with the child's replay log by appending a fake request not
        // covered by the recorded replay_hash.
        let log = ReplayLog::new(&child_dir).unwrap();
        log.append_request(&sample_request("evil", ReplayKind::Sft, 999))
            .unwrap();

        let resolver = |name: &str| tmp.path().join(name);
        let err = verify_chain_integrity(&child_dir, resolver).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("replay_hash mismatch"), "got: {msg}");
    }

    #[test]
    fn verify_chain_integrity_detects_parent_pointer_tamper() {
        let tmp = TempDir::new().unwrap();
        let (_, child_dir, _, _) = build_two_step_chain(tmp.path());

        // Rewrite the child's lineage with a wrong parent replay_hash; the
        // child's own replay_hash still validates against its requests, but
        // the chain walk will catch the mismatch.
        let mut lineage = read_lineage(&child_dir).unwrap();
        lineage.parent_lora.as_mut().unwrap().replay_hash = "wrongparent".to_string();
        write_lineage(&child_dir, &lineage).unwrap();

        let resolver = |name: &str| tmp.path().join(name);
        let err = verify_chain_integrity(&child_dir, resolver).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("parent_lora.replay_hash mismatch"),
            "got: {msg}"
        );
    }

    #[test]
    fn replay_kind_serializes_lowercase() {
        let v = serde_json::to_value(ReplayKind::Sft).unwrap();
        assert_eq!(v, serde_json::json!("sft"));
        let v = serde_json::to_value(ReplayKind::Grpo).unwrap();
        assert_eq!(v, serde_json::json!("grpo"));
    }

    #[test]
    fn record_round_trips_through_jsonl() {
        let tmp = TempDir::new().unwrap();
        let log = ReplayLog::new(tmp.path()).unwrap();
        let req = sample_request("r", ReplayKind::Grpo, 314);
        let outcome = sample_outcome("r", OutcomeStatus::Failed);
        log.append_request(&req).unwrap();
        log.append_outcome(&outcome).unwrap();

        let records = read_replay_log(tmp.path()).unwrap();
        assert_eq!(records.len(), 2);
        match &records[0] {
            ReplayRecord::Request(r) => assert_eq!(r, &req),
            _ => panic!("first must be request"),
        }
        match &records[1] {
            ReplayRecord::Outcome(o) => assert_eq!(o, &outcome),
            _ => panic!("second must be outcome"),
        }
    }
}

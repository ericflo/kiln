//! On-disk dataset registry — SFT/GRPO JSONL files users upload to power
//! eval synthesis and the judgment flywheel.
//!
//! Layout: `<adapter_dir>/.eval/datasets/<name>/`
//! - `data.jsonl` — the uploaded JSONL, kept verbatim so re-synthesis is
//!   reproducible and users can prune individual rows later.
//! - `manifest.json` — line count, format, byte size, upload timestamp.
//!
//! All operations are append-only with the exception of `remove_row`,
//! which writes a new `data.jsonl` skipping the requested line numbers and
//! atomically renames it into place. The provenance contract (the user's
//! "remove bad data and retrain" loop) hinges on this being lossless.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use kiln_eval::synthesis::SftConversation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub name: String,
    pub format: DatasetFormat,
    #[serde(default)]
    pub description: Option<String>,
    pub num_rows: u64,
    pub size_bytes: u64,
    pub created_at: String,
    pub updated_at: String,
    /// Histogram of role patterns observed in the dataset — useful for the
    /// UI to suggest a synthesis strategy ("looks like multi-turn tool-use"
    /// → recommend EveryAssistantTurn).
    #[serde(default)]
    pub stats: DatasetStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    pub num_assistant_turns: u64,
    pub num_tool_messages: u64,
    pub num_with_tool_calls: u64,
    pub max_messages_per_conv: u32,
    pub max_content_chars: u32,
    pub avg_messages_per_conv: f64,
    /// Sample of the first few role patterns. Truncated to keep the
    /// manifest small.
    #[serde(default)]
    pub sample_role_patterns: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormat {
    /// `{"messages": [{role, content, …}, ...]}` per line.
    SftChat,
    /// GRPO group: `{"messages": [...], "completions": [{text, reward}]}`.
    GrpoGroups,
    /// User-supplied raw JSONL — we don't try to interpret beyond counting lines.
    Raw,
}

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io: {0}")]
    Io(String),
    #[error("invalid dataset name: {0}")]
    InvalidName(String),
    #[error("dataset not found: {0}")]
    NotFound(String),
    #[error("dataset already exists: {0}")]
    AlreadyExists(String),
    #[error("invalid jsonl: {0}")]
    InvalidJsonl(String),
    #[error("io quota: {0}")]
    QuotaExceeded(String),
}

/// Maximum decompressed dataset size we accept in a single upload (1 GiB).
pub const DATASET_MAX_BYTES: u64 = 1024 * 1024 * 1024;
/// Max rows analyzed during the stat pass — beyond this we sample.
const STAT_SCAN_ROWS: u64 = 1000;

pub struct DatasetRegistry {
    root: PathBuf,
}

impl DatasetRegistry {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn ensure_root(&self) -> Result<(), DatasetError> {
        std::fs::create_dir_all(&self.root).map_err(|e| DatasetError::Io(format!("{e}")))?;
        Ok(())
    }

    pub fn dataset_dir(&self, name: &str) -> Result<PathBuf, DatasetError> {
        validate_name(name)?;
        Ok(self.root.join(name))
    }

    /// Create a new dataset from JSONL bytes. Returns the manifest. Rejects
    /// names that already exist (use `replace` if you want overwrite).
    pub fn create(
        &self,
        name: &str,
        format: DatasetFormat,
        description: Option<String>,
        jsonl_bytes: &[u8],
    ) -> Result<DatasetManifest, DatasetError> {
        self.ensure_root()?;
        let dir = self.dataset_dir(name)?;
        if dir.exists() {
            return Err(DatasetError::AlreadyExists(name.to_string()));
        }
        if jsonl_bytes.len() as u64 > DATASET_MAX_BYTES {
            return Err(DatasetError::QuotaExceeded(format!(
                "dataset exceeds {} GiB limit",
                DATASET_MAX_BYTES / (1024 * 1024 * 1024)
            )));
        }
        std::fs::create_dir_all(&dir).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let path = dir.join("data.jsonl");
        // Atomic write via tmp + rename.
        let tmp = dir.join("data.jsonl.tmp");
        std::fs::write(&tmp, jsonl_bytes).map_err(|e| DatasetError::Io(format!("{e}")))?;
        std::fs::rename(&tmp, &path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        // Scan for stats + line count.
        let (num_rows, stats) = analyze_jsonl(&path, format)?;
        let now = chrono::Utc::now().to_rfc3339();
        let manifest = DatasetManifest {
            name: name.to_string(),
            format,
            description,
            num_rows,
            size_bytes: jsonl_bytes.len() as u64,
            created_at: now.clone(),
            updated_at: now,
            stats,
        };
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    /// Append one row to an existing dataset (used by the judgment flywheel
    /// when the user submits a new A/B preference). The on-disk JSONL is
    /// fsynced after the append so a crash doesn't lose the judgment.
    pub fn append_row(&self, name: &str, row_json: &str) -> Result<DatasetManifest, DatasetError> {
        let dir = self.dataset_dir(name)?;
        if !dir.exists() {
            return Err(DatasetError::NotFound(name.to_string()));
        }
        let path = dir.join("data.jsonl");
        if !path.exists() {
            return Err(DatasetError::NotFound(format!("{name}/data.jsonl")));
        }
        // Sanity-check that the row parses as JSON.
        serde_json::from_str::<serde_json::Value>(row_json)
            .map_err(|e| DatasetError::InvalidJsonl(format!("{e}")))?;

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        f.write_all(row_json.trim_end_matches('\n').as_bytes())
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        f.write_all(b"\n")
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        f.sync_all().map_err(|e| DatasetError::Io(format!("{e}")))?;
        let mut manifest = self.load_manifest(name)?;
        manifest.num_rows += 1;
        manifest.size_bytes = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(manifest.size_bytes);
        manifest.updated_at = chrono::Utc::now().to_rfc3339();
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    /// Remove a row by its 0-indexed line number. The on-disk file is
    /// rewritten atomically so callers always see a consistent state.
    /// Returns the new manifest.
    ///
    /// This is the foundation of the user-driven "remove bad data → retrain"
    /// loop. The manifest's `updated_at` advances so callers can detect when
    /// a re-train is required.
    pub fn remove_rows(
        &self,
        name: &str,
        line_numbers: &[u64],
    ) -> Result<DatasetManifest, DatasetError> {
        let dir = self.dataset_dir(name)?;
        if !dir.exists() {
            return Err(DatasetError::NotFound(name.to_string()));
        }
        let path = dir.join("data.jsonl");
        let drop_set: std::collections::HashSet<u64> = line_numbers.iter().copied().collect();
        let f = std::fs::File::open(&path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let tmp = dir.join("data.jsonl.tmp");
        let mut out = std::fs::File::create(&tmp).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let mut kept = 0u64;
        for (idx, line) in BufReader::new(f).lines().enumerate() {
            let line = line.map_err(|e| DatasetError::Io(format!("{e}")))?;
            if drop_set.contains(&(idx as u64)) {
                continue;
            }
            out.write_all(line.as_bytes())
                .map_err(|e| DatasetError::Io(format!("{e}")))?;
            out.write_all(b"\n")
                .map_err(|e| DatasetError::Io(format!("{e}")))?;
            kept += 1;
        }
        out.sync_all().map_err(|e| DatasetError::Io(format!("{e}")))?;
        drop(out);
        std::fs::rename(&tmp, &path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let mut manifest = self.load_manifest(name)?;
        manifest.num_rows = kept;
        manifest.size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        manifest.updated_at = chrono::Utc::now().to_rfc3339();
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    pub fn delete(&self, name: &str) -> Result<(), DatasetError> {
        let dir = self.dataset_dir(name)?;
        if !dir.exists() {
            return Err(DatasetError::NotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&dir).map_err(|e| DatasetError::Io(format!("{e}")))?;
        Ok(())
    }

    pub fn list(&self) -> Vec<DatasetManifest> {
        let mut out = Vec::new();
        let read_dir = match std::fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(_) => return out,
        };
        for entry in read_dir.flatten() {
            let name = match entry.file_name().to_str() {
                Some(n) if !n.starts_with('.') => n.to_string(),
                _ => continue,
            };
            if let Ok(m) = self.load_manifest(&name) {
                out.push(m);
            }
        }
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }

    pub fn load_manifest(&self, name: &str) -> Result<DatasetManifest, DatasetError> {
        let path = self.dataset_dir(name)?.join("manifest.json");
        if !path.exists() {
            return Err(DatasetError::NotFound(name.to_string()));
        }
        let bytes = std::fs::read(&path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DatasetError::InvalidJsonl(format!("manifest parse: {e}")))
    }

    fn write_manifest(&self, name: &str, manifest: &DatasetManifest) -> Result<(), DatasetError> {
        let dir = self.dataset_dir(name)?;
        let path = dir.join("manifest.json");
        let tmp = dir.join("manifest.json.tmp");
        let body = serde_json::to_string_pretty(manifest)
            .map_err(|e| DatasetError::Io(format!("manifest serialize: {e}")))?;
        std::fs::write(&tmp, body).map_err(|e| DatasetError::Io(format!("{e}")))?;
        std::fs::rename(&tmp, &path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        Ok(())
    }

    /// Stream the first `n` SFT conversations from a dataset. Used by the
    /// preview endpoint so the UI can render synthesized examples before
    /// committing.
    pub fn head_sft(&self, name: &str, n: usize) -> Result<Vec<SftConversation>, DatasetError> {
        let path = self.dataset_dir(name)?.join("data.jsonl");
        // Distinguish "missing dataset" from real I/O failures so callers
        // can map the former to a 404 without swallowing the latter.
        let f = std::fs::File::open(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => DatasetError::NotFound(name.to_string()),
            _ => DatasetError::Io(format!("{e}")),
        })?;
        let mut out = Vec::new();
        for (idx, line) in BufReader::new(f).lines().enumerate() {
            if out.len() >= n {
                break;
            }
            let line = line.map_err(|e| DatasetError::Io(format!("{e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let conv: SftConversation = serde_json::from_str(&line).map_err(|e| {
                DatasetError::InvalidJsonl(format!("line {}: {e}", idx + 1))
            })?;
            out.push(conv);
        }
        Ok(out)
    }

    /// Stream every SFT conversation from a dataset (no buffering past one
    /// line at a time). Used by the synthesizer.
    pub fn iter_sft<'a>(
        &'a self,
        name: &str,
    ) -> Result<DatasetSftIter<'a>, DatasetError> {
        let path = self.dataset_dir(name)?.join("data.jsonl");
        let f = std::fs::File::open(&path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        Ok(DatasetSftIter {
            reader: BufReader::new(f),
            _marker: std::marker::PhantomData,
        })
    }
}

/// Iterator returned by `DatasetRegistry::iter_sft`. Yields one
/// `SftConversation` per non-empty JSONL line; lines that fail to parse
/// are skipped (with a tracing::warn) so a malformed row in a 1000-line
/// dataset doesn't poison the whole synthesis.
pub struct DatasetSftIter<'a> {
    reader: BufReader<std::fs::File>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for DatasetSftIter<'a> {
    type Item = SftConversation;
    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                return None;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<SftConversation>(trimmed) {
                Ok(conv) => return Some(conv),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed dataset row");
                    continue;
                }
            }
        }
    }
}

fn validate_name(name: &str) -> Result<(), DatasetError> {
    if !crate::eval::util::is_valid_segment_name(name) {
        return Err(DatasetError::InvalidName(name.to_string()));
    }
    Ok(())
}

fn analyze_jsonl(path: &Path, format: DatasetFormat) -> Result<(u64, DatasetStats), DatasetError> {
    let f = std::fs::File::open(path).map_err(|e| DatasetError::Io(format!("{e}")))?;
    let reader = BufReader::new(f);
    let mut num_rows = 0u64;
    let mut stats = DatasetStats::default();
    let mut total_messages = 0u64;
    let mut convs_scanned = 0u64;

    for line in reader.lines() {
        let line = line.map_err(|e| DatasetError::Io(format!("{e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        num_rows += 1;
        if convs_scanned >= STAT_SCAN_ROWS {
            continue;
        }
        match format {
            DatasetFormat::Raw => continue,
            DatasetFormat::SftChat | DatasetFormat::GrpoGroups => {
                let conv: Result<SftConversation, _> = serde_json::from_str(trimmed);
                if let Ok(conv) = conv {
                    convs_scanned += 1;
                    let msgs = &conv.messages;
                    if msgs.len() as u32 > stats.max_messages_per_conv {
                        stats.max_messages_per_conv = msgs.len() as u32;
                    }
                    total_messages += msgs.len() as u64;
                    for m in msgs {
                        if m.role == "assistant" {
                            stats.num_assistant_turns += 1;
                            if m.tool_calls.as_ref().map(|v| !v.is_empty()).unwrap_or(false) {
                                stats.num_with_tool_calls += 1;
                            }
                        }
                        if m.role == "tool" {
                            stats.num_tool_messages += 1;
                        }
                        let content_chars = m.content.chars().count() as u32;
                        if content_chars > stats.max_content_chars {
                            stats.max_content_chars = content_chars;
                        }
                    }
                    if stats.sample_role_patterns.len() < 5 {
                        let pattern: Vec<&str> = msgs.iter().map(|m| m.role.as_str()).collect();
                        // Compress consecutive identical roles into shorthand,
                        // e.g. ["assistant", "tool", "tool", "tool"] becomes
                        // "assistant tool×3".
                        stats.sample_role_patterns.push(compress_pattern(&pattern));
                    }
                }
            }
        }
    }
    if convs_scanned > 0 {
        stats.avg_messages_per_conv = total_messages as f64 / convs_scanned as f64;
    }
    Ok((num_rows, stats))
}

fn compress_pattern(roles: &[&str]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < roles.len() {
        let role = roles[i];
        let mut j = i + 1;
        while j < roles.len() && roles[j] == role {
            j += 1;
        }
        let run = j - i;
        if !out.is_empty() {
            out.push(' ');
        }
        if run == 1 {
            out.push_str(role);
        } else {
            out.push_str(&format!("{role}×{run}"));
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> String {
        let r = |role: &str, content: &str| {
            serde_json::json!({"role": role, "content": content})
        };
        let convs = vec![
            serde_json::json!({"messages": [r("user", "1+1?"), r("assistant", "2")]}),
            serde_json::json!({"messages": [r("user", "hello"), r("assistant", "hi")]}),
        ];
        convs
            .into_iter()
            .map(|c| serde_json::to_string(&c).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn create_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let m = reg
            .create("smoke", DatasetFormat::SftChat, Some("desc".into()), rows().as_bytes())
            .unwrap();
        assert_eq!(m.num_rows, 2);
        assert_eq!(m.format, DatasetFormat::SftChat);
        assert!(m.stats.num_assistant_turns >= 2);
        let listed = reg.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "smoke");
    }

    #[test]
    fn duplicate_create_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        reg.create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let err = reg
            .create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap_err();
        assert!(matches!(err, DatasetError::AlreadyExists(_)));
    }

    #[test]
    fn head_sft_returns_first_n() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        reg.create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let head = reg.head_sft("a", 1).unwrap();
        assert_eq!(head.len(), 1);
        let head = reg.head_sft("a", 99).unwrap();
        assert_eq!(head.len(), 2);
    }

    #[test]
    fn iter_sft_skips_malformed_lines() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let body = format!("{}\nnot json\n{{\"messages\": []}}\n", rows());
        reg.create("a", DatasetFormat::SftChat, None, body.as_bytes())
            .unwrap();
        let convs: Vec<_> = reg.iter_sft("a").unwrap().collect();
        // 2 valid + 1 valid-but-empty messages = 3.
        assert_eq!(convs.len(), 3);
    }

    #[test]
    fn append_row_updates_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        reg.create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let new_row = r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}"#;
        let m = reg.append_row("a", new_row).unwrap();
        assert_eq!(m.num_rows, 3);
    }

    #[test]
    fn remove_rows_rewrites_file() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        reg.create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let m = reg.remove_rows("a", &[0]).unwrap();
        assert_eq!(m.num_rows, 1);
        let head = reg.head_sft("a", 99).unwrap();
        // First row's content was "hello"/"hi"; "1+1?" was at index 0.
        assert!(head.iter().all(|c| {
            c.messages.iter().all(|msg| msg.content != "1+1?")
        }));
    }

    #[test]
    fn invalid_names_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        for bad in &["", "..", "a/b", "a\\b"] {
            let err = reg.create(bad, DatasetFormat::SftChat, None, rows().as_bytes()).unwrap_err();
            assert!(matches!(err, DatasetError::InvalidName(_)));
        }
    }

    #[test]
    fn analyze_compresses_role_patterns() {
        let pattern = compress_pattern(&["assistant", "tool", "tool", "tool", "assistant"]);
        assert_eq!(pattern, "assistant tool×3 assistant");
    }

    #[test]
    fn quota_rejects_oversize_uploads() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let huge = vec![b'x'; (DATASET_MAX_BYTES + 1) as usize];
        let err = reg.create("a", DatasetFormat::Raw, None, &huge).unwrap_err();
        assert!(matches!(err, DatasetError::QuotaExceeded(_)));
    }
}

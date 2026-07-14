//! On-disk dataset registry — SFT/GRPO JSONL files users upload to power
//! eval synthesis and the judgment flywheel.
//!
//! Layout: `<adapter_dir>/.eval/datasets/<name>/`
//! - `data.jsonl` — the uploaded JSONL, kept verbatim so re-synthesis is
//!   reproducible and users can prune individual rows later.
//! - `manifest.json` — content identities, split summary, format, and timestamps.
//! - `identity.json` — one canonical and normalized content identity per row.
//! - `split.json` — deterministic group/session-aware partition assignment.
//! - `{train,validation,holdout}.jsonl` — materialized immutable-by-manifest
//!   partition views used by training and suite synthesis.
//!
//! All operations are append-only with the exception of `remove_row`,
//! which writes a new `data.jsonl` skipping the requested line numbers and
//! atomically renames it into place. The provenance contract (the user's
//! "remove bad data and retrain" loop) hinges on this being lossless.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use kiln_eval::data_identity::{
    DATASET_PROVENANCE_METADATA_KEY, DatasetExampleProvenance, DatasetIdentityIndex,
    DatasetRowIdentity, DatasetSplit, DatasetSplitConfig, DatasetSplitCounts, DatasetSplitManifest,
    build_identity_index_from_rows, build_split_manifest, row_identity, sha256_bytes,
};
use kiln_eval::synthesis::SftConversation;
use serde::{Deserialize, Serialize};

pub const DATASET_MANIFEST_SCHEMA_VERSION: u32 = 2;
const DATA_FILENAME: &str = "data.jsonl";
const IDENTITY_FILENAME: &str = "identity.json";
const SPLIT_FILENAME: &str = "split.json";

fn default_manifest_schema_version() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    #[serde(default = "default_manifest_schema_version")]
    pub schema_version: u32,
    pub name: String,
    pub format: DatasetFormat,
    #[serde(default)]
    pub description: Option<String>,
    pub num_rows: u64,
    pub size_bytes: u64,
    pub created_at: String,
    pub updated_at: String,
    /// Ordered aggregate over canonical row identities.
    #[serde(default)]
    pub corpus_sha256: String,
    /// Ordered aggregate after conservative case/whitespace normalization.
    #[serde(default)]
    pub normalized_corpus_sha256: String,
    /// Digest of the persisted typed split manifest.
    #[serde(default)]
    pub split_manifest_sha256: String,
    #[serde(default)]
    pub split_config: DatasetSplitConfig,
    #[serde(default)]
    pub split_counts: DatasetSplitCounts,
    #[serde(default)]
    pub num_groups: u64,
    #[serde(default)]
    pub num_sessions: u64,
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
}

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
        std::fs::create_dir_all(&dir).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let result = (|| {
            let path = dir.join(DATA_FILENAME);
            write_bytes_atomic(&path, jsonl_bytes)?;
            let now = chrono::Utc::now().to_rfc3339();
            self.rebuild_artifacts(
                name,
                format,
                description,
                now.clone(),
                now,
                DatasetSplitConfig::default(),
            )
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        result
    }

    /// Append one row to an existing dataset (used by the judgment flywheel
    /// when the user submits a new A/B preference). The on-disk JSONL is
    /// fsynced after the append so a crash doesn't lose the judgment.
    pub fn append_row(&self, name: &str, row_json: &str) -> Result<DatasetManifest, DatasetError> {
        let dir = self.dataset_dir(name)?;
        if !dir.exists() {
            return Err(DatasetError::NotFound(name.to_string()));
        }
        let path = dir.join(DATA_FILENAME);
        if !path.exists() {
            return Err(DatasetError::NotFound(format!("{name}/data.jsonl")));
        }
        // Sanity-check that the row parses as JSON.
        serde_json::from_str::<serde_json::Value>(row_json)
            .map_err(|e| DatasetError::InvalidJsonl(format!("{e}")))?;

        let manifest = self.load_manifest(name)?;
        let tmp = dir.join("data.jsonl.tmp");
        let mut input = std::fs::File::open(&path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let mut output =
            std::fs::File::create(&tmp).map_err(|e| DatasetError::Io(format!("{e}")))?;
        std::io::copy(&mut input, &mut output).map_err(|e| DatasetError::Io(format!("{e}")))?;
        if manifest.size_bytes > 0 {
            output
                .write_all(b"\n")
                .map_err(|e| DatasetError::Io(format!("{e}")))?;
        }
        output
            .write_all(row_json.trim_end_matches('\n').as_bytes())
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        output
            .write_all(b"\n")
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        output
            .sync_all()
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        drop(output);
        std::fs::rename(&tmp, &path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        self.rebuild_artifacts(
            name,
            manifest.format,
            manifest.description,
            manifest.created_at,
            chrono::Utc::now().to_rfc3339(),
            manifest.split_config,
        )
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
        let path = dir.join(DATA_FILENAME);
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
        out.sync_all()
            .map_err(|e| DatasetError::Io(format!("{e}")))?;
        drop(out);
        std::fs::rename(&tmp, &path).map_err(|e| DatasetError::Io(format!("{e}")))?;
        let manifest = self.load_manifest(name)?;
        debug_assert_eq!(
            kept,
            manifest.num_rows.saturating_sub(drop_set.len() as u64)
        );
        self.rebuild_artifacts(
            name,
            manifest.format,
            manifest.description,
            manifest.created_at,
            chrono::Utc::now().to_rfc3339(),
            manifest.split_config,
        )
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
        let manifest: DatasetManifest = serde_json::from_slice(&bytes)
            .map_err(|e| DatasetError::InvalidJsonl(format!("manifest parse: {e}")))?;
        if manifest.schema_version > DATASET_MANIFEST_SCHEMA_VERSION {
            return Err(DatasetError::InvalidJsonl(format!(
                "manifest schema_version {} is newer than supported version {}",
                manifest.schema_version, DATASET_MANIFEST_SCHEMA_VERSION
            )));
        }
        if manifest.schema_version < DATASET_MANIFEST_SCHEMA_VERSION
            || manifest.corpus_sha256.is_empty()
            || manifest.split_manifest_sha256.is_empty()
        {
            return self.rebuild_artifacts(
                name,
                manifest.format,
                manifest.description,
                manifest.created_at,
                chrono::Utc::now().to_rfc3339(),
                manifest.split_config,
            );
        }
        Ok(manifest)
    }

    fn write_manifest(&self, name: &str, manifest: &DatasetManifest) -> Result<(), DatasetError> {
        let dir = self.dataset_dir(name)?;
        let path = dir.join("manifest.json");
        let body = serde_json::to_vec_pretty(manifest)
            .map_err(|e| DatasetError::Io(format!("manifest serialize: {e}")))?;
        write_bytes_atomic(&path, &body)
    }

    pub fn load_identity(&self, name: &str) -> Result<DatasetIdentityIndex, DatasetError> {
        let manifest = self.load_manifest(name)?;
        let path = self.dataset_dir(name)?.join(IDENTITY_FILENAME);
        let value: DatasetIdentityIndex = read_json(&path)?;
        if value.dataset_name != name
            || value.corpus_sha256 != manifest.corpus_sha256
            || value.normalized_corpus_sha256 != manifest.normalized_corpus_sha256
        {
            return Err(DatasetError::InvalidJsonl(format!(
                "dataset identity for {name:?} does not match manifest"
            )));
        }
        Ok(value)
    }

    pub fn load_split(&self, name: &str) -> Result<DatasetSplitManifest, DatasetError> {
        let manifest = self.load_manifest(name)?;
        let path = self.dataset_dir(name)?.join(SPLIT_FILENAME);
        let value: DatasetSplitManifest = read_json(&path)?;
        let bytes = serde_json::to_vec_pretty(&value)
            .map_err(|e| DatasetError::InvalidJsonl(format!("split serialize: {e}")))?;
        if value.dataset_name != name
            || value.corpus_sha256 != manifest.corpus_sha256
            || sha256_bytes(&bytes) != manifest.split_manifest_sha256
        {
            return Err(DatasetError::InvalidJsonl(format!(
                "dataset split for {name:?} does not match manifest"
            )));
        }
        Ok(value)
    }

    pub fn set_split(
        &self,
        name: &str,
        config: DatasetSplitConfig,
    ) -> Result<DatasetSplitManifest, DatasetError> {
        config.validate().map_err(DatasetError::InvalidJsonl)?;
        let manifest = self.load_manifest(name)?;
        let updated = self.rebuild_artifacts(
            name,
            manifest.format,
            manifest.description,
            manifest.created_at,
            chrono::Utc::now().to_rfc3339(),
            config,
        )?;
        debug_assert_eq!(updated.schema_version, DATASET_MANIFEST_SCHEMA_VERSION);
        self.load_split(name)
    }

    pub fn split_path(&self, name: &str, split: DatasetSplit) -> Result<PathBuf, DatasetError> {
        let manifest = self.load_manifest(name)?;
        if manifest.split_counts.get(split) == 0 {
            return Err(DatasetError::InvalidJsonl(format!(
                "dataset {name:?} {} split is empty; choose another split or upload enough independent groups for a three-way partition",
                split.as_str()
            )));
        }
        let path = self
            .dataset_dir(name)?
            .join(format!("{}.jsonl", split.as_str()));
        if !path.is_file() {
            return Err(DatasetError::NotFound(format!(
                "{name}/{}.jsonl",
                split.as_str()
            )));
        }
        Ok(path)
    }

    /// Stream the first `n` SFT conversations from a dataset. Used by the
    /// preview endpoint so the UI can render synthesized examples before
    /// committing.
    pub fn head_sft(&self, name: &str, n: usize) -> Result<Vec<SftConversation>, DatasetError> {
        self.head_sft_path(name, None, n)
    }

    pub fn head_sft_split(
        &self,
        name: &str,
        split: DatasetSplit,
        n: usize,
    ) -> Result<Vec<SftConversation>, DatasetError> {
        self.head_sft_path(name, Some(split), n)
    }

    fn head_sft_path(
        &self,
        name: &str,
        split: Option<DatasetSplit>,
        n: usize,
    ) -> Result<Vec<SftConversation>, DatasetError> {
        let mut iter = self.sft_iter(name, split)?;
        let mut out = Vec::new();
        while out.len() < n {
            let Some(conv) = iter.next() else { break };
            out.push(conv);
        }
        Ok(out)
    }

    fn sft_iter<'a>(
        &'a self,
        name: &str,
        split: Option<DatasetSplit>,
    ) -> Result<DatasetSftIter<'a>, DatasetError> {
        let path = match split {
            Some(split) => self.split_path(name, split)?,
            None => self.dataset_dir(name)?.join(DATA_FILENAME),
        };
        // Distinguish "missing dataset" from real I/O failures so callers
        // can map the former to a 404 without swallowing the latter.
        let f = std::fs::File::open(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => DatasetError::NotFound(name.to_string()),
            _ => DatasetError::Io(format!("{e}")),
        })?;
        let mut identities = HashMap::<String, VecDeque<DatasetRowIdentity>>::new();
        if let Some(split) = split {
            for row in self
                .load_split(name)?
                .rows
                .into_iter()
                .filter(|row| row.split == split)
            {
                identities
                    .entry(row.content_sha256.clone())
                    .or_default()
                    .push_back(DatasetRowIdentity {
                        row_number: row.row_number,
                        content_sha256: row.content_sha256,
                        normalized_sha256: row.normalized_sha256,
                        group_id: row.group_id,
                        session_id: row.session_id,
                    });
            }
        }
        Ok(DatasetSftIter {
            reader: BufReader::new(f),
            dataset_name: split.map(|_| name.to_string()),
            source_split: split,
            identities,
            _marker: std::marker::PhantomData,
        })
    }

    /// Stream every SFT conversation from a dataset (no buffering past one
    /// line at a time). Used by the synthesizer.
    pub fn iter_sft<'a>(&'a self, name: &str) -> Result<DatasetSftIter<'a>, DatasetError> {
        self.sft_iter(name, None)
    }

    pub fn iter_sft_split<'a>(
        &'a self,
        name: &str,
        split: DatasetSplit,
    ) -> Result<DatasetSftIter<'a>, DatasetError> {
        self.sft_iter(name, Some(split))
    }

    fn rebuild_artifacts(
        &self,
        name: &str,
        format: DatasetFormat,
        description: Option<String>,
        created_at: String,
        updated_at: String,
        split_config: DatasetSplitConfig,
    ) -> Result<DatasetManifest, DatasetError> {
        split_config
            .validate()
            .map_err(DatasetError::InvalidJsonl)?;
        let dir = self.dataset_dir(name)?;
        let data_path = dir.join(DATA_FILENAME);
        let identities = scan_row_identities(&data_path, format)?;
        if identities.is_empty() {
            return Err(DatasetError::InvalidJsonl(
                "dataset must contain at least one non-empty JSONL row".to_string(),
            ));
        }
        let identity = build_identity_index_from_rows(name, identities);
        let split = build_split_manifest(&identity, split_config.clone())
            .map_err(DatasetError::InvalidJsonl)?;
        materialize_split_files(&data_path, &dir, &split)?;

        let identity_bytes = serde_json::to_vec_pretty(&identity)
            .map_err(|e| DatasetError::Io(format!("identity serialize: {e}")))?;
        write_bytes_atomic(&dir.join(IDENTITY_FILENAME), &identity_bytes)?;
        let split_bytes = serde_json::to_vec_pretty(&split)
            .map_err(|e| DatasetError::Io(format!("split serialize: {e}")))?;
        write_bytes_atomic(&dir.join(SPLIT_FILENAME), &split_bytes)?;

        let (num_rows, stats) = analyze_jsonl(&data_path, format)?;
        if num_rows != identity.rows.len() as u64 {
            return Err(DatasetError::InvalidJsonl(format!(
                "dataset row count changed while artifacts were built: stats={num_rows}, identities={}",
                identity.rows.len()
            )));
        }
        let groups = identity
            .rows
            .iter()
            .filter_map(|row| row.group_id.as_deref())
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;
        let sessions = identity
            .rows
            .iter()
            .filter_map(|row| row.session_id.as_deref())
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;
        let manifest = DatasetManifest {
            schema_version: DATASET_MANIFEST_SCHEMA_VERSION,
            name: name.to_string(),
            format,
            description,
            num_rows,
            size_bytes: std::fs::metadata(&data_path)
                .map_err(|e| DatasetError::Io(format!("inspect data.jsonl: {e}")))?
                .len(),
            created_at,
            updated_at,
            corpus_sha256: identity.corpus_sha256,
            normalized_corpus_sha256: identity.normalized_corpus_sha256,
            split_manifest_sha256: sha256_bytes(&split_bytes),
            split_config,
            split_counts: split.counts,
            num_groups: groups,
            num_sessions: sessions,
            stats,
        };
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }
}

/// Iterator returned by `DatasetRegistry::iter_sft`. Yields one
/// `SftConversation` per non-empty JSONL line; lines that fail to parse
/// are skipped (with a tracing::warn) so a malformed row in a 1000-line
/// dataset doesn't poison the whole synthesis.
pub struct DatasetSftIter<'a> {
    reader: BufReader<std::fs::File>,
    dataset_name: Option<String>,
    source_split: Option<DatasetSplit>,
    identities: HashMap<String, VecDeque<DatasetRowIdentity>>,
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
            let value = match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(value) => value,
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed dataset row");
                    continue;
                }
            };
            match serde_json::from_value::<SftConversation>(value.clone()) {
                Ok(mut conv) => {
                    if let (Some(dataset), Some(source_split)) =
                        (self.dataset_name.as_deref(), self.source_split)
                    {
                        let content_sha256 = kiln_eval::sha256_json(&value);
                        let Some(row) = self
                            .identities
                            .get_mut(&content_sha256)
                            .and_then(VecDeque::pop_front)
                        else {
                            tracing::error!(
                                dataset,
                                split = source_split.as_str(),
                                content_sha256,
                                "split row is missing its persisted dataset identity"
                            );
                            continue;
                        };
                        let provenance = DatasetExampleProvenance {
                            dataset: dataset.to_string(),
                            source_split,
                            row,
                        };
                        if let Ok(value) = serde_json::to_value(provenance) {
                            conv.extra
                                .insert(DATASET_PROVENANCE_METADATA_KEY.to_string(), value);
                        }
                    }
                    return Some(conv);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed dataset row");
                    continue;
                }
            }
        }
    }
}

fn scan_row_identities(
    path: &Path,
    format: DatasetFormat,
) -> Result<Vec<DatasetRowIdentity>, DatasetError> {
    let file = std::fs::File::open(path).map_err(|e| DatasetError::Io(format!("{e}")))?;
    let mut rows = Vec::new();
    for (physical_line, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| DatasetError::Io(format!("{e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(value) => value,
            Err(_error) if matches!(format, DatasetFormat::Raw) => {
                serde_json::Value::String(trimmed.to_string())
            }
            Err(error) => {
                return Err(DatasetError::InvalidJsonl(format!(
                    "line {}: {error}",
                    physical_line + 1
                )));
            }
        };
        match format {
            DatasetFormat::SftChat => {
                serde_json::from_value::<SftConversation>(value.clone()).map_err(|error| {
                    DatasetError::InvalidJsonl(format!(
                        "line {} is not an SFT chat row: {error}",
                        physical_line + 1
                    ))
                })?;
            }
            DatasetFormat::GrpoGroups => {
                serde_json::from_value::<kiln_train::GrpoGroup>(value.clone()).map_err(
                    |error| {
                        DatasetError::InvalidJsonl(format!(
                            "line {} is not a GRPO group: {error}",
                            physical_line + 1
                        ))
                    },
                )?;
            }
            DatasetFormat::Raw => {}
        }
        rows.push(row_identity(rows.len() as u64 + 1, &value));
    }
    Ok(rows)
}

fn materialize_split_files(
    data_path: &Path,
    dir: &Path,
    manifest: &DatasetSplitManifest,
) -> Result<(), DatasetError> {
    let assignments = manifest
        .rows
        .iter()
        .map(|row| (row.row_number, row.split))
        .collect::<HashMap<_, _>>();
    let outputs = [
        (
            DatasetSplit::Train,
            std::fs::File::create(dir.join("train.jsonl.tmp")),
        ),
        (
            DatasetSplit::Validation,
            std::fs::File::create(dir.join("validation.jsonl.tmp")),
        ),
        (
            DatasetSplit::Holdout,
            std::fs::File::create(dir.join("holdout.jsonl.tmp")),
        ),
    ]
    .map(|(split, file)| {
        file.map(|file| (split, file))
            .map_err(|e| DatasetError::Io(format!("create {} split: {e}", split.as_str())))
    });
    let mut outputs = outputs
        .into_iter()
        .collect::<Result<Vec<_>, DatasetError>>()?;
    let input = std::fs::File::open(data_path).map_err(|e| DatasetError::Io(format!("{e}")))?;
    let mut row_number = 0u64;
    for line in BufReader::new(input).lines() {
        let line = line.map_err(|e| DatasetError::Io(format!("{e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        row_number += 1;
        let split = assignments.get(&row_number).ok_or_else(|| {
            DatasetError::InvalidJsonl(format!(
                "split manifest has no assignment for row {row_number}"
            ))
        })?;
        let output = outputs
            .iter_mut()
            .find(|(candidate, _)| candidate == split)
            .map(|(_, file)| file)
            .expect("all split writers exist");
        output
            .write_all(line.as_bytes())
            .and_then(|_| output.write_all(b"\n"))
            .map_err(|e| DatasetError::Io(format!("write {} split: {e}", split.as_str())))?;
    }
    if row_number != manifest.rows.len() as u64 {
        return Err(DatasetError::InvalidJsonl(format!(
            "data has {row_number} rows but split manifest has {}",
            manifest.rows.len()
        )));
    }
    for (split, file) in &mut outputs {
        file.sync_all()
            .map_err(|e| DatasetError::Io(format!("sync {} split: {e}", split.as_str())))?;
    }
    drop(outputs);
    for split in [
        DatasetSplit::Train,
        DatasetSplit::Validation,
        DatasetSplit::Holdout,
    ] {
        std::fs::rename(
            dir.join(format!("{}.jsonl.tmp", split.as_str())),
            dir.join(format!("{}.jsonl", split.as_str())),
        )
        .map_err(|e| DatasetError::Io(format!("publish {} split: {e}", split.as_str())))?;
    }
    Ok(())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), DatasetError> {
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
    ));
    let mut file = std::fs::File::create(&tmp).map_err(|e| DatasetError::Io(format!("{e}")))?;
    file.write_all(bytes)
        .map_err(|e| DatasetError::Io(format!("{e}")))?;
    file.sync_all()
        .map_err(|e| DatasetError::Io(format!("{e}")))?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(|e| DatasetError::Io(format!("{e}")))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DatasetError> {
    let bytes = std::fs::read(path).map_err(|e| DatasetError::Io(format!("{e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| DatasetError::InvalidJsonl(format!("{}: {e}", path.display())))
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
                            if m.tool_calls
                                .as_ref()
                                .map(|v| !v.is_empty())
                                .unwrap_or(false)
                            {
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
        let r = |role: &str, content: &str| serde_json::json!({"role": role, "content": content});
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
            .create(
                "smoke",
                DatasetFormat::SftChat,
                Some("desc".into()),
                rows().as_bytes(),
            )
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
    fn typed_dataset_rejects_malformed_rows_without_leaving_an_orphan() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let body = format!("{}\nnot json\n{{\"messages\": []}}\n", rows());
        let error = reg
            .create("a", DatasetFormat::SftChat, None, body.as_bytes())
            .unwrap_err();
        assert!(matches!(error, DatasetError::InvalidJsonl(_)));
        assert!(!dir.path().join("a").exists());
    }

    #[test]
    fn append_row_updates_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        reg.create("a", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let new_row =
            r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}"#;
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
        assert!(
            head.iter()
                .all(|c| { c.messages.iter().all(|msg| msg.content != "1+1?") })
        );
    }

    #[test]
    fn invalid_names_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        for bad in &["", "..", "a/b", "a\\b"] {
            let err = reg
                .create(bad, DatasetFormat::SftChat, None, rows().as_bytes())
                .unwrap_err();
            assert!(matches!(err, DatasetError::InvalidName(_)));
        }
    }

    #[test]
    fn analyze_compresses_role_patterns() {
        let pattern = compress_pattern(&["assistant", "tool", "tool", "tool", "assistant"]);
        assert_eq!(pattern, "assistant tool×3 assistant");
    }

    #[test]
    fn create_persists_identity_and_group_aware_partitions() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let body = (0..6)
            .map(|index| {
                serde_json::json!({
                    "group_id": format!("group-{}", index / 2),
                    "session_id": format!("session-{}", index / 2),
                    "messages": [
                        {"role": "user", "content": format!("question {index}")},
                        {"role": "assistant", "content": format!("answer {index}")}
                    ]
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let manifest = reg
            .create("partitioned", DatasetFormat::SftChat, None, body.as_bytes())
            .unwrap();
        assert_eq!(manifest.schema_version, DATASET_MANIFEST_SCHEMA_VERSION);
        assert!(manifest.corpus_sha256.starts_with("sha256:"));
        assert_eq!(manifest.num_groups, 3);
        assert_eq!(manifest.num_sessions, 3);
        assert!(manifest.split_counts.train > 0);
        assert!(manifest.split_counts.validation > 0);
        assert!(manifest.split_counts.holdout > 0);

        let split = reg.load_split("partitioned").unwrap();
        for pair in split.rows.chunks_exact(2) {
            assert_eq!(pair[0].group_id, pair[1].group_id);
            assert_eq!(pair[0].split, pair[1].split);
        }
        let holdout = reg
            .head_sft_split("partitioned", DatasetSplit::Holdout, 10)
            .unwrap();
        assert!(!holdout.is_empty());
        let provenance = holdout[0]
            .extra
            .get(DATASET_PROVENANCE_METADATA_KEY)
            .unwrap();
        assert_eq!(provenance["dataset"], "partitioned");
        assert_eq!(provenance["source_split"], "holdout");
    }

    #[test]
    fn legacy_manifest_is_upgraded_with_derived_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let reg = DatasetRegistry::new(dir.path().to_path_buf());
        let dataset = dir.path().join("legacy");
        std::fs::create_dir_all(&dataset).unwrap();
        std::fs::write(dataset.join(DATA_FILENAME), rows()).unwrap();
        std::fs::write(
            dataset.join("manifest.json"),
            serde_json::json!({
                "name": "legacy",
                "format": "sft_chat",
                "num_rows": 2,
                "size_bytes": rows().len(),
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z"
            })
            .to_string(),
        )
        .unwrap();
        let manifest = reg.load_manifest("legacy").unwrap();
        assert_eq!(manifest.schema_version, DATASET_MANIFEST_SCHEMA_VERSION);
        assert!(dataset.join(IDENTITY_FILENAME).is_file());
        assert!(dataset.join(SPLIT_FILENAME).is_file());
        assert!(dataset.join("train.jsonl").is_file());
        assert!(dataset.join("holdout.jsonl").is_file());
    }
}

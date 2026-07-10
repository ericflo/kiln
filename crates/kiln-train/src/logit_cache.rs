//! Logit cache (grand plan §3.3).
//!
//! Prefix-keyed on-disk cache of teacher top-K logprobs so canonical
//! prompts get distilled once and reused forever. Mirrors a
//! [`LogitSource`] interface but reads from disk before falling
//! through to a wrapped inner source.
//!
//! # Storage layout
//!
//! Each entry is a JSON file at:
//!
//! ```text
//! <cache_root>/v2-<teacher_sha256>/v2-<tokenizer_sha256>/<prefix_hash>/<position>.json
//! ```
//!
//! Teacher and tokenizer identifiers are hashed into fixed-size path
//! components, and their exact values are recorded in every entry and checked
//! on read. This keeps arbitrary identifiers out of filesystem paths without
//! making a digest collision an identity alias. The former lossy path format
//! is intentionally not read: existing entries remain on disk but become cold
//! misses after upgrading to the v2 layout.
//!
//! The file contents:
//!
//! ```json
//! {
//!   "indices": [...],
//!   "logprobs": [...],
//!   "top_k": K,
//!   "produced_at": "2026-05-15T...",
//!   "_kiln_cache_identity": {
//!     "path_version": 2,
//!     "teacher_id": "...",
//!     "tokenizer_hash": "...",
//!     "prefix_hash": N,
//!     "position": N
//!   }
//! }
//! ```
//!
//! Why JSON-on-disk instead of RocksDB for milestone-10: zero new
//! workspace deps, trivial to inspect on disk, trivial to tarball
//! for §3.3's "canonical-domain prepopulated cache" distribution
//! pattern. A RocksDB-backed store is a paid optimisation if and
//! when cardinality > 1M entries actually surfaces a hot-path
//! issue.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::logit_source::{
    LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs,
    validate_full_vocab_logprobs_batch, validate_logit_request, validate_topk_logprob_row,
    validate_topk_logprobs_batch,
};

/// One on-disk cache entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub indices: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub top_k: usize,
    pub produced_at: String,
}

impl CacheEntry {
    pub fn approximate_bytes(&self) -> u64 {
        let payload = self.indices.len() * 4 + self.logprobs.len() * 4 + 32;
        payload as u64
    }
}

const CACHE_PATH_VERSION: u8 = 2;
const CACHE_PATH_PREFIX: &str = "v2-";

/// Exact identifier identity stored beside each cached value. The path digest
/// bounds path length; this record makes an identifier-digest collision fail
/// closed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheIdentity {
    path_version: u8,
    teacher_id: String,
    tokenizer_hash: String,
    prefix_hash: u64,
    position: usize,
}

impl CacheIdentity {
    fn new(teacher_id: &str, tokenizer_hash: &str, prefix_hash: u64, position: usize) -> Self {
        Self {
            path_version: CACHE_PATH_VERSION,
            teacher_id: teacher_id.to_owned(),
            tokenizer_hash: tokenizer_hash.to_owned(),
            prefix_hash,
            position,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCacheEntry {
    #[serde(flatten)]
    entry: CacheEntry,
    #[serde(rename = "_kiln_cache_identity", default)]
    identity: Option<CacheIdentity>,
}

impl StoredCacheEntry {
    fn new(entry: &CacheEntry, identity: CacheIdentity) -> Self {
        Self {
            entry: entry.clone(),
            identity: Some(identity),
        }
    }

    fn verify_identity(&self, expected: &CacheIdentity) -> Result<()> {
        let actual = self.identity.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "cache entry has no v{} identity metadata; legacy cache entries are invalidated",
                CACHE_PATH_VERSION
            )
        })?;
        anyhow::ensure!(
            actual == expected,
            "cache identity mismatch: expected {expected:?}, found {actual:?}"
        );
        Ok(())
    }
}

/// Compute the FNV-1a prefix hash used as a cache key.
pub fn hash_prefix(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// On-disk logit cache. Thread-safe (filesystem path operations are).
#[derive(Debug, Clone)]
pub struct LogitCache {
    root: PathBuf,
}

impl LogitCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Build the canonical on-disk path for an entry.
    fn entry_path(
        &self,
        teacher_id: &str,
        tokenizer_hash: &str,
        prefix_hash: u64,
        position: usize,
    ) -> PathBuf {
        self.root
            .join(identity_path_segment(teacher_id))
            .join(identity_path_segment(tokenizer_hash))
            .join(format!("{prefix_hash:016x}"))
            .join(format!("{position}.json"))
    }

    /// Write one entry. Creates parent directories as needed.
    pub fn put(
        &self,
        teacher_id: &str,
        tokenizer_hash: &str,
        prefix_hash: u64,
        position: usize,
        entry: &CacheEntry,
    ) -> Result<()> {
        let path = self.entry_path(teacher_id, tokenizer_hash, prefix_hash, position);
        let identity = CacheIdentity::new(teacher_id, tokenizer_hash, prefix_hash, position);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create cache parent {}", parent.display()))?;
        }
        // Never overwrite a value whose exact logical identity differs. This
        // is primarily a fail-closed guard for a SHA-256 collision, but also
        // catches misplaced or tampered cache files.
        if path.exists() {
            let existing = std::fs::read(&path)
                .with_context(|| format!("read existing cache entry {}", path.display()))?;
            let stored: StoredCacheEntry = serde_json::from_slice(&existing)
                .with_context(|| format!("parse existing cache entry {}", path.display()))?;
            stored.verify_identity(&identity).with_context(|| {
                format!("verify existing cache entry identity {}", path.display())
            })?;
        }
        let stored = StoredCacheEntry::new(entry, identity);
        let bytes = serde_json::to_vec(&stored).context("serialize cache entry")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("write cache entry {}", path.display()))?;
        Ok(())
    }

    /// Read one entry. `Ok(None)` for cache miss.
    pub fn get(
        &self,
        teacher_id: &str,
        tokenizer_hash: &str,
        prefix_hash: u64,
        position: usize,
    ) -> Result<Option<CacheEntry>> {
        let path = self.entry_path(teacher_id, tokenizer_hash, prefix_hash, position);
        if !path.exists() {
            return Ok(None);
        }
        let bytes =
            std::fs::read(&path).with_context(|| format!("read cache entry {}", path.display()))?;
        let stored: StoredCacheEntry = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse cache entry {}", path.display()))?;
        let expected = CacheIdentity::new(teacher_id, tokenizer_hash, prefix_hash, position);
        stored
            .verify_identity(&expected)
            .with_context(|| format!("verify cache entry identity {}", path.display()))?;
        Ok(Some(stored.entry))
    }

    /// Walk the cache and compute size statistics.
    pub fn stats(&self) -> Result<CacheStats> {
        let mut total_entries: usize = 0;
        let mut total_bytes: u64 = 0;
        let mut per_teacher: std::collections::BTreeMap<String, usize> = Default::default();
        if !self.root.exists() {
            return Ok(CacheStats::default());
        }
        for entry in walkdir(&self.root)? {
            if entry.is_file() {
                total_entries += 1;
                if let Ok(meta) = entry.metadata() {
                    total_bytes += meta.len();
                }
                // New entries retain the exact teacher ID in their identity
                // metadata. Fall back to the first path component for legacy
                // or corrupt files so stats remains best-effort.
                let teacher_id = std::fs::read(&entry)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<StoredCacheEntry>(&bytes).ok())
                    .and_then(|stored| stored.identity)
                    .filter(|identity| identity.path_version == CACHE_PATH_VERSION)
                    .map(|identity| identity.teacher_id)
                    .or_else(|| {
                        entry
                            .strip_prefix(&self.root)
                            .ok()
                            .and_then(|rel| rel.components().next())
                            .and_then(|component| match component {
                                std::path::Component::Normal(segment) => {
                                    Some(segment.to_string_lossy().into_owned())
                                }
                                _ => None,
                            })
                    });
                if let Some(teacher_id) = teacher_id {
                    *per_teacher.entry(teacher_id).or_insert(0) += 1;
                }
            }
        }
        Ok(CacheStats {
            total_entries,
            total_bytes,
            per_teacher,
        })
    }

    /// Tar the cache directory into the given output path. Returns the
    /// number of bytes written.
    pub fn export_to_tar(&self, output: &Path) -> Result<u64> {
        if !self.root.exists() {
            return Err(anyhow::anyhow!(
                "cache root {} does not exist",
                self.root.display()
            ));
        }
        let file = std::fs::File::create(output)
            .with_context(|| format!("create tar output {}", output.display()))?;
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut tb = tar::Builder::new(gz);
        tb.append_dir_all("logit-cache", &self.root)
            .context("append cache root to tar")?;
        let writer = tb.into_inner().context("finalize tar")?;
        let written_compressed = writer.finish().context("finish gz")?;
        let bytes = written_compressed.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(bytes)
    }

    /// Untar a cache tarball into the cache root. Returns the number
    /// of entry files written.
    pub fn import_from_tar(&self, input: &Path) -> Result<usize> {
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("create cache root {}", self.root.display()))?;
        let file = std::fs::File::open(input)
            .with_context(|| format!("open tar input {}", input.display()))?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut tb = tar::Archive::new(gz);
        let mut written = 0;
        for entry in tb.entries().context("iter tar entries")? {
            let mut entry = entry?;
            let path = entry.path()?.into_owned();
            // Strip the leading "logit-cache/" prefix written by
            // export_to_tar; anything that doesn't start with it
            // we skip (defensive against tampered tarballs).
            let rel = match path.strip_prefix("logit-cache/") {
                Ok(p) => p.to_path_buf(),
                Err(_) => continue,
            };
            let dst = self.root.join(&rel);
            if entry.header().entry_type().is_dir() {
                std::fs::create_dir_all(&dst).ok();
                continue;
            }
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            entry.unpack(&dst)?;
            written += 1;
        }
        Ok(written)
    }
}

/// Cache statistics returned by [`LogitCache::stats`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_bytes: u64,
    /// Per-teacher entry counts. Use to identify hottest teachers.
    pub per_teacher: std::collections::BTreeMap<String, usize>,
}

/// A [`LogitSource`] that consults a local [`LogitCache`] before
/// falling through to an inner source. §3.3 of the grand plan
/// ("CachedTeacher").
#[derive(Debug)]
pub struct CachedLogitSource<Inner> {
    cache: LogitCache,
    inner: Inner,
}

impl<Inner> CachedLogitSource<Inner>
where
    Inner: LogitSource,
{
    pub fn new(cache: LogitCache, inner: Inner) -> Self {
        Self { cache, inner }
    }
}

impl<Inner: LogitSource + std::fmt::Debug> LogitSource for CachedLogitSource<Inner> {
    fn capabilities(&self) -> LogitSourceCaps {
        self.inner.capabilities()
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let caps = self.inner.capabilities();
        validate_logit_request(&caps, tokens, positions, top_k)?;
        let Some(requested_k) = top_k else {
            // Full-vocabulary rows are intentionally not stored in this sparse
            // cache. Preserve the wrapped source's advertised capability.
            let batch = self.inner.fetch_logprobs(tokens, positions, None)?;
            validate_full_vocab_logprobs_batch(&caps, tokens, positions, &batch)?;
            return Ok(batch);
        };
        let flat_capacity = positions.len().checked_mul(requested_k).ok_or_else(|| {
            LogitSourceError::invalid(&caps.teacher_id, "top-K response shape overflows usize")
        })?;
        let prefix_hash = hash_prefix(tokens);
        let tokenizer_hash = caps.tokenizer_hash.as_deref().unwrap_or("default");

        #[derive(Debug, Clone)]
        struct SparseRow {
            indices: Vec<u32>,
            logprobs: Vec<f32>,
        }

        let mut result_slots: Vec<Option<SparseRow>> = vec![None; positions.len()];
        let mut resolved_rows: HashMap<usize, SparseRow> = HashMap::new();
        let mut miss_index_by_position: HashMap<usize, usize> = HashMap::new();
        let mut miss_positions: Vec<usize> = Vec::new();
        let mut miss_slots: Vec<Vec<usize>> = Vec::new();

        for (slot_index, &position) in positions.iter().enumerate() {
            if let Some(row) = resolved_rows.get(&position) {
                result_slots[slot_index] = Some(row.clone());
                continue;
            }
            if let Some(&miss_index) = miss_index_by_position.get(&position) {
                miss_slots[miss_index].push(slot_index);
                continue;
            }

            let cached = self
                .cache
                .get(&caps.teacher_id, tokenizer_hash, prefix_hash, position)
                .map_err(|error| {
                    LogitSourceError::invalid(
                        &caps.teacher_id,
                        format!("read cache position {position}: {error:#}"),
                    )
                })?;
            if let Some(entry) = cached {
                // Validate the full stored row before considering a prefix of a
                // wider entry. This prevents corrupt legacy tails from staying
                // latent until a later, larger-K query.
                validate_topk_logprob_row(
                    &caps,
                    entry.top_k,
                    slot_index,
                    &entry.indices,
                    &entry.logprobs,
                )?;
                if entry.top_k >= requested_k {
                    let row = SparseRow {
                        indices: entry.indices[..requested_k].to_vec(),
                        logprobs: entry.logprobs[..requested_k].to_vec(),
                    };
                    result_slots[slot_index] = Some(row.clone());
                    resolved_rows.insert(position, row);
                    continue;
                }
            }

            let miss_index = miss_positions.len();
            miss_positions.push(position);
            miss_slots.push(vec![slot_index]);
            miss_index_by_position.insert(position, miss_index);
        }

        let mut new_cache_entries = Vec::with_capacity(miss_positions.len());
        if !miss_positions.is_empty() {
            let batch = self
                .inner
                .fetch_logprobs(tokens, &miss_positions, Some(requested_k))?;
            let LogprobBatch::TopK(batch) = batch else {
                return Err(LogitSourceError::invalid(
                    &caps.teacher_id,
                    "cache pass-through requested top-K but inner source returned full-vocab",
                ));
            };
            // The complete inner result is checked before constructing the
            // first write-through entry, which makes validation failures
            // failure-atomic with respect to the cache.
            validate_topk_logprobs_batch(&caps, tokens, &miss_positions, requested_k, &batch)?;

            let produced_at = chrono::Utc::now().to_rfc3339();
            for (miss_index, &position) in miss_positions.iter().enumerate() {
                let start = miss_index * requested_k;
                let end = start + requested_k;
                let row = SparseRow {
                    indices: batch.indices[start..end].to_vec(),
                    logprobs: batch.logprobs[start..end].to_vec(),
                };
                for &slot_index in &miss_slots[miss_index] {
                    result_slots[slot_index] = Some(row.clone());
                }
                new_cache_entries.push((
                    position,
                    CacheEntry {
                        indices: row.indices,
                        logprobs: row.logprobs,
                        top_k: requested_k,
                        produced_at: produced_at.clone(),
                    },
                ));
            }
        }

        let mut all_indices = Vec::with_capacity(flat_capacity);
        let mut all_logprobs = Vec::with_capacity(flat_capacity);
        for (slot_index, row) in result_slots.into_iter().enumerate() {
            let row = row.ok_or_else(|| {
                LogitSourceError::invalid(
                    &caps.teacher_id,
                    format!("cache result slot {slot_index} was not populated"),
                )
            })?;
            all_indices.extend(row.indices);
            all_logprobs.extend(row.logprobs);
        }

        let output = TopKLogprobs {
            indices: all_indices,
            logprobs: all_logprobs,
            top_k: requested_k,
        };
        validate_topk_logprobs_batch(&caps, tokens, positions, requested_k, &output)?;

        // All hits, the complete inner response, and the assembled position
        // order have now been validated. Validation errors therefore write no
        // new entries; cache I/O remains best-effort after this boundary.
        for (position, entry) in &new_cache_entries {
            if let Err(error) = self.cache.put(
                &caps.teacher_id,
                tokenizer_hash,
                prefix_hash,
                *position,
                entry,
            ) {
                tracing::warn!(error = %error, position, "logit cache write-through failed");
            }
        }

        Ok(LogprobBatch::TopK(output))
    }
}

/// Map an arbitrary UTF-8 identifier to one fixed-size normal path component.
/// The original value is retained in [`CacheIdentity`] and verified on reads.
fn identity_path_segment(identity: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let digest = Sha256::digest(identity.as_bytes());
    let mut segment = String::with_capacity(CACHE_PATH_PREFIX.len() + digest.len() * 2);
    segment.push_str(CACHE_PATH_PREFIX);
    for byte in digest {
        segment.push(HEX[(byte >> 4) as usize] as char);
        segment.push(HEX[(byte & 0x0f) as usize] as char);
    }
    segment
}

fn walkdir(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p).with_context(|| format!("read_dir {}", p.display()))? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logit_source::FixtureLogitSource;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    #[derive(Debug, Clone)]
    enum ScriptedResponse {
        ByPosition,
        Exact(LogprobBatch),
    }

    #[derive(Debug, Clone)]
    struct ScriptedSource {
        caps: LogitSourceCaps,
        response: ScriptedResponse,
        calls: Arc<Mutex<Vec<Vec<usize>>>>,
    }

    impl ScriptedSource {
        fn by_position() -> Self {
            Self {
                caps: test_caps(),
                response: ScriptedResponse::ByPosition,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn exact(batch: LogprobBatch) -> Self {
            Self {
                caps: test_caps(),
                response: ScriptedResponse::Exact(batch),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn calls(&self) -> Vec<Vec<usize>> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl LogitSource for ScriptedSource {
        fn capabilities(&self) -> LogitSourceCaps {
            self.caps.clone()
        }

        fn fetch_logprobs(
            &self,
            _tokens: &[u32],
            positions: &[usize],
            top_k: Option<usize>,
        ) -> std::result::Result<LogprobBatch, LogitSourceError> {
            self.calls.lock().unwrap().push(positions.to_vec());
            match &self.response {
                ScriptedResponse::Exact(batch) => Ok(batch.clone()),
                ScriptedResponse::ByPosition => {
                    let top_k = top_k.ok_or_else(|| LogitSourceError::FullVocabUnsupported {
                        teacher_id: self.caps.teacher_id.clone(),
                    })?;
                    let mut indices = Vec::with_capacity(positions.len() * top_k);
                    let mut logprobs = Vec::with_capacity(positions.len() * top_k);
                    for &position in positions {
                        let entry = valid_entry(position, top_k);
                        indices.extend(entry.indices);
                        logprobs.extend(entry.logprobs);
                    }
                    Ok(LogprobBatch::TopK(TopKLogprobs {
                        indices,
                        logprobs,
                        top_k,
                    }))
                }
            }
        }
    }

    fn test_caps() -> LogitSourceCaps {
        LogitSourceCaps {
            teacher_id: "teacher".into(),
            vocab_size: 64,
            max_top_k: 4,
            supports_full_vocab: false,
            supports_batched: true,
            tokenizer_hash: Some("tok-v1".into()),
        }
    }

    fn valid_entry(position: usize, top_k: usize) -> CacheEntry {
        CacheEntry {
            indices: (0..top_k)
                .map(|candidate| (position * 8 + candidate) as u32)
                .collect(),
            logprobs: vec![-2.0; top_k],
            top_k,
            produced_at: "2026-05-15T00:00:00Z".into(),
        }
    }

    fn topk(batch: LogprobBatch) -> TopKLogprobs {
        match batch {
            LogprobBatch::TopK(batch) => batch,
            LogprobBatch::FullVocab { .. } => panic!("expected top-K batch"),
        }
    }

    #[test]
    fn put_get_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let entry = CacheEntry {
            indices: vec![1, 2, 3, 4],
            logprobs: vec![-2.0, -2.1, -2.2, -2.3],
            top_k: 4,
            produced_at: "2026-05-15T00:00:00Z".into(),
        };
        cache.put("teacher", "tok-v1", 0xdeadbeef, 17, &entry)?;
        let got = cache.get("teacher", "tok-v1", 0xdeadbeef, 17)?.unwrap();
        assert_eq!(got.indices, entry.indices);
        assert_eq!(got.top_k, 4);
        Ok(())
    }

    #[test]
    fn get_returns_none_on_miss() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        assert!(cache.get("x", "y", 0, 0)?.is_none());
        Ok(())
    }

    #[test]
    fn stats_aggregates_per_teacher() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let e = CacheEntry {
            indices: vec![1, 2],
            logprobs: vec![-0.1, -0.2],
            top_k: 2,
            produced_at: "".into(),
        };
        cache.put("t1", "v1", 1, 0, &e)?;
        cache.put("t1", "v1", 1, 1, &e)?;
        cache.put("t2", "v1", 1, 0, &e)?;
        let stats = cache.stats()?;
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.per_teacher.get("t1"), Some(&2));
        assert_eq!(stats.per_teacher.get("t2"), Some(&1));
        Ok(())
    }

    #[test]
    fn export_then_import_round_trip() -> Result<()> {
        let src_dir = tempdir()?;
        let cache = LogitCache::new(src_dir.path());
        let e = CacheEntry {
            indices: vec![10, 20, 30, 40],
            logprobs: vec![-0.5, -0.6, -0.7, -0.8],
            top_k: 4,
            produced_at: "".into(),
        };
        cache.put("teacher@A", "tok-v1", 0xabc, 5, &e)?;
        cache.put("teacher@A", "tok-v1", 0xabc, 6, &e)?;

        let tar_dir = tempdir()?;
        let tar_path = tar_dir.path().join("cache.tar.gz");
        cache.export_to_tar(&tar_path)?;
        assert!(tar_path.exists());
        assert!(tar_path.metadata()?.len() > 0);

        let dst_dir = tempdir()?;
        let cache2 = LogitCache::new(dst_dir.path());
        let written = cache2.import_from_tar(&tar_path)?;
        assert_eq!(written, 2);
        let restored = cache2.get("teacher@A", "tok-v1", 0xabc, 5)?.unwrap();
        assert_eq!(restored.indices, e.indices);
        assert_eq!(restored.top_k, 4);
        Ok(())
    }

    #[test]
    fn identity_path_segment_is_fixed_size_and_stable() {
        assert_eq!(
            identity_path_segment("teacher"),
            "v2-1057a9604e04b274da5a4de0c8f4b4868d9b230989f8c8c6a28221143cc5a755"
        );
        assert_eq!(identity_path_segment("teacher").len(), 67);
        assert!(
            identity_path_segment(&"unbounded/identifier".repeat(100_000))
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        );
    }

    #[test]
    fn identities_that_collided_in_legacy_paths_have_distinct_v2_paths() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let legacy_collision_pairs = [
            ("a/b".to_owned(), "a_b".to_owned()),
            ("a\\b".to_owned(), "a_b".to_owned()),
            ("with space".to_owned(), "with_space".to_owned()),
            (
                format!("{}a", "x".repeat(255)),
                format!("{}b", "x".repeat(255)),
            ),
        ];

        for (left, right) in legacy_collision_pairs {
            assert_ne!(
                cache.entry_path(&left, "tokenizer", 1, 0),
                cache.entry_path(&right, "tokenizer", 1, 0),
                "distinct identifier bytes must select distinct cache paths"
            );
        }
    }

    #[test]
    fn dot_and_separator_identities_cannot_escape_cache_root() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());

        for identity in [".", "..", "../outside", "/absolute", "a/b", "a\\b", ""] {
            let path = cache.entry_path(identity, "../../tokenizer", 0, 0);
            let relative = path.strip_prefix(dir.path()).unwrap();
            assert_eq!(relative.components().count(), 4);
            assert!(
                relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
            );
        }
    }

    #[test]
    fn dangerous_identity_round_trip_preserves_exact_identity() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let entry = valid_entry(1, 2);
        let teacher_id = "../models/model:a?b/\u{0}long teacher";
        let tokenizer_hash = "../../tokenizers/hash\\one";

        cache.put(teacher_id, tokenizer_hash, 0xabc, 1, &entry)?;
        let restored = cache.get(teacher_id, tokenizer_hash, 0xabc, 1)?.unwrap();
        assert_eq!(restored.indices, entry.indices);

        let bytes = std::fs::read(cache.entry_path(teacher_id, tokenizer_hash, 0xabc, 1))?;
        let stored: StoredCacheEntry = serde_json::from_slice(&bytes)?;
        assert_eq!(
            stored.identity,
            Some(CacheIdentity::new(teacher_id, tokenizer_hash, 0xabc, 1))
        );
        Ok(())
    }

    #[test]
    fn identity_mismatch_at_v2_path_fails_closed() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let path = cache.entry_path("teacher", "tokenizer", 7, 3);
        std::fs::create_dir_all(path.parent().unwrap())?;
        let misplaced = StoredCacheEntry::new(
            &valid_entry(3, 2),
            CacheIdentity::new("different-teacher", "tokenizer", 7, 3),
        );
        std::fs::write(&path, serde_json::to_vec(&misplaced)?)?;

        let get_error = cache.get("teacher", "tokenizer", 7, 3).unwrap_err();
        assert!(
            get_error
                .to_string()
                .contains("verify cache entry identity")
        );
        let put_error = cache
            .put("teacher", "tokenizer", 7, 3, &valid_entry(3, 2))
            .unwrap_err();
        assert!(
            put_error
                .to_string()
                .contains("verify existing cache entry identity")
        );
        Ok(())
    }

    #[test]
    fn legacy_lossy_layout_is_deliberately_invalidated() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let legacy_path = dir
            .path()
            .join("teacher")
            .join("tok-v1")
            .join("0000000000000001")
            .join("0.json");
        std::fs::create_dir_all(legacy_path.parent().unwrap())?;
        std::fs::write(&legacy_path, serde_json::to_vec(&valid_entry(0, 2))?)?;

        assert!(cache.get("teacher", "tok-v1", 1, 0)?.is_none());
        assert!(
            cache
                .entry_path("teacher", "tok-v1", 1, 0)
                .starts_with(dir.path())
        );
        Ok(())
    }

    #[test]
    fn cached_source_wraps_fixture_and_writes_through() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let mut fixture = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let tokens = vec![10u32, 20, 30, 40];
        let h = FixtureLogitSource::hash_tokens(&tokens);
        fixture.insert(h, 1, vec![1, 2, 3, 4], vec![-2.0, -2.1, -2.2, -2.3]);
        fixture.insert(h, 2, vec![5, 6, 7, 8], vec![-2.0, -2.1, -2.2, -2.3]);

        let cached = CachedLogitSource::new(cache.clone(), fixture);
        let first = cached.fetch_logprobs(&tokens, &[1, 2], Some(4)).unwrap();
        match &first {
            LogprobBatch::TopK(t) => {
                assert_eq!(t.indices, vec![1, 2, 3, 4, 5, 6, 7, 8]);
            }
            _ => panic!("expected TopK"),
        }
        // Now the cache should have entries for both positions.
        let stats = cache.stats().unwrap();
        assert_eq!(stats.total_entries, 2);
        // A second call must succeed even though the fixture would
        // also produce the same answer; we just want to confirm the
        // cache-hit path is the no-op success.
        let second = cached.fetch_logprobs(&tokens, &[1, 2], Some(4)).unwrap();
        match &second {
            LogprobBatch::TopK(t) => {
                assert_eq!(t.indices, vec![1, 2, 3, 4, 5, 6, 7, 8]);
            }
            _ => panic!("expected TopK"),
        }
    }

    #[test]
    fn cache_preserves_miss_then_hit_order() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let prefix_hash = hash_prefix(&tokens);
        cache
            .put("teacher", "tok-v1", prefix_hash, 2, &valid_entry(2, 2))
            .unwrap();
        let inner = ScriptedSource::by_position();
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache, inner);

        let batch = topk(source.fetch_logprobs(&tokens, &[1, 2], Some(2)).unwrap());
        assert_eq!(batch.indices, vec![8, 9, 16, 17]);
        assert_eq!(calls.calls(), vec![vec![1]]);
    }

    #[test]
    fn cache_preserves_hit_miss_hit_order() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let prefix_hash = hash_prefix(&tokens);
        for position in [1, 3] {
            cache
                .put(
                    "teacher",
                    "tok-v1",
                    prefix_hash,
                    position,
                    &valid_entry(position, 2),
                )
                .unwrap();
        }
        let inner = ScriptedSource::by_position();
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache, inner);

        let batch = topk(source.fetch_logprobs(&tokens, &[1, 2, 3], Some(2)).unwrap());
        assert_eq!(batch.indices, vec![8, 9, 16, 17, 24, 25]);
        assert_eq!(calls.calls(), vec![vec![2]]);
    }

    #[test]
    fn duplicate_misses_use_one_inner_row_and_one_cache_entry() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let prefix_hash = hash_prefix(&tokens);
        cache
            .put("teacher", "tok-v1", prefix_hash, 1, &valid_entry(1, 2))
            .unwrap();
        let inner = ScriptedSource::by_position();
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache.clone(), inner);

        let batch = topk(source.fetch_logprobs(&tokens, &[2, 1, 2], Some(2)).unwrap());
        assert_eq!(batch.indices, vec![16, 17, 8, 9, 16, 17]);
        assert_eq!(calls.calls(), vec![vec![2]]);
        assert_eq!(cache.stats().unwrap().total_entries, 2);
    }

    #[test]
    fn malformed_legacy_hit_fails_before_inner_fetch_or_new_write() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let prefix_hash = hash_prefix(&tokens);
        let malformed = CacheEntry {
            indices: vec![16],
            logprobs: vec![-2.0],
            top_k: 2,
            produced_at: String::new(),
        };
        cache
            .put("teacher", "tok-v1", prefix_hash, 2, &malformed)
            .unwrap();
        let inner = ScriptedSource::by_position();
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache.clone(), inner);

        assert!(source.fetch_logprobs(&tokens, &[1, 2], Some(2)).is_err());
        assert!(calls.calls().is_empty());
        assert!(
            cache
                .get("teacher", "tok-v1", prefix_hash, 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(cache.stats().unwrap().total_entries, 1);
    }

    #[test]
    fn malformed_inner_shape_writes_no_cache_entries() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let malformed = LogprobBatch::TopK(TopKLogprobs {
            indices: vec![8, 9],
            logprobs: vec![-2.0, -2.0],
            top_k: 2,
        });
        let source = CachedLogitSource::new(cache.clone(), ScriptedSource::exact(malformed));

        assert!(source.fetch_logprobs(&tokens, &[1, 2], Some(2)).is_err());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }

    #[test]
    fn malformed_later_inner_row_cannot_partially_write() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let tokens = [1, 2, 3, 4];
        let malformed = LogprobBatch::TopK(TopKLogprobs {
            indices: vec![8, 9, 16, 16],
            logprobs: vec![-2.0; 4],
            top_k: 2,
        });
        let source = CachedLogitSource::new(cache.clone(), ScriptedSource::exact(malformed));

        assert!(source.fetch_logprobs(&tokens, &[1, 2], Some(2)).is_err());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }

    #[test]
    fn invalid_k_never_reads_through_to_inner() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let inner = ScriptedSource::by_position();
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache, inner);
        let tokens = [1, 2, 3, 4];

        assert!(source.fetch_logprobs(&tokens, &[1], Some(0)).is_err());
        assert!(source.fetch_logprobs(&tokens, &[1], Some(5)).is_err());
        assert!(calls.calls().is_empty());
    }

    #[test]
    fn full_vocab_passthrough_is_validated_and_not_cached() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let uniform = -(4.0f32).ln();
        let mut inner = ScriptedSource::exact(LogprobBatch::FullVocab {
            logprobs: vec![uniform; 4],
            vocab_size: 4,
        });
        inner.caps.vocab_size = 4;
        inner.caps.supports_full_vocab = true;
        let source = CachedLogitSource::new(cache.clone(), inner);

        let batch = source.fetch_logprobs(&[0, 1], &[0], None).unwrap();
        assert!(matches!(batch, LogprobBatch::FullVocab { .. }));
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }

    #[test]
    fn malformed_full_vocab_passthrough_is_rejected() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let mut inner = ScriptedSource::exact(LogprobBatch::FullVocab {
            logprobs: vec![-10.0; 4],
            vocab_size: 4,
        });
        inner.caps.vocab_size = 4;
        inner.caps.supports_full_vocab = true;
        let source = CachedLogitSource::new(cache.clone(), inner);

        assert!(source.fetch_logprobs(&[0, 1], &[0], None).is_err());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }
}

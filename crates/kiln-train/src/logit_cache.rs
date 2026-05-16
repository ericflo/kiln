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
//! <cache_root>/<teacher_id>/<tokenizer_hash>/<prefix_hash>/<position>.json
//! ```
//!
//! The file contents:
//!
//! ```json
//! {
//!   "indices": [...],
//!   "logprobs": [...],
//!   "top_k": K,
//!   "produced_at": "2026-05-15T...",
//!   "size_bytes": N
//! }
//! ```
//!
//! Why JSON-on-disk instead of RocksDB for milestone-10: zero new
//! workspace deps, trivial to inspect on disk, trivial to tarball
//! for §3.3's "canonical-domain prepopulated cache" distribution
//! pattern. A RocksDB-backed store is a paid optimisation if and
//! when cardinality > 1M entries actually surfaces a hot-path
//! issue.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::logit_source::{LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs};

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
            .join(safe_path_segment(teacher_id))
            .join(safe_path_segment(tokenizer_hash))
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
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create cache parent {}", parent.display()))?;
        }
        let bytes =
            serde_json::to_vec(entry).context("serialize cache entry")?;
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
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read cache entry {}", path.display()))?;
        Ok(Some(
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parse cache entry {}", path.display()))?,
        ))
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
                // Teacher id is the first child of root.
                if let Ok(rel) = entry.strip_prefix(&self.root) {
                    if let Some(first) = rel.components().next() {
                        if let std::path::Component::Normal(s) = first {
                            *per_teacher
                                .entry(s.to_string_lossy().to_string())
                                .or_insert(0) += 1;
                        }
                    }
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
        let gz =
            flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut tb = tar::Builder::new(gz);
        tb.append_dir_all("logit-cache", &self.root)
            .context("append cache root to tar")?;
        let writer = tb.into_inner().context("finalize tar")?;
        let written_compressed = writer.finish().context("finish gz")?;
        let bytes = written_compressed
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);
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
        let requested_k = top_k.unwrap_or(caps.max_top_k);
        let prefix_hash = hash_prefix(tokens);

        let mut all_indices: Vec<u32> = Vec::with_capacity(positions.len() * requested_k);
        let mut all_logprobs: Vec<f32> = Vec::with_capacity(positions.len() * requested_k);
        let mut misses: Vec<usize> = Vec::new();

        for &pos in positions {
            match self
                .cache
                .get(
                    &caps.teacher_id,
                    caps.tokenizer_hash.as_deref().unwrap_or("default"),
                    prefix_hash,
                    pos,
                )
                .map_err(|e| LogitSourceError::invalid(&caps.teacher_id, e.to_string()))?
            {
                Some(entry) if entry.top_k >= requested_k => {
                    all_indices.extend_from_slice(&entry.indices[..requested_k]);
                    all_logprobs.extend_from_slice(&entry.logprobs[..requested_k]);
                }
                _ => {
                    misses.push(pos);
                }
            }
        }

        if !misses.is_empty() {
            // Fetch the misses from the inner source, then write them
            // through to cache.
            let batch = self.inner.fetch_logprobs(tokens, &misses, top_k)?;
            let LogprobBatch::TopK(t) = batch else {
                return Err(LogitSourceError::invalid(
                    &caps.teacher_id,
                    "cache pass-through only supports TopK batches",
                ));
            };
            for (i, &pos) in misses.iter().enumerate() {
                let start = i * t.top_k;
                let end = start + t.top_k;
                let entry = CacheEntry {
                    indices: t.indices[start..end].to_vec(),
                    logprobs: t.logprobs[start..end].to_vec(),
                    top_k: t.top_k,
                    produced_at: chrono::Utc::now().to_rfc3339(),
                };
                if let Err(e) = self.cache.put(
                    &caps.teacher_id,
                    caps.tokenizer_hash.as_deref().unwrap_or("default"),
                    prefix_hash,
                    pos,
                    &entry,
                ) {
                    tracing::warn!(error = %e, "logit cache write-through failed");
                }
                all_indices.extend_from_slice(&t.indices[start..end]);
                all_logprobs.extend_from_slice(&t.logprobs[start..end]);
            }
        }

        Ok(LogprobBatch::TopK(TopKLogprobs {
            indices: all_indices,
            logprobs: all_logprobs,
            top_k: requested_k,
        }))
    }
}

/// File-system-safe path segment encoding. Strips path separators and
/// limits length so we can't get pathological cache file paths.
fn safe_path_segment(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | '\0' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_ascii_whitespace() => '_',
            c => c,
        })
        .take(255)
        .collect()
}

fn walkdir(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p)
            .with_context(|| format!("read_dir {}", p.display()))?
        {
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
    use tempfile::tempdir;

    #[test]
    fn put_get_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let cache = LogitCache::new(dir.path());
        let entry = CacheEntry {
            indices: vec![1, 2, 3, 4],
            logprobs: vec![-0.1, -0.2, -0.3, -0.4],
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
    fn safe_path_segment_handles_dangerous_chars() {
        assert_eq!(safe_path_segment("a/b"), "a_b");
        assert_eq!(safe_path_segment("a\\b"), "a_b");
        assert_eq!(safe_path_segment("with space"), "with_space");
    }

    #[test]
    fn cached_source_wraps_fixture_and_writes_through() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let mut fixture = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let tokens = vec![100u32, 200, 300, 400];
        let h = FixtureLogitSource::hash_tokens(&tokens);
        fixture.insert(h, 1, vec![1, 2, 3, 4], vec![-0.1, -0.2, -0.3, -0.4]);
        fixture.insert(h, 2, vec![5, 6, 7, 8], vec![-0.5, -0.6, -0.7, -0.8]);

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
}

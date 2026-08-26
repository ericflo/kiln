//! Identity-bound, causal-prefix logit cache.
//!
//! Cache entries are reusable only when the source supplies a complete
//! [`TeacherIdentityV1`](crate::TeacherIdentityV1). A display alias or a
//! tokenizer string is not sufficient provenance for persisted logits.
//!
//! # Storage layout
//!
//! ```text
//! <root>/v3-<teacher-identity-sha256>/<causal-prefix-sha256>/<position>.json
//! ```
//!
//! The prefix digest is a domain-separated SHA-256 over `tokens[..=position]`.
//! Consequently, logits for a causal row remain reusable when only later
//! suffix tokens change. The complete identity and all derived key material
//! are repeated in each strict, canonical JSON entry and checked before a hit.
//! The old v1 and v2 layouts are never inspected and are permanent cold misses.

use std::collections::HashMap;
use std::io::{Error as IoError, ErrorKind, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::TeacherIdentityV1;
use crate::logit_source::{
    LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs,
    validate_full_vocab_logprobs_batch, validate_logit_request, validate_topk_logprob_row,
    validate_topk_logprobs_batch,
};

const CACHE_SCHEMA_V3: &str = "kiln.logit-cache.v3";
const CACHE_PATH_PREFIX_V3: &str = "v3-";
const PREFIX_HASH_DOMAIN_V3: &[u8] = b"kiln.logit-cache.causal-prefix.v3\0";
const MAX_CACHE_ENTRY_BYTES: u64 = 8 * 1024 * 1024;
/// Bound directory traversal for stats/export so an accidentally pointed cache
/// root cannot turn one API request into an unbounded filesystem walk.
pub const MAX_CACHE_SCAN_FILES: usize = 1_000_000;
const MAX_CACHE_SCAN_DIRECTORIES: usize = 1_000_000;
/// Maximum validated source bytes accepted by one archive export.
pub const MAX_CACHE_EXPORT_SOURCE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// One sparse row accepted by the cache write API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

/// Complete v3 file contents. Field order is the canonical JSON order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct StoredCacheEntryV3 {
    schema: String,
    teacher_identity: TeacherIdentityV1,
    teacher_revision_sha256: String,
    prefix_sha256: String,
    token_count: usize,
    position: usize,
    top_k: usize,
    indices: Vec<u32>,
    logprobs: Vec<f32>,
    produced_at: String,
}

impl StoredCacheEntryV3 {
    fn new(
        identity: &TeacherIdentityV1,
        tokens: &[u32],
        position: usize,
        entry: &CacheEntry,
    ) -> Result<Self> {
        validate_cache_row(identity, tokens, position, entry)?;
        Ok(Self {
            schema: CACHE_SCHEMA_V3.to_owned(),
            teacher_identity: identity.clone(),
            teacher_revision_sha256: identity.content_revision(),
            prefix_sha256: hash_prefix(&tokens[..=position]),
            token_count: position + 1,
            position,
            top_k: entry.top_k,
            indices: entry.indices.clone(),
            logprobs: entry.logprobs.clone(),
            produced_at: entry.produced_at.clone(),
        })
    }

    fn as_entry(&self) -> CacheEntry {
        CacheEntry {
            indices: self.indices.clone(),
            logprobs: self.logprobs.clone(),
            top_k: self.top_k,
            produced_at: self.produced_at.clone(),
        }
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).context("serialize canonical v3 cache entry")
    }

    fn validate_self(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema == CACHE_SCHEMA_V3,
            "unsupported cache schema {:?}",
            self.schema
        );
        anyhow::ensure!(
            self.teacher_revision_sha256 == self.teacher_identity.content_revision(),
            "teacher revision does not match the embedded identity"
        );
        anyhow::ensure!(
            is_lower_sha256(&self.prefix_sha256),
            "prefix_sha256 must be 64 lowercase hexadecimal characters"
        );
        anyhow::ensure!(
            self.token_count
                == self
                    .position
                    .checked_add(1)
                    .context("cache position overflow")?,
            "token_count {} does not match causal position {}",
            self.token_count,
            self.position
        );
        validate_stored_row(self)?;
        validate_timestamp(&self.produced_at)?;
        Ok(())
    }

    fn validate_request(
        &self,
        identity: &TeacherIdentityV1,
        tokens: &[u32],
        position: usize,
    ) -> Result<()> {
        self.validate_self()?;
        anyhow::ensure!(
            &self.teacher_identity == identity,
            "embedded teacher identity does not match requested identity"
        );
        anyhow::ensure!(
            self.teacher_revision_sha256 == identity.content_revision(),
            "teacher revision does not match requested identity"
        );
        anyhow::ensure!(
            position < tokens.len(),
            "position {position} is outside token sequence"
        );
        let expected_prefix = hash_prefix(&tokens[..=position]);
        anyhow::ensure!(
            self.prefix_sha256 == expected_prefix,
            "prefix digest does not match requested causal tokens"
        );
        anyhow::ensure!(self.token_count == position + 1, "token count mismatch");
        anyhow::ensure!(self.position == position, "position mismatch");
        Ok(())
    }
}

/// Domain-separated SHA-256 of a causal token prefix, encoded as lowercase
/// hexadecimal. Length and tokens use fixed-width big-endian encodings.
pub fn hash_prefix(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    digest.update(PREFIX_HASH_DOMAIN_V3);
    digest.update((tokens.len() as u64).to_be_bytes());
    for token in tokens {
        digest.update(token.to_be_bytes());
    }
    hex_digest(digest.finalize().as_slice())
}

/// Identity-bound on-disk logit cache.
#[derive(Debug, Clone)]
pub struct LogitCache {
    root: PathBuf,
}

#[derive(Debug)]
struct ValidatedEntry {
    path: PathBuf,
    stored: StoredCacheEntryV3,
    bytes: Vec<u8>,
}

impl LogitCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry_path(
        &self,
        identity: &TeacherIdentityV1,
        tokens: &[u32],
        position: usize,
    ) -> Result<PathBuf> {
        anyhow::ensure!(
            position < tokens.len(),
            "position {position} is outside token sequence"
        );
        Ok(self.path_from_parts(
            &identity.content_revision(),
            &hash_prefix(&tokens[..=position]),
            position,
        ))
    }

    fn path_from_parts(&self, revision: &str, prefix: &str, position: usize) -> PathBuf {
        self.root
            .join(format!("{CACHE_PATH_PREFIX_V3}{revision}"))
            .join(prefix)
            .join(format!("{position}.json"))
    }

    /// Atomically insert a row or upgrade an existing row to a wider support.
    ///
    /// Existing and candidate rows must agree bit-for-bit over their shared
    /// support. Equal or wider existing rows win; a consistent wider candidate
    /// replaces a narrower row. Any conflicting result fails closed.
    pub fn put(
        &self,
        identity: &TeacherIdentityV1,
        tokens: &[u32],
        position: usize,
        entry: &CacheEntry,
    ) -> Result<()> {
        let candidate = StoredCacheEntryV3::new(identity, tokens, position, entry)?;
        let candidate_bytes = candidate.canonical_bytes()?;
        anyhow::ensure!(
            candidate_bytes.len() as u64 <= MAX_CACHE_ENTRY_BYTES,
            "serialized cache entry is {} bytes; maximum is {}",
            candidate_bytes.len(),
            MAX_CACHE_ENTRY_BYTES
        );
        let path = self.entry_path(identity, tokens, position)?;
        validate_existing_path_bound(&path)?;

        kiln_resource::locked_update(&path, |existing| {
            let Some(existing_bytes) = existing else {
                return Ok(candidate_bytes.clone());
            };
            let existing = parse_canonical_entry(existing_bytes).map_err(invalid_data)?;
            existing
                .validate_request(identity, tokens, position)
                .map_err(invalid_data)?;
            merge_entries(&existing, &candidate).map_err(invalid_data)
        })
        .with_context(|| format!("atomically update cache entry {}", path.display()))?;
        Ok(())
    }

    /// Read and validate one row. Missing v3 files are cache misses; malformed
    /// or misplaced v3 files are errors, never misses.
    pub fn get(
        &self,
        identity: &TeacherIdentityV1,
        tokens: &[u32],
        position: usize,
    ) -> Result<Option<CacheEntry>> {
        let path = self.entry_path(identity, tokens, position)?;
        let bytes = match read_bounded_regular_file(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("read cache entry {}", path.display()));
            }
        };
        let stored = parse_canonical_entry(&bytes)
            .with_context(|| format!("parse canonical cache entry {}", path.display()))?;
        stored
            .validate_request(identity, tokens, position)
            .with_context(|| format!("validate cache entry {}", path.display()))?;
        anyhow::ensure!(
            self.canonical_path_for_stored(&stored) == path,
            "cache entry metadata derives to a different path"
        );
        Ok(Some(stored.as_entry()))
    }

    fn canonical_path_for_stored(&self, stored: &StoredCacheEntryV3) -> PathBuf {
        self.path_from_parts(
            &stored.teacher_revision_sha256,
            &stored.prefix_sha256,
            stored.position,
        )
    }

    /// Walk and validate all v3 entries before reporting statistics. Legacy
    /// layouts and lock/temp files are ignored.
    pub fn stats(&self) -> Result<CacheStats> {
        let mut stats = CacheStats::default();
        for path in self.validated_entry_paths()? {
            let entry = self.read_validated_entry(path)?;
            anyhow::ensure!(
                self.canonical_path_for_stored(&entry.stored) == entry.path,
                "cache entry {} is misplaced",
                entry.path.display()
            );
            stats.total_entries += 1;
            stats.total_bytes = stats.total_bytes.saturating_add(entry.bytes.len() as u64);
            *stats
                .per_teacher
                .entry(entry.stored.teacher_revision_sha256)
                .or_insert(0) += 1;
        }
        Ok(stats)
    }

    /// Export only strict, canonical, correctly placed v3 regular files.
    pub fn export_to_tar(&self, output: &Path) -> Result<u64> {
        if !self.root.exists() {
            anyhow::bail!("cache root {} does not exist", self.root.display());
        }
        ensure_export_outside_cache_root(&self.root, output)?;
        let paths = self.validated_entry_paths()?;
        write_validated_archive(self, output, paths, MAX_CACHE_EXPORT_SOURCE_BYTES)
    }

    /// Direct archive extraction is deliberately disabled. A future importer
    /// must stage every member, validate canonical v3 contents and derived
    /// paths, then merge via [`Self::put`].
    pub fn import_from_tar(&self, _input: &Path) -> Result<usize> {
        anyhow::bail!(
            "logit cache archive import is disabled: no staged v3 validation path is implemented"
        )
    }

    fn validated_entry_paths(&self) -> Result<Vec<PathBuf>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut paths = walk_regular_files(&self.root)?;
        paths.sort();
        paths.retain(|path| {
            let Ok(relative) = path.strip_prefix(&self.root) else {
                return false;
            };
            looks_like_v3_entry_path(relative)
        });
        Ok(paths)
    }

    fn read_validated_entry(&self, path: PathBuf) -> Result<ValidatedEntry> {
        let bytes = read_bounded_regular_file(&path)
            .with_context(|| format!("read bounded cache entry {}", path.display()))?;
        let stored = parse_canonical_entry(&bytes)
            .with_context(|| format!("parse canonical cache entry {}", path.display()))?;
        stored
            .validate_self()
            .with_context(|| format!("validate cache entry {}", path.display()))?;
        anyhow::ensure!(
            self.canonical_path_for_stored(&stored) == path,
            "cache entry {} is misplaced",
            path.display()
        );
        Ok(ValidatedEntry {
            path,
            stored,
            bytes,
        })
    }
}

fn ensure_export_outside_cache_root(root: &Path, output: &Path) -> Result<()> {
    let canonical_root = std::fs::canonicalize(root)
        .with_context(|| format!("canonicalize cache root {}", root.display()))?;
    let canonical_output = if output.exists() {
        std::fs::canonicalize(output)
            .with_context(|| format!("canonicalize cache export output {}", output.display()))?
    } else {
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        let canonical_parent = std::fs::canonicalize(parent)
            .with_context(|| format!("canonicalize cache export parent {}", parent.display()))?;
        let name = output
            .file_name()
            .context("cache export output must name a file")?;
        canonical_parent.join(name)
    };
    anyhow::ensure!(
        !canonical_output.starts_with(&canonical_root),
        "cache export output {} must be outside cache root {}",
        output.display(),
        root.display()
    );
    Ok(())
}

fn write_validated_archive(
    cache: &LogitCache,
    output: &Path,
    paths: Vec<PathBuf>,
    max_source_bytes: u64,
) -> Result<u64> {
    let file = std::fs::File::create(output)
        .with_context(|| format!("create tar output {}", output.display()))?;
    let gz = flate2::GzBuilder::new()
        .mtime(0)
        .write(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(gz);
    let mut source_bytes = 0u64;
    for path in paths {
        let entry = cache.read_validated_entry(path)?;
        source_bytes = source_bytes
            .checked_add(entry.bytes.len() as u64)
            .context("cache export source byte count overflow")?;
        anyhow::ensure!(
            source_bytes <= max_source_bytes,
            "cache export source is larger than the {} byte limit",
            max_source_bytes
        );
        let relative = entry
            .path
            .strip_prefix(&cache.root)
            .with_context(|| format!("derive relative cache path {}", entry.path.display()))?;
        let archive_path = Path::new("logit-cache").join(relative);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_size(entry.bytes.len() as u64);
        header.set_mode(0o644);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_cksum();
        archive
            .append_data(&mut header, &archive_path, entry.bytes.as_slice())
            .with_context(|| format!("append validated cache entry {}", entry.path.display()))?;
    }
    let writer = archive.into_inner().context("finalize cache tar")?;
    let output_file = writer.finish().context("finish cache gzip stream")?;
    Ok(output_file
        .metadata()
        .context("stat completed cache gzip stream")?
        .len())
}

/// Cache statistics. `per_teacher` is keyed by authoritative identity revision,
/// not a mutable display alias.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_bytes: u64,
    pub per_teacher: std::collections::BTreeMap<String, usize>,
}

/// A source wrapper that reads and writes identity-bound sparse rows.
#[derive(Debug)]
pub struct CachedLogitSource<Inner> {
    cache: LogitCache,
    inner: Inner,
    identity: TeacherIdentityV1,
}

impl<Inner> CachedLogitSource<Inner>
where
    Inner: LogitSource,
{
    /// Construct a cache only for a source with authoritative provenance.
    pub fn new(cache: LogitCache, inner: Inner) -> Result<Self, LogitSourceError> {
        let caps = inner.capabilities();
        let identity = inner
            .authoritative_teacher_identity()
            .cloned()
            .ok_or_else(|| {
                LogitSourceError::invalid(
                    &caps.teacher_id,
                    "persistent logit caching requires an authoritative teacher identity",
                )
            })?;
        validate_identity_caps(&caps, &identity)?;
        Ok(Self {
            cache,
            inner,
            identity,
        })
    }
}

impl<Inner: LogitSource + std::fmt::Debug> LogitSource for CachedLogitSource<Inner> {
    fn capabilities(&self) -> LogitSourceCaps {
        self.inner.capabilities()
    }

    fn authoritative_teacher_identity(&self) -> Option<&TeacherIdentityV1> {
        Some(&self.identity)
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let caps = self.inner.capabilities();
        validate_identity_caps(&caps, &self.identity)?;
        if self.inner.authoritative_teacher_identity() != Some(&self.identity) {
            return Err(LogitSourceError::invalid(
                &caps.teacher_id,
                "wrapped source authoritative identity changed after cache construction",
            ));
        }
        validate_logit_request(&caps, tokens, positions, top_k)?;
        let Some(requested_k) = top_k else {
            let batch = self.inner.fetch_logprobs(tokens, positions, None)?;
            validate_full_vocab_logprobs_batch(&caps, tokens, positions, &batch)?;
            return Ok(batch);
        };
        let flat_capacity = positions.len().checked_mul(requested_k).ok_or_else(|| {
            LogitSourceError::invalid(&caps.teacher_id, "top-K response shape overflows usize")
        })?;

        #[derive(Debug, Clone)]
        struct SparseRow {
            indices: Vec<u32>,
            logprobs: Vec<f32>,
        }

        let mut result_slots: Vec<Option<SparseRow>> = vec![None; positions.len()];
        let mut resolved_rows: HashMap<usize, SparseRow> = HashMap::new();
        let mut miss_index_by_position: HashMap<usize, usize> = HashMap::new();
        let mut miss_positions = Vec::new();
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
                .get(&self.identity, tokens, position)
                .map_err(|error| {
                    LogitSourceError::invalid(
                        &caps.teacher_id,
                        format!("read cache position {position}: {error:#}"),
                    )
                })?;
            if let Some(entry) = cached
                && entry.top_k >= requested_k
            {
                let row = SparseRow {
                    indices: entry.indices[..requested_k].to_vec(),
                    logprobs: entry.logprobs[..requested_k].to_vec(),
                };
                validate_topk_logprob_row(
                    &caps,
                    requested_k,
                    slot_index,
                    &row.indices,
                    &row.logprobs,
                )?;
                result_slots[slot_index] = Some(row.clone());
                resolved_rows.insert(position, row);
                continue;
            }

            let miss_index = miss_positions.len();
            miss_positions.push(position);
            miss_slots.push(vec![slot_index]);
            miss_index_by_position.insert(position, miss_index);
        }

        let mut new_entries = Vec::with_capacity(miss_positions.len());
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
                new_entries.push((
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

        for (position, entry) in &new_entries {
            self.cache
                .put(&self.identity, tokens, *position, entry)
                .map_err(|error| {
                    LogitSourceError::invalid(
                        &caps.teacher_id,
                        format!("write cache position {position}: {error:#}"),
                    )
                })?;
        }
        Ok(LogprobBatch::TopK(output))
    }
}

fn validate_identity_caps(
    caps: &LogitSourceCaps,
    identity: &TeacherIdentityV1,
) -> Result<(), LogitSourceError> {
    let invalid = |message| LogitSourceError::invalid(&caps.teacher_id, message);
    if caps.vocab_size != identity.vocab_size() as usize {
        return Err(invalid(format!(
            "source vocab_size {} does not match authoritative identity vocab_size {}",
            caps.vocab_size,
            identity.vocab_size()
        )));
    }
    if caps.max_top_k > identity.max_top_k() as usize {
        return Err(invalid(format!(
            "source max_top_k {} exceeds authoritative identity max_top_k {}",
            caps.max_top_k,
            identity.max_top_k()
        )));
    }
    if let Some(tokenizer_hash) = &caps.tokenizer_hash
        && tokenizer_hash != identity.tokenizer_vocab_sha256()
    {
        return Err(invalid(format!(
            "source tokenizer hash {:?} does not match authoritative identity tokenizer vocabulary hash {:?}",
            tokenizer_hash,
            identity.tokenizer_vocab_sha256()
        )));
    }
    Ok(())
}

fn validate_cache_row(
    identity: &TeacherIdentityV1,
    tokens: &[u32],
    position: usize,
    entry: &CacheEntry,
) -> Result<()> {
    anyhow::ensure!(
        position < tokens.len(),
        "position {position} is outside token sequence"
    );
    validate_timestamp(&entry.produced_at)?;
    let caps = identity_caps(identity);
    validate_topk_logprob_row(
        &caps,
        entry.top_k,
        position,
        &entry.indices,
        &entry.logprobs,
    )
    .map_err(anyhow::Error::from)
}

fn validate_stored_row(stored: &StoredCacheEntryV3) -> Result<()> {
    let caps = identity_caps(&stored.teacher_identity);
    validate_topk_logprob_row(
        &caps,
        stored.top_k,
        stored.position,
        &stored.indices,
        &stored.logprobs,
    )
    .map_err(anyhow::Error::from)
}

fn identity_caps(identity: &TeacherIdentityV1) -> LogitSourceCaps {
    LogitSourceCaps {
        teacher_id: identity.content_revision(),
        vocab_size: identity.vocab_size() as usize,
        max_top_k: identity.max_top_k() as usize,
        supports_full_vocab: false,
        supports_batched: true,
        tokenizer_hash: Some(identity.tokenizer_vocab_sha256().to_owned()),
    }
}

fn validate_timestamp(value: &str) -> Result<()> {
    anyhow::ensure!(!value.is_empty(), "produced_at must not be empty");
    chrono::DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("produced_at {value:?} is not RFC 3339"))?;
    Ok(())
}

fn parse_canonical_entry(bytes: &[u8]) -> Result<StoredCacheEntryV3> {
    let stored: StoredCacheEntryV3 =
        serde_json::from_slice(bytes).context("deserialize strict v3 cache entry")?;
    stored.validate_self()?;
    anyhow::ensure!(
        stored.canonical_bytes()?.as_slice() == bytes,
        "cache entry is not canonical compact JSON"
    );
    Ok(stored)
}

fn merge_entries(existing: &StoredCacheEntryV3, candidate: &StoredCacheEntryV3) -> Result<Vec<u8>> {
    anyhow::ensure!(
        existing.teacher_identity == candidate.teacher_identity
            && existing.teacher_revision_sha256 == candidate.teacher_revision_sha256
            && existing.prefix_sha256 == candidate.prefix_sha256
            && existing.token_count == candidate.token_count
            && existing.position == candidate.position,
        "cannot merge entries with different cache identities"
    );
    let shared = existing.top_k.min(candidate.top_k);
    let indices_match = existing.indices[..shared] == candidate.indices[..shared];
    let logprobs_match = existing.logprobs[..shared]
        .iter()
        .zip(&candidate.logprobs[..shared])
        .all(|(left, right)| left.to_bits() == right.to_bits());
    anyhow::ensure!(
        indices_match && logprobs_match,
        "conflicting logits for the same teacher identity and causal prefix"
    );
    if existing.top_k >= candidate.top_k {
        existing.canonical_bytes()
    } else {
        candidate.canonical_bytes()
    }
}

fn looks_like_v3_entry_path(relative: &Path) -> bool {
    let components = relative.components().collect::<Vec<_>>();
    components.len() == 3
        && components[0]
            .as_os_str()
            .to_str()
            .is_some_and(|value| value.starts_with(CACHE_PATH_PREFIX_V3))
        && components[2]
            .as_os_str()
            .to_str()
            .is_some_and(|value| value.ends_with(".json"))
}

fn walk_regular_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut directories = 0usize;
    while let Some(path) = stack.pop() {
        directories = directories
            .checked_add(1)
            .context("cache directory count overflow")?;
        anyhow::ensure!(
            directories <= MAX_CACHE_SCAN_DIRECTORIES,
            "cache scan exceeds the {} directory limit",
            MAX_CACHE_SCAN_DIRECTORIES
        );
        for entry in
            std::fs::read_dir(&path).with_context(|| format!("read_dir {}", path.display()))?
        {
            let entry = entry?;
            let child = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(child);
            } else if file_type.is_file() {
                files.push(child);
                anyhow::ensure!(
                    files.len() <= MAX_CACHE_SCAN_FILES,
                    "cache scan exceeds the {} file limit",
                    MAX_CACHE_SCAN_FILES
                );
            }
        }
    }
    Ok(files)
}

fn validate_existing_path_bound(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("stat cache entry {}", path.display()));
        }
    };
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "cache entry {} is not a regular file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= MAX_CACHE_ENTRY_BYTES,
        "cache entry {} is {} bytes; maximum is {}",
        path.display(),
        metadata.len(),
        MAX_CACHE_ENTRY_BYTES
    );
    Ok(())
}

fn read_bounded_regular_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let path_metadata = std::fs::symlink_metadata(path)?;
    if !path_metadata.file_type().is_file() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("cache entry {} is not a regular file", path.display()),
        ));
    }
    if path_metadata.len() > MAX_CACHE_ENTRY_BYTES {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "cache entry {} is {} bytes; maximum is {}",
                path.display(),
                path_metadata.len(),
                MAX_CACHE_ENTRY_BYTES
            ),
        ));
    }

    let file = std::fs::File::open(path)?;
    let opened_metadata = file.metadata()?;
    if !opened_metadata.file_type().is_file() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "cache entry {} did not open as a regular file",
                path.display()
            ),
        ));
    }
    if opened_metadata.len() > MAX_CACHE_ENTRY_BYTES {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "cache entry {} is {} bytes; maximum is {}",
                path.display(),
                opened_metadata.len(),
                MAX_CACHE_ENTRY_BYTES
            ),
        ));
    }

    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.take(MAX_CACHE_ENTRY_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_CACHE_ENTRY_BYTES {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "cache entry {} exceeded the {} byte maximum while reading",
                path.display(),
                MAX_CACHE_ENTRY_BYTES
            ),
        ));
    }
    if bytes.len() as u64 != opened_metadata.len() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("cache entry {} changed size while reading", path.display()),
        ));
    }
    Ok(bytes)
}

fn invalid_data(error: anyhow::Error) -> IoError {
    IoError::new(ErrorKind::InvalidData, format!("{error:#}"))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};
    use tempfile::tempdir;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn identity(model: &str, base: &str) -> TeacherIdentityV1 {
        TeacherIdentityV1::new(model, base, B, C, None, 64, 8, 4096, 64, "test/1", D).unwrap()
    }

    fn valid_entry(position: usize, top_k: usize) -> CacheEntry {
        CacheEntry {
            indices: (0..top_k)
                .map(|candidate| (position * 8 + candidate) as u32)
                .collect(),
            logprobs: (0..top_k)
                .map(|candidate| -2.0 - candidate as f32)
                .collect(),
            top_k,
            produced_at: "2026-05-15T00:00:00Z".into(),
        }
    }

    #[derive(Debug, Clone)]
    struct ScriptedSource {
        caps: LogitSourceCaps,
        identity: Option<TeacherIdentityV1>,
        calls: Arc<Mutex<Vec<Vec<usize>>>>,
    }

    impl ScriptedSource {
        fn new(alias: &str, identity: Option<TeacherIdentityV1>) -> Self {
            Self {
                caps: LogitSourceCaps {
                    teacher_id: alias.into(),
                    vocab_size: 64,
                    max_top_k: 8,
                    supports_full_vocab: false,
                    supports_batched: true,
                    tokenizer_hash: Some(B.into()),
                },
                identity,
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

        fn authoritative_teacher_identity(&self) -> Option<&TeacherIdentityV1> {
            self.identity.as_ref()
        }

        fn fetch_logprobs(
            &self,
            _tokens: &[u32],
            positions: &[usize],
            top_k: Option<usize>,
        ) -> Result<LogprobBatch, LogitSourceError> {
            let top_k = top_k.ok_or_else(|| {
                LogitSourceError::invalid(&self.caps.teacher_id, "test source is sparse")
            })?;
            self.calls.lock().unwrap().push(positions.to_vec());
            let mut indices = Vec::new();
            let mut logprobs = Vec::new();
            for position in positions {
                let row = valid_entry(*position, top_k);
                indices.extend(row.indices);
                logprobs.extend(row.logprobs);
            }
            Ok(LogprobBatch::TopK(TopKLogprobs {
                indices,
                logprobs,
                top_k,
            }))
        }
    }

    fn topk(batch: LogprobBatch) -> TopKLogprobs {
        match batch {
            LogprobBatch::TopK(batch) => batch,
            LogprobBatch::FullVocab { .. } => panic!("expected top-K"),
        }
    }

    #[test]
    fn prefix_hash_is_domain_separated_fixed_width_sha256() {
        assert_eq!(hash_prefix(&[]).len(), 64);
        assert!(is_lower_sha256(&hash_prefix(&[1, 2, 3])));
        assert_ne!(hash_prefix(&[1, 2]), hash_prefix(&[1, 2, 0]));
        assert_ne!(hash_prefix(&[1, 23]), hash_prefix(&[12, 3]));
        assert_eq!(
            hash_prefix(&[1, 2, 3]),
            "b7406ca42b3cd294d001cfc8549fc986fa3590b0fb3ca7101a1e97eade910ccf"
        );
    }

    #[test]
    fn v3_put_get_round_trip_and_path_are_identity_bound() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [10, 20, 30, 40];
        cache
            .put(&identity, &tokens, 2, &valid_entry(2, 4))
            .unwrap();

        let restored = cache.get(&identity, &tokens, 2).unwrap().unwrap();
        assert_eq!(restored, valid_entry(2, 4));
        let path = cache.entry_path(&identity, &tokens, 2).unwrap();
        let relative = path.strip_prefix(dir.path()).unwrap();
        assert_eq!(relative.components().count(), 3);
        assert!(relative.to_string_lossy().starts_with("v3-"));
    }

    #[test]
    fn every_semantic_identity_field_mutation_is_a_cold_miss() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let first = identity("served", A);
        let tokens = [1, 2, 3];
        cache.put(&first, &tokens, 1, &valid_entry(1, 2)).unwrap();

        let variants = vec![
            TeacherIdentityV1::new("served-2", A, B, C, None, 64, 8, 4096, 64, "test/1", D),
            TeacherIdentityV1::new("served", D, B, C, None, 64, 8, 4096, 64, "test/1", D),
            TeacherIdentityV1::new("served", A, A, C, None, 64, 8, 4096, 64, "test/1", D),
            TeacherIdentityV1::new("served", A, B, D, None, 64, 8, 4096, 64, "test/1", D),
            TeacherIdentityV1::new(
                "served",
                A,
                B,
                C,
                Some(crate::TeacherAdapterIdentityV1::new("adapter", A, B).unwrap()),
                64,
                8,
                4096,
                64,
                "test/1",
                D,
            ),
            TeacherIdentityV1::new("served", A, B, C, None, 65, 8, 4096, 64, "test/1", D),
            TeacherIdentityV1::new("served", A, B, C, None, 64, 7, 4096, 64, "test/1", D),
            TeacherIdentityV1::new("served", A, B, C, None, 64, 8, 4097, 64, "test/1", D),
            TeacherIdentityV1::new("served", A, B, C, None, 64, 8, 4096, 65, "test/1", D),
            TeacherIdentityV1::new("served", A, B, C, None, 64, 8, 4096, 64, "test/2", D),
            TeacherIdentityV1::new("served", A, B, C, None, 64, 8, 4096, 64, "test/1", A),
        ];
        for changed in variants {
            let changed = changed.unwrap();
            assert_ne!(first.content_revision(), changed.content_revision());
            assert!(
                cache.get(&changed, &tokens, 1).unwrap().is_none(),
                "identity mutation selected the original cache entry: {}",
                changed.canonical_json()
            );
        }
    }

    #[test]
    fn alias_changes_do_not_prevent_identity_reuse() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("immutable-served-model", A);
        let tokens = [1, 2, 3, 4];

        let first_inner = ScriptedSource::new("registration-alias-a", Some(identity.clone()));
        let first_calls = first_inner.clone();
        let first = CachedLogitSource::new(cache.clone(), first_inner).unwrap();
        first.fetch_logprobs(&tokens, &[1], Some(2)).unwrap();
        assert_eq!(first_calls.calls(), vec![vec![1]]);

        let second_inner = ScriptedSource::new("registration-alias-b", Some(identity));
        let second_calls = second_inner.clone();
        let second = CachedLogitSource::new(cache, second_inner).unwrap();
        second.fetch_logprobs(&tokens, &[1], Some(2)).unwrap();
        assert!(second_calls.calls().is_empty());
    }

    #[test]
    fn causal_prefix_reuses_rows_across_suffix_changes() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let source_inner = ScriptedSource::new("alias", Some(identity));
        let calls = source_inner.clone();
        let source = CachedLogitSource::new(cache, source_inner).unwrap();

        source.fetch_logprobs(&[1, 2, 3, 4], &[1], Some(2)).unwrap();
        source
            .fetch_logprobs(&[1, 2, 9, 9, 9], &[1], Some(2))
            .unwrap();
        assert_eq!(calls.calls(), vec![vec![1]]);
    }

    #[test]
    fn changing_the_causal_prefix_is_a_miss() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let inner = ScriptedSource::new("alias", Some(identity));
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache, inner).unwrap();

        source.fetch_logprobs(&[1, 2, 3], &[1], Some(2)).unwrap();
        source.fetch_logprobs(&[1, 7, 3], &[1], Some(2)).unwrap();
        assert_eq!(calls.calls(), vec![vec![1], vec![1]]);
    }

    #[test]
    fn missing_authoritative_identity_is_rejected_at_construction() {
        let dir = tempdir().unwrap();
        let error = CachedLogitSource::new(
            LogitCache::new(dir.path()),
            ScriptedSource::new("alias", None),
        )
        .unwrap_err();
        assert!(error.to_string().contains("authoritative teacher identity"));
    }

    #[test]
    fn cached_source_forwards_the_exact_authoritative_identity() {
        let dir = tempdir().unwrap();
        let identity = identity("served", A);
        let source = CachedLogitSource::new(
            LogitCache::new(dir.path()),
            ScriptedSource::new("alias", Some(identity.clone())),
        )
        .unwrap();
        assert_eq!(source.authoritative_teacher_identity(), Some(&identity));
    }

    #[test]
    fn wider_rows_upgrade_and_narrower_rows_never_downgrade() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 2))
            .unwrap();
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 4))
            .unwrap();
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 3))
            .unwrap();
        assert_eq!(cache.get(&identity, &tokens, 1).unwrap().unwrap().top_k, 4);
    }

    #[test]
    fn conflicting_rows_are_rejected_without_clobbering() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        let original = valid_entry(1, 2);
        cache.put(&identity, &tokens, 1, &original).unwrap();
        let mut conflicting = valid_entry(1, 4);
        conflicting.logprobs[0] = -3.0;
        let error = cache.put(&identity, &tokens, 1, &conflicting).unwrap_err();
        assert!(error.to_string().contains("atomically update cache entry"));
        assert_eq!(cache.get(&identity, &tokens, 1).unwrap().unwrap(), original);
    }

    #[test]
    fn concurrent_consistent_upgrades_converge_to_the_widest_row() {
        let dir = tempdir().unwrap();
        let cache = Arc::new(LogitCache::new(dir.path()));
        let identity = identity("served", A);
        let tokens = Arc::new(vec![1, 2, 3]);
        let barrier = Arc::new(Barrier::new(8));
        let handles = (1..=8)
            .map(|top_k| {
                let cache = Arc::clone(&cache);
                let identity = identity.clone();
                let tokens = Arc::clone(&tokens);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    cache
                        .put(&identity, &tokens, 1, &valid_entry(1, top_k))
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(cache.get(&identity, &tokens, 1).unwrap().unwrap().top_k, 8);
        let path = cache.entry_path(&identity, &tokens, 1).unwrap();
        parse_canonical_entry(&std::fs::read(path).unwrap()).unwrap();
    }

    #[test]
    fn concurrent_conflict_has_one_winner_and_no_torn_file() {
        let dir = tempdir().unwrap();
        let cache = Arc::new(LogitCache::new(dir.path()));
        let identity = identity("served", A);
        let tokens = Arc::new(vec![1, 2, 3]);
        let barrier = Arc::new(Barrier::new(2));
        let handles = [0u32, 1u32]
            .into_iter()
            .map(|offset| {
                let cache = Arc::clone(&cache);
                let identity = identity.clone();
                let tokens = Arc::clone(&tokens);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut row = valid_entry(1, 2);
                    row.indices[0] += offset * 20;
                    barrier.wait();
                    cache.put(&identity, &tokens, 1, &row)
                })
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        cache.get(&identity, &tokens, 1).unwrap().unwrap();
    }

    #[test]
    fn corrupt_unknown_nonfinite_and_noncanonical_entries_fail_closed() {
        let variants = ["unknown", "nonfinite", "noncanonical", "truncated"];
        for variant in variants {
            let dir = tempdir().unwrap();
            let cache = LogitCache::new(dir.path());
            let identity = identity("served", A);
            let tokens = [1, 2, 3];
            cache
                .put(&identity, &tokens, 1, &valid_entry(1, 2))
                .unwrap();
            let path = cache.entry_path(&identity, &tokens, 1).unwrap();
            let mut bytes = std::fs::read(&path).unwrap();
            match variant {
                "unknown" => {
                    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("extra".into(), true.into());
                    bytes = serde_json::to_vec(&value).unwrap();
                }
                "nonfinite" => {
                    let text = String::from_utf8(bytes).unwrap();
                    bytes = text.replacen("-2.0", "1e400", 1).into_bytes();
                }
                "noncanonical" => bytes.push(b'\n'),
                "truncated" => bytes.truncate(bytes.len() / 2),
                _ => unreachable!(),
            }
            std::fs::write(path, bytes).unwrap();
            assert!(cache.get(&identity, &tokens, 1).is_err(), "{variant}");
        }
    }

    #[test]
    fn oversized_entries_are_rejected_before_reading_or_parsing() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        let path = cache.entry_path(&identity, &tokens, 1).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, vec![b'x'; MAX_CACHE_ENTRY_BYTES as usize + 1]).unwrap();

        let get_error = cache.get(&identity, &tokens, 1).unwrap_err();
        assert!(format!("{get_error:#}").contains("maximum"));
        let stats_error = cache.stats().unwrap_err();
        assert!(stats_error.to_string().contains("bounded cache entry"));
        assert!(
            cache
                .put(&identity, &tokens, 1, &valid_entry(1, 2))
                .unwrap_err()
                .to_string()
                .contains("maximum")
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_cache_entries_are_not_read_or_exported() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        let stored = StoredCacheEntryV3::new(&identity, &tokens, 1, &valid_entry(1, 2)).unwrap();
        let outside_path = outside.path().join("entry.json");
        std::fs::write(&outside_path, stored.canonical_bytes().unwrap()).unwrap();
        let cache_path = cache.entry_path(&identity, &tokens, 1).unwrap();
        std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        symlink(&outside_path, &cache_path).unwrap();

        assert!(cache.get(&identity, &tokens, 1).is_err());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
        let archive_path = outside.path().join("cache.tar.gz");
        cache.export_to_tar(&archive_path).unwrap();
        let decoder = flate2::read::GzDecoder::new(std::fs::File::open(archive_path).unwrap());
        assert_eq!(tar::Archive::new(decoder).entries().unwrap().count(), 0);
    }

    #[test]
    fn misplaced_entry_fails_stats_and_export_validation() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 2))
            .unwrap();
        let canonical = cache.entry_path(&identity, &tokens, 1).unwrap();
        let misplaced = canonical.with_file_name("2.json");
        std::fs::rename(&canonical, &misplaced).unwrap();
        assert!(cache.stats().is_err());
        assert!(cache.export_to_tar(&dir.path().join("bad.tar.gz")).is_err());
    }

    #[test]
    fn v1_and_v2_files_are_permanent_cold_misses_and_ignored_by_stats() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        for path in [
            dir.path().join("teacher/tokenizer/deadbeef/1.json"),
            dir.path().join(format!("v2-{A}/v2-{B}/deadbeef/1.json")),
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"{\"top_k\":2}").unwrap();
        }
        assert!(cache.get(&identity, &tokens, 1).unwrap().is_none());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }

    #[test]
    fn export_contains_only_validated_v3_entries_and_import_fails_closed() {
        let dir = tempdir().unwrap();
        let output = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 2))
            .unwrap();
        let legacy = dir.path().join("legacy/entry.json");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, b"legacy").unwrap();

        let archive_path = output.path().join("cache-a.tar.gz");
        let second_archive_path = output.path().join("cache-b.tar.gz");
        assert!(cache.export_to_tar(&archive_path).unwrap() > 0);
        assert!(cache.export_to_tar(&second_archive_path).unwrap() > 0);
        assert_eq!(
            std::fs::read(&archive_path).unwrap(),
            std::fs::read(&second_archive_path).unwrap(),
            "cache exports must be byte-deterministic"
        );
        let file = std::fs::File::open(&archive_path).unwrap();
        let decoder = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(decoder);
        let mut entries = archive.entries().unwrap();
        let entry = entries.next().unwrap().unwrap();
        assert!(
            entry
                .path()
                .unwrap()
                .to_string_lossy()
                .starts_with("logit-cache/v3-")
        );
        assert_eq!(entry.header().uid().unwrap(), 0);
        assert_eq!(entry.header().gid().unwrap(), 0);
        assert_eq!(entry.header().mtime().unwrap(), 0);
        assert_eq!(entry.header().mode().unwrap(), 0o644);
        assert!(entries.next().is_none());
        assert!(cache.import_from_tar(&archive_path).is_err());
    }

    #[test]
    fn export_refuses_to_write_inside_the_live_cache_root() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let error = cache
            .export_to_tar(&dir.path().join("export.tar.gz"))
            .unwrap_err();
        assert!(error.to_string().contains("must be outside cache root"));
        assert!(!dir.path().join("export.tar.gz").exists());
    }

    #[test]
    fn archive_writer_revalidates_after_path_discovery() {
        let dir = tempdir().unwrap();
        let output = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 2))
            .unwrap();
        let paths = cache.validated_entry_paths().unwrap();
        std::fs::write(&paths[0], b"substituted after discovery").unwrap();

        let archive_path = output.path().join("snapshot.tar.gz");
        let error =
            write_validated_archive(&cache, &archive_path, paths, MAX_CACHE_EXPORT_SOURCE_BYTES)
                .unwrap_err();
        assert!(format!("{error:#}").contains("parse canonical cache entry"));
    }

    #[test]
    fn archive_writer_enforces_source_byte_limit_before_appending_overflow() {
        let dir = tempdir().unwrap();
        let output = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 2))
            .unwrap();
        let paths = cache.validated_entry_paths().unwrap();
        let entry_bytes = std::fs::metadata(&paths[0]).unwrap().len();

        let error = write_validated_archive(
            &cache,
            &output.path().join("bounded.tar.gz"),
            paths,
            entry_bytes - 1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("source is larger"));
    }

    #[test]
    fn cached_source_handles_mixed_hits_misses_duplicates_and_wider_hits() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3, 4];
        cache
            .put(&identity, &tokens, 1, &valid_entry(1, 4))
            .unwrap();
        let inner = ScriptedSource::new("alias", Some(identity));
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache, inner).unwrap();

        let batch = topk(source.fetch_logprobs(&tokens, &[2, 1, 2], Some(2)).unwrap());
        assert_eq!(batch.indices, vec![16, 17, 8, 9, 16, 17]);
        assert_eq!(calls.calls(), vec![vec![2]]);
    }

    #[test]
    fn cached_source_propagates_conflicting_write_through_rows() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2, 3];
        let mut narrow = valid_entry(1, 1);
        narrow.logprobs[0] = -3.0;
        cache.put(&identity, &tokens, 1, &narrow).unwrap();

        let inner = ScriptedSource::new("alias", Some(identity));
        let calls = inner.clone();
        let source = CachedLogitSource::new(cache.clone(), inner).unwrap();
        let error = source.fetch_logprobs(&tokens, &[1], Some(2)).unwrap_err();
        assert!(error.to_string().contains("write cache position 1"));
        assert!(error.to_string().contains("conflicting logits"));
        assert_eq!(calls.calls(), vec![vec![1]]);
        assert_eq!(
            cache
                .get(source.authoritative_teacher_identity().unwrap(), &tokens, 1)
                .unwrap()
                .unwrap(),
            narrow
        );
    }

    #[test]
    fn invalid_cache_inputs_are_rejected_before_disk_access() {
        let dir = tempdir().unwrap();
        let cache = LogitCache::new(dir.path());
        let identity = identity("served", A);
        let tokens = [1, 2];
        assert!(
            cache
                .put(&identity, &tokens, 2, &valid_entry(2, 2))
                .is_err()
        );
        let mut bad = valid_entry(1, 2);
        bad.logprobs[0] = f32::NAN;
        assert!(cache.put(&identity, &tokens, 1, &bad).is_err());
        bad = valid_entry(1, 2);
        bad.produced_at.clear();
        assert!(cache.put(&identity, &tokens, 1, &bad).is_err());
        assert_eq!(cache.stats().unwrap().total_entries, 0);
    }
}

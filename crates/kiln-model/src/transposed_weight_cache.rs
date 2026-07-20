use std::fs;
#[cfg(test)]
use std::io::Read;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use memmap2::Mmap;

use crate::forward::transposed_weight_bytes_2d;
use crate::weights::WeightTensor;

const CACHE_VERSION: u32 = 2;
const CACHE_MAGIC: &[u8; 8] = b"KILNTRP1";
const CACHE_PAYLOAD_ALIGN: usize = 8;

static CACHE_WRITE_SUCCEEDED: AtomicU64 = AtomicU64::new(0);
static CACHE_WRITE_FAILED: AtomicU64 = AtomicU64::new(0);
static CACHE_READ_FAILED: AtomicU64 = AtomicU64::new(0);
static CACHE_ENTRY_INVALID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct CacheKey {
    file_path: PathBuf,
    key_bytes: Vec<u8>,
    expected_len: usize,
    shape: [usize; 2],
}

pub(crate) struct CachedTransposedWeightBytes {
    storage: CachedTransposedWeightStorage,
    shape: [usize; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransposedWeightCacheMissPolicy {
    /// The caller is still loading the base model and has not declared readiness.
    PersistBeforeReadiness,
    /// Existing entries may be read, but a miss has no disk-side effects.
    ReadOnly,
}

enum CachedTransposedWeightStorage {
    Owned(Arc<[u8]>),
    Mapped { mmap: Mmap, payload: Range<usize> },
}

impl CachedTransposedWeightBytes {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            CachedTransposedWeightStorage::Owned(bytes) => &bytes[..],
            CachedTransposedWeightStorage::Mapped { mmap, payload } => &mmap[payload.clone()],
        }
    }

    pub(crate) fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

pub(crate) fn transposed_weight_bytes_2d_cached_bytes(
    weight: &WeightTensor,
    miss_policy: TransposedWeightCacheMissPolicy,
) -> Result<CachedTransposedWeightBytes> {
    let Some(cache_key) = cache_key(weight) else {
        let (bytes, shape) = transposed_weight_bytes_2d(weight)?;
        return Ok(CachedTransposedWeightBytes {
            storage: CachedTransposedWeightStorage::Owned(Arc::from(bytes)),
            shape,
        });
    };

    if let Some(bytes) = try_mmap_cached(&cache_key, miss_policy) {
        tracing::debug!(
            tensor = %weight
                .source
                .as_ref()
                .map(|s| s.tensor_name.as_str())
                .unwrap_or("<unknown>"),
            path = %weight
                .source
                .as_ref()
                .map(|s| s.shard_path.display().to_string())
                .unwrap_or_else(|| "<unknown>".to_string()),
            "mapped transposed weight from disk cache"
        );
        return Ok(bytes);
    }

    let (bytes, shape) = transposed_weight_bytes_2d(weight)?;
    Ok(finish_cache_miss(cache_key, miss_policy, bytes, shape))
}

fn finish_cache_miss(
    cache_key: CacheKey,
    miss_policy: TransposedWeightCacheMissPolicy,
    bytes: Vec<u8>,
    shape: [usize; 2],
) -> CachedTransposedWeightBytes {
    let _ = handle_cache_miss(miss_policy, &cache_key, &bytes);
    CachedTransposedWeightBytes {
        storage: CachedTransposedWeightStorage::Owned(Arc::from(bytes)),
        shape,
    }
}

fn cache_key(weight: &WeightTensor) -> Option<CacheKey> {
    let source = weight.source.as_ref()?;
    let [rows, cols]: [usize; 2] = weight.shape.as_slice().try_into().ok()?;
    let elem_size = weight.dtype.size_bytes();
    let expected_len = rows.checked_mul(cols)?.checked_mul(elem_size)?;
    let key_bytes = format!(
        "version={CACHE_VERSION}\npath={}\nsize={}\nmtime_ns={}\ntensor={}\ndtype={}\nshape={},{}\n",
        source.shard_path.display(),
        source.shard_size,
        source.shard_mtime_ns,
        source.tensor_name,
        weight.dtype,
        rows,
        cols
    )
    .into_bytes();
    let file_path = cache_root_dir()
        .join(format!("v{CACHE_VERSION}"))
        .join(format!("{:016x}.bin", fnv1a64(&key_bytes)));
    Some(CacheKey {
        file_path,
        key_bytes,
        expected_len,
        shape: [cols, rows],
    })
}

fn try_mmap_cached(
    cache_key: &CacheKey,
    miss_policy: TransposedWeightCacheMissPolicy,
) -> Option<CachedTransposedWeightBytes> {
    let file = match fs::File::open(&cache_key.file_path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            record_cache_read_failure(cache_key, "open", &err);
            return None;
        }
    };

    // SAFETY: Cache files are immutable after an atomic rename into place.
    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(mmap) => mmap,
        Err(err) => {
            record_cache_read_failure(cache_key, "mmap", &err);
            return None;
        }
    };

    match cached_payload_range(cache_key, &mmap[..]) {
        Ok(payload) => Some(CachedTransposedWeightBytes {
            storage: CachedTransposedWeightStorage::Mapped { mmap, payload },
            shape: cache_key.shape,
        }),
        Err(err) => {
            let invalid_entries = increment_counter(&CACHE_ENTRY_INVALID);
            tracing::debug!(
                path = %cache_key.file_path.display(),
                error = %err,
                invalid_entries,
                "ignoring invalid transposed weight cache entry"
            );
            if miss_policy == TransposedWeightCacheMissPolicy::PersistBeforeReadiness {
                let _ = fs::remove_file(&cache_key.file_path);
            }
            None
        }
    }
}

fn record_cache_read_failure(cache_key: &CacheKey, stage: &'static str, err: &std::io::Error) {
    let read_failures = increment_counter(&CACHE_READ_FAILED);
    tracing::debug!(
        path = %cache_key.file_path.display(),
        error = %err,
        stage,
        read_failures,
        "failed to read optional transposed weight cache entry; using canonical weight"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachePersistence {
    SkippedReadOnly,
    AlreadyPresent,
    Persisted,
    Failed,
}

fn handle_cache_miss(
    miss_policy: TransposedWeightCacheMissPolicy,
    cache_key: &CacheKey,
    data: &[u8],
) -> CachePersistence {
    if miss_policy == TransposedWeightCacheMissPolicy::ReadOnly {
        return CachePersistence::SkippedReadOnly;
    }

    if cache_key.file_path.exists() {
        return CachePersistence::AlreadyPresent;
    }

    match try_write_cached(cache_key, data) {
        Ok(()) => {
            let write_successes = increment_counter(&CACHE_WRITE_SUCCEEDED);
            tracing::debug!(
                path = %cache_key.file_path.display(),
                write_successes,
                "persisted transposed weight cache entry"
            );
            CachePersistence::Persisted
        }
        Err(err) => {
            let write_failures = increment_counter(&CACHE_WRITE_FAILED);
            tracing::debug!(
                path = %cache_key.file_path.display(),
                error = %err,
                write_failures,
                "failed to persist transposed weight cache entry; continuing with computed weight"
            );
            CachePersistence::Failed
        }
    }
}

fn increment_counter(counter: &AtomicU64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(1))
        })
        .unwrap_or_else(|current| current)
        .saturating_add(1)
}

#[cfg(test)]
fn read_cached_payload<R: Read>(cache_key: &CacheKey, reader: &mut R) -> Result<Vec<u8>> {
    let mut magic = [0u8; CACHE_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .context("cache entry truncated before magic")?;
    anyhow::ensure!(&magic == CACHE_MAGIC, "cache entry magic mismatch");

    let version = read_u32_le_reader(reader).context("reading cache entry version")?;
    anyhow::ensure!(version == CACHE_VERSION, "cache entry version mismatch");

    let key_len = read_u32_le_reader(reader).context("reading cache entry key length")? as usize;
    anyhow::ensure!(
        key_len == cache_key.key_bytes.len(),
        "cache key length mismatch: expected {}, got {}",
        cache_key.key_bytes.len(),
        key_len
    );
    let mut cached_key = vec![0u8; key_len];
    reader
        .read_exact(&mut cached_key)
        .context("cache entry truncated in key")?;
    anyhow::ensure!(
        cached_key.as_slice() == cache_key.key_bytes.as_slice(),
        "cache key mismatch"
    );

    let data_len_u64 = read_u64_le_reader(reader).context("reading cache entry payload length")?;
    let data_len = usize::try_from(data_len_u64).context("cache payload length overflows usize")?;
    anyhow::ensure!(
        data_len == cache_key.expected_len,
        "cache payload length mismatch: expected {}, got {}",
        cache_key.expected_len,
        data_len
    );

    let payload_offset = CACHE_MAGIC.len() + 4 + 4 + key_len + 8;
    let padding_len = padding_for_alignment(payload_offset, CACHE_PAYLOAD_ALIGN);
    let mut padding = vec![0u8; padding_len];
    reader
        .read_exact(&mut padding)
        .context("cache entry truncated in payload alignment padding")?;
    anyhow::ensure!(
        padding.iter().all(|&byte| byte == 0),
        "cache entry payload alignment padding is non-zero"
    );

    let mut payload = vec![0u8; data_len];
    reader
        .read_exact(&mut payload)
        .context("cache entry truncated in payload")?;

    let mut trailing = [0u8; 1];
    anyhow::ensure!(
        reader
            .read(&mut trailing)
            .context("checking cache entry trailing bytes")?
            == 0,
        "cache entry contains trailing bytes"
    );
    Ok(payload)
}

fn cached_payload_range(cache_key: &CacheKey, bytes: &[u8]) -> Result<Range<usize>> {
    let mut offset = 0usize;
    let magic = take_bytes(
        bytes,
        &mut offset,
        CACHE_MAGIC.len(),
        "cache entry truncated before magic",
    )?;
    anyhow::ensure!(magic == CACHE_MAGIC, "cache entry magic mismatch");

    let version = read_u32_le_slice(bytes, &mut offset, "cache entry truncated before version")?;
    anyhow::ensure!(version == CACHE_VERSION, "cache entry version mismatch");

    let key_len = read_u32_le_slice(
        bytes,
        &mut offset,
        "cache entry truncated before key length",
    )? as usize;
    anyhow::ensure!(
        key_len == cache_key.key_bytes.len(),
        "cache key length mismatch: expected {}, got {}",
        cache_key.key_bytes.len(),
        key_len
    );
    let cached_key = take_bytes(bytes, &mut offset, key_len, "cache entry truncated in key")?;
    anyhow::ensure!(
        cached_key == cache_key.key_bytes.as_slice(),
        "cache key mismatch"
    );

    let data_len_u64 = read_u64_le_slice(
        bytes,
        &mut offset,
        "cache entry truncated before payload length",
    )?;
    let data_len = usize::try_from(data_len_u64).context("cache payload length overflows usize")?;
    anyhow::ensure!(
        data_len == cache_key.expected_len,
        "cache payload length mismatch: expected {}, got {}",
        cache_key.expected_len,
        data_len
    );

    let padding_len = padding_for_alignment(offset, CACHE_PAYLOAD_ALIGN);
    let padding = take_bytes(
        bytes,
        &mut offset,
        padding_len,
        "cache entry truncated in payload alignment padding",
    )?;
    anyhow::ensure!(
        padding.iter().all(|&byte| byte == 0),
        "cache entry payload alignment padding is non-zero"
    );

    let payload_start = offset;
    let payload_end = payload_start
        .checked_add(data_len)
        .context("cache payload range overflow")?;
    anyhow::ensure!(
        payload_end <= bytes.len(),
        "cache entry truncated in payload"
    );
    anyhow::ensure!(
        payload_end == bytes.len(),
        "cache entry contains trailing bytes"
    );
    Ok(payload_start..payload_end)
}

fn take_bytes<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
    truncated: &'static str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .context("cache entry offset overflow")?;
    anyhow::ensure!(end <= bytes.len(), truncated);
    let out = &bytes[*offset..end];
    *offset = end;
    Ok(out)
}

fn read_u32_le_slice(bytes: &[u8], offset: &mut usize, truncated: &'static str) -> Result<u32> {
    let raw = take_bytes(bytes, offset, 4, truncated)?;
    Ok(u32::from_le_bytes(raw.try_into().expect("u32 byte width")))
}

fn read_u64_le_slice(bytes: &[u8], offset: &mut usize, truncated: &'static str) -> Result<u64> {
    let raw = take_bytes(bytes, offset, 8, truncated)?;
    Ok(u64::from_le_bytes(raw.try_into().expect("u64 byte width")))
}

fn padding_for_alignment(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (alignment - (offset & (alignment - 1))) & (alignment - 1)
}

#[cfg(test)]
fn read_u32_le_reader(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(test)]
fn read_u64_le_reader(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn try_write_cached(cache_key: &CacheKey, data: &[u8]) -> Result<()> {
    anyhow::ensure!(
        data.len() == cache_key.expected_len,
        "cache payload length mismatch: expected {}, got {}",
        cache_key.expected_len,
        data.len()
    );

    if let Some(parent) = cache_key.file_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("creating transposed weight cache dir {}", parent.display())
        })?;
    }

    let mut temp_path = cache_key.file_path.clone();
    let suffix = format!(".{}.{}", std::process::id(), unique_nanos());
    let temp_name = temp_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}{suffix}"))
        .unwrap_or_else(|| format!("cache{suffix}"));
    temp_path.set_file_name(temp_name);

    let mut file = match fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp_path)
    {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => return Ok(()),
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "creating transposed weight cache temp file {}",
                    temp_path.display()
                )
            });
        }
    };

    file.write_all(CACHE_MAGIC)
        .context("writing transposed weight cache magic")?;
    file.write_all(&CACHE_VERSION.to_le_bytes())
        .context("writing transposed weight cache version")?;
    file.write_all(&(cache_key.key_bytes.len() as u32).to_le_bytes())
        .context("writing transposed weight cache key length")?;
    file.write_all(&cache_key.key_bytes)
        .context("writing transposed weight cache key")?;
    file.write_all(&(data.len() as u64).to_le_bytes())
        .context("writing transposed weight cache payload length")?;
    let payload_offset = CACHE_MAGIC.len() + 4 + 4 + cache_key.key_bytes.len() + 8;
    let padding_len = padding_for_alignment(payload_offset, CACHE_PAYLOAD_ALIGN);
    if padding_len > 0 {
        file.write_all(&[0u8; CACHE_PAYLOAD_ALIGN][..padding_len])
            .context("writing transposed weight cache payload alignment padding")?;
    }
    file.write_all(data)
        .context("writing transposed weight cache payload")?;
    match fs::rename(&temp_path, &cache_key.file_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&temp_path);
            Ok(())
        }
        Err(err) => {
            let _ = fs::remove_file(&temp_path);
            Err(err).with_context(|| {
                format!(
                    "renaming transposed weight cache {} -> {}",
                    temp_path.display(),
                    cache_key.file_path.display()
                )
            })
        }
    }
}

fn cache_root_dir() -> &'static std::path::Path {
    kiln_resource::application_cache_root()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn unique_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn test_cache_key(expected_len: usize) -> CacheKey {
        CacheKey {
            file_path: PathBuf::from("unused.bin"),
            key_bytes: b"test-key".to_vec(),
            expected_len,
            shape: [2, 2],
        }
    }

    fn cache_entry(cache_key: &CacheKey, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CACHE_MAGIC);
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(cache_key.key_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&cache_key.key_bytes);
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let padding_len = padding_for_alignment(bytes.len(), CACHE_PAYLOAD_ALIGN);
        bytes.resize(bytes.len() + padding_len, 0);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn read_cached_payload_returns_payload_without_trailing_bytes() -> Result<()> {
        let cache_key = test_cache_key(4);
        let payload = [1u8, 2, 3, 4];
        let bytes = cache_entry(&cache_key, &payload);

        let got = read_cached_payload(&cache_key, &mut Cursor::new(bytes))?;

        assert_eq!(got, payload);
        Ok(())
    }

    #[test]
    fn try_mmap_cached_returns_payload_without_allocating_vec() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.key_bytes = b"odd-key".to_vec();
        cache_key.file_path = dir.path().join("entry.bin");
        let payload = [1u8, 2, 3, 4];
        std::fs::write(&cache_key.file_path, cache_entry(&cache_key, &payload))?;

        let got = try_mmap_cached(&cache_key, TransposedWeightCacheMissPolicy::ReadOnly)
            .context("expected cache hit")?;

        assert_eq!(got.as_bytes(), payload);
        assert_eq!(got.as_bytes().as_ptr() as usize % CACHE_PAYLOAD_ALIGN, 0);
        assert_eq!(got.shape(), [2, 2]);
        Ok(())
    }

    #[test]
    fn read_only_policy_leaves_invalid_cache_entry_untouched() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = dir.path().join("entry.bin");
        let invalid_entry = b"invalid cache entry";
        std::fs::write(&cache_key.file_path, invalid_entry)?;

        let got = try_mmap_cached(&cache_key, TransposedWeightCacheMissPolicy::ReadOnly);

        assert!(got.is_none());
        assert_eq!(std::fs::read(&cache_key.file_path)?, invalid_entry);
        Ok(())
    }

    #[test]
    fn persist_before_readiness_removes_structurally_invalid_entry() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = dir.path().join("entry.bin");
        std::fs::write(&cache_key.file_path, b"invalid cache entry")?;

        let got = try_mmap_cached(
            &cache_key,
            TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
        );

        assert!(got.is_none());
        assert!(!cache_key.file_path.exists());
        Ok(())
    }

    #[test]
    fn transient_cache_read_failure_is_nonfatal_and_never_removes_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = dir.path().join("entry.bin");
        std::fs::create_dir(&cache_key.file_path)?;

        for miss_policy in [
            TransposedWeightCacheMissPolicy::ReadOnly,
            TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
        ] {
            assert!(try_mmap_cached(&cache_key, miss_policy).is_none());
            assert!(cache_key.file_path.is_dir());
        }
        Ok(())
    }

    #[test]
    fn persist_policy_writes_payload_synchronously() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = dir.path().join("entry.bin");
        let payload = vec![1u8, 2, 3, 4];

        assert_eq!(
            handle_cache_miss(
                TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
                &cache_key,
                &payload,
            ),
            CachePersistence::Persisted
        );

        let mut file = std::fs::File::open(&cache_key.file_path)?;
        let got = read_cached_payload(&cache_key, &mut file)?;
        assert_eq!(got, payload.as_slice());
        assert_eq!(
            handle_cache_miss(
                TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
                &cache_key,
                &payload,
            ),
            CachePersistence::AlreadyPresent
        );
        Ok(())
    }

    #[test]
    fn read_only_policy_skips_cache_persistence() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = dir.path().join("entry.bin");
        let computed = finish_cache_miss(
            cache_key.clone(),
            TransposedWeightCacheMissPolicy::ReadOnly,
            vec![1u8, 2, 3, 4],
            [2, 2],
        );

        assert_eq!(computed.as_bytes(), [1, 2, 3, 4]);
        assert_eq!(computed.shape(), [2, 2]);
        assert!(!cache_key.file_path.exists());
        Ok(())
    }

    #[test]
    fn persist_policy_keeps_write_failure_nonfatal() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let parent_is_file = dir.path().join("not-a-directory");
        std::fs::write(&parent_is_file, b"block cache directory creation")?;
        let mut cache_key = test_cache_key(4);
        cache_key.file_path = parent_is_file.join("entry.bin");
        let payload = vec![1u8, 2, 3, 4];

        let computed = finish_cache_miss(
            cache_key.clone(),
            TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
            payload,
            [2, 2],
        );

        assert_eq!(computed.as_bytes(), [1, 2, 3, 4]);
        assert_eq!(computed.shape(), [2, 2]);
        assert!(!cache_key.file_path.exists());
        Ok(())
    }

    #[test]
    fn read_cached_payload_rejects_trailing_bytes() {
        let cache_key = test_cache_key(4);
        let payload = [1u8, 2, 3, 4];
        let mut bytes = cache_entry(&cache_key, &payload);
        bytes.push(5);

        let err = read_cached_payload(&cache_key, &mut Cursor::new(bytes)).unwrap_err();

        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn read_cached_payload_rejects_key_length_before_allocating_key() {
        let cache_key = test_cache_key(4);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CACHE_MAGIC);
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());

        let err = read_cached_payload(&cache_key, &mut Cursor::new(bytes)).unwrap_err();

        assert!(err.to_string().contains("cache key length mismatch"));
    }

    #[test]
    fn read_cached_payload_rejects_payload_length_mismatch() {
        let cache_key = test_cache_key(4);
        let payload = [1u8, 2, 3];
        let bytes = cache_entry(&cache_key, &payload);

        let err = read_cached_payload(&cache_key, &mut Cursor::new(bytes)).unwrap_err();

        assert!(err.to_string().contains("cache payload length mismatch"));
    }
}

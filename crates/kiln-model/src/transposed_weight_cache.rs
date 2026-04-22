use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};

use crate::forward::transposed_weight_bytes_2d;
use crate::weights::WeightTensor;

const CACHE_VERSION: u32 = 1;
const CACHE_MAGIC: &[u8; 8] = b"KILNTRP1";

struct CacheKey {
    file_path: PathBuf,
    key_bytes: Vec<u8>,
    expected_len: usize,
    shape: [usize; 2],
}

pub(crate) fn transposed_weight_bytes_2d_cached(
    weight: &WeightTensor,
) -> Result<(Vec<u8>, [usize; 2])> {
    let Some(cache_key) = cache_key(weight) else {
        return transposed_weight_bytes_2d(weight);
    };

    if let Some(bytes) = try_read_cached(&cache_key)? {
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
            "loaded transposed weight from disk cache"
        );
        return Ok((bytes, cache_key.shape));
    }

    let result = transposed_weight_bytes_2d(weight)?;
    if let Err(err) = try_write_cached(&cache_key, &result.0) {
        tracing::debug!(error = %err, "failed to write transposed weight cache entry");
    }
    Ok(result)
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

fn try_read_cached(cache_key: &CacheKey) -> Result<Option<Vec<u8>>> {
    let bytes = match fs::read(&cache_key.file_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "reading transposed weight cache {}",
                    cache_key.file_path.display()
                )
            });
        }
    };

    match parse_cached_bytes(cache_key, &bytes) {
        Ok(payload) => Ok(Some(payload)),
        Err(err) => {
            tracing::debug!(
                path = %cache_key.file_path.display(),
                error = %err,
                "ignoring invalid transposed weight cache entry"
            );
            let _ = fs::remove_file(&cache_key.file_path);
            Ok(None)
        }
    }
}

fn parse_cached_bytes(cache_key: &CacheKey, bytes: &[u8]) -> Result<Vec<u8>> {
    let mut cursor = 0usize;
    let magic = take_chunk(bytes, &mut cursor, CACHE_MAGIC.len())?;
    anyhow::ensure!(magic == CACHE_MAGIC, "cache entry magic mismatch");

    let version = u32::from_le_bytes(take_chunk(bytes, &mut cursor, 4)?.try_into().unwrap());
    anyhow::ensure!(version == CACHE_VERSION, "cache entry version mismatch");

    let key_len =
        u32::from_le_bytes(take_chunk(bytes, &mut cursor, 4)?.try_into().unwrap()) as usize;
    let cached_key = take_chunk(bytes, &mut cursor, key_len)?;
    anyhow::ensure!(
        cached_key == cache_key.key_bytes.as_slice(),
        "cache key mismatch"
    );

    let data_len =
        u64::from_le_bytes(take_chunk(bytes, &mut cursor, 8)?.try_into().unwrap()) as usize;
    anyhow::ensure!(
        data_len == cache_key.expected_len,
        "cache payload length mismatch: expected {}, got {}",
        cache_key.expected_len,
        data_len
    );

    let payload = take_chunk(bytes, &mut cursor, data_len)?;
    anyhow::ensure!(cursor == bytes.len(), "cache entry contains trailing bytes");
    Ok(payload.to_vec())
}

fn take_chunk<'a>(bytes: &'a [u8], cursor: &mut usize, n: usize) -> Result<&'a [u8]> {
    let start = *cursor;
    let end = start
        .checked_add(n)
        .context("cache entry length overflow")?;
    anyhow::ensure!(end <= bytes.len(), "cache entry truncated");
    *cursor = end;
    Ok(&bytes[start..end])
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

fn cache_root_dir() -> &'static PathBuf {
    static CACHE_ROOT: OnceLock<PathBuf> = OnceLock::new();
    CACHE_ROOT.get_or_init(|| {
        if let Some(dir) = env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(dir).join("kiln");
        }

        if let Some(home) = env::var_os("HOME") {
            #[cfg(target_os = "macos")]
            {
                return PathBuf::from(home)
                    .join("Library")
                    .join("Caches")
                    .join("kiln");
            }
            #[cfg(not(target_os = "macos"))]
            {
                return PathBuf::from(home).join(".cache").join("kiln");
            }
        }

        env::temp_dir().join("kiln")
    })
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

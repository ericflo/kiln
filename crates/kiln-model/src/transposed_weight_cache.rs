use std::env;
use std::fs;
use std::io::{Read, Write};
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
    let mut file = match fs::File::open(&cache_key.file_path) {
        Ok(file) => file,
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

    match read_cached_payload(cache_key, &mut file) {
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

fn read_cached_payload<R: Read>(cache_key: &CacheKey, reader: &mut R) -> Result<Vec<u8>> {
    let mut magic = [0u8; CACHE_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .context("cache entry truncated before magic")?;
    anyhow::ensure!(&magic == CACHE_MAGIC, "cache entry magic mismatch");

    let version = read_u32_le(reader).context("reading cache entry version")?;
    anyhow::ensure!(version == CACHE_VERSION, "cache entry version mismatch");

    let key_len = read_u32_le(reader).context("reading cache entry key length")? as usize;
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

    let data_len_u64 = read_u64_le(reader).context("reading cache entry payload length")?;
    let data_len = usize::try_from(data_len_u64).context("cache payload length overflows usize")?;
    anyhow::ensure!(
        data_len == cache_key.expected_len,
        "cache payload length mismatch: expected {}, got {}",
        cache_key.expected_len,
        data_len
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

fn read_u32_le(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64> {
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

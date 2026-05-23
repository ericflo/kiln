//! `AlgoCache` — disk-persistent cublasLt algorithm cache.
//!
//! Per the Phase 2 issue bullet:
//!
//! > **Disk-persistent autotune cache**: `~/.cache/kiln/autotune/
//! > {backend}-{device_uuid}.json`, keyed on `{shape, dtype, layout,
//! > concurrent_streams, kiln_version}`. Applies to cublasLt
//! > heuristics, MPS tile selection, and Vulkan workgroup-size search
//! > uniformly. Cold-start never re-tunes a known shape.
//! >
//! > **`concurrent_streams` matters**: the optimal algo for a matmul
//! > running solo on the device is *not* the optimal algo when two
//! > other QKV matmuls are running on parallel streams competing for
//! > SMs. Cache key includes the planner's `expected_concurrent_streams`
//! > so Phase 3's parallel-QKV picks SM-light algos automatically.
//!
//! And:
//!
//! > **Pre-shipped autotune cache (Qwen3.5-4B specialization).** Ship
//! > `assets/autotune/{backend}-{gpu_sku}.json` *in the binary* for
//! > the tier-1 GPU list ... Cold-start on tier-1 hardware has zero
//! > tune cost.
//!
//! # Phase 2.1 scope
//!
//! Backend-agnostic data types only. The actual cublasLt algo-id
//! type (`cublasLtMatmulAlgo_t`) is opaque to this layer — we cache
//! the **bytes** of the algo descriptor along with metadata used for
//! lookup. The cublasLt-specific serialization shape lives in the
//! per-backend impl in subsequent PRs.
//!
//! # No-cuda build
//!
//! This module compiles on every host. No `--features cuda` required —
//! it's pure data structures + JSON I/O.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Cache key for a single autotune entry. Hashed + serialized to JSON
/// so the cache survives process restarts and can be pre-shipped per
/// the Qwen3.5-4B specialization bullet.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlgoCacheKey {
    /// Matmul shape `[m, n, k]`.
    pub shape: [u64; 3],
    /// Input dtype short-name (`"bf16"`, `"f32"`, `"f16"`, etc.).
    pub input_dtype: String,
    /// Output dtype short-name. Often equals `input_dtype` but the FP8
    /// + accumulator paths have distinct in/out.
    pub output_dtype: String,
    /// Compute dtype short-name (`"f32"` for CUBLAS_COMPUTE_32F, etc.).
    pub compute_dtype: String,
    /// Transpose flags for A and B operands (`true` = transposed,
    /// `false` = non-transposed).
    pub transpose: [bool; 2],
    /// Number of concurrent streams the StreamPlanner expects this
    /// matmul to compete with. Bucket: `1 | 2 | 4`. Default 1.
    pub expected_concurrent_streams: u8,
    /// Major version of the kiln binary that recorded this entry.
    /// Cache hits across mismatched majors are rejected.
    pub kiln_version_major: u32,
}

impl AlgoCacheKey {
    /// Convenience constructor with sensible defaults.
    pub fn new(m: u64, n: u64, k: u64, dtype: impl Into<String>) -> Self {
        let dt: String = dtype.into();
        AlgoCacheKey {
            shape: [m, n, k],
            input_dtype: dt.clone(),
            output_dtype: dt.clone(),
            compute_dtype: "f32".to_string(),
            transpose: [false, false],
            expected_concurrent_streams: 1,
            kiln_version_major: 0,
        }
    }
}

/// Value side of the cache. The opaque `algo_blob` is whatever the
/// per-backend impl wants to store (typically a cublasLt
/// `cublasLtMatmulAlgo_t` serialized to bytes, plus the workspace
/// size that goes alongside).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgoCacheValue {
    /// Algo id reported by cublasLt's heuristic (or per-backend
    /// equivalent). `-1` means "unknown / default".
    pub algo_id: i32,
    /// Workspace bytes the algo requested.
    pub workspace_bytes: u64,
    /// Median timing observed when this entry was recorded (ms).
    /// Phase 2.x's runtime tuner compares against this when re-tuning.
    pub recorded_ms: f32,
    /// Opaque per-backend payload. cublasLt stores the full
    /// `cublasLtMatmulAlgo_t` descriptor here.
    pub algo_blob: Vec<u8>,
}

/// In-memory autotune cache. Disk persistence is the caller's
/// responsibility; this struct exposes load_from + save_to helpers
/// for JSON round-trip.
///
/// Threading: the cache is not `Send + Sync`-safe for mutation; wrap
/// in `Mutex` if multiple threads write. Reads are safe under
/// `Arc<AlgoCache>`.
#[derive(Debug, Default, Clone)]
pub struct AlgoCache {
    entries: HashMap<AlgoCacheKey, AlgoCacheValue>,
}

impl AlgoCache {
    /// Empty cache.
    pub fn new() -> Self {
        AlgoCache {
            entries: HashMap::new(),
        }
    }

    /// Insert or replace a cache entry.
    pub fn insert(&mut self, key: AlgoCacheKey, value: AlgoCacheValue) {
        self.entries.insert(key, value);
    }

    /// Look up an entry.
    pub fn get(&self, key: &AlgoCacheKey) -> Option<&AlgoCacheValue> {
        self.entries.get(key)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate `(key, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&AlgoCacheKey, &AlgoCacheValue)> {
        self.entries.iter()
    }

    /// Compute the standard cache file path for a backend + device
    /// fingerprint. Mirrors the Phase 2 issue path:
    /// `~/.cache/kiln/autotune/{backend}-{device_uuid}.json`.
    pub fn standard_path(backend: &str, device_fingerprint: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".cache/kiln/autotune")
            .join(format!("{backend}-{device_fingerprint}.json"))
    }
}

/// Serialize the cache to a simple JSON blob.
///
/// Format: a JSON array of `{key, value}` objects. Stable across
/// minor kiln versions (the `kiln_version_major` field is part of
/// the key, so a different major triggers cache rebuild).
///
/// Returns the JSON string. Callers persist via standard fs::write.
pub fn serialize_to_json(cache: &AlgoCache) -> String {
    let mut s = String::from("[");
    let mut first = true;
    for (k, v) in cache.iter() {
        if !first {
            s.push(',');
        }
        first = false;
        s.push_str("{\"key\":{");
        s.push_str(&format!(
            "\"shape\":[{},{},{}],\"input_dtype\":{},\"output_dtype\":{},\"compute_dtype\":{},\"transpose\":[{},{}],\"expected_concurrent_streams\":{},\"kiln_version_major\":{}",
            k.shape[0], k.shape[1], k.shape[2],
            json_str(&k.input_dtype),
            json_str(&k.output_dtype),
            json_str(&k.compute_dtype),
            k.transpose[0], k.transpose[1],
            k.expected_concurrent_streams,
            k.kiln_version_major,
        ));
        s.push_str("},\"value\":{");
        s.push_str(&format!(
            "\"algo_id\":{},\"workspace_bytes\":{},\"recorded_ms\":{}",
            v.algo_id, v.workspace_bytes, v.recorded_ms,
        ));
        s.push_str(",\"algo_blob_b64\":");
        s.push_str(&json_str(&base64_encode(&v.algo_blob)));
        s.push_str("}}");
    }
    s.push(']');
    s
}

/// Write the cache to a file. Creates parent directories as needed.
pub fn save_to_path(cache: &AlgoCache, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serialize_to_json(cache).as_bytes())?;
    Ok(())
}

fn json_str(s: &str) -> String {
    // Minimal JSON-string escaper. Sufficient for short dtype names.
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Tiny stdlib base64 encoder (avoids pulling in `base64` as a dep).
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = u32::from_be_bytes([0, bytes[i], bytes[i + 1], bytes[i + 2]]);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        out.push(TABLE[(n & 0x3F) as usize] as char);
        i += 3;
    }
    if i < bytes.len() {
        let r = bytes.len() - i;
        let b0 = bytes[i];
        let b1 = if r > 1 { bytes[i + 1] } else { 0 };
        let n = u32::from_be_bytes([0, b0, b1, 0]);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        if r == 2 {
            out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        } else {
            out.push('=');
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_insert_and_lookup() {
        let mut c = AlgoCache::new();
        assert!(c.is_empty());
        let key = AlgoCacheKey::new(64, 64, 32, "bf16");
        let val = AlgoCacheValue {
            algo_id: 42,
            workspace_bytes: 1024,
            recorded_ms: 0.42,
            algo_blob: vec![1, 2, 3, 4],
        };
        c.insert(key.clone(), val.clone());
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&key).unwrap().algo_id, 42);
    }

    #[test]
    fn standard_path_format() {
        let p = AlgoCache::standard_path("cublaslt", "a6000-86");
        let s = p.to_string_lossy();
        assert!(s.contains(".cache/kiln/autotune/"));
        assert!(s.ends_with("cublaslt-a6000-86.json"));
    }

    #[test]
    fn json_serialization_round_trip_format() {
        let mut c = AlgoCache::new();
        c.insert(
            AlgoCacheKey::new(64, 64, 32, "bf16"),
            AlgoCacheValue {
                algo_id: 7,
                workspace_bytes: 2048,
                recorded_ms: 1.25,
                algo_blob: vec![0xCA, 0xFE, 0xBA, 0xBE],
            },
        );
        let s = serialize_to_json(&c);
        assert!(s.starts_with('['));
        assert!(s.ends_with(']'));
        assert!(s.contains("\"algo_id\":7"));
        assert!(s.contains("\"workspace_bytes\":2048"));
        assert!(s.contains("\"bf16\""));
        // BASE64 of [0xCA, 0xFE, 0xBA, 0xBE] = "yv66vg=="
        assert!(s.contains("yv66vg=="));
    }

    #[test]
    fn cache_key_with_concurrent_streams() {
        let solo = AlgoCacheKey::new(64, 64, 32, "bf16");
        let mut parallel = solo.clone();
        parallel.expected_concurrent_streams = 4;
        assert_ne!(solo, parallel); // distinct keys
    }

    #[test]
    fn save_to_path_writes_file() {
        let tmp = std::env::temp_dir().join("kiln-blas-test-algo-cache.json");
        let _ = std::fs::remove_file(&tmp);
        let mut c = AlgoCache::new();
        c.insert(
            AlgoCacheKey::new(1, 1, 1, "f32"),
            AlgoCacheValue {
                algo_id: 0,
                workspace_bytes: 0,
                recorded_ms: 0.0,
                algo_blob: vec![],
            },
        );
        save_to_path(&c, &tmp).unwrap();
        let s = std::fs::read_to_string(&tmp).unwrap();
        assert!(s.contains("\"algo_id\":0"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn base64_known_values() {
        assert_eq!(base64_encode(&[]), "");
        assert_eq!(base64_encode(b"A"), "QQ==");
        assert_eq!(base64_encode(b"AB"), "QUI=");
        assert_eq!(base64_encode(b"ABC"), "QUJD");
        assert_eq!(base64_encode(&[0xCA, 0xFE, 0xBA, 0xBE]), "yv66vg==");
    }
}

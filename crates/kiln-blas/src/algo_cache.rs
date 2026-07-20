//! `AlgoCache` — disk-persistent cublasLt algorithm cache.
//!
//! Per the Phase 2 issue bullet:
//!
//! > **Disk-persistent autotune cache**: `<paths.cache_root>/autotune/
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
    /// Number of strided-batch matrices. `1` means non-batched GEMM.
    pub batch_count: u64,
    /// Strides, in elements, for A/B/C between batch members.
    pub batch_strides: [u64; 3],
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
            batch_count: 1,
            batch_strides: [0, 0, 0],
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
#[derive(Debug, Clone, PartialEq)]
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

/// Runtime visibility for algorithm-cache behavior. `entries` describes the
/// persistent cache contents; hit/miss/insert counters are owned by runtime
/// handles and are intentionally not serialized to disk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AlgoCacheStats {
    /// Number of entries currently present in the cache snapshot.
    pub entries: usize,
    /// Runtime cache lookups that reused a non-empty cached algo blob.
    pub hits: u64,
    /// Runtime cache lookups that had to run the backend heuristic path.
    pub misses: u64,
    /// Runtime heuristic results inserted into the cache.
    pub inserts: u64,
}

impl AlgoCacheStats {
    /// Total runtime cache lookups observed by the owning handle.
    pub fn lookups(&self) -> u64 {
        self.hits + self.misses
    }

    /// Fraction of runtime lookups served from a cached algo blob.
    pub fn hit_rate(&self) -> Option<f64> {
        let lookups = self.lookups();
        if lookups == 0 {
            None
        } else {
            Some(self.hits as f64 / lookups as f64)
        }
    }
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

    /// Snapshot cache contents as stats. Runtime hit/miss/insert counters are
    /// supplied by the backend handle that owns lookups.
    pub fn stats(&self) -> AlgoCacheStats {
        AlgoCacheStats {
            entries: self.entries.len(),
            ..AlgoCacheStats::default()
        }
    }

    /// Compute the standard cache file path for a backend + device
    /// fingerprint under the process-wide typed application cache root.
    pub fn standard_path(backend: &str, device_fingerprint: &str) -> PathBuf {
        kiln_resource::application_cache_root()
            .join("autotune")
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
            "\"shape\":[{},{},{}],\"batch_count\":{},\"batch_strides\":[{},{},{}],\"input_dtype\":{},\"output_dtype\":{},\"compute_dtype\":{},\"transpose\":[{},{}],\"expected_concurrent_streams\":{},\"kiln_version_major\":{}",
            k.shape[0], k.shape[1], k.shape[2],
            k.batch_count,
            k.batch_strides[0], k.batch_strides[1], k.batch_strides[2],
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
///
/// Cross-process safety is part of the cache contract: concurrent Kiln starts
/// must never expose readers to a truncated JSON file and should not clobber
/// each other's freshly tuned entries. Writes therefore take a best-effort
/// lock file, merge with the current on-disk cache, write a process-unique temp
/// file, fsync it, and atomically rename it into place.
pub fn save_to_path(cache: &AlgoCache, path: &Path) -> std::io::Result<()> {
    kiln_resource::locked_update(path, |existing| {
        let mut merged = existing
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(deserialize_from_json)
            .unwrap_or_default();
        for (k, v) in cache.iter() {
            merged.insert(k.clone(), v.clone());
        }
        Ok(serialize_to_json(&merged).into_bytes())
    })
}

/// Parse a JSON cache blob produced by [`serialize_to_json`]. Returns an
/// empty cache on any parse error so a corrupt or incompatible on-disk file
/// never blocks startup. Parsed with `serde_json` (the writer is hand-rolled,
/// but the format is plain one-level JSON), then field-extracted so missing or
/// renamed fields degrade gracefully rather than rejecting the whole file.
pub fn deserialize_from_json(json: &str) -> AlgoCache {
    let mut cache = AlgoCache::new();
    let val: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return cache,
    };
    let Some(arr) = val.as_array() else {
        return cache;
    };
    for entry in arr {
        let (Some(k), Some(v)) = (entry.get("key"), entry.get("value")) else {
            continue;
        };
        let shape = match k.get("shape").and_then(|s| s.as_array()) {
            Some(a) if a.len() == 3 => [
                a[0].as_u64().unwrap_or(0),
                a[1].as_u64().unwrap_or(0),
                a[2].as_u64().unwrap_or(0),
            ],
            _ => continue,
        };
        let transpose = match k.get("transpose").and_then(|t| t.as_array()) {
            Some(a) if a.len() == 2 => [
                a[0].as_bool().unwrap_or(false),
                a[1].as_bool().unwrap_or(false),
            ],
            _ => [false, false],
        };
        let batch_strides = match k.get("batch_strides").and_then(|s| s.as_array()) {
            Some(a) if a.len() == 3 => [
                a[0].as_u64().unwrap_or(0),
                a[1].as_u64().unwrap_or(0),
                a[2].as_u64().unwrap_or(0),
            ],
            _ => [0, 0, 0],
        };
        let key = AlgoCacheKey {
            shape,
            batch_count: k.get("batch_count").and_then(|x| x.as_u64()).unwrap_or(1),
            batch_strides,
            input_dtype: str_field(k, "input_dtype"),
            output_dtype: str_field(k, "output_dtype"),
            compute_dtype: str_field(k, "compute_dtype"),
            transpose,
            expected_concurrent_streams: k
                .get("expected_concurrent_streams")
                .and_then(|x| x.as_u64())
                .unwrap_or(1) as u8,
            kiln_version_major: k
                .get("kiln_version_major")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as u32,
        };
        let value = AlgoCacheValue {
            algo_id: v.get("algo_id").and_then(|x| x.as_i64()).unwrap_or(-1) as i32,
            workspace_bytes: v
                .get("workspace_bytes")
                .and_then(|x| x.as_u64())
                .unwrap_or(0),
            recorded_ms: v.get("recorded_ms").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            algo_blob: v
                .get("algo_blob_b64")
                .and_then(|x| x.as_str())
                .map(base64_decode)
                .unwrap_or_default(),
        };
        cache.insert(key, value);
    }
    cache
}

/// Load a cache from `path`. Returns an empty cache (no panic) if the file is
/// missing, unreadable, or unparseable — a stale cache must never break boot.
pub fn load_from_path(path: &Path) -> AlgoCache {
    match std::fs::read_to_string(path) {
        Ok(json) => deserialize_from_json(&json),
        Err(_) => AlgoCache::new(),
    }
}

fn str_field(obj: &serde_json::Value, field: &str) -> String {
    obj.get(field)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
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
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

/// Inverse of [`base64_encode`] — tolerant stdlib base64 decoder. Ignores
/// `=` padding and whitespace; stops at the first invalid character.
fn base64_decode(s: &str) -> Vec<u8> {
    fn dval(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s
        .bytes()
        .filter(|&c| c != b'=' && !c.is_ascii_whitespace())
        .collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut buf = [0u8; 4];
        let mut ok = true;
        for (i, &c) in chunk.iter().enumerate() {
            match dval(c) {
                Some(x) => buf[i] = x,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            break;
        }
        let triple = ((buf[0] as u32) << 18)
            | ((buf[1] as u32) << 12)
            | ((buf[2] as u32) << 6)
            | (buf[3] as u32);
        if chunk.len() >= 2 {
            out.push((triple >> 16) as u8);
        }
        if chunk.len() >= 3 {
            out.push((triple >> 8) as u8);
        }
        if chunk.len() >= 4 {
            out.push(triple as u8);
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
    fn cache_stats_reports_entries_and_hit_rate() {
        let mut c = AlgoCache::new();
        assert_eq!(c.stats(), AlgoCacheStats::default());
        assert_eq!(c.stats().hit_rate(), None);

        c.insert(
            AlgoCacheKey::new(64, 64, 32, "bf16"),
            AlgoCacheValue {
                algo_id: 42,
                workspace_bytes: 1024,
                recorded_ms: 0.42,
                algo_blob: vec![1, 2, 3, 4],
            },
        );
        assert_eq!(
            c.stats(),
            AlgoCacheStats {
                entries: 1,
                hits: 0,
                misses: 0,
                inserts: 0,
            }
        );

        let runtime = AlgoCacheStats {
            entries: 1,
            hits: 3,
            misses: 1,
            inserts: 1,
        };
        assert_eq!(runtime.lookups(), 4);
        assert_eq!(runtime.hit_rate(), Some(0.75));
    }

    #[test]
    fn standard_path_format() {
        let p = AlgoCache::standard_path("cublaslt", "a6000-86");
        assert_eq!(
            p,
            kiln_resource::application_cache_root()
                .join("autotune")
                .join("cublaslt-a6000-86.json")
        );
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
    fn concurrent_save_to_path_merges_entries_without_partial_files() {
        let dir =
            std::env::temp_dir().join(format!("kiln-blas-concurrent-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cublaslt-sm00-dev0.json");

        let handles = (0..8u64)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let mut c = AlgoCache::new();
                    c.insert(
                        AlgoCacheKey::new(i + 1, i + 2, i + 3, "f32"),
                        AlgoCacheValue {
                            algo_id: i as i32,
                            workspace_bytes: i * 1024,
                            recorded_ms: i as f32,
                            algo_blob: vec![i as u8],
                        },
                    );
                    save_to_path(&c, &path).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for h in handles {
            h.join().unwrap();
        }

        let loaded = load_from_path(&path);
        assert_eq!(loaded.len(), 8);
        assert_eq!(
            std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
                .count(),
            0
        );
        assert!(!kiln_resource::lock_path_for(&path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base64_known_values() {
        assert_eq!(base64_encode(&[]), "");
        assert_eq!(base64_encode(b"A"), "QQ==");
        assert_eq!(base64_encode(b"AB"), "QUI=");
        assert_eq!(base64_encode(b"ABC"), "QUJD");
        assert_eq!(base64_encode(&[0xCA, 0xFE, 0xBA, 0xBE]), "yv66vg==");
    }

    #[test]
    fn base64_decode_inverts_encode() {
        for case in [
            &b""[..],
            b"A",
            b"AB",
            b"ABC",
            &[0xCA, 0xFE, 0xBA, 0xBE][..],
            &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03][..],
        ] {
            assert_eq!(base64_decode(&base64_encode(case)), case.to_vec());
        }
    }

    #[test]
    fn deserialize_round_trips_serialize_all_fields() {
        let mut c = AlgoCache::new();
        // Non-default values in every field so the round-trip exercises them all.
        let key = AlgoCacheKey {
            shape: [2048, 18432, 2560],
            batch_count: 8,
            batch_strides: [5_242_880, 47_185_920, 37_748_736],
            input_dtype: "bf16".to_string(),
            output_dtype: "f32".to_string(),
            compute_dtype: "f32".to_string(),
            transpose: [false, true],
            expected_concurrent_streams: 4,
            kiln_version_major: 1,
        };
        let val = AlgoCacheValue {
            algo_id: 5,
            workspace_bytes: 4_194_304,
            recorded_ms: 0.25, // exactly representable in f32
            algo_blob: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
        };
        c.insert(key.clone(), val.clone());
        let restored = deserialize_from_json(&serialize_to_json(&c));
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get(&key), Some(&val));
    }

    #[test]
    fn deserialize_negative_algo_id_round_trips() {
        let mut c = AlgoCache::new();
        let key = AlgoCacheKey::new(64, 64, 32, "f16");
        c.insert(
            key.clone(),
            AlgoCacheValue {
                algo_id: -1,
                workspace_bytes: 0,
                recorded_ms: 0.0,
                algo_blob: vec![],
            },
        );
        let restored = deserialize_from_json(&serialize_to_json(&c));
        assert_eq!(restored.get(&key).unwrap().algo_id, -1);
    }

    #[test]
    fn load_from_path_empty_on_missing_or_corrupt() {
        let missing = std::path::Path::new("/tmp/kiln-nonexistent-algocache-xyz.json");
        let _ = std::fs::remove_file(missing);
        assert!(load_from_path(missing).is_empty());

        let corrupt = std::env::temp_dir().join("kiln-corrupt-algocache.json");
        std::fs::write(&corrupt, b"not valid json {{{").unwrap();
        assert!(load_from_path(&corrupt).is_empty());
        let _ = std::fs::remove_file(&corrupt);
    }

    #[test]
    fn save_then_load_round_trip_via_disk() {
        let tmp = std::env::temp_dir().join("kiln-algocache-disk-roundtrip.json");
        let _ = std::fs::remove_file(&tmp);
        let mut c = AlgoCache::new();
        c.insert(
            AlgoCacheKey::new(1024, 9216, 2560, "bf16"),
            AlgoCacheValue {
                algo_id: 11,
                workspace_bytes: 65536,
                recorded_ms: 0.5,
                algo_blob: vec![1, 2, 3, 4, 5],
            },
        );
        save_to_path(&c, &tmp).unwrap();
        let loaded = load_from_path(&tmp);
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded
                .get(&AlgoCacheKey::new(1024, 9216, 2560, "bf16"))
                .unwrap()
                .algo_id,
            11
        );
        let _ = std::fs::remove_file(&tmp);
    }
}

//! Content-addressed Parameter identity.
//!
//! Per the Phase 2.5 issue bullet:
//!
//! > **Parameter content checksum (xxhash3) for safe hot-swap.**
//! > `Parameter::content_hash() -> u64` is a content-addressed
//! > fingerprint of `forward_storage` (post-quantization,
//! > post-LoRA-merge). Required for: (a) multi-process serving safety
//! > — a second kiln process attaching to the same Qwen3.5-4B model
//! > via CUDA IPC can verify the in-memory weights match its expected
//! > checkpoint; (b) hot-swap of a fine-tune over a running serve —
//! > request mid-flight cache invalidation when the content_hash
//! > changes; (c) adapter cache invalidation on the serve side.
//!
//! # Today: stdlib `DefaultHasher`
//!
//! Phase 2.5 ships the API + a stdlib `DefaultHasher` implementation;
//! the xxhash3 swap is a Phase 2.5.x follow-up that depends on adding
//! the `xxhash-rust` crate. The hash interface is stable; only the
//! underlying algorithm changes, and the swap is a one-line constant
//! switch.
//!
//! Why scaffold rather than pull xxhash3 today: the directive on #1082
//! is to ship verifiable code; stdlib hash always compiles. The
//! xxhash3 PR follow-up verifies the dep + the bench on the same
//! Qwen3.5-4B 4 GiB BF16 master that the issue cites
//! ("sub-millisecond on the 4 GiB master").

use std::hash::{Hash, Hasher};

use kiln_tensor::{Result, Storage, StorageBackend};

/// Compute a content hash of a storage's bytes.
///
/// **For now**: stdlib `DefaultHasher` (deterministic within a Rust
/// version + platform). Phase 2.5.x replaces this with xxhash3 for
/// stability across Rust versions and faster hashing on the 4 GiB
/// Qwen3.5-4B BF16 master.
///
/// CPU-only: requires downcasting to `CpuStorage`. GPU storages are
/// hashed by first staging to host in a follow-up PR (Phase 1.12's
/// pinned-host pool is the staging target).
pub fn content_hash_storage(storage: &Storage) -> Result<u64> {
    let cpu = storage
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .ok_or_else(|| {
            kiln_tensor::Error::from_str(
                "content_hash_storage: only CpuStorage is hashable in Phase 2.5; \
                 GPU storage hashing lands with the pinned-host pool",
            )
        })?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Include the dtype short-name in the hash so two storages with
    // identical bytes but different dtypes hash distinctly. Important
    // for the Marlin-packed-int4 vs raw-bf16 disambiguation.
    storage.dtype().short_name().hash(&mut hasher);
    cpu.as_bytes().hash(&mut hasher);
    Ok(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{cpu_zeros, DType};
    use std::sync::Arc;

    fn make_storage(dtype: DType, bytes: Vec<u8>) -> Storage {
        let cpu = kiln_tensor::CpuStorage::from_bytes(dtype, bytes).unwrap();
        Arc::new(cpu)
    }

    #[test]
    fn identical_bytes_same_hash() {
        let a = make_storage(DType::F32, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_storage(DType::F32, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            content_hash_storage(&a).unwrap(),
            content_hash_storage(&b).unwrap()
        );
    }

    #[test]
    fn different_bytes_different_hash() {
        let a = make_storage(DType::F32, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_storage(DType::F32, vec![1, 2, 3, 4, 5, 6, 7, 9]);
        assert_ne!(
            content_hash_storage(&a).unwrap(),
            content_hash_storage(&b).unwrap()
        );
    }

    #[test]
    fn dtype_distinguishes_identical_bytes() {
        // Same bytes, different dtype -> different hash. Important
        // because Marlin-packed int4 and raw bf16 can share a byte
        // pattern but represent very different weights.
        let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let a = make_storage(DType::F32, bytes.clone());
        let b = make_storage(DType::BF16, bytes);
        assert_ne!(
            content_hash_storage(&a).unwrap(),
            content_hash_storage(&b).unwrap()
        );
    }

    #[test]
    fn zeros_hash_is_dtype_specific() {
        let zf = cpu_zeros(DType::F32, 16);
        let zb = cpu_zeros(DType::BF16, 32); // same 64 bytes
        assert_ne!(
            content_hash_storage(&zf).unwrap(),
            content_hash_storage(&zb).unwrap()
        );
    }
}

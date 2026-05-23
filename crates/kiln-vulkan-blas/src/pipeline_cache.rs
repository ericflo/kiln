//! `VkPipelineCacheKey` — hash key for the disk-persistent Vulkan
//! pipeline cache.
//!
//! Per the Phase 2 issue bullet:
//!
//! > **Vulkan: disk-persistent pipeline cache.**
//! > `~/.cache/kiln/vulkan/pipeline-cache-{device_uuid}.bin`. Eliminates
//! > SPIR-V re-compile on every cold start (0.5–5 s currently); reuses
//! > the disk-persistent autotune cache pattern above.
//!
//! The cache key identifies a unique combination of:
//! - device UUID (so cache files survive driver swaps but not GPU swaps)
//! - SPIR-V shader hash (so cache invalidates on shader edits)
//! - kiln binary version (so old caches are rejected on update)
//!
//! Phase 2.x's `kiln_vulkan_blas::MatmulHandle` reads this cache at
//! startup before any pipeline creation.

use std::path::PathBuf;

/// Cache file key. Maps 1:1 to one `pipeline-cache-{key}.bin` on disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VkPipelineCacheKey {
    /// Vulkan device UUID (16 bytes from
    /// `vkGetPhysicalDeviceProperties::pipelineCacheUUID`).
    pub device_uuid: [u8; 16],
    /// Hash of all SPIR-V shaders the pipeline uses. xxhash3 64-bit;
    /// today's stub uses stdlib `DefaultHasher` (Phase 2.5.x swap to
    /// xxhash3 covers this file too).
    pub shader_hash: u64,
    /// Kiln binary version major. Cache hits across mismatched
    /// majors are rejected.
    pub kiln_version_major: u32,
}

impl VkPipelineCacheKey {
    /// Format the standard cache file path.
    pub fn cache_path(&self) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let uuid_hex = hex_lower(&self.device_uuid);
        PathBuf::from(home)
            .join(".cache/kiln/vulkan")
            .join(format!(
                "pipeline-cache-{uuid_hex}-{:016x}-v{}.bin",
                self.shader_hash, self.kiln_version_major
            ))
    }

    /// True iff this key's `kiln_version_major` matches `current`.
    /// Phase 2.x's loader uses this to reject mismatched-version
    /// cache files cleanly.
    pub const fn matches_version(&self, current: u32) -> bool {
        self.kiln_version_major == current
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let hi = b >> 4;
        let lo = b & 0x0F;
        out.push(hex_char(hi));
        out.push(hex_char(lo));
    }
    out
}

const fn hex_char(nibble: u8) -> char {
    if nibble < 10 {
        (b'0' + nibble) as char
    } else {
        (b'a' + (nibble - 10)) as char
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_contains_uuid_hex() {
        let k = VkPipelineCacheKey {
            device_uuid: [
                0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED, 0xFA, 0xCE, //
                0xCA, 0xFE, 0xBA, 0xBE, 0xFA, 0x11, 0xCA, 0xB1,
            ],
            shader_hash: 0xABCDEF0123456789,
            kiln_version_major: 7,
        };
        let s = k.cache_path().to_string_lossy().to_string();
        assert!(s.contains(".cache/kiln/vulkan/"));
        assert!(s.contains("deadbeeffeedface"));
        assert!(s.contains("abcdef0123456789"));
        assert!(s.ends_with("-v7.bin"));
    }

    #[test]
    fn matches_version_predicate() {
        let k = VkPipelineCacheKey {
            device_uuid: [0u8; 16],
            shader_hash: 0,
            kiln_version_major: 3,
        };
        assert!(k.matches_version(3));
        assert!(!k.matches_version(2));
        assert!(!k.matches_version(4));
    }

    #[test]
    fn hex_lower_known_values() {
        assert_eq!(hex_lower(&[]), "");
        assert_eq!(hex_lower(&[0x00]), "00");
        assert_eq!(hex_lower(&[0xFF]), "ff");
        assert_eq!(hex_lower(&[0xAB, 0xCD]), "abcd");
        assert_eq!(hex_lower(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
    }
}

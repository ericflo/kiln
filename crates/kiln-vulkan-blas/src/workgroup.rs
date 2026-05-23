//! `VkWorkgroupConfig` — Vulkan workgroup-size + subgroup-config cache entry.
//!
//! Vulkan compute pipelines pick performance from:
//! - local workgroup size (X, Y, Z)
//! - subgroup size (warp / wavefront equivalent)
//! - cooperative-matrix tile (when `VK_KHR_cooperative_matrix` is available)
//!
//! `VkWorkgroupConfig` carries this configuration in a compact form that
//! serializes cleanly to the algo_blob byte slot in
//! [`kiln_blas::AlgoCacheValue`].

use std::convert::TryInto;

/// Vulkan compute-pipeline configuration for one matmul shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VkWorkgroupConfig {
    /// Local workgroup size along X.
    pub local_x: u32,
    /// Local workgroup size along Y.
    pub local_y: u32,
    /// Local workgroup size along Z.
    pub local_z: u32,
    /// Subgroup size in shader invocations (typically 32 on NVIDIA,
    /// 32/64 on AMD RDNA, 8/16/32 on Intel Arc).
    pub subgroup_size: u32,
    /// Whether `VK_KHR_cooperative_matrix` is used by this pipeline.
    pub uses_cooperative_matrix: bool,
}

impl VkWorkgroupConfig {
    /// Default 64-thread workgroup with subgroup_size=32 and no
    /// cooperative matrix. Reasonable starting point for the runtime
    /// heuristic to refine against.
    pub const DEFAULT: Self = VkWorkgroupConfig {
        local_x: 64,
        local_y: 1,
        local_z: 1,
        subgroup_size: 32,
        uses_cooperative_matrix: false,
    };

    /// Serialize to a 17-byte blob:
    /// `[local_x: u32 LE][local_y: u32 LE][local_z: u32 LE]
    ///  [subgroup_size: u32 LE][uses_cooperative_matrix: 1 byte]`.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(17);
        out.extend_from_slice(&self.local_x.to_le_bytes());
        out.extend_from_slice(&self.local_y.to_le_bytes());
        out.extend_from_slice(&self.local_z.to_le_bytes());
        out.extend_from_slice(&self.subgroup_size.to_le_bytes());
        out.push(self.uses_cooperative_matrix as u8);
        out
    }

    /// Reverse of `to_blob`. Returns `None` on a malformed blob.
    pub fn from_blob(blob: &[u8]) -> Option<Self> {
        if blob.len() != 17 {
            return None;
        }
        Some(VkWorkgroupConfig {
            local_x: u32::from_le_bytes(blob[0..4].try_into().ok()?),
            local_y: u32::from_le_bytes(blob[4..8].try_into().ok()?),
            local_z: u32::from_le_bytes(blob[8..12].try_into().ok()?),
            subgroup_size: u32::from_le_bytes(blob[12..16].try_into().ok()?),
            uses_cooperative_matrix: blob[16] != 0,
        })
    }

    /// Total invocations per workgroup (`local_x * local_y * local_z`).
    /// Capped by the device's `maxComputeWorkGroupInvocations` —
    /// callers consult Vulkan device info before recording.
    pub const fn invocations_per_workgroup(self) -> u32 {
        self.local_x * self.local_y * self.local_z
    }
}

impl Default for VkWorkgroupConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_64x1x1_subgroup32() {
        let c = VkWorkgroupConfig::DEFAULT;
        assert_eq!(c.local_x, 64);
        assert_eq!(c.local_y, 1);
        assert_eq!(c.local_z, 1);
        assert_eq!(c.subgroup_size, 32);
        assert!(!c.uses_cooperative_matrix);
        assert_eq!(c.invocations_per_workgroup(), 64);
    }

    #[test]
    fn blob_round_trip() {
        let c = VkWorkgroupConfig {
            local_x: 32,
            local_y: 16,
            local_z: 1,
            subgroup_size: 32,
            uses_cooperative_matrix: true,
        };
        let blob = c.to_blob();
        assert_eq!(blob.len(), 17);
        assert_eq!(VkWorkgroupConfig::from_blob(&blob).unwrap(), c);
    }

    #[test]
    fn invocations_arithmetic() {
        let c = VkWorkgroupConfig {
            local_x: 16,
            local_y: 8,
            local_z: 4,
            subgroup_size: 32,
            uses_cooperative_matrix: false,
        };
        assert_eq!(c.invocations_per_workgroup(), 512);
    }

    #[test]
    fn blob_rejects_wrong_size() {
        assert!(VkWorkgroupConfig::from_blob(&[]).is_none());
        assert!(VkWorkgroupConfig::from_blob(&[0u8; 16]).is_none());
        assert!(VkWorkgroupConfig::from_blob(&[0u8; 18]).is_none());
    }
}

//! Vulkan-side [`BackendMatmul`] adapter.
//!
//! Phase 2.3 of #1082. Resolves a [`VkWorkgroupConfig`] for the
//! requested shape and packs it as the cache `algo_blob`. The
//! actual `vkCmdDispatch` lands behind the Vulkan feature when the
//! `kiln-vulkan-kernel` matmul wrapper extension ships.

use kiln_blas::{AlgoCacheValue, BackendMatmul, MatmulOutcome, MatmulRequest};

use crate::{VkCooperativeMatrixSupport, VkWorkgroupConfig};

/// Vulkan-side adapter. Knows about cooperative-matrix availability;
/// the resolved [`VkWorkgroupConfig`] is what runtime dispatch
/// consumes.
#[derive(Debug, Clone)]
pub struct VulkanBackendMatmul {
    /// What the runtime has detected for cooperative-matrix support.
    /// On startup, kiln-vulkan-kernel queries
    /// `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and writes
    /// the result here.
    pub coop_matrix: VkCooperativeMatrixSupport,
}

impl Default for VulkanBackendMatmul {
    fn default() -> Self {
        VulkanBackendMatmul {
            coop_matrix: VkCooperativeMatrixSupport::Unavailable,
        }
    }
}

impl VulkanBackendMatmul {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_coop_matrix(coop: VkCooperativeMatrixSupport) -> Self {
        VulkanBackendMatmul { coop_matrix: coop }
    }

    /// Heuristic workgroup pick. When cooperative-matrix is
    /// available + the shape is large, use a 256-thread workgroup
    /// with `uses_cooperative_matrix=true`. Smaller / unsupported
    /// paths use the 64-thread DEFAULT.
    fn workgroup_for(&self, req: &MatmulRequest) -> VkWorkgroupConfig {
        let large = req.m * req.n >= 1024 * 1024;
        let dtype_supported = match req.dtype.as_str() {
            "bf16" => self.coop_matrix.supports_bf16(),
            "f16" => self.coop_matrix.supports_fp16(),
            _ => false,
        };
        let use_coop = dtype_supported && large;
        if use_coop {
            VkWorkgroupConfig {
                local_x: 32,
                local_y: 8,
                local_z: 1,
                subgroup_size: 32,
                uses_cooperative_matrix: true,
            }
        } else if large {
            VkWorkgroupConfig {
                local_x: 128,
                local_y: 1,
                local_z: 1,
                subgroup_size: 32,
                uses_cooperative_matrix: false,
            }
        } else {
            VkWorkgroupConfig::DEFAULT
        }
    }
}

impl BackendMatmul for VulkanBackendMatmul {
    fn backend_name(&self) -> &'static str {
        "vulkan"
    }

    fn plan(&self, req: &MatmulRequest) -> MatmulOutcome {
        let cfg = self.workgroup_for(req);
        let bytes_per_element: u64 = match req.dtype.as_str() {
            "f32" => 4,
            "bf16" | "f16" => 2,
            "u8" | "f8_e4m3" | "f8_e5m2" => 1,
            _ => 4,
        };
        MatmulOutcome {
            bytes_written: req.m * req.n * bytes_per_element,
            elapsed_ms: None,
            algo_blob: AlgoCacheValue {
                algo_id: -1,
                workspace_bytes: 0,
                recorded_ms: 0.0,
                algo_blob: cfg.to_blob(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_blas::{Epilogue, MatmulLayout, MatmulRequest};

    fn req(m: u64, n: u64, k: u64) -> MatmulRequest {
        MatmulRequest {
            m,
            n,
            k,
            dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        }
    }

    #[test]
    fn backend_name_is_vulkan() {
        assert_eq!(VulkanBackendMatmul::new().backend_name(), "vulkan");
    }

    #[test]
    fn coop_matrix_unsupported_falls_back_to_scalar() {
        let h = VulkanBackendMatmul::new();
        let outcome = h.plan(&req(2048, 18432, 2560));
        let cfg = VkWorkgroupConfig::from_blob(&outcome.algo_blob.algo_blob).unwrap();
        assert!(!cfg.uses_cooperative_matrix);
    }

    #[test]
    fn coop_matrix_supported_uses_it_for_large_bf16_shapes() {
        let h = VulkanBackendMatmul::with_coop_matrix(
            VkCooperativeMatrixSupport::Bf16Fp32,
        );
        let outcome = h.plan(&req(2048, 18432, 2560));
        let cfg = VkWorkgroupConfig::from_blob(&outcome.algo_blob.algo_blob).unwrap();
        assert!(cfg.uses_cooperative_matrix);
    }

    #[test]
    fn coop_matrix_supported_skips_coop_for_small_shapes() {
        let h = VulkanBackendMatmul::with_coop_matrix(
            VkCooperativeMatrixSupport::Bf16Fp32,
        );
        let outcome = h.plan(&req(32, 32, 64));
        let cfg = VkWorkgroupConfig::from_blob(&outcome.algo_blob.algo_blob).unwrap();
        assert!(!cfg.uses_cooperative_matrix);
    }

    #[test]
    fn coop_matrix_bf16_only_rejects_f16_request() {
        let h = VulkanBackendMatmul::with_coop_matrix(
            VkCooperativeMatrixSupport::Bf16Fp32,
        );
        let mut r = req(2048, 18432, 2560);
        r.dtype = "f16".to_string();
        let outcome = h.plan(&r);
        let cfg = VkWorkgroupConfig::from_blob(&outcome.algo_blob.algo_blob).unwrap();
        assert!(!cfg.uses_cooperative_matrix);
    }

    #[test]
    fn bytes_written_matches_request_shape() {
        let h = VulkanBackendMatmul::new();
        let outcome = h.plan(&req(8, 16, 64));
        assert_eq!(outcome.bytes_written, 8 * 16 * 2);
    }
}

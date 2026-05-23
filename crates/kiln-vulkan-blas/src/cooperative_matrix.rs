//! `VkCooperativeMatrixSupport` — feature-detection for
//! `VK_KHR_cooperative_matrix`.
//!
//! Per the Phase 2 issue bullet:
//!
//! > **Vulkan: `VK_KHR_cooperative_matrix` for the matmul backend.**
//! > Audit confirms today's `kiln-vulkan-kernel::vk_ops/matmul*` uses
//! > subgroup-scalar arithmetic; cooperative matrix exposes
//! > NVIDIA/AMD/Intel matrix cores via standard Vulkan (same primitive
//! > ggml-vulkan and VkFFT use for 4–8× matmul throughput). Lifts
//! > Vulkan from "works" to "competitive with CUDA at the same shape."

/// Per-device cooperative-matrix capability detection.
///
/// `#[non_exhaustive]` — Phase 8.x may add `Tile16x16` etc. variants
/// once specific tile shapes are bench-tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VkCooperativeMatrixSupport {
    /// `VK_KHR_cooperative_matrix` not present (e.g. older drivers).
    /// Matmul falls back to subgroup-scalar.
    Unavailable,
    /// Extension present with BF16 input + FP32 accumulator support.
    /// The Qwen3.5-4B production path uses this.
    Bf16Fp32,
    /// Extension present with FP16 input + FP32 accumulator support.
    /// Useful for the candle-Mac path during the migration.
    Fp16Fp32,
    /// Extension present with both Bf16Fp32 + Fp16Fp32. Best-case;
    /// matmul handler picks per-shape.
    Bf16AndFp16,
}

impl VkCooperativeMatrixSupport {
    /// Stable short name for logging.
    pub const fn name(self) -> &'static str {
        match self {
            VkCooperativeMatrixSupport::Unavailable => "unavailable",
            VkCooperativeMatrixSupport::Bf16Fp32 => "bf16_fp32",
            VkCooperativeMatrixSupport::Fp16Fp32 => "fp16_fp32",
            VkCooperativeMatrixSupport::Bf16AndFp16 => "bf16_and_fp16",
        }
    }

    /// Can cooperative-matrix BF16 input be used?
    pub const fn supports_bf16(self) -> bool {
        matches!(
            self,
            VkCooperativeMatrixSupport::Bf16Fp32 | VkCooperativeMatrixSupport::Bf16AndFp16
        )
    }

    /// Can cooperative-matrix FP16 input be used?
    pub const fn supports_fp16(self) -> bool {
        matches!(
            self,
            VkCooperativeMatrixSupport::Fp16Fp32 | VkCooperativeMatrixSupport::Bf16AndFp16
        )
    }

    /// Is any cooperative-matrix path available?
    pub const fn is_available(self) -> bool {
        !matches!(self, VkCooperativeMatrixSupport::Unavailable)
    }
}

impl Default for VkCooperativeMatrixSupport {
    fn default() -> Self {
        VkCooperativeMatrixSupport::Unavailable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_strings() {
        assert_eq!(VkCooperativeMatrixSupport::Unavailable.name(), "unavailable");
        assert_eq!(VkCooperativeMatrixSupport::Bf16Fp32.name(), "bf16_fp32");
        assert_eq!(VkCooperativeMatrixSupport::Fp16Fp32.name(), "fp16_fp32");
        assert_eq!(VkCooperativeMatrixSupport::Bf16AndFp16.name(), "bf16_and_fp16");
    }

    #[test]
    fn dtype_support_predicates() {
        // Bf16Fp32 supports BF16 only
        let b = VkCooperativeMatrixSupport::Bf16Fp32;
        assert!(b.supports_bf16());
        assert!(!b.supports_fp16());
        assert!(b.is_available());

        // Fp16Fp32 supports FP16 only
        let f = VkCooperativeMatrixSupport::Fp16Fp32;
        assert!(!f.supports_bf16());
        assert!(f.supports_fp16());
        assert!(f.is_available());

        // Bf16AndFp16 supports both
        let bf = VkCooperativeMatrixSupport::Bf16AndFp16;
        assert!(bf.supports_bf16());
        assert!(bf.supports_fp16());
        assert!(bf.is_available());

        // Unavailable supports neither
        let u = VkCooperativeMatrixSupport::Unavailable;
        assert!(!u.supports_bf16());
        assert!(!u.supports_fp16());
        assert!(!u.is_available());
    }

    #[test]
    fn default_is_unavailable() {
        assert_eq!(
            VkCooperativeMatrixSupport::default(),
            VkCooperativeMatrixSupport::Unavailable
        );
    }
}

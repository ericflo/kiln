//! `kiln_tensor::Device` — the four backends kiln-tensor dispatches over.
//!
//! Replaces `candle_core::Device` at the 91 call sites the Phase 0.1 audit
//! captured: `Device` (30), `Device::Cpu` (24), `Device::Metal` (18),
//! `Device::Cuda` (14), `Device::new_cuda` (5).
//!
//! # Multi-GPU stays out of scope (anti-pattern 12)
//!
//! The variant body for the GPU backends carries a `device_index: usize`
//! so we never hardcode `0`. The Phase 0.6 audit (#1088) showed 126
//! `Device::new_cuda(0)` sites — only 2 production, the rest test/example.
//! Phase 1 callers must reach this enum through a centralized accessor
//! (`kiln_core::device::primary_cuda()` for production; a test-helper for
//! tests). A future TP rewrite swaps the accessor; no per-call-site
//! changes.

use core::fmt;

/// The four backends kiln-tensor dispatches over.
///
/// The variant body carries the device index where applicable. Multi-GPU
/// stays out of scope for #1082 (anti-pattern 12), but the index is in
/// the type so two kiln processes on one box can pin different GPUs and
/// a future TP rewrite does not have to revisit any call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// CPU — the canonical numerical reference. Always available.
    /// Per the issue's DoD: "CPU is a tested backend, not just a
    /// numerical reference."
    Cpu,
    /// CUDA device at the given index. Created via cudarc / candle's
    /// CUDA backend in Phase 1.5.
    Cuda(usize),
    /// Apple Metal device at the given index. Created via objc2-metal
    /// in Phase 1.6.
    Metal(usize),
    /// Vulkan device at the given index. Created via ash in Phase 1.7;
    /// lifts the existing `kiln-vulkan-kernel` VulkanDevice.
    Vulkan(usize),
}

impl Device {
    /// Stable short name. Used in env-var-driven config + bench JSON
    /// `device` fields. Format:
    ///
    /// - `Cpu` → `"cpu"`
    /// - `Cuda(0)` → `"cuda:0"`
    /// - `Metal(0)` → `"metal:0"`
    /// - `Vulkan(0)` → `"vulkan:0"`
    pub fn short_name(self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(i) => format!("cuda:{i}"),
            Device::Metal(i) => format!("metal:{i}"),
            Device::Vulkan(i) => format!("vulkan:{i}"),
        }
    }

    /// Is this device the canonical CPU reference?
    pub const fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Is this device any kind of GPU (CUDA / Metal / Vulkan)?
    pub const fn is_gpu(self) -> bool {
        !self.is_cpu()
    }

    /// Device index. `None` for CPU; `Some(i)` for GPU backends.
    pub const fn index(self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(i) | Device::Metal(i) | Device::Vulkan(i) => Some(i),
        }
    }

    /// Stable backend tag for cross-backend comparisons. Two
    /// `Device::Cuda(i)` with different `i` share a backend; two
    /// `Device::Cuda(0)` and `Device::Metal(0)` do not.
    pub const fn backend(self) -> Backend {
        match self {
            Device::Cpu => Backend::Cpu,
            Device::Cuda(_) => Backend::Cuda,
            Device::Metal(_) => Backend::Metal,
            Device::Vulkan(_) => Backend::Vulkan,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.short_name())
    }
}

/// Cross-backend tag. The variant of [`Device`] with the index stripped.
///
/// Used by parity-test harnesses and the `BackendRuntime` dispatcher:
/// the harness picks one device per backend, then runs the same parity
/// test against all four.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl Backend {
    /// All four backends in canonical iteration order (matches the
    /// `bench-results/parity-tolerance.csv` column order).
    pub const ALL: [Backend; 4] = [Backend::Cpu, Backend::Cuda, Backend::Metal, Backend::Vulkan];

    /// Stable short name — `"cpu"`, `"cuda"`, `"metal"`, `"vulkan"`.
    /// Identical to [`Device::short_name`] minus the `:<index>` suffix.
    pub const fn short_name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_name_format() {
        assert_eq!(Device::Cpu.short_name(), "cpu");
        assert_eq!(Device::Cuda(0).short_name(), "cuda:0");
        assert_eq!(Device::Cuda(3).short_name(), "cuda:3");
        assert_eq!(Device::Metal(1).short_name(), "metal:1");
        assert_eq!(Device::Vulkan(2).short_name(), "vulkan:2");
    }

    #[test]
    fn is_cpu_is_gpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_gpu());
        for d in [Device::Cuda(0), Device::Metal(0), Device::Vulkan(0)] {
            assert!(!d.is_cpu());
            assert!(d.is_gpu());
        }
    }

    #[test]
    fn index_of() {
        assert_eq!(Device::Cpu.index(), None);
        assert_eq!(Device::Cuda(0).index(), Some(0));
        assert_eq!(Device::Cuda(5).index(), Some(5));
        assert_eq!(Device::Metal(2).index(), Some(2));
        assert_eq!(Device::Vulkan(1).index(), Some(1));
    }

    #[test]
    fn backend_strips_index() {
        assert_eq!(Device::Cpu.backend(), Backend::Cpu);
        assert_eq!(Device::Cuda(0).backend(), Backend::Cuda);
        assert_eq!(Device::Cuda(3).backend(), Backend::Cuda);
        assert_eq!(Device::Metal(0).backend(), Backend::Metal);
        assert_eq!(Device::Vulkan(2).backend(), Backend::Vulkan);
    }

    #[test]
    fn backend_all_order() {
        let names: Vec<&str> = Backend::ALL.iter().map(|b| b.short_name()).collect();
        assert_eq!(names, ["cpu", "cuda", "metal", "vulkan"]);
    }

    #[test]
    fn display_uses_short_name() {
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
        assert_eq!(format!("{}", Backend::Vulkan), "vulkan");
    }
}

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
    /// AMD ROCm/HIP device at the given index. Created via the `kiln-hip`
    /// binding (Phase R.1); storage in `rocm_storage.rs` (Phase R.3).
    Rocm(usize),
}

impl Device {
    /// Stable short name. Used in env-var-driven config + bench JSON
    /// `device` fields. Format:
    ///
    /// - `Cpu` → `"cpu"`
    /// - `Cuda(0)` → `"cuda:0"`
    /// - `Metal(0)` → `"metal:0"`
    /// - `Vulkan(0)` → `"vulkan:0"`
    /// - `Rocm(0)` → `"rocm:0"`
    pub fn short_name(self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(i) => format!("cuda:{i}"),
            Device::Metal(i) => format!("metal:{i}"),
            Device::Vulkan(i) => format!("vulkan:{i}"),
            Device::Rocm(i) => format!("rocm:{i}"),
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
            Device::Cuda(i) | Device::Metal(i) | Device::Vulkan(i) | Device::Rocm(i) => Some(i),
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
            Device::Rocm(_) => Backend::Rocm,
        }
    }

    /// Physical-device locator, mirroring `candle_core::Device::location`.
    ///
    /// candle distinguishes a logical [`Device`] (of which several can
    /// share one physical device, e.g. CUDA streams) from a
    /// [`DeviceLocation`] (the physical device). `forward.rs` uses
    /// `device().location()` two ways after the #1082 flip:
    ///
    /// 1. In `{:?}` error messages — any `Debug` locator suffices.
    /// 2. Destructured as `DeviceLocation::Cuda { gpu_id }` to read the
    ///    GPU index when building the paged KV cache.
    ///
    /// To keep both call shapes type-checking, this returns a
    /// [`DeviceLocation`] whose variants match candle's field-for-field
    /// (`Cpu`, `Cuda { gpu_id }`, `Metal { gpu_id }`), plus a kt-only
    /// `Vulkan { gpu_id }` for the fourth backend kt dispatches over.
    pub const fn location(self) -> DeviceLocation {
        match self {
            Device::Cpu => DeviceLocation::Cpu,
            Device::Cuda(gpu_id) => DeviceLocation::Cuda { gpu_id },
            Device::Metal(gpu_id) => DeviceLocation::Metal { gpu_id },
            Device::Vulkan(gpu_id) => DeviceLocation::Vulkan { gpu_id },
            Device::Rocm(gpu_id) => DeviceLocation::Rocm { gpu_id },
        }
    }
}

/// A physical-device locator, mirroring `candle_core::DeviceLocation`.
///
/// candle's enum has `Cpu`, `Cuda { gpu_id }`, `Metal { gpu_id }`; kt
/// adds `Vulkan { gpu_id }` for the fourth backend it dispatches over.
/// The struct-variant field is named `gpu_id` to match candle exactly
/// so flip-site destructuring (`DeviceLocation::Cuda { gpu_id }`)
/// type-checks unchanged. Returned by [`Device::location`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceLocation {
    /// The CPU.
    Cpu,
    /// A CUDA device, identified by its GPU index.
    Cuda {
        /// The CUDA device index.
        gpu_id: usize,
    },
    /// An Apple Metal device, identified by its GPU index.
    Metal {
        /// The Metal device index.
        gpu_id: usize,
    },
    /// A Vulkan device, identified by its GPU index (kt-only — candle
    /// has no Vulkan backend).
    Vulkan {
        /// The Vulkan device index.
        gpu_id: usize,
    },
    /// An AMD ROCm/HIP device, identified by its GPU index (kt-only —
    /// candle has no ROCm backend).
    Rocm {
        /// The ROCm device index.
        gpu_id: usize,
    },
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
    Rocm,
}

impl Backend {
    /// All backends in canonical iteration order (matches the
    /// `bench-results/parity-tolerance.csv` column order). `Rocm` is
    /// appended last so existing column indices are unchanged.
    pub const ALL: [Backend; 5] = [
        Backend::Cpu,
        Backend::Cuda,
        Backend::Metal,
        Backend::Vulkan,
        Backend::Rocm,
    ];

    /// Stable short name — `"cpu"`, `"cuda"`, `"metal"`, `"vulkan"`,
    /// `"rocm"`. Identical to [`Device::short_name`] minus the `:<index>`
    /// suffix.
    pub const fn short_name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
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
        assert_eq!(Device::Rocm(0).short_name(), "rocm:0");
        assert_eq!(Device::Rocm(4).short_name(), "rocm:4");
    }

    #[test]
    fn is_cpu_is_gpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_gpu());
        for d in [Device::Cuda(0), Device::Metal(0), Device::Vulkan(0), Device::Rocm(0)] {
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
        assert_eq!(Device::Rocm(3).index(), Some(3));
    }

    #[test]
    fn backend_strips_index() {
        assert_eq!(Device::Cpu.backend(), Backend::Cpu);
        assert_eq!(Device::Cuda(0).backend(), Backend::Cuda);
        assert_eq!(Device::Cuda(3).backend(), Backend::Cuda);
        assert_eq!(Device::Metal(0).backend(), Backend::Metal);
        assert_eq!(Device::Vulkan(2).backend(), Backend::Vulkan);
        assert_eq!(Device::Rocm(0).backend(), Backend::Rocm);
    }

    #[test]
    fn backend_all_order() {
        let names: Vec<&str> = Backend::ALL.iter().map(|b| b.short_name()).collect();
        assert_eq!(names, ["cpu", "cuda", "metal", "vulkan", "rocm"]);
    }

    #[test]
    fn display_uses_short_name() {
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
        assert_eq!(format!("{}", Backend::Vulkan), "vulkan");
    }

    // --- DeviceLocation (#1082 flip gaps) ------------------------------

    #[test]
    fn location_maps_each_variant() {
        assert_eq!(Device::Cpu.location(), DeviceLocation::Cpu);
        assert_eq!(Device::Cuda(0).location(), DeviceLocation::Cuda { gpu_id: 0 });
        assert_eq!(Device::Cuda(3).location(), DeviceLocation::Cuda { gpu_id: 3 });
        assert_eq!(Device::Metal(1).location(), DeviceLocation::Metal { gpu_id: 1 });
        assert_eq!(
            Device::Vulkan(2).location(),
            DeviceLocation::Vulkan { gpu_id: 2 }
        );
        assert_eq!(Device::Rocm(1).location(), DeviceLocation::Rocm { gpu_id: 1 });
    }

    #[test]
    fn location_destructures_gpu_id() {
        // The exact flip-site pattern: `match dev.location() {
        //   DeviceLocation::Cuda { gpu_id } => ... }`.
        let gpu_id = match Device::Cuda(5).location() {
            DeviceLocation::Cuda { gpu_id } => gpu_id,
            other => panic!("expected Cuda location, got {other:?}"),
        };
        assert_eq!(gpu_id, 5);
    }

    #[test]
    fn location_is_debug_for_error_messages() {
        // forward.rs `{:?}`-prints `device().location()` in bail! messages.
        assert_eq!(format!("{:?}", Device::Cpu.location()), "Cpu");
        assert_eq!(
            format!("{:?}", Device::Cuda(0).location()),
            "Cuda { gpu_id: 0 }"
        );
        assert_eq!(
            format!("{:?}", Device::Metal(2).location()),
            "Metal { gpu_id: 2 }"
        );
    }
}

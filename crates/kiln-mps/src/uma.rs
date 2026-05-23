//! `MpsUmaHint` — Apple-Silicon UMA storage-mode hint for matmul inputs.
//!
//! Per the Phase 1 issue bullet on Apple Silicon UMA:
//!
//! > On M-series, CPU and GPU share physical memory; `MTLStorageModeShared`
//! > buffers are addressable from both. kiln-tensor exposes
//! > `Tensor::is_unified_memory()` and `Tensor::as_host_slice()`
//! > (zero-copy on UMA, errors elsewhere) so the safetensors loader
//! > and the optimizer don't pay a copy round-trip on Mac.
//!
//! The MPS matmul path picks a storage mode for its inputs / output
//! based on whether they will be touched from the CPU side. This hint
//! crystallizes that policy in a typed enum that the per-shape algo
//! cache can record.

/// Apple-Silicon UMA storage hint.
///
/// `#[non_exhaustive]` — Phase 8.x may add `Managed` (mac-only,
/// discrete-GPU optimization for the rare non-Apple-Silicon path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MpsUmaHint {
    /// Touched only from the GPU. Use `MTLStorageModePrivate` for
    /// fastest GPU access. Default for intermediate matmul outputs.
    PrivateGpuOnly,
    /// Touched from both CPU and GPU (host upload, host readback).
    /// Use `MTLStorageModeShared` — zero-copy on UMA.
    SharedUma,
    /// Optimizer master / activation offload buffer; we'd ideally
    /// keep this in pinned-host memory and let the GPU read across
    /// the unified-memory boundary. Mac-specific Phase 6.5 hook.
    HostMirror,
}

impl MpsUmaHint {
    /// Stable short name (for parity-tolerance.csv keying + JSON).
    pub const fn name(self) -> &'static str {
        match self {
            MpsUmaHint::PrivateGpuOnly => "private_gpu_only",
            MpsUmaHint::SharedUma => "shared_uma",
            MpsUmaHint::HostMirror => "host_mirror",
        }
    }

    /// Returns `true` iff this hint requires a Shared / UMA-visible
    /// buffer. The MPS storage-mode picker reads this.
    pub const fn needs_uma_visible(self) -> bool {
        matches!(self, MpsUmaHint::SharedUma | MpsUmaHint::HostMirror)
    }
}

impl Default for MpsUmaHint {
    fn default() -> Self {
        MpsUmaHint::PrivateGpuOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_strings() {
        assert_eq!(MpsUmaHint::PrivateGpuOnly.name(), "private_gpu_only");
        assert_eq!(MpsUmaHint::SharedUma.name(), "shared_uma");
        assert_eq!(MpsUmaHint::HostMirror.name(), "host_mirror");
    }

    #[test]
    fn uma_visible_classification() {
        assert!(!MpsUmaHint::PrivateGpuOnly.needs_uma_visible());
        assert!(MpsUmaHint::SharedUma.needs_uma_visible());
        assert!(MpsUmaHint::HostMirror.needs_uma_visible());
    }

    #[test]
    fn default_is_private() {
        assert_eq!(MpsUmaHint::default(), MpsUmaHint::PrivateGpuOnly);
    }
}

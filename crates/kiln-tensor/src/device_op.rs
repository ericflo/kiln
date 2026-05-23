//! `kiln_tensor::DeviceOp` — the CustomOpN replacement.
//!
//! Replaces candle's `CustomOp1` / `CustomOp2` / `CustomOp3` traits at
//! the 15 `impl CustomOpN for ...` sites Phase 0.2 audited (#1090).
//! Per the issue's Phase 1 bullet:
//!
//! > **`CustomOpN` replacement.** Define `kiln-tensor`'s forward/backward
//! > closure shape so the kernel crates port once (not once-for-forward,
//! > then-again-for-backward). Single integration point with
//! > `kiln-autograd`'s `BackwardOp` trait.
//!
//! # Trait shape
//!
//! Three traits — `DeviceOp1` / `DeviceOp2` / `DeviceOp3` — keyed on
//! input arity. Each has per-backend forward methods returning
//! `Result<Option<Tensor>>`:
//!
//! - `Ok(Some(tensor))` — op produced an output on this backend.
//! - `Ok(None)` — backend fallthrough: try the next backend in the
//!   device's preference order (matches today's `BackendRuntime`
//!   contract at `kiln-model/src/backend/mod.rs:114`).
//! - `Err(...)` — op failed for a real reason; abort the dispatch.
//!
//! Plus three sibling methods:
//!
//! - `name() -> &'static str` — used by NVTX range labels +
//!   bench-results CSV keys.
//! - `determinism() -> Determinism` — per anti-pattern's audit
//!   classification.
//! - `bwd() -> Option<Box<dyn BackwardOp>>` — autograd integration
//!   (None == forward-only op like Marlin per anti-pattern in Phase 2.5).
//!
//! # BackwardOp
//!
//! Bridge type to `kiln-autograd` (Phase 6a). Today's stub trait carries
//! a single `apply` method that walks the tape; the full implementation
//! lands when kiln-autograd is lifted from `vk_autograd.rs:1-173`.

use crate::{Determinism, Device, Result, Tensor};

/// Single-input op (replaces `candle_core::CustomOp1`).
///
/// The Phase 0.2 audit found 4 impls of this arity: `FlceCustomOp`,
/// `VulkanLinearOp`, `GdnGateBetaOp`, `VulkanRmsNormOp` + others.
pub trait DeviceOp1: Send + Sync + core::fmt::Debug {
    /// Stable name. Matches the NVTX range name + the
    /// bench-results parity-tolerance.csv `op` column.
    fn name(&self) -> &'static str;

    /// Determinism category. See [`crate::Determinism`].
    fn determinism(&self) -> Determinism;

    /// CPU forward.
    ///
    /// CPU is the canonical numerical reference; every `DeviceOp` impl
    /// must provide this even if the GPU paths are the production
    /// hot-path. Phase 9's parity tests A/B against the CPU result.
    fn cpu_fwd(&self, input: &Tensor) -> Result<Option<Tensor>>;

    /// CUDA forward. Default: returns `None` (backend fallthrough).
    fn cuda_fwd(&self, _input: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Metal forward. Default: returns `None`.
    fn metal_fwd(&self, _input: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Vulkan forward. Default: returns `None`.
    fn vulkan_fwd(&self, _input: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Backward closure registration. `None` for forward-only ops.
    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Two-input op (replaces `candle_core::CustomOp2`).
///
/// Phase 0.2 found 2 impls at this arity: `CudaSigmoidMulTrainingBf16`,
/// `RmsNormCustomOp`.
pub trait DeviceOp2: Send + Sync + core::fmt::Debug {
    fn name(&self) -> &'static str;
    fn determinism(&self) -> Determinism;

    fn cpu_fwd(&self, a: &Tensor, b: &Tensor) -> Result<Option<Tensor>>;
    fn cuda_fwd(&self, _a: &Tensor, _b: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn metal_fwd(&self, _a: &Tensor, _b: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn vulkan_fwd(&self, _a: &Tensor, _b: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Three-input op (replaces `candle_core::CustomOp3`).
///
/// Phase 0.2 found 9 impls at this arity — the largest cluster:
/// `VulkanLoraOp`, `CudaLoraAddF32`, `CudaLoraLinearBf16`,
/// `CudaLoraAddBf16`, `CudaFlashAttentionTrainingBf16`,
/// `CudaRotaryOneBf16`, `GdnGateGOp`, etc.
pub trait DeviceOp3: Send + Sync + core::fmt::Debug {
    fn name(&self) -> &'static str;
    fn determinism(&self) -> Determinism;

    fn cpu_fwd(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Option<Tensor>>;
    fn cuda_fwd(&self, _a: &Tensor, _b: &Tensor, _c: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn metal_fwd(&self, _a: &Tensor, _b: &Tensor, _c: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn vulkan_fwd(&self, _a: &Tensor, _b: &Tensor, _c: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Backward op registration. Bridges per-op backward closures into
/// `kiln-autograd`'s tape (Phase 6a).
///
/// **Status**: scaffold only. The trait's signature is minimal so
/// existing kernel-crate `bwd` paths have something stable to plug
/// into during the migration. The full method surface — gradient-of
/// inputs, gradient-of outputs, version-counter assertions, etc. —
/// lands when `kiln-autograd` is lifted from `vk_autograd.rs:1-173`.
///
/// Anti-pattern 16 ("in-place mutation invalidates the tape") will be
/// enforced by the version-counter assertions added to `apply` in
/// Phase 6a.
pub trait BackwardOp: Send + Sync + core::fmt::Debug {
    /// Stable name — matches the forward op's `name()`. Used by the
    /// tape's `Display` debug output.
    fn name(&self) -> &'static str;
}

// ----------------------------------------------------------------------
// Helper: pick which backend fwd to call based on a Device.
// ----------------------------------------------------------------------

/// Dispatch a [`DeviceOp1`] to the right backend method based on the
/// input's [`Device`]. CPU fallthrough is automatic: if the GPU-side
/// `*_fwd` returns `None`, we re-dispatch on CPU.
///
/// Returns an error if the CPU fallback itself returns `None` — that
/// indicates a buggy `DeviceOp` impl since CPU is the mandatory
/// reference path.
pub fn dispatch1<Op: DeviceOp1 + ?Sized>(op: &Op, input: &Tensor) -> Result<Tensor> {
    let result = match input.device() {
        Device::Cpu => op.cpu_fwd(input)?,
        Device::Cuda(_) => op.cuda_fwd(input)?,
        Device::Metal(_) => op.metal_fwd(input)?,
        Device::Vulkan(_) => op.vulkan_fwd(input)?,
    };
    if let Some(t) = result {
        return Ok(t);
    }
    // Fallthrough: GPU backend returned None — try CPU.
    if !input.device().is_cpu() {
        if let Some(t) = op.cpu_fwd(input)? {
            return Ok(t);
        }
    }
    Err(crate::Error::Msg(format!(
        "DeviceOp1 {:?}: no backend produced output for device {}",
        op.name(),
        input.device()
    )))
}

/// Dispatch a [`DeviceOp2`] — same semantics as [`dispatch1`].
///
/// Both inputs must be on the same device; mismatched-device dispatch
/// errors per the anti-pattern that storage moves are explicit.
pub fn dispatch2<Op: DeviceOp2 + ?Sized>(op: &Op, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device() != b.device() {
        return Err(crate::Error::Msg(format!(
            "DeviceOp2 {:?}: inputs on different devices: a={}, b={}",
            op.name(),
            a.device(),
            b.device()
        )));
    }
    let result = match a.device() {
        Device::Cpu => op.cpu_fwd(a, b)?,
        Device::Cuda(_) => op.cuda_fwd(a, b)?,
        Device::Metal(_) => op.metal_fwd(a, b)?,
        Device::Vulkan(_) => op.vulkan_fwd(a, b)?,
    };
    if let Some(t) = result {
        return Ok(t);
    }
    if !a.device().is_cpu() {
        if let Some(t) = op.cpu_fwd(a, b)? {
            return Ok(t);
        }
    }
    Err(crate::Error::Msg(format!(
        "DeviceOp2 {:?}: no backend produced output for device {}",
        op.name(),
        a.device()
    )))
}

/// Dispatch a [`DeviceOp3`] — same semantics as [`dispatch1`] / [`dispatch2`].
pub fn dispatch3<Op: DeviceOp3 + ?Sized>(
    op: &Op,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
) -> Result<Tensor> {
    if a.device() != b.device() || a.device() != c.device() {
        return Err(crate::Error::Msg(format!(
            "DeviceOp3 {:?}: inputs on different devices: a={}, b={}, c={}",
            op.name(),
            a.device(),
            b.device(),
            c.device()
        )));
    }
    let result = match a.device() {
        Device::Cpu => op.cpu_fwd(a, b, c)?,
        Device::Cuda(_) => op.cuda_fwd(a, b, c)?,
        Device::Metal(_) => op.metal_fwd(a, b, c)?,
        Device::Vulkan(_) => op.vulkan_fwd(a, b, c)?,
    };
    if let Some(t) = result {
        return Ok(t);
    }
    if !a.device().is_cpu() {
        if let Some(t) = op.cpu_fwd(a, b, c)? {
            return Ok(t);
        }
    }
    Err(crate::Error::Msg(format!(
        "DeviceOp3 {:?}: no backend produced output for device {}",
        op.name(),
        a.device()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Layout, Storage, TensorId};
    use std::sync::Arc;

    /// Minimal op-1 impl that returns its input cloned (verifies the
    /// trait shape compiles + dispatch wires the right backend).
    #[derive(Debug)]
    struct IdentityOp;
    impl DeviceOp1 for IdentityOp {
        fn name(&self) -> &'static str {
            "test/identity"
        }
        fn determinism(&self) -> Determinism {
            Determinism::Constructive
        }
        fn cpu_fwd(&self, input: &Tensor) -> Result<Option<Tensor>> {
            Ok(Some(input.clone()))
        }
    }

    /// Op-2 impl that returns the first input (verifies arity-2 surface).
    #[derive(Debug)]
    struct ReturnFirst;
    impl DeviceOp2 for ReturnFirst {
        fn name(&self) -> &'static str {
            "test/return_first"
        }
        fn determinism(&self) -> Determinism {
            Determinism::Constructive
        }
        fn cpu_fwd(&self, a: &Tensor, _b: &Tensor) -> Result<Option<Tensor>> {
            Ok(Some(a.clone()))
        }
    }

    /// Op that returns None — exercises the fallthrough-error path.
    #[derive(Debug)]
    struct AlwaysNone;
    impl DeviceOp1 for AlwaysNone {
        fn name(&self) -> &'static str {
            "test/always_none"
        }
        fn determinism(&self) -> Determinism {
            Determinism::Constructive
        }
        fn cpu_fwd(&self, _input: &Tensor) -> Result<Option<Tensor>> {
            Ok(None)
        }
    }

    fn cpu_tensor() -> Tensor {
        Tensor::zeros_cpu(vec![2, 3], DType::F32)
    }

    fn fake_cuda_tensor() -> Tensor {
        // Build a Tensor that *claims* Device::Cuda(0) — we don't need
        // a real CUDA context, just the dispatcher to pick the right
        // method.
        //
        // We can't construct a real CudaStorage without `cuda` feature,
        // so we cheat: wrap a CpuStorage but lie about the device by
        // implementing a test-only storage that returns Cuda(0).
        #[derive(Debug)]
        struct LyingCpu(crate::CpuStorage);
        impl crate::StorageBackend for LyingCpu {
            fn device(&self) -> Device {
                Device::Cuda(0)
            }
            fn dtype(&self) -> DType {
                self.0.dtype()
            }
            fn byte_len(&self) -> usize {
                self.0.byte_len()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let cpu = crate::CpuStorage::zeros(DType::F32, 6);
        let storage: Storage = Arc::new(LyingCpu(cpu));
        Tensor::from_parts(storage, Layout::contiguous(vec![2, 3]), TensorId::next()).unwrap()
    }

    #[test]
    fn dispatch1_cpu_input_calls_cpu_fwd() {
        let op = IdentityOp;
        let t = cpu_tensor();
        let out = dispatch1(&op, &t).unwrap();
        assert_eq!(out.shape(), t.shape());
        assert_eq!(out.dtype(), t.dtype());
    }

    #[test]
    fn dispatch1_gpu_falls_back_to_cpu_when_gpu_returns_none() {
        // Default cuda_fwd impl is `Ok(None)`. For a Cuda-device input,
        // dispatch should fall through to cpu_fwd and succeed.
        let op = IdentityOp;
        let t = fake_cuda_tensor();
        let out = dispatch1(&op, &t).unwrap();
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn dispatch1_errors_when_all_backends_return_none() {
        let op = AlwaysNone;
        let t = cpu_tensor();
        let e = dispatch1(&op, &t).unwrap_err();
        assert!(e.to_string().contains("no backend produced output"));
    }

    #[test]
    fn dispatch2_mismatched_devices_errors() {
        let op = ReturnFirst;
        let cpu = cpu_tensor();
        let fake_cuda = fake_cuda_tensor();
        let e = dispatch2(&op, &cpu, &fake_cuda).unwrap_err();
        assert!(e.to_string().contains("different devices"));
    }

    #[test]
    fn dispatch2_same_device_succeeds() {
        let op = ReturnFirst;
        let a = cpu_tensor();
        let b = cpu_tensor();
        let out = dispatch2(&op, &a, &b).unwrap();
        assert_eq!(out.shape(), a.shape());
    }

    #[test]
    fn determinism_classification_is_carried() {
        let op = IdentityOp;
        assert!(op.determinism().is_constructive());
        assert_eq!(op.name(), "test/identity");
    }
}

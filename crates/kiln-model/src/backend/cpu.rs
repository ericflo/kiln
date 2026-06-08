//! Portable fallback backend: every kernel method returns `Ok(None)` so
//! the caller falls back to the candle-op composition that runs on any
//! device. Used on CPU, on Metal until Phase 2 adds a real backend, and
//! as a safe default for any future device.

use super::{
    AttentionBackend, BackendIdentity, BackendMatmulLayout, BackendRuntime, ConvBackend,
    GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend, ReplayBackend, ResidencyBackend,
    SamplingBackend, StartupBackend, TrainingLossBackend, requested_matmul_layout,
};

#[derive(Debug)]
pub struct CpuBackend {
    /// The kt device this backend was constructed for. Returned by the
    /// `BackendRuntime::device()` trait method. (#1082 DoD-100 step 4: the
    /// formerly-cached candle `device` field was dropped — `new` now takes a
    /// `kiln_tensor::Device` directly and `name()` matches on it.)
    device_kt: kiln_tensor::Device,
}

impl CpuBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        Self { device_kt: device }
    }
}

impl BackendIdentity for CpuBackend {
    fn runtime_name(&self) -> &'static str {
        match self.device_kt {
            kiln_tensor::Device::Cpu => "cpu",
            kiln_tensor::Device::Metal(_) => "metal-portable",
            kiln_tensor::Device::Cuda(_) => "cuda-portable",
            kiln_tensor::Device::Vulkan(_) => "vulkan-portable",
            // `kiln_tensor::Device` is #[non_exhaustive]; future device kinds
            // route through the portable CPU fallback backend.
            _ => "portable",
        }
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl StartupBackend for CpuBackend {}

impl AttentionBackend for CpuBackend {}

impl GdnBackend for CpuBackend {}

impl ConvBackend for CpuBackend {}

impl LinearBackend for CpuBackend {
    fn runtime_matmul(
        &self,
        req: &super::capability::MatmulRequest,
        lhs: &kiln_tensor::Tensor,
        rhs: &kiln_tensor::Tensor,
    ) -> anyhow::Result<Option<kiln_tensor::Tensor>> {
        if !matches!(lhs.device(), kiln_tensor::Device::Cpu)
            || !matches!(rhs.device(), kiln_tensor::Device::Cpu)
            || req.out_dtype != lhs.dtype()
            || req.lhs_dtype != req.rhs_dtype
        {
            return Ok(None);
        }

        let Some(layout) = requested_matmul_layout(req, lhs, rhs) else {
            return Ok(None);
        };
        let out = match layout {
            BackendMatmulLayout::Plain => kiln_tensor::ops::matmul(lhs, rhs)?,
            BackendMatmulLayout::LhsTransposed => {
                kiln_tensor::ops::matmul_lhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::RhsTransposed => {
                kiln_tensor::ops::matmul_rhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::BothTransposed => {
                let rank = lhs.rank();
                let lhs_t = lhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                let rhs_t = rhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                kiln_tensor::ops::matmul(&lhs_t, &rhs_t)?
            }
        };
        Ok(Some(out))
    }
}

impl super::residency::ResidentRegistry for CpuBackend {}

impl ResidencyBackend for CpuBackend {}

impl SamplingBackend for CpuBackend {}

impl OptimizerBackend for CpuBackend {}

impl PagedKvBackend for CpuBackend {}

impl ReplayBackend for CpuBackend {}

impl TrainingLossBackend for CpuBackend {}

impl BackendRuntime for CpuBackend {}

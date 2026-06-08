//! Core Metal backend runtime helpers.
//!
//! This module keeps shared Metal runtime utilities out of the operation-heavy
//! backend module: pipeline host identity, kt tensor downcasting, and output
//! tensor allocation.

use anyhow::{Context, Result};

use kiln_tensor::MetalStorage;
use kiln_tensor::metal_types::{MetalCompanion, MetalRawDevice};

/// Host abstraction for the per-device MSL pipeline / library caches.
pub(crate) trait MetalPipelineHost {
    /// The raw substrate device for library / pipeline construction.
    fn pipeline_raw_device(&self) -> &MetalRawDevice;
    /// Stable per-device cache key (`MTLDevice::registryID`).
    fn pipeline_cache_key(&self) -> u64;
}

impl MetalPipelineHost for MetalCompanion {
    fn pipeline_raw_device(&self) -> &MetalRawDevice {
        self.device()
    }

    fn pipeline_cache_key(&self) -> u64 {
        self.device_id()
    }
}

/// Downcast a kt `Tensor`'s storage to `&MetalStorage` so Metal helpers can
/// reach the raw buffer and companion.
#[inline]
pub(super) fn kt_metal(t: &kiln_tensor::Tensor) -> Result<&MetalStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .context("expected a Metal-backed kiln_tensor::Tensor")
}

/// Allocate a fresh contiguous Metal output tensor on the same device as
/// `like`'s companion.
pub(super) fn kt_metal_alloc(
    like: &MetalStorage,
    dtype: kiln_tensor::DType,
    dims: &[usize],
) -> Result<kiln_tensor::Tensor> {
    let companion = like.companion()?;
    let n: usize = dims.iter().product();
    let storage = MetalStorage::zeros_kt(companion.device(), like.device_index(), dtype, n)?;
    kiln_tensor::Tensor::from_parts(
        std::sync::Arc::new(storage),
        kiln_tensor::Layout::contiguous(dims.to_vec()),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| anyhow::anyhow!("kt_metal_alloc: {e}"))
}

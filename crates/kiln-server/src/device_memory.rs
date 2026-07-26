//! Device-allocator memory probes used to cap GPU allocations.
//!
//! `kiln-memory` reports the OS/driver-wide memory snapshot. That is the right
//! common signal for metrics, pressure, and coexisting workloads, but some
//! backend allocators expose a narrower "can this allocation succeed right now?"
//! heap. CUDA and ROCm both provide that via `cuMemGetInfo` / `hipMemGetInfo`.

use kiln_model::{GpuAllocatorMemoryProbe, GpuAllocatorMemoryProbePolicy};
use kiln_tensor::Device;

#[derive(Debug, Clone, Copy)]
pub(crate) struct AllocatorMemorySnapshot {
    /// Bytes this process can try to allocate without exceeding the backend's
    /// live allocator view. For ROCm this includes reusable spare bytes already
    /// reserved by Kiln's own HIP pool, because `hipMemGetInfo` counts those
    /// retained pages as unavailable even though future allocations can reuse
    /// them.
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub source: &'static str,
    pub pool_reserved_bytes: Option<u64>,
    pub pool_used_bytes: Option<u64>,
}

pub(crate) fn allocator_memory_snapshot(
    policy: GpuAllocatorMemoryProbePolicy,
    device: &Device,
) -> Option<AllocatorMemorySnapshot> {
    match policy.probe {
        GpuAllocatorMemoryProbe::None => None,
        GpuAllocatorMemoryProbe::CudaMemGetInfo => cuda_allocator_memory_snapshot(device),
        GpuAllocatorMemoryProbe::RocmMemGetInfo { include_pool_spare } => {
            rocm_allocator_memory_snapshot(device, include_pool_spare)
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_allocator_memory_snapshot(device: &Device) -> Option<AllocatorMemorySnapshot> {
    let Device::Cuda(idx) = *device else {
        return None;
    };
    match kiln_tensor::cuda_mem_get_info(idx) {
        Ok((free, total)) => Some(AllocatorMemorySnapshot {
            free_bytes: free as u64,
            total_bytes: total as u64,
            source: "cuMemGetInfo",
            pool_reserved_bytes: None,
            pool_used_bytes: None,
        }),
        Err(err) => {
            tracing::warn!(
                device = %device.short_name(),
                error = %err,
                "CUDA allocator memory probe failed; falling back to OS memory snapshot"
            );
            None
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_allocator_memory_snapshot(_device: &Device) -> Option<AllocatorMemorySnapshot> {
    None
}

#[cfg(feature = "rocm")]
fn rocm_allocator_memory_snapshot(
    device: &Device,
    include_pool_spare: bool,
) -> Option<AllocatorMemorySnapshot> {
    let Device::Rocm(idx) = *device else {
        return None;
    };
    match kiln_tensor::rocm_mem_get_info(idx) {
        Ok((free, total)) => {
            let (pool_reserved, pool_used) = if include_pool_spare {
                kiln_tensor::rocm_pool_stats(idx)
                    .map(|(reserved, used)| (Some(reserved), Some(used)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
            let pool_spare = pool_reserved
                .zip(pool_used)
                .map(|(reserved, used)| reserved.saturating_sub(used))
                .unwrap_or(0);
            Some(AllocatorMemorySnapshot {
                free_bytes: (free as u64).saturating_add(pool_spare),
                total_bytes: total as u64,
                source: if pool_reserved.is_some() {
                    "hipMemGetInfo+hipMemPool"
                } else {
                    "hipMemGetInfo"
                },
                pool_reserved_bytes: pool_reserved,
                pool_used_bytes: pool_used,
            })
        }
        Err(err) => {
            tracing::warn!(
                device = %device.short_name(),
                error = %err,
                "ROCm allocator memory probe failed; falling back to OS memory snapshot"
            );
            None
        }
    }
}

#[cfg(not(feature = "rocm"))]
fn rocm_allocator_memory_snapshot(
    _device: &Device,
    _include_pool_spare: bool,
) -> Option<AllocatorMemorySnapshot> {
    None
}

pub(crate) fn allocator_safe_available_bytes(
    policy: GpuAllocatorMemoryProbePolicy,
    governor: &kiln_memory::MemoryGovernor,
    device: &Device,
) -> Option<u64> {
    allocator_safe_available_bytes_with_soft_reserved(
        policy,
        device,
        governor.config().floor_bytes,
        governor.soft_reserved_bytes(),
    )
}

/// Initial loading and the all-process residency probe can outlive the cached
/// sample deadline, especially while an outer controller freezes the cgroup.
/// Startup owns synchronous probe I/O, so refresh immediately before admission.
pub(crate) fn refresh_governor_for_kv_admission(
    governor: &kiln_memory::MemoryGovernor,
) -> anyhow::Result<kiln_memory::MemoryGovernorObservation> {
    governor.refresh_startup_capacity();
    let observation = governor.cached_observation();
    anyhow::ensure!(
        observation.sample_status.healthy,
        "selected-device memory observation is unhealthy after the synchronous initial KV admission refresh (stale={}, sampler_required={}, sampler_running={}); refusing to allocate",
        observation.sample_status.stale,
        observation.sample_status.sampler_required,
        observation.sample_status.sampler_running,
    );
    tracing::info!(
        governor_free_gb = observation.snapshot.free_bytes as f64 / 1e9,
        governor_available_gb = observation.available_bytes as f64 / 1e9,
        sample_age_ms = observation.sample_status.age.as_secs_f64() * 1000.0,
        sample_max_age_ms = observation.sample_status.max_age.as_secs_f64() * 1000.0,
        "initial KV admission memory observation refreshed"
    );
    Ok(observation)
}

pub(crate) fn allocator_safe_available_bytes_with_soft_reserved(
    policy: GpuAllocatorMemoryProbePolicy,
    device: &Device,
    floor_bytes: u64,
    soft_reserved_bytes: u64,
) -> Option<u64> {
    let snap = allocator_memory_snapshot(policy, device)?;
    Some(safe_available_bytes_from_free(
        snap.free_bytes,
        floor_bytes,
        soft_reserved_bytes,
    ))
}

pub(crate) fn allocator_kv_budget_bytes_for_fraction(
    policy: GpuAllocatorMemoryProbePolicy,
    governor: &kiln_memory::MemoryGovernor,
    device: &Device,
    fraction: f64,
) -> Option<u64> {
    let snap = allocator_memory_snapshot(policy, device)?;
    Some(kv_budget_bytes_from_free(
        snap.free_bytes,
        governor.config().floor_bytes,
        governor.soft_reserved_bytes(),
        fraction,
    ))
}

fn safe_available_bytes_from_free(
    free_bytes: u64,
    floor_bytes: u64,
    soft_reserved_bytes: u64,
) -> u64 {
    free_bytes
        .saturating_sub(floor_bytes)
        .saturating_sub(soft_reserved_bytes)
}

fn kv_budget_bytes_from_free(
    free_bytes: u64,
    floor_bytes: u64,
    soft_reserved_bytes: u64,
    fraction: f64,
) -> u64 {
    let safe = safe_available_bytes_from_free(free_bytes, floor_bytes, soft_reserved_bytes);
    let fraction = fraction.clamp(0.0, 1.0);
    let fractional = (free_bytes as f64 * fraction) as u64;
    safe.min(fractional)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn safe_available_subtracts_floor_and_reservations() {
        assert_eq!(safe_available_bytes_from_free(16 * GB, GB, 2 * GB), 13 * GB);
        assert_eq!(safe_available_bytes_from_free(2 * GB, 4 * GB, GB), 0);
    }

    #[test]
    fn kv_budget_respects_fraction_and_safety_cap() {
        assert_eq!(
            kv_budget_bytes_from_free(16 * GB, GB, 0, 0.70),
            16 * GB * 7 / 10
        );
        assert_eq!(kv_budget_bytes_from_free(16 * GB, GB, 0, 1.0), 15 * GB);
        assert_eq!(kv_budget_bytes_from_free(16 * GB, GB, 3 * GB, 1.0), 12 * GB);
    }

    #[test]
    fn initial_kv_admission_refresh_replaces_the_prior_cached_budget() {
        struct SequentialSource {
            calls: AtomicUsize,
        }

        impl kiln_memory::MemorySource for SequentialSource {
            fn probe(&self) -> kiln_memory::MemorySnapshot {
                let call = self.calls.fetch_add(1, Ordering::SeqCst);
                let free_bytes = if call == 0 { GB } else { 3 * GB };
                kiln_memory::MemorySnapshot {
                    total_bytes: 4 * GB,
                    used_bytes: 4 * GB - free_bytes,
                    free_bytes,
                    source: kiln_memory::vram::VramSource::None,
                    unified: false,
                    observations: Default::default(),
                }
            }
        }

        let governor = kiln_memory::MemoryGovernor::with_source(
            Box::new(SequentialSource {
                calls: AtomicUsize::new(0),
            }),
            kiln_memory::GovernorConfig::default(),
        );
        assert_eq!(governor.cached_available_bytes(), 0);

        let refreshed = refresh_governor_for_kv_admission(&governor).unwrap();
        assert!(refreshed.sample_status.healthy);
        assert!(!refreshed.sample_status.stale);
        assert_eq!(refreshed.snapshot.free_bytes, 3 * GB);
        assert_eq!(refreshed.available_bytes, 2 * GB);
    }
}

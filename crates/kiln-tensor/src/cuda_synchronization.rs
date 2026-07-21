//! Fixed-cardinality CUDA host-synchronization telemetry.
//!
//! CUDA stream/context waits are correctness boundaries, not generic helper
//! calls. This module makes every production wait choose a stable reason and
//! records its scope, outcome, and host wall time without synchronizing to read
//! the counters.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

#[cfg(feature = "cuda")]
use crate::{Error, Result};

/// Fixed metric dimensions for host-visible CUDA synchronization boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CudaSyncReason {
    ExplicitDeviceDrain,
    ExplicitStreamDrain,
    TensorHandoff,
    ExternalYield,
    InPlaceMutation,
    MemoryReclaim,
    GraphBoundary,
    FullAttentionHandoff,
    ModelHandoff,
    HostReadback,
    AllocationLifetime,
    GlobalStateMutation,
}

impl CudaSyncReason {
    pub const ALL: [Self; CUDA_SYNC_REASON_COUNT] = [
        Self::ExplicitDeviceDrain,
        Self::ExplicitStreamDrain,
        Self::TensorHandoff,
        Self::ExternalYield,
        Self::InPlaceMutation,
        Self::MemoryReclaim,
        Self::GraphBoundary,
        Self::FullAttentionHandoff,
        Self::ModelHandoff,
        Self::HostReadback,
        Self::AllocationLifetime,
        Self::GlobalStateMutation,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitDeviceDrain => "explicit_device_drain",
            Self::ExplicitStreamDrain => "explicit_stream_drain",
            Self::TensorHandoff => "tensor_handoff",
            Self::ExternalYield => "external_yield",
            Self::InPlaceMutation => "in_place_mutation",
            Self::MemoryReclaim => "memory_reclaim",
            Self::GraphBoundary => "graph_boundary",
            Self::FullAttentionHandoff => "full_attention_handoff",
            Self::ModelHandoff => "model_handoff",
            Self::HostReadback => "host_readback",
            Self::AllocationLifetime => "allocation_lifetime",
            Self::GlobalStateMutation => "global_state_mutation",
        }
    }

    #[cfg(any(feature = "cuda", test))]
    const fn index(self) -> usize {
        self as usize
    }
}

pub const CUDA_SYNC_REASON_COUNT: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaSyncReasonStats {
    pub reason: CudaSyncReason,
    pub device_wait_count: u64,
    pub stream_wait_count: u64,
    pub failure_count: u64,
    pub waited_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSyncTelemetrySnapshot {
    pub reasons: [CudaSyncReasonStats; CUDA_SYNC_REASON_COUNT],
}

impl CudaSyncTelemetrySnapshot {
    pub fn total_wait_count(&self) -> u64 {
        self.reasons.iter().fold(0u64, |total, stats| {
            total
                .saturating_add(stats.device_wait_count)
                .saturating_add(stats.stream_wait_count)
        })
    }

    pub fn total_failure_count(&self) -> u64 {
        self.reasons.iter().fold(0u64, |total, stats| {
            total.saturating_add(stats.failure_count)
        })
    }

    pub fn total_waited_ns(&self) -> u64 {
        self.reasons
            .iter()
            .fold(0u64, |total, stats| total.saturating_add(stats.waited_ns))
    }
}

#[derive(Debug)]
struct CudaSyncTelemetry {
    device_wait_counts: [AtomicU64; CUDA_SYNC_REASON_COUNT],
    stream_wait_counts: [AtomicU64; CUDA_SYNC_REASON_COUNT],
    failure_counts: [AtomicU64; CUDA_SYNC_REASON_COUNT],
    waited_ns: [AtomicU64; CUDA_SYNC_REASON_COUNT],
}

impl Default for CudaSyncTelemetry {
    fn default() -> Self {
        Self {
            device_wait_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            stream_wait_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            failure_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            waited_ns: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
enum CudaSyncScope {
    Device,
    Stream,
}

impl CudaSyncTelemetry {
    #[cfg(any(feature = "cuda", test))]
    fn saturating_add(counter: &AtomicU64, value: u64) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(value))
        });
    }

    #[cfg(any(feature = "cuda", test))]
    fn record(&self, reason: CudaSyncReason, scope: CudaSyncScope, waited_ns: u64, failed: bool) {
        let index = reason.index();
        let waits = match scope {
            CudaSyncScope::Device => &self.device_wait_counts[index],
            CudaSyncScope::Stream => &self.stream_wait_counts[index],
        };
        Self::saturating_add(waits, 1);
        Self::saturating_add(&self.waited_ns[index], waited_ns);
        if failed {
            Self::saturating_add(&self.failure_counts[index], 1);
        }
    }

    fn snapshot(&self) -> CudaSyncTelemetrySnapshot {
        CudaSyncTelemetrySnapshot {
            reasons: std::array::from_fn(|index| CudaSyncReasonStats {
                reason: CudaSyncReason::ALL[index],
                device_wait_count: self.device_wait_counts[index].load(Ordering::Relaxed),
                stream_wait_count: self.stream_wait_counts[index].load(Ordering::Relaxed),
                failure_count: self.failure_counts[index].load(Ordering::Relaxed),
                waited_ns: self.waited_ns[index].load(Ordering::Relaxed),
            }),
        }
    }
}

static CUDA_SYNC_TELEMETRY: OnceLock<Mutex<HashMap<usize, Arc<CudaSyncTelemetry>>>> =
    OnceLock::new();

fn telemetry_for(device_index: usize) -> Arc<CudaSyncTelemetry> {
    let registry = CUDA_SYNC_TELEMETRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        registry
            .entry(device_index)
            .or_insert_with(|| Arc::new(CudaSyncTelemetry::default())),
    )
}

#[cfg(feature = "cuda")]
fn timed_synchronize(
    device_index: usize,
    reason: CudaSyncReason,
    scope: CudaSyncScope,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let started = Instant::now();
    let result = synchronize();
    let waited_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    telemetry_for(device_index).record(reason, scope, waited_ns, result.is_err());
    result
}

/// Snapshot one device's synchronization counters without touching the driver.
pub fn cuda_sync_telemetry_snapshot(device_index: usize) -> CudaSyncTelemetrySnapshot {
    telemetry_for(device_index).snapshot()
}

/// Perform and account for a device-wide CUDA context wait.
#[cfg(feature = "cuda")]
pub fn cuda_synchronize_context_for(
    device_index: usize,
    context: &CudaContext,
    reason: CudaSyncReason,
) -> Result<()> {
    timed_synchronize(device_index, reason, CudaSyncScope::Device, || {
        context.synchronize().map_err(|error| {
            Error::Msg(format!(
                "CUDA device synchronization failed for {} on device {device_index}: {error:?}",
                reason.as_str()
            ))
        })
    })
}

/// Perform and account for a CUDA stream wait.
#[cfg(feature = "cuda")]
pub fn cuda_synchronize_stream_for(
    device_index: usize,
    stream: &CudaStream,
    reason: CudaSyncReason,
) -> Result<()> {
    timed_synchronize(device_index, reason, CudaSyncScope::Stream, || {
        stream.synchronize().map_err(|error| {
            Error::Msg(format!(
                "CUDA stream synchronization failed for {} on device {device_index}: {error:?}",
                reason.as_str()
            ))
        })
    })
}

/// Synchronize the primary context's default stream with a fixed reason.
#[cfg(feature = "cuda")]
pub fn cuda_synchronize_default_stream_for(
    device_index: usize,
    reason: CudaSyncReason,
) -> Result<()> {
    let context = crate::primary_cuda_context(device_index)?;
    cuda_synchronize_stream_for(device_index, &context.default_stream(), reason)
}

/// Synchronize a tensor's active stream with a fixed reason.
#[cfg(feature = "cuda")]
pub fn cuda_synchronize_tensor_stream_for(
    tensor: &crate::Tensor,
    stream: &CudaStream,
    reason: CudaSyncReason,
) -> Result<()> {
    let device_index = tensor
        .device()
        .index()
        .ok_or_else(|| Error::Msg("CUDA tensor has no device index".into()))?;
    cuda_synchronize_stream_for(device_index, stream, reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_labels_are_complete_and_unique() {
        let labels = CudaSyncReason::ALL.map(CudaSyncReason::as_str);
        assert_eq!(labels.len(), CUDA_SYNC_REASON_COUNT);
        let unique = labels.into_iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), CUDA_SYNC_REASON_COUNT);
    }

    #[test]
    fn telemetry_accounts_scope_failure_and_saturating_totals() {
        let telemetry = CudaSyncTelemetry::default();
        telemetry.record(
            CudaSyncReason::ExternalYield,
            CudaSyncScope::Device,
            7,
            false,
        );
        telemetry.record(
            CudaSyncReason::ExternalYield,
            CudaSyncScope::Stream,
            11,
            true,
        );
        let snapshot = telemetry.snapshot();
        let stats = snapshot.reasons[CudaSyncReason::ExternalYield.index()];
        assert_eq!(stats.device_wait_count, 1);
        assert_eq!(stats.stream_wait_count, 1);
        assert_eq!(stats.failure_count, 1);
        assert_eq!(stats.waited_ns, 18);
        assert_eq!(snapshot.total_wait_count(), 2);
        assert_eq!(snapshot.total_failure_count(), 1);
        assert_eq!(snapshot.total_waited_ns(), 18);
    }
}

use kiln_memory::{
    AutomaticReclaimStats, CachedSampleStatus, MemoryGovernor, MemoryPressure, MemorySnapshot,
    MemorySnapshotObservations,
};

/// One probe-free governor observation for a control-plane response.
///
/// Request handlers capture this once, then serialize or format only these
/// published values. The background sampler owns driver and operating-system
/// probes; health and metrics endpoints must never refresh it synchronously.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedMemoryGovernorObservation {
    pub snapshot: MemorySnapshot,
    pub available_bytes: u64,
    pub soft_reserved_bytes: u64,
    pub pressure: MemoryPressure,
    pub sample_status: CachedSampleStatus,
    pub automatic_monitor_enabled: bool,
    pub automatic_reclaim: AutomaticReclaimStats,
}

impl CachedMemoryGovernorObservation {
    /// Capture the process governor only when startup has already initialized
    /// it for the same accelerator selected by this server state.
    pub fn capture_global_for(selector: kiln_memory::VramProbeSelector) -> Self {
        let Some(observation) = MemoryGovernor::try_global_cached_observation() else {
            return Self::default();
        };

        // `try_global_cached_observation` returning `Some` proves the OnceLock is
        // initialized, so this accessor cannot construct or probe a source.
        let governor = MemoryGovernor::global();
        if MemoryGovernor::global_configuration().selector != selector {
            return Self::default();
        }
        Self::from_published_observation(governor, observation)
    }

    #[cfg(test)]
    fn capture(governor: &MemoryGovernor) -> Self {
        Self::from_published_observation(governor, governor.cached_observation())
    }

    fn from_published_observation(
        governor: &MemoryGovernor,
        observation: kiln_memory::MemoryGovernorObservation,
    ) -> Self {
        Self {
            snapshot: observation.snapshot,
            available_bytes: observation.available_bytes,
            soft_reserved_bytes: observation.soft_reserved_bytes,
            pressure: observation.pressure,
            sample_status: observation.sample_status,
            automatic_monitor_enabled: governor.monitor_started(),
            automatic_reclaim: governor.automatic_reclaim_stats(),
        }
    }
}

impl Default for CachedMemoryGovernorObservation {
    fn default() -> Self {
        Self {
            snapshot: MemorySnapshot {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                source: kiln_memory::vram::VramSource::None,
                unified: false,
                observations: MemorySnapshotObservations::default(),
            },
            available_bytes: 0,
            soft_reserved_bytes: 0,
            pressure: MemoryPressure::Critical,
            sample_status: CachedSampleStatus::default(),
            automatic_monitor_enabled: false,
            automatic_reclaim: AutomaticReclaimStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use kiln_memory::{GovernorConfig, MemorySource};

    use super::*;

    struct CountingSource {
        probes: Arc<AtomicUsize>,
        snapshot: MemorySnapshot,
    }

    impl MemorySource for CountingSource {
        fn probe(&self) -> MemorySnapshot {
            self.probes.fetch_add(1, Ordering::Relaxed);
            self.snapshot
        }
    }

    #[test]
    fn capture_uses_one_published_snapshot_without_probing() {
        let probes = Arc::new(AtomicUsize::new(0));
        let snapshot = MemorySnapshot {
            total_bytes: 2_000,
            used_bytes: 1_000,
            free_bytes: 1_000,
            source: kiln_memory::vram::VramSource::LinuxDrmSysfs,
            unified: false,
            observations: MemorySnapshotObservations::default(),
        };
        let governor = MemoryGovernor::with_source(
            Box::new(CountingSource {
                probes: Arc::clone(&probes),
                snapshot,
            }),
            GovernorConfig {
                ttl: Duration::ZERO,
                floor_bytes: 100,
                ..GovernorConfig::default()
            },
        );
        let reservation = governor.reserve(50);

        let observation = CachedMemoryGovernorObservation::capture(&governor);

        assert_eq!(probes.load(Ordering::Relaxed), 1);
        assert_eq!(observation.snapshot, snapshot);
        assert_eq!(observation.available_bytes, 850);
        assert_eq!(observation.soft_reserved_bytes, 50);
        assert_eq!(observation.pressure, MemoryPressure::Comfortable);
        assert!(observation.sample_status.healthy);
        assert!(!observation.sample_status.stale);
        assert!(!observation.sample_status.sampler_required);
        assert_eq!(
            observation.available_bytes,
            observation
                .snapshot
                .free_bytes
                .saturating_sub(governor.config().floor_bytes)
                .saturating_sub(observation.soft_reserved_bytes)
        );
        assert!(!observation.automatic_monitor_enabled);
        drop(reservation);
    }
}

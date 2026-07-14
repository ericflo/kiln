//! Startup installation and diagnostics for accelerator execution policy.

use anyhow::{Context, Result};
use serde::Serialize;

#[cfg(feature = "rocm")]
use crate::config::RocmSynchronizationMode;
use crate::config::{ResolvedAcceleratorRuntimePolicy, RocmGraphMode};

/// Stable backend-health detail used when ROCm cannot prove safe cleanup.
pub const ROCM_CLEANUP_QUARANTINE_REASON: &str =
    "ROCm execution or cleanup state is unsafe; the device is quarantined until process restart";

/// Fixed-cardinality counters for one ROCm synchronization reason.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RocmSynchronizationReasonStats {
    pub reason: &'static str,
    pub device_wait_count: u64,
    pub stream_wait_count: u64,
    pub waited_ns: u64,
    pub skipped_count: u64,
}

/// Point-in-time ROCm synchronization telemetry.
///
/// Reading this object loads atomics from the already-created primary context.
/// It never synchronizes the device or probes the driver.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RocmSynchronizationRuntimeStats {
    pub active: bool,
    pub telemetry_available: bool,
    /// A fatal execution or cleanup failure left possibly in-flight HIP
    /// resources unsafe to destroy. Execution remains fail-closed until the
    /// process restarts; later recovery drains cannot clear this device flag.
    pub cleanup_quarantined: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_error: Option<String>,
    pub total_device_wait_count: u64,
    pub total_stream_wait_count: u64,
    pub total_waited_ns: u64,
    pub total_skipped_count: u64,
    pub reasons: Vec<RocmSynchronizationReasonStats>,
}

impl RocmSynchronizationRuntimeStats {
    fn inactive() -> Self {
        Self {
            active: false,
            telemetry_available: false,
            cleanup_quarantined: false,
            telemetry_error: None,
            total_device_wait_count: 0,
            total_stream_wait_count: 0,
            total_waited_ns: 0,
            total_skipped_count: 0,
            reasons: Vec::new(),
        }
    }

    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            active: true,
            telemetry_available: false,
            cleanup_quarantined: false,
            telemetry_error: Some(error.into()),
            total_device_wait_count: 0,
            total_stream_wait_count: 0,
            total_waited_ns: 0,
            total_skipped_count: 0,
            reasons: Vec::new(),
        }
    }

    /// Return the sticky backend-health reason required by this live snapshot.
    /// Inactive non-ROCm backends do not require synchronization telemetry.
    pub fn fail_closed_reason(&self) -> Option<String> {
        if !self.active {
            return None;
        }
        if !self.telemetry_available || self.telemetry_error.is_some() {
            let detail = self
                .telemetry_error
                .as_deref()
                .unwrap_or("the primary ROCm context did not provide a telemetry snapshot");
            return Some(format!(
                "ROCm synchronization telemetry is unavailable; backend safety cannot be established: {detail}"
            ));
        }
        self.cleanup_quarantined
            .then(|| ROCM_CLEANUP_QUARANTINE_REASON.to_string())
    }
}

impl Default for RocmSynchronizationRuntimeStats {
    fn default() -> Self {
        Self::inactive()
    }
}

/// Convert the server's resolved graph policy to the model runner contract.
pub fn model_rocm_graph_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> Result<kiln_model::RocmGraphExecutionPolicy> {
    let mode = match policy.rocm_graph_mode.effective {
        RocmGraphMode::Disabled => kiln_model::RocmGraphExecutionMode::Disabled,
        RocmGraphMode::WarmupThenEager => kiln_model::RocmGraphExecutionMode::WarmupThenEager,
        RocmGraphMode::LazyCaptureReplay => kiln_model::RocmGraphExecutionMode::LazyCaptureReplay,
        RocmGraphMode::Profile => {
            anyhow::bail!(
                "resolved accelerator policy retained the unresolved ROCm graph profile sentinel"
            )
        }
    };
    kiln_model::RocmGraphExecutionPolicy::try_new(
        mode,
        policy.rocm_graph_cache_entries.effective,
        false,
    )
    .context("invalid resolved ROCm graph execution policy")
}

/// Install the immutable ROCm policy before any tensor creates the primary
/// context. Non-ROCm devices are unchanged.
pub fn install_startup_policy(
    device: kiln_tensor::Device,
    policy: ResolvedAcceleratorRuntimePolicy,
) -> Result<()> {
    let kiln_tensor::Device::Rocm(device_index) = device else {
        return Ok(());
    };

    #[cfg(feature = "rocm")]
    {
        let synchronization_mode = match policy.rocm_synchronization_mode.effective {
            RocmSynchronizationMode::LegacyHostBarriers => {
                kiln_tensor::RocmSynchronizationMode::LegacyHostBarriers
            }
            RocmSynchronizationMode::StreamOrdered => {
                kiln_tensor::RocmSynchronizationMode::StreamOrdered
            }
        };
        kiln_tensor::primary_rocm_context_with_execution_policy(
            device_index,
            kiln_tensor::RocmExecutionPolicy::new(synchronization_mode),
        )
        .with_context(|| {
            format!(
                "failed to install ROCm execution policy on device {device_index} before initialization"
            )
        })?;
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        let _ = (device_index, policy);
        anyhow::bail!("cannot install a ROCm policy in a build without the `rocm` feature")
    }
}

/// Snapshot reasoned ROCm synchronization counters without waiting on device
/// work. Non-ROCm devices report an inactive, empty object.
pub fn rocm_synchronization_runtime_stats(
    device: kiln_tensor::Device,
) -> RocmSynchronizationRuntimeStats {
    let kiln_tensor::Device::Rocm(device_index) = device else {
        return RocmSynchronizationRuntimeStats::inactive();
    };

    #[cfg(feature = "rocm")]
    {
        let snapshot = match kiln_tensor::rocm_sync_telemetry_snapshot(device_index) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return RocmSynchronizationRuntimeStats::unavailable(format!("{error:#}"));
            }
        };
        let reasons = snapshot
            .reasons
            .iter()
            .map(|stats| RocmSynchronizationReasonStats {
                reason: stats.reason.as_str(),
                device_wait_count: stats.device_wait_count,
                stream_wait_count: stats.stream_wait_count,
                waited_ns: stats.waited_ns,
                skipped_count: stats.skipped_count,
            })
            .collect();
        RocmSynchronizationRuntimeStats {
            active: true,
            telemetry_available: true,
            cleanup_quarantined: snapshot.cleanup_quarantined,
            telemetry_error: None,
            total_device_wait_count: snapshot.reasons.iter().fold(0u64, |total, stats| {
                total.saturating_add(stats.device_wait_count)
            }),
            total_stream_wait_count: snapshot.reasons.iter().fold(0u64, |total, stats| {
                total.saturating_add(stats.stream_wait_count)
            }),
            total_waited_ns: snapshot.total_waited_ns(),
            total_skipped_count: snapshot.total_skipped_count(),
            reasons,
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        let _ = device_index;
        RocmSynchronizationRuntimeStats::unavailable(
            "ROCm device selected in a build without the `rocm` feature",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AcceleratorRuntimeConfig, ServingProfile, ServingProfileSetting};

    #[test]
    fn default_graph_policy_is_eager_for_stable_and_lazy_only_for_experimental() {
        let config = AcceleratorRuntimeConfig::default();
        let stable = config.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Stable,
            crate::config::ConfigValueSource::Default,
        ));
        let experimental = config.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Experimental,
            crate::config::ConfigValueSource::ConfigFile,
        ));

        assert_eq!(
            model_rocm_graph_policy(stable).unwrap().mode(),
            kiln_model::RocmGraphExecutionMode::Disabled
        );
        assert_eq!(
            model_rocm_graph_policy(experimental).unwrap().mode(),
            kiln_model::RocmGraphExecutionMode::LazyCaptureReplay
        );
    }

    #[test]
    fn non_rocm_telemetry_is_inactive_without_creating_a_context() {
        assert_eq!(
            rocm_synchronization_runtime_stats(kiln_tensor::Device::Cpu),
            RocmSynchronizationRuntimeStats::inactive()
        );
    }

    #[test]
    fn runtime_health_reason_is_inert_off_rocm_and_fail_closed_on_rocm_faults() {
        let inactive = RocmSynchronizationRuntimeStats::inactive();
        assert_eq!(inactive.fail_closed_reason(), None);

        let unavailable = RocmSynchronizationRuntimeStats::unavailable("injected query failure");
        let unavailable_reason = unavailable.fail_closed_reason().unwrap();
        assert!(unavailable_reason.contains("telemetry is unavailable"));
        assert!(unavailable_reason.contains("injected query failure"));

        let mut quarantined = RocmSynchronizationRuntimeStats::unavailable("unused");
        quarantined.telemetry_available = true;
        quarantined.telemetry_error = None;
        quarantined.cleanup_quarantined = true;
        assert_eq!(
            quarantined.fail_closed_reason().as_deref(),
            Some(ROCM_CLEANUP_QUARANTINE_REASON)
        );
    }
}

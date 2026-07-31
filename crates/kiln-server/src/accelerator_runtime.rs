//! Startup installation and diagnostics for accelerator execution policy.

use anyhow::{Context, Result};
use serde::Serialize;

#[cfg(feature = "metal")]
use crate::config::MetalKernelProfile;
#[cfg(feature = "rocm")]
use crate::config::RocmSynchronizationMode;
#[cfg(feature = "cuda")]
use crate::config::{CudaFlashBackwardMode, CudaKernelProfile, CudaMarlinProfile};
use crate::config::{KtApiMode, ResolvedAcceleratorRuntimePolicy, RocmGraphMode};

/// Stable backend-health detail used when ROCm cannot prove safe cleanup.
pub const ROCM_CLEANUP_QUARANTINE_REASON: &str =
    "ROCm execution or cleanup state is unsafe; the device is quarantined until process restart";

/// Fixed-cardinality counters for one CUDA synchronization reason.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CudaSynchronizationReasonStats {
    pub reason: &'static str,
    pub device_wait_count: u64,
    pub stream_wait_count: u64,
    pub failure_count: u64,
    pub waited_ns: u64,
}

/// Point-in-time CUDA synchronization telemetry loaded from process atomics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CudaSynchronizationRuntimeStats {
    pub active: bool,
    pub telemetry_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_error: Option<String>,
    pub total_device_wait_count: u64,
    pub total_stream_wait_count: u64,
    pub total_failure_count: u64,
    pub total_waited_ns: u64,
    pub reasons: Vec<CudaSynchronizationReasonStats>,
}

impl CudaSynchronizationRuntimeStats {
    fn inactive() -> Self {
        Self {
            active: false,
            telemetry_available: false,
            telemetry_error: None,
            total_device_wait_count: 0,
            total_stream_wait_count: 0,
            total_failure_count: 0,
            total_waited_ns: 0,
            reasons: Vec::new(),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            active: true,
            telemetry_available: false,
            telemetry_error: Some(error.into()),
            total_device_wait_count: 0,
            total_stream_wait_count: 0,
            total_failure_count: 0,
            total_waited_ns: 0,
            reasons: Vec::new(),
        }
    }
}

impl Default for CudaSynchronizationRuntimeStats {
    fn default() -> Self {
        Self::inactive()
    }
}

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
        policy.rocm_graph_cache_max_bytes.effective,
        false,
    )
    .context("invalid resolved ROCm graph execution policy")
}

fn model_kt_api_mode(policy: ResolvedAcceleratorRuntimePolicy) -> kiln_model::KtApiMode {
    match policy.kt_api_mode.effective {
        KtApiMode::Auto => kiln_model::KtApiMode::Auto,
        KtApiMode::All => kiln_model::KtApiMode::All,
        KtApiMode::Disabled => kiln_model::KtApiMode::Disabled,
    }
}

#[cfg(feature = "cuda")]
fn model_cuda_kernel_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_model::CudaKernelPolicy {
    match policy.cuda_kernel_profile.effective {
        CudaKernelProfile::NativeDefault => kiln_model::CudaKernelPolicy::native_default(),
        CudaKernelProfile::PortableFallback => kiln_model::CudaKernelPolicy::portable_fallback(),
    }
}

#[cfg(feature = "cuda")]
fn model_cuda_marlin_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_model::CudaMarlinPolicy {
    match policy.cuda_marlin_profile.effective {
        CudaMarlinProfile::Disabled => kiln_model::CudaMarlinPolicy::disabled(),
        CudaMarlinProfile::AttentionMlp => kiln_model::CudaMarlinPolicy::attention_mlp(),
        CudaMarlinProfile::AttentionMlpGdn => kiln_model::CudaMarlinPolicy::attention_mlp_gdn(),
    }
}

#[cfg(feature = "cuda")]
fn model_cuda_training_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_model::CudaTrainingPolicy {
    match policy.cuda_flash_backward_mode.effective {
        CudaFlashBackwardMode::Fast => kiln_model::CudaTrainingPolicy::fast(),
        CudaFlashBackwardMode::Deterministic => kiln_model::CudaTrainingPolicy::deterministic(),
    }
}

#[cfg(feature = "metal")]
fn model_metal_kernel_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_model::MetalKernelPolicy {
    match policy.metal_kernel_profile.effective {
        MetalKernelProfile::NativeDefault => kiln_model::MetalKernelPolicy::native_default(),
        MetalKernelProfile::PortableFallback => kiln_model::MetalKernelPolicy::portable_fallback(),
    }
}

#[cfg(feature = "rocm")]
fn model_rocm_kernel_policy(
    policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_model::RocmKernelPolicy {
    match policy.rocm_kernel_profile.effective {
        crate::config::RocmKernelProfile::NativeDefault => {
            kiln_model::RocmKernelPolicy::native_default()
        }
        crate::config::RocmKernelProfile::PortableFallback => {
            kiln_model::RocmKernelPolicy::portable_fallback()
        }
    }
}

#[cfg(feature = "rocm")]
fn tensor_rocm_kernel_policy(
    _policy: ResolvedAcceleratorRuntimePolicy,
) -> kiln_tensor::RocmTensorKernelPolicy {
    kiln_tensor::RocmTensorKernelPolicy::portable_fallback()
}

/// Install accelerator policy needed to select or create a primary device.
///
/// This must run before device probing: Vulkan physical-device selection and
/// validation change which instance/device the process is allowed to create.
pub fn install_pre_device_startup_policy(policy: ResolvedAcceleratorRuntimePolicy) -> Result<()> {
    #[cfg(feature = "vulkan")]
    kiln_model::install_vulkan_device_policy(kiln_model::VulkanDevicePolicy {
        device_index: policy.vulkan_device_index.effective,
        validation: policy.vulkan_validation.effective,
    })
    .context("failed to install Vulkan device policy before device selection")?;

    #[cfg(not(feature = "vulkan"))]
    let _ = policy;

    Ok(())
}

/// Install immutable accelerator policy before model execution or primary
/// device-context creation.
pub fn install_startup_policy(
    device: kiln_tensor::Device,
    policy: ResolvedAcceleratorRuntimePolicy,
) -> Result<()> {
    install_pre_device_startup_policy(policy)?;

    kiln_model::install_full_attention_score_budget_mib(
        policy.full_attention_score_budget_mib.effective,
    )
    .context("failed to install full-attention score-budget policy")?;

    if matches!(
        device,
        kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Rocm(_)
    ) {
        kiln_model::install_kt_api_mode(model_kt_api_mode(policy))
            .context("failed to install kiln-tensor API route policy")?;
    }

    if matches!(device, kiln_tensor::Device::Cuda(_)) {
        #[cfg(feature = "cuda")]
        {
            kiln_model::install_cuda_kernel_policy(model_cuda_kernel_policy(policy))
                .context("failed to install immutable CUDA backend-kernel policy")?;
            kiln_model::install_cuda_marlin_policy(model_cuda_marlin_policy(policy))
                .context("failed to install immutable CUDA Marlin policy")?;
            kiln_model::install_cuda_training_policy(model_cuda_training_policy(policy))
                .context("failed to install immutable CUDA training policy")?;
            return Ok(());
        }

        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("cannot install a CUDA policy in a build without the `cuda` feature");
    }

    if matches!(device, kiln_tensor::Device::Metal(_)) {
        #[cfg(feature = "metal")]
        {
            kiln_model::install_metal_kernel_policy(model_metal_kernel_policy(policy))
                .context("failed to install immutable Metal backend-kernel policy")?;
            return Ok(());
        }

        #[cfg(not(feature = "metal"))]
        anyhow::bail!("cannot install a Metal policy in a build without the `metal` feature");
    }

    let kiln_tensor::Device::Rocm(device_index) = device else {
        return Ok(());
    };

    #[cfg(feature = "rocm")]
    {
        kiln_model::install_rocm_kernel_policy(model_rocm_kernel_policy(policy))
            .context("failed to install immutable ROCm model-kernel policy")?;
        let synchronization_mode = match policy.rocm_synchronization_mode.effective {
            RocmSynchronizationMode::LegacyHostBarriers => {
                kiln_tensor::RocmSynchronizationMode::LegacyHostBarriers
            }
            RocmSynchronizationMode::StreamOrdered => {
                kiln_tensor::RocmSynchronizationMode::StreamOrdered
            }
        };
        let strided_batched_mode = match policy.rocm_strided_batched_matmul_mode.effective {
            crate::config::RocmStridedBatchedMatmulMode::Enabled => {
                kiln_tensor::RocmStridedBatchedMatmulMode::Enabled
            }
            crate::config::RocmStridedBatchedMatmulMode::Disabled => {
                kiln_tensor::RocmStridedBatchedMatmulMode::Disabled
            }
        };
        let bf16_output_mode = match policy.rocm_bf16_matmul_output_mode.effective {
            crate::config::RocmBf16MatmulOutputMode::NativeBf16 => {
                kiln_tensor::RocmBf16MatmulOutputMode::NativeBf16
            }
            crate::config::RocmBf16MatmulOutputMode::F32ThenCast => {
                kiln_tensor::RocmBf16MatmulOutputMode::F32ThenCast
            }
        };
        kiln_tensor::primary_rocm_context_with_execution_policy(
            device_index,
            kiln_tensor::RocmExecutionPolicy::new(synchronization_mode).with_matmul_policy(
                kiln_tensor::RocmMatmulPolicy::new(strided_batched_mode, bf16_output_mode),
            )
            .with_tensor_kernel_policy(tensor_rocm_kernel_policy(policy)),
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

/// Snapshot reasoned CUDA synchronization counters without touching the
/// driver. Non-CUDA devices report an inactive, empty object.
pub fn cuda_synchronization_runtime_stats(
    device: kiln_tensor::Device,
) -> CudaSynchronizationRuntimeStats {
    let kiln_tensor::Device::Cuda(device_index) = device else {
        return CudaSynchronizationRuntimeStats::inactive();
    };

    #[cfg(feature = "cuda")]
    {
        let snapshot = kiln_tensor::cuda_sync_telemetry_snapshot(device_index);
        let reasons = snapshot
            .reasons
            .iter()
            .map(|stats| CudaSynchronizationReasonStats {
                reason: stats.reason.as_str(),
                device_wait_count: stats.device_wait_count,
                stream_wait_count: stats.stream_wait_count,
                failure_count: stats.failure_count,
                waited_ns: stats.waited_ns,
            })
            .collect();
        CudaSynchronizationRuntimeStats {
            active: true,
            telemetry_available: true,
            telemetry_error: None,
            total_device_wait_count: snapshot.reasons.iter().fold(0u64, |total, stats| {
                total.saturating_add(stats.device_wait_count)
            }),
            total_stream_wait_count: snapshot.reasons.iter().fold(0u64, |total, stats| {
                total.saturating_add(stats.stream_wait_count)
            }),
            total_failure_count: snapshot.total_failure_count(),
            total_waited_ns: snapshot.total_waited_ns(),
            reasons,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        CudaSynchronizationRuntimeStats::unavailable(
            "CUDA device selected in a build without the `cuda` feature",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AcceleratorRuntimeConfig, ConfigValueSource, KtApiModeSetting, ServingProfile,
        ServingProfileSetting,
    };
    #[cfg(feature = "cuda")]
    use crate::config::{
        CudaFlashBackwardMode, CudaFlashBackwardModeSetting, CudaKernelProfile,
        CudaKernelProfileSetting, CudaMarlinProfile, CudaMarlinProfileSetting,
    };
    #[cfg(feature = "metal")]
    use crate::config::{MetalKernelProfile, MetalKernelProfileSetting};
    #[test]
    fn kt_api_modes_map_exactly_to_the_model_authority() {
        for (configured, expected) in [
            (KtApiMode::Auto, kiln_model::KtApiMode::Auto),
            (KtApiMode::All, kiln_model::KtApiMode::All),
            (KtApiMode::Disabled, kiln_model::KtApiMode::Disabled),
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.kt_api_mode = KtApiModeSetting::new(configured, ConfigValueSource::ConfigFile);
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Experimental,
                ConfigValueSource::ConfigFile,
            ));
            assert_eq!(model_kt_api_mode(policy), expected);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernel_profiles_map_exactly_to_the_model_policy() {
        for configured in [
            CudaKernelProfile::NativeDefault,
            CudaKernelProfile::PortableFallback,
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.cuda_kernel_profile =
                CudaKernelProfileSetting::new(configured, ConfigValueSource::ConfigFile);
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Stable,
                ConfigValueSource::Default,
            ));
            let expected = match configured {
                CudaKernelProfile::NativeDefault => kiln_model::CudaKernelPolicy::native_default(),
                CudaKernelProfile::PortableFallback => {
                    kiln_model::CudaKernelPolicy::portable_fallback()
                }
            };
            assert_eq!(model_cuda_kernel_policy(policy), expected);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_marlin_profiles_map_exactly_to_the_model_policy() {
        for configured in [
            CudaMarlinProfile::Disabled,
            CudaMarlinProfile::AttentionMlp,
            CudaMarlinProfile::AttentionMlpGdn,
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.cuda_marlin_profile =
                CudaMarlinProfileSetting::new(configured, ConfigValueSource::ConfigFile);
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Stable,
                ConfigValueSource::Default,
            ));
            let expected = match configured {
                CudaMarlinProfile::Disabled => kiln_model::CudaMarlinPolicy::disabled(),
                CudaMarlinProfile::AttentionMlp => kiln_model::CudaMarlinPolicy::attention_mlp(),
                CudaMarlinProfile::AttentionMlpGdn => {
                    kiln_model::CudaMarlinPolicy::attention_mlp_gdn()
                }
            };
            assert_eq!(model_cuda_marlin_policy(policy), expected);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_flash_backward_modes_map_exactly_to_the_model_policy() {
        for configured in [
            CudaFlashBackwardMode::Fast,
            CudaFlashBackwardMode::Deterministic,
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.cuda_flash_backward_mode =
                CudaFlashBackwardModeSetting::new(configured, ConfigValueSource::ConfigFile);
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Stable,
                ConfigValueSource::Default,
            ));
            let expected = match configured {
                CudaFlashBackwardMode::Fast => kiln_model::CudaTrainingPolicy::fast(),
                CudaFlashBackwardMode::Deterministic => {
                    kiln_model::CudaTrainingPolicy::deterministic()
                }
            };
            assert_eq!(model_cuda_training_policy(policy), expected);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_profiles_map_exactly_to_the_model_policy() {
        for configured in [
            MetalKernelProfile::NativeDefault,
            MetalKernelProfile::PortableFallback,
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.metal_kernel_profile =
                MetalKernelProfileSetting::new(configured, ConfigValueSource::ConfigFile);
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Stable,
                ConfigValueSource::Default,
            ));
            let expected = match configured {
                MetalKernelProfile::NativeDefault => {
                    kiln_model::MetalKernelPolicy::native_default()
                }
                MetalKernelProfile::PortableFallback => {
                    kiln_model::MetalKernelPolicy::portable_fallback()
                }
            };
            assert_eq!(model_metal_kernel_policy(policy), expected);
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_product_profiles_map_to_narrow_model_policies() {
        for (configured, expected) in [
            (
                crate::config::RocmKernelProfile::NativeDefault,
                kiln_model::RocmKernelPolicy::native_default(),
            ),
            (
                crate::config::RocmKernelProfile::PortableFallback,
                kiln_model::RocmKernelPolicy::portable_fallback(),
            ),
        ] {
            let mut config = AcceleratorRuntimeConfig::default();
            config.rocm_kernel_profile = crate::config::RocmKernelProfileSetting::new(
                configured,
                ConfigValueSource::ConfigFile,
            );
            let policy = config.resolved_policy(ServingProfileSetting::new(
                ServingProfile::Experimental,
                ConfigValueSource::ConfigFile,
            ));
            assert_eq!(model_rocm_kernel_policy(policy), expected);
            assert_eq!(
                tensor_rocm_kernel_policy(policy),
                kiln_tensor::RocmTensorKernelPolicy::portable_fallback()
            );
        }
        assert_eq!(
            kiln_model::PORTABLE_ROCM_KERNEL_POLICY,
            kiln_model::RocmKernelPolicy::portable_fallback()
        );
    }

    #[test]
    fn default_graph_policy_is_lazy_for_serving_and_off_for_maintenance() {
        let config = AcceleratorRuntimeConfig::default();
        let stable = config.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Stable,
            crate::config::ConfigValueSource::Default,
        ));
        let experimental = config.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Experimental,
            crate::config::ConfigValueSource::ConfigFile,
        ));
        let maintenance = config.resolved_policy(ServingProfileSetting::new(
            ServingProfile::Maintenance,
            crate::config::ConfigValueSource::ConfigFile,
        ));

        assert_eq!(
            model_rocm_graph_policy(stable).unwrap().mode(),
            kiln_model::RocmGraphExecutionMode::LazyCaptureReplay
        );
        assert_eq!(
            model_rocm_graph_policy(experimental).unwrap().mode(),
            kiln_model::RocmGraphExecutionMode::LazyCaptureReplay
        );
        assert_eq!(
            model_rocm_graph_policy(maintenance).unwrap().mode(),
            kiln_model::RocmGraphExecutionMode::Disabled
        );
        assert_eq!(
            model_rocm_graph_policy(experimental)
                .unwrap()
                .max_retained_bytes(),
            crate::config::DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES
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
    fn non_cuda_telemetry_is_inactive_without_creating_a_context() {
        assert_eq!(
            cuda_synchronization_runtime_stats(kiln_tensor::Device::Cpu),
            CudaSynchronizationRuntimeStats::inactive()
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_device_in_non_cuda_build_reports_unavailable_without_driver_access() {
        let stats = cuda_synchronization_runtime_stats(kiln_tensor::Device::Cuda(0));
        assert!(stats.active);
        assert!(!stats.telemetry_available);
        assert!(stats.reasons.is_empty());
        assert!(
            stats
                .telemetry_error
                .as_deref()
                .is_some_and(|error| error.contains("without the `cuda` feature"))
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

//! Immutable startup snapshot of external accelerator visibility controls.
//!
//! These names belong to CUDA/ROCm/oneAPI/Vulkan drivers, not Kiln's public
//! configuration surface. Reading them is a fail-closed device-identity safety
//! check. The first accelerator validation freezes their presence so later
//! inference and training validation cannot observe mutable process state.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub(crate) enum DeviceRemapFamily {
    Nvidia,
    Rocm,
    Intel,
    Vulkan,
}

const NVIDIA_DEVICE_REMAP_ENV: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
];
const ROCM_DEVICE_REMAP_ENV: &[&str] = &[
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
];
const INTEL_DEVICE_REMAP_ENV: &[&str] = &[
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
];
const VULKAN_DEVICE_REMAP_ENV: &[&str] = &[
    "MESA_VK_DEVICE_SELECT",
    "DRI_PRIME",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
];
const ALL_DEVICE_REMAP_ENV: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "MESA_VK_DEVICE_SELECT",
    "DRI_PRIME",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
];

#[derive(Debug)]
struct DeviceRemapEnvironment {
    present: Vec<&'static str>,
}

impl DeviceRemapEnvironment {
    fn capture_with(mut is_present: impl FnMut(&str) -> bool) -> Self {
        Self {
            present: ALL_DEVICE_REMAP_ENV
                .iter()
                .copied()
                .filter(|name| is_present(name))
                .collect(),
        }
    }

    fn present_for(&self, family: DeviceRemapFamily) -> Vec<&'static str> {
        let family_names = match family {
            DeviceRemapFamily::Nvidia => NVIDIA_DEVICE_REMAP_ENV,
            DeviceRemapFamily::Rocm => ROCM_DEVICE_REMAP_ENV,
            DeviceRemapFamily::Intel => INTEL_DEVICE_REMAP_ENV,
            DeviceRemapFamily::Vulkan => VULKAN_DEVICE_REMAP_ENV,
        };
        family_names
            .iter()
            .copied()
            .filter(|name| self.present.contains(name))
            .collect()
    }
}

static DEVICE_REMAP_ENVIRONMENT: OnceLock<DeviceRemapEnvironment> = OnceLock::new();

pub(crate) fn present_device_remap_variables(family: DeviceRemapFamily) -> Vec<&'static str> {
    DEVICE_REMAP_ENVIRONMENT
        .get_or_init(|| {
            DeviceRemapEnvironment::capture_with(|name| std::env::var_os(name).is_some())
        })
        .present_for(family)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn closed_snapshot_preserves_family_order_and_overlap() {
        let family_union = [
            NVIDIA_DEVICE_REMAP_ENV,
            ROCM_DEVICE_REMAP_ENV,
            INTEL_DEVICE_REMAP_ENV,
            VULKAN_DEVICE_REMAP_ENV,
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
        let all = ALL_DEVICE_REMAP_ENV
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(ALL_DEVICE_REMAP_ENV.len(), all.len());
        assert_eq!(all, family_union);

        let snapshot = DeviceRemapEnvironment::capture_with(|name| {
            matches!(
                name,
                "CUDA_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES" | "VK_DRIVER_FILES"
            )
        });
        assert_eq!(
            snapshot.present_for(DeviceRemapFamily::Nvidia),
            vec!["CUDA_VISIBLE_DEVICES"]
        );
        assert_eq!(
            snapshot.present_for(DeviceRemapFamily::Rocm),
            vec!["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        );
        assert!(snapshot.present_for(DeviceRemapFamily::Intel).is_empty());
        assert_eq!(
            snapshot.present_for(DeviceRemapFamily::Vulkan),
            vec!["VK_DRIVER_FILES"]
        );
    }
}

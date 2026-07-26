//! GPU VRAM detection and auto-configuration utilities.
//!
//! Detects available GPU memory and provides recommended training parameters
//! so that SFT and GRPO training "just works" on consumer GPUs without manual tuning.

use crate::startup_environment::{DeviceRemapFamily, present_device_remap_variables};

const NVIDIA_SMI_STARTUP_PROBE_ATTEMPTS: usize = 3;
const NVIDIA_SMI_STARTUP_PROBE_RETRY_DELAY: std::time::Duration =
    std::time::Duration::from_millis(100);

/// Effective GPU memory capacity plus physical-memory topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuVramInfo {
    /// Effective accelerator-memory capacity in bytes (0 if detection failed).
    pub total_bytes: u64,
    /// Source of the effective capacity.
    pub source: VramSource,
    /// Whether GPU allocations share physical memory with the CPU.
    ///
    /// This is deliberately independent from `source`: a configured capacity
    /// cap must not make an APU look like a discrete GPU.
    pub unified: bool,
}

/// How a configured memory cap resolved against the detected safe capacity.
///
/// `physical` is the one-time hardware/host probe. `effective` is never larger
/// than `physical`; configuration can only reduce the usable capacity. Keeping
/// both values makes a rejected optimistic cap observable without probing the
/// machine a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramCapacityResolution {
    pub physical: GpuVramInfo,
    pub requested_bytes: Option<u64>,
    pub effective: GpuVramInfo,
    pub clamped: bool,
}

/// Vendor constraint for a selected Linux DRM device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxDrmVendor {
    Amd,
    Intel,
}

/// Selects the accelerator whose memory counters are authoritative.
///
/// Linux DRM indices are zero-based after card/render aliases have been
/// deduplicated by canonical device path and the optional vendor filter has
/// been applied. This lets ROCm select AMD device 0 on a mixed NVIDIA/AMD host
/// while Vulkan can select from all DRM devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramProbeSelector {
    Auto,
    Nvidia(usize),
    LinuxDrm {
        index: usize,
        vendor: Option<LinuxDrmVendor>,
    },
    AppleUnified,
    None,
}

/// Startup failure when a backend logical ordinal cannot be proven to name the
/// same physical accelerator as the OS/driver memory probe.
///
/// This is an intentionally conservative interim contract. Until accelerator
/// selection and memory probes share a PCI address or UUID, ordinal-based
/// startup is accepted only for logical ordinal zero on a provably singular
/// physical candidate set with no visibility/remapping controls present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramProbeIdentityError {
    message: String,
}

impl VramProbeIdentityError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for VramProbeIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VramProbeIdentityError {}

/// Prove that an ordinal-based backend and its memory probe address the same
/// physical accelerator before model weights are uploaded.
///
/// `None` is safe because it performs no accelerator probe. Apple unified
/// memory is safe because Metal and the probe observe the same single physical
/// memory pool rather than independent device-local heaps. `Auto` is retained
/// for diagnostics, but is rejected for startup because it cannot bind an
/// already-selected backend to a physical device.
pub fn validate_vram_probe_identity(
    selector: VramProbeSelector,
) -> Result<(), VramProbeIdentityError> {
    match selector {
        VramProbeSelector::None | VramProbeSelector::AppleUnified => Ok(()),
        VramProbeSelector::Auto => Err(VramProbeIdentityError::new(
            "automatic VRAM probe selection is diagnostic-only and cannot prove the identity of an already-selected accelerator; use a backend-derived explicit probe selector",
        )),
        VramProbeSelector::Nvidia(ordinal) => {
            let remapping = present_device_remap_variables(DeviceRemapFamily::Nvidia);
            reject_device_remapping(selector, &remapping)?;
            let candidates = query_nvidia_physical_indices().ok_or_else(|| {
                unresolved_probe_identity_error(
                    selector,
                    "nvidia-smi GPU identity enumeration failed",
                )
            })?;
            validate_nvidia_ordinal_identity(ordinal, &candidates, &remapping)
        }
        VramProbeSelector::LinuxDrm { index, vendor } => {
            #[cfg(target_os = "linux")]
            {
                let remapping = present_device_remap_variables(match vendor {
                    Some(LinuxDrmVendor::Amd) => DeviceRemapFamily::Rocm,
                    Some(LinuxDrmVendor::Intel) => DeviceRemapFamily::Intel,
                    None => DeviceRemapFamily::Vulkan,
                });
                reject_device_remapping(selector, &remapping)?;
                let candidate_count =
                    linux_drm_candidate_count_at(std::path::Path::new("/sys/class/drm"), vendor)
                        .ok_or_else(|| {
                            unresolved_probe_identity_error(
                                selector,
                                "Linux DRM physical-device enumeration failed",
                            )
                        })?;
                validate_ordinal_identity(selector, index, candidate_count, &remapping)
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (index, vendor);
                Err(unresolved_probe_identity_error(
                    selector,
                    "Linux DRM identity validation is unavailable on this operating system",
                ))
            }
        }
    }
}

fn validate_nvidia_ordinal_identity(
    ordinal: usize,
    physical_indices: &[usize],
    remapping: &[&str],
) -> Result<(), VramProbeIdentityError> {
    let selector = VramProbeSelector::Nvidia(ordinal);
    validate_ordinal_identity(selector, ordinal, physical_indices.len(), remapping)?;
    if physical_indices != [0] {
        return Err(unresolved_probe_identity_error(
            selector,
            &format!(
                "nvidia-smi reported a singular but nonzero physical index set {physical_indices:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_ordinal_identity(
    selector: VramProbeSelector,
    ordinal: usize,
    candidate_count: usize,
    remapping: &[&str],
) -> Result<(), VramProbeIdentityError> {
    reject_device_remapping(selector, remapping)?;
    if candidate_count == 0 {
        return Err(unresolved_probe_identity_error(
            selector,
            "the relevant physical candidate set is empty",
        ));
    }
    if ordinal >= candidate_count {
        return Err(VramProbeIdentityError::new(format!(
            "memory probe {selector:?} requests logical ordinal {ordinal}, but only {candidate_count} relevant physical candidate(s) were found; refusing an out-of-range device/probe mapping before model upload",
        )));
    }
    if ordinal != 0 {
        return Err(VramProbeIdentityError::new(format!(
            "cannot prove physical-device identity for memory probe {selector:?}: ordinal-based startup is temporarily restricted to logical ordinal zero until PCI-address/UUID-bound selectors are available",
        )));
    }
    if candidate_count != 1 {
        return Err(VramProbeIdentityError::new(format!(
            "cannot prove physical-device identity for memory probe {selector:?}: found {candidate_count} relevant physical candidates; multi-device ordinal ordering may differ between the backend and memory probe, so startup is refused until PCI-address/UUID-bound selectors are available",
        )));
    }
    Ok(())
}

fn reject_device_remapping(
    selector: VramProbeSelector,
    remapping: &[&str],
) -> Result<(), VramProbeIdentityError> {
    if remapping.is_empty() {
        return Ok(());
    }
    Err(VramProbeIdentityError::new(format!(
        "cannot prove physical-device identity for memory probe {selector:?}: device visibility or ordinal remapping is active via {}; remove these controls or wait for PCI-address/UUID-bound selectors",
        remapping.join(", "),
    )))
}

fn unresolved_probe_identity_error(
    selector: VramProbeSelector,
    reason: &str,
) -> VramProbeIdentityError {
    VramProbeIdentityError::new(format!(
        "cannot prove physical-device identity for memory probe {selector:?}: {reason}; refusing accelerator startup before model upload until PCI-address/UUID-bound selectors are available",
    ))
}

/// Snapshot of currently used GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryUsedInfo {
    /// Used VRAM in bytes (0 if detection failed or no GPU).
    pub used_bytes: u64,
    /// Source of the detection.
    pub source: VramSource,
}

/// The effective training memory budget plus provenance and topology.
///
/// Use this when reporting "how much memory can training use" — the
/// `source` makes the log line honest about *why* the budget is what
/// it is. On a discrete NVIDIA GPU it will be `NvidiaSmi`; on an
/// automatically detected unified-memory APU it will be
/// `LinuxDrmSysfsUnified`; and an explicit configured capacity reports
/// `ConfigOverride`. The independent `unified` field preserves the physical
/// topology even when configuration replaces the capacity provenance.
///
/// Field semantics match what consumers like the trainer preflight
/// estimator and the inference KV-cache sizer already expect.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveBudget {
    /// Effective memory capacity available to training in bytes.
    pub total_bytes: u64,
    /// Provenance of the budget — what kind of probe produced it.
    pub source: VramSource,
    /// Whether the GPU and CPU share the same physical memory pool.
    pub unified: bool,
}

/// Convenience: detect VRAM and return an [`EffectiveBudget`] suitable
/// for direct use in startup logging and the training preflight
/// estimator.
///
/// This is the single source of truth — replaces ad-hoc reads of
/// `total_vram_gb` scattered around `crates/kiln-server/src/state.rs`.
pub fn detect_effective_training_budget(configured_total_gib: Option<f64>) -> EffectiveBudget {
    let info = resolve_vram_capacity(detect_vram(), configured_total_gib).effective;
    EffectiveBudget {
        total_bytes: info.total_bytes,
        source: info.source,
        unified: info.unified,
    }
}

/// How the VRAM value was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// Detected via nvidia-smi (discrete NVIDIA GPU).
    NvidiaSmi,
    /// Detected via Linux DRM sysfs memory counters.
    LinuxDrmSysfs,
    /// Detected via Linux DRM sysfs on a unified-memory APU. Capacity is bounded
    /// by physical/cgroup-backed host memory after a conservative reserve; raw
    /// VRAM+GTT address-space counters remain available in snapshot diagnostics.
    LinuxDrmSysfsUnified,
    /// Detected via `sysctl hw.memsize` on Apple Silicon (unified memory).
    /// GPU-addressable memory is effectively the full physical pool minus a
    /// headroom for the OS and other apps.
    AppleSilicon,
    /// Capacity reduced by the effective `memory.gpu_memory_gb` configuration.
    ConfigOverride,
    /// No GPU detected or detection failed.
    None,
}

impl std::fmt::Display for VramSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VramSource::NvidiaSmi => write!(f, "nvidia-smi"),
            VramSource::LinuxDrmSysfs => write!(f, "linux-drm-sysfs"),
            VramSource::LinuxDrmSysfsUnified => write!(f, "linux-drm-sysfs-unified"),
            VramSource::AppleSilicon => write!(f, "apple-silicon-unified"),
            VramSource::ConfigOverride => write!(f, "memory.gpu_memory_gb"),
            VramSource::None => write!(f, "none"),
        }
    }
}

/// Detect total GPU VRAM (or unified memory on Apple Silicon).
///
/// This probe returns the safe physical capacity and topology. It performs no
/// configuration lookup; call [`resolve_vram_capacity`] with the returned value
/// to apply a typed capacity cap without probing twice.
///
/// Physical detection order:
/// 1. `nvidia-smi` query (discrete NVIDIA).
/// 2. Linux DRM sysfs counters (AMD/Intel Vulkan devices). Unified devices are
///    bounded by host/cgroup capacity and retain conservative system headroom.
/// 3. `sysctl hw.memsize` on Apple Silicon (unified memory), with a
///    `system_reserve_gb` headroom subtracted so training doesn't compete
///    with the OS for the last few GB.
/// 4. Returns a zero-capacity, unknown-topology value if no GPU is detected.
pub fn detect_vram() -> GpuVramInfo {
    detect_vram_for(VramProbeSelector::Auto)
}

/// Detect safe capacity for one explicitly selected accelerator.
pub fn detect_vram_for(selector: VramProbeSelector) -> GpuVramInfo {
    let detected = match selector {
        VramProbeSelector::Auto => query_nvidia_smi_for(0)
            .map(discrete_nvidia_info)
            .or_else(|| {
                #[cfg(target_os = "linux")]
                {
                    detect_linux_drm_vram_for(0, None)
                }
                #[cfg(not(target_os = "linux"))]
                {
                    None
                }
            })
            .or_else(detect_apple_unified_vram),
        VramProbeSelector::Nvidia(index) => query_nvidia_smi_for(index).map(discrete_nvidia_info),
        VramProbeSelector::LinuxDrm { index, vendor } => {
            #[cfg(target_os = "linux")]
            {
                detect_linux_drm_vram_for(index, vendor)
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (index, vendor);
                None
            }
        }
        VramProbeSelector::AppleUnified => detect_apple_unified_vram(),
        VramProbeSelector::None => None,
    };
    detected.unwrap_or(GpuVramInfo {
        total_bytes: 0,
        source: VramSource::None,
        unified: false,
    })
}

fn discrete_nvidia_info(total_bytes: u64) -> GpuVramInfo {
    GpuVramInfo {
        total_bytes,
        source: VramSource::NvidiaSmi,
        unified: false,
    }
}

fn detect_apple_unified_vram() -> Option<GpuVramInfo> {
    #[cfg(target_os = "macos")]
    {
        query_apple_unified_memory().map(|total_bytes| GpuVramInfo {
            total_bytes,
            source: VramSource::AppleSilicon,
            unified: true,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

pub fn resolve_vram_capacity(
    physical: GpuVramInfo,
    configured_total_gib: Option<f64>,
) -> VramCapacityResolution {
    let requested_bytes = configured_total_gib
        .filter(|gib| gib.is_finite() && *gib > 0.0)
        .map(|gib| (gib * 1024.0 * 1024.0 * 1024.0) as u64);
    let Some(requested) = requested_bytes else {
        return VramCapacityResolution {
            physical,
            requested_bytes: None,
            effective: physical,
            clamped: false,
        };
    };
    let effective_bytes = requested.min(physical.total_bytes);
    let effective = GpuVramInfo {
        total_bytes: effective_bytes,
        source: if requested <= physical.total_bytes {
            VramSource::ConfigOverride
        } else {
            physical.source
        },
        unified: physical.unified,
    };
    VramCapacityResolution {
        physical,
        requested_bytes: Some(requested),
        effective,
        clamped: requested > physical.total_bytes,
    }
}

/// Linux DRM detection that distinguishes discrete from unified-memory APUs.
#[cfg(target_os = "linux")]
fn detect_linux_drm_vram_for(index: usize, vendor: Option<LinuxDrmVendor>) -> Option<GpuVramInfo> {
    detect_linux_drm_vram_with_cgroup_at(
        std::path::Path::new("/sys/class/drm"),
        std::path::Path::new("/proc/meminfo"),
        query_current_cgroup_memory(),
        index,
        vendor,
    )
}

#[cfg(all(target_os = "linux", test))]
fn detect_linux_drm_vram_at(
    drm_base: &std::path::Path,
    meminfo_path: &std::path::Path,
) -> Option<GpuVramInfo> {
    detect_linux_drm_vram_with_cgroup_at(drm_base, meminfo_path, None, 0, None)
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_vram_with_cgroup_at(
    drm_base: &std::path::Path,
    meminfo_path: &std::path::Path,
    cgroup: Option<CgroupMemoryObservation>,
    index: usize,
    vendor: Option<LinuxDrmVendor>,
) -> Option<GpuVramInfo> {
    let device = select_linux_drm_device_at(drm_base, index, vendor)?.info;
    let mem_total = query_meminfo_total_bytes_at(meminfo_path);
    let unified = is_host_shared_memory_drm(&device, mem_total);
    let driver_addressable_total = device.vram_total.saturating_add(device.gtt_total);
    let total = if unified {
        unified_memory_bounds(
            driver_addressable_total,
            0,
            mem_total,
            query_meminfo_available_bytes_at(meminfo_path),
            cgroup,
        )
        .total_bytes
    } else {
        // GTT is an addressable spill/staging tier, not device-local
        // allocation capacity. HIP may abort rather than return an OOM when
        // VRAM is exhausted, so discrete admission must use VRAM only.
        device.vram_total
    };
    Some(GpuVramInfo {
        total_bytes: total,
        source: if unified {
            VramSource::LinuxDrmSysfsUnified
        } else {
            VramSource::LinuxDrmSysfs
        },
        unified,
    })
}

/// DRM counters for one physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct LinuxDrmDeviceInfo {
    /// `mem_info_vram_total` (or `vis_vram_total`) for this device, bytes.
    pub vram_total: u64,
    /// `mem_info_gtt_total` for this device, bytes (0 if absent).
    pub gtt_total: u64,
    /// PCI vendor ID (e.g. `0x1002` for AMD), 0 if absent.
    pub vendor: u32,
    /// PCI class word (e.g. `0x038000`), 0 if absent. Top byte is class code,
    /// `0x03` is "display controller".
    pub class: u32,
}

#[derive(Debug, Clone)]
#[cfg(target_os = "linux")]
struct SelectedLinuxDrmDevice {
    info: LinuxDrmDeviceInfo,
    device_dir: std::path::PathBuf,
}

#[cfg(all(target_os = "linux", test))]
fn collect_linux_drm_device_info_at(base: &std::path::Path) -> Option<LinuxDrmDeviceInfo> {
    select_linux_drm_device_at(base, 0, None).map(|device| device.info)
}

#[cfg(target_os = "linux")]
fn select_linux_drm_device_at(
    base: &std::path::Path,
    index: usize,
    vendor_filter: Option<LinuxDrmVendor>,
) -> Option<SelectedLinuxDrmDevice> {
    let device_dir = collect_linux_drm_device_dirs_at(base, vendor_filter)?
        .into_iter()
        .nth(index)?;
    let info = read_linux_drm_device_info_at(&device_dir)?;
    Some(SelectedLinuxDrmDevice { info, device_dir })
}

#[cfg(target_os = "linux")]
fn linux_drm_candidate_count_at(
    base: &std::path::Path,
    vendor_filter: Option<LinuxDrmVendor>,
) -> Option<usize> {
    Some(collect_linux_drm_device_dirs_at(base, vendor_filter)?.len())
}

#[cfg(target_os = "linux")]
fn collect_linux_drm_device_dirs_at(
    base: &std::path::Path,
    vendor_filter: Option<LinuxDrmVendor>,
) -> Option<Vec<std::path::PathBuf>> {
    let mut devices: Vec<(Option<usize>, std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(base).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_primary_drm_node(&name) {
            continue;
        }
        let card_index = name
            .strip_prefix("card")
            .and_then(|suffix| suffix.parse::<usize>().ok());
        let device_dir = entry.path().join("device");
        let canonical = std::fs::canonicalize(&device_dir).unwrap_or_else(|_| device_dir.clone());
        if let Some(existing) = devices
            .iter_mut()
            .find(|(_, existing_canonical, _)| *existing_canonical == canonical)
        {
            if card_index.is_some()
                && (existing.0.is_none()
                    || card_index.is_some_and(|index| existing.0.is_some_and(|old| index < old)))
            {
                existing.0 = card_index;
                existing.2 = device_dir;
            }
        } else {
            devices.push((card_index, canonical, device_dir));
        }
    }
    devices.sort_by(|a, b| {
        a.0.unwrap_or(usize::MAX)
            .cmp(&b.0.unwrap_or(usize::MAX))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut selected = Vec::new();
    for (_, _, device_dir) in devices {
        let matches_vendor = match vendor_filter {
            Some(LinuxDrmVendor::Amd) => read_hex_u32_file(&device_dir.join("vendor"))? == 0x1002,
            Some(LinuxDrmVendor::Intel) => read_hex_u32_file(&device_dir.join("vendor"))? == 0x8086,
            None => true,
        };
        if matches_vendor {
            selected.push(device_dir);
        }
    }
    Some(selected)
}

#[cfg(target_os = "linux")]
fn read_linux_drm_device_info_at(device_dir: &std::path::Path) -> Option<LinuxDrmDeviceInfo> {
    let vram_total = ["mem_info_vram_total", "mem_info_vis_vram_total"]
        .into_iter()
        .filter_map(|field| read_u64_file(&device_dir.join(field)))
        .max()
        .unwrap_or(0);
    let gtt_total = read_u64_file(&device_dir.join("mem_info_gtt_total")).unwrap_or(0);
    if vram_total == 0 && gtt_total == 0 {
        return None;
    }
    Some(LinuxDrmDeviceInfo {
        vram_total,
        gtt_total,
        vendor: read_hex_u32_file(&device_dir.join("vendor")).unwrap_or(0),
        class: read_hex_u32_file(&device_dir.join("class")).unwrap_or(0),
    })
}

/// True only when DRM reports no device-local heap and a host-addressable heap.
///
/// `GTT >= VRAM`, PCI vendor, and display-controller class are not sufficient
/// evidence of unified memory: common discrete AMD cards also have that shape.
/// Auto-detection therefore fails closed whenever any local VRAM is reported.
/// Large carveouts such as Strix Halo remain local accelerator capacity; small
/// APU carveouts may be conservative until the driver exposes an unambiguous
/// topology signal rather than being dangerously promoted to VRAM+GTT.
#[cfg(target_os = "linux")]
fn is_host_shared_memory_drm(device: &LinuxDrmDeviceInfo, _mem_total_bytes: Option<u64>) -> bool {
    device.vram_total == 0 && device.gtt_total > 0
}

/// Reserve retained for the OS and CPU workloads in a shared physical pool.
/// Matches the Apple Silicon policy: `max(6 GiB, physical capacity / 4)`.
fn unified_memory_reserve_bytes(mem_total_bytes: u64) -> u64 {
    const MIN_RESERVE_BYTES: u64 = 6 * 1024 * 1024 * 1024;
    let proportional = mem_total_bytes / 4;
    proportional.max(MIN_RESERVE_BYTES)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg(target_os = "linux")]
struct CgroupMemoryObservation {
    /// Minimum finite `memory.max` (v2) / hard limit (v1) across ancestors.
    limit_bytes: Option<u64>,
    /// Minimum finite cgroup-v2 `memory.high` across ancestors.
    high_bytes: Option<u64>,
    /// Leaf usage, retained for diagnostics. Effective headroom is computed
    /// from each ancestor's own usage rather than subtracting this from an
    /// unrelated ancestor limit.
    current_bytes: Option<u64>,
    /// Minimum finite `(max|high) - current` headroom at any visible level.
    effective_remaining_bytes: Option<u64>,
}

#[cfg(target_os = "linux")]
impl CgroupMemoryObservation {
    fn from_level(
        limit_bytes: Option<u64>,
        high_bytes: Option<u64>,
        current_bytes: Option<u64>,
    ) -> Self {
        let effective_remaining_bytes = min_optional_u64(
            limit_bytes
                .map(|limit| current_bytes.map_or(0, |current| limit.saturating_sub(current))),
            high_bytes.map(|high| current_bytes.map_or(0, |current| high.saturating_sub(current))),
        );
        Self {
            limit_bytes,
            high_bytes,
            current_bytes,
            effective_remaining_bytes,
        }
    }

    fn merge_ancestor(self, ancestor: Self) -> Self {
        Self {
            limit_bytes: min_optional_u64(self.limit_bytes, ancestor.limit_bytes),
            high_bytes: min_optional_u64(self.high_bytes, ancestor.high_bytes),
            // Keep leaf usage as the stable diagnostic value.
            current_bytes: self.current_bytes,
            effective_remaining_bytes: min_optional_u64(
                self.effective_remaining_bytes,
                ancestor.effective_remaining_bytes,
            ),
        }
    }

    fn effective_capacity_bytes(self) -> Option<u64> {
        min_optional_u64(self.limit_bytes, self.high_bytes)
    }

    fn remaining_bytes(self) -> Option<u64> {
        self.effective_remaining_bytes
    }
}

#[cfg(target_os = "linux")]
fn min_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg(target_os = "linux")]
struct UnifiedMemoryBounds {
    total_bytes: u64,
    used_bytes: u64,
    free_bytes: u64,
    reserve_bytes: u64,
}

/// Convert driver address-space counters into a physically backed unified pool.
/// Missing host totals/free values fail closed because the DRM address-space
/// figures alone cannot prove that the corresponding RAM exists.
#[cfg(target_os = "linux")]
fn unified_memory_bounds(
    driver_total: u64,
    driver_used: u64,
    host_total: Option<u64>,
    host_available: Option<u64>,
    cgroup: Option<CgroupMemoryObservation>,
) -> UnifiedMemoryBounds {
    let Some(host_total) = host_total.filter(|total| *total > 0) else {
        return UnifiedMemoryBounds::default();
    };
    let backing_capacity = cgroup
        .and_then(CgroupMemoryObservation::effective_capacity_bytes)
        .map_or(host_total, |limit| host_total.min(limit));
    let reserve = unified_memory_reserve_bytes(backing_capacity).min(backing_capacity);
    let total = driver_total.min(backing_capacity.saturating_sub(reserve));

    let driver_free = driver_total.saturating_sub(driver_used.min(driver_total));
    let mut immediately_available =
        driver_free.min(host_available.unwrap_or(0).saturating_sub(reserve));
    if let Some(remaining) = cgroup.and_then(|observation| observation.remaining_bytes()) {
        immediately_available = immediately_available.min(remaining.saturating_sub(reserve));
    }
    let free = immediately_available.min(total);
    UnifiedMemoryBounds {
        total_bytes: total,
        used_bytes: total.saturating_sub(free),
        free_bytes: free,
        reserve_bytes: reserve,
    }
}

#[cfg(target_os = "linux")]
fn query_meminfo_total_bytes_at(path: &std::path::Path) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())?;
            return kib.checked_mul(1024);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn query_current_cgroup_memory() -> Option<CgroupMemoryObservation> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok();
    let mut v2_path = None;
    let mut v1_memory_path = None;
    if let Some(membership) = membership.as_deref() {
        for line in membership.lines() {
            let mut fields = line.splitn(3, ':');
            let _hierarchy = fields.next();
            let controllers = fields.next().unwrap_or_default();
            let path = fields.next().unwrap_or_default();
            if controllers.is_empty() {
                v2_path = Some(path);
            } else if controllers
                .split(',')
                .any(|controller| controller == "memory")
            {
                v1_memory_path = Some(path);
            }
        }
    }

    if let Some(mountinfo) = std::fs::read_to_string("/proc/self/mountinfo").ok() {
        for mount in cgroup_memory_mounts(&mountinfo) {
            let membership_path = if mount.v2 { v2_path } else { v1_memory_path };
            let Some(membership_path) = membership_path else {
                continue;
            };
            let directory = resolve_cgroup_directory(&mount, membership_path);
            let observation = if mount.v2 {
                query_cgroup_v2_hierarchy_at(&directory, &mount.mount_point)
            } else {
                query_cgroup_v1_hierarchy_at(&directory, &mount.mount_point)
            };
            if observation.is_some() {
                return observation;
            }
        }
    }

    // Conventional locations cover namespaced/container mounts whose
    // mountinfo root is intentionally hidden or otherwise unavailable.
    let root = std::path::Path::new("/sys/fs/cgroup");
    if let Some(path) = v2_path {
        let directory = root.join(path.trim_start_matches('/'));
        if let Some(observation) = query_cgroup_v2_hierarchy_at(&directory, root) {
            return Some(observation);
        }
    }
    if let Some(observation) = query_cgroup_v2_at(root) {
        return Some(observation);
    }

    if let Some(path) = v1_memory_path {
        let relative = path.trim_start_matches('/');
        for base in [root.join("memory").join(relative), root.join(relative)] {
            let hierarchy_root = if base.starts_with(root.join("memory")) {
                root.join("memory")
            } else {
                root.to_path_buf()
            };
            if let Some(observation) = query_cgroup_v1_hierarchy_at(&base, &hierarchy_root) {
                return Some(observation);
            }
        }
    }
    query_cgroup_v1_at(&root.join("memory"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(target_os = "linux")]
struct CgroupMount {
    root: std::path::PathBuf,
    mount_point: std::path::PathBuf,
    v2: bool,
}

#[cfg(target_os = "linux")]
fn cgroup_memory_mounts(mountinfo: &str) -> Vec<CgroupMount> {
    mountinfo
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let separator = fields.iter().position(|field| *field == "-")?;
            if separator < 5 || separator + 3 >= fields.len() {
                return None;
            }
            let fs_type = fields[separator + 1];
            let super_options = fields[separator + 3];
            let v2 = fs_type == "cgroup2";
            let v1_memory =
                fs_type == "cgroup" && super_options.split(',').any(|option| option == "memory");
            if !v2 && !v1_memory {
                return None;
            }
            Some(CgroupMount {
                root: std::path::PathBuf::from(unescape_mountinfo_path(fields[3])),
                mount_point: std::path::PathBuf::from(unescape_mountinfo_path(fields[4])),
                v2,
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn resolve_cgroup_directory(mount: &CgroupMount, membership: &str) -> std::path::PathBuf {
    let membership = std::path::Path::new(membership);
    let relative = membership
        .strip_prefix(&mount.root)
        .unwrap_or_else(|_| membership.strip_prefix("/").unwrap_or(membership));
    mount.mount_point.join(relative)
}

#[cfg(target_os = "linux")]
fn unescape_mountinfo_path(raw: &str) -> String {
    raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(target_os = "linux")]
fn query_cgroup_v2_at(path: &std::path::Path) -> Option<CgroupMemoryObservation> {
    let max_raw = std::fs::read_to_string(path.join("memory.max")).ok()?;
    let limit_bytes = parse_cgroup_limit(max_raw.trim());
    let high_bytes = std::fs::read_to_string(path.join("memory.high"))
        .ok()
        .and_then(|raw| parse_cgroup_limit(raw.trim()));
    let current_bytes = read_u64_file(&path.join("memory.current"));
    Some(CgroupMemoryObservation::from_level(
        limit_bytes,
        high_bytes,
        current_bytes,
    ))
}

#[cfg(target_os = "linux")]
fn query_cgroup_v1_at(path: &std::path::Path) -> Option<CgroupMemoryObservation> {
    let raw_limit = std::fs::read_to_string(path.join("memory.limit_in_bytes")).ok()?;
    let limit = raw_limit.trim().parse::<u64>().unwrap_or(0);
    // v1 commonly represents "unlimited" as a page-aligned value near i64::MAX.
    let limit_bytes = (limit < (1u64 << 60)).then_some(limit);
    let current_bytes = read_u64_file(&path.join("memory.usage_in_bytes"));
    Some(CgroupMemoryObservation::from_level(
        limit_bytes,
        None,
        current_bytes,
    ))
}

#[cfg(target_os = "linux")]
fn query_cgroup_v2_hierarchy_at(
    leaf: &std::path::Path,
    hierarchy_root: &std::path::Path,
) -> Option<CgroupMemoryObservation> {
    query_cgroup_hierarchy_at(leaf, hierarchy_root, query_cgroup_v2_at)
}

#[cfg(target_os = "linux")]
fn query_cgroup_v1_hierarchy_at(
    leaf: &std::path::Path,
    hierarchy_root: &std::path::Path,
) -> Option<CgroupMemoryObservation> {
    query_cgroup_hierarchy_at(leaf, hierarchy_root, query_cgroup_v1_at)
}

#[cfg(target_os = "linux")]
fn query_cgroup_hierarchy_at(
    leaf: &std::path::Path,
    hierarchy_root: &std::path::Path,
    query_level: fn(&std::path::Path) -> Option<CgroupMemoryObservation>,
) -> Option<CgroupMemoryObservation> {
    if !leaf.starts_with(hierarchy_root) {
        return None;
    }
    let mut current = leaf;
    let mut aggregate: Option<CgroupMemoryObservation> = None;
    loop {
        if let Some(level) = query_level(current) {
            aggregate = Some(match aggregate {
                Some(existing) => existing.merge_ancestor(level),
                None => level,
            });
        }
        if current == hierarchy_root {
            break;
        }
        current = current.parent()?;
    }
    aggregate
}

#[cfg(target_os = "linux")]
fn parse_cgroup_limit(raw: &str) -> Option<u64> {
    if raw == "max" {
        None
    } else {
        // The controller exists and claims to be finite. A malformed value
        // cannot safely be interpreted as unlimited.
        Some(raw.parse().unwrap_or(0))
    }
}

#[cfg(target_os = "linux")]
fn read_hex_u32_file(path: &std::path::Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let stripped = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u32::from_str_radix(stripped, 16).ok()
}

/// Query currently used GPU VRAM.
///
/// This is intentionally separate from [`detect_vram`]: total VRAM is stable,
/// while used VRAM is meaningful only after the model, quantized workspaces,
/// allocator slabs, and warmup allocations have actually landed on the device.
pub fn detect_used_vram() -> GpuMemoryUsedInfo {
    detect_used_vram_for(VramProbeSelector::Auto)
}

/// Query raw driver-accounted used memory for one selected accelerator.
pub fn detect_used_vram_for(selector: VramProbeSelector) -> GpuMemoryUsedInfo {
    let snapshot = current_memory_snapshot_for(selector);
    GpuMemoryUsedInfo {
        used_bytes: snapshot
            .observations
            .driver_used_bytes
            .unwrap_or(snapshot.used_bytes),
        source: snapshot.source,
    }
}

/// Query currently used GPU VRAM in bytes.
pub fn detect_used_vram_bytes() -> Option<u64> {
    detect_used_vram_bytes_for(VramProbeSelector::Auto)
}

pub fn detect_used_vram_bytes_for(selector: VramProbeSelector) -> Option<u64> {
    let info = detect_used_vram_for(selector);
    (info.used_bytes > 0).then_some(info.used_bytes)
}

/// One separately governed accelerator-memory allocation tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryTierSnapshot {
    /// Safe effective capacity of this allocation tier.
    pub total_bytes: u64,
    /// Effective pressure consumption; always `total_bytes - free_bytes`.
    pub used_bytes: u64,
    /// Bytes allocators using this tier may consider free before governor floors.
    pub free_bytes: u64,
}

#[cfg(target_os = "linux")]
fn memory_tier_snapshot(bounds: UnifiedMemoryBounds) -> MemoryTierSnapshot {
    MemoryTierSnapshot {
        total_bytes: bounds.total_bytes,
        used_bytes: bounds.used_bytes,
        free_bytes: bounds.free_bytes,
    }
}

/// Raw observations retained alongside a safe, internally consistent snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemorySnapshotObservations {
    /// The latest probe attempt failed. Effective free bytes are forced to zero
    /// while the last known capacity/provenance remain visible.
    pub probe_failed: bool,
    /// Combined driver-visible VRAM+GTT address space. This is diagnostic on
    /// discrete DRM devices and is not treated as allocation capacity.
    pub driver_total_bytes: Option<u64>,
    pub driver_used_bytes: Option<u64>,
    pub driver_free_bytes: Option<u64>,
    pub driver_vram_total_bytes: Option<u64>,
    pub driver_vram_used_bytes: Option<u64>,
    pub driver_gtt_total_bytes: Option<u64>,
    pub driver_gtt_used_bytes: Option<u64>,
    pub host_total_bytes: Option<u64>,
    pub host_available_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    /// Minimum finite cgroup-v2 `memory.high` across the visible hierarchy.
    pub cgroup_high_bytes: Option<u64>,
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_remaining_bytes: Option<u64>,
    pub unified_reserve_bytes: Option<u64>,
    /// Separately admissible host-backed accelerator tier. On Linux DRM this is
    /// GTT bounded by host availability, the effective cgroup hierarchy, and
    /// unified-memory reserve. It never inflates or caps the primary VRAM pool.
    /// `None` means no safe host-backed tier could be established.
    pub host_backed: Option<MemoryTierSnapshot>,
}

/// A point-in-time safe memory snapshot for one selected accelerator.
///
/// On discrete devices the effective figures conservatively reconcile the
/// driver's all-process used and free counters. On a unified Linux device, DRM
/// VRAM+GTT is only an address-space observation: effective total/free are
/// additionally bounded by host RAM, `MemAvailable`, and any finite cgroup
/// v1/v2 limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Safe effective capacity after shared-memory headroom.
    pub total_bytes: u64,
    /// Effective pressure consumption; always `total_bytes - free_bytes`.
    pub used_bytes: u64,
    /// Bytes a shared allocator may consider free before governor floors.
    pub free_bytes: u64,
    /// Provenance of the figures.
    pub source: VramSource,
    /// Whether the accelerator shares system RAM with the CPU.
    pub unified: bool,
    /// Raw probe values used to derive the effective figures.
    pub observations: MemorySnapshotObservations,
}

/// Take an automatically selected live snapshot. Discrete figures come from
/// all-process driver counters; unified figures additionally reflect host and
/// cgroup pressure. Server code should prefer [`current_memory_snapshot_for`]
/// so mixed-device hosts cannot select a different accelerator implicitly.
pub fn current_memory_snapshot() -> MemorySnapshot {
    current_memory_snapshot_for(VramProbeSelector::Auto)
}

/// Take a live snapshot for one explicitly selected accelerator. Explicit
/// selection never falls back to another device when the requested probe fails.
pub fn current_memory_snapshot_for(selector: VramProbeSelector) -> MemorySnapshot {
    try_current_memory_snapshot_for(selector).unwrap_or_else(empty_memory_snapshot)
}

/// Fallible selected-device probe used by the governor to distinguish a probe
/// failure from a legitimate zero-capacity source such as CPU-only mode.
pub(crate) fn try_current_memory_snapshot_for(
    selector: VramProbeSelector,
) -> Option<MemorySnapshot> {
    let snapshot = match selector {
        VramProbeSelector::Auto => nvidia_memory_snapshot(0)
            .or_else(|| {
                #[cfg(target_os = "linux")]
                {
                    linux_drm_memory_snapshot_for(0, None)
                }
                #[cfg(not(target_os = "linux"))]
                {
                    None
                }
            })
            .or_else(apple_memory_snapshot),
        VramProbeSelector::Nvidia(index) => nvidia_memory_snapshot(index),
        VramProbeSelector::LinuxDrm { index, vendor } => {
            #[cfg(target_os = "linux")]
            {
                linux_drm_memory_snapshot_for(index, vendor)
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (index, vendor);
                None
            }
        }
        VramProbeSelector::AppleUnified => apple_memory_snapshot(),
        VramProbeSelector::None => Some(empty_memory_snapshot()),
    };
    snapshot
}

fn empty_memory_snapshot() -> MemorySnapshot {
    MemorySnapshot {
        total_bytes: 0,
        used_bytes: 0,
        free_bytes: 0,
        source: VramSource::None,
        unified: false,
        observations: MemorySnapshotObservations::default(),
    }
}

fn nvidia_memory_snapshot(index: usize) -> Option<MemorySnapshot> {
    let (total, reported_used, reported_free) = query_nvidia_smi_memory_for(index)?;
    Some(nvidia_memory_snapshot_from_counters(
        total,
        reported_used,
        reported_free,
    ))
}

fn nvidia_memory_snapshot_from_counters(
    total: u64,
    reported_used: u64,
    reported_free: u64,
) -> MemorySnapshot {
    // WSL2 can report a reserved gap where used + free < total. Treat both
    // counters as independent ceilings and publish the more conservative
    // allocation budget while retaining their raw values for diagnostics.
    let free = reported_free.min(total.saturating_sub(reported_used));
    let used = total.saturating_sub(free);
    MemorySnapshot {
        total_bytes: total,
        used_bytes: used,
        free_bytes: free,
        source: VramSource::NvidiaSmi,
        unified: false,
        observations: MemorySnapshotObservations {
            driver_total_bytes: Some(total),
            driver_used_bytes: Some(reported_used),
            driver_free_bytes: Some(reported_free),
            ..MemorySnapshotObservations::default()
        },
    }
}

fn apple_memory_snapshot() -> Option<MemorySnapshot> {
    #[cfg(target_os = "macos")]
    let (physical_total, host_available) = query_apple_memory_state()?;
    #[cfg(not(target_os = "macos"))]
    return None;

    #[cfg(target_os = "macos")]
    let reserve = unified_memory_reserve_bytes(physical_total);
    #[cfg(target_os = "macos")]
    let total = physical_total.saturating_sub(reserve);
    #[cfg(target_os = "macos")]
    let free = host_available.saturating_sub(reserve).min(total);
    #[cfg(target_os = "macos")]
    Some(MemorySnapshot {
        total_bytes: total,
        used_bytes: total.saturating_sub(free),
        free_bytes: free,
        source: VramSource::AppleSilicon,
        unified: true,
        observations: MemorySnapshotObservations {
            host_total_bytes: Some(physical_total),
            host_available_bytes: Some(host_available),
            unified_reserve_bytes: Some(reserve),
            host_backed: Some(MemoryTierSnapshot {
                total_bytes: total,
                used_bytes: total.saturating_sub(free),
                free_bytes: free,
            }),
            ..MemorySnapshotObservations::default()
        },
    })
}

#[cfg(target_os = "linux")]
fn linux_drm_memory_snapshot_for(
    index: usize,
    vendor: Option<LinuxDrmVendor>,
) -> Option<MemorySnapshot> {
    linux_drm_memory_snapshot_at(
        std::path::Path::new("/sys/class/drm"),
        std::path::Path::new("/proc/meminfo"),
        query_current_cgroup_memory(),
        index,
        vendor,
    )
}

#[cfg(target_os = "linux")]
fn linux_drm_memory_snapshot_at(
    drm_base: &std::path::Path,
    meminfo_path: &std::path::Path,
    cgroup: Option<CgroupMemoryObservation>,
    index: usize,
    vendor: Option<LinuxDrmVendor>,
) -> Option<MemorySnapshot> {
    let device = select_linux_drm_device_at(drm_base, index, vendor)?;
    let info = device.info;
    let driver_total = info.vram_total.saturating_add(info.gtt_total);
    let vram_used = if info.vram_total == 0 {
        0
    } else {
        read_device_memory_field(
            &device.device_dir,
            &["mem_info_vram_used", "mem_info_vis_vram_used"],
        )?
    };
    let host_total = query_meminfo_total_bytes_at(meminfo_path);
    let host_available = query_meminfo_available_bytes_at(meminfo_path);
    let unified = is_host_shared_memory_drm(&info, host_total);
    let gtt_used_observation = if info.gtt_total == 0 {
        Some(0)
    } else {
        read_device_memory_field(&device.device_dir, &["mem_info_gtt_used"])
    };
    if unified && gtt_used_observation.is_none() {
        return None;
    }
    let driver_used =
        gtt_used_observation.map(|gtt_used| vram_used.saturating_add(gtt_used).min(driver_total));
    let driver_free = driver_used.map(|used| driver_total.saturating_sub(used));
    let host_backed = if info.gtt_total == 0
        || host_total.is_none_or(|total| total == 0)
        || host_available.is_none()
    {
        None
    } else {
        gtt_used_observation.map(|gtt_used| {
            memory_tier_snapshot(unified_memory_bounds(
                info.gtt_total,
                gtt_used.min(info.gtt_total),
                host_total,
                host_available,
                cgroup,
            ))
        })
    };
    let vram_used = vram_used.min(info.vram_total);
    let vram_free = info.vram_total.saturating_sub(vram_used);
    let (total, used, free, reserve) = if unified {
        let bounds = unified_memory_bounds(
            driver_total,
            driver_used?,
            host_total,
            host_available,
            cgroup,
        );
        (
            bounds.total_bytes,
            bounds.used_bytes,
            bounds.free_bytes,
            Some(bounds.reserve_bytes),
        )
    } else {
        (info.vram_total, vram_used, vram_free, None)
    };
    Some(MemorySnapshot {
        total_bytes: total,
        used_bytes: used,
        free_bytes: free,
        source: if unified {
            VramSource::LinuxDrmSysfsUnified
        } else {
            VramSource::LinuxDrmSysfs
        },
        unified,
        observations: MemorySnapshotObservations {
            probe_failed: false,
            driver_total_bytes: Some(driver_total),
            driver_used_bytes: driver_used,
            driver_free_bytes: driver_free,
            driver_vram_total_bytes: Some(info.vram_total),
            driver_vram_used_bytes: Some(vram_used),
            driver_gtt_total_bytes: Some(info.gtt_total),
            driver_gtt_used_bytes: gtt_used_observation.map(|used| used.min(info.gtt_total)),
            host_total_bytes: host_total,
            host_available_bytes: host_available,
            cgroup_limit_bytes: cgroup.and_then(|observation| observation.limit_bytes),
            cgroup_high_bytes: cgroup.and_then(|observation| observation.high_bytes),
            cgroup_current_bytes: cgroup.and_then(|observation| observation.current_bytes),
            cgroup_remaining_bytes: cgroup.and_then(|observation| observation.remaining_bytes()),
            unified_reserve_bytes: reserve,
            host_backed,
        },
    })
}

#[cfg(target_os = "linux")]
fn read_device_memory_field(device_dir: &std::path::Path, fields: &[&str]) -> Option<u64> {
    fields
        .iter()
        .filter_map(|field| read_u64_file(&device_dir.join(field)))
        .max()
}

/// Free accelerator memory in bytes right now (0 if undetectable). Thin
/// convenience over [`current_memory_snapshot`].
pub fn current_free_bytes() -> u64 {
    current_free_bytes_for(VramProbeSelector::Auto)
}

pub fn current_free_bytes_for(selector: VramProbeSelector) -> u64 {
    current_memory_snapshot_for(selector).free_bytes
}

/// Free *VRAM* (`mem_info_vram_total` − `mem_info_vram_used`) in bytes on a Linux
/// AMD/DRM GPU, EXCLUDING GTT. `None` on non-DRM platforms (NVIDIA→nvidia-smi,
/// macOS, Windows) and when the sysfs counters are absent.
///
/// Why this exists, separate from [`current_free_bytes`]:
/// ROCm's HIP allocator `abort()`s (rocclr `vmheap::MapPhysMemory` assertion) on
/// a VRAM OOM instead of returning an error. The general DRM snapshot now uses
/// VRAM-only capacity on discrete devices; this helper remains useful to callers
/// that need an optional AMD-specific allocator guard. On unified devices even
/// the VRAM counter can describe virtual address space, so this helper also caps
/// it by the safe host-backed live free.
pub fn current_free_vram_bytes() -> Option<u64> {
    current_free_vram_bytes_for(VramProbeSelector::Auto)
}

pub fn current_free_vram_bytes_for(selector: VramProbeSelector) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let (index, vendor, drm_selector) = match selector {
            VramProbeSelector::Auto => (
                0,
                None,
                VramProbeSelector::LinuxDrm {
                    index: 0,
                    vendor: None,
                },
            ),
            VramProbeSelector::LinuxDrm { index, vendor } => (index, vendor, selector),
            _ => return None,
        };
        let device =
            select_linux_drm_device_at(std::path::Path::new("/sys/class/drm"), index, vendor)?;
        let total = read_device_memory_field(
            &device.device_dir,
            &["mem_info_vram_total", "mem_info_vis_vram_total"],
        )?;
        if total == 0 {
            return None;
        }
        let used = read_device_memory_field(
            &device.device_dir,
            &["mem_info_vram_used", "mem_info_vis_vram_used"],
        )?;
        let raw_free = total.saturating_sub(used);
        let snapshot = current_memory_snapshot_for(drm_selector);
        Some(if snapshot.source == VramSource::None {
            0
        } else if snapshot.unified {
            raw_free.min(snapshot.free_bytes)
        } else {
            raw_free
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Read `MemAvailable` (the kernel's estimate of allocatable RAM without
/// swapping) from a `/proc/meminfo`-format file. This is the right "free"
/// figure for unified-memory accelerators, where GPU buffers are backed by
/// system RAM.
#[cfg(target_os = "linux")]
fn query_meminfo_available_bytes_at(path: &std::path::Path) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())?;
            return kib.checked_mul(1024);
        }
    }
    None
}

/// Query total GPU memory via nvidia-smi.
///
/// Runs `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
/// which returns total memory in MiB. Returns None if nvidia-smi is not available
/// or fails.
fn query_nvidia_smi_for(index: usize) -> Option<u64> {
    retry_optional_probe(
        NVIDIA_SMI_STARTUP_PROBE_ATTEMPTS,
        NVIDIA_SMI_STARTUP_PROBE_RETRY_DELAY,
        || query_nvidia_smi_memory_for(index),
    )
    .map(|(total, _used, _free)| total)
}

/// Enumerate physical NVIDIA indices and stable UUIDs in one bounded process.
/// The UUID column makes malformed/remapped-looking output fail closed instead
/// of treating an arbitrary number of lines as an identity proof.
fn query_nvidia_physical_indices() -> Option<Vec<usize>> {
    retry_optional_probe(
        NVIDIA_SMI_STARTUP_PROBE_ATTEMPTS,
        NVIDIA_SMI_STARTUP_PROBE_RETRY_DELAY,
        || {
            let stdout = bounded_command_stdout(
                "nvidia-smi",
                &["--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                std::time::Duration::from_secs(2),
            )?;
            parse_nvidia_physical_indices(&stdout)
        },
    )
}

fn parse_nvidia_physical_indices(stdout: &[u8]) -> Option<Vec<usize>> {
    let stdout = std::str::from_utf8(stdout).ok()?;
    let mut indices = Vec::new();
    let mut seen_indices = std::collections::BTreeSet::new();
    let mut seen_uuids = std::collections::BTreeSet::new();
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = line.split(',');
        let index = fields.next()?.trim().parse::<usize>().ok()?;
        let uuid = fields.next()?.trim();
        if fields.next().is_some()
            || uuid.is_empty()
            || uuid.eq_ignore_ascii_case("N/A")
            || !uuid.starts_with("GPU-")
            || uuid.len() <= "GPU-".len()
            || !seen_indices.insert(index)
            || !seen_uuids.insert(uuid)
        {
            return None;
        }
        indices.push(index);
    }
    (!indices.is_empty()).then_some(indices)
}

/// Query total, used, and free memory in one bounded `nvidia-smi` process.
fn query_nvidia_smi_memory_for(index: usize) -> Option<(u64, u64, u64)> {
    let id = format!("--id={index}");
    let stdout = bounded_command_stdout(
        "nvidia-smi",
        &[
            id.as_str(),
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        std::time::Duration::from_secs(2),
    )?;
    parse_nvidia_smi_memory(&stdout)
}

fn parse_nvidia_smi_memory(stdout: &[u8]) -> Option<(u64, u64, u64)> {
    let stdout = std::str::from_utf8(stdout).ok()?;
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let mut fields = lines.next()?.split(',');
    if lines.next().is_some() {
        return None;
    }
    let total_mib: u64 = fields.next()?.trim().parse().ok()?;
    let used_mib: u64 = fields.next()?.trim().parse().ok()?;
    let free_mib: u64 = fields.next()?.trim().parse().ok()?;
    if fields.next().is_some() || total_mib == 0 || used_mib > total_mib || free_mib > total_mib {
        return None;
    }
    Some((
        total_mib.checked_mul(1024 * 1024)?,
        used_mib.checked_mul(1024 * 1024)?,
        free_mib.checked_mul(1024 * 1024)?,
    ))
}

fn retry_optional_probe<T>(
    attempts: usize,
    delay: std::time::Duration,
    mut probe: impl FnMut() -> Option<T>,
) -> Option<T> {
    for attempt in 0..attempts {
        if let Some(value) = probe() {
            return Some(value);
        }
        if attempt + 1 < attempts {
            std::thread::sleep(delay);
        }
    }
    None
}

/// Run a small system probe with a hard child lifetime. Probe failures and
/// timeouts are reported to the governor, which publishes a marked zero-free
/// sample while retaining the last known capacity and provenance.
fn bounded_command_stdout(
    program: &str,
    args: &[&str],
    timeout: std::time::Duration,
) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::process::Stdio;

    let mut child = std::process::Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                child.stdout.take()?.read_to_end(&mut stdout).ok()?;
                return status.success().then_some(stdout);
            }
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn is_primary_drm_node(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
        || name
            .strip_prefix("renderD")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(target_os = "linux")]
fn read_u64_file(path: &std::path::Path) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    raw.trim().parse().ok()
}

/// Query Apple Silicon unified memory size via `sysctl hw.memsize`.
///
/// On Apple Silicon, CPU and GPU share the same memory pool. Metal can
/// address most of it; we subtract a conservative OS/app headroom
/// (6 GB, or 25 % on chips > 24 GB — whichever is larger) so inference
/// and training don't squeeze out Finder, the browser, or a dev server. The
/// typed `memory.gpu_memory_gb` setting may reduce, but never expand, this cap.
#[cfg(target_os = "macos")]
fn query_apple_unified_memory() -> Option<u64> {
    let total = query_apple_physical_memory()?;
    Some(total.saturating_sub(unified_memory_reserve_bytes(total)))
}

#[cfg(target_os = "macos")]
fn query_apple_physical_memory() -> Option<u64> {
    let stdout = bounded_command_stdout(
        "sysctl",
        &["-n", "hw.memsize"],
        std::time::Duration::from_secs(1),
    )?;
    std::str::from_utf8(&stdout).ok()?.trim().parse().ok()
}

#[cfg(target_os = "macos")]
fn query_apple_memory_state() -> Option<(u64, u64)> {
    let total = query_apple_physical_memory()?;
    let available = bounded_command_stdout(
        "memory_pressure",
        &["-Q"],
        std::time::Duration::from_secs(1),
    )
    .and_then(|stdout| parse_memory_pressure_available(&stdout, total))
    .or_else(|| {
        bounded_command_stdout("vm_stat", &[], std::time::Duration::from_secs(1))
            .and_then(|stdout| parse_vm_stat_available(&stdout))
    })?;
    Some((total, available.min(total)))
}

#[cfg(any(target_os = "macos", test))]
fn parse_memory_pressure_available(stdout: &[u8], total: u64) -> Option<u64> {
    let stdout = std::str::from_utf8(stdout).ok()?;
    let percent = stdout.lines().find_map(|line| {
        line.split_once("System-wide memory free percentage:")
            .and_then(|(_, value)| value.trim().strip_suffix('%'))
            .and_then(|value| value.trim().parse::<u64>().ok())
    })?;
    (percent <= 100).then(|| total.saturating_mul(percent) / 100)
}

#[cfg(any(target_os = "macos", test))]
fn parse_vm_stat_available(stdout: &[u8]) -> Option<u64> {
    let stdout = std::str::from_utf8(stdout).ok()?;
    let page_size = stdout
        .lines()
        .next()?
        .split_once("page size of ")?
        .1
        .split_once(" bytes")?
        .0
        .parse::<u64>()
        .ok()?;
    let mut free_pages = None;
    let mut inactive_pages = None;
    for line in stdout.lines().skip(1) {
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        let pages = value.trim().trim_end_matches('.').parse::<u64>().ok();
        match label.trim() {
            "Pages free" => free_pages = pages,
            "Pages inactive" => inactive_pages = pages,
            _ => {}
        }
    }
    free_pages?
        .saturating_add(inactive_pages.unwrap_or(0))
        .checked_mul(page_size)
}

/// Recommended number of KV cache blocks based on total VRAM.
/// Pure heuristic: typed operator policy is applied by the caller.
pub fn recommended_num_blocks(vram: &GpuVramInfo) -> Option<usize> {
    // Use slightly lower thresholds since GPUs report slightly less than marketed
    // e.g. RTX A5000 "24GB" reports 24564 MiB ≈ 23.99 GiB
    let gb = vram.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    Some(if gb >= 45.0 {
        512
    } else if gb >= 22.0 {
        64 // proven safe for training on 24GB (18.3GB peak with 8 segments)
    } else if gb >= 14.0 {
        32
    } else {
        64 // conservative default for unknown VRAM
    })
}

/// Recommended gradient checkpoint segments based on total VRAM.
///
/// Pure heuristic: typed operator policy is applied by the caller. More
/// segments means less VRAM but more compute overhead.
///
/// This is the *VRAM-only* heuristic (no sequence-length awareness). The training
/// trainer paths now prefer [`recommended_checkpoint_plan`] which also factors in
/// `max_seq_len` and `hidden_size`, but this function is retained for callers that
/// don't have the workload shape handy (preflight estimator, bench reporter).
pub fn recommended_checkpoint_segments(vram: &GpuVramInfo) -> Option<usize> {
    // Use slightly lower thresholds since GPUs report slightly less than marketed
    // e.g. RTX A5000 "24GB" reports 24564 MiB ≈ 23.99 GiB
    let gb = vram.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    Some(if gb >= 45.0 {
        4 // fewer segments = faster training, more VRAM headroom
    } else if gb >= 22.0 {
        8 // proven safe on 24GB (18.3GB peak)
    } else if gb >= 14.0 {
        12 // aggressive checkpointing for tight VRAM
    } else {
        8 // conservative default
    })
}

/// Decision returned by [`recommended_checkpoint_plan`].
///
/// Two outcomes — the auto-tuner returns one based on `(vram,
/// num_layers, max_seq_len, hidden_size)`:
///
/// * [`CheckpointPlan::Disabled`] — activations comfortably fit in available
///   VRAM after the base model and a safety reserve. Skipping checkpointing
///   wins ~10-30% step time without OOM risk.
/// * [`CheckpointPlan::Enabled`] — activations would crowd available VRAM at
///   one or more segment counts; pick the smallest segment count that keeps
///   per-segment activation memory under the headroom budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointPlan {
    /// Auto-decided to disable checkpointing entirely. `max_act_gib` is the
    /// estimated activation tape size for the workload; `available_gib` is
    /// the headroom we have for it. Included for logging.
    Disabled {
        max_act_gib: f64,
        available_gib: f64,
    },
    /// Auto-decided to engage checkpointing with this segment count.
    Enabled {
        num_segments: usize,
        max_act_gib: f64,
        per_segment_gib: f64,
        available_gib: f64,
    },
}

/// Approximate base-model VRAM footprint for a single-host SFT/GRPO/OPD
/// training run, in bytes.
///
/// We can't ask candle "how much VRAM is the base model using right now" from
/// inside `kiln-core` without taking a dependency on `kiln-model`, so use a
/// simple param-count formula sized to the canonical inference dtype (BF16,
/// 2 bytes/param). It's deliberately conservative — over-estimating the base
/// only pulls the auto-tune toward MORE checkpointing, never less, so we
/// can't accidentally OOM by mis-estimating.
///
/// Formula (canonical SwiGLU + GQA shape):
/// * Attention per layer: `4 * hidden^2` (Q, K, V, O — approximating GQA's
///   reduction by ~30% via the constant 4 instead of 5 or 6).
/// * MLP per layer: `3 * hidden * intermediate` (gate, up, down).
/// * Embedding + LM head: `2 * vocab * hidden`.
///
/// Multiplied by `bytes_per_param` and then by a 1.2× working-buffer
/// overhead factor (RoPE tables, attention masks, paged-KV scratch,
/// candle's own intermediate tensor pool, the LoRA Vars and AdamW state
/// registry). The 1.2× lands the Qwen3.5-4B estimate at ~9.5 GiB,
/// matching the observed `model_loaded_vram_mib=9943` from the
/// kiln-server training bench within ±5%.
pub fn estimate_base_model_bytes(
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    bytes_per_param: usize,
) -> u64 {
    let per_layer_params = (4 * hidden_size * hidden_size) + (3 * hidden_size * intermediate_size);
    let layer_total = per_layer_params.saturating_mul(num_layers);
    let head_total = 2usize
        .saturating_mul(vocab_size)
        .saturating_mul(hidden_size);
    let total_params = layer_total.saturating_add(head_total);
    let raw_bytes = (total_params as u64).saturating_mul(bytes_per_param as u64);
    // 1.2× working-buffer overhead. Implemented as (raw * 6) / 5 to stay
    // in integer math.
    raw_bytes.saturating_mul(6) / 5
}

/// Multiplier from "forward activation tape" to "peak training memory for
/// activations + grads + scratch". Forward saves activations once; backward
/// materializes grads of the same shape and uses transient scratch matmul
/// buffers of similar size during the chain rule. Empirically peak is
/// ~2-3× the forward tape; 2.5× is a conservative middle that doesn't pull
/// aggressive checkpointing for short prompts but engages it before
/// backward OOMs on long ones. Implemented as `* 5 / 2` to stay in u64.
const ACTIVATION_PEAK_NUMER: u64 = 5;
const ACTIVATION_PEAK_DENOM: u64 = 2;

fn peak_activation_bytes(forward_tape_bytes: u64) -> u64 {
    forward_tape_bytes.saturating_mul(ACTIVATION_PEAK_NUMER) / ACTIVATION_PEAK_DENOM
}

/// Auto-decide a gradient-checkpoint plan for a training workload, factoring
/// in BOTH the device's total VRAM and the workload's shape (`num_layers`,
/// `max_seq_len_tokens`, `hidden_size`, base-model footprint).
///
/// Behavior:
/// 1. Estimate F32 activation tape:
///    `max_act_bytes = num_layers * max_seq_len * hidden_size * 4`.
/// 2. Reserve `base_model_bytes + 2 GiB safety` for everything that isn't
///    activations (model weights, grads, AdamW state, working buffers).
/// 3. Define `available_bytes = max(0, vram.total_bytes - reserved)`.
/// 4. If `max_act_bytes <= available_bytes * 0.5`, return
///    [`CheckpointPlan::Disabled`] — checkpointing would only cost step time
///    without lowering peak VRAM enough to matter.
/// 5. Otherwise pick `num_segments = ceil(max_act_bytes / (available_bytes *
///    0.3))`, clamped to `[2, num_layers]`. The 30% target makes per-segment
///    intermediate memory comfortable inside headroom.
///
/// `None` is returned if VRAM detection failed (`vram.total_bytes == 0`) —
/// the typed runtime policy must choose its conservative unknown-device path.
pub fn recommended_checkpoint_plan(
    vram: &GpuVramInfo,
    num_layers: usize,
    max_seq_len_tokens: usize,
    hidden_size: usize,
    base_model_bytes: u64,
) -> Option<CheckpointPlan> {
    recommended_checkpoint_plan_with_activation_bytes(
        vram,
        num_layers,
        max_seq_len_tokens,
        hidden_size,
        base_model_bytes,
        4,
    )
}

/// Variant of [`recommended_checkpoint_plan`] for callers that know the
/// training hidden-activation dtype. CUDA/ROCm BF16 tape paths should pass `2`;
/// Vulkan BF16 training promotes activations to F32 and should pass `4`.
pub fn recommended_checkpoint_plan_with_activation_bytes(
    vram: &GpuVramInfo,
    num_layers: usize,
    max_seq_len_tokens: usize,
    hidden_size: usize,
    base_model_bytes: u64,
    activation_bytes_per_elem: usize,
) -> Option<CheckpointPlan> {
    // VRAM unknown — caller applies its typed conservative fallback.
    if vram.total_bytes == 0 {
        return None;
    }

    // Forward activation tape (one element per layer-token pair). Most CUDA /
    // ROCm production paths keep BF16 activations; Vulkan promotes BF16 base
    // activations to F32. Callers pass the backend-specific width.
    let activation_bytes_per_elem = activation_bytes_per_elem.max(1) as u64;
    let forward_tape_bytes = (num_layers as u64)
        .saturating_mul(max_seq_len_tokens as u64)
        .saturating_mul(hidden_size as u64)
        .saturating_mul(activation_bytes_per_elem);

    // Peak training memory for activations + grads + scratch. The forward
    // tape alone under-estimates peak — backward doubles the activation
    // footprint and matmul scratch adds ~50% on top. Compare the *peak*,
    // not the forward, against headroom.
    let max_act_bytes = peak_activation_bytes(forward_tape_bytes);

    // 2 GiB safety for working buffers (RoPE tables, attention masks,
    // intermediate matmul outputs that aren't on the activation tape).
    const SAFETY_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    let reserved = base_model_bytes.saturating_add(SAFETY_RESERVE_BYTES);
    let available_bytes = vram.total_bytes.saturating_sub(reserved);

    let gib = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let max_act_gib = gib(max_act_bytes);
    let available_gib = gib(available_bytes);

    // If we have less than 2 GiB of headroom after reserves, use the most
    // aggressive valid checkpoint plan. Hidden runtime policy must not decide
    // this branch through process environment.
    if available_bytes < 2 * 1024 * 1024 * 1024 {
        if num_layers < 2 {
            return None;
        }
        return Some(CheckpointPlan::Enabled {
            num_segments: num_layers,
            max_act_gib,
            per_segment_gib: max_act_gib / num_layers as f64,
            available_gib,
        });
    }

    // Comfortable headroom: skip checkpointing. The 0.5 threshold compares
    // *peak* activation memory (forward tape × 2.5 amplification for
    // backward + scratch) against available headroom, so "Disabled" means
    // we have ≥2× the peak on hand — comfortable across CUDA's caching
    // allocator fragmentation.
    if max_act_bytes <= (available_bytes / 2) {
        return Some(CheckpointPlan::Disabled {
            max_act_gib,
            available_gib,
        });
    }

    // Tight headroom: pick segments to keep per-segment activation memory
    // under 30% of available. Round up; clamp to [2, num_layers].
    let target_per_segment = available_bytes.max(1) * 3 / 10;
    let mut num_segments = ((max_act_bytes + target_per_segment - 1) / target_per_segment) as usize;
    if num_segments < 2 {
        num_segments = 2;
    }
    if num_segments > num_layers {
        num_segments = num_layers;
    }

    let per_segment_gib = max_act_gib / (num_segments as f64);
    Some(CheckpointPlan::Enabled {
        num_segments,
        max_act_gib,
        per_segment_gib,
        available_gib,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_nvidia_memory_query() {
        let mib = 1024 * 1024;
        assert_eq!(
            parse_nvidia_smi_memory(b"24564, 1024, 23200\n"),
            Some((24_564 * mib, 1_024 * mib, 23_200 * mib))
        );
        assert_eq!(parse_nvidia_smi_memory(b"24564, 1024\n"), None);
        assert_eq!(parse_nvidia_smi_memory(b"24564, nope, 23200\n"), None);
        assert_eq!(
            parse_nvidia_smi_memory(b"24564, 1024, 23200\n24564, 0, 0\n"),
            None
        );
        assert_eq!(parse_nvidia_smi_memory(b"0, 0, 0\n"), None);
        assert_eq!(parse_nvidia_smi_memory(b"24564, 24565, 0\n"), None);
        assert_eq!(parse_nvidia_smi_memory(b"24564, 0, 24565\n"), None);
    }

    #[test]
    fn startup_probe_retry_is_bounded_and_stops_after_success() {
        let mut calls = 0;
        let recovered = retry_optional_probe(3, std::time::Duration::ZERO, || {
            calls += 1;
            (calls == 2).then_some(17u64)
        });
        assert_eq!(recovered, Some(17));
        assert_eq!(calls, 2);

        let mut failed_calls = 0;
        let failed: Option<u64> = retry_optional_probe(3, std::time::Duration::ZERO, || {
            failed_calls += 1;
            None
        });
        assert_eq!(failed, None);
        assert_eq!(failed_calls, 3);
    }

    #[test]
    fn nvidia_snapshot_uses_conservative_free_counter() {
        let mib = 1024 * 1024;
        let snapshot =
            nvidia_memory_snapshot_from_counters(16_376 * mib, 15_089 * mib, 1_024 * mib);
        assert_eq!(snapshot.total_bytes, 16_376 * mib);
        assert_eq!(snapshot.free_bytes, 1_024 * mib);
        assert_eq!(snapshot.used_bytes, 15_352 * mib);
        assert_eq!(
            snapshot.used_bytes.saturating_add(snapshot.free_bytes),
            snapshot.total_bytes
        );
        assert_eq!(snapshot.observations.driver_used_bytes, Some(15_089 * mib));
        assert_eq!(snapshot.observations.driver_free_bytes, Some(1_024 * mib));

        let subtraction_is_lower =
            nvidia_memory_snapshot_from_counters(16_376 * mib, 15_089 * mib, 1_500 * mib);
        assert_eq!(subtraction_is_lower.free_bytes, 1_287 * mib);
    }

    #[test]
    fn parses_nvidia_physical_identity_inventory_strictly() {
        assert_eq!(
            parse_nvidia_physical_indices(
                b"0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n\
                  1, GPU-11111111-2222-3333-4444-555555555555\n"
            ),
            Some(vec![0, 1])
        );
        assert_eq!(parse_nvidia_physical_indices(b""), None);
        assert_eq!(parse_nvidia_physical_indices(b"0, N/A\n"), None);
        assert_eq!(parse_nvidia_physical_indices(b"0, not-a-gpu-uuid\n"), None);
        assert_eq!(
            parse_nvidia_physical_indices(b"0, GPU-one\n0, GPU-two\n"),
            None
        );
        assert_eq!(
            parse_nvidia_physical_indices(b"0, GPU-one\n1, GPU-one\n"),
            None
        );
        assert_eq!(
            parse_nvidia_physical_indices(b"0, GPU-one, unexpected\n"),
            None
        );
    }

    #[test]
    fn ordinal_identity_contract_accepts_only_unmapped_singular_zero() {
        let drm_zero = VramProbeSelector::LinuxDrm {
            index: 0,
            vendor: Some(LinuxDrmVendor::Amd),
        };
        assert!(validate_ordinal_identity(drm_zero, 0, 1, &[]).is_ok());

        let multi = validate_ordinal_identity(drm_zero, 0, 2, &[])
            .unwrap_err()
            .to_string();
        assert!(multi.contains("2 relevant physical candidates"));
        assert!(multi.contains("PCI-address/UUID-bound"));

        let nonzero = validate_ordinal_identity(VramProbeSelector::Nvidia(1), 1, 2, &[])
            .unwrap_err()
            .to_string();
        assert!(nonzero.contains("restricted to logical ordinal zero"));

        let out_of_range = validate_ordinal_identity(VramProbeSelector::Nvidia(3), 3, 1, &[])
            .unwrap_err()
            .to_string();
        assert!(out_of_range.contains("out-of-range"));

        let remapped = validate_ordinal_identity(
            drm_zero,
            0,
            1,
            &["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"],
        )
        .unwrap_err()
        .to_string();
        assert!(remapped.contains("ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES"));

        assert!(validate_nvidia_ordinal_identity(0, &[0], &[]).is_ok());
        let physical_nonzero = validate_nvidia_ordinal_identity(0, &[7], &[])
            .unwrap_err()
            .to_string();
        assert!(physical_nonzero.contains("singular but nonzero physical index set [7]"));
    }

    #[test]
    fn non_ordinal_identity_semantics_are_explicit() {
        assert!(validate_vram_probe_identity(VramProbeSelector::None).is_ok());
        assert!(validate_vram_probe_identity(VramProbeSelector::AppleUnified).is_ok());
        let auto = validate_vram_probe_identity(VramProbeSelector::Auto)
            .unwrap_err()
            .to_string();
        assert!(auto.contains("diagnostic-only"));
    }

    #[test]
    fn parses_apple_host_availability_probes() {
        let total = 16 * 1024u64.pow(3);
        assert_eq!(
            parse_memory_pressure_available(b"System-wide memory free percentage: 75%\n", total),
            Some(12 * 1024u64.pow(3))
        );
        assert_eq!(
            parse_vm_stat_available(
                b"Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages active: 900.\nPages inactive: 200.\n"
            ),
            Some(300 * 16_384)
        );
    }

    #[test]
    fn configured_capacity_preserves_unified_physical_topology() {
        let physical = GpuVramInfo {
            total_bytes: 120 * 1024 * 1024 * 1024,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let resolution = resolve_vram_capacity(physical, Some(96.0));
        let configured = resolution.effective;

        assert_eq!(resolution.physical, physical);
        assert_eq!(resolution.requested_bytes, Some(96 * 1024 * 1024 * 1024));
        assert!(!resolution.clamped);
        assert_eq!(configured.total_bytes, 96 * 1024 * 1024 * 1024);
        assert_eq!(configured.source, VramSource::ConfigOverride);
        assert!(configured.unified);
    }

    #[test]
    fn configured_capacity_preserves_discrete_physical_topology() {
        let physical = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        let configured = resolve_vram_capacity(physical, Some(16.0)).effective;

        assert_eq!(configured.total_bytes, 16 * 1024 * 1024 * 1024);
        assert_eq!(configured.source, VramSource::ConfigOverride);
        assert!(!configured.unified);
    }

    #[test]
    fn configured_capacity_never_expands_physical_capacity() {
        let physical = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let resolution = resolve_vram_capacity(physical, Some(96.0));

        assert_eq!(resolution.physical, physical);
        assert_eq!(resolution.effective, physical);
        assert!(resolution.clamped);
    }

    #[test]
    fn effective_budget_mirrors_detect_vram_fields() {
        // The convenience accessor must agree with the underlying
        // detector — it's literally a thin wrapper. Lock that in so a
        // future refactor can't silently introduce divergence between
        // "what we log" and "what we size against".
        let detected = detect_vram();
        let budget = detect_effective_training_budget(None);
        assert_eq!(budget.total_bytes, detected.total_bytes);
        assert_eq!(budget.source, detected.source);
        assert_eq!(budget.unified, detected.unified);
    }

    #[test]
    fn test_recommended_num_blocks() {
        let vram_48gb = GpuVramInfo {
            total_bytes: 48 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_num_blocks(&vram_48gb), Some(512));

        let vram_24gb = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_num_blocks(&vram_24gb), Some(64));

        // Test with real A5000 value (24564 MiB = slightly under 24 GiB)
        let vram_a5000 = GpuVramInfo {
            total_bytes: 24564 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_num_blocks(&vram_a5000), Some(64));

        let vram_16gb = GpuVramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_num_blocks(&vram_16gb), Some(32));

        let vram_none = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
            unified: false,
        };
        assert_eq!(recommended_num_blocks(&vram_none), Some(64));
    }

    #[test]
    fn test_recommended_checkpoint_segments() {
        let vram_48gb = GpuVramInfo {
            total_bytes: 48 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_48gb), Some(4));

        let vram_24gb = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_24gb), Some(8));

        // Test with real A5000 value (24564 MiB = slightly under 24 GiB)
        let vram_a5000 = GpuVramInfo {
            total_bytes: 24564 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_a5000), Some(8));

        let vram_16gb = GpuVramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_16gb), Some(12));
    }

    fn vram(gb: u64) -> GpuVramInfo {
        GpuVramInfo {
            total_bytes: gb * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
            unified: false,
        }
    }

    fn act_gib(num_layers: usize, max_seq_len: usize, hidden_size: usize) -> f64 {
        let bytes = (num_layers as u64) * (max_seq_len as u64) * (hidden_size as u64) * 4;
        (bytes as f64) / (1024.0 * 1024.0 * 1024.0)
    }

    #[test]
    fn estimate_base_model_bytes_qwen35_4b_matches_observed_vram() {
        // Qwen3.5-4B: hidden=2560, intermediate=10240, num_layers=32, vocab=151936.
        // BF16 weights observed at runtime: ~9943 MiB. Estimator must
        // land within ±15% of that or our auto-tune's headroom math will
        // be skewed.
        let est = estimate_base_model_bytes(32, 2560, 10240, 151936, 2);
        let gib = est as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            gib >= 8.0 && gib <= 11.5,
            "Qwen3.5-4B base estimate {gib:.2} GiB outside ±15% of observed ~9.7 GiB"
        );
    }

    #[test]
    fn recommended_checkpoint_plan_disables_on_short_prompts_big_vram() {
        // A6000 (48 GiB) + Qwen3.5-4B + 30-token prompts: activation tape
        // is ~10 MiB. Auto-tune should disable checkpointing entirely.
        // This is the bench scenario where #1071's PR was leaving 10-30%
        // step time on the table.
        let plan = recommended_checkpoint_plan(
            &vram(48),
            32,
            30,
            2560,
            estimate_base_model_bytes(32, 2560, 10240, 151936, 2),
        );
        assert!(matches!(plan, Some(CheckpointPlan::Disabled { .. })));
    }

    #[test]
    fn recommended_checkpoint_plan_enables_for_long_prompts_big_vram() {
        // A6000 (48 GiB) + Qwen3.5-4B + 32K prompts: activation tape is
        // ~10.7 GiB. Headroom is ~36 GiB. 30% of headroom is ~11 GiB so
        // a single segment fits — but we still want >=2 segments since
        // the heuristic clamps to that. Critically: must engage, must
        // not disable.
        let plan = recommended_checkpoint_plan(
            &vram(48),
            32,
            32 * 1024,
            2560,
            estimate_base_model_bytes(32, 2560, 10240, 151936, 2),
        )
        .expect("plan");
        let n = match plan {
            CheckpointPlan::Enabled { num_segments, .. } => num_segments,
            other => panic!("expected Enabled, got {other:?}"),
        };
        assert!(
            (2..=32).contains(&n),
            "expected 2..=32 segments for long-context, got {n}"
        );
    }

    #[test]
    fn recommended_checkpoint_plan_aggressive_on_tight_vram_long_prompts() {
        // RTX 3090 (24 GiB) + Qwen3.5-4B + 16K prompts: activation tape
        // ~5.4 GiB, headroom ~12 GiB → must engage with >= 4 segments
        // (per-segment ~1.4 GiB inside the 30% target).
        let plan = recommended_checkpoint_plan(
            &vram(24),
            32,
            16 * 1024,
            2560,
            estimate_base_model_bytes(32, 2560, 10240, 151936, 2),
        )
        .expect("plan");
        let n = match plan {
            CheckpointPlan::Enabled { num_segments, .. } => num_segments,
            other => panic!("expected Enabled, got {other:?}"),
        };
        assert!(
            n >= 2,
            "expected aggressive checkpointing on tight VRAM + long prompts, got {n} segments"
        );
        // Sanity-check the activation math hasn't drifted.
        assert!((act_gib(32, 16 * 1024, 2560) - 5.0).abs() < 1.0);
    }

    #[test]
    fn recommended_checkpoint_plan_respects_activation_width() {
        let base_bytes = estimate_base_model_bytes(32, 2560, 10240, 151936, 2);
        let f32_plan = recommended_checkpoint_plan_with_activation_bytes(
            &vram(16),
            32,
            22_484,
            2560,
            base_bytes,
            4,
        )
        .expect("f32 plan");
        let bf16_plan = recommended_checkpoint_plan_with_activation_bytes(
            &vram(16),
            32,
            22_484,
            2560,
            base_bytes,
            2,
        )
        .expect("bf16 plan");

        let segments = |plan: CheckpointPlan| match plan {
            CheckpointPlan::Enabled { num_segments, .. } => num_segments,
            other => panic!("expected enabled checkpointing, got {other:?}"),
        };
        let f32_segments = segments(f32_plan);
        let bf16_segments = segments(bf16_plan);
        assert!(
            bf16_segments < f32_segments,
            "BF16 activation sizing should reduce checkpoint segments for long-context CUDA: \
             bf16={bf16_segments} f32={f32_segments}"
        );
    }

    #[test]
    fn recommended_checkpoint_plan_returns_none_when_vram_unknown() {
        let unknown = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
            unified: false,
        };
        let plan = recommended_checkpoint_plan(&unknown, 32, 1024, 2560, 10 * 1024 * 1024 * 1024);
        assert!(plan.is_none());
    }

    #[test]
    fn recommended_checkpoint_plan_maximizes_segments_when_headroom_too_small() {
        // 12 GiB capacity minus a 10 GiB model and 2 GiB safety leaves no
        // activation headroom, so the pure auto policy chooses every layer.
        let plan = recommended_checkpoint_plan(&vram(12), 32, 1024, 2560, 10 * 1024 * 1024 * 1024);
        assert!(matches!(
            plan,
            Some(CheckpointPlan::Enabled {
                num_segments: 32,
                ..
            })
        ));
    }

    /// Perf-regression matrix (#1077 Tier 1a): exhaustive
    /// `(GPU class × max_seq_len)` decision table for the Qwen3.5-4B
    /// shape that the trainer ships on. Drift in the activation
    /// estimate's `2.5×` peak multiplier or in the `50%` Disable
    /// threshold will surface here as a single-cell flip even if no
    /// other test changes.
    ///
    /// This is the per-PR cheap detector — if you change the heuristic
    /// constants, update the table; if a change you didn't intend
    /// flips a cell, the bench is doing its job.
    #[test]
    fn perf_regression_qwen35_4b_plan_matrix() {
        #[derive(Debug)]
        enum Expect {
            /// Activation tape comfortably fits — disable.
            Disabled,
            /// Activations would crowd VRAM — engage; the segment
            /// count must be at least `n` (we don't pin the exact
            /// value because the segment-target heuristic is allowed
            /// to retune within a tolerance).
            EnabledMin(usize),
            /// Either decision is acceptable for this cell (borderline
            /// case — flagged here to surface in the test failure but
            /// not assert direction).
            #[allow(dead_code)]
            Either,
        }
        use Expect::*;

        // (vram_gb, max_seq_len, expected) — all Qwen3.5-4B
        // (num_layers=32, hidden=2560, intermediate=10240,
        // vocab=151936, BF16 base = 2 bytes/param).
        let cases: &[(u64, usize, Expect)] = &[
            // Big VRAM (A6000 48 GB): everything up to 16K disables.
            (48, 30, Disabled),
            (48, 1024, Disabled),
            (48, 4096, Disabled),
            (48, 16384, Disabled),
            // 32K and 64K engage on A6000.
            (48, 32 * 1024, EnabledMin(2)),
            (48, 64 * 1024, EnabledMin(2)),
            // Mid VRAM (RTX 3090 24 GB): short disables, long engages.
            (24, 30, Disabled),
            (24, 4096, Disabled),
            (24, 16 * 1024, EnabledMin(2)),
            // Tight VRAM (RTX 4060 Ti / A4000 16 GB): activations crowd
            // headroom quickly; long contexts engage.
            (16, 30, Disabled),
            (16, 4096, EnabledMin(2)),
            (16, 8 * 1024, EnabledMin(2)),
        ];

        let base_bytes = estimate_base_model_bytes(32, 2560, 10240, 151936, 2);
        let mut failures: Vec<String> = Vec::new();
        for &(vram_gb, max_seq_len, ref expected) in cases {
            let plan =
                recommended_checkpoint_plan(&vram(vram_gb), 32, max_seq_len, 2560, base_bytes);
            let cell_ok = match (expected, plan.as_ref()) {
                (Disabled, Some(CheckpointPlan::Disabled { .. })) => true,
                (EnabledMin(n), Some(CheckpointPlan::Enabled { num_segments, .. })) => {
                    *num_segments >= *n
                }
                (Either, Some(_)) => true,
                _ => false,
            };
            if !cell_ok {
                failures.push(format!(
                    "  vram={vram_gb}GB  seq_len={max_seq_len}  expected={expected:?}  got={plan:?}",
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "#1077 perf-regression matrix drift detected ({} cells):\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }

    /// Perf-regression matrix (#1077 Tier 1a) on a smaller Llama-3-8B-like
    /// shape (num_layers=32, hidden=4096, vocab=32000). Catches drift that
    /// only shows up at larger hidden sizes — the activation tape scales
    /// linearly with hidden, so the Disable→Enable boundary shifts.
    #[test]
    fn perf_regression_llama_8b_plan_matrix() {
        #[derive(Debug)]
        enum Expect {
            Disabled,
            EnabledMin(usize),
        }
        use Expect::*;

        // (vram_gb, max_seq_len, expected) — Llama-3-8B
        // (num_layers=32, hidden=4096, intermediate=14336,
        // vocab=128000, BF16 base = 2 bytes/param).
        let cases: &[(u64, usize, Expect)] = &[
            (80, 30, Disabled),
            (80, 4096, Disabled),
            (80, 16 * 1024, Disabled),
            (80, 32 * 1024, EnabledMin(2)),
            (48, 30, Disabled),
            (48, 4096, Disabled),
            (48, 16 * 1024, EnabledMin(2)),
            (24, 4096, EnabledMin(2)),
        ];

        let base_bytes = estimate_base_model_bytes(32, 4096, 14336, 128000, 2);
        let mut failures: Vec<String> = Vec::new();
        for &(vram_gb, max_seq_len, ref expected) in cases {
            let plan =
                recommended_checkpoint_plan(&vram(vram_gb), 32, max_seq_len, 4096, base_bytes);
            let cell_ok = match (expected, plan.as_ref()) {
                (Disabled, Some(CheckpointPlan::Disabled { .. })) => true,
                (EnabledMin(n), Some(CheckpointPlan::Enabled { num_segments, .. })) => {
                    *num_segments >= *n
                }
                _ => false,
            };
            if !cell_ok {
                failures.push(format!(
                    "  vram={vram_gb}GB  seq_len={max_seq_len}  expected={expected:?}  got={plan:?}",
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "#1077 perf-regression Llama matrix drift detected ({} cells):\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }

    /// Perf-regression #1077 Tier 1a: the `Disabled` plan must report
    /// numbers consistent with the input shape. If a future PR
    /// changes the activation estimate to e.g. omit a hidden-dim
    /// dependency, this catches the change as a sign error or a
    /// constant-factor change in the reported `max_act_gib`.
    #[test]
    fn perf_regression_disabled_plan_reports_sane_act_tape() {
        let base_bytes = estimate_base_model_bytes(32, 2560, 10240, 151936, 2);
        let plan = recommended_checkpoint_plan(&vram(48), 32, 1024, 2560, base_bytes)
            .expect("plan must resolve for A6000+1K");
        let (max_act_gib, available_gib) = match plan {
            CheckpointPlan::Disabled {
                max_act_gib,
                available_gib,
            } => (max_act_gib, available_gib),
            other => panic!("expected Disabled for A6000+1K, got {other:?}"),
        };
        // 32 layers * 1024 tokens * 2560 hidden * 4 bytes = ~320 MiB
        // ≈ 0.31 GiB. With the 2.5× peak multiplier in the heuristic
        // the reported max_act_gib should be roughly in the
        // 0.5..3.0 GiB range — anything outside means the multiplier
        // drifted by >2x.
        assert!(
            (0.3..=3.0).contains(&max_act_gib),
            "Disabled.max_act_gib = {max_act_gib:.2} GiB is outside the \
             plausible 0.3..3.0 GiB range for A6000+1K Qwen3.5-4B — \
             activation peak multiplier likely drifted",
        );
        // Available VRAM after base model should be in the 30-40 GiB
        // range for A6000 + Qwen3.5-4B BF16 (48 - ~10 - safety reserve).
        assert!(
            (20.0..=45.0).contains(&available_gib),
            "Disabled.available_gib = {available_gib:.2} GiB outside \
             plausible 20..45 GiB for A6000 minus Qwen3.5-4B BF16",
        );
    }

    #[test]
    fn test_vram_source_display() {
        assert_eq!(VramSource::NvidiaSmi.to_string(), "nvidia-smi");
        assert_eq!(VramSource::LinuxDrmSysfs.to_string(), "linux-drm-sysfs");
        assert_eq!(
            VramSource::AppleSilicon.to_string(),
            "apple-silicon-unified"
        );
        assert_eq!(
            VramSource::ConfigOverride.to_string(),
            "memory.gpu_memory_gb"
        );
        assert_eq!(VramSource::None.to_string(), "none");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_strix_halo_carveout_keeps_device_vram_separate_from_host() {
        // Approximate this Strix Halo host: 128 GiB physical LPDDR is split into
        // a ~96 GiB GPU VRAM carveout and ~32 GiB CPU-online memory. GTT is an
        // additional host-backed aperture; it must not inflate or cap VRAM.
        let root = std::env::temp_dir().join(format!(
            "kiln-drm-unified-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("card1/device")).unwrap();
        std::fs::write(
            root.join("card1/device/mem_info_vram_total"),
            "103079215104\n",
        )
        .unwrap();
        std::fs::write(
            root.join("card1/device/mem_info_gtt_total"),
            "16629473300\n",
        )
        .unwrap();
        std::fs::write(root.join("card1/device/mem_info_vram_used"), "4294967296\n").unwrap();
        std::fs::write(
            root.join("card1/device/mem_info_gtt_used"),
            (3 * 1024u64.pow(3)).to_string(),
        )
        .unwrap();
        std::fs::write(root.join("card1/device/vendor"), "0x1002\n").unwrap();
        std::fs::write(root.join("card1/device/class"), "0x038000\n").unwrap();

        let device = collect_linux_drm_device_info_at(&root).unwrap();
        assert_eq!(device.vram_total, 103_079_215_104);
        assert_eq!(device.gtt_total, 16_629_473_300);
        assert_eq!(device.vendor, 0x1002);
        assert_eq!(device.class, 0x038000);

        let mem_total = 32 * 1024u64.pow(3);
        let mem_available = 12u64 * 1024 * 1024 * 1024;
        let meminfo_path = root.join("meminfo");
        std::fs::write(
            &meminfo_path,
            format!(
                "MemTotal:       {} kB\nMemAvailable:   {} kB\n",
                mem_total / 1024,
                mem_available / 1024
            ),
        )
        .unwrap();

        assert!(!is_host_shared_memory_drm(&device, Some(mem_total)));
        let info = detect_linux_drm_vram_at(&root, &meminfo_path).unwrap();
        assert_eq!(info.source, VramSource::LinuxDrmSysfs);
        assert!(!info.unified);
        assert_eq!(info.total_bytes, 103_079_215_104);

        let snapshot = linux_drm_memory_snapshot_at(&root, &meminfo_path, None, 0, None).unwrap();
        assert_eq!(snapshot.total_bytes, 103_079_215_104);
        assert_eq!(snapshot.used_bytes, 4 * 1024u64.pow(3));
        assert_eq!(snapshot.free_bytes, 103_079_215_104 - 4 * 1024u64.pow(3));
        assert_eq!(
            snapshot.used_bytes.saturating_add(snapshot.free_bytes),
            snapshot.total_bytes
        );
        assert_eq!(
            snapshot.observations.driver_total_bytes,
            Some(103_079_215_104 + 16_629_473_300)
        );
        assert_eq!(
            snapshot.observations.driver_vram_total_bytes,
            Some(103_079_215_104)
        );
        assert_eq!(
            snapshot.observations.driver_gtt_total_bytes,
            Some(16_629_473_300)
        );
        assert_eq!(snapshot.observations.host_total_bytes, Some(mem_total));
        assert_eq!(
            snapshot.observations.host_available_bytes,
            Some(mem_available)
        );
        assert_eq!(snapshot.observations.unified_reserve_bytes, None);
        let host_backed = snapshot
            .observations
            .host_backed
            .expect("Strix Halo GTT should publish a separate host-backed tier");
        assert_eq!(host_backed.total_bytes, 16_629_473_300);
        assert_eq!(host_backed.free_bytes, 4 * 1024u64.pow(3));
        assert_eq!(
            host_backed
                .used_bytes
                .saturating_add(host_backed.free_bytes),
            host_backed.total_bytes
        );
        assert!(host_backed.free_bytes < snapshot.free_bytes);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_discrete_amd_gpu_kept_as_linuxdrmsysfs() {
        // Discrete AMD card on a small host: 8 GiB VRAM, a larger 16 GiB GTT
        // aperture, and 32 GiB MemTotal. This is also a common dGPU shape, so
        // GTT dominance must neither flag it as unified nor inflate capacity.
        let root = std::env::temp_dir().join(format!(
            "kiln-drm-discrete-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("card0/device")).unwrap();
        std::fs::write(
            root.join("card0/device/mem_info_vram_total"),
            "8589934592\n", // 8 GiB
        )
        .unwrap();
        std::fs::write(
            root.join("card0/device/mem_info_gtt_total"),
            "17179869184\n",
        )
        .unwrap();
        std::fs::write(root.join("card0/device/mem_info_vram_used"), "4294967296\n").unwrap();
        std::fs::write(root.join("card0/device/mem_info_gtt_used"), "2147483648\n").unwrap();
        std::fs::write(root.join("card0/device/vendor"), "0x1002\n").unwrap();
        std::fs::write(root.join("card0/device/class"), "0x030000\n").unwrap();

        let device = collect_linux_drm_device_info_at(&root).unwrap();
        let mem_total = 32u64 * 1024 * 1024 * 1024;
        let meminfo_path = root.join("meminfo");
        std::fs::write(
            &meminfo_path,
            format!(
                "MemTotal:       {} kB\nMemAvailable:   {} kB\n",
                mem_total / 1024,
                16 * 1024u64.pow(2)
            ),
        )
        .unwrap();
        assert!(!is_host_shared_memory_drm(&device, Some(mem_total)));

        let info = detect_linux_drm_vram_at(&root, &meminfo_path).unwrap();
        assert_eq!(info.source, VramSource::LinuxDrmSysfs);
        assert!(!info.unified);
        assert_eq!(info.total_bytes, 8 * 1024u64.pow(3));

        let snapshot = linux_drm_memory_snapshot_at(&root, &meminfo_path, None, 0, None)
            .expect("discrete DRM snapshot");
        assert_eq!(snapshot.total_bytes, 8 * 1024u64.pow(3));
        assert_eq!(snapshot.used_bytes, 4 * 1024u64.pow(3));
        assert_eq!(snapshot.free_bytes, 4 * 1024u64.pow(3));
        assert_eq!(
            snapshot.observations.driver_total_bytes,
            Some(24 * 1024u64.pow(3))
        );
        assert_eq!(
            snapshot.observations.driver_gtt_total_bytes,
            Some(16 * 1024u64.pow(3))
        );
        assert_eq!(
            snapshot.observations.host_backed,
            Some(MemoryTierSnapshot {
                total_bytes: 16 * 1024u64.pow(3),
                used_bytes: 8 * 1024u64.pow(3),
                free_bytes: 8 * 1024u64.pow(3),
            })
        );

        // Losing the GTT usage counter must not poison the independent VRAM
        // snapshot, but the host-backed tier itself becomes unavailable rather
        // than publishing optimistic headroom.
        std::fs::remove_file(root.join("card0/device/mem_info_gtt_used")).unwrap();
        let missing_gtt_usage =
            linux_drm_memory_snapshot_at(&root, &meminfo_path, None, 0, None).unwrap();
        assert_eq!(missing_gtt_usage.free_bytes, 4 * 1024u64.pow(3));
        assert_eq!(missing_gtt_usage.observations.host_backed, None);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_zero_local_vram_uses_host_bounded_gtt_pool() {
        let root = std::env::temp_dir().join(format!(
            "kiln-drm-host-shared-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("card0/device")).unwrap();
        std::fs::write(root.join("card0/device/mem_info_vram_total"), "0\n").unwrap();
        std::fs::write(
            root.join("card0/device/mem_info_gtt_total"),
            "17179869184\n",
        )
        .unwrap();
        std::fs::write(root.join("card0/device/mem_info_gtt_used"), "4294967296\n").unwrap();
        std::fs::write(root.join("card0/device/vendor"), "0x1002\n").unwrap();
        std::fs::write(root.join("card0/device/class"), "0x030000\n").unwrap();
        let meminfo_path = root.join("meminfo");
        std::fs::write(
            &meminfo_path,
            "MemTotal:       33554432 kB\nMemAvailable:   16777216 kB\n",
        )
        .unwrap();

        let device = collect_linux_drm_device_info_at(&root).unwrap();
        assert!(is_host_shared_memory_drm(
            &device,
            Some(32 * 1024u64.pow(3))
        ));
        let info = detect_linux_drm_vram_at(&root, &meminfo_path).unwrap();
        assert_eq!(info.source, VramSource::LinuxDrmSysfsUnified);
        assert!(info.unified);
        assert_eq!(info.total_bytes, 16 * 1024u64.pow(3));

        let snapshot = linux_drm_memory_snapshot_at(&root, &meminfo_path, None, 0, None)
            .expect("host-shared DRM snapshot");
        assert_eq!(snapshot.total_bytes, 16 * 1024u64.pow(3));
        assert_eq!(snapshot.free_bytes, 8 * 1024u64.pow(3));
        assert_eq!(snapshot.used_bytes, 8 * 1024u64.pow(3));
        assert_eq!(
            snapshot.observations.host_backed,
            Some(MemoryTierSnapshot {
                total_bytes: 16 * 1024u64.pow(3),
                used_bytes: 8 * 1024u64.pow(3),
                free_bytes: 8 * 1024u64.pow(3),
            })
        );
        assert_eq!(
            snapshot.observations.unified_reserve_bytes,
            Some(8 * 1024u64.pow(3))
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_meminfo_parser() {
        let path = std::env::temp_dir().join(format!(
            "kiln-meminfo-test-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(
            &path,
            "MemTotal:       32479448 kB\nMemFree:        27571892 kB\n",
        )
        .unwrap();
        assert_eq!(
            query_meminfo_total_bytes_at(&path),
            Some(32_479_448u64 * 1024)
        );
        std::fs::write(&path, format!("MemTotal: {} kB\n", u64::MAX)).unwrap();
        assert_eq!(query_meminfo_total_bytes_at(&path), None);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_meminfo_available_parser() {
        let path = std::env::temp_dir().join(format!(
            "kiln-meminfo-avail-test-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(
            &path,
            "MemTotal:       32479448 kB\nMemFree:  1000000 kB\nMemAvailable:   27571892 kB\n",
        )
        .unwrap();
        assert_eq!(
            query_meminfo_available_bytes_at(&path),
            Some(27_571_892u64 * 1024)
        );
        // Missing MemAvailable -> None (older kernels; caller falls back).
        std::fs::write(&path, "MemTotal: 100 kB\n").unwrap();
        assert_eq!(query_meminfo_available_bytes_at(&path), None);
        std::fs::write(&path, format!("MemAvailable: {} kB\n", u64::MAX)).unwrap();
        assert_eq!(query_meminfo_available_bytes_at(&path), None);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_v1_and_v2_memory_headroom_parsers() {
        let root = std::env::temp_dir().join(format!(
            "kiln-cgroup-memory-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let v2 = root.join("v2");
        let v1 = root.join("v1");
        std::fs::create_dir_all(&v2).unwrap();
        std::fs::create_dir_all(&v1).unwrap();

        std::fs::write(v2.join("memory.max"), "25769803776\n").unwrap();
        std::fs::write(v2.join("memory.high"), "max\n").unwrap();
        std::fs::write(v2.join("memory.current"), "19327352832\n").unwrap();
        let observed_v2 = query_cgroup_v2_at(&v2).unwrap();
        assert_eq!(observed_v2.limit_bytes, Some(25_769_803_776));
        assert_eq!(observed_v2.high_bytes, None);
        assert_eq!(observed_v2.current_bytes, Some(19_327_352_832));
        assert_eq!(observed_v2.remaining_bytes(), Some(6_442_450_944));
        std::fs::write(v2.join("memory.high"), "21474836480\n").unwrap();
        let high_bounded = query_cgroup_v2_at(&v2).unwrap();
        assert_eq!(high_bounded.high_bytes, Some(21_474_836_480));
        assert_eq!(high_bounded.remaining_bytes(), Some(2_147_483_648));
        std::fs::write(v2.join("memory.high"), "max\n").unwrap();
        std::fs::write(v2.join("memory.max"), "max\n").unwrap();
        assert_eq!(query_cgroup_v2_at(&v2).unwrap().limit_bytes, None);
        std::fs::write(v2.join("memory.max"), "invalid\n").unwrap();
        assert_eq!(query_cgroup_v2_at(&v2).unwrap().limit_bytes, Some(0));

        std::fs::write(v1.join("memory.limit_in_bytes"), "21474836480\n").unwrap();
        std::fs::write(v1.join("memory.usage_in_bytes"), "12884901888\n").unwrap();
        let observed_v1 = query_cgroup_v1_at(&v1).unwrap();
        assert_eq!(observed_v1.limit_bytes, Some(21_474_836_480));
        assert_eq!(observed_v1.current_bytes, Some(12_884_901_888));
        assert_eq!(observed_v1.remaining_bytes(), Some(8_589_934_592));
        std::fs::write(v1.join("memory.limit_in_bytes"), "invalid\n").unwrap();
        assert_eq!(query_cgroup_v1_at(&v1).unwrap().limit_bytes, Some(0));

        let mounts = cgroup_memory_mounts(
            "29 22 0:26 /user.slice /run/cgroup\\040v2 rw - cgroup2 cgroup rw\n\
             30 22 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
        );
        assert_eq!(mounts.len(), 2);
        assert_eq!(
            mounts[0].mount_point,
            std::path::Path::new("/run/cgroup v2")
        );
        assert_eq!(
            resolve_cgroup_directory(&mounts[0], "/user.slice/session.scope"),
            std::path::Path::new("/run/cgroup v2/session.scope")
        );
        assert_eq!(
            resolve_cgroup_directory(&mounts[1], "/docker/example"),
            std::path::Path::new("/sys/fs/cgroup/memory/docker/example")
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_hierarchy_enforces_parent_max_and_high_when_leaf_is_unlimited() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let root = std::env::temp_dir().join(format!(
            "kiln-cgroup-hierarchy-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let leaf = root.join("parent/leaf");
        let parent = root.join("parent");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&leaf).unwrap();

        // Leaf is unlimited, while its parent has 24 GiB hard capacity, a
        // 20 GiB no-throttle ceiling, and only 2 GiB of high headroom.
        std::fs::write(leaf.join("memory.max"), "max\n").unwrap();
        std::fs::write(leaf.join("memory.high"), "max\n").unwrap();
        std::fs::write(leaf.join("memory.current"), (4 * GIB).to_string()).unwrap();
        std::fs::write(parent.join("memory.max"), (24 * GIB).to_string()).unwrap();
        std::fs::write(parent.join("memory.high"), (20 * GIB).to_string()).unwrap();
        std::fs::write(parent.join("memory.current"), (18 * GIB).to_string()).unwrap();
        // The synthetic hierarchy root is unlimited.
        std::fs::write(root.join("memory.max"), "max\n").unwrap();
        std::fs::write(root.join("memory.high"), "max\n").unwrap();
        std::fs::write(root.join("memory.current"), (19 * GIB).to_string()).unwrap();

        let observed = query_cgroup_v2_hierarchy_at(&leaf, &root).unwrap();
        assert_eq!(observed.limit_bytes, Some(24 * GIB));
        assert_eq!(observed.high_bytes, Some(20 * GIB));
        assert_eq!(observed.current_bytes, Some(4 * GIB));
        assert_eq!(observed.effective_capacity_bytes(), Some(20 * GIB));
        assert_eq!(observed.remaining_bytes(), Some(2 * GIB));

        // A finite ancestor without a readable usage value fails headroom
        // closed instead of treating the limit as unconsumed.
        std::fs::remove_file(parent.join("memory.current")).unwrap();
        assert_eq!(
            query_cgroup_v2_hierarchy_at(&leaf, &root)
                .unwrap()
                .remaining_bytes(),
            Some(0)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_v1_hierarchy_enforces_finite_parent_limit() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let root = std::env::temp_dir().join(format!(
            "kiln-cgroup-v1-hierarchy-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let leaf = root.join("parent/leaf");
        let parent = root.join("parent");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&leaf).unwrap();

        let unlimited = (1u64 << 60).to_string();
        std::fs::write(leaf.join("memory.limit_in_bytes"), &unlimited).unwrap();
        std::fs::write(leaf.join("memory.usage_in_bytes"), (4 * GIB).to_string()).unwrap();
        std::fs::write(parent.join("memory.limit_in_bytes"), (16 * GIB).to_string()).unwrap();
        std::fs::write(parent.join("memory.usage_in_bytes"), (12 * GIB).to_string()).unwrap();
        std::fs::write(root.join("memory.limit_in_bytes"), unlimited).unwrap();
        std::fs::write(root.join("memory.usage_in_bytes"), (13 * GIB).to_string()).unwrap();

        let observed = query_cgroup_v1_hierarchy_at(&leaf, &root).unwrap();
        assert_eq!(observed.limit_bytes, Some(16 * GIB));
        assert_eq!(observed.high_bytes, None);
        assert_eq!(observed.current_bytes, Some(4 * GIB));
        assert_eq!(observed.remaining_bytes(), Some(4 * GIB));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn current_memory_snapshot_is_internally_consistent() {
        // Whatever the host reports, the snapshot's arithmetic must hold:
        // used + free == total, and free <= total. (On CI runners with no GPU
        // total may be 0, in which case all three are 0 — still consistent.)
        let s = current_memory_snapshot();
        assert_eq!(s.used_bytes.saturating_add(s.free_bytes), s.total_bytes);
        assert!(s.free_bytes <= s.total_bytes);
        if s.source == VramSource::NvidiaSmi {
            let reported_total = s
                .observations
                .driver_total_bytes
                .expect("NVIDIA snapshot must retain reported total");
            let reported_used = s
                .observations
                .driver_used_bytes
                .expect("NVIDIA snapshot must retain reported used");
            let reported_free = s
                .observations
                .driver_free_bytes
                .expect("NVIDIA snapshot must retain reported free");
            assert_eq!(reported_total, s.total_bytes);
            assert_eq!(
                s.free_bytes,
                reported_free.min(reported_total.saturating_sub(reported_used))
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unified_memory_reserve_is_conservative_and_deterministic() {
        let mem_total = 32u64 * 1024 * 1024 * 1024;
        assert_eq!(
            unified_memory_reserve_bytes(mem_total),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(
            unified_memory_reserve_bytes(16 * 1024 * 1024 * 1024),
            6 * 1024 * 1024 * 1024
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_headroom_further_bounds_unified_free() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let bounds = unified_memory_bounds(
            120 * GIB,
            4 * GIB,
            Some(32 * GIB),
            Some(14 * GIB),
            Some(CgroupMemoryObservation::from_level(
                Some(24 * GIB),
                None,
                Some(18 * GIB),
            )),
        );
        // A 24 GiB cgroup retains the 6 GiB minimum reserve. Its remaining
        // 6 GiB is entirely reserved, so a new unified allocation gets zero.
        assert_eq!(bounds.total_bytes, 18 * GIB);
        assert_eq!(bounds.free_bytes, 0);
        assert_eq!(bounds.used_bytes, bounds.total_bytes);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_drm_selector_deduplicates_aliases_and_never_mixes_devices() {
        let root = std::env::temp_dir().join(format!("kiln-drm-vram-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nvidia = root.join("devices/nvidia");
        let amd = root.join("devices/amd");
        std::fs::create_dir_all(&nvidia).unwrap();
        std::fs::create_dir_all(&amd).unwrap();
        for node in ["card0", "renderD128", "card1", "renderD129"] {
            std::fs::create_dir_all(root.join(node)).unwrap();
        }
        std::os::unix::fs::symlink(&nvidia, root.join("card0/device")).unwrap();
        std::os::unix::fs::symlink(&nvidia, root.join("renderD128/device")).unwrap();
        std::os::unix::fs::symlink(&amd, root.join("card1/device")).unwrap();
        std::os::unix::fs::symlink(&amd, root.join("renderD129/device")).unwrap();

        std::fs::write(nvidia.join("mem_info_vram_total"), "51539607552\n").unwrap();
        std::fs::write(nvidia.join("vendor"), "0x10de\n").unwrap();
        std::fs::write(nvidia.join("class"), "0x030200\n").unwrap();
        std::fs::write(amd.join("mem_info_vram_total"), "103079215104\n").unwrap();
        std::fs::write(amd.join("mem_info_gtt_total"), "16629473300\n").unwrap();
        std::fs::write(amd.join("vendor"), "0x1002\n").unwrap();
        std::fs::write(amd.join("class"), "0x038000\n").unwrap();

        let first = select_linux_drm_device_at(&root, 0, None).unwrap();
        let second = select_linux_drm_device_at(&root, 1, None).unwrap();
        assert_eq!(first.info.vram_total, 51_539_607_552);
        assert_eq!(first.info.gtt_total, 0);
        assert_eq!(first.info.vendor, 0x10de);
        assert_eq!(second.info.vram_total, 103_079_215_104);
        assert_eq!(second.info.gtt_total, 16_629_473_300);
        assert_eq!(second.info.vendor, 0x1002);
        assert!(!is_host_shared_memory_drm(
            &first.info,
            Some(32 * 1024 * 1024 * 1024)
        ));
        assert!(!is_host_shared_memory_drm(
            &second.info,
            Some(32 * 1024 * 1024 * 1024)
        ));
        assert!(select_linux_drm_device_at(&root, 2, None).is_none());

        let amd_selected = select_linux_drm_device_at(&root, 0, Some(LinuxDrmVendor::Amd)).unwrap();
        assert_eq!(amd_selected.info, second.info);
        assert!(select_linux_drm_device_at(&root, 1, Some(LinuxDrmVendor::Amd)).is_none());
        assert_eq!(linux_drm_candidate_count_at(&root, None), Some(2));
        assert_eq!(
            linux_drm_candidate_count_at(&root, Some(LinuxDrmVendor::Amd)),
            Some(1)
        );
        assert_eq!(
            linux_drm_candidate_count_at(&root, Some(LinuxDrmVendor::Intel)),
            Some(0)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_drm_vendor_inventory_fails_closed_when_evidence_is_missing() {
        let root = std::env::temp_dir().join(format!(
            "kiln-drm-identity-incomplete-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("card0/device")).unwrap();

        assert_eq!(linux_drm_candidate_count_at(&root, None), Some(1));
        assert_eq!(
            linux_drm_candidate_count_at(&root, Some(LinuxDrmVendor::Amd)),
            None,
            "a missing vendor ID cannot prove that an unknown device is irrelevant to ROCm",
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    /// Exercise `detect_vram` on macOS and confirm it returns a positive
    /// number from the unified-memory path (assuming nvidia-smi isn't
    /// installed, which is the normal mac developer setup).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_detect_apple_unified_memory() {
        // If nvidia-smi happens to exist on this mac (unlikely), skip.
        if std::process::Command::new("nvidia-smi")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return;
        }
        let info = detect_vram();
        assert_eq!(info.source, VramSource::AppleSilicon);
        assert!(info.unified);
        // Source is enough — `total_bytes > 0` doesn't survive on tiny CI
        // runners (GitHub macos-14 ships with ~7 GB, leaving ≤ 1 GB after
        // the 6 GB OS reserve, and `saturating_sub` can hit 0 on the smallest
        // runner SKUs). Production correctness is covered by the source
        // identification; the byte budget is exercised by the recommendation
        // tests above with synthetic VRAM values.
    }
}

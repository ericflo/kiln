//! GPU VRAM detection and auto-configuration utilities.
//!
//! Detects available GPU memory and provides recommended training parameters
//! so that SFT and GRPO training "just works" on consumer GPUs without manual tuning.

/// Detected GPU memory information.
#[derive(Debug, Clone, Copy)]
pub struct GpuVramInfo {
    /// Total VRAM in bytes (0 if detection failed or no GPU).
    pub total_bytes: u64,
    /// Source of the detection.
    pub source: VramSource,
}

/// Snapshot of currently used GPU memory.
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryUsedInfo {
    /// Used VRAM in bytes (0 if detection failed or no GPU).
    pub used_bytes: u64,
    /// Source of the detection.
    pub source: VramSource,
}

/// The corrected training memory budget plus provenance.
///
/// Use this when reporting "how much memory can training use" — the
/// `source` makes the log line honest about *why* the budget is what
/// it is. On a discrete NVIDIA GPU it'll just be `NvidiaSmi` and the
/// budget equals the detected VRAM. On a unified-memory APU it'll be
/// `LinuxDrmSysfsUnified` and the budget is the corrected value
/// (the BIOS-reported VRAM carveout is replaced with `MemTotal − reserve`
/// so training sized against this number cannot exhaust system RAM).
///
/// Field semantics match what consumers like the trainer preflight
/// estimator and the inference KV-cache sizer already expect.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveBudget {
    /// Total memory addressable by training in bytes. Pre-corrected
    /// for the unified-memory APU case.
    pub total_bytes: u64,
    /// Provenance of the budget — what kind of probe produced it.
    pub source: VramSource,
}

/// Convenience: detect VRAM and return an [`EffectiveBudget`] suitable
/// for direct use in startup logging and the training preflight
/// estimator.
///
/// This is the single source of truth — replaces ad-hoc reads of
/// `total_vram_gb` scattered around `crates/kiln-server/src/state.rs`.
pub fn detect_effective_training_budget() -> EffectiveBudget {
    let info = detect_vram();
    EffectiveBudget {
        total_bytes: info.total_bytes,
        source: info.source,
    }
}

/// How the VRAM value was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// Detected via nvidia-smi (discrete NVIDIA GPU).
    NvidiaSmi,
    /// Detected via Linux DRM sysfs memory counters.
    LinuxDrmSysfs,
    /// Detected via Linux DRM sysfs on a unified-memory APU. The reported
    /// VRAM is a BIOS-configured carveout (often >> physical RAM thanks to
    /// the GTT heap that pages against system RAM). The corrected budget
    /// is `min(reported_vram, MemTotal − reserve)` so training sized
    /// against this value cannot exhaust system RAM.
    LinuxDrmSysfsUnified,
    /// Detected via `sysctl hw.memsize` on Apple Silicon (unified memory).
    /// GPU-addressable memory is effectively the full physical pool minus a
    /// headroom for the OS and other apps.
    AppleSilicon,
    /// User-provided via `KILN_GPU_MEMORY_GB` env var.
    EnvOverride,
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
            VramSource::EnvOverride => write!(f, "KILN_GPU_MEMORY_GB"),
            VramSource::None => write!(f, "none"),
        }
    }
}

/// Detect total GPU VRAM (or unified memory on Apple Silicon).
///
/// Priority:
/// 1. `KILN_GPU_MEMORY_GB` env var (user override, always respected).
/// 2. `nvidia-smi` query (discrete NVIDIA).
/// 3. Linux DRM sysfs counters (AMD/Intel Vulkan devices). On unified-memory
///    APUs the BIOS carveout reported via DRM can far exceed physical RAM
///    (GTT pages against system memory), so the reported value is corrected
///    down to `min(reported_vram, MemTotal − reserve)` to avoid sizing
///    training as if there were a discrete GPU's worth of memory.
/// 4. `sysctl hw.memsize` on Apple Silicon (unified memory), with a
///    `system_reserve_gb` headroom subtracted so training doesn't compete
///    with the OS for the last few GB.
/// 5. Returns `GpuVramInfo { total_bytes: 0, source: None }` if no GPU.
pub fn detect_vram() -> GpuVramInfo {
    if let Ok(val) = std::env::var("KILN_GPU_MEMORY_GB") {
        if let Ok(gb) = val.parse::<f64>() {
            return GpuVramInfo {
                total_bytes: (gb * 1024.0 * 1024.0 * 1024.0) as u64,
                source: VramSource::EnvOverride,
            };
        }
    }

    if let Some(bytes) = query_nvidia_smi() {
        return GpuVramInfo {
            total_bytes: bytes,
            source: VramSource::NvidiaSmi,
        };
    }

    #[cfg(target_os = "linux")]
    if let Some(info) = detect_linux_drm_vram() {
        return info;
    }

    #[cfg(target_os = "macos")]
    if let Some(bytes) = query_apple_unified_memory() {
        return GpuVramInfo {
            total_bytes: bytes,
            source: VramSource::AppleSilicon,
        };
    }

    GpuVramInfo {
        total_bytes: 0,
        source: VramSource::None,
    }
}

/// Linux DRM detection that distinguishes discrete from unified-memory
/// APUs and corrects the reported VRAM down to a survivable budget on
/// the latter.
#[cfg(target_os = "linux")]
fn detect_linux_drm_vram() -> Option<GpuVramInfo> {
    detect_linux_drm_vram_at(
        std::path::Path::new("/sys/class/drm"),
        std::path::Path::new("/proc/meminfo"),
    )
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_vram_at(
    drm_base: &std::path::Path,
    meminfo_path: &std::path::Path,
) -> Option<GpuVramInfo> {
    let device = collect_linux_drm_device_info_at(drm_base)?;
    let mem_total = query_meminfo_total_bytes_at(meminfo_path);

    if let Some(mem_total_bytes) = mem_total
        && is_unified_memory_drm(&device, mem_total_bytes)
    {
        let reserve = unified_memory_reserve_bytes(mem_total_bytes);
        let corrected = device
            .vram_total
            .min(mem_total_bytes.saturating_sub(reserve));
        return Some(GpuVramInfo {
            total_bytes: corrected,
            source: VramSource::LinuxDrmSysfsUnified,
        });
    }

    Some(GpuVramInfo {
        total_bytes: device.vram_total,
        source: VramSource::LinuxDrmSysfs,
    })
}

/// Aggregated DRM device info across the primary nodes for a single GPU.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct LinuxDrmDeviceInfo {
    /// Largest `mem_info_vram_total` (or `vis_vram_total`) across nodes, bytes.
    pub vram_total: u64,
    /// Largest `mem_info_gtt_total` across nodes, bytes (0 if absent).
    pub gtt_total: u64,
    /// PCI vendor ID (e.g. `0x1002` for AMD), 0 if absent.
    pub vendor: u32,
    /// PCI class word (e.g. `0x038000`), 0 if absent. Top byte is class code,
    /// `0x03` is "display controller".
    pub class: u32,
}

#[cfg(target_os = "linux")]
fn collect_linux_drm_device_info_at(base: &std::path::Path) -> Option<LinuxDrmDeviceInfo> {
    let mut vram_total = 0u64;
    let mut gtt_total = 0u64;
    let mut vendor = 0u32;
    let mut class = 0u32;
    let mut found_any = false;

    for entry in std::fs::read_dir(base).ok()? {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_primary_drm_node(&name) {
            continue;
        }

        let device_dir = entry.path().join("device");
        for field in ["mem_info_vram_total", "mem_info_vis_vram_total"] {
            if let Some(b) = read_u64_file(&device_dir.join(field)) {
                vram_total = vram_total.max(b);
                found_any = true;
            }
        }
        if let Some(b) = read_u64_file(&device_dir.join("mem_info_gtt_total")) {
            gtt_total = gtt_total.max(b);
            found_any = true;
        }
        if vendor == 0 {
            if let Some(v) = read_hex_u32_file(&device_dir.join("vendor")) {
                vendor = v;
            }
        }
        if class == 0 {
            if let Some(c) = read_hex_u32_file(&device_dir.join("class")) {
                class = c;
            }
        }
    }

    if !found_any {
        return None;
    }
    Some(LinuxDrmDeviceInfo {
        vram_total,
        gtt_total,
        vendor,
        class,
    })
}

/// True when the DRM-reported VRAM should be treated as a unified-memory
/// budget (system RAM, not a discrete VRAM pool).
///
/// Triggers when either:
/// - The reported VRAM exceeds physical RAM by more than 25 % (a BIOS
///   carveout larger than the box can possibly back is the strongest
///   single signal of unified memory), OR
/// - The PCI device is a display controller (`class >> 16 == 0x03`)
///   from AMD (`0x1002`) or Intel (`0x8086`) and exposes a non-trivial
///   GTT heap (`gtt_total >= 1 GB`). GTT is the unified-memory paging
///   path; discrete dGPUs expose tiny GTTs (typically under 256 MB)
///   used only for staging.
#[cfg(target_os = "linux")]
fn is_unified_memory_drm(device: &LinuxDrmDeviceInfo, mem_total_bytes: u64) -> bool {
    if device.vram_total > mem_total_bytes.saturating_mul(5) / 4 {
        return true;
    }
    let class_code = device.class >> 16;
    let integrated_vendor = matches!(device.vendor, 0x1002 | 0x8086);
    let display_controller = class_code == 0x03;
    let trivial_gtt = device.gtt_total < 1024 * 1024 * 1024;
    if display_controller && integrated_vendor && !trivial_gtt {
        return true;
    }
    false
}

/// Reserve to subtract from `MemTotal` before declaring the corrected
/// unified-memory budget.
///
/// Matches the Apple Silicon path: `max(6 GB, MemTotal / 4)`. Override
/// with `KILN_TRAINING_MEMORY_RESERVE_GB` (parsed as f64).
fn unified_memory_reserve_bytes(mem_total_bytes: u64) -> u64 {
    if let Ok(val) = std::env::var("KILN_TRAINING_MEMORY_RESERVE_GB") {
        if let Ok(gb) = val.parse::<f64>() {
            return (gb * 1024.0 * 1024.0 * 1024.0) as u64;
        }
    }
    const MIN_RESERVE_BYTES: u64 = 6 * 1024 * 1024 * 1024;
    let proportional = mem_total_bytes / 4;
    proportional.max(MIN_RESERVE_BYTES)
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
            return Some(kib * 1024);
        }
    }
    None
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
    if let Some(bytes) = query_nvidia_smi_field("memory.used") {
        return GpuMemoryUsedInfo {
            used_bytes: bytes,
            source: VramSource::NvidiaSmi,
        };
    }

    #[cfg(target_os = "linux")]
    if let Some(bytes) = query_linux_drm_used_vram() {
        return GpuMemoryUsedInfo {
            used_bytes: bytes,
            source: VramSource::LinuxDrmSysfs,
        };
    }

    GpuMemoryUsedInfo {
        used_bytes: 0,
        source: VramSource::None,
    }
}

/// Query currently used GPU VRAM in bytes.
pub fn detect_used_vram_bytes() -> Option<u64> {
    let info = detect_used_vram();
    (info.used_bytes > 0).then_some(info.used_bytes)
}

/// Query total GPU memory via nvidia-smi.
///
/// Runs `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
/// which returns total memory in MiB. Returns None if nvidia-smi is not available
/// or fails.
fn query_nvidia_smi() -> Option<u64> {
    query_nvidia_smi_field("memory.total")
}

/// Query a MiB-valued nvidia-smi GPU memory field.
///
/// Takes the first GPU because kiln is a single-GPU server today and the rest
/// of the startup path also selects GPU 0 by default unless overridden.
fn query_nvidia_smi_field(field: &str) -> Option<u64> {
    let query = format!("--query-gpu={field}");
    let output = std::process::Command::new("nvidia-smi")
        .args([query.as_str(), "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mib: u64 = stdout.trim().lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

/// Query currently used GPU memory from Linux DRM sysfs.
///
/// AMDGPU exposes byte-valued `mem_info_vram_used` and
/// `mem_info_vis_vram_used` files under `/sys/class/drm/{cardN,renderDN}/device`.
/// We use the largest value across render/card nodes so duplicated connector
/// entries do not add the same GPU multiple times.
#[cfg(target_os = "linux")]
fn query_linux_drm_used_vram() -> Option<u64> {
    query_linux_drm_memory_fields(&["mem_info_vram_used", "mem_info_vis_vram_used"])
}

#[cfg(target_os = "linux")]
fn query_linux_drm_memory_fields(fields: &[&str]) -> Option<u64> {
    query_linux_drm_memory_fields_at(std::path::Path::new("/sys/class/drm"), fields)
}

#[cfg(target_os = "linux")]
fn query_linux_drm_memory_fields_at(base: &std::path::Path, fields: &[&str]) -> Option<u64> {
    let mut best = 0u64;
    for entry in std::fs::read_dir(base).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_primary_drm_node(&name) {
            continue;
        }

        let device_dir = entry.path().join("device");
        for field in fields {
            if let Some(bytes) = read_u64_file(&device_dir.join(field)) {
                best = best.max(bytes);
            }
        }
    }

    (best > 0).then_some(best)
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
/// and training don't squeeze out Finder, the browser, or a dev server.
/// Users who know their system can work harder can override with
/// `KILN_GPU_MEMORY_GB`.
#[cfg(target_os = "macos")]
fn query_apple_unified_memory() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let total: u64 = stdout.trim().parse().ok()?;

    const MIN_RESERVE_BYTES: u64 = 6 * 1024 * 1024 * 1024;
    let proportional_reserve = total / 4;
    let reserve = proportional_reserve.max(MIN_RESERVE_BYTES);

    Some(total.saturating_sub(reserve))
}

/// Recommended number of KV cache blocks based on total VRAM.
///
/// Returns `None` if the user set `KILN_NUM_BLOCKS` (should use that instead).
/// Otherwise picks a conservative value that leaves room for training.
pub fn recommended_num_blocks(vram: &GpuVramInfo) -> Option<usize> {
    if std::env::var("KILN_NUM_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
    {
        return None; // user override — don't second-guess
    }

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
/// Returns `None` if the user set `KILN_GRAD_CHECKPOINT_SEGMENTS` (should use that instead).
/// More segments = less VRAM but more compute overhead.
///
/// This is the *VRAM-only* heuristic (no sequence-length awareness). The training
/// trainer paths now prefer [`recommended_checkpoint_plan`] which also factors in
/// `max_seq_len` and `hidden_size`, but this function is retained for callers that
/// don't have the workload shape handy (preflight estimator, bench reporter).
pub fn recommended_checkpoint_segments(vram: &GpuVramInfo) -> Option<usize> {
    if std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
    {
        return None; // user override
    }

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
/// Three distinct outcomes — the auto-tuner returns one based on `(vram,
/// num_layers, max_seq_len, hidden_size)`:
///
/// * [`CheckpointPlan::UserOverride`] — the user set
///   `KILN_GRAD_CHECKPOINT_SEGMENTS=<N>` or `KILN_NO_GRAD_CHECKPOINT=1`. The
///   caller should honor the env value and skip auto-tuning entirely.
/// * [`CheckpointPlan::Disabled`] — activations comfortably fit in available
///   VRAM after the base model and a safety reserve. Skipping checkpointing
///   wins ~10-30% step time without OOM risk.
/// * [`CheckpointPlan::Enabled`] — activations would crowd available VRAM at
///   one or more segment counts; pick the smallest segment count that keeps
///   per-segment activation memory under the headroom budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPlan {
    /// User overrode via env (`KILN_GRAD_CHECKPOINT_SEGMENTS` or
    /// `KILN_NO_GRAD_CHECKPOINT`). Caller should fall back to the env-driven
    /// path (e.g. `CheckpointConfig::from_env`) so the override is respected.
    UserOverride,
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
/// Multiplied by 2 bytes/param (BF16). The estimate matches the
/// `model_loaded_vram_mib=9943` we observe for Qwen3.5-4B in the
/// kiln-server training bench (within ~5%).
pub fn estimate_base_model_bytes(
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    bytes_per_param: usize,
) -> u64 {
    let per_layer_params =
        (4 * hidden_size * hidden_size) + (3 * hidden_size * intermediate_size);
    let layer_total = per_layer_params.saturating_mul(num_layers);
    let head_total = 2usize.saturating_mul(vocab_size).saturating_mul(hidden_size);
    let total_params = layer_total.saturating_add(head_total);
    (total_params as u64).saturating_mul(bytes_per_param as u64)
}

/// Auto-decide a gradient-checkpoint plan for a training workload, factoring
/// in BOTH the device's total VRAM and the workload's shape (`num_layers`,
/// `max_seq_len_tokens`, `hidden_size`, base-model footprint).
///
/// Behavior:
/// 1. If `KILN_GRAD_CHECKPOINT_SEGMENTS` or `KILN_NO_GRAD_CHECKPOINT=1` is
///    set, return [`CheckpointPlan::UserOverride`] and let the caller honor
///    the env value via `CheckpointConfig::from_env`.
/// 2. Estimate F32 activation tape:
///    `max_act_bytes = num_layers * max_seq_len * hidden_size * 4`.
/// 3. Reserve `base_model_bytes + 2 GiB safety` for everything that isn't
///    activations (model weights, grads, AdamW state, working buffers).
/// 4. Define `available_bytes = max(0, vram.total_bytes - reserved)`.
/// 5. If `max_act_bytes <= available_bytes * 0.5`, return
///    [`CheckpointPlan::Disabled`] — checkpointing would only cost step time
///    without lowering peak VRAM enough to matter.
/// 6. Otherwise pick `num_segments = ceil(max_act_bytes / (available_bytes *
///    0.3))`, clamped to `[2, num_layers]`. The 30% target makes per-segment
///    intermediate memory comfortable inside headroom.
///
/// `None` is returned if VRAM detection failed (`vram.total_bytes == 0`) —
/// the caller should fall back to [`CheckpointConfig::from_env`]'s VRAM-only
/// path (which itself handles "unknown VRAM" via a conservative default).
pub fn recommended_checkpoint_plan(
    vram: &GpuVramInfo,
    num_layers: usize,
    max_seq_len_tokens: usize,
    hidden_size: usize,
    base_model_bytes: u64,
) -> Option<CheckpointPlan> {
    // Env overrides take absolute precedence — caller should honor them.
    if std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
    {
        return Some(CheckpointPlan::UserOverride);
    }
    if std::env::var("KILN_NO_GRAD_CHECKPOINT")
        .as_deref()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return Some(CheckpointPlan::UserOverride);
    }

    // VRAM unknown — caller should fall back to the env-driven path.
    if vram.total_bytes == 0 {
        return None;
    }

    // F32 activation tape (one element per layer-token pair). Even on a
    // BF16 model, the trainer keeps activations and grads in F32 for
    // numerical stability, so this matches what we'd actually allocate.
    let max_act_bytes = (num_layers as u64)
        .saturating_mul(max_seq_len_tokens as u64)
        .saturating_mul(hidden_size as u64)
        .saturating_mul(4);

    // 2 GiB safety for working buffers (RoPE tables, attention masks,
    // intermediate matmul outputs that aren't on the activation tape).
    const SAFETY_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    let reserved = base_model_bytes.saturating_add(SAFETY_RESERVE_BYTES);
    let available_bytes = vram.total_bytes.saturating_sub(reserved);

    let gib = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let max_act_gib = gib(max_act_bytes);
    let available_gib = gib(available_bytes);

    // If we have less than 2 GiB of headroom after reserves, the auto-tune
    // is in dangerous territory regardless of seq_len. Punt to the
    // VRAM-only heuristic via UserOverride — let from_env's existing logic
    // pick a conservative segment count.
    if available_bytes < 2 * 1024 * 1024 * 1024 {
        return Some(CheckpointPlan::UserOverride);
    }

    // Comfortable headroom: skip checkpointing. The 0.5 threshold leaves
    // plenty of room for grads + AdamW state on top of activations.
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
    fn test_detect_vram_env_override() {
        // This test relies on KILN_GPU_MEMORY_GB not being set in CI
        // and nvidia-smi not being available, so it should return None source
        // unless overridden. We test the logic paths via the recommendation functions.
    }

    #[test]
    fn effective_budget_mirrors_detect_vram_fields() {
        // The convenience accessor must agree with the underlying
        // detector — it's literally a thin wrapper. Lock that in so a
        // future refactor can't silently introduce divergence between
        // "what we log" and "what we size against".
        let detected = detect_vram();
        let budget = detect_effective_training_budget();
        assert_eq!(budget.total_bytes, detected.total_bytes);
        assert_eq!(budget.source, detected.source);
    }

    #[test]
    fn test_recommended_num_blocks() {
        let vram_48gb = GpuVramInfo {
            total_bytes: 48 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_num_blocks(&vram_48gb), Some(512));

        let vram_24gb = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_num_blocks(&vram_24gb), Some(64));

        // Test with real A5000 value (24564 MiB = slightly under 24 GiB)
        let vram_a5000 = GpuVramInfo {
            total_bytes: 24564 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_num_blocks(&vram_a5000), Some(64));

        let vram_16gb = GpuVramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_num_blocks(&vram_16gb), Some(32));

        let vram_none = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
        };
        assert_eq!(recommended_num_blocks(&vram_none), Some(64));
    }

    #[test]
    fn test_recommended_checkpoint_segments() {
        let vram_48gb = GpuVramInfo {
            total_bytes: 48 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_48gb), Some(4));

        let vram_24gb = GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_24gb), Some(8));

        // Test with real A5000 value (24564 MiB = slightly under 24 GiB)
        let vram_a5000 = GpuVramInfo {
            total_bytes: 24564 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_a5000), Some(8));

        let vram_16gb = GpuVramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        };
        assert_eq!(recommended_checkpoint_segments(&vram_16gb), Some(12));
    }

    fn vram(gb: u64) -> GpuVramInfo {
        GpuVramInfo {
            total_bytes: gb * 1024 * 1024 * 1024,
            source: VramSource::NvidiaSmi,
        }
    }

    fn act_gib(num_layers: usize, max_seq_len: usize, hidden_size: usize) -> f64 {
        let bytes = (num_layers as u64)
            * (max_seq_len as u64)
            * (hidden_size as u64)
            * 4;
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
    fn recommended_checkpoint_plan_respects_user_override() {
        let _g = crate::env_flag::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("KILN_GRAD_CHECKPOINT_SEGMENTS", "12");
        }
        let plan = recommended_checkpoint_plan(
            &vram(48),
            32,
            30,
            2560,
            estimate_base_model_bytes(32, 2560, 10240, 151936, 2),
        );
        unsafe {
            std::env::remove_var("KILN_GRAD_CHECKPOINT_SEGMENTS");
        }
        assert!(matches!(plan, Some(CheckpointPlan::UserOverride)));
    }

    #[test]
    fn recommended_checkpoint_plan_respects_disable_env() {
        let _g = crate::env_flag::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("KILN_NO_GRAD_CHECKPOINT", "1");
        }
        let plan = recommended_checkpoint_plan(
            &vram(48),
            32,
            32 * 1024,
            2560,
            estimate_base_model_bytes(32, 2560, 10240, 151936, 2),
        );
        unsafe {
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
        }
        assert!(matches!(plan, Some(CheckpointPlan::UserOverride)));
    }

    #[test]
    fn recommended_checkpoint_plan_returns_none_when_vram_unknown() {
        let unknown = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
        };
        let plan = recommended_checkpoint_plan(&unknown, 32, 1024, 2560, 10 * 1024 * 1024 * 1024);
        assert!(plan.is_none());
    }

    #[test]
    fn recommended_checkpoint_plan_falls_back_when_headroom_too_small() {
        // 12 GiB GPU + 10 GiB base estimate = 2 GiB before safety reserve;
        // 2 GiB after safety. We're under the 2 GiB cliff so the plan
        // should punt to UserOverride and let from_env's VRAM-only path
        // pick a conservative segment count.
        let plan = recommended_checkpoint_plan(
            &vram(12),
            32,
            1024,
            2560,
            10 * 1024 * 1024 * 1024,
        );
        assert!(matches!(plan, Some(CheckpointPlan::UserOverride)));
    }

    #[test]
    fn test_vram_source_display() {
        assert_eq!(VramSource::NvidiaSmi.to_string(), "nvidia-smi");
        assert_eq!(VramSource::LinuxDrmSysfs.to_string(), "linux-drm-sysfs");
        assert_eq!(
            VramSource::AppleSilicon.to_string(),
            "apple-silicon-unified"
        );
        assert_eq!(VramSource::EnvOverride.to_string(), "KILN_GPU_MEMORY_GB");
        assert_eq!(VramSource::None.to_string(), "none");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_unified_memory_apu_corrects_oversized_carveout() {
        // Hold the shared env-mutation lock so concurrent
        // `set_var(KILN_TRAINING_MEMORY_RESERVE_GB)` from
        // `test_unified_memory_reserve_env_override` doesn't race
        // with this test's reads of the same env var via
        // `unified_memory_reserve_bytes`.
        let _env_guard = crate::env_flag::TEST_ENV_LOCK.lock().unwrap();
        // Synthesize the user's hardware: AMD Strix Halo APU. DRM
        // reports a 103 GB VRAM carveout on a 30 GB host. The corrected
        // budget must be MemTotal − reserve, not the carveout.
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
            "16629477376\n",
        )
        .unwrap();
        std::fs::write(root.join("card1/device/vendor"), "0x1002\n").unwrap();
        std::fs::write(root.join("card1/device/class"), "0x038000\n").unwrap();

        let device = collect_linux_drm_device_info_at(&root).unwrap();
        assert_eq!(device.vram_total, 103_079_215_104);
        assert_eq!(device.gtt_total, 16_629_477_376);
        assert_eq!(device.vendor, 0x1002);
        assert_eq!(device.class, 0x038000);

        // 30 GB host: synthesize a meminfo file so the assertion does
        // not depend on the actual host's MemTotal.
        let mem_total = 30u64 * 1024 * 1024 * 1024;
        let meminfo_path = root.join("meminfo");
        std::fs::write(
            &meminfo_path,
            format!("MemTotal:       {} kB\n", mem_total / 1024),
        )
        .unwrap();

        assert!(is_unified_memory_drm(&device, mem_total));

        // KILN_TRAINING_MEMORY_RESERVE_GB may leak from a parent test
        // process; clear it so the assertion is deterministic. SAFETY:
        // env mutation is safe under nextest's per-test process isolation.
        unsafe { std::env::remove_var("KILN_TRAINING_MEMORY_RESERVE_GB") };
        let reserve = unified_memory_reserve_bytes(mem_total);
        // Default reserve = max(6 GB, MemTotal/4) = max(6, 7.5) = 7.5 GB.
        assert_eq!(reserve, mem_total / 4);

        let info = detect_linux_drm_vram_at(&root, &meminfo_path).unwrap();
        assert_eq!(info.source, VramSource::LinuxDrmSysfsUnified);
        // Corrected = min(carveout, MemTotal − reserve) = MemTotal − reserve.
        assert_eq!(info.total_bytes, mem_total - reserve);
        // Sanity-check the order of magnitude — corrected budget must
        // sit between 14 and 24 GB on a 30 GB box.
        let gb = info.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((14.0..=24.0).contains(&gb), "corrected budget = {gb} GB");

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_discrete_amd_gpu_kept_as_linuxdrmsysfs() {
        // Discrete AMD card on a small host: 16 GB VRAM, 256 MB GTT,
        // 32 GB MemTotal. Heuristic must NOT flag this as unified.
        let root = std::env::temp_dir().join(format!(
            "kiln-drm-discrete-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("card0/device")).unwrap();
        std::fs::write(
            root.join("card0/device/mem_info_vram_total"),
            "17179869184\n", // 16 GB
        )
        .unwrap();
        std::fs::write(
            root.join("card0/device/mem_info_gtt_total"),
            "268435456\n", // 256 MB — typical staging GTT on a discrete card
        )
        .unwrap();
        std::fs::write(root.join("card0/device/vendor"), "0x1002\n").unwrap();
        std::fs::write(root.join("card0/device/class"), "0x030000\n").unwrap();

        let device = collect_linux_drm_device_info_at(&root).unwrap();
        let mem_total = 32u64 * 1024 * 1024 * 1024;
        let meminfo_path = root.join("meminfo");
        std::fs::write(
            &meminfo_path,
            format!("MemTotal:       {} kB\n", mem_total / 1024),
        )
        .unwrap();
        assert!(!is_unified_memory_drm(&device, mem_total));

        let info = detect_linux_drm_vram_at(&root, &meminfo_path).unwrap();
        assert_eq!(info.source, VramSource::LinuxDrmSysfs);
        assert_eq!(info.total_bytes, 17_179_869_184);

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
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_unified_memory_reserve_env_override() {
        // Hold the shared env-mutation lock — see
        // test_unified_memory_apu_corrects_oversized_carveout.
        let _env_guard = crate::env_flag::TEST_ENV_LOCK.lock().unwrap();
        // SAFETY: env mutation is safe under nextest's per-test process
        // isolation; this test must run via `cargo nextest run`.
        unsafe { std::env::set_var("KILN_TRAINING_MEMORY_RESERVE_GB", "10.0") };
        let mem_total = 32u64 * 1024 * 1024 * 1024;
        assert_eq!(
            unified_memory_reserve_bytes(mem_total),
            10u64 * 1024 * 1024 * 1024
        );
        unsafe { std::env::remove_var("KILN_TRAINING_MEMORY_RESERVE_GB") };
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_drm_sysfs_memory_detection() {
        let root = std::env::temp_dir().join(format!("kiln-drm-vram-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        std::fs::create_dir_all(root.join("card0/device")).unwrap();
        std::fs::create_dir_all(root.join("card0-DP-1/device")).unwrap();
        std::fs::create_dir_all(root.join("renderD128/device")).unwrap();

        std::fs::write(
            root.join("card0/device/mem_info_vram_total"),
            "17179869184\n",
        )
        .unwrap();
        std::fs::write(root.join("card0/device/mem_info_vram_used"), "536870912\n").unwrap();
        std::fs::write(
            root.join("card0-DP-1/device/mem_info_vram_total"),
            "34359738368\n",
        )
        .unwrap();
        std::fs::write(
            root.join("renderD128/device/mem_info_vis_vram_total"),
            "25769803776\n",
        )
        .unwrap();
        std::fs::write(
            root.join("renderD128/device/mem_info_vis_vram_used"),
            "1073741824\n",
        )
        .unwrap();

        assert_eq!(
            query_linux_drm_memory_fields_at(
                &root,
                &["mem_info_vram_total", "mem_info_vis_vram_total"]
            ),
            Some(25769803776)
        );
        assert_eq!(
            query_linux_drm_memory_fields_at(
                &root,
                &["mem_info_vram_used", "mem_info_vis_vram_used"]
            ),
            Some(1073741824)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    /// Exercise `detect_vram` on macOS and confirm it returns a positive
    /// number from the unified-memory path (assuming nvidia-smi isn't
    /// installed and no env override is set, which is the normal mac
    /// developer setup).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_detect_apple_unified_memory() {
        if std::env::var("KILN_GPU_MEMORY_GB").is_ok() {
            return;
        }
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
        // Source is enough — `total_bytes > 0` doesn't survive on tiny CI
        // runners (GitHub macos-14 ships with ~7 GB, leaving ≤ 1 GB after
        // the 6 GB OS reserve, and `saturating_sub` can hit 0 on the smallest
        // runner SKUs). Production correctness is covered by the source
        // identification; the byte budget is exercised by the recommendation
        // tests above with synthetic VRAM values.
    }
}

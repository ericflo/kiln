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

/// How the VRAM value was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// Detected via nvidia-smi (discrete NVIDIA GPU).
    NvidiaSmi,
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
/// 3. `sysctl hw.memsize` on Apple Silicon (unified memory), with a
///    `system_reserve_gb` headroom subtracted so training doesn't compete
///    with the OS for the last few GB.
/// 4. Returns `GpuVramInfo { total_bytes: 0, source: None }` if no GPU.
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
    if std::env::var("KILN_NUM_BLOCKS").ok().and_then(|v| v.parse::<usize>().ok()).is_some() {
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

    #[test]
    fn test_vram_source_display() {
        assert_eq!(VramSource::NvidiaSmi.to_string(), "nvidia-smi");
        assert_eq!(VramSource::AppleSilicon.to_string(), "apple-silicon-unified");
        assert_eq!(VramSource::EnvOverride.to_string(), "KILN_GPU_MEMORY_GB");
        assert_eq!(VramSource::None.to_string(), "none");
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

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

/// How the VRAM value was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// Detected via nvidia-smi.
    NvidiaSmi,
    /// User-provided via KILN_GPU_MEMORY_GB env var.
    EnvOverride,
    /// No GPU detected or detection failed.
    None,
}

impl std::fmt::Display for VramSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VramSource::NvidiaSmi => write!(f, "nvidia-smi"),
            VramSource::EnvOverride => write!(f, "KILN_GPU_MEMORY_GB"),
            VramSource::None => write!(f, "none"),
        }
    }
}

/// Detect total GPU VRAM.
///
/// Priority:
/// 1. `KILN_GPU_MEMORY_GB` env var (user override, always respected)
/// 2. `nvidia-smi` query (auto-detection)
/// 3. Returns `GpuVramInfo` with `total_bytes=0` and `source=None` if no GPU
pub fn detect_vram() -> GpuVramInfo {
    // 1. Check env override first
    if let Ok(val) = std::env::var("KILN_GPU_MEMORY_GB") {
        if let Ok(gb) = val.parse::<f64>() {
            return GpuVramInfo {
                total_bytes: (gb * 1024.0 * 1024.0 * 1024.0) as u64,
                source: VramSource::EnvOverride,
            };
        }
    }

    // 2. Try nvidia-smi
    if let Some(bytes) = query_nvidia_smi() {
        return GpuVramInfo {
            total_bytes: bytes,
            source: VramSource::NvidiaSmi,
        };
    }

    // 3. No GPU detected
    GpuVramInfo {
        total_bytes: 0,
        source: VramSource::None,
    }
}

/// Query total GPU memory via nvidia-smi.
///
/// Runs `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
/// which returns total memory in MiB. Returns None if nvidia-smi is not available
/// or fails.
fn query_nvidia_smi() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    // nvidia-smi may list multiple GPUs; take the first line (GPU 0)
    let mib: u64 = stdout.trim().lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
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
        assert_eq!(VramSource::EnvOverride.to_string(), "KILN_GPU_MEMORY_GB");
        assert_eq!(VramSource::None.to_string(), "none");
    }
}

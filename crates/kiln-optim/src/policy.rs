//! Per-Parameter optimizer policy knobs.
//!
//! - [`MomentLocation`] — where AdamW running moments live. Phase 6.5
//!   issue bullet: "Optimizer-state location seam. AdamW per-parameter
//!   moments are 8 bytes/param at FP32; for Qwen3.5-4B that's 32 GiB
//!   optimizer state — the binding constraint on 16 GiB and the
//!   second-binding on 24 GiB."
//! - [`StochasticRoundingPolicy`] — round-to-nearest vs stochastic.
//!   Phase 6.5 issue bullet: "Round-to-nearest loses precision at the
//!   master-update step under small learning rates; stochastic
//!   rounding (round up or down with probability proportional to the
//!   fractional position) preserves the in-expectation update."

/// Where AdamW running moments live for a given Parameter.
///
/// `#[non_exhaustive]` — future Phase 8.x may add `IpcShared` for
/// multi-process training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MomentLocation {
    /// On the same device as the parameter's `backward_storage`.
    /// Fastest but consumes ~8 bytes/param at FP32.
    /// Default on 80 GiB tier.
    Device,
    /// In pinned host RAM. Async-paged for the optimizer step.
    /// Default on 16 GiB tier (per the Phase 6.5 auto-sizer).
    PinnedHost,
    /// Mmapped to NVMe scratch. For very-long-running fine-tunes on
    /// 16 GiB tier where even pinned-host pressure forces it.
    MmappedDisk,
}

impl MomentLocation {
    /// Stable short name. Used by checkpoint receipts + Phase 9
    /// audit logs.
    pub const fn name(self) -> &'static str {
        match self {
            MomentLocation::Device => "device",
            MomentLocation::PinnedHost => "pinned_host",
            MomentLocation::MmappedDisk => "mmapped_disk",
        }
    }

    /// `true` iff the moments are physically resident on the same
    /// device as the parameter (no offload cost on the step path).
    pub const fn is_resident(self) -> bool {
        matches!(self, MomentLocation::Device)
    }
}

impl Default for MomentLocation {
    fn default() -> Self {
        MomentLocation::Device
    }
}

/// Rounding policy for BF16 master updates.
///
/// Per the Phase 6.5 issue bullet: stochastic rounding preserves the
/// in-expectation update under small learning rates. Gated by
/// `KILN_BF16_STOCHASTIC_ROUND=1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StochasticRoundingPolicy {
    /// Standard IEEE-754 round-to-nearest-even.
    RoundToNearest,
    /// Stochastic rounding: round up or down with probability
    /// proportional to the fractional position.
    Stochastic { seed: u64 },
}

impl StochasticRoundingPolicy {
    pub const fn name(self) -> &'static str {
        match self {
            StochasticRoundingPolicy::RoundToNearest => "round_to_nearest",
            StochasticRoundingPolicy::Stochastic { .. } => "stochastic",
        }
    }

    /// Read from the env. Returns `Stochastic { seed: 42 }` for
    /// `KILN_BF16_STOCHASTIC_ROUND=1|true|yes`; otherwise
    /// `RoundToNearest`. The seed default is intentionally fixed so
    /// runs are reproducible at the same seed; callers can override
    /// with [`Self::stochastic_with_seed`].
    pub fn from_env() -> Self {
        match std::env::var("KILN_BF16_STOCHASTIC_ROUND").ok().as_deref() {
            Some(v) => {
                let v = v.trim().to_ascii_lowercase();
                if matches!(v.as_str(), "1" | "true" | "yes") {
                    StochasticRoundingPolicy::Stochastic { seed: 42 }
                } else {
                    StochasticRoundingPolicy::RoundToNearest
                }
            }
            None => StochasticRoundingPolicy::RoundToNearest,
        }
    }

    /// Construct a stochastic-rounding policy with an explicit seed.
    pub const fn stochastic_with_seed(seed: u64) -> Self {
        StochasticRoundingPolicy::Stochastic { seed }
    }
}

impl Default for StochasticRoundingPolicy {
    fn default() -> Self {
        StochasticRoundingPolicy::RoundToNearest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moment_location_names() {
        assert_eq!(MomentLocation::Device.name(), "device");
        assert_eq!(MomentLocation::PinnedHost.name(), "pinned_host");
        assert_eq!(MomentLocation::MmappedDisk.name(), "mmapped_disk");
    }

    #[test]
    fn moment_location_residency() {
        assert!(MomentLocation::Device.is_resident());
        assert!(!MomentLocation::PinnedHost.is_resident());
        assert!(!MomentLocation::MmappedDisk.is_resident());
    }

    #[test]
    fn default_moment_location_is_device() {
        assert_eq!(MomentLocation::default(), MomentLocation::Device);
    }

    #[test]
    fn stochastic_with_seed() {
        let p = StochasticRoundingPolicy::stochastic_with_seed(7);
        assert_eq!(p.name(), "stochastic");
        match p {
            StochasticRoundingPolicy::Stochastic { seed } => assert_eq!(seed, 7),
            _ => panic!(),
        }
    }

    #[test]
    fn default_round_policy_is_nearest() {
        assert_eq!(
            StochasticRoundingPolicy::default(),
            StochasticRoundingPolicy::RoundToNearest
        );
    }

    #[test]
    fn round_policy_from_env() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap();
        unsafe { std::env::remove_var("KILN_BF16_STOCHASTIC_ROUND") };
        assert_eq!(
            StochasticRoundingPolicy::from_env(),
            StochasticRoundingPolicy::RoundToNearest
        );
        unsafe { std::env::set_var("KILN_BF16_STOCHASTIC_ROUND", "1") };
        assert!(matches!(
            StochasticRoundingPolicy::from_env(),
            StochasticRoundingPolicy::Stochastic { .. }
        ));
        unsafe { std::env::set_var("KILN_BF16_STOCHASTIC_ROUND", "no") };
        assert_eq!(
            StochasticRoundingPolicy::from_env(),
            StochasticRoundingPolicy::RoundToNearest
        );
        unsafe { std::env::remove_var("KILN_BF16_STOCHASTIC_ROUND") };
    }
}

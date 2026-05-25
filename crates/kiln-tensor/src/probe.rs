//! Runtime backend probes — "is CUDA/Metal usable on this host right now?"
//!
//! Phase-7 candle-removal callers (e.g. `kiln-server`'s startup banner
//! and `device::select_device_with_options`) used to ask
//! `candle_core::utils::cuda_is_available()` /
//! `candle_core::utils::metal_is_available()` directly. Those two functions
//! are simple "the cuda/metal feature is compiled in AND the driver is
//! present" gates — they don't touch candle's Tensor surface at all, so
//! they can be reflected behind a kt-tensor probe today without waiting
//! for the rest of #1082.
//!
//! Under `--features cuda` we delegate to `candle_core::utils::cuda_is_available()`
//! (kiln-tensor already pulls candle-core in for storage), so the probe
//! returns whatever candle decided was available. When the cuda feature
//! is off the probe trivially returns `false`. Same shape for metal.
//!
//! Phase 7 (candle removal) replaces the bodies with a direct cudarc /
//! objc2-metal probe; callers do not have to change.

/// `true` iff this build was compiled with `--features cuda` AND a CUDA
/// device is currently visible to the driver.
///
/// Equivalent to `candle_core::utils::cuda_is_available()`. Callers
/// should prefer this over reaching into candle directly so we have a
/// single seam to replace in Phase 7 of #1082.
#[inline]
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        candle_core::utils::cuda_is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// `true` iff this build was compiled with `--features metal` AND a
/// Metal device is currently visible to the driver.
///
/// Equivalent to `candle_core::utils::metal_is_available()`. Callers
/// should prefer this over reaching into candle directly so we have a
/// single seam to replace in Phase 7 of #1082.
#[inline]
pub fn metal_is_available() -> bool {
    #[cfg(feature = "metal")]
    {
        candle_core::utils::metal_is_available()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_build_returns_false_for_both_probes() {
        // When neither cuda nor metal is compiled in, both probes are
        // const-false. When one is compiled in, the value is whatever
        // candle reports — we don't pin that here because parity-test
        // hosts may or may not have a driver visible. The point of this
        // test is just to guarantee the function is callable on every
        // feature configuration without panicking.
        let _ = cuda_is_available();
        let _ = metal_is_available();
    }
}

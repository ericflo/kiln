//! Runtime backend probes — "is CUDA/Metal usable on this host right now?"
//!
//! These are simple "the cuda/metal feature is compiled in" gates that
//! don't touch candle's Tensor surface at all. Both bodies are
//! candle-free as of #1082: identical semantics to candle's
//! `cuda_is_available()` / `metal_is_available()` (which are themselves
//! just `cfg!(feature = "...")` returns).
//!
//! Callers (e.g. `kiln-server`'s startup banner and
//! `device::select_device_with_options`) use these so we have a single
//! seam — even though the bodies are trivial.

/// `true` iff this build was compiled with `--features cuda`.
///
/// Equivalent to `candle_core::utils::cuda_is_available()` (which is
/// itself just `cfg!(feature = "cuda")`). The fn is kept so callers
/// route through `kiln_tensor` rather than naming candle directly.
#[inline]
pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

/// `true` iff this build was compiled with `--features metal`.
///
/// Equivalent to `candle_core::utils::metal_is_available()` (which is
/// itself just `cfg!(feature = "metal")`). The fn is kept so callers
/// route through `kiln_tensor` rather than naming candle directly.
#[inline]
pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
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

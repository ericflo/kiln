//! Thin NVTX range wrapper for nsys per-call-site attribution.
//!
//! Two design goals:
//!
//! 1. **Zero overhead when the `nvtx` feature is off.** The disabled
//!    `Range::push` is `#[inline]` and constructs a unit struct; no FFI, no
//!    allocation, no string formatting. The optimizer collapses it to nothing
//!    in release builds, so non-CUDA / non-nsys builds (CI, training-only
//!    laptops) pay nothing.
//!
//! 2. **No external NVTX dependency.** We call `nvtxRangePushA` /
//!    `nvtxRangePop` directly. `libnvToolsExt.so` ships with the CUDA toolkit;
//!    `build.rs` adds it to the link line when the feature is on.
//!
//! Range names are `&'static CStr` literals at the call site so the hot path
//! never allocates. Use the [`range!`] macro for the common case.

#![cfg_attr(not(feature = "nvtx"), allow(dead_code))]

use std::ffi::CStr;

/// RAII guard that holds an open NVTX range.  Drop pops it.
///
/// When the `nvtx` feature is disabled this is a zero-sized type and `push`
/// is a no-op; the optimizer eliminates the construct/drop pair in release
/// builds.
pub struct Range {
    // Marker so the user cannot construct one outside this module.
    _priv: (),
}

#[cfg(feature = "nvtx")]
mod ffi {
    use std::os::raw::{c_char, c_int};
    unsafe extern "C" {
        pub fn nvtxRangePushA(message: *const c_char) -> c_int;
        pub fn nvtxRangePop() -> c_int;
    }
}

impl Range {
    /// Push an NVTX range named `name` (a NUL-terminated static string) and
    /// return a guard.  Drop the guard to close the range.
    ///
    /// Using `&'static CStr` keeps the hot path allocation-free and lets the
    /// caller hard-code the range name as a `c"..."` literal (Rust 1.77+).
    #[inline]
    pub fn push(name: &'static CStr) -> Self {
        #[cfg(feature = "nvtx")]
        unsafe {
            let _ = ffi::nvtxRangePushA(name.as_ptr());
        }
        let _ = name;
        Self { _priv: () }
    }
}

impl Drop for Range {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "nvtx")]
        unsafe {
            let _ = ffi::nvtxRangePop();
        }
    }
}

/// Open a named NVTX range bound to the enclosing scope.
///
/// The argument must be a `c"..."` C-string literal so we can pass it to NVTX
/// without allocation.
///
/// ```ignore
/// fn forward(...) -> Result<...> {
///     kiln_nvtx::range!(c"kiln/attn/full/decode_fused");
///     // ... body ...
/// }
/// ```
///
/// Expands to `let _g = Range::push(...);` so the range closes when the scope
/// ends.
#[macro_export]
macro_rules! range {
    ($name:expr) => {
        let _kiln_nvtx_guard = $crate::Range::push($name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the disabled-feature path must compile and run without
    /// CUDA or `libnvToolsExt`.  When the feature is on the calls are real
    /// FFI; we just verify they don't crash.
    #[test]
    fn push_pop_does_not_crash() {
        let _g = Range::push(c"kiln/test/unit");
        // Drop runs at end of scope.
    }
}

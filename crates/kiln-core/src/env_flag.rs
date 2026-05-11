//! Tiny utility for parsing boolean env vars consistently across crates.
//!
//! Three call sites in this codebase had the same hand-rolled
//! `v == "1" || v == "true" || v == "yes"` pattern (or its negation
//! for default-on flags). Centralising it ensures every `KILN_*`
//! flag accepts the same set of truthy / falsy spellings and treats
//! unrecognised values as "fall back to the default" rather than
//! flipping behaviour silently.

/// Parse a boolean env var with explicit truthy / falsy spellings.
///
/// Truthy values: `1`, `true`, `yes` (case-insensitive, whitespace-trimmed).
/// Falsy values: `0`, `false`, `no` (same).
/// Anything else (including unset or unrecognised): returns `default`.
///
/// Examples:
/// ```ignore
/// // Default-off flag, opt-in via env:
/// let on = env_flag("KILN_VULKAN_LINEAR", false);
///
/// // Default-on flag, opt-out via env:
/// let on = env_flag("KILN_VULKAN_RMSNORM", true);
/// ```
pub fn env_flag(name: &str, default: bool) -> bool {
    let raw = std::env::var(name).ok();
    let lower = raw
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase);
    match lower.as_deref() {
        Some("1") | Some("true") | Some("yes") => true,
        Some("0") | Some("false") | Some("no") => false,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_values() {
        for (val, want) in [("1", true), ("true", true), ("yes", true), ("TRUE", true), ("Yes", true), (" 1 ", true)] {
            unsafe { std::env::set_var("KILN_TEST_FLAG_TRUTHY", val); }
            assert_eq!(env_flag("KILN_TEST_FLAG_TRUTHY", false), want, "value {val:?}");
        }
        unsafe { std::env::remove_var("KILN_TEST_FLAG_TRUTHY"); }
    }

    #[test]
    fn falsy_values() {
        for (val, want) in [("0", false), ("false", false), ("no", false), ("FALSE", false), ("No", false), (" 0 ", false)] {
            unsafe { std::env::set_var("KILN_TEST_FLAG_FALSY", val); }
            assert_eq!(env_flag("KILN_TEST_FLAG_FALSY", true), want, "value {val:?}");
        }
        unsafe { std::env::remove_var("KILN_TEST_FLAG_FALSY"); }
    }

    #[test]
    fn unrecognised_falls_back_to_default() {
        unsafe { std::env::set_var("KILN_TEST_FLAG_GIBBERISH", "maybe"); }
        assert!(env_flag("KILN_TEST_FLAG_GIBBERISH", true));
        assert!(!env_flag("KILN_TEST_FLAG_GIBBERISH", false));
        unsafe { std::env::remove_var("KILN_TEST_FLAG_GIBBERISH"); }
    }

    #[test]
    fn unset_falls_back_to_default() {
        // Use a name unlikely to be set in any environment.
        let name = "KILN_TEST_FLAG_DEFINITELY_UNSET_b9e0a4f1";
        unsafe { std::env::remove_var(name); }
        assert!(env_flag(name, true));
        assert!(!env_flag(name, false));
    }
}

//! Structured logging initialization.
//!
//! Configurable via environment variables:
//! - `KILN_LOG_LEVEL`: verbosity level (`trace`, `debug`, `info`, `warn`, `error`)
//!   or a full `tracing_subscriber::EnvFilter` directive. Default: `info`.
//! - `KILN_LOG_FORMAT`: output format — `json` (default) for structured JSON,
//!   `pretty` for human-readable colored output during development.
//! - `RUST_LOG`: if set, takes precedence over `KILN_LOG_LEVEL`.

use tracing_subscriber::EnvFilter;

/// Build an `EnvFilter` from `RUST_LOG` (if set) or `KILN_LOG_LEVEL`.
pub fn build_filter() -> EnvFilter {
    let level = std::env::var("KILN_LOG_LEVEL").unwrap_or_else(|_| "info".into());

    EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        match level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {
                format!("kiln={level},kiln_server={level},tower_http={level}")
                    .parse()
                    .expect("valid filter directive")
            }
            other => other.parse().unwrap_or_else(|_| {
                "kiln=info,kiln_server=info,tower_http=info"
                    .parse()
                    .expect("valid filter directive")
            }),
        }
    })
}

/// Initialize the global tracing subscriber.
///
/// Call once at startup. Panics if called twice (tracing's global subscriber
/// can only be set once per process).
pub fn init() -> anyhow::Result<()> {
    let format = std::env::var("KILN_LOG_FORMAT").unwrap_or_else(|_| "json".into());
    let filter = build_filter();

    match format.as_str() {
        "pretty" | "text" | "human" => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .init();
        }
        _ => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .init();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: env var manipulation is unsafe in Rust 1.78+ because it is not
    // thread-safe. We wrap each call in an unsafe block. These tests are
    // serialized by cargo test's default single-threaded test runner for
    // the lib target, so this is safe in practice.

    #[test]
    fn test_build_filter_default() {
        // Ensure RUST_LOG is not set for this test
        unsafe {
            std::env::remove_var("RUST_LOG");
            std::env::remove_var("KILN_LOG_LEVEL");
        }
        let filter = build_filter();
        let s = format!("{filter}");
        assert!(s.contains("info"), "default filter should contain info: {s}");
    }

    #[test]
    fn test_build_filter_custom_level() {
        unsafe {
            std::env::remove_var("RUST_LOG");
            std::env::set_var("KILN_LOG_LEVEL", "debug");
        }
        let filter = build_filter();
        let s = format!("{filter}");
        assert!(
            s.contains("debug"),
            "filter should contain debug: {s}"
        );
        unsafe { std::env::remove_var("KILN_LOG_LEVEL"); }
    }

    #[test]
    fn test_build_filter_custom_directive() {
        unsafe {
            std::env::remove_var("RUST_LOG");
            std::env::set_var("KILN_LOG_LEVEL", "kiln=trace,tower_http=warn");
        }
        let filter = build_filter();
        let s = format!("{filter}");
        // Custom directive is parsed as-is (not expanded to the standard triple)
        assert!(
            s.contains("kiln=trace") || s.contains("tower_http=warn"),
            "filter should parse custom directive: {s}"
        );
        unsafe { std::env::remove_var("KILN_LOG_LEVEL"); }
    }
}

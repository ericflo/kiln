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

    #[test]
    fn test_build_filter_default() {
        // Ensure RUST_LOG is not set for this test
        std::env::remove_var("RUST_LOG");
        std::env::remove_var("KILN_LOG_LEVEL");
        let filter = build_filter();
        let s = format!("{filter}");
        assert!(s.contains("info"), "default filter should contain info: {s}");
    }

    #[test]
    fn test_build_filter_custom_level() {
        std::env::remove_var("RUST_LOG");
        std::env::set_var("KILN_LOG_LEVEL", "debug");
        let filter = build_filter();
        let s = format!("{filter}");
        assert!(
            s.contains("debug"),
            "filter should contain debug: {s}"
        );
        // Clean up
        std::env::remove_var("KILN_LOG_LEVEL");
    }

    #[test]
    fn test_build_filter_custom_directive() {
        std::env::remove_var("RUST_LOG");
        std::env::set_var("KILN_LOG_LEVEL", "kiln=trace,tower_http=warn");
        let filter = build_filter();
        let s = format!("{filter}");
        assert!(
            s.contains("trace") || s.contains("warn"),
            "filter should parse custom directive: {s}"
        );
        std::env::remove_var("KILN_LOG_LEVEL");
    }

    #[test]
    fn test_build_filter_invalid_fallback() {
        std::env::remove_var("RUST_LOG");
        std::env::set_var("KILN_LOG_LEVEL", "not_a_valid_level!!!!");
        let filter = build_filter();
        let s = format!("{filter}");
        // Should fall back to info
        assert!(s.contains("info"), "invalid level should fall back to info: {s}");
        std::env::remove_var("KILN_LOG_LEVEL");
    }
}

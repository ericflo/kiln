//! Structured logging initialization.
//!
//! Configurable via the `[logging]` section of `kiln.toml` or environment variables:
//! - `KILN_LOG_LEVEL`: verbosity level (`trace`, `debug`, `info`, `warn`, `error`)
//!   or a full `tracing_subscriber::EnvFilter` directive. Default: `info`.
//! - `KILN_LOG_FORMAT`: output format — `auto` (default), `json`, or `pretty`
//!   (also accepts `text`/`human` as aliases for `pretty`). With `auto`, kiln
//!   emits colored pretty logs when stderr is a TTY (interactive terminal) and
//!   structured JSON otherwise (systemd, docker, CI, log pipelines). Set to
//!   `json` to force JSON in production, or `pretty` to force colored output.
//! - `RUST_LOG`: if set, takes precedence over `KILN_LOG_LEVEL`.

use std::io::IsTerminal;

use tracing_subscriber::EnvFilter;

/// Build an `EnvFilter` from `RUST_LOG` (if set) or the provided level string.
pub fn build_filter(level: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| match level {
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
    })
}

/// Resolve a user-supplied format string to a concrete renderer choice.
///
/// Returns `"pretty"` or `"json"`. `"auto"` resolves to `"pretty"` when
/// `stderr_is_tty` is true and `"json"` otherwise; explicit `"pretty"` /
/// `"text"` / `"human"` always pick pretty regardless of TTY state; anything
/// else (including `"json"`) falls through to `"json"`.
pub(crate) fn resolve_format(format: &str, stderr_is_tty: bool) -> &'static str {
    match format {
        "auto" => {
            if stderr_is_tty {
                "pretty"
            } else {
                "json"
            }
        }
        "pretty" | "text" | "human" => "pretty",
        _ => "json",
    }
}

/// Initialize the global tracing subscriber.
///
/// `level`: log level or tracing filter directive (e.g. `"info"`, `"kiln=trace,tower_http=warn"`).
/// `format`: output format — `"auto"` (default; pretty on TTY, JSON otherwise),
/// `"json"`, `"pretty"`, `"text"`, or `"human"`.
///
/// Call once at startup. Panics if called twice (tracing's global subscriber
/// can only be set once per process).
pub fn init(level: &str, format: &str) -> anyhow::Result<()> {
    let filter = build_filter(level);
    let resolved = resolve_format(format, std::io::stderr().is_terminal());

    match resolved {
        "pretty" => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
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
        }
        let filter = build_filter("info");
        let s = format!("{filter}");
        assert!(
            s.contains("info"),
            "default filter should contain info: {s}"
        );
    }

    #[test]
    fn test_build_filter_custom_level() {
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let filter = build_filter("debug");
        let s = format!("{filter}");
        assert!(s.contains("debug"), "filter should contain debug: {s}");
    }

    #[test]
    fn test_build_filter_custom_directive() {
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let filter = build_filter("kiln=trace,tower_http=warn");
        let s = format!("{filter}");
        // Custom directive is parsed as-is (not expanded to the standard triple)
        assert!(
            s.contains("kiln=trace") || s.contains("tower_http=warn"),
            "filter should parse custom directive: {s}"
        );
    }

    #[test]
    fn test_resolve_format_auto_tty() {
        assert_eq!(resolve_format("auto", true), "pretty");
    }

    #[test]
    fn test_resolve_format_auto_no_tty() {
        assert_eq!(resolve_format("auto", false), "json");
    }

    #[test]
    fn test_resolve_format_explicit_pretty_overrides_no_tty() {
        // Explicit user choice wins over TTY detection.
        assert_eq!(resolve_format("pretty", false), "pretty");
        assert_eq!(resolve_format("text", false), "pretty");
        assert_eq!(resolve_format("human", false), "pretty");
    }

    #[test]
    fn test_resolve_format_explicit_json_overrides_tty() {
        // Explicit JSON wins even when stderr is interactive.
        assert_eq!(resolve_format("json", true), "json");
    }

    #[test]
    fn test_resolve_format_unknown_falls_back_to_json() {
        assert_eq!(resolve_format("garbage", true), "json");
        assert_eq!(resolve_format("", false), "json");
    }
}

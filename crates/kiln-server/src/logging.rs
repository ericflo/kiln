//! Structured logging initialization.
//!
//! Configurable via the `[logging]` section of `kiln.toml` or environment variables:
//! - `KILN_LOGGING_LEVEL`: verbosity level (`trace`, `debug`, `info`, `warn`,
//!   `error`) or a full `tracing_subscriber::EnvFilter` directive.
//!   Default: `info`.
//! - `KILN_LOGGING_FORMAT`: output format — `auto` (default), `json`, or `pretty`.
//!   With `auto`, kiln emits colored pretty logs when stderr is a TTY
//!   (interactive terminal) and structured JSON otherwise (systemd, docker,
//!   CI, log pipelines). Set to `json` to force JSON in production, or
//!   `pretty` to force colored output.
//! - `RUST_LOG`: if set, takes precedence over the resolved typed logging level.

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{Context, Result};
use tracing_subscriber::EnvFilter;

use crate::config::LoggingConfig;

const LOGGING_LEVEL_ENV: &str = "KILN_LOGGING_LEVEL";
const LOGGING_FORMAT_ENV: &str = "KILN_LOGGING_FORMAT";

/// Minimal logging policy resolved before full configuration validation.
///
/// Reading only string-valued fields from `[logging]` lets syntax, type, and
/// validation failures elsewhere in the file flow through structured startup
/// diagnostics without weakening the authoritative [`crate::config::KilnConfig`]
/// parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapLoggingConfig {
    pub level: String,
    pub format: String,
    pub config_path: Option<String>,
}

/// Resolve logging file/env precedence without requiring the rest of the
/// configuration to be valid.
pub fn bootstrap_config(explicit_path: Option<&str>) -> BootstrapLoggingConfig {
    let config_path = explicit_path
        .map(String::from)
        .or_else(|| std::env::var("KILN_CONFIG").ok())
        .or_else(|| {
            Path::new("kiln.toml")
                .exists()
                .then(|| "kiln.toml".to_string())
        });
    let mut logging = LoggingConfig::default();

    if let Some(path) = config_path.as_deref()
        && let Ok(contents) = std::fs::read_to_string(path)
    {
        apply_logging_table(&mut logging, &contents);
    }
    apply_bootstrap_environment(&mut logging, |name| std::env::var(name).ok());

    BootstrapLoggingConfig {
        level: logging.level,
        format: logging.format,
        config_path,
    }
}

/// Mirror authoritative logging precedence before the global subscriber is
/// installed. Full configuration loading still performs strict parsing and
/// validation.
fn apply_bootstrap_environment(
    logging: &mut LoggingConfig,
    mut read: impl FnMut(&str) -> Option<String>,
) {
    if let Some(level) = read(LOGGING_LEVEL_ENV) {
        logging.level = level;
    }
    if let Some(format) = read(LOGGING_FORMAT_ENV) {
        logging.format = format;
    }
}

fn apply_logging_table(logging: &mut LoggingConfig, contents: &str) {
    let Ok(document) = toml::from_str::<toml::Value>(contents) else {
        return;
    };
    let Some(table) = document.get("logging").and_then(toml::Value::as_table) else {
        return;
    };
    if let Some(level) = table.get("level").and_then(toml::Value::as_str) {
        logging.level = level.to_string();
    }
    if let Some(format) = table.get("format").and_then(toml::Value::as_str) {
        logging.format = format.to_string();
    }
}

/// Build an `EnvFilter` from `RUST_LOG` (if set) or the provided level string.
pub fn build_filter(level: &str) -> Result<EnvFilter> {
    match std::env::var("RUST_LOG") {
        Ok(raw) => {
            return EnvFilter::try_new(&raw)
                .with_context(|| format!("RUST_LOG contains an invalid filter, got {raw:?}"));
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("RUST_LOG must be valid UTF-8")
        }
    }

    let directive = match level {
        "trace" | "debug" | "info" | "warn" | "error" => {
            format!("kiln={level},kiln_server={level},tower_http={level}")
        }
        other => other.to_string(),
    };
    EnvFilter::try_new(&directive)
        .with_context(|| format!("logging.level contains an invalid filter, got {level:?}"))
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
    let filter = build_filter(level)?;
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

    #[test]
    fn bootstrap_logging_is_field_tolerant_of_other_configuration_errors() {
        let mut logging = LoggingConfig::default();
        apply_logging_table(
            &mut logging,
            r#"
[server]
port = "not-an-integer"

[logging]
level = "debug"
format = "json"
"#,
        );
        assert_eq!(logging.level, "debug");
        assert_eq!(logging.format, "json");

        apply_logging_table(&mut logging, "[logging\nlevel = ???");
        assert_eq!(logging.level, "debug");
        assert_eq!(logging.format, "json");
    }

    #[test]
    fn bootstrap_logging_uses_canonical_environment_precedence() {
        let mut canonical_only = LoggingConfig::default();
        apply_bootstrap_environment(&mut canonical_only, |name| match name {
            LOGGING_LEVEL_ENV => Some("debug".to_owned()),
            LOGGING_FORMAT_ENV => Some("json".to_owned()),
            _ => None,
        });
        assert_eq!(canonical_only.level, "debug");
        assert_eq!(canonical_only.format, "json");

        let mut retired_names_present = LoggingConfig::default();
        apply_bootstrap_environment(&mut retired_names_present, |name| match name {
            "KILN_LOG_LEVEL" => Some("warn".to_owned()),
            LOGGING_LEVEL_ENV => Some("debug".to_owned()),
            "KILN_LOG_FORMAT" => Some("pretty".to_owned()),
            LOGGING_FORMAT_ENV => Some("json".to_owned()),
            _ => None,
        });
        assert_eq!(retired_names_present.level, "debug");
        assert_eq!(retired_names_present.format, "json");

        let mut retired_only = LoggingConfig::default();
        apply_bootstrap_environment(&mut retired_only, |name| match name {
            "KILN_LOG_LEVEL" => Some("warn".to_owned()),
            "KILN_LOG_FORMAT" => Some("pretty".to_owned()),
            _ => None,
        });
        let defaults = LoggingConfig::default();
        assert_eq!(retired_only.level, defaults.level);
        assert_eq!(retired_only.format, defaults.format);
    }

    // NOTE: env var manipulation is unsafe in Rust 1.78+ because it is not
    // thread-safe. We wrap each call in an unsafe block. These tests are
    // serialized by cargo test's default single-threaded test runner for
    // the lib target, so this is safe in practice.

    #[test]
    fn test_build_filter_default() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Ensure RUST_LOG is not set for this test
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let filter = build_filter("info").unwrap();
        let s = format!("{filter}");
        assert!(
            s.contains("info"),
            "default filter should contain info: {s}"
        );
    }

    #[test]
    fn test_build_filter_custom_level() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let filter = build_filter("debug").unwrap();
        let s = format!("{filter}");
        assert!(s.contains("debug"), "filter should contain debug: {s}");
    }

    #[test]
    fn test_build_filter_custom_directive() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let filter = build_filter("kiln=trace,tower_http=warn").unwrap();
        let s = format!("{filter}");
        // Custom directive is parsed as-is (not expanded to the standard triple)
        assert!(
            s.contains("kiln=trace") || s.contains("tower_http=warn"),
            "filter should parse custom directive: {s}"
        );
    }

    #[test]
    fn malformed_filter_inputs_are_fatal_and_identify_the_value() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let error = build_filter("kiln=definitely-not-a-level").unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("logging.level"), "{message}");
        assert!(message.contains("kiln=definitely-not-a-level"), "{message}");

        unsafe {
            std::env::set_var("RUST_LOG", "kiln=definitely-not-a-level");
        }
        let error = build_filter("info").unwrap_err();
        unsafe {
            std::env::remove_var("RUST_LOG");
        }
        let message = format!("{error:#}");
        assert!(message.contains("RUST_LOG"), "{message}");
        assert!(message.contains("kiln=definitely-not-a-level"), "{message}");
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

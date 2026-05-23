//! `kiln_tensor::Error` / `Result` and the public [`bail!`] / [`ensure!`] macros.
//!
//! These are the migration target for `candle_core::Error` (106 sites),
//! `candle_core::Result` (43 sites), `candle_core::bail!` (491 sites), and
//! `candle_core::ensure!` — together over 640 of the 1,799 candle call sites
//! the Phase 0.1 audit captured.
//!
//! The shape mirrors candle's so the swap is mechanical in most call sites:
//!
//! ```ignore
//! use kiln_tensor as kt;
//!
//! fn check(shape: &[usize]) -> kt::Result<()> {
//!     kt::ensure!(!shape.is_empty(), "shape cannot be empty");
//!     kt::ensure!(shape.iter().all(|&d| d > 0),
//!                 "shape {shape:?} contains zero or negative dims");
//!     Ok(())
//! }
//!
//! fn lookup(name: &str) -> kt::Result<i32> {
//!     match name {
//!         "answer" => Ok(42),
//!         other => kt::bail!("unknown name {other:?}"),
//!     }
//! }
//! ```
//!
//! `Error::Msg` is the analogue of `candle_core::Error::Msg`; the typed
//! variants are scaffolded for Phase 1.2 / 1.3 / 1.4 to populate.

use std::fmt;

/// kiln-tensor's error type.
///
/// Variants are split between:
///
/// - [`Error::Msg`] — free-form string. The migration's mechanical
///   substitute for `candle_core::Error::Msg`.
/// - Typed variants — added in subsequent Phase 1 PRs (mismatched
///   dtype, bad shape, allocator OOM, etc.).
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Free-form error message. Use [`bail!`] / [`ensure!`] to construct.
    #[error("{0}")]
    Msg(String),
}

impl Error {
    /// Construct a [`Error::Msg`] from any `Display` value.
    #[inline]
    pub fn msg<T: fmt::Display>(value: T) -> Self {
        Error::Msg(value.to_string())
    }

    /// Wrap a borrowed [`str`] into a [`Error::Msg`]. Marginally cheaper
    /// than [`Error::msg`] for static-string call sites.
    #[inline]
    pub fn from_str(s: &str) -> Self {
        Error::Msg(s.to_owned())
    }

    /// Attach a static `&'static str` context phrase to an existing
    /// error. Mirrors `anyhow::Context::context`; intended for the
    /// per-call-site context-add idiom common in candle code.
    #[inline]
    pub fn with_context<C: fmt::Display>(self, context: C) -> Self {
        Error::Msg(format!("{context}: {self}"))
    }
}

/// kiln-tensor's [`std::result::Result`] alias.
///
/// Replaces `candle_core::Result<T>` at the 43+ call sites that name it
/// explicitly, plus the implicit returns from `bail!` / `ensure!`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Return early with an [`Error::Msg`].
///
/// Mirrors `candle_core::bail!` ergonomically:
///
/// ```ignore
/// kiln_tensor::bail!("got {} dims, expected at least {}", got, want);
/// ```
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return ::core::result::Result::Err($crate::Error::Msg(::std::format!($msg)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return ::core::result::Result::Err($crate::Error::Msg(::std::format!($fmt, $($arg)*)))
    };
}

/// Return early with an [`Error::Msg`] if a condition is false.
///
/// Mirrors `candle_core::ensure!`:
///
/// ```ignore
/// kiln_tensor::ensure!(x.len() == y.len(), "lengths differ: {} vs {}", x.len(), y.len());
/// ```
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:literal $(,)?) => {
        if !($cond) {
            return ::core::result::Result::Err($crate::Error::Msg(::std::format!($msg)));
        }
    };
    ($cond:expr, $fmt:expr, $($arg:tt)*) => {
        if !($cond) {
            return ::core::result::Result::Err($crate::Error::Msg(::std::format!($fmt, $($arg)*)));
        }
    };
}

// ----------------------------------------------------------------------
// Conversions for ergonomic interop with anyhow / std error types.
// ----------------------------------------------------------------------

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_owned())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Msg(format!("io: {e}"))
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(e: std::num::ParseIntError) -> Self {
        Error::Msg(format!("parse int: {e}"))
    }
}

impl From<std::num::ParseFloatError> for Error {
    fn from(e: std::num::ParseFloatError) -> Self {
        Error::Msg(format!("parse float: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn returns_bail() -> Result<()> {
        bail!("bail with format: {} + {}", 1, 2);
    }

    fn returns_ensure(cond: bool) -> Result<()> {
        ensure!(cond, "ensure says no");
        Ok(())
    }

    #[test]
    fn bail_returns_msg() {
        let e = returns_bail().unwrap_err();
        match e {
            Error::Msg(m) => assert_eq!(m, "bail with format: 1 + 2"),
        }
    }

    #[test]
    fn ensure_truthy_is_ok() {
        assert!(returns_ensure(true).is_ok());
    }

    #[test]
    fn ensure_falsy_returns_msg() {
        let e = returns_ensure(false).unwrap_err();
        match e {
            Error::Msg(m) => assert_eq!(m, "ensure says no"),
        }
    }

    #[test]
    fn with_context_prefixes() {
        let e = Error::from_str("inner cause").with_context("while reading model.safetensors");
        assert_eq!(
            e.to_string(),
            "while reading model.safetensors: inner cause"
        );
    }

    #[test]
    fn from_str_and_string_construct_msg() {
        let a: Error = "literal".into();
        let b: Error = String::from("owned").into();
        match (a, b) {
            (Error::Msg(x), Error::Msg(y)) => {
                assert_eq!(x, "literal");
                assert_eq!(y, "owned");
            }
        }
    }

    #[test]
    fn io_error_wraps_into_msg() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let e: Error = io_err.into();
        match e {
            Error::Msg(m) => assert!(m.starts_with("io: ")),
        }
    }
}

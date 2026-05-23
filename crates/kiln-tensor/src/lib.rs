//! kiln-tensor — in-house Tensor + Storage substrate for kiln.
//!
//! # Status
//!
//! **Phase 1.1 scaffold** — see [GitHub issue #1082][epic] for the full
//! migration plan. Today this crate ships only [`Error`], [`Result`],
//! and the [`bail!`] macro. Subsequent Phase 1 PRs add `DType`,
//! `TensorId`, `Layout`, `Storage`, and `Tensor`.
//!
//! # Public-API stability
//!
//! `kiln-tensor` is **internal-only — no semver commitment**. Other
//! `kiln-*` crates may break against minor version bumps until candle is
//! removed and the surface stabilizes (Phase 7 of #1082). Once Phase 7
//! lands and the migration is complete, the public API of this crate
//! freezes against the next major version bump.
//!
//! [epic]: https://github.com/ericflo/kiln/issues/1082

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod dtype;
mod error;
mod layout;
mod tensor_id;

pub use dtype::DType;
pub use error::{Error, Result};
pub use layout::Layout;
pub use tensor_id::TensorId;

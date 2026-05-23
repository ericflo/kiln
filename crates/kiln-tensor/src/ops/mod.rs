//! Concrete `DeviceOp` implementations.
//!
//! Each submodule ships one logical op family and provides:
//!
//! - A struct implementing one of `DeviceOp1` / `DeviceOp2` / `DeviceOp3`.
//! - A `fn op_name(&Tensor, ...) -> Result<Tensor>` convenience that
//!   wraps `dispatch{1,2,3}` so call sites read like regular functions.
//! - A CPU forward (the canonical reference; backend-fallthrough hook).
//! - GPU forwards default to `Ok(None)` until the per-backend impl
//!   lands in Phase 2+.
//! - Parity tests in `#[cfg(test)]` that exercise the public API.
//!
//! Today: `embedding`. Phase 4 fills out the rest of the "glue" ops
//! (RMSNorm, residual add, final norm + LM head, sampling).

pub mod activation;
pub mod elementwise;
pub mod embedding;
pub mod rmsnorm;
pub mod softmax;

pub use activation::{sigmoid, silu, ActivationOp, UnaryKind};
pub use elementwise::{add, div, mul, sub, BinaryKind, ElementwiseOp};
pub use embedding::{embedding, EmbeddingOp};
pub use rmsnorm::{rms_norm, RmsNormOp};
pub use softmax::{softmax_last_dim, SoftmaxLastDimOp};

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
pub mod argmax;
pub mod cast;
pub mod cross_entropy;
pub mod elementwise;
pub mod embedding;
pub mod index_select;
pub mod l2norm;
pub mod logit_dry;
pub mod logit_mirostat;
pub mod logit_misc;
pub mod logit_modern;
pub mod logit_penalties;
pub mod logit_processor;
pub mod logit_xtc;
pub mod mask;
pub mod matmul;
pub mod reduce;
pub mod rmsnorm;
pub mod rope;
pub mod scatter_add;
pub mod silu_mul;
pub mod softmax;

pub use activation::{sigmoid, silu, ActivationOp, UnaryKind};
pub use argmax::{argmax_last_dim, ArgmaxLastDimOp};
pub use cast::{cast, CastOp};
pub use cross_entropy::{cross_entropy, CrossEntropyOp};
pub use elementwise::{add, div, mul, sub, BinaryKind, ElementwiseOp};
pub use embedding::{embedding, EmbeddingOp};
pub use index_select::{index_select, IndexSelectOp};
pub use l2norm::{l2_norm, L2NormOp};
pub use mask::{causal_mask, masked_fill, MaskedFillOp};
pub use matmul::{matmul, MatmulOp};
pub use reduce::{mean_all, mean_axis, sum_all, sum_axis, ReduceOp, ReductionKind, ReductionScope};
pub use rmsnorm::{rms_norm, RmsNormOp};
pub use rope::{rope, RopeOp};
pub use scatter_add::{scatter_add, ScatterAddOp};
pub use silu_mul::{mul_sigmoid_gate, MulSigmoidGateOp};
pub use softmax::{softmax_last_dim, SoftmaxLastDimOp};

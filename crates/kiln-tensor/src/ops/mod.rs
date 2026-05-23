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
pub mod broadcast;
pub mod cast;
pub mod clamp_pow;
pub mod concat;
pub mod cross_entropy;
pub mod cumsum;
pub mod dropout;
pub mod elementwise;
pub mod embedding;
pub mod gumbel_sample;
pub mod hyperbolic;
pub mod index_select;
pub mod l2norm;
pub mod layernorm;
pub mod log_variants;
pub mod logit_dry;
pub mod logit_mirostat;
pub mod logit_misc;
pub mod logit_modern;
pub mod logit_penalties;
pub mod logit_processor;
pub mod logit_xtc;
pub mod mask;
pub mod matmul;
pub mod max_min_axis;
pub mod one_hot;
pub mod range_ctors;
pub mod reduce;
pub mod repeat;
pub mod rmsnorm;
pub mod rope;
pub mod scatter_add;
pub mod sign_and_round;
pub mod silu_mul;
pub mod softmax;
pub mod stack;
pub mod top_k;
pub mod trig;
pub mod unary_arith;
pub mod where_select;

pub use activation::{gelu, relu, sigmoid, silu, tanh, ActivationOp, UnaryKind};
pub use argmax::{argmax_last_dim, ArgmaxLastDimOp};
pub use broadcast::broadcast_to;
pub use cast::{cast, CastOp};
pub use clamp_pow::{clamp, pow};
pub use concat::{concat, ConcatOp};
pub use cross_entropy::{cross_entropy, CrossEntropyOp};
pub use cumsum::cumsum;
pub use dropout::dropout;
pub use elementwise::{add, div, mul, sub, BinaryKind, ElementwiseOp};
pub use embedding::{embedding, EmbeddingOp};
pub use gumbel_sample::GumbelSampler;
pub use hyperbolic::{cosh, sinh};
pub use index_select::{index_select, IndexSelectOp};
pub use l2norm::{l2_norm, L2NormOp};
pub use layernorm::{layer_norm, LayerNormOp};
pub use log_variants::{exp2, expm1, log10, log1p, log2};
pub use mask::{causal_mask, masked_fill, MaskedFillOp};
pub use matmul::{matmul, MatmulOp};
pub use max_min_axis::{max_axis, min_axis, MinMaxKind};
pub use one_hot::one_hot;
pub use range_ctors::{arange, linspace};
pub use reduce::{mean_all, mean_axis, sum_all, sum_axis, ReduceOp, ReductionKind, ReductionScope};
pub use repeat::repeat;
pub use rmsnorm::{rms_norm, RmsNormOp};
pub use rope::{rope, RopeOp};
pub use scatter_add::{scatter_add, ScatterAddOp};
pub use sign_and_round::{ceil, floor, reciprocal, round, sign, trunc};
pub use silu_mul::{mul_sigmoid_gate, MulSigmoidGateOp};
pub use softmax::{softmax_last_dim, SoftmaxLastDimOp};
pub use stack::stack;
pub use top_k::top_k;
pub use trig::{acos, asin, atan, cos, sin, tan, TrigKind};
pub use unary_arith::{abs, exp, ln, neg, sqrt, UnaryArithKind};
pub use where_select::where_select;

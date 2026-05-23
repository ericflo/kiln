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
pub mod binary_minmax;
pub mod bool_reduce;
pub mod broadcast;
pub mod cast;
pub mod categorical;
pub mod clamp_pow;
pub mod compare;
pub mod concat;
pub mod cosine_sim;
pub mod cross_entropy;
pub mod cumsum;
pub mod diag;
pub mod dot;
pub mod dropout;
pub mod elementwise;
pub mod embedding;
pub mod eye;
pub mod grad_clip;
pub mod glu;
pub mod gumbel_sample;
pub mod hyperbolic;
pub mod index_select;
pub mod init;
pub mod l2norm;
pub mod layernorm;
pub mod leaky_activations;
pub mod like;
pub mod linear;
pub mod log_softmax;
pub mod log_variants;
pub mod logit_dry;
pub mod logit_mirostat;
pub mod logit_misc;
pub mod logit_modern;
pub mod logit_penalties;
pub mod logit_processor;
pub mod logit_xtc;
pub mod losses;
pub mod mask;
pub mod matmul;
pub mod max_min_axis;
pub mod mha;
pub mod norms;
pub mod one_hot;
pub mod outer;
pub mod precision;
pub mod random;
pub mod range_ctors;
pub mod reduce;
pub mod repeat;
pub mod rmsnorm;
pub mod rope;
pub mod rope_init;
pub mod scalar;
pub mod scatter_add;
pub mod sdpa;
pub mod sign_and_round;
pub mod silu_mul;
pub mod softmax;
pub mod stack;
pub mod top_k;
pub mod trace;
pub mod triangular;
pub mod trig;
pub mod unary_arith;
pub mod where_select;

pub use activation::{gelu, relu, sigmoid, silu, tanh, ActivationOp, UnaryKind};
pub use argmax::{argmax_last_dim, ArgmaxLastDimOp};
pub use binary_minmax::{maximum, minimum};
pub use bool_reduce::{all_axis, all_reduce, any_axis, any_reduce, BoolReduce};
pub use broadcast::broadcast_to;
pub use cast::{cast, CastOp};
pub use categorical::Multinomial;
pub use clamp_pow::{clamp, pow};
pub use compare::{eq, ge, gt, le, lt, ne, CmpKind};
pub use concat::{concat, ConcatOp};
pub use cosine_sim::cosine_similarity;
pub use cross_entropy::{cross_entropy, CrossEntropyOp};
pub use cumsum::cumsum;
pub use diag::{diag, diagonal};
pub use dot::dot;
pub use dropout::dropout;
pub use elementwise::{add, div, mul, sub, BinaryKind, ElementwiseOp};
pub use embedding::{embedding, EmbeddingOp};
pub use eye::eye;
pub use glu::{geglu, glu, reglu, swiglu};
pub use grad_clip::clip_grad_norm;
pub use gumbel_sample::GumbelSampler;
pub use hyperbolic::{cosh, sinh};
pub use index_select::{index_select, IndexSelectOp};
pub use init::{kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform};
pub use l2norm::{l2_norm, L2NormOp};
pub use layernorm::{layer_norm, LayerNormOp};
pub use leaky_activations::{elu, leaky_relu, mish, softplus};
pub use like::{full_like, ones_like, zeros_like};
pub use linear::linear;
pub use log_softmax::log_softmax_last_dim;
pub use log_variants::{exp2, expm1, log10, log1p, log2};
pub use losses::{bce_with_logits, huber_loss, kl_div_log_probs, l1_loss, mse_loss, nll_loss};
pub use mask::{causal_mask, masked_fill, MaskedFillOp};
pub use matmul::{matmul, MatmulOp};
pub use max_min_axis::{max_axis, min_axis, MinMaxKind};
pub use mha::multi_head_attention;
pub use norms::{frobenius_norm, mean_squared, vector_norm};
pub use one_hot::one_hot;
pub use outer::outer;
pub use precision::{to_bf16, to_f16, to_f32};
pub use random::{rand_normal, rand_uniform};
pub use range_ctors::{arange, linspace};
pub use reduce::{mean_all, mean_axis, sum_all, sum_axis, ReduceOp, ReductionKind, ReductionScope};
pub use repeat::repeat;
pub use rmsnorm::{rms_norm, RmsNormOp};
pub use rope::{rope, RopeOp};
pub use rope_init::precompute_rope_freqs;
pub use scalar::{add_scalar, div_scalar, mul_scalar, sub_scalar};
pub use scatter_add::{scatter_add, ScatterAddOp};
pub use sdpa::{causal_scaled_dot_product_attention, scaled_dot_product_attention};
pub use sign_and_round::{ceil, floor, reciprocal, round, sign, trunc};
pub use silu_mul::{mul_sigmoid_gate, MulSigmoidGateOp};
pub use softmax::{softmax_last_dim, SoftmaxLastDimOp};
pub use stack::stack;
pub use top_k::top_k;
pub use trace::trace;
pub use triangular::{tril, tril_mask, triu, triu_mask};
pub use trig::{acos, asin, atan, cos, sin, tan, TrigKind};
pub use unary_arith::{abs, exp, ln, neg, sqrt, UnaryArithKind};
pub use where_select::where_select;

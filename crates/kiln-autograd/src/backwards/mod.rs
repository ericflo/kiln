//! Concrete `BackwardOp` implementations for the core differentiable
//! ops in `kiln_tensor::ops::*`.
//!
//! Each module ships one logical op's backward. The forward op records
//! a `Box<dyn BackwardOp>` into the tape carrying whatever saved
//! tensors are needed (e.g. `a` and `b` for matmul; only the shapes
//! for add/sub).
//!
//! All apply paths route through the kiln-tensor op surface, so each
//! backward is exercised in CPU parity tests against finite-difference
//! reference values.

pub mod activation;
pub mod broadcast;
pub mod clamp_pow;
pub mod concat;
pub mod cross_entropy;
pub mod cross_entropy_kt;
pub mod cumsum;
pub mod dropout;
pub mod elementwise;
pub mod embedding;
pub mod gather;
pub mod hyperbolic;
pub mod index_ops;
pub mod inject_gradient;
pub mod l2norm;
pub mod layernorm;
pub mod leaky_activations;
pub mod lerp;
pub mod log_variants;
pub mod lora_delta_add;
pub mod mask;
pub mod matmul;
pub mod max_axis;
pub mod maximum;
pub mod narrow;
pub mod precision;
pub mod reduce;
pub mod repeat;
pub mod rmsnorm;
pub mod rope;
pub mod rope_split_half;
pub mod stack;
pub mod stride;
pub mod swiglu;
pub mod trig;
pub mod unary_arith;
pub mod unsqueeze;
pub mod where_select;

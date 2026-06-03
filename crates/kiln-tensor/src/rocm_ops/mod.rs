//! ROCm op wrappers (Phase R.5) — one module per `csrc` kernel family, each
//! wrapping the hipcc-compiled `extern "C"` kernels behind a kt-Tensor API and
//! mirroring the corresponding `cuda_*` ops. Reduction kernels route through the
//! wave-size-agnostic `kiln_block_reduce_*` primitive (see `kt_gpu_compat.cuh`),
//! validated on real wave64. Each submodule re-exports its `rocm_*` functions.

pub mod activation;
pub mod argmax_last_axis;
pub mod binary_minmax;
pub mod cast;
pub mod clamp_pow;
pub mod compare;
pub mod concat;
pub mod cross_entropy;
pub mod diag;
pub mod dropout;
pub mod elementwise;
pub mod index_select;
pub mod is_finite_reduce;
pub mod layernorm;
pub mod lerp;
pub mod masked_fill;
pub mod reduce_arbitrary_axis;
pub mod reduce_last_axis;
pub mod rmsnorm;
pub mod rope;
pub mod scalar_op;
pub mod scan_axis;
pub mod scatter_add;
pub mod where_select;

pub use activation::*;
pub use argmax_last_axis::*;
pub use binary_minmax::*;
pub use cast::*;
pub use clamp_pow::*;
pub use compare::*;
pub use concat::*;
pub use cross_entropy::*;
pub use diag::*;
pub use dropout::*;
pub use elementwise::*;
pub use index_select::*;
pub use is_finite_reduce::*;
pub use layernorm::*;
pub use lerp::*;
pub use masked_fill::*;
pub use reduce_arbitrary_axis::*;
pub use reduce_last_axis::*;
pub use rmsnorm::*;
pub use rope::*;
pub use scalar_op::*;
pub use scan_axis::*;
pub use scatter_add::*;
pub use where_select::*;

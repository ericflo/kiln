//! kiln-autograd — tape-based reverse-mode autograd over kiln-tensor.
//!
//! Phase 6a of #1082 lifts `crates/kiln-vulkan-kernel/src/vk_autograd.rs`
//! (173 LOC) into a backend-generic substrate keyed on
//! `kiln_tensor::TensorId`.
//!
//! # Design
//!
//! **Eager-tape, PyTorch-style.** Each forward op records a tape node
//! `{output_id, op, input_ids}` via [`Tape::record`]. At backward time,
//! [`Tape::backward`] walks the tape in reverse topo order, accumulating
//! gradients in a [`GradStore`] keyed on `TensorId`.
//!
//! Tape state is **external** to `kiln_tensor::Tensor` (per anti-pattern 1
//! — no autograd fields on the Tensor type). A thread-local or
//! explicitly-passed `Tape` handle captures the graph.
//!
//! # Anti-pattern 16 hook
//!
//! Per the issue:
//!
//! > **In-place mutation invalidates the tape.** Any in-place op
//! > (optimizer step, residual accumulate-in-place, in-place norm)
//! > bumps a per-tensor version counter; the backward path asserts
//! > the version is unchanged from when the tape recorded the
//! > forward. Failing the assertion is a programming error, not a
//! > tolerated mode — the tape was holding a stale view.
//!
//! [`TapeNode`] carries the input-tensor version counters at record
//! time. [`Tape::backward`] re-reads each input's current version and
//! asserts equality before calling the op's `bwd`. Phase 1.x doesn't
//! ship the version counter yet (Tensor is immutable so far); this
//! crate records `0` for the version today and the assertion is a
//! no-op. When in-place ops land (optimizer step, residual fuse), the
//! Tensor version field plus this assertion together enforce the
//! invariant.
//!
//! # Selective-recompute hook
//!
//! Phase 6.5's policy reads
//! [`kiln_tensor::selective_recompute_recommendation`] for each
//! activation referenced by a [`TapeNode`]. Today's `Tape::backward`
//! always saves; the recompute path lands in Phase 6.5.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod backward_op;
pub mod backwards;
mod grad_store;
mod tape;

pub use backward_op::{BackwardOp, BoxedBackwardOp};
pub use backwards::activation::{SigmoidBackward, SiluBackward, SoftmaxLastDimBackward};
pub use backwards::cross_entropy::CrossEntropyBackward;
pub use backwards::elementwise::{AddBackward, DivBackward, MulBackward, SubBackward};
pub use backwards::embedding::EmbeddingBackward;
pub use backwards::matmul::MatmulBackward;
pub use backwards::rmsnorm::RmsNormBackward;
pub use grad_store::GradStore;
pub use tape::{Tape, TapeNode};

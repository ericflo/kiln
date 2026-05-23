//! Full-surface `BackwardOp` trait — successor to the Phase 1.12
//! scaffold (`kiln_tensor::BackwardOp`).
//!
//! Per the Phase 6a issue bullet:
//!
//! > `BackwardOp` trait moved to `kiln-autograd`; `VkBackwardOp` becomes
//! > a re-export
//!
//! # Method surface (full)
//!
//! - `name() -> &'static str` — stable label (NVTX + parity-tolerance.csv key)
//! - `apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>`
//!   — produces one gradient per input, in the order the forward op
//!   consumed them. `None` for non-differentiable inputs (token ids,
//!   indices).
//! - `input_count(&self) -> usize` — declared arity; checked by the
//!   tape walker.
//!
//! Optional method:
//!
//! - `requires_input(&self, idx: usize) -> bool` — declares which inputs
//!   the backward needs to read (the rest can be released after forward,
//!   informing Phase 6.5's selective-recompute policy).

use kiln_tensor::{Result, Tensor};

/// One backward closure produced by a forward op. Boxed and stored in
/// the tape; called from [`crate::Tape::backward`] in reverse topo
/// order.
pub trait BackwardOp: Send + Sync + std::fmt::Debug {
    /// Stable name. Matches the forward op's `name()`.
    fn name(&self) -> &'static str;

    /// Arity — how many inputs the forward op consumed.
    fn input_count(&self) -> usize;

    /// Compute gradients with respect to each input, given the
    /// gradient with respect to the output.
    ///
    /// Returns a `Vec` of length `input_count()`:
    /// - `Some(grad)` — gradient w.r.t. the i-th input.
    /// - `None` — input is non-differentiable (token ids, scalar
    ///   indices, masks). The tape walker silently skips these.
    ///
    /// `apply` MUST NOT mutate the input Tensors. In-place mutation
    /// is detected by [`crate::Tape::backward`] via the per-tensor
    /// version counter (anti-pattern 16) — a violation panics with
    /// the op's `name()` and tape position.
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;

    /// Declare whether the `idx`-th forward input is read by this
    /// backward. Default: assume yes. Phase 6.5's selective-recompute
    /// policy reads this to decide whether to keep the input
    /// activation around.
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

/// Owning handle to a [`BackwardOp`]. The tape stores these.
pub type BoxedBackwardOp = Box<dyn BackwardOp>;

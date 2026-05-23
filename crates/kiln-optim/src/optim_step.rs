//! `OptimStep` — the unified trait every optimizer variant implements.
//!
//! Per the Phase 6.5 issue bullet:
//!
//! > `kiln-optim` crate with `OptimStep` trait: AdamW, SGD, Lion, Muon
//!
//! The trait operates on `kiln_param::Parameter` so that the AMP
//! policy (per-parameter dtype tuple), forward/backward storages,
//! and stable `TensorId` are all visible without auxiliary lookups.

use kiln_param::Parameter;
use kiln_tensor::Tensor;

/// One optimizer step over a [`Parameter`] with its accumulated grad.
///
/// Implementations:
/// - [`crate::AdamW`] — CPU reference (this PR).
/// - `Sgd`, `Lion`, `Muon` — subsequent PRs.
/// - Per-backend GPU impls plug into the same trait.
pub trait OptimStep: Send + Sync + std::fmt::Debug {
    /// Stable name. Used by NVTX labels + checkpoint receipts.
    fn name(&self) -> &'static str;

    /// Apply one optimizer step.
    ///
    /// `grad` is the accumulated gradient w.r.t. `param`. Implementations:
    /// - read `param.amp_policy()` for dispatch dtype decisions
    /// - update `param.backward_storage_mut()` (the master copy) in
    ///   place
    /// - on backends with quantized forward (Marlin / FP8), schedule
    ///   a re-quantization pass before returning so `param.forward()`
    ///   in the next iteration sees a fresh forward_storage. Phase
    ///   6.5.x ships the actual re-quant fusion.
    fn step(&mut self, param: &mut Parameter, grad: &Tensor) -> Result<(), StepError>;

    /// Reset internal state (moments, momentum buffers). Called when
    /// the training loop restarts or rolls back to a checkpoint.
    fn reset(&mut self);
}

/// Errors a step may raise. Wraps [`kiln_tensor::Error`] plus
/// optimizer-specific cases.
#[derive(thiserror::Error, Debug)]
pub enum StepError {
    /// Backward storage missing on an inference-only Parameter.
    #[error("OptimStep: parameter has no backward_storage (inference-only) — cannot step")]
    NoBackwardStorage,

    /// Gradient shape doesn't match master shape.
    #[error(
        "OptimStep: grad shape {grad_shape:?} != master shape {master_shape:?}"
    )]
    GradShapeMismatch {
        grad_shape: Vec<usize>,
        master_shape: Vec<usize>,
    },

    /// Gradient dtype doesn't match the policy's `backward_compute_dtype`.
    #[error(
        "OptimStep: grad dtype {grad_dtype} != policy backward_compute_dtype \
         {policy_dtype} for parameter"
    )]
    GradDtypeMismatch {
        grad_dtype: kiln_tensor::DType,
        policy_dtype: kiln_tensor::DType,
    },

    /// Numerical anomaly (NaN / Inf) detected in grad. Raised under
    /// `KILN_DETECT_ANOMALY=1` (Phase 9 deliverable).
    #[error("OptimStep: non-finite values in grad at parameter {tensor_id:?}")]
    NonFiniteGrad { tensor_id: kiln_tensor::TensorId },

    /// Underlying kiln_tensor op failed.
    #[error("OptimStep: tensor op: {0}")]
    Tensor(#[from] kiln_tensor::Error),
}

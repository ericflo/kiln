//! `KtForwardOp` — generic candle [`CustomOp`] shim for production-caller
//! migrations from candle-typed kernel entry points to kt-typed kernels
//! ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # The problem this shim solves
//!
//! After PR #341da876 (`RmsNormCustomOp::bwd`), PR #d99a15a3
//! (`CudaRotaryOneBf16::bwd`), PR #0c1be227 (`OpdLossCustomOp::bwd`), and
//! PR #ab2da23f (`FlceCustomOp::bwd`), four production CustomOp backward
//! bodies have been migrated to the kt bridge. Each one is hand-written:
//!
//! 1. Borrow each candle input tensor as a kt tensor (zero-copy via
//!    [`crate::kt_tensor_from_candle_cuda_borrow`]).
//! 2. Call a kt-typed kernel that returns kt tensor(s).
//! 3. Copy each kt output tensor back to candle storage via
//!    [`crate::kt_tensor_to_candle_cuda_copy`].
//!
//! These bodies live inside existing `CustomOp::bwd` impls in
//! `kiln-rmsnorm-kernel`, `kiln-model::forward`, `kiln-opd-loss-kernel`,
//! and `kiln-flce-kernel`. The pattern is the same; only the kernel
//! function and the arity (which inputs are inputs, which are saved
//! op state) differs.
//!
//! The production callers in `kiln-train` (e.g. `src/opd.rs:68` which
//! imports `opd_top_k_reverse_kl_phase_a_per_position`) still go
//! through the **candle-typed** kernel entry, because:
//!   - kt-typed entries return `kiln_tensor::Tensor`, which has no
//!     autograd (no `.backward()` chain through `Tape`-or-equivalent).
//!   - The CustomOp wrappers (`OpdLossCustomOp`, etc.) provide candle
//!     autograd integration but couple a forward and backward into one
//!     concrete object.
//!
//! `KtForwardOp` is the generic, reusable substitute: a candle
//! [`candle_core::CustomOp1`] / [`candle_core::CustomOp2`] /
//! [`candle_core::CustomOp3`] **parameterized over closures**, so every
//! production caller migration is a per-site mechanical transformation
//! (~50-80 LOC of forward-closure + backward-closure definition,
//! ~3 lines at the call site).
//!
//! # Design contract
//!
//! - **CUDA-only fast path**. The CPU/Metal forward paths return
//!   [`candle_core::Error::BackwardNotSupported`] (or an explicit
//!   "no implementation" error). The shim is built for production
//!   training loops that run on GPU; the candle-typed CustomOp
//!   wrappers remain the parity oracle for non-CUDA paths.
//! - **Saved state lives in the closures (or the shim instance)**.
//!   The forward closure receives candle inputs and returns a candle
//!   output. The backward closure receives the same inputs plus the
//!   upstream gradient and returns one gradient tensor per input
//!   (or `None` for inputs that don't receive a gradient).
//! - **No assumption about input/output arity or saved tensors
//!   beyond the kt round-trip.** The closures are responsible for
//!   force-contiguous, dtype cast, borrow, kt-call, copy-back — the
//!   shim's job is the CustomOp glue.
//! - **Bridge failures bubble up via `candle_core::Error::Msg`.**
//!   Callers can wrap with a kill switch + fallback if they want
//!   "fall back to candle on bridge failure" semantics (same pattern
//!   as the existing 4 migrations).
//!
//! # Why closures, not a trait
//!
//! Two reasons.
//!
//! 1. **Each kernel returns different kt types**. RmsNorm returns
//!    `(grad_x, grad_w)`; OPD returns `dhidden`; FLCE returns
//!    `dhidden`; rotary_one returns `grad_x`. A trait would have to
//!    pin a fixed associated type and re-introduce per-kernel
//!    glue at trait impl time. Closures sidestep that.
//! 2. **Saved-tensor shapes are kernel-specific**. RmsNorm saves
//!    `(x, weight)`; OPD saves `(hidden, head_t, indices, ...)`;
//!    FLCE saves `(hidden, head_t, input_ids, label_mask)`.
//!    A trait would need an opaque `Saved` associated type, plus
//!    boilerplate to thread it through. With closures, captured
//!    variables in the closure are the saved state — no extra
//!    machinery.
//!
//! # Arity
//!
//! This module provides single-input ([`KtForwardOp1`]), binary
//! ([`KtForwardOp2`]), and ternary ([`KtForwardOp3`]) variants —
//! matching candle's [`candle_core::CustomOp1`] /
//! [`candle_core::CustomOp2`] / [`candle_core::CustomOp3`] traits.
//! Adding higher arity would follow the same pattern; the existing
//! migrations cover unary (RmsNorm, OPD, FLCE — all wrap a single
//! `hidden`) and ternary (rotary_one — `(x, cos, sin)`).
//!
//! # Example (single-input)
//!
//! ```ignore
//! use kiln_kt_bridge::forward_op::KtForwardOp1;
//! use kiln_kt_bridge::{kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy};
//! use candle_core::{Result, Tensor};
//! use std::sync::Arc;
//!
//! // Forward: y = scale * x  via a kt-typed scalar mul.
//! // Backward: dx = scale * dy.
//! let scale = 2.0f32;
//! let op = KtForwardOp1::new(
//!     "demo-scalar-mul",
//!     move |x: &Tensor| -> Result<Tensor> {
//!         let x_kt = kt_tensor_from_candle_cuda_borrow(x)
//!             .map_err(|e| candle_core::Error::Msg(format!("borrow x: {e}")))?;
//!         let y_kt = my_kt_kernel_scale(&x_kt, scale)
//!             .map_err(|e| candle_core::Error::Msg(format!("kt call: {e}")))?;
//!         kt_tensor_to_candle_cuda_copy(&y_kt)
//!             .map_err(|e| candle_core::Error::Msg(format!("copy-back: {e}")))
//!     },
//!     move |_x: &Tensor, _y: &Tensor, grad_y: &Tensor| -> Result<Tensor> {
//!         // dx = scale * grad_y — same kt-bridge round-trip as forward.
//!         let g_kt = kt_tensor_from_candle_cuda_borrow(grad_y)
//!             .map_err(|e| candle_core::Error::Msg(format!("borrow gy: {e}")))?;
//!         let dx_kt = my_kt_kernel_scale(&g_kt, scale)
//!             .map_err(|e| candle_core::Error::Msg(format!("kt bwd: {e}")))?;
//!         kt_tensor_to_candle_cuda_copy(&dx_kt)
//!             .map_err(|e| candle_core::Error::Msg(format!("copy-back: {e}")))
//!     },
//! );
//! let y = x.apply_op1_arc(Arc::new(Box::new(op)))?;
//! ```

use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result as CandleResult,
    Shape, Tensor,
};

// kiln-kt-bridge unconditionally enables candle-core's `cuda` feature
// (see Cargo.toml — there is no non-CUDA build of this crate), so the
// CUDA-only `cuda_fwd` paths below do not need feature gates. The
// crate's `metal_fwd` defaults from the trait remain in place (returning
// "no Metal implementation" — matching the documented CUDA-only contract
// of the kt bridge surface).

// ----------------------------------------------------------------------
// Closure type aliases
// ----------------------------------------------------------------------
//
// Forward closures take borrowed candle `Tensor`s and return a candle
// `Tensor`. The shim drives this via the CustomOp{1,2,3} `cuda_fwd`
// hook by constructing leaf candle `Tensor`s from the supplied
// `(CudaStorage, Layout)` pairs (zero-copy via `try_clone`) and then
// invoking the closure. The output `Tensor`'s storage is unwrapped
// back into a `(CudaStorage, Shape)` pair to return through the
// CustomOp interface.
//
// Backward closures take borrowed candle input `Tensor`s + result +
// upstream gradient, and return one gradient `Tensor` per input.
// Returning `None` signals "no grad for this input" (matching candle's
// CustomOp{1,2,3}::bwd return convention).

/// Forward closure for [`KtForwardOp1`].
pub type Fwd1Closure = dyn Fn(&Tensor) -> CandleResult<Tensor> + Send + Sync + 'static;

/// Backward closure for [`KtForwardOp1`].
///
/// Receives `(arg, res, grad_res)` and returns `Option<grad_arg>`,
/// matching candle's [`CustomOp1::bwd`] signature.
pub type Bwd1Closure =
    dyn Fn(&Tensor, &Tensor, &Tensor) -> CandleResult<Option<Tensor>> + Send + Sync + 'static;

/// Forward closure for [`KtForwardOp2`].
pub type Fwd2Closure = dyn Fn(&Tensor, &Tensor) -> CandleResult<Tensor> + Send + Sync + 'static;

/// Backward closure for [`KtForwardOp2`].
///
/// Receives `(arg1, arg2, res, grad_res)` and returns
/// `(Option<grad_arg1>, Option<grad_arg2>)`.
pub type Bwd2Closure = dyn Fn(
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)>
    + Send
    + Sync
    + 'static;

/// Forward closure for [`KtForwardOp3`].
pub type Fwd3Closure =
    dyn Fn(&Tensor, &Tensor, &Tensor) -> CandleResult<Tensor> + Send + Sync + 'static;

/// Backward closure for [`KtForwardOp3`].
///
/// Receives `(arg1, arg2, arg3, res, grad_res)` and returns
/// `(Option<grad_arg1>, Option<grad_arg2>, Option<grad_arg3>)`.
pub type Bwd3Closure = dyn Fn(
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
    )
        -> CandleResult<(Option<Tensor>, Option<Tensor>, Option<Tensor>)>
    + Send
    + Sync
    + 'static;

// ----------------------------------------------------------------------
// Shared CudaStorage → output unwrap helper
// ----------------------------------------------------------------------

/// Unwrap a candle output `Tensor` back to a `(CudaStorage, Shape)`
/// pair so it can be returned through the CustomOp interface.
///
/// The CustomOp `*_fwd` hooks receive `(&CudaStorage, &Layout)` pairs
/// and must return a `(CudaStorage, Shape)` pair. But our forward
/// closures return a candle `Tensor` (which is what the production
/// callers want). The unwrap is a `try_clone` of the storage so the
/// returned `CudaStorage` is independent of the candle `Tensor`'s
/// arc-shared storage (the latter goes out of scope when the closure
/// returns).
fn cuda_output_to_storage_shape(out: Tensor) -> CandleResult<(CudaStorage, Shape)> {
    use candle_core::backend::BackendStorage;
    let shape = Shape::from(out.dims().to_vec());
    let (storage_guard, layout) = out.storage_and_layout();
    let cuda = match &*storage_guard {
        candle_core::Storage::Cuda(c) => c,
        _ => {
            return Err(candle_core::Error::Msg(
                "KtForwardOp: closure produced non-CUDA storage; expected CUDA".to_string(),
            ));
        }
    };
    let cloned = cuda.try_clone(layout)?;
    Ok((cloned, shape))
}

/// Same as [`cuda_output_to_storage_shape`] but reads a leaf candle
/// CUDA `Tensor` from a borrowed `(&CudaStorage, &Layout)` pair so the
/// forward closure can be invoked with proper `Tensor`s.
///
/// SAFETY/COST: `try_clone` is a single device-to-device copy on CUDA
/// (`Clone.map(...)` in candle's CUDA backend). For zero-copy borrow we
/// would need to wrap the `CudaStorage` in a `Tensor` without cloning,
/// but candle's `Tensor::from_storage` takes ownership of `Storage`.
/// This single dtod copy is the cost of the shim's generic-ness; it
/// is bounded by `numel * dtype_size_in_bytes` per call.
fn cuda_input_to_leaf_tensor(s: &CudaStorage, l: &Layout) -> CandleResult<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::{op::BackpropOp, Storage};
    let cloned = s.try_clone(l)?;
    let shape = Shape::from(l.dims().to_vec());
    // Build a leaf tensor (no backprop op attached — the autograd
    // graph extension happens at apply_op{1,2,3} time, not here).
    Ok(Tensor::from_storage(
        Storage::Cuda(cloned),
        shape,
        BackpropOp::none(),
        false,
    ))
}

// ----------------------------------------------------------------------
// KtForwardOp1 — single-input shim
// ----------------------------------------------------------------------

/// Generic candle [`CustomOp1`] shim that drives a kt-typed kernel
/// forward + backward via closures.
///
/// See module-level docs for the full design rationale. Pass
/// [`Box<dyn Fwd1Closure>`] / [`Box<dyn Bwd1Closure>`] to [`Self::new`]
/// and use [`Tensor::apply_op1_arc`] to register the op with candle's
/// autograd.
pub struct KtForwardOp1 {
    name: &'static str,
    forward: Box<Fwd1Closure>,
    backward: Box<Bwd1Closure>,
}

impl KtForwardOp1 {
    /// Construct a new shim with the given forward and backward
    /// closures. The closures are stored as `Box<dyn ... + Send + Sync>`
    /// so the shim can be wrapped in `Arc<Box<dyn CustomOp1 + Send +
    /// Sync>>` for [`Tensor::apply_op1_arc`].
    pub fn new<F, B>(name: &'static str, forward: F, backward: B) -> Self
    where
        F: Fn(&Tensor) -> CandleResult<Tensor> + Send + Sync + 'static,
        B: Fn(&Tensor, &Tensor, &Tensor) -> CandleResult<Option<Tensor>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name,
            forward: Box::new(forward),
            backward: Box::new(backward),
        }
    }
}

impl CustomOp1 for KtForwardOp1 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(format!(
            "KtForwardOp1 '{}': no CPU implementation (kt-bridge is CUDA-only); \
             callers should provide a candle-path fallback for CPU inputs",
            self.name
        )))
    }

    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> CandleResult<(CudaStorage, Shape)> {
        let x_leaf = cuda_input_to_leaf_tensor(s, l)?;
        let y = (self.forward)(&x_leaf)?;
        cuda_output_to_storage_shape(y)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> CandleResult<Option<Tensor>> {
        (self.backward)(arg, res, grad_res)
    }
}

// ----------------------------------------------------------------------
// KtForwardOp2 — binary-input shim
// ----------------------------------------------------------------------

/// Generic candle [`CustomOp2`] shim that drives a kt-typed kernel
/// forward + backward via closures.
///
/// Forward closure: `Fn(&Tensor, &Tensor) -> Result<Tensor>`.
/// Backward closure: `Fn(&Tensor, &Tensor, &Tensor, &Tensor) ->
/// Result<(Option<Tensor>, Option<Tensor>)>` (matches
/// [`CustomOp2::bwd`]).
pub struct KtForwardOp2 {
    name: &'static str,
    forward: Box<Fwd2Closure>,
    backward: Box<Bwd2Closure>,
}

impl KtForwardOp2 {
    pub fn new<F, B>(name: &'static str, forward: F, backward: B) -> Self
    where
        F: Fn(&Tensor, &Tensor) -> CandleResult<Tensor> + Send + Sync + 'static,
        B: Fn(
                &Tensor,
                &Tensor,
                &Tensor,
                &Tensor,
            ) -> CandleResult<(Option<Tensor>, Option<Tensor>)>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name,
            forward: Box::new(forward),
            backward: Box::new(backward),
        }
    }
}

impl CustomOp2 for KtForwardOp2 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(format!(
            "KtForwardOp2 '{}': no CPU implementation (kt-bridge is CUDA-only)",
            self.name
        )))
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let a = cuda_input_to_leaf_tensor(s1, l1)?;
        let b = cuda_input_to_leaf_tensor(s2, l2)?;
        let y = (self.forward)(&a, &b)?;
        cuda_output_to_storage_shape(y)
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)> {
        (self.backward)(arg1, arg2, res, grad_res)
    }
}

// ----------------------------------------------------------------------
// KtForwardOp3 — ternary-input shim
// ----------------------------------------------------------------------

/// Generic candle [`CustomOp3`] shim that drives a kt-typed kernel
/// forward + backward via closures.
///
/// Use this for the rotary_one shape — `(x, cos, sin) -> y` with
/// `(grad_x, None, None)` from the backward.
pub struct KtForwardOp3 {
    name: &'static str,
    forward: Box<Fwd3Closure>,
    backward: Box<Bwd3Closure>,
}

impl KtForwardOp3 {
    pub fn new<F, B>(name: &'static str, forward: F, backward: B) -> Self
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> CandleResult<Tensor> + Send + Sync + 'static,
        B: Fn(
                &Tensor,
                &Tensor,
                &Tensor,
                &Tensor,
                &Tensor,
            ) -> CandleResult<(
                Option<Tensor>,
                Option<Tensor>,
                Option<Tensor>,
            )>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name,
            forward: Box::new(forward),
            backward: Box::new(backward),
        }
    }
}

impl CustomOp3 for KtForwardOp3 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(format!(
            "KtForwardOp3 '{}': no CPU implementation (kt-bridge is CUDA-only)",
            self.name
        )))
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let a = cuda_input_to_leaf_tensor(s1, l1)?;
        let b = cuda_input_to_leaf_tensor(s2, l2)?;
        let c = cuda_input_to_leaf_tensor(s3, l3)?;
        let y = (self.forward)(&a, &b, &c)?;
        cuda_output_to_storage_shape(y)
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        (self.backward)(arg1, arg2, arg3, res, grad_res)
    }
}

// ----------------------------------------------------------------------
// Unit tests — CPU-only, structural / type-level
// ----------------------------------------------------------------------
//
// The CUDA round-trip tests live in `tests/cuda_forward_op_parity.rs`
// (run only when a CUDA device is available). These CPU tests verify
// that:
//   - The shim implements `CustomOp1` / `CustomOp2` / `CustomOp3` as
//     expected.
//   - The CPU-forward error message is well-formed (matching the
//     contract that production callers must provide a candle-path
//     fallback for CPU inputs).
//   - The `name` is plumbed correctly.

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn kt_forward_op1_name_is_plumbed() {
        let op = KtForwardOp1::new(
            "test-op1",
            |x: &Tensor| Ok(x.clone()),
            |_x: &Tensor, _y: &Tensor, _g: &Tensor| Ok(None),
        );
        assert_eq!(CustomOp1::name(&op), "test-op1");
    }

    #[test]
    fn kt_forward_op2_name_is_plumbed() {
        let op = KtForwardOp2::new(
            "test-op2",
            |a: &Tensor, _b: &Tensor| Ok(a.clone()),
            |_a, _b, _r, _g| Ok((None, None)),
        );
        assert_eq!(CustomOp2::name(&op), "test-op2");
    }

    #[test]
    fn kt_forward_op3_name_is_plumbed() {
        let op = KtForwardOp3::new(
            "test-op3",
            |a: &Tensor, _b: &Tensor, _c: &Tensor| Ok(a.clone()),
            |_a, _b, _c, _r, _g| Ok((None, None, None)),
        );
        assert_eq!(CustomOp3::name(&op), "test-op3");
    }

    #[test]
    fn kt_forward_op1_cpu_fwd_returns_msg_error() {
        let op = KtForwardOp1::new(
            "cpu-rejection",
            |x: &Tensor| Ok(x.clone()),
            |_x, _y, _g| Ok(None),
        );
        let x = Tensor::from_slice(&[1.0f32, 2.0], (2,), &candle_core::Device::Cpu).unwrap();
        let (st, layout) = x.storage_and_layout();
        let st = match &*st {
            candle_core::Storage::Cpu(c) => c,
            _ => panic!("expected cpu storage"),
        };
        let err = CustomOp1::cpu_fwd(&op, st, &layout).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("cpu-rejection"), "msg was: {msg}");
        assert!(msg.contains("CUDA-only"), "msg was: {msg}");
    }
}

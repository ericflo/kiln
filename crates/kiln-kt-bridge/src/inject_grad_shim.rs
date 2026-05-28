//! Candle `CustomOp1` shim for gradient injection — non-cuda-gated.
//!
//! This module hosts the `InjectGradientCandleShim` + the
//! `inject_gradient_via_shim` helper that lets a caller splice a
//! precomputed gradient into candle's backward walk for an
//! intermediate tensor. It mirrors the historical
//! `kiln-train::trainer::InjectTensorGradient` impl byte-for-byte.
//!
//! # Why this lives here (and not in `tape_bridge`)
//!
//! The `tape_bridge` module is gated on `feature = "cuda"` because
//! it depends on `kt_tensor_from_candle_cuda_borrow` and friends for
//! the kt-tape side channel. The shim itself is pure candle — no
//! cuda crate refs — so it can compile on every feature combination.
//! Hoisting it out of `tape_bridge` lets `kiln-train::trainer`'s
//! `InjectTensorGradient::apply_op1` call sites flip onto this shim
//! on all builds, not just CUDA. Once the flip is complete and
//! `InjectTensorGradient` is deleted, `kiln-train` drops its lone
//! production `candle_core::*` reference and `candle-core` can move
//! to `[dev-dependencies]`.
//!
//! # Relationship to `tape_bridge::inject_gradient_kt`
//!
//! `inject_gradient_kt` (CUDA-only) wraps `inject_gradient_via_shim`
//! and additionally records the gradient on a kt-tape as a side
//! channel for migration tracking + future kiln-kt-bridge-only
//! execution. The GradStore population is identical between the two
//! entry points; the cuda variant just emits an extra tape record.
//! (#1082)

use candle_core::Tensor;

/// Candle `CustomOp1` that injects a precomputed gradient into the
/// backward walk. Forward returns a scalar F32 zero (placeholder).
/// Backward returns the held `upstream` tensor (with `to_device +
/// to_dtype` matched to `arg`), exactly mirroring the contract of
/// the historical in-trainer `InjectTensorGradient::bwd`.
///
/// See [`inject_gradient_via_shim`] for the apply helper. (#1082)
#[derive(Clone)]
pub struct InjectGradientCandleShim {
    /// The precomputed gradient to emit as `arg`'s grad during
    /// candle's backward walk. Lives here so `bwd` doesn't have to
    /// reach into a thread-local / lookup table.
    pub upstream: Tensor,
}

impl std::fmt::Debug for InjectGradientCandleShim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectGradientCandleShim")
            .field("upstream_dtype", &self.upstream.dtype())
            .field("upstream_dims", &self.upstream.dims())
            .finish()
    }
}

impl candle_core::CustomOp1 for InjectGradientCandleShim {
    fn name(&self) -> &'static str {
        "kiln-inject-gradient-candle-shim"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Ok((
            candle_core::CpuStorage::F32(vec![0.0]),
            candle_core::Shape::from(()),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        let device = storage.device();
        let out_slice = device.clone_htod(&[0.0f32])?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
            candle_core::Shape::from(()),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        if self.upstream.dims() != arg.dims() {
            candle_core::bail!(
                "InjectGradientCandleShim shape mismatch: upstream {:?}, arg {:?}",
                self.upstream.dims(),
                arg.dims()
            );
        }
        let upstream = self.upstream.to_device(arg.device())?;
        let grad = if upstream.dtype() == arg.dtype() {
            upstream
        } else {
            upstream.to_dtype(arg.dtype())?
        };
        Ok(Some(grad))
    }
}

/// Apply [`InjectGradientCandleShim`] to `arg`. Returns a candle
/// scalar-zero Tensor whose `.backward()` populates
/// `grads[arg.id()]` with `upstream` (after device + dtype matching).
///
/// This is the candle-only entry point — works on every feature
/// combination, including CPU-only builds. For the CUDA path that
/// additionally records a kt-tape side channel, see
/// [`crate::tape_bridge::inject_gradient_kt`]. (#1082)
#[inline]
pub fn inject_gradient_via_shim(
    arg: &Tensor,
    upstream: &Tensor,
) -> candle_core::Result<Tensor> {
    if arg.dims() != upstream.dims() {
        candle_core::bail!(
            "inject_gradient_via_shim shape mismatch: arg {:?} != upstream {:?}",
            arg.dims(),
            upstream.dims()
        );
    }
    arg.apply_op1(InjectGradientCandleShim {
        upstream: upstream.clone(),
    })
}

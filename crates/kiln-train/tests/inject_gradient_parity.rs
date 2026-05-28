//! Parity test for the kt-tape replacement of
//! `kiln-train::trainer::InjectTensorGradient` (#1082, CP-4 step 2).
//!
//! # Two paths under test
//!
//! 1. **Candle baseline** — locally-defined `InjectTensorGradientRef`
//!    that mirrors the production trainer's
//!    `InjectTensorGradient::bwd` byte-for-byte (forward returns scalar
//!    zero, backward emits `upstream.to_device(arg.device()).to_dtype(arg.dtype())`
//!    as the grad for `arg`). We can't import the trainer's private
//!    struct directly, so the test inlines a faithful copy. The
//!    structural equivalence is the assertion: if the trainer ever
//!    diverges from "emit upstream regardless of grad_res", this test
//!    must update to match — that change is exactly the contract
//!    `InjectGradientBackward` is replacing.
//!
//! 2. **kt-tape via the bridge** — calls
//!    `kiln_kt_bridge::tape_bridge::inject_gradient_kt(arg, upstream)`
//!    inside a `with_tape_scope_emit_to_grad_store` scope, then
//!    `.backward()` on the returned candle scalar. The bridge:
//!    - registers `(arg_kt.id ↔ arg.id())` and
//!      `(out_kt.id ↔ candle_scalar.id())` mappings,
//!    - lets candle's `loss.backward()` propagate a grad for the
//!      scalar leaf,
//!    - walks the tape with that seed (which `InjectGradientBackward`
//!      ignores),
//!    - inserts the emitted kt grad back into the candle `GradStore`
//!      under `arg.id()`.
//!
//! Both paths must produce a candle `GradStore` whose `arg.id()` entry
//! is bit-equivalent (or numerically tight: a single device→device
//! memcpy plus optional dtype cast, no arithmetic operations between
//! the input and output). We use `max-abs-diff == 0.0` because every
//! intermediate is just a memcpy or a candle to_dtype call applied at
//! the same point in both paths.
//!
//! # Skip behaviour
//!
//! Non-CUDA builds: `#[cfg(feature = "cuda")]` no-ops the entire file.
//! CUDA build without a visible device: the test bails out early.
//!
//! # Substrate-only
//!
//! The trainer's `InjectTensorGradient` call sites are NOT flipped to
//! the kt-tape adapter in this PR — that's CP-4 step 4. This parity
//! test exists to prove the substrate is correct so the call-site flip
//! is a mechanical follow-up.

#![cfg(feature = "cuda")]

use candle_core::backprop::GradStore;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor, Var};

/// Local copy of the trainer's `InjectTensorGradient`. The trainer's
/// version is private to `trainer.rs`; we mirror its three method
/// bodies here so the test can construct the candle baseline without
/// depending on internal trainer types.
///
/// Any change to the trainer's semantics (forward placeholder shape,
/// backward dtype/device conversion) MUST be reflected here too — the
/// purpose of `InjectGradientBackward` is to replace this exact
/// contract.
#[derive(Clone)]
struct InjectTensorGradientRef {
    upstream: Tensor,
}

impl std::fmt::Debug for InjectTensorGradientRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectTensorGradientRef")
            .field("upstream_dtype", &self.upstream.dtype())
            .field("upstream_dims", &self.upstream.dims())
            .finish()
    }
}

impl CustomOp1 for InjectTensorGradientRef {
    fn name(&self) -> &'static str {
        "kiln-inject-tensor-gradient-test-ref"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Ok((CpuStorage::F32(vec![0.0]), Shape::from(())))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        let device = storage.device();
        let out_slice = device.clone_htod(&[0.0f32])?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
            Shape::from(()),
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
                "InjectTensorGradientRef shape mismatch: upstream {:?}, arg {:?}",
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

fn cuda_device() -> Option<Device> {
    Device::new_cuda(0).ok()
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn random_bf16_vec(len: usize, seed: u64, scale: f32) -> Vec<half::bf16> {
    let mut state = seed;
    (0..len)
        .map(|_| half::bf16::from_f32(lcg(&mut state) * scale))
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).expect("a -> f32");
    let bf = b.to_dtype(DType::F32).expect("b -> f32");
    let diff = (&af - &bf).expect("diff").abs().expect("abs");
    let flat = diff.flatten_all().expect("flat");
    let values = flat.to_vec1::<f32>().expect("diff vec");
    values.iter().cloned().fold(0.0f32, f32::max)
}

/// Build a (arg_var, upstream) pair on the given device. `arg_var` is
/// a tracked candle `Var` (so candle's GradStore allocates a slot for
/// it under `.id()`), and `upstream` is a sibling F32 tensor we'll
/// pass as the precomputed grad.
fn build_inputs(device: &Device, shape: &[usize]) -> (Var, Tensor) {
    let n: usize = shape.iter().product();
    // Use bf16 for arg (matching the trainer's tile dtypes); upstream
    // we keep as f32 to also exercise the `to_dtype` leg in both paths.
    let arg_host = random_bf16_vec(n, 0x1234_5678_dead_beef, 0.5);
    let arg_t = Tensor::from_vec(arg_host, shape, &Device::Cpu)
        .expect("arg cpu")
        .to_device(device)
        .expect("arg -> device")
        .contiguous()
        .expect("arg contig");
    // Var wrapping makes track_op() == true so candle's backward walk
    // populates the GradStore for it.
    let arg_var = Var::from_tensor(&arg_t).expect("Var::from_tensor(arg)");

    let mut state: u64 = 0xabcd_ef01_2345_6789;
    let upstream_host: Vec<f32> = (0..n).map(|_| lcg(&mut state)).collect();
    let upstream = Tensor::from_vec(upstream_host, shape, &Device::Cpu)
        .expect("upstream cpu")
        .to_device(device)
        .expect("upstream -> device")
        .contiguous()
        .expect("upstream contig");

    (arg_var, upstream)
}

/// Drive the candle baseline path. Returns the GradStore so the caller
/// can pull the grad keyed on `arg_var.id()`.
fn run_candle_baseline(arg_var: &Var, upstream: &Tensor) -> GradStore {
    let injected = arg_var
        .as_tensor()
        .apply_op1(InjectTensorGradientRef {
            upstream: upstream.clone(),
        })
        .expect("apply_op1");
    injected.backward().expect("candle backward")
}

/// Drive the kt-tape bridge path. Returns the GradStore.
fn run_kt_tape(arg_var: &Var, upstream: &Tensor) -> GradStore {
    // Bridge wrapper expects `forward()` to return (payload, loss).
    // For this parity test the "loss" is the scalar zero placeholder
    // returned by `inject_gradient_kt` (mirroring the candle path's
    // `injected` scalar).
    let arg_t: Tensor = arg_var.as_tensor().clone();
    let upstream_clone: Tensor = upstream.clone();
    let ((), grads) =
        kiln_kt_bridge::tape_bridge::with_tape_scope_emit_to_grad_store(move || {
            let injected =
                kiln_kt_bridge::tape_bridge::inject_gradient_kt(&arg_t, &upstream_clone)
                    .map_err(|e| {
                        kiln_kt_bridge::BridgeError::new(format!(
                            "inject_gradient_kt: {e}"
                        ))
                    })?;
            Ok(((), injected))
        })
        .expect("with_tape_scope_emit_to_grad_store");
    grads
}

#[test]
fn inject_gradient_kt_matches_candle_custom_op1() {
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("inject_gradient parity: no CUDA device — skipping");
            return;
        }
    };

    // Small but non-trivial shape: a 2D tile that's representative of
    // the trainer's tile sizes without burning GPU time.
    let shape = [4usize, 16];

    // -------- Path A: candle baseline.
    let (arg_var_a, upstream_a) = build_inputs(&device, &shape);
    let grads_a = run_candle_baseline(&arg_var_a, &upstream_a);
    let grad_a = grads_a
        .get(arg_var_a.as_tensor())
        .expect("candle baseline must produce a grad for arg")
        .clone();
    assert_eq!(grad_a.dims(), &shape);
    assert_eq!(grad_a.dtype(), DType::BF16, "grad dtype must match arg dtype");

    // -------- Path B: kt-tape bridge.
    // Re-seed inputs identically — the LCG seeds match `build_inputs`
    // exactly, so the second build produces byte-identical tensors.
    // (Var allocates a fresh TensorId per build; the grad value must
    // still be the same because the input bytes match.)
    let (arg_var_b, upstream_b) = build_inputs(&device, &shape);
    let grads_b = run_kt_tape(&arg_var_b, &upstream_b);
    let grad_b = grads_b
        .get(arg_var_b.as_tensor())
        .expect("kt-tape bridge must produce a grad for arg")
        .clone();
    assert_eq!(grad_b.dims(), &shape);
    assert_eq!(grad_b.dtype(), DType::BF16, "grad dtype must match arg dtype");

    // -------- Bit-equivalence.
    //
    // The candle baseline does: upstream.to_device(arg.device()).to_dtype(arg.dtype()).
    // The kt-tape path does the same to_device + to_dtype on the candle
    // side BEFORE copying into kt, then memcpys back into a fresh
    // candle Tensor for the GradStore insert. Both routes therefore
    // see identical intermediate bytes; max-abs-diff must be exactly 0.
    let diff = max_abs_diff(&grad_a, &grad_b);
    assert_eq!(
        diff, 0.0,
        "kt-tape inject_gradient must be bit-equivalent to candle CustomOp1 path \
         (max-abs-diff was {diff}). Both paths apply the same to_device + \
         to_dtype on candle's side; any nonzero diff is a wiring bug."
    );
}

#[test]
fn inject_gradient_kt_skip_dtype_cast_when_matched() {
    // Exercise the "upstream already matches arg dtype" branch in both
    // paths. We pass an upstream tensor that is already bf16 and on
    // the arg's device, so neither path takes its to_device/to_dtype
    // leg.
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("inject_gradient parity (same dtype): no CUDA device — skipping");
            return;
        }
    };

    let shape = [8usize, 8];
    let n: usize = shape.iter().product();
    let arg_host = random_bf16_vec(n, 0xface_b00c_dead_beef, 0.25);
    let upstream_host = random_bf16_vec(n, 0xc0ff_eef0_0d0d_b00b, 0.5);

    // Helper that builds (Var, Tensor) on-device with the SAME bytes
    // every call.
    let build = || {
        let arg_t = Tensor::from_vec(arg_host.clone(), &shape[..], &Device::Cpu)
            .expect("arg cpu")
            .to_device(&device)
            .expect("arg -> device")
            .contiguous()
            .expect("arg contig");
        let arg_var = Var::from_tensor(&arg_t).expect("Var::from_tensor");
        let upstream = Tensor::from_vec(upstream_host.clone(), &shape[..], &Device::Cpu)
            .expect("upstream cpu")
            .to_device(&device)
            .expect("upstream -> device")
            .contiguous()
            .expect("upstream contig");
        (arg_var, upstream)
    };

    let (arg_var_a, upstream_a) = build();
    let grads_a = run_candle_baseline(&arg_var_a, &upstream_a);
    let grad_a = grads_a
        .get(arg_var_a.as_tensor())
        .expect("candle: grad exists")
        .clone();

    let (arg_var_b, upstream_b) = build();
    let grads_b = run_kt_tape(&arg_var_b, &upstream_b);
    let grad_b = grads_b
        .get(arg_var_b.as_tensor())
        .expect("kt-tape: grad exists")
        .clone();

    assert_eq!(grad_a.dtype(), DType::BF16);
    assert_eq!(grad_b.dtype(), DType::BF16);
    let diff = max_abs_diff(&grad_a, &grad_b);
    assert_eq!(
        diff, 0.0,
        "same-dtype path must also be bit-equivalent: max-abs-diff {diff}"
    );
}

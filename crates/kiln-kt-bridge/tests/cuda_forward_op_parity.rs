//! CUDA-side end-to-end parity test for [`kiln_kt_bridge::forward_op`]
//! ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # What this test crate covers
//!
//! Exercises a trivial kt-typed forward+backward kernel pair
//! (element-wise scalar multiply: `y = scale * x`; `dx = scale * dy`)
//! wired through the [`KtForwardOp1`] generic shim, and compares the
//! end-to-end candle [`Tensor::backward`] result against:
//!
//! 1. A direct candle [`Tensor::affine`] reference (`y = x * scale`,
//!    same scalar multiplication path).
//! 2. The analytical gradient `scale * grad_y` computed on CPU.
//!
//! The goal is not to validate the underlying kt kernel (the kt
//! `mul_scalar` op has its own parity tests); it's to validate that
//! the shim's CustomOp glue — leaf-tensor construction in `cuda_fwd`,
//! storage unwrap in the closure return, autograd graph extension via
//! `apply_op1_arc`, backward closure dispatch through `CustomOp1::bwd`
//! — wires up correctly on real CUDA storage.
//!
//! Gated on `CUDA_VISIBLE_DEVICES` via `try_cuda_device()`; silently
//! skips when no CUDA device is reachable so the file is harmless on
//! non-CUDA hosts (the workspace builds it under `--features cuda`
//! anyway because `kiln-kt-bridge` always pulls the cuda feature).
//!
//! # Coverage matrix
//!
//! - **f32 forward parity** — single mul-scalar through the shim,
//!   compared to candle `affine`.
//! - **f32 backward parity** — `.backward()` through the shim,
//!   compared to the analytical `scale * dy` on CPU.
//! - **bf16 forward parity** — same, in bf16 to exercise the dtype
//!   path. Tolerance loosened to absorb bf16 rounding (5e-2).
//! - **CPU-input rejection** — shim returns
//!   `Error::Msg("...no CPU implementation...")` when the input
//!   tensor is on CPU. Matches the documented contract.

use std::sync::Arc;

use candle_core::Device as CandleDevice;
use candle_core::Tensor;

use kiln_kt_bridge::forward_op::KtForwardOp1;
use kiln_kt_bridge::{kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy};
use kiln_tensor::ops::mul_scalar;
use kiln_tensor::{cuda_to_host_copy, CpuStorage};

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn try_cuda_device() -> Option<Arc<candle_core::CudaDevice>> {
    let dev = CandleDevice::new_cuda(0).ok()?;
    match dev {
        CandleDevice::Cuda(c) => Some(Arc::new(c)),
        _ => None,
    }
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E37_9B97_F4A7_C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEAD_BEEF).wrapping_mul(0x9E37_9B97_F4A7_C15);
        let f = ((s as u32 % 1024) as f32 - 512.0) / 5120.0;
        out.push(f);
    }
    out
}

fn cuda_tensor_to_vec_f32(t: &kiln_tensor::Tensor) -> Vec<f32> {
    let host = cuda_to_host_copy(t).expect("cuda → host copy");
    let cpu_st = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("CPU storage post copy");
    let bytes = cpu_st.as_bytes();
    assert_eq!(bytes.len() % 4, 0, "expected F32 layout");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn candle_cuda_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let t_cpu = t.to_device(&CandleDevice::Cpu).expect("D2H copy");
    t_cpu.flatten_all().unwrap().to_vec1::<f32>().expect("vec1<f32>")
}

fn candle_cuda_to_vec_bf16(t: &Tensor) -> Vec<f32> {
    let t_cpu = t.to_device(&CandleDevice::Cpu).expect("D2H copy");
    let t_f32 = t_cpu.to_dtype(candle_core::DType::F32).expect("bf16→f32 cast");
    t_f32.flatten_all().unwrap().to_vec1::<f32>().expect("vec1<f32>")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ----------------------------------------------------------------------
// Build a `KtForwardOp1` that performs `y = scale * x` via kt.
// ----------------------------------------------------------------------

fn make_scale_op(scale: f32) -> KtForwardOp1 {
    KtForwardOp1::new(
        "test-scale-op1",
        move |x: &Tensor| -> candle_core::Result<Tensor> {
            // Forward: y = scale * x  via kt's mul_scalar.
            // Force contiguous before the borrow (cheap arc clone if
            // already contiguous).
            let x_c = x.contiguous()?;
            let x_kt = kt_tensor_from_candle_cuda_borrow(&x_c).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op fwd borrow x: {e}"))
            })?;
            let y_kt = mul_scalar(&x_kt, scale).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op fwd kt mul_scalar: {e}"))
            })?;
            kt_tensor_to_candle_cuda_copy(&y_kt).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op fwd copy-back y: {e}"))
            })
        },
        move |_arg: &Tensor, _res: &Tensor, grad_res: &Tensor| -> candle_core::Result<Option<Tensor>> {
            // dx = scale * grad_res — same kt round-trip.
            let g_c = grad_res.contiguous()?;
            let g_kt = kt_tensor_from_candle_cuda_borrow(&g_c).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op bwd borrow gy: {e}"))
            })?;
            let dx_kt = mul_scalar(&g_kt, scale).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op bwd kt mul_scalar: {e}"))
            })?;
            let dx = kt_tensor_to_candle_cuda_copy(&dx_kt).map_err(|e| {
                candle_core::Error::Msg(format!("scale-op bwd copy-back dx: {e}"))
            })?;
            Ok(Some(dx))
        },
    )
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[test]
fn forward_op1_cpu_input_returns_typed_error() {
    // CPU tensors must be rejected by the shim with the documented
    // "no CPU implementation" message — production callers are
    // expected to dispatch on device type and fall back to the candle
    // path themselves.
    let op = make_scale_op(2.0);

    let x_cpu = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &CandleDevice::Cpu).unwrap();
    let res = x_cpu.apply_op1_arc(Arc::new(Box::new(op)));
    let err = res.expect_err("expected typed error for CPU input");
    let msg = format!("{err}");
    assert!(msg.contains("test-scale-op1"), "msg: {msg}");
    assert!(msg.contains("CUDA-only"), "msg: {msg}");
}

#[test]
fn forward_op1_cuda_fwd_parity_f32() {
    let Some(cuda) = try_cuda_device() else {
        eprintln!("[skip] no CUDA device available");
        return;
    };
    let device = CandleDevice::Cuda((*cuda).clone());

    let n = 1024usize;
    let scale = 2.5f32;
    let data = pattern(n, 42);

    let x = Tensor::from_slice(&data, (n,), &device).unwrap();

    // Path A: through the KtForwardOp1 shim.
    let op = make_scale_op(scale);
    let y_shim = x.apply_op1_arc(Arc::new(Box::new(op))).unwrap();
    let y_shim_host = candle_cuda_to_vec_f32(&y_shim);

    // Path B: direct candle affine reference.
    let y_ref = x.affine(scale as f64, 0.0).unwrap();
    let y_ref_host = candle_cuda_to_vec_f32(&y_ref);

    let max_diff = max_abs_diff(&y_shim_host, &y_ref_host);
    assert!(
        max_diff < 1e-5,
        "f32 forward parity drifted: max abs diff = {max_diff}"
    );
}

#[test]
fn forward_op1_cuda_bwd_parity_f32() {
    let Some(cuda) = try_cuda_device() else {
        eprintln!("[skip] no CUDA device available");
        return;
    };
    let device = CandleDevice::Cuda((*cuda).clone());

    let n = 256usize;
    let scale = 3.0f32;
    let data = pattern(n, 7);

    // The shim path: y = shim(x); l = sum(y); l.backward() yields dx.
    let x_param = candle_core::Var::from_slice(&data, (n,), &device).unwrap();
    let x_t = x_param.as_tensor();
    let op = make_scale_op(scale);
    let y_shim = x_t.apply_op1_arc(Arc::new(Box::new(op))).unwrap();
    let loss = y_shim.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dx_shim = grads
        .get(x_t)
        .expect("dx for x must be present in the grad map");
    let dx_shim_host = candle_cuda_to_vec_f32(dx_shim);

    // Analytical: l = sum(scale * x)  ⇒  dl/dx = scale * 1 everywhere.
    let expected: Vec<f32> = (0..n).map(|_| scale).collect();

    let max_diff = max_abs_diff(&dx_shim_host, &expected);
    assert!(
        max_diff < 1e-5,
        "f32 backward parity drifted: max abs diff = {max_diff} \
         (got front: {:?}, expected scale={scale})",
        &dx_shim_host[..dx_shim_host.len().min(4)]
    );
}

#[test]
fn forward_op1_cuda_bwd_parity_chained_with_more_ops() {
    // More demanding: l = sum( (shim(x))^2 ).  dl/dx = 2 * scale * scale * x.
    // This proves the shim's autograd integration (via apply_op1_arc +
    // CustomOp1::bwd) chains correctly through a downstream op.
    let Some(cuda) = try_cuda_device() else {
        eprintln!("[skip] no CUDA device available");
        return;
    };
    let device = CandleDevice::Cuda((*cuda).clone());

    let n = 64usize;
    let scale = 1.5f32;
    let data = pattern(n, 17);

    let x_param = candle_core::Var::from_slice(&data, (n,), &device).unwrap();
    let x_t = x_param.as_tensor();
    let op = make_scale_op(scale);
    let y_shim = x_t.apply_op1_arc(Arc::new(Box::new(op))).unwrap();
    let y_sq = (&y_shim * &y_shim).unwrap();
    let loss = y_sq.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dx_shim = grads
        .get(x_t)
        .expect("dx for x must be present in the grad map");
    let dx_shim_host = candle_cuda_to_vec_f32(dx_shim);

    // Analytical: y = scale*x; loss = sum(y^2).
    // dL/dx = dL/dy * dy/dx = (2y) * scale = 2 * scale * scale * x.
    let factor = 2.0 * scale * scale;
    let expected: Vec<f32> = data.iter().map(|&v| factor * v).collect();

    let max_diff = max_abs_diff(&dx_shim_host, &expected);
    assert!(
        max_diff < 1e-3,
        "chained bwd parity drifted: max abs diff = {max_diff}"
    );
}

#[test]
fn forward_op1_cuda_fwd_parity_bf16() {
    // Cast to bf16 to exercise the dtype path.  We compare the shim
    // output to the candle reference at bf16, both round-tripped to
    // f32 for the actual comparison.
    let Some(cuda) = try_cuda_device() else {
        eprintln!("[skip] no CUDA device available");
        return;
    };
    let device = CandleDevice::Cuda((*cuda).clone());

    let n = 1024usize;
    let scale = 2.0f32;
    let data = pattern(n, 99);

    let x_f32 = Tensor::from_slice(&data, (n,), &device).unwrap();
    let x = x_f32.to_dtype(candle_core::DType::BF16).unwrap();

    let op = make_scale_op(scale);
    let y_shim = x.apply_op1_arc(Arc::new(Box::new(op))).unwrap();
    let y_shim_host = candle_cuda_to_vec_bf16(&y_shim);

    let y_ref = x.affine(scale as f64, 0.0).unwrap();
    let y_ref_host = candle_cuda_to_vec_bf16(&y_ref);

    let max_diff = max_abs_diff(&y_shim_host, &y_ref_host);
    assert!(
        max_diff < 5e-2,
        "bf16 forward parity drifted: max abs diff = {max_diff}"
    );
}

#[test]
fn forward_op1_used_twice_keeps_separate_outputs() {
    // Sanity: applying the shim twice in sequence (y = shim(shim(x)))
    // should yield scale^2 * x, and the autograd through both
    // applications should give scale^2 * dy.
    let Some(cuda) = try_cuda_device() else {
        eprintln!("[skip] no CUDA device available");
        return;
    };
    let device = CandleDevice::Cuda((*cuda).clone());

    let n = 32usize;
    let scale = 1.5f32;
    let data = pattern(n, 5);

    let x_param = candle_core::Var::from_slice(&data, (n,), &device).unwrap();
    let x_t = x_param.as_tensor();

    let op1 = make_scale_op(scale);
    let op2 = make_scale_op(scale);
    let y1 = x_t.apply_op1_arc(Arc::new(Box::new(op1))).unwrap();
    let y2 = y1.apply_op1_arc(Arc::new(Box::new(op2))).unwrap();

    let y2_host = candle_cuda_to_vec_f32(&y2);
    let expected_y: Vec<f32> = data.iter().map(|&v| v * scale * scale).collect();
    let fwd_diff = max_abs_diff(&y2_host, &expected_y);
    assert!(
        fwd_diff < 1e-5,
        "chained-shim fwd diff = {fwd_diff}"
    );

    let loss = y2.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dx = grads.get(x_t).expect("dx must be present");
    let dx_host = candle_cuda_to_vec_f32(dx);
    let expected_dx = scale * scale;
    let bwd_diff = dx_host
        .iter()
        .map(|&v| (v - expected_dx).abs())
        .fold(0.0f32, f32::max);
    assert!(
        bwd_diff < 1e-5,
        "chained-shim bwd diff = {bwd_diff}"
    );

    // Avoid unused warning for kt-tensor crate; force a use.
    let _ = cuda_tensor_to_vec_f32;
}

//! Parity test: kt CUDA `cuda_activation_unary` (extended with the
//! unary-math kinds 5..=14 in #1082) vs kt CPU references.
//!
//! Phase 4 substrate validation. Confirms the per-kind branches of
//! `csrc/activation.cu` produce values matching the CPU forwards of
//! `ops::unary_arith`, `ops::trig`, `ops::hyperbolic` to bit-tight
//! tolerance on F32 and within BF16 round-trip slop on BF16.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_activation_unary, ops, CpuStorage, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Deterministic data with a moderate range. Each new unary kind picks
/// a subrange suited to its domain (sqrt/log clamp negatives away).
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // [-3.0, 3.0] — keeps exp/sinh/cosh inside finite range
        // and away from f16 overflow.
        let f = ((s as u32 % 4096) as f32 - 2048.0) / 2048.0 * 3.0;
        out.push(f);
    }
    out
}

fn positive_pattern(n: usize, seed: u64) -> Vec<f32> {
    // [0.01, 5.0] — for log/sqrt where negatives produce NaN.
    pattern(n, seed)
        .into_iter()
        .map(|x| 0.01_f32 + x.abs())
        .collect()
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run CPU and CUDA forwards through the same `ops::*` entry points,
/// confirm equality within tolerance. This exercises `DeviceOp1`'s
/// dispatch1 fallthrough: CPU input -> cpu_fwd; CUDA input -> cuda_fwd.
fn check_op<F>(name: &str, op: F, data: &[f32], dtype: CandleDType, tol: f32)
where
    F: Fn(&Tensor) -> kiln_tensor::Result<Tensor>,
{
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping {name}");
        return;
    };
    let n = data.len();

    // CPU reference: build CPU kt-Tensor, call op (cpu_fwd path).
    let cpu_in = Tensor::from_slice(data, vec![n]).unwrap();
    let cpu_out = op(&cpu_in).expect("cpu op");
    let cpu_vec = read_f32(&cpu_out);

    // CUDA path: build candle CUDA tensor at the requested dtype,
    // borrow into kt, dispatch via the same ops::* function (which
    // routes through cuda_fwd -> cuda_activation_unary).
    let x_cd = CandleTensor::from_vec(data.to_vec(), (n,), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = op(&x_kt).expect("cuda op");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(cpu_vec.len(), got_vec.len(), "{name}: len mismatch");
    let mut max_abs = 0.0_f32;
    for (i, (a, b)) in cpu_vec.iter().zip(got_vec.iter()).enumerate() {
        // Treat NaN==NaN — sqrt/log of negatives at the dtype-promote
        // boundary produces NaN on both sides; skip the diff there.
        if a.is_nan() && b.is_nan() {
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < tol,
            "{name} idx {i}: cpu={a} cuda={b} diff={d} > tol={tol}"
        );
    }
    eprintln!("{name} (n={n} dtype={dtype:?}): max_abs={max_abs}");
}

// ---- unary_arith --------------------------------------------------------

#[test]
fn cuda_abs_f32_parity() {
    let data = pattern(257, 1);
    check_op("abs", ops::abs, &data, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_abs_bf16_parity() {
    let data = pattern(513, 2);
    check_op("abs", ops::abs, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_neg_f32_parity() {
    // Negation is bit-exact, but check_op uses strict `<` so a 0.0
    // tolerance fails on the first equal pair. Use the smallest
    // positive f32 as the strictness floor.
    let data = pattern(129, 3);
    check_op("neg", ops::neg, &data, CandleDType::F32, f32::EPSILON);
}

#[test]
fn cuda_neg_bf16_parity() {
    let data = pattern(129, 4);
    check_op("neg", ops::neg, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_exp_f32_parity() {
    // Bounded input keeps exp finite; CPU uses libm exp, CUDA __expf
    // — tolerance reflects __expf's ~2 ulp accuracy.
    let data = pattern(257, 5);
    check_op("exp", ops::exp, &data, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_exp_bf16_parity() {
    // BF16's 7-bit mantissa quantizes large exp() outputs (e^3 ≈ 20)
    // to multiples of the local ULP — ~0.2 worst-case absolute slop.
    let data = pattern(257, 6);
    check_op("exp", ops::exp, &data, CandleDType::BF16, 3e-1);
}

#[test]
fn cuda_ln_f32_parity() {
    let data = positive_pattern(257, 7);
    check_op("ln", ops::ln, &data, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_ln_bf16_parity() {
    let data = positive_pattern(257, 8);
    check_op("ln", ops::ln, &data, CandleDType::BF16, 1e-1);
}

#[test]
fn cuda_sqrt_f32_parity() {
    let data = positive_pattern(257, 9);
    check_op("sqrt", ops::sqrt, &data, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_sqrt_bf16_parity() {
    let data = positive_pattern(257, 10);
    check_op("sqrt", ops::sqrt, &data, CandleDType::BF16, 1e-1);
}

// ---- trig ---------------------------------------------------------------

#[test]
fn cuda_sin_f32_parity() {
    let data = pattern(257, 11);
    check_op("sin", ops::sin, &data, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_sin_bf16_parity() {
    let data = pattern(257, 12);
    check_op("sin", ops::sin, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_cos_f32_parity() {
    let data = pattern(257, 13);
    check_op("cos", ops::cos, &data, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_cos_bf16_parity() {
    let data = pattern(257, 14);
    check_op("cos", ops::cos, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_tan_f32_parity() {
    // tan is unbounded near (k+0.5)π; clamp input to keep finite.
    let data: Vec<f32> = pattern(129, 15).into_iter().map(|x| x * 0.3).collect();
    check_op("tan", ops::tan, &data, CandleDType::F32, 1e-4);
}

// ---- hyperbolic --------------------------------------------------------

#[test]
fn cuda_sinh_f32_parity() {
    let data = pattern(257, 16);
    check_op("sinh", ops::sinh, &data, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_sinh_bf16_parity() {
    let data = pattern(257, 17);
    check_op("sinh", ops::sinh, &data, CandleDType::BF16, 2e-1);
}

#[test]
fn cuda_cosh_f32_parity() {
    let data = pattern(257, 18);
    check_op("cosh", ops::cosh, &data, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_cosh_bf16_parity() {
    let data = pattern(257, 19);
    check_op("cosh", ops::cosh, &data, CandleDType::BF16, 2e-1);
}

// ---- direct cuda_activation_unary FFI smoke test -----------------------

#[test]
fn cuda_activation_unary_log_direct_call() {
    // Confirm the FFI bounds-check accepts the new kinds (was kind <= 4).
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let n = data.len();
    let x_cd = CandleTensor::from_vec(data.clone(), (n,), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // KIND_LOG = 5 (new); should succeed.
    let out_kt = cuda_activation_unary(&x_kt, 5).expect("KIND_LOG");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, &g) in got.iter().enumerate() {
        let want = (i as f32 + 1.0).ln();
        assert!((g - want).abs() < 1e-5, "i={i}: got {g}, want {want}");
    }

    // KIND_MAX+1 (=15) must still error.
    assert!(cuda_activation_unary(&x_kt, 15).is_err());
}

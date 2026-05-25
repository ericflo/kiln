//! Parity test: kt CUDA `cuda_activation_unary` (extended with the
//! sign/round/reciprocal kinds 22..=NN in #1082 follow-up) vs kt CPU
//! references.
//!
//! Phase 4 substrate validation. Confirms the per-kind branches of
//! `csrc/activation.cu` produce values matching the CPU forwards in
//! `ops::sign_and_round` to bit-tight tolerance on F32 and within
//! BF16 round-trip slop on BF16.
//!
//! `recip` ships first (kind 22, unblocks the
//! `(variance + eps).sqrt().recip()` RMSNorm-style pattern). The
//! other five kinds (sign/floor/ceil/round/trunc) land in
//! subsequent commits of the same #1082 series and append to this
//! file.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_activation_unary, ops, CpuStorage, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Deterministic data with a moderate range.
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // [-3.0, 3.0]
        let f = ((s as u32 % 4096) as f32 - 2048.0) / 2048.0 * 3.0;
        out.push(f);
    }
    out
}

/// `[0.1, 5.1]` — avoids the zero singularity for `recip` so that
/// CPU and CUDA don't disagree on `1/0 = ±inf` vs the BF16 narrow
/// path (which clamps to BF16's representable range).
fn nonzero_pattern(n: usize, seed: u64) -> Vec<f32> {
    pattern(n, seed)
        .into_iter()
        .map(|x| 0.1_f32 + x.abs())
        .collect()
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check_op<F>(name: &str, op: F, data: &[f32], dtype: CandleDType, tol: f32)
where
    F: Fn(&Tensor) -> kiln_tensor::Result<Tensor>,
{
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping {name}");
        return;
    };
    let n = data.len();

    // Run the CPU reference at the target dtype so it sees the same
    // narrowed inputs the CUDA path does. floor/ceil/round/trunc are
    // integer-discontinuous: an F32 input like 1.99 has `floor = 1`,
    // but the BF16-narrowed input rounds to 2.0 and `floor = 2`.
    // Building the CPU tensor directly at BF16/F16 (instead of F32
    // and letting the CUDA copy do the narrowing) keeps both sides
    // honest. Smooth ops tolerate either form; this normalization
    // just makes the comparison apples-to-apples for every kind.
    let cpu_in = match dtype {
        CandleDType::F32 => Tensor::from_slice(data, vec![n]).unwrap(),
        CandleDType::BF16 => {
            let bf: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            Tensor::from_slice(&bf, vec![n]).unwrap()
        }
        CandleDType::F16 => {
            let h: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            Tensor::from_slice(&h, vec![n]).unwrap()
        }
        other => panic!("unsupported test dtype {other:?}"),
    };
    let cpu_out = op(&cpu_in).expect("cpu op");
    let cpu_vec: Vec<f32> = match dtype {
        CandleDType::F32 => read_f32(&cpu_out),
        CandleDType::BF16 => {
            let cpu = cpu_out
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .unwrap();
            cpu.as_bytes()
                .chunks(2)
                .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect()
        }
        CandleDType::F16 => {
            let cpu = cpu_out
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .unwrap();
            cpu.as_bytes()
                .chunks(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect()
        }
        _ => unreachable!(),
    };

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
        // Treat NaN==NaN — domain-edge cases (e.g. 1/0) may produce
        // matching NaN/inf on both sides; skip the diff there.
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
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

// ---- reciprocal (#1082: kind 22) ---------------------------------------

#[test]
fn cuda_reciprocal_f32_parity() {
    let data = nonzero_pattern(257, 1);
    check_op("recip", ops::reciprocal, &data, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_reciprocal_bf16_parity() {
    // Both CPU and CUDA compute `1/x` in F32 then narrow to BF16,
    // so any diff reflects the F32→BF16 rounding mode rather than
    // the math. Tolerance matches what `cuda_neg_bf16_parity`
    // accepts on a pure round-trip. Restrict the input range to
    // `[1.1, 6.1]` so `1/x ∈ (0.16, 0.91)` and the BF16 ulp stays
    // below ~0.01.
    let data: Vec<f32> = nonzero_pattern(257, 2).into_iter().map(|x| x + 1.0).collect();
    check_op("recip", ops::reciprocal, &data, CandleDType::BF16, 1e-1);
}

#[test]
fn cuda_reciprocal_negative_values_f32_parity() {
    // Exercise the negative half of the domain so the sign bit
    // round-trips correctly through the kernel.
    let data: Vec<f32> = nonzero_pattern(129, 3).into_iter().map(|x| -x).collect();
    check_op("recip-neg", ops::reciprocal, &data, CandleDType::F32, 1e-5);
}

// ---- sign (#1082: kind 23) ---------------------------------------------

#[test]
fn cuda_sign_f32_parity() {
    // Mix of positive, negative, and a few zeros to exercise the
    // three-way branch.
    let mut data = pattern(257, 10);
    // Salt in some exact zeros and very small values straddling zero.
    data[0] = 0.0;
    data[1] = -0.0;
    data[2] = f32::MIN_POSITIVE;
    data[3] = -f32::MIN_POSITIVE;
    check_op("sign", ops::sign, &data, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_sign_bf16_parity() {
    // BF16: -1.0 / 0.0 / 1.0 are all exactly representable so this
    // is a bit-tight assertion modulo the F32→BF16 narrowing on the
    // input side (which may flush the smallest subnormals to 0).
    let data = pattern(513, 11);
    check_op("sign", ops::sign, &data, CandleDType::BF16, 1e-3);
}

// ---- direct cuda_activation_unary FFI smoke for KIND_SIGN --------------

#[test]
fn cuda_activation_unary_recip_direct_call() {
    // Confirm the FFI bounds-check accepts the existing kind 22.
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

    // KIND_RECIP = 22; should succeed.
    let out_kt = cuda_activation_unary(&x_kt, 22).expect("KIND_RECIP");
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
        let want = 1.0_f32 / (i as f32 + 1.0);
        assert!((g - want).abs() < 1e-5, "i={i}: got {g}, want {want}");
    }
}

#[test]
fn cuda_activation_unary_sign_direct_call() {
    // Confirm the FFI bounds-check accepts kind 23.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Mix of positive, negative, and zero.
    let data: Vec<f32> = (0..64)
        .map(|i| (i as f32 - 32.0) * 0.25)
        .collect();
    let n = data.len();
    let x_cd = CandleTensor::from_vec(data.clone(), (n,), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // KIND_SIGN = 23; should succeed.
    let out_kt = cuda_activation_unary(&x_kt, 23).expect("KIND_SIGN");
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
        let v = data[i];
        let want = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        assert!(
            (g - want).abs() < 1e-6,
            "i={i}: v={v} got {g}, want {want}"
        );
    }
}

// ---- floor (#1082: kind 24) --------------------------------------------

#[test]
fn cuda_floor_f32_parity() {
    let data = pattern(257, 20);
    check_op("floor", ops::floor, &data, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_floor_bf16_parity() {
    let data = pattern(257, 21);
    check_op("floor", ops::floor, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_activation_unary_floor_direct_call() {
    // Confirm the FFI bounds-check accepts kind 24.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
    let n = data.len();
    let x_cd = CandleTensor::from_vec(data.clone(), (n,), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // KIND_FLOOR = 24; should succeed.
    let out_kt = cuda_activation_unary(&x_kt, 24).expect("KIND_FLOOR");
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
        let want = data[i].floor();
        assert!(
            (g - want).abs() < 1e-6,
            "i={i}: v={} got {g}, want {want}",
            data[i]
        );
    }
}

// ---- ceil (#1082: kind 25) ---------------------------------------------

#[test]
fn cuda_ceil_f32_parity() {
    let data = pattern(257, 22);
    check_op("ceil", ops::ceil, &data, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_ceil_bf16_parity() {
    let data = pattern(257, 23);
    check_op("ceil", ops::ceil, &data, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_activation_unary_ceil_direct_call() {
    // Confirm the FFI bounds-check accepts kind 25 (new KIND_MAX)
    // and rejects 26 (one past the current max).
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.41).collect();
    let n = data.len();
    let x_cd = CandleTensor::from_vec(data.clone(), (n,), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // KIND_CEIL = 25 (new max); should succeed.
    let out_kt = cuda_activation_unary(&x_kt, 25).expect("KIND_CEIL");
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
        let want = data[i].ceil();
        assert!(
            (g - want).abs() < 1e-6,
            "i={i}: v={} got {g}, want {want}",
            data[i]
        );
    }

    // KIND_MAX+1 (=26) must still error.
    assert!(cuda_activation_unary(&x_kt, 26).is_err());
}

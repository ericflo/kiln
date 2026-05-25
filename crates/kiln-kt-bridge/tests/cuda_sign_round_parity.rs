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

    let cpu_in = Tensor::from_slice(data, vec![n]).unwrap();
    let cpu_out = op(&cpu_in).expect("cpu op");
    let cpu_vec = read_f32(&cpu_out);

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

// ---- direct cuda_activation_unary FFI smoke for KIND_RECIP -------------

#[test]
fn cuda_activation_unary_recip_direct_call() {
    // Confirm the FFI bounds-check accepts kind 22 (new KIND_MAX)
    // and rejects 23 (one past the current max).
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

    // KIND_RECIP = 22 (new max); should succeed for nonzero inputs.
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

    // KIND_MAX+1 (=23) must still error.
    assert!(cuda_activation_unary(&x_kt, 23).is_err());
}

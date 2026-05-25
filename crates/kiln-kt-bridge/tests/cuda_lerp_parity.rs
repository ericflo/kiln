//! Issue #1082: parity test for elementwise linear interpolation
//! `lerp(a, b, w) = a + w * (b - a)` on CUDA. Validates `cuda_lerp`
//! (and the transparent dispatch through `ops::lerp` when both
//! inputs live on CUDA) against the canonical CPU reference.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_lerp, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_lerp_ref(a: &[f32], b: &[f32], shape: &[usize], w: f32) -> Vec<f32> {
    let ta = Tensor::from_slice(a, shape.to_vec()).unwrap();
    let tb = Tensor::from_slice(b, shape.to_vec()).unwrap();
    let out = ops::lerp(&ta, &tb, w).unwrap();
    let cpu = out
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let n = out.element_count();
    let mut o = Vec::with_capacity(n);
    for i in 0..n {
        o.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    o
}

fn cast_data(data: &[f32], dtype: CandleDType) -> Vec<f32> {
    match dtype {
        CandleDType::F32 => data.to_vec(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype"),
    }
}

fn run_parity(shape: &[usize], weight: f32, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let a = pattern(n, 101);
    let b = pattern(n, 103);
    let shape_tuple: Vec<usize> = shape.to_vec();
    let a_cd = CandleTensor::from_vec(a.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = cuda_lerp(&a_kt, &b_kt, weight).expect("cuda_lerp");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let a_ref = cast_data(&a, dtype);
    let b_ref = cast_data(&b, dtype);
    let ref_vec = cpu_lerp_ref(&a_ref, &b_ref, shape, weight);

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(ref_vec.len(), got_vec.len(), "len mismatch");
    let mut max_abs = 0.0f32;
    for (x, y) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "lerp: shape={shape:?} w={weight} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

// --------------- F32 ----------------

#[test]
fn cuda_lerp_1d_f32_mid() {
    run_parity(&[1024], 0.5, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_lerp_2d_f32_quarter() {
    run_parity(&[8, 256], 0.25, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_lerp_2d_f32_extrap_above() {
    // Extrapolation past b (weight > 1).
    run_parity(&[8, 256], 1.5, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_lerp_2d_f32_extrap_below() {
    // Extrapolation below a (weight < 0).
    run_parity(&[8, 256], -0.5, CandleDType::F32, 1e-6);
}

#[test]
fn cuda_lerp_3d_f32() {
    run_parity(&[2, 16, 64], 0.7, CandleDType::F32, 1e-6);
}

// --------------- BF16 ----------------

#[test]
fn cuda_lerp_2d_bf16_mid() {
    run_parity(&[8, 256], 0.5, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_lerp_2d_bf16_ema_decay() {
    // EMA-style weight value — common in trainer code (decay = 0.99).
    run_parity(&[8, 256], 0.01, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_lerp_large_bf16() {
    run_parity(&[2, 1024, 64], 0.3, CandleDType::BF16, 1e-2);
}

// --------------- F16 ----------------

#[test]
fn cuda_lerp_2d_f16() {
    run_parity(&[8, 256], 0.4, CandleDType::F16, 1e-2);
}

// --------------- weight boundaries ----------------

#[test]
fn cuda_lerp_weight_zero_returns_a() {
    // lerp(a, b, 0) == a — sanity check the boundary.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 64;
    let a = pattern(n, 121);
    let b = pattern(n, 123);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let out_kt = cuda_lerp(&a_kt, &b_kt, 0.0).unwrap();
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
        assert!(
            (g - a[i]).abs() < 1e-6,
            "i={i}: weight=0 should return a; got {g}, want {}",
            a[i]
        );
    }
}

#[test]
fn cuda_lerp_weight_one_returns_b() {
    // lerp(a, b, 1) == b — sanity check the boundary.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 64;
    let a = pattern(n, 131);
    let b = pattern(n, 133);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let out_kt = cuda_lerp(&a_kt, &b_kt, 1.0).unwrap();
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
        // weight=1 *should* give exactly b, but the (a + (b - a)) form
        // has a tiny ULP drift relative to direct `b`. Tolerance picked
        // to absorb that single FMA-rounding step.
        assert!(
            (g - b[i]).abs() < 1e-5,
            "i={i}: weight=1 should return b; got {g}, want {}",
            b[i]
        );
    }
}

// --------------- dispatch through ops::lerp ----------------

#[test]
fn ops_lerp_dispatches_to_cuda() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 256;
    let a = pattern(n, 141);
    let b = pattern(n, 143);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    // Pre-#1082 this would error with "lerp: a must be CpuStorage" because
    // both tensors are CUDA-backed.
    let out_kt = ops::lerp(&a_kt, &b_kt, 0.3).expect("ops::lerp on CUDA");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_lerp_ref(&a, &b, &[n], 0.3);
    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (x, y) in ref_vec.iter().zip(got_vec.iter()) {
        assert!((x - y).abs() < 1e-6, "x={x}, y={y}");
    }
}

// --------------- FFI shape/dtype validation ----------------

#[test]
fn cuda_lerp_rejects_shape_mismatch() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_cd = CandleTensor::from_vec(pattern(16, 1), (16,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(pattern(8, 2), (8,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    assert!(cuda_lerp(&a_kt, &b_kt, 0.5).is_err());
}

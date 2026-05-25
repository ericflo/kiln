//! Issue #1082: parity test for elementwise `minimum` / `maximum`
//! on CUDA. Validates `cuda_binary_minmax` (and the transparent
//! dispatch through `ops::minimum` / `ops::maximum` when both inputs
//! live on CUDA) against the canonical CPU reference.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_binary_minmax, ops, Tensor};

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

fn cpu_minmax_ref(a: &[f32], b: &[f32], shape: &[usize], is_min: bool) -> Vec<f32> {
    let ta = Tensor::from_slice(a, shape.to_vec()).unwrap();
    let tb = Tensor::from_slice(b, shape.to_vec()).unwrap();
    let out = if is_min {
        ops::minimum(&ta, &tb).unwrap()
    } else {
        ops::maximum(&ta, &tb).unwrap()
    };
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

fn run_parity(
    shape: &[usize],
    dtype: CandleDType,
    is_min: bool,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let a = pattern(n, if is_min { 61 } else { 63 });
    let b = pattern(n, if is_min { 71 } else { 73 });
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

    let kind = if is_min { 0 } else { 1 };
    let out_kt = cuda_binary_minmax(&a_kt, &b_kt, kind).expect("cuda_binary_minmax");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let a_ref = cast_data(&a, dtype);
    let b_ref = cast_data(&b, dtype);
    let ref_vec = cpu_minmax_ref(&a_ref, &b_ref, shape, is_min);

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(ref_vec.len(), got_vec.len(), "len mismatch");
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let label = if is_min { "minimum" } else { "maximum" };
    assert!(
        max_abs < tolerance,
        "{label}: shape={shape:?} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

// --------------- minimum ----------------

#[test]
fn cuda_minimum_1d_f32() {
    run_parity(&[1024], CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_minimum_2d_f32() {
    run_parity(&[8, 256], CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_minimum_3d_bf16() {
    run_parity(&[4, 8, 16], CandleDType::BF16, true, 1e-2);
}

#[test]
fn cuda_minimum_f16() {
    run_parity(&[128], CandleDType::F16, true, 1e-3);
}

#[test]
fn cuda_minimum_large_f32() {
    run_parity(&[2, 1024, 64], CandleDType::F32, true, 1e-6);
}

// --------------- maximum ----------------

#[test]
fn cuda_maximum_1d_f32() {
    run_parity(&[1024], CandleDType::F32, false, 1e-6);
}

#[test]
fn cuda_maximum_2d_f32() {
    run_parity(&[8, 256], CandleDType::F32, false, 1e-6);
}

#[test]
fn cuda_maximum_3d_bf16() {
    run_parity(&[4, 8, 16], CandleDType::BF16, false, 1e-2);
}

#[test]
fn cuda_maximum_f16() {
    run_parity(&[128], CandleDType::F16, false, 1e-3);
}

#[test]
fn cuda_maximum_large_f32() {
    run_parity(&[2, 1024, 64], CandleDType::F32, false, 1e-6);
}

// --------------- exact ties (semantics) ----------------

#[test]
fn cuda_minimum_picks_smaller_explicit_ties() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Picks the smaller value when both differ; matches at parity returns
    // the canonical (both equal) value.
    let a_data: Vec<f32> = vec![1.0, 5.0, 3.0, 7.0, 0.0, -1.0];
    let b_data: Vec<f32> = vec![2.0, 4.0, 6.0, 0.5, 0.0, -2.0];
    let n = a_data.len();
    let a_cd = CandleTensor::from_vec(a_data.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = cuda_binary_minmax(&a_kt, &b_kt, 0).unwrap();
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
    let want: Vec<f32> = vec![1.0, 4.0, 3.0, 0.5, 0.0, -2.0];
    for (i, &g) in got.iter().enumerate() {
        assert!(
            (g - want[i]).abs() < 1e-6,
            "i={i}: got={g}, want={}",
            want[i]
        );
    }
}

// --------------- dispatch through ops::minimum / ops::maximum ----------------

#[test]
fn ops_minimum_dispatches_to_cuda() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 256;
    let a = pattern(n, 81);
    let b = pattern(n, 83);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::minimum(&a_kt, &b_kt).expect("ops::minimum on CUDA");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_minmax_ref(&a, &b, &[n], true);
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

#[test]
fn ops_maximum_dispatches_to_cuda() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 256;
    let a = pattern(n, 85);
    let b = pattern(n, 87);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::maximum(&a_kt, &b_kt).expect("ops::maximum on CUDA");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_minmax_ref(&a, &b, &[n], false);
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

// --------------- FFI bounds check ----------------

#[test]
fn cuda_binary_minmax_rejects_invalid_kind() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 16;
    let a = pattern(n, 91);
    let b = pattern(n, 93);
    let a_cd = CandleTensor::from_vec(a.clone(), (n,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), (n,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    // kind=2 is past the max (max=1).
    assert!(cuda_binary_minmax(&a_kt, &b_kt, 2).is_err());
    // kind=-1 also rejected.
    assert!(cuda_binary_minmax(&a_kt, &b_kt, -1).is_err());
}

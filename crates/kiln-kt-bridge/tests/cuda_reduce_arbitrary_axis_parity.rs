//! Issue #1082: parity test for sum/mean reductions over any axis,
//! not just the last one. Validates `ReduceOp::cuda_fwd` /
//! `cuda_sum_axis` / `cuda_mean_axis` against the canonical CPU
//! reference in `ops::sum_axis` / `ops::mean_axis`.
//!
//! Before #1082, the CUDA path only handled `axis == rank - 1`.
//! Non-last-axis fell through to CPU. These tests prove the new
//! arbitrary-axis kernel produces matching results.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_mean_axis, cuda_sum_axis, ops, Tensor};

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

fn cpu_sum_ref(data: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape.to_vec()).unwrap();
    let s = ops::sum_axis(&x, axis).unwrap();
    let cpu_storage = s
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let n = s.element_count();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
}

fn cpu_mean_ref(data: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape.to_vec()).unwrap();
    let m = ops::mean_axis(&x, axis).unwrap();
    let cpu_storage = m
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let n = m.element_count();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
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

fn run_sum_parity(shape: &[usize], axis: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 41);
    let shape_tuple: Vec<usize> = shape.to_vec();
    let x_cd = CandleTensor::from_vec(data.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_sum_axis(&x_kt, axis).expect("sum_axis");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_data = cast_data(&data, dtype);
    let ref_vec = cpu_sum_ref(&ref_data, shape, axis);

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(
        ref_vec.len(),
        got_vec.len(),
        "shape={shape:?} axis={axis} dtype={dtype:?}: len mismatch"
    );
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "sum_axis: shape={shape:?} axis={axis} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

fn run_mean_parity(shape: &[usize], axis: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 43);
    let shape_tuple: Vec<usize> = shape.to_vec();
    let x_cd = CandleTensor::from_vec(data.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_mean_axis(&x_kt, axis).expect("mean_axis");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_data = cast_data(&data, dtype);
    let ref_vec = cpu_mean_ref(&ref_data, shape, axis);

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(
        ref_vec.len(),
        got_vec.len(),
        "shape={shape:?} axis={axis} dtype={dtype:?}: len mismatch"
    );
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "mean_axis: shape={shape:?} axis={axis} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

// --------------- sum: non-last axes ----------------

#[test]
fn cuda_sum_axis0_2d_f32() {
    // [3, 4] sum axis 0 → [4]
    run_sum_parity(&[3, 4], 0, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_sum_axis0_3d_f32() {
    // [2, 3, 4] sum axis 0 → [3, 4]
    run_sum_parity(&[2, 3, 4], 0, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_sum_axis1_3d_f32() {
    // [2, 3, 4] sum axis 1 → [2, 4]
    run_sum_parity(&[2, 3, 4], 1, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_sum_axis0_3d_bf16() {
    run_sum_parity(&[2, 3, 4], 0, CandleDType::BF16, 5e-2);
}

#[test]
fn cuda_sum_axis1_3d_bf16() {
    run_sum_parity(&[2, 3, 4], 1, CandleDType::BF16, 5e-2);
}

#[test]
fn cuda_sum_axis1_large_bf16() {
    // Exercise the strided reduction with a larger axis_dim.
    run_sum_parity(&[8, 128, 16], 1, CandleDType::BF16, 5e-1);
}

#[test]
fn cuda_sum_axis_last_still_works() {
    // Sanity: axis == rank-1 still routes correctly.
    run_sum_parity(&[2, 3, 4], 2, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_sum_axis0_4d_f32() {
    run_sum_parity(&[2, 3, 4, 5], 0, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_sum_axis2_4d_f32() {
    run_sum_parity(&[2, 3, 4, 5], 2, CandleDType::F32, 1e-3);
}

// --------------- mean: non-last axes ----------------

#[test]
fn cuda_mean_axis0_3d_f32() {
    run_mean_parity(&[2, 3, 4], 0, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_mean_axis1_3d_f32() {
    run_mean_parity(&[2, 3, 4], 1, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_mean_axis0_3d_bf16() {
    run_mean_parity(&[2, 3, 4], 0, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_mean_axis1_3d_bf16() {
    run_mean_parity(&[2, 3, 4], 1, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_mean_axis2_4d_f32() {
    run_mean_parity(&[2, 3, 4, 5], 2, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_mean_axis1_f16() {
    run_mean_parity(&[2, 3, 4], 1, CandleDType::F16, 1e-2);
}

// --------------- dispatch through ReduceOp ----------------

#[test]
fn cuda_reduce_op_dispatches_for_non_last_axis() {
    use kiln_tensor::dispatch1;
    use kiln_tensor::ops::reduce::{ReduceOp, ReductionKind, ReductionScope};

    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let data = pattern(n, 77);
    let x_cd = CandleTensor::from_vec(data.clone(), (shape[0], shape[1], shape[2]), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // Reduce axis 1 (the middle axis) — pre-#1082 this would fall
    // back to CPU and fail because the tensor is on CUDA.
    let op = ReduceOp::new(ReductionKind::Sum, ReductionScope::Axis(1));
    let out_kt = dispatch1(&op, &x_kt).expect("ReduceOp axis=1");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_sum_ref(&data, &shape, 1);
    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(ref_vec.len(), got_vec.len());
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        assert!((a - b).abs() < 1e-4, "mismatch a={a} b={b}");
    }
}

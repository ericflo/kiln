//! Issue #1082: parity test for min/max reductions over any axis.
//! Validates `cuda_min_axis` / `cuda_max_axis` against the canonical
//! CPU reference in `ops::min_axis` / `ops::max_axis`, and the
//! transparent dispatch from `ops::max_axis` / `ops::min_axis` when
//! given a CUDA-backed tensor.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_max_axis, cuda_min_axis, ops, Tensor};

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

fn cpu_minmax_ref(data: &[f32], shape: &[usize], axis: usize, is_min: bool) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape.to_vec()).unwrap();
    let s = if is_min {
        ops::min_axis(&x, axis).unwrap()
    } else {
        ops::max_axis(&x, axis).unwrap()
    };
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

fn run_minmax_parity(
    shape: &[usize],
    axis: usize,
    dtype: CandleDType,
    is_min: bool,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, if is_min { 51 } else { 53 });
    let shape_tuple: Vec<usize> = shape.to_vec();
    let x_cd = CandleTensor::from_vec(data.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = if is_min {
        cuda_min_axis(&x_kt, axis).expect("cuda_min_axis")
    } else {
        cuda_max_axis(&x_kt, axis).expect("cuda_max_axis")
    };
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_data = cast_data(&data, dtype);
    let ref_vec = cpu_minmax_ref(&ref_data, shape, axis, is_min);

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(
        ref_vec.len(),
        got_vec.len(),
        "shape={shape:?} axis={axis} dtype={dtype:?} is_min={is_min}: len mismatch"
    );
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let label = if is_min { "min_axis" } else { "max_axis" };
    assert!(
        max_abs < tolerance,
        "{label}: shape={shape:?} axis={axis} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

// --------------- min: non-last axes ----------------

#[test]
fn cuda_min_axis0_2d_f32() {
    run_minmax_parity(&[3, 4], 0, CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_min_axis0_3d_f32() {
    run_minmax_parity(&[2, 3, 4], 0, CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_min_axis1_3d_f32() {
    run_minmax_parity(&[2, 3, 4], 1, CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_min_last_axis_f32() {
    run_minmax_parity(&[2, 3, 4], 2, CandleDType::F32, true, 1e-6);
}

#[test]
fn cuda_min_axis1_3d_bf16() {
    run_minmax_parity(&[2, 3, 4], 1, CandleDType::BF16, true, 1e-2);
}

#[test]
fn cuda_min_axis0_large_bf16() {
    run_minmax_parity(&[8, 128, 16], 0, CandleDType::BF16, true, 1e-2);
}

#[test]
fn cuda_min_axis2_4d_f16() {
    run_minmax_parity(&[2, 3, 4, 5], 2, CandleDType::F16, true, 1e-2);
}

// --------------- max: non-last axes ----------------

#[test]
fn cuda_max_axis0_2d_f32() {
    run_minmax_parity(&[3, 4], 0, CandleDType::F32, false, 1e-6);
}

#[test]
fn cuda_max_axis1_3d_f32() {
    run_minmax_parity(&[2, 3, 4], 1, CandleDType::F32, false, 1e-6);
}

#[test]
fn cuda_max_last_axis_f32() {
    run_minmax_parity(&[2, 3, 4], 2, CandleDType::F32, false, 1e-6);
}

#[test]
fn cuda_max_axis0_3d_bf16() {
    run_minmax_parity(&[2, 3, 4], 0, CandleDType::BF16, false, 1e-2);
}

#[test]
fn cuda_max_axis1_large_bf16() {
    run_minmax_parity(&[8, 128, 16], 1, CandleDType::BF16, false, 1e-2);
}

#[test]
fn cuda_max_axis2_4d_f16() {
    run_minmax_parity(&[2, 3, 4, 5], 2, CandleDType::F16, false, 1e-2);
}

// --------------- dispatch through ops::min_axis / ops::max_axis ----------------

#[test]
fn ops_min_axis_dispatches_to_cuda() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let data = pattern(n, 91);
    let x_cd = CandleTensor::from_vec(data.clone(), (shape[0], shape[1], shape[2]), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // Calls `ops::min_axis(t, 1)` where `t` is a CUDA-backed tensor —
    // the CUDA fast path inside `max_min_axis::apply` should route
    // straight through `cuda_min_axis`. (Pre-#1082 this returned a
    // "storage must be CpuStorage" error.)
    let out_kt = ops::min_axis(&x_kt, 1).expect("ops::min_axis on CUDA");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_minmax_ref(&data, &shape, 1, true);
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
        assert!((a - b).abs() < 1e-5, "mismatch a={a} b={b}");
    }
}

#[test]
fn ops_max_axis_dispatches_to_cuda() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let data = pattern(n, 93);
    let x_cd = CandleTensor::from_vec(data.clone(), (shape[0], shape[1], shape[2]), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::max_axis(&x_kt, 1).expect("ops::max_axis on CUDA");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_minmax_ref(&data, &shape, 1, false);
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
        assert!((a - b).abs() < 1e-5, "mismatch a={a} b={b}");
    }
}

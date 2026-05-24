//! Parity test: kt CUDA reduce-last-axis (`ReduceOp::cuda_fwd` /
//! `cuda_sum_last_axis` / `cuda_mean_last_axis`) vs kt CPU reference
//! (`ops::sum_axis` / `ops::mean_axis`).
//!
//! Phase 4 substrate validation. Confirms the kernels in
//! `csrc/reduce_last_axis.cu` (sum + mean over the trailing axis)
//! produce per-row results matching the canonical CPU reference.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_mean_last_axis, cuda_sum_last_axis, ops, Tensor};

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

fn cpu_sum_axis(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let x = Tensor::from_slice(data, vec![n_rows, n_cols]).unwrap();
    let s = ops::sum_axis(&x, 1).unwrap();
    let cpu_storage = s
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
}

fn cpu_mean_axis(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let x = Tensor::from_slice(data, vec![n_rows, n_cols]).unwrap();
    let m = ops::mean_axis(&x, 1).unwrap();
    let cpu_storage = m
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
}

fn run_sum_parity(n_rows: usize, n_cols: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let data = pattern(n, 41);
    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_sum_last_axis(&x_kt).expect("sum");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_data: Vec<f32> = match dtype {
        CandleDType::F32 => data.clone(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype"),
    };
    let ref_vec = cpu_sum_axis(&ref_data, n_rows, n_cols);

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "sum: rows={n_rows} cols={n_cols} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

fn run_mean_parity(n_rows: usize, n_cols: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let data = pattern(n, 43);
    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_mean_last_axis(&x_kt).expect("mean");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_data: Vec<f32> = match dtype {
        CandleDType::F32 => data.clone(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype"),
    };
    let ref_vec = cpu_mean_axis(&ref_data, n_rows, n_cols);

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "mean: rows={n_rows} cols={n_cols} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

// --------------- sum ----------------

#[test]
fn cuda_sum_last_axis_f32_4_rows_512_cols() {
    run_sum_parity(4, 512, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_sum_last_axis_bf16_8_rows_128_cols() {
    run_sum_parity(8, 128, CandleDType::BF16, 5e-1);
}

#[test]
fn cuda_sum_last_axis_bf16_2_rows_2048_cols() {
    // Larger row size — exercises the strided per-thread reduction.
    run_sum_parity(2, 2048, CandleDType::BF16, 2.0);
}

// --------------- mean ----------------

#[test]
fn cuda_mean_last_axis_f32_4_rows_512_cols() {
    run_mean_parity(4, 512, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_mean_last_axis_bf16_8_rows_128_cols() {
    run_mean_parity(8, 128, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_mean_last_axis_bf16_2_rows_2048_cols() {
    run_mean_parity(2, 2048, CandleDType::BF16, 1e-2);
}

// --------------- dispatch through ReduceOp ----------------

#[test]
fn cuda_sum_dispatches_through_ops_sum_axis_last() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4usize;
    let n_cols = 128usize;
    let data = pattern(n_rows * n_cols, 53);

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::sum_axis(&x_kt, 1).expect("dispatch sum_axis");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let ref_vec = cpu_sum_axis(&data, n_rows, n_cols);
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        assert!((a - b).abs() < 1e-3, "cpu={a} cuda={b}");
    }
}

#[test]
fn cuda_mean_dispatches_through_ops_mean_axis_last() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4usize;
    let n_cols = 256usize;
    let data = pattern(n_rows * n_cols, 59);

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::mean_axis(&x_kt, 1).expect("dispatch mean_axis");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let ref_vec = cpu_mean_axis(&data, n_rows, n_cols);
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        assert!((a - b).abs() < 1e-5, "cpu={a} cuda={b}");
    }
}

#[test]
fn cuda_sum_3d_drops_trailing_axis() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // shape [2, 3, 4] -> sum_axis(-1) -> [2, 3]
    let n = 2 * 3 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data.clone(), (2, 3, 4), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_sum_last_axis(&x_kt).expect("sum");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(out_kt.shape(), &[2usize, 3usize]);
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((6,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Row sums of [0..24] taken in chunks of 4:
    // 0+1+2+3=6, 4+5+6+7=22, 8+9+10+11=38, 12+13+14+15=54, 16+17+18+19=70, 20+21+22+23=86
    let expected = vec![6.0f32, 22.0, 38.0, 54.0, 70.0, 86.0];
    for (a, b) in expected.iter().zip(got.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {a}, got {b}");
    }
}

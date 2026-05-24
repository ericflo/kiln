//! Parity test: kt CUDA softmax_last_axis vs kt CPU softmax_last_dim.
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/softmax.cu` produces per-row probability distributions
//! matching the canonical CPU reference (kt's own naive triple-loop).

#![cfg(feature = "cuda")]

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_softmax_last_axis, ops, Tensor};

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

fn cpu_reference_f32(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    // kt CPU softmax via ops::softmax_last_dim. Build a CPU kt-Tensor
    // first, then run the op.
    let x = Tensor::from_slice(data, vec![n_rows, n_cols]).unwrap();
    let y = ops::softmax_last_dim(&x).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..(n_rows * n_cols) {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

fn run_softmax_parity(n_rows: usize, n_cols: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let data = pattern(n, 11);

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = cuda_softmax_last_axis(&x_kt).expect("softmax");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Reference: kt CPU softmax over the same F32 values.
    let ref_vec = cpu_reference_f32(&data, n_rows, n_cols);

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
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
        "rows={n_rows} cols={n_cols} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );

    // Each row should sum to ~1.
    for row in 0..n_rows {
        let row_sum: f32 = got_vec[row * n_cols..(row + 1) * n_cols].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-2,
            "row {row} sum = {row_sum}, expected ≈ 1.0"
        );
    }
}

#[test]
fn cuda_softmax_bf16_8_rows_64_cols() {
    run_softmax_parity(8, 64, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_softmax_f32_4_rows_512_cols() {
    run_softmax_parity(4, 512, CandleDType::F32, 1e-5);
}

#[test]
fn cuda_softmax_bf16_2_rows_2048_cols() {
    // Larger row size — exercises the strided per-thread reduction
    // when n_cols > MAX_THREADS.
    run_softmax_parity(2, 2048, CandleDType::BF16, 1e-2);
}

#[test]
fn cuda_softmax_dispatches_through_ops_softmax_last_dim() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4;
    let n_cols = 256;
    let x_cd = CandleTensor::from_vec(pattern(n_rows * n_cols, 21), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::softmax_last_dim(&x_kt).expect("dispatch");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows * n_cols,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for row in 0..n_rows {
        let s: f32 = got_vec[row * n_cols..(row + 1) * n_cols].iter().sum();
        assert!((s - 1.0).abs() < 1e-2, "row sum = {s}");
    }
}

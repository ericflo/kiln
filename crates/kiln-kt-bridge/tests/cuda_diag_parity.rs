//! Parity tests for the CUDA `diagonal` extract / `diag` construct
//! kernels vs CPU reference.
//!
//! The CUDA kernels (`csrc/diag.cu`) do per-element byte copies of
//! the diagonal entries. The CPU reference does the same scalar
//! copy in `ops::diag`. Parity should be bit-exact since no math is
//! performed. (#1082)

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::ops;

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 4096) as f32 - 2048.0) / 256.0;
        out.push(f);
    }
    out
}

#[test]
fn cuda_diagonal_f32_extracts_main_diagonal() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 5;
    let data: Vec<f32> = (0..(n * n)).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data, (n, n), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::diagonal(&x_kt).expect("diagonal");
    assert_eq!(out_kt.shape(), &[n]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // For row-major 5x5 with values 0..25, diagonal is [0, 6, 12, 18, 24].
    assert_eq!(got, vec![0.0, 6.0, 12.0, 18.0, 24.0]);
}

#[test]
fn cuda_diagonal_f32_parity_vs_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 32;
    let data = pattern_f32(n * n, 42);

    // CPU reference
    let x_cpu = kiln_tensor::Tensor::from_slice(&data, vec![n, n]).unwrap();
    let out_cpu = ops::diagonal(&x_cpu).unwrap();
    let want: Vec<f32> = out_cpu
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap()
        .as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // CUDA path
    let x_cd = CandleTensor::from_vec(data, (n, n), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let out_kt = ops::diagonal(&x_kt).unwrap();

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want);
}

#[test]
fn cuda_diag_f32_builds_diagonal_matrix() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let v_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let v_cd = CandleTensor::from_vec(v_data.clone(), (4,), &dev).unwrap();
    let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_cd).unwrap();

    let out_kt = ops::diag(&v_kt).expect("diag");
    assert_eq!(out_kt.shape(), &[4, 4]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Expected 4x4 matrix with [1, 2, 3, 4] on the diagonal.
    let mut want = vec![0.0f32; 16];
    for i in 0..4 {
        want[i * 4 + i] = v_data[i];
    }
    assert_eq!(got, want);
}

#[test]
fn cuda_diag_f32_parity_vs_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 19;
    let data = pattern_f32(n, 99);

    // CPU reference
    let v_cpu = kiln_tensor::Tensor::from_slice(&data, vec![n]).unwrap();
    let out_cpu = ops::diag(&v_cpu).unwrap();
    let want: Vec<f32> = out_cpu
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap()
        .as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // CUDA path
    let v_cd = CandleTensor::from_vec(data, (n,), &dev).unwrap();
    let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_cd).unwrap();
    let out_kt = ops::diag(&v_kt).unwrap();
    assert_eq!(out_kt.shape(), &[n, n]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want);
}

#[test]
fn cuda_diag_bf16_parity_vs_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 8;
    let data = pattern_f32(n, 31);

    // CPU reference (bf16).
    let v_bf16: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let v_cpu = kiln_tensor::Tensor::from_slice(&v_bf16, vec![n]).unwrap();
    let out_cpu = ops::diag(&v_cpu).unwrap();
    let cpu_bytes = out_cpu
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap()
        .as_bytes()
        .to_vec();
    let want: Vec<f32> = cpu_bytes
        .chunks(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();

    // CUDA path.
    let v_cd = CandleTensor::from_vec(v_bf16, (n,), &dev).unwrap();
    let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_cd).unwrap();
    let out_kt = ops::diag(&v_kt).unwrap();
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::BF16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want);
}

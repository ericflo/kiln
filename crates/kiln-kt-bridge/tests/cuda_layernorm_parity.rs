//! Parity test: kt CUDA LayerNorm (`LayerNormOp::cuda_fwd` /
//! `cuda_layernorm_last_axis`) vs kt CPU reference (`ops::layer_norm`).
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/layernorm.cu` (per-row mean + variance, then per-element
//! scale by `weight` plus `bias`) produces outputs matching the
//! canonical CPU reference.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_layernorm_last_axis, ops, Tensor};

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

fn pattern_weight(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = 0.5 + ((s as u32 % 1024) as f32) / 1024.0;
        out.push(f);
    }
    out
}

fn pattern_bias(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // Small offsets in [-0.5, 0.5).
        let f = ((s as u32 % 1024) as f32 - 512.0) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_reference(
    x_data: &[f32],
    w_data: &[f32],
    b_data: &[f32],
    n_rows: usize,
    n_cols: usize,
    eps: f32,
) -> Vec<f32> {
    let x = Tensor::from_slice(x_data, vec![n_rows, n_cols]).unwrap();
    let w = Tensor::from_slice(w_data, vec![n_cols]).unwrap();
    let b = Tensor::from_slice(b_data, vec![n_cols]).unwrap();
    let y = ops::layer_norm(&x, &w, &b, eps).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..(n_rows * n_cols) {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
}

fn run_layernorm_parity(
    n_rows: usize,
    n_cols: usize,
    dtype: CandleDType,
    eps: f32,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let x_data = pattern(n, 13);
    let w_data = pattern_weight(n_cols, 47);
    let b_data = pattern_bias(n_cols, 89);

    let x_cd = CandleTensor::from_vec(x_data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let w_cd = CandleTensor::from_vec(w_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = cuda_layernorm_last_axis(&x_kt, &w_kt, &b_kt, eps).expect("layernorm");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_reference(&x_data, &w_data, &b_data, n_rows, n_cols, eps);

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
        "rows={n_rows} cols={n_cols} dtype={dtype:?} eps={eps} max_abs={max_abs} > {tolerance}"
    );
}

#[test]
fn cuda_layernorm_f32_4_rows_512_cols() {
    run_layernorm_parity(4, 512, CandleDType::F32, 1e-6, 1e-4);
}

#[test]
fn cuda_layernorm_f32_8_rows_1024_cols() {
    // BERT-base hidden dim (768) and BERT-large hidden dim (1024)
    // are both realistic LayerNorm sizes.
    run_layernorm_parity(8, 1024, CandleDType::F32, 1e-6, 1e-4);
}

#[test]
fn cuda_layernorm_bf16_8_rows_64_cols() {
    run_layernorm_parity(8, 64, CandleDType::BF16, 1e-6, 5e-2);
}

#[test]
fn cuda_layernorm_bf16_2_rows_2048_cols() {
    // Stress: large rows force the strided per-thread reduction.
    run_layernorm_parity(2, 2048, CandleDType::BF16, 1e-6, 5e-2);
}

#[test]
fn cuda_layernorm_dispatches_through_ops_layer_norm() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4;
    let n_cols = 256;
    let eps = 1e-6_f32;
    let x_data = pattern(n_rows * n_cols, 23);
    let w_data = pattern_weight(n_cols, 71);
    let b_data = pattern_bias(n_cols, 103);
    let x_cd = CandleTensor::from_vec(x_data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let w_cd = CandleTensor::from_vec(w_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    // ops::layer_norm should pick the CUDA path automatically via
    // LayerNormOp::cuda_fwd through dispatch3.
    let out_kt = ops::layer_norm(&x_kt, &w_kt, &b_kt, eps).expect("dispatch");

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

    let ref_vec = cpu_reference(&x_data, &w_data, &b_data, n_rows, n_cols, eps);
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(max_abs < 5e-2, "dispatch parity max_abs={max_abs}");
}

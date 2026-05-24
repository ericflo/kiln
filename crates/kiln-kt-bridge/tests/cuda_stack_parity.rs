//! Parity test: kt CUDA stack (`ops::stack` on all-CUDA inputs) vs kt
//! CPU reference (`ops::stack` on CPU tensors).
//!
//! The CUDA fast path in `ops::stack` composes `unsqueeze(axis)` +
//! `cuda_concat(axis)` — reuses the existing per-axis concat kernel,
//! no new CUDA code. Parity should be exact for all dtypes since the
//! kernel does byte-wise copies. (#1082)

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

/// CPU reference via `ops::stack`, return flat F32.
fn cpu_reference_f32(inputs: &[(Vec<usize>, Vec<f32>)], axis: usize) -> Vec<f32> {
    let kt_tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(shape, data)| Tensor::from_slice(data, shape.clone()).unwrap())
        .collect();
    let refs: Vec<&Tensor> = kt_tensors.iter().collect();
    let y = ops::stack(&refs, axis).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let n: usize = y.shape().iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

#[test]
fn cuda_stack_two_rank1_axis_0_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // a [3] + b [3] stacked axis=0 → [2, 3].
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b_data: Vec<f32> = vec![4.0, 5.0, 6.0];
    let a_cd = CandleTensor::from_vec(a_data.clone(), (3,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (3,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt, &b_kt], 0).expect("stack axis=0");
    assert_eq!(out_kt.shape(), &[2, 3]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[(vec![3], a_data.clone()), (vec![3], b_data.clone())],
        0,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_stack_two_rank1_axis_1_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // a [3] + b [3] stacked axis=1 → [3, 2].
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0];
    let a_cd = CandleTensor::from_vec(a_data.clone(), (3,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (3,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt, &b_kt], 1).expect("stack axis=1");
    assert_eq!(out_kt.shape(), &[3, 2]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[(vec![3], a_data.clone()), (vec![3], b_data.clone())],
        1,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_stack_three_rank2_axis_0_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data = pattern_f32(2 * 3, 11);
    let b_data = pattern_f32(2 * 3, 13);
    let c_data = pattern_f32(2 * 3, 17);
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 3), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (2, 3), &dev).unwrap();
    let c_cd = CandleTensor::from_vec(c_data.clone(), (2, 3), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let c_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&c_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt, &b_kt, &c_kt], 0).expect("stack 3-way axis=0");
    assert_eq!(out_kt.shape(), &[3, 2, 3]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[
            (vec![2, 3], a_data),
            (vec![2, 3], b_data),
            (vec![2, 3], c_data),
        ],
        0,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_stack_two_rank2_axis_middle_f32() {
    // 2 inputs of shape [2, 3] stacked at axis 1 → [2, 2, 3].
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data = pattern_f32(2 * 3, 41);
    let b_data = pattern_f32(2 * 3, 43);
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 3), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (2, 3), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt, &b_kt], 1).expect("stack axis=middle");
    assert_eq!(out_kt.shape(), &[2, 2, 3]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[(vec![2, 3], a_data), (vec![2, 3], b_data)],
        1,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_stack_bf16_axis_0() {
    // BF16 path — same byte-wise copy, but goes through the BF16
    // concat kernel branch.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data = pattern_f32(4 * 8, 71);
    let b_data = pattern_f32(4 * 8, 73);
    let a_cd = CandleTensor::from_vec(a_data.clone(), (4, 8), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (4, 8), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt, &b_kt], 0).expect("stack bf16");
    assert_eq!(out_kt.shape(), &[2, 4, 8]);
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::BF16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Quantize reference through BF16.
    let bf16_to_f32 = |v: &f32| -> f32 { half::bf16::from_f32(*v).to_f32() };
    let a_bf16: Vec<f32> = a_data.iter().map(bf16_to_f32).collect();
    let b_bf16: Vec<f32> = b_data.iter().map(bf16_to_f32).collect();
    let want = cpu_reference_f32(
        &[(vec![4, 8], a_bf16), (vec![4, 8], b_bf16)],
        0,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_stack_single_input_adds_singleton_axis() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_cd = CandleTensor::from_vec(a_data.clone(), (4,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();

    let out_kt = ops::stack(&[&a_kt], 0).expect("stack singleton");
    assert_eq!(out_kt.shape(), &[1, 4]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((4,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a_data, got);
}

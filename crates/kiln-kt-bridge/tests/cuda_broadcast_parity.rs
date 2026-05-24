//! Parity test: kt CUDA `broadcast_to` vs kt CPU reference.
//!
//! Issue #1082 — sub-phase: broadcast cuda_fwd wiring.
//! `BroadcastOp::cuda_fwd` flattens the input to 1D, builds a
//! flat-output → flat-input gather map on the host, ships it to
//! CUDA, and uses `cuda_index_select_dim0` on the flattened input.
//! This confirms the GPU path produces byte-identical output to the
//! canonical CPU per-element byte-copy loop in
//! `crates/kiln-tensor/src/ops/broadcast.rs`.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, Tensor};

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

fn cpu_reference_f32(data: &[f32], in_shape: Vec<usize>, target_shape: &[usize]) -> Vec<f32> {
    let x = Tensor::from_slice(data, in_shape).unwrap();
    let y = ops::broadcast_to(&x, target_shape).unwrap();
    let cpu = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

fn run_broadcast_parity(in_shape: Vec<usize>, target_shape: Vec<usize>, dtype: CandleDType) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = in_shape.iter().product();
    let data = pattern(n, 31);

    let x_cd = match in_shape.as_slice() {
        [a] => CandleTensor::from_vec(data.clone(), (*a,), &dev).unwrap(),
        [a, b] => CandleTensor::from_vec(data.clone(), (*a, *b), &dev).unwrap(),
        [a, b, c] => CandleTensor::from_vec(data.clone(), (*a, *b, *c), &dev).unwrap(),
        other => panic!("unsupported test shape {other:?}"),
    };
    let x_cd = x_cd.to_dtype(dtype).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt =
        ops::broadcast_to(&x_kt, &target_shape).expect("broadcast_to dispatch");
    assert_eq!(out_kt.shape(), target_shape.as_slice());

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n_out: usize = target_shape.iter().product();
    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_out,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference at the same cast dtype to match BF16/F16 precision.
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
        other => panic!("unsupported test dtype {other:?}"),
    };
    let ref_v = cpu_reference_f32(&ref_data, in_shape, &target_shape);
    let tol = match dtype {
        CandleDType::F32 => 0.0f32,
        CandleDType::BF16 => 1e-2f32,
        CandleDType::F16 => 1e-3f32,
        _ => 0.0,
    };
    assert_eq!(got_vec.len(), ref_v.len());
    for (i, (a, b)) in ref_v.iter().zip(got_vec.iter()).enumerate() {
        if tol == 0.0 {
            assert_eq!(a, b, "idx={i} ref={a} got={b}");
        } else {
            assert!((a - b).abs() <= tol, "idx={i} ref={a} got={b} tol={tol}");
        }
    }
}

#[test]
fn cuda_broadcast_rank1_size1_to_3_f32() {
    run_broadcast_parity(vec![1], vec![3], CandleDType::F32);
}

#[test]
fn cuda_broadcast_rank2_axis0_only_f32() {
    // [1, 4] -> [3, 4]: replicate row 3 times.
    run_broadcast_parity(vec![1, 4], vec![3, 4], CandleDType::F32);
}

#[test]
fn cuda_broadcast_rank2_axis1_only_f32() {
    // [3, 1] -> [3, 4]: replicate column 4 times.
    run_broadcast_parity(vec![3, 1], vec![3, 4], CandleDType::F32);
}

#[test]
fn cuda_broadcast_rank2_both_axes_f32() {
    // [1, 1] -> [3, 4]: replicate single element across all positions.
    run_broadcast_parity(vec![1, 1], vec![3, 4], CandleDType::F32);
}

#[test]
fn cuda_broadcast_rank3_f32() {
    // [1, 4, 1] -> [3, 4, 5]: replicate along axes 0 and 2.
    run_broadcast_parity(vec![1, 4, 1], vec![3, 4, 5], CandleDType::F32);
}

#[test]
fn cuda_broadcast_bf16() {
    // [1, 3] -> [4, 3], BF16 path.
    run_broadcast_parity(vec![1, 3], vec![4, 3], CandleDType::BF16);
}

#[test]
fn cuda_broadcast_f16() {
    // [2, 1] -> [2, 5], F16 path.
    run_broadcast_parity(vec![2, 1], vec![2, 5], CandleDType::F16);
}

#[test]
fn cuda_broadcast_identity_shape_match() {
    // No expansion — shapes match exactly. Should be a zero-data-movement
    // reshape on the cuda_fwd fast path.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data, (2, 3), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::broadcast_to(&x_kt, &[2, 3]).expect("broadcast identity");
    assert_eq!(out_kt.shape(), &[2, 3]);
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::F32);
    assert!(matches!(out_kt.device(), kiln_tensor::Device::Cuda(_)));
}

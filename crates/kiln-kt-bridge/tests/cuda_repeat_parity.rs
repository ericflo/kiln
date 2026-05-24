//! Parity test: kt CUDA `repeat(axis=0, n)` vs kt CPU reference.
//!
//! Phase 4 substrate validation. `RepeatOp::cuda_fwd` routes through
//! `cuda_index_select_dim0` with a tiled
//! `[0,1,…,d-1, 0,1,…,d-1, …]` index buffer; this confirms the
//! gather produces byte-identical output to the canonical CPU
//! per-slab byte-copy loop in
//! `crates/kiln-tensor/src/ops/repeat.rs`.
//!
//! Non-zero-axis repeats fall through to CPU on the CUDA path (the
//! standalone DeviceOp returns `Ok(None)` for those; `dispatch1`
//! then re-runs the CPU forward against host storage). Those code
//! paths are exercised in the in-crate `ops::repeat::tests` unit
//! tests.

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

fn cpu_reference_f32(data: &[f32], shape: Vec<usize>, n_rep: usize) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape).unwrap();
    let y = ops::repeat(&x, 0, n_rep).unwrap();
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

fn run_repeat_axis0_parity(shape: Vec<usize>, n_rep: usize, dtype: CandleDType) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 31);

    let x_cd = match shape.as_slice() {
        [a] => CandleTensor::from_vec(data.clone(), (*a,), &dev).unwrap(),
        [a, b] => CandleTensor::from_vec(data.clone(), (*a, *b), &dev).unwrap(),
        [a, b, c] => CandleTensor::from_vec(data.clone(), (*a, *b, *c), &dev).unwrap(),
        other => panic!("unsupported test shape {other:?}"),
    };
    let x_cd = x_cd.to_dtype(dtype).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::repeat(&x_kt, 0, n_rep).expect("repeat axis=0 dispatch");
    let mut expected_shape = shape.clone();
    expected_shape[0] = shape[0] * n_rep;
    assert_eq!(out_kt.shape(), expected_shape.as_slice());

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n_out: usize = expected_shape.iter().product();
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
    let ref_v = cpu_reference_f32(&ref_data, shape.clone(), n_rep);
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
fn cuda_repeat_axis0_rank1_f32() {
    run_repeat_axis0_parity(vec![5], 3, CandleDType::F32);
}

#[test]
fn cuda_repeat_axis0_rank2_f32_addmm_pattern() {
    // Mirrors the addmm bias-broadcast pattern: repeat([1, N], 0, M).
    run_repeat_axis0_parity(vec![1, 32], 8, CandleDType::F32);
}

#[test]
fn cuda_repeat_axis0_rank2_bf16() {
    run_repeat_axis0_parity(vec![4, 7], 3, CandleDType::BF16);
}

#[test]
fn cuda_repeat_axis0_rank3_f16() {
    run_repeat_axis0_parity(vec![3, 5, 4], 2, CandleDType::F16);
}

#[test]
fn cuda_repeat_axis0_n_one_preserves_shape_and_dtype() {
    // `n=1` is identity — `cuda_fwd` reshape-clones the storage,
    // so the device stays the same and no data movement happens.
    // We don't round-trip through `kt_tensor_to_candle_cuda_copy`
    // because that path doesn't support Borrowed CudaStorage today;
    // byte-identity for the real (n > 1) repeats is covered by the
    // `cuda_repeat_axis0_*` parity tests above.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data, (2, 3), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::repeat(&x_kt, 0, 1).expect("repeat n=1 dispatch");
    assert_eq!(out_kt.shape(), &[2, 3]);
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::F32);
    assert!(matches!(out_kt.device(), kiln_tensor::Device::Cuda(_)));
}

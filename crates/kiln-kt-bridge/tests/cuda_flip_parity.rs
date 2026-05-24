//! Parity test: kt CUDA `flip(axes=[0])` vs kt CPU reference.
//!
//! Phase 4 substrate validation. `FlipOp::cuda_fwd` routes through
//! `cuda_index_select_dim0` with a reversed `[n-1, n-2, ..., 0]`
//! index buffer; this confirms the gather produces byte-identical
//! output to the canonical CPU byte-copy loop in
//! `crates/kiln-tensor/src/ops/flip.rs`.
//!
//! Multi-axis flips and non-zero-axis flips fall through to CPU on
//! the CUDA path (the standalone DeviceOp returns `Ok(None)` for
//! those; `dispatch1` then re-runs the CPU forward against host
//! storage). Those code paths are exercised in the in-crate
//! `ops::flip::tests` unit tests.

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

fn cpu_reference_f32(data: &[f32], shape: Vec<usize>) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape).unwrap();
    let y = ops::flip(&x, &[0]).unwrap();
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

fn run_flip_axis0_parity(shape: Vec<usize>, dtype: CandleDType) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 13);

    // Build the candle tensor and convert to kt borrowed-CUDA storage.
    let x_cd = match shape.as_slice() {
        [a] => CandleTensor::from_vec(data.clone(), (*a,), &dev).unwrap(),
        [a, b] => CandleTensor::from_vec(data.clone(), (*a, *b), &dev).unwrap(),
        [a, b, c] => CandleTensor::from_vec(data.clone(), (*a, *b, *c), &dev).unwrap(),
        other => panic!("unsupported test shape {other:?}"),
    };
    let x_cd = x_cd.to_dtype(dtype).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::flip(&x_kt, &[0]).expect("flip axis=0 dispatch");
    assert_eq!(out_kt.shape(), shape.as_slice());

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt).unwrap();
    let got_vec: Vec<f32> = got_cd
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
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
    let ref_v = cpu_reference_f32(&ref_data, shape.clone());
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
fn cuda_flip_axis0_rank1_f32() {
    run_flip_axis0_parity(vec![16], CandleDType::F32);
}

#[test]
fn cuda_flip_axis0_rank2_f32() {
    run_flip_axis0_parity(vec![8, 5], CandleDType::F32);
}

#[test]
fn cuda_flip_axis0_rank3_bf16() {
    run_flip_axis0_parity(vec![6, 4, 3], CandleDType::BF16);
}

#[test]
fn cuda_flip_axis0_rank2_f16() {
    run_flip_axis0_parity(vec![5, 7], CandleDType::F16);
}

#[test]
fn cuda_flip_empty_axes_preserves_shape_and_dtype() {
    // No-op on `axes=[]` — `cuda_fwd` reshape-clones the storage,
    // so the device stays the same and no data movement happens.
    // We don't round-trip through `kt_tensor_to_candle_cuda_copy`
    // because that path doesn't support Borrowed CudaStorage today;
    // a full byte-identity check for the non-empty axes case is
    // covered by the `cuda_flip_axis0_*` parity tests above.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data, (2, 3), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::flip(&x_kt, &[]).expect("flip [] dispatch");
    assert_eq!(out_kt.shape(), &[2, 3]);
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::F32);
    // Device should remain Cuda(0).
    assert!(matches!(out_kt.device(), kiln_tensor::Device::Cuda(_)));
}

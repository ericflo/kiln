//! Parity test: kt CUDA element-wise comparison ops (eq / ne / lt /
//! le / gt / ge) vs kt CPU reference.
//!
//! The kernel `csrc/compare.cu` does per-element F32-promoted
//! comparison and writes a U8 mask. Parity is bit-exact for F32 and
//! exact-up-to-rounding for BF16/F16 (since both CPU and CUDA promote
//! the same way and the comparison ops are monotonic). (#1082)

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
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 128.0;
        out.push(f);
    }
    out
}

fn cpu_ref(kind: &str, a: &[f32], b: &[f32], shape: &[usize]) -> Vec<u8> {
    let at = kiln_tensor::Tensor::from_slice(a, shape.to_vec()).unwrap();
    let bt = kiln_tensor::Tensor::from_slice(b, shape.to_vec()).unwrap();
    let out = match kind {
        "eq" => ops::eq(&at, &bt).unwrap(),
        "ne" => ops::ne(&at, &bt).unwrap(),
        "lt" => ops::lt(&at, &bt).unwrap(),
        "le" => ops::le(&at, &bt).unwrap(),
        "gt" => ops::gt(&at, &bt).unwrap(),
        "ge" => ops::ge(&at, &bt).unwrap(),
        other => panic!("unknown kind {other}"),
    };
    let cpu = out
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    cpu.as_bytes().to_vec()
}

fn run_kind(kind: &str, dev: &CandleDevice, a: &[f32], b: &[f32], shape: (usize, usize)) {
    let a_cd = CandleTensor::from_vec(a.to_vec(), shape, dev).unwrap();
    let b_cd = CandleTensor::from_vec(b.to_vec(), shape, dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = match kind {
        "eq" => ops::eq(&a_kt, &b_kt).unwrap(),
        "ne" => ops::ne(&a_kt, &b_kt).unwrap(),
        "lt" => ops::lt(&a_kt, &b_kt).unwrap(),
        "le" => ops::le(&a_kt, &b_kt).unwrap(),
        "gt" => ops::gt(&a_kt, &b_kt).unwrap(),
        "ge" => ops::ge(&a_kt, &b_kt).unwrap(),
        other => panic!("unknown kind {other}"),
    };
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::U8);
    assert_eq!(out_kt.shape(), &[shape.0, shape.1]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n = shape.0 * shape.1;
    let got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    let shape_vec = vec![shape.0, shape.1];
    let want = cpu_ref(kind, a, b, &shape_vec);
    assert_eq!(want, got, "kind={kind}");
}

#[test]
fn cuda_compare_all_kinds_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = (4, 8);
    let a = pattern_f32(shape.0 * shape.1, 11);
    let b = pattern_f32(shape.0 * shape.1, 13);
    for kind in ["eq", "ne", "lt", "le", "gt", "ge"] {
        run_kind(kind, &dev, &a, &b, shape);
    }
}

#[test]
fn cuda_compare_eq_self_all_ones() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_cd = CandleTensor::from_vec(a.clone(), (4,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();

    let out_kt = ops::eq(&a_kt, &a_kt).unwrap();
    let cuda_dev = match dev {
        CandleDevice::Cuda(c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(got, vec![1, 1, 1, 1]);
}

#[test]
fn cuda_compare_lt_simple() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![2.0, 2.0, 2.0];
    let a_cd = CandleTensor::from_vec(a, (3,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b, (3,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let cuda_dev = match dev {
        CandleDevice::Cuda(c) => c,
        _ => unreachable!(),
    };

    let lt_out = ops::lt(&a_kt, &b_kt).unwrap();
    cuda_dev.synchronize().unwrap();
    let lt_got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&lt_out)
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(lt_got, vec![1, 0, 0]);

    let le_out = ops::le(&a_kt, &b_kt).unwrap();
    cuda_dev.synchronize().unwrap();
    let le_got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&le_out)
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(le_got, vec![1, 1, 0]);

    let gt_out = ops::gt(&a_kt, &b_kt).unwrap();
    cuda_dev.synchronize().unwrap();
    let gt_got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&gt_out)
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(gt_got, vec![0, 0, 1]);
}

#[test]
fn cuda_compare_bf16_matches_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = (4, 8);
    let a = pattern_f32(shape.0 * shape.1, 41);
    let b = pattern_f32(shape.0 * shape.1, 43);
    let a_cd = CandleTensor::from_vec(a.clone(), shape, &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b.clone(), shape, &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::lt(&a_kt, &b_kt).unwrap();
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::U8);

    let cuda_dev = match dev {
        CandleDevice::Cuda(c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n = shape.0 * shape.1;
    let got: Vec<u8> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();

    // CPU reference: quantize through BF16 then compare.
    let a_bf16: Vec<half::bf16> = a.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let b_bf16: Vec<half::bf16> = b.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let a_kt_cpu = kiln_tensor::Tensor::from_slice(&a_bf16, vec![shape.0, shape.1]).unwrap();
    let b_kt_cpu = kiln_tensor::Tensor::from_slice(&b_bf16, vec![shape.0, shape.1]).unwrap();
    let ref_kt = ops::lt(&a_kt_cpu, &b_kt_cpu).unwrap();
    let ref_cpu = ref_kt
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let want: Vec<u8> = ref_cpu.as_bytes().to_vec();
    assert_eq!(want, got);
}

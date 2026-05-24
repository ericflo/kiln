//! Parity tests for the CUDA `where_select` kernel vs CPU reference.
//!
//! The CUDA kernel (`csrc/where_select.cu`) does per-element select
//! over F32 / BF16 / F16 (mask is U8). The CPU reference is exact
//! byte-copy, so the values must match bit-for-bit. (#1082)

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::ops;

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn read_f32(t: &kiln_tensor::Tensor) -> Vec<f32> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
fn cuda_where_select_f32_picks_per_position() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let mask_data = vec![1u8, 0, 1, 0];
    let t_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let f_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let m_cd = CandleTensor::from_vec(mask_data.clone(), (4,), &dev).unwrap();
    let t_cd = CandleTensor::from_vec(t_data.clone(), (4,), &dev).unwrap();
    let f_cd = CandleTensor::from_vec(f_data.clone(), (4,), &dev).unwrap();

    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();
    let f_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&f_cd).unwrap();

    let out_kt = ops::where_select(&m_kt, &t_kt, &f_kt).expect("where_select");
    assert_eq!(out_kt.shape(), &[4]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, vec![1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn cuda_where_select_f32_all_mask_set_returns_t() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 16;
    let mask_data = vec![1u8; n];
    let t_data = pattern_f32(n, 1);
    let f_data = pattern_f32(n, 2);

    let m_cd = CandleTensor::from_vec(mask_data, (n,), &dev).unwrap();
    let t_cd = CandleTensor::from_vec(t_data.clone(), (n,), &dev).unwrap();
    let f_cd = CandleTensor::from_vec(f_data.clone(), (n,), &dev).unwrap();

    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();
    let f_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&f_cd).unwrap();

    let out_kt = ops::where_select(&m_kt, &t_kt, &f_kt).unwrap();

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, t_data);
}

#[test]
fn cuda_where_select_f32_parity_vs_cpu_2d() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let rows = 4;
    let cols = 17;
    let n = rows * cols;
    // alternating mask
    let mask_data: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
    let t_data = pattern_f32(n, 7);
    let f_data = pattern_f32(n, 11);

    // CPU reference
    let m_cpu = kiln_tensor::Tensor::from_slice(&mask_data, vec![rows, cols]).unwrap();
    let t_cpu = kiln_tensor::Tensor::from_slice(&t_data, vec![rows, cols]).unwrap();
    let f_cpu = kiln_tensor::Tensor::from_slice(&f_data, vec![rows, cols]).unwrap();
    let out_cpu = ops::where_select(&m_cpu, &t_cpu, &f_cpu).unwrap();
    let want = read_f32(&out_cpu);

    // CUDA path
    let m_cd = CandleTensor::from_vec(mask_data, (rows, cols), &dev).unwrap();
    let t_cd = CandleTensor::from_vec(t_data, (rows, cols), &dev).unwrap();
    let f_cd = CandleTensor::from_vec(f_data, (rows, cols), &dev).unwrap();

    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();
    let f_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&f_cd).unwrap();

    let out_kt = ops::where_select(&m_kt, &t_kt, &f_kt).unwrap();
    assert_eq!(out_kt.shape(), &[rows, cols]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt).unwrap();
    let got: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "mismatch at index {i}");
    }
}

#[test]
fn cuda_where_select_bf16_parity_vs_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 32;
    let mask_data: Vec<u8> = (0..n).map(|i| (i % 2 == 0) as u8).collect();
    let t_data = pattern_f32(n, 13);
    let f_data = pattern_f32(n, 17);

    // CPU reference (in bf16).
    let tv_bf16: Vec<half::bf16> = t_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let fv_bf16: Vec<half::bf16> = f_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let m_cpu = kiln_tensor::Tensor::from_slice(&mask_data, vec![n]).unwrap();
    let t_cpu = kiln_tensor::Tensor::from_slice(&tv_bf16, vec![n]).unwrap();
    let f_cpu = kiln_tensor::Tensor::from_slice(&fv_bf16, vec![n]).unwrap();
    let out_cpu = ops::where_select(&m_cpu, &t_cpu, &f_cpu).unwrap();
    let cpu_bytes = out_cpu
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap()
        .as_bytes()
        .to_vec();

    // CUDA path.
    let m_cd = CandleTensor::from_vec(mask_data, (n,), &dev).unwrap();
    let t_cd = CandleTensor::from_vec(tv_bf16, (n,), &dev).unwrap();
    let f_cd = CandleTensor::from_vec(fv_bf16, (n,), &dev).unwrap();
    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();
    let f_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&f_cd).unwrap();

    let out_kt = ops::where_select(&m_kt, &t_kt, &f_kt).unwrap();
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::BF16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Pull back to CPU via candle to_dtype(F32) then compare to bf16-quantized CPU ref.
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want: Vec<f32> = cpu_bytes
        .chunks(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "bf16 mismatch at {i}: got {g}, want {w}");
    }
}

//! Round-trip test: CUDA buffer → host kt-Tensor → byte-identical
//! to a CPU kt-Tensor built from the same source data.

use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_to_host_copy, CpuStorage};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

#[test]
fn cuda_to_host_copy_round_trip_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 1.0).collect();
    let cd = CandleTensor::from_vec(data.clone(), (4, 8), &dev).unwrap();

    // Borrow into kt; copy to host.
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();
    let host = cuda_to_host_copy(&kt).expect("D2H");

    assert_eq!(host.shape(), &[4, 8]);
    assert_eq!(host.dtype(), kiln_tensor::DType::F32);
    assert!(matches!(host.device(), kiln_tensor::Device::Cpu));

    let cpu_storage = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("CpuStorage");
    let bytes = cpu_storage.as_bytes();
    assert_eq!(bytes.len(), 4 * 8 * 4);
    for i in 0..32 {
        let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(v, data[i], "byte {i} mismatch");
    }
}

#[test]
fn cuda_to_host_copy_round_trip_bf16() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let cd = CandleTensor::from_vec(data.clone(), (4, 4), &dev)
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();

    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();
    let host = cuda_to_host_copy(&kt).expect("D2H bf16");

    assert_eq!(host.shape(), &[4, 4]);
    assert_eq!(host.dtype(), kiln_tensor::DType::BF16);

    let cpu_storage = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("CpuStorage");
    let bytes = cpu_storage.as_bytes();
    assert_eq!(bytes.len(), 4 * 4 * 2);
    for i in 0..16 {
        let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        // BF16 quantization tolerance ~1/128 for these small values.
        let expected = half::bf16::from_f32(data[i]).to_f32();
        assert!((v - expected).abs() < 1e-6, "byte {i}: got {v} want {expected}");
    }
}

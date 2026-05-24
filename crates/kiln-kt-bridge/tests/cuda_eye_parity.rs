//! Parity tests for `eye_on_device` against the CPU reference.
//!
//! `eye_on_device(n, dtype, Cuda(_))` builds an identity matrix on
//! the host then copies to CUDA via `host_to_cuda_copy`. The bytes
//! must match the host `eye(n, dtype)` exactly — no rounding paths
//! involved. (#1082)

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice};

use kiln_tensor::ops;

fn try_cuda() -> Option<(CandleDevice, std::sync::Arc<candle_core::cuda_backend::CudaDevice>)> {
    let dev = CandleDevice::new_cuda(0).ok()?;
    let cuda = match &dev {
        CandleDevice::Cuda(c) => std::sync::Arc::new(c.clone()),
        _ => return None,
    };
    Some((dev, cuda))
}

#[test]
fn cuda_eye_f32_matches_cpu() {
    let Some((dev, cuda)) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let n = 8;
    let eye_cuda =
        ops::eye_on_device(n, kiln_tensor::DType::F32, kiln_tensor::Device::Cuda(0), cuda.clone())
            .expect("eye_on_device f32");
    assert_eq!(eye_cuda.shape(), &[n, n]);
    assert_eq!(eye_cuda.dtype(), kiln_tensor::DType::F32);
    assert_eq!(eye_cuda.device(), kiln_tensor::Device::Cuda(0));

    cuda.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&eye_cuda)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Build expected: identity
    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        want[i * n + i] = 1.0;
    }
    assert_eq!(got, want);
    let _ = dev;
}

#[test]
fn cuda_eye_bf16_matches_cpu() {
    let Some((dev, cuda)) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let n = 4;
    let eye_cuda =
        ops::eye_on_device(n, kiln_tensor::DType::BF16, kiln_tensor::Device::Cuda(0), cuda.clone())
            .expect("eye_on_device bf16");
    assert_eq!(eye_cuda.dtype(), kiln_tensor::DType::BF16);

    cuda.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&eye_cuda)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        want[i * n + i] = 1.0;
    }
    assert_eq!(got, want);
    let _ = dev;
}

#[test]
fn cuda_eye_dispatches_to_cpu_when_requested() {
    // CPU path of eye_on_device returns the same as the host-only `eye`.
    let host = ops::eye(3, kiln_tensor::DType::F32).unwrap();
    // Even without CUDA available, the CPU branch should work.
    let Some((dev, cuda)) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let on_cpu =
        ops::eye_on_device(3, kiln_tensor::DType::F32, kiln_tensor::Device::Cpu, cuda.clone())
            .expect("eye_on_device cpu");
    assert_eq!(on_cpu.shape(), host.shape());
    assert_eq!(on_cpu.device(), kiln_tensor::Device::Cpu);
    let _ = dev;
}

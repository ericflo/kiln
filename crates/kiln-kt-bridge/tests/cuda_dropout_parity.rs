//! Parity test: kt CUDA `dropout` vs contract (distribution + scale).
//!
//! Issue #1082 — sub-phase: dropout cuda_fwd wiring.
//!
//! Note: the CUDA path uses a per-element splitmix64 hash of `(seed,
//! i)` (parallel-friendly), while the CPU path uses a sequential
//! splitmix64 chain. The masks are NOT bit-identical across devices,
//! so this test verifies the dropout *contract* on the GPU:
//!
//! - p == 0 → identity output, all-ones mask
//! - p > 0  → drop rate within sampling tolerance of p
//! - Surviving elements scaled by 1/(1-p)
//! - Dropped elements are exactly 0
//! - Mask dtype = U8; output shape preserved
//! - Determinism: same (seed, input) → byte-identical (y, mask)
//! - Different seeds produce different masks

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, DType, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn make_cuda_input(n: usize, value: f32, dev: &CandleDevice) -> Tensor {
    let data = vec![value; n];
    let x_cd = CandleTensor::from_vec(data, (n,), dev).unwrap();
    kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap()
}

fn read_f32_from_cuda(t: &Tensor) -> Vec<f32> {
    let n: usize = t.shape().iter().product();
    kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(t)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn read_u8_from_cuda(t: &Tensor) -> Vec<u8> {
    let n: usize = t.shape().iter().product();
    kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(t)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<u8>()
        .unwrap()
}

#[test]
fn cuda_dropout_p_zero_is_identity() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let x = make_cuda_input(16, 2.5, &dev);
    let (y, mask) = ops::dropout(&x, 0.0, 42).expect("dropout p=0");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(y.shape(), x.shape());
    assert_eq!(mask.shape(), x.shape());
    assert_eq!(mask.dtype(), DType::U8);

    let y_vals = read_f32_from_cuda(&y);
    let m_vals = read_u8_from_cuda(&mask);
    for v in &y_vals {
        assert!((*v - 2.5).abs() < 1e-6, "expected identity, got {v}");
    }
    for m in &m_vals {
        assert_eq!(*m, 1u8);
    }
}

#[test]
fn cuda_dropout_drop_rate_within_tolerance() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 4096;
    let p = 0.25f32;
    let x = make_cuda_input(n, 1.0, &dev);
    let (_y, mask) = ops::dropout(&x, p, 7).expect("dropout");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let m = read_u8_from_cuda(&mask);
    let dropped: usize = m.iter().filter(|&&v| v == 0).count();
    let drop_rate = dropped as f32 / n as f32;
    // 3-sigma band for Bernoulli(p, n): sigma = sqrt(p*(1-p)/n).
    let sigma = (p * (1.0 - p) / (n as f32)).sqrt();
    let band = 4.0 * sigma; // generous
    assert!(
        (drop_rate - p).abs() < band,
        "drop_rate {drop_rate} outside expected band p±{band:.4} (p={p})"
    );
}

#[test]
fn cuda_dropout_surviving_elements_scaled() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 1024;
    let p = 0.5f32;
    let inv_keep = 1.0 / (1.0 - p);
    let x = make_cuda_input(n, 3.0, &dev);
    let (y, mask) = ops::dropout(&x, p, 13).expect("dropout");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let y_vals = read_f32_from_cuda(&y);
    let m_vals = read_u8_from_cuda(&mask);
    for i in 0..n {
        if m_vals[i] == 1 {
            let expected = 3.0 * inv_keep;
            assert!(
                (y_vals[i] - expected).abs() < 1e-4,
                "i={i}: kept, expected {expected}, got {}",
                y_vals[i]
            );
        } else {
            assert_eq!(y_vals[i], 0.0, "i={i}: dropped, expected 0, got {}", y_vals[i]);
        }
    }
}

#[test]
fn cuda_dropout_deterministic_same_seed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 256;
    let x = make_cuda_input(n, 1.0, &dev);
    let (y1, m1) = ops::dropout(&x, 0.3, 99).expect("dropout 1");
    let (y2, m2) = ops::dropout(&x, 0.3, 99).expect("dropout 2");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let y1v = read_f32_from_cuda(&y1);
    let y2v = read_f32_from_cuda(&y2);
    let m1v = read_u8_from_cuda(&m1);
    let m2v = read_u8_from_cuda(&m2);
    assert_eq!(y1v, y2v, "outputs not deterministic for same seed");
    assert_eq!(m1v, m2v, "masks not deterministic for same seed");
}

#[test]
fn cuda_dropout_different_seeds_diverge() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = 256;
    let x = make_cuda_input(n, 1.0, &dev);
    let (_, m1) = ops::dropout(&x, 0.5, 1).expect("dropout 1");
    let (_, m2) = ops::dropout(&x, 0.5, 100).expect("dropout 2");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();
    let m1v = read_u8_from_cuda(&m1);
    let m2v = read_u8_from_cuda(&m2);
    assert!(m1v != m2v, "two different seeds produced identical masks");
}

#[test]
fn cuda_dropout_2d_shape_preserved() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x_cd = CandleTensor::from_vec(data, (3, 4), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let (y, m) = ops::dropout(&x_kt, 0.5, 5).expect("dropout 2d");
    assert_eq!(y.shape(), &[3, 4]);
    assert_eq!(m.shape(), &[3, 4]);
    assert_eq!(m.dtype(), DType::U8);
}

#[test]
fn cuda_dropout_bf16_path() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = vec![2.0; 64];
    let x_cd = CandleTensor::from_vec(data, (64,), &dev).unwrap()
        .to_dtype(CandleDType::BF16).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let (y, m) = ops::dropout(&x_kt, 0.0, 42).expect("dropout bf16");
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(m.dtype(), DType::U8);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let y_vals = read_f32_from_cuda(&y);
    for v in &y_vals {
        assert!((v - 2.0).abs() < 1e-2, "p=0 should be identity, got {v}");
    }
}

#[test]
fn cuda_dropout_f16_path() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = vec![4.0; 32];
    let x_cd = CandleTensor::from_vec(data, (32,), &dev).unwrap()
        .to_dtype(CandleDType::F16).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let (y, _) = ops::dropout(&x_kt, 0.0, 13).expect("dropout f16");
    assert_eq!(y.dtype(), DType::F16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();
    let y_vals = read_f32_from_cuda(&y);
    for v in &y_vals {
        assert!((v - 4.0).abs() < 1e-2, "p=0 should be identity, got {v}");
    }
}

#[test]
fn cuda_dropout_rejects_bad_p() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let x = make_cuda_input(8, 1.0, &dev);
    let e = ops::dropout(&x, 1.0, 1).unwrap_err();
    assert!(e.to_string().contains("p must be"));
    let e = ops::dropout(&x, -0.1, 1).unwrap_err();
    assert!(e.to_string().contains("p must be"));
}

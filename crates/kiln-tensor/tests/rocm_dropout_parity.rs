//! Phase R.5b — dropout CPU-vs-ROCm parity (distribution / scale / mask).
//!
//! `dropout.cu` is a pure elementwise map with a counter-based splitmix64 RNG
//! (per-element hash of `(seed, idx)`) — NO curand/hiprand dependency and no
//! cross-lane reduction, so it is wave32/wave64-correct as-is. The ROCm RNG
//! stream differs from the CPU op's sequential chain, so this is NOT a
//! bit-identical test; it checks the documented contract instead:
//!   1. inverted-dropout scaling: surviving elements are `x * 1/(1-p)`,
//!   2. dropped elements are exactly 0 with a `0` mask byte,
//!   3. determinism: same seed → same `(y, mask)`,
//!   4. distribution: empirical drop rate ≈ p,
//!   5. p = 0 is a pass-through (every element survives).
//!
//! Run: cargo test -p kiln-tensor --features rocm --test rocm_dropout_parity
#![cfg(feature = "rocm")]

use kiln_tensor::ops::dropout;
use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5b dropout parity test");
        true
    } else {
        false
    }
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    let h = kiln_tensor::rocm_to_host_copy(t).expect("to_host");
    h.to_vec::<f32>().expect("to_vec f32")
}

fn host_u8(t: &Tensor) -> Vec<u8> {
    let h = kiln_tensor::rocm_to_host_copy(t).expect("to_host");
    h.to_vec::<u8>().expect("to_vec u8")
}

#[test]
fn dropout_scale_and_mask_consistency() {
    if no_rocm() {
        return;
    }
    let n = 4096usize;
    let x_data = vec![1.0f32; n];
    let p = 0.3f32;
    let inv_keep = 1.0 / (1.0 - p);

    let x = Tensor::from_vec_on(Device::Rocm(0), x_data, vec![n]).unwrap();
    let (y, mask) = dropout(&x, p, 1234).unwrap();
    assert_eq!(mask.dtype(), DType::U8);

    let yv = host_f32(&y);
    let mv = host_u8(&mask);
    assert_eq!(yv.len(), n);
    assert_eq!(mv.len(), n);

    let mut dropped = 0usize;
    for i in 0..n {
        match mv[i] {
            0 => {
                assert_eq!(yv[i], 0.0, "dropped element {i} not zeroed: {}", yv[i]);
                dropped += 1;
            }
            1 => {
                let diff = (yv[i] - inv_keep).abs();
                assert!(
                    diff < 1e-4,
                    "survivor {i} scale wrong: got {} want {inv_keep}",
                    yv[i]
                );
            }
            other => panic!("mask byte {other} not in {{0,1}} at {i}"),
        }
    }
    // Empirical drop rate within a generous band of p.
    let rate = dropped as f32 / n as f32;
    assert!(
        (rate - p).abs() < 0.05,
        "drop rate {rate} too far from p={p}"
    );
    eprintln!("dropout scale/mask consistency ok (drop rate {rate:.3} vs p {p})");
}

#[test]
fn dropout_deterministic_same_seed() {
    if no_rocm() {
        return;
    }
    let n = 1000usize;
    let x = Tensor::from_vec_on(Device::Rocm(0), vec![2.0f32; n], vec![n]).unwrap();
    let (y1, m1) = dropout(&x, 0.5, 42).unwrap();
    let (y2, m2) = dropout(&x, 0.5, 42).unwrap();
    assert_eq!(host_f32(&y1), host_f32(&y2), "same seed → same y");
    assert_eq!(host_u8(&m1), host_u8(&m2), "same seed → same mask");
}

#[test]
fn dropout_different_seeds_diverge() {
    if no_rocm() {
        return;
    }
    let n = 1000usize;
    let x = Tensor::from_vec_on(Device::Rocm(0), vec![1.0f32; n], vec![n]).unwrap();
    let (_, m1) = dropout(&x, 0.5, 1).unwrap();
    let (_, m2) = dropout(&x, 0.5, 99).unwrap();
    assert_ne!(host_u8(&m1), host_u8(&m2), "different seeds → different masks");
}

#[test]
fn dropout_p_zero_is_passthrough() {
    if no_rocm() {
        return;
    }
    let x = Tensor::from_vec_on(Device::Rocm(0), vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let (y, mask) = dropout(&x, 0.0, 7).unwrap();
    assert_eq!(host_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(host_u8(&mask), vec![1, 1, 1, 1]);
}

#[test]
fn dropout_bf16_scale_and_mask() {
    if no_rocm() {
        return;
    }
    let n = 2048usize;
    let bf: Vec<half::bf16> = (0..n).map(|_| half::bf16::from_f32(1.0)).collect();
    let x = Tensor::from_vec_on(Device::Rocm(0), bf, vec![n]).unwrap();
    let p = 0.25f32;
    let (y, mask) = dropout(&x, p, 555).unwrap();
    assert_eq!(y.dtype(), DType::BF16);

    let yh = kiln_tensor::rocm_to_host_copy(&y).unwrap();
    let yv = yh.to_vec::<half::bf16>().unwrap();
    let mv = host_u8(&mask);
    let inv_keep = 1.0 / (1.0 - p);
    for i in 0..n {
        let yf = yv[i].to_f32();
        if mv[i] == 0 {
            assert_eq!(yf, 0.0, "bf16 dropped {i} not zeroed");
        } else {
            // bf16 has ~3 significant digits; loose tolerance.
            assert!((yf - inv_keep).abs() < 0.05, "bf16 survivor {i}: {yf}");
        }
    }
}

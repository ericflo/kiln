//! Deterministic CUDA dense-matmul parity against a scalar CPU oracle.
//!
//! Normal developer runs skip without CUDA. Qualification mode makes missing
//! hardware a failure so a passing receipt always represents real execution.
#![cfg(feature = "cuda")]

use half::bf16;
use kiln_tensor::{DType, Device, Tensor, ops};

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn cuda_available_or_skip() -> bool {
    if kiln_tensor::cuda_is_available() {
        return true;
    }
    if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
        panic!("CUDA device unavailable while KILN_QUALIFICATION=1");
    }
    eprintln!("no CUDA device available; skipping CUDA matmul parity test");
    false
}

fn value(index: usize, offset: usize) -> f32 {
    (((index.wrapping_mul(37).wrapping_add(offset)) % 97) as f32 / 97.0 - 0.5) * 0.5
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + column];
            }
            output[row * n + column] = sum;
        }
    }
    output
}

fn assert_close(got: &[f32], expected: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: output length");
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let error = (got - expected).abs();
        let tolerance = atol + rtol * expected.abs();
        assert!(
            error <= tolerance,
            "{label}: index={index} got={got} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn host_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .expect("CUDA output to CPU")
        .to_dtype(DType::F32)
        .expect("CUDA output to F32")
        .flatten_all()
        .expect("flatten CUDA output")
        .to_vec1::<f32>()
        .expect("read CUDA output")
}

#[test]
fn qualification_mode_is_exact_opt_in() {
    assert!(qualification_required(Some("1")));
    assert!(!qualification_required(None));
    assert!(!qualification_required(Some("")));
    assert!(!qualification_required(Some("0")));
    assert!(!qualification_required(Some("true")));
}

#[test]
fn cuda_matmul_f32_matches_cpu() {
    if !cuda_available_or_skip() {
        return;
    }
    for &(m, k, n) in &[(1, 64, 64), (16, 16, 16), (7, 65, 33), (32, 128, 64)] {
        let a: Vec<f32> = (0..m * k).map(|index| value(index, 11)).collect();
        let b: Vec<f32> = (0..k * n).map(|index| value(index, 29)).collect();
        let expected = cpu_matmul(&a, &b, m, k, n);
        let a = Tensor::from_vec_on(Device::Cuda(0), a, vec![m, k]).expect("CUDA lhs");
        let b = Tensor::from_vec_on(Device::Cuda(0), b, vec![k, n]).expect("CUDA rhs");
        let output = ops::matmul(&a, &b).expect("CUDA F32 matmul");
        assert_eq!(output.device(), Device::Cuda(0));
        assert_eq!(output.shape(), &[m, n]);
        assert_close(
            &host_f32(&output),
            &expected,
            2e-4,
            2e-4,
            &format!("F32 {m}x{k}x{n}"),
        );
    }
    eprintln!("[CUDA MATMUL PARITY PASS] dtype=f32 shapes=4");
}

#[test]
fn cuda_matmul_bf16_matches_rounded_cpu_inputs() {
    if !cuda_available_or_skip() {
        return;
    }
    for &(m, k, n) in &[(1, 256, 512), (16, 64, 96), (32, 128, 64)] {
        let a: Vec<bf16> = (0..m * k)
            .map(|index| bf16::from_f32(value(index, 7)))
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|index| bf16::from_f32(value(index, 43)))
            .collect();
        let a_f32: Vec<f32> = a.iter().map(|item| item.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|item| item.to_f32()).collect();
        let expected = cpu_matmul(&a_f32, &b_f32, m, k, n);
        let a = Tensor::from_vec_on(Device::Cuda(0), a, vec![m, k]).expect("CUDA BF16 lhs");
        let b = Tensor::from_vec_on(Device::Cuda(0), b, vec![k, n]).expect("CUDA BF16 rhs");
        let output = ops::matmul(&a, &b).expect("CUDA BF16 matmul");
        assert_eq!(output.device(), Device::Cuda(0));
        assert_eq!(output.dtype(), DType::BF16);
        assert_close(
            &host_f32(&output),
            &expected,
            3e-2,
            k as f32 * 1e-3,
            &format!("BF16 {m}x{k}x{n}"),
        );
    }
    eprintln!("[CUDA MATMUL PARITY PASS] dtype=bfloat16 shapes=3");
}

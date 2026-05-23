//! Parameterized activation variants: `leaky_relu`, `elu`, `softplus`,
//! `mish`.
//!
//! Compose with the unparameterized `silu/sigmoid/gelu/tanh/relu`
//! from `activation.rs`.
//!
//! - **leaky_relu**(x, α) = `x if x >= 0 else α*x` — avoids "dying
//!   ReLU" by passing a small fraction of negative gradient.
//! - **elu**(x, α) = `x if x >= 0 else α*(exp(x) - 1)` — smooth
//!   negative tail.
//! - **softplus**(x) = `log(1 + exp(x))` — smooth ReLU. Numerically
//!   stable form for large x.
//! - **mish**(x) = `x * tanh(softplus(x))` — self-regularized
//!   non-monotonic activation (Misra 2019).

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn apply(f: impl Fn(f32) -> f32, x: &Tensor, name: &str) -> Result<Tensor> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("leaky_activations: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = x.element_count();
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; n * per];
    for i in 0..n {
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        let y = f(v);
        match dtype {
            DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())
}

pub fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    apply(
        move |v| if v >= 0.0 { v } else { alpha * v },
        x,
        "leaky_relu",
    )
}

pub fn elu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    apply(
        move |v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) },
        x,
        "elu",
    )
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    // Stable form: max(x, 0) + log(1 + exp(-|x|))
    apply(
        |v| v.max(0.0) + (-v.abs()).exp().ln_1p(),
        x,
        "softplus",
    )
}

pub fn mish(x: &Tensor) -> Result<Tensor> {
    apply(
        |v| {
            let sp = v.max(0.0) + (-v.abs()).exp().ln_1p();
            v * sp.tanh()
        },
        x,
        "mish",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn leaky_relu_passes_positives_attenuates_negatives() {
        let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
        let y = read_f32(&leaky_relu(&x, 0.1).unwrap());
        assert_eq!(y, vec![-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn elu_known_values() {
        // elu(0) = 0; elu(1) = 1; elu(-1) = α*(1/e - 1) ≈ -0.632 with α=1.
        let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0], vec![3]).unwrap();
        let y = read_f32(&elu(&x, 1.0).unwrap());
        assert!((y[0] - (1.0f32.exp().recip() - 1.0)).abs() < 1e-5);
        assert!(y[1].abs() < 1e-6);
        assert!((y[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softplus_at_zero_is_log2() {
        let x = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let y = read_f32(&softplus(&x).unwrap());
        assert!((y[0] - 2.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn softplus_stable_at_large_x() {
        // softplus(100) ≈ 100 (not Inf).
        let x = Tensor::from_slice(&[100.0f32], vec![1]).unwrap();
        let y = read_f32(&softplus(&x).unwrap());
        assert!(y[0].is_finite());
        assert!((y[0] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn mish_at_zero_is_zero() {
        let x = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let y = read_f32(&mish(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
    }

    #[test]
    fn mish_known_values() {
        // mish(1) ≈ 0.8651. Reference from misra paper.
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let y = read_f32(&mish(&x).unwrap());
        assert!((y[0] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [-1.0f32, 0.0, 1.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![3]).unwrap();
        for op_name in ["leaky_relu", "elu", "softplus", "mish"] {
            let y = match op_name {
                "leaky_relu" => leaky_relu(&x, 0.1).unwrap(),
                "elu" => elu(&x, 1.0).unwrap(),
                "softplus" => softplus(&x).unwrap(),
                "mish" => mish(&x).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(y.dtype(), DType::BF16, "{op_name}");
        }
    }
}

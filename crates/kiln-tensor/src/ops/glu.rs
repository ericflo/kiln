//! GLU family ops: `glu`, `swiglu`, `geglu`, `reglu`.
//!
//! Gated Linear Units — split the trailing axis in half and gate one
//! half with an activation applied to the other:
//!
//! ```text
//! split(x) → (a, b), each of shape [..., D/2]
//! glu(x)     = a * sigmoid(b)
//! swiglu(x)  = a * silu(b)
//! geglu(x)   = a * gelu(b)
//! reglu(x)   = a * relu(b)
//! ```
//!
//! Input trailing axis must be even. Used in Phi / PaLM / LLaMA-style
//! MLP blocks instead of bare ReLU MLP.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy)]
enum GluKind {
    Glu,
    SwiGLU,
    GeGLU,
    ReGLU,
}

impl GluKind {
    fn name(self) -> &'static str {
        match self {
            GluKind::Glu => "glu",
            GluKind::SwiGLU => "swiglu",
            GluKind::GeGLU => "geglu",
            GluKind::ReGLU => "reglu",
        }
    }
    fn activate(self, x: f32) -> f32 {
        match self {
            GluKind::Glu => 1.0 / (1.0 + (-x).exp()),
            GluKind::SwiGLU => x / (1.0 + (-x).exp()),
            GluKind::GeGLU => {
                const C: f32 = 0.7978845608_f32;
                let inner = C * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            GluKind::ReGLU => x.max(0.0),
        }
    }
}

fn apply(kind: GluKind, x: &Tensor) -> Result<Tensor> {
    if x.rank() == 0 {
        bail!("{}: input must have rank ≥ 1", kind.name());
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{}: dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("{}: input must be contiguous", kind.name());
    }
    let shape = x.shape().to_vec();
    let last = *shape.last().unwrap();
    if !last.is_multiple_of(2) {
        bail!(
            "{}: trailing axis {last} must be even (will split in half)",
            kind.name()
        );
    }
    let half = last / 2;
    let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let dtype = x.dtype();
    let per = dtype.size_in_bytes();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("glu: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; outer * half * per];
    for o in 0..outer {
        for i in 0..half {
            let a_idx = o * last + i;
            let b_idx = o * last + i + half;
            let av = match dtype {
                DType::F32 => f32::from_le_bytes(bytes[a_idx * 4..a_idx * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[a_idx * 2..a_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[a_idx * 2..a_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            let bv = match dtype {
                DType::F32 => f32::from_le_bytes(bytes[b_idx * 4..b_idx * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[b_idx * 2..b_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[b_idx * 2..b_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            let y = av * kind.activate(bv);
            let dst = (o * half + i) * per;
            match dtype {
                DType::F32 => out[dst..dst + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out[dst..dst + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => out[dst..dst + 2]
                    .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let mut out_shape = shape;
    *out_shape.last_mut().unwrap() = half;
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
}

pub fn glu(x: &Tensor) -> Result<Tensor> {
    apply(GluKind::Glu, x)
}
pub fn swiglu(x: &Tensor) -> Result<Tensor> {
    apply(GluKind::SwiGLU, x)
}
pub fn geglu(x: &Tensor) -> Result<Tensor> {
    apply(GluKind::GeGLU, x)
}
pub fn reglu(x: &Tensor) -> Result<Tensor> {
    apply(GluKind::ReGLU, x)
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
    fn glu_basic_shape() {
        // [4] → [2] (halved trailing axis).
        let x = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 100.0], vec![4]).unwrap();
        let y = glu(&x).unwrap();
        assert_eq!(y.shape(), &[2]);
        // a = [1, 2]; b = [0, 100]
        // sigmoid(0) = 0.5; sigmoid(100) ≈ 1
        // → [1 * 0.5, 2 * 1.0] = [0.5, 2.0]
        let v = read_f32(&y);
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn swiglu_at_zero() {
        // a = [1]; b = [0]. silu(0) = 0. → 0.
        let x = Tensor::from_slice(&[1.0f32, 0.0], vec![2]).unwrap();
        let y = read_f32(&swiglu(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
    }

    #[test]
    fn reglu_kills_negatives() {
        // a = [1]; b = [-5]. relu(-5) = 0. → 0.
        let x = Tensor::from_slice(&[1.0f32, -5.0], vec![2]).unwrap();
        let y = read_f32(&reglu(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
    }

    #[test]
    fn geglu_smoke_test() {
        // Just sanity that geglu runs.
        let x = Tensor::from_slice(&[1.0f32, 0.5], vec![2]).unwrap();
        let y = read_f32(&geglu(&x).unwrap());
        // gelu(0.5) ≈ 0.345; product = 1 * 0.345 = 0.345.
        assert!((y[0] - 0.345).abs() < 0.01);
    }

    #[test]
    fn glu_rank2_per_row() {
        // [B=2, D=4] → [B=2, D=2]; per-row halving.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 100.0, 5.0, 5.0, 100.0, 0.0], vec![2, 4]).unwrap();
        let y = glu(&x).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
    }

    #[test]
    fn glu_odd_axis_rejected() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = glu(&x).unwrap_err();
        assert!(e.to_string().contains("must be even"));
    }

    #[test]
    fn glu_rank_0_rejected() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let e = glu(&x).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }
}

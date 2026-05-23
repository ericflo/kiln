//! Backwards for the leaky-activation family: `leaky_relu`, `elu`,
//! `softplus`, `mish`.
//!
//! | Op          | Backward                                          |
//! |-------------|---------------------------------------------------|
//! | leaky_relu  | `dx = dy * 1` if x > 0 else `dy * α`              |
//! | elu(α)      | `dx = dy * 1` if x > 0 else `dy * α * exp(x)`     |
//! | softplus    | `dx = dy * sigmoid(x)`                            |
//! | mish        | see below; saved input.                           |
//!
//! `mish(x) = x * tanh(softplus(x))`. Its derivative:
//! `dmish/dx = exp(x) * (4(x+1) + 4*exp(2x) + exp(3x) + exp(x)*(4x+6)) /
//! (2*exp(x) + exp(2x) + 2)^2`.
//! We compute it via the stable form `tanh_sp + x * sech²(sp) * sigmoid`.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

fn validate_same(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch");
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    Ok(())
}

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("leaky_activations_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; values.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

#[derive(Debug)]
pub struct LeakyReluBackward {
    pub x: Tensor,
    pub negative_slope: f32,
}
impl BackwardOp for LeakyReluBackward {
    fn name(&self) -> &'static str {
        "leaky_relu_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "leaky_relu_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let a = self.negative_slope;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| if xi > 0.0 { dyi } else { dyi * a })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
}

#[derive(Debug)]
pub struct EluBackward {
    pub x: Tensor,
    pub alpha: f32,
}
impl BackwardOp for EluBackward {
    fn name(&self) -> &'static str {
        "elu_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "elu_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let a = self.alpha;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                if xi > 0.0 {
                    dyi
                } else {
                    dyi * a * xi.exp()
                }
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
}

#[derive(Debug)]
pub struct SoftplusBackward {
    pub x: Tensor,
}
impl BackwardOp for SoftplusBackward {
    fn name(&self) -> &'static str {
        "softplus_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "softplus_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        // sigmoid(x) computed stably.
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                let s = if xi >= 0.0 {
                    1.0 / (1.0 + (-xi).exp())
                } else {
                    let e = xi.exp();
                    e / (1.0 + e)
                };
                dyi * s
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
}

#[derive(Debug)]
pub struct MishBackward {
    pub x: Tensor,
}
impl BackwardOp for MishBackward {
    fn name(&self) -> &'static str {
        "mish_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "mish_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                // softplus = log(1 + exp(x)) — stable form.
                let sp = if xi >= 0.0 {
                    xi + (1.0 + (-xi).exp()).ln()
                } else {
                    (1.0 + xi.exp()).ln()
                };
                let t = sp.tanh();
                let sigmoid = if xi >= 0.0 {
                    1.0 / (1.0 + (-xi).exp())
                } else {
                    let e = xi.exp();
                    e / (1.0 + e)
                };
                let sech_sq = 1.0 - t * t;
                dyi * (t + xi * sech_sq * sigmoid)
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::Tensor;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn first(g: Vec<Option<Tensor>>) -> Tensor {
        g.into_iter().next().unwrap().unwrap()
    }

    #[test]
    fn leaky_relu_positive_branch() {
        let bwd = LeakyReluBackward {
            x: Tensor::from_slice(&[2.0f32], vec![1]).unwrap(),
            negative_slope: 0.1,
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn leaky_relu_negative_branch() {
        let bwd = LeakyReluBackward {
            x: Tensor::from_slice(&[-2.0f32], vec![1]).unwrap(),
            negative_slope: 0.1,
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn elu_positive_branch_equals_dy() {
        let bwd = EluBackward {
            x: Tensor::from_slice(&[1.0f32], vec![1]).unwrap(),
            alpha: 1.0,
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn elu_at_zero_is_alpha() {
        // At x=0: d/dx elu = α * exp(0) = α
        let bwd = EluBackward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
            alpha: 1.5,
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        // x=0 is the > 0 branch (false → uses alpha * exp(0) = alpha).
        assert!((read_f32(&g)[0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn softplus_at_zero_is_half() {
        // d/dx softplus(x) at x=0 = sigmoid(0) = 0.5
        let bwd = SoftplusBackward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn softplus_finite_difference() {
        let x_val = 0.3f32;
        let eps = 1e-3f32;
        let sp = |x: f32| (1.0 + x.exp()).ln();
        let analytic = 1.0 / (1.0 + (-x_val).exp());
        let numeric = (sp(x_val + eps) - sp(x_val - eps)) / (2.0 * eps);
        assert!((analytic - numeric).abs() < 1e-3);

        let bwd = SoftplusBackward {
            x: Tensor::from_slice(&[x_val], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - analytic).abs() < 1e-4);
    }

    #[test]
    fn mish_finite_difference() {
        let x_val = 0.7f32;
        let eps = 1e-3f32;
        let mish = |x: f32| x * (1.0 + x.exp()).ln().tanh();
        let analytic = first(
            MishBackward {
                x: Tensor::from_slice(&[x_val], vec![1]).unwrap(),
            }
            .apply(&Tensor::from_slice(&[1.0f32], vec![1]).unwrap())
            .unwrap(),
        );
        let analytic_v = read_f32(&analytic)[0];
        let numeric = (mish(x_val + eps) - mish(x_val - eps)) / (2.0 * eps);
        assert!(
            (analytic_v - numeric).abs() < 1e-3,
            "analytic {analytic_v} vs numeric {numeric}"
        );
    }
}

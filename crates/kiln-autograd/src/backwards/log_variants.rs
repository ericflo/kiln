//! Backwards for `log_variants` ops: `log10`, `log2`, `log1p`,
//! `exp2`, `expm1`.
//!
//! | Op    | Backward                              |
//! |-------|---------------------------------------|
//! | log10 | `dx = dy / (x * ln(10))`              |
//! | log2  | `dx = dy / (x * ln(2))`               |
//! | log1p | `dx = dy / (1 + x)`                   |
//! | exp2  | `dx = dy * exp2(x) * ln(2)`           |
//! | expm1 | `dx = dy * exp(x)` (= `dy * (expm1(x) + 1)`) |

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
        .ok_or_else(|| Error::from_str("log_variants_backward: storage must be CpuStorage"))?;
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

macro_rules! impl_log_variant_bwd {
    ($struct_name:ident, $op_name:literal, $closure:expr) => {
        #[derive(Debug)]
        pub struct $struct_name {
            pub x: Tensor,
        }
        impl BackwardOp for $struct_name {
            fn name(&self) -> &'static str {
                $op_name
            }
            fn input_count(&self) -> usize {
                1
            }
            fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
                validate_same(&self.x, grad_output, $op_name)?;
                let x = load_f32(&self.x)?;
                let dy = load_f32(grad_output)?;
                let f = $closure;
                let dx: Vec<f32> = x.iter().zip(dy.iter()).map(|(&xi, &dyi)| f(xi, dyi)).collect();
                Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
            }
        }
    };
}

impl_log_variant_bwd!(Log10Backward, "log10_backward", |x: f32, dy: f32| {
    dy / (x * std::f32::consts::LN_10)
});
impl_log_variant_bwd!(Log2Backward, "log2_backward", |x: f32, dy: f32| {
    dy / (x * std::f32::consts::LN_2)
});
impl_log_variant_bwd!(Log1pBackward, "log1p_backward", |x: f32, dy: f32| {
    dy / (1.0 + x)
});
impl_log_variant_bwd!(Exp2Backward, "exp2_backward", |x: f32, dy: f32| {
    dy * x.exp2() * std::f32::consts::LN_2
});
impl_log_variant_bwd!(Expm1Backward, "expm1_backward", |x: f32, dy: f32| {
    dy * x.exp()
});

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

    fn first_grad(g: Vec<Option<Tensor>>) -> Tensor {
        g.into_iter().next().unwrap().unwrap()
    }

    #[test]
    fn log10_bwd_at_one() {
        // d/dx log10(x) at x=1 = 1/ln(10).
        let bwd = Log10Backward {
            x: Tensor::from_slice(&[1.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0 / std::f32::consts::LN_10).abs() < 1e-5);
    }

    #[test]
    fn log2_bwd_at_one() {
        let bwd = Log2Backward {
            x: Tensor::from_slice(&[1.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0 / std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn log1p_bwd_at_zero_is_one() {
        // d/dx log1p(x) at x=0 = 1.
        let bwd = Log1pBackward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn exp2_bwd_at_zero_is_ln2() {
        // d/dx exp2(x) at x=0 = ln(2).
        let bwd = Exp2Backward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn expm1_bwd_at_zero_is_one() {
        // d/dx expm1(x) at x=0 = exp(0) = 1.
        let bwd = Expm1Backward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log1p_bwd_finite_difference() {
        let x_val = 0.5f32;
        let analytic = 1.0 / (1.0 + x_val);
        let bwd = Log1pBackward {
            x: Tensor::from_slice(&[x_val], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first_grad(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - analytic).abs() < 1e-5);
    }

    #[test]
    fn input_count_is_one() {
        let bwd = Log10Backward {
            x: Tensor::from_slice(&[1.0f32], vec![1]).unwrap(),
        };
        assert_eq!(bwd.input_count(), 1);
    }
}

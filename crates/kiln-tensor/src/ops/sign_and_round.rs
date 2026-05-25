//! `sign`, `floor`, `ceil`, `round`, `trunc`, `reciprocal`.
//!
//! Elementwise primitives. All non-differentiable in the
//! mathematical sense (piecewise constant or sparse-derivative),
//! so no BackwardOp.
//!
//! # CUDA wiring (#1082)
//!
//! Routes through the shared `cuda_activation_unary` kernel.
//! All six kinds now route through CUDA via the shared kernel:
//! `recip` (22), `sign` (23), `floor` (24), `ceil` (25),
//! `round` (26), `trunc` (27) — all landed in the #1082 series.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignRoundKind {
    Sign,
    Floor,
    Ceil,
    Round,
    Trunc,
    Reciprocal,
}

impl SignRoundKind {
    const fn name(self) -> &'static str {
        match self {
            SignRoundKind::Sign => "sign",
            SignRoundKind::Floor => "floor",
            SignRoundKind::Ceil => "ceil",
            SignRoundKind::Round => "round",
            SignRoundKind::Trunc => "trunc",
            SignRoundKind::Reciprocal => "reciprocal",
        }
    }

    fn apply_f32(self, v: f32) -> f32 {
        match self {
            SignRoundKind::Sign => {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            SignRoundKind::Floor => v.floor(),
            SignRoundKind::Ceil => v.ceil(),
            SignRoundKind::Round => v.round(),
            SignRoundKind::Trunc => v.trunc(),
            SignRoundKind::Reciprocal => 1.0 / v,
        }
    }

    /// CUDA kernel kind tag matching the `KIND_*` constants in
    /// `csrc/activation.cu`. All six kinds in this family now have
    /// CUDA paths: recip (22), sign (23), floor (24), ceil (25),
    /// round (26), trunc (27) — the last (#1082) landed with this
    /// commit.
    #[cfg(feature = "cuda")]
    const fn cuda_kind_tag(self) -> i32 {
        match self {
            SignRoundKind::Reciprocal => 22,
            SignRoundKind::Sign => 23,
            SignRoundKind::Floor => 24,
            SignRoundKind::Ceil => 25,
            SignRoundKind::Round => 26,
            SignRoundKind::Trunc => 27,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SignRoundOp {
    kind: SignRoundKind,
}

impl DeviceOp1 for SignRoundOp {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        let t = cpu_apply(self.kind, x)?;
        Ok(Some(t))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        Ok(Some(crate::cuda_activation_unary(x, self.kind.cuda_kind_tag())?))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Metal): once a Metal sign/round kernel
        // ships, dispatch via the kind tag and route through
        // `crate::metal_activation_unary`. Until then, fall through
        // to CPU.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): once a Vulkan sign/round
        // kernel ships, dispatch via the kind tag and route through
        // `crate::vulkan_activation_unary`. Until then, fall
        // through to CPU.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

fn validate(x: &Tensor, name: &str) -> Result<()> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    Ok(())
}

fn cpu_apply(kind: SignRoundKind, x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("sign_and_round: storage must be CpuStorage"))?;
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
        let y = kind.apply_f32(v);
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

pub fn sign(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Sign }, x)
}

pub fn floor(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Floor }, x)
}

pub fn ceil(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Ceil }, x)
}

pub fn round(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Round }, x)
}

pub fn trunc(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Trunc }, x)
}

pub fn reciprocal(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SignRoundOp { kind: SignRoundKind::Reciprocal }, x)
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
    fn sign_three_way() {
        let x = Tensor::from_slice(&[-3.0f32, -0.5, 0.0, 0.5, 3.0], vec![5]).unwrap();
        assert_eq!(read_f32(&sign(&x).unwrap()), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn floor_ceil_round_trunc() {
        let x = Tensor::from_slice(&[1.3f32, -1.3, 2.7, -2.7, 0.5, -0.5], vec![6]).unwrap();
        let f = read_f32(&floor(&x).unwrap());
        let c = read_f32(&ceil(&x).unwrap());
        let r = read_f32(&round(&x).unwrap());
        let t = read_f32(&trunc(&x).unwrap());
        assert_eq!(f, vec![1.0, -2.0, 2.0, -3.0, 0.0, -1.0]);
        assert_eq!(c, vec![2.0, -1.0, 3.0, -2.0, 1.0, 0.0]);
        // Rust's round() rounds half away from zero.
        assert_eq!(r, vec![1.0, -1.0, 3.0, -3.0, 1.0, -1.0]);
        assert_eq!(t, vec![1.0, -1.0, 2.0, -2.0, 0.0, 0.0]);
    }

    #[test]
    fn reciprocal_inverts() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 4.0, -0.5], vec![4]).unwrap();
        let y = read_f32(&reciprocal(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 0.5).abs() < 1e-6);
        assert!((y[2] - 0.25).abs() < 1e-6);
        assert!((y[3] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.5f32, -1.5]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        assert_eq!(floor(&x).unwrap().dtype(), DType::BF16);
        assert_eq!(sign(&x).unwrap().dtype(), DType::BF16);
    }
}

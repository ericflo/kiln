//! Tensor-scalar elementwise ops: `add_scalar`, `sub_scalar`,
//! `mul_scalar`, `div_scalar`.
//!
//! Different from `add`/`mul` which require two same-shape tensors.
//! These take a tensor + an f32 scalar; output has the same shape.
//!
//! Saves having to broadcast_to a same-shape filled tensor for every
//! call site that just wants `x * 0.5`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

/// Tensor-scalar elementwise op kind. Carried by [`ScalarOp`].
///
/// Tags `0..=3` cover the four named entry points. Tags `4..=7`
/// are extra variants the CUDA kernel supports for future use
/// (`c - x`, `c / x`, `max(x, c)`, `min(x, c)`) but are not exposed
/// as public Rust entry points yet — they exist so the kernel ABI is
/// future-proof per #1082.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    AddScalar = 0,
    SubScalar = 1,
    MulScalar = 2,
    DivScalar = 3,
}

impl ScalarKind {
    pub const fn name(self) -> &'static str {
        match self {
            ScalarKind::AddScalar => "add_scalar",
            ScalarKind::SubScalar => "sub_scalar",
            ScalarKind::MulScalar => "mul_scalar",
            ScalarKind::DivScalar => "div_scalar",
        }
    }

    fn apply_f32(self, x: f32, c: f32) -> f32 {
        match self {
            ScalarKind::AddScalar => x + c,
            ScalarKind::SubScalar => x - c,
            ScalarKind::MulScalar => x * c,
            ScalarKind::DivScalar => x / c,
        }
    }

    fn cuda_tag(self) -> i32 {
        match self {
            ScalarKind::AddScalar => 0,
            ScalarKind::SubScalar => 1,
            ScalarKind::MulScalar => 2,
            ScalarKind::DivScalar => 3,
        }
    }
}

/// Tensor-scalar elementwise op. `out[i] = op(x[i], c)`.
///
/// Multiple [`ScalarKind`] variants share one struct so the call
/// sites in [`add_scalar`] / [`sub_scalar`] / [`mul_scalar`] /
/// [`div_scalar`] go through the same dispatch path and benefit
/// from the same CUDA kernel (`csrc/scalar_op.cu`).
#[derive(Debug, Clone, Copy)]
pub struct ScalarOp {
    kind: ScalarKind,
    c: f32,
}

impl ScalarOp {
    pub const fn new(kind: ScalarKind, c: f32) -> Self {
        ScalarOp { kind, c }
    }
    pub const fn kind(self) -> ScalarKind {
        self.kind
    }
    pub const fn scalar(self) -> f32 {
        self.c
    }
}

impl DeviceOp1 for ScalarOp {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind)?;
        let dtype = x.dtype();
        let cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("ScalarOp: storage must be CpuStorage"))?;
        let bytes = cpu.as_bytes();
        let n = x.element_count();
        let per = dtype.size_in_bytes();
        let mut out = vec![0u8; n * per];
        let c = self.c;
        for i in 0..n {
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            let y = self.kind.apply_f32(v, c);
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
        let t = Tensor::from_parts(storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())?;
        Ok(Some(t))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind)?;
        let dtype = x.dtype();
        if !x.is_contiguous() {
            // Fallthrough to CPU when input is not contiguous —
            // dispatch1 will retry on CPU automatically.
            return Ok(None);
        }
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        Ok(Some(crate::cuda_scalar_op(x, self.kind.cuda_tag(), self.c)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

fn validate(x: &Tensor, kind: ScalarKind) -> Result<()> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "ScalarOp({}): dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() && x.device().is_cpu() {
        bail!("ScalarOp({}): input must be contiguous", kind.name());
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Convenience wrappers (public entry points). Signatures unchanged
// from the pre-#1082 function-only form so callers (addmm, sdpa, etc.)
// continue to compile.
// ----------------------------------------------------------------------

/// `out = x + c`.
pub fn add_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    dispatch1(&ScalarOp::new(ScalarKind::AddScalar, c), x)
}

/// `out = x - c`.
pub fn sub_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    dispatch1(&ScalarOp::new(ScalarKind::SubScalar, c), x)
}

/// `out = x * c`.
pub fn mul_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    dispatch1(&ScalarOp::new(ScalarKind::MulScalar, c), x)
}

/// `out = x / c`. `c` must be non-zero.
pub fn div_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    if c == 0.0 {
        bail!("div_scalar: divisor must be non-zero");
    }
    dispatch1(&ScalarOp::new(ScalarKind::DivScalar, c), x)
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
    fn add_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = add_scalar(&x, 10.0).unwrap();
        assert_eq!(read_f32(&y), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn sub_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = sub_scalar(&x, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn mul_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = mul_scalar(&x, 0.5).unwrap();
        assert_eq!(read_f32(&y), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn div_scalar_works() {
        let x = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![3]).unwrap();
        let y = div_scalar(&x, 2.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn div_by_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = div_scalar(&x, 0.0).unwrap_err();
        assert!(e.to_string().contains("non-zero"));
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        assert_eq!(add_scalar(&x, 1.0).unwrap().dtype(), DType::BF16);
    }

    #[test]
    fn op_kind_round_trips() {
        let op = ScalarOp::new(ScalarKind::MulScalar, 0.25);
        assert_eq!(op.kind(), ScalarKind::MulScalar);
        assert!((op.scalar() - 0.25).abs() < f32::EPSILON);
        assert_eq!(op.name(), "mul_scalar");
        assert!(matches!(op.determinism(), Determinism::Constructive));
    }
}

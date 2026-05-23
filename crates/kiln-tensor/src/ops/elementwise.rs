//! Element-wise binary ops: `add`, `sub`, `mul`, `div`.
//!
//! Replaces candle's `Tensor::{add, sub, mul, div}` and matches the
//! 14 `kiln/residual` NVTX call sites (`kiln/residual`,
//! `kiln/residual_batch_decode`, etc.) from Phase 0.7's preserve-list.
//!
//! # Semantics
//!
//! Both inputs share the same shape and dtype. The output is the
//! pointwise application of the binary op, same shape, same dtype.
//!
//! Broadcast and stride-aware iteration land in a follow-up — Phase
//! 1.15 ships the exact-shape contiguous path, which covers every
//! residual call site in `forward.rs`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Binary element-wise operation kind. Carried by [`ElementwiseOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryKind {
    pub const fn name(self) -> &'static str {
        match self {
            BinaryKind::Add => "add",
            BinaryKind::Sub => "sub",
            BinaryKind::Mul => "mul",
            BinaryKind::Div => "div",
        }
    }

    fn apply_f32(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryKind::Add => a + b,
            BinaryKind::Sub => a - b,
            BinaryKind::Mul => a * b,
            BinaryKind::Div => a / b,
        }
    }
}

/// Element-wise binary op. Same-shape, same-dtype.
///
/// Multiple [`BinaryKind`] variants share one struct so the call sites
/// in [`add`] / [`sub`] / [`mul`] / [`div`] go through the same
/// dispatch path.
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseOp {
    kind: BinaryKind,
}

impl ElementwiseOp {
    pub const fn new(kind: BinaryKind) -> Self {
        ElementwiseOp { kind }
    }
    pub const fn kind(self) -> BinaryKind {
        self.kind
    }
}

impl DeviceOp2 for ElementwiseOp {
    fn name(&self) -> &'static str {
        // Static-string per kind — these are referenced by NVTX range
        // names + parity-tolerance CSV keys.
        match self.kind {
            BinaryKind::Add => "add",
            BinaryKind::Sub => "sub",
            BinaryKind::Mul => "mul",
            BinaryKind::Div => "div",
        }
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
        validate(a, b, self.kind)?;
        let dtype = a.dtype();
        let a_cpu = downcast_cpu(a, "a")?;
        let b_cpu = downcast_cpu(b, "b")?;
        let a_bytes = a_cpu.as_bytes();
        let b_bytes = b_cpu.as_bytes();
        let n_elements = a.element_count();
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; n_elements * per];

        match dtype {
            DType::F32 => {
                for i in 0..n_elements {
                    let av = read_f32(a_bytes, i);
                    let bv = read_f32(b_bytes, i);
                    let cv = self.kind.apply_f32(av, bv);
                    out_bytes[i * 4..i * 4 + 4].copy_from_slice(&cv.to_le_bytes());
                }
            }
            DType::BF16 => {
                for i in 0..n_elements {
                    let av = read_bf16(a_bytes, i).to_f32();
                    let bv = read_bf16(b_bytes, i).to_f32();
                    let cv = self.kind.apply_f32(av, bv);
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(cv).to_le_bytes());
                }
            }
            DType::F16 => {
                for i in 0..n_elements {
                    let av = read_f16(a_bytes, i).to_f32();
                    let bv = read_f16(b_bytes, i).to_f32();
                    let cv = self.kind.apply_f32(av, bv);
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(cv).to_le_bytes());
                }
            }
            other => bail!(
                "ElementwiseOp({}): dtype {other} not supported (need F32/BF16/F16)",
                self.kind.name()
            ),
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let layout = Layout::contiguous(a.shape().to_vec());
        let out = Tensor::from_parts(storage, layout, TensorId::next())?;
        Ok(Some(out))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

// ----------------------------------------------------------------------
// Convenience wrappers
// ----------------------------------------------------------------------

/// `out = a + b` (same shape, same dtype). The migration target for
/// `kiln/residual` and friends.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    dispatch2(&ElementwiseOp::new(BinaryKind::Add), a, b)
}

/// `out = a - b`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    dispatch2(&ElementwiseOp::new(BinaryKind::Sub), a, b)
}

/// `out = a * b` (element-wise, NOT matmul).
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    dispatch2(&ElementwiseOp::new(BinaryKind::Mul), a, b)
}

/// `out = a / b`. Caller is responsible for non-zero divisor;
/// element-wise division by zero produces ±inf or NaN per IEEE-754.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    dispatch2(&ElementwiseOp::new(BinaryKind::Div), a, b)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(a: &Tensor, b: &Tensor, kind: BinaryKind) -> Result<()> {
    if a.shape() != b.shape() {
        bail!(
            "ElementwiseOp({}): shape mismatch {:?} vs {:?} (broadcasting not in Phase 1.15)",
            kind.name(),
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!(
            "ElementwiseOp({}): dtype mismatch {} vs {}",
            kind.name(),
            a.dtype(),
            b.dtype()
        );
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "ElementwiseOp({}): dtype must be F32/BF16/F16, got {}",
            kind.name(),
            a.dtype()
        );
    }
    if !a.is_contiguous() {
        bail!(
            "ElementwiseOp({}): a must be contiguous (stride-aware path not in Phase 1.15)",
            kind.name()
        );
    }
    if !b.is_contiguous() {
        bail!(
            "ElementwiseOp({}): b must be contiguous",
            kind.name()
        );
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("ElementwiseOp: {label} storage must be CpuStorage")))
}

fn read_f32(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
}

fn read_bf16(bytes: &[u8], i: usize) -> half::bf16 {
    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
}

fn read_f16(bytes: &[u8], i: usize) -> half::f16 {
    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_back_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn add_f32_exact() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(read_back_f32(&out), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sub_f32() {
        let a = Tensor::from_slice(&[5.0f32, 10.0, 15.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let out = sub(&a, &b).unwrap();
        assert_eq!(read_back_f32(&out), vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn mul_f32() {
        let a = Tensor::from_slice(&[2.0f32, 3.0, 4.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0], vec![3]).unwrap();
        let out = mul(&a, &b).unwrap();
        assert_eq!(read_back_f32(&out), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn div_f32() {
        let a = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![3]).unwrap();
        let out = div(&a, &b).unwrap();
        assert_eq!(read_back_f32(&out), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn add_bf16_round_trip() {
        let av: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let bv: Vec<half::bf16> = [0.5f32, 1.5, 2.5]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let a = Tensor::from_slice(&av, vec![3]).unwrap();
        let b = Tensor::from_slice(&bv, vec![3]).unwrap();
        let out = add(&a, &b).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        for (i, &expected) in [1.5f32, 3.5, 5.5].iter().enumerate() {
            let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
            // BF16 ULP tolerance per parity-tolerance.csv elementwise row.
            assert!((v.to_f32() - expected).abs() < 1e-2, "{} vs {}", v.to_f32(), expected);
        }
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = add(&a, &b).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn add_rejects_dtype_mismatch() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bv: Vec<half::bf16> = [1.0f32, 2.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let b = Tensor::from_slice(&bv, vec![2]).unwrap();
        let e = add(&a, &b).unwrap_err();
        assert!(e.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn add_rejects_packed_dtype() {
        // We can't construct a packed tensor via from_slice (no Element
        // impl), so reject indirectly via the validate dtype-allowlist.
        let a = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let e = add(&a, &b).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn binary_kind_name_strings() {
        assert_eq!(BinaryKind::Add.name(), "add");
        assert_eq!(BinaryKind::Sub.name(), "sub");
        assert_eq!(BinaryKind::Mul.name(), "mul");
        assert_eq!(BinaryKind::Div.name(), "div");
    }

    #[test]
    fn op_metadata() {
        let op = ElementwiseOp::new(BinaryKind::Add);
        assert_eq!(op.name(), "add");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
        assert_eq!(op.kind(), BinaryKind::Add);
    }
}

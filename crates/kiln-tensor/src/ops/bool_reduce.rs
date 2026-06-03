//! `all` / `any` — boolean reduction along an axis (or full reduction).
//!
//! Takes a U8 mask (typical output of `eq` / `lt` / etc.) and
//! reduces it along an axis (or all axes). Output is U8.
//!
//! - `all(mask, axis)` returns 1 iff every element along `axis` is
//!   nonzero.
//! - `any(mask, axis)` returns 1 iff at least one element is nonzero.
//!
//! Same shape rules as `sum_axis` (axis removed); full-reduction
//! variant produces a rank-0 scalar.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Materialize `t` on CPU. CUDA inputs are D2H-copied via
/// `cuda_to_host_copy`; CPU inputs are cheap `Arc` bumps. `all` /
/// `any` reduce U8 masks that are typically the output of `eq` /
/// `lt` / `compare` — those ops have CUDA fast-paths now (see
/// `ops/compare.rs`), so the masks land on CUDA. The boolean
/// reduction here runs in Rust; the host copy lets it accept CUDA
/// masks transparently. See `#1082`.
fn to_cpu(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_to_host_copy(t);
        }
    }
    Ok(t.clone())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolReduce {
    All,
    Any,
}

impl BoolReduce {
    pub const fn name(self) -> &'static str {
        match self {
            BoolReduce::All => "all",
            BoolReduce::Any => "any",
        }
    }
}

fn apply_axis(kind: BoolReduce, mask: &Tensor, axis: usize) -> Result<Tensor> {
    if mask.dtype() != DType::U8 {
        bail!(
            "{}: mask dtype must be U8, got {}",
            kind.name(),
            mask.dtype()
        );
    }
    if axis >= mask.rank() {
        bail!(
            "{}: axis {axis} out of range for rank-{}",
            kind.name(),
            mask.rank()
        );
    }
    if !mask.is_contiguous() {
        bail!("{}: mask must be contiguous", kind.name());
    }
    let shape = mask.shape().to_vec();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_dim = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let mask_host = to_cpu(mask)?;
    let cpu = mask_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("bool_reduce: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = match kind {
                BoolReduce::All => true,
                BoolReduce::Any => false,
            };
            for a in 0..axis_dim {
                let idx = (o * axis_dim + a) * inner + i;
                let b = bytes[idx] != 0;
                acc = match kind {
                    BoolReduce::All => acc && b,
                    BoolReduce::Any => acc || b,
                };
            }
            out[o * inner + i] = if acc { 1 } else { 0 };
        }
    }
    let mut out_shape = shape;
    out_shape.remove(axis);
    let cpu_out = CpuStorage::from_bytes(DType::U8, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
}

fn apply_all_axes(kind: BoolReduce, mask: &Tensor) -> Result<Tensor> {
    if mask.dtype() != DType::U8 {
        bail!(
            "{}: mask dtype must be U8, got {}",
            kind.name(),
            mask.dtype()
        );
    }
    if !mask.is_contiguous() {
        bail!("{}: mask must be contiguous", kind.name());
    }
    let mask_host = to_cpu(mask)?;
    let cpu = mask_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("bool_reduce: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let result = match kind {
        BoolReduce::All => bytes.iter().all(|&b| b != 0),
        BoolReduce::Any => bytes.iter().any(|&b| b != 0),
    };
    let cpu_out = CpuStorage::from_bytes(DType::U8, vec![if result { 1u8 } else { 0u8 }])?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

/// ROCm: correctness-first host round-trip for a boolean reduction. Stage the
/// mask to host, run `f` (the recursive CPU path), and move the U8 result back
/// to the input device so the op is transparent to ROCm callers. No-op (returns
/// `None`) for non-ROCm inputs.
#[cfg(feature = "rocm")]
fn rocm_roundtrip(
    mask: &Tensor,
    f: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Option<Tensor>> {
    if matches!(mask.device(), crate::Device::Rocm(_)) {
        let dev = mask.device();
        let host = crate::rocm_to_host_copy(mask)?;
        let out_host = f(&host)?;
        return Ok(Some(out_host.to_device(dev)?));
    }
    Ok(None)
}

pub fn all_axis(mask: &Tensor, axis: usize) -> Result<Tensor> {
    // ROCm fast path: native per-axis boolean reduction (`kind == 0` is ALL).
    // Requires a U8, contiguous, rank>=1 ROCm mask with `axis < rank` — exactly
    // what the native `reduce_arbitrary_axis.cu` kernel accepts (Phase R.5). No
    // host round-trip. The `axis_dim > 0` guard avoids the kernel's empty-axis
    // early-return (which leaves the uninit output unwritten); the empty-axis
    // identity (ALL -> 1) is handled by the host path below. Any guard failure
    // falls through to the host path.
    #[cfg(feature = "rocm")]
    if matches!(mask.device(), crate::Device::Rocm(_))
        && mask.dtype() == DType::U8
        && mask.is_contiguous()
        && mask.rank() >= 1
        && axis < mask.rank()
        && mask.shape()[axis] > 0
    {
        return crate::rocm_bool_reduce_axis(mask, axis, 0);
    }
    #[cfg(feature = "rocm")]
    if let Some(out) = rocm_roundtrip(mask, |m| all_axis(m, axis))? {
        return Ok(out);
    }
    apply_axis(BoolReduce::All, mask, axis)
}
pub fn any_axis(mask: &Tensor, axis: usize) -> Result<Tensor> {
    // ROCm fast path: native per-axis boolean reduction (`kind == 1` is ANY).
    // Same guards as `all_axis` (incl. the `axis_dim > 0` empty-axis guard);
    // falls through to host on any guard miss.
    #[cfg(feature = "rocm")]
    if matches!(mask.device(), crate::Device::Rocm(_))
        && mask.dtype() == DType::U8
        && mask.is_contiguous()
        && mask.rank() >= 1
        && axis < mask.rank()
        && mask.shape()[axis] > 0
    {
        return crate::rocm_bool_reduce_axis(mask, axis, 1);
    }
    #[cfg(feature = "rocm")]
    if let Some(out) = rocm_roundtrip(mask, |m| any_axis(m, axis))? {
        return Ok(out);
    }
    apply_axis(BoolReduce::Any, mask, axis)
}
pub fn all_reduce(mask: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    if let Some(out) = rocm_roundtrip(mask, all_reduce)? {
        return Ok(out);
    }
    apply_all_axes(BoolReduce::All, mask)
}
pub fn any_reduce(mask: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    if let Some(out) = rocm_roundtrip(mask, any_reduce)? {
        return Ok(out);
    }
    apply_all_axes(BoolReduce::Any, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u8(t: &Tensor) -> Vec<u8> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes().to_vec()
    }

    #[test]
    fn all_axis_simple() {
        // [[1, 1, 1], [1, 0, 1]] all axis 1 → [1, 0]
        let m = Tensor::from_slice(&[1u8, 1, 1, 1, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&all_axis(&m, 1).unwrap()), vec![1, 0]);
    }

    #[test]
    fn any_axis_simple() {
        // [[1, 0, 0], [0, 0, 0]] any axis 1 → [1, 0]
        let m = Tensor::from_slice(&[1u8, 0, 0, 0, 0, 0], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&any_axis(&m, 1).unwrap()), vec![1, 0]);
    }

    #[test]
    fn all_axis_0() {
        // [[1, 1, 0], [1, 1, 1]] all axis 0 → [1, 1, 0]
        let m = Tensor::from_slice(&[1u8, 1, 0, 1, 1, 1], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&all_axis(&m, 0).unwrap()), vec![1, 1, 0]);
    }

    #[test]
    fn all_reduce_scalar() {
        let m = Tensor::from_slice(&[1u8, 1, 1, 1], vec![4]).unwrap();
        let y = all_reduce(&m).unwrap();
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(read_u8(&y), vec![1]);
        let m = Tensor::from_slice(&[1u8, 0, 1, 1], vec![4]).unwrap();
        assert_eq!(read_u8(&all_reduce(&m).unwrap()), vec![0]);
    }

    #[test]
    fn any_reduce_scalar() {
        let m = Tensor::from_slice(&[0u8; 4], vec![4]).unwrap();
        assert_eq!(read_u8(&any_reduce(&m).unwrap()), vec![0]);
        let m = Tensor::from_slice(&[0u8, 0, 1, 0], vec![4]).unwrap();
        assert_eq!(read_u8(&any_reduce(&m).unwrap()), vec![1]);
    }

    #[test]
    fn mask_dtype_errors() {
        let m = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = all_axis(&m, 0).unwrap_err();
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn kind_names() {
        assert_eq!(BoolReduce::All.name(), "all");
        assert_eq!(BoolReduce::Any.name(), "any");
    }

    /// CUDA parity: lifting a U8 mask onto CUDA and reducing it
    /// produces byte-equal output vs the CPU path for both
    /// per-axis and full reductions, for both All and Any.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_bool_reduce_parity() {
        // Skip if no CUDA device available at runtime (cfg-feature gate
        // only confirms the build was compiled with --features cuda).
        if crate::primary_cuda_context(0).is_err() {
            return;
        }

        // [[1, 1, 1], [1, 0, 1]] — same pattern as all_axis_simple.
        let mask_cpu = Tensor::from_slice(&[1u8, 1, 1, 1, 0, 1], vec![2, 3]).unwrap();
        let mask_cuda = mask_cpu
            .to_device(crate::Device::Cuda(0))
            .unwrap();

        // Per-axis reductions.
        assert_eq!(
            read_u8(&all_axis(&mask_cpu, 1).unwrap()),
            read_u8(&all_axis(&mask_cuda, 1).unwrap()),
            "all_axis 1 cuda parity"
        );
        assert_eq!(
            read_u8(&any_axis(&mask_cpu, 1).unwrap()),
            read_u8(&any_axis(&mask_cuda, 1).unwrap()),
            "any_axis 1 cuda parity"
        );
        assert_eq!(
            read_u8(&all_axis(&mask_cpu, 0).unwrap()),
            read_u8(&all_axis(&mask_cuda, 0).unwrap()),
            "all_axis 0 cuda parity"
        );

        // Full reductions.
        assert_eq!(
            read_u8(&all_reduce(&mask_cpu).unwrap()),
            read_u8(&all_reduce(&mask_cuda).unwrap()),
            "all_reduce cuda parity"
        );
        assert_eq!(
            read_u8(&any_reduce(&mask_cpu).unwrap()),
            read_u8(&any_reduce(&mask_cuda).unwrap()),
            "any_reduce cuda parity"
        );
    }
}


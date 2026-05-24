//! `concat` — join N tensors along a given axis.
//!
//! Required for sequence batching, attention head concatenation, and
//! the residual-stream merge in transformer blocks.
//!
//! # Semantics
//!
//! Given `inputs: [T1, T2, …, TN]` and `axis: A`:
//!
//! - Every input must share the same rank and the same shape along
//!   every axis **except** `A`.
//! - The output has shape `inputs[0].shape` with `axis` replaced by
//!   the sum of the inputs' `axis` sizes.
//! - All inputs must share dtype.
//!
//! # Variable arity
//!
//! Concat is naturally variable-arity, which doesn't fit the
//! `DeviceOp{1,2,3}` trait surface. The op is exposed as a free
//! function plus a [`ConcatOp`] handle carrying the `axis`. Per-
//! backend impls plug in via the convenience function:
//! - CPU path: byte-wise per-outer-slab copy (in `concat()`).
//! - CUDA path: dispatches through `cuda_concat` (in
//!   `crates/kiln-tensor/src/cuda_storage.rs`) when all inputs are
//!   CUDA-backed. The kernel performs the same per-(outer, axis,
//!   inner) byte copy as the CPU reference, in a single launch.
//!
//! # Determinism
//!
//! Constructive. Fixed iteration order over (input_index, slice
//! offset); bit-identical at the same input dtype.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Concat op handle. Carries the concat axis. Convenience for naming
/// + future per-backend dispatch.
#[derive(Debug, Clone, Copy)]
pub struct ConcatOp {
    axis: usize,
}

impl ConcatOp {
    pub const fn new(axis: usize) -> Self {
        ConcatOp { axis }
    }
    pub const fn axis(self) -> usize {
        self.axis
    }
    pub fn name(&self) -> &'static str {
        "concat"
    }

    /// Forward via the variable-arity dispatch in [`concat`]. Mirrors
    /// the `cpu_fwd`/`cuda_fwd` surface of single-input ops for code
    /// that wants to invoke through the `ConcatOp` handle. The
    /// dispatch picks the CUDA fast path automatically when all inputs
    /// are CUDA-backed.
    pub fn fwd(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        concat(inputs, self.axis)
    }

    /// CUDA-only forward: routes directly through the CUDA substrate
    /// kernel. Returns `Ok(None)` (rather than erroring) when any
    /// input is not on CUDA — keeps the per-Op surface symmetric with
    /// `DeviceOp{1,2,3}::cuda_fwd`.
    #[cfg(feature = "cuda")]
    pub fn cuda_fwd(&self, inputs: &[&Tensor]) -> Result<Option<Tensor>> {
        if inputs.is_empty() {
            return Ok(None);
        }
        if inputs.len() > 32 {
            return Ok(None);
        }
        if !inputs
            .iter()
            .all(|t| matches!(t.device(), crate::Device::Cuda(_)))
        {
            return Ok(None);
        }
        let out = crate::cuda_concat(inputs, self.axis)?;
        Ok(Some(out))
    }

    /// Metal-only forward scaffold: symmetric with `cuda_fwd` but
    /// currently returns `Ok(None)` so dispatch falls through to the
    /// CPU path. Phase 4 Metal kernel author drops in
    /// `crate::metal_concat(inputs, self.axis)` once it ships.
    #[cfg(feature = "metal")]
    pub fn metal_fwd(&self, inputs: &[&Tensor]) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so the future
        // MSL kernel can drop in without changing the call site:
        //   - non-empty inputs
        //   - <= 32 inputs (mirrors CUDA substrate's MAX_INPUTS cap;
        //     beyond that the CPU path's per-slab byte copy is fine)
        //   - all inputs on Metal device
        if inputs.is_empty() {
            return Ok(None);
        }
        if inputs.len() > 32 {
            return Ok(None);
        }
        if !inputs
            .iter()
            .all(|t| matches!(t.device(), crate::Device::Metal(_)))
        {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Metal): implement
        // `crate::metal_concat(inputs, self.axis)` analogous to
        // `crate::cuda_concat` above. Until that kernel lands, fall
        // through to the CPU path so the op still produces correct
        // results on Mac (numerics-correct, performance-wrong;
        // requires a per-input CPU staging copy at the dispatch
        // boundary).
        // Candidate implementations:
        //   1. Custom MSL kernel: grid over (outer, axis_total,
        //      inner); each thread copies one `per`-byte element
        //      from `inputs[k][o, axis_off + a, i]` to
        //      `out[o, a, i]`. Resolves the per-input axis offset
        //      with a small uniform of length `inputs.len()`.
        //   2. Multi-dispatch fallback: one Metal blit-encoder copy
        //      per input slab (no kernel needed). Cheaper to wire
        //      but trades kernel launches for blits — fine for
        //      attention head concat where N is small.
        //   3. MPS Graph: `concat(_:dimension:)` for one-shot
        //      dispatch. Higher per-call overhead than a custom
        //      kernel.
        Ok(None)
    }

    /// Vulkan-only forward scaffold: symmetric with `cuda_fwd` but
    /// currently returns `Ok(None)` so dispatch falls through to the
    /// CPU path. Phase 4 Vulkan kernel author drops in
    /// `crate::vulkan_concat(inputs, self.axis)` once it ships.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_fwd(&self, inputs: &[&Tensor]) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - non-empty inputs
        //   - <= 32 inputs (mirrors CUDA substrate's MAX_INPUTS cap)
        //   - all inputs on Vulkan device
        if inputs.is_empty() {
            return Ok(None);
        }
        if inputs.len() > 32 {
            return Ok(None);
        }
        if !inputs
            .iter()
            .all(|t| matches!(t.device(), crate::Device::Vulkan(_)))
        {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_concat(inputs, self.axis)` analogous to
        // `crate::cuda_concat` above. Until that wrapper lands, fall
        // through to the CPU path (numerics-correct,
        // performance-wrong; requires a per-input CPU staging copy
        // at the dispatch boundary).
        // Candidate implementations:
        //   1. SPIR-V compute shader: grid over (outer, axis_total,
        //      inner); each invocation copies one `per`-byte element
        //      from `inputs[k][o, axis_off + a, i]` to
        //      `out[o, a, i]`. Push the per-input axis offsets via
        //      push-constants or a uniform buffer.
        //   2. Multi-dispatch fallback: one `vkCmdCopyBuffer`
        //      region per (input, outer-slab). No shader needed,
        //      but trades kernel launches for queue submissions —
        //      fine for attention head concat where N is small.
        //   3. Dtype matrix gap: concat is dtype-agnostic (byte
        //      copy), so the `VkDType` matrix doesn't constrain
        //      this op. Both shader and copy-region paths work on
        //      raw bytes.
        Ok(None)
    }
}

/// Concatenate `inputs` along `axis`. All inputs must share rank,
/// dtype, and every non-axis dim.
pub fn concat(inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
    if inputs.is_empty() {
        bail!("concat: at least one input required");
    }
    let rank = inputs[0].rank();
    if axis >= rank {
        bail!(
            "concat: axis {axis} out of range for rank-{rank} inputs (shape {:?})",
            inputs[0].shape()
        );
    }
    let dtype = inputs[0].dtype();
    if dtype.is_packed() {
        bail!("concat: packed dtype {dtype} is not supported");
    }
    let per = dtype.size_in_bytes();
    if per == 0 {
        bail!("concat: zero-size dtype {dtype}");
    }
    for (i, t) in inputs.iter().enumerate() {
        if t.rank() != rank {
            bail!(
                "concat: input {i} rank {} != input 0 rank {rank}",
                t.rank()
            );
        }
        if t.dtype() != dtype {
            bail!(
                "concat: input {i} dtype {} != input 0 dtype {dtype}",
                t.dtype()
            );
        }
        if !t.is_contiguous() {
            bail!("concat: input {i} must be contiguous");
        }
        for (d, (&a, &b)) in t.shape().iter().zip(inputs[0].shape()).enumerate() {
            if d != axis && a != b {
                bail!(
                    "concat: input {i} shape {:?} differs from input 0 shape {:?} along axis {d}",
                    t.shape(),
                    inputs[0].shape()
                );
            }
        }
    }

    // CUDA fast path: if all inputs are CUDA-backed (and within the
    // substrate's MAX_INPUTS=32), route through `cuda_concat` to
    // avoid a CPU staging copy. Mirrors the per-Op `cuda_fwd` pattern
    // for variable-arity ops like this one.
    #[cfg(feature = "cuda")]
    {
        let all_cuda = inputs.len() <= 32
            && inputs
                .iter()
                .all(|t| matches!(t.device(), crate::Device::Cuda(_)));
        if all_cuda {
            return crate::cuda_concat(inputs, axis);
        }
    }

    // Output shape: input 0's shape with axis dim = sum.
    let mut out_shape = inputs[0].shape().to_vec();
    let axis_total: usize = inputs.iter().map(|t| t.shape()[axis]).sum();
    out_shape[axis] = axis_total;

    let outer: usize = out_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);

    let mut out_bytes = vec![0u8; outer * axis_total * inner * per];

    // For each outer slab, copy each input's axis-slab into the
    // running output offset.
    for o in 0..outer {
        let mut axis_offset = 0usize;
        for t in inputs {
            let t_axis = t.shape()[axis];
            let t_cpu = t
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::from_str("concat: storage must be CpuStorage"))?;
            let t_bytes = t_cpu.as_bytes();
            // For this outer index o, copy the contiguous block
            // `t_bytes[o*t_axis*inner*per .. (o+1)*t_axis*inner*per]`
            // into `out_bytes[(o*axis_total + axis_offset)*inner*per .. ...]`.
            let src_start = o * t_axis * inner * per;
            let src_end = src_start + t_axis * inner * per;
            let dst_start = (o * axis_total + axis_offset) * inner * per;
            let dst_end = dst_start + t_axis * inner * per;
            out_bytes[dst_start..dst_end].copy_from_slice(&t_bytes[src_start..src_end]);
            axis_offset += t_axis;
        }
    }

    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
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
    fn concat_two_rank1_tensors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0], vec![2]).unwrap();
        let out = concat(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape(), &[5]);
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_two_rank2_axis_0() {
        // [[1,2],[3,4]] + [[5,6]] along axis 0 = [[1,2],[3,4],[5,6]]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0], vec![1, 2]).unwrap();
        let out = concat(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_two_rank2_axis_1() {
        // [[1,2],[3,4]] + [[5],[6]] along axis 1 = [[1,2,5],[3,4,6]]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0], vec![2, 1]).unwrap();
        let out = concat(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn concat_three_tensors_rank3_axis_middle() {
        // [B=2, H=1, D=1] + [B=2, H=2, D=1] + [B=2, H=1, D=1] along axis 1
        // = [B=2, H=4, D=1]
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2, 1, 1]).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 4.0, 5.0, 6.0], vec![2, 2, 1]).unwrap();
        let c = Tensor::from_slice(&[7.0f32, 8.0], vec![2, 1, 1]).unwrap();
        let out = concat(&[&a, &b, &c], 1).unwrap();
        assert_eq!(out.shape(), &[2, 4, 1]);
        // Batch 0: [a, b[0], b[1], c[0]] = [1, 3, 4, 7]
        // Batch 1: [a, b[2], b[3], c[1]] = [2, 5, 6, 8]
        assert_eq!(read_f32(&out), vec![1.0, 3.0, 4.0, 7.0, 2.0, 5.0, 6.0, 8.0]);
    }

    #[test]
    fn concat_bf16_round_trips() {
        let av: Vec<half::bf16> = [1.0f32, 2.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let bv: Vec<half::bf16> = [3.0f32, 4.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let a = Tensor::from_slice(&av, vec![2]).unwrap();
        let b = Tensor::from_slice(&bv, vec![2]).unwrap();
        let out = concat(&[&a, &b], 0).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn concat_empty_inputs_errors() {
        let e = concat(&[], 0).unwrap_err();
        assert!(e.to_string().contains("at least one input"));
    }

    #[test]
    fn concat_axis_out_of_range_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = concat(&[&a], 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn concat_rank_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 4.0], vec![1, 2]).unwrap();
        let e = concat(&[&a, &b], 0).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn concat_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bv = vec![half::bf16::from_f32(3.0)];
        let b = Tensor::from_slice(&bv, vec![1]).unwrap();
        let e = concat(&[&a, &b], 0).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn concat_shape_mismatch_on_other_axis_errors() {
        // axis 0 concat; axis-1 sizes must match
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0], vec![1, 2]).unwrap();
        let e = concat(&[&a, &b], 0).unwrap_err();
        assert!(e.to_string().contains("differs"));
    }

    #[test]
    fn op_metadata() {
        let op = ConcatOp::new(1);
        assert_eq!(op.name(), "concat");
        assert_eq!(op.axis(), 1);
    }
}

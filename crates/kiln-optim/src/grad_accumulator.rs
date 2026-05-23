//! `GradAccumulator` — accumulate gradients across micro-batches
//! before stepping.
//!
//! Per the Phase 6.5 issue bullet:
//!
//! > **Gradient accumulation across micro-batches.** Trainer
//! > accumulates over N micro-batches before stepping
//! > (`accumulate_grad` / `accumulate_grads_except` in
//! > `vk_train.rs:808,823`; `opd.rs:2767`). The grad buffer lives in
//! > `Parameter` and accumulates atomically across micro-batches;
//! > `optimizer.step()` consumes-and-zeros.
//!
//! This CPU reference impl owns one `Tensor` per `TensorId` and
//! exposes `accumulate(id, grad)` plus `take_and_clear(id)` for the
//! optimizer step. The "atomic" qualifier in the issue refers to the
//! GPU backend's atomicAdd semantics — on CPU we are single-writer
//! per id, so a plain in-place add is the canonical reference.
//!
//! The contract:
//! - First `accumulate(id, grad)` for a given id stores `grad`.
//! - Subsequent calls element-wise add into the stored tensor (must
//!   match shape + dtype).
//! - `take_and_clear(id)` returns the accumulated tensor and removes
//!   the entry (the next accumulate starts fresh).
//! - `len()` / `is_empty()` report how many ids are currently
//!   accumulating.

use std::collections::HashMap;
use std::sync::Arc;

use kiln_tensor::{
    CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId,
};
use kiln_tensor::bail;

#[derive(Debug, Default)]
pub struct GradAccumulator {
    inner: HashMap<TensorId, Tensor>,
}

impl GradAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn contains(&self, id: TensorId) -> bool {
        self.inner.contains_key(&id)
    }

    /// Add `grad` into the slot for `id`. First call stores; later
    /// calls element-wise accumulate.
    pub fn accumulate(&mut self, id: TensorId, grad: &Tensor) -> Result<()> {
        if let Some(existing) = self.inner.get_mut(&id) {
            if existing.shape() != grad.shape() {
                bail!(
                    "GradAccumulator: shape mismatch for id {:?} — stored {:?} vs new {:?}",
                    id,
                    existing.shape(),
                    grad.shape()
                );
            }
            if existing.dtype() != grad.dtype() {
                bail!(
                    "GradAccumulator: dtype mismatch for id {:?} — stored {} vs new {}",
                    id,
                    existing.dtype(),
                    grad.dtype()
                );
            }
            *existing = add_inplace_clone(existing, grad)?;
            Ok(())
        } else {
            // First write: keep a clean copy keyed by `id` so the
            // caller's tensor can drop while we hold our accumulation
            // state.
            self.inner.insert(id, clone_tensor(grad)?);
            Ok(())
        }
    }

    /// Pop the accumulated tensor for `id` and clear the slot.
    /// Returns `None` if nothing was accumulated.
    pub fn take_and_clear(&mut self, id: TensorId) -> Option<Tensor> {
        self.inner.remove(&id)
    }

    /// Clear every slot. The optimizer step calls this implicitly via
    /// `take_and_clear`; this is the bulk-reset helper for
    /// checkpoint-rollback paths.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

fn clone_tensor(t: &Tensor) -> Result<Tensor> {
    if !t.is_contiguous() {
        bail!("GradAccumulator: input must be contiguous");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| kiln_tensor::Error::from_str(
            "GradAccumulator: storage must be CpuStorage",
        ))?;
    let bytes = cpu.as_bytes().to_vec();
    let dtype = t.dtype();
    let cpu_out = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(t.shape().to_vec()), TensorId::next())
}

fn add_inplace_clone(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Allocate a fresh tensor whose storage is a + b. We don't yet
    // have a true in-place tensor mutation; this keeps the semantics
    // pure-functional while the version-counter contract (anti-pattern
    // 16) ships across the rest of the autograd tape.
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("GradAccumulator: both inputs must be contiguous");
    }
    let dtype = a.dtype();
    let a_cpu = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| kiln_tensor::Error::from_str("GradAccumulator: A must be CpuStorage"))?;
    let b_cpu = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| kiln_tensor::Error::from_str("GradAccumulator: B must be CpuStorage"))?;
    let ab = a_cpu.as_bytes();
    let bb = b_cpu.as_bytes();
    let n = a.element_count();
    let mut out_bytes = vec![0u8; n * dtype.size_in_bytes()];
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let va = f32::from_le_bytes(ab[i * 4..i * 4 + 4].try_into().unwrap());
                let vb = f32::from_le_bytes(bb[i * 4..i * 4 + 4].try_into().unwrap());
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&(va + vb).to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let va =
                    half::bf16::from_le_bytes(ab[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                let vb =
                    half::bf16::from_le_bytes(bb[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                let sum = half::bf16::from_f32(va + vb);
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&sum.to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let va = half::f16::from_le_bytes(ab[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let vb = half::f16::from_le_bytes(bb[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let sum = half::f16::from_f32(va + vb);
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&sum.to_le_bytes());
            }
        }
        other => bail!("GradAccumulator: unsupported dtype {other}"),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(a.shape().to_vec()), TensorId::next())
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

    #[test]
    fn accumulator_first_write() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        acc.accumulate(id, &g).unwrap();
        assert_eq!(acc.len(), 1);
        assert!(acc.contains(id));
    }

    #[test]
    fn accumulator_adds_subsequent_writes() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        let g1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let g2 = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        acc.accumulate(id, &g1).unwrap();
        acc.accumulate(id, &g2).unwrap();
        let popped = acc.take_and_clear(id).unwrap();
        assert_eq!(read_f32(&popped), vec![11.0, 22.0, 33.0]);
        assert!(acc.is_empty());
    }

    #[test]
    fn accumulator_take_and_clear_returns_none_when_empty() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        assert!(acc.take_and_clear(id).is_none());
    }

    #[test]
    fn accumulator_take_removes_id() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        let g = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        acc.accumulate(id, &g).unwrap();
        let _ = acc.take_and_clear(id);
        assert!(!acc.contains(id));
    }

    #[test]
    fn accumulator_multiple_ids() {
        let mut acc = GradAccumulator::new();
        let id1 = TensorId::next();
        let id2 = TensorId::next();
        let g1 = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g2 = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        acc.accumulate(id1, &g1).unwrap();
        acc.accumulate(id2, &g2).unwrap();
        assert_eq!(acc.len(), 2);
        assert_eq!(read_f32(&acc.take_and_clear(id1).unwrap()), vec![1.0]);
        assert_eq!(read_f32(&acc.take_and_clear(id2).unwrap()), vec![5.0]);
    }

    #[test]
    fn accumulator_shape_mismatch_errors() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        let g1 = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let g2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        acc.accumulate(id, &g1).unwrap();
        let e = acc.accumulate(id, &g2).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn accumulator_dtype_mismatch_errors() {
        let mut acc = GradAccumulator::new();
        let id = TensorId::next();
        let g1 = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let g2 = Tensor::from_slice(
            &[half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)],
            vec![2],
        )
        .unwrap();
        acc.accumulate(id, &g1).unwrap();
        let e = acc.accumulate(id, &g2).unwrap_err();
        assert!(e.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn accumulator_clear_drops_all() {
        let mut acc = GradAccumulator::new();
        for _ in 0..3 {
            let id = TensorId::next();
            let g = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
            acc.accumulate(id, &g).unwrap();
        }
        assert_eq!(acc.len(), 3);
        acc.clear();
        assert!(acc.is_empty());
    }
}

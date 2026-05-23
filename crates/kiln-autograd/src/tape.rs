//! `Tape` — the autograd recording surface.
//!
//! Records per-op tape nodes during forward; walks them in reverse
//! topo order during backward. Lifted from `vk_autograd::vk_backward`.
//!
//! # Threading model
//!
//! The Tape is **not thread-local** in this scaffold — callers pass it
//! explicitly through the forward path. Phase 6a.x adds a
//! thread-local tape handle for parity with PyTorch's `torch.autograd`
//! ergonomics if the explicit-pass turns out clumsy in practice.
//!
//! # Per-node version tracking (anti-pattern 16 hook)
//!
//! Each [`TapeNode`] records the **version** of every input at record
//! time. [`Tape::backward`] re-reads each input's current version and
//! asserts equality before calling the op's `bwd`. Today's
//! `kiln_tensor::Tensor` has no version field, so versions are
//! always `0` and the assertion is a no-op. When in-place ops land
//! (optimizer step, residual fuse), the version + this assertion
//! together enforce the invariant.

use std::collections::HashSet;

use kiln_tensor::{Error, Result, Tensor, TensorId};

use crate::{BoxedBackwardOp, GradStore};

/// One recorded forward op.
#[derive(Debug)]
pub struct TapeNode {
    /// `TensorId` of the op's output. Tape walker keys grads on this.
    pub output_id: TensorId,
    /// `TensorId`s of the op's inputs, in declaration order.
    pub input_ids: Vec<TensorId>,
    /// Input version counters at record time. See module doc.
    /// Length matches `input_ids`. Today always `0`.
    pub input_versions: Vec<u64>,
    /// Owning backward closure.
    pub op: BoxedBackwardOp,
}

/// Autograd tape.
///
/// One handle per forward pass. Cleared between training steps via
/// [`Tape::clear`].
#[derive(Debug, Default)]
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    /// Empty tape.
    pub fn new() -> Self {
        Tape { nodes: Vec::new() }
    }

    /// Number of recorded ops. Useful for sanity checks + debug.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True iff no ops have been recorded.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Drop all recorded nodes. Called between training steps.
    ///
    /// Per anti-pattern 16: `Tape::clear()` MUST run before any
    /// in-place mutation of recorded tensors. Failing to clear lets a
    /// stale node reference a post-mutation tensor and the backward
    /// path produces silent corruption.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Record a forward op.
    ///
    /// Captures `output_id`, `input_ids`, and the current `input_versions`
    /// (always `0` until Tensor gets a version field in Phase 1.x).
    pub fn record(&mut self, output: &Tensor, inputs: &[&Tensor], op: BoxedBackwardOp) {
        let input_ids = inputs.iter().map(|t| t.id()).collect::<Vec<_>>();
        let input_versions = inputs.iter().map(|_| 0u64).collect::<Vec<_>>();
        self.nodes.push(TapeNode {
            output_id: output.id(),
            input_ids,
            input_versions,
            op,
        });
    }

    /// Borrow the recorded nodes (for debugging / introspection).
    pub fn nodes(&self) -> &[TapeNode] {
        &self.nodes
    }

    /// Run the backward pass.
    ///
    /// `seed_grad` is `dLoss/dLoss` (typically ones-of-shape-`loss` for
    /// scalar losses). The walker:
    ///
    /// 1. Initializes a per-output-id gradient map keyed on `output_id`,
    ///    seeded with `(loss_id, seed_grad)`.
    /// 2. Walks `nodes` in reverse insertion order.
    /// 3. For each node, reads `grad_output` from the map, asserts that
    ///    each input's current version matches the recorded version
    ///    (anti-pattern 16), calls `op.apply(grad_output)`, and
    ///    accumulates per-input gradients back into the map.
    /// 4. Returns the final map as a [`GradStore`].
    ///
    /// `accumulator` is a function `(grad_a, grad_b) -> grad_sum` that
    /// adds two same-shape Tensors. The caller supplies it so this
    /// crate doesn't depend on `kiln_tensor::ops` directly (avoids a
    /// crate-cycle when `kiln-tensor::ops` themselves want to use the
    /// autograd tape). Typically: `|a, b| kiln_tensor::ops::add(a, b)`.
    pub fn backward<F>(
        &self,
        loss_id: TensorId,
        seed_grad: Tensor,
        mut accumulator: F,
    ) -> Result<GradStore>
    where
        F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    {
        // Per-output-id accumulated gradient map.
        let mut grads: std::collections::HashMap<TensorId, Tensor> =
            std::collections::HashMap::new();
        grads.insert(loss_id, seed_grad);

        // Walk nodes in reverse insertion order. (Insertion order is
        // already topo-sorted producer-before-consumer because each
        // forward op records before its consumers.)
        for node in self.nodes.iter().rev() {
            // 1. Anti-pattern 16: input version check. Today's
            //    `current_version()` is a stub returning 0; once
            //    Tensor has a real version counter, this asserts that
            //    no in-place mutation happened since `record()`.
            for (i, recorded_version) in node.input_versions.iter().enumerate() {
                let current = current_version_for_id(node.input_ids[i]);
                if current != *recorded_version {
                    return Err(Error::Msg(format!(
                        "kiln_autograd: tape node {} input {} version drifted \
                         (recorded {}, now {}). Anti-pattern 16: in-place \
                         mutation invalidated the tape.",
                        node.op.name(),
                        i,
                        recorded_version,
                        current
                    )));
                }
            }

            // 2. Pull the grad for this node's output. If `None`,
            //    the output has no upstream gradient — skip.
            let Some(grad_output) = grads.remove(&node.output_id) else {
                continue;
            };

            // 3. Compute per-input grads.
            let per_input = node.op.apply(&grad_output)?;
            if per_input.len() != node.input_count_decl() {
                return Err(Error::Msg(format!(
                    "kiln_autograd: tape node {} returned {} grads for {} inputs",
                    node.op.name(),
                    per_input.len(),
                    node.input_count_decl()
                )));
            }

            // 4. Accumulate into the per-id grad map.
            for (i, maybe_grad) in per_input.into_iter().enumerate() {
                let Some(g) = maybe_grad else { continue };
                let input_id = node.input_ids[i];
                match grads.remove(&input_id) {
                    Some(existing) => {
                        let summed = accumulator(&existing, &g)?;
                        grads.insert(input_id, summed);
                    }
                    None => {
                        grads.insert(input_id, g);
                    }
                }
            }
        }

        // Collect the remaining (leaf) grads into the public store.
        let mut store = GradStore::new();
        for (id, g) in grads.drain() {
            store.insert(id, g);
        }
        Ok(store)
    }

    /// Walk the tape and return the set of reachable `TensorId`s from
    /// `root`. Used by `Tape::backward` (implicitly via the topo walk)
    /// and exposed publicly for selective-recompute analysis in Phase 6.5.
    pub fn reachable_from(&self, root: TensorId) -> HashSet<TensorId> {
        let mut reached: HashSet<TensorId> = HashSet::new();
        reached.insert(root);
        // Walk in reverse so producers are visited before consumers.
        // (Insertion order has consumers after producers; reversal
        // walks consumers first.)
        for node in self.nodes.iter().rev() {
            if reached.contains(&node.output_id) {
                for &id in &node.input_ids {
                    reached.insert(id);
                }
            }
        }
        reached
    }
}

impl TapeNode {
    fn input_count_decl(&self) -> usize {
        self.op.input_count()
    }
}

/// **Stub**: returns the current version counter for a `TensorId`.
///
/// Today there is no version field on `Tensor`, so this always returns 0
/// and the anti-pattern 16 assertion in `Tape::backward` is a no-op.
///
/// Phase 1.x adds the version field; the stub gets replaced with a
/// real lookup. The signature stays the same so this PR's tape walker
/// is forward-compatible.
fn current_version_for_id(_id: TensorId) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackwardOp;
    use kiln_tensor::{DType, Tensor};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test-only `BackwardOp` that records how many times its `apply`
    /// was called and returns a fixed gradient per input.
    #[derive(Debug)]
    struct CountingOp {
        name: &'static str,
        input_count: usize,
        calls: std::sync::Arc<AtomicUsize>,
    }

    impl BackwardOp for CountingOp {
        fn name(&self) -> &'static str {
            self.name
        }
        fn input_count(&self) -> usize {
            self.input_count
        }
        fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // Echo grad_output as each input's gradient.
            Ok((0..self.input_count).map(|_| Some(grad_output.clone())).collect())
        }
    }

    fn cpu_tensor() -> Tensor {
        Tensor::zeros_cpu(vec![1], DType::F32)
    }

    fn passthrough_accumulator(_a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Tests don't need real arithmetic — just return one of the inputs.
        Ok(b.clone())
    }

    #[test]
    fn empty_tape_backward_no_ops() {
        let tape = Tape::new();
        let loss = cpu_tensor();
        let store = tape
            .backward(loss.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap();
        // Just the seed grad ends up in the store.
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn record_increments_len() {
        let mut tape = Tape::new();
        assert!(tape.is_empty());
        let out = cpu_tensor();
        let inp = cpu_tensor();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let op = CountingOp {
            name: "test/op1",
            input_count: 1,
            calls: calls.clone(),
        };
        tape.record(&out, &[&inp], Box::new(op));
        assert_eq!(tape.len(), 1);
        assert!(!tape.is_empty());
    }

    #[test]
    fn backward_calls_apply_on_recorded_op() {
        let mut tape = Tape::new();
        let out = cpu_tensor();
        let inp = cpu_tensor();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        tape.record(
            &out,
            &[&inp],
            Box::new(CountingOp {
                name: "test/identity",
                input_count: 1,
                calls: calls.clone(),
            }),
        );
        let _ = tape
            .backward(out.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn backward_walks_in_reverse_topo_order() {
        // Build: a → op_first → b → op_second → c
        // Forward records op_first before op_second.
        // Backward must call op_second first, then op_first.
        let mut tape = Tape::new();
        let a = cpu_tensor();
        let b = cpu_tensor();
        let c = cpu_tensor();
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));

        #[derive(Debug)]
        struct RecordingOp {
            name: &'static str,
            input_count: usize,
            order: std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl BackwardOp for RecordingOp {
            fn name(&self) -> &'static str {
                self.name
            }
            fn input_count(&self) -> usize {
                self.input_count
            }
            fn apply(&self, g: &Tensor) -> Result<Vec<Option<Tensor>>> {
                self.order.lock().unwrap().push(self.name);
                Ok((0..self.input_count).map(|_| Some(g.clone())).collect())
            }
        }

        tape.record(
            &b,
            &[&a],
            Box::new(RecordingOp {
                name: "op_first",
                input_count: 1,
                order: order.clone(),
            }),
        );
        tape.record(
            &c,
            &[&b],
            Box::new(RecordingOp {
                name: "op_second",
                input_count: 1,
                order: order.clone(),
            }),
        );
        let _ = tape
            .backward(c.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap();
        let recorded = order.lock().unwrap().clone();
        assert_eq!(recorded, vec!["op_second", "op_first"]);
    }

    #[test]
    fn clear_resets_length() {
        let mut tape = Tape::new();
        tape.record(
            &cpu_tensor(),
            &[&cpu_tensor()],
            Box::new(CountingOp {
                name: "test",
                input_count: 1,
                calls: std::sync::Arc::new(AtomicUsize::new(0)),
            }),
        );
        assert_eq!(tape.len(), 1);
        tape.clear();
        assert_eq!(tape.len(), 0);
    }

    #[test]
    fn reachable_from_walks_inputs() {
        let mut tape = Tape::new();
        let a = cpu_tensor();
        let b = cpu_tensor();
        let c = cpu_tensor();
        tape.record(
            &b,
            &[&a],
            Box::new(CountingOp {
                name: "a_to_b",
                input_count: 1,
                calls: std::sync::Arc::new(AtomicUsize::new(0)),
            }),
        );
        tape.record(
            &c,
            &[&b],
            Box::new(CountingOp {
                name: "b_to_c",
                input_count: 1,
                calls: std::sync::Arc::new(AtomicUsize::new(0)),
            }),
        );
        let r = tape.reachable_from(c.id());
        assert!(r.contains(&a.id()));
        assert!(r.contains(&b.id()));
        assert!(r.contains(&c.id()));
    }

    #[test]
    fn backward_arity_mismatch_errors() {
        let mut tape = Tape::new();
        let out = cpu_tensor();
        let inp1 = cpu_tensor();
        let inp2 = cpu_tensor();
        // Declare arity=2 but record only 1 input? — actually record
        // 2 inputs but `apply` returns 1 grad: we'll force the mismatch
        // by declaring `input_count=2` but emitting 1-grad in apply.
        #[derive(Debug)]
        struct BuggyOp;
        impl BackwardOp for BuggyOp {
            fn name(&self) -> &'static str {
                "buggy"
            }
            fn input_count(&self) -> usize {
                2
            }
            fn apply(&self, g: &Tensor) -> Result<Vec<Option<Tensor>>> {
                Ok(vec![Some(g.clone())])
            }
        }
        tape.record(&out, &[&inp1, &inp2], Box::new(BuggyOp));
        let e = tape
            .backward(out.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap_err();
        assert!(e.to_string().contains("returned"));
    }
}

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
//! # Per-node version tracking (anti-pattern 16 hook) — wired in Phase 1.32
//!
//! Each [`TapeNode`] records the version of every input at record time
//! AND keeps an `Arc<AtomicU64>` handle to the live version. On backward,
//! the live value is loaded and compared against the snapshot; if they
//! differ, in-place mutation happened between forward and backward and
//! the tape is stale (anti-pattern 16 violation).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Length matches `input_ids`.
    pub input_versions: Vec<u64>,
    /// Live handles to each input's version counter. The tape walker
    /// loads from these on backward and compares against
    /// `input_versions` to enforce anti-pattern 16.
    pub input_version_handles: Vec<Arc<AtomicU64>>,
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
    /// Captures `output_id`, `input_ids`, the current `input_versions`,
    /// and `Arc<AtomicU64>` handles to each input's live version.
    pub fn record(&mut self, output: &Tensor, inputs: &[&Tensor], op: BoxedBackwardOp) {
        let input_ids: Vec<TensorId> = inputs.iter().map(|t| t.id()).collect();
        let input_versions: Vec<u64> = inputs.iter().map(|t| t.current_version()).collect();
        let input_version_handles: Vec<Arc<AtomicU64>> =
            inputs.iter().map(|t| t.version_handle()).collect();
        self.nodes.push(TapeNode {
            output_id: output.id(),
            input_ids,
            input_versions,
            input_version_handles,
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
        accumulator: F,
    ) -> Result<GradStore>
    where
        F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    {
        // Delegate to the multi-seed walker with a singleton seed map.
        // Keeps the public single-loss API stable while letting the
        // bridge feed multiple downstream seed grads in one walk (Phase
        // 6a/CP-4 #1082 — see `Tape::backward_with_seeds`).
        let mut seeds: std::collections::HashMap<TensorId, Tensor> =
            std::collections::HashMap::new();
        seeds.insert(loss_id, seed_grad);
        self.backward_with_seeds(seeds, accumulator)
    }

    /// Run the backward pass with a *map* of seed gradients.
    ///
    /// Same walker semantics as [`Tape::backward`] but the caller
    /// supplies `(output_id → seed_grad)` for **any** tape-recorded
    /// node's output (or for any `TensorId` that appears as an input to
    /// a tape node, in which case it short-circuits into the per-input
    /// accumulation directly). This is the entry point used by the
    /// `kiln-kt-bridge::tape_emit` bridge (Phase 6a/CP-4 #1082):
    /// `loss.backward()` produces candle grads for the candle Tensors
    /// returned by each tape adapter; the bridge feeds them in as the
    /// per-tape-output seeds, lets `Tape::backward` walk the tape, and
    /// merges the resulting per-input kt grads back into candle's
    /// `GradStore` under the matching candle `TensorId`s.
    ///
    /// `seeds` may contain entries for `TensorId`s that the tape
    /// never recorded as an output — they're carried through to the
    /// returned `GradStore` (so the caller can ask "what's the grad of
    /// `x`?" without first proving x was a tape output).
    pub fn backward_with_seeds<F>(
        &self,
        seeds: std::collections::HashMap<TensorId, Tensor>,
        mut accumulator: F,
    ) -> Result<GradStore>
    where
        F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    {
        // Per-output-id accumulated gradient map.
        let mut grads: std::collections::HashMap<TensorId, Tensor> = seeds;

        // Read KILN_DETECT_ANOMALY once up-front so the per-node loop
        // doesn't pay the env-var lookup cost per iteration. When
        // set, each backward op's gradient outputs are scanned for
        // NaN/Inf via `Tensor::all_finite()`; the first violation
        // panics with `anomaly_panic` so corruption surfaces at the
        // op that produced the non-finite value, not 100 steps
        // later when loss diverges. Off-by-default (~5% per-step
        // cost when on); CI training-parity tests opt in.
        let anomaly = crate::anomaly::anomaly_detection_enabled();

        // #1082 CP-4 diagnostic: tape size + per-node processing trace.
        let cp4_dbg = std::env::var("KILN_CP4_DEBUG").is_ok();
        let trace_node_timings = std::env::var("KILN_TRACE_TAPE_BACKWARD_TIMINGS")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no" || v == "off")
            })
            .unwrap_or(false);
        if cp4_dbg {
            eprintln!(
                "[CP4-DEBUG] backward_with_seeds: {} nodes on tape, {} seeds",
                self.nodes.len(),
                grads.len()
            );
            for (i, node) in self.nodes.iter().enumerate() {
                eprintln!(
                    "[CP4-DEBUG]   node[{i}] op={} out={} ins={:?}",
                    node.op.name(),
                    node.output_id.as_raw(),
                    node.input_ids
                        .iter()
                        .map(|id| id.as_raw())
                        .collect::<Vec<_>>(),
                );
            }
        }

        // Walk nodes in reverse insertion order. (Insertion order is
        // already topo-sorted producer-before-consumer because each
        // forward op records before its consumers.)
        for (node_index, node) in self.nodes.iter().enumerate().rev() {
            if cp4_dbg && !grads.contains_key(&node.output_id) {
                eprintln!(
                    "[CP4-DEBUG]   SKIP node[{node_index}] op={} out={} (no upstream grad)",
                    node.op.name(),
                    node.output_id.as_raw(),
                );
            }
            // 1. Anti-pattern 16: input version check via the live
            //    Arc<AtomicU64> handles captured at record() time.
            for (i, recorded_version) in node.input_versions.iter().enumerate() {
                let current = node.input_version_handles[i].load(Ordering::Relaxed);
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
            let node_started = if trace_node_timings {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let grad_shape = if trace_node_timings {
                Some(grad_output.shape().to_vec())
            } else {
                None
            };
            let grad_dtype = if trace_node_timings {
                Some(grad_output.dtype())
            } else {
                None
            };
            let per_input = node.op.apply(&grad_output)?;
            if let Some(started) = node_started {
                eprintln!(
                    "kiln_tape_backward_timing node_index={} op={} grad_shape={:?} \
                     grad_dtype={:?} elapsed_ms={:.3}",
                    node_index,
                    node.op.name(),
                    grad_shape.unwrap_or_default(),
                    grad_dtype,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if per_input.len() != node.input_count_decl() {
                return Err(Error::Msg(format!(
                    "kiln_autograd: tape node {} returned {} grads for {} inputs",
                    node.op.name(),
                    per_input.len(),
                    node.input_count_decl()
                )));
            }

            // 3b. KILN_DETECT_ANOMALY: scan each gradient output for
            //     NaN/Inf and panic at the producing op's tape
            //     position on the first violation. CPU tensors use
            //     the strided walker; CUDA tensors bridge through a
            //     D2H copy until the per-backend is_finite reduction
            //     kernels land. Unsupported devices are skipped so the
            //     debug trap stays opt-in instead of breaking backward.
            if anomaly {
                for (i, maybe_grad) in per_input.iter().enumerate() {
                    let Some(g) = maybe_grad.as_ref() else {
                        continue;
                    };
                    match g.all_finite() {
                        Ok(true) => {}
                        Ok(false) => {
                            crate::anomaly::anomaly_panic(
                                node_index,
                                node.op.name(),
                                &format!(
                                    "input #{i} gradient contained NaN or Inf \
                                     (shape {:?}, dtype {:?})",
                                    g.shape(),
                                    g.dtype()
                                ),
                            );
                        }
                        Err(_) => {
                            // Unsupported storage/backend — the dedicated
                            // is_finite kernels are follow-up work.
                        }
                    }
                }
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

// The Phase 1.12 stub `current_version_for_id` was replaced in Phase
// 1.32: `Tape::backward` now loads the live version directly from each
// node's `input_version_handles` Arc<AtomicU64>. No process-wide
// registry needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackwardOp;
    use kiln_tensor::{DType, Tensor};
    use std::ffi::OsString;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: Rust 2024 marks env mutation unsafe because it is
            // process-global. Tests that call this helper hold ENV_LOCK.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: see `EnvVarGuard::set`; the guard is dropped
            // before ENV_LOCK is released.
            unsafe {
                match self.previous.as_ref() {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

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
            Ok((0..self.input_count)
                .map(|_| Some(grad_output.clone()))
                .collect())
        }
    }

    #[derive(Debug)]
    struct NonFiniteGradOp {
        name: &'static str,
    }

    impl BackwardOp for NonFiniteGradOp {
        fn name(&self) -> &'static str {
            self.name
        }

        fn input_count(&self) -> usize {
            1
        }

        fn apply(&self, _grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
            Ok(vec![Some(Tensor::from_slice(&[f32::NAN], vec![1])?)])
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
    fn backward_detects_anti_pattern_16_version_drift() {
        // Wire the Phase 1.32 version counter end-to-end:
        // record an op, bump the input's version (simulating in-place
        // mutation between forward and backward), then call backward.
        // Expected: error message mentions anti-pattern 16 + the op name.
        let mut tape = Tape::new();
        let out = cpu_tensor();
        let inp = cpu_tensor();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        tape.record(
            &out,
            &[&inp],
            Box::new(CountingOp {
                name: "test/some_op",
                input_count: 1,
                calls,
            }),
        );
        // Simulate in-place mutation of `inp` between forward and backward.
        inp.bump_version();
        let e = tape
            .backward(out.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.contains("Anti-pattern 16"),
            "expected anti-pattern 16 error, got: {msg}"
        );
        assert!(msg.contains("test/some_op"));
    }

    #[test]
    fn backward_ok_when_versions_unchanged() {
        // No in-place mutation between forward and backward -> backward succeeds.
        let mut tape = Tape::new();
        let out = cpu_tensor();
        let inp = cpu_tensor();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        tape.record(
            &out,
            &[&inp],
            Box::new(CountingOp {
                name: "test/clean_op",
                input_count: 1,
                calls: calls.clone(),
            }),
        );
        // No bump — versions stay at 0.
        let _ = tape
            .backward(out.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn backward_detects_non_finite_grad_when_anomaly_enabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(crate::ENV_DETECT_ANOMALY, Some("1"));

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut tape = Tape::new();
            let out = cpu_tensor();
            let inp = cpu_tensor();
            tape.record(
                &out,
                &[&inp],
                Box::new(NonFiniteGradOp {
                    name: "test/non_finite_grad",
                }),
            );
            let _ = tape
                .backward(out.id(), cpu_tensor(), passthrough_accumulator)
                .unwrap();
        }));

        let panic = result.expect_err("expected anomaly detector to panic");
        let msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            msg.contains("kiln_autograd: anomaly detected at tape position 0"),
            "expected anomaly prefix and tape position, got: {msg}"
        );
        assert!(
            msg.contains("op `test/non_finite_grad`"),
            "expected op name, got: {msg}"
        );
        assert!(
            msg.contains("input #0 gradient contained NaN or Inf"),
            "expected gradient detail, got: {msg}"
        );
    }

    #[test]
    fn backward_allows_non_finite_grad_when_anomaly_disabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(crate::ENV_DETECT_ANOMALY, None);

        let mut tape = Tape::new();
        let out = cpu_tensor();
        let inp = cpu_tensor();
        tape.record(
            &out,
            &[&inp],
            Box::new(NonFiniteGradOp {
                name: "test/non_finite_grad",
            }),
        );

        let store = tape
            .backward(out.id(), cpu_tensor(), passthrough_accumulator)
            .unwrap();
        let grad = store.get(inp.id()).expect("input grad");
        assert!(!grad.all_finite().unwrap());
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

    #[test]
    fn backward_with_seeds_supports_multi_output_seeding() {
        // CP-4 #1082 bridge precondition: when the kt-tape graph has
        // multiple sub-roots (e.g. two production-caller adapters each
        // record a node whose candle output flows independently into
        // the loss), the bridge feeds *each* output's seed grad as a
        // separate map entry. Confirm the walker honours both seeds
        // and accumulates correctly per input.
        //
        // Graph:
        //   a → op1 → out1   (seeded externally)
        //   b → op2 → out2   (seeded externally)
        //
        // Backward must call op1 with the seed for out1 and op2 with
        // the seed for out2; each input grad map must contain a, b.
        let mut tape = Tape::new();
        let a = cpu_tensor();
        let b = cpu_tensor();
        let out1 = cpu_tensor();
        let out2 = cpu_tensor();

        let calls1 = std::sync::Arc::new(AtomicUsize::new(0));
        let calls2 = std::sync::Arc::new(AtomicUsize::new(0));
        tape.record(
            &out1,
            &[&a],
            Box::new(CountingOp {
                name: "test/op1",
                input_count: 1,
                calls: calls1.clone(),
            }),
        );
        tape.record(
            &out2,
            &[&b],
            Box::new(CountingOp {
                name: "test/op2",
                input_count: 1,
                calls: calls2.clone(),
            }),
        );

        let mut seeds: std::collections::HashMap<TensorId, Tensor> =
            std::collections::HashMap::new();
        seeds.insert(out1.id(), cpu_tensor());
        seeds.insert(out2.id(), cpu_tensor());

        let store = tape
            .backward_with_seeds(seeds, passthrough_accumulator)
            .unwrap();

        assert_eq!(calls1.load(Ordering::SeqCst), 1, "op1 must run once");
        assert_eq!(calls2.load(Ordering::SeqCst), 1, "op2 must run once");
        assert!(store.get(a.id()).is_some(), "a must have a grad");
        assert!(store.get(b.id()).is_some(), "b must have a grad");
    }

    #[test]
    fn backward_with_seeds_carries_unmatched_seeds_through() {
        // Bridge contract: seeds for `TensorId`s that the tape never
        // recorded as outputs are preserved in the returned GradStore.
        // The bridge relies on this so it can ask "what's the grad of
        // a candle parameter that fed *into* a tape op but had no
        // tape op record itself" — i.e., the kt-side input ID grad
        // accumulator behaviour at the leaf nodes.
        let tape = Tape::new();
        let leaf = cpu_tensor();
        let mut seeds: std::collections::HashMap<TensorId, Tensor> =
            std::collections::HashMap::new();
        seeds.insert(leaf.id(), cpu_tensor());

        let store = tape
            .backward_with_seeds(seeds, passthrough_accumulator)
            .unwrap();
        assert!(
            store.get(leaf.id()).is_some(),
            "unmatched seed must survive an empty backward"
        );
    }
}

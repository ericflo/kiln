//! `StreamPlanner` — per-op stream affinity + cross-stream dependency
//! tracking.
//!
//! Per the Phase 1 issue bullet:
//!
//! > **`kiln-tensor::StreamPlanner`** — per-op stream affinity,
//! > automatic event/semaphore insertion, stream topology stable
//! > across `capture()` / `replay()`. Without this, "QKV on three
//! > streams" in Phase 3 won't generalize and graph capture won't
//! > compose with multi-stream forward.
//!
//! # Phase 1.26 scope (this PR)
//!
//! Backend-agnostic types only. Per-backend wires in subsequent PRs:
//!
//! - CUDA: per-op `cudaStream_t` selection + `cudaStreamWaitEvent`
//!   for cross-stream dependencies. The `StreamId` maps to a
//!   `cudarc::driver::CudaStream` index in `kiln-tensor::CudaStorage`.
//! - Metal: per-op `MTLCommandQueue` selection + `MTLEvent` waits.
//! - Vulkan: per-op `vk::Queue` selection (compute vs transfer) +
//!   `vk::Semaphore` waits. Lifts from `kiln-vulkan-kernel::cmd_batch.rs`.
//!
//! # Anti-pattern: stream-handle aliasing
//!
//! Per the issue's Phase 1 bullet:
//!
//! > CUDA inherits candle's cuBLAS handle + default stream. ...
//! > **One cuBLAS(Lt) handle per stream**, not per device —
//! > `cublasSetStream` is stateful and serializes; per-stream
//! > handles are the only way the StreamPlanner gets real parallelism.
//!
//! The `StreamId` carries enough information for the per-backend
//! impl to select its per-stream handle — the planner itself stays
//! backend-agnostic.
//!
//! # Phase 3 hook
//!
//! The Phase 3 attention port's "QKV on three independent streams" is
//! the canonical user. Each of `Q proj`, `K proj`, `V proj` reserves a
//! distinct [`StreamId`]; the planner records that they have no
//! data dependency on each other; the per-backend dispatch issues
//! them in parallel.

use std::collections::HashMap;

use crate::TensorId;

/// Opaque stream / queue identifier. Maps 1:1 to a per-backend
/// stream handle (CUDA `cudaStream_t` / Metal `MTLCommandQueue` /
/// Vulkan `vk::Queue`).
///
/// `StreamId(0)` is conventionally the "default" stream; planner
/// users requesting a fresh stream call [`StreamPlanner::reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(u32);

impl StreamId {
    /// The default stream id. Backend impls map this to their
    /// per-device default stream (cudarc's `default_stream()`,
    /// Metal's `MTLCommandQueue::new()`, Vulkan's queue family 0).
    pub const DEFAULT: StreamId = StreamId(0);

    pub const fn from_raw(raw: u32) -> Self {
        StreamId(raw)
    }
    pub const fn as_raw(self) -> u32 {
        self.0
    }
    pub const fn is_default(self) -> bool {
        self.0 == 0
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_default() {
            f.write_str("stream:default")
        } else {
            write!(f, "stream:{}", self.0)
        }
    }
}

/// Per-op scheduling record. Stored in the [`StreamPlanner`] and
/// consumed by the per-backend dispatcher.
#[derive(Debug, Clone)]
pub struct StreamRecord {
    /// Stream the op runs on.
    pub stream: StreamId,
    /// Tensors this op reads. Used to compute cross-stream wait
    /// dependencies.
    pub inputs: Vec<TensorId>,
    /// Tensor this op writes. Future ops that read this id must
    /// wait on this op's completion event before launching on a
    /// different stream.
    pub output: TensorId,
}

/// Per-op scheduler. Backend-agnostic; tracks per-stream op queues
/// and emits the wait-list for each op based on its inputs' producer
/// stream.
#[derive(Debug, Default)]
pub struct StreamPlanner {
    /// All ops, in record order.
    ops: Vec<StreamRecord>,
    /// For each output `TensorId`, the stream that produced it.
    producer_of: HashMap<TensorId, StreamId>,
    /// Next stream id to hand out from [`StreamPlanner::reserve`].
    next_stream: u32,
}

impl StreamPlanner {
    /// Empty planner. Default stream is `StreamId::DEFAULT`.
    pub fn new() -> Self {
        StreamPlanner {
            ops: Vec::new(),
            producer_of: HashMap::new(),
            next_stream: 1, // 0 is DEFAULT
        }
    }

    /// Reserve a fresh stream id. The per-backend impl creates a
    /// matching stream/queue when the planner is wired through (Phase
    /// 1.x cudarc CudaStream pool, Metal MTLCommandQueue pool,
    /// Vulkan vk::Queue pool).
    pub fn reserve(&mut self) -> StreamId {
        let id = StreamId(self.next_stream);
        self.next_stream += 1;
        id
    }

    /// Record that `op` runs on `stream`, reads `inputs`, writes
    /// `output`. The planner updates its producer-of map so future
    /// ops reading `output` can compute cross-stream waits.
    pub fn record(&mut self, stream: StreamId, inputs: Vec<TensorId>, output: TensorId) {
        self.producer_of.insert(output, stream);
        self.ops.push(StreamRecord {
            stream,
            inputs,
            output,
        });
    }

    /// Borrow the recorded ops.
    pub fn ops(&self) -> &[StreamRecord] {
        &self.ops
    }

    /// Number of recorded ops.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// True iff no ops recorded.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// For the op at index `i`, return the set of stream ids it must
    /// wait on before launching. Computed from the producers of each
    /// of its inputs.
    ///
    /// The output set:
    /// - Excludes the op's own stream (no self-wait).
    /// - Excludes inputs whose producer is unknown (model weights,
    ///   constants — no wait needed; they're already resident).
    pub fn waits_for(&self, op_idx: usize) -> Vec<StreamId> {
        let op = &self.ops[op_idx];
        let mut waits: Vec<StreamId> = op
            .inputs
            .iter()
            .filter_map(|id| self.producer_of.get(id))
            .copied()
            .filter(|s| *s != op.stream)
            .collect();
        waits.sort_unstable();
        waits.dedup();
        waits
    }

    /// Clear all recorded ops + producer map. Called between forward
    /// passes (the planner is per-forward, not per-process).
    pub fn clear(&mut self) {
        self.ops.clear();
        self.producer_of.clear();
        self.next_stream = 1;
    }

    /// Return the set of unique stream ids the planner has emitted
    /// records on. Used by the per-backend dispatcher to know how
    /// many distinct streams to allocate / synchronize.
    pub fn unique_streams(&self) -> Vec<StreamId> {
        let mut set: std::collections::HashSet<StreamId> = std::collections::HashSet::new();
        for op in &self.ops {
            set.insert(op.stream);
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stream_is_zero() {
        assert!(StreamId::DEFAULT.is_default());
        assert_eq!(StreamId::DEFAULT.as_raw(), 0);
    }

    #[test]
    fn reserve_hands_out_unique_ids() {
        let mut p = StreamPlanner::new();
        let a = p.reserve();
        let b = p.reserve();
        let c = p.reserve();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert!(!a.is_default());
    }

    #[test]
    fn record_and_len() {
        let mut p = StreamPlanner::new();
        assert!(p.is_empty());
        let s = p.reserve();
        p.record(s, vec![TensorId::from_raw(1)], TensorId::from_raw(2));
        assert_eq!(p.len(), 1);
        assert!(!p.is_empty());
    }

    #[test]
    fn waits_for_returns_producer_stream() {
        // Build: Q_proj on s_q (reads x), K_proj on s_k (reads x),
        // V_proj on s_v (reads x). Then attn_score on s_attn reads
        // (Q, K) -> must wait for s_q AND s_k.
        let mut p = StreamPlanner::new();
        let s_q = p.reserve();
        let s_k = p.reserve();
        let s_v = p.reserve();
        let s_attn = p.reserve();
        let x = TensorId::from_raw(1);
        let q = TensorId::from_raw(2);
        let k = TensorId::from_raw(3);
        let v = TensorId::from_raw(4);
        let scores = TensorId::from_raw(5);

        p.record(s_q, vec![x], q);
        p.record(s_k, vec![x], k);
        p.record(s_v, vec![x], v);
        p.record(s_attn, vec![q, k], scores);

        let waits = p.waits_for(3); // attn_score is index 3
        assert_eq!(waits, vec![s_q, s_k]); // sorted by raw id
    }

    #[test]
    fn waits_for_skips_own_stream() {
        // Two ops on the same stream don't need a cross-stream wait.
        let mut p = StreamPlanner::new();
        let s = p.reserve();
        let x = TensorId::from_raw(1);
        let y = TensorId::from_raw(2);
        let z = TensorId::from_raw(3);
        p.record(s, vec![x], y);
        p.record(s, vec![y], z);
        // Op 1 reads y (produced by op 0 on the same stream) — no wait.
        assert_eq!(p.waits_for(1), Vec::<StreamId>::new());
    }

    #[test]
    fn waits_for_skips_unknown_producer() {
        // Inputs without a known producer (weights, constants) need
        // no wait.
        let mut p = StreamPlanner::new();
        let s = p.reserve();
        let weight = TensorId::from_raw(99);
        let out = TensorId::from_raw(100);
        p.record(s, vec![weight], out);
        assert!(p.waits_for(0).is_empty());
    }

    #[test]
    fn unique_streams_returns_distinct_set() {
        let mut p = StreamPlanner::new();
        let s1 = p.reserve();
        let s2 = p.reserve();
        let s3 = p.reserve();
        let x = TensorId::from_raw(1);
        p.record(s1, vec![], TensorId::from_raw(10));
        p.record(s2, vec![x], TensorId::from_raw(11));
        p.record(s1, vec![x], TensorId::from_raw(12));
        p.record(s3, vec![x], TensorId::from_raw(13));
        let mut got = p.unique_streams();
        got.sort();
        let mut expected = vec![s1, s2, s3];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn clear_resets_state() {
        let mut p = StreamPlanner::new();
        let s = p.reserve();
        p.record(s, vec![TensorId::from_raw(1)], TensorId::from_raw(2));
        p.clear();
        assert!(p.is_empty());
        // After clear, next reserve restarts at 1 (DEFAULT is still 0).
        assert_eq!(p.reserve().as_raw(), 1);
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", StreamId::DEFAULT), "stream:default");
        assert_eq!(format!("{}", StreamId::from_raw(7)), "stream:7");
    }
}

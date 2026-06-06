//! kiln-graph — backend-agnostic command-list / graph capture.
//!
//! This crate is the shared replay vocabulary, not the current
//! production replay authority. Production decode replay still lives in
//! model-level runners:
//!
//! - `crates/kiln-model/src/cuda_graph.rs` for CUDA graphs.
//! - `crates/kiln-model/src/rocm_graph.rs` for HIP graphs.
//! - `crates/kiln-model/src/metal_graph.rs` plus
//!   `kiln_graph_metal::MetalCapturedGraph` for Metal ICB replay.
//! - `crates/kiln-model/src/vk_decode_resident.rs` and
//!   `kiln-vulkan-kernel/src/cmd_batch.rs` for Vulkan resident command
//!   batching.
//!
//! The backend `kiln-graph-*` crates are scaffolds and small reusable
//! replay objects until Phase 5 moves or wraps those production runners
//! behind one authoritative replay contract.
//!
//! Phase 5 of #1082. Per the issue:
//!
//! > **kiln-tensor allocator "freeze-pointers" mode.** Active for the
//! > duration of `capture()` … `replay()`. Allocations during freeze
//! > come from a pre-sized pool with no reclamation. This is the
//! > structural fix that makes batched cuda-graph capture possible;
//! > without it Phase 8 doesn't land.
//!
//! And:
//!
//! > **Capture-lifetime programming model (the dangling-pointer rule).**
//! > Any tensor whose `.device_ptr()` enters a captured graph must
//! > outlive every replay of that graph. Lifetimes are encoded in the
//! > type system where possible (`CapturedGraph<'a>` borrows from a
//! > `FrozenAllocator<'a>`) and enforced by a debug-assertion
//! > `kiln_tensor::audit_captured_pointers()` that walks the graph and
//! > verifies every recorded pointer still resolves to a live
//! > allocation.
//!
//! # Phase 5 scope (this PR)
//!
//! - [`CapturedGraph`] trait — `replay()` + per-backend metadata
//!   (`backend_name`, `replay_count`).
//! - [`CaptureSession`] — RAII guard for a capture lifetime. Records
//!   the set of pinned pointers; verifies on drop that no pointer
//!   was freed mid-capture.
//! - [`AllocatorMode`] enum — `Owned` / `Pool` / `Frozen` per the
//!   issue's "Allocator design — three modes" bullet.
//! - [`PinnedPointer`] + audit walker — debug-assert any pointer
//!   referenced by a captured graph still resolves to a live
//!   allocation.
//! - [`CaptureError`] — typed error surface for the per-backend impls.
//!
//! # What this PR does NOT do
//!
//! - The actual CUDA `cudaGraph_t` wrapping (`kiln-graph-cuda`,
//!   Phase 5.x).
//! - The Metal `MTLIndirectCommandBuffer` wrapping (`kiln-graph-metal`).
//! - The Vulkan secondary-command-buffer wrapping (extends
//!   `kiln-vulkan-kernel::cmd_batch.rs`).
//! - The bs>1 cuda-graph stream-DAG dispatch (revives the dead-code
//!   `CudaBatchedGraphKey` at `kiln-model/src/cuda_graph.rs:178`) —
//!   that lands once `kiln-graph-cuda` ships.
//! - AOT graph serialization via `cuGraphSerialize` (Phase 5
//!   final bullet) — Phase 5.x follow-up.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod allocator_mode;
mod capture_session;
mod captured_graph;
mod error;

pub use allocator_mode::AllocatorMode;
pub use capture_session::{CaptureSession, PinnedPointer};
pub use captured_graph::CapturedGraph;
pub use error::CaptureError;

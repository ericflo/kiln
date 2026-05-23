//! `AllocatorMode` — re-exported from `kiln-tensor`.
//!
//! As of Phase 1.27, the canonical home of [`AllocatorMode`] is
//! `kiln-tensor` (the allocator IS a kiln-tensor concern; kiln-graph
//! consumes it via the [`crate::CaptureSession`] surface).
//!
//! This module exists so existing `use kiln_graph::AllocatorMode`
//! imports keep working — `kiln-graph` re-exports the canonical type.

pub use kiln_tensor::AllocatorMode;

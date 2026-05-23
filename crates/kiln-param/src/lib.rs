//! kiln-param — the unified Parameter handle.
//!
//! **One logical parameter, one stable `TensorId`, multiple physical
//! storages.** Replaces today's bookkeeping spread across:
//!
//! - `crates/kiln-model/src/packed_weight_registry.rs` (489 LOC)
//! - `crates/kiln-model/src/transposed_weight_cache.rs` (730 LOC)
//! - `crates/kiln-model/src/marlin_proj.rs`
//! - `crates/kiln-model/src/fp8.rs`
//! - `crates/kiln-model/src/lora_loader.rs`
//!
//! per the Phase 2.5 bullet in #1082.
//!
//! # Phase 2.5 scope
//!
//! This PR ships the **type definitions and the storage-coherence
//! state machine**:
//!
//! - `Parameter` — owns `forward_storage` + optional `backward_storage`
//!   + optional `transposed_cache` + optional `lora_delta`, all keyed
//!   by one stable [`kiln_tensor::TensorId`].
//! - `ForwardStorage` enum — `Bf16Tensor`, `Marlin`, `Fp8`, plus a
//!   `Fp4Packed` scaffold for Phase 8.10.
//! - `AmpPolicy` — per-Parameter declaration of
//!   `{forward_compute_dtype, backward_compute_dtype, master_dtype,
//!   accumulation_dtype}`. Per the issue: AMP is a property of the
//!   parameter, not an implicit behaviour of the call site.
//! - `OutputHead` registry — multiple output heads sharing trunk
//!   storage (LM head + MTP head + future value/reward heads).
//! - `Parameter::content_hash()` — xxhash3 placeholder for the
//!   Phase 2.5 content-addressed weight identity. Today returns
//!   a deterministic hash of the forward storage bytes; full
//!   xxhash3 integration is a Phase 2.5.x follow-up.
//!
//! # What this PR does NOT do
//!
//! - The actual re-quantization kernel on backward (`dequant → update
//!   → requant`). That lives in `kiln-optim` (Phase 6.5).
//! - Storage-coherence enforcement at runtime (require_fresh /
//!   mark_stale state machine). Today's API exposes the slots and
//!   policy; Phase 2.5.x wires the runtime invariants.
//! - LoRA hot-swap semantics. Scaffold (`lora_delta` slot present);
//!   merge / swap kernels live downstream.
//!
//! Each lands as a separate small PR per anti-pattern 5.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod amp_policy;
mod content_hash;
mod parameter;

pub use amp_policy::AmpPolicy;
pub use content_hash::content_hash_storage;
pub use parameter::{ForwardStorage, OutputHead, OutputHeadRole, Parameter};

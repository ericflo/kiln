//! `kiln-memory` — Kiln's cross-engine memory awareness layer.
//!
//! This crate is the single home for everything about *how much memory there is
//! and how it's being used* on whatever accelerator kiln is running on. It is
//! **backend-agnostic by construction**: the probes read memory at the OS level
//! (`nvidia-smi`, AMD/Intel DRM sysfs, Apple `sysctl`, host `/proc/meminfo`),
//! not through any one GPU API, so a single set of figures is correct for CUDA,
//! ROCm, Vulkan, Metal, and CPU alike — and it even sees memory used by other
//! processes sharing the device.
//!
//! # Layers
//!
//! * [`vram`] — the low-level probes and [`vram::current_memory_snapshot`], the
//!   keystone every other subsystem reads (KV-cache sizing, graph-capture
//!   headroom, the budget arbiter, allocator pressure response). Unified-memory
//!   aware: on an APU / Apple Silicon, "free" tracks host RAM, not a VRAM
//!   carveout.
//!
//! Subsequent layers (the continuous `MemoryGovernor`, the pressure-aware
//! allocator hooks, and the training/inference budget arbiter) build on top of
//! `vram` and land in this crate as they're implemented, so memory governance
//! stays in one place rather than scattered across the server and model crates.

pub mod governor;
pub mod vram;

pub use governor::{
    AutomaticReclaimStats, GovernorConfig, MEMORY_RECLAIM_MODE_ENV, MemoryGovernor, MemoryPressure,
    MemoryReclaimMode, MemorySource, OsProbe, Reservation,
};
pub use vram::{MemorySnapshot, current_free_bytes, current_memory_snapshot};

/// Process-wide lock serializing tests that mutate environment variables that
/// the memory probes read (e.g. `KILN_GPU_MEMORY_GB`,
/// `KILN_TRAINING_MEMORY_RESERVE_GB`). `cargo nextest` runs each test in its own
/// process so this is belt-and-suspenders for `cargo test`'s shared-process
/// model. Moved here with `vram` from `kiln-core::env_flag`.
pub static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

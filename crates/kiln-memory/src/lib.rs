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
//!   aware when the driver exposes an unambiguous host-shared topology; Apple
//!   free tracks host pressure, while ambiguous Linux DRM heaps fail closed to
//!   device-local VRAM rather than promoting GTT into allocation capacity.
//!
//! Subsequent layers (the continuous `MemoryGovernor`, the pressure-aware
//! allocator hooks, and the training/inference budget arbiter) build on top of
//! `vram` and land in this crate as they're implemented, so memory governance
//! stays in one place rather than scattered across the server and model crates.

pub mod governor;
pub mod vram;

pub use governor::{
    AutomaticReclaimStats, CachedSampleStatus, GlobalGovernorConfiguration,
    GlobalGovernorConfigurationError, GovernorConfig, MemoryGovernor, MemoryGovernorObservation,
    MemoryPressure, MemoryReclaimMode, MemorySource, OsProbe, Reservation,
};
pub use vram::{
    LinuxDrmVendor, MemorySnapshot, MemorySnapshotObservations, MemoryTierSnapshot,
    VramCapacityResolution, VramProbeIdentityError, VramProbeSelector, current_free_bytes,
    current_free_bytes_for, current_free_vram_bytes, current_free_vram_bytes_for,
    current_memory_snapshot, current_memory_snapshot_for, detect_used_vram, detect_used_vram_for,
    detect_vram, detect_vram_for, resolve_vram_capacity, validate_vram_probe_identity,
};

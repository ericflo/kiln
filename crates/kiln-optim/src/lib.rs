//! kiln-optim — fused per-backend optimizer step.
//!
//! Phase 6.5 of #1082. Per the issue's bullet:
//!
//! > Without an explicit shared crate the fused-optimizer logic gets
//! > rebuilt three times — Vulkan already has `adamw_step_bf16.comp`;
//! > the CUDA path goes through `Var::set` via candle; Metal has nothing.
//!
//! # Phase 6.5 scope (this PR)
//!
//! - [`OptimStep`] trait — generic over `Parameter`. Variants:
//!   `AdamW`, `Sgd`, `Lion`, `Muon`.
//! - [`MomentLocation`] enum — `Device` / `PinnedHost` / `MmappedDisk`,
//!   per the issue's "Optimizer-state location seam" bullet.
//! - [`StochasticRoundingPolicy`] — explicit programmatic policy for reference
//!   updates; ordinary product construction is round-to-nearest.
//! - [`AdamW`] CPU reference impl. F32-master + F32-moments, runs on
//!   `Parameter::backward_storage`. The migration target for the
//!   existing `crates/kiln-train/src/trainer.rs`
//!   `HashMap<TensorId, AdamWMoments>` (Phase 0.1 audit, 30 sites).
//! - [`Muon`] CPU reference impl — momentum-orthogonalized SGD with the
//!   gram-space P-accumulator Newton-Schulz. This is the **oracle** the
//!   per-backend fused Muon kernels (CUDA / ROCm / Vulkan / Metal) are
//!   validated against. [`newton_schulz`] is exported for those parity
//!   tests.
//!
//! Per-backend (CUDA / Metal / Vulkan) on-device impls plug in via the
//! `OptimizerBackend::runtime_dispatch_*_step` seam in `kiln-model`,
//! falling back to these CPU references when operands aren't resident.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod accumulate_step;
mod adamw;
mod grad_accumulator;
mod lion_muon;
pub mod lr_schedule;
mod optim_step;
mod policy;
mod sgd;

pub use accumulate_step::accumulate_then_step;
pub use adamw::{AdamW, AdamWHyperparameters, AdamWMoments};
pub use grad_accumulator::GradAccumulator;
pub use lion_muon::{Lion, LionEma, Muon, MuonState, newton_schulz};
pub use optim_step::{OptimStep, StepError};
pub use policy::{MomentLocation, StochasticRoundingPolicy};
pub use sgd::{Sgd, SgdHyperparameters, SgdMomentum};

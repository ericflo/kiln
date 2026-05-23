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
//! - [`StochasticRoundingPolicy`] — read at step time; toggled by
//!   `KILN_BF16_STOCHASTIC_ROUND=1`.
//! - [`AdamW`] CPU reference impl. F32-master + F32-moments, runs on
//!   `Parameter::backward_storage`. The migration target for the
//!   existing `crates/kiln-train/src/trainer.rs`
//!   `HashMap<TensorId, AdamWMoments>` (Phase 0.1 audit, 30 sites).
//!
//! Per-backend (CUDA / Metal / Vulkan) impls plug in via the same
//! `OptimStep` trait in subsequent PRs.

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
pub use lion_muon::{Lion, LionEma, Muon, MuonState};
pub use optim_step::{OptimStep, StepError};
pub use policy::{MomentLocation, StochasticRoundingPolicy};
pub use sgd::{Sgd, SgdHyperparameters, SgdMomentum};

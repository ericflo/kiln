//! #1082 Phase 2.7 — **every** optimizer bumps the `Parameter` epoch
//! counter at end-of-optimizer-step.
//!
//! The epoch counter is the version a live serving thread polls (via
//! `Parameter::version_handle`) to learn that the master advanced and
//! its cached forward view is stale. If any `OptimStep` impl finishes a
//! step without bumping, that signal is silently lost — so this is the
//! cross-optimizer guard: a future optimizer that forgets
//! `param.bump_epoch()` fails CI here.

use kiln_optim::{AdamW, AdamWHyperparameters, Lion, Muon, OptimStep, Sgd, SgdHyperparameters};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor::Tensor;

/// 2×2 master so Muon's Newton–Schulz (square-matrix) path is happy;
/// the elementwise optimizers accept the same shape.
fn make_param() -> Parameter {
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let master = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    Parameter::trainable(ForwardStorage::Plain(t), master, AmpPolicy::fp32_reference())
}

fn grad() -> Tensor {
    Tensor::from_slice(&[0.1f32, 0.1, 0.1, 0.1], vec![2, 2]).unwrap()
}

fn assert_steps_bump_epoch(opt: &mut dyn OptimStep, name: &str) {
    let mut p = make_param();
    assert_eq!(p.current_epoch(), 0, "{name}: fresh parameter starts at epoch 0");
    opt.step(&mut p, &grad()).unwrap();
    assert_eq!(
        p.current_epoch(),
        1,
        "{name}: one optimizer step must bump the epoch to 1"
    );
    opt.step(&mut p, &grad()).unwrap();
    assert_eq!(p.current_epoch(), 2, "{name}: a second step bumps to 2");
}

#[test]
fn adamw_step_bumps_epoch() {
    let mut opt = AdamW::new(AdamWHyperparameters::default());
    assert_steps_bump_epoch(&mut opt, "AdamW");
}

#[test]
fn sgd_step_bumps_epoch() {
    let mut opt = Sgd::new(SgdHyperparameters::default());
    assert_steps_bump_epoch(&mut opt, "Sgd");
}

#[test]
fn lion_step_bumps_epoch() {
    let mut opt = Lion::new(1e-3, 0.9, 0.99, 0.0);
    assert_steps_bump_epoch(&mut opt, "Lion");
}

#[test]
fn muon_step_bumps_epoch() {
    let mut opt = Muon::new(1e-3, 0.9, 5);
    assert_steps_bump_epoch(&mut opt, "Muon");
}

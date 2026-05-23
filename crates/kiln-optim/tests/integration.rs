//! kiln-optim integration tests.
//!
//! Exercises `OptimStep::step(&mut Parameter, &Tensor)` end-to-end
//! with multiple parameters, multiple step kinds, and the
//! anti-pattern 11 contract (stable `TensorId` keying).

use kiln_optim::{AdamW, AdamWHyperparameters, OptimStep, Sgd, SgdHyperparameters};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor as kt;

fn fresh_param(values: &[f32]) -> Parameter {
    let fwd = kt::Tensor::from_slice(values, vec![values.len()]).unwrap();
    let master = kt::Tensor::from_slice(values, vec![values.len()]).unwrap();
    Parameter::trainable(ForwardStorage::Plain(fwd), master, AmpPolicy::fp32_reference())
}

#[test]
fn adamw_multiple_parameters_keep_separate_state() {
    // Anti-pattern 11: AdamW's HashMap<TensorId, AdamWMoments> must
    // keep per-parameter state distinct. Step two params with different
    // grads and verify their moments diverge.
    let mut opt = AdamW::default_hp();

    let mut p1 = fresh_param(&[1.0, 2.0, 3.0, 4.0]);
    let mut p2 = fresh_param(&[10.0, 20.0, 30.0, 40.0]);

    let g1 = kt::Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
    let g2 = kt::Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], vec![4]).unwrap();

    opt.step(&mut p1, &g1).unwrap();
    opt.step(&mut p2, &g2).unwrap();

    let m1 = opt.moments(p1.tensor_id()).expect("p1 moments");
    let m2 = opt.moments(p2.tensor_id()).expect("p2 moments");

    // After one step: m[i] = (1 - beta1) * g[i] = 0.1 * g[i]
    for (i, &expected) in [0.01f32, 0.02, 0.03, 0.04].iter().enumerate() {
        assert!((m1.m[i] - expected).abs() < 1e-6, "p1 m[{i}]={}", m1.m[i]);
    }
    for v in &m2.m {
        assert!((v - 0.05).abs() < 1e-6, "p2 m={v}");
    }

    // Stepping p1 a second time advances p1.step but not p2.
    opt.step(&mut p1, &g1).unwrap();
    let m1_after = opt.moments(p1.tensor_id()).unwrap();
    let m2_after = opt.moments(p2.tensor_id()).unwrap();
    assert_eq!(m1_after.step, 2);
    assert_eq!(m2_after.step, 1, "p2 step should not advance when p1 steps");
    assert_eq!(opt.parameter_count(), 2);
}

#[test]
fn adamw_lora_swap_preserves_optimizer_state() {
    // Anti-pattern 11: replacing `lora_delta` (or any other slot)
    // does NOT change tensor_id; therefore AdamW state survives.
    let mut opt = AdamW::default_hp();
    let mut p = fresh_param(&[1.0, 2.0, 3.0, 4.0]);
    let g = kt::Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();

    opt.step(&mut p, &g).unwrap();
    let id_before = p.tensor_id();
    let step_before = opt.moments(id_before).unwrap().step;
    assert_eq!(step_before, 1);

    // Hot-swap a LoRA delta (anti-pattern 11: tensor_id unchanged).
    let delta = kt::Tensor::zeros_cpu(vec![4], kt::DType::F32);
    p.set_lora_delta(Some(delta));
    assert_eq!(p.tensor_id(), id_before);

    // Step again. AdamW must still find the existing moments under
    // the stable tensor_id.
    opt.step(&mut p, &g).unwrap();
    let step_after = opt.moments(id_before).unwrap().step;
    assert_eq!(
        step_after, 2,
        "AdamW state orphaned by LoRA swap — anti-pattern 11 violation"
    );
}

#[test]
fn adamw_forward_storage_swap_preserves_optimizer_state() {
    // Anti-pattern 11: replacing forward_storage (e.g. BF16 → Marlin)
    // does NOT change tensor_id.
    let mut opt = AdamW::default_hp();
    let mut p = fresh_param(&[1.0, 2.0, 3.0, 4.0]);
    let g = kt::Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();

    opt.step(&mut p, &g).unwrap();
    let id_before = p.tensor_id();
    assert_eq!(opt.moments(id_before).unwrap().step, 1);

    // Swap forward storage from Plain to Marlin shape (we lie about
    // dtypes since we never actually use the new forward storage in
    // this test — the OptimizerStep::step path only reads
    // backward_storage).
    let packed = kt::Tensor::zeros_cpu(vec![4], kt::DType::Int4Packed);
    let scales = kt::Tensor::zeros_cpu(vec![4], kt::DType::BF16);
    p.replace_forward_storage(ForwardStorage::Marlin { packed, scales });
    assert_eq!(p.tensor_id(), id_before);
    assert_eq!(p.forward_storage().kind_name(), "marlin");

    // Step again. The AdamW state should survive the storage swap.
    opt.step(&mut p, &g).unwrap();
    assert_eq!(
        opt.moments(id_before).unwrap().step,
        2,
        "AdamW state orphaned by forward_storage swap"
    );
}

#[test]
fn sgd_with_momentum_runs_through_optim_step_trait() {
    // Verify Sgd is usable through the OptimStep trait — same shape
    // as AdamW so dispatch code can be generic.
    let mut opt: Box<dyn OptimStep> = Box::new(Sgd::new(SgdHyperparameters {
        lr: 0.01,
        momentum: 0.9,
        weight_decay: 0.0,
        nesterov: false,
    }));
    assert_eq!(opt.name(), "sgd");
    let mut p = fresh_param(&[1.0, 2.0]);
    let g = kt::Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
    opt.step(&mut p, &g).unwrap();
    opt.reset();
    // reset() works through the trait too.
    opt.step(&mut p, &g).unwrap();
}

#[test]
fn optim_step_dispatch_works_via_trait_object() {
    // Pattern: a training loop has `Vec<Box<dyn OptimStep>>` indexed
    // by parameter group. Verify both AdamW and Sgd compose through
    // the trait without per-impl knowledge.
    let mut optimizers: Vec<Box<dyn OptimStep>> = vec![
        Box::new(AdamW::new(AdamWHyperparameters {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        })),
        Box::new(Sgd::new(SgdHyperparameters {
            lr: 1e-2,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        })),
    ];
    let names: Vec<&'static str> = optimizers.iter().map(|o| o.name()).collect();
    assert_eq!(names, vec!["adamw", "sgd"]);

    let mut p = fresh_param(&[1.0f32, 2.0]);
    let g = kt::Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
    for opt in optimizers.iter_mut() {
        opt.step(&mut p, &g).unwrap();
    }
}

#[test]
fn optim_step_rejects_grad_shape_mismatch_under_trait_dispatch() {
    // Same error surface for both AdamW and Sgd.
    let cases: Vec<Box<dyn OptimStep>> = vec![
        Box::new(AdamW::default_hp()),
        Box::new(Sgd::default_hp()),
    ];
    for mut opt in cases.into_iter() {
        let mut p = fresh_param(&[1.0f32, 2.0, 3.0]);
        let g = kt::Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        assert!(
            e.to_string().contains("shape"),
            "{} should report shape mismatch, got: {e}",
            opt.name()
        );
    }
}

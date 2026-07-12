//! Compare Kiln's portable F32 AdamW reference with a source-pinned PyTorch
//! trajectory, including epsilon-dominated gradients and both moment buffers.

use kiln_optim::{AdamW, AdamWHyperparameters, OptimStep};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor::{CpuStorage, DType, Tensor};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/adamw_pytorch_oracle_v1.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    tolerances: Tolerances,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    float32_absolute: f32,
    float32_relative: f32,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    class: String,
    dtype: String,
    hyperparameters: Hyperparameters,
    stored_initial_parameter: Vec<f32>,
    stored_gradients: Vec<Vec<f32>>,
    state_contract: StateContract,
    trajectory: Vec<Step>,
}

#[derive(Debug, Deserialize)]
struct Hyperparameters {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Debug, Deserialize)]
struct StateContract {
    parameter_dtype: String,
    first_moment_dtype: String,
    second_moment_dtype: String,
    separate_master_parameter: bool,
}

#[derive(Debug, Deserialize)]
struct Step {
    step: u64,
    parameter: Vec<f32>,
    exp_avg: Vec<f32>,
    exp_avg_sq: Vec<f32>,
}

fn read_cpu_f32(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), DType::F32);
    let storage = tensor
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("oracle master must remain CPU-backed");
    storage
        .as_bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_close(
    case: &str,
    step: u64,
    field: &str,
    actual: &[f32],
    expected: &[f32],
    absolute: f32,
    relative: f32,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case} step {step} {field} length"
    );
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let error = (got - want).abs();
        let tolerance = absolute + relative * want.abs();
        assert!(
            error <= tolerance,
            "{case} step {step} {field}[{index}]: got={got:.12e} \
             want={want:.12e} error={error:.12e} tolerance={tolerance:.12e}"
        );
    }
}

#[test]
fn portable_f32_adamw_matches_pytorch_low_and_ordinary_trajectories() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid AdamW oracle fixture");
    assert_eq!(fixture.schema, "kiln.adamw-pytorch-oracle.v1");
    let cases: Vec<&Case> = fixture
        .cases
        .iter()
        .filter(|case| case.dtype == "float32")
        .collect();
    assert_eq!(cases.len(), 2, "fixture must contain two F32 cases");

    for case in cases {
        assert!(matches!(case.class.as_str(), "ordinary" | "low_gradient"));
        assert_eq!(case.state_contract.parameter_dtype, "float32");
        assert_eq!(case.state_contract.first_moment_dtype, "float32");
        assert_eq!(case.state_contract.second_moment_dtype, "float32");
        assert!(!case.state_contract.separate_master_parameter);
        assert_eq!(case.stored_gradients.len(), case.trajectory.len());

        let shape = vec![case.stored_initial_parameter.len()];
        let forward = Tensor::from_slice(&case.stored_initial_parameter, shape.clone()).unwrap();
        let master = Tensor::from_slice(&case.stored_initial_parameter, shape.clone()).unwrap();
        let mut parameter = Parameter::trainable(
            ForwardStorage::Plain(forward),
            master,
            AmpPolicy::fp32_reference(),
        );
        let id = parameter.tensor_id();
        let hp = &case.hyperparameters;
        let mut optimizer = AdamW::new(AdamWHyperparameters {
            lr: hp.lr,
            beta1: hp.beta1,
            beta2: hp.beta2,
            eps: hp.eps,
            weight_decay: hp.weight_decay,
        });

        for (gradient, expected) in case.stored_gradients.iter().zip(&case.trajectory) {
            let gradient = Tensor::from_slice(gradient, shape.clone()).unwrap();
            optimizer.step(&mut parameter, &gradient).unwrap();
            let state = optimizer.moments(id).expect("AdamW moments");
            assert_eq!(state.step, expected.step, "{} step counter", case.id);
            let actual_parameter = read_cpu_f32(
                parameter
                    .backward_storage()
                    .expect("AdamW must retain an F32 master"),
            );
            assert_close(
                &case.id,
                expected.step,
                "parameter",
                &actual_parameter,
                &expected.parameter,
                fixture.tolerances.float32_absolute,
                fixture.tolerances.float32_relative,
            );
            assert_close(
                &case.id,
                expected.step,
                "exp_avg",
                &state.m,
                &expected.exp_avg,
                fixture.tolerances.float32_absolute,
                fixture.tolerances.float32_relative,
            );
            assert_close(
                &case.id,
                expected.step,
                "exp_avg_sq",
                &state.v,
                &expected.exp_avg_sq,
                fixture.tolerances.float32_absolute,
                fixture.tolerances.float32_relative,
            );
        }
    }
}

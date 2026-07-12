//! Compare each production optimizer kernel with the pinned PyTorch AdamW
//! trajectory at the backend's declared native parameter/moment dtype.

#![cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan"
))]

use anyhow::{Context, Result};
use kiln_model::backend::{OptimizerBackend, ResidencyBackend};
use kiln_tensor::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::BTreeMap;

const FIXTURE: &str = include_str!("../../kiln-optim/tests/fixtures/adamw_pytorch_oracle_v1.json");

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
    bfloat16_parameter_max_ulp: u16,
    bfloat16_first_moment_max_ulp: u16,
    bfloat16_second_moment_max_ulp: u16,
    bfloat16_reason: String,
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
    step: u32,
    parameter: Vec<f32>,
    exp_avg: Vec<f32>,
    exp_avg_sq: Vec<f32>,
}

fn qualification_required() -> bool {
    std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1")
}

fn unavailable(backend: &str, reason: &str) {
    if qualification_required() {
        panic!("required {backend} AdamW oracle unavailable: {reason}");
    }
    eprintln!("skip {backend} AdamW PyTorch oracle: {reason}");
}

fn tensor_from_f32(values: &[f32], dtype: DType, device: Device) -> Result<Tensor> {
    let shape = vec![values.len()];
    Ok(match dtype {
        DType::F32 => Tensor::from_vec_on(device, values.to_vec(), shape)?,
        DType::BF16 => Tensor::from_vec_on(
            device,
            values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        )?,
        other => anyhow::bail!("unsupported AdamW oracle dtype {other:?}"),
    })
}

fn read_current<B: ResidencyBackend>(
    backend: &B,
    tensor: &Tensor,
    dtype: DType,
) -> Result<Vec<f32>> {
    let current = ResidencyBackend::runtime_resolve_resident_activation(
        backend,
        tensor,
        tensor.dims(),
        dtype,
    )?
    .unwrap_or_else(|| tensor.clone());
    current
        .to_device(Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .map_err(Into::into)
}

fn assert_f32_close(
    backend: &str,
    case: &str,
    step: u32,
    field: &str,
    actual: &[f32],
    expected: &[f32],
    absolute: f32,
    relative: f32,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let error = (got - want).abs();
        let tolerance = absolute + relative * want.abs();
        assert!(
            error <= tolerance,
            "{backend} {case} step {step} {field}[{index}]: got={got:.12e} \
             want={want:.12e} error={error:.12e} tolerance={tolerance:.12e}"
        );
    }
}

fn bf16_ulp_distance(left: f32, right: f32) -> u16 {
    if left == right {
        return 0;
    }
    let left = half::bf16::from_f32(left).to_bits();
    let right = half::bf16::from_f32(right).to_bits();
    left.abs_diff(right)
}

fn assert_bf16_close(
    backend: &str,
    case: &str,
    step: u32,
    field: &str,
    actual: &[f32],
    expected: &[f32],
    max_observed: &mut BTreeMap<String, (u16, String)>,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let distance = bf16_ulp_distance(got, want);
        let observed = max_observed.entry(field.to_string()).or_default();
        if distance > observed.0 {
            *observed = (
                distance,
                format!(
                    "{backend} {case} step {step} {field}[{index}]: \
                     got={got:.12e} want={want:.12e}"
                ),
            );
        }
    }
}

fn assert_values(
    backend: &str,
    case: &str,
    step: u32,
    field: &str,
    dtype: DType,
    actual: &[f32],
    expected: &[f32],
    tolerances: &Tolerances,
    max_bf16_ulp: &mut BTreeMap<String, (u16, String)>,
) {
    match dtype {
        DType::F32 => assert_f32_close(
            backend,
            case,
            step,
            field,
            actual,
            expected,
            tolerances.float32_absolute,
            tolerances.float32_relative,
        ),
        DType::BF16 => {
            assert_bf16_close(backend, case, step, field, actual, expected, max_bf16_ulp)
        }
        other => panic!("unsupported AdamW oracle dtype {other:?}"),
    }
}

fn run_backend_oracle<B>(backend: &B, device: Device, dtype: DType, name: &str) -> Result<()>
where
    B: OptimizerBackend + ResidencyBackend,
{
    let fixture: Fixture = serde_json::from_str(FIXTURE).context("parse AdamW oracle fixture")?;
    anyhow::ensure!(
        fixture.schema == "kiln.adamw-pytorch-oracle.v1",
        "unexpected AdamW oracle schema"
    );
    let dtype_name = match dtype {
        DType::F32 => "float32",
        DType::BF16 => "bfloat16",
        other => anyhow::bail!("unsupported native AdamW dtype {other:?}"),
    };
    let cases: Vec<&Case> = fixture
        .cases
        .iter()
        .filter(|case| case.dtype == dtype_name)
        .collect();
    anyhow::ensure!(cases.len() == 2, "expected two {dtype_name} cases");
    let mut total_steps = 0usize;
    let mut max_bf16_ulp = BTreeMap::new();

    for case in cases {
        anyhow::ensure!(matches!(case.class.as_str(), "ordinary" | "low_gradient"));
        anyhow::ensure!(case.state_contract.parameter_dtype == dtype_name);
        anyhow::ensure!(case.state_contract.first_moment_dtype == dtype_name);
        anyhow::ensure!(case.state_contract.second_moment_dtype == dtype_name);
        anyhow::ensure!(!case.state_contract.separate_master_parameter);
        anyhow::ensure!(case.stored_gradients.len() == case.trajectory.len());

        let parameter = tensor_from_f32(&case.stored_initial_parameter, dtype, device)?;
        let first_moment = tensor_from_f32(
            &vec![0.0; case.stored_initial_parameter.len()],
            dtype,
            device,
        )?;
        let second_moment = tensor_from_f32(
            &vec![0.0; case.stored_initial_parameter.len()],
            dtype,
            device,
        )?;
        ResidencyBackend::runtime_register_resident_activation(backend, &parameter)?;
        ResidencyBackend::runtime_register_resident_activation(backend, &first_moment)?;
        ResidencyBackend::runtime_register_resident_activation(backend, &second_moment)?;

        for (gradient, expected) in case.stored_gradients.iter().zip(&case.trajectory) {
            let gradient = tensor_from_f32(gradient, dtype, device)?;
            ResidencyBackend::runtime_register_resident_activation(backend, &gradient)?;
            let hp = &case.hyperparameters;
            let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
                backend,
                &parameter,
                &gradient,
                &first_moment,
                &second_moment,
                hp.lr,
                hp.beta1,
                hp.beta2,
                hp.eps,
                hp.weight_decay,
                expected.step,
            )?;
            ResidencyBackend::runtime_evict_resident_activation(backend, &gradient);
            anyhow::ensure!(
                dispatched,
                "{name} declined required {} step {}",
                case.id,
                expected.step
            );

            let actual_parameter = read_current(backend, &parameter, dtype)?;
            let actual_first = read_current(backend, &first_moment, dtype)?;
            let actual_second = read_current(backend, &second_moment, dtype)?;
            assert_values(
                name,
                &case.id,
                expected.step,
                "parameter",
                dtype,
                &actual_parameter,
                &expected.parameter,
                &fixture.tolerances,
                &mut max_bf16_ulp,
            );
            assert_values(
                name,
                &case.id,
                expected.step,
                "exp_avg",
                dtype,
                &actual_first,
                &expected.exp_avg,
                &fixture.tolerances,
                &mut max_bf16_ulp,
            );
            assert_values(
                name,
                &case.id,
                expected.step,
                "exp_avg_sq",
                dtype,
                &actual_second,
                &expected.exp_avg_sq,
                &fixture.tolerances,
                &mut max_bf16_ulp,
            );
            total_steps += 1;
        }

        ResidencyBackend::runtime_evict_resident_activation(backend, &parameter);
        ResidencyBackend::runtime_evict_resident_activation(backend, &first_moment);
        ResidencyBackend::runtime_evict_resident_activation(backend, &second_moment);
    }

    let observed_summary: BTreeMap<&str, u16> = max_bf16_ulp
        .iter()
        .map(|(field, (distance, _))| (field.as_str(), *distance))
        .collect();
    if dtype == DType::BF16 {
        let allowed = BTreeMap::from([
            ("exp_avg", fixture.tolerances.bfloat16_first_moment_max_ulp),
            (
                "exp_avg_sq",
                fixture.tolerances.bfloat16_second_moment_max_ulp,
            ),
            ("parameter", fixture.tolerances.bfloat16_parameter_max_ulp),
        ]);
        for (field, (distance, context)) in &max_bf16_ulp {
            let limit = allowed
                .get(field.as_str())
                .with_context(|| format!("missing BF16 tolerance for {field}"))?;
            anyhow::ensure!(
                distance <= limit,
                "{name} BF16 AdamW exceeded PyTorch {field} tolerance: \
                 observed={distance} allowed={limit} at {context}; {}",
                fixture.tolerances.bfloat16_reason
            );
        }
    }

    println!(
        "[ADAMW PYTORCH ORACLE PASS] backend={name} dtype={dtype_name} \
         cases=2 steps={total_steps} max_bf16_ulp={observed_summary:?}"
    );
    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_adamw_matches_pytorch_native_bf16_contract() -> Result<()> {
    use kiln_model::backend::rocm::RocmBackend;
    if !kiln_tensor::rocm_is_available() {
        unavailable("rocm", "no ROCm device");
        return Ok(());
    }
    let device = Device::Rocm(0);
    run_backend_oracle(&RocmBackend::new(device), device, DType::BF16, "rocm")
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_adamw_matches_pytorch_native_bf16_contract() -> Result<()> {
    use kiln_model::backend::cuda::CudaBackend;
    if !kiln_tensor::cuda_is_available() {
        unavailable("cuda", "no CUDA device");
        return Ok(());
    }
    let device = Device::Cuda(0);
    run_backend_oracle(&CudaBackend::new(device), device, DType::BF16, "cuda")
}

#[cfg(feature = "metal")]
#[test]
fn metal_adamw_matches_pytorch_native_bf16_contract() -> Result<()> {
    use kiln_model::backend::metal::MetalBackend;
    if !kiln_tensor::metal_is_available() {
        unavailable("metal", "no Metal device");
        return Ok(());
    }
    let device = Device::Metal(0);
    run_backend_oracle(&MetalBackend::new(device), device, DType::BF16, "metal")
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_adamw_matches_pytorch_native_f32_contract() -> Result<()> {
    use kiln_model::backend::vulkan::{VulkanBackend, vulkan_is_available};
    if !vulkan_is_available() {
        unavailable("vulkan", "no Vulkan device");
        return Ok(());
    }
    run_backend_oracle(
        &VulkanBackend::new(Device::Cpu),
        Device::Cpu,
        DType::F32,
        "vulkan",
    )
}

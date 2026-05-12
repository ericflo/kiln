//! Minimal CUDA-native training loop helpers.
//!
//! This module is intentionally tiny while the CUDA-native Qwen path is still
//! being built. It bridges `kiln_model::cuda_train` tensor/autograd primitives
//! to an optimizer step in the training crate, mirroring the direction of the
//! Vulkan `vk_train` module without claiming full model coverage yet.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, TensorId};
use kiln_model::cuda_train::{
    CudaAdamWConfig, CudaAdamWState, CudaTrainArena, CudaTrainTensor,
    cuda_adamw_step_from_store, cuda_backward, cuda_matmul, cuda_mul, cuda_sum_all,
};
use std::collections::HashMap;
use std::path::Path;

pub type CudaAdamWBook = HashMap<TensorId, CudaAdamWState>;

pub fn allocate_cuda_adamw_state(params: &[CudaTrainTensor]) -> Result<CudaAdamWBook> {
    let mut states = HashMap::new();
    for param in params {
        let Some(param_id) = param.param_id() else {
            continue;
        };
        states.insert(param_id, CudaAdamWState::zeros_like(param)?);
    }
    Ok(states)
}

/// Run one native CUDA linear training step for `loss = sum((input @ weight)^2)`.
pub fn cuda_linear_sum_square_adamw_step(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
) -> Result<f32> {
    let mut arena = CudaTrainArena::new(input.as_tensor().device())?;
    cuda_linear_sum_square_adamw_step_with_arena(input, weight, adamw_state, cfg, &mut arena)
}

/// Run one native CUDA linear training step using caller-owned arena accounting.
pub fn cuda_linear_sum_square_adamw_step_with_arena(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        weight.param_id().is_some(),
        "cuda_linear_sum_square_adamw_step requires a parameter weight"
    );
    let output = arena.track(cuda_matmul(input, weight).context("cuda linear forward")?)?;
    let squared = arena.track(cuda_mul(&output, &output).context("cuda linear square loss")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda linear reduce loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda linear backward")?;
    let updated = cuda_adamw_step_from_store(&[weight.clone()], &grads, adamw_state, cfg)
        .context("cuda linear AdamW step")?;
    ensure!(
        updated == 1,
        "cuda_linear_sum_square_adamw_step expected one updated parameter, got {updated}"
    );
    Ok(loss_value)
}

/// Save named CUDA training tensors to safetensors after one CUDA-to-CPU readback.
pub fn save_cuda_training_tensors(
    weights: &[(&str, CudaTrainTensor)],
    output_path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, weight) in weights {
        ensure!(!name.is_empty(), "CUDA training safetensors key must not be empty");
        let tensor = weight
            .as_tensor()
            .to_dtype(DType::F32)
            .with_context(|| format!("convert CUDA training tensor {name} to f32"))?
            .to_device(&Device::Cpu)
            .with_context(|| format!("read CUDA training tensor {name} to CPU"))?;
        tensors.insert((*name).to_string(), tensor);
    }
    candle_core::safetensors::save(&tensors, output_path)
        .with_context(|| format!("save CUDA training tensors {}", output_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn cuda_linear_adamw_train_step_decreases_loss() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear AdamW smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let cfg = CudaAdamWConfig {
            lr: 0.1,
            ..CudaAdamWConfig::default()
        };

        let first = cuda_linear_sum_square_adamw_step(&input, &weight, &mut adamw, cfg)?;
        let second = cuda_linear_sum_square_adamw_step(&input, &weight, &mut adamw, cfg)?;
        assert!(
            second < first,
            "expected native CUDA linear AdamW loss to decrease: first={first} second={second}"
        );
        assert_eq!(adamw.get(&weight_id).expect("state").step, 2);
        Ok(())
    }

    #[test]
    fn cuda_linear_adamw_train_step_uses_arena_accounting() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear arena smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let mut arena = CudaTrainArena::new(&device)?;

        let loss = cuda_linear_sum_square_adamw_step_with_arena(
            &input,
            &weight,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.1,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;
        assert!(loss > 0.0);
        assert_eq!(arena.allocation_count(), 3);
        assert!(arena.allocated_bytes() >= 12);
        arena.clear();
        assert_eq!(arena.allocation_count(), 0);
        Ok(())
    }

    #[test]
    fn cuda_linear_weight_save_reflects_updated_cuda_tensor() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear save smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let mut arena = CudaTrainArena::new(&device)?;

        for _ in 0..2 {
            let loss = cuda_linear_sum_square_adamw_step_with_arena(
                &input,
                &weight,
                &mut adamw,
                CudaAdamWConfig {
                    lr: 0.1,
                    ..CudaAdamWConfig::default()
                },
                &mut arena,
            )?;
            assert!(loss.is_finite());
            arena.clear();
        }

        let expected = weight.to_vec_f32()?;
        let tmp = std::env::temp_dir().join(format!(
            "kiln-cuda-linear-weight-{}.safetensors",
            std::process::id()
        ));
        save_cuda_training_tensors(&[("linear.weight", weight.clone())], &tmp)?;

        let loaded = candle_core::safetensors::load(&tmp, &Device::Cpu)?;
        let saved = loaded
            .get("linear.weight")
            .context("missing saved linear.weight")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(saved, expected);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }
}

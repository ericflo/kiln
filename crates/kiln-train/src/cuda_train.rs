//! Minimal CUDA-native training loop helpers.
//!
//! This module is intentionally tiny while the CUDA-native Qwen path is still
//! being built. It bridges `kiln_model::cuda_train` tensor/autograd primitives
//! to an optimizer step in the training crate, mirroring the direction of the
//! Vulkan `vk_train` module without claiming full model coverage yet.

use anyhow::{Context, Result, ensure};
use candle_core::TensorId;
use kiln_model::cuda_train::{
    CudaAdamWConfig, CudaAdamWState, CudaTrainTensor, cuda_adamw_step_from_store, cuda_backward,
    cuda_matmul, cuda_mul, cuda_sum_all,
};
use std::collections::HashMap;

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
    ensure!(
        weight.param_id().is_some(),
        "cuda_linear_sum_square_adamw_step requires a parameter weight"
    );
    let output = cuda_matmul(input, weight).context("cuda linear forward")?;
    let squared = cuda_mul(&output, &output).context("cuda linear square loss")?;
    let loss = cuda_sum_all(&squared).context("cuda linear reduce loss")?;
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
}

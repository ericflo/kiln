//! Minimal CUDA-native training loop helpers.
//!
//! This module is intentionally tiny while the CUDA-native Qwen path is still
//! being built. It bridges `kiln_model::cuda_train` tensor/autograd primitives
//! to an optimizer step in the training crate, mirroring the direction of the
//! Vulkan `vk_train` module without claiming full model coverage yet.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, TensorId};
use kiln_model::cuda_train::{
    CudaAdamWConfig, CudaAdamWState, CudaFullAttentionLayer, CudaRopeTables, CudaTrainArena,
    CudaTrainTensor, cuda_adamw_step_from_store, cuda_backward, cuda_full_attention_layer,
    cuda_matmul, cuda_mul, cuda_sum_all,
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

/// Run one native CUDA FullAttention-layer training step for `loss = sum(layer(input)^2)`.
pub fn cuda_full_attention_sum_square_adamw_step_with_arena(
    input: &CudaTrainTensor,
    weights: &CudaFullAttentionLayer<'_>,
    trainable_params: &[CudaTrainTensor],
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        !trainable_params.is_empty(),
        "cuda_full_attention_sum_square_adamw_step requires trainable params"
    );
    for param in trainable_params {
        ensure!(
            param.param_id().is_some(),
            "cuda_full_attention_sum_square_adamw_step trainable params must be parameters"
        );
    }

    let output = arena
        .track(cuda_full_attention_layer(input, weights).context("cuda FullAttention forward")?)?;
    let squared = arena.track(cuda_mul(&output, &output).context("cuda FullAttention square")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda FullAttention reduce loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda FullAttention backward")?;
    let updated = cuda_adamw_step_from_store(trainable_params, &grads, adamw_state, cfg)
        .context("cuda FullAttention AdamW step")?;
    ensure!(
        updated > 0,
        "cuda_full_attention_sum_square_adamw_step expected at least one updated parameter"
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
    fn cuda_full_attention_adamw_train_step_updates_projection_weight() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda FullAttention AdamW smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.5f32, -1.0, 1.5, 0.25],
            (2usize, 2usize),
            &device,
        )?)?;
        let input_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.3, 0.05, 0.4, 0.4, 0.1, -0.2, 0.3],
            (2usize, 4usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let q_weight = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k_tensor = Tensor::from_vec(vec![0.1f32, 0.6, 0.8, -0.2], (2usize, 2usize), &device)?;
        let k_id = k_tensor.id();
        let k_weight = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v_tensor = Tensor::from_vec(vec![0.7f32, -0.2, -0.5, 0.6], (2usize, 2usize), &device)?;
        let v_id = v_tensor.id();
        let v_weight = CudaTrainTensor::parameter(v_tensor, v_id)?;
        let q_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let k_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let o_tensor = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let o_id = o_tensor.id();
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let gate_tensor =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let gate_id = gate_tensor.id();
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_tensor = Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
        let up_id = up_tensor.id();
        let up_weight = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down_tensor =
            Tensor::from_vec(vec![0.6f32, -0.25, 0.15, 0.5], (2usize, 2usize), &device)?;
        let down_id = down_tensor.id();
        let down_weight = CudaTrainTensor::parameter(down_tensor, down_id)?;
        let cos = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 0.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let sin = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 1.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let weights = CudaFullAttentionLayer {
            input_norm_weight: &input_norm,
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            o_weight: &o_weight,
            post_norm_weight: &post_norm,
            gate_weight: &gate_weight,
            up_weight: &up_weight,
            down_weight: &down_weight,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: true,
            rope: Some(CudaRopeTables {
                cos: &cos,
                sin: &sin,
                rotary_dim: 2,
            }),
        };
        let trainable = vec![
            q_weight.clone(),
            k_weight.clone(),
            v_weight.clone(),
            o_weight.clone(),
            gate_weight.clone(),
            up_weight.clone(),
            down_weight.clone(),
        ];
        let mut adamw = allocate_cuda_adamw_state(&trainable)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let q_before = q_weight.to_vec_f32()?;

        let loss = cuda_full_attention_sum_square_adamw_step_with_arena(
            &input,
            &weights,
            &trainable,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;

        assert!(loss.is_finite() && loss > 0.0);
        assert_ne!(q_weight.to_vec_f32()?, q_before);
        assert_eq!(adamw.get(&q_id).expect("q state").step, 1);
        assert_eq!(arena.allocation_count(), 3);
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

//! Vulkan-native training step.
//!
//! Self-contained training loop on `VkTensor` parameters: forward via
//! `vk_model_forward_loss`, backward via `vk_backward`, optimizer step
//! via the on-device AdamW kernel called directly with `VulkanBuffer`
//! handles (skipping the candle TensorId → registry indirection).
//!
//! For a full Qwen3.5 SFT run, the caller wires `VkModelWeights` from
//! their existing `GpuWeights` (one-time upload at training start) and
//! constructs `VkLoraLayer` per layer. The trainer then drives:
//!
//! ```text
//! for epoch:
//!   for batch (input_ids):
//!     loss = vk_model_forward_loss(weights, lora, ids)
//!     grads = vk_step_backward(loss)
//!     for (param_id, grad) in grads:
//!       lookup VkAdamWState by param_id
//!       dispatch_adamw_step_f32 in place
//! at end:
//!   for each lora pair: VkTensor.to_candle() → safetensors save
//! ```

use anyhow::{Context, Result};
use candle_core::TensorId;
use kiln_model::vk_forward::{
    vk_model_forward_loss, vk_step_backward, VkLoraLayer, VkLoraPair, VkModelWeights,
};
use kiln_vulkan_kernel::kernels::dispatch_adamw_step_f32;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanBuffer, VulkanDevice};
use std::collections::HashMap;
use std::sync::Arc;

/// Per-parameter AdamW state held entirely on the GPU.
pub struct VkAdamWState {
    pub m: Arc<VulkanBuffer>,
    pub v: Arc<VulkanBuffer>,
    pub n_elements: usize,
}

impl VkAdamWState {
    pub fn zeros_for(device: &Arc<VulkanDevice>, n_elements: usize) -> Result<Self> {
        let bytes = (n_elements * 4).max(4) as u64;
        let m = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes,
        )
        .context("VkAdamWState: alloc m")?;
        let v = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes,
        )
        .context("VkAdamWState: alloc v")?;
        // Zero them via the existing fill shader.
        let zero_bytes: Vec<u8> = vec![0u8; n_elements * 4];
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &m,
            &zero_bytes,
        )?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &v,
            &zero_bytes,
        )?;
        Ok(Self {
            m: Arc::new(m),
            v: Arc::new(v),
            n_elements,
        })
    }
}

/// All AdamW state for a model — one entry per trainable param.
pub type VkAdamWBook = HashMap<TensorId, VkAdamWState>;

pub fn allocate_adamw_state(
    device: &Arc<VulkanDevice>,
    lora_layers: &[VkLoraLayer],
) -> Result<VkAdamWBook> {
    let mut book = HashMap::new();
    for layer in lora_layers {
        for proj in [
            layer.q_proj.as_ref(),
            layer.k_proj.as_ref(),
            layer.v_proj.as_ref(),
            layer.o_proj.as_ref(),
            layer.gate_proj.as_ref(),
            layer.up_proj.as_ref(),
            layer.down_proj.as_ref(),
        ]
        .iter()
        .flatten()
        {
            book.insert(proj.a_id, VkAdamWState::zeros_for(device, proj.a.num_elements())?);
            book.insert(proj.b_id, VkAdamWState::zeros_for(device, proj.b.num_elements())?);
        }
    }
    Ok(book)
}

#[derive(Clone, Copy, Debug)]
pub struct VkAdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for VkAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

fn lora_pairs<'a>(layers: &'a [VkLoraLayer]) -> impl Iterator<Item = &'a VkLoraPair> + 'a {
    layers.iter().flat_map(|l| {
        [
            l.q_proj.as_ref(),
            l.k_proj.as_ref(),
            l.v_proj.as_ref(),
            l.o_proj.as_ref(),
            l.gate_proj.as_ref(),
            l.up_proj.as_ref(),
            l.down_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
    })
}

/// Run one training step end-to-end on the GPU.
///
/// Returns the scalar loss value.
pub fn vk_train_step(
    weights: &VkModelWeights,
    lora_layers: &[VkLoraLayer],
    input_ids: &[u32],
    adamw_state: &mut VkAdamWBook,
    cfg: &VkAdamWConfig,
    step: u32,
) -> Result<f32> {
    let loss = vk_model_forward_loss(weights, lora_layers, input_ids)?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = vk_step_backward(&loss)?;

    // Dispatch AdamW per parameter. We assume F32 storage; BF16
    // variant just swaps the kernel name.
    for pair in lora_pairs(lora_layers) {
        for (param, pid) in [(&pair.a, pair.a_id), (&pair.b, pair.b_id)] {
            let Some(grad) = grads.get(pid) else { continue };
            anyhow::ensure!(
                param.dtype() == VkDType::F32 && grad.dtype() == VkDType::F32,
                "vk_train_step: AdamW F32 only for Phase F (got {:?}/{:?})",
                param.dtype(),
                grad.dtype()
            );
            anyhow::ensure!(
                param.num_elements() == grad.num_elements(),
                "vk_train_step: param/grad element-count mismatch"
            );
            let state = adamw_state
                .get(&pid)
                .with_context(|| format!("missing AdamW state for param {:?}", pid))?;
            anyhow::ensure!(
                state.n_elements == param.num_elements(),
                "AdamW state size mismatch"
            );
            dispatch_adamw_step_f32(
                weights.embed_tokens.device(),
                param.buffer(),
                grad.buffer(),
                &state.m,
                &state.v,
                param.num_elements(),
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                step,
            )
            .context("dispatch_adamw_step_f32")?;
        }
    }

    Ok(loss_val)
}

/// Save LoRA adapter to safetensors via candle. Each VkTensor is read
/// back to CPU once.
pub fn save_vk_lora_adapter(
    lora_layers: &[VkLoraLayer],
    rank: usize,
    alpha: f32,
    output_path: &std::path::Path,
) -> Result<()> {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (li, layer) in lora_layers.iter().enumerate() {
        for (name, proj) in [
            ("q_proj", layer.q_proj.as_ref()),
            ("k_proj", layer.k_proj.as_ref()),
            ("v_proj", layer.v_proj.as_ref()),
            ("o_proj", layer.o_proj.as_ref()),
            ("gate_proj", layer.gate_proj.as_ref()),
            ("up_proj", layer.up_proj.as_ref()),
            ("down_proj", layer.down_proj.as_ref()),
        ] {
            let Some(p) = proj else { continue };
            let a_t = p.a.to_candle()?.to_device(&Device::Cpu)?;
            let b_t = p.b.to_candle()?.to_device(&Device::Cpu)?;
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_A.weight",
                    li, name
                ),
                a_t,
            );
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_B.weight",
                    li, name
                ),
                b_t,
            );
        }
    }
    candle_core::safetensors::save(&tensors, output_path)
        .with_context(|| format!("save_vk_lora_adapter: {}", output_path.display()))?;
    let _ = (rank, alpha); // adapter_config.json could be written here if desired
    Ok(())
}

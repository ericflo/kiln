//! Vulkan decode-weight cache and prewarm helpers.
//!
//! These helpers own kt `TensorId`-keyed f32/BF16-packed VulkanBuffer caches
//! and non-destructive decode-weight prewarming. The
//! runtime facade in `vulkan.rs` delegates here so operation dispatch remains
//! separate from explicit weight residency plumbing.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::DecodeWeightPrewarmPolicy;
use super::vulkan::VulkanBackend;
use crate::forward::{GpuAttentionWeights, GpuWeights};

const PREWARM_CANCELLATION_POLL: Duration = Duration::from_millis(25);

struct DecodeWeightPrewarmPacer<'a> {
    policy: &'a DecodeWeightPrewarmPolicy,
    started: Instant,
}

impl<'a> DecodeWeightPrewarmPacer<'a> {
    fn new(policy: &'a DecodeWeightPrewarmPolicy) -> Result<Self> {
        policy.ensure_active()?;
        Ok(Self {
            policy,
            started: Instant::now(),
        })
    }

    fn settle(&self, materialized_bytes: usize) -> Result<()> {
        self.policy.ensure_active()?;
        let Some(rate) = self.policy.max_bytes_per_second() else {
            return Ok(());
        };
        let target = prewarm_target_elapsed(materialized_bytes, rate);
        loop {
            self.policy.ensure_active()?;
            let elapsed = self.started.elapsed();
            if elapsed >= target {
                return Ok(());
            }
            let remaining = target.saturating_sub(elapsed);
            std::thread::sleep(remaining.min(PREWARM_CANCELLATION_POLL));
        }
    }
}

fn prewarm_target_elapsed(materialized_bytes: usize, bytes_per_second: u64) -> Duration {
    let target_nanos = (materialized_bytes as u128)
        .saturating_mul(1_000_000_000)
        .div_ceil(bytes_per_second as u128);
    Duration::from_nanos(u64::try_from(target_nanos).unwrap_or(u64::MAX))
}

/// kt-native f32 weight buffer cache: keys the buffer
/// cache on the **kt** `TensorId` (stable for the model's lifetime) and
/// extracts f32 bytes straight from kt storage on a miss - no candle
/// bridge, so a cache hit (every token after the first) does zero copy
/// work. (#1082)
pub(super) fn cached_f32_weight_buffer_kt(
    backend: &VulkanBackend,
    weight: &kiln_tensor::Tensor,
) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let vk_device = backend
        .vulkan_device
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let key = weight.id();
    {
        let cache = backend
            .weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan kt weight cache mutex poisoned"))?;
        if let Some(buffer) = cache.get(&key) {
            return Ok(Arc::clone(buffer));
        }
    }
    let weight_f32_data: Vec<f32> = weight
        .flatten_all()
        .context("kt weight flatten_all")?
        .to_dtype(kiln_tensor::DType::F32)
        .context("kt weight to f32")?
        .to_vec1::<f32>()
        .context("kt weight to_vec1 f32")?;
    let buffer = Arc::new(
        kiln_vulkan_kernel::kernels::upload_f32_buffer_from_slice(vk_device, &weight_f32_data)
            .context("upload kt f32 weight to Vulkan")?,
    );
    let mut cache = backend
        .weight_cache_kt
        .lock()
        .map_err(|_| anyhow::anyhow!("Vulkan kt weight cache mutex poisoned"))?;
    Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
}

/// kt-native bf16-packed weight buffer cache. Stable-kt-id keying;
/// extracts bf16 straight from kt storage on a miss. (#1082)
pub(super) fn cached_bf16_packed_weight_buffer_kt(
    backend: &VulkanBackend,
    weight: &kiln_tensor::Tensor,
) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let vk_device = backend
        .vulkan_device
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let key = weight.id();
    {
        let cache = backend
            .bf16_packed_weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan kt packed bf16 weight cache mutex poisoned"))?;
        if let Some(buffer) = cache.get(&key) {
            return Ok(Arc::clone(buffer));
        }
    }
    anyhow::ensure!(
        weight.dtype() == kiln_tensor::DType::BF16,
        "packed bf16 upload requires BF16 kt tensor, got {:?}",
        weight.dtype()
    );
    let weight_bf16_data: Vec<half::bf16> = weight
        .flatten_all()
        .context("kt bf16 weight flatten_all")?
        .to_vec1::<half::bf16>()
        .context("kt bf16 weight to_vec1")?;
    let buffer = Arc::new(
        kiln_vulkan_kernel::kernels::upload_bf16_packed_buffer_from_slice(
            vk_device,
            &weight_bf16_data,
        )
        .context("upload kt packed BF16 weight to Vulkan")?,
    );
    let mut cache = backend
        .bf16_packed_weight_cache_kt
        .lock()
        .map_err(|_| anyhow::anyhow!("Vulkan kt packed bf16 weight cache mutex poisoned"))?;
    Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
}

/// kt-native: whether to use the bf16-packed linear-weight decode path.
pub(super) fn use_bf16_packed_linear_weight_kt(
    backend: &VulkanBackend,
    weight: &kiln_tensor::Tensor,
) -> bool {
    backend.bf16_packed_linear_weights_enabled && weight.dtype() == kiln_tensor::DType::BF16
}

pub(super) fn use_bf16_packed_gdn_in_proj_weights_kt(
    backend: &VulkanBackend,
    weights: &[&kiln_tensor::Tensor],
) -> bool {
    backend.bf16_packed_gdn_in_proj_weights_enabled
        && weights
            .iter()
            .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
}

pub(super) fn use_bf16_packed_full_attn_qkv_weights_kt(
    backend: &VulkanBackend,
    weights: &[&kiln_tensor::Tensor],
) -> bool {
    backend.bf16_packed_full_attn_qkv_weights_enabled
        && weights
            .iter()
            .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
}

pub(super) fn use_bf16_packed_mlp_decode_weights_kt(
    backend: &VulkanBackend,
    weights: &[&kiln_tensor::Tensor],
) -> bool {
    backend.bf16_packed_mlp_decode_weights_enabled
        && weights
            .iter()
            .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
}

pub(super) fn prewarm_f32_weight_kt(
    backend: &VulkanBackend,
    name: &str,
    weight: &kiln_tensor::Tensor,
    count: &mut usize,
    bytes: &mut usize,
) -> Result<()> {
    backend
        .cached_f32_weight_buffer_kt(weight)
        .with_context(|| format!("prewarm Vulkan decode weight {name}"))?;
    *count += 1;
    *bytes += weight.elem_count() * std::mem::size_of::<f32>();
    Ok(())
}

pub(super) fn prewarm_bf16_packed_weight_kt(
    backend: &VulkanBackend,
    name: &str,
    weight: &kiln_tensor::Tensor,
    count: &mut usize,
    bytes: &mut usize,
) -> Result<()> {
    backend
        .cached_bf16_packed_weight_buffer_kt(weight)
        .with_context(|| format!("prewarm Vulkan packed BF16 decode weight {name}"))?;
    *count += 1;
    *bytes += weight.elem_count().div_ceil(2) * std::mem::size_of::<u32>();
    Ok(())
}

pub(super) fn prewarm_linear_weight_kt(
    backend: &VulkanBackend,
    name: &str,
    weight: &kiln_tensor::Tensor,
    f32_count: &mut usize,
    f32_bytes: &mut usize,
    bf16_count: &mut usize,
    bf16_bytes: &mut usize,
) -> Result<()> {
    if use_bf16_packed_linear_weight_kt(backend, weight) {
        prewarm_bf16_packed_weight_kt(backend, name, weight, bf16_count, bf16_bytes)
    } else {
        prewarm_f32_weight_kt(backend, name, weight, f32_count, f32_bytes)
    }
}

pub(super) fn prewarm_gdn_in_proj_weight_kt(
    backend: &VulkanBackend,
    name: &str,
    weight: &kiln_tensor::Tensor,
    f32_count: &mut usize,
    f32_bytes: &mut usize,
    bf16_count: &mut usize,
    bf16_bytes: &mut usize,
) -> Result<()> {
    if use_bf16_packed_gdn_in_proj_weights_kt(backend, &[weight]) {
        prewarm_bf16_packed_weight_kt(backend, name, weight, bf16_count, bf16_bytes)
    } else {
        prewarm_f32_weight_kt(backend, name, weight, f32_count, f32_bytes)
    }
}

// Flat prewarm ABI: one tensor per weight plus the f32/bf16 count/byte accumulators.
#[allow(clippy::too_many_arguments)]
pub(super) fn prewarm_full_attn_qkv_weights_kt(
    backend: &VulkanBackend,
    layer_idx: usize,
    q_weight_t: &kiln_tensor::Tensor,
    k_weight_t: &kiln_tensor::Tensor,
    v_weight_t: &kiln_tensor::Tensor,
    f32_count: &mut usize,
    f32_bytes: &mut usize,
    bf16_count: &mut usize,
    bf16_bytes: &mut usize,
) -> Result<()> {
    let weights = [
        ("q_proj_t", q_weight_t),
        ("k_proj_t", k_weight_t),
        ("v_proj_t", v_weight_t),
    ];
    if use_bf16_packed_full_attn_qkv_weights_kt(backend, &[q_weight_t, k_weight_t, v_weight_t]) {
        for (suffix, weight) in weights {
            prewarm_bf16_packed_weight_kt(
                backend,
                &format!("layers.{layer_idx}.attention.{suffix}"),
                weight,
                bf16_count,
                bf16_bytes,
            )?;
        }
    } else {
        for (suffix, weight) in weights {
            prewarm_f32_weight_kt(
                backend,
                &format!("layers.{layer_idx}.attention.{suffix}"),
                weight,
                f32_count,
                f32_bytes,
            )?;
        }
    }
    Ok(())
}

// Flat prewarm ABI: one tensor per weight plus the f32/bf16 count/byte accumulators.
#[allow(clippy::too_many_arguments)]
pub(super) fn prewarm_mlp_decode_weights_kt(
    backend: &VulkanBackend,
    layer_idx: usize,
    gate_weight_t: &kiln_tensor::Tensor,
    up_weight_t: &kiln_tensor::Tensor,
    down_weight_t: &kiln_tensor::Tensor,
    f32_count: &mut usize,
    f32_bytes: &mut usize,
    bf16_count: &mut usize,
    bf16_bytes: &mut usize,
) -> Result<()> {
    let weights = [
        ("gate_proj_t", gate_weight_t),
        ("up_proj_t", up_weight_t),
        ("down_proj_t", down_weight_t),
    ];
    if use_bf16_packed_mlp_decode_weights_kt(backend, &[gate_weight_t, up_weight_t, down_weight_t])
    {
        for (suffix, weight) in weights {
            prewarm_bf16_packed_weight_kt(
                backend,
                &format!("layers.{layer_idx}.mlp.{suffix}"),
                weight,
                bf16_count,
                bf16_bytes,
            )?;
        }
        if backend.mlp_bf16_gate_up_f32_down_enabled {
            prewarm_f32_weight_kt(
                backend,
                &format!("layers.{layer_idx}.mlp.down_proj_t"),
                down_weight_t,
                f32_count,
                f32_bytes,
            )?;
        }
    } else {
        for (suffix, weight) in weights {
            prewarm_f32_weight_kt(
                backend,
                &format!("layers.{layer_idx}.mlp.{suffix}"),
                weight,
                f32_count,
                f32_bytes,
            )?;
        }
    }
    Ok(())
}

pub(super) fn prewarm_decode_weights(
    backend: &VulkanBackend,
    weights: &GpuWeights,
    policy: &DecodeWeightPrewarmPolicy,
) -> Result<()> {
    if !backend.has_vulkan() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    let pacer = DecodeWeightPrewarmPacer::new(policy)?;
    let mut count = 0usize;
    let mut bytes = 0usize;
    let mut bf16_packed_count = 0usize;
    let mut bf16_packed_bytes = 0usize;

    prewarm_linear_weight_kt(
        backend,
        "embed_tokens_t",
        &weights.embed_tokens_t,
        &mut count,
        &mut bytes,
        &mut bf16_packed_count,
        &mut bf16_packed_bytes,
    )?;
    pacer.settle(bytes.saturating_add(bf16_packed_bytes))?;

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        match &layer.attention {
            GpuAttentionWeights::Full(attn) => {
                prewarm_full_attn_qkv_weights_kt(
                    backend,
                    layer_idx,
                    &attn.q_proj_t,
                    &attn.k_proj_t,
                    &attn.v_proj_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
                prewarm_linear_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.o_proj_t"),
                    &attn.o_proj_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
            }
            GpuAttentionWeights::Linear(attn) => {
                prewarm_gdn_in_proj_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.in_proj_qkv_t"),
                    &attn.in_proj_qkv_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
                prewarm_gdn_in_proj_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.in_proj_z_t"),
                    &attn.in_proj_z_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
                prewarm_gdn_in_proj_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.in_proj_a_t"),
                    &attn.in_proj_a_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
                prewarm_gdn_in_proj_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.in_proj_b_t"),
                    &attn.in_proj_b_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
                prewarm_linear_weight_kt(
                    backend,
                    &format!("layers.{layer_idx}.attention.out_proj_t"),
                    &attn.out_proj_t,
                    &mut count,
                    &mut bytes,
                    &mut bf16_packed_count,
                    &mut bf16_packed_bytes,
                )?;
            }
        }

        prewarm_mlp_decode_weights_kt(
            backend,
            layer_idx,
            &layer.mlp.gate_proj_t,
            &layer.mlp.up_proj_t,
            &layer.mlp.down_proj_t,
            &mut count,
            &mut bytes,
            &mut bf16_packed_count,
            &mut bf16_packed_bytes,
        )?;
        pacer.settle(bytes.saturating_add(bf16_packed_bytes))?;
    }

    tracing::info!(
        weights = count,
        f32_cache_mb = bytes / (1024 * 1024),
        bf16_packed_weights = bf16_packed_count,
        bf16_packed_cache_mb = bf16_packed_bytes / (1024 * 1024),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Vulkan decode weight cache prewarmed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LinearBackend;
    use crate::forward::{
        GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights,
    };
    use kiln_tensor::{DType, Device, Tensor};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn prewarm_pacer_converts_materialized_bytes_to_exact_elapsed_budget() {
        assert_eq!(
            prewarm_target_elapsed(256 * 1024 * 1024, 512 * 1024 * 1024),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn prewarm_pacer_rejects_work_after_shutdown() -> Result<()> {
        let cancellation = Arc::new(AtomicBool::new(true));
        let policy = DecodeWeightPrewarmPolicy::paced(1024, cancellation)?;
        let error = DecodeWeightPrewarmPacer::new(&policy)
            .err()
            .expect("cancelled policy must reject prewarm");
        assert!(
            error
                .downcast_ref::<super::super::DecodeWeightPrewarmCancelled>()
                .is_some()
        );
        Ok(())
    }

    fn bf16_tensor(values: Vec<f32>, shape: Vec<usize>, device: Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(values, shape)?
            .to_dtype(DType::BF16)?
            .to_device(device)?)
    }

    fn patterned_bf16(shape: &[usize], seed: usize, device: Device) -> Result<Tensor> {
        let values = (0..shape.iter().product())
            .map(|index| (((index + seed) % 17) as f32 - 8.0) * 0.03125)
            .collect();
        bf16_tensor(values, shape.to_vec(), device)
    }

    fn prewarm_fixture(device: Device) -> Result<GpuWeights> {
        let hidden = 4usize;
        let intermediate = 5usize;
        let vocab = 6usize;
        let q_proj_t = bf16_tensor(
            vec![
                0.5, -0.25, 0.125, -0.5, 0.75, 0.25, 1.0, -0.125, 0.375, -0.75, 0.625, 0.5,
            ],
            vec![hidden, 3],
            device,
        )?;
        let layer = GpuLayerWeights {
            input_layernorm: patterned_bf16(&[hidden], 1, device)?,
            post_attention_layernorm: patterned_bf16(&[hidden], 2, device)?,
            attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj: patterned_bf16(&[3, hidden], 3, device)?,
                k_proj: patterned_bf16(&[2, hidden], 4, device)?,
                v_proj: patterned_bf16(&[2, hidden], 5, device)?,
                o_proj: patterned_bf16(&[hidden, 3], 6, device)?,
                q_norm: patterned_bf16(&[2], 7, device)?,
                k_norm: patterned_bf16(&[2], 8, device)?,
                q_proj_t,
                k_proj_t: patterned_bf16(&[hidden, 2], 9, device)?,
                v_proj_t: patterned_bf16(&[hidden, 2], 10, device)?,
                qkv_proj_t: None,
                o_proj_t: patterned_bf16(&[3, hidden], 11, device)?,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj: patterned_bf16(&[intermediate, hidden], 12, device)?,
                up_proj: patterned_bf16(&[intermediate, hidden], 13, device)?,
                down_proj: patterned_bf16(&[hidden, intermediate], 14, device)?,
                gate_proj_t: patterned_bf16(&[hidden, intermediate], 15, device)?,
                up_proj_t: patterned_bf16(&[hidden, intermediate], 16, device)?,
                down_proj_t: patterned_bf16(&[intermediate, hidden], 17, device)?,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        };
        Ok(GpuWeights {
            source_content_sha256: None,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            embed_tokens: patterned_bf16(&[vocab, hidden], 18, device)?,
            embed_tokens_t: patterned_bf16(&[hidden, vocab], 19, device)?,
            lm_head_w8: None,
            layers: vec![layer],
            final_norm: patterned_bf16(&[hidden], 20, device)?,
            rotary_inv_freq: Tensor::from_vec(vec![1.0_f32, 0.01], 2)?.to_device(device)?,
            mtp: None,
        })
    }

    fn assert_prewarm_retains_projection(
        backend: &VulkanBackend,
        weight_device: Device,
    ) -> Result<()> {
        let weights = prewarm_fixture(weight_device)?;
        let q_proj_t = match &weights.layers[0].attention {
            GpuAttentionWeights::Full(attention) => &attention.q_proj_t,
            GpuAttentionWeights::Linear(_) => unreachable!(),
        };
        let before_id = q_proj_t.id();
        let before_bytes = q_proj_t.storage().byte_len();
        let before_values = q_proj_t
            .to_device(Device::Cpu)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;

        LinearBackend::runtime_prewarm_decode_weights(backend, &weights)?;

        let q_proj_t = match &weights.layers[0].attention {
            GpuAttentionWeights::Full(attention) => &attention.q_proj_t,
            GpuAttentionWeights::Linear(_) => unreachable!(),
        };
        assert_eq!(q_proj_t.id(), before_id);
        assert_eq!(q_proj_t.device(), weight_device);
        assert_eq!(q_proj_t.storage().byte_len(), before_bytes);
        assert!(q_proj_t.is_contiguous());
        assert_eq!(
            q_proj_t
                .to_device(Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::bf16>()?,
            before_values
        );
        assert!(
            backend
                .bf16_packed_weight_cache_kt
                .lock()
                .map_err(|_| anyhow::anyhow!("packed cache mutex poisoned"))?
                .contains_key(&before_id)
        );

        let x = Tensor::from_vec(
            vec![0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
            (1, 2, 4),
        )?
        .to_device(weight_device)?;
        let output = LinearBackend::runtime_linear_decode(backend, &x, q_proj_t)?
            .context("prewarmed projection should remain executable")?;
        assert_eq!(output.device(), weight_device);
        assert_eq!(
            output
                .to_device(Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![0.375, 0.09375, 0.6875, -0.375, -0.09375, -0.6875]
        );
        Ok(())
    }

    #[test]
    fn default_prewarm_retains_authoritative_serving_and_resident_weights() -> Result<()> {
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            return Ok(());
        }
        assert!(
            crate::backend::vulkan::vulkan_is_available(),
            "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
        );
        let backend = VulkanBackend::new(Device::Cpu);
        // Serving keeps model weights on CPU and uses the backend-private
        // packed cache. Training may keep the same authoritative tensors
        // resident. Prewarm must preserve both representations exactly.
        assert_prewarm_retains_projection(&backend, Device::Cpu)?;
        assert_prewarm_retains_projection(&backend, Device::Vulkan(0))?;
        Ok(())
    }
}

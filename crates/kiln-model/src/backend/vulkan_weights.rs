//! Vulkan decode-weight cache and prewarm helpers.
//!
//! These helpers own kt `TensorId`-keyed f32/BF16-packed VulkanBuffer caches,
//! decode-weight prewarming, and post-upload BF16 host-storage stubbing. The
//! runtime facade in `vulkan.rs` delegates here so operation dispatch remains
//! separate from explicit weight residency plumbing.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::vulkan::VulkanBackend;
use crate::forward::{GpuAttentionWeights, GpuWeights};

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
        for (suffix, weight) in weights {
            prewarm_f32_weight_kt(
                backend,
                &format!("layers.{layer_idx}.mlp.{suffix}"),
                weight,
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

pub(super) fn prewarm_decode_weights(backend: &VulkanBackend, weights: &GpuWeights) -> Result<()> {
    if !backend.has_vulkan() || !backend.weight_prewarm_enabled {
        return Ok(());
    }

    let start = std::time::Instant::now();
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

/// Phase 4.x residency: drop the CPU storage of every
/// pre-transposed weight cache (`*_proj_t`, `embed_tokens_t`)
/// whose BF16-packed bytes are already resident in
/// `bf16_packed_weight_cache_kt`. Replace each with a
/// 1-element BF16 stub and re-key the cache so subsequent
/// lookups against the new kt `TensorId` still find the same
/// `Arc<VulkanBuffer>`.
///
/// Saves ~6-7 GB peak RSS on Qwen3.5-4B training at T=918 - the
/// transposed-cache copies are the dominant remaining
/// CPU-side residency item documented in
/// `docs/audits/candle_cpu_residency_2026-05-11.md`.
///
/// Safe because:
/// - The bf16-packed Vulkan code paths read the weight via the
///   `Arc<VulkanBuffer>` looked up in `bf16_packed_weight_cache_kt`.
///   They never re-read the CPU storage of the source tensor
///   after the buffer is cached.
/// - `VulkanLinearOp::bwd` for BF16 weights routes through the
///   transposed Vulkan kernel (also buffer-backed). The F32
///   fallback bwd path that *does* read the source weight tensor cannot
///   fire for BF16 weights.
/// - Non-BF16 tensors and tensors not in the cache are skipped.
pub(super) fn drop_uploaded_bf16_weights(
    backend: &VulkanBackend,
    weights: &mut crate::forward::GpuWeights,
    device: &kiln_tensor::Device,
) -> Result<usize> {
    if !backend.has_vulkan() {
        return Ok(0);
    }
    // Broadcast-base for cheap shape-preserving stubs. Source has
    // 2 bytes of storage; broadcast_as(target_shape) creates views
    // with stride [0, 0] sharing the same backing storage. Each per-
    // weight stub costs ~24 bytes of metadata (Layout + Tensor
    // struct), not `hidden * out_dim * 2` bytes. The weights are
    // kt-typed (#1082 forward-flip), and the Vulkan buffer cache is
    // re-keyed directly from the old kt TensorId to the stub's kt
    // TensorId.
    let broadcast_base = kiln_tensor::Tensor::zeros(
        (1usize, 1usize),
        kiln_tensor::DType::BF16,
        kiln_tensor::Device::Cpu,
    )
    .context("drop_uploaded_bf16_weights: create broadcast base")?;
    let _ = device;
    let mut bf16_cache = backend
        .bf16_packed_weight_cache_kt
        .lock()
        .map_err(|_| anyhow::anyhow!("bf16 weight cache mutex poisoned"))?;
    let mut f32_cache = backend
        .weight_cache_kt
        .lock()
        .map_err(|_| anyhow::anyhow!("f32 weight cache mutex poisoned"))?;

    // Per-tensor replacement closure. Returns true if the tensor
    // was stubbed (was BF16, rank-2, and in the cache).
    //
    // - Reads the original `[hidden, out_dim]` shape from `t.dims()`
    //   *before* replacement.
    // - Creates a shape-preserving stub by broadcasting the
    //   2-byte base to that shape (so downstream `weight_t.dims2()`
    //   reads continue to return the right shape, but the storage
    //   bytes drop to ~zero).
    // - Re-keys the packed cache and any F32 shadow cache entry so
    //   subsequent kt-native lookups by the stub's new TensorId still find
    //   the original `Arc<VulkanBuffer>`s.
    fn replace(
        t: &mut kiln_tensor::Tensor,
        bf16_cache: &mut std::collections::HashMap<
            kiln_tensor::TensorId,
            Arc<kiln_vulkan_kernel::VulkanBuffer>,
        >,
        f32_cache: &mut std::collections::HashMap<
            kiln_tensor::TensorId,
            Arc<kiln_vulkan_kernel::VulkanBuffer>,
        >,
        broadcast_base: &kiln_tensor::Tensor,
    ) -> bool {
        if t.dtype() != kiln_tensor::DType::BF16 {
            return false;
        }
        let dims = t.dims();
        if dims.len() != 2 {
            return false; // Only rank-2 transposed-cache tensors are stubbable.
        }
        let (d0, d1) = (dims[0], dims[1]);
        let old_id = t.id();
        let Some(bf16_buf) = bf16_cache.remove(&old_id) else {
            return false;
        };
        let f32_buf = f32_cache.remove(&old_id);
        let Ok(new_stub) = broadcast_base.broadcast_as((d0, d1)) else {
            bf16_cache.insert(old_id, bf16_buf); // restore on failure
            if let Some(buf) = f32_buf {
                f32_cache.insert(old_id, buf);
            }
            return false;
        };
        let new_id = new_stub.id();
        *t = new_stub;
        bf16_cache.insert(new_id, bf16_buf);
        if let Some(buf) = f32_buf {
            f32_cache.insert(new_id, buf);
        }
        true
    }

    let mut stubbed = 0usize;

    // Intentionally NOT stubbing `weights.embed_tokens_t`:
    // `embedding_lookup_from_transposed_index` calls
    // `embed_tokens_t.index_select(idx, 1)` which reads the
    // tensor's data (not just shape), so a 1-element stub would
    // make the embedding lookup return garbage. The other `*_proj_t`
    // caches go through the kt TensorId -> Arc<VulkanBuffer> packed cache,
    // so they only need shape/dtype metadata locally. Embedding savings
    // (~750 MB) are small
    // next to the per-layer transposes (~5-6 GB across 32 layers).

    // Per-layer attention + MLP transposes.
    for layer in weights.layers.iter_mut() {
        match &mut layer.attention {
            crate::forward::GpuAttentionWeights::Full(attn) => {
                for t in [
                    &mut attn.q_proj_t,
                    &mut attn.k_proj_t,
                    &mut attn.v_proj_t,
                    &mut attn.o_proj_t,
                ] {
                    if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                        stubbed += 1;
                    }
                }
                if let Some(qkv_t) = attn.qkv_proj_t.as_mut() {
                    if replace(qkv_t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                        stubbed += 1;
                    }
                }
            }
            crate::forward::GpuAttentionWeights::Linear(attn) => {
                for t in [
                    &mut attn.in_proj_qkv_t,
                    &mut attn.in_proj_z_t,
                    &mut attn.in_proj_a_t,
                    &mut attn.in_proj_b_t,
                    &mut attn.out_proj_t,
                ] {
                    if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                        stubbed += 1;
                    }
                }
                if let Some(ab_t) = attn.in_proj_ab_t.as_mut() {
                    if replace(ab_t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                        stubbed += 1;
                    }
                }
            }
        }
        for t in [
            &mut layer.mlp.gate_proj_t,
            &mut layer.mlp.up_proj_t,
            &mut layer.mlp.down_proj_t,
        ] {
            if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                stubbed += 1;
            }
        }
    }

    tracing::info!(
        stubbed,
        "dropped CPU storage of pre-transposed bf16 weight caches"
    );
    Ok(stubbed)
}

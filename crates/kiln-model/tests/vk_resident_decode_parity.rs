//! Gate (d) of `docs/vk_resident_decode_plan.md`: end-to-end parity
//! between the Vulkan-resident decode path
//! ([`model_forward_paged_last_token_resident`]) and the legacy
//! [`model_forward_paged_last_token`] on Qwen3.5-4B.
//!
//! Each path is run on the same prompt + KV-cache state; the resident
//! logits must be within `≤ 1e-4` relative error of the non-resident
//! logits.
//!
//! Activation: gated on `KILN_RESIDENT_DECODE_PARITY_MODEL`, which
//! must point at a Qwen3.5-4B checkpoint directory. Without the env
//! var the test is skipped silently, so workspace `cargo test` on a
//! host without the model still passes. The non-Vulkan workspace
//! build skips at the `cfg(feature = "vulkan")` gate.
//!
//! The resident entry point is a strict superset of the legacy fn
//! today: when the per-layer resident wiring lands, this test gates
//! correctness; when the entry point is still delegating to the
//! legacy fn, the test verifies the delegation contract stays
//! bit-identical.

#![cfg(feature = "vulkan")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_model::backend;
use kiln_model::forward::{
    GpuWeights, model_forward_paged_last_token, model_forward_paged_last_token_resident,
};
use kiln_model::paged_kv_cache::PagedKvCache;
use kiln_model::{LoadModelOptions, load_model_with_options};

const MODEL_ENV: &str = "KILN_RESIDENT_DECODE_PARITY_MODEL";

#[test]
fn vk_resident_decode_matches_nonresident_on_qwen35_4b() {
    let Some(model_dir) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        eprintln!(
            "[vk_resident_decode_parity] skipped — set {MODEL_ENV}=/path/to/Qwen3.5-4B to enable"
        );
        return;
    };
    run(&model_dir).expect("vk-resident decode parity failed");
}

fn run(model_dir: &std::path::Path) -> Result<()> {
    let config = ModelConfig::qwen3_5_4b();
    let opts = LoadModelOptions { load_mtp: false };
    let model_weights = load_model_with_options(model_dir, &config, opts)?;

    let device = Device::Cpu;
    let runtime = backend::for_device(&device);

    if !runtime.supports_resident_decode() {
        eprintln!(
            "[vk_resident_decode_parity] skipped — Backend::supports_resident_decode() is false; \
             this build doesn't include the Vulkan backend"
        );
        return Ok(());
    }
    if !runtime.decode_resident_pool_ready(config.hidden_size, config.intermediate_size, 64) {
        eprintln!(
            "[vk_resident_decode_parity] skipped — decode_resident_pool_ready returned false; \
             not enough device-local memory for the resident ring"
        );
        return Ok(());
    }

    let weights = GpuWeights::from_model_weights(&model_weights, &config, &device)
        .context("transfer weights to backend")?;
    drop(model_weights);
    runtime.prewarm_decode_weights(&weights)?;

    // Small prompt + a single decode step. The cache is sized for one short
    // sequence: 8 tokens of prompt + 1 decode token + headroom = 4 blocks of
    // 16 each = 64 slots, comfortably within the 16-block test pool.
    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let block_size = 16usize;
    let num_blocks = 16usize;
    let cache_legacy = build_cache(&config, &device, num_blocks, block_size)?;
    let cache_resident = build_cache(&config, &device, num_blocks, block_size)?;
    let mut bt_legacy = BlockTable::new();
    let mut bt_resident = BlockTable::new();
    // Allocate the same two blocks (0, 1) to each table — the caches are
    // independent so block IDs don't conflict.
    bt_legacy.push(0);
    bt_legacy.push(1);
    bt_resident.push(0);
    bt_resident.push(1);

    // Prefill on both caches identically — same KV state going into the
    // single decode step we actually compare.
    let mut lin_legacy = kiln_model::forward::LinearAttentionState::new_for_inference(&config, &device)?;
    let mut lin_resident = kiln_model::forward::LinearAttentionState::new_for_inference(&config, &device)?;
    let _ = model_forward_paged_last_token(
        runtime.as_ref(),
        &prompt_tokens,
        &weights,
        &config,
        &cache_legacy,
        &bt_legacy,
        0,
        Some(&mut lin_legacy),
        None,
        None,
    )
    .context("legacy prefill")?;
    let _ = model_forward_paged_last_token(
        runtime.as_ref(),
        &prompt_tokens,
        &weights,
        &config,
        &cache_resident,
        &bt_resident,
        0,
        Some(&mut lin_resident),
        None,
        None,
    )
    .context("resident-prefill setup")?;

    // One additional decode step on each path. We pick a fixed next-token
    // id (100) so legacy and resident see the same input.
    let next_token: [u32; 1] = [100];
    let legacy_logits = model_forward_paged_last_token(
        runtime.as_ref(),
        &next_token,
        &weights,
        &config,
        &cache_legacy,
        &bt_legacy,
        prompt_tokens.len(),
        Some(&mut lin_legacy),
        None,
        None,
    )
    .context("legacy decode step")?;
    let resident_logits = model_forward_paged_last_token_resident(
        runtime.as_ref(),
        &next_token,
        &weights,
        &config,
        &cache_resident,
        &bt_resident,
        prompt_tokens.len(),
        Some(&mut lin_resident),
        None,
        None,
    )
    .context("resident decode step")?;

    let legacy_v: Vec<f32> = legacy_logits.flatten_all()?.to_vec1()?;
    let resident_v: Vec<f32> = resident_logits.flatten_all()?.to_vec1()?;
    assert_eq!(
        legacy_v.len(),
        resident_v.len(),
        "logits length mismatch: legacy={} resident={}",
        legacy_v.len(),
        resident_v.len()
    );

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (a, b)) in legacy_v.iter().zip(resident_v.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(b.abs()).max(1e-6f32);
        if rel > max_rel {
            max_rel = rel;
            max_abs = abs;
            worst_idx = i;
        }
    }
    eprintln!(
        "[vk_resident_decode_parity] worst diff @ {worst_idx}: abs={max_abs:e} rel={max_rel:e} \
         (legacy={} resident={})",
        legacy_v[worst_idx], resident_v[worst_idx],
    );
    assert!(
        max_rel <= 1e-4,
        "vk-resident logits diverge: max relative error {max_rel:e} > 1e-4 at index {worst_idx}"
    );
    Ok(())
}

fn build_cache(
    config: &ModelConfig,
    device: &Device,
    num_blocks: usize,
    block_size: usize,
) -> Result<PagedKvCache> {
    // Map kiln_core::config::DType → candle_core::DType.
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };
    PagedKvCache::new(
        config.num_full_attention_layers,
        num_blocks,
        block_size,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
    )
}

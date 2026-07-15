//! Gate (d) of `docs/vk_resident_decode_plan.md`: end-to-end parity
//! between the Vulkan-resident decode path
//! ([`model_forward_paged_last_token_resident`]) and the legacy
//! [`model_forward_paged_last_token`] on Qwen3.5-4B.
//!
//! Each path is run on the same prompt + KV-cache state; the resident
//! logits must be within `≤ 1e-4` relative error of the non-resident
//! logits.
//!
//! Activation: gated on `KILN_QUALIFICATION_MODEL_PATH`, which
//! must point at a Qwen3.5-4B checkpoint directory. Normal developer
//! runs without the model skip with a diagnostic; `KILN_QUALIFICATION=1`
//! makes a missing model, runtime, or resident pool fail closed. The
//! non-Vulkan workspace build skips at the `cfg(feature = "vulkan")` gate.
//!
//! The resident entry point is a strict superset of the non-resident fn
//! today: when the per-layer resident wiring lands, this test gates
//! correctness; when the entry point is still delegating to the
//! non-resident fn, the test verifies the delegation contract stays
//! bit-identical.

#![cfg(feature = "vulkan")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_model::backend::{self, LinearBackend, ReplayBackend};
use kiln_model::forward::{
    GpuWeights, model_forward_paged_last_token, model_forward_paged_last_token_resident,
};
use kiln_model::{LoadModelOptions, PagedKvCacheKt as PagedKvCache, load_model_with_options};
use kiln_tensor::{DType, Device};

const MODEL_ENV: &str = "KILN_QUALIFICATION_MODEL_PATH";
const HF_LOGITS_ENV: &str = "KILN_QUALIFICATION_HF_LOGITS_PATH";
const HF_ORACLE_SCHEMA: &str = "kiln.qwen35-hf-full-logits.v1";
const HF_INPUT_TOKEN_IDS: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 100];

#[test]
fn vk_resident_decode_matches_nonresident_on_qwen35_4b() {
    let Some(model_dir) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        if qualification_required() {
            panic!("{MODEL_ENV} is required while KILN_QUALIFICATION=1");
        }
        eprintln!(
            "[vk_resident_decode_parity] skipped - set {MODEL_ENV}=/path/to/Qwen3.5-4B to enable"
        );
        return;
    };
    run(&model_dir).expect("vk-resident decode parity failed");
}

fn qualification_required() -> bool {
    std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1")
}

fn run(model_dir: &std::path::Path) -> Result<()> {
    let config = ModelConfig::qwen3_5_4b();
    let opts = LoadModelOptions { load_mtp: false };
    let model_weights = load_model_with_options(model_dir, &config, opts)?;

    let device = Device::Cpu;
    let runtime = backend::for_device_kt(&device);

    if !ReplayBackend::runtime_supports_resident_decode(runtime.as_ref()) {
        if qualification_required() {
            anyhow::bail!("Vulkan resident decode runtime unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!(
            "[vk_resident_decode_parity] skipped - ReplayBackend::runtime_supports_resident_decode() is false; \
             this build doesn't include the Vulkan backend"
        );
        return Ok(());
    }
    if !ReplayBackend::runtime_decode_resident_pool_ready(
        runtime.as_ref(),
        config.hidden_size,
        config.intermediate_size,
        64,
    ) {
        if qualification_required() {
            anyhow::bail!("Vulkan resident decode pool unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!(
            "[vk_resident_decode_parity] skipped - decode_resident_pool_ready returned false; \
             not enough device-local memory for the resident ring"
        );
        return Ok(());
    }

    let weights = GpuWeights::from_model_weights(&model_weights, &config, &device)
        .context("transfer weights to backend")?;
    drop(model_weights);
    LinearBackend::runtime_prewarm_decode_weights(runtime.as_ref(), &weights)?;

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
    let mut lin_legacy =
        kiln_model::forward::LinearAttentionState::new_for_inference(&config, &device)?;
    let mut lin_resident =
        kiln_model::forward::LinearAttentionState::new_for_inference(&config, &device)?;
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
    eprintln!(
        "KILN_VULKAN_RESIDENT_LOGIT_PARITY_PASS max_abs={max_abs:e} max_rel={max_rel:e} vocab={}",
        legacy_v.len()
    );
    if let Some(reference_path) = std::env::var_os(HF_LOGITS_ENV).map(PathBuf::from) {
        compare_hf_full_logits(&reference_path, &resident_v)?;
    }
    Ok(())
}

fn compare_hf_full_logits(reference_path: &std::path::Path, kiln_logits: &[f32]) -> Result<()> {
    let data = std::fs::read(reference_path)
        .with_context(|| format!("read HF reference {}", reference_path.display()))?;
    let (_, metadata) =
        safetensors::SafeTensors::read_metadata(&data).context("parse HF reference metadata")?;
    let user_metadata = metadata
        .metadata()
        .as_ref()
        .context("HF reference has no user metadata")?;
    anyhow::ensure!(
        user_metadata.get("schema").map(String::as_str) == Some(HF_ORACLE_SCHEMA),
        "HF reference schema is not {HF_ORACLE_SCHEMA}"
    );
    anyhow::ensure!(
        user_metadata
            .get("linear_attention_implementation")
            .map(String::as_str)
            == Some("transformers_torch_fallback"),
        "HF reference did not use the pinned Transformers torch fallback"
    );
    anyhow::ensure!(
        user_metadata
            .get("attention_implementation")
            .map(String::as_str)
            == Some("eager"),
        "HF reference did not use eager full attention"
    );

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .context("deserialize HF reference safetensors")?;
    anyhow::ensure!(
        tensors.names().len() == 2
            && tensors.names().contains(&"input_ids")
            && tensors.names().contains(&"logits"),
        "HF reference must contain exactly input_ids and logits"
    );
    let input_ids = tensors.tensor("input_ids").context("HF input_ids tensor")?;
    anyhow::ensure!(
        input_ids.dtype() == safetensors::Dtype::I64,
        "HF input_ids must be I64"
    );
    anyhow::ensure!(
        input_ids.shape() == [1, HF_INPUT_TOKEN_IDS.len()],
        "HF input_ids shape mismatch: {:?}",
        input_ids.shape()
    );
    let input_values: Vec<u32> = input_ids
        .data()
        .chunks_exact(8)
        .map(|bytes| {
            let value = i64::from_le_bytes(bytes.try_into().expect("exact I64 chunk"));
            u32::try_from(value).context("HF input token does not fit u32")
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(
        input_values == HF_INPUT_TOKEN_IDS,
        "HF reference input IDs do not match the Vulkan test: {input_values:?}"
    );

    let logits = tensors.tensor("logits").context("HF logits tensor")?;
    anyhow::ensure!(
        logits.dtype() == safetensors::Dtype::F32,
        "HF logits must be F32"
    );
    anyhow::ensure!(
        logits.shape() == [kiln_logits.len()],
        "HF logits shape {:?} does not match Kiln vocab {}",
        logits.shape(),
        kiln_logits.len()
    );
    let hf_logits: Vec<f32> = logits
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact F32 chunk")))
        .collect();
    anyhow::ensure!(
        hf_logits.iter().all(|value| value.is_finite())
            && kiln_logits.iter().all(|value| value.is_finite()),
        "HF or Kiln logits contain non-finite values"
    );

    let mut max_abs = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut hf_norm = 0.0_f64;
    let mut kiln_norm = 0.0_f64;
    for (&hf, &kiln) in hf_logits.iter().zip(kiln_logits) {
        let hf = f64::from(hf);
        let kiln = f64::from(kiln);
        let abs = (hf - kiln).abs();
        max_abs = max_abs.max(abs);
        abs_sum += abs;
        dot += hf * kiln;
        hf_norm += hf * hf;
        kiln_norm += kiln * kiln;
    }
    let mean_abs = abs_sum / hf_logits.len() as f64;
    let cosine = dot / (hf_norm.sqrt() * kiln_norm.sqrt());
    let hf_argmax = argmax(&hf_logits);
    let kiln_argmax = argmax(kiln_logits);
    let argmax_equal = usize::from(hf_argmax == kiln_argmax);
    let hf_top10 = top_k(&hf_logits, 10);
    let kiln_top10 = top_k(kiln_logits, 10);
    let top10_overlap = hf_top10
        .iter()
        .filter(|index| kiln_top10.contains(index))
        .count();

    eprintln!(
        "KILN_VULKAN_HF_FULL_LOGIT_PASS vocab={} argmax_equal={argmax_equal} \
         hf_argmax={hf_argmax} kiln_argmax={kiln_argmax} top10_overlap={top10_overlap} \
         max_abs={max_abs:e} mean_abs={mean_abs:e} cosine={cosine:.12}",
        hf_logits.len()
    );
    anyhow::ensure!(
        argmax_equal == 1,
        "Vulkan argmax {kiln_argmax} differs from HF argmax {hf_argmax}"
    );
    anyhow::ensure!(
        top10_overlap >= 9,
        "Vulkan/HF top-10 overlap {top10_overlap} is below 9: HF={hf_top10:?} Kiln={kiln_top10:?}"
    );
    anyhow::ensure!(
        max_abs <= 0.5,
        "Vulkan/HF maximum absolute logit error {max_abs:e} exceeds 0.5"
    );
    anyhow::ensure!(
        mean_abs <= 0.05,
        "Vulkan/HF mean absolute logit error {mean_abs:e} exceeds 0.05"
    );
    anyhow::ensure!(
        cosine >= 0.9999,
        "Vulkan/HF full-logit cosine {cosine:.12} is below 0.9999"
    );
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

fn top_k(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    indices
}

fn build_cache(
    config: &ModelConfig,
    device: &Device,
    num_blocks: usize,
    block_size: usize,
) -> Result<PagedKvCache> {
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => DType::BF16,
        kiln_core::config::DType::FP16 => DType::F16,
        kiln_core::config::DType::FP32 => DType::F32,
    };
    // (#1082) Allocate pools on the runtime `Device`. The Vulkan resident-decode
    // path keeps its kt pools CPU-resident (the real KV bytes live in
    // `VkPagedKvCache`), so a Vulkan device routes to the host-resident default.
    PagedKvCache::new(
        config.num_full_attention_layers,
        num_blocks,
        block_size,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        *device,
    )
}

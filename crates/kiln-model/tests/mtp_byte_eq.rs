//! Phase B2 byte-equality test for native MTP weights.
//!
//! Loads a Qwen3.5-4B checkpoint via the kiln loader and re-reads each MTP
//! tensor directly from the underlying safetensors shard, then asserts that
//! the bytes, shape, and dtype match exactly. Catches any silent rewrite or
//! reshape that the loader might have introduced between disk and the
//! `MtpWeights` struct that `mtp_forward_step` consumes.
//!
//! Why this exists: Phase B (PR #261) saw α = 0.154 with 6/13 draft steps
//! exhibiting an identity-prediction bias. The leading hypothesis is that
//! `mtp.fc` weights one half of `concat(norm_emb, norm_h)` disproportionately
//! over the other; a secondary hypothesis is that the two pre-fc RMSNorm
//! tensors are swapped at load time. Byte-equal comparison rules out a
//! loader-side rewrite for both classes of bug — if any tensor here fails,
//! the loader is the bug and there is no point chasing forward-pass math.
//!
//! Activation: gated on the `KILN_MTP_BYTE_EQ_MODEL` env var, which must
//! point at a HuggingFace-style model directory containing one or more
//! `model*.safetensors` files plus an optional `model.safetensors.index.json`.
//! Without the env var the test is skipped (logs a one-line note and
//! returns), so workspace `cargo test` on a host without the model still
//! passes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use kiln_core::config::ModelConfig;
use kiln_model::weights::{AttentionWeights, WeightTensor};
use kiln_model::{LoadModelOptions, load_model_with_options};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const MODEL_ENV: &str = "KILN_MTP_BYTE_EQ_MODEL";

#[test]
fn mtp_weights_match_safetensors_byte_for_byte() {
    let Some(model_dir) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        eprintln!(
            "[mtp_byte_eq] skipped — set {MODEL_ENV}=/path/to/Qwen3.5-4B to enable"
        );
        return;
    };
    run(&model_dir).expect("byte-eq comparison failed");
}

fn run(model_dir: &Path) -> Result<()> {
    let config = qwen35_4b_config()
        .context("could not build Qwen3.5-4B model config for byte-eq test")?;

    // 1. Load MTP via the kiln loader.
    let opts = LoadModelOptions { load_mtp: true };
    let weights = load_model_with_options(model_dir, &config, opts)
        .with_context(|| format!("kiln load_model failed for {}", model_dir.display()))?;
    let mtp = weights
        .mtp
        .as_ref()
        .context("checkpoint did not yield MTP weights — wrong model?")?;

    // 2. Re-read every MTP-prefixed tensor directly from safetensors.
    let shards = discover_shards(model_dir)?;
    let mmaps: Vec<Mmap> = shards
        .iter()
        .map(|p| {
            let f = std::fs::File::open(p)
                .with_context(|| format!("open shard {}", p.display()))?;
            // SAFETY: read-only mmap of an immutable file for the duration of
            // this test. Standard safetensors loader pattern.
            let mmap = unsafe {
                Mmap::map(&f).with_context(|| format!("mmap shard {}", p.display()))?
            };
            anyhow::Ok(mmap)
        })
        .collect::<Result<Vec<_>>>()?;
    let parsed: Vec<SafeTensors<'_>> = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m).context("safetensors parse"))
        .collect::<Result<Vec<_>>>()?;
    let mut raw: HashMap<String, TensorView<'_>> = HashMap::new();
    for st in &parsed {
        for (name, view) in st.tensors() {
            raw.insert(name, view);
        }
    }

    // The loader auto-detects `mtp.` (bare) vs `model.language_model.mtp.`;
    // probe both here so this test stays in lockstep with the loader rules
    // in `loader.rs::detect_mtp_prefix`.
    let mtp_prefix = if raw.contains_key("mtp.fc.weight") {
        "mtp.".to_string()
    } else if raw.contains_key("model.language_model.mtp.fc.weight") {
        "model.language_model.mtp.".to_string()
    } else {
        bail!("checkpoint at {} has no MTP fc tensor", model_dir.display());
    };
    eprintln!("[mtp_byte_eq] using mtp_prefix = {mtp_prefix}");

    // Stock Qwen3.5-4B uses `mtp.norm.weight`; older docs / earlier kiln
    // comments name it `mtp.final_layernorm.weight`. Mirror the loader.
    let final_ln_name = {
        let a = format!("{mtp_prefix}norm.weight");
        let b = format!("{mtp_prefix}final_layernorm.weight");
        if raw.contains_key(&a) {
            a
        } else if raw.contains_key(&b) {
            b
        } else {
            bail!("MTP final RMSNorm not found under {mtp_prefix}");
        }
    };

    // 3. Compare each tensor: dtype, shape, and bytes.
    let mut report = Vec::new();
    let mut all_ok = true;

    let cases: Vec<(&str, String, &WeightTensor)> = vec![
        (
            "fc",
            format!("{mtp_prefix}fc.weight"),
            &mtp.fc,
        ),
        (
            "pre_fc_norm_embedding",
            format!("{mtp_prefix}pre_fc_norm_embedding.weight"),
            &mtp.pre_fc_norm_embedding,
        ),
        (
            "pre_fc_norm_hidden",
            format!("{mtp_prefix}pre_fc_norm_hidden.weight"),
            &mtp.pre_fc_norm_hidden,
        ),
        (
            "final_layernorm",
            final_ln_name.clone(),
            &mtp.final_layernorm,
        ),
    ];

    for (label, raw_name, loaded) in &cases {
        let view = raw
            .get(raw_name.as_str())
            .with_context(|| format!("safetensors raw missing {raw_name}"))?;
        let ok = compare(label, raw_name, view, loaded, &mut report);
        all_ok &= ok;
    }

    // Layer-level: compare the inner MTP transformer block tensors that are
    // most likely to drive draft predictions. q/k/v/o + their norms, gate/up/
    // down, and the two layer norms. The layer prefix in safetensors is
    // `{mtp_prefix}layers.0.`.
    let layer_prefix = format!("{mtp_prefix}layers.0.");
    let layer_cases: Vec<(&str, String, &WeightTensor)> = vec![
        (
            "layer.input_layernorm",
            format!("{layer_prefix}input_layernorm.weight"),
            &mtp.layer.input_layernorm,
        ),
        (
            "layer.post_attention_layernorm",
            format!("{layer_prefix}post_attention_layernorm.weight"),
            &mtp.layer.post_attention_layernorm,
        ),
        (
            "layer.mlp.gate_proj",
            format!("{layer_prefix}mlp.gate_proj.weight"),
            &mtp.layer.mlp.gate_proj,
        ),
        (
            "layer.mlp.up_proj",
            format!("{layer_prefix}mlp.up_proj.weight"),
            &mtp.layer.mlp.up_proj,
        ),
        (
            "layer.mlp.down_proj",
            format!("{layer_prefix}mlp.down_proj.weight"),
            &mtp.layer.mlp.down_proj,
        ),
    ];
    for (label, raw_name, loaded) in &layer_cases {
        let view = raw
            .get(raw_name.as_str())
            .with_context(|| format!("safetensors raw missing {raw_name}"))?;
        let ok = compare(label, raw_name, view, loaded, &mut report);
        all_ok &= ok;
    }

    // Q/K/V/O for the MTP layer — only present on full-attention. The loader
    // already bails if MTP came up linear, so this `match` is exhaustive in
    // practice.
    if let AttentionWeights::Full(attn) = &mtp.layer.attention {
        let attn_cases: Vec<(&str, String, &WeightTensor)> = vec![
            (
                "layer.attn.q_proj",
                format!("{layer_prefix}self_attn.q_proj.weight"),
                &attn.q_proj,
            ),
            (
                "layer.attn.k_proj",
                format!("{layer_prefix}self_attn.k_proj.weight"),
                &attn.k_proj,
            ),
            (
                "layer.attn.v_proj",
                format!("{layer_prefix}self_attn.v_proj.weight"),
                &attn.v_proj,
            ),
            (
                "layer.attn.o_proj",
                format!("{layer_prefix}self_attn.o_proj.weight"),
                &attn.o_proj,
            ),
            (
                "layer.attn.q_norm",
                format!("{layer_prefix}self_attn.q_norm.weight"),
                &attn.q_norm,
            ),
            (
                "layer.attn.k_norm",
                format!("{layer_prefix}self_attn.k_norm.weight"),
                &attn.k_norm,
            ),
        ];
        for (label, raw_name, loaded) in &attn_cases {
            let view = raw
                .get(raw_name.as_str())
                .with_context(|| format!("safetensors raw missing {raw_name}"))?;
            let ok = compare(label, raw_name, view, loaded, &mut report);
            all_ok &= ok;
        }
    } else {
        bail!("MTP layer attention is Linear; loader should have bailed");
    }

    eprintln!("[mtp_byte_eq] === report ===");
    for line in &report {
        eprintln!("[mtp_byte_eq] {line}");
    }

    if !all_ok {
        bail!("one or more MTP tensors failed byte-eq — see report above");
    }
    eprintln!("[mtp_byte_eq] PASS — all MTP tensors byte-equal to safetensors");
    Ok(())
}

fn compare(
    label: &str,
    raw_name: &str,
    view: &TensorView<'_>,
    loaded: &WeightTensor,
    report: &mut Vec<String>,
) -> bool {
    let raw_dtype = format!("{:?}", view.dtype());
    let raw_shape: Vec<usize> = view.shape().to_vec();
    let raw_bytes = view.data();
    let loaded_dtype = format!("{}", loaded.dtype);
    let shape_ok = raw_shape == loaded.shape;
    let len_ok = raw_bytes.len() == loaded.data.len();
    let bytes_ok = len_ok && raw_bytes == loaded.data.as_slice();

    // Dtype label sanity: `safetensors::Dtype::BF16` → "BF16",
    // `kiln_model::TensorDType::BF16` → "bf16". Compare case-insensitively.
    let dtype_ok = raw_dtype.eq_ignore_ascii_case(&loaded_dtype);

    let verdict = if shape_ok && dtype_ok && bytes_ok {
        "PASS"
    } else {
        "FAIL"
    };
    let mut diff_summary = String::new();
    if !shape_ok {
        diff_summary.push_str(&format!(
            " shape_diff(raw={:?},loaded={:?})",
            raw_shape, loaded.shape
        ));
    }
    if !dtype_ok {
        diff_summary.push_str(&format!(
            " dtype_diff(raw={raw_dtype},loaded={loaded_dtype})"
        ));
    }
    if !bytes_ok {
        let first = first_diff_byte(raw_bytes, &loaded.data);
        diff_summary.push_str(&format!(
            " bytes_diff(raw_len={},loaded_len={},first_diff_at={:?})",
            raw_bytes.len(),
            loaded.data.len(),
            first
        ));
    }
    report.push(format!(
        "{verdict} {label} ({raw_name}) shape={raw_shape:?} dtype={raw_dtype}{diff_summary}"
    ));
    shape_ok && dtype_ok && bytes_ok
}

fn first_diff_byte(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .or_else(|| {
            if a.len() != b.len() {
                Some(a.len().min(b.len()))
            } else {
                None
            }
        })
}

fn discover_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    if !index.exists() {
        bail!(
            "no model.safetensors or model.safetensors.index.json under {}",
            model_dir.display()
        );
    }
    let s = std::fs::read_to_string(&index).context("read index.json")?;
    let v: serde_json::Value = serde_json::from_str(&s).context("parse index.json")?;
    let mut shards: Vec<PathBuf> = Vec::new();
    if let Some(map) = v.get("weight_map").and_then(|w| w.as_object()) {
        for shard in map.values() {
            if let Some(s) = shard.as_str() {
                let p = model_dir.join(s);
                if !shards.contains(&p) {
                    shards.push(p);
                }
            }
        }
    }
    if shards.is_empty() {
        bail!(
            "weight_map empty in {}/model.safetensors.index.json",
            model_dir.display()
        );
    }
    shards.sort();
    Ok(shards)
}

fn qwen35_4b_config() -> Result<ModelConfig> {
    // Build the same shape kiln uses everywhere else for Qwen3.5-4B. Fields
    // here mirror kiln_core::config::ModelConfig defaults at the time of
    // PR #261; if the config schema gains required fields they need to be
    // mirrored here so the byte-eq test keeps building.
    Ok(ModelConfig::qwen3_5_4b())
}

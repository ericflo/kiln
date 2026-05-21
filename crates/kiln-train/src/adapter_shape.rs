//! Base-adapter shape validation for continued training.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
#[cfg(test)]
use candle_core::{DType, Device, Tensor};
use kiln_core::config::ModelConfig;
use kiln_model::lora_loader::AdapterConfig;

pub const ALLOW_ADAPTER_SHAPE_CONVERSION_FLAG: &str = "--allow-adapter-shape-conversion";

pub const TRAINABLE_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseAdapterCompatibility {
    pub adapter_dir: PathBuf,
    pub rank: usize,
    pub target_modules: Vec<String>,
    pub tensor_count: usize,
}

pub fn resolve_base_adapter_dir(base_adapter: &str, adapter_parent_dir: &Path) -> PathBuf {
    let direct = Path::new(base_adapter);
    if direct.exists() {
        direct.to_path_buf()
    } else {
        adapter_parent_dir.join(base_adapter)
    }
}

pub fn validate_base_adapter_compatibility(
    adapter_dir: &Path,
    model_config: &ModelConfig,
    expected_rank: usize,
    allow_adapter_shape_conversion: bool,
) -> Result<BaseAdapterCompatibility> {
    if expected_rank == 0 {
        bail!("training LoRA rank must be greater than zero");
    }
    let config_path = adapter_dir.join("adapter_config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read base adapter config {}", config_path.display()))?;
    let adapter_config: AdapterConfig = serde_json::from_str(&config_text)
        .with_context(|| format!("parse base adapter config {}", config_path.display()))?;

    if adapter_config.r != expected_rank {
        return incompatible(
            allow_adapter_shape_conversion,
            format!(
                "base adapter rank mismatch: training rank is {expected_rank}, adapter rank is {}",
                adapter_config.r
            ),
        );
    }

    let expected_targets = target_module_set(TRAINABLE_TARGET_MODULES.iter().copied());
    let adapter_targets =
        target_module_set(adapter_config.target_modules.iter().map(String::as_str));
    if adapter_targets != expected_targets {
        return incompatible(
            allow_adapter_shape_conversion,
            format!(
                "base adapter target_modules mismatch: expected {:?}, found {:?}",
                expected_targets, adapter_targets
            ),
        );
    }

    let st_path = adapter_dir.join("adapter_model.safetensors");
    let st_data = std::fs::read(&st_path)
        .with_context(|| format!("read base adapter tensors {}", st_path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&st_data)
        .with_context(|| format!("deserialize base adapter tensors {}", st_path.display()))?;

    let mut actual = BTreeMap::new();
    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("read base adapter tensor metadata {name}"))?;
        actual.insert(name.to_string(), tensor.shape().to_vec());
    }

    let expected = expected_base_adapter_tensors(model_config, expected_rank);
    for (name, expected_shape) in &expected {
        let Some(actual_shape) = actual.remove(name) else {
            return incompatible(
                allow_adapter_shape_conversion,
                format!("base adapter missing tensor {name}"),
            );
        };
        if actual_shape != *expected_shape {
            return incompatible(
                allow_adapter_shape_conversion,
                format!(
                    "base adapter tensor shape mismatch for {name}: expected {:?}, found {:?}",
                    expected_shape, actual_shape
                ),
            );
        }
    }

    if let Some((name, shape)) = actual.iter().next() {
        return incompatible(
            allow_adapter_shape_conversion,
            format!("base adapter has unexpected tensor {name} with shape {shape:?}"),
        );
    }

    Ok(BaseAdapterCompatibility {
        adapter_dir: adapter_dir.to_path_buf(),
        rank: adapter_config.r,
        target_modules: adapter_targets.into_iter().collect(),
        tensor_count: expected.len(),
    })
}

fn incompatible<T>(allow_adapter_shape_conversion: bool, message: String) -> Result<T> {
    if allow_adapter_shape_conversion {
        bail!(
            "{message}; {ALLOW_ADAPTER_SHAPE_CONVERSION_FLAG} was set, but adapter shape conversion is not implemented"
        );
    }
    bail!("{message}");
}

fn target_module_set<'a>(modules: impl Iterator<Item = &'a str>) -> BTreeSet<String> {
    modules.map(str::to_string).collect()
}

fn expected_base_adapter_tensors(
    config: &ModelConfig,
    rank: usize,
) -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::new();
    let hidden = config.hidden_size;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let q_out_dim = config.num_attention_heads * config.head_dim;
    let linear_v_dim = config.linear_v_dim();

    for layer_idx in 0..config.num_layers {
        if config.is_full_attention_layer(layer_idx) {
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "q_proj",
                hidden,
                config.full_attn_q_proj_dim(),
                rank,
            );
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "k_proj",
                hidden,
                kv_dim,
                rank,
            );
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "v_proj",
                hidden,
                kv_dim,
                rank,
            );
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "o_proj",
                q_out_dim,
                hidden,
                rank,
            );
        } else {
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "in_proj_qkv",
                hidden,
                config.linear_qkv_dim(),
                rank,
            );
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "in_proj_z",
                hidden,
                linear_v_dim,
                rank,
            );
            insert_lora_pair(
                &mut expected,
                layer_idx,
                "self_attn",
                "out_proj",
                linear_v_dim,
                hidden,
                rank,
            );
        }

        insert_lora_pair(
            &mut expected,
            layer_idx,
            "mlp",
            "gate_proj",
            hidden,
            config.intermediate_size,
            rank,
        );
        insert_lora_pair(
            &mut expected,
            layer_idx,
            "mlp",
            "up_proj",
            hidden,
            config.intermediate_size,
            rank,
        );
        insert_lora_pair(
            &mut expected,
            layer_idx,
            "mlp",
            "down_proj",
            config.intermediate_size,
            hidden,
            rank,
        );
    }

    expected
}

fn insert_lora_pair(
    expected: &mut BTreeMap<String, Vec<usize>>,
    layer_idx: usize,
    submodule: &str,
    module: &str,
    in_features: usize,
    out_features: usize,
    rank: usize,
) {
    let prefix = format!("base_model.model.model.layers.{layer_idx}.{submodule}.{module}");
    expected.insert(format!("{prefix}.lora_A.weight"), vec![rank, in_features]);
    expected.insert(format!("{prefix}.lora_B.weight"), vec![out_features, rank]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_hybrid_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_5_4b();
        config.hidden_size = 4;
        config.num_layers = 2;
        config.num_attention_heads = 2;
        config.num_kv_heads = 1;
        config.head_dim = 2;
        config.intermediate_size = 6;
        config.num_full_attention_layers = 1;
        config.full_attention_interval = 2;
        config.attn_output_gate = false;
        config.linear_num_key_heads = 1;
        config.linear_key_head_dim = 2;
        config.linear_num_value_heads = 1;
        config.linear_value_head_dim = 2;
        config
    }

    fn write_adapter(
        dir: &Path,
        config: &ModelConfig,
        rank: usize,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<usize>>, &mut Vec<String>),
    ) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut target_modules = TRAINABLE_TARGET_MODULES
            .iter()
            .map(|module| module.to_string())
            .collect::<Vec<_>>();
        let mut shapes = expected_base_adapter_tensors(config, rank);
        mutate(&mut shapes, &mut target_modules);
        let adapter_config = serde_json::json!({
            "r": rank,
            "lora_alpha": (rank * 2) as f32,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "peft_type": "LORA",
        });
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&adapter_config)?,
        )?;
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        for (name, shape) in shapes {
            tensors.insert(name, Tensor::zeros(shape, DType::F32, &device)?);
        }
        candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
        Ok(())
    }

    #[test]
    fn base_adapter_shape_valid_match_passes() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        write_adapter(tmp.path(), &config, 2, |_, _| {})?;

        let result = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)?;

        assert_eq!(result.rank, 2);
        assert_eq!(
            result.tensor_count,
            expected_base_adapter_tensors(&config, 2).len()
        );
        Ok(())
    }

    #[test]
    fn base_adapter_rank_mismatch_fails() -> Result<()> {
        // Issue 40 regression: rank mismatches must fail in the
        // pre-training compatibility check, before optimizer setup.
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        write_adapter(tmp.path(), &config, 4, |_, _| {})?;

        let err = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("rank mismatch"));
        assert!(err.contains("training rank is 2"));
        assert!(err.contains("adapter rank is 4"));
        Ok(())
    }

    #[test]
    fn base_adapter_target_modules_mismatch_fails() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        write_adapter(tmp.path(), &config, 2, |_, target_modules| {
            target_modules.retain(|module| module != "out_proj");
        })?;

        let err = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("target_modules mismatch"));
        assert!(err.contains("out_proj"));
        Ok(())
    }

    #[test]
    fn base_adapter_missing_tensor_fails_with_name() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        let missing = "base_model.model.model.layers.0.self_attn.in_proj_qkv.lora_A.weight";
        write_adapter(tmp.path(), &config, 2, |shapes, _| {
            shapes.remove(missing);
        })?;

        let err = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("missing tensor"));
        assert!(err.contains(missing));
        Ok(())
    }

    #[test]
    fn base_adapter_extra_tensor_fails_with_name() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        let extra = "base_model.model.model.layers.0.mlp.extra_proj.lora_A.weight";
        write_adapter(tmp.path(), &config, 2, |shapes, _| {
            shapes.insert(extra.to_string(), vec![2, 4]);
        })?;

        let err = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("unexpected tensor"));
        assert!(err.contains(extra));
        Ok(())
    }

    #[test]
    fn base_adapter_shape_mismatch_fails_with_name() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let config = tiny_hybrid_config();
        let bad = "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight";
        write_adapter(tmp.path(), &config, 2, |shapes, _| {
            shapes.insert(bad.to_string(), vec![5, 2]);
        })?;

        let err = validate_base_adapter_compatibility(tmp.path(), &config, 2, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("shape mismatch"));
        assert!(err.contains(bad));
        Ok(())
    }
}

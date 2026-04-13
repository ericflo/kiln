use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use kiln_core::config::ModelConfig;

use crate::weights::*;

/// Weight name prefix for the language model within the VL checkpoint.
/// Qwen3.5-4B ships as a vision-language model; we only load the LM portion.
const LM_PREFIX: &str = "model.language_model.";

/// Safetensors index file for sharded checkpoints.
const INDEX_FILENAME: &str = "model.safetensors.index.json";

/// Load Qwen3.5-4B language model weights from a directory of safetensors files.
///
/// Handles both single-file (`model.safetensors`) and sharded checkpoints
/// (`model-00001-of-NNNNN.safetensors` with an index file).
///
/// The model directory should be a HuggingFace model download containing
/// the safetensors files. Only language model weights (prefixed with
/// `model.language_model.`) are loaded; vision encoder weights are skipped.
pub fn load_model(model_dir: &Path, config: &ModelConfig) -> Result<ModelWeights> {
    let shards = discover_shards(model_dir)?;
    tracing::info!(
        "Loading model from {} ({} shard{})",
        model_dir.display(),
        shards.len(),
        if shards.len() == 1 { "" } else { "s" }
    );

    // Memory-map all shards and parse safetensors headers.
    let mmaps = mmap_shards(&shards)?;
    let parsed: Vec<SafeTensors<'_>> = mmaps
        .iter()
        .map(|mmap| {
            SafeTensors::deserialize(mmap)
                .context("Failed to parse safetensors file")
        })
        .collect::<Result<Vec<_>>>()?;

    // Build a unified name -> (shard_idx, tensor_view) lookup.
    // We collect (String, TensorView) pairs and then reference them by &str.
    let mut all_tensors: Vec<(String, usize, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (shard_idx, st) in parsed.iter().enumerate() {
        for (name, view) in st.tensors() {
            all_tensors.push((name, shard_idx, view));
        }
    }
    let mut tensor_map: HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)> = HashMap::new();
    for (name, shard_idx, view) in &all_tensors {
        tensor_map.insert(name.as_str(), (*shard_idx, view));
    }

    // Auto-detect prefix: try "model.language_model." first, fall back to "model.".
    let prefix = detect_prefix(&tensor_map);
    tracing::info!("Using weight name prefix: \"{prefix}\"");

    // Load embedding.
    let embed_tokens = extract_tensor(&tensor_map, &format!("{prefix}embed_tokens.weight"))?;
    validate_shape(&embed_tokens, &[config.vocab_size, config.hidden_size], "embed_tokens")?;

    // Load layers.
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer_prefix = format!("{prefix}layers.{i}.");
        let layer = load_layer(&tensor_map, &layer_prefix, i, config)?;
        layers.push(layer);
    }

    // Load final norm.
    let final_norm = extract_tensor(&tensor_map, &format!("{prefix}norm.weight"))?;
    validate_shape(&final_norm, &[config.hidden_size], "final_norm")?;

    let weights = ModelWeights {
        embedding: EmbeddingWeights { embed_tokens },
        layers,
        final_norm,
    };

    let total_mb = weights.total_bytes() as f64 / (1024.0 * 1024.0);
    let total_params_m = weights.total_params() as f64 / 1_000_000.0;
    tracing::info!(
        "Loaded {:.0}M parameters ({:.0} MB) across {} layers",
        total_params_m,
        total_mb,
        config.num_layers,
    );

    Ok(weights)
}

/// Discover safetensors shard files in the model directory.
fn discover_shards(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index_path = model_dir.join(INDEX_FILENAME);

    if index_path.exists() {
        // Sharded model: read index to find shard filenames.
        let index_bytes = fs::read(&index_path)
            .with_context(|| format!("Failed to read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes)
            .context("Failed to parse safetensors index JSON")?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("Index file missing 'weight_map' object")?;

        // Collect unique shard filenames, preserving order.
        let mut seen = std::collections::HashSet::new();
        let mut shard_files = Vec::new();
        for filename in weight_map.values() {
            let filename = filename
                .as_str()
                .context("weight_map value is not a string")?;
            if seen.insert(filename.to_string()) {
                let shard_path = model_dir.join(filename);
                if !shard_path.exists() {
                    bail!(
                        "Shard file {} referenced in index but not found on disk",
                        shard_path.display()
                    );
                }
                shard_files.push(shard_path);
            }
        }

        if shard_files.is_empty() {
            bail!("Index file has an empty weight_map");
        }

        Ok(shard_files)
    } else {
        // Single-file model.
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            Ok(vec![single])
        } else {
            // Try glob for sharded files without index.
            let mut shards: Vec<_> = fs::read_dir(model_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().is_some_and(|ext| ext == "safetensors")
                })
                .collect();
            shards.sort();

            if shards.is_empty() {
                bail!(
                    "No .safetensors files found in {}",
                    model_dir.display()
                );
            }
            Ok(shards)
        }
    }
}

/// Memory-map all shard files.
fn mmap_shards(paths: &[std::path::PathBuf]) -> Result<Vec<Mmap>> {
    paths
        .iter()
        .map(|path| {
            let file = fs::File::open(path)
                .with_context(|| format!("Failed to open {}", path.display()))?;
            // SAFETY: We assume the file is not modified while mapped.
            // This is standard practice for model weight loading.
            unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap {}", path.display()))
        })
        .collect()
}

/// Detect the weight name prefix by checking which keys exist.
fn detect_prefix(tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>) -> String {
    // Try the VL model prefix first (Qwen3.5-4B ships as VL).
    if tensor_map.contains_key(&format!("{LM_PREFIX}embed_tokens.weight") as &str) {
        return LM_PREFIX.to_string();
    }
    // Fall back to bare prefix for text-only checkpoints.
    if tensor_map.contains_key("model.embed_tokens.weight") {
        return "model.".to_string();
    }
    // Last resort: no prefix.
    if tensor_map.contains_key("embed_tokens.weight") {
        return String::new();
    }
    // Default to VL prefix and let it fail with a clear error.
    LM_PREFIX.to_string()
}

/// Load one transformer layer's weights.
fn load_layer(
    tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<LayerWeights> {
    let input_layernorm = extract_tensor(tensor_map, &format!("{prefix}input_layernorm.weight"))?;
    validate_shape(&input_layernorm, &[config.hidden_size], &format!("layer {layer_idx} input_layernorm"))?;

    let post_attention_layernorm = extract_tensor(tensor_map, &format!("{prefix}post_attention_layernorm.weight"))?;
    validate_shape(&post_attention_layernorm, &[config.hidden_size], &format!("layer {layer_idx} post_attn_layernorm"))?;

    let mlp = load_ffn(tensor_map, prefix, layer_idx, config)?;

    let attention = if config.is_full_attention_layer(layer_idx) {
        AttentionWeights::Full(load_full_attention(tensor_map, prefix, layer_idx, config)?)
    } else {
        AttentionWeights::Linear(load_linear_attention(tensor_map, prefix, layer_idx, config)?)
    };

    Ok(LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    })
}

/// Load full GQA attention weights.
fn load_full_attention(
    tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<FullAttentionWeights> {
    let attn = format!("{prefix}self_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} self_attn.{name}");

    let q_proj_dim = config.full_attn_q_proj_dim();
    let q_out_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let q_proj = extract_tensor(tensor_map, &format!("{attn}q_proj.weight"))?;
    validate_shape(&q_proj, &[q_proj_dim, config.hidden_size], &ctx("q_proj"))?;

    let k_proj = extract_tensor(tensor_map, &format!("{attn}k_proj.weight"))?;
    validate_shape(&k_proj, &[kv_dim, config.hidden_size], &ctx("k_proj"))?;

    let v_proj = extract_tensor(tensor_map, &format!("{attn}v_proj.weight"))?;
    validate_shape(&v_proj, &[kv_dim, config.hidden_size], &ctx("v_proj"))?;

    let o_proj = extract_tensor(tensor_map, &format!("{attn}o_proj.weight"))?;
    validate_shape(&o_proj, &[config.hidden_size, q_out_dim], &ctx("o_proj"))?;

    let q_norm = extract_tensor(tensor_map, &format!("{attn}q_norm.weight"))?;
    validate_shape(&q_norm, &[config.head_dim], &ctx("q_norm"))?;

    let k_norm = extract_tensor(tensor_map, &format!("{attn}k_norm.weight"))?;
    validate_shape(&k_norm, &[config.head_dim], &ctx("k_norm"))?;

    Ok(FullAttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
    })
}

/// Load Gated DeltaNet linear attention weights.
fn load_linear_attention(
    tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<LinearAttentionWeights> {
    let attn = format!("{prefix}linear_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} linear_attn.{name}");

    // Linear attention uses asymmetric Q/K/V dimensions from config.
    let fused_qkv_dim = config.linear_qkv_dim(); // Q + K + V total
    let v_dim = config.linear_v_dim();
    let num_heads = config.linear_num_value_heads;

    let in_proj_qkv = extract_tensor(tensor_map, &format!("{attn}in_proj_qkv.weight"))?;
    validate_shape(&in_proj_qkv, &[fused_qkv_dim, config.hidden_size], &ctx("in_proj_qkv"))?;

    let in_proj_z = extract_tensor(tensor_map, &format!("{attn}in_proj_z.weight"))?;
    validate_shape(&in_proj_z, &[v_dim, config.hidden_size], &ctx("in_proj_z"))?;

    let out_proj = extract_tensor(tensor_map, &format!("{attn}out_proj.weight"))?;
    validate_shape(&out_proj, &[config.hidden_size, v_dim], &ctx("out_proj"))?;

    let in_proj_a = extract_tensor(tensor_map, &format!("{attn}in_proj_a.weight"))?;
    validate_shape(&in_proj_a, &[num_heads, config.hidden_size], &ctx("in_proj_a"))?;

    let in_proj_b = extract_tensor(tensor_map, &format!("{attn}in_proj_b.weight"))?;
    validate_shape(&in_proj_b, &[num_heads, config.hidden_size], &ctx("in_proj_b"))?;

    // conv1d shape is [channels, 1, kernel_size] — channels = fused QKV dim.
    let conv1d = extract_tensor(tensor_map, &format!("{attn}conv1d.weight"))?;
    if conv1d.shape.len() != 3 || conv1d.shape[0] != fused_qkv_dim || conv1d.shape[1] != 1 {
        bail!(
            "Shape mismatch for {}: expected [{fused_qkv_dim}, 1, *], got {:?}",
            ctx("conv1d"),
            conv1d.shape
        );
    }

    let norm = extract_tensor(tensor_map, &format!("{attn}norm.weight"))?;
    // Group norm weight — per-head normalization.
    validate_shape(&norm, &[config.linear_key_head_dim], &ctx("norm"))?;

    let a_log = extract_tensor(tensor_map, &format!("{attn}A_log"))?;
    if a_log.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("A_log"),
            a_log.numel(),
            a_log.shape
        );
    }

    let dt_bias = extract_tensor(tensor_map, &format!("{attn}dt_bias"))?;
    if dt_bias.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("dt_bias"),
            dt_bias.numel(),
            dt_bias.shape
        );
    }

    Ok(LinearAttentionWeights {
        in_proj_qkv,
        in_proj_z,
        out_proj,
        in_proj_a,
        in_proj_b,
        conv1d,
        norm,
        a_log,
        dt_bias,
    })
}

/// Load FFN/MLP weights.
fn load_ffn(
    tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<FfnWeights> {
    let mlp = format!("{prefix}mlp.");
    let ctx = |name: &str| format!("layer {layer_idx} mlp.{name}");

    let gate_proj = extract_tensor(tensor_map, &format!("{mlp}gate_proj.weight"))?;
    validate_shape(&gate_proj, &[config.intermediate_size, config.hidden_size], &ctx("gate_proj"))?;

    let up_proj = extract_tensor(tensor_map, &format!("{mlp}up_proj.weight"))?;
    validate_shape(&up_proj, &[config.intermediate_size, config.hidden_size], &ctx("up_proj"))?;

    let down_proj = extract_tensor(tensor_map, &format!("{mlp}down_proj.weight"))?;
    validate_shape(&down_proj, &[config.hidden_size, config.intermediate_size], &ctx("down_proj"))?;

    Ok(FfnWeights {
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// Extract a tensor by name from the unified tensor map.
fn extract_tensor(
    tensor_map: &HashMap<&str, (usize, &safetensors::tensor::TensorView<'_>)>,
    name: &str,
) -> Result<WeightTensor> {
    let (_shard_idx, view) = tensor_map
        .get(name)
        .with_context(|| format!("Weight tensor not found: {name}"))?;

    let dtype = convert_dtype(view.dtype())
        .with_context(|| format!("Unsupported dtype {:?} for tensor {name}", view.dtype()))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data().to_vec();

    Ok(WeightTensor { data, shape, dtype })
}

/// Convert safetensors dtype to our dtype enum.
fn convert_dtype(dt: safetensors::Dtype) -> Result<TensorDType> {
    match dt {
        safetensors::Dtype::F16 => Ok(TensorDType::F16),
        safetensors::Dtype::BF16 => Ok(TensorDType::BF16),
        safetensors::Dtype::F32 => Ok(TensorDType::F32),
        other => bail!("Unsupported dtype: {other:?}"),
    }
}

/// Validate that a tensor has the expected shape.
fn validate_shape(tensor: &WeightTensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape != expected {
        bail!(
            "Shape mismatch for {name}: expected {expected:?}, got {:?}",
            tensor.shape
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::Dtype as StDtype;
    use std::collections::HashMap as StdMap;

    /// Create a minimal safetensors file with the given tensors.
    fn create_test_safetensors(
        tensors: &[(&str, Vec<usize>, StDtype)],
    ) -> Vec<u8> {
        let mut data_map: StdMap<String, Vec<u8>> = StdMap::new();
        let mut views = Vec::new();

        for (name, shape, dtype) in tensors {
            let numel: usize = shape.iter().product();
            let elem_size = match dtype {
                StDtype::BF16 | StDtype::F16 => 2,
                StDtype::F32 => 4,
                _ => panic!("unsupported dtype in test"),
            };
            let data = vec![0u8; numel * elem_size];
            data_map.insert(name.to_string(), data);
            views.push((name.to_string(), shape.clone(), *dtype));
        }

        let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = views
            .iter()
            .map(|(name, shape, dtype)| {
                let data = data_map.get(name).unwrap();
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_views
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();

        safetensors::tensor::serialize(refs, &None).unwrap()
    }

    /// Build a list of tensor specs for a tiny test model.
    fn tiny_model_tensors(prefix: &str) -> Vec<(String, Vec<usize>, StDtype)> {
        let config = tiny_model_config();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;

        // Full attention dims
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_out_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        // Linear attention dims
        let fused_qkv_dim = config.linear_qkv_dim();
        let v_dim = config.linear_v_dim();
        let num_linear_heads = config.linear_num_value_heads;
        let conv_size = config.linear_conv_kernel_dim;

        let mut tensors: Vec<(String, Vec<usize>, StDtype)> = Vec::new();
        let bf16 = StDtype::BF16;

        // Embedding + final norm
        tensors.push((format!("{prefix}embed_tokens.weight"), vec![vocab, hidden], bf16));
        tensors.push((format!("{prefix}norm.weight"), vec![hidden], bf16));

        // 4 layers: layers 0,1,2 are linear, layer 3 is full attention
        for i in 0..config.num_layers {
            let lp = format!("{prefix}layers.{i}.");
            tensors.push((format!("{lp}input_layernorm.weight"), vec![hidden], bf16));
            tensors.push((format!("{lp}post_attention_layernorm.weight"), vec![hidden], bf16));
            tensors.push((format!("{lp}mlp.gate_proj.weight"), vec![intermediate, hidden], bf16));
            tensors.push((format!("{lp}mlp.up_proj.weight"), vec![intermediate, hidden], bf16));
            tensors.push((format!("{lp}mlp.down_proj.weight"), vec![hidden, intermediate], bf16));

            if config.is_full_attention_layer(i) {
                // Full attention layer
                tensors.push((format!("{lp}self_attn.q_proj.weight"), vec![q_proj_dim, hidden], bf16));
                tensors.push((format!("{lp}self_attn.k_proj.weight"), vec![kv_dim, hidden], bf16));
                tensors.push((format!("{lp}self_attn.v_proj.weight"), vec![kv_dim, hidden], bf16));
                tensors.push((format!("{lp}self_attn.o_proj.weight"), vec![hidden, q_out_dim], bf16));
                tensors.push((format!("{lp}self_attn.q_norm.weight"), vec![config.head_dim], bf16));
                tensors.push((format!("{lp}self_attn.k_norm.weight"), vec![config.head_dim], bf16));
            } else {
                // Linear attention layer
                tensors.push((format!("{lp}linear_attn.in_proj_qkv.weight"), vec![fused_qkv_dim, hidden], bf16));
                tensors.push((format!("{lp}linear_attn.in_proj_z.weight"), vec![v_dim, hidden], bf16));
                tensors.push((format!("{lp}linear_attn.out_proj.weight"), vec![hidden, v_dim], bf16));
                tensors.push((format!("{lp}linear_attn.in_proj_a.weight"), vec![num_linear_heads, hidden], bf16));
                tensors.push((format!("{lp}linear_attn.in_proj_b.weight"), vec![num_linear_heads, hidden], bf16));
                tensors.push((format!("{lp}linear_attn.conv1d.weight"), vec![fused_qkv_dim, 1, conv_size], bf16));
                tensors.push((format!("{lp}linear_attn.norm.weight"), vec![config.linear_key_head_dim], bf16));
                tensors.push((format!("{lp}linear_attn.A_log"), vec![num_linear_heads], bf16));
                tensors.push((format!("{lp}linear_attn.dt_bias"), vec![num_linear_heads], bf16));
            }
        }

        tensors
    }

    fn tiny_model_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_num_key_heads: 4,
            linear_key_head_dim: 8,
            linear_num_value_heads: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
        }
    }

    #[test]
    fn test_load_single_shard() {
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        let safetensors_path = dir.path().join("model.safetensors");
        fs::write(&safetensors_path, &bytes).unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();

        // Check structure.
        assert_eq!(weights.layers.len(), 4);

        // Layers 0, 1, 2 should be linear attention.
        for i in 0..3 {
            assert!(
                matches!(weights.layers[i].attention, AttentionWeights::Linear(_)),
                "Layer {i} should be linear attention"
            );
        }

        // Layer 3 should be full attention.
        assert!(
            matches!(weights.layers[3].attention, AttentionWeights::Full(_)),
            "Layer 3 should be full attention"
        );

        // Check embedding shape.
        assert_eq!(weights.embedding.embed_tokens.shape, vec![256, 64]);
        assert_eq!(weights.embedding.embed_tokens.dtype, TensorDType::BF16);

        // Check final norm.
        assert_eq!(weights.final_norm.shape, vec![64]);

        // Check total params > 0.
        assert!(weights.total_params() > 0);
        assert!(weights.total_bytes() > 0);
    }

    #[test]
    fn test_load_bare_prefix() {
        // Test with "model." prefix (text-only checkpoint).
        let tensors = tiny_model_tensors("model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert_eq!(weights.layers.len(), 4);
    }

    #[test]
    fn test_load_sharded() {
        let all_tensors = tiny_model_tensors("model.language_model.");
        let mid = all_tensors.len() / 2;

        let shard1_tensors: Vec<(&str, Vec<usize>, StDtype)> = all_tensors[..mid]
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let shard2_tensors: Vec<(&str, Vec<usize>, StDtype)> = all_tensors[mid..]
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes1 = create_test_safetensors(&shard1_tensors);
        let bytes2 = create_test_safetensors(&shard2_tensors);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), &bytes1).unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), &bytes2).unwrap();

        // Create index file.
        let mut weight_map = serde_json::Map::new();
        for (name, _, _) in &all_tensors[..mid] {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("model-00001-of-00002.safetensors".into()),
            );
        }
        for (name, _, _) in &all_tensors[mid..] {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("model-00002-of-00002.safetensors".into()),
            );
        }
        let index = serde_json::json!({ "weight_map": weight_map });
        fs::write(
            dir.path().join(INDEX_FILENAME),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert_eq!(weights.layers.len(), 4);
    }

    #[test]
    fn test_shape_mismatch_detected() {
        let mut tensors = tiny_model_tensors("model.language_model.");
        // Corrupt embedding shape.
        tensors[0].1 = vec![999, 64];

        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let result = load_model(dir.path(), &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Shape mismatch"), "Error should mention shape mismatch: {err}");
    }

    #[test]
    fn test_missing_tensor_detected() {
        // Create tensors but skip the embedding.
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .skip(1) // skip embed_tokens
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let result = load_model(dir.path(), &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should mention missing tensor: {err}"
        );
    }
}

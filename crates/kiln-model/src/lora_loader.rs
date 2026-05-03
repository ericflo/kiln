//! LoRA adapter weight loading from PEFT-compatible safetensors format.
//!
//! Loads LoRA A/B matrices from safetensors files and adapter_config.json,
//! organizing them into per-layer structs for use during forward pass.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Configuration from PEFT's adapter_config.json.
#[derive(Debug, Deserialize)]
pub struct AdapterConfig {
    /// LoRA rank (r).
    pub r: usize,
    /// LoRA alpha scaling factor.
    pub lora_alpha: f32,
    /// Target modules (e.g., ["q_proj", "k_proj", "v_proj", "o_proj", ...]).
    pub target_modules: Vec<String>,
    /// Task type (optional, e.g., "CAUSAL_LM").
    #[serde(default)]
    pub task_type: Option<String>,
}

/// LoRA A/B weight pair for a single linear projection.
pub struct LoraProjectionWeights {
    /// A matrix: [rank, in_features]
    pub a: Tensor,
    /// B matrix: [out_features, rank]
    pub b: Tensor,
}

/// LoRA weights for all targeted modules in one transformer layer.
#[derive(Default)]
pub struct LoraLayerWeights {
    pub q_proj: Option<LoraProjectionWeights>,
    pub k_proj: Option<LoraProjectionWeights>,
    pub v_proj: Option<LoraProjectionWeights>,
    pub o_proj: Option<LoraProjectionWeights>,
    pub gate_proj: Option<LoraProjectionWeights>,
    pub up_proj: Option<LoraProjectionWeights>,
    pub down_proj: Option<LoraProjectionWeights>,
}

/// Complete LoRA adapter weights for all layers.
pub struct LoraWeights {
    /// Per-layer LoRA weights, indexed by layer number.
    pub layers: Vec<LoraLayerWeights>,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha (scaling factor).
    pub alpha: f32,
    /// Precomputed scale = alpha / rank.
    pub scale: f32,
}

impl LoraWeights {
    /// Load a PEFT-compatible LoRA adapter from a directory.
    ///
    /// The directory must contain:
    /// - `adapter_config.json`: PEFT configuration with rank, alpha, target_modules
    /// - `adapter_model.safetensors`: LoRA A/B weight matrices
    ///
    /// Weight keys follow the PEFT naming convention:
    /// `base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight`
    /// `base_model.model.model.layers.{i}.self_attn.{module}.lora_B.weight`
    /// and similarly for MLP modules under `.mlp.{module}.`.
    pub fn load(adapter_dir: &Path, num_layers: usize, device: &Device) -> Result<Self> {
        // Load adapter config
        let config_path = adapter_dir.join("adapter_config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config: AdapterConfig =
            serde_json::from_str(&config_str).context("failed to parse adapter_config.json")?;

        let rank = config.r;
        let alpha = config.lora_alpha;
        let scale = alpha / rank as f32;

        // Load safetensors
        let st_path = adapter_dir.join("adapter_model.safetensors");
        let st_data = std::fs::read(&st_path)
            .with_context(|| format!("failed to read {}", st_path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .context("failed to deserialize safetensors")?;

        // Parse all tensor names into a map: (layer_idx, module_name, "A"|"B") -> tensor_name
        let mut tensor_map: HashMap<(usize, String, String), String> = HashMap::new();

        for name in tensors.names() {
            if let Some(parsed) = parse_peft_key(name) {
                tensor_map.insert((parsed.layer, parsed.module, parsed.ab), name.to_string());
            }
        }

        // Build per-layer weights
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut layer = LoraLayerWeights::default();

            for module in &config.target_modules {
                let a_key = tensor_map.get(&(layer_idx, module.clone(), "A".to_string()));
                let b_key = tensor_map.get(&(layer_idx, module.clone(), "B".to_string()));

                if let (Some(a_name), Some(b_name)) = (a_key, b_key) {
                    let a_view = tensors
                        .tensor(a_name)
                        .with_context(|| format!("failed to get tensor {a_name}"))?;
                    let b_view = tensors
                        .tensor(b_name)
                        .with_context(|| format!("failed to get tensor {b_name}"))?;

                    let a = safetensor_to_candle(&a_view, device)
                        .with_context(|| format!("converting {a_name}"))?;
                    let b = safetensor_to_candle(&b_view, device)
                        .with_context(|| format!("converting {b_name}"))?;

                    let proj = LoraProjectionWeights { a, b };
                    match module.as_str() {
                        "q_proj" => layer.q_proj = Some(proj),
                        "k_proj" => layer.k_proj = Some(proj),
                        "v_proj" => layer.v_proj = Some(proj),
                        "o_proj" => layer.o_proj = Some(proj),
                        "gate_proj" => layer.gate_proj = Some(proj),
                        "up_proj" => layer.up_proj = Some(proj),
                        "down_proj" => layer.down_proj = Some(proj),
                        _ => {
                            tracing::warn!("unknown LoRA target module: {module}, skipping");
                        }
                    }
                }
            }

            layers.push(layer);
        }

        Ok(Self {
            layers,
            rank,
            alpha,
            scale,
        })
    }
}

/// Parsed PEFT weight key components.
struct ParsedKey {
    layer: usize,
    module: String,
    ab: String, // "A" or "B"
}

/// Parse a PEFT-style safetensors key into layer index, module name, and A/B indicator.
///
/// Expected patterns:
/// - `base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight`
/// - `base_model.model.model.layers.{i}.mlp.{module}.lora_{A|B}.weight`
fn parse_peft_key(key: &str) -> Option<ParsedKey> {
    // Look for "layers.{i}." and "lora_{A|B}.weight"
    let parts: Vec<&str> = key.split('.').collect();

    // Find "layers" followed by a number
    let layer_pos = parts.iter().position(|&p| p == "layers")?;
    let layer_idx: usize = parts.get(layer_pos + 1)?.parse().ok()?;

    // Find "lora_A" or "lora_B"
    let lora_pos = parts
        .iter()
        .position(|p| *p == "lora_A" || *p == "lora_B")?;
    let ab = if parts[lora_pos] == "lora_A" {
        "A".to_string()
    } else {
        "B".to_string()
    };

    // The module name is the part just before "lora_A" or "lora_B"
    let module = parts.get(lora_pos.checked_sub(1)?)?.to_string();

    Some(ParsedKey {
        layer: layer_idx,
        module,
        ab,
    })
}

/// Convert a safetensors tensor view to a candle Tensor.
fn safetensor_to_candle(
    view: &safetensors::tensor::TensorView<'_>,
    device: &Device,
) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = match view.dtype() {
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F32 => DType::F32,
        other => anyhow::bail!("unsupported safetensors dtype: {:?}", other),
    };
    let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)
        .context("failed to create tensor from safetensors data")?;
    Ok(tensor)
}

/// Apply a LoRA delta to a linear projection output.
///
/// Computes: `base_output + (x @ A^T @ B^T) * scale`
///
/// - `x`: input tensor [batch, seq_len, in_features] (or [seq_len, in_features])
/// - `proj`: LoRA A/B weight pair
/// - `scale`: alpha / rank
///
/// Returns: the LoRA delta tensor (same shape as base_output)
pub fn compute_lora_delta(x: &Tensor, proj: &LoraProjectionWeights, scale: f32) -> Result<Tensor> {
    // x: [..., in_features]
    // A: [rank, in_features] -> A^T: [in_features, rank]
    // B: [out_features, rank] -> B^T: [rank, out_features]
    // delta = x @ A^T @ B^T * scale
    //
    // Phase 10: cast A/B to x's dtype (BF16 typically; F32 when MTP fp32-head is
    // armed) and let cuBLAS run BF16-input + FP32-accumulate on tensor cores.
    // See docs/audits/PHASE10_LORA_PRECISION_STUDY.md §5.
    let a = proj.a.to_dtype(x.dtype())?;
    let b = proj.b.to_dtype(x.dtype())?;

    let hidden = x.broadcast_matmul(&a.t()?)?; // [..., rank]
    let delta = hidden.broadcast_matmul(&b.t()?)?; // [..., out_features]
    let delta = (delta * scale as f64)?;

    // Final cast to input dtype (no-op when already matching).
    let delta = delta.to_dtype(x.dtype())?;
    Ok(delta)
}

fn cpu_needs_f32_matmul(lhs: &Tensor, rhs: &Tensor) -> bool {
    matches!(lhs.device(), Device::Cpu) && (lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32)
}

fn broadcast_matmul_cpu_compatible(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if cpu_needs_f32_matmul(lhs, rhs) {
        let lhs_f32 = lhs.to_dtype(DType::F32)?;
        let rhs_f32 = rhs.to_dtype(DType::F32)?;
        Ok(lhs_f32.broadcast_matmul(&rhs_f32)?)
    } else {
        Ok(lhs.broadcast_matmul(rhs)?)
    }
}

/// Apply a LoRA-augmented linear projection.
///
/// Computes: `(x @ W^T) + (x @ A^T @ B^T) * scale`
///
/// If no LoRA weights are provided for this projection, just returns `x @ W^T`.
pub fn linear_with_lora(
    x: &Tensor,
    base_weight: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    scale: f32,
) -> Result<Tensor> {
    let base_weight_t = base_weight.t()?;
    let base_output = broadcast_matmul_cpu_compatible(x, &base_weight_t)?;
    if let Some(proj) = lora {
        let delta = compute_lora_delta(x, proj, scale)?;
        Ok((base_output + delta)?)
    } else {
        Ok(base_output)
    }
}

/// Apply a LoRA-augmented linear projection using a pre-transposed base weight.
///
/// Takes `base_weight_t` = `base_weight.t().contiguous()` (shape `[in, out]`) and
/// computes `x @ base_weight_t` directly, avoiding the per-call transpose copy
/// (`ucopy_bf16`) that would otherwise be materialized on every step.
///
/// The LoRA delta path is unchanged.
///
/// Phase C12: when the MTP fp32-head TLS flag is armed (see
/// [`crate::mtp_debug::is_mtp_fp32_head_armed`]), the base matmul is
/// promoted to f32. Inputs and weights are upcast to f32, matmul runs in
/// f32, and the result is cast back to the input dtype before the LoRA
/// delta is added. The flag is only set inside `mtp_forward_step` while
/// the MTP inner transformer block is running, so every non-MTP call site
/// takes the legacy bf16 broadcast_matmul path unchanged.
pub fn linear_with_lora_t(
    x: &Tensor,
    base_weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    scale: f32,
) -> Result<Tensor> {
    let cpu_f32_matmul = cpu_needs_f32_matmul(x, base_weight_t);
    let base_output = if crate::mtp_debug::is_mtp_fp32_head_armed() || cpu_f32_matmul {
        let in_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let w_f32 = base_weight_t.to_dtype(DType::F32)?;
        let out = x_f32.broadcast_matmul(&w_f32)?;
        if cpu_f32_matmul {
            out
        } else {
            out.to_dtype(in_dtype)?
        }
    } else {
        x.broadcast_matmul(base_weight_t)?
    };
    if let Some(proj) = lora {
        let delta = compute_lora_delta(x, proj, scale)?;
        Ok((base_output + delta)?)
    } else {
        Ok(base_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_parse_peft_key_self_attn() {
        let key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer, 5);
        assert_eq!(parsed.module, "q_proj");
        assert_eq!(parsed.ab, "A");
    }

    #[test]
    fn test_parse_peft_key_mlp() {
        let key = "base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer, 12);
        assert_eq!(parsed.module, "gate_proj");
        assert_eq!(parsed.ab, "B");
    }

    #[test]
    fn test_parse_peft_key_invalid() {
        assert!(parse_peft_key("random.key.name").is_none());
        assert!(parse_peft_key("layers.abc.q_proj.lora_A.weight").is_none());
    }

    #[test]
    fn test_compute_lora_delta_known_values() -> Result<()> {
        let device = Device::Cpu;

        // x: [1, 2, 4] (batch=1, seq_len=2, in_features=4)
        let x = Tensor::new(&[[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], &device)?
            .unsqueeze(0)?;

        // A: [2, 4] (rank=2, in_features=4) — identity-like
        let a = Tensor::new(&[[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], &device)?;

        // B: [3, 2] (out_features=3, rank=2)
        let b = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;

        let proj = LoraProjectionWeights { a, b };
        let delta = compute_lora_delta(&x, &proj, 2.0)?;

        // x[0] = [1,0,0,0] -> x@A^T = [1,0] -> @B^T = [1,0,1] -> *2 = [2,0,2]
        // x[1] = [0,1,0,0] -> x@A^T = [0,1] -> @B^T = [0,1,1] -> *2 = [0,2,2]
        let vals = delta.squeeze(0)?.to_vec2::<f32>()?;
        assert!((vals[0][0] - 2.0).abs() < 1e-5);
        assert!((vals[0][1] - 0.0).abs() < 1e-5);
        assert!((vals[0][2] - 2.0).abs() < 1e-5);
        assert!((vals[1][0] - 0.0).abs() < 1e-5);
        assert!((vals[1][1] - 2.0).abs() < 1e-5);
        assert!((vals[1][2] - 2.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_linear_with_lora_adds_delta() -> Result<()> {
        let device = Device::Cpu;

        // x: [1, 1, 4]
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0, 4.0]], &device)?.unsqueeze(0)?;

        // W: [3, 4] — base weight
        let w = Tensor::zeros((3, 4), DType::F32, &device)?;

        // A: [2, 4], B: [3, 2]
        let a = Tensor::new(&[[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], &device)?;
        let b = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [0.5, 0.5]], &device)?;

        let proj = LoraProjectionWeights { a, b };

        // Without LoRA: output should be all zeros (zero weight)
        let out_base = linear_with_lora(&x, &w, None, 1.0)?;
        let vals = out_base.squeeze(0)?.to_vec2::<f32>()?;
        assert!((vals[0][0]).abs() < 1e-5);

        // With LoRA (scale=1.0):
        // x@A^T = [1,2] (first two elements of x), @B^T = [1, 2, 1.5], *1.0
        let out_lora = linear_with_lora(&x, &w, Some(&proj), 1.0)?;
        let vals = out_lora.squeeze(0)?.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.0).abs() < 1e-5);
        assert!((vals[0][1] - 2.0).abs() < 1e-5);
        assert!((vals[0][2] - 1.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_load_from_directory() -> Result<()> {
        // Create a temporary directory with mock adapter files
        let dir = tempfile::tempdir()?;
        let adapter_dir = dir.path();

        // Write adapter_config.json
        let config = serde_json::json!({
            "r": 4,
            "lora_alpha": 8.0,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM"
        });
        std::fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&config)?,
        )?;

        // Create minimal safetensors with A/B for layer 0 q_proj and v_proj
        let rank = 4usize;
        let in_features = 8usize;
        let out_features = 8usize;

        let mut tensors: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = Vec::new();

        // Helper: create f32 tensor data
        let make_data = |rows: usize, cols: usize| -> Vec<u8> {
            let vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        };

        // q_proj A: [rank, in_features]
        tensors.push((
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            make_data(rank, in_features),
            vec![rank, in_features],
            safetensors::Dtype::F32,
        ));
        // q_proj B: [out_features, rank]
        tensors.push((
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            make_data(out_features, rank),
            vec![out_features, rank],
            safetensors::Dtype::F32,
        ));
        // v_proj A: [rank, in_features]
        tensors.push((
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            make_data(rank, in_features),
            vec![rank, in_features],
            safetensors::Dtype::F32,
        ));
        // v_proj B: [out_features, rank]
        tensors.push((
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            make_data(out_features, rank),
            vec![out_features, rank],
            safetensors::Dtype::F32,
        ));

        // Serialize to safetensors format
        let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, data, shape, dtype)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone()))
            .collect();

        let serialized = safetensors::tensor::serialize(refs, None)?;
        std::fs::write(adapter_dir.join("adapter_model.safetensors"), &serialized)?;

        // Load
        let device = Device::Cpu;
        let weights = LoraWeights::load(adapter_dir, 1, &device)?;

        assert_eq!(weights.rank, 4);
        assert!((weights.alpha - 8.0).abs() < 1e-5);
        assert!((weights.scale - 2.0).abs() < 1e-5);
        assert_eq!(weights.layers.len(), 1);

        let layer = &weights.layers[0];
        assert!(layer.q_proj.is_some());
        assert!(layer.k_proj.is_none());
        assert!(layer.v_proj.is_some());
        assert!(layer.o_proj.is_none());

        // Verify shapes
        let q_a = &layer.q_proj.as_ref().unwrap().a;
        assert_eq!(q_a.dims(), &[rank, in_features]);
        let q_b = &layer.q_proj.as_ref().unwrap().b;
        assert_eq!(q_b.dims(), &[out_features, rank]);

        Ok(())
    }
}

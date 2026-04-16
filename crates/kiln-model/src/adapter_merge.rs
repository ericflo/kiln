//! Adapter merging via weighted (linear-interpolation) averaging.
//!
//! Given a set of `(adapter, weight)` pairs that share the same rank,
//! target modules, and tensor layout, produces a new PEFT-compatible
//! LoRA adapter whose A/B matrices are the weighted sum of the inputs:
//!
//! ```text
//! merged_A = Σᵢ wᵢ · Aᵢ
//! merged_B = Σᵢ wᵢ · Bᵢ
//! ```
//!
//! This is the "weighted average" / linear-interpolation merge mode.
//! TIES and concatenation merging are deferred to follow-up PRs.
//!
//! All merging is performed in `f32` on CPU using raw safetensors I/O,
//! so it does not depend on candle / CUDA and can be unit-tested without
//! a GPU.

use anyhow::{Context, Result, anyhow, bail};
use safetensors::Dtype;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// On-disk PEFT LoRA adapter loaded into memory as raw `f32` tensors.
///
/// `config` is the parsed `adapter_config.json` as a `serde_json::Value`,
/// preserving every field (including ones Kiln does not interpret) so the
/// merged output stays compatible with downstream PEFT consumers.
///
/// `tensors` is keyed by the safetensors weight name (e.g.
/// `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`) and
/// holds row-major `f32` data plus the original shape. We use a
/// `BTreeMap` so saved safetensors files are deterministic.
#[derive(Debug)]
pub struct PeftLora {
    pub config: Value,
    pub tensors: BTreeMap<String, MergeTensor>,
}

/// Raw `f32` tensor data plus shape, used for merge arithmetic.
#[derive(Clone, Debug)]
pub struct MergeTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl MergeTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }
}

impl PeftLora {
    /// Load a PEFT adapter directory into memory for merging.
    ///
    /// Reads `adapter_config.json` (preserved verbatim) and decodes every
    /// tensor in `adapter_model.safetensors` to `f32`.
    pub fn load(adapter_dir: &Path) -> Result<Self> {
        let config_path = adapter_dir.join("adapter_config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config: Value = serde_json::from_str(&config_str)
            .with_context(|| format!("failed to parse {}", config_path.display()))?;

        let st_path = adapter_dir.join("adapter_model.safetensors");
        let st_data = std::fs::read(&st_path)
            .with_context(|| format!("failed to read {}", st_path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .with_context(|| format!("failed to deserialize {}", st_path.display()))?;

        let mut out: BTreeMap<String, MergeTensor> = BTreeMap::new();
        for name in tensors.names() {
            let view = tensors
                .tensor(name)
                .with_context(|| format!("missing tensor view for {name}"))?;
            let shape = view.shape().to_vec();
            let data = decode_tensor_to_f32(view.dtype(), view.data())
                .with_context(|| format!("decoding tensor {name}"))?;
            let expected = shape.iter().product::<usize>();
            if data.len() != expected {
                bail!(
                    "tensor {name} has shape {:?} ({expected} elements) but decoded {} values",
                    shape,
                    data.len()
                );
            }
            out.insert(name.to_string(), MergeTensor { shape, data });
        }

        Ok(Self {
            config,
            tensors: out,
        })
    }

    /// Save this adapter to `adapter_dir` in PEFT-compatible format.
    ///
    /// Writes `adapter_config.json` (pretty-printed) and
    /// `adapter_model.safetensors`. All tensors are stored as `f32`.
    pub fn save(&self, adapter_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(adapter_dir)
            .with_context(|| format!("creating adapter dir {}", adapter_dir.display()))?;

        let config_path = adapter_dir.join("adapter_config.json");
        let config_str = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(&config_path, config_str)
            .with_context(|| format!("writing {}", config_path.display()))?;

        // Encode each tensor as little-endian f32 bytes for safetensors.
        let mut byte_storage: Vec<(String, Vec<usize>, Vec<u8>)> =
            Vec::with_capacity(self.tensors.len());
        for (name, t) in &self.tensors {
            let mut bytes = Vec::with_capacity(t.data.len() * 4);
            for v in &t.data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            byte_storage.push((name.clone(), t.shape.clone(), bytes));
        }

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = byte_storage
            .iter()
            .map(|(name, shape, bytes)| {
                let view = safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map_err(|e| anyhow!("building safetensors view for {name}: {e}"))?;
                Ok::<_, anyhow::Error>((name.clone(), view))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone()))
            .collect();

        let serialized = safetensors::tensor::serialize(refs, &None)
            .context("serializing merged safetensors")?;
        let st_path = adapter_dir.join("adapter_model.safetensors");
        std::fs::write(&st_path, serialized)
            .with_context(|| format!("writing {}", st_path.display()))?;

        Ok(())
    }

    /// Convenience accessor for `r` from the parsed config.
    pub fn rank(&self) -> Option<u64> {
        self.config.get("r")?.as_u64()
    }

    /// Convenience accessor for `target_modules` (array of strings) from
    /// the parsed config.
    pub fn target_modules(&self) -> Vec<String> {
        self.config
            .get("target_modules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Convenience accessor for `base_model_name_or_path` from the parsed
    /// config (PEFT's canonical field name).
    pub fn base_model(&self) -> Option<String> {
        self.config
            .get("base_model_name_or_path")
            .and_then(|v| v.as_str())
            .map(String::from)
    }
}

/// Linearly interpolate (weighted-average) a set of LoRA adapters.
///
/// Each input is a `(&PeftLora, weight)` pair. The output adapter has:
/// - the same `adapter_config.json` as `adapters[0]` (preserving fields
///   PEFT cares about that Kiln does not interpret);
/// - tensors equal to the elementwise weighted sum of the inputs:
///   `out[k] = Σᵢ wᵢ · adapters[i].tensors[k]`.
///
/// Validation (returns `Err` with an actionable message):
/// - at least one adapter is supplied;
/// - all adapters share the same `r` (rank) — required for v1 merge;
/// - all adapters share the same set of `target_modules`;
/// - all adapters share the same `base_model_name_or_path` (when present
///   on more than one input);
/// - all adapters have identical tensor key sets and matching shapes.
pub fn merge_linear(adapters: &[(&PeftLora, f32)]) -> Result<PeftLora> {
    if adapters.is_empty() {
        bail!("merge_linear requires at least one source adapter");
    }

    let (first, _) = adapters[0];
    let first_rank = first.rank();
    let first_targets = first.target_modules();
    let first_base = first.base_model();

    let mut keys: Vec<&String> = first.tensors.keys().collect();
    keys.sort();

    // Validate every adapter against the first.
    for (idx, (adapter, _)) in adapters.iter().enumerate().skip(1) {
        if adapter.rank() != first_rank {
            bail!(
                "adapter rank mismatch: source[0] has r={:?}, source[{idx}] has r={:?} \
                 (linear interpolation merge requires identical rank)",
                first_rank,
                adapter.rank()
            );
        }
        let targets = adapter.target_modules();
        if !same_string_set(&first_targets, &targets) {
            bail!(
                "adapter target_modules mismatch: source[0] has {:?}, source[{idx}] has {:?}",
                first_targets,
                targets
            );
        }
        if let (Some(a), Some(b)) = (&first_base, &adapter.base_model()) {
            if a != b {
                bail!(
                    "adapter base_model_name_or_path mismatch: source[0] is {a:?}, \
                     source[{idx}] is {b:?}"
                );
            }
        }

        // Tensor key sets must match exactly.
        let other_keys: Vec<&String> = adapter.tensors.keys().collect();
        if first.tensors.len() != adapter.tensors.len() {
            bail!(
                "adapter tensor count mismatch: source[0] has {} tensors, source[{idx}] has {}",
                first.tensors.len(),
                adapter.tensors.len()
            );
        }
        for key in &other_keys {
            if !first.tensors.contains_key(key.as_str()) {
                bail!(
                    "adapter tensor key mismatch: source[{idx}] has tensor {:?} not present in source[0]",
                    key
                );
            }
        }

        // Shape match per tensor.
        for key in &keys {
            let a = &first.tensors[key.as_str()];
            let b = adapter.tensors.get(key.as_str()).ok_or_else(|| {
                anyhow!("adapter tensor key mismatch: source[{idx}] missing tensor {:?}", key)
            })?;
            if a.shape != b.shape {
                bail!(
                    "tensor shape mismatch for {key:?}: source[0] is {:?}, source[{idx}] is {:?}",
                    a.shape,
                    b.shape
                );
            }
        }
    }

    // Compute weighted sum per tensor.
    let mut merged: BTreeMap<String, MergeTensor> = BTreeMap::new();
    for key in &keys {
        let key_str = key.as_str();
        let template = &first.tensors[key_str];
        let n = template.numel();
        let mut acc = vec![0.0_f32; n];
        for (adapter, weight) in adapters {
            let t = &adapter.tensors[key_str];
            // Defensive: shape was validated above, but recheck length.
            if t.data.len() != n {
                bail!(
                    "internal error: tensor {key:?} has {} elements, expected {n}",
                    t.data.len()
                );
            }
            let w = *weight;
            for (i, v) in t.data.iter().enumerate() {
                acc[i] += w * v;
            }
        }
        merged.insert(
            key_str.to_string(),
            MergeTensor {
                shape: template.shape.clone(),
                data: acc,
            },
        );
    }

    Ok(PeftLora {
        config: first.config.clone(),
        tensors: merged,
    })
}

fn same_string_set(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort();
    b_sorted.sort();
    a_sorted == b_sorted
}

fn decode_tensor_to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                bail!("F32 tensor byte length {} is not a multiple of 4", bytes.len());
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().expect("chunk size 4");
                out.push(f32::from_le_bytes(arr));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                bail!("F16 tensor byte length {} is not a multiple of 2", bytes.len());
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().expect("chunk size 2");
                out.push(f16_le_bytes_to_f32(arr));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                bail!("BF16 tensor byte length {} is not a multiple of 2", bytes.len());
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().expect("chunk size 2");
                out.push(bf16_le_bytes_to_f32(arr));
            }
            Ok(out)
        }
        other => bail!("unsupported safetensors dtype for adapter merge: {other:?}"),
    }
}

fn f16_le_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut m = mant;
            let mut e: i32 = 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + e) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn bf16_le_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    // bf16 occupies the upper 16 bits of an f32; lower 16 bits are zero.
    let bits = (u16::from_le_bytes(bytes) as u32) << 16;
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_adapter(
        rank: usize,
        a_data: Vec<f32>,
        b_data: Vec<f32>,
    ) -> PeftLora {
        // Single layer, single target module (q_proj), in_features=4, out_features=3.
        // A: [rank, 4], B: [3, rank]
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            MergeTensor {
                shape: vec![rank, 4],
                data: a_data,
            },
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            MergeTensor {
                shape: vec![3, rank],
                data: b_data,
            },
        );

        let config = json!({
            "r": rank,
            "lora_alpha": (rank * 2) as f32,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B"
        });

        PeftLora { config, tensors }
    }

    #[test]
    fn test_merge_linear_weighted_average() -> Result<()> {
        // rank 2: A is [2,4]=8 elements, B is [3,2]=6 elements.
        let a1 = vec![1.0_f32; 8];
        let b1 = vec![2.0_f32; 6];

        let a2 = vec![3.0_f32; 8];
        let b2 = vec![4.0_f32; 6];

        let adapter1 = make_adapter(2, a1, b1);
        let adapter2 = make_adapter(2, a2, b2);

        let merged = merge_linear(&[(&adapter1, 0.5), (&adapter2, 0.5)])?;

        // Check config preserved (same rank).
        assert_eq!(merged.rank(), Some(2));
        assert_eq!(merged.target_modules(), vec!["q_proj".to_string()]);

        let merged_a = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        let merged_b = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];

        // 0.5*1 + 0.5*3 = 2 for A
        for v in &merged_a.data {
            assert!((*v - 2.0).abs() < 1e-6, "expected 2.0, got {v}");
        }
        // 0.5*2 + 0.5*4 = 3 for B
        for v in &merged_b.data {
            assert!((*v - 3.0).abs() < 1e-6, "expected 3.0, got {v}");
        }

        // Shapes preserved.
        assert_eq!(merged_a.shape, vec![2, 4]);
        assert_eq!(merged_b.shape, vec![3, 2]);

        Ok(())
    }

    #[test]
    fn test_merge_linear_uneven_weights() -> Result<()> {
        let a1 = vec![1.0_f32; 8];
        let b1 = vec![10.0_f32; 6];
        let a2 = vec![5.0_f32; 8];
        let b2 = vec![20.0_f32; 6];

        let adapter1 = make_adapter(2, a1, b1);
        let adapter2 = make_adapter(2, a2, b2);

        // 0.25 * a1 + 0.75 * a2 = 0.25 + 3.75 = 4.0
        // 0.25 * b1 + 0.75 * b2 = 2.5 + 15.0 = 17.5
        let merged = merge_linear(&[(&adapter1, 0.25), (&adapter2, 0.75)])?;

        let merged_a = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        let merged_b = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];

        for v in &merged_a.data {
            assert!((*v - 4.0).abs() < 1e-6, "expected 4.0, got {v}");
        }
        for v in &merged_b.data {
            assert!((*v - 17.5).abs() < 1e-6, "expected 17.5, got {v}");
        }

        Ok(())
    }

    #[test]
    fn test_merge_linear_three_adapters_uniform() -> Result<()> {
        let adapter1 = make_adapter(2, vec![3.0_f32; 8], vec![6.0_f32; 6]);
        let adapter2 = make_adapter(2, vec![6.0_f32; 8], vec![9.0_f32; 6]);
        let adapter3 = make_adapter(2, vec![9.0_f32; 8], vec![12.0_f32; 6]);

        let third = 1.0 / 3.0;
        let merged = merge_linear(&[
            (&adapter1, third),
            (&adapter2, third),
            (&adapter3, third),
        ])?;

        let merged_a = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        // (3+6+9)/3 = 6
        for v in &merged_a.data {
            assert!((*v - 6.0).abs() < 1e-5, "expected 6.0, got {v}");
        }
        let merged_b = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
        // (6+9+12)/3 = 9
        for v in &merged_b.data {
            assert!((*v - 9.0).abs() < 1e-5, "expected 9.0, got {v}");
        }

        Ok(())
    }

    #[test]
    fn test_merge_linear_rank_mismatch_errors() {
        let a1 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        // rank 4: A is [4,4]=16, B is [3,4]=12
        let a2 = make_adapter(4, vec![1.0_f32; 16], vec![1.0_f32; 12]);

        let err = merge_linear(&[(&a1, 0.5), (&a2, 0.5)]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("rank mismatch"),
            "expected rank mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_merge_linear_target_modules_mismatch_errors() {
        let a1 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        let mut a2 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        // Mutate config to change target_modules.
        a2.config["target_modules"] = json!(["k_proj"]);

        let err = merge_linear(&[(&a1, 0.5), (&a2, 0.5)]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("target_modules mismatch"),
            "expected target_modules mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_merge_linear_base_model_mismatch_errors() {
        let a1 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        let mut a2 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        a2.config["base_model_name_or_path"] = json!("meta-llama/Llama-3-8B");

        let err = merge_linear(&[(&a1, 0.5), (&a2, 0.5)]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("base_model_name_or_path mismatch"),
            "expected base model mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_merge_linear_empty_errors() {
        let err = merge_linear(&[]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("at least one source adapter"),
            "expected empty error, got: {msg}"
        );
    }

    #[test]
    fn test_merge_linear_single_source_passthrough() -> Result<()> {
        let adapter = make_adapter(2, vec![7.0_f32; 8], vec![11.0_f32; 6]);
        let merged = merge_linear(&[(&adapter, 1.0)])?;
        let merged_a = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        for v in &merged_a.data {
            assert!((*v - 7.0).abs() < 1e-6);
        }
        let merged_b = &merged.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
        for v in &merged_b.data {
            assert!((*v - 11.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_save_and_load_roundtrip() -> Result<()> {
        let dir = tempdir()?;
        let adapter_dir = dir.path();

        let adapter = make_adapter(2, (0..8).map(|i| i as f32 * 0.5).collect(), vec![3.0_f32; 6]);
        adapter.save(adapter_dir)?;

        // Files written.
        assert!(adapter_dir.join("adapter_config.json").exists());
        assert!(adapter_dir.join("adapter_model.safetensors").exists());

        // Round-trip.
        let loaded = PeftLora::load(adapter_dir)?;
        assert_eq!(loaded.rank(), Some(2));
        assert_eq!(loaded.target_modules(), vec!["q_proj".to_string()]);
        assert_eq!(loaded.base_model().as_deref(), Some("Qwen/Qwen3.5-4B"));
        assert_eq!(loaded.tensors.len(), 2);

        let a = &loaded.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        assert_eq!(a.shape, vec![2, 4]);
        for (i, v) in a.data.iter().enumerate() {
            assert!((*v - (i as f32 * 0.5)).abs() < 1e-6);
        }
        let b = &loaded.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
        assert_eq!(b.shape, vec![3, 2]);
        for v in &b.data {
            assert!((*v - 3.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_merge_then_save_and_load() -> Result<()> {
        let dir = tempdir()?;
        let merged_dir = dir.path().join("merged");

        let adapter1 = make_adapter(2, vec![1.0_f32; 8], vec![2.0_f32; 6]);
        let adapter2 = make_adapter(2, vec![3.0_f32; 8], vec![4.0_f32; 6]);

        let merged = merge_linear(&[(&adapter1, 0.5), (&adapter2, 0.5)])?;
        merged.save(&merged_dir)?;

        let loaded = PeftLora::load(&merged_dir)?;
        let a = &loaded.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        for v in &a.data {
            assert!((*v - 2.0).abs() < 1e-6);
        }
        let b = &loaded.tensors
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
        for v in &b.data {
            assert!((*v - 3.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_merged_loadable_by_lora_weights() -> Result<()> {
        // The merged adapter should be loadable by the existing
        // LoraWeights::load() pipeline (which is what production code
        // uses to apply adapters during inference). This guards against
        // accidental format drift.
        use crate::lora_loader::LoraWeights;
        use candle_core::Device;

        let dir = tempdir()?;
        let merged_dir = dir.path().join("merged");

        let a1 = make_adapter(2, vec![1.0_f32; 8], vec![1.0_f32; 6]);
        let a2 = make_adapter(2, vec![3.0_f32; 8], vec![3.0_f32; 6]);
        let merged = merge_linear(&[(&a1, 0.5), (&a2, 0.5)])?;
        merged.save(&merged_dir)?;

        let device = Device::Cpu;
        let loaded = LoraWeights::load(&merged_dir, 1, &device)?;
        assert_eq!(loaded.rank, 2);
        assert!(loaded.layers[0].q_proj.is_some());

        Ok(())
    }
}

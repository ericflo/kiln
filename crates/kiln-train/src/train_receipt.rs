//! Stable training receipts for GRPO/SFT adapter runs.
//!
//! This is intentionally separate from `receipt.json`: `receipt.json` is the
//! older high-level reproducibility artifact used by distillation recipes,
//! while `train_receipt.json` is the machine-readable forensic record that cap
//! scripts can parse without scraping logs.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const TRAIN_RECEIPT_FILENAME: &str = "train_receipt.json";
pub const TRAIN_RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainReceiptStatus {
    Success,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainReceipt {
    pub schema_version: u32,
    pub receipt_type: String,
    pub adapter_name: String,
    pub produced_at: String,
    pub status: TrainReceiptStatus,
    pub failure_reason: Option<String>,
    pub kiln: KilnSourceReceipt,
    pub model: ModelReceipt,
    pub tokenizer: TokenizerReceipt,
    pub adapters: AdapterReceiptSet,
    pub training_data: TrainingDataReceipt,
    pub hyperparameters: HyperparameterReceipt,
    pub grpo: Option<GrpoReceipt>,
    pub echo: EchoReceipt,
    pub no_policy_loss: bool,
    pub data: DataStatsReceipt,
    pub rewards: RewardStatsReceipt,
    pub token_counts: TokenCountReceipt,
    pub runtime: RuntimeReceipt,
    pub lora_delta_norms: Vec<LoraDeltaNormSummary>,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KilnSourceReceipt {
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub git_source: Option<String>,
    pub package_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelReceipt {
    pub path: Option<String>,
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerReceipt {
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterReceiptSet {
    pub base: AdapterFileReceipt,
    pub output: AdapterFileReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterFileReceipt {
    pub path: Option<String>,
    pub adapter_model_sha256: Option<String>,
    pub adapter_model_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingDataReceipt {
    pub source: String,
    pub path: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperparameterReceipt {
    pub mode: String,
    pub rank: usize,
    pub alpha: f32,
    pub alpha_over_rank: Option<f32>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoReceipt {
    pub kl_coeff: f64,
    pub clip_epsilon: f64,
    pub clip_eps_high: Option<f64>,
    pub dynamic_sampling: bool,
    pub dynamic_groups_filtered: usize,
    pub advantage_mode: serde_json::Value,
    pub loss_aggregation: serde_json::Value,
    pub kl_estimator: serde_json::Value,
    pub is_level: serde_json::Value,
    pub reference_policy: serde_json::Value,
    pub entropy_aware_kl_quantile: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EchoReceipt {
    pub enabled: bool,
    pub lambda: Option<f64>,
    pub env_mask_mode: Option<String>,
    pub warning_filter: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DataStatsReceipt {
    pub examples_read: usize,
    pub examples_filtered: usize,
    pub examples_trained: usize,
    pub groups_read: usize,
    pub groups_filtered: usize,
    pub groups_trained: usize,
    pub completions_read: usize,
    pub completions_trained: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardStatsReceipt {
    pub count: usize,
    pub mean: Option<f64>,
    pub stdev: Option<f64>,
    pub group_variance_histogram: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramBucket {
    pub label: String,
    pub min_inclusive: Option<f64>,
    pub max_inclusive: Option<f64>,
    pub count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TokenCountReceipt {
    pub action_tokens: u64,
    pub env_tokens: u64,
    pub context_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeReceipt {
    pub wall_clock_ms: u64,
    pub peak_vram_mib: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraDeltaNormSummary {
    pub module: String,
    pub pair_count: usize,
    pub a_l2_mean: f64,
    pub a_l2_max: f64,
    pub b_l2_mean: f64,
    pub b_l2_max: f64,
    pub delta_l2_upper_bound_mean: f64,
    pub delta_l2_upper_bound_max: f64,
}

impl Default for RewardStatsReceipt {
    fn default() -> Self {
        Self {
            count: 0,
            mean: None,
            stdev: None,
            group_variance_histogram: variance_histogram(&[]),
        }
    }
}

impl TrainReceipt {
    pub fn new(
        adapter_name: impl Into<String>,
        mode: impl Into<String>,
        model_config: &ModelConfig,
        tokenizer: &KilnTokenizer,
        hyperparameters: HyperparameterReceipt,
        config: serde_json::Value,
    ) -> Self {
        Self {
            schema_version: TRAIN_RECEIPT_SCHEMA_VERSION,
            receipt_type: "kiln_train_receipt".to_string(),
            adapter_name: adapter_name.into(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            status: TrainReceiptStatus::Success,
            failure_reason: None,
            kiln: detect_kiln_source(),
            model: ModelReceipt {
                path: detect_model_path(),
                config_hash: sha256_json_serializable(model_config),
            },
            tokenizer: TokenizerReceipt {
                config_hash: tokenizer.config_sha256().ok(),
            },
            adapters: AdapterReceiptSet {
                base: AdapterFileReceipt::none(),
                output: AdapterFileReceipt::none(),
            },
            training_data: TrainingDataReceipt {
                source: mode.into(),
                path: None,
                sha256: None,
            },
            hyperparameters,
            grpo: None,
            echo: EchoReceipt::disabled(),
            no_policy_loss: false,
            data: DataStatsReceipt::default(),
            rewards: RewardStatsReceipt::default(),
            token_counts: TokenCountReceipt::default(),
            runtime: RuntimeReceipt {
                wall_clock_ms: 0,
                peak_vram_mib: None,
            },
            lora_delta_norms: Vec::new(),
            config,
        }
    }

    pub fn mark_failed(mut self, reason: impl Into<String>) -> Self {
        self.status = TrainReceiptStatus::Failed;
        self.failure_reason = Some(reason.into());
        self
    }

    pub fn write_to_adapter_dir(&self, adapter_dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(adapter_dir).with_context(|| {
            format!(
                "create adapter dir {} for train receipt",
                adapter_dir.display()
            )
        })?;
        let path = adapter_dir.join(TRAIN_RECEIPT_FILENAME);
        let json = serde_json::to_string_pretty(self).context("serialize train receipt")?;
        std::fs::write(&path, json)
            .with_context(|| format!("write train receipt {}", path.display()))?;
        Ok(path)
    }

    pub fn read_from_adapter_dir(adapter_dir: &Path) -> Result<Option<Self>> {
        let path = adapter_dir.join(TRAIN_RECEIPT_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let bytes =
            std::fs::read(&path).with_context(|| format!("read train receipt {}", path.display()))?;
        let receipt = serde_json::from_slice(&bytes)
            .with_context(|| format!("deserialize train receipt {}", path.display()))?;
        Ok(Some(receipt))
    }
}

impl AdapterFileReceipt {
    pub fn none() -> Self {
        Self {
            path: None,
            adapter_model_sha256: None,
            adapter_model_bytes: None,
        }
    }
}

impl EchoReceipt {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            lambda: None,
            env_mask_mode: None,
            warning_filter: None,
        }
    }
}

pub fn adapter_file_receipt(adapter_dir: Option<&Path>) -> AdapterFileReceipt {
    let Some(dir) = adapter_dir else {
        return AdapterFileReceipt::none();
    };
    let adapter_model = dir.join("adapter_model.safetensors");
    let (hash, bytes) = if adapter_model.exists() {
        (
            sha256_file(&adapter_model).ok(),
            adapter_model.metadata().ok().map(|m| m.len()),
        )
    } else {
        (None, None)
    };
    AdapterFileReceipt {
        path: Some(dir.display().to_string()),
        adapter_model_sha256: hash,
        adapter_model_bytes: bytes,
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open {} for sha256", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {} for sha256", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("sha256:{}", hex_digest(h.finalize().as_slice())))
}

pub fn sha256_json_value(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let digest = Sha256::digest(&bytes);
    format!("sha256:{}", hex_digest(digest.as_slice()))
}

pub fn sha256_json_serializable<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .map(|value| sha256_json_value(&value))
}

pub fn reward_stats_from_groups<'a, I>(groups: I) -> RewardStatsReceipt
where
    I: IntoIterator<Item = &'a [f64]>,
{
    let mut rewards = Vec::new();
    let mut variances = Vec::new();
    for group_rewards in groups {
        rewards.extend_from_slice(group_rewards);
        if !group_rewards.is_empty() {
            variances.push(population_variance(group_rewards));
        }
    }
    let count = rewards.len();
    if count == 0 {
        return RewardStatsReceipt::default();
    }
    let mean = rewards.iter().sum::<f64>() / count as f64;
    let variance = rewards
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;
    RewardStatsReceipt {
        count,
        mean: Some(mean),
        stdev: Some(variance.sqrt()),
        group_variance_histogram: variance_histogram(&variances),
    }
}

pub fn lora_delta_norm_summary_from_adapter(
    adapter_dir: &Path,
    alpha_over_rank: f64,
) -> Result<Vec<LoraDeltaNormSummary>> {
    let adapter_model = adapter_dir.join("adapter_model.safetensors");
    if !adapter_model.exists() {
        return Ok(Vec::new());
    }
    let tensors = candle_core::safetensors::load(&adapter_model, &Device::Cpu)
        .with_context(|| format!("load adapter tensors {}", adapter_model.display()))?;

    let mut pairs: BTreeMap<(usize, String), ProjectionPair> = BTreeMap::new();
    for (key, tensor) in tensors {
        let Some(parsed) = parse_peft_lora_key(&key) else {
            continue;
        };
        let norm = tensor_l2_norm(&tensor)
            .with_context(|| format!("compute l2 norm for adapter tensor {key}"))?;
        let pair = pairs
            .entry((parsed.layer, parsed.module))
            .or_default();
        match parsed.kind {
            LoraTensorKind::A => pair.a_l2 = Some(norm),
            LoraTensorKind::B => pair.b_l2 = Some(norm),
        }
    }

    let mut by_module: BTreeMap<String, ModuleNormAccumulator> = BTreeMap::new();
    for ((_layer, module), pair) in pairs {
        let (Some(a_l2), Some(b_l2)) = (pair.a_l2, pair.b_l2) else {
            continue;
        };
        by_module
            .entry(module)
            .or_default()
            .push(a_l2, b_l2, alpha_over_rank);
    }

    Ok(by_module
        .into_iter()
        .map(|(module, acc)| acc.finish(module))
        .collect())
}

fn detect_kiln_source() -> KilnSourceReceipt {
    let repo_root = std::env::var("KILN_REPO_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).map(Path::to_path_buf));

    let (git_commit, git_dirty, git_source) = if let Some(root) = repo_root.as_deref() {
        let commit = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .filter(|s| !s.is_empty());
        let dirty = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| !out.stdout.is_empty());
        let source = commit.as_ref().map(|_| root.display().to_string());
        (commit, dirty, source)
    } else {
        (None, None, None)
    };

    KilnSourceReceipt {
        git_commit: git_commit.or_else(|| Some(crate::replay::kiln_commit())),
        git_dirty,
        git_source,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

fn detect_model_path() -> Option<String> {
    ["KILN_MODEL_PATH", "MODEL_PATH", "HF_MODEL_PATH"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok())
        .filter(|value| !value.trim().is_empty())
}

fn variance_histogram(variances: &[f64]) -> Vec<HistogramBucket> {
    let specs = [
        ("zero", Some(0.0), Some(0.0)),
        ("tiny", Some(f64::MIN_POSITIVE), Some(1e-6)),
        ("low", Some(1e-6), Some(0.01)),
        ("medium", Some(0.01), Some(0.25)),
        ("high", Some(0.25), Some(1.0)),
        ("extreme", Some(1.0), None),
    ];
    specs
        .into_iter()
        .map(|(label, min, max)| {
            let count = variances
                .iter()
                .filter(|value| histogram_contains(**value, min, max))
                .count();
            HistogramBucket {
                label: label.to_string(),
                min_inclusive: min,
                max_inclusive: max,
                count,
            }
        })
        .collect()
}

fn histogram_contains(value: f64, min: Option<f64>, max: Option<f64>) -> bool {
    if value == 0.0 {
        return min == Some(0.0) && max == Some(0.0);
    }
    if let Some(min) = min {
        if value <= min {
            return false;
        }
    }
    if let Some(max) = max {
        if value > max {
            return false;
        }
    }
    true
}

fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64
}

fn tensor_l2_norm(tensor: &Tensor) -> Result<f64> {
    let sum_sq = tensor
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok((sum_sq as f64).sqrt())
}

fn parse_peft_lora_key(key: &str) -> Option<ParsedLoraKey> {
    let parts: Vec<&str> = key.split('.').collect();
    let layer_pos = parts.iter().position(|part| *part == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse().ok()?;
    let lora_pos = parts
        .iter()
        .position(|part| *part == "lora_A" || *part == "lora_B")?;
    if parts.get(lora_pos + 1)? != &"weight" || lora_pos == 0 {
        return None;
    }
    let module = parts.get(lora_pos - 1)?.to_string();
    let kind = match parts[lora_pos] {
        "lora_A" => LoraTensorKind::A,
        "lora_B" => LoraTensorKind::B,
        _ => return None,
    };
    Some(ParsedLoraKey {
        layer,
        module,
        kind,
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct ParsedLoraKey {
    layer: usize,
    module: String,
    kind: LoraTensorKind,
}

#[derive(Clone, Copy)]
enum LoraTensorKind {
    A,
    B,
}

#[derive(Default)]
struct ProjectionPair {
    a_l2: Option<f64>,
    b_l2: Option<f64>,
}

#[derive(Default)]
struct ModuleNormAccumulator {
    pair_count: usize,
    a_l2_sum: f64,
    a_l2_max: f64,
    b_l2_sum: f64,
    b_l2_max: f64,
    delta_l2_upper_bound_sum: f64,
    delta_l2_upper_bound_max: f64,
}

impl ModuleNormAccumulator {
    fn push(&mut self, a_l2: f64, b_l2: f64, alpha_over_rank: f64) {
        let delta = a_l2 * b_l2 * alpha_over_rank;
        self.pair_count += 1;
        self.a_l2_sum += a_l2;
        self.a_l2_max = self.a_l2_max.max(a_l2);
        self.b_l2_sum += b_l2;
        self.b_l2_max = self.b_l2_max.max(b_l2);
        self.delta_l2_upper_bound_sum += delta;
        self.delta_l2_upper_bound_max = self.delta_l2_upper_bound_max.max(delta);
    }

    fn finish(self, module: String) -> LoraDeltaNormSummary {
        let denom = self.pair_count.max(1) as f64;
        LoraDeltaNormSummary {
            module,
            pair_count: self.pair_count,
            a_l2_mean: self.a_l2_sum / denom,
            a_l2_max: self.a_l2_max,
            b_l2_mean: self.b_l2_sum / denom,
            b_l2_max: self.b_l2_max,
            delta_l2_upper_bound_mean: self.delta_l2_upper_bound_sum / denom,
            delta_l2_upper_bound_max: self.delta_l2_upper_bound_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn train_receipt_success_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let receipt = TrainReceipt::new(
            "adapter-a",
            "sft",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(42),
            },
            serde_json::json!({"epochs": 1}),
        );
        let path = receipt.write_to_adapter_dir(dir.path())?;
        assert_eq!(path.file_name().and_then(|n| n.to_str()), Some(TRAIN_RECEIPT_FILENAME));
        let loaded = TrainReceipt::read_from_adapter_dir(dir.path())?.expect("receipt exists");
        assert_eq!(loaded.schema_version, TRAIN_RECEIPT_SCHEMA_VERSION);
        assert_eq!(loaded.status, TrainReceiptStatus::Success);
        assert_eq!(loaded.adapter_name, "adapter-a");
        assert_eq!(loaded.hyperparameters.rank, 8);
        Ok(())
    }

    #[test]
    fn train_receipt_failed_status_is_stable() -> Result<()> {
        let dir = tempdir()?;
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer()?;
        let receipt = TrainReceipt::new(
            "adapter-b",
            "grpo",
            &model,
            &tokenizer,
            HyperparameterReceipt {
                mode: "grpo".to_string(),
                rank: 4,
                alpha: 8.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-5,
                epochs: 1,
                seed: None,
            },
            serde_json::json!({}),
        )
        .mark_failed("no valid GRPO groups after tokenization");
        receipt.write_to_adapter_dir(dir.path())?;
        let json = std::fs::read_to_string(dir.path().join(TRAIN_RECEIPT_FILENAME))?;
        assert!(json.contains("\"status\": \"failed\""));
        assert!(json.contains("no valid GRPO groups"));
        Ok(())
    }

    #[test]
    fn sha256_json_value_is_prefixed_and_stable() {
        let a = sha256_json_value(&serde_json::json!({"x": 1, "y": [true, false]}));
        let b = sha256_json_value(&serde_json::json!({"x": 1, "y": [true, false]}));
        assert!(a.starts_with("sha256:"));
        assert_eq!(a, b);
    }

    #[test]
    fn reward_stats_include_variance_histogram() {
        let groups = [vec![1.0, 0.0], vec![0.25, 0.25], vec![2.0, -2.0]];
        let slices: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
        let stats = reward_stats_from_groups(slices);
        assert_eq!(stats.count, 6);
        assert!(stats.mean.unwrap().abs() > 0.0);
        assert_eq!(stats.group_variance_histogram.len(), 6);
        assert_eq!(
            stats
                .group_variance_histogram
                .iter()
                .map(|b| b.count)
                .sum::<usize>(),
            3
        );
    }

    fn minimal_tokenizer() -> Result<KilnTokenizer> {
        let json = br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]"
            }
        }"#;
        KilnTokenizer::from_bytes(json).map_err(|err| anyhow::anyhow!("{err}"))
    }
}

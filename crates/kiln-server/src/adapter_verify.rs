//! Offline LoRA adapter verification for the `kiln adapter verify` CLI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use kiln_model::adapter_merge::{MergeTensor, PeftLora};

#[derive(Debug, Clone)]
pub struct AdapterVerifyOptions {
    pub input: String,
    pub adapter_dir: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub struct AdapterVerifyReceipt {
    pub status: &'static str,
    pub input: String,
    pub name: Option<String>,
    pub adapter_dir: Option<String>,
    pub adapter_path: String,
    pub checks: Vec<AdapterVerifyCheck>,
    pub files: AdapterVerifyFiles,
    pub lora: AdapterVerifyLora,
    pub tensor_summary: AdapterTensorSummary,
    pub logit_delta_summary: AdapterLogitDeltaSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<AdapterVerifyServerReceipt>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdapterVerifyCheck {
    pub name: &'static str,
    pub pass: bool,
    pub message: String,
}

#[derive(Debug, Default, Serialize)]
pub struct AdapterVerifyFiles {
    pub adapter_config_sha256: Option<String>,
    pub adapter_model_sha256: Option<String>,
    pub adapter_model_size_bytes: Option<u64>,
}

#[derive(Debug, Default, Serialize)]
pub struct AdapterVerifyLora {
    pub rank: Option<u64>,
    pub alpha: Option<f64>,
    pub alpha_over_rank: Option<f64>,
    pub target_modules: Vec<String>,
    pub base_model_name_or_path: Option<String>,
    pub tensor_count: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct AdapterTensorSummary {
    pub paired_projection_count: usize,
    pub tensor_count: usize,
    pub nonzero_tensor_count: usize,
    pub max_abs_weight: f64,
    pub l1_weight_sum: f64,
    pub l2_weight_norm: f64,
    pub lora_update_l2_upper_bound: f64,
}

#[derive(Debug, Serialize)]
pub struct AdapterLogitDeltaSummary {
    pub mode: &'static str,
    pub prompt: &'static str,
    pub measurable: bool,
    pub max_abs_delta_proxy: f64,
    pub l2_delta_proxy: f64,
}

impl Default for AdapterLogitDeltaSummary {
    fn default() -> Self {
        Self {
            mode: "offline_lora_delta_norm_proxy",
            prompt: DEFAULT_VERIFY_PROMPT,
            measurable: false,
            max_abs_delta_proxy: 0.0,
            l2_delta_proxy: 0.0,
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct AdapterVerifyServerReceipt {
    pub url: String,
    pub adapter_name: String,
    pub checks: Vec<AdapterVerifyCheck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_output: Option<String>,
}

pub const DEFAULT_VERIFY_PROMPT: &str =
    "In one short sentence, answer with the word kiln and one adjective.";

pub fn verify_adapter_offline(options: AdapterVerifyOptions) -> AdapterVerifyReceipt {
    let resolved = resolve_adapter_path(&options.input, options.adapter_dir.as_deref());
    let name = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    let mut receipt = AdapterVerifyReceipt {
        status: "failed",
        input: options.input,
        name,
        adapter_dir: options
            .adapter_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        adapter_path: resolved.display().to_string(),
        checks: Vec::new(),
        files: AdapterVerifyFiles::default(),
        lora: AdapterVerifyLora::default(),
        tensor_summary: AdapterTensorSummary::default(),
        logit_delta_summary: AdapterLogitDeltaSummary::default(),
        server: None,
    };

    let exists = resolved.exists() && resolved.is_dir();
    push_check(
        &mut receipt.checks,
        "adapter_directory_exists",
        exists,
        if exists {
            format!("adapter directory exists at {}", resolved.display())
        } else {
            format!("adapter directory does not exist at {}", resolved.display())
        },
    );
    if !exists {
        finalize_status(&mut receipt);
        return receipt;
    }

    let canonical = resolved.canonicalize().unwrap_or(resolved.clone());
    receipt.adapter_path = canonical.display().to_string();

    let config_path = canonical.join("adapter_config.json");
    let weights_path = canonical.join("adapter_model.safetensors");
    let has_config = config_path.is_file();
    let has_weights = weights_path.is_file();
    let mut layout_message = if has_config && has_weights {
        "adapter directory contains adapter_config.json and adapter_model.safetensors".to_string()
    } else {
        let mut missing = Vec::new();
        if !has_config {
            missing.push("adapter_config.json");
        }
        if !has_weights {
            missing.push("adapter_model.safetensors");
        }
        let mut msg = format!("missing required file(s): {}", missing.join(", "));
        if let Some(child) = find_single_nested_adapter_dir(&canonical) {
            msg.push_str(&format!(
                "; found nested adapter directory at {}",
                child.display()
            ));
        }
        msg
    };
    if has_config {
        receipt.files.adapter_config_sha256 = sha256_file_hex(&config_path).ok();
    }
    if has_weights {
        receipt.files.adapter_model_sha256 = sha256_file_hex(&weights_path).ok();
        receipt.files.adapter_model_size_bytes = std::fs::metadata(&weights_path)
            .ok()
            .map(|meta| meta.len());
    }
    push_check(
        &mut receipt.checks,
        "adapter_layout",
        has_config && has_weights,
        std::mem::take(&mut layout_message),
    );
    if !(has_config && has_weights) {
        finalize_status(&mut receipt);
        return receipt;
    }

    match read_config_metadata(&config_path) {
        Ok(config) => {
            receipt.lora.rank = config.rank;
            receipt.lora.alpha = config.alpha;
            receipt.lora.alpha_over_rank = config.alpha_over_rank;
            receipt.lora.target_modules = config.target_modules;
            receipt.lora.base_model_name_or_path = config.base_model_name_or_path;
            push_check(
                &mut receipt.checks,
                "adapter_config_json",
                true,
                "adapter_config.json parsed".to_string(),
            );
        }
        Err(err) => {
            push_check(
                &mut receipt.checks,
                "adapter_config_json",
                false,
                format!("adapter_config.json is invalid: {err}"),
            );
            finalize_status(&mut receipt);
            return receipt;
        }
    }

    let rank_ok = receipt.lora.rank.is_some_and(|rank| rank > 0);
    push_check(
        &mut receipt.checks,
        "rank_declared",
        rank_ok,
        match receipt.lora.rank {
            Some(rank) if rank > 0 => format!("rank declared as {rank}"),
            Some(rank) => format!("rank must be positive; got {rank}"),
            None => "adapter_config.json is missing integer field `r`".to_string(),
        },
    );

    let alpha_ok = receipt
        .lora
        .alpha
        .is_some_and(|alpha| alpha.is_finite() && alpha > 0.0);
    push_check(
        &mut receipt.checks,
        "alpha_declared",
        alpha_ok,
        match receipt.lora.alpha {
            Some(alpha) if alpha.is_finite() && alpha > 0.0 => {
                format!("lora_alpha declared as {alpha}")
            }
            Some(alpha) => format!("lora_alpha must be finite and positive; got {alpha}"),
            None => "adapter_config.json is missing numeric field `lora_alpha`".to_string(),
        },
    );

    let targets_ok = !receipt.lora.target_modules.is_empty();
    push_check(
        &mut receipt.checks,
        "target_modules_declared",
        targets_ok,
        if targets_ok {
            format!(
                "target_modules declared: {}",
                receipt.lora.target_modules.join(", ")
            )
        } else {
            "adapter_config.json must declare at least one target module".to_string()
        },
    );

    match PeftLora::load(&canonical) {
        Ok(adapter) => {
            receipt.lora.tensor_count = adapter.tensors.len();
            push_check(
                &mut receipt.checks,
                "adapter_model_safetensors",
                true,
                format!(
                    "adapter_model.safetensors parsed with {} tensors",
                    adapter.tensors.len()
                ),
            );
            validate_tensors(&adapter, &mut receipt);
        }
        Err(err) => {
            push_check(
                &mut receipt.checks,
                "adapter_model_safetensors",
                false,
                format!("failed to load adapter_model.safetensors: {err}"),
            );
        }
    }

    finalize_status(&mut receipt);
    receipt
}

pub fn finalize_status(receipt: &mut AdapterVerifyReceipt) {
    let offline_ok = receipt.checks.iter().all(|check| check.pass);
    let server_ok = receipt
        .server
        .as_ref()
        .map(|server| server.checks.iter().all(|check| check.pass))
        .unwrap_or(true);
    receipt.status = if offline_ok && server_ok { "ok" } else { "failed" };
}

pub fn push_check(
    checks: &mut Vec<AdapterVerifyCheck>,
    name: &'static str,
    pass: bool,
    message: String,
) {
    checks.push(AdapterVerifyCheck {
        name,
        pass,
        message,
    });
}

fn resolve_adapter_path(input: &str, adapter_dir: Option<&Path>) -> PathBuf {
    let input_path = Path::new(input);
    if input_path.exists()
        || input_path.is_absolute()
        || input.contains('/')
        || input.contains('\\')
        || input == "."
        || input == ".."
    {
        input_path.to_path_buf()
    } else if let Some(adapter_dir) = adapter_dir {
        adapter_dir.join(input)
    } else {
        input_path.to_path_buf()
    }
}

fn find_single_nested_adapter_dir(parent: &Path) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path.join("adapter_config.json").is_file()
            && path.join("adapter_model.safetensors").is_file()
        {
            matches.push(path);
        }
    }
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

#[derive(Default)]
struct AdapterConfigMetadata {
    rank: Option<u64>,
    alpha: Option<f64>,
    alpha_over_rank: Option<f64>,
    target_modules: Vec<String>,
    base_model_name_or_path: Option<String>,
}

fn read_config_metadata(path: &Path) -> Result<AdapterConfigMetadata, String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| err.to_string())?;
    let rank = config.get("r").and_then(|v| v.as_u64());
    let alpha = config.get("lora_alpha").and_then(|v| v.as_f64());
    let alpha_over_rank = match (alpha, rank) {
        (Some(alpha), Some(rank)) if rank != 0 => Some(alpha / rank as f64),
        _ => None,
    };
    let target_modules = config
        .get("target_modules")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let base_model_name_or_path = config
        .get("base_model_name_or_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(AdapterConfigMetadata {
        rank,
        alpha,
        alpha_over_rank,
        target_modules,
        base_model_name_or_path,
    })
}

fn validate_tensors(adapter: &PeftLora, receipt: &mut AdapterVerifyReceipt) {
    let mut pairs: BTreeMap<(usize, String), ProjectionPair> = BTreeMap::new();
    let mut malformed = Vec::new();
    let rank = receipt.lora.rank.unwrap_or(0) as usize;

    for (name, tensor) in &adapter.tensors {
        accumulate_tensor_summary(tensor, &mut receipt.tensor_summary);
        match parse_peft_lora_key(name) {
            Some(parsed) => {
                if !receipt
                    .lora
                    .target_modules
                    .iter()
                    .any(|target| target == &parsed.module)
                {
                    malformed.push(format!(
                        "{name}: module `{}` is not listed in target_modules",
                        parsed.module
                    ));
                }
                if tensor.shape.len() != 2 {
                    malformed.push(format!(
                        "{name}: expected a 2D LoRA tensor, got shape {:?}",
                        tensor.shape
                    ));
                    continue;
                }
                match parsed.kind {
                    LoraTensorKind::A if rank != 0 && tensor.shape[0] != rank => {
                        malformed.push(format!(
                            "{name}: lora_A rank dimension {} does not match config r={rank}",
                            tensor.shape[0]
                        ));
                    }
                    LoraTensorKind::B if rank != 0 && tensor.shape[1] != rank => {
                        malformed.push(format!(
                            "{name}: lora_B rank dimension {} does not match config r={rank}",
                            tensor.shape[1]
                        ));
                    }
                    _ => {}
                }
                let pair = pairs.entry((parsed.layer, parsed.module)).or_default();
                let l2 = tensor_l2_norm(tensor);
                match parsed.kind {
                    LoraTensorKind::A => {
                        pair.a = Some(tensor.shape.clone());
                        pair.a_l2 = l2;
                    }
                    LoraTensorKind::B => {
                        pair.b = Some(tensor.shape.clone());
                        pair.b_l2 = l2;
                    }
                }
            }
            None => malformed.push(format!("{name}: unsupported PEFT LoRA tensor key")),
        }
    }

    receipt.tensor_summary.tensor_count = adapter.tensors.len();

    for ((layer, module), pair) in &pairs {
        match (&pair.a, &pair.b) {
            (Some(a), Some(b)) => {
                receipt.tensor_summary.paired_projection_count += 1;
                receipt.tensor_summary.lora_update_l2_upper_bound += pair.a_l2 * pair.b_l2;
                if a[0] != b[1] {
                    malformed.push(format!(
                        "layer {layer} module {module}: A rank {} does not match B rank {}",
                        a[0], b[1]
                    ));
                }
            }
            (None, Some(_)) => malformed.push(format!(
                "layer {layer} module {module}: missing lora_A tensor for lora_B"
            )),
            (Some(_), None) => malformed.push(format!(
                "layer {layer} module {module}: missing lora_B tensor for lora_A"
            )),
            (None, None) => {}
        }
    }

    let tensors_consistent = malformed.is_empty() && !pairs.is_empty();
    push_check(
        &mut receipt.checks,
        "safetensors_consistency",
        tensors_consistent,
        if tensors_consistent {
            format!(
                "{} LoRA projection pair(s) match config rank and target modules",
                receipt.tensor_summary.paired_projection_count
            )
        } else if malformed.is_empty() {
            "adapter_model.safetensors contains no PEFT LoRA A/B tensor pairs".to_string()
        } else {
            malformed.join("; ")
        },
    );

    let measurable = receipt.tensor_summary.nonzero_tensor_count > 0
        && receipt.tensor_summary.l2_weight_norm > 0.0
        && receipt.tensor_summary.lora_update_l2_upper_bound > 0.0;
    receipt.logit_delta_summary.measurable = measurable;
    receipt.logit_delta_summary.max_abs_delta_proxy = receipt.tensor_summary.max_abs_weight;
    receipt.logit_delta_summary.l2_delta_proxy =
        receipt.tensor_summary.lora_update_l2_upper_bound;
    push_check(
        &mut receipt.checks,
        "measurable_adapter_effect",
        measurable,
        if measurable {
            format!(
                "nonzero LoRA tensors found; delta proxy l2 upper bound {:.6}",
                receipt.tensor_summary.lora_update_l2_upper_bound
            )
        } else {
            "no measurable adapter effect: all LoRA tensors are zero or no A/B pairs were found"
                .to_string()
        },
    );
}

fn accumulate_tensor_summary(tensor: &MergeTensor, summary: &mut AdapterTensorSummary) {
    let mut any_nonzero = false;
    let mut sum_sq = 0.0f64;
    for value in &tensor.data {
        let v = *value as f64;
        let abs = v.abs();
        if abs > 0.0 {
            any_nonzero = true;
        }
        summary.max_abs_weight = summary.max_abs_weight.max(abs);
        summary.l1_weight_sum += abs;
        sum_sq += v * v;
    }
    if any_nonzero {
        summary.nonzero_tensor_count += 1;
    }
    summary.l2_weight_norm = (summary.l2_weight_norm.powi(2) + sum_sq).sqrt();
}

fn tensor_l2_norm(tensor: &MergeTensor) -> f64 {
    tensor
        .data
        .iter()
        .map(|value| {
            let v = *value as f64;
            v * v
        })
        .sum::<f64>()
        .sqrt()
}

#[derive(Default)]
struct ProjectionPair {
    a: Option<Vec<usize>>,
    b: Option<Vec<usize>>,
    a_l2: f64,
    b_l2: f64,
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

fn parse_peft_lora_key(key: &str) -> Option<ParsedLoraKey> {
    let parts: Vec<&str> = key.split('.').collect();
    let layer_pos = parts.iter().position(|part| *part == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse().ok()?;
    let lora_pos = parts.iter().position(|part| *part == "lora_A" || *part == "lora_B")?;
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

fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

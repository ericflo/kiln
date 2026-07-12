//! LoRA adapter weight loading from PEFT-compatible safetensors format.
//!
//! Loads LoRA A/B matrices from safetensors files and adapter_config.json,
//! organizing them into per-layer structs for use during forward pass.
//!
//! #1082: the A/B matrices are `kiln_tensor::Tensor` (kt). They are the
//! tape Parameter leaves the LoRA forward consumes, so the recorded tape
//! leaf ids equal the param's kt `tensor_id`. The safetensors load goes
//! straight to kt via [`kiln_tensor::safetensors::tensor_from_view`] (the
//! standalone `safetensors` crate parses the file format; the kt helper
//! maps dtype + copies the byte slice into a `CpuStorage`). The tensor
//! lands on CPU first, then `.to_device(device)` migrates to GPU — the
//! same pattern the base-model weights use.

use crate::backend::{BackendRuntime, ResidencyBackend};
use anyhow::{Context, Result, ensure};
use kiln_tensor::Tensor as KtTensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

const ADAPTER_WEIGHTS_IDENTITY_DOMAIN: &[u8] = b"kiln.adapter-weights.v1\0";
const ADAPTER_CONTENT_REVISION_DOMAIN: &[u8] = b"kiln.adapter-content-revision.v1\0";
const PEFT_SAFETENSORS_FILENAME: &str = "adapter_model.safetensors";

pub const SUPPORTED_LORA_TARGET_MODULES: &[&str] = &[
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
    pub a: KtTensor,
    /// B matrix: [out_features, rank]
    pub b: KtTensor,
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
    pub in_proj_qkv: Option<LoraProjectionWeights>,
    pub in_proj_z: Option<LoraProjectionWeights>,
    pub gdn_out_proj: Option<LoraProjectionWeights>,
}

impl LoraLayerWeights {
    pub fn has_mlp(&self) -> bool {
        self.gate_proj.is_some() || self.up_proj.is_some() || self.down_proj.is_some()
    }

    pub fn has_mlp_gate_up(&self) -> bool {
        self.gate_proj.is_some() || self.up_proj.is_some()
    }

    pub fn has_gdn_attention(&self) -> bool {
        self.in_proj_qkv.is_some() || self.in_proj_z.is_some() || self.gdn_out_proj.is_some()
    }

    /// Iterate over every present `LoraProjectionWeights` in this
    /// layer, calling `f` with each. Order matches
    /// the train/save target order.
    pub fn for_each_projection<F: FnMut(&LoraProjectionWeights)>(&self, mut f: F) {
        for proj in [
            self.q_proj.as_ref(),
            self.k_proj.as_ref(),
            self.v_proj.as_ref(),
            self.o_proj.as_ref(),
            self.gate_proj.as_ref(),
            self.up_proj.as_ref(),
            self.down_proj.as_ref(),
            self.in_proj_qkv.as_ref(),
            self.in_proj_z.as_ref(),
            self.gdn_out_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            f(proj);
        }
    }
}

/// Complete LoRA adapter weights for all layers.
pub struct LoraWeights {
    /// Per-layer LoRA weights, indexed by layer number.
    pub layers: Vec<LoraLayerWeights>,
    /// LoRA weights for the native MTP (multi-token-prediction) draft
    /// block, when the adapter was trained with MTP alignment. Keyed in
    /// the safetensors as `...mtp.layers.0.{self_attn,mlp}.{module}...`.
    /// `None` for adapters that predate MTP training — the draft block
    /// then runs base weights (correct, but acceptance degrades as the
    /// tuned model diverges from the base).
    pub mtp: Option<LoraLayerWeights>,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha (scaling factor).
    pub alpha: f32,
    /// Precomputed scale = alpha / rank.
    pub scale: f32,
    /// Exact source-byte identity when these weights came from
    /// [`LoraWeights::load`]. Training-time tensor views have no disk source
    /// and therefore carry `None` until they are serialized and loaded again.
    pub source_identity: Option<LoraSourceIdentity>,
}

/// Loader-owned identity of the exact PEFT files used to construct a LoRA.
///
/// Both digests are raw lowercase SHA-256. `weights_sha256` uses Kiln's
/// versioned adapter-weight framing contract, shared with
/// `scripts/vllm_teacher.py`, rather than naming or rescanning a mutable path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraSourceIdentity {
    weights_sha256: String,
    config_sha256: String,
}

impl LoraSourceIdentity {
    pub fn new(
        weights_sha256: impl Into<String>,
        config_sha256: impl Into<String>,
    ) -> Result<Self> {
        let value = Self {
            weights_sha256: weights_sha256.into(),
            config_sha256: config_sha256.into(),
        };
        for (field, digest) in [
            ("adapter weights", value.weights_sha256.as_str()),
            ("adapter config", value.config_sha256.as_str()),
        ] {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "{field} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
            );
        }
        Ok(value)
    }

    /// Fingerprint and validate the exact PEFT files currently present in an
    /// adapter directory without uploading tensors to a device. Registration
    /// can pin this value; the production load later publishes its own value
    /// from the bytes it actually consumed and must match exactly.
    pub fn from_adapter_dir(adapter_dir: &Path) -> Result<Self> {
        let source = read_lora_source(adapter_dir)?;
        Ok(source.identity)
    }

    /// Fingerprint an adapter only after its complete A/B tensor structure and
    /// every projection shape are proven compatible with `model_config`.
    ///
    /// # Safety
    ///
    /// Both PEFT files must remain immutable until this function returns. The
    /// weight file is memory-mapped to avoid a heap copy proportional to its
    /// size; concurrent truncation of a mapped file can terminate the process.
    pub unsafe fn from_immutable_adapter_dir_for_model(
        adapter_dir: &Path,
        model_config: &kiln_core::config::ModelConfig,
    ) -> Result<Self> {
        inspect_immutable_adapter_dir(adapter_dir, model_config)
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    /// Canonical revision of the exact PEFT config and weight identities.
    ///
    /// The domain separator and length framing make this safe to use as a
    /// stable cache/queue identity rather than concatenating two digests at
    /// each caller. The returned value is raw lowercase SHA-256.
    pub fn content_revision(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(ADAPTER_CONTENT_REVISION_DOMAIN);
        feed_len_prefixed(&mut digest, self.weights_sha256.as_bytes());
        feed_len_prefixed(&mut digest, self.config_sha256.as_bytes());
        hex_digest(&digest.finalize())
    }
}

fn inspect_immutable_adapter_dir(
    adapter_dir: &Path,
    model_config: &kiln_core::config::ModelConfig,
) -> Result<LoraSourceIdentity> {
    let config_path = adapter_dir.join("adapter_config.json");
    let config_metadata = std::fs::symlink_metadata(&config_path)
        .with_context(|| format!("failed to stat {}", config_path.display()))?;
    ensure!(
        !config_metadata.file_type().is_symlink() && config_metadata.file_type().is_file(),
        "{} is not a regular file",
        config_path.display()
    );
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: AdapterConfig =
        serde_json::from_slice(&config_bytes).context("failed to parse adapter_config.json")?;

    let weights_path = adapter_dir.join(PEFT_SAFETENSORS_FILENAME);
    let weights_metadata = std::fs::symlink_metadata(&weights_path)
        .with_context(|| format!("failed to stat {}", weights_path.display()))?;
    ensure!(
        !weights_metadata.file_type().is_symlink() && weights_metadata.file_type().is_file(),
        "{} is not a regular file",
        weights_path.display()
    );
    let weights_file = std::fs::File::open(&weights_path)
        .with_context(|| format!("failed to open {}", weights_path.display()))?;
    // SAFETY: the caller supplies a stable regular adapter file and this
    // mapping is read-only. Identity-only validation avoids a second heap
    // allocation proportional to the complete PEFT weight file.
    let weights = unsafe { memmap2::MmapOptions::new().map(&weights_file) }
        .with_context(|| format!("failed to map {}", weights_path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&weights)
        .context("failed to deserialize safetensors")?;
    validate_lora_structure(&config, &tensors, model_config)?;
    Ok(LoraSourceIdentity {
        weights_sha256: adapter_weights_identity_sha256(PEFT_SAFETENSORS_FILENAME, &weights),
        config_sha256: sha256_hex(&config_bytes),
    })
}

#[derive(Default)]
struct LoraPairMetadata {
    a: Option<(Vec<usize>, safetensors::Dtype)>,
    b: Option<(Vec<usize>, safetensors::Dtype)>,
}

fn validate_lora_structure(
    config: &AdapterConfig,
    tensors: &safetensors::SafeTensors<'_>,
    model_config: &kiln_core::config::ModelConfig,
) -> Result<()> {
    ensure!(config.r > 0, "adapter LoRA rank must be greater than zero");
    ensure!(
        config.lora_alpha.is_finite() && config.lora_alpha > 0.0,
        "adapter lora_alpha must be finite and greater than zero"
    );
    if let Some(task_type) = config.task_type.as_deref() {
        ensure!(
            task_type == "CAUSAL_LM",
            "adapter task_type must be CAUSAL_LM, found {task_type:?}"
        );
    }
    ensure!(
        !config.target_modules.is_empty(),
        "adapter target_modules must not be empty"
    );
    let targets = config
        .target_modules
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(
        targets.len() == config.target_modules.len(),
        "adapter target_modules contains duplicates"
    );
    let supported = SUPPORTED_LORA_TARGET_MODULES
        .iter()
        .map(|module| (*module).to_string())
        .collect::<BTreeSet<_>>();
    let unsupported = targets.difference(&supported).cloned().collect::<Vec<_>>();
    ensure!(
        unsupported.is_empty(),
        "adapter targets unsupported modules {unsupported:?}; supported modules are {supported:?}"
    );

    let mut pairs = BTreeMap::<(bool, usize, String), LoraPairMetadata>::new();
    let mut observed_targets = BTreeSet::new();
    for name in tensors.names() {
        let parsed = parse_peft_key_strict(name)
            .with_context(|| format!("unsupported PEFT tensor key {name:?}"))?;
        ensure!(
            targets.contains(&parsed.module),
            "PEFT tensor {name:?} targets module {:?} absent from adapter_config.json",
            parsed.module
        );
        ensure!(
            !parsed.is_mtp || parsed.layer == 0,
            "PEFT MTP tensor {name:?} uses unsupported MTP layer {}",
            parsed.layer
        );
        ensure!(
            parsed.is_mtp || parsed.layer < model_config.num_layers,
            "PEFT tensor {name:?} addresses layer {} but the resident model has {} layers",
            parsed.layer,
            model_config.num_layers
        );
        let view = tensors
            .tensor(name)
            .with_context(|| format!("read PEFT tensor metadata {name:?}"))?;
        ensure!(
            matches!(
                view.dtype(),
                safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
            ),
            "PEFT tensor {name:?} uses unsupported dtype {:?}; expected F32, F16, or BF16",
            view.dtype()
        );
        let pair = pairs
            .entry((parsed.is_mtp, parsed.layer, parsed.module.clone()))
            .or_default();
        let slot = if parsed.ab == "A" {
            &mut pair.a
        } else {
            &mut pair.b
        };
        ensure!(
            slot.is_none(),
            "duplicate PEFT projection tensor for {name:?}"
        );
        *slot = Some((view.shape().to_vec(), view.dtype()));
        observed_targets.insert(parsed.module);
    }
    ensure!(
        !pairs.is_empty(),
        "adapter contains no supported LoRA A/B tensors"
    );
    ensure!(
        observed_targets == targets,
        "adapter target_modules differs from tensors: declared {targets:?}, observed {observed_targets:?}"
    );

    for ((is_mtp, layer, module), pair) in pairs {
        let a = pair.a.with_context(|| {
            format!("adapter projection layer={layer} module={module} is missing lora_A")
        })?;
        let b = pair.b.with_context(|| {
            format!("adapter projection layer={layer} module={module} is missing lora_B")
        })?;
        ensure!(
            a.0.len() == 2 && b.0.len() == 2,
            "adapter projection layer={layer} module={module} must use 2D A/B tensors, found A={:?}, B={:?}",
            a.0,
            b.0
        );
        ensure!(
            a.0[0] == config.r && b.0[1] == config.r,
            "adapter projection layer={layer} module={module} rank differs from config r={}: A={:?}, B={:?}",
            config.r,
            a.0,
            b.0
        );
        ensure!(
            a.1 == b.1,
            "adapter projection layer={layer} module={module} mixes A/B dtypes {:?} and {:?}",
            a.1,
            b.1
        );
        let (input, output) =
            expected_lora_projection_shape(model_config, layer, &module, is_mtp).with_context(
                || {
                    format!(
                        "adapter projection layer={layer} module={module} is incompatible with the resident model"
                    )
                },
            )?;
        let expected_a = [config.r, input];
        let expected_b = [output, config.r];
        ensure!(
            a.0 == expected_a && b.0 == expected_b,
            "adapter projection layer={layer} module={module} shape mismatch: expected A={expected_a:?}, B={expected_b:?}; found A={:?}, B={:?}",
            a.0,
            b.0
        );
    }
    Ok(())
}

fn expected_lora_projection_shape(
    config: &kiln_core::config::ModelConfig,
    layer: usize,
    module: &str,
    is_mtp: bool,
) -> Option<(usize, usize)> {
    let hidden = config.hidden_size;
    let full_attention = is_mtp || config.is_full_attention_layer(layer);
    match module {
        "q_proj" if full_attention => Some((hidden, config.full_attn_q_proj_dim())),
        "k_proj" | "v_proj" if full_attention => {
            Some((hidden, config.num_kv_heads * config.head_dim))
        }
        "o_proj" if full_attention => Some((config.num_attention_heads * config.head_dim, hidden)),
        "in_proj_qkv" if !full_attention => Some((hidden, config.linear_qkv_dim())),
        "in_proj_z" if !full_attention => Some((hidden, config.linear_v_dim())),
        "out_proj" if !full_attention => Some((config.linear_v_dim(), hidden)),
        "gate_proj" | "up_proj" => Some((hidden, config.intermediate_size)),
        "down_proj" => Some((config.intermediate_size, hidden)),
        _ => None,
    }
}

impl LoraWeights {
    /// Identity of the exact PEFT source bytes, when loaded from disk.
    pub fn source_identity(&self) -> Option<&LoraSourceIdentity> {
        self.source_identity.as_ref()
    }

    /// Phase 4.1: register every LoRA A and B tensor in the backend's
    /// resident activation registry. After this, the inference path's
    /// `add_lora_delta_to_base` will dispatch through
    /// `lora_delta_resident` (on-device LoRA matmul) instead of
    /// candle CPU `compute_lora_delta`.
    ///
    /// Caller invokes this once after [`Self::load`], typically at
    /// adapter-load time. No-op on backends without registry support.
    /// Inverse: [`Self::evict_from_backend`] for cleanup.
    pub fn register_with_backend(&self, backend: &dyn BackendRuntime) -> anyhow::Result<()> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(());
        }
        for layer in self.layers.iter().chain(self.mtp.as_ref()) {
            let mut maybe_err: Option<anyhow::Error> = None;
            layer.for_each_projection(|proj| {
                if maybe_err.is_some() {
                    return;
                }
                // #1082: `proj.a` / `proj.b` are kt tensors and the
                // `register_resident_activation` backend hook now takes
                // `&kiln_tensor::Tensor`, so pass the kt tensor directly
                // (no kt -> candle bridge copy).
                let register = |t: &KtTensor| -> anyhow::Result<()> {
                    ResidencyBackend::runtime_register_resident_activation(backend, t)
                };
                if let Err(e) = register(&proj.a) {
                    maybe_err = Some(e);
                    return;
                }
                if let Err(e) = register(&proj.b) {
                    maybe_err = Some(e);
                }
            });
            if let Some(e) = maybe_err {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Inverse of [`Self::register_with_backend`].
    pub fn evict_from_backend(&self, backend: &dyn BackendRuntime) {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return;
        }
        for layer in self.layers.iter().chain(self.mtp.as_ref()) {
            layer.for_each_projection(|proj| {
                // #1082: `evict_resident_activation` now takes a kt
                // `&kiln_tensor::Tensor`, so pass `proj.a` / `proj.b`
                // directly (no kt -> candle bridge copy). The resident
                // registry is GPU-only behind
                // `supports_resident_activation()`.
                ResidencyBackend::runtime_evict_resident_activation(backend, &proj.a);
                ResidencyBackend::runtime_evict_resident_activation(backend, &proj.b);
            });
        }
    }
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
    ///
    /// #1082: `device` is a kt [`kiln_tensor::Device`]. The A/B matrices
    /// are loaded to CPU via the kt safetensors helper and then moved to
    /// `device`.
    pub fn load(
        adapter_dir: &Path,
        num_layers: usize,
        device: kiln_tensor::Device,
    ) -> Result<Self> {
        Self::load_from_source(adapter_dir, num_layers, device, None)
    }

    /// Load only when the exact PEFT bytes match a registration-time identity.
    /// The comparison happens after both files are read but before any tensor
    /// is deserialized or copied to the target device, so mutable adapter paths
    /// cannot silently change a queued teacher.
    pub fn load_pinned(
        adapter_dir: &Path,
        num_layers: usize,
        device: kiln_tensor::Device,
        expected_source: &LoraSourceIdentity,
    ) -> Result<Self> {
        Self::load_from_source(adapter_dir, num_layers, device, Some(expected_source))
    }

    fn load_from_source(
        adapter_dir: &Path,
        num_layers: usize,
        device: kiln_tensor::Device,
        expected_source: Option<&LoraSourceIdentity>,
    ) -> Result<Self> {
        let source = read_lora_source(adapter_dir)?;
        if let Some(expected) = expected_source {
            ensure!(
                &source.identity == expected,
                "adapter source identity changed: expected weights sha256:{} and config sha256:{}, observed weights sha256:{} and config sha256:{}",
                expected.weights_sha256(),
                expected.config_sha256(),
                source.identity.weights_sha256(),
                source.identity.config_sha256()
            );
        }
        let config = source.config;

        let rank = config.r;
        let alpha = config.lora_alpha;
        let scale = alpha / rank as f32;

        // Load safetensors
        let st_data = source.weights;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .context("failed to deserialize safetensors")?;

        // Parse all tensor names into a map: (layer_idx, module_name, "A"|"B") -> tensor_name
        // MTP-block keys (`...mtp.layers.0...`) go to their own map — they
        // would otherwise collide with main layer 0.
        let mut tensor_map: HashMap<(usize, String, String), String> = HashMap::new();
        let mut mtp_tensor_map: HashMap<(String, String), String> = HashMap::new();

        for name in tensors.names() {
            if let Some(parsed) = parse_peft_key(name) {
                if parsed.is_mtp {
                    mtp_tensor_map.insert((parsed.module, parsed.ab), name.to_string());
                } else {
                    tensor_map.insert((parsed.layer, parsed.module, parsed.ab), name.to_string());
                }
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

                    let a = safetensor_to_kt(&a_view, device)
                        .with_context(|| format!("converting {a_name}"))?;
                    let b = safetensor_to_kt(&b_view, device)
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
                        "in_proj_qkv" => layer.in_proj_qkv = Some(proj),
                        "in_proj_z" => layer.in_proj_z = Some(proj),
                        "out_proj" => layer.gdn_out_proj = Some(proj),
                        _ => {
                            tracing::warn!("unknown LoRA target module: {module}, skipping");
                        }
                    }
                }
            }

            layers.push(layer);
        }

        // Optional MTP draft-block LoRA (one full-attention layer, k=1).
        let mut mtp_layer = LoraLayerWeights::default();
        let mut mtp_any = false;
        for module in &config.target_modules {
            let a_key = mtp_tensor_map.get(&(module.clone(), "A".to_string()));
            let b_key = mtp_tensor_map.get(&(module.clone(), "B".to_string()));
            if let (Some(a_name), Some(b_name)) = (a_key, b_key) {
                let a_view = tensors
                    .tensor(a_name)
                    .with_context(|| format!("failed to get tensor {a_name}"))?;
                let b_view = tensors
                    .tensor(b_name)
                    .with_context(|| format!("failed to get tensor {b_name}"))?;
                let a = safetensor_to_kt(&a_view, device)
                    .with_context(|| format!("converting {a_name}"))?;
                let b = safetensor_to_kt(&b_view, device)
                    .with_context(|| format!("converting {b_name}"))?;
                let proj = LoraProjectionWeights { a, b };
                mtp_any = true;
                match module.as_str() {
                    "q_proj" => mtp_layer.q_proj = Some(proj),
                    "k_proj" => mtp_layer.k_proj = Some(proj),
                    "v_proj" => mtp_layer.v_proj = Some(proj),
                    "o_proj" => mtp_layer.o_proj = Some(proj),
                    "gate_proj" => mtp_layer.gate_proj = Some(proj),
                    "up_proj" => mtp_layer.up_proj = Some(proj),
                    "down_proj" => mtp_layer.down_proj = Some(proj),
                    _ => {
                        mtp_any = mtp_any
                            && (mtp_layer.q_proj.is_some()
                                || mtp_layer.k_proj.is_some()
                                || mtp_layer.v_proj.is_some()
                                || mtp_layer.o_proj.is_some()
                                || mtp_layer.gate_proj.is_some()
                                || mtp_layer.up_proj.is_some()
                                || mtp_layer.down_proj.is_some());
                        tracing::warn!("unsupported MTP LoRA target module: {module}, skipping");
                    }
                }
            }
        }

        Ok(Self {
            layers,
            mtp: mtp_any.then_some(mtp_layer),
            rank,
            alpha,
            scale,
            source_identity: Some(source.identity),
        })
    }
}

struct LoadedLoraSource {
    config: AdapterConfig,
    weights: Vec<u8>,
    identity: LoraSourceIdentity,
}

fn read_lora_source(adapter_dir: &Path) -> Result<LoadedLoraSource> {
    let config_path = adapter_dir.join("adapter_config.json");
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: AdapterConfig =
        serde_json::from_slice(&config_bytes).context("failed to parse adapter_config.json")?;

    let weights_path = adapter_dir.join(PEFT_SAFETENSORS_FILENAME);
    let weights = std::fs::read(&weights_path)
        .with_context(|| format!("failed to read {}", weights_path.display()))?;
    safetensors::SafeTensors::deserialize(&weights).context("failed to deserialize safetensors")?;
    let identity = LoraSourceIdentity {
        weights_sha256: adapter_weights_identity_sha256(PEFT_SAFETENSORS_FILENAME, &weights),
        config_sha256: sha256_hex(&config_bytes),
    };
    Ok(LoadedLoraSource {
        config,
        weights,
        identity,
    })
}

fn adapter_weights_identity_sha256(filename: &str, bytes: &[u8]) -> String {
    let raw_weights_sha256 = Sha256::digest(bytes);
    let mut aggregate = Sha256::new();
    aggregate.update(ADAPTER_WEIGHTS_IDENTITY_DOMAIN);
    aggregate.update(1u64.to_le_bytes());
    feed_len_prefixed(&mut aggregate, filename.as_bytes());
    aggregate.update((bytes.len() as u64).to_le_bytes());
    aggregate.update(raw_weights_sha256);
    hex_digest(&aggregate.finalize())
}

fn feed_len_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_digest(&Sha256::digest(bytes))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

/// Parsed PEFT weight key components.
struct ParsedKey {
    layer: usize,
    module: String,
    ab: String, // "A" or "B"
    /// Key addresses the MTP draft block (`...mtp.layers.{i}...`) rather
    /// than a main-model layer.
    is_mtp: bool,
}

/// Parse a PEFT-style safetensors key into layer index, module name, and A/B indicator.
///
/// Expected patterns:
/// - `base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight`
/// - `base_model.model.model.layers.{i}.mlp.{module}.lora_{A|B}.weight`
fn parse_peft_key(key: &str) -> Option<ParsedKey> {
    // Look for "layers.{i}." and "lora_{A|B}.weight". Production loading
    // retains this permissive parser for compatibility with existing PEFT
    // names; verified import applies `parse_peft_key_strict` first.
    let parts: Vec<&str> = key.split('.').collect();

    let layer_pos = parts.iter().position(|&p| p == "layers")?;
    let layer_idx: usize = parts.get(layer_pos + 1)?.parse().ok()?;

    let lora_pos = parts
        .iter()
        .position(|p| *p == "lora_A" || *p == "lora_B")?;
    let ab = if parts[lora_pos] == "lora_A" {
        "A".to_string()
    } else {
        "B".to_string()
    };
    let module = parts.get(lora_pos.checked_sub(1)?)?.to_string();
    let is_mtp = parts[..layer_pos].contains(&"mtp");

    Some(ParsedKey {
        layer: layer_idx,
        module,
        ab,
        is_mtp,
    })
}

fn parse_peft_key_strict(key: &str) -> Option<ParsedKey> {
    let parts = key.split('.').collect::<Vec<_>>();
    if parts.last() != Some(&"weight") || parts.len() < 6 {
        return None;
    }
    let lora_pos = parts.len().checked_sub(2)?;
    if !matches!(parts[lora_pos], "lora_A" | "lora_B") {
        return None;
    }
    let layer_positions = parts
        .iter()
        .enumerate()
        .filter_map(|(index, part)| (*part == "layers").then_some(index))
        .collect::<Vec<_>>();
    if layer_positions.len() != 1 || lora_pos <= layer_positions[0] + 2 {
        return None;
    }
    parse_peft_key(key)
}

/// Convert a safetensors tensor view to a kt [`kiln_tensor::Tensor`].
///
/// #1082: the standalone `safetensors` crate parses the file format and
/// hands us a `TensorView`; [`kiln_tensor::safetensors::tensor_from_view`]
/// maps the dtype and copies the byte slice into a `CpuStorage`-backed
/// CPU tensor. We then migrate to `device` with an explicit
/// `to_device` (no-op when `device` is `Cpu`).
fn safetensor_to_kt(
    view: &safetensors::tensor::TensorView<'_>,
    device: kiln_tensor::Device,
) -> Result<KtTensor> {
    let cpu = kiln_tensor::safetensors::tensor_from_view(view)
        .context("failed to build kt tensor from safetensors view")?;
    let tensor = cpu
        .to_device(device)
        .context("failed to move LoRA tensor to device")?;
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
pub fn compute_lora_delta(
    x: &KtTensor,
    proj: &LoraProjectionWeights,
    scale: f32,
) -> Result<KtTensor> {
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

    let hidden = matmul_last_dim_rhs_transposed(x, &a)?; // [..., rank]
    let delta = matmul_last_dim_rhs_transposed(&hidden, &b)?; // [..., out_features]
    // Keep the scalar on the tensor's device. The overloaded `Tensor * f64`
    // path materializes a CPU broadcast tensor on non-CPU substrates, which
    // made frozen LoRA reference forwards fail mid-GRPO on Vulkan.
    let delta = kiln_tensor::ops::mul_scalar(&delta, scale)?;

    // Final cast to input dtype (no-op when already matching).
    let delta = delta.to_dtype(x.dtype())?;
    Ok(delta)
}

fn matmul_last_dim_rhs_transposed(lhs: &KtTensor, rhs: &KtTensor) -> Result<KtTensor> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() < 2 || rhs_shape.len() != 2 {
        anyhow::bail!(
            "matmul_last_dim_rhs_transposed: expected lhs rank >= 2 and rhs rank 2, got lhs={lhs_shape:?} rhs={rhs_shape:?}"
        );
    }
    let in_features = *lhs_shape
        .last()
        .context("matmul_last_dim_rhs_transposed lhs missing last dim")?;
    let out_features = rhs_shape[0];
    if rhs_shape[1] != in_features {
        anyhow::bail!(
            "matmul_last_dim_rhs_transposed: inner dim mismatch lhs last dim={in_features} rhs={rhs_shape:?}"
        );
    }
    let rows = lhs_shape[..lhs_shape.len() - 1]
        .iter()
        .copied()
        .product::<usize>();
    let lhs_2d = lhs.reshape((rows, in_features))?.contiguous()?;
    let out_2d = kiln_tensor::ops::matmul_rhs_transposed(&lhs_2d, rhs)?;
    let mut out_shape = lhs_shape[..lhs_shape.len() - 1].to_vec();
    out_shape.push(out_features);
    Ok(out_2d.reshape(out_shape)?)
}

fn needs_f32_matmul_fallback(lhs: &KtTensor, rhs: &KtTensor) -> bool {
    lhs.dtype() != rhs.dtype()
        || matches!(lhs.device(), kiln_tensor::Device::Cpu)
            && (lhs.dtype() != kiln_tensor::DType::F32 || rhs.dtype() != kiln_tensor::DType::F32)
}

/// Apply a LoRA-augmented linear projection.
///
/// Computes: `(x @ W^T) + (x @ A^T @ B^T) * scale`
///
/// If no LoRA weights are provided for this projection, just returns `x @ W^T`.
pub fn linear_with_lora(
    x: &KtTensor,
    base_weight: &KtTensor,
    lora: Option<&LoraProjectionWeights>,
    scale: f32,
) -> Result<KtTensor> {
    let f32_matmul_fallback = needs_f32_matmul_fallback(x, base_weight);
    let base_output = if f32_matmul_fallback {
        let x_f32 = x.to_dtype(kiln_tensor::DType::F32)?;
        let w_f32 = base_weight.to_dtype(kiln_tensor::DType::F32)?;
        matmul_last_dim_rhs_transposed(&x_f32, &w_f32)?
    } else {
        matmul_last_dim_rhs_transposed(x, base_weight)?
    };
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
    x: &KtTensor,
    base_weight_t: &KtTensor,
    lora: Option<&LoraProjectionWeights>,
    scale: f32,
) -> Result<KtTensor> {
    // (#1082) GDN-on-Vulkan: some GDN-block intermediates land on `Device::Cpu`
    // while the frozen projection weight stays on the accelerator. The kt
    // `broadcast_matmul` op requires both operands co-located, so align the
    // weight to the activation device. No-op on the matched-device paths
    // (CUDA / Metal / full-attn / decode), so they stay byte-identical.
    let base_weight_aligned;
    let base_weight_t = if x.device() != base_weight_t.device() {
        base_weight_aligned = base_weight_t.to_device(x.device())?;
        &base_weight_aligned
    } else {
        base_weight_t
    };
    let f32_matmul_fallback = needs_f32_matmul_fallback(x, base_weight_t);
    let base_output = if crate::mtp_debug::is_mtp_fp32_head_armed() || f32_matmul_fallback {
        let in_dtype = x.dtype();
        let x_f32 = x.to_dtype(kiln_tensor::DType::F32)?;
        let w_f32 = base_weight_t.to_dtype(kiln_tensor::DType::F32)?;
        let out = x_f32.broadcast_matmul(&w_f32)?;
        if f32_matmul_fallback {
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
    use kiln_tensor::Device;

    #[test]
    fn adapter_weight_identity_matches_cross_runtime_golden() {
        assert_eq!(
            adapter_weights_identity_sha256(PEFT_SAFETENSORS_FILENAME, b"weights"),
            "f9cf9b7ba0de3353dc9baf8047675fe8e5077518c16b293d50a2f23e50aa5c15"
        );
    }

    #[test]
    fn mixed_cpu_linear_fallback_promotes_without_backend_help() -> Result<()> {
        let x = KtTensor::from_vec(
            vec![0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
            (1, 2, 4),
        )?;
        let weight = KtTensor::from_vec(
            vec![
                0.5_f32, -0.25, 0.125, -0.5, 0.75, 0.25, 1.0, -0.125, 0.375, -0.75, 0.625, 0.5,
            ],
            (4, 3),
        )?
        .to_dtype(kiln_tensor::DType::BF16)?;
        let output = linear_with_lora_t(&x, &weight, None, 0.0)?;
        assert_eq!(output.dtype(), kiln_tensor::DType::F32);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            vec![0.375, 0.09375, 0.6875, -0.375, -0.09375, -0.6875]
        );
        Ok(())
    }

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
        let namespaced = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight";
        assert!(parse_peft_key(namespaced).is_some());
        assert!(parse_peft_key_strict(namespaced).is_none());
        assert!(
            parse_peft_key_strict("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
                .is_some()
        );
    }

    #[test]
    fn import_validation_rejects_missing_pairs_targets_and_layers() -> Result<()> {
        let mut resident = kiln_core::config::ModelConfig::qwen3_5_4b();
        resident.hidden_size = 4;
        resident.num_layers = 1;
        resident.num_attention_heads = 1;
        resident.num_kv_heads = 1;
        resident.head_dim = 4;
        resident.num_full_attention_layers = 1;
        resident.full_attention_interval = 1;
        resident.attn_output_gate = false;
        let config = AdapterConfig {
            r: 1,
            lora_alpha: 2.0,
            target_modules: vec!["q_proj".to_string()],
            task_type: Some("CAUSAL_LM".to_string()),
        };
        let a_bytes = vec![0u8; 4 * std::mem::size_of::<f32>()];
        let b_bytes = vec![0u8; 4 * std::mem::size_of::<f32>()];
        let a =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 4], &a_bytes)?;
        let missing_b = safetensors::tensor::serialize(
            [("base.model.layers.0.self_attn.q_proj.lora_A.weight", a)],
            None,
        )?;
        let tensors = safetensors::SafeTensors::deserialize(&missing_b)?;
        let error = validate_lora_structure(&config, &tensors, &resident).unwrap_err();
        assert!(error.to_string().contains("missing lora_B"), "{error:#}");

        let a =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 4], &a_bytes)?;
        let b =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 1], &b_bytes)?;
        let wrong_layer = safetensors::tensor::serialize(
            [
                ("base.model.layers.1.self_attn.q_proj.lora_A.weight", a),
                ("base.model.layers.1.self_attn.q_proj.lora_B.weight", b),
            ],
            None,
        )?;
        let tensors = safetensors::SafeTensors::deserialize(&wrong_layer)?;
        let error = validate_lora_structure(&config, &tensors, &resident).unwrap_err();
        assert!(error.to_string().contains("has 1 layers"), "{error:#}");

        let unsupported = AdapterConfig {
            target_modules: vec!["in_proj_a".to_string()],
            ..config
        };
        let error = validate_lora_structure(&unsupported, &tensors, &resident).unwrap_err();
        assert!(
            error.to_string().contains("unsupported modules"),
            "{error:#}"
        );
        Ok(())
    }

    /// MTP draft-block keys must parse as MTP — without the flag they'd
    /// alias main layer 0 and silently corrupt its LoRA.
    #[test]
    fn test_parse_peft_key_mtp() {
        let key = "base_model.model.model.mtp.layers.0.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert!(parsed.is_mtp);
        assert_eq!(parsed.layer, 0);
        assert_eq!(parsed.module, "q_proj");

        let main_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        assert!(!parse_peft_key(main_key).unwrap().is_mtp);
    }

    /// Full load round-trip: an adapter carrying both main-layer and MTP
    /// keys loads with `mtp: Some(...)`; an adapter without MTP keys
    /// (every adapter trained before MTP alignment) loads `mtp: None`.
    #[test]
    fn test_load_adapter_with_and_without_mtp_keys() -> Result<()> {
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj", "gate_proj"]}"#,
        )?;

        let a_data: Vec<f32> = vec![0.1; 2 * 4]; // [rank=2, in=4]
        let b_data: Vec<f32> = vec![0.2; 4 * 2]; // [out=4, rank=2]
        fn bytemuck_cast_slice_f32(data: &[f32]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            }
        }
        fn mk<'a>(data: &'a [f32], shape: Vec<usize>) -> safetensors::tensor::TensorView<'a> {
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape,
                bytemuck_cast_slice_f32(data),
            )
            .unwrap()
        }

        let mut tensors: BTreeMap<String, safetensors::tensor::TensorView<'_>> = BTreeMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
            mk(&a_data, vec![2, 4]),
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
            mk(&b_data, vec![4, 2]),
        );
        tensors.insert(
            "base_model.model.model.mtp.layers.0.self_attn.q_proj.lora_A.weight".into(),
            mk(&a_data, vec![2, 4]),
        );
        tensors.insert(
            "base_model.model.model.mtp.layers.0.self_attn.q_proj.lora_B.weight".into(),
            mk(&b_data, vec![4, 2]),
        );
        tensors.insert(
            "base_model.model.model.mtp.layers.0.mlp.gate_proj.lora_A.weight".into(),
            mk(&a_data, vec![2, 4]),
        );
        tensors.insert(
            "base_model.model.model.mtp.layers.0.mlp.gate_proj.lora_B.weight".into(),
            mk(&b_data, vec![4, 2]),
        );
        let st = safetensors::serialize(&tensors, None)?;
        std::fs::write(dir.path().join("adapter_model.safetensors"), st)?;

        let loaded = LoraWeights::load(dir.path(), 1, Device::Cpu)?;
        assert!(loaded.layers[0].q_proj.is_some(), "main layer parsed");
        assert!(
            loaded.layers[0].gate_proj.is_none(),
            "MTP keys must not bleed into main layer 0"
        );
        let mtp = loaded.mtp.as_ref().expect("MTP block parsed");
        assert!(mtp.q_proj.is_some());
        assert!(mtp.gate_proj.is_some());
        assert!(mtp.o_proj.is_none());

        // Legacy adapter (no MTP keys) → mtp: None.
        let dir2 = tempfile::tempdir()?;
        std::fs::write(
            dir2.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj"]}"#,
        )?;
        let mut legacy: BTreeMap<String, safetensors::tensor::TensorView<'_>> = BTreeMap::new();
        legacy.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
            mk(&a_data, vec![2, 4]),
        );
        legacy.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
            mk(&b_data, vec![4, 2]),
        );
        let st2 = safetensors::serialize(&legacy, None)?;
        std::fs::write(dir2.path().join("adapter_model.safetensors"), st2)?;
        let legacy_loaded = LoraWeights::load(dir2.path(), 1, Device::Cpu)?;
        assert!(legacy_loaded.mtp.is_none());
        Ok(())
    }

    #[test]
    fn test_compute_lora_delta_known_values() -> Result<()> {
        // x: [1, 2, 4] (batch=1, seq_len=2, in_features=4)
        // kt `Tensor::new` is rank-1 only; higher-rank host data flips to
        // `from_slice` + `reshape` (#1082).
        let x = KtTensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            (1usize, 2usize, 4usize),
        )?;

        // A: [2, 4] (rank=2, in_features=4) — identity-like
        let a = KtTensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            (2usize, 4usize),
        )?;

        // B: [3, 2] (out_features=3, rank=2)
        let b = KtTensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3usize, 2usize))?;

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
        let x = KtTensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (1usize, 1usize, 4usize))?;

        // W: [3, 4] — base weight
        let w = KtTensor::zeros((3usize, 4usize), kiln_tensor::DType::F32, device)?;

        // A: [2, 4], B: [3, 2]
        let a = KtTensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            (2usize, 4usize),
        )?;
        let b = KtTensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5], (3usize, 2usize))?;

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
        let config_bytes = serde_json::to_vec_pretty(&config)?;
        std::fs::write(adapter_dir.join("adapter_config.json"), &config_bytes)?;

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
        let weights = LoraWeights::load(adapter_dir, 1, device)?;

        assert_eq!(weights.rank, 4);
        assert!((weights.alpha - 8.0).abs() < 1e-5);
        assert!((weights.scale - 2.0).abs() < 1e-5);
        assert_eq!(weights.layers.len(), 1);
        let source_identity = weights
            .source_identity
            .as_ref()
            .expect("disk-loaded LoRA publishes exact source identity");
        assert_eq!(
            LoraSourceIdentity::from_adapter_dir(adapter_dir)?,
            *source_identity
        );
        let mut resident = kiln_core::config::ModelConfig::qwen3_5_4b();
        resident.hidden_size = in_features;
        resident.num_layers = 1;
        resident.num_attention_heads = 2;
        resident.num_kv_heads = 2;
        resident.head_dim = 4;
        resident.num_full_attention_layers = 1;
        resident.full_attention_interval = 1;
        resident.attn_output_gate = false;
        // SAFETY: this test owns the temporary directory and does not mutate
        // either PEFT file while the validation call is active.
        assert_eq!(
            unsafe {
                LoraSourceIdentity::from_immutable_adapter_dir_for_model(adapter_dir, &resident)?
            },
            *source_identity
        );
        assert_eq!(source_identity.config_sha256(), sha256_hex(&config_bytes));
        assert_eq!(
            source_identity.weights_sha256(),
            adapter_weights_identity_sha256(PEFT_SAFETENSORS_FILENAME, &serialized)
        );
        let original_revision = source_identity.content_revision();
        assert_eq!(original_revision.len(), 64);
        assert!(
            original_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        assert_eq!(original_revision, source_identity.content_revision());
        LoraWeights::load_pinned(adapter_dir, 1, device, source_identity)?;

        let mut changed_config = config.clone();
        changed_config["lora_alpha"] = serde_json::json!(9.0);
        std::fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::to_vec_pretty(&changed_config)?,
        )?;
        assert_ne!(
            LoraSourceIdentity::from_adapter_dir(adapter_dir)?.content_revision(),
            original_revision
        );
        let error = match LoraWeights::load_pinned(adapter_dir, 1, device, source_identity) {
            Ok(_) => panic!("changed adapter config must not satisfy the pinned identity"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("adapter source identity changed")
        );

        // The identity belongs to the bytes already parsed, not to the mutable
        // directory name. Later replacement cannot silently rewrite it.
        let loaded_identity = source_identity.clone();
        std::fs::write(adapter_dir.join("adapter_config.json"), b"{}")?;
        std::fs::write(adapter_dir.join(PEFT_SAFETENSORS_FILENAME), b"replaced")?;
        assert_eq!(weights.source_identity.as_ref(), Some(&loaded_identity));

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

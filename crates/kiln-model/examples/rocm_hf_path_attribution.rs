//! Attribute one retained HF next-token divergence across Kiln's ROCm paths.

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("rocm_hf_path_attribution requires --features rocm");
    std::process::exit(2);
}

#[cfg(feature = "rocm")]
mod rocm {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use kiln_core::block::BlockTable;
    use kiln_core::config::{DType as ModelDType, ModelConfig};
    use kiln_memory::GovernorConfig;
    use kiln_model::backend::{self, LinearBackend};
    use kiln_model::forward::{
        GpuWeights, LinearAttentionState, model_forward_paged_last_token,
        model_forward_paged_last_token_greedy, model_forward_paged_last_token_with_layer_snapshots,
        model_forward_paged_next_token_greedy,
    };
    use kiln_model::rocm_graph::RocmGraphRunner;
    use kiln_model::{
        InferenceMemoryRuntime, KtApiMode, LoadModelOptions, PagedKvCacheKt,
        RocmGraphExecutionPolicy, RocmKernelPolicy, install_kt_api_mode,
        install_rocm_kernel_policy, load_model_with_options,
    };
    use kiln_tensor::{
        DType, Device, RocmExecutionPolicy, RocmSynchronizationMode, RocmTensorKernelPolicy,
        Tensor, primary_rocm_context_with_execution_policy,
    };
    use safetensors::SafeTensors;
    use serde::Serialize;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const MARKER: &str = "KILN_ROCM_HF_PATH_ATTRIBUTION ";
    const LAYER_MARKER: &str = "KILN_ROCM_HF_LAYER_ATTRIBUTION ";
    const RESULT_SCHEMA: &str = "kiln.rocm-hf-path-attribution.v1";
    const LAYER_RESULT_SCHEMA: &str = "kiln.rocm-hf-layer-attribution.v1";
    const HF_SCHEMA: &str = "kiln.qwen35-hf-full-logits.v1";
    const HF_LAYER_SCHEMA: &str = "kiln.qwen35-hf-layer-last-rows.v1";
    const BLOCK_SIZE: usize = 16;
    const NUM_BLOCKS: usize = 32;
    const GRAPH_ROW_PRIME: u64 = 1;
    const GRAPH_ROW_FULL: u64 = 2;
    const GRAPH_ROW_GREEDY: u64 = 3;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum KernelProfile {
        FusedNormMlpFallback,
        FusedNormMlpOnly,
        FusedRmsnormFallback,
        FusedMlpSiluMulFallback,
        FusedMlpGateUpPrefillFallback,
        GdnFallback,
        ModelFallback,
        NonGdnFallback,
        Qualified,
        PortableFallback,
        SplitQGateFallback,
        SplitQGateOnly,
        TensorFallback,
    }

    impl KernelProfile {
        fn parse(value: &str) -> Result<Self> {
            match value {
                "fused_norm_mlp_fallback" => Ok(Self::FusedNormMlpFallback),
                "fused_norm_mlp_only" => Ok(Self::FusedNormMlpOnly),
                "fused_rmsnorm_fallback" => Ok(Self::FusedRmsnormFallback),
                "fused_mlp_silu_mul_fallback" => Ok(Self::FusedMlpSiluMulFallback),
                "fused_mlp_gate_up_prefill_fallback" => Ok(Self::FusedMlpGateUpPrefillFallback),
                "gdn_fallback" => Ok(Self::GdnFallback),
                "model_fallback" => Ok(Self::ModelFallback),
                "non_gdn_fallback" => Ok(Self::NonGdnFallback),
                "qualified" => Ok(Self::Qualified),
                "portable_fallback" => Ok(Self::PortableFallback),
                "split_q_gate_fallback" => Ok(Self::SplitQGateFallback),
                "split_q_gate_only" => Ok(Self::SplitQGateOnly),
                "tensor_fallback" => Ok(Self::TensorFallback),
                _ => anyhow::bail!(
                    "--kernel-profile must be fused_mlp_gate_up_prefill_fallback, fused_mlp_silu_mul_fallback, fused_norm_mlp_fallback, fused_norm_mlp_only, fused_rmsnorm_fallback, gdn_fallback, model_fallback, non_gdn_fallback, qualified, portable_fallback, split_q_gate_fallback, split_q_gate_only, or tensor_fallback, got {value}"
                ),
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::FusedNormMlpFallback => "fused_norm_mlp_fallback",
                Self::FusedNormMlpOnly => "fused_norm_mlp_only",
                Self::FusedRmsnormFallback => "fused_rmsnorm_fallback",
                Self::FusedMlpSiluMulFallback => "fused_mlp_silu_mul_fallback",
                Self::FusedMlpGateUpPrefillFallback => "fused_mlp_gate_up_prefill_fallback",
                Self::GdnFallback => "gdn_fallback",
                Self::ModelFallback => "model_fallback",
                Self::NonGdnFallback => "non_gdn_fallback",
                Self::Qualified => "qualified",
                Self::PortableFallback => "portable_fallback",
                Self::SplitQGateFallback => "split_q_gate_fallback",
                Self::SplitQGateOnly => "split_q_gate_only",
                Self::TensorFallback => "tensor_fallback",
            }
        }

        const fn model_policy(self) -> RocmKernelPolicy {
            match self {
                Self::FusedNormMlpFallback => RocmKernelPolicy::fused_norm_mlp_fallback(),
                Self::FusedNormMlpOnly => RocmKernelPolicy::fused_norm_mlp_only(),
                Self::FusedRmsnormFallback => RocmKernelPolicy::fused_rmsnorm_fallback(),
                Self::FusedMlpSiluMulFallback => RocmKernelPolicy::fused_mlp_silu_mul_fallback(),
                Self::FusedMlpGateUpPrefillFallback => {
                    RocmKernelPolicy::fused_mlp_gate_up_prefill_fallback()
                }
                Self::GdnFallback => RocmKernelPolicy::gdn_fallback(),
                Self::ModelFallback => RocmKernelPolicy::portable_fallback(),
                Self::NonGdnFallback => RocmKernelPolicy::non_gdn_fallback(),
                Self::Qualified => RocmKernelPolicy::qualified(),
                Self::PortableFallback => RocmKernelPolicy::portable_fallback(),
                Self::SplitQGateFallback => RocmKernelPolicy::split_q_gate_fallback(),
                Self::SplitQGateOnly => RocmKernelPolicy::split_q_gate_only(),
                Self::TensorFallback => RocmKernelPolicy::qualified(),
            }
        }

        const fn tensor_policy(self) -> RocmTensorKernelPolicy {
            match self {
                Self::FusedNormMlpFallback => RocmTensorKernelPolicy::qualified(),
                Self::FusedNormMlpOnly => RocmTensorKernelPolicy::qualified(),
                Self::FusedRmsnormFallback => RocmTensorKernelPolicy::qualified(),
                Self::FusedMlpSiluMulFallback => RocmTensorKernelPolicy::qualified(),
                Self::FusedMlpGateUpPrefillFallback => RocmTensorKernelPolicy::qualified(),
                Self::GdnFallback => RocmTensorKernelPolicy::qualified(),
                Self::ModelFallback => RocmTensorKernelPolicy::qualified(),
                Self::NonGdnFallback => RocmTensorKernelPolicy::qualified(),
                Self::Qualified => RocmTensorKernelPolicy::qualified(),
                Self::PortableFallback => RocmTensorKernelPolicy::portable_fallback(),
                Self::SplitQGateFallback => RocmTensorKernelPolicy::qualified(),
                Self::SplitQGateOnly => RocmTensorKernelPolicy::qualified(),
                Self::TensorFallback => RocmTensorKernelPolicy::portable_fallback(),
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        kernel_profile: KernelProfile,
        layer_attribution: bool,
        model: PathBuf,
        request: PathBuf,
        reference: PathBuf,
    }

    #[derive(Debug)]
    struct Request {
        id: String,
        input_token_ids_sha256: String,
        prompt: Vec<u32>,
        continuation: Vec<u32>,
        candidates: Vec<u32>,
    }

    #[derive(Debug)]
    struct Reference {
        boundary_names: Option<Vec<String>>,
        layer_last_rows: Option<Vec<Vec<f32>>>,
        logits: Vec<f32>,
    }

    #[derive(Debug, Serialize)]
    struct CandidateLogit {
        logit: f32,
        rank: usize,
        token_id: u32,
    }

    #[derive(Debug, Serialize)]
    struct LogitComparison {
        argmax: u32,
        argmax_equal: bool,
        candidate_tokens: Vec<CandidateLogit>,
        cosine_similarity: f64,
        logits_sha256: String,
        max_abs_error: f64,
        mean_abs_error: f64,
        top10_overlap: usize,
    }

    #[derive(Debug, Serialize)]
    struct FullPathResult {
        comparison: LogitComparison,
        observed_next_tokens: Vec<u32>,
    }

    #[derive(Debug, Serialize)]
    struct GreedyPathResult {
        final_token_matches_reference: bool,
        observed_next_tokens: Vec<u32>,
    }

    #[derive(Debug, Serialize)]
    struct GraphEvidence {
        cache_admission_successes: u64,
        capture_attempts: u64,
        capture_failures: u64,
        capture_successes: u64,
        enabled: bool,
        fallbacks: u64,
        replay_attempts: u64,
        replay_failures: u64,
        replay_successes: u64,
    }

    #[derive(Debug, Serialize)]
    struct ResultMarker {
        attribution: &'static str,
        eager_full: FullPathResult,
        eager_greedy: GreedyPathResult,
        graph: GraphEvidence,
        graph_full: FullPathResult,
        graph_greedy: GreedyPathResult,
        hf_argmax: u32,
        input_token_count: usize,
        input_token_ids_sha256: String,
        kernel_policy: &'static str,
        request_id: String,
        schema: &'static str,
    }

    #[derive(Debug, Serialize)]
    struct BoundaryComparison {
        cosine_similarity: f64,
        hf_sha256: String,
        index: usize,
        kiln_sha256: String,
        max_abs_error: f64,
        mean_abs_error: f64,
        name: String,
        reference_root_mean_square: f64,
        relative_root_mean_square_error: f64,
        root_mean_square_error: f64,
    }

    #[derive(Debug, Serialize)]
    struct ErrorGrowthBoundary {
        index: usize,
        name: String,
        relative_root_mean_square_error_delta: f64,
    }

    #[derive(Debug, Serialize)]
    struct LayerResultMarker {
        boundaries: Vec<BoundaryComparison>,
        final_logits_sha256: String,
        hf_layer_last_rows_sha256: String,
        input_token_count: usize,
        input_token_ids_sha256: String,
        kernel_policy: &'static str,
        largest_relative_error_growth: ErrorGrowthBoundary,
        observed_next_tokens: Vec<u32>,
        request_id: String,
        schema: &'static str,
    }

    pub fn main() -> Result<()> {
        let args = parse_args()?;
        let request = load_request(&args.request)?;
        let reference = load_reference(&args.reference, &request, args.layer_attribution)?;
        let hf_argmax = argmax(&reference.logits)?;
        let kernel_policy = args.kernel_profile.label();

        install_kt_api_mode(KtApiMode::Auto).context("install qualified kt API mode")?;
        install_rocm_kernel_policy(args.kernel_profile.model_policy())
            .with_context(|| format!("install {kernel_policy} ROCm model policy"))?;
        primary_rocm_context_with_execution_policy(
            0,
            RocmExecutionPolicy::new(RocmSynchronizationMode::LegacyHostBarriers)
                .with_tensor_kernel_policy(args.kernel_profile.tensor_policy()),
        )
        .with_context(|| format!("install {kernel_policy} ROCm tensor policy"))?;

        anyhow::ensure!(
            kiln_tensor::rocm_is_available(),
            "ROCm device is unavailable"
        );
        let device = Device::Rocm(0);
        let _memory_runtime = InferenceMemoryRuntime::initialize(
            device,
            GovernorConfig {
                capacity_limit_bytes: None,
                ..GovernorConfig::default()
            },
        )
        .context("initialize ROCm memory governance")?;
        let config = ModelConfig::qwen3_5_4b();
        anyhow::ensure!(
            config.dtype == ModelDType::BF16,
            "expected BF16 model config"
        );
        let raw_weights = load_model_with_options(
            &args.model,
            &config,
            LoadModelOptions {
                load_mtp: false,
                ..Default::default()
            },
        )
        .context("load Qwen3.5 model")?;
        let weights = GpuWeights::from_model_weights(&raw_weights, &config, &device)
            .context("upload Qwen3.5 weights to ROCm")?;
        drop(raw_weights);
        let runtime = backend::for_device_kt(&device);
        LinearBackend::runtime_prewarm_decode_weights(runtime.as_ref(), &weights)
            .with_context(|| format!("prewarm {kernel_policy} ROCm decode weights"))?;

        if args.layer_attribution {
            let result = run_layer_attribution(
                runtime.as_ref(),
                &weights,
                &config,
                &device,
                &request,
                &reference,
                kernel_policy,
            )?;
            println!("{LAYER_MARKER}{}", serde_json::to_string(&result)?);
            return Ok(());
        }

        let eager_full = run_eager_full(
            runtime.as_ref(),
            &weights,
            &config,
            &device,
            &request,
            &reference,
        )?;
        let eager_greedy = run_eager_greedy(
            runtime.as_ref(),
            &weights,
            &config,
            &device,
            &request,
            hf_argmax,
        )?;
        let (graph_full, graph_greedy, graph) = run_graph_paths(
            runtime.as_ref(),
            &weights,
            &config,
            &device,
            &request,
            &reference,
            hf_argmax,
        )?;

        let eager_full_matches = eager_full.comparison.argmax_equal;
        let graph_full_matches = graph_full.comparison.argmax_equal;
        let eager_greedy_matches = eager_greedy.final_token_matches_reference;
        let graph_greedy_matches = graph_greedy.final_token_matches_reference;
        let attribution = if !eager_full_matches {
            "eager_full_logits"
        } else if !graph_full_matches {
            "hip_graph_full_logits"
        } else if !eager_greedy_matches {
            "eager_greedy_selection"
        } else if !graph_greedy_matches {
            "hip_graph_greedy_selection"
        } else {
            "serving_only_or_not_reproduced"
        };
        let result = ResultMarker {
            attribution,
            eager_full,
            eager_greedy,
            graph,
            graph_full,
            graph_greedy,
            hf_argmax,
            input_token_count: request.prompt.len() + request.continuation.len(),
            input_token_ids_sha256: request.input_token_ids_sha256,
            kernel_policy,
            request_id: request.id,
            schema: RESULT_SCHEMA,
        };
        println!("{MARKER}{}", serde_json::to_string(&result)?);
        Ok(())
    }

    fn parse_args() -> Result<Args> {
        let mut values = std::env::args_os().skip(1);
        let mut parsed = BTreeMap::new();
        let mut layer_attribution = false;
        while let Some(flag) = values.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("arguments must be UTF-8"))?;
            if flag == "--layer-attribution" {
                anyhow::ensure!(!layer_attribution, "duplicate --layer-attribution");
                layer_attribution = true;
                continue;
            }
            anyhow::ensure!(
                matches!(
                    flag.as_str(),
                    "--model" | "--request" | "--hf-reference" | "--kernel-profile"
                ),
                "unknown argument {flag}"
            );
            let value = values
                .next()
                .with_context(|| format!("{flag} requires a value"))?;
            anyhow::ensure!(
                parsed.insert(flag.clone(), value).is_none(),
                "duplicate {flag}"
            );
        }
        let take = |flag: &str| -> Result<PathBuf> {
            let path = PathBuf::from(
                parsed
                    .get(flag)
                    .with_context(|| format!("missing {flag}"))?,
            );
            anyhow::ensure!(path.is_absolute(), "{flag} must be absolute");
            anyhow::ensure!(path.is_file(), "{flag} is not a regular file");
            Ok(path)
        };
        let model = PathBuf::from(parsed.get("--model").context("missing --model")?);
        anyhow::ensure!(
            model.is_absolute() && model.is_dir(),
            "--model must be an absolute directory"
        );
        let kernel_profile = parsed
            .get("--kernel-profile")
            .map(|value| {
                value
                    .to_str()
                    .context("--kernel-profile must be UTF-8")
                    .and_then(KernelProfile::parse)
            })
            .transpose()?
            .unwrap_or(KernelProfile::Qualified);
        anyhow::ensure!(
            layer_attribution || kernel_profile == KernelProfile::Qualified,
            "diagnostic kernel profiles are supported only with --layer-attribution"
        );
        Ok(Args {
            kernel_profile,
            layer_attribution,
            model,
            request: take("--request")?,
            reference: take("--hf-reference")?,
        })
    }

    fn load_request(path: &Path) -> Result<Request> {
        let document: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            document.get("schema").and_then(Value::as_str) == Some("kiln.hf-next-token-request.v1"),
            "request schema mismatch"
        );
        let id = string_field(&document, "id")?.to_owned();
        let input_token_ids_sha256 = string_field(&document, "input_token_ids_sha256")?.to_owned();
        let input = u32_array_field(&document, "input_token_ids")?;
        let prompt = u32_array_field(
            document
                .get("prompt")
                .context("request.prompt is missing")?,
            "token_ids",
        )?;
        let continuation = token_id_array(
            document
                .get("continuation_prefix")
                .context("request.continuation_prefix is missing")?,
        )?;
        let candidates = token_id_array(
            document
                .get("candidates")
                .context("request.candidates is missing")?,
        )?;
        anyhow::ensure!(
            continuation.len() == 3 && candidates.len() == 2,
            "request must declare three common continuation tokens and two candidates"
        );
        let reconstructed: Vec<u32> = prompt.iter().chain(&continuation).copied().collect();
        anyhow::ensure!(input == reconstructed, "request input IDs are inconsistent");
        Ok(Request {
            id,
            input_token_ids_sha256,
            prompt,
            continuation,
            candidates,
        })
    }

    fn string_field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
        value
            .get(name)
            .and_then(Value::as_str)
            .with_context(|| format!("{name} must be a string"))
    }

    fn u32_array_field(value: &Value, name: &str) -> Result<Vec<u32>> {
        let values = value
            .get(name)
            .and_then(Value::as_array)
            .with_context(|| format!("{name} must be an array"))?;
        values
            .iter()
            .map(|item| {
                let value = item
                    .as_u64()
                    .context("token ID must be an unsigned integer")?;
                u32::try_from(value).context("token ID does not fit u32")
            })
            .collect()
    }

    fn token_id_array(value: &Value) -> Result<Vec<u32>> {
        let rows = value.as_array().context("token rows must be an array")?;
        rows.iter()
            .map(|row| {
                let token = row
                    .get("token_id")
                    .and_then(Value::as_u64)
                    .context("token row requires an unsigned token_id")?;
                u32::try_from(token).context("token_id does not fit u32")
            })
            .collect()
    }

    fn load_reference(
        path: &Path,
        request: &Request,
        layer_attribution: bool,
    ) -> Result<Reference> {
        let data =
            std::fs::read(path).with_context(|| format!("read HF reference {}", path.display()))?;
        let (_, metadata) = SafeTensors::read_metadata(&data).context("read HF metadata")?;
        let user = metadata
            .metadata()
            .as_ref()
            .context("HF metadata is absent")?;
        let expected_schema = if layer_attribution {
            HF_LAYER_SCHEMA
        } else {
            HF_SCHEMA
        };
        anyhow::ensure!(
            user.get("schema").map(String::as_str) == Some(expected_schema),
            "HF reference schema mismatch"
        );
        anyhow::ensure!(
            user.get("attention_implementation").map(String::as_str) == Some("eager")
                && user
                    .get("linear_attention_implementation")
                    .map(String::as_str)
                    == Some("transformers_torch_fallback"),
            "HF reference implementation mismatch"
        );
        let tensors = SafeTensors::deserialize(&data).context("deserialize HF reference")?;
        let expected_names: Vec<&str> = if layer_attribution {
            vec!["input_ids", "layer_last_rows", "logits"]
        } else {
            vec!["input_ids", "logits"]
        };
        anyhow::ensure!(
            tensors.names().len() == expected_names.len()
                && expected_names
                    .iter()
                    .all(|name| tensors.names().contains(name)),
            "HF reference tensor inventory is inconsistent"
        );
        let input = tensors.tensor("input_ids")?;
        anyhow::ensure!(
            input.dtype() == safetensors::Dtype::I64,
            "HF input_ids must be I64"
        );
        let input_ids: Vec<u32> = input
            .data()
            .chunks_exact(8)
            .map(|bytes| {
                let value = i64::from_le_bytes(bytes.try_into().expect("I64 chunk"));
                u32::try_from(value).context("HF input token does not fit u32")
            })
            .collect::<Result<_>>()?;
        let expected: Vec<u32> = request
            .prompt
            .iter()
            .chain(&request.continuation)
            .copied()
            .collect();
        anyhow::ensure!(input_ids == expected, "HF input IDs do not match request");
        let tensor = tensors.tensor("logits")?;
        anyhow::ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "HF logits must be F32"
        );
        let logits: Vec<f32> = tensor
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 chunk")))
            .collect();
        anyhow::ensure!(
            tensor.shape() == [logits.len()] && logits.iter().all(|value| value.is_finite()),
            "HF logits are malformed or non-finite"
        );
        let (boundary_names, layer_last_rows) = if layer_attribution {
            let names: Vec<String> = serde_json::from_str(
                user.get("boundary_names")
                    .context("HF layer reference omits boundary_names")?,
            )
            .context("parse HF boundary_names")?;
            let layer_tensor = tensors.tensor("layer_last_rows")?;
            anyhow::ensure!(
                layer_tensor.dtype() == safetensors::Dtype::F32,
                "HF layer_last_rows must be F32"
            );
            let shape = layer_tensor.shape();
            anyhow::ensure!(
                shape.len() == 2 && shape[0] == names.len() && shape[1] > 0 && names.len() == 34,
                "HF layer_last_rows shape or boundary count is invalid"
            );
            let values: Vec<f32> = layer_tensor
                .data()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 chunk")))
                .collect();
            anyhow::ensure!(
                values.len() == shape[0] * shape[1] && values.iter().all(|value| value.is_finite()),
                "HF layer_last_rows are malformed or non-finite"
            );
            let rows = values.chunks_exact(shape[1]).map(<[f32]>::to_vec).collect();
            (Some(names), Some(rows))
        } else {
            (None, None)
        };
        Ok(Reference {
            boundary_names,
            layer_last_rows,
            logits,
        })
    }

    fn new_cache(config: &ModelConfig, device: Device) -> Result<PagedKvCacheKt> {
        let dtype = match config.dtype {
            ModelDType::BF16 => DType::BF16,
            ModelDType::FP16 => DType::F16,
            ModelDType::FP32 => DType::F32,
        };
        PagedKvCacheKt::new(
            config.num_full_attention_layers,
            NUM_BLOCKS,
            BLOCK_SIZE,
            config.num_kv_heads,
            config.head_dim,
            dtype,
            device,
        )
    }

    fn block_table(token_count: usize) -> BlockTable {
        BlockTable {
            blocks: (0..token_count.div_ceil(BLOCK_SIZE) as u32).collect(),
        }
    }

    fn prefill_full(
        runtime: &dyn kiln_model::BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        cache: &PagedKvCacheKt,
        table: &BlockTable,
        state: &mut LinearAttentionState,
        prompt: &[u32],
    ) -> Result<(Tensor, u32)> {
        let logits = model_forward_paged_last_token(
            runtime,
            prompt,
            weights,
            config,
            cache,
            table,
            0,
            Some(state),
            None,
            None,
        )?;
        let token = tensor_argmax(&logits)?;
        Ok((logits, token))
    }

    fn run_eager_full(
        runtime: &dyn kiln_model::BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        device: &Device,
        request: &Request,
        reference: &Reference,
    ) -> Result<FullPathResult> {
        let cache = new_cache(config, *device)?;
        let table = block_table(request.prompt.len() + request.continuation.len() + 1);
        let mut state = LinearAttentionState::new_for_inference(config, device)?;
        let (mut logits, first) = prefill_full(
            runtime,
            weights,
            config,
            &cache,
            &table,
            &mut state,
            &request.prompt,
        )?;
        let mut observed = vec![first];
        for (offset, &token) in request.continuation.iter().enumerate() {
            logits = model_forward_paged_last_token(
                runtime,
                &[token],
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                Some(&mut state),
                None,
                None,
            )?;
            observed.push(tensor_argmax(&logits)?);
        }
        let values = tensor_logits(&logits)?;
        Ok(FullPathResult {
            comparison: compare_logits(&values, &reference.logits, &request.candidates)?,
            observed_next_tokens: observed,
        })
    }

    fn run_layer_attribution(
        runtime: &dyn kiln_model::BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        device: &Device,
        request: &Request,
        reference: &Reference,
        kernel_policy: &'static str,
    ) -> Result<LayerResultMarker> {
        let reference_rows = reference
            .layer_last_rows
            .as_ref()
            .context("HF layer reference rows are absent")?;
        let boundary_names = reference
            .boundary_names
            .as_ref()
            .context("HF layer boundary names are absent")?;
        let mut expected_names = vec!["embedding".to_owned()];
        expected_names.extend((0..config.num_layers).map(|index| {
            let layer_type = if config.is_full_attention_layer(index) {
                "full_attention"
            } else {
                "linear_attention"
            };
            format!("layer_{index:02}_{layer_type}")
        }));
        expected_names.push("final_norm".to_owned());
        anyhow::ensure!(
            boundary_names == &expected_names && reference_rows.len() == expected_names.len(),
            "HF layer boundaries do not match the Qwen3.5 model configuration"
        );

        let cache = new_cache(config, *device)?;
        let table = block_table(request.prompt.len() + request.continuation.len() + 1);
        let mut state = LinearAttentionState::new_for_inference(config, device)?;
        let (_, first) = prefill_full(
            runtime,
            weights,
            config,
            &cache,
            &table,
            &mut state,
            &request.prompt,
        )?;
        let mut observed = vec![first];
        for (offset, &token) in request.continuation[..2].iter().enumerate() {
            let logits = model_forward_paged_last_token(
                runtime,
                &[token],
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                Some(&mut state),
                None,
                None,
            )?;
            observed.push(tensor_argmax(&logits)?);
        }
        let (logits, snapshots) = model_forward_paged_last_token_with_layer_snapshots(
            runtime,
            &request.continuation[2..],
            weights,
            config,
            &cache,
            &table,
            request.prompt.len() + 2,
            Some(&mut state),
            None,
        )?;
        observed.push(tensor_argmax(&logits)?);
        anyhow::ensure!(
            snapshots.len() == reference_rows.len(),
            "Kiln and HF layer snapshot counts differ"
        );

        let mut boundaries = Vec::with_capacity(snapshots.len());
        for (index, ((snapshot, expected), name)) in snapshots
            .iter()
            .zip(reference_rows)
            .zip(boundary_names)
            .enumerate()
        {
            let actual = tensor_logits(snapshot)?;
            boundaries.push(compare_boundary(index, name, &actual, expected)?);
        }
        let mut previous = 0.0_f64;
        let mut largest_index = 0;
        let mut largest_delta = f64::NEG_INFINITY;
        for boundary in &boundaries {
            let delta = boundary.relative_root_mean_square_error - previous;
            if delta > largest_delta {
                largest_delta = delta;
                largest_index = boundary.index;
            }
            previous = boundary.relative_root_mean_square_error;
        }
        let final_values = tensor_logits(&logits)?;
        Ok(LayerResultMarker {
            boundaries,
            final_logits_sha256: vector_sha256(&final_values),
            hf_layer_last_rows_sha256: matrix_sha256(reference_rows),
            input_token_count: request.prompt.len() + request.continuation.len(),
            input_token_ids_sha256: request.input_token_ids_sha256.clone(),
            kernel_policy,
            largest_relative_error_growth: ErrorGrowthBoundary {
                index: largest_index,
                name: boundary_names[largest_index].clone(),
                relative_root_mean_square_error_delta: largest_delta,
            },
            observed_next_tokens: observed,
            request_id: request.id.clone(),
            schema: LAYER_RESULT_SCHEMA,
        })
    }

    fn compare_boundary(
        index: usize,
        name: &str,
        actual: &[f32],
        reference: &[f32],
    ) -> Result<BoundaryComparison> {
        anyhow::ensure!(
            actual.len() == reference.len() && !actual.is_empty(),
            "layer boundary width mismatch"
        );
        let mut max_abs = 0.0_f64;
        let mut abs_sum = 0.0_f64;
        let mut squared_error = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut actual_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        for (&observed, &expected) in actual.iter().zip(reference) {
            let observed = f64::from(observed);
            let expected = f64::from(expected);
            let error = observed - expected;
            let abs = error.abs();
            max_abs = max_abs.max(abs);
            abs_sum += abs;
            squared_error += error * error;
            dot += observed * expected;
            actual_squared += observed * observed;
            reference_squared += expected * expected;
        }
        let count = actual.len() as f64;
        let root_mean_square_error = (squared_error / count).sqrt();
        let reference_root_mean_square = (reference_squared / count).sqrt();
        anyhow::ensure!(
            reference_root_mean_square > 0.0 && actual_squared > 0.0,
            "layer boundary has zero RMS magnitude"
        );
        Ok(BoundaryComparison {
            cosine_similarity: dot / (actual_squared.sqrt() * reference_squared.sqrt()),
            hf_sha256: vector_sha256(reference),
            index,
            kiln_sha256: vector_sha256(actual),
            max_abs_error: max_abs,
            mean_abs_error: abs_sum / count,
            name: name.to_owned(),
            reference_root_mean_square,
            relative_root_mean_square_error: root_mean_square_error / reference_root_mean_square,
            root_mean_square_error,
        })
    }

    fn run_eager_greedy(
        runtime: &dyn kiln_model::BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        device: &Device,
        request: &Request,
        hf_argmax: u32,
    ) -> Result<GreedyPathResult> {
        let cache = new_cache(config, *device)?;
        let table = block_table(request.prompt.len() + request.continuation.len() + 1);
        let mut state = LinearAttentionState::new_for_inference(config, device)?;
        let first = model_forward_paged_last_token_greedy(
            runtime,
            &request.prompt,
            weights,
            config,
            &cache,
            &table,
            0,
            Some(&mut state),
            None,
            None,
        )?;
        let mut observed = vec![first];
        for (offset, &token) in request.continuation.iter().enumerate() {
            observed.push(model_forward_paged_next_token_greedy(
                runtime,
                token,
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                Some(&mut state),
                None,
                None,
            )?);
        }
        Ok(GreedyPathResult {
            final_token_matches_reference: observed.last().copied() == Some(hf_argmax),
            observed_next_tokens: observed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_graph_paths(
        runtime: &dyn kiln_model::BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        device: &Device,
        request: &Request,
        reference: &Reference,
        hf_argmax: u32,
    ) -> Result<(FullPathResult, GreedyPathResult, GraphEvidence)> {
        let cache = new_cache(config, *device)?;
        let table = block_table(request.prompt.len() + request.continuation.len() + 16);
        let mut runner =
            RocmGraphRunner::new(device, RocmGraphExecutionPolicy::lazy_capture_replay());

        // Prime the exact FA2 bucket, then release the logical row. The exact
        // comparison rows must reuse a retained slot and execute replay, as the
        // source serving process had already completed its warmup request.
        let mut prime_state = LinearAttentionState::new_for_inference(config, device)?;
        let _ = prefill_full(
            runtime,
            weights,
            config,
            &cache,
            &table,
            &mut prime_state,
            &request.prompt,
        )?;
        let mut prime_tokens = request.continuation.clone();
        prime_tokens.extend(request.candidates.iter().copied().cycle().take(6));
        for (offset, token) in prime_tokens.into_iter().enumerate() {
            let _ = runner.decode_step_paged(
                runtime,
                token,
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                &mut prime_state,
                None,
                GRAPH_ROW_PRIME,
            )?;
            let stats = runner.stats();
            if stats.capture_successes > 0 && stats.replay_successes > 0 {
                break;
            }
        }
        let primed = runner.stats();
        anyhow::ensure!(
            primed.capture_successes > 0
                && primed.cache_admission_successes > 0
                && primed.replay_successes > 0
                && primed.failures == 0,
            "HIP graph priming did not produce one retained capture and replay: {primed:?}"
        );
        runner.release_decode_row(GRAPH_ROW_PRIME);

        let mut full_state = LinearAttentionState::new_for_inference(config, device)?;
        let (mut logits, first) = prefill_full(
            runtime,
            weights,
            config,
            &cache,
            &table,
            &mut full_state,
            &request.prompt,
        )?;
        let mut full_observed = vec![first];
        let replays_before_full = runner.stats().replay_successes;
        for (offset, &token) in request.continuation.iter().enumerate() {
            logits = runner.decode_step_paged(
                runtime,
                token,
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                &mut full_state,
                None,
                GRAPH_ROW_FULL,
            )?;
            full_observed.push(tensor_argmax(&logits)?);
        }
        anyhow::ensure!(
            runner.stats().replay_successes - replays_before_full
                == request.continuation.len() as u64,
            "graph-full exact row did not replay every continuation step"
        );
        let graph_values = tensor_logits(&logits)?;
        runner.release_decode_row(GRAPH_ROW_FULL);

        let mut greedy_state = LinearAttentionState::new_for_inference(config, device)?;
        let first = model_forward_paged_last_token_greedy(
            runtime,
            &request.prompt,
            weights,
            config,
            &cache,
            &table,
            0,
            Some(&mut greedy_state),
            None,
            None,
        )?;
        let mut greedy_observed = vec![first];
        let replays_before_greedy = runner.stats().replay_successes;
        for (offset, &token) in request.continuation.iter().enumerate() {
            greedy_observed.push(runner.decode_step_paged_greedy(
                runtime,
                token,
                weights,
                config,
                &cache,
                &table,
                request.prompt.len() + offset,
                &mut greedy_state,
                None,
                GRAPH_ROW_GREEDY,
            )?);
        }
        anyhow::ensure!(
            runner.stats().replay_successes - replays_before_greedy
                == request.continuation.len() as u64,
            "graph-greedy exact row did not replay every continuation step"
        );
        runner.release_decode_row(GRAPH_ROW_GREEDY);
        let stats = runner.stats();
        anyhow::ensure!(
            stats.failures == 0 && stats.fallbacks.total == 0,
            "graph path fell back"
        );
        Ok((
            FullPathResult {
                comparison: compare_logits(&graph_values, &reference.logits, &request.candidates)?,
                observed_next_tokens: full_observed,
            },
            GreedyPathResult {
                final_token_matches_reference: greedy_observed.last().copied() == Some(hf_argmax),
                observed_next_tokens: greedy_observed,
            },
            GraphEvidence {
                cache_admission_successes: stats.cache_admission_successes,
                capture_attempts: stats.capture_attempts,
                capture_failures: stats.capture_failures,
                capture_successes: stats.capture_successes,
                enabled: stats.enabled,
                fallbacks: stats.fallbacks.total,
                replay_attempts: stats.replay_attempts,
                replay_failures: stats.replay_failures,
                replay_successes: stats.replay_successes,
            },
        ))
    }

    fn tensor_logits(tensor: &Tensor) -> Result<Vec<f32>> {
        let values = tensor
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "Kiln logits are non-finite"
        );
        Ok(values)
    }

    fn tensor_argmax(tensor: &Tensor) -> Result<u32> {
        argmax(&tensor_logits(tensor)?)
    }

    fn argmax(values: &[f32]) -> Result<u32> {
        let (index, _) = values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .context("logits are empty")?;
        u32::try_from(index).context("argmax does not fit u32")
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

    fn vector_sha256(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        sha256_hex(hasher)
    }

    fn matrix_sha256(rows: &[Vec<f32>]) -> String {
        let mut hasher = Sha256::new();
        for value in rows.iter().flatten() {
            hasher.update(value.to_le_bytes());
        }
        sha256_hex(hasher)
    }

    fn sha256_hex(hasher: Sha256) -> String {
        let digest = hasher.finalize();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("sha256:{hex}")
    }

    fn compare_logits(
        actual: &[f32],
        reference: &[f32],
        candidates: &[u32],
    ) -> Result<LogitComparison> {
        anyhow::ensure!(
            actual.len() == reference.len() && !actual.is_empty(),
            "logit vocabulary mismatch"
        );
        let mut max_abs = 0.0_f64;
        let mut abs_sum = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut actual_norm = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        for (&observed, &expected) in actual.iter().zip(reference) {
            let observed = f64::from(observed);
            let expected = f64::from(expected);
            let abs = (observed - expected).abs();
            max_abs = max_abs.max(abs);
            abs_sum += abs;
            dot += observed * expected;
            actual_norm += observed * observed;
            reference_norm += expected * expected;
        }
        let actual_argmax = argmax(actual)?;
        let reference_argmax = argmax(reference)?;
        let actual_top10 = top_k(actual, 10);
        let reference_top10 = top_k(reference, 10);
        let mut candidate_tokens = Vec::with_capacity(candidates.len());
        for &token_id in candidates {
            let index = usize::try_from(token_id)?;
            let value = *actual
                .get(index)
                .context("candidate token exceeds vocabulary")?;
            let rank = actual
                .iter()
                .enumerate()
                .filter(|&(other_index, other)| {
                    *other > value || (*other == value && other_index < index)
                })
                .count()
                + 1;
            candidate_tokens.push(CandidateLogit {
                logit: value,
                rank,
                token_id,
            });
        }
        Ok(LogitComparison {
            argmax: actual_argmax,
            argmax_equal: actual_argmax == reference_argmax,
            candidate_tokens,
            cosine_similarity: dot / (actual_norm.sqrt() * reference_norm.sqrt()),
            logits_sha256: vector_sha256(actual),
            max_abs_error: max_abs,
            mean_abs_error: abs_sum / actual.len() as f64,
            top10_overlap: actual_top10
                .iter()
                .filter(|index| reference_top10.contains(index))
                .count(),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn diagnostic_profiles_split_model_and_tensor_route_groups() {
            let model = KernelProfile::parse("model_fallback").unwrap();
            assert_eq!(model.model_policy(), RocmKernelPolicy::portable_fallback());
            assert_eq!(model.tensor_policy(), RocmTensorKernelPolicy::qualified());
            assert_eq!(model.label(), "model_fallback");

            let tensor = KernelProfile::parse("tensor_fallback").unwrap();
            assert_eq!(tensor.model_policy(), RocmKernelPolicy::qualified());
            assert_eq!(
                tensor.tensor_policy(),
                RocmTensorKernelPolicy::portable_fallback()
            );
            assert_eq!(tensor.label(), "tensor_fallback");

            let fused_fallback = KernelProfile::parse("fused_norm_mlp_fallback").unwrap();
            assert_eq!(
                fused_fallback.model_policy(),
                RocmKernelPolicy::fused_norm_mlp_fallback()
            );
            assert_eq!(
                fused_fallback.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(fused_fallback.label(), "fused_norm_mlp_fallback");

            let fused_only = KernelProfile::parse("fused_norm_mlp_only").unwrap();
            assert_eq!(
                fused_only.model_policy(),
                RocmKernelPolicy::fused_norm_mlp_only()
            );
            assert_eq!(
                fused_only.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(fused_only.label(), "fused_norm_mlp_only");

            let fused_rmsnorm = KernelProfile::parse("fused_rmsnorm_fallback").unwrap();
            assert_eq!(
                fused_rmsnorm.model_policy(),
                RocmKernelPolicy::fused_rmsnorm_fallback()
            );
            assert_eq!(
                fused_rmsnorm.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(fused_rmsnorm.label(), "fused_rmsnorm_fallback");

            let fused_silu = KernelProfile::parse("fused_mlp_silu_mul_fallback").unwrap();
            assert_eq!(
                fused_silu.model_policy(),
                RocmKernelPolicy::fused_mlp_silu_mul_fallback()
            );
            assert_eq!(
                fused_silu.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(fused_silu.label(), "fused_mlp_silu_mul_fallback");

            let fused_gate_up = KernelProfile::parse("fused_mlp_gate_up_prefill_fallback").unwrap();
            assert_eq!(
                fused_gate_up.model_policy(),
                RocmKernelPolicy::fused_mlp_gate_up_prefill_fallback()
            );
            assert_eq!(
                fused_gate_up.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(fused_gate_up.label(), "fused_mlp_gate_up_prefill_fallback");

            let gdn = KernelProfile::parse("gdn_fallback").unwrap();
            assert_eq!(gdn.model_policy(), RocmKernelPolicy::gdn_fallback());
            assert_eq!(gdn.tensor_policy(), RocmTensorKernelPolicy::qualified());
            assert_eq!(gdn.label(), "gdn_fallback");

            let non_gdn = KernelProfile::parse("non_gdn_fallback").unwrap();
            assert_eq!(non_gdn.model_policy(), RocmKernelPolicy::non_gdn_fallback());
            assert_eq!(non_gdn.tensor_policy(), RocmTensorKernelPolicy::qualified());
            assert_eq!(non_gdn.label(), "non_gdn_fallback");

            let split_fallback = KernelProfile::parse("split_q_gate_fallback").unwrap();
            assert_eq!(
                split_fallback.model_policy(),
                RocmKernelPolicy::split_q_gate_fallback()
            );
            assert_eq!(
                split_fallback.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(split_fallback.label(), "split_q_gate_fallback");

            let split_only = KernelProfile::parse("split_q_gate_only").unwrap();
            assert_eq!(
                split_only.model_policy(),
                RocmKernelPolicy::split_q_gate_only()
            );
            assert_eq!(
                split_only.tensor_policy(),
                RocmTensorKernelPolicy::qualified()
            );
            assert_eq!(split_only.label(), "split_q_gate_only");

            assert!(KernelProfile::parse("individual_switches").is_err());
        }

        #[test]
        fn tensor_logits_explicitly_casts_bf16() {
            let tensor = Tensor::from_vec(vec![1.25_f32, -2.5], (2,))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            assert_eq!(tensor_logits(&tensor).unwrap(), vec![1.25, -2.5]);
        }
    }
}

#[cfg(feature = "rocm")]
fn main() {
    if let Err(error) = rocm::main() {
        eprintln!("ROCm/HF path attribution failed: {error:#}");
        std::process::exit(1);
    }
}

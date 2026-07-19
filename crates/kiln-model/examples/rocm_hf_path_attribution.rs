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
        model_forward_paged_last_token_greedy, model_forward_paged_next_token_greedy,
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
    const RESULT_SCHEMA: &str = "kiln.rocm-hf-path-attribution.v1";
    const HF_SCHEMA: &str = "kiln.qwen35-hf-full-logits.v1";
    const BLOCK_SIZE: usize = 16;
    const NUM_BLOCKS: usize = 32;
    const GRAPH_ROW_PRIME: u64 = 1;
    const GRAPH_ROW_FULL: u64 = 2;
    const GRAPH_ROW_GREEDY: u64 = 3;

    #[derive(Debug)]
    struct Args {
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
    struct CgroupEvidence {
        memory_current_bytes: u64,
        memory_high_events: u64,
        memory_max_bytes: u64,
        memory_max_events: u64,
        memory_oom_events: u64,
        memory_oom_kill_events: u64,
        memory_peak_bytes: u64,
        memory_swap_bytes: u64,
        memory_swap_max_bytes: u64,
    }

    #[derive(Debug, Serialize)]
    struct ResultMarker {
        attribution: &'static str,
        containment: CgroupEvidence,
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

    pub fn main() -> Result<()> {
        let args = parse_args()?;
        let request = load_request(&args.request)?;
        let reference = load_reference(&args.reference, &request)?;
        let hf_argmax = argmax(&reference.logits)?;

        install_kt_api_mode(KtApiMode::Auto).context("install qualified kt API mode")?;
        install_rocm_kernel_policy(RocmKernelPolicy::qualified())
            .context("install qualified ROCm model policy")?;
        primary_rocm_context_with_execution_policy(
            0,
            RocmExecutionPolicy::new(RocmSynchronizationMode::LegacyHostBarriers)
                .with_tensor_kernel_policy(RocmTensorKernelPolicy::qualified()),
        )
        .context("install qualified ROCm tensor policy")?;

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
        let raw_weights =
            load_model_with_options(&args.model, &config, LoadModelOptions { load_mtp: false })
                .context("load Qwen3.5 model")?;
        let weights = GpuWeights::from_model_weights(&raw_weights, &config, &device)
            .context("upload Qwen3.5 weights to ROCm")?;
        drop(raw_weights);
        let runtime = backend::for_device_kt(&device);
        LinearBackend::runtime_prewarm_decode_weights(runtime.as_ref(), &weights)
            .context("prewarm qualified ROCm decode weights")?;

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
            containment: read_cgroup_evidence()?,
            eager_full,
            eager_greedy,
            graph,
            graph_full,
            graph_greedy,
            hf_argmax,
            input_token_count: request.prompt.len() + request.continuation.len(),
            input_token_ids_sha256: request.input_token_ids_sha256,
            kernel_policy: "qualified",
            request_id: request.id,
            schema: RESULT_SCHEMA,
        };
        println!("{MARKER}{}", serde_json::to_string(&result)?);
        Ok(())
    }

    fn parse_args() -> Result<Args> {
        let mut values = std::env::args_os().skip(1);
        let mut parsed = BTreeMap::new();
        while let Some(flag) = values.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("arguments must be UTF-8"))?;
            anyhow::ensure!(
                matches!(flag.as_str(), "--model" | "--request" | "--hf-reference"),
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
        Ok(Args {
            model,
            request: take("--request")?,
            reference: take("--hf-reference")?,
        })
    }

    fn read_cgroup_evidence() -> Result<CgroupEvidence> {
        let membership = std::fs::read_to_string("/proc/self/cgroup")?;
        let relative = membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .context("unified cgroup membership is absent")?;
        let cgroup = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
        let read_u64 = |name: &str| -> Result<u64> {
            let value = std::fs::read_to_string(cgroup.join(name))?;
            anyhow::ensure!(value.trim() != "max", "{name} must be bounded");
            value
                .trim()
                .parse::<u64>()
                .with_context(|| format!("parse {name}"))
        };
        let events = std::fs::read_to_string(cgroup.join("memory.events"))?;
        let event = |name: &str| -> Result<u64> {
            events
                .lines()
                .find_map(|line| {
                    let mut fields = line.split_ascii_whitespace();
                    (fields.next() == Some(name))
                        .then(|| fields.next())
                        .flatten()
                })
                .with_context(|| format!("memory.events omits {name}"))?
                .parse::<u64>()
                .with_context(|| format!("parse memory.events {name}"))
        };
        let evidence = CgroupEvidence {
            memory_current_bytes: read_u64("memory.current")?,
            memory_high_events: event("high")?,
            memory_max_bytes: read_u64("memory.max")?,
            memory_max_events: event("max")?,
            memory_oom_events: event("oom")?,
            memory_oom_kill_events: event("oom_kill")?,
            memory_peak_bytes: read_u64("memory.peak")?,
            memory_swap_bytes: read_u64("memory.swap.current")?,
            memory_swap_max_bytes: read_u64("memory.swap.max")?,
        };
        anyhow::ensure!(
            evidence.memory_high_events == 0
                && evidence.memory_max_events == 0
                && evidence.memory_oom_events == 0
                && evidence.memory_oom_kill_events == 0
                && evidence.memory_swap_bytes == 0
                && evidence.memory_swap_max_bytes == 0,
            "cgroup memory containment was not clean: {evidence:?}"
        );
        Ok(evidence)
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

    fn load_reference(path: &Path, request: &Request) -> Result<Reference> {
        let data =
            std::fs::read(path).with_context(|| format!("read HF reference {}", path.display()))?;
        let (_, metadata) = SafeTensors::read_metadata(&data).context("read HF metadata")?;
        let user = metadata
            .metadata()
            .as_ref()
            .context("HF metadata is absent")?;
        anyhow::ensure!(
            user.get("schema").map(String::as_str) == Some(HF_SCHEMA),
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
        anyhow::ensure!(
            tensors.names().len() == 2
                && tensors.names().contains(&"input_ids")
                && tensors.names().contains(&"logits"),
            "HF reference must contain exactly input_ids and logits"
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
        Ok(Reference { logits })
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
        let mut hasher = Sha256::new();
        for value in actual {
            hasher.update(value.to_le_bytes());
        }
        let digest = hasher.finalize();
        let logits_sha256 = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(LogitComparison {
            argmax: actual_argmax,
            argmax_equal: actual_argmax == reference_argmax,
            candidate_tokens,
            cosine_similarity: dot / (actual_norm.sqrt() * reference_norm.sqrt()),
            logits_sha256: format!("sha256:{logits_sha256}"),
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

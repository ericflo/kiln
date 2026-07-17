use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

type GraphContractResult = Result<(), kiln_graph::CaptureError>;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    manifest_dir()
        .parent()
        .and_then(Path::parent)
        .expect("kiln-model should live under workspace crates/")
        .to_path_buf()
}

fn assert_supplemental_command(gate: &Value, scope_fragment: &str, command_fragment: &str) {
    let supplemental_commands = gate["supplemental_commands"]
        .as_array()
        .expect("supplemental_commands should be an array");
    assert!(
        supplemental_commands.iter().any(|entry| {
            let scope = entry["scope"].as_str().unwrap_or("");
            let command = entry["command"].as_str().unwrap_or("");
            scope.contains(scope_fragment) && command.contains(command_fragment)
        }),
        "gate should include supplemental {scope_fragment} command fragment {command_fragment}"
    );
}

#[test]
fn generated_capability_report_uses_typed_support_states() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");
    let valid = [
        "Native",
        "NativeWithConstraints",
        "HostFallbackAllowed",
        "Declined",
        "Unsupported",
        "DisabledByEnv",
        "RequiresFeature",
    ];
    let mut failures = Vec::new();

    let backends = report["backends"]
        .as_object()
        .expect("report backends should be an object");
    for (backend, info) in backends {
        let support_methods = info["support_methods"]
            .as_object()
            .expect("support_methods should be an object");
        for (method, entry) in support_methods {
            let Some(state) = entry["support_state"].as_str() else {
                failures.push(format!("{backend}.{method} missing support_state"));
                continue;
            };
            if !valid.contains(&state) {
                failures.push(format!(
                    "{backend}.{method} has invalid support_state={state}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "invalid generated support states:\n{}",
        failures.join("\n")
    );
}

#[test]
fn generated_capability_report_lists_backend_source_modules() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let backends = report["backends"]
        .as_object()
        .expect("report backends should be an object");
    for backend in ["cuda", "rocm", "metal", "vulkan"] {
        let source_modules = backends[backend]["source_modules"]
            .as_array()
            .expect("backend source_modules should be an array")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert!(
            !source_modules.is_empty(),
            "{backend} should list at least one source module"
        );
    }

    let cuda_sources = backends["cuda"]["source_modules"]
        .as_array()
        .expect("cuda source_modules should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        cuda_sources.contains(&"crates/kiln-model/src/backend/cuda_rocm_common.rs"),
        "CUDA backend source modules should include the shared CUDA/ROCm helper module"
    );

    let rocm_sources = backends["rocm"]["source_modules"]
        .as_array()
        .expect("rocm source_modules should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        rocm_sources.contains(&"crates/kiln-model/src/backend/cuda_rocm_common.rs"),
        "ROCm backend source modules should include the shared CUDA/ROCm helper module"
    );

    let metal_sources = backends["metal"]["source_modules"]
        .as_array()
        .expect("metal source_modules should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_training.rs"),
        "Metal backend source modules should include the extracted training module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_attention.rs"),
        "Metal backend source modules should include the extracted attention module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_residency.rs"),
        "Metal backend source modules should include the extracted residency module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_config.rs"),
        "Metal backend source modules should include the extracted config module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_conv1d.rs"),
        "Metal backend source modules should include the extracted conv1d module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_core.rs"),
        "Metal backend source modules should include the extracted core module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_dense.rs"),
        "Metal backend source modules should include the extracted dense projection module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_gdn.rs"),
        "Metal backend source modules should include the extracted GDN module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_icb.rs"),
        "Metal backend source modules should include the extracted ICB module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_lm_head.rs"),
        "Metal backend source modules should include the extracted lm-head module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_msl.rs"),
        "Metal backend source modules should include the extracted MSL source module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_norm.rs"),
        "Metal backend source modules should include the extracted norm/rotary module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_paged.rs"),
        "Metal backend source modules should include the extracted paged attention/KV module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_pipeline.rs"),
        "Metal backend source modules should include the extracted pipeline module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_precompile.rs"),
        "Metal backend source modules should include the extracted precompile module"
    );
    assert!(
        metal_sources.contains(&"crates/kiln-model/src/backend/metal_runtime.rs"),
        "Metal backend source modules should include the extracted runtime facade module"
    );

    let vulkan_sources = backends["vulkan"]["source_modules"]
        .as_array()
        .expect("vulkan source_modules should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_attention.rs"),
        "Vulkan backend source modules should include the extracted attention module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_training.rs"),
        "Vulkan backend source modules should include the extracted training module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_config.rs"),
        "Vulkan backend source modules should include the extracted config module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_conv1d.rs"),
        "Vulkan backend source modules should include the extracted conv1d module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_decode_state.rs"),
        "Vulkan backend source modules should include the extracted decode state module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_dense.rs"),
        "Vulkan backend source modules should include the extracted dense projection module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_device.rs"),
        "Vulkan backend source modules should include the extracted device module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_gdn.rs"),
        "Vulkan backend source modules should include the extracted GDN module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_linear.rs"),
        "Vulkan backend source modules should include the extracted linear module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_residency.rs"),
        "Vulkan backend source modules should include the extracted residency module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_resources.rs"),
        "Vulkan backend source modules should include the extracted resource cache module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_tensor_bridge.rs"),
        "Vulkan backend source modules should include the extracted tensor bridge module"
    );
    assert!(
        vulkan_sources.contains(&"crates/kiln-model/src/backend/vulkan_weights.rs"),
        "Vulkan backend source modules should include the extracted weights module"
    );
}

#[test]
fn generated_capability_report_lists_request_descriptors() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let descriptors = report["request_descriptors"]
        .as_object()
        .expect("request_descriptors should be an object");
    for name in [
        "AttentionRequest",
        "MatmulRequest",
        "MatmulBlasRequest",
        "LinearRequest",
        "ReplayRequest",
    ] {
        assert!(
            descriptors.contains_key(name),
            "{name} should be present in request_descriptors"
        );
        let descriptor = &descriptors[name];
        for flag in [
            "has_dtype",
            "has_shape",
            "has_layout",
            "has_batch",
            "has_replay_safe",
        ] {
            assert_eq!(
                descriptor[flag], true,
                "{name} should expose {flag} in the typed request audit"
            );
        }
    }

    let replay_fields = descriptors["ReplayRequest"]["fields"]
        .as_array()
        .expect("ReplayRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(replay_fields.contains(&"dtype"));
    assert!(replay_fields.contains(&"replay_safe"));
    assert_eq!(
        descriptors["ReplayRequest"]["has_shape"], true,
        "ReplayRequest should carry replay shape metadata for replay key derivation"
    );
    assert_eq!(
        descriptors["ReplayRequest"]["has_layout"], true,
        "ReplayRequest should carry replay resource layout metadata"
    );
    assert!(
        replay_fields.contains(&"replay_shape"),
        "ReplayRequest should include replay_shape"
    );
    assert!(
        replay_fields.contains(&"layout"),
        "ReplayRequest should include layout"
    );

    let attention = &descriptors["AttentionRequest"];
    assert_eq!(
        attention["has_shape"], true,
        "AttentionRequest should carry shape metadata for prefill/decode capability queries"
    );
    assert_eq!(
        attention["has_layout"], true,
        "AttentionRequest should carry layout metadata for prefill/decode capability queries"
    );
    let attention_fields = attention["fields"]
        .as_array()
        .expect("AttentionRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in ["q_shape", "k_shape", "v_shape", "output_shape", "layout"] {
        assert!(
            attention_fields.contains(&field),
            "AttentionRequest should include {field}"
        );
    }

    let linear = &descriptors["LinearRequest"];
    assert_eq!(
        linear["has_shape"], true,
        "LinearRequest should carry shape metadata for decode/lm-head capability queries"
    );
    assert_eq!(
        linear["has_layout"], true,
        "LinearRequest should carry layout metadata for decode/lm-head capability queries"
    );
    let linear_fields = linear["fields"]
        .as_array()
        .expect("LinearRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in ["input_shape", "weight_shape", "output_shape", "layout"] {
        assert!(
            linear_fields.contains(&field),
            "LinearRequest should include {field}"
        );
    }

    let matmul_blas_fields = descriptors["MatmulBlasRequest"]["fields"]
        .as_array()
        .expect("MatmulBlasRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "m",
        "n",
        "k",
        "dtype",
        "lhs_dtype",
        "rhs_dtype",
        "out_dtype",
        "accumulation",
        "epilogue",
        "replay_safe",
        "concurrent_streams",
    ] {
        assert!(
            matmul_blas_fields.contains(&field),
            "MatmulBlasRequest should include {field}"
        );
    }

    let request_queries = report["request_capability_queries"]
        .as_array()
        .expect("request_capability_queries should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(request_queries.contains(&"backend_capabilities"));
    assert!(request_queries.contains(&"supports_attention_request"));
    assert!(request_queries.contains(&"supports_matmul_request"));
    assert!(request_queries.contains(&"supports_linear_request"));
    assert!(request_queries.contains(&"supports_replay_request"));

    let capability_descriptors = report["capability_descriptors"]
        .as_object()
        .expect("capability_descriptors should be an object");
    for name in [
        "BackendCapabilities",
        "StorageCapabilities",
        "ProjectionLoadPolicy",
        "GpuMemoryDetectionPolicy",
        "GpuMemoryBudgetPolicy",
        "GpuAllocatorMemoryProbePolicy",
        "GpuMemoryReclaimPolicy",
        "KvCacheAutoBlockPolicy",
        "KvCacheMemoryTierBlockCap",
        "KvCacheFp8Policy",
        "StartupCapabilities",
        "MatmulCapabilities",
        "AttentionCapabilities",
        "StreamingPrefillBackendPolicy",
        "GdnCapabilities",
        "InferenceRecurrentStatePolicy",
        "DecodeCapabilities",
        "SpeculativeDecodePolicy",
        "DecodeBatcherPolicy",
        "BackendTrainingCapabilities",
        "ServerTrainingDispatchPolicy",
        "TrainingAccelerationProfilePolicy",
        "ReplayCapabilities",
        "ReplayAuthority",
        "BackendFallbackCapabilities",
        "TrainingOptimizerSupport",
    ] {
        assert!(
            capability_descriptors.contains_key(name),
            "{name} should be present in capability_descriptors"
        );
    }
    let backend_capability_fields = capability_descriptors["BackendCapabilities"]["fields"]
        .as_array()
        .expect("BackendCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        backend_capability_fields.contains(&"startup"),
        "BackendCapabilities should expose startup/prewarm policy"
    );
    assert!(
        backend_capability_fields.contains(&"streaming_prefill"),
        "BackendCapabilities should expose streaming-prefill backend policy"
    );
    let startup_capability_fields = capability_descriptors["StartupCapabilities"]["fields"]
        .as_array()
        .expect("StartupCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for (field, message) in [
        (
            "run_inference_prewarm",
            "StartupCapabilities should own whether startup runs inference prewarm",
        ),
        (
            "require_inference_prewarm_for_health",
            "StartupCapabilities should own health readiness prewarm policy",
        ),
        (
            "precompile_custom_kernels",
            "StartupCapabilities should own startup custom-kernel precompile policy",
        ),
        (
            "native_training_default_enabled",
            "StartupCapabilities should own native-training default enablement",
        ),
        (
            "native_training_env",
            "StartupCapabilities should own native-training env override",
        ),
        (
            "decode_weight_prewarm_when_native_training",
            "StartupCapabilities should own native-training decode-weight prewarm routing",
        ),
    ] {
        assert!(startup_capability_fields.contains(&field), "{message}");
    }
    let replay_capability_fields = capability_descriptors["ReplayCapabilities"]["fields"]
        .as_array()
        .expect("ReplayCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        replay_capability_fields.contains(&"authority"),
        "ReplayCapabilities should expose typed replay authority"
    );
    let storage_capability_fields = capability_descriptors["StorageCapabilities"]["fields"]
        .as_array()
        .expect("StorageCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        storage_capability_fields.contains(&"kv_cache_device_memory_pressure"),
        "StorageCapabilities should own KV cache device-memory pressure policy"
    );
    assert!(
        storage_capability_fields.contains(&"projection_load_policy"),
        "StorageCapabilities should own backend-specific projection load policy"
    );
    assert!(
        storage_capability_fields.contains(&"gpu_memory_detection_policy"),
        "StorageCapabilities should own backend-specific GPU memory detection fallback policy"
    );
    assert!(
        storage_capability_fields.contains(&"gpu_memory_budget_policy"),
        "StorageCapabilities should own backend-specific GPU memory budget policy"
    );
    assert!(
        storage_capability_fields.contains(&"gpu_allocator_memory_probe_policy"),
        "StorageCapabilities should own backend-specific allocator memory probe policy"
    );
    assert!(
        storage_capability_fields.contains(&"gpu_memory_reclaim_policy"),
        "StorageCapabilities should own backend-specific GPU memory reclaim policy"
    );
    assert!(
        storage_capability_fields.contains(&"kv_sizing_residency_model_multiplier"),
        "StorageCapabilities should own backend-specific KV sizing residency reserve policy"
    );
    assert!(
        storage_capability_fields.contains(&"kv_auto_block_policy"),
        "StorageCapabilities should own backend-specific KV auto block cap policy"
    );
    assert!(
        storage_capability_fields.contains(&"kv_cache_fp8_policy"),
        "StorageCapabilities should own backend-specific KV FP8 cache policy"
    );
    let projection_load_policy_fields = capability_descriptors["ProjectionLoadPolicy"]["fields"]
        .as_array()
        .expect("ProjectionLoadPolicy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "direct_transposed_upload_for_cached_weights",
        "parallel_transposed_projection_upload",
        "parallel_transposed_projection_upload_disable_env",
        "parallel_auxiliary_weight_upload",
        "parallel_auxiliary_weight_upload_disable_env",
        "cache_full_attention_qkv_transpose_concat",
        "cache_linear_attention_ab_transpose_concat",
        "cache_mlp_gate_up_transpose_concat",
        "pack_w8a16_projection_rows",
        "stub_embedding_table_after_transposed_upload",
        "drop_projection_originals",
        "drop_projection_transposes",
        "synchronize_after_dropping_originals",
        "keep_projection_originals_env",
        "drop_projection_originals_env",
        "native_training_env",
        "keep_projection_transposes_env",
    ] {
        assert!(
            projection_load_policy_fields.contains(&field),
            "ProjectionLoadPolicy should include {field}"
        );
    }
    let gpu_memory_detection_policy_fields =
        capability_descriptors["GpuMemoryDetectionPolicy"]["fields"]
            .as_array()
            .expect("GpuMemoryDetectionPolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for field in [
        "detected_total_log_message",
        "missing_total_warning",
        "missing_total_fallback_bytes",
    ] {
        assert!(
            gpu_memory_detection_policy_fields.contains(&field),
            "GpuMemoryDetectionPolicy should include {field}"
        );
    }
    let gpu_memory_budget_policy_fields = capability_descriptors["GpuMemoryBudgetPolicy"]["fields"]
        .as_array()
        .expect("GpuMemoryBudgetPolicy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "use_live_memory_snapshot",
        "cap_kv_blocks_by_live_budget",
        "retry_kv_allocation_after_reclaim",
    ] {
        assert!(
            gpu_memory_budget_policy_fields.contains(&field),
            "GpuMemoryBudgetPolicy should include {field}"
        );
    }
    let gpu_allocator_memory_probe_policy_fields =
        capability_descriptors["GpuAllocatorMemoryProbePolicy"]["fields"]
            .as_array()
            .expect("GpuAllocatorMemoryProbePolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    assert!(
        gpu_allocator_memory_probe_policy_fields.contains(&"probe"),
        "GpuAllocatorMemoryProbePolicy should expose the selected allocator heap probe"
    );
    let gpu_memory_reclaim_policy_fields =
        capability_descriptors["GpuMemoryReclaimPolicy"]["fields"]
            .as_array()
            .expect("GpuMemoryReclaimPolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    assert!(
        gpu_memory_reclaim_policy_fields.contains(&"reclaimer"),
        "GpuMemoryReclaimPolicy should expose the selected reclaimer"
    );
    let kv_auto_block_policy_fields = capability_descriptors["KvCacheAutoBlockPolicy"]["fields"]
        .as_array()
        .expect("KvCacheAutoBlockPolicy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "context_window_cap",
        "static_max_blocks",
        "memory_tier_cap",
        "allow_min_blocks_below_live_budget",
    ] {
        assert!(
            kv_auto_block_policy_fields.contains(&field),
            "KvCacheAutoBlockPolicy should include {field}"
        );
    }
    let kv_cache_fp8_policy_fields = capability_descriptors["KvCacheFp8Policy"]["fields"]
        .as_array()
        .expect("KvCacheFp8Policy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "allow_when_requested_by_default",
        "explicit_enable_env",
        "disabled_reason",
    ] {
        assert!(
            kv_cache_fp8_policy_fields.contains(&field),
            "KvCacheFp8Policy should include {field}"
        );
    }
    let attention_capability_fields = capability_descriptors["AttentionCapabilities"]["fields"]
        .as_array()
        .expect("AttentionCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        attention_capability_fields.contains(&"flash_prefill_consumes_grouped_kv"),
        "AttentionCapabilities should own flash-prefill grouped-KV ABI routing"
    );
    assert!(
        attention_capability_fields.contains(&"detached_chunked_prefill"),
        "AttentionCapabilities should own detached chunked prefill routing"
    );
    let streaming_prefill_policy_fields =
        capability_descriptors["StreamingPrefillBackendPolicy"]["fields"]
            .as_array()
            .expect("StreamingPrefillBackendPolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for field in [
        "auto_dispatch",
        "base_tile_tokens",
        "tape_tile_tokens",
        "detached_full_attn_tile_tokens",
        "detached_full_attn_boundary_tile_tokens",
        "detached_full_attn_tape_replay_tile_tokens",
    ] {
        assert!(
            streaming_prefill_policy_fields.contains(&field),
            "StreamingPrefillBackendPolicy should include {field}"
        );
    }
    let decode_capability_fields = capability_descriptors["DecodeCapabilities"]["fields"]
        .as_array()
        .expect("DecodeCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        decode_capability_fields.contains(&"mtp_speculative_generation"),
        "DecodeCapabilities should expose native MTP speculative generation support"
    );
    assert!(
        decode_capability_fields.contains(&"speculative_policy"),
        "DecodeCapabilities should expose backend-owned speculative decode thresholds"
    );
    let speculative_decode_policy_fields =
        capability_descriptors["SpeculativeDecodePolicy"]["fields"]
            .as_array()
            .expect("SpeculativeDecodePolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for (field, message) in [
        (
            "mtp_max_prompt_tokens",
            "SpeculativeDecodePolicy should own the native MTP prompt threshold",
        ),
        (
            "long_prompt_skip_layer_min_prompt_tokens",
            "SpeculativeDecodePolicy should own the long-prompt skip-layer crossover",
        ),
        (
            "long_prompt_skip_layer_min_output_tokens",
            "SpeculativeDecodePolicy should own the skip-layer output threshold",
        ),
    ] {
        assert!(
            speculative_decode_policy_fields.contains(&field),
            "{message}"
        );
    }
    let gdn_capability_fields = capability_descriptors["GdnCapabilities"]["fields"]
        .as_array()
        .expect("GdnCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        gdn_capability_fields.contains(&"recurrent_step_f32"),
        "GdnCapabilities should own dtype-specific recurrent-step routing"
    );
    assert!(
        gdn_capability_fields.contains(&"inference_recurrent_state"),
        "GdnCapabilities should own inference recurrent-state dtype policy"
    );
    assert!(
        gdn_capability_fields.contains(&"chunk_pre_permute_bf16"),
        "GdnCapabilities should own GDN chunk pre-permute policy"
    );
    assert!(
        gdn_capability_fields.contains(&"gated_rms_norm_preserves_tape_residency"),
        "GdnCapabilities should own active-tape GDN RMSNorm residency policy"
    );
    let inference_recurrent_state_policy_fields =
        capability_descriptors["InferenceRecurrentStatePolicy"]["fields"]
            .as_array()
            .expect("InferenceRecurrentStatePolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    assert!(
        inference_recurrent_state_policy_fields.contains(&"bf16")
            && inference_recurrent_state_policy_fields.contains(&"f16"),
        "InferenceRecurrentStatePolicy should expose BF16 and F16 support"
    );
    let decode_batcher_policy_fields = capability_descriptors["DecodeBatcherPolicy"]["fields"]
        .as_array()
        .expect("DecodeBatcherPolicy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        decode_batcher_policy_fields.contains(&"require_native_decode_attention"),
        "DecodeBatcherPolicy should own native decode-attention fallback requirements"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"allow_portable_lora_decode"),
        "DecodeBatcherPolicy should own the correctness-qualified LoRA decode route"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"partition_noncontiguous_gdn_kv_tiles"),
        "DecodeBatcherPolicy should own GDN KV contiguity partition routing"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"prefer_direct_paged_decode_attention"),
        "DecodeBatcherPolicy should own direct paged-decode attention path preference"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"direct_paged_decode_attention_env_gate"),
        "DecodeBatcherPolicy should own direct paged-decode attention env gates"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"allow_prefix_cache_split_snapshot"),
        "DecodeBatcherPolicy should own prefix-cache split snapshot routing"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"paged_decode_requires_contiguous_kv_chunks"),
        "DecodeBatcherPolicy should own paged-decode KV chunk contiguity requirements"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"use_greedy_token_decode"),
        "DecodeBatcherPolicy should own greedy-token decode shortcut routing"
    );
    assert!(
        decode_batcher_policy_fields.contains(&"use_native_sampled_contiguous_decode"),
        "DecodeBatcherPolicy should own sampled contiguous decode routing"
    );
    assert!(
        decode_batcher_policy_fields
            .contains(&"sampled_contiguous_decode_requires_resident_decode"),
        "DecodeBatcherPolicy should own sampled contiguous resident-decode requirements"
    );
    for (field, message) in [
        (
            "rendezvous_default_enabled",
            "DecodeBatcherPolicy should own direct-rendezvous enable defaults",
        ),
        (
            "use_decode_width_prefill_admission",
            "DecodeBatcherPolicy should own prefill admission width defaults",
        ),
        (
            "burst_prefill_admission",
            "DecodeBatcherPolicy should own burst prefill admission defaults",
        ),
        (
            "batching_engine_default_enabled",
            "DecodeBatcherPolicy should own server batching-engine defaults",
        ),
        (
            "warm_resident_decode_pool_on_startup",
            "DecodeBatcherPolicy should own resident decode pool startup warmup",
        ),
    ] {
        assert!(decode_batcher_policy_fields.contains(&field), "{message}");
    }
    let backend_training_fields = capability_descriptors["BackendTrainingCapabilities"]["fields"]
        .as_array()
        .expect("BackendTrainingCapabilities fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        backend_training_fields.contains(&"server_dispatch"),
        "BackendTrainingCapabilities should expose server-side training dispatch policy"
    );
    assert!(
        backend_training_fields.contains(&"acceleration_profile"),
        "BackendTrainingCapabilities should expose startup training acceleration profile policy"
    );
    let server_training_dispatch_fields =
        capability_descriptors["ServerTrainingDispatchPolicy"]["fields"]
            .as_array()
            .expect("ServerTrainingDispatchPolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for field in [
        "native_route",
        "native_training_env",
        "native_training_default_enabled",
    ] {
        assert!(
            server_training_dispatch_fields.contains(&field),
            "ServerTrainingDispatchPolicy should include {field}"
        );
    }
    let training_acceleration_profile_fields =
        capability_descriptors["TrainingAccelerationProfilePolicy"]["fields"]
            .as_array()
            .expect("TrainingAccelerationProfilePolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for field in [
        "log_message",
        "linear",
        "sdpa",
        "rmsnorm_inference",
        "rmsnorm_training",
        "flce_provider",
        "resident_activation",
        "sgd_step_on_device",
    ] {
        assert!(
            training_acceleration_profile_fields.contains(&field),
            "TrainingAccelerationProfilePolicy should include {field}"
        );
    }
    let replay_authority_fields = capability_descriptors["ReplayAuthority"]["fields"]
        .as_array()
        .expect("ReplayAuthority fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "backend",
        "production_authority",
        "native_primitive",
        "graph_crate_role",
    ] {
        assert!(
            replay_authority_fields.contains(&field),
            "ReplayAuthority should include {field}"
        );
    }

    let resident_resource_descriptors = report["resident_resource_descriptors"]
        .as_object()
        .expect("resident_resource_descriptors should be an object");
    let resident_resource_fields = resident_resource_descriptors["ResidentResource"]["fields"]
        .as_array()
        .expect("ResidentResource fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in [
        "tensor_id",
        "backend",
        "device",
        "ownership",
        "state",
        "replay_stability",
    ] {
        assert!(
            resident_resource_fields.contains(&field),
            "ResidentResource should include {field}"
        );
    }

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let valid_statuses = ["covered", "partial", "gap", "fixture_required"];
    let gate_names = conformance_gates
        .iter()
        .filter_map(|gate| gate["gate"].as_str())
        .collect::<Vec<_>>();
    for name in [
        "storage_round_trip",
        "host_transfer_to_device_parity",
        "device_op_parity",
        "matmul_linear_parity",
        "attention_gdn_conv_parity",
        "optimizer_parity",
        "replay_parity",
        "one_step_training_proof",
        "no_unexpected_host_fallback",
        "decode_submit_or_replay_count",
        "matmul_algorithm_cache_reporting",
        "hardware_latency_thresholds",
        "generated_capability_dashboard",
    ] {
        assert!(
            gate_names.contains(&name),
            "conformance_gates should include {name}"
        );
    }
    for gate in conformance_gates {
        let status = gate["status"]
            .as_str()
            .expect("conformance gate status should be a string");
        assert!(
            valid_statuses.contains(&status),
            "invalid conformance gate status {status}"
        );
        let supplemental_commands = gate["supplemental_commands"]
            .as_array()
            .expect("supplemental_commands should be an array");
        let coverage_blockers = gate["coverage_blockers"]
            .as_array()
            .expect("coverage_blockers should be an array");
        for supplemental in supplemental_commands {
            assert!(
                !supplemental["scope"].as_str().unwrap_or("").is_empty(),
                "supplemental command scope should be non-empty"
            );
            assert!(
                !supplemental["command"].as_str().unwrap_or("").is_empty(),
                "supplemental command should be non-empty"
            );
        }
        if status == "covered" {
            assert!(
                !gate["evidence_present"]
                    .as_array()
                    .expect("evidence_present should be an array")
                    .is_empty(),
                "covered conformance gate should have evidence"
            );
            assert!(
                coverage_blockers.is_empty(),
                "covered conformance gate should not cite coverage blockers: {coverage_blockers:?}"
            );
        }
    }
    let storage_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "storage_round_trip")
        .expect("storage round-trip gate should be present");
    assert_eq!(storage_gate["status"], "covered");
    let storage_command = storage_gate["command"]
        .as_str()
        .expect("storage round-trip command should be a string");
    assert!(storage_command.contains("kiln-tensor --features rocm --test rocm_storage_smoke"));
    assert!(storage_command.contains("kiln-vulkan-kernel --test vk_tensor_parity"));
    assert_supplemental_command(
        storage_gate,
        "ROCm",
        "kiln-tensor --features rocm --test rocm_storage_smoke",
    );
    assert_supplemental_command(
        storage_gate,
        "Vulkan",
        "kiln-vulkan-kernel --test vk_tensor_parity",
    );

    let decode_submit_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "decode_submit_or_replay_count")
        .expect("decode submit/replay gate should be present");
    assert_eq!(decode_submit_gate["status"], "covered");
    let decode_submit_evidence = decode_submit_gate["evidence_present"]
        .as_array()
        .expect("decode submit/replay evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/src/generate.rs",
        "crates/kiln-server/src/metrics.rs",
        "crates/kiln-server/src/api/health.rs",
        "crates/kiln-server/src/api/debug_model_state.rs",
        "crates/kiln-graph/src/replay_plan.rs",
    ] {
        assert!(
            decode_submit_evidence.contains(&path),
            "decode submit/replay gate should cite {path}"
        );
    }

    let host_transfer_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "host_transfer_to_device_parity")
        .expect("host transfer gate should be present");
    assert_eq!(host_transfer_gate["status"], "covered");
    let host_transfer_evidence = host_transfer_gate["evidence_present"]
        .as_array()
        .expect("host transfer evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        host_transfer_evidence.contains(&"crates/kiln-tensor/src/tensor.rs"),
        "host transfer gate should cite Tensor::to_device support classification"
    );
    let host_transfer_command = host_transfer_gate["command"]
        .as_str()
        .expect("host transfer command should be a string");
    assert!(
        host_transfer_command.contains("device_transfer_support_classifies_explicit_transitions")
    );
    assert!(
        !host_transfer_command.contains("cuda_resize_copy_primitives"),
        "CUDA feature-gated resize coverage should live in supplemental commands"
    );
    assert_supplemental_command(
        host_transfer_gate,
        "CUDA",
        "kiln-tensor --no-default-features --features cuda --test cuda_resize_copy_primitives",
    );
    assert_supplemental_command(
        host_transfer_gate,
        "ROCm",
        "kiln-tensor --features rocm --test rocm_compare_parity",
    );
    assert_supplemental_command(
        host_transfer_gate,
        "Metal",
        "kiln-tensor --features metal --test metal_ops_parity",
    );
    assert_supplemental_command(
        host_transfer_gate,
        "Vulkan",
        "kiln-vulkan-kernel --test vk_tensor_parity",
    );

    let device_op_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "device_op_parity")
        .expect("DeviceOp parity gate should be present");
    assert_supplemental_command(
        device_op_gate,
        "ROCm",
        "kiln-tensor --features rocm --test rocm_scalar_op_parity",
    );
    assert_supplemental_command(
        device_op_gate,
        "Metal",
        "kiln-tensor --features metal --test metal_ops_parity",
    );

    let matmul_cache_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "matmul_algorithm_cache_reporting")
        .expect("matmul algorithm/cache reporting gate should be present");
    assert_eq!(matmul_cache_gate["status"], "covered");
    let matmul_cache_evidence = matmul_cache_gate["evidence_present"]
        .as_array()
        .expect("matmul cache evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-blas/src/algo_cache.rs",
        "crates/kiln-blas/src/cublaslt_handle.rs",
        "crates/kiln-rocblas/src/algo_cache.rs",
        "crates/kiln-rocblas/src/hipblaslt_handle.rs",
    ] {
        assert!(
            matmul_cache_evidence.contains(&path),
            "matmul cache reporting gate should cite {path}"
        );
    }

    let attention_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "attention_gdn_conv_parity")
        .expect("attention/GDN/conv parity gate should be present");
    assert_eq!(attention_gate["status"], "covered");
    let attention_command = attention_gate["command"]
        .as_str()
        .expect("attention/GDN/conv command should be a string");
    for command_fragment in [
        "kiln-model --no-default-features --features rocm --test rocm_flash_attn_bwd_gradcheck",
        "kiln-flash-attn --no-default-features --features rocm --test rocm_flash_attn_parity",
        "kiln-gdn-kernel --no-default-features --features rocm --test rocm_gdn_parity",
        "kiln-conv1d-kernel --no-default-features --features rocm --test rocm_conv1d_parity",
        "kiln-vulkan-kernel --test vk_attention_parity",
        "kiln-vulkan-kernel --test vk_sdpa_prefill_kernel_parity",
        "kiln-vulkan-kernel --test vk_gdn_foundation_parity",
        "kiln-vulkan-kernel --test vk_gdn_backward_parity",
        "kiln-vulkan-kernel --test gdn_parity",
    ] {
        assert!(
            attention_command.contains(command_fragment),
            "attention/GDN/conv parity gate command should run {command_fragment}"
        );
    }
    let attention_evidence = attention_gate["evidence_present"]
        .as_array()
        .expect("attention/GDN/conv evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/tests/rocm_flash_attn_bwd_gradcheck.rs",
        "crates/kiln-flash-attn/tests/rocm_flash_attn_parity.rs",
        "crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs",
        "crates/kiln-conv1d-kernel/tests/rocm_conv1d_parity.rs",
        "crates/kiln-vulkan-kernel/tests/vk_attention_parity.rs",
        "crates/kiln-vulkan-kernel/tests/vk_sdpa_prefill_kernel_parity.rs",
        "crates/kiln-vulkan-kernel/tests/vk_gdn_foundation_parity.rs",
        "crates/kiln-vulkan-kernel/tests/vk_gdn_backward_parity.rs",
        "crates/kiln-vulkan-kernel/tests/gdn_parity.rs",
    ] {
        assert!(
            attention_evidence.contains(&path),
            "attention/GDN/conv parity gate should cite {path}"
        );
    }
}

#[test]
fn gpu_memory_budget_policy_routes_live_budget_probes() {
    use kiln_model::GpuMemoryBudgetPolicy;
    use kiln_tensor::Device;

    assert_eq!(
        GpuMemoryBudgetPolicy::for_backend("cpu", Device::Cpu),
        GpuMemoryBudgetPolicy::HOST_MEMORY_ONLY
    );
    for (name, device) in [
        ("cuda", Device::Cuda(0)),
        ("rocm", Device::Rocm(0)),
        ("metal", Device::Metal(0)),
        ("vulkan", Device::Vulkan(0)),
    ] {
        assert_eq!(
            GpuMemoryBudgetPolicy::for_backend(name, device),
            GpuMemoryBudgetPolicy::DEVICE_MEMORY_AWARE,
            "{name} should use live device-memory budget policy"
        );
    }
}

#[test]
fn gpu_allocator_memory_probe_policy_routes_heap_probes() {
    use kiln_model::{GpuAllocatorMemoryProbe, GpuAllocatorMemoryProbePolicy};
    use kiln_tensor::Device;

    assert_eq!(
        GpuAllocatorMemoryProbePolicy::for_backend("cuda", Device::Cuda(0)).probe,
        GpuAllocatorMemoryProbe::CudaMemGetInfo
    );
    assert_eq!(
        GpuAllocatorMemoryProbePolicy::for_backend("rocm", Device::Rocm(0)).probe,
        GpuAllocatorMemoryProbe::RocmMemGetInfo {
            include_pool_spare: true
        }
    );
    for (name, device) in [
        ("cpu", Device::Cpu),
        ("metal", Device::Metal(0)),
        ("vulkan", Device::Vulkan(0)),
    ] {
        assert_eq!(
            GpuAllocatorMemoryProbePolicy::for_backend(name, device).probe,
            GpuAllocatorMemoryProbe::None,
            "{name} should not expose a backend allocator heap probe"
        );
    }
}

#[test]
fn gpu_memory_reclaim_policy_routes_backend_hooks() {
    use kiln_model::{GpuMemoryReclaimPolicy, GpuMemoryReclaimer};
    use kiln_tensor::Device;

    assert_eq!(
        GpuMemoryReclaimPolicy::for_backend("cuda", Device::Cuda(0)).reclaimer,
        GpuMemoryReclaimer::CudaTrimPool
    );
    assert_eq!(
        GpuMemoryReclaimPolicy::for_backend("rocm", Device::Rocm(0)).reclaimer,
        GpuMemoryReclaimer::RocmTrimPool
    );
    assert_eq!(
        GpuMemoryReclaimPolicy::for_backend("metal", Device::Metal(0)).reclaimer,
        GpuMemoryReclaimer::LoggedNoop {
            log_message: GpuMemoryReclaimPolicy::METAL_LOGGED_NOOP_MESSAGE,
        }
    );
    assert_eq!(
        GpuMemoryReclaimPolicy::for_backend("vulkan", Device::Vulkan(0)).reclaimer,
        GpuMemoryReclaimer::VulkanTrimPool
    );
    assert_eq!(
        GpuMemoryReclaimPolicy::for_backend("cpu", Device::Cpu).reclaimer,
        GpuMemoryReclaimer::None
    );
}

#[test]
fn training_acceleration_profile_policy_routes_vulkan_startup_log() {
    use kiln_model::{TrainingAccelerationProfileLogMessage, TrainingAccelerationProfilePolicy};
    use kiln_tensor::Device;

    let vulkan = TrainingAccelerationProfilePolicy::for_backend("vulkan", Device::Vulkan(0));
    assert_eq!(
        vulkan.log_message,
        TrainingAccelerationProfileLogMessage::Vulkan
    );
    assert_eq!(vulkan.linear, "on (qualified policy)");
    assert_eq!(vulkan.sdpa, "on (qualified policy)");
    assert_eq!(vulkan.rmsnorm_inference, "on (qualified policy)");

    for (name, device) in [
        ("cpu", Device::Cpu),
        ("cuda", Device::Cuda(0)),
        ("rocm", Device::Rocm(0)),
        ("metal", Device::Metal(0)),
    ] {
        assert_eq!(
            TrainingAccelerationProfilePolicy::for_backend(name, device).log_message,
            TrainingAccelerationProfileLogMessage::None,
            "{name} should not log the Vulkan training acceleration profile"
        );
    }
}

#[test]
fn generated_capability_report_lists_streaming_prefill_backend_policy() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let policies = report["streaming_prefill_backend_policy"]
        .as_object()
        .expect("streaming_prefill_backend_policy should be an object");
    for backend in ["cpu", "cuda", "rocm", "metal", "vulkan"] {
        assert!(
            policies.contains_key(backend),
            "{backend} should be present in streaming_prefill_backend_policy"
        );
    }

    assert_eq!(policies["cpu"]["auto_dispatch"]["kind"], "never");
    assert_eq!(
        policies["cpu"]["auto_dispatch"]["minimum_prompt_tokens"],
        Value::Null
    );
    assert_eq!(
        policies["cuda"]["auto_dispatch"]["kind"],
        "prompt_tokens_at_least"
    );
    assert_eq!(
        policies["cuda"]["auto_dispatch"]["minimum_prompt_tokens"],
        2_048
    );
    assert_eq!(policies["cuda"]["base_tile_tokens"], 1_024);
    assert_eq!(
        policies["cuda"]["detached_full_attn_boundary_tile_tokens"],
        65_536
    );
    assert_eq!(policies["rocm"]["base_tile_tokens"], 1_024);
    assert_eq!(policies["metal"]["base_tile_tokens"], 2_048);
    assert_eq!(policies["vulkan"]["auto_dispatch"]["kind"], "never");
    assert_eq!(policies["vulkan"]["base_tile_tokens"], 2_048);
}

#[test]
fn generated_capability_report_lists_training_precision_policy() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let policies = report["training_precision_policy"]
        .as_object()
        .expect("training_precision_policy should be an object");
    for backend in ["cpu", "cuda", "rocm", "metal", "vulkan"] {
        assert!(
            policies.contains_key(backend),
            "{backend} should be present in training_precision_policy"
        );
    }

    let vulkan = &policies["vulkan"];
    assert_eq!(vulkan["name"], "vulkan_mixed_f32_bf16");
    assert_eq!(vulkan["loss_accumulation_dtype"], "F32");
    assert_eq!(vulkan["mixed_rms_norm_weight_dtype"], "BF16");
    assert_eq!(vulkan["mixed_precision"], true);
    assert_eq!(
        policies["cuda"]["mixed_rms_norm_weight_dtype"],
        Value::Null,
        "CUDA should keep equal-dtype RMSNorm training policy"
    );
    for policy in policies.values() {
        let policy = policy
            .as_object()
            .expect("training precision policy entry should be an object");
        for field in [
            "streaming_prefill_tile_tokens",
            "tape_streaming_tile_tokens",
            "detached_full_attn_tile_tokens",
            "detached_full_attn_boundary_tile_tokens",
            "detached_full_attn_tape_replay_tile_tokens",
            "paged_prefill_medium_tile_tokens",
            "paged_prefill_medium_tile_max_tokens",
        ] {
            assert!(
                !policy.contains_key(field),
                "execution field {field} must not appear in training_precision_policy"
            );
        }
    }

    let activation_dtypes = vulkan["activation_dtypes"]
        .as_array()
        .expect("vulkan activation_dtypes should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert_eq!(activation_dtypes, ["F32"]);

    let base_weight_dtypes = vulkan["base_weight_dtypes"]
        .as_array()
        .expect("vulkan base_weight_dtypes should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(base_weight_dtypes.contains(&"BF16"));
}

#[test]
fn generated_capability_report_lists_training_loss_policy() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let policies = report["training_loss_policy"]
        .as_object()
        .expect("training_loss_policy should be an object");
    for backend in ["cpu", "cuda", "rocm", "metal", "vulkan"] {
        assert!(
            policies.contains_key(backend),
            "{backend} should be present in training_loss_policy"
        );
    }

    assert_eq!(
        policies["cpu"]["sft_flce_loss_route"], "full_logits",
        "CPU should keep the portable SFT full-logits route"
    );
    assert_eq!(
        policies["cpu"]["tape_forward_backward_route"], "unsupported",
        "CPU should not advertise kt tape-authoritative forward/backward"
    );
    assert_eq!(
        policies["cpu"]["grpo_loss_route"], "kt_composite",
        "CPU should keep the shared GRPO kt-composite route"
    );
    assert_eq!(
        policies["cpu"]["grpo_kl_auxiliary_route"], "host_composite",
        "CPU should keep GRPO KL auxiliaries on the host-composite route"
    );
    assert_eq!(
        policies["cpu"]["opd_loss_route"], "unsupported",
        "CPU should not advertise a portable OPD training route"
    );
    assert_eq!(
        policies["cpu"]["opd_phase_b_backward_route"], "unsupported",
        "CPU should not advertise OPD Phase-B backward"
    );
    assert_eq!(
        policies["cpu"]["final_rmsnorm_backward_route"], "kt_composite",
        "CPU should use the kt-composite final RMSNorm backward route"
    );
    assert_eq!(
        policies["metal"]["sft_flce_loss_route"], "full_logits",
        "Metal should keep the portable SFT full-logits route"
    );
    assert_eq!(
        policies["metal"]["tape_forward_backward_route"], "kt_tape_authoritative",
        "Metal should advertise kt tape-authoritative forward/backward"
    );
    assert_eq!(
        policies["metal"]["grpo_loss_route"], "kt_composite",
        "Metal should keep the shared GRPO kt-composite route"
    );
    assert_eq!(
        policies["metal"]["grpo_kl_auxiliary_route"], "host_composite",
        "Metal should keep GRPO KL auxiliaries on the host-composite route"
    );
    assert_eq!(
        policies["metal"]["opd_loss_route"], "kt_tape_phase_b",
        "Metal should use the shared OPD kt-tape Phase-B route"
    );
    assert_eq!(
        policies["metal"]["opd_phase_b_backward_route"], "kt_composite",
        "Metal should use the device-agnostic kt composite OPD Phase-B backward"
    );
    assert_eq!(
        policies["metal"]["final_rmsnorm_backward_route"], "kt_composite",
        "Metal should use the kt-composite final RMSNorm backward route"
    );
    assert_eq!(
        policies["cuda"]["sft_flce_loss_route"], "kt_tape_flce",
        "CUDA should use the kt-tape SFT FLCE route"
    );
    assert_eq!(
        policies["cuda"]["tape_forward_backward_route"], "kt_tape_authoritative",
        "CUDA should advertise kt tape-authoritative forward/backward"
    );
    assert_eq!(
        policies["cuda"]["grpo_loss_route"], "kt_composite",
        "CUDA should use the shared GRPO kt-composite route"
    );
    assert_eq!(
        policies["cuda"]["grpo_kl_auxiliary_route"], "cuda_rocm_device_fast_path",
        "CUDA should advertise device fast paths for GRPO KL auxiliaries"
    );
    assert_eq!(
        policies["cuda"]["opd_loss_route"], "kt_tape_phase_b",
        "CUDA should use the shared OPD kt-tape Phase-B route"
    );
    assert_eq!(
        policies["cuda"]["opd_phase_b_backward_route"], "cuda_rocm_fused_unit_grad",
        "CUDA should advertise the fused OPD Phase-B unit-gradient leaf"
    );
    assert_eq!(
        policies["cuda"]["final_rmsnorm_backward_route"], "cuda_rocm_fused_tail",
        "CUDA should advertise the fused final RMSNorm tail route"
    );
    assert_eq!(
        policies["rocm"]["sft_flce_loss_route"], "kt_tape_flce",
        "ROCm should use the shared kt-tape SFT FLCE route"
    );
    assert_eq!(
        policies["rocm"]["tape_forward_backward_route"], "kt_tape_authoritative",
        "ROCm should advertise kt tape-authoritative forward/backward"
    );
    assert_eq!(
        policies["rocm"]["grpo_loss_route"], "kt_composite",
        "ROCm should use the shared GRPO kt-composite route"
    );
    assert_eq!(
        policies["rocm"]["grpo_kl_auxiliary_route"], "cuda_rocm_device_fast_path",
        "ROCm should advertise device fast paths for GRPO KL auxiliaries"
    );
    assert_eq!(
        policies["rocm"]["opd_loss_route"], "kt_tape_phase_b",
        "ROCm should use the shared OPD kt-tape Phase-B route"
    );
    assert_eq!(
        policies["rocm"]["opd_phase_b_backward_route"], "cuda_rocm_fused_unit_grad",
        "ROCm should advertise the fused OPD Phase-B unit-gradient leaf"
    );
    assert_eq!(
        policies["rocm"]["final_rmsnorm_backward_route"], "cuda_rocm_fused_tail",
        "ROCm should advertise the fused final RMSNorm tail route"
    );
    assert_eq!(
        policies["vulkan"]["sft_flce_loss_route"], "vulkan_active_rows",
        "Vulkan should use its active-row SFT FLCE route"
    );
    assert_eq!(
        policies["vulkan"]["tape_forward_backward_route"], "kt_tape_authoritative",
        "Vulkan should advertise kt tape-authoritative forward/backward"
    );
    assert_eq!(
        policies["vulkan"]["grpo_loss_route"], "vulkan_active_rows",
        "Vulkan should use its active-row GRPO route"
    );
    assert_eq!(
        policies["vulkan"]["grpo_kl_auxiliary_route"], "host_composite",
        "Vulkan should keep GRPO KL auxiliaries on the host-composite route"
    );
    assert_eq!(
        policies["vulkan"]["opd_loss_route"], "vulkan_active_hidden",
        "Vulkan should use its active-hidden OPD route"
    );
    assert_eq!(
        policies["vulkan"]["opd_phase_b_backward_route"], "vulkan_active_hidden",
        "Vulkan should advertise active-hidden OPD loss/backward routing"
    );
    assert_eq!(
        policies["vulkan"]["final_rmsnorm_backward_route"], "kt_composite",
        "Vulkan should use the kt-composite final RMSNorm backward route"
    );
}

#[test]
fn generated_capability_report_lists_optimizer_dispatch_policy() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let dispatch = report["optimizer_dispatch"]
        .as_object()
        .expect("optimizer_dispatch should be an object");
    for (backend, sgd, adamw, muon) in [
        ("cuda", "overridden", "overridden", "overridden"),
        ("rocm", "overridden", "overridden", "overridden"),
        ("metal", "default_decline", "overridden", "overridden"),
        ("vulkan", "overridden", "overridden", "overridden"),
    ] {
        let info = dispatch
            .get(backend)
            .unwrap_or_else(|| panic!("{backend} optimizer dispatch should be present"));
        assert_eq!(info["sgd_step"], sgd, "{backend} SGD dispatch drifted");
        assert_eq!(
            info["adamw_step"], adamw,
            "{backend} AdamW dispatch drifted"
        );
        assert_eq!(info["muon_step"], muon, "{backend} Muon dispatch drifted");
    }

    let fallback = report["training_optimizer_fallback_policy"]
        .as_object()
        .expect("training_optimizer_fallback_policy should be an object");
    assert_eq!(fallback["cpu"]["default_policy"], "CorrectnessAllowed");
    assert_eq!(
        fallback["cpu"]["optimizer_parameter_dtypes"]["adam_w"],
        serde_json::json!(["F32"])
    );
    assert_eq!(
        fallback["cpu"]["rounding_modes"],
        serde_json::json!(["round_to_nearest"])
    );
    assert_eq!(fallback["cpu"]["muon_min_lora_rank"], 2);
    assert!(fallback["cpu"]["muon_max_lora_rank"].is_null());
    for backend in ["cuda", "rocm", "metal", "vulkan"] {
        assert_eq!(
            fallback[backend]["default_policy"], "NativeRequired",
            "{backend} optimizer fallback policy should require native dispatch"
        );
        assert!(
            fallback[backend].get("debug_opt_in").is_none(),
            "{backend} optimizer fallback policy must not advertise a mutable override"
        );
        assert_eq!(
            fallback[backend]["rounding_modes"],
            serde_json::json!(["round_to_nearest"]),
            "{backend} product optimizer rounding must be immutable"
        );
        assert_eq!(fallback[backend]["muon_min_lora_rank"], 2);
    }
    assert_eq!(fallback["cuda"]["muon_max_lora_rank"], 48);
    assert_eq!(fallback["rocm"]["muon_max_lora_rank"], 48);
    assert_eq!(fallback["metal"]["muon_max_lora_rank"], 32);
    assert_eq!(fallback["vulkan"]["muon_max_lora_rank"], 32);
    assert_eq!(
        fallback["metal"]["optimizer_parameter_dtypes"]["sgd"],
        serde_json::json!([])
    );
    assert_eq!(
        fallback["metal"]["optimizer_parameter_dtypes"]["adam_w"],
        serde_json::json!(["F32", "BF16"])
    );
    assert_eq!(
        fallback["metal"]["optimizer_parameter_dtypes"]["muon"],
        serde_json::json!(["F32", "BF16"])
    );
    assert_eq!(
        fallback["metal"]["product_executable_base_to_lora"],
        serde_json::json!({"BF16": "BF16"})
    );
    assert_eq!(
        fallback["metal"]["product_executable_optimizer_kinds"],
        serde_json::json!(["adam_w", "muon"])
    );
    assert_eq!(
        fallback["vulkan"]["product_executable_base_to_lora"],
        serde_json::json!({"F32": "F32", "BF16": "F32"})
    );

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let optimizer_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "optimizer_parity")
        .expect("optimizer parity gate should be present");
    assert_eq!(optimizer_gate["status"], "covered");
    let evidence = optimizer_gate["evidence_present"]
        .as_array()
        .expect("optimizer parity evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-optim/tests/integration.rs",
        "crates/kiln-model/src/backend/cuda.rs",
        "crates/kiln-model/src/backend/rocm.rs",
        "crates/kiln-model/src/backend/metal_training.rs",
        "crates/kiln-model/src/backend/vulkan.rs",
        "crates/kiln-model/src/backend/vulkan_training.rs",
        "crates/kiln-train/src/trainer.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "optimizer parity gate should cite {path}"
        );
    }
    assert_supplemental_command(
        optimizer_gate,
        "CUDA plus Vulkan",
        "kiln-train --features cuda,vulkan --test vk_cuda_opd_parity",
    );
}

#[test]
fn generated_capability_report_gates_replay_contract() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let replay_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "replay_parity")
        .expect("replay parity gate should be present");
    assert_eq!(replay_gate["status"], "covered");
    let command = replay_gate["command"]
        .as_str()
        .expect("replay parity command should be a string");
    for command_fragment in [
        "kiln-graph replay",
        "kiln-graph --test capture_lifetime",
        "kiln-graph-cuda replay",
        "kiln-graph-metal replay",
        "kiln-graph-vulkan replay",
        "kiln-model --features vulkan --test vk_resident_decode_parity",
        "kiln-tensor --features rocm --test rocm_capture_arena",
        "kiln-model --test backend_capability_contract",
    ] {
        assert!(
            command.contains(command_fragment),
            "replay parity gate command should run {command_fragment}"
        );
    }
    let evidence = replay_gate["evidence_present"]
        .as_array()
        .expect("replay parity evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-graph/src/replay_plan.rs",
        "crates/kiln-graph/src/captured_graph.rs",
        "crates/kiln-graph/tests/capture_lifetime.rs",
        "crates/kiln-graph-cuda/src/lib.rs",
        "crates/kiln-graph-metal/src/lib.rs",
        "crates/kiln-graph-vulkan/src/lib.rs",
        "crates/kiln-model/src/backend/capability.rs",
        "crates/kiln-model/src/backend/residency.rs",
        "crates/kiln-model/tests/vk_resident_decode_parity.rs",
        "crates/kiln-tensor/tests/rocm_capture_arena.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "replay parity gate should cite {path}"
        );
    }

    let request_queries = report["request_capability_queries"]
        .as_array()
        .expect("request_capability_queries should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(request_queries.contains(&"supports_replay_request"));
    assert!(request_queries.contains(&"replay_key_for_request"));
}

#[test]
fn generated_capability_report_lists_replay_authority() {
    let root = workspace_root();
    let report_path = root.join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");
    let replay_authority = report["replay_authority"]
        .as_object()
        .expect("replay_authority should be an object");

    for (backend, primitive, runner_path) in [
        ("cuda", "CUDA graph", "crates/kiln-model/src/cuda_graph.rs"),
        ("rocm", "HIP graph", "crates/kiln-model/src/rocm_graph.rs"),
        ("metal", "Metal ICB", "crates/kiln-model/src/metal_graph.rs"),
        (
            "vulkan",
            "Vulkan CommandBatch",
            "crates/kiln-model/src/vk_decode_resident.rs",
        ),
    ] {
        let info = replay_authority
            .get(backend)
            .unwrap_or_else(|| panic!("replay_authority should list {backend}"));
        assert_eq!(
            info["native_primitive"].as_str(),
            Some(primitive),
            "{backend} should report its native replay primitive"
        );
        let runners = info["runner_paths"]
            .as_array()
            .expect("runner_paths should be an array")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert!(
            runners.contains(&runner_path),
            "{backend} should cite production runner path {runner_path}"
        );
        let missing = info["evidence_missing"]
            .as_array()
            .expect("evidence_missing should be an array");
        assert!(
            missing.is_empty(),
            "{backend} replay authority should not cite missing paths: {missing:?}"
        );
    }

    assert!(
        replay_authority["rocm"]["graph_crate_paths"]
            .as_array()
            .expect("rocm graph_crate_paths should be an array")
            .is_empty(),
        "ROCm should honestly report that no kiln-graph-rocm crate exists yet"
    );

    let metal_tests = replay_authority["metal"]["parity_tests"]
        .as_array()
        .expect("metal parity_tests should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        metal_tests.contains(&"test_metal_graph_batched_decode_matches_eager_and_replays_bucket"),
        "Metal replay authority should cite the batched eager-vs-replay graph test"
    );
}

#[test]
fn generated_capability_report_separates_rocm_legacy_cuda_env_aliases() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let rocm = &report["backends"]["rocm"];
    let native_env_gates = rocm["native_env_gates"]
        .as_array()
        .expect("rocm native_env_gates should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let legacy_env_aliases = rocm["legacy_env_aliases"]
        .as_array()
        .expect("rocm legacy_env_aliases should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();

    assert!(
        native_env_gates
            .iter()
            .all(|gate| !gate.starts_with("KILN_DISABLE_CUDA_")),
        "ROCm native env gates should not include legacy CUDA aliases"
    );
    for alias in [
        "KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT",
        "KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM",
        "KILN_DISABLE_CUDA_GDN_PREFILL_GATES",
        "KILN_DISABLE_CUDA_LORA_DECODE_ADD",
    ] {
        assert!(
            legacy_env_aliases.contains(&alias),
            "ROCm should list {alias} as a legacy compatibility alias"
        );
    }
}

#[test]
fn generated_capability_report_gates_matmul_linear_contract() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let matmul_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "matmul_linear_parity")
        .expect("matmul/linear parity gate should be present");
    assert_eq!(matmul_gate["status"], "covered");
    let command = matmul_gate["command"]
        .as_str()
        .expect("matmul/linear parity command should be a string");
    for command_fragment in [
        "kiln-tensor --features rocm --test rocm_matmul_parity",
        "kiln-vulkan-kernel --test vk_matmul_parity",
        "kiln-vulkan-kernel --test linear_decode_argmax",
        "kiln-vulkan-kernel --test linear_decode_sample",
        "kiln-blas --features cublaslt --tests",
    ] {
        assert!(
            command.contains(command_fragment),
            "matmul/linear parity command should run {command_fragment}"
        );
    }
    assert_supplemental_command(
        matmul_gate,
        "CUDA cublasLt",
        "kiln-blas --features cublaslt --test cublaslt_handle_smoke",
    );
    assert_supplemental_command(
        matmul_gate,
        "Metal",
        "kiln-tensor --features metal --test metal_ops_parity",
    );
    assert!(
        matmul_gate["evidence_missing"]
            .as_array()
            .expect("matmul/linear missing evidence should be an array")
            .is_empty(),
        "covered matmul/linear parity gate should not have missing evidence"
    );

    let evidence = matmul_gate["evidence_present"]
        .as_array()
        .expect("matmul/linear evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/src/backend/capability.rs",
        "crates/kiln-model/src/backend/mod.rs",
        "crates/kiln-blas/tests/cublaslt_handle_smoke.rs",
        "crates/kiln-tensor/tests/rocm_matmul_parity.rs",
        "crates/kiln-tensor/tests/metal_ops_parity.rs",
        "crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs",
        "crates/kiln-vulkan-kernel/tests/linear_decode_argmax.rs",
        "crates/kiln-vulkan-kernel/tests/linear_decode_sample.rs",
        "crates/kiln-model/tests/tape_forward_parity.rs",
        "crates/kiln-model/tests/marlin_qproj_parity.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "matmul/linear parity gate should cite {path}"
        );
    }

    let request_queries = report["request_capability_queries"]
        .as_array()
        .expect("request_capability_queries should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(request_queries.contains(&"supports_matmul_request"));
    assert!(request_queries.contains(&"supports_linear_request"));
}

#[test]
fn generated_capability_report_tracks_one_step_training_proof_gate() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let training_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "one_step_training_proof")
        .expect("one-step training proof gate should be present");
    let command = training_gate["command"]
        .as_str()
        .expect("one-step training proof command should be a string");
    assert!(command.contains("kiln-optim --test end_to_end_training"));
    assert!(command.contains("kiln-model --test backend_capability_contract"));
    assert!(
        !command.contains("cuda_sft_step_proof"),
        "backend-specific SFT proof commands should live in supplemental commands"
    );
    assert_supplemental_command(
        training_gate,
        "CUDA",
        "kiln-model --features cuda --test cuda_sft_step_proof",
    );
    assert_supplemental_command(
        training_gate,
        "ROCm",
        "kiln-model --features rocm --test rocm_sft_step_proof",
    );
    assert_supplemental_command(
        training_gate,
        "Metal",
        "kiln-model --features metal --test metal_sft_step_proof",
    );
    assert_supplemental_command(training_gate, "Vulkan", "KILN_TENSOR_VULKAN_TEST=1");
    assert_supplemental_command(
        training_gate,
        "Vulkan",
        "kiln-model --features vulkan --test vk_sft_step_proof",
    );

    let evidence = training_gate["evidence"]
        .as_array()
        .expect("one-step training proof evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/tests/cuda_sft_step_proof.rs",
        "crates/kiln-model/tests/metal_sft_step_proof.rs",
        "crates/kiln-model/tests/vk_sft_step_proof.rs",
        "crates/kiln-model/tests/rocm_sft_step_proof.rs",
        "crates/kiln-optim/tests/end_to_end_training.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "one-step training proof gate should cite {path}"
        );
    }

    let evidence_present = training_gate["evidence_present"]
        .as_array()
        .expect("one-step training proof present evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/tests/cuda_sft_step_proof.rs",
        "crates/kiln-model/tests/metal_sft_step_proof.rs",
        "crates/kiln-model/tests/vk_sft_step_proof.rs",
        "crates/kiln-model/tests/rocm_sft_step_proof.rs",
        "crates/kiln-optim/tests/end_to_end_training.rs",
    ] {
        assert!(
            evidence_present.contains(&path),
            "one-step training proof gate should find existing {path}"
        );
    }

    let evidence_missing = training_gate["evidence_missing"]
        .as_array()
        .expect("one-step training proof missing evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let backend_proofs = [
        "crates/kiln-model/tests/cuda_sft_step_proof.rs",
        "crates/kiln-model/tests/metal_sft_step_proof.rs",
        "crates/kiln-model/tests/vk_sft_step_proof.rs",
        "crates/kiln-model/tests/rocm_sft_step_proof.rs",
    ];
    match training_gate["status"].as_str() {
        Some("covered") => {
            assert!(
                evidence_missing.is_empty(),
                "covered one-step training proof gate must not have missing evidence"
            );
        }
        Some("partial") => {
            assert!(
                !evidence_missing.is_empty(),
                "partial one-step training proof gate should name missing evidence"
            );
            for path in backend_proofs {
                if !evidence_present.contains(&path) {
                    assert!(
                        evidence_missing.contains(&path),
                        "partial one-step training proof gate should name missing {path}"
                    );
                }
            }
        }
        other => panic!("unexpected one-step training proof status {other:?}"),
    }
}

#[test]
fn replay_plan_cpu_mock_parity_gate_runs_in_unification_contract() -> GraphContractResult {
    use kiln_graph::{
        CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
        ReplayResourceStability, ReplayState, ResidentResourceRef,
    };
    use kiln_tensor::{Backend, DType, Tensor};

    fn cpu_mock_decode_eager(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; 4];
        for row in 0..2 {
            for col in 0..2 {
                let mut acc = 0.0_f32;
                for k in 0..3 {
                    acc += lhs[row * 3 + k] * rhs[k * 2 + col];
                }
                out[row * 2 + col] = acc;
            }
        }
        out
    }

    struct MockCpuDecodeReplayPlan {
        key: ReplayKey,
        state: ReplayState,
        replay_count: u64,
        lhs: Vec<f32>,
        rhs: Vec<f32>,
        last_output: Option<Vec<f32>>,
    }

    impl std::fmt::Debug for MockCpuDecodeReplayPlan {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCpuDecodeReplayPlan")
                .field("key", &self.key)
                .field("replay_count", &self.replay_count)
                .finish_non_exhaustive()
        }
    }

    impl ReplayPlan for MockCpuDecodeReplayPlan {
        fn backend(&self) -> Backend {
            self.key.backend
        }

        fn key(&self) -> ReplayKey {
            self.key.clone()
        }

        fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
            self.state.validate(inputs.key, inputs.resources)
        }

        fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
            self.state.validate(inputs.key, inputs.resources)?;
            self.replay_count += 1;
            self.last_output = Some(cpu_mock_decode_eager(&self.lhs, &self.rhs));
            Ok(ReplayOutputs::new(
                inputs.resources.to_vec(),
                self.replay_count,
            ))
        }

        fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
            self.state.invalidate_reason(&state.key, &state.inputs)
        }
    }

    let lhs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let eager = cpu_mock_decode_eager(&lhs, &rhs);
    let lhs_tensor = Tensor::from_slice(&lhs, vec![2, 3])?;
    let rhs_tensor = Tensor::from_slice(&rhs, vec![3, 2])?;
    let key = ReplayKey::new(
        Backend::Cpu,
        "mock_decode_matmul",
        vec![2, 3, 2],
        Some(DType::F32),
        1,
        true,
    );
    let resources = vec![
        ResidentResourceRef::from_tensor(
            &lhs_tensor,
            Backend::Cpu,
            ReplayResourceStability::StableAcrossReplay,
        ),
        ResidentResourceRef::from_tensor(
            &rhs_tensor,
            Backend::Cpu,
            ReplayResourceStability::StableAcrossReplay,
        ),
    ];
    let mut plan = MockCpuDecodeReplayPlan {
        key: key.clone(),
        state: ReplayState::new(key.clone(), resources.clone()),
        replay_count: 0,
        lhs,
        rhs,
        last_output: None,
    };

    plan.validate_inputs(ReplayInputs::new(&key, &resources))?;
    let outputs = kiln_graph::ReplayPlan::replay(&mut plan, ReplayInputs::new(&key, &resources))?;
    assert_eq!(outputs.replay_count, 1);
    assert_eq!(outputs.resources, resources);
    let replayed = plan
        .last_output
        .as_ref()
        .expect("mock replay should record its CPU output");
    assert_eq!(
        replayed, &eager,
        "CPU/mock ReplayPlan parity gate should compare replayed output to eager output"
    );
    Ok(())
}

#[test]
fn generated_capability_report_check_mode_is_non_mutating_and_enforced() {
    let root = workspace_root();
    let script_path = root.join("scripts/generate_backend_capability_report.py");

    let self_test = Command::new("python3")
        .arg(&script_path)
        .arg("--self-test")
        .current_dir(&root)
        .output()
        .expect("capability report generator self-test should run");
    assert!(
        self_test.status.success(),
        "capability report generator self-test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&self_test.stdout),
        String::from_utf8_lossy(&self_test.stderr)
    );

    let markdown_path = root.join("docs/backend-capability-report.md");
    let json_path = root.join("docs/backend-capability-report.json");
    let markdown_before =
        fs::read_to_string(&markdown_path).expect("capability report markdown should be readable");
    let json_before =
        fs::read_to_string(&json_path).expect("capability report json should be readable");

    let check = Command::new("python3")
        .arg(&script_path)
        .arg("--check")
        .current_dir(&root)
        .output()
        .expect("capability report generator check should run");
    assert!(
        check.status.success(),
        "capability report generator --check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert_eq!(
        fs::read_to_string(&markdown_path)
            .expect("capability report markdown should remain readable"),
        markdown_before,
        "capability report --check should not rewrite Markdown"
    );
    assert_eq!(
        fs::read_to_string(&json_path).expect("capability report json should remain readable"),
        json_before,
        "capability report --check should not rewrite JSON"
    );

    let report: Value =
        serde_json::from_str(&json_before).expect("capability report json should parse");
    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let dashboard_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "generated_capability_dashboard")
        .expect("generated capability dashboard gate should be present");
    let command = dashboard_gate["command"]
        .as_str()
        .expect("generated dashboard command should be a string");
    assert!(command.contains("--self-test"));
    assert!(command.contains("--check"));
    let evidence_present = dashboard_gate["evidence_present"]
        .as_array()
        .expect("generated dashboard evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        evidence_present.contains(&"scripts/generate_backend_capability_report.py"),
        "generated dashboard gate should cite the generator itself"
    );
    assert!(
        evidence_present.contains(&"scripts/check_unification_gates.sh"),
        "generated dashboard gate should cite the local unification gate"
    );
}

/// Every backend must keep the prefix-cache split snapshot enabled.
///
/// The prefill-split snapshot is the ONLY producer of block-aligned
/// strict-prefix cache entries, and `RealPrefixCache` can only serve a
/// longer next-turn prompt from a block-aligned entry. A backend arm that
/// sets `allow_prefix_cache_split_snapshot: false` therefore silently
/// disables multi-turn prefix caching wholesale: every agent turn
/// re-prefills its entire conversation history from scratch (40s+ per turn
/// at 16K tokens on Strix Halo). Commit 002af558 did exactly that to ROCm
/// while optimizing long-context prefill, and nothing caught it.
///
/// If a backend ever genuinely cannot capture a mid-prefill
/// `LinearAttentionState` snapshot, gate it behind an env override with a
/// startup warning — never a silent policy `false`.
#[test]
fn every_backend_allows_prefix_cache_split_snapshot() {
    use kiln_model::backend::capability::DecodeBatcherPolicy;

    let arms = [
        ("cuda", kiln_tensor::Device::Cuda(0)),
        ("rocm", kiln_tensor::Device::Rocm(0)),
        ("metal", kiln_tensor::Device::Metal(0)),
        ("vulkan", kiln_tensor::Device::Vulkan(0)),
        ("cpu", kiln_tensor::Device::Cpu),
    ];
    for (name, device) in arms {
        let policy = DecodeBatcherPolicy::for_backend(name, device);
        assert!(
            policy.allow_prefix_cache_split_snapshot,
            "backend `{name}` disables the prefix-cache split snapshot, which kills \
             multi-turn prefix caching (every turn fully re-prefills its history); \
             see the field comment on allow_prefix_cache_split_snapshot"
        );
    }
}

#[test]
fn portable_lora_decode_is_an_explicit_vulkan_only_capability() {
    use kiln_model::backend::capability::DecodeBatcherPolicy;

    for (name, device, expected) in [
        ("cuda", kiln_tensor::Device::Cuda(0), false),
        ("rocm", kiln_tensor::Device::Rocm(0), false),
        ("metal", kiln_tensor::Device::Metal(0), false),
        ("vulkan", kiln_tensor::Device::Vulkan(0), true),
        ("cpu", kiln_tensor::Device::Cpu, false),
    ] {
        assert_eq!(
            DecodeBatcherPolicy::for_backend(name, device).allow_portable_lora_decode,
            expected,
            "backend `{name}` portable LoRA decode policy drifted"
        );
    }
}

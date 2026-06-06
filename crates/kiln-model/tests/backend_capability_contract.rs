use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

#[derive(Debug)]
struct FunctionDef {
    body: String,
    line: usize,
}

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

fn source_between<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("source should contain marker {start_marker}"));
    let rest = &source[start..];
    let end = rest.find(end_marker).unwrap_or_else(|| {
        panic!("source should contain marker {end_marker} after {start_marker}")
    });
    &rest[..end]
}

fn find_matching_brace(source: &str, open_idx: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut i = open_idx;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut in_string = false;
    let mut in_char = false;
    let mut escape = false;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        let next = bytes.get(i + 1).copied().map(char::from).unwrap_or('\0');
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
        } else if in_block_comment {
            if ch == '*' && next == '/' {
                in_block_comment = false;
                i += 1;
            }
        } else if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if in_char {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '\'' {
                in_char = false;
            }
        } else if ch == '/' && next == '/' {
            in_line_comment = true;
            i += 1;
        } else if ch == '/' && next == '*' {
            in_block_comment = true;
            i += 1;
        } else if ch == '"' {
            in_string = true;
        } else if ch == '\'' {
            in_char = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn parse_functions(path: &Path) -> HashMap<String, FunctionDef> {
    let source = fs::read_to_string(path).expect("backend source should be readable");
    let mut out = HashMap::new();
    let mut offset = 0usize;
    while let Some(relative_fn) = source[offset..].find("fn ") {
        let fn_start = offset + relative_fn;
        let name_start = fn_start + 3;
        let Some(first_non_name) =
            source[name_start..].find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        else {
            break;
        };
        let name_end = name_start + first_non_name;
        let name = &source[name_start..name_end];
        if name.is_empty() {
            offset = name_end;
            continue;
        }
        let Some(open_relative) = source[name_end..].find('{') else {
            break;
        };
        let open = name_end + open_relative;
        let Some(close) = find_matching_brace(&source, open) else {
            break;
        };
        let line = source[..fn_start].bytes().filter(|b| *b == b'\n').count() + 1;
        out.insert(
            name.to_string(),
            FunctionDef {
                body: source[open + 1..close].to_string(),
                line,
            },
        );
        offset = close + 1;
    }
    out
}

fn body_without_comments(body: &str) -> String {
    let mut out = String::new();
    let mut chars = body.chars().peekable();
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    while let Some(ch) = chars.next() {
        let next = chars.peek().copied().unwrap_or('\0');
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
                out.push(ch);
            }
        } else if in_block_comment {
            if ch == '*' && next == '/' {
                in_block_comment = false;
                chars.next();
            }
        } else if ch == '/' && next == '/' {
            in_line_comment = true;
            chars.next();
        } else if ch == '/' && next == '*' {
            in_block_comment = true;
            chars.next();
        } else {
            out.push(ch);
        }
    }
    out
}

fn compact_body(body: &str) -> String {
    body_without_comments(body)
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect()
}

fn support_is_literal_true(body: &str) -> bool {
    compact_body(body) == "true"
}

fn body_always_declines(body: &str) -> bool {
    matches!(compact_body(body).as_str(), "Ok(None)" | "returnOk(None);")
}

fn paired_method_name(support_method: &str) -> Option<&'static str> {
    match support_method {
        "supports_flash_attn_prefill" => Some("flash_attn_prefill"),
        "supports_flash_attn_paged_decode" => Some("flash_attn_paged_decode"),
        "supports_linear_decode_argmax" => Some("linear_decode_argmax"),
        "supports_linear_decode_argmax_batch" => Some("linear_decode_argmax_batch"),
        "supports_gdn_forward_substitution" => Some("gdn_forward_substitution"),
        "supports_gdn_recurrent_step" => Some("gdn_recurrent_step"),
        "supports_gdn_chunk_prep" => Some("gdn_chunk_prep"),
        "supports_gdn_chunk_scan" => Some("gdn_chunk_scan"),
        "supports_gdn_full_chunk_forward" => Some("gdn_full_chunk_forward"),
        "supports_gdn_gates" => Some("gdn_gates"),
        "supports_gdn_gated_rms_norm" => Some("gdn_gated_rms_norm"),
        _ => None,
    }
}

#[test]
fn literal_true_support_predicates_do_not_pair_with_always_declining_methods() {
    let backend_dir = manifest_dir().join("src/backend");
    let backends = ["cuda.rs", "rocm.rs", "metal.rs", "vulkan.rs"];
    let mut failures = Vec::new();

    for backend in backends {
        let path = backend_dir.join(backend);
        let functions = parse_functions(&path);
        for (name, support_fn) in functions
            .iter()
            .filter(|(name, _)| name.starts_with("supports_"))
        {
            if !support_is_literal_true(&support_fn.body) {
                continue;
            }
            let Some(pair) = paired_method_name(name) else {
                continue;
            };
            let Some(paired_fn) = functions.get(pair) else {
                continue;
            };
            if body_always_declines(&paired_fn.body) {
                failures.push(format!(
                    "{}:{} `{}` returns true but `{}` at line {} always returns Ok(None)",
                    backend, support_fn.line, name, pair, paired_fn.line
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "backend support predicate mismatches:\n{}",
        failures.join("\n")
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
fn cuda_rocm_resident_membership_stays_in_shared_helper() {
    let backend_dir = manifest_dir().join("src/backend");
    let common_path = backend_dir.join("cuda_rocm_common.rs");
    let common_source =
        fs::read_to_string(&common_path).expect("cuda_rocm_common.rs should be readable");

    for required in [
        "ResidentTensorIdRegistry",
        "kt_tensor_id",
        "mark_resident_activation",
        "evict_resident_activation",
        "has_resident_activation",
    ] {
        assert!(
            common_source.contains(required),
            "cuda_rocm_common.rs should own shared resident membership helper `{required}`"
        );
    }

    for backend_file in ["cuda.rs", "rocm.rs"] {
        let path = backend_dir.join(backend_file);
        let source = fs::read_to_string(&path).expect("backend source should be readable");
        let functions = parse_functions(&path);
        let register = functions
            .get("register_resident_activation")
            .unwrap_or_else(|| panic!("{backend_file} missing register_resident_activation"));
        let update = functions
            .get("update_resident_activation")
            .unwrap_or_else(|| panic!("{backend_file} missing update_resident_activation"));
        let evict = functions
            .get("evict_resident_activation")
            .unwrap_or_else(|| panic!("{backend_file} missing evict_resident_activation"));
        let has = functions
            .get("has_resident_activation")
            .unwrap_or_else(|| panic!("{backend_file} missing has_resident_activation"));

        assert!(
            compact_body(&register.body).contains("cuda_rocm_common::mark_resident_activation("),
            "{backend_file} should register residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&update.body).contains("cuda_rocm_common::mark_resident_activation("),
            "{backend_file} should update residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&evict.body).contains("cuda_rocm_common::evict_resident_activation("),
            "{backend_file} should evict residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&has.body).contains("cuda_rocm_common::has_resident_activation("),
            "{backend_file} should query residency through cuda_rocm_common"
        );

        assert!(
            !source.contains("with_cuda_resident_ids")
                && !source.contains("with_rocm_resident_ids")
                && !source.contains("fn kt_id("),
            "{backend_file} should not keep copied resident TensorId registry helpers"
        );
    }
}

#[test]
fn cuda_rocm_optimizer_arg_validation_stays_in_shared_helper() {
    let backend_dir = manifest_dir().join("src/backend");
    let common_path = backend_dir.join("cuda_rocm_common.rs");
    let common_source =
        fs::read_to_string(&common_path).expect("cuda_rocm_common.rs should be readable");
    for required in [
        "optimizer_tensors_supported_for_kt",
        "optimizer_args_ready_for_kt",
        "supports_optimizer_step_kt",
    ] {
        assert!(
            common_source.contains(required),
            "cuda_rocm_common.rs should own shared optimizer argument validation `{required}`"
        );
    }

    for (backend_file, helper) in [
        ("cuda.rs", "cuda_optimizer_args_ready_for_kt"),
        ("rocm.rs", "rocm_optimizer_args_ready_for_kt"),
    ] {
        let path = backend_dir.join(backend_file);
        let source = fs::read_to_string(&path).expect("backend source should be readable");
        let functions = parse_functions(&path);
        let sgd = functions
            .get("dispatch_sgd_step")
            .unwrap_or_else(|| panic!("{backend_file} missing dispatch_sgd_step"));
        let adamw = functions
            .get("dispatch_adamw_step")
            .unwrap_or_else(|| panic!("{backend_file} missing dispatch_adamw_step"));

        assert!(
            compact_body(&sgd.body).contains(&format!("{helper}(&[param,grad])")),
            "{backend_file} SGD dispatch should use shared optimizer readiness validation"
        );
        assert!(
            compact_body(&adamw.body).contains(&format!(
                "{helper}(&[param,grad,first_moment,second_moment])"
            )),
            "{backend_file} AdamW dispatch should use shared optimizer readiness validation"
        );
        assert!(
            !source.contains("supports_optimizer_step_kt(&[")
                && !source.contains("optimizer_tensors_supported_for_kt(&["),
            "{backend_file} should not keep copied optimizer argument validation"
        );
    }
}

#[test]
fn cuda_rocm_kt_bridge_device_checks_stay_in_shared_helper() {
    let backend_dir = manifest_dir().join("src/backend");
    let common_path = backend_dir.join("cuda_rocm_common.rs");
    let common_source =
        fs::read_to_string(&common_path).expect("cuda_rocm_common.rs should be readable");
    assert!(
        common_source.contains("tensors_on_backend_device"),
        "cuda_rocm_common.rs should own shared kt tensor device checks"
    );

    for (backend_file, helper, variant) in [
        ("cuda.rs", "cuda_tensors_on_device", "Cuda"),
        ("rocm.rs", "rocm_tensors_on_device", "Rocm"),
    ] {
        let path = backend_dir.join(backend_file);
        let source = fs::read_to_string(&path).expect("backend source should be readable");
        let compact_source = compact_body(&source);
        let functions = parse_functions(&path);
        let linear = functions
            .get("linear_prefill_apply")
            .unwrap_or_else(|| panic!("{backend_file} missing linear_prefill_apply"));
        let offset = functions
            .get("linear_prefill_apply_offset")
            .unwrap_or_else(|| panic!("{backend_file} missing linear_prefill_apply_offset"));
        let lora = functions
            .get("lora_delta_resident")
            .unwrap_or_else(|| panic!("{backend_file} missing lora_delta_resident"));

        assert!(
            compact_body(&linear.body).contains(&format!("{helper}(&[x,weight_t])")),
            "{backend_file} linear_prefill_apply should use shared kt device validation"
        );
        assert!(
            compact_body(&offset.body).contains(&format!("{helper}(&[x,full_weight_t])")),
            "{backend_file} linear_prefill_apply_offset should use shared kt device validation"
        );
        assert!(
            compact_body(&lora.body).contains(&format!("{helper}(&[x,a,b])")),
            "{backend_file} lora_delta_resident should use shared kt device validation"
        );

        for copied_check in [
            format!("!matches!(x.device(),kiln_tensor::Device::{variant}(_))"),
            format!("!matches!(weight_t.device(),kiln_tensor::Device::{variant}(_))"),
            format!("!matches!(full_weight_t.device(),kiln_tensor::Device::{variant}(_))"),
            format!("!matches!(a.device(),kiln_tensor::Device::{variant}(_))"),
            format!("!matches!(b.device(),kiln_tensor::Device::{variant}(_))"),
        ] {
            assert!(
                !compact_source.contains(&copied_check),
                "{backend_file} should not keep copied kt device check `{copied_check}`"
            );
        }
    }
}

#[test]
fn cuda_rocm_blaslt_request_conversion_stays_shared() {
    let tensor_dir = workspace_root().join("crates/kiln-tensor/src");
    let helper_path = tensor_dir.join("blaslt_request.rs");
    let helper_source =
        fs::read_to_string(&helper_path).expect("blaslt_request.rs should be readable");
    for required in [
        "BlasLtMatmulRequest",
        "BlasLtMatmulLayout",
        "BlasLtEpilogue",
        "blaslt_dtype_name",
    ] {
        assert!(
            helper_source.contains(required),
            "blaslt_request.rs should own shared CUDA/ROCm BLASLt request conversion `{required}`"
        );
    }

    for (backend_file, concrete_mapper) in [
        ("cuda_matmul.rs", "cuda_blaslt_request"),
        ("rocm_matmul.rs", "rocm_blaslt_request"),
    ] {
        let path = tensor_dir.join(backend_file);
        let source = fs::read_to_string(&path).expect("matmul source should be readable");
        let compact_source = compact_body(&source);

        assert!(
            compact_source.contains(&format!(
                "letrequest={concrete_mapper}(BlasLtMatmulRequest::new("
            )),
            "{backend_file} should build BLASLt requests through the shared descriptor"
        );
        assert!(
            compact_source.contains("blaslt_dtype_name(dtype,"),
            "{backend_file} should use the shared BLASLt dtype envelope"
        );
        assert!(
            !source.contains("let request = MatmulRequest {"),
            "{backend_file} should not keep copied request construction at dispatch sites"
        );
        assert!(
            !source.contains("fn dtype_str(")
                && !source.contains("DType::F32 => \"f32\"")
                && !source.contains("DType::BF16 => \"bf16\"")
                && !source.contains("DType::F16 => \"f16\""),
            "{backend_file} should not keep copied CUDA/ROCm BLASLt dtype conversion"
        );
    }
}

#[test]
fn cuda_rocm_forward_in_projection_gates_use_rocm_names() {
    let forward_path = manifest_dir().join("src/forward.rs");
    let source = fs::read_to_string(&forward_path).expect("forward.rs should be readable");
    for required in [
        "cuda_rocm_disable_env_set_for_device",
        "cuda_rocm_gdn_ab_in_proj_enabled",
        "cuda_rocm_gdn_prefill_ab_in_proj_enabled",
        "CUDA_ROCM_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
        "cuda_rocm_full_attn_qkv_in_proj_enabled",
        "KILN_DISABLE_ROCM_GDN_AB_IN_PROJ",
        "KILN_DISABLE_ROCM_GDN_PREFILL_AB_IN_PROJ",
        "KILN_DISABLE_ROCM_FULL_ATTN_QKV_IN_PROJ",
    ] {
        assert!(
            source.contains(required),
            "forward.rs should use ROCm-native names for shared CUDA/ROCm in-projection gate `{required}`"
        );
    }
    for stale_helper in [
        "fn cuda_gdn_ab_in_proj_enabled(",
        "fn cuda_gdn_prefill_ab_in_proj_enabled(",
        "fn cuda_full_attn_qkv_in_proj_enabled(",
        "CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
    ] {
        assert!(
            !source.contains(stale_helper),
            "forward.rs should not keep stale CUDA-only shared gate `{stale_helper}`"
        );
    }
}

#[test]
fn graph_crates_disclose_scaffold_authority_boundary() {
    let root = workspace_root();
    let required_by_file: &[(&str, &[&str])] = &[
        (
            "crates/kiln-graph/src/lib.rs",
            &[
                "shared replay vocabulary",
                "not the current",
                "production replay authority",
                "model-level runners",
            ],
        ),
        (
            "crates/kiln-graph-cuda/src/lib.rs",
            &[
                "scaffold",
                "production CUDA decode graph",
                "not yet the authoritative replay layer",
            ],
        ),
        (
            "crates/kiln-graph-metal/src/lib.rs",
            &[
                "scaffold plus a reusable ICB replay object",
                "production Metal replay orchestration",
                "not yet",
                "authoritative replay layer",
            ],
        ),
        (
            "crates/kiln-graph/Cargo.toml",
            &[
                "Backend-agnostic replay vocabulary",
                "Production decode replay still lives",
                "model-level",
            ],
        ),
        (
            "crates/kiln-graph-vulkan/src/lib.rs",
            &[
                "scaffold",
                "production Vulkan replay path",
                "not yet",
                "authoritative replay layer",
            ],
        ),
        (
            "crates/kiln-graph-cuda/Cargo.toml",
            &[
                "CUDA CapturedGraph scaffold",
                "It does not yet wrap",
                "production CUDA decode",
                "graph runner still lives",
            ],
        ),
        (
            "crates/kiln-graph-metal/Cargo.toml",
            &[
                "Metal CapturedGraph scaffold",
                "production Metal replay orchestration still lives",
                "moves or wraps it",
            ],
        ),
        (
            "crates/kiln-graph-vulkan/Cargo.toml",
            &[
                "Vulkan CapturedGraph scaffold",
                "Production Vulkan replay",
                "still lives",
                "command batching remains",
            ],
        ),
    ];

    for (relative_path, required_phrases) in required_by_file {
        let source = fs::read_to_string(root.join(relative_path))
            .unwrap_or_else(|err| panic!("{relative_path} should be readable: {err}"));
        for required in *required_phrases {
            assert!(
                source.contains(required),
                "{relative_path} should disclose the current graph authority boundary with `{required}`"
            );
        }
    }

    for relative_path in [
        "crates/kiln-graph-cuda/Cargo.toml",
        "crates/kiln-graph-metal/Cargo.toml",
        "crates/kiln-graph-vulkan/Cargo.toml",
    ] {
        let source = fs::read_to_string(root.join(relative_path))
            .unwrap_or_else(|err| panic!("{relative_path} should be readable: {err}"));
        for stale in [
            "wraps cudarc CudaGraph + CudaGraphExec for the Phase 5 capture/replay surface",
            "for the production capture/replay path",
            "extends kiln-vulkan-kernel cmd_batch.rs for the Phase 5 capture/replay surface",
        ] {
            assert!(
                !source.contains(stale),
                "{relative_path} should not imply kiln-graph-* is already production replay authority: `{stale}`"
            );
        }
    }
}

#[test]
fn cuda_rocm_support_predicates_stay_in_shared_helper() {
    let backend_dir = manifest_dir().join("src/backend");
    let common_path = backend_dir.join("cuda_rocm_common.rs");
    let common_source =
        fs::read_to_string(&common_path).expect("cuda_rocm_common.rs should be readable");
    assert!(
        common_source.contains("CudaRocmSupportPredicates"),
        "cuda_rocm_common.rs should own the shared CUDA/ROCm support predicate table"
    );

    let shared_support_methods = [
        "supports_flash_attn_prefill",
        "supports_flash_attn_paged_decode",
        "supports_strict_paged_decode_contiguous_batch",
        "supports_gdn_forward_substitution",
        "supports_gdn_recurrent_step",
        "supports_gdn_chunk_prep",
        "supports_gdn_chunk_scan",
        "supports_gdn_full_chunk_forward",
        "supports_gdn_decode_gates_recurrent_unexpanded_qk",
        "supports_gdn_decode_qk_norm_gates_recurrent",
        "supports_gdn_gates",
        "supports_gdn_gated_rms_norm",
        "supports_causal_conv1d_update",
        "supports_causal_conv1d_prefill",
    ];

    for backend_file in ["cuda.rs", "rocm.rs"] {
        let path = backend_dir.join(backend_file);
        let functions = parse_functions(&path);
        let support_factory = functions
            .get("support_predicates")
            .unwrap_or_else(|| panic!("{backend_file} missing support_predicates helper"));
        assert!(
            support_factory.body.contains("CudaRocmSupportPredicates"),
            "{backend_file} should construct shared CUDA/ROCm support predicates"
        );

        for method in shared_support_methods {
            let body = functions
                .get(method)
                .unwrap_or_else(|| panic!("{backend_file} missing {method}"));
            assert!(
                compact_body(&body.body).contains(&format!("support_predicates().{method}()")),
                "{backend_file} `{method}` should delegate to cuda_rocm_common"
            );
        }
    }
}

#[test]
fn vulkan_gdn_runtime_methods_stay_in_gdn_module() {
    let backend_dir = manifest_dir().join("src/backend");
    let vulkan_rs = backend_dir.join("vulkan.rs");
    let functions = parse_functions(&vulkan_rs);
    let delegated_methods = [
        "supports_gdn_forward_substitution",
        "supports_gdn_recurrent_step",
        "supports_gdn_recurrent_prefill_native_head_last",
        "supports_gdn_recurrent_qk_norm_prefill_native_head_last",
        "supports_gdn_chunk_prep",
        "supports_gdn_chunk_scan",
        "supports_gdn_full_chunk_forward",
        "supports_gdn_gates",
        "supports_gdn_gated_rms_norm",
        "gdn_in_proj_decode",
        "gdn_decode_gates_recurrent_rmsnorm",
        "gdn_forward_substitution",
        "gdn_recurrent_prefill_native_head_last",
        "gdn_recurrent_qk_norm_prefill_native_head_last",
        "gdn_recurrent_step",
        "gdn_chunkwise_forward",
        "gdn_chunk_prep",
        "gdn_chunk_scan",
        "gdn_full_chunk_forward",
        "gdn_gates",
        "gdn_gated_rms_norm",
    ];
    let mut failures = Vec::new();

    for method in delegated_methods {
        let Some(function) = functions.get(method) else {
            failures.push(format!("vulkan.rs is missing `{method}`"));
            continue;
        };
        let body = compact_body(&function.body);
        let delegation = format!("vulkan_gdn::{method}(");
        if !body.starts_with(&delegation) {
            failures.push(format!(
                "vulkan.rs:{} `{method}` should delegate to `{delegation}`",
                function.line
            ));
        }
        if body.contains("kiln_vulkan_kernel::") || body.contains("std::env::var(") {
            failures.push(format!(
                "vulkan.rs:{} `{method}` should not retain kernel/env dispatch logic",
                function.line
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Vulkan GDN facade regression:\n{}",
        failures.join("\n")
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
        "MatmulCapabilities",
        "AttentionCapabilities",
        "GdnCapabilities",
        "DecodeCapabilities",
        "BackendTrainingCapabilities",
        "ReplayCapabilities",
        "BackendFallbackCapabilities",
    ] {
        assert!(
            capability_descriptors.contains_key(name),
            "{name} should be present in capability_descriptors"
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
        if status == "covered" {
            assert!(
                !gate["evidence_present"]
                    .as_array()
                    .expect("evidence_present should be an array")
                    .is_empty(),
                "covered conformance gate should have evidence"
            );
        }
    }
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
    assert_eq!(vulkan["mixed_precision"], true);

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
fn generated_capability_report_lists_optimizer_dispatch_policy() {
    let report_path = workspace_root().join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let dispatch = report["optimizer_dispatch"]
        .as_object()
        .expect("optimizer_dispatch should be an object");
    for (backend, sgd, adamw) in [
        ("cuda", "overridden", "overridden"),
        ("rocm", "overridden", "overridden"),
        ("metal", "default_decline", "overridden"),
        ("vulkan", "overridden", "overridden"),
    ] {
        let info = dispatch
            .get(backend)
            .unwrap_or_else(|| panic!("{backend} optimizer dispatch should be present"));
        assert_eq!(info["sgd_step"], sgd, "{backend} SGD dispatch drifted");
        assert_eq!(
            info["adamw_step"], adamw,
            "{backend} AdamW dispatch drifted"
        );
    }

    let fallback = report["training_optimizer_fallback_policy"]
        .as_object()
        .expect("training_optimizer_fallback_policy should be an object");
    assert_eq!(fallback["cpu"]["default_policy"], "CorrectnessAllowed");
    for backend in ["cuda", "rocm", "metal", "vulkan"] {
        assert_eq!(
            fallback[backend]["default_policy"], "NativeRequired",
            "{backend} optimizer fallback policy should require native dispatch"
        );
    }

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
        "crates/kiln-model/src/backend/metal.rs",
        "crates/kiln-model/src/backend/vulkan.rs",
        "crates/kiln-model/src/backend/vulkan_training.rs",
        "crates/kiln-train/src/trainer.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "optimizer parity gate should cite {path}"
        );
    }
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
fn capability_queries_consume_focused_backend_facets() {
    let root = workspace_root();
    let capability_source =
        fs::read_to_string(root.join("crates/kiln-model/src/backend/capability.rs"))
            .expect("capability.rs should be readable");

    for required in [
        "pub trait BackendCapabilityQueries:",
        "AttentionBackend",
        "SamplingBackend",
        "ReplayBackend",
        "TrainingLossBackend",
        "AttentionBackend::runtime_supports_flash_attn_prefill",
        "SamplingBackend::runtime_supports_linear_decode_argmax",
        "ReplayBackend::runtime_supports_replay_request",
        "ReplayBackend::runtime_replay_key_for_request",
    ] {
        assert!(
            capability_source.contains(required),
            "BackendCapabilityQueries should consume focused facet surface {required}"
        );
    }

    let backend_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend/mod.rs should be readable");
    let replay_impl_start = backend_source
        .find("impl<T: BackendRuntime + ?Sized> ReplayBackend for T")
        .expect("ReplayBackend blanket impl should be present");
    let replay_impl = &backend_source[replay_impl_start..];
    let replay_support_start = replay_impl
        .find("fn runtime_supports_replay_request")
        .expect("ReplayBackend runtime_supports_replay_request should be implemented");
    let replay_key_start = replay_impl
        .find("fn runtime_replay_key_for_request")
        .expect("ReplayBackend runtime_replay_key_for_request should follow support mapping");
    let replay_support = &replay_impl[replay_support_start..replay_key_start];
    assert!(
        !replay_support.contains("BackendCapabilityQueries::supports_replay_request"),
        "ReplayBackend must own replay request support mapping instead of recursing through BackendCapabilityQueries"
    );
    assert!(
        replay_support.contains("AttentionBackend::runtime_supports_flash_attn_paged_decode"),
        "paged-decode replay support should be derived from the focused attention facet"
    );
}

#[test]
fn resident_registry_consumes_focused_residency_facet() {
    let root = workspace_root();
    let backend_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend/mod.rs should be readable");
    let registry_impl_start = backend_source
        .find("impl<T> residency::ResidentRegistry for T")
        .expect("ResidentRegistry blanket impl should be present");
    let registry_impl = &backend_source[registry_impl_start..];
    let registry_impl_end = registry_impl
        .find("impl<T: BackendRuntime + ?Sized> OptimizerBackend for T")
        .expect("OptimizerBackend blanket impl should follow ResidentRegistry adapter");
    let registry_impl = &registry_impl[..registry_impl_end];

    for required in [
        "T: BackendRuntime + ResidencyBackend + ?Sized",
        "ResidencyBackend::runtime_register_resident_activation",
        "ResidencyBackend::runtime_update_resident_activation",
        "ResidencyBackend::runtime_evict_resident_activation",
        "ResidencyBackend::runtime_resident_activation_resource",
    ] {
        assert!(
            registry_impl.contains(required),
            "ResidentRegistry adapter should consume focused residency facet surface {required}"
        );
    }

    for forbidden in [
        "BackendRuntime::register_resident_activation",
        "BackendRuntime::update_resident_activation",
        "BackendRuntime::evict_resident_activation",
        "BackendRuntime::resident_activation_resource",
    ] {
        assert!(
            !registry_impl.contains(forbidden),
            "ResidentRegistry adapter should not reach around ResidencyBackend via {forbidden}"
        );
    }
}

#[test]
fn trainer_optimizer_updates_consume_optimizer_backend_facet() {
    let root = workspace_root();
    let trainer_source = fs::read_to_string(root.join("crates/kiln-train/src/trainer.rs"))
        .expect("trainer.rs should be readable");

    assert!(
        trainer_source.contains("OptimizerBackend"),
        "trainer should import the focused optimizer facet"
    );

    for (fn_name, fn_body, focused_call, forbidden_call) in [
        (
            "apply_sgd_update_kt",
            source_between(
                &trainer_source,
                "fn apply_sgd_update_kt",
                "fn apply_adamw_update_kt",
            ),
            "OptimizerBackend::runtime_dispatch_sgd_step",
            "backend.dispatch_sgd_step",
        ),
        (
            "apply_adamw_update_kt",
            source_between(
                &trainer_source,
                "fn apply_adamw_update_kt",
                "/// (#1082) Accumulate kt gradients",
            ),
            "OptimizerBackend::runtime_dispatch_adamw_step",
            "backend.dispatch_adamw_step",
        ),
    ] {
        assert!(
            fn_body.contains(focused_call),
            "{fn_name} should route optimizer dispatch through {focused_call}"
        );
        assert!(
            !fn_body.contains(forbidden_call),
            "{fn_name} should not call broad BackendRuntime optimizer method {forbidden_call}"
        );
    }
}

#[test]
fn lora_residency_call_sites_consume_residency_backend_facet() {
    let root = workspace_root();
    let trainer_source = fs::read_to_string(root.join("crates/kiln-train/src/trainer.rs"))
        .expect("trainer.rs should be readable");
    let lora_source = fs::read_to_string(root.join("crates/kiln-model/src/lora_loader.rs"))
        .expect("lora_loader.rs should be readable");

    assert!(
        trainer_source.contains("ResidencyBackend"),
        "trainer should import the focused residency facet"
    );
    assert!(
        lora_source.contains("ResidencyBackend"),
        "LoRA loader should import the focused residency facet"
    );

    let trainer_lora_section = source_between(
        &trainer_source,
        "impl TrainableLoraParams",
        "/// (#1082) Allocate AdamW optimizer state.",
    );
    let trainer_optimizer_section = source_between(
        &trainer_source,
        "impl OptimizerState",
        "/// (#1082) Build `Option<OptimizerState>`",
    );
    let trainer_optimizer_updates = source_between(
        &trainer_source,
        "fn sgd_step(",
        "/// Gradient checkpointing configuration.",
    );
    let lora_loader_section = source_between(
        &lora_source,
        "impl LoraWeights {\n    /// Phase 4.1",
        "impl LoraWeights {\n    /// Load",
    );

    let trainer_residency_sections =
        format!("{trainer_lora_section}\n{trainer_optimizer_section}\n{trainer_optimizer_updates}");
    for required in [
        "ResidencyBackend::runtime_supports_resident_activation",
        "ResidencyBackend::runtime_register_resident_activation",
        "ResidencyBackend::runtime_evict_resident_activation",
        "ResidencyBackend::runtime_has_resident_activation",
        "ResidencyBackend::runtime_resolve_resident_activation",
        "ResidencyBackend::runtime_update_resident_activation",
    ] {
        assert!(
            trainer_residency_sections.contains(required),
            "trainer LoRA residency paths should consume focused residency facet method {required}"
        );
    }

    for required in [
        "ResidencyBackend::runtime_supports_resident_activation",
        "ResidencyBackend::runtime_register_resident_activation",
        "ResidencyBackend::runtime_evict_resident_activation",
    ] {
        assert!(
            lora_loader_section.contains(required),
            "LoRA loader residency paths should consume focused residency facet method {required}"
        );
    }

    let combined = format!("{trainer_residency_sections}\n{lora_loader_section}");
    for forbidden in [
        "backend.supports_resident_activation",
        "backend.register_resident_activation",
        "backend.evict_resident_activation",
        "backend.update_resident_activation",
        "backend.has_resident_activation",
        "backend.resolve_resident_activation",
    ] {
        assert!(
            !combined.contains(forbidden),
            "LoRA residency paths should not call broad BackendRuntime method {forbidden}"
        );
    }
}

#[test]
fn runtime_policy_call_sites_consume_focused_capability_surfaces() {
    let root = workspace_root();
    let trainer_source = fs::read_to_string(root.join("crates/kiln-train/src/trainer.rs"))
        .expect("trainer.rs should be readable");
    let generate_source = fs::read_to_string(root.join("crates/kiln-model/src/generate.rs"))
        .expect("generate.rs should be readable");

    assert!(
        trainer_source.contains("BackendCapabilityQueries"),
        "trainer should import the request-shaped backend capability query surface"
    );
    assert!(
        generate_source.contains("TrainingLossBackend"),
        "ModelRunner should import the focused training loss/capability facet"
    );
    assert!(
        generate_source.contains("BackendCapabilityQueries"),
        "decode fallback policy should import the shared backend capability query surface"
    );

    let decode_policy_section = source_between(
        &generate_source,
        "fn decode_hot_path_fallback_policy(",
        "fn decode_hot_path_debug_fallback_enabled(",
    );
    assert!(
        decode_policy_section.contains("BackendCapabilityQueries::backend_capabilities"),
        "decode fallback policy should come from the shared backend capability aggregate"
    );
    assert!(
        !generate_source.contains("fn decode_hot_path_fallback_policy_for"),
        "generate should not keep a duplicate backend-name/device decode fallback policy table"
    );
    assert!(
        !decode_policy_section.contains("match device"),
        "decode fallback policy should not branch directly on device kind"
    );

    let trainer_policy_section = source_between(
        &trainer_source,
        "fn training_optimizer_fallback_policy(",
        "fn ensure_training_optimizer_fallback_allowed(",
    );
    assert!(
        trainer_policy_section.contains("BackendCapabilityQueries::backend_capabilities"),
        "trainer optimizer fallback policy should come from the shared backend capability aggregate"
    );
    assert!(
        !trainer_policy_section.contains("training_optimizer_fallback_policy_for"),
        "trainer should not keep a duplicate backend-name fallback policy table"
    );
    assert!(
        !trainer_policy_section.contains("match device"),
        "trainer optimizer fallback policy should not branch directly on device kind"
    );

    let runner_new_section = source_between(
        &generate_source,
        "pub fn new_with_options(",
        "// Phase A.5: registry + decode-buffer config",
    );
    assert!(
        runner_new_section.contains("TrainingLossBackend::runtime_training_capabilities"),
        "ModelRunner training capability logging should consume the focused training facet"
    );
    assert!(
        !runner_new_section.contains("backend.training_capabilities()"),
        "ModelRunner should not call the broad BackendRuntime training capability method directly"
    );
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
    assert!(command.contains("cuda_sft_step_proof"));
    assert!(command.contains("metal_sft_step_proof"));

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
fn generated_capability_report_tracks_hardware_latency_fixture_contract() {
    let root = workspace_root();
    let report_path = root.join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let hardware_gate = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "hardware_latency_thresholds")
        .expect("hardware latency threshold gate should be present");

    assert_eq!(hardware_gate["status"], "fixture_required");
    let command = hardware_gate["command"]
        .as_str()
        .expect("hardware latency command should be a string");
    assert!(command.contains("check_backend_latency_fixtures.py"));
    assert!(command.contains("--require-covered"));

    let evidence_present = hardware_gate["evidence_present"]
        .as_array()
        .expect("hardware latency present evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "docs/backend-latency-fixtures.json",
        "scripts/check_backend_latency_fixtures.py",
        "crates/kiln-server/examples/flce_preflight_bench.rs",
        "crates/kiln-server/examples/flce_phase_a_validation_bench.rs",
        "crates/kiln-tensor/tests/metal_matmul_bench.rs",
        "crates/kiln-tensor/tests/metal_sdpa_bench.rs",
        "crates/kiln-tensor/tests/rocm_latency_bench.rs",
        "crates/kiln-vulkan-kernel/src/bin/vulkan_decode_microbench.rs",
    ] {
        assert!(
            evidence_present.contains(&path),
            "hardware latency gate should cite existing {path}"
        );
    }

    let evidence_missing = hardware_gate["evidence_missing"]
        .as_array()
        .expect("hardware latency missing evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        evidence_missing.is_empty(),
        "hardware latency gate should have no missing fixture source evidence: {evidence_missing:?}"
    );

    let manifest_path = root.join("docs/backend-latency-fixtures.json");
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(&manifest_path).expect("latency fixture manifest should be readable"),
    )
    .expect("latency fixture manifest should parse");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["status"], "fixture_required");

    let required_backends = manifest["required_backends"]
        .as_array()
        .expect("required_backends should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for backend in ["cuda", "rocm", "metal", "vulkan"] {
        assert!(
            required_backends.contains(&backend),
            "latency fixture manifest should require {backend}"
        );
    }

    let fixtures = manifest["fixtures"]
        .as_array()
        .expect("fixtures should be an array");
    let fixture_backends = fixtures
        .iter()
        .filter_map(|fixture| fixture["backend"].as_str())
        .collect::<Vec<_>>();
    for backend in ["cuda", "rocm", "metal", "vulkan"] {
        assert!(
            fixture_backends.contains(&backend),
            "latency fixture manifest should have a {backend} fixture"
        );
    }
    for fixture in fixtures {
        assert_eq!(fixture["threshold_state"], "pending_fixture_result");
        let source = fixture["source"]
            .as_str()
            .expect("fixture source should be a string");
        assert!(
            root.join(source).is_file(),
            "fixture source should exist: {source}"
        );
        let metrics = fixture["metrics"]
            .as_array()
            .expect("fixture metrics should be an array");
        assert!(
            !metrics.is_empty(),
            "fixture should declare at least one latency metric"
        );
        for metric in metrics {
            assert!(
                metric["max"].is_null(),
                "pending fixture metrics should not pretend thresholds are locked"
            );
        }
    }

    let missing_slots = manifest["missing_fixture_slots"]
        .as_array()
        .expect("missing_fixture_slots should be an array");
    assert!(
        missing_slots.is_empty(),
        "latency fixture manifest should have no missing fixture slots: {missing_slots:?}"
    );
}

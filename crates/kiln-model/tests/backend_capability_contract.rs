use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn trait_method_names(trait_source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut offset = 0usize;
    while let Some(relative_fn) = trait_source[offset..].find("fn ") {
        let name_start = offset + relative_fn + 3;
        let Some(name_len) =
            trait_source[name_start..].find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        else {
            break;
        };
        let name = &trait_source[name_start..name_start + name_len];
        if !name.is_empty() && !names.iter().any(|existing| existing == name) {
            names.push(name.to_string());
        }
        offset = name_start + name_len;
    }
    names
}

fn assert_no_broad_backend_runtime_calls(
    backend_source: &str,
    sources: &[(&str, &str)],
    receiver_prefixes: &[&str],
) {
    let runtime_trait_source = source_between(
        backend_source,
        "pub trait BackendRuntime",
        "pub trait BackendIdentity",
    );
    let methods = trait_method_names(runtime_trait_source);
    let mut failures = Vec::new();

    for (label, source) in sources {
        let compact_source = compact_body(source);
        for method in &methods {
            let mut forbidden = Vec::new();
            for receiver in receiver_prefixes {
                forbidden.push(format!("{receiver}.{method}("));
                forbidden.push(format!("{receiver}.as_ref().{method}("));
            }
            forbidden.push(format!("BackendRuntime::{method}("));
            forbidden.push(format!("backend::BackendRuntime::{method}("));
            forbidden.push(format!("crate::backend::BackendRuntime::{method}("));

            for pattern in forbidden {
                if compact_source.contains(&pattern) {
                    failures.push(format!(
                        "{label} should not call broad facade method {pattern}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "orchestration call sites should consume focused backend facets:\n{}",
        failures.join("\n")
    );
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
        "DecodeBatcherPolicy",
        "BackendTrainingCapabilities",
        "ReplayCapabilities",
        "ReplayAuthority",
        "BackendFallbackCapabilities",
    ] {
        assert!(
            capability_descriptors.contains_key(name),
            "{name} should be present in capability_descriptors"
        );
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
    let decode_batcher_policy_fields = capability_descriptors["DecodeBatcherPolicy"]["fields"]
        .as_array()
        .expect("DecodeBatcherPolicy fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
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
    assert_eq!(vulkan["exact_gdn_backward_tile_tokens"], Value::Null);
    assert_eq!(vulkan["mixed_precision"], true);
    assert_eq!(
        policies["cuda"]["exact_gdn_backward_tile_tokens"], 1024,
        "CUDA should own its exact-GDN backward tile default in the training policy"
    );
    assert_eq!(
        policies["metal"]["exact_gdn_backward_tile_tokens"],
        Value::Null,
        "Metal should inherit the streaming tile default for exact-GDN backward"
    );

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
            "crates/kiln-vulkan-kernel/src/cmd_batch.rs",
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

    let report_md = fs::read_to_string(root.join("docs/backend-capability-report.md"))
        .expect("capability report md should be readable");
    assert!(
        report_md.contains("## Replay Authority"),
        "Markdown report should expose Phase 5 replay authority"
    );
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
        "ReplayBackend::runtime_replay_authority",
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
fn generated_capability_report_lists_focused_backend_facets() {
    let root = workspace_root();
    let report_path = root.join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");
    let facets = report["focused_backend_facets"]
        .as_object()
        .expect("focused_backend_facets should be an object");

    for (facet, required_method) in [
        ("BackendIdentity", "runtime_name"),
        ("AttentionBackend", "runtime_flash_attn_prefill"),
        ("PagedKvBackend", "runtime_paged_kv_head_major_read"),
        ("GdnBackend", "runtime_gdn_recurrent_step"),
        ("ConvBackend", "runtime_causal_conv1d_update"),
        ("LinearBackend", "runtime_linear_decode"),
        ("SamplingBackend", "runtime_linear_decode_sample"),
        ("ResidencyBackend", "runtime_register_resident_activation"),
        ("OptimizerBackend", "runtime_dispatch_adamw_step"),
        ("TrainingLossBackend", "runtime_training_precision_policy"),
        ("TrainingLossBackend", "runtime_tape_forward_backward_route"),
        ("TrainingLossBackend", "runtime_grpo_loss_route"),
        ("TrainingLossBackend", "runtime_grpo_kl_auxiliary_route"),
        ("TrainingLossBackend", "runtime_opd_loss_route"),
        ("TrainingLossBackend", "runtime_opd_phase_b_backward_route"),
        (
            "TrainingLossBackend",
            "runtime_final_rmsnorm_backward_route",
        ),
        ("ReplayBackend", "runtime_supports_replay_request"),
    ] {
        let info = facets
            .get(facet)
            .unwrap_or_else(|| panic!("focused_backend_facets should list {facet}"));
        assert_eq!(
            info["forwarding_impl"].as_str(),
            Some("blanket_backend_runtime"),
            "{facet} should keep its BackendRuntime blanket forwarding impl while the facade remains"
        );
        assert!(
            info["method_count"].as_u64().unwrap_or_default() > 0,
            "{facet} should report at least one method"
        );
        let methods = info["methods"]
            .as_array()
            .expect("focused facet methods should be an array")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert!(
            methods.contains(&required_method),
            "{facet} should report focused method {required_method}"
        );
    }

    let report_md = fs::read_to_string(root.join("docs/backend-capability-report.md"))
        .expect("capability report md should be readable");
    assert!(
        report_md.contains("## Focused Backend Facets"),
        "Markdown report should expose Phase 1 focused backend facets"
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
    let backend_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend/mod.rs should be readable");
    let capability_source =
        fs::read_to_string(root.join("crates/kiln-model/src/backend/capability.rs"))
            .expect("backend/capability.rs should be readable");
    let trainer_source = fs::read_to_string(root.join("crates/kiln-train/src/trainer.rs"))
        .expect("trainer.rs should be readable");
    let grpo_tape_source = fs::read_to_string(root.join("crates/kiln-train/src/grpo_tape_shim.rs"))
        .expect("grpo_tape_shim.rs should be readable");
    let opd_source = fs::read_to_string(root.join("crates/kiln-train/src/opd.rs"))
        .expect("opd.rs should be readable");
    let generate_source = fs::read_to_string(root.join("crates/kiln-model/src/generate.rs"))
        .expect("generate.rs should be readable");
    let lora_source = fs::read_to_string(root.join("crates/kiln-model/src/lora_loader.rs"))
        .expect("lora_loader.rs should be readable");
    let speculative_source = fs::read_to_string(root.join("crates/kiln-model/src/speculative.rs"))
        .expect("speculative.rs should be readable");
    let cuda_graph_source = fs::read_to_string(root.join("crates/kiln-model/src/cuda_graph.rs"))
        .expect("cuda_graph.rs should be readable");
    let rocm_graph_source = fs::read_to_string(root.join("crates/kiln-model/src/rocm_graph.rs"))
        .expect("rocm_graph.rs should be readable");
    let metal_graph_source = fs::read_to_string(root.join("crates/kiln-model/src/metal_graph.rs"))
        .expect("metal_graph.rs should be readable");
    let tape_forward_source =
        fs::read_to_string(root.join("crates/kiln-model/src/tape_forward.rs"))
            .expect("tape_forward.rs should be readable");
    let forward_source = fs::read_to_string(root.join("crates/kiln-model/src/forward.rs"))
        .expect("forward.rs should be readable");

    assert_no_broad_backend_runtime_calls(
        &backend_source,
        &[
            ("crates/kiln-model/src/generate.rs", &generate_source),
            ("crates/kiln-model/src/lora_loader.rs", &lora_source),
            ("crates/kiln-model/src/speculative.rs", &speculative_source),
            ("crates/kiln-model/src/cuda_graph.rs", &cuda_graph_source),
            ("crates/kiln-model/src/rocm_graph.rs", &rocm_graph_source),
            ("crates/kiln-model/src/metal_graph.rs", &metal_graph_source),
            (
                "crates/kiln-model/src/tape_forward.rs",
                &tape_forward_source,
            ),
            ("crates/kiln-model/src/forward.rs", &forward_source),
            ("crates/kiln-train/src/trainer.rs", &trainer_source),
            ("crates/kiln-train/src/opd.rs", &opd_source),
        ],
        &[
            "backend",
            "_backend",
            "fallback_backend",
            "backend_rt",
            "runtime",
            "rt",
            "self.backend",
            "runner.backend",
        ],
    );

    assert!(
        trainer_source.contains("BackendCapabilityQueries"),
        "trainer should import the request-shaped backend capability query surface"
    );
    assert!(
        trainer_source.contains("BackendIdentity"),
        "trainer should import the focused backend identity facet"
    );
    assert!(
        generate_source.contains("BackendIdentity"),
        "generate should import the focused backend identity facet"
    );
    assert!(
        forward_source.contains("BackendIdentity"),
        "forward should import the focused backend identity facet"
    );
    assert!(
        forward_source.contains("BackendCapabilityQueries"),
        "forward should import the shared backend capability query surface"
    );
    assert!(
        generate_source.contains("TrainingLossBackend"),
        "ModelRunner should import the focused training loss/capability facet"
    );
    assert!(
        generate_source.contains("BackendCapabilityQueries"),
        "decode fallback policy should import the shared backend capability query surface"
    );
    assert!(
        speculative_source.contains("try_device_logits_to_probs"),
        "speculative rejection sampling should expose a device-dispatched logits-to-probs path"
    );
    assert!(
        speculative_source.contains("kiln_tensor::ops::div_scalar")
            && speculative_source.contains("softmax_last_dim"),
        "speculative rejection sampling should use kt device-dispatched scalar and softmax ops"
    );
    for forbidden in [
        "try_kt_logits_to_probs",
        "kiln_tensor::cuda_scalar_op",
        "kiln_tensor::cuda_softmax_last_axis",
        "matches!(logits.device(), kiln_tensor::Device::Cuda(_))",
    ] {
        assert!(
            !speculative_source.contains(forbidden),
            "speculative rejection sampling should not depend on CUDA-only logits probability path {forbidden}"
        );
    }
    assert!(
        generate_source.contains("DecodeBatcherPolicy"),
        "decode batcher defaults should import the backend-owned policy surface"
    );
    assert!(
        generate_source.contains("rowwise_retry_env"),
        "decode batcher rowwise retry should be backend-owned policy, not a local backend-name branch"
    );
    assert!(
        generate_source.contains("greedy_token_decode_enabled")
            && generate_source.contains("use_greedy_token_decode"),
        "greedy-token decode routing should read DecodeBatcherPolicy"
    );
    assert!(
        generate_source.contains("prefix_cache_split_snapshot_allowed")
            && generate_source.contains("allow_prefix_cache_split_snapshot"),
        "prefix-cache split snapshot routing should read DecodeBatcherPolicy"
    );
    assert!(
        generate_source.contains("ReplayBackend"),
        "generate decode residency gates should import the focused replay facet"
    );
    assert!(
        generate_source.contains("ReplayNativePrimitive")
            && generate_source.contains("ReplayRequest")
            && generate_source.contains("paged_decode_replay_primitive_enabled"),
        "generate graph replay routing should consume typed replay requests and replay authority"
    );
    assert!(
        generate_source.contains("ResidencyBackend"),
        "generate GDN recurrent residency scopes should import the focused residency facet"
    );
    assert!(
        generate_source.contains("SamplingBackend"),
        "generate decode sampling gates should import the focused sampling facet"
    );
    assert!(
        generate_source.contains("LinearBackend"),
        "generate decode weight residency helpers should import the focused linear facet"
    );
    assert!(
        metal_graph_source.contains("SamplingBackend"),
        "Metal graph decode sampling gates should import the focused sampling facet"
    );
    assert!(
        forward_source.contains("LinearBackend"),
        "forward dense decode helpers should import the focused linear facet"
    );
    assert!(
        forward_source.contains("ConvBackend"),
        "forward conv decode helpers should import the focused conv facet"
    );
    assert!(
        forward_source.contains("GdnBackend"),
        "forward GDN helpers should import the focused GDN facet"
    );
    assert!(
        forward_source.contains("direct_paged_decode_attention_enabled")
            && capability_source.contains("prefer_direct_paged_decode_attention")
            && capability_source.contains("direct_paged_decode_attention_env_gate"),
        "forward direct paged-decode attention routing should read DecodeBatcherPolicy"
    );
    assert!(
        forward_source.contains("flash_prefill_consumes_grouped_kv")
            && capability_source.contains("flash_prefill_consumes_grouped_kv"),
        "forward flash-prefill GQA routing should read AttentionCapabilities"
    );
    assert!(
        forward_source.contains("paged_decode_requires_contiguous_kv_chunks")
            && capability_source.contains("paged_decode_requires_contiguous_kv_chunks"),
        "forward paged-decode KV contiguity routing should read DecodeBatcherPolicy"
    );
    assert!(
        forward_source.contains("gdn_recurrent_step_supports_dtype")
            && capability_source.contains("recurrent_step_f32"),
        "forward GDN recurrent-step dtype routing should read GdnCapabilities"
    );
    assert!(
        forward_source.contains("ResidencyBackend"),
        "forward GDN recurrent residency helpers should import the focused residency facet"
    );
    assert!(
        forward_source.contains("SamplingBackend"),
        "forward lm-head decode helpers should import the focused sampling facet"
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
    assert!(
        decode_policy_section.contains("decode_hot_path_debug_env"),
        "decode debug fallback opt-in should be read from BackendFallbackCapabilities"
    );
    let decode_debug_policy_section = source_between(
        &generate_source,
        "fn decode_hot_path_debug_fallback_enabled(",
        "fn decode_hot_path_fallback_disabled_context(",
    );
    assert!(
        !decode_debug_policy_section.contains("\"metal\"")
            && !decode_debug_policy_section.contains("\"vulkan\"")
            && !decode_debug_policy_section.contains("\"rocm\"")
            && !decode_debug_policy_section.contains("match backend_name"),
        "decode debug fallback opt-in should not keep a local backend-name env table"
    );

    let decode_batcher_config_section = source_between(
        &generate_source,
        "pub fn from_env_for_backend_kt(",
        "pub fn from_env_for_device_kt(",
    );
    assert!(
        decode_batcher_config_section.contains("DecodeBatcherPolicy::for_backend"),
        "decode batcher backend-aware defaults should come from the shared policy object"
    );
    let decode_batcher_retry_section = source_between(
        &generate_source,
        "fn decode_batcher_rowwise_retry_enabled(",
        "fn decode_batch_generic_fallback_enabled(",
    );
    assert!(
        decode_batcher_retry_section.contains("BackendCapabilityQueries::backend_capabilities"),
        "decode batcher rowwise retry should come from the shared backend capability aggregate"
    );
    assert!(
        !decode_batcher_retry_section.contains("runtime_name")
            && !decode_batcher_retry_section.contains("\"vulkan\""),
        "decode batcher rowwise retry should not branch on backend name locally"
    );
    let prefix_cache_split_section = source_between(
        &generate_source,
        "let capture_prefix_split =",
        "let split_pos = capture_prefix_split",
    );
    assert!(
        prefix_cache_split_section
            .contains("prefix_cache_split_snapshot_allowed(self.backend.as_ref())"),
        "prefix-cache split snapshot routing should read the backend policy helper"
    );
    assert!(
        !prefix_cache_split_section.contains("Device::Rocm")
            && !prefix_cache_split_section.contains("self.weights.embed_tokens.device()"),
        "prefix-cache split snapshot routing should not branch on ROCm device identity"
    );
    let direct_paged_decode_attention_helper = source_between(
        &forward_source,
        "fn direct_paged_decode_attention_enabled(",
        "/// Try the fused paged-decode flash-attention kernel.",
    );
    assert!(
        direct_paged_decode_attention_helper
            .contains("BackendCapabilityQueries::backend_capabilities")
            && direct_paged_decode_attention_helper
                .contains("AttentionBackend::runtime_supports_flash_attn_paged_decode"),
        "direct paged-decode attention routing should combine backend policy with focused attention capability"
    );
    let direct_paged_decode_attention_section = source_between(
        &forward_source,
        "let use_direct_paged_decode =",
        "#[cfg(feature = \"cuda\")]",
    );
    assert!(
        direct_paged_decode_attention_section
            .contains("direct_paged_decode_attention_enabled(backend)"),
        "try_flash_attn_paged_decode should read the shared direct paged-decode attention helper"
    );
    for forbidden in [
        "BackendIdentity::runtime_name(backend) == \"cuda\"",
        "BackendIdentity::runtime_name(backend) == \"vulkan\"",
        "BackendIdentity::runtime_name(backend) == \"rocm\"",
        "cuda_direct_paged_decode_disabled()",
        "rocm_paged_decode_enabled()",
    ] {
        assert!(
            !direct_paged_decode_attention_section.contains(forbidden),
            "try_flash_attn_paged_decode should not branch locally on backend identity/env helper: {forbidden}"
        );
    }
    let flash_prefill_gqa_section = source_between(
        &forward_source,
        "fn flash_attention_forward(",
        "let Some(attn_output)",
    );
    assert!(
        flash_prefill_gqa_section.contains("flash_prefill_consumes_grouped_kv(backend)"),
        "flash_attention_forward should route grouped-KV expansion through AttentionCapabilities"
    );
    assert!(
        !flash_prefill_gqa_section.contains("BackendIdentity::runtime_name(backend) != \"cuda\""),
        "flash_attention_forward should not branch on CUDA backend identity for grouped-KV expansion"
    );
    let gdn_recurrent_step_dtype_section = source_between(
        &forward_source,
        "// Single-token decode fast path.",
        "if use_backend_recurrent_step {",
    );
    assert!(
        gdn_recurrent_step_dtype_section
            .contains("gdn_recurrent_step_supports_dtype(backend, dtype)"),
        "single-token GDN recurrent-step routing should ask the dtype capability helper"
    );
    assert!(
        !gdn_recurrent_step_dtype_section
            .contains("BackendIdentity::runtime_name(backend) == \"vulkan\"")
            && !gdn_recurrent_step_dtype_section
                .contains("vulkan_gdn_recurrent_step_f32_enabled()"),
        "single-token GDN recurrent-step routing should not branch on Vulkan identity/env locally"
    );
    let paged_decode_kv_contiguity_section = source_between(
        &forward_source,
        "// Verify intra-chunk contiguity.",
        "// Build a padded block_table tensor",
    );
    assert!(
        paged_decode_kv_contiguity_section
            .contains("paged_decode_requires_contiguous_kv_chunks(backend)"),
        "paged-decode KV contiguity guard should read the backend policy helper"
    );
    assert!(
        !paged_decode_kv_contiguity_section
            .contains("BackendIdentity::runtime_name(backend) != \"vulkan\""),
        "paged-decode KV contiguity guard should not branch on Vulkan backend identity"
    );
    for removed_helper in [
        "fn default_decode_batcher_max_batch_kt",
        "fn default_decode_batcher_allow_mixed_seq_lens_kt",
        "fn default_decode_batcher_wait_kt",
    ] {
        assert!(
            !generate_source.contains(removed_helper),
            "generate should not keep local decode batcher backend policy helper {removed_helper}"
        );
    }
    let gdn_contiguity_partition_section = source_between(
        &generate_source,
        "// #1082 PERF + CRASHER FIX (per-row contiguity partition).",
        "let pc_guard = lock_paged_cache(paged_cache)?;",
    );
    assert!(
        gdn_contiguity_partition_section.contains("BackendCapabilityQueries::backend_capabilities"),
        "GDN KV contiguity partition should read DecodeBatcherPolicy"
    );
    assert!(
        gdn_contiguity_partition_section.contains("partition_noncontiguous_gdn_kv_tiles"),
        "GDN KV contiguity partition should be controlled by backend policy"
    );
    assert!(
        !gdn_contiguity_partition_section.contains("self.backend_name() == \"cuda\""),
        "GDN KV contiguity partition should not branch on backend name in ModelRunner"
    );
    let sampled_contiguous_decode_section = source_between(
        &generate_source,
        "let sampled_contiguous_resident_decode_ready =",
        "// R.9: ROCm HIP-graph single-row decode",
    );
    assert!(
        sampled_contiguous_decode_section.contains("use_native_sampled_contiguous_decode")
            && sampled_contiguous_decode_section
                .contains("sampled_contiguous_decode_requires_resident_decode"),
        "sampled contiguous decode routing should read DecodeBatcherPolicy"
    );
    assert!(
        !sampled_contiguous_decode_section.contains("self.backend_name() == \"vulkan\""),
        "sampled contiguous decode routing should not branch on backend name in ModelRunner"
    );
    assert!(
        !sampled_contiguous_decode_section
            .contains("matches!(self.backend_device(), kiln_tensor::Device::Metal(_))"),
        "sampled contiguous decode routing should not branch on Metal device identity in ModelRunner"
    );
    for forbidden in [
        "matches!(self.backend_device(), kiln_tensor::Device::Metal(_))",
        "matches!(self.backend_device(), kiln_tensor::Device::Rocm(_))",
    ] {
        assert!(
            !generate_source.contains(forbidden),
            "ModelRunner generation routing should not branch on backend device identity: {forbidden}"
        );
    }

    let graph_replay_routing_section = source_between(
        &generate_source,
        "fn paged_decode_replay_primitive_enabled(",
        "fn decode_hot_path_fallback_disabled_context(",
    );
    assert!(
        graph_replay_routing_section.contains("ReplayBackend::runtime_supports_replay_request")
            && graph_replay_routing_section.contains("ReplayBackend::runtime_replay_authority")
            && graph_replay_routing_section.contains("native_support_enabled"),
        "paged decode graph replay routing should use the focused ReplayBackend facet"
    );

    let batched_rocm_graph_section = source_between(
        &generate_source,
        "// R.9: ROCm HIP-graph single-row decode for the batched/batching-engine",
        "let pc_guard = lock_paged_cache(paged_cache)?;",
    );
    assert!(
        batched_rocm_graph_section.contains("paged_decode_replay_primitive_enabled")
            && batched_rocm_graph_section.contains("ReplayNativePrimitive::HipGraph"),
        "batched ROCm graph routing should use replay primitive policy"
    );
    assert!(
        !batched_rocm_graph_section
            .contains("matches!(self.backend_device(), kiln_tensor::Device::Rocm(_))"),
        "batched ROCm graph routing should not branch on device identity"
    );

    let greedy_metal_graph_section = source_between(
        &generate_source,
        "fn decode_next_token_paged_greedy_metal_graph(",
        "fn decode_next_token_paged_sample_metal_graph(",
    );
    assert!(
        greedy_metal_graph_section.contains("paged_decode_replay_primitive_enabled")
            && greedy_metal_graph_section.contains("ReplayNativePrimitive::MetalIcb"),
        "single-row Metal graph routing should use replay primitive policy"
    );
    assert!(
        !greedy_metal_graph_section
            .contains("matches!(self.backend_device(), kiln_tensor::Device::Metal(_))"),
        "single-row Metal graph routing should not branch on device identity"
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
    assert!(
        trainer_policy_section.contains("training_optimizer_debug_env"),
        "trainer optimizer debug fallback opt-in should be read from BackendFallbackCapabilities"
    );
    let trainer_debug_policy_section = source_between(
        &trainer_source,
        "fn training_optimizer_debug_fallback_enabled(",
        "fn training_optimizer_fallback_policy(",
    );
    assert!(
        !trainer_debug_policy_section.contains("\"cuda\"")
            && !trainer_debug_policy_section.contains("\"metal\"")
            && !trainer_debug_policy_section.contains("\"vulkan\"")
            && !trainer_debug_policy_section.contains("\"rocm\"")
            && !trainer_debug_policy_section.contains("match backend_name"),
        "trainer optimizer debug fallback opt-in should not keep a local backend-name env table"
    );

    let orchestration_identity_sources =
        format!("{trainer_source}\n{generate_source}\n{forward_source}");
    let orchestration_identity_compact = orchestration_identity_sources
        .split_whitespace()
        .collect::<String>();
    for required in [
        "BackendIdentity::runtime_name",
        "BackendIdentity::runtime_device",
        "BackendIdentity::runtime_as_any",
    ] {
        assert!(
            orchestration_identity_sources.contains(required),
            "orchestration identity reads should consume focused identity facet method {required}"
        );
    }
    for forbidden in ["backend.name()", "backend.device()", "backend.as_any()"] {
        assert!(
            !orchestration_identity_compact.contains(forbidden),
            "orchestration identity reads should not call broad BackendRuntime method {forbidden}"
        );
    }

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
    assert!(
        trainer_source.contains("TrainingLossBackend::runtime_sft_flce_loss_route"),
        "SFT FLCE routing should consume the focused training-loss capability facet"
    );
    assert!(
        trainer_source.contains("TrainingLossBackend::runtime_grpo_loss_route"),
        "GRPO loss routing should consume the focused training-loss capability facet"
    );
    assert!(
        trainer_source.contains("TrainingLossBackend::runtime_grpo_kl_auxiliary_route")
            && grpo_tape_source.contains("grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute")
            && grpo_tape_source.contains("grpo_loss_with_kl_auxiliary_route")
            && grpo_tape_source.contains("GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath"),
        "GRPO KL auxiliary routing should consume backend policy and route shim fast paths through it"
    );
    assert!(
        trainer_source.contains("TrainingLossBackend::runtime_final_rmsnorm_backward_route"),
        "final RMSNorm tail backward routing should consume the focused training-loss capability facet"
    );
    assert!(
        opd_source.contains("TrainingLossBackend::runtime_opd_loss_route"),
        "OPD loss routing should consume the focused training-loss capability facet"
    );
    assert!(
        trainer_source.contains("TrainingPrecisionPolicy"),
        "trainer should import the backend-owned training precision policy"
    );
    assert!(
        trainer_source.contains("fn training_precision_policy_for_device("),
        "trainer should centralize device-family precision policy lookup"
    );
    assert!(
        trainer_source.contains("fn training_precision_policy_for_backend("),
        "trainer production paths should centralize backend precision policy lookup"
    );
    assert!(
        !trainer_source.contains("fn is_metal_device(")
            && !trainer_source.contains("fn is_cuda_device("),
        "trainer should not keep local backend identity helpers for training policy"
    );
    let lora_init_section = source_between(
        &trainer_source,
        "pub fn initialize_seeded(",
        "// Kaiming uniform bound",
    );
    assert!(
        lora_init_section.contains("initialize_seeded_with_precision_policy")
            && compact_body(lora_init_section).contains(
                "precision_policy.lora_parameter_dtype_for_base_weight(weights.embed_tokens.dtype())"
            ),
        "LoRA initialization should choose parameter dtype through TrainingPrecisionPolicy"
    );
    assert!(
        trainer_source.contains("training_precision_policy_for_backend(backend.as_ref())")
            && trainer_source
                .contains("TrainableLoraParams::initialize_seeded_with_precision_policy")
            && opd_source.contains("training_precision_policy_for_backend(backend_rt.as_ref())")
            && opd_source.contains("TrainableLoraParams::initialize_seeded_with_precision_policy"),
        "SFT/GRPO/OPD production setup should pass backend precision policy into LoRA initialization"
    );
    assert!(
        !lora_init_section.contains("let lora_dtype = if is_vulkan_device(device)"),
        "LoRA initialization should not hard-code Vulkan for precision policy"
    );
    let activation_bytes_section = source_between(
        &trainer_source,
        "fn training_activation_bytes_per_elem_for_policy(",
        "pub(crate) fn training_activation_bytes_per_elem(",
    );
    assert!(
        activation_bytes_section.contains("uses_f32_activations_for_mixed_base_weights"),
        "training activation sizing should read F32-activation policy from TrainingPrecisionPolicy"
    );
    assert!(
        compact_body(&trainer_source).contains(
            "training_activation_bytes_per_elem_for_policy(weights,training_precision_policy)"
        ) && compact_body(&opd_source).contains("training_activation_bytes_per_elem_for_backend(")
            && compact_body(&opd_source).contains("backend_rt.as_ref()"),
        "SFT/GRPO/OPD production checkpoint sizing should consume backend precision policy"
    );
    let exact_gdn_tile_section = source_between(
        &trainer_source,
        "fn exact_gdn_backward_tile_tokens_for(",
        "// (#1082) Deleted three orphaned residues",
    );
    assert!(
        exact_gdn_tile_section.contains("training_precision_policy_for_device"),
        "exact-GDN backward tile defaults should read backend training policy"
    );
    assert!(
        !exact_gdn_tile_section.contains("is_cuda_device"),
        "exact-GDN backward tile defaults should not hard-code CUDA in the trainer"
    );
    let base_dtype_support_section = source_between(
        &trainer_source,
        "fn base_dtype_supports_tape_for_policy(",
        "/// (#1082 Increment-0 PR2) kt-native sibling",
    );
    assert!(
        base_dtype_support_section.contains("uses_f32_activations_for_mixed_base_weights"),
        "base dtype tape support should read mixed F32 activation policy from TrainingPrecisionPolicy"
    );
    assert!(
        trainer_source.contains("base_dtype_supports_tape_for_backend(weights, backend)"),
        "SFT/GRPO tape eligibility should consume backend precision policy"
    );
    assert!(
        trainer_source.contains("fn backend_supports_tape_forward_backward(")
            && trainer_source
                .contains("TrainingLossBackend::runtime_tape_forward_backward_route(backend)")
            && trainer_source.contains("backend_supports_tape_forward_backward(backend)"),
        "SFT/GRPO tape eligibility should consume backend tape-forward/backward capability"
    );
    assert!(
        !compact_body(&trainer_source).contains(
            "matches!(device,kiln_tensor::Device::Cuda(_)|kiln_tensor::Device::Metal(_)|kiln_tensor::Device::Vulkan(_)|kiln_tensor::Device::Rocm(_))"
        ),
        "trainer SFT/GRPO tape eligibility should not hard-code GPU device families"
    );
    assert!(
        trainer_source.contains("SftFlceLossRoute::KtTapeFlce")
            && trainer_source.contains("SftFlceLossRoute::VulkanActiveRows"),
        "trainer SFT FLCE routing should match on typed backend-owned loss routes"
    );
    let checkpointed_sft_section = source_between(
        &trainer_source,
        "fn checkpointed_forward_backward_tape_authoritative_kt(",
        "fn grpo_step_forward_backward_tape_authoritative_kt(",
    );
    assert!(
        checkpointed_sft_section.contains("final_rmsnorm_backward_route_for_backend(backend)"),
        "checkpointed SFT tail should route final RMSNorm backward through TrainingLossBackend"
    );
    for forbidden in [
        "use_sft_flce && is_cuda_device",
        "use_sft_flce && is_vulkan_device",
    ] {
        assert!(
            !trainer_source.contains(forbidden),
            "trainer SFT FLCE routing should not branch directly on device checks: {forbidden}"
        );
    }
    let grpo_step_section = source_between(
        &trainer_source,
        "fn grpo_step_forward_backward_tape_authoritative_kt(",
        "fn checkpointed_grpo_forward_backward_tape_authoritative_kt(",
    );
    assert!(
        grpo_step_section.contains("TrainingLossBackend::runtime_grpo_loss_route"),
        "GRPO tape-authoritative step should route fused loss roots through TrainingLossBackend"
    );
    assert!(
        !grpo_step_section.contains("if is_vulkan_device(device)"),
        "GRPO tape-authoritative step should not hard-code Vulkan loss routing"
    );
    let checkpointed_grpo_section = source_between(
        &trainer_source,
        "fn checkpointed_grpo_forward_backward_tape_authoritative_kt(",
        "fn entropy_aware_kl_threshold_from_policy_log_probs(",
    );
    assert!(
        checkpointed_grpo_section.contains("TrainingLossBackend::runtime_grpo_loss_route"),
        "checkpointed GRPO tail should route fused loss roots through TrainingLossBackend"
    );
    assert!(
        checkpointed_grpo_section.contains("final_rmsnorm_backward_route_for_backend(backend)"),
        "checkpointed GRPO tail should route final RMSNorm backward through TrainingLossBackend"
    );
    assert!(
        !checkpointed_grpo_section.contains("if is_vulkan_device(device)"),
        "checkpointed GRPO tail should not hard-code Vulkan loss routing"
    );
    let opd_step_section = source_between(
        &opd_source,
        "fn opd_step_forward_backward_tape_authoritative(",
        "fn checkpointed_opd_step_forward_backward_tape_authoritative(",
    );
    assert!(
        opd_step_section.contains("TrainingLossBackend::runtime_opd_loss_route"),
        "OPD tape-authoritative step should route fused loss roots through TrainingLossBackend"
    );
    assert!(
        !opd_step_section.contains("matches!(normed.device(), kiln_tensor::Device::Vulkan(_))"),
        "OPD tape-authoritative step should not hard-code Vulkan loss routing"
    );
    let checkpointed_opd_section = source_between(
        &opd_source,
        "fn checkpointed_opd_step_forward_backward_tape_authoritative(",
        "fn write_opd_train_receipt_best_effort(",
    );
    assert!(
        checkpointed_opd_section.contains("TrainingLossBackend::runtime_opd_loss_route"),
        "checkpointed OPD tail should route fused loss roots through TrainingLossBackend"
    );
    assert!(
        checkpointed_opd_section
            .contains("TrainingLossBackend::runtime_opd_phase_b_backward_route"),
        "checkpointed OPD tail should route Phase-B backward through TrainingLossBackend"
    );
    assert!(
        checkpointed_opd_section
            .contains("TrainingLossBackend::runtime_final_rmsnorm_backward_route"),
        "checkpointed OPD tail should route final RMSNorm backward through TrainingLossBackend"
    );
    assert!(
        !checkpointed_opd_section
            .contains("matches!(normed.device(), kiln_tensor::Device::Vulkan(_))"),
        "checkpointed OPD tail should not hard-code Vulkan loss routing"
    );
    assert!(
        !checkpointed_opd_section
            .contains("kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Rocm(_)"),
        "checkpointed OPD tail should not hard-code CUDA/ROCm Phase-B backward routing"
    );

    let inference_decode_residency_sources =
        format!("{generate_source}\n{metal_graph_source}\n{forward_source}");
    for required in [
        "ReplayBackend::runtime_supports_resident_decode",
        "ReplayBackend::runtime_decode_resident_pool_ready",
    ] {
        assert!(
            inference_decode_residency_sources.contains(required),
            "inference decode gates should consume focused capability facet method {required}"
        );
    }
    for forbidden in [".supports_resident_decode(", ".decode_resident_pool_ready("] {
        assert!(
            !inference_decode_residency_sources.contains(forbidden),
            "inference decode gates should not call broad BackendRuntime method {forbidden}"
        );
    }

    let gdn_recurrent_residency_sources = format!("{generate_source}\n{forward_source}");
    for required in [
        "ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope",
        "ResidencyBackend::runtime_exit_gdn_recurrent_resident_state_scope",
        "ResidencyBackend::runtime_materialize_gdn_recurrent_resident_state",
        "ResidencyBackend::runtime_evict_gdn_recurrent_resident_state",
        "ResidencyBackend::runtime_has_gdn_recurrent_resident_state",
        "ResidencyBackend::runtime_assemble_gdn_recurrent_resident_batch_rows",
        "ResidencyBackend::runtime_scatter_gdn_recurrent_resident_batch_rows",
        "ResidencyBackend::runtime_assemble_linear_attn_gdn_state_batch_kt",
        "ResidencyBackend::runtime_scatter_linear_attn_gdn_state_batch_kt",
        "ResidencyBackend::runtime_seed_linear_attn_gdn_state_kt",
        "ResidencyBackend::runtime_has_linear_attn_gdn_state_kt",
    ] {
        assert!(
            gdn_recurrent_residency_sources.contains(required),
            "GDN recurrent residency should consume focused residency facet method {required}"
        );
    }
    for forbidden in [
        ".enter_gdn_recurrent_resident_state_scope(",
        ".exit_gdn_recurrent_resident_state_scope(",
        ".materialize_gdn_recurrent_resident_state(",
        ".evict_gdn_recurrent_resident_state(",
        ".has_gdn_recurrent_resident_state(",
        ".assemble_gdn_recurrent_resident_batch_rows(",
        ".scatter_gdn_recurrent_resident_batch_rows(",
        ".assemble_linear_attn_gdn_state_batch_kt(",
        ".scatter_linear_attn_gdn_state_batch_kt(",
        ".seed_linear_attn_gdn_state_kt(",
        ".has_linear_attn_gdn_state_kt(",
    ] {
        assert!(
            !gdn_recurrent_residency_sources.contains(forbidden),
            "GDN recurrent residency should not call broad BackendRuntime method {forbidden}"
        );
    }

    let inference_decode_sampling_sources =
        format!("{generate_source}\n{metal_graph_source}\n{forward_source}");
    for required in [
        "SamplingBackend::runtime_supports_linear_decode_sample",
        "SamplingBackend::runtime_linear_decode_sample",
        "SamplingBackend::runtime_linear_decode_argmax",
    ] {
        assert!(
            inference_decode_sampling_sources.contains(required),
            "inference decode sampling should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".supports_linear_decode_sample(",
        ".supports_linear_decode_sample_batch(",
        ".supports_linear_decode_argmax_batch(",
        ".linear_decode_argmax(",
        ".linear_decode_argmax_batch(",
        ".linear_decode_sample(",
        ".linear_decode_sample_batch(",
    ] {
        assert!(
            !inference_decode_sampling_sources.contains(forbidden),
            "inference decode sampling should not call broad BackendRuntime method {forbidden}"
        );
    }

    let inference_linear_sources = format!("{generate_source}\n{forward_source}");
    for required in [
        "LinearBackend::runtime_prewarm_decode_weights",
        "LinearBackend::runtime_drop_uploaded_bf16_weights",
        "LinearBackend::runtime_lora_decode_add",
        "LinearBackend::runtime_lora_delta_resident",
        "LinearBackend::runtime_linear_prefill_apply",
        "LinearBackend::runtime_linear_decode",
        "LinearBackend::runtime_full_attn_qkv_decode",
        "LinearBackend::runtime_mlp_decode",
        "LinearBackend::runtime_mlp_gate_up_decode",
    ] {
        assert!(
            inference_linear_sources.contains(required),
            "inference dense linear paths should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".prewarm_decode_weights(",
        ".drop_uploaded_bf16_weights(",
        ".lora_decode_add(",
        ".lora_delta_resident(",
        ".linear_prefill_apply(",
        ".linear_decode(",
        ".full_attn_qkv_decode(",
        ".mlp_decode(",
        ".mlp_gate_up_decode(",
    ] {
        assert!(
            !inference_linear_sources.contains(forbidden),
            "inference dense linear paths should not call broad BackendRuntime method {forbidden}"
        );
    }

    for required in [
        "AttentionBackend::runtime_supports_flash_attn_prefill",
        "AttentionBackend::runtime_supports_flash_attn_prefill_head_major",
        "AttentionBackend::runtime_supports_flash_attn_paged_decode",
        "AttentionBackend::runtime_supports_strict_paged_decode_contiguous_batch",
        "AttentionBackend::runtime_flash_attn_prefill",
        "AttentionBackend::runtime_flash_attn_prefill_head_major",
        "AttentionBackend::runtime_flash_attn_paged_decode_contiguous",
        "AttentionBackend::runtime_flash_attn_paged_decode_contiguous_batch",
        "AttentionBackend::runtime_flash_attn_paged_decode",
        "ReplayBackend::runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs",
    ] {
        assert!(
            forward_source.contains(required),
            "forward attention paths should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".supports_flash_attn_prefill(",
        ".supports_flash_attn_prefill_head_major(",
        ".supports_flash_attn_paged_decode(",
        ".supports_strict_paged_decode_contiguous_batch(",
        ".flash_attn_prefill(",
        ".flash_attn_prefill_head_major(",
        ".flash_attn_paged_decode_contiguous(",
        ".flash_attn_paged_decode_contiguous_batch(",
        ".flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(",
        ".flash_attn_paged_decode(",
    ] {
        assert!(
            !forward_source.contains(forbidden),
            "forward attention paths should not call broad BackendRuntime method {forbidden}"
        );
    }

    for required in [
        "PagedKvBackend::runtime_supports_paged_kv_head_major_read",
        "PagedKvBackend::runtime_supports_paged_kv_head_major_read_append_token_major",
        "PagedKvBackend::runtime_paged_kv_head_major_read",
        "PagedKvBackend::runtime_paged_kv_head_major_read_append_token_major",
    ] {
        assert!(
            forward_source.contains(required),
            "forward paged-KV paths should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".supports_paged_kv_head_major_read(",
        ".supports_paged_kv_head_major_read_append_token_major(",
        ".paged_kv_head_major_read(",
        ".paged_kv_head_major_read_append_token_major(",
    ] {
        assert!(
            !forward_source.contains(forbidden),
            "forward paged-KV paths should not call broad BackendRuntime method {forbidden}"
        );
    }

    for required in [
        "GdnBackend::runtime_supports_gdn_gated_rms_norm",
        "GdnBackend::runtime_gdn_gated_rms_norm",
        "GdnBackend::runtime_supports_gdn_forward_substitution",
        "GdnBackend::runtime_gdn_forward_substitution",
        "GdnBackend::runtime_supports_gdn_recurrent_step",
        "GdnBackend::runtime_gdn_recurrent_step",
        "GdnBackend::runtime_gdn_chunkwise_forward",
        "GdnBackend::runtime_supports_gdn_full_chunk_forward",
        "GdnBackend::runtime_gdn_full_chunk_forward",
        "GdnBackend::runtime_supports_gdn_chunk_prep",
        "GdnBackend::runtime_gdn_chunk_prep",
        "GdnBackend::runtime_supports_gdn_chunk_scan",
        "GdnBackend::runtime_gdn_chunk_scan",
        "GdnBackend::runtime_supports_gdn_recurrent_prefill_head_last",
        "GdnBackend::runtime_gdn_recurrent_prefill_head_last",
        "GdnBackend::runtime_supports_gdn_recurrent_prefill_native_head_last",
        "GdnBackend::runtime_gdn_recurrent_prefill_native_head_last",
        "GdnBackend::runtime_supports_gdn_full_chunk_forward_head_last",
        "GdnBackend::runtime_gdn_full_chunk_forward_head_last_into",
        "GdnBackend::runtime_gdn_in_proj_decode",
        "GdnBackend::runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk",
        "GdnBackend::runtime_supports_gdn_decode_qk_norm_gates_recurrent",
        "GdnBackend::runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last",
        "GdnBackend::runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm",
        "GdnBackend::runtime_gdn_decode_qk_norm_gates_recurrent",
        "GdnBackend::runtime_gdn_decode_gates_recurrent",
        "GdnBackend::runtime_supports_gdn_gates",
        "GdnBackend::runtime_gdn_gates",
        "GdnBackend::runtime_gdn_recurrent_qk_norm_prefill_native_head_last",
    ] {
        assert!(
            forward_source.contains(required),
            "forward GDN paths should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".supports_gdn_",
        ".gdn_forward_substitution(",
        ".gdn_recurrent_step(",
        ".gdn_chunkwise_forward(",
        ".gdn_chunk_prep(",
        ".gdn_chunk_scan(",
        ".gdn_full_chunk_forward(",
        ".gdn_full_chunk_forward_head_last_into(",
        ".gdn_recurrent_prefill_head_last(",
        ".gdn_recurrent_prefill_native_head_last(",
        ".gdn_recurrent_qk_norm_prefill_native_head_last(",
        ".gdn_decode_gates_recurrent(",
        ".gdn_decode_qk_norm_gates_recurrent(",
        ".gdn_decode_qk_norm_gates_recurrent_rmsnorm(",
        ".gdn_in_proj_decode(",
        ".gdn_gates(",
        ".gdn_gated_rms_norm(",
    ] {
        assert!(
            !forward_source.contains(forbidden),
            "forward GDN paths should not call broad BackendRuntime method {forbidden}"
        );
    }

    for required in [
        "ConvBackend::runtime_supports_causal_conv1d_update",
        "ConvBackend::runtime_causal_conv1d_update",
        "ConvBackend::runtime_supports_causal_conv1d_prefill",
        "ConvBackend::runtime_causal_conv1d_prefill",
    ] {
        assert!(
            forward_source.contains(required),
            "forward conv paths should consume focused capability facet method {required}"
        );
    }
    for forbidden in [
        ".supports_causal_conv1d_update(",
        ".causal_conv1d_update(",
        ".supports_causal_conv1d_prefill(",
        ".causal_conv1d_prefill(",
    ] {
        assert!(
            !forward_source.contains(forbidden),
            "forward conv paths should not call broad BackendRuntime method {forbidden}"
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
    assert!(command.contains("run_backend_latency_fixture.py"));
    assert!(command.contains("write_backend_latency_result_artifact.py"));
    assert!(command.contains("lock_backend_latency_thresholds.py"));
    assert!(command.contains("--self-test"));
    assert!(command.contains("--require-covered"));

    let checker_source = fs::read_to_string(root.join("scripts/check_backend_latency_fixtures.py"))
        .expect("latency fixture checker should be readable");
    for required in [
        "def validate_result_artifact(",
        "def metric_threshold_passes(",
        "MANIFEST_KEYS",
        "REQUIRED_COVERED_GATE_POLICY",
        "manifest contains unknown keys",
        "policy.covered_gate_requires must match",
        "status covered requires --require-covered",
        "artifact_schema_version",
        "created_at_utc",
        "git_commit",
        "git_commit_exists",
        "git_file_sha256_at_commit",
        "git_commit must exist in the local repository",
        "source must exist at git_commit",
        "source_sha256 must match source at git_commit",
        "git_path_is_tracked",
        "result_artifact must be tracked by git",
        "raw_log must be tracked by git",
        "git_tracked_dirty",
        "git_tracked_dirty must be false",
        "fixture_id",
        "is_repo_relative_path",
        "is_canonical_result_artifact_path",
        "is_canonical_raw_log_path",
        "must be repo-relative",
        "LATENCY_RESULT_ARTIFACT_DIR",
        "LATENCY_RAW_LOG_DIR",
        ".json extension",
        ".log extension",
        "require_raw_log_file",
        "raw_log must exist",
        "status must be passed",
        "manifest_schema_version",
        "fixture_spec_sha256",
        "source_sha256",
        "source_sha256 does not match source",
        "parse_metric_log",
        "RESULT_ARTIFACT_KEYS",
        "contains unknown keys",
        "contains undeclared metrics",
        "must match raw_log value",
        "raw_log missing metric",
        "finite numeric",
        "raw_log_sha256",
        "does not satisfy",
    ] {
        assert!(
            checker_source.contains(required),
            "latency fixture checker should validate covered artifact contract: {required}"
        );
    }

    let runner_source = fs::read_to_string(root.join("scripts/run_backend_latency_fixture.py"))
        .expect("latency fixture runner should be readable");
    for required in [
        "def run_fixture(",
        "subprocess.Popen",
        "KILN_LATENCY_METRIC",
        "raw_log",
        "git_commit",
        "git_tracked_dirty",
        "--self-test",
    ] {
        assert!(
            runner_source.contains(required),
            "latency fixture runner should capture raw fixture logs and materialize artifacts: {required}"
        );
    }

    let writer_source =
        fs::read_to_string(root.join("scripts/write_backend_latency_result_artifact.py"))
            .expect("latency result artifact writer should be readable");
    for required in [
        "def parse_metric_log(",
        "KILN_LATENCY_METRIC",
        "def build_result_artifact(",
        "ARTIFACT_SCHEMA_VERSION",
        "GIT_COMMIT_RE",
        "current_git_commit",
        "git_commit_exists",
        "cat-file",
        "git_file_sha256_at_commit",
        "git_path_is_tracked",
        "git_output_bytes",
        "ls-files",
        "git_output_bytes([\"show\", f\"{commit}:{path}\"])",
        "tracked_git_dirty",
        "rev-parse",
        "--untracked-files=no",
        "LATENCY_RESULT_ARTIFACT_DIR",
        "LATENCY_RAW_LOG_DIR",
        "bench-results/backend-latency",
        "is_canonical_result_artifact_path",
        "is_canonical_raw_log_path",
        "created_at_utc",
        "manifest_schema_version",
        "fixture_spec_sha256",
        "source_sha256",
        "math.isfinite",
        "raw_log_sha256",
        "fixture_id",
        "--self-test",
    ] {
        assert!(
            writer_source.contains(required),
            "latency result artifact writer should materialize fixture logs: {required}"
        );
    }

    let threshold_locker_source =
        fs::read_to_string(root.join("scripts/lock_backend_latency_thresholds.py"))
            .expect("latency threshold locker should be readable");
    for required in [
        "def lock_manifest_thresholds(",
        "def validate_manifest_header_for_lock(",
        "schema_version must be 1 before thresholds can lock",
        "status must be one of",
        "required_backends",
        "def validate_required_backend_coverage(",
        "required backend has no fixture before thresholds can lock",
        "locked_threshold",
        "artifact_schema_version",
        "created_at_utc",
        "git_commit",
        "git_commit_exists",
        "git_file_sha256_at_commit",
        "git_commit must exist in the local repository",
        "source must exist at git_commit",
        "source_sha256 must match source at git_commit",
        "git_tracked_dirty",
        "git_tracked_dirty must be false",
        "is_repo_relative_path",
        "is_canonical_result_artifact_path",
        "is_canonical_raw_log_path",
        "must be repo-relative",
        "LATENCY_RESULT_ARTIFACT_DIR",
        "LATENCY_RAW_LOG_DIR",
        ".json extension",
        ".log extension",
        "raw_log must exist",
        "raw_log_sha256 does not match raw_log",
        "status",
        "manifest_schema_version",
        "fixture_spec_sha256",
        "source_sha256",
        "source_sha256 does not match source",
        "parse_metric_log",
        "RESULT_ARTIFACT_KEYS",
        "contains unknown keys",
        "contains undeclared metrics",
        "must match raw_log value",
        "finite numeric",
        "covered",
        "--headroom",
        "--self-test",
    ] {
        assert!(
            threshold_locker_source.contains(required),
            "latency threshold locker should materialize covered fixture manifests: {required}"
        );
    }

    let perf_workflow =
        fs::read_to_string(root.join(".github/workflows/perf-regression-nightly.yml"))
            .expect("perf regression workflow should be readable");
    for required in [
        "Verify backend latency fixture contract",
        "scripts/run_backend_latency_fixture.py --self-test",
        "scripts/write_backend_latency_result_artifact.py --self-test",
        "scripts/lock_backend_latency_thresholds.py --self-test",
        "scripts/check_backend_latency_fixtures.py --self-test",
        "scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json",
        "docs/backend-latency-fixtures.json",
        "docs/backend-latency-result-schema.md",
    ] {
        assert!(
            perf_workflow.contains(required),
            "perf workflow should run non-hardware latency fixture validation: {required}"
        );
    }

    let evidence_present = hardware_gate["evidence_present"]
        .as_array()
        .expect("hardware latency present evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "docs/backend-latency-fixtures.json",
        "docs/backend-latency-result-schema.md",
        ".github/workflows/perf-regression-nightly.yml",
        "scripts/run_backend_latency_fixture.py",
        "scripts/write_backend_latency_result_artifact.py",
        "scripts/lock_backend_latency_thresholds.py",
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

    let schema_doc = fs::read_to_string(root.join("docs/backend-latency-result-schema.md"))
        .expect("latency result schema doc should be readable");
    for required in [
        "fixture_id",
        "schema_version: 1",
        "artifact_schema_version",
        "created_at_utc",
        "required_backends",
        "git_commit",
        "git_tracked_dirty",
        "40-character git commit",
        "local repository",
        "source bytes at that commit",
        "tracked by git",
        "backend",
        "status",
        "manifest_schema_version",
        "fixture_spec_sha256",
        "source_sha256",
        "KILN_LATENCY_METRIC",
        "must match the raw log",
        "unknown artifact keys",
        "undeclared metrics",
        "finite numeric",
        "hardware",
        "source",
        "command",
        "raw_log_sha256",
        "bench-results/backend-latency",
        ".json",
        ".log",
        "repo-relative",
        "files to exist",
        "metrics",
        "KILN_LATENCY_METRIC",
        "run_backend_latency_fixture.py",
        "write_backend_latency_result_artifact.py",
        "lock_backend_latency_thresholds.py",
        "--require-covered",
        "--self-test",
    ] {
        assert!(
            schema_doc.contains(required),
            "latency result schema doc should describe {required}"
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
    let policy = manifest["policy"]
        .as_object()
        .expect("latency fixture manifest policy should be an object");
    let covered_gate_requires = policy["covered_gate_requires"]
        .as_array()
        .expect("covered_gate_requires should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for requirement in [
        "Every required backend has at least one known hardware fixture.",
        "Every fixture has locked numeric thresholds for its required measurements.",
        "Every locked threshold has a checked hardware-result artifact from the named fixture.",
        "Default-feature local tests must not mark the hardware latency gate covered.",
    ] {
        assert!(
            covered_gate_requires.contains(&requirement),
            "latency fixture manifest policy should include {requirement}"
        );
    }

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
        let source_text = fs::read_to_string(root.join(source))
            .unwrap_or_else(|err| panic!("fixture source should be readable: {source}: {err}"));
        assert!(
            root.join(source).is_file(),
            "fixture source should exist: {source}"
        );
        assert!(
            source_text.contains("KILN_LATENCY_METRIC"),
            "fixture source should emit machine-readable latency metric lines: {source}"
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
            let metric_name = metric["name"]
                .as_str()
                .expect("fixture metric name should be a string");
            let source_mentions_metric = source_text.contains(metric_name)
                || metric_name
                    .strip_suffix("_us")
                    .is_some_and(|base| source_text.contains(base));
            assert!(
                source_mentions_metric,
                "fixture source should emit or derive metric {metric_name}: {source}"
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

#[test]
fn generated_capability_report_tracks_migration_phase_status() {
    let root = workspace_root();
    let report_path = root.join("docs/backend-capability-report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(&report_path).expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    let phases = report["migration_phase_status"]
        .as_array()
        .expect("migration_phase_status should be an array");
    assert_eq!(
        phases.len(),
        9,
        "migration phase status should cover phases 0-8"
    );

    let valid_statuses = ["covered", "partial", "gap", "fixture_required"];
    let phase_numbers = phases
        .iter()
        .filter_map(|phase| phase["phase"].as_u64())
        .collect::<Vec<_>>();
    for phase in 0..=8 {
        assert!(
            phase_numbers.contains(&phase),
            "migration phase status should include Phase {phase}"
        );
    }
    for phase in phases {
        let status = phase["status"]
            .as_str()
            .expect("migration phase status should be a string");
        assert!(
            valid_statuses.contains(&status),
            "invalid migration phase status {status}"
        );
        assert!(
            !phase["deliverables"]
                .as_array()
                .expect("migration phase deliverables should be an array")
                .is_empty(),
            "migration phase should list deliverables"
        );
        assert!(
            !phase["evidence_present"]
                .as_array()
                .expect("migration phase evidence_present should be an array")
                .is_empty(),
            "migration phase should cite source evidence"
        );
        assert!(
            phase["evidence_missing"]
                .as_array()
                .expect("migration phase evidence_missing should be an array")
                .is_empty(),
            "migration phase should not cite missing source evidence"
        );
    }

    let phase0 = phases
        .iter()
        .find(|phase| phase["phase"] == 0)
        .expect("Phase 0 should be present");
    assert_eq!(phase0["status"], "covered");
    let phase0_evidence = phase0["evidence_present"]
        .as_array()
        .expect("Phase 0 evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "docs/backend-engine-unification-plan.md",
        "scripts/generate_backend_capability_report.py",
        "docs/backend-capability-report.json",
    ] {
        assert!(
            phase0_evidence.contains(&path),
            "Phase 0 should cite {path}"
        );
    }

    let phase7 = phases
        .iter()
        .find(|phase| phase["phase"] == 7)
        .expect("Phase 7 should be present");
    assert_eq!(phase7["status"], "covered");
    let phase7_evidence = phase7["evidence_present"]
        .as_array()
        .expect("Phase 7 evidence should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for path in [
        "crates/kiln-model/src/backend/metal_attention.rs",
        "crates/kiln-model/src/backend/vulkan_tensor_bridge.rs",
        "crates/kiln-model/src/backend/cuda_rocm_common.rs",
    ] {
        assert!(
            phase7_evidence.contains(&path),
            "Phase 7 should cite backend decomposition evidence {path}"
        );
    }

    let phase8 = phases
        .iter()
        .find(|phase| phase["phase"] == 8)
        .expect("Phase 8 should be present");
    assert_eq!(phase8["status"], "fixture_required");
    let phase8_remaining = phase8["remaining"]
        .as_array()
        .expect("Phase 8 remaining should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        phase8_remaining
            .iter()
            .any(|item| item.contains("hardware_latency_thresholds")),
        "Phase 8 should keep the known-hardware latency artifact requirement visible"
    );

    let report_md = fs::read_to_string(root.join("docs/backend-capability-report.md"))
        .expect("capability report markdown should be readable");
    assert!(
        report_md.contains("## Migration Phase Status"),
        "Markdown report should expose migration phase status"
    );
    assert!(
        report_md.contains("| Phase 8 | Conformance and performance gates | `fixture_required` |"),
        "Markdown report should keep Phase 8 fixture requirement visible"
    );
}

#[test]
fn backend_engine_unification_plan_matches_current_training_status() {
    let root = workspace_root();
    let plan_source = fs::read_to_string(root.join("docs/backend-engine-unification-plan.md"))
        .expect("backend engine unification plan should be readable");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(root.join("docs/backend-capability-report.json"))
            .expect("capability report json should be readable"),
    )
    .expect("capability report json should parse");

    for stale in [
        "`is_cuda_device`",
        "`is_metal_device`",
        "no native fused FLCE",
        "host-stages down to CPU",
        "returns true while the current `linear_decode_argmax` override returns",
        "SGD is not currently overridden",
    ] {
        assert!(
            !plan_source.contains(stale),
            "backend unification plan should not keep stale implementation claim: {stale}"
        );
    }

    assert!(
        plan_source.contains("TrainingLossBackend::runtime_sft_flce_loss_route")
            && plan_source.contains("TrainingLossBackend::runtime_grpo_loss_route")
            && plan_source.contains("TrainingLossBackend::runtime_grpo_kl_auxiliary_route")
            && plan_source.contains("TrainingLossBackend::runtime_opd_loss_route")
            && plan_source.contains("TrainingLossBackend::runtime_opd_phase_b_backward_route")
            && plan_source.contains("TrainingLossBackend::runtime_final_rmsnorm_backward_route")
            && plan_source.contains("TrainingLossBackend::runtime_tape_forward_backward_route")
            && plan_source.contains("TrainingLossBackend::runtime_training_precision_policy"),
        "training source map should describe backend-owned loss and precision routing"
    );
    assert!(
        plan_source.contains("generated capability report"),
        "plan should point readers at the generated report for current completion status"
    );
    assert_eq!(
        report["training_loss_policy"]["rocm"]["sft_flce_loss_route"], "kt_tape_flce",
        "ROCm report should agree with the plan's kt-tape SFT FLCE route"
    );
    assert_eq!(
        report["optimizer_dispatch"]["metal"]["sgd_step"], "default_decline",
        "Metal optimizer status should agree with the plan's default-decline language"
    );
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

    let script_source =
        fs::read_to_string(&script_path).expect("capability report generator should be readable");
    for required in [
        "def load_toml_feature_subset(",
        "GITHUB_HEAD_REF",
        "def check_report_files(",
        "def run_self_test(",
        "fallback TOML feature self-test failed",
        "Markdown report is stale",
        "write_report_files(json_text, markdown_text)",
    ] {
        assert!(
            script_source.contains(required),
            "capability report generator should keep check-mode contract: {required}"
        );
    }

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
}

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

fn production_source_before_tests(source: &str) -> &str {
    source
        .split_once("#[cfg(test)]\nmod tests")
        .map(|(production, _)| production)
        .unwrap_or(source)
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

#[test]
fn matmul_transposed_ops_route_through_deviceop_contract() {
    let matmul_source =
        fs::read_to_string(workspace_root().join("crates/kiln-tensor/src/ops/matmul.rs"))
            .expect("matmul op source should be readable");
    let production_source = production_source_before_tests(&matmul_source);

    for required in [
        "pub struct MatmulLhsTransposedOp",
        "impl DeviceOp2 for MatmulLhsTransposedOp",
        "dispatch2(&MatmulLhsTransposedOp, a, b)",
        "pub struct MatmulRhsTransposedOp",
        "impl DeviceOp2 for MatmulRhsTransposedOp",
        "dispatch2(&MatmulRhsTransposedOp, a, b)",
    ] {
        assert!(
            production_source.contains(required),
            "transposed matmul should route through DeviceOp2 contract: missing {required}"
        );
    }

    for forbidden in [
        "Device::Cuda",
        "Device::Rocm",
        "Device::Metal",
        "Device::Vulkan",
        "match self.name()",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "production matmul op source should not choose a backend by identity: {forbidden}"
        );
    }
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
        "resident_activation_resource",
        "ResidentResource",
        "HashMap<TensorId",
    ] {
        assert!(
            common_source.contains(required),
            "cuda_rocm_common.rs should own shared resident metadata helper `{required}`"
        );
    }

    for backend_file in ["cuda.rs", "rocm.rs"] {
        let path = backend_dir.join(backend_file);
        let source = fs::read_to_string(&path).expect("backend source should be readable");
        let functions = parse_functions(&path);
        let register = functions.get("register_resource").unwrap_or_else(|| {
            panic!("{backend_file} missing ResidentRegistry::register_resource")
        });
        let update = functions
            .get("update_resource")
            .unwrap_or_else(|| panic!("{backend_file} missing ResidentRegistry::update_resource"));
        let evict = functions
            .get("evict_resource")
            .unwrap_or_else(|| panic!("{backend_file} missing ResidentRegistry::evict_resource"));
        let resident = functions.get("resident_resource").unwrap_or_else(|| {
            panic!("{backend_file} missing ResidentRegistry::resident_resource")
        });

        assert!(
            compact_body(&register.body).contains("cuda_rocm_common::mark_resident_activation("),
            "{backend_file} registry should register residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&update.body).contains("cuda_rocm_common::mark_resident_activation("),
            "{backend_file} registry should update residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&evict.body).contains("cuda_rocm_common::evict_resident_activation("),
            "{backend_file} registry should evict residency through cuda_rocm_common"
        );
        assert!(
            compact_body(&resident.body)
                .contains("cuda_rocm_common::resident_activation_resource("),
            "{backend_file} registry should query resident metadata through cuda_rocm_common"
        );

        assert!(
            !source.contains("with_cuda_resident_ids")
                && !source.contains("with_rocm_resident_ids")
                && !source.contains("fn kt_id("),
            "{backend_file} should not keep copied resident TensorId registry helpers"
        );
    }

    let metal_source = fs::read_to_string(backend_dir.join("metal_residency.rs"))
        .expect("metal_residency.rs should be readable");
    assert!(
        metal_source.contains("HashMap<TensorId, super::residency::ResidentResource>")
            && metal_source.contains("ResidentResourceState")
            && metal_source.contains("ReplayStability::StableWithinStep"),
        "Metal resident registry should persist resource lifecycle and replay metadata"
    );
    let vulkan_residency_source = fs::read_to_string(backend_dir.join("vulkan_residency.rs"))
        .expect("vulkan_residency.rs should be readable");
    let vulkan_source =
        fs::read_to_string(backend_dir.join("vulkan.rs")).expect("vulkan.rs should be readable");
    assert!(
        vulkan_residency_source.contains("ResidentActivationEntry")
            && vulkan_residency_source.contains("resource: super::residency::ResidentResource")
            && vulkan_source.contains("ReplayStability::StableAcrossReplay")
            && vulkan_source.contains("with_resident_allocation"),
        "Vulkan resident registry should persist actual allocation metadata with replay stability"
    );
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
            .get("runtime_dispatch_sgd_step")
            .unwrap_or_else(|| panic!("{backend_file} missing runtime_dispatch_sgd_step"));
        let adamw = functions
            .get("runtime_dispatch_adamw_step")
            .unwrap_or_else(|| panic!("{backend_file} missing runtime_dispatch_adamw_step"));

        let sgd_body = compact_body(&sgd.body);
        assert!(
            sgd_body.contains(&format!("{helper}(&self.resident_tensor_ids,"))
                && sgd_body.contains("&[param,grad]"),
            "{backend_file} SGD dispatch should use shared optimizer readiness validation"
        );
        let adamw_body = compact_body(&adamw.body);
        assert!(
            adamw_body.contains(&format!("{helper}(&self.resident_tensor_ids,"))
                && adamw_body.contains("&[param,grad,first_moment,second_moment]"),
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
            .get("runtime_linear_prefill_apply")
            .unwrap_or_else(|| panic!("{backend_file} missing runtime_linear_prefill_apply"));
        let offset = functions
            .get("runtime_linear_prefill_apply_offset")
            .unwrap_or_else(|| {
                panic!("{backend_file} missing runtime_linear_prefill_apply_offset")
            });
        let lora = functions
            .get("runtime_lora_delta_resident")
            .unwrap_or_else(|| panic!("{backend_file} missing runtime_lora_delta_resident"));

        assert!(
            compact_body(&linear.body).contains(&format!("{helper}(&[x,weight_t])")),
            "{backend_file} runtime_linear_prefill_apply should use shared kt device validation"
        );
        assert!(
            compact_body(&offset.body).contains(&format!("{helper}(&[x,full_weight_t])")),
            "{backend_file} runtime_linear_prefill_apply_offset should use shared kt device validation"
        );
        assert!(
            compact_body(&lora.body).contains(&format!("{helper}(&[x,a,b])")),
            "{backend_file} runtime_lora_delta_resident should use shared kt device validation"
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
    let cuda_source = fs::read_to_string(manifest_dir().join("src/backend/cuda.rs"))
        .expect("cuda backend source should be readable");
    assert!(
        cuda_source.contains("cuda_full_attn_qkv_in_proj_enabled")
            && cuda_source.contains("KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ"),
        "CUDA full-attention QKV gate should live behind the LinearBackend implementation"
    );
    for required in [
        "cuda_gdn_ab_in_proj_enabled",
        "cuda_gdn_prefill_ab_in_proj_enabled",
        "CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
        "KILN_DISABLE_CUDA_GDN_AB_IN_PROJ",
        "KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ",
    ] {
        assert!(
            cuda_source.contains(required),
            "CUDA GDN A/B in-projection gate should live behind the GdnBackend implementation `{required}`"
        );
    }
    let rocm_source = fs::read_to_string(manifest_dir().join("src/backend/rocm.rs"))
        .expect("rocm backend source should be readable");
    for required in [
        "rocm_full_attn_qkv_in_proj_enabled",
        "KILN_DISABLE_ROCM_FULL_ATTN_QKV_IN_PROJ",
        "KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ",
    ] {
        assert!(
            rocm_source.contains(required),
            "ROCm full-attention QKV gate should live behind the LinearBackend implementation and preserve legacy CUDA alias `{required}`"
        );
    }
    for required in [
        "rocm_gdn_ab_in_proj_enabled",
        "rocm_gdn_prefill_ab_in_proj_enabled",
        "ROCM_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
        "KILN_DISABLE_ROCM_GDN_AB_IN_PROJ",
        "KILN_DISABLE_ROCM_GDN_PREFILL_AB_IN_PROJ",
        "KILN_DISABLE_CUDA_GDN_AB_IN_PROJ",
        "KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ",
    ] {
        assert!(
            rocm_source.contains(required),
            "ROCm GDN A/B in-projection gate should live behind the GdnBackend implementation and preserve legacy CUDA aliases `{required}`"
        );
    }
    for stale_helper in [
        "fn cuda_gdn_ab_in_proj_enabled(",
        "fn cuda_gdn_prefill_ab_in_proj_enabled(",
        "fn cuda_rocm_disable_env_set_for_device(",
        "fn cuda_rocm_gdn_ab_in_proj_enabled(",
        "fn cuda_rocm_gdn_prefill_ab_in_proj_enabled(",
        "CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
        "CUDA_ROCM_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
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

    let shared_gdn_support_methods = [
        (
            "runtime_supports_gdn_forward_substitution",
            "supports_gdn_forward_substitution",
        ),
        (
            "runtime_supports_gdn_recurrent_step",
            "supports_gdn_recurrent_step",
        ),
        ("runtime_supports_gdn_chunk_prep", "supports_gdn_chunk_prep"),
        ("runtime_supports_gdn_chunk_scan", "supports_gdn_chunk_scan"),
        (
            "runtime_supports_gdn_full_chunk_forward",
            "supports_gdn_full_chunk_forward",
        ),
        (
            "runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk",
            "supports_gdn_decode_gates_recurrent_unexpanded_qk",
        ),
        (
            "runtime_supports_gdn_decode_qk_norm_gates_recurrent",
            "supports_gdn_decode_qk_norm_gates_recurrent",
        ),
        ("runtime_supports_gdn_gates", "supports_gdn_gates"),
        (
            "runtime_supports_gdn_gated_rms_norm",
            "supports_gdn_gated_rms_norm",
        ),
    ];
    let shared_attention_support_methods = [
        (
            "runtime_supports_flash_attn_prefill",
            "supports_flash_attn_prefill",
        ),
        (
            "runtime_supports_flash_attn_paged_decode",
            "supports_flash_attn_paged_decode",
        ),
        (
            "runtime_supports_strict_paged_decode_contiguous_batch",
            "supports_strict_paged_decode_contiguous_batch",
        ),
    ];
    let shared_conv_support_methods = [
        (
            "runtime_supports_causal_conv1d_update",
            "supports_causal_conv1d_update",
        ),
        (
            "runtime_supports_causal_conv1d_prefill",
            "supports_causal_conv1d_prefill",
        ),
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

        for (runtime_method, predicate_method) in shared_gdn_support_methods {
            let body = functions
                .get(runtime_method)
                .unwrap_or_else(|| panic!("{backend_file} missing {runtime_method}"));
            assert!(
                compact_body(&body.body)
                    .contains(&format!("support_predicates().{predicate_method}()")),
                "{backend_file} `{runtime_method}` should delegate to cuda_rocm_common"
            );
        }

        for (runtime_method, predicate_method) in shared_attention_support_methods {
            let body = functions
                .get(runtime_method)
                .unwrap_or_else(|| panic!("{backend_file} missing {runtime_method}"));
            assert!(
                compact_body(&body.body)
                    .contains(&format!("support_predicates().{predicate_method}()")),
                "{backend_file} `{runtime_method}` should delegate to cuda_rocm_common"
            );
        }

        for (runtime_method, predicate_method) in shared_conv_support_methods {
            let body = functions
                .get(runtime_method)
                .unwrap_or_else(|| panic!("{backend_file} missing {runtime_method}"));
            assert!(
                compact_body(&body.body)
                    .contains(&format!("support_predicates().{predicate_method}()")),
                "{backend_file} `{runtime_method}` should delegate to cuda_rocm_common"
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
        (
            "runtime_supports_gdn_forward_substitution",
            "supports_gdn_forward_substitution",
        ),
        (
            "runtime_supports_gdn_recurrent_step",
            "supports_gdn_recurrent_step",
        ),
        (
            "runtime_supports_gdn_recurrent_prefill_native_head_last",
            "supports_gdn_recurrent_prefill_native_head_last",
        ),
        (
            "runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last",
            "supports_gdn_recurrent_qk_norm_prefill_native_head_last",
        ),
        ("runtime_supports_gdn_chunk_prep", "supports_gdn_chunk_prep"),
        ("runtime_supports_gdn_chunk_scan", "supports_gdn_chunk_scan"),
        (
            "runtime_supports_gdn_full_chunk_forward",
            "supports_gdn_full_chunk_forward",
        ),
        ("runtime_supports_gdn_gates", "supports_gdn_gates"),
        (
            "runtime_supports_gdn_gated_rms_norm",
            "supports_gdn_gated_rms_norm",
        ),
        ("runtime_gdn_in_proj_decode", "gdn_in_proj_decode"),
        (
            "runtime_gdn_decode_gates_recurrent_rmsnorm",
            "gdn_decode_gates_recurrent_rmsnorm",
        ),
        (
            "runtime_gdn_forward_substitution",
            "gdn_forward_substitution",
        ),
        (
            "runtime_gdn_recurrent_prefill_native_head_last",
            "gdn_recurrent_prefill_native_head_last",
        ),
        (
            "runtime_gdn_recurrent_qk_norm_prefill_native_head_last",
            "gdn_recurrent_qk_norm_prefill_native_head_last",
        ),
        ("runtime_gdn_recurrent_step", "gdn_recurrent_step"),
        ("runtime_gdn_chunkwise_forward", "gdn_chunkwise_forward"),
        ("runtime_gdn_chunk_prep", "gdn_chunk_prep"),
        ("runtime_gdn_chunk_scan", "gdn_chunk_scan"),
        ("runtime_gdn_full_chunk_forward", "gdn_full_chunk_forward"),
        ("runtime_gdn_gates", "gdn_gates"),
        ("runtime_gdn_gated_rms_norm", "gdn_gated_rms_norm"),
    ];
    let mut failures = Vec::new();

    for (method, delegate) in delegated_methods {
        let Some(function) = functions.get(method) else {
            failures.push(format!("vulkan.rs is missing `{method}`"));
            continue;
        };
        let body = compact_body(&function.body);
        let delegation = format!("vulkan_gdn::{delegate}(");
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
fn vulkan_recurrent_scatter_stages_rows_before_mutating_destinations() {
    let backend_dir = manifest_dir().join("src/backend");
    let vulkan_rs = backend_dir.join("vulkan.rs");
    let functions = parse_functions(&vulkan_rs);
    let scatter = functions
        .get("runtime_scatter_gdn_recurrent_resident_batch_rows")
        .expect("VulkanBackend should expose recurrent resident scatter");
    let body = compact_body(&scatter.body);

    assert!(
        body.contains("ifrow_buffers.len()!=destinations.len(){returnOk(false);}"),
        "Vulkan recurrent scatter should reject short split results before mutating destinations"
    );
    let stage_pos = body
        .find("staged_rows.push((old_id,new_id,placeholder,row_buffer));")
        .expect("Vulkan recurrent scatter should stage destination replacements");
    let assignment_pos = body
        .find("**dst=placeholder;")
        .expect("Vulkan recurrent scatter should assign staged placeholders");
    let replacement_pos = body
        .find("replace_recurrent_state_resident_buffer(old_id,new_id,row_buffer);")
        .expect("Vulkan recurrent scatter should replace resident row buffers");
    assert!(
        stage_pos < assignment_pos && assignment_pos < replacement_pos,
        "Vulkan recurrent scatter should validate/stage every row before destination mutation"
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
        "TrainingAccelerationEnvFlagPolicy",
        "ReplayCapabilities",
        "ReplayAuthority",
        "BackendFallbackCapabilities",
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
    let training_acceleration_env_fields =
        capability_descriptors["TrainingAccelerationEnvFlagPolicy"]["fields"]
            .as_array()
            .expect("TrainingAccelerationEnvFlagPolicy fields should be an array")
            .iter()
            .filter_map(|field| field["name"].as_str())
            .collect::<Vec<_>>();
    for field in ["env", "default_on"] {
        assert!(
            training_acceleration_env_fields.contains(&field),
            "TrainingAccelerationEnvFlagPolicy should include {field}"
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
        GpuMemoryReclaimer::LoggedNoop {
            log_message: GpuMemoryReclaimPolicy::VULKAN_LOGGED_NOOP_MESSAGE,
        }
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
    assert_eq!(
        vulkan.linear.expect("linear env policy").env,
        "KILN_VULKAN_LINEAR"
    );
    assert_eq!(
        vulkan.sdpa.expect("sdpa env policy").env,
        "KILN_VULKAN_SDPA"
    );
    assert_eq!(
        vulkan
            .rmsnorm_inference
            .expect("rmsnorm inference env policy")
            .env,
        "KILN_VULKAN_RMSNORM"
    );

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
    let replay_trait = source_between(
        &backend_source,
        "pub trait ReplayBackend",
        "// (#1082 candle removal) The candle-typed `for_device` shim was deleted",
    );
    let replay_support_start = replay_trait
        .find("fn runtime_supports_replay_request")
        .expect("ReplayBackend runtime_supports_replay_request should be implemented");
    let replay_key_start = replay_trait
        .find("fn runtime_replay_key_for_request")
        .expect("ReplayBackend runtime_replay_key_for_request should follow support mapping");
    let replay_support = &replay_trait[replay_support_start..replay_key_start];
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
        ("StartupBackend", "runtime_precompile_startup_kernels"),
        ("ExternalYieldBackend", "runtime_synchronize_external_yield"),
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
        let expected_forwarding = if matches!(
            facet,
            "BackendIdentity"
                | "StartupBackend"
                | "ExternalYieldBackend"
                | "AttentionBackend"
                | "GdnBackend"
                | "ConvBackend"
                | "LinearBackend"
                | "ResidencyBackend"
                | "SamplingBackend"
                | "OptimizerBackend"
                | "PagedKvBackend"
                | "ReplayBackend"
                | "TrainingLossBackend"
        ) {
            "concrete_authoritative"
        } else {
            "blanket_backend_runtime"
        };
        assert_eq!(
            info["forwarding_impl"].as_str(),
            Some(expected_forwarding),
            "{facet} should report its current focused-trait implementation route"
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

    for facet in [
        "BackendIdentity",
        "StartupBackend",
        "ExternalYieldBackend",
        "AttentionBackend",
        "GdnBackend",
        "ConvBackend",
        "LinearBackend",
        "ResidencyBackend",
        "SamplingBackend",
        "OptimizerBackend",
        "PagedKvBackend",
        "ReplayBackend",
        "TrainingLossBackend",
    ] {
        let info = facets
            .get(facet)
            .unwrap_or_else(|| panic!("focused_backend_facets should list {facet}"));
        let concrete_impls = info["concrete_impls"]
            .as_array()
            .unwrap_or_else(|| panic!("{facet} concrete_impls should be an array"))
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        for backend in [
            "CpuBackend",
            "CudaBackend",
            "RocmBackend",
            "MetalBackend",
            "VulkanBackend",
        ] {
            assert!(
                concrete_impls.contains(&backend),
                "{facet} should be implemented directly by {backend}"
            );
        }
    }

    let backend_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend/mod.rs should be readable");
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> BackendIdentity for T"),
        "BackendIdentity should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> StartupBackend for T"),
        "StartupBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> ExternalYieldBackend for T"),
        "ExternalYieldBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> AttentionBackend for T"),
        "AttentionBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> GdnBackend for T"),
        "GdnBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> ConvBackend for T"),
        "ConvBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> LinearBackend for T"),
        "LinearBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> ResidencyBackend for T"),
        "ResidencyBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> SamplingBackend for T"),
        "SamplingBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> OptimizerBackend for T"),
        "OptimizerBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> PagedKvBackend for T"),
        "PagedKvBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> ReplayBackend for T"),
        "ReplayBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    assert!(
        !backend_source.contains("impl<T: BackendRuntime + ?Sized> TrainingLossBackend for T"),
        "TrainingLossBackend should not regress to a blanket BackendRuntime forwarding impl"
    );
    let runtime_trait_source = source_between(
        &backend_source,
        "pub trait BackendRuntime",
        "pub trait BackendIdentity",
    );
    for supertrait in [
        "BackendIdentity",
        "StartupBackend",
        "ExternalYieldBackend",
        "AttentionBackend",
        "GdnBackend",
        "ConvBackend",
        "LinearBackend",
        "ResidencyBackend",
        "SamplingBackend",
        "OptimizerBackend",
        "PagedKvBackend",
        "ReplayBackend",
        "TrainingLossBackend",
    ] {
        assert!(
            runtime_trait_source.contains(supertrait),
            "BackendRuntime should inherit {supertrait} from focused facets"
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
fn external_yield_synchronization_is_backend_owned_and_device_wide() {
    let root = workspace_root();
    let backend_dir = root.join("crates/kiln-model/src/backend");
    let read = |name: &str| {
        fs::read_to_string(backend_dir.join(name))
            .unwrap_or_else(|err| panic!("{name} should be readable: {err}"))
    };

    let cpu = read("cpu.rs");
    let cpu_impl = source_between(
        &cpu,
        "impl ExternalYieldBackend for CpuBackend",
        "impl AttentionBackend for CpuBackend",
    );
    assert!(
        compact_body(cpu_impl).contains("Ok(())"),
        "CPU external-yield synchronization should remain a synchronous no-op"
    );

    let cuda = read("cuda.rs");
    let cuda_impl = source_between(
        &cuda,
        "impl ExternalYieldBackend for CudaBackend",
        "impl AttentionBackend for CudaBackend",
    );
    let cuda_impl = compact_body(cuda_impl);
    assert!(
        cuda_impl.contains("kiln_tensor::primary_cuda_context(device_index)")
            && cuda_impl.contains("context.synchronize()"),
        "CUDA external yields must drain the complete CUDA context"
    );
    assert!(
        !cuda_impl.contains("cuda_synchronize_default_stream"),
        "CUDA external yields must not drain only the default stream"
    );

    let rocm = read("rocm.rs");
    let rocm_impl = source_between(
        &rocm,
        "impl ExternalYieldBackend for RocmBackend",
        "impl AttentionBackend for RocmBackend",
    );
    assert!(
        compact_body(rocm_impl)
            .contains("kiln_tensor::rocm_synchronize_default_stream(device_index)"),
        "ROCm external yields must use the device-wide hipDeviceSynchronize wrapper"
    );

    let metal = read("metal_runtime.rs");
    let metal_impl = source_between(
        &metal,
        "impl ExternalYieldBackend for MetalBackend",
        "impl ConvBackend for MetalBackend",
    );
    let metal_impl = compact_body(metal_impl);
    assert!(
        metal_impl.contains("kiln_tensor::primary_metal_companion(device_index)")
            && metal_impl.contains("companion.wait_until_completed()"),
        "Metal external yields must commit and drain the backend command queue"
    );

    let vulkan = read("vulkan.rs");
    let vulkan_impl = source_between(
        &vulkan,
        "impl ExternalYieldBackend for VulkanBackend",
        "impl ConvBackend for VulkanBackend",
    );
    let vulkan_impl = compact_body(vulkan_impl);
    assert!(
        vulkan_impl.contains("self.vulkan_device.as_ref()")
            && vulkan_impl.contains("kiln_tensor::Device::Vulkan(device_index)")
            && vulkan_impl.contains("kiln_tensor::vulkan_synchronize_queue(device_index)")
            && vulkan_impl.contains("device.synchronize_queue(\"externalmodelyield\")"),
        "Vulkan external yields must drain both the tensor companion queue and the logical device owned by VulkanBackend"
    );
    let companion_pos = vulkan_impl
        .find("kiln_tensor::vulkan_synchronize_queue(device_index)")
        .expect("Vulkan tensor companion synchronization should be present");
    let backend_pos = vulkan_impl
        .find("device.synchronize_queue(\"externalmodelyield\")")
        .expect("Vulkan backend synchronization should be present");
    assert!(
        companion_pos < backend_pos,
        "Vulkan external yields must settle tensor companion work before the backend-private queue"
    );

    let vulkan_device = fs::read_to_string(root.join("crates/kiln-vulkan-kernel/src/device.rs"))
        .expect("Vulkan device source should be readable");
    let owned_queue_sync = source_between(
        &vulkan_device,
        "pub fn synchronize_queue(&self, label: &str)",
        "fn terminally_lost_message",
    );
    for required in [
        "self.check_alive()?",
        "self.device.queue_wait_idle(self.queue)",
        "self.mark_terminally_lost()",
    ] {
        assert!(
            compact_body(owned_queue_sync).contains(&compact_body(required)),
            "owned Vulkan queue synchronization should preserve {required}"
        );
    }
}

#[test]
fn vulkan_decode_weight_prewarm_is_nondestructive() {
    let root = workspace_root();
    let backend_mod = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend facade source should be readable");
    let vulkan = fs::read_to_string(root.join("crates/kiln-model/src/backend/vulkan.rs"))
        .expect("Vulkan backend source should be readable");
    let weights = fs::read_to_string(root.join("crates/kiln-model/src/backend/vulkan_weights.rs"))
        .expect("Vulkan weight-cache source should be readable");
    let generate = fs::read_to_string(root.join("crates/kiln-model/src/generate.rs"))
        .expect("model runner source should be readable");
    let sources = format!("{backend_mod}\n{vulkan}\n{weights}\n{generate}");

    assert!(
        weights.contains("pub(super) fn prewarm_decode_weights(")
            && weights.contains("weights: &GpuWeights"),
        "Vulkan decode prewarm must borrow model weights immutably"
    );
    assert!(
        generate.contains("pub fn prewarm_backend_decode_weights(&self)")
            && generate.contains("LinearBackend::runtime_prewarm_decode_weights"),
        "ModelRunner prewarm must be a non-mutating focused-backend call"
    );
    for forbidden in [
        "runtime_drop_uploaded_bf16_weights",
        "drop_uploaded_bf16_weights",
        "dropped_bf16_weight_stub",
    ] {
        assert!(
            !sources.contains(forbidden),
            "authoritative serving/training weights must not be replaced after prewarm: {forbidden}"
        );
    }
}

#[test]
fn residency_facade_delegates_to_authoritative_registries() {
    let root = workspace_root();
    let backend_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/mod.rs"))
        .expect("backend/mod.rs should be readable");

    assert!(
        !backend_source.contains("impl<T> residency::ResidentRegistry for T"),
        "ResidentRegistry should not be a blanket adapter over ResidencyBackend"
    );

    let residency_trait = source_between(
        &backend_source,
        "pub trait ResidencyBackend:",
        "/// Focused `OptimizerBackend`",
    );

    for required in [
        "BackendIdentity + residency::ResidentRegistry",
        "residency::ResidentRegistry::register_resource",
        "residency::ResidentRegistry::update_resource",
        "residency::ResidentRegistry::evict_resource",
        "residency::ResidentRegistry::has_resident_resource",
        "residency::ResidentRegistry::resident_resource",
        "residency::ResidentRegistry::resolve_resource",
    ] {
        assert!(
            residency_trait.contains(required),
            "ResidencyBackend activation facade should delegate through {required}"
        );
    }

    for (backend_file, backend_type) in [
        ("cpu.rs", "CpuBackend"),
        ("cuda.rs", "CudaBackend"),
        ("rocm.rs", "RocmBackend"),
        ("metal_runtime.rs", "MetalBackend"),
        ("vulkan.rs", "VulkanBackend"),
    ] {
        let backend_source =
            fs::read_to_string(root.join(format!("crates/kiln-model/src/backend/{backend_file}")))
                .expect("backend source should be readable");
        assert!(
            backend_source.contains(&format!(
                "impl super::residency::ResidentRegistry for {backend_type}"
            )),
            "{backend_file} should implement ResidentRegistry directly"
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
    let lora_functions = parse_functions(&root.join("crates/kiln-model/src/lora_loader.rs"));

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
    let lora_loader_section = ["register_with_backend", "evict_from_backend"]
        .into_iter()
        .map(|name| {
            lora_functions
                .get(name)
                .unwrap_or_else(|| panic!("lora_loader.rs should define {name}"))
                .body
                .as_str()
        })
        .collect::<Vec<_>>()
        .join("\n");

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
    let opd_path = root.join("crates/kiln-train/src/opd.rs");
    let opd_source = fs::read_to_string(&opd_path).expect("opd.rs should be readable");
    let opd_functions = parse_functions(&opd_path);
    let generate_source = fs::read_to_string(root.join("crates/kiln-model/src/generate.rs"))
        .expect("generate.rs should be readable");
    let lora_source = fs::read_to_string(root.join("crates/kiln-model/src/lora_loader.rs"))
        .expect("lora_loader.rs should be readable");
    let speculative_source = fs::read_to_string(root.join("crates/kiln-model/src/speculative.rs"))
        .expect("speculative.rs should be readable");
    let sampling_source = fs::read_to_string(root.join("crates/kiln-model/src/sampling.rs"))
        .expect("sampling.rs should be readable");
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
    let server_state_source = fs::read_to_string(root.join("crates/kiln-server/src/state.rs"))
        .expect("kiln-server state.rs should be readable");
    let server_device_memory_path = root.join("crates/kiln-server/src/device_memory.rs");
    let server_kv_autoscaler_source =
        fs::read_to_string(root.join("crates/kiln-server/src/kv_autoscaler.rs"))
            .expect("kv_autoscaler.rs should be readable");
    let server_main_source = fs::read_to_string(root.join("crates/kiln-server/src/main.rs"))
        .expect("kiln-server main.rs should be readable");
    let server_batching_source =
        fs::read_to_string(root.join("crates/kiln-server/src/batching_engine.rs"))
            .expect("kiln-server batching_engine.rs should be readable");
    let server_completions_source =
        fs::read_to_string(root.join("crates/kiln-server/src/api/completions.rs"))
            .expect("kiln-server api/completions.rs should be readable");
    let server_bench_source = fs::read_to_string(root.join("crates/kiln-server/src/bench.rs"))
        .expect("kiln-server bench.rs should be readable");
    let server_training_queue_path = root.join("crates/kiln-server/src/training_queue.rs");
    let server_training_queue_source = fs::read_to_string(&server_training_queue_path)
        .expect("kiln-server training_queue.rs should be readable");
    let server_training_queue_functions = parse_functions(&server_training_queue_path);

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
        sampling_source.contains("pub fn try_topk_on_device("),
        "sampling should expose the shared device top-k helper for non-generation callers"
    );
    let entropy_kl_threshold_section = source_between(
        &trainer_source,
        "fn entropy_aware_kl_threshold_from_policy_log_probs(",
        "let plp_host:",
    );
    assert!(
        entropy_kl_threshold_section.contains("try_topk_on_device(&flat, idx + 1)"),
        "entropy-aware KL threshold should consume the shared device top-k helper"
    );
    for forbidden in [
        "matches!(flat.device(), Device::Cuda(_))",
        "matches!(flat.device(), Device::Rocm(_))",
        "cuda_topk_last_axis",
        "rocm_topk_last_axis",
    ] {
        assert!(
            !entropy_kl_threshold_section.contains(forbidden),
            "entropy-aware KL threshold should not branch locally on backend top-k dispatch: {forbidden}"
        );
    }
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
        forward_source.contains("gated_rms_norm_preserves_tape_residency")
            && capability_source.contains("gated_rms_norm_preserves_tape_residency"),
        "forward GDN gated RMSNorm active-tape routing should read GdnCapabilities"
    );
    let gated_rms_norm_section = source_between(
        &forward_source,
        "fn gated_rms_norm(",
        "fn gated_rms_norm_fallback(",
    );
    for forbidden in [
        "BackendIdentity::runtime_device(backend)",
        "kiln_tensor::Device::Vulkan(_)",
    ] {
        assert!(
            !gated_rms_norm_section.contains(forbidden),
            "gated_rms_norm should not branch locally on backend/device residency policy: {forbidden}"
        );
    }
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
        forward_source.contains("detached_chunked_prefill_supported")
            && capability_source.contains("detached_chunked_prefill"),
        "forward detached chunked prefill routing should read AttentionCapabilities"
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
        forward_source.contains("chunk_pre_permute_bf16")
            && capability_source.contains("chunk_pre_permute_bf16"),
        "forward GDN chunk pre-permute routing should read GdnCapabilities"
    );
    assert!(
        forward_source.contains("InferenceRecurrentStatePolicy")
            && capability_source.contains("inference_recurrent_state"),
        "forward inference recurrent-state dtype routing should read GdnCapabilities"
    );
    let inference_recurrent_dtype_section = source_between(
        &forward_source,
        "fn inference_recurrent_dtype(",
        "fn new_with_batch_and_recurrent_dtype(",
    );
    assert!(
        inference_recurrent_dtype_section.contains("InferenceRecurrentStatePolicy")
            && inference_recurrent_dtype_section.contains("policy.bf16")
            && inference_recurrent_dtype_section.contains("policy.f16"),
        "LinearAttentionState inference recurrent dtype should consume the backend policy"
    );
    for forbidden in [
        "backend_name == Some(\"vulkan\")",
        "KILN_DISABLE_CUDA_BF16_INFERENCE_STATE",
        "KILN_DISABLE_ROCM_BF16_INFERENCE_STATE",
        "KILN_DISABLE_VULKAN_BF16_INFERENCE_STATE",
    ] {
        assert!(
            !inference_recurrent_dtype_section.contains(forbidden),
            "LinearAttentionState inference recurrent dtype should not branch locally on backend/env policy: {forbidden}"
        );
    }
    assert!(
        generate_source.contains("new_with_batch_for_inference_runtime(")
            && generate_source.contains("self.backend.as_ref()"),
        "ModelRunner linear-state creation should pass the active backend to recurrent-state policy"
    );
    assert!(
        trainer_source.contains("new_with_batch_for_inference_runtime(")
            && !trainer_source.contains("Some(BackendIdentity::runtime_name(backend))"),
        "trainer adapter smoke linear-state creation should pass the active backend to recurrent-state policy"
    );
    let cached_transpose_policy_section = source_between(
        &forward_source,
        "fn cached_transpose_for_weight(",
        "fn dropped_weight_stub(",
    );
    assert!(
        cached_transpose_policy_section.contains("ProjectionLoadPolicy::for_model_loader_device")
            && cached_transpose_policy_section
                .contains("direct_transposed_upload_for_cached_weights"),
        "forward cached transpose upload should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "matches!(device, Device::Metal(_))",
        "crate::backend::vulkan_active()",
    ] {
        assert!(
            !cached_transpose_policy_section.contains(forbidden),
            "cached transpose upload should not keep local backend policy: {forbidden}"
        );
    }
    let projection_load_cache_section = source_between(
        &forward_source,
        "struct ProjectionLoadCache",
        "fn projection_tensors_for_load(",
    );
    assert!(
        projection_load_cache_section.contains("ProjectionLoadPolicy")
            && projection_load_cache_section.contains("for_model_loader_device")
            && projection_load_cache_section.contains("drop_projection_originals")
            && projection_load_cache_section.contains("drop_projection_transposes"),
        "projection load cache should be backed by ProjectionLoadPolicy"
    );
    for forbidden in [
        "KILN_DROP_PROJECTION_ORIGINALS",
        "KILN_KEEP_PROJECTION_ORIGINALS",
        "KILN_KEEP_PROJECTION_TRANSPOSES",
        "KILN_VK_NATIVE_TRAINING",
        "matches!(device, Device::Metal",
        "crate::backend::vulkan_active()",
    ] {
        assert!(
            !projection_load_cache_section.contains(forbidden),
            "projection load cache should not keep local backend/env policy: {forbidden}"
        );
    }
    let aux_load_policy_section = source_between(
        &forward_source,
        "fn aux_tensors_for_load_batch(",
        "/// Cache a transpose for repeated GEMMs.",
    );
    assert!(
        aux_load_policy_section.contains("ProjectionLoadPolicy::for_model_loader_device")
            && aux_load_policy_section.contains("parallel_auxiliary_weight_upload_enabled"),
        "auxiliary weight batch upload should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "matches!(device, Device::Metal",
        "KILN_DISABLE_PARALLEL_AUX_LOAD",
    ] {
        assert!(
            !aux_load_policy_section.contains(forbidden),
            "auxiliary weight batch upload should not keep local backend/env policy: {forbidden}"
        );
    }
    let embedding_load_policy_section = source_between(
        &forward_source,
        "pub fn from_model_weights(",
        "let lm_head_w8 =",
    );
    assert!(
        embedding_load_policy_section.contains("ProjectionLoadPolicy::for_model_loader_device")
            && embedding_load_policy_section
                .contains("stub_embedding_table_after_transposed_upload"),
        "embedding table upload/stub decision should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "stub_embed_tokens_after_upload(",
        "matches!(device, Device::Metal",
        "crate::backend::vulkan_active()",
    ] {
        assert!(
            !embedding_load_policy_section.contains(forbidden),
            "embedding table upload/stub decision should not keep local backend policy: {forbidden}"
        );
    }
    let full_qkv_concat_policy_section = source_between(
        &forward_source,
        "let qkv_proj_t = {",
        "// KILN_W4A16=1 opt-in: queue q_proj",
    );
    assert!(
        full_qkv_concat_policy_section
            .contains("projection_load_policy.cache_full_attention_qkv_transpose_concat"),
        "full-attention qkv_proj_t cache should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "cuda_or_rocm_device(*device)",
        "matches!(device, Device::Cuda",
        "matches!(device, Device::Rocm",
    ] {
        assert!(
            !full_qkv_concat_policy_section.contains(forbidden),
            "full-attention qkv_proj_t cache should not keep local backend policy: {forbidden}"
        );
    }
    let linear_ab_concat_policy_section = source_between(
        &forward_source,
        "let in_proj_ab_t = {",
        "let in_proj_qkvzab_w8 =",
    );
    assert!(
        linear_ab_concat_policy_section
            .contains("projection_load_policy.cache_linear_attention_ab_transpose_concat"),
        "linear-attention in_proj_ab_t cache should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "let mut should_cache",
        "matches!(device, Device::Cuda",
        "matches!(device, Device::Metal",
        "matches!(device, Device::Rocm",
    ] {
        assert!(
            !linear_ab_concat_policy_section.contains(forbidden),
            "linear-attention in_proj_ab_t cache should not keep local backend policy: {forbidden}"
        );
    }
    let mlp_gate_up_concat_policy_section = source_between(
        &forward_source,
        "let gate_up_proj_t = {",
        "let (gate_up_proj_w8, down_proj_w8) =",
    );
    assert!(
        mlp_gate_up_concat_policy_section
            .contains("projection_load_policy.cache_mlp_gate_up_transpose_concat"),
        "MLP gate/up transpose cache should consume ProjectionLoadPolicy"
    );
    for forbidden in [
        "cuda_or_rocm_device(*device)",
        "matches!(device, Device::Cuda",
        "matches!(device, Device::Rocm",
    ] {
        assert!(
            !mlp_gate_up_concat_policy_section.contains(forbidden),
            "MLP gate/up transpose cache should not keep local backend policy: {forbidden}"
        );
    }
    let projection_w8_pack_policy_section = source_between(
        &forward_source,
        "pub fn from_model_weights(",
        "if w4a16_enabled && !marlin_pack_inputs.is_empty()",
    );
    assert!(
        projection_w8_pack_policy_section
            .contains("projection_load_policy.pack_w8a16_projection_rows"),
        "W8A16 projection row packing should consume ProjectionLoadPolicy"
    );
    assert!(
        !projection_w8_pack_policy_section.contains("matches!(*device, Device::Rocm"),
        "W8A16 projection row packing should not keep local ROCm device checks"
    );
    let gdn_chunk_pre_permute_policy_section = source_between(
        &forward_source,
        "let pre_permute_chunks =",
        "let pre_permuted:",
    );
    assert!(
        gdn_chunk_pre_permute_policy_section
            .contains("BackendCapabilityQueries::backend_capabilities(backend)")
            && gdn_chunk_pre_permute_policy_section.contains("chunk_pre_permute_bf16"),
        "GDN chunk pre-permute decision should consume GdnCapabilities"
    );
    for forbidden in [
        "cfg!(feature = \"cuda\")",
        "matches!(device, Device::Cuda",
        "matches!(*device, Device::Cuda",
    ] {
        assert!(
            !gdn_chunk_pre_permute_policy_section.contains(forbidden),
            "GDN chunk pre-permute decision should not keep local CUDA policy: {forbidden}"
        );
    }
    assert!(
        forward_source.contains("ResidencyBackend"),
        "forward GDN recurrent residency helpers should import the focused residency facet"
    );
    assert!(
        forward_source.contains("SamplingBackend"),
        "forward lm-head decode helpers should import the focused sampling facet"
    );

    let decode_policy_section = source_between(
        &capability_source,
        "pub(crate) fn decode_hot_path_fallback_policy_for_backend(",
        "pub(crate) fn decode_hot_path_generic_fallback_enabled_for_backend(",
    );
    assert!(
        decode_policy_section.contains("BackendCapabilities::from_backend(backend)"),
        "decode fallback policy should come from the shared backend capability aggregate"
    );
    assert!(
        !generate_source.contains("fn decode_hot_path_fallback_policy(")
            && !generate_source.contains("fn decode_batch_generic_fallback_enabled(")
            && !forward_source.contains("fn decode_batch_generic_fallback_enabled("),
        "forward/generate should not keep duplicate decode hot-path fallback helpers"
    );
    assert!(
        !decode_policy_section.contains("match device"),
        "decode fallback policy should not branch directly on device kind"
    );
    assert!(
        decode_policy_section.contains("decode_hot_path_debug_fallback_enabled()"),
        "decode fallback policy should use the shared BackendFallbackCapabilities debug opt-in"
    );
    let decode_buffer_max_batch_section = source_between(
        &generate_source,
        "fn decode_buffer_max_batch(",
        "enum PrefillSampleSource",
    );
    assert!(
        decode_buffer_max_batch_section
            .contains("BackendCapabilityQueries::backend_capabilities(backend)")
            && decode_buffer_max_batch_section.contains(".decode_batcher")
            && decode_buffer_max_batch_section.contains(".max_batch")
            && !decode_buffer_max_batch_section.contains("std::env")
            && !decode_buffer_max_batch_section.contains("KILN_DECODE_BUFFER_MAX_BATCH"),
        "decode buffer max-batch should use injected/backend policy without a model-local environment override"
    );
    assert!(
        !decode_buffer_max_batch_section.contains("DecodeBatcherPolicy::for_backend")
            && !decode_buffer_max_batch_section.contains("backend_name"),
        "decode buffer max-batch default should not rebuild policy from a backend name"
    );
    assert!(
        generate_source.contains(
            "decode_buffer_max_batch(selected_backend.as_ref(), options.max_decode_batch)",
        ) && generate_source.contains("self.decode_buffer_max_batch"),
        "decode buffer config should resolve the active backend under the injected hard ceiling once at construction"
    );
    let decode_debug_policy_section = source_between(
        &capability_source,
        "pub(crate) fn decode_hot_path_debug_fallback_enabled_for_backend(",
        "pub(crate) fn decode_hot_path_debug_fallback_env_for_backend(",
    );
    assert!(
        !decode_debug_policy_section.contains("\"metal\"")
            && !decode_debug_policy_section.contains("\"vulkan\"")
            && !decode_debug_policy_section.contains("\"rocm\"")
            && !decode_debug_policy_section.contains("match backend_name"),
        "decode debug fallback opt-in should not keep a local backend-name env table"
    );
    let decode_debug_env_section = source_between(
        &capability_source,
        "pub(crate) fn decode_hot_path_debug_fallback_env_for_backend(",
        "pub(crate) fn decode_hot_path_fallback_policy_for_backend(",
    );
    assert!(
        decode_debug_env_section.contains("decode_hot_path_debug_env"),
        "decode debug fallback env name should be read from BackendFallbackCapabilities"
    );

    let decode_batcher_config_section = source_between(
        &generate_source,
        "pub struct DecodeBatcherConfig {",
        "fn env_flag_value(",
    );
    assert!(
        decode_batcher_config_section.contains("pub max_batch: usize")
            && decode_batcher_config_section.contains("pub wait: std::time::Duration")
            && decode_batcher_config_section.contains("pub allow_mixed_seq_lens: bool"),
        "decode batcher should expose only its injected execution values"
    );
    for removed_constructor in [
        "pub fn from_env() -> Self",
        "from_env_for_policy(",
        "from_env_for_policy_with_max_batch(",
        "from_env_for_backend_kt(",
        "from_env_for_device_kt(",
        "enabled_for_device_kt(",
    ] {
        assert!(
            !generate_source.contains(removed_constructor),
            "decode batcher execution config must not retain runtime environment constructor {removed_constructor}"
        );
    }
    for legacy_env in [
        "KILN_DECODE_BATCHER",
        "KILN_DECODE_BATCH_MAX",
        "KILN_DECODE_BATCH_WAIT_US",
        "KILN_DECODE_BATCH_MIXED_SEQ",
    ] {
        let quoted = format!("\"{legacy_env}\"");
        assert!(
            !generate_source.contains(&quoted),
            "kiln-model generate must receive typed decode-batcher values instead of reading {legacy_env}"
        );
        assert!(
            !server_state_source.contains(&quoted),
            "kiln-server state must receive centralized typed configuration instead of reading {legacy_env}"
        );
    }
    assert!(
        capability_source.contains("pub rendezvous_default_enabled: bool"),
        "DecodeBatcherPolicy should own the direct-rendezvous enable default"
    );
    assert!(
        server_state_source.contains("runner.backend_capabilities()")
            && server_state_source.contains("decode_batcher_policy")
            && server_state_source.contains("decode_batcher_config")
            && server_state_source.contains("DecodeBatcher::spawn")
            && server_state_source.contains("warm_resident_decode_pool_on_startup")
            && server_state_source.contains("batching_engine_default_enabled")
            && server_state_source.contains("kv_cache_device_memory_pressure"),
        "kiln-server startup should resolve backend policy once and inject decode-batcher execution config"
    );
    let server_prefix_cache_state_section = source_between(
        &server_state_source,
        "fn linear_attention_state_bytes(",
        "fn default_prefix_cache_max_entries(",
    );
    assert!(
        server_state_source.contains("runner.backend_capabilities().gdn.inference_recurrent_state")
            && server_prefix_cache_state_section.contains("InferenceRecurrentStatePolicy")
            && server_prefix_cache_state_section.contains("policy.supports_dtype"),
        "kiln-server prefix-cache state sizing should consume the backend-owned inference recurrent-state policy"
    );
    let server_kv_sizing_reserve_section = source_between(
        &server_state_source,
        "let mut sizing_residency_bytes = post_load_used_vram.max(estimated_model_bytes);",
        "if post_load_used_vram > 0 {",
    );
    assert!(
        server_state_source.contains("runner.backend_capabilities().storage")
            && server_kv_sizing_reserve_section.contains("kv_sizing_residency_model_multiplier"),
        "kiln-server KV sizing residency reserve should consume StorageCapabilities"
    );
    let server_kv_auto_sizing_section = source_between(
        &server_state_source,
        "let compute_blocks_for_fraction = |fraction: f64| -> usize {",
        "let fp8_enabled = {",
    );
    assert!(
        server_state_source.contains("kv_auto_block_policy")
            && server_kv_auto_sizing_section.contains("kv_auto_block_policy")
            && server_kv_auto_sizing_section.contains("allow_min_blocks_below_live_budget"),
        "kiln-server KV auto-sizing caps should consume StorageCapabilities.kv_auto_block_policy"
    );
    for forbidden in [
        "let is_metal",
        "let is_rocm",
        "is_metal,",
        "is_rocm,",
        "matches!(device_kt, kiln_tensor::Device::Rocm(_))",
    ] {
        assert!(
            !server_kv_auto_sizing_section.contains(forbidden),
            "kiln-server KV auto-sizing should not keep a local backend/device cap table: {forbidden}"
        );
    }
    let server_kv_auto_sizing_helpers = source_between(
        &server_state_source,
        "fn auto_num_blocks_for_fraction(",
        "fn runtime_used_vram_for_policy(",
    );
    assert!(
        server_kv_auto_sizing_helpers.contains("KvCacheAutoBlockPolicy")
            && server_kv_auto_sizing_helpers.contains("runtime_cap_blocks")
            && server_kv_auto_sizing_helpers.contains("allow_min_blocks_below_live_budget"),
        "kiln-server KV auto-sizing helpers should be parameterized by the backend-owned policy"
    );
    for forbidden in [
        "is_metal: bool",
        "is_rocm: bool",
        "if is_metal",
        "else if is_rocm",
        "metal_auto_max_kv_blocks",
        "ROCM_AUTO_MAX_KV_BLOCKS",
        "METAL_AUTO_MAX_KV_BLOCKS",
        "matches!(device, kiln_tensor::Device::Rocm(_))",
    ] {
        assert!(
            !server_kv_auto_sizing_helpers.contains(forbidden),
            "kiln-server KV auto-sizing helpers should not keep a local backend/device cap table: {forbidden}"
        );
    }
    let server_kv_fp8_section = source_between(
        &server_state_source,
        "let fp8_enabled = {",
        "// Allocation closure: try to build the paged KV cache for `n` blocks.",
    );
    assert!(
        server_kv_fp8_section.contains("kv_cache_fp8_policy")
            && server_kv_fp8_section.contains("policy.enabled(requested)")
            && server_kv_fp8_section.contains("explicit_enable_env")
            && server_kv_fp8_section.contains("disabled_reason"),
        "kiln-server KV FP8 routing should consume StorageCapabilities.kv_cache_fp8_policy"
    );
    for forbidden in [
        "KILN_ALLOW_FP8_ON_METAL",
        "is_metal_device",
        "Device::Metal",
        "Backend::Metal",
        "metal_override",
        "on Metal",
    ] {
        assert!(
            !server_kv_fp8_section.contains(forbidden),
            "kiln-server KV FP8 routing should not keep a local backend/env policy table: {forbidden}"
        );
    }
    let server_gpu_memory_capacity_section = source_between(
        &server_state_source,
        "pub fn ensure_accelerator_memory_capacity(",
        "pub fn ensure_accelerator_memory_floor(",
    );
    assert!(
        server_state_source
            .contains("resolve_vram_capacity(physical_vram, memory_cfg.gpu_memory_gb)")
            && server_gpu_memory_capacity_section.contains("capacity.total_bytes")
            && server_gpu_memory_capacity_section.contains("cap-only"),
        "kiln-server GPU memory capacity should come from the typed physical-probe/configured-cap resolution"
    );
    for forbidden in [
        "Device::Cuda",
        "Device::Metal",
        "Backend::Cuda",
        "Backend::Metal",
        "24 * 1024 * 1024 * 1024",
        "16 * 1024 * 1024 * 1024",
        "assuming 24GB",
        "assuming 16GB",
    ] {
        assert!(
            !server_gpu_memory_capacity_section.contains(forbidden),
            "kiln-server GPU memory detection should not keep backend fallback policy locally: {forbidden}"
        );
    }
    let server_memory_snapshot_section = source_between(
        &server_state_source,
        "let snap = if",
        "let total_vram = vram_info.total_bytes;",
    );
    assert!(
        server_memory_snapshot_section.contains("gpu_memory_budget_policy")
            && server_memory_snapshot_section.contains("use_live_memory_snapshot"),
        "kiln-server live memory snapshot routing should consume StorageCapabilities.gpu_memory_budget_policy"
    );
    for forbidden in ["device_kt.backend()", "Backend::Cpu", "governor_backend"] {
        assert!(
            !server_memory_snapshot_section.contains(forbidden),
            "kiln-server live memory snapshot routing should not keep a local backend table: {forbidden}"
        );
    }
    let server_live_budget_section = source_between(
        &server_state_source,
        "let compute_blocks_for_fraction = |fraction: f64| -> usize {",
        "let fp8_enabled = {",
    );
    assert!(
        server_live_budget_section.contains("gpu_memory_budget_policy")
            && server_live_budget_section.contains("cap_kv_blocks_by_live_budget"),
        "kiln-server KV live-budget caps should consume StorageCapabilities.gpu_memory_budget_policy"
    );
    for forbidden in ["device_kt.backend()", "Backend::Cpu", "governor_backend"] {
        assert!(
            !server_live_budget_section.contains(forbidden),
            "kiln-server KV live-budget caps should not keep local backend policy: {forbidden}"
        );
    }
    assert!(
        server_state_source.contains("gpu_allocator_memory_probe_policy")
            && server_live_budget_section.contains("gpu_allocator_memory_probe_policy")
            && server_live_budget_section.contains("allocator_kv_budget_bytes_for_fraction("),
        "kiln-server KV live-budget caps should consume StorageCapabilities.gpu_allocator_memory_probe_policy"
    );
    let server_allocator_snapshot_section = source_between(
        &server_state_source,
        "let allocator_memory_snapshot = crate::device_memory::allocator_memory_snapshot(",
        "if let Some(allocator) = allocator_memory_snapshot {",
    );
    assert!(
        server_allocator_snapshot_section.contains("gpu_allocator_memory_probe_policy"),
        "kiln-server allocator memory snapshot logging should consume the backend-owned allocator probe policy"
    );
    let server_allocator_probe_functions = parse_functions(&server_device_memory_path);
    let allocator_memory_snapshot = server_allocator_probe_functions
        .get("allocator_memory_snapshot")
        .expect("device_memory should define allocator_memory_snapshot");
    let allocator_memory_snapshot_body = body_without_comments(&allocator_memory_snapshot.body);
    assert!(
        allocator_memory_snapshot_body.contains("policy.probe")
            && allocator_memory_snapshot_body.contains("GpuAllocatorMemoryProbe::CudaMemGetInfo")
            && allocator_memory_snapshot_body.contains("GpuAllocatorMemoryProbe::RocmMemGetInfo"),
        "device_memory allocator snapshot should dispatch through GpuAllocatorMemoryProbe"
    );
    assert!(
        !allocator_memory_snapshot_body.contains("match *device"),
        "device_memory allocator snapshot should not choose allocator probe policy from Device identity"
    );
    let compact_server_kv_autoscaler_source = compact_body(&server_kv_autoscaler_source);
    assert!(
        server_kv_autoscaler_source.contains("GpuAllocatorMemoryProbePolicy")
            && server_kv_autoscaler_source.contains("gpu_allocator_memory_probe_policy")
            && compact_server_kv_autoscaler_source
                .contains("live_resize_memory_snapshot(gpu_allocator_memory_probe_policy,"),
        "KV autoscaler should receive and consume the backend-owned allocator probe policy"
    );
    for forbidden in [
        "GpuAllocatorMemoryProbePolicy::for_backend",
        "Device::Cuda",
        "Device::Rocm",
        "Backend::Cuda",
        "Backend::Rocm",
        "device.backend()",
    ] {
        assert!(
            !server_kv_autoscaler_source.contains(forbidden),
            "KV autoscaler should not choose allocator probe policy locally: {forbidden}"
        );
    }
    let server_allocation_retry_section = source_between(
        &server_state_source,
        "let allocate_cache = |n: usize| -> anyhow::Result<PagedKvCacheKt> {",
        "// Determine num_blocks + paged cache:",
    );
    assert!(
        server_allocation_retry_section.contains("gpu_memory_budget_policy")
            && server_allocation_retry_section.contains("retry_kv_allocation_after_reclaim"),
        "kiln-server KV allocation retry should consume StorageCapabilities.gpu_memory_budget_policy"
    );
    assert!(
        server_allocation_retry_section.contains("gpu_allocator_memory_probe_policy")
            && server_allocation_retry_section
                .contains("validate_kv_allocation_against_live_allocator("),
        "kiln-server KV allocation validation should consume StorageCapabilities.gpu_allocator_memory_probe_policy"
    );
    for forbidden in ["device_kt.backend()", "Backend::Cpu", "governor_backend"] {
        assert!(
            !server_allocation_retry_section.contains(forbidden),
            "kiln-server KV allocation retry should not keep local backend policy: {forbidden}"
        );
    }
    let server_memory_reclaim_call_site_section = source_between(
        &server_state_source,
        "static GOVERNOR_WIRED: std::sync::OnceLock<()>",
        "let kv_autoscaler = if",
    );
    let compact_server_memory_reclaim_call_site_section =
        compact_body(server_memory_reclaim_call_site_section);
    assert!(
        server_memory_reclaim_call_site_section.contains("gpu_memory_reclaim_policy")
            && compact_server_memory_reclaim_call_site_section.contains(
                "register_backend_memory_reclaimer(gpu_memory_reclaim_policy,device_kt,gpu_lock.clone(),backend_health.clone(),batching_engine.clone(),)"
            ),
        "kiln-server memory-governor startup should consume the backend reclaim policy and the shared GPU/actor coordination surfaces"
    );
    let batching_engine_start = server_state_source
        .find("let batching_engine =")
        .expect("kiln-server state should construct the batching engine");
    let governor_wiring_start = server_state_source
        .find("static GOVERNOR_WIRED: std::sync::OnceLock<()>")
        .expect("kiln-server state should wire the memory governor");
    assert!(
        governor_wiring_start > batching_engine_start,
        "automatic allocator reclaim must be wired only after batching actor coordination exists"
    );
    for forbidden in [
        "Device::Cuda",
        "Device::Rocm",
        "Device::Metal",
        "Device::Vulkan",
        "matches!(device_kt",
        "cuda_set_pool_release_threshold",
        "cuda_trim_pool",
        "rocm_trim_pool",
        "metal reclaimer",
        "vulkan reclaimer",
    ] {
        assert!(
            !server_memory_reclaim_call_site_section.contains(forbidden),
            "kiln-server memory-governor startup should not keep backend reclaimer policy locally: {forbidden}"
        );
    }
    let server_memory_reclaim_helper_section = source_between(
        &server_state_source,
        "fn register_backend_memory_reclaimer(",
        "/// Auto-size the KV cache by trying",
    );
    assert!(
        server_memory_reclaim_helper_section.contains("GpuMemoryReclaimPolicy")
            && server_memory_reclaim_helper_section.contains("GpuMemoryReclaimer::CudaTrimPool")
            && server_memory_reclaim_helper_section.contains("GpuMemoryReclaimer::RocmTrimPool")
            && server_memory_reclaim_helper_section.contains("GpuMemoryReclaimer::LoggedNoop"),
        "kiln-server memory-governor helper should dispatch through the backend-owned reclaim policy"
    );
    for forbidden in [
        "Device::Metal",
        "Device::Vulkan",
        "matches!(device",
        "metal reclaimer",
        "vulkan reclaimer",
    ] {
        assert!(
            !server_memory_reclaim_helper_section.contains(forbidden),
            "kiln-server memory-governor helper should not keep Metal/Vulkan no-op policy locally: {forbidden}"
        );
    }
    let server_training_acceleration_profile_call_section = source_between(
        &server_state_source,
        "let backend_name = runner.backend_name();",
        "let prefix_cache_max_blocks = if prefix_cache_cfg.enabled {",
    );
    assert!(
        server_training_acceleration_profile_call_section
            .contains("backend_capabilities.training.acceleration_profile")
            && server_training_acceleration_profile_call_section
                .contains("log_backend_training_acceleration_profile("),
        "kiln-server training acceleration startup profile should consume BackendTrainingCapabilities"
    );
    for forbidden in [
        "VramSource::LinuxDrmSysfs",
        "VramSource::LinuxDrmSysfsUnified",
        "KILN_VULKAN_LINEAR",
        "KILN_VULKAN_SDPA",
        "KILN_VULKAN_RMSNORM",
        "env_flag = |",
    ] {
        assert!(
            !server_training_acceleration_profile_call_section.contains(forbidden),
            "kiln-server training acceleration startup profile should not keep local Vulkan/DRM policy: {forbidden}"
        );
    }
    let server_training_acceleration_profile_helper_section = source_between(
        &server_state_source,
        "fn training_acceleration_env_flag_status(",
        "fn linear_attention_state_bytes(",
    );
    assert!(
        server_training_acceleration_profile_helper_section
            .contains("TrainingAccelerationProfilePolicy")
            && server_training_acceleration_profile_helper_section
                .contains("TrainingAccelerationProfileLogMessage::Vulkan")
            && server_training_acceleration_profile_helper_section.contains("policy.env")
            && server_training_acceleration_profile_helper_section.contains("policy.default_on"),
        "kiln-server training acceleration profile helper should format the backend-owned policy"
    );
    for forbidden in [
        "KILN_VULKAN_LINEAR",
        "KILN_VULKAN_SDPA",
        "KILN_VULKAN_RMSNORM",
        "VramSource::LinuxDrmSysfs",
    ] {
        assert!(
            !server_training_acceleration_profile_helper_section.contains(forbidden),
            "kiln-server training acceleration profile helper should not own backend env/source tables: {forbidden}"
        );
    }
    assert!(
        !server_kv_sizing_reserve_section
            .contains("matches!(device_kt, kiln_tensor::Device::Vulkan(_))"),
        "kiln-server KV sizing residency reserve should not branch locally on Vulkan device identity"
    );
    for forbidden in [
        "KILN_DISABLE_CUDA_BF16_INFERENCE_STATE",
        "KILN_DISABLE_ROCM_BF16_INFERENCE_STATE",
        "KILN_DISABLE_VULKAN_BF16_INFERENCE_STATE",
        "match device.backend()",
        "compact_recurrent_state",
    ] {
        assert!(
            !server_prefix_cache_state_section.contains(forbidden),
            "kiln-server prefix-cache state sizing should not keep a local backend/env policy table: {forbidden}"
        );
    }
    let server_startup_policy_section = source_between(
        &server_state_source,
        "let backend_name = runner.backend_name();",
        "let decode_batcher = if let Some(config) = decode_batcher_config {",
    );
    for forbidden in [
        "backend_name == \"vulkan\"",
        "backend_name == \"metal\"",
        "Some(kiln_tensor::Device::Rocm(_))",
        "Some(kiln_tensor::Device::Cuda(_))",
    ] {
        assert!(
            !server_startup_policy_section.contains(forbidden),
            "kiln-server startup should not branch locally on backend/device policy: {forbidden}"
        );
    }
    assert!(
        server_state_source.contains("require_inference_prewarm_for_health"),
        "kiln-server health prewarm readiness should consume StartupCapabilities"
    );
    assert!(
        !server_state_source.contains("fn device_needs_inference_prewarm("),
        "kiln-server should not keep a local backend/device table for prewarm readiness"
    );
    let server_prewarm_policy_section = source_between(
        &server_main_source,
        "fn spawn_backend_prewarm(",
        "tokio::spawn(async move {",
    );
    assert!(
        server_prewarm_policy_section.contains("runner_guard.backend_capabilities().startup")
            && server_prewarm_policy_section.contains("run_inference_prewarm")
            && server_prewarm_policy_section.contains("decode_weight_prewarm_when_native_training")
            && server_prewarm_policy_section.contains("native_training_enabled_for_startup"),
        "kiln-server inference prewarm routing should consume StartupCapabilities"
    );
    assert!(
        server_main_source.contains("precompile_backend_startup_kernels")
            && !server_main_source.contains("fn precompile_metal_custom_kernels")
            && !server_main_source.contains("fn precompile_vulkan_custom_kernels")
            && !server_main_source
                .contains("kiln_model::backend::metal::precompile_custom_kernels")
            && !server_main_source
                .contains("kiln_model::backend::vulkan::precompile_custom_kernels"),
        "kiln-server startup custom-kernel precompile should route through StartupBackend"
    );
    for forbidden in [
        "backend_name() == \"vulkan\"",
        "Device::Metal",
        "Device::Rocm",
        "is_vulkan",
        "is_metal",
        "is_rocm",
    ] {
        assert!(
            !server_prewarm_policy_section.contains(forbidden),
            "kiln-server prewarm routing should not branch locally on backend/device policy: {forbidden}"
        );
    }
    let server_training_dispatch_section = &server_training_queue_functions
        .get("execute_job")
        .expect("kiln-server training_queue.rs should define execute_job")
        .body;
    assert!(
        server_training_dispatch_section
            .contains("backend_capabilities().training.server_dispatch")
            && server_training_dispatch_section.contains("native_route_enabled")
            && server_training_queue_source.contains("ServerTrainingDispatchPolicy")
            && server_training_queue_source.contains("ServerTrainingNativeRoute"),
        "kiln-server SFT/GRPO dispatch should consume BackendTrainingCapabilities server policy"
    );
    assert!(
        !server_training_dispatch_section.contains("KILN_CUDA_NATIVE_TRAINING")
            && !server_training_dispatch_section
                .contains("native_training_env_enabled(\"KILN_CUDA_NATIVE_TRAINING\")"),
        "kiln-server SFT/GRPO dispatch should not read the legacy CUDA native-training env locally"
    );
    let bench_training_dispatch_section = source_between(
        &server_bench_source,
        "fn bench_training(",
        "/// Render one `key  value [unit]` line at the indent used by the summary.",
    );
    assert!(
        bench_training_dispatch_section.contains("ServerTrainingDispatchPolicy")
            && bench_training_dispatch_section.contains("native_route_enabled")
            && server_bench_source.contains("backend_capabilities.training.server_dispatch"),
        "kiln-bench SFT dispatch should consume BackendTrainingCapabilities server policy"
    );
    assert!(
        !bench_training_dispatch_section.contains("std::env::var(\"KILN_CUDA_NATIVE_TRAINING\")")
            && !bench_training_dispatch_section.contains("let cuda_native"),
        "kiln-bench SFT dispatch should not read the legacy CUDA native-training env locally"
    );
    assert!(
        server_batching_source.contains("resolve_decode_runtime_config")
            && server_batching_source.contains("BatchingBackendPolicy")
            && server_batching_source.contains("BatchingActorAdmissionConfig")
            && server_state_source.contains("batching_config.resolve(")
            && server_state_source.contains("batching_runtime_config.actor_admission_config()")
            && server_state_source.contains("batching_engine_default_enabled")
            && server_state_source.contains("use_decode_width_prefill_admission")
            && server_state_source.contains("burst_prefill_admission"),
        "kiln-server batching startup should resolve DecodeBatcherPolicy once and project narrow actor admission settings"
    );
    for legacy_name in [
        "KILN_BATCHING_ENGINE",
        "KILN_BATCH_DECODE_ROWWISE",
        "KILN_BATCH_PREFIX_AWARE_ADMISSION",
        "KILN_BATCH_PREFILL_ADMISSION_QUANTUM",
    ] {
        let direct_read = format!("std::env::var(\"{legacy_name}\")");
        assert!(
            !server_batching_source.contains(&direct_read)
                && !server_state_source.contains(&direct_read),
            "production batching must not reread legacy environment control {legacy_name}"
        );
    }
    for forbidden in [
        "env_max_decode_batch_for_backend",
        "env_prefill_admission_quantum_for_backend",
        "Some(\"vulkan\")",
        "Some(\"cuda\")",
        "Some(\"metal\")",
    ] {
        assert!(
            !server_batching_source.contains(forbidden),
            "kiln-server batching engine should not keep backend-name default tables: {forbidden}"
        );
    }
    let decode_batcher_retry_section = source_between(
        &generate_source,
        "fn decode_batcher_rowwise_retry_enabled(",
        "fn greedy_token_decode_enabled(",
    );
    assert!(
        decode_batcher_retry_section.contains("BackendCapabilityQueries::backend_capabilities")
            && decode_batcher_retry_section
                .contains("decode_hot_path_fallback_policy_for_backend(backend)"),
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
    assert!(
        direct_paged_decode_attention_helper.contains("fn native_decode_attention_required(")
            && direct_paged_decode_attention_helper.contains("require_native_decode_attention")
            && forward_source
                .contains("decode_hot_path_debug_fallback_enabled_for_backend(backend)")
            && forward_source.contains("decode_hot_path_debug_fallback_env_for_backend(backend)"),
        "decode attention native-required routing should combine DecodeBatcherPolicy with BackendFallbackCapabilities"
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
    let native_decode_attention_decline_section = source_between(
        &forward_source,
        "let out_opt = {",
        "// Open the fallback-decode range",
    );
    assert!(
        native_decode_attention_decline_section
            .contains("native_decode_attention_required(backend)")
            && native_decode_attention_decline_section
                .contains("decode_hot_path_debug_fallback_enabled_for_backend(backend)"),
        "paged decode attention decline handling should read backend-owned native/fallback policy"
    );
    for forbidden in [
        "BackendIdentity::runtime_name(backend) == \"vulkan\"",
        "vulkan_decode_generic_fallback_enabled()",
        "KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK",
    ] {
        assert!(
            !native_decode_attention_decline_section.contains(forbidden),
            "paged decode attention decline handling should not keep local Vulkan policy/env branch: {forbidden}"
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
    let detached_chunked_prefill_section = source_between(
        &forward_source,
        "fn transformer_block_detached_prefill_chunked(",
        "let (_batch, seq_len, _hidden) = x.dims3()?",
    );
    assert!(
        detached_chunked_prefill_section.contains("detached_chunked_prefill_supported(backend)"),
        "detached chunked prefill should route through AttentionCapabilities"
    );
    assert!(
        !detached_chunked_prefill_section
            .contains("BackendIdentity::runtime_name(backend) != \"cuda\""),
        "detached chunked prefill should not branch on CUDA backend identity"
    );
    assert!(
        capability_source.contains("\"cuda\" | \"rocm\" => Support::NativeWithConstraints"),
        "detached chunked prefill should advertise both CUDA and ROCm; otherwise ROCm long-context training falls back to monolithic full attention"
    );
    assert!(
        forward_source
            .contains("#[cfg(any(feature = \"cuda\", feature = \"metal\", feature = \"rocm\"))]"),
        "full-attention tape flash call sites must include ROCm so ROCm training uses the gradchecked flash tape path instead of monolithic SDPA"
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
    let native_resident_decode_required_section = source_between(
        &forward_source,
        "fn native_resident_decode_required(",
        "/// Strict batched single-token paged decode",
    );
    assert!(
        native_resident_decode_required_section
            .contains("ReplayBackend::runtime_supports_resident_decode(backend)"),
        "native resident decode requirement should be gated by ReplayBackend capabilities"
    );
    assert!(
        !native_resident_decode_required_section.contains("BackendIdentity::runtime_name(backend)"),
        "native resident decode requirement should not branch on backend identity"
    );
    assert!(
        !forward_source.contains("fn vulkan_decode_generic_fallback_enabled")
            && !forward_source.contains("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK"),
        "forward decode fallback opt-in should come from BackendFallbackCapabilities, not a local Vulkan env helper"
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
    let batched_full_attention_decline_section = source_between(
        &forward_source,
        "match transformer_block_paged_decode_contiguous_batch(",
        "tracing::debug!(",
    );
    assert!(
        batched_full_attention_decline_section
            .contains("native_decode_attention_required(backend)")
            && batched_full_attention_decline_section
                .contains("decode_hot_path_debug_fallback_enabled_for_backend(backend)"),
        "batched full-attention decode decline should read backend-owned native/fallback policy"
    );
    assert!(
        !batched_full_attention_decline_section
            .contains("BackendIdentity::runtime_name(backend) == \"vulkan\"")
            && !batched_full_attention_decline_section
                .contains("vulkan_decode_generic_fallback_enabled()")
            && !batched_full_attention_decline_section
                .contains("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK"),
        "batched full-attention decline should not branch locally on Vulkan identity/env"
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
    let mtp_paged_cache_device_section = source_between(
        &generate_source,
        "fn paged_cache_device(",
        "fn fast_batched_linear_state_scatter_enabled(",
    );
    assert!(
        mtp_paged_cache_device_section
            .contains("BackendCapabilityQueries::backend_capabilities(backend)")
            && mtp_paged_cache_device_section.contains("mtp_speculative_generation"),
        "native MTP paged-cache allocation should read DecodeCapabilities"
    );
    assert!(
        !mtp_paged_cache_device_section.contains("Device::Cuda")
            && !mtp_paged_cache_device_section.contains("kiln_tensor::Device::Cuda"),
        "native MTP paged-cache allocation should not branch locally on CUDA device identity"
    );
    assert!(
        generate_source.contains("paged_cache_device(self.backend.as_ref(),"),
        "native MTP paged-cache allocation call sites should pass the active backend"
    );
    assert!(
        !server_completions_source.contains("ResolvedSpeculativeMode")
            && !server_completions_source.contains("resolve_speculative_mode")
            && !server_completions_source.contains("generate_paged_speculative_shared_tokens"),
        "request dispatch must not contain speculative serving machinery before local accelerator qualification"
    );
    for forbidden in [
        "KILN_ENABLE_METAL_NATIVE_MTP",
        "Device::Metal",
        "kiln_tensor::Device::Metal",
    ] {
        assert!(
            !server_completions_source.contains(forbidden),
            "server request dispatch should not keep a local speculative backend/env policy table: {forbidden}"
        );
    }
    assert!(
        !capability_source.contains("KILN_ENABLE_METAL_NATIVE_MTP")
            && capability_source.contains("\"cuda\" => Support::Declined")
            && capability_source.contains("\"metal\" => Support::Declined"),
        "native MTP must remain declined until cancel-aware external-yield settlement is qualified"
    );
    let bench_mtp_resolver_section = source_between(
        &server_bench_source,
        "fn resolve_bench_spec_method(",
        "fn resolve_bench_spec_method_with_force(",
    );
    assert!(
        server_bench_source.contains("BackendCapabilityQueries::backend_capabilities")
            && server_bench_source.contains("mtp_speculative_generation")
            && server_bench_source.contains("speculative_policy")
            && bench_mtp_resolver_section.contains("native_mtp_allowed"),
        "kiln-bench speculative resolution should receive DecodeCapabilities-derived support"
    );
    for forbidden in [
        "KILN_ENABLE_METAL_NATIVE_MTP",
        "bench_native_mtp_allowed",
        "bench_long_prompt_skip_layer_min_prompt_tokens",
        "BENCH_LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL",
    ] {
        assert!(
            !server_bench_source.contains(forbidden),
            "kiln-bench speculative resolution should not keep a local backend/env policy table: {forbidden}"
        );
    }
    let bench_graph_replay_helper = source_between(
        &server_bench_source,
        "fn bench_paged_decode_replay_primitive_enabled(",
        "// (#1082) Deleted `bench_kt_tensor_to_candle`",
    );
    assert!(
        bench_graph_replay_helper.contains("ReplayRequest::paged_decode_graph_outputs")
            && bench_graph_replay_helper.contains("ReplayBackend::runtime_supports_replay_request")
            && bench_graph_replay_helper.contains("ReplayBackend::runtime_replay_authority"),
        "kiln-bench graph replay routing should ask the focused ReplayBackend facet"
    );
    let bench_latency_graph_section = source_between(
        &server_bench_source,
        "let hip_graph_decode_enabled =",
        "// #1082 forward-flip: `LinearAttentionState::new_with_batch_for_inference_backend`",
    );
    assert!(
        bench_latency_graph_section.contains("bench_paged_decode_replay_primitive_enabled")
            && bench_latency_graph_section.contains("ReplayNativePrimitive::HipGraph"),
        "kiln-bench ROCm graph routing should use replay primitive policy"
    );
    assert!(
        !bench_graph_replay_helper.contains("Device::Rocm")
            && !bench_latency_graph_section.contains("Device::Rocm")
            && !bench_graph_replay_helper
                .contains("matches!(device_kt, kiln_tensor::Device::Rocm(_))")
            && !bench_latency_graph_section
                .contains("matches!(device_kt, kiln_tensor::Device::Rocm(_))"),
        "kiln-bench graph routing should not branch on ROCm device identity"
    );
    let bench_paged_latency_section = source_between(
        &server_bench_source,
        "fn bench_latency_paged(",
        "fn bench_latency_skiplayer(",
    );
    assert!(
        bench_paged_latency_section.contains("backend_capabilities.decode.linear_argmax")
            && bench_paged_latency_section
                .contains("backend_capabilities.decode_batcher.use_greedy_token_decode")
            && bench_paged_latency_section.contains("greedy_token_decode_enabled"),
        "kiln-bench greedy paged latency routing should consume DecodeCapabilities/DecodeBatcherPolicy"
    );
    for forbidden in [
        "device_is_metal",
        "Backend::Metal",
        "supports_linear_decode_argmax()",
    ] {
        assert!(
            !bench_paged_latency_section.contains(forbidden),
            "kiln-bench greedy paged latency routing should not keep a local backend/support table: {forbidden}"
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

    let batched_rocm_graph_route_selection = source_between(
        &generate_source,
        "let hip_graph_single_row_ready =",
        "let greedy_route =",
    );
    assert!(
        batched_rocm_graph_route_selection.contains("paged_decode_replay_primitive_enabled")
            && batched_rocm_graph_route_selection.contains("ReplayNativePrimitive::HipGraph"),
        "batched ROCm graph readiness should use replay primitive policy"
    );
    let batched_rocm_graph_section = source_between(
        &generate_source,
        "// R.9: ROCm HIP-graph single-row decode for the batched/batching-engine",
        "let pc_guard = lock_paged_cache(paged_cache)?;",
    );
    assert!(
        batched_rocm_graph_section.contains("hip_graph_single_row_ready"),
        "batched ROCm graph routing should consume capability-derived graph readiness"
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
    assert!(
        server_bench_source.contains("BackendIdentity::runtime_name")
            && !server_bench_source.contains("backend.name()"),
        "kiln-bench backend identity reads should consume focused BackendIdentity"
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
        backend_source.contains("pub fn training_precision_policy_for_device_kt")
            && backend_source.contains("TrainingLossBackend::runtime_training_precision_policy"),
        "device precision-policy lookup should delegate through the focused TrainingLossBackend facet"
    );
    assert!(
        backend_source.contains("pub fn training_tape_route_for_device_kt")
            && backend_source.contains("TrainingLossBackend::runtime_tape_forward_backward_route"),
        "device tape-forward route lookup should delegate through the focused TrainingLossBackend facet"
    );
    for (path, source) in [
        (
            "crates/kiln-model/src/forward.rs",
            production_source_before_tests(&forward_source),
        ),
        (
            "crates/kiln-model/src/tape_forward.rs",
            production_source_before_tests(&tape_forward_source),
        ),
        (
            "crates/kiln-train/src/trainer.rs",
            production_source_before_tests(&trainer_source),
        ),
    ] {
        assert!(
            !source.contains("TrainingPrecisionPolicy::for_device_family"),
            "{path} production code should not call the device-family compatibility helper"
        );
    }
    let tape_forward_production = production_source_before_tests(&tape_forward_source);
    let tape_forward_compact = compact_body(tape_forward_production);
    assert!(
        tape_forward_production.contains("fn tape_forward_device_supported")
            && tape_forward_production.contains("training_tape_route_for_device_kt(device)"),
        "tape-forward device support should use the backend-owned tape route helper"
    );
    assert!(
        !tape_forward_compact.contains(
            "kiln_tensor::Device::Cuda(_)|kiln_tensor::Device::Metal(_)|kiln_tensor::Device::Vulkan(_)|kiln_tensor::Device::Rocm(_)"
        ),
        "tape-forward production adapters should not hard-code accelerator device families"
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
            "training_activation_bytes_per_elem_for_policy(weights,training_precision_policy,model_config_has_linear_attention(model_config),"
        ) && compact_body(&opd_source).contains("training_activation_bytes_per_elem_for_backend(")
            && compact_body(&opd_source).contains("backend_rt.as_ref()"),
        "SFT/GRPO/OPD production checkpoint sizing should consume backend precision policy"
    );
    let streaming_tile_section = source_between(
        &forward_source,
        "pub fn streaming_tile_tokens_for(",
        "fn trace_model_segment_timings()",
    );
    assert!(
        streaming_tile_section.contains("StreamingPrefillExecutionPolicy::for_device"),
        "streaming prefill tile defaults should read the backend-owned streaming policy"
    );
    assert!(
        !streaming_tile_section.contains("training_precision_policy_for_device_kt")
            && !streaming_tile_section.contains("TrainingPrecisionPolicy"),
        "streaming prefill tile defaults should not retain the superseded training-precision authority"
    );
    assert!(
        !streaming_tile_section.contains("match streaming_prefill_device_kind"),
        "streaming tile defaults should not keep a local backend device table"
    );
    let training_precision_fields = source_between(
        &backend_source,
        "pub struct TrainingPrecisionPolicy {",
        "impl TrainingPrecisionPolicy {",
    );
    for field in [
        "exact_gdn_backward_tile_tokens",
        "streaming_prefill_tile_tokens",
        "tape_streaming_tile_tokens",
        "detached_full_attn_tile_tokens",
        "detached_full_attn_boundary_tile_tokens",
        "detached_full_attn_tape_replay_tile_tokens",
        "paged_prefill_medium_tile_tokens",
        "paged_prefill_medium_tile_max_tokens",
    ] {
        assert!(
            !training_precision_fields.contains(field),
            "retired execution field {field} must not appear in TrainingPrecisionPolicy"
        );
    }
    assert!(
        !trainer_source.contains("KILN_EXACT_GDN_TILE_BACKWARD")
            && !trainer_source.contains("KILN_EXACT_GDN_BACKWARD_TILE_TOKENS"),
        "retired exact-GDN environment controls must not return to the trainer"
    );
    let embedding_activation_cast_section = source_between(
        &forward_source,
        "fn cast_embedding_output_to_policy_activation(",
        "fn embedding_lookup_from_weights_with_index(",
    );
    assert!(
        embedding_activation_cast_section.contains("training_precision_policy_for_device_kt")
            && embedding_activation_cast_section.contains("activation_dtype_for_embedding_output"),
        "embedding activation cast should consume TrainingPrecisionPolicy"
    );
    assert!(
        !embedding_activation_cast_section.contains("TrainingPrecisionPolicy::for_device_family"),
        "embedding activation cast should not call the device-family compatibility helper"
    );
    let embedding_lookup_section = source_between(
        &forward_source,
        "fn embedding_lookup_from_weights(",
        "fn raw_embedding_lookup_from_weights_with_index(",
    );
    assert!(
        embedding_lookup_section.contains("cast_embedding_output_to_policy_activation("),
        "every weight-aware host-index embedding lookup must apply activation precision policy"
    );
    let indexed_embedding_lookup_section = source_between(
        &forward_source,
        "fn embedding_lookup_from_weights_with_index(",
        "fn embedding_lookup_from_transposed(",
    );
    assert!(
        indexed_embedding_lookup_section.contains("cast_embedding_output_to_policy_activation("),
        "every weight-aware device-index embedding lookup must apply activation precision policy"
    );
    assert_eq!(
        forward_source
            .matches("raw_embedding_lookup_from_weights(")
            .count(),
        2,
        "raw host-index embedding lookup must remain private to its policy wrapper"
    );
    assert_eq!(
        forward_source
            .matches("raw_embedding_lookup_from_weights_with_index(")
            .count(),
        2,
        "raw device-index embedding lookup must remain private to its policy wrapper"
    );
    for forbidden in [
        "vulkan_cast_activation_to_f32",
        "matches!(hidden.device(), Device::Vulkan(_))",
        "hidden.dtype() == DType::BF16",
    ] {
        assert!(
            !embedding_activation_cast_section.contains(forbidden),
            "embedding activation cast should not keep a local Vulkan dtype policy: {forbidden}"
        );
    }
    let base_dtype_support_section = source_between(
        &trainer_source,
        "fn base_dtype_supports_tape_for_policy(",
        "/// (#1082 Increment-0 PR2) kt-native sibling",
    );
    assert!(
        base_dtype_support_section.contains("uses_f32_activations_for_mixed_base_weights"),
        "base dtype tape support should read mixed F32 activation policy from TrainingPrecisionPolicy"
    );
    let tape_rms_norm_section = source_between(
        &tape_forward_source,
        "pub fn try_tape_rms_norm_kt(",
        "/// kt-native matmul tape recorder",
    );
    assert!(
        tape_rms_norm_section.contains("training_precision_policy_for_device_kt")
            && tape_rms_norm_section.contains("supports_rms_norm_weight_dtype_for_activation"),
        "RMSNorm tape dtype routing should consume TrainingPrecisionPolicy"
    );
    assert!(
        !tape_rms_norm_section.contains("TrainingPrecisionPolicy::for_device_family"),
        "RMSNorm tape dtype routing should not call the device-family compatibility helper"
    );
    for forbidden in [
        "vk_f32x_bf16w",
        "matches!(x.device(), kiln_tensor::Device::Vulkan(_))",
        "x.dtype() == kiln_tensor::DType::F32",
        "weight.dtype() == kiln_tensor::DType::BF16",
    ] {
        assert!(
            !tape_rms_norm_section.contains(forbidden),
            "RMSNorm tape dtype routing should not keep a local Vulkan mixed-dtype policy: {forbidden}"
        );
    }
    let tape_lora_linear_section = source_between(
        &tape_forward_source,
        "pub fn try_tape_lora_linear_kt(",
        "let out_kt = out_kt.context(\"tape_forward::try_tape_lora_linear_kt",
    );
    assert!(
        tape_lora_linear_section.contains("training_precision_policy_for_device_kt")
            && tape_lora_linear_section.contains("supports_mixed_base_weight_dtype_for_activation"),
        "LoRA tape mixed-base routing should consume TrainingPrecisionPolicy"
    );
    assert!(
        !tape_lora_linear_section.contains("TrainingPrecisionPolicy::for_device_family"),
        "LoRA tape mixed-base routing should not call the device-family compatibility helper"
    );
    for forbidden in [
        "vk_bf16_base",
        "matches!(x.device(), kiln_tensor::Device::Vulkan(_))",
        "x.dtype() == kiln_tensor::DType::F32",
        "weight_t.dtype() == kiln_tensor::DType::BF16",
    ] {
        assert!(
            !tape_lora_linear_section.contains(forbidden),
            "LoRA tape mixed-base routing should not keep a local Vulkan mixed-dtype policy: {forbidden}"
        );
    }
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
    let opd_step_section = &opd_functions
        .get("opd_step_forward_backward_tape_authoritative")
        .expect("opd.rs should define opd_step_forward_backward_tape_authoritative")
        .body;
    assert!(
        opd_step_section.contains("TrainingLossBackend::runtime_opd_loss_route"),
        "OPD tape-authoritative step should route fused loss roots through TrainingLossBackend"
    );
    assert!(
        !opd_step_section.contains("matches!(normed.device(), kiln_tensor::Device::Vulkan(_))"),
        "OPD tape-authoritative step should not hard-code Vulkan loss routing"
    );
    let checkpointed_opd_section = &opd_functions
        .get("checkpointed_opd_step_forward_backward_tape_authoritative")
        .expect("opd.rs should define checkpointed_opd_step_forward_backward_tape_authoritative")
        .body;
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
        "LinearBackend::runtime_lora_decode_add",
        "LinearBackend::runtime_lora_delta_resident",
        "LinearBackend::runtime_linear_prefill_apply",
        "LinearBackend::runtime_linear_decode",
        "LinearBackend::runtime_full_attn_qkv_combined_decode",
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
        "GdnBackend::runtime_gdn_ab_in_proj_prefill",
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
fn forward_weight_kt_accessors_are_device_agnostic_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let full_attn_accessors = source_between(
        &forward_source,
        "impl GpuFullAttentionWeights {",
        "#[derive(Clone)]\npub struct GpuLinearAttentionWeights",
    );
    let gdn_accessors = source_between(
        &forward_source,
        "impl GpuLinearAttentionWeights {",
        "#[derive(Clone)]\npub struct GpuFfnWeights",
    );
    let ffn_accessors = source_between(
        &forward_source,
        "impl GpuFfnWeights {",
        "/// State for Gated DeltaNet linear attention layers.",
    );
    let embed_tokens_accessor = source_between(
        &forward_source,
        "pub fn embed_tokens_kt(&self) -> Result<KtTensor> {",
        "/// kt-native view of the pre-transposed token-embedding table",
    );
    let embed_tokens_t_accessor = source_between(
        &forward_source,
        "pub fn embed_tokens_t_kt(&self) -> Result<KtTensor> {",
        "/// Convert `ModelWeights`",
    );

    for (name, section) in [
        ("full-attention kt accessors", full_attn_accessors),
        ("GDN kt accessors", gdn_accessors),
        ("FFN kt accessors", ffn_accessors),
        ("embedding kt accessor", embed_tokens_accessor),
        ("LM-head kt accessor", embed_tokens_t_accessor),
    ] {
        assert!(
            section.contains("kt_contiguous("),
            "{name} should use the shared native kt contiguous accessor"
        );
        for forbidden in [
            "Device::Cuda",
            "cuda_or_rocm_device",
            "must be on CUDA",
            "zero-copy CUDA bridge",
        ] {
            assert!(
                !section.contains(forbidden),
                "{name} should not decide backend identity locally: {forbidden}"
            );
        }
    }
}

#[test]
fn forward_lora_delta_routes_through_device_ops_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let lora_delta_helper = source_between(
        &forward_source,
        "fn try_kt_lora_delta(",
        "/// Phase 7 (#1082) — **kt-native** LM head matmul core.",
    );

    assert!(
        lora_delta_helper.contains("kiln_tensor::ops::matmul_rhs_transposed"),
        "LoRA delta helper should route both transposed matmuls through DeviceOp dispatch"
    );
    assert!(
        lora_delta_helper.contains("kiln_tensor::ops::mul_scalar"),
        "LoRA delta helper should route scale multiplication through DeviceOp dispatch"
    );
    for forbidden in [
        "kiln_tensor::cuda_matmul_rhs_transposed",
        "kiln_tensor::cuda_scalar_op",
    ] {
        assert!(
            !lora_delta_helper.contains(forbidden),
            "LoRA delta helper should not bind directly to CUDA primitives: {forbidden}"
        );
    }
}

#[test]
fn forward_lm_head_matmul_routes_through_matmul_op_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");

    let lm_head_core = source_between(
        &forward_source,
        "fn kt_lm_head_native(",
        "/// Phase 7 (#1082) — kt-API LM head migration helper.",
    );
    assert!(
        lm_head_core.contains("kiln_tensor::ops::matmul"),
        "LM-head kt core should route matmul through the MatmulOp contract"
    );
    assert!(
        !lm_head_core.contains("kiln_tensor::cuda_matmul("),
        "LM-head kt core should not bind directly to CUDA matmul"
    );

    let lm_head_helper = source_between(
        &forward_source,
        "fn try_kt_lm_head(",
        "fn lm_head_forward_backend_decode_if(",
    );
    for forbidden in [
        "matches!(x.device(), Device::Cuda(_))",
        "matches!(embed_tokens_t.device(), Device::Cuda(_))",
        "kiln_tensor::cuda_matmul(",
    ] {
        assert!(
            !lm_head_helper.contains(forbidden),
            "LM-head kt helper should not select the matmul path by CUDA identity: {forbidden}"
        );
    }

    let lm_head_argmax = source_between(
        &forward_source,
        "fn try_kt_lm_head_argmax(",
        "/// Phase 7 (#1082) — kt-API argmax migration helper.",
    );
    assert!(
        lm_head_argmax.contains("kiln_tensor::ops::matmul"),
        "fused LM-head argmax helper should route its matmul through MatmulOp"
    );
    for forbidden in [
        "matches!(x.device(), Device::Cuda(_))",
        "matches!(embed_tokens_t.device(), Device::Cuda(_))",
        "kiln_tensor::cuda_matmul(",
    ] {
        assert!(
            !lm_head_argmax.contains(forbidden),
            "fused LM-head argmax helper should not select the matmul path by CUDA identity: {forbidden}"
        );
    }
}

#[test]
fn forward_lm_head_argmax_fallbacks_keep_backend_linear_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");

    let scalar_fallback = source_between(
        &forward_source,
        "fn lm_head_argmax_backend_decode_if(",
        "/// Phase 7 (#1082) — kt-API sampler argmax migration helper",
    );
    assert!(
        scalar_fallback.contains("lm_head_argmax_with_backend(backend, x, embed_tokens_t)"),
        "scalar LM-head argmax fallback should keep logits on the backend-aware linear path"
    );
    assert!(
        !scalar_fallback.contains("lm_head_argmax(x, embed_tokens_t)"),
        "scalar LM-head argmax fallback should not bypass the backend-aware linear path"
    );

    let rows_fallback = source_between(
        &forward_source,
        "fn lm_head_argmax_rows_backend_decode_if(",
        "fn lm_head_weighted_prep_argmax(",
    );
    assert!(
        rows_fallback.contains("lm_head_argmax_rows_with_backend(backend, x, embed_tokens_t)"),
        "batched LM-head argmax fallback should keep logits on the backend-aware linear path"
    );
    assert!(
        !rows_fallback.contains("lm_head_argmax_rows(x, embed_tokens_t)"),
        "batched LM-head argmax fallback should not bypass the backend-aware linear path"
    );
}

#[test]
fn forward_gqa_sdpa_matmuls_route_through_matmul_ops_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let sdpa_helper = source_between(
        &forward_source,
        "fn try_kt_gqa_sdpa_matmuls(",
        "pub fn gqa_attention_core_prefill(",
    );

    assert!(
        sdpa_helper.contains("kiln_tensor::ops::matmul_rhs_transposed(q, k)"),
        "GQA SDPA score matmul should route through MatmulRhsTransposedOp"
    );
    assert!(
        sdpa_helper.contains("kiln_tensor::ops::matmul(&p_contig, v)"),
        "GQA SDPA value matmul should route through MatmulOp"
    );
    for forbidden in [
        "kiln_tensor::cuda_matmul_rhs_transposed",
        "kiln_tensor::cuda_matmul(",
    ] {
        assert!(
            !sdpa_helper.contains(forbidden),
            "GQA SDPA helper should not bind directly to CUDA matmul primitives: {forbidden}"
        );
    }
}

#[test]
fn forward_packed_mlp_prefill_routes_gate_up_through_matmul_request_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let gate_up_prefill = source_between(
        &forward_source,
        "kiln/mlp/gate_up_fused_prefill",
        "gate_silu_hidden_mul_packed",
    );

    let request_idx = gate_up_prefill
        .find("runtime_matmul_no_broadcast_copy(backend, x, gate_up_proj_t)")
        .expect("packed MLP gate+up prefill should try the LinearBackend matmul request first");
    let fallback_idx = gate_up_prefill
        .find("broadcast_matmul_cpu_compatible(x, gate_up_proj_t)")
        .expect("packed MLP gate+up prefill should retain the portable broadcast fallback");
    assert!(
        request_idx < fallback_idx,
        "packed MLP gate+up prefill should try request routing before broadcast fallback"
    );
}

#[test]
fn forward_nonpaged_full_attention_mlp_routes_inference_through_backend() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let transformer_block = source_between(
        &forward_source,
        "pub fn transformer_block(",
        "fn transformer_block_detached_prefill_chunked(",
    );

    assert!(
        transformer_block.contains("swiglu_ffn_backend_profiled(")
            && transformer_block.contains("tape_scope_active"),
        "nonpaged full-attention inference must route MLP projections through LinearBackend while preserving the tape-training route"
    );
}

#[test]
fn vulkan_mixed_matmul_capability_is_bound_to_the_resident_dispatch_contract() {
    let root = workspace_root();
    let vulkan_source = fs::read_to_string(root.join("crates/kiln-model/src/backend/vulkan.rs"))
        .expect("Vulkan backend source should be readable");
    let vulkan_linear_source =
        fs::read_to_string(root.join("crates/kiln-model/src/backend/vulkan_linear.rs"))
            .expect("Vulkan linear source should be readable");

    let capability_impl = source_between(
        &vulkan_source,
        "fn runtime_supports_matmul_request(",
        "fn runtime_matmul(",
    );
    assert!(
        capability_impl.contains("vulkan_linear::matmul_request_support(req)"),
        "Vulkan capability reporting should delegate to the linear route's request contract"
    );

    let mixed_predicate = source_between(
        &vulkan_linear_source,
        "pub(super) fn resident_mixed_rank2_request_supported(",
        "pub(super) fn matmul_request_support(",
    );
    for required in [
        "req.rank() == Some(2)",
        "MatmulOperandLayout::RowMajor",
        "MatmulEpilogue::Identity",
        "req.lhs_dtype == kiln_tensor::DType::F32",
        "req.rhs_dtype == kiln_tensor::DType::BF16",
        "req.out_dtype == kiln_tensor::DType::F32",
    ] {
        assert!(
            mixed_predicate.contains(required),
            "resident mixed Vulkan request contract should contain {required}"
        );
    }
    assert!(
        !mixed_predicate.contains("std::env") && !mixed_predicate.contains("KILN_"),
        "mixed Vulkan capability truth must not depend on a test or runtime environment gate"
    );

    let support_impl = source_between(
        &vulkan_linear_source,
        "pub(super) fn matmul_request_support(",
        "pub(super) fn matmul(",
    );
    let dispatch_impl = source_between(
        &vulkan_linear_source,
        "pub(super) fn matmul(",
        "fn resident_matmul(",
    );
    assert!(
        support_impl.contains("resident_mixed_rank2_request_supported(req)")
            && dispatch_impl.contains("resident_mixed_rank2_request_supported(req)"),
        "capability reporting and resident dispatch should share the exact mixed request predicate"
    );
    assert!(
        dispatch_impl.contains("kiln_tensor::Device::Vulkan(_)")
            && dispatch_impl.contains("return resident_matmul(req, lhs, rhs, layout)"),
        "mixed capability support must keep Vulkan residency checks and homogeneous-route rollback"
    );
}

#[test]
fn forward_mtp_matmuls_route_through_linear_backend_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let mtp_forward = source_between(
        &forward_source,
        "pub fn mtp_forward_step(",
        "fn model_forward_paged_inner(",
    );

    for required in [
        "runtime_matmul_or_broadcast(backend, &concat_f32, &fc_t_f32)",
        "runtime_matmul_or_broadcast(backend, &concat, &mtp.fc_t)",
        "lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)",
    ] {
        assert!(
            mtp_forward.contains(required),
            "MTP forward should route matmul through LinearBackend before portable fallback: {required}"
        );
    }
    for forbidden in [
        "concat_f32.broadcast_matmul(&fc_t_f32)",
        "concat.broadcast_matmul(&mtp.fc_t)",
        "lm_head_forward(&normed, &weights.embed_tokens_t)",
    ] {
        assert!(
            !mtp_forward.contains(forbidden),
            "MTP forward should not bypass the backend matmul contract: {forbidden}"
        );
    }
}

#[test]
fn forward_full_attn_qkv_combined_routes_through_linear_backend_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let qkv_helper = source_between(
        &forward_source,
        "fn full_attn_qkv_proj_decode_if(",
        "/// CUDA-compatible softmax on last dimension.",
    );

    assert!(
        qkv_helper.contains("LinearBackend::runtime_full_attn_qkv_combined_decode"),
        "combined full-attention QKV projection should route through LinearBackend"
    );
    assert!(
        qkv_helper.contains("LinearBackend::runtime_full_attn_qkv_decode"),
        "split full-attention QKV projection should continue to route through LinearBackend"
    );
    for forbidden in [
        "cuda_rocm_full_attn_qkv_in_proj_enabled",
        "cuda_or_rocm_device(x.device())",
        "cuda_or_rocm_device(qkv_proj_t.device())",
        "matches!(x.device(), Device::Rocm(_))",
        "crate::rocm_w8_proj::matmul_bf16",
        "broadcast_matmul_cpu_compatible(x, qkv_proj_t)",
    ] {
        assert!(
            !qkv_helper.contains(forbidden),
            "forward full-attention QKV helper should not select combined matmul by backend identity: {forbidden}"
        );
    }
}

#[test]
fn forward_gdn_ab_in_proj_routes_through_gdn_backend_contract() {
    let forward_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/forward.rs"))
            .expect("forward source should be readable");
    let gdn_helper = source_between(
        &forward_source,
        "fn gated_deltanet_forward_decode_if_inner(",
        "// Phase B11b tap: `gdn_in_proj`.",
    );

    assert!(
        gdn_helper.contains("GdnBackend::runtime_gdn_ab_in_proj_prefill"),
        "GDN A/B in-projection should route through GdnBackend"
    );
    for forbidden in [
        "cuda_rocm_gdn_ab_in_proj_enabled",
        "cuda_rocm_gdn_prefill_ab_in_proj_enabled",
        "CUDA_ROCM_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS",
        "crate::backend::metal::metal_gdn_prefill_ab_in_proj_supports",
        "crate::backend::metal::metal_gdn_prefill_ab_in_proj_bf16",
        "cuda_or_rocm_device(in_proj_ab_t.device())",
        "broadcast_matmul_cpu_compatible(x, in_proj_ab_t)",
    ] {
        assert!(
            !gdn_helper.contains(forbidden),
            "forward GDN A/B in-projection should not select combined matmul by backend identity: {forbidden}"
        );
    }
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
    for path in [
        "crates/kiln-model/tests/cuda_sft_step_proof.rs",
        "crates/kiln-model/tests/rocm_sft_step_proof.rs",
    ] {
        let source =
            fs::read_to_string(workspace_root().join(path)).expect("SFT proof should be readable");
        assert!(
            source.contains("OptimizerBackend::runtime_dispatch_adamw_step")
                && source.contains("ResidencyBackend::runtime_register_resident_activation"),
            "{path} should route its AdamW proof through the backend optimizer facet"
        );
        assert!(
            !source.contains("kiln_rmsnorm_kernel::adamw_step_f32_kt"),
            "{path} should not bypass OptimizerBackend with a direct AdamW kernel call"
        );
    }
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

    let manifest_path = root.join("docs/backend-latency-fixtures.json");
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(&manifest_path).expect("latency fixture manifest should be readable"),
    )
    .expect("latency fixture manifest should parse");
    let manifest_status = manifest["status"]
        .as_str()
        .expect("latency fixture manifest status should be a string");
    assert!(
        manifest_status == "covered" || manifest_status == "fixture_required",
        "unexpected latency fixture manifest status {manifest_status}"
    );
    assert_eq!(
        hardware_gate["status"], manifest["status"],
        "hardware latency gate status should be derived from the fixture manifest"
    );
    let command = hardware_gate["command"]
        .as_str()
        .expect("hardware latency command should be a string");
    assert!(command.contains("check_backend_latency_fixtures.py"));
    assert!(command.contains("run_backend_latency_fixture.py"));
    assert!(command.contains("write_backend_latency_result_artifact.py"));
    assert!(command.contains("import_backend_latency_artifact.py"));
    assert!(command.contains("lock_backend_latency_thresholds.py"));
    assert!(command.contains("plan_backend_latency_fixture_dispatch.py"));
    assert!(command.contains("--self-test"));
    assert!(command.contains("--require-covered"));

    let checker_source = fs::read_to_string(root.join("scripts/check_backend_latency_fixtures.py"))
        .expect("latency fixture checker should be readable");
    for required in [
        "def validate_result_artifact(",
        "def metric_threshold_passes(",
        "MANIFEST_KEYS",
        "FIXTURE_KEYS",
        "REQUIRED_COVERED_GATE_POLICY",
        "manifest contains unknown keys",
        "runner_labels",
        "must be a non-empty string array",
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
        "raw_log_tail",
        "raw log tail:",
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
        "git_dirty_status_lines",
        "tracked_git_dirty",
        "rev-parse",
        "--untracked-files=all",
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

    let importer_source =
        fs::read_to_string(root.join("scripts/import_backend_latency_artifact.py"))
            .expect("latency artifact importer should be readable");
    for required in [
        "def import_backend_latency_artifact(",
        "def safe_extract_zip(",
        "zipfile",
        "artifact_bundle",
        "locate_raw_log",
        "validate_result_artifact",
        "require_threshold_pass=False",
        "raw log checksum does not match",
        "LATENCY_RESULT_ARTIFACT_DIR",
        "LATENCY_RAW_LOG_DIR",
        "is_canonical_result_artifact_path",
        "is_canonical_raw_log_path",
        "fixture_spec_sha256",
        "git_tracked_dirty",
        "--force",
        "--fixture-id",
        "--self-test",
    ] {
        assert!(
            importer_source.contains(required),
            "latency artifact importer should validate downloaded workflow artifacts: {required}"
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

    let local_gate_source = fs::read_to_string(root.join("scripts/check_unification_gates.sh"))
        .expect("local unification gate should be readable");
    for required in [
        "set -euo pipefail",
        "PYTHON_BIN",
        "CARGO_BIN",
        "scripts/generate_backend_capability_report.py",
        "--check",
        "test --locked -p kiln-model --test backend_capability_contract",
        "test --locked -p kiln-tensor tensor::tests",
        "test --locked -p kiln-tensor device_op::tests",
        "test --locked -p kiln-tensor matmul_matrix_core",
        "test --locked -p kiln-optim --test integration",
        "test --locked -p kiln-optim --test end_to_end_training",
        "test --locked -p kiln-graph replay",
        "test --locked -p kiln-graph --test capture_lifetime",
        "scripts/run_backend_latency_fixture.py --self-test",
        "scripts/write_backend_latency_result_artifact.py --self-test",
        "scripts/import_backend_latency_artifact.py --self-test",
        "scripts/lock_backend_latency_thresholds.py --self-test",
        "scripts/check_backend_latency_fixtures.py --self-test",
        "scripts/plan_backend_latency_fixture_dispatch.py --self-test",
        "docs/backend-latency-fixtures.json",
        "--require-covered",
    ] {
        assert!(
            local_gate_source.contains(required),
            "local unification gate should run {required}"
        );
    }

    let planner_source =
        fs::read_to_string(root.join("scripts/plan_backend_latency_fixture_dispatch.py"))
            .expect("latency fixture dispatch planner should be readable");
    for required in [
        "def dispatch_plans(",
        "gh workflow run",
        "latency_fixture_id",
        "latency_runner_labels_json",
        "runner_labels",
        "needs_runner_labels",
        "artifact_name_template",
        "gh_run_download",
        "import_artifact",
        "lock_threshold",
        "covered_gate_check",
        "RUN_ID",
        "github_runner_check",
        "matching_online_idle",
        "no_matching_runner",
        "--check-runners",
        "--github-repo",
        "gh api",
        "actions/runners",
        "--runner-labels-json",
        "--shell",
        "--self-test",
    ] {
        assert!(
            planner_source.contains(required),
            "latency fixture dispatch planner should expose workflow commands: {required}"
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
        "scripts/check_unification_gates.sh",
        "scripts/run_backend_latency_fixture.py",
        "scripts/write_backend_latency_result_artifact.py",
        "scripts/import_backend_latency_artifact.py",
        "scripts/lock_backend_latency_thresholds.py",
        "scripts/check_backend_latency_fixtures.py",
        "scripts/plan_backend_latency_fixture_dispatch.py",
        "crates/kiln-tensor/tests/cuda_latency_bench.rs",
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
        "untracked repo files",
        "--untracked-files=all",
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
        "import_backend_latency_artifact.py",
        "lock_backend_latency_thresholds.py",
        "workflow_dispatch",
        "latency_fixture_id",
        "latency_runner_labels_json",
        "runner_labels",
        "cuda-rtx4090",
        "rocm-gfx1151",
        "vulkan-strix-halo",
        "fixture_spec_sha256",
        "plan_backend_latency_fixture_dispatch.py",
        "gh_run_download",
        "import_artifact",
        "lock_threshold",
        "covered_gate_check",
        "RUN_ID",
        "github_runner_check",
        "--check-runners",
        "--github-repo",
        "gh api",
        "bench-results/backend-latency/*.json",
        "bench-results/backend-latency/raw/*.log",
        "--force",
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
    let coverage_blockers = hardware_gate["coverage_blockers"]
        .as_array()
        .expect("hardware latency coverage blockers should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if manifest_status == "covered" {
        assert!(
            coverage_blockers.is_empty(),
            "covered hardware latency gate should not expose fixture blockers: {coverage_blockers:?}"
        );
    } else {
        assert!(
            !coverage_blockers.is_empty(),
            "fixture-required hardware latency gate should explain its blockers"
        );
        assert!(
            coverage_blockers
                .iter()
                .any(|blocker| blocker.contains("manifest status is 'fixture_required'")),
            "fixture-required gate should identify the manifest status: {coverage_blockers:?}"
        );
        for fixture in manifest["fixtures"]
            .as_array()
            .expect("fixtures should be an array")
        {
            if fixture["threshold_state"] == "pending_fixture_result" {
                let fixture_id = fixture["id"]
                    .as_str()
                    .expect("pending fixture id should be a string");
                assert!(
                    coverage_blockers
                        .iter()
                        .any(|blocker| blocker.contains(fixture_id)),
                    "pending fixture {fixture_id} should be named in coverage blockers: {coverage_blockers:?}"
                );
            }
        }
    }

    assert_eq!(manifest["schema_version"], 1);
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
        let threshold_state = fixture["threshold_state"]
            .as_str()
            .expect("fixture threshold_state should be a string");
        assert!(
            threshold_state == "pending_fixture_result" || threshold_state == "locked_threshold",
            "fixture threshold_state should be pending or locked: {threshold_state}"
        );
        let source = fixture["source"]
            .as_str()
            .expect("fixture source should be a string");
        let source_text = fs::read_to_string(root.join(source))
            .unwrap_or_else(|err| panic!("fixture source should be readable: {source}: {err}"));
        assert!(
            root.join(source).is_file(),
            "fixture source should exist: {source}"
        );
        let expected_runner_labels: &[&str] = match fixture["id"].as_str() {
            Some("cuda_rtx4090_matmul_qwen35_4b") => &["self-hosted", "linux", "cuda-rtx4090"],
            Some("rocm_gfx1151_matmul_qwen35_4b") => &["self-hosted", "linux", "rocm-gfx1151"],
            Some("vulkan_strix_halo_decode_microbench") => {
                &["self-hosted", "linux", "vulkan-strix-halo"]
            }
            _ => &[],
        };
        if !expected_runner_labels.is_empty() {
            let fixture_id = fixture["id"]
                .as_str()
                .expect("fixture id should be a string");
            let runner_labels = fixture["runner_labels"]
                .as_array()
                .unwrap_or_else(|| panic!("{fixture_id} should declare stable runner labels"))
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            for label in expected_runner_labels {
                assert!(
                    runner_labels.contains(label),
                    "{fixture_id} should declare runner label {label}"
                );
            }
        }
        assert!(
            source_text.contains("KILN_LATENCY_METRIC"),
            "fixture source should emit machine-readable latency metric lines: {source}"
        );
        let metrics = fixture["metrics"]
            .as_array()
            .expect("fixture metrics should be an array");
        let command = fixture["command"]
            .as_str()
            .expect("fixture command should be a string");
        assert!(
            !command.contains("/home/") && !command.contains("/Users/"),
            "fixture command should be runner-portable because it is part of the stable fixture digest: {command}"
        );
        assert!(
            !command.contains("/path/to"),
            "fixture command should not contain placeholder paths because workflow_dispatch runs it verbatim: {command}"
        );
        assert!(
            command.contains("cargo "),
            "fixture command should invoke cargo through the runner PATH: {command}"
        );
        assert!(
            !metrics.is_empty(),
            "fixture should declare at least one latency metric"
        );
        for metric in metrics {
            if threshold_state == "pending_fixture_result" {
                assert!(
                    metric["max"].is_null(),
                    "pending fixture metrics should not pretend thresholds are locked"
                );
            } else {
                let max = metric["max"]
                    .as_f64()
                    .expect("locked fixture metric max should be numeric");
                assert!(
                    max.is_finite() && max > 0.0,
                    "locked fixture metric max should be finite and positive: {max}"
                );
            }
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
    let conformance_gates = report["conformance_gates"]
        .as_array()
        .expect("conformance_gates should be an array");
    let hardware_latency_status = conformance_gates
        .iter()
        .find(|gate| gate["gate"] == "hardware_latency_thresholds")
        .and_then(|gate| gate["status"].as_str())
        .expect("hardware latency threshold gate should have a status");
    let expected_phase8_status = if conformance_gates
        .iter()
        .all(|gate| gate["status"] == "covered")
    {
        "covered"
    } else if hardware_latency_status == "fixture_required"
        && conformance_gates.iter().all(|gate| {
            gate["gate"] == "hardware_latency_thresholds" || gate["status"] == "covered"
        })
    {
        "fixture_required"
    } else {
        "partial"
    };
    assert_eq!(
        phases.len(),
        9,
        "migration phase status should cover phases 0-8"
    );

    let valid_statuses = ["covered", "partial", "gap", "fixture_required"];
    let valid_contract_states = ["landed", "absent"];
    let valid_migration_states = ["none", "partial", "complete"];
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
        let contract = phase["contract"]
            .as_str()
            .expect("migration phase contract state should be a string");
        assert!(
            valid_contract_states.contains(&contract),
            "invalid migration phase contract state {contract}"
        );
        let migration = phase["migration"]
            .as_str()
            .expect("migration phase migration state should be a string");
        assert!(
            valid_migration_states.contains(&migration),
            "invalid migration phase migration state {migration}"
        );
        assert!(
            phase["genuine"].is_boolean(),
            "migration phase genuine flag should be boolean"
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
    assert_eq!(phase0["contract"], "landed");
    assert_eq!(phase0["migration"], "complete");
    assert_eq!(phase0["genuine"], true);
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

    let phase1 = phases
        .iter()
        .find(|phase| phase["phase"] == 1)
        .expect("Phase 1 should be present");
    assert_eq!(
        phase1["status"], "covered",
        "Phase 1 should be covered once focused facets are authoritative and BackendRuntime is identity-only"
    );
    assert_eq!(phase1["contract"], "landed");
    assert_eq!(phase1["migration"], "complete");
    assert_eq!(phase1["genuine"], true);
    let phase1_signals = phase1["migration_signals"]
        .as_array()
        .expect("Phase 1 should list machine signals");
    let identity_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "backend_identity_facet_authoritative")
        .expect("Phase 1 should include BackendIdentity authoritative signal");
    assert_eq!(
        identity_signal["passed"], true,
        "BackendIdentity should be a completed W1 family slice"
    );
    let startup_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "startup_backend_facet_authoritative")
        .expect("Phase 1 should include StartupBackend authoritative signal");
    assert_eq!(
        startup_signal["passed"], true,
        "StartupBackend should be a completed W1 family slice"
    );
    let conv_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "conv_backend_facet_authoritative")
        .expect("Phase 1 should include ConvBackend authoritative signal");
    assert_eq!(
        conv_signal["passed"], true,
        "ConvBackend should be a completed W1 family slice"
    );
    let attention_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "attention_backend_facet_authoritative")
        .expect("Phase 1 should include AttentionBackend authoritative signal");
    assert_eq!(
        attention_signal["passed"], true,
        "AttentionBackend should be a completed W1 family slice"
    );
    let gdn_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "gdn_backend_facet_authoritative")
        .expect("Phase 1 should include GdnBackend authoritative signal");
    assert_eq!(
        gdn_signal["passed"], true,
        "GdnBackend should be a completed W1 family slice"
    );
    let linear_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "linear_backend_facet_authoritative")
        .expect("Phase 1 should include LinearBackend authoritative signal");
    assert_eq!(
        linear_signal["passed"], true,
        "LinearBackend should be a completed W1 family slice"
    );
    let residency_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "residency_backend_facet_authoritative")
        .expect("Phase 1 should include ResidencyBackend authoritative signal");
    assert_eq!(
        residency_signal["passed"], true,
        "ResidencyBackend should be a completed W1 family slice"
    );
    let sampling_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "sampling_backend_facet_authoritative")
        .expect("Phase 1 should include SamplingBackend authoritative signal");
    assert_eq!(
        sampling_signal["passed"], true,
        "SamplingBackend should be a completed W1 family slice"
    );
    let optimizer_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "optimizer_backend_facet_authoritative")
        .expect("Phase 1 should include OptimizerBackend authoritative signal");
    assert_eq!(
        optimizer_signal["passed"], true,
        "OptimizerBackend should be a completed W1 family slice"
    );
    let paged_kv_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "paged_kv_backend_facet_authoritative")
        .expect("Phase 1 should include PagedKvBackend authoritative signal");
    assert_eq!(
        paged_kv_signal["passed"], true,
        "PagedKvBackend should be a completed W1 family slice"
    );
    let replay_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "replay_backend_facet_authoritative")
        .expect("Phase 1 should include ReplayBackend authoritative signal");
    assert_eq!(
        replay_signal["passed"], true,
        "ReplayBackend should be a completed W1 family slice"
    );
    let training_loss_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "training_loss_backend_facet_authoritative")
        .expect("Phase 1 should include TrainingLossBackend authoritative signal");
    assert_eq!(
        training_loss_signal["passed"], true,
        "TrainingLossBackend should be a completed W1 family slice"
    );
    let shim_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "focused_trait_forwarding_shims_removed")
        .expect("Phase 1 should include all-shims-removed signal");
    assert_eq!(
        shim_signal["passed"], true,
        "Focused trait blanket shims should stay removed once all W1 families are authoritative"
    );
    let method_count_signal = phase1_signals
        .iter()
        .find(|signal| signal["name"] == "backend_runtime_method_count_below_gate")
        .expect("Phase 1 should include BackendRuntime method-count gate");
    assert_eq!(
        method_count_signal["passed"], true,
        "Phase 1 should stay complete only while BackendRuntime remains below the method-count gate"
    );
    assert_eq!(
        method_count_signal["observed"]
            .as_u64()
            .expect("BackendRuntime method count should be numeric"),
        3,
        "BackendRuntime should stay identity-only after W1 deletes compatibility methods"
    );

    let phase2 = phases
        .iter()
        .find(|phase| phase["phase"] == 2)
        .expect("Phase 2 should be present");
    assert_eq!(
        phase2["status"], "covered",
        "Phase 2 should report complete only after fallback policy helpers are centralized"
    );
    assert_eq!(phase2["contract"], "landed");
    assert_eq!(phase2["migration"], "complete");
    assert_eq!(phase2["genuine"], true);
    let phase2_signals = phase2["migration_signals"]
        .as_array()
        .expect("Phase 2 should list fallback-policy migration signals");
    for signal_name in [
        "decode_hot_path_duplicate_helpers_removed",
        "decode_hot_path_fallback_delegates_to_backend_capability",
    ] {
        let signal = phase2_signals
            .iter()
            .find(|signal| signal["name"] == signal_name)
            .unwrap_or_else(|| panic!("Phase 2 should include migration signal {signal_name}"));
        assert_eq!(
            signal["passed"], true,
            "Phase 2 signal {signal_name} should pass before Phase 2 can be genuine"
        );
    }

    let phase3 = phases
        .iter()
        .find(|phase| phase["phase"] == 3)
        .expect("Phase 3 should be present");
    assert_eq!(
        phase3["status"], "covered",
        "Phase 3 should be covered only when registry routing, ownership, and lifecycle metadata signals pass"
    );
    assert_eq!(phase3["contract"], "landed");
    assert_eq!(phase3["migration"], "complete");
    assert_eq!(phase3["genuine"], true);
    let phase3_signals = phase3["migration_signals"]
        .as_array()
        .expect("Phase 3 should list registry migration signals");
    for signal_name in [
        "resident_registry_blanket_adapter_removed",
        "production_backends_implement_resident_registry",
        "residency_backend_facade_delegates_to_registry",
        "resident_registry_process_global_statics_removed",
        "resident_registry_drop_drains_test_present",
        "resident_registry_lifecycle_metadata_persisted",
    ] {
        let signal = phase3_signals
            .iter()
            .find(|signal| signal["name"] == signal_name)
            .unwrap_or_else(|| panic!("Phase 3 should include migration signal {signal_name}"));
        assert_eq!(
            signal["passed"], true,
            "Phase 3 signal {signal_name} should pass before Phase 3 can be genuine"
        );
    }

    let phase4 = phases
        .iter()
        .find(|phase| phase["phase"] == 4)
        .expect("Phase 4 should be present");
    assert_eq!(
        phase4["status"], "covered",
        "Phase 4 should report complete only after request routing and identity-dispatch guards pass"
    );
    assert_eq!(phase4["contract"], "landed");
    assert_eq!(phase4["migration"], "complete");
    assert_eq!(phase4["genuine"], true);
    let phase4_signals = phase4["migration_signals"]
        .as_array()
        .expect("Phase 4 should list machine signals");
    for signal_name in [
        "matmul_request_descriptor_w4_1_lossless",
        "matmul_support_query_delegates_to_linear_backend",
        "matmul_transposed_request_contract_present",
    ] {
        let signal = phase4_signals
            .iter()
            .find(|signal| signal["name"] == signal_name)
            .unwrap_or_else(|| panic!("Phase 4 should include migration signal {signal_name}"));
        assert_eq!(
            signal["passed"], true,
            "Phase 4 signal {signal_name} should pass before W4 partial progress is advertised"
        );
    }
    let identity_signal = phase4_signals
        .iter()
        .find(|signal| signal["name"] == "matmul_linear_identity_dispatch_removed")
        .expect("Phase 4 should include the final identity-dispatch migration signal");
    assert_eq!(
        identity_signal["passed"], true,
        "Phase 4 should be genuine only when scoped matmul/linear identity dispatch is removed"
    );
    assert_eq!(
        identity_signal["observed"], 0,
        "Phase 4 identity-dispatch guard should observe no W4-owned backend identity branches"
    );

    let phase5 = phases
        .iter()
        .find(|phase| phase["phase"] == 5)
        .expect("Phase 5 should be present");
    assert_eq!(
        phase5["status"], "covered",
        "Phase 5 should report complete only after replay routing and parity guards pass"
    );
    assert_eq!(phase5["contract"], "landed");
    assert_eq!(phase5["migration"], "complete");
    assert_eq!(phase5["genuine"], true);
    let phase5_signals = phase5["migration_signals"]
        .as_array()
        .expect("Phase 5 should list migration signals");
    let replay_contract_signal = phase5_signals
        .iter()
        .find(|signal| signal["name"] == "replay_contract_w5_0_fixed")
        .expect("Phase 5 should include W5.0 contract signal");
    assert_eq!(
        replay_contract_signal["passed"], true,
        "Phase 5 W5.0 contract signal should pass after replay contract bugs are fixed"
    );
    let production_replay_signal = phase5_signals
        .iter()
        .find(|signal| signal["name"] == "production_replay_paths_use_replay_plan")
        .expect("Phase 5 should include production replay wiring signal");
    assert_eq!(
        production_replay_signal["passed"], true,
        "Phase 5 production wiring signal should pass only when all decode replay runner families use ReplayPlan"
    );
    assert_eq!(
        production_replay_signal["observed"], 4,
        "Phase 5 should report CUDA, ROCm, Metal, and Vulkan production replay runner slices as wired"
    );
    assert_eq!(
        production_replay_signal["expected"], 4,
        "Phase 5 production wiring should require CUDA, ROCm, Metal, and Vulkan runner families"
    );
    let replay_parity_signal = phase5_signals
        .iter()
        .find(|signal| signal["name"] == "replay_parity_w5_3_live_gate")
        .expect("Phase 5 should include W5.3 eager-vs-replay parity signal");
    assert_eq!(
        replay_parity_signal["passed"], true,
        "Phase 5 must not become genuine until replay parity is live rather than a skip-only scaffold"
    );
    assert_eq!(
        replay_parity_signal["observed"], 3,
        "Phase 5 should count the local CPU/mock contract plus Metal and CUDA parity gates as present"
    );
    assert_eq!(
        replay_parity_signal["expected"], 3,
        "Phase 5 W5.3 should require local CPU/mock, Metal, and live CUDA replay parity gates"
    );

    let phase6 = phases
        .iter()
        .find(|phase| phase["phase"] == 6)
        .expect("Phase 6 should be present");
    assert_eq!(
        phase6["status"], "covered",
        "Phase 6 should report complete only after training precision routing leaves production device-family lookup"
    );
    assert_eq!(phase6["contract"], "landed");
    assert_eq!(phase6["migration"], "complete");
    assert_eq!(phase6["genuine"], true);
    let phase6_signals = phase6["migration_signals"]
        .as_array()
        .expect("Phase 6 should list migration signals");
    let device_family_signal = phase6_signals
        .iter()
        .find(|signal| {
            signal["name"] == "training_precision_for_device_family_removed_from_production"
        })
        .expect("Phase 6 should include the production for_device_family removal signal");
    assert_eq!(
        device_family_signal["passed"], true,
        "Phase 6 should not be genuine while production training paths call for_device_family"
    );
    assert_eq!(
        device_family_signal["observed"], 0,
        "Phase 6 should observe no production TrainingPrecisionPolicy::for_device_family call sites"
    );
    let backend_trait_signal = phase6_signals
        .iter()
        .find(|signal| signal["name"] == "training_precision_policy_delegates_to_backend_trait")
        .expect("Phase 6 should include the backend trait precision-policy signal");
    assert_eq!(
        backend_trait_signal["passed"], true,
        "Phase 6 should select precision policy through TrainingLossBackend"
    );
    let tape_guard_signal = phase6_signals
        .iter()
        .find(|signal| signal["name"] == "tape_forward_gpu_family_guards_removed")
        .expect("Phase 6 should include the tape-forward device-family guard removal signal");
    assert_eq!(
        tape_guard_signal["passed"], true,
        "Phase 6 should not be genuine while tape-forward adapters hard-code accelerator device families"
    );
    assert_eq!(
        tape_guard_signal["observed"], 0,
        "Phase 6 should observe no tape-forward accelerator-family allowlist guards"
    );
    let tape_route_signal = phase6_signals
        .iter()
        .find(|signal| signal["name"] == "tape_forward_route_delegates_to_backend_trait")
        .expect("Phase 6 should include the tape-forward backend trait routing signal");
    assert_eq!(
        tape_route_signal["passed"], true,
        "Phase 6 should select tape-forward support through TrainingLossBackend"
    );
    let sft_proof_signal = phase6_signals
        .iter()
        .find(|signal| signal["name"] == "sft_step_proofs_route_optimizer_backend")
        .expect("Phase 6 should include the SFT proof optimizer routing signal");
    assert_eq!(
        sft_proof_signal["passed"], true,
        "Phase 6 should require CUDA/ROCm SFT proofs to route AdamW through OptimizerBackend"
    );

    let phase7 = phases
        .iter()
        .find(|phase| phase["phase"] == 7)
        .expect("Phase 7 should be present");
    assert_eq!(phase7["status"], "covered");
    assert_eq!(phase7["contract"], "landed");
    assert_eq!(phase7["migration"], "complete");
    assert_eq!(phase7["genuine"], true);
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
    assert_eq!(
        phase8["status"], expected_phase8_status,
        "Phase 8 should derive its status from the conformance gates"
    );
    assert_eq!(phase8["contract"], "landed");
    let phase8_complete = expected_phase8_status == "covered";
    assert_eq!(
        phase8["migration"],
        if phase8_complete {
            "complete"
        } else {
            "partial"
        }
    );
    assert_eq!(phase8["genuine"], phase8_complete);
    let phase8_remaining = phase8["remaining"]
        .as_array()
        .expect("Phase 8 remaining should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if phase8_complete {
        assert!(
            phase8_remaining.is_empty(),
            "covered Phase 8 should not list remaining hardware latency work: {phase8_remaining:?}"
        );
    } else {
        assert!(
            phase8_remaining
                .iter()
                .any(|remaining| remaining.contains("hardware_latency_thresholds")),
            "incomplete Phase 8 should name remaining hardware latency work: {phase8_remaining:?}"
        );
    }

    let report_md = fs::read_to_string(root.join("docs/backend-capability-report.md"))
        .expect("capability report markdown should be readable");
    let generator_source =
        fs::read_to_string(root.join("scripts/generate_backend_capability_report.py"))
            .expect("capability report generator should be readable");
    assert!(
        !generator_source.contains("\"status\": \"covered\""),
        "capability report generator should derive covered status instead of hardcoding it"
    );
    assert!(
        !generator_source.contains("\"genuine\": true"),
        "capability report generator should compute genuine flags instead of hardcoding them"
    );
    assert!(
        !generator_source.contains("\"genuine\": True"),
        "capability report generator should compute genuine flags instead of hardcoding them"
    );
    assert!(
        report_md.contains("## Migration Phase Status"),
        "Markdown report should expose migration phase status"
    );
    assert!(
        report_md.contains(
            "| Phase 1 | Introduce focused backend traits | `covered` | `landed` | `complete` | yes |"
        ),
        "Markdown report should expose Phase 1 as complete and genuine"
    );
    let expected_phase8_row = format!(
        "| Phase 8 | Conformance and performance gates | `{expected_phase8_status}` | `landed` | `{}` | {} |",
        if phase8_complete {
            "complete"
        } else {
            "partial"
        },
        if phase8_complete { "yes" } else { "no" }
    );
    assert!(
        report_md.contains(&expected_phase8_row),
        "Markdown report should match the derived Phase 8 state: {expected_phase8_row}"
    );
}

#[test]
fn cuda_graph_replay_hidden_routes_through_replay_plan_contract() {
    let cuda_graph_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/cuda_graph.rs"))
            .expect("cuda_graph.rs should be readable");
    let replay_section = source_between(
        &cuda_graph_source,
        "if let Some(captured) = self.captured.get(&cache_key)",
        "if self.captured.len() >= Self::max_cached_graphs()",
    );

    assert!(
        cuda_graph_source.contains("struct CudaDecodeReplayPlan")
            && cuda_graph_source.contains("impl ReplayPlan for CudaDecodeReplayPlan"),
        "CUDA graph runner should expose a production ReplayPlan adapter"
    );
    assert!(
        cuda_graph_source.contains("fn replay_state_for_capture(")
            && cuda_graph_source.contains("Backend::Cuda")
            && cuda_graph_source.contains("ReplayState::new(replay_key, resources)")
            && cuda_graph_source.contains("ReplayResourceStability::StableAcrossReplay"),
        "CUDA graph capture should persist shared replay key/resource validation state"
    );
    assert!(
        cuda_graph_source.contains("let owner = CudaGraphOwner::from_row_id(graph_row_id)")
            && cuda_graph_source
                .contains("let cache_key = CudaGraphCacheKey::new(owner, requested_key.clone())")
            && replay_section.contains("self.captured.get(&cache_key)")
            && replay_section.contains("self.captured.remove(&cache_key)"),
        "CUDA graph replay should be keyed by decode-row owner, not only graph shape"
    );
    assert!(
        replay_section.contains("CudaDecodeReplayPlan::new(captured)")
            && replay_section.contains("ReplayInputs::new")
            && replay_section.contains("kiln_graph::ReplayPlan::replay(&mut plan"),
        "CUDA graph replay should execute through ReplayPlan::replay"
    );
    assert!(
        !replay_section.contains("captured.graph.launch()"),
        "CUDA graph replay should not launch the native graph outside ReplayPlan::replay"
    );
}

#[test]
fn rocm_graph_replay_hidden_routes_through_replay_plan_contract() {
    let rocm_graph_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/rocm_graph.rs"))
            .expect("rocm_graph.rs should be readable");
    let replay_hidden_section = source_between(
        &rocm_graph_source,
        "fn replay_hidden(",
        "// --- per-replay in-place buffer refresh",
    );
    assert!(
        rocm_graph_source.contains("struct RocmDecodeReplayPlan")
            && rocm_graph_source.contains("impl ReplayPlan for RocmDecodeReplayPlan"),
        "ROCm graph runner should expose a production ReplayPlan adapter"
    );
    assert!(
        rocm_graph_source.contains("fn replay_state_for_capture(")
            && rocm_graph_source.contains("ReplayState::new(replay_key, resources)")
            && rocm_graph_source.contains("ReplayResourceStability::StableAcrossReplay"),
        "ROCm graph capture should persist shared replay key/resource validation state"
    );
    assert!(
        replay_hidden_section.contains("RocmDecodeReplayPlan::new(captured)")
            && replay_hidden_section.contains("ReplayInputs::new")
            && replay_hidden_section.contains("kiln_graph::ReplayPlan::replay(&mut plan"),
        "ROCm graph replay_hidden should execute through ReplayPlan::replay"
    );
    assert!(
        !replay_hidden_section.contains(".exec\n            .launch("),
        "ROCm graph replay_hidden should not launch the native graph outside ReplayPlan::replay"
    );
    let replay_plan_section = source_between(
        &rocm_graph_source,
        "impl ReplayPlan for RocmDecodeReplayPlan",
        "enum RocmCaptureStep",
    );
    assert!(
        replay_hidden_section.contains("record_event(&captured.replay_inputs_ready_event)")
            && replay_hidden_section.contains("wait_event(&captured.replay_inputs_ready_event)"),
        "ROCm graph replay inputs should use a default-to-capture event dependency"
    );
    assert!(
        replay_plan_section.contains("record_event(&self.captured.replay_complete_event)")
            && replay_plan_section.contains("wait_event(&self.captured.replay_complete_event)"),
        "ROCm graph replay outputs should use a capture-to-default event dependency"
    );
    assert!(
        !replay_hidden_section.contains("rocm_synchronize_default_stream")
            && !replay_plan_section.contains(".synchronize()"),
        "steady-state ROCm graph replay must not host-synchronize either stream"
    );
}

#[test]
fn metal_graph_icb_replay_routes_through_replay_plan_contract() {
    let root = workspace_root();
    let metal_graph_source = fs::read_to_string(root.join("crates/kiln-model/src/metal_graph.rs"))
        .expect("metal_graph.rs should be readable");
    let metal_icb_source =
        fs::read_to_string(root.join("crates/kiln-model/src/backend/metal_icb.rs"))
            .expect("metal_icb.rs should be readable");
    let forward_source = fs::read_to_string(root.join("crates/kiln-model/src/forward.rs"))
        .expect("forward.rs should be readable");
    let icb_attention_section = source_between(
        &forward_source,
        "fn try_metal_paged_decode_icb_attention(",
        "/// Grouped-query attention using a paged KV cache.",
    );

    assert!(
        metal_icb_source.contains("struct MetalPagedDecodeReplayPlan")
            && metal_icb_source.contains("impl ReplayPlan for MetalPagedDecodeReplayPlan"),
        "Metal ICB graph should expose a production ReplayPlan adapter"
    );
    assert!(
        metal_icb_source.contains("metal_paged_decode_replay_state(")
            && metal_icb_source.contains("ReplayState::new(replay_key, resources)")
            && metal_icb_source.contains("ReplayResourceStability::StableAcrossReplay"),
        "Metal ICB graph capture should persist shared replay key/resource validation state"
    );
    assert!(
        metal_graph_source.contains("fn replay_paged_decode_icb_graph_through_replay_plan(")
            && metal_graph_source.contains("ReplayInputs::new")
            && metal_graph_source.contains("kiln_graph::ReplayPlan::replay(&mut plan"),
        "Metal graph runner should execute ICB replay through ReplayPlan::replay"
    );
    assert!(
        icb_attention_section.contains("replay_paged_decode_icb_graph_through_replay_plan("),
        "forward Metal ICB attention should route through the model-level replay-plan helper"
    );
    assert!(
        !icb_attention_section.contains(".replay(max_seqlen_k as u32, softmax_scale)"),
        "forward Metal ICB attention should not call the native ICB replay directly"
    );
}

#[test]
fn vulkan_resident_decode_routes_command_batch_through_replay_plan_contract() {
    let vk_source =
        fs::read_to_string(workspace_root().join("crates/kiln-model/src/vk_decode_resident.rs"))
            .expect("vk_decode_resident.rs should be readable");
    let production = production_source_before_tests(&vk_source);
    let replay_plan_impl = source_between(
        production,
        "impl ReplayPlan for VulkanCommandBatchReplayPlan",
        "fn replay_vulkan_command_batch(",
    );
    let production_outside_replay_plan = production.replacen(replay_plan_impl, "", 1);

    assert!(
        production.contains("struct VulkanCommandBatchReplayPlan")
            && production.contains("impl ReplayPlan for VulkanCommandBatchReplayPlan"),
        "Vulkan resident decode should expose a production ReplayPlan adapter"
    );
    assert!(
        production.contains("ReplayState::new(replay_key.clone(), resources)")
            && production.contains("ReplayResourceStability::StableWithinStep")
            && production.contains("vk_replay_resource("),
        "Vulkan resident decode should persist shared replay key/resource validation state"
    );
    assert!(
        production.contains("ReplayInputs::new")
            && production.contains("kiln_graph::ReplayPlan::replay(&mut plan"),
        "Vulkan resident decode should execute command batches through ReplayPlan::replay"
    );
    assert!(
        replay_plan_impl.contains(".submit_and_wait(self.label)"),
        "Vulkan ReplayPlan adapter should own the native CommandBatch submit"
    );
    assert!(
        !production_outside_replay_plan.contains(".submit_and_wait("),
        "Vulkan resident decode production paths should not submit CommandBatch outside ReplayPlan::replay"
    );
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
    for forbidden in [
        "GITHUB_HEAD_REF",
        "GITHUB_REF_NAME",
        "branch\", \"--show-current",
    ] {
        assert!(
            !script_source.contains(forbidden),
            "capability report generator should not depend on branch-specific metadata: {forbidden}"
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

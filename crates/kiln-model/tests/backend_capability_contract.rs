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
    }

    let replay_fields = descriptors["ReplayRequest"]["fields"]
        .as_array()
        .expect("ReplayRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    assert!(replay_fields.contains(&"dtype"));
    assert!(replay_fields.contains(&"replay_safe"));

    let matmul_blas_fields = descriptors["MatmulBlasRequest"]["fields"]
        .as_array()
        .expect("MatmulBlasRequest fields should be an array")
        .iter()
        .filter_map(|field| field["name"].as_str())
        .collect::<Vec<_>>();
    for field in ["m", "n", "k", "dtype", "epilogue", "concurrent_streams"] {
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
        assert_eq!(info["adamw_step"], adamw, "{backend} AdamW dispatch drifted");
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
        "crates/kiln-train/src/trainer.rs",
    ] {
        assert!(
            evidence.contains(&path),
            "optimizer parity gate should cite {path}"
        );
    }
}

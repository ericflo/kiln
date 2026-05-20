//! Integration tests for `kiln adapter verify` offline receipt validation.

use serde_json::json;

use kiln_server::adapter_verify::{AdapterVerifyOptions, verify_adapter_offline};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write_tiny_adapter(path: &std::path::Path, rank: u64, tensor_rank: usize, zero: bool) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(
        path.join("adapter_config.json"),
        serde_json::to_vec_pretty(&json!({
            "r": rank,
            "lora_alpha": 4.0,
            "target_modules": ["q_proj"],
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        }))
        .unwrap(),
    )
    .unwrap();

    let a_values = if zero {
        vec![0.0; tensor_rank * 3]
    } else {
        vec![0.25; tensor_rank * 3]
    };
    let b_values = if zero {
        vec![0.0; 4 * tensor_rank]
    } else {
        vec![0.5; 4 * tensor_rank]
    };
    let storage = [
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            vec![tensor_rank, 3],
            f32_bytes(&a_values),
        ),
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            vec![4, tensor_rank],
            f32_bytes(&b_values),
        ),
    ];
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = storage
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect();
    let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, view)| (name.as_str(), view.clone()))
        .collect();
    let serialized = safetensors::tensor::serialize(refs, None).unwrap();
    std::fs::write(path.join("adapter_model.safetensors"), serialized).unwrap();
}

fn check_passed(receipt: &kiln_server::adapter_verify::AdapterVerifyReceipt, name: &str) -> bool {
    receipt
        .checks
        .iter()
        .find(|check| check.name == name)
        .is_some_and(|check| check.pass)
}

fn check_message<'a>(
    receipt: &'a kiln_server::adapter_verify::AdapterVerifyReceipt,
    name: &str,
) -> &'a str {
    receipt
        .checks
        .iter()
        .find(|check| check.name == name)
        .map(|check| check.message.as_str())
        .unwrap_or("")
}

#[test]
fn adapter_verify_accepts_known_good_tiny_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = tmp.path().join("good");
    write_tiny_adapter(&adapter, 2, 2, false);

    let receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: adapter.display().to_string(),
        adapter_dir: None,
    });

    assert_eq!(receipt.status, "ok");
    assert!(check_passed(&receipt, "adapter_layout"));
    assert!(check_passed(&receipt, "safetensors_consistency"));
    assert!(check_passed(&receipt, "measurable_adapter_effect"));
    assert_eq!(receipt.lora.rank, Some(2));
    assert_eq!(receipt.lora.alpha, Some(4.0));
    assert_eq!(receipt.lora.alpha_over_rank, Some(2.0));
    assert_eq!(receipt.lora.target_modules, vec!["q_proj"]);
    assert_eq!(receipt.lora.tensor_count, 2);
    assert_eq!(receipt.tensor_summary.paired_projection_count, 1);
    assert!(receipt.files.adapter_model_sha256.unwrap().len() == 64);
    assert!(receipt.logit_delta_summary.measurable);
}

#[test]
fn adapter_verify_rejects_parent_of_nested_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    let parent = tmp.path().join("run-output");
    let nested = parent.join("actual-adapter");
    write_tiny_adapter(&nested, 2, 2, false);

    let receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: parent.display().to_string(),
        adapter_dir: None,
    });

    assert_eq!(receipt.status, "failed");
    assert!(!check_passed(&receipt, "adapter_layout"));
    assert!(check_message(&receipt, "adapter_layout").contains("nested adapter directory"));
}

#[test]
fn adapter_verify_rejects_missing_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = tmp.path().join("missing-weights");
    std::fs::create_dir_all(&adapter).unwrap();
    std::fs::write(
        adapter.join("adapter_config.json"),
        br#"{"r":2,"lora_alpha":4.0,"target_modules":["q_proj"]}"#,
    )
    .unwrap();

    let receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: adapter.display().to_string(),
        adapter_dir: None,
    });

    assert_eq!(receipt.status, "failed");
    assert!(!check_passed(&receipt, "adapter_layout"));
    assert!(check_message(&receipt, "adapter_layout").contains("adapter_model.safetensors"));
}

#[test]
fn adapter_verify_rejects_rank_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = tmp.path().join("rank-mismatch");
    write_tiny_adapter(&adapter, 4, 2, false);

    let receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: adapter.display().to_string(),
        adapter_dir: None,
    });

    assert_eq!(receipt.status, "failed");
    assert!(!check_passed(&receipt, "safetensors_consistency"));
    assert!(check_message(&receipt, "safetensors_consistency").contains("config r=4"));
}

#[test]
fn adapter_verify_rejects_zero_effect_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = tmp.path().join("zero-effect");
    write_tiny_adapter(&adapter, 2, 2, true);

    let receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: adapter.display().to_string(),
        adapter_dir: None,
    });

    assert_eq!(receipt.status, "failed");
    assert!(check_passed(&receipt, "safetensors_consistency"));
    assert!(!check_passed(&receipt, "measurable_adapter_effect"));
    assert!(check_message(&receipt, "measurable_adapter_effect").contains("no measurable"));
}

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{ChatMessage, SftExample, trainer::tokenize_for_training};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    oracle: Oracle,
    mask_contract: MaskContract,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Oracle {
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
}

#[derive(Debug, Deserialize)]
struct MaskContract {
    version: String,
    add_generation_prompt: bool,
    ignore_index: i64,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    messages: Vec<ChatMessage>,
    rendered: String,
    input_ids: Vec<u32>,
    assistant_mask: Vec<u8>,
    labels: Vec<i64>,
}

fn sha256(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn model_path() -> PathBuf {
    std::env::var_os("KILN_QWEN35_MODEL_PATH").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Qwen3.5-4B"),
        PathBuf::from,
    )
}

#[test]
#[ignore = "requires the source-pinned Qwen3.5-4B tokenizer artifacts"]
fn qwen35_hf_token_and_assistant_label_goldens() -> Result<()> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/qwen35_sft_oracle_v1.json"))?;
    ensure!(fixture.schema == "kiln.qwen35-sft-oracle.v1");
    ensure!(fixture.mask_contract.version == "kiln.qwen35-assistant-only.v1");
    ensure!(!fixture.mask_contract.add_generation_prompt);
    ensure!(fixture.mask_contract.ignore_index == -100);

    let model_path = model_path();
    let tokenizer_bytes = std::fs::read(model_path.join("tokenizer.json")).with_context(|| {
        format!(
            "read source-pinned tokenizer from {}; set KILN_QWEN35_MODEL_PATH if needed",
            model_path.display()
        )
    })?;
    let tokenizer_config_bytes = std::fs::read(model_path.join("tokenizer_config.json"))?;
    let template = std::fs::read_to_string(model_path.join("chat_template.jinja"))?;
    ensure!(sha256(&tokenizer_bytes) == fixture.oracle.tokenizer_sha256);
    ensure!(sha256(&tokenizer_config_bytes) == fixture.oracle.tokenizer_config_sha256);
    ensure!(sha256(template.as_bytes()) == fixture.oracle.chat_template_sha256);

    let tokenizer = KilnTokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| anyhow::anyhow!("load Qwen tokenizer: {error}"))?
        .with_chat_template(template);

    for case in fixture.cases {
        let rendered = tokenizer
            .apply_chat_template_for_training(&case.messages)
            .map_err(|error| anyhow::anyhow!("render {}: {error}", case.name))?;
        assert_eq!(rendered, case.rendered, "{} rendered text", case.name);

        let example = SftExample {
            messages: case.messages,
        };
        let (input_ids, label_mask) = tokenize_for_training(&example, &tokenizer)
            .with_context(|| format!("tokenize {}", case.name))?;
        let expected_mask = case
            .assistant_mask
            .iter()
            .map(|value| *value == 1)
            .collect::<Vec<_>>();
        assert_eq!(input_ids, case.input_ids, "{} input IDs", case.name);
        assert_eq!(label_mask, expected_mask, "{} assistant mask", case.name);

        let labels = input_ids
            .iter()
            .zip(&label_mask)
            .map(|(token_id, active)| {
                if *active {
                    i64::from(*token_id)
                } else {
                    fixture.mask_contract.ignore_index
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(labels, case.labels, "{} labels", case.name);
    }

    Ok(())
}

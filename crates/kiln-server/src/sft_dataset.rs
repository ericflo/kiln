use std::{fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{SftInvalidRowPolicy, SftPreparedDataset};

fn open_sft_jsonl(path: &Path) -> Result<BufReader<File>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open SFT dataset_path {}", path.display()))?;
    Ok(BufReader::new(file))
}

/// Parse and tokenize a complete SFT JSONL source through the shared
/// `kiln-train` admission contract. The caller may retain only the receipt at
/// submission time and repeat this operation in the worker to detect mutation.
pub(crate) fn prepare_sft_jsonl(
    path: &Path,
    tokenizer: &KilnTokenizer,
    policy: SftInvalidRowPolicy,
    source: &str,
    source_locator: Option<String>,
) -> Result<SftPreparedDataset> {
    kiln_train::prepare_sft_jsonl(
        open_sft_jsonl(path)?,
        tokenizer,
        policy,
        source,
        source_locator,
    )
    .with_context(|| format!("ingest SFT JSONL {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_ingestion_preserves_agentic_message_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agentic.jsonl");
        let row = serde_json::json!({
            "messages": [
                {"role": "user", "content": "a"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"x\":1}"}
                    }]
                },
                {
                    "role": "tool",
                    "content": [{"type": "text", "text": "a"}],
                    "name": "calculator",
                    "tool_call_id": "call_1"
                },
                {"role": "assistant", "content": "b"}
            ]
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();

        let tokenizer = crate::api::test_tokenizer().with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );
        let prepared = prepare_sft_jsonl(
            &path,
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "dataset_path",
            Some(path.display().to_string()),
        )
        .unwrap();
        assert_eq!(prepared.examples.len(), 1);
        assert_eq!(prepared.examples[0].messages[1].content, "");
        assert_eq!(
            prepared.examples[0].messages[1]
                .tool_calls
                .as_ref()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(prepared.examples[0].messages[2].content, "a");
        assert_eq!(
            prepared.examples[0].messages[2].name.as_deref(),
            Some("calculator")
        );
        assert_eq!(
            prepared.examples[0].messages[2].tool_call_id.as_deref(),
            Some("call_1")
        );
    }
}

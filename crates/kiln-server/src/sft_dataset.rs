use std::{fs::File, io::Read, path::Path};

use anyhow::{Context, Result};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{SftInvalidRowPolicy, SftPreparedDataset};

pub(crate) const MAX_SFT_JSONL_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_SFT_JSONL_ROW_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_SFT_JSONL_ROWS: usize = 100_000;

/// Parse and tokenize a complete SFT JSONL source through the shared
/// `kiln-train` admission contract. The caller queues the owned examples and
/// receipt together, so the worker never reopens a mutable source path.
pub(crate) fn prepare_sft_jsonl(
    path: &Path,
    tokenizer: &KilnTokenizer,
    policy: SftInvalidRowPolicy,
    source: &str,
    source_locator: Option<String>,
) -> Result<SftPreparedDataset> {
    let file = File::open(path)
        .with_context(|| format!("failed to open SFT dataset_path {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect SFT dataset_path {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "SFT dataset_path {} must be a regular file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= MAX_SFT_JSONL_BYTES,
        "SFT dataset_path {} is {} bytes; maximum is {MAX_SFT_JSONL_BYTES}",
        path.display(),
        metadata.len()
    );
    let expected_bytes = metadata.len();
    let mut bytes = Vec::with_capacity(expected_bytes as usize);
    file.take(MAX_SFT_JSONL_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read SFT dataset_path {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_SFT_JSONL_BYTES && bytes.len() as u64 == expected_bytes,
        "SFT dataset_path {} changed while it was being admitted: expected {expected_bytes} bytes, read {}",
        path.display(),
        bytes.len()
    );
    let mut rows = 0usize;
    for (line_index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        anyhow::ensure!(
            line.len() <= MAX_SFT_JSONL_ROW_BYTES,
            "SFT dataset_path {} line {} is {} bytes; maximum row size is {MAX_SFT_JSONL_ROW_BYTES}",
            path.display(),
            line_index + 1,
            line.len()
        );
        if !line.iter().all(u8::is_ascii_whitespace) {
            rows = rows.saturating_add(1);
            anyhow::ensure!(
                rows <= MAX_SFT_JSONL_ROWS,
                "SFT dataset_path {} exceeds the {MAX_SFT_JSONL_ROWS} row limit",
                path.display()
            );
        }
    }
    kiln_train::prepare_sft_jsonl(
        std::io::Cursor::new(bytes),
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

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result, anyhow};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::SftExample;

use crate::training_preflight;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SftJsonlStats {
    pub examples: usize,
    pub max_seq_len: usize,
    pub max_supervised_tokens: usize,
}

fn parse_sft_jsonl_line(path: &Path, line_no: usize, line: &str) -> Result<Option<SftExample>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let example: SftExample = serde_json::from_str(trimmed).with_context(|| {
        format!(
            "invalid SFT JSONL example at line {} in {}",
            line_no,
            path.display()
        )
    })?;
    if example.messages.is_empty() {
        return Err(anyhow!(
            "SFT JSONL example at line {} in {} has no messages",
            line_no,
            path.display()
        ));
    }
    Ok(Some(example))
}

fn open_sft_jsonl(path: &Path) -> Result<BufReader<File>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open SFT dataset_path {}", path.display()))?;
    Ok(BufReader::new(file))
}

pub(crate) fn scan_sft_jsonl_stats(
    path: &Path,
    tokenizer: Option<&KilnTokenizer>,
) -> Result<SftJsonlStats> {
    let reader = open_sft_jsonl(path)?;
    let mut examples = 0usize;
    let mut max_seq_len = 0usize;
    let mut max_supervised_tokens = 0usize;

    for (idx, line) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read SFT dataset_path {} line {}",
                path.display(),
                line_no
            )
        })?;
        let Some(example) = parse_sft_jsonl_line(path, line_no, &line)? else {
            continue;
        };
        examples += 1;
        max_seq_len = max_seq_len.max(training_preflight::approximate_max_seq_len_sft(
            std::slice::from_ref(&example),
            tokenizer,
        ));
        max_supervised_tokens =
            max_supervised_tokens.max(training_preflight::approximate_max_supervised_tokens_sft(
                std::slice::from_ref(&example),
                tokenizer,
            ));
    }

    if examples == 0 {
        return Err(anyhow!(
            "SFT dataset_path {} contains no examples",
            path.display()
        ));
    }

    Ok(SftJsonlStats {
        examples,
        max_seq_len,
        max_supervised_tokens,
    })
}

pub(crate) fn load_sft_jsonl_examples(path: &Path) -> Result<Vec<SftExample>> {
    let reader = open_sft_jsonl(path)?;
    let mut examples = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read SFT dataset_path {} line {}",
                path.display(),
                line_no
            )
        })?;
        if let Some(example) = parse_sft_jsonl_line(path, line_no, &line)? {
            examples.push(example);
        }
    }
    if examples.is_empty() {
        return Err(anyhow!(
            "SFT dataset_path {} contains no examples",
            path.display()
        ));
    }
    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_loader_preserves_agentic_message_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agentic.jsonl");
        let row = serde_json::json!({
            "messages": [
                {"role": "user", "content": "calculate"},
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
                    "content": [{"type": "text", "text": "1"}],
                    "name": "calculator",
                    "tool_call_id": "call_1"
                },
                {"role": "assistant", "content": "done"}
            ]
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();

        let examples = load_sft_jsonl_examples(&path).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].messages[1].content, "");
        assert_eq!(
            examples[0].messages[1].tool_calls.as_ref().unwrap().len(),
            1
        );
        assert_eq!(examples[0].messages[2].content, "1");
        assert_eq!(examples[0].messages[2].name.as_deref(), Some("calculator"));
        assert_eq!(
            examples[0].messages[2].tool_call_id.as_deref(),
            Some("call_1")
        );
    }
}

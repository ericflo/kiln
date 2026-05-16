//! `tokenize_opd_prompts` — apply kiln's chat template + tokenizer to a
//! list of OpdPrompts and emit (tokens, active_positions) JSONL. Used by
//! the OPD fixture builder so the Python-side top-K extraction operates
//! on EXACTLY the token IDs and active positions kiln will compute at
//! train time.
//!
//! Usage:
//!     tokenize_opd_prompts \
//!         --model-path /workspace/kiln/Qwen3.5-4B \
//!         --in datasets/train.opd.jsonl \
//!         --out datasets/train.opd.tokens.jsonl

use anyhow::{Context, Result};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::trainer::tokenize_for_training;
use kiln_train::{ChatMessage, SftExample};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct InRow {
    messages: Vec<MsgRow>,
}

#[derive(serde::Deserialize)]
struct MsgRow {
    role: String,
    content: String,
}

#[derive(serde::Serialize)]
struct OutRow {
    tokens: Vec<u32>,
    active_positions: Vec<usize>,
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mut model_path = None;
    let mut in_path = None;
    let mut out_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-path" => model_path = args.next().map(PathBuf::from),
            "--in" => in_path = args.next().map(PathBuf::from),
            "--out" => out_path = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    Ok((
        model_path.context("--model-path required")?,
        in_path.context("--in required")?,
        out_path.context("--out required")?,
    ))
}

fn main() -> Result<()> {
    let (model_path, in_path, out_path) = parse_args()?;
    let tokenizer_path = model_path.join("tokenizer.json");
    let mut tokenizer = KilnTokenizer::from_file(
        tokenizer_path
            .to_str()
            .context("tokenizer path not utf8")?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tmpl = model_path.join("chat_template.jinja");
    if tmpl.exists() {
        tokenizer = tokenizer.with_chat_template(std::fs::read_to_string(&tmpl)?);
    }

    let raw =
        std::fs::read_to_string(&in_path).with_context(|| format!("read {}", in_path.display()))?;
    let mut out = String::new();
    let mut n_done = 0usize;
    let mut n_skipped = 0usize;
    for (li, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: InRow = serde_json::from_str(line)
            .with_context(|| format!("parse line {}", li + 1))?;
        let messages: Vec<ChatMessage> = row
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let example = SftExample { messages };
        match tokenize_for_training(&example, &tokenizer) {
            Ok((tokens, label_mask)) => {
                let active_positions: Vec<usize> = label_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| if m { Some(i) } else { None })
                    .collect();
                let o = OutRow {
                    tokens,
                    active_positions,
                };
                out.push_str(&serde_json::to_string(&o)?);
                out.push('\n');
                n_done += 1;
            }
            Err(e) => {
                eprintln!("line {}: skipped — {e}", li + 1);
                // Emit an empty placeholder so the indices align with the input.
                out.push_str(&serde_json::to_string(&OutRow {
                    tokens: vec![],
                    active_positions: vec![],
                })?);
                out.push('\n');
                n_skipped += 1;
            }
        }
    }
    std::fs::write(&out_path, out).with_context(|| format!("write {}", out_path.display()))?;
    eprintln!("wrote {} rows ({} skipped) to {}", n_done, n_skipped, out_path.display());
    Ok(())
}

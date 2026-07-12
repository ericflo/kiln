use std::path::PathBuf;

use anyhow::{Context, Result};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::{SftExample, trainer::tokenize_for_training};

fn tokenize_for_stats(example: &SftExample, tokenizer: &KilnTokenizer) -> Result<(usize, usize)> {
    let (input_ids, label_mask) = tokenize_for_training(example, tokenizer)?;
    Ok((
        input_ids.len(),
        label_mask.iter().filter(|active| **active).count(),
    ))
}

fn load_tokenizer(model_dir: &str) -> Result<KilnTokenizer> {
    let tokenizer_path = PathBuf::from(model_dir).join("tokenizer.json");
    let mut tokenizer = KilnTokenizer::from_file(
        tokenizer_path
            .to_str()
            .context("tokenizer path is not valid UTF-8")?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    let template_path = PathBuf::from(model_dir).join("chat_template.jinja");
    if template_path.exists() {
        let template = std::fs::read_to_string(&template_path)
            .with_context(|| format!("reading {}", template_path.display()))?;
        tokenizer = tokenizer.with_chat_template(template);
    }
    Ok(tokenizer)
}

fn percentile(sorted: &[usize], pct: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let file = args
        .next()
        .context("usage: sft_token_stats <jsonl> <model_dir>")?;
    let model_dir = args
        .next()
        .context("usage: sft_token_stats <jsonl> <model_dir>")?;
    let tokenizer = load_tokenizer(&model_dir)?;

    let raw = std::fs::read_to_string(&file).with_context(|| format!("reading {file}"))?;
    let mut token_counts = Vec::new();
    let mut assistant_label_counts = Vec::new();
    let mut no_label = 0usize;

    for (line_idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let example: SftExample = serde_json::from_str(line)
            .with_context(|| format!("parsing JSONL line {}", line_idx + 1))?;
        let (tokens, supervised) = tokenize_for_stats(&example, &tokenizer)
            .with_context(|| format!("tokenizing JSONL line {}", line_idx + 1))?;
        if supervised == 0 {
            no_label += 1;
        }
        token_counts.push(tokens);
        assistant_label_counts.push(supervised);
    }

    token_counts.sort_unstable();
    assistant_label_counts.sort_unstable();
    let total_tokens: usize = token_counts.iter().sum();
    let total_assistant_labels: usize = assistant_label_counts.iter().sum();

    println!("examples={}", token_counts.len());
    println!("no_label_examples={no_label}");
    println!(
        "tokens min={} p50={} p90={} p99={} max={} total={}",
        token_counts.first().copied().unwrap_or(0),
        percentile(&token_counts, 0.50),
        percentile(&token_counts, 0.90),
        percentile(&token_counts, 0.99),
        token_counts.last().copied().unwrap_or(0),
        total_tokens
    );
    println!(
        "assistant_label_tokens min={} p50={} p90={} p99={} max={} total={}",
        assistant_label_counts.first().copied().unwrap_or(0),
        percentile(&assistant_label_counts, 0.50),
        percentile(&assistant_label_counts, 0.90),
        percentile(&assistant_label_counts, 0.99),
        assistant_label_counts.last().copied().unwrap_or(0),
        total_assistant_labels
    );

    Ok(())
}

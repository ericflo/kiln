//! Cross-check the PythonDoctest scorer against an existing GRPO dataset's
//! baked-in rewards. The humaneval-derived datasets in
//! `capabilities/sft/python-algo/datasets/` carry reward labels presumably
//! produced by a similar doctest-based verifier; running our scorer over
//! the same completions should reproduce those labels.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use kiln_eval::scorers::{NoopJudgeRunner, Scorer, score_completion};
use kiln_eval::suite::{EvalChatMessage, EvalExample};

#[derive(serde::Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}
#[derive(serde::Deserialize)]
struct ScoredCompletion {
    text: String,
    reward: f64,
}
#[derive(serde::Deserialize)]
struct GrpoGroup {
    messages: Vec<ChatMessage>,
    completions: Vec<ScoredCompletion>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .context("usage: doctest_cross_check <dataset.jsonl> [max_groups]")?;
    let max_groups: usize = args
        .next()
        .map(|s| s.parse().unwrap_or(usize::MAX))
        .unwrap_or(usize::MAX);

    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let scorer = Scorer::PythonDoctest {
        timeout_seconds: 5.0,
        python_bin: None,
    };
    let runner = NoopJudgeRunner;

    let mut agree = 0usize;
    let mut disagree = 0usize;
    let mut total = 0usize;
    let mut my_pass = 0usize;
    let mut dataset_pass = 0usize;
    let mut invalid = 0usize;

    for (i, line) in BufReader::new(f).lines().enumerate() {
        if i >= max_groups {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let g: GrpoGroup = serde_json::from_str(&line)
            .with_context(|| format!("parsing line {}", i + 1))?;
        let example = EvalExample {
            id: Some(format!("group_{i}")),
            messages: g
                .messages
                .into_iter()
                .map(|m| EvalChatMessage::new(m.role, m.content))
                .collect(),
            target: None,
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: None,
        };
        for (j, c) in g.completions.iter().enumerate() {
            total += 1;
            let outcome = score_completion(&scorer, &example, &c.text, &runner)?;
            let my_passed = matches!(outcome.kind, kiln_eval::result::EvalOutcomeKind::Pass);
            let ds_passed = c.reward >= 0.99;
            if my_passed {
                my_pass += 1;
            }
            if ds_passed {
                dataset_pass += 1;
            }
            if matches!(outcome.kind, kiln_eval::result::EvalOutcomeKind::Invalid) {
                invalid += 1;
            }
            if my_passed == ds_passed {
                agree += 1;
            } else {
                disagree += 1;
                if disagree <= 5 {
                    eprintln!(
                        "disagree group={} comp={} mine={} ds={} reward={} detail={:?}",
                        i, j, my_passed, ds_passed, c.reward, outcome.detail
                    );
                }
            }
        }
    }
    println!(
        "total={total} agree={agree} disagree={disagree} invalid={invalid} my_pass={my_pass} dataset_pass={dataset_pass} agreement={:.2}%",
        100.0 * agree as f32 / total.max(1) as f32
    );
    Ok(())
}

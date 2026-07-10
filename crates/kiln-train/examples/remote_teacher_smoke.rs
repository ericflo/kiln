//! Smoke test: query a running vLLM `/v1/completions` server through
//! kiln's `RemoteTeacher` and print the top-K logprobs at a few
//! positions.
//!
//! Run with:
//! ```bash
//! cargo run -p kiln-train --release --example remote_teacher_smoke -- \
//!   http://localhost:8002 qwen3.6-27b-fp8 8
//! ```
//!
//! Args: `[vllm_url] [model_id] [top_k]`. The authoritative vocabulary size,
//! top-K cap, tokenizer identity, and model revision are discovered from the
//! server rather than supplied by the operator.

use kiln_train::logit_source::{LogitSource, LogprobBatch};
use kiln_train::{RemoteProvider, RemoteTeacher, RemoteTeacherConfig, discover_vllm_identity};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .unwrap_or_else(|| "http://localhost:8002".to_string());
    let model = args.next().unwrap_or_else(|| "qwen3.6-27b-fp8".to_string());
    let requested_top_k = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    anyhow::ensure!(args.next().is_none(), "expected at most three arguments");

    let mut cfg = RemoteTeacherConfig {
        provider: RemoteProvider::Vllm,
        model: model.clone(),
        url: url.clone(),
        api_key_env: None,
        teacher_id: format!("vllm/{model}"),
        expected_identity: None,
        tokenizer_hash: None,
        max_top_k: 0,
        vocab_size: 0,
        max_cost_usd: None,
        timeout_ms: 60_000,
    };

    let identity = discover_vllm_identity(&cfg)?;
    let top_k = requested_top_k.unwrap_or_else(|| (identity.max_top_k() as usize).min(8));
    anyhow::ensure!(top_k > 0, "top_k must be greater than zero");
    anyhow::ensure!(
        top_k <= identity.max_top_k() as usize,
        "requested top_k {top_k} exceeds the verified server cap {}",
        identity.max_top_k()
    );
    println!("Verified teacher revision: {}", identity.content_revision());
    println!("  served model:      {}", identity.served_model_id());
    println!("  implementation:    {}", identity.implementation());
    println!("  vocabulary size:   {}", identity.vocab_size());
    println!("  tokenizer SHA-256: {}", identity.tokenizer_vocab_sha256());
    println!("  maximum top-K:     {}", identity.max_top_k());
    println!("  maximum model len: {}", identity.max_model_len());

    cfg.expected_identity = Some(identity);
    let teacher = RemoteTeacher::new(cfg)?;
    let caps = teacher.capabilities();
    println!("RemoteTeacher capabilities: {caps:?}");

    // Token ID zero is valid for the numeric model-input protocol and keeps
    // this transport smoke independent of any one model's natural-language
    // vocabulary.
    let tokens: Vec<u32> = vec![0; 7];
    let positions: Vec<usize> = vec![0, 3, tokens.len() - 1];

    let batch = teacher.fetch_logprobs(&tokens, &positions, Some(top_k))?;
    match batch {
        LogprobBatch::TopK(tk) => {
            println!(
                "TopK: positions={} top_k={} indices.len()={} logprobs.len()={}",
                positions.len(),
                tk.top_k,
                tk.indices.len(),
                tk.logprobs.len()
            );
            for (i, &pos) in positions.iter().enumerate() {
                println!("  position {pos}:");
                for j in 0..tk.top_k {
                    let idx = tk.indices[i * tk.top_k + j];
                    let lp = tk.logprobs[i * tk.top_k + j];
                    println!("    id={idx:>6}  lp={lp:>8.4}");
                }
            }
        }
        other => anyhow::bail!("unexpected batch variant: {other:?}"),
    }

    Ok(())
}

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
//! Args: <vllm_url> <model_id> <top_k>

use kiln_train::logit_source::{LogitSource, LogprobBatch};
use kiln_train::{RemoteProvider, RemoteTeacher, RemoteTeacherConfig};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let url = args.next().unwrap_or_else(|| "http://localhost:8002".to_string());
    let model = args.next().unwrap_or_else(|| "qwen3.6-27b-fp8".to_string());
    let top_k: usize = args.next().unwrap_or_else(|| "8".to_string()).parse()?;

    let cfg = RemoteTeacherConfig {
        provider: RemoteProvider::Vllm,
        model: model.clone(),
        url: url.clone(),
        api_key_env: None,
        teacher_id: format!("vllm/{model}"),
        tokenizer_hash: None,
        max_top_k: top_k,
        vocab_size: 0,
        max_cost_usd: None,
        timeout_ms: 60_000,
    };
    let teacher = RemoteTeacher::new(cfg);
    let caps = teacher.capabilities();
    println!("RemoteTeacher capabilities: {caps:?}");

    // A few arbitrary Qwen vocab ids — "Hello world<assistant turn marker>"-ish.
    // The point isn't semantics, just that vLLM produces top-K logprobs
    // at every prompt position we ask about.
    let tokens: Vec<u32> = vec![9707, 1879, 2, 30246, 11, 1246, 525];
    let positions: Vec<usize> = vec![1, 3, 5];

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
        other => println!("unexpected batch variant: {other:?}"),
    }

    Ok(())
}

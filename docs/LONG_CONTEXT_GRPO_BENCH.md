# Long-Context GRPO Benchmark

`crates/kiln-train/examples/long_context_grpo_bench.rs` measures synthetic
GRPO training shaped like pi-compaction workloads: a short action, a long tool
observation, and a final compact answer. The default length sweep is 8K, 16K,
32K, and 64K tokens per completion.

Dry CPU/tokenization run:

```bash
cargo run -p kiln-train --example long_context_grpo_bench -- \
  --dry-run \
  --lengths 1024,8192 \
  --output /tmp/long-context-grpo-dry.json
```

Pass `--model /workspace/Qwen3.5-4B` to use the real Qwen tokenizer. When
`--model` is omitted in dry mode, the benchmark uses a built-in byte tokenizer
with a Qwen-shaped chat template so normal CPU-only dev environments can run a
small smoke without model assets.

CUDA one-step GRPO run:

```bash
KILN_CUDA_ARCHS=86 cargo run --release -p kiln-train --features cuda \
  --example long_context_grpo_bench -- \
  --model /workspace/Qwen3.5-4B \
  --cuda \
  --lengths 8192,16384,32768,65536 \
  --segments 4 \
  --output /tmp/long-context-grpo-cuda.json
```

Each result is emitted as one JSON line on stdout. `--output` additionally
writes the full record array. Records include:

- `report.timings.tokenize_ms`
- `report.timings.mask_build_ms`
- `report.timings.reference_forward_ms`
- `report.timings.policy_forward_ms`
- `report.timings.backward_ms`
- `report.timings.optimizer_ms`
- `peak_vram_mib`
- `kernel_launch_count` (`null` until a launch counter is wired)
- `report.tokens_per_sec`

Use `--lengths` for shorter smoke runs in CI or development environments. Use
`--segments 0` to disable checkpointing for small CUDA comparisons; leave
checkpointing enabled for compaction-like long contexts.

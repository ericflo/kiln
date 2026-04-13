# Kiln

A single-model LLM inference server with live online learning. Train while you serve.

## What is this?

Kiln is an inference server purpose-built for one model. The entire inference stack is tuned to that model — no abstractions, no multi-model support, no plugin system. Just fast, long-context inference with one killer feature: **you can submit training examples (SFT or GRPO) and the server immediately starts serving the updated weights via LoRA hot-swap.**

The idea: you run this on your home server, collect examples of failures, submit corrections or scored completions, and get an improved model within seconds — all without restarting the server or managing separate training infrastructure.

## Target Model

**Qwen3-4B** (pure transformer, GQA, 128K extended context) for Phase 1. Qwen3.5-4B (hybrid linear attention, 262K native context) planned for Phase 2 once framework support matures.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  KILN SERVER                     │
│                                                  │
│  HTTP API (axum)                                 │
│    ├── /v1/chat/completions  (OpenAI-compat)    │
│    ├── /v1/models                                │
│    ├── /v1/adapters          (LoRA management)   │
│    ├── /v1/train/sft         (submit examples)   │
│    └── /v1/train/grpo        (submit scored)     │
│                                                  │
│  Scheduler (continuous batching, chunked prefill)│
│    └── Block Manager (paged KV cache)            │
│                                                  │
│  Engine (model forward pass + LoRA application)  │
│                                                  │
│  Training Sidecar (Python, shared GPU memory)    │
│    ├── SFT (cross-entropy on LoRA)              │
│    └── GRPO (group relative policy optimization) │
└─────────────────────────────────────────────────┘
```

- **Inference**: Rust (axum HTTP server, iteration-level scheduler, paged KV cache)
- **Training**: Python sidecar (PyTorch/PEFT, communicates via unix socket)
- **LoRA hot-swap**: New adapter weights are swapped atomically at iteration boundaries

## Key Design Decisions

- **Single model, fully tuned.** No runtime model switching. The scheduler, memory management, and kernels are all optimized for one architecture.
- **Continuous batching with chunked prefill.** Sarathi-style: decode requests are never stalled by long prefills. The prefill is chunked across iterations.
- **Paged KV cache.** Virtual memory-style block allocation eliminates fragmentation. Each request gets a block table mapping logical positions to physical blocks.
- **LoRA-only training.** All learning happens through LoRA adapters. The base model weights are never modified. This means changes are reversible, composable, and cheap.
- **GRPO as a first-class citizen.** Submit a batch of scored completions and the server runs a GRPO update — no separate training pipeline needed. The client controls the reward function.

## The GRPO Loop

```python
# 1. Generate completions
responses = client.generate_batch(prompts, n=8, temperature=0.7)

# 2. Score them (your reward function — regex, code exec, another model, human)
for r in responses:
    r.reward = my_reward_fn(r.text)

# 3. Submit GRPO update — server trains and hot-swaps immediately
client.train_grpo(responses)

# 4. Next inference call uses updated weights
answer = client.chat("What is 2+2?")
```

## Memory Budget (24GB GPU)

Kiln targets **128K context** (131,072 tokens) as the default. Qwen3-4B uses GQA with 8 KV heads, so KV cache is ~144 KB/token in BF16.

| Scenario | Weights | KV Cache | LoRA + Training | Total |
|---|---|---|---|---|
| **128K ctx, 1 seq, inference only** | 8 GB | ~18 GB | — | ~26 GB ⚠️ |
| **128K ctx, 1 seq, inference + FP8 KV** | 8 GB | ~9 GB | — | ~17 GB ✓ |
| **64K ctx, 2 seq, inference only** | 8 GB | ~18 GB | — | ~26 GB ⚠️ |
| **32K ctx, 4 seq, inference + training** | 8 GB | ~4.5 GB | ~5 GB | ~18 GB ✓ |
| **32K ctx, 4 seq, INT4 weights + training** | 2.5 GB | ~4.5 GB | ~5 GB | ~12 GB ✓ |

Full 128K on a 24GB GPU requires FP8 KV cache quantization (Phase 6) or INT4 model weights. Until then, 32K-64K context with multiple concurrent sequences is the sweet spot. The scheduler and block manager are designed for the full 131,072 from day one — the only constraint is VRAM.

## Project Structure

```
crates/
  kiln-core/       Core types: block manager, config, request, sampling
  kiln-model/      Engine trait, model loading, LoRA management
  kiln-scheduler/  Continuous batching scheduler with chunked prefill
  kiln-server/     HTTP server (axum), OpenAI-compatible API
  kiln-train/      Training API types and sidecar coordination
```

## Status

**Phase 1 (in progress):** Scaffold, scheduler, HTTP API with mock engine.

- [x] Cargo workspace with 5 crates
- [x] Block manager with paged KV cache
- [x] Iteration-level scheduler with chunked prefill
- [x] OpenAI-compatible chat completions API
- [x] Training API types (SFT + GRPO)
- [x] LoRA adapter management API
- [ ] Real model loading (safetensors, Qwen3 architecture)
- [ ] Real inference engine (candle or CUDA)
- [ ] Tokenizer integration
- [ ] SSE streaming

**Phase 2:** LoRA serving and hot-swap  
**Phase 3:** Python training sidecar (SFT)  
**Phase 4:** GRPO training  
**Phase 5:** Polish (quantization, adapter merging, web UI, client libraries)

## Building

```bash
cargo build --release
```

## Running

```bash
# Start with mock engine (for development/testing)
KILN_PORT=8420 cargo run --release --bin kiln

# Test
curl http://localhost:8420/health
curl http://localhost:8420/v1/models
curl -X POST http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Prior Art

- [Tinker API](https://thinkingmachines.ai/blog/announcing-tinker/) by Thinking Machines Lab — cloud-hosted LoRA training + serving. Kiln is the self-hosted, single-model, open-source version of this idea.
- [vLLM](https://github.com/vllm-project/vllm) — production inference server with PagedAttention. Kiln borrows the paged KV cache and continuous batching concepts.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — 1200-line reference implementation of continuous batching. Proves the core can be simple.
- [S-LoRA](https://arxiv.org/abs/2311.03285) / [LoRAX](https://github.com/predibase/lorax) — multi-LoRA serving. Kiln's LoRA serving approach draws from these.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — introduced GRPO. Kiln implements GRPO as a first-class training method.

## License

MIT

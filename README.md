<p align="center">
  <img src="assets/logo.png" alt="Kiln" width="200">
</p>

<h1 align="center">Kiln</h1>

<p align="center">
  A single-model LLM inference server with live online learning.<br>
  <strong>Train while you serve.</strong>
</p>

## What is this?

Kiln is an inference server purpose-built for one model. The entire inference stack is tuned to that model — no abstractions, no multi-model support, no plugin system. Just fast, long-context inference with one killer feature: **you can submit training examples (SFT or GRPO) and the server immediately starts serving the updated weights via LoRA hot-swap.**

The idea: you run this on your home server, collect examples of failures, submit corrections or scored completions, and get an improved model within seconds — all without restarting the server or managing separate training infrastructure.

## Target Model

**Qwen3.5-4B** — a hybrid architecture combining Gated DeltaNet linear attention with standard GQA attention. This is the ideal model for Kiln because:

- **262K native context** — long-context is a first-class capability, not an afterthought
- **Hybrid attention** — 24 of 32 layers use linear attention (O(1) state), only 8 use full GQA attention. KV cache is ~32 KB/token instead of ~128 KB/token. This is what makes 128K+ context practical on a single 24GB GPU.
- **4B parameters** — large enough to be genuinely capable, small enough for consumer hardware. LoRA training completes in seconds, not hours.
- **248K vocabulary** — 202+ language support out of the box
- **Built-in vision encoder** — future path to multimodal fine-tuning

| Spec | Value |
|---|---|
| Layers | 32 (24 linear + 8 full attention) |
| Hidden size | 2560 |
| Attention heads | 16 Q / 4 KV (GQA, 4:1 ratio) |
| Head dim | 256 |
| FFN intermediate | 9216 |
| Context length | 262,144 tokens (native) |
| KV cache / token | ~32 KB (BF16, 8 full-attn layers only) |

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
│    ├── 24× Gated DeltaNet layers (linear attn)  │
│    └──  8× GQA layers (full attention + KV)     │
│                                                  │
│  In-Process Training (pure Rust, candle)          │
│    ├── SFT (cross-entropy on LoRA)              │
│    └── GRPO (group relative policy optimization) │
└─────────────────────────────────────────────────┘
```

- **Inference**: Rust (axum HTTP server, iteration-level scheduler, paged KV cache)
- **Training**: In-process LoRA training (pure Rust via candle, no Python dependency)
- **LoRA hot-swap**: New adapter weights are swapped atomically at iteration boundaries

## Key Design Decisions

- **Single model, fully tuned.** No runtime model switching. The scheduler, memory management, and kernels are all optimized for Qwen3.5-4B's hybrid architecture.
- **Hybrid attention aware.** The engine knows which layers are linear (Gated DeltaNet) and which are full attention (GQA). KV cache is only allocated for the 8 full-attention layers. Linear layers maintain fixed-size recurrent state.
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

Qwen3.5-4B's hybrid architecture is the key enabler. Because only 8 of 32 layers need KV cache, the memory footprint at long context is ~4× lower than a pure transformer.

| Scenario | Weights | KV Cache | LoRA + Training | Total |
|---|---|---|---|---|
| **128K ctx, 1 seq, inference only** | 8 GB | ~4 GB | — | ~13 GB ✓ |
| **128K ctx, 1 seq, inference + training** | 8 GB | ~4 GB | ~5 GB | ~18 GB ✓ |
| **128K ctx, 4 seq, inference only** | 8 GB | ~16 GB | — | ~25 GB ⚠️ |
| **64K ctx, 4 seq, inference + training** | 8 GB | ~8 GB | ~5 GB | ~22 GB ✓ |
| **32K ctx, 8 seq, inference + training** | 8 GB | ~8 GB | ~5 GB | ~22 GB ✓ |
| **128K ctx, 4 seq, INT4 weights** | 2.5 GB | ~16 GB | — | ~19 GB ✓ |

**128K context with a single sequence fits on a 24GB GPU with room to spare for training.** This is the hybrid architecture payoff — what would require 26+ GB on a pure transformer needs only 13 GB here.

## Project Structure

```
crates/
  kiln-core/       Core types: block manager, config, request, sampling
  kiln-model/      Engine trait, model loading, LoRA management
  kiln-scheduler/  Continuous batching scheduler with chunked prefill
  kiln-server/     HTTP server (axum), OpenAI-compatible API
  kiln-train/      In-process LoRA training (pure Rust, candle)
```

## Status

**Phase 1 (in progress):** Scaffold, scheduler, HTTP API with mock engine.

- [x] Cargo workspace with 5 crates
- [x] Block manager with paged KV cache
- [x] Iteration-level scheduler with chunked prefill
- [x] OpenAI-compatible chat completions API
- [x] Training API types (SFT + GRPO)
- [x] LoRA adapter management API
- [ ] Real model loading (safetensors, Qwen3.5 architecture)
- [ ] Real inference engine (hybrid linear + full attention)
- [ ] Tokenizer integration
- [ ] SSE streaming

**Phase 2:** LoRA serving and hot-swap  
**Phase 3:** In-process LoRA training (SFT)  
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

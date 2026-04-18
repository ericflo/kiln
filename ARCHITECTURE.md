# Architecture Deep-Dive

This document explains how Kiln works internally. It is aimed at contributors and power users who want to understand what happens between an HTTP request arriving and a token being generated — or a model being trained.

## System Overview

Kiln is a single Rust binary built as a Cargo workspace with six crates:

```
kiln
├── kiln-core          Core types: block manager, KV cache config, tokenizer, request lifecycle
├── kiln-flash-attn    Vendored Flash-Attention-2 CUDA kernels with thin C-ABI/Rust FFI
├── kiln-model         Model loading, forward pass, LoRA, sampling, KV cache, CUDA graphs
├── kiln-scheduler     Sarathi-style continuous batching scheduler with prefix caching
├── kiln-server        Axum HTTP server, CLI, training queue, metrics, configuration
└── kiln-train         SFT and GRPO training loops with gradient checkpointing
```

The dependency graph flows downward:

```
kiln-server
├── kiln-model
│   ├── kiln-core
│   └── kiln-flash-attn
├── kiln-scheduler
│   └── kiln-core
└── kiln-train
    ├── kiln-core
    └── kiln-model
```

Everything runs in a single OS process. Inference and training share the same GPU memory and model weights. There is no Python sidecar, no second model copy, no separate training service.

### Startup Flow

1. Parse CLI args and load `kiln.toml` config (merged with env vars and defaults)
2. Initialize structured logging (JSON or pretty, configurable level)
3. Load tokenizer from file, HuggingFace Hub, or model directory
4. **Real mode** (model path provided): load safetensors weights, transfer to GPU, detect VRAM, auto-size KV cache blocks, create `ModelRunner`
5. **Mock mode** (no model path): create scheduler with mock engine for testing
6. Spawn training queue worker (background tokio task)
7. Bind Axum HTTP server, register signal handlers for graceful shutdown

See `crates/kiln-server/src/main.rs` for the full startup sequence.

## Inference Pipeline

The journey from HTTP request to generated text:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    kiln-server                           │
                              │                                                         │
  HTTP Request ──────────────►│  POST /v1/chat/completions                              │
  (OpenAI-compatible)         │       │                                                 │
                              │       ▼                                                 │
                              │  Apply chat template ──► Tokenize prompt                │
                              │       │                                                 │
                              │       ▼                                                 │
                              │  ensure_adapter() ──► Two-phase LoRA load if needed     │
                              │       │                                                 │
                              │       ▼                                                 │
                              │  ┌─────────────────────────────────────────────────┐    │
                              │  │  Acquire GPU read lock (concurrent inference OK) │    │
                              │  │                                                  │    │
                              │  │  ModelRunner::generate_paged()                   │    │
                              │  │       │                                          │    │
                              │  │       ▼                                          │    │
                              │  │  ┌────────────────────────┐                      │    │
                              │  │  │  PREFILL               │                      │    │
                              │  │  │  Embed prompt tokens   │                      │    │
                              │  │  │  Forward through 32    │                      │    │
                              │  │  │  layers (GDN + GQA)    │                      │    │
                              │  │  │  Write KV cache        │                      │    │
                              │  │  │  Sample first token    │                      │    │
                              │  │  └────────┬───────────────┘                      │    │
                              │  │           │                                      │    │
                              │  │           ▼                                      │    │
                              │  │  ┌────────────────────────┐                      │    │
                              │  │  │  DECODE (loop)         │◄──── CUDA Graph      │    │
                              │  │  │  Embed 1 token         │      Replay          │    │
                              │  │  │  Forward through 32    │      (after warmup)   │    │
                              │  │  │  layers, read KV cache │                      │    │
                              │  │  │  Sample next token     │                      │    │
                              │  │  │  Check stop conditions │                      │    │
                              │  │  └────────┬───────────────┘                      │    │
                              │  │           │ (EOS / max_tokens / stop sequence)    │    │
                              │  │           ▼                                      │    │
                              │  │  Return generated text + usage stats             │    │
                              │  └──────────────────────────────────────────────────┘    │
                              │       │                                                 │
                              │       ▼                                                 │
  HTTP Response ◄─────────────│  JSON response (or SSE stream)                          │
  (or SSE stream)             └─────────────────────────────────────────────────────────┘
```

**Streaming** uses a tokio mpsc channel: a blocking thread runs generation and sends tokens as they are produced, while an async task forwards them as SSE events. A timeout races against the channel receiver for cancellation.

See `crates/kiln-server/src/api/completions.rs` for the HTTP handler and `crates/kiln-model/src/generate.rs` for the generation loop.

### Scheduler: Sarathi-Style Chunked Prefill

The scheduler (`crates/kiln-scheduler/src/scheduler.rs`) implements continuous batching with a strict priority order per iteration:

1. **Decode requests first** — each active decode request gets exactly 1 token. These have absolute priority because stalling a decode request means latency for a waiting user.
2. **Continue partial prefills** — if a prefill was chunked (prompt too large for one iteration's token budget), continue it with remaining budget.
3. **Start new prefills** — promote waiting requests and begin their prefill, chunking to fit the remaining budget.

The token budget (`max_batch_tokens`, default 8192) caps the total tokens per forward pass. A 50-token prompt with a budget of 30 gets split into two chunks (30 + 20). During the first chunk, any active decode requests still get their 1-token slot.

```
Iteration token budget: 8192
┌──────────────────────────────────────────────────────────┐
│ Decode tokens (1 each) │ Partial prefill │ New prefills  │
│   Highest priority     │   Continue      │  Remaining    │
│                        │   chunked work  │  budget       │
└──────────────────────────────────────────────────────────┘
```

**Request state machine:**

```
Waiting ──► Prefilling(tokens_processed) ──► Decoding ──► Complete
                                     │
               (if all tokens cached) └──────► Decoding ──► Complete
```

## Memory Management

### Paged KV Cache

Kiln uses paged virtual memory for the KV cache, inspired by vLLM's PagedAttention. Physical memory is divided into fixed-size blocks (default 16 tokens each). Each request gets a **block table** that maps logical token positions to physical block slots.

```
                         Block Table (per request)
                         ┌──────┬──────┬──────┬──────┐
  Logical block index:   │  0   │  1   │  2   │  3   │
                         ├──────┼──────┼──────┼──────┤
  Physical block ID:     │  7   │  23  │  4   │  15  │
                         └──┬───┴──┬───┴──┬───┴──┬───┘
                            │      │      │      │
                            ▼      ▼      ▼      ▼
  Physical Block Pool:  ┌──────────────────────────────────┐
  [total_slots × num_kv_heads × head_dim]                  │
  │ block 0 │ ... │ block 4 │ ... │ block 7 │ ... │ ...    │
  └──────────────────────────────────────────────────────────┘

  Address translation:
    token_pos 35 (block_size=16) → logical block 2 → physical block 4 → offset 3
    slot = physical_block_id × block_size + offset = 4 × 16 + 3 = 67
```

The `BlockManager` (`crates/kiln-core/src/block.rs`) maintains a FIFO free list. Allocation and deallocation are O(1). The `PagedKvCache` (`crates/kiln-model/src/paged_kv_cache.rs`) stores the actual K and V tensors in a pre-allocated GPU pool shaped `[total_slots, num_kv_heads, head_dim]`.

**Why paging matters:** Without paging, each request would need a contiguous allocation sized for `max_sequence_length`. With paging, memory is allocated incrementally as tokens are generated. Multiple concurrent requests share the same physical pool with no fragmentation.

### FP8 KV Cache Quantization

The KV cache can optionally use FP8 (E4M3FN) format — 1 byte per element instead of 2 bytes for BF16. This halves KV cache memory, doubling the effective context length or concurrent request capacity.

- **128K context in BF16:** ~4 GB KV cache (only 8 layers need KV cache)
- **128K context in FP8:** ~2 GB KV cache

FP8 quantization is per-tensor with absmax scaling. Roundtrip error is ~5-10%, which is acceptable for attention computations. See `crates/kiln-model/src/fp8.rs`.

### Prefix Caching

When multiple requests share a common prompt prefix (e.g., a system prompt), the prefix cache (`crates/kiln-core/src/prefix_cache.rs`) avoids recomputing KV entries. It uses **hash chaining** for position-aware matching:

```
Block 0 hash: H(0, tokens[0..16])
Block 1 hash: H(block_0_hash, tokens[16..32])
Block 2 hash: H(block_1_hash, tokens[32..48])
```

The same tokens at different positions produce different hashes. Lookups walk the chain until a hash miss. Cached blocks are reference-counted and LRU-evicted when the cache is full.

### VRAM Budget

At startup, Kiln detects GPU memory via `nvidia-smi` (overridable with `KILN_GPU_MEMORY_GB`) and auto-configures:

| GPU VRAM | KV Cache Blocks | Grad Checkpoint Segments |
|----------|----------------|------------------------|
| >= 45 GB | 512 | 4 |
| >= 22 GB | 64 | 8 |
| >= 14 GB | 32 | 12 |

See `crates/kiln-core/src/vram.rs` for the recommendation logic.

## Attention Architecture

Kiln targets Qwen3.5-4B, a hybrid transformer with two attention mechanisms:

```
Layer 0:  Gated DeltaNet (linear)  ─┐
Layer 1:  Gated DeltaNet (linear)   │ 24 linear attention layers
Layer 2:  Gated DeltaNet (linear)   │ O(1) recurrent state per layer
Layer 3:  Full GQA Attention     ◄──┤ Every 4th layer
Layer 4:  Gated DeltaNet (linear)   │
...                                 │
Layer 30: Gated DeltaNet (linear)   │
Layer 31: Full GQA Attention     ◄──┘ 8 full attention layers total
```

### Full GQA Attention (8 layers)

Standard grouped-query attention with 16 query heads sharing 4 KV heads (group size = 4). Each layer:

1. RMSNorm on Q and K (QK-norm)
2. Rotary position embeddings (RoPE) on first 64 of 256 head dimensions
3. FlashAttention-2 CUDA kernel (prefill) or paged attention (decode)
4. Optional output gate: `output * sigmoid(gate)` — enabled for Qwen3.5-4B
5. Output projection with optional LoRA delta

Only these 8 layers need KV cache. This is the key architectural advantage: KV cache memory scales with 8 layers instead of 32, enabling 128K context in ~4 GB.

### Gated DeltaNet Linear Attention (24 layers)

Each GDN layer maintains a fixed-size recurrent state matrix `S` of shape `[batch, num_value_heads, key_head_dim, value_head_dim]` — independent of sequence length. The forward pass per timestep:

```
Input: x (hidden state)
  │
  ├─► in_proj_qkv(x) ──► Causal Conv1d ──► SiLU ──► Split into Q, K, V
  ├─► in_proj_a(x)   ──► Compute decay gate γ = -exp(A_log) * softplus(a + dt_bias)
  ├─► in_proj_b(x)   ──► Compute write gate β = sigmoid(b)
  └─► in_proj_z(x)   ──► Output gate z
         │
         ▼
  L2-normalize Q, K; scale Q by 1/√dk
         │
         ▼
  Sequential recurrence (per timestep t):
    S *= exp(γ_t)                           ← Decay old state
    memory = S @ k_t                        ← Read from state
    delta = (v_t - memory) * β_t            ← Delta rule update
    S += k_t ⊗ delta                        ← Write to state
    output_t = S @ q_t                      ← Query state
         │
         ▼
  Gated RMSNorm: norm(output) * silu(z)
         │
         ▼
  Output projection ──► Residual connection
```

The causal Conv1d uses a sliding window of size 4. During prefill, it processes the full sequence. During decode, it maintains a small buffer of the last `kernel_size - 1` values.

See `crates/kiln-model/src/forward.rs` — functions `gated_deltanet_forward()`, `causal_conv1d_prefill()`, `causal_conv1d_decode()`.

### FlashAttention-2 Integration

Kiln vendors the Flash-Attention-2 CUDA kernels directly, with no PyTorch dependency. The `kiln-flash-attn` crate (`crates/kiln-flash-attn/`) provides:

- A thin C-ABI wrapper over the core CUDA source files
- Rust FFI bindings for forward and backward passes
- Support for BF16, head dimensions 128 and 256, causal masking
- Forward returns `softmax_lse` (log-sum-exp) needed by the backward pass

The build uses `cc` crate to compile CUDA via nvcc with CUTLASS headers. The instantiation matrix is trimmed to only what Qwen3.5-4B needs (BF16, hdim128/256, causal).

## Backend Abstraction

Most of `kiln-model`'s forward pass is expressed as portable `candle_core::Tensor` ops that run on any candle device. A small set of ops with no candle equivalent sits behind the `BackendRuntime` trait in `kiln-model::backend`. The trait abstracts four platform-specific kernel ops: FlashAttention-2 prefill, FlashAttention-2 paged decode, Gated DeltaNet chunkwise forward-substitution, and the Gated DeltaNet single-token recurrent step. Each method returns `Result<Option<Tensor>>`: returning `Ok(None)` means the backend declines the call, and the caller falls back to a portable candle-op path. This keeps `#[cfg(feature = "...")]` gates out of every call site in `forward.rs`, `generate.rs`, `paged_kv_cache.rs`, and the training loop.

Three backends implement the trait. `CudaBackend` dispatches to the vendored `kiln-flash-attn` kernels plus the `kiln-gdn-kernel` fused recurrent/forward-substitution kernels. `MetalBackend` uses candle's native `scaled_dot_product_attention` on Apple Silicon and declines the GDN ops (the portable candle path handles them). `CpuBackend` declines every op and routes all work through the portable fallback — used in mock mode and on platforms without a GPU feature enabled.

Backend selection is build-time via Cargo features: `--features cuda` pulls in `CudaBackend`, `--features metal` pulls in `MetalBackend`, and omitting both yields the CPU-only fallback. At runtime, `backend::for_device()` picks the concrete backend for the active candle `Device`.

Reference: `crates/kiln-model/src/backend/mod.rs`.

## LoRA System

### Adapter Loading

LoRA adapters use the PEFT-compatible format:

```
adapters/
└── my-adapter/
    ├── adapter_config.json      # rank, alpha, target_modules
    └── adapter_model.safetensors  # LoRA A and B matrices
```

Loading parses PEFT key names (`base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight`) and maps them to per-layer, per-module weight pairs. See `crates/kiln-model/src/lora_loader.rs`.

### Per-Request Adapter Selection

Each inference request can specify an adapter via the `adapter` field (a Kiln extension to the OpenAI API). The server ensures the correct adapter is loaded before generation begins.

### Hot-Swap at Iteration Boundary

Adapter swapping uses a two-phase RwLock pattern to avoid blocking inference during I/O:

```
Phase 1: Brief read lock → extract device info, layer count
         Drop read lock immediately

Phase 2: Load weights from disk (no lock held)
         This is the slow part — disk I/O, tensor conversion

Phase 3: Brief write lock → atomically swap adapter weights
         This is just a pointer swap — sub-millisecond
```

CUDA graphs are invalidated on adapter swap because weight tensor pointers change. See `crates/kiln-model/src/lora.rs` and the `ensure_adapter()` function in `crates/kiln-server/src/api/completions.rs`.

### LoRA Delta Application

During the forward pass, LoRA deltas are computed as:

```
delta = x @ A^T @ B^T × (alpha / rank)
output = base_output + delta
```

Applied to: q_proj, k_proj, v_proj, o_proj (attention), gate_proj, up_proj, down_proj (FFN). If no LoRA exists for a module, the delta computation is skipped. See `compute_lora_delta()` and `linear_with_lora()` in `crates/kiln-model/src/forward.rs`.

### Adapter Management API

```
GET    /v1/adapters              List active + available adapters
POST   /v1/adapters/load         Load adapter from disk
POST   /v1/adapters/unload       Revert to base model
POST   /v1/adapters/merge        Merge multiple adapters via weighted average
DELETE /v1/adapters/{name}       Delete adapter from disk
```

See `crates/kiln-server/src/api/adapters.rs`.

### Adapter Merging

Multiple PEFT adapters that share the same base model, rank, and target modules can be combined via linear interpolation:

```
merged = Σᵢ wᵢ · adapter_i        # element-wise on every (A, B) tensor
```

Request:

```json
POST /v1/adapters/merge
{
  "mode": "weighted_average",
  "sources": [
    {"name": "code-fix",   "weight": 0.6},
    {"name": "doc-style",  "weight": 0.4}
  ],
  "output_name": "code-fix-doc-style"
}
```

Sources must share `r`, `target_modules`, `base_model_name_or_path`, and per-tensor shapes. The merged adapter is written to disk in the same PEFT format (`adapter_config.json` + `adapter_model.safetensors`, f32) and can immediately be loaded via `POST /v1/adapters/load`. Merging happens off the async runtime via `spawn_blocking` and the helper lives at `crates/kiln-model/src/adapter_merge.rs::merge_linear`. TIES and concatenation merging are deferred to follow-up phases.

## Training Pipeline

Training runs in-process on a background thread, sharing the GPU with inference. The GPU is coordinated via an `RwLock<()>`:

- **Inference** acquires a read lock (multiple concurrent readers OK)
- **Training** acquires a write lock (exclusive — blocks inference during gradient computation)

### SFT (Supervised Fine-Tuning)

Submit training examples via `POST /v1/train/sft`:

```json
{
  "adapter_name": "my-adapter",
  "examples": [
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ],
  "num_epochs": 3,
  "learning_rate": 1e-4,
  "rank": 8
}
```

The training loop (`crates/kiln-train/src/trainer.rs`):

1. Initialize fresh LoRA parameters (Kaiming uniform for A, zero for B)
2. Tokenize examples with chat template, extract assistant-only label mask
3. For each epoch, for each example:
   - Forward pass through all 32 layers with LoRA applied
   - Compute cross-entropy loss **only on assistant tokens** (label masking)
   - Backward pass computes gradients for LoRA A and B matrices only
   - SGD step: `param -= learning_rate * gradient`
4. Save adapter in PEFT format
5. Optionally auto-load the trained adapter for immediate use

### GRPO (Group Relative Policy Optimization)

Submit scored completions via `POST /v1/train/grpo`:

```json
{
  "adapter_name": "my-adapter",
  "groups": [
    {
      "prompt": [{"role": "user", "content": "..."}],
      "completions": [
        {"text": "answer A", "reward": 1.0},
        {"text": "answer B", "reward": 0.2}
      ]
    }
  ]
}
```

The GRPO loop:

1. For each group:
   - Forward pass under **policy** (base + LoRA) → compute log-probabilities
   - Forward pass under **reference** (base only, no LoRA) → compute reference log-probabilities
   - Normalize rewards within group: `advantage = (reward - mean) / std`
   - Compute clipped importance-sampling loss with KL penalty:

```
importance_weight = exp(log_prob_policy - log_prob_reference)
loss = -E[clip(importance_weight, 1-ε, 1+ε) × advantage] + kl_coeff × KL(policy ‖ reference)
```

2. Backward + SGD step (same as SFT)

### Gradient Checkpointing

To fit training on 24 GB consumer GPUs (RTX 3090/4090), Kiln implements activation checkpointing. Instead of storing activations for all 32 layers during the forward pass, it divides layers into segments and recomputes activations during the backward pass:

```
Without checkpointing: Store all 32 layers of activations (~20+ GB)
With 8 segments:       Store 8 checkpoint boundaries, recompute within each segment
                       Peak VRAM: ~12-16 GB (model 8 GB + checkpointed activations + LoRA grads)
```

More segments = less VRAM but more computation. The number of segments is auto-tuned based on detected VRAM. Controlled by `KILN_GRAD_CHECKPOINT_SEGMENTS` or disabled with `KILN_NO_GRAD_CHECKPOINT=1`.

See `model_forward_segment()` in `crates/kiln-model/src/forward.rs`.

### Training Queue

Training jobs are queued FIFO and executed one at a time (`crates/kiln-server/src/training_queue.rs`). A background tokio task polls every 500ms, picks the next job, and runs it on the thread pool with an exclusive GPU write lock.

```
POST /v1/train/sft ──► Queue (FIFO) ──► Worker thread ──► GPU (write lock)
POST /v1/train/grpo ──►                                     │
                                                             ▼
GET /v1/train/status/{id} ◄── TrainingJobInfo (progress, loss, state)
```

Jobs can be cancelled while queued but not while running.

## Performance Optimizations

### CUDA Graphs for Decode

After a warmup step, the decode forward pass is captured into a CUDA graph and replayed on subsequent steps. This eliminates kernel launch overhead and provides 10-15% speedup for decode.

The position value (for RoPE) is updated via `cudaMemcpyHtoDAsync` to a pre-allocated GPU buffer before each graph replay. The graph reads from the same device pointer, seeing the updated position each step.

CUDA graphs are invalidated when LoRA adapters are swapped (different weight pointers). See `crates/kiln-model/src/cuda_graph.rs`.

### GPTQ INT4 Quantization

Kiln loads GPTQ-quantized models with packed INT4 weights. Each `u32` stores 8 4-bit weights with per-group scales and zero points:

```
Dequantization: (weight_int4 - zero_int4) × scale → BF16
```

Currently dequantized to BF16 on CPU during loading. Auto-detected via `quantize_config.json` in the model directory. See `crates/kiln-model/src/quantized.rs`.

### Speculative Decoding

Self-speculative decoding uses the first N layers (default 8) as a draft model — no separate model needed, no extra VRAM:

```
Phase 1: Draft — run first 8 layers K times, propose K candidate tokens
Phase 2: Verify — run full 32 layers on all K+1 positions in one forward pass
Phase 3: Accept/Reject — for each draft token:
           if random() < min(1, p_target / p_draft): accept
           else: resample from max(0, p_target - p_draft) and stop
Bonus:   If all K accepted, sample one extra token from position K
```

Expected speedup: 1.5-2.5x depending on acceptance rate. Configured via `kiln.toml`:

```toml
[speculative_decoding]
enabled = true
num_speculative_tokens = 4
draft_layers = 8
```

See `crates/kiln-model/src/speculative.rs`.

## Configuration

Kiln uses layered configuration with priority: environment variables > TOML file > defaults.

```toml
# kiln.toml

[server]
host = "0.0.0.0"
port = 8080
request_timeout_secs = 300
shutdown_timeout_secs = 30

[model]
path = "/models/Qwen3.5-4B"        # omit for mock mode
model_id = "qwen3.5-4b"
adapter_dir = "./adapters"

[memory]
# num_blocks = 64                   # auto-detected from VRAM if omitted
inference_memory_fraction = 0.7     # fraction of remaining VRAM for KV cache
kv_cache_fp8 = false                # halve KV cache memory with FP8
cuda_graphs = true                  # 10-15% decode speedup

[training]
# grad_checkpoint_segments = 8      # auto-tuned if omitted
checkpoint_interval = 100           # save checkpoint every N training steps

[logging]
level = "info"                      # trace, debug, info, warn, error
format = "json"                     # json or pretty

[prefix_cache]
enabled = true
max_blocks = 128                    # default: 25% of total blocks

[speculative_decoding]
enabled = false
num_speculative_tokens = 4
draft_layers = 8
```

### Key Environment Variables

| Variable | Description |
|----------|-------------|
| `KILN_CONFIG` | Path to config file (default: `kiln.toml`) |
| `KILN_GPU_MEMORY_GB` | Override GPU VRAM detection |
| `KILN_NUM_BLOCKS` | Override KV cache block count |
| `KILN_GRAD_CHECKPOINT_SEGMENTS` | Override gradient checkpoint segments |
| `KILN_NO_GRAD_CHECKPOINT` | Set to `1` to disable gradient checkpointing |
| `KILN_LOG_LEVEL` | Override log level |
| `KILN_LOG_FORMAT` | Override log format (`json` or `pretty`) |

See `crates/kiln-server/src/config.rs` for the full configuration schema and validation.

## Key Data Structures

### ModelConfig (`crates/kiln-core/src/config.rs`)

Hardcoded for Qwen3.5-4B. Key values:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 2560 | Model width |
| `num_layers` | 32 | 24 GDN + 8 GQA |
| `num_attention_heads` | 16 | Q heads (full attention) |
| `num_kv_heads` | 4 | KV heads (GQA group size = 4) |
| `head_dim` | 256 | Per-head dimension |
| `vocab_size` | 248320 | Token vocabulary |
| `max_position_embeddings` | 262144 | 256K max context |
| `rope_theta` | 10,000,000 | RoPE base frequency |
| `partial_rotary_factor` | 0.25 | 64 of 256 dims get RoPE |
| `linear_num_key_heads` | 16 | GDN Q/K heads |
| `linear_key_head_dim` | 128 | GDN Q/K per-head dim |
| `linear_num_value_heads` | 32 | GDN V heads |
| `linear_value_head_dim` | 128 | GDN V per-head dim |

KV cache per token: 32 KB (BF16) or 16 KB (FP8) — only across 8 full-attention layers. 128K context fits in ~4 GB (BF16) or ~2 GB (FP8).

### Request (`crates/kiln-core/src/request.rs`)

Tracks the lifecycle of a single inference request: prompt tokens, growing output tokens, allocated block IDs, sampling parameters, optional LoRA adapter ID, and current state (Waiting/Prefilling/Decoding/Complete/Cancelled).

### GpuWeights (`crates/kiln-model/src/forward.rs`)

GPU-resident model weights. Contains embedding tensor, 32 layer weight structs (each dispatched as `GpuAttentionWeights::Full` or `::Linear`), and final layer norm. Loaded from safetensors via `crates/kiln-model/src/loader.rs`.

### AppState (`crates/kiln-server/src/state.rs`)

Server-wide shared state. In real mode: `Arc<RwLock<ModelRunner>>`, `Arc<Mutex<BlockManager>>`, `Arc<Mutex<PagedKvCache>>`, GPU memory budget, training job tracker, metrics, and the GPU coordination lock.

## Monitoring

### Prometheus Metrics (`GET /metrics`)

Atomic counters and gauges with no external dependencies:

- `kiln_requests_total{status=ok|error|timeout}` — request counts
- `kiln_tokens_generated_total` — total tokens produced
- `kiln_request_duration_seconds` — request latency (count + sum)
- `kiln_training_*_total{status=completed|failed|cancelled}` — training job counts
- `kiln_scheduler_*` — waiting/running requests, block usage
- `kiln_gpu_memory_*_bytes` — VRAM breakdown (total, model, KV cache, training budget)

See `crates/kiln-server/src/metrics.rs`.

### Health Endpoint (`GET /v1/health`)

Returns uptime, model info, scheduler statistics, GPU memory breakdown, active adapter, and training queue state.

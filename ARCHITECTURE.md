# Architecture Deep-Dive

This document explains how Kiln works internally. It is aimed at contributors and power users who want to understand what happens between an HTTP request arriving and a token being generated — or a model being trained.

## Where to go next

- Use this architecture deep-dive when you want to understand Kiln's scheduler, model runner, LoRA hot-swap, training queue, and CUDA kernel layout.
- Start with [Quickstart](QUICKSTART.md) when you want to install Kiln, run the server, or try the common API flows before reading internals.
- Read [docs/GRPO_GUIDE.md](docs/GRPO_GUIDE.md) when you want the generate → score → train loop, reward-shaping examples, and GRPO request shapes.
- Skim [README.md](README.md) when you want the shorter overview, feature map, install command, and links to the rest of the docs.
- If setup or API behavior is confusing, use the website [Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html).

## System Overview

Kiln is a single Rust binary built as a Cargo workspace with twelve crates — seven portable crates plus five CUDA kernel crates that are only compiled when `--features cuda` is enabled:

```
kiln
├── kiln-core             Core types: block manager, prefix cache, KV cache config, request lifecycle
├── kiln-model            Model loading, forward pass, LoRA, sampling, KV cache, CUDA graphs
├── kiln-scheduler        Sarathi-style continuous batching scheduler with chunked prefill
├── kiln-server           Axum HTTP server, CLI, training queue, metrics, configuration
├── kiln-train            SFT and GRPO training loops with gradient checkpointing
├── kiln-nvtx             Thin NVTX range wrapper for nsys attribution (zero overhead when off)
├── kiln-flce-kernel      Fused Linear Cross-Entropy: chunked CE without materializing [T, V] logits
└── (CUDA-only, --features cuda)
    ├── kiln-flash-attn   Vendored Flash-Attention-2 CUDA kernels with C-ABI/Rust FFI
    ├── kiln-gdn-kernel   Vendored Gated DeltaNet chunk forward-substitution kernel (mamba-ssm port)
    ├── kiln-conv1d-kernel Vendored mamba-ssm causal_conv1d_update decode kernel
    ├── kiln-rmsnorm-kernel Fused RMSNorm CUDA kernel (Liger-style, ~11 launches → 1)
    └── kiln-marlin-gemm  Vendored IST-DASLab/marlin W4A16 GEMM CUDA kernel
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
                              │                    kiln-server                          │
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
                              │  ┌──────────────────────────────────────────────────┐   │
                              │  │  Acquire GPU read lock (concurrent inference OK) │   │
                              │  │                                                  │   │
                              │  │  ModelRunner::generate_paged()                   │   │
                              │  │       │                                          │   │
                              │  │       ▼                                          │   │
                              │  │  ┌────────────────────────┐                      │   │
                              │  │  │  PREFILL               │                      │   │
                              │  │  │  Embed prompt tokens   │                      │   │
                              │  │  │  Forward through 32    │                      │   │
                              │  │  │  layers (GDN + GQA)    │                      │   │
                              │  │  │  Write KV cache        │                      │   │
                              │  │  │  Sample first token    │                      │   │
                              │  │  └────────┬───────────────┘                      │   │
                              │  │           │                                      │   │
                              │  │           ▼                                      │   │
                              │  │  ┌────────────────────────┐                      │   │
                              │  │  │  DECODE (loop)         │◄──── CUDA Graph      │   │
                              │  │  │  Embed 1 token         │      Replay          │   │
                              │  │  │  Forward through 32    │      (after warmup)  │   │
                              │  │  │  layers, read KV cache │                      │   │
                              │  │  │  Sample next token     │                      │   │
                              │  │  │  Check stop conditions │                      │   │
                              │  │  └────────┬───────────────┘                      │   │
                              │  │           │ (EOS / max_tokens / stop sequence)   │   │
                              │  │           ▼                                      │   │
                              │  │  Return generated text + usage stats             │   │
                              │  └──────────────────────────────────────────────────┘   │
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
  └────────────────────────────────────────────────────────┘

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

When multiple requests share a common prompt prefix (e.g., a system prompt), the prefix cache avoids recomputing KV entries. Kiln currently has two implementations:

- **Radix prefix tree** (`crates/kiln-core/src/prefix_cache.rs`, PR #512). A SGLang-style trie over block-aligned token-hash edges. Each node is one cached block; siblings share their longest common prefix; lookups walk the tree from the root and return the longest matching block run. LRU eviction is leaf-only so internal shared prefixes survive until every descendant is evicted. This is the long-term structure used by the mock-backend scheduler today.
- **Flat `RealPrefixCache`** (`crates/kiln-server/src/state.rs`, PRs #515 / #520 / #521). A linear-scan cache over registered (token-prefix → physical-block-id) entries that backs the production `/v1/chat/completions` path. PR #520 added streaming-reuse so partial decodes can register their KVs incrementally, PR #521 made the cache CUDA-graph-compatible by keeping reused block pointers stable across the graph capture, and PR #518 added a runtime warning when a request configuration would silently bypass the cache (e.g. CUDA graphs replaying with a different block table).

Both caches use the same block-aligned hash scheme (each block hash mixes the parent block's hash with its own 16 tokens, so identical token runs at different positions still produce different hashes). Cached blocks are reference-counted and LRU-evicted when the budget is full. Future work consolidates the two paths so the radix tree backs production as well.

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

### GDN Kernel Implementation

The 24 Gated DeltaNet layers run on the vendored `kiln-gdn-kernel` crate (PR #80, ported from `mamba-ssm`'s `chunk_gla_fwd`). It exposes two CUDA entry points consumed by `CudaBackend`: a chunkwise prefill kernel that processes the full sequence with forward-substitution, and a single-token recurrent decode kernel that updates the per-layer state matrix `S` in place. Decode-side fusion has been pushed as far as the architecture allows:

- **PR #158 — fused gates** (merged). Decay-gate (`γ = -exp(A_log) * softplus(a + dt_bias)`) and write-gate (`β = sigmoid(b)`) are computed in one kernel instead of two candle ops.
- **PR #173 — fused L2-QK norm** (opt-in, null median). Available behind `KILN_ENABLE_FUSED_L2_QK_NORM=1`. Bench-neutral on A6000 under CUDA graphs (variance reduction only — graph replay already amortizes the launch cost the fusion saved).
- **PR #176 — big-fusion across recurrent + qk_norm + gated_norm** (closed null). Step 6 (gates) and Step 8 (gated RMSNorm) are separated by Step 7 (the in-place recurrence), so a single mega-kernel was architecturally infeasible.

Cross-stack audit (PR #525) compared `kiln-gdn-kernel` against vLLM's Triton `fused_recurrent_gated_delta_rule_packed_decode_kernel` on A6000. Under CUDA graphs the math ceiling for vendoring vLLM's tile shape is below the 1.05× floor — no portable wins were available, so kiln stays on the mamba-ssm port.

See `crates/kiln-gdn-kernel/` and the `gated_deltanet_forward()` dispatch in `crates/kiln-model/src/forward.rs`.

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
GET    /v1/adapters                       List active + available adapters
POST   /v1/adapters/load                  Load adapter from disk
POST   /v1/adapters/unload                Revert to base model
POST   /v1/adapters/merge                 Merge adapters (weighted_average | ties | concat)
POST   /v1/adapters/upload                Multipart tar.gz import (PR #577)
GET    /v1/adapters/{name}/download       Streaming tar.gz export (PR #575)
DELETE /v1/adapters/{name}                Delete adapter from disk
```

See `crates/kiln-server/src/api/adapters.rs`.

`download_adapter` builds a `tar.gz` of the adapter directory on a `spawn_blocking` thread and pushes chunks through a bounded mpsc channel so the response streams without buffering the whole archive in memory. `upload_adapter` accepts a multipart/form-data body up to 2 GiB, extracts into a `.upload-tmp-*` staging directory, enforces caps on total extracted bytes (4 GiB) and entry count (100 000), and atomically renames into place. Path traversal, symlinks, devices, and sockets are rejected at extract time. Together these endpoints make adapters portable: train somewhere, download, upload to another kiln instance, hot-swap.

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

Sources must share `r`, `target_modules`, `base_model_name_or_path`, and per-tensor shapes. The merged adapter is written to disk in the same PEFT format (`adapter_config.json` + `adapter_model.safetensors`, f32) and can immediately be loaded via `POST /v1/adapters/load`. Merging happens off the async runtime via `spawn_blocking` and the helper lives at `crates/kiln-model/src/adapter_merge.rs::merge_linear`.

Two additional merge modes shipped in Phase 8 and live in the same crate:

**`ties`** (Yadav et al. 2023, arXiv 2306.01708, PR #578) reduces destructive interference between adapters via a three-phase per-tensor pipeline: (1) **trim** — for each adapter, keep only the top `density` fraction of values by absolute magnitude and zero the rest; (2) **elect sign** — at each parameter position, take the sign of `Σⱼ wⱼ · trimmed_j[i]`; (3) **disjoint merge** — weight-average only the trimmed values whose sign agrees with the elected sign. The request accepts an optional `density` in `(0.0, 1.0]` (default 0.2, the TIES paper's recommendation). Shape requirements are identical to `weighted_average`. Helper: `merge_ties` in `crates/kiln-model/src/adapter_merge.rs`.

**`concat`** (PR #579) preserves each source's contribution by stacking ranks rather than averaging. `lora_A` is row-concatenated to shape `[Σᵢ rᵢ, in_features]`; `lora_B` is column-concatenated to shape `[out_features, Σᵢ rᵢ]` with each block scaled by its source weight. The product `B_concat @ A_concat` then equals `Σᵢ wᵢ · (Bᵢ @ Aᵢ)` — the same effective rank-update each source would have applied independently, materialized as one rank-`r_total` adapter. Unlike `weighted_average` and `ties`, source ranks are allowed to differ — that is the whole point. The merged `lora_alpha` is rescaled to `alpha_first × r_total / r_first` so the inference-time `alpha / r` factor is preserved. Tensor names must end in `lora_A.weight` or `lora_B.weight` (embedding LoRAs and DoRA magnitude vectors fall back to `weighted_average` or `ties`). Helper: `merge_concat` in `crates/kiln-model/src/adapter_merge.rs`.

### Per-Request Adapter Composition

Sometimes you want to stack multiple LoRAs at inference time without writing a new merged adapter to disk first. Phase 8 (PR #581) added a per-request composition spec to `/v1/chat/completions` and `/v1/completions/batch`: instead of `"adapter": "name"`, pass `"adapters": [{"name": "code-fix", "scale": 1.0}, {"name": "doc-style", "scale": 0.5}]`. The two fields are mutually exclusive — a request specifies either a single adapter or a composition list, not both.

Composition is implemented as a **cached `merge_concat`** on the request path. The server hashes the `(name, scale)` pairs, looks up an existing composed adapter under `adapter_dir/.composed/<hash>/`, and synthesizes a new one only if no cache entry exists. Synthesis is `merge_concat` with the per-source `scale` values used as weights, so the inference-time effect is exactly `Σᵢ scaleᵢ · (Bᵢ @ Aᵢ)`. The composed adapter is then loaded and hot-swapped through the existing iteration-boundary swap path. Subsequent requests with the same composition reuse the cached adapter without recomputation. See `synthesize_composed_adapter` and `composition_hash` in `crates/kiln-server/src/api/completions.rs`.

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

### Webhook Notifications

Phase 8 (PR #582) added an opt-in completion webhook so external schedulers, GRPO workers, and dashboards can react the moment a training job finishes — without polling `/v1/train/status/{id}`. The webhook URL is server-wide config (set via `KILN_TRAINING_WEBHOOK_URL` or the `[training] webhook_url` field in `kiln.toml`), not per-request. When set, every training job — both successful completions and failures — fires a fire-and-forget `POST` containing a `TrainingCompletionEvent` payload:

```json
{
  "job_id": "uuid",
  "job_type": "sft" | "grpo",
  "status": "completed" | "failed",
  "adapter_name": "my-adapter",
  "adapter_path": "/var/kiln/adapters/my-adapter",
  "error": null,
  "timestamp": "2026-04-26T01:23:45Z"
}
```

The HTTP `POST` runs on a tokio task with a 5-second client timeout and is best-effort: 4xx, 5xx, and transport errors are logged at WARN but never propagate, so a successful training job stays "completed" even if the notifier 5xxs. There is no built-in retry — clients that need at-least-once semantics should re-poll `/v1/train/status/{id}` on the receiving end. See `fire_completion_webhook` and `TrainingCompletionEvent` in `crates/kiln-server/src/training_queue.rs`.

## Batch Generation API

Phase 8 (PR #583) added `POST /v1/completions/batch`, a multi-prompt completion endpoint designed for the GRPO loop. GRPO normalizes advantages within a group of `n` completions per prompt, and issuing N separate HTTP requests per group adds non-trivial overhead per iteration. The batch endpoint takes the whole group in one round-trip and lets the iteration-level scheduler interleave the underlying prefill/decode steps:

```json
POST /v1/completions/batch
{
  "prompts": [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the capital of France?"}]
  ],
  "n": 4,
  "temperature": 0.8,
  "seed": 42,
  "adapter": "my-adapter"
}
```

The response carries `prompts.len() × n` items, each tagged with `prompt_index` and `completion_index`:

```json
{
  "id": "...",
  "completions": [
    {"prompt_index": 0, "completion_index": 0, "text": "...", "finish_reason": "stop", "usage": {...}},
    {"prompt_index": 0, "completion_index": 1, "text": "...", "finish_reason": "stop", "usage": {...}},
    ...
  ],
  "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
}
```

When `seed` is set, each completion's effective seed is `seed + (prompt_index * n + completion_index)` so completions are deterministic across runs but distinct within a group — without that, identical prompts plus a fixed seed would produce identical outputs even at temperature > 0. Total output count is capped at 64 per request (`BATCH_MAX_TOTAL_OUTPUTS`); over the cap, the request is rejected with `batch_too_large` (HTTP 400) so a runaway client cannot pin the engine for an unbounded number of iterations. `stream: true` is not supported on this endpoint — only the aggregated final result is returned. The entire batch shares a single adapter (or composition, or none); per-prompt adapter override is a future extension. See `BatchCompletionRequest`, `BatchCompletionResponse`, and `batch_completions` in `crates/kiln-server/src/api/completions.rs`.

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

### Marlin W4A16 GEMM

The `kiln-marlin-gemm` crate (PR #146, vendored from the IST-DASLab Marlin kernel) provides a hand-tuned W4A16 GEMM that runs the GPTQ-packed weights directly on tensor cores without dequantizing to BF16. It is opt-in via `KILN_W4A16=1` and, when enabled, dispatches the four highest-volume projections through Marlin: `q_proj` plus the MLP `gate_proj`, `up_proj`, and `down_proj`. `k_proj`, `v_proj`, and `o_proj` stay on the BF16 matmul path.

Two follow-on cleanups landed alongside the kernel:

- **PR #210 — Marlin pack determinism + speed**. The 96 MLP projections used to pack serially in ~42.8 s at model load; PR #210 made the pack deterministic and parallelized it down to ~16.9 s. (See [`docs/archive/benchmarks/MARLIN_MLP_BENCH.md`](docs/archive/benchmarks/MARLIN_MLP_BENCH.md) for the per-projection numbers.)
- **PR #206 — BF16 weight VRAM cleanup**. Previously the BF16 MLP weights stayed resident alongside the packed Marlin weights even when `KILN_W4A16=1` (~4.4 GB unused). PR #206 drops the BF16 tensors after packing.

See `crates/kiln-marlin-gemm/` for the kernel and `crates/kiln-model/src/marlin_proj.rs` for the BF16-Linear-compatible wrapper used by the forward path.

### Speculative Decoding

Kiln has two self-speculative-decode paths, both off by default. Both use the same generic verify loop in `crates/kiln-model/src/speculative.rs`; the dispatch is selected at server startup via `KILN_SPEC_METHOD={off|skip_layer|mtp}` (see `crates/kiln-server/src/config.rs`).

- **`skip_layer` (legacy)** — uses the first N layers of the main model as a draft. No extra VRAM, no separate checkpoint. Acceptance rate is workload-dependent and the default is `off`.
- **`mtp` (native MTP, attempted, null on A6000)** — Qwen3.5-4B ships with a single pretrained MTP (Multi-Token Prediction) head (`mtp.*` tensors in the checkpoint). PRs #535 / #536 vendored this head, ran the existing draft-then-verify loop with the MTP head as the drafter, and benchmarked end-to-end self-spec decode on A6000 bs=1.

Result for native MTP: measured acceptance α = **0.69**, below the 0.72 break-even ceiling implied by the kiln-native verify cost (see `PROFILING-MTP-C40*.md`). PR #536 merged the implementation behind `KILN_ENABLE_MTP=0` (default off) so the code path stays exercised but the production decode path is unaffected. The cross-stack audits in PRs #532 (SGLang) and #533 (vLLM), plus the HF-transformers α microbench in PR #534, all corroborated kiln's native α and confirmed there was no missed implementation win — the 0.72 ceiling is a property of the Qwen3.5-4B MTP head, not a kiln bug.

This supersedes the older skip-layer self-spec design described in the agent note `kiln-speculative-decoding-design`. Per-token configuration still flows through `[speculative_decoding]` in `kiln.toml`:

```toml
[speculative_decoding]
enabled = false                # KILN_SPEC_ENABLED
method = "off"                 # KILN_SPEC_METHOD: off | skip_layer | mtp
num_speculative_tokens = 4     # KILN_SPEC_NUM_TOKENS
draft_layers = 8               # KILN_SPEC_DRAFT_LAYERS (skip_layer only)
```

See `crates/kiln-model/src/speculative.rs` for the verify loop and `crates/kiln-model/src/mtp_debug.rs` for the per-step instrumentation used during the α investigation.

## Configuration

Kiln uses layered configuration with priority: environment variables > TOML file > defaults.

```toml
# kiln.toml

[server]
host = "0.0.0.0"
port = 8420
request_timeout_secs = 300
shutdown_timeout_secs = 30

[model]
path = "/models/Qwen3.5-4B"        # omit for mock mode
model_id = "Qwen/Qwen3.5-4B"       # HuggingFace model ID; served as "qwen3.5-4b-kiln" by default
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
format = "auto"                     # auto (pretty on TTY, JSON otherwise), json, pretty, text, human

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

## Phase status (2026-04-25)

Phase 6 (performance optimization) is closed. The post-#534 perf shortlist concluded with PRs #525 / #526 (SGLang RadixAttention), #210 / #206 (Marlin pack determinism + BF16 cleanup), #222 (FP8 KV opt-in), and #536 (native MTP self-spec, null at α=0.69). Active work is now Phase 7 (developer experience). For current decode numbers see `BENCHMARKS.md`; for the live profiling hotspot table see `PROFILING.md`.

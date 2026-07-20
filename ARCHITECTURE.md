# Architecture Deep-Dive

This document explains how Kiln works internally. It is aimed at contributors and power users who want to understand what happens between an HTTP request arriving and a token being generated — or a model being trained.

## Where to go next

- Use this architecture deep-dive when you want to understand Kiln's scheduler, model runner, LoRA hot-swap, training queue, and CUDA kernel layout.
- Start with [Quickstart](QUICKSTART.md) when you want to install Kiln, run the server, or try the common API flows before reading internals.
- Read [docs/GRPO_GUIDE.md](docs/GRPO_GUIDE.md) when you want the generate → score → train loop, reward-shaping examples, and GRPO request shapes.
- Read [docs/EVAL_GUIDE.md](docs/EVAL_GUIDE.md) when you want the scorer reference, the `POST /v1/eval/*` API, the `kiln-eval` CLI, and post-training auto-eval hooks.
- Skim [README.md](README.md) when you want the shorter overview, feature map, install command, and links to the rest of the docs.
- If setup or API behavior is confusing, use the website [Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html).

## System Overview

> **Migration note (Phase 1+ of #1082).** The substrate Kiln is being
> migrated to lives in six new crates that ship in parallel with the
> existing ones: `kiln-tensor` (Tensor + Storage), `kiln-blas` (CUDA
> BLAS), `kiln-param` (unified Parameter handle), `kiln-autograd`
> (tape-based reverse-mode), `kiln-optim` (fused optimizer step), and
> `kiln-graph` (command-list capture). See the [#1082 migration
> substrate](#1082-migration-substrate-phase-1-onward) section below
> for the dispatch flow + per-crate breakdown.

Kiln is a single Rust binary built as a Cargo workspace with thirteen crates — eight portable crates plus five CUDA kernel crates that are only compiled when `--features cuda` is enabled:

```
kiln
├── kiln-core             Core types: block manager, prefix cache, KV cache config, request lifecycle
├── kiln-model            Model loading, forward pass, LoRA, sampling, KV cache, CUDA graphs
├── kiln-scheduler        Sarathi-style continuous batching scheduler with chunked prefill
├── kiln-server           Axum HTTP server, CLI, training queue, eval queue, metrics, configuration
├── kiln-train            SFT and GRPO training loops with gradient checkpointing
├── kiln-eval             Suite + scorer + result types for LoRA evaluation (see docs/EVAL_GUIDE.md)
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
├── kiln-train
│   ├── kiln-core
│   ├── kiln-model
│   └── kiln-eval         (post_eval hook on SftRequest / GrpoRequest)
└── kiln-eval             (suite, scorers, results — pure CPU)
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

### Scheduler: Token-Budgeted Chunked Prefill

Real-model serving is scheduled by the actor in
`crates/kiln-server/src/batching_engine.rs`; resumable paged-prefill ownership
lives in `crates/kiln-model/src/generate.rs`. The `kiln-scheduler` crate drives
mock-mode execution and retains the same scheduling policy. Each cycle has a
strict priority order:

1. **Decode requests first** — each active decode request gets exactly 1 token. These have absolute priority because stalling a decode request means latency for a waiting user.
2. **Continue partial prefills** — if a prefill was chunked (prompt too large for one iteration's token budget), continue it with remaining budget.
3. **Start new prefills** — promote waiting requests and begin their prefill, chunking to fit the remaining budget.

The resolved `server.max_batch_tokens` budget (default 512) caps combined decode
and prefill work per actor cycle. A 50-token prompt with a budget of 30 gets
split into bounded chunks; if two decode rows are ready, they consume two tokens
and that cycle's prefill quantum is at most 28. Partial prefills are selected
round-robin, so a 16K prompt cannot repeatedly hide a shorter prefill.

```
Actor-cycle token budget: 512
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

The batching actor keeps cache admission separate from numerical prefill
geometry. Backends that support block-aligned prefix snapshots always preserve
the same final prompt split on an enabled miss and on a disabled-cache fresh
prefill. Disabling the cache suppresses lookup, snapshot, registration,
retention, and rolling-snapshot work, but it cannot silently change token or
layer quanta. This is required for hybrid GDN models: a resumable chunk boundary
is also a recurrent-state precision boundary, so coupling it to a storage
toggle can change greedy output without any cache hit.

### Conservative Terminal Drain

The batching engine's public `active_decode` and related drain gauges cover
terminal model cleanup, not only membership in the actor's scheduling vector.
A row remains visible in the last published snapshot while `finish_request` or
`discard_request` releases its graph owner, recurrent-state lease, prefix-cache
ownership, and private KV blocks. Only after that cleanup returns does the actor
publish the row's removal and queue terminal delivery. This makes a zero-active
health snapshot a resource-ownership boundary: graph, cache, and block-pool
drain assertions cannot race a row that is no longer schedulable but is still
being finalized.

### VRAM Budget

At startup, Kiln detects GPU memory via the active backend (overridable with
`KILN_MEMORY_GPU_MEMORY_GB`) and auto-configures:

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

Most of `kiln-model`'s forward pass is expressed as portable `candle_core::Tensor` ops that run on any candle device. A small set of ops with no candle equivalent sits behind the `BackendRuntime` trait in `kiln-model::backend`. The trait abstracts platform-specific attention, paged-KV, Gated DeltaNet, conv1d, and norm kernels. Each method returns `Result<Option<Tensor>>`: returning `Ok(None)` means the backend declines the call, and the caller falls back to a portable candle-op path. This keeps `#[cfg(feature = "...")]` gates out of every call site in `forward.rs`, `generate.rs`, `paged_kv_cache.rs`, and the training loop.

Four backends implement the trait. `CudaBackend` dispatches to the vendored `kiln-flash-attn` kernels plus the `kiln-gdn-kernel` fused recurrent/chunk kernels. `MetalBackend` uses candle-metal plus Kiln's Metal shader family on Apple Silicon. `VulkanBackend` owns an `ash` Vulkan device for AMD/Intel GPUs, embeds SPIR-V compute shaders at build time, caches compute pipelines on the selected Vulkan device, and currently accelerates validated GDN gates, gated RMSNorm, recurrent decode, chunk prep, and chunk scan paths while declining unvalidated shaders. `CpuBackend` declines every op and routes all work through the portable fallback — used in mock mode and on platforms without a GPU feature enabled.

Backend selection is build-time via Cargo features: `--features cuda` pulls in `CudaBackend`, `--features vulkan` pulls in `VulkanBackend`, `--features metal` pulls in `MetalBackend`, and omitting all three yields the CPU-only fallback. At runtime, `backend::for_device()` picks the concrete backend for the active candle `Device`; Vulkan is detected separately because candle-core 0.10 has no native Vulkan tensor device.

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
  "examples": [
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ],
  "config": {
    "training_profile": "native_online_lora_v1",
    "output_name": "my-adapter",
    "epochs": 3,
    "lora_rank": 8,
    "lora_alpha": 16.0
  }
}
```

Native SFT is the bounded
[`native_online_lora_v1` profile](docs/NATIVE_SFT_PROFILE.md), not a general
trainer. Its main loop (`crates/kiln-train/src/trainer.rs`):

1. Initialize fresh LoRA parameters (Kaiming uniform for A, zero for B)
2. Tokenize examples with chat template, extract assistant-only label mask
3. For each epoch, for each example:
   - Forward pass through all 32 layers with LoRA applied
   - Compute cross-entropy loss **only on assistant tokens** (label masking)
   - Backward pass computes gradients for LoRA A and B matrices only
   - One configured optimizer update at a constant learning rate
4. Save adapter in PEFT format
5. Optionally auto-load the trained adapter for immediate use

One conversation is one microbatch and one update. There is no gradient
accumulation, warmup, decay, or gradient clipping. Unknown general-trainer
fields are rejected rather than ignored.

#### Backend-owned SFT loss routing

SFT does not select its cross-entropy implementation from request JSON, TOML,
CLI input, or process environment. `TrainingLossBackend` reports a typed
capability from the resident runner: CUDA and ROCm use `kt_tape_flce`, Vulkan
uses `vulkan_active_rows`, and Metal uses `full_logits`. The generated
[backend capability report](docs/backend-capability-report.md) is the
inspectable source-level support matrix. It is not a substitute for a passing
device qualification receipt.

Submission and execution bind that capability through the complete lifecycle:

```text
resident ModelRunner capability
  -> route-specific saturating working-set estimate
  -> checkpoint-route compatibility check
  -> queue entry with PreparedSftAdmission.loss_route
  -> resident-runner recheck before governor reservation/reclamation
  -> job-local TrainingRuntimeContext with the admitted route
  -> fresh execution-backend recheck before resident/trainable allocation
  -> every standard or checkpointed SFT loss step receives the pinned enum
  -> train receipt + exact-checkpoint planning identity
```

Admission accounts for the actual algorithm rather than a generic loss
constant. `kt_tape_flce` includes active-row and bounded chunk buffers, the
full hidden gradient, and CUDA/ROCm F32 head promotion when required.
`vulkan_active_rows` includes the largest legal F32 vocabulary chunk, its
weight transpose, active-row buffers, and full hidden gradient. `full_logits`
includes dense `[T, V]` logits and the portable cross-entropy forward/backward
residency. Every multiplication and sum saturates toward rejection. Automatic
checkpoint fitting may search multiple legal segment counts only for a
checkpoint-compatible route; a `full_logits` plan remains one segment.

An estimate above the current training budget returns HTTP 413 before queue
publication. The error names the route in its `loss workspace` line alongside
estimated/available capacity, activations and boundaries, LoRA gradients,
optimizer state, and residency scratch. A multi-segment `full_logits` plan
instead returns `training_invalid_request`, because checkpoint tails execute
outside an active kt tape. The trainer repeats this invariant before a forward.

The old `KILN_USE_FLCE` override is removed with no compatibility alias or
replacement. There is no loss-route field under `[training]`, so there is no
mechanically derived environment name. The selected route is observable in a
new SFT `train_receipt.json` at `runtime.sft_loss_route` and in SFT exact
checkpoint planning identity v4. It is intentionally not exposed as mutable
intent by `/v1/config` or health. GRPO and OPD retain planning identity v3;
their independent loss routes are not controlled by the SFT capability.

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

More segments = less VRAM but more computation. The number of segments is
auto-tuned based on detected VRAM. Override it with
`KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS`, or disable checkpointing with
`KILN_TRAINING_NO_GRAD_CHECKPOINT=1`.

SFT auto-tuning is also constrained by the backend-owned loss route.
`kt_tape_flce` and `vulkan_active_rows` support multi-segment execution;
`full_logits` does not. Kiln rejects a checkpointed `full_logits` plan instead
of estimating a fused route and executing the dense portable route.

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

## Evaluation Pipeline

The eval pipeline lives in two places:

- **`kiln-eval`** (pure CPU crate): suite/scorer/result types, the synthesis engine that turns SFT/GRPO datasets into eval suites, and the `PostEvalConfig` knob embedded in `SftRequest`/`GrpoRequest`.
- **`kiln-server::eval`** module: the queue + worker + registry plumbing, the dataset and judgment stores, and the HTTP surface at `/v1/eval/*` and `/v1/judgments/*`.

Full reference (request shapes, scorer reference, CLI usage): [docs/EVAL_GUIDE.md](docs/EVAL_GUIDE.md).

### Stores on disk

Three on-disk registries persist the eval system, all rooted under `[eval] root` (defaults to `<state_dir>/eval`):

| Path | Type | Purpose |
|------|------|---------|
| `<root>/suites/<name>.json` | `SuiteRegistry` | Versioned eval suites (examples + scorers + generation defaults) |
| `<root>/datasets/<name>/` | `DatasetRegistry` | Uploaded SFT/GRPO JSONL with `manifest.json` + sampled stats |
| `<root>/judgments/<name>/` | `JudgmentStore` | A/B/Tie/Skip rows + manifest for the human-judge flywheel |

`SuiteRegistry::load` returns a deserialized `EvalSuite` from `kiln-eval`; `DatasetRegistry` keeps file size and row counts cheap by sampling the first N rows on upload, so the UI can render a preview without re-walking 100k-line files.

### Scorers

Scorers are variants of the serde-tagged `kiln_eval::scorers::Scorer` enum that compute a 0/1 outcome from `(generated_text, target, metadata)`. The on-disk `kind` field selects the variant; the kind labels (snake-cased) and use cases:

| `kind` | Variant | Use case |
|--------|---------|----------|
| `exact_match` | `Scorer::ExactMatch` | Single-source-of-truth answers (math, classification labels) |
| `contains` / `regex` | `Scorer::Contains` / `Scorer::Regex` | Free-text key-phrase or pattern matching |
| `json_validity` / `multiple_choice` / `numeric_tolerance` | `Scorer::JsonValidity` / `MultipleChoice` / `NumericTolerance` | Structured outputs and graded numeric responses |
| `tool_call` | `Scorer::ToolCall` | Trajectory/tool-arg matching (selected tool + arg shape + inner-content quality on string args) |
| `code` | `Scorer::Code` | Code-output evals via `CodeStyle` (Jaccard, AST, output-match), with `bash` introspection for `python -c`/`node -e`/`uv run` inlined inside `bash` |
| `llm_judge` | `Scorer::LlmJudge` | Local judge LoRA scoring with `judge_adapter`/`template`/`score_regex` |
| `all` / `any` | `Scorer::All` / `Scorer::Any` | Composite — pass-all or pass-any over a list of sub-scorers |

Adding a scorer means: append a `Scorer` variant, add the `kind_label()` arm, implement the scoring logic in `crates/kiln-eval/src/scorers/<name>.rs`, and dispatch it in `score_completion`. The wire format is then derived for free; synthesis and the UI pick it up via `kind_label`.

### Synthesis: dataset → suite

`POST /v1/eval/datasets/<name>/synthesize` (driven by `eval/synthesis_driver.rs`) decomposes an SFT or GRPO file into a graded suite. Strategies live in `kiln_eval::synthesis`:

- **`final_assistant`** — last assistant turn becomes the target; everything before becomes the prompt.
- **`first_assistant_turn`** — grade only the model's opening reply.
- **`every_assistant_turn`** — emit one example per assistant turn (multi-turn rollouts).
- **`tool_call_predict`** — for trajectories ending in a tool call: predict `{tool_name, args}` and score with `ToolCall` (with optional inner-content quality scorers on string args).

Auto-detect chooses the scorer from the target shape (JSON → `JsonValidity` + structural checks, short literal → `ExactMatch`, code block → `Code`, free text → `Contains` / `LlmJudge`). A reservoir-sampled `max_examples` cap keeps suites runnable in seconds even when the source dataset has hundreds of thousands of rows.

### Eval queue + worker

The eval queue mirrors the training queue's design — a single FIFO worker, terminal-job GC, structured progress callbacks — but runs concurrently with training and inference because it consumes only the inference path (read lock on the GPU):

```
POST /v1/eval/run ──┐
POST /v1/eval/compare ──► EvalQueue (FIFO) ──► spawn_eval_worker
POST /v1/train/{sft,grpo}                            │   (read-locks GPU
  with post_eval: {…} ──► enqueue_post_training_eval │    while generating)
                                                     ▼
                                  EvalJobInfo (Queued → Running → Completed/Failed)
                                  ├── progress: EvalProgress { running_accuracy, … }
                                  ├── finished_runs: Vec<SuiteResult>
                                  └── headline_accuracy
```

`QueuedEvalJob` has three variants — `Registered { suite_name, adapter, … }`, `Inline { suite, adapter, … }`, and `Compare(spec)` (one suite × N adapters). The worker reads the next entry, resolves the suite from the registry (or uses the inline copy), instantiates a generator that drives the in-process inference path through `crate::eval::generator::generator_from_state`, and runs `run_suite_against_adapter` per adapter. Progress callbacks update `EvalJobInfo.progress` after every example so the UI's drill-in modal can stream a running accuracy.

Cancellation flows through an `Arc<AtomicBool>` checked between examples; the API's cancel endpoint marks the job `Cancelled` and the worker either skips it (if still queued) or stops at the next example boundary (if running). Terminal jobs are evicted from `eval_jobs` once `tracked_job_ttl` elapses (`gc_eval_jobs`) so long-running servers don't grow the map unbounded.

### Post-training auto-eval hook

Every `SftRequest` and `GrpoRequest` accepts an optional `post_eval: { suite, include_baseline, generation }`. When the training job completes successfully, `enqueue_post_training_eval` (in `crates/kiln-server/src/training_queue.rs`) immediately pushes one or two eval jobs onto the eval queue:

- The new adapter, against the named suite (always).
- A baseline run with no adapter (when `include_baseline: true`), so the UI can render the delta side-by-side without a second request.

The eval job IDs are recorded on `TrainingJobInfo.linked_eval_job_ids` at queue time — not at eval completion — so the training-detail panel can link to the in-flight eval the moment it appears, not after it finishes. The eval worker deliberately does not push back into the training job's link list, since the assignment already happened upstream and a second push would duplicate every ID.

### Drill-in modal flow

The UI uses three peer endpoints with a consistent shape so the drill-in modals can share a renderer:

| Endpoint | Returns | Modal |
|----------|---------|-------|
| `GET /v1/eval/jobs/{id}` | `EvalJobInfo` (state + progress + per-example outcomes + headline) | Eval drill |
| `GET /v1/train/jobs/{job_id}` | `TrainingJobDetail` (`serde(flatten)`-composed with linked eval IDs and downsampled loss history) | Training drill |
| `GET /v1/adapters/{name}/detail` | `AdapterDetail` (file list + training history + eval history) | Adapter drill |

Each modal polls only while the underlying job is non-terminal and uses a content-key change-detection guard (e.g. `evalDrillLastKey`) so the SVG charts and example tables only re-render when something actually changed. The training-job loss history is downsampled when it crosses 2× the cap, giving amortized O(1) `push_loss_sample` while still showing a smooth curve over thousands of steps.

### Judgment flywheel

The flywheel turns user A/B preferences into a local judge LoRA — no frontier LLM ever called:

```
A/B compare playground  ──► POST /v1/judgments/{name}/rows         (one row per pick)
                                       │
                                       ▼
                            JudgmentStore on disk (manifest + JSONL)
                                       │
              POST /v1/judgments/{name}/compile
                                       │
                                       ▼
                       SFT examples in chat-template form
                                       │
                                       ▼
                  POST /v1/train/sft (judge-lora-name)
                                       │
                                       ▼
                  Scorer::LlmJudge { judge_adapter: "judge-lora-name", template, score_regex }  ──► used by any future suite
```

`compile_judgments_to_sft` formats each `(prompt, response_a, response_b, winner)` row into a `format_judge_prompt` chat with a single-token verdict label. `build_validation_suite` produces a held-out slice so `POST /v1/judgments/{name}/validate` can score the freshly trained judge against the user's own preferences. Once the judge LoRA is trained, any suite can name it via `Scorer::LlmJudge { judge_adapter: … }` and the eval worker resolves it the same way the inference path resolves any other adapter — same hot-swap, same scheduler, same machine.

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

## Thinking Budgets

`ThinkingBudget` in `kiln-core::sampling` is a request-local decode state
machine shared by the sampling clones used by the engine and response layer.
It activates only when the rendered prompt ends inside `<think>`. At each token
boundary it observes the reasoning-token count and elapsed decode time, lets a
natural `</think>` win, or replaces the sampled candidate with the next token
of the tokenizer-validated close sequence. It also reserves enough of the
completion-wide `max_tokens` allowance to emit that sequence atomically.

The replacement happens before EOS and stop checks. Each forced token is then
accepted through the normal decode path, enters KV and generated-token history,
and is visible to the next model step. Once the sequence is complete, the state
machine becomes inert and ordinary answer generation resumes. The batching
actor applies the same ordering to its first-token and decode-token paths. The
ordinary flat, paged, streaming, and batched loops enforce the controller.
Speculative serving is unavailable, and the public high-level speculative and
MTP model entry points fail closed before work, so multi-token acceptance cannot
skip a thinking-budget boundary.

Request limits use three states per dimension: omitted inherits the server
default, explicit JSON `null` is unlimited, and a nonnegative number is a
request limit (`0` closes immediately). Time starts when the first decode
candidate is available, excluding queue and prefill. Time-budgeted requests do
not use deterministic response caches; token budgets are part of cache identity
and cached values retain their original closure outcome. Chat choices, batch
items, final SSE chunks, eval results, health/config APIs, the Playground, the
desktop settings UI, and rollout/eval CLIs all expose the same semantics.

## Performance Optimizations

### CUDA Graphs for Decode

`memory.cuda_graphs` requests single-row CUDA decode capture, while
`memory.cuda_graph_cache_entries` bounds retained graph geometries and their
stable device buffers to `1..=64` (default `8`). The owning server resolves
both fields before device selection and injects one immutable
`CudaGraphExecutionPolicy`; decode never reads process environment. The
default `stable` serving profile still resolves execution to eager because it
does not permit live graph capture.

After a warmup step, an eligible decode forward pass is captured into a CUDA
graph and replayed on subsequent steps. Historical NVIDIA measurements found
lower launch overhead, but current performance remains a machine-local NVIDIA
qualification claim rather than a portable percentage guarantee.

The position value (for RoPE) is updated via `cudaMemcpyHtoDAsync` to a pre-allocated GPU buffer before each graph replay. The graph reads from the same device pointer, seeing the updated position each step.

Paged block-table, sequence-length, KV-slot, rotary, and attention-output
buffers are always graph-stable and refreshed in place. There is no opt-out:
transient metadata previously caused stale-pointer faults under concurrent
block reuse. The in-tree batched capture implementation remains unavailable
after poisoning the CUDA context in real concurrent serving; no environment
switch can activate it. Re-enabling it requires a source change and NVIDIA
correctness, resilience, and throughput evidence.

CUDA graphs are invalidated when LoRA adapters are swapped (different weight pointers). See `crates/kiln-model/src/cuda_graph.rs`.

### ROCm Graphs for Decode

`accelerator.rocm_graph_mode = "lazy_capture_replay"` requests HIP graph
capture for eligible single-row and contiguous BF16 multi-row decode. A batch
graph is keyed by batch width and the bucketed FlashAttention paged-decode
geometry, so changing cohorts can reuse one graph without changing any captured
device pointer. The owner slot refreshes recurrent and convolution state in
place for GDN layers; full-attention layers use stable RoPE, block-table,
sequence-length, KV-slot, attention-output, and softmax scratch tensors.

The captured region ends at the pre-final-norm hidden state. Final norm, LM
head, penalties, and sampling execute eagerly so one graph can serve greedy and
sampled rows without capturing host-visible token selection. Each replay orders
default-stream input refresh into the private graph stream with an event, then
orders graph completion back to eager work without a host wait. The paged-KV
allocation identity and generation must still match before launch.

Every new multi-row graph must also pass an automatic first-launch oracle before
cache admission. The runner retains the eager warm pass's hidden output, all GDN
recurrent and convolution state, and only the current rows from every K/V pool;
it restores the pre-warm GDN state, launches the new graph once, and compares all
of those values exactly on device. The snapshots, current-row gathers, and
single U8 equality mask are reserved from the matching device's memory governor
before the first snapshot allocation. A mismatch or comparison error settles
device work, restores the pre-capture GDN state, discards the graph, disables
graphs for that runner, and permits only the existing contained eager retry.
`event="rocm_graph_capture_parity_check"` reports a `passed`, `failed`, or
`error` outcome, whether comparison completed, compared bytes, duration, and
the first mismatching state/K/V layer or comparison error. This guard is
unconditional; there is no tuning or environment switch that can bypass it.

Graph-slot liveness is row ownership, not cache residency. Health counts only a
slot with `assigned_row=Some(id)` as active. A retained single-row slot or a
slot reserved for a batched width is idle between cohorts even though its
graphs, recurrent state, and width mapping remain resident and byte-accounted.
At a drained boundary both the active-slot and tracked-owner counts must be
zero; positive idle slots are bounded reusable capacity. The
`multi_row_batch_unsupported` fallback remains a closed legacy reason so old
receipts remain valid; supported current batch graphs must leave it at zero and
show successful capture/replay. See `crates/kiln-model/src/rocm_graph.rs`.

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

Kiln retains two speculative implementations as research and qualification
substrate: `skip_layer` uses the first `draft_layers` model layers as a draft,
and `mtp` uses checkpoint MTP heads. Both share the verifier in
`crates/kiln-model/src/speculative.rs`. Neither implementation has a serving
route. Streaming, non-streaming, and batched requests all remain on the
ordinary decode path.

The only serving policy currently accepted is effective `off`. `kiln config`
and server startup reject every effective non-off policy before model weights
are loaded. This includes `enabled = true` with `method = "off"`, which resolves
to the legacy `skip_layer` fallback. A dormant method value is retained when
`enabled = false`, but its effective method remains `off`. The production
loader is called with `load_mtp = false`, which retains a deferred checkpoint
source but does not upload MTP weights or prewarm either implementation.
Public high-level model-library speculative entry points also reject before
work. Server SFT normalizes omitted `train_mtp` to false and rejects explicit
true, so training cannot materialize that deferred slot inside a live server.

The draft window defaults to K=4 and is hard-capped at K=4 in both server
configuration and the low-level model API. Promotion requires local
accelerator qualification at K=1, K=2, and K=4, including cancellation,
device-owner settlement, EOS, rejection and full-acceptance behavior,
near-context capacity, timeout/panic quarantine, burst admission, and
throughput evidence. Any speculative benchmark must run in an isolated
qualification harness; the benchmark-only path is not a serving bypass.

The historical A6000 native-MTP experiment in PRs #535 / #536 measured
acceptance α=0.69, below the approximately 0.72 break-even point for that
implementation. Those results remain useful benchmark evidence, but do not
establish present serving eligibility. The SGLang, vLLM, and Hugging Face
cross-stack investigations are summarized in `BENCHMARKS.md`.

The typed startup configuration remains visible so intent can be validated and
reported consistently:

```toml
[speculative]
enabled = false
method = "off"
num_speculative_tokens = 4      # default and hard ceiling
draft_layers = 8                # qualification geometry only
```

See `crates/kiln-model/src/speculative.rs` for the verify loop. The temporary
per-step Phase B/C instrumentation used during the acceptance investigation was
retired after its conclusions were archived in `PROFILING.md`; the remaining
MTP-only single-token attention invariant is a private scoped runtime guard.

## Configuration

Kiln uses layered configuration with priority: environment variables > TOML file > defaults.

```toml
# kiln.toml

[server]
host = "0.0.0.0"
port = 8420
request_timeout_secs = 300
shutdown_timeout_secs = 30

[batching]
mode = "auto"                         # backend-owned actor selection
rowwise_decode = false                # one true batched forward per cohort
prefix_aware_admission = true         # retain strict-prefix reuse opportunity
prefill_admission_quantum = "auto"    # backend-owned prompt-admission cadence

[accelerator]
rocm_kernel_profile = "qualified"     # complete immutable ROCm model-kernel route set

[model]
path = "/models/Qwen3.5-4B"        # omit for mock mode
model_id = "Qwen/Qwen3.5-4B"       # HuggingFace model ID; served as "Qwen3.5-4B" by default
adapter_dir = "./adapters"

[memory]
# num_blocks = 64                   # auto-detected from VRAM if omitted
inference_memory_fraction = 0.7     # fraction of remaining VRAM for KV cache
kv_cache_fp8 = false                # halve KV cache memory with FP8
cuda_graphs = true                  # request qualified single-row capture
cuda_graph_cache_entries = 8        # retain at most this many captured shapes

[training]
# grad_checkpoint_segments = 8      # auto-tuned if omitted
recompute_checkpoint_boundaries = "auto"  # auto at threshold; enabled; disabled
recompute_boundary_threshold_tokens = 8192
checkpoint_boundary_anchor_stride = "auto" # or a positive segment stride
checkpoint_boundary_cache_gb = 6.0          # auto-stride memory target
checkpoint_interval = 100           # save checkpoint every N training steps

[logging]
level = "info"                      # trace, debug, info, warn, error
format = "auto"                     # auto (pretty on TTY, JSON otherwise), json, pretty, text, human

[prefix_cache]
enabled = true
# max_blocks = 128                  # omitted by default; auto resolves to half the KV block pool

[speculative]
enabled = false
method = "off"
num_speculative_tokens = 4
draft_layers = 8

[streaming_prefill]
mode = "auto"                              # backend-qualified long-prefill dispatch
threshold_tokens = "auto"                  # backend crossover, or a positive override
tile_tokens = "auto"                       # ordinary inference/non-tape tile
tape_tile_tokens = "auto"                  # tape-authoritative training tile
detached_full_attn_tile_tokens = "auto"    # materialized full-attention tile
last_token_lm_head = true
```

Startup selects the built-in `Qwen3.5-4B` defaults profile and logs it. The
profile preserves the official template thinking default for ordinary serving,
uses `enable_thinking=false` as the eval-mode default for deterministic
tool-agent loops, resolves adapters from `model.adapter_dir` or
`<model.path>/adapters`, and supports the official Qwen chat template from
`chat_template.jinja` or `tokenizer_config.json` with `enable_thinking` and
tool-call rendering.

### Key Environment Variables

| Variable | Description |
|----------|-------------|
| `KILN_CONFIG` | Path to config file (default: `kiln.toml`) |
| `KILN_BATCHING_MODE` | `auto`, `enabled`, or `disabled` actor selection |
| `KILN_BATCHING_ROWWISE_DECODE` | Strict boolean emergency rowwise comparison |
| `KILN_BATCHING_PREFIX_AWARE_ADMISSION` | Strict boolean strict-prefix admission policy |
| `KILN_BATCHING_PREFILL_ADMISSION_QUANTUM` | `auto` or 1–65536 prompt admissions per actor cycle |
| `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE` | `auto`, `enabled`, or `disabled` fallback direct-stream worker |
| `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH` | `auto` or 1–65536 fallback rows, clamped to effective decode width |
| `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US` | `auto` or non-negative collection delay in microseconds |
| `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS` | `auto` or strict boolean mixed-position fallback cohorts |
| `KILN_STREAMING_PREFILL_MODE` | `auto`, `enabled`, or `disabled` tiled-prefill dispatch |
| `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` | `auto` or a positive automatic-dispatch crossover |
| `KILN_STREAMING_PREFILL_TILE_TOKENS` | `auto` or a positive multiple-of-64 base tile |
| `KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS` | `auto` or a positive multiple-of-64 tape tile |
| `KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS` | `auto` or a positive multiple-of-64 detached full-attention tile |
| `KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD` | Strict boolean final-tile LM-head optimization |
| `KILN_ACCELERATOR_ROCM_KERNEL_PROFILE` | `qualified`, `portable_fallback`, or experimental-only `experimental_multiblock` complete ROCm route set |
| `KILN_MEMORY_GPU_MEMORY_GB` | Override GPU VRAM detection |
| `KILN_MEMORY_NUM_BLOCKS` | Override KV cache block count |
| `KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS` | Override gradient checkpoint segments |
| `KILN_TRAINING_NO_GRAD_CHECKPOINT` | Set to `1` to disable gradient checkpointing |
| `KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES` | `auto`, `enabled`, or `disabled` SFT boundary replay |
| `KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS` | Positive automatic-replay sequence-length crossover |
| `KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE` | `auto` or a positive sparse-anchor segment stride |
| `KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB` | Finite positive GiB target for automatic anchor spacing |
| `KILN_LOGGING_LEVEL` | Override log level |
| `KILN_LOGGING_FORMAT` | Override log format (`json` or `pretty`) |

The canonical startup override rule is mechanical:
`KILN_<SECTION>_<FIELD>`. The eight older batching spellings are deprecated
compatibility aliases, warn at startup, and cannot disagree with their
canonical counterpart. Startup resolves one immutable
`BatchingRuntimeConfig`. The state boundary uses its mode to construct the
actor, the decode-forward boundary applies its rowwise selector, and the actor
receives only the projected admission policy it owns. None of those production
paths rereads the process variables. The same resolved object constructs the
fallback direct-stream rendezvous worker without an environment reread. `auto`
enables the actor on CUDA, ROCm,
Vulkan, and CPU but disables it on Metal. CUDA/Vulkan use effective decode
width as the automatic admission quantum; ROCm/Metal/CPU use 4, clamped to that
effective width. CUDA alone uses backend-owned burst admission.

The fallback worker's separate `auto` policy is enabled on every real backend.
Its `(max_batch, wait_us, mixed_seq_lens)` defaults are CPU `(8, 0, false)`,
CUDA `(1, 0, false)`, ROCm `(8, 0, false)`, Metal `(8, 100, true)`, and Vulkan
`(64, 5000, true)`; width is then clamped to effective decode width. It is a
narrow compatibility route, not the production scheduling abstraction. Only
direct streaming effectively-greedy requests can submit to it, and only when
the batching actor is absent. Sampled, non-streaming, and actor-routed requests
bypass it. The worker is constructed independently, so it can be active but
unroutable while the actor is active.

Runtime observability preserves intent separately from execution:
`GET /v1/config` returns `configuration`, `actor_active`, and actual fallback
state under `batching`; health returns the same policy at
`decode_runtime.batching_configuration` and fallback state at
`decode_runtime.direct_decode_rendezvous`; trusted debug uses
`batching_engine.configuration` and
`batching_engine.direct_decode_rendezvous`. The actual state distinguishes
backend availability, actor activity, worker activity, and route availability.
See
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the complete schema,
validation, provenance, restart, and compatibility contract.

Accelerator execution has the same single-authority boundary. Resolved schema
`kiln.accelerator-runtime-policy.v12` includes
`accelerator.cuda_kernel_profile` and `accelerator.rocm_kernel_profile`.
Before constructing a CUDA backend, the server installs one immutable
fourteen-route CUDA policy. `native_default` preserves the prior default-on
full-attention QKV, GDN projection/prefill/gates/recurrent/normalization/decode,
fused convolution, decode LoRA add, and multi-block GDN routes without claiming
current hardware qualification. `portable_fallback` declines all fourteen.
CUDA backend dispatch no longer reads its former per-kernel environment
switches; a same-value policy reinstall is idempotent and a conflict fails.

The server also installs one complete 75-leaf
model/tensor policy before constructing a ROCm context or backend. `qualified` enables the production
full-attention QKV, GDN projection/gate/recurrent/normalization/decode,
head-major prefill, fused convolution, and LoRA decode routes, excluding the
slower gfx1151 multi-block GDN prefill experiment, plus qualified tensor-level
paged-attention splitting and GQA specializations. `portable_fallback` declines
all forty-five accelerated ROCm model/tensor routes while retaining thirty
correctness and bounded-work leaves for training, paged attention, concat,
finite checks, RMSNorm, and flash-attention geometry. `experimental_multiblock` adds that unqualified
prefill route and requires the experimental serving profile. Model dispatch
reads process-lifetime policy and low-level operations read their owning
context; the C++ paged-attention ABI receives explicit route values. No layer
or kernel rereads environment or admits per-operation mixtures. Retired route
variables and their legacy CUDA-spelled fallbacks are not accepted as ROCm
profile aliases. CUDA controls outside the fourteen backend-owned routes remain
separate migration work and are not represented by this profile.

Streaming prefill follows the same authority boundary. After the backend is
selected, startup resolves one immutable `StreamingPrefillRuntimeConfig` and
injects its private execution policy into `ModelRunner` and the server-owned
training runtime. Inference, SFT, GRPO, OPD, checkpoint planning, GDN segment
forwards, tape forwards, and detached full-attention forwards consume that
value; none reads public streaming environment variables while work is in
flight. A restart is required to change it.

`auto` dispatches at 256 prompt tokens on ROCm, 2048 on CUDA and Metal, and
never on CPU or Vulkan. The base/tape defaults are 1024/1024 on CUDA,
256/256 on ROCm, 2048/2048 on Metal and Vulkan, and 8192/8192 on CPU. Detached full-attention
defaults to 8192 everywhere; CUDA alone raises its boundary and tape-replay
variants to 65536. An explicit base tile is inherited by any `auto` tape or
detached field, including detached boundary/replay variants. A separate
detached value overrides all three detached variants. Every concrete tile is a
positive multiple of the 64-token GDN chunk size.

ROCm also owns a cross-subsystem numerical contract: when its production
batching actor is effective, the actor prefill ceiling must equal the 256-token
streaming tile, direct streaming must become eligible no later than that first
split, and the combined actor budget must fit one full tile beside every
effective decode row. Startup rejects violations before serving traffic. This
keeps actor and direct routes from silently applying different deterministic
prompt partitions.

The resolved structure is intentionally richer than one enabled flag. It
retains configured source, backend policy, effective dispatch rule, threshold
override applicability, each effective tile and inheritance source, and the
last-token LM-head selector. Ordinary generation, prompt-logprob scoring,
native training, local OPD teachers, MTP alignment, checkpoint planning, and
the benchmark all consume this one value. The prompt-logprob teacher identity
hashes every field under inference-contract v2, so policy drift changes the
teacher revision and cannot reuse identity-bound logits. `/v1/config` exposes
the policy at `streaming_prefill`, health at
`prefill_runtime.streaming_prefill`, and trusted debug at top-level
`streaming_prefill`. This makes a long-prefill pause attributable to scheduling,
tiled model work, or memory activity without guessing from an ambient shell.

Checkpointed SFT has a parallel typed authority boundary. Startup resolves the
four `[training]` fields into one integral, copyable
`CheckpointBoundaryPolicy`: replay mode defaults to `auto`, the automatic
crossover defaults to 8192 sequence tokens, anchor stride defaults to `auto`,
and its memory target defaults to 6.0 GiB. Automatic replay starts at
`seq_len >= threshold`; `enabled` always replays sparse boundaries and
`disabled` retains every boundary. A concrete positive stride wins. Otherwise,
an automatic one-segment shape uses stride 1; for larger shapes the resolver
sizes one boundary from sequence length, hidden width, and element size,
reserves a replay slot, and spaces the remaining anchors across the checkpoint
segments within the cache target.

The canonical overrides derive mechanically as `KILN_TRAINING_<FIELD>`. The
four old unsectioned names (`KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES`,
`KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS`,
`KILN_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE`, and
`KILN_CHECKPOINT_BOUNDARY_CACHE_GB`) remain strict warning aliases. Invalid,
non-Unicode, or conflicting inputs fail startup. After resolution, neither
training admission nor the trainer reads those variables: SFT memory preflight
and boundary execution call the same policy methods, preventing estimate/runtime
drift. GRPO and OPD do not execute the SFT boundary-spooling path.

Every training mode records this policy in its planning identity. GRPO and OPD
use `kiln.training-checkpoint-planning.v3`; SFT uses v4, which adds the pinned
backend loss route described above. A field change, a checkpoint carrying the
former v2 planning schema, or an SFT checkpoint carrying v3 is exact-resume
drift and fails closed rather than continuing with different memory/replay or
loss behavior. The immutable runtime object appears as
`training.checkpoint_boundary_policy` in
`GET /v1/config`, `GET /health`, and trusted `GET /v1/debug/model-state`; the
serialized fields are `recompute_mode`, `recompute_threshold_tokens`,
`anchor_stride` (`null` means automatic), and `cache_target_bytes`. The
dashboard renders the same resolved fields under Runtime config → Training. A
restart is required to change them.

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

Returns uptime, model info, scheduler statistics, GPU memory breakdown, active
adapter, training queue state, and the immutable SFT checkpoint-boundary policy.

### Debug Model State (`GET /v1/debug/model-state`)

Trusted eval/debug endpoint for answering "what model state am I actually
hitting?" It is available only when typed `server.debug_model_state=true` or
`server.eval_mode=true`, and returns no prompt or user message data. The
response includes model path, served model id, active defaults profile,
active/loaded adapters with adapter weight hashes, config hashes, selected
`KILN_*` flags, batching-engine status, thinking defaults, the resolved SFT
checkpoint-boundary policy, and aggregate cache counts.

## Phase status (2026-05-13)

Phase 6 (performance optimization) is closed. The post-#534 perf shortlist concluded with PRs #525 / #526 (SGLang RadixAttention), #210 / #206 (Marlin pack determinism + BF16 cleanup), #222 (FP8 KV opt-in), and #536 (native MTP self-spec, null at α=0.69). Phase 11 (eval as a first-class surface — `kiln-eval` crate, `/v1/eval/*` API, dataset → suite synthesis, judgment flywheel, post-training auto-eval, drill-in UI) shipped as a single bundled commit. Active work is now Phase 7 (developer experience). For current decode numbers see `BENCHMARKS.md`; for the live profiling hotspot table see `PROFILING.md`.

## #1082 migration substrate (Phase 1 onward)

The candle removal epic ([#1082](https://github.com/ericflo/kiln/issues/1082))
introduces six new crates that together replace candle's tensor /
autograd / optimizer / graph-capture surfaces. They ship in parallel
with the existing crates; the migration flips per-op call sites onto
the new substrate over many small PRs while the candle path stays
buildable behind feature flags. Phase 7 of the issue deletes
`vendor/candle-core/` once every backend has a matching kiln-tensor
implementation.

### Six new crates

| Crate | Replaces | What's shipped today |
|---|---|---|
| `kiln-tensor` | `candle_core::{Tensor, Storage, Device, DType, TensorId, Layout, Error, Result}` + `bail!` / `ensure!` macros + safetensors load + 11 CPU op families + `StreamPlanner` + `Allocator` trait + `CpuAllocator` + `Activation` registry | Full Phase 1 scaffold, GPU storage impls behind `cuda` / `metal` / `vulkan` features |
| `kiln-blas` | candle's locked `CUBLAS_GEMM_DEFAULT_TENSOR_OP` GEMM | Phase 0.8 cublasLt probe; production matmul lands in Phase 2 |
| `kiln-param` | `packed_weight_registry.rs` + `transposed_weight_cache.rs` + `marlin_proj.rs` + `fp8.rs` + `lora_loader.rs`'s weight-side concerns | `Parameter`, `ForwardStorage`, `OutputHead`, `AmpPolicy`, `content_hash_storage` |
| `kiln-autograd` | `candle_core::{Var, GradStore, BackpropOp}` + `vk_autograd.rs` lift | `Tape`, `GradStore`, `BackwardOp`, reverse-topo walker |
| `kiln-optim` | `kiln-train::trainer.rs::AdamWMoments` HashMap + Vulkan's `adamw_step_bf16.comp` | `OptimStep` trait, `AdamW` CPU reference, `MomentLocation` + `StochasticRoundingPolicy` |
| `kiln-graph` | `cuda_graph.rs:33-96` blocked bs>1 path + `vk_cmd_batch.rs` lift | `CapturedGraph` trait, `CaptureSession`, `AllocatorMode` (re-exports kiln-tensor's), `CaptureError::DanglingPointer` |

### Dispatch flow

```
User call site
  └─ kiln_tensor::ops::matmul(&a, &b)
       └─ dispatch2(&MatmulOp, &a, &b)
            └─ match a.device() {
                 Device::Cpu     -> op.cpu_fwd(&a, &b),
                 Device::Cuda(_) -> op.cuda_fwd(&a, &b) (default None ⇒ fallback to cpu_fwd),
                 Device::Metal(_)-> op.metal_fwd(...),
                 Device::Vulkan(_)-> op.vulkan_fwd(...),
               }
            └─ Op records on the autograd tape (Phase 6a) if requires_grad
            └─ Forward returns kiln_tensor::Tensor (zero-copy views via Layout)
```

Training flow:

```
forward → kiln_autograd::Tape::record per op
       → loss tensor
       → tape.backward(loss_id, seed_grad, accumulator) → GradStore
       → optimizer.step(parameter, grad) per parameter
            └─ reads parameter.amp_policy() for dispatch dtype
            └─ updates parameter.backward_storage() in place
            └─ (later) fused dequant → update → requant kernel for
              quantized forward_storage (Marlin / FP8)
       → Tape::clear() (anti-pattern 16: required before next forward)
```

Graph-capture flow (Phase 5):

```
allocator.warm(dtype, n, count) for every shape the captured graph needs
session = CaptureSession::begin()
allocator.set_mode(AllocatorMode::Frozen)
... per-backend graph capture ops, each session.pin(&tensor) ...
session.finalize()
// On every replay:
session.audit_pinned(&live) → Err(DanglingPointer{tensor_id}) if pinned id dropped
captured.replay()
```

### CPU op families (current)

Eleven `DeviceOp` impls (each with a CPU reference + parity tests):

| Op family | Trait | Migration target |
|---|---|---|
| `embedding` | `DeviceOp2` | `candle_core::Tensor::index_select`, `candle_nn::Embedding` |
| `rmsnorm` | `DeviceOp2` | candle's `RmsNorm` + 16 NVTX call sites |
| `add` / `sub` / `mul` / `div` (`ElementwiseOp`) | `DeviceOp2` | `Tensor::{add, sub, mul, div}`; 14 `kiln/residual` NVTX sites |
| `silu` / `sigmoid` (`ActivationOp`) | `DeviceOp1` | `Tensor::{silu, sigmoid}`, `Activation::SiLU` |
| `softmax_last_dim` | `DeviceOp1` | `candle_nn::ops::softmax_last_dim` |
| `matmul` | `DeviceOp2` | `Tensor::matmul` (canonical reference for kiln-blas) |
| `argmax_last_dim` | `DeviceOp1` | `Tensor::argmax_keepdim(-1)` + sampler greedy path |
| `cast` | `DeviceOp1` | `Tensor::to_dtype` (F32 ↔ BF16 ↔ F16; U32 ↔ I64) |
| `rope` | `DeviceOp3` | `candle_nn::rotary_emb::rope` (partial-rotary 0.25 for Qwen3.5-4B) |
| `l2_norm` | `DeviceOp1` | `Tensor::l2_normalize(-1)` (QK-norm path) |
| `mul_sigmoid_gate` | `DeviceOp2` | the packed silu*mul kernel from the MLP fusion work (PRs e44c2c84/2a44953a/da1b0467) |

Each impl exposes the same shape: `name()`, `determinism()`,
`cpu_fwd(...) -> Result<Option<Tensor>>` (the canonical reference),
default-`None` GPU forwards, and `bwd() -> Option<Box<dyn BackwardOp>>`.
Dispatchers in `kiln_tensor::device_op` (`dispatch1` / `dispatch2` /
`dispatch3`) pick the right backend method based on `Tensor::device()`
and fall through to CPU on `Ok(None)`.

### Determinism stance + parity tolerance

`kiln_tensor::Determinism` (Phase 1.11) classifies each op as either
`Constructive` (bit-identical across runs) or
`ToleranceBounded { dtype_band_key }`. The band keys reference rows
in `bench-results/parity-tolerance.csv` (Phase 0.4) which carries 416
`{op, dtype, backend}` cells. `server.deterministic = true` (or the strict
`KILN_SERVER_DETERMINISTIC=1` override) freezes the process-wide selector that kernel
implementations can consume. The metadata and selector do not, by themselves,
prove that every tolerance-bounded op has or selects a deterministic variant.

Anti-pattern 2 (every materializing `contiguous()` is logged) is wired
through `kiln_tensor::profile::emit_contiguous_copy()`, which
`Tensor::contiguous()` bumps on its non-fast-path branch.

### Reproducibility & anomaly-detection flags

The following controls cover different parts of the reproducibility and
fail-fast surface. They are **off by default**.

**`server.deterministic = true` / `KILN_SERVER_DETERMINISTIC=1`** is the typed,
immutable serving-repeatability selector
(`crates/kiln-tensor/src/determinism.rs`, `deterministic_enabled()`). The server
validates it before tensor initialization and forces the batching actor's
effective decode width to one. Multi-row BF16 GEMMs are individually bounded
and valid, but changing the live request cohort changes their shape and can
flip a close greedy-logit boundary; single-row decode removes that
scheduler-dependent numerical path and is covered by same-hardware restart
receipts.

This is not yet a blanket bitwise-deterministic kernel or training guarantee.
Kiln does not currently export `CUBLAS_WORKSPACE_CONFIG` from this setting, and
several tolerance-bounded backward implementations do not route through the
selector. Those require explicit backend controls and local hardware evidence;
the `Determinism` metadata is the audit inventory, not proof of implementation.

**`config.detect_anomaly = true`** is a request-local NaN/Inf trap for native
SFT, GRPO, and OPD. The trainer captures the value in immutable
`TapeOptions` when it opens each full or checkpoint-segment tape; there is no
process-global environment reader. `Tape::backward` scans every
`BackwardOp::apply` gradient before propagation, and the first violation
panics with the offending op name and tape position. This is more precise than
the mandatory optimizer-boundary finite check, but it adds a reduction and
potential device synchronization per produced gradient, so the default is
`false`. The CLI exposes `--detect-anomaly` and the browser exposes the same
opt-in under each training form's advanced controls. The effective job config
and exact checkpoint retain the selection.

Other training execution choices follow the same request boundary. SFT and
GRPO carry optional `adapter_smoke_prompts` as request data; GRPO records the
`shared_prefix_reference` policy; OPD records `sampler_segments` and the
`rollout_prompt_rendering` algorithm; and ECHO is represented only by the
typed `loss.echo` object. The training library does not reread process state
for any of those choices. The only runtime environment adapter retained in
`kiln-train` is the deliberately narrow remote-teacher credential provider,
which resolves a named secret without changing execution policy.

### What's NOT yet ported (subsequent PRs)

- Per-backend GPU `DeviceOp` impls — Phase 2 fills these in (CUDA via
  `kiln-blas` cublasLt, Metal via MPSMatrixMultiplication, Vulkan via
  `kiln-vulkan-kernel`'s existing compute pipelines + 33 vk_ops
  backward impls).
- Per-backend `Allocator` impls (`CudaAllocator` over `cudaMemPool_t`,
  `MetalAllocator` over `MTLHeap`, `VulkanAllocator` lifting
  `buffer_pool.rs`).
- Per-backend `CapturedGraph` impls (`kiln-graph-cuda` over
  `cudaGraph_t`, etc.).
- `Tensor` interior-mutability story (in-place ops + per-tensor
  version counter for anti-pattern 16 runtime enforcement).
- The actual call-site swaps from `candle_core::*` to
  `kiln_tensor::*` in `crates/kiln-model/src/`.

Phase 9's bench-gate enforces non-regression on every kiln-tensor
op's parity test + the per-tier baselines under
`bench-results/pre-migration-baseline/` (Phase 0.10).

### Phase 7 kt_api production wiring snapshot (2026-05-25)

The Tier-1 leaf crates (see `docs/CANDLE_REMOVAL_PLAN.md`) now expose
kt-typed entrypoints at every production call site. Concretely:

- **5/5 Tier-1 kernel crates have kt_api production wires** —
  `kiln-rmsnorm-kernel`, `kiln-conv1d-kernel`, `kiln-marlin-gemm`,
  `kiln-flash-attn`, and `kiln-gdn-kernel`. The candle-typed surface on
  each crate stays in place for back-compat but is no longer called
  from `kiln-model::forward` or `kiln-train::cuda_train`. gdn closed at
  10/10 wires (PR-equivalent commit `d4a9ec33`); flash-attn at 5/5;
  rmsnorm at 25/25 smoke tests green; marlin + conv1d ditto.
- **kt-API helpers wired at ~95 production call sites** across
  `crates/kiln-model/src/forward.rs` (decode + prefill kernels,
  paged-decode, gdn chunk/recurrent/qk-norm/gates/gated-rms-norm,
  flash-attn paged decode variants) and `crates/kiln-train/src/cuda_train.rs`
  (training-side rmsnorm, flash-attn fwd/bwd, marlin matmul, gdn
  forward-substitution/gates). Each wire is a borrow-zero-copy path
  through `kiln_kt_bridge`; no extra alloc on the hot path.
- **DeviceBuffer Metal variant landed.** `kiln_tensor::DeviceBuffer`
  now has a `Metal(MetalStorage)` arm alongside `Cpu` / `Cuda` /
  `Vulkan`, giving every kt-typed op a uniform handle on Apple
  Silicon. Paired with the `MetalAllocator` (see "kt-tensor allocator
  impls" in `bench-results/cuda-graph-status.md`), the Metal backend
  reaches the same lifecycle contract as CPU + CUDA + Vulkan.
- **The bs>1 CUDA graph implementation is unavailable.** Historical
  allocation suspects are closed, but real concurrent serving still poisoned
  the CUDA context. It is not a hidden opt-in. Re-entry requires a source
  change plus NVIDIA sanitizer, parity, resilience, and throughput evidence;
  `bench-results/cuda-graph-status.md` retains the historical investigation.
- **9 Vulkan ops + 15 Metal kind tags wired through real kernels.**
  The Vulkan backend has 9 ops on real `kiln-vulkan-kernel` compute
  pipelines (the rest are `Ok(None)` fall-through to CPU reference);
  the Metal backend has 15 op-kind tags wired through real MPS / MSL
  kernels via `kiln-mps`. Together with the kt_api wires above,
  `KILN_USE_KT_API_ALL=1` is end-to-end runnable on CUDA today and
  partially runnable on Metal + Vulkan.

The remaining `kiln-model` candle dependency is the diffuse 27-file
forward.rs surface (PagedKvCacheKt finalization, `try_kt_*` gate
demotion, fallback-branch deletion) and the Tier-2 crates
(`kiln-opd-loss-kernel`, `kiln-flce-kernel` Phase B closeout,
`kiln-vulkan-kernel` shim cleanup). See `docs/CANDLE_REMOVAL_PLAN.md`
for the full sequence to vendor delete.

### Phase 7 Metal backend snapshot (2026-05-31)

The candle→kt forward-flip (driven on the Linux CUDA box) had left the
Metal build uncompilable — 80 errors where `forward.rs` passed kt
`Tensor`s into `backend/metal.rs` kernel helpers that still took
`candle_core::Tensor`. No one could catch it without Apple Silicon. This
is now resolved and validated on an M1 (16 GiB, the consumer-floor tier):

- **`cargo build --features metal` is green** and the entire forward.rs
  **free-function Metal kernel path is candle-CORE-free.** 50 fused
  kernels (MLP gate+up / silu*mul, transposed-coop GEMV + fused-QKV,
  LM head + argmax, RoPE, LoRA-add, GDN gates / qk-norm / decode /
  prefill / recurrent, gated-RMSNorm, attn-gate) source storage via the
  kt `MetalStorage` downcast, device/encoder/pipelines via the kt
  `MetalCompanion`, and output via `MetalStorage::zeros_kt` — all over
  the `candle_metal_kernels` + `objc2_metal` substrate the `metal`
  feature already pulls (and that Phase 7 keeps). Pipeline getters take
  `&dyn MetalPipelineHost`, impl'd for both the candle `MetalDevice`
  (transitional) and the kt `MetalCompanion` (candle-free), so one
  compiled pipeline serves both with no call-site churn.
- **kt Metal host I/O + sync substrate** lands in `kiln-tensor`:
  `MetalCompanion::wait_until_completed` (the host-read sync point),
  `host_to_metal_copy` / `metal_to_host_copy`, and Metal arms on
  `Tensor::{to_device, from_vec_on, zeros_on, from_raw_bytes_on,
  contiguous}` — all over `StorageModeShared` UMA (zero-copy memcpy,
  no PCIe hop). Generic `DeviceOp{1,2,3}` ops fall back to a Metal-only
  host round-trip when no native kernel exists (CUDA/Vulkan keep their
  loud "implement the kernel" failure, so missing GPU kernels aren't
  masked).
- **Two on-M1 e2e gates pass**
  (`crates/kiln-server/tests/real_model_integration.rs`): the FP32
  naive path and a BF16 variant that drives the fused decode kernels
  end-to-end through the real `ModelRunner` + HTTP layer. Plus the first
  Metal parity suite, `crates/kiln-tensor/tests/metal_ops_parity.rs`
  (9 tests, kt-Metal vs the canonical kt-CPU reference).
- **Remaining for DoD-101 (candle-free Metal):** the trait-method path
  (paged-attn d256, GDN recurrent/chunk/conv1d, gdn_in_proj, paged-kv
  read) still bridges kt→candle→kt (~111 host-copy sites); flipping it
  is both candle removal and a Metal decode perf win. The candle-era
  in-file kernel tests in `forward.rs` + `metal.rs` need migration to
  kt-native inputs (they broke from the flip; tracked follow-up). Phase
  2 Metal perf (`simdgroup_matrix`, `MTLHeap`) and Phase 6c Metal
  backward parity remain.

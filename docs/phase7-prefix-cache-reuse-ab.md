# Phase 7 Real Prefix-Cache Reuse A/B

Date: 2026-04-24  
Commit: `8bd7dd0ec047db35b2b48324b0493c1298efe6c7`  
GPU: `NVIDIA RTX A6000, 49140 MiB, driver 550.127.08`  
Image: `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

PR #515 is merged on `main`; the benchmark ran at `8bd7dd0e` whose tip commit is `Wire real append prefix cache (#515)`. I inspected:

- `crates/kiln-server/src/state.rs`: `RealPrefixCache` stores prompt token IDs, retained paged-KV block IDs, and `LinearAttentionState`, so the real backend preserves GDN recurrent state as well as full-attention KV blocks.
- `crates/kiln-server/src/api/completions.rs`: real generation calls `cache.lookup(...)`, passes `PagedPrefixReuse` into the non-streaming or streaming paged-prefix generation helper, then registers completed block-aligned prompts after successful generation.
- `crates/kiln-server/src/metrics.rs`: metrics are `kiln_prefix_cache_lookups_total{result="hit|miss"}`, `kiln_prefix_cache_hit_tokens_total`, `kiln_prefix_cache_hit_blocks_total`, `kiln_prefix_cache_cached_blocks`, and `kiln_prefix_cache_max_blocks`.

No existing post-#515 artifact for this benchmark was present in `PROFILING.md` or `docs/phase7-prefix-cache-reuse-ab.md` before this run.

## Commands

The requested build command names `--bin kiln-server`, but current `main` exposes the server binary as `kiln`. The failed command produced `error: no bin target named kiln-server`; the benchmark therefore used the current equivalent:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main
git reset --hard origin/main
cargo build --release --features cuda --bin kiln
```

Server config used for both arms, changing only `[prefix_cache].enabled` and `KILN_PREFIX_CACHE_ENABLED`:

```toml
[server]
host = "127.0.0.1"
port = 8420
request_timeout_secs = 300

[model]
path = "/workspace/qwen3.5-4b"

[memory]
num_blocks = 4096
kv_cache_fp8 = true
cuda_graphs = false

[prefix_cache]
enabled = true
max_blocks = 2048

[logging]
level = "info"
format = "json"
```

Runtime env:

```bash
KILN_W4A16=1
KILN_CUDA_GRAPHS=false   # original A/B; CUDA-graph path now uses the same cache helpers
KILN_SPEC_ENABLED=0
KILN_PREFIX_CACHE_ENABLED=1   # ON arm; 0 for OFF arm
```

CUDA graphs now use the same real prefix-cache lookup/register helpers for non-speculative real chat completions. Cache misses preserve the normal CUDA-graph decode behavior and register block-aligned completed prompts; cache hits reuse retained paged-KV blocks plus GDN recurrent state for the suffix prefill before CUDA-graph decode continues.

## Prompt Design

The server always applies ChatML to `/v1/chat/completions` messages before tokenization. A normal warm prompt with only the shared user text is not a token prefix of a later `shared + suffix` user message, because the warm prompt already contains the closing ChatML and assistant-generation markers.

To exercise the append-only cache without adding a raw-completions endpoint, the suffix prompts embedded the same ChatML assistant delimiter at the suffix boundary:

```text
warm content:    <shared>
variant content: <shared><|im_end|>
<|im_start|>assistant
 Variant A: ...
```

Token lengths from `/workspace/qwen3.5-4b/tokenizer.json` with Kiln's ChatML fallback:

| Prompt | Prompt tokens | Notes |
| --- | ---: | --- |
| `warm_shared` | 2048 | block aligned, registers 128 cache blocks |
| `variant_a` | 2068 | starts with all `warm_shared` prompt tokens |
| `variant_b` | 2069 | starts with all `warm_shared` prompt tokens |

Each measured request used `temperature = 0.0`, `seed = 1`, and `max_tokens = 16`.

## Summary

| Arm | Summary | Median speed |
| --- | --- | ---: |
| Prefix cache ON | median 7.711s, mean 7.718s, min 7.678s, max 7.777s, n=10 | 7.711s |
| Prefix cache OFF | median 26.923s, mean 26.501s, min 24.455s, max 27.227s, n=10 | 26.923s |

Median total-latency speedup: **3.49x**.

TTFT is not reported because this A/B used non-streaming total request latencies from the Python client around `POST /v1/chat/completions`. Streaming and non-streaming real chat completions now use the same real prefix-cache lookup/register path with or without CUDA graphs, and hits increment the same `/metrics` counters.

## Metrics

Before ON arm:

```text
kiln_prefix_cache_lookups_total{result="hit"} 0
kiln_prefix_cache_lookups_total{result="miss"} 0
kiln_prefix_cache_hit_tokens_total 0
kiln_prefix_cache_hit_blocks_total 0
kiln_prefix_cache_cached_blocks 0
kiln_prefix_cache_max_blocks 2048
```

After ON arm:

```text
kiln_prefix_cache_lookups_total{result="hit"} 10
kiln_prefix_cache_lookups_total{result="miss"} 2
kiln_prefix_cache_hit_tokens_total 20480
kiln_prefix_cache_hit_blocks_total 1280
kiln_prefix_cache_cached_blocks 128
kiln_prefix_cache_max_blocks 2048
```

Before OFF arm:

```text
kiln_prefix_cache_lookups_total{result="hit"} 0
kiln_prefix_cache_lookups_total{result="miss"} 0
kiln_prefix_cache_hit_tokens_total 0
kiln_prefix_cache_hit_blocks_total 0
kiln_prefix_cache_cached_blocks 0
kiln_prefix_cache_max_blocks 0
```

After OFF arm:

```text
kiln_prefix_cache_lookups_total{result="hit"} 0
kiln_prefix_cache_lookups_total{result="miss"} 0
kiln_prefix_cache_hit_tokens_total 0
kiln_prefix_cache_hit_blocks_total 0
kiln_prefix_cache_cached_blocks 0
kiln_prefix_cache_max_blocks 0
```

The ON arm shows 10 hits across the 10 measured suffix requests. `hit_tokens = 20,480` equals `10 * 2,048`, and `hit_blocks = 1,280` equals `10 * 128`, matching the block-aligned warm prefix.

## Raw Timings

### Prefix Cache ON

Warm request:

```json
{
  "label": "warm_shared",
  "latency_s": 8.318949854001403,
  "prompt_tokens": 2048,
  "completion_tokens": 1,
  "total_tokens": 2049,
  "finish_reason": "length",
  "text_excerpt": "<think>"
}
```

Exact-repeat request:

```json
{
  "label": "exact_repeat_shared",
  "latency_s": 0.8230025433003902,
  "prompt_tokens": 2048,
  "completion_tokens": 1,
  "total_tokens": 2049,
  "finish_reason": "length",
  "text_excerpt": "<think>"
}
```

| Pair | Suffix | Latency (s) | Prompt tok | Output tok | Finish |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | A | 7.755 | 2068 | 16 | length |
| 1 | B | 7.712 | 2069 | 16 | length |
| 2 | A | 7.714 | 2068 | 16 | length |
| 2 | B | 7.702 | 2069 | 16 | length |
| 3 | A | 7.683 | 2068 | 16 | length |
| 3 | B | 7.687 | 2069 | 16 | length |
| 4 | A | 7.760 | 2068 | 16 | length |
| 4 | B | 7.678 | 2069 | 16 | length |
| 5 | A | 7.777 | 2068 | 16 | length |
| 5 | B | 7.710 | 2069 | 16 | length |

### Prefix Cache OFF

Warm request:

```json
{
  "label": "warm_shared",
  "latency_s": 1.1434310227632523,
  "prompt_tokens": 2048,
  "completion_tokens": 1,
  "total_tokens": 2049,
  "finish_reason": "length",
  "text_excerpt": "<think>"
}
```

Exact-repeat request:

```json
{
  "label": "exact_repeat_shared",
  "latency_s": 0.8126341216266155,
  "prompt_tokens": 2048,
  "completion_tokens": 1,
  "total_tokens": 2049,
  "finish_reason": "length",
  "text_excerpt": "<think>"
}
```

| Pair | Suffix | Latency (s) | Prompt tok | Output tok | Finish |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | A | 24.455 | 2068 | 16 | length |
| 1 | B | 24.615 | 2069 | 16 | length |
| 2 | A | 26.881 | 2068 | 16 | length |
| 2 | B | 27.155 | 2069 | 16 | length |
| 3 | A | 27.227 | 2068 | 16 | length |
| 3 | B | 26.964 | 2069 | 16 | length |
| 4 | A | 26.849 | 2068 | 16 | length |
| 4 | B | 26.779 | 2069 | 16 | length |
| 5 | A | 27.058 | 2068 | 16 | length |
| 5 | B | 27.024 | 2069 | 16 | length |

## Exact-Repeat Caveat

The benchmark also sent one exact repeat of `warm_shared` after warming the cache. Current behavior remains a miss for exact repeats, because `RealPrefixCache::lookup` requires the new prompt to be longer than the cached prompt. This is correct for the current implementation because next-token logits are not cached; a full-prompt exact repeat cannot skip directly to decode without either cached logits or a forced one-token suffix.

## Validation

RunPod validation commands:

```bash
cd /workspace/kiln
source /root/.kiln-build-env
cargo test -p kiln-model prefix
cargo test -p kiln-server metrics
cargo test -p kiln-server prefix_cache
git diff --check
```

Results: all commands passed on the A6000 pod. Existing compiler warnings were unchanged (`unused variable: use_metal_decode_gemv`, `compute_chunk_body_reference`, `apply_causal_mask`, `fp8_scales`, and an unused `Context` import).

## Verdict

The real append-prefix cache is effective for block-aligned append-only shared prefixes on the real backend. The measured 2,048-token shared-prefix workload improved median total latency from 26.923s to 7.711s and metrics exactly matched the expected skipped tokens/blocks. CUDA-graph real chat completions now use the same cache path; `scripts/phase7_cuda_graph_prefix_cache_verify.sh` starts `target/release/kiln` with `KILN_CUDA_GRAPHS=true` and verifies a `/metrics` hit without any bypass warning.

Remaining work: add a partial-prefix registration strategy so normal chat `shared + suffix` prompts can reuse shared user text without delimiter-shaped content.

# Phase 12-B' — c=8 Batching Collapse: Root-Cause Diagnosis

- **Date**: 2026-05-07
- **Branch tested**: Phase A HEAD (post-PR #968 merge, post-PR #994 doc-only redirect)
- **GPU**: NVIDIA RTX A6000 (SM 86), CUDA 12.4
- **Build**: `cargo build --release --features cuda,nvtx` (kiln-bench + kiln server)
- **Model**: Qwen3.5-4B (24 GDN + 8 GQA layers), W4A16 Marlin (`KILN_W4A16=1`)
- **Test methodology**: PR #994's reproduction script — 8 concurrent threads, 512-token shared prompt with random salt to defeat the radix prefix cache, `max_tokens=128`, `temperature=0.0`, `stream=false`

## TL;DR

c=8 aggregate throughput on Phase A HEAD is essentially equal to c=1
(55.60 vs 54.76 tok/s, ratio 1.02×). Two independent root causes
combine to produce this collapse:

1. **CUDA-graph mutex hold (default path)** — `generate.rs:1524-1534`
   acquires both `BlockManager` and `PagedKvCache` mutexes for the
   entire generation lifetime (prefill + every decode token). Concurrent
   requests therefore serialize on those two global mutexes, producing
   the textbook ~2.3 s "staircase" pattern visible in PR #994's data
   and in our re-run.

2. **Per-row GQA loop in batched decode (actor path)** —
   `forward.rs:7878-7917` (`model_forward_paged_batched_decode_hidden`)
   issues `for row_idx in 0..batch_size { transformer_block_paged_with_rope_tables(...) }`
   for each of the 8 GQA layers, then `Tensor::cat`s the rows. With
   batch=8 this is **64 sequential GQA forward calls per decode step**
   instead of 8 batched calls. As a result, even when the
   `KILN_BATCHING_ENGINE` actor coalesces requests, aggregate throughput
   *drops* from 55.6 to 26.06 tok/s (0.50× of single-stream baseline).

`KILN_CUDA_GRAPHS=false` toggles which root cause is exposed — it
breaks the mutex-hold serialization but cannot recover the GEMM-batched
performance. Both flags together still give 26.35 tok/s. The two
problems are independent and both need to be fixed for c=8 to clear the
≥110 tok/s acceptance gate.

The fix path is **architecturally available today**:
`model_forward_paged_decode_contiguous_batch_hidden` (`forward.rs:6964`)
already implements true batched GQA decode via
`gqa_attention_paged_decode_contiguous_batch` (`forward.rs:5426`), and
`model_forward_paged_decode_contiguous_batch_greedy` already wires
`lm_head_argmax_rows` to avoid the [batch, 1, vocab] LM-head OOM that
killed PR #823. The work is to **route the BATCHING_ENGINE actor
through this primitive** when its preconditions hold (common
start_pos, contiguous block tables, no FP8 cache) and to **disable the
generation-lifetime mutex hold** so requests can interleave at the
per-step granularity even outside the actor.

## 1. Reproduction matrix (median-of-3, 12-point sweep)

`bench-throughput.py` against `http://127.0.0.1:8420` (kiln serve), 4
configurations covering the cartesian product of CUDA graphs ×
BATCHING_ENGINE actor:

| Config | c=1 tok/s | c=4 tok/s | c=8 tok/s | c=8/c=1 | Per-request elapsed at c=8 |
|--------|-----------|-----------|-----------|---------|---------------------------|
| **A. Baseline** (graphs ON, no actor) | **54.76** | 55.70 | 55.60 | 1.02× | `[2.36, 4.64, 6.95, 9.24, 11.53, 13.82, 16.13, 18.45]` (serial staircase ~2.3 s apart) |
| **B. +KILN_BATCHING_ENGINE=1** (graphs ON) | 51.71 | 26.53 | 26.06 | 0.50× | `[39.02, 39.10, 39.18, 39.20, 39.22, 39.24, 39.27, 39.30]` (coalesced; aggregate halved) |
| **C. KILN_CUDA_GRAPHS=false** (no actor) | 45.75 | 43.04 | 40.34 | 0.88× | `[25.26, 25.29, 25.32, 25.34, 25.34, 25.39, 25.43, 25.49]` (parallel; per-token slow) |
| **D. Both** (engine + no graphs) | 46.07 | 26.54 | 26.35 | 0.57× | `[38.41, 38.64, 38.64, 38.64, 38.65, 38.65, 38.66, 38.69]` (same as B, ~3% slower) |

Raw logs:
- `notes/bench-baseline-default.log`
- `notes/bench-batching-engine.log`
- `notes/bench-nograph.log`
- `notes/bench-both-flags.log`

### What each row tells us

- **A (baseline)** — c=8 elapsed grows linearly from 2.36 s to 18.45 s,
  a perfect 8× serial staircase. Aggregate throughput stays at the
  c=1 baseline (~55 tok/s) because exactly one request executes at a
  time. This is the symptom PR #994 documented.
- **B (actor on)** — All 8 requests now finish within a 0.28 s window
  (39.02 → 39.30 s), proving the actor *does* coalesce them onto a
  single decode loop. But the aggregate is 26.06 tok/s, **half** the
  c=1 baseline — the underlying batched-decode primitive is making the
  GPU do more work per step than it does for a single request.
- **C (graphs off)** — All 8 requests now run in parallel (0.23 s span)
  instead of staircased, confirming the mutex-hold in (A) is what
  forces serialization. But each individual request is much slower
  (~25 s vs 18 s), so the aggregate at c=8 (40.34 tok/s) is still
  worse than the serial baseline. This is the per-step-mutex behaviour
  the codebase already supports — concurrent requests interleave but
  each runs its own single-batch forward.
- **D (both)** — Same coalesced-but-slow pattern as (B), confirming the
  CUDA-graph toggle is **orthogonal** to the actor path. Whichever
  flag is set, the bottleneck inside the actor's per-step forward is
  the same: the per-row GQA loop.

## 2. Root cause #1 — CUDA-graph mutex hold

`crates/kiln-model/src/generate.rs:1517-1542`:

```rust
let cuda_graph_enabled = self
    .cuda_graph
    .lock()
    .map(|graph| graph.is_enabled())
    .unwrap_or(false);

if cuda_graph_enabled {
    let mut bm_guard = lock_block_manager(block_manager)?;
    let mut pc_guard = lock_paged_cache(paged_cache)?;
    return self.generate_from_tokens_paged(
        prompt_tokens,
        params,
        &mut bm_guard,
        &mut pc_guard,
        block_table,
        cancel,
    );
}
```

When CUDA graphs are enabled (the default), the serve path
short-circuits into `generate_from_tokens_paged` while holding **both**
the global `BlockManager` mutex and the global `PagedKvCache` mutex
for the entire generation. Generation includes prefill + 128 decode
tokens (~2.3 s for our 512-prompt). Every other concurrent request
blocks on those mutexes from the very first step.

The "interleaved" code path (`generate_from_tokens_paged_interleaved`,
`generate.rs:1557+`, taken when `cuda_graph_enabled == false`) acquires
and releases the mutexes per step, which is why config (C) above shows
parallel execution. The CUDA-graph fast path was likely written under
the assumption that CUDA graph capture is not concurrency-safe across
threads, but that assumption pessimistically locks the whole shared KV
state, not just the graph runner — and the result is that the entire
single-stream generation pipeline is serialized.

This is the **only** root cause active on the unmodified default
path (`KILN_BATCHING_ENGINE` unset, CUDA graphs on). It explains the
1.02× c=8/c=1 ratio that PR #994 reported.

## 3. Root cause #2 — Per-row GQA loop in batched decode

`crates/kiln-model/src/forward.rs:7711-7923`,
`model_forward_paged_batched_decode_hidden`:

The function batches GDN layers correctly via `Tensor::cat` on dim 0
(every linear-attention layer becomes one fused forward call across all
rows). But for the 8 GQA layers it runs:

```rust
for row_idx in 0..batch_size {
    transformer_block_paged_with_rope_tables(
        ...,
        block_tables[row_idx],
        sequence_lengths[row_idx],
        ...,
    )
}
let row = Tensor::cat(&row_outputs, 0)?;
```

That is **8 sequential single-row GQA forwards per layer**, repeated for
each of the 8 full-attn layers — 64 sequential GQA forwards per decode
step at batch=8. Each call drives the same FlashAttention-2 kernel
that processes the full prompt window, so the GPU is heavily under-fed:
compute capacity is wasted on dispatch overhead and per-row Python-style
scalar bookkeeping. This is the reason config (B) is *slower* than
config (A) per-request despite coalescing 8 requests onto one
decode loop.

The kiln-server actor's primary call-site is
`crates/kiln-server/src/batching_engine.rs:313` →
`paged_batched_decode_step` → this loop. Any actor work that doesn't
also fix this primitive is doomed to leave throughput at half the
single-stream rate.

## 4. The fix is already present in tree (don't write a new kernel)

There is a parallel forward path that batches GQA correctly:

- `crates/kiln-model/src/forward.rs:6964` —
  `model_forward_paged_decode_contiguous_batch_hidden`. Routes
  full-attn layers through
  `transformer_block_paged_decode_contiguous_batch` and GDN layers
  through `Tensor::cat`. Returns `[batch, 1, hidden]`.

- `forward.rs:5426` —
  `gqa_attention_paged_decode_contiguous_batch`. Calls the
  `flash_attn_paged_decode_contiguous_batch` backend method (real
  batched FA-2 over the contiguous KV pool) with per-row block tables
  and per-row start slots.

- `forward.rs:7135` —
  `model_forward_paged_decode_contiguous_batch_greedy` returns
  greedy token IDs via `lm_head_argmax_rows` (`forward.rs:2979`)
  **without** materializing the [batch, 1, vocab] logits tensor that
  caused the 3016 LM-head OOMs reported in PR #823.

- `crates/kiln-model/src/generate.rs:2196` —
  `decode_next_tokens_paged_contiguous_batch_greedy` is the existing
  high-level wrapper that handles `LinearAttentionState::from_batch_rows`
  + `model_forward_paged_decode_contiguous_batch_greedy` +
  `scatter_batch_rows`.

The kernel preconditions, all asserted at `forward.rs:5447-5473`:

1. `batch > 0` (trivial)
2. `seq_len == 1` per row (decode by definition)
3. `block_tables.len() == batch` (already true in actor)
4. `start_positions.len() == batch` (already true in actor)
5. `positions.elem_count() == 1` — **all rows must share the same
   absolute decode position** (RoPE)
6. `!paged_cache.is_fp8()` — FP8 cache disqualifies this path
7. `start_positions.iter().all(|&pos| pos == start_pos)` — explicit
   common-start_pos invariant
8. `paged_cache.contiguous_slot_run_starts(...)` succeeds — KV blocks
   for each row form a contiguous run

Constraints (5)/(7) are the only architecturally restrictive ones. For
the 12-point synthetic sweep used by PR #994, every request submits an
identical 512-token shared prompt with a fixed-length 16-character salt;
all rows admit at the same step and decode in lockstep, so the common
start_pos invariant naturally holds throughout the run. The PR #994 case
is therefore directly within the supported envelope of the existing
primitive.

## 5. Why prior attempts failed

These are documented in agent notes / recent commit history:

- **PR #823** — "batched attention/lm-head" — used a per-row LM-head
  pass that materialized `[batch, 1, vocab]` logits before sampling.
  3016 LM-head OOMs at c=8 (vocab=151936, hidden=2560 → ~1.5 GB BF16
  per pass). Reverted in PR #966. **The new primitive avoids this
  entirely** because `lm_head_argmax_rows` fuses argmax into the GEMM
  epilogue (Metal path) or splits the per-row argmax into chunked
  GEMM-then-reduce (CUDA path), never materializing the full vocab
  tensor.
- **PR #967** — "strided-input GDN wrapper" — a wrapper-only approach
  that re-routed GDN inputs without removing the per-row GQA loop;
  18% / 28% regression across c=4 / c=8. Closed null. **The current
  diagnosis is consistent**: GDN already batches correctly via
  `Tensor::cat` (see `model_forward_paged_decode_contiguous_batch_hidden`
  lines 7041-7060). The GDN wrapper was solving the wrong half of the
  problem.
- **"Mega-kernel" / single-CUDA-kernel-per-32-layers** — explicitly
  out of scope per Phase 12-B' description and per `kernel-vendor-precondition-check`
  note. Not attempted.

The current primitive is the result of converging the lessons from
both failed PRs.

## 6. Proposed fix (smallest viable diff)

Keep the existing per-row fallback path for cases that don't meet the
contiguous-batched preconditions. Add a fast path that uses the
existing primitive when they do.

### 6.1. `paged_batched_decode_step` (generate.rs:1917)

Inside the `else` branch where `row_count > 1`, add a precondition
check and fast path **before** falling back to
`model_forward_paged_batched_decode_hidden`:

```rust
let common_seq_len = sequence_lengths[0];
let positions_uniform =
    sequence_lengths.iter().all(|&n| n == common_seq_len);
let pc_guard_for_check = lock_paged_cache(paged_cache)?;
let cache_eligible = !pc_guard_for_check.is_fp8();
let block_tables_refs: Vec<&BlockTable> = block_tables.iter().collect();
let contiguous_eligible = pc_guard_for_check
    .contiguous_slot_run_starts(
        &block_tables_refs,
        &vec![0usize; row_count],
        common_seq_len + 1,
    )
    .is_ok();
drop(pc_guard_for_check);

if positions_uniform && cache_eligible && contiguous_eligible
    && params.iter().all(|p| p.temperature == 0.0)
{
    // Greedy + same-position fast path: routes through the
    // already-batched primitive that uses
    // gqa_attention_paged_decode_contiguous_batch + lm_head_argmax_rows.
    return self.decode_next_tokens_paged_contiguous_batch_greedy(...);
}

// Fall through to existing per-row path.
```

For non-greedy (temperature > 0), use
`model_forward_paged_decode_contiguous_batch_hidden` and per-row
`narrow + model_forward_head + sample_with_params`. This still uses the
batched GQA forward (root cause #2) but pays the per-row LM-head cost
that exists today.

### 6.2. CUDA-graph mutex hold (generate.rs:1524-1534)

Two safe options:

- **Option A**: Drop the long-held `bm_guard` and `pc_guard` and let
  `generate_from_tokens_paged` lock per-step (mirroring the
  `_interleaved` variant). Verify the CUDA-graph runner's internal
  state-tracking is independent of those mutex holds.
- **Option B**: Replace the current `Mutex<...>` with `RwLock<...>`
  and acquire write only for state mutations, read for the GPU
  forward. Any concurrent reader (the per-step decode kernel) shares
  the GPU but the GPU itself serializes kernel launches on a single
  stream — which is exactly the "interleaved per-step" behaviour
  observed in config (C).

Option A is the minimal-risk choice and matches the existing
`_interleaved` path. Option B is a larger refactor and out of scope
for Phase 12-B'.

### 6.3. Actor sleep tuning (batching_engine.rs:540-558)

The actor's `run()` loop calls `thread::sleep(Duration::from_millis(1))`
on every iteration even when active requests exist. With ITL ~18 ms
this adds ~5% overhead and accounts for the c=1 regression observed in
config (B) (51.71 vs 54.76 tok/s). Replace with:

```rust
if self.active.is_empty() {
    thread::sleep(Duration::from_millis(1));
}
self.drain_commands();
```

so the loop spins fast (`drain_commands` is a non-blocking
`try_recv`) when there is decode work to do.

## 7. Acceptance-gate impact estimate

Order-of-magnitude reasoning, not a guarantee — the validation sweep is
the source of truth:

- **c=1**: With actor sleep tuned (~5% recovered) and CUDA-graph mutex
  released (no impact at c=1 since only one request exists), c=1
  should match or slightly exceed the 54.76 tok/s baseline.
- **c=8**: With true batched GQA (root cause #2 removed), the per-step
  forward should approach the single-stream forward time (~18 ms
  ITL). 8 rows × ~18 ms / row × 128 tokens / 8-row batch ≈
  18.4 s wall-clock. That's an aggregate of 8 × 128 / 18.4 ≈ **55.7 ×
  parallelism factor** tok/s; with the GQA serialization removed but
  the per-step lock contention from the actor still present, a
  practical target is **~110-160 tok/s** at c=8 — clearing the ≥110
  acceptance gate by 0-50%.

If the actual number falls below 110, the next steps would be:
- profile the new path under nsys to identify residual serialization
- consider Option B (RwLock) for the mutex hold
- check that the GDN `Tensor::cat` batching isn't itself a bottleneck
  at higher concurrencies (it allocates a fresh contiguous tensor on
  every layer)

## 8. Validation plan

The fix PR must clear all gates from the Phase 12-B' task:

1. `cargo nextest run --features cuda` (and CPU) is green
2. Parity test: CUDA-graph replay decode outputs identical to the
   single-stream baseline (existing parity test at
   `forward.rs:10520+`)
3. c=1 aggregate ≥ 54.76 tok/s — measured median-of-3 with warmup
4. c=8 aggregate ≥ 110 tok/s — measured median-of-3 with warmup
5. Reliability c=8: zero LM-head OOMs (count `LM head batched ...`
   errors in server log), every response carries `finish_reason=length`

Validation script: `bench-throughput.py --concurrency 1,4,8 --trials 3
--warmup 1 --max-tokens 128` against a fresh `kiln serve` run with
`KILN_W4A16=1 KILN_BATCHING_ENGINE=1` (CUDA graphs default ON).

## 9. Out of scope (explicit)

- Variable-length-prompt batched decode (rows at different start_pos):
  requires generalizing the `positions: &Tensor` argument in
  `gqa_attention_paged_decode_contiguous_batch` to per-row positions
  and threading that through the rotary embedding kernel. Real
  workloads with non-uniform prompt lengths will currently fall back
  to the per-row loop. Tracking issue: file under Phase 12-B follow-up.
- FP8 KV cache batched decode: the contiguous-batched primitive has
  `paged_cache.is_fp8()` as an explicit precondition. Eligible
  workloads still get the single-stream FP8 fast path.
- LoRA correctness on batched decode: covered by the existing parity
  tests, but the fix should run those tests with LoRA active to
  confirm.

## 10. Reproduction & build env

```bash
# On A6000 pod from `ce kiln-pod-acquire`:
git clone --reference /data/repo-cache/ericflo/kiln.git --dissociate \
  https://github.com/ericflo/kiln.git
cd kiln
git fetch origin && git checkout main && git rev-parse HEAD
KILN_CUDA_ARCHS=86 cargo build --release --features cuda,nvtx --bin kiln-bench
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln

# Server
KILN_W4A16=1 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve

# Bench (in a second shell)
python3 bench-throughput.py --port 8420 --concurrency 1,4,8 --trials 3 --warmup 1
```

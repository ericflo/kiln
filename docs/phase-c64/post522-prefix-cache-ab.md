# Phase 7 A/B: is the radix prefix cache the source of the post-#521 decode regression?

Date: 2026-04-24
HEAD under test: 821ccd3 (`phase7: refresh post-#521 profiling artifacts (#522)`, docs-only)
Binary: `target/release/kiln-bench` built from 0fda0e6 (PR #521, identical Rust as 821ccd3)
GPU / pod: RunPod A6000 (`kiln-pool-1777034537883864691`, `mfk88l8i8tab02`), `ghcr.io/ericflo/kiln-runpod:latest`
Image tooling: CUDA 12.4.1, nsys 2023.4.4, rustc 1.95.0, `KILN_CUDA_ARCHS=86`

## Motivation

PR #522 flagged a −7.9% decode tok/s regression on `kiln-bench` vs the post-#166 closing
baseline (49.76 → 45.85 tok/s, mean ITL 20.10 → 21.81 ms). Its parsimonious hypothesis
was that the radix prefix-cache lookup/registration hooks added in PRs #515/#520/#521 now
fire on every CUDA-graph chat-completion request.

This doc runs the A/B isolation the task brief asked for and reports a **structural +
empirical null result**: the prefix cache **cannot** be responsible for the `kiln-bench`
regression, because `kiln-bench` never enters the prefix-cache code path at all.

## TL;DR

- Toggling `KILN_PREFIX_CACHE_ENABLED` 0 ↔ 1 moves `kiln-bench` decode tok/s by less
  than the intra-arm run-to-run noise (48.84 vs 47.65 tok/s median-of-last-2, spread
  ≈ 2.5%; intra-arm spread is already 3–7 tok/s per arm).
- Neither arm recovers to the post-#166 49.76 tok/s baseline; both cluster near 45–51
  tok/s. So the "−7.9%" gap is real at the single-sample level but partly run-to-run
  noise overlaid on a smaller (~2–4%) steady-state drift.
- `crates/kiln-model/src/forward.rs` and `crates/kiln-server/src/bench.rs` contain
  **zero** references to `PrefixCache`, `RealPrefixCache`, or `prefix_cache`. The cache
  only fires through `generate_*_with_prefix_cache()` in `crates/kiln-model/src/generate.rs`,
  which is only called from `crates/kiln-server/src/api/completions.rs` (the HTTP
  OpenAI-compatible endpoints). `kiln-bench --paged` routes through
  `model_forward_paged*` directly and never touches those generate entry points.
- **Conclusion**: the prefix cache hooks are not the source of the bench-visible
  regression. Do not fix by gating / amortizing the cache. The next bisection should
  target CUDA-decode landings between post-#166 and post-#521 (candidates listed
  below).

## Methodology

Same shape as the post-#521 profile that #522 captured:

```bash
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_PREFIX_CACHE_ENABLED=<0|1> \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged \
  --prompt-tokens 512 \
  --max-output-tokens 128 \
  --skip-training \
  --prompt-subset humaneval \
  --chat-template \
  --latency-only \
  --temperature 0.0 \
  --seed 1
```

- Arm A: `KILN_PREFIX_CACHE_ENABLED=0` (explicit disable)
- Arm B: `KILN_PREFIX_CACHE_ENABLED=1` (current default; matches the #522 baseline)
- 3 back-to-back runs per arm in the same bench session
- Discard run 1 of each arm as cold-start TTFT artifact (per agent note
  `kiln-bench-prefix-warmup-required`)
- Report median-of-last-2 (i.e. mean of run 2 and run 3)

## Raw results

All three decode numbers per arm:

| Arm | Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms | TTFT ms |
|-----|-----|--------------|-------------|------------|------------|------------|---------|
| A (disabled) | 1 (cold, discard) | 45.94 | 21.77 | 21.62 | 27.25 | 362.2 | 362.2 |
| A (disabled) | 2 | 50.57 | 19.77 | 19.44 | 26.55 | 343.5 | 343.5 |
| A (disabled) | 3 | 47.10 | 21.23 | 21.09 | 26.46 | 360.5 | 360.5 |
| B (enabled)  | 1 (cold, discard) | 43.78 | 22.84 | 22.63 | 28.05 | 364.3 | 364.3 |
| B (enabled)  | 2 | 51.26 | 19.51 | 19.33 | 23.25 | 337.8 | 337.8 |
| B (enabled)  | 3 | 44.05 | 22.70 | 22.49 | 28.98 | 356.2 | 356.2 |

Median-of-last-2 per arm:

| Arm | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
|-----|--------------|-------------|------------|------------|------------|
| A (`KILN_PREFIX_CACHE_ENABLED=0`) | **48.84** | **20.50** | 20.27 | 26.51 | 352.0 |
| B (`KILN_PREFIX_CACHE_ENABLED=1`) | **47.65** | **21.11** | 20.91 | 26.11 | 347.0 |
| Δ (A − B)                          | **+1.19** (+2.5%) | **−0.61** (−2.9%) | −0.64 | +0.40 | +5.0 |

Reference baselines for context:

| Source | Decode tok/s | Mean ITL ms | P99 ITL ms |
|--------|--------------|-------------|------------|
| post-#166 closing baseline (PROFILING.md) | 49.76 | 20.10 | 25.46 |
| post-#521 single-sample in PR #522 | 45.85 | 21.81 | 26.36 |
| this A/B Arm A median | 48.84 | 20.50 | 26.51 |
| this A/B Arm B median | 47.65 | 21.11 | 26.11 |

Run-level intra-arm variance:

| Arm | min–max decode tok/s | spread | spread % of mean |
|-----|----------------------|--------|------------------|
| A runs 2–3 | 47.10–50.57 | 3.47 | 7.1% |
| B runs 2–3 | 44.05–51.26 | 7.21 | 15.1% |

The inter-arm delta (**+1.19 tok/s**, 2.5%) is **smaller than the intra-arm spread in
both arms**. Taking run 2 vs run 3 of each arm, Arm B run 2 (51.26) actually beat Arm A
run 2 (50.57) and Arm A run 3 (47.10). This is consistent with the flag having zero
effect and the observed noise being driven by something else (GPU frequency scaling,
CUDA graph capture reuse, mmap warm-up, etc.).

## Structural confirmation

Grep counts on the two files that define what `kiln-bench --paged` actually executes:

| File | Count of `prefix_cache`/`PrefixCache`/`RealPrefixCache` |
|------|--------------------------------------------------------|
| `crates/kiln-model/src/forward.rs` | 0 |
| `crates/kiln-server/src/bench.rs` | 0 |
| `crates/kiln-model/src/generate.rs` | 17 |

`kiln-bench --paged` routes the latency phase through `model_forward_paged` and its
last-token / streaming / greedy variants (`crates/kiln-server/src/bench.rs:22–25,
764–847`). None of those ever reach `generate_from_tokens_paged_interleaved_with_prefix_cache`
or its streaming sibling. The prefix cache is only wired into the HTTP chat / completion
handlers:

```
$ rg -n 'generate_.*_with_prefix_cache' --type=rust
crates/kiln-model/src/generate.rs:762   pub fn generate_paged_shared_tokens_with_prefix_cache(
crates/kiln-model/src/generate.rs:2764  pub fn generate_streaming_paged_shared_tokens_with_prefix_cache(
crates/kiln-server/src/api/completions.rs:523  let result = runner_guard.generate_paged_shared_tokens_with_prefix_cache(
crates/kiln-server/src/api/completions.rs:749  .generate_streaming_paged_shared_tokens_with_prefix_cache(
```

So the flag we just toggled has nothing to attach to in this workload. The A/B result
is consistent with the structural claim: we are measuring the same code with different
(unread) env vars.

Reading `crates/kiln-server/src/config.rs:492` confirms `KILN_PREFIX_CACHE_ENABLED`
feeds `PrefixCacheConfig::enabled`, which in turn gates the scheduler's `prefix_cache`
construction (`crates/kiln-scheduler/src/scheduler.rs:99–104`) and is plumbed to the
HTTP state (`crates/kiln-server/src/main.rs:180`). `kiln-bench` does not instantiate
the scheduler or the HTTP state — it builds `GpuWeights` + `PagedKvCache` and calls
`model_forward_paged` directly. The flag therefore cannot alter the work that bench
does.

## So where is the regression?

The hypothesis in PR #522 is falsified for the bench path. Candidates to bisect next
are CUDA-decode landings between the post-#166 baseline commit `c2579a1` and current
main `821ccd3` that modify `crates/kiln-model/src/forward.rs` or the backing CUDA
kernels. Restricting to CUDA-affecting PRs (not Metal-only):

1. **#461 Mmap transposed weight cache hits** — changes how the transposed weight
   cache is backed. The transposed cache is on the decode hot path for Marlin and MLP
   GEMVs. Prior baselines include "cached transposed projections eliminated 50.7% of
   decode GPU time" (agent note `kiln-cached-transpose-prefill-tradeoff`). Any change
   to hit/miss/mmap path can move decode.
2. **#506 + #508 transposed cache reliability / deferred writer** — same cache, landed
   one day before the #522 profile was captured. `#506` added 87 lines to
   `crates/kiln-model/src/transposed_weight_cache.rs`; `#508` deferred writes by 46
   lines. Either can add per-layer latency on decode if it changed when the cache is
   actually hit.
3. **#486 "enable fused GDN qk norm by default"** — this moved an opt-in fused kernel
   onto the default path. If the fused kernel happens to be neutral-or-slightly-slower
   at Qwen3.5-4B decode shapes (a known hazard for fusion with CUDA graph dispatch
   amortization; see agent note `kernel-vendor-precondition-check`), it would appear
   as a post-#166 decode regression.
4. **#466 "Fuse CUDA GDN gated RMSNorm"**, **#500 "CUDA GDN qk norm GQA fast path"**,
   **#498 "opt-in CUDA GDN decode fuse hook"** — three landings that added/changed
   CUDA GDN fusions on `forward.rs`. #498 in particular says "opt-in" but any dispatch
   branch it installed is always evaluated.
5. **#509 "Default CUDA streaming prefill for long prompts"** and **#519 "Speed up
   greedy paged prefill defaults"** — both explicitly prefill-targeted. Unlikely to
   move decode, but worth verifying by checking whether decode latency numbers in
   the #522 profile have any prefill-leakage (e.g. first-decode-token including
   residual prefill work).

The 45.85 → 49.76 delta is also partially noise, which the single-sample #522 capture
cannot separate. A proper post-#522 re-profile with median-of-3 (discarding cold) would
let us pin down whether the true steady-state regression is ~2–4% or the full 7.9%.

## Recommended next task

A kernel-level kill-switch bisection is cheap:

```bash
# Baseline (everything on, default post-#521)
KILN_W4A16=1 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench ...

# Kill GDN fusions one at a time, look for a single toggle that recovers ≥49 tok/s
KILN_DISABLE_FUSED_GDN_GATES=1 ...
KILN_DISABLE_GDN_KERNEL=1 ...
KILN_DISABLE_RMSNORM_KERNEL=1 ...
KILN_DISABLE_FUSED_CONV1D=1 ...
KILN_DISABLE_FUSED_PAGED_DECODE=1 ...
```

If a single kill-switch recovers the baseline, that isolates the regression to that
fusion's default path and we can decide between reverting it, making it opt-in, or
accepting the smaller gap. If none do, the regression is in the transposed-cache
plumbing (#461/#506/#508) or in something more subtle (mmap cold-start for the
transposed cache blobs, scheduler-less bench not seeing some optimization, etc.) and
the next step is `git bisect` on the bench binary across the 70+ post-#166 commits.

## What did NOT change

We explicitly did **not** modify the prefix-cache code in this PR:

- No code changes to `crates/kiln-core/src/radix/*`
- No code changes to `crates/kiln-model/src/generate.rs`
- No code changes to `crates/kiln-server/src/api/completions.rs`

This PR is findings-only per the task brief: *"If Arm A does NOT recover: findings-only
PR, no code change"*. Arm A did not recover (48.84 < 49.76, within noise of Arm B's
47.65), so no fix is in scope.

## Reproducing

Pod pool: `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`
Clone + build: `kiln-setup --clone` on the pod, then
`KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench`.
Run the two commands above (with `KILN_PREFIX_CACHE_ENABLED=0` and `=1`) three times
each, discard run 1 of each arm, report median-of-last-2. Expect similar noise;
the arm-level delta should again be smaller than the intra-arm spread.

## Bench logs

Archived to the session dir and uploaded as a task artifact under this PR's CI run for
reproducibility.

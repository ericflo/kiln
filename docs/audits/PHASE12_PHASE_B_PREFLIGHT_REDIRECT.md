# Phase 12 — Phase B (attention layer clusters fusion) preflight redirect

Date: 2026-05-07
Status: **Preflight failed gates. No code changes. Doc-only redirect PR (precedent: PR #131, #163, #164, #170).**
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM), CUDA 12.4
Pod: RunPod pool lease `pod-357e8de3004fca40f7d884ec` (runpod `6034jwt7rzsgt1`)
Base commit: `718f8021c3e81c163369c5fed968e51972ca040c` (Phase A hot-path foundation, PR #968)

## TL;DR

**Phase B as scoped — fuse the 8 full-attn (GQA) layers' pre-attn pipeline (RMSNorm → QKV
projection epilogue, then QK-norm → RoPE → paged-decode setup) within a 120 min / $60
wall-clock cap — cannot mathematically clear its own +5 % c=8 acceptance gate via
dispatch-level fusion under CUDA graphs.** The fusion-target region is ~13 % of decode
wall-clock; even an aggressive 50 % region saving lands the c=8 aggregate at ~66.2 tok/s,
just barely above the 65.6 tok/s floor — and only if Phase A's c=8/c=1 ≈ 1.04× batching
ratio holds (so per-step speedup translates 1:1 to aggregate). Dispatch-only fusion
under CUDA graphs is consistently null per PR #141 (`gated_rms_norm`) and PR #173
(`L2_QK_norm`), so the 50 % region saving requires a brand-new fused CUDA kernel
(Marlin W4A16 epilogue + RoPE + RMS), which is multi-day kernel-engineering work, not
120-min budget work.

In addition, this preflight produced two independently important measurements about
the Phase A merge (#968):

1. **The `kiln-pr968-batching-regression` agent note's c=1 = 36.69 tok/s collapse does
   NOT reproduce on this fresh A6000 build.** Median-of-3 c=1 throughput is 57.0 tok/s,
   *above* the pre-Phase-A baseline of 54.76 tok/s. Phase A's hot path is healthy at c=1.

2. **The same note's c=8 = 27.31 tok/s flat-batching collapse also does NOT reproduce.**
   Median-of-3 c=8 aggregate via concurrent OpenAI-compat HTTP load is 62.20 tok/s,
   matching the note's stated baseline of 62.45 within 0.4 %. **However** the c=8/c=1
   ratio is only **1.04×** (62.20 / 59.77), confirming that effective batching is
   essentially absent on Phase A HEAD. This is the latent finding worth following up
   on next: a working scheduler at c=8 should compound aggregate to several × c=1, not
   1.04×.

The implementation phase was therefore not entered; pod was used only for preflight
measurement and released. Pod cost ≈ $0.40 (≈ 50 min on a $0.49/hr A6000).

## Acceptance gates this preflight evaluated

The Phase B task spec required ALL of the following to hold to merge a code PR:

| Gate | Floor | Status this preflight |
|---|---|---|
| Parity (BF16 logits at full-attn taps, KILN_DISABLE_FUSED_ATTN_PRE A/B, 1e-3 tolerance) | pass | not exercised — implementation skipped |
| `cargo nextest run` green | pass | not exercised |
| c=1 no regression (Shape A) | ≥ 49.3 tok/s (-10 % floor on 54.76 baseline) | **57.0 tok/s** (median-of-3) — clears trivially |
| c=8 improvement (Shape A) | ≥ 65.6 tok/s (+5 % floor on 62.45 baseline) | baseline measured **62.20 tok/s** (median-of-3) — Phase B math ceiling sits at ~66.2 tok/s, clearance margin ≤ +0.6 tok/s without a brand-new fused CUDA kernel |
| Reliability (zero LM-head OOM, all c=8+ finish_reason=length) | pass | n/a (no harness run on a fused build) |
| nsys diff (fewer `ucopy_bf16`/rope ranges) | pass | n/a |
| CUDA graph replay outputs identical | pass | n/a |

Two of the gates (c=8 floor and the implicit "fits in 120 min / $60") are mathematically
incompatible with what Phase B can deliver via dispatch-level fusion alone. See § Math.

## Math: why dispatch-level fusion cannot clear the c=8 gate

The latest re-profile in PROFILING.md (post-PR #534, 2026-04-25, A6000) gives the
following NVTX wall-clock shares for Phase B's target regions in decode-dominated
capture:

| NVTX range | Decode share | Notes |
|---|---|---|
| `:kiln/proj/qkv` | ~3.1 % | full-attn QKV projection (GDN uses `gdn/in_proj`, accounted separately at 9.4 %) |
| `:kiln/attn/rope` | **8.4 %** | full-attn-only (the 24 GDN layers do not use RoPE) |
| `:kiln/attn/qk_norm` | < 2.6 % | full-attn QK norm (not in top-10) |
| transpose / reshape / setup | < 1 % | not in top-10 |
| **Total Phase B target region** | **~13 %** | upper bound; assumes everything outside `attn/full/decode_fused` (the paged-attention kernel itself, 3.8 %, which the spec says to PRESERVE not fuse) |

Amdahl ceiling on per-step latency under perfect (100 %) elimination of the target
region is `1 / (1 − 0.13) = 1.149×`. That is the absolute upper bound and would
require zero residual cost from the fused replacement.

Realistic dispatch-level fusion under CUDA graphs (the only kind feasible in 120 min):

| Saving on target region | Per-step speedup | c=8 aggregate (62.20 baseline × per-step speedup, since c=8/c=1 ≈ 1.0) | Gate (≥ 65.6) |
|---|---|---|---|
| 30 % | 1.041× | 64.7 tok/s | **fail** |
| 40 % | 1.055× | 65.6 tok/s | borderline |
| 50 % | 1.069× | 66.5 tok/s | pass with +0.9 tok/s margin |
| 70 % | 1.099× | 68.4 tok/s | pass |
| 100 % | 1.149× | 71.4 tok/s | pass |

Empirically, prior dispatch-level fusion attempts on similar regions under CUDA-graph
capture have closed null:

- **PR #141** — fused GDN gated RMSNorm: closed null. Graph dispatch already amortized.
- **PR #173** — opt-in fused L2 QK norm (`KILN_ENABLE_FUSED_L2_QK_NORM`): null median,
  variance reduction only.
- **PR #176** — big-fusion across recurrent + qk_norm + gated_norm: closed null
  ($14.99 burn).

These rejections are the strongest available evidence that **30–40 % effective region
saving is the realistic ceiling for dispatch-level fusion under CUDA graphs**, which
puts the c=8 outcome in the fail-to-borderline band — and well below the +5 % floor
required to merge.

The 50–70 % region-saving regime — which is what the gate actually requires — is only
reachable with a real fused CUDA kernel (one Marlin W4A16 GEMM epilogue + RoPE rotation
+ fused RMSNorm split across Q and K outputs, all in a single device kernel, sharing
register-resident intermediates instead of round-tripping through HBM via the
`OnceLock<DecodeBuffers>` registry that Phase A introduced). That is multi-day
kernel-engineering work, not 120-min budget work.

## Measurements: c=1 baseline (median-of-3)

Per the canonical Phase 6 bench protocol from the kiln skill:

```
KILN_W4A16=1 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training
```

| Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms |
|---:|---:|---:|---:|---:|
| 1 | 57.03 | 17.5 | 17.15 | 23.70 |
| 2 | 58.59 | 17.1 | 16.98 | 20.11 |
| 3 | 56.85 | 17.6 | 17.37 | 22.38 |
| **median** | **57.03** | 17.5 | 17.15 | 22.38 |

**Comparison vs prior baselines:**

| Source | c=1 decode tok/s | Δ vs this preflight |
|---|---:|---:|
| Pre-Phase-A baseline (kiln skill, post-PR #166) | 54.76 | this preflight is **+4.1 %** |
| `kiln-pr968-batching-regression` note (claimed) | 36.69 | this preflight is **+55.4 %** — does not reproduce |

Phase A's c=1 hot path is healthy on this fresh A6000 build. The note's −33 % c=1
regression appears to be either a transient measurement issue at the time the note
was saved, a cold-start / warmup artifact, or a different harness configuration. It
should not gate further Phase 12 work.

## Measurements: c=8 batching baseline (median-of-3)

Concurrent OpenAI-compat HTTP load against `kiln serve` (in-pod), shared ~512 token
prompt with random per-request salt to defeat radix prefix-output caching, max_tokens
= 128, temperature = 0.0:

```
KILN_MODEL_PATH=/workspace/qwen3.5-4b KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  KILN_PREFIX_CACHE_ENABLED=0 \
  /workspace/kiln/target/release/kiln serve --quiet
```

```
ThreadPoolExecutor(max_workers=c) submitting c POSTs to /v1/chat/completions
with body {"model": ..., "messages": [{"role":"user","content": SHARED+random_salt}],
"max_tokens": 128, "temperature": 0.0, "stream": false}
```

| c | Trial 0 (tok/s) | Trial 1 | Trial 2 | **Median agg.** | c/c=1 ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.19 | 59.77 | 59.02 | **59.77** | 1.00× |
| 4 | 61.78 | 62.10 | 61.95 | **61.95** | 1.04× |
| 8 | 62.20 | 62.20 | 62.18 | **62.20** | 1.04× |

Variance is extremely tight (≤ 0.04 % across the c=8 trials), so this is a stable
measurement of the current scheduler behaviour, not noise.

**Comparison vs the `kiln-pr968-batching-regression` note:**

| Source | c=8 aggregate (tok/s) | Status |
|---|---:|---|
| This preflight, fresh build, median-of-3 | 62.20 | — |
| Note: claimed Phase A baseline | 62.45 | reproduces (within 0.4 %) |
| Note: claimed Phase A regression | 27.31 | **does NOT reproduce** |
| Note: claimed Phase A "flat 27.31 across c=4..32" | 27.31 (flat) | also does not reproduce; my c=4 = 61.95, c=8 = 62.20 |

The note's collapse claim does not reproduce on this build. The note's *baseline*
claim does. Whatever produced the 27.31 figure was not the merged Phase A code on a
healthy A6000 with the canonical bench protocol.

## The latent finding: c=8/c=1 = 1.04× is the real Phase 12 problem

Even though the regression-note collapse does not reproduce, **the c=8/c=1 batching
ratio of 1.04× confirms there is essentially no batching happening on Phase A HEAD.**
A correctly-batching paged-decode engine at c=8 with same-shape requests should
compound aggregate to several × c=1 (LM-head, MLP, and Marlin GEMM all benefit linearly
from batch dim under decode), bounded by the per-request KV-page-fetch latency for
paged attention.

Per-request elapsed times in the c=8 run are perfectly serial:

```
elapsed_s sorted: [2.10, 4.15, 6.18, 8.21, 10.24, 12.28, 14.39]
```

Each request completes ~2.1 s after the prior — exactly the c=1 single-stream
latency. This is the signature of a scheduler that is admitting requests one at a
time, not coalescing them into batched decode steps.

This dwarfs anything Phase B as scoped could deliver. **Restoring effective batching
from 1.04× to even 2× would be a +90 % aggregate win at c=8 — 18× the +5 % Phase B
floor — for substantially less kernel-engineering complexity than fusing the full-attn
pre-attn pipeline.**

## Recommendations

This preflight does NOT itself queue follow-up tasks (per the kiln skill's
"informational, not queueing" pattern for re-profile and audit artifacts). The next
planning cycle should pick from:

1. **Phase 12-B′ (recommended next)**: investigate why c=8/c=1 ≈ 1.04× on Phase A HEAD.
   This is the highest-leverage Phase 12 target. Likely suspects: scheduler admitting
   one request per decode step, paged-metadata shape-cache misses (the
   `kiln-pr968-batching-regression` note observed "hundreds of `CUDA graph capture
   skipped: paged metadata shape cache is full` warnings" — this preflight did not
   capture server logs but the cache size is documented as 8 entries, and 8 distinct
   per-request salts + the batched shapes could plausibly collide), or actor-level
   serialization that Phase A's `OnceLock<DecodeBuffers>` made worse via
   serialized first-touch on the request path.

2. **Phase 12-B (deferred)**: only revisit if and when (a) c=8 batching is restored to
   ≥ 2× c=1, AND (b) the work is scoped as a multi-day dedicated CUDA-kernel project
   with a brand-new fused Marlin-epilogue + RoPE + RMS kernel, not a 120-min
   dispatch-level refactor.

3. **Re-targeted Phase B alternative (if dispatch-level fusion is desired in a short
   budget)**: the MLP trio `:kiln/mlp/{gate, up, down}` is **5.1 % + 5.0 % + 4.6 % = 14.7 %**
   of decode and is structurally amenable to a SwiGLU-style fused Marlin gate+up+down
   pipeline. Same Amdahl class as Phase B but with one well-trodden fusion pattern
   (Liger / vLLM both ship variants), instead of three back-to-back primitives that
   need a custom kernel.

4. **Independent**: KV-cache FP8 default-on regression sweep — long-context capability
   win that does not interact with the batching investigation.

## Why this is a doc-only redirect, not "implement and discover failure"

Per the kiln skill's `kernel-vendor-precondition-check` and `phase6-kernel-vendor-preflight-pattern`
notes, Phase 6 / Phase 12 kernel-fusion work that fails its math-ceiling check at
preflight should ship as a doc-only redirect PR with $0 GPU spend on the
implementation phase. The precedent PRs are #131, #163, #164, and #170. This preflight
followed that pattern: ~50 min of pod time (≈ $0.40) on baseline measurement and math,
zero on speculative implementation.

## Appendix A — exact build, env, and harness

Build:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin && git reset --hard 718f8021c3e81c163369c5fed968e51972ca040c
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln
```

c=1 latency bench:

```bash
export KILN_W4A16=1 KILN_CUDA_GRAPHS=true
for i in 1 2 3; do
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
done
```

c=4/c=8 batched bench (Python harness uploaded to pod at `/tmp/c8_med3.py`):

```python
# Same SHARED prompt across all c requests, plus per-request random salt
# (to defeat radix prefix-output caching that returns identical responses
# for byte-identical requests).
# Submitted via concurrent.futures.ThreadPoolExecutor(max_workers=c).
# Aggregate throughput = sum(completion_tokens) / wall_clock_s.
```

## Appendix B — what the kiln-bench "Inference throughput" section actually measures

The c=4 / c=8 numbers reported in `kiln-bench`'s "Inference throughput" section
(the rows like `✓ 53.2 tok/s aggregate`) are **sequential per-shape runs at
synthetic concurrency 1, 4, 8, 16**, not concurrent batched throughput. They report
what one stream observes when the harness sweeps configurations in series. The actual
c=N batched throughput requires concurrent HTTP load against `kiln serve`, which is
what the Python harness above measures.

This is worth documenting because the regression note's "c=8 = 27.31" figure may
itself have come from `kiln-bench`'s sequential throughput section under unusual
conditions, not from concurrent HTTP batching, which would explain the divergence
between this preflight and the note.

## Appendix C — pod and lease

| Field | Value |
|---|---|
| Pool lease | `pod-357e8de3004fca40f7d884ec` |
| RunPod pod id | `6034jwt7rzsgt1` |
| GPU | NVIDIA RTX A6000 (49 140 MiB VRAM, 48 669 MiB free) |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| CUDA | 12.4 |
| Driver | as supplied by RunPod template |
| Cost rate | $0.49 / hr |
| Wall time | ≈ 50 min |
| Cost | ≈ $0.40 |

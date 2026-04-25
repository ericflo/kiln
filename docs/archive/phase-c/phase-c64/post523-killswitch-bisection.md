# Phase 7 kill-switch bisection: does any single fused-kernel default path account for the post-#166 decode regression?

Date: 2026-04-24
HEAD under test: `d22eb00` (`phase7: prefix-cache A/B rules out cache hooks as bench regression source (#523)`, docs-only on top of `821ccd3`)
Binary: `target/release/kiln-bench` rebuilt from `d22eb00` (identical Rust content vs `821ccd3` / `0fda0e6`)
GPU / pod: RunPod A6000 (`pod-66eb55349e1403350e6c342d`, runpod id `mfk88l8i8tab02`), `ghcr.io/ericflo/kiln-runpod:latest`
Image tooling: CUDA 12.4.1, `KILN_CUDA_ARCHS=86`, rustc stable

## Motivation

PR #522 observed a −7.9% decode tok/s gap (49.76 → 45.85) between the post-#166
closing baseline and a single-sample profile on post-#521 `main`. PR #523 ran an
`KILN_PREFIX_CACHE_ENABLED` A/B and falsified the parsimonious hypothesis
(prefix-cache hooks) both empirically and structurally — `kiln-bench --paged`
never enters the prefix-cache code path at all — and called for a kill-switch
bisection across the CUDA fused kernels as the next step.

This PR executes that bisection.

## TL;DR

**Null result. No single fused-kernel default path accounts for the gap.**

- None of the 5 kill-switch arms recovers decode tok/s to ≥49.0 (the post-#166
  baseline was 49.76). Two arms (`KILN_DISABLE_FUSED_GDN_GATES=1` and
  `KILN_DISABLE_FUSED_PAGED_DECODE=1`) appear +4.2–4.7% above this run's own
  baseline A, but that delta is well inside baseline A's own 8.2% intra-arm
  run-to-run spread.
- Baseline A median-of-last-2 in this sweep was **44.43 tok/s**, which is ~10%
  below the post-#166 baseline (49.76) and ~9% below the post-#522 Arm A median
  (48.84). A single pre-sweep sanity run (before the 18-run sweep started) hit
  **50.48 tok/s** — i.e. post-#166 territory — which is consistent with GPU
  frequency scaling / thermal drift and/or CUDA-graph-capture warmth differences
  across back-to-back bench invocations dominating the signal.
- The two intentional "sanity" arms C and D (disable whole-GDN-kernel and
  disable RMSNorm kernel respectively) fall back to candle paths and lose
  8–11% decode, confirming the kill switches are wired and take effect. Arm C
  also shows prefill time blowing up 7× (352 ms → 2357 ms median), which is
  the expected candle GDN-linear-attention fallback cost.

The measured "regression" between post-#166 and post-#521 is best explained as
run-to-run noise and GPU-state drift, not as a single kernel's default path.
Do **not** revert / gate any of the fused kernels based on this data.

## Methodology

Same shape as the post-#522 A/B (matches `crates/kiln-server/src/bench.rs`
production-path, `--paged` flag):

```bash
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
<ONE OF: KILN_DISABLE_FUSED_GDN_GATES=1 | KILN_DISABLE_GDN_KERNEL=1 | KILN_DISABLE_RMSNORM_KERNEL=1 | KILN_DISABLE_FUSED_CONV1D=1 | KILN_DISABLE_FUSED_PAGED_DECODE=1 | (none)> \
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

- 6 arms × 3 runs back-to-back, same bench binary, same model load path
- Discard run 1 per arm as cold-start TTFT artifact (per agent note
  `kiln-bench-prefill-warmup-required`)
- Report median-of-last-2 (mean of run 2 and run 3)
- Each arm reset all kill-switch env vars with `env -u` before applying its
  single extra kill switch, so arms never stack accidentally
- Bench sweep took ~9 min wall-clock on a pool-acquired A6000 (pre-warmed) with
  the binary already built; total task wall-clock including build + analysis
  ~20 min
- Pre-sweep sanity run (before the 18-run sweep began): decode 50.48 tok/s,
  mean ITL 19.81 ms, p99 ITL 24.53 ms. See "So where did the regression go?"
  below.

## Raw results

All 18 runs (decode tok/s, mean ITL ms, p99 ITL ms, prefill ms):

| Arm | Run | decode tok/s | mean ITL ms | p99 ITL ms | prefill ms |
|-----|-----|--------------|-------------|------------|------------|
| A (baseline)                          | 1 (cold, discard) | 45.61 | 21.93 | 26.75 | 353.0 |
| A (baseline)                          | 2 | 42.60 | 23.47 | 27.81 | 359.3 |
| A (baseline)                          | 3 | 46.26 | 21.62 | 27.88 | 373.2 |
| B (`DISABLE_FUSED_GDN_GATES=1`)       | 1 (cold, discard) | 45.06 | 22.19 | 31.08 | 353.4 |
| B (`DISABLE_FUSED_GDN_GATES=1`)       | 2 | 45.74 | 21.86 | 26.67 | 353.0 |
| B (`DISABLE_FUSED_GDN_GATES=1`)       | 3 | 47.30 | 21.14 | 24.25 | 346.5 |
| C (`DISABLE_GDN_KERNEL=1`)            | 1 (cold, discard) | 40.76 | 24.53 | 27.10 | 2348.2 |
| C (`DISABLE_GDN_KERNEL=1`)            | 2 | 41.92 | 23.86 | 28.75 | 2306.5 |
| C (`DISABLE_GDN_KERNEL=1`)            | 3 | 39.53 | 25.30 | 28.92 | 2415.7 |
| D (`DISABLE_RMSNORM_KERNEL=1`)        | 1 (cold, discard) | 39.91 | 25.05 | 35.04 | 359.2 |
| D (`DISABLE_RMSNORM_KERNEL=1`)        | 2 | 39.71 | 25.18 | 27.60 | 351.8 |
| D (`DISABLE_RMSNORM_KERNEL=1`)        | 3 | 39.21 | 25.51 | 36.55 | 366.9 |
| E (`DISABLE_FUSED_CONV1D=1`)          | 1 (cold, discard) | 46.46 | 21.52 | 23.81 | 366.9 |
| E (`DISABLE_FUSED_CONV1D=1`)          | 2 | 38.02 | 26.30 | 31.75 | 386.0 |
| E (`DISABLE_FUSED_CONV1D=1`)          | 3 | 46.46 | 21.52 | 26.07 | 370.4 |
| F (`DISABLE_FUSED_PAGED_DECODE=1`)    | 1 (cold, discard) | 47.97 | 20.84 | 31.65 | 350.9 |
| F (`DISABLE_FUSED_PAGED_DECODE=1`)    | 2 | 47.49 | 21.06 | 37.36 | 347.4 |
| F (`DISABLE_FUSED_PAGED_DECODE=1`)    | 3 | 45.07 | 22.19 | 27.96 | 351.3 |

Median-of-last-2 per arm:

| Arm | Kill switch | decode tok/s | Δ vs A | mean ITL ms | Δ vs A | p99 ITL ms | prefill ms |
|-----|-------------|--------------|--------|-------------|--------|------------|------------|
| A | none (baseline) | **44.43** | +0.0% | **22.54** | +0.0% | 27.84 | 366.2 |
| B | `DISABLE_FUSED_GDN_GATES=1` | **46.52** | **+4.7%** | 21.50 | −4.6% | 25.46 | 349.7 |
| C | `DISABLE_GDN_KERNEL=1` | 40.72 | −8.3% | 24.58 | +9.0% | 28.83 | 2361.1 |
| D | `DISABLE_RMSNORM_KERNEL=1` | 39.46 | −11.2% | 25.34 | +12.4% | 32.07 | 359.3 |
| E | `DISABLE_FUSED_CONV1D=1` | 42.24 | −4.9% | 23.91 | +6.1% | 28.91 | 378.2 |
| F | `DISABLE_FUSED_PAGED_DECODE=1` | **46.28** | **+4.2%** | 21.62 | −4.1% | 32.66 | 349.3 |

Reference baselines (for context):

| Source | decode tok/s | mean ITL ms | p99 ITL ms |
|--------|--------------|-------------|------------|
| post-#166 closing baseline (PROFILING.md) | 49.76 | 20.10 | 25.46 |
| post-#521 single-sample in PR #522 | 45.85 | 21.81 | 26.36 |
| post-#522 A/B Arm A median (PR #523) | 48.84 | 20.50 | 26.51 |
| post-#522 A/B Arm B median (PR #523) | 47.65 | 21.11 | 26.11 |
| **this sweep, pre-sweep sanity run** | **50.48** | **19.81** | **24.53** |
| **this sweep, baseline A median-of-last-2** | **44.43** | **22.54** | **27.84** |

Intra-arm variance (runs 2-3 min/max spread):

| Arm | run 2 | run 3 | spread | spread % of mean |
|-----|-------|-------|--------|------------------|
| A (baseline) | 42.60 | 46.26 | 3.66 | **8.2%** |
| B | 45.74 | 47.30 | 1.56 | 3.3% |
| C | 41.92 | 39.53 | 2.39 | 5.9% |
| D | 39.71 | 39.21 | 0.51 | 1.3% |
| E | 38.02 | 46.46 | **8.44** | **20.0%** |
| F | 47.49 | 45.07 | 2.41 | 5.2% |

Baseline A's 8.2% and Arm E's 20.0% spreads are both **larger** than the +4.2%
/ +4.7% signal seen in arms B and F. Taking Arm A's run 3 (46.26) against
Arm B's run 2 (45.74) or Arm F's run 3 (45.07), the "improvement" flips sign.
The signal is inside the noise band.

## So where did the regression go?

The single **pre-sweep sanity run** (one `kiln-bench` invocation issued right
after the `cargo build` completed and before the 18-run sweep began) produced
**50.48 tok/s / 19.81 ms mean ITL / 24.53 ms p99 ITL** — i.e. at or above the
post-#166 closing baseline. The subsequent 18 runs then cluster between ~38
and ~47 tok/s with large per-run spread.

Plausible explanations, in decreasing order of likelihood:

1. **GPU clock / thermal drift.** The A6000 was idle on the pod before the
   sanity run and hot after 18 consecutive bench invocations. Each run does
   ~24 s of model load (mostly DRAM + PCIe, not GPU-hot) followed by ~4 s of
   prefill + decode (GPU-hot). Cumulative heating over ~9 minutes of sweep
   likely dropped sustained clocks.
2. **CUDA graph capture reuse.** `kiln-bench` builds a fresh process per
   invocation and captures CUDA graphs fresh each time. Graph capture overhead
   is amortized over the 128 decode tokens within a run, but the first handful
   of decoded tokens of each run pay a higher cost (hence discarding run 1 is
   not enough to fully remove it).
3. **`mmap` page-in for the transposed weight cache** (`#461`, `#506`, `#508`).
   The first process after a fresh lease walks the cache cold; subsequent
   processes on the same host may page-in faster or slower depending on
   kernel page reclaim.

None of these are addressable by toggling a fused kernel default.

## Sanity check: do the kill switches work?

Yes. The kill switches for GDN kernel (C), RMSNorm kernel (D), and to a
lesser extent FUSED_CONV1D (E) all degrade decode as expected when flipped,
falling back to slower candle / non-fused paths. The GDN kernel fallback
in arm C additionally triples total prefill time (352 → 2357 ms median),
which is the known cost of routing the 24 Gated-DeltaNet layers through the
candle linear-attention reference path instead of the vendored CUDA kernel
from PR #80. This rules out the "kill switch has no effect" failure mode for
the whole sweep.

## Decision

Per the task brief:

> - If any arm recovers decode ≥ 49.0 tok/s **AND** beats baseline by ≥ 3% (above
>   noise floor documented in #523): that kernel's default path is the
>   regression source. Open a follow-up PR making the kill switch the new
>   default OR fix the fused path.
> - If **no** arm moves more than ±3% from baseline: regression is likely noise
>   / GPU frequency scaling / graph-capture reuse, not a single kernel. Ship
>   findings-only PR […]

No arm crosses the ≥ 49.0 tok/s bar. B and F cross ±3% in decode tok/s vs
this sweep's own baseline A (44.43), but that baseline is itself already
below the reported post-#166 baseline (49.76) and well below the pre-sweep
sanity run on the same binary in the same shell (50.48). Interpreting B
or F as "recovers the regression" requires treating the 44.43 reading as
authoritative, and the other runs on the very same binary (sanity run,
Arm A run 3) directly contradict that.

**Ship as findings-only. No Rust changes. No kill-switch becomes default.**

## Interesting-but-inconclusive signals

Two observations are worth flagging for a future higher-N re-run:

1. **Arm B p99 ITL exactly matches post-#166.** Arm B
   (`KILN_DISABLE_FUSED_GDN_GATES=1`) has median-of-last-2 p99 ITL of
   **25.46 ms**, which is *identical* to the post-#166 closing baseline
   p99 ITL (PROFILING.md). Baseline A's p99 in this sweep is 27.84 ms.
   This is possibly a real but small regression from the fused GDN gates
   default path (landed via `#486` "enable fused GDN qk norm by default" and
   follow-ups), hiding inside the tok/s noise. A median-of-5+ targeted A/B
   between arms A and B on the same pod would either confirm or reject this.

2. **Arm F has bimodal p99.** `DISABLE_FUSED_PAGED_DECODE=1` improved
   mean ITL to 21.62 ms (vs A 22.54) but its p99 worsened to 32.66 ms
   (vs A 27.84). This suggests the fused paged decode path has tighter
   per-token latency variance than the fallback, even if the fallback is
   competitive at the mean. Not a regression source, but a useful data point
   for the decoder's variance budget.

Both are observations, not regressions. They do **not** justify making
either kill switch the default.

## Recommended next step

Stop chasing a single-kernel culprit for the post-#166 decode gap. The two
alternative axes from the skill doc's "Current Phase 6 frontier" section
remain viable and are not confounded by run-to-run GPU noise in the same
way:

- **KV cache FP8** (`KILN_KV_CACHE_FP8`) — halves KV memory, opens more batch
  concurrency. Direct decode speedup from lower memory bandwidth is small
  but robust.
- **Marlin packing latency cleanup** (~58 s at model load across 96 MLP
  projections) — cache packed weights to disk, or parallelize the pack loop.
  Cleanup task, not a decode-hot-path change, but removes a known wart.
- **Marlin BF16 weight residency cleanup** (+1.3 GB unused VRAM when
  `KILN_W4A16=1`) — drop BF16 tensors from VRAM after Marlin packing.

If someone still wants to close the post-#166 gap specifically, the next
step is:

1. Instrument a pinned GPU clock / thermal state between runs (clock capping
   + cooldown pause), so baseline A is reproducible within ~2% across 10+
   runs.
2. Run median-of-5 (or higher) for each arm — 3 runs cannot distinguish a
   3–5% effect from 8–20% noise.
3. If arm B still shows a structurally lower p99 at median-of-5, a targeted
   fused-GDN-gates A/B with numerical parity against the candle fallback
   becomes the next focused PR.

None of this should block Phase 7 work on more promising axes.

## What did NOT change

This PR is findings-only:

- No code changes to `crates/kiln-model/src/forward.rs`
- No code changes to any CUDA kernel crate
- No changes to any kill-switch default (`KILN_DISABLE_*` env vars remain
  the same — opt-in for parity testing, never-on in production)

## Reproducing

```bash
# 1. Acquire a pool A6000
ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'

# 2. On pod:
kiln-setup --clone                                     # fetch kiln + sccache+B2
cd /workspace/kiln && git checkout d22eb00              # or current main
source /root/.kiln-build-env
cargo build --release --features cuda --bin kiln-bench  # ~90 s warm

# 3. Run 6 arms × 3 runs (see /tmp/run-killswitch.sh in PR sources
#    for the full driver; each arm resets all kill-switch env vars before
#    applying its one extra switch, so arms never stack)
for SWITCH in "" \
              "KILN_DISABLE_FUSED_GDN_GATES=1" \
              "KILN_DISABLE_GDN_KERNEL=1" \
              "KILN_DISABLE_RMSNORM_KERNEL=1" \
              "KILN_DISABLE_FUSED_CONV1D=1" \
              "KILN_DISABLE_FUSED_PAGED_DECODE=1"; do
  for RUN in 1 2 3; do
    env -u KILN_DISABLE_FUSED_GDN_GATES -u KILN_DISABLE_GDN_KERNEL \
        -u KILN_DISABLE_RMSNORM_KERNEL -u KILN_DISABLE_FUSED_CONV1D \
        -u KILN_DISABLE_FUSED_PAGED_DECODE \
      bash -c "KILN_W4A16=1 KILN_CUDA_GRAPHS=true $SWITCH \
        ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
          --paged --prompt-tokens 512 --max-output-tokens 128 \
          --skip-training --prompt-subset humaneval --chat-template \
          --latency-only --temperature 0.0 --seed 1"
  done
done

# 4. Release the lease
ce kiln-pod-release --lease <leaseID>
```

Wall-clock cost on the pool A6000: ~9 min for the sweep, ~90 s for the
cached rebuild, ~25 s for each bench's model load (included in run time).
Total pod cost at the pool's $0.49/hr on-demand rate: well under $1.

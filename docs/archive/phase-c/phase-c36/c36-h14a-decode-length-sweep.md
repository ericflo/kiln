# Phase C36 — H14a decode-length α stability sweep

**Date:** 2026-04-22
**Hardware:** RunPod A6000 (49 140 MB), pod `sq3exutrqy2cuw`, image `ghcr.io/ericflo/kiln-runpod:latest`
**Build:** branch `ce/mtp-c36-h14a-decode-length` rebased on `main` @ `07d2477` (post C35 merge, no kernel changes)
**Wall-clock:** 4 337 s (~72 min) for the 12-run matrix; pod cost ≈ $0.49/hr × ~2.5 h ≈ $1.20 incl. build

## 1. Question

C35 Cell D (chat-template × FP32 argmax × prompt_tokens=512 × decode=128) measured median α = **0.6076**, leaving a **gap of 0.112** to the paper-floor target α = 0.72. C35 §4.2 listed three H14 candidates; this run interrogates the cheapest:

> **H14a** — α at 128-token decode is a transient under-measurement caused by a too-short observation window. The MTP head needs a longer decode horizon to reach its stable-regime acceptance, and median α should rise toward 0.72 as decode length grows.

## 2. Method

Single configuration cell from C35 (Cell D), held fixed:

```
KILN_W4A16=1 KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp KILN_MTP_ARGMAX_FP32=1 KILN_CUDA_GRAPHS=true
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template \
  --prompt-tokens 512 --max-output-tokens {128|256|512|1024} \
  --skip-training --seed {0|1|2}
```

Matrix: 4 decode lengths × 3 seeds = **12 bench invocations** against the same warm binary. No code changes; the harness already accepts `--max-output-tokens` and `--seed`.

## 3. Per-seed results

```
    L  s   alpha   tok/s   p50ms   p99ms
  128  0  0.5875   33.63   19.34   61.39
  128  1  0.7067   40.22   18.03   63.32
  128  2  0.7162   41.48   17.89   59.36
  256  0  0.7114   37.05   19.71   67.51
  256  1  0.6887   38.29   19.22   67.58
  256  2  0.6452   39.18   17.65   58.41
  512  0  0.6570   36.28   19.54   65.11
  512  1  0.6484   36.54   18.89   61.82
  512  2  0.6570   41.91   16.72   53.23
 1024  0  0.6814   37.64   19.10   64.60
 1024  1  0.6814   39.56   17.99   57.85
 1024  2  0.6982   39.07   18.77   62.96
```

## 4. Per-decode-length medians

| L    | median α | median tok/s | median p50 ITL (ms) | median p99 ITL (ms) | Δα vs C35 anchor (0.6076) | gap to paper (0.72) |
|------|----------|--------------|----------------------|----------------------|----------------------------|----------------------|
| 128  | 0.7067   | 40.22        | 18.03                | 61.39                | **+0.0991**                | +0.0133              |
| 256  | 0.6887   | 38.29        | 19.22                | 67.51                | +0.0811                    | +0.0313              |
| 512  | 0.6570   | 36.54        | 18.89                | 61.82                | +0.0494                    | +0.0630              |
| 1024 | 0.6814   | 39.07        | 18.77                | 62.96                | +0.0738                    | +0.0386              |

C35 Cell D anchor for reference: α = 0.6076, tok/s = 37.48 (chat × FP32 × prompt=512 × decode=128, N=3 seeds).

## 5. Verdict

**Verdict on H14a: INCONCLUSIVE — H14a is *not* the dominant driver.**

Decision rules (verbatim from the C36 task spec):

- **Confirm** if any decode-length median α ≥ 0.72 OR shows monotonic upward trend across {128, 256, 512, 1024}.
- **Refute** if median α stays within ± noise (~0.06) of 0.608 at all lengths.
- **Inconclusive** otherwise.

Evidence:

| Test                                          | Result |
|-----------------------------------------------|--------|
| Any median α ≥ 0.72                           | No (max = 0.7067 at L=128, gap +0.0133) |
| Monotonic upward across L                     | No — sequence 0.7067 → 0.6887 → 0.6570 → 0.6814 dips at L=256/512 then partially recovers |
| All medians within ±0.06 of 0.608             | No — L=128 median sits at 0.7067 (Δ +0.0991), well outside the refute window |

Neither confirm nor refute fires; we land in the inconclusive bucket. The more important finding is **what the data does show**:

1. **The C35 Cell D anchor of 0.6076 looks like a low-variance fluke at seed 0.** This run's L=128 seed-0 reproduces it at α = 0.5875 (well within noise of 0.6076), but seeds 1 and 2 at the *identical* configuration land at 0.7067 and 0.7162 — within 0.013 of the paper floor. The L=128 median in C36 (0.7067) sits **0.0991 above** the C35 median, despite zero config drift.
2. **Seed-to-seed variance dominates the decode-length signal at every length.** Per-seed α ranges:
   - L=128: [0.5875, 0.7162] — spread Δ = 0.1287
   - L=256: [0.6452, 0.7114] — spread Δ = 0.0662
   - L=512: [0.6484, 0.6570] — spread Δ = 0.0086 (tightest)
   - L=1024: [0.6814, 0.6982] — spread Δ = 0.0168
   The longer-decode runs have tighter inter-seed spread (more tokens averaged → narrower CI), but the *centre* of the distribution is roughly stable around 0.66–0.71. Decode length does not move the long-run mean toward 0.72; it merely tightens the estimator.
3. **N=3 is structurally insufficient at L=128 / L=256.** The C35 → C36 swing in the L=128 median (0.6076 → 0.7067, same config) shows that N=3 medians have ~±0.10 variance for short decodes. C35 conclusions framed off N=3 short-decode medians are therefore over-confident.
4. **Throughput is roughly flat across L** (median tok/s ∈ [36.5, 40.2]), which is the expected speculative-decoding behaviour: longer windows do not change per-token economics once steady-state.

Mechanistically: the data is consistent with the MTP head reaching its acceptance regime within the first ~64 decode tokens (well below the L=128 floor). Once there, additional decode tokens just shrink the variance of the α estimator — they do not raise its mean. The "decode length is too short" hypothesis is not supported.

## 6. Handoff to C37

The blocker is no longer "is the decode horizon long enough?" — it is **"why does seed-to-seed variance swamp the configuration signal at this measurement budget?"**. Two C37 tracks, with the second prioritised because it directly tests one of the remaining §4.2 candidates from C35:

### C37 (recommended): H14b — prompt-content distribution

Replace the current synthetic prompt with canonical task distributions and re-bench at fixed L=512 (smallest variance band in C36) with **N ≥ 5 seeds**:

- **GSM8K** (math word problems) — short answers, structured arithmetic, MTP draft should align tightly
- **HumanEval** (Python code completion) — token co-occurrence is highly predictable for the head
- **C4 prose continuation** — high-entropy baseline

Hypothesis: prompt-content distribution shifts the α distribution by ≥ 0.05 (similar magnitude to the seed-to-seed swing), and at least one of {GSM8K, HumanEval} reproducibly clears α ≥ 0.72 with the same Cell D config. If true, the paper-floor gap is a prompt-distribution artifact, not a kernel/sampler defect, and we close out the gap with a corpus selection note rather than further kernel work.

### C37b (parallel, cheap): re-anchor C35 Cell D with N=5

Re-run **only** the L=128 Cell D point with seeds 0..4 (5 runs, ≈10 min) to pin down whether the C35 median of 0.6076 was a one-tail draw or whether C36's 0.7067 is. This is a direct sanity check on whether C35 §4.1 needs a numeric correction. Cost is trivial; pairs naturally with C37 to share a pod lease.

If H14b also lands inconclusive at N=5, fall back to the third §4.2 candidate (residual MTP head weight quality) — but only after ruling out variance-driven measurement noise.

## 7. Reproducibility

Raw JSONs are saved off-pod for replay; rerun with the exact harness invocation in §2 against `main` @ `07d2477`. Sentinel-driven runner: `run_c36_bench.sh` (committed alongside this doc for traceability).

# Phase C39 — HumanEval-only N=20 domain-focused α re-bench

## TL;DR

HumanEval MTP α is **structurally below the Qwen3.5 paper floor of 0.72**. N=20
median 0.6933, bootstrap 95% CI [0.6602, 0.7162] — the entire CI sits below
the floor, resolving the ambiguity left by C38's straddling N=30 CI.

- **Verdict:** HumanEval fails the paper floor. The C38 N=30 combined
  CI [0.6996, 0.7760] straddled 0.72 ambiguously; restricting to code prompts
  tightens the band and lands it below the floor.
- **Decision input for C40:** the code-distribution gap is real and domain-
  specific, not variance. HumanEval α ≈ 0.69 vs GSM8K 0.789 is a
  prompt-distribution effect, not a seed-sampling artifact.

## Anchor

Cell D, identical to C37 / C38:

| Knob | Value |
| --- | --- |
| Quantization | `KILN_W4A16=1` |
| Argmax dtype | `KILN_MTP_ARGMAX_FP32=1` (FP32) |
| Prompt framing | `--chat-template` (Qwen ChatML) |
| KV path | `--paged` |
| Prompt tokens | 512 |
| Decode tokens | 128 |
| Sampling | `temperature=0` (greedy) |
| CUDA graphs | `KILN_CUDA_GRAPHS=true` |
| Spec method | `KILN_SPEC_METHOD=mtp` |

## Methodology change vs C38

C38 used a 30-prompt domain-balanced pool (indices 0-9 GSM8K, 10-19
HumanEval, 20-29 C4) with `seed % 30` indexing, so N=30 gave 10 seeds
per domain. C39 adds a `--prompt-subset` CLI flag that filters the pool
by domain:

- `--prompt-subset humaneval` indexes into `PROMPT_POOL[10..20]` via
  `seed % 10`, so seeds 0..9 cover all 10 code prompts once and
  seeds 10..19 cover them again. N=20 therefore visits every HumanEval
  prompt exactly twice, keeping per-seed variance a pure RNG effect
  rather than a prompt re-hit artifact.
- `--prompt-subset` defaults to `all`, so all prior C38 / C37 runs are
  reproducible byte-for-byte.

Only the MTP bench arm consumes the subset; off / skip-layer still pull
from the full pool.

## Per-seed results (N=20, HumanEval subset)

| seed | α      | decode_tps | mean_itl_ms |
| ---- | ------ | ---------- | ----------- |
|    0 | 0.7162 |      41.63 |      24.019 |
|    1 | 0.7162 |      45.74 |      21.861 |
|    2 | 0.6711 |      43.49 |      22.996 |
|    3 | 0.6494 |      39.28 |      25.458 |
|    4 | 0.7397 |      41.60 |      24.039 |
|    5 | 0.7778 |      43.80 |      22.829 |
|    6 | 0.6933 |      41.53 |      24.081 |
|    7 | 0.6933 |      40.57 |      24.650 |
|    8 | 0.6933 |      41.80 |      23.924 |
|    9 | 0.5679 |      35.70 |      28.010 |
|   10 | 0.7534 |      43.71 |      22.879 |
|   11 | 0.7297 |      42.55 |      23.500 |
|   12 | 0.7162 |      42.39 |      23.589 |
|   13 | 0.6494 |      39.53 |      25.300 |
|   14 | 0.6282 |      36.72 |      27.234 |
|   15 | 0.7067 |      45.40 |      22.029 |
|   16 | 0.6203 |      37.15 |      26.915 |
|   17 | 0.6711 |      39.45 |      25.348 |
|   18 | 0.6933 |      39.24 |      25.485 |
|   19 | 0.6494 |      38.75 |      25.807 |

## Summary statistics

| Metric | Value |
| --- | --- |
| N | 20 |
| median α | **0.6933** |
| mean α | 0.6868 |
| stdev α | 0.0499 |
| min α | 0.5679 (seed 9) |
| max α | 0.7778 (seed 5) |
| 95% CI (bootstrap 10k resamples, rng=12345) | **[0.6602, 0.7162]** |
| median ITL | ≈ 24.0 ms |
| median decode tps | ≈ 41.6 tok/s |

## Floor check (Qwen3.5 paper, α ≥ 0.72)

| Bound | Value | vs 0.72 | Verdict |
| --- | --- | --- | --- |
| CI lo | 0.6602 | < 0.72 | fail |
| median | 0.6933 | < 0.72 | fail (by 0.0267) |
| CI hi | 0.7162 | < 0.72 | fail |

The **entire** 95% CI sits below the paper floor — a cleaner fail than
C38's [0.6996, 0.7760] which straddled it.

## Cross-phase α comparison (Cell D anchor)

| Phase | N | pool | median α | 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| C37 | 10 | 8-prose | 0.6779 | [0.63, 0.72] | below, but N too small |
| C38 (full) | 30 | 30-domain | 0.7297 | [0.6996, 0.7760] | straddled 0.72 |
| C38 (GSM8K slice) | 10 | 30-domain slice | 0.789 | n/a (too narrow) | above floor |
| C38 (HumanEval slice) | 10 | 30-domain slice | 0.6888 | n/a (too narrow) | below floor |
| C38 (C4 slice) | 10 | 30-domain slice | 0.716 | n/a (too narrow) | borderline |
| **C39 (HumanEval-only)** | **20** | **10-HumanEval** | **0.6933** | **[0.6602, 0.7162]** | **below floor, CI clean** |

The C39 median (0.6933) is statistically indistinguishable from the C38
HumanEval-slice median (0.6888) — doubling the code-prompt sample size
just tightens the band without shifting the center. The gap to the
paper floor is ~0.027 in the median, and ~0.06 at the CI upper bound.

## Interpretation

1. **The paper-floor shortfall is a code-distribution effect, not seed
   variance.** Doubling the sample size on code prompts did not raise
   the α estimate; it only narrowed the CI. This is the expected
   behavior when the underlying population mean is genuinely below the
   target.
2. **Domain gap ordering is stable:** GSM8K > C4 > HumanEval, with
   roughly 0.07–0.09 α spread between the best and worst domains. MTP
   acceptance is most reliable on natural-prose math problems, moderate
   on general English, and structurally weaker on Python code.
3. **Nothing about the anchor is suspect.** The same W4A16 + FP32-argmax
   + chat-template + paged decode stack that hits 0.789 on GSM8K drops
   to 0.69 on HumanEval. The degradation is in the token distribution
   the MTP head is being asked to predict, not in the numerical path
   that runs it.

## C40 recommendation

The original C40 queue from this phase is now complete enough to close
the "easy gap-closer" branch and keep only the documentation update.

### What the completed C40 work says

- **C40a falsified the prompt-framing hypothesis.** Removing
  `--chat-template` collapsed HumanEval α from the C39 anchor rather
  than recovering it, so chat-template is load-bearing for code.
- **C40b falsified the low-temperature hypothesis.**
  `temperature=0.1` did not rescue HumanEval α, so there is no cheap
  sampler knob that closes the gap.
- **C40e found a real paired seed-0 BF16 win.** On the same binary/pod
  and flags, BF16 beat W4A16 materially for that one HumanEval seed.
- **C40f showed that C40e does not generalize distributionally.** The
  BF16 N=20 HumanEval distribution on current `main` landed at median
  `0.6888`, mean `0.6939`, bootstrap 95% CI `[0.6517, 0.7232]`, with
  only `6/20` seeds at or above `0.72`. That does not materially beat
  the C39 W4A16 HumanEval distribution.

### Updated recommendation

- Keep the documentation conclusion from C39/C40: the paper's `0.72`
  reference is an aggregate, prose-heavy benchmark anchor, while the
  current-main HumanEval code distribution sits lower.
- Do **not** present C40a/C40b/C40c as still-pending next steps. Those
  branches are now either falsified or superseded by C40d/C40e/C40f.
- Do **not** treat BF16 as the resolved HumanEval load-mode winner.
  C40e remains an important paired single-seed result, but C40f is the
  correct distributional readout for the BF16 question on current
  `main`.
- Keep the code-domain issue open as a modeling / implementation
  investigation, not as an unresolved queue of cheap benchmark toggles.

## Artifacts

- Branch: `ce/mtp-c39-humaneval-domain`
- CLI addition: `crates/kiln-server/src/bench.rs` — `PromptSubset`
  enum + `--prompt-subset` flag + JSON `prompt_subset` tag in
  `LatencyResult`.
- Raw per-seed JSON: pod path `/workspace/c39-results/seed-{0..19}.json`
  (pod `sq3exutrqy2cuw`, lease `pod-da455a3428335b8b2f650773`).
- Analysis script: `analyze_c39.py` (session-local, not committed).

## Reproduction

```bash
cd /workspace/kiln
for seed in $(seq 0 19); do
  KILN_W4A16=1 \
  KILN_SPEC_ENABLED=1 \
  KILN_SPEC_METHOD=mtp \
  KILN_MTP_ARGMAX_FP32=1 \
  KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --chat-template --skip-training \
    --prompt-tokens 512 --max-output-tokens 128 \
    --prompt-subset humaneval \
    --seed $seed
done
```

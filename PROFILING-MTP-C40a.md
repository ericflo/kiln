# Phase C40a — HumanEval + chat-template OFF N=20 α re-bench

## TL;DR

Removing `--chat-template` **catastrophically lowers** HumanEval MTP α.
N=20 median **0.2814**, bootstrap 95% CI **[0.2562, 0.3026]** — far
**below** the 0.72 paper floor and sharply below the C39 chat-template-ON
baseline median of 0.6933.

- **Hypothesis pre-test (from C39/C40 synthesis):** chat-template wraps
  Python in a natural-language user turn, potentially shifting the MTP
  draft off its training distribution. If α rose to ≥ 0.72 with
  chat-template off, the paper floor would be reachable via prompt
  framing alone.
- **Result:** the hypothesis is **inverted**. Chat-template framing is
  *critical* to MTP draft acceptance on code. Stripping it drops α by
  **~0.412** absolute (median). **Every seed is below C39.**
- **Decision input for C40b/c:** chat-template is not the bottleneck.
  The HumanEval α gap at C35 Cell D is not a framing artifact; it is
  a combination of a code-distribution weakness in the MTP head and,
  possibly, other anchor-level effects (sampling temperature, W4A16).
  C40b (temperature=0.1) and C40c (BF16) remain open as distinct
  interventions, but C40a rules out the cheapest hypothesis.

## Anchor

Cell D, identical to C35/C37/C38/C39 **except** chat-template off:

| Knob | Value | vs C39 |
| --- | --- | --- |
| Quantization | `KILN_W4A16=1` | same |
| Argmax dtype | `KILN_MTP_ARGMAX_FP32=1` (FP32) | same |
| Prompt framing | **`--chat-template` OFF (raw PROMPT_POOL strings)** | **flipped** |
| KV path | `--paged` | same |
| Prompt tokens | 512 | same |
| Decode tokens | 128 | same |
| Sampling | `temperature=0` (greedy) | same |
| CUDA graphs | `KILN_CUDA_GRAPHS=true` | same |
| Spec method | `KILN_SPEC_METHOD=mtp` | same |
| Prompt subset | `--prompt-subset humaneval` | same |

### Why chat-template is the only knob

C40 synthesis (PR #372) identified chat-template as the single highest-
leverage empirical intervention: it requires no code change (a CLI
flag flip), it is orthogonal to numerical knobs (W4A16, FP32 argmax),
and the pre-test hypothesis (natural-language framing shifting code
tokens off the MTP training distribution) pointed at a *rise* in α.

N=20 matches C39 exactly for direct seed-vs-seed comparison.

## GPU note

C39 ran on NVIDIA A6000 (sm_86). C40a ran on NVIDIA L40S (sm_89) because
A6000 supply was constrained on RunPod at task time. MTP α is computed
on CPU from argmax comparison of two token IDs per decode step with
`KILN_MTP_ARGMAX_FP32=1`; GPU arch cannot materially affect the
measurement at greedy temperature=0 given identical weights. Throughput
numbers (decode tps, ITL) are not directly comparable to C39 across the
arch boundary; α is.

## Per-seed results (N=20, HumanEval subset, chat-template OFF)

| seed | α      | decode_tps | mean_itl_ms | vs C39 α |
| ---- | ------ | ---------- | ----------- | -------- |
|    0 | 0.3368 |      27.06 |      36.954 | -0.3794 |
|    1 | 0.1869 |      21.77 |      45.930 | -0.5293 |
|    2 | 0.2451 |      23.38 |      42.765 | -0.4260 |
|    3 | 0.2800 |      23.72 |      42.152 | -0.3694 |
|    4 | 0.3229 |      25.40 |      39.376 | -0.4168 |
|    5 | 0.2959 |      23.73 |      42.139 | -0.4819 |
|    6 | 0.2959 |      23.43 |      42.683 | -0.3974 |
|    7 | 0.2700 |      22.77 |      43.911 | -0.4233 |
|    8 | 0.3093 |      27.99 |      35.733 | -0.3840 |
|    9 | 0.1852 |      23.85 |      41.932 | -0.3827 |
|   10 | 0.3368 |      25.71 |      38.892 | -0.4166 |
|   11 | 0.1651 |      21.08 |      47.437 | -0.5646 |
|   12 | 0.2451 |      23.01 |      43.455 | -0.4711 |
|   13 | 0.2673 |      23.46 |      42.623 | -0.3821 |
|   14 | 0.3229 |      25.27 |      39.577 | -0.3053 |
|   15 | 0.2959 |      24.74 |      40.427 | -0.4108 |
|   16 | 0.2828 |      24.13 |      41.436 | -0.3375 |
|   17 | 0.2700 |      23.52 |      42.509 | -0.4011 |
|   18 | 0.3093 |      25.04 |      39.932 | -0.3840 |
|   19 | 0.1852 |      19.98 |      50.050 | -0.4642 |

Every single seed is below the matching C39 seed by a large margin.
Per-seed deltas range from −0.31 (seed 14) to −0.56 (seed 11).

## Summary statistics

| Metric | C40a (chat-off) | C39 (chat-on) | Δ |
| --- | --- | --- | --- |
| N | 20 | 20 | — |
| median α | **0.2814** | 0.6933 | **-0.4119** |
| mean α | 0.2704 | 0.6868 | -0.4164 |
| stdev α | 0.0530 | 0.0499 | +0.0031 |
| min α | 0.1651 (seed 11) | 0.5679 (seed 9) | -0.4028 |
| max α | 0.3368 (seeds 0,10) | 0.7778 (seed 5) | -0.4410 |
| 95% CI | **[0.2562, 0.3026]** | [0.6602, 0.7162] | — |
| median ITL | ≈ 42.1 ms | ≈ 24.0 ms | +18.1 ms |
| median decode tps | ≈ 23.7 tok/s | ≈ 41.6 tok/s | -17.9 tok/s |

ITL / tps deltas reflect both the lower α (more verifier-only steps)
**and** the L40S vs A6000 arch change; do not attribute them solely
to the chat-template flip.

## Floor check (Qwen3.5 paper, α ≥ 0.72)

| Bound | Value | vs 0.72 | Verdict |
| --- | --- | --- | --- |
| CI lo | 0.2562 | < 0.72 | fail (by 0.464) |
| median | 0.2814 | < 0.72 | fail (by 0.439) |
| CI hi | 0.3026 | < 0.72 | fail (by 0.417) |

The entire 95% CI sits ~0.42–0.46 below the paper floor. The CI
upper bound is still further from 0.72 than the C39 lower bound was —
chat-template OFF makes the gap *worse*, not better.

## Cross-phase α comparison (Cell D anchor variants)

| Phase | N | pool / subset | chat-template | median α | 95% CI | verdict vs 0.72 |
| --- | --- | --- | --- | --- | --- | --- |
| C37 | 10 | 8-prose | on | 0.6779 | [0.63, 0.72] | below (N small) |
| C38 (full) | 30 | 30-domain | on | 0.7297 | [0.6996, 0.7760] | straddled |
| C38 (GSM8K slice) | 10 | 30-domain slice | on | 0.789 | n/a | above |
| C38 (HumanEval slice) | 10 | 30-domain slice | on | 0.6888 | n/a | below |
| C38 (C4 slice) | 10 | 30-domain slice | on | 0.716 | n/a | borderline |
| C39 | 20 | 10-HumanEval | **on** | 0.6933 | [0.6602, 0.7162] | below, CI clean |
| **C40a** | **20** | **10-HumanEval** | **off** | **0.2814** | **[0.2562, 0.3026]** | **far below, ~0.43 gap** |

## Interpretation

1. **Chat-template is load-bearing for HumanEval MTP α.** Stripping the
   Qwen ChatML wrapper around the Python prompts drops median α by
   ~0.412 absolute — a direction and magnitude that completely
   contradict the C40 pre-test hypothesis. The natural-language framing
   is *helping*, not hurting, MTP draft acceptance on code tokens.
2. **C40 hypothesis inversion.** The C40 synthesis suggested that
   chat-template might be shifting code tokens off the MTP training
   distribution. Empirically, the opposite holds: the bare code prompt,
   without any framing, is what looks foreign to the MTP draft head.
   One plausible cause: the MTP draft head was distilled / trained on
   chat-formatted sequences at scale, so the ChatML preamble tokens
   drive hidden states that the head predicts well; stripping them
   exposes a much narrower code-only distribution that the head
   underperforms on.
3. **Domain ordering likely stable.** C40a does not re-measure GSM8K
   or C4 with chat-template off, but the same mechanism that drops
   HumanEval α by ~0.41 would be expected to drop GSM8K and C4 as
   well. The conclusion that chat-template is the MTP's default
   high-α regime is consistent with the C38 cross-domain ordering
   (GSM8K > C4 > HumanEval, all at chat-template ON).
4. **Anchor is still not suspect, but the anchor's default is now
   justified.** C39 showed that the same Cell D anchor that scores 0.789
   on GSM8K drops to 0.69 on HumanEval. C40a adds: removing chat-template
   is the wrong way to close that gap.

## Per-domain floor implications (update to C40 proposal)

C40 (PR #372) proposed per-domain floors derived from C38/C39 data:
GSM8K ≥ 0.78, C4 ≥ 0.70, HumanEval ≥ 0.65. C40a does not change those
floors — it only strengthens the claim that those floors hold at the
chat-template-ON default, and that turning chat-template off would
violate the HumanEval floor catastrophically (observed 0.2814 vs the
0.65 proposed floor — a 0.37 shortfall). Net effect: if kiln codifies
per-domain floors, the default-anchor **must** include `--chat-template`
(or the tokenizer's chat_template application) on every MTP code path;
that framing is not a nice-to-have but a load-bearing part of reaching
paper-floor compliance even under the lower domain-specific bar.

## What C40a rules out

1. Chat-template stripping as a path to α ≥ 0.72 on HumanEval. This was
   the highest-leverage / cheapest / single-knob hypothesis coming out
   of C40. **Dead.**
2. Framing-drift as the dominant cause of the HumanEval code gap. The
   gap is *not* that chat-template confuses the MTP head on code; the
   gap is that, even *with* chat-template smoothing things, pure-code
   content still underperforms natural language prose (GSM8K /
   C4 prose).

## Remaining open questions (C40b+)

C40a pushes the highest-leverage empirical knob off the table. The
remaining hypotheses in the C40 queue are all unchanged:

1. **C40b — greedy vs temperature=0.1 on HumanEval.** Tests whether
   greedy decoding is uniquely harmful for low-entropy code tokens;
   MTP heads are trained over sampled distributions.
2. **C40c — W4A16 vs BF16 on HumanEval.** Tests whether Marlin's W4
   quantization is specifically hurting code α (vs. natural language,
   where C37/C38 GSM8K α was ≥ 0.78 under the same W4A16 path).
3. **Per-domain floors (C40 documentation proposal).** Still stands;
   C40a confirms those floors are dependent on chat-template ON, so
   any MTP bench or gate should default chat-template on.

A new question surfaces from C40a:

4. **Does chat-template ON also raise GSM8K α above its current
   0.789 ceiling, or is GSM8K already saturated?** If the ChatML
   wrapper helps code because it brings the prompt closer to the
   MTP training distribution, GSM8K prose should be near-saturated
   (no further α headroom), while code has residual headroom. A tiny
   10-seed GSM8K-vs-HumanEval A/B at chat-template-on (already measured)
   vs. an alternate framing (e.g. ChatML without system turn) could
   probe whether "chat-template" specifically or just "any preamble"
   is the key.

## Artifacts

- Branch: `ce/mtp-c40a-humaneval-no-chat-template`
- Raw per-seed JSON: `docs/phase-c40a/seed-{0..19}.json`
- Analysis script: `docs/phase-c40a/analyze_c40a.py` (ships with PR for
  reproducibility)

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
    --paged --skip-training \
    --prompt-tokens 512 --max-output-tokens 128 \
    --prompt-subset humaneval \
    --seed $seed
done
```

Note the **absence** of `--chat-template`. All other flags match C39 to
enable direct chat-template ON vs OFF comparison.

## Cost & environment

- GPU: NVIDIA L40S ($0.86/hr; A6000 supply-constrained at task time)
- Pod: `hlxekl8ah1gjqa` on RunPod, on-demand
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- Build: cached via sccache + B2 (incremental ~2 min)
- Bench run: ~38 min wall for N=20 × (model load 25s + bench 60s) per seed
- Total pod cost: ≈ $0.75–1.00

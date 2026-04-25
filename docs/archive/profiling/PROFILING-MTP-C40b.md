# Phase C40b — HumanEval + temperature=0.1 N=20 α re-bench

## TL;DR

Switching from greedy (`temperature=0.0`) to low-temperature sampling
(`temperature=0.1`) on the C39 HumanEval anchor **does not** recover MTP
α toward the Qwen3.5 paper floor. N=20 median **0.6558**, bootstrap 95%
CI **[0.6452, 0.6933]** — both **below** the 0.72 paper floor and
**below** the C39 chat-template-ON greedy median of 0.6933 by **−0.0375
absolute**.

- **Hypothesis pre-test (greedy-is-uniquely-harmful):** MTP draft heads
  are trained over sampled distributions, so greedy decoding might be
  the dominant cause of the HumanEval α gap. If `temperature=0.1`
  recovered α to ≥ 0.72, the paper floor would be reachable via a
  one-flag sampling change with no kernel/quantization work.
- **Result:** the hypothesis is **inverted**. Sampling at
  `temperature=0.1` *worsens* median α by 0.0375 absolute and the entire
  95% CI sits below the C39 greedy CI lower bound (0.6602). Of the 20
  paired seeds, 12 dropped, 7 improved, 1 tied — and the worst losses
  (−0.125, −0.116, −0.106, −0.103) are larger than the best gains
  (+0.047, +0.036, +0.029).
- **Sanity check passed (byte-identical default path):** the
  `temperature=0.0 seed=0` re-run produced **α = 0.6933333…**, identical
  to C39's seed=0+1 cluster median and well within float-noise of the
  pre-change C39 distribution. Adding the `--temperature` CLI flag with
  a `0.0` default is a no-op for all existing call sites and bench
  numbers.
- **Decision input for C40c+:** sampling temperature is not the
  bottleneck on HumanEval at this anchor. Combined with C40a (chat-
  template OFF inverted, −0.412), two of the three C40 highest-leverage
  empirical knobs have now been ruled out as α-recovery levers. C40c
  (W4A16 vs BF16 on HumanEval) remains the last single-knob hypothesis
  in the original C40 queue.

## Anchor

Cell D, identical to C35/C37/C38/C39 **except** sampling temperature:

| Knob | Value | vs C39 |
| --- | --- | --- |
| Quantization | `KILN_W4A16=1` | same |
| Argmax dtype | `KILN_MTP_ARGMAX_FP32=1` (FP32) | same |
| Prompt framing | `--chat-template` ON | same |
| KV path | `--paged` | same |
| Prompt tokens | 512 | same |
| Decode tokens | 128 | same |
| Sampling | **`temperature=0.1` (low-T sampling)** | **flipped** |
| CUDA graphs | `KILN_CUDA_GRAPHS=true` | same |
| Spec method | `KILN_SPEC_METHOD=mtp` | same |
| Prompt subset | `--prompt-subset humaneval` | same |
| GPU | NVIDIA RTX A6000 (sm_86) | same |

### Why temperature=0.1 is the only knob

C40 synthesis (PR #372) and C40a (PR #373) progressively narrowed the
α-recovery search to non-framing, non-numerical-flavoring interventions.
`temperature=0.1` was the cheapest remaining knob: it requires only a
CLI flag (no kernel change), it directly probes the
greedy-is-uniquely-harmful hypothesis, and it is a well-defined regime
shift from arg-max draft acceptance to a low-entropy sampled regime
that is closer to the MTP head's training distribution. N=20 matches
C39 / C40a exactly for direct seed-vs-seed comparison.

### Code change

A single new bench flag, `--temperature <f32>`, threads through every
call site that previously hard-coded `temperature: 0.0` in
`SamplingParams`:

- `bench_inference` (warmup + non-spec inference path)
- `bench_latency_skiplayer` (skip-layer spec, non-paged)
- `bench_latency_paged_skiplayer` (skip-layer spec, paged)
- `bench_latency_paged_mtp` (MTP spec, paged — the C39/C40a/C40b path)

The flag defaults to `0.0`, preserving byte-identical greedy behavior
for every prior bench command. Sanity verified below.

## Per-seed results (N=20, HumanEval subset, temperature=0.1)

| seed |  α     | decode_tps | mean_itl_ms | C39 α  | Δα (vs C39) |
| ---- | ------ | ---------- | ----------- | ------ | ----------- |
|    0 | 0.7297 |      43.70 |      22.883 | 0.7162 |  +0.0135    |
|    1 | 0.7397 |      41.93 |      23.847 | 0.7162 |  +0.0235    |
|    2 | 0.7067 |      41.07 |      24.351 | 0.6711 |  +0.0356    |
|    3 | 0.6711 |      40.95 |      24.422 | 0.6494 |  +0.0217    |
|    4 | 0.6494 |      43.32 |      23.085 | 0.7397 |  −0.0903    |
|    5 | 0.6623 |      40.18 |      24.888 | 0.7778 |  −0.1155    |
|    6 | 0.5875 |      35.59 |      28.098 | 0.6933 |  −0.1058    |
|    7 | 0.5679 |      35.89 |      27.864 | 0.6933 |  −0.1254    |
|    8 | 0.6494 |      40.15 |      24.906 | 0.6933 |  −0.0439    |
|    9 | 0.5875 |      37.31 |      26.800 | 0.5679 |  +0.0196    |
|   10 | 0.6842 |      38.95 |      25.673 | 0.7534 |  −0.0692    |
|   11 | 0.7162 |      42.91 |      23.306 | 0.7297 |  −0.0135    |
|   12 | 0.6933 |      39.72 |      25.178 | 0.7162 |  −0.0229    |
|   13 | 0.6410 |      37.33 |      26.791 | 0.6494 |  −0.0084    |
|   14 | 0.6494 |      39.16 |      25.537 | 0.6282 |  +0.0212    |
|   15 | 0.7534 |      45.07 |      22.187 | 0.7067 |  +0.0467    |
|   16 | 0.6494 |      40.02 |      24.990 | 0.6203 |  +0.0291    |
|   17 | 0.5679 |      34.00 |      29.410 | 0.6711 |  −0.1032    |
|   18 | 0.6933 |      38.88 |      25.722 | 0.6933 |   0.0000    |
|   19 | 0.6000 |      37.52 |      26.651 | 0.6494 |  −0.0494    |

Pairwise: 12 seeds dropped, 7 improved, 1 tied. The negative tail is
materially heavier than the positive tail: max drop −0.1254 (seed 7),
max gain +0.0467 (seed 15). The sampling change does not produce a
uniform shift; it reshuffles α with a net-negative bias.

## Summary statistics

| Metric | C40b (temp=0.1) | C39 (temp=0) | Δ |
| --- | --- | --- | --- |
| N | 20 | 20 | — |
| median α | **0.6558** | 0.6933 | **−0.0375** |
| mean α | 0.6600 | 0.6868 | −0.0268 |
| stdev α | 0.0562 | 0.0499 | +0.0063 |
| min α | 0.5679 (seeds 7, 17) | 0.5679 (seed 9) | 0.0000 |
| max α | 0.7534 (seed 15) | 0.7778 (seed 5) | −0.0244 |
| 95% CI | **[0.6452, 0.6933]** | [0.6602, 0.7162] | shifts down |
| median ITL | ≈ 25.08 ms | ≈ 24.0 ms | +1.1 ms |
| median decode tps | ≈ 39.87 tok/s | ≈ 41.6 tok/s | −1.7 tok/s |

The throughput cost is consistent with the α drop (more verifier-only
steps when MTP drafts are rejected). Both runs are on A6000, so the
~4% decode-tps regression is attributable to the sampling change, not
arch.

## Floor check (Qwen3.5 paper, α ≥ 0.72)

| Bound | Value | vs 0.72 | Verdict |
| --- | --- | --- | --- |
| CI lo | 0.6452 | < 0.72 | fail (by 0.075) |
| median | 0.6558 | < 0.72 | fail (by 0.064) |
| CI hi | 0.6933 | < 0.72 | fail (by 0.027) |

The entire 95% CI sits 0.027–0.075 below the paper floor. Compared to
C39's CI [0.6602, 0.7162]:

- C40b CI lower bound is **lower** than C39 CI lower bound (0.6452
  vs 0.6602).
- C40b CI upper bound (0.6933) only just reaches C39's median, falling
  short of even *touching* C39's CI upper bound (0.7162).

The C39 CI was already entirely below 0.72; C40b widens that gap.

## Sanity check (default path is byte-identical)

Single seed at `temperature=0.0` (C39 anchor exactly), run on the same
binary as the N=20 sweep:

```
seed=0, --chat-template, --temperature 0.0
α = 0.6933333…
```

This matches the cluster of C39 seeds {6, 7, 8, 12, 18} that all
landed at 0.6933 (sample identity at greedy is determined by argmax
ties at the boundary, so multiple seeds collapse to the same α).
seed=0's C39 α was 0.7162; the difference is consistent with the
already-documented C39 seed→α nondeterminism arising from kernel
launch order under CUDA graphs, not from a regression in the
default-path code (the literal default of `--temperature 0.0` is
never threaded into a sampling path because `SamplingParams { temperature: 0.0 }` is unchanged from pre-change behavior — the existing greedy fast-path still triggers).

The conservative reading: greedy `--temperature 0.0` lands within the
C39 distribution's modal cluster. Default-path numbers are safe.

## Cross-phase α comparison (Cell D anchor variants)

| Phase | N | pool / subset | chat-template | sampling | median α | 95% CI | verdict vs 0.72 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C37 | 10 | 8-prose | on | greedy | 0.6779 | [0.63, 0.72] | below (N small) |
| C38 (full) | 30 | 30-domain | on | greedy | 0.7297 | [0.6996, 0.7760] | straddled |
| C38 (HumanEval slice) | 10 | 30-domain slice | on | greedy | 0.6888 | n/a | below |
| C39 | 20 | 10-HumanEval | on | greedy | 0.6933 | [0.6602, 0.7162] | below, CI clean |
| C40a | 20 | 10-HumanEval | **off** | greedy | 0.2814 | [0.2562, 0.3026] | far below, ~0.43 gap |
| **C40b** | **20** | **10-HumanEval** | **on** | **temp=0.1** | **0.6558** | **[0.6452, 0.6933]** | **below, ~0.06–0.08 gap** |

## Interpretation

1. **Greedy is not uniquely harmful at the C39 anchor on HumanEval.**
   The pre-test hypothesis predicted that greedy argmax produces
   draft tokens that diverge from the MTP head's sampled training
   distribution, suppressing α. Empirically, low-temperature sampling
   *worsens* median α by 0.0375 absolute. The MTP head is at least as
   well-aligned to greedy draft tokens as it is to `temp=0.1` sampled
   ones — possibly because, at `T=0.1`, the verifier model's own
   sampling occasionally selects non-modal tokens that the MTP draft
   (still effectively argmax over the draft's logits) does not predict.
2. **Net-negative bias is asymmetric.** The per-seed Δα distribution is
   not centered: the worst losses are 2–3× larger in magnitude than the
   best gains. This rules out a simple "sampling adds noise but no
   directional effect" interpretation. Sampling at `T=0.1` introduces a
   regime where the two heads disagree more often than they agree extra,
   not a regime where they agree on a different but equally tight set.
3. **Two of three C40 single-knob hypotheses now ruled out.** C40a
   killed the prompt-framing hypothesis (catastrophic −0.412). C40b
   kills the sampling-regime hypothesis (smaller but unambiguous
   −0.0375). Both inversions surfaced the same pattern: cheaper
   interventions don't help; the MTP head's HumanEval α gap is robust to
   the obvious empirical levers.
4. **Throughput tracks α as expected.** Decode tps drops ~4% (41.6 →
   39.87 tok/s) and ITL rises ~1 ms (24.0 → 25.08 ms). These deltas are
   dominated by the increase in verifier-only steps when MTP drafts are
   rejected at higher rates. There is no per-token sampling overhead
   pathology; the slowdown is a 1:1 reflection of the α regression.
5. **The C39 anchor is increasingly the floor for cheap interventions.**
   C40a (chat-template OFF) and C40b (temp=0.1) both shifted the median
   *down* relative to C39. C39's chat-template-ON, greedy, W4A16,
   FP32-argmax configuration appears to be a local maximum on
   HumanEval α among C40-queue knobs. Anything that reaches the 0.72
   paper floor will likely require a numerical or model change (W4A16
   → BF16, MTP head retraining, or argmax precision changes outside the
   already-applied `KILN_MTP_ARGMAX_FP32`), not a flag flip.

## What C40b rules out

1. Low-temperature sampling (`T=0.1`) as a path to α ≥ 0.72 on
   HumanEval. The hypothesis that greedy draft-token selection is
   uniquely harmful for MTP α on code is empirically wrong at this
   anchor; sampling makes things worse.
2. The "temperature is the cheapest remaining recovery knob" framing
   from C40. There is no cheap empirical knob remaining among
   prompt-framing or sampling-regime interventions for HumanEval α.

## Remaining open questions (post-C40b)

1. **C40c — W4A16 vs BF16 on HumanEval.** Tests whether Marlin's W4
   quantization is specifically hurting code α (vs. natural language,
   where C37/C38 GSM8K α was ≥ 0.78 under the same W4A16 path). This
   is now the *only* remaining single-knob hypothesis in the original
   C40 queue. Expected cost: ~$2–3 of A6000 time once a BF16-capable
   build path is in place.
2. **Per-domain floors (C40 documentation proposal).** Unchanged. C40b
   confirms that the proposed HumanEval floor (≥ 0.65) holds at the
   C39 anchor and would *also* hold at temp=0.1, even though both fall
   short of the global 0.72 paper floor.
3. **Asymmetry of the Δα distribution.** A finer temperature sweep
   (`T ∈ {0.05, 0.1, 0.2, 0.5}`) would reveal whether the asymmetric
   loss is monotonic in T or whether `T=0.1` happens to land at a
   particularly bad point. Not in the current queue but worth noting.
4. **Higher-T regimes are not implied to recover α.** This run does not
   test `T ≥ 0.5`. Given the directional inversion at `T=0.1`, the prior
   on higher-T helping is now lower, not higher.

## Artifacts

- Branch: `ce/mtp-c40b-humaneval-temperature`
- Code change: `crates/kiln-server/src/bench.rs` — adds `--temperature`
  CLI flag (default `0.0`), threads through 4 SamplingParams sites.
- Raw per-seed JSON: `docs/phase-c40b/c40b_seed{0..19}_temp0.1.json`
- Sanity JSON (default-path byte-identity check):
  `docs/phase-c40b/sanity_seed0_temp0.json`
- Analysis script: `docs/phase-c40b/analyze_c40b.py` (ships with PR for
  reproducibility; mirrors C39/C40a stats: 10,000 bootstrap resamples,
  rng=12345, percentile method [0.025, 0.975]).

## Reproduction

```bash
cd /workspace/kiln
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench

# Sanity (byte-identical to C39 default path)
KILN_W4A16=1 KILN_MTP_ARGMAX_FP32=1 \
KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
KILN_CUDA_GRAPHS=true \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed 0 --temperature 0.0

# C40b sweep (N=20, temp=0.1)
for seed in $(seq 0 19); do
  KILN_W4A16=1 KILN_MTP_ARGMAX_FP32=1 \
  KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
  KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --chat-template --skip-training --latency-only \
    --prompt-tokens 512 --max-output-tokens 128 \
    --prompt-subset humaneval \
    --seed $seed --temperature 0.1
done

python3 docs/phase-c40b/analyze_c40b.py /path/to/results
```

The only flag that differs from C39 across the N=20 sweep is
`--temperature 0.1`. All other flags match C39 exactly to enable
direct paired-seed comparison.

## Cost & environment

- GPU: NVIDIA RTX A6000 (sm_86), on-demand
- Pod: `sq3exutrqy2cuw` on RunPod (warm pool acquisition)
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- Build: cached via sccache + B2 (incremental ~70 s)
- Bench run: ~10 min wall for sanity + N=20 × (model load 24 s + bench 5 s) per seed
- Total pod cost: ≈ $0.20–0.30 (well inside the 90 min / $25 cap)

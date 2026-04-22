# Phase C37 — variance re-anchor at C35 Cell D (L=128, N=10)

**Date:** 2026-04-22
**Hardware:** RunPod A6000 (49 140 MB), pod `sq3exutrqy2cuw`, image `ghcr.io/ericflo/kiln-runpod:latest`
**Build:** branch `ce/mtp-c37-variance-reanchor` based on `main` @ `35c1c54` (post-C36 merge; Metal-only deltas since C36 runs)
**Wall-clock:** ~19 min for the 10-seed sweep; pod time ≈ $0.18 incremental on an already-warm lease

## 1. Question

Both C35 (§4.1) and C36 (§6) flagged that the measured α distribution at Cell D (chat-template × FP32 argmax × `prompt_tokens=512` × `max_output_tokens=128`) was too thinly sampled to decide whether the MTP head clears the paper-floor target α = 0.72 under W4A16. Concretely:

- C35 reported median α = **0.6076** at this cell. The C36 doc describes this anchor as N = 3 seeds; that attribution is being revisited here because C36 seed 0 at the identical cell landed at α = 0.5875.
- C36 re-measured the same cell with N = 3 seeds {0, 1, 2} and got median α = **0.7067** — a +0.0991 swing with zero config drift. C36 §6 explicitly called out that N = 3 is structurally insufficient at this decode length.

C37 pins this down directly: re-measure Cell D with **N = 10 independent seeds** and report the median, P10–P90 band, standard deviation, and 95% CI so we know what the Cell D α distribution actually is before spending more kernel-fusion budget hunting the 0.72 gap.

## 2. Method

Single configuration cell, held fixed across all 10 runs:

```
KILN_W4A16=1 KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp KILN_MTP_ARGMAX_FP32=1 KILN_CUDA_GRAPHS=true
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template \
  --prompt-tokens 512 --max-output-tokens 128 \
  --skip-training --seed {0..9}
```

This is the canonical C35 / C36 L=128 Cell D invocation. `--seed s` selects prompt `PROMPT_POOL[s % 8]` (see `crates/kiln-server/src/bench.rs:259-303`) AND seeds the sampling `StdRng` threaded into the MTP draft path. Both are taken at their default Cell D values (chat template on, BF16 weights, FP32 argmax, Marlin W4A16 for q_proj and MLP, paged KV, CUDA graphs on).

10 bench invocations against the same warm binary; no code changes; no pod restarts between runs.

## 3. Per-seed results

```
seed  prompt  alpha   decode tok/s  p50 ITL ms  p99 ITL ms
   0       0  0.5679         35.20       17.79       69.54
   1       1  0.7778         44.38       17.70       59.03
   2       2  0.5679         33.75       19.01       62.62
   3       3  0.6933         40.82       17.88       59.12
   4       4  0.6842         39.35       18.66       61.24
   5       5  0.6076         39.73       16.89       52.84
   6       6  0.5610         37.95       16.78       55.75
   7       7  0.5488         37.39       16.87       56.68
   8       0  0.6076         39.79       16.85       57.30
   9       1  0.6711         41.02       17.33       56.68
```

## 4. Aggregate statistics (N = 10)

| Metric          | α        |
|-----------------|----------|
| Median          | **0.6076** |
| Mean            | 0.6287   |
| Std dev         | 0.0749   |
| P10             | 0.5598   |
| P90             | 0.7018   |
| Min             | 0.5488   |
| Max             | 0.7778   |
| 95 % CI (mean ± 1.96 σ/√N) | [0.5823, 0.6752] |

Decode throughput (N = 10): median **39.54 tok/s**, mean 38.94, σ = 3.04.

Gap to paper floor α = 0.72:
- Median α: **−0.1124** below floor
- P90 α: −0.0182 (just below floor)
- Max α: +0.0578 (only one seed clears)

## 5. Cross-phase comparison

| Source | N | L=128 Cell D median α | Notes |
|--------|---|------------------------|-------|
| C35 §4.1 (as anchored in C36 doc)  | 3 (per C36 doc) | 0.6076 | Single-cell baseline; N-attribution is mildly unclear between phases |
| C36 §3 (L=128 row)                 | 3               | 0.7067 | Seeds {0, 1, 2}; seed 0 = 0.5875 |
| **C37 (this run)**                 | **10**          | **0.6076** | Seeds {0..9}; median coincides with C35 anchor |

Key observations:

1. **C36's 0.7067 median is outside the C37 95 % CI [0.5823, 0.6752].** Treating the C37 distribution as the ground truth under this prompt pool, C36's N = 3 draw was an upper-tail fluke. The earlier +0.0991 "gain" over C35 is a sampling artifact, not a real shift.
2. **C35's 0.6076 anchor lands on the C37 median exactly.** That is a coincidence (C37 seed 0 = 0.5679, not 0.6076), but it means C35's published number is well inside the C37 distribution and does not need a retroactive correction.
3. **Greedy MTP is *not* bit-deterministic across runs with the same prompt.** Seed 0 and seed 8 both hit `PROMPT_POOL[0]` with temperature = 0.0, and C37 still measures α = 0.5679 vs 0.6076 — a Δ of 0.0397. The sampling `StdRng::seed_from_u64(seed)` at `bench.rs:882` and the analogous MTP-path RNG at `bench.rs:1383` thread into the draft path somewhere, so seed-level variance is not purely prompt-selection variance. This invalidates the earlier intuition that N > 8 is wasted under greedy.
4. **The 8-prompt pool (plus RNG noise) still caps the effective variance budget.** Even N = 10 is only 8 distinct prompts × 2 RNG noise samples. Widening the prompt pool is required before tighter CIs can be trusted.

## 6. Verdict

**Decision-branch match (from the task spec):**

| Branch | Trigger | Fires? |
|--------|---------|--------|
| A | median α ≥ 0.68 | No (0.6076) |
| B | median α ∈ [0.55, 0.68] with wide P10–P90 | **Yes** — median 0.6076, P10–P90 spread 0.142 |
| C | median α < 0.55 | No |

→ **Branch B fires.** C36's 0.7067 was a lucky 3-seed upper-tail draw; the true Cell D α distribution under the current prompt pool sits around 0.61 with a std of ~0.07, and clears the 0.72 paper floor in only 1 of 10 seeds. Continuing to hunt kernel-side explanations for the gap without first widening the prompt distribution would be a directed search against noise.

**C35 Cell D stands as the correct anchor. C36's L=128 median should be flagged as under-sampled; the longer-decode rows (L = 512, 1024) in C36 are less noisy (spread Δ ≤ 0.017) and still do not clear 0.72.**

## 7. Recommended C38

**C38 (H14b — prompt-content distribution), the C36 §6 preferred track, now with a concrete N and pool:**

1. Extend `PROMPT_POOL` beyond the current 8 synthetic prose bases with at least three canonical task distributions:
   - **GSM8K** short math word-problem prompts
   - **HumanEval** Python completion prompts
   - **C4** prose continuation prompts
2. Re-bench Cell D with **N = 30 seeds** drawn from the widened pool so each prompt sees ≥ 2 RNG samples, giving a 95 % CI width below ±0.03.
3. Decision rule:
   - If **any** prompt-bucket median α ≥ 0.72 reproducibly (e.g. GSM8K median clears) → close the paper-floor gap as a corpus-selection artifact and document Cell D's applicability envelope.
   - If **no** bucket clears 0.72 → fall back to C36 §6's third candidate (residual MTP head weight quality) and plan a direct head-replacement A/B against the reference HF-Qwen3.5-4B MTP head.

Explicitly **deprioritised** after this run:
- Per-layer W4A16 ablation (Branch A plan from the C37 task spec). With median α well below 0.68, no kernel-side ablation is going to bridge the 0.11 gap under a prompt pool this narrow.
- Any further L-sweep work. C36 §5 already showed the α mean is flat across {128, 256, 512, 1024} once variance is accounted for.

## 8. Notes on task-spec data accuracy

The C37 task description referenced two numbers that do not match the primary sources:

- Stated "C36 seed 0 α = 0.6076." C36 §3 (`docs/phase-c36/c36-h14a-decode-length-sweep.md`) records seed 0 at L = 128 as α = **0.5875**. The 0.6076 value belongs to the C35 Cell D anchor.
- Stated "C35 median 0.6076, N = 1." The C36 doc's reference table attributes the 0.6076 anchor to **N = 3 seeds**. The C35 source doc should be cross-checked for N, but neither number invalidates any C37 conclusion here — the C37 distribution contains both 0.5875 and 0.6076 well within its ±1 σ band.

Neither misattribution changes the C37 verdict. Flagging both so downstream phases quote from the primary C35 / C36 docs, not the C37 task spec.

## 9. Reproducibility

Raw JSONs (`seed-0.json` … `seed-9.json`) and the driver script (`run_c37_bench.sh`) are pulled off-pod and committed alongside this doc. Rerun with the exact invocation in §2 against `main` @ `35c1c54` on an A6000. All 10 runs completed rc = 0; per-run total wall-clock ≈ 110 s (dominated by model load).

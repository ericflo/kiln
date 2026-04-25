# Phase C40f — HumanEval BF16-only N=20 alpha distribution re-bench on current main

## TL;DR

BF16 does **not** close the C40 quantization question distributionally on
current `main`.

- C40e's paired `seed=0` result still reproduces cleanly: BF16
  `0.7887` vs the C40e W4A16 `0.6282`.
- But the BF16 **N=20 HumanEval distribution** on current `main`
  lands at median **`0.6888`**, mean `0.6939`, bootstrap 95% CI
  **`[0.6517, 0.7232]`**, with only **6/20** seeds clearing `0.72`.
- That does **not** beat the already-landed C39 W4A16 distribution
  (median `0.6933`, CI `[0.6602, 0.7162]`) in a way that supports a
  BF16-favored conclusion. The BF16 median is slightly lower and the CI
  still straddles the 0.72 floor.

Verdict: **the single-seed C40e BF16 win does not generalize to an
N=20 HumanEval distribution on current `main`.** Do not close the C40
quantization question in BF16's favor from this evidence.

## Preflight

All required stop-condition checks passed before pod spend:

1. `ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status pending`
   returned no pending tasks for this project.
2. `ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status running`
   showed only this C40f task.
3. `gh pr list --repo ericflo/kiln --limit 15 --state all` showed
   `#378` as the newest relevant MTP PR; no later PR had already landed
   an N>=10 HumanEval BF16 sweep.
4. In a fresh clone, all required bench-surface files existed:
   - `crates/kiln-server/src/bench.rs`
   - `PROFILING-MTP-C40e.md`
   - `docs/archive/phase-c/phase-c40e/`
5. `origin/main` was still exactly the `#378` merge commit
   `199d16bcc65251c9a34fc60249c9a17840d15514`, so no post-C40e bench-path
   drift existed to confound the comparison.

## Validation

Required command:

```bash
source /root/.kiln-build-env && export KILN_CUDA_ARCHS=86 CARGO_PROFILE_DEV_DEBUG=0 && cd /workspace/kiln && cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

Outcome: success on the benchmark pod before the sweep.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-04-22 |
| Commit under test | `199d16bcc65251c9a34fc60249c9a17840d15514` |
| Short SHA | `199d16b` |
| Pod | `8xti6zgiceq76v` |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Requested GPU | NVIDIA RTX A6000 |
| Actual GPU | NVIDIA A40 |
| A6000 fallback reason | RunPod `SUPPLY_CONSTRAINT`; A40 is the next allowed fallback in project policy |
| Checkpoint source | `hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b` |
| Binary SHA-256 | `a7faee2612f87bdf1cf6a17d52754b7c4270f40a0f9e6d03ee84ff4a0d93c7f4` |

## Command Surface

Env for every run:

```bash
KILN_W4A16=0
KILN_MTP_ARGMAX_FP32=1
KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp
KILN_CUDA_GRAPHS=true
```

Command for every run:

```bash
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --temperature 0.0 \
  --seed <0..19>
```

## Per-seed results

| seed | alpha | decode_tps | mean_itl_ms |
| ---: | ---: | ---: | ---: |
| 0 | 0.7887 | 44.36 | 22.540 |
| 1 | 0.6842 | 38.98 | 25.652 |
| 2 | 0.6282 | 36.52 | 27.385 |
| 3 | 0.7067 | 34.29 | 29.160 |
| 4 | 0.6410 | 36.84 | 27.146 |
| 5 | 0.7397 | 40.80 | 24.512 |
| 6 | 0.6623 | 37.56 | 26.627 |
| 7 | 0.6410 | 34.85 | 28.698 |
| 8 | 0.6933 | 39.31 | 25.441 |
| 9 | 0.7534 | 41.44 | 24.131 |
| 10 | 0.7887 | 43.94 | 22.758 |
| 11 | 0.6842 | 38.94 | 25.679 |
| 12 | 0.6282 | 36.18 | 27.638 |
| 13 | 0.7067 | 37.23 | 26.863 |
| 14 | 0.6410 | 32.00 | 31.246 |
| 15 | 0.7397 | 41.82 | 23.914 |
| 16 | 0.6623 | 36.88 | 27.118 |
| 17 | 0.6410 | 36.34 | 27.518 |
| 18 | 0.6933 | 39.52 | 25.306 |
| 19 | 0.7534 | 39.06 | 25.600 |

Observation: seeds `0/10`, `1/11`, ..., `9/19` produced identical alpha
values, consistent with HumanEval subset cycling every 10 prompts under
greedy BF16 decode. Timing metrics still varied run-to-run.

## Summary statistics

Bootstrap method: 10,000 median resamples with fixed RNG seed `12345`,
sampling from the 20 observed alpha values with replacement.

| Metric | Value |
| --- | --- |
| N | 20 |
| median alpha | **0.6888** |
| mean alpha | 0.6939 |
| stdev alpha | 0.0521 |
| min alpha | 0.6282 |
| max alpha | 0.7887 |
| 95% CI on median | **[0.6517, 0.7232]** |
| seeds with alpha >= 0.72 | **6 / 20** |
| median decode tok/s | 38.25 |
| median ITL | 26.153 ms |

## Cross-phase comparison

### Against C39 W4A16 HumanEval N=20

| Phase | Load mode | GPU | median alpha | mean alpha | 95% CI | seeds >= 0.72 |
| --- | --- | --- | --- | --- | --- | --- |
| C39 | W4A16 (`KILN_W4A16=1`) | A6000 | 0.6933 | 0.6868 | [0.6602, 0.7162] | 4 / 20 |
| C40f | BF16 (`KILN_W4A16=0`) | A40 | 0.6888 | 0.6939 | [0.6517, 0.7232] | 6 / 20 |

Distributional read:

- BF16 **does not materially beat** the C39 W4A16 HumanEval distribution.
- The BF16 median is slightly **lower** than C39 (`-0.0045` absolute).
- The BF16 mean is slightly higher (`+0.0071`), but that lift is driven by
  a few high seeds rather than a broad shift in the center.
- The BF16 CI still straddles `0.72`, so the HumanEval floor remains
  unresolved distributionally.

Decode tok/s is not a clean cross-phase comparison here because C39 ran
on A6000 and C40f fell back to A40 after an A6000 supply constraint.

### Against C40e paired seed 0

| Phase | seed | W4A16 alpha | BF16 alpha | BF16 delta |
| --- | --- | --- | --- | --- |
| C40e | 0 | 0.6282051282 | 0.7887323944 | +0.1605272662 |
| C40f | 0 | n/a | 0.7887323944 | matches C40e BF16 exactly |

This is the contradiction C40f was asked to resolve:

- The C40e single-seed BF16 win is real and reproducible.
- But it does **not** imply a BF16-favored N=20 HumanEval distribution.

## Floor check

Decision rule target: median-alpha CI lower bound `>= 0.72`.

| Bound | Value | vs 0.72 | Verdict |
| --- | --- | --- | --- |
| CI lo | 0.6517 | below | fail |
| median | 0.6888 | below | fail |
| CI hi | 0.7232 | above | straddles |

BF16 therefore does **not** clear the 0.72 HumanEval floor
distributionally on current `main`.

## Verdict

**Contradiction documented explicitly:** BF16 wins the paired seed-0
comparison from C40e, but BF16 **fails to beat** the C39 W4A16
HumanEval N=20 distribution in a way that supports a BF16-favored
decision.

That means:

1. The C40e seed-0 BF16 result should be treated as a strong
   single-seed datapoint, not a distributional conclusion.
2. C40f does **not** justify closing the HumanEval quantization question
   in BF16's favor.
3. The HumanEval floor remains distributionally ambiguous under BF16 on
   current `main`.

## Recommendation

Do **not** replace the C39 distributional conclusion with a BF16-favored
claim. Keep C40e as the clean single-seed paired result, but use C40f as
the distributional verdict: **BF16 does not generalize into a better
HumanEval N=20 distribution on current `main`.**

## Artifacts

- Raw BF16 outputs and stderr logs:
  `docs/archive/phase-c/phase-c40f/seed-{0..19}.json`,
  `docs/archive/phase-c/phase-c40f/seed-{0..19}.log`,
  `docs/archive/phase-c/phase-c40f/seed-{0..19}.exit`
- Exact env and command:
  `docs/archive/phase-c/phase-c40f/common-env.txt`,
  `docs/archive/phase-c/phase-c40f/command-template.txt`,
  `scripts/phase-c40f/run-all-seeds.sh`
- Binary hash:
  `docs/archive/phase-c/phase-c40f/kiln-bench.sha256`
- Analysis script:
  `scripts/phase-c40f/analyze.py`

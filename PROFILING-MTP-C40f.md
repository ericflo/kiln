# Phase C40f — HumanEval BF16-only N=20 alpha distribution re-bench on current main

## TL;DR

This re-ran the C40e BF16 HumanEval path on current `main`, but across
an N=20 seed distribution instead of just the paired seed-0 point.

Result on `main` commit `199d16bcc65251c9a34fc60249c9a17840d15514`
(PR `#378`, 2026-04-22):

| Metric | BF16 C40f |
| --- | --- |
| median alpha | `0.6887719298` |
| mean alpha | `0.6938687678` |
| stdev alpha | `0.0521101775` |
| min / max alpha | `0.6282051282` / `0.7887323944` |
| 95% bootstrap CI on median | `[0.6516816517, 0.7231963470]` |
| seeds clearing `0.72` | `6 / 20` |
| median decode tok/s | `37.6990` |
| median mean ITL ms | `26.5262` |

Primary verdict: **BF16 does not beat the already-landed C39 W4A16
HumanEval N=20 distribution on the median**, even though the CI now
barely straddles the `0.72` floor. That directly contradicts the
single-seed C40e story as a distributional claim: the seed-0 BF16 gain
does not generalize cleanly across N=20 on current `main`.

## Preflight

Before pod spend, the required stop conditions were checked.

1. **No newer BF16 N>=10 HumanEval sweep after `#378`**
   - GitHub PR search over `ericflo/kiln` found no open or merged PR
     newer than `#378` covering this exact BF16 HumanEval distribution
     re-bench.
2. **No overlapping kiln PR/task for this exact sweep**
   - No open kiln PR matched `c40f` / BF16 HumanEval distributional
     re-bench.
   - The project planning task was running, but it was not executing
     this exact benchmark.
3. **No post-`#378` bench-path drift**
   - Fresh clone of `ericflo/kiln` landed at `199d16b`, the merge commit
     for `#378`.
   - So current `main` had **no commits after C40e**; there was no new
     drift to confound a same-path comparison.
4. **Bench surface still exists in a fresh clone**
   - Verified in:
     - `crates/kiln-server/src/bench.rs`
     - `crates/kiln-model/src/speculative.rs`
     - `crates/kiln-model/src/generate.rs`
     - `crates/kiln-core/src/sampling.rs`

Because all four checks passed, the sweep remained valid to run.

## Required validation

Required command:

```bash
source /root/.kiln-build-env && export KILN_CUDA_ARCHS=86 CARGO_PROFILE_DEV_DEBUG=0 && cd /workspace/kiln && cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

Outcome: success on the same RTX A6000 pod before the sweep.

## Environment

| Item | Value |
| --- | --- |
| Date | `2026-04-22` |
| Commit under test | `199d16bcc65251c9a34fc60249c9a17840d15514` |
| Short SHA | `199d16b` |
| GPU | `NVIDIA RTX A6000` |
| Pod | `dcdaeukr7dvskz` |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Checkpoint | `Qwen/Qwen3.5-4B` at `/workspace/qwen3.5-4b` |
| Binary SHA-256 | `c4f75910cb71718a2846fd2bc4a9bbf7c498ab94a1e6ce8b9b39e6a84cfea3a0` |

Note: the pod image does not preload the checkpoint, so the public
`Qwen/Qwen3.5-4B` weights were downloaded to the same path C40e used
before starting the sweep. The bench command surface itself stayed
unchanged.

## Exact command surface

Env:

```bash
KILN_W4A16=0
KILN_MTP_ARGMAX_FP32=1
KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp
KILN_CUDA_GRAPHS=true
```

Command:

```bash
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed <SEED> --temperature 0.0
```

This matches the current-main C40e BF16 leg, except expanded to seeds
`0..19`.

## Per-seed results

| seed | alpha | decode tok/s | mean ITL ms |
| --- | --- | --- | --- |
| 0 | `0.7887323944` | `45.8126` | `21.8281` |
| 1 | `0.6842105263` | `37.5755` | `26.6131` |
| 2 | `0.6282051282` | `35.7954` | `27.9366` |
| 3 | `0.7066666667` | `40.8491` | `24.4803` |
| 4 | `0.6410256410` | `31.0214` | `32.2358` |
| 5 | `0.7397260274` | `35.3660` | `28.2758` |
| 6 | `0.6623376623` | `37.8225` | `26.4393` |
| 7 | `0.6410256410` | `35.3006` | `28.3281` |
| 8 | `0.6933333333` | `42.4089` | `23.5799` |
| 9 | `0.7534246575` | `44.2104` | `22.6191` |
| 10 | `0.7887323944` | `37.4989` | `26.6675` |
| 11 | `0.6842105263` | `35.9969` | `27.7802` |
| 12 | `0.6282051282` | `38.2936` | `26.1140` |
| 13 | `0.7066666667` | `38.9331` | `25.6851` |
| 14 | `0.6410256410` | `35.2459` | `28.3721` |
| 15 | `0.7397260274` | `42.9373` | `23.2898` |
| 16 | `0.6623376623` | `35.9644` | `27.8052` |
| 17 | `0.6410256410` | `33.7404` | `29.6381` |
| 18 | `0.6933333333` | `38.9687` | `25.6616` |
| 19 | `0.7534246575` | `39.2624` | `25.4697` |

## Summary statistics

| Metric | Value |
| --- | --- |
| N | `20` |
| median alpha | `0.6887719298` |
| mean alpha | `0.6938687678` |
| stdev alpha | `0.0521101775` |
| min alpha | `0.6282051282` |
| max alpha | `0.7887323944` |
| 95% CI (bootstrap 10k resamples, rng=`12345`) | `[0.6516816517, 0.7231963470]` |
| seeds clearing `0.72` | `6 / 20` |
| median decode tok/s | `37.6990270692` |
| median mean ITL ms | `26.5261682480` |

## Bootstrap method

Method matches C39/C40a/C40b:

- statistic: median alpha
- resamples: `10,000`
- RNG seed: `12345`
- CI: percentile bootstrap (`p2.5` / `p97.5`)

Reproducible from committed raw JSONs via
`docs/phase-c40f/analyze_c40f.py`.

## Cross-phase comparison

### vs C39 HumanEval W4A16 N=20 (`#371`)

| Metric | C39 W4A16 | C40f BF16 | Delta |
| --- | --- | --- | --- |
| median alpha | `0.6933` | `0.6888` | `-0.0045` |
| mean alpha | `0.6868` | `0.6939` | `+0.0071` |
| stdev alpha | `0.0499` | `0.0521` | `+0.0022` |
| 95% CI | `[0.6602, 0.7162]` | `[0.6517, 0.7232]` | lower `-0.0085`, upper `+0.0070` |
| seeds clearing `0.72` | `4 / 20` | `6 / 20` | `+2` |
| median decode tok/s | `41.6` | `37.7` | `-3.9` |
| median mean ITL ms | `24.0` | `26.5` | `+2.5` |

This is the core contradiction:

- BF16 raises the upper tail enough for the CI to **barely cross 0.72**.
- But BF16 does **not** improve the center of the distribution; the
  median is slightly **worse** than the C39 W4A16 median.
- Throughput also regresses relative to C39.

So the honest distributional read is not "BF16 wins"; it is "BF16
changes the shape of the distribution, but does not beat W4A16 on the
primary median statistic."

### vs C40e paired seed 0 (`#378`)

| Metric | C40e seed-0 BF16 | C40f seed-0 BF16 | Delta |
| --- | --- | --- | --- |
| alpha | `0.7887323944` | `0.7887323944` | `0.0000` |
| vs paired W4A16 seed 0 | `+0.1605272662` | same | same |

This confirms the C40f seed-0 leg reproduced the C40e BF16 paired point
exactly. The contradiction is therefore **not** a failed reproduction of
C40e; it is that the strong seed-0 win does not survive as an N=20
distributional claim.

## Floor check (`0.72`)

| Bound | Value | vs `0.72` | Verdict |
| --- | --- | --- | --- |
| CI lo | `0.6517` | below | fail |
| median | `0.6888` | below | fail |
| CI hi | `0.7232` | above | straddles |

BF16 on current `main` therefore **does not clear the HumanEval floor
distributionally**. It improved just enough in the upper tail to make
the floor question ambiguous again, but not enough in the center to
claim success.

## Interpretation

1. **C40e was real, but narrow.** Seed 0 reproduced byte-for-byte and
   still shows a large BF16 win over W4A16.
2. **That win does not generalize cleanly.** Across N=20, BF16 median
   alpha is slightly below the C39 W4A16 median, so the single-seed
   paired result is not a valid substitute for a distributional claim.
3. **BF16 changes the tails more than the center.** `6/20` BF16 seeds
   clear `0.72` vs `4/20` for C39 W4A16, and the CI upper bound now
   crosses `0.72`. But the lower bound worsens and the median slips.
4. **The quantization question remains unresolved distributionally.**
   The best current evidence is internally contradictory:
   - paired seed 0: strong BF16 win
   - N=20 median: no BF16 win
   - N=20 CI: still ambiguous on the floor

## Verdict

**BF16 on current `main` does not clear the HumanEval `0.72` floor
distributionally, and it does not beat the C39 W4A16 N=20 distribution
on the median.**

That means the correct C40f conclusion is:

- **not** "BF16 wins"
- **not** "BF16 clears the floor"
- **not** "the question is closed in BF16's favor"

Instead:

> **The C40e seed-0 BF16 gain is real but non-generalizing. On the N=20
> HumanEval distribution, BF16 produces an ambiguous floor result and a
> slightly worse median than C39 W4A16.**

Per the task decision rule, this contradiction is documented directly
and no new recovery knob is proposed here.

## Recommendation

Do **not** close the C40 quantization question in BF16's favor from this
evidence. The distributional result is mixed and the primary median
comparison went the wrong way.

## Artifacts

- `docs/phase-c40f/seed-{0..19}.json`
- matching `docs/phase-c40f/seed-{0..19}.log`
- `docs/phase-c40f/seed-{0..19}.exit`
- `docs/phase-c40f/common-env.txt`
- `docs/phase-c40f/command.txt`
- `docs/phase-c40f/run_c40f.sh`
- `docs/phase-c40f/git-head.txt`
- `docs/phase-c40f/kiln-bench.sha256`
- `docs/phase-c40f/analyze_c40f.py`

The `seed-*.exit` files are the raw driver outputs and contain the
literal text `0n`; the substantive success signals are the completed
JSON/log pairs for all 20 seeds plus `done.txt`.

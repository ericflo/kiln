# Phase C40e — HumanEval BF16 vs W4A16 paired re-bench after anchor-drift audit

## TL;DR

C40d removed the exact-value `0.6933333333` seed-0 gate, so this task
ran the missing paired comparison directly on the **same current-main
binary**, on the **same pod**, with the **same command line**, flipping
only `KILN_W4A16`.

Result on `main` commit `dab806eb87d5d811dae6bdfc114967e54ab3495e`
(post-C40d, 2026-04-22):

| Leg | `KILN_W4A16` | alpha | vs 0.72 floor | decode tok/s | mean ITL ms |
| --- | --- | --- | --- | --- | --- |
| A | `1` (current default path) | `0.6282051282` | fail by `0.0918` | `38.34` | `26.08` |
| B | `0` (BF16 dense weights) | `0.7887323944` | pass by `0.0687` | `46.71` | `21.41` |

Paired delta: **BF16 beats W4A16 by `+0.1605272662` alpha** on the same
seed-0 HumanEval run. Both legs exited successfully. BF16 also improved
decode throughput by `+8.38 tok/s` and reduced mean ITL by `-4.68 ms`.

Direct recommendation: **for the C40 HumanEval path on current main,
replace W4A16 with BF16.** This exact paired run is the cleanest answer
the C40 queue asked for: same binary, same pod, same flags, only
`KILN_W4A16` changed, and BF16 both recovers alpha and clears the 0.72
paper floor while W4A16 misses it.

This is still a **single-seed** result, not an N=20 distributional
claim. It is strong enough to choose the BF16 path for C40 follow-up,
but not strong enough by itself to rewrite all broader quantization
guidance.

## Preflight

Before pod spend, the requested stop conditions were checked:

1. **No duplicate PR after #376**
   - GitHub PR search over `ericflo/kiln` found no open or merged PR
     after #376 containing the actual BF16-vs-W4A16 paired comparison.
   - Relevant recent PRs were:
     - `#375` C40c abort
     - `#376` C40d anchor-drift audit
2. **No overlapping task**
   - Project running/pending task state showed no other in-flight task
     performing this exact paired re-bench.
3. **Current `main` still matches post-C40d**
   - `origin/main` was identical to C40d merge commit
     `dab806eb87d5d811dae6bdfc114967e54ab3495e`.
   - There were **no commits after #376**, so the stale-comparison stop
     condition did not trigger.

Because all three gates passed, the benchmark remained valid to run.

## Required validation

Required command:

```bash
source /root/.kiln-build-env && export KILN_CUDA_ARCHS=86 CARGO_PROFILE_DEV_DEBUG=0 && cd /workspace/kiln && cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

Outcome: success on the same RTX A6000 pod used for the paired bench.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-04-22 |
| Commit under test | `dab806eb87d5d811dae6bdfc114967e54ab3495e` |
| Short SHA | `dab806e` |
| GPU | NVIDIA RTX A6000 |
| Pod | `sunq14rou988pi` |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Checkpoint | `Qwen/Qwen3.5-4B` at `/workspace/qwen3.5-4b` |
| Binary SHA-256 | `c4f75910cb71718a2846fd2bc4a9bbf7c498ab94a1e6ce8b9b39e6a84cfea3a0` |

## Exact command surface

Common env for both legs:

```bash
KILN_MTP_ARGMAX_FP32=1 \
KILN_SPEC_ENABLED=1 \
KILN_SPEC_METHOD=mtp \
KILN_CUDA_GRAPHS=true
```

Common command for both legs:

```bash
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed 0 --temperature 0.0
```

Leg-specific env:

- Leg A: `KILN_W4A16=1`
- Leg B: `KILN_W4A16=0`

No other flag or env change was made between the two runs.

## Paired result

### Leg A — current default path (`KILN_W4A16=1`)

| Metric | Value |
| --- | --- |
| Exit code | `0` |
| Alpha | `0.6282051282051282` |
| Accepted / attempts | `49 / 78` |
| Decode tok/s | `38.336376878417745` |
| Mean ITL ms | `26.084885464566963` |
| Prefill tok/s | `60.322802269159546` |
| Prefill time ms | `8222.429684` |
| Model load secs | `22.193330076` |
| Model VRAM MB | `18011` |

### Leg B — BF16 dense weights (`KILN_W4A16=0`)

| Metric | Value |
| --- | --- |
| Exit code | `0` |
| Alpha | `0.7887323943661971` |
| Accepted / attempts | `56 / 71` |
| Decode tok/s | `46.714152981300984` |
| Mean ITL ms | `21.406788653543302` |
| Prefill tok/s | `1487.4179807812102` |
| Prefill time ms | `333.463765` |
| Model load secs | `2.132304785` |
| Model VRAM MB | `16795` |

### Delta (BF16 minus W4A16)

| Metric | Delta |
| --- | --- |
| Alpha | `+0.1605272661610689` |
| Decode tok/s | `+8.377776102883239` |
| Mean ITL ms | `-4.678096811023661` |
| Model VRAM MB | `-1216` |

## Interpretation

1. **BF16 recovers alpha on this C40 path.**
   This is the exact direct comparison C40c could not complete while the
   historical singleton anchor was treated as blocking. Once C40d
   removed that block, the paired answer is clear: BF16 wins.
2. **The paired design matters.**
   The historical seed-0 exact-value check was unstable enough to be a
   bad gate. This run avoids that problem by comparing the two load
   modes directly on the same binary and same warm pod.
3. **The result is materially large, not marginal.**
   `+0.1605` alpha is far larger than the `~0.02–0.05` drift observed in
   the recent anchor audit. On this seed, W4A16 lands below the 0.72
   paper floor while BF16 clears it.
4. **This is still one seed.**
   The recommendation here is targeted: use BF16 for the C40 HumanEval
   path. A future broader policy change should still be supported by a
   short paired multi-seed follow-up.

## Recommendation

**Yes: BF16 should replace W4A16 for the C40 HumanEval path on current
main.**

Reasoning:

- W4A16 (`0.6282`) fails the 0.72 floor on this paired run.
- BF16 (`0.7887`) clears the 0.72 floor on the same binary/pod/flags.
- The alpha gain (`+0.1605`) is large enough that the C40d
  anchor-instability caveat does not change the decision for this path.

Practical reading: if the goal is the best current-main HumanEval MTP
alpha on the C40 setup, BF16 is the right load mode.

## Artifacts

- [docs/phase-c40e/command.txt](docs/phase-c40e/command.txt)
- [docs/phase-c40e/common-env.txt](docs/phase-c40e/common-env.txt)
- [docs/phase-c40e/leg-a.env](docs/phase-c40e/leg-a.env)
- [docs/phase-c40e/leg-b.env](docs/phase-c40e/leg-b.env)
- [docs/phase-c40e/kiln-bench.sha256](docs/phase-c40e/kiln-bench.sha256)
- [docs/phase-c40e/leg-a.exit](docs/phase-c40e/leg-a.exit)
- [docs/phase-c40e/leg-b.exit](docs/phase-c40e/leg-b.exit)
- [docs/phase-c40e/leg-a-w4a16.json](docs/phase-c40e/leg-a-w4a16.json)
- [docs/phase-c40e/leg-a-w4a16.log](docs/phase-c40e/leg-a-w4a16.log)
- [docs/phase-c40e/leg-b-bf16.json](docs/phase-c40e/leg-b-bf16.json)
- [docs/phase-c40e/leg-b-bf16.log](docs/phase-c40e/leg-b-bf16.log)

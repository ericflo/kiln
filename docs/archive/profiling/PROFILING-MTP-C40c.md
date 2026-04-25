# Phase C40c — HumanEval + BF16 (W4A16 off) N=20 α re-bench

## TL;DR

This run was **aborted before the BF16 sweep** because the required
default-path sanity check failed on current `origin/main`
(`8870dd838a1182ce7f9d484ec321007f2bdebae3`, PR #374 merged).

The task contract required the C39 anchor to reproduce
`α = 0.6933333` for:

- `KILN_W4A16=1`
- `KILN_MTP_ARGMAX_FP32=1`
- `KILN_SPEC_ENABLED=1`
- `KILN_SPEC_METHOD=mtp`
- `KILN_CUDA_GRAPHS=true`
- `--paged --chat-template --skip-training --latency-only`
- `--prompt-tokens 512 --max-output-tokens 128`
- `--prompt-subset humaneval`
- `--seed 0 --temperature 0.0`

Two consecutive runs on the same pod and same current-main commit did
not reproduce that anchor:

| run | observed α | expected α | Δ vs expected |
| --- | --- | --- | --- |
| sanity #1 | `0.6710526316` | `0.6933333333` | `-0.0222807018` |
| sanity #2 | `0.6623376623` | `0.6933333333` | `-0.0309956710` |

Because the default `W4A16=1` path was already off-anchor before any
`KILN_W4A16=0` change, the planned N=20 BF16 sweep was **not run**. Any
BF16-vs-W4A16 comparison would have been confounded by an unexplained
regression in the baseline path.

## Preflight

All pre-pod gates passed:

1. No existing C40c PR was in flight.
2. `origin/main:crates/kiln-server/src/bench.rs` includes the
   `--temperature` flag from PR #374.
3. `origin/main:PROFILING-MTP-C40b.md` ends with the "Remaining open
   questions" section listing C40c as item 1.
4. Static code reading confirms dense BF16 is the default weight-loader
   path and that MTP loading is independent of `KILN_W4A16`:
   - dense checkpoint load in
     `crates/kiln-model/src/loader.rs:75-91` and `:94-182`
   - MTP dense load in `crates/kiln-model/src/loader.rs:663-750`
   - `KILN_W4A16` only toggles Marlin packing during GPU upload in
     `crates/kiln-model/src/forward.rs:998-1199`
   - MTP bench requires loaded MTP weights in
     `crates/kiln-server/src/bench.rs:1404-1421`
5. The pod did not already have `/workspace/qwen3.5-4b`, but the public
   HF repo `Qwen/Qwen3.5-4B` was reachable and downloaded successfully
   to the pod in under 5 minutes.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-04-22 |
| Commit under test | `8870dd838a1182ce7f9d484ec321007f2bdebae3` |
| GPU | NVIDIA RTX A6000 |
| Pod | `sl53yvx5seviyx` |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Checkpoint source | HF `Qwen/Qwen3.5-4B` |
| Build | `cargo build --release --features cuda --bin kiln-bench` |

## Sanity Runs

Command used for both runs:

```bash
KILN_W4A16=1 \
KILN_MTP_ARGMAX_FP32=1 \
KILN_SPEC_ENABLED=1 \
KILN_SPEC_METHOD=mtp \
KILN_CUDA_GRAPHS=true \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed 0 --temperature 0.0
```

Observed metrics:

| run | α | draft accepted | attempts | decode tok/s | mean ITL ms | prefill tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| sanity #1 | `0.6710526316` | 51 | 76 | 44.81 | 22.32 | 68.48 |
| sanity #2 | `0.6623376623` | 51 | 77 | 35.46 | 28.20 | 1464.91 |

Notes:

- Both runs loaded the same two-shard dense checkpoint and reported
  "Native MTP head detected and loaded (k=1 draft depth)".
- Both runs used Marlin on the default path:
  `marlin batch pack: 104/104 projections`.
- The acceptance-rate miss is larger than trivial float noise and moves
  in the wrong direction on rerun.
- Prefill throughput differed sharply between the two sanity runs, but
  that does not explain the contract failure: both α values remain below
  the required `0.6933333` anchor.

## Why The BF16 Sweep Was Skipped

The C40c method required a byte-identity check for the unchanged
`KILN_W4A16=1` path before flipping the single knob to `KILN_W4A16=0`.
That gate failed twice on current main, so the experiment could no
longer isolate "W4A16 off" as the only changing variable.

Running the planned `KILN_W4A16=0` N=20 sweep in this state would not
answer the C40c hypothesis cleanly. A BF16 result could improve or
worsen for reasons unrelated to quantization if the anchor itself has
already drifted.

## Verdict

`C40c` is **blocked by baseline drift on current main**. The code-path
preflight says BF16 + MTP is supported, and the checkpoint is available,
but the unchanged `W4A16=1` sanity anchor no longer reproduces the C39 /
C40b expectation. The next task should diagnose why current `main`
misses `α = 0.6933333` on the seed-0 HumanEval anchor before resuming
the BF16-vs-W4A16 comparison.

## Artifacts

- Sanity JSON:
  `docs/phase-c40c/sanity_seed0_temp0.json`
- Sanity rerun JSON:
  `docs/phase-c40c/sanity_seed0_temp0_rerun.json`
- Sanity stderr log:
  `docs/phase-c40c/sanity_seed0_temp0.log`
- Sanity rerun stderr log:
  `docs/phase-c40c/sanity_seed0_temp0_rerun.log`

## Cost Summary

- Wall-clock: about 7 minutes from pod acquire to abort decision
- GPU cost: about `$0.06` at `$0.49/hr`
- Status: stopped well inside the 90 min / $25 cap

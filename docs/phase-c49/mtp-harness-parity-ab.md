# C49 MTP harness-parity A/B

Date: 2026-04-24

Commit: `d93d456024efc2db631fde175ffc8b618aed8a71`

GPU: RunPod on-demand NVIDIA RTX A6000, mandatory `ghcr.io/ericflo/kiln-runpod:latest` image.

## Purpose

C48 refreshed forced-MTP on current `origin/main`, but used a different harness shape than C40f. This run compares the C48-style command against the C40f-style prompt/decode flags on the same commit and same A6000 to isolate whether the C48 regression was mostly a harness/workload comparability artifact.

## Setup

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin
git reset --hard origin/main
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench
cargo test --locked -p kiln-server bench -- --nocapture || true
python3 -m py_compile scripts/mtp_compare.py scripts/mtp_h_main_reference_dump.py
```

`cargo test --locked -p kiln-server bench -- --nocapture` exited `0`; `python3 -m py_compile` also passed.

## Arms

Arm A, C48-style:

```bash
env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed <seed>
```

Arm B, C40f-style:

```bash
env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed <seed> \
  --chat-template --latency-only --prompt-subset humaneval --temperature 0.0
```

## Per-seed results

| Arm | Seed | Prompt subset | Prompt tokens | α | Decode tok/s | Mean ITL ms | Prefill ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| C48-style | 0 | all | 494 | 0.740 | 43.12 | 23.19 | 356.14 |
| C48-style | 1 | all | 508 | 0.309 | 26.46 | 37.79 | 352.72 |
| C48-style | 2 | all | 501 | 0.396 | 29.76 | 33.60 | 368.38 |
| C48-style | 3 | all | 496 | 0.270 | 25.38 | 39.40 | 350.26 |
| C48-style | 4 | all | 512 | 0.391 | 28.73 | 34.81 | 376.33 |
| C40f-style | 0 | humaneval | 496 | 0.789 | 46.42 | 21.54 | 349.54 |
| C40f-style | 1 | humaneval | 515 | 0.707 | 42.45 | 23.56 | 382.58 |
| C40f-style | 2 | humaneval | 510 | 0.588 | 37.36 | 26.77 | 350.43 |
| C40f-style | 3 | humaneval | 494 | 0.568 | 36.24 | 27.59 | 347.65 |
| C40f-style | 4 | humaneval | 502 | 0.740 | 44.46 | 22.49 | 347.39 |

## Summary

| Metric | C48-style median | C40f-style median | C40f-style mean | Paired median delta | Paired mean delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acceptance α | 0.391 | 0.707 | 0.678 | +0.298 | +0.257 |
| Decode tok/s | 28.73 | 42.45 | 41.39 | +10.86 | +10.70 |
| Mean ITL ms | 34.81 | 23.56 | 24.39 | -11.81 | -9.37 |
| Prefill ms | 356.14 | 349.54 | 355.52 | -5.71 | -5.25 |

C40f-style decode is `1.48x` the C48-style median. Against the C40f historical median of `38.25 tok/s`, the C40f-style arm is `1.11x`; it is well inside the required 10% band and actually faster on this run.

## Recommendation

The C40f-style flags restore median α to `0.707`, clearing the `0.65` gate, and restore median decode to `42.45 tok/s`, clearing the 10%-of-`38.25 tok/s` gate. Treat the C48 drop as primarily a harness/workload comparability artifact, not a model-math regression signal.

Resume Phase 6 MTP performance/profiling work from the C40f-style harness anchor. Do not reopen model-math investigation from C48 alone. If future C40f-style runs fall below the gate, the next boundary should be a single-factor split across the restored knobs in this order: `--prompt-subset humaneval`, `--chat-template`, `KILN_MTP_ARGMAX_FP32=1`, then `--temperature 0.0`.

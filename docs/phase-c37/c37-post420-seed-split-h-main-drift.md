# Phase C37 — post-#420 seed0-vs-seed1 `h_main` drift audit

**Date:** 2026-04-23  
**Current main:** `59ea3b5` (PR #419 merged; PR #420 already on `main`)  
**Hardware:** RunPod `NVIDIA A40` via `ghcr.io/ericflo/kiln-runpod:latest`  
**A6000 note:** direct A6000 allocation and pool resume both failed on 2026-04-23 with RunPod supply-constraint errors, so this rerun used the project-policy fallback order (`A6000 -> A40`) rather than stalling the task indefinitely.

## Preflight

1. **No newer open or merged PR already lands this artifact:** satisfied. Fresh `origin/main` was still `59ea3b5`, and recent PR history after #420 contains no post-#420 seed0-vs-seed1 `h_main` drift artifact for the standard native-MTP workload.
2. **Current main still shows the post-#420 seed split:** satisfied on a fresh rerun of the standard native-MTP bench. Seed 0 stayed near the paper floor while seed 1 stayed in the low-α regime.

## Commands run

Build:

```bash
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
cd /workspace/kiln
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
```

Per-seed standard workload capture:

```bash
for seed in 0 1; do
  rm -rf profiling-artifacts/post420_20260423_seed${seed}_captures
  mkdir -p \
    profiling-artifacts/post420_20260423_seed${seed}_captures/mtp_pos-0 \
    profiling-artifacts/post420_20260423_seed${seed}_captures/mtp_pos-2

  KILN_SPEC_METHOD=mtp \
  KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 \
  KILN_MTP_DUMP_SPLICE_POS=0,2 \
  KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 \
  KILN_MTP_DUMP_PATH=/workspace/kiln/profiling-artifacts/post420_20260423_seed${seed}_captures/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > profiling-artifacts/post420_20260423_seed${seed}.bench.json \
    2> profiling-artifacts/post420_20260423_seed${seed}.bench.stderr
done
```

Audit (existing tooling only):

```bash
python3 scripts/c15_h_main_drift_audit.py \
  --checkpoint /workspace/qwen3.5-4b \
  --captures-root /workspace/kiln/profiling-artifacts/post420_20260423_seed0_captures \
  --out /workspace/kiln/profiling-artifacts/post420_20260423_seed0_audit_manual

python3 scripts/c15_h_main_drift_audit.py \
  --checkpoint /workspace/qwen3.5-4b \
  --captures-root /workspace/kiln/profiling-artifacts/post420_20260423_seed0_captures \
  --out /workspace/kiln/profiling-artifacts/post420_20260423_seed0_audit_manual \
  --fp32

python3 scripts/c15_h_main_drift_audit.py \
  --checkpoint /workspace/qwen3.5-4b \
  --captures-root /workspace/kiln/profiling-artifacts/post420_20260423_seed1_captures \
  --out /workspace/kiln/profiling-artifacts/post420_20260423_seed1_audit_manual

python3 scripts/c15_h_main_drift_audit.py \
  --checkpoint /workspace/qwen3.5-4b \
  --captures-root /workspace/kiln/profiling-artifacts/post420_20260423_seed1_captures \
  --out /workspace/kiln/profiling-artifacts/post420_20260423_seed1_audit_manual \
  --fp32
```

The committed artifacts are the FP32 audit outputs copied verbatim from the existing script:

- [`profiling-artifacts/post420_20260423_hmain_seed0.json`](../../profiling-artifacts/post420_20260423_hmain_seed0.json)
- [`profiling-artifacts/post420_20260423_hmain_seed1.json`](../../profiling-artifacts/post420_20260423_hmain_seed1.json)

## Fresh seed split

| Seed | blocks | actual prompt tokens | mean ITL ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 47 | 494 | 24.9 | 40.1 | 0.740 |
| 1 | 48 | 508 | 43.7 | 22.9 | 0.309 |

This reproduces the post-#420 split on fresh current `main`: seed 0 escapes, seed 1 stays trapped.

## Audit summary at `mtp_pos=0`

FP32 is the source-of-truth verdict here because C15 already established that BF16-only thresholds can overstate harmless noise. BF16 is listed only as a corroborating cross-check.

| Seed | FP32 status | FP32 min cos | FP32 last-step cos | FP32 drift ratio | BF16 min cos |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `clean` | 0.999067 | 0.999675 | 0.348x | 0.998639 |
| 1 | `drift` | 0.990691 | 0.998943 | 3.425x | 0.985204 |

### Readout

- **Seed 0:** the escaped case stays effectively aligned in the base hidden-state path. The FP32 audit is `clean` at `mtp_pos=0`, with every captured step above the strict `0.999` bar and the drift ratio shrinking across steps.
- **Seed 1:** the trapped case shows a real late-step `h_main` divergence at `mtp_pos=0`. FP32 cosine falls from `0.999691` at step 0 to `0.990691` at step 6, and the drift ratio grows `3.425x` from step 0 to step 7.

## Pos-2 caveat

The existing `scripts/c15_h_main_drift_audit.py` output for `mtp_pos=2` is not usable on these per-seed roots. It builds a chained sequence of length 4 from the isolated `pos=2` files, then indexes with `base_pos` values around `502` / `520`, which produces the committed `index out of range` errors in both seeds. I did not add new instrumentation or patch the script in this task. The honest source-of-truth verdict is therefore limited to `mtp_pos=0`, which is also the critical path for distinguishing the escaping seed from the trapped one.

## Verdict

**Yes: the post-#420 seed split is already visible in the base-model hidden-state path at `mtp_pos=0`.** The escaping seed 0 does **not** show materially growing `h_main` drift under the FP32 audit, while seed 1 does. That means the current split is not solely a downstream accept/reject artifact layered on top of identical `h_main` quality; by step 5-6 the low-α seed is feeding the MTP head a measurably worse base hidden state than the escaping seed.

The next recommendation should stay on the base-stack / `h_main` production side for the low-α branch, not on another decode-kernel retry. The narrow follow-up is to bisect the first seed1-only divergence upstream of the `h_main` tap on the standard workload, using the existing hidden-state dump path and a layer/sub-op comparator, rather than reopening speculative acceptance math in isolation.

## Validation

- `python3 -m py_compile scripts/c15_h_main_drift_audit.py`
- Fresh RunPod GPU rerun of the standard native-MTP workload for seeds 0 and 1 with splice dumps enabled
- Re-copied the script outputs and verified the committed JSON files match the generated FP32 audit JSON contents (newline-normalized locally)

# Phase C39 — full replay context for post-#423 seed bisect

**Date:** 2026-04-23  
**Branch under test:** `ce/phase-c39-full-replay-context` (`906666e` before artifacts)  
**Hardware:** RunPod `NVIDIA RTX A6000` using `ghcr.io/ericflo/kiln-runpod:latest`

## Goal

Fix the post-#423 replay contract so every B10/B11/B12 dump carries the full
conditioned token sequence needed for an honest HF replay, then rerun the
standard native-MTP seed-0 / seed-1 investigation on fresh `main`.

## Validation

Local:

```bash
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
```

Pod:

```bash
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
cargo build --release --features cuda --bin kiln-bench
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
python3 -m pip install --upgrade numpy transformers sentencepiece safetensors
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
```

Fresh workload rerun and compare generation:

```bash
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_METHOD=mtp \
KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 \
KILN_MTP_DUMP_SPLICE_POS=0,2 \
KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 \
KILN_MTP_DUMP_B11_TAPS=1 \
KILN_MTP_DUMP_B12_GQA_TAPS=1 \
KILN_MTP_DUMP_PATH=/workspace/c39-work/seed${seed}_captures/mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed ${seed}

python3 scripts/mtp_h_main_reference_dump.py --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump <dump> --out <ref> --device cuda --b11-taps --b12-taps
python3 scripts/mtp_h_main_reference_dump.py --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump <dump> --out <ref-fp32> --device cuda --b11-taps --b12-taps --fp32

python3 scripts/mtp_compare.py --b10 ...
python3 scripts/mtp_compare.py --b11 ...
python3 scripts/mtp_compare.py --b12 ...
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 445.2 | 36.88 | 0.7067 |
| 1 | 508 | 453.1 | 24.03 | 0.2929 |

The seed split still reproduces on the fixed replay contract.

## Replay-contract result

The dump fix worked. Every captured row now carries the full conditioned
sequence length instead of collapsing to the local verifier slice:

- seed 0: `494 -> 495 -> 496 -> 497`, then `502 -> 503 -> 504` at `mtp_pos=2`
- seed 1: `508 -> 509 -> 510 -> 511 -> 512 -> 513 -> 514 -> 515`, then
  `520 -> 521 -> 522` at `mtp_pos=2`

Source of truth: [replay_lengths.json](./replay_lengths.json)

## Verdict

**The replay bug from PR #423 is fixed, but it does not uncover a seed-1-only
first upstream boundary or sub-op.**

With fully conditioned HF replays, both seeds land on the same comparator
verdicts in both bf16 and fp32:

- B10: first diverging boundary tap is `h_layer_8` at `0_s0`
- B11b: all layer-0 GDN sub-ops stay within `cos_sim >= 0.95`
- B12: late-stack drift first appears at `h_layer_24`, not layer 31

Artifacts:

- [seed0_compare_bf16.txt](./seed0_compare_bf16.txt)
- [seed0_compare_fp32.txt](./seed0_compare_fp32.txt)
- [seed1_compare_bf16.txt](./seed1_compare_bf16.txt)
- [seed1_compare_fp32.txt](./seed1_compare_fp32.txt)
- [seed0.bench.json](./seed0.bench.json)
- [seed1.bench.json](./seed1.bench.json)

## Recommendation

Do not frame the next task as "find the first seed-1-only upstream `h_main`
boundary" anymore; C39 refutes that route on the current build. The next
narrow task should either:

1. bisect the shared early-GDN drift between `h_layer_0` and `h_layer_8`, or
2. treat the seed split as downstream of this shared `h_main` boundary and
   move the seed-specific audit back into accept/reject dynamics rather than
   another replay-contract pass.

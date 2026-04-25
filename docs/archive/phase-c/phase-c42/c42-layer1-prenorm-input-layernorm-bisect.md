# Phase C42 — layer 1 pre-norm vs input-layernorm bisect after PR #428

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


**Date:** 2026-04-23  
**Current main preflight base:** `e569d22` (PR #428 on `main`)  
**Hardware:** RunPod `NVIDIA A100-SXM4-80GB` via `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

1. **Fresh-main source of truth still valid:** satisfied. Fresh `origin/main`
   still pointed at PR #428 and
   [`docs/archive/phase-c/phase-c41/c41-layer1-subop-bisect.md`](../phase-c41/c41-layer1-subop-bisect.md)
   remained the latest committed source of truth for this bisect path.
2. **Committed C41 verdict still held:** satisfied. C41 still recorded
   `layer_1_post_input_norm` as the earliest shared bad layer-1 tap for both
   seeds.
3. **No committed C42 localization already landed:** satisfied. Fresh main did
   not already contain committed C42 taps or a doc that localized the
   remaining span to either the residual input entering block 1 or the
   layer-1 `input_layernorm` numerics.

## Code change

This task adds a minimal, opt-in C42 bisect path for the layer-1 norm
boundary only:

- `crates/kiln-model/src/mtp_debug.rs`
  adds the dedicated `KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS=1` switch, the
  canonical C42 tap list, a dedicated thread-local sink, and metadata
  serialization via `meta__c42_tap_ids`.
- `crates/kiln-model/src/forward.rs`
  arms C42 only on the base-model replay path and captures the explicit
  norm-boundary tap set at layer 1:
  `layer_1_residual_input`, `layer_1_input_norm_rms_inv`,
  `layer_1_input_norm_pre_weight`, and `layer_1_post_input_norm`.
- `scripts/mtp_h_main_reference_dump.py`
  mirrors the same tap set under `--c42-taps`, resolves tap ordering from kiln
  metadata, and captures the matching HF-side layer-1 input-layernorm
  intermediates on the same last-token replay contract.
- `scripts/mtp_compare.py`
  adds `--c42` mode and narrows reporting to the explicit `c42__*` tap set so
  the earliest shared bad norm-boundary tensor is reported cleanly.

Default decode remains unchanged when
`KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS` is unset.

## Validation commands run

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

## Standard workload commands run

Build and checkpoint setup on the RunPod fallback GPU:

```bash
cd /workspace/kiln
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=80
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 -m pip install transformers safetensors sentencepiece
```

Standard native-MTP rerun with the C42 tap flag enabled:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}_captures
  rm -rf "$root"
  mkdir -p "$root"/mtp_pos-0 "$root"/mtp_pos-2

  KILN_W4A16=1 \
  KILN_CUDA_GRAPHS=true \
  KILN_SPEC_METHOD=mtp \
  KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 \
  KILN_MTP_DUMP_SPLICE_POS=0,2 \
  KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 \
  KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}.bench.stderr
done
```

Representative C41 divergence points were then re-referenced:

- seed 0 representative dump:
  `profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1 representative dump:
  `profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference regeneration and compare commands:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post428_c42_20260423_seed0_ref_bf16.safetensors \
  --device cuda \
  --c42-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post428_c42_20260423_seed0_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c42-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post428_c42_20260423_seed1_ref_bf16.safetensors \
  --device cuda \
  --c42-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post428_c42_20260423_seed1_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c42-taps

python3 scripts/mtp_compare.py --c42 \
  --pair seed0:profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed1_ref_bf16.safetensors \
  --out profiling-artifacts/post428_c42_20260423_compare_bf16.txt

python3 scripts/mtp_compare.py --c42 \
  --pair seed0:profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed0_ref_fp32.safetensors \
  --pair seed1:profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed1_ref_fp32.safetensors \
  --out profiling-artifacts/post428_c42_20260423_compare_fp32.txt
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10116.0 | 35.3 | 0.716 |
| 1 | 508 | 395.3 | 21.2 | 0.293 |

The post-#428 seed split still reproduces.

## Committed artifacts

- [`profiling-artifacts/post428_c42_20260423_seed0.bench.json`](../../profiling-artifacts/post428_c42_20260423_seed0.bench.json)
- [`profiling-artifacts/post428_c42_20260423_seed0.bench.stderr`](../../profiling-artifacts/post428_c42_20260423_seed0.bench.stderr)
- [`profiling-artifacts/post428_c42_20260423_seed1.bench.json`](../../profiling-artifacts/post428_c42_20260423_seed1.bench.json)
- [`profiling-artifacts/post428_c42_20260423_seed1.bench.stderr`](../../profiling-artifacts/post428_c42_20260423_seed1.bench.stderr)
- [`profiling-artifacts/post428_c42_20260423_compare_bf16.txt`](../../profiling-artifacts/post428_c42_20260423_compare_bf16.txt)
- [`profiling-artifacts/post428_c42_20260423_compare_fp32.txt`](../../profiling-artifacts/post428_c42_20260423_compare_fp32.txt)

## Verdict

**The earliest shared bad layer-1 norm-boundary tap is
`layer_1_input_norm_pre_weight`.**

Across both bf16 and fp32 reference comparisons:

| Reference dtype | Seed 0 verdict | Seed 1 verdict | Shared earliest bad tap |
| --- | --- | --- | --- |
| bf16 | `layer_1_input_norm_pre_weight` | `layer_1_input_norm_pre_weight` | `layer_1_input_norm_pre_weight` |
| fp32 | `layer_1_input_norm_pre_weight` | `layer_1_input_norm_pre_weight` | `layer_1_input_norm_pre_weight` |

Representative C42 rows:

- seed 0 bf16:
  `layer_1_residual_input` remains `ok`,
  `layer_1_input_norm_rms_inv` remains `ok`,
  and `layer_1_input_norm_pre_weight` is already `DIV`
- seed 1 fp32:
  `layer_1_residual_input` remains `ok`,
  `layer_1_input_norm_rms_inv` remains `ok`,
  and `layer_1_input_norm_pre_weight` is already `DIV`

This closes the C41 remaining span. The first shared bad tensor is **not**
already present in the residual entering transformer block 1; it appears
inside layer 1 `input_layernorm`.

## Narrowest remaining span

C42 localizes the remaining bad span to the norm-local step between:

- `layer_1_input_norm_rms_inv` (last shared-good tap)
- `layer_1_input_norm_pre_weight` (earliest shared-bad tap)

So the residual input entering block 1 is no longer a viable culprit. The
remaining work, if any, should stay inside layer-1 input-layernorm numerics:
the normalization / pre-weight scaling path before the final `(1 + weight)`
application.

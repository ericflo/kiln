# Phase C41 — transformer block 1 sub-op bisect after PR #427

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


**Date:** 2026-04-23  
**Current main preflight base:** `5b79676` (PR #427 on `main`)  
**Hardware:** RunPod `NVIDIA A40` via `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

1. **Fresh-main source of truth still valid:** satisfied. Fresh `origin/main`
   still pointed at PR #427 and
   [`docs/archive/phase-c/phase-c40/c40-early-hmain-bisect.md`](../phase-c40/c40-early-hmain-bisect.md)
   remained the latest committed source of truth for this bisect path.
2. **Committed C40 verdict still held:** satisfied. C40 still recorded
   `h_layer_0` as matching and `h_layer_1` as the first shared bad layer for
   both seeds.
3. **No committed layer-1 sub-op localization already landed:** satisfied.
   Fresh main did not already contain committed C41 taps or a localized
   transformer-block-1 earliest-bad boundary.

## Code change

This task adds a minimal, opt-in C41 bisect path for transformer block 1:

- `crates/kiln-model/src/mtp_debug.rs`
  adds the dedicated `KILN_MTP_DUMP_C41_LAYER1_TAPS=1` capture switch, the
  canonical C41 tap list, thread-local capture plumbing, and safetensors
  metadata serialization via `meta__c41_tap_ids`.
- `crates/kiln-model/src/forward.rs`
  arms C41 only on the base-model replay path and captures the explicit
  transformer-block-1 boundaries:
  `layer_1_post_input_norm`, GDN in-proj / conv / qk-norm / gates /
  recurrent output / gated norm / out-proj, `layer_1_post_attn_residual`, and
  `layer_1_output`.
- `scripts/mtp_h_main_reference_dump.py`
  mirrors the same boundary set under `--c41-taps`, resolves tap ordering from
  kiln metadata, and slices the captured HF tensors to the final replay token
  so the reference dump matches kiln's `[1, 1, ...]` shape contract.
- `scripts/mtp_compare.py`
  adds `--c41` mode and narrows the per-pair / summary report to the explicit
  `c41__*` tap set so the earliest shared bad layer-1 boundary is reported
  cleanly.

Default decode remains unchanged when `KILN_MTP_DUMP_C41_LAYER1_TAPS` is
unset.

## Validation commands run

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

## Standard workload commands run

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post427_c41_20260423_seed${seed}_captures
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
  KILN_MTP_DUMP_C41_LAYER1_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post427_c41_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post427_c41_20260423_seed${seed}.bench.stderr
done
```

Representative C40 divergence points were then re-referenced:

- seed 0 representative dump:
  `profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1 representative dump:
  `profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference regeneration and compare commands:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed0_ref_bf16.safetensors \
  --device cuda \
  --c41-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed0_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c41-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed1_ref_bf16.safetensors \
  --device cuda \
  --c41-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed1_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c41-taps

python3 scripts/mtp_compare.py --c41 \
  --pair seed0:profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed1_ref_bf16.safetensors \
  --out profiling-artifacts/post427_c41_20260423_compare_bf16.txt

python3 scripts/mtp_compare.py --c41 \
  --pair seed0:profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed0_ref_fp32.safetensors \
  --pair seed1:profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed1_ref_fp32.safetensors \
  --out profiling-artifacts/post427_c41_20260423_compare_fp32.txt
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8900.8 | 39.0 | 0.740 |
| 1 | 508 | 479.8 | 42.5 | 0.283 |

The post-#427 seed split still reproduces.

## Committed artifacts

- [`profiling-artifacts/post427_c41_20260423_seed0.bench.json`](../../profiling-artifacts/post427_c41_20260423_seed0.bench.json)
- [`profiling-artifacts/post427_c41_20260423_seed0.bench.stderr`](../../profiling-artifacts/post427_c41_20260423_seed0.bench.stderr)
- [`profiling-artifacts/post427_c41_20260423_seed1.bench.json`](../../profiling-artifacts/post427_c41_20260423_seed1.bench.json)
- [`profiling-artifacts/post427_c41_20260423_seed1.bench.stderr`](../../profiling-artifacts/post427_c41_20260423_seed1.bench.stderr)
- [`profiling-artifacts/post427_c41_20260423_compare_bf16.txt`](../../profiling-artifacts/post427_c41_20260423_compare_bf16.txt)
- [`profiling-artifacts/post427_c41_20260423_compare_fp32.txt`](../../profiling-artifacts/post427_c41_20260423_compare_fp32.txt)

## Verdict

**The earliest shared bad layer-1 sub-op boundary is
`layer_1_post_input_norm`.**

Across both bf16 and fp32 reference comparisons:

| Reference dtype | Seed 0 verdict | Seed 1 verdict | Shared earliest bad tap |
| --- | --- | --- | --- |
| bf16 | `layer_1_post_input_norm` | `layer_1_post_input_norm` | `layer_1_post_input_norm` |
| fp32 | `layer_1_post_input_norm` | `layer_1_post_input_norm` | `layer_1_post_input_norm` |

Representative C41 rows:

- seed 0 bf16:
  `layer_1_post_input_norm` is already `DIV`, `gdn_gate_beta` and
  `gdn_recur_out` remain `ok`, and `layer_1_post_attn_residual` returns to
  `ok`
- seed 1 fp32:
  `layer_1_post_input_norm` is already `DIV`, `gdn_conv` remains `ok`,
  `gdn_gate_beta` remains `ok`, and `layer_1_post_attn_residual` remains `ok`

This is enough to close the C40 task premise. The first observed bad tensor is
not “somewhere inside transformer block 1” anymore; it is already present at
the output of layer 1 input_layernorm.

## Narrowest remaining span

Because the earliest bad tap is the first explicit C41 boundary, the remaining
uncaptured span is narrower than the GDN internals:

- the residual input entering transformer block 1
- the layer-1 `input_layernorm` application itself

If there is a follow-up, it should instrument the pre-norm residual input or
audit the layer-1 input-layernorm numerics directly. It should **not** widen
the tracing deeper into later layer-1 sub-ops again.

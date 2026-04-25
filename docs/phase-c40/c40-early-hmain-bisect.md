# Phase C40 — dense early h_main sweep after PR #426

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


**Date:** 2026-04-23  
**Current main:** `4df936c` (PR #426 on `main`)  
**Hardware:** RunPod `NVIDIA RTX A6000` via `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

1. **Fresh-main baseline:** satisfied. Work started from fresh `origin/main`
   at `4df936c`, the merge commit for PR #426.
2. **No committed dense early sweep already present:** satisfied. Fresh main
   still only localized the shared upstream drift as `h_layer_0 -> h_layer_8`.
3. **No open PR already covering this exact sweep:** satisfied. No open PR
   landed a C40-style early-layer artifact.

## Code change

The h_main capture path now supports an opt-in dense early sweep with
`KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1`. When that flag is set:

- kiln emits `h_layer_1..8` in addition to the historical B10 boundary taps
- kiln serializes the actual emitted layer order as `meta__boundary_layers`
- `scripts/mtp_h_main_reference_dump.py` mirrors the same boundary set from
  the dump metadata
- `scripts/mtp_compare.py --b10` derives the verdict from the real emitted
  sequence and can now report the exact earliest layer when the last matching
  tap and first diverging tap are adjacent

Default B10/B12 behavior is unchanged when the new flag is unset.

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
  root=/workspace/kiln/profiling-artifacts/post426_c40_20260423_seed${seed}_captures
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
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post426_c40_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post426_c40_20260423_seed${seed}.bench.stderr
done
```

Reference regeneration and comparison:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump <dump> \
  --out <bf16-ref> \
  --device cuda

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump <dump> \
  --out <fp32-ref> \
  --device cuda \
  --fp32

python3 scripts/mtp_compare.py --b10 --pair <label>:<kiln>,<ref>
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8795.4 | 39.57 | 0.6842 |
| 1 | 508 | 386.8 | 23.95 | 0.2959 |

The seed split still reproduces on fresh `main`.

## Committed artifacts

- [`profiling-artifacts/post426_c40_20260423_seed0.bench.json`](../../profiling-artifacts/post426_c40_20260423_seed0.bench.json)
- [`profiling-artifacts/post426_c40_20260423_seed0.bench.stderr`](../../profiling-artifacts/post426_c40_20260423_seed0.bench.stderr)
- [`profiling-artifacts/post426_c40_20260423_seed1.bench.json`](../../profiling-artifacts/post426_c40_20260423_seed1.bench.json)
- [`profiling-artifacts/post426_c40_20260423_seed1.bench.stderr`](../../profiling-artifacts/post426_c40_20260423_seed1.bench.stderr)
- [`profiling-artifacts/post426_c40_20260423_seed0_compare_bf16.txt`](../../profiling-artifacts/post426_c40_20260423_seed0_compare_bf16.txt)
- [`profiling-artifacts/post426_c40_20260423_seed0_compare_fp32.txt`](../../profiling-artifacts/post426_c40_20260423_seed0_compare_fp32.txt)
- [`profiling-artifacts/post426_c40_20260423_seed1_compare_bf16.txt`](../../profiling-artifacts/post426_c40_20260423_seed1_compare_bf16.txt)
- [`profiling-artifacts/post426_c40_20260423_seed1_compare_fp32.txt`](../../profiling-artifacts/post426_c40_20260423_seed1_compare_fp32.txt)

## Verdict

**The dense early sweep localizes the first shared upstream drift to layer 1
for both seeds.**

Across both bf16 and fp32 HF references:

| Seed | First diverging tap | Position label | Last matching tap | Verdict |
| --- | --- | --- | --- | --- |
| 0 | `h_layer_1` | `mtp_pos-0-step-1` | `h_layer_0` | `DIVERGENCE FIRST APPEARS AT LAYER 1` |
| 1 | `h_layer_1` | `mtp_pos-2-step-1` | `h_layer_0` | same |

Representative bf16 rows from the committed compare outputs:

- seed 0: `h_layer_0` stays `ok` across all captured rows, while `h_layer_1`
  is already `DIV` by `mtp_pos-0-step-1`
- seed 1: `h_layer_0` stays `ok` across all captured rows, while `h_layer_1`
  first drops below tolerance at `mtp_pos-2-step-1`

That means C39's coarse `h_layer_0 -> h_layer_8` span is now closed. The
shared drift does **not** start at layer 8; it starts immediately after
transformer block 1.

## Next narrower span

Because both seeds share the same earliest bad layer, the remaining span is no
longer per-layer. It is **inside transformer block 1**:

- layer-1 input norm output
- layer-1 GDN in-proj / conv / qk-norm / gates / recurrent kernel
- layer-1 gated norm / out-proj / residual handoff

The next task should capture layer-1 sub-op taps on the same replay contract
instead of extending the boundary sweep again.

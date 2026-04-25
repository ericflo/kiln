# Phase C38 — post-#422 seed1 upstream base-stack bisect

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


**Date:** 2026-04-23  
**Current main:** `c66cf41` (PR #422 on `main`)  
**Hardware:** RunPod `NVIDIA A100 PCIe` via `ghcr.io/ericflo/kiln-runpod:latest`  
**A6000 note:** `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'` failed because the hibernated pool pod could not resume (`There are not enough free GPUs on the host machine to start this pod`). Direct A6000 launch returned `SUPPLY_CONSTRAINT`, and direct A40 launch returned a RunPod internal error. Per project fallback order, the fresh rerun used `A100 PCIe`.

## Preflight

1. **No newer PR already lands this artifact:** satisfied. Fresh `origin/main` is `c66cf41` (PR #422), and recent PR history contains no newer open or merged PR that already commits a post-#422 seed1-only upstream base-stack bisect artifact for the standard native-MTP workload.
2. **Current `main` still reproduces the seed split:** satisfied on a fresh standard-workload rerun. Seed 0 stayed near the paper floor while seed 1 remained trapped in the low-α regime.
3. **Current `PROFILING.md` still recommends an upstream `h_main` bisect, not another decode-kernel retry:** satisfied. The Phase C37 section still says the next narrow task should bisect the first seed1-only divergence upstream of the `h_main` tap.

## Commands run

Build and model download:

```bash
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=80
cd /workspace/kiln
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 -m pip install --upgrade numpy transformers sentencepiece safetensors
```

Fresh per-seed capture on the standard workload:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post422_20260423_seed${seed}_captures
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
  KILN_MTP_DUMP_B11_TAPS=1 \
  KILN_MTP_DUMP_B12_GQA_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post422_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post422_20260423_seed${seed}.bench.stderr
done
```

Matched HF reference dumps with the existing script:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post422_20260423_seed${seed}_captures
  while IFS= read -r dump; do
    rel=${dump#${root}/}
    out_bf16=/workspace/kiln/profiling-artifacts/post422_20260423_refs_bf16/seed${seed}/${rel}
    out_fp32=/workspace/kiln/profiling-artifacts/post422_20260423_refs_fp32/seed${seed}/${rel}
    mkdir -p "$(dirname "$out_bf16")" "$(dirname "$out_fp32")"

    python3 scripts/mtp_h_main_reference_dump.py \
      --checkpoint /workspace/qwen3.5-4b \
      --kiln-dump "$dump" \
      --out "$out_bf16" \
      --device cuda \
      --b11-taps \
      --b12-taps

    python3 scripts/mtp_h_main_reference_dump.py \
      --checkpoint /workspace/qwen3.5-4b \
      --kiln-dump "$dump" \
      --out "$out_fp32" \
      --device cuda \
      --b11-taps \
      --b12-taps \
      --fp32
  done < <(find "$root" -type f -name '*.safetensors' | sort)
done
```

Comparator pass (mechanically reproduced for both seeds and both reference dtypes):

```bash
python3 scripts/mtp_compare.py --b10 ...
python3 scripts/mtp_compare.py --b11 ...
python3 scripts/mtp_compare.py --b12 ...
```

Committed comparator outputs:

- [`profiling-artifacts/post422_20260423_seed0_compare_bf16.txt`](../../profiling-artifacts/post422_20260423_seed0_compare_bf16.txt)
- [`profiling-artifacts/post422_20260423_seed1_compare_bf16.txt`](../../profiling-artifacts/post422_20260423_seed1_compare_bf16.txt)
- [`profiling-artifacts/post422_20260423_seed0_compare_fp32.txt`](../../profiling-artifacts/post422_20260423_seed0_compare_fp32.txt)
- [`profiling-artifacts/post422_20260423_seed1_compare_fp32.txt`](../../profiling-artifacts/post422_20260423_seed1_compare_fp32.txt)

Supporting bench artifacts:

- [`profiling-artifacts/post422_20260423_seed0.bench.json`](../../profiling-artifacts/post422_20260423_seed0.bench.json)
- [`profiling-artifacts/post422_20260423_seed1.bench.json`](../../profiling-artifacts/post422_20260423_seed1.bench.json)

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 5602.8 | 49.98 | 0.7162 |
| 1 | 508 | 377.2 | 30.82 | 0.2828 |

Condition 2 still holds on fresh `main`: seed 0 escapes, seed 1 does not.

## What the comparator outputs actually prove

The existing B10/B11/B12 path is only trustworthy on the **first** `mtp_pos=0` dump (`0_s0`) for each seed.

Why:

- `mtp_h_main_reference_dump.py` forwards the `prompt_tokens` tensor embedded in each kiln dump.
- On the standard splice dumps, only the first `mtp_pos=0` capture carries the full prompt (`494` tokens for seed 0, `508` for seed 1).
- Every later `mtp_pos=0` capture carries `prompt_tokens` length `1`, and the `mtp_pos=2` captures carry lengths `2` then `1`.
- So the comparator's later rows are comparing a late-step kiln hidden state against an HF forward on an under-conditioned 1-token or 2-token sequence. That is not the same absolute position and cannot localize a seed-specific upstream bug.

This is visible directly in the committed reports:

- seed 0 fp32: `0_s0` has `prompt_tokens 494`, then `0_s1/0_s2/0_s3/2_s1/2_s2` all drop to `1`, with `2_s0` at `2`
- seed 1 fp32: `0_s0` has `prompt_tokens 508`, then every later `0_s*` row drops to `1`, with `2_s0` at `2`

That makes the later B10/B11/B12 medians **mechanically generated but not source-of-truth** for this task.

## Valid step-0 readout

On the only fully-contexted comparison (`mtp_pos=0`, `step=0`), seed 0 and seed 1 do **not** separate cleanly:

| Surface | Seed 0 fp32 (`0_s0`) | Seed 1 fp32 (`0_s0`) | Seed-specific? |
| --- | ---: | ---: | --- |
| `h_layer_0` | `1.00` | `1.00` | no |
| `h_layer_24` | `0.985` | `0.976` | no clear split |
| `h_layer_31` | `0.958` | `0.957` | no |
| B11 `gdn_conv` | `1.00` | `1.00` | no |

The same story holds in bf16: the control and failing seed are materially similar at the only valid comparator row.

## Mechanical full-report outputs

If you ignore the context problem and let the existing comparators consume every splice file verbatim, they produce the same nominal localization for **both** seeds:

- B10: `DIVERGENCE AT LAYER 0`
- B11: first sub-op below the bar is `gdn_conv`
- B12: first below-bar late-stack layer is `h_layer_24`

Those are not seed1-only findings. They are the product of feeding the same under-conditioned later-step inputs into the HF reference path on both seeds.

## Verdict

**The existing B10/B11/B12 dump/reference/comparator path is structurally insufficient to localize a first seed1-only upstream divergence after PR #422.**

Fresh `main` still reproduces the seed split, and the current handoff is still correctly pointed upstream of `h_main`, but the existing reference dumper only has one valid fully-contexted row per seed (`0_s0`). That valid row does **not** expose a seed1-only first divergent layer or sub-op. Everything later in the comparator reports is contaminated by incomplete history (`prompt_tokens` length `1` or `2`) and cannot be used as a source-of-truth localization verdict.

## Next recommendation

Do **not** queue a kernel fix from this task.

The next narrow task should keep the same B10/B11/B12 comparator stack but fix the input contract first, either by:

1. serializing the full chained prompt history into every splice dump, or
2. teaching the HF reference side to reconstruct the chained history C15-style before re-running the existing comparators.

Until one of those happens, the honest result is: **seed 1 is still the failing branch, but the current base-stack comparator path cannot isolate its first upstream-only divergence beyond step 0.**

## Validation

- `python3 -m py_compile scripts/mtp_compare.py scripts/mtp_h_main_reference_dump.py scripts/c15_h_main_drift_audit.py`
- fresh RunPod rerun on `c66cf41` of the standard native-MTP workload for seeds `0` and `1`
- fresh kiln hidden-state dumps with `KILN_MTP_DUMP_HIDDEN_STATES=1`, `KILN_MTP_DUMP_B11_TAPS=1`, and `KILN_MTP_DUMP_B12_GQA_TAPS=1`
- fresh HF reference dumps from `scripts/mtp_h_main_reference_dump.py` in both bf16 and fp32
- fresh B10/B11/B12 comparator runs for both seeds against both reference dtypes


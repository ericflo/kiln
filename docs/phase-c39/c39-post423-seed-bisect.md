# Phase C39 — post-#423 replay-context fix + fresh seed bisect

**Date:** 2026-04-23  
**Current main:** `61badb6` (PR #423 on `main`)  
**Hardware:** RunPod `NVIDIA A40` via `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

1. **Fresh-main baseline:** satisfied. Work started from fresh `origin/main`
   at `61badb6`, the merge commit for PR #423.
2. **No existing post-#423 artifact already closes this gap:** satisfied.
   Fresh `main` still lacked a full per-step replay contract for later
   B10/B11/B12 rows.
3. **Only fix the missing replay contract if it is still missing:** satisfied.
   An initial rerun on the unchanged bench path still emitted later
   `replay_tokens_len=1` / `2`, proving the under-conditioned replay problem
   was still present outside the kiln-model generation loop.

## Code change

The dump contract now carries `replay_tokens` plus
`meta__replay_tokens_len`, and the native-MTP bench loop seeds the per-step
replay prefix before each `speculative_mtp_decode_step`.

That change matters because the kiln dump still records the immediate
`prompt_tokens` slice (`1` token at most later in the run), but the HF
reference path now has the **fully conditioned sequence** needed to replay
the exact absolute position seen at each dumped step.

## Validation commands run

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py scripts/c15_h_main_drift_audit.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --features cuda --lib -- --nocapture
```

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post423_20260423_seed${seed}_captures
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
    > /workspace/kiln/profiling-artifacts/post423_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post423_20260423_seed${seed}.bench.stderr
done
```

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post423_20260423_seed${seed}_captures
  while IFS= read -r dump; do
    rel=${dump#${root}/}
    out_bf16=/workspace/kiln/profiling-artifacts/post423_20260423_refs_bf16/seed${seed}/${rel}
    out_fp32=/workspace/kiln/profiling-artifacts/post423_20260423_refs_fp32/seed${seed}/${rel}
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

Comparator pass:

```bash
python3 scripts/mtp_compare.py --b10 ...
python3 scripts/mtp_compare.py --b11 ...
python3 scripts/mtp_compare.py --b12 ...
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8734.0 | 36.13 | 0.7162 |
| 1 | 508 | 452.6 | 22.92 | 0.2828 |

The seed split still reproduces on fresh `main`.

## Replay-contract proof

The updated dump path now emits full replay lengths on later rows instead of
the previous 1-token / 2-token under-conditioned slices:

- seed 0: `step-1 replay_tokens_len=495`, `step-2=496`, `mtp_pos=2 step-0=502`
- seed 1: `step-1 replay_tokens_len=509`, `step-7=515`, `mtp_pos=2 step-0=520`

The evidence is in:

- [`profiling-artifacts/post423_20260423_seed0.bench.stderr`](../../profiling-artifacts/post423_20260423_seed0.bench.stderr)
- [`profiling-artifacts/post423_20260423_seed1.bench.stderr`](../../profiling-artifacts/post423_20260423_seed1.bench.stderr)

## Refreshed compare outputs

- [`profiling-artifacts/post423_20260423_seed0_compare_bf16.txt`](../../profiling-artifacts/post423_20260423_seed0_compare_bf16.txt)
- [`profiling-artifacts/post423_20260423_seed0_compare_fp32.txt`](../../profiling-artifacts/post423_20260423_seed0_compare_fp32.txt)
- [`profiling-artifacts/post423_20260423_seed1_compare_bf16.txt`](../../profiling-artifacts/post423_20260423_seed1_compare_bf16.txt)
- [`profiling-artifacts/post423_20260423_seed1_compare_fp32.txt`](../../profiling-artifacts/post423_20260423_seed1_compare_fp32.txt)

Across both bf16 and fp32 HF references, the refreshed B10/B11/B12 verdicts
are the same for seed 0 and seed 1:

| Surface | Seed 0 | Seed 1 |
| --- | --- | --- |
| B10 first diverging tap | `h_layer_8` | `h_layer_8` |
| B10 verdict | `DIVERGENCE IN EARLY GDN STACK` | `DIVERGENCE IN EARLY GDN STACK` |
| B11 verdict | `ALL layer-0 GDN sub-ops match within cos_sim >= 0.95` | same |
| B12 first below-bar layer | `h_layer_24` | `h_layer_24` |
| B12 verdict | `DRIFT FIRST APPEARS AT 'h_layer_24' (not layer 31)` | same |

Representative fp32 medians:

- seed 0 B12 first below-bar layer: `h_layer_24` median cos `0.985383`
- seed 1 B12 first below-bar layer: `h_layer_24` median cos `0.984894`

## Verdict

**C39 fixes the replay-contract blocker from C38, but it does not expose a
seed1-only first boundary or sub-op.**

With fully conditioned HF replays, both the control seed and the failing seed
land the same first shared bad boundary:

1. `h_layer_0` still matches.
2. The first bad B10 boundary is `h_layer_8`.
3. B11 does not isolate a bad layer-0 GDN sub-op.
4. The later-stack B12 drift still first appears at `h_layer_24` for both
   seeds.

That is an honest negative result for the original seed-specific bisect
question: the refreshed post-#423 artifact does **not** justify a seed1-only
fix target.

## Next recommendation

Do **not** claim a seed1-only localization from this artifact.

The next narrow task should bisect inside the shared early-GDN span between
`h_layer_0` and `h_layer_8` on the updated replay contract, because that is
now the first trustworthy bad boundary on both seeds.

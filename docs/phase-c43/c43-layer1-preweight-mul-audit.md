# Phase C43 — layer 1 pre-weight multiply audit after PR #429

## Remaining-work preflight

Fresh `origin/main` on 2026-04-23 still satisfied the task gate:

1. PR #429 / Phase C42 was present on `origin/main`.
2. No newer open or merged kiln PR had already localized the shared
   layer-1 divergence beyond `layer_1_input_norm_pre_weight`.
3. The narrow remaining span from C42 was still exactly:
   `layer_1_input_norm_rms_inv` (shared-good) ->
   `layer_1_input_norm_pre_weight` (shared-bad).

So this task stayed inside the layer-1 input-layernorm pre-weight path and did
not widen into deeper transformer-block or decode work.

## Scope

The C43 instrumentation adds a dedicated opt-in tap namespace under
`KILN_MTP_DUMP_C43_LAYER1_PREWEIGHT_TAPS=1`:

- `layer_1_residual_input`
- `layer_1_input_norm_rms_inv`
- `layer_1_input_norm_pre_weight_broadcast_mul`
- `layer_1_input_norm_pre_weight_scalar_affine`
- `layer_1_post_input_norm`

The critical new check is the split pre-weight site:

- `broadcast_mul`: kiln's existing `x_f32.broadcast_mul(rms_inv)` path
- `scalar_affine`: an independently computed equivalent that selects each row,
  extracts the per-row scalar `rms_inv`, and applies a scalar `affine`
  multiply instead of the broadcast path

If `broadcast_mul` diverged but `scalar_affine` stayed clean, the lead
hypothesis would become layout / row selection around the broadcast path. If
both diverged, the bad span would remain on the normalized pre-weight values
themselves.

## Validation

Run on the required RunPod kiln image on an on-demand `NVIDIA RTX A6000`.

Required pre-bench validation:

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

Both passed on the final A6000 run after fixing the `write_mtp_dump` test call
sites for the new `c43_taps` parameter.

## Standard workload

Build + model prep:

```bash
cd /workspace/kiln
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 -m pip install transformers safetensors sentencepiece
```

Seeded C43 capture sweep:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post429_c43_20260423_seed${seed}_captures
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
  KILN_MTP_DUMP_C43_LAYER1_PREWEIGHT_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post429_c43_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post429_c43_20260423_seed${seed}.bench.stderr
done
```

Representative compare dumps:

- seed 0: `profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference generation + compare:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post429_c43_20260423_seed0_ref_bf16.safetensors \
  --device cuda \
  --c43-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post429_c43_20260423_seed0_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c43-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post429_c43_20260423_seed1_ref_bf16.safetensors \
  --device cuda \
  --c43-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post429_c43_20260423_seed1_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c43-taps

python3 scripts/mtp_compare.py --c43 \
  --pair seed0:profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post429_c43_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post429_c43_20260423_seed1_ref_bf16.safetensors \
  --out profiling-artifacts/post429_c43_20260423_compare_bf16.txt

python3 scripts/mtp_compare.py --c43 \
  --pair seed0:profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post429_c43_20260423_seed0_ref_fp32.safetensors \
  --pair seed1:profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post429_c43_20260423_seed1_ref_fp32.safetensors \
  --out profiling-artifacts/post429_c43_20260423_compare_fp32.txt
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10057.8 | 38.9 | 0.693 |
| 1 | 508 | 414.0 | 21.3 | 0.293 |

The post-#429 seed split still reproduces on the final A6000 run.

## Verdict

**The earliest shared bad tap stays on the pre-weight path, and the
independently computed equivalent is also bad.**

Across both bf16 and fp32 reference compares:

| Reference dtype | Seed 0 earliest bad | Seed 1 earliest bad | Independent equivalent |
| --- | --- | --- | --- |
| bf16 | `layer_1_input_norm_pre_weight_broadcast_mul` | `layer_1_input_norm_pre_weight_broadcast_mul` | `layer_1_input_norm_pre_weight_scalar_affine` is also `DIV` |
| fp32 | `layer_1_input_norm_pre_weight_broadcast_mul` | `layer_1_input_norm_pre_weight_broadcast_mul` | `layer_1_input_norm_pre_weight_scalar_affine` is also `DIV` |

Shared-good taps remained unchanged:

- `layer_1_residual_input`
- `layer_1_input_norm_rms_inv`

Shared-bad taps remained:

- `layer_1_input_norm_pre_weight_broadcast_mul`
- `layer_1_input_norm_pre_weight_scalar_affine`
- `layer_1_post_input_norm`

So C43 rejects the "broadcast layout / row selection only" hypothesis. The bad
span is still the normalized pre-weight values themselves, not a quirk of the
specific `broadcast_mul` path.

## Evidence

- `profiling-artifacts/post429_c43_20260423_seed0.bench.json`
- `profiling-artifacts/post429_c43_20260423_seed0.bench.stderr`
- `profiling-artifacts/post429_c43_20260423_seed1.bench.json`
- `profiling-artifacts/post429_c43_20260423_seed1.bench.stderr`
- `profiling-artifacts/post429_c43_20260423_compare_bf16.txt`
- `profiling-artifacts/post429_c43_20260423_compare_fp32.txt`

## Discarded attempt

An earlier H200 NVL attempt was discarded and is **not** the source of truth
for this phase. With `KILN_W4A16=1`, the run failed inside the verify forward
path with `CUDA_ERROR_ILLEGAL_INSTRUCTION` before the audit could complete.
The final verdict above comes from the successful A6000 rerun only.

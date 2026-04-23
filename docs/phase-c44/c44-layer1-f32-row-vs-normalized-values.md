# Phase C44 — layer 1 F32 row vs normalized values after PR #430

## Remaining-work preflight

Fresh `origin/main` on 2026-04-23 still satisfied the task gate:

1. PR #430 / Phase C43 was present on `origin/main`.
2. No newer open or merged kiln PR had already localized the shared layer-1
   divergence beyond the C43 boundary.
3. Fresh main still did **not** prove whether the bad values were already
   present in the F32-cast row before scaling or only appeared when the
   shared-good `rms_inv` scalar was applied.

So this task proceeded and stayed strictly inside layer-1 input-layernorm
numerics.

## Scope

The C44 instrumentation adds a dedicated opt-in tap namespace under
`KILN_MTP_DUMP_C44_LAYER1_F32_ROW_TAPS=1`:

- `layer_1_residual_input_f32_row`
- `layer_1_input_norm_rms_inv_scalar`
- `layer_1_input_norm_pre_weight_row_scalar_affine`

The tap set is deliberately narrower than C43:

- capture only the **last replay row** after `x.to_dtype(F32)`
- capture only the matching row-local `rms_inv` scalar
- capture only that row after scalar-affine normalization

This answers one question only:

- if `layer_1_residual_input_f32_row` were already bad, the bad values would
  already exist before normalization
- if the F32 row stayed shared-good and only the normalized row diverged, the
  first bad values would appear when normalization is applied

## Validation

Run on the required RunPod kiln image on an on-demand `NVIDIA RTX A6000`.

Required pre-bench validation:

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

Both passed on the final A6000 run after fixing one remaining
`write_mtp_dump(...)` unit-test call site for the new `c44_taps` parameter.

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

Seeded C44 capture sweep:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post430_c44_20260423_seed${seed}_captures
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
  KILN_MTP_DUMP_C44_LAYER1_F32_ROW_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post430_c44_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post430_c44_20260423_seed${seed}.bench.stderr
done
```

Representative compare dumps:

- seed 0: `profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference generation + compare:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post430_c44_20260423_seed0_ref_bf16.safetensors \
  --device cuda \
  --c44-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post430_c44_20260423_seed0_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c44-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post430_c44_20260423_seed1_ref_bf16.safetensors \
  --device cuda \
  --c44-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post430_c44_20260423_seed1_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c44-taps

python3 scripts/mtp_compare.py --c44 \
  --pair seed0:profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post430_c44_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post430_c44_20260423_seed1_ref_bf16.safetensors \
  --out profiling-artifacts/post430_c44_20260423_compare_bf16.txt

python3 scripts/mtp_compare.py --c44 \
  --pair seed0:profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post430_c44_20260423_seed0_ref_fp32.safetensors \
  --pair seed1:profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post430_c44_20260423_seed1_ref_fp32.safetensors \
  --out profiling-artifacts/post430_c44_20260423_compare_fp32.txt
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10046.8 | 38.3 | 0.716 |
| 1 | 508 | 397.5 | 23.0 | 0.270 |

The post-#430 seed split still reproduces on the final A6000 run.

## Verdict

**The earliest shared bad C44 tap is _not_ the F32-cast row. The row stays
shared-good and the first shared divergence appears only when normalization is
applied to produce the pre-weight values.**

Across both bf16 and fp32 reference compares:

| Reference dtype | Seed 0 earliest bad | Seed 1 earliest bad | Shared-good row taps |
| --- | --- | --- | --- |
| bf16 | `layer_1_input_norm_pre_weight_row_scalar_affine` | `layer_1_input_norm_pre_weight_row_scalar_affine` | `layer_1_residual_input_f32_row`, `layer_1_input_norm_rms_inv_scalar` |
| fp32 | `layer_1_input_norm_pre_weight_row_scalar_affine` | `layer_1_input_norm_pre_weight_row_scalar_affine` | `layer_1_residual_input_f32_row`, `layer_1_input_norm_rms_inv_scalar` |

Representative C44 rows:

- seed 0 bf16:
  `layer_1_residual_input_f32_row` is `ok`,
  `layer_1_input_norm_rms_inv_scalar` is `ok`,
  and `layer_1_input_norm_pre_weight_row_scalar_affine` is already `DIV`
- seed 1 fp32:
  `layer_1_residual_input_f32_row` is `ok`,
  `layer_1_input_norm_rms_inv_scalar` is `ok`,
  and `layer_1_input_norm_pre_weight_row_scalar_affine` is already `DIV`

So C44 answers the task question directly:

- the bad values are **not** already present in the layer-1 residual row after
  `x.to_dtype(F32)`
- the row stays shared-good and the first shared divergence appears only when
  applying normalization to produce the row's pre-weight values

## Recommendation

Do not widen beyond layer-1 input-layernorm numerics yet.

The remaining culprit span is now tighter than C43:

- `layer_1_residual_input_f32_row` (shared-good)
- `layer_1_input_norm_rms_inv_scalar` (shared-good)
- `layer_1_input_norm_pre_weight_row_scalar_affine` (earliest shared-bad)

So the next slice, if any, should stay inside the row-local normalization
application itself rather than revisiting the F32 cast or the RMS reduction.

## Evidence

- `profiling-artifacts/post430_c44_20260423_seed0.bench.json`
- `profiling-artifacts/post430_c44_20260423_seed0.bench.stderr`
- `profiling-artifacts/post430_c44_20260423_seed1.bench.json`
- `profiling-artifacts/post430_c44_20260423_seed1.bench.stderr`
- `profiling-artifacts/post430_c44_20260423_compare_bf16.txt`
- `profiling-artifacts/post430_c44_20260423_compare_fp32.txt`

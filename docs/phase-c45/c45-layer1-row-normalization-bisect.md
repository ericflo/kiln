# Phase C45 — post-#432 healthy-RunPod rerun

## Remaining-work preflight

Fresh `origin/main` on 2026-04-23 still satisfied the rerun gate:

1. PR #432 / Phase C45 instrumentation was present on `origin/main`
   (`5574aed`, merged 2026-04-23 14:34 UTC).
2. No newer open or merged kiln PR had already recorded a successful post-#432
   C45 rerun or an earlier shared-bad C45 tap.
3. No separate pending or running kiln task already covered this exact rerun.

So this task proceeded.

## Scope

The C45 instrumentation adds a dedicated opt-in tap namespace under
`KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1`:

- `layer_1_residual_input_f32_row_values`
- `layer_1_input_norm_rms_inv_scalar`
- `layer_1_input_norm_pre_weight_row_scalar_values`
- `layer_1_input_norm_pre_weight_row_reconstructed`

This keeps the task diagnostic-only and splits the previously-bad C44 site one
step further:

- selected residual row values
- the row-local `rms_inv` scalar
- the flat row-scalar multiply result
- the reconstructed row-shaped output right before the existing post-input-norm
  path

The goal is to tell whether the first shared drift now appears in the selected
row values, in the scalar multiply itself, or only when reconstructing the row
shape.

## Local validation

Completed locally before any GPU spend:

```bash
cargo test --locked -p kiln-model c45 -- --nocapture
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
```

Both passed on 2026-04-23.

## GPU validation

The rerun completed on the first healthy allowed pod:

- `NVIDIA RTX A6000`: `SUPPLY_CONSTRAINT`
- `NVIDIA A40`: healthy on-demand pod, SSH ready, final verdict source

Pod / image:

- image: `ghcr.io/ericflo/kiln-runpod:latest`
- GPU: `NVIDIA A40` (`46068 MB` VRAM from `nvidia-smi`)

Validation commands run on pod:

```bash
python3 $RP bg $POD_ID /tmp/build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file $POD_ID /workspace/kiln/target/release/kiln-bench --timeout 3600

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && cargo test --locked -p kiln-model c45 -- --nocapture'
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py'
```

All three passed.

Model / Python prep required on the healthy pod:

```bash
python3 $RP ssh $POD_ID 'hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b'
python3 $RP ssh $POD_ID 'python3 -m pip install transformers sentencepiece'
```

The exact C45 dump rerun that produced the verdict was:

```bash
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
  KILN_MTP_DUMP_PATH=profiling-artifacts/post432_c45_seed0_captures/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 0 \
  > profiling-artifacts/post432_c45_seed0.bench.json \
  2> profiling-artifacts/post432_c45_seed0.bench.stderr'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
  KILN_MTP_DUMP_PATH=profiling-artifacts/post432_c45_seed1_captures/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 1 \
  > profiling-artifacts/post432_c45_seed1.bench.json \
  2> profiling-artifacts/post432_c45_seed1.bench.stderr'
```

Representative compare inputs:

- seed 0 kiln dump:
  `profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1 kiln dump:
  `profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference generation + compare:

```bash
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_h_main_reference_dump.py \
    --checkpoint /workspace/qwen3.5-4b \
    --kiln-dump profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors \
    --out profiling-artifacts/post432_c45_seed0_ref.safetensors \
    --device cuda \
    --c45-taps'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_compare.py --c45 \
    --kiln profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors \
    --ref profiling-artifacts/post432_c45_seed0_ref.safetensors \
    > profiling-artifacts/post432_c45_seed0_compare.txt'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_h_main_reference_dump.py \
    --checkpoint /workspace/qwen3.5-4b \
    --kiln-dump profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors \
    --out profiling-artifacts/post432_c45_seed1_ref.safetensors \
    --device cuda \
    --c45-taps'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_compare.py --c45 \
    --kiln profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors \
    --ref profiling-artifacts/post432_c45_seed1_ref.safetensors \
    > profiling-artifacts/post432_c45_seed1_compare.txt'
```

## Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 384.8 | 36.8 | 0.740 |
| 1 | 508 | 393.4 | 24.2 | 0.296 |

## Verdict

The post-#432 rerun produced a stable two-seed verdict:

- seed 0 earliest shared bad tap:
  `layer_1_input_norm_pre_weight_row_scalar_values`
- seed 1 earliest shared bad tap:
  `layer_1_input_norm_pre_weight_row_scalar_values`
- last shared-good tap on both seeds:
  `layer_1_input_norm_rms_inv_scalar`

So the first shared C45 drift is **not** in the selected row values and **not**
in the row-local `rms_inv` scalar. It first appears in the flat row-scalar
multiply values, then remains bad in the reconstructed row-shaped output.

## Recommendation

Do not widen to C46 or expand the tap contract. The next audit should stay
inside the row-local scalar multiply that produces
`layer_1_input_norm_pre_weight_row_scalar_values`; row selection and the
row-local `rms_inv` scalar are provisionally cleared on current `main`.

## Evidence

- `profiling-artifacts/post432_c45_seed0.bench.json`
- `profiling-artifacts/post432_c45_seed0.bench.stderr`
- `profiling-artifacts/post432_c45_seed1.bench.json`
- `profiling-artifacts/post432_c45_seed1.bench.stderr`
- `profiling-artifacts/post432_c45_seed0_compare.txt`
- `profiling-artifacts/post432_c45_seed1_compare.txt`

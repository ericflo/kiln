# Phase C45 — post-#433 row-local scalar-multiply follow-up

## Scope

PR #433 localized the first shared bad C45 tap to
`layer_1_input_norm_pre_weight_row_scalar_values`, with
`layer_1_input_norm_rms_inv_scalar` as the last shared-good tap. This
follow-up keeps the audit strictly inside that row-local scalar-multiply
boundary under `KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1`.

The narrowed tap contract is:

- `layer_1_input_norm_rms_inv_scalar`
- `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_scalar_values`
- `layer_1_input_norm_pre_weight_row_reconstructed`

Interpretation:

- If `layer_1_input_norm_rms_inv_scalar` is first bad, the shared-good PR #433
  boundary regressed and the problem is upstream of scalar extraction.
- If `layer_1_input_norm_rms_inv_scalar_extracted_values` is first bad, the
  row-local scalar tensor is still shared-good and the next target is the
  Rust-side scalar extraction / per-batch replay path.
- If `layer_1_input_norm_pre_weight_row_scalar_values` is first bad, scalar
  extraction is cleared and the next target stays on the actual row-local
  scalar multiply.
- If `layer_1_input_norm_pre_weight_row_reconstructed` is first bad, the flat
  multiply is cleared and the next target stays only on reshape /
  concatenation of the multiplied row.

## Commands

Local validation:

```bash
cargo test --locked -p kiln-model c45 -- --nocapture
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
```

Representative GPU rerun on RunPod:

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

python3 $RP status
POD_ID=$(python3 $RP launch --kiln-image --gpu "NVIDIA RTX A6000" \
  --name kiln-c45-row-scalar-narrow --disk-gb 80 | tail -1 | jq -r .id)
trap 'python3 $RP terminate $POD_ID 2>/dev/null || true' ERR INT TERM
python3 $RP wait $POD_ID

B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh $POD_ID \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --clone"
python3 $RP bg $POD_ID /tmp/build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file $POD_ID /workspace/kiln/target/release/kiln-bench --timeout 3600

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && cargo test --locked -p kiln-model c45 -- --nocapture'
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py'
```

Representative seed reruns:

```bash
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
  KILN_MTP_DUMP_PATH=profiling-artifacts/post433_c45_row_scalar_seed0_captures/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 0 \
  > profiling-artifacts/post433_c45_row_scalar_seed0.bench.json \
  2> profiling-artifacts/post433_c45_row_scalar_seed0.bench.stderr'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
  KILN_MTP_DUMP_PATH=profiling-artifacts/post433_c45_row_scalar_seed1_captures/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 1 \
  > profiling-artifacts/post433_c45_row_scalar_seed1.bench.json \
  2> profiling-artifacts/post433_c45_row_scalar_seed1.bench.stderr'
```

Representative compare inputs:

- seed 0: `profiling-artifacts/post433_c45_row_scalar_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post433_c45_row_scalar_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference generation + compare:

```bash
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_h_main_reference_dump.py \
    --checkpoint /workspace/qwen3.5-4b \
    --kiln-dump profiling-artifacts/post433_c45_row_scalar_seed0_captures/mtp_pos-0/step-1.safetensors \
    --out profiling-artifacts/post433_c45_row_scalar_seed0_ref.safetensors \
    --device cuda \
    --c45-taps'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_compare.py --c45-row-scalar \
    --kiln profiling-artifacts/post433_c45_row_scalar_seed0_captures/mtp_pos-0/step-1.safetensors \
    --ref profiling-artifacts/post433_c45_row_scalar_seed0_ref.safetensors \
    > profiling-artifacts/post433_c45_row_scalar_seed0_compare.txt'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_h_main_reference_dump.py \
    --checkpoint /workspace/qwen3.5-4b \
    --kiln-dump profiling-artifacts/post433_c45_row_scalar_seed1_captures/mtp_pos-2/step-1.safetensors \
    --out profiling-artifacts/post433_c45_row_scalar_seed1_ref.safetensors \
    --device cuda \
    --c45-taps'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && \
  python3 scripts/mtp_compare.py --c45-row-scalar \
    --kiln profiling-artifacts/post433_c45_row_scalar_seed1_captures/mtp_pos-2/step-1.safetensors \
    --ref profiling-artifacts/post433_c45_row_scalar_seed1_ref.safetensors \
    > profiling-artifacts/post433_c45_row_scalar_seed1_compare.txt'
```

## Expected Outputs After Rerun

- `profiling-artifacts/post433_c45_row_scalar_seed0_compare.txt`
- `profiling-artifacts/post433_c45_row_scalar_seed1_compare.txt`

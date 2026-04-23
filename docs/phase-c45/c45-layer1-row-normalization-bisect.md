# Phase C45 — layer 1 row normalization bisect after PR #431

## Remaining-work preflight

Fresh `origin/main` on 2026-04-23 still satisfied the task gate:

1. PR #431 / Phase C44 was present on `origin/main`.
2. No newer open or merged kiln PR already localized the shared layer-1
   row-level normalization boundary beyond C44.

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

## Planned GPU validation

The intended source-of-truth workload stayed exactly aligned with the task:

```bash
python3 $RP bg $POD_ID /tmp/build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file $POD_ID /workspace/kiln/target/release/kiln-bench --timeout 3600

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && cargo test --locked -p kiln-model c45 -- --nocapture'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && export KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 && ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training > profiling-artifacts/post431_c45_seed0.bench.json 2> profiling-artifacts/post431_c45_seed0.bench.stderr'
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && export KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 KILN_SEED=1 && ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training > profiling-artifacts/post431_c45_seed1.bench.json 2> profiling-artifacts/post431_c45_seed1.bench.stderr'

python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && python3 scripts/mtp_h_main_reference_dump.py --c45-taps profiling-artifacts/<fresh-kiln-dump>.safetensors profiling-artifacts/post431_c45_seed0_ref.safetensors'
python3 $RP ssh $POD_ID 'source /root/.kiln-build-env && cd /workspace/kiln && python3 scripts/mtp_compare.py --c45 profiling-artifacts/post431_c45_seed0_compare.txt profiling-artifacts/post431_c45_seed1_compare.txt'
```

## RunPod blocker

GPU validation did **not** complete on 2026-04-23 because RunPod never
provided a healthy on-demand pod with a usable runtime / SSH endpoint.

Observed failures:

- direct launches for `NVIDIA RTX A6000`, `NVIDIA A40`, `NVIDIA A100 80GB PCIe`,
  `NVIDIA RTX 6000 Ada Generation`, and `NVIDIA L40S` all failed immediately
  with `SUPPLY_CONSTRAINT`
- fallback launches on `NVIDIA H200 NVL`, `NVIDIA RTX PRO 4500 Blackwell`, and
  `NVIDIA H100 PCIe` did create pods, but each pod stayed in:
  `desiredStatus=RUNNING` with `runtime: null`, so `runpod_api.py wait` never
  returned an SSH endpoint

That means no build, no bench run, no kiln dump, no HF replay dump, and no C45
compare report were produced in this task.

The raw launch / provisioning evidence is committed in:

- `profiling-artifacts/post431_c45_20260423_local_validation.txt`
- `profiling-artifacts/post431_c45_20260423_runpod_blocker.txt`

## Verdict

No fresh C45 verdict yet.

The code plumbing and local validation are complete, but the required GPU
artifact capture was blocked by RunPod provisioning failures on 2026-04-23.

## Recommendation

Re-run the committed C45 workflow on the next healthy on-demand RunPod pod and
fill in the earliest shared bad C45 tap. The code changes in this PR are ready
for that rerun; the missing piece is infrastructure, not instrumentation.

# Phase 6 C56 post-#476 native-MTP decode profile (2026-04-24)

## Result

C56 did **not** produce a valid native-MTP decode-window profile. Current `origin/main` at `f1530071bb489b4d72bbf8e6ad0062281b52c0bf` built and the focused conv1d CUDA tests passed on an on-demand RTX A6000, but the required C52/C54 workload failed during native-MTP prefill before the first `:kiln/mtp/step` decode range:

```text
Error: MTP latency benchmark failed

Caused by:
    0: MTP prefill (paged with last-hidden) failed
    1: gated deltanet layer 0 (linear attention, paged)
    2: causal_conv1d_prefill kernel failed
    3: kiln_causal_conv1d_prefill_bf16_f32 failed with status 3
```

The same failure reproduced without CUDA graphs (`KILN_CUDA_GRAPHS=false`), so this is not a CUDA graph capture-only issue.

## Setup

- Repo: `ericflo/kiln`
- Commit: `f1530071bb489b4d72bbf8e6ad0062281b52c0bf` (`phase6: add CUDA conv1d prefill fast path`)
- Branch: `phase6-post-476-mtp-profile`
- Pod: `2cetn7tf9zckij`
- GPU: `NVIDIA RTX A6000`, compute capability `8.6`, `49140 MiB`, driver `550.127.08`
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- Nsight Systems: `2024.5.1.113-245134619542v0`

The baked image still had Nsight Systems `2023.4.4` in `/usr/local/cuda/bin/nsys`, so I installed only `nsight-systems-2024.5.1` from the existing NVIDIA CUDA apt repo before profiling.

## Precondition Check

Before launching RunPod I inspected:

- `PROFILING.md`
- `docs/archive/phase-c/phase-c54/conv-child-nvtx-profile.md`
- open PRs for `ericflo/kiln`

There was no existing post-#476 C56/post-`f153007` profile section or artifact, and there were no open PRs at the time of inspection.

## Validation Commands

```bash
cd /workspace/kiln
git fetch origin main && git reset --hard origin/main
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
cargo test -p kiln-conv1d-kernel --release -- --nocapture
cargo test -p kiln-model --release --features cuda test_causal_conv1d_update_matches_fallback -- --nocapture
cargo test -p kiln-model --release --features cuda causal_conv1d_prefill -- --nocapture
cargo build --release --features cuda,nvtx --bin kiln-bench
```

Validation passed:

- `kiln-conv1d-kernel`: 4 passed
- `test_causal_conv1d_update_matches_fallback`: 1 passed
- `test_causal_conv1d_prefill_matches_fallback`: 1 passed, max abs diff `1.4901161e-8`, state parity max abs diff `0`
- `kiln-bench`: built in release mode with `cuda,nvtx`

Reduced validation log: `build-validation-reduced.log`.

## Profile Command

```bash
mkdir -p docs/archive/phase-c/phase-c56 /tmp/kiln-c56
NSYS=/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys
$NSYS profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-memory-usage=false --output /tmp/kiln-c56/c56-post-476-mtp \
  env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1 \
  2>&1 | tee docs/archive/phase-c/phase-c56/nsys-profile.log
$NSYS stats --report nvtx_sum,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum --format csv --output docs/archive/phase-c/phase-c56/c56-failed-prefill /tmp/kiln-c56/c56-post-476-mtp.nsys-rep \
  2>&1 | tee docs/archive/phase-c/phase-c56/nsys-stats-failed-prefill.log
```

## Decode-Window Method

The intended method is the C52/C54 decode window: first `:kiln/mtp/step` start through final decode NVTX end, excluding the `:kiln/mtp/step` parent wrapper from the denominator.

That window is unavailable in C56 because the run fails during native-MTP prefill before any `:kiln/mtp/step` range appears. Therefore decode tok/s, mean ITL, MTP α, top decode NVTX ranges, and top decode-window CUDA kernels are not validly measurable from this run.

## Fallback Prefill Check

` :kiln/gdn/conv/fallback_prefill` does **not** appear in the failed prefill trace, but this does not answer the original decode-window question because decode never starts. The fast CUDA prefill path is reached as `:kiln/gdn/conv/prefill_update`, then the CUDA wrapper returns status `3`.

Observed conv child ranges before failure:

| Range | Wall time | Instances |
| --- | ---: | ---: |
| `:kiln/gdn/conv/layout` | 0.053620 ms | 1 |
| `:kiln/gdn/conv/prefill_update` | 0.189640 ms | 1 |

## Failed-Prefill NVTX Ranges

These are **not decode-window hotspots**; they are the reduced trace before the prefill failure.

| Rank | NVTX range | Wall time | Share | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/in_proj` | 144.576983 ms | 98.9% | 1 |
| 2 | `:kiln/norm/pre_attn` | 1.097066 ms | 0.8% | 1 |
| 3 | `:kiln/gdn/conv` | 0.250360 ms | 0.2% | 1 |

Source: `c56-failed-prefill_nvtx_sum.csv` and `c56-failed-prefill_nvtx_pushpop_trace.csv`.

## Failed-Prefill CUDA Kernels

These are **not decode-window kernel totals**; they are the reduced trace before the prefill failure.

| Rank | CUDA kernel | Total time | Share | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `ucopy_bf16` | 192.268138 ms | 88.2% | 250 |
| 2 | `cast_bf16_f32` | 25.365062 ms | 11.6% | 104 |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8` | 0.200449 ms | 0.1% | 1 |
| 4 | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nn` | 0.099648 ms | 0.0% | 1 |
| 5 | `is_u32_bf16` | 0.026048 ms | 0.0% | 1 |

Source: `c56-failed-prefill_cuda_gpu_kern_sum.csv`.

## Artifacts

Committed artifacts are intentionally reduced and small:

- `build-validation-reduced.log`
- `nsys-profile.log`
- `bench-no-cuda-graphs.log`
- `nsys-stats-failed-prefill.log`
- `c56-failed-prefill_nvtx_sum.csv`
- `c56-failed-prefill_nvtx_pushpop_trace.csv`
- `c56-failed-prefill_cuda_gpu_kern_sum.csv`
- `c56-failed-prefill_cuda_api_sum.csv`
- `c56-failed-prefill_cuda_kern_exec_sum.csv`
- `summary.json`

The raw `.nsys-rep` and SQLite exports were not committed.

## Recommendation

Do **not** select a new Phase 6 decode optimization target from C56 yet. The post-#476 fast path blocks the required native-MTP workload before decode on a real Qwen3.5-4B A6000 run, even though the focused parity tests pass.

Next focused task: fix or guard `kiln_causal_conv1d_prefill_bf16_f32` for the real native-MTP prefill shape that fails with status `3`, then rerun C56. Until a post-fix run reaches `:kiln/mtp/step`, C54 remains the last valid native-MTP decode-window source of truth.

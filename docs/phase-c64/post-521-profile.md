# Phase 7 post-#521 current-main profile refresh

## Scope

Refresh the Phase 7 source-of-truth profile after PR #521
(`Use prefix cache with CUDA graphs`) landed on `main`. PR #521 closes
the 18-PR window of radix-cache, streaming-prefill, Metal-fusion, and
codex tuning work (PRs #503–#521) since the post-#502 refresh. This is
artifact-only profiling; no optimization code changed.

## Hardware and tooling

- Commit: `0fda0e667636bd782bb0a29feb06f2ff3d31d917` (`Use prefix cache with CUDA graphs (#521)`)
- Pod: `mfk88l8i8tab02`, lease `pod-66eb55349e1403350e6c342d`
- RunPod on-demand `NVIDIA RTX A6000`, `ghcr.io/ericflo/kiln-runpod:latest`
- GPU software: driver `550.127.08`, CUDA toolkit `12.4`, `KILN_CUDA_ARCHS=86`
- Profiler: Nsight Systems `2024.6.2.225-246235244400v0` from
  `/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys`
  (the baked image's `nsys 2023.4.4` is broken for stats import; per
  agent notes `kiln-nsys-baked-importer-broken` and
  `kiln-nsight-profiling-gotchas`, install 2024.6.2 from the NVIDIA apt
  repo before profiling on this image)
- Runtime flags: `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`

## Commands

### Build

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main
git reset --hard origin/main
export KILN_CUDA_ARCHS=86
cargo build --release --features cuda,nvtx --bin kiln-bench
```

### Decode bench (median-of-3, no profiler)

```bash
export KILN_W4A16=1 KILN_CUDA_GRAPHS=true
for i in 1 2 3; do
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged \
    --prompt-tokens 512 \
    --max-output-tokens 128 \
    --skip-training \
    --prompt-subset humaneval \
    --chat-template \
    --latency-only \
    --temperature 0.0 \
    --seed 1
done
```

Each invocation runs in a fresh process so the cold-start TTFT artifact
appears in run 1 only.

### Decode profile (single representative run, with nsys)

```bash
NSYS=/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys
mkdir -p /tmp/kiln-post521
$NSYS profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  --output /tmp/kiln-post521/post521_decode \
  env KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_CUDA_ARCHS=86 \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged \
    --prompt-tokens 512 \
    --max-output-tokens 128 \
    --skip-training \
    --prompt-subset humaneval \
    --chat-template \
    --latency-only \
    --temperature 0.0 \
    --seed 1
```

Profiled run reported `23.47 tok/s decode, 42.60 ms mean ITL` — the
~50 % slowdown vs the unprofiled bench is the documented nsys overhead
(post-#481 reported the same ~22.4 tok/s under nsys vs ~46 tok/s
unprofiled), so attribution percentages are honest while wall-clock
latencies from this run are not.

### Stats export

```bash
$NSYS stats --force-overwrite=true \
  --report cuda_gpu_kern_sum --report nvtx_kern_sum --report nvtx_pushpop_sum \
  --format csv --output /tmp/kiln-post521/post521_decode \
  /tmp/kiln-post521/post521_decode.nsys-rep
```

Outputs become:
- `post521_decode_cuda_gpu_kern_sum.csv`
- `post521_decode_nvtx_kern_sum.csv`
- `post521_decode_nvtx_pushpop_sum.csv`

## Decode bench median-of-3

| Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.85 | 21.81 | 21.68 | 25.43 | 9304.9 (cold-start TTFT artifact) |
| 2 | 46.18 | 21.65 | 21.69 | 26.99 | 353.7 |
| 3 | 45.71 | 21.88 | 21.87 | 26.36 | 351.7 |
| **median** | **45.85** | **21.81** | **21.69** | **26.36** | **352.7** (warm, runs 2–3 mean) |

## Decode top NVTX ranges

Source: `post521_decode_nvtx_pushpop_sum.csv`. Capture shape: 1 paged
prefill (494 prompt tokens) + 128 paged decode steps; the prefill
`:kiln/attn/full/prefill_initial` range is only 0.8 % of total
wall-clock, so this ranking is decode-dominated.

| Rank | NVTX range | Wall-clock share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `:kiln/gdn/gates` | **14.5%** | 3096 |
| 2 | `:kiln/gdn/gated_norm` | **13.9%** | 3096 |
| 3 | `:kiln/gdn/qk_norm` | **11.9%** | 3096 |
| 4 | `:kiln/gdn/in_proj` | **9.5%** | 3096 |
| 5 | `:kiln/attn/rope` | **8.3%** | 1032 |
| 6 | `:kiln/mlp/gate` | **5.0%** | 4128 |
| 7 | `:kiln/mlp/up` | **5.0%** | 4128 |
| 8 | `:kiln/mlp/down` | **4.7%** | 4128 |
| 9 | `:kiln/attn/full/decode_fused` | **3.8%** | 1024 |
| 10 | `:kiln/gdn/head_expand` | **2.9%** | 3096 |

## Decode top CUDA kernels

Source: `post521_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `ucopy_bf16` | **15.9%** | 5585 |
| 2 | `Marlin<(256,1,8,8,4,8)>` (W4A16 decode) | **13.3%** | 13312 |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | **10.9%** | 129 |
| 4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | **9.0%** | 3072 |
| 5 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **8.4%** | 6144 |
| 6 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | **4.9%** | 3072 |
| 7 | `bmul_f32` | **2.9%** | 23865 |
| 8 | `cast_bf16_f32` | **2.8%** | 17648 |
| 9 | `gdn_full_chunk_forward_kernel` (vendored GDN, PR #80) | **2.8%** | 168 |
| 10 | `fused_rmsnorm_kernel` | **2.6%** | 10449 |

## Comparison vs prior baselines

| Region / Kernel | post-#166 | post-#481 | post-#521 | Notes |
| --- | ---: | ---: | ---: | --- |
| `:kiln/gdn/gates` | 18.0% | 12.0% | 14.5% | Post-#481 included `qk_norm` at 24.4% which compressed the rest; post-#521 redistributes after PR #486. |
| `:kiln/gdn/gated_norm` | 17.5% | 11.2% | 13.9% | Same redistribution effect. |
| `:kiln/gdn/qk_norm` | 14.9% | 24.4% | 11.9% | PR #486 default-on fused QK norm is the dominant shipped delta. |
| `:kiln/gdn/in_proj` | 7.4% | n/a | 9.5% | |
| `ucopy_bf16` (kernel) | n/a | 15.4% | 15.9% | Stable; per `kiln-ucopy-bf16-exhausted` no green-lit follow-up. |
| `Marlin<(256,1,8,8,4,8)>` (kernel) | n/a | 13.6% | 13.3% | Stable. |
| Decode tok/s (median, no profiler) | **49.76** | n/a | **45.85** | **−7.9 %** vs post-#166 closing baseline — flag for investigation. |
| Mean ITL (median, no profiler) | 20.10 ms | n/a | 21.81 ms | +8.5 % vs post-#166. |
| P99 ITL (median, no profiler) | 25.46 ms | n/a | 26.36 ms | +3.5 % vs post-#166. |

The post-#481 row reports nsys-profiled wall-clock (22.4 tok/s) for its
plain-decode trace, not unprofiled bench numbers, so it cannot be
directly compared to the unprofiled medians.

## Implications

- **Investigate the −7.9 % decode tok/s regression vs post-#166 before
  any new kernel-fusion task.** Most parsimonious hypothesis:
  radix-prefix-cache lookup/registration hooks now run on every
  CUDA-graph chat-completion request (PRs #515 / #520 / #521). Run a
  `KILN_PREFIX_CACHE_ENABLED=0` vs `=1` A/B on the same shape to isolate
  this from other sources before touching kernels.
- The `gdn/gates` and `gdn/gated_norm` rebound is a redistribution
  effect from PR #486 shrinking `qk_norm` — their absolute
  ms/region budgets are essentially unchanged. PR #173 (`gates`) and
  PR #176 (`gates + gated_norm + recurrent`) both closed null; do not
  re-queue without sub-range NVTX evidence of real HBM traffic.
- `ucopy_bf16` remains the kernel king at 15.9 % but the remaining
  un-attempted sites yield ≤ 0.080 speedup at 1.5× local
  (`kiln-ucopy-bf16-exhausted`); not the productive next axis.
- Phase 7's productive next axes: prefix-cache regression A/B, KV
  cache FP8 long-context capability, and self-spec end-to-end
  benching.

## Artifacts in this directory

- `post-521-profile.md` — this file
- `post521_decode_nvtx_pushpop_sum.csv` — top NVTX ranges by wall-clock
- `post521_decode_cuda_gpu_kern_sum.csv` — top GPU kernels by wall-clock
- `post521_decode_nvtx_kern_sum.csv` — kernel attribution by NVTX range

The full `post521_decode.nsys-rep` (110 MB) is not committed because it
exceeds the in-repo artifact-size convention; it remains at
`/tmp/kiln-post521/post521_decode.nsys-rep` for the lifetime of pod
`mfk88l8i8tab02` (lease `pod-66eb55349e1403350e6c342d`).

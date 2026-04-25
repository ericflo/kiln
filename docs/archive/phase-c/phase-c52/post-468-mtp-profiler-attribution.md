# C52 Post-#468 MTP Profiler Attribution

Date: 2026-04-24

Commit: `55a5d9f26e6ca4be3d5f448935786d6a97e16a24`

GPU: RunPod on-demand `NVIDIA RTX A6000`, image `ghcr.io/ericflo/kiln-runpod:latest`.

## Purpose

C51 confirmed post-#466 C40f-style native-MTP benchmark medians, but its `top_decode_ranges` were null because the baked Nsight Systems 2023.4.4 importer failed with the known QuadD wrong-event-order bug. C52 repairs that attribution gap without implementing a CUDA kernel.

## Profiler Repair

The baked tool remained:

```text
NVIDIA Nsight Systems version 2023.4.4.54-234433681190v0 
```

C52 installed the compatible newer Nsight package from the already-configured NVIDIA CUDA apt repository:

```bash
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-2024.5.1
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys --version
```

Selected profiler:

```text
NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0 
```

This bypassed the 2023.4.4 importer failure: `nsys profile` produced `/tmp/kiln-post468-mtp.nsys-rep`, and all 2024 report exports used for this artifact exited successfully. Note that Nsight 2024 report names are `nvtx_sum`, `cuda_gpu_kern_sum`, and `cuda_api_sum`; the older aliases `nvtxsum`, `gpukernsum`, and `cudaapisum` print "Report could not be found" under 2024.5.1 and are preserved as compatibility-ladder evidence.

## Profile Command

```bash
KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 \
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/kiln-post468-mtp \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed 1 --chat-template --latency-only --prompt-subset humaneval --temperature 0.0
```

## Benchmark Result

| Metric | Value |
| --- | ---: |
| Prompt tokens | 515 |
| Generated tokens | 128 |
| Prefill time | 7921.2 ms |
| Decode throughput | 25.31 tok/s |
| Mean ITL | 39.51 ms |
| Draft acceptance α | 0.707 |

The profiled seed-1 run is slower than the unprofiled C51 median because Nsight tracing overhead is included; use it for attribution, not throughput gating.

## Decode Window Method

The decode window is derived from `nvtx_pushpop_trace`: first `:kiln/mtp/step` start (`12058641289` ns) through the final decode NVTX end (`17128527253` ns), covering `5069.9` ms and `75` MTP steps. Range percentages below sum rows wholly inside that window and exclude the `:kiln/mtp/step` parent wrapper from the denominator to avoid double-counting a high-level parent around child work.

## Top Decode NVTX Ranges

| Rank | Range | Share | Time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/conv` | 15.4% | 562.361 ms | 2328 |
| 2 | `:kiln/gdn/gates` | 14.2% | 518.513 ms | 2328 |
| 3 | `:kiln/gdn/gated_norm` | 13.4% | 487.295 ms | 2328 |
| 4 | `:kiln/gdn/qk_norm` | 10.9% | 396.837 ms | 2328 |
| 5 | `:kiln/attn/rope` | 8.5% | 308.079 ms | 851 |
| 6 | `:kiln/gdn/in_proj` | 4.5% | 164.971 ms | 2328 |
| 7 | `:kiln/attn/full/prefill` | 3.4% | 124.870 ms | 600 |
| 8 | `:kiln/attn/gdn/chunk_prep` | 3.0% | 109.453 ms | 1800 |
| 9 | `:kiln/residual` | 2.8% | 100.982 ms | 6358 |
| 10 | `:kiln/gdn/qkv_split` | 2.5% | 92.293 ms | 2328 |

## Top Decode Kernels

Kernel shares use CUDA GPU trace rows inside the same decode window. The multi-MB raw CUDA trace was reduced to `decode-window-kernels.csv` for the committed artifact set; the compact `cuda_gpu_kern_sum` / `cuda_kern_exec_sum` raw summaries are also committed.

| Rank | Kernel | Share | Time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | 23.6% | 563.199 ms | 7977 |
| 2 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8>(T1::Params)` | 12.3% | 293.366 ms | 1655 |
| 3 | `ucopy_bf16` | 7.6% | 181.148 ms | 19612 |
| 4 | `ampere_bf16_s16816gemm_bf16_128x64_sliced1x2_ldg8_f2f_stages_64x3_nn` | 6.5% | 154.203 ms | 2400 |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_nn_align8>(T1::Params)` | 5.6% | 133.066 ms | 74 |
| 6 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8>(T1::Params)` | 4.5% | 107.316 ms | 3335 |
| 7 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | 4.5% | 106.534 ms | 2928 |
| 8 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_nn_align8>(T1::Params)` | 3.2% | 75.305 ms | 3600 |
| 9 | `bmul_f32` | 3.1% | 73.891 ms | 27620 |
| 10 | `[CUDA memcpy Host-to-Device]` | 2.7% | 65.108 ms | 80434 |

## Interpretation

C52 changes the Phase 6 source of truth from stale pre-#466 carry-forward percentages to current post-#468 attribution. The top decode bucket is now `:kiln/gdn/conv` (`15.4%`), followed by `:kiln/gdn/gates` (`14.2%`) and `:kiln/gdn/gated_norm` (`13.4%`). `:kiln/gdn/qk_norm` is still material at `10.9%`, but it is no longer top-3 in this successful capture.

The next implementation target should therefore come from this current GDN decode cluster. Start with a static audit of `:kiln/gdn/conv` to avoid retrying an already-exhausted causal-conv/chunk-vendoring path; if that audit shows low leverage, target the adjacent `:kiln/gdn/gates` + `:kiln/gdn/gated_norm` cluster. Do not select a kernel from stale C50/C51 percentages.

## Artifacts

- `environment.txt` — commit, GPU, CUDA, baked and selected Nsight versions.
- `nsys-profile.log` — full profile command and bench JSON.
- `nsys-stats-v2024-status.txt` — successful 2024 report export status.
- `nvtx_pushpop_trace_nvtx_pushpop_trace.csv` — raw NVTX trace used for decode-window attribution.
- `decode-window-nvtx-ranges.csv` — compact derived top decode ranges.
- `decode-window-kernels.csv` — compact derived top decode kernels.
- `nsys-nvtxsum.txt`, `nsys-gpukernsum.txt`, `nsys-cudaapisum.txt` — failed old-report-name fallback evidence under Nsight 2024.5.1.

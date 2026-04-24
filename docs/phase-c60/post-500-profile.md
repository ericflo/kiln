# Phase C60 post-#500 current-main profile refresh

## Scope

Refresh the Phase 6 source-of-truth profile after PR #500 (`[codex] add CUDA GDN qk_norm GQA fast path`) landed on `main`. This is artifact-only profiling; no optimization code changed.

## Hardware and tooling

- Commit: `faf8cb3e1d5ae79fb9a2b3111c3f78052c6a51e7` (`faf8cb3`)
- Pod: `93wtgcujidv9ky`, RunPod on-demand `NVIDIA RTX A6000`, `ghcr.io/ericflo/kiln-runpod:latest`
- GPU software: driver `550.127.08`, CUDA toolkit `12.4`, `KILN_CUDA_ARCHS=86`
- Profiler: Nsight Systems `2024.6.2.225-246235244400v0` from `/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys`
- Runtime flags: `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`; native-MTP decode also used `KILN_SPEC_METHOD=mtp`, `KILN_BENCH_FORCE_MTP=1`, and `KILN_MTP_ARGMAX_FP32=1`

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

### Native-MTP decode profile

```bash
/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  --output /tmp/kiln-c60/post500_mtp_decode \
  env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
```

### Prompt-heavy prefill profile

```bash
/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  --output /tmp/kiln-c60/post500_prefill \
  env KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 4096 --max-output-tokens 16 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
```

### Stats export

```bash
for base in post500_mtp_decode post500_prefill; do
  /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys stats --force-overwrite=true \
    --report cuda_gpu_kern_sum --report nvtx_kern_sum --report nvtx_pushpop_sum \
    --format csv --output /tmp/kiln-c60/$base /tmp/kiln-c60/$base.nsys-rep
done
```

## Native-MTP decode result

- Prompt tokens: `515`
- Prefill: `419.1 ms` (`1229 tok/s`)
- Generated tokens: `128`
- Decode: `23.5 tok/s`, `42.5 ms` mean ITL
- MTP acceptance rate: `0.716`

### Decode top NVTX ranges

Source: `top_decode_nvtx.csv`. The `:kiln/mtp/step` parent wrapper is excluded from this ranking.

| Rank | NVTX range | Wall-clock share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `:kiln/gdn/gates` | **12.8%** | 2304 |
| 2 | `:kiln/gdn/gated_norm` | **12.1%** | 2304 |
| 3 | `:kiln/gdn/qk_norm` | **10.0%** | 2304 |
| 4 | `:kiln/attn/rope` | **7.7%** | 842 |
| 5 | `:kiln/gdn/in_proj` | **7.7%** | 2304 |

### Decode top CUDA kernels

Source: `top_decode_kernels.csv`.

| Rank | Kernel | GPU-kernel share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `ucopy_bf16` | **18.2%** | 20961 |
| 2 | `void Marlin<(int)256, (int)1, (int)8, (int)8, (int)4, (int)8>(const int4 *, const i...` | **10.0%** | 9880 |
| 3 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_...` | **8.7%** | 244 |
| 4 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_nn_...` | **6.3%** | 74 |
| 5 | `ampere_bf16_s16816gemm_bf16_128x64_sliced1x2_ldg8_f2f_stages_64x3_nn` | **5.4%** | 1776 |

## Prompt-heavy prefill result

- Prompt tokens: `4090`
- Prefill: `1635.6 ms` (`2501 tok/s`)
- Generated tokens: `17`
- Decode tail: `22.5 tok/s`, `44.5 ms` mean ITL

### Prefill top NVTX ranges

Source: `top_prefill_nvtx.csv`.

| Rank | NVTX range | Wall-clock share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `:kiln/gdn/in_proj` | **23.1%** | 408 |
| 2 | `:kiln/gdn/gates` | **10.7%** | 408 |
| 3 | `:kiln/gdn/gated_norm` | **9.2%** | 408 |
| 4 | `:kiln/gdn/qk_norm` | **8.9%** | 408 |
| 5 | `:kiln/attn/rope` | **6.0%** | 136 |

### Prefill top CUDA kernels

Source: `top_prefill_kernels.csv`.

| Rank | Kernel | GPU-kernel share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `<unnamed>::gdn_full_chunk_forward_kernel(const __nv_bfloat16 *, const __nv_bfloat16...` | **27.9%** | 1512 |
| 2 | `ucopy_bf16` | **22.3%** | 10065 |
| 3 | `void Marlin<(int)256, (int)4, (int)16, (int)4, (int)4, (int)8>(const int4 *, const ...` | **9.8%** | 520 |
| 4 | `ucopy_f32` | **3.7%** | 1704 |
| 5 | `bmul_f32` | **2.6%** | 3145 |

## Interpretation

Post-#500 native-MTP decode is no longer led by `:kiln/gdn/qk_norm`: the top NVTX buckets are `:kiln/gdn/gates` (**12.8%**), `:kiln/gdn/gated_norm` (**12.1%**), and `:kiln/gdn/qk_norm` (**10.0%**). This supports the PR #500 conclusion that the qk_norm GQA fusion shape is effectively exhausted and should not be retried.

Prompt-heavy prefill remains dominated by GDN work. The largest NVTX range is `:kiln/gdn/in_proj` (**23.1%**), and the largest kernel bucket is `gdn_full_chunk_forward_kernel` (**27.9%**), followed by `ucopy_bf16` (**22.3%**). The next Phase 6 target should therefore be the vendored GDN full-chunk/recurrent gated-delta path: audit kiln's current CUDA kernel against upstream flash-linear-attention/vLLM tuning and port a minimal bf16/F32 forward-path kernel win if one is available.

## Artifacts

- `summary.json`: machine-readable commit, hardware, latency, and top hotspots
- `top_decode_nvtx.csv`, `top_prefill_nvtx.csv`: curated top NVTX ranges
- `top_decode_kernels.csv`, `top_prefill_kernels.csv`: curated top CUDA kernels
- `post500_*_nvtx_pushpop_sum.csv`, `post500_*_nvtx_kern_sum.csv`, `post500_*_cuda_gpu_kern_sum.csv`: raw Nsight CSV exports
- `post500_mtp_decode.log`, `post500_prefill.log`, `post500_mtp_decode_stats.log`, `post500_prefill_stats.log`, `env.log`: profiler and environment logs

# C50 C40f-Style Native MTP Decode Profile

Date: 2026-04-24

Commit: `7c638e7e2d69cb16772619a2a32d6114767bf7e2`

GPU: RunPod on-demand NVIDIA RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

## Purpose

C49 showed that the C48 native-MTP regression was a harness/workload comparability artifact. C50 re-runs the restored C40f-style native-MTP harness on current `origin/main`, records the new benchmark anchor, and selects the next Phase 6 implementation target from the freshest comparable data.

## Setup

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin
git reset --hard origin/main
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench
cargo test --locked -p kiln-server bench -- --nocapture || true
python3 -m py_compile scripts/mtp_compare.py scripts/mtp_h_main_reference_dump.py
```

`cargo test --locked -p kiln-server bench -- --nocapture` exited successfully; the filtered bench tests reported `8 passed`. The Python compile check also passed.

## Benchmark Command

```bash
env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed <seed> \
  --chat-template --latency-only --prompt-subset humaneval --temperature 0.0
```

## Benchmark Summary

| Seed | Prompt subset | α | Decode tok/s | Mean ITL ms | Prefill ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | humaneval | 0.789 | 48.43 | 20.65 | 362.85 |
| 1 | humaneval | 0.707 | 44.00 | 22.73 | 371.45 |
| 2 | humaneval | 0.588 | 38.41 | 26.03 | 350.41 |

| Metric | Median | Mean |
| --- | ---: | ---: |
| Acceptance α | 0.707 | 0.694 |
| Decode tok/s | 44.00 | 43.61 |
| Mean ITL ms | 22.73 | 23.14 |
| Prefill ms | 362.85 | 361.57 |

The C40f-style harness still clears both C49 gates: median α `0.707` is above `0.65`, and median decode `44.00 tok/s` is above the 10%-below-`38.25 tok/s` floor (`34.43 tok/s`). This means C50 should not reopen C48 model-math investigation.

## Nsight Status

The requested `nsys profile` command ran on the restored C40f-style native-MTP workload, but the baked Nsight Systems 2023.4.4 importer failed before producing a `.nsys-rep` stats database:

```text
Importer error status: Importation failed.
ErrorText = "Wrong event order has been detected when adding events to the collection"
```

This happened for full-run capture, CUDA-only capture, CUDA-graph-granularity capture, no-CUDA-graphs capture, and a decode-window capture. `nvprof` is unsupported on the A6000, and `ncu` could not access performance counters under the pod permissions. The committed `nsys-profile.log`, `nsys-stats.txt`, `nsys-gpu-kernsum.txt`, and `nsys-cuda-api.txt` preserve the failed C50 capture attempt rather than committing multi-MB `.qdstrm` files.

## Top Decode Hotspots

Because the C50 `.nsys-rep` importer failed, the decode hotspot table uses the latest successful current-main decode NVTX attribution already checked in from the post-#442 profile, while C50 supplies the restored-harness MTP benchmark anchor on current main.

| Rank | Decode range | Wall-clock share | Interpretation |
| ---: | --- | ---: | --- |
| 1 | `:kiln/gdn/gates` | 17.9% | Largest GDN-side decode bucket. |
| 2 | `:kiln/gdn/gated_norm` | 17.3% | Adjacent GDN normalization bucket with near-equal cost. |
| 3 | `:kiln/gdn/qk_norm` | 15.0% | Third GDN normalization bucket; still larger than full-attention decode. |

Kernel-level post-#442 cross-check had `Marlin<(256,1,8,8,4,8)>` at `14.4%`, `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64...` at `11.6%`, and `ampere_bf16_s16816gemm_bf16_128x64...` at `9.7%`. The full-attention projection ranges remained much smaller (`:kiln/proj/qkv` `3.1%`, `:kiln/proj/o` `0.9%`).

## Recommendation

Queue exactly one next implementation target: **fuse or vendor the GDN gate/gated-norm decode path**.

Rationale: C50 restores native MTP to a usable harness anchor (median α `0.707`, median decode `44.00 tok/s`), so the next speed work should target the dominant GDN decode wall-clock cluster instead of C48 model-math. The top two ranges, `:kiln/gdn/gates` and `:kiln/gdn/gated_norm`, are adjacent and together account for `35.2%` of the last successful current-main decode NVTX attribution; they are higher leverage than FlashInfer/full-attention decode or another acceptance-rate investigation unless a future C40f-style run falls below the gates.

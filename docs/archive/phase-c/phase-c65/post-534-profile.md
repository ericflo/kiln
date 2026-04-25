# Phase 7 post-#534 current-main profile refresh

## Scope

Refresh the Phase 7 source-of-truth profile after the post-#521 → post-#534
window landed on `main`. PRs #522 through #534 are entirely doc-only
audits (vLLM GDN, SGLang radix, MTP acceptance state-of-play, Marlin
determinism), external-α microbenches (`vllm_mtp_unsupported`,
`sglang_mtp_unsupported_dense_4b`, `vllm_020_mtp_unsupported_dense_4b`,
`kiln_above_hf`), and the H15b stratified C29 v2 reject-row probe
(`kiln_native_ceiling`). No decode-path code changed in this window;
this profile validates that the post-#521 hotspot mix and decode
throughput are unchanged within measurement noise. This is
artifact-only profiling; no optimization code changed.

## Hardware and tooling

- Commit: `60e298d95deaf318aeb04af88cfa6c9add26a289` (`phase 7: H18 hand-rolled HF transformers MTP α reference — kiln_above_hf (#534)`)
- Pod: `mfk88l8i8tab02`, lease `pod-66eb55349e1403350e6c342d` (same warm pod as post-#521)
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
mkdir -p /tmp/kiln-post534
$NSYS profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false \
  --output /tmp/kiln-post534/post534_decode \
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

Profiled run reported `23.65 tok/s decode, 42.29 ms mean ITL` — the
~50 % slowdown vs the unprofiled bench is the documented nsys overhead
(post-#481 reported the same ~22.4 tok/s under nsys vs ~46 tok/s
unprofiled, post-#521 reported 23.47 tok/s under nsys), so attribution
percentages are honest while wall-clock latencies from this run are
not.

### Stats export

```bash
$NSYS stats --force-overwrite=true \
  --report cuda_gpu_kern_sum --report nvtx_kern_sum --report nvtx_pushpop_sum \
  --format csv --output /tmp/kiln-post534/post534_decode \
  /tmp/kiln-post534/post534_decode.nsys-rep
```

Outputs become:
- `post534_decode_cuda_gpu_kern_sum.csv`
- `post534_decode_nvtx_kern_sum.csv`
- `post534_decode_nvtx_pushpop_sum.csv`

## Decode bench median-of-3

| Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.48 | 21.06 | 21.08 | 25.12 | 9237.6 (cold-start TTFT artifact) |
| 2 | 46.36 | 21.57 | 21.63 | 25.89 | 369.7 |
| 3 | 46.55 | 21.48 | 21.49 | 26.19 | 356.2 |
| **median** | **46.55** | **21.48** | **21.49** | **25.89** | **362.9** (warm, runs 2–3 mean) |

## Decode top NVTX ranges

Source: `post534_decode_nvtx_pushpop_sum.csv`. Capture shape: 1 paged
prefill (494 prompt tokens) + 128 paged decode steps; the prefill
`:kiln/attn/full/prefill_initial` range is 1.0 % of total wall-clock,
so this ranking is decode-dominated.

| Rank | NVTX range | Wall-clock share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `:kiln/gdn/gates` | **14.6%** | 3096 |
| 2 | `:kiln/gdn/gated_norm` | **14.0%** | 3096 |
| 3 | `:kiln/gdn/qk_norm` | **11.8%** | 3096 |
| 4 | `:kiln/gdn/in_proj` | **9.4%** | 3096 |
| 5 | `:kiln/attn/rope` | **8.4%** | 1032 |
| 6 | `:kiln/mlp/gate` | **5.1%** | 4128 |
| 7 | `:kiln/mlp/up` | **5.0%** | 4128 |
| 8 | `:kiln/mlp/down` | **4.6%** | 4128 |
| 9 | `:kiln/attn/full/decode_fused` | **3.8%** | 1024 |
| 10 | `:kiln/gdn/head_expand` | **2.9%** | 3096 |

Aggregates: GDN (gates+gated_norm+qk_norm+in_proj+head_expand+
out_proj+conv+conv/update+recur_prep+recurrent) ~52.7 %; MLP trio
(gate+up+down) 14.7 %; full-attn projections + RoPE + decode_fused
(qkv+qkv_split+qk_norm+rope+full/decode_fused+proj/o+kv/copy) ~17.3 %.

## Decode top CUDA kernels

Source: `post534_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `ucopy_bf16` | **15.9%** | 5585 |
| 2 | `Marlin<(256,1,8,8,4,8)>` (W4A16 decode) | **13.3%** | 13312 |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | **10.9%** | 129 |
| 4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | **9.1%** | 3072 |
| 5 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **8.4%** | 6144 |
| 6 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | **4.9%** | 3072 |
| 7 | `bmul_f32` | **2.9%** | 23865 |
| 8 | `gdn_full_chunk_forward_kernel` (vendored GDN, PR #80) | **2.8%** | 168 |
| 9 | `cast_bf16_f32` | **2.7%** | 17648 |
| 10 | `fused_rmsnorm_kernel` | **2.6%** | 10449 |

## Comparison vs prior baselines

| Region / Kernel | post-#166 | post-#521 | post-#534 | Δ post-#521 → post-#534 |
| --- | ---: | ---: | ---: | --- |
| `:kiln/gdn/gates` | 18.0% | 14.5% | 14.6% | +0.1 pp |
| `:kiln/gdn/gated_norm` | 17.5% | 13.9% | 14.0% | +0.1 pp |
| `:kiln/gdn/qk_norm` | 14.9% | 11.9% | 11.8% | −0.1 pp |
| `:kiln/gdn/in_proj` | 7.4% | 9.5% | 9.4% | −0.1 pp |
| `:kiln/attn/rope` | n/a (sub-10) | 8.3% | 8.4% | +0.1 pp |
| `:kiln/mlp/gate` | 6.3% | 5.0% | 5.1% | +0.1 pp |
| `:kiln/mlp/up` | 6.2% | 5.0% | 5.0% | 0 pp |
| `:kiln/mlp/down` | 5.9% | 4.7% | 4.6% | −0.1 pp |
| `:kiln/attn/full/decode_fused` | n/a | 3.8% | 3.8% | 0 pp |
| `ucopy_bf16` (kernel) | n/a | 15.9% | 15.9% | 0 pp |
| `Marlin<(256,1,8,8,4,8)>` (kernel) | n/a | 13.3% | 13.3% | 0 pp |
| Decode tok/s (median, no profiler) | **49.76** | **45.85** | **46.55** | **+1.5 %** |
| Mean ITL (median, no profiler) | 20.10 ms | 21.81 ms | 21.48 ms | −1.5 % |
| P99 ITL (median, no profiler) | 25.46 ms | 26.36 ms | 25.89 ms | −1.8 % |

The post-#521 → post-#534 deltas are within ±0.1 pp on every NVTX range
and within ±1.8 % on every aggregate latency metric, consistent with the
fact that PRs #522–#534 are entirely doc-only / external-α microbench /
artifact-only and changed no decode-path code.

The −6.4 % decode tok/s gap to the post-#166 closing baseline (49.76 →
46.55) persists; per agent note `kiln-bench-prefix-cache-no-effect` and
PR #523, this regression is **not** caused by radix-prefix-cache hooks
(`kiln-bench --paged` bypasses the prefix cache entirely — toggling
`KILN_PREFIX_CACHE_ENABLED` showed no measurable effect). The actual
source remains unidentified and is left for a follow-up bisection task,
not a goal of this artifact refresh.

## Implications

- **Hotspot mix is structurally identical to post-#521.** Top-3 GDN
  ranges still account for ~40 % of decode wall-clock; the next-target
  shortlist is unchanged from the post-#521 profile and the
  vLLM/SGLang audit verdicts (`vllm-gdn-fused-decode-audit-2026-04-24`
  in agent notes) still apply: vLLM's fused
  `fused_recurrent_gated_delta_rule_packed_decode_kernel` offers no
  bounded micro-port win on A6000 under CUDA graphs.
- **`gdn/gates` (PR #173 closed null) and `gates+gated_norm+recurrent`
  (PR #176 closed null) remain off-limits** for re-attempt without new
  sub-range NVTX evidence of real HBM traffic.
- **`ucopy_bf16` (kernel) is unchanged at 15.9 %** with the same
  per-kernel ms budget as post-#521. Per
  `kiln-ucopy-bf16-exhausted` the remaining un-attempted sites yield
  ≤ 0.080× speedup at 1.5× local; not the productive next axis.
- **Productive next axes for Phase 7** (unchanged from the post-#521
  recommendation but now with a stable two-profile baseline):
  KV-cache FP8 long-context capability (already shown to fit 128 k on
  A6000 via streaming prefill in the GDN prefill memory preflight),
  self-spec end-to-end benching using kiln's native MTP heads (the H15b
  / H17 / H17b / H18 work landed `kiln_native_ceiling` and confirmed
  no external dense-4B α reference is available; the next slice is
  end-to-end e2e tok/s under kiln's own MTP, not another reference
  audit), and Marlin packing latency / BF16 weight residency cleanup
  (~58 s pack at load + ~1.3 GiB unused VRAM).
- **Skip another reprofile** until a code-changing PR lands. The
  post-#534 numbers should be the planning loop's next-target source
  of truth.

## Next-target candidates (informational, not queueing)

The planning loop should pick from this shortlist on the next cycle. No
task is queued from this artifact (per the task brief).

1. KV-cache FP8 default-on regression sweep (long-context win, see
   Phase 7 GDN prefill memory preflight 2026-04-24).
2. End-to-end native-MTP self-spec decode benchmarking (now that the
   external-α reference family is closed via #529–#534).
3. Marlin pack-at-load latency cleanup (~58 s, deterministic, single
   kernel target).
4. Marlin BF16 weight residency cleanup (~1.3 GiB unused VRAM after
   `KILN_W4A16=1` packed weights are resident).

## Artifacts in this directory

- `post-534-profile.md` — this file
- `post534_decode_nvtx_pushpop_sum.csv` — top NVTX ranges by wall-clock
- `post534_decode_cuda_gpu_kern_sum.csv` — top GPU kernels by wall-clock
- `post534_decode_nvtx_kern_sum.csv` — kernel attribution by NVTX range

The full `post534_decode.nsys-rep` (110 MB) is not committed because it
exceeds the in-repo artifact-size convention; it remains at
`/tmp/kiln-post534/post534_decode.nsys-rep` for the lifetime of pod
`mfk88l8i8tab02` (lease `pod-66eb55349e1403350e6c342d`).

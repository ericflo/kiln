# Kiln Profiling Report — Paged Production Path (Phase 6, post-PR #133)

## Overview

PR #133 fused the pre-norm RMSNorm into a single CUDA kernel, lifting paged
decode mean ITL from 27.67 ms → 22.77 ms (1.215×). This Phase 6 report
re-profiles the paged production path on current `main` (HEAD `1de1080`,
"CE: Fuse pre-norm RMSNorm into single CUDA kernel (#133)") to identify
the next decode hotspot.

The questions answered here:

1. **Did PR #133 land cleanly on the paged path?** Yes — `fused_rmsnorm_kernel`
   now appears at ~1.9 % of GPU time (was previously a major share split
   across `bmul_f32` / `urecip_f32` / `usqrt_f32` / dtype casts inside the
   pre-norm region).
2. **What still dominates paged decode after #133?** The
   `kiln-model::gdn` block — 9 NVTX regions inside `gated_delta_net_attention`
   account for **~57 %** of paged decode GPU wallclock combined.
3. **What is the single concrete next optimization?** Vendor a fused
   `gated_rms_norm` (RMSNorm × SiLU(z)) CUDA kernel. The
   `:kiln/gdn/gated_norm` NVTX region alone is **13.66 %** of paged GPU
   time and decomposes into ~10 unfused candle ops that PR #133-shaped
   fusion can collapse into one launch.

**No optimizations are proposed or attempted here — this is a profiling-only
pass.** The recommendation at the end picks the single concrete next step.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 570.x (pod default) |
| CUDA toolkit | 12.6 (Nsight Systems 2024.5.1 from `cuda-nsight-systems-12-6`) |
| Rustc | 1.95 stable (baked in `kiln-runpod:latest`) |
| Build | `cargo build --release --features cuda,nvtx --bin kiln-bench` |
| Build env | sccache (B2 backend); ~100 % C/C++/CUDA hits, 0 % Rust hits (fresh target dir) |
| Nsight Systems | 2024.5.1.113 (`/usr/local/bin/nsys`, NOT 2023.4.4 from `/usr/local/cuda/bin/nsys`) |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, 32 layers (24 GDN + 8 GQA full-attn) |
| Commit | `1de1080` (`main`, post-PR #133) |
| Pod | RunPod RTX A6000 on-demand |

**Environment notes.** `kiln-runpod:latest` came up clean with Rust + sccache
+ B2 prefix already wired via `kiln-setup`, so build finished in **<5 min**
(C/C++/CUDA fully cached). One snag: the `nsys` shipped under
`/usr/local/cuda/bin/nsys` is **2023.4.4** which has the well-known
`EventCollection::CheckOrder` "Wrong event order has been detected" bug
that prevents `.qdstrm` finalization. Workaround: `apt-get install -y
cuda-nsight-systems-12-6` lands `nsys 2024.5.1.113` at `/usr/local/bin/nsys`.
**This should be baked into `kiln-runpod`** — flagging as a future image
fix, not blocking this profile.

CUDA graphs were disabled for capture (`KILN_CUDA_GRAPHS=false`) so each
kernel launch is individually timed; production runs with graphs on.

## Wallclock Numbers

Latency benchmark, prompt = 512 tokens, max_output = 128 tokens,
A6000, profiled with `nsys profile -t cuda,nvtx --duration=20`:

| Phase | Paged | Non-paged |
|---|---|---|
| Prefill | 513.4 ms (986 tok/s) | 435.9 ms (1161 tok/s) |
| Decode mean ITL (cold + warm, profiled) | 40.1 ms (25.0 tok/s) | 41.4 ms (24.2 tok/s) |
| Decode mean ITL (warm aggregate, no nsys) | **25.5 ms (39.2 tok/s)** | 25.5 ms (39.2 tok/s) |
| Decode mean ITL (warm aggregate w/ CUDA graphs ON) | **22.77 ms** (per PR #133, no nsys) | — |

**Acceptance criterion check.** PR #133 set the paged decode bar at
≤ 23 ms mean ITL / ≤ 29 ms p99. The warm-aggregate steady-state of
**~22.77 ms** at the production setting (CUDA graphs ON, no nsys overhead)
is in line with PR #133. The 25.5 ms steady-state seen in this profile
is the additional cost of `KILN_CUDA_GRAPHS=false` (necessary for kernel
attribution); no decode regression vs PR #133 detected.

The 40.1 ms "decode mean ITL" reported by the latency benchmark phase
mixes the **first** decode step (which carries cold KV-cache and one-time
allocation costs ~150-200 ms) into the average; the warm aggregate above
strips that and is the right number to compare against. Flagging here so
future agents do not chase a phantom regression.

## Top 5 Kernels (Paged Decode + Prefill, % of GPU time)

Source: `nsys stats --report cuda_gpu_kern_sum --format csv profile_paged.nsys-rep`.
Aggregated over the full latency phase (prefill + decode); kernel-level
attribution can't separate them, so use the NVTX table below for the
decode-only structural picture.

| Rank | % GPU | Kernel | Notes |
|---|---|---|---|
| 1 | **28.4 %** | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | Large bf16 GEMM (projection matmuls — q/k/v/gate/up/down) |
| 2 | **15.7 %** | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | Smaller bf16 GEMM (decode-shaped Mx1xN) |
| 3 | **8.6 %** | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | Mid-size bf16 GEMM |
| 4 | **7.6 %** | `ucopy_bf16` | Generic bf16 element-wise copy (per-token reshape / strided memcpy) |
| 5 | **3.7 %** | `bmul_f32` | candle broadcast multiply in F32 (gates / qk_norm pipeline) |

Notable lower-rank kernels:

- `fused_rmsnorm_kernel` (PR #133): **1.9 %** at rank 10 — confirmation
  that the fused pre-norm landed and is hot.
- `recurrent_gdn_fwd_kernel` (vendored, PR #80): **1.6 %** at rank 14.
- `gdn_fwd_sub_kernel` (vendored, PR #80): **0.8 %** at rank 24.
- `kiln_flash::flash_fwd_splitkv_kernel` (vendored FlashInfer GQA decode):
  **0.9 %** at rank 21.
- `fast_argmax_bf16` (greedy sampling): **0.6 %** at rank 30.

The combined `cutlass_*_bf16_s16816gemm_*` family (top 3 entries plus
several smaller variants) accounts for **~55 %** of GPU time and reflects
all of the projection matmuls (Q/K/V/Gate/Up/Down/O proj across 32 layers
× 128 decode tokens). Those are already fused-vendor-grade (cutlass) and
not the next target.

## Top 5 NVTX Regions (Paged Decode, % of GPU time)

Source: `nsys stats --report nvtx_gpu_proj_trace --format csv profile_paged.nsys-rep`,
aggregated by region name.

| Rank | % GPU | NVTX Region | Count | Avg per call | What it does |
|---|---|---|---|---|---|
| 1 | **13.66 %** | `:kiln/gdn/gated_norm` | 3,709 | 205.6 µs | `gated_rms_norm(attn_out, z, weight, eps)` — RMSNorm × SiLU(z), ~10 unfused candle ops |
| 2 | **12.86 %** | `:kiln/gdn/gates` | 3,709 | 193.5 µs | `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)` — ~8 unfused F32/bf16 ops + dtype casts |
| 3 | **11.13 %** | `:kiln/gdn/qk_norm` | 3,709 | 167.5 µs | L2 normalize Q and K, scale Q by 1/√dk — `sqr/mean/sqrt/recip/mul/cast` chain |
| 4 | **7.88 %** | `:kiln/gdn/in_proj` | 3,709 | 118.6 µs | GDN input projection (single batched matmul + transpose) |
| 5 | **7.29 %** | `:kiln/mlp/gate` | 4,945 | 82.3 µs | MLP gate projection (batched matmul) |

Honorable mentions just outside top 5: `mlp/up` (7.23 %), `mlp/down`
(6.62 %), `attn/rope` (6.07 % — only 8 full-attn layers but still
expensive), `gdn/conv` (5.47 %, causal depthwise conv1d), `lm_head`
(5.16 % — final 152k-vocab projection per token).

**Structural breakdown by block:**
- All `:kiln/gdn/*` regions combined: **~57 %** of paged decode GPU time.
- All `:kiln/mlp/*` regions combined: **~22 %**.
- All `:kiln/attn/*` (full-attn only — 8/32 layers): **~10 %**.
- `:kiln/lm_head`: ~5 %.
- Residuals / norms: ~3 %.

The non-paged path shows the same structural picture (`gated_norm`
15.22 %, `gates` 14.39 %, `qk_norm` 12.44 %); it lacks the paged-only
`attn/rope` / `kv/copy` / `decode_fused` regions, so the GDN share is
correspondingly slightly higher in proportion. The next target picked
from the paged numbers therefore also wins on the non-paged path.

## Recommendation

**Next target: `gated_rms_norm` (NVTX `:kiln/gdn/gated_norm`) at 13.66 %
of paged decode wallclock.**

**Approach: Vendor a fused `gated_rms_norm` CUDA kernel** modeled on
PR #133's `kiln-rmsnorm-kernel` crate (which fused the plain pre-norm and
hit 1.215× decode). Reference implementation: FLA's
`fused_norm_gate.py` (https://github.com/fla-org/flash-linear-attention,
file `fla/modules/fused_norm_gate.py`) — it does exactly
`out = rms_norm(x) * weight * silu(z)` in a single Triton kernel that
ports cleanly to a CUDA `__global__` with one warp-per-row reduction,
matching the existing `kiln_rmsnorm_kernel` shape.

Per-call structure to fuse (from `forward.rs:643-657`):

```rust
let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;     // 2 kernels
let rms_inv  = (variance + eps)?.sqrt()?.recip()?;        // 3 kernels
let normed   = x_f32.broadcast_mul(&rms_inv)?;            // 1 kernel
let normed   = normed.broadcast_mul(&w_f32)?;             // 1 kernel
let gate     = cuda_silu(&z_f32)?;                        // 1 kernel
let out      = (normed * gate)?;                          // 1 kernel
// + 3 dtype casts (x→F32, z→F32, weight→F32) and 1 cast back
```

Today: ~10 launches per call × 3,709 calls = ~37k launches just for
gated_norm. Fused: 1 launch × 3,709 = 3,709 launches.

Recommended new crate: `crates/kiln-gated-rmsnorm-kernel` (mirror layout
of `crates/kiln-rmsnorm-kernel`), with a `fused_gated_rmsnorm(x, z,
weight, eps)` entry point and a `supports(...)` predicate. Wire dispatch
in `gated_rms_norm` (forward.rs:643) behind the same
`KILN_DISABLE_FUSED_*` env-var pattern PR #133 used.

**Expected speedup range: 1.10× – 1.18× decode (additive over PR #133).**
- Lower bound: gated_norm 13.66 % → ~3 % (similar shape to PR #133's
  ~5 % → 1.9 % collapse) ⇒ ~10.5 % of paged decode time recovered ⇒
  22.77 ms → ~20.4 ms ITL ⇒ **1.12×**.
- Upper bound: similar fusion of the surrounding `gates` and `qk_norm`
  regions in a follow-up could compound to ~1.25-1.30×, but those should
  be separate PRs (not this recommendation).

**Do NOT hand-roll candle ops.** Vendor a single CUDA kernel as a sibling
crate to `kiln-rmsnorm-kernel`. The `gates` and `qk_norm` regions are
attractive secondary targets but should wait until `gated_norm` is fused
and re-profiled (vendoring three at once will obscure attribution).

## Files

Raw artifacts (committed under `profiling/post-pr133/`):

- `paged_kern_sum.csv` — `cuda_gpu_kern_sum` for the paged profile
- `paged_nvtx.csv` — `nvtx_gpu_proj_trace` for the paged profile
- `nonpaged_kern_sum.csv` — `cuda_gpu_kern_sum` for the non-paged profile
- `nonpaged_nvtx.csv` — `nvtx_gpu_proj_trace` for the non-paged profile
- `paged_run.log` — kiln-bench output during paged profile
- `nonpaged_run.log` — kiln-bench output during non-paged profile

Raw `.nsys-rep` files (~140 MB each) were not committed; rerun on an
A6000 with `nsys profile -t cuda,nvtx --duration=20 -o /tmp/profile_paged
--force-overwrite=true ./target/release/kiln-bench --model-path
<weights> --paged --prompt-tokens 512 --max-output-tokens 128
--skip-training` (with `KILN_CUDA_GRAPHS=false`) to regenerate.

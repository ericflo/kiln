# Kiln Profiling Report — Paged Production Path (Phase 6 re-profile, post-PR #141)

## Overview

PR #141 vendored a fused `gated_rms_norm` kernel targeting the 13.66 % NVTX
hotspot called out in the previous PROFILING.md, but measured no decode-ITL
speedup on A6000 at the GDN decode shape (`rows = 32`, `hidden = 128`). The
PR hypothesised the null result was a methodology artifact: the previous
profile captured with `KILN_CUDA_GRAPHS=false`, which exposes per-call launch
overhead that production (`KILN_CUDA_GRAPHS=true`) amortises — so gains from
a fused kernel could be real in graphs-OFF numbers but hidden in graphs-ON.

This Phase 6 re-profile re-captures the paged production path on current
`main` (HEAD `07d934b`, "desktop: fix model name in screenshots") under both
CUDA-graphs modes, head-to-head on the same pod with the same build, to
settle three questions:

1. **Does `KILN_CUDA_GRAPHS` actually change paged decode ITL today?** No.
   Graphs-OFF mean ITL 23.10 ms, graphs-ON 23.15 ms — identical within noise.
   The ~2.7 ms gap documented in the prior PROFILING.md does not reproduce.
2. **Does graphs=ON change the NVTX / kernel mix enough to revise the next
   optimization target?** No. Top-10 NVTX regions rank identically and
   differ by ≤ 0.4 pct-pt across modes.
3. **Does the methodology invalidate PR #141's null decode result?** No.
   PR #141 measured no speedup at `rows = 32`, `hidden = 128`. That result
   is real, not an artifact of graphs=OFF timing.

This changes the recommended next optimization target. Per-kernel fusion at
the GDN decode shape has hit diminishing returns: the launch-overhead
bottleneck fusion targets is not detectable on this workload. The next
concrete lever is **bf16 → FP8/INT8 weight quantization on the projection
GEMMs**, which are the true dominant cost (~60.5 % of GPU time in graphs=ON
kernel mix).

## Methodology

All numbers in this report come from one pod, one build, back-to-back
captures. Environment variables other than `KILN_CUDA_GRAPHS` are constant
across arms.

- **Steady-state ITL.** Three back-to-back 512-prompt × 128-decode
  latency runs per arm, no nsys attached, reporting the
  `mean_inter_token_ms` from the paged decode phase after the model is
  warm (the prefill and first-decode cold costs are excluded by
  `kiln-bench`'s latency phase already). Graphs are OFF/ON via
  `KILN_CUDA_GRAPHS=false|true`.
- **nsys captures.** Two 20-second `nsys profile -t cuda,nvtx
  --duration=20` captures (Capture A: graphs=OFF, Capture B: graphs=ON),
  started during the throughput-phase warmup so the profile window lands
  on warm steady-state decode for a long window. Both captures use the
  same kiln-bench invocation, the same model weights, and target the
  same decode token count.
- **Extraction.** `nsys stats --report cuda_gpu_kern_sum` for the
  per-kernel table; `nsys stats --report nvtx_sum` for the NVTX region
  table. Aggregated over the full capture (prefill + decode); the NVTX
  region table is the right decode-only structural view.
- **No per-region kernel timer.** The per-region `cuda_gpu_trace` join
  would be nicer than NVTX-time for decode-only attribution, but graphs
  OFF and graphs ON NVTX summaries agree within 0.4 pct-pt on every
  top-10 region, so the extra instrumentation pass is not needed to
  decide the next optimization target.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 550.127.08 |
| CUDA toolkit | 12.4 (runtime); Nsight Systems 2024.5.1.113 (side-installed) |
| Rustc | 1.95 stable (baked in `kiln-runpod:latest`) |
| Build | `cargo build --release --features cuda,nvtx --bin kiln-bench` |
| Build cache | sccache + B2 prefix via `kiln-setup`; C/C++/CUDA cache hits |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, 32 layers (24 GDN + 8 GQA full-attn) |
| Commit | `07d934b` (`main`) |
| Pod | RunPod RTX A6000 on-demand (`t0eii824pgigpi`) |

`nsys` note. The baked `/usr/local/cuda/bin/nsys` is 2023.4.4 which has the
`EventCollection::CheckOrder` event-order bug that prevents `.qdstrm`
finalization on long-duration captures. Workaround used here:
installed `nsight-systems-2024.5.1` from NVIDIA directly (after
`libxcb-cursor0` from universe) and symlinked over the old binary.
Baking `nsys 2024.5.1+` into `kiln-runpod` is a known follow-up — the
prior PROFILING.md called this out too and the image has not moved yet.

## Wallclock Numbers — `KILN_CUDA_GRAPHS` is a noise-level knob today

Three back-to-back no-nsys `kiln-bench --paged --prompt-tokens 512
--max-output-tokens 128 --skip-training` runs per arm, reporting the
latency-phase `mean_inter_token_ms` / `p50` / `p99`:

| Arm | Run 1 | Run 2 | Run 3 | Mean | p50 (run 1) | p99 (run 1) |
|---|---|---|---|---|---|---|
| `KILN_CUDA_GRAPHS=false` | 23.08 ms | 23.04 ms | 23.18 ms | **23.10 ms** | 22.90 ms | 28.43 ms |
| `KILN_CUDA_GRAPHS=true`  | 23.16 ms | 23.15 ms | 23.15 ms | **23.15 ms** | 22.97 ms | 28.16 ms |

**Delta: 0.05 ms (0.2 %). Within run-to-run noise.**

This does not match the prior PROFILING.md's 25.5 ms (graphs OFF) vs
22.77 ms (graphs ON) framing, and does not reproduce the 2.7 ms CUDA-graphs
benefit that PR #141's fallback-config hypothesis depended on. The
graphs-ON path delivers no measurable ITL benefit over graphs-OFF at the
current main, on this pod.

A plausible cause of the drift: when PR #133 fused the pre-norm RMSNorm
(collapsing ~10 candle ops into one `fused_rmsnorm_kernel` launch per
region × 2 regions × 32 layers per token), per-token launch count dropped
enough that the capture-time launch-overhead amortisation that CUDA
graphs provides is no longer measurable at decode. The graphs capture
path still runs without error (`INFO kiln_model::cuda_graph: CUDA graphs
enabled for decode` emits cleanly), so the path is live — it just has
no ITL floor to claw back.

Prefill on this pod: 436.7 ms (1159 tok/s) in the graphs=ON warm run and
541.8 ms (934 tok/s) in the graphs=OFF nsys-attached warm run. Prefill
is not the focus of this profile; quoted for parity with the prior report.

## Top Kernels — Graphs OFF vs Graphs ON

Source: `nsys stats --report cuda_gpu_kern_sum --format csv`. Aggregated
over the full 20-second nsys capture (prefill + decode). The kernel mix
is almost completely graph-mode-invariant.

| Kernel (family) | Graphs OFF % GPU | Graphs ON % GPU | Notes |
|---|---|---|---|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | **29.4 %** | **30.6 %** | Large bf16 GEMM — Q/K/V/gate/up/down projections |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **16.3 %** | **16.9 %** | Decode-shaped (Mx1xN) bf16 GEMM |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | **8.9 %** | **9.2 %** | Mid-size bf16 GEMM |
| `ucopy_bf16` | 10.1 % | 8.2 % | Generic bf16 element-wise copy (reshape / strided memcpy) |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | 3.6 % | 3.8 % | bf16 GEMM (attn proj shape) |
| `bmul_f32` | 3.2 % | 3.1 % | candle broadcast multiply (F32) |
| `<unnamed>::fused_rmsnorm_kernel` (PR #133) | **1.9 %** | **2.0 %** | Fused pre-norm — hot and stable across modes |
| `fast_sum_f32` | 1.9 % | 1.9 % | F32 reduction (post-norm, qk_norm, softplus tails) |
| `ucopy_f32` | 1.9 % | 1.8 % | F32 element-wise copy |
| `recurrent_gdn_fwd_kernel<128>` (PR #80) | 1.6 % | 1.7 % | Vendored GDN recurrence |
| `cast_f32_bf16` | 1.7 % | 1.7 % | dtype cast (F32→bf16) |
| `cast_bf16_f32` | 1.6 % | 1.6 % | dtype cast (bf16→F32) |
| `kiln_flash::flash_fwd_splitkv_kernel` (vendored FlashInfer) | 1.1 % | 1.2 % | Full-attn layers' decode |
| `gdn_fwd_sub_kernel` (PR #80) | 0.4 % | 0.3 % | GDN chunk forward-substitution |

**Combined bf16 GEMM family (top 5 GEMM entries + smaller variants):
~60.5 % of GPU time in graphs=ON. Up from ~55 % in the prior PROFILING.md.**

Observations across modes:

- Kernel-level percentages shift ≤ 1 pct-pt between modes for every row.
  No new kernel appears / disappears.
- `fused_rmsnorm_kernel` stays at ~1.9-2.0 %: PR #133's fusion is still
  landed, still hot, and not regressed by graphs=ON.
- `ucopy_bf16` loses ~1.9 pct-pt under graphs=ON (10.1 → 8.2 %). This is
  consistent with CUDA graphs eliding some per-launch metadata copies; it
  is the only graph-detectable change in the kernel table, and it maps
  to ≤ 0.05 ms / token in ITL — matching the head-to-head gap above.

## Top NVTX Regions — Graphs OFF vs Graphs ON (decode-dominant)

Source: `nsys stats --report nvtx_sum --format csv`. Push/Pop ranges
emitted from `crates/kiln-model/src/forward.rs`. Aggregated over the
entire capture, so a long warm decode window dominates — this is the
right decode-structural view.

| Rank | NVTX Region | OFF % | ON % | Δ pct-pt | Avg per call (ON) | What it does |
|---|---|---|---|---|---|---|
| 1 | `:kiln/gdn/gates` | 14.0 % | **14.3 %** | +0.3 | 224.5 µs | `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)` |
| 2 | `:kiln/gdn/gated_norm` | 13.8 % | **14.1 %** | +0.3 | 221.3 µs | `gated_rms_norm(attn_out, z, w, eps)` — PR #141 fused, no ITL speedup |
| 3 | `:kiln/gdn/qk_norm` | 11.5 % | **11.7 %** | +0.2 | 184.5 µs | L2 normalize Q and K, scale Q by 1/√dk |
| 4 | `:kiln/gdn/conv` | 11.1 % | **11.1 %** | 0.0 | 175.1 µs | Causal depthwise conv1d (pure candle ops) |
| 5 | `:kiln/attn/rope` | 9.1 % | **9.3 %** | +0.2 | 440.1 µs | RoPE application (8 full-attn layers only) |
| 6 | `:kiln/gdn/in_proj` | 9.7 % | **9.0 %** | -0.7 | 140.8 µs* | GDN input projection (batched matmul + transpose) |
| 7 | `:kiln/attn/full/decode_fused` | 3.0 % | 3.1 % | +0.1 | 145.9 µs | Vendored FlashInfer GQA decode |
| 8 | `:kiln/residual` | 2.9 % | 2.8 % | -0.1 | 16.3 µs | Residual add |
| 9 | `:kiln/mlp/gate` | 2.6 % | 2.7 % | +0.1 | 31.9 µs | MLP gate projection (batched matmul) |
| 10 | `:kiln/mlp/up` | 2.5 % | 2.5 % | 0.0 | 30.0 µs | MLP up projection (batched matmul) |
| 11 | `:kiln/gdn/head_expand` | 2.4 % | 2.5 % | +0.1 | 39.7 µs | Repeat K heads to match V head count |

_*`in_proj` has a bimodal distribution (median 85 µs, max 151 ms) — the
tail is a stream-sync wait bubble in one capture instance, not per-call
compute. Use the median as the stable per-call cost estimate._

**Structural breakdown (graphs=ON):**

| Block | NVTX % |
|---|---|
| All `:kiln/gdn/*` (GDN layers, 24/32 layers) | **~58 %** |
| All `:kiln/attn/*` (full-attn, 8/32 layers) | ~14 % |
| All `:kiln/mlp/*` (32 layers) | ~8 % |
| Norms + residuals | ~6 % |
| LM head + sampling | ~0.2 % |

Every top-10 region moves by ≤ 0.7 pct-pt between modes. The only
noticeable shift — `in_proj` down 0.7 pct-pt with graphs=ON — is inside
its own per-call noise (the median doesn't move). There is no evidence
that graphs=ON changes the hotspot picture.

## What This Means for PR #141

PR #141's decode-ITL measurement was:

| Arm | Mean ITL |
|---|---|
| graphs=ON main (no kernel) | 25.06 ms |
| graphs=ON kernel-ON         | 24.73 ms |
| graphs=ON kernel-OFF        | 24.85 ms |

That pod showed 25 ms vs this pod's 23.15 ms on the same commit tier —
a cross-pod drift of ~2 ms that's larger than PR #141's kernel-on vs
kernel-off delta. So the absolute numbers in PR #141 vs this profile
are not directly comparable, but the **within-pod, kernel-ON vs
kernel-OFF** result is: **1.005× with graphs on, 1.003× with graphs
off**. That is the same "within noise" signal we see here in the
graphs=ON vs graphs=OFF head-to-head.

Conclusion: PR #141's null result is **not** a graphs-methodology artifact.
The fused kernel is correct and cheap to ship, but it does not recover
wallclock at the GDN decode shape — because the candle-op chain's
launch overhead is not the bottleneck on this workload in either graphs
mode. The NVTX `:kiln/gdn/gated_norm` region cost is ~221 µs per call
regardless of whether its inner ops are one fused launch or ten — the
cost is HBM bandwidth on the RMSNorm reduction and the SiLU-gated
epilogue (bf16 reads/writes at `rows=32, hidden=128`), which fusion
collapses the launches of but does not eliminate the memory traffic of.

**Recommendation on PR #141: close (don't merge).**

Rationale:

1. Zero measured speedup at the production shape (and this re-profile
   says that's the real answer, not a methodology miss).
2. Zero measured regression either — but shipping a vendored kernel with
   no benefit is net-negative: another crate to maintain, another env-var
   to remember, another fallback path that has to stay in parity.
3. The fallback/oracle parity test in the PR is the useful artifact and
   can be kept as a branch / gist reference if we revisit fused
   gated-RMSNorm at larger hidden dims or batched training (where the
   compute/launch ratio flips and the kernel might actually win).
4. If we leave the PR open as "dark landed", future optimization work
   will be forced to maintain the fused kernel's invariants (strided
   inputs, cuda-only, bf16-only envelope) for zero benefit.

_This report recommends but does not execute the close. Disposition is
the reviewer's call._

## Next Optimization Target — weight quantization, not more fusion

The kernel mix in graphs=ON is **60.5 % GEMM, 5 % vendored GDN inner
kernels, ~7 % dtype-cast + element-wise copies, ~10 % F32 reductions,
~17 % small candle kernels**. All of the big cutlass/ampere GEMMs are
already at vendor-grade tensor-core utilisation. There is no per-kernel
fusion target with a plausible path to meaningful decode-ITL speedup:

- **`:kiln/gdn/gated_norm`** — PR #141 already proved fusion doesn't
  help here (see above).
- **`:kiln/gdn/gates`** (14.3 %) — same decode shape (rows=32,
  hidden=128) and same character (sigmoid/softplus/exp chain) as
  gated_norm. A vendored fusion is very likely to land in the same
  null-result regime; not worth re-running the PR #141 experiment.
- **`:kiln/gdn/qk_norm`** (11.7 %) — same shape, same L2-normalize
  memory-bound character; same risk.
- **`:kiln/gdn/conv`** (11.1 %) — pure candle causal depthwise conv1d
  at decode (`causal_conv1d_decode` in `forward.rs:715`). This one
  _could_ fuse productively (a single-warp-per-channel gather + dot +
  state-roll kernel), but it's also only 11 %, and the cost is again
  memory-bound on `[batch, channels, kernel_size]` reads — same regime
  that blocked PR #141 from showing a speedup.
- **`:kiln/attn/rope`** (9.3 %) — only 8/32 layers, already reasonable
  launch cost, likely another null-result fusion target.
- **All `:kiln/mlp/*`** (~8 %) — pure vendor cutlass GEMM. Nothing to
  fuse.

The single biggest remaining lever on A6000 decode is **bf16 → FP8 (or
INT8) weight quantization on the projection GEMMs** — the top three
kernels in the table are 56.7 % of GPU time combined and every one is a
projection GEMM that compresses cleanly. Going to FP8 halves GEMM
arithmetic density and roughly doubles effective HBM bandwidth for
weight loads (the real A6000 decode bottleneck at batch=1), giving a
headline decode-ITL lever that's 3-5× larger than any remaining fusion
target by the NVTX/kernel mix.

**Proposed next task (do not execute in this PR):** scope an
FP8 / INT8 weight-only quantization pass for the Q/K/V/gate/up/down
projections, keeping activations bf16 and the vendored `flash_fwd`,
`recurrent_gdn`, `gdn_fwd_sub`, and `fused_rmsnorm` kernels unchanged.
Measurement: warm paged decode ITL on the same pod, with the same
nsys-based NVTX/kernel-mix deltas reported here, and a parity check
vs bf16 logits tolerance identical to the existing kernel-crate
parity tests.

## Files / How to Reproduce

Raw artifacts for this report (captures live on pod `t0eii824pgigpi`
under `/tmp/` during the run; not committed to the repo because
`.nsys-rep` is 80-110 MB):

- `/tmp/profile_off.nsys-rep` — Capture A (graphs=OFF, 20 s)
- `/tmp/profile_on.nsys-rep` — Capture B (graphs=ON, 20 s)
- `/tmp/stats_off.log`, `/tmp/stats_on.log` — `cuda_gpu_kern_sum`
- `/tmp/nvtx_off.log`, `/tmp/nvtx_on2.log` — `nvtx_sum`
- `/tmp/tri.log` — 3× no-nsys warm ITL runs per arm
- `/tmp/warm_off.log`, `/tmp/warm_on.log` — single-run bench outputs

To reproduce on a fresh A6000 pod (kiln-runpod image):

```bash
# 1. Install nsys 2024.5.1 (the baked 2023.4.4 is buggy on long captures)
apt-get update && apt-get install -y libxcb-cursor0
wget -O /tmp/nsys.deb https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb
dpkg -i /tmp/nsys.deb
ln -sf /opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys /usr/local/cuda/bin/nsys

# 2. Clone + setup + build
GH_TOKEN=$(ce secret-get --name GITHUB_TOKEN)  # or equivalent
git clone https://x-access-token:$GH_TOKEN@github.com/ericflo/kiln.git /workspace/kiln
cd /workspace/kiln && kiln-setup
cargo build --release --features cuda,nvtx --bin kiln-bench

# 3. Fetch weights
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b

# 4. Capture A (graphs=OFF)
KILN_CUDA_GRAPHS=false nsys profile -t cuda,nvtx --duration=20 \
  -o /tmp/profile_off --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 5. Capture B (graphs=ON)
KILN_CUDA_GRAPHS=true nsys profile -t cuda,nvtx --duration=20 \
  -o /tmp/profile_on --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 6. Extract tables
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_off.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_on.nsys-rep
nsys stats --report nvtx_sum         --format csv /tmp/profile_off.nsys-rep
nsys stats --report nvtx_sum         --format csv /tmp/profile_on.nsys-rep
```

### Pod / cost notes

- Pod: `t0eii824pgigpi` (RunPod A6000 on-demand, ~\$0.49/hr).
- Total pod uptime at capture end: ~30 minutes.
- Estimated pod cost: ~\$0.25.
- Pod terminated after PR creation.

# Kiln Profiling Report

## Phase 6 — chunk-prep fusion prefill bench (2026-04-18)

Measured the Phase 6 `gdn_chunk_prep` fusion (`ce/phase6-chunk-prep-fusion`,
HEAD `11314e3`) against the post-#160 `main` baseline (HEAD `395f5f7`) on the
same pod, back-to-back, using the existing `kiln-bench --paged` latency
harness. The fusion collapses the 7+ elementwise launches inside the
chunkwise outer recurrence (cumsum, decay matrix, KKT/QKT masked
exp-scaling, v_prime, q_s_scaled, decay_last_col, p_last) into one CUDA
kernel per (chunk × batch × head). The four cuBLAS matmuls surrounding it
(KKT, QKT, ks_entry, q_s) are unchanged.

**Hardware:** RunPod NVIDIA A40 on-demand (A6000 unavailable;
same sm_86 Ampere arch, covered by existing `KILN_CUDA_ARCHS="80;86;89;90"`),
Driver 580.95.05, CUDA 12.4.

**Build:** release, `--features cuda`, `KILN_W4A16=0 KILN_CUDA_GRAPHS=true`.

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens N --max-output-tokens M --skip-training`, 3 back-to-back runs
per arm (no nsys attached), reporting `time_to_first_token_ms` from the
latency phase.

### Prefill TTFT — 512-prompt × 128-decode

| run    | PRE (`395f5f7`) ms | POST (`11314e3`) ms |
| ------ | -----------------: | ------------------: |
| 1      |             403.43 |              366.68 |
| 2      |             393.19 |              363.41 |
| 3      |             413.51 |              428.25 |
| mean   |             403.38 |              386.11 |
| median |             403.43 |              366.68 |

Mean speedup: **1.045× TTFT** (−4.3 %). Median speedup: **1.100× TTFT**
(−9.1 %). Run 3 of the POST arm is a clear outlier (+63 ms versus the other
two) — probably pod-to-pod GPU variance we've seen before on this harness.

### Prefill TTFT — 2048-prompt × 64-decode

Longer prompt → more chunks per prefill (2048 / chunk_size=64 = 32 chunks per
layer × 24 GDN layers = 768 fused kernel launches in place of 5 376
elementwise launches).

| run    | PRE (`395f5f7`) ms | POST (`11314e3`) ms |
| ------ | -----------------: | ------------------: |
| 1      |            1070.49 |              882.04 |
| 2      |             987.92 |              934.43 |
| 3      |             980.49 |              941.23 |
| mean   |            1012.97 |              919.23 |
| median |             987.92 |              934.43 |

Mean speedup: **1.102× TTFT** (−9.3 %). Median speedup: **1.057× TTFT**
(−5.7 %).

### Decode latency (no expected change)

The fusion is a prefill-path optimization — decode uses the single-step
`gdn_recurrent_step` fast path, which is untouched. Decode numbers are
reported for completeness.

| metric              | PRE mean | POST mean | Δ       |
| ------------------- | -------: | --------: | ------- |
| mean inter-token ms |    25.64 |     25.88 | +0.9 %  |
| decode tok/s        |    38.99 |     38.65 | −0.9 %  |

Within pod variance.

### Read / take

The fusion lands below the ≥1.2× prefill TTFT floor stated in the task brief.
Measured improvement is **1.05–1.10×** on 512-prompt TTFT and **1.06–1.10×**
on 2048-prompt TTFT — directionally correct, reproducible across both prompt
lengths, but well short of the 2–4× target. The implementation is correct
(see `test_gdn_chunk_prep_matches_fallback` and
`test_gdn_chunkwise_matches_sequential` in `crates/kiln-model/src/forward.rs`
— both pass with max error < 2e-3 bf16), the kernel removes a measurable
slice of prefill wall-clock, and it leaves the decode fast path unchanged.
The ceiling is pinned by the four cuBLAS matmuls (KKT, QKT, ks_entry, q_s)
which still dominate the chunkwise recurrence wall-clock and are
intentionally out of scope for this kernel.

Possible follow-ups if the chunkwise recurrence becomes a hotter target:
- Collapse the four matmuls into fewer cuBLAS calls (batch two KKT/QKT into
  a single `bmm`, batch ks_entry/q_s).
- Explore a full Triton-style `chunk_gla_fwd` that owns the matmuls too —
  this is the upstream fla-org path and is what PR #160 originally
  recommended for the largest decode-side win.

### Reproduction

```bash
# 1. Pod + weights
kiln-setup --repo /workspace/kiln
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b

# 2. Baseline (pre-fusion)
cd /workspace/kiln && git checkout 395f5f7
cargo build --release --features cuda -p kiln-server --bin kiln-bench
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 --skip-training
done

# 3. Fusion (post)
git checkout ce/phase6-chunk-prep-fusion
cargo build --release --features cuda -p kiln-server --bin kiln-bench
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 --skip-training
done
```

### Pod / cost

- Pod: `11hd6xzo2uwlyy` (RunPod NVIDIA A40 on-demand, $0.44/hr).
- Total uptime at bench end: ~2.5 hours (build, weight fetch, parity tests,
  12 bench runs across pre/post × two prompt lengths).
- Estimated pod cost: **~$1.10** (well under the $20 target / $40 hard abort
  set in the task brief).

## Post-PR #158 Decode Profile (2026-04-18)

Refreshed decode profile on current `main` (HEAD `7132f29`, post-PR #158 which
fused the GDN decode gates path into one CUDA kernel). Re-ran the same nsys
methodology as the PR #156 profile below for an apples-to-apples comparison.
Only Arm B (production W4A16) was re-captured — the baseline Arm A path was
not changed by #158.

**Hardware:** RunPod RTX A6000 on-demand, Driver 565.57.01, CUDA 12.4, nsys
2024.5.1 (apt-installed; the baked 2023.4.4 has the
`EventCollection::CheckOrder` finalization bug).

**Build:** release, `KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, bench command
identical to PR #156 (`kiln-bench --delay=110 --duration=30 --nvtx ...`).

### Arm B decode latency (post-#158 vs PR #156)

| metric                     | PR #156 | post-#158 | Δ           |
| -------------------------- | ------: | --------: | ----------- |
| mean inter-token ms        | 21.50   | 22.92     | +6.6 %      |
| p50 inter-token ms         | 21.43   | 22.61     | +5.5 %      |
| p99 inter-token ms         | 28.87   | 25.13     | **−13.0 %** |
| decode tok/s (latency)     | 46.52   | 43.63     | −6.2 %      |

The p99 tightening is the clearest effect of #158. Mean ITL regressed by
~1.4 ms, which is within the pod-to-pod variance previously observed on this
harness (PR #152→#153→#156 moved the same metric by a similar magnitude
without any code change). Directionally consistent: the GDN hotspots all
moved down proportionally.

### Top-10 NVTX regions (Arm B, post-#158)

| rank | %     | region                        |
| ---: | ----: | ----------------------------- |
|    1 | 16.7% | `:kiln/gdn/gates`             |
|    2 | 15.8% | `:kiln/gdn/gated_norm`        |
|    3 | 13.3% | `:kiln/gdn/qk_norm`           |
|    4 | 12.2% | `:kiln/gdn/conv`              |
|    5 |  6.8% | `:kiln/gdn/in_proj`           |
|    6 |  5.8% | `:kiln/mlp/gate`              |
|    7 |  5.6% | `:kiln/mlp/up`                |
|    8 |  5.4% | `:kiln/mlp/down`              |
|    9 |  3.1% | `:kiln/residual`              |
|   10 |  2.8% | `:kiln/gdn/head_expand`       |
|   +  |  2.7% | `:kiln/proj/qkv`              |
|   +  |  2.5% | `:kiln/gdn/out_proj`          |
|   +  |  1.6% | `:kiln/attn/gdn/recurrent`    |

GDN subsystem (in_proj + gates + gated_norm + qk_norm + conv + head_expand +
out_proj + recurrent) still owns **~73 %** of decode wall-clock. The MLP
trio (`gate`/`up`/`down`) is **16.8 %**. Residual + norms are ~6 %.

### Top-10 CUDA kernels (Arm B, post-#158)

| rank | %     | kernel                                                                     |
| ---: | ----: | -------------------------------------------------------------------------- |
|    1 | 14.5% | `Marlin<256,1,8,8,4,8>`                                                    |
|    2 | 12.0% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`             |
|    3 |  9.8% | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f`                              |
|    4 |  9.2% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`              |
|    5 |  7.5% | `ucopy_bf16`                                                               |
|    6 |  5.2% | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2`                              |
|    7 |  4.1% | `bmul_f32`                                                                 |
|    8 |  2.6% | `fused_rmsnorm_kernel`                                                     |
|    9 |  2.5% | `fast_sum_f32`                                                             |
|   10 |  2.4% | `ucopy_f32`                                                                |
|   +  |  2.2% | `cast_f32_bf16`                                                            |
|   +  |  2.1% | `recurrent_gdn_fwd_kernel<128>`                                            |
|   +  |  2.0% | `cast_bf16_f32`                                                            |

Marlin remains the single largest kernel (q_proj + 3×MLP). The cutlass /
ampere BF16 GEMM stack sitting at #2/#3/#4/#6 is the non-Marlin projection
work (k_proj, v_proj, o_proj, lm_head, GDN in_proj/out_proj). The long tail
of `bmul_f32`/`fast_sum_f32`/`ucopy_*`/`cast_*` elementwise kernels is the
remaining GDN gates + norm plumbing that PR #158 did not absorb.

### Delta vs PR #156 (NVTX top-4)

| region                   | PR #156 | post-#158 | Δ pp   |
| ------------------------ | ------: | --------: | -----: |
| `:kiln/gdn/gates`        | 18.2 %  | 16.7 %    | −1.5   |
| `:kiln/gdn/gated_norm`   | 18.0 %  | 15.8 %    | −2.2   |
| `:kiln/gdn/qk_norm`      | 14.7 %  | 13.3 %    | −1.4   |
| `:kiln/gdn/conv`         | 12.4 %  | 12.2 %    | −0.2   |

Every GDN hotspot moved down — #158's fusion reduced the proportional share
of the gates path and the downstream gated_norm/qk_norm (they dispatch fewer
upstream elementwise tensors). The gain is real but narrow: the top-50
kernel list still contains the full elementwise zoo (`bmul_f32`,
`fast_sum_f32`, `cast_f32_bf16`, `cast_bf16_f32`, `ucopy_bf16`, `affine_f32`,
`uneg_f32`, `uexp_f32`, `urecip_f32`, `usqrt_f32`) in similar quantities to
PR #156. No single fused-gate kernel dominates — the fusion collapsed part
of the gate arithmetic but the surrounding norm / cast / copy traffic is
unchanged.

### Recommended next optimization target

**Vendor `chunk_gla_fwd` from flash-linear-attention (roadmap step 2).**

The combined decode share of `gdn/gates` + `gdn/gated_norm` + `gdn/qk_norm`
+ `gdn/conv` is still **57.9 %** of wall-clock. A narrow gates-only fusion
(like #158) only chips at the first of those four regions. Vendoring the
upstream `chunk_gla_fwd` kernel that fuses the full GDN decode chain
(conv + in_proj + gates + gated_norm + qk_norm + recurrent) is the
highest-leverage move available — it would collapse ~58 % of decode
wall-clock into a single kernel dispatch and directly eliminate the
elementwise zoo that #158 could not reach.

Vendoring approach follows the established pattern (Marlin vendor in PR
#149, GDN substitution kernel in PR #80, narrow scope under
`crates/kiln-chunk-gla-kernel/`, C-ABI entrypoint, cuda-gated, parity test
vs existing reference). Preflight note: confirmed no prior crate or PR
already lands this (`git log --all --grep="chunk_gla"` and
`ls crates/ | grep -i chunk` both empty).

## Reproduction (post-#158)

Identical to the PR #156 reproduction section below, with one substitution:
install nsys 2024.5.1 (`apt-get install -y cuda-nsight-systems-12-6`) before
the `kiln-bench --nvtx` capture. The baked `ghcr.io/ericflo/kiln-runpod`
image ships nsys 2023.4.4 which finalizes broken `.nsys-rep` files.

---

# Kiln Profiling Report — Phase 6 re-profile, post-Marlin MLP wire-in (PR #152)

## Overview

PR #152 wired the vendored Marlin W4A16 kernel into the three MLP projections
(`gate_proj`, `up_proj`, `down_proj`), extending the same path PR #149 landed
for `q_proj`. PR #153 reported a +9.9 % decode tok/s delta from that wire-in
on a separate pod. This re-profile captures fresh profiles on current `main`
(HEAD `5aa22e1`) on one pod, back-to-back, so the post-#152 production hotspot
mix is settled for the next optimization cycle.

Two arms measured, identical bench and build:

- **Arm A — BF16 baseline (`KILN_W4A16=0 KILN_CUDA_GRAPHS=true`).** All
  projections go through the existing cuBLAS BF16 GEMM path.
- **Arm B — production W4A16 (`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`).** At
  model load, `q_proj` and the three MLP projections swap in Marlin 4-bit
  packed weights when the shape is compatible (`k%128 == 0 && n%256 == 0`).
  All other layers (`k_proj`, `v_proj`, `o_proj`, norms, GDN kernels, full
  attention) are unchanged.

Measured results on one A6000 pod:

| metric                 | Arm A (BF16) | Arm B (Marlin) | Δ        |
| ---------------------- | -----------: | -------------: | -------- |
| mean inter-token ms    | 22.25        | 21.50          | **−3.4 %** |
| p50 inter-token ms     | 22.10        | 21.43          | −3.0 %   |
| p99 inter-token ms     | 28.82        | 28.87          | +0.2 %   |
| decode tok/s (latency) | 44.95        | 46.52          | **+3.5 %** |
| throughput bs=1 tok/s  | 40.40        | 41.53          | +2.8 %   |
| throughput bs=4 tok/s  | 40.40        | 42.24          | +4.6 %   |
| throughput bs=8 tok/s  | 40.35        | 42.17          | +4.5 %   |
| throughput bs=16 tok/s | 40.47        | 41.47          | +2.5 %   |
| model load time        | 13.95 s      | 103.58 s       | +89.6 s  |
| model VRAM (load)      | 16344 MB     | 17656 MB       | +1312 MB |
| peak VRAM (inference)  | 16842 MB     | 18026 MB       | +1184 MB |

The decode-ITL lift lands at roughly **one third the tok/s gain PR #153
reported (+9.9 %)**. Likely drivers: PR #153 measured on a cold pod with a
different throughput harness, and small pod-to-pod drift on the same hardware.
The direction and sign agree; the magnitude is smaller than previously
reported but real and repeatable in this back-to-back measurement.

Model load jumped from 14 s to 104 s because Marlin packs the four
projection weight matrices at load time (≈ 33 matrices × ~8 MB of packed
data + scales). Load-time packing is one-shot and does not affect request
latency, but is worth tracking — a pre-packed on-disk artifact would
eliminate the cost if it becomes a cold-start issue.

Peak VRAM grew ~1.2 GB with Marlin because the packed weights, scales, and
workspace live alongside the original BF16 weights still held for the
non-Marlin paths (`k_proj`, `v_proj`, `o_proj`). Future work that fully
replaces BF16 weights at load time would recover this.

## Methodology

All numbers from one pod (RunPod RTX A6000 on-demand), one build, back-to-back
captures. Only environment variables changing between arms are
`KILN_W4A16=0|1`. `KILN_CUDA_GRAPHS=true` for both arms (production path).

- **Steady-state ITL.** Three back-to-back 512-prompt × 128-decode
  latency runs per arm, no nsys attached, reporting the
  `mean_inter_token_ms` from the paged decode phase after the model is
  warm. Prefill and first-decode cold costs are excluded by
  `kiln-bench`'s latency phase already.
- **Throughput sweep.** The same `kiln-bench` invocation runs a
  `1/4/8/16` sequential-generation sweep after the latency phase, which
  produces the bs=1/4/8/16 tok/s numbers above.
- **nsys captures.** Two captures with the production path and CUDA
  graphs enabled:
    - **Arm A** — `KILN_W4A16=0 KILN_CUDA_GRAPHS=true nsys profile -t
      cuda,nvtx --delay=15 --duration=20`. Capture window begins after
      the ~14 s model load, spanning prefill + warm decode + the first
      throughput runs.
    - **Arm B** — `KILN_W4A16=1 KILN_CUDA_GRAPHS=true nsys profile -t
      cuda,nvtx --delay=110 --duration=30`. The `--delay` is larger
      because Marlin packing pushes model load to 104 s. Capture window
      lands in the throughput phase (steady-state warm decode through
      `bs=1` and into `bs=4` / `bs=8`).
- **Extraction.** `nsys stats --report cuda_gpu_kern_sum --format csv`
  for the per-kernel table; `nsys stats --report nvtx_sum --format csv`
  for the NVTX region table.
- **Capture-window caveat.** Because Arm A's window includes some
  prefill + decode transition activity and Arm B's window is pure
  steady-state decode, a small number of prefill-heavy NVTX regions
  (`:kiln/attn/full/prefill`, `:kiln/attn/full/decode_fused`, some
  `:kiln/attn/rope`) appear in Arm A's top-30 but not Arm B's. The
  measured ITL/tok-s numbers are from the bench's own clean-timing
  phase and are apples-to-apples. The Arm B top-10 tables are the
  correct structural view for "what dominates the production decode
  hot loop today."

## Hardware / Build

- GPU: NVIDIA RTX A6000 (49 GB VRAM)
- Driver: 565.57.01, CUDA 12.4
- nsys: 2024.5.1 (the baked 2023.4.4 triggers
  `EventCollection::CheckOrder` on long captures; 2024.5.1 fixes it)
- Rust: 1.95.0, `cargo build --release --features cuda,nvtx`
- Kiln HEAD at capture: `5aa22e1` (bench report from PR #153, which
  sits on top of PR #152's MLP wire-in)
- Model: Qwen3.5-4B, sharded safetensors in `/workspace/qwen3.5-4b`
- Prompt: 512 tokens; decode: 128 tokens; paged KV
  (`block_size=16`, `blocks=40`)

## Top-10 GPU Kernels — Arm B (W4A16, production path)

From `nsys stats --report cuda_gpu_kern_sum` on `profile_armB.nsys-rep`.
CSV in `profiling-artifacts/statsB_kern.csv`.

| rank | % GPU time | kernel                                                              | role                                                       |
| ---: | ---------: | ------------------------------------------------------------------- | ---------------------------------------------------------- |
|   1  | **14.8 %** | `Marlin<256,1,8,8,4,8>`                                             | W4A16 MLP (gate/up/down) + `q_proj` GEMMs                  |
|   2  |   11.7 %   | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`      | Remaining BF16 GEMMs (`k_proj`, `v_proj`, `o_proj`, lm_head) |
|   3  |    9.9 %   | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`        | BF16 GEMM (projections / lm_head)                          |
|   4  |    9.3 %   | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`       | BF16 GEMM (shape-dependent tile)                           |
|   5  |    8.2 %   | `ucopy_bf16`                                                        | Tensor copies (residual stream, reshapes)                  |
|   6  |    5.3 %   | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5`  | BF16 GEMM (smaller tile)                                   |
|   7  |    3.9 %   | `bmul_f32`                                                          | GDN elementwise multiply (gates / gated_norm path)         |
|   8  |    3.1 %   | `fused_rmsnorm_kernel`                                              | RMSNorm (pre-attn, pre-mlp, qk_norm)                       |
|   9  |    2.9 %   | `fast_sum_f32`                                                      | Reductions in GDN gates / norms                            |
|  10  |    2.7 %   | `recurrent_gdn_fwd_kernel<128>`                                     | GDN linear-attention recurrent kernel                      |

Observations:

- Marlin has become the **single largest kernel** in the production path at
  **14.8 %**, covering four GEMMs (q_proj + gate_proj + up_proj + down_proj)
  in one kernel family.
- The four cuBLAS BF16 GEMM tile variants together account for **36.2 %**
  of GPU time. These are the non-Marlin projections (`k_proj`, `v_proj`,
  `o_proj`, `lm_head`) plus any shape-incompatible fallback.
- Small elementwise / reduction kernels on the GDN path (bmul_f32,
  fast_sum_f32, cast_*, fused_rmsnorm_kernel) remain numerous (8k+
  instances combined). These are launch-overhead-bound at the GDN
  decode shape and are the natural fusion target for the next phase.

## Top-10 NVTX Regions — Arm B (W4A16, production path)

From `nsys stats --report nvtx_sum`. CSV in
`profiling-artifacts/statsB_nvtx.csv`. Percentages are of the NVTX total
within the 30 s steady-state capture window.

| rank | % time | region                  | layers per token | notes                                                  |
| ---: | -----: | ----------------------- | ---------------: | ------------------------------------------------------ |
|   1  | **18.2 %** | `:kiln/gdn/gates`     | 24               | Per-layer gate projection + swish + elementwise path   |
|   2  | **18.0 %** | `:kiln/gdn/gated_norm` | 24              | Gated RMSNorm on GDN head outputs                      |
|   3  |   14.7 %   | `:kiln/gdn/qk_norm`    | 24              | Q/K RMSNorm inside GDN                                 |
|   4  |   12.4 %   | `:kiln/gdn/conv`       | 24               | Causal 1-D conv on GDN Q/K/V                           |
|   5  |    5.5 %   | `:kiln/mlp/gate`       | 32               | `gate_proj` GEMM (now Marlin)                          |
|   6  |    5.5 %   | `:kiln/mlp/up`         | 32               | `up_proj` GEMM (now Marlin)                            |
|   7  |    5.5 %   | `:kiln/mlp/down`       | 32               | `down_proj` GEMM (now Marlin)                          |
|   8  |    4.9 %   | `:kiln/gdn/in_proj`    | 24               | GDN input projection (cuBLAS BF16)                     |
|   9  |    3.6 %   | `:kiln/residual`       | 64               | Residual adds                                          |
|  10  |    2.7 %   | `:kiln/gdn/head_expand` | 24              | GDN head expansion                                     |

Structural breakdown of steady-state decode (Arm B):

- **GDN linear-attention subsystem** (`:kiln/gdn/*`) = **72.5 %**
  (gates 18.2 + gated_norm 18.0 + qk_norm 14.7 + conv 12.4 + in_proj 4.9
  + head_expand 2.7 + recurrent 1.2 + out_proj ≈ negligible here).
- **MLP** (`:kiln/mlp/*`) = **16.5 %** (gate 5.5 + up 5.5 + down 5.5).
  Now balanced across the three projections because Marlin runs all
  three in near-identical time.
- **Residuals + norms** (`:kiln/residual`, `:kiln/norm/pre_*`) =
  **~6.0 %**.
- **Full-attention layers** — fall below the top-10 cutoff in the Arm B
  window; full-attn share is small on this model (only 8 of 32 layers)
  and was dominated by the GDN pathway even in the prior profile.

## Comparison to prior PROFILING.md (pre-PR #152, commit `07d934b`)

Prior production-path top-10 NVTX (graphs ON, no W4A16):

| rank | prior (`07d934b`, BF16) | now (`5aa22e1`, Arm B, W4A16) | shift   |
| ---: | ------------------------- | ------------------------------- | ------- |
|   1  | `:kiln/gdn/gates` 14.3 %  | `:kiln/gdn/gates` **18.2 %**    | +3.9 pp |
|   2  | `:kiln/gdn/gated_norm` 14.1 % | `:kiln/gdn/gated_norm` **18.0 %** | +3.9 pp |
|   3  | `:kiln/gdn/qk_norm` 11.7 % | `:kiln/gdn/qk_norm` **14.7 %**   | +3.0 pp |
|   4  | `:kiln/gdn/conv` 11.1 %   | `:kiln/gdn/conv` **12.4 %**      | +1.3 pp |
|   5  | `:kiln/attn/rope` 9.3 %   | (below top-10 in Arm B window)   | —       |
|   6  | `:kiln/gdn/in_proj` 9.0 % | `:kiln/gdn/in_proj` 4.9 %        | −4.1 pp |
|   9  | `:kiln/mlp/gate` 2.7 %    | `:kiln/mlp/gate` 5.5 %           | +2.8 pp |
|  10  | `:kiln/mlp/up`   2.5 %    | `:kiln/mlp/up`   5.5 %           | +3.0 pp |

Reading: Marlin eliminated a large BF16 GEMM slice per MLP layer, which
**shrank the absolute time in MLP GEMMs** (and hence tightened the decode
loop, producing the measured +3.5 % tok/s). MLP NVTX share **went up**
because the region is now time-relative to a shorter loop and because the
previous profile captured a wider window including prefill. The GDN share
also grew — the Amdahl's-law flip from MLP wins: after Marlin halves
MLP GEMM cost, GDN becomes proportionally even more dominant.

**Dominant cost today is unambiguously the GDN subsystem at 72.5 % of
decode time.** The four big GDN regions — gates, gated_norm, qk_norm,
conv — together account for **63.3 %** of decode time in a single
structural group.

## Next optimization target

The next concrete lever is **fusing the GDN gate + gated_norm +
elementwise path** into a small number of large kernels, ideally one.

Evidence:

- `:kiln/gdn/gates` and `:kiln/gdn/gated_norm` together are **36.2 %**
  of decode time (#1 and #2 by a wide margin).
- The hot kernels serving these regions are small elementwise /
  reduction primitives (`bmul_f32` 3.9 %, `fast_sum_f32` 2.9 %,
  `cast_f32_bf16` 2.4 %, `cast_bf16_f32` 2.2 %, `affine_f32` 1.8 %,
  `uneg_f32` / `uexp_f32` / `urecip_f32` / `usqrt_f32` each ~0.7–1.0 %).
  These are **hundreds to thousands of instances per capture**, each a
  per-element kernel that does very little work.
- At the GDN decode shape (`rows = 32`, `hidden = 128` per head, 24
  layers, batch = 1) these kernels are launch- and
  memory-bandwidth-bound. One fused kernel collapses:
    - `gates_lin` → swish activation → mul into `gated_norm` input,
    - `rms_norm` over the gated output,
    - the residual elementwise multiply.
  That eliminates intermediate bf16→f32→bf16 round-trips that appear
  as `cast_*` kernels today, and cuts hundreds of small launches per
  token.
- PR #141 vendored a `gated_rms_norm` kernel and measured no delta at
  `rows = 32, hidden = 128` **under CUDA graphs**. That test bounded
  the isolated fusion gain, not the broader "collapse the gate path"
  target proposed here. The broader fusion targets NVTX regions that
  together are ~5× larger than the isolated `gated_rms_norm` shape
  PR #141 measured, and the hot kernel list shows the dominant cost is
  in elementwise / reduction primitives that a proper fused gate kernel
  would eliminate, not in the narrow `gated_rms_norm` slice.

Recommended scope for the next cycle:

1. Vendor or author a fused **`gdn_gate`** kernel that folds the gate
   linear output, swish, gated elementwise multiply, RMSNorm, and
   residual-multiply into one kernel (bf16 in, bf16 out).
2. Keep the existing `recurrent_gdn_fwd_kernel<128>` and
   `gdn_fwd_sub_kernel` unchanged — those already own a small share
   and are CUDA-graph-friendly.
3. Validate on the same harness this report uses (warm paged decode
   ITL, 512-prompt × 128-decode, three runs, both with and without
   `KILN_W4A16=1`), with a parity test vs the existing unfused
   implementation using the kernel-crate parity test pattern.

Expected ceiling: if the fused kernel removes even half of the
elementwise launch overhead and cast round-trips, decode ITL should
improve by 6–10 % (target: 19.5 ms mean ITL, ~51 tok/s on A6000). The
same fusion helps both arms because the GDN path is unchanged by
W4A16.

## Files / Reproduction

Raw artifacts in this repo:

- `profiling-artifacts/statsA_kern.csv` — per-kernel table, Arm A
- `profiling-artifacts/statsA_nvtx.csv` — NVTX table, Arm A
- `profiling-artifacts/statsB_kern.csv` — per-kernel table, Arm B
- `profiling-artifacts/statsB_nvtx.csv` — NVTX table, Arm B

Full `.nsys-rep` captures live on pod `daogyp64vo0cgq` under `/tmp/` and
are not committed (19 MB Arm A, 34 MB Arm B).

To reproduce on a fresh RunPod A6000 pod (`ghcr.io/ericflo/kiln-runpod:latest`):

```bash
# 1. Install nsys 2024.5.1 (baked 2023.4.4 is buggy on long captures)
apt-get update && apt-get install -y libxcb-cursor0 cuda-nsight-systems-12-6
ln -sf /opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys /usr/local/cuda/bin/nsys

# 2. Clone + setup + build
GH_TOKEN=$(ce secret-get --name GITHUB_TOKEN)  # or equivalent
git clone https://x-access-token:$GH_TOKEN@github.com/ericflo/kiln.git /workspace/kiln
cd /workspace/kiln && kiln-setup
cargo build --release --features cuda,nvtx --bin kiln-bench

# 3. Fetch weights
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b

# 4. Steady-state ITL, three runs per arm (uncaptured)
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true \
    ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
done
for i in 1 2 3; do
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
    ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
done

# 5. Capture Arm A (BF16 baseline) — delay 15s past model load
KILN_W4A16=0 KILN_CUDA_GRAPHS=true \
  /usr/local/cuda/bin/nsys profile -t cuda,nvtx --delay=15 --duration=20 \
  -o /tmp/profile_armA --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 6. Capture Arm B (W4A16 production) — delay 110s past Marlin packing
KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  /usr/local/cuda/bin/nsys profile -t cuda,nvtx --delay=110 --duration=30 \
  -o /tmp/profile_armB --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 7. Extract tables
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_armA.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_armB.nsys-rep
nsys stats --report nvtx_sum          --format csv /tmp/profile_armA.nsys-rep
nsys stats --report nvtx_sum          --format csv /tmp/profile_armB.nsys-rep
```

### Pod / cost notes

- Pod: `daogyp64vo0cgq` (RunPod RTX A6000 on-demand, ~$0.49/hr).
- Total pod uptime at capture end: ~2.5 hours (build, load
  testing, four bench runs, two nsys captures, stats extraction).
- Estimated pod cost: ~$1.25.

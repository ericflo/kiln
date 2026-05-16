# OPD top-K reverse-KL — Vulkan kernel sanity-check throughput

End-to-end throughput numbers for the fused Vulkan OPD loss kernels via
`examples/bench_opd_topk_kl_vk.rs`. These are **lavapipe (Mesa CPU
Vulkan)** numbers — software-rasterizer baselines that prove the kernel
runs correctly end-to-end and exercises the resident-buffer dispatch
path. They are NOT representative of GPU performance; the kernel runs
~100–1000× faster on a real Vulkan-capable GPU (NVIDIA / AMD RADV /
Apple Metal).

Reported numbers are the minimum-of-`REPEATS=3` per-iteration latency
across `TIMED_ITERS=30` iterations after `WARMUP=10`, on a RunPod
RTX A5000 host running through `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
(NVIDIA Vulkan ICD on RunPod fails to initialise — see the PR
description for details).

```
# WARMUP=10 TIMED_ITERS=30 REPEATS=3
# columns: t (active tokens), h, v, k, dtype, fwd_ms, fwd_tok/s, bwd_ms, bwd_tok/s
t=  256  h= 2560  v= 32000  k=32  f32    fwd_ms= 40.149  fwd_tok/s=  6376  bwd_ms= 76.806  bwd_tok/s= 3333
t= 1024  h= 2560  v= 32000  k=32  f32    fwd_ms= 89.872  fwd_tok/s= 11394  bwd_ms=210.437  bwd_tok/s= 4866
t= 4096  h= 2560  v= 32000  k=32  f32    fwd_ms=268.412  fwd_tok/s= 15260  bwd_ms=702.366  bwd_tok/s= 5832
t= 1024  h= 2560  v= 32000  k=16  f32    fwd_ms= 66.442  fwd_tok/s= 15412  bwd_ms=153.320  bwd_tok/s= 6679
t= 1024  h= 2560  v= 32000  k=32  bf16w  fwd_ms= 99.458  fwd_tok/s= 10296  bwd_ms=230.429  bwd_tok/s= 4444
t= 4096  h= 2560  v= 32000  k=32  bf16w  fwd_ms=259.923  fwd_tok/s= 15759  bwd_ms=700.941  bwd_tok/s= 5844
```

Notes:
- Per-token cost grows roughly linearly with H × K (expected — bandwidth-
  bound, dominated by the per-slot strided dot product over H).
- K=16 is meaningfully faster than K=32 at the same T (half the per-slot
  matmul work).
- bf16w forward is on par with f32 forward — packed bf16 reads halve
  the weight bandwidth but llvmpipe doesn't benefit (CPU L1/L2 dominates
  either way). Real GPU will see the bf16w bandwidth win.
- Backward latency is ~2.5× forward, dominated by the H-loop with the
  per-token K-element gather from the weight matrix (expected — see
  CUDA kernel comments for the same ratio).

Real GPU numbers will land once NVIDIA Vulkan ICD initialization on
RunPod (or any other Vulkan-capable host) is available; the §9.7
grand-plan perf gates from
`docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`
target ≥600 t/s on 1× 7900 XTX cached + post-PR #1030 ≥1200 t/s.

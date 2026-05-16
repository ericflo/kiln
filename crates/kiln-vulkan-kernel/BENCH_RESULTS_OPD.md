# OPD top-K reverse-KL — Vulkan kernel throughput

End-to-end forward + analytic-backward throughput for the fused Vulkan
OPD loss kernels via `examples/bench_opd_topk_kl_vk.rs`.

Reported: minimum-of-`REPEATS=3` per-iteration latency across
`TIMED_ITERS=30` iterations after `WARMUP=10`. Hidden=2560 matches the
Qwen3.5-4B student; vocab=32K (subset that fits comfortably on A6000).

## NVIDIA RTX A6000 — Vulkan (driver 565.57.01, Vulkan 1.3.289)

RunPod A6000 host, `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
image with `NVIDIA_DRIVER_CAPABILITIES=all` and the LunarG Vulkan SDK
1.3.290 installed (Ubuntu 22.04's default `libvulkan1` 1.3.204 is too
old for the driver's 1.3.289 ICD and `vk_icdGetInstanceProcAddr` returns
NULL for `vkCreateInstance` — see the PR description for the diagnosis).

```
# WARMUP=10 TIMED_ITERS=30 REPEATS=3
# t (active tokens), h=2560, v=32000, k, dtype, fwd_ms, fwd_tok/s, bwd_ms, bwd_tok/s
t=  256  k=32  f32    fwd_ms= 0.366  fwd_tok/s=   700_113  bwd_ms= 0.500  bwd_tok/s=  511_930
t= 1024  k=32  f32    fwd_ms= 1.256  fwd_tok/s=   814_989  bwd_ms= 1.699  bwd_tok/s=  602_578
t= 4096  k=32  f32    fwd_ms= 4.455  fwd_tok/s=   919_400  bwd_ms= 5.803  bwd_tok/s=  705_815
t= 1024  k=16  f32    fwd_ms= 0.587  fwd_tok/s= 1_745_227  bwd_ms= 0.910  bwd_tok/s= 1_125_822
t= 1024  k=32  bf16w  fwd_ms= 0.714  fwd_tok/s= 1_434_377  bwd_ms= 1.102  bwd_tok/s=  928_942
t= 4096  k=32  bf16w  fwd_ms= 2.425  fwd_tok/s= 1_689_218  bwd_ms= 3.341  bwd_tok/s= 1_226_094
```

Observations:
- **bf16w gives a clean ~2× over f32** on both fwd and bwd at K=32, T∈{1024, 4096} — packed bf16 reads halve the weight bandwidth and the kernel is bandwidth-bound on the per-slot strided weight reads (as the CUDA reference noted: `For T=4096, K=32, head reads dominate at ~670 MiB scattered`).
- **K=16 doubles fwd tok/s vs K=32** at the same T (half the per-slot work).
- **Backward is consistently ~1.4× slower than forward** (CUDA reference observed the same ratio — the H-loop with per-token K-element gather dominates).
- §9.7 grand-plan perf-gate targets `≥600 t/s cached / ≥250 t/s local Q4 teacher` on 1× 7900 XTX Vulkan — the loss kernel itself, isolated, is two orders of magnitude above that, so it won't be the bottleneck even in the full OPD-trainer wall-clock.

## Mesa lavapipe (CPU software Vulkan, RunPod RTX A5000 host)

Software-rasteriser baseline that exercises the resident-buffer
dispatch path on hosts where no GPU Vulkan ICD is available. Used as
the portable parity-validation target.

```
t=  256  k=32  f32    fwd_ms= 40.149  fwd_tok/s=  6_376  bwd_ms= 76.806  bwd_tok/s= 3_333
t= 1024  k=32  f32    fwd_ms= 89.872  fwd_tok/s= 11_394  bwd_ms=210.437  bwd_tok/s= 4_866
t= 4096  k=32  f32    fwd_ms=268.412  fwd_tok/s= 15_260  bwd_ms=702.366  bwd_tok/s= 5_832
t= 1024  k=16  f32    fwd_ms= 66.442  fwd_tok/s= 15_412  bwd_ms=153.320  bwd_tok/s= 6_679
t= 1024  k=32  bf16w  fwd_ms= 99.458  fwd_tok/s= 10_296  bwd_ms=230.429  bwd_tok/s= 4_444
t= 4096  k=32  bf16w  fwd_ms=259.923  fwd_tok/s= 15_759  bwd_ms=700.941  bwd_tok/s= 5_844
```

NVIDIA-vs-lavapipe ratio: **~60× fwd / ~120× bwd at T=4096, K=32, f32**.
The lavapipe numbers are not perf claims — they exist to gate
correctness on a portable software Vulkan implementation.

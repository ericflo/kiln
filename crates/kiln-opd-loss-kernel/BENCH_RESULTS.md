# `kiln-opd-loss-kernel` benchmark results

§9.7 of `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`
gates shipping each engine on per-engine perf targets. This file captures
the validated numbers from the A6000 pod, against the canonical
production shape **H=2560, V=32000, K=32**.

## Methodology

- `crates/kiln-opd-loss-kernel/examples/bench_opd_topk_kl.rs` — sweeps
  T ∈ {256, 512, 1024, 4096}, K ∈ {16, 32}, dtype ∈ {f32, bf16}.
- For each shape: 1 warm-up call, then `iters` measured calls
  (`iters = 20` for T ≤ 512, `iters = 5` otherwise).
- `device.synchronize()` before and after each timing window.
- Two paths compared:
  - **Kernel** — `opd_top_k_reverse_kl_phase_b_per_position` routed
    through the fused CUDA kernel (`cuda_kernel_supports(K, dtype)`
    returns `true`).
  - **Candle** — `opd_top_k_reverse_kl_phase_a_per_position`, the
    autograd-aware reference path executed on CUDA storage via
    candle's matmul / log_softmax / etc.

## A6000 (RTX A6000, 49140 MiB, CUDA 12.4, driver 570.195.03)

Pod: `8v9c4kq0uvcjjw` / lease `pod-4e82038bc1bff16fa7fa9fca` /
commit `60db09ff`.

```
# kiln-opd-loss-kernel throughput bench
# device: Cuda(CudaDevice(DeviceId(1)))
# header: shape  K  dtype  iters  kernel_ms  candle_ms  speedup_x  kernel_tok_s
T=  256  H=2560  V=32000  K=32  F32  iters= 20  kernel=  0.563ms  candle=  1.334ms   2.37x     455014 tok/s
T=  512  H=2560  V=32000  K=32  F32  iters= 20  kernel=  0.871ms  candle=  2.789ms   3.20x     587770 tok/s
T= 1024  H=2560  V=32000  K=32  F32  iters=  5  kernel=  1.756ms  candle=  7.656ms   4.36x     583043 tok/s
T= 4096  H=2560  V=32000  K=32  F32  iters=  5  kernel= 16.975ms  candle= 29.860ms   1.76x     241297 tok/s
T= 1024  H=2560  V=32000  K=16  F32  iters=  5  kernel=  4.219ms  candle=  4.880ms   1.16x     242727 tok/s
T= 1024  H=2560  V=32000  K=32  BF16  iters=  5  kernel=  1.570ms  candle= 10.372ms   6.61x     652428 tok/s
T= 4096  H=2560  V=32000  K=32  BF16  iters=  5  kernel=  5.102ms  candle= 31.690ms   6.21x     802808 tok/s
```

### Headline reads

- **Production path (bf16, K=32)**: 6× speedup at both T=1024 and T=4096,
  hitting **803K tok/s at T=4096**. This is the §6 default the trainer
  will hit.
- **F32 path (K=32)**: 1.76–4.36× speedup depending on shape; smaller
  wins at T=4096 (matmul-bound regime where candle's cuBLAS is harder
  to beat with a custom kernel).
- **K=16 path (f32)**: only 1.16× speedup. The K=16 kernel doesn't fill
  the SMs as well as K=32 (16 warps × 32 threads = 512 threads/block
  vs the 1024-thread peak), so candle's optimised path is closer.
  Still positive — the kernel doesn't hurt anywhere.

### Throughput target (§9.7 "today" CUDA column)

The grand-plan §9.7 target for CUDA on a 4090 is ≥600 tok/s cached / ≥1500
local — those numbers are for the **full OPD training step**, not just
the loss kernel. The loss kernel is ~10% of the step (per §9.1), so for
the §9.7 step target to be feasible the loss kernel needs to clear
roughly **6000 tok/s at production shape** to not be the bottleneck.

Our bf16 path at T=4096 hits **803K tok/s** — far in excess of that
budget. The kernel is not on the critical path.

### Performance regression gate

The numbers above are the **A6000 baseline**. A 5%+ regression on the
K=32 bf16 row at T=1024 or T=4096 should fail CI per §9.9 of the grand
plan. The bench is reproducible via the example binary; cron + diff
against this file is the simplest gating path.

## Next-hardware columns

Pending:
- 4090 (Ada / SM 89) — primary prosumer target.
- 7900 XTX (Vulkan) — milestone 7 once the Vulkan kernel lands.
- M3 Max (Metal) — milestone 8.
- H200 (multi-GPU, full-vocab path) — corporate-tier validation.

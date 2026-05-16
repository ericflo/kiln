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

## A6000 — forward + backward (kernel-bwd lands)

Pod: same lease (`pod-4e82038bc1bff16fa7fa9fca`), commit `77833003`.
16/16 tests pass (10 CPU + 3 fwd-CUDA + 3 bwd-CUDA parity tests). The
`FWD+BWD` rows include autograd graph construction + the kernel
backward (`OpdLossCustomOp::bwd → cuda_kernel_backward`); the candle
column reflects the analytic backward (`backward_inner`).

```
# kiln-opd-loss-kernel throughput bench (FWD only)
T=  256  H=2560  V=32000  K=32  F32   kernel=  0.560ms  candle=  1.347ms   2.40x   457K tok/s
T=  512  H=2560  V=32000  K=32  F32   kernel=  0.848ms  candle=  2.541ms   3.00x   604K tok/s
T= 1024  H=2560  V=32000  K=32  F32   kernel=  1.510ms  candle=  7.370ms   4.88x   678K tok/s
T= 4096  H=2560  V=32000  K=32  F32   kernel=  9.786ms  candle= 43.660ms   4.46x   419K tok/s
T= 1024  H=2560  V=32000  K=16  F32   kernel=  4.029ms  candle=  4.580ms   1.14x   254K tok/s
T= 1024  H=2560  V=32000  K=32  BF16  kernel=  1.303ms  candle= 11.991ms   9.21x   786K tok/s
T= 4096  H=2560  V=32000  K=32  BF16  kernel=  5.268ms  candle= 31.707ms   6.02x   778K tok/s

# Forward + backward (the trainer's actual step)
T=  256  H=2560  V=32000  K=32  F32   kernel=  2.153ms  candle=  3.717ms   1.73x   119K tok/s
T=  512  H=2560  V=32000  K=32  F32   kernel=  5.592ms  candle=  7.426ms   1.33x    92K tok/s
T= 1024  H=2560  V=32000  K=32  F32   kernel= 16.212ms  candle= 18.443ms   1.14x    63K tok/s
T= 4096  H=2560  V=32000  K=32  F32   kernel= 94.060ms  candle= 82.857ms   0.88x    44K tok/s   ← regression
T= 1024  H=2560  V=32000  K=16  F32   kernel= 13.529ms  candle= 10.136ms   0.75x    76K tok/s   ← regression
T= 1024  H=2560  V=32000  K=32  BF16  kernel=  4.616ms  candle= 21.493ms   4.66x   222K tok/s
T= 4096  H=2560  V=32000  K=32  BF16  kernel= 17.541ms  candle= 82.693ms   4.71x   234K tok/s
```

### Headline reads

- **Production path (bf16 K=32)**: **4.7× end-to-end speedup** at both
  T=1024 and T=4096. The trainer's per-step cost on the loss kernel
  drops from ~80ms to ~17ms at T=4096. This is the run rate at which
  the rest of the §3.1 training step's components (rollout, teacher
  query, autograd graph teardown, AdamW) become the limiting factor.
- **Forward-only bf16 K=32**: **9.2× speedup** at T=1024, 6.0× at
  T=4096. Pure forward (metrics pass, validation, judge LoRA scoring)
  is dramatically faster.
- **F32 K=32 T=4096 regression**: kernel is **0.88×** at this shape —
  candle's cuBLAS-backed analytic backward beats us at this size on
  scattered head_t reads. Documented but **production-irrelevant**:
  Qwen3.5-4B training is bf16; the f32 path is only used by the
  CPU-only reference (`KILN_OPD_LOSS_PHASE_A=1`) and the parity
  oracle. Pit-of-success guidance: leave the kernel on; if a future
  f32 workload appears, set `KILN_DISABLE_OPD_LOSS_KERNEL=1` until
  the bwd kernel is retuned for that regime.
- **K=16 regression**: also documented; K=16 doesn't fill the SMs as
  efficiently. K=32 is the §6 default and the recommended path.

### Adversarial self-check

If I think this is done — what would make it more complete?
- ✔ Both forward and backward kernels parity-tested on A6000.
- ✔ Production dtype (bf16) wins by >4× end-to-end at the canonical
  shape range.
- ✔ Kill-switch documented for the f32 regression.
- ◯ Per-engine bench gate as CI feature (§9.9) — needs a CI runner
  with GPU access; tracked as part of §9.9 implementation.
- ◯ Save-not-recompute optimization: the bwd kernel currently
  recomputes forward state (s_logits, p_hat, etc.) — saving them
  from the forward run would halve the work. Future optimization.
- ◯ K=64 support (corporate full-vocab tier) — separate task.

## Next-hardware columns

Pending:
- 4090 (Ada / SM 89) — primary prosumer target.
- 7900 XTX (Vulkan) — under active development on a separate branch.
- M3 Max (Metal) — deferred per goal.
- H200 (multi-GPU, full-vocab path) — corporate-tier validation.

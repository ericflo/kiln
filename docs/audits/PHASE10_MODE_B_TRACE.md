# Phase 10 §2 prep — Mode B per-allocation trace on A6000

Date: 2026-04-29
Status: **DIAGNOSED**. The Mode B failing allocation at T=8192 seg=16 is
the **FLCE Phase A `logits_chunk` / `shifted` / `shifted.exp()` triplet
of `[num_active=8180, chunk_len=4096]` f32 = 127.81 MiB each, held live
across 60+ vocab chunks for autograd**. The closure audit (PR #645,
`PHASE10_S1_CLOSURE_STATUS.md`) ranked this candidate #4 ("Backward
saved-tensor for FLCE final-projection chunk"). The trace shows it is
actually #1 by a wide margin.

Hardware: NVIDIA RTX A6000 (49,140 MiB total VRAM, sm_86, driver
550.127.08), CUDA 12.4. Pod: RunPod direct-launch (`rh885mmucajgxw`)
on the kiln-runpod image (pool A6000 returned `SUPPLY_CONSTRAINT` on
resume, fell back to direct launch per the kiln skill).
Branch: `ce/phase10-mode-b-trace`
Commit: `55f9493` (PR #645 closure audit; HEAD of main at task start)
Bench: single-cell `T=8192, seg=16, RMSNorm CustomOp ON` patch of
`crates/kiln-server/examples/phase10_rmsnorm_bench.rs` (idempotent
patch in `scripts/mode_b_trace_patch.py`).
Model: Qwen3.5-4B (V=248320, hidden=2560, 32 layers, 24 GDN + 8 GQA),
bf16, no W4A16. FLCE on (`KILN_USE_FLCE=1`). LoRA rank=8.
`KILN_GRAD_CHECKPOINT_SEGMENTS=16`.

## TL;DR

* **The single allocation that triggers Mode B OOM is a `[num_active=8180,
  chunk_len=4096]` f32 tensor of size 134,021,120 bytes (127.81 MiB)
  requested via `cuMemAllocAsync` and rejected with
  `CUDA_ERROR_OUT_OF_MEMORY` (return code 2).** Captured in the LD_PRELOAD
  shim's alloc log at the very last line:
  ```
  10.414768 cuMemAllocAsync size=134021120 MiB=127 ptr=0x0 ret=2
  ```

* **The pattern leading up to OOM is unambiguous.** During the FLCE
  forward pass, kiln issues a strict triplet of allocations per vocab
  chunk: 1× 40 MiB + 3× 127.81 MiB. Across the captured 19 ms burst (10.395
  s — 10.414 s elapsed in the bench process), **only two unique allocation
  sizes appear**: 92× of 134,021,120 B (127.81 MiB) and 30× of 41,943,040 B
  (40 MiB). The OOM fires on the 93rd 127.81 MiB request.

* **Sizes match the FLCE Phase A intermediates exactly:**
  - **40 MiB** = `[hidden=2560, chunk_len=4096]` f32 = **`head_chunk`**
    (the contiguous `head_t_f32.narrow(1, chunk_start, chunk_len)`
    materialized at `crates/kiln-flce-kernel/src/lib.rs:175-179`).
  - **127.81 MiB** = `[num_active=8180, chunk_len=4096]` f32 = each of
    `logits_chunk`, `shifted`, `shifted.exp()` (one alloc each, three per
    chunk) at `crates/kiln-flce-kernel/src/lib.rs:181-205`.
  - `num_active = 8180` = `seq_len(8192) − ChatML framing(12)`. The bench
    uses `<|im_start|>user\nSummarize.<|im_end|>\n<|im_start|>assistant\n`
    framing which tokenizes to ~12 non-loss positions in the mask.

* **Why these sizes accumulate.** FLCE Phase A keeps `logits_chunk`,
  `shifted`, and `shifted.exp()` live for autograd's backward pass. The
  scalar loss depends on `running_sumexp` which is the sum-tree across
  ALL chunks' `shifted.exp()` — so candle's autograd refuses to free any
  of them until the LM-head/CE backward runs. With V=248320 / chunk_size
  =4096 = **61 chunks**, the theoretical FLCE-only steady state is
  61 × 3 × 127.81 MiB = **~23 GiB just to hold FLCE intermediates live**
  alongside the model weights (16 GiB), the LoRA grad set, and the 16
  per-segment activation snapshots saved by gradient checkpointing.

* **Verdict:** **Mode B is FLCE Phase A's documented autograd-graph
  bottleneck**, exactly as the kernel's own source-comment predicts:
  > *Phase A keeps these as `Tensor`s so autograd can backprop into
  > `active_hidden`; Phase B will detach + recompute in a CustomOp.*
  >  
  > — `crates/kiln-flce-kernel/src/lib.rs:147-150`

* **Right next slice is NEITHER §2 (Liger RoPE) NOR §3 (Liger
  SwiGLU/GeGLU).** The right next slice is **FLCE Phase B — implement
  a manual-backward CustomOp for `fused_linear_cross_entropy` that
  recomputes chunk intermediates instead of holding them live for
  autograd.** This is already-planned work per the kernel's own source.
  See [`#recommended-next-slice`](#recommended-next-slice) for sizing
  and rationale.

## Purpose

PR #645's closure audit verified that PR #644's VRAM-gate Path 1
remediation works on A6000 but **failed to close T=8192 SFT** because
both Mode A (ceiling-OOM) and Mode B (low-peak OOM) still fire. The
closure audit listed four candidate sources for the Mode B single-
allocation failure but stopped short of running a per-allocation trace
to identify which one was the actual trigger:

1. MLP gate-projection scratch
2. F32 cross-entropy logits or backward grads (FLCE chunk slices)
3. GDN recurrent-state intermediates
4. Backward saved-tensor for FLCE final-projection chunk

This audit answers: **"Which single allocation fails first when running
T=8192 seg=16 RMSNorm-on SFT on A6000?"** — and does so without any
kernel implementation work, so the trace can drive the §2/§3 priority
decision before any pod-time on a fusion implementation is spent.

## Method

`nsys profile --cuda-memory-usage=true` was attempted first, but **nsys
2023.4.4 (the version shipped in the kiln-runpod image) has a known
event-ordering import bug** on this exact bench:

```
Importer error status: Importation failed.
Wrong event order has been detected when adding events to the collection
```

The `.qdstrm` is generated, but the `.nsys-rep` conversion fails for
every trace combination tried (`--trace=cuda,nvtx,osrt`,
`--trace=cuda,nvtx`, `--trace=cuda` alone, with/without
`--cuda-memory-usage=true`, with/without `--backtrace=none`). Both the
`nsys` CLI and a direct `QdstrmImporter` invocation reproduce the bug
identically — it is not a flag issue. nsys is unusable on this pod for
this binary. Audit-time wall-clock spent on nsys: ~8 min, ~$0.07.

**Fallback:** an LD_PRELOAD shim that intercepts the CUDA driver pool
allocators (`cuMemAllocAsync`, `cuMemAlloc_v2`, `cuMemAllocFromPoolAsync`,
`cuMemAllocManaged`) plus the matching frees, logs every call ≥4 MiB or
any non-zero return code with a monotonic timestamp. Source for the
shim is at `/tmp/alloc_log.c` on pod `rh885mmucajgxw`; built with
`gcc -O2 -fPIC -shared -ldl`. The shim is ~150 lines, has no external
deps, and adds <1 µs per intercepted call. With `LD_PRELOAD=
/tmp/alloc_log.so ALLOC_LOG=/tmp/alloc.txt`, the bench process emits a
clean per-allocation timeline through OOM.

The shim's `fopen("path", "w")` produces a sparse file (header at offset
0, then a long stretch of NULs from buffered-write reordering, then the
real records). `strings` recovers the textual events without loss; the
123 lines after `strings` were verified line-by-line against `od -c`
to ensure no records were dropped.

Single-cell bench patch (idempotent, in `scripts/mode_b_trace_patch.py`):

```python
# Replaces the 6-cell main() body with one cell:
#   rows.push(run_one(8192, true, ...))
# Same shape as the closure audit's single-cell variants.
```

Run command:
```
KILN_GRAD_CHECKPOINT_SEGMENTS=16 \
RUST_LOG=info,kiln=debug,kiln_train=debug,kiln_model=debug \
ALLOC_LOG=/tmp/mode_b_pre/alloc.txt \
LD_PRELOAD=/tmp/alloc_log.so \
./target/release/examples/phase10_rmsnorm_bench \
  --model-path /workspace/qwen3.5-4b
```

Bench result without nsys overhead:

```
| T target | T actual | RMSNorm op | status | peak (MiB) | delta (MiB) | step (s) | final loss |
| 8192     | 8192     | on         | OOM    | 46248      | 29904       | 6.27     | -          |
```

Peak 46,248 MiB sits between Mode A (~48,500 MiB ceiling) and the
PR #645 Mode B re-run peak (39,048 MiB). The 50 ms `nvidia-smi` poller
in the bench is not synchronous with allocation events, so the peak it
reports is a sample of the high-water mark rather than the moment-of-
OOM allocator state. The shim's per-allocation log is the canonical
record.

## Results

### Failing allocation

```
10.414768 cuMemAllocAsync size=134021120 MiB=127 ptr=0x0 ret=2
```

* `size=134021120` bytes = **127.81 MiB** (= 8180 × 4096 × 4 = bf16 dim
  8180, f32 chunk 4096, exact)
* `ptr=0x0` (device pointer not assigned — alloc failed)
* `ret=2` = `CUDA_ERROR_OUT_OF_MEMORY`

### Allocation size distribution

```
=== UNIQUE SIZES + COUNTS ===
     92 size=134021120 MiB=127      <-- FLCE chunk f32 intermediates
     30 size=41943040  MiB=40       <-- FLCE head_chunk f32
```

**Only two unique allocation sizes appear in the entire 19 ms burst.**
122 successful allocs, then the 123rd (a 127 MiB request) fails. No
other class of allocation appears at the OOM site: not GDN in_proj
(8192-dim bf16 ≈ 128 MiB), not MLP gate (9216-dim bf16 ≈ 144 MiB), not
attention scores (would be `[bs, h, T, T]` ≈ 2 GiB and FA-2 would not
materialize them), and not RoPE intermediates (Q/K rotations are bf16
and would be 33-67 MiB).

### Burst pattern (last 30 events before OOM)

```
10.397049 cuMemAllocAsync size=134021120 MiB=127  ret=0   # logits_chunk[N]
10.397773 cuMemAllocAsync size=134021120 MiB=127  ret=0   # shifted[N]
10.398418 cuMemAllocAsync size=134021120 MiB=127  ret=0   # shifted.exp()[N]
10.398639 cuMemAllocAsync size=41943040  MiB=40   ret=0   # head_chunk[N+1]
10.399525 cuMemAllocAsync size=134021120 MiB=127  ret=0   # logits_chunk[N+1]
... pattern continues ~30 times ...
10.413605 cuMemAllocAsync size=41943040  MiB=40   ret=0   # head_chunk[31]
10.414231 cuMemAllocAsync size=134021120 MiB=127  ret=0   # logits_chunk[31]
10.414768 cuMemAllocAsync size=134021120 MiB=127  ret=2   # shifted[31] FAILS
```

**Per-chunk allocation triplet:** `head_chunk → logits_chunk → shifted →
shifted.exp()`. The pattern repeats once per FLCE vocab chunk. With 92
successful 127 MiB allocs, FLCE made it through ~30 chunks (= ~92/3) of
its 61-chunk forward pass before the allocator's free pool segment
could no longer satisfy a 128 MiB-class request.

### Mapping each size to a kiln source line

`crates/kiln-flce-kernel/src/lib.rs`:

| Size         | Source line         | Tensor                              | Shape                     | Dtype |
|-------------:|:--------------------|:------------------------------------|:--------------------------|:------|
| 40 MiB       | `lib.rs:175-179`    | `head_chunk` (post-`.contiguous()`) | `[2560, 4096]`            | f32   |
| 127.81 MiB   | `lib.rs:181-184`    | `logits_chunk = active_hidden_f32 @ head_chunk` | `[8180, 4096]` | f32   |
| 127.81 MiB   | `lib.rs:202`        | `shifted = logits_chunk - new_max.broadcast_as(...)` | `[8180, 4096]` | f32 |
| 127.81 MiB   | `lib.rs:203`        | `chunk_sumexp_intermediate = shifted.exp()` | `[8180, 4096]` | f32 |

The first chunk's `running_max` and `running_sumexp` initialization
(lines 195-200) also produces `shifted` and `shifted.exp()`, matching
the same pattern. Subsequent chunks reuse the running accumulators in
the `(Some, Some)` branch (lines 204-216) but still allocate a fresh
`shifted` and `shifted.exp()` per chunk — `prev_scale` and `scaled_prev`
are small (`[num_active, 1]` f32 = ~32 KiB).

### Why all three live tensors are kept across chunks

```rust
// crates/kiln-flce-kernel/src/lib.rs:147-150
// Phase A keeps these as `Tensor`s so autograd can backprop into
// `active_hidden`; Phase B will detach + recompute in a CustomOp.
```

The scalar `loss = correct_logit_sum - log(running_sumexp).sum()`. The
running accumulator `new_sumexp = scaled_prev + chunk_sumexp` is built
chunk-by-chunk, and **every chunk's `shifted.exp()` is a leaf in the
autograd graph for `running_sumexp.log()`**. Candle's autograd cannot
free any of them until backward runs, because each is a saved tensor
for the chain `shifted → exp → sum_keepdim → broadcast_add → ... →
running_sumexp`. Equivalently: each chunk's `logits_chunk` is saved
because `shifted = logits_chunk - new_max.broadcast_as(...)` is a node
of the autograd graph and `logits_chunk` is its left operand.

## Failure mode classification

This trace is closer to **Mode A (ceiling-OOM)** than to the closure
audit's reported Mode B (peak 39,048 MiB). Two reasons the peaks differ:

1. **No nsys / nvtx perturbation in the closure audit's Mode B run.**
   This audit's bench was built with `--features cuda` (no `nvtx`) but
   was run with `LD_PRELOAD` overhead on every CUDA alloc. The closure
   audit's bench was vanilla. Both still hit the FLCE Phase A wall, but
   at slightly different watermarks because the candle CUDA caching
   allocator's pool decisions are sensitive to small amounts of extra
   resident memory (precedent: agent note
   `kiln-cuda-kernel-crate-allocator-perturbation`).

2. **The 50 ms `nvidia-smi` poller is asynchronous to allocation
   events.** It samples the GPU's reported `memory.used`, which lags
   the candle allocator's per-stream pool requests by tens of ms. Both
   peaks are LOWER bounds on the actual allocator high-water; the per-
   allocation log is the canonical record.

The underlying bottleneck is identical in Mode A and Mode B: **FLCE
Phase A's autograd-graph holds 3× per-chunk f32 intermediates live for
all 61 vocab chunks at T=8192**, totaling ~23 GiB before the head/CE
backward releases them. Whichever specific chunk's allocation finally
fails depends on the allocator's pool fragmentation state at the
crossover.

## Recommended next slice

**FLCE Phase B (manual-backward CustomOp), NOT §2 (RoPE) or §3
(SwiGLU/GeGLU).**

### Why not §2 (Liger RoPE)

RoPE rotates Q/K. At T=8192:
* Full-attn (8 layers): Q `[1, 8192, 16, 256]` bf16 = 67 MiB, K
  `[1, 8192, 4, 256]` bf16 = 17 MiB. Per-layer transient: ~84 MiB.
* GDN linear-attn (24 layers): Q/K `[1, 8192, 16, 128]` bf16 = 34 MiB
  each. Per-layer transient: ~67 MiB.

Neither layer-class produces a **127.81 MiB single allocation**. RoPE
fusion would save roughly 67-84 MiB per layer of intermediate scratch
times 32 layers = ~2-2.7 GiB at T=8192 — meaningful, but **does not
unblock T=8192 SFT** because the dominant pressure (~23 GiB of FLCE
intermediates) still exceeds whatever room a §2 fusion creates.

### Why not §3 (Liger SwiGLU/GeGLU)

MLP gate-projection at T=8192 with `intermediate_size=9216`:
`[1, 8192, 9216]` bf16 = 1 × 8192 × 9216 × 2 = 150,994,944 bytes =
**144 MiB per layer** — close to but distinct from the observed 127.81
MiB. Per the trace, **no allocation of size 150,994,944 appears in
the OOM burst**, which is strong evidence that the MLP gate-projection
intermediate is NOT the failing allocation. (FLCE runs after the
last segment's MLP, so the MLP intermediates have already been freed
by the candle allocator before FLCE starts its chunk burst.)

§3 would still pay off if applied — it eliminates ~144 MiB × 32 layers
= ~4.6 GiB of MLP transients during forward — but again, **does not
close the FLCE Phase A gap**.

### Why FLCE Phase B IS the right next slice

`crates/kiln-flce-kernel/src/lib.rs:20-23` already plans Phase B:

> Phase B (follow-up) will replace `fused_linear_cross_entropy` with a
> CustomOp that does the chunked forward without saving tensors; its
> manual `bwd()` recomputes the chunks to produce `dhidden` without
> retaining intermediates.

Phase B implementation outline:
* Forward pass produces only `loss: f32` (scalar) + a small saved set:
  the running `correct_logit` accumulator (`[num_active]` f32 = 32 KiB)
  and the final `running_max`, `running_sumexp` (`[num_active, 1]` f32
  = 32 KiB each). Total saved: ~96 KiB.
* Backward pass receives `loss.backward()`'s upstream `dloss = 1.0`
  and recomputes each chunk's `logits_chunk → shifted → softmax_chunk
  → grad_logits_chunk` on the fly, producing `dhidden` and `dW_chunk`
  without keeping any chunk intermediate live across chunks.

Estimated savings at T=8192:
* Phase A holds: 3 × 127.81 MiB × 61 chunks = **~23 GiB** of FLCE
  chunk intermediates live for autograd (theoretical max if backward
  is delayed to the very end of the step).
* Phase B holds: at most 3 × 127.81 MiB at any instant (current chunk
  only) = **~384 MiB**.
* Net peak-VRAM saving: **~22 GiB at T=8192**, which is more than
  enough to close the gap to the A6000 48 GiB ceiling (the closure
  audit's Mode A hit peak ~48,500 MiB; subtracting 22 GiB leaves
  ~26,500 MiB peak — comfortably below the ceiling and consistent with
  the T=2048 seg=8 baseline of ~42 GiB peak).

**Phase B is the only audit-supported path to T=8192 SFT closure on
A6000 within the Phase 10 plan.** It is also explicitly ranked in the
kernel source as the planned follow-up to Phase A — implementing it is
not a new design decision, just executing on existing scope.

### Companion improvements to file alongside Phase B (smaller wins)

These do not unblock T=8192 alone but should be considered part of the
same patch series since they touch FLCE call sites:

* **Larger FLCE chunk size at long T.** `DEFAULT_CHUNK_SIZE=4096` was
  chosen for V=151936 (per the source comment) but Qwen3.5-4B has
  V=248320. With Phase B's recompute path, chunk_size could be 8192 or
  16384 without keeping live tensors, which reduces per-chunk launch
  overhead. Out of scope for the audit but worth noting.
* **Skip the `shifted.exp()` materialization** by fusing
  `chunk_sumexp = shifted.exp().sum_keepdim(D::Minus1)` into a single
  reduction. Even in Phase A this would drop one of the 3× per-chunk
  127.81 MiB tensors. Optional — Phase B subsumes this.

### What §2 and §3 are still good for (post-Phase-B)

§2 RoPE and §3 SwiGLU/GeGLU remain valid Phase 10 slices for
**decode-time speedups** (per kiln's existing roadmap commentary on
Liger fusions delivering throughput improvements distinct from peak
memory). They should NOT be queued ahead of Phase B for SFT-on-A6000
purposes, but they remain on the post-Phase-B menu.

## Cost

* Pod: A6000 on-demand at $0.49/hr (direct-launch fallback per pool
  capacity exhaustion).
* Time on pod: ~65 min wall-clock end-to-end (kiln-setup + clone + 2
  builds + 4 bench runs including 3 nsys attempts and the LD_PRELOAD
  capture + write-up).
* Cost: ~$0.53, well within the 75 min / $25 cap for this audit.
* No SSH-wedge incidents (used `python3 $RP bg` + `python3 $RP wait-file`
  for all background tasks; no `until ssh ...` polling loops, no bare
  `trap ... EXIT`).

## Process

This audit followed the kiln skill's "MONEY-BURNING ANTI-PATTERNS"
section to the letter:
* All long-lived tasks via `python3 $RP bg` + `python3 $RP wait-file
  --timeout`.
* No `until ssh ...` or `while ssh ... grep` loops.
* `trap ... ERR INT TERM` only (no bare `EXIT`).
* On-demand pod, never spot.

When nsys's import bug blocked the planned trace path within ~8
minutes, the audit pivoted to LD_PRELOAD instead of burning time on
nsys workarounds. The pivot saved roughly 30 minutes of pod time
relative to attempting nsys re-installs or container rebuilds, and
produced a strictly more useful artifact (the shim runs in any future
audit on any nsys version).

## References

* `crates/kiln-flce-kernel/src/lib.rs` — chunk loop and Phase A/B
  documentation comments.
* `crates/kiln-train/src/trainer.rs` — FLCE call sites (lines 1401,
  1717, 2135, 2188), all using `DEFAULT_CHUNK_SIZE=4096`.
* `docs/audits/PHASE10_S1_CLOSURE_STATUS.md` (PR #645) — closure audit
  that motivated this trace.
* `docs/audits/PHASE10_FLCE_PREFLIGHT.md` (PR #631) — FLCE Phase A
  preflight on A6000 (the source of `kiln-flce-phase-a-validation-2026
  -04-29` agent note).
* Agent note `kiln-flce-is-prerequisite-not-optimization` — FLCE preflight
  found logits tensor dominates peak VRAM at long T (this audit refines:
  it's not just the materialized logits, it's the per-chunk intermediates
  held live for autograd).
* Agent note `kiln-cuda-kernel-crate-allocator-perturbation` — allocator
  perturbation precedent for why nsys+nvtx vs vanilla bench peaks differ.

# vLLM `fused_recurrent_gated_delta_rule` vs kiln `kiln-gdn-kernel` — audit

**Date:** 2026-04-24  
**Scope:** doc-only; $0 pod spend  
**Trigger:** project-goals item _"Audit vLLM's Triton `fused_recurrent_gated_delta_rule`
GDN kernel and port any wins back into kiln's vendored CUDA GDN kernel."_
Unstarted prior to this PR. Kernel-vendor-precondition-check requires a desk
audit before any paid-pod port task.

## Summary

**Verdict: no Class A portable wins.** vLLM's decode-path fusion recipe is
algorithmically identical to kiln's where it matters numerically. The
remaining upstream advantages are **structural** (Triton tile shape, state
memory layout, paged-state plumbing, speculative-decode multi-token-per-launch)
and would require a full re-port of kiln's decode kernels, not a bounded
micro-port.

Under CUDA graphs on A6000, the specific gains vLLM reports for fusing
L2-qk-norm / gates / scale into the recurrent kernel are below the 1.05× floor
by three orders of magnitude — they amount to eliminating 1–2 extra kernel
launches and a ~1 MiB bf16 roundtrip per decode step, both of which CUDA graph
capture already amortizes to near-zero. This matches PR #173's null-median
result for kiln's own L2-qk-norm fusion.

**Two items worth recording for the future**, neither an implementation
target for Phase 7:

- When the MTP speculative-decode work lands, vLLM's `IS_SPEC_DECODING` +
  `num_accepted_tokens` pattern (multi-token advance per kernel launch with
  state reload at the accepted-prefix index) is the natural reference for
  kiln's recurrent rollback.
- If kiln ever moves to paged GDN state (a continuous-batching dependency),
  vLLM's `ssm_state_indices` indirection is the reference for block-table-
  style state addressing.

**Recommendation:** close the goal. Do not queue a follow-up port task from
this audit.

## Methodology

- **kiln source at audit time:** `9f1525c449` (HEAD of branch
  `phase7-vllm-gdn-audit` off main, post-PR #524).
  - `crates/kiln-gdn-kernel/src/lib.rs` — Rust wrapper (1922 lines).
  - `crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.cu` — seq_len=1 recurrent
    kernel + fused gates+recurrent variant (337 lines). This is the direct
    counterpart to vLLM's recurrent kernels.
  - `crates/kiln-gdn-kernel/csrc/gdn_gates.cu` — standalone gates kernel
    (PR #158, merged).
  - `crates/kiln-model/src/forward.rs` — dispatch path around lines 2499–2530
    (`gdn_qk_norm`, `l2_normalize`) and 2804–2929
    (`gdn_single_token_recurrence`, `gdn_chunkwise_recurrence` seq_len==1 fast
    path calling `backend.gdn_recurrent_step`).
- **vLLM source at audit time:** HEAD `333529deae5` (post 2026-04-24).
  - `vllm/model_executor/layers/fla/ops/fused_recurrent.py` last touched at
    commit `d4cb783c10` on 2026-04-17
    (#39064 — NULL_BLOCK_ID=0 CUDA graph padding bugfix).
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` for the decode
    dispatch site (call at line 1051).
- **History reviewed:** full `git log -- vllm/model_executor/layers/fla/ops/fused_recurrent.py`
  back to the initial port commit `1aa427fdc` (#24518, 2025-09-10 vintage).
  Perf-relevant commits:
  - `22dffca98` (#31722, 2026-01-06) — `[PERF] Speed-up of GDN attention decode
    part (Qwen3-Next)`. Diff is **one line**: `BV = min(next_power_of_2(V), 8)
    → 32`. Tile-size tuning.
  - `824058076` (#33291, 2026-02-04) — `[PERF] Change GDN Attention State Layout
    from [N, HV, K, V] to [N, HV, V, K]`. Transposed state dims to match the
    updated tile layout after BV was widened.
  - `9e19f8338` (#36596, 2026-03-12) — `[Perf] add packed recurrent fast path
    for decode`. Adds `fused_recurrent_gated_delta_rule_packed_decode_kernel`
    that inlines `a/b/A_log/dt_bias → g/β` gate math, L2-qk-norm, the
    `1/sqrt(K)` scale, the state step, and the output projection into a
    single Triton launch.
  - `a16133a0f` (#37338) — `torch.LongTensor → torch.Tensor` type annotation
    fix; no algorithmic change.
  - `d4cb783c1` (#39064) — guard against NULL_BLOCK_ID=0 in CUDA graph padding
    (bugfix; not a perf change).
- **Kiln state of art assumed by this audit:**
  - PR #80 (#0c9c519) — vendored `kiln-gdn-kernel` crate (full-chunk forward
    + recurrent seq_len=1). Per `PROFILING.md`, **decode is dominated by GDN
    body regions** (~69% wall-clock post #521).
  - PR #158 — fused gates kernel (`:kiln/gdn/gates`, merged).
  - PR #173 — opt-in fused L2-qk-norm (`kiln_rmsnorm_kernel::fused_l2_qk_norm`)
    — null median, shipped behind `KILN_ENABLE_FUSED_L2_QK_NORM` for variance
    reduction only.
  - PR #141 — `:kiln/gdn/gated_norm` fusion closed null.
  - PR #524 — single-kernel kill-switch bisection for the post-#166 decode
    regression closed null; best explanation is thermal/clock drift, not a
    single kernel.

## Findings

Axis-by-axis comparison. Expected speedup numbers are ceilings under the
stated model (`B=1`, Qwen3.5-4B on A6000, 24 GDN layers, 32 value heads,
`dk=dv=128`).

| # | Axis | kiln (current) | vLLM (current HEAD) | Class | Expected speedup floor | Expected speedup ceiling |
|---|---|---|---|---|---|---|
| 1 | Decode-step fusion boundary | **3 kernel launches:** gates (PR #158) → L2-qk-norm (PR #173 opt-in) → recurrent. Each exchanges bf16 tensors via global memory. | **1 kernel launch** (`packed_decode_kernel`): inlines `a/b/A_log/dt_bias`-derived gates, L2-qk-norm, scale, state decay+update, output projection. | **D** (already done as much as physically matters) | n/a | n/a — PR #173's null median is the evidence; deeper fusion saves ~0.006% of decode under CUDA graphs (see calc below) |
| 2 | State memory layout | `[B, HV, dk, dv]` with `dv` innermost. Inner stride `dv`; thread-per-column → `s_base[i * dv + tid]` is coalesced. | Was `[N, HV, K, V]`. Flipped to `[N, HV, V, K]` in #33291 to re-coalesce after BV widened to 32 (tile `[BV, BK]` where `BK=K` is inner). | **C** | n/a | n/a — kiln's layout is already optimal for its thread-per-column mapping; flipping it without adopting vLLM's tile algorithm would **hurt**, not help |
| 3 | Tile shape / parallelism | One block per `(B, HV)`, `blockDim.x = dv = 128` (4 warps), per-thread `s_local[128] F32` in registers (≈128 32-bit regs/thread, at the sm_86 2048-reg-per-warp budget), fully unrolled dk loop. | Grid `(NV=4, B*HV)`, `BV=32`, `BK=128`, `num_warps=1` (one 32-thread warp), `num_stages=3` async pipelining; state tile `[BV=32, BK=128] F32` in shared memory / mma frags. | **B** (portable in principle, but requires full re-port with unknown speedup) | n/a | n/a — see "Class B details" below |
| 4 | L2 qk-norm fusion | Separate kernel (`kiln_rmsnorm_kernel::fused_l2_qk_norm`, PR #133+#173). Writes normalized bf16 q,k back to memory. Opt-in fused path. | Inline in `packed_decode_kernel` (`USE_QK_L2NORM_IN_KERNEL=True`). | **D** | n/a | PR #173 already shipped a standalone fused version; deeper fusion into the recurrent kernel saves only the q,k bf16 roundtrip (≈96 KiB/step system-wide) and one launch. At ~700 GB/s HBM, the roundtrip is ~140 ns; CUDA graphs amortize the launch. Combined ~0.001% of a 20 ms decode step. |
| 5 | Gate math fusion (sigmoid+softplus+exp+mul) | Standalone kernel `kiln_gdn_gates_bf16` (PR #158). Writes bf16 `beta`, `g` to memory. | Inline in `packed_decode_kernel`. | **D** | n/a | Same argument as #4. `beta` + `g` are [B, T, HV] bf16 = 128 B/head × 32 heads × 24 layers ≈ 96 KiB roundtrip/step. Amortized. |
| 6 | Scale `1/sqrt(K)` application | Applied inside L2-qk-norm path (`(q * scale)?.to_dtype(input_dtype)?`). | Applied inline in recurrent kernel (`b_q = b_q * scale`). | **D** | n/a | Zero-sum move — kiln already pays it once in the normalization path |
| 7 | Gate `g` numerical path | `gdn_decode_gates_recurrent_rmsnorm_bf16_kernel` computes `g` in F32 then **roundtrips through bf16** before `exp`: `scalars[0] = expf(bf16_to_f32(f32_to_bf16(g)))`. Parity shim for matching Python ref that writes g as bf16 before the recurrent step. | Computes `g` in F32 and applies `exp(g_val)` directly; no roundtrip. | **C — correctness** (not perf) | n/a | n/a — removing the roundtrip breaks kiln's parity oracle against its own split-kernel path; perf win is immeasurable (a couple of FPU ops on a scalar) |
| 8 | Speculative decode support | None in the recurrent kernel. (Spec-decode design is tracked in `kiln-spec-decode-gdn-rollback` note; not shipped.) | `IS_SPEC_DECODING=True` path loops `for i_t in range(0, T)` with state-reload from `num_accepted_tokens + i_n`, enabling multi-token-per-launch verify with rollback. | **C** — future reference | n/a | n/a — kiln MTP spec decode is not in scope for Phase 7; when it lands, this is the design to mirror |
| 9 | Paged GDN state (continuous batching) | Contiguous `[B, HV, dk, dv]`. No `ssm_state_indices` indirection. | `ssm_state_indices + i_n * stride_indices_seq * stride_init_state_token` → block-table style addressing. | **C** — future infrastructure | n/a | n/a — requires continuous-batching rework in kiln-scheduler; not a GDN-kernel perf port |
| 10 | `num_warps` / `num_stages` tuning | Static `blockDim=128` (4 warps), no async pipelining | Triton autotune-friendly (`num_warps=1`, `num_stages=3`); vLLM experimentally tuned BV from 8 → 32 in #31722 | **B** — bundled with #3 | n/a | n/a — cannot port single knob; Triton's async pipeline has no direct CUDA-C equivalent short of manual CUTLASS-style overlap |

### Back-of-envelope: why deeper fusion saves nothing under CUDA graphs

Decode median at post-PR #521: **~49 tok/s**, i.e. ~20 ms/token on A6000 BF16
+Marlin, 24 GDN layers × 32 heads × (dk=dv=128).

- Extra memory exchanged by kiln's split design, per decode step:
  - `β`, `g` (gate outputs, bf16, [B=1, T=1, HV=32, per-layer×24]) ≈ 6 KiB
  - normalized `q`, `k` (bf16, [1, 1, 32, 128]×24) ≈ 384 KiB
  - roundtrip = write + read ≈ **≤ 1 MiB/step total**
- A6000 HBM bandwidth ≈ 700 GB/s → ~1.4 µs to move 1 MiB
- → **~0.007% of a 20 ms step**, even if the roundtrip is fully unhidden

Kernel launch overhead: CUDA graph capture collapses 24+ GDN layers' kernel
dispatches into a single `cuLaunchGraph`. The per-launch dispatch cost is
amortized to ≲ 1 µs total. Fusing 2 kernels into 1 (PR #173 L2-qk-norm
fusion) saves ~24 dispatches = ≲ 1 µs. Graph replay timing dominated by
kernel execution, not launch.

### Class B details: the tile redesign

vLLM's recurrent kernel is a **tile-level SIMT** algorithm:
- State tile `b_h[BV, BK] F32` in fast memory (shmem or register frags).
- Each step: scalar or vector `load` of `b_g, b_beta, b_q[BK], b_k[BK], b_v[BV]`
  (Triton tile loads).
- `b_h *= exp(b_g)` — broadcast scalar multiply.
- `b_v -= tl.sum(b_h * b_k[None, :], axis=1)` — reduction of `[BV, BK]`
  against `[BK]` → `[BV]`.
- `b_v *= b_beta; b_h += b_v[:, None] * b_k[None, :]` — outer-product update.
- `b_o = tl.sum(b_h * b_q[None, :], 1)` — `[BV, BK]` ⋅ `[BK]` → `[BV]`.

Kiln's kernel is a **thread-per-column ILP** algorithm:
- Each thread (of 128 threads = dv) owns one `s_local[128] F32` column in
  its own register file.
- Phase A: `decayed[i] = decay * s_local[i]; v_pred += k_smem[i] * decayed[i]`
  — fully unrolled ILP.
- Phase B: `new_s = decayed[i] + k_smem[i] * delta; out_acc += q_smem[i]
  * new_s; s_base[i*dv + tid] = bf16(new_s)` — same.

They compute the same math. Differences:
- **Occupancy**: kiln at ~128 F32-regs/thread × 128 threads = heavy register
  pressure (very close to the sm_86 255-regs/thread cap). Two blocks per SM
  is plausible at best. vLLM's `num_warps=1` × 32 regs/thread (plus shmem
  for the `[BV=32, BK=128]` tile) can usually keep 6–8 blocks resident per
  SM on A6000.
- **Parallelism for small batches**: at B=1, kiln has 32 blocks total
  (HV=32). vLLM has 4 × 32 = 128 blocks. A6000 has 84 SMs — vLLM keeps
  more SMs busy at B=1.
- **Hidden latency via async pipelining**: vLLM's `num_stages=3` overlaps
  HBM loads with compute. Kiln's kernel has no explicit async prefetch;
  the unrolled dk=128 loop issues plain loads that stall on L1 misses.

**Potential upside** of a tile-redesign re-port: 1.1×–1.5× on the recurrent
kernel itself, which is a fraction of the GDN region (`kiln/attn/gdn/recurrent`
vs the larger `:kiln/gdn/*` ensemble). End-to-end decode win is harder to
estimate without fresh nsys profiling of the recurrent kernel in isolation.

**Cost**: new ~400-line CUDA kernel (or a Triton dependency), parity test
suite against the existing kernel, nsys capture on A6000.

**Why Phase 7 shouldn't pick this up now**: per the _Current optimization
queue_ in the project description and the post-#524 null-bisection finding,
the post-#166 decode gap is most likely thermal drift, not a kernel
inefficiency. Picking a new Class B re-port without a fresh profile that
isolates `kiln/attn/gdn/recurrent` as the top hotspot risks another
null-median PR.

## What we are NOT recommending and why

- **Fusing L2-qk-norm + gates + recurrent into one kernel (mirror of vLLM
  `packed_decode_kernel`)**: already evidenced as null by PR #173. The
  roundtrip + launch saves ~0.01% of decode under CUDA graphs on A6000.
- **Flipping state layout to `[B, HV, dv, dk]`**: only makes sense coupled
  with a tile redesign; isolated, it would hurt kiln's thread-per-column
  coalescing.
- **Porting `IS_SPEC_DECODING` now**: kiln's native-MTP spec-decode is not
  yet scoped; porting the kernel-side plumbing in advance of the scheduler
  work is speculative.
- **Porting `ssm_state_indices` paged state now**: requires a scheduler /
  block-manager change outside the kernel-vendor envelope; out of scope
  for a GDN kernel PR.
- **Dropping the `expf(bf16_to_f32(f32_to_bf16(g)))` roundtrip**: parity
  concern, not a perf concern. If kiln's parity oracle is updated to match
  FLA's "compute g in F32, apply exp in F32" convention, the roundtrip can
  go — but this is bit-exactness work, not optimization.

## Recommendation

Close the _"Audit vLLM's Triton `fused_recurrent_gated_delta_rule` GDN kernel"_
project goal. Record the structural (Class B) tile-redesign option as a
_future_ candidate that requires:

1. A fresh `nsys` capture isolating `kiln/attn/gdn/recurrent` as the top
   decode hotspot (i.e. at least ~10% of wall-clock so a 1.1× recurrent
   speedup clears the math-ceiling floor).
2. A parity test suite against the existing `recurrent_gdn_fwd_kernel`.
3. An explicit speedup floor (e.g. 1.05× decode tok/s) and abort threshold.

Until those preconditions are met, **no new kernel-port task should be
queued against the recurrent kernel** — the Phase 6/7 history (PR #141, #173,
#176, #524) is clear that additional fusion in this region has saturated.

## Appendix: commits referenced

- **kiln:** PR #80 (chunk vendor), #133 (rmsnorm), #141 (null), #158 (gates),
  #166 (regression origin), #173 (L2-qk-norm opt-in, null median), #176 (null),
  #500/#502 (post-fusion refresh), #521 (prefix cache), #522 (post-521 refresh),
  #523 (prefix-cache A/B null), #524 (kill-switch bisection null).
- **vLLM:** #24518 (initial FLA port), #25432 (jit compile fix), #31722 (BV
  8→32), #33291 (state layout flip), #33326 (negative offset bugfix), #36596
  (packed_decode_kernel), #37338 (Tensor type fix), #39064 (NULL_BLOCK_ID
  guard).

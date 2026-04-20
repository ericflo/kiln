# Kiln Profiling Report


## Phase 6 preflight 2026-04-20: Marlin wrapper BF16-native I/O epilogue (KILL)

**Outcome: KILL. Phase 6 is complete. Advancing to Phase 7.**

This preflight gates the final un-attempted Phase 6 decode lever surfaced
by the [#224](https://github.com/ericflo/kiln/pull/224) frontier audit:
eliminating the per-call BF16↔FP16 cast pair and FP16 intermediate buffer
around every Marlin W4A16 GEMM call in the decode path. The audit
projected region-local speedups in the 1.05–1.10× range with a
contribution of 0.017–0.033 (plausibly clearing the 0.05 floor under two
compounding assumptions). This preflight, $0 and doc-only, tests both
assumptions against the post-#210 direct nsys evidence and kills on two of
the three required checks.

No pod spent. No code changed. Precedent for doc-only redirect: PRs
[#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223).

### Check 1 — BF16-native Marlin tile exists upstream: **PASS**

vLLM's templated Marlin (`csrc/quantization/marlin/`, the successor to the
IST-DASLab FP16-only kernel that kiln currently vendors via PR #146)
carries a full BF16 dtype specialization:

- `csrc/quantization/marlin/marlin_dtypes.cuh`: defines
  `MarlinScalarType<vllm::kBFloat16.id()>` with `nv_bfloat16` /
  `nv_bfloat162` `FragA` / `FragB` / `FragS` types (lines 54–92). Aligned
  on the same 16-lane fragment structure as the FP16 specialization, so
  the tile shape (`Marlin<256,1,8,8,4,8>`) is compatible with the BF16
  variant without geometric changes.
- `csrc/quantization/marlin/marlin_mma.h` lines 66–75: emits
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` under
  `__CUDA_ARCH__ >= 800`. RTX A6000 is sm_86 — supported. H100 (sm_90) and
  A100 (sm_80) are supported. Blackwell (sm_120a) is not the target arch
  for kiln under CUDA 12.4.
- `csrc/quantization/marlin/dequant.h` lines 174–193: provides
  `dequant<nv_bfloat162, vllm::kU4B8.id(), true/false>` specializations
  for both symmetric and asymmetric 4-bit dequant.

Vendoring path is precedented: PR [#80](https://github.com/ericflo/kiln/pull/80)
pulled `chunk_gla_fwd` from fla upstream into `kiln-gdn-kernel` with a
similar scope (several hundred lines of templated CUDA + dtype
specializations). A BF16 port would fork
`crates/kiln-marlin-gemm/csrc/marlin_kernel.cu` and add a parallel
`marlin_bf16_kernel.cu` with the dequant and mma branches swapped out.

**Check 1 is not the blocker.**

### Check 2 — wrapper cast+alloc ≥5% of each Marlin region wall-clock: **PARTIAL**

The direct evidence that exists is kernel-level, not NVTX sub-range. From
`profiling-artifacts/post210_kern.csv`:

```
%  total_ns     instances  avg_ns  name
15.7  279,139,141  13,527  20,636  void Marlin<(int)256,(int)1,(int)8,(int)8,(int)4,(int)8>(...)
 1.0   17,877,596  13,527   1,322  cast_bf16_f16
 1.0   17,695,841  13,527   1,308  cast_f16_bf16
```

The 13,527-instance count matches exactly across all three rows — every
Marlin call on the decode path is bracketed by one `cast_bf16_f16` on the
input side and one `cast_f16_bf16` on the output side, confirming the
wrapper structure at `crates/kiln-marlin-gemm/src/lib.rs` (the actual cast
pair lives one crate below `kiln-model/src/marlin_proj.rs`: lines 168
`a.to_dtype(DType::F16)?.contiguous()?`, 173
`Tensor::zeros((m,n), DType::F16, ...)`, 268
`c_fp16.to_dtype(DType::BF16)?`).

Cast-pair wall-clock is 35.57 ms vs Marlin kernel body 279.14 ms, so the
cast pair alone is **12.74% of the Marlin kernel body time**. Taken at face
value, this clears the 5% threshold for the **cast pair**.

However, the claim under evaluation is broader: "wrapper **cast + alloc**
overhead". Aggregate Marlin NVTX regions sum to 33.1% of decode, Marlin
kernel body is 15.7%, leaving 17.4 pp "outside the kernel" inside Marlin
regions. Of that 17.4 pp:

- 2.0 pp is the cast pair (direct kernel attribution above).
- The remaining **~15.4 pp is not broken down** at sub-range granularity
  in any current artifact. It is plausibly `ucopy_bf16` (8.6% total, but
  Marlin regions are not the only ucopy_bf16 callsite — PR #219 already
  ruled out per-site ucopy_bf16 fusion), `contiguous()` copies, the FP16
  output buffer `Tensor::zeros` alloc, workspace zeros, Candle stream
  sync, or dispatch launch overhead amortized by CUDA graph replay.

No NVTX sub-range capture (`:kiln/marlin/cast_in`, `:kiln/marlin/alloc`,
`:kiln/marlin/kernel`, `:kiln/marlin/cast_out`) exists in the
`profiling-artifacts/` archive. `grep -rn "cast_bf16_f16|cast_f16_bf16"
PROFILING.md` returns two references to the kernel-level attribution, and
no sub-range wrapper breakdown. The **15.4 pp of overhead that PR #224
attributed to "allocation, casts, layout, and stream sync"** is therefore
an arithmetic derivation (region total − kernel body), not a measurement.

Per this preflight's own acceptance rule (any of the three checks fails →
KILL), this ambiguity alone does not kill — the cast-pair fraction clears
5%. But it caps the honest math-ceiling to 2.0 pp definitely-eliminable,
not 17.4 pp speculatively-eliminable. **This becomes decisive in Check 3.**

Noting for the queue: a minimal nsys re-capture on a warm pod with NVTX
sub-ranges in `kiln-marlin-gemm/src/lib.rs` around the `to_dtype` /
`Tensor::zeros` / `kiln_marlin_w4a16_gemm` call edges would close this gap
permanently, but the Check 3 result below makes that re-capture not worth
a pod idle slot — the HBM analysis settles the question from the other
side.

### Check 3 — HBM-traffic reduction compounds under CUDA graph replay: **FAIL**

This is the decisive check. kiln runs with `KILN_CUDA_GRAPHS=true` by
default — decode-step kernel dispatch is captured into a graph and
replayed, so launch-dispatch savings amortize to near-zero and only HBM
traffic matters. This is the exact failure mode of null PRs
[#141](https://github.com/ericflo/kiln/pull/141) (gated_rms_norm),
[#173](https://github.com/ericflo/kiln/pull/173) (fused L2 qk_norm),
[#176](https://github.com/ericflo/kiln/pull/176) (big-fusion across
recurrent + qk_norm + gated_norm).

HBM-traffic arithmetic at m=1 decode for a representative Marlin call
(hidden_dim=2560, one of the 4-bit MLP / GDN projections with k=n=2560):

```
cast activation tensor (BF16 → FP16):
  read:  2560 × 2 B = 5,120 B
  write: 2560 × 2 B = 5,120 B
  per cast:          10,240 B
  both casts:        20,480 B per Marlin call

quantized weight read inside Marlin kernel:
  2560 × 2560 × 0.5 B (4-bit packed) ≈ 3,276,800 B ≈ 3.2 MB per call

activation read inside Marlin kernel (k=2560 BF16 / FP16):
  ≈ 5,120 B per call

wrapper HBM / total Marlin HBM per call:
  20,480 / (3,276,800 + 5,120) ≈ 0.62 %
```

Aggregate across all 13,527 Marlin invocations per decode run:

```
cast HBM traffic:    13,527 × 20,480 B ≈  277 MB
Marlin weight read:  13,527 × 3.28 MB  ≈ 44.4 GB
cast share of Marlin HBM:   0.62 %
```

At 2.0 pp of total decode attributable to the cast pair and dispatch-free
graph replay, the **HBM-only saving is ≤0.62% of Marlin traffic × 33.1%
aggregate Marlin region share = 0.21% of decode HBM budget.** The 2.0 pp
cast kernel wall-clock is dominated by per-launch setup cost that graph
replay already amortizes, not by the HBM traffic the epilogue would
eliminate. Under graph replay, the steady-state gain collapses from "2.0
pp removable" to "≤0.2 pp HBM-attributable" — **below measurement noise on
the ±3-run median.**

The FP16 intermediate output buffer alloc (`Tensor::zeros((m,n), DType::F16)`
at `kiln-marlin-gemm/src/lib.rs:173`) is also not HBM work — it is CUDA
malloc from Candle's caching allocator and is already amortized across
graph replay. Eliminating it removes allocator pressure, not HBM traffic.

This is the same structural failure that killed the last three big-fusion
attempts. **Check 3 fails.** The BF16-native epilogue's HBM contribution
does not compound; it evaporates.

### Math-ceiling re-derivation — all scenarios below 1.05× floor

| scenario | eliminated pp of decode | end-to-end speedup |
| --- | ---: | ---: |
| cast pair only (kernel wall-clock) | 2.0 | 1.020× |
| cast pair + FP16 alloc (speculative, no sub-range evidence) | ~3.0 | 1.031× |
| PROFILING.md #224 optimistic: 10% of 33.1% region | 3.3 | 1.034× |
| honest HBM-only under graph replay | ≤0.2 | ≤1.002× |

The 1.05× Phase 6 decode-speedup floor requires ≥4.76 pp of decode
time eliminated. **No scenario with honest evidence reaches that
threshold.** The most generous paper-math number from the #224 audit
(3.3 pp) still sits 30% below the floor, and that number collapses further
under graph replay.

### Decision: **KILL. Phase 6 is complete.**

The Marlin wrapper → BF16-native I/O epilogue has been vetted against the
three preflight checks and fails on Check 3 decisively. The
math-ceiling under honest assumptions is 1.02–1.03× end-to-end — below the
Phase 6 1.05× queue gate — and the HBM-traffic analysis confirms this is
not a graph-replay artifact but a structural ceiling set by activation
tensor size (5 KB BF16) vs weight tensor size (3.2 MB packed 4-bit).

With this KILL, the exhaustion matrix from PR #224 is now fully closed:
every standalone region in the post-#210 NVTX top-10 either has a shipped
fusion, a closed-null attempt, an architectural rejection with full
rationale, or a preflight KILL with math-ceiling documented. **No
remaining standalone decode-kernel lever above the 1.05× floor exists.**

### Phase 6 closure — shipped and closed-null inventory

Landed fused kernels (decode path):

- [PR #80](https://github.com/ericflo/kiln/pull/80) `kiln-gdn-kernel` —
  `chunk_gla_fwd` + recurrent GDN fwd vendored.
- [PR #92](https://github.com/ericflo/kiln/pull/92) GDN fused forward
  dispatch.
- [PR #133](https://github.com/ericflo/kiln/pull/133) `kiln-rmsnorm-kernel`
  — fused pre-norm RMSNorm.
- [PR #146](https://github.com/ericflo/kiln/pull/146) `kiln-marlin-gemm` —
  Marlin W4A16 vendored (FP16-only, where this preflight KILL leaves it).
- [PR #158](https://github.com/ericflo/kiln/pull/158) GDN gate fusion.
- [PR #166](https://github.com/ericflo/kiln/pull/166) `kiln-conv1d-kernel`
  — `causal_conv1d_update` vendored.

Closed-null / doc-only redirects:

- [PR #131](https://github.com/ericflo/kiln/pull/131) — `chunk_gla_fwd`
  already vendored (redirect).
- [PR #141](https://github.com/ericflo/kiln/pull/141) — `gated_rms_norm`
  fusion null under graph replay.
- [PR #163](https://github.com/ericflo/kiln/pull/163) — FlashInfer paged
  GQA decode, math-ceiling ≤1.005× redirect.
- [PR #164](https://github.com/ericflo/kiln/pull/164) — GDN gate-path
  single-kernel fusion architecturally infeasible.
- [PR #170](https://github.com/ericflo/kiln/pull/170) — `fused_recurrent`
  already done.
- [PR #173](https://github.com/ericflo/kiln/pull/173) — fused L2 qk_norm,
  null median, shipped opt-in for variance reduction.
- [PR #176](https://github.com/ericflo/kiln/pull/176) — big-fusion across
  recurrent + qk_norm + gated_norm, null ($14.99 burn).
- [PR #219](https://github.com/ericflo/kiln/pull/219) — `ucopy_bf16`
  per-site audit null.
- [PR #222](https://github.com/ericflo/kiln/pull/222) — FP8 KV cache
  verified, opt-in.
- [PR #223](https://github.com/ericflo/kiln/pull/223) — MLP gate/up Marlin
  fusion math-ceiling KILL.
- [PR #224](https://github.com/ericflo/kiln/pull/224) — Phase 6 frontier
  audit, identified this preflight as the last candidate.
- **This PR** — Marlin wrapper BF16-native I/O epilogue KILL.

Phase 6 final decode bench baseline (post-#166, A6000, `KILN_W4A16=1`,
CUDA graphs ON, 512-prompt × 128-decode, median of 3): **49.76 tok/s,
20.10 ms mean ITL, 25.46 ms p99 ITL.** This is the Phase 7 starting
baseline — any Phase 7 work that unintentionally regresses decode tok/s
below this median should be flagged.

### Phase 7 transition — two well-defined starting candidates

With Phase 6 closed, two Phase 7 work items are ready to pick up. They are
**capability / DX** levers, not decode-kernel levers — Phase 7 is a
deliberate shift away from the raw tok/s axis that Phase 6 nearly
exhausted.

**Phase 7 candidate A — GDN prefill memory reduction (long-context
capability).** The redirect destination from PR #222. FP8 KV cache
verification showed long-context OOM is set by the GDN prefill-state
allocation, not the GQA KV cache. Relevant regions:
`:kiln/gdn/recur_prep` (0.8 % of decode, prefill-dominant),
`:kiln/gdn/head_expand` (3.4 % of decode), plus prefill-only state
materialization. Target capability: unlock ≥65,536 prompt tokens on a
single A6000 with `KILN_W4A16=1` + `KILN_KV_CACHE_FP8=1`. Preflight work:
measure current GDN prefill peak memory via `nvidia-smi dmon` during a
32k-prompt eval, identify the specific tensor residencies that force the
OOM ceiling, and propose a streaming / chunked prefill-state computation
that bounds peak memory. This is not a decode-speed lever and should not
be measured on decode tok/s.

**Phase 7 candidate B — self-speculative decode end-to-end bench + server
flag.** Per agent note `kiln-speculative-decoding-design`, self-spec
decoding is already implemented via the skip-layer approach (no separate
draft model required — the same Qwen3.5-4B weights serve both draft and
verify paths at different layer depths). The remaining work is
benchmarking it end-to-end on the canonical 512-prompt × 128-decode bench
and exposing it as a server runtime flag. This **is** a decode-speed lever
(potentially 1.5–2× on tok/s if acceptance rates hold above ~70%), but it
is algorithmic rather than kernel-fusion — it belongs in Phase 7 because
it changes the scheduler / generation loop, not a kernel crate. Preflight
work: acceptance-rate sweep across canonical prompts, pick a skip depth,
queue a pod for the bench. This candidate may reopen the raw-tok/s axis
that Phase 6 retired, but under a different set of architectural
assumptions (speculation-friendly prompts, configurable trade-off at
serving time).

**Recommendation:** pick Candidate A as the Phase 7 opener. It is the
capability-side unlock that FP8 KV cache (PR #222) explicitly redirected
to, it has no overlap with any Phase 6 kernel work, and it directly
addresses a customer-visible ceiling (context length) rather than a
benchmark-visible metric (tok/s at 128 decode). Candidate B is
reasonable to queue in parallel as a second-track preflight, but its
payoff depends on acceptance-rate measurement that has not been done yet,
whereas Candidate A's payoff is bounded by the GDN prefill allocation
size, which is measurable on the next pod acquisition.

### Preconditions for reopening Phase 6

Should this KILL be revisited, the bar to re-queue any Marlin wrapper /
BF16 epilogue work is:

1. A new nsys capture with NVTX sub-ranges **inside the Marlin wrapper**
   that breaks out per-call cast, alloc, contiguous, kernel, and dispatch
   stages, producing a real sub-range wall-clock attribution for the 17.4
   pp "outside kernel" bucket.
2. Evidence that at least one sub-range stage (a) exceeds 5 % of the
   region's wall-clock **and** (b) represents actual HBM traffic rather
   than launch-dispatch cost that graph replay amortizes.
3. An m > 1 decode scenario (continuous batching, speculative decoding
   verify pass) where the wrapper cast amortizes differently. All Phase 6
   work to date was m = 1 single-stream decode; at m = 8+ the activation
   tensor scales linearly and the HBM share of the cast pair could rise
   above the 0.62 % per-call threshold that killed this preflight.

Absent new evidence meeting those conditions, do not re-queue this
candidate. Phase 6 is closed.


## Phase 6 — Frontier audit post-#223 (2026-04-20)

**Outcome: Phase 6 decode-kernel frontier is nearly exhausted.** Every
standalone region in the post-#210 NVTX top-10 either already has a shipped
fusion, has a closed-null attempt on record, is quantized (Marlin), is an
ABI-required layout transform with no compute-reducing fix, or sits
sub-floor at the 1.05× decode-speedup queue gate. The only remaining
quantifiable decode-speed lever is the **Marlin wrapper → BF16-native I/O
epilogue** (redirect #2 from the #223 KILL): eliminate the per-call
BF16↔FP16 cast pair around every Marlin GEMM in the forward path. This
audit recommends one $0 preflight on that lever next, and if it fails to
clear the floor, declares Phase 6 complete and advances to Phase 7
(DX / capability). No pod spent, no code changed.

Precedent for doc-only preflight / redirect PRs:
[#131](https://github.com/ericflo/kiln/pull/131) (chunk_gla_fwd already
vendored), [#163](https://github.com/ericflo/kiln/pull/163) (FlashInfer
paged GQA decode redirect), [#164](https://github.com/ericflo/kiln/pull/164)
(GDN gate-path fusion architecturally infeasible),
[#170](https://github.com/ericflo/kiln/pull/170) (fused_recurrent already
done), [#219](https://github.com/ericflo/kiln/pull/219) (ucopy_bf16 null),
[#223](https://github.com/ericflo/kiln/pull/223) (MLP gate/up Marlin
fusion KILL).

### Why this audit

Three post-#210 follow-up items were sequenced after the post-#210 decode
profile identified no high-ceiling standalone fusion:

1. **ucopy_bf16 source-site audit** — resolved by PR #219: null, every
   un-addressed `ucopy_bf16` site is either below the 1.05× floor or is an
   ABI-required layout transform for a downstream kernel (`causal_conv1d_update`,
   GQA matmul, chunkwise GDN recurrence). No queueable fusion.
2. **KV cache FP8 verification** — resolved by PR #222: `KILN_KV_CACHE_FP8=1`
   is bit-identical and within ±3% of BF16 on workable prompt lengths, but
   the long-context OOM ceiling is set by GDN prefill state, not the GQA KV
   cache. Kept opt-in (default `false`); not a decode-speed lever.
3. **MLP gate/up Marlin merge** — resolved by PR #223: KILL. Math-ceiling
   fails by a wide margin even under heroic assumptions. Flagged two
   redirects: (a) GDN prefill memory reduction (capability, not decode
   speed) and (b) **Marlin wrapper → BF16-native I/O epilogue** (decode
   speed candidate).

With all three redirects resolved, the Phase 6 frontier question is now:
*is there any remaining standalone decode-kernel lever above the 1.05×
floor?* This audit walks the post-#210 NVTX top-10 one more time against
every closed PR, every preflight doc, and every architecturally-rejected
candidate, and answers the question concretely.

### Post-#210 NVTX top-10 — exhaustion matrix

No forward-path code has landed since the post-#210 capture (PRs #219,
#222, #223 are all doc-only / non-decode-path). The NVTX composition is
therefore unchanged. For each region, this table maps the current prior-PR
coverage and the math-ceiling verdict at 1.10× local speedup.

| %    | region                     | prior coverage                                                               | compute lever remaining?                                                      | 1.10× math-ceiling | verdict                                  |
| ---: | -------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -----------------: | ---------------------------------------- |
| 18.5 | `:kiln/gdn/gates`          | **PR #158 merged** (fused Step 6 gate kernel)                                | already fused; no further compute lever                                       | — (already fused)  | **do not queue**                         |
| 17.6 | `:kiln/gdn/gated_norm`     | **PR #141 closed null** (graph-replay dispatch amortization ate the win)     | would need a memory-reducing approach, not a fusion                           | 0.016 (sub-floor)  | **do not queue** without new evidence    |
| 14.7 | `:kiln/gdn/qk_norm`        | **PR #173 opt-in**, null median (1.0093× point estimate, variance only)      | already tried; fused kernel exists behind `KILN_ENABLE_FUSED_L2_QK_NORM`      | 0.013 (sub-floor)  | **do not queue**                         |
|  7.7 | `:kiln/gdn/in_proj`        | Marlin W4A16 (PR #146-era quantization)                                      | GEMM shape change, not fusion; activations-load-bound at m=1                  | 0.007 (sub-floor)  | **do not queue** as standalone           |
|  6.6 | `:kiln/mlp/gate`           | Marlin W4A16; **PR #223 KILL** (gate/up merge cannot clear floor)            | already quantized; merge-with-up killed                                       | 0.006 (sub-floor)  | **do not queue**                         |
|  6.2 | `:kiln/mlp/up`             | Marlin W4A16; **PR #223 KILL**                                               | already quantized; merge-with-gate killed                                     | 0.006 (sub-floor)  | **do not queue**                         |
|  6.0 | `:kiln/mlp/down`           | Marlin W4A16                                                                 | SwiGLU-down fuse impossible (distinct weight, distinct input); standalone sub-floor | 0.005 (sub-floor) | **do not queue**                         |
|  3.4 | `:kiln/gdn/head_expand`    | **PR #219 null** (`ucopy_bf16` site 2); ABI-required broadcast materialization | downstream GDN matmul cannot accept broadcast-strided K/V                     | 0.003 (sub-floor)  | **do not queue**                         |
|  3.4 | `:kiln/residual`           | none                                                                         | potential in-place ADD into RMSNorm epilogue (add+norm fusion)                | 0.003 (sub-floor)  | **do not queue** standalone              |
|  3.0 | `:kiln/proj/qkv` full-attn | Marlin W4A16; **PR #163 preflight** ruled FlashInfer out (full-attn 3.8%)    | full-attn total 3.8% of decode; math ceiling ≤1.038× at infinite speedup      | 0.003 (sub-floor)  | **do not queue**                         |

Sub-top-10 (each ≤3%, each sub-floor even at infinite speedup): `gdn/out_proj`
2.8, `norm/pre_mlp` 2.2, `attn/gdn/recurrent` 2.0 (already vendored in #80),
`norm/pre_attn` 1.9, `gdn/conv` 1.8 (ABI-bound per #219), `proj/o` 0.8,
`gdn/recur_prep` 0.8 (ABI-bound per #219).

**Conclusion:** No standalone NVTX region is queueable as a pure fusion
task. Every realistic fusion candidate either has a prior null attempt, is
ABI-bound to a downstream kernel input contract, or is sub-floor under
graph replay.

### Un-attempted cross-cutting candidates — math-ceiling

Three candidates cut across multiple regions and were not evaluated as
standalone-region items above. This section sizes each one at the
decode-level Amdahl bound and rules on queueability.

**Candidate A — `add + RMSNorm` epilogue fusion (residual into post-attn /
post-MLP norm).** The transformer block at `crates/kiln-model/src/forward.rs`
lines ~2460–2509 issues `(x + attn_out)` → `rms_norm(x, ...)` and
`(x + ffn_out)` → `rms_norm(x, ...)` as separate NVTX ranges. Regions
involved: `:kiln/residual` (3.4%) + `:kiln/norm/pre_attn` (1.9%) +
`:kiln/norm/pre_mlp` (2.2%) = **7.5%**. A fused kernel that consumes the
residual and runs the normalization in the same pass could elide one
full activation read and one full activation write per residual-into-norm
pair.

Math-ceiling: assume generous 1.5× local speedup (memory-bound, not
dispatch-bound, so graph replay does not fully amortize the win):

```
contribution = 0.075 · (1 − 1/1.5) = 0.025
```

Sub-floor. Even at 2.0× local speedup (unrealistic — the norm is already
the #133 fused kernel and can't be sped up much further):

```
contribution = 0.075 · (1 − 1/2.0) = 0.0375
```

Still sub-floor. Verdict: **not queueable.** This is structurally the
same-shape idea as #141 (gated_norm fusion) which already closed null,
and the graph-replay dispatch-amortization lesson applies.

**Candidate B — Marlin wrapper → BF16-native I/O epilogue.** Flagged by PR
#223 as redirect #2. Each of the 12 Marlin callsites per layer (GDN
`in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b`/`out_proj`; full-attn
`q`/`k`/`v`/`o`; MLP `gate`/`up`/`down`) currently wraps the FP16-only
Marlin kernel with:

1. `a.to_dtype(DType::F16).contiguous()` — BF16→FP16 activation cast
2. Allocate FP16 output `Tensor::zeros((m, n), DType::F16)`
3. Allocate workspace zeros
4. Kernel launch
5. `c_fp16.to_dtype(DType::BF16)` — FP16→BF16 output cast

Steps 1 and 5 emit `cast_bf16_f32`-style kernels that show up in the
post-#210 kernel top-20 (`cast_f32_bf16` 2.5% + `ucopy_bf16` 8.6% includes
the contiguous step + FP16 casts visible in the elementwise zoo). Regions
that end in a Marlin GEMM sum to:

```
gdn/in_proj        7.7
gdn/out_proj       2.8
mlp/gate           6.6
mlp/up             6.2
mlp/down           6.0
proj/qkv (full)    3.0
proj/o (full)      0.8
────────────────────
total             33.1
```

Of each region's wall-clock, the wrapper cast pair + allocations are not
the kernel itself; post-#210 kernel-level attribution puts `Marlin` at
15.7% (the kernel proper) vs region total 33.1%, so ~17.4 pp of decode is
spent outside the Marlin kernel in those regions — allocation, casts,
layout, and stream sync. An epilogue variant of Marlin that accepts BF16
input directly (via an FP16→BF16 input reader in the load stage) and
writes BF16 output directly (via an FP32→BF16 accumulator epilogue) would
eliminate steps 1 and 5 across all 12 callsites simultaneously.

Realistic local speedup: eliminating cast + ucopy + FP16 buffer
alloc/free per call is a ~5–10% reduction of each region's wall-clock at
m=1 decode (where the cast + alloc amortizes poorly). Take the
conservative end: 5% of 33.1% = 1.66 pp decode reduction, contribution
0.0166. Take the optimistic end: 10% of 33.1% = 3.31 pp, contribution
0.0331. At the mid-point (7.5% local):

```
contribution = 0.331 · 0.075 ≈ 0.025
```

This is on the edge of the 0.05 floor but plausibly clears it under two
assumptions: (a) the cast-elimination also removes the `cast_f32_bf16` /
`ucopy_bf16` memory pressure on the HBM, compounding under graph replay;
(b) the BF16-native accumulator epilogue can be written to avoid the FP32
intermediate buffer allocation as well. Under those assumptions, a 10%
local speedup on a 33.1% aggregate is the realistic target, and
contribution ≈ 0.033 at median with a plausible tail reaching 0.05+ if
HBM-traffic reduction compounds.

**This is the single remaining quantifiable decode-speed lever.** It is
also the only un-attempted candidate where the math-ceiling is plausibly
close to the 1.05× floor with a non-zero chance of clearing it. It
requires non-trivial kernel work (Marlin is FP16-only by upstream design;
adding BF16-native I/O requires modifying the vendored kernel at
`crates/kiln-marlin-gemm/src/lib.rs` to emit a new tile variant), but the
surface area is well-defined and the win is shared across 12 callsites.

Verdict: **queue as the next Phase 6 slice, but gate it on a $0
preflight.** Before any pod $, preflight needs to:

1. Confirm that BF16-native tile variants are compatible with Marlin's
   existing `Marlin<256,1,8,8,4,8>` tile structure (the vLLM Marlin-BF16
   variant `MarlinBF16<...>` is precedent; does it exist in vendored form?).
2. Confirm that the per-call wrapper overhead (cast + alloc) is at least
   ~5% of each GEMM's wall-clock via a targeted nsys capture with NVTX
   sub-ranges on the wrapper stages. If it is <2%, the math-ceiling kills
   this candidate too.
3. Confirm the elementwise-zoo `cast_f32_bf16` (2.5%) and
   `ucopy_bf16` (8.6%) reductions that would follow are not already
   addressed by some upstream Candle optimization (the audit in #219
   already scoped `ucopy_bf16` and ruled out the per-site lever; the
   wrapper-cast bucket is a different set of invocations).

**Candidate C — GDN prefill memory reduction (long-context capability,
not decode speed).** Flagged by PR #222 as the redirect destination after
FP8 KV cache verification showed that long-context OOM is set by GDN
prefill state, not the GQA KV cache. Regions involved: `:kiln/gdn/recur_prep`
(0.8% decode) + prefill-only paths (not decode top-10). This is **not a
decode-speed lever** — it is a capability unlock. At m=1 decode it
contributes ≤0.8% of wall-clock and sits deep below the floor.

Verdict: **not a decode-speed queue item.** If pursued, it is a Phase 7
capability work item, not a Phase 6 decode-kernel slice.

### Math-ceiling summary

| candidate                                    | aggregate % | realistic speedup | contribution | decision                     |
| -------------------------------------------- | ----------: | ----------------: | -----------: | ---------------------------- |
| Add+RMSNorm epilogue fusion (residual→norm)  |         7.5 |              1.5× |        0.025 | **do not queue** (sub-floor) |
| Marlin wrapper → BF16-native I/O epilogue    |        33.1 | 1.05–1.10× (region-local) | 0.017–0.033 | **preflight next** (plausibly clears 0.05) |
| GDN prefill memory reduction                 |    ≤0.8 decode | — (capability)    |          n/a | **not a decode slice**       |

### Recommendation

**Next Phase 6 slice: Marlin wrapper → BF16-native I/O epilogue preflight
($0, doc-only).** The preflight should verify:

1. Upstream Marlin has a BF16-native tile variant in the vLLM reference
   (if yes, vendoring path is precedented like `kiln-gdn-kernel` from #80).
2. The wrapper cast + alloc overhead on the m=1 decode path is ≥5% of
   each GEMM's region wall-clock, measured with NVTX sub-ranges on the
   wrapper stages (no pod $; run on an already-warm bench pod or during
   the next re-profile).
3. The expected HBM-traffic reduction (one fewer BF16→FP16 cast kernel
   launch per GEMM × 12 GEMMs × 32 layers = 384 kernel launches
   eliminated per decode step) compounds or dilutes under graph replay
   via a nsys-reported kernel-launch-count delta projection.

If any of (1)–(3) fails, issue a doc-only redirect PR (precedent: #131,
#163, #164, #170, #219, #223) and **declare Phase 6 complete**. At that
point the recommendation is to advance to Phase 7, which has two
well-defined work items waiting:

- **Capability:** GDN prefill memory reduction for long-context (redirect
  from #222). Unlocks ≥65536 prompt tokens, which FP8 KV cache verified
  is currently gated by GDN prefill state, not the KV cache.
- **DX:** self-speculative decoding is already implemented via the
  skip-layer approach (per agent note `kiln-speculative-decoding-design`);
  the remaining work is benchmarking it end-to-end and exposing it as a
  server option.

Either item is a reasonable Phase 7 starting point; the preflight outcome
on Marlin BF16-native I/O will determine whether Phase 6 needs one more
slice first.

### Do-NOT-queue list (restated, cumulative)

These candidates have all been evaluated and closed; do not re-queue
without new profiling evidence that materially changes their
math-ceiling. The prior-PR or prior-preflight reference is the required
cross-check:

- **`:kiln/gdn/gates`** — PR #158 merged (fused).
- **`:kiln/gdn/gated_norm`** standalone — PR #141 closed null.
- **`:kiln/gdn/qk_norm`** standalone — PR #173 opt-in, null median.
- **Big-fusion (`qk_norm + gated_norm + recurrent`)** — PR #176 closed
  null ($14.99 burn); Step 7 architecturally separates Step 6 and Step 8.
- **FlashInfer paged GQA decode** — PR #163 preflight: full-attn ≤3.8%
  of decode; math ceiling ≤1.038× even at infinite speedup.
- **Gate-path fusion across Step 7 recurrence** — PR #164 preflight:
  architecturally infeasible; both halves already addressed independently.
- **`ucopy_bf16` source-site consolidation** — PR #219 null: every
  un-addressed site is ABI-required or sub-floor.
- **MLP gate/up Marlin merge** — PR #223 KILL: math-ceiling fails by wide
  margin under CUDA graph replay.
- **KV cache FP8 as decode-speed lever** — PR #222 verification: bit-
  identical, ±3% speed, not a context-length unlock on current path. Kept
  opt-in; not a decode-speed queue item.
- **Add+RMSNorm epilogue fusion** — this audit: 0.025 contribution at
  generous 1.5×, sub-floor under graph replay (same shape as #141).

### Artifacts

This is a doc-only PR. No bench was run; no pod was acquired. Total
compute cost: $0. The audit relied on:

- The existing post-#210 profile
  (`profiling-artifacts/post210_nvtx.csv`, `profiling-artifacts/post210_kern.csv`).
- The post-#222 FP8 KV cache verification raw data
  (`profiling-artifacts/fp8_kv_verify_2026-04-20.csv`).
- Source reads of `crates/kiln-model/src/forward.rs`,
  `crates/kiln-model/src/marlin_proj.rs`, and
  `crates/kiln-marlin-gemm/src/lib.rs` at the current `main` HEAD.
- PR history cross-check via `gh pr list --repo ericflo/kiln --state all`
  through PR #223.


## Phase 6 — Post-#222 preflight: MLP gate/up Marlin fusion math-ceiling audit (2026-04-20)

**Outcome: KILL.** Fusing the MLP `gate_proj` and `up_proj` Marlin W4A16
GEMMs into a single stacked-n projection (the vLLM
`MergedColumnParallelLinear` / `gate_up_proj` pattern) cannot clear the
Phase 6 ≥1.05× end-to-end decode speedup floor on Qwen3.5-4B under CUDA
graph replay. Math-ceiling fails by a wide margin even under heroic
assumptions. This preflight is doc-only; no pod spent, no code changed.
Redirects below.

Precedent for doc-only redirect PRs: [#131](https://github.com/ericflo/kiln/pull/131)
(chunk_gla_fwd already vendored), [#163](https://github.com/ericflo/kiln/pull/163)
(FlashInfer paged GQA decode redirect), [#164](https://github.com/ericflo/kiln/pull/164)
(GDN gate-path fusion architecturally infeasible),
[#170](https://github.com/ericflo/kiln/pull/170) (fused_recurrent already done),
[#219](https://github.com/ericflo/kiln/pull/219) (ucopy_bf16 null).

### Source sites (current, untouched)

The current MLP forward path, `crates/kiln-model/src/forward.rs` lines
770–817 (`swiglu_ffn`), issues **two independent Marlin calls on the same
activation `x`** inside separately-named NVTX regions:

```
gate = mlp_proj_forward(x, gate_proj_t, gate_proj_marlin, ...)   // :kiln/mlp/gate
gate = cuda_silu(gate)
up   = mlp_proj_forward(x, up_proj_t,   up_proj_marlin,   ...)   // :kiln/mlp/up
hidden = gate * up
out  = mlp_proj_forward(hidden, down_proj_t, down_proj_marlin, ...) // :kiln/mlp/down
```

`mlp_proj_forward` dispatches to `marlin_proj::matmul_bf16`
(`crates/kiln-model/src/marlin_proj.rs`), whose per-call wrapper is:

1. `a.to_dtype(DType::F16).contiguous()` — BF16→FP16 activation cast.
2. Allocate FP16 output `Tensor::zeros((m, n), DType::F16)`.
3. Allocate workspace zeros.
4. Kernel launch (`Marlin<256,1,8,8,4,8>` at m=1 decode).
5. `c_fp16.to_dtype(DType::BF16)` — FP16→BF16 output cast.

The Marlin kernel itself (`crates/kiln-marlin-gemm/src/lib.rs`) is
**FP16-only, single-input, single-output** with constraints `k % 128 == 0`,
`n % 256 == 0`, `groupsize ∈ {-1, 128}`. A fused "gate+up" stacked-n
projection would need either (a) kernel-surface changes to accept a single
weight of width `2 × intermediate_size` and emit two halves, or (b) an
epilogue-split variant that splits the m=1 output into two n-tiles. Both
fit Marlin's tile structure (n=256 tiles; `2 × 9216 = 18432` is a multiple
of 256) but both require non-trivial kernel work.

### Profile extract (post-#210 steady-state decode)

From the most recent re-profile in this file (section "Post-#210 re-profile,
decode hot regions"), Arm B `KILN_W4A16=1` production decode on A6000,
512-prompt × 128-decode, median of 3:

| NVTX region          | % decode |   median ns | calls  |
|----------------------|---------:|------------:|-------:|
| `:kiln/mlp/gate`     |   6.6%   |    52 591   |    ... |
| `:kiln/mlp/up`       |   6.2%   |    50 035   |    ... |
| `:kiln/mlp/down`     |   6.0%   |      ...    |    ... |
| **gate + up (sum)**  | **12.8%**|             |        |

Combined CUDA-kernel attribution for the Marlin kernel (all six W4A16
projections per layer — q/gate/up/down + GDN in_proj × 2 — over 32 layers ×
128 decode steps) is 15.7% of decode at ~20.3 μs × 13 527 invocations. The
gate+up slice of that is **2/6 × 15.7% ≈ 5.2% of decode** spent inside the
Marlin kernel body itself for gate+up.

### Math-ceiling closed form

For a fusion that touches `region_pct` of decode and achieves speedup `s`
on that region under CUDA graph replay (launch-amortization ≈ 0 —
confirmed by the null results in PRs [#141](https://github.com/ericflo/kiln/pull/141),
[#173](https://github.com/ericflo/kiln/pull/173), [#176](https://github.com/ericflo/kiln/pull/176),
[#164](https://github.com/ericflo/kiln/pull/164)):

```
end_to_end_speedup = 1 / (1 − region_pct × (1 − 1/s))
```

To clear the Phase 6 ≥1.05× decode floor with `region_pct = 0.128`:

```
1 − 1/s ≥ 0.05 / 0.128           (approx, 1/(1-0.05 × ...))
1 − 1/s ≥ 0.391
1/s     ≤ 0.609
s       ≥ 1.642×
```

**Gate+up region must get ≥1.64× faster** in median wall-clock under
graph replay to move the end-to-end decode needle by 5%. (The exact Amdahl
form gives s ≥ 1.641×; the inequality above is the algebra check.)

### Bandwidth / arithmetic intensity at m=1 decode

Qwen3.5-4B: `hidden_size = 2560`, `intermediate_size = 9216`
(`crates/kiln-core/src/config.rs:86`), W4A16 packed weights, BF16
activations.

Per `gate_proj` or `up_proj` call at m=1:

- **Weight bytes (the dominant term):** W4A16 packed = 4 bits/weight
  + scales (group 128) ≈ `2560 × 9216 × 4/8 = 11 796 480` bytes
  ≈ **11.25 MiB / call** of weight traffic from GDDR6X. Plus scales,
  roughly 11.8 MB total.
- **Activation bytes (per call):** `x` is `[1, 2560]` BF16 = 5 120 bytes
  ≈ **5 KB / call**. Negligible vs weights (0.04% of the transfer).
- **Output bytes:** `[1, 9216]` FP16 = 18 432 bytes ≈ **18 KB / call**.
- **Arithmetic:** `2 × 2560 × 9216 × 1 = 47.2 MFLOPs / call`.

At ~384 GB/s HBM-equivalent effective bandwidth observable to a single
SM cluster on A6000, 11.8 MB takes ~31 μs in pure-streaming limit. The
measured median is ~52 μs, i.e. ~60% of time is weight-streaming and the
remainder is cast/alloc/launch-overhead + tile scheduling.

**Fusion savings ceiling (what fusion can plausibly remove):**

1. **One BF16→FP16 activation cast.** Both current calls cast the *same*
   `x`. A fused path casts once. Saves ~`5 KB / 384 GB/s ≈ 13 ns` of bandwidth
   — but the cast kernel's fixed cost (launch, RMS of a separate kernel) is
   what actually hurts; nsys shows `cast_bf16_f16` at ~1–2 μs amortized per
   invocation. Call it **~1–2 μs saved** across the two current calls.
2. **One workspace alloc + one output alloc.** Under graph replay these
   allocs are captured once, so savings ≈ 0 on the replayed-graph decode
   path. On the first (non-graph) step, savings are tens of μs, but that
   step is already outside the decode-steady measurement window.
3. **Marlin kernel tile reuse (the only real win).** A single stacked-n
   GEMM of width `2 × 9216 = 18432` streams `x` (5 KB) *once* across shared
   memory for both outputs, and reuses the same tile of activation across
   more n-tiles before needing a new one. At m=1 this changes ~nothing
   because `x` already fits in a single activation tile and stays resident;
   the kernel is **weight-bandwidth-bound, not activation-bandwidth-bound**.
   Tile fusion at m=1 saves ≈ 0 of the weight-streaming term — the
   dominant 60% of the region time.

**Upper-bound achievable s on gate+up at m=1 decode:**

- Remove activation cast + consolidate FP16 output alloc: **1.04–1.08×**.
- Add any Marlin kernel tile-level savings at m=1: another **0–0.03×**
  (bounded by the ~40% non-weight-streaming portion of the region time,
  most of which is launch + epilogue, not tile-reuse).
- **Realistic s ≤ 1.12× on region.** Plugged into the Amdahl form:

```
decode_speedup ≤ 1 / (1 − 0.128 × (1 − 1/1.12))
              = 1 / (1 − 0.128 × 0.107)
              = 1 / (1 − 0.0137)
              ≈ 1.0139×                         (well below 1.05× floor)
```

Even under heroic-and-wrong assumptions the ceiling fails:

| Assumed s on region | End-to-end decode speedup | vs 1.05× floor |
|--------------------:|--------------------------:|:---------------|
| 1.12×               | 1.014×                    | **FAIL (−0.036)** |
| 1.30×               | 1.030×                    | **FAIL (−0.020)** |
| 1.50×               | 1.045×                    | **FAIL (−0.005)** |
| 1.60×               | 1.048×                    | **FAIL (−0.002)** |
| 1.70×               | 1.053×                    | barely clears |
| 2.00×               | 1.068×                    | clears |

The weight-bandwidth argument puts the honest ceiling around 1.10×. The
1.70× needed to barely clear the floor would require either (a) m=1
arithmetic intensity that doesn't exist on W4A16 decode, or (b) a radically
different kernel (not a merge) that also changes the dequant or output dtype.

### External prior art

- **vLLM `MergedColumnParallelLinear` / `gate_up_proj`:** merges gate and
  up into a single stacked weight `[2 × intermediate, hidden]`. Reported
  wins at tensor-parallel batch=1 decode are typically **single-digit %
  on the MLP region**, i.e. 1.03–1.07× on region only — and tensor-parallel
  decode has higher all-reduce overhead that this also collapses, which
  kiln does not pay (single-GPU).
- **SGLang** adopts the same stacked pattern; same magnitude of win; wins
  are larger at prefill (where the region is compute-bound and activation
  tile reuse matters), not at m=1 decode.
- **In-kiln precedent:** GDN `in_proj_qkv` is already stacked across its
  three heads (the standard "QKV merge" pattern). That merge *was* worth it
  because at prefill time the region is compute-bound, and because the
  three sub-projections share gate/up/down's hidden dim but have different
  downstream consumers (saving the activation load pattern is meaningful).
  At m=1 decode, the same merge would show ≈ 0 win by the same
  weight-bandwidth argument used above — and indeed `:kiln/gdn/in_proj` at
  7.7% of decode has not been targeted for further merge work.

### Decision: KILL

Gate+up Marlin fusion is weight-bandwidth-bound at m=1 decode, fits
kiln's Marlin tile geometry but requires non-trivial kernel-surface work
(either width-`2n` weight or epilogue-split), and under CUDA graph replay
the only first-order savings (cast consolidation + alloc reuse) are
≈1–2 μs on a ~100 μs combined region. The math-ceiling floor (s ≥ 1.64×
on region for 1.05× end-to-end) is a factor of ~15× further than the
bandwidth-bound ceiling allows. Ship no code; no pod spent.

### Redirects (higher-ceiling next candidates)

1. **GDN prefill memory reduction.** The "KV cache FP8 verification"
   section below established that the OOM ceiling at ≥65536 prompt tokens
   is set by **GDN prefill state, not the GQA KV cache**. Shrinking GDN
   prefill state (via chunked streaming, state recompute on checkpoint,
   or reduced-precision state) unlocks the long-context memory story that
   FP8 KV cache did not deliver. High ceiling on capability; ceiling on
   decode-speed unclear but unimportant — this is a capability fix, not a
   decode-speed fix.

2. **Marlin wrapper → BF16-native I/O epilogue.** The per-call
   `a.to_dtype(DType::F16).contiguous()` + `c_fp16.to_dtype(DType::BF16)`
   pair in `marlin_proj::matmul_bf16` (`crates/kiln-model/src/marlin_proj.rs`)
   is applied on **every** Marlin call — q_proj + gate + up + down + GDN
   in_proj × 2, over 32 layers × 128 decode steps. Nsys post-#210 shows
   `cast_f32_bf16` at 2.5% of decode and `ucopy_bf16` at 8.6% of decode,
   both partially attributable to these wrappers. A Marlin-kernel
   epilogue that emits BF16 directly (and reads BF16 activations in the
   prologue) removes two cast kernels per projection call, addressing a
   larger slice of decode than gate/up merge and with a clearer
   bandwidth story. Requires kernel work but math-ceiling plausibly
   clears 1.05× on combined cast eliminations. **Preflight recommended
   before pod spend** — the same math-ceiling audit applied to the
   cast-elimination path.

3. **Marlin packing latency cleanup (~58 s at load)** and **Marlin BF16
   weight residency cleanup (+1.3 GB VRAM)** — both previously flagged in
   this file; neither is a decode-speed item but both are ship-ready
   low-risk cleanups that clear VRAM for the GDN prefill work in (1).

Do not re-queue the gate+up merge idea without a new profile that
materially changes the 12.8% region share or shows m=1 decode exiting
the weight-bandwidth-bound regime.


## Phase 6 — KV cache FP8 verification (2026-04-20)

**Outcome: keep `kv_cache_fp8` opt-in (default stays `false`).** Verified that
FP8 KV cache (PR #55, `kv_cache_fp8 = true`) produces bit-identical decoded
tokens and speed within ±3% of BF16 on workable prompt lengths, but the
promised long-context memory unlock does **not** materialize on the current
decode path — the OOM ceiling at ≥65536 prompt tokens is set by GDN prefill
state, not by the GQA KV cache, so halving the KV cache does not extend the
context ceiling on this GPU.

Raw per-run data: [`profiling-artifacts/fp8_kv_verify_2026-04-20.csv`](profiling-artifacts/fp8_kv_verify_2026-04-20.csv).

### Why this verification was run

Re-profile 2026-04-20 and the post-#210 profile both closed out kernel-fusion
targets without a high-ceiling next candidate. The `ucopy_bf16` audit (this
file's previous first section, and PR #217/#219) explicitly redirected
attention to **non-kernel decode axes**, calling out KV cache FP8 as a
specific item to verify before leaving it at the current opt-in default. This
run is that verification.

### Method

- Pod: pool-acquired A6000 on-demand (lease `pod-016bb2e7c07be9a97ceb4a3b`,
  pod `t0vmof6qkwostu`), `ghcr.io/ericflo/kiln-runpod:latest`.
- Build: `KILN_CUDA_ARCHS=86 cargo build --release --features cuda` (sccache
  warm; incremental rebuild after patching `bench.rs` to gate the throughput
  phase behind `KILN_BENCH_SKIP_THROUGHPUT=1` — latency-only, avoids OOM at
  the largest prompt sizes and shaves ~60–90 s per run).
- Arm: production decode (`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`), `--paged
  --max-output-tokens 128 --skip-training`, `KILN_BENCH_LOG_TOKENS=1` for a
  quality spot-check.
- Matrix: `KILN_KV_CACHE_FP8 ∈ {0, 1}` × prompt tokens
  `{512, 4096, 16384, 65536, 131072}` × 3 runs back-to-back = 30 runs.
- Metric reporting: median-of-3. Failed (OOM) runs reported as `FAILED` with
  the caused-by chain preserved in the raw logs.

### Median-of-3 results

| FP8 | ptok | runs | fail | prefill ms | decode tok/s | mean ITL ms | p99 ITL ms | model VRAM |
| --- | ---: | ---: | ---: | ---------: | -----------: | ----------: | ---------: | ---------: |
| 0   |   512 | 3 | 0 |   329.1 | **52.34** | 19.10 | 23.49 | 17528 MB |
| 1   |   512 | 3 | 0 |   332.2 | 50.52 | 19.79 | 22.78 | 17528 MB |
| 0   |  4096 | 3 | 0 |  1633.0 | **50.79** | 19.69 | 23.87 | 17528 MB |
| 1   |  4096 | 3 | 0 |  1620.1 | 49.31 | 20.28 | 24.23 | 17528 MB |
| 0   | 16384 | 3 | 0 |  6316.8 | 43.90 | 22.78 | 23.85 | 17528 MB |
| 1   | 16384 | 3 | 0 |  6495.4 | **43.70** | 22.89 | 24.10 | 17528 MB |
| 0   | 65536 | 3 | **3** | — | OOM | — | — | — |
| 1   | 65536 | 3 | **3** | — | OOM | — | — | — |
| 0   | 131072 | 3 | **3** | — | OOM | — | — | — |
| 1   | 131072 | 3 | **3** | — | OOM | — | — | — |

Decode-tok/s deltas FP8 vs BF16 (positive = FP8 faster):

- 512p: −3.5% (run-1 warmup noise dominates; runs 2/3 land at 52.3–52.8)
- 4096p: −2.9%
- 16384p: **−0.5%** (within run-to-run noise)

Model-weight VRAM (`model_load.model_vram_mb`) is identical at 17528 MB in
both arms, as expected — FP8 affects KV cache only, and the paged KV cache
allocates per-prefill rather than at load.

### Quality spot-check (first 16 decoded token ids)

| ptok | FP8=0 first 16 ids | FP8=1 first 16 ids |
| ---: | :--- | :--- |
|   512 | `561, 29144, 6165, 27050, 279, 24460, 3377, 303, 3150, 854, 13, 2838, 5947, 264, 15352, 6157` | **identical** |
|  4096 | `54134, 10782, 264, 491, 9140, 314, 5365, 7559, 64, 7397, 303, 279, 15979, 20915, 13, 561` | **identical** |
| 16384 | `561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 3043, 279, 14367, 5883, 13, 54134, 10782` | **identical** |

Bit-identical tokens across the three workable prompt lengths confirm the FP8
E4M3 path is numerically safe for production decode at BF16 parity.

### Why ≥65536p OOMs in *both* arms

The 131072p logs surface the root cause:

```
Caused by:
    0: paged prefill forward pass failed
    1: gated deltanet layer 0 (linear attention, paged)
    2: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The 65536p failure is the same OOM on the same GDN layer-0 allocation (the
error chain is elided by a log overwrite in our capture script but the raw
traceback matches). The OOM happens **inside GDN's linear-attention prefill
buffers, before the paged GQA KV cache is even touched**. FP8 quantizes only
the GQA KV cache; it leaves GDN's recurrent state and chunk buffers
BF16-sized. On the current decode path, GDN prefill is the memory-dominant
term at long prompts, so halving the GQA KV cache does not shift the OOM
ceiling upward.

This matches the architectural note in ARCHITECTURE.md that 24 of 32 layers
are GDN with per-layer chunk/recurrent state, vs 8 GQA layers with
kv-head count = 4.

### Decision rubric (as specified in the verification task)

| Criterion | Threshold | Observed | Pass? |
| --- | --- | --- | :---: |
| Decode tok/s within ±3% at 512p / 4096p | ±3% either direction | 512p −3.5% (warmup), 4096p −2.9%, 16384p −0.5% | partial |
| Peak VRAM at 65536p FP8 ≤60% of BF16 | ≤60% | Not measurable — both OOM in GDN prefill before KV cache is sized | no |
| Output tokens coherent / match BF16 | bit-identical | identical first-16 ids at 512 / 4096 / 16384 | yes |
| 131072p unlocks FP8 while OOMing BF16 | FP8 succeeds, BF16 fails | both OOM in GDN layer 0 | no |

Two of four criteria fail and one passes only marginally, so the rubric
specifies **keep opt-in**. We keep `kv_cache_fp8 = false` as the default and
leave the flag documented in `kiln.example.toml` for users who explicitly
need BF16 parity with halved GQA KV storage (e.g. doubling a workload that
is KV-bound on batch count rather than single-request context).

### What this means for the Phase 6 frontier

FP8 KV cache is **not** the lever that unlocks 65536p+ context on this
pipeline, because GDN prefill memory is the binding constraint. Follow-on
work that could change that conclusion:

1. **GDN prefill memory reduction.** Audit the `kiln/gdn/in_proj`,
   `chunk_prep`, and `recurrent` state buffers in
   `crates/kiln-model/src/forward.rs` for chunked allocation or
   activation-checkpointing-style recomputation. This is the only path that
   actually raises the context ceiling on single-request A6000 decode.
2. **FP8 KV path re-verification if GDN is bounded.** If (1) lands and 65536p
   becomes workable in BF16, re-run this matrix — at that point the 60% KV
   memory headroom from FP8 could genuinely extend context to 131072p.
3. **Multi-request / batched decode.** FP8 KV still has real value when KV
   memory scales by concurrent sequences rather than by single-sequence
   prompt length. That regime wasn't exercised by this matrix (which ran
   single-request) and is not a reason to change the default until it is
   measured in a multi-tenant scheduling context.

Until one of (1)–(3) produces evidence that FP8 flips the context ceiling
or improves steady-state decode at real batch fan-out, the default stays
BF16 and the flag stays opt-in.




**$0 preflight follow-up to the post-#210 profile** (this file's first section).
That section recommended a `ucopy_bf16` source-site audit before any kernel
work; this is the audit. It is a **null result**: every remaining
`ucopy_bf16`-emitting site in the decode hot path is either already addressed
by a prior PR, sits below the 1.05× decode-floor on its own (even at infinite
local speedup), or is required by a downstream kernel ABI that cannot be
relaxed without redesigning the kernel itself. **Do not queue a `ucopy_bf16`
fusion task.** Phase 6 should move to KV cache FP8 next; the MLP gate/up
Marlin-merge speculation needs a separate activation-load profile before it
can be sized.

### Inputs

- Source: `crates/kiln-model/src/forward.rs` (read in full at this audit's
  branch point, c480886) plus `crates/kiln-model/src/marlin_proj.rs`,
  `crates/kiln-model/src/lora_loader.rs`.
- Wall-clock attribution: `profiling-artifacts/post210_nvtx.csv` and
  `profiling-artifacts/post210_kern.csv` (the post-#210 capture). Note: the
  post-#210 capture intentionally omitted `--cuda-graph-trace=node` (it
  corrupted event ordering), so per-region kernel attribution is not
  available — region totals and kernel totals are correlated below by
  reading the source.
- Prior PR scope: `git log --all --oneline | grep -iE
  "ucopy|transpose|cast_bf16|in_proj|qkv|recurrent|gates|gated|qk_norm"`.

### Per-site enumeration

For each `ucopy_bf16`-emitting site in the decode hot path: NVTX region,
prior-PR coverage, why it remains, and the per-site math-ceiling at a
generous 1.5× local speedup (`p · (1 − 1/1.5)`).

| # | site (file:line)                                     | NVTX region              | region % | prior PR coverage                       | why it remains                                                | local 1.5× ceiling |
| - | ---------------------------------------------------- | ------------------------ | -------: | --------------------------------------- | ------------------------------------------------------------- | -----------------: |
| 1 | `forward.rs:1462` `mixed_qkv.transpose(1,2).contiguous()` | `:kiln/gdn/conv`         |      1.8 | none                                    | required input layout for `causal_conv1d_update` kernel ABI   |              0.006 |
| 2 | `forward.rs:1521-1535` `.unsqueeze().expand().contiguous()` (q,k) and `q.contiguous()`/`k.contiguous()` fallback | `:kiln/gdn/head_expand`  |      3.4 | none                                    | downstream matmul cannot operate on broadcast-strided tensor  |              0.011 |
| 3 | `forward.rs:1655-1659` 5× `.transpose(1,2)` (q,k,v,beta,g) | `:kiln/gdn/recur_prep`   |      0.8 | none                                    | lazy strides; downstream `gdn_chunkwise_recurrence` consumer materializes | 0.003 |
| 4 | `forward.rs` chunkwise recurrence interior `.contiguous()` | `:kiln/attn/gdn/recurrent` |    2.0 | PR #80 (vendored chunkwise GDN), PR #74 (matmul readout), PR #75 (analytical recurrence) | internal compute layout for the vendored recurrence kernel    |              0.007 |
| 5 | `forward.rs:1580-1581` `(q*scale).to_dtype()`, `k.to_dtype()` | `:kiln/gdn/qk_norm`      |     14.7 | **PR #173** opt-in fused L2-QK norm; null median 1.0093× | already attempted; null. The fused kernel exists behind `KILN_ENABLE_FUSED_L2_QK_NORM`. | (see PR #173)      |
| 6 | `forward.rs:1693` `attn_out.reshape().to_dtype()`    | `:kiln/gdn/gated_norm`   |     17.6 | **PR #141** closed null                 | already attempted; null. Future re-visit needs new memory-reducing approach | (see PR #141)      |
| 7 | `forward.rs:1844-1846,1862-1873,1895-1898` Q/K/V/output `transpose().contiguous()` | `:kiln/attn/full/*` slow path | ≤3.8 | **PR #100** fused paged-decode (skipped on hot path) | only fires on FA2 fallback (rare); paged-decode kernel never materializes these | ~0 on hot path     |
| 8 | `marlin_proj.rs:296,315` `x.contiguous()`            | callsite-attributed to `:kiln/gdn/in_proj`, MLP `gate/up/down`, full-attn `qkv` | (covered by callsites) | required by `marlin_w4a16_gemm` kernel ABI | kernel reads contiguous strides; cannot be relaxed without redesigning Marlin | sub-floor |
| 9 | GDN linear projections (in_proj_qkv/z/a/b, out_proj) | `:kiln/gdn/in_proj`, `:kiln/gdn/out_proj` |     10.5 (combined) | **PR #130** pre-transposed `*_t` cache  | already eliminated at load time (`weights.in_proj_*_t`)        | already addressed  |
| 10 | MLP / full-attn projections (gate/up/down, q/k/v/o) | `:kiln/mlp/*`, `:kiln/proj/*` |    22.6 (combined) | **PR #128** pre-transposed `*_t` cache  | already eliminated at load time                                | already addressed  |
| 11 | lm_head                                             | `:kiln/lm_head` 0.2%     |      0.2 | **PR #117** precompute `embed_tokens.t()` once at load | already eliminated                                            | already addressed  |
| 12 | `linear_with_lora` (non-`_t` variant)               | n/a (tests only)         |        0 | PR #128/#130 hot-path migration to `_t` | only used in `lora_loader.rs` tests; no decode callsite        | already addressed  |

(Sites 9–12 confirm the historical wins from PRs #117/#128/#130 are still in
place on current `main`; the residual 8.6% kernel-level `ucopy_bf16` is the
sum of the un-cached layout transforms in sites 1–7.)

### Combined math-ceiling

Sum of un-addressed, un-attempted, in-hot-path sites (1 + 2 + 3 + 4):
`1.8% + 3.4% + 0.8% + 2.0% = 8.0%` of decode wall-clock time. At a generous
1.5× local speedup applied to **all four sites simultaneously**:

```
combined contribution = 0.080 · (1 − 1/1.5) = 0.027
```

Even **fully eliminating every un-addressed site at infinite local speedup**:

```
combined contribution = 0.080 · (1 − 1/∞) = 0.080
```

Both are below the project's 0.05 (i.e. ≥1.05×) decode-speedup queue floor.
The "infinite" case is also unrealistic — `ucopy_bf16` is memory-bound, not
launch-bound, so the local speedup ceiling is set by HBM bandwidth, not by
launch dispatch. Under CUDA graph replay (production decode path), launch
amortization further compresses the achievable win toward the conservative
end of this range. The same lesson applies as in #141, #173, #176, and #164.

Sites 5 and 6 are also above the floor on paper (14.7% + 17.6% = 32.3%), but
they have already been attempted (#173 opt-in null median 1.0093×; #141
closed null). They are listed for completeness, not as queueable targets.

### Why "consolidating call sites" does not help

The `ucopy_bf16` kernel total (8.6% / 4164 invocations) looks consolidatable
but in fact splits cleanly by tensor shape and dtype across the seven hot-path
sites above. The four un-addressed sites (1, 2, 3, 4) operate on different
ranks ([B,T,qkv_dim] vs [B,T,nv,dk] vs [B,nv,T,*]) and cannot share a fused
kernel: each one is a layout transform that exists because the *next* kernel
in the chain (`causal_conv1d_update`, the GQA matmul, the chunkwise GDN
recurrence) requires that specific stride pattern. Removing the layout
transform requires changing the downstream kernel's input contract — which
is exactly what PR #128 and PR #130 already did for the projection weights
(by caching the pre-transposed weight tensor at load time so the per-step
transpose disappears).

There is no equivalent "cache it at load time" trick available for sites 1–4
because they operate on per-step *activations*, not per-load *weights*. An
`ucopy_bf16`-eliminating fix at site 2 (head_expand) would need to be a
custom CUDA matmul kernel that accepts broadcast-strided K/V inputs — which
is the same class of work as #100 (fused paged decode FA2 hot path), and at
3.4% region-share it does not clear the floor.

### Recommendation

This audit closes the post-#210 PROFILING.md follow-up #1
(`ucopy_bf16` audit). Proceed with the post-#210 follow-up #2 instead:

- **KV cache FP8 (`KILN_KV_CACHE_FP8=1`)** is the qualified next slice. It
  is already wired (`kiln-server/src/config.rs`); needs a correctness +
  quality benchmark, not a fusion. Distinct win category from the
  decode-kernel work, not gated by the math-ceiling floor (context-length
  is a product feature, not a decode-speed metric).

The post-#210 follow-up #3 (MLP gate/up Marlin merge) remains marked "needs
further math" — needs a profile that separates activation-load time from
GEMM time on the MLP trio before pod $ can be queued.

### Marlin BF16 residency cleanup — not a `ucopy_bf16` lever

PR #206 already dropped the redundant BF16 `*_proj_t` tensors after Marlin
packing, reclaiming the +1.3 GB VRAM that the post-#166 baseline reported.
There is no further BF16-residency-related `ucopy_bf16` work pending — the
remaining decode-path BF16 weight tensors (GDN `*_t` projections, attention
`*_t` projections kept around for non-Marlin builds, and embeddings) are
either consumed by `broadcast_matmul` directly (no `ucopy_bf16` emitted) or
gated on `*_marlin.is_some()` to skip the BF16 path entirely.

### Artifacts

This is a doc-only PR. No bench was run; no pod was acquired. Total compute
cost: $0. The audit relied entirely on the existing post-#210 profile
(`profiling-artifacts/post210_*.csv`) and a source read of
`crates/kiln-model/src/{forward,marlin_proj,lora_loader}.rs` at commit
c480886.


## Phase 6 — Post-PR #210 decode profile (2026-04-20)

Fresh nsys capture on current `main` after the post-#166 cluster of Marlin /
load-time / VRAM / bench-infra changes. Supersedes the post-#166 breakdown as
the canonical "next target" reference. This is the first profile since #173
(opt-in fused L2-QK norm, null median) and #176 (closed-null big-fusion) were
resolved, so it is also the first profile that lets us re-check whether any
GDN gate-path fusion has meaningfully shifted proportions at the
graph-replay layer.

**Preflight** (kernel-vendor-precondition-check, before pod $):

- Recent PRs reviewed:
  - #204 bench-only pool verification (no code delta vs #166 NVTX snapshot)
  - #206 drop BF16 MLP weights when Marlin is resident (VRAM cleanup; no
    decode-path compute delta)
  - #210 parallelize Marlin pack loop (load-time only; no forward-path
    compute delta)
  - #211–#215 desktop / CI / docs (no runtime impact)
- Net: no forward-path code deltas since the post-#166 NVTX snapshot. The
  NVTX composition is expected to be within run-to-run variance of post-#166,
  and this capture confirms that — top-3 region ordering is unchanged
  (gates → gated_norm → qk_norm) and their summed share moves from 50.4% →
  50.8% (within noise). **The profile's job is not to find a shifted
  hotspot — it is to rule out new candidates before we consider re-opening
  any closed-null kernel work.**
- Not-a-target list (verified against `gh pr list --repo ericflo/kiln
  --state all`): qk_norm standalone (#173 null 1.0093×), gated_norm
  standalone (#141 null), big-fusion qk_norm+gated_norm+recurrent (#176
  null, $14.99 burn), FlashInfer paged GQA decode (#163 math-ceiling
  ≤1.005×), gate-path fusion across Step 7 recurrence (#164 architecturally
  infeasible — two halves already shipped/closed independently).

**Hardware:** Pool-acquired RunPod NVIDIA RTX A6000 on-demand (pod
`t0vmof6qkwostu`, $0.49/hr), Driver 580.95.05, CUDA 12.8,
`ghcr.io/ericflo/kiln-runpod:latest`.

**Build:** release, `--features cuda,nvtx`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (Backblaze B2 bucket).

**Bench & profile:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
clean bench runs (no nsys) followed by a single nsys capture run. nsys
2024.6.2 with `--trace=cuda,nvtx` (NO `--cuda-graph-trace=node` — that flag
previously corrupted event ordering and produced an
`EventCollection::CheckOrder` finalization failure; CUDA-graph attribution
is intentionally sacrificed so we get a valid trace at all).

### Decode bench — 512-prompt × 128-decode (Arm B, 3 clean runs)

| run    | decode tok/s | mean ITL ms | prefill (ms) | model load (s) |
| ------ | -----------: | ----------: | -----------: | -------------: |
| 1      |        52.29 |       19.13 |        324.3 |          23.76 |
| 2      |        50.92 |       19.65 |        329.6 |          22.75 |
| 3      |        52.88 |       18.90 |        327.7 |          22.90 |
| median |    **52.29** |   **19.13** |    **327.7** |      **22.90** |

Resident VRAM: 17528 MB (post-#206 drop; matches the expected −1.3 GB vs
pre-#206). Marlin pack latency at load: ~22.9 s (post-#210 parallelized;
was ~58 s on the pre-#210 baseline — confirms the load-time win landed).

### Δ vs post-#166 direct-RunPod baseline (median)

| metric            | post-#166 direct | 2026-04-20 post-#210 | Δ         |
| ----------------- | ---------------: | -------------------: | --------: |
| decode tok/s      |            49.76 |                52.29 |    +5.1 % |
| mean ITL ms       |            20.10 |                19.13 |    −4.8 % |
| model load s      |             ~50+ |                22.90 | ~ −54 %   |
| resident VRAM MB  |           ~18800 |                17528 |    −6.8 % |

The decode tok/s uplift is within run-to-run variance (post-#204 pool re-run
also saw a median of 51.37 tok/s on identical code — variance band is ~±3%
across sessions). Model-load and VRAM deltas are real and attributable to
#210 and #206 respectively. **Decode hotspots are materially unchanged.**

### NVTX region top-10 (decode, wall-clock %, post-#210 nsys capture)

Source: `profiling-artifacts/post210_nvtx.csv` (130 decode iterations
captured, 3120 GDN instances = 130 × 24 GDN layers, 4160 MLP instances =
130 × 32 MLP layers, 1041 full-attn `qkv` instances ≈ 130 × 8 GQA layers +
1 prefill).

| %        | region                          | med (ns) | notes                      |
| -------: | ------------------------------- | -------: | -------------------------- |
| **18.5** | `:kiln/gdn/gates`               |  198,347 | GDN Step 6 (fused in #158) |
| **17.6** | `:kiln/gdn/gated_norm`          |  188,730 | #141 closed null           |
| **14.7** | `:kiln/gdn/qk_norm`             |  157,995 | #173 opt-in only (null)    |
|      7.7 | `:kiln/gdn/in_proj`             |   81,101 | Marlin GEMM                |
|      6.6 | `:kiln/mlp/gate`                |   52,591 | Marlin GEMM                |
|      6.2 | `:kiln/mlp/up`                  |   50,035 | Marlin GEMM                |
|      6.0 | `:kiln/mlp/down`                |   47,815 | Marlin GEMM                |
|      3.4 | `:kiln/gdn/head_expand`         |   36,927 | reshape/broadcast          |
|      3.4 | `:kiln/residual`                |   13,636 | 8321 instances             |
|      3.0 | `:kiln/proj/qkv`                |   97,379 | full-attn Marlin (8 lyrs)  |

Sub-10 regions worth noting: `gdn/out_proj` 2.8%, `norm/pre_mlp` 2.2%,
`attn/gdn/recurrent` 2.0%, `norm/pre_attn` 1.9%, `gdn/conv` 1.8%,
`proj/o` 0.8%, `gdn/recur_prep` 0.8%.

**Aggregates:**
- **GDN total:** ~68.5% (`gates` 18.5 + `gated_norm` 17.6 + `qk_norm` 14.7
  + `in_proj` 7.7 + `head_expand` 3.4 + `out_proj` 2.8 + `recurrent` 2.0 +
  `conv` 1.8 + `recur_prep` 0.8)
- **MLP trio:** 18.8% (`gate` 6.6 + `up` 6.2 + `down` 6.0)
- **Full-attn projections:** ~3.8% (`qkv` 3.0 + `o` 0.8) — **still below
  the 10% FlashInfer threshold** (this re-confirms the #163 preflight).

### CUDA kernel top-10 (`--trace=cuda,nvtx`, post-#210)

Source: `profiling-artifacts/post210_kern.csv` (nsys `cuda_gpu_kern_sum`).

| %        | kernel                                                           | med (ns) | notes                                |
| -------: | ---------------------------------------------------------------- | -------: | ------------------------------------ |
| **15.7** | `Marlin<256,1,8,8,4,8>`                                          |   20,288 | 13527 invocations                    |
| **12.6** | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`   | 1,715,259 | 130 invocations = **lm_head** |
|     10.5 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`     |   60,000 | 3121 invocations                     |
|      9.8 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`    |   36,208 | 6244 invocations                     |
|  **8.6** | `ucopy_bf16`                                                     |   32,897 | 4164 invocations — cross-cutting     |
|      5.7 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | 32,480 | 3121 invocations                 |
|      3.4 | `bmul_f32`                                                       |    2,784 | 24979 invocations (elementwise zoo)  |
|      3.2 | `fused_rmsnorm_kernel`                                           |    6,176 | 10536 invocations (PR #133 kernel)   |
|      2.7 | `recurrent_gdn_fwd_kernel<128>`                                  |   15,360 | 3122 invocations (PR #80 kernel)     |
|      2.5 | `cast_f32_bf16`                                                  |    1,312 | 20814 invocations                    |

Sub-top-10 notable: `ucopy_f32` 2.1%, `cast_bf16_f32` 1.8%, `affine_f32`
1.6%, `fast_sum_f32` 1.3%, `bmul_bf16` 0.8%, `badd_bf16` 0.8%. Together the
elementwise zoo (casts + ucopy + affine + sum + mul) accounts for ~20% of
kernel time, scattered across the GDN + norm + residual paths.

### Graph-replay-aware analysis (per region)

The dispatch-amortization lesson from #141, #173, #176, and #164 applies
with full force: under CUDA graph replay the per-iteration kernel-launch
cost goes to near-zero, so a fusion that only collapses launches shows up
as a null delta. **For a fusion to clear the 1.05× decode floor, it must
reduce real compute or memory traffic, not just dispatch cost.** The
math-ceiling check below uses the decode-level Amdahl bound: a fusion
eliminating fraction *p* of decode time at speedup *s* yields a total
speedup of `1 / (1 - p·(1 - 1/s))`, and the win-contribution is
`p · (1 - 1/s)`. We require that contribution ≥ 0.05 (i.e. ≥5% decode
speedup) to even queue a pod-$ attempt.

| region                           | % | already attempted?                   | realistic compute-reducing fusion? | win contribution if 1.10× | queue?     |
| -------------------------------- | -: | ------------------------------------ | :--------------------------------: | ------------------------: | ---------- |
| `:kiln/gdn/gates`                | 18.5 | fused in #158 (merged)             | already fused                      |                         — | **no**     |
| `:kiln/gdn/gated_norm`           | 17.6 | #141 closed null                   | would need new memory-reducing approach | 0.016                | **no** (re-visit needs new idea, not new attempt) |
| `:kiln/gdn/qk_norm`              | 14.7 | #173 opt-in, null median           | already tried and null             |                     0.013 | **no** |
| `:kiln/gdn/in_proj`              |  7.7 | Marlin GEMM (already quantized)    | GEMM shape change, not fusion      |                     0.007 | **no**     |
| MLP trio (gate+up+down)          | 18.8 | none                               | partial fusion (swiglu gate/up merge into a single Marlin call) plausible | 0.017 | **no** (sub-floor) |
| `:kiln/gdn/in_proj` + `out_proj` | 10.5 | none                               | share layout work, potential combined load  | 0.009        | **no** (sub-floor) |
| `ucopy_bf16` (kernel-level)      |  8.6 kernel-% | none                    | many sites (residual, reshape, cast-adjacent); consolidation could pay if source sites are identified | depends on which sites | **investigate** |
| elementwise zoo (aggregate)      | ~20 kernel-% | none                     | cross-cutting; no single natural fusion site | depends on grouping | **investigate** |
| `:kiln/residual`                 |  3.4 | none                               | potential in-place ADD into RMSNorm epilogue | 0.003            | **no**     |
| full-attn `:kiln/proj/qkv`       |  3.0 | FlashInfer #163 ruled null         | already quantized (Marlin)         |                     0.003 | **no**     |

None of the "queue?" column says **yes**. Every standalone region either
already has a shipped fusion, has a closed-null attempt on record, or sits
below the 1.10× × region-% floor of 0.05 decode speedup.

### Next target — recommendation

**Do not queue another standalone decode-kernel fusion.** The top three
GDN regions (gates / gated_norm / qk_norm, summed 50.8%) have all been
attacked and each either shipped (#158) or closed null (#141, #173, #176).
The MLP trio is the only sizable region that has not been touched, but at
18.8% it requires a 1.362× per-region speedup to clear the 5% decode floor,
and that's the theoretical ceiling — real speedups under graph replay fall
well below the nominal kernel-level number.

Three qualified candidates for the next Phase 6 slice, in priority order:

1. **`ucopy_bf16` source-site audit (investigate-only, $0 preflight
   follow-up).** `ucopy_bf16` is 8.6% of kernel time spread across 4164
   invocations per decode step. If the source sites (candle's transpose /
   contiguous / cast-adjacent copies) can be identified and consolidated,
   the math-ceiling case is: 8.6% × (1 − 1/1.5) ≈ 0.029, still sub-floor
   *if we only save kernel time*, but attacking it means reducing memory
   traffic which *does* compound under graph replay. This needs an
   instrumentation pass first — add `NVTX_RANGE!` around each call site —
   before any kernel work.

2. **KV cache FP8 (`KILN_KV_CACHE_FP8=1`).** Non-kernel axis; not about
   decode tok/s, about doubling effective context. Already wired
   (`KILN_KV_CACHE_FP8` in `kiln-server/src/config.rs`). Needs a
   correctness + quality benchmark, not a fusion. Distinct win category
   from the decode-kernel work; would not be starved by the math-ceiling
   floor because context-length is a product feature, not a decode-speed
   metric.

3. **MLP gate/up Marlin merge (speculative).** `:kiln/mlp/gate` (6.6%) and
   `:kiln/mlp/up` (6.2%) run on the same input activation and are
   immediately fused by a SwiGLU elementwise op. A single Marlin call
   producing both outputs in parallel should halve the activation load
   cost. Math-ceiling: 12.8% × (1 − 1/1.5) = 0.043 — still sub-floor, so
   **not a queued task yet**. Flag it as "needs further math" — specifically
   needs a profile that separates activation-load time from GEMM time on
   the MLP trio. Do that profiling before committing pod $.

### Do NOT target (restated)

- **`:kiln/gdn/qk_norm`** standalone — PR #173 shipped opt-in, null median
  (1.0093× point estimate, variance-reducing only). Do not re-queue without
  new evidence of a memory-reducing approach that wasn't tried in #173.
- **`:kiln/gdn/gated_norm`** standalone — PR #141 closed null. Do not
  re-queue.
- **Big-fusion (`qk_norm` + `gated_norm` + `recurrent`)** — PR #176 closed
  null, $14.99 burn. Step 7 (recurrent) is architecturally separated from
  Steps 6 and 8 and cannot be single-kernel fused. Do not re-queue.
- **FlashInfer paged GQA decode** — PR #163 preflight: full-attn total is
  3.8% of decode this capture (was 3.5% post-#166), math ceiling ≤1.038×
  even at infinite speedup on that region. Not viable on Qwen3.5-4B's
  24 GDN + 8 GQA architecture.
- **Gate-path fusion across Step 7 recurrence** — PR #164 preflight
  concluded architecturally infeasible; the two halves (Step 6 gates, Step
  8 gated_norm) are separated by the recurrent scan and cannot share a
  kernel. Both halves are already addressed independently (#158 merged,
  #141 closed null). Do not re-queue as a combined task.

### Artifacts

- `profiling-artifacts/post210_nvtx.csv` — full NVTX region table
  (24 rows).
- `profiling-artifacts/post210_kern.csv` — full CUDA kernel table
  (52 rows).
- Raw `.nsys-rep` (112 MB) retained on pod only (too large to commit).


## Re-profile 2026-04-20 (pool-verification baseline)

Bench-only re-run on a fresh pod acquired through the new Cloud Eric kiln pod
pool (`ce kiln-pod-acquire` / `ce kiln-pod-release`). Purpose:

1. Verify the pool path produces results materially identical to the
   direct-RunPod path (post-PR #166 baseline).
2. Validate the new long-running bench supervision pattern in
   `skills/kiln/resources/runpod-workflow.md` (sentinel-file polling via
   `runpod_api.py wait-file`, no ad-hoc `until ssh ... kill -0` loops). The
   prior attempt at this re-profile (task `c8a18546…`) burned $13.76 in a
   silent SSH-polling deadlock — see agent note `kiln-ssh-polling-deadlock`.

No code changes between this run and the post-PR #166 baseline, so the NVTX
top-10 and CUDA-kernel top-10 from the
[Post-PR #166 section](#phase-6--post-pr-166-decode-profile-2026-04-18) below
remain canonical. This section adds only fresh bench numbers from the
pool-verification run.

**Hardware:** Pool-acquired RunPod NVIDIA RTX A6000 on-demand (pod
`t0vmof6qkwostu`, $0.49/hr), Driver 580.95.05, CUDA 12.8,
`ghcr.io/ericflo/kiln-runpod:latest`.

**Build:** release, `--features cuda,nvtx`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (Backblaze B2 bucket).
Cold first-clone build on the freshly-spawned pool pod (sccache hit rate
low on first build; not a hot cache).

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
runs in a single nohup'd shell script with sentinel-file completion
signaling. NVTX-feature build but no nsys capture this run (pool-verification
scope only).

### Decode tok/s — 512-prompt × 128-decode (Arm B, 3 runs, pool path)

| run    | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ------ | -----------: | ----------: | ---------: | ---------: |
| 1      |        49.15 |       20.35 |      20.12 |      24.19 |
| 2      |        51.37 |       19.47 |      19.38 |      22.39 |
| 3      |        52.39 |       19.09 |      18.83 |      22.40 |
| mean   |    **50.97** |   **19.64** |  **19.44** |  **23.00** |
| median |    **51.37** |   **19.47** |  **19.38** |  **22.40** |

Run 1 here is the slowest (cold prefill: 7540 ms, 67 tok/s for the first
single-shot prefill — pure CUDA-graph capture + JIT + page-in warmup; runs
2–3 settle to 327–337 ms / ~1500 tok/s). The decode tok/s ordering is
inverted vs the post-#166 baseline where run 1 was fastest; this is exactly
the run-to-run variance the `kiln-bench-prefill-warmup-required` note
warns about. Decode tok/s + ITL averages over many tokens so all three runs
are individually trustworthy, but single-shot TTFT is not.

### Δ vs post-#166 direct-RunPod baseline (median)

| metric            | post-#166 direct | 2026-04-20 pool | Δ        |
| ----------------- | ---------------: | --------------: | -------: |
| decode tok/s      |            49.76 |           51.37 |   +3.2 % |
| mean ITL ms       |            20.10 |           19.47 |   −3.1 % |
| p50 ITL ms        |            19.91 |           19.38 |   −2.7 % |
| p99 ITL ms        |            25.46 |           22.40 |  −12.0 % |

Within run-to-run variance — the post-#166 mean was 51.15 tok/s, this run's
mean is 50.97 tok/s (a 0.4% delta). The pool path produces equivalent
decode performance to the direct-launch path. **No regression.** Pool
verification: passed.

### Pool-path operational metrics

- Pool acquire → SSH ready: ~6 min (fresh spawn, not a warm reuse — pool
  was empty for A6000 at acquire time).
- Cold build (`cargo build --release --features cuda,nvtx --bin kiln-bench`):
  ~22 min (sccache cold, full ~330-crate build).
- Model download (Qwen/Qwen3.5-4B, ~8.8 GB) ran in parallel with build,
  finished before build completed.
- Bench (3× back-to-back, sentinel-driven): ~7 min.
- Total wall-clock from acquire to PR-ready: ~36 min.
- Bench supervision pattern (sentinel + `runpod_api.py wait-file`) worked
  cleanly — no SSH wedging, no silent polling stalls, no $13+ surprise.

Subsequent re-profiles on a warm pool pod (or a hibernated-then-resumed
pod) should hit ~10 min total wall-clock by skipping the cold build.


## Phase 6 — Post-PR #166 decode profile (2026-04-18)

Fresh nsys capture on current `main` (HEAD `c2579a1`, PR #166 vendored
`causal_conv1d_update` into `kiln-conv1d-kernel` and wired
`BackendRuntime::causal_conv1d_update` in `kiln-model/src/backend/cuda.rs`).
This is now the source-of-truth decode breakdown for the next optimization
task — it supersedes the "Post-PR #158 Decode Profile" section below for
planning purposes. Methodology matches the post-#158 profile (Arm B only,
W4A16 + CUDA graphs production path).

**Hardware:** RunPod NVIDIA RTX A6000 on-demand ($0.49/hr), Driver
580.95.05, CUDA 12.8, `ghcr.io/ericflo/kiln-runpod:latest` (CUDA 12.4
toolchain, nsys 2024.5.1 — the baked 2023.4.4 still has the
`EventCollection::CheckOrder` finalization bug noted in prior sections).

**Build:** release, `--features cuda`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (backblaze B2 bucket).

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
uncaptured runs for tok/s + ITL, 1 separate `nsys profile --delay=110
--duration=30 --capture-range=nvtx -p 'kiln/*' ...` capture for NVTX +
CUDA kernel stats. Single-token paged decode path
(`model_forward_paged`) exactly as served by the HTTP scheduler.

### Decode tok/s — 512-prompt × 128-decode (Arm B, 3 runs)

| run    | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ------ | -----------: | ----------: | ---------: | ---------: |
| 1      |        54.24 |       18.44 |      18.43 |      20.90 |
| 2      |        49.76 |       20.10 |      19.91 |      25.46 |
| 3      |        49.45 |       20.22 |      20.10 |      25.59 |
| mean   |    **51.15** |   **19.59** |  **19.48** |  **23.98** |
| median |    **49.76** |   **20.10** |  **19.91** |  **25.46** |

Run 1 is again the fastest and tightest — consistent with the pattern seen
in the earlier #166 A/B bench where the first POST run was slower; here
sccache / graph-cache warmth appears to favor run 1 instead. Runs 2–3
converge to ~49.5 tok/s.

### Top-10 NVTX regions (Arm B, post-#166)

| rank | %     | region                        |
| ---: | ----: | ----------------------------- |
|    1 | 18.0% | `:kiln/gdn/gates`             |
|    2 | 17.5% | `:kiln/gdn/gated_norm`        |
|    3 | 14.9% | `:kiln/gdn/qk_norm`           |
|    4 |  7.4% | `:kiln/gdn/in_proj`           |
|    5 |  6.3% | `:kiln/mlp/gate`              |
|    6 |  6.2% | `:kiln/mlp/up`                |
|    7 |  5.9% | `:kiln/mlp/down`              |
|    8 |  3.8% | `:kiln/gdn/head_expand`       |
|    9 |  3.4% | `:kiln/residual`              |
|   10 |  3.0% | `:kiln/proj/qkv`              |
|   +  |  2.9% | `:kiln/gdn/out_proj`          |
|   +  |  2.1% | `:kiln/norm/pre_mlp`          |
|   +  |  1.9% | `:kiln/norm/pre_attn`         |
|   +  |  1.8% | `:kiln/attn/gdn/recurrent`    |
|   +  |  1.8% | `:kiln/gdn/conv`              |

GDN (in_proj + gates + gated_norm + qk_norm + conv + head_expand + out_proj
+ recurrent) now owns **~69%** of decode wall-clock (down from ~73% post-#158
as conv collapsed). MLP trio is **18.4%**. Norms/residual ≈ 7.4%. Full-attn
projections (`:kiln/proj/qkv` + `:kiln/proj/o`) are ~3.5% combined — still
below the 10% threshold that would justify FlashInfer paged GQA decode
(see `flashinfer-decode-preflight-kiln-2026-04-18`).

### Top-10 CUDA kernels (Arm B, post-#166)

| rank | %     | kernel                                                                     |
| ---: | ----: | -------------------------------------------------------------------------- |
|    1 | 14.6% | `Marlin<256,1,8,8,4,8>`                                                    |
|    2 | 11.8% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`             |
|    3 |  9.9% | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`               |
|    4 |  9.2% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`              |
|    5 |  8.3% | `ucopy_bf16`                                                               |
|    6 |  5.4% | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn`      |
|    7 |  4.0% | `bmul_f32`                                                                 |
|    8 |  2.9% | `fused_rmsnorm_kernel`                                                     |
|    9 |  2.4% | `recurrent_gdn_fwd_kernel<128>`                                            |
|   10 |  2.4% | `cast_f32_bf16`                                                            |
|   +  |  2.2% | `ucopy_f32`                                                                |
|   +  |  1.7% | `cast_bf16_f32`                                                            |
|   +  |  1.6% | `affine_f32`                                                               |
|   +  |  1.3% | `fast_sum_f32`                                                             |
|   +  |  0.3% | `kiln_conv1d_update_k4_kernel<true>` *(vendored in PR #166)*               |

The vendored `kiln_conv1d_update_k4_kernel<true>` now runs in **0.3%** of
decode time (30.3 ms / 17,560 invocations, avg 1.7 μs per call). The
Marlin + cutlass + ampere GEMM stack composition is essentially unchanged
vs post-#158; the displaced conv1d time (was ~10% as an
elementwise-kernel tail) has been absorbed proportionally across the
remaining GDN gate-path tensors — the top-50 kernel list still shows the
same elementwise zoo (`bmul_f32`, `cast_*`, `ucopy_*`, `affine_f32`,
`fast_sum_f32`, `bdiv_f32`, `badd_f32`) at similar totals.

### Δ vs post-#158 (NVTX)

| region                   | post-#158 | post-#166 | Δ pp     |
| ------------------------ | --------: | --------: | -------: |
| `:kiln/gdn/conv`         |   12.2 %  |    1.8 %  | **−10.4** |
| `:kiln/gdn/gates`        |   16.7 %  |   18.0 %  |    +1.3  |
| `:kiln/gdn/gated_norm`   |   15.8 %  |   17.5 %  |    +1.7  |
| `:kiln/gdn/qk_norm`      |   13.3 %  |   14.9 %  |    +1.6  |
| `:kiln/gdn/in_proj`      |    6.8 %  |    7.4 %  |    +0.6  |
| `:kiln/mlp/gate`         |    5.8 %  |    6.3 %  |    +0.5  |
| `:kiln/mlp/up`           |    5.6 %  |    6.2 %  |    +0.6  |
| `:kiln/mlp/down`         |    5.4 %  |    5.9 %  |    +0.5  |
| `:kiln/gdn/head_expand`  |    2.8 %  |    3.8 %  |    +1.0  |
| `:kiln/residual`         |    3.1 %  |    3.4 %  |    +0.3  |
| `:kiln/proj/qkv`         |    2.7 %  |    3.0 %  |    +0.3  |

`:kiln/gdn/conv` dropped from the #4 hottest region to the #15 — the sole
intended effect of PR #166. Every other region's share went up by
~0.3–1.7 pp as the freed ~10 pp redistributed proportionally; this is the
expected "proportional rebasing" pattern and is not a regression of those
regions in wall-clock terms. Decode tok/s (median 49.76 vs #158's 43.63 on
the same methodology) is **+14.1%** faster.

### Recommended next optimization target

`:kiln/gdn/gates` + `:kiln/gdn/gated_norm` + `:kiln/gdn/qk_norm` =
**50.4 %** of decode, which is above the 40 % threshold the Phase 6 brief
set as the trigger for the next vendor task. The natural next step is to
vendor **`chunk_gla_fwd`** from
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
— it fuses the gated-linear-attention chunked forward path (gates + qk_norm
+ gated_norm) that is currently three separate Candle-dispatched region
groups on our side. Per the `kernel-vendoring-call-pattern-regression` note,
the scope should be narrowed to the **decode-only (single-token)** call
pattern so the vendored kernel does not regress prefill / chunk paths — a
decode-specialized split of `chunk_gla_fwd` (or its underlying block-GLA
step kernel) is the concrete target.

> **Update 2026-04-19:** A standalone fused `qk_norm` kernel was attempted
> as a narrower scope (the 14.9 % `:kiln/gdn/qk_norm` slice in isolation)
> and produced a **null result** — see "Phase 6 — fused L2-norm Q+K
> decode kernel (NULL RESULT, 2026-04-19)" below. The kernel is correct
> but the dispatch overhead under `KILN_CUDA_GRAPHS=true` already
> amortizes most of the candle-launch cost at replay time, leaving only
> ~6 % of the qk_norm region to recover. **Do not propose another
> standalone `qk_norm` fusion** without first invalidating the
> dispatch-amortization argument. The `chunk_gla_fwd` direction (or a
> fused `gated_norm + qk_norm` kernel that runs as a single dispatch
> across both regions, claiming the combined **32.4 %**) is the only
> remaining sensible scope here.

Do **not** redirect to FlashInfer paged GQA decode: full-attn projections
on Qwen3.5-4B (24 GDN + 8 full-attn layers) are still only ~3.5 % of
decode, bounded at ≤1.035× overall speedup — well below the 1.15 %
abort threshold. The `flashinfer-decode-preflight-kiln-2026-04-18` note
remains the governing preflight for any future FlashInfer proposal.

### Preflight record

- HEAD verified `c2579a1` (`git log --oneline -1`).
- `crates/kiln-conv1d-kernel/` exists (Cargo.toml, build.rs, csrc/,
  src/lib.rs) — confirms PR #166 landed as advertised.
- `BackendRuntime::causal_conv1d_update` present in
  `crates/kiln-model/src/backend/cuda.rs` (line 20 declaration, lines
  185-205 dispatch).
- `gh pr list` showed no open PR touching GDN / conv paths at capture
  time; `git log --all --grep='chunk_gla'` returned no prior vendor work
  for the next recommended target.
- nsys 2024.5.1 used (baked 2023.4.4 avoided for the
  `EventCollection::CheckOrder` bug).
- `profiling-artifacts/post166_nvtx.csv` and
  `profiling-artifacts/post166_kern.csv` are the raw tables the above
  rankings were derived from (nsys stats `nvtxsum` / `cuda_gpu_kern_sum`
  reports).

### Pod / cost

- Pod: `povpyv0bkwqrte` (RunPod NVIDIA RTX A6000 on-demand, $0.49/hr).
- Total uptime at capture end: ~35 minutes (pull + clone + warmup + 3
  bench runs + 1 nsys capture + stats extraction).
- Estimated pod cost: **~$0.30** (well under the Phase 6 budget).


## Phase 6 — fused L2-norm Q+K decode kernel (NULL RESULT, 2026-04-19)

Vendored a single-launch CUDA kernel that fuses
`l2_normalize(Q) + scale(Q) + l2_normalize(K) + dtype-cast` (the
`:kiln/gdn/qk_norm` region) into `kiln-rmsnorm-kernel` as
`kiln_fused_l2_qk_norm`. The kernel is **correct** (3 parity tests at
decode/prefill/batch shapes pass; the full 128-test `kiln-model`
nextest suite passes) but **misses the task's 1.05× decode tok/s abort
floor**. It is shipped **opt-in only** via
`KILN_ENABLE_FUSED_L2_QK_NORM=1` — the candle reference path remains the
production default. See `crates/kiln-model/src/forward.rs` step-5 doc
comment for the in-tree pointer.

### Bench result (Arm B, RTX A6000, 3 paired runs, 2026-04-19)

`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, paged 512/128. Baseline disables
the kernel; fused enables it. Same binary; same warm-up.

| run        | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ---------- | -----------: | ----------: | ---------: | ---------: |
| baseline 1 |        51.67 |       19.35 |      19.12 |      23.85 |
| baseline 2 |        40.19 |       24.88 |      24.93 |      31.17 |
| baseline 3 |        50.81 |       19.68 |      19.63 |      24.78 |
| **median** |    **50.81** |   **19.68** |  **19.63** |  **24.78** |
| fused 1    |        50.64 |       19.75 |      19.71 |      20.25 |
| fused 2    |        51.28 |       19.50 |      19.47 |      19.87 |
| fused 3    |        51.63 |       19.37 |      19.39 |      22.53 |
| **median** |    **51.28** |   **19.50** |  **19.47** |  **20.25** |

| Metric                     | Baseline (median) | Fused (median) | Delta            |
| -------------------------- | ----------------: | -------------: | :--------------- |
| decode tok/s               |             50.81 |          51.28 | **+0.93 % (1.0093×)** |
| mean ITL                   |          19.68 ms |       19.50 ms | -0.18 ms (-0.92 %) |
| p50 ITL                    |          19.63 ms |       19.47 ms | -0.16 ms (-0.81 %) |
| p99 ITL                    |          24.78 ms |       20.25 ms | -4.54 ms (-18.3 %) |
| best-of-3 decode tok/s     |             51.67 |          51.63 | -0.07 % (0.9994×) |
| run-to-run range           |   11.48 (40-52)   |   1.00 (50-52) | -10.5 (much tighter) |

**Median speedup = 1.0093× — well below the task's 1.05× hard abort
floor.** Best-of-3 is actually a hair slower (0.9994×). The only clearly
positive signal is p99 ITL: tail latency tightens by ~4.5 ms (-18 %),
and run-to-run variance collapses from an 11 tok/s spread to a 1 tok/s
spread, suggesting the fused launch removes a small but real source of
dispatch jitter. That p99 win is not enough to clear the bar.

### Why the kernel won the math but lost the wallclock

Per the post-PR #166 profile above, `:kiln/gdn/qk_norm` was 14.9 % of
decode = ~2.93 ms of the median 19.68 ms ITL. Even fully eliminating
that NVTX region would yield at most:

    1 / (1 - 0.149) ≈ **1.175× decode speedup**

The actual 0.18 ms median ITL improvement implies the kernel only
shaved ~6 % of `:kiln/gdn/qk_norm`'s wallclock — far less than the
~85 %+ savings that would be needed to land at the 1.05× floor. The
qk_norm region is dominated not by the per-row arithmetic (rows = 16,
hidden = 128 — trivially memory-bound at decode shape) but by:

1. **Candle dispatch + graph-replay overhead** that survives the launch
   collapse. The CUDA-graph capture at decode already amortizes most of
   the ~11 underlying launches; collapsing them to 1 in the captured
   graph saves graph nodes but not host-side cost per replay.
2. **HBM round-trips** for the F32 intermediate buffers (`q_f32`,
   `k_f32`, `q_squared`, `k_squared`, `q_sum_keepdim`, etc.) that
   candle materializes. The fused kernel skips those, but the savings
   per row are tiny relative to the per-block scheduling cost.

The first point is the dominant explanation: under `KILN_CUDA_GRAPHS=true`
the qk_norm chain is captured once and replayed every decode step, so
the per-step "11 launches → 1" benefit at host code already mostly
collapses to a no-op at replay time. That is also why p99 (which
correlates with off-graph dispatch jitter) shows a clear win while mean
tok/s does not.

### Why ship the kernel as opt-in instead of deleting it

1. **It is correct** — parity tests pass at decode (16×128), prefill
   (1×512×128), and small-batch (4×8×128) shapes; the full nextest
   suite is green. No regression.
2. **Tail-latency improvement is real** — -18 % p99 ITL and a 10×
   tighter run-to-run distribution. Workloads sensitive to ITL tail
   variance (live token streaming, latency-SLO serving) may want to
   opt in.
3. **Future re-evaluation is cheap** — if a later optimization (e.g.
   an opt-out of CUDA graphs at qk_norm, or a wider qk_norm region
   that includes scale-fused dispatch) shifts the dispatch-overhead
   balance, the kernel is already wired and tested. No re-implementation
   needed.

The kernel sits behind `KILN_ENABLE_FUSED_L2_QK_NORM=1` (note: enable,
not disable). Default decode behavior is unchanged from PR #166.

### Re-visit triggers

A future PR can re-default this kernel to ON only if **all three**
hold on a fresh `nsys` profile of current `main` using the same
production path:

1. `:kiln/gdn/qk_norm` NVTX region ≥ **10 %** of decode wall-clock
   (it is 14.9 % today, but graph-replay amortization may have already
   reduced it).
2. The fused kernel produces a median **≥ 1.03×** decode tok/s
   speedup on a 3-run paired bench (currently 1.0093×).
3. There is a documented reason the dispatch-overhead amortization
   no longer dominates (e.g. CUDA graphs disabled at this region for
   a different reason, or a much wider fused region that includes
   the upstream `q = a.broadcast_mul(...)` step).

### Code shipped (branch `ce/phase6-qk-norm-fusion`)

- `crates/kiln-rmsnorm-kernel/csrc/fused_l2_qk_norm.{h,cu}` — single-block
  per-row kernel, two warp-shuffle reductions sharing scratch shmem,
  bf16 in/out, F32 reductions, hidden ≤ 8192 envelope.
- `crates/kiln-rmsnorm-kernel/build.rs` — adds the new `.cu` to the cc
  build.
- `crates/kiln-rmsnorm-kernel/src/lib.rs` — `kiln_fused_l2_qk_norm` FFI
  decl (line ~69), `supports_l2_qk_norm` capability gate, candle
  wrapper `fused_l2_qk_norm`, `reference_l2_qk_norm` parity oracle, 3
  `parity_l2_qk_norm_*` unit tests.
- `crates/kiln-model/src/forward.rs` — step-5 region updated with the
  opt-in dispatch (`KILN_ENABLE_FUSED_L2_QK_NORM`) and an in-tree
  comment pointing here.

### Preflight performed

- HEAD verified `c2579a1` (post-#166).
- `crates/kiln-rmsnorm-kernel/` already exists from the earlier RMSNorm
  vendor; this is an additive `.cu` + extern, not a new crate.
- `gh pr list` showed no overlapping open PR at preflight time.
- Nextest suite passed before the bench (128 tests, full kiln-model
  surface).

### Pod / cost

- Pod: `7s9x0e53pjoglc` (RunPod NVIDIA RTX A6000 on-demand, $0.79/hr —
  spot was unavailable; on-demand mandated by `runpod-always-on-demand`
  agent note).
- Total uptime at capture end: ~75 minutes (image pull + manual clone +
  `kiln-setup` + first build + 6 paired bench runs + result download).
- Estimated pod cost: **~$1.00**.

### Note on diagnostic capture

`nsys 2023.4.4` (baked in `ghcr.io/ericflo/kiln-runpod:latest`) hits
the `EventCollection::CheckOrder` finalization bug on this workload —
the same bug noted in the earlier post-#158 / post-#166 sections. Both
captures generated valid `.qdstrm` traces but `QdstrmImporter` exited
non-zero before producing `.nsys-rep`. Bench data alone is the abort
criterion per the task brief, so the missing NVTX share confirmation
does not change the result; this is recorded for future captures
(`nsys 2024.5.1` was used in the post-#166 profile and avoided the bug).


## Not next target — fused_recurrent GDN decode vendor (preflight, 2026-04-18)

Doc-only preflight-redirect. Third in the Phase 6 preflight-redundant
series after PR #163 (FlashInfer paged GQA decode) and PR #164 (GDN
gate-path fusion). **$0 pod spend.**

### Scope the task asked for

Vendor fla-org/flash-linear-attention's **`fused_recurrent_gated_delta_rule_fwd`**
(Triton → raw CUDA C) into a new `kiln-gdn-recurrent-kernel` crate, wrap
with a thin C-ABI + Rust FFI, and dispatch in kiln's GDN decode path.
Scope: bf16 activations, F32 state accumulators, causal, forward only,
per-token (`seq_len = 1`) decode, Qwen3.5-4B head dim. Expected speedup
1.3–2.0× decode tok/s; abort floor 1.10×.

### Why it is redundant

**PR #92 "Vendor fla fused_recurrent_gated_delta_rule_fwd for GDN decode"**
(merged 2026-04-17, commit `7aa62f1`) already shipped exactly this kernel
with exactly this scope.

| Component asked for                                      | Status in current HEAD (`c2579a1`) |
| -------------------------------------------------------- | ---------------------------------- |
| Fused per-token GDN recurrent CUDA kernel                | `crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.cu` (added by PR #92) |
| Thin C-ABI wrapper                                       | `crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.h` — `kiln_gdn_recurrent_forward(...)` extern "C" entry point |
| Rust FFI binding                                         | `crates/kiln-gdn-kernel/src/lib.rs` — `pub fn gdn_recurrent_forward(...)` at line 259, `kiln_gdn_recurrent_forward` FFI decl at line 80 |
| Dispatch in decode path                                  | `crates/kiln-model/src/forward.rs:1077-1079` — `backend.gdn_recurrent_step(...)` inside `kiln/attn/gdn/recurrent` NVTX range (the `seq_len == 1` fast-path branch of `gdn_chunkwise_recurrence`) |
| bf16 activations / F32 accumulators                      | Exactly this — see `recurrent_gdn_fwd.cu:49` (`__nv_bfloat16` inputs) and the F32 `s_local[MAX_DK]` state column |
| Causal only, forward only                                | Yes — single-token forward, no backward path |
| Qwen3.5-4B head dim                                      | `MAX_DK ∈ {128, 256}`; kiln configures `dk = dv = 128` |
| Per-thread scope (one block per `(batch, head)`)         | Exactly the launch geometry in `recurrent_gdn_fwd.cu:169/181` |
| `cargo build --release --features cuda` wired            | `crates/kiln-gdn-kernel/build.rs:44` compiles `recurrent_gdn_fwd.cu` via the `cc` crate |
| Parity test vs candle fallback                           | `test_gdn_kernel_matches_fallback` (oracle = `kiln-model::forward::compute_w_chunk_fallback`) |

The crate's top-level doc comment (`crates/kiln-gdn-kernel/src/lib.rs:33-35`)
states plainly:

> `gdn_recurrent_forward` — seq_len==1 decode fast path. Collapses the
> single-token GDN recurrence (decay, delta, state-update, output
> projection) into one block per (batch, head).

That is the function the task asked me to vendor. It already exists.

### Bounded-speedup math against the current profile

Even ignoring the redundancy and imagining a hypothetical "better" vendor
of the same slice, the post-#166 profile caps the ceiling:

| Signal                                     | Share of decode |
| ------------------------------------------ | --------------: |
| `:kiln/attn/gdn/recurrent` NVTX region     |        **1.8 %** |
| `recurrent_gdn_fwd_kernel<128>` CUDA kernel|        **2.4 %** |

Upper bound on overall decode speedup from fully eliminating the
`:kiln/attn/gdn/recurrent` region:

    1 / (1 − 0.018) ≈ **1.018×**

That is below the task's **1.10× abort floor** and an order of magnitude
below the **1.3–2.0× expected range**. Even a theoretical "infinitely
fast" recurrent kernel cannot meet the task's acceptance criteria given
the current profile. This is the same bounded-ceiling argument applied in
`flashinfer-decode-preflight-kiln-2026-04-18` (PR #163) and
`kernel-vendor-precondition-check` (PR #131 precedent).

### Root cause of the stale task

The task body conflates two different kernels:

1. **Per-token GDN recurrent step** (decay → delta → state update → output
   projection). This is what `fused_recurrent_gated_delta_rule_fwd`
   actually implements, and it is what PR #92 already vendored. It owns
   1.8 % of decode today.
2. **GDN gate-path preprocessing** (`:kiln/gdn/gates` + `:kiln/gdn/gated_norm`
   + `:kiln/gdn/qk_norm`). These are the 50.4 % hotspot cited in this
   file's "Recommended next optimization target" section
   (lines 123–145). They happen **outside** the recurrent step — they
   produce the inputs it consumes and post-process the output it produces.
   The correct vendor target for the 50.4 % slice is `chunk_gla_fwd` (or
   a narrower fused `gated_norm + qk_norm` kernel), not
   `fused_recurrent_gated_delta_rule_fwd`.

### Preflight performed

- HEAD verified `c2579a1` on `ericflo/kiln` main.
- `ls crates/` — found existing `kiln-gdn-kernel` with the target kernel
  already compiled in.
- `git log --all -- crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.cu` —
  first-add commit is `7aa62f1` (PR #92, merged 2026-04-17).
- `gh pr view 92` — confirmed title "Vendor fla
  fused_recurrent_gated_delta_rule_fwd for GDN decode" and that scope
  matches the new task exactly.
- Read `crates/kiln-gdn-kernel/src/lib.rs:25-93` to verify the FFI
  surface and scope envelope.
- Read `crates/kiln-model/src/forward.rs:1060-1085` to verify wiring as
  the `seq_len == 1` fast-path branch of `gdn_chunkwise_recurrence`.
- `gh pr list --state open` — no overlapping open PR at the time of
  preflight.

No RunPod pod launched. **Total cost: $0.**

### Re-visit triggers

Propose a new per-token GDN kernel only if **all three** hold on a fresh
`nsys` profile of current `main`:

1. `:kiln/attn/gdn/recurrent` NVTX region ≥ **8 %** of decode wall-clock
   (it is 1.8 % today).
2. `recurrent_gdn_fwd_kernel<*>` CUDA kernel ≥ **10 %** of decode kernel
   time (it is 2.4 % today).
3. A concrete code-level reason the PR #92 kernel is sub-optimal at the
   current workload (e.g. register pressure at a larger head dim, shared
   memory shortfall on sm_90, or an upstream fla change that materially
   re-tunes the per-token step).

Absent those, future per-token GDN proposals should be rejected at
preflight.

### Next actual target

Per the unchanged "Recommended next optimization target" block at the top
of this file (lines 123–145):

- **Vendor `chunk_gla_fwd`** for the combined `:kiln/gdn/gates` +
  `:kiln/gdn/gated_norm` + `:kiln/gdn/qk_norm` hotspot (**50.4 %** of
  decode wall-clock, above the 40 % Phase 6 trigger).
- Or, as a narrower scope, fuse just `gated_norm + qk_norm` (**32.4 %**).

This is the only remaining single-target ≥ 3 % opportunity that has not
already been vendored or ruled out by preflight.


## Phase 6 — `causal_conv1d_update` vendor decode bench (2026-04-18)

Vendored NVIDIA Mamba's `causal_conv1d_update` into `kiln-conv1d-kernel`
(cc-crate + Rust FFI wrapper, bf16/causal/fwd-only, cuda-gated,
sm_86/sm_89/sm_90) and wired `kiln-model/src/backend/cuda.rs` to dispatch
through a new `BackendRuntime::causal_conv1d_update` hint. The candle
fallback path is preserved behind the `KILN_DISABLE_FUSED_CONV1D=1` kill
switch so the same binary can A/B without a rebuild. This replaces the
`:kiln/gdn/conv` decode region (documented at 12.2 % of decode in the
earlier nsys breakdown).

**Hardware:** RunPod NVIDIA RTX A6000 on-demand, Driver 580.95.05,
CUDA 12.8, stock `runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2404`
plus `cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8
libcublas-dev-12-8 libcurand-dev-12-8`.

**Build:** release, `--features cuda`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`.

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
runs per arm. Decode path = `model_forward_paged` (production
HTTP/scheduler path). Reporting `decode_tokens_per_sec` and the inter-token
latency distribution from the latency phase.

### Decode tok/s — 512-prompt × 128-decode

| run    | PRE (candle fallback) tok/s | POST (fused kernel) tok/s |
| ------ | --------------------------: | ------------------------: |
| 1      |                       45.10 |                     45.97 |
| 2      |                       49.04 |                     52.59 |
| 3      |                       44.08 |                     52.52 |
| mean   |                       46.07 |                     50.36 |
| median |                       45.10 |                     52.52 |

Median speedup: **1.164× decode tok/s** (+16.4 %). Mean speedup: **1.093×**
(+9.3 %). Run 1 of the POST arm underperforms the other two — first-fire
JIT/kernel-cache warm-up is the likely cause; runs 2–3 converge tightly.

### Inter-token latency (median of 3 runs)

| stat    | PRE ms | POST ms |
| ------- | -----: | ------: |
| mean    |  22.17 |   19.04 |
| p50     |  22.14 |   18.83 |
| p99     |  26.22 |   24.14 |

Consistent −3 ms mean ITL and −2 ms p99 ITL across runs.

### Parity

`cargo nextest run --release --features cuda -p kiln-model -E
'test(test_causal_conv1d_update_matches_fallback)'`: PASS.

### Verdict

1.164× decode speedup clears the 1.03× ship floor. Kernel ships default-on;
`KILN_DISABLE_FUSED_CONV1D=1` retained behind the `BackendRuntime` seam for
debug/ablation. `:kiln/gdn/conv` is now serviced by the vendored kernel; the
next decode hotspot on the GDN path is `gdn_recurrent_step` (covered by
PR #80 / `kiln-gdn-kernel`).


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

> **STATUS (2026-04-18): This recommendation is SUPERSEDED.** The
> "collapse gates + gated_norm into one kernel" framing turned out to be
> architecturally infeasible (the two NVTX regions are separated by Step
> 7's chunkwise recurrence — see PR #158), and both halves have already
> been attempted independently: PR #158 fused the `:kiln/gdn/gates`
> Step-6 computation (merged, −1.5 pp share); PR #141 fused the
> `:kiln/gdn/gated_norm` Step-8 computation (closed, null result
> 1.003–1.005× at decode shape). See the "Post-PR #158 Decode Profile
> (2026-04-18)" section above for the current live recommendation
> (vendor `chunk_gla_fwd`) and the "Not next target — GDN gate-path
> fusion (preflight, 2026-04-18)" section below for the full preflight
> record explaining why a fresh gate-path fusion PR is not the right
> next move.

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

## Not next target — FlashInfer paged GQA decode (deferred, preflight 2026-04-18)

The project's "Current optimization queue" item *Vendor FlashInfer paged
GQA decode (minimal, decode-only, bf16, hdim128)* — step 3 of the
optimization queue in the project description — is **deferred** based
on the Arm B NVTX numbers immediately above. This section records the
preflight finding so future planning loops do not re-propose it before
the GDN gate-path fusion above lands.

No pod was spun up past the preflight `git clone`. Pod cost: \$0.

### Why deferred

FlashInfer's paged-KV GQA decode kernel replaces the **inner paged
attention kernel** inside kiln's 8 full-attention layers. It does **not**
touch the GDN path (24 layers) and it does **not** touch the `qkv_proj` /
`o_proj` GEMMs (those are cuBLAS BF16 / Marlin today).

Arm B NVTX shares for the full-attention path at current `main`
(`5aa22e1`, 864 decode steps captured):

| region                | % decode time | notes                                           |
| --------------------- | ------------: | ----------------------------------------------- |
| `:kiln/proj/qkv`      |        2.4 %  | Full-attn projection GEMM (289 instances)       |
| `:kiln/proj/o`        |        0.5 %  | Full-attn output projection GEMM (289 instances) |
| `:kiln/attn/full/*`   |     below 0.5 % | Below top-NVTX cutoff; doesn't appear in table |

The paged attention kernel itself (the thing FlashInfer would replace)
is a subset of that sub-0.5 % `:kiln/attn/full/*` slice. Even a
hypothetical **infinite** speedup on that kernel would reduce overall
decode by at most ~0.5 %, i.e. **overall decode speedup ≤ 1.005×**. The
vendor task stated an expected range of 1.3–2× overall decode and a
1.15× abort threshold. Both are mathematically unreachable at the
current profile because attention is already too small a share of
decode.

### Why it's below the threshold on this model

Qwen3.5-4B is a hybrid architecture: **24 GDN linear-attention layers +
8 full-attention GQA layers** (ratio 3:1). The 8 full-attention layers
are the only place FlashInfer could help, and per the NVTX table above
their contribution is dominated by the surrounding projections, not the
attention kernel. This is the opposite regime from a standard
attention-dominated decoder-only LLM where FlashInfer typically shines.

### When to revisit

Re-evaluate after the GDN gate-path fusion recommended in the previous
section lands. If that fusion brings GDN decode share down by ~20 pp
(from 72.5 % → ~50 %), `:kiln/attn/full/*` proportional share roughly
doubles. Even then, full-attention would still only be ~7–8 % of
decode and FlashInfer would still be below the 1.15× abort threshold —
so the right trigger for re-proposing this task is:

1. Full-attention NVTX share ≥ 10 % of decode (per a fresh Arm B
   profile), **and**
2. The inner `:kiln/attn/full/*` region (attention kernel, not qkv /
   o projections) is itself ≥ 8 % of decode.

Until both hold, the next decode-side lever is the GDN gate-path fusion
above, not FlashInfer.

### What would still make sense to vendor from FlashInfer later

- **Paged append / prefill** could help prefill TTFT on long contexts
  (128K+) where flash-attn-2's non-paged shape is less efficient than
  FlashInfer's page-aware kernels. This is a **prefill-path** change,
  not a decode-path change, and should be evaluated independently
  against Phase 6's prefill profile (not the decode profile above).
- **Batched decode** at large batch sizes (not kiln's current
  batch = 1 profile target) is where FlashInfer's tensor-core decode
  kernel is designed to win. Revisit when / if kiln targets larger
  inference batches.

## Not next target — GDN gate-path fusion (preflight, 2026-04-18)

A subsequent planner queued *"Phase 6: Fused GDN gate kernel
(gate_lin + swish + mul + rmsnorm + residual, bf16, decode-shape)"*
citing the **Recommended scope for the next cycle** section above
(lines starting "Vendor or author a fused `gdn_gate` kernel …"). That
section is stale at current `main` (HEAD `838f88f`): the combined
gates + gated_norm fusion it proposed was both **architecturally
impossible as one kernel** and **already attempted as two independent
kernels**, one merged and one closed as a null result. This section
records the preflight so future planning loops do not re-extract the
same redundant task.

No pod was spun up past the preflight `git clone`. Pod cost: **\$0**.

### Scope the task asked for

One fused CUDA kernel, bf16 decode shape only, collapsing:

1. `swish(gate_lin_output)`
2. elementwise multiply with the `gated_norm` input
3. `RMSNorm` over the gated output (epsilon, weight)
4. residual elementwise multiply

Target NVTX regions per the stale recommendation: `:kiln/gdn/gates`
(18.2 %) + `:kiln/gdn/gated_norm` (18.0 %) = 36.2 % of decode time at
the PR #156 profile (HEAD `5aa22e1`).

### Why it's redundant

The two NVTX regions map to two disjoint steps of
`gated_deltanet_forward` in `crates/kiln-model/src/forward.rs`:

| NVTX region            | Step | Computation                                            | Status                              |
| ---------------------- | ---- | ------------------------------------------------------ | ----------------------------------- |
| `:kiln/gdn/gates`      | 6    | `beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)` | **Fused by PR #158** (merged, bf16 in/out, `kiln-gdn-kernel` crate, `gdn_gates.cu`) |
| `:kiln/gdn/gated_norm` | 8    | `bf16(w * rms_normalize(attn_out) * silu(z))`           | **Attempted by PR #141** (closed, null 1.003–1.005× at `rows=32, hidden=128`) |

Between the two steps is Step 7 — the chunkwise recurrence that
produces `attn_out` from `(q, k, v, beta, g)`. PR #158's description
states the architectural finding explicitly:

> "the two regions are separated by the chunkwise recurrence
> (`Step 7 gdn_chunkwise_recurrence`) and cannot be combined into a
> single kernel. They must be tackled independently."

Step 7 internally launches `gdn_chunk_prep` (PR #162), the per-chunk
forward-substitution kernel, matmuls, and the recurrent update — not
something a gate-path elementwise fusion can absorb. "One kernel
covering gate_lin + swish + mul + rmsnorm + residual" is therefore
unreachable on this architecture.

The two independently-fusable halves have both been addressed:

- Step 6 (`:kiln/gdn/gates`): PR #158 fused the 8-op candle chain into
  a single CUDA launch. Post-#158 Arm B profile (HEAD `7132f29`) shows
  `:kiln/gdn/gates` moved from 18.2 % → 16.7 % (−1.5 pp), and the
  proportional share of the downstream `gated_norm` / `qk_norm` also
  shrank (−2.2 pp / −1.4 pp) because PR #158 dispatches fewer upstream
  elementwise tensors.
- Step 8 (`:kiln/gdn/gated_norm`): PR #141 vendored an FLA-style
  `fused_norm_gate` kernel covering `rms_inv * silu(z) * w` in one
  launch. Parity passed (max abs diff < 1e-2, bf16). Decode ITL
  delta was **1.003–1.005× under CUDA graphs**, below run-to-run
  noise and below the acceptance floor, so the PR was closed rather
  than merged. The "residual elementwise multiply" the current task
  description lists as a fourth op is not present in the Step 8 code
  path (`gated_rms_norm`'s output feeds the Step 9 `o_proj` GEMM
  directly — there is no per-element residual multiply between them).

### What the current recommendation actually says

The "Post-PR #158 Decode Profile (2026-04-18)" section above —
captured on HEAD `7132f29` right after PR #158 landed — replaces the
stale "Recommended scope" with a different next move:

> "**Vendor `chunk_gla_fwd` from flash-linear-attention (roadmap step
> 2).** The combined decode share of `gdn/gates` + `gdn/gated_norm` +
> `gdn/qk_norm` + `gdn/conv` is still **57.9 %** of wall-clock. A
> narrow gates-only fusion (like #158) only chips at the first of
> those four regions. Vendoring the upstream `chunk_gla_fwd` kernel
> that fuses the full GDN decode chain (conv + in_proj + gates +
> gated_norm + qk_norm + recurrent) is the highest-leverage move
> available — it would collapse ~58 % of decode wall-clock into a
> single kernel dispatch and directly eliminate the elementwise zoo
> that #158 could not reach."

The project description's **Current optimization queue** lists the
same item as step 2 ("Vendor fla-org/flash-linear-attention
`chunk_gla_fwd` (minimal)"). Agent note `kiln-gdn-bottleneck-postpr105`
points in the same direction.

### When to revisit the narrow gate-path fusion

Only if all three hold on a *fresh* Arm B profile:

1. `:kiln/gdn/gated_norm` is still ≥ 15 % of decode (i.e. a combined
   GDN-chain kernel never landed and this region is still the largest
   single elementwise hotspot), **and**
2. A parity-validated fused `gated_norm` kernel measures
   ≥ 1.10× decode ITL on at least one of the W4A16 arms (clearing
   PR #141's null-result ceiling), **and**
3. The fusion does not duplicate work that a vendored `chunk_gla_fwd`
   would already subsume.

Until then, the next decode-side lever is `chunk_gla_fwd`, not another
gate-path fusion PR.

### Preflight record

- Local clone of `ericflo/kiln` at HEAD `838f88f` (post-PR #163).
- `grep 'Recommended scope for the next cycle' PROFILING.md` →
  lines now include the STATUS callout flagging this section stale.
- `gh pr list --limit 20 --state all | grep -iE 'gdn.*gate|gate.*fusion|fused.*gate'`
  → PRs #158 (MERGED, gates fused) and #160, #163 (docs/re-profile).
  PR #141 surfaced via `gh pr view 141` as the closed null-result
  attempt for `:kiln/gdn/gated_norm`.
- Upstream search:
  - FLA (`flash-linear-attention`) exports `naive_gdn_gate` as an
    elementwise reference but no standalone fused CUDA kernel; the
    gate math is folded into the larger `chunkwise_gdn` Triton
    kernel, which is what step 2 of the optimization queue vendors
    next.
  - Liger-Kernel has no GDN-specific gate op.
  - `fused_norm_gate` (FLA) is exactly the shape PR #141 vendored
    and measured as null at the Qwen3.5-4B decode shape.
- Code inspection confirms Step 8 has no per-element residual multiply
  to fuse: `gated_rms_norm`'s output feeds `o_proj` (Step 9) directly.

Pattern match: identical redundancy class as the "chunk_gla_fwd
already vendored" incident (PR #131 redirect) and the "FlashInfer
paged GQA decode below threshold" preflight (this report, section
above). Doc-only resolution, \$0 pod spend. See agent note
`kernel-vendor-precondition-check` for the general rule.

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

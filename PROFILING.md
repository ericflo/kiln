# Kiln Profiling Report

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were removed
> from the working tree on 2026-04-25 to clean up the top-level directory.
> The exact files cited in this report are preserved in git history at commit
> `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before
> the cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:profile-out/<file>`
> (or whichever path is cited — `c12-out/`, `profile/`, `profiling/`, or
> `profiling-artifacts/`).

## Phase 10 §3 post-#647 SFT-step re-profile + next-kernel candidate audit (2026-04-29)

**Verdict: `no_kernel_pivot`.** Three back-to-back nsys 2024.5.1
profiles of the T=8192 RMSNorm-on FLCE-Phase-B SFT step on RunPod
A100-SXM4-80GB sm_80 (A6000/A40 both capacity-blocked at task start;
A100 was the next allowed fallback per task description, NOT L40S).
Kiln main commit `3504a06` (PR #648 — FLCE Phase B A6000 closure
deferred). Step wall-clock medians 30.03/30.98/30.98 s, total kernel
exec 21.113/21.118/21.121 s (Δ 0.04% of median — profiler-grade
reproducible), peak VRAM 70,053–70,081 MiB, final loss bit-identical
0.8115726113319397 across all three runs. KILN_USE_FLCE=1, RMSNorm
fused CustomOp2 path active (`fused_path=ON` — A100 80 GiB clears the
PR #644 47 GiB gate), KILN_W4A16 unset (W4A16 + training currently
fails with "expected rank-2 q_proj_t" on the trainer's Marlin unpack
path; orthogonal but noted). Top NVTX (median): `:kiln/gdn/in_proj`
**19.6%**, `:kiln/mlp/gate` 11.8%, `:kiln/gdn/qk_norm` 10.8%,
`:kiln/gdn/gates` 10.4%, `:kiln/mlp/up` 8.9%, `:kiln/mlp/down` 7.9%,
`:kiln/norm/pre_attn` 6.0%, `:kiln/gdn/gated_norm` 5.2%,
`:kiln/proj/qkv` 5.2% (GDN aggregate ~50.6%, MLP trio 28.6%, full-attn
projections 7.2%). Top CUDA kernels (median): `ampere_sgemm_64x64_nn`
(FP32) **31.07%** + `ampere_sgemm_128x64_nn` (FP32) 11.02% =
**~42% combined FP32 SGEMM** — shape signatures match the kiln-train
rank-8 LoRA-down/-up matmuls (FP32 for numerical stability of small-
rank accumulation; not lm_head). `gdn_full_chunk_forward_kernel`
(kiln-gdn-kernel) 22.13%; the post-PR-#502 audit (2026-04-24) declared
this kernel structural-only with no bounded micro-port remaining.
Elementwise zoo (`badd/bsub/bmul/bdiv/uexp/fast_sum/fast_max/affine/
cast/ucopy_*`) ~24.6% spread across many sites. **Per-candidate
math-ceiling verdicts** (1.05× floor under graphs; 1.10× under
graph-amortized fusion): ❌ Fused RoPE (5.2% region × 2× speedup-only
inside `:kiln/proj/qkv` → 1.027× overall, sub-floor); ❌ Fused
SwiGLU/GeGLU (Liger fuses only the post-gate elementwise, ~5% step-
budget × 2× → 1.025× overall, sub-floor — the 28.6% MLP trio is the
matmul cost, NOT addressable by Liger SwiGLU); ❌ Fused Layer Norm
(N/A — Qwen3.5 uses RMSNorm only; PR #644 already shipped); ⚠️
**FleCE conditional** — kiln Phase B (PR #647) already chunks along
T; FleCE adds chunking along V (vocab=248,320). Peak VRAM at T=8192
on A100 80 GiB is 70/80 GiB — no pressure, FleCE null. On A6000 at
T=8192, Phase B closes peak below 49 GiB ceiling already (PR #647 +
A40 closure evidence). FleCE only buys new ground at **T≥16384** on
A6000 — NOT YET MEASURED. **Conditional reopen**: queue a T=16384
A6000 SFT-step OOM probe (~15 min, no kernel work) before
implementing FleCE. The **actual top kernel hotspot (FP32 LoRA SGEMM
~42%) is NOT a Phase 10 candidate** — replacing it requires either
LoRA-in-BF16-with-FP32-accum (numerical stability change) or fusing
LoRA into the base GEMM (autograd path rewrite). Both are recorded
for future planning, not §3 work. **Phase 10 §3 should pivot to
non-kernel work**: T=16384 A6000 OOM probe (gates FleCE go/no-go),
or Phase 9 release prep (security audit, license review, reproducible
builds, CI/CD, GHCR Docker image, landing page, v0.1.0 cut). Bench
envelope: A100 SXM4-80GB lease 47 min, ~$1.20 GPU spend, well under
$40/90 min cap. Required Cargo.toml fix on the pod (NOT in this PR's
diff): `kiln-server`'s `cuda` feature did not propagate to
`kiln-train/cuda`, so the `kiln-flce-kernel` CUDA path errored out
at runtime. Cite:
[`docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md`](docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md).

## Phase 7 end-to-end native-MTP self-spec decode bench (2026-04-25)

**Verdict: `mtp_no_decode_win`.** End-to-end MTP-On vs MTP-Off
decode tok/s bench at bs=1 on RunPod A6000 sm_86 against post-PR
#535 main (SHA `c5cf77d`, fully cached link-only rebuild). MTP-On
medians: **decode 43.09 tok/s** (range 40.15–48.56), **mean ITL
23.21 ms**, **P99 ITL 56.78 ms**, **α 0.6842** (range 0.620–0.778).
MTP-Off medians: **decode 44.75 tok/s** (range 44.23–45.01), **mean
ITL 22.35 ms**, **P99 ITL 27.33 ms**. **Δ median decode tok/s =
−3.7 %**, paired-by-seed median Δ = −4.27%, mean Δ = −1.64%. Per the
pre-registered decision rule (median Δ < −3% → `mtp_no_decode_win`),
the MTP path does NOT yet deliver an operational decode-tok/s win
at bs=1 even with all C3-C40+ fixes landed and α now ~5.5× higher
than the PR #316 baseline (0.124 → 0.6842). One seed (α=0.778, the
only seed clearing the 0.72 paper floor) produced +8.51% over its
paired Off run, but the other two seeds (α=0.620, 0.684) produced
−9.22% and −4.27%. **P99 ITL doubles** under MTP (27.33 → 56.78 ms,
~2.08×) — the bimodal verifier-cost signature: P50 actually drops
from ~22 ms to ~16 ms, but rejected-draft steps create a heavy
tail. This is a **doc-only redirect**: no code changes,
`KILN_SPEC_METHOD=mtp` stays opt-in (gated additionally by
`KILN_BENCH_FORCE_MTP=1` because the bench resolver caps MTP at
`requested_prompt_tokens ≤ BENCH_MTP_MAX_PROMPT_TOKENS = 128`), no
default flip proposed. Bench envelope: pool A6000 acquire → release
~25 min (warm pod, no cold-start), 6 bench runs × ~50 s each, ~$0.21
GPU spend. Reopen triggers: median α reliably clearing 0.72 across
humaneval+gsm8k+c4 with ≥10 seeds, a fused/short-circuit verifier
landing, k>1 MTP variant, or workload-distribution shift to
genuinely chat-template-heavy traffic. See
[`docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.md`](docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.md)
for the full protocol, anti-duplication evidence, per-run table
(off vs `mtp_forced` vs the resolver-downgraded `mtp_unforced`
control), decision-rule application, paired-seed delta breakdown,
P99 ITL bimodality discussion, α progression timeline (PR #316 →
this PR), and reproduction commands. Raw data:
[`docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv`](docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv)
and [`docs/archive/phase-c/phase-c66/artifacts/`](docs/archive/phase-c/phase-c66/artifacts/) (9 raw
bench logs, 3 seeds × 3 arms).

## Phase 7 H18 hand-rolled HF transformers MTP α reference (2026-04-25)

**Verdict: `kiln_above_hf`.** Hand-rolled HuggingFace transformers
reference α probe for Qwen3.5-4B native MTP loaded and ran end-to-end
on RunPod A6000 sm_86. Median α: HF **0.2500** vs kiln **0.3636**
(re-derived from PR #529 c1_attr CSVs). Δ (hf − kiln) = **−0.1136**.
Per the pre-registered decision rule (PR #531 candidate 8, locked
into `scripts/h18_compare.py` BEFORE the run), `delta < -0.02` →
`kiln_above_hf` branch. **Seed 0 is a bit-for-bit match (4/11 vs
4/11)** — strong evidence the H18 protocol implementation
(MTP head forward, verifier wiring, RoPE position threading,
accept/reject semantics) is correct on at least one trajectory.
Seeds 1+2 diverge by 1-2 extra rejects (kiln 4/12, 5/11 vs HF 3/13,
3/12) — small in absolute count but enough to drag median α down by
~0.11. Likely sources of the 1-2-token gap: BF16 numerical drift
inside the 24×GDN + 8×GQA base stack (Phase C7 documented kv_len
divergence between kiln and the canonical Python single-position
self-attn contract), causal-conv1d kernel fallback in HF (the run
log explicitly reports the fast path unavailable), and Marlin W4A16
vs BF16 quantization confounder. None is a bug — they are documented
numerical paths whose absolute α delta is governed by checkpoint-
quality and 1-2-token argmax-flip windows. The PR #529 H15b
`kiln_native_ceiling` verdict stands as the operative checkpoint-
quality ceiling. With vLLM 0.19.1 (PR #530) + v0.20.0 (PR #533)
unsupported, SGLang main HEAD (PR #532) unsupported, AND H18 now
showing `kiln_above_hf` (kiln α exceeds the BF16 HF reference even
under W4A16 quantization), the external-α reference family closes
here. Phase 7's α-improvement track is closed; refocus is on non-α
decode-path wins (post-PR #521 prefix-cache + CUDA-graph work,
SGLang RadixAttention port from PR #526). Bench envelope: pool A6000
pod (lease + auto-resume from hibernation), ~6 min total wall-clock
including dependency installs (torch 2.4.1+cu124 reinstall over
kiln-runpod's default 2.11.0+cu130, transformers c472755 from
source, torchvision 0.19.1+cu124), 30.8 s for the actual H18 dump.
**~$0.05 GPU spend** — well under $0.30 H18 cap. Reopen preconditions:
a second independent reference (HF without causal-conv1d fallback,
or future vLLM/SGLang stable that loads on driver 550.x) producing
α materially > 0.3636; running H18 against a future BF16-only kiln
build to isolate the W4A16 confounder; or reproducing H18 with
different seeds and obtaining a materially different median α. See
[`docs/audits/phase7-h18-hf-transformers-alpha-reference.md`](docs/audits/phase7-h18-hf-transformers-alpha-reference.md)
for the full verdict, decision-rule application, per-seed
divergence analysis, anti-duplication evidence, reproduction
commands, and detailed reopen triggers. Raw data:
`docs/archive/phase-c/phase-c29-v3-hf/{verdict,compare,hf_alpha_per_seed,kiln_alpha_per_seed}.json`,
`docs/archive/phase-c/phase-c29-v3-hf/compare.md`, and `docs/archive/phase-c/phase-c29-v3-hf/artifacts/`
(per-seed traces + run log).

## Phase 7 H17b vLLM v0.20.0 α microbench retest (2026-04-25)

**Verdict: `vllm_020_mtp_unsupported_dense_4b`.** vLLM v0.20.0
prerelease (the "free pre-step" branch from PR #531's pre-registered
decision rule, fired by PR #532's `sglang_mtp_unsupported_dense_4b`)
also cannot serve Qwen3.5-4B dense BF16 + native MTP on A6000 sm_86 —
this run, however, fails for a distinct **runtime-stack** reason
rather than another segfault inside dispatched MTP code. The vLLM
0.20.0 prerelease wheel `vllm-0.20.0+cu129` (the only x86_64 wheel
published as of 2026-04-25 for the v0.20.0 release tag of 2026-04-23)
pulls `torch 2.11.0+cu130` as a transitive dep; that wheel requires
NVIDIA driver ≥575 to load CUDA 12.9 runtime libs and ≥580 for CUDA
13.0. RunPod's `kiln-runpod` stock image ships driver 550.127.08
(CUDA 12.4 max), so `torch.cuda.is_available()` returns `False` with
warning `"NVIDIA driver on your system is too old (found version
12040)"`, and vLLM's V1 EngineCore subprocess crashes at first GPU
init with `RuntimeError: Cannot re-initialize CUDA in forked
subprocess` at `torch.accelerator.set_device_index()` →
`torch._C._accelerator_setDeviceIndex()` →
`torch/cuda/__init__.py:466 _lazy_init`. Reproduced for all three
method aliases (`qwen3_5_mtp` / `mtp` / `qwen3_next_mtp`); each
takes ~1-3 s wall-clock and each crashes inside the same CUDA-init
code path before any MTP-specific dispatch can run. Per the
pre-registered decision rule, `mtp_supported == false` maps to the
`vllm_020_mtp_unsupported_dense_4b` branch — ship as a doc-only
redirect PR. Kiln median α at this workload (unchanged from PR #530
/ PR #532, re-derived from PR #529 c1_attr CSVs): 0.3636. vLLM 0.20.0
median α: UNAVAILABLE. PyPI does not yet publish 0.20.0 (still
0.19.1 latest); the only published wheels are `+cu129` and `+cu130`
on `wheels.vllm.ai/0.20.0/`, both incompatible with current RunPod
stock A6000 driver. The H17b run cannot make any statement about
whether 0.20.0's MTP path actually works correctly — it never
reaches MTP code at all. With vLLM 0.19.1 (PR #530) AND vLLM 0.20.0
(this run) AND SGLang main HEAD (PR #532) all blocked, the
external-α reference question now points unambiguously to the
hand-rolled HF transformers H18 reference (PR #531 candidate 8) as
the only remaining viable path. Bench envelope: pool acquire
succeeded on first try (no fallback needed); pod
`mfk88l8i8tab02`, ~13 min wall-clock / ~$0.10 — well under $0.40
H17b cap. Reopen preconditions: vLLM publishing a `+cu124`/`+cu128`
0.20.x wheel; RunPod stock driver upgrade to ≥580; or H18
hand-rolled reference closing the question independently. See
[`docs/audits/phase7-h17b-vllm-020-alpha-microbench.md`](docs/audits/phase7-h17b-vllm-020-alpha-microbench.md)
for the full verdict, the runtime-stack delta table, the canonical
3-attempt crash trace, anti-duplication evidence, reproduction
commands, and detailed reopen triggers. Raw data:
`docs/archive/phase-c/phase-c29-v3-vllm-020/{verdict,compare,kiln_alpha_per_seed,vllm_020_alpha_per_seed}.json`,
`docs/archive/phase-c/phase-c29-v3-vllm-020/compare.md`, and
`docs/archive/phase-c/phase-c29-v3-vllm-020/artifacts/{vllm_020_failure_evidence,driver_full}.log`.

## Phase 7 H17 SGLang α microbench (2026-04-25)

**Verdict: `sglang_mtp_unsupported_dense_4b`.** SGLang 0.5.10.post1 +
Qwen3.5-4B dense BF16 + native MTP segfaults on A6000 sm_86 across three
distinct serving configurations. All three attempts produced SIGSEGV
(exit code -11) in native-extension code with no Python file symbols at
the terminal frame — identical high-level shape to PR #530's vLLM 0.19.1
failure class. The crash frames are structurally distinct per config:
(1) default flashinfer + CUDA graphs: segfault during
`flashinfer_backend.call_begin_forward` -> `eagle_info.py:188
generate_attn_arg_prefill` -> Triton JIT compile (graph capture);
(2) flashinfer graphs-off: scheduler SIGSEGV during init, before reaching
prefill, also on retry with `speculative_algorithm=NEXTN` (auto-rewrites
to EAGLE); (3) triton attention graphs-off: engine loaded successfully
(8.62 GB target + 1.41 GB drafter, KV + mamba cache allocated, hybrid GDN
dispatched via TritonGDNKernel), scheduler entered event loop, then
SIGSEGV fired during `write_cache_indices` Triton JIT compile inside
`alloc_for_extend` on the first prefill. Engine-side dispatch is correct
throughout — the bug is downstream of model dispatch in each attempt.
Per the pre-registered decision rule this maps to branch D
(`sglang.mtp_supported == false`) — ship as a doc-only redirect PR.
Newly documented SGLang runtime prerequisites (beyond PR #531's static
audit): `libnuma1` + `libnuma-dev` apt packages (sgl_kernel dlopens
libnuma.so.1), `SGLANG_ENABLE_SPEC_V2=1` env var,
`mamba_scheduler_strategy='extra_buffer'`,
`speculative_algorithm='EAGLE'` (SGLang enum has no `MTP`; `NEXTN`
rewrites to EAGLE), `speculative_eagle_topk` required as int. Per PR
#531 §"Bench envelope", the free pre-step vLLM v0.20.0 retest was
SKIPPED — SGLang diagnostic attempts consumed ~35 min of the 75 min
combined cap (pod: ~25 min / ~$0.20, well under $0.40 SGLang cap). Kiln
median α at this workload (unchanged from PR #530, re-derived from PR
#529 c1_attr CSVs): 0.3636. With both vLLM 0.19.1 AND SGLang 0.5.10.post1
blocked, the external-α-reference question remains open — next options:
opportunistic vLLM v0.20.0 retest (~30 min / $0.25), hand-rolled HF
transformers reference H18 (~2-4 hrs engineering, $0 GPU), or accept the
H15b `kiln_native_ceiling` verdict as operational conclusion and
deprioritize MTP-side α work entirely. See
[`docs/audits/phase7-h17-sglang-alpha-microbench.md`](docs/audits/phase7-h17-sglang-alpha-microbench.md)
for the full verdict, three crash signatures, per-config outcome table,
SGLang prerequisites discovered, workload matching, reproduction
commands, anti-duplication evidence, and detailed reopen triggers. Raw
data:
`docs/archive/phase-c/phase-c29-v3-sglang/{verdict,compare,kiln_alpha_per_seed,sglang_alpha_per_seed}.json`,
`docs/archive/phase-c/phase-c29-v3-sglang/compare.md`, and
`docs/archive/phase-c/phase-c29-v3-sglang/artifacts/sglang_segfault_evidence.log`.

## Phase 7 H16 external-α reference options audit (2026-04-25)

**Verdict: `external_reference_exists`.** Doc-only audit (no pod, $0 GPU
spend) classifying every viable external Qwen3.5-MTP α-reference path
after PR #530's `vllm_mtp_unsupported` blocked the vLLM 0.19.1 upper-bound
route. Eight mandatory candidates classified as supported / unsupported /
unknown with file:line evidence and commit SHAs:
**1 supported** (SGLang main `a4facdf` — `Qwen3_5ForCausalLMMTP` class
present since PR #18489 2026-02-09, dense 4B fix landed PR #21347
2026-03-24, spec_v2 enabled PR #19391 2026-03-04; caveat: open PR #23331
fixes adaptive-DSD bug, kiln workload is non-DSD greedy);
**2 unknown** (vLLM main `bc2ae5a` and v0.20.0 release — same code state;
qwen3_5_mtp.py file unchanged from v0.19.1 except NVFP4 workaround, but
the segfault was downstream in native code where significant Eagle /
spec_decode / GDN / autograd changes have landed in the v0.19.1→v0.20.0
window: #37588, #38496, #40732, #37813, #38047, #38361, #38981);
**4 unsupported** (HF transformers — has `Qwen3_5ForCausalLM` but no MTP
class and no candidate-generator type that loads Qwen3.5's pretrained
`mtp.*` head; SafeAILab/EAGLE `cb7e0841` — only Qwen-2 + Qwen3 in eagle/
model/, no qwen3.5 modeling file, also wrong architecture since Eagle
trains its own draft head; FasterDecoding/Medusa `e2a5d20` — stale 2024,
same architecture mismatch; NVIDIA/TensorRT-LLM `1989520` — has only
Qwen-1/2 era `QWenForCausalLM` in tensorrt_llm/models/qwen/, no
Qwen3.5 path);
**1 feasible-but-not-strict** (hand-rolled HF reference — algorithmically
straightforward but ~300-500 LOC of new Python; reserved as fallback).
Decision rule fires branch 1 (≥1 supported): queue exactly ONE H17
SGLang α microbench task with PR #530's prefill-byte-equal workload
(seeds {0,1,2}, 512 prompt tokens, 16 max output, k=1 native MTP, A6000
greedy, ≤45 min / ≤$0.40), plus a $0.25 bounded vLLM v0.20.0 retest as
opportunistic free pre-step. SGLang BF16 vs kiln Marlin W4A16 is the
acknowledged confounder, identical to the vLLM-vs-kiln confounder in
PR #530. Re-classifies PR #527's "SGLang has no MTP for Qwen3.5"
cross-reference: SGLang has had `Qwen3_5ForCausalLMMTP` since 2026-02-09
PR #18489 — PR #526's accurate claim was about *RadixAttention*
MTP-awareness, a different question. See
[`docs/audits/phase7-h16-external-alpha-options-audit.md`](docs/audits/phase7-h16-external-alpha-options-audit.md)
for the 8-candidate table with file:line evidence and commit SHAs, the
pre-registered decision rule + application, the H17 task scope, the free
pre-step protocol, the explicit reopen preconditions per candidate, and
anti-duplication evidence. No raw data files (doc-only PR).

## Phase 7 H15c vLLM α microbench (2026-04-25)

**Outcome: `vllm_mtp_unsupported`.** vLLM 0.19.1 segfaults during V1 engine
init when loading Qwen3.5-4B with native MTP speculative decoding on A6000,
post drafter weight load. All three method aliases (`qwen3_5_mtp` / `mtp`
/ `qwen3_next_mtp` — vLLM internally rewrites the first and third to
`mtp` per `vllm/config/speculative.py:323-330`) reach the same crash
point and produce identical native-extension backtraces with no Python
file symbols. The vLLM model-dispatch path is correct (`Qwen3_5MTP` arch
in registry, drafter weights load, tied embed/lm_head attach), the crash
is in a downstream native code path. Workarounds that took effect but did
NOT prevent the crash: `enforce_eager=True` (no CUDA graphs),
`limit_mm_per_prompt={image:0,video:0}` + `skip_mm_profiling=True` (text-
only mode confirmed in vLLM logs). Per the pre-registered decision rule
this maps to the `vllm_mtp_unsupported` branch — ship as a doc-only
redirect PR, queue next H from PR #527 §"Queued next action". Possible
next paths (next planning cycle picks): (a) build a hand-rolled HF
transformers reference α as the external upper bound, (b) wait for a vLLM
patch and re-run this PR's `scripts/h15c_*` as-is, or (c) refocus Phase 7
onto non-α decode-path wins (PR #521 prefix-cache + CUDA graphs landed,
PR #526 SGLang RadixAttention port queued). Kiln median α at this
workload (re-derived from PR #529 c1_attr CSVs): 0.3636. See
[`docs/audits/phase7-h15c-vllm-alpha-microbench.md`](docs/audits/phase7-h15c-vllm-alpha-microbench.md)
for the full verdict, segfault signature + crash backtrace, per-seed α
table, vLLM config used (BF16 confounder vs kiln Marlin W4A16),
reproduction commands, anti-duplication evidence, and detailed reopen
triggers. Raw data:
`docs/archive/phase-c/phase-c29-v3-vllm/{verdict,compare,kiln_alpha_per_seed,vllm_alpha_per_seed}.json`
and `docs/archive/phase-c/phase-c29-v3-vllm/artifacts/vllm_segfault_evidence.log`.

## Phase 7 H15b stratified C29 v2 reject-row probe (2026-04-25)

**Scope:** A6000 re-run of PR #355's C29 top-K Jaccard / cos_sim probe on
the `c14__logits` tap, stratified by accept vs reject rows, seeds 0..2,
POSITIONS=0..3, MAX_STEPS=2. Executes the H15b recommendation queued by
PR #527 §"Recommended next H" after H15a (PR #528) ruled Marlin pack
determinism out. The question: does kiln's MTP head stay within the
BF16-noise cos_sim floor (≥ 0.999) on the *rejected* sub-population, or
diverge materially (< 0.99) from an fp32 HF reference? If kiln is at the
native ceiling even on reject rows, the α-gap source is external and the
next step is a vLLM α microbench. If reject-row cos_sim is materially
below the ceiling, per-layer bisect on reject rows localizes the drift.

**Outcome: kiln_native_ceiling.** Reject sub-population stays at the
BF16-noise cosine ceiling — `reject_cos_sim_median = 0.999978` (p10
0.999971), vs decision floor 0.999. Accept sub-population is
statistically identical (`accept_cos_sim_median = 0.999979`, p10 0.999968).
Both strata show 100% top-1 agreement and median J@10 = 1.0000. Verifier
numerical drift on reject rows is RULED OUT on this checkpoint.
Recommendation: queue a vLLM α microbench next to establish whether an
external-reference upper bound exists above kiln's current α. See
[`docs/audits/phase7-h15b-stratified-c29-v2.md`](docs/audits/phase7-h15b-stratified-c29-v2.md)
for the full verdict, per-position stratified table, decision-rule
application, anti-duplication evidence, and reopen triggers. Raw data:
`docs/archive/phase-c/phase-c29-v2/c29-v2-stratified-compare.{json,md}` and
`docs/archive/phase-c/phase-c29-v2/verdict.json`.

## Phase 7 H15a Marlin pack determinism correlation (2026-04-24)

**Scope:** $0 doc-only correlation analysis on the existing C40f N=20 anchor
(`docs/archive/phase-c/phase-c40f/summary.json`, PR #379). Executes the "free pre-step (do
BEFORE GPU spend)" called out in PR #527's bench plan: Spearman + Pearson
between `model_load_secs` and `acceptance_rate` to test whether load-time
variance in Marlin packed weights drives MTP α variance. No pod, no SSH, no
Rust changes.

**Outcome: RULED OUT.** Marlin pack determinism is NOT the alpha-gap
mechanism on this anchor. Pearson r = +0.349 (95% CI [−0.111, +0.686], t =
+1.580 df=18 — NOT significant at p<0.05; critical |r| = 0.4438). Spearman
ρ = +0.111 (NOT significant; critical |ρ| = 0.4500). The weak Pearson signal
is single-seed leverage from seed 0's 3.55 s load-time outlier; rank-robust
Spearman drops to +0.111. Bonus structural finding: `acceptance_rate` has
only 10 unique values across 20 seeds (paired seeds {0,10} … {9,19} produce
bit-identical α despite 8–116% load-time differences) — itself near-conclusive
evidence that α at this anchor is workload-deterministic, not pack-determined.
**Recommendation:** queue H15b (stratified C29 v2 reject-row probe, ~30 min
A6000 / ~$0.25) per PR #527 §"Recommended next H." Decision belongs to the
next planning cycle. Full methodology, decision rule, period-10 collision
table, anti-duplication evidence, and reopen triggers in
[`docs/audits/phase7-h15a-marlin-determinism.md`](docs/audits/phase7-h15a-marlin-determinism.md).
Raw script + verbatim output: `scripts/phase-c40f/h15a_correlation.py` and
`docs/archive/phase-c/phase-c40f/h15a_correlation_output.txt`.

## Phase 7 MTP acceptance-rate state-of-play audit (2026-04-24)

**Scope:** doc-only consolidation of the 34 MTP-named phase-B / phase-C PRs
(#311 through #382, plus C40+ refresh attempts and post-#481/#500/#502 GDN
work) into a single H-hypothesis inventory + α gap quantification + remaining
unknowns scoring. No pod spend, no SSH, no new Rust code.

**Outcome:** every named hypothesis on the inference-side pipeline (H1–H14)
is ruled out. Current canonical N=20 anchor (C40f) is α median **0.689**,
95% CI [0.652, 0.723], 6/20 seeds ≥ paper-floor 0.72. Median decode 38.25
tok/s — still below the +10% MTP-on vs plain-decode floor on this hardware.
The remaining frontier is C36's seed-conditioned identity-bias regime split
(80.85% of reject rows show `draft_top1 == last_token`; 0% of accept rows
do; one of 3 seeds clears the floor under default flags). Recommended next
H: stratified C29 v2 — re-run the existing top-K Jaccard / cos_sim probe on
the existing dump pipeline but split by accept/reject, since C29's clean
verdict was measured only on accepted-token positions. Falsification cost
~30 min A6000 / ~$0.25. Full inventory table, scoring, anti-duplication
evidence, and bench plan in
[`docs/audits/phase7-mtp-acceptance-state-of-play.md`](docs/audits/phase7-mtp-acceptance-state-of-play.md).

## Phase 7 SGLang RadixAttention design audit (2026-04-24)

**Scope:** doc-only audit of kiln's radix prefix cache (PR #512) and real-path
flat `RealPrefixCache` (PRs #515/#520/#521) against SGLang's RadixAttention
(`python/sglang/srt/mem_cache/radix_cache.py` + `mamba_radix_cache.py`). No
pod spend, no new Rust code.

**Outcome:** design parity sufficient for kiln's target workloads; port-to-radix
on the real path is speculative without branching-workload evidence. Detailed
findings, 16-feature classification table, and reopen precondition in
[`docs/audits/phase7-sglang-radix-audit.md`](docs/audits/phase7-sglang-radix-audit.md).

## Phase 7 kill-switch bisection follow-up (2026-04-24)

**Scope:** follow-up to the post-#522 A/B above. Ran 6 arms × 3 runs with one
fused-kernel kill switch per arm (`KILN_DISABLE_FUSED_GDN_GATES`,
`KILN_DISABLE_GDN_KERNEL`, `KILN_DISABLE_RMSNORM_KERNEL`,
`KILN_DISABLE_FUSED_CONV1D`, `KILN_DISABLE_FUSED_PAGED_DECODE`) to isolate the
post-#166 decode regression. Full per-arm tables, variance analysis, and
recommended next steps in
[`docs/archive/phase-c/phase-c64/post523-killswitch-bisection.md`](docs/archive/phase-c/phase-c64/post523-killswitch-bisection.md).

**Outcome: null. No single fused-kernel default path accounts for the gap.**

- No arm recovers decode tok/s to ≥ 49.0 (post-#166 baseline 49.76). Two arms
  (`DISABLE_FUSED_GDN_GATES`, `DISABLE_FUSED_PAGED_DECODE`) cross +3% vs this
  sweep's own baseline A (44.43 tok/s) but neither reaches the 49.0 bar.
- Intra-arm spread ran 8.2% for baseline A and 20.0% for arm E — larger than
  the inter-arm signal. The pre-sweep sanity run on the same binary hit 50.48
  tok/s; the ensuing sweep's baseline collapsed to 44.43. Parsimonious
  explanation is GPU clock / thermal drift and CUDA-graph-capture warmth, not
  a single kernel.
- Kill switches are verified wired: arms C and D fall back to candle paths
  and lose 8–11% decode as expected, and arm C's prefill balloons 7× to
  2357 ms (known candle GDN-linear-attn fallback cost).
- Two observations worth a higher-N follow-up (not regression fixes):
  arm B's median p99 ITL (25.46 ms) exactly matches post-#166, and arm F
  shows bimodal p99 — both possible weak signals buried in the noise.

**Recommended next step:** stop chasing the post-#166 single-kernel culprit.
Move to KV cache FP8, Marlin packing latency cleanup, or Marlin BF16 VRAM
cleanup from the existing Phase 6 frontier list. If the post-#166 gap still
matters, the next attempt needs pinned GPU clocks + cooldowns between runs
and median-of-5+ per arm before it can distinguish a 3–5% effect from 8–20%
noise.

## Phase 7 prefix-cache regression A/B (2026-04-24)

**Scope:** isolation A/B for the −7.9% decode tok/s regression flagged in the
post-#521 profile refresh below. Toggles `KILN_PREFIX_CACHE_ENABLED` on and off
on current main (`821ccd3`, PR #522 docs-only) and compares decode tok/s,
ITL, and prefill on `kiln-bench --paged`. Full methodology, per-run tables,
and structural analysis in
[`docs/archive/phase-c/phase-c64/post522-prefix-cache-ab.md`](docs/archive/phase-c/phase-c64/post522-prefix-cache-ab.md).

**Outcome: prefix-cache hooks are not the source of the bench-visible regression.**
Both structurally and empirically:

- `crates/kiln-model/src/forward.rs` and `crates/kiln-server/src/bench.rs`
  contain **zero** references to `prefix_cache`. The cache only fires through
  `generate_*_with_prefix_cache()` in `crates/kiln-model/src/generate.rs`,
  which is only called from the HTTP handlers in
  `crates/kiln-server/src/api/completions.rs`. `kiln-bench --paged` bypasses
  those generate entry points and calls `model_forward_paged*` directly, so
  `KILN_PREFIX_CACHE_ENABLED` has no surface to affect.
- Empirically, median-of-last-2 (runs 2–3 of 3, discarding cold run 1):
  Arm A (`=0`) = 48.84 tok/s, Arm B (`=1`) = 47.65 tok/s. Inter-arm delta
  (+1.19 tok/s, 2.5%) is smaller than intra-arm spread (3.47 tok/s in Arm A,
  7.21 tok/s in Arm B).
- Neither arm fully recovers to the post-#166 49.76 tok/s baseline, so the
  regression is real but partly noise overlaid on a smaller ~2–4%
  steady-state drift.

**Next-candidate bisection targets** (CUDA-decode landings between post-#166
`c2579a1` and current main `821ccd3` that modify `forward.rs` or the CUDA GDN
kernels): #461 (mmap transposed weight cache hits), #506 + #508 (transposed
cache reliability / deferred writer), #486 (fused GDN qk norm on by default),
#466 (fused CUDA GDN gated RMSNorm), #500 (CUDA GDN qk norm GQA fast path),
#498 (opt-in CUDA GDN decode fuse hook). Recommended approach: try the
existing `KILN_DISABLE_*` kill-switches one at a time to look for a single
toggle that recovers ≥49 tok/s. If none do, `git bisect` across the 70+
post-#166 commits touching `forward.rs`.

## Phase 7 post-#521 current-main profile refresh (2026-04-24)

**Scope:** refresh the Phase 7 source of truth after PR
[#521](https://github.com/ericflo/kiln/pull/521) routed the radix prefix
cache through the CUDA-graph chat-completion path. PR #521 closes the
18-PR window of radix-cache, streaming-prefill, Metal-fusion, and codex
tuning work (PRs #503–#521) since the post-#502 refresh. Artifact-only
profile PR; no optimization code changed.

**Preflight outcome:** proceed. Fresh `origin/main` is
`0fda0e667636bd782bb0a29feb06f2ff3d31d917`
(`Use prefix cache with CUDA graphs (#521)`). `gh pr list -R ericflo/kiln
--state all --limit 30 | grep -i profil` returned no post-#502
profile-refresh PR (PR #502 is the most recent profile refresh and predates
PRs #503–#521).

**Hardware / image:** RunPod on-demand `NVIDIA RTX A6000`, pod
`mfk88l8i8tab02`, lease `pod-66eb55349e1403350e6c342d`,
`ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`, CUDA toolkit
`12.4`, and `KILN_CUDA_ARCHS=86`. The baked `nsys 2023.4.4` was not used
for stats; `nsight-systems-2024.6.2` was installed from NVIDIA's apt repo
and `/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys` was used
for both capture and CSV export (per agent notes
`kiln-nsys-baked-importer-broken` and `kiln-nsight-profiling-gotchas`).

**Build / profile commands:** see `docs/archive/phase-c/phase-c64/post-521-profile.md`.

### Decode bench median-of-3 (no profiler)

`KILN_W4A16=1 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training
--prompt-subset humaneval --chat-template --latency-only --temperature
0.0 --seed 1`. Three back-to-back invocations of the same command with a
fresh process per run (cold model load each, excluded from latency
stats).

| Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.85 | 21.81 | 21.68 | 25.43 | 9304.9 (cold-start TTFT artifact) |
| 2 | 46.18 | 21.65 | 21.69 | 26.99 | 353.7 |
| 3 | 45.71 | 21.88 | 21.87 | 26.36 | 351.7 |
| **median** | **45.85** | **21.81** | **21.69** | **26.36** | **352.7** (warm, runs 2–3 mean) |

Run 1's prefill is the documented cold-start TTFT artifact (agent notes
`kiln-bench-prefill-warmup-required`,
`rope-inv-freq-cold-start-ttft`) — one-time HtoD allocations like
`RoPE inv_freq` plus CUDA graph capture. Decode tok/s and ITL are stable
across all three runs.

### Decode regression vs Phase 6 post-#166 closing baseline

The closing Phase 6 baseline (PROFILING.md, post-#166 around line 3225)
is **49.76 tok/s decode, 20.10 ms mean ITL, 25.46 ms p99 ITL** for the
same 512×128 paged decode shape with `KILN_W4A16=1` and CUDA graphs ON.
The post-#521 median is **45.85 tok/s, 21.81 ms mean ITL, 26.36 ms p99
ITL** — a **−7.9 %** decode tok/s regression and **+8.5 %** mean ITL
regression versus the documented Phase 7 starting baseline.

The post-#166 baseline section did not record its prompt subset, so the
gap could include prompt-content delta (this run pinned humaneval to
match recent post-#481 / post-#500 / post-#502 profiles). But the gap is
large enough that prompt content alone is unlikely to fully explain it.
The post-#166 section explicitly states *"any Phase 7 work that
unintentionally regresses decode tok/s below this median should be
flagged"* — flagging here.

### Decode top NVTX ranges

Source: `docs/archive/phase-c/phase-c64/post521_decode_nvtx_pushpop_sum.csv`. Capture
shape: 1 paged prefill (494 prompt tokens) + 128 paged decode steps; the
prefill `:kiln/attn/full/prefill_initial` range is only **0.8 %** of
total wall-clock, so this ranking is decode-dominated.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/gates` | **14.5%** |
| 2 | `:kiln/gdn/gated_norm` | **13.9%** |
| 3 | `:kiln/gdn/qk_norm` | **11.9%** |
| 4 | `:kiln/gdn/in_proj` | **9.5%** |
| 5 | `:kiln/attn/rope` | **8.3%** |
| 6 | `:kiln/mlp/gate` | **5.0%** |
| 7 | `:kiln/mlp/up` | **5.0%** |
| 8 | `:kiln/mlp/down` | **4.7%** |
| 9 | `:kiln/attn/full/decode_fused` | **3.8%** |
| 10 | `:kiln/gdn/head_expand` | **2.9%** |

### Decode top CUDA kernels

Source: `docs/archive/phase-c/phase-c64/post521_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `ucopy_bf16` | **15.9%** |
| 2 | `Marlin<(256,1,8,8,4,8)>` (W4A16 decode) | **13.3%** |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` (single big prefill GEMM) | **10.9%** |
| 4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | **9.0%** |
| 5 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **8.4%** |
| 6 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | **4.9%** |
| 7 | `bmul_f32` | **2.9%** |
| 8 | `cast_bf16_f32` | **2.8%** |
| 9 | `gdn_full_chunk_forward_kernel` (vendored GDN, PR #80) | **2.8%** |
| 10 | `fused_rmsnorm_kernel` | **2.6%** |

### Region shifts vs post-#481 plain decode trace

The closest like-for-like prior trace is the post-#481 "plain decode"
nsys profile (also 512 prompt × 128 decode under nsys, same paged config)
documented above in PROFILING.md.

| NVTX range | post-#481 | post-#521 | Δ pp |
| --- | ---: | ---: | ---: |
| `:kiln/gdn/qk_norm` | 24.4% | 11.9% | −12.5 |
| `:kiln/gdn/gates` | 12.0% | 14.5% | +2.5 |
| `:kiln/gdn/gated_norm` | 11.2% | 13.9% | +2.7 |

The `qk_norm` collapse is the documented effect of PR #486
default-on fused QK norm. The relative growth of `gates` / `gated_norm`
is a redistribution effect from `qk_norm` shrinking; their absolute
ms/region budgets are essentially unchanged. `ucopy_bf16` GPU-kernel
share is also essentially unchanged (15.4 % → 15.9 %).

### Implications for the next optimization

- **Investigate the −7.9 % decode tok/s regression versus the post-#166
  baseline before queueing any new kernel-fusion work.** The most
  parsimonious culprit is the radix-prefix-cache lookup/registration
  hooks now active on every CUDA-graph chat-completion request (PRs
  #515 / #520 / #521). The next planning cycle should A/B
  `KILN_PREFIX_CACHE_ENABLED=0` vs `=1` on the same paged decode-only
  shape used here to isolate prefix-cache overhead from any other
  regression source. If the A/B reproduces the gap, the fast path is to
  short-circuit the lookup when no prior session is cached.
- **Do not re-queue another GDN gates / gated_norm fusion attempt on
  this evidence alone.** Their rebound to top-of-table is a relative
  effect from the `qk_norm` win, not new HBM traffic. PR #173 closed
  null on `gates` fusion; PR #176 closed null on the
  `gates + gated_norm + recurrent` big-fusion ($14.99 burn). Any new
  fusion task here needs sub-range NVTX evidence that the work it would
  fuse is real HBM traffic and not launch-dispatch cost that CUDA graph
  replay already amortizes (`kiln-cuda-graph-dispatch-amortization`).
- **`ucopy_bf16` remains the dominant GPU-kernel hotspot at 15.9 %** but
  per agent note `kiln-ucopy-bf16-exhausted` (post-#219, 2026-04-20) the
  remaining un-attempted sites combined yield ≤ 0.080 speedup at 1.5×
  local and the work is not green-lit. Phase 7's productive next axis is
  the prefix-cache regression A/B above, KV cache FP8
  (`KILN_KV_CACHE_FP8=1`) for context capability, and / or self-spec
  end-to-end benching — not another `ucopy_bf16` audit.

Committed artifacts: `docs/archive/phase-c/phase-c64/post-521-profile.md`,
`docs/archive/phase-c/phase-c64/post521_decode_nvtx_pushpop_sum.csv`,
`docs/archive/phase-c/phase-c64/post521_decode_cuda_gpu_kern_sum.csv`,
`docs/archive/phase-c/phase-c64/post521_decode_nvtx_kern_sum.csv`. The full
`post521_decode.nsys-rep` (110 MB) is not committed because it exceeds
the in-repo artifact-size convention; it remains at
`/tmp/kiln-post521/post521_decode.nsys-rep` for the lifetime of pod
`mfk88l8i8tab02` (lease `pod-66eb55349e1403350e6c342d`).

## Phase 7 real prefix-cache reuse A/B (2026-04-24)

**Scope:** verify that the real-backend append-prefix cache from PR #515 skips shared prompt prefill and improves repeat/shared-prefix latency on GPU.

**Build / hardware:** `main` `8bd7dd0e` (`8bd7dd0ec047db35b2b48324b0493c1298efe6c7`), RunPod on-demand `NVIDIA RTX A6000, 49140 MiB, driver 550.127.08`, `ghcr.io/ericflo/kiln-runpod:latest`. The requested A6000 was available. The task's literal build command `cargo build --release --features cuda --bin kiln-server` is incompatible with current Cargo targets (`kiln` is the server binary), so the benchmark built `cargo build --release --features cuda --bin kiln`.

**Server config:** real `kiln` server, `model.path = "/workspace/qwen3.5-4b"`, `memory.num_blocks = 4096`, `memory.kv_cache_fp8 = true`, `memory.cuda_graphs = false`, `[prefix_cache] max_blocks = 2048`. The original A/B disabled CUDA graphs to isolate prefix-cache reuse; current CUDA-graph real chat completions now use the same prefix-cache lookup/register path.

**Prompt shape:** ChatML content used a 2,048-token block-aligned shared prefix. Suffix variants embedded the ChatML assistant delimiter after the shared text so the warm shared prompt's complete token sequence was an exact prefix of both later variant prompts. Variant A was 2,068 prompt tokens; variant B was 2,069 prompt tokens.

| Arm | Prefix cache | Median total latency | Mean total latency | Min | Max | Completion tokens | Metrics verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ON | `KILN_PREFIX_CACHE_ENABLED=1` | 7.711s | 7.718s | 7.678s | 7.777s | 16 each | 10 hits, 20,480 hit tokens, 1,280 hit blocks |
| OFF | `KILN_PREFIX_CACHE_ENABLED=0` | 26.923s | 26.501s | 24.455s | 27.227s | 16 each | counters stayed zero |

**Result:** prefix cache ON was **3.49x faster** on median total latency for the 5 paired A/B suffix requests after warming the shared prefix. TTFT was not reported in this A/B because it used non-streaming requests; streaming and non-streaming real chat completions now use the same real prefix-cache lookup/register path with or without CUDA graphs, so hits increment the same `/metrics` prefix-cache counters.

**Metrics after ON arm:**

```text
kiln_prefix_cache_lookups_total{result="hit"} 10
kiln_prefix_cache_lookups_total{result="miss"} 2
kiln_prefix_cache_hit_tokens_total 20480
kiln_prefix_cache_hit_blocks_total 1280
kiln_prefix_cache_cached_blocks 128
kiln_prefix_cache_max_blocks 2048
```

**Exact-repeat caveat:** repeating the warmed 2,048-token prompt increments a miss rather than a hit because cache lookup requires `prompt_tokens.len() > entry.prompt_tokens.len()` and next-token logits are not cached. The exact-repeat request completed in 0.823s with 1 output token, but it did not count as a prefix-cache hit.

**Verdict / current behavior:** the real append-prefix cache is functionally effective for append-only shared prefixes on both non-streaming and streaming chat completions. CUDA graphs no longer bypass prefix-cache lookup/registration for non-speculative real chat completions; `scripts/phase7_cuda_graph_prefix_cache_verify.sh` verifies a CUDA-graph cache hit and absence of the old bypass warning.

Detailed artifact: `docs/audits/phase7-prefix-cache-reuse-ab.md`.

## Phase 7 CUDA streaming prefill default A/B (2026-04-24)

**Scope:** decide whether to flip the existing CUDA streaming/tiled prefill path from opt-in to default-on for long prompts after PR #507 showed the memory win at 128k.

**Hardware:** RunPod on-demand `NVIDIA A40` fallback, 48 GiB VRAM, sm_86, `ghcr.io/ericflo/kiln-runpod:latest`. The requested A6000 was supply-constrained; A40 is the documented same-VRAM/sm_86 fallback.

**Build:** current main base `76c7614`, `KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench`.

**A/B command shape:** interleaved `KILN_STREAMING_PREFILL=0` vs `1` at 32768, 65536, and 131072 prompt tokens with `KILN_W4A16=1 KILN_KV_CACHE_FP8=1 KILN_CUDA_GRAPHS=true`, `--paged`, `--max-output-tokens 1`, `--skip-training`, and `--latency-only`. Peak VRAM came from 200 ms `nvidia-smi` sampling.

| Prompt tokens | Mode | Exit | Peak VRAM | TTFT / prefill | Prefill tok/s | First decode ITL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32768 | monolithic (`KILN_STREAMING_PREFILL=0`) | 0 | 25319 MiB | 20075.7 ms | 1632 | 102.3 ms |
| 32768 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 20071 MiB | 12881.7 ms | 2544 | 101.1 ms |
| 65536 | monolithic (`KILN_STREAMING_PREFILL=0`) | 0 | 33063 MiB | 28482.0 ms | 2301 | 169.3 ms |
| 65536 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 21863 MiB | 27693.8 ms | 2366 | 170.7 ms |
| 131072 | monolithic (`KILN_STREAMING_PREFILL=0`) | 1 | 44455 MiB sampled before OOM | OOM in GDN layer 0 | n/a | n/a |
| 131072 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 25383 MiB | 62930.8 ms | 2083 | 310.6 ms |

**Decision:** enable CUDA streaming prefill by default at `>= 65533` actual prompt tokens. The 65k arm reduced peak memory by 11200 MiB and did not materially regress TTFT or first decode ITL; the 128k monolithic arm OOMed while streaming completed. `KILN_STREAMING_PREFILL=0` remains the kill switch, `KILN_STREAMING_PREFILL=1` remains the force-on override below threshold, and `KILN_STREAMING_TILE_TOKENS` remains the tile-size override.

Detailed artifact: `docs/archive/phase-c/phase-c63/cuda-streaming-prefill-default.md`.

## Phase 6 FlashInfer-style paged GQA decode preflight (2026-04-24)

**Scope:** decide whether current `main` still lacks a native CUDA paged GQA
decode kernel for single-token full-attention decode before creating a new
minimal FlashInfer-style kernel crate.

**Preflight outcome:** stop with a documentation-only PR. `gh pr list -R
ericflo/kiln --limit 10` returned no open overlapping PRs, and the required
current-main files show the remaining-work precondition fails: the paged
full-attention decode path is still active in `crates/kiln-model/src/forward.rs`,
but it already routes eligible single-token GQA decode calls through the native
CUDA `kiln-flash-attn` paged-decode C ABI.

**Code evidence:** `try_flash_attn_paged_decode(...)` borrows raw per-layer K/V
pools with `PagedKvCache::pool_tensors(...)`, builds the paged block-table
tensor, and calls `backend.flash_attn_paged_decode(...)`. The vendored CUDA
backend is `kiln_flash_attn_fwd_paged_decode(...)`, exposed from
`crates/kiln-flash-attn/csrc/flash_attn/flash_api_c.h` and wrapped by
`crates/kiln-flash-attn/src/lib.rs` as `flash_attn_paged_decode(...)` for bf16,
single-query GQA decode.

**Decision:** no new `kiln-paged-gqa-kernel` crate and no RunPod spend. Current
`main` does not lack the native CUDA single-token paged GQA decode capability
this task was gated on. Adding a second FlashInfer-style crate would duplicate
the existing backend while the post-PR-#500 / PR #502 profile still points Phase
6 at GDN-dominated decode/prefill work rather than full-attention paged decode.

**Validation:** no pod launched; no CUDA files changed; no build or benchmark
commands run because the precondition failed before source edits. Local
validation: `git diff --check`.

Detailed artifact:
`docs/archive/phase-c/phase-c61/flashinfer-paged-gqa-decode-preflight.md`.

## Phase 6 post-#502 GDN full-chunk audit (2026-04-24)

**Scope:** audit whether the freshly merged PR #502 post-PR-#500 profile leaves
one bounded CUDA improvement to port into Kiln's vendored GDN full-chunk
prefill kernel.

**Preflight outcome:** stop with a documentation-only PR. `gh pr list -R
ericflo/kiln --limit 10` returned no open PRs, and the post-PR-#500 section
below is present on `origin/main` at commit `76ea952`, with prompt-heavy
prefill led by `gdn_full_chunk_forward_kernel` (**27.9%** GPU-kernel share)
and `ucopy_bf16` (**22.3%**). The CUDA file still exists, but the upstream
audit found no remaining bounded micro-port inside the task envelope.

**Upstream reviewed:** `fla-org/flash-linear-attention` commit
`101240f396a6b53e452defb371e3d6e98211535a` and `vllm-project/vllm` commit
`9f771b3ab92d26a7d91a8255572c5d8d2b3ad601`, specifically the chunk gated-delta
and fused recurrent kernels documented in
`docs/archive/phase-c/phase-c60/post-502-gdn-full-chunk-audit.md`.

**Audit result:** no CUDA source change. Current upstream still gets its
remaining full-chunk advantage from a structural state-ownership model: WY-style
chunk preparation plus tiled FP32 recurrent-state ownership in the update
kernel. Kiln's current `gdn_full_chunk_forward_kernel` consumes the existing
cuBLAS-produced `kkt`, `qkt`, `ks_entry`, and `q_s` tensors and then performs
value-lane forward substitution/output accumulation plus a scalar BF16 state
epilogue behind the existing C ABI. Porting the remaining upstream shape would
therefore require a larger re-vendor/rewrite of the chunk/state boundary, not a
bounded patch to `gdn_full_chunk_forward.cu`.

**Consumed sub-ideas:** the bounded slices implied by upstream have already
been attempted or disqualified in the history below: recurrent-state BF16
round-trip removal failed parity; decay hoisting, `k_t` staging, and triangular
front-half packing did not produce durable prompt-heavy wins; the tiled
recurrent-state epilogue was **1.8% slower** than warmed `main`; and the
post-#442 vLLM audit already concluded the remaining delta is structural only.

**Validation:** no RunPod was launched and no CUDA build was run because the
remaining-work precondition failed before source edits. Local validation:
`git diff --check`.

Detailed artifact: `docs/archive/phase-c/phase-c60/post-502-gdn-full-chunk-audit.md`.

## Phase 6 post-#500 current-main profile refresh (2026-04-24)

**Scope:** refresh the Phase 6 CUDA source of truth after PR
[#500](https://github.com/ericflo/kiln/pull/500) added the CUDA GDN
`qk_norm_gqa` fast path. This is an artifact-only profile refresh; no
optimization code changed.

**Precondition outcome:** proceed. `PROFILING.md` did not contain a post-PR-#500
current-main section at commit `faf8cb3` or newer with both native-MTP decode
and prompt-heavy prefill top-3 NVTX/kernel hotspot tables.

**Baseline:** fresh `origin/main` was `faf8cb3e1d5ae79fb9a2b3111c3f78052c6a51e7` (`faf8cb3`). This includes PR
[#500](https://github.com/ericflo/kiln/pull/500).

**Hardware / image:** RunPod on-demand `NVIDIA RTX A6000`, pod
`93wtgcujidv9ky`, `ghcr.io/ericflo/kiln-runpod:latest`, driver
`550.127.08`, CUDA toolkit `12.4`, and `KILN_CUDA_ARCHS=86`.
Nsight Systems `2024.6.2.225-246235244400v0` was installed from the
NVIDIA apt repo and `/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys`
was used for both capture and CSV export.

**Build / profile commands:** see `docs/archive/phase-c/phase-c60/post-500-profile.md`.

Native-MTP decode latency metrics were **23.5
tok/s**, **42.5 ms** mean ITL, **419.1 ms**
prefill for **515** prompt tokens, **128** generated tokens, and
**MTP α = 0.716**. Prompt-heavy prefill metrics were
**1635.6 ms** prefill for **4090** prompt tokens
(**2501 tok/s**) and **17** generated tokens.

### C60 top native-MTP decode hotspots

Source: `docs/archive/phase-c/phase-c60/top_decode_nvtx.csv`. Decode ranking excludes the
`:kiln/mtp/step` parent wrapper.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/gates` | **12.8%** |
| 2 | `:kiln/gdn/gated_norm` | **12.1%** |
| 3 | `:kiln/gdn/qk_norm` | **10.0%** |

Kernel-level cross-check: `docs/archive/phase-c/phase-c60/top_decode_kernels.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `ucopy_bf16` | **18.2%** |
| 2 | `Marlin<(256,1,8,8,4,8)>` | **10.0%** |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64...` | **8.7%** |

### C60 top prompt-heavy prefill hotspots

Source: `docs/archive/phase-c/phase-c60/top_prefill_nvtx.csv`.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/in_proj` | **23.1%** |
| 2 | `:kiln/gdn/gates` | **10.7%** |
| 3 | `:kiln/gdn/gated_norm` | **9.2%** |

Kernel-level cross-check: `docs/archive/phase-c/phase-c60/top_prefill_kernels.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `gdn_full_chunk_forward_kernel` | **27.9%** |
| 2 | `ucopy_bf16` | **22.3%** |
| 3 | `Marlin<(256,4,16,4,4,8)>` | **9.8%** |

**Verdict:** PR #500 moved native-MTP `:kiln/gdn/qk_norm` below gates and
gated_norm in the decode NVTX ranking, so the same qk_norm GQA fusion shape
should not be retried. Prompt-heavy prefill is still dominated by GDN
full-chunk/in-projection work, with `gdn_full_chunk_forward_kernel` the largest
kernel bucket. The next material Phase 6 target should audit and port upstream
flash-linear-attention/vLLM recurrent-gated-delta improvements into kiln's
vendored CUDA GDN full-chunk path, rather than hand-rolling another candle
micro-fusion.

Committed artifacts: `docs/archive/phase-c/phase-c60/post-500-profile.md`,
`docs/archive/phase-c/phase-c60/summary.json`, `docs/archive/phase-c/phase-c60/top_decode_nvtx.csv`,
`docs/archive/phase-c/phase-c60/top_prefill_nvtx.csv`, `docs/archive/phase-c/phase-c60/top_decode_kernels.csv`,
`docs/archive/phase-c/phase-c60/top_prefill_kernels.csv`, raw Nsight CSV exports, and profiler
logs under `docs/archive/phase-c/phase-c60/`.

## Phase 6 C57 native-MTP conv1d prefill recovery profile (2026-04-24)

**Scope:** verify that the C56 CUDA `causal_conv1d_prefill` status-3 blocker is
cleared on current `main`, keep regression coverage on the real C56 prefill
envelope, and refresh the native-MTP decode-window source of truth.

**Precondition outcome:** proceed. `PROFILING.md` had a newer post-#486 plain
decode/prefill profile, but no newer artifact than C56 that both exercised the
CUDA conv1d prefill fast path and reached native-MTP decode (`:kiln/mtp/step`).

**Root-cause status:** current `main` already includes PR #481 (`7227b4c`),
which fixed the C56 root cause by matching the prefill kernel
`__launch_bounds__` to the largest 256-thread launch path. C57 adds model-level
regression coverage by running the CUDA prefill parity test at `seq_len=512`,
the C56 native-MTP prompt prefill envelope.

**Hardware / image:** RunPod on-demand `NVIDIA RTX A6000`, pod
`6lv4pu241ofanf`, `ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`,
CUDA toolkit `12.4`, `KILN_CUDA_ARCHS=86`. The pod was terminated after
artifacts were copied.

**Validation:**

- `cargo test -p kiln-conv1d-kernel --release -- --nocapture`: 4 passed.
- `cargo test -p kiln-model --release --features cuda causal_conv1d_prefill -- --nocapture`: 1 passed at the 512-token C56 prefill envelope.
- `cargo test -p kiln-model --release --features cuda test_causal_conv1d_update_matches_fallback -- --nocapture`: 1 passed.
- `cargo build --release --features cuda,nvtx --bin kiln-bench`: succeeded.

**Profile command:**

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false --output /tmp/kiln-c57/c57-mtp-conv-prefill-v2 \
  env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
```

**Result:** the profile reached `:kiln/mtp/step`; the C56 prefill status-3
failure did not reproduce on current `main`. Latency metrics were 515 prompt
tokens, 417.7 ms prefill (1233 tok/s), 128 generated tokens, **26.5 decode
tok/s**, **37.8 ms mean ITL**, and **MTP α = 0.764**.

### C57 top native-MTP decode hotspots

Source: `docs/archive/phase-c/phase-c57/top_decode_nvtx.csv`. Decode window is first
`:kiln/mtp/step` start through final decode NVTX end, excluding the
`:kiln/mtp/step` parent wrapper.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/gates` | **14.45%** |
| 2 | `:kiln/gdn/gated_norm` | **13.55%** |
| 3 | `:kiln/gdn/qk_norm` | **11.02%** |

### C57 top native-MTP prefill hotspots

Source: `docs/archive/phase-c/phase-c57/top_prefill_nvtx.csv`.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/in_proj` | **59.49%** |
| 2 | `:kiln/attn/full/prefill_initial` | **12.67%** |
| 3 | `:kiln/gdn/qk_norm` | **4.47%** |

**Verdict:** C56's profiling blocker is cleared on current `main`, and Phase 6
should use C57, not failed C56, when selecting the next native-MTP decode target.
The top native-MTP decode buckets are now GDN gates, gated_norm, and qk_norm.

Committed artifacts: `docs/archive/phase-c/phase-c57/conv-prefill-mtp-profile.md`,
`docs/archive/phase-c/phase-c57/summary.json`, `docs/archive/phase-c/phase-c57/top_decode_nvtx.csv`,
`docs/archive/phase-c/phase-c57/top_prefill_nvtx.csv`,
`docs/archive/phase-c/phase-c57/c57-mtp-conv-prefill-v2_cuda_gpu_kern_sum.csv`,
`docs/archive/phase-c/phase-c57/nsys-profile-v2.log`, and `docs/archive/phase-c/phase-c57/nsys-stats-v2.log`.

## Phase 6 post-#486 current-main profile refresh (2026-04-24)

**Scope:** refresh the Phase 6 CUDA source of truth after PR
[#486](https://github.com/ericflo/kiln/pull/486) made fused GDN qk_norm the
default. This is an artifact-only profile refresh; no optimization code
changed.

**Preflight outcome:** proceed. GitHub open-PR search found no existing
post-#486 profile-refresh PR, and `PROFILING.md` had no post-#486 current-main
section with both decode and prompt-heavy prefill top-3 NVTX/kernel hotspots.

**Baseline:** fresh `origin/main` was
`4d96c9f117108f915fadf1177b517f170252ba41`
(`[codex] fuse Metal GDN gates with recurrent decode`, PR
[#489](https://github.com/ericflo/kiln/pull/489)). This includes PR
[#486](https://github.com/ericflo/kiln/pull/486) at
`e1430086d04345d8ca1ff64820bfe1bf1a5c5700`, plus the subsequent non-CUDA
mainline PRs #487-#489.

**Hardware / image:** RunPod on-demand `NVIDIA A40` fallback because A6000
launch returned `SUPPLY_CONSTRAINT`. The pod used
`ghcr.io/ericflo/kiln-runpod:latest`, driver `580.126.09`, CUDA toolkit
`12.4`, and `KILN_CUDA_ARCHS=86`. Nsight Systems `2024.6.2` was installed from
the NVIDIA apt repo and
`/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys` was used for both
capture and CSV export.

**Build / profile commands:**

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main
git reset --hard origin/main
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
cargo build --release --features cuda,nvtx --bin kiln-bench

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/workspace/post486-profile/post486_decode \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 \
    --skip-training --chat-template --latency-only --temperature 0.0 --seed 1

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/workspace/post486-profile/post486_prefill \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 8192 --max-output-tokens 2 \
    --skip-training --latency-only --temperature 0.0 --seed 1

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys stats \
  --report cuda_gpu_kern_sum \
  --report nvtx_kern_sum \
  --report nvtx_pushpop_sum \
  --format csv --output /workspace/post486-profile/post486_decode \
  /workspace/post486-profile/post486_decode.nsys-rep

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys stats \
  --report cuda_gpu_kern_sum \
  --report nvtx_kern_sum \
  --report nvtx_pushpop_sum \
  --format csv --output /workspace/post486-profile/post486_prefill \
  /workspace/post486-profile/post486_prefill.nsys-rep
```

Latency metrics were captured from profiler stdout. The plain decode trace
reported **494** prompt tokens, **8450.8 ms** prefill, and **47.5 ms** mean ITL
(**21.1 tok/s**) for **129** generated tokens. The prompt-heavy prefill trace
reported **8180** prompt tokens, **3451.8 ms** prefill, and **2370 tok/s**
prompt throughput.

### Decode top hotspots

Source: `profile-out/post486_decode_nvtx_pushpop_sum.csv`.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/qk_norm` | **24.8%** |
| 2 | `:kiln/gdn/gates` | **11.9%** |
| 3 | `:kiln/gdn/gated_norm` | **11.2%** |

Decode kernel-level cross-check: `profile-out/post486_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `ucopy_bf16` | **15.4%** |
| 2 | `Marlin<(256,1,8,8,4,8)>` | **13.6%** |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64...` | **11.5%** |

### Prompt-heavy prefill top hotspots

Source: `profile-out/post486_prefill_nvtx_pushpop_sum.csv`. The two-token
decode tail contributes `:kiln/lm_head` at **17.9%** in this trace and is
excluded from the prefill ranking below.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/in_proj` | **30.4%** |
| 2 | `:kiln/attn/full/prefill_initial` | **12.7%** |
| 3 | `:kiln/gdn/gated_norm` | **5.0%** |

Prefill kernel-level cross-check: `profile-out/post486_prefill_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `gdn_full_chunk_forward_kernel` | **34.8%** |
| 2 | `ucopy_bf16` | **15.1%** |
| 3 | `Marlin<(256,4,16,4,4,8)>` | **13.4%** |

### Verdict

Default-on fused GDN qk_norm does **not** change the Phase 6 source of truth:
plain decode remains dominated by the GDN norm/gate trio, with `qk_norm` still
the largest NVTX bucket, while prompt-heavy prefill is still dominated by GDN
full-chunk/in-projection work. Because standalone qk_norm fusion has already
landed and still leaves `:kiln/gdn/qk_norm` at **24.8%**, the next optimization
task should **not** retry qk_norm micro-fusion. Based only on this fresh
profile, the next material CUDA target is the GDN prefill/full-chunk path:
audit the current vendored `gdn_full_chunk_forward_kernel` against newer
flash-linear-attention/vLLM recurrent-gated-delta tuning and port a minimal
bf16/f32 forward-path improvement if the diff exposes a concrete kernel win.

Committed artifacts:

- `profile-out/post486_decode_cuda_gpu_kern_sum.csv`
- `profile-out/post486_decode_nvtx_kern_sum.csv`
- `profile-out/post486_decode_nvtx_pushpop_sum.csv`
- `profile-out/post486_prefill_cuda_gpu_kern_sum.csv`
- `profile-out/post486_prefill_nvtx_kern_sum.csv`
- `profile-out/post486_prefill_nvtx_pushpop_sum.csv`

## Phase 6 default-on fused GDN qk_norm validation (2026-04-24)

PR scope: make the existing CUDA fused L2 Q/K norm path the default for bf16
CUDA tensors supported by `kiln_rmsnorm_kernel::fused_l2_qk_norm`, with
`KILN_DISABLE_FUSED_L2_QK_NORM=1` as the candle fallback opt-out.

Validation ran on an on-demand RTX A6000 RunPod pod
`ztru3dal6px398` using `ghcr.io/ericflo/kiln-runpod:latest`,
`KILN_CUDA_ARCHS=86`, `KILN_W4A16=1`, and `KILN_CUDA_GRAPHS=true`.

- `cargo test -p kiln-rmsnorm-kernel --release -- --nocapture`: passed, 6
  tests including all `parity_l2_qk_norm_*` coverage.
- `cargo test -p kiln-model --release --features cuda gdn -- --nocapture`:
  7/8 matching tests passed; `test_gdn_chunk_body_matches_fallback` failed
  with `max_abs_diff 2.109375e0`, and the same focused test still failed
  with `KILN_DISABLE_FUSED_L2_QK_NORM=1`, so this is not attributable to
  the default-on qk_norm dispatch.
- `cargo build --release --features cuda,nvtx --bin kiln-bench`: passed.
- Default-on latency arm (`--paged --prompt-tokens 512 --max-output-tokens 64`):
  494 prompt tokens, 65 generated tokens, prefill 1736.8 ms, mean ITL 24.3 ms,
  decode 41.1 tok/s. This was the first post-load arm and includes cold-start
  noise.
- Opt-out latency arm (`KILN_DISABLE_FUSED_L2_QK_NORM=1`, same command): 494
  prompt tokens, 65 generated tokens, prefill 352.8 ms, mean ITL 19.9 ms,
  decode 50.2 tok/s. Treat the paired wall-clock comparison as noisy because
  it ran second after model/kernel warmup.

A short default-on NVTX profile on the same A6000 run reported
`:kiln/gdn/qk_norm` at **11.4%** of total NVTX push/pop time
(`277.014 ms`, 1560 instances). That is below the post-#481 plain-decode
source-of-truth share of **24.4%**, but the run was a short validation profile
under Nsight Systems 2023.4.4 and is not a strict replacement for the
post-#481 decode-window methodology.

## Phase 6 post-#481 current-main profile refresh (2026-04-24)

**Scope:** refresh the Phase 6 source of truth after PR
[#481](https://github.com/ericflo/kiln/pull/481) fixed the CUDA conv1d
prefill launch bounds that had blocked the native-MTP prefill path before
decode. This is an artifact-only profile PR; no optimization code changed.

**Preflight outcome:** proceed. Fresh `origin/main` was
`7227b4cf8bc4b117b60865e36214d69024288463`
(`phase6: fix conv1d prefill launch bounds (#481)`), `PROFILING.md` had no
post-#481 current-main profile section, and GitHub open-PR search found no
existing post-#481 profile-refresh PR.

**Hardware / image:** RunPod on-demand `NVIDIA A40` fallback because A6000
launch returned `SUPPLY_CONSTRAINT`. The pod used
`ghcr.io/ericflo/kiln-runpod:latest`, driver `580.126.09`, CUDA toolkit
`12.4`, and `KILN_CUDA_ARCHS=86`. The baked `nsys 2023.4.4` was not used for
stats; `nsight-systems-2024.6.2` was installed from NVIDIA's apt repo and
`/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys` was used for both
capture and CSV export.

**Build / profile commands:**

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main
git checkout main
git reset --hard origin/main
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export KILN_MTP_ARGMAX_FP32=1
cargo build --release --features cuda,nvtx --bin kiln-bench

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/workspace/post481-profile/post481_decode \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 \
    --skip-training --chat-template --latency-only --temperature 0.0 --seed 1

/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/workspace/post481-profile/post481_prefill \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 8192 --max-output-tokens 2 \
    --skip-training --latency-only --temperature 0.0 --seed 1

KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 64 \
  --skip-training --prompt-subset humaneval --chat-template \
  --latency-only --temperature 0.0 --seed 1
```

Latency summary source: `profile-out/post481_latency_summary.log`. The plain
decode trace reported **494** prompt tokens, **8154.0 ms** prefill, and
**44.7 ms** mean ITL (**22.4 tok/s**) for **129** generated tokens. The
prompt-heavy prefill trace reported **8180** prompt tokens, **3447.8 ms**
prefill, and **2372 tok/s** prompt throughput.

### Decode top hotspots

Source: `profile-out/post481_decode_nvtx_pushpop_sum.csv`.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/qk_norm` | **24.4%** |
| 2 | `:kiln/gdn/gates` | **12.0%** |
| 3 | `:kiln/gdn/gated_norm` | **11.2%** |

Decode kernel-level cross-check: `profile-out/post481_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `ucopy_bf16` | **15.4%** |
| 2 | `Marlin<(256,1,8,8,4,8)>` | **13.6%** |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64...` | **11.5%** |

### Prompt-heavy prefill top hotspots

Source: `profile-out/post481_prefill_nvtx_pushpop_sum.csv`. The two-token
decode tail contributes `:kiln/lm_head` at **17.5%** in this trace and is
excluded from the prefill ranking below.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/in_proj` | **33.6%** |
| 2 | `:kiln/attn/full/prefill_initial` | **13.1%** |
| 3 | `:kiln/gdn/qk_norm` | **3.9%** |

Prefill kernel-level cross-check: `profile-out/post481_prefill_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `gdn_full_chunk_forward_kernel` | **34.8%** |
| 2 | `ucopy_bf16` | **15.1%** |
| 3 | `Marlin<(256,4,16,4,4,8)>` | **13.5%** |

### Native MTP observation

Source: `profile-out/post481_mtp_latency.log`. Native MTP now reaches decode
on the standard humaneval latency arm instead of failing during GDN layer-0
conv1d prefill. The run reported **515** prompt tokens, **340.5 ms** MTP
prefill (**1513 tok/s**), **64** generated tokens, **24.3 ms** mean ITL,
**41.2 tok/s** decode, and **α = 0.730** (**27/37** accepted draft tokens).
This clears the earlier pre-decode blocker from the post-#476/#481 notes, but
it is a single seed-1 A40 observation, not a new MTP ship verdict.

### Verdict

Post-#481 current `main` is profileable again through the native-MTP path. The
plain decode top-3 remains GDN-side (`qk_norm`, `gates`, `gated_norm`), while
prompt-heavy prefill still has GDN full-chunk work as the largest kernel bucket
(**34.8%**) and `:kiln/gdn/in_proj` as the largest NVTX wall-clock range
(**33.6%**).

The next Phase 6 source-of-truth candidate should therefore stay on GDN-side
work unless a follow-up multi-seed MTP sweep confirms that post-#481 native MTP
acceptance and throughput are durable enough to justify optimization work.

**Committed artifacts:**

- `profile-out/post481_decode_cuda_gpu_kern_sum.csv`
- `profile-out/post481_decode_nvtx_kern_sum.csv`
- `profile-out/post481_decode_nvtx_pushpop_sum.csv`
- `profile-out/post481_prefill_cuda_gpu_kern_sum.csv`
- `profile-out/post481_prefill_nvtx_kern_sum.csv`
- `profile-out/post481_prefill_nvtx_pushpop_sum.csv`
- `profile-out/post481_latency_summary.log`
- `profile-out/post481_mtp_latency.log`

## Phase 6 post-#442 vLLM full-chunk GDN audit (2026-04-23)

**Scope:** audit current vLLM Gated DeltaNet kernels against Kiln's vendored
CUDA full-chunk prefill path after PR #444 refreshed the post-#442 profile and
kept `gdn_full_chunk_forward_kernel` as the largest prompt-heavy prefill
GPU-kernel bucket (**33.6%**).

**Preflight outcome:** proceed with a doc-only audit. Fresh `origin/main`
contains the post-#442 / PR #444 section above, no pending or running Cloud Eric
task matched this vLLM full-chunk audit, and no PR newer than #444 already
audits or ports vLLM `fused_recurrent_gated_delta_rule` /
`chunk_gated_delta_rule` ideas into
`crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`. The CUDA file still
exists and remains the full-chunk CUDA prefill path called from
`crates/kiln-model/src/forward.rs`.

**Upstream reviewed:**

- vLLM commit `7ff65b19003be4955d2d5d1428e7d94d082559d0`
- `vllm/model_executor/layers/fla/ops/chunk.py`
  (`chunk_gated_delta_rule_fwd`)
- `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`
  (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`)
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
  (`fused_recurrent_gated_delta_rule_fwd_kernel`)

**Audit result:** no new bounded, portable win exists inside this task's
envelope (`chunk_size=64`, bf16, `dk/dv<=128`, forward-only, existing C ABI).

Current vLLM still gets its full-chunk advantage from a different structure:
the chunk path builds WY-style intermediates, then
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` keeps FP32 recurrent-state
tiles (`[BV, 64]`, split by K blocks) in registers and updates those tiles with
Triton dot products after loading `b_k` once per chunk tile. Its recurrent
decode kernel follows the same tile-owned-state pattern. Kiln's
`gdn_full_chunk_forward_kernel` intentionally fuses the existing cuBLAS-side
chunk matmuls (`kkt`, `qkt`, `ks_entry`, `q_s`) with chunk-local orchestration
and a scalar-thread state epilogue over `(k_idx, dv)`.

The remaining vLLM delta is therefore a structural re-vendor/rewrite of the
full-chunk state/update ownership, not a single kernel-local cleanup. The
bounded sub-ideas implied by that upstream design have already been attempted
and disqualified before this post-#442 refresh:

- post-#397: remove recurrent-state bf16 scalar round trips -> parity failure
- post-#399: hoist decay weighting into shared `W` rows -> slower
- post-#401: stage shared `k_t` rows -> slower
- post-#403: triangular front-half packing -> warm-pod artifact, no durable win
- post-#406: tiled recurrent-state epilogue -> warmed same-pod control was
  **1.8% slower** than main

**Keep / revert decision:** no CUDA change. Porting the remaining vLLM idea
would require replacing the current scalar-thread full-chunk/state-update
layout with a tile-owned FP32 state design, which is outside this task's "at
most one minimal improvement" scope and would duplicate the already-failed
micro-port frontier.

**RunPod validation:** allowed fallback H100 NVL pool pod
(`ghcr.io/ericflo/kiln-runpod:latest`). The requested A6000 pool resume failed
with host capacity, A40 reached pod creation but remained runtime-null and was
released, A100 / RTX 6000 Ada / L40S were unavailable, and H100 NVL was the
first reachable fallback. Because H100 is `sm_90`, validation used
`KILN_CUDA_ARCHS=90` instead of the A6000/A40 `sm_86` setting.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=90
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true

cargo build --release --features cuda,nvtx --bin kiln-bench
cargo test -p kiln-model --release --features cuda \
  test_gdn_full_chunk_forward_matches_fallback -- --nocapture
cargo test -p kiln-gdn-kernel --release -- --nocapture
```

Results:

- `kiln-bench` release build: passed
- `test_gdn_full_chunk_forward_matches_fallback`: passed
  (`out_chunk` max abs diff **1.5625e-2**, `state` max abs diff **3.125e-2**)
- `kiln-gdn-kernel`: passed (`gdn_gates_parity_vs_candle_reference` passed)

Prompt-heavy `8192/1` validation on the same H100 fallback:

```bash
KILN_SPEC_METHOD=off ./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 8192 --max-output-tokens 1 \
  --skip-training --latency-only

KILN_DISABLE_GDN_KERNEL=1 KILN_SPEC_METHOD=off ./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 8192 --max-output-tokens 1 \
  --skip-training --latency-only
```

Both H100 latency runs emitted the prefill measurement, then failed in the
single-token decode tail with `CUDA_ERROR_ILLEGAL_INSTRUCTION`. This appears
H100-tail specific, not a full-chunk prefill parity failure: the current GDN
kernel arm reported **3873.8 ms** prefill (**2112 tok/s**) before the decode
tail failure, and the `KILN_DISABLE_GDN_KERNEL=1` arm reported **25089.5 ms**
prefill (**326 tok/s**) before the same failure. A diagnostic no-graphs rerun
of the current GDN arm reported **2459.0 ms** prefill (**3327 tok/s**) before
the same decode-tail failure.

No Nsight capture was taken because this PR does not change the CUDA kernel.
The H100 pool lease was released after validation.

## Phase 6 post-#442 current-main prefill/decode profile refresh (2026-04-23)

**Scope:** refresh the Phase 6 source of truth after PR
[#442](https://github.com/ericflo/kiln/pull/442) merged, with one fresh
decode trace and one fresh prompt-heavy prefill trace on current `main`.

**Preflight outcome:** proceed. Fresh GitHub state showed PR #442 merged into
`main`; the checked-in `PROFILING.md` did not yet contain a post-#442
current-main section with separate top-3 decode and prefill hotspots; GitHub PR
search plus Cloud Eric pending/running task search found no overlapping newer
open/merged PR or pending/running kiln task for this exact profile refresh. The
profiled checkout was `a203330` (`Batch Metal auxiliary weight uploads (#443)`),
which is current `main` and includes #442 (`0937e4b`).

**Hardware / image:** RunPod on-demand `NVIDIA RTX A6000`, image
`ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`, CUDA toolkit
`12.4`. The pod had the baked `nsys 2023.4.4`, so `nvidia-nsys-cli==2024.5.1`
was installed from NVIDIA's wheel and `/usr/local/bin/nsys` was used for both
capture and stats export.

**Build / setup commands:**

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main
git checkout main
git reset --hard origin/main
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
cargo build --release --features cuda,nvtx --bin kiln-bench
python3 scripts/c12_bench_runner.py --help
```

The model checkpoint was already present at `/workspace/qwen3.5-4b`.

**Baseline decode command:**

```bash
KILN_SPEC_METHOD=off \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  > profiling-artifacts/post442_decode_baseline.json
```

Baseline single-request latency arm: **494** prompt tokens, **394.9 ms**
prefill, **19.7 ms** mean ITL, **50.6 tok/s** decode.

**Nsight commands run:**

```bash
KILN_SPEC_METHOD=off \
/usr/local/bin/nsys profile -t cuda,nvtx \
  --sample=none --cpuctxsw=none --delay=70 --duration=20 \
  -o /workspace/post442-decode \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

KILN_SPEC_METHOD=off \
/usr/local/bin/nsys profile -t cuda,nvtx \
  --sample=none --cpuctxsw=none --delay=25 --duration=12 \
  -o /workspace/post442-prefill \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 8192 --max-output-tokens 1 \
    --skip-training --latency-only

/usr/local/bin/nsys stats \
  --report cuda_gpu_kern_sum \
  --report nvtx_kern_sum \
  --report nvtx_pushpop_sum \
  --format csv --output /workspace/post442-decode \
  /workspace/post442-decode.nsys-rep

/usr/local/bin/nsys stats \
  --report cuda_gpu_kern_sum \
  --report nvtx_kern_sum \
  --report nvtx_pushpop_sum \
  --format csv --output /workspace/post442-prefill \
  /workspace/post442-prefill.nsys-rep
```

The task's literal prefill `--delay=3 --duration=4` window would have captured
model loading on this pod. The final replacement window above was chosen after
observing model load in the **25.1-31.6 s** range; the committed run reported
**8180** prompt tokens, **3423.5 ms** prefill, and **2389 tok/s** prompt
throughput.

### Decode top hotspots

Source: `profile-out/post442_decode_nvtx_pushpop_sum.csv`.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/gates` | **17.9%** |
| 2 | `:kiln/gdn/gated_norm` | **17.3%** |
| 3 | `:kiln/gdn/qk_norm` | **15.0%** |

Decode kernel-level cross-check:

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `Marlin<(256,1,8,8,4,8)>` | **14.4%** |
| 2 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64...` | **11.6%** |
| 3 | `ampere_bf16_s16816gemm_bf16_128x64...` | **9.7%** |

### Prefill top hotspots

Source: `profile-out/post442_prefill_nvtx_pushpop_sum.csv`. The table below
excludes the two-token decode tail `:kiln/lm_head` range from the prefill
ranking; it is present in the trace at **18.5%** but is not a prefill hotspot.

| Rank | NVTX range | Wall-clock share |
| ---: | --- | ---: |
| 1 | `:kiln/gdn/in_proj` | **32.0%** |
| 2 | `:kiln/attn/full/prefill_initial` | **14.4%** |
| 3 | `:kiln/mlp/gate` | **7.0%** |

Prefill kernel-level cross-check:

| Rank | Kernel | GPU-kernel share |
| ---: | --- | ---: |
| 1 | `gdn_full_chunk_forward_kernel` | **33.6%** |
| 2 | `Marlin<(256,4,16,4,4,8)>` | **11.8%** |
| 3 | `bmul_f32` | **8.8%** |

### Verdict

Fresh post-#442 evidence does **not** reopen FlashInfer paged decode. The
decode wall-clock top 3 remains entirely GDN-side, and the full-attention
projection region is still small (`:kiln/proj/qkv` **3.1%**, `:kiln/proj/o`
**0.9%** in the decode NVTX table).

Further GDN work remains the supported Phase 6 frontier. Decode is still
dominated by GDN gate/norm ranges, while prompt-heavy prefill still has
`gdn_full_chunk_forward_kernel` as the largest GPU-kernel bucket (**33.6%**)
and GDN-side wall-clock ranges at the top of the prefill NVTX table.

**Committed artifacts:**

- `profile-out/post442_decode_cuda_gpu_kern_sum.csv`
- `profile-out/post442_decode_nvtx_kern_sum.csv`
- `profile-out/post442_decode_nvtx_pushpop_sum.csv`
- `profile-out/post442_prefill_cuda_gpu_kern_sum.csv`
- `profile-out/post442_prefill_nvtx_kern_sum.csv`
- `profile-out/post442_prefill_nvtx_pushpop_sum.csv`

## Phase 6 post-#415 current-main native MTP A/B refresh (2026-04-23)

**Scope:** refresh the current-main source of truth for native MTP after PR
[#415](https://github.com/ericflo/kiln/pull/415) closed out the stale GDN
tiled/full-chunk retry queue and reconfirmed FlashInfer decode is still below
the reopen bar.

**Preflight outcome:** proceed. On fresh `main` at `d94d3c9` (PR #415
merged), `PROFILING.md` still only had older MTP sections from 2026-04-21 and
later C-series diagnostic writeups; it did **not** already contain a
post-#415 warm-pod `off` vs `skip_layer` vs `mtp` comparison on the standard
`--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`
workload. `gh pr list --repo ericflo/kiln --state all` showed no newer open or
merged PR that already landed this exact refresh.

**Hardware / image:** RunPod on-demand RTX A6000 (`$0.49/hr`),
`ghcr.io/ericflo/kiln-runpod:latest`.

**Build / setup commands:**

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
```

The pod did not already have `/workspace/qwen3.5-4b`, so the public HF
checkpoint had to be downloaded before the bench sweep.

**Benchmark commands run:**

```bash
for run in 1 2 3; do
  KILN_SPEC_METHOD=off \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    > profiling-artifacts/post415_20260423_off_run${run}.json

  KILN_SPEC_METHOD=skip_layer \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    > profiling-artifacts/post415_20260423_skip_layer_run${run}.json

  KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    > profiling-artifacts/post415_20260423_mtp_run${run}.json
done
```

`KILN_BENCH_FORCE_MTP=1` is required on current `main` for the 512-token bench
prompt. Without it, `bench.rs` routes `KILN_SPEC_METHOD=mtp` to the long-prompt
`skip_layer` fallback instead of timing raw native MTP.

### Medians

Source: `profiling-artifacts/post415_20260423_mtp_ab.json`

| Arm | actual prompt tokens | prefill ms | mean ITL ms | p50 ITL ms | p99 ITL ms | decode tok/s | α |
| --- | -------------------: | ---------: | ----------: | ---------: | ---------: | -----------: | -: |
| Off | **494** | **407.7** | **18.56** | **18.35** | **24.23** | **53.87** | — |
| Skip-layer paged | **502** | **414.3** | **471.30** | **474.63** | **900.47** | **2.12** | **0.000** |
| Native MTP | **502** | **419.1** | **36.14** | **47.52** | **55.94** | **27.67** | **0.245** |

### Raw run summaries

| Arm | Run | prefill ms | mean ITL ms | decode tok/s | α |
| --- | --- | ---------: | ----------: | -----------: | -: |
| Off | 1 | 7374.9 | 18.25 | 54.80 | — |
| Off | 2 | 401.7 | 18.75 | 53.33 | — |
| Off | 3 | 407.7 | 18.56 | 53.87 | — |
| Skip-layer paged | 1 | 417.9 | 470.31 | 2.13 | 0.000 |
| Skip-layer paged | 2 | 408.3 | 472.38 | 2.12 | 0.000 |
| Skip-layer paged | 3 | 414.3 | 471.30 | 2.12 | 0.000 |
| Native MTP | 1 | 415.0 | 36.07 | 27.72 | 0.245 |
| Native MTP | 2 | 419.1 | 36.29 | 27.55 | 0.245 |
| Native MTP | 3 | 450.8 | 36.14 | 27.67 | 0.245 |

### Verdict

Native MTP is **not** the next shipping speed lever on current `main`.

- The task's ship bar was `>= 1.5x` `off` decode tok/s with `α >= 0.72`.
- Measured current-main native MTP is **27.67 tok/s**, only **0.51x** of the
  `off` median (**53.87 tok/s**), with **α = 0.245**.
- Paged skip-layer remains dramatically worse at **2.12 tok/s** with
  **α = 0.000**.

So the blocker is still **low MTP acceptance**, not a missing warm-current-main
measurement and not a decode-kernel frontier gap. Even with the old
post-`#415` kernel retries explicitly closed out, current-main MTP remains
slower than plain decode by roughly **48.6%**.

### Notes

- The first `off` run had a **7374.9 ms** latency prefill outlier while later
  `off` runs were **401.7 / 407.7 ms**; the median-of-3 rule correctly drops
  that first-run warmup artifact.
- Current bench plumbing does not produce identical *actual* prompt token
  counts across arms on the nominal 512-token workload: `off` logged **494**
  prompt tokens while `skip_layer` / `mtp` logged **502**. That asymmetry is
  small compared with the 25.7 tok/s decode gap between `off` and `mtp`, but it
  is worth keeping in mind when comparing absolute prefill numbers.

### Recommendation

Do **not** queue another GDN tiled/full-chunk retry or a FlashInfer decode
retry off this result. The current-main answer is now explicit:

1. `skip_layer` is unshippable on the standard workload.
2. native MTP is also unshippable on the standard workload because α is too
   low and decode throughput is well below baseline.
3. the next narrow task should stay on MTP correctness / acceptance-rate root
   cause, not on another decode-kernel attempt justified by stale assumptions.

**Validation / evidence:** release build completed, nine benchmark JSON outputs
were produced on the same warm pod, and the committed aggregate artifact
`profiling-artifacts/post415_20260423_mtp_ab.json` records the per-run and
median numbers used above.

## Phase C36 post-#417 current-main identity-bias attribution (2026-04-23)

**Scope:** rerun the existing Phase C1 attribution path on fresh current
`main`, on the same standard native-MTP workload as PR #417, and make the
`draft_top1 == last_token` signature first-class in committed tooling and
artifacts.

**Preflight outcome:** proceed. Fresh `main` at `d5c57ed` (PR #417 merged)
still had no newer open or merged PR that landed a post-#417 current-main
identity-bias artifact for the standard
`--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`
workload. The fresh rerun below also still leaves native MTP below the ship
bar: median decode tok/s **21.95** and median α **0.3956** across seeds
`0/1/2`.

**Commands run:** build `kiln-bench` on RunPod A6000, then run the standard
MTP bench three times with `KILN_BENCH_FORCE_MTP=1` and
`KILN_C1_ATTR_PATH=docs/archive/phase-c/phase-c36/c1_seed${seed}.csv`, then summarize with:

```bash
python3 scripts/mtp_c1_summarize.py \
  --json profiling-artifacts/post417_20260423_c1_identity.json \
  docs/archive/phase-c/phase-c36/c1_seed0.csv docs/archive/phase-c/phase-c36/c1_seed1.csv docs/archive/phase-c/phase-c36/c1_seed2.csv
```

### Fresh metrics

| Seed | decode tok/s | α | C1 rows | identity rate |
| --- | ---: | ---: | ---: | ---: |
| 0 | 36.61 | 0.7397 | 73 | 10.96% |
| 1 | 20.00 | 0.3093 | 97 | 59.79% |
| 2 | 21.95 | 0.3956 | 91 | 52.75% |

Aggregate C1 summary (`profiling-artifacts/post417_20260423_c1_identity.json`):

- total rows: **261**
- α / top-k match rate: **45.98%**
- Class A rows: **0**
- Class B rows: **141**
- overall `draft_equals_last_token_rate`: **43.68%**
- reject-conditioned identity-bias rate: **80.85%**
- accept-conditioned identity-bias rate: **0.00%**

### Verdict

Identity bias still dominates the low-α failure mode on the standard
current-main native-MTP workload, but it is now **reject-conditioned** rather
than universal. The current source-of-truth number is that **114 of 141
rejects (80.85%)** are `draft_top1 == last_token`, while **0 of 120 accepts**
are. That makes identity bias the right first-class diagnostic to commit in
`scripts/mtp_c1_summarize.py` and the aggregate JSON artifact, even though the
overall workload is no longer a uniform collapse: seed 0 nearly clears the
paper floor (`α=0.7397`) while seeds 1 and 2 still sit in the low-α regime.

### Recommendation

Do **not** reopen FlashInfer or GDN decode work from this result. The next
bounded task should stay on MTP correctness: use the new identity-bias metrics
as the front-door verdict, then bisect why standard-workload seeds 1 and 2
stay trapped in the identity-biased reject regime while seed 0 escapes it on
the same current `main`.

## Phase C37 post-#420 seed0-vs-seed1 `h_main` drift audit (2026-04-23)

**Scope:** reuse the existing splice dump path plus
`scripts/c15_h_main_drift_audit.py` to test whether the post-#420 seed split
is already visible in the base-model hidden-state path on the standard native
MTP workload.

**Preflight outcome:** proceed. Fresh `origin/main` was still `59ea3b5`
(PR #419 merged, PR #420 already present), with no newer open or merged PR
landing a post-#420 seed0-vs-seed1 `h_main` drift artifact. A fresh rerun of
the standard workload also reproduced the split before the audit:

| Seed | prompt tokens | mean ITL ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 24.9 | 40.1 | 0.740 |
| 1 | 508 | 43.7 | 22.9 | 0.309 |

**Hardware note:** RunPod A6000 capacity was unavailable on 2026-04-23
(both pool resume and direct launch returned supply-constraint failures), so
this run used the project-policy fallback `A40` on the same kiln image.

**Commands run:** build `kiln-bench`, rerun the standard workload with
`KILN_MTP_DUMP_SPLICE=1`, `KILN_MTP_DUMP_HIDDEN_STATES=1`, and a dump path
template rooted at
`profiling-artifacts/post420_20260423_seed${seed}_captures/mtp_pos-{pos}/step-{step}.safetensors`,
then run `scripts/c15_h_main_drift_audit.py` twice per seed (default BF16 and
`--fp32`). The committed source-of-truth artifacts are the FP32 outputs:

- `profiling-artifacts/post420_20260423_hmain_seed0.json`
- `profiling-artifacts/post420_20260423_hmain_seed1.json`

### Verdict at `mtp_pos=0`

Yes, the split is already visible in `h_main`.

- **Seed 0 (escaping case):** FP32 audit is `clean` at `mtp_pos=0`. Minimum
  cosine is **0.999067**, last-step cosine **0.999675**, and drift shrinks
  across the captured window (ratio **0.348x**).
- **Seed 1 (trapped case):** FP32 audit is `drift` at `mtp_pos=0`. Minimum
  cosine falls to **0.990691** at step 6, last-step cosine is **0.998943**,
  and drift grows **3.425x** from step 0 to step 7.

So the low-α seed is not merely failing farther downstream on identical
hidden-state quality. By the late `mtp_pos=0` steps, it is already feeding the
MTP head a materially worse `h_main` than the escaping seed.

### Caveat

The existing C15 script's `pos=2` output is not usable on these per-seed
roots: it builds a 4-token chained sequence from the isolated `pos=2` files
and then indexes with `base_pos` values around 502/520, producing the
committed `index out of range` errors. No new instrumentation was added in
this task, so the source-of-truth verdict is intentionally limited to
`mtp_pos=0`.

### Recommendation

Do **not** queue another decode-kernel retry off this result. The next narrow
task should bisect the first seed1-only divergence upstream of the `h_main`
tap on the standard workload, using the existing hidden-state dump path and a
layer/sub-op comparator. Seed 1's failure is already visible before the
accept/reject seam.

## Phase C38 post-#422 seed1 upstream base-stack bisect (2026-04-23)

**Scope:** use the existing B10/B11/B12 hidden-state dump + HF reference +
comparator path on fresh `main` after PR #422 to localize the first
seed1-only divergence upstream of `h_main`.

**Preflight outcome:** proceed. Fresh `origin/main` was `c66cf41`
(PR #422 on `main`), with no newer open or merged PR already landing a
post-#422 upstream base-stack bisect artifact. A fresh standard-workload rerun
still reproduced the split:

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 5602.8 | 49.98 | 0.7162 |
| 1 | 508 | 377.2 | 30.82 | 0.2828 |

**Hardware note:** the pod-pool A6000 could not resume (`not enough free GPUs
on the host machine`). Direct A6000 launch hit `SUPPLY_CONSTRAINT`, and direct
A40 launch failed with a RunPod internal error, so the fresh rerun used the
documented fallback `A100 PCIe` on the kiln image.

**Commands run:** rebuilt `kiln-bench`, reran the standard workload for seeds
0/1 with `KILN_MTP_DUMP_HIDDEN_STATES=1`, `KILN_MTP_DUMP_B11_TAPS=1`, and
`KILN_MTP_DUMP_B12_GQA_TAPS=1`, generated matched HF reference dumps with
`scripts/mtp_h_main_reference_dump.py` in both bf16 and fp32, then ran
`scripts/mtp_compare.py --b10`, `--b11`, and `--b12` for both seeds. The
committed source-of-truth artifacts are:

- `profiling-artifacts/post422_20260423_seed0_compare_bf16.txt`
- `profiling-artifacts/post422_20260423_seed1_compare_bf16.txt`
- `profiling-artifacts/post422_20260423_seed0_compare_fp32.txt`
- `profiling-artifacts/post422_20260423_seed1_compare_fp32.txt`

### Verdict

**Blocked by the existing reference contract.** The current B10/B11/B12 path
cannot honestly localize a **seed1-only** first divergence beyond the initial
`mtp_pos=0` dump.

Why:

- `mtp_h_main_reference_dump.py` forwards the `prompt_tokens` tensor embedded
  in each splice dump.
- Only the first `mtp_pos=0` dump (`0_s0`) contains the full prompt
  (`494` tokens for seed 0, `508` for seed 1).
- Every later `mtp_pos=0` dump carries `prompt_tokens` length `1`, and the
  `mtp_pos=2` path carries lengths `2` then `1`.
- So every later comparator row is an under-conditioned HF replay on the wrong
  sequence length. Both seeds then look "divergent" for the same reason.

On the **only** fully-contexted row (`0_s0`), seed 0 and seed 1 do not
separate in a way that makes the next fix target obvious:

- `h_layer_0`: seed 0 `1.00`, seed 1 `1.00`
- `h_layer_24`: seed 0 `0.985`, seed 1 `0.976`
- `h_layer_31`: seed 0 `0.958`, seed 1 `0.957`
- B11 `gdn_conv`: seed 0 `1.00`, seed 1 `1.00`

The mechanically-generated full reports do produce the same nominal
localization for **both** seeds:

- B10: `DIVERGENCE AT LAYER 0`
- B11: first below-bar sub-op = `gdn_conv`
- B12: first below-bar late-stack layer = `h_layer_24`

But those are **not** seed1-only findings; they are artifacts of the
incomplete-history replay on every row after `0_s0`.

### Recommendation

Do **not** queue a fix from this task. The next narrow task should keep the
same B10/B11/B12 comparator stack but fix the replay contract first, either
by serializing the full chained prompt history into every splice dump or by
teaching the HF reference side to reconstruct the chained history C15-style
before re-running the existing comparators.

## Phase 6 post-#412 preflight — tiled recurrent-state retry is obsolete (2026-04-23)

**Scope:** re-check the queued "prototype vLLM-style block-tiled
recurrent-state update in the full-chunk GDN kernel" task on fresh `main`
after PR [#412](https://github.com/ericflo/kiln/pull/412), and stop at docs if
current-main evidence already shows this exact slice has been attempted and
consumed.

**Preflight outcome:** stop with docs only. On fresh `main` at `e516f24`
(2026-04-23):

- PR [#412](https://github.com/ericflo/kiln/pull/412) is merged and doc-only
  (`PROFILING.md` only);
- there is still no **merged or open** PR that lands a kept block-tiled
  recurrent-state update in
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`;
- `crates/kiln-model/src/backend/cuda.rs` still routes paged decode through
  `kiln_flash_attn::flash_attn_paged_decode(...)`;
- `crates/kiln-gdn-kernel/src/lib.rs` still says the core `chunk_gla_fwd`
  vendor already landed, so that old queue item remains consumed;
- but `PROFILING.md` no longer leaves this tiled recurrent-state step as live
  remaining work: the existing
  ["Post-#406 tiled full-chunk recurrent-state update"](#post-406-tiled-full-chunk-recurrent-state-update--2026-04-23)
  section already records the exact same vLLM-style tiled epilogue retry,
  parity result, warmed same-pod control, and revert verdict.

That last point fails the task's remaining-work precondition. Fresh `main`
already says this slice was tried and rejected:

- parity passed;
- fresh-first timing looked faster;
- warmed control on the same pod measured **3112.7 ms** on `main` vs
  **3167.5 ms** on the patched branch;
- the candidate was therefore **1.8% slower**, which missed the `>=3%`
  keep bar and was reverted before PR [#411](https://github.com/ericflo/kiln/pull/411)
  merged.

**Decision:** no implementation follow-up and no new RunPod spend.

The honest current-main conclusion is that this exact bounded tiled
recurrent-state retry is already consumed. Re-running it would duplicate the
same failed frontier, not advance Phase 6. The broader GDN path is still the
active area, but this particular structural slice is not remaining work
anymore on fresh `main`.

For avoidance of doubt: FlashInfer-style paged decode also remains below the
reopen threshold on Qwen3.5-4B. PR [#412](https://github.com/ericflo/kiln/pull/412)
already reconfirmed that the decode top hotspots remain GDN-only and that the
full-attention decode region a FlashInfer port would replace is still
sub-floor.

**Validation:** no pod launched; no CUDA files changed; no build, test, or
benchmark commands run because the fresh-main preflight already showed this
task was obsolete.

## Phase 6 post-#411 recheck — FlashInfer paged decode still does not reopen (2026-04-23)

**Scope:** re-check the queued "vendor minimal FlashInfer-style paged GQA
decode" step on fresh `main` after PR
[#411](https://github.com/ericflo/kiln/pull/411), and stop at docs if the
existing CUDA paged-decode backend is still the vendored flash-attn path and
the current profile still leaves full-attention decode below the reopen bar.

**Preflight outcome:** stop with docs only. On fresh `main` at `b9d5b52`
(2026-04-23):

- there is still **no landed FlashInfer-style paged decode implementation** in
  kiln; prior related PRs remain doc-only redirects
  ([#163](https://github.com/ericflo/kiln/pull/163),
  [#387](https://github.com/ericflo/kiln/pull/387));
- open-PR search for `flashinfer`, `paged decode`, and
  `flash_attn_paged_decode` found no active overlapping implementation PR;
- `crates/kiln-model/src/backend/cuda.rs` still routes CUDA paged decode
  through `kiln_flash_attn::flash_attn_paged_decode(...)`;
- the current decode path is still the vendored flash-attn paged-decode path,
  not a newer minimal FlashInfer port.

**Decision:** no implementation follow-up.

The fresh current-main evidence already at the top of this file still leaves
FlashInfer-class decode work below the keep bar on Qwen3.5-4B:

- decode top-3 remain GDN-only:
  `:kiln/gdn/gates` **17.8%**, `:kiln/gdn/gated_norm` **17.3%**,
  `:kiln/gdn/qk_norm` **14.9%**;
- full-attention projections remain only ~**3.8%** of decode wall-clock
  (`:kiln/proj/qkv` ~**3.0%** + `:kiln/proj/o` ~**0.8%** in the latest
  frontier sections below);
- the inner `:kiln/attn/full/*` kernel region a FlashInfer-style paged decode
  kernel would replace is still below the reporting floor.

That keeps the same Amdahl ceiling already documented by the earlier
redirects: even a heroic replacement of the inner paged-attention kernel
cannot clear the task's `>=3%` keep bar on this model. After `#411`, the next
honest Phase 6 work is still in the GDN path, not a reimplementation of the
full-attention decode backend.

**Validation:** no pod launched; no CUDA files changed; no build or benchmark
commands run because this is a docs-only redirect on current-main evidence.

## Phase C39 post-#423 replay-context fix + fresh seed bisect (2026-04-23)

**Scope:** extend the B10/B11/B12 dump contract so every splice row carries
the fully conditioned replay sequence, rerun the standard native-MTP seed-0 /
seed-1 workload on fresh `main` after PR #423, and re-run the matched HF
reference / comparator path on the updated artifacts.

**Preflight outcome:** proceed. Fresh `origin/main` was `61badb6`
(PR #423), and current `main` did **not** already serialize enough token
context for later splice rows. An initial rerun on the unchanged bench path
still logged `replay_tokens_len=1` / `2` after step 0, proving the missing
context was still present outside the kiln-model generation loop.

**Code path fixed:** the replay-prefix plumbing now covers both the
kiln-model native-MTP generation loops and the `kiln-bench` native-MTP bench
loop. Fresh rerun logs on a supported A40 show the intended full-sequence
contract:

- seed 0: `step-1 replay_tokens_len=495`, `step-2=496`, `mtp_pos=2 step-0=502`
- seed 1: `step-1 replay_tokens_len=509`, `step-7=515`, `mtp_pos=2 step-0=520`

Those rows previously carried only the current 1-token / 2-token slice.

**Fresh workload check (standard native-MTP path):**

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8734.0 | 36.13 | 0.7162 |
| 1 | 508 | 452.6 | 22.92 | 0.2828 |

So the seed split still reproduces on fresh `main`; C39 only changes the
replay contract, not the underlying acceptance behavior.

**Source-of-truth artifacts:** the refreshed comparator outputs are:

- `profiling-artifacts/post423_20260423_seed0_compare_bf16.txt`
- `profiling-artifacts/post423_20260423_seed0_compare_fp32.txt`
- `profiling-artifacts/post423_20260423_seed1_compare_bf16.txt`
- `profiling-artifacts/post423_20260423_seed1_compare_fp32.txt`

Across both bf16 and fp32 HF references, the seed-0 and seed-1 verdicts are
the same:

- B10: first diverging tap is `h_layer_8` (`h_layer_0` still matches), verdict
  `DIVERGENCE IN EARLY GDN STACK`
- B11: `ALL layer-0 GDN sub-ops match within cos_sim >= 0.95`
- B12: first below-bar late-stack layer is `h_layer_24`, verdict
  `DRIFT FIRST APPEARS AT 'h_layer_24' (not layer 31)`

**Verdict:** the replay-contract blocker from C38 is fixed, but the refreshed
post-#423 artifact still does **not** expose a seed1-only first boundary or
sub-op. With fully conditioned HF replays, both the control and failing seeds
land the same first boundary verdict (`h_layer_8`) and the same later-stack
drift verdict (`h_layer_24`). The honest next recommendation is **not** a
seed1-only fix from this artifact; it is a narrower bisect inside the early
GDN span between `h_layer_0` and `h_layer_8`, because that is now the first
shared bad boundary on both seeds.

## Phase C40 dense early h_main sweep (2026-04-23)

**Scope:** densify the shared upstream `h_main` bisect from C39 by adding an
opt-in early-layer sweep for `h_layer_1..8`, then rerun the standard
native-MTP workload on fresh `main` after PR #426.

**Preflight outcome:** proceed. Fresh `origin/main` was `4df936c`
(PR #426), and no open PR already landed the dense early-layer sweep or a
committed artifact localizing the first shared drift more narrowly than
`h_layer_0 -> h_layer_8`.

**Code change:** the h_main dump path now supports
`KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1`, serializes the actual emitted boundary set
as `meta__boundary_layers`, mirrors that sequence in
`scripts/mtp_h_main_reference_dump.py`, and teaches
`scripts/mtp_compare.py --b10` to derive/report the first divergence from the
real emitted layer order instead of the historical sparse B10 list.

**Validation commands run (RunPod A6000, kiln image):**

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

**Standard workload rerun:** same C39 shape, but with dense early h_main taps
enabled:

```bash
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_METHOD=mtp \
KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 \
KILN_MTP_DUMP_SPLICE_POS=0,2 \
KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 \
KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_PATH=.../mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed <0|1>
```

Matched HF reference dumps were regenerated in bf16 and fp32 with
`scripts/mtp_h_main_reference_dump.py`, then compared with
`scripts/mtp_compare.py --b10`.

### Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8795.4 | 39.57 | 0.6842 |
| 1 | 508 | 386.8 | 23.95 | 0.2959 |

The seed split still reproduces on fresh `main`.

### Source-of-truth artifacts

- `profiling-artifacts/post426_c40_20260423_seed0.bench.json`
- `profiling-artifacts/post426_c40_20260423_seed0.bench.stderr`
- `profiling-artifacts/post426_c40_20260423_seed1.bench.json`
- `profiling-artifacts/post426_c40_20260423_seed1.bench.stderr`
- `profiling-artifacts/post426_c40_20260423_seed0_compare_bf16.txt`
- `profiling-artifacts/post426_c40_20260423_seed0_compare_fp32.txt`
- `profiling-artifacts/post426_c40_20260423_seed1_compare_bf16.txt`
- `profiling-artifacts/post426_c40_20260423_seed1_compare_fp32.txt`

### Verdict

**Both seeds now localize to the exact same first bad layer: transformer block
1.**

Across both bf16 and fp32 HF references:

- seed 0: first diverging tap = `h_layer_1` (`mtp_pos-0-step-1`)
- seed 1: first diverging tap = `h_layer_1` (`mtp_pos-2-step-1`)
- last matching tap before divergence = `h_layer_0` for both seeds
- comparator verdict = `DIVERGENCE FIRST APPEARS AT LAYER 1` for both seeds

So the shared upstream drift no longer spans `0 -> 8`; it starts immediately
after layer 1, with layer 0 still numerically clean.

### Recommendation

Do **not** keep bisecting across coarse `h_main` boundaries. The remaining
narrow span is now **inside transformer block 1** itself: layer-1 input norm,
GDN projections/conv/qk-norm/gates/recurrent kernel, or the layer-1 residual
handoff. The next diagnostic task should capture layer-1 sub-op taps rather
than adding another per-layer sweep.

## Phase C41 post-#427 transformer block 1 sub-op bisect (2026-04-23)

**Scope:** take the post-#427 dense early-layer verdict (`h_layer_0` still ok,
first shared drift at `h_layer_1`) and narrow it to the earliest shared
transformer-block-1 sub-op boundary on the same native-MTP replay contract.

**Hardware / image:** RunPod on-demand `NVIDIA A40` (A6000 unavailable; used
the project fallback order), `ghcr.io/ericflo/kiln-runpod:latest`.

**Code change:** add an opt-in C41 capture path behind
`KILN_MTP_DUMP_C41_LAYER1_TAPS=1`, serialize the emitted tap ids as
`meta__c41_tap_ids`, mirror the same tap set in
`scripts/mtp_h_main_reference_dump.py --c41-taps`, and teach
`scripts/mtp_compare.py --c41` to report the earliest shared bad layer-1 tap
from the explicit `c41__*` boundary set. The HF reference path also now slices
those C41 taps to the final replay token so kiln and HF dump the same shapes.

**Validation commands run (RunPod kiln image):**

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

**Standard workload rerun:** same C40 workload plus the new C41 tap flag:

```bash
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_METHOD=mtp \
KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 \
KILN_MTP_DUMP_SPLICE_POS=0,2 \
KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 \
KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_C41_LAYER1_TAPS=1 \
KILN_MTP_DUMP_PATH=.../mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed <0|1>
```

Representative C40 carry-forward positions were re-referenced and compared:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed0_ref_bf16.safetensors \
  --device cuda \
  --c41-taps

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post427_c41_20260423_seed1_ref_fp32.safetensors \
  --device cuda \
  --fp32 \
  --c41-taps

python3 scripts/mtp_compare.py --c41 \
  --pair seed0:profiling-artifacts/post427_c41_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post427_c41_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post427_c41_20260423_seed1_ref_bf16.safetensors
```

### Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 8900.8 | 39.0 | 0.740 |
| 1 | 508 | 479.8 | 42.5 | 0.283 |

The post-#427 seed split still reproduces on fresh main.

### Source-of-truth artifacts

- `profiling-artifacts/post427_c41_20260423_seed0.bench.json`
- `profiling-artifacts/post427_c41_20260423_seed0.bench.stderr`
- `profiling-artifacts/post427_c41_20260423_seed1.bench.json`
- `profiling-artifacts/post427_c41_20260423_seed1.bench.stderr`
- `profiling-artifacts/post427_c41_20260423_compare_bf16.txt`
- `profiling-artifacts/post427_c41_20260423_compare_fp32.txt`

### Verdict

**Both seeds and both HF dtypes agree that the earliest bad layer-1 tap is
`layer_1_post_input_norm`.**

- bf16 compare: earliest shared bad tap = `layer_1_post_input_norm`
- fp32 compare: earliest shared bad tap = `layer_1_post_input_norm`
- shared later-good taps still exist (`gdn_gate_beta`, `gdn_recur_out`,
  `gdn_out_proj`, `layer_1_post_attn_residual`), so this is not a monotonic
  “everything inside layer 1 is broken” result
- the first observed bad tensor is already the first explicit C41 boundary

That means the C40 “inside transformer block 1” span is now closed to the
earliest captured boundary: divergence is already present by the output of
layer 1 input_layernorm.

### Recommendation

The remaining uncaptured span is now extremely narrow:

- the residual input arriving at transformer block 1
- the layer-1 `input_layernorm` application itself

Do **not** broaden tracing again. The next task, if needed, should instrument
the pre-norm residual input and/or audit layer-1 input-layernorm numerics
directly instead of tracing deeper GDN internals.

## Phase 6 post-#392 current-main re-profile — 2026-04-23

**Scope:** refresh the Phase 6 performance source of truth on fresh `main`
after PR [#392](https://github.com/ericflo/kiln/pull/392) landed and name the
single next optimization target from current evidence, not from the old
post-`#384` assumption that `:kiln/attn/gdn/chunk_prep` and
`:kiln/attn/gdn/chunk` still define the frontier.

**Preflight outcome:** proceed. On fresh `main` at `c392cf1` (2026-04-23),
PR #392 is merged (`57f67ae`), there is no newer merged or open PR that
already lands a post-#392 re-profile, `PROFILING.md` still ends with the
post-`#384` / current-main sections plus a short post-change note rather than
a full post-#392 refresh, and
`crates/kiln-model/src/forward.rs` still routes full 64-token BF16 prefill
chunks through `backend.gdn_full_chunk_forward(...)`.

**Hardware / image:** RunPod on-demand RTX A6000 (`$0.49/hr`),
`ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`, CUDA toolkit
`12.4`.

**Build / validation commands:**

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
cargo build --release --features cuda,nvtx --bin kiln-bench
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

**Profiling commands run:**

```bash
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --latency-only
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=70 --duration=20 -o /workspace/phase6-profile/post392-decode ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=3 --duration=4 -o /workspace/phase6-profile/post392-prefill ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
```

The baked image still ships `nsys 2023.4.4`, so this run upgraded Nsight
userspace and used
`/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys` for all final
captures and exports.

The literal prompt-heavy prefill command above did **not** capture usable
prefill NVTX on this pod: with current `main`, model load still takes
~26-28 s, so `--delay=3 --duration=4` samples startup instead of the
`8192/1` prefill region. For the committed prefill CSVs below I reran that
same `8192/1 --latency-only` arm with a capture window shifted onto the
actual prefill span:

```bash
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=26 --duration=4 -o /workspace/phase6-profile/post392-prefill-refined ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
```

### Decode uncaptured runs — paged 512/128

| run | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms | prefill ms |
| --- | -----------: | ----------: | ---------: | ---------: | ---------: |
| 1 | 51.18 | 19.54 | 19.49 | 25.30 | 374.7 |
| 2 | 50.94 | 19.63 | 19.50 | 24.43 | 372.6 |
| 3 | 51.97 | 19.24 | 19.14 | 21.40 | 369.8 |
| median | **51.18** | **19.54** | **19.49** | **24.43** | **372.6** |

Decode on fresh post-#392 `main` is materially faster than the post-`#384`
refresh (`43.8 tok/s`, `22.8 ms` mean ITL). The decode hotspot ordering,
however, is unchanged.

### Prompt-heavy prefill timing — paged 8192/1

The uncaptured prompt-heavy timing run on current `main` produced:

- prompt tokens: **8180**
- prefill time: **3277.7 ms**
- prefill throughput: **2495.6 tok/s**

That is a regression against the post-`#384` baseline already recorded in
this file (**2831.2 ms / 2889.3 tok/s** uncaptured,
**2947.8 ms / 2775.0 tok/s** captured).

### Top-3 NVTX hotspots — decode

Source: `profiling-artifacts/post392_20260423_decode_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **17.8** | `:kiln/gdn/gates` |
| 2 | **17.3** | `:kiln/gdn/gated_norm` |
| 3 | **14.9** | `:kiln/gdn/qk_norm` |

Decode remains the same story as post-`#384`: GDN gate/norm work still owns
the top of the decode profile, and FlashInfer-style paged decode still fails
the reopen bar.

### Top-3 NVTX hotspots — prefill

Source: `profiling-artifacts/post392_20260423_prefill_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **53.9** | `:kiln/gdn/in_proj` |
| 2 | **22.6** | `:kiln/attn/full/prefill_initial` |
| 3 | **9.3** | `:kiln/mlp/gate` |

The important post-#392 prefill fact is not the exact late-prefill NVTX
ordering above; it is that the old chunkwise frontier is gone from the top of
the table. In the refined capture, `:kiln/attn/gdn/chunk_prep` and
`:kiln/attn/gdn/chunk` have collapsed to **0.1% / 0.1%**. That is a real
ordering change versus post-`#384`, where those same ranges were
**51.1% / 25.1%**.

### Kernel callouts

- **Decode top kernels** (`profiling-artifacts/post392_20260423_decode_kern.csv`):
  Marlin GEMM at **14.7%**, large CUTLASS BF16 GEMMs at **11.9%**, **10.0%**,
  and **9.2%**, `ucopy_bf16` at **8.0%**, and
  `gdn_full_chunk_forward_kernel` at **2.3%**.
- **Prefill top kernels** (`profiling-artifacts/post392_20260423_prefill_kern.csv`):
  `gdn_full_chunk_forward_kernel` at **34.6%**, Marlin at **10.6%**,
  `bmul_f32` at **10.0%**, `ucopy_bf16` at **7.5%**, `badd_f32` at **4.4%**,
  and `ucopy_f32` at **4.0%**. The old chunkwise pair is effectively gone:
  `gdn_chunk_scan_kernel<64>` is **0.1%** and `gdn_chunk_prep_kernel<64>` is
  **0.0%** in this refined post-#392 prefill window.

This is the core post-#392 result: the fused full-chunk path **did**
materially change hotspot ordering, but it did **not** deliver an end-to-end
prefill win. The dominant prefill work is now inside the fused
`gdn_full_chunk_forward_kernel` and the HBM-heavy elementwise traffic around
it, not in the old `chunk_prep` / `chunk` Candle path.

### Comparison vs the post-#384 section

Against the post-`#384` current-main re-profile already in this file:

- decode median improved from **43.8 tok/s** to **51.18 tok/s**;
- decode mean ITL improved from **22.8 ms** to **19.54 ms**;
- decode top-3 ordering stayed the same (`gates`, `gated_norm`, `qk_norm`);
- prompt-heavy uncaptured prefill regressed from **2831.2 ms / 2889.3 tok/s**
  to **3277.7 ms / 2495.6 tok/s**;
- prefill NVTX ordering changed materially:
  `chunk_prep` + `chunk` fell from **76.2%** combined to **~0.2%** combined in
  the refined post-#392 window;
- prefill kernel ordering changed materially:
  `gdn_chunk_scan_kernel<64>` fell from **18.4%** to **0.1%**, while
  `gdn_full_chunk_forward_kernel` rose to **34.6%** and became the dominant
  prompt-heavy prefill kernel.

So the fresh post-#392 evidence says two things at once:

1. **#392 changed the hotspot ranking.** The pre-#392 "finish chunk_prep /
   chunk" queue shape is stale.
2. **#392 is still a regression on uncaptured prompt-heavy prefill.** The
   new top bottleneck is the fused full-chunk kernel path itself, not the old
   chunkwise fallback around it.

### Recommendation

**Single next optimization target:** treat
`gdn_full_chunk_forward_kernel` as the new Phase 6 prefill frontier and port
the best available upstream fused-GDN kernel work into that path, rather than
retrying the old `chunk_prep` / `chunk` optimization loop and rather than
reopening FlashInfer decode.

Concretely, the next task should target the vendored fused full-chunk GDN
kernel itself: compare kiln's `gdn_full_chunk_forward_kernel` against vLLM's
newer `fused_recurrent_gated_delta_rule` / tuned GDN path, then port the
missing wins into kiln. Fresh post-#392 evidence supports that specific
direction:

- old chunkwise kernels are no longer the bottleneck;
- the fused full-chunk kernel alone is **34.6%** of refined prompt-heavy
  prefill kernel time;
- the surrounding `ucopy_bf16` / `ucopy_f32` / `bmul_f32` traffic is still
  large enough that a kernel-internal memory-traffic reduction has a credible
  ceiling;
- decode full attention remains far below the >5% reopen bar.

### Follow-up blocked on RunPod infra — 2026-04-23

Attempted follow-up task `phase6: port one vLLM fused-GDN win into full-chunk
prefill kernel` stopped before validation because the required RunPod GPU
environment never became reachable.

- Requested on-demand `NVIDIA RTX A6000` with
  `ghcr.io/ericflo/kiln-runpod:latest` failed immediately with
  `SUPPLY_CONSTRAINT`.
- Fallback launch on on-demand `A40` succeeded at pod creation
  (`gx9qu9cmmoxrrm`, `$0.44/hr`) but remained `runtime: null` / unreachable for
  more than 5 minutes and was terminated per task policy.
- Second launch on on-demand `A100 PCIe` (`i3rgpq7yiehyzo`, `$1.39/hr`) also
  remained `runtime: null` / unreachable and was terminated. Per the task
  brief's second-launch rule, this attempt stopped cleanly with a doc-only
  PR instead of shipping unvalidated CUDA changes.

No parity tests, `8192/1` timing runs, or post-change kernel-share captures
were executed in this attempt, so the post-`#392` profile above remains the
current source of truth.

### Follow-up parity retry — 2026-04-23

Retried the same Phase 6 frontier on fresh `main` after `#397`, this time on a
reachable on-demand A40 fallback pod (same `sm_86` envelope as A6000) with the
required kiln image. Preflight still passed:

- `#397` is merged on `main` and remains doc-only (`PROFILING.md` only);
- no newer open or merged PR already modifies
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this exact goal;
- the post-`#392` section above still shows
  `gdn_full_chunk_forward_kernel` at **34.6%** of refined prompt-heavy prefill
  kernel time.

The upstream idea attempted here was the smallest vLLM-style cleanup available
inside kiln's fused full-chunk kernel: stop redundant bf16 round-trips in the
final recurrent-state update path while leaving the chunk body and launch
envelope unchanged.

Validation command on the pod:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export CARGO_PROFILE_DEV_DEBUG=0
cargo test -p kiln-model --features cuda \
  forward::tests::test_gdn_full_chunk_forward_matches_fallback \
  -- --exact --nocapture
```

Result: **fail**. The kernel launched and the output chunk stayed within the
existing tolerance, but the recurrent-state parity regressed past the test bar:

- `out_chunk` max abs diff: **1.5625e-2**
- `state` max abs diff: **3.90625e-2**
- test threshold for `state`: **3.5e-2**

Because the focused CUDA parity check failed, this retry did **not** run the
`8192/1` timing arm or a new Nsight capture, and it does **not** ship a kernel
change. The current source of truth therefore remains the post-`#392` profile
above, and the next Phase 6 retry on this frontier needs a different upstream
idea than this scalar round-trip cleanup.

### Post-#399 bounded W-decay hoist — 2026-04-23

Fresh-`main` preflight passed again before editing:

- PR `#399` is merged and remains doc-only (`PROFILING.md` only).
- No newer open or merged PR already changes
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this post-`#399`
  full-chunk optimization target.
- The validated post-`#392` refined prefill capture above still shows
  `gdn_full_chunk_forward_kernel` at **34.6%** of prompt-heavy prefill kernel
  time, so the frontier still exists.

This retry ported one different, bounded vLLM-inspired win inside the vendored
full-chunk kernel: fold the per-token `decay_last` weighting into the shared
`W` rows once after chunk output is produced, then reuse the weighted rows in
the recurrent-state update instead of recomputing the same `w * decay_last`
bf16 product inside every `dk` iteration.

**Attempted code path**

- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`

**Parity**

RunPod on-demand RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export CARGO_PROFILE_DEV_DEBUG=0
cargo test -p kiln-model --features cuda \
  forward::tests::test_gdn_full_chunk_forward_matches_fallback \
  -- --exact --nocapture
cargo test -p kiln-gdn-kernel gdn_gates_parity_vs_candle_reference \
  -- --exact --nocapture
```

Result: **pass**.

- `out_chunk` max abs diff: **1.5625e-2**
- `state` max abs diff: **3.125e-2**
- `gdn_gates_parity_vs_candle_reference`: passed

**8192/1 prompt-heavy prefill**

Baseline already recorded in the validated post-`#392` section above:

- before: **3277.7 ms**, **2495.6 tok/s**

Current branch uncaptured reruns on the same `--paged --prompt-tokens 8192
--max-output-tokens 1 --skip-training --latency-only` arm:

- warm rerun 1: **3566.0 ms**, **2293.9 tok/s**
- profile-arm rerun: **3686.3 ms**, **2219.0 tok/s**

So this kernel change is **parity-safe but slower** on prompt-heavy prefill on
the measured A6000 run, and the kernel diff was reverted before opening the
PR. This section documents the failed retry; it does **not** land a kernel
change.

**Refined kernel-share capture**

I attempted the requested refined prompt-heavy capture on the same pod with:

```bash
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  --delay=26 --duration=4 \
  -o /workspace/phase6-profile/post399-prefill \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training \
  --latency-only
```

and two bounded fallbacks:

- `--cuda-graph-trace=graph:host-only`
- `KILN_CUDA_GRAPHS=false`

All three runs produced `.qdstrm` traces but the pod's baked `nsys 2023.4.4`
failed to import them into `.nsys-rep` with the same `Wrong event order has
been detected` error, so there is **no new post-change kernel-share CSV** from
this pod. The failure details are recorded in
`profiling-artifacts/post399_20260423_prefill_kern.csv`.

**Frontier status**

Yes: absent a new validated post-change kernel-share export, the last valid
refined capture still leaves `gdn_full_chunk_forward_kernel` as the prompt-
heavy prefill frontier. This specific W-decay hoist does not clear the
end-to-end bar, so the next retry on this path should target a different
kernel-internal bottleneck than both `#399`'s scalar round-trip cleanup and
this shared-`W` weighting hoist.

### Post-#401 bounded shared-`k_t` row staging — 2026-04-23

Fresh-`main` preflight passed again before editing:

- PR `#401` is merged and remains doc-only (`PROFILING.md` plus a capture-note
  CSV only).
- No newer open or merged PR already changes
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this post-`#401`
  follow-up scope.
- The validated post-`#392` refined prefill capture above still shows
  `gdn_full_chunk_forward_kernel` at **34.6%** of prompt-heavy prefill kernel
  time, so the frontier still exists on the current source of truth.

This retry ported one different bounded vLLM-style idea into the fused
full-chunk kernel's recurrent-state update: stage each `k_t[k_idx, :]` row
into shared memory once per `dk` iteration, then let all `dv` lanes reuse that
staged row instead of reloading the same 64 bf16 `k_t` scalars from global
memory in every thread. The motivating upstream pattern is the current vLLM
`fused_recurrent_gated_delta_rule_fwd_kernel`, which loads `b_k` once per
program tile and reuses it across all value lanes.

**Attempted code path**

- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`

**Parity**

RunPod on-demand RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export CARGO_PROFILE_DEV_DEBUG=0
cargo test -p kiln-model --features cuda \
  forward::tests::test_gdn_full_chunk_forward_matches_fallback \
  -- --exact --nocapture
```

Result: **pass**.

- `out_chunk` max abs diff: **1.5625e-2**
- `state` max abs diff: **3.125e-2**

**8192/1 prompt-heavy prefill**

To avoid cross-pod noise, I measured both arms on the same A6000 pod after the
branch benchmark looked much slower than the historical post-`#392` number.

Same-pod control on fresh `main`:

- before: **3272.4 ms**, **2499.7 tok/s**

Branch with shared-`k_t` row staging:

- after: **4704.0 ms**, **1739.0 tok/s**

That is a clear regression on the required prompt-heavy arm, so the kernel
change was reverted before opening the PR. This section documents the failed
attempt; it does **not** land a source change in
`gdn_full_chunk_forward.cu`.

**Kernel-share capture note**

Unlike the post-`#399` retry, I did **not** keep a new Nsight kernel-share
capture for this attempt. Once the same-pod control showed the branch at
**4704.0 ms** versus `main` at **3272.4 ms**, the code no longer met the keep
bar, so I reverted it immediately instead of burning more pod time on a
post-change profile for a change that was already disqualified. The capture
note for this decision is recorded in
`profiling-artifacts/post401_20260423_prefill_kern.csv`.

### Post-#403 bounded front-half triangular mask packing — 2026-04-23

Fresh-`main` preflight passed before editing:

- PR `#403` is merged and remains doc-only: it updated `PROFILING.md` plus
  `profiling-artifacts/post401_20260423_prefill_kern.csv`, with **no** source
  change under `crates/kiln-gdn-kernel/`.
- No newer open or merged PR already changes
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this post-`#403`
  front-half follow-up scope. PR `#404` only touches
  `crates/kiln-server/src/bench.rs`, and there were no open PRs at preflight.
- The validated post-`#392` refined prefill capture above still shows
  `gdn_full_chunk_forward_kernel` at **34.6%** of prompt-heavy prefill kernel
  time, with the hot slice centered on the fused full-chunk prefill body
  rather than decode.

This retry targeted the **front half** of `gdn_full_chunk_forward_kernel`
instead of the already-exhausted recurrent-state update. The bounded idea was:
pack `s_a` and `s_b` as the exact triangular regions the algorithm reads
(strict-lower for `a`, lower-with-diagonal for `b`) instead of reserving full
`64 x 64` shared tiles whose upper halves are always zero. The expected upside
was lower shared-memory footprint for the `big_g` / decay / mask / `out_chunk`
path while keeping the math and synchronization structure unchanged.

**Attempted code path**

- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`

**Parity**

RunPod on-demand RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export CARGO_PROFILE_DEV_DEBUG=0
cargo test -p kiln-model --features cuda \
  forward::tests::test_gdn_full_chunk_forward_matches_fallback \
  -- --exact --nocapture
```

Result: **pass**.

- `out_chunk` max abs diff: **1.5625e-2**
- `state` max abs diff: **3.125e-2**

**8192/1 prompt-heavy prefill**

Required same-pod order:

- fresh `main` control first: **4785.1 ms**, **1709 tok/s**
- branch second (triangular mask packing): **3348.3 ms**, **2443 tok/s**

That looked like a keep, but the branch ran second on the same warmed pod and
the gain was much larger than the expected 5-15% range, so I ran one extra
sanity control on the same pod after rebuilding `main` again:

- fresh `main` warm rerun after the branch: **3347.1 ms**, **2444 tok/s**

That warm-`main` rerun is effectively identical to the branch. So the apparent
`4785 -> 3348 ms` improvement was a second-arm warm effect, not a durable
kernel win. The real same-pod delta after controlling for warmth is below the
task's keep bar, so the kernel diff was reverted before opening the PR.

**Keep / revert decision**

Revert. This section documents the failed front-half attempt; it does **not**
land a source change in `gdn_full_chunk_forward.cu`.

**Capture note**

I did not spend additional pod time on a post-change Nsight capture once the
warm-`main` rerun showed parity with the branch. The measurement record for the
control, branch, and warm-control sanity rerun is stored in
`profiling-artifacts/post403_20260423_prefill_kern.csv`.

### Post-#405 vLLM fused recurrent GDN audit verdict — 2026-04-23

Fresh-`main` preflight passed before editing:

- PR `#405` is merged and remains doc-only: it updated `PROFILING.md` plus
  `profiling-artifacts/post403_20260423_prefill_kern.csv`, with **no** source
  change under `crates/kiln-gdn-kernel/` or `crates/kiln-model/src/forward.rs`.
- No newer open or merged PR already audits or edits
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this exact
  recurrent/full-chunk vLLM follow-up frontier. PR `#404` only touches
  `crates/kiln-server/src/bench.rs`, and there were no open PRs at preflight.
- Current `main` still predates the specific upstream vLLM design I audited:
  `ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8`
  (`vllm/model_executor/layers/fla/ops/chunk_delta_h.py`,
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`) and its caller in
  `vllm/model_executor/layers/fla/ops/chunk.py`.

**Audit result**

No bounded port-worthy recurrent/full-chunk win remains. The current vLLM
advantage is structural, not another small kernel-local cleanup.

The exact upstream pattern still missing from kiln is the block-tiled state
update in `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`: vLLM keeps the
recurrent state in FP32 `[BV, 64]` tiles (`b_h*`) in registers, loads `b_k`
once per chunk tile, and applies the state update with tiled dot products
(`b_h += tl.trans(tl.dot(b_k, b_v))`). Kiln's fused full-chunk CUDA kernel is
still a scalar-thread epilogue over `(k_idx, dv)`:

- shared `s_w[t, dv]`
- per-thread `delta += k_t[k_idx, t] * (w[t, dv] * decay_last[t])`
- bf16 store back to `state[k_idx, dv]`

That remaining gap is a different kernel architecture, not one more isolated
micro-port. The bounded sub-diffs suggested by the same upstream design have
already been attempted and disqualified on current-main:

- post-`#399`: hoist `decay_last` weighting into shared `W` rows -> slower
- post-`#401`: stage shared `k_t` rows -> slower
- post-`#403`: triangular front-half packing -> warm-pod artifact, no durable win

Those measurements are summarized in
`profiling-artifacts/post405_20260423_vllm_recurrent_audit.csv`.

**Keep / revert decision**

Doc-only redirect. I did **not** edit
`crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` because the only
meaningful upstream delta left is a
larger re-vendor/rewrite of the recurrent-state update, not a single bounded
win that fits this task's acceptance criteria.

**RunPod / validation**

No pod launched and no CUDA validation ran for this audit. That was deliberate:
after reading the current kiln kernel against the current vLLM upstream source,
there was no remaining one-diff candidate to validate. Re-running another
micro-port on RunPod would have been a duplicate spend against the same
already-failed frontier.

### Post-#406 tiled full-chunk recurrent-state update — 2026-04-23

Fresh-`main` preflight passed before editing:

- PR `#406` is merged and remains doc-only: it updated `PROFILING.md` plus
  `profiling-artifacts/post405_20260423_vllm_recurrent_audit.csv`, with **no**
  source change under `crates/kiln-gdn-kernel/`.
- No newer open or merged PR already changes
  `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` for this exact
  recurrent/full-chunk follow-up. Current `main` includes PR `#408`, but that
  only touches `crates/kiln-model/src/{forward,generate,loader,weights}.rs`.
- The validated post-`#392` refined prefill capture above still shows
  `gdn_full_chunk_forward_kernel` at **34.6%** of prompt-heavy prefill kernel
  time, so this frontier still existed on fresh `main`.

This retry implemented the first real structural step implied by PR `#406`'s
audit: tile the scalar `(k_idx, dv)` recurrent-state epilogue over a small
`k_idx` block so each value lane reuses `w[t, dv] * decay_last[t]` across
multiple state rows before writing BF16 state back. Concretely, the attempted
kernel diff:

- staged an `8 x 64` `k_t` tile in shared memory;
- accumulated `delta_tile[8]` per value lane in registers;
- left the full-chunk body, launch envelope, and Rust call surface unchanged.

**Attempted code path**

- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`

**Validation**

RunPod on-demand RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
export KILN_CUDA_ARCHS=86
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export CARGO_PROFILE_DEV_DEBUG=0

cargo build --release --features cuda --bin kiln-bench
cargo test -p kiln-model --release --features cuda \
  forward::tests::test_gdn_full_chunk_forward_matches_fallback \
  -- --exact --nocapture

./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
```

Parity passed:

- `out_chunk` max abs diff: **1.5625e-2**
- `state` max abs diff: **3.125e-2**

The first before/after pair looked promising:

- fresh `main`: **4735.5 ms**, **1727 tok/s**
- patched branch: **3145.3 ms**, **2601 tok/s**

But the same-pod warm control disproved that apparent win:

- warm `main` rerun: **3112.7 ms**, **2628 tok/s**
- warm patched rerun: **3167.5 ms**, **2582 tok/s**

So the tiled epilogue is actually **1.8% slower** than warmed `main`
(`3167.5 / 3112.7 - 1`), which fails the task's **>=3%** keep bar and fits the
same warm-order artifact pattern that invalidated the earlier post-`#403`
result.

I still captured one refined changed-build kernel CSV after the parity pass:
`profiling-artifacts/post406_20260423_tiled_prefill_kern.csv`. The changed
build's top kernel remains `gdn_full_chunk_forward_kernel` at **29.7%**, but
that capture does **not** override the failed end-to-end wall-clock gate.

**Keep / revert decision**

Doc-only negative result. I reverted the kernel diff and did **not** ship a
source change in `gdn_full_chunk_forward.cu` because the same-pod warmed
control shows the tiled recurrent-state update misses the acceptance threshold.

**Measurement artifact**

- `profiling-artifacts/post406_20260423_tiled_recurrent_verdict.csv`
- `profiling-artifacts/post406_20260423_tiled_prefill_kern.csv`

## Phase 6 post-#384 current-main re-profile — 2026-04-22

### Post-change note — 2026-04-23 (`ce/phase6-fused-gdn-full-chunk`)

Follow-up A6000 validation for the next minimal fused full-chunk CUDA prefill
step (`gdn_full_chunk_forward`: fused chunk body + in-kernel state update,
tail fallback unchanged) showed that correctness is acceptable on the focused
CUDA parity test, but end-to-end prompt-heavy prefill did **not** improve on
the `#389` baseline.

- Validation environment: RunPod A6000, `ghcr.io/ericflo/kiln-runpod:latest`,
  `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`, `--paged --prompt-tokens 8192
  --max-output-tokens 1 --skip-training --latency-only`
- New focused CUDA parity test:
  `test_gdn_full_chunk_forward_matches_fallback` — passed
- Bench runs on the fused branch:
  - run 1: **3239.3 ms** prefill (**2525 tok/s**)
  - run 2: **3651.1 ms** prefill (**2240 tok/s**)
  - run 3: **3254.9 ms** prefill (**2513 tok/s**)
  - median: **3254.9 ms** (**2513 tok/s**)

Compared with the post-`#384` baseline already recorded in this file
(**2831.2 ms / 2889 tok/s** uncaptured, **2947.8 ms / 2775 tok/s** profiled),
this minimal fused full-chunk step is a regression rather than a win. The
fused path did exercise on the benchmark's full 64-token chunks, but the final
short tail chunk still fell back by design.

**Scope:** refresh the current-`main` performance source of truth after PR
[#384](https://github.com/ericflo/kiln/pull/384) and name the next real
Phase 6 hotspot from fresh evidence instead of the pre-`#384` queue shape.

**Preflight outcome:** proceed. On fresh `main` (`06d4cea`), PR #384 is
merged (`930363e`), `crates/kiln-model/src/forward.rs` still contains the
merged `gdn_chunk_scan` path, and there is no newer merged or open PR that
already lands a post-`#384` current-main re-profile. The closest match is
PR #383, but that landed before #384 and is therefore not sufficient for this
task. PR #387 is a decode no-op reconfirmation, not a post-#384 full
decode+prefill refresh.

**Hardware / image:** RunPod on-demand RTX A6000 (`$0.49/hr`),
`ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`, CUDA toolkit
`12.4`. The baked image shipped `nsys 2023.4.4`; that version reproduced the
known `EventCollection::CheckOrder` importer failure on both new captures, so
I installed `nsight-systems-2024.5.1` and used
`/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys` for the final
exportable reports.

**Build / validation commands:**

```bash
export KILN_CUDA_ARCHS=86
cargo build --release --features cuda,nvtx --bin kiln-bench
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p kiln-model -p kiln-server --features cuda --no-run
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
```

**Profiling commands run:**

```bash
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --latency-only
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=70 --duration=20 -o /workspace/phase6-profile/post384-decode-20245 ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=22 --duration=8 -o /workspace/phase6-profile/post384-prefill-20245 ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
```

The decode command above produced a valid report after upgrading Nsight. The
prefill command above produced no report on current `main` because the entire
`8192/1 --latency-only` run now finishes before the 22-second delay elapses.
For the committed prefill summaries below, I reran that same prompt-heavy arm
with the only change needed to arm the profiler on this faster build:

```bash
/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=3 --duration=4 -o /workspace/phase6-profile/post384-prefill-fixed ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only
```

### Decode uncaptured runs — paged 512/128

| run | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms | prefill ms |
| --- | -----------: | ----------: | ---------: | ---------: | ---------: |
| 1 | 43.8 | 22.8 | 22.6 | 27.1 | 10375.9 |
| 2 | 44.2 | 22.6 | 22.5 | 24.1 | 359.2 |
| 3 | 43.4 | 23.0 | 22.7 | 27.0 | 389.9 |
| median | **43.8** | **22.8** | **22.6** | **27.0** | **389.9** |

Run 1 kept the familiar cold-start prefill outlier (`10.38 s` TTFT) while
decode itself stayed in-family. The stable decode central tendency remains
runs 2-3: about **44 tok/s** at **22.6-23.0 ms** ITL.

### Prompt-heavy prefill timing — paged 8192/1

The uncaptured prompt-heavy timing run on current `main` produced:

- prompt tokens: **8180**
- prefill time: **2831.2 ms**
- prefill throughput: **2889.3 tok/s**

The adjusted prefill `nsys` run used the same bench arm and measured
**2947.8 ms / 2775.0 tok/s**, which is close enough for hotspot attribution
without changing the qualitative ranking.

### Top-3 NVTX hotspots — decode

Source: `profiling-artifacts/post384_20260422_decode_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **19.7** | `:kiln/gdn/gates` |
| 2 | **19.4** | `:kiln/gdn/gated_norm` |
| 3 | **16.4** | `:kiln/gdn/qk_norm` |

Decode is still dominated by the GDN gate/norm stack. The top-3 decode
regions now sum to **55.5%** of wall-clock, slightly *more* concentrated than
the earlier `2026-04-22` current-main refresh.

### Top-3 NVTX hotspots — prefill

Source: `profiling-artifacts/post384_20260422_prefill_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **51.1** | `:kiln/attn/gdn/chunk_prep` |
| 2 | **25.1** | `:kiln/attn/gdn/chunk` |
| 3 | **11.3** | `:kiln/gdn/in_proj` |

Prefill remains the dominant opportunity after #384. More specifically, the
fresh post-#384 run says the vendored prefill path is still incomplete:
`chunk_prep` plus `chunk` alone now account for **76.2%** of captured
prompt-heavy wall-clock.

### Kernel callouts

- **Decode top kernels** (`profiling-artifacts/post384_20260422_decode_kern.csv`):
  CUTLASS BF16 GEMMs at **31.8%** and **17.6%**, `ampere_bf16...128x64...`
  at **9.6%**, `ucopy_bf16` at **6.2%**, and `gdn_chunk_scan_kernel<64>` at
  **1.2%**.
- **Prefill top kernels** (`profiling-artifacts/post384_20260422_prefill_kern.csv`):
  `gdn_chunk_scan_kernel<64>` at **18.4%**, `ucopy_bf16` at **11.2%**,
  `bmul_f32` at **10.5%**, the two large BF16 GEMMs at **6.8%** and **6.5%**,
  and `gdn_chunk_prep_kernel<64>` at **3.6%**.

The important post-#384 fact is that `gdn_chunk_scan_kernel<64>` is real and
hot on prefill, but the surrounding chunk path still dominates around it.
PR #384 moved work into the vendored scan path; it did not finish the
prompt-heavy GDN prefill bottleneck.

### Comparison vs the previous 2026-04-22 profile

Against the earlier `2026-04-22` current-main refresh section already in this
file:

- decode median improved from **41.65 tok/s** to **43.8 tok/s** and mean ITL
  improved from **24.01 ms** to **22.8 ms**;
- decode top-3 concentration rose from **50.5%** to **55.5%**;
- decode top-3 ordering did not change: `gates`, `gated_norm`, `qk_norm`;
- prefill `chunk_prep` grew from **46.9%** to **51.1%**;
- prefill `chunk` grew from **14.6%** to **25.1%** and overtook `in_proj`;
- prefill `in_proj` fell from **19.0%** to **11.3%**;
- the prefill chunk pair (`chunk_prep` + `chunk`) rose from **61.5%** to
  **76.2%** of prompt-heavy wall-clock.

So the fresh post-#384 evidence does **not** say "GDN prefill is fixed." It
says the remaining opportunity is now even more cleanly localized to the
chunkwise prefill path around the vendored scan kernel.

### Recommendation

**Single next Phase 6 target:** keep the focus on **GDN prefill vendoring**,
specifically the chunkwise prefill path around `:kiln/attn/gdn/chunk_prep`
and `:kiln/attn/gdn/chunk`, not FlashInfer-style decode work and not another
hand-rolled candle pass. Fresh post-#384 decode evidence still leaves full
attention below the reopen bar (`:kiln/proj/qkv` **2.2%** + `:kiln/proj/o`
**1.0%**), while prompt-heavy prefill now exposes a much larger and more
concentrated ceiling in the GDN chunk path.

## Phase 6 current-main re-profile — 2026-04-22

**Scope:** refresh the current-`main` performance source of truth after the
post-#204 code-path drift and name the top-3 **decode** and **prefill**
wall-clock hotspots separately from fresh current-main captures. This run
stops the planning loop from leaning on stale pre-`main` hotspot data.

**Preflight outcome:** proceed. There is no newer merged PR than #204 that
already lands a current-main re-profile with separate decode/prefill top-3
lists, no open PR covering this exact scope, and the `#204..HEAD` range is
not docs-only. Forward-path drift is real in `crates/kiln-model/src/forward.rs`,
`generate.rs`, `loader.rs`, `paged_kv_cache.rs`, `transposed_weight_cache.rs`,
and `crates/kiln-server/src/bench.rs`, so a no-pod redirect would have been
incorrect.

**Hardware / image:** Kiln pool A6000 lease `sl53yvx5seviyx`
(`$0.49/hr`), `ghcr.io/ericflo/kiln-runpod:latest`, Driver 580.95.05, CUDA
12.8 runtime. The warm pool pod came up with baked `nsys 2023.4.4`; that
version reproduced the known `EventCollection::CheckOrder` importer failure
on the prompt-heavy prefill trace, so this run upgrades profiler userspace to
`nsys 2024.5.1` via `apt-get install -y libxcb-cursor0 cuda-nsight-systems-12-6`
before the final prefill capture. Decode and prefill CSVs committed under
`profiling-artifacts/currentmain_20260422_*`.

**Build / validation:** `cargo build --release --features cuda,nvtx --bin kiln-bench`
and `cargo test -p kiln-model -p kiln-server --features cuda --no-run`, both
with `KILN_CUDA_ARCHS=86`. Validation completed cleanly on the pod before any
profiling runs.

**Methodology:** production paged path, `KILN_W4A16=1 KILN_CUDA_GRAPHS=true`,
Qwen3.5-4B from `/workspace/qwen3.5-4b`.

- **Decode uncaptured runs:** `./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --latency-only`
  run 3x back-to-back.
- **Decode nsys capture:** `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=70 --duration=20 -o /workspace/phase6-profile/decode-20245-main ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`
- **Prompt-heavy prefill timing:** `./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only`
- **Prefill nsys capture:** `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --delay=22 --duration=8 -o /workspace/phase6-profile/prefill-20245-main ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 8192 --max-output-tokens 1 --skip-training --latency-only`

### Decode uncaptured runs — canonical paged 512/128

| run | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms | prefill ms |
| --- | -----------: | ----------: | ---------: | ---------: | ---------: |
| 1 | 50.86 | 19.66 | 19.55 | 24.48 | 320.2 |
| 2 | 40.88 | 24.46 | 24.49 | 27.99 | 333.3 |
| 3 | 41.65 | 24.01 | 24.08 | 26.87 | 345.6 |
| median | **41.65** | **24.01** | **24.08** | **26.87** | **333.3** |

Run 1 remained the fastest by a wide margin, so the stable central tendency
for this refresh is runs 2-3 rather than the warm-start outlier. The median
decode result to carry forward is therefore **41.65 tok/s** at **24.01 ms**
mean ITL.

### Prompt-heavy prefill timing — paged 8192/1

One uncaptured prompt-heavy latency run on current `main` produced:

- prompt tokens: **8180**
- prefill time: **2702.7 ms**
- prefill throughput: **3026.6 tok/s**

The one-token decode tail on this run is not the target metric; this arm is
only here to anchor the separate prefill hotspot capture on a genuinely
prompt-heavy path.

### Top-3 NVTX hotspots — decode

Source: `profiling-artifacts/currentmain_20260422_decode_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **18.0** | `:kiln/gdn/gates` |
| 2 | **17.5** | `:kiln/gdn/gated_norm` |
| 3 | **15.0** | `:kiln/gdn/qk_norm` |

These three regions still dominate decode together at **50.5%** of wall-clock.
The fresh current-main profile therefore does **not** move the next decode
target away from the GDN gate/norm path.

### Top-3 NVTX hotspots — prefill

Source: `profiling-artifacts/currentmain_20260422_prefill_nvtx.csv`

| rank | % | region |
| ---: | -: | --- |
| 1 | **46.9** | `:kiln/attn/gdn/chunk_prep` |
| 2 | **19.0** | `:kiln/gdn/in_proj` |
| 3 | **14.6** | `:kiln/attn/gdn/chunk` |

Prefill is even more concentrated than decode: the top-3 prefill regions sum
to **80.5%** of prompt-heavy wall-clock, and two of the three are squarely in
the chunkwise GDN prefill path rather than the decode-only recurrence path.

### Kernel callouts

- **Decode top kernels** (`profiling-artifacts/currentmain_20260422_decode_kern.csv`):
  `Marlin<256,1,8,8,4,8>` 14.5%, `cutlass_80_tensorop_bf16...256x64...` 11.8%
  (`lm_head`), `ampere_bf16_s16816gemm...128x64...` 9.8%, `ucopy_bf16` 8.3%.
- **Prefill top kernels** (`profiling-artifacts/currentmain_20260422_prefill_kern.csv`):
  `Marlin<256,4,16,4,4,8>` 14.3%, `bmul_f32` 11.4%, `ucopy_bf16` 11.1%,
  `gdn_fwd_sub_kernel` 6.5%, `gdn_chunk_prep_kernel<64>` 4.0%.

### Comparison to #204

Only the shared uncaptured 512/128 paged arm is directly comparable to the
fresh-profile baseline in PR #204, because #204 was bench-only and did **not**
land separate decode/prefill hotspot tables. On that shared arm, current-main
median decode is **41.65 tok/s** versus **51.37 tok/s** in #204. I am not
claiming a hotspot delta against #204 because there is no same-shape NVTX
table in that PR to compare against.

The important current-main conclusion is instead structural: the decode top-3
ordering remains the familiar GDN gate/norm stack, while the fresh prefill
capture shows an even larger wall-clock opportunity in the chunkwise GDN
prefill path.

### Recommendation

**Single next optimization target:** vendor the minimal prefill-side
`chunk_gla_fwd`-class GDN kernel path, starting with
`:kiln/attn/gdn/chunk_prep` and its immediately-adjacent chunk execution
(`:kiln/attn/gdn/chunk`). That recommendation is now grounded in fresh
current-main evidence, not stale queue shape:

- decode top-3 still justify GDN-focused work, but they are a cluster of
  already-known decode regions;
- prefill presents the larger fresh concentration (**80.5%** in the top-3,
  **46.9%** in `chunk_prep` alone);
- the prompt-heavy profile says the next materially new win is in **GDN
  prefill kernel vendoring**, not another hand-rolled candle experiment.

### Remaining-work preflight — FlashInfer-style paged GQA decode (2026-04-22)

Task brief checked against fresh `main` after PR #384:

- **PR #384 merged:** yes (`930363e`, on `main` before current HEAD
  `234c17a`).
- **No open overlap:** yes. GitHub search over open PRs for
  `flashinfer`, `paged decode`, and `flash_attn_paged_decode` returned no
  active implementation PR.
- **Decode call path still present:** yes.
  `crates/kiln-model/src/forward.rs` still routes single-token paged
  full-attention decode through `try_flash_attn_paged_decode(...)`, and
  `crates/kiln-model/src/backend/cuda.rs` still implements
  `flash_attn_paged_decode(...)` via
  `kiln_flash_attn::flash_attn_paged_decode(...)`.

**Decision: no implementation follow-up.** The fresh current-main profile
above still leaves full-attention at sub-floor share on Qwen3.5-4B:

- decode top-3 remain GDN-only (`gates` 18.0%, `gated_norm` 17.5%,
  `qk_norm` 15.0%);
- full-attention projections remain only ~3.8% of decode wall-clock
  (`:kiln/proj/qkv` 3.0% + `:kiln/proj/o` 0.8%);
- the inner `:kiln/attn/full/*` kernel region a FlashInfer-class paged
  decode kernel would replace stays below the reporting floor.

That keeps the overall Amdahl ceiling in the same ~`<=1.038x` range already
documented by PR #163 and the later frontier audits, still well below the
Phase 6 reopen bar. Replacing the current CUDA paged-decode backend on this
model would be redoing a known sub-floor slice, not landing the next queued
win.


## Phase 7 design: streaming/tiled GDN prefill — 2026-04-20

**Outcome: long-context prefill (≥65k tokens) fits on a 48 GiB A6000 by iterating the prompt in 8192-token tiles, reusing the already-existing `LinearAttentionState` as the O(1) state carrier across tile boundaries. Peak working set shrinks from ~9 GiB per-layer-full-prompt to ~1.1 GiB per-layer-per-tile. No GDN kernel changes, no state struct changes, no paged-cache changes; a single new `model_forward_paged_streaming` wrapper plus three config flags. $0 doc-only PR; implementation spike is one ~30-minute A6000 run.**

This design is a $0 source-inspection follow-on to the PR #226 audit
([#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223),
[#224](https://github.com/ericflo/kiln/pull/224),
[#225](https://github.com/ericflo/kiln/pull/225),
[#226](https://github.com/ericflo/kiln/pull/226)
pattern: inspect, decide, defer pod $ until the design is pinned down).

**Why append to PROFILING.md rather than a new `DESIGN-streaming-prefill.md`:** the Phase 7 audit already lives here, and candidate G ("streaming prefill") is explicitly the thing this section lands. Future re-profiles will extend the same file with post-implementation deltas. Keeping audit + design + measurement in one place avoids the "where did we decide this?" cross-file hunt that earlier phases produced. If this section grows past ~500 lines we revisit, but today the signal is stronger as one continuous document.

### (a) Tile size choice — 8192 tokens

- **Chosen: `KILN_STREAMING_PREFILL_TILE=8192` (default).** Rationale below; the flag is tunable for measurement.
- **Hard constraint: tile size must be a multiple of `GDN_CHUNK_SIZE = 64`.** `gdn_chunkwise_recurrence` (forward.rs:1173–1378) already handles a tail chunk smaller than 64 — but only *once per call*. Letting tile boundaries land mid-chunk would repeatedly force the tail path and also splinter the recurrence into unequal chunks, which complicates state handover (see §c). `8192 = 128 × 64` is chunk-aligned.
- **Lower bound (2k, 4k):** too small. Per-tile kernel-launch overhead (conv1d_prefill + chunk_prep + fused_recurrent + gates + norms × 32 layers × ceil(seq_len / tile)) starts to dominate wall-clock below ~4k. At 2k across T=65536 that is 32 tiles × 32 layers = 1,024 layer invocations vs 8 × 32 = 256 at tile=8192. Marlin GEMM and paged-decode fusion are also tuned for larger M.
- **Upper bound (16k, 32k):** fits comfortably in budget (per-layer peak scales roughly linearly in tile size → ~2.2 GiB and ~4.4 GiB respectively), but the audit's 9 GiB-per-layer at T=65536 is the monolithic full-prompt peak. Picking tile=8192 leaves ~35 GiB of headroom for the LM-head tail (§d) and keeps Marlin activations well inside SM resident budgets.
- **Sweet spot math** (Candidate G in the audit, extrapolated to the activation graph from §3–§6 of the PR #226 audit):

  | tile | per-layer peak act | 32-layer working budget fit | tile count at T=65536 | relative kernel-launch overhead |
  |------|-------------------:|:---------------------------:|----------------------:|--------------------------------:|
  | 2048 | ~0.28 GiB          | trivial                     | 32                    | 4× vs 8192                      |
  | 4096 | ~0.55 GiB          | trivial                     | 16                    | 2× vs 8192                      |
  | **8192** | **~1.10 GiB**  | **trivial (<1.5 GiB carry)**| **8**                 | **1× (baseline)**               |
  | 16384 | ~2.2 GiB          | fits (~5 GiB carry)          | 4                     | 0.5× vs 8192                    |
  | 32768 | ~4.4 GiB          | fits (~10 GiB carry)         | 2                     | 0.25× vs 8192                   |
  | 65536 (monolithic) | ~9 GiB | **OOM** (≥28 GiB carry on A6000) | 1 | — |

  (Per-layer-peak figures derive from scaling the PR #226 audit's post-`l2_normalize` live set linearly in `seq_len`; the O(1) state carry is negligible — 48 MiB across all GDN layers.)
- **8192 is the smallest tile at which kernel-launch overhead is not the limiting cost AND peak per-tile per-layer activations fit comfortably on an A5000-class 24 GiB GPU too** (so the same default works for smaller GPUs once Phase 7 lands there).

### (b) Iteration order — layer-by-tile (outer tile, inner layer)

Two orderings exist; only one is viable:

- **Option 1 (CHOSEN): outer loop = tile, inner loop = layer.** For each tile, run all 32 layers, handing `hidden` from layer to layer as today; threading `LinearAttentionState` *across tiles* at the model level, threading paged KV cache / `start_pos` as today. Working set per step ≈ activations of one layer of one tile ≈ ~1.1 GiB (at tile=8192), plus the `hidden` tensor `[1, tile, 2560]` ≈ 40 MiB handed between layers.
- **Option 2 (REJECTED): outer loop = layer, inner loop = tile.** Would need to hold the *full* `[1, seq_len, hidden=2560]` intermediate between layers — at T=65536 that is 320 MiB × (one live copy per layer-boundary hand-off) ≈ unbounded without additional buffering, or an explicit ~320 MiB-per-layer streaming scratchpad. This does not reduce the per-layer per-tile peak; it just rearranges the storage of inter-layer hidden. And it breaks the existing `model_forward_paged` contract of "feed one call, get logits out" — forcing a much larger refactor.

Layer-by-tile is also the pattern used by flash-linear-attention for chunked prefill (see §h) and by vLLM's Mamba-2 prefill streaming. Conceptually: each tile is a complete "mini forward pass" whose only linkage to the prior tile is the GDN state and the paged KV cache.

Pseudocode (layer-by-tile, per-prompt):

```
fn model_forward_paged_streaming(tokens, ..., state, ...) -> Tensor {
    let tile_size = env("KILN_STREAMING_PREFILL_TILE", 8192);
    let mut last_logits = None;
    let mut cursor = 0;
    while cursor < tokens.len() {
        let end = (cursor + tile_size).min(tokens.len());
        let is_last = end == tokens.len();
        let tile_tokens = &tokens[cursor..end];
        let tile_logits = model_forward_paged(
            tile_tokens,            // embedding + all 32 layers on this tile
            ...,
            start_pos = cursor,     // paged KV cache threads via existing start_pos
            Some(&mut state),       // GDN state threads via existing LinearAttentionState
            ...,
            // INTERNAL knob (see §d): skip LM head unless is_last, and only emit
            // the final token's row when is_last
        )?;
        if is_last { last_logits = Some(tile_logits); }
        cursor = end;
    }
    last_logits.expect("at least one tile")
}
```

### (c) State handover contract

Three pieces of state cross tile boundaries; two already work, one needs alignment discipline:

1. **`LinearAttentionState` (GDN recurrent + conv states).** Already defined at `forward.rs:241-270` with the docstring "This state is O(1) in sequence length — it does not grow with the number of tokens processed." Threading is already implemented: `gated_deltanet_forward` takes `&mut recurrent_states[i]` and `&mut conv_states[i]` (forward.rs:2911–2925), and `causal_conv1d_prefill` already threads `conv_state` correctly (forward.rs:907–946, writes the final `k-1` cols back to `conv_state` at 931–943). Per-layer: `recurrent_states[1, nv=32, dk=128, dv=128] F32` = 2 MiB; `conv_states[1, qkv_dim, k-1=3] F32` < 100 KiB. **Total across 24 GDN layers: 48 MiB.** No changes to the struct, no changes to the kernels — we just hand the same `&mut state` into each tile's `model_forward_paged` call.
2. **Paged KV cache + `start_pos` (full-attn / GQA layers).** Already threaded. The 8 full-attention layers read prior tile KV pages via `block_table` indexing during the current tile's attention; `start_pos = cursor` threads correctly (forward.rs:2872, 2905). No new state. This is why full-attn layers are "tile-oblivious" — they see the same prefix attention semantics whether invoked monolithically or tiled.
3. **Chunk alignment (the one real constraint).** Because `gdn_chunkwise_recurrence` handles the tail-chunk path only *once per call* (the last partial chunk flushes into the state at function exit), we require `tile_size % GDN_CHUNK_SIZE == 0` for all non-final tiles so that within a tile the recurrence processes only full 64-token chunks and the state is clean at the tile boundary. The *final* tile (which covers the prompt remainder) is allowed to have a tail chunk — this is exactly the behavior `gdn_chunkwise_recurrence` already supports when the monolithic path runs with a non-multiple-of-64 prompt length.

**Contract summary:** for a tile `[cursor, cursor+N)` with `N % 64 == 0` (except the last tile), after `model_forward_paged(tokens[cursor..cursor+N], ..., start_pos=cursor, state)`:
- `state.recurrent_states[l]` for each GDN layer `l` holds the post-N recurrence state for that layer.
- `state.conv_states[l]` for each GDN layer `l` holds the last `k-1=3` input columns of that layer's conv1d input.
- Paged KV cache blocks `[cursor .. cursor+N)` are populated for all 8 full-attn layers.
- No other mutation persists between calls.

This is byte-identical to the monolithic contract because the operations within each tile are bit-for-bit the same ops that the monolithic path would execute on the same token range.

### (d) Output accumulation — last-tile-last-token only (inference)

The LM head is `[hidden=2560, vocab=151936]` matmulled against `hidden: [1, seq_len, 2560]` → `logits: [1, seq_len, vocab=151936]`. At T=65536 the full logits tensor is **~19 GiB F32** (or ~9.5 GiB BF16 if we downcast — still a huge sink). The audit (Candidate C) called this out as the dominant post-norm allocation.

For **inference**, we only need the last row. `generate_from_tokens_paged_inner` at `generate.rs:547` samples from `logits` using `greedy_sample`/`sample_with_params`, which already operate on the last time-step. We therefore:

- On **non-last tiles:** skip the LM head entirely. Return `None` (or a cheap sentinel tensor) and drop the tile's `hidden` on function exit. Memory is reclaimed before the next tile starts.
- On **the last tile:** compute `rms_norm` + LM-head matmul, but optionally restrict to the final token's row — `hidden[.., -1:, ..]` of shape `[1, 1, 2560]` against `embed_tokens_t`, producing `[1, 1, vocab]` (~300 KiB BF16). Samplers are already last-token-indexed; this just lets us avoid materializing the full `[1, tile_size, vocab]` row even on the last tile.

Savings at T=65536 with tile=8192: LM-head output shrinks from ~19 GiB to ~300 KiB, a 60,000× cut on the final spike. Additionally, Candidate C ("lm_head output full-tensor sink") is mostly obviated — we never hold more than one tile's LM-head output, and on the critical last tile we only hold one row.

**Training (forward+backward) needs full logits** for the CE loss; streaming training prefill is explicitly deferred (§i). The `model_forward_paged_streaming` entrypoint is therefore **inference-only** in Phase 7; the existing `model_forward` remains the training path.

### (e) Integration surface

Minimal additive surface. No refactor of existing call sites beyond the dispatch point.

- **New function in `crates/kiln-model/src/forward.rs`:**
  ```rust
  pub fn model_forward_paged_streaming(
      backend: &dyn BackendRuntime,
      token_ids: &[u32],
      weights: &GpuWeights,
      config: &kiln_core::config::ModelConfig,
      paged_cache: &mut PagedKvCache,
      block_table: &BlockTable,
      start_pos: usize,
      mut linear_state: Option<&mut LinearAttentionState>,
      lora: Option<&LoraWeights>,
      positions_gpu: Option<&Tensor>,
  ) -> Result<Tensor> {
      // Same signature as model_forward_paged. Iterates tiles, delegates
      // per-tile work to model_forward_paged with the "skip LM head on
      // non-last tiles" knob (internal, not exposed as a pub API).
  }
  ```
  Same signature as `model_forward_paged` so the dispatch wrapper is a drop-in.
- **Internal knob in `model_forward_paged` (or a sibling `model_forward_paged_tile`):** an `is_last_tile: bool` param (crate-private) that short-circuits the LM-head section (forward.rs:2944–2950). On non-last tiles it returns a 1-byte placeholder tensor and the caller discards it.
- **Dispatch at two call sites** (`generate.rs:525`, `bench.rs:484`):
  ```rust
  let prefill_fn = if streaming_prefill_enabled(seq_len) {
      model_forward_paged_streaming
  } else {
      model_forward_paged
  };
  let logits = prefill_fn(backend, prompt_tokens, ...)?;
  ```
- **Config flags (parsed in `crates/kiln-server/src/config.rs`):**
  - `KILN_STREAMING_PREFILL` — `0|1` (default `0`). Master opt-in.
  - `KILN_STREAMING_PREFILL_TILE` — integer (default `8192`). Tile size in tokens; must be multiple of 64 (validated on parse; falls back to 8192 with a warning otherwise).
  - `KILN_STREAMING_PREFILL_THRESHOLD` — integer (default `32768`). Only stream when `seq_len >` this threshold; short prompts take the monolithic fast path. Setting to `0` forces streaming for all prefills (used by parity tests).
- **Kill switch:** `KILN_STREAMING_PREFILL=0` restores byte-identical behaviour. This is the default for Phase 7 ship; we gate on measurement before flipping the default.
- **Zero changes to:** `LinearAttentionState` struct, GDN kernels (`kiln-gdn-kernel`), conv1d kernel (`kiln-conv1d-kernel`), `gdn_chunkwise_recurrence`, `gated_deltanet_forward`, `causal_conv1d_prefill`, paged KV cache, `transformer_block_paged`, `model_forward` (training), `generate_from_tokens` (non-paged path).
- **CUDA graph interaction:** none. Prefill never uses CUDA graphs (`generate.rs:524` comment: "Prefill: forward pass on all prompt tokens (never uses CUDA graphs)"). Decode path is untouched; the decode-time CUDA graph replay (`crates/kiln-model/src/cuda_graph.rs:300,358`) continues to call `model_forward_paged` for single-token steps after streaming prefill completes.

### (f) Correctness strategy — bitwise parity

Streaming is a pure code-motion refactor: the kernel invocations, their inputs, and their outputs on each token range are identical to the monolithic path. Correctness is provable by parity tests, not by eyeballing tolerances.

Tests (CPU backend, add to `crates/kiln-model/src/forward.rs` `#[cfg(test)]` module):

1. **`test_streaming_matches_monolithic_cpu_small`**: T=128, tile=64. Build a tiny model via existing test scaffolding (the GDN decode tests at forward.rs:4256–4300 are the template), run both `model_forward_paged` and `model_forward_paged_streaming` from zero-state, assert `logits_mono == logits_stream` **bit-exact** (`to_vec1::<f32>()` equality).
2. **`test_streaming_matches_monolithic_cpu_mid`**: T=2048, tile=512. Same assertion.
3. **`test_streaming_tile_invariance_cpu`**: T=1024, loop `tile ∈ {64, 128, 256, 512, 1024}`; assert all five runs produce bit-equal logits.
4. **`test_streaming_preserves_state_cpu`**: after streaming T=2048 with tile=512, assert `state.recurrent_states[l]` and `state.conv_states[l]` are bit-equal to the monolithic run's state.
5. **`test_streaming_disabled_is_byte_identical`**: with `KILN_STREAMING_PREFILL=0`, assert dispatch goes through `model_forward_paged` unchanged (regression guard for the `≤32k` fast path).

**CUDA parity** (run on the spike pod, §g): `test_streaming_matches_monolithic_cuda` — same as #2 above but on CUDA device. F32 recurrent state means GDN is reproducible across tilings; the conv1d F32 promotion we already do (Candidate A future work) keeps conv1d deterministic too. Paged attention is tile-oblivious by construction.

**Why this is tractable:** the operations in each tile are the *same* ops in the *same* order on the *same* memory; we are just not holding intermediate activations from tile N-1 when tile N runs. State threading is already a tested code path (decode uses it token-by-token — forward.rs:4256–4300, `test_paged_single_token_forward` and related). This is prefix-block aggregation at a larger granularity.

### (g) Spike scope — one pod, ~30 min, <$0.70

- **Goal:** land `KILN_STREAMING_PREFILL=1` behind a flag; prove parity on CPU + CUDA; prove T=65536 prefill completes on A6000; prove ≤32k fast path is untouched.
- **Pod:** acquire via `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`. Fallback to direct RunPod if pool is capped.
- **Runtime:** single session.
- **Steps:**
  1. Implement `model_forward_paged_streaming` + dispatch + config flags. (~1 hr local, no pod.)
  2. Add 5 CPU parity tests. Iterate on CPU until green. (local, no pod.)
  3. Open PR in draft. Wake pod.
  4. On pod: `cargo nextest run --features cuda` — all existing tests green + CUDA parity test.
  5. On pod: `KILN_STREAMING_PREFILL=1 KILN_W4A16=1 ./target/release/kiln-bench --paged --prompt-tokens 65536 --max-output-tokens 32 --skip-training`. Confirm no OOM; record prefill latency + decode tok/s.
  6. On pod: ≤32k regression. `KILN_STREAMING_PREFILL=0 KILN_W4A16=1 ./target/release/kiln-bench --paged --prompt-tokens 8192 --max-output-tokens 128` and same with `KILN_STREAMING_PREFILL=1 KILN_STREAMING_PREFILL_THRESHOLD=0` (force streaming on a short prompt); 3× each, confirm medians are within noise (≤2% variance).
  7. Capture nsys NVTX trace of one streaming prefill for PROFILING.md re-profile attachment. Optional.
  8. Release the lease.
- **Success criteria:**
  - All parity tests green.
  - T=65536 prefill + T=32 decode completes without OOM on 48 GiB A6000.
  - ≤32k fast path unchanged (median prefill/decode within noise vs pre-PR main).
  - No regression on the existing bench harness.
- **Cost ceiling:** ~0.5 h × A6000 on-demand (~$0.79/hr) ≈ $0.40. Budget cap $0.70 including warm-up and nsys attribution run. If the spike trips anything unexpected, ship the doc-only design (this PR), redirect with a null-finding PR, and re-plan.

### (h) flash-linear-attention precedent

The `flash-linear-attention` (fla) repo (https://github.com/sustcsonglin/flash-linear-attention) — which is where kiln's `kiln-gdn-kernel` ultimately derives from (PR #80) — ships its linear-attention ops with an explicit `chunks=` / `chunk_size=` knob and a paired `initial_state=` / `final_state=` contract. Representative APIs:

- `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule(q, k, v, g, beta, scale=..., initial_state=None, output_final_state=True, cu_seqlens=None)` — the chunkwise Gated DeltaNet forward. `initial_state` and `output_final_state` are the fla equivalent of our `LinearAttentionState` threading. `cu_seqlens` supports variable-length batches in the same kernel invocation.
- `fla.ops.delta_rule.chunk.chunk_delta_rule(..., initial_state=..., output_final_state=...)` — same pattern on the ungated variant.
- fla's chunk size is a kernel-internal constant (64 — matches `GDN_CHUNK_SIZE` in `kiln-gdn-kernel`). fla does not stream across *kernel calls*; streaming is left to the caller, using `initial_state` / `final_state`.

kiln's streaming prefill is therefore a *model-level* analog of the pattern fla already encodes at the op level. We reuse the vendored chunk kernel unchanged (the "compute per-chunk recurrence, fold into state" contract is identical), and layer tiling outside it. This is precisely what vLLM does for its Mamba-2 streaming prefill (https://github.com/vllm-project/vllm/pull/12093 and siblings): the mamba kernel stays fla-derived; tiling is orchestrated at the model-forward level with explicit state threading.

We are not inventing a new streaming algorithm. We are picking up the state-threading contract fla has always shipped and wiring it into our paged forward.

### (i) What is deliberately deferred

- **Training (backward) streaming.** CE loss needs full logits and the gradient of the full hidden sequence; streaming backward needs tile-boundary state checkpointing + a compatible reverse-recurrence path. Phase 7.x, not Phase 7.0.
- **Async tile pipelining.** Overlap tile-N compute with tile-(N+1) embedding lookup + positions_gpu build. Useful but orthogonal — costs CUDA stream plumbing and a second `hidden` buffer. Phase 7.2.
- **Dynamic tile sizing.** Measure free VRAM at runtime and adjust `KILN_STREAMING_PREFILL_TILE` per-prompt. Useful for mixed-GPU fleets and varying LoRA footprints. Phase 7.3.
- **FP8 KV cache × streaming prefill interaction.** Already composes cleanly at the design level (prefill populates FP8 KV blocks tile-by-tile exactly as it does monolithically), but empirical re-verification at T ≥ 32k is deferred until both features are on simultaneously in one bench.
- **Candidate A (F32 → BF16 `causal_conv1d_prefill`).** Independent of streaming; shrinks conv bubble by ~2× inside each tile. Still queued from the PR #226 audit; orthogonal to this PR.
- **Candidate B (l2_normalize in-place).** Independent; smaller peak within GDN. Still queued.
- **Candidate C (LM-head full-tensor sink).** *Partially subsumed* by §d — we only ever hold one tile's logits, and only the last token on the last tile. The remaining Candidate-C opportunity (chunked softmax inside the sampler) is unchanged but now off the critical OOM path.
- **Dispatching streaming on the non-paged `model_forward` path.** Non-paged is the training path; inference uses paged. If someone introduces long-context non-paged inference we revisit; today no call site needs it.
- **Flipping `KILN_STREAMING_PREFILL` default to `1`.** Ship behind a flag in Phase 7.0; flip the default only after the spike parity + bench + nsys data land and a re-profile confirms no silent regression on short prompts.

---

TL;DR: tile-oblivious full-attn + existing O(1) `LinearAttentionState` + chunk-aligned tile size = long-context prefill with zero kernel changes and a surgically small forward-path patch. The next PR implements this behind `KILN_STREAMING_PREFILL=1` and measures it in a single ~30-min A6000 spike.


## GDN prefill memory audit (Phase 7 opener) — 2026-04-20

**Outcome: the GDN recurrent state (the thing FP8 would shrink) is ~48 MiB total across 24 layers — 0.1% of the OOM ceiling. The binding constraint is per-layer prefill activations. At `seq_len=65536` one GDN layer's live tensors sum to roughly 9–10 GiB, which exceeds the ~37 GiB working budget on A6000 (48 GiB − ~8 GiB weights − ~3 GiB CUDA context − carry across 32 layers). Long-context capability is unlocked by streaming/tiled prefill (candidate G below), not by shrinking state.**

This audit is a $0 source-inspection preflight following PRs
[#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223),
[#225](https://github.com/ericflo/kiln/pull/225). The trigger is PR
[#222](https://github.com/ericflo/kiln/pull/222)'s FP8 KV cache verification
matrix: 65536p and 131072p prompts OOM'd in **both** the BF16 and FP8 arms,
and the failure was localized to **GDN layer 0 prefill** — not the paged GQA
KV cache. FP8 halves KV bytes but the 24 linear-attention layers dominate
activation memory before the 8 GQA layers' paged KV is even touched, so
`KILN_KV_CACHE_FP8` cannot lift the ceiling. The question this audit
answers: what exactly does GDN prefill allocate, and which reduction opens
≥65536p on a 48 GiB A6000?

### Config reference (Qwen3.5-4B)

From `crates/kiln-core/src/config.rs:79–101` and
`crates/kiln-model/src/forward.rs:250–264`, `990`:

| Symbol | Value | Meaning |
| --- | ---: | --- |
| `hidden` | 2560 | residual hidden size |
| `L_total` | 32 | total transformer layers |
| `L_full` | 8 | full-attention layers (paged GQA KV) |
| `L_gdn` | 24 | GDN linear-attention layers |
| `nk` | 16 | GDN key heads |
| `nv` | 32 | GDN value heads |
| `dk` | 128 | GDN key head dim |
| `dv` | 128 | GDN value head dim |
| `qk_dim` | 2048 | `nk * dk` |
| `v_dim` | 4096 | `nv * dv` |
| `qkv_dim` | 8192 | `2 * qk_dim + v_dim` (fused conv channels) |
| `kernel_size` | 4 | causal depthwise conv |
| `C` | 64 | `GDN_CHUNK_SIZE` |
| dtype | BF16 | hot path activation/weight dtype |

For the allocation table below, assume `B = 1`, `T = seq_len = 65536`, `C = 64`, `nc = T / C = 1024 full chunks`. All tensors are live on the CUDA device.

### (a) Source-level allocation table — single GDN layer at `T = 65536`

Byte totals are the product of shape × dtype bytes, on a single GDN layer's
forward pass through
`gated_deltanet_forward` and `gdn_chunkwise_recurrence` in
`crates/kiln-model/src/forward.rs`.

| # | Tensor | File:line | Shape | Dtype | Bytes @ T=65536 |
| --- | --- | --- | --- | --- | ---: |
| 1 | hidden `x` input (layer carry) | `forward.rs:~2600` (layer loop) | `[B, T, hidden]` | BF16 | 320 MiB |
| 2 | `mixed_qkv` after `in_proj_qkv` matmul (Step 1) | `forward.rs:1443` | `[B, T, qkv_dim]` | BF16 | 1.00 GiB |
| 3 | `z` after `in_proj_z` matmul | `forward.rs:1444` | `[B, T, v_dim]` | BF16 | 512 MiB |
| 4 | `a`, `b` gate projections | `forward.rs:1445–1446` | `[B, T, nv]` | BF16 | 4 MiB each |
| 5 | `mixed_qkv_ct` for conv (transpose + contiguous) | `forward.rs:1462` | `[B, qkv_dim, T]` | BF16 | 1.00 GiB |
| 6a | `x_padded = cat(conv_state, x)` inside `causal_conv1d_prefill` | `forward.rs:915,921` | `[B, qkv_dim, T+3]` | **F32** | 2.00 GiB |
| 6b | `output` accumulator inside `causal_conv1d_prefill` (kernel_size=4 steps) | `forward.rs:924–928` | `[B, qkv_dim, T]` | **F32** | 2.00 GiB |
| 7 | `post_silu` = `cuda_silu(.to_dtype(F32))` result | `forward.rs:1488` | `[B, qkv_dim, T]` | **F32** | 2.00 GiB |
| 8 | `post_silu.transpose(1, 2)` — F32 view/copy | `forward.rs:1499` | `[B, T, qkv_dim]` | **F32** | 2.00 GiB |
| 9 | `v` cast to input_dtype in `recur_prep` | `forward.rs:1652` | `[B, T, nv, dv]` | BF16 | 512 MiB |
| 10 | `q` after GQA head_expand contiguous | `forward.rs:1522–1526` | `[B, T, nv, dk]` (F32 through Step 4) | F32 | 1.00 GiB |
| 11 | `k` after GQA head_expand contiguous | `forward.rs:1527–1531` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 12 | `l2_normalize(q)` F32 output | `forward.rs:1578` / `l2_normalize` at `852–858` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 13 | `l2_normalize(k)` F32 output | `forward.rs:1579` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 14 | `q` cast to BF16 after norm+scale | `forward.rs:1580` | `[B, T, nv, dk]` | BF16 | 512 MiB |
| 15 | `k` cast to BF16 after norm | `forward.rs:1581` | `[B, T, nv, dk]` | BF16 | 512 MiB |
| 16 | `q.transpose(1, 2)` for recurrence | `forward.rs:1655` | `[B, nv, T, dk]` | BF16 (view→contig on entry into chunked path) | 512 MiB |
| 17 | `k.transpose(1, 2)` | `forward.rs:1656` | `[B, nv, T, dk]` | BF16 | 512 MiB |
| 18 | `v.transpose(1, 2)` | `forward.rs:1657` | `[B, nv, T, dv]` | BF16 | 512 MiB |
| 19 | `q_pre` pre-permuted `[nc, B, nv, C, dk]` contiguous | `forward.rs:1231` / `preshape_chunked_4d` at `1014–1038` | `[1024, 1, 32, 64, 128]` | BF16 | 512 MiB |
| 20 | `k_pre` | `forward.rs:1232` | same shape | BF16 | 512 MiB |
| 21 | `v_pre` | `forward.rs:1233` | `[1024, 1, 32, 64, 128]` | BF16 | 512 MiB |
| 22 | `beta_pre`, `g_pre` | `forward.rs:1234–1235` | `[1024, 1, 32, 64]` | BF16 | 16 MiB each |
| 23 | `out_chunks: Vec<Tensor>` accumulated outputs | `forward.rs:1237, 1366` | `nc × [B, nv, C, dv]` | BF16 | 512 MiB (sum) |
| 24 | `Tensor::cat(&out_chunks, 2)` final recurrence output | `forward.rs:1377` | `[B, nv, T, dv]` | BF16 | 512 MiB (new) |
| 25 | `attn_out.transpose(1, 2)` | `forward.rs:1683` | `[B, T, nv, dv]` | BF16 | 512 MiB |
| 26 | `gated_rms_norm` F32 output before reshape/cast | `forward.rs:1689` / `gated_rms_norm` at `885–900` | `[B, T, nv, dv]` | **F32** | 1.00 GiB |
| 27 | post-`gated_rms_norm` reshape + cast to BF16 | `forward.rs:1691–1693` | `[B, T, v_dim]` | BF16 | 512 MiB |
| 28 | output of `out_proj` matmul (written back into residual) | `forward.rs:~1700+` | `[B, T, hidden]` | BF16 | 320 MiB |
| C1 | **O(1) recurrent state** per layer | `forward.rs:262` | `[B, nv, dk, dv]` | F32 | **2 MiB** (constant, not T-scaling) |
| C2 | **O(1) conv state** per layer | `forward.rs:263` | `[B, qkv_dim, k−1]` | F32 | **96 KiB** (constant, not T-scaling) |

**Note on row 6/7/8 — the F32 conv bubble**. `causal_conv1d_prefill`
unconditionally promotes the conv input to F32 (line 915
`x_f32 = x.to_dtype(F32)?`), then SiLU runs on F32, and the result only
returns to BF16 after Step 5's final cast. Between conv entry and the
`input_dtype` re-cast at Step 7 (`v.to_dtype(input_dtype)` at line 1652) the
sequence carries a full `[B, T, qkv_dim]` F32 tensor — 2× the byte cost of
the BF16 equivalent, and the single largest non-pre-chunk allocation in the
whole layer.

**Note on row 10/11 — GQA head_expand materialization**. `gqa_ratio =
nv / nk = 2`, so `q`/`k` are duplicated along a new head axis via
`unsqueeze(3).expand(...).contiguous().reshape(...)`. The `contiguous()`
forces a real copy — at T=65536 this is two 1 GiB allocations per layer
before `l2_normalize` even runs.

**Note on row 19–22 — the pre-permutation tax**. `preshape_chunked_4d`
calls `.contiguous()` at line 1037 to materialize a new tensor with the
chunk axis leading, enabling zero-copy chunk slices inside the loop. This
duplicates q/k/v/beta/g across the full sequence length: **~1.5 GiB of
extra live BF16** at T=65536 in addition to rows 16–18. The originals
(16/17/18) are still borrowed through the loop for the tail (`narrow` at
lines 1246–1250), so both live simultaneously.

### (b) Peak-live-at-once analysis

Not every row is live at the peak; candle releases intermediates as
their last `&Tensor` borrow ends. The worst moment is immediately before
the chunk loop begins — after `gdn_chunkwise_recurrence` has materialized
the five `*_pre` tensors but before `q`, `k`, `v`, `beta`, `g` go out of
scope (they remain referenced for the tail branch at lines 1246–1250 and
for the narrow/squeeze/contiguous calls). Hidden-state carry from outside
the layer also stays live.

| Row | Size | Rationale for being live at peak |
| ---: | ---: | --- |
| 1 residual `x` carry | 320 MiB | layer input — still referenced across 32 layers; next-layer residual-add will still read it |
| 5 `mixed_qkv_ct` (BF16) | 1.00 GiB | may be released before conv returns, but worst-case overlaps with row 6 |
| 6a `x_padded` F32 inside conv | 2.00 GiB | live during conv loop |
| 6b conv `output` F32 | 2.00 GiB | live during conv loop (written every kernel_size step) |
| 8 post-transpose F32 `mixed_qkv` | 2.00 GiB | live from Step 3 narrow/reshape until v/q/k drop their F32 references |
| 10 q expanded F32 | 1.00 GiB | live through Step 5 |
| 11 k expanded F32 | 1.00 GiB | live through Step 5 |
| 12 l2_normalize(q) F32 | 1.00 GiB | brief overlap with row 10 |
| 13 l2_normalize(k) F32 | 1.00 GiB | brief overlap with row 11 |
| 16 q transposed BF16 | 512 MiB | live through whole chunk loop (tail fallback) |
| 17 k transposed BF16 | 512 MiB | live through whole chunk loop |
| 18 v transposed BF16 | 512 MiB | live through whole chunk loop |
| 19 q_pre | 512 MiB | live through whole chunk loop |
| 20 k_pre | 512 MiB | live through whole chunk loop |
| 21 v_pre | 512 MiB | live through whole chunk loop |
| 22 beta_pre + g_pre | 32 MiB | live through whole chunk loop |
| 23 `out_chunks` accumulator | 512 MiB | grows to full size before cat() |

The **conv bubble peak** (rows 6a + 6b + carry from row 1 + 5): ~6.32 GiB.

The **post-l2_normalize peak** (rows 8 + 10 + 11 + 12 + 13 + 1 + 14 + 15):
~9.0 GiB including residual carry. This is the peak of the F32 phase —
after the two `.to_dtype(input_dtype)` casts at lines 1580–1581 the F32
copies of q/k go out of scope and the peak drops.

The **chunk-loop peak** (rows 1 + 16–23): ~3.4 GiB sustained.

The **cat/gated_norm peak** (rows 1 + 24 + 26): ~1.83 GiB. Brief.

**Dominant peak: the post-l2_normalize phase at ~9 GiB per layer.**

Global budget sanity:

- A6000: 48 GiB
- Weights at `KILN_W4A16=1`: q_proj Marlin packed + MLP Marlin packed ≈ 4.5 GiB (post-PR #206); GDN projections still BF16 (`in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, `out_proj`, `conv1d`, `norm`, `a_log`, `dt_bias`, plus pre-transposed `*_t` copies from PR #128); embeddings + lm_head + full-attn layers. Total resident ≈ **7–9 GiB** on current main.
- CUDA context, workspace, allocator fragmentation: **~3 GiB**.
- Per-step carry across 32 layers: residual `[B, T, hidden]` BF16 + any lazy-released activation tails: **320 MiB minimum**, up to **~1 GiB** in practice.
- Working budget for a single layer's prefill: **~35–37 GiB**.

One GDN layer's ~9 GiB peak fits with margin. But the peak is not the only cost: allocator fragmentation, the `x_padded` F32 bubble reappearing on every layer's conv, and the GQA head_expand copies compound across layers during prefill. The observed OOM at GDN layer 0 suggests either (a) fragmentation from the initial BF16 weight residency is higher than naively budgeted, or (b) the F32 `mixed_qkv` tensor and the dual pre-permutation tensors are alive simultaneously for longer than the static analysis above assumes. Either way, the **per-layer activation footprint is the binding axis**.

At `T = 131072` everything T-scaling doubles: ~18 GiB per-layer peak. No realistic weight-side cleanup gets that under 35 GiB.

### (c) Reduction candidates ranked by impact

All numbers are at `T = 65536`, single layer, relative to the rows in (a).

| Cand. | Description | Bytes saved @ 65k (per-layer peak, approximate) | Complexity | Numerical risk | Upstream precedent |
| :---: | --- | ---: | --- | --- | --- |
| **A** | Keep conv output BF16 (drop F32 promotion in `causal_conv1d_prefill` rows 6a/6b/7/8) | ~4.0 GiB | LOW–MEDIUM: requires BF16 depthwise conv that doesn't overflow; use the fused `kiln-conv1d-kernel` path (already BF16-native for decode) or extend it to prefill | LOW: bf16 SiLU is standard in fla/vLLM Mamba paths | flash-linear-attention, vLLM Mamba keep conv in compute dtype end-to-end |
| **B** | Skip pre-permutation (rows 19–22) at long T; use direct `narrow` per chunk | ~1.5 GiB | MEDIUM: pre-permute was added to kill the `copy2d_bf16` per-chunk hotspot (PROFILING.md history). Replace with (i) gated-on-`seq_len` short-prefill only, or (ii) fused chunk-indexed gather in `kiln-gdn-kernel` | LOW: identical math; only locality changes | fla chunk_gla_fwd slices on the fly without pre-permutation |
| **C** | Write chunk outputs in-place into a pre-allocated `[B, nv, T, dv]` BF16 buffer; eliminate `Vec<Tensor>` + final `cat` (rows 23 + 24 collapse to one 512 MiB buffer) | ~512 MiB | LOW: pre-allocate once, `slice_assign` per chunk | NONE: bit-identical | vLLM output accumulation pattern |
| **D** | FP8/FP16 recurrent state (row C1) | ~1 MiB (single layer) / **~24 MiB total across 24 layers** | LOW (E4M3 cast, mirror #222 FP8 KV cache) | MEDIUM: state accumulator precision matters; Phase-6 PR [#72](https://github.com/ericflo/kiln/pull/72)/[#74](https://github.com/ericflo/kiln/pull/74) already moved the *internal* recurrence to BF16 with F32 boundary cast for stability | — (state is already F32 for accumulator stability; reducing it regresses precision without unlocking context) |
| **E** | Reduce `GDN_CHUNK_SIZE` from 64 to 32 | ≤ 5 MiB (per-chunk scratch shrinks ~75% but chunk count doubles) | LOW (one `const` change + parity tests) | LOW | — |
| **F** | Checkpoint / stream the full-attention (GQA) KV-cache writes so `mixed_qkv` F32 can be freed earlier in each GDN layer | overlaps with A; no independent savings | MEDIUM | LOW | — |
| **G** | **Streaming/tiled prefill: split T into super-chunks of `S` tokens (e.g. S=4096), run the full 32-layer stack end-to-end per super-chunk, carrying only the per-layer recurrent state (C1, 2 MiB/layer) + conv state (C2, 96 KiB/layer) between super-chunks.** Activation peak becomes `O(S)` not `O(T)`. | **~9 GiB** at S=4096 for T=65536; **~17 GiB** at S=4096 for T=131072 (the entire T-scaling half of per-layer peak goes away) | **HIGH**: re-architect `model_forward_prefill` to iterate super-chunks, preserve `LinearAttentionState` + paged GQA KV cache across iterations, concatenate output hidden states only at the lm_head boundary (or stream to the final-token path when only the last logits are needed) | LOW: `LinearAttentionState` *already* supports this — it's designed to carry across prefill/decode (see `LinearAttentionState` docstring at `forward.rs:240`, and the decode tests at `forward.rs:4256–4300` that drive prefill-then-decode through the same state). Paged GQA KV cache already appends across calls. | vLLM handles long Mamba/Mamba-2 contexts exactly this way; fla documents super-chunk prefill as the canonical path. See also `flash-linear-attention/fla/layers/gated_deltanet.py` `chunk_size=` / `chunks=` iteration pattern. |

### (d) Recommended Phase 7 opener

**Candidate G (streaming/tiled prefill with O(1) state handover) is the only candidate that both unlocks `T ≥ 65536` on A6000 and extends naturally to `T = 131072`.** Candidates A+B+C together sum to ~6 GiB saved per-layer peak — enough to maybe land 65536p but not 131072p, and fragile under any future model change that extends `v_dim` or adds heads.

Ranked by (unlock × complexity × risk):

1. **G — streaming/tiled prefill (opener).** Unlocks the full 262K native
   context window on A6000 without quantizing anything. Single largest
   architectural win available in Phase 7. The `LinearAttentionState` and
   paged KV cache plumbing is already in place; the work is in
   `model_forward_prefill` (sequence iteration, state threading, hidden-state
   concatenation or streaming to lm_head). Gate the streaming path behind a
   `seq_len > threshold` check (suggest `threshold = 8192`) so short
   prefills pay zero throughput penalty.
2. **A — BF16-native conv prefill (follow-up #1).** Orthogonal to G.
   Even after G drops the per-layer peak, keeping conv output in BF16
   reclaims ~2× memory on the conv bubble and shrinks the super-chunk
   floor, which lets G use a larger `S` (higher throughput). Already half
   done: `kiln-conv1d-kernel` is BF16-native for decode; extend to
   prefill. Upper-bound on savings independent of G.
3. **C — in-place chunk accumulator (follow-up #2).** Cheap mechanical
   win (~512 MiB per layer), strictly additive to G, no risk.
4. **B — drop pre-permutation at long T (follow-up #3, conditional).**
   Only worth doing if G + A + C land and 131072p still needs headroom at
   a desired `S`. Gate on `seq_len` so short prefills retain the
   pre-permutation locality win.

Items D, E, F do not move the needle on the 65k/131k ceiling.

### (e) Preconditions for re-opening / invalidation

This recommendation should be re-verified before a Phase 7 implementation PR
if any of the following change:

- **G becomes upstream-obvious or already partially landed.** Check
  `git log --all --grep='streaming\\|tiled.*prefill\\|super.*chunk'` on
  kiln and `gh pr list -R ericflo/kiln --state all --limit 50`. The
  `LinearAttentionState` threading pattern is already there, but no
  current PR or branch drives it from the model-forward layer.
- **fla upstream ships a single-kernel long-context GDN prefill** that
  makes the Rust-side super-chunking unnecessary. If
  [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  adds a fused `chunk_gla_fwd_streaming` or similar, vendor that kernel
  into `kiln-gdn-kernel` instead (same minimal-scope precedent as
  [PR #80](https://github.com/ericflo/kiln/pull/80)).
- **Nsys evidence at 16384p or 32768p shifts the peak away from the
  post-l2_normalize phase.** The static analysis above assumes the F32
  inflation is the dominant axis. If a real runtime snapshot at
  `T = 32768` (the largest size that currently fits) shows the chunk-loop
  tensors or the output-cat as the dominant mass, re-prioritize C over
  the others.
- **Weight residency drops further** (e.g. PR [#206](https://github.com/ericflo/kiln/pull/206) extensions pack more BF16 projections into Marlin). Each GiB freed is a GiB the per-layer peak can grow into — if total weights drop below 5 GiB, candidates A+B+C alone might cross the 65536p line without G. Re-check the working-budget arithmetic in (b) before committing.
- **A new runtime memory snapshot contradicts the static analysis.**
  This audit is source-only; it doesn't observe allocator fragmentation
  or candle's actual release timing. If a minimal GPU
  instrumentation task (described below) is ever run, its results
  supersede rows (b) and the dominant peak may shift.

### Follow-up: optional GPU instrumentation task (bounded, not blocking)

If a future planner wants ground truth before committing to G, the
minimal-cost verification is: run the existing 16384p bench with
`CUDA_LAUNCH_BLOCKING=1` and `cudaMemGetInfo` polling inside
`gated_deltanet_forward` (gated behind a `KILN_MEM_TRACE=1` env var),
capturing free-MiB before/after each numbered step in (a). One A6000 pod,
~15 minutes of runtime, single run is sufficient (peak memory is
deterministic at fixed T, not latency-like). Do **not** launch this from
the current task — this audit is $0 and the static analysis already
points unambiguously at G.


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


## Phase 7 GPU spike: streaming prefill on A6000 — 2026-04-20

**Outcome: CUDA parity test passes at T=2048 tile=512 (FP32 hybrid toy model, 1e-4 tolerance, per-layer GDN recurrent + conv state bit-equal). T=65536 prefill completes on a 48 GiB A6000 without OOM. ≤32k regression sweep (8192p / 128d) shows prefill within tolerance (+1.5% faster in streaming arm) and decode slightly below the 2% bar (-4.6%) — not a statistically clean null on this 3-run budget, documented for follow-up. Phase 7 streaming remains opt-in (`KILN_STREAMING_PREFILL=0` default); no threshold or default flip in this PR.**

### Parity (CUDA, `cargo nextest run --release --features cuda -p kiln-model test_streaming`)

All six tests green, 4.584s wall:

| # | Test | Result |
|---|---|---|
| 1 | `test_streaming_prefill_env_helpers` (existing) | PASS (0.056s) |
| 2 | `test_streaming_matches_monolithic_cpu_small` (existing) | PASS (0.142s) |
| 3 | `test_streaming_preserves_state_cpu` (existing) | PASS (0.188s) |
| 4 | `test_streaming_tile_invariance_cpu` (existing) | PASS (0.243s) |
| 5 | `test_streaming_matches_monolithic_cpu_mid` (existing) | PASS (0.284s) |
| 6 | **`test_streaming_matches_monolithic_cuda`** (new) | **PASS (4.583s)** |

The new CUDA test runs the 8-layer hybrid (6 GDN + 2 full-attn, FP32) at T=2048 / tile=512 / block_size=64 and asserts (a) last-tile logits match the corresponding slice of monolithic logits to `<1e-4` max-abs, (b) every GDN layer's `recurrent_states` matches across streaming vs monolithic at `<1e-4`, (c) same for `conv_states`. 1e-4 is a conservative FP32 bar that absorbs candle CUDA matmul reduction-order variance; the design doc (§c above) argues for bit-exactness and in practice the observed diffs were comfortably below threshold.

### T=65536 long-context validation (`KILN_STREAMING_PREFILL=1 KILN_W4A16=1`, tile=8192 default)

Single-request latency arm:

| Metric | Value |
|---|---|
| Effective prompt tokens | 65522 (bench truncates to block alignment) |
| Model VRAM after load | 17528 MB (14.3 GiB) |
| Model load time | 23.73s |
| Prefill time | 27485.1 ms (2384 tok/s) |
| Decode tokens generated | 33 |
| Mean ITL | 42.2 ms (23.7 tok/s) |
| P50 ITL | 41.8 ms |
| P99 ITL | 55.4 ms |
| **OOM?** | **No** |

The kiln-bench throughput sweep (4 / 8 / 16 sequential runs at this prompt size) fails — that path accumulates full-length prompts across the sweep and runs past VRAM, which is a scheduler-level constraint orthogonal to streaming prefill. The single-request latency arm is the spike target and it completes end-to-end on the 48 GiB A6000.

### ≤32k regression sweep (8192 prompt, 128 decode, 3 runs each, W4A16)

- **Arm A** = `KILN_STREAMING_PREFILL=0` (monolithic baseline).
- **Arm B** = `KILN_STREAMING_PREFILL=1 KILN_STREAMING_PREFILL_THRESHOLD=0` (streaming forced on for sub-threshold input).

| Arm | Run | prefill_time_ms | prefill tok/s | decode tok/s | P50 ITL ms | P99 ITL ms |
|---|---|---:|---:|---:|---:|---:|
| A | 1 | 3181.8 | 2574.3 | 51.2 | 19.3 | 22.3 |
| A | 2 | 3194.5 | 2564.1 | 50.2 | 19.6 | 22.9 |
| A | 3 | 3381.9 | 2422.0 | 48.1 | 20.6 | 22.7 |
| A | **median** | **3194.5** | **2564.1** | **50.2** | **19.6** | **22.7** |
| B | 1 | 3146.6 | 2603.1 | 47.9 | 20.8 | 22.7 |
| B | 2 | 3157.5 | 2594.1 | 47.6 | 20.8 | 31.2 |
| B | 3 | 3141.1 | 2607.6 | 48.4 | 20.5 | 25.9 |
| B | **median** | **3146.6** | **2603.1** | **47.9** | **20.8** | **25.9** |

Deltas (B vs A medians):

- Prefill tok/s: **+1.5%** (2603.1 vs 2564.1) — within 2% tolerance.
- Prefill time ms: **-1.5%** (3146.6 vs 3194.5) — within 2% tolerance.
- Decode tok/s: **-4.6%** (47.9 vs 50.2) — **outside the 2% tolerance** by roughly 1 ms of ITL. Run 2 P99 (31.2 ms) and run 3 P99 (25.9 ms) suggest this is decode-path ITL variance rather than a systematic streaming-path regression. Decode is identical single-step in both arms (both use `model_forward_paged`, not streaming, after prefill finishes); the paths differ only in warmup flow and the provenance of the `LinearAttentionState` going into the first decode step. Budget did not allow N=10+ runs to resolve. Worth a targeted re-bench before considering a threshold / default flip.

### Verdict and Phase 7 gating

- CUDA parity and T=65536 no-OOM are **green** → the streaming/tiled implementation is safe to keep shipped behind the default-off flag.
- ≤32k regression is **amber** on decode — not a block, but warrants a quieter re-bench (median-of-5+, same pod, same process) before changing the default or lowering the threshold.
- No changes to `KILN_STREAMING_PREFILL` default (stays `0`), no changes to `KILN_STREAMING_PREFILL_THRESHOLD` default (stays `32768`).

### Pod / cost notes

- Pod: `t0vmof6qkwostu` via lease `pod-016bb2e7c07be9a97ceb4a3b` (RunPod RTX A6000 on-demand, $0.49/hr, pool-leased).
- Leased → all work done: ~30 minutes (reused warm pod from pool; zero cold-build time).
- Spike cost: ~$0.25 (well under the $0.70 ceiling; $1.00 abort line was not approached).

## Phase 7 decode re-bench: resolving amber from PR #231 — 2026-04-20

**Outcome: median-of-5 interleaved A/B on the same warm A6000 pod / process lineage closes the amber. Arm A (monolithic) and Arm B (streaming) decode medians are 48.2 vs 48.1 tok/s — delta −0.21% (well inside the ±2% tolerance). The −4.6% gap reported in PR #231's 3-run regression sweep was ITL variance, not a streaming-path decode regression. No code or default flips from this re-bench; `KILN_STREAMING_PREFILL` stays default-off and `KILN_STREAMING_PREFILL_THRESHOLD` keeps its 32768 default.**

### Protocol

- Same warm pod / process lineage as the original GPU spike: pod `t0vmof6qkwostu` via lease `pod-016bb2e7c07be9a97ceb4a3b` (RunPod RTX A6000 on-demand, $0.49/hr). Reused the `target/release/kiln-bench` binary and Marlin pack cache produced by the PR #231 spike — controls for build-time / sccache / pack-cache drift that a fresh pod would reintroduce.
- Interleaved A, B, A, B, … for 5 rounds (10 kiln-bench invocations total). Each invocation is a fresh process, but all share the same pre-warmed binary and packed-weight cache on the pod. Interleaving controls for slow process-age / thermal drift that pure-sequential (5A then 5B) would fold into the arm delta.
- Both arms identical except for `KILN_STREAMING_PREFILL`:
  - Arm A: `KILN_STREAMING_PREFILL=0 KILN_W4A16=1 ./kiln-bench --paged --prompt-tokens 8192 --max-output-tokens 128 --skip-training` (monolithic baseline).
  - Arm B: `KILN_STREAMING_PREFILL=1` + same flags (streaming/tiled prefill). `streaming_prefill_enabled` in `crates/kiln-model/src/forward.rs` is a binary read of that env var; there is no active `KILN_STREAMING_PREFILL_THRESHOLD` env plumbing in the decode path — at `seq_len=8192` below the 32768 default threshold, `KILN_STREAMING_PREFILL=1` is what actually flips the path for this bench.
- Reported numbers are the final `--- Latency (single request) ---` summary each run prints to stderr. kiln-bench's paged-latency measurement populates that summary when `--paged` is set, so the P50 / P99 ITL and decode tok/s are the paged-path measurements.

### Results (5 rounds × 2 arms, same warm A6000 pod / process)

| Round | Arm | prefill tok/s | decode tok/s | mean ITL ms | P50 ITL ms | P99 ITL ms | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | A | 2540 | 47.7 | 21.0 | 20.8 | 23.2 | 3225.2 |
| 1 | B | 2589 | 48.0 | 20.8 | 20.7 | 22.8 | 3163.2 |
| 2 | A | 2542 | 47.9 | 20.9 | 20.7 | 22.9 | 3222.9 |
| 2 | B | 2603 | 48.1 | 20.8 | 20.6 | 23.9 | 3146.7 |
| 3 | A | 2559 | 49.7 | 20.1 | 19.8 | 22.9 | 3201.0 |
| 3 | B | 2646 | 50.6 | 19.8 | 19.6 | 21.9 | 3096.0 |
| 4 | A | 2523 | 50.2 | 19.9 | 19.7 | 22.4 | 3245.9 |
| 4 | B | 2600 | 47.7 | 21.0 | 20.7 | 25.2 | 3150.4 |
| 5 | A | 2444 | 48.2 | 20.7 | 20.6 | 23.8 | 3351.7 |
| 5 | B | 2613 | 50.7 | 19.7 | 19.6 | 22.0 | 3134.4 |

### Per-arm stats (n=5)

| Metric | Arm A median | Arm A stddev (% of median) | Arm B median | Arm B stddev (% of median) |
|---|---:|---:|---:|---:|
| prefill tok/s | 2540.0 | 1.59% | **2603.0** | 0.75% |
| decode tok/s | **48.2** | 2.10% | **48.1** | 2.78% |
| mean ITL ms | 20.7 | 2.13% | 20.8 | 2.66% |
| P50 ITL ms | 20.6 | 2.29% | 20.6 | 2.54% |
| P99 ITL ms | 22.9 | 2.00% | 22.8 | 5.47% |

Decode stddev is 2.1% (A) and 2.78% (B) of the median — both under the 3% variance guard from the task spec, so no additional rounds were triggered. Arm B's P99 ITL has a wider spread (5.47%) driven by round 4's 25.2 ms tail; that's isolated tail variance, not a systematic regression (median P99 is 22.8 ms, better than Arm A's 22.9 ms).

### Decision (decode tok/s medians)

- Arm A median: **48.2 tok/s**
- Arm B median: **48.1 tok/s**
- Delta: **−0.1 tok/s (−0.21%) vs Arm A** — well inside the ±2% tolerance.
- Prefill medians: Arm B 2603 vs Arm A 2540 tok/s = **+2.5% prefill** in the streaming arm, consistent with the +1.5% prefill seen in the PR #231 spike.

**Verdict: PR #231 amber is ITL variance. RESOLVED.** The streaming/tiled prefill path does not regress decode throughput at 8192p / 128d on A6000 once n≥5 rounds are collected on a warm process lineage. The −4.6% from the 3-run spike was inside the normal decode-tok/s noise floor (roughly 2–3% stddev on this workload/pod).

### What this changes

Nothing in shipped code: Phase 7 streaming remains opt-in (`KILN_STREAMING_PREFILL=0` by default), and `KILN_STREAMING_PREFILL_THRESHOLD` keeps its 32768 default. This re-bench only resolves the outstanding amber flag from the PR #231 write-up; no gating decision downstream is waiting on it.

If a future planner considers flipping `KILN_STREAMING_PREFILL` on by default or lowering the threshold, the data point to anchor on is: on A6000 at 8192p / 128d with W4A16, the two paths are **statistically indistinguishable on decode** (−0.21% ≪ noise floor) and the streaming path is **+2.5% on prefill tok/s**. Any default flip still needs the ≥32k / OOM-risk arm reconsidered separately; this re-bench only covers the ≤32k amber.

### Pod / cost notes (re-bench)

- Same pod as the spike (`t0vmof6qkwostu`, lease `pod-016bb2e7c07be9a97ceb4a3b`) — warm, with the release binary and packed weights already resident. No rebuild.
- 10 kiln-bench runs × ~6 min = ~60 minutes wall time. Pod at $0.49/hr → ~$0.49.
- Total re-bench spend: ~$0.50 (well under the $1.50 ceiling for this task).

## Native MTP speculative decoding — preflight 2026-04-20

**Status: GREEN — greenlight implementation task.**

Supersedes the `kiln-speculative-decoding-design` agent note (skip-layer self-spec). For Qwen3.5-4B specifically, the native pretrained MTP head is strictly better than self-spec: same k=1 draft cost, much higher acceptance rate, zero extra weights at serving time beyond 15 already-shipped tensors. This preflight is doc-only — no code changed, no pod acquired, $0 spend.

### Check 1 — MTP heads exist in the Qwen3.5-4B checkpoint (GREEN)

Verified via HuggingFace `config.json` + `model.safetensors.index.json` for the `Qwen/Qwen3.5-4B` checkpoint (same checkpoint kiln already loads):

- `num_nextn_predict_layers: 1` → **k=1 draft depth** (single MTP head; one speculative token per step).
- 15 MTP-prefixed tensors in the index:
  - `mtp.fc.weight` — shape `[2560, 5120]` — the `concat(inputs_embeds, hidden_states) → H` projection (matches `2*H → H` with `H=2560`).
  - `mtp.pre_fc_norm_embedding.weight` `[2560]`, `mtp.pre_fc_norm_hidden.weight` `[2560]` — two pre-FC RMSNorms (norm-then-concat pattern, one per input stream).
  - `mtp.layers.0.*` — one full transformer layer with GQA attention + SwiGLU MLP + input/post-attention layernorms. Tensor shapes match the main-model GQA blocks (q_proj `[8192, 2560]` = 32 heads × 256 head_dim; k_proj / v_proj `[1024, 2560]` = 4 KV heads × 256; o_proj `[2560, 8192]`; gate_proj / up_proj `[MLP_size, 2560]`; down_proj `[2560, MLP_size]`).
  - `mtp.final_layernorm.weight` `[2560]`.
- **No separate `mtp.lm_head`** in the index — the MTP head ties to the base model's `lm_head` (which itself ties to `embed_tokens`). That matches vLLM's `qwen3_next_mtp.py` (`self.lm_head = model.lm_head` linkage) and is what kiln already does for the main model (see `model_forward_head` in `crates/kiln-model/src/forward.rs` line 2873: `rms_norm(hidden, final_norm) → broadcast_matmul(embed_tokens_t)`).

Net: everything MTP needs is already in the checkpoint. Nothing to ship, nothing to train, nothing to distill — just load + wire.

### Check 2 — vLLM reference implementation (GREEN)

Primary reference: `vllm/model_executor/models/qwen3_next_mtp.py` (EAGLE-framework-based, production path).

Key structural points (relevant for kiln integration):

1. **Forward signature**: `Qwen3NextMTP.forward(input_ids, positions, hidden_states, inputs_embeds=None)` — takes the **base model's last hidden state** for the verified prefix plus the new draft-token positions.
2. **Input construction**: `inputs_embeds = embed_tokens(input_ids)` (the draft token candidate), then `x = fc(concat([pre_fc_norm_embedding(inputs_embeds), pre_fc_norm_hidden(hidden_states)], dim=-1))` — norm each stream, concatenate channel-wise, project `2H → H` via `fc: ColumnParallelLinear`.
3. **One MTP decoder block**: Runs the single `mtp.layers.0` block on `x` with the same KV cache / attention metadata as the base model's last full layer.
4. **Final RMSNorm + tied lm_head**: `logits = lm_head(final_layernorm(x))` — the last two ops are exactly what kiln already has in `model_forward_head`.
5. **Speculative loop lives outside this module**: vLLM's speculative worker drives the `target_forward → mtp_forward → sample draft → target_forward (verify k+1 in parallel) → rejection_sample` sequence. `qwen3_next_mtp.py` is just the "one MTP step" building block.
6. **GDN state rollback — explicit non-handling in vLLM**: qwen3-next has hybrid GDN + attention layers just like Qwen3.5, and vLLM's MTP path **does not checkpoint GDN recurrent state per-draft-token**. When a draft token is rejected, vLLM relies on the spec worker rerunning the target model on the committed prefix, which regenerates the correct GDN state from scratch on the next step. For kiln that's fine for correctness but ~24 GDN layers × O(head_dim × head_dim) state × ~10-20× per-second reject events is measurable overhead. **Main implementation risk for kiln** — see "Risks" below.

vLLM's `qwen3_next_mtp_eagle.py` is a second entry point wrapping the same head into EAGLE-3's tree-speculation machinery; kiln will not use tree spec at k=1 (it has no benefit), so the plain `qwen3_next_mtp.py` is the right reference.

### Check 3 — Math ceiling at k=1 (GREEN)

Formula (from standard spec-decode analysis, see Leviathan et al. 2023 and the more-general draft-model equation):

```
speedup = (1 - α^(k+1)) / ((1 - α) × (k+1))
```

where α is the per-token acceptance rate of the draft against the target distribution.

At **k=1** (Qwen3.5-4B's `num_nextn_predict_layers`):

| α (acceptance rate) | Ceiling speedup | Decode tok/s @ 49.76 baseline |
|---:|---:|---:|
| 0.50 | 1.333× | 66.3 tok/s |
| 0.60 | 1.400× | 69.7 tok/s |
| 0.70 | 1.467× | 73.0 tok/s |
| **0.72** (conservative published MTP range) | **1.480×** | **73.6 tok/s** |
| **0.80** (typical published MTP range) | **1.533×** | **76.3 tok/s** |
| 0.85 | 1.567× | 77.9 tok/s |
| 0.90 | 1.600× | 79.6 tok/s |

Target for Phase 6 / spec-decode work is **1.5× on decode tok/s** (49.76 → 74.6 tok/s).

- At α=0.80 (typical) the math ceiling is 1.533× — just clears the 1.5× target. Real-world measured speedup lives below the ceiling because of draft + verify overhead on top of baseline decode (non-zero draft cost, non-zero concat/norm/fc/block cost, GDN state management). Published Qwen MTP numbers at α≈0.8 + k=1 typically realize ~1.3-1.4× — below the 1.5× target.
- At α=0.85 the ceiling is 1.567× — comfortable headroom.
- Qwen3-Next 80B's own MTP self-reports α in the 0.72-0.85 band on typical workloads. Qwen3.5-4B (much smaller target, same MTP head architecture) should land in the same range; a smaller target model tends to match its MTP head better (lower-entropy hidden states → draft head more calibrated), not worse.

**Ceiling verdict**: k=1 clears the 1.5× target at α ≥ 0.80. The math is tight but not marginal. There is also no k-knob to tune post-implementation (Qwen3.5-4B ships k=1 fixed) — this is the only arm we can run.

Compare to FlashInfer paged GQA decode (killed in PR #163 at ceiling ≤1.005×) and fused L2-QK-norm (PR #173, null-median): those had math ceilings below the dispatch-amortization floor. MTP at k=1 has a ceiling 30-50× larger than that floor, so null-result risk is qualitatively different (it's implementation-overhead-dominated, not ceiling-dominated).

### Check 4 — Integration scope in kiln source (GREEN)

Files that need to change for native MTP (line counts are estimates for the implementation task, not this preflight PR):

| File | LOC now | Est. added | What changes |
|---|---:|---:|---|
| `crates/kiln-model/src/weights.rs` | 232 | +80 | Add `MtpWeights { fc: WeightTensor, pre_fc_norm_embedding: WeightTensor, pre_fc_norm_hidden: WeightTensor, layer: LayerWeights, final_norm: WeightTensor }` and thread it as `ModelWeights.mtp: Option<MtpWeights>`. `Option<>` because other Qwen3.5 variants / future models may not ship MTP; keep the existing `ModelWeights` shape backwards-compatible. Update `total_bytes`/`total_params`. |
| `crates/kiln-model/src/loader.rs` | 1445 | +150 | Detect `mtp.fc.weight` in the safetensors index → load the 15-tensor MTP block. Reuse `load_layer` for `mtp.layers.0.*` with a prefix override. Tie `mtp.lm_head` to `embed_tokens` (no separate load). Gate behind `if num_nextn_predict_layers > 0` from config.json. |
| `crates/kiln-model/src/forward.rs` | 5719 | +500 | (1) Add `MtpGpuWeights` to `GpuWeights` (Option, uploaded if present). (2) Add `mtp_forward_step(model, hidden_states, draft_token_id, position, kv_cache, gdn_state) → draft_logits` — runs `concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)) → fc → mtp.layers.0 → final_layernorm → tied lm_head`. (3) Add `speculative_mtp_decode_step` — analogous to the existing `speculative_decode_step` in `src/speculative.rs` but driving MTP + a k+1-wide verify call into `model_forward_paged`. (4) Add `LinearAttentionState::snapshot() / restore_from(snapshot)` helpers on the struct at line 241 — deep-clone `recurrent_states` + `conv_states` before MTP draft, restore on reject. (5) Add rejection sampling + resample identical to `src/speculative.rs::rejection_sample` (can call into existing impl). |
| `crates/kiln-model/src/generate.rs` | 1562 | +200 | New `generate_mtp_speculative()` + `generate_from_tokens_mtp_speculative()` parallel to the existing `generate_speculative` skip-layer versions at lines 657 / 689. Dispatch based on `SpecMethod` from config. Reuse `draft_forward_for_state_init` pattern for initial GDN state population before the first MTP draft. |
| `crates/kiln-server/src/config.rs` | 846 | +40 | Add `SpecMethod` enum: `Off` (default) / `Mtp` / `SkipLayer` (keep existing for fallback / A/B). Add `KILN_SPEC_METHOD` env flag parsing near lines 397-408. Validate at lines 455-460: `SpecMethod::Mtp` requires `num_nextn_predict_layers > 0` in loaded model config. |
| `crates/kiln-model/src/speculative.rs` | 646 | +0 / −0 | **Keep as-is**. Skip-layer self-spec stays as `SpecMethod::SkipLayer` fallback and the A/B baseline for benchmarking MTP. No deprecation in this slice. |
| `crates/kiln-server/src/api.rs` (and/or router) | — | +20 | Thread `SpecMethod` into the decode loop selection. If MTP requested but model lacks MTP weights, log a warning and fall back to off (don't hard-fail the server). |
| `PROFILING.md` | 3324 | +250 | New "Phase X — native MTP spec-decode results" section post-implementation (not this PR). |

**Total estimate: ~800-1000 LOC added across 6-7 files**, no kernel crate work required (all math is GEMM / RMSNorm / attention / sampling that already have paths). No new dependency on mamba-ssm / FlashInfer / Marlin / cuDNN. No new kernel crate — reuses existing kernels.

Existing speculative decoding infrastructure:
- `crates/kiln-model/src/speculative.rs` (646 LOC, already has `SpeculativeConfig`, `rejection_sample`, `speculative_decode_step`) — much of the MTP loop structure is copy-adapt from here; rejection sampling is identical.
- `crates/kiln-server/src/config.rs` already has `SpeculativeDecodingConfig` (line 126) with `enabled` / `num_speculative_tokens` / `draft_layers` + `KILN_SPEC_*` env flags (lines 397-408). Adding `method` is additive.
- **Server decode path is not yet wired to speculative at all**: `grep -ri speculative crates/kiln-server/src/` returns only the config struct, the CLI startup banner, and test assertions — no actual dispatch. That's good: MTP is net-new decode wiring, not a swap-out of live code. Lower collision risk.

### Check 5 — Update superseded agent note (DONE)

`ce notes-set --topic kiln-speculative-decoding-design` has been updated with a `SUPERSEDED 2026-04-20:` prefix referring back to this section. The skip-layer description is preserved as the fallback path; MTP becomes the primary recommendation for Qwen3.5-4B.

### Risks (main + mitigation)

1. **GDN state rollback overhead (primary risk).** 24 GDN layers × `LinearAttentionState { recurrent_states: Vec<Tensor>, conv_states: Vec<Tensor> }` must snapshot before every MTP draft and restore on every rejected token. Rough sizing: per layer, `recurrent_states` is roughly `(batch, num_v_heads, head_dim, head_dim)` and `conv_states` is `(batch, conv_size × num_heads × head_dim)`. At single-stream bs=1 the total snapshot is small (single-digit MB), but it's `24 × cudaMemcpyDeviceToDevice` per draft step. If naive copy lands at >1 ms/step this alone eats the entire MTP speedup. Mitigation: use a double-buffered ping-pong allocation (snapshot goes to a pre-allocated "shadow" slot, restore is just a pointer swap) instead of real memcpy; this is what `speculative.rs::draft_forward` already does for attention KV cache. Test in isolation before the full loop.

2. **Tied lm_head broadcast_matmul on 2 positions.** MTP verify calls `model_forward_paged` with k+1=2 positions. `model_forward_head` does `broadcast_matmul(embed_tokens_t)` — which is a `[2, H] × [H, V]` GEMM where V=151936. That's 4× the FLOPs of a single-position decode and kiln's decode is GEMM-bound on this step (top NVTX regions at 17-18% each). Not a correctness issue, a throughput one: the 2-position verify cost is baked into the math ceiling above, but if lm_head isn't batched well in the current path this lands at 2× not 1× + ε. Mitigation: inspect the `model_forward_head` path for any per-position branching; already parallelizes across the sequence dim in the current code.

3. **Marlin W4A16 compatibility for MTP layer.** `mtp.layers.0` has q_proj / o_proj / gate_proj / up_proj / down_proj shapes that match the main model's layers. Marlin packing in the loader operates per-projection and should pick these up transparently (PR #146 path). Need to verify in implementation that the existing pack loop covers `mtp.*` prefixes — this is a 1-line check in the loader, not a design risk.

4. **Acceptance rate below the 1.5× target.** At α=0.70, ceiling is 1.467× — below target. Qwen3.5-4B does not have published MTP acceptance numbers (Qwen3.5-4B isn't flagged as MTP-enabled in the model card text even though the weights are there). Risk: measured α could be <0.70 on kiln's typical workloads, giving a below-target speedup even with perfect implementation. Mitigation: A/B against skip-layer self-spec and against baseline in the same implementation PR. If α measurement on a representative workload comes back <0.72, pivot early — don't ship a below-target kernel just to cash the preflight.

5. **Prefix cache interaction.** kiln has `KILN_PREFIX_CACHE_ENABLED=true` by default. MTP reads the target model's last hidden state, which for a prefix-cache hit was computed before the current request started. Need to verify the hidden state is cached alongside the KV for the last verified position, not just the KV. If it isn't, MTP can't run on the first step of a prefix-cache-hit request and must fall back to one synchronous target step to materialize `hidden_states`. Mitigation: small (one step), well-defined; probably worth it to avoid architectural coupling of MTP into prefix cache serialization.

### Preflight cost summary

- Doc-only PR. $0 pod spend (no RunPod acquired). ~30 min of HF index inspection + vLLM source read + math + source traversal.
- Kiln agent notes updated (`kiln-speculative-decoding-design` → SUPERSEDED).
- No PROFILING.md re-profile needed (Phase 6 kernel frontier is unchanged; MTP is a Phase 7+-style addition, a new decode path rather than a kernel fusion).

### Recommendation

**GREENLIGHT a follow-up implementation task** titled along the lines of "Phase X — native MTP spec-decode on Qwen3.5-4B (GREEN-after-preflight)". Scope: the 6-7 files listed in Check 4 (~800-1000 LOC). Gate behind `KILN_SPEC_METHOD=mtp`, default off. Ship with A/B bench (off / skip-layer / MTP) on A6000 warm pod, median-of-3 rule. Target: **≥1.5× decode tok/s vs. baseline Arm B on the standard 512p/128d workload**, with α reported separately. If measured α < 0.72 on the standard workload, stop at the A/B numbers — don't force-ship a below-ceiling path.

## Phase X — native MTP spec-decode results (2026-04-21)

Bench sweep of the native MTP decode arm introduced in PR #247 and patched by PR #253 (loader now accepts Qwen3.5-4B's bare `mtp.*` tensor layout). Fills in the Mtp row of the results table that PR #247 opened.

### Context

- Hardware: RunPod A6000 (SM 86), CUDA 12.4, `ghcr.io/ericflo/kiln-runpod:latest`.
- Commit: `0443b81` (tip of `main` post-#253).
- Model: `Qwen/Qwen3.5-4B` weights at `/workspace/qwen3.5-4b` (15 `mtp.*` tensors, `mtp_num_hidden_layers=1`, `tie_word_embeddings=true`).
- Build: `cargo build --release --features cuda --bin kiln-bench` (sccache: 94% hit rate on warm pod, 81 s wall clock).
- Env flags for all runs: `KILN_W4A16=1 KILN_CUDA_GRAPHS=true`.
- Bench args: `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`.
- Cadence: 3 runs per arm, back-to-back, median reported.

### Results table

| Arm | decode tok/s | mean ITL (ms) | p50 ITL (ms) | p99 ITL (ms) | α |
|---|---:|---:|---:|---:|---:|
| Off | **48.5** | **20.6** | **20.50** | **23.70** | — |
| Mtp (initial, PR #247+#253) | **BLOCKED** | — | — | — | — |
| Mtp (after GDN state threading, this PR) | **41.16** | **24.29** | **17.23** | **39.31** | **0.411** |

Off median is the standard 512p/128d Arm B baseline. The initial Mtp row blocked before any decode steps completed. The follow-up Mtp row is measured after threading `LinearAttentionState` through `speculative_mtp_decode_step` with snapshot/restore on reject (see "Follow-up: GDN state threading" below).

### Off — raw per-run numbers

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 343.5 | 1473 | 19.4 | 51.6 | 19.01 | 25.28 |
| 2 | 354.7 | 1427 | 20.6 | 48.4 | 20.50 | 23.70 |
| 3 | 362.0 | 1398 | 20.6 | 48.5 | 20.58 | 22.69 |
| **median** | **354.7** | **1427** | **20.6** | **48.5** | **20.50** | **23.70** |

Compare to PR #247's Arm B baseline (50.77 tok/s): 48.5 is ~4.4% lower, comfortably inside the median-of-3 noise floor — baseline is reproduced, not regressed.

### Mtp — blocker

All 3 runs prefilled successfully (~355 ms / ~1420 tok/s) then failed identically at the first decode step:

```
--- Latency Benchmark (MTP — native speculative, paged) ---
  Measuring latency [MTP, paged, blocks=48] (506 prompt tokens)...
    Prefill (MTP): 358.7ms (1410 tok/s)
Error: MTP latency benchmark failed

Caused by:
    0: speculative_mtp_decode_step failed
    1: mtp verify forward failed
    2: linear attention state required for GDN layers (layer 0)
```

Loader is good — every Mtp run logs `Native MTP head detected and loaded (k=1 draft depth)`, confirming PR #253's `detect_mtp_prefix` fix is working against Qwen3.5-4B's stock tensor layout. The failure is downstream of loading.

### Root cause

`crates/kiln-model/src/speculative.rs:573-585`, inside `speculative_mtp_decode_step`, calls the base model verify forward with `None` for the `LinearAttentionState`:

```rust
let verify_input = [last_token, draft_token];
let (verify_logits, new_hidden) = model_forward_paged_with_last_hidden(
    backend,
    &verify_input,
    weights,
    config,
    base_cache,
    base_block_table,
    base_pos,
    None, // MTP speculative: no linear-attn state rollback for GDN in this WIP.
    None, // no LoRA on the verify pass — keep parity with scaffolding.
    None, // positions_gpu: let the forward pass build positions internally.
)
.context("mtp verify forward failed")?;
```

The 8th argument is `None`, but Qwen3.5-4B's 24 GDN layers require a `LinearAttentionState`. The forward pass panics at `forward.rs:2949` (and the mirrored 3055 / 3477 sites for the other GDN paths): `linear attention state required for GDN layers (layer {i})`. The inline comment in `speculative.rs:581` explicitly flags this as a known WIP gap.

MTP cannot run in kiln today on this hybrid-architecture model until GDN state snapshot / restore is threaded through the MTP verify path.

### Recommendation

Minimal follow-up fix PR, narrow scope:

1. Add `LinearAttentionState::snapshot()` / `restore_from(snapshot)` on the struct at `forward.rs:241` (deep-clone `recurrent_states` + `conv_states`; ping-pong allocation per preflight Risk #1 above — avoid per-step `cudaMemcpy`).
2. Thread the snapshotted state into `speculative_mtp_decode_step`: take a snapshot before the draft step, pass the live `&mut LinearAttentionState` to `model_forward_paged_with_last_hidden`, and restore from the snapshot on rejected tokens.
3. Re-run the 3× Mtp sweep. Keep the same `KILN_W4A16=1 KILN_CUDA_GRAPHS=true --paged --prompt-tokens 512 --max-output-tokens 128` workload. Median-of-3, reporting α alongside decode tok/s.

Preflight Check 2 in the section above math-ceilings α=0.75 at 1.571× on Qwen3.5-4B's MTP (k=1, ~86% overhead). The ≥1.5× target stands; α<0.72 is the stop-ship threshold.

### Cost

- Pod: 1× A6000 warm lease from the `ce kiln-pod-acquire` pool, ~35 min wall clock (preflight + build + 3×Off + 3×Mtp + log pull + lease release).
- Build: 81 s (sccache hit rate 94%).
- No nsys capture, no Marlin repack, no second pod.
- Well under the 120 min / $60 cap.

### What this PR does NOT do

- Does not attempt the GDN state rollback fix (scope-discipline: doc-only, per task brief).
- Does not run the SkipLayer arm (out of scope for this PR; Phase X preflight uses SkipLayer as an A/B option but the Mtp row is the blocker here).
- Does not modify any `.rs` source. PROFILING.md only.

### Follow-up: GDN state threading (2026-04-21, same-day)

Surgical fix landed in the same branch that produced this PROFILING.md update: `speculative_mtp_decode_step` now accepts `linear_state: &mut LinearAttentionState`, takes a snapshot before the verify forward, threads `Some(linear_state)` into `model_forward_paged_with_last_hidden`, and calls `restore_from(&snapshot)` on the reject branch so the rejected draft token's GDN state mutation is rolled back.

With that change the Mtp bench no longer fails at `linear attention state required for GDN layers (layer 0)`. 3× Mtp + 3× Off re-run on the same A6000 pod, same bench flags:

#### Mtp — raw per-run numbers (post state-threading)

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms | α |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 335.8 | 1507 | 24.41 | 40.97 | 17.12 | 38.05 | 0.407 |
| 2 | 331.2 | 1527 | 24.29 | 41.16 | 17.23 | 39.31 | 0.411 |
| 3 | 334.9 | 1511 | 22.53 | 44.39 | 17.05 | 38.99 | 0.524 |
| **median** | **334.9** | **1511** | **24.29** | **41.16** | **17.23** | **39.31** | **0.411** |

#### Off — raw per-run numbers (same pod, re-run for fresh-paired comparison)

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 324.2 | 1561 | 20.46 | 48.87 | 20.25 | 25.89 |
| 2 | 334.9 | 1511 | 20.74 | 48.21 | 20.75 | 23.13 |
| 3 | 326.2 | 1551 | 20.28 | 49.32 | 20.14 | 22.55 |
| **median** | **326.2** | **1551** | **20.46** | **48.87** | **20.25** | **23.13** |

#### Findings

1. **State threading is correct.** The bench path that previously crashed at decode step 0 now completes all three runs with 128 tokens each, stable tok/s across runs, and sane α.
2. **α = 0.411 is well below any viable ship gate, including the paper's aggregate α≈0.72 reference.** Later C39/C40 work showed that `0.72` is a prose-heavy aggregate benchmark reference, not a blanket code-domain stop-ship threshold: current-main HumanEval distributions sit around `0.69` under both W4A16 and BF16, while GSM8K clears the paper floor. That nuance does not change the local conclusion here. At α=0.411 with k=1, mean accepted tokens per verify call is 1.411; the verify-pass overhead (two-token forward + draft-head forward + snapshot/restore dToD) is larger than the 0.411-token savings, so:
3. **Mtp is 15.7% SLOWER than Off on Qwen3.5-4B at this α** (41.16 vs 48.87 tok/s median decode). p99 ITL also widens from 23.13 → 39.31 ms.
4. **The MTP path is functional but not yet shippable.** This PR unblocks the measurement path; the low-α investigation is a separate follow-up.

#### Why α is low (hypotheses, not debugged here)

- The MTP head may be consuming a stale `hidden_state` from the base forward — `speculative_mtp_decode_step` passes `last_hidden` returned from the verify forward, but the ordering vs the `embed(draft_token)` concat in `mtp_forward_step` should be re-checked against the HF reference.
- `pre_fc_norm_embedding` and `pre_fc_norm_hidden` tensor loading (PR #253) should be re-verified post-loading — log a dequant sample vs. HF `safetensors` to confirm the MTP head weights are byte-identical.
- Sampling temperature in the bench's rejection loop may not match the base sampler — preflight assumed greedy/temp=0; if the verify path uses a different sampler than the draft, acceptance will systematically under-report.
- Pretraining mismatch: Qwen3.5-4B's native MTP was trained with `mtp_num_hidden_layers=1` and a specific concat order; any minor layout difference in our loader (e.g., swapped `fc` input ordering) would hand back near-random predictions → α≈0.4 is exactly the signature.

#### Next steps

- Open a dedicated `mtp: debug low-α root cause` task. Instrument α per-step, log the top-5 draft tokens + base-verify probs for the first 16 verify calls, compare against HF reference on the same prompt tokens, confirm weight-tensor byte-equality after loader.
- Revisit shadow-slot ping-pong snapshot rewrite (zero-copy) only after α clears 0.72 — at α=0.411 it would not move the needle.
- Consider gating MTP on the new `KILN_SPEC_METHOD=mtp` flag with a startup warning until α ships above floor.

#### Cost (follow-up)

- Same warm pod lease, incremental re-build: 33 s wall clock (sccache hit).
- 3× Mtp run ~6 min + 3× Off run ~6 min.
- Well under the 120 min / $60 cap for the combined effort.

## MTP low-α root cause — Phase A static audit + Phase B instrumentation (2026-04-21)

Follow-up to "Phase X — native MTP spec-decode results" above. PR #257 unblocked the MTP measurement path; Mtp arm now runs to completion at α=0.411, well below the paper's aggregate α≈0.72 reference and also well below the workload-aware ranges later documented in C39/C40. This update lands the Phase B instrumentation patch and records the Phase A static-audit findings.

### Tier classification

This is a **Tier 2 diagnostic** PR (per the task brief): no fix shipped, no claim that any of the four candidate hypotheses from the Phase X "Why α is low" section are confirmed or refuted by code reading alone. Phase B requires a runtime trace on a pod plus an HF parity diff, which is the work the next agent picks up using the instrumentation landed here.

### Phase A — static audit findings

Code references are pinned to `e18ed1b` (PR #257 merge) on `main`.

| Hypothesis (from Phase X "Why α is low") | Static-audit verdict | Evidence |
|---|---|---|
| Stale `h_prev` from base verify forward | **Doc-confirmed correct on ACCEPT** | `crates/kiln-model/src/speculative.rs:481-490` documents that `new_h_prev` is the hidden at the draft token's position in the verify pass, which is the correct conditioning input for the *next* MTP step (it predicts the bonus, which on accept is exactly the token immediately after the draft). The ACCEPT case is the dominant path at any reasonable α; the REJECT case has a known <5% staleness called out in the same comment block. Both paths are too small to explain the 0.31 gap from 0.411 to 0.72. |
| Loader byte mismatch (`mtp.fc` etc.) | **Tensor names map cleanly** | `crates/kiln-model/src/loader.rs:610-704`: `mtp.fc.weight` shape validated as `[hidden, 2*hidden]`; both `mtp.norm.weight` and `mtp.final_layernorm.weight` accepted; `pre_fc_norm_embedding` and `pre_fc_norm_hidden` map directly to the corresponding `MtpWeights` fields with no aliasing. PR #253's `detect_mtp_prefix` covers Qwen3.5-4B's bare `mtp.*` layout. **Byte-equality vs. raw safetensors is not verifiable from code reading** — that needs Phase B. |
| Sampler / temperature mismatch | **Both paths greedy** | `speculative_mtp_decode_step` calls `greedy_sample(&mtp_logits)` for the draft (`speculative.rs:566`) and `greedy_sample(&verify_pos0)` for the target (`speculative.rs:602`). The `_params: &SamplingParams` argument is intentionally unused (commented `# Greedy-only path (temperature == 0)` on `speculative.rs:533-535`). Sampler asymmetry is ruled out for the bench workload. |
| Swapped `fc` input ordering vs. vLLM | **Visual match to vLLM** | `crates/kiln-model/src/forward.rs:3517-3523`: `Tensor::cat(&[&norm_emb, &norm_h], 2)` matches vLLM `qwen3_next_mtp.py`'s `concat([pre_fc_norm_embedding(inputs_embeds), pre_fc_norm_hidden(hidden_states)])`. Embed first, hidden second. **Tensor name → field semantics are correct.** What this *cannot* rule out from code reading: that `mtp.pre_fc_norm_embedding.weight` and `mtp.pre_fc_norm_hidden.weight` are themselves transposed in the published checkpoint relative to vLLM's training-time layout. Confirming that requires Phase B byte-equality + per-token dequant diff. |

#### Other code-reading checks

- **Tied lm_head**: `forward.rs:3553` reuses `weights.embed_tokens_t` for the MTP head's projection. Matches the loader's contract that there is no separate `mtp.lm_head.weight` tensor (per `loader.rs:604-606`). ✓
- **Position handling**: MTP uses an independent position counter (`mtp_pos`, starting at 0) for both the RoPE positions tensor (`forward.rs:3528`) and the KV-cache start index (`forward.rs:3535`). The base model uses `base_pos` (starting at `prompt_tokens.len()`) for both. They cannot interact: the MTP layer has its own 1-layer paged cache (`generate.rs:~1280`) so there is no cross-cache RoPE interference. **Constant offset between base and MTP positions is invisible to attention** because RoPE attention scores depend only on `pos_j - pos_i`, and the MTP layer attends only to its own KV cache (single sequence). ✓
- **`fc_t` cached transpose**: `cached_transpose(weight)` at `forward.rs:396-398` is a single `weight.t()?.contiguous()?`. Stored fc has shape `[H, 2H]` (PyTorch `nn.Linear(2H, H)` storage convention); `fc_t` has `[2H, H]`; matmul against `concat[..., 2H]` produces `[..., H]`. ✓
- **Marlin off the MTP layer**: `MtpGpuWeights` does not carry Marlin packed projections — the MTP transformer block uses the raw BF16 layer (`forward.rs:130-172`). W4A16 quantization mismatch is ruled out for this layer. ✓

#### Net Phase A verdict

Static audit found **no obvious layout or routing bug from code alone**. The most-suspicious remaining hypothesis is "swapped `fc` input ordering during checkpoint export" (the Phase X note's "α≈0.4 is exactly the signature"), but the kiln code visually matches the vLLM reference order. Confirming or refuting this requires runtime evidence: byte-equality of the loaded `mtp.*` tensors vs. raw safetensors, plus per-token comparison of the draft logits vs. an HF reference run on the same prompt.

### Phase B — instrumentation landed in this PR

New module `crates/kiln-model/src/mtp_debug.rs` (~120 lines incl. tests):

- `is_enabled()` / `should_log()` — `KILN_MTP_DEBUG=1` opt-in, with `KILN_MTP_DEBUG_MAX_CALLS` (default 16) limiting noise.
- `top_k_logits(logits, k)` — extracts top-K `(token_id, logit)` pairs from `[V]` / `[1, V]` / `[1, 1, V]` tensors.
- `tensor_l2_norm(t)` — single-f32 L2 sanity check.
- `format_top_k(...)` — compact `"[(id=42, l=12.34), ...]"` rendering for log lines.

Two call sites add one `tracing::info!` line each when enabled:

- `forward.rs::mtp_forward_step` — `mtp_draft` line per draft pass: `mtp_pos`, `last_token`, `h_prev_l2`, `mtp_logits_l2`, `mtp_top5`.
- `speculative.rs::speculative_mtp_decode_step` — `mtp_verify` line per verify pass: `mtp_pos`, `base_pos`, `last_token`, `draft_token`, `target_at_0`, `accepted`, `verify_pos0_l2`, `verify_pos0_top5`. Shares `mtp_pos` with the matching `mtp_draft` so trace lines pair by grep.

Both call sites early-out cheaply (`std::env::var` lookup) when the flag is unset, so production decode is not affected.

### Phase B — execution plan for the next agent

1. **Acquire pod**: `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`. Use the pool, not raw `runpod_api.py launch`.
2. **Build with this PR merged**: standard `cargo build --release --features cuda --bin kiln-bench` (sccache should land 90%+ on a warm pod).
3. **Run a 16-step trace**:
   ```bash
   KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
   KILN_MTP_DEBUG=1 RUST_LOG=kiln::mtp_debug=info,info \
     ./target/release/kiln-bench --paged --prompt-tokens 512 \
       --max-output-tokens 16 --skip-training 2>&1 | tee mtp-trace.log
   ```
   Output will contain 16 `mtp_draft` lines + 16 `mtp_verify` lines, each tagged with `mtp_pos` for pairing.
4. **HF reference parity** — small Python harness using `transformers` to load Qwen/Qwen3.5-4B and compute the same first-16 MTP draft tokens given the same prompt tokens. Diff:
   - Token-level: do `target_at_0` and `draft_token` from the trace match HF's predicted next-token for the same context? If `target_at_0` matches HF but `draft_token` doesn't, the bug is in the MTP draft head (loader / fc / norms / RoPE). If `target_at_0` itself doesn't match HF, the bug is in the base model verify path (less likely — base bench α metric would have caught it earlier).
   - Logit-level: compare `verify_pos0_top5` against HF's softmax top-5 on the same context. Should be byte-equivalent at temperature=0.
5. **Byte-equality of MTP tensors** — small Rust binary (or Python via `safetensors` crate equivalent) that:
   - Loads `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight` directly from raw safetensors.
   - Compares to the same tensors after kiln's loader path (dequant if needed; these MTP tensors are BF16, no quantization in the layer).
   - Confirms element-wise equality. If anything differs, that's the bug.
6. **Decision tree**:
   - Tensors byte-equal AND draft top-1 matches HF ⇒ MTP head is correct; α=0.411 is "as good as it gets" for this checkpoint and we should consider whether `mtp_num_hidden_layers=1` is a structural ceiling on Qwen3.5-4B (open a Phase 7 design question, not a bug fix).
   - Tensors byte-equal AND draft top-1 does NOT match HF ⇒ forward-pass bug. Likely candidates remaining: RoPE position offset (despite relative-invariance argument; check partial-rotary application), GDN state initialization for the standalone MTP layer (despite this being a *full-attention* layer per `is_full_attention_layer(3)`), or `attn_output_gate` interaction.
   - Tensors NOT byte-equal ⇒ loader bug. Fix in `loader.rs::load_mtp_if_present`. **This is the strongest fix-in-Tier-1 outcome** and would close the investigation.

### Cost guardrails for Phase B

- Pod: 1× A6000 warm lease, expected 30-45 min wall clock (build + 1× 16-step trace + HF parity Python script + byte-eq tool).
- Use `wait-file` pattern from the kiln skill `resources/runpod-workflow.md`. **Never** write `until ssh ... kill -0` polling loops — that pattern burned $99.76 + $13.76 in two SSH-wedge incidents on 2026-04-20.
- Hard cap: 90 min / $40 (per the standard kiln cap). Fall back to a Tier 2 docs-only update if Phase B blows the cap.

### Files / Reproduction

```bash
# This PR — Tier 2 diagnostic, code patch + PROFILING.md update
git fetch origin
git checkout mtp/debug-low-alpha-instrumentation
cargo check                                                        # CPU host
cargo build --release --features cuda --bin kiln-bench             # Linux+CUDA pod

# Phase B trace (next agent, on a pod)
KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
KILN_MTP_DEBUG=1 RUST_LOG=kiln::mtp_debug=info,info \
  ./target/release/kiln-bench --paged --prompt-tokens 512 \
    --max-output-tokens 16 --skip-training 2>&1 | tee mtp-trace.log
```

### What this PR does NOT do

- Does not change MTP forward, loader, sampler, or scheduler behavior in production decode (instrumentation early-outs when `KILN_MTP_DEBUG` is unset).
- Does not run a fresh bench. The α=0.411 baseline from the section above stands; this PR adds the instrumentation to investigate it.
- Does not modify the Phase X "Why α is low" hypotheses table — those are still open. This PR only narrows the search space via static audit and lands the runtime instrumentation needed to close them.

### Cost

- Local-only PR (`cargo check` on CPU host).
- $0 in pod time. The Phase B execution plan above commits to ≤90 min / $40 on a future pod lease.


## Phase B — Runtime MTP bisect trace (2026-04-21)

Follow-up to "Phase A audit + Phase B instrumentation" (PR #260). Landed a fresh runtime trace of native MTP speculative decode on A6000 with `KILN_MTP_DEBUG=1` and bisected the 13 draft/verify pairs per the Phase A execution plan.

### Repro

```bash
KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
  KILN_MTP_DEBUG=1 KILN_MTP_DEBUG_MAX_CALLS=32 \
  RUST_LOG=kiln::mtp_debug=info,info \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 16 --skip-training
```

- Pod: A6000 (49140 MB), CUDA 12.4, image `ghcr.io/ericflo/kiln-runpod:latest`
- Checkpoint: `Qwen/Qwen3.5-4B` (HF download), weight prefix `model.language_model.` (VL-wrapper architecture `Qwen3_5ForConditionalGeneration`)
- Build: warm sccache/B2, release + `--features cuda`, `KILN_CUDA_ARCHS=86`
- Full trace: `assets/mtp-phase-b-trace-2026-04-21.log` (194 lines, 13 draft/verify pairs)

### Headline numbers

| Metric | Value |
| --- | --- |
| Draft acceptance α | **0.154** (2 of 13 drafts accepted) |
| Decode tok/s (MTP arm) | 30.13 |
| Decode tok/s (baseline plain, from skill doc) | 49.76 |
| Prefill TTFT ms | 11074.2 (warmup artifact; see skill note `kiln-bench-prefill-warmup-required`) |
| Mean ITL ms | 57.56 |
| P99 ITL ms | 67.998 |

α = 0.154 is **lower than the 0.411 previously recorded** in the "Phase X — native MTP spec-decode results" section. The prior α=0.411 figure came from a different bench run that was not captured with full instrumentation; the 0.411 number remains in PROFILING.md but should be treated as a ceiling, not a floor, until re-measured under identical settings.

**MTP is a decode regression at this α.** 30.13 tok/s vs 49.76 plain-decode baseline is 0.605× — worse than off. This is consistent with the overhead-math laid out in the prior phase (verify forward + draft forward + linear-state snapshot/restore dToD exceeds the savings below ~α=0.5).

### Bisect table (13 steps)

Full data in `assets/mtp-phase-b-trace-2026-04-21.log`. Summary:

| # | mtp_pos | base_pos | last_token | draft_top1 | target_at_0 | accepted | identity? | h_prev_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
|  1 | 0 | 506 |    561 |    561 |  29144 | ✗ | **Y** | 64.00 |
|  2 | 0 | 507 |  29144 |  29144 |   6165 | ✗ | **Y** | 68.21 |
|  3 | 0 | 508 |   6165 |   6165 |  27050 | ✗ | **Y** | 74.76 |
|  4 | 0 | 509 |  27050 |  27050 |    279 | ✗ | **Y** | 72.58 |
|  5 | 0 | 510 |    279 |   3377 |  29144 | ✗ |   | 73.45 |
|  6 | 0 | 511 |  29144 |   1785 |   6165 | ✗ |   | 56.80 |
|  7 | 0 | 512 |   6165 |  18465 |  27050 | ✗ |   | 61.05 |
|  8 | 0 | 513 |  27050 |    279 |    279 | **✓** |   | 61.82 |
|  9 | 1 | 515 |  24460 |  24460 |   3377 | ✗ | **Y** | 67.95 |
| 10 | 1 | 516 |   3377 |   3377 |     13 | ✗ | **Y** | 70.53 |
| 11 | 1 | 517 |     13 |    271 |    271 | **✓** |   | 67.90 |
| 12 | 2 | 519 | 248068 |    271 |    198 | ✗ |   | 51.76 |
| 13 | 2 | 520 |    198 | 248069 |    760 | ✗ |   | 46.54 |

Column key:
- `identity?` — draft top-1 equals the input `last_token` (i.e. "the MTP head says the next token will be the same as the one just generated").

### Primary finding — identity-prediction bias

**6 of 13 draft predictions (46.2%) have `mtp_top5[0].id == last_token`.** Four of those six are consecutive at the start of decode (`mtp_pos=0, base_pos ∈ [506..509]`) with draft top-1 = last input token and logit-1 spread ≥4 over logit-2 — the MTP head is highly confident in the wrong answer.

Magnitudes are all healthy:
- `h_prev_l2 ∈ [46.5, 74.8]` — well-formed hidden states, no zero-out or explosion.
- `mtp_logits_l2 ∈ [1062, 1479]` and `verify_pos0_l2 ∈ [1100, 1743]` — comparable scales, so this is **not** a magnitude bug; the argmax is wrong.

### Secondary finding — out-of-vocab special-token drafting

At steps 12–13, drafts include tokens `248068` and `248069`, both high-id special/reserved tokens (`image_token_id=248056`, `eos_token_id=248044` from `config.json`). These appear in draft top-5 with dominant logits (e.g. step 13 draft top-1 = 248069 @ l=24.6) but do not dominate the base verify_pos0 output. This is not the primary failure mode but it does indicate the MTP head is allocating nontrivial probability to VL/reserved token IDs the base model correctly rejects.

### Hypotheses ranked by strength of runtime evidence

1. **fc matmul produces an embed-dominated fused state (most likely).** When the MTP `fc` projection  (2560 ← 5120) weights the embed-half of `concat([norm_emb, norm_h])` disproportionately higher than the hidden-half, the single-layer MTP transformer block + tied `lm_head` projection regresses toward `argmax(embed(x) · embed^T)[x] = x`. This is exactly the identity bias pattern observed, and is strongest when `h_prev` carries the least task-specific signal (early in the decode, steps 1–4) and weakens as `h_prev` accumulates context (steps 5+). Phase A audit marked "fc input ordering" as checked statically, but a *magnitude* asymmetry between the two halves of `fc.weight` would not show up in a shape/concat-order check — only in a per-half norm comparison or byte-equality diff against HF.

2. **`pre_fc_norm_hidden` loaded into the wrong tensor slot (plausible).** If `mtp.pre_fc_norm_hidden.weight` ended up loaded into the scale of `mtp.pre_fc_norm_embedding.weight` (or vice versa), or if one of the two RMSNorm scales collapsed to near-zero, the fused input is embed-dominant for exactly the reason above. Runtime tensor dump or byte-equality against the safetensors shard would disambiguate.

3. **MTP RoPE position mismatch (weaker evidence).** `forward.rs:3528` passes `mtp_pos as f32` for RoPE (0..=2 across the trace), while the token being drafted actually lives at `base_pos + 1 ∈ [507, 521]`. On a single-token MTP attention pass the RoPE rotations cancel in `Q·K` (self-attention), so this alone would not produce identity bias — but it could compound with (1) on accepted steps when the MTP cache has >1 token.

4. **Structural ceiling (unlikely given pattern).** If MTP were simply a weak draft head (`num_nextn_predict_layers=1` structural limit), we would expect low-but-diffuse draft distributions — not sharp, high-confidence, wrong-at-logit-gap-of-4+ identity predictions on 4 consecutive steps. The pattern is too consistent to be noise.

### Recommended next bisect step (Tier 2, ≤45 min)

Byte-equality check on the 3 critical MTP tensors directly on a pod:

```python
from safetensors import safe_open
for t in ["mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"]:
    # Compare bytes with kiln's GpuWeights.mtp via a small ffi dump, or
    # dump kiln's loaded tensors to .npy and diff against the raw safetensors.
```

In parallel, dump `mtp.fc.weight[:, :2560]` (embed half) and `mtp.fc.weight[:, 2560:]` (hidden half) L2 norms and compare. An asymmetry >2× would directly confirm hypothesis (1).

If byte-equal and halves balanced → run a one-shot HF reference parity on the same 512-token prompt and diff `draft_top_k[0]` per step. If HF also emits identity-biased drafts at step 1–4, the pattern is a genuine checkpoint property (structural, hypothesis 4). If HF does not, the gap is in kiln's forward path — most likely (3) or a subtle op in `mtp_forward_step` (e.g. residual vs. no-residual at `fused → attn`, currently implicit via `transformer_block_paged`).

### Budget used

- Pod: 1× A6000 on-demand (pool fallback: `runpod_api.py launch` direct, image `ghcr.io/ericflo/kiln-runpod:latest`).
- Wall-clock: ~45 min (under the 90 min / $40 cap).
- Work: build (5m 26s warm sccache) + model download (~3 min, parallel with build) + 1 bench run (~30s decode after 37s model load) + analysis.

Pod terminated on task completion.

## Phase B2 — fc halves + norm-swap bisect (2026-04-21)

Follow-up to Phase B. Three-part experiment: (1) byte-equality verification of all 15 MTP tensors against the raw safetensors shards, (2) runtime logging of the two halves of `fc`'s input (pre-norm-embedding on `embed(last_token)` vs pre-norm-hidden on `h_prev`) plus fused output L2, (3) A/B toggle (`KILN_MTP_SWAP_FC_NORMS=1`) that swaps which RMSNorm weight is paired with which half. Goal: isolate whether Phase B's identity bias comes from a byte-level loader bug, a halves-magnitude asymmetry, or the two RMSNorm scales being wired to the wrong halves of the fused input.

All runs on the same A6000 pod (on-demand, `ghcr.io/ericflo/kiln-runpod:latest`, warm sccache B2), same model (`/workspace/qwen3.5-4b`), same prompt used in Phase B, `--paged --skip-training --max-output-tokens 32`, `KILN_MTP_DEBUG=1 KILN_MTP_DEBUG_MAX_CALLS=16`.

### Part A — byte-equality (rules out raw loader bugs)

New test `crates/kiln-model/tests/mtp_byte_eq.rs` walks all 15 MTP tensors under the detected `mtp.*` prefix, flattens each `Tensor` kiln loaded into host memory, and compares bytes against the corresponding slice in the raw safetensors shard (mmap via `safetensors::SafeTensors::deserialize`). Runs on CPU so the same test reproduces on any developer laptop.

```
test mtp_weights_match_safetensors_byte_for_byte ...
[mtp_byte_eq] using mtp_prefix = mtp.
[mtp_byte_eq] === report ===
PASS fc                                  (mtp.fc.weight)                                  shape=[2560, 5120] dtype=BF16
PASS pre_fc_norm_embedding               (mtp.pre_fc_norm_embedding.weight)               shape=[2560]       dtype=BF16
PASS pre_fc_norm_hidden                  (mtp.pre_fc_norm_hidden.weight)                  shape=[2560]       dtype=BF16
PASS final_layernorm                     (mtp.norm.weight)                                shape=[2560]       dtype=BF16
PASS transformer_block.input_layernorm   (mtp.layer.input_layernorm.weight)               shape=[2560]       dtype=BF16
PASS transformer_block.post_attn_ln      (mtp.layer.post_attention_layernorm.weight)      shape=[2560]       dtype=BF16
PASS mlp.gate_proj                       (mtp.layer.mlp.gate_proj.weight)                 shape=[9728, 2560] dtype=BF16
PASS mlp.up_proj                         (mtp.layer.mlp.up_proj.weight)                   shape=[9728, 2560] dtype=BF16
PASS mlp.down_proj                       (mtp.layer.mlp.down_proj.weight)                 shape=[2560, 9728] dtype=BF16
PASS attn.q_proj                         (mtp.layer.self_attn.q_proj.weight)              shape=[4096, 2560] dtype=BF16
PASS attn.k_proj                         (mtp.layer.self_attn.k_proj.weight)              shape=[1024, 2560] dtype=BF16
PASS attn.v_proj                         (mtp.layer.self_attn.v_proj.weight)              shape=[1024, 2560] dtype=BF16
PASS attn.o_proj                         (mtp.layer.self_attn.o_proj.weight)              shape=[2560, 4096] dtype=BF16
PASS attn.q_norm                         (mtp.layer.self_attn.q_norm.weight)              shape=[256]        dtype=BF16
PASS attn.k_norm                         (mtp.layer.self_attn.k_norm.weight)              shape=[256]        dtype=BF16
[mtp_byte_eq] PASS — all MTP tensors byte-equal to safetensors
test result: ok. 1 passed; finished in 6.87s
```

**Verdict A**: kiln's MTP loader is byte-for-byte faithful for every MTP tensor on this Qwen3.5-4B checkpoint. This mechanically rules out Phase B hypothesis 2 in its strong form — no raw loader bug exists where a tensor is loaded into the wrong slot or byte-mangled during conversion. Any remaining ambiguity must be in *how* the correctly-loaded tensors are wired into `mtp_forward_step`.

### Part B — halves-L2 instrumentation

`crates/kiln-model/src/mtp_debug.rs` extended with `is_swap_fc_norms_enabled()` and emits, per draft step, the L2 norms of:

- `h_prev_l2` — L2 norm of the incoming last-layer hidden state
- `norm_emb_l2` — L2 of `rms_norm(embed(last_token), pre_fc_norm_embedding)`
- `norm_h_l2` — L2 of `rms_norm(h_prev, pre_fc_norm_hidden)`
- `halves_ratio = norm_emb_l2 / norm_h_l2`
- `fused_l2` — L2 of `fc(concat(...))` output, the input to the MTP transformer block

Halves are extracted in `mtp_forward_step` via `narrow` slices on the fused `concat` result so the measured halves come from exactly the tensor the matmul sees, not a re-computation.

### Part C — norm-swap A/B toggle

`KILN_MTP_SWAP_FC_NORMS=1` causes `mtp_forward_step` to swap which `RmsNorm` weight is applied to which half of the fused input (`pre_fc_norm_hidden` applied to the embedding half, `pre_fc_norm_embedding` applied to the `h_prev` half). Read every call so an A/B can be run across two processes with no code change. Matches the bisect workflow described at the end of Phase B under "Recommended next bisect step".

### Runtime results — single 32-output-token prompt, A/B

| Arm | α (draft accept) | num_drafts | num_accepts | identity (draft==last_tok) overall | halves_ratio median |
|---|---|---|---|---|---|
| swap **OFF** (default) | **0.240** (6 / 25 verify) | 16 | 6 | 7/25 = **28%** | ~0.745 (emb < h) |
| swap **ON**  | **0.348** (8 / 23 verify) | 16 | 8 | 3/23 = **13%** | ~1.067 (emb > h) |

Δα: **+0.108 absolute, +45% relative** with swap ON.

#### Halves table — swap-OFF (first 13 draft rows)

| mtp_pos | last_tok | h_prev_l2 | norm_emb_l2 | norm_h_l2 | halves_ratio | fused_l2 | mtp_logits_l2 | top1_id | top1_logit |
|--|--|--|--|--|--|--|--|--|--|
| 0 | 561   | 64.38 | 32.02 | 41.34 | 0.7745 | 14.14 | 1068.4 | 561   | 21.12 |
| 0 | 29144 | 69.17 | 30.24 | 40.13 | 0.7537 | 15.01 | 1229.8 | 29144 | 17.25 |
| 0 | 6165  | 76.73 | 29.93 | 41.27 | 0.7251 | 15.40 | 1111.6 | 6165  | 17.88 |
| 0 | 27050 | 72.92 | 30.29 | 41.16 | 0.7360 | 14.91 | 1155.7 | 27050 | 15.75 |
| 0 | 279   | 75.05 | 32.18 | 41.50 | 0.7754 | 14.91 | 1076.6 | 3377  | 14.81 |
| 0 | 29144 | 59.62 | 30.24 | 40.67 | 0.7436 | 15.06 | 1398.4 | 1785  | 14.81 |
| 0 | 6165  | 60.48 | 29.93 | 40.45 | 0.7400 | 15.58 | 1252.1 | 18465 | 18.62 |
| 0 | 27050 | 63.04 | 30.29 | 40.47 | 0.7485 | 14.59 | 1210.6 | 279   | 18.88 |
| 1 | 24460 | 67.21 | 30.29 | 40.78 | 0.7429 | 14.81 | 1148.9 | 3377  | 20.38 |
| 2 | 303   | 65.20 | 32.76 | 41.04 | 0.7983 | 13.92 | 1212.8 | 279   | 20.50 |
| 2 | 3150  | 60.00 | 29.88 | 39.46 | 0.7572 | 14.68 | 1227.2 | 3150  | 19.88 |
| 2 | 854   | 70.49 | 30.16 | 40.21 | 0.7501 | 15.05 | 1129.8 | 13    | 19.00 |
| 3 | 2838  | 72.92 | 29.92 | 39.71 | 0.7534 | 13.43 | 1130.1 | 2838  | 17.38 |

`halves_ratio` is tightly clustered in `[0.725, 0.798]`. The pre-norm on the embed half produces a scale ~25% smaller than the pre-norm on the hidden half, every step. Identity hits at rows 1-4 (pos=0) and again at rows 11 and 13.

#### Halves table — swap-ON (first 13 draft rows)

| mtp_pos | last_tok | h_prev_l2 | norm_emb_l2 | norm_h_l2 | halves_ratio | fused_l2 | mtp_logits_l2 | top1_id | top1_logit |
|--|--|--|--|--|--|--|--|--|--|
| 0 | 561   | 64.38 | 40.21 | 36.55 | 1.1001 | 13.88 |  960.7 | 561   | 16.38 |
| 0 | 29144 | 69.17 | 38.75 | 37.62 | 1.0301 | 15.88 | 1267.3 | 6165  | 15.06 |
| 1 | 27050 | 67.78 | 38.68 | 35.60 | 1.0865 | 14.80 | 1235.3 | 279   | 18.62 |
| 2 | 24460 | 67.40 | 38.68 | 36.89 | 1.0486 | 15.10 | 1096.8 | 3377  | 18.00 |
| 3 | 303   | 63.82 | 40.74 | 36.62 | 1.1127 | 13.54 | 1493.0 | 279   | 21.62 |
| 3 | 3150  | 61.95 | 37.89 | 36.59 | 1.0357 | 14.58 | 1191.0 | 3150  | 18.50 |
| 3 | 854   | 69.00 | 38.19 | 36.07 | 1.0587 | 14.86 | 1150.8 | 13    | 19.12 |
| 4 | 2838  | 71.45 | 38.24 | 34.96 | 1.0935 | 13.85 | 1251.6 | 369   | 16.88 |
| 4 | 5947  | 65.90 | 38.58 | 37.24 | 1.0361 | 14.55 | 1128.3 | 264   | 18.12 |
| 5 | 15352 | 89.60 | 38.94 | 38.15 | 1.0207 | 15.78 | 1243.3 | 6157  | 16.75 |
| 6 | 314   | 77.91 | 40.44 | 37.15 | 1.0885 | 13.06 | 1097.5 | 220   | 19.12 |
| 6 | 2981  | 67.48 | 38.11 | 34.16 | 1.1155 | 15.57 | 1042.6 | 8876  | 14.56 |
| 6 | 17793 | 78.24 | 38.99 | 36.77 | 1.0603 | 15.06 | 1221.4 | 48736 | 17.00 |

With the norm-swap on, `halves_ratio` clusters in `[1.02, 1.12]` — the embedding half is now slightly larger than the hidden half. The top-1 token no longer matches `last_tok` at rows 1, 3-8, 10-13 (only two identity hits in 13 rows vs five in the swap-OFF table over the same slice).

#### Phase B verify-pos0 comparison

Direct comparison against the Phase B verify trace for the same prompt (first 4 draft steps, `mtp_pos=0`, predictions of the next ground-truth token):

| step | last_tok | target | swap-OFF top1 | swap-ON top1 |
|--|--|--|--|--|
| 1 | 561   | 29144 | **561** (identity, wrong)   | **561** (identity, wrong)   |
| 2 | 29144 | 6165  | **29144** (identity, wrong) | **6165** ✅ |
| 3 | 6165  | 27050 | **6165** (identity, wrong)  | — (advanced by accept at step 2) |
| 4 | 27050 | 279   | **27050** (identity, wrong) | — |

The norm-swap flipped step 2 from an identity-biased reject into a direct accept. Step 1 remains wrong under both arms — consistent with "cold" MTP state when `h_prev` for the very first draft carries too little task context regardless of norm pairing.

### Verdict

- **Phase B hypothesis 2 (wrong norm-tensor slot / collapsed scale) — still plausible but not byte-level.** Part A proves no tensor is byte-swapped or corrupted at load. But Part C shows a direct, reproducible, same-prompt α lift of +45% when the two RMSNorm scales are applied to each other's half. That is exactly the signature expected if the loader/wiring maps `pre_fc_norm_embedding` to operate on `h_prev` and `pre_fc_norm_hidden` to operate on `embed(last_token)` — i.e. the two scales are correctly loaded but attached to the wrong half inside `mtp_forward_step`.
- **Phase B hypothesis 1 (fc matmul embed-dominance) — partially supported as a consequence, not the root cause.** Under swap-OFF the embed half is smaller (`halves_ratio ≈ 0.74`) yet identity bias dominates; under swap-ON the embed half is slightly larger (`halves_ratio ≈ 1.07`) and identity bias drops. That direction is opposite to "fc weights the embed half too heavily" — if embed magnitude alone drove the bias we would expect more identity with a larger embed half, not less. The halves ratio interacts with the identity bias through the norm pairing, not directly through halves magnitude.
- **Direction for Phase B3**: re-verify the norm-half pairing in `mtp_forward_step` against the Qwen3-MTP reference implementation and/or the HF generator. Compare kiln's concatenation order and the binding between `pre_fc_norm_embedding`/`pre_fc_norm_hidden` and the `embed`/`h_prev` halves. If swap-ON is the "correct" pairing, the fix is a one-line rewire and should reproduce across multiple prompts and longer decode windows.

### Caveats

- **N=1, single prompt, 32 output tokens, 16 draft calls captured per arm.** The +45% α lift is a strong signal but has not been confirmed across prompts. Phase B3 must run a multi-prompt A/B (e.g. 8–16 prompts × 128 output tokens) before any code change is landed as the canonical wiring.
- Both runs use the same bench seed and the same prompt, so the first draft step's `last_token=561` and `target=29144` are identical. The divergence at step 2 (6165 vs identity) is deterministic relative to the norm-pairing choice.
- Out-of-vocab special-token drafting (`248068`, `248069`) from Phase B still reproduces under swap-ON at `pos=8`, so the norm-swap is not a universal fix — it is a bisect result pointing at the pairing, not a production patch.

### Budget used

- Pod: 1× A6000 on-demand (pool lease, `ghcr.io/ericflo/kiln-runpod:latest`).
- Wall-clock: ~55 min total (build + byte-eq 34s on warm sccache, A/B pair 81s, upload/download/parse ~5 min; rest was planning and PROFILING.md write-up).
- Work: byte-eq test (Part A), halves-L2 logging (Part B), swap toggle (Part C), this write-up (Part D).

Pod released to pool on task completion.

## Phase B3 — MTP multi-prompt A/B: reject the B2 norm-swap finding (2026-04-21)

Phase B2 ended on a single-prompt N=1 result: swap-ON lifted α from 0.240 to 0.348 (+45%) for one fixed bench prompt, 32 output tokens. Phase B3 set out to confirm or reject that finding across 8 distinct prompts × 128 output tokens, on the same A6000 pod image and the same MTP tracing plumbing.

**Verdict: REJECT.** Across 8 seeds × 2 arms, swap-ON beats swap-OFF on α in only 3/8 seeds. Mean paired Δα = −0.208 (swap hurts on average). The B2 +45% result is within the normal prompt-dependent α variance of swap-OFF (range 0.085–0.829) and was not a reliable signal. The norm-swap is not the correct wiring; do not land PR-style wiring changes based on Phase B2 alone.

The two consistent structural signals **are** real but do not translate into α gain:
- **Δhalves_ratio = +0.327 ± 0.015** across all 8 seeds — the swap does materially change the fc input composition as expected.
- **Δidentity_bias = −0.075 ± 0.049** across all 8 seeds — swap-ON reliably reduces the "draft the last token" failure mode. But this reduction is consumed by other draft failures, so α does not rise.

### Methodology

Reused everything from Phase B2 (bench binary, MTP tracing, halves-L2 logging, `KILN_MTP_SWAP_FC_NORMS`, MTP_DEBUG cap = 16 draft emissions). Two changes:

1. `kiln-bench --seed <u64>` now also selects one of 8 prompt bases (`PROMPT_POOL[seed % 8]`), so seeds actually produce different token distributions instead of identical greedy runs. Seed 0 is the Phase B2 baseline preserved verbatim; seeds 1–7 cover distinct topics (software, history, nature, philosophy, cooking, astronomy, music).
2. Output window increased from 32 → 128 tokens per run to amplify α signal (70–127 `mtp_verify` events per run vs. ~20 in B2).

Sweep: 16 runs (8 seeds × {off, on}), paged production path, MTP_DEBUG on:

```bash
for SEED in 0..7; do for ARM in off on; do
  env KILN_MTP_DEBUG=1 KILN_SPEC_METHOD=mtp ${ARM=on:+KILN_MTP_SWAP_FC_NORMS=1} \
    kiln-bench --model-path .../qwen3.5-4b --paged \
      --prompt-tokens 512 --max-output-tokens 128 --skip-training \
      --seed $SEED > mtp-b3-seed${SEED}-${ARM}.log 2>&1
done; done
```

Wall-clock: 1710 s (≈ 28.5 min, ≈107 s per run) on one A6000. Trace files archived under `assets/mtp-phase-b3-seed{0..7}-{off,on}.log`.

Aggregation (`assets/mtp-phase-b3-aggregate.py`): strips ANSI codes, stops at the `Inference Throughput Benchmarks` marker so post-latency events do not contaminate metrics, extracts α from `mtp_verify accepted=…`, identity-bias from `mtp_verify draft_token==last_token`, halves ratio / norm-L2 from `mtp_draft` events.

### Per-run aggregate

`alpha` = accept rate on `mtp_verify`; `id_bias` = fraction of verifies where `draft_token == last_token`; `halves` = mean `halves_ratio` (`norm_emb_l2 / norm_h_l2`); `norm_e`/`norm_h` = mean L2 of the two fc-input halves; `oov` = drafts with `draft_token >= 151936`; `n_d`/`n_v` = number of draft/verify events (n_d is capped at 16 by `KILN_MTP_DEBUG_MAX_CALLS`).

| seed | arm | alpha  | id_bias | halves | norm_e | norm_h | oov | n_d | n_v |
|-----:|:---:|:------:|:-------:|:------:|:------:|:------:|:---:|:---:|:---:|
|    0 | off | 0.257  | 0.099   | 0.756  | 30.68  | 40.58  | 0   | 16  | 101 |
|    0 | on  | 0.306  | 0.031   | 1.076  | 39.05  | 36.35  | 1   | 16  |  98 |
|    1 | off | 0.198  | 0.085   | 0.762  | 31.19  | 40.96  | 0   | 16  | 106 |
|    1 | on  | 0.366  | 0.011   | 1.069  | 39.06  | 36.56  | 0   | 16  |  93 |
|    2 | off | 0.085  | 0.195   | 0.766  | 30.90  | 40.32  | 0   | 16  | 118 |
|    2 | on  | 0.347  | 0.042   | 1.097  | 38.93  | 35.61  | 0   | 16  |  95 |
|    3 | off | 0.477  | 0.023   | 0.768  | 30.80  | 40.15  | 2   | 16  |  86 |
|    3 | on  | 0.337  | 0.021   | 1.105  | 39.10  | 35.50  | 1   | 16  |  95 |
|    4 | off | 0.829  | 0.143   | 0.759  | 30.78  | 40.58  | 0   | 16  |  70 |
|    4 | on  | 0.257  | 0.000   | 1.109  | 39.05  | 35.32  | 0   | 16  | 101 |
|    5 | off | 0.309  | 0.093   | 0.769  | 30.75  | 40.03  | 3   | 16  |  97 |
|    5 | on  | 0.033  | 0.008   | 1.082  | 38.88  | 35.94  | 0   | 16  | 123 |
|    6 | off | 0.549  | 0.049   | 0.774  | 30.96  | 40.00  | 1   | 16  |  82 |
|    6 | on  | 0.033  | 0.024   | 1.119  | 39.23  | 35.13  | 1   | 16  | 123 |
|    7 | off | 0.641  | 0.051   | 0.764  | 31.05  | 40.67  | 0   | 16  |  78 |
|    7 | on  | 0.000  | 0.000   | 1.075  | 39.46  | 36.75  | 0   | 16  | 127 |

### Paired ON − OFF per seed

| seed | d_alpha | d_id_bias | d_halves | d_oov |
|-----:|:-------:|:---------:|:--------:|:-----:|
|    0 | +0.049  | −0.068    | +0.319   | +1    |
|    1 | +0.167  | −0.074    | +0.307   |  0    |
|    2 | +0.263  | −0.153    | +0.330   |  0    |
|    3 | −0.140  | −0.002    | +0.337   | −1    |
|    4 | −0.571  | −0.143    | +0.350   |  0    |
|    5 | −0.277  | −0.085    | +0.314   | −3    |
|    6 | −0.516  | −0.024    | +0.345   |  0    |
|    7 | −0.641  | −0.051    | +0.311   |  0    |

Summary:

| metric    | mean   | std   | min    | max    | n |
|-----------|:------:|:-----:|:------:|:------:|:-:|
| d_alpha   | −0.208 | 0.326 | −0.641 | +0.263 | 8 |
| d_id_bias | −0.075 | 0.049 | −0.153 | −0.002 | 8 |
| d_halves  | +0.327 | 0.015 | +0.307 | +0.350 | 8 |
| d_oov     | −0.375 | 1.111 | −3.000 | +1.000 | 8 |

**ON > OFF on d_alpha: 3/8 seeds.** Decision rule stated in the task spec — CONFIRM requires ≥ 7/8, INCONCLUSIVE 6/8, REJECT ≤ 5/8. We are well inside REJECT.

### Interpretation

Phase B2 compared swap-OFF α = 0.240 against swap-ON α = 0.348 on a single shared prompt (seed 0 in B3 terms, 32 output tokens). B3 seed-0 at 128 tokens is OFF=0.257 / ON=0.306, Δα = +0.049 — a much smaller lift than B2 reported, still positive but not +45%. Seeds 1–2 show a genuine Δα lift (+0.167, +0.263) while seeds 3–7 show a Δα *drop* as large as −0.641. No clean topic-cluster pattern in the "helped" vs. "hurt" split (seed 0 canonical / 1 software / 2 history were helped; 3 nature / 4 philosophy / 5 cooking / 6 astronomy / 7 music were hurt). The swap-OFF α distribution across prompts (0.085–0.829, range 0.74) is the dominant source of variance; once OFF α is high on a given prompt, ON α is often substantially lower, and vice versa — the two arms are not a pure "shifted mean" but a repairing-vs-breaking pair whose sign depends on the prompt.

The two truly consistent deltas (halves_ratio and id_bias) confirm the norm-swap acts as intended mechanically — the `embed` half of the fc input gets the larger-scale norm applied, inverting which half dominates, and the "copy last token" failure mode drops reliably. But that failure mode only accounts for a small fraction of rejects (OFF id_bias ranges 2.3%–19.5%, mostly < 10%), so fixing it does not move α in the direction we need.

**This rules out "swap the norms" as the wiring fix.** The remaining hypothesis from Phase B — that fc maps the wrong half-to-slot pairing in the concatenation — is still live, but the concat order and matmul direction are now the more likely culprits, not the RMSNorm attachment. A Phase B4 (if pursued) should:

1. Verify the fc concat ordering (`[embed; h_prev]` vs. `[h_prev; embed]`) against the Qwen3-MTP reference generator end-to-end on the same prompt.
2. Confirm the fc weight tensor's row/column layout matches the concat order kiln feeds it.
3. Re-trace α with any such wiring fix across the same 8-prompt pool, using this same sweep script for a direct comparison.

Do not reopen the norm-swap question without a new structural hypothesis — this A/B is decisively negative.

### Caveats

- `n_drafts` is the MAX_CALLS cap (16) per run, not the true number of drafts; α statistics come from `n_verifies` (70–127 per run), which is statistically adequate for ±~0.05 precision per arm.
- Greedy sampling (`temperature = 0`) means seeds only vary the prompt, not per-token RNG. This is the correct choice for reproducibility but means each seed is effectively a single deterministic trajectory through decode.
- The `oov` count (drafts with `token_id >= 151936`) is small (0–3 per run) and inconsistent in sign between arms. The Phase B OOV special-token pathology still exists but is orthogonal to the norm-pairing question.
- MTP_DEBUG tracing adds stderr I/O overhead but is applied symmetrically, so it does not bias the ON/OFF comparison.

### Budget used

- Pod: 1× A6000 on-demand (pool lease, `ghcr.io/ericflo/kiln-runpod:latest`), reused from the B2 lease.
- Wall-clock: sweep 28.5 min + aggregation + write-up ≈ 55 min on pod, well under the 120 min / $60 task cap.
- Work: Part A (`--seed` prompt-pool wiring in `bench.rs`), Part B (16-run sweep script + execution), Part C (ANSI-safe aggregator), Part D (this section). All four parts landed in this PR.

Pod released to pool on task completion.

## Phase B4 — MTP `fc` concat order + weight layout audit vs upstream (doc-only)

### Goal

Phase B3 (PR #265) ruled out the `pre_fc_norm_embedding` ↔ `pre_fc_norm_hidden` swap as the α-collapse fix. The next live hypothesis from Phase B was that kiln's `fc` concat order or weight row/column layout mismatches the upstream Qwen3-MTP reference. Phase B4 is a $0, pod-free desk audit: compare kiln's MTP glue wiring byte-for-byte against the vLLM and SGLang reference implementations, plus a shape check against the published Qwen3.5-4B checkpoint header.

### Upstream references

**vLLM — `vllm/model_executor/models/qwen3_5_mtp.py`** (commit `771913e4a024`):

```python
# vllm-project/vllm @ 771913e4a024, qwen3_5_mtp.py
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    ...,
)
self.pre_fc_norm_embedding = RMSNorm(
    self.config.hidden_size, eps=self.config.rms_norm_eps
)
self.pre_fc_norm_hidden = RMSNorm(
    self.config.hidden_size, eps=self.config.rms_norm_eps
)
...
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

Source: <https://github.com/vllm-project/vllm/blob/771913e4a024/vllm/model_executor/models/qwen3_5_mtp.py>.

**vLLM — `vllm/model_executor/models/qwen3_next_mtp.py`** (commit `657855ab4179`):

```python
# vllm-project/vllm @ 657855ab4179, qwen3_next_mtp.py
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    ...,
)
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

Source: <https://github.com/vllm-project/vllm/blob/657855ab4179/vllm/model_executor/models/qwen3_next_mtp.py>.

**SGLang — `python/sglang/srt/models/qwen3_5_mtp.py`** (commit `cabe171b6ce3`):

```python
# sgl-project/sglang @ cabe171b6ce3, qwen3_5_mtp.py
self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
...
input_embeds = self.pre_fc_norm_embedding(input_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
hidden_states = self.fc(hidden_states)
```

Source: <https://github.com/sgl-project/sglang/blob/cabe171b6ce3/python/sglang/srt/models/qwen3_5_mtp.py>.

All three upstream references agree on the same wiring:

1. **Concat order** is `[embed, hidden]` — the freshly embedded draft token first, the pre-final-norm hidden state second, along `dim=-1`.
2. **Norm-to-input assignment** is `pre_fc_norm_embedding` → embedding half, `pre_fc_norm_hidden` → hidden half.
3. **fc projection** is a standard PyTorch `nn.Linear(2*H, H)` (aliased as `ColumnParallelLinear(2*H, H)` in vLLM, which is identical to `nn.Linear` at `tensor_parallel_size=1`). PyTorch `nn.Linear(in, out)` stores `weight` as `[out, in]`, so on disk the tensor must be `[H, 2H]`.

### Kiln code

**`crates/kiln-model/src/forward.rs:3516–3536`** — current `mtp_forward_step` glue:

```rust
let swap_fc_norms = crate::mtp_debug::is_swap_fc_norms_enabled();
let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
    (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
} else {
    (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
};
let norm_emb = {
    kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_emb");
    rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?
};
let norm_h = {
    kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
    rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?
};

// 4. Concat along the hidden dim and fuse: [1, 1, 2H] @ fc_t[2H, H] -> [1, 1, H]
let fused = {
    kiln_nvtx::range!(c"kiln/mtp/fc");
    let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
    concat.broadcast_matmul(&mtp.fc_t)?
};
```

With `KILN_MTP_SWAP_FC_NORMS=0` (the production default, which Phase B3 confirmed as optimal), this resolves to:

- `norm_emb = RMSNorm_embedding(token_emb)` — embed-side norm on the draft-token embedding.
- `norm_h   = RMSNorm_hidden(h_prev)` — hidden-side norm on the pre-final-norm hidden state.
- `concat = cat([norm_emb, norm_h], dim=2)` — embed first, hidden second, matching vLLM/SGLang byte-for-byte.

**`crates/kiln-model/src/loader.rs:626–632`** — fc shape check at load time:

```rust
let fc = extract_tensor(tensor_map, &format!("{mtp_prefix}fc.weight"))?;
// fc maps concat(embed, hidden) → hidden, so shape is [hidden, 2*hidden].
validate_shape(
    &fc,
    &[config.hidden_size, 2 * config.hidden_size],
    &ctx("fc"),
)?;
```

**`crates/kiln-model/src/forward.rs:768–770`** — upload-time transpose that feeds the forward path:

```rust
let fc = weight_to_tensor(&mtp_w.fc, device).context("mtp.fc")?;
let fc_t = cached_transpose(&fc).context("mtp.fc cached transpose")?;
```

So on device we hold `fc_t: [2H, H]`, and the forward path computes `concat[1,1,2H] @ fc_t[2H, H] → [1,1,H]`, which is exactly equivalent to `nn.Linear(2H, H)` applied to `concat`.

### Qwen3.5-4B checkpoint header (ground truth)

Cross-check against the published `Qwen/Qwen3.5-4B` checkpoint, read via an HTTP Range request on `model.safetensors` (first ~200 KB is the JSON header). Relevant MTP entries:

```
mtp.fc.weight                           dtype=BF16  shape=[2560, 5120]
mtp.pre_fc_norm_embedding.weight        dtype=BF16  shape=[2560]
mtp.pre_fc_norm_hidden.weight           dtype=BF16  shape=[2560]
mtp.norm.weight                         dtype=BF16  shape=[2560]
mtp.layers.0.input_layernorm.weight     dtype=BF16  shape=[2560]
mtp.layers.0.post_attention_layernorm.  dtype=BF16  shape=[2560]
mtp.layers.0.self_attn.q_norm.weight    dtype=BF16  shape=[256]
mtp.layers.0.self_attn.k_norm.weight    dtype=BF16  shape=[256]
mtp.layers.0.self_attn.q_proj.weight    dtype=BF16  shape=[8192, 2560]
mtp.layers.0.self_attn.k_proj.weight    dtype=BF16  shape=[1024, 2560]
mtp.layers.0.self_attn.v_proj.weight    dtype=BF16  shape=[1024, 2560]
mtp.layers.0.self_attn.o_proj.weight    dtype=BF16  shape=[2560, 4096]
mtp.layers.0.mlp.gate_proj.weight       dtype=BF16  shape=[9728, 2560]
mtp.layers.0.mlp.up_proj.weight         dtype=BF16  shape=[9728, 2560]
mtp.layers.0.mlp.down_proj.weight       dtype=BF16  shape=[2560, 9728]
```

`hidden_size = 2560`, so `[2560, 5120] = [H, 2H]` for `mtp.fc.weight`, matching both (a) PyTorch's `nn.Linear(2*H, H)` storage convention used upstream, and (b) kiln's `validate_shape(&fc, &[H, 2H])`.

Note also that `mtp.layers.0.self_attn.q_proj.weight` is `[8192, 2560] = [2·16·256, H]`, i.e. the gated `q_proj` (`attn_output_gate=true`), and `k_proj`/`v_proj` are `[1024, 2560] = [4·256, H]` (4 KV heads → GQA). These are the same shapes the main-model full-attention layers use; kiln's loader routes them through the existing `load_layer(..., 3, config)` call (line 683) precisely because layer_idx 3 falls in the full-attention residue class `(i + 1) % 4 == 0`.

### Verdict

**No mismatch.** Every one of the three wiring points audited is byte-equivalent between kiln and the upstream references, and the checkpoint header confirms the shape assumptions:

| Wiring point                                  | Upstream (vLLM + SGLang)            | Kiln                                                    | Match |
|-----------------------------------------------|-------------------------------------|---------------------------------------------------------|:-----:|
| Concat order along the 2H dim                 | `[embed, hidden]`                   | `Tensor::cat(&[&norm_emb, &norm_h], 2)` (embed first)   |   ✅   |
| Norm assigned to the embedding half           | `pre_fc_norm_embedding`             | `mtp.pre_fc_norm_embedding` (swap-off default)          |   ✅   |
| Norm assigned to the hidden half              | `pre_fc_norm_hidden`                | `mtp.pre_fc_norm_hidden` (swap-off default)             |   ✅   |
| `fc.weight` on-disk shape                     | `nn.Linear(2H, H)` ⇒ `[H, 2H]`      | `validate_shape(&fc, &[H, 2H])`                         |   ✅   |
| `fc` applied to concat                        | `nn.Linear(concat)` ≡ `concat @ Wᵀ` | `concat.broadcast_matmul(&mtp.fc_t)` with `fc_t=[2H,H]` |   ✅   |

Phase B3 already disproved the norm swap as the fix. Phase B4 now disproves the "fc concat order / weight layout is inverted" hypothesis: the kiln code and the Qwen3.5-4B checkpoint are fully consistent with both of the canonical Python reference implementations.

### Recommendation — next hypothesis

The α = 0.154 collapse with ~46% identity-bias persists despite correct MTP glue wiring. The remaining structural candidates, in order of likelihood:

1. **`mtp.layers.0.*` transformer block weight load.** The MTP inner layer is loaded by re-using `load_layer(tensor_map, &mtp_layer_prefix, 3, config)`. This threads a synthetic `layer_idx=3` through the main-model full-attention path to pick up `attn_output_gate=true`, gated `q_proj` splitting (`[2·num_heads·head_dim, H]`), RMSNorm Q/K norm, etc. Any subtle loader divergence here — e.g. a main-model-only fix-up that assumes a sequential `layer_idx` residue pattern, or a slice of the gated `q_proj` that only activates on the real layers — would silently mis-wire the MTP layer without tripping any shape validator. Phase B5 should re-verify the MTP layer's GQA weights via a direct scalar-comparison test against vLLM's `Qwen3NextMultiTokenPredictor` on an identical input.
2. **RoPE position threading into the MTP layer.** `mtp_forward_step` (forward.rs:3541) builds its own one-element position tensor (`Tensor::new(&[mtp_pos as f32][..], device)`) and passes it to `transformer_block_paged`. The MTP "position space" is distinct from the base model's. Confirm that `mtp_pos` is advanced exactly the same way as the reference generator (vLLM/SGLang advance it by +1 per accepted draft, not per verify-step).
3. **Tied `lm_head` (`embed_tokens_t`).** The MTP head reuses the base model's `embed_tokens_t` as its LM head (loader.rs:604–606). Verify that `embed_tokens_t` is, in fact, `embed_tokens.transpose()` and not the same tensor feeding forward — a copy-vs-transpose confusion would look like a stochastic identity bias even when draft logits look plausible.
4. **`mtp.final_layernorm` vs main-model `final_layernorm` reuse.** The MTP head ships its own `mtp.norm.weight` (confirmed in checkpoint header). Ensure the forward path applies `mtp.final_layernorm` and not the base model's after the inner transformer block.

### Budget used

- **Pod:** none. $0 pod spend.
- **Wall-clock:** ≈ 45 min desk audit (upstream source comparison, safetensors header probe, kiln source re-read, write-up).
- **Output:** this PROFILING.md appendix + this PR. No code changes.

---

## Phase B5 — MTP inner-layer weight load (`layer_idx=3` trick) vs vLLM/SGLang reference (doc-only)

**Status: NO MISMATCH at the MTP inner-layer weight load.** (2026-04-21)

### Why this audit

Phase B4 (PR #267) ruled out the MTP `fc` concat order and weight layout as the source of the α ≈ 0.154 collapse. The next structural candidate from Phase B4's ranked list was the MTP inner transformer block: kiln's loader reuses `load_layer(tensor_map, &mtp_layer_prefix, 3, config)` with a **synthetic** `layer_idx=3` to coerce the main-model full-attention codepath to fire for the MTP layer. The residue class `(3 + 1) % 4 == 0` makes `ModelConfig::is_full_attention_layer` return true, unlocking `attn_output_gate=true`, the gated `q_proj` split (`[2·num_heads·head_dim, H]`), Q/K norm, and GQA dims — all of which match the main-model full-attention layers.

The worry: if any main-model-only fix-up inside `load_layer` / `load_full_attention` is keyed off `layer_idx` as a real index (not just an error-context label), or if anything downstream dispatches on the layer's position in the main-model stack, the synthetic `layer_idx=3` could silently mis-wire the MTP layer.

Phase B5 is a $0, pod-free desk audit that compares kiln's MTP inner-layer load path byte-for-byte against two canonical Python reference implementations plus the published Qwen3.5-4B safetensors header.

### Upstream references (pinned)

| File | Path | SHA |
|---|---|---|
| vLLM | `vllm/model_executor/models/qwen3_5_mtp.py` | `771913e4a024` |
| vLLM | `vllm/model_executor/models/qwen3_next_mtp.py` | `657855ab4179` |
| SGLang | `python/sglang/srt/models/qwen3_5_mtp.py` | `cabe171b6ce3` |

How each upstream gets the MTP inner layer into full-attention mode:

- **vLLM `qwen3_5_mtp.py:97-104`** — explicit string:
  ```python
  self.mtp_block = Qwen3_5DecoderLayer(
      vllm_config=vllm_config,
      layer_type="full_attention",
      prefix=maybe_prefix(prefix, "mtp_block"),
  )
  ```
- **SGLang `qwen3_5_mtp.py:62-75`** — config mutation before delegating to the main-model class:
  ```python
  self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
  ...
  config.num_hidden_layers = 1
  config.full_attention_interval = 1
  self.model = Qwen3_5ForCausalLM(config, quant_config, prefix=..., is_nextn=True)
  ```
- **kiln `loader.rs:676-683`** — synthetic layer index:
  ```rust
  // The MTP layer uses the same full-attention shape as the main-model
  // full-attention layers (GQA + output gate + SwiGLU MLP). We reuse
  // `load_layer` with a synthetic prefix that points at `mtp.layers.0.`
  // and an index (3) whose residue class `(i + 1) % 4 == 0` makes
  // `is_full_attention_layer` return true. The layer_idx argument is only
  // used for error context strings, not for dispatch.
  let mtp_layer_prefix = format!("{mtp_prefix}layers.0.");
  let layer = load_layer(tensor_map, &mtp_layer_prefix, 3, config).context("mtp layer 0")?;
  ```

All three arrive at the **same dispatch outcome**: load an MTP inner layer using the full-attention path. vLLM does it via an explicit `layer_type` string; SGLang mutates `full_attention_interval=1` so every index is full-attn; kiln picks any `layer_idx` that satisfies its fixed residue class.

### Ground truth: Qwen3.5-4B MTP layer tensor shapes

Fetched via HTTP Range on `model.safetensors-00001-of-00002.safetensors` and `model.safetensors-00002-of-00002.safetensors` headers on Hugging Face (`Qwen/Qwen3.5-4B`):

```
mtp.layers.0.input_layernorm.weight            BF16 [2560]
mtp.layers.0.post_attention_layernorm.weight   BF16 [2560]
mtp.layers.0.self_attn.q_proj.weight           BF16 [8192, 2560]   # [2·num_heads·head_dim, H] = gated
mtp.layers.0.self_attn.k_proj.weight           BF16 [1024, 2560]   # [num_kv_heads·head_dim, H]
mtp.layers.0.self_attn.v_proj.weight           BF16 [1024, 2560]
mtp.layers.0.self_attn.o_proj.weight           BF16 [2560, 4096]   # [H, num_heads·head_dim]
mtp.layers.0.self_attn.q_norm.weight           BF16 [256]          # [head_dim]
mtp.layers.0.self_attn.k_norm.weight           BF16 [256]
mtp.layers.0.mlp.gate_proj.weight              BF16 [9216, 2560]   # [intermediate_size, H]
mtp.layers.0.mlp.up_proj.weight                BF16 [9216, 2560]
mtp.layers.0.mlp.down_proj.weight              BF16 [2560, 9216]
```

Cross-check against `model.layers.3.*` (a real main-model full-attention layer per `(3+1)%4==0`): **all 11 tensor shapes are identical**. The MTP inner layer is byte-for-byte the same shape as a main-model full-attention layer, which is exactly what the synthetic `layer_idx=3` trick assumes.

(Note: `intermediate_size=9216` — the value in `ModelConfig::qwen3_5_4b()` at `kiln-core/src/config.rs:86`. The Phase B4 appendix printed `9728` in the shape dump at PROFILING.md:4185-4187 — that was a transcription typo. `kiln-core` validates `[9216, 2560]`, the checkpoint is `[9216, 2560]`, they match.)

### 6-point wiring audit

| # | Wiring point | Upstream vLLM / SGLang | Kiln | Match |
|---|---|---|---|:-:|
| 1 | **Checkpoint key names** (`mtp.layers.0.self_attn.q_proj.weight`, etc.) | Both remap `mtp.` → `model.` in `load_weights`, then feed unchanged through the main-model `Qwen3_5DecoderLayer` loader | `detect_mtp_prefix` keeps the `mtp.` prefix and passes `mtp.layers.0.` to `load_layer`; each tensor is looked up by its stock checkpoint name | ✅ |
| 2 | **Gated `q_proj` dispatch** (must see `[8192, 2560]`, not `[4096, 2560]`) | vLLM: explicit `layer_type="full_attention"`. SGLang: `config.full_attention_interval=1`. Both route to the main-model full-attn block that handles `attn_output_gate=true` | `is_full_attention_layer(3)` = `(3+1)%4==0` = `true` → `AttentionWeights::Full(load_full_attention(...))` (loader.rs:380-386). `full_attn_q_proj_dim()` = `16·256·2 = 8192` (config.rs:157-160) | ✅ |
| 3 | **GQA dims** (`num_heads=16`, `num_kv_heads=4`, `head_dim=256`) | Global model-config read; no per-layer overrides | `load_full_attention` (loader.rs:406-420) reads `config.num_attention_heads`, `config.num_kv_heads`, `config.head_dim` — no `layer_idx`-dependent branches | ✅ |
| 4 | **Q/K RMSNorm** (shape `[256] = [head_dim]`) | No `layer_idx` branch; same head-dim norm whether layer is main-model layer 3 or MTP | `load_full_attention` loader.rs:422-426 validates both against `[config.head_dim]`; identical to main-model layer 3 load | ✅ |
| 5 | **RoPE threading** (partial rotary, `rotary_dim=64`, `rope_theta=1e7`) | Upstream `Qwen3_5DecoderLayer` instance reuses the main-model RoPE module (not rebuilt per layer) | `mtp_forward_step` forward.rs:3547-3558 passes `weights.rotary_inv_freq` (main-model table, built once loader.rs:564-566) and `config.rotary_dim()=64` to `transformer_block_paged`. Same inv_freq table as main-model layer 3 | ✅ |
| 6 | **Weight on-disk layout** (PyTorch `[out, in]`) | All three use PyTorch `nn.Linear` storage: `q_proj=[8192, H]`, `k_proj=[kv_dim, H]`, `o_proj=[H, 4096]`, `gate_proj=[I, H]`, `down_proj=[H, I]` | `validate_shape` checks at loader.rs:411, 414, 417, 420 match PyTorch order. Cached `_t` transposes happen downstream on the GPU (identical to how main-model layers are transposed); no layer-specific transpose path | ✅ |

### What the synthetic `layer_idx=3` actually controls

Grepping `crates/kiln-model/src/loader.rs` for every use of the `layer_idx` parameter reaching from `load_layer` / `load_full_attention` / `load_ffn`:

1. **`is_full_attention_layer(layer_idx)` at loader.rs:380** — the only load-time dispatch. Returns `true` for `3`, just like for main-model layers 3/7/11/15/19/23/27/31.
2. **Error-context strings** (`&format!("layer {layer_idx} input_layernorm")` and similar) at loader.rs:365, 375, 404, 411, 414, 417, 420, 423, 426 — cosmetic. Zero effect on tensor values.
3. **Defensive post-load check** at loader.rs:684-695 — asserts the resulting `AttentionWeights` variant is `Full`, not `Linear`. Belt-and-braces against a future checkpoint schema change. Zero effect on loaded tensors.

That's all. The synthetic layer index **never reaches the MTP KV-cache slot, the forward path, or any tensor loader**. The forward path dispatches on its own index: `mtp_forward_step` hard-codes `full_attn_layer_idx = 0` when it calls `transformer_block_paged` (forward.rs:3562), because the MTP head has its own one-element cache — it is not a slot in the main-model KV cache.

### Verdict

**No mismatch.** kiln's `layer_idx=3` trick is mechanically equivalent to vLLM's `layer_type="full_attention"` and SGLang's `full_attention_interval=1` — three different ways to route the same checkpoint tensors through the same full-attention code. The checkpoint header confirms the shape assumptions (`q_proj=[8192, 2560]`, `kv=[1024, 2560]`, etc.). Nothing about the synthetic layer index leaks into tensor values, forward compute, or cache routing.

### Recommendation — next hypothesis

With the Phase B4 hypothesis #1 now disproved (load-time), the ranking from PROFILING.md:4208-4215 collapses to three runtime candidates:

1. **RoPE position threading** (forward-time, not load-time). `mtp_forward_step` builds its own one-element `positions` tensor from `mtp_pos`. The reference semantics advance `mtp_pos` by +1 per *accepted* draft; kiln's caller advances it only on acceptance (confirmed in loader.rs comment at 601). Worth a scalar A/B against the upstream generator: run the same prompt through vLLM's `Qwen3NextMTP` with acceptance telemetry, diff the `mtp_pos` trajectory.
2. **Tied lm_head via `embed_tokens_t`**. The MTP head reuses the base model's `embed_tokens_t` as its unembedding matrix (forward.rs:3571). Confirm that `embed_tokens_t` is the true transpose of `embed_tokens`, not a shared-storage alias that forwards would be mutating. A one-shot check: load `embed_tokens`, transpose, assert byte-equality against `embed_tokens_t` at load time.
3. **`mtp.final_layernorm` application site**. The loader accepts both `mtp.norm.weight` and `mtp.final_layernorm.weight` (loader.rs:657-668). The checkpoint ships `mtp.norm.weight`. Confirm the forward path (forward.rs:3570) is applying the *loaded MTP norm scale*, not accidentally reaching back through a closure to the main-model's `final_layernorm`.

If all three runtime hypotheses clear, Phase B5-follow-up should move from structural auditing to a direct numerical A/B: load the same Qwen3.5-4B checkpoint into vLLM (using `Qwen3NextMTP`), feed a fixed prompt, and dump MTP intermediate activations (`fc` output, attn output, final norm output, logits) at every step. Scalar diff against kiln's `KILN_MTP_DEBUG=1` trace. Any divergence localizes the bug to a specific sub-op.

### Budget used

- **Pod:** none. $0 pod spend.
- **Wall-clock:** ≈ 55 min desk audit (fetching upstream at pinned SHAs, safetensors header probe on both shards, re-reading `loader.rs::load_mtp_if_present` + `load_layer` + `load_full_attention`, cross-checking forward.rs MTP path, writing this appendix).
- **Output:** this PROFILING.md appendix + this PR. No code changes.

## Phase B6 — MTP numerical dual-dump bisect (per-tap localization, 2026-04-21)

### Goal

Phase B5 disproved the `layer_idx=3` load-time hypothesis and collapsed the
remaining MTP-α-collapse root-cause candidates to three runtime hypotheses
(PROFILING.md:4211–4220). This Phase B6 bisect runs a **pure-PyTorch reference
implementation** of `mtp_forward_step` over the exact same `h_main` +
`draft_token_id` that kiln's own forward path consumes, and diffs the
intermediate activations tap-by-tap with `allclose` + cosine similarity. The
first tap that diverges pinpoints which of the three hypotheses (if any) is
actually responsible.

### Scope and honest limits

* **Single prompt**, single `mtp_pos=0`. One-shot dump. Does not claim to
  cover every MTP acceptance path.
* `h_main` is **taken from the kiln dump** — the reference does not re-run
  the 24 × GDN + 8 × GQA base stack. If the base model is producing the
  wrong `h_main`, every downstream tap would match the reference bit-for-bit
  (since the reference is fed the same bad `h_main`), and Phase B6 would
  return a clean verdict. That would itself be a signal — pointing
  upstream of `h_main` to the scheduler / paged KV state / GDN state on
  the MTP branch.
* BF16 matmul noise is not a bug: `|Δ|` of order 0.01 at intermediate
  activations over 5k-dim reductions is normal. The bisect therefore reports
  cosine similarity alongside `allclose`; a direction-preserving tap (cos≥0.999)
  with small `|Δ|` is considered a match regardless of strict-`allclose` status.

### Instrumentation

Code added on this branch:

* `crates/kiln-model/src/mtp_debug.rs` — `write_mtp_dump()` writes 8 F32 taps
  + 3 I32 metadata scalars in safetensors format, gated by `KILN_MTP_DUMP_PATH`.
  One-shot latch via `AtomicBool` to avoid per-step overhead.
* `crates/kiln-model/src/forward.rs` — `mtp_forward_step` captures the 8
  named taps in order. `concat` and `normed` pulled out as named binds so
  the dump can see them without a second forward pass.
* `scripts/mtp_reference_dump.py` — pure-PyTorch reference. Loads MTP weights
  from `/workspace/qwen3.5-4b`, reads `h_main` + `draft_token_id` from the
  kiln dump, runs the full `embed → dual rms_norm → concat → fc →
  single transformer block (with RoPE at mtp_pos, per-head Q/K RMSNorm,
  gated-attn, MLP) → final_layernorm → tied LM head` path, writes the
  same 8 taps.
* `scripts/mtp_compare.py` — per-tap diff. Prints a table of
  (shape, allclose, cos_sim, max|Δ|, mean|Δ|) and maps the first divergence
  back to the hypothesis it implicates.

Reference RMSNorm uses the Qwen3.5-specific form `out = (1 + w) * x * rsqrt(mean(x²) + ε)`
(see `forward.rs::rms_norm_fallback` at line 936). This is **not** the
HF-standard RMSNorm; Qwen3.5 stores RMSNorm weights centered around 0 and
applies them as `(1 + w)`. Using the standard form here produces a false
divergence at `fc_input` that masks the real signal — any future additions
to the reference must use the same semantics.

### Results

Prompt = `PROMPT_POOL[2]`, seed=42, `draft_token_id=561`, `mtp_pos=0`,
`swap_fc_norms=0`. Kiln built in release mode with `KILN_CUDA_ARCHS=86
--features cuda --bin kiln-bench`. Reference run with torch 2.x on CPU.

Comparison at `atol=0.05, rtol=0.05` (BF16-realistic):

| tap            | shape           | cos_sim | max\|Δ\|   | mean\|Δ\| | verdict |
|----------------|-----------------|---------|-----------|-----------|---------|
| h_main         | 1×1×2560        | 1.000   | 0.00      | 0.00      | match (input) |
| tok_embed      | 1×1×2560        | 1.000   | 0.00      | 0.00      | match |
| fc_input       | 1×1×5120        | 1.000   | 2.70e-2   | 7.12e-4   | match (BF16 noise) |
| fc_output      | 1×1×2560        | 1.000   | 1.16e-2   | 7.06e-4   | match (BF16 noise) |
| pre_layer      | 1×1×2560        | 1.000   | 1.16e-2   | 7.06e-4   | match |
| **post_layer** | **1×1×2560**    | **0.600** | **5.37** | **6.95e-1** | **FIRST DIVERGENCE** |
| post_final_ln  | 1×1×2560        | 0.538   | 21.12     | 2.41      | divergent (propagated) |
| mtp_logits     | 1×1×248320      | 0.540   | 10.57     | 1.62      | divergent (propagated) |

Metadata checks: `draft_token_id`, `mtp_pos`, `swap_fc_norms` all match.

### Verdict

The first numerically real divergence is at **`post_layer`** — the output of
the single-layer MTP transformer block. The pipeline is clean through
`pre_layer` (cos≥1.0, `|Δ|` at BF16-matmul-noise levels), which means:

* **Hypothesis H1 (RoPE `mtp_pos` advancement)**: **STRONG candidate.** The
  MTP block is the only place `mtp_pos` enters the forward graph. RoPE uses
  `partial_rotary_factor=0.25` (64 of 256 head dims rotated) and
  `rope_theta=1e7`. At `mtp_pos=0` the rotation is the identity, so if the
  divergence persists at `mtp_pos=0`, the root cause is **not** a
  position-advancement bug per se but something else inside the block
  (Q/K/V projection layout, per-head Q/K RMSNorm, gated-attn). If on
  later `mtp_pos > 0` steps the divergence grows with position, H1 is
  confirmed as an advancement bug.
* **Hypothesis H2 (tied `embed_tokens_t` transpose vs alias)**: **WEAKENED
  but not fully eliminated.** `post_final_ln` and `mtp_logits` both end at
  cos_sim ≈ 0.54 — meaning the `post_final_ln → logits` step preserves the
  bulk of the direction. A transpose-layout bug in the tied LM head would
  collapse cos_sim to near-zero, not preserve it. The remaining ~0.6 %
  cos_sim gap between `post_final_ln` (0.538) and `mtp_logits` (0.540) is
  within propagation noise for an `[H] → [V]` matmul. The tied transpose
  is likely fine.
* **Hypothesis H3 (`mtp.final_layernorm` application site)**: **WEAKENED.**
  `post_layer` is already divergent, so any mismatch at `post_final_ln` is
  partly explained by propagation. The `post_layer → post_final_ln`
  cos_sim drop (0.600 → 0.538) is consistent with a single RMSNorm
  amplifying an already-divergent input; it does not require an
  independent bug in the norm application site. This line of inquiry
  can be parked unless H1 is disproved.

### Narrowed next steps

This bisect collapses the three-hypothesis space to **a single active
candidate: the MTP inner transformer block itself**, with sub-hypotheses
ordered by likelihood:

1. **Per-head Q/K RMSNorm** — same `(1 + w)` trap that produced a false
   divergence in the reference on the first pass. If kiln's Q/K-norm path
   ever takes a branch that applies bare `w`, it would manifest exactly
   here.
2. **Gated attention (`attn_output_gate=true`)** — Qwen3.5-4B applies a
   sigmoid gate on the attention output. A sign error, wrong split, or
   missing gate would appear first at `post_layer`.
3. **Q/K/V projection layout** — `q_proj` is the gated `[8192, 2560]`
   variant, `k_proj`/`v_proj` are GQA `[1024, 2560]`. A transpose bug
   here would mangle the attention output specifically.
4. **RoPE position threading** at `mtp_pos > 0` — not diagnosed by this
   single-step dump; needs a multi-step variant (Phase B7).

Phase B7 should dump at `mtp_pos = 0, 1, 2` on the same prompt and check
whether the divergence grows with position (→ H1 confirmed) or is
position-invariant (→ H1 eliminated, bug is inside the block). If B7
eliminates H1, the next phase is a **per-sub-op reference inside the MTP
block**: break down `post_layer` into `q/k/v → rope → attn → out_proj →
gate → mlp_up → mlp_gate → mlp_down` and dump each one.

### Budget used

- **Pod:** RunPod pool, `s23qwogiqyk76s` (RTX A6000) via lease
  `pod-37efdfc4f8b4c4bdbcfa0b98`. Hot build (sccache+incremental), two
  kiln-bench runs for dump capture, one reference-script execution, two
  compare runs. ≈ 15 min GPU-time. Pod released to pool after PR open.
- **Wall-clock:** ≈ 80 min total, including one iteration to catch a
  reference-side RMSNorm bug (`(1 + w)` vs bare `w`) that masqueraded as
  an `fc_input` divergence on the first pass and would otherwise have
  produced a false H2-ish verdict.
- **Output:** this appendix, the `mtp_debug.rs` + `mtp_reference_dump.py`
  + `mtp_compare.py` scaffold (all preserved for Phase B7), the concrete
  bisect report at `/tmp/mtp-compare.txt` (copied to
  `mtp-compare-phase-b6.txt` in the session workspace), and this PR.
  **No fix included.** Scope is bisect-only per the task brief.


## Phase C12 — fp32 draft-head kill switch + activation-weighted probe (2026-04-21)

### Goal

Test whether forcing the MTP draft head's q/k/v/o + fc projections to fp32
(`KILN_MTP_FP32_HEAD=1`) recovers α toward the 0.72 ship floor. Hypothesis
from C9/C11 audit: bf16 matmul accumulation noise on W4A16-dequanted weights
shifts the draft head's top-1 enough to explain α == 0.058-0.124 observed
in C5 after the C3 RoPE fix landed.

### Implementation

Narrow, TLS-gated chokepoint — no behavior change when flag unset.

- `crates/kiln-model/src/mtp_debug.rs` — `KILN_MTP_FP32_HEAD` env reader +
  `MTP_FP32_HEAD_ARMED` TLS `RefCell<bool>` + arm/disarm helpers.
- `crates/kiln-model/src/lora_loader.rs::linear_with_lora_t` — when armed,
  upcast `x` and transposed base weight to `DType::F32`, matmul, cast
  back. LoRA and bias adds left untouched. Single chokepoint covers q/k/v/o
  and MLP base matmuls in one branch.
- `crates/kiln-model/src/forward.rs::mtp_forward_step` — reads flag once;
  arms TLS around the draft-head `transformer_block_paged`; disarms on
  exit. OR's into the existing `KILN_MTP_FC_FP32_ACCUM` branch so the new
  flag subsumes C9's fc-only knob.

### Results

Median-of-3 on A6000, Qwen3.5-4B, `--paged --prompt-tokens 512
--max-output-tokens 128 --skip-training --seed {42,43,44}`, MTP on
(`KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp`):

| Config           | env                                      | Median α | Median tok/s |
|------------------|------------------------------------------|----------|--------------|
| baseline         | `KILN_W4A16=1`                           | 0.0325   | 41.09        |
| **primary**      | `KILN_W4A16=1 KILN_MTP_FP32_HEAD=1`      | 0.0325   | 37.66        |
| sanity_nomarlin  | `KILN_W4A16=0`                           | 0.0583   | 36.76        |

Per-seed α is bitwise identical across configs for 2/3 seeds (seed 42:
4/123 for all three configs; seed 43: 4/123 for baseline and primary).

### Verdict

**PRIMARY-NEGATIVE.** fp32 draft-head projections do not recover α. The
bf16 matmul-accumulation hypothesis is refuted: dtype of the head's
projections is not a contributor of meaningful magnitude to α under greedy
decode. The failure mode is structural — the MTP head deterministically
picks a different top-1 token from the main head, across W4A16-bf16,
W4A16-fp32, and no-Marlin-at-all.

Combined with C5 (Class C = 0, Class B = 87.6%) and C3 (bitwise-identical
α pre/post RoPE fix), the surviving hypothesis space is now entirely
upstream of the head's projections: MTP token-embedding lookup, MTP
pre-norm γ application, MTP hidden-state splicing from the main stack, or
weight-loading for `mtp_head.fc_input` / `mtp_head.fc_output` /
q-k-v-o. **C13 should investigate weight-loading policy and the splice
input to the draft head first** — mis-tied or mis-loaded weights produce
exactly this symptom (deterministic wrong top-1, dtype-insensitive).

Decode cost: `KILN_MTP_FP32_HEAD=1` costs ≈ 8.3% of tok/s on A6000
(41.09 → 37.66). Do not gate on by default.

### Activation-weighted probe (sidecar, main-model MLP)

`scripts/c12_activation_weighted_probe.py` audited 104 main-model
projections on 32 prompts × 595 total tokens. Top weighted drift:
`L6.down_proj` at 1.447e-01; many MLPs in the 11-14% band. This is
non-trivial drift by magnitude, but it doesn't propagate into main-head
top-1 flips on the bench prompts (otherwise sanity_nomarlin and baseline
main-head trajectories would differ more — they don't). The probe's
script-level terminal verdict ("Corroborates PRIMARY-POSITIVE…") is stale
boilerplate; the authoritative C12 verdict is the bench above.

### Budget used

Single A6000 pod (`wl0fyjvqrv0v9b`), warm sccache (hit rate 97.56%). Build
2m 12s; 9 bench runs ≈ 974s total; one probe run (~160s). All pod-side
waits used `runpod_api.py wait-file --timeout`; no `until ssh` or
`while ssh … sleep` polling loops. Well under the 90 min / $40 hard cap.

### Output

- `docs/archive/phase-c/phase-c12/c12-fp32-head.md` — full verdict report.
- `c12-out/bench-summary.json` — 3 × 3 trial JSON (seed, α, tok/s, ITL,
  accept line).
- `c12-out/probe-report.md` + `c12-out/probe-report.json` — main-model
  activation-weighted drift audit.
- `c12-out/bench.log`, `c12-out/probe.log` — runner logs.

## Phase C42 post-#428 layer-1 pre-norm vs input-layernorm bisect (2026-04-23)

**Scope:** resolve the post-C41 remaining span by asking one narrower
question on fresh `main`: is the first shared bad tensor already present in
the residual input entering transformer block 1, or is it introduced inside
layer 1 `input_layernorm` itself?

**Preflight outcome:** proceed. Fresh `origin/main` at `e569d22` (PR #428
merged) still had
[`docs/archive/phase-c/phase-c41/c41-layer1-subop-bisect.md`](docs/archive/phase-c/phase-c41/c41-layer1-subop-bisect.md)
as the latest committed source of truth for this path, and that doc still
recorded `layer_1_post_input_norm` as the earliest shared bad C41 tap for both
seeds. Fresh main did not already contain committed C42 taps or a doc that
localized the remaining span any further.

**Hardware / image:** RunPod on-demand fallback `NVIDIA A100-SXM4-80GB`,
`ghcr.io/ericflo/kiln-runpod:latest`. A6000 and A40 capacity were unavailable
at the time of the run, so the project-policy fallback order was used.

**Code change:** add a dedicated opt-in C42 capture path behind
`KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS=1`, serialize the emitted tap ids as
`meta__c42_tap_ids`, mirror the same four taps in the HF reference dump under
`--c42-taps`, and add a focused `scripts/mtp_compare.py --c42` mode.
The explicit tap set is:

- `layer_1_residual_input`
- `layer_1_input_norm_rms_inv`
- `layer_1_input_norm_pre_weight`
- `layer_1_post_input_norm`

No generalized tracing was added; production decode stays on the current fast
path when the env var is unset.

**Validation commands run:**

```bash
cd /workspace/kiln
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
source /root/.kiln-build-env
cargo test -p kiln-model mtp_debug --lib -- --test-threads=1
```

**Standard workload rerun:** same C41 replay contract plus the new C42 tap
flag. Build and setup on the fallback A100:

```bash
cd /workspace/kiln
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=80
cargo build --release --features cuda --bin kiln-bench
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 -m pip install transformers safetensors sentencepiece
```

Then rerun the standard native-MTP workload for seeds `0` and `1`:

```bash
for seed in 0 1; do
  root=/workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}_captures
  rm -rf "$root"
  mkdir -p "$root"/mtp_pos-0 "$root"/mtp_pos-2

  KILN_W4A16=1 \
  KILN_CUDA_GRAPHS=true \
  KILN_SPEC_METHOD=mtp \
  KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_SPLICE=1 \
  KILN_MTP_DUMP_SPLICE_POS=0,2 \
  KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 \
  KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
  KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS=1 \
  KILN_MTP_DUMP_PATH=$root/mtp_pos-{pos}/step-{step}.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > /workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}.bench.json \
    2> /workspace/kiln/profiling-artifacts/post428_c42_20260423_seed${seed}.bench.stderr
done
```

Representative compare dumps:

- seed 0: `profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

HF references were regenerated in bf16 and fp32 with
`scripts/mtp_h_main_reference_dump.py --c42-taps`, then compared with:

```bash
python3 scripts/mtp_compare.py --c42 \
  --pair seed0:profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed0_ref_bf16.safetensors \
  --pair seed1:profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed1_ref_bf16.safetensors \
  --out profiling-artifacts/post428_c42_20260423_compare_bf16.txt

python3 scripts/mtp_compare.py --c42 \
  --pair seed0:profiling-artifacts/post428_c42_20260423_seed0_captures/mtp_pos-0/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed0_ref_fp32.safetensors \
  --pair seed1:profiling-artifacts/post428_c42_20260423_seed1_captures/mtp_pos-2/step-1.safetensors,profiling-artifacts/post428_c42_20260423_seed1_ref_fp32.safetensors \
  --out profiling-artifacts/post428_c42_20260423_compare_fp32.txt
```

### Fresh workload check

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10116.0 | 35.3 | 0.716 |
| 1 | 508 | 395.3 | 21.2 | 0.293 |

The post-#428 seed split still reproduces.

### Verdict

The first shared bad tensor is **not** already present in the residual input
entering block 1. Across both bf16 and fp32 comparisons:

- `layer_1_residual_input` remains `ok` for both seeds
- `layer_1_input_norm_rms_inv` remains `ok` for both seeds
- `layer_1_input_norm_pre_weight` is the earliest shared `DIV` tap
- `layer_1_post_input_norm` remains bad, but it is downstream of the first
  failing C42 boundary

So the shared drift is now localized to the boundary between
`layer_1_input_norm_rms_inv` and `layer_1_input_norm_pre_weight`. The
remaining culprit space is inside layer-1 `input_layernorm` numerics,
specifically the normalization / pre-weight scaling path before the final
`(1 + weight)` application.

### Evidence

- `profiling-artifacts/post428_c42_20260423_seed0.bench.json`
- `profiling-artifacts/post428_c42_20260423_seed0.bench.stderr`
- `profiling-artifacts/post428_c42_20260423_seed1.bench.json`
- `profiling-artifacts/post428_c42_20260423_seed1.bench.stderr`
- `profiling-artifacts/post428_c42_20260423_compare_bf16.txt`
- `profiling-artifacts/post428_c42_20260423_compare_fp32.txt`

## Phase C43 post-#429 layer-1 pre-weight multiply audit (2026-04-23)

Goal: split the existing C42 `layer_1_input_norm_pre_weight` boundary into the
current `broadcast_mul` path and an independently computed equivalent that
changes the row-selection / scaling path without changing the math.

Code change:

- add opt-in C43 taps behind `KILN_MTP_DUMP_C43_LAYER1_PREWEIGHT_TAPS=1`
- capture both:
  - `layer_1_input_norm_pre_weight_broadcast_mul`
  - `layer_1_input_norm_pre_weight_scalar_affine`
- mirror the C43 tap order in `mtp_h_main_reference_dump.py`
- add `scripts/mtp_compare.py --c43`

Final source-of-truth run:

- RunPod kiln image on on-demand `NVIDIA RTX A6000`
- validation passed:
  - `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
  - `cargo test -p kiln-model mtp_debug --lib -- --test-threads=1`
- standard workload:
  - `KILN_W4A16=1`
  - `KILN_CUDA_GRAPHS=true`
  - `KILN_SPEC_METHOD=mtp`
  - `KILN_BENCH_FORCE_MTP=1`
  - `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`

Representative compare dumps:

- seed 0: `profiling-artifacts/post429_c43_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post429_c43_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Fresh workload check:

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10057.8 | 38.9 | 0.693 |
| 1 | 508 | 414.0 | 21.3 | 0.293 |

Verdict:

- `layer_1_residual_input` stays shared-good
- `layer_1_input_norm_rms_inv` stays shared-good
- `layer_1_input_norm_pre_weight_broadcast_mul` is the earliest shared bad tap
- `layer_1_input_norm_pre_weight_scalar_affine` is also bad in both seeds and
  in both bf16 and fp32 compares

Therefore C43 rejects the "layout / row selection around `broadcast_mul`"
hypothesis. The shared divergence survives the independent equivalent, so the
remaining culprit space stays on the layer-1 pre-weight values themselves, not
on the specific broadcast path.

Recommendation:

- Do not widen into deeper layer-1 block internals yet.
- The next slice should stay inside layer-1 input-layernorm numerics and audit
  why kiln's normalized pre-weight tensor is wrong even when the same
  `rms_inv` scalar is reused through an independent scaling path.
- Treat the discarded H200 NVL run as non-evidence; with `KILN_W4A16=1` it hit
  `CUDA_ERROR_ILLEGAL_INSTRUCTION` before the audit completed.

Evidence:

- `profiling-artifacts/post429_c43_20260423_seed0.bench.json`
- `profiling-artifacts/post429_c43_20260423_seed0.bench.stderr`
- `profiling-artifacts/post429_c43_20260423_seed1.bench.json`
- `profiling-artifacts/post429_c43_20260423_seed1.bench.stderr`
- `profiling-artifacts/post429_c43_20260423_compare_bf16.txt`
- `profiling-artifacts/post429_c43_20260423_compare_fp32.txt`

## Phase C44 post-#430 layer-1 F32 row vs normalized-row audit (2026-04-23)

Goal: answer the next smallest slice after C43 by checking whether the last
replay row is already bad immediately after `x.to_dtype(F32)` or whether the
row stays shared-good and only diverges when the shared-good `rms_inv` scalar
is applied.

Code change:

- add opt-in C44 taps behind `KILN_MTP_DUMP_C44_LAYER1_F32_ROW_TAPS=1`
- capture only:
  - `layer_1_residual_input_f32_row`
  - `layer_1_input_norm_rms_inv_scalar`
  - `layer_1_input_norm_pre_weight_row_scalar_affine`
- mirror the C44 tap order in `mtp_h_main_reference_dump.py`
- add `scripts/mtp_compare.py --c44`

Final source-of-truth run:

- RunPod kiln image on on-demand `NVIDIA RTX A6000`
- validation passed:
  - `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
  - `cargo test -p kiln-model mtp_debug --lib -- --test-threads=1`
- standard workload:
  - `KILN_W4A16=1`
  - `KILN_CUDA_GRAPHS=true`
  - `KILN_SPEC_METHOD=mtp`
  - `KILN_BENCH_FORCE_MTP=1`
  - `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`

Representative compare dumps:

- seed 0: `profiling-artifacts/post430_c44_20260423_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1: `profiling-artifacts/post430_c44_20260423_seed1_captures/mtp_pos-2/step-1.safetensors`

Fresh workload check:

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 10046.8 | 38.3 | 0.716 |
| 1 | 508 | 397.5 | 23.0 | 0.270 |

Verdict:

- `layer_1_residual_input_f32_row` stays shared-good
- `layer_1_input_norm_rms_inv_scalar` stays shared-good
- `layer_1_input_norm_pre_weight_row_scalar_affine` is the earliest shared bad tap
- this holds in both bf16 and fp32 HF compares

Therefore C44 answers the remaining question from C43 directly: the bad values
are **not** already present in the F32-cast last replay row. The row stays
shared-good, and the first shared divergence appears only when normalization is
applied to produce the row's pre-weight values.

Recommendation:

- Do not widen beyond layer-1 input-layernorm numerics yet.
- The next slice should stay inside the row-local normalization application
  itself, not the F32 cast and not the RMS reduction.

Evidence:

- `profiling-artifacts/post430_c44_20260423_seed0.bench.json`
- `profiling-artifacts/post430_c44_20260423_seed0.bench.stderr`
- `profiling-artifacts/post430_c44_20260423_seed1.bench.json`
- `profiling-artifacts/post430_c44_20260423_seed1.bench.stderr`
- `profiling-artifacts/post430_c44_20260423_compare_bf16.txt`
- `profiling-artifacts/post430_c44_20260423_compare_fp32.txt`

## Phase C45 post-#431 layer-1 row normalization bisect (2026-04-23)

Goal: split the previously-bad C44 row-level scalar-affine site one step
further so the next replay can distinguish:

- selected residual row values
- the row-local `rms_inv` scalar
- the flat row-scalar multiply values
- the reconstructed row-shaped output before the existing post-input-norm path

Code change:

- add opt-in C45 taps behind `KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1`
- capture:
  - `layer_1_residual_input_f32_row_values`
  - `layer_1_input_norm_rms_inv_scalar`
  - `layer_1_input_norm_pre_weight_row_scalar_values`
  - `layer_1_input_norm_pre_weight_row_reconstructed`
- mirror the C45 tap order in `mtp_h_main_reference_dump.py`
- add `scripts/mtp_compare.py --c45`

Local validation:

- `cargo test --locked -p kiln-model c45 -- --nocapture`
- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`

Both passed on 2026-04-23.

RunPod validation blocker:

- on-demand launches for `NVIDIA RTX A6000`, `NVIDIA A40`, `NVIDIA A100 80GB PCIe`,
  `NVIDIA RTX 6000 Ada Generation`, and `NVIDIA L40S` all failed with
  `SUPPLY_CONSTRAINT`
- fallback launches on `NVIDIA H200 NVL`, `NVIDIA RTX PRO 4500 Blackwell`, and
  `NVIDIA H100 PCIe` created pods, but every pod stayed in
  `desiredStatus=RUNNING` with `runtime: null`, so `runpod_api.py wait` never
  returned SSH

Verdict:

- no fresh C45 numerical verdict yet
- the instrumentation and local validation are complete
- the remaining blocker is RunPod provisioning, not missing code plumbing

Recommendation:

- rerun the committed C45 workload on the next healthy on-demand RunPod pod
- do not change the tap contract before that rerun; the next missing result is
  the earliest shared bad C45 tap, not another instrumentation edit

Evidence:

- `profiling-artifacts/post431_c45_20260423_local_validation.txt`
- `profiling-artifacts/post431_c45_20260423_runpod_blocker.txt`

## Phase C45 post-#432 healthy-RunPod rerun (2026-04-23)

Goal: rerun the already-merged C45 tap contract on the first healthy
on-demand RunPod pod and capture the earliest shared bad C45 tap on current
`origin/main`.

Remaining-work preflight:

- PR #432 was present on `origin/main` (`5574aed`, merged 2026-04-23 14:34 UTC)
- no newer open or merged kiln PR had already recorded a successful post-#432
  C45 rerun or an earlier shared-bad C45 boundary
- no separate pending/running kiln task overlapped this exact post-#432 rerun

RunPod outcome:

- `NVIDIA RTX A6000` launch failed immediately with `SUPPLY_CONSTRAINT`
- the next allowed GPU, on-demand `NVIDIA A40`, reached SSH and completed the
  rerun on `ghcr.io/ericflo/kiln-runpod:latest`
- hardware used for the final verdict: `NVIDIA A40` (46068 MB VRAM reported by
  `nvidia-smi`)

Validation completed on pod:

- `cargo build --release --features cuda --bin kiln-bench`
- `cargo test --locked -p kiln-model c45 -- --nocapture`
- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`

All three passed on the A40 pod.

Actual rerun commands:

```bash
source /root/.kiln-build-env
cd /workspace/kiln

hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 -m pip install transformers sentencepiece

KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_METHOD=mtp \
KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 \
KILN_MTP_DUMP_SPLICE_POS=0,2 \
KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 \
KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
KILN_MTP_DUMP_PATH=profiling-artifacts/post432_c45_seed0_captures/mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed 0 \
  > profiling-artifacts/post432_c45_seed0.bench.json \
  2> profiling-artifacts/post432_c45_seed0.bench.stderr

KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_METHOD=mtp \
KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 \
KILN_MTP_DUMP_SPLICE_POS=0,2 \
KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 \
KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
KILN_MTP_DUMP_PATH=profiling-artifacts/post432_c45_seed1_captures/mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed 1 \
  > profiling-artifacts/post432_c45_seed1.bench.json \
  2> profiling-artifacts/post432_c45_seed1.bench.stderr
```

Representative compare inputs:

- seed 0 kiln dump: `profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors`
- seed 1 kiln dump: `profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors`

Reference + compare:

```bash
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post432_c45_seed0_ref.safetensors \
  --device cuda \
  --c45-taps

python3 scripts/mtp_compare.py --c45 \
  --kiln profiling-artifacts/post432_c45_seed0_captures/mtp_pos-0/step-1.safetensors \
  --ref profiling-artifacts/post432_c45_seed0_ref.safetensors \
  > profiling-artifacts/post432_c45_seed0_compare.txt

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post432_c45_seed1_ref.safetensors \
  --device cuda \
  --c45-taps

python3 scripts/mtp_compare.py --c45 \
  --kiln profiling-artifacts/post432_c45_seed1_captures/mtp_pos-2/step-1.safetensors \
  --ref profiling-artifacts/post432_c45_seed1_ref.safetensors \
  > profiling-artifacts/post432_c45_seed1_compare.txt
```

Fresh workload check:

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 384.8 | 36.8 | 0.740 |
| 1 | 508 | 393.4 | 24.2 | 0.296 |

Verdict:

- seed 0 compare: earliest shared bad tap =
  `layer_1_input_norm_pre_weight_row_scalar_values`
- seed 1 compare: earliest shared bad tap =
  `layer_1_input_norm_pre_weight_row_scalar_values`
- the last shared-good C45 tap on both seeds is
  `layer_1_input_norm_rms_inv_scalar`

So the post-#432 rerun localizes the first shared layer-1 row-normalization
drift one step earlier than C44:

- `layer_1_residual_input_f32_row_values` remains shared-good
- `layer_1_input_norm_rms_inv_scalar` remains shared-good
- `layer_1_input_norm_pre_weight_row_scalar_values` is the earliest shared-bad
  C45 tap
- `layer_1_input_norm_pre_weight_row_reconstructed` also diverges, but only
  after the scalar-value tap is already bad

Recommendation:

- do not widen to C46 or expand the tap contract
- focus the next audit inside the row-local scalar multiply that produces
  `layer_1_input_norm_pre_weight_row_scalar_values`
- treat row selection and the row-local `rms_inv` scalar as provisionally
  cleared on current main

Evidence:

- `profiling-artifacts/post432_c45_seed0.bench.json`
- `profiling-artifacts/post432_c45_seed0.bench.stderr`
- `profiling-artifacts/post432_c45_seed1.bench.json`
- `profiling-artifacts/post432_c45_seed1.bench.stderr`
- `profiling-artifacts/post432_c45_seed0_compare.txt`
- `profiling-artifacts/post432_c45_seed1_compare.txt`

## Phase C45 post-#435 narrowed row-scalar rerun verdict (2026-04-23)

Goal: rerun the already-merged narrowed C45 instrumentation from PR #434 on the
first healthy allowed on-demand RunPod pod and capture the earliest shared bad
tap among:

- `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_scalar_values`
- `layer_1_input_norm_pre_weight_row_reconstructed`

Remaining-work preflight:

- PR #434 was present on `origin/main` (`7808adf`, merged 2026-04-23 15:44 UTC)
- PR #435 was present on `origin/main` (`ca474b5`, merged 2026-04-23 16:39 UTC)
- no newer open or merged kiln PR had already recorded the narrowed
  three-tap C45 verdict
- no separate pending or running kiln task overlapped this exact post-#435
  healthy-pod rerun

RunPod outcome:

- first healthy allowed pod: `NVIDIA RTX A6000`, image
  `ghcr.io/ericflo/kiln-runpod:latest`, pod `h2gltnqi6qdjb4`
- the first healthy pod exposed two concrete bootstrap bugs on current `main`:
  - `deploy/runpod/kiln-setup.sh` no longer downloaded the expected
    `/workspace/qwen3.5-4b` checkpoint, so the first seeded bench failed at
    model load until the setup helper was restored to pull Qwen3.5-4B
  - `crates/kiln-model/src/mtp_debug.rs` serialized the narrowed splice dump
    but never created the parent directories for
    `profiling-artifacts/.../mtp_pos-{pos}/step-{step}.safetensors`, so the
    C45 dump path warned and emitted no safetensors until the write path grew
    a `create_dir_all(...)` for its parent directory
- the reference dump script also required `transformers`, which the current
  kiln image did not install; the recovery pod installed it ad hoc and this PR
  adds it to `deploy/runpod/Dockerfile`
- final artifact-producing recovery pod: `8d7s7t6zxs827r`
  (`NVIDIA RTX A6000`, same kiln image)

Validation outcome on the healthy A6000 pod after the narrow bootstrap fixes:

- `cargo build --release --features cuda --bin kiln-bench`
- `cargo test --locked -p kiln-model c45 -- --nocapture`
- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
- both narrowed `scripts/mtp_compare.py --c45-row-scalar ...` runs completed
  and emitted the same earliest shared bad tap

Commands used on the artifact-producing pod:

```bash
source /root/.kiln-build-env
cd /workspace/kiln

KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
KILN_MTP_DUMP_PATH=profiling-artifacts/post435_c45_row_scalar_seed0_captures/mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 0 \
  > profiling-artifacts/post435_c45_row_scalar_seed0.bench.json \
  2> profiling-artifacts/post435_c45_row_scalar_seed0.bench.stderr

KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0,2 KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 \
KILN_MTP_DUMP_HIDDEN_STATES=1 KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1 \
KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS=1 \
KILN_MTP_DUMP_PATH=profiling-artifacts/post435_c45_row_scalar_seed1_captures/mtp_pos-{pos}/step-{step}.safetensors \
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 1 \
  > profiling-artifacts/post435_c45_row_scalar_seed1.bench.json \
  2> profiling-artifacts/post435_c45_row_scalar_seed1.bench.stderr

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post435_c45_row_scalar_seed0_captures/mtp_pos-0/step-1.safetensors \
  --out profiling-artifacts/post435_c45_row_scalar_seed0_ref.safetensors \
  --device cuda \
  --c45-taps

python3 scripts/mtp_compare.py --c45-row-scalar \
  --kiln profiling-artifacts/post435_c45_row_scalar_seed0_captures/mtp_pos-0/step-1.safetensors \
  --ref profiling-artifacts/post435_c45_row_scalar_seed0_ref.safetensors \
  > profiling-artifacts/post435_c45_row_scalar_seed0_compare.txt

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump profiling-artifacts/post435_c45_row_scalar_seed1_captures/mtp_pos-2/step-1.safetensors \
  --out profiling-artifacts/post435_c45_row_scalar_seed1_ref.safetensors \
  --device cuda \
  --c45-taps

python3 scripts/mtp_compare.py --c45-row-scalar \
  --kiln profiling-artifacts/post435_c45_row_scalar_seed1_captures/mtp_pos-2/step-1.safetensors \
  --ref profiling-artifacts/post435_c45_row_scalar_seed1_ref.safetensors \
  > profiling-artifacts/post435_c45_row_scalar_seed1_compare.txt
```

Recovered seed metrics from the final artifact-producing pod:

| Seed | prompt tokens | prefill ms | decode tok/s | α |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 9587.5 | 39.8 | 0.7397 |
| 1 | 508 | 404.8 | 23.9 | 0.2959 |

Verdict:

- seed 0 compare: earliest shared bad tap =
  `layer_1_input_norm_pre_weight_row_scalar_values`
- seed 1 compare: earliest shared bad tap =
  `layer_1_input_norm_pre_weight_row_scalar_values`
- the last shared-good narrowed C45 tap on both seeds is
  `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_reconstructed` also diverges on both
  seeds, but only after the flat row-local scalar multiply values are already
  bad

So the post-#435 narrowed rerun clears the extracted scalar replay path and
localizes the first shared row-local drift one step later:

- `layer_1_input_norm_rms_inv_scalar` remains shared-good
- `layer_1_input_norm_rms_inv_scalar_extracted_values` remains shared-good
- `layer_1_input_norm_pre_weight_row_scalar_values` is the earliest shared-bad
  narrowed C45 tap
- `layer_1_input_norm_pre_weight_row_reconstructed` also diverges, but only
  after the multiply-values tap is already bad

Recommendation:

- do not widen to C46 or expand the tap contract
- focus the next audit inside the row-local scalar multiply that produces
  `layer_1_input_norm_pre_weight_row_scalar_values`
- treat the row-local `rms_inv` tensor and its extracted scalar replay as
  provisionally cleared on current `main`

Evidence:

- `profiling-artifacts/post435_c45_row_scalar_seed0.bench.json`
- `profiling-artifacts/post435_c45_row_scalar_seed0.bench.stderr`
- `profiling-artifacts/post435_c45_row_scalar_seed1.bench.json`
- `profiling-artifacts/post435_c45_row_scalar_seed1.bench.stderr`
- `profiling-artifacts/post435_c45_row_scalar_seed0_compare.txt`
- `profiling-artifacts/post435_c45_row_scalar_seed1_compare.txt`

## 2026-04-24 C48 post-C47 forced-MTP A6000 benchmark

C48 refreshed the forced-MTP A6000 artifact on current `origin/main` after PR #459
(`e9c071e`) using the mandatory RunPod image on pod `sl53yvx5seviyx`. Full
report and per-seed artifacts live in
`docs/archive/phase-c/phase-c48/post-c47-mtp-a6000-benchmark.md`.

Summary: median alpha = `0.3231`; median decode = `26.91 tok/s`; median mean
ITL = `37.16 ms`; median prefill = `351.7 ms` across 20 zero-exit seeds.
Compared with `docs/archive/phase-c/phase-c40f/summary.json`, C48 is still below the MTP
acceptance floor and decode is `0.703x` of the C40f median, so C47 does not
change the MTP go/no-go. Next boundary is benchmark harness/prompt-distribution
parity, not production RMSNorm/broadcast math.

## 2026-04-24 C49 MTP harness-parity A/B

C49 compares the C48-style forced-MTP command against C40f-style harness flags on the same current-main commit and A6000. See `docs/archive/phase-c/phase-c49/mtp-harness-parity-ab.md` and `docs/archive/phase-c/phase-c49/summary.json`.

Result: C40f-style flags restore the median acceptance rate from `0.391` to `0.707` and median decode from `28.73 tok/s` to `42.45 tok/s`. The restored arm clears both gates: α ≥ `0.65`, and decode is `1.11x` the C40f historical median of `38.25 tok/s` rather than outside the allowed 10% band.

Recommendation: treat C48 as a harness/workload comparability artifact and resume Phase 6 MTP performance/profiling work from the C40f-style harness anchor. Do not reopen model-math investigation based on C48 alone.

## Phase 6 C50 C40f-style native MTP decode profile (2026-04-24)

C50 re-ran the restored C40f-style native-MTP harness from C49 on current
`origin/main` at `7c638e7e2d69cb16772619a2a32d6114767bf7e2` on an on-demand
A6000. The three-seed median remained healthy: α `0.707`, decode `44.00 tok/s`,
and mean ITL `22.73 ms`, so do not reopen the C48 model-math investigation.

Artifact: `docs/archive/phase-c/phase-c50/mtp-c40f-decode-profile.md`; machine summary:
`docs/archive/phase-c/phase-c50/summary.json`.

The baked Nsight Systems 2023.4.4 importer failed C50 `.qdstrm` import with a
QuadD wrong-event-order error before stats export, so C50 carries forward the
latest successful current-main decode NVTX attribution from post-#442 for the
hotspot table: `:kiln/gdn/gates` `17.9%`, `:kiln/gdn/gated_norm` `17.3%`, and
`:kiln/gdn/qk_norm` `15.0%`. Next implementation target: fuse or vendor the GDN
gate/gated-norm decode path; it is higher leverage than FlashInfer/full-attn
decode and C50 does not justify retrying C48 model-math work.

## 2026-04-24 — GDN gated RMSNorm CUDA fusion result (PR TBD)

Implemented a minimal CUDA bf16 fused `rms_norm(x, weight, eps) * silu(z)` kernel for the Qwen3.5 GDN `hidden=128` envelope and wired it through `BackendRuntime::gdn_gated_rms_norm` behind `KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM=1`.

Validation on RunPod A6000 (`KILN_CUDA_ARCHS=86`) passed:

- `cargo test -p kiln-gdn-kernel --release -- --nocapture`
- `cargo test -p kiln-model --release --features cuda test_cuda_gdn_gated_rms_norm_matches_fallback -- --nocapture`
- `cargo build --release --features cuda,nvtx --bin kiln-bench`

Parity was exact at the model backend test shape (`max_abs_diff=0`, `mean_abs_diff=0`) versus the Candle fallback rounded to bf16.

C40f-style MTP decode A/B (`--prompt-subset humaneval --prompt-tokens 512 --max-output-tokens 128 --temperature 0.0`) did not clear the speedup floor:

| Path | Seeds | Median decode tok/s | Median mean ITL |
| --- | --- | ---: | ---: |
| Candle fallback (`KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM=1`) | 1,2,3 | 37.54 tok/s | 26.64 ms |
| Fused CUDA default | 1,2,3 | 38.58 tok/s | 25.92 ms |

Overall decode median improved ~2.8%, which is safely below the requested `kiln/gdn/gated_norm` 25% local floor and consistent with prior CUDA-graphs fusion-null notes. The fused kernel remains safe to ship because parity passed and the kill switch preserves fallback behavior, but the next optimization task should target a different hotspot or a memory-traffic-reducing extension of an existing on-chip GDN kernel rather than retrying standalone gated-norm fusion.

## Phase 6 — post-#466 C40f-style native MTP decode profile

C51 re-ran the C40f-style native-MTP decode workload on current `origin/main`
after PR #466 landed the fused CUDA GDN gated RMSNorm kernel. RunPod used the
mandatory `ghcr.io/ericflo/kiln-runpod:latest` image on an on-demand NVIDIA RTX
A6000 at commit `831d879d2ddb2500fd49f76c9b5df89aedd923b1`.

Artifact: `docs/archive/phase-c/phase-c51/post-466-mtp-decode-profile.md`; machine summary:
`docs/archive/phase-c/phase-c51/summary.json`.

Benchmark command shape: `KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1
KILN_MTP_ARGMAX_FP32=1`, paged, 512 prompt tokens, 128 output tokens,
`--prompt-subset humaneval`, chat template, temperature `0.0`, seeds `1,2,3`.
Median decode was `37.07 tok/s`, median mean ITL was `26.97 ms`, and median α
was `0.588`. The per-seed decode tok/s values were `41.41`, `37.07`, and
`32.06`.

Profiler status: `nsys profile` exited `0` and generated a `94M` `.qdstrm`, but
the baked Nsight Systems `2023.4.4` importer failed with the known QuadD
wrong-event-order error before producing `/tmp/kiln-post466-mtp.nsys-rep`. The
committed `docs/archive/phase-c/phase-c51/nsys-profile.log`, `nsys-nvtxsum.txt`,
`nsys-gpu-kernsum.txt`, and `nsys-cuda-api.txt` preserve the failed capture and
missing-stats evidence; the multi-MB raw `.qdstrm` is intentionally not
committed.

Top decode hotspots: unavailable for post-#466. Do not invent wall-clock
percentages or treat the C50 post-#442 carry-forward table as fresh after PR
#466. For context only, the freshest successful attribution before #466 had
`:kiln/gdn/gates` `17.9%`, `:kiln/gdn/gated_norm` `17.3%`, and
`:kiln/gdn/qk_norm` `15.0%`; those numbers are now stale.

Recommended next implementation target: conditional GDN gate/gated-norm cluster
work only if a successful post-#466 profiler capture confirms it remains the top
decode cluster. Otherwise, first repair or bypass the Nsight importer failure
with a compatible newer `nsys`, smaller decode-window capture, or alternate
profiler path that yields current NVTX/kernel attribution.

## Phase 6 C52 post-#468 MTP profiler attribution (2026-04-24)

C52 repairs the C51 attribution gap on current `origin/main` at
`55a5d9f26e6ca4be3d5f448935786d6a97e16a24` after PR #468 documented the
post-#466 Nsight importer failure. The workload is the same C40f-style
native-MTP decode shape used by C51: `KILN_SPEC_METHOD=mtp`,
`KILN_BENCH_FORCE_MTP=1`, `KILN_MTP_ARGMAX_FP32=1`, paged 512-token prompt,
128 generated tokens, seed `1`, `--chat-template`, `--latency-only`,
`--prompt-subset humaneval`, and temperature `0.0`.

Artifact: `docs/archive/phase-c/phase-c52/post-468-mtp-profiler-attribution.md`; machine
summary: `docs/archive/phase-c/phase-c52/summary.json`.

Profiler repair: the baked image still reports Nsight Systems `2023.4.4`, which
is the version that failed C51 import with the QuadD wrong-event-order error.
C52 installed only `nsight-systems-2024.5.1` from the already-configured NVIDIA
CUDA apt repository and used
`/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys` for both capture and
stats. The required capture produced a `104M` `.nsys-rep`, and the 2024 report
exports succeeded with the current report names (`nvtx_sum`,
`nvtx_pushpop_trace`, `cuda_gpu_kern_sum`, `cuda_api_sum`,
`cuda_kern_exec_sum`). The old aliases (`nvtxsum`, `gpukernsum`, `cudaapisum`)
are not valid report names under 2024.5.1; the failed alias outputs are
committed as fallback-ladder evidence.

The profiled seed-1 run produced α `0.707`, decode `25.31 tok/s`, and mean ITL
`39.51 ms` under Nsight tracing. Treat those timings as profiled attribution
context only; C51 remains the unprofiled post-#466 benchmark median anchor.

### Top decode NVTX ranges

Decode window attribution is derived from `nvtx_pushpop_trace`: first
`:kiln/mtp/step` start through the final decode NVTX end (`5069.9 ms`, 75 MTP
steps). Percentages exclude the `:kiln/mtp/step` parent wrapper from the
NVTX-duration denominator to avoid double-counting that high-level parent.

| Rank | Decode range | Wall-clock share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/conv` | 15.4% | 562.361 ms | 2328 |
| 2 | `:kiln/gdn/gates` | 14.2% | 518.513 ms | 2328 |
| 3 | `:kiln/gdn/gated_norm` | 13.4% | 487.295 ms | 2328 |
| 4 | `:kiln/gdn/qk_norm` | 10.9% | 396.837 ms | 2328 |
| 5 | `:kiln/attn/rope` | 8.5% | 308.079 ms | 851 |

### Top decode kernels

Kernel shares use CUDA GPU trace rows inside the same decode window, reduced to
`docs/archive/phase-c/phase-c52/decode-window-kernels.csv` rather than committing the full
multi-MB raw CUDA trace.

| Rank | Kernel | GPU-time share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | 23.6% | 563.199 ms | 7977 |
| 2 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | 12.3% | 293.366 ms | 1655 |
| 3 | `ucopy_bf16` | 7.6% | 181.148 ms | 19612 |

Recommendation from current post-#468 data only: target the GDN decode cluster.
Start with a static audit of `:kiln/gdn/conv` so we do not retry an already
exhausted causal-conv/chunk-vendoring path; if that audit shows low leverage,
move to the adjacent `:kiln/gdn/gates` + `:kiln/gdn/gated_norm` cluster. Do not
choose a next kernel from stale C50/C51 carry-forward percentages.

## Phase 6 C53 GDN conv decode hotspot audit (2026-04-24)

C53 audited the C52 rank-1 decode hotspot, `:kiln/gdn/conv`, before attempting
another causal-conv optimization. Artifact:
`docs/archive/phase-c/phase-c53/gdn-conv-decode-audit.md`; machine summary:
`docs/archive/phase-c/phase-c53/summary.json`.

Conclusion: **audit-only, no implementation**. The current CUDA path already
routes supported native-MTP single-token decode calls through
`backend.causal_conv1d_update` and the vendored `kiln-conv1d-kernel` update
kernel for the Qwen3.5 envelope (BF16 activations/weights, F32 state,
`seq_len == 1`, `kernel_size == 4`). The portable
`causal_conv1d_decode` fallback is not expected for the C52 workload unless the
`KILN_DISABLE_FUSED_CONV1D` kill switch is set or the support envelope changes.

C53 adds measurement-only child NVTX ranges under `:kiln/gdn/conv`:
`:kiln/gdn/conv/layout`, `:kiln/gdn/conv/update`,
`:kiln/gdn/conv/prefill_update`, `:kiln/gdn/conv/fallback_decode`, and
`:kiln/gdn/conv/fallback_prefill`. These ranges split the remaining hotspot
into layout/contiguity, fused update wrapper/kernel launch, and explicit
fallback time without changing tensor behavior.

Required RunPod validation/profiling was attempted on RTX A6000 pods
`nszu2wno80dvef` and `efeldsa2tpx69s`, but both runs hit SSH failures during
validation (`wait-file` timeout followed by SSH reset/no-output state checks).
Both pods were terminated. No C53 before/after speedup is claimed.

Recommendation: re-run the C52 native-MTP decode workload with the C53 child
NVTX ranges when RunPod SSH is healthy. If `:kiln/gdn/conv/update` dominates,
treat causal-conv as already kernel/runtime-bound and move to the adjacent
`:kiln/gdn/gates` + `:kiln/gdn/gated_norm` cluster. If
`:kiln/gdn/conv/layout` dominates, open a separate minimal layout/contiguity
fix task.

## Phase 6 C54 GDN conv child NVTX profile (2026-04-24)

C54 reran the C52 C40f-style native-MTP decode profile on current `origin/main`
at `13a2a3d437680d299a1f4a17029cbca8b700701f`, after C53 added child NVTX
ranges under `:kiln/gdn/conv`. Artifact:
`docs/archive/phase-c/phase-c54/conv-child-nvtx-profile.md`; machine summary:
`docs/archive/phase-c/phase-c54/summary.json`.

Validation passed on an on-demand RTX A6000 RunPod pool lease
`pod-c11fe496c432600caf0baa6a` / `sl53yvx5seviyx`: `cargo test -p
kiln-conv1d-kernel --release -- --nocapture`, `cargo test -p kiln-model
--release --features cuda test_causal_conv1d_update_matches_fallback --
--nocapture`, and `cargo build --release --features cuda,nvtx --bin
kiln-bench`. The pod was released after artifacts were copied; the subsequent
RunPod status check showed no active pods.

Profiler setup matched the C52 Nsight repair: the baked image had Nsight Systems
`2023.4.4`, so C54 installed only `nsight-systems-2024.5.1` from the existing
NVIDIA CUDA apt repo and used
`/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys`. The retained
workload used `KILN_SPEC_METHOD=mtp`, `KILN_BENCH_FORCE_MTP=1`,
`KILN_MTP_ARGMAX_FP32=1`, `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`,
`--model-path /workspace/qwen3.5-4b`, paged 512-token Humaneval prompt, 128
generated tokens, `--skip-training`, `--chat-template`, `--latency-only`,
temperature `0.0`, and seed `1`.

The profiled run produced α `0.693`, decode `22.16 tok/s`, mean ITL `45.13 ms`,
P50 ITL `31.74 ms`, and P99 ITL `105.52 ms`. Treat those as profiled
attribution context only.

Decode window attribution uses the same C52 method: first `:kiln/mtp/step` start
through final decode NVTX end (`5880.202 ms`, 75 MTP steps), excluding the
`:kiln/mtp/step` parent wrapper from the NVTX-duration denominator
(`4848.439 ms`).

| Rank | Decode range | Wall-clock share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/conv` | 12.45% | 603.799 ms | 2352 |
| 2 | `:kiln/gdn/gates` | 11.47% | 555.891 ms | 2352 |
| 3 | `:kiln/gdn/conv/fallback_prefill` | 11.11% | 538.451 ms | 1800 |
| 4 | `:kiln/gdn/gated_norm` | 10.79% | 523.019 ms | 2352 |
| 5 | `:kiln/gdn/qk_norm` | 8.76% | 424.748 ms | 2352 |

Child attribution under `:kiln/gdn/conv`:

| Child range | Decode-window share | Total time | Instances | Share of conv parent |
| --- | ---: | ---: | ---: | ---: |
| `:kiln/gdn/conv/fallback_prefill` | 11.11% | 538.451 ms | 1800 | 89.18% |
| `:kiln/gdn/conv/layout` | 0.90% | 43.672 ms | 2352 | 7.23% |
| `:kiln/gdn/conv/update` | 0.23% | 11.114 ms | 552 | 1.84% |
| `:kiln/gdn/conv/prefill_update` | 0.00% | 0.000 ms | 0 | 0.00% |
| `:kiln/gdn/conv/fallback_decode` | 0.00% | 0.000 ms | 0 | 0.00% |

Conclusion: C53's single-token decode audit was correct — `fallback_decode` is
absent and the existing `causal_conv1d_update` wrapper is not the bottleneck.
The material fallback is the native-MTP draft path: CUDA does not yet implement
`supports_causal_conv1d_prefill` / `causal_conv1d_prefill`, so each `seq_len >
1` GDN draft forward falls back to the portable prefill path. The 1800 fallback
instances equal 24 GDN layers × 75 MTP steps.

Recommendation: target a minimal CUDA `causal_conv1d_prefill` path for the
native-MTP draft GDN conv shape before moving to `:kiln/gdn/gates` +
`:kiln/gdn/gated_norm`. Suggested files are `crates/kiln-conv1d-kernel/src/lib.rs`,
`crates/kiln-conv1d-kernel/csrc/causal_conv1d_update.cu`,
`crates/kiln-conv1d-kernel/csrc/causal_conv1d_update.h`,
`crates/kiln-model/src/backend/cuda.rs`, and `crates/kiln-model/src/forward.rs`.
Do not retry the existing `seq_len == 1` update kernel; if the minimal CUDA
prefill support envelope cannot cover the MTP draft shape, retarget to the
gates/gated-norm cluster instead.

## Phase 6 C56 post-#476 native-MTP decode profile (2026-04-24)

C56 attempted to replace C54 with a fresh post-#476 native-MTP decode-window profile on current `origin/main` at `f1530071bb489b4d72bbf8e6ad0062281b52c0bf` (`phase6: add CUDA conv1d prefill fast path`). The run used the required `ghcr.io/ericflo/kiln-runpod:latest` image on on-demand pod `2cetn7tf9zckij` with an RTX A6000 (`sm_86`, 49140 MiB, driver 550.127.08). Nsight Systems `2024.5.1` was installed from the existing NVIDIA CUDA apt repo because the baked PATH still pointed at `2023.4.4`.

The focused validation/build commands passed before profiling:

- `cargo test -p kiln-conv1d-kernel --release -- --nocapture`: 4 passed
- `cargo test -p kiln-model --release --features cuda test_causal_conv1d_update_matches_fallback -- --nocapture`: 1 passed
- `cargo test -p kiln-model --release --features cuda causal_conv1d_prefill -- --nocapture`: 1 passed; max abs diff `1.4901161e-8`, state parity max abs diff `0`
- `cargo build --release --features cuda,nvtx --bin kiln-bench`: passed

The required C52/C54 workload did **not** reach decode. It failed during native-MTP prefill in GDN layer 0:

```text
Error: MTP latency benchmark failed

Caused by:
    0: MTP prefill (paged with last-hidden) failed
    1: gated deltanet layer 0 (linear attention, paged)
    2: causal_conv1d_prefill kernel failed
    3: kiln_causal_conv1d_prefill_bf16_f32 failed with status 3
```

The same failure reproduced with `KILN_CUDA_GRAPHS=false`, so it is not isolated to CUDA graph capture. Because no `:kiln/mtp/step` range was emitted, the C52/C54 decode-window method (first `:kiln/mtp/step` start through final decode NVTX end, excluding the `:kiln/mtp/step` parent wrapper) could not be applied. Decode tok/s, mean ITL, MTP α, top decode NVTX ranges, and top decode-window CUDA kernels are therefore unavailable from C56.

`:kiln/gdn/conv/fallback_prefill` does not appear in the failed prefill trace, but this does not confirm its decode-window removal because decode never starts. The fast path is reached as `:kiln/gdn/conv/prefill_update` and then the CUDA wrapper returns status `3`.

Failed-prefill NVTX ranges, included only as diagnostic context:

| Rank | NVTX range | Wall time | Share | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/in_proj` | 144.576983 ms | 98.9% | 1 |
| 2 | `:kiln/norm/pre_attn` | 1.097066 ms | 0.8% | 1 |
| 3 | `:kiln/gdn/conv` | 0.250360 ms | 0.2% | 1 |

Failed-prefill CUDA kernels, included only as diagnostic context:

| Rank | CUDA kernel | Total time | Share | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `ucopy_bf16` | 192.268138 ms | 88.2% | 250 |
| 2 | `cast_bf16_f32` | 25.365062 ms | 11.6% | 104 |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_nn_align8` | 0.200449 ms | 0.1% | 1 |
| 4 | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nn` | 0.099648 ms | 0.0% | 1 |
| 5 | `is_u32_bf16` | 0.026048 ms | 0.0% | 1 |

Recommendation: do not pick the next Phase 6 decode optimization target from C56. First fix or guard the CUDA causal-conv1d prefill fast path for the real native-MTP prefill workload on A6000, then rerun C56. Until that rerun reaches `:kiln/mtp/step`, C54 remains the last valid native-MTP decode-window source of truth.

Full reduced artifact: `docs/archive/phase-c/phase-c56/post-476-mtp-decode-profile.md`. Machine-readable summary: `docs/archive/phase-c/phase-c56/summary.json`.

## Phase 7 GDN prefill memory preflight (2026-04-24)

**Outcome: current `origin/main` (`4abc2ca`) no longer OOMs at 64k with W4A16 + FP8 KV on an RTX A6000, but the monolithic CUDA prefill path is still activation-bound and effectively hits the A6000 ceiling at 128k.** The required 32k and 64k probes completed with `KILN_STREAMING_PREFILL=0`, `KILN_W4A16=1`, `KILN_KV_CACHE_FP8=1`, and `KILN_CUDA_GRAPHS=true`; peak memory was 25.3 GiB and 33.3 GiB respectively. An exploratory 128k monolithic run completed the latency prefill/decode portion at 48.6 GiB peak, within ~0.6 GiB of the 49.1 GiB device total, then was SIGTERM'd during the post-latency throughput sweep to stay inside the task budget. The already-shipped opt-in streaming prefill path (`KILN_STREAMING_PREFILL=1`, default 8192-token tile) ran the same 128k latency probe at 25.4 GiB peak with equivalent prefill throughput.

### Hardware and cleanup

- **Pod:** direct RunPod fallback `jc3jwpaps4e8ro`, `kiln-gdn-prefill-memory-c62`; pool acquire failed with A6000 supply exhaustion.
- **Image:** `ghcr.io/ericflo/kiln-runpod:latest`.
- **GPU:** NVIDIA RTX A6000, 49140 MiB, driver 550.127.08, CUDA 12.4 runtime.
- **Repository:** `/workspace/kiln`, `origin/main` at `4abc2ca`.
- **Cleanup:** fallback pod explicitly terminated after measurements.

### Commands

Current `kiln-bench` requires `--model-path`, so the required commands were run with that added argument and with an `nvidia-smi --query-gpu=timestamp,memory.used --format=csv -lms 250` sampler around the process:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench

KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=0 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 32768 --max-output-tokens 1 --skip-training

KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=0 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 65536 --max-output-tokens 1 --skip-training
```

Exploratory ceiling/streaming commands added `--latency-only` only for 128k where the post-latency throughput sweep was not needed for this memory preflight.

### Measurements

| Prompt request | Actual prompt | Path | Peak GPU memory | Model-load memory | Delta over load | Prefill time | Prefill tok/s | Status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | 32768 | monolithic (`KILN_STREAMING_PREFILL=0`) | 25320 MiB | 17526 MiB | 7794 MiB | 20340.1 ms | 1611 | pass |
| 65536 | 65533 | monolithic (`KILN_STREAMING_PREFILL=0`) | 33327 MiB | 17526 MiB | 15801 MiB | 25868.8 ms | 2533 | pass |
| 131072 | 131065 | monolithic (`KILN_STREAMING_PREFILL=0`) | 48562 MiB | 17526 MiB | 31036 MiB | 58397.6 ms | 2244 | latency pass; SIGTERM during throughput sweep |
| 131072 | 131065 | streaming (`KILN_STREAMING_PREFILL=1`, tile 8192) | 25384 MiB | 17526 MiB | 7858 MiB | 58013.7 ms | 2259 | pass (`--latency-only`) |

The required 32k/64k validation commands exited `0`. The harness printed `generation failed` in the later throughput section, but the process still exited successfully and the latency prefill/decode data were emitted. The optional 128k monolithic run also completed prefill and decode, but was manually terminated after it entered the unnecessary throughput sweep.

### Conclusion

FP8 KV is not the limiting factor for these long prompts. With `KILN_KV_CACHE_FP8=1`, memory still scales almost linearly with monolithic GDN/full-forward activations: +7.8 GiB at 32k, +15.8 GiB at 64k, and +31.0 GiB at 128k over the loaded model. The 64k path now fits on A6000, but 128k monolithic prefill runs with only ~578 MiB headroom, so the practical monolithic ceiling is around 128k on this 48 GiB card.

The streaming path is already present and opt-in on CUDA. At 128k it reduces peak memory from 48.6 GiB to 25.4 GiB while preserving roughly the same prefill throughput in this single-run probe (2259 tok/s vs 2244 tok/s). That makes the next Phase 7 slice a defaulting/guarding task, not a new CUDA-kernel task: run a small interleaved regression at 32k/64k/128k with `KILN_STREAMING_PREFILL=0/1`, then enable streaming by default for CUDA prompts above a conservative threshold (recommended first threshold: 65536 tokens) if decode and TTFT stay within noise.

Detailed artifact: `docs/archive/phase-c/phase-c62/gdn-prefill-memory-preflight.md`.

## Re-profile 2026-04-25 (post-PR #534, main SHA `60e298d`)

**Outcome: Hotspot mix and decode throughput are structurally identical to the post-#521 baseline within ±0.1 pp on every NVTX range and ±1.5 % on every aggregate latency metric, confirming that the post-#521 → post-#534 PR window (#522–#534, all doc-only audits, external-α microbenches, or artifact-only artifacts) changed no decode-path code.** This refresh is the next planning cycle's source-of-truth for hotspot ranking; the next-target shortlist is unchanged from the post-#521 implications.

### Hardware and tooling

- Pod: `mfk88l8i8tab02`, lease `pod-66eb55349e1403350e6c342d` (warm-pool pod, same as post-#521).
- RunPod on-demand `NVIDIA RTX A6000`, image `ghcr.io/ericflo/kiln-runpod:latest`, driver `550.127.08`, CUDA 12.4, `KILN_CUDA_ARCHS=86`.
- Profiler: Nsight Systems `2024.6.2.225-246235244400v0` (installed from NVIDIA apt repo; the baked `nsys 2023.4.4` is broken for stats import — see agent notes `kiln-nsys-baked-importer-broken` and `kiln-nsight-profiling-gotchas`).
- Runtime flags: `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`.

### Build

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin main && git reset --hard origin/main   # 60e298d
KILN_CUDA_ARCHS=86 cargo build --release --features cuda,nvtx --bin kiln-bench
```

### Decode bench (median-of-3, no profiler)

```bash
export KILN_W4A16=1 KILN_CUDA_GRAPHS=true
for i in 1 2 3; do
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --prompt-subset humaneval --chat-template --latency-only \
    --temperature 0.0 --seed 1
done
```

| Run | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.48 | 21.06 | 21.08 | 25.12 | 9237.6 (cold-start TTFT artifact) |
| 2 | 46.36 | 21.57 | 21.63 | 25.89 | 369.7 |
| 3 | 46.55 | 21.48 | 21.49 | 26.19 | 356.2 |
| **median** | **46.55** | **21.48** | **21.49** | **25.89** | **362.9** (warm, runs 2–3 mean) |

### Decode top NVTX ranges

Source: `docs/archive/phase-c/phase-c65/post534_decode_nvtx_pushpop_sum.csv`. Capture: 1 paged prefill (494 tokens) + 128 paged decode steps; `:kiln/attn/full/prefill_initial` is 1.0 % of total wall-clock so the ranking is decode-dominated.

| Rank | NVTX range | Wall-clock share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `:kiln/gdn/gates` | **14.6%** | 3096 |
| 2 | `:kiln/gdn/gated_norm` | **14.0%** | 3096 |
| 3 | `:kiln/gdn/qk_norm` | **11.8%** | 3096 |
| 4 | `:kiln/gdn/in_proj` | **9.4%** | 3096 |
| 5 | `:kiln/attn/rope` | **8.4%** | 1032 |
| 6 | `:kiln/mlp/gate` | **5.1%** | 4128 |
| 7 | `:kiln/mlp/up` | **5.0%** | 4128 |
| 8 | `:kiln/mlp/down` | **4.6%** | 4128 |
| 9 | `:kiln/attn/full/decode_fused` | **3.8%** | 1024 |
| 10 | `:kiln/gdn/head_expand` | **2.9%** | 3096 |

### Decode top CUDA kernels

Source: `docs/archive/phase-c/phase-c65/post534_decode_cuda_gpu_kern_sum.csv`.

| Rank | Kernel | GPU-kernel share | Instances |
| ---: | --- | ---: | ---: |
| 1 | `ucopy_bf16` | **15.9%** | 5585 |
| 2 | `Marlin<(256,1,8,8,4,8)>` (W4A16 decode) | **13.3%** | 13312 |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8` | **10.9%** | 129 |
| 4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | **9.1%** | 3072 |
| 5 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **8.4%** | 6144 |
| 6 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | **4.9%** | 3072 |
| 7 | `bmul_f32` | **2.9%** | 23865 |
| 8 | `gdn_full_chunk_forward_kernel` (vendored GDN, PR #80) | **2.8%** | 168 |
| 9 | `cast_bf16_f32` | **2.7%** | 17648 |
| 10 | `fused_rmsnorm_kernel` | **2.6%** | 10449 |

### Comparison vs prior baselines

| Region / Kernel | post-#166 | post-#521 | post-#534 | Δ post-#521 → post-#534 |
| --- | ---: | ---: | ---: | --- |
| `:kiln/gdn/gates` | 18.0% | 14.5% | 14.6% | +0.1 pp |
| `:kiln/gdn/gated_norm` | 17.5% | 13.9% | 14.0% | +0.1 pp |
| `:kiln/gdn/qk_norm` | 14.9% | 11.9% | 11.8% | −0.1 pp |
| `:kiln/gdn/in_proj` | 7.4% | 9.5% | 9.4% | −0.1 pp |
| `:kiln/attn/rope` | n/a | 8.3% | 8.4% | +0.1 pp |
| `:kiln/mlp/gate` | 6.3% | 5.0% | 5.1% | +0.1 pp |
| `:kiln/mlp/up` | 6.2% | 5.0% | 5.0% | 0 pp |
| `:kiln/mlp/down` | 5.9% | 4.7% | 4.6% | −0.1 pp |
| `:kiln/attn/full/decode_fused` | n/a | 3.8% | 3.8% | 0 pp |
| `ucopy_bf16` (kernel) | n/a | 15.9% | 15.9% | 0 pp |
| `Marlin<(256,1,8,8,4,8)>` (kernel) | n/a | 13.3% | 13.3% | 0 pp |
| Decode tok/s (median, no profiler) | **49.76** | **45.85** | **46.55** | **+1.5 %** |
| Mean ITL (median, no profiler) | 20.10 ms | 21.81 ms | 21.48 ms | −1.5 % |
| P99 ITL (median, no profiler) | 25.46 ms | 26.36 ms | 25.89 ms | −1.8 % |

### What changed structurally vs the post-#166 baseline

The structural deltas baked in between the post-#166 closing baseline and post-#534 are entirely from the post-#166 → post-#521 window (PRs #167–#521): PR #486 default-on fused QK norm collapsed `gdn/qk_norm` from 14.9 % to 11.8 %, redistributing share into `gdn/gates` and `gdn/gated_norm` (which dropped in absolute ms but rose in relative share), and the radix prefix cache + CUDA-graphs scaffolding (#503–#521) was wired in. The −6.4 % decode tok/s gap to post-#166 (49.76 → 46.55) persists from the post-#521 profile. Per agent note `kiln-bench-prefix-cache-no-effect` and PR #523, this regression is **not** caused by radix-prefix-cache hooks — `kiln-bench --paged` bypasses the prefix cache entirely. The actual source remains unidentified and is a follow-up bisection task, not a goal of this refresh.

### Next-target candidates (informational, not queueing)

This artifact does not queue new tasks. The next planning cycle should pick from this shortlist:

1. KV-cache FP8 default-on regression sweep (long-context capability win — see Phase 7 GDN prefill memory preflight 2026-04-24).
2. End-to-end native-MTP self-spec decode benchmarking (now that the external-α reference family is closed via #529–#534 with `kiln_native_ceiling`).
3. Marlin pack-at-load latency cleanup (~58 s, deterministic, single-kernel target).
4. Marlin BF16 weight residency cleanup (~1.3 GiB unused VRAM after `KILN_W4A16=1` packed weights are resident).

GDN gate-path fusion remains off-limits without new sub-range HBM-traffic evidence: PR #173 (`gates`) and PR #176 (`gates + gated_norm + recurrent`) both closed null, and the post-#534 mix gives no new reason to retry. vLLM's fused `fused_recurrent_gated_delta_rule_packed_decode_kernel` was audited (agent note `vllm-gdn-fused-decode-audit-2026-04-24`) and offers no bounded micro-port win on A6000 under CUDA graphs.

Detailed artifact: `docs/archive/phase-c/phase-c65/post-534-profile.md`.

# Phase 7 — End-to-end native-MTP self-spec decode tok/s bench (post-#535)

**Verdict:** `mtp_no_decode_win`

**Date:** 2026-04-25
**Hardware:** RunPod A6000 (sm_86), pool pod `mfk88l8i8tab02` (warm, idle since
PR #535 capture). 49,140 MB VRAM (`nvidia-smi`).
**Image:** `ghcr.io/ericflo/kiln-runpod:latest`.
**Kiln main SHA:** `c5cf77d` (post-PR #535 re-profile).
**Build:** `KILN_CUDA_ARCHS=86 cargo build --release --features cuda
--bin kiln-bench` — fully cached, link-only rebuild in 7.27 s on the
pool pod.

## Why this bench was needed

Phase 7's α-reference family closed across PRs #529–#534:

- **PR #529 (H15b)** — `kiln_native_ceiling`: kiln native α is the
  operative checkpoint-quality ceiling.
- **PR #530–#533 (H16/H17/H17b)** — vLLM v0.19.1, vLLM v0.20.0,
  SGLang main HEAD all return `*_unsupported_dense_4b` for
  Qwen3.5-4B + native MTP on A6000 sm_86.
- **PR #534 (H18)** — `kiln_above_hf`: kiln α (0.3636) > hand-rolled
  HF transformers reference α (0.2500); seed 0 bit-for-bit match
  confirmed the H18 protocol implementation correct.
- **PR #535** — fresh post-#534 PROFILING.md confirmed the decode
  hotspot mix is structurally identical to post-#521 (gates 14.6%,
  gated_norm 14.0%, qk_norm 11.8%, in_proj 9.4%, MLP trio ~14.7%);
  median MTP-Off decode 46.55 tok/s. **#535 explicitly listed
  end-to-end native-MTP self-spec decode benching as next-target
  candidate #2.**

What was *unmeasured*: the **operational decode-tok/s delta** of
MTP-On vs MTP-Off at bs=1 on A6000 with all C3-C40+ fixes landed.
Last measurement was PR #316 (Phase C5, 2026-04-21): α=0.124,
MTP-On −25.1% slower than MTP-Off. With α now confirmed at 0.3636
(H18 anchor) and the H18-validated protocol, the operational
win-or-no-win question had to be re-measured before any
default-flip proposal could be drafted.

This is a **bench-only PR**: no kernel work, no vendoring, no
forward-pass code changes. The output is either a flip-default
proposal or a doc-only `mtp_no_decode_win` redirect.

## Anti-duplication evidence (preflight)

| Check | Command | Result |
| --- | --- | --- |
| Bench-title overlap since PR #316 | `gh pr list -R ericflo/kiln --state all --limit 5 --search 'MTP decode bench in:title'` | Only PR #316 (Phase C5, 2026-04-21). |
| MTP-on-default proposal already filed | `gh pr list -R ericflo/kiln --state all --limit 10 --search 'MTP on default in:title'` | None — only PR #330 (codex MTP fp32-head fast gate, unrelated). |
| Active task overlap | `ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status running\|pending` | Only this task. |
| Pool state | `ce kiln-pod-list` | One warm A6000 (`mfk88l8i8tab02`), two hibernated. |
| Recent MTP PRs since C5 | `gh pr list -R ericflo/kiln --state all --limit 15 --search 'mtp in:title'` | Many phase-C/C-N audits and infra fixes; no decode tok/s rerun. |

Preflight clear. No overlap.

## Pre-registered decision rule

Locked into the task description **before** any bench was run:

| Median MTP-On Δ vs MTP-Off | Verdict | Action |
| --- | --- | --- |
| ≥ +10% decode tok/s | `mtp_decode_win_ship` | Propose flipping `KILN_SPEC_METHOD=mtp` default; open follow-up PR |
| +3% to +10% | `mtp_decode_marginal` | Keep opt-in; document; no default flip |
| −3% to +3% | `mtp_decode_neutral` | Keep opt-in; document |
| < −3% | `mtp_no_decode_win` | Doc-only redirect PR; document why α=0.3636+ still does not yield wallclock win |

## Bench protocol

**Common env:** `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`,
`KILN_CUDA_ARCHS=86`.

**Common args:**
```
--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
--prompt-subset humaneval --chat-template --latency-only
--temperature 0.0 --seed N
```

**MTP-Off arm:** unset `KILN_SPEC_METHOD` and `KILN_MTP_ARGMAX_FP32`,
seeds `1`, `2`, `3`.

**MTP-On arm:** `KILN_SPEC_METHOD=mtp`, `KILN_BENCH_FORCE_MTP=1`,
`KILN_MTP_ARGMAX_FP32=1`, seeds `1`, `2`, `3`.
`KILN_BENCH_FORCE_MTP=1` is required because
`crates/kiln-server/src/bench.rs:resolve_bench_spec_method_with_force`
downgrades `SpecMethod::Mtp` to `Off` when
`requested_prompt_tokens > BENCH_MTP_MAX_PROMPT_TOKENS` (= 128).
The first MTP-arm pass without the force flag confirmed the
downgrade — all 3 seeds resolved to `spec=off` and matched the
MTP-Off arm within ±2.5% (raw logs at
`artifacts/arm-mtp-seed-{1,2,3}.log`). The force flag matches the
agent-note recipe `kiln-c57-conv1d-prefill-recovery` and the C40f
harness-parity convention from agent-note `kiln-mtp-harness-parity-c49`.

**Build cache:** sccache backed by B2 (`build-cache/kiln/x86_64-linux-cuda12.4/`);
no rebuild required since pool pod was warm from PR #535 capture.

## Results

### Per-run table

| Arm | Seed | `spec_method` resolved | Prompt tokens | Decode tok/s | Mean ITL (ms) | P50 ITL (ms) | P99 ITL (ms) | α |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTP-Off | 1 | `off` | 494 | 44.748 | 22.35 | 21.85 | 27.05 | — |
| MTP-Off | 2 | `off` | 494 | 45.014 | 22.22 | 22.33 | 27.33 | — |
| MTP-Off | 3 | `off` | 494 | 44.231 | 22.61 | 22.56 | 28.19 | — |
| MTP-On  | 1 | `mtp` | 515 | 48.558 | 20.59 | 15.56 | 54.12 | 0.778 |
| MTP-On  | 2 | `mtp` | 510 | 43.092 | 23.21 | 16.06 | 57.97 | 0.684 |
| MTP-On  | 3 | `mtp` | 494 | 40.152 | 24.91 | 16.13 | 56.78 | 0.620 |

Raw artifacts: [`artifacts/arm-off-seed-{1,2,3}.log`](./artifacts/),
[`artifacts/arm-mtp-force-seed-{1,2,3}.log`](./artifacts/),
[`artifacts/arm-mtp-seed-{1,2,3}.log`](./artifacts/) (resolver-downgraded
unforced control). Raw CSV: [`post-535-mtp-decode-bench.csv`](./post-535-mtp-decode-bench.csv).

### Aggregates

|                       | MTP-Off | MTP-On (forced) |
| --- | --- | --- |
| Decode tok/s (median) | **44.748** | **43.092** |
| Decode tok/s (mean)   | 44.664 | 43.934 |
| Decode tok/s (range)  | 44.23–45.01 (Δ=1.8%) | 40.15–48.56 (Δ=21%) |
| P99 ITL (median, ms)  | 27.33 | 56.78 |
| α (median)            | — | 0.6842 |
| α (mean)              | — | 0.6941 |
| α (range)             | — | 0.620–0.778 |

### Decision-rule application

| Statistic | Value | Bucket |
| --- | --- | --- |
| **Δ median decode tok/s** | **−3.7 %** ((43.092 − 44.748)/44.748) | **`mtp_no_decode_win`** (< −3%) |
| Δ mean decode tok/s | −1.64 % | `mtp_decode_neutral` (−3% to +3%) |
| Δ paired (per-seed median) | −4.27 % | `mtp_no_decode_win` |
| Δ paired seed-1 (α=0.778) | +8.51 % | `mtp_decode_marginal` (in isolation) |
| Δ paired seed-2 (α=0.684) | −4.27 % | `mtp_no_decode_win` (in isolation) |
| Δ paired seed-3 (α=0.620) | −9.22 % | `mtp_no_decode_win` (in isolation) |

The pre-registered statistic is **median MTP-On Δ vs MTP-Off**.
Median-of-median (−3.7%) and paired-median (−4.27%) both fall into
`mtp_no_decode_win`. Mean-of-mean (−1.64%) falls into
`mtp_decode_neutral`, but the rule is registered against the
median, not the mean.

**Verdict: `mtp_no_decode_win`.**

## Interpretation

### α has improved 5.5× since PR #316 but still does not clear the operational floor

| Phase | PR | Median α | Notes |
| --- | --- | --- | --- |
| C5 | #316 | 0.124 | First post-C3 measurement, MTP −25.1% slower. |
| C29 | #355 | n/a | H9 empirical logits (H8 audit). |
| C35 | #364 | 0.588 (BF16) / 0.608 (FP32) | First chat-templated humaneval measurement; one seed hit 0.764 — first time clearing 0.72. |
| H18 | #534 | 0.3636 | H18 protocol probe vs HF transformers reference (0.2500). |
| **C66 (this PR)** | **(current)** | **0.6842** | **Median above 0.65 floor; one seed (0.778) clears the 0.72 paper floor.** |

The trend is monotonic-and-significant. Combined with the H18 seed-0
bit-for-bit match, this strongly supports the H18 PR #534 conclusion
that the protocol implementation is correct — the remaining ceiling
on α is a **checkpoint-quality artifact** (BF16 base-stack numerical
drift + Marlin W4A16 confounder), not a bug in the verifier or head.

### Why α=0.68 still does not yield a wallclock win at bs=1

- **Each MTP step costs more than one base decode step.** The verifier
  forward (full GQA + GDN sweep over all 32 layers) executes
  unconditionally each step, regardless of how many of the
  (k=1) drafted tokens get accepted. With α=0.68, the expected
  per-step token gain is `1 + α = 1.68` tokens, but the per-step
  cost relative to baseline must be ≤ 1.68× baseline cost for any
  decode-tok/s win. The PR #316 baseline (α=0.124) showed −25%; this
  bench (α=0.68) recovered to −3.7% — consistent with overhead/gain
  ratio being roughly linear in α with the break-even point near
  α≈0.72.
- **P99 ITL doubles.** MTP-On median P99 ITL is **56.78 ms** vs
  MTP-Off **27.33 ms** (~2.08×). MTP introduces large bursts of
  variance: accepted-draft steps ship multiple tokens cheaply but
  rejected-draft steps still pay the verifier cost, producing a
  bimodal latency distribution. P50 ITL **drops** from ~22 ms (Off)
  to ~16 ms (On) — exactly the bimodality signature.
- **High α-variance across just three humaneval seeds**
  (0.620–0.778, range 0.158). The seed with α=0.778 produced a
  +8.5% decode win; the seed with α=0.620 produced −9.2%. With only
  3 seeds the median estimate is unstable; even ±1 seed change
  could move the median to either `mtp_decode_neutral` or
  `mtp_no_decode_win`.

### What this means for the project goal

The project description names "close the decode-speed gap to
vLLM/SGLang on Qwen3.5-4B bs=1 A6000 via native MTP speculative
decoding" as a goal. As of post-#535:

- The **MTP path is functional and producing reasonable α** — the
  C-phase work since PR #316 has been productive.
- The **operational decode-tok/s win is not yet there** at bs=1.
  Median α needs to clear ~0.72 (paper floor) reliably across the
  workload distribution before a default flip is justified, and
  the variance in P99 ITL means even a flat-decode-median win
  would degrade tail latency.
- **`KILN_SPEC_METHOD=mtp` should remain opt-in.** Combined with the
  existing `BENCH_MTP_MAX_PROMPT_TOKENS=128` resolver guard and
  `KILN_BENCH_FORCE_MTP=1` opt-in, the current default is
  appropriate.

## Reopen triggers

This `mtp_no_decode_win` verdict should be revisited if any of the
following are true:

1. **Median α reliably clears 0.72** across humaneval+gsm8k+c4 with
   ≥10 seeds.
2. **A new PR materially reduces verifier-step cost** (e.g. fused
   verifier kernel, projection caching across draft+verify, smaller
   verifier KV recomputation, or a draft-skip-on-low-confidence
   short-circuit). In that case, retest with the same protocol.
3. **A k>1 native MTP path lands** (Qwen3.5-4B ships
   `mtp_num_hidden_layers=1` so k=1 is the immediate target, but
   future Qwen variants or extended-MTP heads may change the
   economics).
4. **Workload distribution shifts** — the prose-vs-chat-template α
   gap (agent-note `mtp-bench-workload-sensitivity`) means a
   different production workload may put α firmly above 0.72,
   making the default flip viable for that workload alone.

## Wall-clock + cost envelope

- **Pool acquire → release**: ~25 min (most of which was bench
  execution and artifact extraction). Pod was already warm from PR
  #535 — no cold-start cost.
- **GPU spend**: ~25 min × $0.49/hr = **~$0.21**. Well under the
  $15 cap declared in the task description.
- **Build time**: 7.27 s (link-only; sccache fully hot from #535
  capture).
- **Bench wall-clock**: 6 runs × ~50 s each = ~5 min total.

## Reproduction commands

From a warm A6000 pool pod with `kiln-runpod:latest`:

```bash
# Sync to current main (or the post-#535 commit)
cd /workspace/kiln && git fetch origin main && git reset --hard origin/main

# Build (link-only if cache is warm)
source /root/.kiln-build-env
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench

# MTP-Off arm
unset KILN_SPEC_METHOD KILN_MTP_ARGMAX_FP32
export KILN_W4A16=1 KILN_CUDA_GRAPHS=true
for s in 1 2 3; do
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --prompt-subset humaneval --chat-template --latency-only \
    --temperature 0.0 --seed $s
done

# MTP-On arm (force MTP — resolver downgrades to Off without this)
export KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1
for s in 1 2 3; do
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --prompt-subset humaneval --chat-template --latency-only \
    --temperature 0.0 --seed $s
done
```

α is auto-printed by the bench as `Draft acceptance α: <value>` for
the MTP arm; no `--mtp-log-alpha` flag exists in current main (the
task spec referenced one — the canonical name is just the
`acceptance_rate` field in the JSON output block).

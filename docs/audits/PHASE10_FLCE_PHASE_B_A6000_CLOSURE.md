# Phase 10 §1.5 — FLCE Phase B A6000 closure (deferred)

Date: 2026-04-29
Status: **DEFERRED — A6000 capacity unavailable.** PR #647 landed FLCE
Phase B with A40 closure validation (capacity-forced fallback from
A6000 `SUPPLY_CONSTRAINT`); the A40 numbers are sufficient evidence
that Phase B unblocks T=8192 RMSNorm-on SFT on Ampere-class hardware
under the unfused RMSNorm path. This audit attempted the production
A6000 closure cell — fused RMSNorm CustomOp2 active (PR #644 VRAM gate
threshold 48,128 MiB ≤ A6000's 49,140 MiB) AND FLCE Phase B's
autograd-graph break — but RunPod A6000 supply remained exhausted
across both pool hosts for the duration of the task budget. The
closure remains gated on capacity; this audit records the attempt
with a stable next-cycle reproducer so the planning loop has an
actionable handoff.

Hardware target (not reached): NVIDIA RTX A6000 (49,140 MiB total VRAM,
sm_86, driver 550.127.08), CUDA 12.4.
Pool state at task start: both A6000 entries hibernated and ready to
wake — `pod-cb7a166f8cd2d98f2f7466db`/`18x6how2blkqon` and
`pod-dfee85941e0b88c8e6a702f8`/`1qnfctm4m9qwoe`.
Branch: `ce/phase10-flce-phase-b-a6000-closure`
Commit at task start: `b82dfde` (PR #647 — FLCE Phase B + A40 closure).

## TL;DR

* **Pool acquire failed for ~12 minutes of contiguous retries.** Both
  hibernated A6000s in the pool returned RunPod GraphQL
  `SUPPLY_CONSTRAINT` on resume — "There are not enough free GPUs on
  the host machine to start this pod." The `runpod_api.py launch`
  fallback was not exercised because the kiln skill requires using
  the pool first, the task description explicitly forbids A40
  fallback (already validated in PR #647), and 12 min of consecutive
  503s indicate a sustained capacity shortage rather than a transient
  resume race.
* **Per-task instruction, this is the closure result.** Task
  description, step 1: *"If the pool returns no A6000 (capacity
  exhausted), STOP — do NOT fall back to A40 again. Report 'A6000
  capacity unavailable' as the closure result."* No GPU work was
  performed. No pod was acquired. No lease was held. Total cumulative
  GPU-side spend on this task: $0.
* **The A40 numbers in PR #647 stand as the closure evidence for FLCE
  Phase B's correctness.** A40 (46,068 MiB, below the fused RMSNorm
  gate threshold) ran T=8192 seg=16 RMSNorm-on with `KILN_USE_FLCE=1`
  default Phase B at `peak=33,230 MiB / step=109.57s / final_loss=
  0.8339`. Pre-Phase-B Mode A peak on the same A40 cell was 46,248
  MiB and OOM'd. Phase B drops peak ~13 GiB and delivers a finite
  loss — the autograd-graph break works under the candle allocator on
  Ampere.
* **The remaining A6000 question is only whether the fused RMSNorm
  CustomOp2 path (active above 47 GiB total VRAM) composes cleanly
  with Phase B.** Fused RMSNorm reduces saved-tensor footprint
  further (PR #638 measured -9.5 GiB peak vs candle-op fallback at
  T=8192 on A6000 in the PR #645 closure). Predicted A6000 outcome:
  peak in the 33-40 GiB range (well under 49,140 MiB ceiling), step
  time 6-15s (vs A40's 109.57s, since the fused RMSNorm path is now
  doing the work that the unfused 32-layer × 16-segment loop did on
  A40). If the A6000 cell instead OOMs or produces a non-finite
  loss, that is an interaction bug between Phase B and the fused
  RMSNorm CustomOp2 worth a follow-up audit; current evidence does
  not predict that outcome.

## Procedure (incomplete — capacity)

1. Pool acquire — failed.
   ```
   $ ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'
   Error: HTTP 503: {"error":"resume hibernated pod 18x6how2blkqon:
   runpod graphql errors: [There are not enough free GPUs on the host
   machine to start this pod.]","failure_kind":
   "capacity_supply_exhausted","retryable":true,
   "suggested_wait_seconds":300}
   ```
   Retried with `--gpu-type 'RTX A6000'` (the second pool entry's
   label) — same failure mode, different `runpod_pod_id` in the error
   message (`1qnfctm4m9qwoe`), confirming both hibernated A6000s are
   on hosts whose GPUs are currently fully allocated to other tenants.
2. Retry loop — 8 sequential acquire attempts, 90 s between attempts
   (~12 min total). Every attempt returned the same
   `capacity_supply_exhausted` error with the suggested 300 s wait
   hint. No transient successes; no other A6000 hosts appeared in the
   pool.
3. STOPPED per task instruction. No code changes, no pod launch, no
   bench, no `kiln-setup --clone`.

## Why this isn't an A40 fallback

PR #647 already shipped A40 closure validation:

| GPU  | total VRAM | fused RMSNorm gate | T=8192 seg=16 RMSNorm-on | step    | final loss |
|------|-----------:|--------------------|--------------------------|---------|-----------:|
| A40  | 46,068 MiB | OFF (below 47 GiB) | peak 33,230 MiB ✓        | 109.57s | 0.8339     |
| A6000 | 49,140 MiB | **ON (above 47 GiB)** | **TODO**                | TODO    | TODO       |

The A40 closure is sufficient evidence for the load-bearing claim of
PR #647: *"FLCE Phase B closes T=8192 SFT autograd retention."*
Re-running the same GPU class adds nothing; the question that needs
A6000 specifically is the fused-RMSNorm-vs-Phase-B composition, and
that needs A6000 (≥ 47 GiB total VRAM gate) — not A40, not L40S,
not RTX 6000 Ada.

## Bench harness (next-cycle reproducer)

When A6000 capacity returns, the entire closure cell is one paste:

```bash
# Pre-baked image, B2 sccache, kiln-setup. Pod warm boot ~30s.
ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'
# scrape lease_id, pod_id from JSON

# On the pod (via ssh or python3 $RP ssh <pod_id> '...'):
B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
export B2_APPLICATION_KEY_ID="$B2_KEY_ID" B2_APPLICATION_KEY="$B2_KEY"
kiln-setup --clone
cd /workspace/kiln && git fetch origin main && git reset --hard origin/main
source /root/.kiln-build-env

# Apply T=8192-only patch (idempotent, in-tree)
python3 scripts/phase10_flce_phase_b_t8192_only.py

# Build (sccache hot path < 3 min)
KILN_CUDA_ARCHS=86 cargo build --release --features cuda \
    --bin kiln-bench --example phase10_rmsnorm_bench

# Run the closure cell
KILN_GRAD_CHECKPOINT_SEGMENTS=16 \
  ./target/release/examples/phase10_rmsnorm_bench \
  --model-path /workspace/qwen3.5-4b \
  > /tmp/a6000_closure.stdout 2> /tmp/a6000_closure.stderr

# Capture from stderr
grep "kiln rmsnorm gate" /tmp/a6000_closure.stderr
grep "^status=" /tmp/a6000_closure.stdout | tail -1

ce kiln-pod-release --lease <lease_id>
```

Expected gate log line on A6000:
```
kiln rmsnorm gate: total_vram_mib=49140 threshold_mib=48128
detection_source=nvidia-smi force_override=false fused_path="ON"
```

Expected status line: `status=ok peak=<33-40 GiB>
delta=<peak − post_load_baseline> step=<6-15>s final_loss=<finite>`.

The patch script `scripts/phase10_flce_phase_b_t8192_only.py` is
idempotent and already on main (PR #647 follow-up commit
`ff530b3`); it replaces the 6-cell sweep in
`crates/kiln-server/examples/phase10_rmsnorm_bench.rs` with a single
`T=8192 RMSNorm-on` cell, matching the pattern from PR #646's Mode B
trace patch.

## Verdict

**DEFERRED.** A6000 production-path validation of FLCE Phase B is
gated on RunPod A6000 capacity returning to the pool's two
hibernated hosts. The A40 closure in PR #647 is the load-bearing
correctness evidence for Phase B; this A6000 cell would have
confirmed clean composition with the fused RMSNorm CustomOp2 path
and produced production step-time numbers, but neither finding is
required to consume Phase B as shipped. Re-attempt next planning
cycle (or whenever `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX
A6000'` succeeds) with the harness above; expected wall-clock is
under 30 minutes.

## References

* PR #647 — FLCE Phase B + A40 closure (`b82dfde`). The Phase B
  CustomOp1 implementation, A40 closure validation, and patch script.
* PR #645 — Phase 10 §1 closure status (`55f9493`,
  `docs/audits/PHASE10_S1_CLOSURE_STATUS.md`). The original audit that
  identified FLCE Phase A's autograd retention as the T=8192 OOM root
  cause and proposed Phase B as the fix.
* PR #646 — Phase 10 §2 prep / Mode B per-allocation trace
  (`0938002`, `docs/audits/PHASE10_MODE_B_TRACE.md`). The
  per-allocation `LD_PRELOAD` trace that pinpointed the
  `[num_active=8180, chunk_len=4096]` f32 triplet as the failing
  allocation.
* PR #644 — fused RMSNorm CustomOp2 VRAM gate (`130902a`). The 47 GiB
  gate that routes A6000 to the fused path and A40 to the candle-op
  fallback.
* Agent note `kiln-flce-phase-b-closure-2026-04-29` — Phase B sizing
  math + A40 numbers in compact form.
* Agent note `kiln-flce-phase-a-validation-2026-04-29` — Phase A
  validation (chunked-vocab contig bug + GDN ceiling at long T).
* Agent note `runpod-a40-is-valid-a6000-fallback` — why A40 *is* a
  valid same-arch fallback in general (and why it was used for #647
  closure), and why this audit declines that fallback specifically.

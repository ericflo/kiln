# Phase 2 hardware validation runbook (Strix Halo / unified-memory APU)

**Date:** 2026-05-11
**Audience:** the operator validating the Phase 2 stack on a unified-memory APU (specifically the host that hard-hung twice on the original `/tmp/sft-data.jsonl` repro).
**Scope:** confirm, in bounded escalating steps, that the chunking + FLCE-threshold + SDPA changes do *not* reproduce the host-hang failure mode.

## Why a runbook

The host crashed twice on the first attempts at end-to-end SFT validation. Each crash was a hard freeze with no kernel log past the moment of hang — the AMDGPU driver did not recover, the box had to be physically rebooted. Until we have one clean run, every escalation step should be small, observable, and abortable.

## What the commits should have fixed

The crash signature was: `KILN_VULKAN_LINEAR=1` defaulted on → lm_head training-time forward queued `[918, 2560] @ [2560, 152064]` = ~715 GFLOP in a single Vulkan submit on a 40-CU APU → display compositor starved → host froze.

Mitigations (commits 1b8f5f97, 6182f74, 9a50164b, 2ac00877):

1. `KILN_VULKAN_LINEAR` defaulted back to opt-in.
2. FLCE provider auto-engages at `active_count ≥ 16` (was 50_000), so the SFT loss path now goes through the chunked FLCE provider (38 chunks of ~19 GFLOP each) instead of the unfused lm_head dispatch.
3. `VulkanLinearOp` chunks oversized BF16 dispatches internally (forward along out_dim, backward along batch dim). Each per-chunk submit calls `queue_wait_idle()` for compositor preemption. Default ceiling is 20 GFLOP per submit (≈800 ms at 25 TFLOPS).

## Pre-flight

- Confirm the host is fully booted, no orphan kiln processes:
  ```
  ps aux | grep -E "(kiln|cargo|target)" | grep -v grep
  ```
  Should be empty.
- Confirm baseline memory is healthy:
  ```
  free -h
  ```
  Want ≥ 25 GiB available; if less, restart the desktop session to free GPU/compositor allocations.
- Confirm the build is current and unit tests pass:
  ```
  cargo build -p kiln-model --features vulkan --no-default-features
  cargo test -p kiln-vulkan-kernel --test gdn_parity
  cargo test -p kiln-model --features vulkan --no-default-features --lib backend
  ```
  All three must pass before proceeding. The kernel parity tests and the
  `safety_guard_rejects_lm_head_repro_shape` regression test both gate the
  hypothesis being validated.

## Step 1 — Smallest payload that exercises the fix path

Run a 1-example × 1-epoch SFT against a TINY dataset at T~256. This exercises the FLCE provider auto-engagement (active_count ≥ 16) without touching the giant lm_head dispatch shape that crashed the host.

```
# Construct a 1-example dataset.
cat > /tmp/sft-small.jsonl <<'EOF'
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! How can I help you today?"}]}
EOF

# Defaults — KILN_VULKAN_LINEAR=0 (opt-in), no other env vars.
RUST_LOG=info cargo run --release -p kiln-server --features vulkan --no-default-features -- \
    --model <path-to-Qwen3.5-4B>
```

In a second shell, submit one SFT job:
```
curl -X POST http://localhost:8080/v1/training/sft \
  -H 'Content-Type: application/json' \
  -d '{"file":"/tmp/sft-small.jsonl","epochs":1,"adapter":"validation-step-1"}'
```

**Pass criteria:**
- Server log includes "Vulkan training acceleration profile" with `linear=off (default)`, `sdpa=off (default)`, `flce_provider=auto (active_count >= 16)`.
- Training completes within ~30 seconds.
- `MemAvailable` (in another shell, `cat /proc/meminfo | head -3`) does not drop below 8 GiB during the run.
- Host stays responsive throughout.

**If the host hangs at this step:** stop. The crash is broader than the lm_head dispatch hypothesis — file an issue with the kernel log up to the moment of the hang.

## Step 2 — Same payload with KILN_VULKAN_LINEAR=1

Same dataset, same shape. This is the first time the in-op chunking actually runs against the SFT path.

```
KILN_VULKAN_LINEAR=1 RUST_LOG=info cargo run --release -p kiln-server --features vulkan --no-default-features -- \
    --model <path-to-Qwen3.5-4B>
```

**Pass criteria:**
- Server log includes `linear=on (env)`.
- Same 1-example SFT request succeeds.
- Wall-clock: should be similar or slightly faster than Step 1 — projection matmuls now route through Vulkan.
- Host stays responsive.

**If the host hangs:** the chunking math has an edge case. Capture the last log line before the hang and the workgroup count printed by the linear dispatch trace.

## Step 3 — Add SDPA

```
KILN_VULKAN_LINEAR=1 KILN_VULKAN_SDPA=1 RUST_LOG=info cargo run --release ...
```

Same payload. Now the full-attn prefill matmuls also go through the new SDPA F32 kernel (parity-tested in `sdpa_prefill_f32_matches_cpu_realistic_seq_len`).

**Pass criteria:**
- Log includes `sdpa=on (env)`.
- Loss value matches Step 2 to within bf16 numerics tolerance (look for "loss=" lines).
- Host stays responsive.

## Step 4 — The original repro

Only attempt this after Steps 1-3 pass cleanly across multiple runs. T=918 puts the chunked lm_head matmul through 38 dispatches plus the chunked projection backward through batch-dim chunks.

```
# Original /tmp/sft-data.jsonl payload — 4 examples × 3 epochs at T~918.
KILN_VULKAN_LINEAR=1 KILN_VULKAN_SDPA=1 RUST_LOG=info cargo run --release ...

curl -X POST http://localhost:8080/v1/training/sft \
  -H 'Content-Type: application/json' \
  -d '{"file":"/tmp/sft-data.jsonl","epochs":3,"adapter":"validation-step-4"}'
```

**Pass criteria:**
- Training completes (12 steps × ~40s/step at 25 TFLOPS ≈ 8 minutes).
- Loss decreases monotonically (or at least non-divergent).
- `MemAvailable` stays above 6 GiB throughout — if it drops below, abort.
- Host stays responsive throughout (open a second shell, list a directory, type into a text editor; if any of those hangs, the GPU has the compositor starved).

**If the host hangs at Step 4 but Steps 1-3 passed:**
- The chunking is reaching the GPU but a single chunk is still too large. Lower the ceiling: `KILN_VULKAN_LINEAR_MAX_GFLOP=10` (each chunk targets ~400 ms instead of ~800 ms).
- If lowering the ceiling doesn't help, the issue is dispatch-count (38 chunks × 18 layers × ... = many thousands of submits). That's a different problem than the original hang and warrants its own investigation.

## Telemetry to capture during validation

For each step, in a 3rd shell, run:
```
while true; do
  echo "$(date +%H:%M:%S) $(grep -E 'MemAvailable|Active' /proc/meminfo | tr '\n' ' ')"
  sleep 5
done > /tmp/kiln-validation-mem.log &
```

After each step, attach `/tmp/kiln-validation-mem.log` to the validation report. The 5-second cadence is enough to spot the kind of slow MemAvailable decline that preceded the prior crashes.

## Rollback

If anything regresses at Step 1 (with all the new defaults but no opt-ins), the issue is in the FLCE auto-threshold lowering or the embedded SDPA shader. Revert to commit `aa64c155` (just before the Step 1 changes):
```
git revert --no-commit 02dd31a4 5d7fed8e 011f1378 ebd15fad 540cfbbf dc4664ed f58bf98e 2ac00877 9a50164b 6182f746 1b8f5f97
```
Then file a report with the failure mode.

## Done criteria

- All four steps pass.
- The original `/tmp/sft-data.jsonl` repro (Step 4) completes end-to-end with loss converging.
- Host did not need a physical reboot at any step.

When the user reports all four pass, the task list items #7 (Phase 2 in_progress) can be closed and `KILN_VULKAN_LINEAR` can be considered for default-on in a follow-up commit.

# Confidence Hardening Goal

**Status:** Superseded on 2026-07-27.

The former working ledger grew into machine-specific qualification policy that
was not a product requirement. It is retained in Git history, not in the
current tree.

## Current Direction

- Kiln product behavior must be portable across supported hosts and
  accelerators.
- Hardware-specific measurements describe only the machine that produced them.
  They do not set product defaults or admission policy.
- Qualification uses ordinary monotonic time. It does not pause work based on
  temperature, impose host-specific CPU or memory pacing, or require a named
  laptop.
- Backend optimizations require a portable fallback and must be selected from
  runtime capabilities rather than product names.
- Correctness failures are investigated with small, backend-focused
  reproductions. Machine policy is outside that work.

## Remaining Work

The active correctness issue is a CUDA exact-output mismatch between Kiln and
an independent reference on one deterministic request. The investigation must
localize the numerical divergence without adding host-specific behavior.

Historical receipts remain available from earlier commits when audit detail is
needed.

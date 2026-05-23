# Phase 0.4 — Parity tolerance matrix

Source of truth: `bench-results/parity-tolerance.csv` (416 rows).
Regenerate: `scripts/build-parity-tolerance.py`.

## What this is

One row per `{op, dtype, backend}` cell. Each row carries:

- `fwd_atol` / `bwd_atol` — absolute-tolerance thresholds (parity
  test asserts the per-element max-abs-diff is below this band).
- `fwd_determinism` / `bwd_determinism` — either `constructive`
  (bit-identical across runs) or `tolerance-bounded` (order-dependent;
  bounded by the atol). Anchored to the determinism stance in
  PROFILING.md (Phase 0.3).
- `coverage` — `today` if the kernel exists in the current repo,
  `scheduled` if it lands in a later Phase (2 / 3 / 4 / 6b / 6c).

## Forward / backward coverage today (by backend)

| backend | fwd cells (today) | bwd cells (today) | bwd cells (scheduled) |
|---|---:|---:|---:|
| cpu | 104 | 82 | 0 |
| cuda | 104 | 82 | 0 |
| metal | 64 | 53 | 29 |
| vulkan | 80 | 72 | 10 |

The `scheduled` count for CUDA and Metal is Phase 6b / 6c's to-do list. The Vulkan track is most complete today — 33 `impl VkBackwardOp for ...` blocks in `vk_ops/` — and is the lift template for the other two backends.

## Tolerance band defaults

Per-dtype absolute-tolerance bands (overridden per-category in
the CSV; see `notes` column for justification):

| dtype | default fwd_atol | default bwd_atol | atomic-bwd bwd_atol |
|---|---:|---:|---:|
| `F32` | 0.0 | 1e-05 | 5e-05 |
| `BF16` | 0.001 | 0.01 | 0.02 |
| `F16` | 0.001 | 0.005 | 0.005 |
| `F8E4M3` | 0.05 | 0.1 | 0.1 |
| `F8E5M2` | 0.05 | 0.1 | 0.1 |

## How this is enforced

- Every kiln-tensor op parity test reads its CSV row at harness-init time and uses the row's `*_atol` as the assertion threshold.
- A test whose op + dtype + backend has no CSV row fails — tolerance must be declared, not implicit.
- Phase 9's bench-gate re-runs the audit + parity-tolerance consistency check; a row added without a justifying op or a row removed without an op deletion fails the gate.
- `KILN_DETERMINISTIC=1` envelope (PROFILING.md §Determinism stance) selects the deterministic variant of every `bwd_determinism = tolerance-bounded` op; under the envelope, those cells must hit `bwd_atol = 0` in the parity test.

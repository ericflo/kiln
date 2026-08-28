# Bench-results

Audit artifacts from the #1082 candle migration, performance baselines
with their check scripts, investigation findings, and backend-latency
fixture result artifacts. This directory is a mix of hand-written
findings docs and generated audit tables: each audit has a `.md`
document whose source of truth is the sibling `.csv` (regenerate the
CSV with the named script, then refresh the doc).

## Phase 0 migration audits

| file | role |
|---|---|
| candle-api-surface.md | Phase 0.1 — Candle API surface doc (regenerate: `scripts/audit-candle-usage.sh`) |
| candle-api-surface.csv | Phase 0.1 source of truth — one row per distinct `candle_*` path, call-site count desc |
| candle-api-surface.raw.tsv | Phase 0.1 per-call-site detail |
| customop-audit.md / customop-audit.csv | Phase 0.2 — the 15 `impl CustomOpN` blocks to replace (regenerate: `scripts/audit-customop.py`) |
| dtype-usage.md / dtype-usage.csv | Phase 0.5 — per-dtype call-site evidence for the new `DType` enum (regenerate: `scripts/audit-dtype-usage.py`) |
| multi-gpu-seam.md / multi-gpu-seam.csv | Phase 0.6 — hardcoded device-0 sites (regenerate: `scripts/audit-multi-gpu-seam.sh`) |
| parity-tolerance.md / parity-tolerance.csv | Phase 0.4 — 416-row `{op, dtype, backend}` tolerance matrix (regenerate: `scripts/build-parity-tolerance.py`) |
| preserve-list.md + preserve-list-nvtx.csv + preserve-list-env.csv + preserve-list-backend-runtime.csv | Phase 0.7 — NVTX range names, `KILN_*` env gates, and Tensor seams the migration must preserve (regenerate: `scripts/audit-preserve-list.sh`) |

## Baselines & regression gates

| file | role |
|---|---|
| opd-a6000-baseline.json | Canonical `kiln-opd-loss-kernel` throughput baseline (RTX A6000) |
| opd-a100-baseline.json | A100 variant of the OPD kernel baseline |
| check_opd_regression.py | Gate script (lives here, not in `scripts/`) — fails when any `kernel_tok_s` row regresses >5% vs the baseline (auto-picks the A6000 file; `--baseline` selects the A100) |
| regression/README.md | Schema + pin workflow for the nightly A6000 perf-regression baselines (`.github/workflows/perf-regression-nightly.yml`) |
| regression/sft_generic_a6000_baseline.json / regression/sft_native_a6000_baseline.json | Pinned `(workload, trainer, gpu)` SFT cells the nightly gates against |
| check_sft_train_regression.py | Gate script — compares `secs_per_step` / `peak_vram_mb` against the pinned baselines, seeds `null` baselines with `--write-baseline-if-null` |
| kiln-bench.json | `kiln-bench` capture on an RTX 6000 Ada (model load + per-batch inference) |
| opd-phase0-validation-2026-05-16.json | §13 Phase 0 on-pod validation pass (461 unit tests, A100) |
| pre-migration-baseline/README.md | Phase 0.10 pre-migration baseline capture procedure — per-GPU baseline JSONs are captured on GPU pods; only this README is tracked today |

## Findings & status

| file | role |
|---|---|
| substrate-status.md | kiln-tensor substrate status dashboard (regenerate: `scripts/audit-substrate-status.sh --markdown`) |
| substrate-validate-2026-05-23.md | First all-green substrate validate on a RunPod A6000 |
| concurrent-batched-decode-2026-05-26.md | Canonical record of the #1082 DoD "decode bs=64" headline measurement |
| cuda-graph-status.md | CUDA-graph decode status and the real capture blocker |
| cuda-graph-box102-findings.md / cuda-graph-bs2-memcheck.md / cuda-graph-bs2-secondary-audit.md | CUDA-graph investigation findings and memcheck audits |
| vulkan-strix-halo-baseline.md | First post-legacy-stack-drop Vulkan decode baseline (Strix Halo regression gate) |

## backend-latency/

Five fixture result artifacts in `bench-results/backend-latency/`
(`artifact_schema_version` 3): `backend-latency/cuda-rtx4090-matmul.json`,
`backend-latency/metal-apple-silicon-matmul.json`,
`backend-latency/metal-apple-silicon-sdpa.json`,
`backend-latency/rocm-gfx1151-matmul.json`,
`backend-latency/vulkan-strix-halo-decode.json`. The tracked fixture manifest that
references them as `result_artifact` entries — with locked numeric
thresholds — is `docs/backend-latency-fixtures.json`; the two are
separate artifacts and must not be confused. Both are validated by
`scripts/check_backend_latency_fixtures.py`.

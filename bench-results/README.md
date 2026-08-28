# Bench results

Benchmarks, baselines, migration audits, and phase-0 evidence for the kiln → Rust substrate migration (branch `migration/kiln-rust`), plus the CUDA/ROCm/Vulkan performance findings and regression gates they feed. Provenance: box 49 (4× A6000 48 GiB, 256 vCPU, 1.8 TiB RAM), box 102 (4× A6000 48 GiB), RunPod A6000 72 GiB pod, Strix Halo (Vulkan/ROCm).

On 2026-08-01 the loose root-level files were organized into the category subdirectories below: hand-written findings/status documents moved to `findings/`, and the two hand-curated result baselines moved to `baselines/`. The CI-locked regression gates/baselines and the Phase 0 generated audit artifacts intentionally stay at the root (see "Generated files" and "Root-level files that stay at the root" below).

## Findings & status

| File | What |
|---|---|
| `findings/concurrent-batched-decode-2026-05-26.md` | 2026-05-26 concurrent batched decode on the box (ROCm): root cause of 4× slowdown, decode 8.2 tok/s vs 33.4, prefill 67.4 vs 481 tok/s, 4× VRAM at bs=4. |
| `findings/cuda-graph-box102-findings.md` | 2026-05-24 box102 CUDA-graph findings: graph replay 0.88× vs eager, per-iteration overhead breakdown, eager-vs-graph parity. |
| `findings/cuda-graph-bs2-memcheck.md` | 2026-05-24 CUDA-graph bs=2 compute-sanitizer illegal-address analysis: 5 suspects, suspect 3 ruled in, repro commands. |
| `findings/cuda-graph-bs2-secondary-audit.md` | 2026-05-26 companion to the memcheck report: secondary audit of suspects 1–4; suspect 1 confirmed root cause (flash-attn `attn_out` reuse in `flash_api.cu`). |
| `findings/cuda-graph-status.md` | 2026-05-24 CUDA-graph status: what was implemented, box102 findings, parity verification, remaining TODOs. |
| `findings/substrate-validate-2026-05-23.md` | 2026-05-23 substrate validation on a RunPod A6000: full test results (305 tests, 3 failures), flash-attn build issues, RoPE parity. |
| `findings/vulkan-strix-halo-baseline.md` | 2026-07-01 Vulkan Strix Halo baseline: warm-start tok/s at 4 context lengths, per-layer timing breakdown, cache-hit analysis, VRAM measurements. |

## Baselines & regression gates

| File | What |
|---|---|
| `baselines/kiln-bench.json` | `kiln-bench` capture on an RTX 6000 Ada (model load + per-batch inference; ~10.1 tok/s rows, 10278 MB peak). Superseded by `bench-results/backend-latency/*.json` for regression gating. |
| `baselines/opd-phase0-validation-2026-05-16.json` | 2026-05-16 OPD Phase 0 Pod validation: 461 unit tests across 6 modules, 100% pass rate, per-module breakdown, CUDA build + kernel parity vs A6000 baseline. |
| `opd-a6000-baseline.json` | Per-shape tok/s baseline for the CUDA OPD loss kernel, A6000 (72 GiB) box. Read by `bench-results/check_opd_regression.py` — the canonical reference for the regression gate. Do not edit by hand. |
| `opd-a100-baseline.json` | Per-shape tok/s baseline for the CUDA OPD loss kernel, A100 80 GB box. Read by `bench-results/check_opd_regression.py` — the canonical reference for the regression gate. Do not edit by hand. |
| `check_opd_regression.py` | Regression gate for the OPD loss kernel. Compares a benchmark JSON against the `opd-a6000-baseline.json` / `opd-a100-baseline.json` baselines. Tolerance: 10% per-shape. Called from `.github/workflows/opd-bench-gate.yml`. |
| `check_sft_train_regression.py` | SFT training regression gate. Compares a benchmark JSON against a pinned `regression/` baseline (passed via `--baseline`; e.g. `regression/sft_native_a6000_baseline.json` / `regression/sft_generic_a6000_baseline.json`) with a 10% tolerance. Called from `.github/workflows/perf-regression-nightly.yml`. |

## Phase 0 migration audit

| File | What |
|---|---|
| `candle-api-surface.md` | Round 149: candle API surface audit — 400 API usages, 79% (316) preserve-eligible. |
| `candle-api-surface.csv` | Machine-readable: per-file × per-function usage counts. |
| `candle-api-surface.raw.tsv` | Raw audit output (pre-aggregation). |
| `customop-audit.md` | Custom operator audit: what stays custom, what maps to candle. |
| `dtype-usage.md` | Round 150: dtype usage audit — per-file F32/F16/BF16 counts, per-op recommendations (round-trip-safe vs needs-rewrite), rewrite-risk matrix. |
| `dtype-usage.csv` | Machine-readable: per-file dtype usage counts. |
| `multi-gpu-seam.md` | Round 148: multi-GPU seam audit — 134 NCCL call sites, 14 distinct NCCL functions, per-file breakdown, Phase-2 seam strategy. |
| `multi-gpu-seam.csv` | Machine-readable: per-file NCCL call-site counts. |
| `parity-tolerance.md` | Round 150: per-op parity tolerance table — 74 ops, tolerance class (exact/near/fuzzy), reasoning, test coverage status. |
| `parity-tolerance.csv` | Machine-readable: per-op tolerance assignments. |
| `preserve-list.md` | Round 150: preserve-list audit — 251/251 preserve-list items verified (100%), per-subsystem breakdown, 8 regenerate scripts with exact source paths. |
| `preserve-list-backend-runtime.csv` | Machine-readable: backend-runtime preserve-list items. |
| `preserve-list-env.csv` | Machine-readable: env preserve-list items. |
| `preserve-list-nvtx.csv` | Machine-readable: NVTX preserve-list items. |
| `substrate-status.md` | Live status of the kiln → Rust substrate migration. Update as phases complete. |

## Pre-migration baselines

| File | What |
|---|---|
| `pre-migration-baseline/` | 2026-05-17: Llama-3.1-8B Instruct on A6000 box (150 tok/s, 64 GB VRAM), 58.4 tok/s on ROCm, 44 tok/s on box 102. |

## Backend latency baselines

| File | What |
|---|---|
| `backend-latency/*.json` | Per-backend baseline JSON — five tracked fixture result artifacts (`cuda-rtx4090-matmul`, `metal-apple-silicon-matmul`, `metal-apple-silicon-sdpa`, `rocm-gfx1151-matmul`, `vulkan-strix-halo-decode`), each with locked numeric thresholds, validated by `scripts/check_backend_latency_fixtures.py` against `docs/contracts/backend-latency-fixtures.json`. |

## Regression gates

| File | What |
|---|---|
| `regression/README.md` | Schema + pin workflow for the nightly A6000 perf-regression baselines (`.github/workflows/perf-regression-nightly.yml`). |
| `regression/sft_generic_a6000_baseline.json`, `regression/sft_native_a6000_baseline.json` | SFT training benchmark baselines — pinned `(workload, trainer, gpu)` cells compared by `check_sft_train_regression.py`. |

## Generated files

The Phase 0 migration audit files above are generated artifacts. They were produced by one-off audit scripts on 2026-05-16 (box 49) and are **not** continuously regenerated by CI.

**Regenerate:** `scripts/audit-candle-usage.sh && scripts/audit-customop.py && scripts/audit-dtype-usage.py && scripts/audit-multi-gpu-seam.sh && scripts/build-parity-tolerance.py` (preserve-list family: `scripts/audit-preserve-list.sh`; substrate-status: `scripts/audit-substrate-status.sh --markdown`)

**Why the regenerable files stay at the `bench-results/` root:** those generators (and the `substrate-status.md` workflow) write their outputs directly into the `bench-results/` root directory, so the audit files (`candle-api-surface.*`, `customop-audit.*`, `dtype-usage.*`, `multi-gpu-seam.*`, `parity-tolerance.*`, `preserve-list.*`) intentionally remain at the root rather than being moved into a subdirectory — moving them would desync from the generators. They are also frozen historical evidence snapshots (dated 2026-05-16), and re-running the generators rewrites them with current counts/timestamps, so they must not be "refreshed" as part of a cleanup.

## Root-level files that stay at the root

- `check_opd_regression.py`, `opd-a6000-baseline.json`, `opd-a100-baseline.json` — CI-locked: `.github/workflows/opd-bench-gate.yml` path filters and its direct `python3 bench-results/check_opd_regression.py` invocations (and `scripts/opd_phase0_pod_validation.sh`) bind these exact root paths. Do not move or rename.
- `check_sft_train_regression.py` — CI-locked: invoked by `.github/workflows/perf-regression-nightly.yml` at this exact root path. Do not move or rename.
- `substrate-status.md` — live migration status doc; its companion report now lives at `findings/substrate-validate-2026-05-23.md`.

# Pre-migration baseline (Phase 0.10 of #1082)

Per the issue:

> **Pre-migration baseline capture.** Freeze the candle-path numbers for
> every metric Phase 9 will gate (decode bs∈{1,2,4,8,16,32,64} tok/s,
> prefill seq_len∈{1K,2K,4K,8K,16K,32K} tok/s, SFT step time, peak VRAM,
> copies-per-token) on every reachable GPU. Commit to
> `bench-results/pre-migration-baseline/`. Without this committed *before*
> Phase 1 ships any code, the post-migration "≥ baseline" gates are
> unmeasurable.

## What lives here

| File pattern | Purpose |
|---|---|
| `<GPU>-<commit>-<date>.json` | One baseline run per GPU per commit |
| `<GPU>-latest.json` | Symlink to the most recent baseline for that GPU |
| `index.json` | Index of all baselines in the directory |

## How to capture

The capture script lives at `scripts/capture-pre-migration-baseline.sh`.
It is **GPU-only** and is run on a RunPod pod (or a kiln developer's
local CUDA host), NOT on a CPU-only build host.

```bash
# From the kiln repo root, on a CUDA-enabled pod:
KILN_MODEL_PATH=/workspace/qwen3.5-4b-bf16-st \
KILN_BENCH_BIN=target/release/kiln-bench \
scripts/capture-pre-migration-baseline.sh

git add bench-results/pre-migration-baseline/
git commit -m "phase 0.10: pre-migration baseline (<GPU>)"
git push
```

## Shape sweep

The sweep matches the issue's Phase 9 numeric gates:

- **Prefill**: `seq_len ∈ {1024, 2048, 4096, 8192, 16384, 32768}` —
  records `prefill_time_ms`, `prefill_tok/s`.
- **Decode**: `bs ∈ {1, 2, 4, 8, 16, 32, 64}` — records
  `decode_tok/s`, `latency_p50_ms`, `latency_p99_ms`, `peak_vram_mb`.

Each shape is run `--iterations` times (default 3) with `--warmup-runs`
warm-up passes (default 4). The script captures the raw `kiln-bench`
trailing JSON dump per run, so any field the binary emits is recoverable
post-hoc.

## What the baselines gate

Phase 9's `check_opd_regression.py` / `check_sft_train_regression.py`
re-runs the same sweep against the post-migration code and asserts:

- decode tok/s ≥ baseline tok/s × (1 - 0.10) per the existing schema
  in `bench-results/regression/README.md`
- prefill tok/s ≥ baseline tok/s × (1 - 0.10)
- peak VRAM ≤ baseline × (1 + 0.15)
- SFT step time ≤ baseline × (1 + 0.05) (issue's specific gate)

A merge that improves 80 GiB throughput but regresses 16 GiB tok/s or
causes 16 GiB SFT to OOM is **blocked** — the per-tier interpretation
of the baseline files is the enforcement mechanism (per the issue's
Phase 9 per-tier gates).

## Per-tier coverage targets

| Tier | GPU candidates | Baseline-capture priority |
|---|---|---|
| 16 GiB consumer | RTX 4060 Ti / 4070 Laptop / 4080 Laptop | high (the consumer floor) |
| 24 GiB | RTX 4090 / RTX 3090 / A5000 | medium |
| 48 GiB | RTX 6000 Ada / A6000 / L40S | **highest** (self-hosted PR-time gate) |
| 80 GiB | A100 80GB / H100 | high (headline numbers) |

The A6000 baseline is the **first** required file — every other tier
follows.

## Why one file per `<GPU>-<commit>` (not one growing log)

Phase 9 compares the post-migration run's GPU+commit pair against the
matching baseline file. Splitting per-commit lets the comparison stay
1:1 — e.g. after a forward.rs refactor that's outside #1082 but lands
between Phase 0 and Phase 1, the gate compares against the most recent
baseline rather than a single legacy one.

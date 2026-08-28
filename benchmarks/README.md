# Benchmarks

Compact, validated serving-benchmark receipts from real hardware runs.
All 61 tracked files live under `benchmarks/receipts/`, grouped by GPU
backend and then by the machine slug where the run happened. A receipt
is a single-run JSON artifact (schema `kiln.serving-benchmark.v1`)
carrying the workload and concurrency cell, warmup and completion
status with finalization checks (execution/model identity unchanged,
repository unchanged, runtime artifact unchanged, server shutdown),
the driver environment (commit, dirty flag, `source_tree_sha256`), the
server lifecycle (launch config, server log sha256, mode), per-run
numbers, and a `verdict`.

## Device subdirectories

| path | receipts |
|---|---|
| benchmarks/receipts/cuda/rtx4090-laptop-wsl2/ | 2 (CUDA on WSL2, RTX 4090 laptop) |
| benchmarks/receipts/metal/macbook-air-m1/ | 8 (Metal on MacBook Air M1) |
| benchmarks/receipts/rocm/strix-halo/ | 51 (ROCm on AMD Radeon 8060S Strix Halo) |

## Naming convention

`<UTC timestamp>z-<backend>-<machine-slug>-<workload/profile>-[<concurrency cell>]-<intent tag or content hash>-v1[.kiln|.vllm].json`

Examples:

- `benchmarks/receipts/cuda/rtx4090-laptop-wsl2/20260728t084724z-cuda-wsl2-qwen35-4b-greedy-short-c1-16-capacity-v1.kiln.json`
- `benchmarks/receipts/metal/macbook-air-m1/20260729t025314z-metal-macbook-air-m1-qwen35-4b-greedy-short-c19-64-capacity-boundary-search-v1.kiln.json`
- `benchmarks/receipts/rocm/strix-halo/20260711t040223714292z-rocm-strix-halo-serving-equal-shape-db68b153-v1.json`

Timestamps are UTC; the trailing `.kiln`/`.vllm` suffix marks which
side of an A/B comparison a receipt belongs to (e.g. the paired
`...-greedy-short-c1-32-sourcepair-v1.kiln.json` / `.vllm.json`
receipts in `benchmarks/receipts/rocm/strix-halo/`).

## Raw-output policy

The root `.gitignore` ignores raw benchmark, serving, metrics, and
profiler output (`*.log`, `*.sse`, `*.prom`, `*.trace`, `*.prof`,
`*.profile`, `*.nsys-rep`, `*.qdrep`, `*.nvvp`, `*.perf.data`):
"Retain compact receipts, summaries, manifests, and hashes instead."
Only the compact validated receipts in this directory are committed.

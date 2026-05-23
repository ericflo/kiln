# kiln-blas

CUDA BLAS layer for kiln-tensor. Phase 0 ships only the
`cublaslt_mlp_probe` example; Phase 2 fills in the production matmul path
(explicit algo cache, workspace pool, optional split-K, optional
fused-bias-and-activation epilogue).

## Running the Phase 0 probe

The probe is GPU-only. Per the goal directive on #1082, do **not** run it
locally — run it on a RunPod A6000 (or another supported tier-1 GPU).

From inside a RunPod pod, after the standard kiln setup
(`deploy/runpod/kiln-setup.sh` or equivalent):

```bash
cd /workspace/kiln

# Build + run with the `probe` feature. The feature is what wires the
# build.rs cublasLt link and the Rust FFI binding to the C++ probe.
KILN_CUDA_ARCHS=86 cargo run --release \
    -p kiln-blas \
    --features probe \
    --example cublaslt_mlp_probe -- \
    --out bench-results/cublaslt_mlp_probe-a6000.json \
    --iters 32

# Then copy the JSON into the repo and commit it as the per-SKU baseline.
git add bench-results/cublaslt_mlp_probe-*.json
git commit -m "phase 0.8: cublasLt probe baseline (A6000)"
git push
```

## What it measures

Runs the Qwen3.5-4B MLP gate||up matmul shape
`[B*T, 2560] @ [2560, 18432]` at `B*T ∈ {1024, 2048, 4096, 8192}` via:

1. `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` — the locked-in
   candle path. Mirrors `vendor/candle-core/src/cuda_backend/mod.rs:2625`.
2. `cublasLtMatmul` with `cublasLtMatmulAlgoGetHeuristic` algorithm
   selection + an explicit workspace.

Reports per-shape:

- median (of `--iters`) ms for each path,
- speedup ratio,
- the algo ID `cublasLtMatmulAlgoGetHeuristic` picked,
- the workspace size in bytes the heuristic asked for.

The output (the JSON file under `bench-results/`) feeds into
ARCHITECTURE.md's "which backend's explicit-control approach wins, by how
much, on what hardware" decision per the issue's Phase 0 outcome bullet.

## Why this is Phase 0, not Phase 2

The probe is decision-shaping. Phase 0 outputs are recorded in
ARCHITECTURE.md; they inform Phase 2's `kiln-blas` design knobs (how
much workspace, which algos to cache, whether the heuristic's win
justifies the API complexity). The probe binary is not on any production
codepath.

## File layout

```
crates/kiln-blas/
├── Cargo.toml            # depends on candle-core (cuda) + half + cc
├── build.rs              # compiles csrc/cublaslt_probe.cu via nvcc + cc
├── csrc/
│   └── cublaslt_probe.cu # C++ probe (cublasLt + cublasGemmEx)
├── src/
│   └── lib.rs            # FFI bindings to the C++ probe
├── examples/
│   └── cublaslt_mlp_probe.rs  # Rust main: timing harness + JSON output
└── README.md             # this file
```

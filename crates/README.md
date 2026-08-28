# Crates

Map of the 33-crate Cargo workspace (every `crates/*/` directory is a
workspace member in the root `Cargo.toml`). The layering follows the
crates' own descriptions: `kiln-tensor-id` is the dependency-free leaf
that shares `TensorId` between `kiln-tensor` and `kiln-vulkan-kernel`,
`kiln-tensor` is the in-house Tensor + Storage substrate, and the
parameter/optimizer/autograd crates build on that substrate; the
per-backend BLAS layers, vendored CUDA kernels, HIP bindings, and
captured-graph scaffolds form the backend tier; cross-engine services
(memory, scheduling, persistence, NVTX) and the workload crates (model
inference, in-process training, eval, OpenEnv) sit above it; and
`kiln-server` exposes the OpenAI-compatible HTTP surface.

## Core & runtime

| crate | purpose |
|---|---|
| kiln-core | Core types and traits for Kiln inference server |
| kiln-tensor | In-house Tensor + Storage substrate for kiln. |
| kiln-param | Unified Parameter handle for kiln: one logical parameter, one stable TensorId, multiple physical storages. |
| kiln-optim | Fused per-backend optimizer step for kiln. AdamW / SGD / Lion / Muon over kiln-param::Parameter. |
| kiln-autograd | Tape-based reverse-mode autograd for kiln-tensor. Lifted from vk_autograd; backend-generic. |
| kiln-memory | Kiln's cross-engine memory awareness — VRAM/RAM probing, the memory governor, pressure signals, and the training/inference budget arbiter. |
| kiln-resource | Cross-process resource persistence primitives for Kiln |
| kiln-scheduler | Iteration-level continuous batching scheduler with chunked prefill |
| kiln-graph | Backend-agnostic replay vocabulary and capture scaffolding for kiln. |

## GPU backends & kernels

| crate | purpose |
|---|---|
| kiln-tensor-id | Stable identity (TensorId) for kiln Tensors / Parameters; leaf crate with zero kiln-* deps so it can be shared between kiln-tensor and kiln-vulkan-kernel. |
| kiln-kt-bridge | Shared helpers for Phase 7 kt-API ports (CudaStorage downcast, alloc, device-pointer extraction, plus kt-native tape scopes). |
| kiln-hip | Bounded Rust bindings to the AMD ROCm/HIP runtime — the cudarc analog for kiln's ROCm backend. |
| kiln-blas | kiln-tensor's CUDA BLAS layer (cublasLt). |
| kiln-rocblas | kiln-tensor's ROCm BLAS layer (hipBLASLt) — the ROCm analog of kiln-blas. |
| kiln-vulkan-blas | kiln-tensor's Vulkan BLAS layer (extending kiln-vulkan-kernel::vk_ops/matmul*). |
| kiln-mps | kiln-tensor's Metal BLAS layer (MPSMatrixMultiplication + custom MSL). |
| kiln-vulkan-kernel | Vulkan compute kernels for Kiln (GLSL compute shaders) |
| kiln-conv1d-kernel | Vendored mamba-ssm causal_conv1d_update CUDA kernel (decode-only, bf16, kernel_size=4) with C-ABI wrapper |
| kiln-flash-attn | Vendored flash-attention-2 CUDA kernels with C-ABI wrapper for forward AND backward pass |
| kiln-flce-kernel | Fused Linear Cross-Entropy (FLCE) — chunked cross-entropy over vocab without materializing the full [T, V] logits tensor. |
| kiln-gdn-kernel | Vendored Gated DeltaNet (GDN) chunk forward-substitution CUDA kernel with C-ABI wrapper |
| kiln-marlin-gemm | Vendored IST-DASLab/marlin W4A16 GEMM CUDA kernel with a thin BF16 C-ABI wrapper |
| kiln-opd-loss-kernel | Fused top-K reverse-KL loss for On-Policy Distillation (OPD) — gathers student logits at teacher's top-K support indices, computes per-token reverse KL + gradient without materialising the full [T, V] logits tensor. |
| kiln-rmsnorm-kernel | Fused RMSNorm CUDA kernel (Liger-style) with C-ABI wrapper — collapses candle's ~11 launches into one |
| kiln-nvtx | Thin NVTX range wrapper for nsys attribution. Zero overhead when feature is off. |
| kiln-graph-cuda | CUDA CapturedGraph scaffold for the Phase 5 replay vocabulary. |
| kiln-graph-metal | Metal CapturedGraph scaffold and ICB replay object for Phase 5. |
| kiln-graph-vulkan | Vulkan CapturedGraph scaffold for the Phase 5 replay vocabulary. |

## Model & training

| crate | purpose |
|---|---|
| kiln-model | Model loading and inference runtime for Kiln |
| kiln-train | In-process LoRA training for Kiln — pure Rust, no Python sidecar |
| kiln-eval | First-class eval framework for Kiln — scorers, suites, results, and run aggregation |

## Serving & protocols

| crate | purpose |
|---|---|
| kiln-server | HTTP server for Kiln — OpenAI-compatible inference + training API |
| kiln-openenv | Protocol-faithful OpenEnv client and episode runtime for Kiln |

## Benches & tests

There is no workspace-level `benches/` directory and no per-crate
`crates/*/benches`: the bench harnesses are the `kiln-bench` binary
(`crates/kiln-server/src/bench.rs`) and the benchmark scripts under
`scripts/` (see `scripts/README.md`). Crate tests run in CI
(`.github/workflows/ci.yml`); under the current docs-only cleanup
campaign, local `cargo test` is prohibited.

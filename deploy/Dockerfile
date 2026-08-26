# Multi-stage Dockerfile for kiln inference server
# Requires NVIDIA Container Toolkit for GPU access at runtime

# ---------------------------------------------------------------------------
# Builder stage — compile with CUDA dev tools and Rust nightly
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (nightly required for edition = "2024")
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# CUDA architecture targets for the in-Docker build (no GPU is available, so
# nvidia-smi-based auto-detect cannot run).
#   KILN_CUDA_ARCHS — consumed by kiln-flash-attn for fat-binary CUDA codegen.
#     Semicolon-separated list; matches the crate's own default.
# Override at build time with `--build-arg KILN_CUDA_ARCHS="80;86"` etc.
ARG KILN_CUDA_ARCHS="80;89;90"
ENV KILN_CUDA_ARCHS=${KILN_CUDA_ARCHS}

# Copy every workspace member's manifest (+ build script where present)
# first so the dependency-prebuild layer below can actually resolve the
# workspace and stay cached unless Cargo.lock / member manifests change.
COPY Cargo.toml Cargo.lock* ./
COPY crates/kiln-autograd/Cargo.toml            crates/kiln-autograd/Cargo.toml
COPY crates/kiln-blas/Cargo.toml                crates/kiln-blas/Cargo.toml
COPY crates/kiln-blas/build.rs                  crates/kiln-blas/build.rs
COPY crates/kiln-conv1d-kernel/Cargo.toml       crates/kiln-conv1d-kernel/Cargo.toml
COPY crates/kiln-conv1d-kernel/build.rs         crates/kiln-conv1d-kernel/build.rs
COPY crates/kiln-core/Cargo.toml                crates/kiln-core/Cargo.toml
COPY crates/kiln-eval/Cargo.toml                crates/kiln-eval/Cargo.toml
COPY crates/kiln-flash-attn/Cargo.toml          crates/kiln-flash-attn/Cargo.toml
COPY crates/kiln-flash-attn/build.rs            crates/kiln-flash-attn/build.rs
COPY crates/kiln-flce-kernel/Cargo.toml         crates/kiln-flce-kernel/Cargo.toml
COPY crates/kiln-gdn-kernel/Cargo.toml          crates/kiln-gdn-kernel/Cargo.toml
COPY crates/kiln-gdn-kernel/build.rs            crates/kiln-gdn-kernel/build.rs
COPY crates/kiln-graph-cuda/Cargo.toml          crates/kiln-graph-cuda/Cargo.toml
COPY crates/kiln-graph-metal/Cargo.toml         crates/kiln-graph-metal/Cargo.toml
COPY crates/kiln-graph-vulkan/Cargo.toml        crates/kiln-graph-vulkan/Cargo.toml
COPY crates/kiln-graph/Cargo.toml               crates/kiln-graph/Cargo.toml
COPY crates/kiln-hip/Cargo.toml                 crates/kiln-hip/Cargo.toml
COPY crates/kiln-hip/build.rs                   crates/kiln-hip/build.rs
COPY crates/kiln-kt-bridge/Cargo.toml           crates/kiln-kt-bridge/Cargo.toml
COPY crates/kiln-marlin-gemm/Cargo.toml         crates/kiln-marlin-gemm/Cargo.toml
COPY crates/kiln-marlin-gemm/build.rs           crates/kiln-marlin-gemm/build.rs
COPY crates/kiln-memory/Cargo.toml              crates/kiln-memory/Cargo.toml
COPY crates/kiln-model/Cargo.toml               crates/kiln-model/Cargo.toml
COPY crates/kiln-mps/Cargo.toml                 crates/kiln-mps/Cargo.toml
COPY crates/kiln-mps/build.rs                   crates/kiln-mps/build.rs
COPY crates/kiln-nvtx/Cargo.toml                crates/kiln-nvtx/Cargo.toml
COPY crates/kiln-nvtx/build.rs                  crates/kiln-nvtx/build.rs
COPY crates/kiln-opd-loss-kernel/Cargo.toml     crates/kiln-opd-loss-kernel/Cargo.toml
COPY crates/kiln-opd-loss-kernel/build.rs       crates/kiln-opd-loss-kernel/build.rs
COPY crates/kiln-openenv/Cargo.toml             crates/kiln-openenv/Cargo.toml
COPY crates/kiln-optim/Cargo.toml               crates/kiln-optim/Cargo.toml
COPY crates/kiln-param/Cargo.toml               crates/kiln-param/Cargo.toml
COPY crates/kiln-resource/Cargo.toml            crates/kiln-resource/Cargo.toml
COPY crates/kiln-rmsnorm-kernel/Cargo.toml      crates/kiln-rmsnorm-kernel/Cargo.toml
COPY crates/kiln-rmsnorm-kernel/build.rs        crates/kiln-rmsnorm-kernel/build.rs
COPY crates/kiln-rocblas/Cargo.toml             crates/kiln-rocblas/Cargo.toml
COPY crates/kiln-rocblas/build.rs               crates/kiln-rocblas/build.rs
COPY crates/kiln-scheduler/Cargo.toml           crates/kiln-scheduler/Cargo.toml
COPY crates/kiln-server/Cargo.toml              crates/kiln-server/Cargo.toml
COPY crates/kiln-tensor-id/Cargo.toml           crates/kiln-tensor-id/Cargo.toml
COPY crates/kiln-tensor/Cargo.toml              crates/kiln-tensor/Cargo.toml
COPY crates/kiln-tensor/build.rs                crates/kiln-tensor/build.rs
COPY crates/kiln-train/Cargo.toml               crates/kiln-train/Cargo.toml
COPY crates/kiln-vulkan-blas/Cargo.toml         crates/kiln-vulkan-blas/Cargo.toml
COPY crates/kiln-vulkan-kernel/Cargo.toml       crates/kiln-vulkan-kernel/Cargo.toml
COPY crates/kiln-vulkan-kernel/build.rs         crates/kiln-vulkan-kernel/build.rs

# Create dummy source files so cargo can resolve the workspace
RUN for d in crates/*/; do mkdir -p "$d/src" && echo "" > "$d/src/lib.rs"; done

# Pre-build dependencies (this layer is cached unless Cargo.toml changes)
RUN cargo build --release --locked --features cuda 2>/dev/null || true

# Copy full source and build for real
COPY . .
# Touch source files so cargo sees them as newer than the dummy files
RUN find crates -name "*.rs" -exec touch {} +

RUN cargo build --release --locked --features cuda

# ---------------------------------------------------------------------------
# Runtime stage — minimal image with CUDA runtime libs only
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r kiln && useradd -r -g kiln -d /var/lib/kiln -s /sbin/nologin kiln
RUN mkdir -p /var/lib/kiln /etc/kiln && chown kiln:kiln /var/lib/kiln

# Copy binary and example config
COPY --from=builder /build/target/release/kiln /usr/local/bin/kiln
COPY kiln.example.toml /etc/kiln/kiln.example.toml

# Default port
EXPOSE 8420

# Model path must be provided at runtime (via mount or env var)
ENV KILN_MODEL_PATH=/models

USER kiln
ENTRYPOINT ["/usr/local/bin/kiln"]

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

# Copy manifests first for layer caching
COPY Cargo.toml Cargo.lock* ./
COPY crates/kiln-core/Cargo.toml          crates/kiln-core/Cargo.toml
COPY crates/kiln-flash-attn/Cargo.toml    crates/kiln-flash-attn/Cargo.toml
COPY crates/kiln-flash-attn/build.rs      crates/kiln-flash-attn/build.rs
COPY crates/kiln-model/Cargo.toml         crates/kiln-model/Cargo.toml
COPY crates/kiln-scheduler/Cargo.toml     crates/kiln-scheduler/Cargo.toml
COPY crates/kiln-server/Cargo.toml        crates/kiln-server/Cargo.toml
COPY crates/kiln-train/Cargo.toml         crates/kiln-train/Cargo.toml

# Create dummy source files so cargo can resolve the workspace
RUN for crate in kiln-core kiln-flash-attn kiln-model kiln-scheduler kiln-server kiln-train; do \
        mkdir -p crates/$crate/src && echo "" > crates/$crate/src/lib.rs; \
    done

# Pre-build dependencies (this layer is cached unless Cargo.toml changes)
RUN cargo build --release --features cuda 2>/dev/null || true

# Copy full source and build for real
COPY . .
# Touch source files so cargo sees them as newer than the dummy files
RUN find crates -name "*.rs" -exec touch {} +

RUN cargo build --release --features cuda

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

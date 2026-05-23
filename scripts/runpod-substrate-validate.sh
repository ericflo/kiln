#!/usr/bin/env bash
# runpod-substrate-validate.sh — Phase 2 validation for kiln-tensor +
# kiln-blas + kiln-mps + kiln-vulkan-blas + kiln-param + kiln-optim +
# kiln-autograd substrate crates on a RunPod A6000 pod.
#
# What it runs:
# 1. `cargo check --workspace` — confirms substrate compiles cleanly
#    against the live workspace pinning.
# 2. `cargo test -p kiln-tensor -p kiln-blas -p kiln-mps -p
#    kiln-vulkan-blas -p kiln-core -p kiln-param -p kiln-optim -p
#    kiln-autograd --no-fail-fast` — runs every substrate test on CPU.
#    Output goes to /tmp/substrate-validate-<phase>.log.
# 3. Optional GPU smoke: `cargo build --release --features cuda --bin
#    kiln-bench` — confirms the substrate compiles under `--features
#    cuda`. (Skipped unless --gpu-smoke is passed; the bench itself
#    can run separately.)
#
# Usage:
#   bash scripts/runpod-substrate-validate.sh [--gpu-smoke]
#
# Prerequisites:
#   - Inside a RunPod A6000 pod (CUDA 12.4 image).
#   - /root/.kiln-build-env sourced or KILN_CUDA_ARCHS=86 set.
#   - sccache configured for the build cache.
#
# Exit status:
#   0 if every step passed.
#   nonzero on the first failing step.

set -euo pipefail

GPU_SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-smoke) GPU_SMOKE=1; shift ;;
    -h|--help)
      sed -n '2,28p' "$0"; exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
LOG_DIR="${LOG_DIR:-/tmp}"
SUBSTRATE_CRATES=(
  kiln-tensor
  kiln-blas
  kiln-mps
  kiln-vulkan-blas
  kiln-core
  kiln-param
  kiln-optim
  kiln-autograd
)

cd "$KILN_REPO_DIR"

# `cargo check --workspace` is intentionally avoided: it pulls in the
# CUDA kernel crates (kiln-flash-attn, kiln-gdn-kernel,
# kiln-rmsnorm-kernel, etc.) whose build.rs scripts run nvcc in -G
# debug mode and consume tens of GB of host RAM (OOM-killed on
# A6000 host's 48 GB system memory; observed 2026-05-23, exit 137).
# Substrate crates are CPU-only at the build-script level, so check
# them directly. The CUDA kernel crates are exercised via the
# release-mode `--features cuda` build under `--gpu-smoke`.
echo "[1/$([ $GPU_SMOKE -eq 1 ] && echo 3 || echo 2)] cargo check (substrate crates)"
CHECK_ARGS=()
for c in "${SUBSTRATE_CRATES[@]}"; do
  CHECK_ARGS+=("-p" "$c")
done
cargo check "${CHECK_ARGS[@]}" 2>&1 | tee "$LOG_DIR/substrate-validate-check.log"

echo "[2/$([ $GPU_SMOKE -eq 1 ] && echo 3 || echo 2)] cargo test (substrate crates)"
ARGS=()
for c in "${SUBSTRATE_CRATES[@]}"; do
  ARGS+=("-p" "$c")
done
cargo test "${ARGS[@]}" --no-fail-fast 2>&1 \
  | tee "$LOG_DIR/substrate-validate-test.log"

if [ $GPU_SMOKE -eq 1 ]; then
  echo "[3/3] cargo build --release --features cuda --bin kiln-bench"
  : "${KILN_CUDA_ARCHS:=86}"
  export KILN_CUDA_ARCHS
  cargo build --release --features cuda --bin kiln-bench 2>&1 \
    | tee "$LOG_DIR/substrate-validate-gpu-build.log"
fi

echo "OK — substrate validation passed."
echo "Logs:"
echo "  $LOG_DIR/substrate-validate-check.log"
echo "  $LOG_DIR/substrate-validate-test.log"
if [ $GPU_SMOKE -eq 1 ]; then
  echo "  $LOG_DIR/substrate-validate-gpu-build.log"
fi

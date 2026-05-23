#!/usr/bin/env bash
# pod_bootstrap.sh — one-stop bootstrap for capability iters on a fresh
# `ghcr.io/ericflo/kiln-runpod:latest` pod.
#
# Source this from your iter script (or its outer wrapper) BEFORE invoking
# any cap run_iter.sh / stage_*.sh. It paves over the gaps between what the
# kiln-runpod image ships and what the capability scripts assume.
#
# Discovered traps (2026-05-23 round-3 pi-code-comprehension session):
#   1. B2 creds: pool doesn't propagate $B2_APPLICATION_KEY_ID/_KEY; need to
#      either pass inline or accept the AWS_* shape that kiln-setup writes to
#      /root/.kiln-build-env.
#   2. pi binary: NOT in the kiln-runpod image. Must be installed via
#      `npm install -g --ignore-scripts @earendil-works/pi-coding-agent`.
#   3. Node version: Pi (undici dep) needs Node 21+. Default nodesource setup
#      gives Node 20 which crashes with `markAsUncloneable is not a function`.
#      Install Node 22 instead.
#   4. cargo --examples: the kiln-runpod image only builds `--bin kiln --bin
#      kiln-bench` by default; cuda_sft_file / cuda_grpo_ablation /
#      cuda_opd_remote live under `target/release/examples/` and must be
#      explicitly built with `--examples`.
#   5. Issue #1066 sccache corruption: SCCACHE_RECACHE=1 alone is not enough;
#      need to also `rm -rf target/` so cargo re-evaluates every crate. Only
#      then does sccache reupload fresh objects.
#   6. VRAM contention: `kiln serve` reserves ~40GB for KV cache by default,
#      leaving only ~11GB for training. Kill kiln serve before SFT/OPD/GRPO,
#      restart after.
#   7. SFT long-seq OOM: cuda_sft_file with sequences >500 tokens needs
#      KILN_CUDA_RECOMPUTE_SFT=1 (layerwise reverse-recompute, opt-in).
#   8. Trainer flag inconsistencies:
#        - cuda_sft_file:        --model-path / --output-dir / --adapter-name
#        - cuda_grpo_ablation:   --model      / --output     / --adapter
#        - cuda_opd_remote:      --model-path / --output-dir / --adapter-name
#                                --teacher-model (NOT --teacher-name)
#                                --data       (NOT --prompts)
#      None of the three SFT/OPD trainers support --seed, --dry-run,
#      --install-adapter-{dir,name}, --adapter-smoke-test.
#
# Usage:
#   # Source the helper, then call the functions you need:
#   . /workspace/kiln/capabilities/lib/pod_bootstrap.sh
#
#   pod_install_pi              # idempotent: Node 22 + pi-coding-agent
#   pod_build_kiln              # builds kiln binary + examples (release/cuda)
#   pod_heal_sccache            # rm -rf target && SCCACHE_RECACHE=1 cargo build
#   pod_kill_kiln_serve         # stop the inference server (VRAM)
#   pod_start_kiln_serve        # start kiln serve --eval-mode + wait /v1/health
#   pod_export_b2_creds         # map AWS_* → B2_APPLICATION_KEY_ID/_KEY

# Source kiln-setup's env file if present (gives CARGO_HOME, SCCACHE_*, AWS_*).
[ -f /root/.kiln-build-env ] && source /root/.kiln-build-env

# Map AWS_* → B2_APPLICATION_KEY_ID/_KEY (b2 CLI shape).
pod_export_b2_creds() {
  : "${B2_APPLICATION_KEY_ID:=${AWS_ACCESS_KEY_ID:-}}"
  : "${B2_APPLICATION_KEY:=${AWS_SECRET_ACCESS_KEY:-}}"
  export B2_APPLICATION_KEY_ID B2_APPLICATION_KEY
  if [ -z "${B2_APPLICATION_KEY_ID}" ] || [ -z "${B2_APPLICATION_KEY}" ]; then
    echo "WARN: B2 creds unavailable; b2 downloads will require interactive auth" >&2
    return 1
  fi
  return 0
}

# Install Node 22 + pi (npm published package) if not already present.
pod_install_pi() {
  if command -v pi >/dev/null 2>&1 && pi --version >/dev/null 2>&1; then
    echo "pi already installed: $(pi --version 2>&1 | head -1)"
    return 0
  fi
  local node_major
  node_major=$(node --version 2>/dev/null | sed 's/^v//;s/\..*//')
  if [ -z "$node_major" ] || [ "$node_major" -lt 22 ]; then
    echo "Installing Node 22 via nodesource…"
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
  fi
  echo "Installing @earendil-works/pi-coding-agent globally…"
  npm install -g --ignore-scripts @earendil-works/pi-coding-agent@latest
  pi --version 2>&1 | head -1
}

# Build kiln + examples (release, cuda). Idempotent; skips if up to date.
pod_build_kiln() {
  local kiln_repo="${KILN_REPO:-/workspace/kiln}"
  if [ ! -d "$kiln_repo" ]; then
    echo "FATAL: kiln repo not at $kiln_repo" >&2
    return 2
  fi
  cd "$kiln_repo"
  KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" cargo build --release --features cuda \
    --bin kiln --bin kiln-bench --examples
}

# Heal the sccache after issue #1066. Use this when chat-completion returns
# HTTP 500 `batched-engine prefill forward pass failed`.
pod_heal_sccache() {
  local kiln_repo="${KILN_REPO:-/workspace/kiln}"
  pod_kill_kiln_serve
  cd "$kiln_repo"
  echo "Nuking target/ and rebuilding with SCCACHE_RECACHE=1 (forces fresh nvcc + re-upload)…"
  rm -rf target
  KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" SCCACHE_RECACHE=1 cargo build \
    --release --features cuda --bin kiln --bin kiln-bench --examples
}

pod_kill_kiln_serve() {
  pkill -f "kiln serve" 2>/dev/null || true
  sleep 2
}

# Start kiln serve --eval-mode and wait until /v1/health returns OK.
# Args: $1 = optional adapter-dir override (default /workspace/adapters)
pod_start_kiln_serve() {
  local adapter_dir="${1:-/workspace/adapters}"
  local model_path="${KILN_MODEL_PATH:-/workspace/Qwen3.5-4B}"
  local log="${KILN_SERVE_LOG:-/workspace/kiln-serve.log}"
  KILN_MODEL_PATH="$model_path" KILN_ADAPTER_DIR="$adapter_dir" \
    nohup /workspace/kiln/target/release/kiln serve --eval-mode > "$log" 2>&1 &
  for i in $(seq 1 60); do
    curl -sf http://localhost:8420/v1/health > /dev/null 2>&1 && {
      echo "kiln serve up after ${i}s"
      return 0
    }
    sleep 1
  done
  echo "FATAL: kiln serve did not come up in 60s; tail of $log:" >&2
  tail -30 "$log" >&2
  return 3
}

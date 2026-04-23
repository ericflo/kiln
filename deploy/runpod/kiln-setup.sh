#!/usr/bin/env bash
# kiln-setup — baked into ghcr.io/ericflo/kiln-runpod:latest.
#
# Configures sccache with B2 as a remote cache backend and (optionally)
# restores cached flash-attn artifacts. Run once per pod *before* the first
# `cargo build`. Designed to work with or without the kiln repo cloned yet:
#
#   kiln-setup                                 # just sets up sccache
#   kiln-setup --repo /workspace/kiln          # also restores flash-attn cache
#   kiln-setup --clone                         # clones kiln to /workspace/kiln then sets up
#
# Required env vars:
#   B2_APPLICATION_KEY_ID  — Backblaze B2 key ID
#   B2_APPLICATION_KEY     — Backblaze B2 application key
#
# Optional env vars:
#   KILN_REPO_DIR          — Path to kiln repo checkout (default: /workspace/kiln)
#   KILN_MODEL_ID          — Hugging Face model ID to download (default: Qwen/Qwen3.5-4B)
#   KILN_MODEL_DIR         — Local model dir (default: /workspace/qwen3.5-4b)
#
# Writes env exports to $KILN_REPO_DIR/.build-cache-env (if repo exists) and
# also to /root/.kiln-build-env for agents to source directly.

set -euo pipefail

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
KILN_MODEL_ID="${KILN_MODEL_ID:-Qwen/Qwen3.5-4B}"
KILN_MODEL_DIR="${KILN_MODEL_DIR:-/workspace/qwen3.5-4b}"
B2_BUCKET="clouderic"
B2_ENDPOINT="https://s3.us-west-002.backblazeb2.com"
B2_REGION="us-west-002"

# Argument parsing
CLONE_REPO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo)   KILN_REPO_DIR="$2"; shift 2 ;;
        --clone)  CLONE_REPO=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Detect architecture string (matches scripts/setup-build-cache.sh)
detect_arch() {
    local cpu os cuda_ver
    cpu="$(uname -m)"; os="linux"
    if command -v nvcc >/dev/null 2>&1; then
        cuda_ver="cuda$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        cuda_ver="cuda$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    else
        cuda_ver="nocuda"
    fi
    echo "${cpu}-${os}-${cuda_ver}"
}

ARCH="$(detect_arch)"
CACHE_PREFIX="build-cache/kiln/${ARCH}"

# Ensure CUDA on PATH
if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
[ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ] && export CUDA_HOME="/usr/local/cuda"

echo "=== kiln-setup ==="
echo "  arch:          ${ARCH}"
echo "  cache prefix:  ${CACHE_PREFIX}"
echo "  repo dir:      ${KILN_REPO_DIR}"

if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set" >&2
    exit 1
fi

# Optional: clone the kiln repo if the caller asked
if [ "$CLONE_REPO" = "1" ] && [ ! -d "${KILN_REPO_DIR}" ]; then
    echo "Cloning kiln into ${KILN_REPO_DIR}..."
    git clone https://github.com/ericflo/kiln.git "${KILN_REPO_DIR}"
fi

if [ ! -f "${KILN_MODEL_DIR}/config.json" ]; then
    echo "Downloading ${KILN_MODEL_ID} into ${KILN_MODEL_DIR}..."
    HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
        hf download "${KILN_MODEL_ID}" --local-dir "${KILN_MODEL_DIR}"
else
    echo "Model already present at ${KILN_MODEL_DIR}"
fi

# Configure sccache environment
export SCCACHE_BUCKET="${B2_BUCKET}"
export SCCACHE_ENDPOINT="${B2_ENDPOINT}"
export SCCACHE_REGION="${B2_REGION}"
export SCCACHE_S3_KEY_PREFIX="${CACHE_PREFIX}/sccache"
export SCCACHE_S3_USE_SSL="true"
export AWS_ACCESS_KEY_ID="${B2_APPLICATION_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_APPLICATION_KEY}"
export RUSTC_WRAPPER="sccache"

# Restart sccache server with the new env
sccache --stop-server >/dev/null 2>&1 || true
sccache --start-server
echo "sccache server started"
sccache --show-stats | head -8

# Restore flash-attn artifacts if a kiln checkout is present
if [ -d "${KILN_REPO_DIR}" ]; then
    b2 account authorize "${B2_APPLICATION_KEY_ID}" "${B2_APPLICATION_KEY}" >/dev/null 2>&1
    mkdir -p "${KILN_REPO_DIR}/target/release/build"
    FILE_COUNT=$(b2 ls -r "b2://${B2_BUCKET}/${CACHE_PREFIX}/artifacts/flash-attn/" 2>/dev/null | wc -l)
    if [ "${FILE_COUNT}" -gt 0 ]; then
        echo "Restoring ${FILE_COUNT} cached flash-attn artifact files..."
        b2 sync "b2://${B2_BUCKET}/${CACHE_PREFIX}/artifacts/flash-attn/" \
            "${KILN_REPO_DIR}/target/release/build/" --skipNewer 2>&1 | tail -5
    else
        echo "No cached flash-attn artifacts (first build will populate)"
    fi

    # Write the env file into the repo for tool discovery
    ENV_FILE="${KILN_REPO_DIR}/.build-cache-env"
    cat > "${ENV_FILE}" <<ENVEOF
export SCCACHE_BUCKET="${B2_BUCKET}"
export SCCACHE_ENDPOINT="${B2_ENDPOINT}"
export SCCACHE_REGION="${B2_REGION}"
export SCCACHE_S3_KEY_PREFIX="${CACHE_PREFIX}/sccache"
export SCCACHE_S3_USE_SSL=true
export AWS_ACCESS_KEY_ID="${B2_APPLICATION_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_APPLICATION_KEY}"
export RUSTC_WRAPPER=sccache
export PATH="/usr/local/cuda/bin:\${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ENVEOF
    echo "Wrote ${ENV_FILE}"
fi

# Always write a root-level env file so agents can source it without a clone
cat > /root/.kiln-build-env <<ENVEOF
export SCCACHE_BUCKET="${B2_BUCKET}"
export SCCACHE_ENDPOINT="${B2_ENDPOINT}"
export SCCACHE_REGION="${B2_REGION}"
export SCCACHE_S3_KEY_PREFIX="${CACHE_PREFIX}/sccache"
export SCCACHE_S3_USE_SSL=true
export AWS_ACCESS_KEY_ID="${B2_APPLICATION_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_APPLICATION_KEY}"
export RUSTC_WRAPPER=sccache
export PATH="/usr/local/cuda/bin:\${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ENVEOF

echo ""
echo "Build cache ready. Source the env file and build:"
echo "  source /root/.kiln-build-env"
echo "  cargo build --release --features cuda"

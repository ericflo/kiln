#!/usr/bin/env bash
# setup-build-cache.sh — Run on a RunPod pod before cargo build.
# Installs sccache, configures it for B2 S3-compat backend, and pulls
# cached flash-attn artifacts if available.
#
# Required env vars:
#   B2_APPLICATION_KEY_ID  — Backblaze B2 key ID
#   B2_APPLICATION_KEY     — Backblaze B2 application key
#
# Optional env vars:
#   KILN_REPO_DIR          — Path to kiln repo checkout (default: /workspace/kiln)
#   SCCACHE_VERSION        — sccache release tag (default: v0.9.1)

set -euo pipefail

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
SCCACHE_VERSION="${SCCACHE_VERSION:-v0.9.1}"
B2_BUCKET="clouderic"
B2_ENDPOINT="https://s3.us-west-002.backblazeb2.com"
B2_REGION="us-west-002"

# --- Detect architecture string ---
detect_arch() {
    local cpu
    cpu="$(uname -m)"
    local os="linux"
    local cuda_ver=""

    if command -v nvcc &>/dev/null; then
        cuda_ver="cuda$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    elif [ -f /usr/local/cuda/version.txt ]; then
        cuda_ver="cuda$(grep -oP '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -1)"
    elif [ -d /usr/local/cuda ]; then
        # Try nvcc from cuda path
        local nvcc_path="/usr/local/cuda/bin/nvcc"
        if [ -x "$nvcc_path" ]; then
            cuda_ver="cuda$($nvcc_path --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
        else
            cuda_ver="cuda-unknown"
        fi
    else
        cuda_ver="nocuda"
    fi

    echo "${cpu}-${os}-${cuda_ver}"
}

ARCH="$(detect_arch)"
CACHE_PREFIX="build-cache/kiln/${ARCH}"

# --- Ensure CUDA is on PATH ---
if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    echo "Added /usr/local/cuda/bin to PATH"
fi
if [ -d /usr/local/cuda ] && [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME="/usr/local/cuda"
    echo "Set CUDA_HOME=/usr/local/cuda"
fi

echo "=== Kiln Build Cache Setup ==="
echo "Architecture: ${ARCH}"
echo "Cache prefix: ${CACHE_PREFIX}"
echo "Repo dir:     ${KILN_REPO_DIR}"

# --- Validate required env vars ---
if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set"
    exit 1
fi

# --- Install sccache ---
install_sccache() {
    if command -v sccache &>/dev/null; then
        echo "sccache already installed: $(sccache --version)"
        return 0
    fi

    echo "Installing sccache ${SCCACHE_VERSION}..."
    local cpu
    cpu="$(uname -m)"
    local sccache_arch
    case "$cpu" in
        x86_64)  sccache_arch="x86_64-unknown-linux-musl" ;;
        aarch64) sccache_arch="aarch64-unknown-linux-musl" ;;
        *)       echo "ERROR: Unsupported CPU architecture: $cpu"; exit 1 ;;
    esac

    local url="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${sccache_arch}.tar.gz"
    local tmpdir
    tmpdir="$(mktemp -d)"

    echo "Downloading from ${url}..."
    curl -fsSL "$url" | tar xz -C "$tmpdir"
    mv "$tmpdir"/sccache-*/sccache /usr/local/bin/sccache
    chmod +x /usr/local/bin/sccache
    rm -rf "$tmpdir"

    echo "Installed: $(sccache --version)"
}

install_sccache

# --- Install b2 CLI ---
install_b2() {
    if command -v b2 &>/dev/null; then
        echo "b2 CLI already installed"
        return 0
    fi

    echo "Installing b2 CLI..."
    pip install --quiet --break-system-packages 'b2[full]' 2>/dev/null \
        || pip3 install --quiet --break-system-packages 'b2[full]' 2>/dev/null \
        || pip install --quiet 'b2[full]' 2>/dev/null \
        || pip3 install --quiet 'b2[full]' 2>/dev/null
    echo "Installed b2 CLI"
}

install_b2

# --- Configure sccache for B2 ---
echo "Configuring sccache for B2..."
export SCCACHE_BUCKET="${B2_BUCKET}"
export SCCACHE_ENDPOINT="${B2_ENDPOINT}"
export SCCACHE_REGION="${B2_REGION}"
export SCCACHE_S3_KEY_PREFIX="${CACHE_PREFIX}/sccache"
export SCCACHE_S3_USE_SSL="true"
export AWS_ACCESS_KEY_ID="${B2_APPLICATION_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_APPLICATION_KEY}"
export RUSTC_WRAPPER="sccache"

# Start sccache server
sccache --stop-server 2>/dev/null || true
sccache --start-server
echo "sccache server started"
sccache --show-stats

# --- Pull cached flash-attn artifacts ---
echo ""
echo "Checking for cached flash-attn artifacts..."
b2 account authorize "${B2_APPLICATION_KEY_ID}" "${B2_APPLICATION_KEY}" >/dev/null 2>&1

# Ensure target dir exists
mkdir -p "${KILN_REPO_DIR}/target/release/build"

# Try to pull cached flash-attn artifacts
FILE_COUNT=$(b2 ls -r "b2://${B2_BUCKET}/${CACHE_PREFIX}/artifacts/flash-attn/" 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -gt 0 ]; then
    echo "Found ${FILE_COUNT} cached flash-attn files, downloading..."
    b2 sync "b2://${B2_BUCKET}/${CACHE_PREFIX}/artifacts/flash-attn/" \
        "${KILN_REPO_DIR}/target/release/build/" \
        --skipNewer 2>&1 | tail -5
    RESTORED=$(find "${KILN_REPO_DIR}/target/release/build/" -maxdepth 1 -type d \( -name 'candle-flash-attn-*' -o -name 'kiln-flash-attn-*' \) -exec find {} -type f \; 2>/dev/null | wc -l)
    echo "Flash-attn artifacts restored (${RESTORED} files)"
else
    echo "No cached flash-attn artifacts found (first build will populate)"
fi

# --- Print env export commands for the caller ---
echo ""
echo "=== Build cache ready ==="
echo ""
echo "To use in your current shell, run:"
echo "  export SCCACHE_BUCKET=${B2_BUCKET}"
echo "  export SCCACHE_ENDPOINT=${B2_ENDPOINT}"
echo "  export SCCACHE_REGION=${B2_REGION}"
echo "  export SCCACHE_S3_KEY_PREFIX=${CACHE_PREFIX}/sccache"
echo "  export SCCACHE_S3_USE_SSL=true"
echo "  export AWS_ACCESS_KEY_ID=${B2_APPLICATION_KEY_ID}"
echo "  export AWS_SECRET_ACCESS_KEY=${B2_APPLICATION_KEY}"
echo "  export RUSTC_WRAPPER=sccache"
echo ""
echo "Then run: cargo build --release --features flash-attn"

# Write env to a sourceable file
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
echo "Env vars also written to ${ENV_FILE} — source it with: source ${ENV_FILE}"

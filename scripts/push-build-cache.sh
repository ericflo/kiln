#!/usr/bin/env bash
# push-build-cache.sh — Run after a successful cargo build to push
# compiled artifacts to B2 for future pods.
#
# Required env vars:
#   B2_APPLICATION_KEY_ID  — Backblaze B2 key ID
#   B2_APPLICATION_KEY     — Backblaze B2 application key
#
# Optional env vars:
#   KILN_REPO_DIR          — Path to kiln repo checkout (default: /workspace/kiln)

set -euo pipefail

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
B2_BUCKET="clouderic"

# --- Detect architecture string (same logic as setup script) ---
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

echo "=== Push Kiln Build Cache ==="
echo "Architecture: ${ARCH}"
echo "Cache prefix: ${CACHE_PREFIX}"

# --- Validate ---
if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set"
    exit 1
fi

BUILD_DIR="${KILN_REPO_DIR}/target/release/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: No build directory at ${BUILD_DIR} — run cargo build first"
    exit 1
fi

# --- Authorize b2 ---
echo "Authorizing B2..."
b2 account authorize "${B2_APPLICATION_KEY_ID}" "${B2_APPLICATION_KEY}" >/dev/null 2>&1

# --- Push flash-attn build artifacts ---
echo ""
echo "Looking for flash-attn build artifacts..."
# Support both candle-flash-attn (external dep) and kiln-flash-attn (vendored)
FLASH_DIRS=$(find "${BUILD_DIR}" -maxdepth 1 -type d \( -name 'candle-flash-attn-*' -o -name 'kiln-flash-attn-*' \) 2>/dev/null || true)

if [ -z "$FLASH_DIRS" ]; then
    echo "No flash-attn build directories found. Skipping flash-attn cache."
else
    echo "Found flash-attn directories:"
    echo "$FLASH_DIRS"
    echo ""
    echo "Uploading flash-attn artifacts to B2..."

    for dir in $FLASH_DIRS; do
        dirname="$(basename "$dir")"
        echo "  Syncing ${dirname}..."
        b2 sync "${dir}/" "b2://${B2_BUCKET}/${CACHE_PREFIX}/artifacts/flash-attn/${dirname}/" \
            --skipNewer 2>&1 | tail -3
    done
    echo "Flash-attn artifacts pushed"
fi

# --- Push key release binaries ---
RELEASE_DIR="${KILN_REPO_DIR}/target/release"
if [ -f "${RELEASE_DIR}/kiln" ]; then
    echo ""
    echo "Uploading kiln binary to B2..."
    b2 file upload "${B2_BUCKET}" "${RELEASE_DIR}/kiln" \
        "${CACHE_PREFIX}/artifacts/target-release/kiln" 2>&1 | tail -3
    echo "Binary uploaded"
fi

# --- Show sccache stats ---
echo ""
echo "=== sccache stats ==="
if command -v sccache &>/dev/null; then
    sccache --show-stats
else
    echo "(sccache not available — stats skipped)"
fi

echo ""
echo "=== Cache push complete ==="
echo "Artifacts stored at: b2://${B2_BUCKET}/${CACHE_PREFIX}/"

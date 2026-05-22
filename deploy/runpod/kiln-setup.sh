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
#   kiln-setup --per-sha-cache                 # isolate sccache namespace per kiln SHA
#                                              # (slower first build, safer cache hygiene;
#                                              # see issue #1066)
#
# Required env vars:
#   B2_APPLICATION_KEY_ID  — Backblaze B2 key ID
#   B2_APPLICATION_KEY     — Backblaze B2 application key
#
# Optional env vars:
#   KILN_REPO_DIR          — Path to kiln repo checkout (default: /workspace/kiln)
#   KILN_MODEL_ID          — Hugging Face model ID to download (default: Qwen/Qwen3.5-4B)
#   KILN_MODEL_DIR         — Local model dir (default: /workspace/Qwen3.5-4B)
#   KILN_SCCACHE_SALT      — Extra string appended to SCCACHE_S3_KEY_PREFIX, e.g.
#                            a content hash or git SHA. Overrides --per-sha-cache
#                            when both are set.
#
# Writes env exports to $KILN_REPO_DIR/.build-cache-env (if repo exists) and
# also to /root/.kiln-build-env for agents to source directly.

set -euo pipefail

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
KILN_MODEL_ID="${KILN_MODEL_ID:-Qwen/Qwen3.5-4B}"
KILN_MODEL_DIR="${KILN_MODEL_DIR:-/workspace/Qwen3.5-4B}"
B2_BUCKET="clouderic"
B2_ENDPOINT="https://s3.us-west-002.backblazeb2.com"
B2_REGION="us-west-002"

# Argument parsing
CLONE_REPO=0
PER_SHA_CACHE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo)            KILN_REPO_DIR="$2"; shift 2 ;;
        --clone)           CLONE_REPO=1; shift ;;
        --per-sha-cache)   PER_SHA_CACHE=1; shift ;;
        -h|--help)
            sed -n '2,29p' "$0"
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

if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set" >&2
    exit 1
fi

# Optional: clone the kiln repo if the caller asked
if [ "$CLONE_REPO" = "1" ] && [ ! -d "${KILN_REPO_DIR}" ]; then
    echo "Cloning kiln into ${KILN_REPO_DIR}..."
    git clone https://github.com/ericflo/kiln.git "${KILN_REPO_DIR}"
fi

# Optional cache salt — isolates the sccache namespace per kiln commit (or
# per arbitrary caller-supplied string). Issue #1066 documented a case where
# a corrupted .o object cached under the shared namespace took down every
# fresh pod until SCCACHE_RECACHE=1 overwrote it. A per-SHA namespace bounds
# the blast radius: a bad object only haunts pods on the same commit.
#
# Resolution order: explicit KILN_SCCACHE_SALT > --per-sha-cache > none.
# Resolved AFTER any --clone so a freshly-cloned repo can supply the SHA.
CACHE_SALT="${KILN_SCCACHE_SALT:-}"
if [ -z "${CACHE_SALT}" ] && [ "${PER_SHA_CACHE}" = "1" ]; then
    if [ -d "${KILN_REPO_DIR}/.git" ]; then
        CACHE_SALT="$(git -C "${KILN_REPO_DIR}" rev-parse --short=12 HEAD 2>/dev/null || true)"
    fi
    if [ -z "${CACHE_SALT}" ]; then
        echo "WARN: --per-sha-cache requested but ${KILN_REPO_DIR} is not a git checkout;" >&2
        echo "      falling back to the shared cache namespace." >&2
    fi
fi
if [ -n "${CACHE_SALT}" ]; then
    # Strip anything that isn't S3-safe (lowercase alnum, dash, dot).
    CACHE_SALT="$(printf '%s' "${CACHE_SALT}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.-' '-' | sed 's/-\+/-/g; s/^-//; s/-$//')"
    CACHE_PREFIX="${CACHE_PREFIX}/sha-${CACHE_SALT}"
fi

# Auto-detect KILN_CUDA_ARCHS from the visible GPU(s) if the caller didn't
# pin it. The default in build.rs is "80;86;89;90" which triples compile
# time AND produces multi-arch fat binaries — a known sccache-with-nvcc
# landmine (issue #1066 suspect-list, item 1 in the issue diagnosis). On a
# single-GPU pod the right value is just the local compute capability.
if [ -z "${KILN_CUDA_ARCHS:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk '{ gsub(/\./, ""); print }' | sort -u | paste -sd';' -)"
    if [ -n "${DETECTED_ARCHS}" ]; then
        export KILN_CUDA_ARCHS="${DETECTED_ARCHS}"
    fi
fi

echo "=== kiln-setup ==="
echo "  arch:          ${ARCH}"
echo "  cache prefix:  ${CACHE_PREFIX}"
echo "  cache salt:    ${CACHE_SALT:-<none>}"
echo "  cuda archs:    ${KILN_CUDA_ARCHS:-<unset, build.rs default>}"
echo "  repo dir:      ${KILN_REPO_DIR}"

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

echo ""
echo "=== pod resources ==="
free -h || true
echo "cpus: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
echo ""

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
# Cap build parallelism: nvcc -O3 can use 4-8GB per TU, and 5 CUDA kernel
# crates × 24 default cargo jobs × multi-arch was OOM-killing sshd.
export CARGO_BUILD_JOBS=4
export NVCC_THREADS=1
# Pin nvcc to the detected GPU arch — keeps the cache key stable across
# shells and avoids the multi-arch fat-binary path (issue #1066 hygiene).
# Override by setting KILN_CUDA_ARCHS before sourcing this file.
${KILN_CUDA_ARCHS:+export KILN_CUDA_ARCHS=\"${KILN_CUDA_ARCHS}\"}
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
# Cap build parallelism: nvcc -O3 can use 4-8GB per TU, and 5 CUDA kernel
# crates × 24 default cargo jobs × multi-arch was OOM-killing sshd.
export CARGO_BUILD_JOBS=4
export NVCC_THREADS=1
# Pin nvcc to the detected GPU arch — keeps the cache key stable across
# shells and avoids the multi-arch fat-binary path (issue #1066 hygiene).
# Override by setting KILN_CUDA_ARCHS before sourcing this file.
${KILN_CUDA_ARCHS:+export KILN_CUDA_ARCHS=\"${KILN_CUDA_ARCHS}\"}
ENVEOF

# Postmortem helper — run after any build wedge / failure to capture state
cat > /root/kiln-postmortem.sh <<'POSTEOF'
#!/usr/bin/env bash
echo "=== free -h ==="; free -h
echo "=== dmesg tail (OOM?) ==="; dmesg 2>/dev/null | tail -200
echo "=== top procs ==="; ps auxf --sort=-%mem | head -30
echo "=== disk ==="; df -h /workspace /tmp 2>/dev/null
POSTEOF
chmod +x /root/kiln-postmortem.sh

echo ""
echo "Build cache ready. Source the env file and build:"
echo "  source /root/.kiln-build-env"
echo "  cargo build --release --features cuda"
echo ""
echo "After the build completes, run a smoke check to confirm every fused"
echo "CUDA kernel works end-to-end (issue #1066 prevention — see Troubleshooting):"
echo "  kiln-smoke-check"
echo ""
echo "If a build wedges or sshd dies mid-build, run:"
echo "  /root/kiln-postmortem.sh"
echo ""
echo "=== Troubleshooting ==="
echo "  Build succeeds but inference returns HTTP 500 with"
echo "  'kiln_<kernel> failed with status <N>'?"
echo "  → The remote sccache returned a corrupted object. Heal the cache:"
echo "       source /root/.kiln-build-env"
echo "       SCCACHE_RECACHE=1 cargo build --release --features cuda --bin kiln-bench"
echo "       kiln-smoke-check"
echo "    If a single kernel crate is implicated, scope the rebuild:"
echo "       SCCACHE_RECACHE=1 cargo build --release --features cuda -p kiln-gdn-kernel"
echo "    If corruption recurs on fresh pods, re-run setup with --per-sha-cache"
echo "    to isolate the sccache namespace per kiln commit."

#!/usr/bin/env bash
# Run one Cargo command without allowing compilation/linking to exhaust a host.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cargo-bounded.sh <cargo-subcommand> [args...]

Runs Cargo with one build job after checking Linux MemAvailable. A transient
systemd scope places Cargo and every compiler/linker child under one aggregate
memory ceiling with swap disabled. It also refuses to overlap another Cargo or
rustc process.

Overrides:
  KILN_CARGO_JOBS                 Build jobs (default: 1)
  KILN_CARGO_MIN_AVAILABLE_GIB    Preflight floor (default: 2/3 host RAM, min 8)
  KILN_CARGO_HOST_RESERVE_GIB     Memory kept outside each child (default: 1/4 host RAM, min 4)
  KILN_CARGO_MAX_MEMORY_GIB       Explicit aggregate ceiling (default: available minus reserve)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    [[ $# -gt 0 ]] && exit 0
    exit 2
fi

for tool in cargo awk ps systemd-run; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' is not available" >&2
        exit 2
    fi
done
if [[ ! -r /proc/meminfo ]]; then
    echo "error: bounded Cargo requires Linux /proc/meminfo" >&2
    exit 2
fi

active_builds="$(ps -C cargo,rustc -o stat=,pid=,args= | awk '$1 !~ /^Z/' || true)"
if [[ -n "$active_builds" ]]; then
    echo "error: refusing to overlap an existing cargo or rustc process" >&2
    printf '%s\n' "$active_builds" >&2
    exit 2
fi

read -r total_kib available_kib < <(
    awk '
        /^MemTotal:/ { total = $2 }
        /^MemAvailable:/ { available = $2 }
        END { print total, available }
    ' /proc/meminfo
)
if [[ -z "${total_kib:-}" || -z "${available_kib:-}" ]]; then
    echo "error: could not read MemTotal and MemAvailable from /proc/meminfo" >&2
    exit 2
fi

total_gib=$((total_kib / 1024 / 1024))
available_gib=$((available_kib / 1024 / 1024))
default_min_gib=$((total_gib * 2 / 3))
(( default_min_gib < 8 )) && default_min_gib=8
default_reserve_gib=$((total_gib / 4))
(( default_reserve_gib < 4 )) && default_reserve_gib=4

jobs="${KILN_CARGO_JOBS:-1}"
min_available_gib="${KILN_CARGO_MIN_AVAILABLE_GIB:-$default_min_gib}"
reserve_gib="${KILN_CARGO_HOST_RESERVE_GIB:-$default_reserve_gib}"
for pair in \
    "KILN_CARGO_JOBS:$jobs" \
    "KILN_CARGO_MIN_AVAILABLE_GIB:$min_available_gib" \
    "KILN_CARGO_HOST_RESERVE_GIB:$reserve_gib"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: $name must be a positive decimal integer, got '$value'" >&2
        exit 2
    fi
done

if (( available_gib < min_available_gib )); then
    echo "error: refusing Cargo with ${available_gib} GiB available; require at least ${min_available_gib} GiB" >&2
    exit 2
fi

default_limit_gib=$((available_gib - reserve_gib))
(( default_limit_gib < 4 )) && default_limit_gib=4
limit_gib="${KILN_CARGO_MAX_MEMORY_GIB:-$default_limit_gib}"
if [[ ! "$limit_gib" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: KILN_CARGO_MAX_MEMORY_GIB must be a positive decimal integer, got '$limit_gib'" >&2
    exit 2
fi
if (( limit_gib + reserve_gib > available_gib )); then
    echo "error: aggregate memory ceiling (${limit_gib} GiB) plus host reserve (${reserve_gib} GiB) exceeds available memory (${available_gib} GiB)" >&2
    exit 2
fi

export CARGO_BUILD_JOBS="$jobs"

echo "bounded-cargo: jobs=$jobs available=${available_gib}GiB reserve=${reserve_gib}GiB aggregate_limit=${limit_gib}GiB swap_limit=0" >&2
exec systemd-run \
    --user \
    --scope \
    --quiet \
    -p "MemoryMax=${limit_gib}G" \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    cargo "$@"

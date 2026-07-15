#!/usr/bin/env bash
# Run one Cargo command without allowing compilation/linking to exhaust a host.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cargo-bounded.sh <cargo-subcommand> [args...]

Runs Cargo with one build job after checking Linux MemAvailable. A transient
systemd scope, or a transient service for PID-namespaced callers, places Cargo
and every compiler/linker child under one aggregate memory ceiling with swap
disabled. It also refuses to overlap another Cargo or rustc process.

Overrides:
  CARGO                           Cargo executable/name (default: PATH, then ~/.cargo/bin/cargo)
  KILN_CARGO_JOBS                 Build jobs (default: 1)
  KILN_CARGO_MIN_AVAILABLE_GIB    Preflight floor (default: 2/3 host RAM, min 8)
  KILN_CARGO_HOST_RESERVE_GIB     Memory kept outside each child (default: 1/4 host RAM, min 4)
  KILN_CARGO_MAX_MEMORY_GIB       Explicit aggregate ceiling (default: available minus reserve)
  KILN_CARGO_EXECUTION_MODE       scope (default) or transient-service
  KILN_CARGO_PRIVATE_NETWORK      1 requires a private network in transient-service mode
  KILN_CARGO_ENVIRONMENT_POLICY   closed-source-build-v1 (transient-service default),
                                  closed-qualification-test-v1, or inherit
  KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS
                                  Hard transient-service deadline (default: 3600)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    [[ $# -gt 0 ]] && exit 0
    exit 2
fi

execution_mode="${KILN_CARGO_EXECUTION_MODE:-scope}"
if [[ "$execution_mode" != "scope" && "$execution_mode" != "transient-service" ]]; then
    echo "error: KILN_CARGO_EXECUTION_MODE must be scope or transient-service, got '$execution_mode'" >&2
    exit 2
fi

for tool in awk ps systemd-run; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' is not available" >&2
        exit 2
    fi
done
if [[ "$execution_mode" == "transient-service" ]] && ! command -v systemctl >/dev/null 2>&1; then
    echo "error: required tool 'systemctl' is not available" >&2
    exit 2
fi

cargo_executable="${CARGO:-}"
if [[ -n "$cargo_executable" ]]; then
    if [[ "$cargo_executable" == */* ]]; then
        if [[ ! -x "$cargo_executable" ]]; then
            echo "error: CARGO='$cargo_executable' is not executable" >&2
            exit 2
        fi
    elif ! cargo_executable="$(command -v "$cargo_executable")"; then
        echo "error: CARGO='${CARGO}' is not available" >&2
        exit 2
    fi
elif cargo_executable="$(command -v cargo 2>/dev/null)"; then
    :
elif [[ -n "${HOME:-}" && -x "$HOME/.cargo/bin/cargo" ]]; then
    cargo_executable="$HOME/.cargo/bin/cargo"
else
    echo "error: required tool 'cargo' is not available" >&2
    exit 2
fi
if [[ "$cargo_executable" -ef "$0" ]]; then
    echo "error: CARGO cannot point to scripts/cargo-bounded.sh itself" >&2
    exit 2
fi
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

private_network="${KILN_CARGO_PRIVATE_NETWORK:-0}"
if [[ "$private_network" != "0" && "$private_network" != "1" ]]; then
    echo "error: KILN_CARGO_PRIVATE_NETWORK must be 0 or 1, got '$private_network'" >&2
    exit 2
fi
if [[ "$execution_mode" == "scope" && "$private_network" != "0" ]]; then
    echo "error: KILN_CARGO_PRIVATE_NETWORK=1 requires transient-service mode" >&2
    exit 2
fi
environment_policy="${KILN_CARGO_ENVIRONMENT_POLICY:-}"
if [[ -z "$environment_policy" ]]; then
    if [[ "$execution_mode" == "transient-service" ]]; then
        environment_policy="closed-source-build-v1"
    else
        environment_policy="inherit"
    fi
fi
if [[ "$environment_policy" != "closed-source-build-v1" && "$environment_policy" != "closed-qualification-test-v1" && "$environment_policy" != "inherit" ]]; then
    echo "error: KILN_CARGO_ENVIRONMENT_POLICY must be closed-source-build-v1, closed-qualification-test-v1, or inherit, got '$environment_policy'" >&2
    exit 2
fi
service_runtime_max_seconds="${KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS:-3600}"
if [[ ! "$service_runtime_max_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS must be a positive decimal integer, got '$service_runtime_max_seconds'" >&2
    exit 2
fi

echo "bounded-cargo: mode=$execution_mode jobs=$jobs available=${available_gib}GiB reserve=${reserve_gib}GiB aggregate_limit=${limit_gib}GiB swap_limit=0 private_network=$private_network environment_policy=$environment_policy" >&2
if [[ "$execution_mode" == "scope" ]]; then
    exec systemd-run \
        --user \
        --scope \
        --quiet \
        -p "MemoryMax=${limit_gib}G" \
        -p MemorySwapMax=0 \
        -p OOMPolicy=kill \
        "$cargo_executable" "$@"
fi

# A process in a bubblewrap PID namespace cannot be attached to the host user
# manager as a scope. Qualification uses a transient service instead: Cargo is
# still in one bounded cgroup, while PrivateNetwork independently preserves the
# offline build boundary. The explicit unit and EXIT trap make a normal timeout
# stop the complete compiler/linker tree; RuntimeMaxSec bounds hard-kill cases.
read -r service_uuid < /proc/sys/kernel/random/uuid
service_unit="kiln-cargo-bounded-${service_uuid//-/}.service"
cleanup_service() {
    systemctl --user stop "$service_unit" >/dev/null 2>&1 || true
}
trap cleanup_service EXIT

environment_args=()
if [[ "$environment_policy" == "inherit" ]]; then
    while IFS= read -r name; do
        if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            environment_args+=("--setenv=$name")
        fi
    done < <(compgen -e)
else
    # This versioned policy is shared with the source-bound qualification
    # driver. It deliberately excludes ambient compiler flags, target paths,
    # credentials, and API tokens so they cannot change or enter the build.
    closed_source_build_environment=(
        CARGO_BUILD_JOBS
        CARGO_HOME
        CARGO_NET_OFFLINE
        DBUS_SESSION_BUS_ADDRESS
        HOME
        KILN_ROCM_ARCHS
        LANG
        LC_ALL
        LC_CTYPE
        LOGNAME
        PATH
        ROCM_PATH
        RUSTUP_HOME
        SHELL
        TMPDIR
        USER
        XDG_RUNTIME_DIR
    )
    if [[ "$environment_policy" == "closed-qualification-test-v1" ]]; then
        # Runner-owned test controls derived from the committed qualification
        # manifest. Product/runtime KILN_* settings remain excluded.
        closed_source_build_environment+=(
            KILN_QUALIFICATION
            KILN_QUALIFICATION_HF_LOGITS_PATH
            KILN_QUALIFICATION_MODEL_PATH
        )
    fi
    for name in "${closed_source_build_environment[@]}"; do
        if [[ -v "$name" ]]; then
            environment_args+=("--setenv=$name")
        fi
    done
fi

private_network_property="PrivateNetwork=no"
if [[ "$private_network" == "1" ]]; then
    private_network_property="PrivateNetwork=yes"
fi

systemd-run \
    --user \
    --wait \
    --collect \
    --pipe \
    --quiet \
    --same-dir \
    --unit "$service_unit" \
    "${environment_args[@]}" \
    -p Type=exec \
    -p "MemoryMax=${limit_gib}G" \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    -p KillMode=control-group \
    -p SendSIGKILL=yes \
    -p TimeoutStopSec=15s \
    -p "RuntimeMaxSec=${service_runtime_max_seconds}s" \
    -p "$private_network_property" \
    "$cargo_executable" "$@"

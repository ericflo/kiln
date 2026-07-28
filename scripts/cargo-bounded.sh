#!/usr/bin/env bash
# Run one Cargo command without allowing compilation/linking to exhaust a host.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cargo-bounded.sh <cargo-subcommand> [args...]

Runs Cargo with one build job after checking host memory availability. A
transient systemd scope/service, or a WSL2 delegated cgroup for contained
qualification, places Cargo and every compiler/linker child under one aggregate
memory ceiling with swap disabled. On macOS, a qualification-owned sandbox,
session, process group, wall-clock deadline, and cleanup boundary must already
exist; the wrapper verifies and inherits that boundary. It also refuses to
overlap another Cargo or rustc process.

Overrides:
  CARGO                           Cargo executable/name (default: PATH, then ~/.cargo/bin/cargo)
  KILN_CARGO_JOBS                 Build jobs (default: 1)
  KILN_CARGO_CPU_QUOTA_PERCENT    Aggregate CPU quota; 100 is one logical CPU (default: disabled)
  KILN_CARGO_MIN_AVAILABLE_GIB    Preflight floor (default: 2/3 host RAM, min 8)
  KILN_CARGO_HOST_RESERVE_GIB     Memory kept outside each child (default: 1/4 host RAM, min 4)
  KILN_CARGO_MAX_MEMORY_GIB       Explicit aggregate ceiling (default: available minus reserve)
  KILN_CARGO_EXECUTION_MODE       scope (default), transient-service,
                                  delegated-cgroup, or macos-contained
  KILN_CARGO_PRIVATE_NETWORK      1 requires private-network containment in service/cgroup mode
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
if [[ "$execution_mode" != "scope" \
    && "$execution_mode" != "transient-service" \
    && "$execution_mode" != "delegated-cgroup" \
    && "$execution_mode" != "macos-contained" ]]; then
    echo "error: KILN_CARGO_EXECUTION_MODE must be scope, transient-service, delegated-cgroup, or macos-contained, got '$execution_mode'" >&2
    exit 2
fi

required_tools=(awk)
if [[ "$execution_mode" == "scope" || "$execution_mode" == "transient-service" ]]; then
    required_tools+=(ps sleep)
    required_tools+=(systemctl systemd-run)
elif [[ "$execution_mode" == "delegated-cgroup" ]]; then
    required_tools+=(ps python3 sleep)
else
    required_tools+=(memory_pressure mkdir python3 rmdir sysctl)
fi
for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' is not available" >&2
        exit 2
    fi
done

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
if [[ "$execution_mode" == "macos-contained" ]]; then
    active_builds=""
else
    if [[ ! -r /proc/meminfo ]]; then
        echo "error: bounded Cargo requires Linux /proc/meminfo" >&2
        exit 2
    fi
    active_builds="$(ps -C cargo,rustc -o stat=,pid=,args= | awk '$1 !~ /^Z/' || true)"
fi
if [[ -n "$active_builds" ]]; then
    echo "error: refusing to overlap an existing cargo or rustc process" >&2
    printf '%s\n' "$active_builds" >&2
    exit 2
fi

if [[ "$execution_mode" == "macos-contained" ]]; then
    total_bytes="$(sysctl -n hw.memsize)"
    free_percent="$(
        memory_pressure \
            | awk -F': |%' '/System-wide memory free percentage:/ { print $2 }'
    )"
    if [[ ! "$total_bytes" =~ ^[1-9][0-9]*$ ]] \
        || [[ ! "$free_percent" =~ ^[0-9]+$ ]] \
        || (( free_percent > 100 )); then
        echo "error: could not read macOS total memory and free percentage" >&2
        exit 2
    fi
    total_kib=$((total_bytes / 1024))
    available_kib=$((total_kib * free_percent / 100))
else
    read -r total_kib available_kib < <(
        awk '
            /^MemTotal:/ { total = $2 }
            /^MemAvailable:/ { available = $2 }
            END { print total, available }
        ' /proc/meminfo
    )
fi
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
cpu_quota_percent="${KILN_CARGO_CPU_QUOTA_PERCENT:-}"
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
if [[ -n "$cpu_quota_percent" ]]; then
    if [[ ! "$cpu_quota_percent" =~ ^[1-9][0-9]*$ ]] \
        || (( cpu_quota_percent > 10000 )); then
        echo "error: KILN_CARGO_CPU_QUOTA_PERCENT must be in 1..=10000, got '$cpu_quota_percent'" >&2
        exit 2
    fi
fi

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
    echo "error: KILN_CARGO_PRIVATE_NETWORK=1 requires a contained execution mode" >&2
    exit 2
fi
if [[ "$execution_mode" == "delegated-cgroup" && "$private_network" != "1" ]]; then
    echo "error: delegated-cgroup mode requires KILN_CARGO_PRIVATE_NETWORK=1" >&2
    exit 2
fi
if [[ "$execution_mode" == "macos-contained" && "$private_network" != "1" ]]; then
    echo "error: macos-contained mode requires KILN_CARGO_PRIVATE_NETWORK=1" >&2
    exit 2
fi
environment_policy="${KILN_CARGO_ENVIRONMENT_POLICY:-}"
if [[ -z "$environment_policy" ]]; then
    if [[ "$execution_mode" == "transient-service" \
        || "$execution_mode" == "delegated-cgroup" \
        || "$execution_mode" == "macos-contained" ]]; then
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

cpu_quota_args=()
cpu_quota_summary="disabled"
if [[ -n "$cpu_quota_percent" ]]; then
    cpu_quota_args=(-p "CPUQuota=${cpu_quota_percent}%")
    cpu_quota_summary="${cpu_quota_percent}%"
fi
memory_summary="aggregate_limit=${limit_gib}GiB"
if [[ "$execution_mode" == "delegated-cgroup" ]] \
    && [[ "${KILN_WSL2_SCOPE_MEMORY_MAX_BYTES:-}" == "0" ]]; then
    memory_summary="aggregate_limit=unbounded admission_budget=${limit_gib}GiB"
elif [[ "$execution_mode" == "macos-contained" ]]; then
    memory_summary="aggregate_limit=unavailable admission_budget=${limit_gib}GiB"
fi
swap_summary="swap_limit=0"
if [[ "$execution_mode" == "macos-contained" ]]; then
    swap_summary="swap_limit=unavailable"
fi
echo "bounded-cargo: mode=$execution_mode jobs=$jobs cpu_quota=$cpu_quota_summary available=${available_gib}GiB reserve=${reserve_gib}GiB $memory_summary $swap_summary private_network=$private_network environment_policy=$environment_policy" >&2
if [[ "$execution_mode" == "scope" || "$execution_mode" == "transient-service" ]]; then
    read -r bounded_uuid < /proc/sys/kernel/random/uuid
    if [[ "$execution_mode" == "scope" ]]; then
        bounded_unit="kiln-cargo-bounded-${bounded_uuid//-/}.scope"
    else
        bounded_unit="kiln-cargo-bounded-${bounded_uuid//-/}.service"
    fi
else
    bounded_unit=""
fi

if [[ "$execution_mode" == "macos-contained" ]]; then
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "error: macos-contained mode requires Darwin" >&2
        exit 2
    fi
    if [[ "${KILN_QUALIFICATION_NETWORK_ISOLATION:-}" != "macos-sandbox-loopback-only-v1" ]]; then
        echo "error: macos-contained mode is not inside the required qualification sandbox" >&2
        exit 2
    fi
    if ! python3 - "$$" <<'PY'
import errno
import os
import select
import socket
import sys

owner = int(sys.argv[1])
if os.getppid() != owner or os.getpgrp() != owner or os.getsid(0) != owner:
    raise SystemExit("qualification wrapper does not own its session/process group")

listener = socket.socket()
client = socket.socket()
try:
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    client.settimeout(1.0)
    client.connect(listener.getsockname())
    accepted, _ = listener.accept()
    accepted.close()
finally:
    client.close()
    listener.close()

external = socket.socket()
try:
    external.setblocking(False)
    result = external.connect_ex(("192.0.2.1", 9))
    if result in {errno.EAGAIN, errno.EINPROGRESS, errno.EWOULDBLOCK}:
        _, writable, exceptional = select.select([], [external], [external], 1.0)
        if not writable and not exceptional:
            raise SystemExit("external connection did not settle")
        result = external.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
finally:
    external.close()
if result not in {errno.EACCES, errno.EPERM}:
    raise SystemExit(f"external connection returned {result}")
PY
    then
        echo "error: Cargo could not reverify the live macOS containment boundary" >&2
        exit 2
    fi
    macos_lock="${TMPDIR:-/tmp}/kiln-cargo-bounded-${UID}.lock"
    if ! mkdir "$macos_lock"; then
        echo "error: refusing to overlap another macOS bounded Cargo invocation; lock exists at $macos_lock" >&2
        exit 2
    fi
    cleanup_macos_lock() {
        rmdir "$macos_lock" >/dev/null 2>&1 || true
    }
    trap cleanup_macos_lock EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    closed_environment_names=(
        CARGO_BUILD_JOBS
        CARGO_HOME
        CARGO_NET_OFFLINE
        HOME
        LANG
        LC_ALL
        LC_CTYPE
        LOGNAME
        PATH
        RUSTUP_HOME
        SHELL
        TMPDIR
        USER
    )
    if [[ "$environment_policy" == "closed-qualification-test-v1" ]]; then
        closed_environment_names+=(
            KILN_QUALIFICATION
            KILN_QUALIFICATION_HF_LOGITS_PATH
            KILN_QUALIFICATION_MODEL_PATH
        )
    elif [[ "$environment_policy" == "inherit" ]]; then
        "$cargo_executable" "$@"
        exit $?
    fi
    closed_environment=()
    for name in "${closed_environment_names[@]}"; do
        if [[ ${!name+x} ]]; then
            closed_environment+=("$name=${!name}")
        fi
    done
    env -i "${closed_environment[@]}" "$cargo_executable" "$@"
    exit $?
fi

if [[ "$execution_mode" == "delegated-cgroup" ]]; then
    if [[ -n "$cpu_quota_percent" ]] && (( cpu_quota_percent > 100 )); then
        echo "error: delegated-cgroup CPU quota must be in 1..=100 percent" >&2
        exit 2
    fi
    pids_max="${KILN_CARGO_PIDS_MAX:-512}"
    if [[ ! "$pids_max" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: KILN_CARGO_PIDS_MAX must be a positive decimal integer" >&2
        exit 2
    fi
    if [[ "${KILN_WSL2_SCOPE_BOUNDARY:-}" != "systemd-user-scope-feedback-v1" ]]; then
        echo "error: delegated-cgroup mode is not inside the required WSL2 user scope" >&2
        exit 2
    fi
    scope_unit="${KILN_WSL2_SCOPE_UNIT:-}"
    if [[ ! "$scope_unit" =~ ^kiln-wsl-scope-[0-9a-f]{32}$ ]]; then
        echo "error: invalid or missing KILN_WSL2_SCOPE_UNIT" >&2
        exit 2
    fi
    scope_memory_max="${KILN_WSL2_SCOPE_MEMORY_MAX_BYTES:-}"
    scope_pids_max="${KILN_WSL2_SCOPE_PIDS_MAX:-}"
    scope_cpu_quota="${KILN_WSL2_SCOPE_CPU_QUOTA_PERCENT:-}"
    scope_host_uid="${KILN_WSL2_SCOPE_HOST_UID:-}"
    if [[ ! "$scope_memory_max" =~ ^(0|[1-9][0-9]*)$ ]] \
        || [[ ! "$scope_pids_max" =~ ^[1-9][0-9]*$ ]] \
        || [[ ! "$scope_host_uid" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: malformed WSL2 scope resource binding" >&2
        exit 2
    fi
    if [[ -n "$cpu_quota_percent" ]]; then
        if [[ "$scope_cpu_quota" != "$cpu_quota_percent" ]]; then
            echo "error: WSL2 scope CPU binding disagrees with Cargo" >&2
            exit 2
        fi
    elif [[ "$scope_cpu_quota" != "0" ]]; then
        echo "error: WSL2 scope applies an undeclared CPU quota" >&2
        exit 2
    fi
    current_cgroup="$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup)"
    expected_cgroup="/user.slice/user-${scope_host_uid}.slice/user@${scope_host_uid}.service/app.slice/${scope_unit}.scope"
    if [[ "$current_cgroup" != "$expected_cgroup" ]]; then
        echo "error: Cargo cgroup '$current_cgroup' does not match '$expected_cgroup'" >&2
        exit 2
    fi
    scope_cgroup="/sys/fs/cgroup$current_cgroup"
    IFS= read -r observed_memory_max < "$scope_cgroup/memory.max" || observed_memory_max=""
    IFS= read -r observed_swap_max < "$scope_cgroup/memory.swap.max" || observed_swap_max=""
    IFS= read -r observed_pids_max < "$scope_cgroup/pids.max" || observed_pids_max=""
    IFS= read -r observed_oom_group < "$scope_cgroup/memory.oom.group" || observed_oom_group=""
    expected_memory_max="$scope_memory_max"
    [[ "$scope_memory_max" == "0" ]] && expected_memory_max="max"
    if [[ "$observed_memory_max" != "$expected_memory_max" ]] \
        || { [[ "$scope_memory_max" != "0" ]] \
            && (( observed_memory_max > limit_gib * 1024 * 1024 * 1024 )); }; then
        echo "error: outer WSL2 scope memory.max '$observed_memory_max' exceeds or contradicts the Cargo ceiling" >&2
        exit 2
    fi
    if [[ "$observed_swap_max" != "0" ]] \
        || [[ "$observed_pids_max" != "$scope_pids_max" ]] \
        || (( observed_pids_max > pids_max )) \
        || [[ "$observed_oom_group" != "1" ]]; then
        echo "error: outer WSL2 scope swap/PID/OOM limits do not match the required boundary" >&2
        exit 2
    fi
    script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    wsl_platform_helper="$script_directory/qualification/wsl_platform.py"
    if [[ ! -f "$wsl_platform_helper" ]]; then
        echo "error: WSL2 platform helper is missing: $wsl_platform_helper" >&2
        exit 2
    fi
    if ! python3 - "$wsl_platform_helper" <<'PY'
import importlib.util
import os
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("kiln_wsl_platform", path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load WSL2 platform helper")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_contained_case(os.environ.get(module.NETWORK_ISOLATION_ENV))
PY
    then
        echo "error: Cargo could not reverify the live WSL2 containment boundary" >&2
        exit 2
    fi
    closed_environment_names=(
        CARGO_BUILD_JOBS
        CARGO_HOME
        CARGO_NET_OFFLINE
        CUDARC_CUDA_VERSION
        HOME
        KILN_CUDA_ARCHS
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
    )
    if [[ "$environment_policy" == "closed-qualification-test-v1" ]]; then
        closed_environment_names+=(
            KILN_QUALIFICATION
            KILN_QUALIFICATION_HF_LOGITS_PATH
            KILN_QUALIFICATION_MODEL_PATH
        )
    elif [[ "$environment_policy" == "inherit" ]]; then
        exec "$cargo_executable" "$@"
    fi
    closed_environment=()
    for name in "${closed_environment_names[@]}"; do
        if [[ ${!name+x} ]]; then
            closed_environment+=("$name=${!name}")
        fi
    done
    exec env -i "${closed_environment[@]}" "$cargo_executable" "$@"
fi

bounded_runner_pid=""
cleanup_unit() {
    systemctl --user stop "$bounded_unit" >/dev/null 2>&1 || true
    if [[ -n "$bounded_runner_pid" ]]; then
        kill -TERM "$bounded_runner_pid" >/dev/null 2>&1 || true
        wait "$bounded_runner_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup_unit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_bounded_runner() {
    if wait "$bounded_runner_pid"; then
        return 0
    else
        return $?
    fi
}

if [[ "$execution_mode" == "scope" ]]; then
    systemd-run \
        --user \
        --scope \
        --quiet \
        --unit "$bounded_unit" \
        "${cpu_quota_args[@]}" \
        -p "MemoryMax=${limit_gib}G" \
        -p MemorySwapMax=0 \
        -p OOMPolicy=kill \
        "$cargo_executable" "$@" &
    bounded_runner_pid=$!
    if wait_for_bounded_runner; then
        exit 0
    else
        exit $?
    fi
fi

# A process in a private PID namespace cannot be attached to the host user
# manager as a scope. Qualification uses a transient service instead: Cargo is
# still in one bounded cgroup, while the same fail-closed namespace helper used
# by qualification supplies loopback-only networking and native executable
# containment. The named unit and EXIT trap make client cancellation stop the
# complete compiler/linker tree; RuntimeMaxSec bounds hard-kill cases.

environment_args=()
if [[ "$environment_policy" == "inherit" ]]; then
    while IFS= read -r name; do
        if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            environment_args+=("--setenv=$name=${!name}")
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
        CUDARC_CUDA_VERSION
        DBUS_SESSION_BUS_ADDRESS
        HOME
        KILN_CUDA_ARCHS
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
        if [[ ${!name+x} ]]; then
            environment_args+=("--setenv=$name=${!name}")
        fi
    done
fi

private_network_prefix=()
if [[ "$private_network" == "1" ]]; then
    for tool in unshare python3 ip; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "error: private-network transient service requires '$tool'" >&2
            exit 2
        fi
    done
    script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    namespace_helper="$script_directory/qualification/linux_namespace_exec.py"
    if [[ -L "$namespace_helper" || ! -f "$namespace_helper" ]]; then
        echo "error: private-network namespace helper is not a regular file: $namespace_helper" >&2
        exit 2
    fi
    private_network_prefix=(
        "$(command -v unshare)"
        --user
        --map-root-user
        --net
        --pid
        --fork
        --kill-child=SIGKILL
        --mount
        --mount-proc=/proc
        "$(command -v python3)"
        "$namespace_helper"
        --
    )
fi

systemd-run \
    --user \
    --wait \
    --collect \
    --pipe \
    --quiet \
    --same-dir \
    --unit "$bounded_unit" \
    "${environment_args[@]}" \
    "${cpu_quota_args[@]}" \
    -p Type=exec \
    -p "MemoryMax=${limit_gib}G" \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    -p KillMode=control-group \
    -p SendSIGKILL=yes \
    -p TimeoutStopSec=15s \
    -p "RuntimeMaxSec=${service_runtime_max_seconds}s" \
    -p PrivateNetwork=no \
    "${private_network_prefix[@]}" \
    "$cargo_executable" "$@" &
bounded_runner_pid=$!
if wait_for_bounded_runner; then
    exit 0
else
    exit $?
fi

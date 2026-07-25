#!/usr/bin/env bash
# Run one Cargo command without allowing compilation/linking to exhaust a host.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cargo-bounded.sh [--host-thermal-policy PATH] <cargo-subcommand> [args...]

Runs Cargo with one build job after checking Linux MemAvailable. A transient
systemd scope/service, or a WSL2 delegated cgroup for contained qualification,
places Cargo and every compiler/linker child under one aggregate memory ceiling
with swap disabled. It also refuses to overlap another Cargo or rustc process.

  --host-thermal-policy PATH      Reuse a content-hashed hard-limit-only
                                  kiln.host-thermal-policy.v1 document.

Overrides:
  CARGO                           Cargo executable/name (default: PATH, then ~/.cargo/bin/cargo)
  KILN_CARGO_JOBS                 Build jobs (default: 1)
  KILN_CARGO_CPU_QUOTA_PERCENT    Aggregate CPU quota; 100 is one logical CPU (default: disabled)
  KILN_CARGO_MIN_AVAILABLE_GIB    Preflight floor (default: 2/3 host RAM, min 8)
  KILN_CARGO_HOST_RESERVE_GIB     Memory kept outside each child (default: 1/4 host RAM, min 4)
  KILN_CARGO_MAX_MEMORY_GIB       Explicit aggregate ceiling (default: available minus reserve)
  KILN_CARGO_EXECUTION_MODE       scope (default), transient-service, or delegated-cgroup
  KILN_CARGO_PRIVATE_NETWORK      1 requires private-network containment in service/cgroup mode
  KILN_CARGO_ENVIRONMENT_POLICY   closed-source-build-v1 (transient-service default),
                                  closed-qualification-test-v1, or inherit
  KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS
                                  Hard transient-service deadline (default: 3600)
  KILN_CARGO_HOST_THERMAL_SENSOR_NAME
  KILN_CARGO_HOST_THERMAL_SENSOR_LABEL
  KILN_CARGO_HOST_THERMAL_LIMIT_MILLICELSIUS
  KILN_CARGO_HOST_THERMAL_POLL_MILLISECONDS
                                  Package-temperature guard for scopes and services. All four
                                  fields must be set together. If omitted, a unique
                                  k10temp/Tctl sensor enables a 97000 mC, 250 ms guard.
  KILN_WSL2_THERMAL_POLICY_SHA256
                                  Required outer thermal supervisor binding in
                                  delegated-cgroup mode.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    [[ $# -gt 0 ]] && exit 0
    exit 2
fi

host_thermal_policy=""
case "${1:-}" in
    --host-thermal-policy)
        if [[ $# -lt 3 || -z "${2:-}" ]]; then
            echo "error: --host-thermal-policy requires a path before the Cargo subcommand" >&2
            exit 2
        fi
        host_thermal_policy="$2"
        shift 2
        ;;
    --host-thermal-policy=*)
        host_thermal_policy="${1#*=}"
        if [[ -z "$host_thermal_policy" ]]; then
            echo "error: --host-thermal-policy requires a nonempty path" >&2
            exit 2
        fi
        shift
        ;;
esac
if [[ $# -eq 0 ]]; then
    echo "error: a Cargo subcommand is required" >&2
    exit 2
fi

execution_mode="${KILN_CARGO_EXECUTION_MODE:-scope}"
if [[ "$execution_mode" != "scope" \
    && "$execution_mode" != "transient-service" \
    && "$execution_mode" != "delegated-cgroup" ]]; then
    echo "error: KILN_CARGO_EXECUTION_MODE must be scope, transient-service, or delegated-cgroup, got '$execution_mode'" >&2
    exit 2
fi

required_tools=(awk ps sleep)
if [[ "$execution_mode" != "delegated-cgroup" ]]; then
    required_tools+=(systemctl systemd-run)
else
    required_tools+=(python3)
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
    echo "error: KILN_CARGO_PRIVATE_NETWORK=1 requires transient-service mode" >&2
    exit 2
fi
if [[ "$execution_mode" == "delegated-cgroup" && "$private_network" != "1" ]]; then
    echo "error: delegated-cgroup mode requires KILN_CARGO_PRIVATE_NETWORK=1" >&2
    exit 2
fi
environment_policy="${KILN_CARGO_ENVIRONMENT_POLICY:-}"
if [[ -z "$environment_policy" ]]; then
    if [[ "$execution_mode" == "transient-service" || "$execution_mode" == "delegated-cgroup" ]]; then
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

thermal_sensor_name="${KILN_CARGO_HOST_THERMAL_SENSOR_NAME:-}"
thermal_sensor_label="${KILN_CARGO_HOST_THERMAL_SENSOR_LABEL:-}"
thermal_limit_millicelsius="${KILN_CARGO_HOST_THERMAL_LIMIT_MILLICELSIUS:-}"
thermal_poll_milliseconds="${KILN_CARGO_HOST_THERMAL_POLL_MILLISECONDS:-}"
thermal_fields_set=0
for value in \
    "$thermal_sensor_name" \
    "$thermal_sensor_label" \
    "$thermal_limit_millicelsius" \
    "$thermal_poll_milliseconds"; do
    [[ -n "$value" ]] && thermal_fields_set=$((thermal_fields_set + 1))
done
if (( thermal_fields_set != 0 && thermal_fields_set != 4 )); then
    echo "error: all four KILN_CARGO_HOST_THERMAL_* fields must be set together" >&2
    exit 2
fi
hwmon_root="${KILN_CARGO_HWMON_ROOT:-/sys/class/hwmon}"
thermal_config_source="explicit"
external_thermal_policy_sha256="${KILN_WSL2_THERMAL_POLICY_SHA256:-}"
if [[ "$execution_mode" == "delegated-cgroup" ]]; then
    if [[ -n "$host_thermal_policy" ]] || (( thermal_fields_set != 0 )); then
        echo "error: delegated-cgroup mode requires the outer WSL2 thermal supervisor, not a Linux hwmon policy" >&2
        exit 2
    fi
    if [[ ! "$external_thermal_policy_sha256" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "error: delegated-cgroup mode requires a valid KILN_WSL2_THERMAL_POLICY_SHA256 binding" >&2
        exit 2
    fi
    thermal_config_source="external-wsl2-policy:$external_thermal_policy_sha256"
elif [[ -n "$host_thermal_policy" ]]; then
    if (( thermal_fields_set != 0 )); then
        echo "error: --host-thermal-policy conflicts with KILN_CARGO_HOST_THERMAL_* fields" >&2
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: --host-thermal-policy requires python3" >&2
        exit 2
    fi
    script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    policy_helper="$script_directory/qualification/prepare_host_thermal_policy.py"
    if [[ ! -f "$policy_helper" ]]; then
        echo "error: host thermal policy helper is missing: $policy_helper" >&2
        exit 2
    fi
    if ! policy_output="$(
        python3 "$policy_helper" \
            --hwmon-root "$hwmon_root" \
            cargo-fields \
            --policy "$host_thermal_policy"
    )"; then
        exit 2
    fi
    mapfile -t policy_fields <<< "$policy_output"
    if (( ${#policy_fields[@]} != 4 )); then
        echo "error: host thermal policy helper returned ${#policy_fields[@]} fields, expected 4" >&2
        exit 2
    fi
    thermal_sensor_name="${policy_fields[0]}"
    thermal_sensor_label="${policy_fields[1]}"
    thermal_limit_millicelsius="${policy_fields[2]}"
    thermal_poll_milliseconds="${policy_fields[3]}"
    thermal_fields_set=4
    thermal_config_source="policy:$host_thermal_policy"
fi
thermal_sensor_path=""
thermal_poll_seconds=""
if (( thermal_fields_set == 0 )) && [[ "$execution_mode" != "delegated-cgroup" ]]; then
    thermal_sensor_name="k10temp"
    thermal_sensor_label="Tctl"
    thermal_limit_millicelsius="97000"
    thermal_poll_milliseconds="250"
    thermal_config_source="automatic"
fi
if (( thermal_fields_set == 4 )) || [[ "$thermal_config_source" == "automatic" ]]; then
    if [[ ! "$thermal_limit_millicelsius" =~ ^[1-9][0-9]*$ ]] \
        || (( thermal_limit_millicelsius > 200000 )); then
        echo "error: KILN_CARGO_HOST_THERMAL_LIMIT_MILLICELSIUS must be in 1..=200000, got '$thermal_limit_millicelsius'" >&2
        exit 2
    fi
    if [[ ! "$thermal_poll_milliseconds" =~ ^[1-9][0-9]*$ ]] \
        || (( thermal_poll_milliseconds < 50 || thermal_poll_milliseconds > 60000 )); then
        echo "error: KILN_CARGO_HOST_THERMAL_POLL_MILLISECONDS must be in 50..=60000, got '$thermal_poll_milliseconds'" >&2
        exit 2
    fi
    thermal_matches=()
    for hwmon_dir in "$hwmon_root"/hwmon*; do
        [[ -d "$hwmon_dir" && -r "$hwmon_dir/name" ]] || continue
        IFS= read -r observed_name < "$hwmon_dir/name" || continue
        [[ "$observed_name" == "$thermal_sensor_name" ]] || continue
        for label_path in "$hwmon_dir"/temp*_label; do
            [[ -r "$label_path" ]] || continue
            IFS= read -r observed_label < "$label_path" || continue
            [[ "$observed_label" == "$thermal_sensor_label" ]] || continue
            input_path="${label_path%_label}_input"
            [[ -r "$input_path" ]] || continue
            thermal_matches+=("$input_path")
        done
    done
    if (( ${#thermal_matches[@]} == 0 )) && [[ "$thermal_config_source" == "automatic" ]]; then
        if [[ "$(cat /proc/sys/kernel/osrelease 2>/dev/null || true)" == *microsoft-standard-WSL2* ]]; then
            echo "error: WSL2 exposes no Linux package-temperature sensor; use delegated-cgroup mode under the required outer thermal supervisor" >&2
            exit 2
        fi
        thermal_sensor_name=""
        thermal_sensor_label=""
        thermal_limit_millicelsius=""
        thermal_poll_milliseconds=""
    elif (( ${#thermal_matches[@]} != 1 )); then
        echo "error: Cargo host thermal selector name='$thermal_sensor_name' label='$thermal_sensor_label' matched ${#thermal_matches[@]} readable sensors under '$hwmon_root'" >&2
        exit 2
    else
        thermal_sensor_path="${thermal_matches[0]}"
        IFS= read -r starting_temperature < "$thermal_sensor_path" || starting_temperature=""
        if [[ ! "$starting_temperature" =~ ^[0-9]+$ ]] \
            || (( starting_temperature == 0 || starting_temperature > 200000 )); then
            echo "error: Cargo host thermal sensor '$thermal_sensor_path' returned implausible reading '$starting_temperature'" >&2
            exit 2
        fi
        if (( starting_temperature >= thermal_limit_millicelsius )); then
            echo "error: refusing Cargo at ${starting_temperature} millicelsius; thermal limit is ${thermal_limit_millicelsius}" >&2
            exit 2
        fi
        thermal_poll_seconds="$(awk -v milliseconds="$thermal_poll_milliseconds" 'BEGIN { printf "%.3f", milliseconds / 1000 }')"
    fi
fi

thermal_summary="disabled"
if [[ -n "$thermal_sensor_path" ]]; then
    thermal_summary="${thermal_config_source}:${thermal_sensor_name}/${thermal_sensor_label}:${thermal_limit_millicelsius}mC@${thermal_poll_milliseconds}ms"
elif [[ "$execution_mode" == "delegated-cgroup" ]]; then
    thermal_summary="$thermal_config_source"
fi
cpu_quota_args=()
cpu_quota_summary="disabled"
if [[ -n "$cpu_quota_percent" ]]; then
    cpu_quota_args=(-p "CPUQuota=${cpu_quota_percent}%")
    cpu_quota_summary="${cpu_quota_percent}%"
fi
echo "bounded-cargo: mode=$execution_mode jobs=$jobs cpu_quota=$cpu_quota_summary available=${available_gib}GiB reserve=${reserve_gib}GiB aggregate_limit=${limit_gib}GiB swap_limit=0 private_network=$private_network environment_policy=$environment_policy thermal=$thermal_summary" >&2
read -r bounded_uuid < /proc/sys/kernel/random/uuid
if [[ "$execution_mode" == "scope" ]]; then
    bounded_unit="kiln-cargo-bounded-${bounded_uuid//-/}.scope"
elif [[ "$execution_mode" == "transient-service" ]]; then
    bounded_unit="kiln-cargo-bounded-${bounded_uuid//-/}.service"
else
    bounded_unit=""
fi

if [[ "$execution_mode" == "delegated-cgroup" ]]; then
    if [[ -z "$cpu_quota_percent" ]]; then
        echo "error: delegated-cgroup mode requires KILN_CARGO_CPU_QUOTA_PERCENT" >&2
        exit 2
    fi
    if (( cpu_quota_percent > 100 )); then
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
    if [[ ! "$scope_memory_max" =~ ^[1-9][0-9]*$ ]] \
        || [[ ! "$scope_pids_max" =~ ^[1-9][0-9]*$ ]] \
        || [[ ! "$scope_host_uid" =~ ^[1-9][0-9]*$ ]] \
        || [[ "$scope_cpu_quota" != "$cpu_quota_percent" ]]; then
        echo "error: malformed WSL2 scope resource binding" >&2
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
    if [[ "$observed_memory_max" != "$scope_memory_max" ]] \
        || (( observed_memory_max > limit_gib * 1024 * 1024 * 1024 )); then
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
    )
    if [[ "$environment_policy" == "closed-qualification-test-v1" ]]; then
        closed_environment_names+=(
            CUDARC_CUDA_VERSION
            KILN_CUDA_ARCHS
            KILN_QUALIFICATION
            KILN_QUALIFICATION_HF_LOGITS_PATH
            KILN_QUALIFICATION_MODEL_PATH
        )
    elif [[ "$environment_policy" == "inherit" ]]; then
        exec "$cargo_executable" "$@"
    fi
    closed_environment=()
    for name in "${closed_environment_names[@]}"; do
        if [[ -v "$name" ]]; then
            closed_environment+=("$name=${!name}")
        fi
    done
    exec env -i "${closed_environment[@]}" "$cargo_executable" "$@"
fi

bounded_runner_pid=""
thermal_watchdog_pid=""
thermal_trip_file="${TMPDIR:-/tmp}/kiln-cargo-bounded-${bounded_uuid//-/}.thermal-trip"
cleanup_unit() {
    if [[ -n "$thermal_watchdog_pid" ]]; then
        kill "$thermal_watchdog_pid" >/dev/null 2>&1 || true
        wait "$thermal_watchdog_pid" >/dev/null 2>&1 || true
    fi
    systemctl --user stop "$bounded_unit" >/dev/null 2>&1 || true
    if [[ -n "$bounded_runner_pid" ]]; then
        kill -TERM "$bounded_runner_pid" >/dev/null 2>&1 || true
        wait "$bounded_runner_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$thermal_trip_file"
}
trap cleanup_unit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_thermal_watchdog() {
    [[ -n "$thermal_sensor_path" ]] || return 0
    (
        while kill -0 "$bounded_runner_pid" >/dev/null 2>&1; do
            IFS= read -r current_temperature < "$thermal_sensor_path" || current_temperature=""
            trip_reason=""
            if [[ ! "$current_temperature" =~ ^[0-9]+$ ]] \
                || (( current_temperature == 0 || current_temperature > 200000 )); then
                trip_reason="sensor '$thermal_sensor_path' returned implausible reading '$current_temperature'"
            elif (( current_temperature >= thermal_limit_millicelsius )); then
                trip_reason="temperature ${current_temperature} millicelsius reached limit ${thermal_limit_millicelsius}"
            fi
            if [[ -n "$trip_reason" ]]; then
                printf '%s\n' "$trip_reason" > "$thermal_trip_file"
                echo "error: Cargo host thermal guard tripped: $trip_reason" >&2
                systemctl --user stop "$bounded_unit" >/dev/null 2>&1 || true
                kill -TERM "$bounded_runner_pid" >/dev/null 2>&1 || true
                exit 0
            fi
            sleep "$thermal_poll_seconds"
        done
    ) &
    thermal_watchdog_pid=$!
}

wait_for_bounded_runner() {
    start_thermal_watchdog
    if wait "$bounded_runner_pid"; then
        bounded_status=0
    else
        bounded_status=$?
    fi
    if [[ -n "$thermal_watchdog_pid" ]]; then
        kill "$thermal_watchdog_pid" >/dev/null 2>&1 || true
        wait "$thermal_watchdog_pid" >/dev/null 2>&1 || true
        thermal_watchdog_pid=""
    fi
    if [[ -f "$thermal_trip_file" ]]; then
        return 3
    fi
    return "$bounded_status"
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

# A process in a bubblewrap PID namespace cannot be attached to the host user
# manager as a scope. Qualification uses a transient service instead: Cargo is
# still in one bounded cgroup, while PrivateNetwork independently preserves the
# offline build boundary. The named unit and EXIT trap make client cancellation
# stop the complete compiler/linker tree; RuntimeMaxSec bounds hard-kill cases.

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
            CUDARC_CUDA_VERSION
            KILN_CUDA_ARCHS
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
    -p "$private_network_property" \
    "$cargo_executable" "$@" &
bounded_runner_pid=$!
if wait_for_bounded_runner; then
    exit 0
else
    exit $?
fi

#!/usr/bin/env bash
# kiln-heartbeat — Phase A of the pod-wedge watchdog system.
#
# Writes /workspace/heartbeat.txt atomically every 30s with ground-truth
# liveness signals (GPU util, load, top procs, build-tree mtime, build log
# sizes). The orchestrator-side watchdog (Phase B) parses this file to
# distinguish a genuinely working pod from one whose sshd has wedged but
# whose actual workload is dead.
#
# The output schema is an API contract — Phase B parsers depend on these
# exact key names. Add new keys at the end; do not rename or reorder.
#
# Robustness rules:
#   - Atomic write: render to .new, then `mv` (rename(2) is atomic on Linux).
#   - Self-healing: every iteration is wrapped in a subshell + `|| true`,
#     so a single bad command never stops the heartbeat.
#   - Silent: all output goes to the file; nothing on stdout/stderr.

set -u  # not -e: a failed sub-command must not kill the loop

HEARTBEAT_FILE="${KILN_HEARTBEAT_FILE:-/workspace/heartbeat.txt}"
HEARTBEAT_INTERVAL="${KILN_HEARTBEAT_INTERVAL:-30}"
TARGET_DIR="${KILN_HEARTBEAT_TARGET_DIR:-/workspace/kiln/target}"
BUILD_LOGS=(${KILN_HEARTBEAT_BUILD_LOGS:-/tmp/build.log /tmp/bench.log})

mkdir -p "$(dirname "$HEARTBEAT_FILE")" 2>/dev/null || true

write_heartbeat() {
    local tmp="${HEARTBEAT_FILE}.new"
    {
        printf 'timestamp_iso: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
        printf 'uptime_s: %s\n' "$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null || echo 0)"

        # Load average — /proc/loadavg is "1m 5m 15m running/total lastpid"
        if read -r l1 l5 l15 _ < /proc/loadavg 2>/dev/null; then
            printf 'load_1m: %s\n' "$l1"
            printf 'load_5m: %s\n' "$l5"
            printf 'load_15m: %s\n' "$l15"
        else
            printf 'load_1m: none\nload_5m: none\nload_15m: none\n'
        fi

        # GPU 0 — nvidia-smi may be absent (CPU-only pod) or fail (no driver).
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu
            gpu="$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
                              --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)"
            if [ -n "$gpu" ]; then
                local util mem_util mem_used mem_total
                util="$(echo "$gpu"      | awk -F, '{gsub(/ /,""); print $1}')"
                mem_util="$(echo "$gpu"  | awk -F, '{gsub(/ /,""); print $2}')"
                mem_used="$(echo "$gpu"  | awk -F, '{gsub(/ /,""); print $3}')"
                mem_total="$(echo "$gpu" | awk -F, '{gsub(/ /,""); print $4}')"
                printf 'gpu0_util_pct: %s\n' "${util:-none}"
                printf 'gpu0_mem_util_pct: %s\n' "${mem_util:-none}"
                printf 'gpu0_mem_used_mib: %s\n' "${mem_used:-none}"
                printf 'gpu0_mem_total_mib: %s\n' "${mem_total:-none}"
            else
                printf 'gpu0_util_pct: none\ngpu0_mem_util_pct: none\ngpu0_mem_used_mib: none\ngpu0_mem_total_mib: none\n'
            fi
        else
            printf 'gpu0_util_pct: none\ngpu0_mem_util_pct: none\ngpu0_mem_used_mib: none\ngpu0_mem_total_mib: none\n'
        fi

        # Top 3 procs by CPU%. Columns: comm %cpu %mem. `ps -e --sort` is
        # procps-only (Ubuntu 22.04 image); on busybox / minimal images the
        # awk pipe yields no rows and we fall through to `none 0.0 0.0` so
        # the section always has at least one parseable line.
        printf 'top_procs:\n'
        local procs
        procs="$(ps -eo comm=,pcpu=,pmem= --sort=-pcpu 2>/dev/null \
                 | awk 'NF>=3 {printf "  %s %s %s\n", $1, $2, $3}' \
                 | head -3)"
        if [ -n "$procs" ]; then
            printf '%s\n' "$procs"
        else
            printf '  none 0.0 0.0\n'
        fi

        # Build-tree freshness — the canonical "are we still compiling" signal.
        if [ -e "$TARGET_DIR" ]; then
            local mtime_epoch now_epoch age
            mtime_epoch="$(stat -c %Y "$TARGET_DIR" 2>/dev/null || echo 0)"
            now_epoch="$(date -u +%s)"
            age=$(( now_epoch - mtime_epoch ))
            printf 'workspace_target_mtime: %s\n' \
                "$(date -u -d "@${mtime_epoch}" +%Y-%m-%dT%H:%M:%S+00:00 2>/dev/null || echo none)"
            printf 'workspace_target_mtime_age_s: %s\n' "$age"
        else
            printf 'workspace_target_mtime: absent\n'
            printf 'workspace_target_mtime_age_s: none\n'
        fi

        # Build log sizes — Phase B watches for log growth as a liveness signal.
        printf 'build_logs:\n'
        local any_log=0
        for log in "${BUILD_LOGS[@]}"; do
            if [ -f "$log" ]; then
                local sz
                sz="$(stat -c %s "$log" 2>/dev/null || echo 0)"
                printf '  %s %s\n' "$log" "$sz"
                any_log=1
            fi
        done
        [ $any_log -eq 0 ] && printf '  none 0\n'
    } > "$tmp" 2>/dev/null

    mv -f "$tmp" "$HEARTBEAT_FILE" 2>/dev/null || true
}

# Outer loop: every iteration is wrapped so any single-cycle failure is
# absorbed and the next cycle still runs. `sleep` itself is outside the
# subshell so an interrupted iteration doesn't cause a tight CPU spin.
while true; do
    ( write_heartbeat ) >/dev/null 2>&1 || true
    sleep "$HEARTBEAT_INTERVAL"
done

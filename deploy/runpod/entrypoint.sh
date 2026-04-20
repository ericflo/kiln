#!/usr/bin/env bash
# entrypoint.sh — install RunPod $PUBLIC_KEY, start kiln-heartbeat, start sshd.
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Generate host keys on first boot
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
fi

# Phase A pod-wedge watchdog: write /workspace/heartbeat.txt every 30s.
# Backgrounded BEFORE the sshd exec so the heartbeat lives on as a child of
# PID 1 (the replacing sshd). The Phase B orchestrator-side watchdog uses
# this file as the ground-truth liveness signal — if the file stops updating
# while the SSH connection still answers TCP, the pod has wedged.
if [ -x /usr/local/bin/kiln-heartbeat ]; then
    /usr/local/bin/kiln-heartbeat > /tmp/kiln-heartbeat.stderr 2>&1 &
fi

exec "$@"

#!/usr/bin/env bash
# entrypoint.sh — install RunPod $PUBLIC_KEY, start sshd.
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

exec "$@"

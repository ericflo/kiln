#!/usr/bin/env bash
# Run a qualification Cargo test in a private bounded native or WSL2 scope.

set -euo pipefail

export CARGO_NET_OFFLINE=true
export KILN_CARGO_ENVIRONMENT_POLICY=closed-qualification-test-v1
export KILN_CARGO_JOBS=1
export KILN_CARGO_PRIVATE_NETWORK=1
export KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS=1740

if [[ "${KILN_WSL2_SCOPE_BOUNDARY:-}" == "systemd-user-scope-feedback-v1" ]]; then
    export KILN_CARGO_EXECUTION_MODE=delegated-cgroup
else
    export KILN_CARGO_EXECUTION_MODE=transient-service
fi

exec scripts/cargo-bounded.sh "$@"

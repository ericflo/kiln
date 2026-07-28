#!/usr/bin/env bash
# Run a qualification Cargo test in the WSL2-safe delegated cgroup.

set -euo pipefail

export CARGO_NET_OFFLINE=true
export KILN_CARGO_ENVIRONMENT_POLICY=closed-qualification-test-v1
export KILN_CARGO_EXECUTION_MODE=delegated-cgroup
export KILN_CARGO_JOBS=1
export KILN_CARGO_PRIVATE_NETWORK=1
export KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS=1740

exec scripts/cargo-bounded.sh "$@"

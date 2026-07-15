#!/usr/bin/env bash
# Run a qualification Cargo test in the PID-namespace-safe bounded service.

set -euo pipefail

export CARGO_NET_OFFLINE=true
export KILN_CARGO_ENVIRONMENT_POLICY=closed-qualification-test-v1
export KILN_CARGO_EXECUTION_MODE=transient-service
export KILN_CARGO_JOBS=1
export KILN_CARGO_CPU_QUOTA_PERCENT=400
export KILN_CARGO_MIN_AVAILABLE_GIB=15
export KILN_CARGO_PRIVATE_NETWORK=1
export KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS=1740

exec scripts/cargo-bounded.sh "$@"

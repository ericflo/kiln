#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CARGO_BIN="${CARGO_BIN:-cargo}"

run_step() {
  printf '\n==> %s\n' "$*"
  "$@"
}

command -v "$PYTHON_BIN" >/dev/null
command -v "$CARGO_BIN" >/dev/null

run_step "$PYTHON_BIN" scripts/generate_backend_capability_report.py --check
run_step "$CARGO_BIN" test --locked -p kiln-model --test backend_capability_contract

run_step "$CARGO_BIN" test --locked -p kiln-tensor tensor::tests
run_step "$CARGO_BIN" test --locked -p kiln-tensor device_op::tests
run_step "$CARGO_BIN" test --locked -p kiln-tensor matmul_matrix_core
run_step "$CARGO_BIN" test --locked -p kiln-optim --test integration
run_step "$CARGO_BIN" test --locked -p kiln-optim --test end_to_end_training
run_step "$CARGO_BIN" test --locked -p kiln-graph replay
run_step "$CARGO_BIN" test --locked -p kiln-graph --test capture_lifetime

run_step "$PYTHON_BIN" scripts/run_backend_latency_fixture.py --self-test
run_step "$PYTHON_BIN" scripts/write_backend_latency_result_artifact.py --self-test
run_step "$PYTHON_BIN" scripts/import_backend_latency_artifact.py --self-test
run_step "$PYTHON_BIN" scripts/lock_backend_latency_thresholds.py --self-test
run_step "$PYTHON_BIN" scripts/check_backend_latency_fixtures.py --self-test
run_step "$PYTHON_BIN" scripts/plan_backend_latency_fixture_dispatch.py --self-test
run_step "$PYTHON_BIN" scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json \
  --require-covered

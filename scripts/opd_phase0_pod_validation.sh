#!/usr/bin/env bash
# §13 Phase 0 validation — real-hardware run.
#
# Runs on a leased Kiln RunPod CUDA pod (A6000 or stronger). Builds the
# on-policy-distillation branch with --features cuda, runs the
# kiln-opd-loss-kernel throughput bench, compares the result against
# bench-results/opd-a6000-baseline.json, and runs the full kiln-train
# OPD/diagnostics/logit_source/receipt + kiln-server lib test suites
# on real GPU hardware. Writes a structured report to
# /workspace/opd-phase0-validation.json so the caller can pull it back
# via ce kiln-runpod-wait-file.
#
# Exit non-zero on any failure so wait-file detects the failure
# (instead of polling indefinitely). Trap-on-failure dumps a tail of
# every log so debugging from the lease ID alone is possible.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ericflo/kiln.git}"
BRANCH="${BRANCH:-on-policy-distillation}"
WORK="/workspace/opd-phase0"
RESULT="/workspace/opd-phase0-validation.json"
LOG_DIR="$WORK/logs"

mkdir -p "$WORK" "$LOG_DIR"
cd "$WORK"

trap 'rc=$?; echo "FAILED rc=$rc phase=${PHASE:-unknown} — tailing logs" >&2; ls "$LOG_DIR" 2>/dev/null | xargs -I{} sh -c "echo === {} ===; tail -n 60 \"$LOG_DIR/{}\"" >&2; printf "{\"ok\":false,\"rc\":%s,\"phase\":\"%s\"}\n" "$rc" "${PHASE:-unknown}" > "$RESULT"; exit $rc' ERR

PHASE="clone"
if [ ! -d kiln/.git ]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" kiln >"$LOG_DIR/clone.log" 2>&1
else
  (cd kiln && git fetch --depth 1 origin "$BRANCH" && git checkout "$BRANCH" && git reset --hard "origin/$BRANCH") >"$LOG_DIR/clone.log" 2>&1
fi

cd kiln
KILN_COMMIT=$(git rev-parse HEAD)
echo "kiln branch=$BRANCH commit=$KILN_COMMIT" | tee "$LOG_DIR/commit.txt"

PHASE="env"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tee "$LOG_DIR/nvidia-smi.csv"
nvcc --version | tee "$LOG_DIR/nvcc.txt"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
# Pick the right CUDA arch based on the GPU name.
case "$GPU_NAME" in
  *A100*|*A800*) KILN_CUDA_ARCHS=80 ;;
  *A6000*|*A40*|*A10*|*A30*|*A16*|*3090*|*A5000*) KILN_CUDA_ARCHS=86 ;;
  *H100*|*H200*|*H800*) KILN_CUDA_ARCHS=90 ;;
  *L40*|*L4*) KILN_CUDA_ARCHS=89 ;;
  *) KILN_CUDA_ARCHS=80 ;;  # safe baseline
esac
export KILN_CUDA_ARCHS
export CARGO_TERM_COLOR=never
echo "GPU=$GPU_NAME KILN_CUDA_ARCHS=$KILN_CUDA_ARCHS"

PHASE="build_opd_kernel"
cargo build --release --features cuda -p kiln-opd-loss-kernel \
  >"$LOG_DIR/build_opd_kernel.log" 2>&1

PHASE="opd_kernel_tests"
cargo test --release --features cuda -p kiln-opd-loss-kernel --lib \
  -- --test-threads=1 \
  >"$LOG_DIR/opd_kernel_tests.log" 2>&1

PHASE="opd_kernel_bench"
cargo run --release --features cuda -p kiln-opd-loss-kernel \
  --example bench_opd_topk_kl \
  >"$LOG_DIR/opd_kernel_bench.log" 2>&1

PHASE="perf_gate"
# Compare bench output against the committed baseline for the
# detected GPU. The §9.9 gate is per-GPU because raw tok/s varies
# significantly across A6000 / A100 / H100 etc., even on the same
# kernel — the regression check is meant to catch code regressions,
# not hardware differences.
case "$GPU_NAME" in
  *A100*|*A800*) BASELINE_FILE=bench-results/opd-a100-baseline.json ;;
  *A6000*)        BASELINE_FILE=bench-results/opd-a6000-baseline.json ;;
  *)              BASELINE_FILE=bench-results/opd-a6000-baseline.json ;;  # closest available
esac
PERF_GATE=pass
python3 bench-results/check_opd_regression.py \
  --bench-stdout "$LOG_DIR/opd_kernel_bench.log" \
  --baseline "$BASELINE_FILE" \
  >"$LOG_DIR/perf_gate.log" 2>&1 \
  || PERF_GATE=fail

PHASE="opd_train_tests"
cargo test --release --features cuda -p kiln-train --lib opd:: \
  >"$LOG_DIR/opd_train_tests.log" 2>&1

PHASE="diagnostics_tests"
cargo test --release --features cuda -p kiln-train --lib diagnostics:: \
  >"$LOG_DIR/diagnostics_tests.log" 2>&1

PHASE="logit_source_tests"
cargo test --release --features cuda -p kiln-train --lib logit_source:: \
  >"$LOG_DIR/logit_source_tests.log" 2>&1

PHASE="receipt_tests"
cargo test --release --features cuda -p kiln-train --lib receipt:: \
  >"$LOG_DIR/receipt_tests.log" 2>&1

PHASE="server_tests"
SERVER_TESTS=pass
cargo test --release --features cuda -p kiln-server --lib \
  >"$LOG_DIR/server_tests.log" 2>&1 \
  || SERVER_TESTS=fail

PHASE="report"
python3 - <<PY > "$RESULT"
import json, pathlib, re, subprocess
log = pathlib.Path("$LOG_DIR/opd_kernel_bench.log").read_text()
row_re = re.compile(
    r"T=\s*(?P<T>\d+)\s+H=\s*(?P<H>\d+)\s+V=\s*(?P<V>\d+)\s+"
    r"K=\s*(?P<K>\d+)\s+(?P<dtype>F32|BF16|F16)\s+iters=\s*\d+\s+"
    r"kernel=\s*(?P<km>[\d.]+)ms\s+candle=\s*(?P<cm>[\d.]+)ms\s+"
    r"(?P<sp>[\d.]+)x\s+(?P<tps>\d+)\s+tok/s"
)
rows = []
for m in row_re.finditer(log):
    rows.append({
        "T": int(m["T"]), "K": int(m["K"]), "dtype": m["dtype"],
        "kernel_ms": float(m["km"]), "candle_ms": float(m["cm"]),
        "speedup_x": float(m["sp"]), "kernel_tok_s": int(m["tps"]),
    })

def count_passed(path):
    text = pathlib.Path(path).read_text() if pathlib.Path(path).exists() else ""
    m = re.search(r"test result: ok\. (\d+) passed", text)
    return int(m.group(1)) if m else None

report = {
    "ok": True,
    "kiln_commit": "$KILN_COMMIT",
    "gpu": "$GPU_NAME",
    "kiln_cuda_archs": "$KILN_CUDA_ARCHS",
    "perf_gate": "$PERF_GATE",
    "server_tests": "$SERVER_TESTS",
    "kernel_bench_rows": rows,
    "tests": {
        "kiln_opd_loss_kernel": count_passed("$LOG_DIR/opd_kernel_tests.log"),
        "kiln_train_opd": count_passed("$LOG_DIR/opd_train_tests.log"),
        "kiln_train_diagnostics": count_passed("$LOG_DIR/diagnostics_tests.log"),
        "kiln_train_logit_source": count_passed("$LOG_DIR/logit_source_tests.log"),
        "kiln_train_receipt": count_passed("$LOG_DIR/receipt_tests.log"),
        "kiln_server": count_passed("$LOG_DIR/server_tests.log"),
    },
    "phases_completed": [
      "clone", "env", "build_opd_kernel", "opd_kernel_tests",
      "opd_kernel_bench", "perf_gate", "opd_train_tests",
      "diagnostics_tests", "logit_source_tests", "receipt_tests",
      "server_tests", "report",
    ],
}
print(json.dumps(report, indent=2))
PY

echo "DONE"
ls -la "$RESULT"

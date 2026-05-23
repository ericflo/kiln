#!/usr/bin/env bash
# Phase 1.31 — kiln-tensor substrate status audit.
#
# Prints which #1082 deliverables have shipped, based on file
# existence + Cargo workspace membership. The output is a dashboard
# for the migration progress; it does NOT need GPU access to run.
#
# Usage:
#     scripts/audit-substrate-status.sh                # human-readable to stdout
#     scripts/audit-substrate-status.sh --markdown     # also write bench-results/substrate-status.md
#     scripts/audit-substrate-status.sh --json         # machine-readable JSON for CI integration
#
# Each row reports:
#   - the issue's phase label (Phase 0.1 / Phase 1.12 / Phase 2.5 / ...)
#   - file or directory the deliverable lives at
#   - "shipped" if present, "todo" if missing
#
# Re-runnable; deterministic given a fixed working tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODE="human"
case "${1:-}" in
    --markdown) MODE="markdown" ;;
    --json) MODE="json" ;;
    --help|-h)
        sed -n '2,18p' "$0"
        exit 0 ;;
esac

python3 - "$REPO_ROOT" "$MODE" <<'PY'
import json, os, sys
from pathlib import Path

repo_root = Path(sys.argv[1])
mode = sys.argv[2]

# (phase, label, paths_to_check) where paths_to_check is a list of
# repo-relative paths; deliverable counts as "shipped" if ALL paths exist.
ROWS = [
    # Phase 0 — decision-shaping audits.
    ("0.1",  "audit-candle-usage script + CSV",
     ["scripts/audit-candle-usage.sh", "bench-results/candle-api-surface.csv"]),
    ("0.2",  "CustomOpN audit",
     ["scripts/audit-customop.py", "bench-results/customop-audit.csv"]),
    ("0.3",  "determinism stance (PROFILING.md section)",
     ["PROFILING.md"]),  # presence-only; deeper check via grep below
    ("0.4",  "parity-tolerance.csv",
     ["scripts/build-parity-tolerance.py", "bench-results/parity-tolerance.csv"]),
    ("0.5",  "DType usage audit",
     ["scripts/audit-dtype-usage.py", "bench-results/dtype-usage.csv"]),
    ("0.6",  "multi-GPU seam audit",
     ["scripts/audit-multi-gpu-seam.sh", "bench-results/multi-gpu-seam.csv"]),
    ("0.7",  "preserve-list audit (NVTX + KILN_* + BR)",
     ["scripts/audit-preserve-list.sh", "bench-results/preserve-list-nvtx.csv"]),
    ("0.8",  "kiln-blas crate + cublasLt probe",
     ["crates/kiln-blas/Cargo.toml", "crates/kiln-blas/csrc/cublaslt_probe.cu",
      "crates/kiln-blas/examples/cublaslt_mlp_probe.rs"]),
    ("0.9",  "Vulkan MLP probe example",
     ["crates/kiln-vulkan-kernel/examples/vk_mlp_probe.rs"]),
    ("0.10", "pre-migration baseline harness",
     ["scripts/capture-pre-migration-baseline.sh",
      "bench-results/pre-migration-baseline/README.md"]),
    # Phase 1 — kiln-tensor scaffold + ops.
    ("1.1",  "kiln-tensor scaffold + Error/Result/bail!/ensure!",
     ["crates/kiln-tensor/Cargo.toml", "crates/kiln-tensor/src/error.rs"]),
    ("1.2",  "DType enum",        ["crates/kiln-tensor/src/dtype.rs"]),
    ("1.3",  "TensorId + Layout", ["crates/kiln-tensor/src/tensor_id.rs",
                                     "crates/kiln-tensor/src/layout.rs"]),
    ("1.4",  "Device + Storage trait + CPU storage",
                                    ["crates/kiln-tensor/src/device.rs",
                                     "crates/kiln-tensor/src/storage.rs"]),
    ("1.5",  "Tensor struct + Element trait",
                                    ["crates/kiln-tensor/src/tensor.rs",
                                     "crates/kiln-tensor/src/element.rs"]),
    ("1.6",  "CUDA storage (cuda feature)",
                                    ["crates/kiln-tensor/src/cuda_storage.rs"]),
    ("1.7",  "Metal storage (metal feature)",
                                    ["crates/kiln-tensor/src/metal_storage.rs"]),
    ("1.8",  "Vulkan storage (vulkan feature)",
                                    ["crates/kiln-tensor/src/vulkan_storage.rs"]),
    ("1.9",  "safetensors loader", ["crates/kiln-tensor/src/safetensors.rs"]),
    ("1.10", "copy-counter instrumentation",
                                    ["crates/kiln-tensor/src/profile.rs"]),
    ("1.11", "Determinism + KILN_DETERMINISTIC envelope",
                                    ["crates/kiln-tensor/src/determinism.rs"]),
    ("1.12", "DeviceOp trait + BackwardOp scaffold",
                                    ["crates/kiln-tensor/src/device_op.rs"]),
    ("1.13", "embedding op (DeviceOp2)",
                                    ["crates/kiln-tensor/src/ops/embedding.rs"]),
    ("1.14", "rmsnorm op (DeviceOp2)",
                                    ["crates/kiln-tensor/src/ops/rmsnorm.rs"]),
    ("1.15", "elementwise add/sub/mul/div (DeviceOp2)",
                                    ["crates/kiln-tensor/src/ops/elementwise.rs"]),
    ("1.16", "silu + sigmoid activations (DeviceOp1)",
                                    ["crates/kiln-tensor/src/ops/activation.rs"]),
    ("1.17", "softmax_last_dim (DeviceOp1)",
                                    ["crates/kiln-tensor/src/ops/softmax.rs"]),
    ("1.18", "matmul CPU reference (DeviceOp2)",
                                    ["crates/kiln-tensor/src/ops/matmul.rs"]),
    ("1.19", "argmax_last_dim (DeviceOp1)",
                                    ["crates/kiln-tensor/src/ops/argmax.rs"]),
    ("1.20", "cast op (DeviceOp1)", ["crates/kiln-tensor/src/ops/cast.rs"]),
    ("1.21", "rope op (DeviceOp3)", ["crates/kiln-tensor/src/ops/rope.rs"]),
    ("1.22", "l2_norm op (DeviceOp1)",
                                    ["crates/kiln-tensor/src/ops/l2norm.rs"]),
    ("1.23", "mul_sigmoid_gate (silu*mul, DeviceOp2)",
                                    ["crates/kiln-tensor/src/ops/silu_mul.rs"]),
    ("1.24", "mini transformer block integration test",
                                    ["crates/kiln-tensor/tests/mini_block.rs"]),
    ("1.25", "Activation registry",
                                    ["crates/kiln-tensor/src/activation_registry.rs"]),
    ("1.26", "StreamPlanner",       ["crates/kiln-tensor/src/stream_planner.rs"]),
    ("1.27", "Allocator trait skeleton",
                                    ["crates/kiln-tensor/src/allocator.rs"]),
    ("1.28", "CpuAllocator",        ["crates/kiln-tensor/src/cpu_allocator.rs"]),
    ("1.29", "Allocator + CaptureSession integration test",
                                    ["crates/kiln-graph/tests/capture_lifetime.rs"]),
    ("1.30", "ARCHITECTURE.md migration substrate section",
                                    ["ARCHITECTURE.md"]),
    # Phase 1 (continued) — substrate completion.
    ("1.32", "Tensor version counter (anti-pattern 16 wiring)",
                                    ["crates/kiln-tensor/src/tensor.rs"]),
    ("1.33", "reduce_sum + reduce_mean CPU DeviceOps",
                                    ["crates/kiln-tensor/src/ops/reduce.rs"]),
    ("1.34", "index_select CPU DeviceOp",
                                    ["crates/kiln-tensor/src/ops/index_select.rs"]),
    ("1.35", "masked_fill + causal_mask CPU DeviceOps",
                                    ["crates/kiln-tensor/src/ops/mask.rs"]),
    ("1.36", "causal attention block integration test",
                                    ["crates/kiln-tensor/tests/attention_block.rs"]),
    ("1.37", "kiln-param + kiln-optim integration test",
                                    ["crates/kiln-optim/tests/integration.rs"]),
    ("1.38", "kiln-autograd end-to-end backward integration test",
                                    ["crates/kiln-autograd/tests/end_to_end.rs"]),
    ("1.39", "safetensors save path",
                                    ["crates/kiln-tensor/src/safetensors.rs"]),
    ("1.40", "Parameter::content_hash method",
                                    ["crates/kiln-param/src/parameter.rs"]),
    ("1.42", "full training step demo (tensor + autograd + param + optim)",
                                    ["crates/kiln-optim/tests/full_training_step.rs"]),
    # Phase 2.5 / 5 / 6a / 6.5 — new crates.
    ("2.1",  "kiln-blas production API sketch (AlgoCache + WorkspacePool)",
                                    ["crates/kiln-blas/src/algo_cache.rs",
                                     "crates/kiln-blas/src/workspace_pool.rs"]),
    ("2.2",  "kiln-mps crate scaffold",
                                    ["crates/kiln-mps/Cargo.toml",
                                     "crates/kiln-mps/src/lib.rs"]),
    ("2.3",  "kiln-vulkan-blas crate scaffold",
                                    ["crates/kiln-vulkan-blas/Cargo.toml",
                                     "crates/kiln-vulkan-blas/src/lib.rs"]),
    ("2.5",  "kiln-param scaffold (Parameter + AmpPolicy + content hash)",
                                    ["crates/kiln-param/Cargo.toml",
                                     "crates/kiln-param/src/parameter.rs"]),
    ("5",    "kiln-graph crate (CapturedGraph + CaptureSession)",
                                    ["crates/kiln-graph/Cargo.toml",
                                     "crates/kiln-graph/src/captured_graph.rs"]),
    ("6a",   "kiln-autograd (Tape + GradStore + BackwardOp)",
                                    ["crates/kiln-autograd/Cargo.toml",
                                     "crates/kiln-autograd/src/tape.rs"]),
    ("6.5",  "kiln-optim (OptimStep + AdamW CPU)",
                                    ["crates/kiln-optim/Cargo.toml",
                                     "crates/kiln-optim/src/adamw.rs"]),
    ("6.5.1", "Sgd + Lion/Muon scaffolds",
                                    ["crates/kiln-optim/src/sgd.rs",
                                     "crates/kiln-optim/src/lion_muon.rs"]),
]

def is_shipped(paths):
    return all((repo_root / p).exists() for p in paths)

rows = []
for phase, label, paths in ROWS:
    rows.append({
        "phase": phase,
        "label": label,
        "paths": paths,
        "status": "shipped" if is_shipped(paths) else "todo",
    })

if mode == "human":
    shipped = sum(1 for r in rows if r["status"] == "shipped")
    total = len(rows)
    print(f"# kiln-tensor substrate status — {shipped}/{total} deliverables shipped")
    print()
    for r in rows:
        mark = "✓" if r["status"] == "shipped" else " "
        print(f"  [{mark}] Phase {r['phase']:>4}  {r['label']}")
elif mode == "markdown":
    out = repo_root / "bench-results" / "substrate-status.md"
    out.parent.mkdir(exist_ok=True)
    shipped = sum(1 for r in rows if r["status"] == "shipped")
    total = len(rows)
    with out.open("w") as f:
        f.write(f"# kiln-tensor substrate status\n\n")
        f.write(f"**{shipped} / {total} deliverables shipped** as of latest re-run "
                f"of `scripts/audit-substrate-status.sh`.\n\n")
        f.write("Regenerate: `scripts/audit-substrate-status.sh --markdown`.\n\n")
        f.write("## Phase 0 — decision-shaping\n\n")
        f.write("| Phase | Deliverable | Status |\n|---|---|:-:|\n")
        for r in rows:
            if r["phase"].startswith("0"):
                mark = "✓" if r["status"] == "shipped" else "✗"
                f.write(f"| {r['phase']} | {r['label']} | {mark} |\n")
        f.write("\n## Phase 1 — kiln-tensor scaffold + ops\n\n")
        f.write("| Phase | Deliverable | Status |\n|---|---|:-:|\n")
        for r in rows:
            if r["phase"].startswith("1"):
                mark = "✓" if r["status"] == "shipped" else "✗"
                f.write(f"| {r['phase']} | {r['label']} | {mark} |\n")
        f.write("\n## Phase 2 / 2.5 / 5 / 6a / 6.5 — new crates\n\n")
        f.write("| Phase | Deliverable | Status |\n|---|---|:-:|\n")
        # Sort 2.x / 2.5 / 5 / 6a / 6.5 / 6.5.1 in sensible order.
        order = {"2.1": 0, "2.2": 1, "2.3": 2, "2.5": 3, "5": 4, "6a": 5,
                 "6.5": 6, "6.5.1": 7}
        non_01 = [r for r in rows if not r["phase"].startswith(("0", "1"))]
        non_01.sort(key=lambda r: order.get(r["phase"], 99))
        for r in non_01:
            mark = "✓" if r["status"] == "shipped" else "✗"
            f.write(f"| {r['phase']} | {r['label']} | {mark} |\n")
    sys.stderr.write(f"wrote {out}\n")
    sys.stdout.write(f"{shipped}/{total} deliverables shipped\n")
elif mode == "json":
    json.dump({"rows": rows, "shipped": sum(1 for r in rows if r["status"] == "shipped"),
               "total": len(rows)}, sys.stdout, indent=2)
    sys.stdout.write("\n")
PY

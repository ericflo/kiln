#!/usr/bin/env python3
"""
Phase 0.4 — Per-op parity tolerance matrix.

Builds `bench-results/parity-tolerance.csv` with one row per
`{op, dtype, backend}` cell. Each row carries:

    op, dtype, backend, fwd_atol, bwd_atol, fwd_determinism, bwd_determinism, notes

where:
    fwd_atol / bwd_atol are absolute-tolerance thresholds (NOT relative),
    fwd_determinism / bwd_determinism are one of:
        "constructive"           — bit-identical across runs
        "tolerance-bounded"      — order-dependent; bounded by atol
        "n/a"                    — op has no forward / no backward
    notes is a short justification referencing the op's source file or a
    cited tolerance comment in-repo.

This is the operational form of the determinism stance the markdown in
PROFILING.md describes. Phase 9's bench-gate reads this CSV and asserts
each parity test stays within its row's `*_atol`.

The op + backend coverage is derived from:
    - 33 `impl VkBackwardOp for ...` blocks in `vk_ops/`
    - 15 `impl CustomOpN for ...` from Phase 0.2's audit
    - Core ops that always exist on a tensor library (matmul, softmax, ...)

Dtype × backend support is over-declared (we list every cell even if a
particular backend doesn't have that op today) so the CSV doubles as a
to-do list for Phase 6b/6c (Metal + CUDA backward parity).
"""

import csv
from pathlib import Path

# ----------------------------------------------------------------------
# Op inventory.
# ----------------------------------------------------------------------

# (op, fwd_dtypes, bwd_dtypes, category)
# category determines the default tolerance + determinism profile:
#   "matmul"           — fwd/bwd via cuBLAS / MPS / Vulkan compute; deterministic when workspace is pinned
#   "matmul-bwd-atomic"  — dW path uses atomicAdd on a per-channel grad accumulator
#   "atomic-bwd"       — bwd uses atomicAdd (embedding, scatter)
#   "reduction"        — softmax / RMSNorm / cross-entropy; fixed-tree reduction
#   "shape-only"       — reshape / transpose / permute / narrow — no math, no tolerance
#   "elementwise"      — silu / sigmoid / add / mul / cast — bit-stable in fwd, atomic-free bwd
#   "rope"             — rotate; bit-stable
#   "attention"        — flash-attn; deterministic variant via the build flag
#   "conv1d"           — causal conv; deterministic
#   "gdn"              — GatedDeltaNet; per-chunk fixed-stride
#   "loss"             — opd / flce; cross-entropy-shaped (reduction)
OPS = [
    # Identity / shape ops (no math, no tolerance).
    ("reshape",              ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "shape-only"),
    ("transpose",            ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "shape-only"),
    ("permute",              ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "shape-only"),
    ("narrow",               ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "shape-only"),
    ("contiguous",           ["F32", "BF16", "F16"], [],                     "shape-only"),
    ("cast",                 ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "shape-only"),

    # Element-wise (bit-stable in fwd; atomic-free bwd through bcast-aware reduction).
    ("add",                  ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "elementwise"),
    ("sub",                  ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "elementwise"),
    ("mul",                  ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "elementwise"),
    ("div",                  ["F32", "BF16", "F16"], ["F32", "BF16", "F16"], "elementwise"),
    ("silu",                 ["F32", "BF16"],        ["F32", "BF16"],        "elementwise"),
    ("sigmoid",              ["F32", "BF16"],        ["F32", "BF16"],        "elementwise"),
    ("mul_sigmoid_gate",     ["F32", "BF16"],        ["F32", "BF16"],        "elementwise"),

    # Reductions (fixed-tree, deterministic).
    ("softmax_last_dim",     ["F32", "BF16"],        ["F32", "BF16"],        "reduction"),
    ("sum_all",              ["F32", "BF16"],        ["F32", "BF16"],        "reduction"),
    ("mean_all",             ["F32", "BF16"],        ["F32", "BF16"],        "reduction"),
    ("rmsnorm",              ["F32", "BF16"],        ["F32", "BF16"],        "reduction"),
    ("l2norm",               ["F32", "BF16"],        ["F32", "BF16"],        "reduction"),

    # Matmul family (deterministic when workspace pinned; bwd-dW uses
    # per-column atomic accumulation in some backends).
    ("matmul",               ["F32", "BF16", "F16"], ["F32", "BF16"],        "matmul-bwd-atomic"),
    ("matmul_batched",       ["F32", "BF16", "F16"], ["F32", "BF16"],        "matmul-bwd-atomic"),
    # Marlin W4A16: forward-only.
    ("matmul_bf16w_marlin",  ["BF16"],               [],                     "matmul"),
    # Linear / Lora ops in kiln-model (Cuda + Vulkan today).
    ("linear",               ["F32", "BF16"],        ["F32", "BF16"],        "matmul-bwd-atomic"),
    ("lora_add",             ["F32", "BF16"],        ["F32", "BF16"],        "matmul-bwd-atomic"),
    ("lora_linear",          ["F32", "BF16"],        ["F32", "BF16"],        "matmul-bwd-atomic"),

    # Embedding / scatter (atomicAdd on the bwd).
    ("embedding",            ["F32", "BF16", "F16"], ["F32", "BF16"],        "atomic-bwd"),
    ("index_select_rows",    ["F32", "BF16", "F16"], ["F32", "BF16"],        "atomic-bwd"),

    # RoPE — rotate, bit-stable.
    ("rope",                 ["F32", "BF16"],        ["F32", "BF16"],        "rope"),

    # Attention (FA-2; FA-3 in Phase 8.12).
    ("flash_attn",           ["BF16"],               ["BF16"],               "attention"),
    ("flash_attn_paged",     ["BF16", "F8E4M3"],     [],                     "attention"),
    ("sdpa",                 ["F32", "BF16"],        ["F32", "BF16"],        "attention"),

    # Causal conv1d (GDN).
    ("causal_conv1d_update", ["F32", "BF16"],        ["F32", "BF16"],        "conv1d"),
    ("causal_conv1d_prefill", ["F32", "BF16"],       ["F32", "BF16"],        "conv1d"),

    # GatedDeltaNet kernel cluster.
    ("gdn_chunk_prep",       ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),
    ("gdn_chunk_scan",       ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),
    ("gdn_gates",            ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),
    ("gdn_gated_rms_norm",   ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),
    ("gdn_chunkwise",        ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),
    ("gdn_recurrent_step",   ["F32", "BF16"],        ["F32", "BF16"],        "gdn"),

    # Loss kernels.
    ("flce_loss",            ["F32", "BF16"],        ["F32", "BF16"],        "loss"),
    ("opd_loss",             ["F32", "BF16"],        ["F32", "BF16"],        "loss"),

    # KV-paged ops (forward-only on the read side; the write side is in-place).
    ("paged_kv_read",        ["F32", "BF16", "F8E4M3"], [],                  "shape-only"),
    ("paged_kv_write_slot",  ["F32", "BF16", "F8E4M3"], [],                  "shape-only"),

    # Sampling (forward-only; consumed by the sampler kernel chain).
    ("argmax",               ["F32", "BF16"],        [],                     "shape-only"),
    ("topk",                 ["F32", "BF16"],        [],                     "reduction"),
    ("apply_penalties",      ["F32", "BF16"],        [],                     "elementwise"),
]

BACKENDS = ["cpu", "cuda", "metal", "vulkan"]

# ----------------------------------------------------------------------
# Tolerance + determinism profiles.
# ----------------------------------------------------------------------

# Defaults are absolute-tolerance bands keyed on dtype. Categories override.
DEFAULT_FWD_ATOL = {
    "F32":     0.0,     # bit-identical on a deterministic kernel
    "BF16":    1e-3,    # ~1 ULP in BF16 at order-1 magnitudes
    "F16":     1e-3,
    "F8E4M3":  0.05,    # FP8 has 3-bit mantissa; band must reflect that
    "F8E5M2":  0.05,
}
DEFAULT_BWD_ATOL = {
    "F32":     1e-5,
    "BF16":    1e-2,    # accommodates the atomicAdd zone documented at rmsnorm-kernel:5036
    "F16":     5e-3,
    "F8E4M3":  0.1,
    "F8E5M2":  0.1,
}

# Category-specific tightening:
TIGHTEN_FWD = {
    "shape-only": {"F32": 0.0, "BF16": 0.0, "F16": 0.0},  # no math
}
TIGHTEN_BWD = {
    "shape-only": {"F32": 0.0, "BF16": 0.0, "F16": 0.0},  # bwd is inverse layout
    "rope":       {"F32": 0.0},                            # rotate-inverse is exact for F32
}
LOOSEN_BWD = {
    "atomic-bwd":        {"BF16": 2e-2, "F32": 5e-5},
    "matmul-bwd-atomic": {"BF16": 2e-2},   # dW path atomic on some backends
}

DETERMINISM_FWD = {
    "shape-only":         "constructive",
    "elementwise":        "constructive",
    "reduction":          "constructive",
    "matmul":             "constructive",
    "matmul-bwd-atomic":  "constructive",
    "atomic-bwd":         "constructive",
    "rope":               "constructive",
    "attention":          "constructive",
    "conv1d":             "constructive",
    "gdn":                "constructive",
    "loss":               "constructive",
}
DETERMINISM_BWD = {
    "shape-only":         "constructive",
    "elementwise":        "constructive",
    "reduction":          "constructive",
    "matmul":             "constructive",
    "matmul-bwd-atomic":  "tolerance-bounded",
    "atomic-bwd":         "tolerance-bounded",
    "rope":               "constructive",
    "attention":          "tolerance-bounded",  # FA bwd has the recorded deterministic variant
    "conv1d":             "constructive",
    "gdn":                "constructive",
    "loss":               "tolerance-bounded",
}

NOTES = {
    "shape-only":        "no arithmetic; layout-only.",
    "elementwise":       "bit-stable in forward; atomic-free in backward via bcast-aware reduction.",
    "reduction":         "fixed-tree reduction; deterministic.",
    "matmul":            "deterministic under CUBLAS_WORKSPACE_CONFIG=:4096:8; the workspace pin is mandatory.",
    "matmul-bwd-atomic": "dW path uses per-column atomic accumulation on some backends; tolerance reflects BF16-ULP band.",
    "atomic-bwd":        "atomic-add bwd; deterministic variant available under KILN_DETERMINISTIC=1.",
    "rope":              "rotate; no reduction.",
    "attention":         "flash-attn bwd uses a deterministic variant under build flag; default path is tolerance-bounded.",
    "conv1d":            "fixed-window stride; deterministic.",
    "gdn":               "per-chunk fixed-stride reduction; deterministic within chunk.",
    "loss":              "cross-entropy-shaped reduction; bwd uses atomicAdd on the unreduced-grad accumulator.",
}

# ----------------------------------------------------------------------
# Backend support map.
# ----------------------------------------------------------------------

# Mark which {op, backend} cells have an implementation today vs are
# scheduled for Phase N. The CSV records both with explicit `coverage`
# column so the matrix doubles as a Phase 6b/6c to-do.
#
# An op is in `today_on` for a backend if a real kernel exists today;
# otherwise it lands in `phase_<n>` for the backend phase that ships it.
TODAY_ON = {
    "cpu":    {op for (op, *_rest) in OPS},  # CPU is the canonical reference; everything must work there
    "cuda":   {
        "reshape", "transpose", "permute", "narrow", "contiguous", "cast",
        "add", "sub", "mul", "div", "silu", "sigmoid", "mul_sigmoid_gate",
        "softmax_last_dim", "sum_all", "mean_all", "rmsnorm", "l2norm",
        "matmul", "matmul_batched", "matmul_bf16w_marlin", "linear",
        "lora_add", "lora_linear",
        "embedding", "index_select_rows",
        "rope",
        "flash_attn", "flash_attn_paged", "sdpa",
        "causal_conv1d_update", "causal_conv1d_prefill",
        "gdn_chunk_prep", "gdn_chunk_scan", "gdn_gates", "gdn_gated_rms_norm",
        "gdn_chunkwise", "gdn_recurrent_step",
        "flce_loss", "opd_loss",
        "paged_kv_read", "paged_kv_write_slot",
        "argmax", "topk", "apply_penalties",
    },
    "metal":  {
        "reshape", "transpose", "permute", "narrow", "contiguous", "cast",
        "add", "sub", "mul", "div", "silu", "sigmoid",
        "softmax_last_dim", "sum_all", "mean_all", "rmsnorm",
        "matmul", "matmul_batched", "linear",
        "embedding", "index_select_rows",
        "rope", "sdpa",
        "argmax", "topk",
    },
    "vulkan": {
        "reshape", "transpose", "permute", "narrow", "contiguous", "cast",
        "add", "sub", "mul", "div", "silu", "sigmoid", "mul_sigmoid_gate",
        "softmax_last_dim", "sum_all", "mean_all", "rmsnorm", "l2norm",
        "matmul", "matmul_batched", "matmul_bf16w_marlin",
        "embedding", "index_select_rows",
        "rope", "flash_attn",  # FlashSdpaBackward exists
        "causal_conv1d_update", "causal_conv1d_prefill",
        "gdn_chunk_prep", "gdn_chunk_scan", "gdn_gates", "gdn_gated_rms_norm",
        "gdn_chunkwise",
        "flce_loss", "opd_loss",
    },
}


def fwd_atol(category, dtype):
    base = DEFAULT_FWD_ATOL.get(dtype, 1e-3)
    if category in TIGHTEN_FWD:
        base = TIGHTEN_FWD[category].get(dtype, base)
    return base


def bwd_atol(category, dtype):
    base = DEFAULT_BWD_ATOL.get(dtype, 1e-2)
    if category in TIGHTEN_BWD:
        base = TIGHTEN_BWD[category].get(dtype, base)
    if category in LOOSEN_BWD:
        base = LOOSEN_BWD[category].get(dtype, base)
    return base


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "bench-results" / "parity-tolerance.csv"
    md = repo_root / "bench-results" / "parity-tolerance.md"
    out.parent.mkdir(exist_ok=True)

    rows = []
    for (op, fwd_dts, bwd_dts, category) in OPS:
        all_dts = sorted(set(fwd_dts + bwd_dts),
                         key=["F32", "BF16", "F16", "F8E4M3", "F8E5M2"].index)
        for dtype in all_dts:
            for backend in BACKENDS:
                has_fwd = dtype in fwd_dts
                has_bwd = dtype in bwd_dts
                coverage = "today" if op in TODAY_ON.get(backend, set()) else "scheduled"
                rows.append({
                    "op": op,
                    "dtype": dtype,
                    "backend": backend,
                    "category": category,
                    "fwd_atol":        fwd_atol(category, dtype) if has_fwd else "",
                    "bwd_atol":        bwd_atol(category, dtype) if has_bwd else "",
                    "fwd_determinism": DETERMINISM_FWD.get(category, "constructive") if has_fwd else "n/a",
                    "bwd_determinism": DETERMINISM_BWD.get(category, "constructive") if has_bwd else "n/a",
                    "coverage":        coverage,
                    "notes":           NOTES.get(category, ""),
                })

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "op", "dtype", "backend", "category",
            "fwd_atol", "bwd_atol",
            "fwd_determinism", "bwd_determinism",
            "coverage", "notes",
        ])
        for r in rows:
            w.writerow([
                r["op"], r["dtype"], r["backend"], r["category"],
                r["fwd_atol"], r["bwd_atol"],
                r["fwd_determinism"], r["bwd_determinism"],
                r["coverage"], r["notes"],
            ])

    # ---- markdown interpretation -------------------------------------
    by_backend = {b: [r for r in rows if r["backend"] == b] for b in BACKENDS}
    bwd_today = {
        b: sum(1 for r in by_backend[b] if r["coverage"] == "today" and r["bwd_atol"] != "")
        for b in BACKENDS
    }
    bwd_scheduled = {
        b: sum(1 for r in by_backend[b] if r["coverage"] == "scheduled" and r["bwd_atol"] != "")
        for b in BACKENDS
    }
    fwd_today = {
        b: sum(1 for r in by_backend[b] if r["coverage"] == "today" and r["fwd_atol"] != "")
        for b in BACKENDS
    }

    with md.open("w", encoding="utf-8") as f:
        f.write("# Phase 0.4 — Parity tolerance matrix\n\n")
        f.write(
            "Source of truth: `bench-results/parity-tolerance.csv` "
            f"({len(rows)} rows).\n"
            "Regenerate: `scripts/build-parity-tolerance.py`.\n\n"
            "## What this is\n\n"
            "One row per `{op, dtype, backend}` cell. Each row carries:\n\n"
            "- `fwd_atol` / `bwd_atol` — absolute-tolerance thresholds (parity\n"
            "  test asserts the per-element max-abs-diff is below this band).\n"
            "- `fwd_determinism` / `bwd_determinism` — either `constructive`\n"
            "  (bit-identical across runs) or `tolerance-bounded` (order-dependent;\n"
            "  bounded by the atol). Anchored to the determinism stance in\n"
            "  PROFILING.md (Phase 0.3).\n"
            "- `coverage` — `today` if the kernel exists in the current repo,\n"
            "  `scheduled` if it lands in a later Phase (2 / 3 / 4 / 6b / 6c).\n\n"
            "## Forward / backward coverage today (by backend)\n\n"
        )
        f.write("| backend | fwd cells (today) | bwd cells (today) | bwd cells (scheduled) |\n")
        f.write("|---|---:|---:|---:|\n")
        for b in BACKENDS:
            f.write(f"| {b} | {fwd_today[b]} | {bwd_today[b]} | {bwd_scheduled[b]} |\n")
        f.write(
            "\nThe `scheduled` count for CUDA and Metal is Phase 6b / 6c's "
            "to-do list. The Vulkan track is most complete today — 33 "
            "`impl VkBackwardOp for ...` blocks in `vk_ops/` — and is the "
            "lift template for the other two backends.\n\n"
        )

        f.write("## Tolerance band defaults\n\n")
        f.write(
            "Per-dtype absolute-tolerance bands (overridden per-category in\n"
            "the CSV; see `notes` column for justification):\n\n"
            "| dtype | default fwd_atol | default bwd_atol | atomic-bwd bwd_atol |\n"
            "|---|---:|---:|---:|\n"
        )
        for dtype in ["F32", "BF16", "F16", "F8E4M3", "F8E5M2"]:
            f_def = DEFAULT_FWD_ATOL[dtype]
            b_def = DEFAULT_BWD_ATOL[dtype]
            b_atomic = LOOSEN_BWD["atomic-bwd"].get(dtype, b_def)
            f.write(f"| `{dtype}` | {f_def} | {b_def} | {b_atomic} |\n")

        f.write("\n## How this is enforced\n\n")
        f.write(
            "- Every kiln-tensor op parity test reads its CSV row at "
            "harness-init time and uses the row's `*_atol` as the assertion "
            "threshold.\n"
            "- A test whose op + dtype + backend has no CSV row fails — "
            "tolerance must be declared, not implicit.\n"
            "- Phase 9's bench-gate re-runs the audit + parity-tolerance "
            "consistency check; a row added without a justifying op or a "
            "row removed without an op deletion fails the gate.\n"
            "- `KILN_DETERMINISTIC=1` envelope (PROFILING.md §Determinism "
            "stance) selects the deterministic variant of every "
            "`bwd_determinism = tolerance-bounded` op; under the envelope, "
            "those cells must hit `bwd_atol = 0` in the parity test.\n"
        )

    print(f"wrote {out}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
